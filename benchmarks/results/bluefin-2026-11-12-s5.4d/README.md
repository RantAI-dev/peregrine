# The 8-bit prompt form on AVX2 (S5.4d, x86 half)

`--fast-prompt`'s 8-bit form had one SIMD body (AVX-512). On an AVX2 machine the mode ran
through the scalar twin: correct, and about as fast as the exact kernels. This is the AVX2
body, its builder, their bit-identity proof, and the same two tables the desktop measured
(`benchmarks/results/desktop-zen5-2026-11-12-s5.4b/`) on this host.

**The result, first.** The AVX2 body is bit-identical to the scalar twin and to the AVX-512
body (the tests below), and it is **37× the twin** — so the "correct but not faster" state is
gone. It is **not** faster than this tier's *exact* Q4_K kernel: 59.7 against 77.4 GMAC/s on
one core. On the 1B file, whose Q4_K matrices hold 67 % of the weights, the fast mode is
therefore **0.86-0.92 of the exact mode** on every cell. It is ahead on the Q6_K part (1.30×
the exact Q6_K kernel) and on the Qwen 0.5B, whose Q4_K_M file keeps most of its weights in
Q5_0 (which the form does not cover), the two modes are level (0.90-1.04). The reason is a
property of the instruction set, not of this implementation: see §"Why AVX2 cannot be ahead"
below.

- Date: 2026-11-12.
- Host: bluefin, AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads, AVX2 + FMA + F16C, no
  AVX-512, no AVX-VNNI, 13 GB RAM, Fedora 44, kernel 7.2.7-200.fc44.x86_64.
- Threads: **4** (one per physical core, `taskset -c 0-3`) and **1** (`taskset -c 0`), as the
  brief asks. Never 8: the SMT siblings halve the per-core rate.
- Measured tree: `origin/main` at `5b394c3` plus this PR's changes; rustc 1.99.1.
- Load average during the runs 2.7-3.6, no other benchmark on the host (the brief's own rule:
  another worker was on the ARM host, not here).
- **Skipped, per the brief and this machine's memory:** the large-model golden tests (phi-4
  Q4_K_M, the 27B/30B files) and the accuracy report of S5.4c, which is another machine's job.
  The 4.4 GB free disk and the full swap of the earlier worker were re-checked: 15 GB free on
  `/var/home`, and the `/tmp` tmpfs was at 80 % because of earlier sessions' files, which were
  cleared before the model tests ran.

## Models

| model | bytes | SHA-256 | Q4_K | Q6_K | Q5_0 | the form covers |
|---|---|---|---|---|---|---|
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 807 694 464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` | 445.5 MiB | 317.1 MiB | — | 830 M + 143 M weights (the Q6_K token embedding is not multiplied) |
| `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | 397 808 192 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` | 28.1 MiB | 40.9 MiB | 165.1 MiB | 69 MiB of 373.7 MiB of weights: the file is mostly Q5_0, which has no form in this step |

Both were already on the host (`~models/`); nothing was downloaded.

## a. The kernel, GMAC/s

```
KQBENCH_PROMPT=1 taskset -c 0-3 ./target/release/examples/kqbench 8 20 4
```

One "row" of work is 8 output rows of one tile over 8 super-blocks (2048 weights per row)
against 128 activation rows, the width of a prompt chunk; the table is the best of four rounds
per thread, aggregate over the cores. `exact` is today's wide kernel (`q4k8_gemm`,
`q6k8_gemm`), `twin` is `kq8::gemm_q8k` (the definition), `avx2` is `kqsimd::q8k_gemm_avx2`
(raw output: `kqbench.txt`).

| format | body | 1 core | 4 cores |
|---|---|---|---|
| Q4_K | exact wide kernel | 77.4 | 307.7 |
| Q4_K | 8-bit scalar twin | 1.6 | 6.2 |
| Q4_K | **8-bit AVX2** | **59.7** | **236.6** |
| Q6_K | exact wide kernel | 45.8 | 181.6 |
| Q6_K | 8-bit scalar twin | 1.6 | 6.2 |
| Q6_K | **8-bit AVX2** | **59.5** | **236.2** |

Against the exact kernel: Q4_K **0.77×**, Q6_K **1.30×**. Against the scalar twin: **37×**
(one core), 38× at four. The 4-core columns scale 3.96-4.07× the single-core ones: the body
is compute-bound and scales with the cores.

## b. End to end, `peregrine bench`

```
# one cell, five times; PEREGRINE_PREFILL_CHUNK is the fast mode's chunk
PEREGRINE_PREFILL_CHUNK=256 taskset -c 0-3 target/release/peregrine bench \
  -m ~models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --prompt-tokens 512 --tokens 1 \
  --threads 4 --fast-prompt --json
# the exact rows: no --fast-prompt and no PEREGRINE_PREFILL_CHUNK (its wide default)
```

`--tokens 1` makes the prefill the measurement. Median of five runs, spread in the raw data
(`runs.jsonl`); every row of the tables below is five runs.

**Llama 3.2 1B Instruct Q4_K_M** — prompt tokens/s:

| tokens | threads | exact | fast 64 | fast 128 | fast 256 | fast 512 |
|---|---|---|---|---|---|---|
| 512 | 1 | 61.7 | 53.3 (0.86) | 54.3 (0.88) | 54.4 (0.88) | 53.9 (0.87) |
| 512 | 4 | 229.0 | 199.8 (0.87) | 204.9 (0.89) | 204.5 (0.89) | 200.9 (0.88) |
| 4096 | 1 | 45.7 | 41.9 (0.92) | 41.7 (0.91) | 42.1 (0.92) | 42.1 (0.92) |
| 4096 | 4 | 175.0 | 157.1 (0.90) | 160.4 (0.92) | 160.4 (0.92) | 158.9 (0.91) |

**Qwen 2.5 0.5B Instruct Q4_K_M** — prompt tokens/s:

| tokens | threads | exact | fast 64 | fast 128 | fast 256 | fast 512 |
|---|---|---|---|---|---|---|
| 512 | 1 | 132.2 | 134.0 (1.01) | 133.4 (1.01) | 131.2 (0.99) | 127.8 (0.97) |
| 512 | 4 | 467.4 | 461.2 (0.99) | 463.1 (0.99) | 445.1 (0.95) | 421.7 (0.90) |
| 4096 | 1 | 88.9 | 92.2 (1.04) | 91.6 (1.03) | 88.7 (1.00) | 86.1 (0.97) |
| 4096 | 4 | 329.1 | 323.8 (0.98) | 326.1 (0.99) | 319.3 (0.97) | 306.0 (0.93) |

**Every cell below 1.0 is named:** on the 1B, *all sixteen* fast cells are below 1.0 (0.86-0.92),
and that is the headline of this step: on AVX2 the mode is slower than the exact engine on a
Q4_K_M file. On the Qwen 0.5B the cells below 1.0 are 512/1 at chunks 256 (0.99) and 512 (0.97);
512/4 at 256 (0.95) and 512 (0.90); 4096/1 at 256 (1.00, rounding) and 512 (0.97); 4096/4 at
256 (0.97) and 512 (0.93). Its three cells above 1.0 are 512/1 at 64 and 128 (1.01) and 4096/1
at 64 and 128 (1.04, 1.03). The Qwen file is mostly Q5_0, so the mode changes only 69 MiB of its
373.7 MiB of weights and the difference is nearly noise — except at chunk 512, where the wide
chunk itself costs (see below).

**The best chunk on this CPU is 128 or 256, tied within 0.2 %** (54.3 against 54.4 at 512/1,
204.9 against 204.5 at 512/4, 160.4 against 160.4 at 4096/4; on the Qwen 128 is ahead on all
four rows). 64 is 2 % behind and 512 is 2-9 % behind; 512's loss grows with the Qwen's chunk
(0.90 at 512/4) where the pass is short enough that a wider chunk costs. The current fast
default (**256**) is within 0.2 % of the best on the 1B and 2-5 % off it on the Qwen, so **the
default does not change**: the brief's rule (a change only on a gain over 5 % on both models)
is not met by any chunk.

## c. The other engines, on the same cells

ik_llama.cpp at its defaults (it re-quantises the files on the fly: it reports 968.30 MiB for
the 1B and 511.65 MiB for the Qwen, i.e. its own repacked sizes) and llama.cpp's default path.
Both with `-n 0`, `taskset -c 0-3` (or `-c 0`), `-r 5` (`-r 3` for the 4096/1 rows, which are
90 s+ per repetition).

```
taskset -c 0-3 /var/tmp/llama.cpp/build/bin/llama-bench        -m <model> -p 512  -n 0 -t 4 -r 5
taskset -c 0-3 /var/tmp/ik-llama-s54/build/bin/llama-bench     -m <model> -p 512  -n 0 -t 4 -r 5
```

| engine | build | path |
|---|---|---|
| llama.cpp | `cd959ba` (v0.9.0-dev) | `/var/tmp/llama.cpp/build/bin/llama-bench` |
| ik_llama.cpp | `883789c` (built here: `cmake -B build -DCMAKE_BUILD_TYPE=Release`, `cmake --build build -j4 --target llama-bench`) | `/var/tmp/ik-llama-s54/build/bin/llama-bench` |

Prompt tokens/s, and Peregrine ÷ each engine:

**Llama 3.2 1B**

| cell | Peregrine exact | Peregrine fast (256) | llama.cpp | ik_llama.cpp | exact ÷ llama | exact ÷ ik | fast ÷ llama | fast ÷ ik |
|---|---|---|---|---|---|---|---|---|
| 512 tok, 1 th | 61.7 | 54.4 | 51.09 | 57.28 | 1.21 | 1.08 | 1.06 | 0.95 |
| 512 tok, 4 th | 229.0 | 204.5 | 192.09 | 220.14 | 1.19 | 1.04 | 1.06 | 0.93 |
| 4096 tok, 1 th | 45.7 | 42.1 | 38.31 | 44.61 | 1.19 | 1.02 | 1.10 | 0.94 |
| 4096 tok, 4 th | 175.0 | 160.4 | 143.03 | 171.60 | 1.22 | 1.02 | 1.12 | 0.93 |

**Qwen 2.5 0.5B** (fast at its best chunk here, 128)

| cell | Peregrine exact | Peregrine fast (128) | llama.cpp | ik_llama.cpp | exact ÷ llama | exact ÷ ik | fast ÷ llama | fast ÷ ik |
|---|---|---|---|---|---|---|---|---|
| 512 tok, 1 th | 132.2 | 133.4 | 81.83 | 127.94 | 1.62 | 1.03 | 1.63 | 1.04 |
| 512 tok, 4 th | 467.4 | 463.1 | 305.53 | 488.96 | 1.53 | 0.96 | 1.52 | 0.95 |
| 4096 tok, 1 th | 88.9 | 91.6 | 63.71 | 92.85 | 1.40 | 0.96 | 1.44 | 0.99 |
| 4096 tok, 4 th | 329.1 | 326.1 | 237.18 | 353.31 | 1.39 | 0.93 | 1.37 | 0.92 |

**Cells below 1.0, named.** Peregrine exact ÷ ik_llama.cpp: 0.96 (Qwen 512/4), 0.96 (Qwen
4096/1), 0.93 (Qwen 4096/4). Peregrine fast ÷ ik_llama.cpp: 0.95, 0.93, 0.94, 0.93 on the 1B;
0.95, 0.99, 0.92 on the Qwen. Every other cell is above 1.0. Nothing is below 1.0 against
llama.cpp: exact is 1.19-1.62 of it, fast is 1.06-1.63.

## d. Load time, memory, and decode

`/usr/bin/time -v` on `bench --prompt-tokens 8 --tokens 1` (three runs each; the pack cache is
warm, so this is the load proper, not a first-run repack):

| | wall | peak RSS |
|---|---|---|
| default | 0.47 s | 1 013 096 KiB |
| `--fast-prompt` | 1.29-1.31 s | 2 001 156 KiB |

**+0.83 s and +987 MiB.** `PEREGRINE_FAST_PROMPT=1 peregrine inspect` reports the form at
**957.0 MiB** (the desktop measured +0.52 s with the AVX-512 builder; the AVX2 builder is
about 0.3 s slower because it packs eight codes at a time instead of sixteen).

**Decode after a fast prompt against after an exact prompt** (1B, 512-token prompt, 32 greedy
tokens, 4 threads, five pairs): 46.53 / 46.56, 46.88 / 46.78, 46.93 / 46.14, 45.98 / 46.56,
46.97 / 46.49 tok/s (exact / fast) — medians 46.88 and 46.56, inside the run-to-run spread.
Level, as it must be: the form is read only by prompt rows.

## e. Accuracy sanity (not the report)

Two plain-text prompts, no chat template, 64 greedy tokens, 1B, 4 threads
(`peregrine run ... --max-tokens 64 --temperature 0`), exact against `--fast-prompt`:

1. *"The lighthouse keeper kept a ledger of every ship that passed the point, noting its name,
   its flag, and the hour."* — the two agree for the first 231 characters (about 50 tokens),
   then exact continues `"...had become an integral part of his daily routine."` and fast
   `"...had become a trusted companion in his role as a keeper."`.
2. *"In 1847 a chemist in Uppsala dissolved a mineral in acid and found a new element in the
   residue."* — they differ at the **first** token: exact `"That's close to the actual
   event..."`, fast `"You're referring to the discovery of Radium..."`.

So no, the first 64 tokens are not equal, on either prompt.

**That is not evidence of a broken body**, and the two facts that say so are worth stating:

- the AVX2 body is *bit-identical* to the scalar twin at every width (the kernel tests) and
  end to end (the tier's fast goldens, recorded with the scalar twin, pass unchanged), so the
  numbers the mode produces on this tier are exactly the numbers the AVX-512 tier produced;
- this host's *exact* mode differs from `PEREGRINE_ISA=scalar`'s exact mode at the same scale:
  prompt 1 at character 73, and on prompt 2 the scalar exact run produces no less than the
  string the AVX2 fast run produces (`"You're referring to the discovery of Radium..."`). The
  tier-to-tier spread of the *exact* engine here is larger than the fast mode's approximation,
  which is the known per-tier golden effect, not the form.

## Why AVX2 cannot be ahead of the exact Q4_K kernel

AVX-512's `vpdpbusd` does 64 products per instruction (8 rows × 8 codes × 1 activation row) and
accumulates in int32, so the 8-bit form's kernel is 1.28× the exact Q4_K kernel there. AVX2 has
no byte dot product: `vpmaddubsw` does 32 products per instruction and *saturates* at int16, so
every group of eight weights needs

1. one `vpsignb` (the sign trick: `|code|` × `sign(code)·a`, `2·128·127 = 32 512 < 32 767`,
   which is how the code's bias and sign are handled without a saturating pair),
2. one `vpmaddubsw`,
3. one `vpmaddwd` (int16 → int32) and one `vpaddd`,
4. plus the activation broadcast, amortised over the tile's eight activation rows.

That is four instructions per 32 products, 8 MAC per instruction. The exact Q4_K kernel's
nibbles are small enough to chain *eight* `vpmaddubsw` in int16 before the widening
(`16 · 15 · 127 = 30 480`), so it gets 21 MAC per instruction. No formulation of the 8-bit form
escapes this: the nibble split of the codes (two `vpmaddubsw` per window, chained deeper) and
the int16 `vpmaddwd` path both land at 6-7 MAC per instruction. The body measures 59.7 GMAC/s
= 14.5 MAC/cycle at 4.1 GHz, which is the machine's two 256-bit vector ops per cycle at 7.1 MAC
per instruction; the exact kernel is at 18.9 MAC/cycle because it is latency-bound, not
throughput-bound. On this tier the form pays off where the exact kernel is weaker (Q6_K, +30 %)
and loses where it is strongest (Q4_K, −23 %). The mode stays off by default, and the design
note's §5d says the same in prose.
