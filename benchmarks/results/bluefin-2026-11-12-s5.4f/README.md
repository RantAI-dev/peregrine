# The 8-bit form where it wins on AVX2 (S5.4f)

`--fast-prompt` took the resident 8-bit form for Q4_K and Q6_K on every tier that had a SIMD
body. On AVX2 that made it **slower than the exact engine**: the form's AVX2 body is 59.7
GMAC/s for every format, this tier's exact Q4_K kernel is 77.4, and a Q4_K_M file is mostly
Q4_K. This step makes the set of formats a property of the tier — one table in
`crates/peregrine-kernels/src/table.rs` — measured here for Q4_K, Q6_K and Q5_0, and adds Q5_0
to the engine's formal set (its own build path included). On AVX2 only **Q6_K** takes the form,
so the Llama 1B file carries a form for its Q6_K matrices only (140.2 MiB, against 957.0 MiB
before) and the mode is **1.05-1.08× the exact engine on every cell** instead of 0.88-0.93.
Qwen 2.5 0.5B, whose Q5_0 matrices are 896 wide, is 1.00-1.04× (its fast Q6_K share is 51.4 MiB).

- Date: 2026-11-12.
- Host: bluefin, AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads, AVX2 + FMA + F16C, no
  AVX-512, no AVX-VNNI, 13 GB RAM, Fedora 44, kernel 7.2.7-200.fc44.x86_64.
- Threads **4** (`taskset -c 0-3`) and **1** (`taskset -c 0`), as S5.4d.
- Measured tree: `s5.4f-form-where-it-wins` from `s5.4d-prompt-kernel-avx2` (PR #250),
  plus this step's changes; rustc 1.99.1.
- Load average during the runs **2.4-6.1** (`runs.jsonl` records it per round; the host had
  Chrome and `kswapd` on it, and the load rose during the 4096-token multi-thread cells). The
  fast and exact runs alternate inside each round, so a drifting load moves both.
- **Skipped, per the brief:** the large-model golden tests (phi-4 Q4_K_M, the 27B/30B files).
  The 1B and Qwen 0.5B files were already on the host (`~models/`); nothing was downloaded.
- Disk/RAM before the runs: 12 GB free on `/var/home`, 6 GB available RAM, 2 GB free swap.

## Models

| model | bytes | SHA-256 |
|---|---|---|
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 807 694 464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |
| `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | 397 808 192 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |

## a. The kernel table (the per-format decision)

```
KQBENCH_PROMPT=1 KQBENCH_Q5_SHAPE=1 taskset -c 0-3 \
  target/release/examples/kqbench 8 20 4
```

One row of work is 8 output rows of an eight-super-block tile (2048 weights per row) against
the given number of activation rows; `exact` is the tier's own wide kernel (`q4k8_gemm`,
`q6k8_gemm`, `q5_08_gemm`), `form` is the 8-bit form's AVX2 body (`q8k_gemm_avx2`). Best of four
rounds inside the run, three runs; medians below. The raw three runs, load averages and the
896-wide section: `kqbench.txt`.

| format | rows | 1 core exact | 1 core form | ratio | 4 cores exact | 4 cores form | ratio |
|---|---|---|---|---|---|---|---|
| Q4_K | 8 | 74.7 | 59.8 | 0.80 | 297.8 | 238.3 | 0.80 |
| Q4_K | 64 | 75.7 | 59.9 | **0.79** | 299.9 | 238.7 | 0.80 |
| Q4_K | 128 | 75.2 | 59.7 | **0.79** | 292.2 | 237.4 | 0.81 |
| Q4_K | 512 | 71.5 | 59.1 | **0.83** | 276.2 | 235.6 | 0.85 |
| Q6_K | 8 | 30.6 | 39.8 | 1.30 | 122.5 | 159.4 | 1.30 |
| Q6_K | 64 | 45.9 | 59.6 | **1.30** | 182.7 | 234.3 | 1.28 |
| Q6_K | 128 | 45.8 | 59.5 | **1.30** | 182.1 | 237.7 | 1.31 |
| Q6_K | 512 | 45.2 | 59.3 | **1.31** | 175.3 | 235.7 | 1.34 |
| Q5_0 | 8 | 40.3 | 39.8 | 0.99 | 161.0 | 159.4 | 0.99 |
| Q5_0 | 64 | 60.1 | 59.5 | **0.99** | 237.4 | 238.0 | 1.00 |
| Q5_0 | 128 | 58.8 | 59.6 | **1.01** | 235.5 | 237.3 | 1.01 |
| Q5_0 | 512 | 57.7 | 59.2 | **1.03** | 145.7 | 234.8 | 1.61 |

GMAC/s. The decision rule is 1.10× at prompt widths (64 rows and up), so **Q6_K takes the form
and Q4_K and Q5_0 do not** on AVX2. Q5_0's 4-core 512 row (1.61) is the exact kernel's odd
slowdown at that width (145.7 against 235 at 128, while the form stays flat); its 1-core ratio
is 1.03, and the shape the Qwen file actually has is worse still:

**Q5_0 at Qwen 2.5's 896-wide row** (28 blocks, a padded fourth super-block; `kqbench.txt`'s
`useful` section, GMAC/s of the 896 useful weights):

| rows | 1 core exact | 1 core form | ratio | 4 cores exact | 4 cores form | ratio |
|---|---|---|---|---|---|---|
| 64 | 56.8 | 49.7 | **0.88** | 231.8 | 207.0 | 0.89 |
| 128 | 57.4 | 51.3 | **0.89** | 229.6 | 207.0 | 0.90 |
| 512 | 52.3 | 50.7 | 0.97 | 218.7 | 205.4 | 0.94 |

The padding a 896-wide row needs is why Q5_0 is not in the AVX2 set even though its whole-block
form is level: at the shape the Qwen file uses it is 0.88-0.90× the exact kernel. The AVX-512
set is unchanged by this step (Q4_K and Q6_K), as the brief requires.

## b. End to end, `peregrine bench`

```
# one cell, five times; both fast builds alternate with the exact engine inside each round
taskset -c 0-3 target/release/peregrine bench -m <model> --prompt-tokens 512 --tokens 1 \
  --threads 4 [--fast-prompt] --json
taskset -c 0-3 /tmp/peregrine-s54d-before bench -m <model> --prompt-tokens 512 --tokens 1 \
  --threads 4 --fast-prompt --json          # S5.4d's build, PR #250's head
```

Median of five runs; the spread (min-max) and each round's load average are in `runs.jsonl`.
`before` is a binary built from the branch this one starts at (S5.4d), `after` this PR's.

**Llama 3.2 1B Instruct Q4_K_M** — prompt tokens/s:

| tokens | threads | exact | fast before | fast after | before ÷ exact | after ÷ exact |
|---|---|---|---|---|---|---|
| 512 | 1 | 59.4 | 54.2 | 64.0 | 0.913 | **1.077** |
| 512 | 4 | 223.3 | 205.2 | 236.6 | 0.919 | **1.059** |
| 4096 | 1 | 47.4 | 44.2 | 50.4 | 0.932 | **1.063** |
| 4096 | 4 | 178.3 | 165.9 | 186.9 | 0.931 | **1.048** |

**Qwen 2.5 0.5B Instruct Q4_K_M** — prompt tokens/s:

| tokens | threads | exact | fast before | fast after | before ÷ exact | after ÷ exact |
|---|---|---|---|---|---|---|
| 512 | 1 | 131.6 | 131.9 | 136.4 | 1.003 | **1.036** |
| 512 | 4 | 470.7 | 454.0 | 473.5 | 0.964 | **1.006** |
| 4096 | 1 | 93.2 | 94.0 | 96.7 | 1.008 | **1.037** |
| 4096 | 4 | 339.7 | 332.2 | 338.9 | 0.978 | **0.998 (level)** |

**The acceptance bar, per cell.** Every Llama cell is clearly above 1.00: the after run's
lowest value is above the exact run's highest on all four (62.6 > 60.0, 231.7 > 225.0,
50.0 > 47.5, 183.6 > 178.6). On the Qwen the cells are 1.006-1.037 except 4096/4, where
after's range (329.3-343.2) contains the exact median (339.7) and exact's range
(338.1-340.5) contains after's median (338.9) — **level, not below**, reported as such.
`before` on the Qwen is level too: it changed only 69 MiB of Q4_K/Q6_K weights, and the Q4_K
part was at 0.79×.

The Qwen 512/1 after cell has one 119.1 tok/s run out of five (its other four are 136.3-137.5);
that is a transient on the host (load 2.93, no outlier in the other mode that round), and the
median is 136.4. The exact 512/4 and 4096/1 cells have one low run each (439.0 and 82.4) for
the same reason; their medians are what the table uses.

**The form's size, before and after** (`PEREGRINE_FAST_PROMPT=1 peregrine inspect`; `forms.txt`):

| model | form before | form after | peak RSS before | peak RSS after |
|---|---|---|---|---|
| Llama 3.2 1B | 957.0 MiB | **140.2 MiB** | 1 999 600 KiB | **1 163 500 KiB** |
| Qwen 2.5 0.5B | 102.9 MiB | **51.4 MiB** | 780 100 KiB | **727 600 KiB** |

A Q4_K_M 1B on AVX2 now carries a form for its Q6_K matrices only; the 957 MiB S5.4d figure
included the Q4_K matrices the tier no longer uses. On a tier with no SIMD body the form is
0.0 MiB and the mode computes exactly what the exact engine computes.

## c. The other engines, on the same cells

`engines.md` has the builds and raw rows. Peregrine's own numbers are the medians of table b.

**Llama 3.2 1B**

| cell | exact | fast | llama.cpp | ik_llama.cpp | exact ÷ llama | exact ÷ ik | fast ÷ llama | fast ÷ ik |
|---|---|---|---|---|---|---|---|---|
| 512 tok, 1 th | 59.4 | 64.0 | 51.07 | 57.09 | 1.16 | 1.04 | 1.25 | 1.12 |
| 512 tok, 4 th | 223.3 | 236.6 | 194.02 | 221.42 | 1.15 | 1.01 | 1.22 | 1.07 |
| 4096 tok, 1 th | 47.4 | 50.4 | 38.14 | 44.62 | 1.24 | 1.06 | 1.32 | 1.13 |
| 4096 tok, 4 th | 178.3 | 186.9 | 143.16 | 171.51 | 1.25 | 1.04 | 1.31 | 1.09 |

**Qwen 2.5 0.5B**

| cell | exact | fast | llama.cpp | ik_llama.cpp | exact ÷ llama | exact ÷ ik | fast ÷ llama | fast ÷ ik |
|---|---|---|---|---|---|---|---|---|
| 512 tok, 1 th | 131.6 | 136.4 | 81.59 | 127.93 | 1.61 | 1.03 | 1.67 | 1.07 |
| 512 tok, 4 th | 470.7 | 473.5 | 305.11 | 487.63 | 1.54 | 0.97 | 1.55 | 0.97 |
| 4096 tok, 1 th | 93.2 | 96.7 | 63.58 | 92.67 | 1.47 | 1.01 | 1.52 | 1.04 |
| 4096 tok, 4 th | 339.7 | 338.9 | 239.70 | 354.36 | 1.42 | 0.96 | 1.41 | 0.96 |

**Every cell below 1.0 is named.** Against llama.cpp nothing is: exact is 1.15-1.61× and fast
1.22-1.67×. Against ik_llama.cpp the four below 1.0 are all on the Qwen at four threads:
exact 0.965 (512 tok) and 0.958 (4096 tok), fast 0.971 and 0.956. Every 1B cell is above 1.0
against both engines.

## d. Load, memory and decode

`/usr/bin/time -v` on `bench --prompt-tokens 8 --tokens 1` (three runs each; `load.txt`):

| model | mode | wall | peak RSS |
|---|---|---|---|
| Llama 3.2 1B | exact | 0.62-0.63 s | 1 012 300 KiB |
| Llama 3.2 1B | fast, after | 0.73-0.86 s | 1 163 500 KiB |
| Llama 3.2 1B | fast, before | 2.21-2.44 s | 1 999 600 KiB |
| Qwen 2.5 0.5B | exact | 0.41-0.42 s | 668 100 KiB |
| Qwen 2.5 0.5B | fast, after | 0.45 s | 727 600 KiB |
| Qwen 2.5 0.5B | fast, before | 0.49 s | 780 100 KiB |

The 1B's fast load costs **+0.16 s and +151 MiB** after this step, against **+1.7 s and
+987 MiB** before; the Qwen's +0.04 s / +60 MiB against +0.08 s / +112 MiB.

**Decode after a fast prompt against after an exact one** (1B, 512-token prompt, 32 greedy
tokens, 4 threads, five pairs, alternating, `decode.jsonl`): exact 46.41-46.80 tok/s, fast
46.25-46.69; medians 46.65 against 46.48 — level (0.996), inside the pair-to-pair spread. The
same pairs' prefills are 225-226 against 230-241 tok/s, the mode's prompt win showing up in the
same run.

## e. Accuracy sanity (not the report)

Two plain-text prompts, 64 greedy tokens, 1B, 4 threads, exact against fast
(`sanity-prompts.md`, `sanity.txt`):

1. *The lighthouse keeper kept a ledger of every ship that passed the point...* — the two agree
   for the first 212 characters (about 45 tokens); exact then has `"...the ledger had become an
   integral part of his daily routine"` and fast `"...the rhythmic tick-tock of the lighthouse's
   beam as it cut through the darkness"`.
2. *In 1847 a chemist in Uppsala dissolved a mineral in acid...* — they differ at the first
   token: exact `"That's close to the actual event..."`, fast `"You're referring to the discovery
   of Radium"`.

As in S5.4d, this is not evidence of a broken body: the Q6_K form is the kernel-tested one, the
tier's fast golden passes, and the host's exact engine differs from `PERGIN_ISA=scalar`'s exact
engine by more than this on the same prompts. S5.4c's report is the number that matters.

## Raw files

| file | what |
|---|---|
| `kqbench.txt` | three runs of table a, including the 896-wide Q5_0 section |
| `runs.jsonl` | every end-to-end run: model, tokens, threads, round, mode, tok/s, load |
| `engines.md` | llama.cpp and ik_llama.cpp raw rows |
| `load.txt` | `/usr/bin/time -v` output for the load/RSS table |
| `forms.txt` | `inspect`'s form size, before and after |
| `decode.jsonl` | the five decode pairs |
| `sanity.txt`, `sanity-prompts.md` | the accuracy sanity runs and their prompts |

