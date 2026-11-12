# The fast prompt pass at long prompts on ARM (S5.4k)

S5.4h left the `--fast-prompt` mode ahead of ik_llama.cpp at 512-token prompts (1.03-1.72×)
and behind at 4 096 (Qwen 2.5 0.5B 0.86× at one thread and 0.91× at ten, Llama 3.2 1B 0.74× at
ten). It named one suspect: the form's bytes, 385.9 MiB on Qwen against the file's 235.5 MiB of
multiplied weights, which ik streams. This round is the account of what a fast pass actually
spends at 512 and at 4 096 tokens, one thread and ten, on both models: the engine's own stage
timers, the form's in-situ rate against its kernel table, attention's split, the chunk, the
thread scaling, and ik's aarch64 wide path read from its source. **Nothing was built**: the
account says the 4 096-token gap is attention's O(n²), not the form's bytes, and the one
fast-mode leftover it did find is worth 2.4 % of one cell.

- Date: 2026-11-12 23:00 – 2026-11-13 01:30 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters: 10 ×
  Cortex-X925 at 3.9 GHz (CPUs 5–9 and 15–19) and 10 × Cortex-A725 at 2.8 GHz, 121 GB unified
  memory, Linux `6.17.0-1008-nvidia`, governor `performance`, rustc 1.99.1. Kernel tier
  `aarch64-neon-i8mm`; `perf` is unusable (`perf_event_paranoid` is 4), so the engine's
  `PEREGRINE_PROF2=full` counters are the profile.
- Peregrine: branch `s5.4k-fast-prompt-long-arm`, on `origin/s5.4h-fast-prompt-q5-arm`
  (`3215a27`, PR #257's head, unmerged). No engine file changed this round; the binary is that
  branch's.
- Models (both regular files under `~/models`, the same SHA-256 as S5.4h):
  `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (397 808 192 B) and
  `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (807 694 464 B).
- ik_llama.cpp `master` at `479074ccac2ac94f06b6d24a9ce26f0f9e9aa6ae`, built CPU-only, at its
  defaults. llama.cpp v0.9.0 `cd959bae99ff4bed22bef2604f45241c09bbb8c8`, CPU only.
- **The host was shared all window**: the 1-minute load average ran 4.0–13.8 and is beside
  every cell. Every cell is therefore **loaded** by the rule that a cell above load 4 is not a
  standing. The one-thread cells are nonetheless stable to ±0.5 % round to round (Qwen 4 096/1
  fast: 176.8, 176.9, 176.9; Llama 512/1: 207.5, 207.4, 207.9), which is why the account's
  per-stage rates are read at one thread; the ten-thread cells swing by up to 2× between rounds
  and are reported with their raw rounds.

## 1. The account

`promptbench` (a text read by the engine, its own `StepStats` and `PEREGRINE_PROF2` slots),
medians of three runs per cell, `--tokens 0` so the prefill is the whole measurement; rows
multiplied are 511 (512-token cells) and 4 095 (4 096). MAC counts are the 2-D matrices a prompt
pass multiplies, from the architecture: Qwen 2.5 0.5B 253.2 M Q5_0 (q, k, v, wo, gate, up) +
2.75 M Q8_0 (attn_v) + 52.3 M Q4_K and 52.3 M Q6_K (ffn_down, twelve layers each); Llama 3.2 1B
704.6 M Q4_K (q, k, v, wo, gate, up) + 268.4 M Q6_K (ffn_down). Attention is causal: the scores
and the mix are `n_head · hd · n(n+1)/2` per layer each.

**Weight matmul by group, GMAC/s over the group's own wall** (`account.jsonl`,
`account_table.py`):

| model | cell | qkv | wo | gate/up | down |
|---|---|---:|---:|---:|---:|
| Qwen 0.5B | 512/1 | Q5_0 199 | Q5_0 213 | Q5_0 221 (0.48 s) | Q4_K+Q6_K 269 (0.20 s) |
| Qwen 0.5B | 4096/1 | Q5_0 204 | Q5_0 216 | Q5_0 225 (3.81 s) | Q4_K+Q6_K 270 (1.58 s) |
| Llama 1B | 512/1 | Q4_K 268 | Q4_K 266 | Q4_K 270 (1.01 s) | Q6_K 277 (0.50 s) |
| Llama 1B | 4096/1 | Q4_K 271 | Q4_K 268 | Q4_K 275 (8.01 s) | Q6_K 279 (3.94 s) |
| Qwen 0.5B | 512/10 | Q5_0 193 | Q5_0 162 | Q5_0 809 (0.13 s) | 499 (0.11 s) |
| Qwen 0.5B | 4096/10 | Q5_0 269 | Q5_0 213 | Q5_0 979 (0.88 s) | 595 (0.72 s) |
| Llama 1B | 512/10 | Q4_K 851 | Q4_K 633 | Q4_K 1501 (0.18 s) | Q6_K 1032 (0.13 s) |
| Llama 1B | 4096/10 | Q4_K 954 | Q4_K 750 | Q4_K 1610 (1.37 s) | Q6_K 1067 (1.03 s) |

**The same pass's other stages** (caller walls; the `scores`/`mix` subtimers are worker-side and
summed over the pool, which for one thread is the wall):

| model | cell | wall | attention | scores | mix | norms | rope | kv write | residual | swiglu | quantise | wait |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 0.5B | 512/1 | 1.30 s | 0.25 s (19.5 %) | 34 | 18 | 0.01 | 0.00 | 0.01 | 0.00 | 0.14 | 0.06 | 0.00 |
| Qwen 0.5B | 4096/1 | 23.30 s | **15.17 s (65.1 %)** | 35 | 18 | 0.10 | 0.04 | 0.05 | 0.03 | 1.14 | 0.45 | 0.00 |
| Llama 1B | 512/1 | 2.51 s | 0.34 s (13.5 %) | 35 | 22 | 0.02 | 0.01 | 0.02 | 0.01 | 0.16 | 0.09 | 0.00 |
| Llama 1B | 4096/1 | 37.09 s | **20.21 s (54.5 %)** | 35 | 23 | 0.16 | 0.05 | 0.13 | 0.06 | 1.26 | 0.68 | 0.00 |
| Qwen 0.5B | 512/10 | 0.88 s | 0.11 s (12.0 %) | 28 | 15 | 0.00 | 0.04 | 0.01 | 0.08 | 0.07 | 0.05 | **0.49** |
| Qwen 0.5B | 4096/10 | 7.48 s | 3.02 s (40.4 %) | 25 | 14 | 0.02 | 0.23 | 0.05 | 0.47 | 0.35 | **3.99** |
| Llama 1B | 512/10 | 0.85 s | 0.09 s (10.7 %) | 34 | 20 | 0.01 | 0.02 | 0.02 | 0.04 | 0.05 | 0.04 | **0.30** |
| Llama 1B | 4096/10 | 8.84 s | 3.83 s (43.4 %) | 28 | 18 | 0.02 | 0.16 | 0.13 | 0.31 | 0.41 | 0.26 | **3.71** |

`scores` and `mix` are GMAC/s of their own half (each is `n_head · hd · n(n+1)/2 · layers`); the
`attention` column's own rate is their sum over the wall (Qwen 4 096/1: 361 GMAC / 15.17 s =
23.8 GMAC/s). "wait" is the caller's post-share wait for the pool, which overlaps the matmul and
attention walls and is never added to them.

**What the table says.**

- **The matmul rate does not depend on the prompt length.** Qwen's Q5_0 groups are 199–221 GMAC/s
  at 512 and 204–225 at 4 096; Llama's Q4_K 266–270 and 271–275. The 4 096-token pass multiplies
  8× the rows and takes 8× the matmul time; nothing about the long pass makes the form slower.
- **Attention is what grows.** Its share of the pass is 19.5 % → 65.1 % on Qwen and 13.5 % →
  54.5 % on the 1B, because its MACs are O(n²) while the matmul's are O(n). At 4 096 tokens the
  score half runs at 35 GMAC/s and the value-mix half (soft-max's max and exponentials included)
  at 18–23 GMAC/s, the S4.9s/S4.9t kernels' measured rates; the same rates at 512.
- **The ten-thread pass does not scale like its stages.** Whole-pass throughput against one
  thread is 2.0× at two, 3.4× at four, 4.5× at six, 5.0× at eight, and 3.1× at ten in this
  window, and the caller spends 3.7–4.0 s of a 7.5–8.8 s pass waiting for its pool. That is the
  host: a co-tenant was on the same cores all window (load 4–14), and the raw rounds move by up
  to 2× (Qwen 4 096/10 fast: 570.5, 619.4, 570.7; ik on the same cell: 592.6, 1021.6, 735.8).
- **The chunk is already right.** At 4 096 tokens and one thread, chunks 64/128/256/512 give
  Qwen 175.1/176.6/177.0/176.5 tok/s (matmul 6.44/6.31/6.27/6.29 s) and Llama
  107.9/110.1/110.6/110.7 (15.27/14.65/14.56/14.56 s): 64 is 2.5 % off on the 1B, and 128–512
  are within 1 %. 512 was chosen on 512-token cells and is not the long prompt's problem.

## 2. What the form reaches inside a pass

The kernel table's rows (S5.4h, `neonbench table`, one worker, `neon-i8mm`) against the same
shapes inside a real pass at one thread (`account.jsonl`):

| format, shape | microbench, 64 rows | microbench, 512 rows | in a 4 096 pass |
|---|---:|---:|---:|
| Q5_0 form, 896-wide (Qwen qkv/wo/gate-up) | 266.3 | 268.3 | **204–225** (0.80–0.84) |
| Q4_K form, whole rows (Qwen ffn_down; Llama qkv/wo/gate-up) | 312.2 | 308.4 | **270** (0.86) / 268–275 (0.87–0.89) |
| Q6_K form, whole rows (Llama ffn_down) | 312.5 | 307.0 | **277–279** (0.89–0.91) |

**Why less than the microbenchmark.** Not the activation's preparation: the interleaved
activation build and the activation quantise together are 0.45 s of Qwen's 23.30 s pass (1.9 %)
and 0.68 s of the 1B's 37.09 s (1.8 %). Not the chunk (§1: ≤1 % across 64–512). Not the
padding alone: the 896-wide row's padded fourth super-block is the 1024/896 = 1.14 traffic S5.4h
measured, and 268.3/1.14 = 235 GMAC/s is the shape's real-row rate, against 204–225 in a pass.
The rest is **DRAM**: the microbenchmark calls the kernel on one tile and nb rows while the tile
is hot, whereas a pass streams the whole 385.9 MiB form (Qwen) or 957.0 MiB (1B) from memory once
per chunk, and the kernel's arithmetic intensity is 8 MAC per activation byte it re-reads per
tile. At one thread the form is 80–91 % of the hot-tile rate; the last 10–20 % is the stream.
At ten threads the same groups read 4.4–5.9× the one-thread rate (gate/up 979/225, down
595/270 on Qwen; 1610/275, 1067/279 on the 1B), not the ten the ten-worker microbenchmark's
2.6–3.1 TMAC/s would allow — and the co-tenant is in that number.

**So the form's 1.64× bytes are not what loses the 4 096-token cell.** At one thread the pass is
65 % attention and 27 % matmul; if the form were free the cell would be 8.2 s, i.e. 500 tok/s,
and ik does 189. ik's wide path is faster on the part both engines still have to compute.

## 3. ik_llama.cpp's aarch64 wide path, read from its source

`ggml/src/iqk/iqk_gemm_kquants.cpp` has a second kernel family under
`// ---- __aarch64__ ----`. `iqk_set_kernels_kquants` (the aarch64 branch at the end of the
file) maps Q4_K and Q5_K to `mul_mat_qX_k_q8_1_x4` with `DequantizerQ4K`/`DequantizerQ5K`, Q6_K
to `mul_mat_q6_k_q8_0_x4`, and leaves every other format to ggml. Those kernels **do not
requantise the matrix into a resident wide form**: `DequantizerQ4K::new_block` reads the file's
own k-quant block (`x[i].scales`, `x[i].qs`), expands it to eight-bit codes in registers
(`Q4bits::prepare`, `HighBit5::apply` for Q5_K), and the MMA is `sdot`/`vmull` against a
`q8_1`/`q8_0` activation built per pass; the `_x4` suffix is four activation rows per call, the
same 4-row granularity as this form's `ACT_GROUP`. So ik streams **0.5625 bytes per Q4_K weight
and 0.8125 per Q6_K weight** — the file's own bytes — and pays the expansion in ALU on every
tile. This form does the opposite: 1.00 byte per weight resident (1.16 padded on Qwen's
896-wide rows), and no expansion. At one thread that trade is strongly in the form's favour
(the exact Q4_K kernel runs at 78 GMAC/s here, the form at 270); at ten threads ik's smaller
byte stream and its per-core kernels win on the models where the pass is bandwidth-bound.

ik's 4 096-token advantage is nevertheless **not** its matmul. Its attention is its own CPU
flash attention (`iqk_flash_attn.cpp`, `iqk_impl.h`: fp16 K/V, a tiled `q·k`/soft-max/`v` over
the cache, with `iqk_repack_k` choosing the K layout), the same design this engine's S4.9s/S4.9t
kernels implement, and it runs ahead of 35/18 GMAC/s.

`perf` is not available on this host (`perf_event_paranoid` = 4), so there is no sampled profile
of either engine; the engine's own counters and this source reading are the profile.

## 4. The eight cells against the two engines

`peregrine bench --tokens 1`, interleaved with the engines inside each round; medians of three
Peregrine rounds per cell (two on Llama 4 096/1, where llama.cpp's own run is 2.5 min) and of
the engine rounds in `llama.txt`/`ik.txt`. Ratios from the raw medians, as always.

| model | cell | exact | fast | fast ÷ exact | llama.cpp | ik | fast ÷ llama.cpp | fast ÷ ik | load |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen 0.5B | 512/1 | 145.3 | **403.2** | 2.77 | 61.8 | 307.0 | **6.53** | **1.31** | 5.4–7.0 |
| Qwen 0.5B | 512/10 | 306.6 | **724.7** | 2.36 | 451.9 | 668.0 | **1.60** | **1.09** | 5.7–6.3 |
| Qwen 0.5B | 4096/1 | 99.0 | **176.9** | 1.79 | 50.6 | 189.0 | **3.50** | 0.94 | 4.0–6.7 |
| Qwen 0.5B | 4096/10 | 246.3 | **570.7** | 2.32 | 321.2 | 735.8 | **1.78** | 0.78 | 6.9–13.6 |
| Llama 1B | 512/1 | 95.8 | **207.5** | 2.17 | 30.2 | 118.8 | **6.87** | **1.75** | 8.7–13.3 |
| Llama 1B | 512/10 | 205.9 | **709.7** | 3.45 | 227.3 | 517.3 | **3.12** | **1.37** | 7.9–9.5 |
| Llama 1B | 4096/1 | 67.8 | **110.8** | 1.64 | 25.9 | 86.7 | **4.28** | **1.28** | 5.4–6.5 |
| Llama 1B | 4096/10 | 292.6 | **551.4** | 1.89 | 186.5 | 584.8 | **2.96** | 0.94 | 6.6–13.8 |

Every cell is **loaded**. Below 1.0, all against ik at 4 096 tokens:

- **Qwen 0.5B 4 096/1 = 0.94×** (ik 188.7, 189.3 in both rounds — the stablest engine cell in
  the table). This is attention: 15.17 s of 23.30 s, with the form's matmul at 80–87 % of its
  hot-tile rate and the exact engine's at 99 tok/s. A 20 % faster score and mix would make the
  cell 20.8 s, 197 tok/s, **1.04×** ik — that is the whole distance.
- **Qwen 0.5B 4 096/10 = 0.78×** (fast 570.7; ik's median 735.8 over rounds 592.6, 1021.6,
  735.8 — a 1.7× spread on a loaded host, so the ratio's true value is anywhere in 0.55–1.2×).
- **Llama 1B 4 096/10 = 0.94×** (fast 551.4 over 558.5, 468.1, 551.4; ik 584.8 over 588.3,
  584.8, 530.6). At 4 096 tokens attention is 43 % of this pass.

Against llama.cpp the fast mode is ahead on all eight (1.60–6.87×).

## 5. What is left, and what it would take

1. **Attention's two kernels at 4 096 tokens (the whole of the one-thread gap).** Scores run at
   35 GMAC/s and the mix at 18–23 against this core's f32/f16 roofs (S4.9t measured the score
   tile at 36.7 and the mix at 25.1 after its four-row bodies). Both are already the four-row
   bodies on `neon-i8mm` and fire on these prompt rows (the 64-row sub-blocks' rows see the same
   tile), so the next step is a different loop shape, not a dispatch miss: a wider store of the
   mix's accumulators (the mix is conversion-bound at one f16→f32 per FMA) and a score tile that
   keeps two K blocks in flight. That is S4.9t-sized work on a new kernel, not a scheduling
   change, and it would move both modes' 4 096-token cells (attention's arithmetic is exact in
   both, and a bit-identical reshape is what the goldens would allow).
2. **The ten-thread cell is the host.** Half of the pass is the caller waiting for a pool whose
   workers the co-tenant deschedules; the same binary measured 879 tok/s at eight threads and
   543 at ten in one sweep. A quiet window is what that cell needs, and this one never had one.
3. **The Q5_0 padded form (2.4 % of the Qwen cell).** The 896-wide row's fourth super-block is
   half zero codes and half zero weights. Skipping its four empty groups is bit-identical (S5.4h
   proved it with a test that rewrites the padded groups and gets the same bits) and would take
   Qwen 4 096/1 from 176.9 to about 181 tok/s (0.96× ik). It was not taken because it changes the
   hot kernel's signature on three architectures for a cell that stays below 1.0, and because the
   account says attention is where the remaining 0.94→1.04 lives.

## What was not done

- No engine change. The chunk (measured, already right), the form's rate (measured, 80–91 % of
  its hot-tile rate at one thread, and the ten-thread number is host-contaminated), and the
  padded tail (2.4 %) were the candidates; the account puts the loss in attention, which is
  S4.9t-sized kernel work rather than a fix this round could take with an A/B inside its budget.
- No `perf` profile: the counter is 4 and the engine's own timers are the profile.
- No golden moved: no file the tests read changed. `git diff --name-status
  origin/main...HEAD -- '*.golden'` is empty.

## Files

`account.jsonl` — one line per `promptbench` run (stage timers, load). `account.sh`,
`account_table.py`, `summarize_account.py` — the driver and the two tables. `round.sh`,
`summarize.py`, `runs.jsonl`, `llama.txt`, `ik.txt` — the eight-cell table and the engines.
`sweep.sh`, `sweeps.txt` — the chunk and thread sweeps.
