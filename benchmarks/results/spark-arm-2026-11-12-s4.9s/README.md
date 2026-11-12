# Qwen3-Embedding F16 at one long text: the value mix's row pair (S4.9s, ARM)

The cache-off ARM table ([`spark-arm-2026-11-12-embeddings-cache-off`](../spark-arm-2026-11-12-embeddings-cache-off/README.md))
has one cell below 1.0: Qwen3-Embedding 0.6B **F16**, one 2 000-token text, **0.96×** llama.cpp
(512 tokens 1.06×, 128 tokens 1.14×, and the Q8_0 file of the same model 1.38–3.20×). This round
is the account of that cell, the one same-bits change the account pointed at, and the re-measured
row. The change: the attention **value mix** now shares each loaded F16 V half-vector across two
query rows and keeps both rows' outputs in registers while the keys walk, instead of loading and
re-storing one row's output per key and reading V once per row.

- Date: 2026-11-12, 17:00–19:30 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters: 10 ×
  Cortex-X925 at 3.9 GHz (CPUs 5–9 and 15–19) and 10 × Cortex-A725 at 2.8 GHz, 121 GB unified
  memory, Linux `6.17.0-1008-nvidia`, governor `performance`, rustc 1.99.1, kernel tier
  `aarch64-neon-i8mm`.
- Peregrine: `origin/main` at `30795cc` plus this round's one commit; the `main` arm is a
  separate worktree at `/var/tmp/pg/wt-s49s-main` with its own target directory, the branch arm
  is the repository build. llama.cpp v0.9.0, `cd959bae99ff4bed22bef2604f45241c09bbb8c8`, the
  harness cache build, CPU only.
- Models:

  | model | bytes | SHA-256 |
  |---|---|---|
  | `qwen3-embed-0.6b-f16.gguf` | 1 197 629 632 | `421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340` |
  | `qwen3-embed-0.6b-q8_0.gguf` | 639 150 592 | `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` |
  | `gte-modernbert-base-F16.gguf` | 299 876 992 | `2cda419ce09e87b1dd5294177a1c7d563e9536700e33652a9bb9e9c503c4e437` |

  Qwen3-Embedding 0.6B is 28 blocks, d_model 1 024, feed-forward 3 072, 16 query heads and 8 KV
  heads of width 128, last-token pooling, a causal `qwen3` file (the embedding path is the
  causal prefill, not the BERT encoder).
- Texts: `/var/tmp/embed-texts/qwen3-embed-0.6b.json` from `benchmarks/embed_texts.py`; every
  timed request uses a different text of the requested length (the harness cycles eight).

## The host is shared, again

A co-tenant ran ClickHouse (eight server threads on the X925 cluster, ~0.9 cores in total), a
Rust build and its service binaries for the whole window; the 1-minute load average was **5–25**,
recorded beside every cell. Two consequences the tables below are built around:

- the **one-thread** cells are stable (round-to-round spread 0.1–1.5 %), so they carry the
  account and the A/B;
- the **ten-thread** cell swings by up to 3.8× for the *same* arm (5.79–22.07 s for one
  2 000-token text), so its absolute medians are host state, not a baseline. The row below is
  therefore reported beside the quiet-host cache-off table, and the change's own effect is
  shown by a paired `--main`/`--branch` run in the same window and by the ten-thread
  attention/matmul ratio (the matmul is not touched by the change).

`perf` is not usable on this host: `perf stat` refuses (`perf_event_paranoid` is 4), so the
engine's own `PEREGRINE_PROF2=full` counters are the profile, and llama.cpp is read from its
timings and source rather than a sampled profile.

## 1. The account, before the change (one thread, core 5)

`cellprof` (the engine's own `StepStats` plus the `PEREGRINE_PROF2=full` buckets), three runs
of the L512 and L2000 cells, medians; the text counts include the special tokens (528 and 2000).
`texts/s` is the cell's wall rate.

| cell | tokens | wall | matmul | attention | of which mix | of which scores | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| L512 b1 | 528 | 7.37 s | 6.25 s (84.8 %) | 0.95 s (12.8 %) | 0.64 s | 0.27 s | 0.17 s (2.3 %) |
| L2000 b1 | 2 000 | 37.44 s | 23.63 s (63.1 %) | 13.14 s (35.1 %) | 8.94 s (23.9 %) | 3.95 s (10.6 %) | 0.64 s (1.7 %) |

The arithmetic, from the model's own shapes (per token per layer: q 2.097, k and v 1.049 each,
o 2.097, gate and up 3.146 each, down 3.146 MMAC; ×28 layers = 440.4 MMAC/token; attention is
causal: 16 heads × 128 dims × n(n+1)/2 scores and the same again in the mix):

| stage | GMAC at 512 | GMAC at 2 000 | GMAC/s at 512 | GMAC/s at 2 000 |
|---|---:|---:|---:|---:|
| matmul (F16) | 232.5 | 880.8 | 37.5 | **37.3** |
| attention | 16.0 | 229.5 | 16.9 | **17.5** |
| — scores | 8.0 | 114.7 | 29.2 | 29.0 |
| — mix | 8.0 | 114.7 | 12.5 | **12.8** |

**The roof.** `fpbench` on this core (one thread, 1 024 × 1 024 and 1 024 × 3 072, 64–128 rows)
puts the packed F16 kernel at **47.0–47.6 GMAC/s**, f32 at 62.6–65.8 and bf16 at 57.7–58.0; the
f32 FMA roof of the core is about 66–72 GMAC/s at 3.9 GHz. So the F16 matrix product is at
**80 % of its own kernel** (37.3 of 47.0) — the kernel, not the pass, is the matmul's limit, and
the kernel is itself 25–30 % below the f32 one because it widens each panel before the FMA loop
(S4.9p's trade).

**The losing stage is the value mix.** At 528 tokens it is 8.7 % of the pass; at 2 000 tokens it
is **23.9 %**, 8.94 s, and runs at **12.8 GMAC/s — 19 % of the core's f32 roof**, below the score
kernel's 29.0 on the same tile and a third of the matmul. It is not arithmetic-bound: reading
`mix_f16` in `crates/peregrine-core/src/attn.rs` shows why. The loop is position-outer,
dimension-inner: for every key it loads the row's 128-float output, multiplies the stored V half
by the softmax weight, adds, and stores the output back — 4 loads and 2 stores of `float32x4_t`
of output per 8 dimensions per key — and a second row sharing the KV head loads the same V
half-vector again. At 2 000 positions one row touches its 512-byte output row 2 000 times per
head slice and V traffic is 2 000 × 2 000 × 128 × 2 B = **1.02 GB per head**, 16 heads × 28
layers = **459 GB** for the mix alone.

**Scaling 1 → 10 threads** (the same window; 1-thread medians above, 10-thread medians of 15
alternating runs, `ab-10t.txt` + `ab-10t-long.txt`, load 9–22):

| stage | 1 thread | 10 threads | scaling |
|---|---:|---:|---:|
| whole pass | 36.76 s | 11.85 s (5.85–17.76) | **3.1×** |
| matmul | 23.63 s | 5.35 s (3.17–7.26) | 4.4× |
| attention | 13.14 s | 2.84 s (1.55–3.42) | 4.6× |

On this loaded host ten workers get about four cores' worth of throughput; the quiet-host
cache-off table reaches 7.7× on the same cell (0.027 → 0.21 texts/s). The stages scale alike
within a run, which is why the same-run attention ÷ matmul ratio is the usable ten-thread
statistic.

**llama.cpp's side.** At 2 000 tokens llama.cpp (v0.9.0) runs the same 1 100 GMAC in about
4.55 s on a quiet host (0.22 texts/s, 242 GMAC/s over ten cores) against Peregrine's 4.76 s
(0.21), i.e. the two are within 5 % there. Its matrix product is `ggml_compute_forward_mul_mat`
with the generic `ggml_vec_dot_f16` (`ggml/src/ggml-cpu/vec.cpp:264`), and its attention is
`ggml_compute_forward_flash_attn_ext_tiled` (`ggml/src/ggml-cpu/ops.cpp:9080`), which walks
**64 query rows × 64 keys** at a time, converting each K and V tile to f32 once
(`K_f32`, `V32`; `GGML_FA_TILE_Q = GGML_FA_TILE_KV = 64`) and reusing it across all 64 query
rows. Peregrine shares a loaded V vector across **two** rows: on the same cell llama.cpp's mix
reads each V tile once per 64 rows, ours once per 2, a 32× difference in V traffic. That is the
structural reason the gap appears at long text and not at 128 tokens, where K/V fit in cache.

## 2. The change: the value mix's row pair

`crates/peregrine-kernels/src/neon.rs` gains `mix_f16_pair` beside its scalar twin
`mix_f16_pair_scalar`, and `crates/peregrine-core/src/attn.rs` calls it from `run_rows` when two
rows of one KV head see the same positions (the F16-direct tile, which is every prefill on
aarch64). One commit.

- **What moves**: the output row is read once per eight dimensions instead of once per key, and
  each stored V half-vector is widened once and used by both rows. The pair measures
  **21.3 GMAC/s against 12.8** on the 2 000-token cell (+66 %), and the mix time falls 8.94 →
  5.39 s.
- **What cannot move**: each row keeps its own accumulator, multiplies by its own α once, and
  adds `p_j · v_j` in ascending key order with a separate multiply and a separate add, the same
  operands in the same order as the single-row body (no fused multiply-add is introduced). The
  test `the_neon_pair_mix_is_each_rows_own_mix` checks it against **both** `mix_f16` and the
  scalar twin for every length 1..=70, 127/128/129 and 511/512/513, at widths 64, 128, 72, 80 and
  16, with the edge alphas a softmax produces (0, 1, −0.0, 3.5) — every output `to_bits` equal.
  No golden file changed (`git diff --stat origin/main...HEAD -- '*.golden'` is empty).

**The A/B, one thread, five interleaved rounds** (`ab-1t.txt`; two builds, `main` and the
branch, alternating inside each round, load 5.4–23.4):

| cell | main | branch | change |
|---|---:|---:|---:|
| L512 pass | 7.28 s | 6.90 s | **−5.2 %** |
| L512 attention | 0.946 s | 0.687 s | −27.4 % |
| L512 mix | 0.638 s | 0.388 s | **−39.2 %** |
| L2000 pass | 36.76 s | 32.97 s | **−10.3 %** |
| L2000 attention | 13.00 s | 9.43 s | −27.5 % |
| L2000 mix | 8.93 s | 5.39 s | **−39.7 %** |
| L2000 scores | 3.94 s | 3.93 s | 0.0 % (−0.3 % across the five rounds) |
| L2000 matmul | 23.09 s | 23.00 s | −0.4 % (untouched) |

Five rounds, and every round of both cells is inside ±1 % of its arm's median (`runs` in
`ab-1t.txt`). **At ten threads** the same arms do not read: 15 alternating runs of the same cell
span 5.79–22.07 s for main and 5.85–17.76 s for the branch as the co-tenant moves, so the
absolute medians (15.61 s against 11.85 s) are not a measurement. The same-run control — the
attention time divided by that run's matmul time, which this commit does not touch — is **0.58
(main) against 0.48 (branch)**, the direction and roughly the size the one-thread A/B predicts
(a 27 % cut of a 35 % share is 9.5 points of the pass).

## 3. The row, re-measured (ten threads, this window)

`run.py` here is the cache-off harness (one server per engine per model, both up, rounds
interleaved by engine, `--opt EMBED_CACHE_MB=0`, one text per request, distinct texts). The
branch's numbers, three rounds per cell, medians of texts/s, against llama.cpp (v0.9.0, `-np 32
-c 2056×32 -b 2048 -ub 2048 -fa auto`, CPU only):

| cell | Peregrine texts/s | llama.cpp texts/s | ratio | Peregrine spread |
|---|---:|---:|---:|---:|
| one text, 16 tok | 14.22 | 4.95 | **2.87×** | 50.7 % |
| one text, 128 tok | 1.27 | 1.45 | **0.88×** | 59.0 % |
| one text, 512 tok | 0.345 | 0.367 | **0.94×** | 31.9 % |
| one text, 2 000 tok | 0.122 | 0.149 | **0.82×** | 12.3 % |
| batch 32 × 128 tok | 1.49 | 1.385 | **1.08×** | 30.7 % |
| mixed batch of 32 | 0.805 | 0.785 | **1.03×** | 44.4 % |

Read this against the quiet-host table, not as a new baseline: every cell is 1.7–2.8× below it
(2 000 tokens 0.122 against 0.21; 512 0.345 against 0.61), and the 16-token cell is a 70–200 ms
request whose spread is 50 %. The **paired** comparison in the same window is the change's
number: back-to-back three-round harness runs of the two builds on the same cells give

| cell | main ÷ llama.cpp | branch ÷ llama.cpp |
|---|---:|---:|
| F16, 512 tok | 0.76× | **0.80×** |
| F16, 2 000 tok | 0.67× | **0.75×** |
| Q8_0, 2 000 tok | 0.82× | **0.93×** |

So on a quiet host, where the same cell read 0.96× before the change, the 1-thread A/B's −10.3 %
of the pass and −27.5 % of attention put it at **about 1.05–1.07×**; the loaded window's
0.67 → 0.75 is the same direction. **The cell below 1.0 has not been re-read on a quiet host in
this window** and is marked so below.

### Guards

| row | 16 | 128 | 512 | 2 000 | 32 × 128 | mixed | quiet-host reference (main) |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-Embedding F16 (branch) | 2.87× | 0.88× | 0.94× | 0.82× | 1.08× | 1.03× | 1.57/1.14/1.06/0.96/1.17/1.10 |
| Qwen3-Embedding Q8_0 (branch) | 4.10× | 1.42× | 1.04× | 0.88× | 1.84× | 1.25× | 3.20/1.97/1.71/1.38/1.99/1.80 |
| gte-modernbert-base F16 (branch) | 2.27× | 1.38× | 1.27× | 1.35× | 1.15× | 1.20× | 1.59/1.40/1.34/1.42/1.33/1.37 |

Ratios are Peregrine ÷ llama.cpp in this window; the last column is the quiet-host cache-off run
of `main`, for shape only. The Q8_0 row executes the same causal path as F16 and gains the pair
too (its 2 000-token cell 0.82× → 0.93× against llama.cpp in the paired runs). **gte-modernbert
is a bidirectional encoder** (`bert`/`modern-bert` path, `bidir.rs`) and never calls the causal
mix: its numbers move only with the host (−13 % to +4 % of the reference, inside this window's
spread), which is the guard's point.

## 4. What is left

- **A quiet-host re-read of the Qwen3-Embedding F16 2 000-token cell** (and the row), which is
  what `docs/benchmark.md` should quote. The change is in; the window was not.
- **The full K/V panel.** llama.cpp's 64 × 64 tile is the next step: widening a 64-key K and V
  tile once for 64 query rows instead of sharing across 2. The account says it is the remaining
  structural gap (V traffic is still 32× llama.cpp's at 64 rows), but it is a rewrite of the
  score and mix orchestration, not a loop shape, and it is not this round's.
- **The F16 matmul's 80 % of its own kernel** and the kernel's 25–30 % gap to f32 (the widening
  of each panel) are S4.9p's trade and were not touched.

## Files

`cellprof-1t.txt`, `cellprof-10t.txt` (the account, before the change), `ab-1t.txt`,
`ab-10t.txt`, `ab-10t-long.txt` (the A/B), `qwen3-f16-*.json`, `qwen3-q8-*.json`,
`gte-f16-*.json` (the harness, per round), `run.py` (the harness copy with a `--cells` filter).
