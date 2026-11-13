# The prompt pass's tails, its form's fill and its KV write, and the int8 key measured (S5.4o)

Where S5.4j left the opt-in 8-bit prompt pass (0.82-0.97× ik_llama.cpp's `-rtr` path, the caller's
wait 10.4 % of a 512-token pass and attention 38 % of a 4 096-token one), the two things the step
was given: **is an int8 key the missing piece at 4 096 tokens** (Part A, measurement only), and
**what the pass's own fork–joins and serial steps are worth at 512 tokens** (Part B, scheduling
only). Part A's answer is **no** — and it corrects S5.4j's premise. Part B takes the two 512-token
twelve-thread cells from **0.823/0.816 to 0.925/0.957** against the fork's `-rtr` path, with no
other cell lower than it was.

- Date: 2026-11-13 (08:00–14:30 UTC+7).
- Host: AMD Ryzen AI 9 HX 370 (Zen 5), 12 physical cores / 24 logical, AVX-512 (VNNI, GFNI, VBMI,
  BF16), WSL2, 11.7 GiB, kernel `6.18.33.2-microsoft-standard-WSL2`. The 1-minute load is recorded
  beside every raw line and read **1.1-11.2** across this session (a co-tenant came and went), so
  every ratio here is recomputed from **this window's** medians and the same-window "before" cell,
  never from a previous report's numbers. The clock still runs ~10 % fast (MONOTONIC), the same for
  every engine.
- Branch: `s5.4o-fast-pass-shape`, from `origin/s5.6-pool-contended` (neither #263's nor #265's
  commits were on `main` when the step started — `git grep -c run_claimed origin/main --
  crates/peregrine-core/src/pool.rs` printed nothing, and so did `pub mod fast`), with `origin/main`
  merged before the PR. Rust 1.99.1, AVX-512 tier.
- ik_llama.cpp: `~/.cache/peregrine-bench/ik_llama.cpp-479074c` (`479074c`), its `llama-bench`.
  llama.cpp: `~/.cache/peregrine-bench/llama.cpp-v0.9.0` (v0.9.0).
- Models: Llama 3.2 1B Instruct Q4_K_M `~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf`
  (807 694 464 B), Qwen3 1.7B Q4_K_M `/var/tmp/pg/s54c/models/Qwen3-1.7B-Q4_K_M.gguf`
  (1 282 439 264 B) — both regular files (`test -L` clean).
- Corpus: `~/.cache/peregrine-bench/s52b/corpus.txt` for the accounts and the `promptbench` cells,
  and wikitext-2's chunks through the S5.4c base files (`/var/tmp/pg/s54c/base/*.base.bin`) for the
  `kld` runs.

## 0. What this round found, in one paragraph

**An int8 key is not what makes the fork's `-rtr` path fast, and our `KV=i8` cache does not close
the 4 096-token gap.** The fork's `-rtr` cells keep **f16** K/V and score them with an f32 × f16 dot
product (`iqk_gemm_default_floats`, 8×8 tiles); the int8 dot S5.4j named is taken only when the
cache is quantised, which its measured cells are not. Our `KvPrec::I8` costs **7-15 %** of the
prompt at 512 tokens and **11-14 %** at 4 096 on twelve threads (its attention is 6-45 % slower
because the tile is dequantised to f32 and the f16-direct read is gone), buys nothing, and costs
**+0.00015 / +0.00056 nats** of mean KL on top of the fast mode's 0.0035/0.0173 — cheap, but for no
speed. The fast order *does* work with the int8 cache (same walk, no fallback, no panic). **At 512
tokens, the pass's own shape was worth three levers**: the dynamic row grain was a 287 µs tail a
join (11 % of the pass) and a 32nd of the row space per worker, capped at 64 rows, recovers
**+5.8-6.7 %**; the form's interleaved activation fill was **78.9 ms, 6.4 %** of a pass (S5.4j's
estimate of 1.5 % was low by the size of the `down` group), and filling it on the pool is worth
**+7.5 %** at 512/12 and +3.5 % at 4 096/12; and the serial KV write (**37.4 ms, 3.0 %**) now rides
the preparation pass's own join, **+3.3 %**. The caller's wait falls from 138 ms (12.5 %) to 92 ms
(8.9 %) and the parks from 1 009 to 173. The two 512-token twelve-thread cells are **0.925 and 0.957**
against `-rtr` (were 0.823 and 0.816), the exact engine gains 1.6 %, decode and the ten-user cell do
not move, and **no golden moved**.

## 1. Method

### Part A: the grid and its tools

Six arms, one window, interleaved inside every round: `exact`, `fast`, `fast` with `KV=i8`, `exact`
with `KV=i8`, ik at its defaults and ik with `-rtr`; two models at 512/12, 4 096/12 and 4 096/1.
Peregrine's arms are `promptbench` (a real text, `--decode 32`, so the same run gives the pass, the
decode and the attention bucket), the fork's are its `llama-bench`; 12-thread cells are pinned to
`taskset -c 0-23` and one-thread cells to cpu0. Medians of five rounds for the 512-token cells and
4 096/12, three for the 4 096-token one-thread cells (they are 2.5-11 minutes an arm). One
`ik -rtr` 512-token cell failed once and was re-run, so that cell has six values.

Two measurement-tool changes made the grid possible, and they are engine-neutral:

- `promptbench --kv f16|f32|i8` — it hard-coded `KvPrec::F16`, so no profile could be taken on the
  cache a cell used. `peregrine bench` builds its session from `SessionCfg::default()` and does
  **not** read `PEREGRINE_KV`, so the env variable alone would not have changed its cache either.
- `kld --ref-kv f16|f32|i8` — `--ref-self` built both caches at the candidate's precision. A3 needs
  the candidate on `i8` against the **exact engine on f16**, which needed the reference cache's
  precision as its own argument.

The `kld` arms of A3 were run on the Part A build; the step's changes are scheduling only (Part B
pins that with the goldens and the parallel/serial byte tests), so the KL numbers are the same
build's bits.

### Part B: the account, then one lever per commit

`promptbench` with `PEREGRINE_PROF2=full` (the caller's `StepStats` buckets, every `prof` slot, and
the pool's join accounting: joins, wait, share, straggler gap) and `PEREGRINE_ADAPTIVE_LOG=1` (the
park histogram) on the merged build first. Then each lever is an A/B of two builds alternating
inside each round, five to seven rounds, on the cell it targets; the step's final standings are a
six-arm sweep (this step's build, the build before it, the exact engine, ik's two paths and
llama.cpp) interleaved in the same rounds, and every ratio is recomputed from those medians.
`scripts/ab.sh` and `scripts/abenv.sh` are the two-build and two-environment A/Bs; `scripts/table.py`
recomputes the tables from the raw files.

## 2. Part A: is an int8 key the missing piece at 4 096 tokens?

### 2.1 What the fork does (A1) — a correction to S5.4j §5k

S5.4j's last paragraph said the fork's flash attention "quantises the query and the key to int8".
That was an inference from timing. The source says otherwise:

- `-rtr` is `--run-time-repack` (`examples/llama-bench/llama-bench.cpp:822`,
  `common/common.cpp:2537`): it repacks the **weights** at load and disables mmap. The cache is not
  touched.
- A `llama-bench` run's cache is **f16** for both K and V by default
  (`examples/llama-bench/llama-bench.cpp:299-300`), and flash attention is on by default (`:309`).
  The S5.4i/S5.4j sweeps passed neither `-ctk`/`-ctv` nor `-fa`.
- With a f16 cache `iqk_flash_helper_T` dispatches `GGML_TYPE_F16 → HelperF16`
  (`ggml/src/iqk/fa/iqk_fa_templates.h:2212`), whose `load` widens the stored halves
  (`:198-208`). The score is `FlashQKfp32::multiply_mask_kq` (`:1262`) →
  `iqk_gemm_default_floats` with the query in `float` on x86 (`:1388-1391`), i.e.
  `mul_mat_Qx_Qy_MxN_fa4<QFT<float,8>, QFT<ggml_half,8>>` (`iqk_gemm_floats.cpp:665`). The value
  mix is f16 values × f32 weights (`FlashQKV::accumulate_qkv`, `:996-1015`).
- The int8 path (`block_q8_KV` `:210`, `HelperQ8KV` `:217`, `iqk_gemm_q8kv_fa` `:1325`) is only
  taken for a **quantised cache** (`-ctk q8_0` → `HelperQ80` `:2216`, `GGML_TYPE_Q8_KV` →
  `HelperQ8KV` `:2229`). The measured `-rtr` cells are not.

The int8 dot is real, but it is the fork's *other* path: on the same binary, `-ctk q8_0 -ctv q8_0`
at 4 096/12 and `-rtr` gives **369.9 / 372.5 / 369.9 → 442.2 / 445.3 / 551.8 tok/s** (three rounds
each) — a 20-49 % gain that the fork's published cells do not take and that our f32-dequantising
cache cannot reach.

### 2.2 The grid (A2)

Prompt tok/s, medians; `fast+i8` is `--fast-prompt` with the Hadamard-rotated int8 cache,
`exact+i8` is the default engine on it:

| model | cell | exact | fast | fast+i8 | exact+i8 | ik | ik `-rtr` |
|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 512 / 12 | 322.2 | 406.7 | 376.5 | 253.8 | 421.7 | 528.7 |
| Llama 3.2 1B | 4 096 / 12 | 247.6 | 329.4 | 293.2 | 212.5 | 361.5 | 370.9 |
| Llama 3.2 1B | 4 096 / 1 | 38.4 | 51.6 | 49.8 | 37.9 | 48.9 | 55.8 |
| Qwen3 1.7B | 512 / 12 | 194.3 | 261.5 | 221.8 | 155.1 | 278.5 | 286.3 |
| Qwen3 1.7B | 4 096 / 12 | 160.3 | 207.0 | 177.8 | 130.1 | 223.5 | 238.1 |
| Qwen3 1.7B | 4 096 / 1 | 26.0 | 34.2 | 32.4 | 25.2 | 32.5 | 37.8 |

The cache's cost and its attention bucket (the profile's `StepStats::attention`, which includes the
32 decoded rows):

| cell | fast+i8 ÷ fast | exact+i8 ÷ exact | attention, fast → fast+i8 | attention, exact → exact+i8 | decode, fast → fast+i8 |
|---|---|---|---|---|---|
| 1B 512 / 12 | 0.926 | 0.788 | 156.6 → 226.4 ms (+45 %) | 141.9 → 375.2 ms (+164 %) | 58.6 → 56.3 |
| 1B 4 096 / 12 | 0.890 | 0.858 | 4 668 → 5 146 ms (+10 %) | 5 153 → 6 455 ms (+25 %) | 49.4 → 45.7 |
| 1B 4 096 / 1 | 0.965 | 0.987 | 32 467 → 34 772 ms (+7 %) | 34 501 → 36 289 ms (+5 %) | 12.5 → 11.1 |
| Qwen3 512 / 12 | 0.848 | 0.798 | 278.2 → 391.7 ms (+41 %) | 269.9 → 645.5 ms (+139 %) | 39.1 → 35.5 |
| Qwen3 4 096 / 12 | 0.859 | 0.812 | 7 620 → 8 094 ms (+6 %) | 8 091 → 10 263 ms (+27 %) | 28.2 → 26.4 |
| Qwen3 4 096 / 1 | 0.947 | 0.969 | 49 343 → 53 981 ms (+9 %) | 53 356 → 56 224 ms (+5 %) | 6.8 → 6.3 |

The mechanism is in the code, not the timing: our scores are an f32 dot over a tile that
`KvCache::decode_block` decodes to f32 in every precision, so the int8 cache adds an `i8 → f32`
widening and a per-vector scale multiply (`decode_block_i8_avx2`) and removes the f16-direct tile
the exact path reads (`run_f16`) — it never introduces an integer dot. At one thread the pass is
matmul-bound and the cache costs only 1-3 %; at twelve the attention bucket is what pays.

### 2.3 Accuracy (A3)

`kld --ref-self --ref-kv f16`, the S5.4c corpus (wikitext-2, 40 chunks × 512 tokens, 10 200
scored positions), the candidate's cache against **the exact engine on f16**:

| model | (2) fast, f16 | (3) fast, i8 vs exact f16 | (4) exact, i8 vs exact f16 |
|---|---|---|---|
| Llama 3.2 1B | 0.003464 ± 0.000047, 96.667 ± 0.178 % | **0.003613 ± 0.000045**, 96.510 ± 0.182 % | 0.002852 ± 0.000038, 96.804 ± 0.174 % |
| Qwen3 1.7B | 0.017336 ± 0.001186, 94.196 ± 0.232 % | **0.017898 ± 0.000486**, 93.922 ± 0.237 % | 0.017263 ± 0.001206, 94.500 ± 0.226 % |

The int8 cache alone is 0.00285 (1B) and 0.01726 (Qwen3) nats from the f16 exact engine; on top of
the fast mode it is +0.00015 (+4.3 % of the fast mode's own distance) and +0.00056 (+3.2 %), with
the same-top-token share down 0.16 and 0.27 points. Inside the re-baselined 0.0040 bound on the 1B;
the Qwen3 number moves from 0.0173 to 0.0179, also inside the run's error. **Cheap in accuracy,
and it buys no speed** — which is the whole question.

### 2.4 The conclusion (A4), as numbers

- **How much of the 4 096-token gap `KV=i8` closes: none.** At 4 096/12 it takes the fast cell from
  329.4 to 293.2 tok/s (1B) and 207.0 to 177.8 (Qwen3), i.e. **−11 % and −14 %**, and the same
  window's `fast ÷ ik -rtr` is 0.888/0.870, so the int8 cache would take those two cells to
  **0.79 and 0.75**.
- **What it costs on top of the fast mode's 0.0035**: +0.00015 nats on the 1B (+4.3 % of itself,
  96.667 → 96.510 % same top token) and +0.00056 on Qwen3 (+3.2 %, 94.196 → 93.922 %).
- **The fast order applies to the int8 path, and does not bypass it.** `run_range_fast` →
  `fast_walk` (`crates/peregrine-core/src/attn.rs:748`) decodes its tile with
  `kv.decode_block(...)` into the f32 scratch and Hadamard-rotates the query when `kv.rotated()`;
  the fast module never looks at the precision, `attn.rs:4173` runs the int8 cache through the same
  walk as a test, and the grid above *is* that path. No panic, no wrong result, no silent fallback.
- **What the gap is**: the fork's f16-key kernel shape (its 8×8 tile reads the stored halves) against
  our fast order's widened f32 tile. An integer dot is a third option — the fork's own quantised
  cache is worth 20 % to it — and it needs a quantising attention kernel with its own accuracy
  report. Not a cache flag.
- **No default changed.** `KV=i8` stays an opt-in measurement cache.

## 3. Part B: the pass's fork–joins and serial steps

### 3.1 The account on the merged build (B1)

Llama 3.2 1B Q4_K_M, `--fast-prompt`, 512 tokens, 12 threads; "before" is the merged S5.6+S5.4j
build, "after" this step's tree, one run each in the same window:

| term | before | after |
|---|---|---|
| joins in the pass (and a layer) | **481** (15.0) | **609** (19.0) |
| the caller's post-share wait | 138 ms, **12.5 %** | 92 ms, **8.9 %** |
| wait per join | 287 µs | 151 µs |
| straggler gap, mean / max | 0.062 / 2.049 ms | 0.033 / 0.416 ms |
| parks in one pass | **1 009** (603 of 1-5 ms) | **173** (55 of 1-5 ms) |
| attention | 114.6 ms | 87.0 ms |
| KV write | **37.4 ms (3.0 %)** | **0.035 ms** |
| the form's interleaved fill | **78.9 ms (6.4 %)** | ~7 ms over 128 jobs |
| the pass | 1.230 s / 415.4 tok/s | **1.033 s / 494.7 tok/s** |

The 15 joins a layer are the four matmul groups, attention, the preparation pass, two residuals,
two norms, SwiGLU, the attention output's quantise — and, new, the form's fill (4 a layer). The
2026-11-13 S5.4o brief's first lever, "q/k/v and gate/up as ONE fork–join", **is already the
merged tree's shape** (`matmul_group`, the fused group since S4.9g); the account has no such join
left to remove. What it did show: the wait was mostly the **tail of the dynamic row partition**,
and the two serial steps were 9.4 % together, not 3.4 %.

### 3.2 Lever 1: the dynamic grain follows the tail

`matmul_group` splits its row space over the pool in `total/(width·8)` chunks clamped to [8, 256].
The caller's share ends when the counter is exhausted, so it then waits for a chunk another worker
claimed just before it — one chunk's compute, 287 µs a join. One 32nd per worker, capped at 64
rows (a finer grain than that pays for its claims; a *decode* pass — `nb == 1`, 224 joins a token,
no tail — keeps the old grain, because the first cut cost **11 % of the decode rate**):

| cell | before | after | ratio |
|---|---|---|---|
| fast 512/12, 5 rounds | 444.5 | **470.1** | **1.058** |
| fast 512/12, second A/B, 5 rounds | 447.1 | **477.2** | **1.067** |
| fast 4 096/12, 5 rounds | 334.1 | **339.7** | **1.017** |
| exact 512/12, 5 rounds | 339.2 | **344.8** | **1.016** |
| exact 512/1, 5 rounds | 52.6 | 52.4 | 0.997 |

### 3.3 Lever 2: the form's activation fill, on the pool

`KqActs::fill_with` was 78.9 ms of a 1.033 s pass — 6.4 %, and 112 MiB of interleaving a pass, not
S5.4j's estimated 24 MiB (the `down` matmul's input is 8 192 wide, 32 super-blocks, four times the
other groups). Each group of four activation rows is an independent transpose, so the layout
definition moved into `kq::fill_group` and `blocks::linear::fill_form` gives each worker disjoint
groups (`Disjoint`), with the whole-block sums only when the 8-bit form asks. Below 256 KiB of
interleaved codes the join costs more than the copy, and the serial fill is kept.

| cell | before | after | ratio |
|---|---|---|---|
| fast 512/12, 5 rounds | 454.4 | **488.7** | **1.075** |
| fast 4 096/12, 5 rounds | 337.2 | **349.0** | **1.035** |

`the_parallel_form_fill_is_the_serial_fill` pins the six buffers byte for byte (71 rows, so the last
group is padded; with and without the 8-bit form's whole-block sums).

### 3.4 Lever 3: the KV write rides the preparation join

The write was 37.4 ms of the pass — 4.2 M f16 conversions written one position at a time into a
cache whose consecutive positions are a page apart, latency-bound at one thread as much as twelve.
A prompt chunk is **one run of consecutive positions**, so each row's write is
`(layer, 0..n_kv_head, pos)` and rows have distinct positions: `prepare_rows_kv` writes the row the
worker just rotated (`KvWriter`, raw buffers with a disjointness argument — the recurrent state's
workers already do this), and only `advance` is left on the caller. A multi-run pass (a prompt
chunk beside other users' decode rows) and a non-consecutive position list keep the serial write.

| cell | before | after | ratio |
|---|---|---|---|
| fast 512/12 (3 rounds) | 476.7 | **491.6** | **1.031** |
| fast 512/12, next A/B, 5 rounds | 488.0 | 484.7 | 0.993 (this window's spread) |
| fast 4 096/12, 5 rounds | 347.1 | **358.5** | **1.033** |

`a_worker_write_is_the_serial_puts_bytes` pins the writer to `put`'s bytes on f16, f32 and int8,
rotated and unrotated.

### 3.5 The three together

| cell | before (S5.6+S5.4j) | after | ratio |
|---|---|---|---|
| fast 512/12, 7 rounds | 429.8 | **481.1** | **1.119** |
| fast 512/1, 7 rounds | 76.2 | 75.6 | 0.992 |
| exact 512/12, 5 rounds | 339.2 | **344.8** | **1.016** |
| exact 512/1, 5 rounds | 52.6 | 52.4 | 0.997 |
| fast 1 024/1 (1B), 3 rounds | 48.6 | **49.9** | **1.025** |
| fast 1 024/1 (Qwen3), 3 rounds | 34.1 | 33.9 | 0.993 |
| fast 512/12, **after merging `origin/main`** (S5.4n and the review fixes), 5 rounds | 442.1 | **484.9** | **1.097** |

**Decode did not move**: after a 512-token fast prompt, 60.4-61.7 tok/s before and 59.6-62.2 after
(three pairs each). **The ten-user cell** (10 slots, 512-token prompts, one pool): prefill
453.4/474.0 → **484.7/524.7**, decode 301.8/300.0 → 293.5/300.0 (level), its pass's attention
534.7/524.8 → 525.3/508.6 ms. A pooled row is never a fast row, so its matmuls are the exact ones,
and its `nb > 1` takes the new grain: prefill up, decode level.

## 4. The eight cells, before and after (B3)

Prompt tokens per second, medians of five interleaved rounds for the four 512-token cells, three
for the 4 096-token 12-thread cells, three for Llama 3.2 1B at 4 096/1 and **one** for Qwen3 1.7B's
(as S5.4j left it: 6.5 minutes an arm at one thread in this window); llama.cpp at 4 096/1 on the 1B
has two rounds. "before" is the merged build timed in the same rounds.

| Model | prompt / threads | exact | fast before | **fast now** | ik (defaults) | ik (`-rtr`) | llama.cpp | fast ÷ exact | fast ÷ ik | fast ÷ **ik `-rtr`** | fast ÷ llama | before → after (fast) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 512 / 12 | 342.0 | 442.7 | **498.0** | 418.0 | 538.3 | 269.9 | 1.456 | 1.191 | **0.925** | 1.845 | +12.5 % |
| Llama 3.2 1B | 512 / 1 | 52.5 | 74.7 | **76.5** | 62.2 | 80.9 | 39.0 | 1.457 | 1.230 | **0.946** | 1.962 | +2.4 % |
| Llama 3.2 1B | 4096 / 12 | 254.9 | 325.5 | **356.1** | 308.4 | 452.2 | 201.8 | 1.397 | 1.155 | **0.787** | 1.765 | +9.4 % |
| Llama 3.2 1B | 4096 / 1 | 38.26 | 51.27 | **51.08** | 49.37 | 57.09 | 30.13 | 1.335 | 1.035 | **0.895** | 1.695 | −0.4 % |
| Qwen3 1.7B | 512 / 12 | 221.8 | 288.3 | **338.2** | 277.0 | 353.5 | 181.2 | 1.525 | 1.221 | **0.957** | 1.867 | +17.3 % |
| Qwen3 1.7B | 512 / 1 | 35.9 | 48.1 | **51.5** | 41.4 | 53.8 | 25.7 | 1.435 | 1.244 | **0.957** | 2.004 | +7.1 % |
| Qwen3 1.7B | 4096 / 12 | 166.6 | 208.2 | **233.3** | 201.5 | 270.1 | 124.6 | 1.400 | 1.158 | **0.864** | 1.872 | +12.0 % |
| Qwen3 1.7B | 4096 / 1 | 25.4 | 34.0 | **33.2** | 32.9 | 37.7 | — | 1.309 | 1.010 | **0.883** | — | −2.2 % (one round) |

**Cells below 1.0, all named.** Against ik's `-rtr` path, **all eight** (0.787-0.957); against ik at
its defaults, **none** (1.010-1.244); against the exact engine and llama.cpp, none. The step's bar —
the two 512-token twelve-thread cells at 0.90 against `-rtr` — is met: **0.925 and 0.957**.

**Where each cell below 1.0 is held.**

- **512 tokens, 12 threads (0.925, 0.957; from 0.823, 0.816)**: the caller's wait (8.9 % of the pass,
  halved) and attention (8.4 %). The wait left is 151 µs over 609 joins; 128 of those joins are the
  form's fill, and folding that fill into the producer joins (norm, SwiGLU, the output quantise) is
  the next lever, worth ~6 % if it keeps the fill's gain and drops its joins.
- **512 tokens, one thread (0.946, 0.957) and 4 096 tokens at one thread (0.895, 0.883)**: the
  kernel's rate (81 % and 55 % of those passes) and attention — scheduling cannot reach either, and
  one thread is level with before (the 1 024-token pair: +2.5 % / −0.7 %).
- **4 096 tokens, 12 threads (0.787, 0.864)**: attention, 45 %/41 % of the pass against the fork's
  ~20 % — the fork reads its f16 halves directly, our fast order widens every key tile to f32. The
  int8 cache does not close it (Part A) and is not the missing piece.
- The 1B 4 096/12 ratio reads lower than S5.4j's 0.890 because **the fork's cell moved in this
  window** (452.2 against 380.2 then) while ours moved 5 %; the same-window before → after is
  0.720 → 0.787, and the raw rounds show the fork's 4 096/12 cell spread 370.9-452.2 across this
  session's windows.

## 5. Bits

- `git diff --name-status origin/main...HEAD -- '*.golden'` prints nothing: **no golden moved**.
- The changes are scheduling: the grain only decides which worker takes a row range (a row's value
  is a function of its activation and weight row), the fill writes every group's own bytes (pinned
  against the serial fill), and the KV writer encodes through the same function `put` does (pinned
  on all three precisions).
- The model suite passes natively, under `PEREGRINE_ISA=avx2` and `scalar`, and with
  `PEREGRINE_ADAPTIVE_STALL=claim` (the claimed partitions cover the fill and the write, because
  they go through the same `run`/`run_claimed` drivers).
- The exact engine is the same engine, only scheduled differently: 512/12 +1.6 %, 512/1 level, and
  every exact golden unchanged.

## 6. What surprised this round

1. **The int8 claim was an inference, and it was wrong.** The fork's `-rtr` cells are f16-keyed
   f32 × f16 kernels; the int8 path is one flag away (`-ctk q8_0`) and is worth 20 % to the *fork*,
   but not to the cells we compare against. Reading the dispatch rather than the timings changed
   the plan in one paragraph.
2. **Our int8 cache is 11-14 % *slower* at 4 096 and 7-15 % slower at 512.** An int8 cache without
   an integer dot is a dequantise on the critical path; the accuracy is nearly free, the time is
   not.
3. **The form's fill was 6.4 %, not 1.5 %.** The estimate counted 24 MiB; the pass interleaves
   112 MiB because the `down` group's input is 32 super-blocks wide. It is now the second largest
   term removed.
4. **A finer dynamic grain is worth 6 % and costs 11 % of decode.** The same knob turned the other
   way is a different workload: one activation row, 224 joins a token, no tail — hence the
   `nb == 1` exception.
5. **The caller's wait was a tail, not (only) parking.** 287 µs a join with a 62 µs straggler gap
   says the caller is waiting for a chunk that was claimed just before it, not for a worker that
   went to sleep.

## 7. What was not done

- **Folding the fill into the producer joins.** It keeps the 7.5 % and drops the 128 joins it added
  (and the ~1 % those joins' waits cost); it needs the act-producing step (norm, SwiGLU, the output
  quantise) to know the next group's form shape, which is a signature change on a shared path.
- **The residual into the matmul's store** (2 joins a layer): an accumulating output pointer in
  every store path of `blocks/linear.rs`, which another worker was rewriting in the same window.
- **A quantising attention kernel.** The int8 cache is not one (A4); the fork's 20 % is the size of
  the prize at 4 096/12 and it needs a Q8 key/query report of its own.
- **An f16-direct fast score body.** The other candidate for the 4 096 gap (read the f16 halves in
  the fast order instead of widening the tile): the fast order's definition would have to say
  which representation each score reads, and the bits move, so it needs its own definition and
  golden.
- **ik at its defaults for the two 4 096-token one-thread cells and llama.cpp for Qwen3 1.7B at
  4 096/1** — S5.4i/S5.4j left the same cells empty and they cost 6.5-11 minutes an arm.

## 8. Exact commands

```sh
# Part A: the six-arm grid (promptbench --kv, kld --ref-kv and ik's llama-bench)
./scripts/sweep-a2.sh 5 raw/a2-512-12.jsonl "$L|512|12" "$Q|512|12"
./scripts/sweep-a2.sh 5 raw/a2-4096-12-1b.jsonl "$L|4096|12"
./scripts/sweep-a2.sh 1 raw/a2-4096-1-q3.jsonl "$Q|4096|1"      # 3 rounds, in halves

# Part A3: the cache against the exact engine on f16
/tmp/kld-s54o --base /var/tmp/pg/s54c/base/llama32-1b.base.bin --model "$L" --threads 12 \
    --kv i8 --ref-kv f16 --ref-self --fast-prompt

# the int8 KV on the fork's own path (A1): 369.9 -> 445.3 tok/s at 4096/12
$IKB -m "$L" -p 4096 -n 0 -t 12 -r 1 -o json -rtr -ctk q8_0 -ctv q8_0

# Part B1: the account on the merged build and on this one
PEREGRINE_FAST_PROMPT=1 PEREGRINE_PROF2=full PEREGRINE_ADAPTIVE_LOG=1 \
    /tmp/pb-s54o-base "$L" ~/.cache/peregrine-bench/s52b/corpus.txt 512 12
PEREGRINE_FAST_PROMPT=1 PEREGRINE_PROF2=full PEREGRINE_ADAPTIVE_LOG=1 \
    /tmp/pb-s54o-final "$L" ~/.cache/peregrine-bench/s52b/corpus.txt 512 12

# Part B: one lever per A/B, alternating the two builds inside each round
./scripts/ab.sh 5 raw/ab-chunk32.jsonl 0-23 /tmp/pg-s54o-base /tmp/pg-s54o-chunk32 "$L|512|12"
./scripts/ab.sh 5 raw/ab-fill.jsonl    0-23 /tmp/pg-s54o-chunk32 /tmp/pg-s54o-fill    "$L|512|12"

# Part B3: the eight cells, six arms
A=exact,fast-before,fast,ik,ik-rtr,llama
./scripts/sweep.sh 5 raw/sweep-512.jsonl "$L|512|12|$A" "$L|512|1|$A" "$Q|512|12|$A" "$Q|512|1|$A"
./scripts/sweep.sh 3 raw/sweep-4096.jsonl "$L|4096|12|$A" "$Q|4096|12|$A"
```

## 9. The files

| file | what |
|---|---|
| `README.md` | this report |
| `scripts/sweep-a2.sh` | one round of Part A's six arms |
| `scripts/sweep.sh` | one round of the eight cells (six arms; S5.4j's driver with this step's builds) |
| `scripts/ab.sh`, `scripts/abenv.sh` | the alternating A/B of two builds, and of two environments on one build |
| `scripts/table.py` | the tables from the raw files, every ratio recomputed |
| `raw/a2-*.jsonl` | Part A's grid (prompt, decode, attention bucket, load) |
| `raw/kld-*.txt` | the accuracy runs |
| `raw/sweep-*.jsonl` | the eight cells |
| `raw/ab-*.jsonl` | each lever's A/B |
| `raw/b1*-*.txt` | the pass accounts on the merged build and on this one |
| `raw/ten-user.txt` | the ten-user cell, before and after |
