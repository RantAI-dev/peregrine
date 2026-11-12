# S5.2c: the f32-tile pair mix, the pass's serial remainder, and the mix against its roof

Raw measurements for `docs/design/s5-prefill-avx512.md` §13, 2026-11-12 (15:20–18:15 local,
UTC+7) on the desktop: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads, AVX-512
F/BW/DQ/VL/VNNI/VBMI/BF16), WSL2, 11.7 GiB.

Peregrine: branch `s5.2c-prefill-serial` = `origin/main` `e4ad034` (S5.2b) plus this
round's three commits; `cargo build --release -p peregrine-bench --example promptbench
--features "arch-llama arch-qwen"` into `~/.cache/peregrine-bench/target-s52c`. Two builds
per part, as the round requires: `pb-before` (= `e4ad034` exactly) and the branch's own
binary, alternating **inside** each round. `pb-after-A` is part A alone; `pb-final` is parts
A and B.

Models: the same two files as every earlier desktop round, by their regular paths — Llama
3.2 1B Instruct Q4_K_M (807,694,464 B) and Qwen 2.5 0.5B Instruct Q4_K_M (491,400,032 B)
under `~/models-ab/` (the `~/.cache/peregrine-bench/models/` entries are symlinks, so they
were not opened). Corpus: the S5.2 prose recipe, `~/.cache/peregrine-bench/s52c/corpus.txt`
(the concatenation in the S5.2b README, this repository's own documentation only).

**The host was shared and heavily loaded throughout.** The other worker's S5.4c accuracy
report ran on the same 24 logical CPUs for the whole window; the one-minute load average is
printed beside every cell and ran **3–23** (mostly 10–20, i.e. one to two other busy threads
per logical CPU). Two consequences, both applied everywhere below:

- every before/after is two builds alternating **within** a round, five rounds (two or three
  for the one-thread 4096-token cells, which cost 1.5–2.5 minutes a run), median and full
  range printed; a cell whose ranges overlap is reported as **no change**, not as a gain;
- absolute tok/s are far below the S5.2b window (Llama 1B 512 tokens at 12 threads: 240–300
  here against 304–336 there). **Every ratio is what this round reports**; the absolute
  numbers carry the host.

This machine's CLOCK_MONOTONIC runs ~10 % fast (S5.2b §"The clock"): every tok/s is ~10 %
conservative in real seconds, and a ratio between two builds timed with the same clock is
untouched. Nothing here is timed with CPU time, so no roof number needs correcting.

Drivers in this directory:

| file | what |
|---|---|
| `ab.py` | the before/after A/B: two builds, arms alternating within a round, `REVERSE=1` flips the order |
| `cfgab.py` | two *environments* of one build, interleaved (the chunk probe) |
| `engines.py` | Peregrine / ik_llama.cpp / llama.cpp, interleaved per round |
| `account.py` | the phase account of a pass, from `promptbench`'s own slots |
| `ab-*.jsonl`, `acct-*.jsonl`, `engines-*.jsonl` | the raw cells |

Everything ran in the foreground; no background job was left. Scratch (binaries, corpus,
probes) is under `~/.cache/peregrine-bench/s52c/`.

---

## Part A: the widened f32 tile's two-row value mix

**What was built.** `mix_pair_same` (`attn.rs`) is `mix_f16_pair_same` for the tile below
`DIRECT_MIN_VIS` — the one a 512-token prompt takes: two rows of one KV head that see the
same positions read each f32 value once and feed both rows' fused multiply-adds, each row
keeping its own four accumulators and its own ascending position order. A sixteen-lane body
for `head_dim % 16 == 0`, an eight-lane body for `% 8` (four lanes per row in flight), and
two solo mixes otherwise. The prefill tile's mix loop pairs a row with the next one when both
see the same position range and the same canonical chunks — the `gqa` query heads of one
batch row always do — and cuts the pass at the same chunk boundaries as before.

**The bits.** `the_prefill_f32_pair_mix_is_each_rows_own_mix` checks the body against `mix`
row by row, bit for bit, on the same grid as the F16 body's test: every visible length 1…70,
127/128/129, 511/512/513 and 4095/4096; head dims 64 and 128 (sixteen lanes), 80 and 72
(eight lanes) and 16; on random scores and on the edge values a softmax produces (all-zero
alphas, alpha 1, a whole row of equal weights, negative zero, a large alpha). The prefill
tile's own invariance tests — `prefill_tiles_are_bit_identical_to_a_solo_decode`,
`a_windowed_rows_bits_are_identical_alone_in_a_chunk_and_at_any_pool_width`,
`the_direct_prefill_threshold_does_not_move_a_bit` — pass unchanged.

**The measurement.** The pass at 512 tokens, both models, five rounds; and the profile's own
`attn mix` bucket (the mix **and** the softmax it is timed with) at one thread, where the
bucket is 4–6 % of the pass:

| model | prompt | threads | before | after | ratio (median) | per-round ratios | n |
|---|---|---:|---:|---:|---|---|---:|
| Llama 1B | 512 | 1 | 35.2 | **36.2** | **1.033** | 1.013, 1.021, 1.033, 1.065, 1.068 | 5 |
| Qwen 0.5B | 512 | 1 | 72.7 | **74.6** | **1.027** | 1.000, 1.008, 1.027, 1.130, 1.203 | 5 |
| Llama 1B | 512 | 6 | 171.9 | 177.9 | 1.057 | 0.999, 1.035, 1.057, 1.089, 1.220 | 5 |
| Llama 1B | 512 | 12 | 244.6 | 262.7 | 1.075 | 0.957, 1.056, 1.075, 1.076, 1.132 | 5 |
| Qwen 0.5B | 512 | 6 | 346.9 | 379.5 | 1.089 | 0.874, 1.015, 1.089, 1.139, 1.163 | 5 |
| Qwen 0.5B | 512 | 12 | 522.9 | 470.4 | 0.955 | 0.899, 0.946, 0.955, 1.001, 1.040 | 5 |

`REVERSE=1` (the after arm first) reads the same cells at 0.995, 1.011, 1.031, 1.029 — so at
6 and 12 threads the two orders **straddle 1.0 and the pass-level result is "no change"**, as
§12.2 predicted for a 4–6 % bucket. At **one thread** the arm order cannot hide it: every
round of both models is ≥ 1.000, the median is +2.7 % (Qwen) and +3.3 % (Llama), and the
bucket itself says where it comes from:

| model | `attn mix` per round, before → after (ns, one thread) | ratio |
|---|---|---:|
| Llama 1B | 622→476, 599→470, 598→484, 599→464, 610→462 | **0.784** (1.28×) |
| Qwen 0.5B | 408→354, 419→357, 428→359, 382→341, 411→335 | **0.861** (1.16×) |

The bucket holds the softmax too (2.2 ns a score, part C), so the *body's* share of it falls
by more than a quarter: the pair body is real, and the pass at 12 threads cannot resolve it.

---

## Part B: the serial remainder, named

### B.1 The timers

The phase slots already exist per call (`norms`, `quantise`, `rope+bias`, `kv write`,
`residual`, `barrier wait`) and are taken around whole passes, never per row. One step had
no timer: **SwiGLU**. `swiglu_rows` in the dense forward is now wrapped in `Slot::Swiglu`
(the decode path's fused variant already was), so the account has no unmeasured step above
0.5 %.

### B.2 The account

`PEREGRINE_PROF2=1`, medians of 2–3 runs per cell, shares of the engine's `prefill_s`; the
load average beside each row was 10–20. `rest` is the sum of the six steps. `joins` is the
fork–join count of the pass (`StepStats::barriers`; 12 per layer per prefill chunk).

| model | prompt | threads | pass ms | tok/s | rest % | norms | quantise | rope+bias | kv write | residual | swiglu | matmul % | attention % | wait % | joins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 1B | 512 | 1 | 12 672 | 40 | 5.8 | 0.25 | 1.89 | 0.22 | 0.23 | 0.10 | 3.10 | 88.3 | 6.3 | 0.0 | 577 |
| | 512 | 6 | 2 598 | 197 | 7.7 | 0.32 | 1.97 | 0.36 | 1.13 | 0.51 | 3.39 | 84.1 | 7.6 | 9.7 | 1 633 |
| | 512 | 12 | 1 994 | 256 | 14.5 | 0.58 | 3.26 | 1.52 | 1.84 | 2.60 | 4.68 | 73.9 | 7.7 | 20.9 | 1 633 |
| | 4096 | 1 | 141 969 | 29 | 4.2 | 0.16 | 1.37 | 0.17 | 0.20 | 0.06 | 2.20 | 64.0 | 32.5 | 0.0 | 4 161 |
| | 4096 | 6 | 29 221 | 140 | 6.2 | 0.15 | 2.08 | 0.30 | 0.92 | 0.42 | 2.30 | 58.5 | 34.5 | 18.4 | 12 385 |
| | 4096 | 12 | 16 967 | 241 | 8.5 | 0.20 | 3.51 | 0.37 | 1.39 | 0.65 | 2.40 | 59.9 | 30.7 | 12.6 | 12 385 |
| Qwen 0.5B | 512 | 1 | 5 449 | 94 | 8.4 | 0.28 | 1.79 | 0.31 | 0.35 | 0.13 | 5.52 | 81.8 | 9.1 | 0.0 | 865 |
| | 512 | 6 | 1 399 | 367 | 11.2 | 0.57 | 1.88 | 0.75 | 0.96 | 0.87 | 6.13 | 77.4 | 9.8 | 11.8 | 2 449 |
| | 512 | 12 | 1 037 | 495 | 15.2 | 1.14 | 2.55 | 1.61 | 1.27 | 2.78 | 5.88 | 72.1 | 8.6 | 21.4 | 2 449 |
| | 4096 | 1 | 80 926 | 51 | 5.4 | 0.19 | 1.11 | 0.22 | 0.18 | 0.09 | 3.57 | 52.4 | 42.1 | 0.0 | 6 241 |
| | 4096 | 6 | 17 975 | 228 | 8.7 | 0.19 | 2.52 | 0.48 | 0.79 | 0.79 | 3.97 | 50.6 | 39.8 | 13.4 | 18 577 |
| | 4096 | 12 | 14 427 | 286 | 18.1 | 0.20 | 6.25 | 1.70 | 0.91 | 4.41 | 4.64 | 46.9 | 30.9 | 31.3 | 18 577 |

`PEREGRINE_PROF2=full`, twelve threads, the same cells (medians of two), splits the attention
bucket. The worker-side sub-timers are summed over the pool and divided by its width:

| model | prompt | attention | scores | mix (+ softmax) | kv decode | the caller's own fork + prologue |
|---|---|---:|---:|---:|---:|---:|
| Llama 1B | 512 | 7.4 % | 1.4 % | 2.2 % | 0.2 % | **3.5 %** |
| | 4096 | 28.7 % | 9.0 % | 12.9 % | 0.1 % | **6.8 %** |
| Qwen 0.5B | 512 | 8.7 % | 1.5 % | 2.4 % | 0.2 % | **4.6 %** |
| | 4096 | 32.8 % | 9.4 % | 14.8 % | 0.1 % | **8.4 %** |

**So the "6–10 % of the caller's serial work" is not serial work.** Of the six steps, five
run on the pool — `norm_rows`, `prepare_rows`, the attention output's quantise, the two
residual adds and `swiglu_rows` all fork — and their wall is what a fork–join costs on this
host (below). Only the **KV write** is serial (`write_kv_rows` takes no pool), and at twelve
threads it is 0.9–1.8 % of a 4096 pass and 1.0–1.8 % of a 512 one. The other 5–16 % is
parallel work paying one join each, plus the caller's own share. The same table from the
quiet window of S5.2b read `rest` 6.3–9.3 %; this window reads 6.2–18.1 %, and the growth is
entirely in the join-bound steps (`quantise`, `residual`), not in their work.

### B.3 Each step over 0.5 % of a pass, and its verdict

| step | 512/12 t | 4096/12 t | can it run on the pool? | fused? skipped? | verdict |
|---|---:|---:|---|---|---|
| `quantise` (attention out → the wo matmul's activation) | 2.6–3.3 % | 3.5–6.3 % | already does (`for_row_chunks_2`) | the fork's own cost; its work is 134 M row-elements a pass at 4096 (14.5 ns each on one thread, 4.5–8 ns at twelve) | no change: the join, not the work, is the cost (B.4) |
| `swiglu` | 4.7–5.9 % | 2.4–4.6 % | already does (prefill branch) | — | no change: 537 M `silu`s a pass at 4096, 7.6× on twelve threads; it is at its floor |
| `residual` | 2.6–2.8 % | 0.7–4.4 % | already does | could be fused into the following norm (one fork instead of two, and one less read/write of `x`) | **dropped**: the estimated saving is 2 joins a layer of 12, ≈ 2 % of the pass, and it needs a pending-add parameter through `attention_rows`, the dense loop and the head in every family; the change is not worth its bit-exactness surface. Written down, not built |
| `rope+bias` | 1.5–1.6 % | 0.4–1.7 % | already does | — | no change: the phase is *cheaper* than the fork that runs it (below) |
| `kv write` | 1.3–1.8 % | 0.9–1.4 % | **yes — it is the only serial step** | — | **dropped**: the write is 65 k f32→f16 conversions a chunk-layer, 20 µs over twelve threads; the fork that would run it costs ~200 µs here (B.4). Threading it cannot pay. Fusing it into `prepare_rows` (the pass immediately before it, which already walks the same rows) would need `&mut Caches` inside a work-splitting closure: two threads would hold `&mut` to one `Shared` cache. For 1 % of a pass that is not a trade this round makes |
| `norms` | 0.6–1.1 % | 0.2 % | already does | — | no change |

### B.4 The barrier wait: what the workers wait for

`Slot::Wait` is the caller's wait *after its own share* of a fork–join, and `barriers` counts
the joins. Twelve threads, this window:

| model | prompt | joins | wait | per join | parks of a worker, this pass | park length | adaptive level |
|---|---:|---:|---:|---:|---:|---|---|
| Llama 1B | 512 | 1 633 | 20.9 % | 0.26 ms | (not logged; the 4096 pass is) | | 0 |
| | 4096 | 12 385 | 12.6 % | 0.17 ms | 6.3 k | 1–5 ms: 6 347; 0.5–1 ms: 2 182; <0.5 ms: 1 361; ≥5 ms: 247 | 0 |
| Qwen 0.5B | 4096 | 18 577 | 31.3 % | 0.24 ms | — (same shape) | | 0 |

`PEREGRINE_ADAPTIVE_LOG=1` on the Llama 4096 pass printed `ladder wall per thread: level0
239 767 ms (23 399 samples), level1 0, level2 0, transitions 0` and the park buckets above.
So:

- **not the spin window.** The ladder never left level 0 (the 1000 µs spin), and S5.2b swept
  `PEREGRINE_SPIN_US` 0/200/1000/4000: parking costs 56–79 %, 4000 µs costs 2–13 %.
- **not uneven work per worker in the attention split.** `plan()` makes 24 tasks
  (`n_kv_head × ceil(2·workers / n_kv_head)`) and `pool.run` splits them over the *pool
  width*, 12 — two tasks each, the caller taking share 0 like any worker. Every task is one
  KV head × a block of whole rows of one chunk, and all rows of a chunk see the same
  ±64-position context, so the tasks are equal to ~1.5 %.
- **not one serial step between two joins** in the sense of a caller loop: the gaps a worker
  sees are the caller's own share of the *previous* job plus the next job's setup
  (microseconds), except that the caller's share ends last by construction (it takes its
  share before waiting).
- **it is the OS.** A worker that finishes its share, spins 1 ms and finds no job **parks**
  (measured: 6–13 k parks a pass, most of them 1–5 ms long). The next job then waits for the
  futex wake-up *and* for the scheduler to put that worker back on a logical CPU. With
  another tenant's 10–20 threads on the same 24 logical CPUs, that is milliseconds. The
  joins are frequent (12 a layer a chunk: 12 385 at 4096) precisely because a 64-row pass is
  read in 64-token pieces, so the pass pays the wake-up ~12 000 times.

**What that makes a scheduling choice, and what was measured.** The only lever the code has
is *fewer joins per pass*, and the pass width is that lever. Chunk 64 against chunk 128 at
4096 tokens, twelve threads, the same build, two rounds (host load 10–19):

| model | round 0 | round 1 |
|---|---:|---:|
| Llama 1B | 208.1 → 212.2 (1.020) | 204.0 → 210.2 (1.030) |
| Qwen 0.5B | 320.9 → 331.9 (1.034) | 325.1 → 332.7 (1.024) |

That is +2–3.4 % in four cells of four, and it is the one scheduling change the round found
that pays. It is **not shipped**: a 128-row pass is twice the prompt-row scratch of every
engine (`WIDE_CHUNK_DEFAULT` is 64 because the scratch already holds 64 speculative rows),
S5.2b measured chunk 128 *worse* at 512 tokens (−3 to −5 %), and a long prompt read beside a
decoding user would double that user's wait. A default chosen from a two-round probe on this
host is what §12 refused to do; it is written down as the open item instead.

---

## Part C: the mix body against its roof

### C.1 The present body, operation for operation

`mix_avx512` (`head_dim 64`, the f32 tile): one 64-dimension block = **four `zmm`
accumulators** (16 lanes each). Per visible position `j`:

```
broadcast  p_j                1 × vbroadcastss
load       v_j[16d+0..16]     4 × vmovups (64 B each, 256 B a position)
fma        acc[d] += p_j·v    4 × vfmadd231ps (64 MACs)
```

Per output element and position: **1 load + 1 fused multiply-add**; per 16 elements and
position, one broadcast. The stores happen once per row, outside the position loop. The F16
body is the same plus a `vcvtph2ps` per 32 halves (one per accumulator per position), and the
pair bodies double the FMAs and halve the loads (and the conversions).

**The dependency chain.** Each output element's additions are `acc = α·o; acc = fma(p1,v1,acc);
acc = fma(p2,v2,acc); …` in ascending `j` — the defined order. A chain of `vis` FMAs, one per
position, 4-cycle latency on Zen 5. Chains in flight per row at `head_dim 64`: **four**
(one per 16-dimension block). Therefore the *one-row* body can issue one FMA a cycle at best
— **16 MAC/cycle = 29.6 GMAC/s** at 1.85 GHz, exactly half the machine's 32 MAC/cycle
(two `vfmadd512` a cycle, the 59.0 GMAC/s roof S5.2b measured). At `head_dim 128` a row has
**eight** chains (two blocks of four), so its chain bound is the roof itself.

**Two rows** doubles the chains (8 at head_dim 64) **without touching any element's order**,
so the chain bound moves to the roof: 32 MAC/cycle. This is why the pair body is the right
shape and a fourth row is not — 16 chains at head_dim 64 are more than two/FMA-a-cycle can
use, they only add latency tolerance, and S5.2b measured four rows at the same total as two
(27.6 against 27.8 GMAC/s).

### C.2 What the bodies actually reach (L1-resident, one core, same loops)

`mixsoft.rs` / `mixbench.rs` in the scratch directory, `rustc -O -C target-cpu=native`,
pinned to CPU 4 (which the other tenant shared, so these are 15–30 % below a quiet core);
`vis = 64`, 200 000 repetitions, the exact loops of `attn.rs`:

| body | head_dim 64 | head_dim 128 |
|---|---:|---:|
| f32, 1 row (4 chains) | 18.8 GMAC/s (32 %) | 14.0–15.9 (24–27 %) |
| f32, 2 rows (8 chains) | 21.2 (36 %) | 19.7–19.9 (33 %) |
| f32, 4 rows (16 chains) | 25.4 (43 %) | 24.3 (41 %) |
| f16, 1 row | 11.8 (20 %) | 15.2 (26 %) |
| f16, 2 rows | 15.0 (25 %) | 23.8 (40 %) |
| f16, 4 rows | 20.2 (34 %) | 22.7 (38 %) |
| f32, 1 row, **two accumulators per element** (a different order, priced only) | 14.7 | 14.2 |
| the softmax alone (`max` + `exp_shift_sum` + `det_exp`) | **2.18 ns a score** | 2.28 |

**The softmax is 39 % of the bucket the design doc calls "the mix".** At `head_dim 64` a
64-position tile costs 218 ns of mix and 139 ns of softmax; a pair of rows costs 386 ns of
mix and 278 ns of softmax. The same bench with the softmax in the loop reads **11.9 GMAC/s**
for the solo body and **12.6** for the pair — against the 9.3 GMAC/s §12 reports in situ. So
the in-situ "a sixth of the roof" decomposes as: the body at **18–21 GMAC/s (32–36 % of the
roof)**, the softmax at 39 % of the bucket, and the last third the tile stream and the pass's
own cuts (a 4096-token row's mix is cut at its five canonical chunk boundaries).

The "two accumulators per element" line prices the **defined order** directly: splitting each
element's chain in half (a different summation, so never shippable) buys *nothing* (14.7
against 18.8 solo). The chain is not what binds; the loads and the conversions are, and those
the pair body already halves.

### C.3 The verdict

The arithmetic says the same-bits ceiling at `head_dim 64` is the roof with **two** rows and
no more, and the measured body is at 32–36 % of the roof solo and 36 % with the pair, with
the remaining gap in the load and conversion stream and the tile read — not in the ascending
order. The one same-bits change the arithmetic still allowed (two rows) is what S5.2b shipped
for the F16 tile and part A ships for the f32 one. **There is no further 10 % of mix time
available from a body with the same order, so part C is this account: no code.**

---

## The table

Medians of five interleaved rounds (two or three for the one-thread 4096 cells), `pb-before`
= `e4ad034` against `pb-final` = this branch, both models, both lengths, 1/6/12 threads. The
ranges are in §A/§B; a cell whose range straddles 1.0 is a dash.

| model | prompt | threads | before | after | ratio | range |
|---|---|---:|---:|---:|---|---|
| Llama 1B | 512 | 1 | 35.2 | **36.2** | **1.033** | 1.013–1.068 |
| | 512 | 6 | 171.9 | 177.9 | — | 0.999–1.220 |
| | 512 | 12 | 244.6 | 262.7 | — | 0.957–1.132 |
| | 4096 | 1 | 33.1 | 33.5 | — | 0.966–1.063 |
| | 4096 | 6 | 127.5 | 130.6 | — | 0.947–1.024 |
| | 4096 | 12 | 195.4 | 195.0 | — | 0.830–1.046 |
| Qwen 0.5B | 512 | 1 | 72.7 | **74.6** | **1.027** | 1.000–1.203 |
| | 512 | 6 | 346.9 | 379.5 | — | 0.874–1.163 |
| | 512 | 12 | 522.9 | 470.4 | — | 0.899–1.040 |
| | 4096 | 1 | 52.0 | 50.4 | — | 0.969–1.224 |
| | 4096 | 6 | 222.0 | 223.6 | — | 0.983–1.030 |
| | 4096 | 12 | 299.8 | 297.1 | — | 0.901–1.154 |

Nothing else moved (five rounds, medians):

| cell (12 threads) | before | after | ratio |
|---|---:|---:|---:|
| Llama 1B decode tok/s, 512 ctx | 41.7 | 39.7 | 0.93 (spread 2.3×, 36–58 after) |
| Llama 1B decode tok/s, 4096 ctx | 37.3 | 32.7 | 0.88 (spread 1.6×) |
| Qwen 0.5B decode tok/s, 512 ctx | 69.6 | 54.3 | 0.78 (spread 2.4×, 35–84 before) |
| Qwen 0.5B decode tok/s, 4096 ctx | 51.1 | 39.4 | 0.77 (spread 2.8×) |
| Llama 1B, 10 users × 512 prompt + 32 tokens | 249.1 | 249.0 | 1.055 (0.813–1.202) |
| Qwen 0.5B, 10 users × 512 prompt + 32 tokens | 384.9 | 431.1 | 1.120 (0.759–1.543) |

The decode cells are **not measurable in this window**: the same arm varies 2.3–2.8× between
rounds, the ratios straddle 1 in both directions, and decode takes no new code at all — the
pair body is inside `run_range`'s prefill tile, and a decode step never reaches it. The
ten-user cells are ±40 % for the same reason (S5.2b's window already showed ±10 %).

### Against the other engines, twelve threads

`engines.py`, interleaved per round with `-r 1` per `llama-bench` cell; ik_llama.cpp
`479074c` at its `-rtr 1 -ub 512`, llama.cpp `v0.9.0` at its defaults; three rounds at 512,
two at 4096 (load 16–22 throughout, which the `±` in the raw files shows):

| model | prompt | Peregrine | ik `-ub 512` | llama.cpp | P ÷ ik | P ÷ llama |
|---|---:|---:|---:|---:|---:|---:|
| Llama 1B | 512 | 238.1 | 397.0 | 238.9 | **0.60** | 1.00 |
| | 4096 | 204.1 | 385.3 | 185.0 | **0.53** | 1.10 |
| Qwen 0.5B | 512 | 332.9 | 896.6 | 372.0 | **0.37** | **0.62** |
| | 4096 | 361.5 | 642.3 | 252.6 | **0.56** | 1.43 |

Every cell below 1.0 is named: **Peregrine is below ik_llama.cpp in all four cells** (0.37–0.60)
and **below llama.cpp in one (Qwen 512, 0.62)**; at 4096 it is above llama.cpp on both. The
spread on this host is large (ik's Qwen 512 cells read 765/955/897, Peregrine's 333/292/480),
so these are the medians of a noisy window, not a settled ranking; §12.4's account of *why*
ik leads (its 512-row requantising kernel at 91 GMAC/s against Peregrine's exact 72–75, and
512-bit kernels throughout) is unchanged by this round.

## Files

| file | what |
|---|---|
| `ab.py`, `cfgab.py`, `engines.py`, `account.py` | the four drivers in this README |
| `ab-512-t1.jsonl`, `ab-512-t6-12.jsonl`, `ab-512-t6-12-rev.jsonl` | part A, both models |
| `ab-4096-*.jsonl` | the 4096 cells (no change expected) |
| `ab-decode.jsonl`, `ab-users.jsonl` | decode and ten users |
| `probe-chunk.jsonl`, `engines-*.jsonl`, `acct-*.jsonl` | the chunk probe, the three engines, the accounts |

The binaries, the corpus, `mixbench.rs`/`mixsoft.rs` and the peak probes are under
`~/.cache/peregrine-bench/s52c/`; the corpus is a concatenation of this repository's own
files (S5.2b's command), reproducible from it.
