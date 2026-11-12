# The long-context attention tile's loop shapes (S4.9t, ARM)

S4.9s left the Qwen3-Embedding F16 2 000-token cell at 0.96× llama.cpp on a quiet host, with
attention 35 % of the pass and two stages at a third of this core's f32 roof: the score kernel
at 29.0 GMAC/s and the value mix at 21.3 (the mix bucket, soft-max included). This round is the
account of where attention's time goes stage by stage, the same-bits bound of each stage that
is limited by its loop shape, the three changes that closed the score, mix and soft-max parts,
and the re-measured rows. The one-thread A/Bs are here in full; the ten-thread engine table
against llama.cpp is **owed**, because this host never gave a cell under load 4.

- Date: 2026-11-12, 19:30–21:15 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters: 10 ×
  Cortex-X925 at 3.9 GHz (CPUs 5–9 and 15–19) and 10 × Cortex-A725 at 2.8 GHz, 121 GB unified
  memory, Linux `6.17.0-1008-nvidia`, governor `performance`, rustc 1.99.1, kernel tier
  `aarch64-neon-i8mm`.
- Peregrine: `origin/s4.9s-qwen3-embed-f16-long` (`cab27cd`, PR #252's head, unmerged) plus this
  round's three commits in order: the four-row score tile, the four-row value mix, the four-row
  soft-max. The baseline arm is a worktree of `cab27cd` with its own target directory.
- **The host was not quiet in this window.** The 1-minute load average ran 2.4–32 (a co-tenant,
  plus the load the round's own runs see on a shared box); it is written beside every cell, and
  every cell here is marked **loaded** by the rule that a cell above load 4 is not a standing.
  The one-thread cells are nonetheless stable to ±1 % round to round, which is why the A/Bs
  below read and the engine table does not exist yet.
- Models: `qwen3-embed-0.6b-f16.gguf` (1 197 629 632 B), `qwen3-embed-0.6b-q8_0.gguf`
  (639 150 592 B), `gte-modernbert-base-F16.gguf` (299 876 992 B), `Llama-3.2-1B-Instruct-Q4_K_M`
  (807 694 464 B). Text: `/var/tmp/embed-texts/qwen3-embed-0.6b.json`; the 2 000-token cell
  counts 2 000 tokens (528 at "512").
- `perf` is not usable (`perf_event_paranoid` is 4), so the engine's own `PERGEGRINE_PROF2=full`
  counters are the profile and `stagebench.rs` here is the kernel-level bench.

## 1. The account

One thread, `cellprof` + `PERGEGRINE_PROF2=full`, medians of three runs per arm, before (the
`cab27cd` arm) and after (the branch). `mix bucket` is what the engine's timer calls the mix: the
value mix **plus** the soft-max's maximum and exponentials, which run between its pieces. The
kernel-level split of that bucket comes from `stagebench.rs` (below), which puts the soft-max at
0.878 ns per score element: 0.787 s at 2 000 tokens and 0.055 s at 528.

**Qwen3-Embedding 0.6B, F16, 1 thread** (28 blocks, d 1 024, ff 3 072, 16 query / 8 KV heads of
128; the embedding path is the causal prefill):

| cell | wall | matmul | attention | scores | mix bucket | of which soft-max | of which mix | caller |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 528 tok, before | 6.862 s | 6.012 s | 0.681 s | 0.272 s | 0.386 s | 0.055 s | 0.331 s | 0.023 s |
| 528 tok, after | 6.828 s | 6.025 s | **0.634 s** | **0.223 s** | 0.387 s | 0.055 s | 0.332 s | 0.024 s |
| 2 000 tok, before | 32.80 s | 22.79 s | 9.356 s | 3.897 s | 5.353 s | 0.787 s | 4.566 s | 0.106 s |
| 2 000 tok, after | 32.10 s | 22.85 s | **8.595 s** | **3.122 s** | 5.360 s | 0.787 s | 4.573 s | 0.113 s |

Arithmetic is the S4.9s count: per layer 440.4 MMAC/token of matrix product, and causal
attention 16 heads × 128 dims × n(n+1)/2 scores and the same again in the mix — 8.0 + 8.0 GMAC at
528 tokens and 114.7 + 114.7 GMAC at 2 000. Stage rates against this core's measured roof
(`fpbench`: packed F16 47.0–47.6 GMAC/s, f32 62.6–65.8, so a f32 FMA roof near 66):

| stage | GMAC/s at 528 | GMAC/s at 2 000 | share of the pass at 2 000 |
|---|---:|---:|---:|
| matmul (F16) | 37.6 | 37.4 | 71 % |
| scores, before | 29.4 | 29.4 | 11.9 % |
| scores, after | **35.9** | **36.7** | **9.7 %** |
| mix, before | 24.2 | 25.1 | 13.9 % |
| mix, after | 24.1 | 25.1 | 13.9 % |
| soft-max | — | — | 2.4 % |
| caller (decode_block, banking, combine) | — | — | 0.3 % |

The same account on the **Q8_0** file of the model (same causal path, so the same kernels) and on
**gte-modernbert-base** (a bidirectional encoder — the `bidir` path, which this round does not
touch and which is the guard):

| cell | wall | matmul | attention | scores | mix bucket | attention share |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 Q8_0 528, before | 3.285 s | 2.275 s | 0.685 s | 0.277 s | 0.388 s | 20.9 % |
| Qwen3 Q8_0 528, after | 3.253 s | 2.293 s | **0.637 s** | **0.227 s** | 0.390 s | 19.6 % |
| Qwen3 Q8_0 2 000, before | 19.35 s | 8.620 s | 9.484 s | 3.981 s | 5.403 s | 49.0 % |
| Qwen3 Q8_0 2 000, after | 18.69 s | 8.708 s | **8.736 s** | **3.202 s** | 5.429 s | 46.7 % |
| gte-modernbert 552 (L512), before | 1.549 s | 1.349 s | 0.129 s | — | — | 8.3 % |
| gte-modernbert 552 (L512), after | 1.557 s | 1.349 s | 0.130 s | — | — | 8.3 % |
| gte-modernbert 2 091 (L2000), after | 6.803 s | 5.019 s | 1.416 s | — | — | 20.8 % |

The encoder's attention is not split by the profiler (its stages were accounted in S4.9r §21:
key panel, scores, row max, exp + denominator, mix), and it executes none of this round's code:
the guard is that its cells do not move.

**What the split says.** Three of attention's four stages are running at a third to a half of
what the arithmetic allows, and each is limited by its loop shape, not by its count of products:

- **scores** (29.4 GMAC/s): per key and row there are 128 products as 32 four-lane FMAs, 32
  widenings and 32 loads of four halves, into **two** accumulators per row (dimensions `i..i+4`
  and `i+4..i+8` of each ascending eight-wide block). Two chains is what a row's tree allows: the
  defined order accumulates the block walk into lane-wise sums in ascending order, then reduces
  `((a0+a1)+(a2+a3)) + ((b0+b1)+(b2+b3))`. A four-cycle FMA latency and two chains leave half of
  the four FMA pipes idle; the row-pair body had eight chains (two rows × two keys × two) and
  measured 31.5, i.e. it is the pair that was nearly covered, not the kernel.
- **mix** (25.1 GMAC/s): per key and row there are 32 multiplies **and** 32 adds — the definition
  is `o += p_j · v_j`, rounded product then rounded add, so no FMA may be introduced — plus 32
  widenings and 32 loads of four halves shared by the group. Two ops per four products means the
  ceiling is half the pipes; and with two accumulators per row over an eight-dimension slice, a
  row-pair has four chains. Measured (`stagebench`, hd 128, 64-key pieces, GMAC/s): pair 28.4,
  pair over a 16-dimension slice (8 chains) 33.7, pair over 32 dimensions 34.0, **four rows 54.8**,
  eight rows 53.2, sixteen 48.1. The knee is eight chains; the four-row body reaches 83 % of the
  f32 roof.
- **soft-max** (2.4 % of the pass, 0.878 ns per score element): the exponentials are vector work
  but the denominator is one ascending `f64` sum per row, and a scalar `f64` add waits four
  cycles; a row alone keeps one add in flight. S4.9r measured this on the encoder (0.14 ns per
  element with three of four cycles idle) and fixed it with four interleaved denominators. In
  the decoder it was still one row at a time. Four rows interleaved measure **0.567 ns** per
  element.
- **the caller's own work** is 0.3 % of the pass: tile decoding, the chunk banking and the merge.
  Not a candidate.

## 2. The bound each loop shape allowed

Written before each body, in the terms the kernels are made of (per element: loads, widenings,
multiplies, accumulators in flight), and what the defined ascending order permits at head dim 64
and 128.

**Scores.** The tree is fixed: `sum over ascending eight-wide blocks` per lane, two four-lane
accumulators, then the pairwise reduction. At 128 dims that is 32 FMAs and 32 widenings per
key-row and 16 loads of four halves (two per eight dims) — no order change is available, so the
only free variable is how many independent chains are in flight. Chains = rows × keys × 2 halves.
Four chains (one row) → 1 FMA/cycle at four-cycle latency = 8 GMAC/s; eight (the pair) → 2/cycle
= 16 GMAC/s of FMAs, which is half the pipes; **sixteen (four rows × two keys)** covers all four
pipes and predicts ~62 GMAC/s ceiling on FMA issue alone, minus the widening and load pressure
(4 loads + 4 widenings + 16 FMAs per two keys per eight dims). Measured 40.3 at hd 128 and 35.0 at
hd 64. Eight keys per row needs 32 accumulators and spills (24.2 measured); four rows × four keys
spills harder (25.8).

**Mix.** The definition is two ops per four products and two accumulators per row over an
eight-dimension slice, so chains = rows × 2 and the pipe ceiling is ~66 GMAC/s only if 8 chains
are live. Four chains (the pair) predicts 33 % of the pipes, which is what 28.4 measures. Eight
chains is the knee. Note what the four-row body may *not* do: it may not mix four positions,
because cutting a row anywhere but its own chunk boundary gives its soft-max an extra step and
moves the denominator's `f64` summation — so the group is rows that share their context, and at
head dim 64 (Llama) that is four query heads of one position while at 128 (Qwen3-Embedding, two
query heads per KV head) it is two. A 16-dimension pair slice was measured as an alternative for
the pair: 28.4 → 33.7 alone, **+0.8 % in the pass** — dropped, see §4.

**Soft-max.** One ascending `f64` add per element, four-cycle latency, one row's chain. Chains =
rows. Four rows → 4 adds in flight → the fold stops being the limit; measured 0.878 → 0.567 ns
per element, and the residual is `exp4` itself. The four rows must share their piece, for the
same reason as the mix.

## 3. What was built

One change per commit, in this order, each with its bit-identity test and its own five-round
alternating A/B at one thread.

1. **`scores_f16_tile4`** (`neon.rs`, scalar twin `scores_f16_tile4_scalar`): four query rows
   over one F16 K tile, two keys per pass. `attn.rs` groups four rows that see exactly the same
   part of the tile and falls back to the pair and then the single row. The four-row body is
   aarch64-only; x86 keeps its own pair. **Test**: `the_four_row_score_tile_is_each_rows_own_score`
   — every length 1..=70, 127/128/129, 511/512/513, at a packed and a padded stride, head dims
   64/128/80/72/16, random keys and the values a saturated soft-max leaves (±0, the smallest
   subnormal, ±1, ±65504), against both `scores_f16` and the twin, `to_bits` equal.
2. **`mix_f16_quad`** (`neon.rs`, twin `mix_f16_quad_scalar`): four rows over one F16 V tile,
   eight-dimension slices. `attn.rs` groups four rows of one KV head that share their context and
   chunk count. **Test**: `the_four_row_mix_is_each_rows_own_mix` — same grid, the alphas a
   soft-max produces (0, 1, −0.0, 3.5), against `mix_f16` and the twin.
3. **`softmax_rows`** calls `bidir::exp_rows4_sums` (S4.9r's encoder body) when four rows share a
   piece. **Test**: `the_four_row_softmax_is_each_rows_own_softmax` — the kernel on the same
   length grid at a padded stride, and the tile walk on every piece length up to `BLOCK` with the
   prior `m`/`l` a previous tile leaves, comparing the tile, the maximum, the denominator and the
   rescale factor bit for bit.

No golden file changed: `git diff --stat origin/main...HEAD -- '*.golden'` prints nothing. A row's
bits are still its own — the ascending key order, the canonical tree and the separate
multiply-then-add are untouched; the bodies share only loads and exact F16 widenings.

## 4. The A/Bs

**The score tile, five interleaved rounds, two builds, one thread** (`ab-scores.txt`; every round
of both arms within ±1 % of its median; load 3.25–3.87):

| cell | base | branch | change |
|---|---:|---:|---:|
| 528 tok pass | 6.967 s | 6.820 s | −2.1 % |
| 528 tok scores | 0.274 s | 0.223 s | **−18.4 %** |
| 2 000 tok pass | 33.23 s | 32.02 s | −3.6 % |
| 2 000 tok scores | 3.914 s | 3.122 s | **−20.2 %** |
| 2 000 tok attention | 9.372 s | 8.606 s | −8.2 % |
| 2 000 tok mix bucket | 5.343 s | 5.370 s | +0.5 % (untouched) |
| 2 000 tok matmul | 23.19 s | 22.79 s | −1.7 % (untouched, host drift) |

**The four-row mix and the four-row soft-max do not form on Qwen3-Embedding** — two query heads
per KV head, one position per row, and rows of different positions may not share a piece — so
their A/Bs are on Llama 3.2 1B Q4_K_M, whose four query heads per KV head do form them, attention
only at 4 096 rows, one thread, five interleaved rounds each (`llama-attn.txt`; load 5.0–18.4,
marked loaded):

| change | base | branch | change |
|---|---:|---:|---:|
| the mix's four-row pass | 176.0 tok/s | 183.5 tok/s | **+4.3 %** |
| the soft-max's four denominators | 183.5 tok/s | 200.8 tok/s | **+9.4 %** |

**Dropped: the pair's 16-dimension slice.** Alone it measures 28.4 → 33.7 GMAC/s (eight chains
for one row-pair), but in the pass the mix bucket moved +0.8 % (5.379 → 5.403 s at 2 000 tokens,
one round, load 18–20) — inside the round's spread and in the wrong direction; the in-pass pieces
are the tile walk's, not the bench's. Reverted, and what is left is in §5.

**The round's totals, two interleaved rounds, both arms, medians** (`abc.txt`; load 5–8; the two
runs of each arm within 0.5 %):

| row | cell | before | after | change |
|---|---|---:|---:|---:|
| Qwen3 F16 | 528 tok pass | 6.862 s | 6.828 s | −0.5 % |
| Qwen3 F16 | 528 tok attention | 0.681 s | 0.634 s | −6.8 % |
| Qwen3 F16 | 2 000 tok pass | 32.80 s | 32.10 s | −2.1 % |
| Qwen3 F16 | 2 000 tok attention | 9.356 s | 8.595 s | −8.1 % |
| Qwen3 Q8_0 | 528 tok pass | 3.285 s | 3.253 s | −1.0 % |
| Qwen3 Q8_0 | 2 000 tok pass | 19.35 s | 18.69 s | **−3.4 %** |
| Qwen3 Q8_0 | 2 000 tok attention | 9.484 s | 8.736 s | −7.9 % |
| gte-modernbert F16 | 552 tok pass | 1.549 s | 1.557 s | +0.5 % (guard) |
| gte-modernbert F16 | 552 tok attention | 0.129 s | 0.130 s | +0.3 % (guard) |

## 5. The decoder guard, and what is owed

`promptbench`, Llama 3.2 1B Q4_K_M, a 4 096-token prompt of this repository's own prose, one and
ten threads (`llama-guard.txt`). One thread: **63.97 → 69.02 tok/s (+7.9 %)**, its attention
24.80 → 20.30 s (−18.2 %) and its untouched matmul 37.18 → 37.20 s. Ten threads: 201.06 → 184.91
tok/s, which is the host and not the change — that run's untouched matmul moved 9.03 → 10.22 s
(+13 %) between the arms and attention stayed flat (5.36 → 5.54 s). Both cells are marked
**loaded**.

Owed:

- **The ten-thread engine table against llama.cpp**: the Qwen3-Embedding F16 and Q8_0 rows at 16,
  128, 512 and 2 000 tokens, the 32 × 128 batch and the mixed batch of 32, and gte-modernbert as
  the guard. Every window in this round had the 1-minute load between 4 and 32, so no cell here is
  a standing and `docs/benchmark.md`'s ARM row is not updated. All that can be said from this
  window is the one-thread A/B above: at 2 000 tokens the scores' −20 % and attention's −8 % are
  the pass's −2.1 % (F16) and −3.4 % (Q8_0); the same change applied to a cell that was 0.96× is
  the shape of a cell that should now clear 1.0, and that must be measured, not asserted.
- **The pair's slice width**, already measured and dropped twice (§4): the isolated gain does not
  survive the tile walk's piece structure. The next step there is not a wider slice but a
  restructured tile whose pieces are the chunk boundaries only (i.e. the mix's group is
  widened to the whole task by making the banked `o` per row a function of the chunk, not the
  piece), which changes the online soft-max's steps and so its bits — a different round.
- **The F16 matmul stays at 80 % of its own kernel** by design (S4.9p's trade) and was not
  touched.

## Files

`ab-scores.txt` (five rounds, the score tile), `abc.txt` (two rounds, all three models, both
arms), `llama-guard.txt` (the decoder guard), `llama-attn.txt` (the two Llama A/Bs),
`stagebench.rs` (the kernel bench: the pair/quad/rows-N mix at hd 64/128, the 1/2/4-row score
bodies, the soft-max stages) and `stage.txt` (its output, one thread, `taskset -c 5,15`).
