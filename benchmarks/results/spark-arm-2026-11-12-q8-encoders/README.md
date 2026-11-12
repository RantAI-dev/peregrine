# Q8_0 encoders on the ARM host's kernel tiers: Peregrine before/after S4.9o vs llama.cpp

S4.9o asks why a Q8_0 encoder file is slow on the two int8 aarch64 tiers, and whether the
route the lower tiers take is faster. This run answers that for `gte-modernbert-base`
(`modern-bert`) and `all-MiniLM-L6-v2` (`bert`), per kernel tier, with llama.cpp beside
them, and then measures the change S4.9o makes.

The finding: a Q8_0 `modern-bert` file took the repacked int8 tiles on every tier, and on
aarch64 those tiles have no `i8mm` body — they run on `sdot`, eight output rows at a time.
The NEON float panel kernel is faster per multiply than that tile even at four times the
weight bytes, so on the tiers that carry those tiles the file is now dequantised once into
f32 panels at load, the route a small Q8_0 `bert` file already took on every tier since
S4.9m. `texts/s` at 128 tokens on the native tier: 12.48 → 23.20, and llama.cpp's 14.32 is
now at 0.62× of Peregrine's.

- Date: 2026-11-12.
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters: 10 ×
  Cortex-X925 at 3.9 GHz (CPUs 5–9 and 15–19) and 10 × Cortex-A725 at 2.8 GHz (the rest),
  121 GB unified memory, Linux `6.17.0-1008-nvidia`, governor `performance`, kernel tier
  `aarch64-neon-i8mm`, rustc 1.99.1.
- Peregrine before: `origin/main` at `dc6beba`'s Q8_0 path, built from a clean worktree,
  `cargo build --release -p peregrine`. (It is the binary of `bf1193e`; no commit between
  the two touches this path.)
- Peregrine after: this branch's code commit `d515b2b`, `cargo build --release -p
  peregrine`.
- llama.cpp: v0.9.0-dev (build 1, commit `cd959ba`), GNU 13.3.0 for Linux aarch64,
  `~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-server`, `CUDA_VISIBLE_DEVICES=""`
  so the CPU path is measured.
- Threads: 10, the Cortex-X925 cluster, and every engine is pinned to it —
  `taskset -c 5-9,15-19`. Host load was 0.3 at the start of the session and rose to ~3
  later (co-tenant `vllm` processes), and an unpinned 10-thread process that the scheduler
  put on the A725 cluster measured 2.1× slower on the same binary; the pinning is what
  makes the arms comparable. The spreads below are the min–max over seven rounds.

## Method

`run.py` in this directory (standard library only). One process per engine, every arm up
at once for a model, and the timed rounds interleave the arms in a rotating order so a
frequency or thermal drift is not charged to one engine. Each cell is one
`POST /v1/embeddings` request, timed wall-clock; `texts/s` is texts ÷ wall. The tables
report the median of seven runs, or eleven for the control.

- **Cells**: one text of 16, 128 and 512 tokens, and one batch of 16 texts of 32 tokens
  (511–512 tokens per request). Every text is built from the family's own fixture texts
  (`crates/peregrine-models/tests/fixtures/modern-bert/texts.txt` for gte,
  `.../bert/texts.txt` for MiniLM) and calibrated with llama-server's `/tokenize`
  (`add_special: true`), so the token counts are exact.
- **Peregrine**: `serve -m <file> -t 10 -c <model ctx> --slots 32 --no-profile --opt
  EMBED_CACHE_MB=0`. The exact embedding-vector cache is **off** for every arm: left on, a
  repeated text is a lookup, not a pass. `PEREGRINE_PACK_CACHE=0` keeps two builds from
  racing on the same pack-cache key; the pack is load-time work and no timed cell sees it.
  The tier arms are the same binary with `PEREGRINE_ISA` set: `native` (`aarch64-neon-i8mm`),
  `neon-dotprod` (`aarch64-neon-dotprod`) and `neon` (`aarch64-neon`).
- **llama.cpp**: `-t 10 -c (ctx + 8) × 32 -b 2048 -ub 2048 -np 32 -fa auto --embedding
  --pooling <the file's own>` — `cls` for gte, `mean` for MiniLM.
- **Vectors**: the run's first response per cell is kept; the cosine is Peregrine's vector
  against llama.cpp's for the same text, min and mean over the cell's texts. The raw
  per-cell vectors are not committed (they are 20 MB of JSON); the summaries are.

Reproduce with:

```sh
PIN_CPUS=5-9,15-19 python3 run.py --out summary.json --runs 7
```

`summary.json` is the interleaved before/after run, `summary-tiers-before.json` the
pre-change tier survey, and `summary-float-route-probe.json` the probe below. Engine
command lines are recorded verbatim in each file, with `~` for the home directory.

## Models

| Model | Architecture | Bytes | SHA-256 |
|---|---|---:|---|
| gte-modernbert-base Q8_0 | `modern-bert` | 160,208,512 (151.1 MiB of weights) | `d3c637bb…93301` |
| gte-modernbert-base F16 | `modern-bert` | 299,876,992 | `2cda419c…4e437` |
| all-MiniLM-L6-v2 Q8_0 | `bert` | 25,008,064 (22.4 MiB of weights) | `263215c3…2f8ed` |
| all-MiniLM-L6-v2 F16 | `bert` | 45,949,216 | `797b70c4…be4662` |
| EmbeddingGemma 300M Q8_0 | `gemma-embedding` | 333,590,944 (311.9 MiB of weights) | `b5ce9d77…90d63` |

## What runs for a Q8_0 encoder, per tier

Read from the code and confirmed by the vectors and the goldens: the family's Q8_0 linears
are handed to `MatKernels::path` (`crates/peregrine-kernels/src/table.rs`), which picks the
first of: packed float panels (`fp_off` and a float format or `fp_kind`), the repacked
k-quant tiles (`kq_off` and `kq_tiles_for`), or the per-row kernel.

| File | Tier | Q8_0 matmul | Activations | Why |
|---|---|---|---|---|
| `bert` (MiniLM, ≤ 64 MiB) | all four aarch64 tiers | **float panels** (`fpack`, F32) | f32, no quantisation | S4.9m's dequantise-at-load rule, family `bert`, every target |
| `modern-bert` (gte) — before S4.9o | `aarch64`, `aarch64-neon` | per-row `q8_0_batch` / `dot_batch` | int8 per 32 weights | no `sdot`, so no tiles: `kq_tiles` is off without `dotprod` |
| `modern-bert` (gte) — before S4.9o | `aarch64-neon-dotprod`, `aarch64-neon-i8mm` | repacked 8-row Q8_0 tile (`neon::q8_08_gemm`, `sdot`) | int8 per 32 weights | `kq_tiles` needs `dotprod`; **Q8_0 has no `i8mm` body**, so the `i8mm` tier runs the same `sdot` tile |
| `modern-bert` (gte) — after S4.9o | `aarch64-neon-dotprod`, `aarch64-neon-i8mm` | **float panels** | f32, no quantisation | S4.9o: the float kernel is faster than the tile, and more exact |
| `modern-bert` (gte) — after S4.9o | `aarch64`, `aarch64-neon` | per-row (unchanged) | int8 | the lower tiers never had the tiles; the change is scoped to the tiers that did |
| `gemma-embedding` (Gemma, 312 MiB) | `aarch64`, `aarch64-neon` | per-row | int8 | no tiles without `dotprod` |
| `gemma-embedding` | `aarch64-neon-dotprod`, `aarch64-neon-i8mm` | repacked 8-row Q8_0 tile | int8 | unchanged: its family is not part of the S4.9o rule |

Two things follow from that table. A Q8_0 encoder is the one place where an `aarch64-neon`
run is *not* the same work as the int8 tiers: it drops the tiles and uses the per-row
kernel, which is 3–4× slower here (25.7 against 78.7 texts/s at 16 tokens on gte). And the
two int8 tiers have no `i8mm` Q8_0 kernel to be faster with, so "the i8mm tier" for a Q8_0
file means "the dotprod tile".

## Part 2: the tier survey, before the change

The pre-change binary, `texts/s`, median of seven runs; `spread` is min–max over the seven.

### gte-modernbert-base Q8_0 (151.1 MiB; its F16 counterpart is 299.9 MB)

| Cell | Tokens | i8mm | dotprod | neon | llama.cpp | i8mm ÷ llama | Spread (i8mm) |
|---|---|---:|---:|---:|---:|---:|---|
| L16 b1 | 16 | 78.703 | 76.987 | 25.678 | 73.691 | **1.07×** | 9.5% |
| L128 b1 | 128 | 12.295 | 12.342 | 3.094 | 14.203 | 0.87× | 2.0% |
| L512 b1 | 512 | 3.132 | 3.135 | 0.800 | 3.586 | 0.87× | 1.0% |
| batch 16×32 | 512 | 47.747 | 48.150 | 13.655 | 62.650 | 0.76× | 1.9% |

The same file in F16 on the native tier: 110.582 / 20.747 / 5.333 / 76.436 texts/s — 39%
to 68% faster than the Q8_0 file with 1.9× the weight bytes. Forcing the `neon` tier makes
the Q8_0 pass **3.1× to 3.9× slower**, not faster: the per-row kernel is not the float
route. So the premise "forcing the lower tier is faster" is false as stated; what is
faster is the float *weights* route, which the `bert` family takes and `modern-bert` did
not. That is the probe below.

### all-MiniLM-L6-v2 Q8_0 (22.4 MiB) — the control that already took the float route

| Cell | Tokens | i8mm | dotprod | neon | llama.cpp | i8mm ÷ llama | Spread (i8mm) |
|---|---|---:|---:|---:|---:|---:|---|
| L16 b1 | 16 | 385.656 | 341.603 | 372.291 | 196.496 | **1.96×** | 15.8% |
| L128 b1 | 128 | 155.653 | 145.642 | 163.183 | 96.121 | **1.62×** | 13.9% |
| L512 b1 | 512 | 40.052 | 41.986 | 43.523 | 23.775 | **1.68×** | 5.9% |
| batch 16×32 | 511 | 685.007 | 664.769 | 686.521 | 455.241 | **1.50×** | 5.9% |

Its F16 file on the native tier: 200–368 / 109–141 / 27.5–38.4 / 569–638 texts/s across
runs. All four tiers run the same float panels and land within the run's spread of each
other, which is what a tier-independent route looks like. (This model's cells are 3–25 ms,
so the 16-token and 128-token spreads reach 15% on this host.)

## Part 3: the float-panel probe, and the change it supports

**The probe.** Before writing any code, the same build was measured with the S4.9m rule
lifted for `modern-bert` at the load step only (a throwaway build, not committed). It ran
the f32 panels on the native tier: `texts/s` 97.725 / 20.634 / 3.928 / 92.357 at
16 / 128 / 512 / batch, against llama.cpp's 71.769 / 14.071 / 3.617 / 63.380 in the same
run. Against the pre-change medians (78.703 / 12.295 / 3.132 / 47.747) it is faster at
**every** cell: +24% at 16 tokens, +68% at 128, +25% at 512, +93% on the batch. So the
change is not shape-dependent and needs no batch- or length-dependent switch.

**The change** (`crates/peregrine-core/src/model.rs`,
`crates/peregrine-core/src/options.rs`): on `aarch64-neon-dotprod` and up — the tiers that
carry the Q8_0 int8 tiles — a `modern-bert` file of at most `Q8_FLOAT_MAX_BYTES_ARM`
(256 MiB; the measured gte file is 151 MiB) is dequantised into f32 panels at load, exactly
as a small `bert` file already was. `scale × code` is exact in f32 and the float path
quantises no activation, so this route is **more exact** than the one it replaces, not
less: the synthetic Q8_0 file's vector is now the scalar float reference's bit for bit on
those two tiers, and the ignored real files below agree. x86 keeps the int8 tiles and the
64 MiB `bert` rule; the lower aarch64 tiers keep the per-row kernel they already ran;
`gemma-embedding`, the causal embedders and `PEREGRINE_Q8_FLOAT=0` are unchanged.

### Before / after, interleaved in one run (gte Q8_0)

`texts/s`, median of seven runs; each arm ran in every round. `ratio` is that row ÷
llama.cpp of the same run.

| Cell | Before | After | Gain | llama.cpp | Ratio before | Ratio after | Spread (after) |
|---|---:|---:|---:|---:|---:|---:|---|
| L16 b1 | 80.915 | **103.196** | 1.28× | 71.961 | 1.12× | **1.43×** | 6.8% |
| L128 b1 | 12.484 | **23.199** | 1.86× | 14.317 | 0.87× | **1.62×** | 2.0% |
| L512 b1 | 3.168 | **6.404** | 2.02× | 3.627 | 0.87× | **1.77×** | 2.8% |
| batch 16×32 | 48.306 | **98.502** | 2.04× | 63.384 | 0.76× | **1.55×** | 3.1% |

The dotprod arm of the same run, 104.111 / 23.346 / 6.459 / 98.789, is within 2% of the
native arm: the two int8 tiers now agree, and both are `aarch64-neon`'s 25.625 / 3.070 /
0.802 / 13.584 × 3–4 (that tier keeps the per-row kernel; see "what was not done"). The
Q8_0 file is now at or above its own F16 sibling from 128 tokens up (23.199 against
20.907; 6.404 against 5.345; 98.502 against 75.710) at 1.9× fewer weight bytes.

Controls in the same run, which no code change touches:

- gte F16 native: 110.582 → 110.246 / 20.747 → 20.907 / 5.333 → 5.345 / 76.436 → 75.710
  (−0.3% to +0.8%, inside the spread).
- MiniLM Q8_0 native: 369.263 → 372.225 / 159.633 → 162.269 / 43.884 → 43.643 /
  696.911 → 697.297 (+0.8% / +1.7% / −0.5% / +0.1%). The file converted before and after,
  so the arms are the same code path; this is the control that the change leaves it alone.
  An eleven-round rerun of this one model alone, with the same two builds, was not usable:
  this host's co-tenant load moved one arm's whole run by 2× at the 3 ms cells, which is
  why the primary run keeps every arm together and pins them.

### Vectors against llama.cpp (gte Q8_0, per cell, min / mean cosine)

| Cell | Before (int8 tile) | After (f32 panels) |
|---|---|---|
| L16 | 0.9998901 / 0.9998901 | 0.9999010 / 0.9999010 |
| L128 | 0.9999010 / 0.9999010 | 0.9998966 / 0.9998966 |
| L512 | 0.9997947 / 0.9997947 | 0.9998605 / 0.9998605 |
| batch 16×32 | 0.9997992 / 0.9998498 | 0.9998114 / 0.9998507 |

llama.cpp's own Q8_0 has its int8 activation quantiser too, so a Q8_0-versus-Q8_0 cosine
measures two quantisers, not exactness; it moves by a few 1e-5 either way. The exactness
claim is against the float reference, and it is in the next table.

### The 200-text fixtures (`--ignored` real-file tests, before → after)

`cargo test --release -p peregrine-models --test embeddings <test> -- --ignored --nocapture`,
on this host (native tier):

| File | vs llama.cpp min / mean | vs the fp32/F16 reference min / mean |
|---|---|---|
| gte-modernbert-base Q8_0, before | 0.9969122 / 0.9998028 | 0.9963540 / 0.9996125 |
| gte-modernbert-base Q8_0, after | **0.9983872 / 0.9998352** | **0.9994969 / 0.9997834** |
| all-MiniLM-L6-v2 Q8_0 (both) | 0.9995759 / 0.9997501 | 0.9995207 / 0.9997637 |
| gte-modernbert-base F16 (both) | 0.9999936 / 0.9999997 | 0.9999936 / 0.9999997 |

The Q8_0 distance to the full-precision reference falls from 0.9964 to 0.9995 at the
minimum and from 0.99961 to 0.99978 on the mean: the remaining gap is the Q8_0 weights
themselves, not the activation quantiser. Peregrine's Q8_0 file is now closer to the full
precision reference than llama.cpp's own Q8_0 vector is (0.9981270 / 0.9996284), and the
synthetic test asserts the stricter statement on both int8 tiers — the Q8_0 vector *is*
the scalar float reference's, bit for bit (max abs 0.0).

## Goldens

The synthetic `modern-bert` Q8_0 golden was recorded on the four aarch64 tiers by S4.9n;
the two int8 tiers' hashes change with this route (the scalar and neon tiers keep the
per-row kernel and their hashes):

| Golden | aarch64, aarch64-neon | dotprod, i8mm before | dotprod, i8mm after |
|---|---|---|---|
| `modern-bert-Q8_0.golden`, mean | `07f65ef98b0e8dc5` (unchanged) | `301cbef7263b78d7` | `51c9434cc8ba546e` |
| `modern-bert-Q8_0.golden`, cls 64 | `a05d5ecbd3b65f03` (unchanged) | `bb1ba15a4583cf63` | `55e17bd3fd81801d` |

`bert-Q8_0.golden` (`b544043b48866d26`, `7dcc2ebe69f0abcd`) and
`bert-heads-Q8_0.golden` (`ca54d90d43e88fb2`, `e4b7b55d6db4b1bd`, `f65b54f0d4866827`,
`79e9c20796542234`) are unchanged on every tier. No x86 golden moves: the rule is
`cfg!(target_arch = "aarch64")` plus `dotprod`.

## EmbeddingGemma 300M Q8_0: not changed, and why

`gemma-embedding` is its own family and stays on the int8 tiles. Its tier table from the
same pinned run (median of seven): 90.579 / 14.627 / 3.567 / 56.208 on the int8 tiers,
28.755 / 3.650 / 0.921 / 15.437 on `neon`, llama.cpp 79.662 / 12.375 / 3.399 / 64.877. Two
reasons not to extend the rule: the S4.9m x86 measurement has the float route losing at
300M (29% slower at 16 tokens, level at 128), and this family's `act_layout` only keeps the
f32 activation rows for a float *file* — a probe that converted its Q8_0 matrices panicked
at `fpack.rs:320` ("float matrix matmul reached with keep_f32 off"). Extending it needs its
own act_layout change and 1.3 GiB of panels, and is not supported by a measurement. The
causal embedders (`qwen3-embed`, the causal `llama`/`qwen3` files) take the int8 tiles as
before: the rule is scoped to one encoder family and to the bidirectional pass.

## What was not done

- The `aarch64` and `aarch64-neon` tiers still run the per-row Q8_0 kernel for
  `modern-bert` (25.6 / 3.07 / 0.80 / 13.6 texts/s). The same measurement says the float
  route would help them too (97.7 / 20.6 / 3.9 / 92.4 in the probe), but the change is
  scoped to the tiers that ran the int8 tiles, so no other tier's golden moves and the
  fallback tier does not pay 4× the weight memory.
- The batch cells are 16 × 32 tokens, not 32 texts of mixed length; the earlier ARM
  embedding run covers mixed batches for the same engines.
- No 2000-token cell: 512 is the longest text here and the shape where the int8 tile lost
  most to the float route. The 128/512 columns carry the claim.
- x86 was not measured and does not change. Granite Embedding R2 is not on this machine;
  gte is the only measured `modern-bert` file, and the constant is what covers it.
- The 64 MiB `bert` constant was not re-measured for bge-m3 (604 MiB, not on this machine);
  it stays at the x86 measurement, and large `bert` Q8_0 files keep their tiles on ARM too.
- A Q8_0 `i8mm` (`smmla`) tile kernel was not written. It is the other way to close the
  gap on the int8 tiers, but the float route is exact and already ahead of llama.cpp, so
  that kernel has no measured reason to exist.
