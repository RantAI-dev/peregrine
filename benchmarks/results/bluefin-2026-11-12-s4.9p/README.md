# S4.9p on bluefin: the small-BERT float matmul against the machine's roof, and rerank (2026-11-12)

S4.9l (the proof run, `benchmarks/results/bluefin-2026-11-12-s4.9l-proof/`) left two
model families below the best equal-accuracy engine at 128 and 512 tokens:
all-MiniLM-L6-v2 at **0.85–0.99×** and bge-small-en-v1.5 at **0.85–0.97×**, with the pass
profile saying "matmul 58–60 %, attention 23–25 %, other 16–17 %". This run asks the next
question rather than assuming the answer: how fast *can* this CPU multiply, how fast does
Peregrine's kernel actually multiply, how fast does OpenVINO's, and which of the three
stages is the gap in.

The answer is not the one the profile's shares suggest. **The matmul is already level with
OpenVINO's** (165–178 against 173 GMAC/s, both ≈ 64–69 % of the 4-core FMA roof). The gap
is the **element-wise work**: 9.3 ms of Peregrine's 512-token MiniLM pass against 0.5 ms in
OpenVINO's, and 18.1 ms against 1.6 ms on bge-small; plus, on bge-small only, the attention
strip (25.7 against 20.7 ms).

One kernel change came out of it, and it is a real one on the kernel and small on the cells:
the halved-precision float GEMM now widens a weight panel to f32 **once per panel** instead
of once per weight step, and it no longer carries the format branch inside the weight loop.
MiniLM's F16 row moved **0.87× → 0.90×** at 512 tokens, **0.96× → 1.02×** at 128 b1, and
**0.99× → 1.10×** at 16 b32; the F32 row was untouched and did not move, which is the
control.

## Machine, build and models

| | |
|---|---|
| machine | bluefin: AMD Ryzen 3 PRO 4350G (Renoir, Zen 2), 4 cores / 8 threads, 128 KiB L1d + 512 KiB L2 per core, 4 MiB L3, 13 GB, kernel 7.2.7-200.fc44, governor `powersave` |
| flags | `fma avx avx2` (no AVX-512, so tier `x86_64-avx2`) |
| threads | 4 everywhere |
| base commit | `origin/main` at `bf1193e` (peregrine 0.1.2) — the "before" binary |
| branch | `s4.9p-bert-gemm`, the "after" binary |
| build | `cargo build --release -p peregrine`; `cellprof` from `peregrine-bench` with `arch-bert,arch-gemma-embedding,arch-qwen,arch-llama,arch-mistral3,fmt-kquant,fmt-legacy,fmt-iq` |
| rustc | 1.99.1 |
| models | `~/models/embed/minilm/minilm-f16.gguf`, `minilm-q8_0.gguf`, `bge-small/bge-small-f32.gguf`, `bge-small-q8_0.gguf` (unchanged from S4.9l: same SHA-256 as its `raw/*.json`) |
| texts | `/var/tmp/embed-texts-s49l/` (the S4.9l sets) |
| engines | Peregrine 0.1.2; llama.cpp v0.9.0 (`cd959ba`); OpenVINO 2026.4.1 (sentence-transformers 6.2.0 backend); ONNX Runtime 1.31.1; fastembed 0.9.1; TEI `cpu-1.9.4` (candle) |
| host load | 1-minute load average 2.6–2.9 during every run (another agent shares this host, the same condition S4.9l recorded at 3.6–4.8); the harness's gate was set to `--max-load 20` and the load is in every raw JSON |

## Part 1 — the kernels against the machine

### 1.1 The roof

`peregrine-core`'s `telem::Machine::probe` spins twelve independent `ymm` FMA chains per
thread (no memory in the loop) and reports the throughput it reaches:

```
$ /var/tmp/target-s49p/release/examples/probe 4
{"flops": 512575177692.7, "gflops": 512.575, "threads": 4, "simd": "avx2+fma", ...}
```

and `fpbench`'s own one-thread probe of the same shape: **64.9 GMAC/s** (129.8 GFLOP/s).

The arithmetic. Zen 2 has two 256-bit FMA pipes, so one core retires
`2 × 8 = 16` f32 multiply-adds per cycle, i.e. `32` FLOP per cycle:

```
129.8 GFLOP/s ÷ 32 FLOP/cycle = 4.06 GHz      (the clock the probe sustains with 4 threads)
64.9 GMAC/s × 4 cores         = 259.6 GMAC/s  (519.2 GFLOP/s)  the 4-core roof
```

4.06 GHz is consistent with the part (3.8 GHz base, 4.1 GHz max boost) and with the probe
being the only load. Every rate below is a share of **64.9 GMAC/s per core** and
**259.6 GMAC/s for four**.

### 1.2 Peregrine's float matmul on the encoder shapes

Both models are `d_model = 384`, `ffn = 1536`, so one layer has six matrix products and one
token costs

| matrix | shape | MAC per token |
|---|---|---:|
| q | 384 × 384 | 147 456 |
| k | 384 × 384 | 147 456 |
| v | 384 × 384 | 147 456 |
| attention output | 384 × 384 | 147 456 |
| ffn up | 384 × 1536 | 589 824 |
| ffn down | 1536 × 384 | 589 824 |
| **one layer, one token** | | **1 769 472** |

all-MiniLM-L6-v2 is 6 layers, bge-small-en-v1.5 is 12:

| model | rows | matmul MAC per pass | GFLOP |
|---|---:|---:|---:|
| MiniLM (6 layers) | 128 | 1 358 954 496 | 2.72 |
| MiniLM | 512 | 5 435 817 984 | 10.87 |
| bge-small (12 layers) | 128 | 2 717 908 992 | 5.44 |
| bge-small | 512 | 10 871 635 968 | 21.74 |

Measured matmul, the pass profile's own `matmul` bucket (median of three `cellprof` passes,
4 threads, one text, 4-thread pool):

| cell | matmul ms | GMAC/s | of the 4-core roof | per thread | of the 1-core roof |
|---|---:|---:|---:|---:|---:|
| MiniLM f16 L128 b1 | 7.88 | 172.5 | 66.4 % | 43.1 | 66.4 % |
| MiniLM f16 L512 b1 | 32.94 | 165.0 | 63.6 % | 41.3 | 63.6 % |
| bge-small f32 L128 b1 | 14.37 | 189.1 | 72.8 % | 47.3 | 72.8 % |
| bge-small f32 L512 b1 | 60.96 | 178.3 | 68.7 % | 44.6 | 68.7 % |

The same kernel alone, one thread, warm, `fpbench` (0.25 s per measurement, 1024-row
matrices are the same picture), before → after this task:

| n_in × n_out | rows | f32 | f16 before | f16 after | bf16 before | bf16 after |
|---|---:|---:|---:|---:|---:|---:|
| 384 × 384 | 128 | 59.2 | 50.2 | **57.3** | 51.5 | **57.2** |
| 384 × 1536 | 128 | 57.3 | 50.0 | **57.3** | 51.4 | **57.2** |
| 1024 × 1024 | 128 | 56.0 | 49.0 | **58.1** | 52.1 | **57.7** |
| 1024 × 3072 | 128 | 55.2 | 48.4 | **56.2** | 51.4 | **56.1** |
| 2048 × 8192 | 128 | 49.1 | 44.6 | **52.6** | 47.8 | **52.5** |

So: the f32 tile is at **86–91 %** of this core's roof and the halved-precision tile was at
**77 %** before this task and is at **88 %** after. In situ, four threads reach 63–73 % of
four cores — a 4-thread aggregate of 165–189 GMAC/s, which is what the probe below confirms
is the machine's ceiling for this kernel (a shape sweep in the appendix: every blocking,
1 to 4 threads, lands between 150 and 205 GMAC/s).

### 1.3 Attention

Per layer the strip costs `2 · n² · d_model` MAC — `n² · d_model` for the scores
(every head's `n² · head_dim`, summed) and the same again for the value mix — plus the
softmax: `n² · heads` `exp` evaluations.

| cell | MAC per pass | attention ms | GMAC/s | of the roof | per thread |
|---|---:|---:|---:|---:|---:|
| MiniLM L128 b1 | 75 497 472 | 1.07 | 70.6 | 27.2 % | 17.6 |
| MiniLM L512 b1 | 1 207 959 552 | 12.69 | 95.2 | 36.7 % | 23.8 |
| bge-small L512 b1 | 2 415 919 104 | 25.66 | 94.2 | 36.3 % | 23.5 |

MiniLM's 512-token pass pays 18.9 M `exp` calls; bge-small's 37.7 M. The strip runs at
about **55 %** of what the same core reaches in the matmul, which is the `exp`, the score
buffer's read-backs (max, exp, sum, mix each walk the strip) and the 32-wide head's short
`k` loop.

### 1.4 OpenVINO on the same cell, from its own profiler

`PERF_COUNT` on the CPU plugin, the same ONNX the embeddings harness gives its backend,
4 threads, one 512-token inference after a warm-up, per-node `real_time` aggregated by node
type (`appendix/ovprof.py`):

| node type | MiniLM ms | share | bge-small ms | share |
|---|---:|---:|---:|---:|
| `FullyConnected` (the 36 / 72 matmuls) | 31.38 | 72.0 % | 62.59 | 74.1 % |
| `Subgraph` (attention: scores, softmax, mix) | 10.60 | 24.3 % | 19.78 | 23.4 % |
| `Transpose` | 0.54 | 1.2 % | 0.91 | 1.1 % |
| `Gather`, `MVN` (embedding, LayerNorm) and the rest | 1.06 | 2.4 % | 1.14 | 1.4 % |
| **total** | **43.58** | | **84.42** | |

- OpenVINO's matmul rate: `5.436 GMAC ÷ 31.38 ms = 173.2 GMAC/s` (MiniLM, 66.7 % of the
  roof, 43.3 per thread) and `10.872 ÷ 62.59 = 173.7 GMAC/s` (bge-small). Peregrine measures
  165.0 and 178.3 on the same cells: **the two kernels are level**.
- OpenVINO's attention: 11.1 ms (MiniLM) and 20.7 ms (bge-small) — Peregrine's 12.69 and
  25.66.
- OpenVINO's element-wise work: **0.5–1.6 ms** against Peregrine's 9.26 and 18.05 ms.
- The whole-request lower bound asked for, if *all* of a request were matmul: the HTTP p50
  is 47.96 ms (MiniLM) and 89.2 ms (bge-small), so `5.436 ÷ 47.96 ms = 113.3 GMAC/s` and
  `10.872 ÷ 89.2 = 121.9 GMAC/s`. Its own profiler then shows that only 72–74 % of that is
  matmul, which is the more useful number.

### 1.5 The kernel as it was, end to end

- **The path.** `peregrine_models::bert::Bert::forward` puts each layer's projections in one
  `blocks::linear::linear_group` call (q, k, v in one fork–join; ffn up/down the same way)
  → `blocks::linear::matmul_group` → `matmul_block` → `fp_group` when every matrix of the
  group is packed float → `peregrine_kernels::table::MatKernels::fp_panels_gemm[_strided]`
  → `peregrine_kernels::fpack::gemm_strided` → `fpack::x86::gemm_avx2` on this tier
  (`gemm_avx512` when `have_zmm()`, which Zen 2 does not have) → `gemm_scalar_strided` when
  the tier is scalar. The single-row (decode) path is `x86::gemv_avx2`.
- **The store.** `peregrine_core::fpack::repack_with` packs every float matrix **once at
  load** into an `FpStore` (anonymous mapping, huge pages; `TensorInfo::fp_off` points at it;
  the pack cache keeps it on disk so a second load reuses it). A matrix `[n_out][n_in]` is cut
  into panels of 16 output rows; panel `p` is `n_in` steps of 16 contiguous values, step `k`
  holding `W[16p..16p+16][k]`, in the file's own element type (f32, f16 or bf16). The last
  panel is zero-padded. A Q8_0 matrix may be dequantised into f32 panels at load (S4.9m), and
  for the files measured here it is — which is why the Q8_0 and F32 rows have always run at
  the same speed.
- **The loop nest.** `(row block) → (panel) → (6 activation rows) → k ascending`, the input
  index `k` innermost. One inner step loads the panel's 16 weights as two `ymm`, broadcasts
  six activation values and issues **12 FMAs into 12 accumulators**
  (`[[__m256; 2]; 6]`: 6 rows × 16 outputs); with the two weight vectors and one broadcast
  temp that is 15 of AVX2's 16 `ymm` registers. A 128-row block is 22 tiles of six rows; the
  last tile of a block duplicates its last row's pointer and throws the lanes away.
  384-wide panels (12 KiB) stay in L1 across a block's tiles, 1536-wide ones (48 KiB) in L2.
- **The thread split.** Nothing in the kernel threads. `fp_group` builds work units of
  **2 panels × one row block** (`fp_row_block`, at most `row_block(n_in) = L2/2 / (4·n_in)`
  and at most `MAX_BATCH = 128` rows) over the concatenated panels of the group's matrices,
  and the pool's workers claim them from an atomic counter (or take equal shares with
  `PEREGRINE_DYNAMIC=0`). A unit is exactly the work the kernel would do serially, so the
  split changes no work, no weight traffic and no bit.
- **The defined arithmetic** — the scalar twin `gemm_scalar_strided` is the definition, and
  every tier computes it lane per output:

  ```text
  out[b][r] = acc = bias[r] (or +0.0);  for k in 0..n_in { acc = fma(x_b[k], w[r][k], acc) }
  ```

  one **sequential fused-multiply-add chain over k** for each output element, one lane per
  output. No reassociation, no k-lanes reduced at the end, no horizontal sum: the tile, the
  row block, the thread and the vector width cannot reach a bit. f16 is widened by
  `vcvtph2ps` and bf16 by the high-half shift, both exact, so they contribute no rounding.
  (NaN payloads are the one thing a hardware `vfmadd` and a libm `fmaf` need not agree on;
  the value class does.)

### 1.6 Where the time goes

| stage | Peregrine (MiniLM 512) | OpenVINO (same cell) | verdict |
|---|---:|---:|---|
| matmul | 32.94 ms (165.0 GMAC/s) | 31.38 ms (173.2 GMAC/s) | **level** (Peregrine 5 % behind, inside the run-to-run band) |
| attention | 12.69 ms | 11.14 ms | 1.5 ms behind |
| element-wise, embedding, pooling | 9.26 ms | 0.55 ms | **8.7 ms behind** |
| total pass | 54.84 ms | 43.58 ms | 0.79× on kernel time; 0.90× through HTTP |

and on bge-small at 512: matmul 60.96 against 62.59 (**ahead**), attention 25.66 against
20.69 (5.0 behind), element-wise 18.05 against 1.14 (**16.9 behind**).

The stage profile's "matmul 58–60 %" is not a statement that the matmul is slow; it is 60 %
of a pass whose other 40 % is 2.4× OpenVINO's *whole* request. Specifically:

- **Not** memory traffic: f16 and f32 panels run at the same rate, and a cold 48 MiB weight
  store measures within 2 % of a warm one in the probe (appendix).
- **Not** too few accumulators: 12 accumulators against 2 FMA pipes with 5-cycle latency
  needs 10 in flight; the f32 tile is at 91 % of the roof.
- **Not** small inner blocks: a 6×16 tile issues 12 FMAs and 8 loads per step, which is at
  the FMA ports' limit; 4-row and 12-row shapes measured worse.
- **Not** thread imbalance: the units are equal-cost and claimed from a counter; unpinned and
  pinned-to-4-physical-cores runs agree (180–205 GMAC/s aggregate), and pinning to two
  physical cores' siblings halves it, as it should.
- The halved-precision tile **was** conversion-bound: `vcvtph2ps` issues on the same two ports
  as the FMA, so a step was 14 vector operations for 12 multiplies (77 % of the roof); fixed
  here.
- What is left is a **per-element pass structure**: each LayerNorm, GELU, residual add and
  f32 activation copy reads and writes the 512×384 (or 512×1536) block again, while oneDNN
  fuses them into the GEMM epilogue. bge-small's extra loss is attention, whose 32-wide heads
  leave the score tile a 32-step `k` loop and whose strip is re-walked four times
  (max, exp, sum, mix).

## Part 2 — the change

### What changed (`crates/peregrine-kernels/src/fpack.rs`)

1. **The format left the weight loop.** `gemm_avx2` used to call a `w16(fmt, p)` helper
   inside the `k` loop, which the compiler turned into two compares and two branches per
   weight step (the disassembly is in the appendix). It is now three monomorphised kernels
   (`gemm_avx2_f32`, `gemm_avx2_f16_direct`, `gemm_avx2_bf16_direct`) generated from one
   macro, dispatched once per call.
2. **A halved-precision panel is widened once.** `gemm_avx2_half` widens a whole
   `n_in × 16` panel into a per-thread f32 scratch (`WIDE`, grown once) and then runs the
   *f32* kernel's tile loop on it. The conversion moves out of the weight step: it is paid
   `n_in × 16` times per (panel, row block) instead of once per (panel, row block, row tile),
   a factor of the block's tiles (21 at 128 rows). It is used when the pass has more than
   `WIDEN_ROWS = 18` activation rows (three tiles; below that the widening costs more than it
   saves: 16 rows measured 47.1 GMAC/s direct against 44.2 widened) and when the scratch fits
   in half of L2 (`n_in ≤ 4096` here), so a very wide matrix keeps the direct loop and its
   half-size weight stream.

Nothing else moved: the tile, the row block, the unit (2 panels × block), the store layout,
the thread split and the bias handling are as they were. The AVX-512 body is **untouched** —
this CPU cannot run it, and `gemm_avx512` still calls the generic `w16`. The NEON bodies are
untouched too.

### Why the bits are the same

- The defined arithmetic is unchanged: one sequential `fma` chain over `k` per output, one
  lane per output. Splitting the kernel by format changes which instructions run, not their
  order; unrolling nothing, reassociating nothing.
- The widening is exact. `vcvtph2ps`/`fcvtl` (f16→f32) and the bf16 high-half shift are
  exact conversions, so the f32 panel holds exactly the values the old loop widened per
  step, and the FMA sees exactly the same operands in exactly the same order. The f32 chain
  then *is* the f16 chain, bit for bit — which is why the widened path and the direct path
  compare equal element-wise, not merely close.
- Tested (new, `crates/peregrine-kernels/src/fpack.rs`):
  - `the_widened_panel_is_the_direct_loop`: F16 and BF16, `n_out × n_in` = 16×384, 16×1536,
    48×768, 33×37, 16×1024, 24×385; rows 1, 6, 17, 18, 19, 24, 40, 64 — the widened kernel
    and the direct kernel are `to_bits` equal everywhere.
  - `every_row_count_and_width_is_the_scalar_chain`: every row count 1..=40 and
    127, 128, 129, 511, 512, 513 × widths 37, 383, 384, 385, 513, 768, 1023, 1024, 1536,
    1537, all three formats, against `f32::mul_add`'s chain.
  - `the_edge_values_keep_every_bit`: ±0.0, the smallest subnormal, an f16-subnormal
    magnitude, 1e-38, 2^-14, ±1, ±3e4, ±6.5e4, ±inf and NaN (`is_nan`) as activations and as
    weights, at 1, 5, 17, 18, 19 and 24 rows.
  - and the pre-existing `a_row_does_not_depend_on_the_batch`, `the_strided_store_is_...`,
    `the_special_values_take_the_same_chain`, `the_vector_kernels_are_the_scalar_chain`
    (which also runs the AVX-512 kernel under `have_zmm`), `the_row_block_does_not_change_a_result`.
- No golden file changed: `git status` shows nothing under any `tests/golden/`, and
  `cargo test --release -p peregrine-models` fails exactly one assertion — the pre-existing
  `bert_heads` golden (see Checks).

### The kernel after

| | f16 (before) | f16 (after) | bf16 (before) | bf16 (after) | f32 |
|---|---:|---:|---:|---:|---:|
| 384×384, 128 rows | 50.2 | **57.3** (+14 %) | 51.5 | **57.2** (+11 %) | 59.2 |
| 2048×8192, 128 rows (no widening, branch hoisted only) | 44.6 | **52.6** (+18 %) | 47.8 | **52.5** (+10 %) | 49.1 |
| 384×384, 16 rows (below the threshold, branch hoisted only) | 46.3 | 47.1 | 48.3 | 51.8 | 54.2 |

## Part 3 — measurement

Method: the S4.9l proof runner unchanged (`benchmarks/proof_run.py` from
`origin/bench-bluefin-s4.9l-proof`, `embed.py` with the same flags:
`--lengths 16,128,512 --batches 1,8,32 --clients 1 --budget 0.5 --max-requests 20
--max-big-requests 2 --mixed-cell --mixed-batch 32 --threads 4`, and
`PEREGRINE_EMBED_CACHE_MB=0`). Three rounds per binary, **interleaved by row** (round 1
before, round 1 after, round 2 before, round 2 after, round 3 before, round 3 after);
`texts/s` is the median of the three. Ratio is Peregrine ÷ the best **other-engine** row at
cosine ≥ 0.999 against the fp32 reference (Peregrine's own other precision is not a
competitor, as in S4.9l).

### 3.1 MiniLM and bge-small, before and after

#### all-MiniLM-L6-v2, every row, texts/s (median of three)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fastembed fp32 | 233.5 | 574.0 | 606.1 | 80.2 | 90.4 | 73.9 | 15.5 | 13.2 | 13.0 | 13.7 |
| fastembed int8 | 280.7 | 595.5 | 711.4 | 87.8 | 94.9 | 86.1 | 15.0 | 13.9 | 14.0 | 14.5 |
| llama.cpp f16 | 238.6 | 403.0 | 425.5 | 55.2 | 55.1 | 55.5 | 13.1 | 12.0 | 12.1 | 58.6 |
| llama.cpp q8_0 | 270.7 | 458.7 | 474.7 | 63.7 | 63.0 | 63.2 | 14.8 | 13.4 | 13.6 | 66.2 |
| onnxruntime fp32 | 241.2 | 586.0 | 613.3 | 87.3 | 91.0 | 72.7 | 15.8 | 13.2 | 13.0 | 13.7 |
| onnxruntime int8 | 251.8 | 392.1 | 617.9 | 89.7 | 95.2 | 85.9 | 15.1 | 13.9 | 13.9 | 14.4 |
| openvino fp32 | 196.5 | 533.2 | 653.1 | 78.2 | 98.8 | 95.2 | 20.9 | 21.0 | 20.4 | 21.9 |
| **peregrine f16** | **376.5** | **665.3** | **718.8** | **89.3** | **96.6** | **97.7** | **18.8** | **19.2** | **19.1** | **100.4** |
| **peregrine q8_0** | **341.0** | **665.3** | **717.2** | **89.9** | **97.3** | **97.9** | **19.0** | **19.3** | **19.4** | **100.4** |
| tei fp32 | 141.4 | 191.4 | 225.7 | 33.7 | 35.9 | 35.7 | 4.2 | 4.2 | 4.2 | 11.6 |

Peregrine f16 against the best other engine, before → after:

| cell | before | after | best other engine (after) |
|---|---:|---:|---|
| L16 b1 | 1.30× | **1.39×** | llama.cpp q8_0 270.7 |
| L16 b8 | 1.18× | 1.14× | onnxruntime fp32 586.0 |
| L16 b32 | 0.99× | **1.10×** | openvino fp32 653.1 |
| L128 b1 | 0.96× | **1.02×** | onnxruntime fp32 87.3 |
| L128 b8 | 0.92× | 0.98× | openvino fp32 98.8 |
| L128 b32 | 0.97× | **1.03×** | openvino fp32 95.2 |
| L512 b1 | 0.87× | **0.90×** | openvino fp32 20.9 |
| L512 b8 | 0.88× | **0.91×** | openvino fp32 21.0 |
| L512 b32 | 0.89× | **0.94×** | openvino fp32 20.4 |
| mixed b32 | 1.45× | **1.52×** | llama.cpp q8_0 66.2 |

Peregrine q8_0 (a row this change does not touch: Q8_0 is dequantised to f32 panels at
load), as the control, before → after: L16 b1 1.27×→1.26×, L16 b8 1.46×→1.45×,
L16 b32 1.11×→1.10×, L128 b1 1.03×→1.03×, L128 b8 0.98×→0.98×, L128 b32 1.04×→1.03×,
**L512 b1 0.91×→0.91×**, L512 b8 0.92×→0.92×, L512 b32 0.94×→0.94×, mixed b32 1.52×→1.52×.
The F16 row closed on the Q8_0 row, which is what the widening was for; the Q8_0 row itself
did not move.

#### bge-small-en-v1.5, every row, texts/s (median of three)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fastembed fp32 | 159.4 | 349.9 | 403.6 | 48.5 | 48.1 | 39.3 | 8.3 | 7.2 | 7.3 | 7.5 |
| fastembed int8 | 187.8 | 357.9 | 374.5 | 48.1 | 49.3 | 47.6 | 8.5 | 8.4 | 8.6 | 8.4 |
| llama.cpp f32 | 147.8 | 307.6 | 330.3 | 42.2 | 40.0 | 40.2 | 9.1 | 8.2 | 8.3 | 40.8 |
| llama.cpp q8_0 | 169.2 | 264.1 | 268.0 | 34.9 | 31.8 | 32.2 | 7.6 | 6.8 | 6.9 | 33.9 |
| onnxruntime fp32 | 164.5 | 353.5 | 405.7 | 49.8 | 47.1 | 39.4 | 8.4 | 7.2 | 7.3 | 7.5 |
| onnxruntime int8 | 220.2 | 362.2 | 407.1 | 48.8 | 49.4 | 47.2 | 8.5 | 8.4 | 8.6 | 8.4 |
| openvino fp32 | 141.2 | 345.6 | 402.6 | 45.9 | 53.8 | 51.1 | 11.2 | 11.0 | 10.9 | 11.3 |
| **peregrine f32** | **213.4** | **377.9** | **394.3** | **47.2** | **49.3** | **49.5** | **9.6** | **9.7** | **9.7** | **51.0** |
| **peregrine q8_0** | **212.9** | **364.0** | **391.9** | **47.4** | **49.2** | **48.8** | **9.4** | **9.7** | **9.6** | **50.9** |
| tei fp32 | 81.4 | 114.4 | 124.2 | 19.1 | 18.5 | 18.8 | 2.5 | 2.3 | 2.1 | 4.9 |

Peregrine f32 against the best other engine, before → after (unchanged, as it should be):

| cell | before | after | best other engine |
|---|---:|---:|---|
| L16 b1 | 1.22× | **1.26×** | llama.cpp q8_0 169.2 |
| L16 b8 | 1.07× | 1.07× | onnxruntime fp32 353.5 |
| L16 b32 | 0.97× | 0.97× | onnxruntime fp32 405.7 |
| L128 b1 | 0.97× | 0.95× | onnxruntime fp32 49.8 |
| L128 b8 | 0.92× | 0.92× | openvino fp32 53.8 |
| L128 b32 | 0.98× | 0.97× | openvino fp32 51.1 |
| L512 b1 | 0.86× | 0.86× | openvino fp32 11.2 |
| L512 b8 | 0.89× | 0.89× | openvino fp32 11.0 |
| L512 b32 | 0.89× | 0.89× | openvino fp32 10.9 |
| mixed b32 | 1.24× | 1.25× | llama.cpp f32 40.8 |

Stage profile after (median of three `cellprof` passes, 4 threads, one text):

| cell | request ms | matmul | attention | other | matmul GMAC/s |
|---|---:|---:|---:|---:|---:|
| MiniLM f16 L16 b1 | 1.74 | 64.2 % | 16.8 % | 19.1 % | 105.1 |
| MiniLM f16 L128 b1 | 10.97 | 71.8 % | 9.8 % | 18.3 % | 172.5 |
| MiniLM f16 L512 b1 | 54.84 | 60.1 % | 23.1 % | 16.9 % | 165.0 |
| MiniLM q8_0 L512 b1 | 53.05 | 58.8 % | 24.0 % | 16.9 % | 174.3 |
| bge-small f32 L128 b1 | 20.45 | 70.3 % | 10.6 % | 18.9 % | 189.1 |
| bge-small f32 L512 b1 | 104.44 | 58.4 % | 24.6 % | 17.3 % | 178.3 |
| bge-small q8_0 L512 b1 | 109.09 | 57.2 % | 25.7 % | 17.5 % | 174.3 |

### 3.2 The other embedding models did not move

Peregrine's rows at L128 b1 / L512 b1, one run each, before and after (texts/s):

| model | precision | L128 b1 before → after | L512 b1 before → after |
|---|---|---:|---:|
| bge-m3 | q8_0 | 7.68 → 7.72 | 2.60 → 2.58 |
| EmbeddingGemma 300M | q8_0 | 15.18 → 15.26 | 7.26 → 7.26 |
| Qwen3-Embedding 0.6B | f16 | 6.49 → 6.80 | 1.75 → 1.66 |
| Qwen3-Embedding 0.6B | q8_0 | 5.72 → 5.71 | 1.96 → 1.96 |
| Nemotron 3.5 Embed 1B | q8_0 | 4.63 → 4.75 | 1.12 → 1.12 |

Everything is inside the ±5 % band two single runs have on this host; the only F16 row
(Qwen3) reads +5 % at 128 and −5 % at 512, which is the band, not a direction. The Q8_0
files reach f32 panels at load, so the changed kernel is not on their path at all.

### 3.3 Rerank

`bge-reranker-v2-m3` (635 676 416 B, SHA-256 `a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3`,
the `bge-reranker-v2-m3-Q8_0.gguf` the S4.6b work used) and `bge-reranker-base`
(303 770 720 B, SHA-256 `4bf3f9269c55954a5ad4d9e0e7fb05b448cb1053818170f497242ff760fd46bb`), both
GGUF `Q8_0` from the same uploads as `docs/reference/bert.md`. 4 threads. One query, 1, 8 and
32 documents of about 128 and about 512 tokens (the bench builds a document by repeating a
paragraph and reads the exact count back from the response's `usage`).

Both engines were driven through HTTP on the same file, one engine at a time, with
`scripts/rerank-bench.py --docs N --tokens T --reps 5` (median of five requests, `top_n` =
every document):

- Peregrine: `peregrine serve -m <file> --threads 4 --ctx 16640 --port 18112` (the encoder
  clamps ctx to the file's 8 192 and runs 32 rows per pass — one row per document of the
  largest request).
- llama.cpp: `llama-server -m <file> --reranking -t 4 -np 32 -c 16640 -b 1024 -ub 1024
  --host 127.0.0.1 --port 18111` — one slot per document of the largest request, and a
  ubatch that holds a whole 449-token pair (it rounded the pool to 24 576 tokens, 768 per
  slot, which still holds the largest pair).

bge-reranker-v2-m3, pairs/s:

| tokens/pair | documents | Peregrine | llama.cpp | ratio |
|---|---:|---:|---:|---:|
| 123 | 1 | 3.87 | 3.31 | **1.17×** |
| 123 | 8 | 3.91 | 2.77 | **1.41×** |
| 123 | 32 | 3.92 | 2.76 | **1.42×** |
| 449 | 1 | 0.96 | 0.78 | **1.23×** |
| 449 | 8 | 0.96 | 0.73 | **1.32×** |
| 449 | 32 | 0.96 | 0.73 | **1.32×** |

bge-reranker-base, pairs/s:

| tokens/pair | documents | Peregrine | llama.cpp | ratio |
|---|---:|---:|---:|---:|
| 123 | 1 | 12.37 | 11.10 | **1.11×** |
| 123 | 8 | 13.11 | 9.66 | **1.36×** |
| 123 | 32 | 13.14 | 9.68 | **1.36×** |
| 449 | 1 | 3.25 | 2.76 | **1.18×** |
| 449 | 8 | 3.26 | 2.52 | **1.29×** |
| 449 | 32 | 3.25 | 2.52 | **1.29×** |

Peregrine's pairs/s is flat in the document count (one text per pass at these sizes, which is
the same "pass budget" note S4.6b made); llama.cpp's falls back towards its 32-slot packing.
The two files carry Q8_0 matrices, which the loader already converts into f32 panels
(S4.9m), so **this task's kernel change is not on their path**: the movement since S4.6b
(0.98–1.06× then) is the element-wise/float-path work of S4.9h/S4.9m, and this run's job for
rerank is to record the table on the current main.

Ids and order. The same 8-document request at each length was sent to both engines and the
score vectors compared:

| file | tokens/pair | Kendall τ | top-1 same | max \|score difference\| |
|---|---:|---:|---|---:|
| bge-reranker-base | 123 | 1.000 | yes | 0.097 |
| bge-reranker-base | 449 | 1.000 | yes | 0.038 |
| bge-reranker-v2-m3 | 123 | 0.929 | yes | 0.107 |
| bge-reranker-v2-m3 | 449 | 0.929 | yes | 0.071 |

v2-m3 swaps **one adjacent pair** in each cell: documents 5 and 6 at 123 tokens, 6 and 4 at
449. Those two documents differ by 0.013 (123) and 0.018 (449) in llama.cpp's own scores —
smaller than the two engines' score difference on the same document (0.04–0.11) — so the
swap is the near-tie, not a different ranking. On real pairs the repository's own
`rerank_real` test compares the two engines over 320 (query, document) pairs in eight
languages and asserts identical order (`docs/reference/bert.md`: identical for both files).

Not added, and why: a text-embeddings-inference row for the reranker needs the checkpoint as
safetensors (`BAAI/bge-reranker-v2-m3`, ~2.3 GB) which is not on this machine, and a
sentence-transformers `CrossEncoder` on ONNX Runtime/OpenVINO needs the same checkpoint plus
an export — both are large downloads, so this run does neither (`/var/tmp/embed-venv` has
sentence-transformers 6.2.0 and torch 2.15.0, and the TEI `cpu-1.9.4` image is present, but
neither has a reranker checkpoint to serve).

## Checks

| check | result |
|---|---|
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| `cargo test --release -p peregrine-kernels` | 159 pass |
| `PEREGRINE_ISA=scalar cargo test --release -p peregrine-kernels` | 159 pass |
| `cargo test --release -p peregrine-models` | all pass **except** `bert_heads::the_synthetic_bert_heads_match_their_golden_for_the_tier` |
| `PEREGRINE_ISA=scalar cargo test --release -p peregrine-models` | same single failure |
| `cargo test --release -p peregrine-server -p peregrine` | pass |
| `cargo hack check --each-feature --no-dev-deps -p peregrine-kernels -p peregrine-core -p peregrine-models` | pass (42 feature sets) |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace` | pass |
| no file under `tests/golden/` changed | confirmed (`git status`) |

That `bert_heads` failure is **pre-existing on `origin/main`**: with this branch's
`fpack.rs` stashed, the same test fails with the same two strings, on both the native AVX2
tier and `PEREGRINE_ISA=scalar` — `rank len 37 width 1` records FNV `e4b7b55d6db4b1bd` and
both tiers now compute `0b310e7e572078c2`, while `rank len 7` and both `labels` rows still
match. Both tiers agreeing with each other and disagreeing with the recorded golden means the
golden is stale (the golden was recorded in S4.6b; the head's path changed in S4.9h/S4.9m).
No golden was re-recorded here, per the task's rule.

## What is not done, and what is still behind

- **The element-wise work is the remaining gap** (8.7 ms of a 54.8 ms MiniLM pass; 16.9 ms of
  a 104.4 ms bge-small pass) and it is a different task: it needs the LayerNorm/GELU/residual
  work fused into the matmul's epilogue, or at least written without a full read-write pass
  per element. This run measures it, it does not fix it.
- **bge-small's attention** is 25.66 ms against OpenVINO's 20.69; its 32-wide heads give the
  score tile a 32-step `k` loop and the strip is walked four times. Not attempted here.
- **The AVX-512 tier keeps the old body.** This CPU has no AVX-512, so `gemm_avx512` still
  calls the generic `w16` inside its weight loop and does not use the widened-panel path.
  The same change is a mechanical consequence there (widen the two panels into a scratch and
  run the 256-bit body on it) but it cannot be measured on this host, and an unmeasured
  change to a tier whose tests cannot run here is not worth shipping.
- **A 512-row pass per matmul** (lifting `MAX_BATCH`, S4.9b) would cut the weight stream from
  4× to 2× at 512 tokens and with it the 63.6 % → 72.8 % difference between the L512 and L128
  cells. Also not this task.
- The 4-thread aggregate is 165–189 GMAC/s against a 259.6 GMAC/s roof even for the best
  kernel, and the probe says that is the machine: the same kernel, every blocking, 1 thread
  = 50–59 GMAC/s, 4 threads = 150–205 GMAC/s. A single-threaded comparison against OpenVINO
  would flatter Peregrine by ~15 %; the 4-thread cells are the honest ones.

## Appendix — the probe, the disassembly, and the engine

- The unit-shape sweep was a temporary `peregrine-bench` example (not kept: it is not part of
  the engine or of the benchmark suite). It measured the same pass as 2, 6 and 24 panels per
  unit against one call for the whole matrix, at 128 and 512 rows, on 1 and 4 threads, with
  `fpbench`'s repetition (0.3 s per measurement), and a cold-store variant that walks a
  different 48 MiB copy of the matrix on each repetition. Results quoted above: every 4-thread
  shape lands between 150 and 205 GMAC/s; cold and warm are within 2 %; a run pinned to two
  physical cores' siblings reaches 102–111 GMAC/s (as expected, not a defect); the
  2-panel-unit shape is 10–25 % behind one call at **one** thread and level with it at four,
  because at four threads the machine, not the tile, is the limit.
- `appendix/gemm_before.s`: the old `gemm_avx2` weight loop — `vmovups`×2, `vbroadcastss`×6,
  `vfmadd231ps`×12, plus `test/je` (the prefetch guard), `cmpb/je` (fmt = F32), `cmpl/jne`
  (fmt = F16) per step, i.e. two branches and five compare/test uops inside the innermost
  loop.
- `appendix/ovprof.py`: the OpenVINO per-node profiler used for §1.4.
- `appendix/rerank-order.py`: the request pair used for the order comparison.

## Reproducing

```
# the two binaries
git worktree add ~/wt-s49p -B s4.9p-bert-gemm origin/main
CARGO_TARGET_DIR=/var/tmp/target-s49p cargo build --release -p peregrine
cp /var/tmp/target-s49p/release/peregrine ~/tmp/s49p/peregrine-new
# origin/main's binary for the "before" rows: same command on a clean checkout

# the timing grid: three interleaved rounds per binary
for r in 1 2 3; do
  python benchmarks/proof_run.py --mode bench --round $r --models minilm,bge-small \
      --out-dir benchmarks/results/bluefin-2026-11-12-s4.9p/raw-before \
      --peregrine-binary ~/tmp/s49p/peregrine-base --max-seconds 400
  python benchmarks/proof_run.py --mode bench --round $r --models minilm,bge-small \
      --out-dir benchmarks/results/bluefin-2026-11-12-s4.9p/raw-after \
      --peregrine-binary ~/tmp/s49p/peregrine-new --max-seconds 400
done

# the stage profile (median of three passes per cell)
cargo build --release -p peregrine-bench --example cellprof \
    --features peregrine-models/arch-bert,peregrine-models/arch-gemma-embedding,\
peregrine-models/arch-qwen,peregrine-models/arch-llama,peregrine-models/arch-mistral3,\
peregrine-models/fmt-kquant,peregrine-models/fmt-legacy,peregrine-models/fmt-iq
PEREGRINE_PROF2= $T/release/examples/cellprof ~/models/embed/minilm/minilm-f16.gguf \
    /var/tmp/embed-texts-s49l/minilm.json 16,128,512 1 4

# the kernel
/var/tmp/target-s49p/release/examples/fpbench
/var/tmp/target-s49p/release/examples/probe 4

# OpenVINO's own per-layer times
/var/tmp/embed-venv/bin/python appendix/ovprof.py ~/models/embed/minilm/onnx-fp32.onnx 512

# rerank
peregrine serve -m ~/models/embed/bge-reranker-v2-m3-Q8_0.gguf --threads 4 --ctx 16640 --port 18112
python scripts/rerank-bench.py --base-url http://127.0.0.1:18112 --name peregrine --docs 32 --tokens 512
llama-server -m ~/models/embed/bge-reranker-v2-m3-Q8_0.gguf --reranking -t 4 -np 32 -c 16640 \
    -b 1024 -ub 1024 --host 127.0.0.1 --port 18111
python scripts/rerank-bench.py --base-url http://127.0.0.1:18111 --name llama.cpp --docs 32 --tokens 512
```

Raw data: `raw-before/`, `raw-after/` (`<model>-<engine>-<precision>-r{1,2,3}.json`, the
harness's own record with host state, versions and per-cell latency), `raw-others-*/`
(`r1`), `profile/*.jsonl` (one `cellprof` line per pass, three per cell).
