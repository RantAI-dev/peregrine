# S4.9r on bluefin: the encoder's element-wise block, threaded and batched (2026-11-12)

S4.9p found that the two small BERT encoders are behind OpenVINO not in the matrix products —
Peregrine's float GEMM is level with oneDNN's, 165–178 against 173 GMAC/s — but in everything
around them: on a 512-token pass, **9.26 ms** of MiniLM's time was element-wise work (embedding,
biases, residuals, LayerNorms, the GELU, pooling, copies) against the 0.55 ms S4.9p counted for
OpenVINO, and **18.05 ms** on bge-small against 1.14 ms; attention was 12.69 against 11.14 ms and
25.66 against 20.69 ms. (Counting every OpenVINO node that is neither a matmul nor attention, the
same profiler gives 2.36 and 3.76 ms — §1.5 below — which is the fairer number and the one this
run works against.)

This run does two things: it takes that "other" time apart step by step, in pass order, beside
OpenVINO's own per-node types; and then it makes the steps it can make faster **without moving a
bit**. The short answer: the element-wise block is now **7.54 ms** (MiniLM) and **15.98 ms**
(bge-small), attention is **10.77** and **21.4 ms**, and the 512-token pass went from 53.4 to
**49.5 ms** and from 104.4 to **98.8 ms**. MiniLM's 512-token cells reached level
(0.82→0.99, 0.91→1.00, 0.93→1.01 against the best equal-accuracy engine); bge-small moved from
0.85–0.90 to 0.92–0.95 and is still behind. Every vector is bit-identical: the same text gives
the same 4 789 floats before and after, at 1 and 4 threads, alone and in a batch of 8.

What did **not** move, and why, is the exact-`erf` GELU: 4.5 ms of MiniLM's remaining 7.5 and
10.7 of bge-small's 16.0. It is one sequential `f64` Horner chain per lane (10, 22 or 36 terms
by `|x/√2|`), measured at 1.1–1.5 `f64` FMAs per cycle against the core's two ports; every
reachable variant either changes a bit or loses to the chain's latency. It is the price of the
model's own arithmetic, and OpenVINO does not pay it because it fuses a fast approximation into
its GEMM epilogue.

## Machine, build and models

| | |
|---|---|
| machine | bluefin: AMD Ryzen 3 PRO 4350G (Renoir, Zen 2), 4 cores / 8 threads, 128 KiB L1d + 512 KiB L2 per core, 4 MiB L3, 13 GB, Fedora 44, kernel 7.2.7-200.fc44, governor `powersave` |
| flags | `fma avx avx2` (no AVX-512; tier `x86_64-avx2`) |
| threads | 4 everywhere |
| before | `origin/s4.9p-bert-gemm` at `b3c9a2b` — the "before" binary and the account below |
| after | this branch, `d776367` on top of `origin/main` `b53ec7f` — the "after" binary |
| build | `cargo build --release -p peregrine`; `cellprof` from `peregrine-bench` with `--features peregrine-models/arch-bert,peregrine-models/arch-gemma-embedding,peregrine-models/arch-qwen,peregrine-models/arch-llama,peregrine-models/arch-mistral3` |
| rustc | 1.99.1 |
| models | `~/models/embed/minilm/minilm-f16.gguf`, `minilm-q8_0.gguf`, `bge-small/bge-small-f32.gguf`, `bge-small-q8_0.gguf` — the same files (same SHA-256) as S4.9l and S4.9p |
| texts | `/var/tmp/embed-texts-s49l/` (the S4.9l sets) |
| engines | Peregrine 0.2.0; llama.cpp v0.9.0 (`cd959ba`); OpenVINO 2026.4.1 (sentence-transformers 6.2.0 backend); ONNX Runtime 1.31.1; fastembed 0.9.1; TEI 1.9.4 (candle) |
| host load | 1-minute load average 2.5–3.6 during every run, with 4–6 GB of RAM held by the desktop session (another agent and the browser share this host; `--max-load 20 --min-avail-gb 3.5`) — the same condition S4.9l and S4.9p recorded, and the reason every table is interleaved and reported as a median |

## Method

**The account.** `peregrine-bench`'s `cellprof` drives the same `Engine` the server does, one
cell at a time, one text per pass, 4 threads. For the step breakdown a temporary thread-local
timer table was added to `Bert::forward_texts` (**not committed**; the working tree that produced
these numbers is `origin/s4.9p-bert-gemm` plus the timers), one `Instant` pair around each step
in pass order, and worker-side atomic accumulators around the parts that run inside a pool task
(the GELU's bias add, the `erf` call and the activation copy; the norm's reduction and its act
copy). A timer pair is ~20 ns against a 30–100 µs step, and the medians are of four or five
`cellprof` invocations after its warm-up pass. Times are wall milliseconds of the pass; the
"worker" times quoted for parallel steps are the summed per-thread times divided by the pool
width, and every table says which one it is.

**OpenVINO beside it.** `appendix/ovprof.py` here (the S4.9p script, extended to print every node
type) reads the same ONNX the embeddings harness gives the OpenVINO backend, compiles it for CPU
with `PERF_COUNT`, runs one warm-up and then one 512-token inference, and sums each node's
`real_time` by type. Three runs per model; the tables below are the last (warm, quiet) one.

**End to end.** The S4.9l proof runner unchanged — `benchmarks/proof_run.py --mode bench`, three
interleaved rounds (round 1 before, round 1 after, round 2 before, …), `embed.py` with
`--lengths 16,128,512 --batches 1,8,32 --clients 1 --budget 0.5 --max-requests 20
--max-big-requests 2 --mixed-cell --mixed-batch 32 --threads 4` and `PEREGRINE_EMBED_CACHE_MB=0`
— medians of the three rounds. Ratio is Peregrine ÷ the best **other-engine** row at cosine ≥
0.999 against the fp32 reference (the S4.9l accuracy files, unchanged because every vector is).
The same runner was given one new flag, `--min-avail-gb` (default 5.0, unchanged), because the
desktop session leaves this 13 GB host at 4.7 GB free.

## Part 1 — the account

### 1.1 MiniLM-L6 F16, 512 tokens, 4 threads (before)

Stage profile: matmul **31.2 ms**, attention **12.74 ms**, element-wise **9.31 ms**, pass
**53.4 ms**. The rows below are the 9.31 ms, in pass order, plus the attention split (1.3).

| step | ms | MiB read+write | GB/s | threads | SIMD |
|---|---:|---:|---:|---|---|
| token embedding lookup (F16 → f32) | 0.49 | 1.13 | 2.4 | 1 | scalar `f16` loop |
| + token-type row 0 | 0.03 | 1.50 | 56 | 1 | auto-vectorised |
| position embedding lookup (F16) | 0.43 | 1.13 | 2.7 | 1 | scalar `f16` loop |
| + position | 0.03 | 1.50 | 56 | 1 | auto-vectorised |
| `token_embd_norm` + activation fill | 0.29 | 3.00 | 10.9 | 4 | AVX2, 8 rows of f64 lanes |
| copy `xb` → `x` (the norm's output) | 0.03 | 1.50 | 60 | 1 | `memcpy` |
| q/k/v bias adds | 0.66 | 4.50 | 7.2 | **1** | auto-vectorised |
| × 1/√32 on q | 0.14 | 2.25 | 17 | **1** | auto-vectorised |
| attn_output bias + residual (`wo` post) | 0.35 | 9.00 | 27 | **1** | auto-vectorised |
| `attn_output_norm` + activation fill | 0.36 | 3.00 | 8.8 | 4 | AVX2 |
| ffn_up bias + exact GELU + activation fill | **4.48** | 54.1 | 12.7 | 4 | AVX2 f64 `erf` |
| ffn_down bias + residual | 0.35 | 13.5 | 40 | **1** | auto-vectorised |
| `layer_output_norm` + activation fill | 0.41 | 3.00 | 7.7 | 4 | AVX2 |
| pooling (mean, f64) + store | 0.04 | 0.75 | 20 | 1 | scalar f64 sum |
| **sum of steps** | **8.64** | | | | |
| thread wake-up (187 fork-joins), buffer checks, stats | 0.67 | — | — | — | — |
| **element-wise total (stage profile)** | **9.31** | | | | |

The GELU pass is 4.48 ms of it: bias add 0.35, the `erf` itself **3.60** (worker sum 14.3 ms over
four threads), the activation copy 0.45, the rest the fork-join. Its 3.60 ms is 4.72 M elements,
i.e. **0.77 ns — 3.1 cycles — per element**: the chain described at the top.

The four steps that run on **one thread** (the q/k/v biases, the query scale, the two
residual+bias reads) are 1.50 ms together; the two F16 embedding lookups are 0.92 ms and run at
2.4 GB/s because the widening is a scalar loop.

### 1.2 bge-small-en-v1.5 F32, 512 tokens, 4 threads (before)

Stage profile: matmul **60.9 ms**, attention **25.48 ms**, element-wise **17.78 ms**, pass
**104.4 ms** (12 layers, twice MiniLM's).

| step | ms | MiB read+write | GB/s | threads | SIMD |
|---|---:|---:|---:|---|---|
| token embedding lookup (F32) | 0.09 | 1.50 | 17.9 | 1 | auto-vectorised |
| + token-type row 0 | 0.03 | 1.50 | 54 | 1 | auto-vectorised |
| position embedding lookup (F32) | 0.08 | 1.50 | 20.7 | 1 | auto-vectorised |
| + position | 0.03 | 1.50 | 56 | 1 | auto-vectorised |
| `token_embd_norm` + activation fill | 0.30 | 3.00 | 10.7 | 4 | AVX2 |
| copy `xb` → `x` | 0.03 | 1.50 | 60 | 1 | `memcpy` |
| q/k/v bias adds | 1.44 | 4.50 | 3.3 | **1** | auto-vectorised |
| × 1/√32 on q | 0.33 | 2.25 | 7.2 | **1** | auto-vectorised |
| attn_output bias + residual (`wo` post) | 0.70 | 28.3 | 42 | **1** | auto-vectorised |
| `attn_output_norm` + activation fill | 0.74 | 3.00 | 4.3 | 4 | AVX2 |
| ffn_up bias + exact GELU + activation fill | **10.66** | 151 | 14.2 | 4 | AVX2 f64 `erf` |
| ffn_down bias + residual | 0.67 | 28.3 | 42 | **1** | auto-vectorised |
| `layer_output_norm` + activation fill | 0.81 | 3.00 | 3.9 | 4 | AVX2 |
| pooling (CLS: a row copy) | 0.001 | 0.001 | — | 1 | `memcpy` |
| **sum of steps** | **16.65** | | | | |
| thread wake-up (373 fork-joins), buffer checks, stats | 1.13 | — | — | — | — |
| **element-wise total (stage profile)** | **17.78** | | | | |

Here the GELU is 10.66 ms (the `erf` itself 8.9) and the four single-threaded steps are
3.14 ms. The norms are 1.55 ms over 25 calls.

### 1.3 The same at 128 tokens

Everything above scales with the row count, so the 128-token pass is the 512 one over four, and
the two totals agree with what the stage profile measured (1.76 and 3.52 ms):

| step | MiniLM 128 | bge-small 128 |
|---|---:|---:|
| token + type + position, lookups and adds | 0.25 | 0.06 |
| embedding LayerNorm + fill + copy | 0.02 | 0.02 |
| q/k/v biases + query scale | 0.14 | 0.28 |
| attn_output bias + residual | 0.09 | 0.13 |
| `attn_output_norm` + fill | 0.06 | 0.13 |
| ffn_up bias + GELU + fill | 1.00 | 2.52 |
| ffn_down bias + residual | 0.06 | 0.13 |
| `layer_output_norm` + fill | 0.08 | 0.19 |
| pooling | 0.01 | 0.001 |
| **sum** | **1.71** | **3.44** |
| unattributed | 0.05 | 0.08 |
| **element-wise total** | **1.76** | **3.52** |
| attention | 1.07 | 2.18 |
| matmul | 5.32 | 10.77 |
| **pass** | **9.79** | **19.71** |

### 1.4 Attention at 512 tokens (before)

The strip is walked per `(text, strip, query head)`: the keys are transposed into a panel, then
the strip's scores are written, read back for the maximum, read and written for the exp and the
denominator, and read again for the value mix.

| step | MiniLM ms | bge-small ms | MiniLM MiB | GB/s | threads | SIMD |
|---|---:|---:|---:|---:|---|---|
| key-panel build (transpose) | 1.05 | 2.07 | 9.0 / 18.9 | 8.9 | 4 | tiled copy |
| scores (`Q·Kᵀ`) | 2.98 | 6.05 | 81 / 170 | 28 | 4 | AVX2 6×16 tile |
| row maximum | 0.42 | 1.00 | 72 / 151 | 179 | 4 | AVX2 `maxps` |
| exp + ascending f64 denominator | **5.30** | **10.50** | 144 / 302 | 28 | 4 | AVX2 `exp8`, **scalar f64 sum** |
| value mix (`P·V`) | 3.66 | 7.03 | 81 / 170 | 23 | 4 | AVX2 6-row tile |
| fork-join gaps | 1.33 | 2.83 | — | — | — | — |
| **attention (stage profile)** | **12.74** | **25.48** | | | | |

The scores run at **202 GMAC/s** and the mix at 168 — at this machine's roof for the shape (the
S4.9p appendix measured 150–205 GMAC/s for every blocking). The softmax is the one step far from
its arithmetic: 37.7 M `exp` calls plus 37.7 M `f64` adds on one serial chain per row, 0.14 ns per
element, which is the add's four-cycle latency and nothing else — three of every four cycles
spent waiting.

### 1.5 OpenVINO on the same cells, from its own profiler

`PERF_COUNT` on the CPU plugin, the same ONNX, 4 threads, one 512-token inference after a
warm-up, per-node `real_time` aggregated by type (`appendix/ovprof.py`):

| node type | MiniLM ms | bge-small ms |
|---|---:|---:|
| `FullyConnected` (36 / 72 matmuls, bias fused) | 35.59 | 72.07 |
| `Subgraph` — attention (scores, softmax, mix) | 12.70 | 25.39 |
| `MVN` (13 / 25 LayerNorms) | 0.40 | 0.81 |
| `Gather` (word + token-type embeddings) | 0.20 | 0.27 |
| `Transpose` (q/k/v, 6 / 12) | 0.49 | 0.90 |
| `Convert` (the masks' dtype) | 0.15 | 0.05 |
| `Broadcast` (the mask) | 0.03 | 0.00 |
| `Subgraph`, non-attention (residual adds, the LayerNorm epilogue, `Where`) | 1.09 | 1.75 |
| `Gelu` (6 / 12 nodes) | **0.00** | **0.00** |
| `Eltwise`/`Add` | — | — |
| **total** | **50.70** | **101.22** |
| **everything but matmul and attention** | **2.36** | **3.76** |

Two things are worth saying plainly. First, the `Gelu` and the residual `Add`s cost OpenVINO
*nothing* as separate nodes because oneDNN fuses them into the `FullyConnected` nodes' post-ops:
that is the structural difference, not a faster GELU. Second, even counting every node type
OpenVINO has, its non-matmul/non-attention work is 2.36 and 3.76 ms against Peregrine's 9.31 and
17.78 — **3.9× and 4.7×**, not the 17× that S4.9p's narrower 0.55/1.14 comparison suggested.

### 1.6 The ranking, by time that could be saved

| rank | step | MiniLM ms | bge-small ms | what could be saved |
|---:|---|---:|---:|---|
| 1 | exact-`erf` GELU | 3.60 | 8.90 | nothing within the bits: a dependency chain on `f64` FMAs |
| 2 | softmax denominator (serial f64 sum) | 5.30 | 10.50 | most of it: four rows' chains in flight |
| 3 | q/k/v biases + query scale, one thread | 0.80 | 1.77 | all but a barrier: rows are independent |
| 4 | the two residual+bias walks, one thread | 0.70 | 1.37 | all but a barrier |
| 5 | F16 embedding widening | 0.92 | — (F32) | nearly all: `vcvtph2ps` |
| 6 | LayerNorms + activation copies | 1.06 | 1.55 | little: already 4-thread and AVX2, bound by the `xf` copy |
| 7 | attention scores, mix, panel | 7.69 | 15.15 | scores and mix are at the roof; 1 ms of panel |
| 8 | pooling, copies | 0.07 | 0.03 | — |

## Part 2 — the fix, same bits

Everything below is one commit (`d776367`). The rule for all of it: the **scalar twin is the
definition**, and each new vector or threaded body is `to_bits`-equal to it.

### 2.1 Four softmax denominators in flight (`exp_rows4_sums`)

The denominator is defined as an ascending `f64` sum of the row's weights, so it cannot be
reassociated — but four **rows** are four independent chains, and a global layer's rows share
their band. `peregrine_kernels::bidir::exp_rows4_sums` takes four rows of scores, one shift each,
computes `exp(x - max)` eight keys at a time with `exp8`, and advances the four sums interleaved
in ascending key order; a row's own fold and its own bits are untouched. The caller
(`blocks/bidir.rs`) groups a strip's rows by four and keeps the per-row path for a sliding
window, whose rows do not share a band.

Saving: softmax **5.30 → 3.18 ms** on MiniLM, **10.50 → 6.46 ms** on bge-small; attention
**12.74 → 10.77** and **25.48 → 21.4**.

Tests (`crates/peregrine-kernels/src/bidir.rs`): every length 0–40 and 127/128/129/383/384/385/
511/512/513, two strides (row-tight and padded), random batches of 20 shifts; the edge values ±0,
the smallest subnormal, 1e-38, ±88 (exp under/overflow), ±inf and NaN; and the bounds asserts.
Each case is compared with `exp_row` + `sum_row` per row, bit for bit, and the padding is checked
untouched.

### 2.2 The q/k/v prologue and the residual reads on four threads

`qkv_prologue` applies bq, bk and bv and the query scale to the three projections in one
fork-join over rows; `add_bias_residual_par` does the attn_output and ffn_down bias+residual the
same way. The arithmetic stays `(out + bias) * scale` and `(out + bias) + residual` per element,
in that order, and a projection whose bias is all zero still skips the add (so a `-0.0` stays a
`-0.0`). The `wo` post moved out of the matmul chunk loop into one pass over the batch: the same
values, one barrier instead of four.

Saving: q/k/v prologue **0.80 → 0.22** (MiniLM) and **1.77 → 0.49**; the two residual walks
**0.70 → 0.35** and **1.37 → 0.70**; the extra barriers cost ~0.1 and ~0.2 ms.

### 2.3 F16 embedding rows through F16C

`dequant_row`'s F16 arm widened through a scalar `f16` loop (2.4 GB/s). It now widens sixteen
halves per step with `vcvtph2ps`, with the scalar loop kept as the twin. F16C's conversion is
exact for every finite half; a NaN's payload is not (the hardware quiets it), so a vector holding
a NaN re-widens those sixteen values with the scalar body — no real weight is a NaN, and the test
holds all 65 536 half patterns to the twin, payloads included.

Saving: the two F16 lookups **0.92 → 0.12 ms** on MiniLM; bge-small is F32 and did not move.

### 2.4 The GELU

The AVX2 body now walks 32 elements per call instead of 16 (two 16-element blocks in one loop
body). That is the whole change: it lets the out-of-order engine overlap the tail of one block
with the next, and it measured 3.60 → 3.45 ms on the 512 pass (4 %). It is left there because
it is free and bit-identical, not because it closes the step.

Tried and rejected, with what was measured:

- **Per-group path selection** (each four-lane group continues to 22 or 36 terms only if one of
  its lanes needs it): 3.7 → 6.0 ns/element in a microbenchmark. The branches keep the twelve
  polynomial chains out of the register file and the shortened chains cannot hide the `f64`
  add's latency.
- **32 lanes with per-half path selection**: at best the same rate. The block already spends
  three of four cycles waiting on the longest `p36` chain (36 terms × 4 cycles); wider blocks
  amortise it but do not shorten it, and 8 groups × 3 polynomials does not fit the 16 vector
  registers.
- A block-wide single polynomial: not bit-equal — a lane in `(1, 2]` is defined by the 22-term
  series, not the 36-term one.

### 2.5 What the block costs after the change

| step (512 tokens) | MiniLM before | after | bge-small before | after |
|---|---:|---:|---:|---:|
| token/position/type embedding | 0.98 | **0.20** | 0.29 | 0.29 |
| embedding LayerNorm + fill + copy | 0.32 | 0.32 | 0.33 | 0.33 |
| q/k/v biases + query scale | 0.80 | **0.22** | 1.77 | **0.49** |
| attn_output bias + residual | 0.35 | **0.16** | 0.70 | **0.31** |
| `attn_output_norm` + fill | 0.36 | 0.37 | 0.74 | 0.72 |
| ffn_up bias + GELU + fill | 4.48 | **4.40** | 10.66 | **10.72** |
| ffn_down bias + residual | 0.35 | **0.19** | 0.67 | **0.39** |
| `layer_output_norm` + fill | 0.41 | 0.44 | 0.81 | 0.80 |
| pooling | 0.04 | 0.04 | 0.001 | 0.001 |
| **element-wise (stage profile)** | **9.31** | **7.54** | **17.78** | **15.98** |
| attention | 12.74 | **10.77** | 25.48 | **21.40** |
| pass | 53.39 | **49.51** | 104.40 | **98.80** |

### 2.6 Why the bits are unchanged

- The softmax kernel advances each row's `f64` accumulator over its keys in ascending order,
  one key at a time, with the same `exp(x − max)` values; the rows are only interleaved in the
  instruction stream, and the tests hold it to `exp_row` + `sum_row` on every length and every
  edge value.
- The threaded prologue and residual passes give a row to one worker; the arithmetic of a row is
  unchanged (`(o + b) * s`, `(o + b) + r`), and the zero-bias skip is preserved.
- The F16 widening is exact per the F16C definition, with the scalar twin for NaN payloads.
- The GELU unroll evaluates the same 16-element blocks; nothing was reordered or fused.
- Measured, not just argued: for both models, `peregrine embed` on the same 512-token text
  produces the same 4 789 / 4 721 floats with the before binary and the after binary, at 1 and 4
  threads, and a batch of 8 matches the same texts embedded alone. `git status` shows no file
  under any `tests/golden/` changed.

## Part 3 — measurement

### 3.1 all-MiniLM-L6-v2, every row, texts/s (median of three interleaved rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fastembed fp32 | 234.6 | 464.7 | 607.4 | 84.7 | 82.9 | 73.8 | 15.6 | 13.2 | 13.2 | 13.6 |
| fastembed int8 | 223.3 | 388.3 | 609.3 | 82.9 | 93.2 | 85.3 | 14.8 | 14.0 | 13.9 | 14.4 |
| llama.cpp f16 | 234.7 | 403.3 | 425.8 | 56.8 | 54.7 | 54.8 | 13.0 | 11.9 | 12.0 | 58.3 |
| llama.cpp q8_0 | 264.5 | 458.6 | 453.8 | 59.3 | 59.6 | 62.0 | 14.7 | 13.1 | 13.4 | 64.8 |
| onnxruntime fp32 | 254.2 | 583.5 | 602.1 | 87.3 | 91.0 | 73.4 | 15.7 | 13.2 | 12.9 | 12.7 |
| onnxruntime int8 | 280.6 | 388.0 | 611.6 | 89.2 | 93.8 | 84.6 | 14.9 | 13.9 | 13.7 | 14.3 |
| openvino fp32 | 195.9 | 536.8 | 644.1 | 77.3 | 98.2 | 93.9 | 20.6 | 20.7 | 20.3 | 21.7 |
| **peregrine f16** | **385.9** | **693.0** | **699.8** | **93.0** | 95.3 | **101.0** | **20.4** | 20.6 | **20.5** | **107.0** |
| **peregrine q8_0** | **334.6** | **678.2** | **653.0** | **92.3** | **100.4** | **100.6** | **20.5** | **20.6** | **20.9** | **103.7** |
| tei fp32 | 156.3 | 199.4 | 258.2 | 35.7 | 36.9 | 38.1 | 4.3 | 4.1 | 4.2 | 9.4 |

Peregrine f16 ÷ best other engine at cosine ≥ 0.999, before → after:

| cell | before | after | best other engine (after) |
|---|---:|---:|---|
| L16 b1 | 1.34× | **1.46×** | llama.cpp q8_0 264.5 |
| L16 b8 | 1.14× | **1.19×** | onnxruntime fp32 583.5 |
| L16 b32 | 1.04× | **1.09×** | openvino fp32 644.1 |
| L128 b1 | 1.02× | **1.07×** | onnxruntime fp32 87.3 |
| L128 b8 | 0.96× | 0.97× | openvino fp32 98.2 |
| L128 b32 | 1.03× | **1.08×** | openvino fp32 93.9 |
| L512 b1 | 0.82× | **0.99×** | openvino fp32 20.6 |
| L512 b8 | 0.91× | **1.00×** | openvino fp32 20.7 |
| L512 b32 | 0.93× | **1.01×** | openvino fp32 20.3 |
| mixed b32 | 1.46× | **1.65×** | llama.cpp q8_0 64.8 |

Peregrine q8_0 (the same engine path; the file reaches f32 panels at load), before → after:
L16 b1 1.27→1.27, b8 1.15→**1.16**, b32 1.05→1.01; L128 b1 1.03→**1.06**, b8 0.99→**1.02**,
b32 1.03→**1.07**; L512 b1 0.88→**0.99**, b8 0.92→**1.00**, b32 0.93→**1.03**; mixed 1.53→**1.60**.

The raw texts/s moved **+21.7 % at L512 b1** (16.76 → 20.40), +10.4 % at b8, +8.5 % at b32,
+11.8 % on the mixed batch, and +3.6 to +4.5 % at 128. The last 1 % of the L512 b1
cell is this host's run-to-run spread: 20.4 against openvino's 20.6 is inside the ±3 % band the
three rounds of that cell span. The 16-token cells lost 2–9 % in one round and gained in the others (their three
rounds span 628–740 texts/s at L16 b32); that is the cell's noise, not a direction.

### 3.2 bge-small-en-v1.5, every row, texts/s (median of three interleaved rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fastembed fp32 | 161.9 | 349.2 | 405.3 | 48.2 | 47.9 | 39.2 | 8.2 | 7.1 | 7.2 | 7.4 |
| fastembed int8 | 196.7 | 284.6 | 371.1 | 46.8 | 47.6 | 47.0 | 8.4 | 8.3 | 8.4 | 8.3 |
| llama.cpp f32 | 147.1 | 309.6 | 326.8 | 41.9 | 40.1 | 40.2 | 9.0 | 8.1 | 8.1 | 40.6 |
| llama.cpp q8_0 | 152.4 | 259.1 | 263.3 | 33.7 | 32.0 | 32.0 | 7.5 | 6.7 | 6.8 | 33.5 |
| onnxruntime fp32 | 165.4 | 351.8 | 376.6 | 49.4 | 48.1 | 39.0 | 8.2 | 7.1 | 7.2 | 7.3 |
| onnxruntime int8 | 168.4 | 278.6 | 364.2 | 46.9 | 47.9 | 46.6 | 8.5 | 8.3 | 8.4 | 8.2 |
| openvino fp32 | 140.8 | 344.1 | 398.7 | 45.6 | 53.4 | 50.0 | 11.1 | 10.8 | 10.8 | 11.1 |
| **peregrine f32** | **198.8** | **377.8** | 386.9 | 46.9 | 49.6 | **50.3** | 10.2 | **10.2** | **10.3** | **53.1** |
| **peregrine q8_0** | **203.1** | **362.7** | **398.2** | **48.1** | 50.1 | **50.5** | **10.1** | **10.3** | **10.2** | **53.1** |
| tei fp32 | 81.8 | 117.0 | 126.1 | 19.2 | 18.8 | 19.4 | 2.2 | 2.2 | 2.1 | 5.9 |

Peregrine f32 ÷ best other engine at cosine ≥ 0.999, before → after:

| cell | before | after | best other engine (after) |
|---|---:|---:|---|
| L16 b1 | 1.26× | 1.20× | onnxruntime fp32 165.4 |
| L16 b8 | 1.04× | **1.07×** | onnxruntime fp32 351.8 |
| L16 b32 | 0.96× | 0.95× | fastembed fp32 405.3 |
| L128 b1 | 0.96× | 0.95× | onnxruntime fp32 49.4 |
| L128 b8 | 0.92× | **0.93×** | openvino fp32 53.4 |
| L128 b32 | 0.98× | **1.01×** | openvino fp32 50.0 |
| L512 b1 | 0.85× | **0.92×** | openvino fp32 11.1 |
| L512 b8 | 0.86× | **0.94×** | openvino fp32 10.8 |
| L512 b32 | 0.90× | **0.95×** | openvino fp32 10.8 |
| mixed b32 | 1.24× | **1.31×** | llama.cpp f32 40.6 |

Peregrine q8_0, before → after: L16 b1 1.26→1.23, b8 0.97→**1.03**, b32 0.96→0.98; L128 b1
0.96→0.97, b8 0.91→**0.94**, b32 0.98→**1.01**; L512 b1 0.83→**0.90**, b8 0.88→**0.95**, b32
0.89→**0.95**; mixed 1.23→**1.31**.

The raw texts/s moved **+8.4 % at L512 b1** (9.40 → 10.19), +10.8 % at b8, +7.2 % at b32, +4.6 %
on the mixed batch, and +2.3 to +3.4 % at 128 b8/b32. bge-small has twice MiniLM's layers, so the
same element-wise saving is a smaller share of a pass whose matmul is already ahead.

### 3.3 The cells still below 1.0×

After, against the best equal-accuracy engine (three decimals):

| model / file | cell | after |
|---|---|---:|
| MiniLM f16 | L128 b8 | 0.970 |
| MiniLM f16 | L512 b1 / b8 | 0.990 / 0.996 |
| MiniLM q8_0 | L512 b1 / b8 | 0.993 / 0.996 |
| bge-small f32 | L16 b32 | 0.955 |
| bge-small f32 | L128 b1 / b8 | 0.950 / 0.928 |
| bge-small f32 | L512 b1 / b8 / b32 | 0.916 / 0.942 / 0.954 |
| bge-small q8_0 | L16 b32 | 0.983 |
| bge-small q8_0 | L128 b1 / b8 | 0.974 / 0.938 |
| bge-small q8_0 | L512 b1 / b8 / b32 | 0.905 / 0.951 / 0.949 |

The remaining bge-small loss is the exact GELU (10.7 ms of a 98.8 ms pass against a fused
approximation) plus the embedding LayerNorm pass; its matmul is 60.9 ms against OpenVINO's 72.1.
MiniLM is level to ahead on every 512-token cell.

### 3.4 The other embedding models did not move

Peregrine's rows at L128 b1 / L512 b1, one run each, before → after (texts/s):

| model | precision | L128 b1 | L512 b1 |
|---|---|---:|---:|
| bge-m3 | q8_0 | 7.50 → 7.36 | 2.56 → 2.59 |
| EmbeddingGemma 300M | q8_0 | 14.03 → 13.68 | 4.79 → 4.73 |
| Qwen3-Embedding 0.6B | f16 | 4.64 → 4.58 | 1.66 → 1.66 |
| Qwen3-Embedding 0.6B | q8_0 | 5.58 → 5.57 | 1.91 → 1.94 |
| Nemotron 3.5 Embed 1B | q8_0 | 4.61 → 4.63 | 1.10 → 1.10 |

Everything is inside the ±5 % band two single runs have on this host. These families do not call
the BERT element-wise block; the softmax kernel is shared with `modern-bert` and
`gemma-embedding`, and their long-text cells would move by a fraction of a millisecond, inside
the band.

### 3.5 Rerank, 8 documents of ~123 tokens

Both files through HTTP on Peregrine alone, `scripts/rerank-bench.py --docs 8 --tokens 123
--reps 5`, two interleaved runs each, medians:

| file | before | after |
|---|---:|---:|
| bge-reranker-v2-m3 (Q8_0) | 3.89 | **3.94** pairs/s |
| bge-reranker-base (Q8_0) | 12.82–13.32 | **13.40–13.43** pairs/s |

Short passes: the element-wise block is a smaller share, so the gain is 1–2 % and inside the
run's spread.

### 3.6 Stage profile after (median of three `cellprof` passes, 4 threads, one text)

| cell | pass ms | matmul | attention | other |
|---|---:|---:|---:|---:|
| MiniLM f16 L128 b1 | 9.34 | 74.4 % | 10.1 % | 15.4 % |
| MiniLM f16 L512 b1 | 49.51 | 62.6 % | 21.7 % | 15.2 % |
| bge-small f32 L128 b1 | 19.16 | 72.8 % | 10.5 % | 16.8 % |
| bge-small f32 L512 b1 | 98.80 | 61.8 % | 21.6 % | 16.2 % |

Before the same runs: 9.79 / 53.39 / 19.71 / 104.40 ms, with other 18.0 / 17.4 / 17.9 / 17.0 %.

## Checks

| check | result |
|---|---|
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| `cargo test --release -p peregrine-kernels` | 168 pass |
| `PEREGRINE_ISA=scalar cargo test --release -p peregrine-kernels` | 168 pass |
| `cargo test --release -p peregrine-core` | pass (all binaries) |
| `cargo test --release -p peregrine-models` (every binary but `golden`) | pass |
| `cargo test --release -p peregrine-models --test golden`, split by model | pass for the 11 models it could load: SmolLM2-360M, Qwen2.5-0.5B, Llama-3.2-1B, gemma-3-1b, Qwen3.5-2B, Llama-2-7B, Mistral-7B (each golden byte-identical) |
| phi-4 and Qwen3-30B-A3B goldens | **not run**: 8.5 GB and 11 GB of weights against 4.4 GB free and a full swap; the test thrashes and the 9-minute bound stops it. The same is true of the before binary. Their golden files are unchanged (`git status`) |
| `PEREGRINE_ISA=scalar cargo test --release -p peregrine-models` (every binary but `golden`; `bert_heads`, `embeddings`, `batch`, `runtime_parity`, `attn_invariance` split out) | pass |
| `cargo test --release -p peregrine-server -p peregrine` | pass |
| `cargo hack check --each-feature --no-dev-deps -p peregrine-kernels -p peregrine-core -p peregrine-models` | pass (42 feature sets) |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace` | pass |
| no file under any `tests/golden/` changed | confirmed (`git status`) |

## Exact commands

```sh
# the worktree (the PR branch is built on origin/s4.9p-bert-gemm, PR #233)
git fetch origin
git worktree add ~/wt-s49r -B s4.9r-encoder-elementwise origin/s4.9p-bert-gemm
git worktree add ~/wt-s49r-before --detach origin/s4.9p-bert-gemm   # the "before" binary

cargo build --release -p peregrine
cargo build --release -p peregrine-bench --example cellprof \
  --features peregrine-models/arch-bert,peregrine-models/arch-gemma-embedding,peregrine-models/arch-qwen,peregrine-models/arch-llama,peregrine-models/arch-mistral3

# the stage profile (before and after)
./target/release/examples/cellprof ~/models/embed/minilm/minilm-f16.gguf \
  /var/tmp/embed-texts-s49l/minilm.json 128,512 1 4
./target/release/examples/cellprof ~/models/embed/bge-small/bge-small-f32.gguf \
  /var/tmp/embed-texts-s49l/bge-small.json 128,512 1 4

# the proof run: three interleaved rounds per binary, MiniLM and bge-small, every engine
for r in 1 2 3; do
  for eng in before after; do
    bin=~/wt-s49r-before/target/release/peregrine
    out=benchmarks/results/bluefin-2026-11-12-s4.9r/raw-before
    [ $eng = after ] && bin=~/wt-s49r/target/release/peregrine
    [ $eng = after ] && out=benchmarks/results/bluefin-2026-11-12-s4.9r/raw-after
    python3 benchmarks/proof_run.py --mode bench --round $r --models minilm,bge-small \
      --peregrine-binary "$bin" --texts-dir /var/tmp/embed-texts-s49l \
      --min-avail-gb 3.5 --out-dir "$out"
  done
done

# the other families: L128 b1 and L512 b1, before and after
python3 benchmarks/embed.py --select bge-m3:peregrine:q8_0 --round 1 --threads 4 \
  --max-load 20 --min-avail-gb 3.5 --port 8402 --texts-dir /var/tmp/embed-texts-s49l \
  --output-dir benchmarks/results/bluefin-2026-11-12-s4.9r/raw-others-after \
  --peregrine-binary ~/wt-s49r/target/release/peregrine \
  --lengths 128,512 --batches 1 --clients 1 --budget 0.5 --ctx-tokens 2048
# …and the same for embeddinggemma:q8_0, qwen3-embed-0.6b:f16, qwen3-embed-0.6b:q8_0,
# nemotron-3.5-embed-1b:q8_0, and for the before binary

# rerank
~/wt-s49r/target/release/peregrine serve -m ~/models/embed/bge-reranker-v2-m3-Q8_0.gguf \
  --threads 4 --ctx 8192 --port 18112
python3 scripts/rerank-bench.py --base-url http://127.0.0.1:18112 \
  --name after-bge-reranker-v2-m3 --docs 8 --tokens 123 --reps 5

# the OpenVINO per-node profile
/var/tmp/embed-venv/bin/python appendix/ovprof.py ~/models/embed/minilm/onnx-fp32.onnx 512
/var/tmp/embed-venv/bin/python appendix/ovprof.py ~/models/embed/bge-small/onnx-fp32.onnx 512
```
