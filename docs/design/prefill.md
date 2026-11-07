# Where Peregrine's prefill time goes

Research note for roadmap S3.4 (faster k-quant kernels) and S3.5 (the prompt path).
No code changes were made to the repository; every number below was measured on the
baseline machine with the pinned engines.

Model: `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (807 694 464 B, 16 layers,
d_model 2048, 32 heads / 8 KV heads, head_dim 64, n_ff 8192, n_vocab 128 256).
Engines: Peregrine `5a7691d` (`cargo build --release -p peregrine`) and
llama.cpp **v0.9.0** (`cd959ba`, CMake Release, AVX2/FMA) from the harness cache in
the harness cache.

---

## 0. Executive summary

| | Peregrine | llama.cpp |
|---|---|---|
| Prompt tok/s (512-token prompt, 12 threads) | **154.0** (3.325 s) | **258.1** (1.984 s) |
| Ratio | **0.60×** (baseline said 0.58×) | — |
| Prompt tok/s, 1 thread | 23.3 | 44.7 |
| Ratio at 1 thread | 0.52× | — |

Peregrine's prefill is **85.7 % weight matmul**, 4.7 % attention, 9.3 % a serial
"everything else" bucket, 0.3 % the output head, and 0.08 % fork-join. It moves
1.67 GB/s of weights — 4 % of the machine's 41.7 GB/s DRAM roof — so it is
**not** bandwidth-bound; it is bound by the k-quant kernel's instruction mix.

The single biggest gap is that **llama.cpp's repacked "8x8" Q4_K GEMM gets 64 MACs
out of each `vpmaddubsw`, while Peregrine's `wide_q4_k` gets 32.** Both kernels
issue `vpmaddubsw` at almost the same rate (0.40 vs 0.44 per cycle), so that 2× is
almost exactly the 1.8× per-core matmul gap.

The second gap is Peregrine's serial non-matmul work (9.3 % of the prefill, scales
at 2.07× from 1 to 12 threads).

The third gap is the prefill chunk size: `PREFILL_CHUNK = 64` costs **9-11 %**
(a 16-token chunk measures 167.7-168.8 tok/s against 150.7-156.3 tok/s at 64).

---

## 1. Machine, and how the two engines were pinned to the same thread count

```
CPU      AMD Ryzen AI 9 HX 370 w/ Radeon 890M, 24 logical CPUs (12 cores + SMT)
Kernel   6.18.33.2-microsoft-standard-WSL2
Build    AVX2 + FMA (no AVX-512 in either binary)
Clock    1.74 GHz  (measured: 5 M dependent `vfmadd` chain = 4 cycles each)
```

Ceilings, from Peregrine's own telemetry probe (`target/release/examples/probe 12`):

```
bandwidth 41.679 GB/s      flops 328.582 GFLOP/s (f32 FMA, 12 threads)      ridge 7.884
```

(An earlier run of the same probe reported 35.43 GB/s and 287.9 GFLOP/s; the probe
is the repo's own instrument and its numbers move by ~8 % between runs, so both
should be quoted with that spread.)

Two additional ceilings were measured, because the prefill is an **int8** kernel and
the f32-FMA roof is not the right yardstick (own micro-benchmarks, one core):

```
12 independent 256-bit FMA chains      52.2 GFLOP/s  = 1.87 FMA/cycle
pure vpmaddubsw+vpmaddwd+vpaddd mix     101.8 G int8-MAC/s = 58.5 MAC/cycle
                                          (this is Peregrine's own instruction mix)
```

So the machine's int8 multiply roof is ≈ 1.22 T MAC/s (12 cores), and Peregrine's
Q4_K kernel reaches 148.8 G MAC/s — **12 % of the int8 roof** — while sitting at
90 % of the *f32* FMA roof. Quote the int8 number, not the f32 one, for prefill.

### Thread pinning (a reproducibility trap)

`peregrine bench` takes no thread flag: `SessionCfg::default()` sets
`threads = std::thread::available_parallelism()`, i.e. **24** on this box. The
benchmark harness passes `RAYON_NUM_THREADS` to the Peregrine CLI
(`benchmarks/engines/peregrine.py:170`) but Peregrine uses its own pool and never
reads it — measured: without pinning, `peregrine bench` reports 196.7 tok/s, which
is the 24-thread number.

So both engines were pinned with `taskset` to the *same* CPU set for every run:

| threads | peregrine | llama.cpp |
|---|---|---|
| 6 | `taskset -c 0-5` | `taskset -c 0-5 … -t 6` |
| 12 | `taskset -c 0-11` | `taskset -c 0-11 … -t 12` |
| 24 | `taskset -c 0-23` | `taskset -c 0-23 … -t 24` |

CPUs 0-11 are the 12 physical cores; 12-23 are their SMT siblings.

---

## 2. Q1 — prompt tok/s, both engines, measured the same way, 3× alternating

Commands (alternated A, B, A, B, …):

```sh
taskset -c 0-11 ./target/release/peregrine bench \
    -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --prompt-tokens 512 --tokens 1 --json --no-profile

taskset -c 0-11 ~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench \
    -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 512 -n 0 -t 12 -r 3 -o json
```

Round-by-round (tok/s):

| round | peregrine | llama.cpp (llama-bench avg ± sd) |
|---|---|---|
| 1 | 149.89 | 260.88 ± 5.09 |
| 2 | 148.78 | 256.85 ± 3.18 |
| 3 | 154.19 | 259.18 ± 1.50 |

**Median 149.9 vs 259.2 tok/s → 0.578×** (the baseline's 0.58× reproduces).

Pooling every clean sample taken during the session (llama-bench samples, not the
process average): peregrine 149.9-156.5 tok/s, llama.cpp 255.4-260.9 tok/s.
Final medians: **peregrine 154.0 tok/s (3.325 s), llama.cpp 258.1 tok/s (1.984 s)
→ 0.60×**.

Two of the `llama-bench` runs produced one corrupt sample (`2.8e-08 tok/s`, i.e.
a broken timer read; the other two samples in that run were normal at 255.7 and
257.8). All other samples were tight (sd ≤ 5 tok/s).

### Thread sweep (both engines, same prompt)

| threads | peregrine tok/s | llama.cpp tok/s | ratio | peregrine scaling | llama.cpp scaling |
|---|---|---|---|---|---|
| 1 | 23.3 | 44.7 | 0.52 | 1.00× | 1.00× |
| 2 | 39.9 | 76.7 | 0.52 | 1.71× | 1.72× |
| 4 | 67.2 | 110.1 | 0.61 | 2.88× | 2.46× |
| 6 | 87.2 | 149.0 | 0.585 | 3.74× | 3.33× |
| 12 | 154.0 | 258.1 | 0.60 | 6.61× | 5.77× |
| 24 | 204.3 | 384.2 | 0.53 | 8.77× | 8.59× |

**The ratio is flat (0.52-0.60) across the whole range, so this is a per-core
kernel-efficiency gap, not a scheduling or parallelism gap.** Both engines scale
the same way; Peregrine simply does less work per cycle.

### Prompt-length sweep (512-token model, 12 threads)

| prompt | peregrine tok/s | peregrine s | llama.cpp tok/s | llama.cpp s |
|---|---|---|---|---|
| 256 | 154.2 | 1.661 | 262.0 | 0.977 |
| 512 | 154.0 | 3.325 | 256.3 | 1.998 |
| 1024 | 145.2 | 7.052 | ~241.5 | 4.240 |
| 2048 | 135.5 | 15.118 | 212.9 | 9.621 |

Peregrine is flat from 256 to 512 (both are 64-token chunks, so the per-chunk
weight pass is amortised over the same 64 rows) and then degrades slowly as
attention grows. llama.cpp is flat to 512 too (one 512-token ubatch) and then
degrades faster, because it re-reads the weights for every ubatch beyond 512.

---

## 3. Q2 — Peregrine's prefill split, by phase

`StepStats` (crates/peregrine-core/src/state.rs:12) has no prefill profiler in the
CLI, so a small external harness (outside the repo) drove
`peregrine_core::model::Model::forward` exactly the way `Session::bench` does
(`crates/peregrine-core/src/session.rs:961-987`: chunks of `PREFILL_CHUNK`, logits
only on the last chunk) and printed the telemetry.

12 threads, 512-token prompt, chunk 64:

```
wall             : 3.4072 s  (150.3 tok/s)
ns_total         : 3.4072 s
ns_matmul        : 2.9214 s  (85.7 %)   498.96 G int8 MACs -> 170.8 G MAC/s
                                       = 341.6 GFLOP/s (2 FLOP per MAC)
                                       1.67 GB/s of weight traffic
ns_head          : 0.0091 s  ( 0.3 %)   the 128 256-row Q6_K head (tied embedding)
ns_attn          : 0.1612 s  ( 4.7 %)   8.61 G MACs -> 106.8 GFLOP/s
ns_other         : 0.3155 s  ( 9.3 %)   norms, rope, residuals, SwiGLU,
                                       activation quantisation, KV writes
barriers         : 1153
bytes_weights    : 4891.4 MB            bytes_kv: 209.5 MB
```

### Threading overhead (barriers / idle)

* **Barriers:** 1153 fork-joins for the whole prefill (144 per 64-token chunk).
  An empty `pool.run` costs **2.37 µs** (measured over 1000 runs), so the whole
  fork-join bill is **2.7 ms = 0.08 %** of the prefill. Barrier cost is *not* a gap.
* **Idle / imbalance:** with the static row split
  (`PEREGRINE_DYNAMIC=0`) the prefill takes **3.962 s instead of 3.407 s** — the
  dynamic row claiming is worth **16 %**. On this hybrid P-core/Zen-5c machine a
  static split makes every matmul wait for the slowest core, so "idle" here is
  load imbalance, not barrier overhead: **up to 0.55 s (16 %) of the prefill.**
* The "other" bucket barely scales: 0.6505 s at 1 thread → 0.3150 s at 12 threads
  (2.07× over 12 threads). **It is effectively serial**, and at 12 threads it is
  9.4 % of the prefill (it would be 1.6 % if it scaled).

### Matmul by format

The layer projections are only two formats (from the GGUF):

| format | tensors | weight bytes | where |
|---|---|---|---|
| Q4_K | 96 | 467.1 MB | attn_q, attn_k, attn_v (8 of 16 layers), attn_output, ffn_gate, ffn_up, ffn_down (8 of 16 layers) |
| Q6_K | 16 | 117.0 MB | attn_v (8 layers), ffn_down (8 layers) |
| F32 | 34 | 0.3 MB | the norms — not matmuls |

(The only other Q6_K tensor is `token_embd.weight`, 215.5 MB, which is also the
tied output head and is run as a **one-row GEMV** at the end of the prefill — that
is `ns_head`, 0.3 % of the time.)

Per-format time inside `ns_matmul` was not directly observable (the telemetry has
one `ns_matmul` counter), so it was attributed by measuring each format's kernel
on the **real weight rows of the model** at the prefill batch width, and projecting:

```
real rows, batch width 64, 1 thread:   Q4_K  k=2048: 4 446-4 580 ns/row Q4_K  k=8192: 18 411-18 325 ns/row
                                       Q6_K  k=2048: 4 699 ns/row
                                       Q6_K  k=8192: 16 670-16 756 ns/row
projected over all tensors x 8 chunks:  Q4_K 14.47 s core-time (86.1 %)
                                       Q6_K  2.34 s core-time (13.9 %)
```

Both the 1-thread and the 12-thread projections agree on the split
(Q4_K 86.1 % / Q6_K 13.9 % and Q4_K 86.5 % / Q6_K 13.5 %). Scaled to the measured
`ns_matmul` of 2.9214 s:

| phase | time | % of the 3.407 s prefill |
|---|---|---|
| Q4_K matmuls | **2.53 s** | **74 %** |
| Q6_K matmuls | **0.39 s** | 11.6 % |
| attention | 0.161 s | 4.7 % |
| norms/rope/SwiGLU/activation-quant/KV-writes (serial) | 0.316 s | 9.3 % |
| output head (Q6_K GEMV) | 0.009 s | 0.3 % |
| fork-join barriers | 0.003 s | 0.08 % |

(Caveat: the two format rows are a projection from measured per-format kernel
rates, not a direct counter. The phase rows above them — matmul / head / attention
/ other — are exact.)

---

## 4. Q3 — which kernel path prefill uses for Q4_K, at what width, and how fast

### The path

```
blocks/linear.rs:141   MatPath::PerRow  ->  mk.row(j.w.kind, row, acts, &mut tmp)
kernels/table.rs:81    MatKernels::path(Q4_K, tiled=false, ...) == MatPath::PerRow
kernels/quant.rs:1147  quant::dot_batch(fmt, w, acts, out)
kernels/simd.rs:1603   simd::dot_batch:
                           Q4_K if acts.len() > TILE(4) && acts.len() <= MAX_WIDE(128)
                                 && wide_enabled()   ->   wide_q4_k(w, acts, out)
kernels/simd.rs:1392   wide_q4_k: the "wide" batched k-quant kernel
```

* **Batch width = 64** (`PREFILL_CHUNK`, crates/peregrine-core/src/serve.rs:40).
  512 prompt tokens ⇒ 8 forward passes of 64 rows each.
* Inside `wide_q4_k` the activations are still processed **4 at a time**
  (`TILE = 4`, simd.rs:472) — the wide kernel's *group* is 4, its *batch* is 64.
* `PEREGRINE_NO_WIDE=1` forces the tiled path (`tile_q4_k`, which re-decodes the
  weight for every group of 4). Measured: **3.543 s vs 3.407 s — only 3.9 % slower.**
  **So the weight decode is not the bottleneck**; the per-activation work is.
* `PEREGRINE_PF=0` (no software prefetch): 3.351 s, i.e. 1.6 % *faster* than the
  default — the prefetcher is not earning its keep either.

### What `wide_q4_k` does per output row (k = 2048, 64 activations)

Per 144-byte Q4_K super-block (256 weights):

* once per super-block: unpack the 8 scales + 8 mins (`unpack_q4k_scales`,
  `q4k_mins16`) and split the nibbles into low/high (`and` / `srli` on 4×32 B) —
  **12 ops, amortised over all 64 activations**;
* per activation: 8 × `vpmaddubsw` (32 MACs each) + 8 × `vpmaddwd` + 8 × `vpaddd`
  + 16 × 32-byte activation loads + `q4k_fold` (7 float/int ops: the min term's
  `vpmaddwd`, two `vcvtdq2ps`, a `vmulps`, a `vfnmaddps`, a `vfmaddps`) + a
  load/store of the float accumulator, which lives in a **128-entry YMM stack
  array** (`acc`, simd.rs:1397) because 64 live accumulators do not fit in
  16 YMM registers.

### Effective rate

| | value |
|---|---|
| whole prefill | 507.09 G int8 MACs in 3.407 s = **148.8 G MAC/s = 297.7 GFLOP/s** (2 FLOP/MAC) |
| matmul only | 498.96 G MACs in 2.921 s = **170.8 G MAC/s = 341.6 GFLOP/s** |
| per core (1 thread) | 24.6 G MAC/s = 14.1 MAC/cycle = **0.44 vpmaddubsw/cycle** |
| per core (12 threads) | 14.2 G MAC/s = 8.2 MAC/cycle |
| vs the machine's int8 multiply roof (101.8 G MAC/s/core) | **24 % single-thread, 12 % at 12 threads** |
| vs the probe's f32 FMA ceiling (328.6 GFLOP/s) | 90.6 % (the wrong yardstick for an int8 kernel) |
| weight bandwidth used | 1.67 GB/s = **4 % of the 41.7 GB/s DRAM roof** |
| arithmetic intensity | 1014 GFLOP / 4.9 GB = 207 FLOP/B (the roof's ridge is 7.9) |

So: **the prefill is compute/issue-bound, not bandwidth-bound.** The kernel
executes `vpmaddubsw` at 0.44/cycle where the same instruction mix with no memory
traffic sustains 1.83/cycle — the loads, the per-activation float fold and the
accumulator spill eat ~4×.

### The chunk-size lever (relevant to S3.5)

Same run, only `PREFILL_CHUNK` changed (12 threads, 512-token prompt):

| chunk | wall | tok/s | weight bytes read |
|---|---|---|---|
| 8 | 3.202-3.217 s | 159.2-159.9 | 38 000 MB |
| **16** | **3.034-3.054 s** | **167.7-168.8** | 18 917 MB |
| 32 | 3.041-3.140 s | 163.1-168.4 | 9 567 MB |
| 64 (default) | 3.275-3.398 s | 150.7-156.3 | 4 891 MB |
| 128 | 3.640 s | 140.7 | 2 554 MB |

**A 16-token chunk is 9-11 % faster than the default 64**, even though it reads the
weights four times as often (18.9 GB vs 4.9 GB) — the extra traffic is free at
1.7-6 GB/s against a 41.7 GB/s roof, while the smaller activation array
(16 × k bytes = 32 KB for k=2048, 128 KB at 64) stays in L1/L2 and the prefetcher
sees 16 streams instead of 64.

---

## 5. Q4 — the same split for llama.cpp

**No per-kernel split is available on this box.** `perf` is not installed
(`/usr/bin/perf` absent, `linux-tools` not installed, `perf_event_paranoid = 2`),
`gdb` is not installed, and llama.cpp v0.9.0 has no phase profiler of its own
(`GGML_SCHED` is a graph scheduler, not a profiler; the server's `timings` block
only splits prompt-eval from token-gen). So llama.cpp's split is reported as its
total plus a length-sweep decomposition.

Measured totals (12 threads, 512-token prompt): **1.984 s = 258.1 tok/s**.

Derived rates for the same 507 G MACs:

| | llama.cpp | Peregrine |
|---|---|---|
| prompt time | 1.984 s | 3.325 s |
| int8 MAC/s | 256.6 G | 148.8 G |
| GFLOP/s (2/MAC) | 513.2 | 297.7 |
| weight bytes read | 800 MB (read **once**) | 4 891 MB (read **8×**) |
| weight bandwidth | 0.40 GB/s | 1.67 GB/s |

llama.cpp processes the whole 512-token prompt as **one ubatch** (`-ub 512`, the
default), so it reads the weights once; Peregrine's 64-token chunks read them
eight times. Neither is close to the DRAM roof, so this is not why llama.cpp is
faster — but it is a free win for Peregrine.

**Length-sweep decomposition** (t(p) = A·p + B·p², from the 256/512/1024/2048
points): a linear + quadratic fit gives, at p = 512, **≈ 96 % linear**
(matmul + norms/rope/activation quantisation) and **≈ 4 % quadratic**
(attention) for llama.cpp — the same shape as Peregrine's measured 85.7 % matmul
+ 9.3 % other + 4.7 % attention.

---

## 6. Q5 — the gaps, biggest first

### Gap 1 — the k-quant GEMM extracts half the MACs per instruction (≈ 76 % of the prefill)

Peregrine's matmul is 2.92 s; llama.cpp's *entire* prompt is 1.98 s. Per core:
Peregrine 24.6 G int8 MAC/s, llama.cpp ≈ 40 G MAC/s (its 1-thread prompt rate is
44.7 tok/s = 44.3 G MAC/s in total, of which ~90 % is matmul).

The reason is visible in the two kernels' instruction mixes. In llama.cpp's
repacked AVX2 Q4_K GEMM (`ggml_gemm_q4_K_8x8_q8_K`,
`ggml/src/ggml-cpu/arch/x86/repack.cpp:2042`, AVX2 branch at 2816) the inner loop
is:

* `for (; y < anr/4; y += 4)` — 4 activation rows,
* `for (x = 0; x < nc/8; x++)` — 8 output columns,
* `for (b = 0; b < nb; b++)` — super-blocks, and inside it `for (sb = 0; sb < QK_K/64; sb++)`
  with **32 `vpmaddubsw` per (super-block, 8 columns, 4 rows)**, covering
  8 columns × 4 rows × 64 weights = **2048 MACs** ⇒ **64 MACs per `vpmaddubsw`**.
  The trick is the `_sp1`/`_sp2` shuffle pair (`_mm256_shuffle_epi32(..., 160/245)`
  on both operands): each 32-byte weight vector is duplicated so that *one*
  `vpmaddubsw` feeds two different (column, row) accumulator groups. The int16
  products are then summed four at a time (`_mm256_add_epi16` ×3) and the
  per-sub-block scale is applied **once per super-block** with a single
  `_mm256_madd_epi16` (repack.cpp:3104); the min term is one `_mm256_madd_epi16`
  against precomputed per-row `bsums` (repack.cpp:3139).

Peregrine's `wide_q4_k` does **32 MACs per `vpmaddubsw`** (one maddubs per
32-weight sub-block per activation) and pays a full float fold
(`q4k_fold`, simd.rs:362) per super-block **per activation**, with the
accumulator spilled to a 128-entry stack array.

Both kernels issue `vpmaddubsw` at nearly the same rate —

| | MACs per `vpmaddubsw` | `vpmaddubsw`/cycle/core | MAC/cycle/core |
|---|---|---|---|
| Peregrine `wide_q4_k` | 32 | 0.44 | 14.1 |
| llama.cpp `ggml_gemm_q4_K_8x8_q8_K` | 64 | 0.40 (derived) | ≈ 23-26 |

— so the 2× MACs-per-instruction is essentially the whole 1.8× per-core gap.
**This is the S3.4 lever:** restructure the k-quant kernels to llama.cpp's "8x8"
form — repack the weights into 8-row tiles at load (Peregrine already has a tile
store for PQ2_0, `crates/peregrine-core/src/model.rs:171`), interleave 4
activation rows into one buffer, and let a single `vpmaddubsw` feed two
accumulator groups. A 2× improvement in the matmul would take the prompt from
154 tok/s to ≈ 265 tok/s and close the gap entirely.

### Gap 2 — the serial non-matmul bucket (9.3 % of the prefill, and it does not scale)

`ns_other` is 0.3155 s at 12 threads but 0.6505 s at 1 thread: it scales 2.07× over
12 threads, i.e. it is a serial section of the prefill (rmsnorm, rope, the
residual adds, SwiGLU, the activation quantisation and the KV writes all run on
one thread over the whole 64-row batch). At 12 threads it is 9.4 % of the prefill;
perfectly parallel it would be 1.6 %. llama.cpp keeps the equivalent work inside
its ≈ 4 % non-matmul share, because it quantises the activations **once per
ubatch** into a shared interleaved buffer (`ggml_quantize_mat_t<INTER_SIZE,
PARAM_TYPE>`, repack.cpp:4700) and keeps the f32 copy for the norm/rope/residual
path, whereas Peregrine re-quantises a fresh `QuantAct` for every matmul group
(`st.acts[b].fill(&st.xb[...])` per layer, `crates/peregrine-models/src/layer.rs`).
**Fix for S3.5:** quantise once per layer (or once per ubatch) into a shared
interleaved buffer and parallelise the norm/rope/residual pass.

### Gap 3 — the prefill chunk size (≈ 9-11 % recoverable today)

`PREFILL_CHUNK = 64` (serve.rs:40) forces eight passes over the weights
(4 891 MB instead of 800 MB) and, more importantly, makes the activation array
128 KB (512 KB for `ffn_down`, k = 8192), which no longer sits in L1/L2. A
16-token chunk measures **167.7-168.8 tok/s against 150.7-156.3 tok/s** at 64
(§4). The cost is not bandwidth — at 18.9 GB / 3.04 s = 6.2 GB/s the machine is
at 15 % of its roof — it is activation locality and prefetcher stream count.
**Fix for S3.5:** make the prefill chunk size a function of the model's activation
size (and of the number of concurrent slots, which is why it is currently a
constant), or make the kernel's activation layout stream-friendly.

### Gap 4 — thread scheduling: load imbalance, not barriers (up to 16 %)

The fork-join itself costs 2.7 ms (0.08 %). But the static row split
(`PEREGRINE_DYNAMIC=0`) costs 16 % (3.962 s vs 3.407 s) on this hybrid
P-core/Zen-5c machine, because the slowest core sets the pace. Any future work
that adds barriers or changes the chunk granularity should keep the dynamic row
claiming, and should be measured with `PEREGRINE_DYNAMIC=0` as the control.

### Not a gap, but worth recording

* The weight decode is *not* the bottleneck: `PEREGRINE_NO_WIDE=1` (which re-decodes
  every super-block for every group of 4 activations) costs only 3.9 %.
* Software prefetch is not helping: `PEREGRINE_PF=0` is 1.6 % *faster*.
* The prefill is not bandwidth-bound: 1.67 GB/s against a 41.7 GB/s roof.
* In-situ format comparison (same Qwen2.5-0.5B model, two quantisations, 512-token
  prefill, 12 threads): the Q4_K_M/Q5_K/Q6_K/Q8_0 mix takes 2.031 s (matmul
  1.562 s, 242.0 GFLOP/s) and the all-Q8_0 version takes 1.836 s (matmul 1.371 s,
  275.8 GFLOP/s) — **the k-quant path is 14 % slower per MAC than Q8_0's**, and it
  moves 33 % fewer bytes.

---

## 7. Method notes, tools, and what could not be measured

* **perf is unavailable** under WSL2 on this host (`perf` binary absent, no
  `linux-tools` package, `perf_event_paranoid = 2`, no `gdb`, no `valgrind`), so
  every per-phase number comes from Peregrine's own `StepStats`
  (`ns_matmul` / `ns_head` / `ns_attn` / `ns_total` / `barriers`) driven by an
  external harness that calls `Model::forward` the same way `Session::bench` does.
  llama.cpp could only be measured as a total.
* `PEREGRINE_PROF=1` exists (options.rs:116) but only the DSpark drafter reads a
  `prof` flag (dspark.rs:850); there is no dense-model phase printer in the CLI,
  and the server's `phases_ms()` (serve.rs:988) reports decode phases only.
* The external harness lives in `/tmp/opencode/prefill-probe` (its own cargo
  project, path-dependencies on `peregrine-core` / `peregrine-models`); the
  repository was not modified (`git status` clean).
* The repo was left on `5a7691d` (`origin/main`, detached) with the release build
  refreshed; no source file was changed.

### Full command list

```sh
# build
cargo build --release -p peregrine
cargo build --release -p peregrine-bench --example probe

# machine ceilings
./target/release/examples/probe 12
/tmp/opencode/prefill-probe/target/release/peak          # clock, FMA and int8 roofs

# the A/B (alternating, 3x each)
taskset -c 0-11 ./target/release/peregrine bench -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
        --prompt-tokens 512 --tokens 1 --json --no-profile
taskset -c 0-11 ~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench \
        -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 512 -n 0 -t 12 -r 3 -o json

# phase split (12 threads, chunk 64; also 8/16/32/128)
taskset -c 0-11 /tmp/opencode/prefill-probe/target/release/prefill-probe phase \
        models/Llama-3.2-1B-Instruct-Q4_K_M.gguf 12 512 64

# kernel micro-benchmarks (one core, real and synthetic rows)
taskset -c 0 /tmp/opencode/prefill-probe/target/release/prefill-probe kernel 2048 4096 5 1,4,8,16,32,64,128
taskset -c 0-11 /tmp/opencode/prefill-probe/target/release/split \
        models/Llama-3.2-1B-Instruct-Q4_K_M.gguf 64 8 12 3

# option A/Bs
PEREGRINE_NO_WIDE=1 / PEREGRINE_DYNAMIC=0 / PEREGRINE_PF=0 ... prefill-probe phase ...
```

### Raw phase numbers used in §3 (12 threads)

| option | wall | ns_matmul | ns_attn | ns_other | tok/s |
|---|---|---|---|---|---|
| default | 3.4072 s | 2.9214 s | 0.1612 s | 0.3155 s | 150.3 |
| `PEREGRINE_NO_WIDE=1` | 3.5428 s | 3.0464 s | 0.1569 s | 0.3308 s | 144.5 |
| `PEREGRINE_DYNAMIC=0` | 3.9620 s | 3.3507 s | 0.1588 s | 0.4458 s | 129.2 |
| `PEREGRINE_PF=0` | 3.3510 s | 2.8859 s | 0.1572 s | 0.2994 s | 152.8 |

### Peregrine prefill thread scaling (phase probe, 512-token prompt, chunk 64)

| threads | wall | ns_matmul | matmul GFLOP/s | ns_attn | ns_other | scaling vs 1 |
|---|---|---|---|---|---|---|
| 1 | 21.978 s | 20.261 s | 50.1 | 1.020 s | 0.651 s | 1.00× |
| 2 | 12.831 s | 11.729 s | 86.5 | 0.598 s | 0.477 s | 1.73× |
| 4 | 7.619 s | 6.831 s | 148.5 | 0.389 s | 0.379 s | 2.97× |
| 6 | 6.051 s | 5.369 s | 188.9 | 0.323 s | 0.344 s | 3.77× |
| 12 | 3.334 s | 2.850 s | 355.9 | 0.160 s | 0.315 s | 7.11× |

(12 threads is 59 % efficient against the 1-thread rate; the "other" column is the
one that barely moves — 0.651 s → 0.315 s.)
