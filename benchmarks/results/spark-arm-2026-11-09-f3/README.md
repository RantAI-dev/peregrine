# aarch64 NEON tier F3: Q6_K on `sdot`

The second follow-up to the NEON tier (S2.8 F3): the Q6_K per-row kernel and the repacked `Q6K8`
tile on aarch64. Branch `neon-4`, stacked on `neon-3` (F1, #129), which is stacked on `neon-2`
(#128). This is the last 42 % of the Llama-3.2-1B Q4_K_M file's matmul bytes.

- Date: 2026-11-09.
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores: 10 × Cortex-X925 (CPUs 5–9,
  15–19) and 10 × Cortex-A725 (CPUs 0–4, 10–14), 121 GB, one NUMA node.
- Peregrine `neon-4` at `452427e` + this step, `release`, `rustc 1.99.1`. Tier
  `aarch64-neon-dotprod`.
- llama.cpp v0.9.0-dev (`0c6dca2`), CPU-only (`CUDA_VISIBLE_DEVICES=""`, `-ngl 0`); the same
  invocations as the F1 run, reused here as context.
- Models and settings as in `../spark-arm-2026-11-09-f1/`: 512-token prompt, 128 greedy decode
  tokens, `peregrine bench`, 5 interleaved rounds, 20 threads (all CPUs) and 10 threads
  (`taskset -c 5-9,15-19`, X925 only).

## What this step changes

| arm | kernels |
|---|---|
| `scalar` | `PEREGRINE_ISA=scalar`: the scalar reference |
| `f1` | the previous PR: Q4_K and Q8_0 tiles, Q5_0 tile and per-row kernels; Q6_K still scalar |
| `native` | the above plus Q6_K: the `Q6K8` tile (GEMV and GEMM) and the per-row `sdot` kernel |

The Q6_K tile is folded exactly like the scalar reference: the run's sixteen-weight integer sum
minus the `32·Σa` offset, scaled by the run's signed eight-bit scale, summed as an integer over the
super-block, then `acc += d · dk · sumi` in the reference's order. Unlike Q5_0's tile (whose fused
`mul_add` moves the last bits), this path is **bit-identical to the scalar kernel**, so no golden
was re-recorded: the Q6_K models are byte-for-byte the same, which the golden test confirms.

## Results

Medians of 5 rounds on the X925 cluster (the clean, reproducible column):

| model | arm | prefill tok/s | decode tok/s | prefill vs scalar | decode vs scalar |
|---|---|---:|---:|---:|---:|
| Llama-3.2-1B | scalar | 83.1 | 57.6 | — | — |
| Llama-3.2-1B | f1 | 203.4 | 68.5 | 2.45× | 1.19× |
| Llama-3.2-1B | **native** | **393.6** | **101.3** | **4.74×** | **1.76×** |
| Qwen2.5-0.5B | scalar | 216.8 | 117.7 | — | — |
| Qwen2.5-0.5B | f1 | 380.9 | 160.6 | 1.76× | 1.36× |
| Qwen2.5-0.5B | **native** | **564.4** | **174.3** | **2.60×** | **1.48×** |

This step against F1, same rounds: Llama prefill **1.94×**, decode **1.48×** (Q6_K is 42 % of its
bytes); Qwen prefill 1.48×, decode 1.09× (Q6_K is 11 % of its bytes).

20 threads, least-contended estimate (best of 10 runs; the host was badly shared during this
measurement — another worker's benchmark held most of the machine in several rounds, so the medians
in `peregrine.csv` are not usable):

| model | arm | prefill tok/s | decode tok/s | prefill vs scalar | decode vs scalar |
|---|---|---:|---:|---:|---:|
| Llama-3.2-1B | scalar | 121.6 | 66.6 | — | — |
| Llama-3.2-1B | f1 | 224.0 | 64.5 | 1.84× | 0.97× |
| Llama-3.2-1B | **native** | **422.7** | **91.1** | **3.48×** | **1.37×** |
| Qwen2.5-0.5B | scalar | 256.4 | 126.8 | — | — |
| Qwen2.5-0.5B | f1 | 434.2 | 163.1 | 1.69× | 1.29× |
| Qwen2.5-0.5B | **native** | **637.1** | **176.2** | **2.48×** | **1.39×** |

The 20-thread mixed pool is still slower per core than the X925-only run, and on this day much
noisier; the X925 medians are what this note stands behind.

## llama.cpp (context)

Best of `-fa 0` and `-fa auto`, CPU-only, same files (from the F1 session):

| model | threads | metric | native | llama.cpp | ratio |
|---|---:|---|---:|---:|---:|
| Llama-3.2-1B | 20 | prompt 512 | 422.7 | 363.9 | 1.16× |
| Llama-3.2-1B | 20 | decode | 91.1 | 99.7 | 0.91× |
| Llama-3.2-1B | 10 (X925) | prompt 512 | 393.6 | 282.5 | 1.39× |
| Llama-3.2-1B | 10 (X925) | decode | 101.3 | 112.4 | 0.90× |
| Qwen2.5-0.5B | 20 | prompt 512 | 637.1 | 706.3 | 0.90× |
| Qwen2.5-0.5B | 20 | decode | 176.2 | 155.3 | 1.13× |
| Qwen2.5-0.5B | 10 (X925) | prompt 512 | 564.4 | 561.2 | 1.01× |
| Qwen2.5-0.5B | 10 (X925) | decode | 174.3 | 206.6 | 0.84× |

The gap that remains is no longer weight formats: on this model pair every matmul format that
dominates the bytes now has an `sdot` kernel. It is the scalar attention, `exp`/softmax, activation
quantisation and RMSNorm on aarch64 (F5/F6), which bounds decode.

## Per-core kernel speed

One thread, synthetic L1/L2-resident data, X925 (`microbench.txt`):

| kernel | scalar | NEON | ratio |
|---|---:|---:|---:|
| Q6_K, one row | 5.8 GMAC/s | 28.5 | 4.9× |
| Q6_K repacked tile, GEMV | 1.0 (tile reference) | 34.9 | 35× |
| Q6_K repacked tile, GEMM ×4 | — | 52.8 | — |
| Q6_K repacked tile, GEMM ×16 | — | 52.8 | — |
| Q5_0 tile, GEMM ×4 (control) | — | 34.6 | — |
| Q4_K tile, GEMM ×4 (control) | — | 75.7 | — |

Q6_K's 52.8 GMAC/s sits between Q4_K (75.7) and Q5_0 (34.6): the six-bit code needs a mask, a
two-bit shift and an OR before each `sdot`, where Q4_K needs a mask or a shift and Q5_0 needs the
fifth-bit spread. A `neon-i8mm` (`smmla`) tile is the follow-up (F2) that would remove the expansion
from the inner loop entirely.

## Batch invariance

- `q6_k_matches_the_scalar_reference_bit_for_bit` (per-row kernel ≡ scalar, `to_bits()`).
- `q6k8_gemv_and_gemm_agree_with_the_reference_at_every_width`: tile GEMV (1–3) ≡ GEMM (4–8) ≡
  `kq::dot_q6k8_ref` at every width, `to_bits()`.
- The same test asserts `dot_q6k8_ref` ≡ `quant::dot_scalar`, so the tile cannot drift from the
  path it replaced — which is also why the goldens do not move.
- `q6k8_matches_the_scalar_on_the_file_bytes_row_by_row`: the tile on the file's bytes against the
  scalar kernel, per row.
- `KqActs` gained the per-16 run sums (`bs16`) the Q6_K kernel needs; the Q4_K kernels still read
  the per-32 sums, and their tests are unchanged.

## Files

- `peregrine.csv`: the 10-thread X925 and the first 20-thread matrix, every run.
- `peregrine-20t.csv`: the second 20-thread matrix (contended; quoted only as best-of).
- `microbench.txt`: `neonbench` output.
- `llama-bench.txt`: the llama.cpp invocations (same as F1).
