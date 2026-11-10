# aarch64 NEON tier F1: Q5_0 on `sdot`

The first follow-up to the NEON tier (S2.8 F1): the repacked `Q5_08` tile and the per-row Q5_0
kernel on aarch64. This is the branch `neon-3`, stacked on `neon-2` (`origin/neon-2`, #128) because
#128 had not reached `main` on the day it was measured. The parent tier's numbers are in
`../spark-arm-2026-11-09/`.

- Date: 2026-11-09.
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters: 10 × Cortex-X925
  (3.9 GHz max, CPUs 5–9 and 15–19) and 10 × Cortex-A725 (2.8 GHz max, CPUs 0–4 and 10–14),
  121 GB unified memory, one NUMA node. `asimd`, `asimddp`, `i8mm`, `sve`, `sve2`, `bf16`.
- Peregrine: `neon-3` on `origin/neon-2`, `release` profile, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
  Kernel tier `aarch64-neon-dotprod`.
- llama.cpp: v0.9.0-dev (build 1, commit `0c6dca2`), GNU 13.3.0, Linux aarch64, CUDA build, CPU
  runs only (`CUDA_VISIBLE_DEVICES=""`, `-ngl 0`).
- Models, under `/var/tmp/pg/models`:
  - `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807694464 bytes,
    `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`;
  - `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, 397808192 bytes,
    `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`.
- Settings: `peregrine bench --prompt-tokens 512 --tokens 128`, 5 rounds per arm, arms interleaved
  round by round. llama.cpp: `llama-bench -p 512 -n 128 -r 3 -t <threads> -ngl 0`, both `-fa 0` and
  `-fa auto`; the better of the two per metric is reported (as the parent run did).
- 20 threads = all CPUs; 10 threads = `taskset -c 5-9,15-19` (X925 only).

## What this step changes

| arm | kernels |
|---|---|
| `scalar` | `PEREGRINE_ISA=scalar`: every dot product is `quant.rs`'s scalar reference |
| `native` | Q4_K, Q8_0 and now **Q5_0** on `sdot`: the repacked `Q5_08` tile (GEMV for 1–3 activation rows, GEMM for 4 or more), the per-row kernel and the eight-row decode kernel |

Qwen2.5-0.5B Q4_K_M is 44 % Q5_0 by weight bytes, so it is the model this step is for. Llama-3.2-1B
has no Q5_0: its row is the regression check (Q4_K is untouched, Q6_K is still scalar here).

## Results

Medians of 5 interleaved rounds. The ratio is the median of the per-round ratios; the range is in
the round-by-round list under the table. Absolute numbers are indicative: two other workers share
this host, and two Llama rounds were visibly hit (prefill 66 tok/s against ~210); the medians and
the clean-round ratios are stable.

| model | threads | arm | prefill tok/s | decode tok/s | prefill vs scalar | decode vs scalar |
|---|---:|---|---:|---:|---:|---:|
| Llama-3.2-1B | 20 | scalar | 124.3 | 67.4 | — | — |
| Llama-3.2-1B | 20 | **native** | **210.9** | **73.4** | **1.69×** | **1.09×** |
| Llama-3.2-1B | 10 (X925) | scalar | 94.7 | 57.5 | — | — |
| Llama-3.2-1B | 10 (X925) | **native** | **205.3** | **69.7** | **2.16×** | **1.21×** |
| Qwen2.5-0.5B | 20 | scalar | 251.4 | 118.0 | — | — |
| Qwen2.5-0.5B | 20 | **native** | **434.3** | **154.2** | **1.71×** | **1.29×** |
| Qwen2.5-0.5B | 10 (X925) | scalar | 218.3 | 118.3 | — | — |
| Qwen2.5-0.5B | 10 (X925) | **native** | **382.3** | **162.1** | **1.76×** | **1.37×** |

Per-round ratios (native ÷ scalar, same round), 20 threads / X925 10 threads:

- Llama prefill: 1.13, 0.98, 1.96, 1.69, 1.89 / 2.16, 2.20, 2.16, 2.16, 2.17.
- Llama decode: 1.15, 2.47, 1.08, 1.07, 1.09 / 1.40, 1.21, 1.21, 1.21, 1.21.
- Qwen prefill: 1.63, 1.67, 1.75, 1.86, 1.71 / 1.76, 1.75, 1.76, 1.77, 1.75.
- Qwen decode: 1.11, 1.25, 1.38, 1.30, 1.29 / 1.36, 1.37, 1.37, 1.37, 1.38.

Against the parent tier on the same host (`../spark-arm-2026-11-09`), Qwen's 20-thread prefill goes
268 → 434 tok/s and decode 132 → 154; Llama is unchanged within noise (its Q5_0 share is zero).
The 20-thread mixed pool is still worse per core than the 10-thread X925 run (210.9/205.3), which
is the scheduling point S2.8 open question 2 records, not this step.

## llama.cpp (context)

Best of the two flash-attention settings, same files, threads and affinity:

| model | threads | metric | peregrine native | llama.cpp | ratio |
|---|---:|---|---:|---:|---:|
| Llama-3.2-1B | 20 | prompt 512 | 210.9 | 363.9 | 0.58× |
| Llama-3.2-1B | 20 | decode | 73.4 | 99.7 | 0.74× |
| Llama-3.2-1B | 10 (X925) | prompt 512 | 205.3 | 282.5 | 0.73× |
| Llama-3.2-1B | 10 (X925) | decode | 69.7 | 112.4 | 0.62× |
| Qwen2.5-0.5B | 20 | prompt 512 | 434.3 | 706.3 | 0.61× |
| Qwen2.5-0.5B | 20 | decode | 154.2 | 155.3 | 0.99× |
| Qwen2.5-0.5B | 10 (X925) | prompt 512 | 382.3 | 561.2 | 0.68× |
| Qwen2.5-0.5B | 10 (X925) | decode | 162.1 | 206.6 | 0.78× |

Qwen prefill closed from 0.37× to 0.61× and decode from 0.93× to 0.99× at 20 threads; Llama is the
Q6_K follow-up (F3), which is still scalar on this branch.

## Per-core kernel speed

One thread, synthetic L1/L2-resident data, X925 cluster (`microbench.txt`):

| kernel | scalar | NEON | ratio |
|---|---:|---:|---:|
| Q5_0, one row | 10.9 GMAC/s | 18.9 | 1.7× |
| Q5_0, eight rows (decode) | 10.9 | 22.7 | 2.1× |
| Q5_0 repacked tile, GEMV | 2.5 (tile reference) | 22.1 | 8.8× |
| Q5_0 repacked tile, GEMM ×4 | — | 36.0 | — |
| Q4_K repacked tile, GEMM ×4 (control) | — | 75.7 | — |

The Q5_0 tile's 36 GMAC/s against Q4_K's 76 is the fifth-bit expansion: each window needs the two
`qh` bits spread into `0x10`/`0` bytes before the `sdot`, where Q4_K's nibble pair is a mask and a
shift. A `neon-i8mm` (`smmla`) tile is the follow-up that would remove that overhead (F2).

## Files

- `peregrine.csv`: every `peregrine bench` run (model, arm, affinity, threads, round, tok/s).
- `microbench.txt`: `neonbench` output.
- `llama-bench.txt`: every llama.cpp invocation.
