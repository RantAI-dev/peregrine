# Benchmark Report

**Date**: 2026-11-07T05:15:51.823304+00:00

## Machine Record

| Property | Value |
|---|---|
| Host / CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M |
| CPU Features | fma, avx, avx2, avx512f, avx_vnni, avx512_bf16 |
| Memory Bandwidth Ceiling | **35.43 GB/s** (Peregrine telemetry probe) |
| Compute Ceiling | 287.9 GFLOP/s (Ridge: 8.1) |
| Kernel / OS | 6.18.33.2-microsoft-standard-WSL2 (Linux) |
| CPU Governor | unknown (VM or fixed frequency) |
| Benchmark Threads | 12 |

## Engines Tested

| Engine | Version / Commit | Build Details |
|---|---|---|
| llama.cpp | `v0.9.0` (`cd959ba`) | CMake Release AVX2/FMA CPU |
| peregrine | `0.1.0` (`ce67e762`) | cargo build --release -p peregrine |

## Model: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (SHA-256: `74a4da8c9fdbcd15...`)

### Benchmark Results: The Four Numbers

Per `docs/targets.md`: single-user decode tok/s, prompt tok/s, total tok/s at 4 and 10 users, and first-token time at 10 users; medians reported with spread (± half-range).

| Context | Actual Tokens | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |
|---|---|---|---|---|---|---|---|---|
| 512 | 510 | llama.cpp | 70.8 ± 1.3 | 78.3% | 419.4 ± 2.0 | 79.6 ± 0.4 | 93.8 ± 0.5 | 11.060 ± 1.270 s |
| 512 | 509 | peregrine | 66.5 ± 1.2 | 73.6% | 271.1 ± 3.6 | 71.1 ± 0.2 | 80.0 ± 0.2 | 8.960 ± 1.560 s |
| 4096 | 4085 | llama.cpp | 46.1 ± 0.7 | 51.1% | 269.9 ± 0.6 | 14.1 ± 0.0 | 13.4 ± 0.0 | 91.030 ± 0.330 s |
| 4096 | 4087 | peregrine | 53.0 ± 0.7 | 58.7% | 200.7 ± 0.3 | 12.1 ± 0.1 | 12.0 ± 0.0 | 121.310 ± 0.400 s |

### Head-to-Head Comparison

| Context | Metric | Peregrine | Competitor | Speedup | Peregrine % Ceiling |
|---|---|---|---|---|---|
| 512 | Single Decode | 66.5 ± 1.2 | 70.8 ± 1.3 (llama.cpp) | 0.94x | 73.6% |
| 512 | Prompt (Prefill) | 271.1 ± 3.6 | 419.4 ± 2.0 | 0.65x | — |
| 512 | 4 Users Throughput | 71.1 ± 0.2 | 79.6 ± 0.4 | 0.89x | — |
| 512 | 10 Users Throughput | 80.0 ± 0.2 | 93.8 ± 0.5 | 0.85x | — |
| 512 | 10 Users TTFT | 8.960 ± 1.560 s | 11.060 ± 1.270 s | 1.23x faster | — |
| 4096 | Single Decode | 53.0 ± 0.7 | 46.1 ± 0.7 (llama.cpp) | 1.15x | 58.7% |
| 4096 | Prompt (Prefill) | 200.7 ± 0.3 | 269.9 ± 0.6 | 0.74x | — |
| 4096 | 4 Users Throughput | 12.1 ± 0.1 | 14.1 ± 0.0 | 0.86x | — |
| 4096 | 10 Users Throughput | 12.0 ± 0.0 | 13.4 ± 0.0 | 0.90x | — |
| 4096 | 10 Users TTFT | 121.310 ± 0.400 s | 91.030 ± 0.330 s | 0.75x faster | — |

## Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (SHA-256: `6f85a640a97cf2bf...`)

### Benchmark Results: The Four Numbers

Per `docs/targets.md`: single-user decode tok/s, prompt tok/s, total tok/s at 4 and 10 users, and first-token time at 10 users; medians reported with spread (± half-range).

| Context | Actual Tokens | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |
|---|---|---|---|---|---|---|---|---|
| 512 | 508 | llama.cpp | 49.4 ± 0.9 | 111.6% | 252.6 ± 8.1 | 53.5 ± 4.9 | 62.5 ± 2.8 | 15.190 ± 4.820 s |
| 512 | 510 | peregrine | 61.4 ± 1.3 | 138.8% | 147.0 ± 1.9 | 51.3 ± 3.4 | 52.0 ± 0.0 | 18.290 ± 0.370 s |
| 4096 | 4077 | llama.cpp | 31.9 ± 0.2 | 72.1% | 187.9 ± 8.1 | 9.2 ± 0.1 | 8.4 ± 0.0 | 142.450 ± 0.290 s |
| 4096 | 4078 | peregrine | 45.5 ± 0.8 | 102.8% | 116.8 ± 0.6 | 7.1 ± 0.0 | 7.0 ± 0.1 | 206.950 ± 2.930 s |

### Head-to-Head Comparison

| Context | Metric | Peregrine | Competitor | Speedup | Peregrine % Ceiling |
|---|---|---|---|---|---|
| 512 | Single Decode | 61.4 ± 1.3 | 49.4 ± 0.9 (llama.cpp) | 1.24x | 138.8% |
| 512 | Prompt (Prefill) | 147.0 ± 1.9 | 252.6 ± 8.1 | 0.58x | — |
| 512 | 4 Users Throughput | 51.3 ± 3.4 | 53.5 ± 4.9 | 0.96x | — |
| 512 | 10 Users Throughput | 52.0 ± 0.0 | 62.5 ± 2.8 | 0.83x | — |
| 512 | 10 Users TTFT | 18.290 ± 0.370 s | 15.190 ± 4.820 s | 0.83x faster | — |
| 4096 | Single Decode | 45.5 ± 0.8 | 31.9 ± 0.2 (llama.cpp) | 1.43x | 102.8% |
| 4096 | Prompt (Prefill) | 116.8 ± 0.6 | 187.9 ± 8.1 | 0.62x | — |
| 4096 | 4 Users Throughput | 7.1 ± 0.0 | 9.2 ± 0.1 | 0.77x | — |
| 4096 | 10 Users Throughput | 7.0 ± 0.1 | 8.4 ± 0.0 | 0.83x | — |
| 4096 | 10 Users TTFT | 206.950 ± 2.930 s | 142.450 ± 0.290 s | 0.69x faster | — |
