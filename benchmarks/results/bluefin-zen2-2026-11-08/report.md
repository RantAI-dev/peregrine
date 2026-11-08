# Benchmark Report

**Date**: 2026-11-08T03:25:14.060205+00:00

## Machine Record

| Property | Value |
|---|---|
| Host / CPU | AMD Ryzen 3 PRO 4350G with Radeon Graphics |
| CPU Features | fma, avx, avx2 |
| Memory Bandwidth Ceiling | **39.27 GB/s** (Peregrine telemetry probe) |
| Compute Ceiling | 478.9 GFLOP/s (Ridge: 12.2) |
| Kernel / OS | 7.2.7-200.fc44.x86_64 (Linux) |
| CPU Governor | powersave |
| Benchmark Threads | 4 |

## Engines Tested

| Engine | Version / Commit | Build Details |
|---|---|---|
| llama.cpp | `v0.9.0` (`cd959ba`) | CMake Release AVX2/FMA CPU |
| peregrine | `0.1.0` (`5f76d488`) | cargo build --release -p peregrine |

## Model: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (SHA-256: `6eb923e7d26e9cea...`)

### Benchmark Results: The Four Numbers

Per `docs/targets.md`: single-user decode tok/s, prompt tok/s, total tok/s at 4 and 10 users, and first-token time at 10 users; medians reported with spread (± half-range).

| Context | Actual Tokens | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |
|---|---|---|---|---|---|---|---|---|
| 512 | 507 | llama.cpp | 74.9 ± 0.3 | 74.8% | 302.9 ± 1.6 | 65.2 ± 0.5 | 68.7 ± 0.2 | 13.650 ± 0.080 s |
| 512 | 509 | peregrine | 52.8 ± 0.1 | 52.7% | 260.0 ± 0.2 | 70.1 ± 0.3 | 71.3 ± 0.4 | 14.880 ± 0.120 s |
| 4096 | 4088 | llama.cpp | 41.0 ± 0.1 | 41.0% | 236.0 ± 0.1 | 10.9 ± 0.0 | 10.1 ± 0.0 | 114.250 ± 4.660 s |
| 4096 | 4089 | peregrine | 45.1 ± 0.5 | 45.1% | 195.2 ± 2.2 | 9.9 ± 0.0 | 9.8 ± 0.0 | 152.010 ± 0.650 s |

### Head-to-Head Comparison

| Context | Metric | Peregrine | Competitor | Speedup | Peregrine % Ceiling |
|---|---|---|---|---|---|
| 512 | Single Decode | 52.8 ± 0.1 | 74.9 ± 0.3 (llama.cpp) | 0.70x | 52.7% |
| 512 | Prompt (Prefill) | 260.0 ± 0.2 | 302.9 ± 1.6 | 0.86x | — |
| 512 | 4 Users Throughput | 70.1 ± 0.3 | 65.2 ± 0.5 | 1.08x | — |
| 512 | 10 Users Throughput | 71.3 ± 0.4 | 68.7 ± 0.2 | 1.04x | — |
| 512 | 10 Users TTFT | 14.880 ± 0.120 s | 13.650 ± 0.080 s | 0.92x faster | — |
| 4096 | Single Decode | 45.1 ± 0.5 | 41.0 ± 0.1 (llama.cpp) | 1.10x | 45.1% |
| 4096 | Prompt (Prefill) | 195.2 ± 2.2 | 236.0 ± 0.1 | 0.83x | — |
| 4096 | 4 Users Throughput | 9.9 ± 0.0 | 10.9 ± 0.0 | 0.91x | — |
| 4096 | 10 Users Throughput | 9.8 ± 0.0 | 10.1 ± 0.0 | 0.96x | — |
| 4096 | 10 Users TTFT | 152.010 ± 0.650 s | 114.250 ± 4.660 s | 0.75x faster | — |

## Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (SHA-256: `6f85a640a97cf2bf...`)

### Benchmark Results: The Four Numbers

Per `docs/targets.md`: single-user decode tok/s, prompt tok/s, total tok/s at 4 and 10 users, and first-token time at 10 users; medians reported with spread (± half-range).

| Context | Actual Tokens | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |
|---|---|---|---|---|---|---|---|---|
| 512 | 509 | llama.cpp | 39.5 ± 0.3 | 79.6% | 188.3 ± 0.6 | 39.5 ± 0.2 | 37.4 ± 0.0 | 22.020 ± 0.050 s |
| 512 | 510 | peregrine | 45.4 ± 0.1 | 91.5% | 196.3 ± 0.6 | 53.5 ± 0.3 | 55.5 ± 0.0 | 18.870 ± 0.090 s |
| 4096 | 4080 | llama.cpp | 13.0 ± 0.1 | 26.2% | 141.2 ± 0.5 | 5.4 | 5.1 ± 0.0 | 185.220 ± 0.180 s |
| 4096 | 4079 | peregrine | 36.7 ± 0.1 | 74.0% | 143.7 ± 0.8 | 7.7 ± 0.0 | 7.7 | 192.590 ± 0.180 s |

### Head-to-Head Comparison

| Context | Metric | Peregrine | Competitor | Speedup | Peregrine % Ceiling |
|---|---|---|---|---|---|
| 512 | Single Decode | 45.4 ± 0.1 | 39.5 ± 0.3 (llama.cpp) | 1.15x | 91.5% |
| 512 | Prompt (Prefill) | 196.3 ± 0.6 | 188.3 ± 0.6 | 1.04x | — |
| 512 | 4 Users Throughput | 53.5 ± 0.3 | 39.5 ± 0.2 | 1.35x | — |
| 512 | 10 Users Throughput | 55.5 ± 0.0 | 37.4 ± 0.0 | 1.49x | — |
| 512 | 10 Users TTFT | 18.870 ± 0.090 s | 22.020 ± 0.050 s | 1.17x faster | — |
| 4096 | Single Decode | 36.7 ± 0.1 | 13.0 ± 0.1 (llama.cpp) | 2.82x | 74.0% |
| 4096 | Prompt (Prefill) | 143.7 ± 0.8 | 141.2 ± 0.5 | 1.02x | — |
| 4096 | 4 Users Throughput | 7.7 ± 0.0 | 5.4 | 1.43x | — |
| 4096 | 10 Users Throughput | 7.7 | 5.1 ± 0.0 | 1.52x | — |
| 4096 | 10 Users TTFT | 192.590 ± 0.180 s | 185.220 ± 0.180 s | 0.96x faster | — |
