# Benchmark Report

> **Read this with the README.** This report compares Peregrine against llama.cpp with `-fa off`
> only. That is the faster setting for single-user decode, so those rows are the published ones;
> for prompt processing and multi-user throughput llama.cpp's default (`-fa auto`) is faster, and
> the published ratios for those rows are in [`README.md`](README.md), not here. Rows showing
> `0.0` for 4096-token, 10-user `-fa off` mean *not measured*: that phase does not finish within
> the 10-minute command cap on this host. The README explains why and uses the default for those
> two rows.

**Date**: 2026-11-08T06:47:55.346518+00:00

## Machine Record

| Property | Value |
|---|---|
| Host / CPU | AMD Ryzen 3 PRO 4350G with Radeon Graphics |
| CPU Features | fma, avx, avx2 |
| Memory Bandwidth Ceiling | **39.25 GB/s** (Peregrine telemetry probe) |
| Compute Ceiling | 513.3 GFLOP/s (Ridge: 13.1) |
| Kernel / OS | 7.2.7-200.fc44.x86_64 (Linux) |
| CPU Governor | powersave |
| Benchmark Threads | 4 |

## Engines Tested

| Engine | Version / Commit | Build Details |
|---|---|---|
| llama.cpp | `v0.9.0` (`cd959ba`) | CMake Release AVX2/FMA CPU |
| peregrine | `0.1.0` (`2a227d2f`) | cargo build --release -p peregrine |

## Model: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (SHA-256: `6eb923e7d26e9cea...`)

### Benchmark Results: The Four Numbers

Per `docs/targets.md`: single-user decode tok/s, prompt tok/s, total tok/s at 4 and 10 users, and first-token time at 10 users; medians reported with spread (± half-range).

| Context | Actual Tokens | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |
|---|---|---|---|---|---|---|---|---|
| 512 | 509 | llama.cpp | 77.4 ± 0.3 | 77.4% | 277.1 ± 1.1 | 55.0 ± 0.4 | 42.5 ± 0.1 | 22.350 ± 0.050 s |
| 512 | 511 | peregrine | 52.9 ± 0.3 | 52.9% | 263.2 ± 1.4 | 70.7 ± 0.4 | 71.7 ± 0.3 | 14.760 ± 0.150 s |
| 4096 | 4090 | llama.cpp | 52.0 ± 0.2 | 52.0% | 192.9 ± 1.0 | 5.0 | 0.0 | 0.000 s |
| 4096 | 4086 | peregrine | 45.2 ± 0.1 | 45.2% | 198.2 ± 0.1 | 10.0 ± 0.1 | 9.8 ± 0.0 | 151.420 ± 0.200 s |

### Head-to-Head Comparison

| Context | Metric | Peregrine | Competitor | Speedup | Peregrine % Ceiling |
|---|---|---|---|---|---|
| 512 | Single Decode | 52.9 ± 0.3 | 77.4 ± 0.3 (llama.cpp) | 0.68x | 52.9% |
| 512 | Prompt (Prefill) | 263.2 ± 1.4 | 277.1 ± 1.1 | 0.95x | — |
| 512 | 4 Users Throughput | 70.7 ± 0.4 | 55.0 ± 0.4 | 1.28x | — |
| 512 | 10 Users Throughput | 71.7 ± 0.3 | 42.5 ± 0.1 | 1.69x | — |
| 512 | 10 Users TTFT | 14.760 ± 0.150 s | 22.350 ± 0.050 s | 1.51x faster | — |
| 4096 | Single Decode | 45.2 ± 0.1 | 52.0 ± 0.2 (llama.cpp) | 0.87x | 45.2% |
| 4096 | Prompt (Prefill) | 198.2 ± 0.1 | 192.9 ± 1.0 | 1.03x | — |
| 4096 | 4 Users Throughput | 10.0 ± 0.1 | 5.0 | 1.99x | — |
| 4096 | 10 Users Throughput | 9.8 ± 0.0 | 0.0 | — | — |
| 4096 | 10 Users TTFT | 151.420 ± 0.200 s | 0.000 s | — | — |

## Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (SHA-256: `6f85a640a97cf2bf...`)

### Benchmark Results: The Four Numbers

Per `docs/targets.md`: single-user decode tok/s, prompt tok/s, total tok/s at 4 and 10 users, and first-token time at 10 users; medians reported with spread (± half-range).

| Context | Actual Tokens | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |
|---|---|---|---|---|---|---|---|---|
| 512 | 508 | llama.cpp | 42.0 ± 1.1 | 85.1% | 173.6 ± 1.0 | 34.2 ± 0.1 | 23.9 ± 0.1 | 35.300 ± 0.040 s |
| 512 | 511 | peregrine | 46.1 ± 0.5 | 93.3% | 197.3 ± 0.7 | 53.8 ± 0.5 | 55.2 ± 0.0 | 18.960 ± 0.010 s |
| 4096 | 4079 | llama.cpp | 29.2 ± 0.1 | 59.2% | 122.2 ± 0.2 | 3.1 ± 0.0 | 0.0 | 0.000 s |
| 4096 | 4081 | peregrine | 36.9 ± 0.1 | 74.6% | 144.6 ± 0.9 | 7.7 ± 0.0 | 7.7 ± 0.0 | 192.540 ± 0.910 s |

### Head-to-Head Comparison

| Context | Metric | Peregrine | Competitor | Speedup | Peregrine % Ceiling |
|---|---|---|---|---|---|
| 512 | Single Decode | 46.1 ± 0.5 | 42.0 ± 1.1 (llama.cpp) | 1.10x | 93.3% |
| 512 | Prompt (Prefill) | 197.3 ± 0.7 | 173.6 ± 1.0 | 1.14x | — |
| 512 | 4 Users Throughput | 53.8 ± 0.5 | 34.2 ± 0.1 | 1.57x | — |
| 512 | 10 Users Throughput | 55.2 ± 0.0 | 23.9 ± 0.1 | 2.31x | — |
| 512 | 10 Users TTFT | 18.960 ± 0.010 s | 35.300 ± 0.040 s | 1.86x faster | — |
| 4096 | Single Decode | 36.9 ± 0.1 | 29.2 ± 0.1 (llama.cpp) | 1.26x | 74.6% |
| 4096 | Prompt (Prefill) | 144.6 ± 0.9 | 122.2 ± 0.2 | 1.18x | — |
| 4096 | 4 Users Throughput | 7.7 ± 0.0 | 3.1 ± 0.0 | 2.47x | — |
| 4096 | 10 Users Throughput | 7.7 ± 0.0 | 0.0 | — | — |
| 4096 | 10 Users TTFT | 192.540 ± 0.910 s | 0.000 s | — | — |
