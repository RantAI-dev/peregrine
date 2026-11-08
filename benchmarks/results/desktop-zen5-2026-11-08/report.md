# Benchmark Report

**Date**: 2026-11-08 (runs 10:16–12:41 local, UTC+07)

This is the S3.2 desktop baseline re-run after the November speed-ups, plus a second
Peregrine pass with the repacked Q4_K kernels forced on (`PEREGRINE_KREPACK=1`).

## Machine Record

| Property | Value |
|---|---|
| Host / CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M |
| CPU Features | fma, avx, avx2, avx512f, avx_vnni, avx512_bf16 |
| Memory Bandwidth Ceiling | **29.29 – 47.91 GB/s** across the 12 invocations (Peregrine telemetry probe) |
| Compute Ceiling | 272.0 – 344.4 GFLOP/s (Ridge: 8.1 – 10.6) |
| Kernel / OS | 6.18.33.2-microsoft-standard-WSL2 (Linux) |
| CPU Governor | unknown (VM or fixed frequency) |
| Benchmark Threads | 12 |

The bandwidth probe is not stable enough to compare across days or even across
invocations: it returned 29.29 GB/s for the first invocation, 47.91 GB/s for the second,
and landed between 30.55 and 45.12 GB/s for the rest. The 2026-11-07 baseline recorded a
single 35.43 GB/s figure. The "% of ceiling" column below therefore uses each run's own
probe value and should be read as a per-run indication only; the raw tok/s numbers are
the comparable ones.

## Engines Tested

| Engine | Version / Commit | Build Details |
|---|---|---|
| llama.cpp | `v0.9.0` (`cd959ba`) | CMake Release AVX2/FMA CPU |
| peregrine | `0.1.0` (`5f76d488`) | cargo build --release -p peregrine |
| peregrine (KREPACK=1) | `0.1.0` (`5f76d488`) | same binary, `PEREGRINE_KREPACK=1` in the server's environment |

## Model: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (SHA-256: `74a4da8c9fdbcd15...`)

### Benchmark Results: The Four Numbers

Per `docs/targets.md`: single-user decode tok/s, prompt tok/s, total tok/s at 4 and 10 users,
and first-token time at 10 users; medians reported with spread (± half-range).

| Context | Actual Tokens | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |
|---|---|---|---|---|---|---|---|---|
| 512 | 509 | llama.cpp | 75.8 ± 0.6 | 101.5% | 435.2 ± 2.1 | 81.8 ± 8.1 | 93.4 ± 0.8 | 9.540 ± 1.560 s |
| 512 | 511 | peregrine | 68.1 ± 1.2 | 91.2% | 270.6 ± 4.9 | 70.6 ± 6.0 | 78.0 ± 2.3 | 12.160 ± 0.960 s |
| 512 | 509 | peregrine KREPACK=1 | 67.8 ± 2.1 | 76.8% | 263.5 ± 6.6 | 80.5 ± 6.6 | 77.0 ± 2.3 | 14.190 ± 0.970 s |
| 4096 | 4087 | peregrine | 51.6 ± 0.1 | 42.2% | 184.9 ± 1.4 | 10.3 ± 0.1 | 10.5 ± 0.1 | 141.170 ± 1.370 s |
| 4096 | 4084 | peregrine KREPACK=1 | 50.5 ± 1.0 | 47.7% | 180.5 ± 1.8 | 10.2 ± 0.0 | 10.4 ± 0.1 | 142.720 ± 0.490 s |
| 4096 | 4091 | llama.cpp | 48.8 ± 0.6 | 42.7% | 279.4 ± 1.4 | 14.2 ± 0.2 | 13.4 ± 0.1 | 90.050 ± 0.890 s |

### Head-to-Head Comparison

| Context | Metric | Peregrine | Peregrine KREPACK=1 | Competitor | Speedup | KREPACK speedup | Peregrine % Ceiling |
|---|---|---|---|---|---|---|---|
| 512 | Single Decode | 68.1 ± 1.2 | 67.8 ± 2.1 | 75.8 ± 0.6 (llama.cpp) | 0.90x | 0.89x | 91.2% |
| 512 | Prompt (Prefill) | 270.6 ± 4.9 | 263.5 ± 6.6 | 435.2 ± 2.1 | 0.62x | 0.61x | — |
| 512 | 4 Users Throughput | 70.6 ± 6.0 | 80.5 ± 6.6 | 81.8 ± 8.1 | 0.86x | 0.98x | — |
| 512 | 10 Users Throughput | 78.0 ± 2.3 | 77.0 ± 2.3 | 93.4 ± 0.8 | 0.83x | 0.82x | — |
| 512 | 10 Users TTFT | 12.160 ± 0.960 s | 14.190 ± 0.970 s | 9.540 ± 1.560 s | 0.78x faster | 0.67x faster | — |
| 4096 | Single Decode | 51.6 ± 0.1 | 50.5 ± 1.0 | 48.8 ± 0.6 (llama.cpp) | 1.06x | 1.04x | 42.2% |
| 4096 | Prompt (Prefill) | 184.9 ± 1.4 | 180.5 ± 1.8 | 279.4 ± 1.4 | 0.66x | 0.65x | — |
| 4096 | 4 Users Throughput | 10.3 ± 0.1 | 10.2 ± 0.0 | 14.2 ± 0.2 | 0.73x | 0.72x | — |
| 4096 | 10 Users Throughput | 10.5 ± 0.1 | 10.4 ± 0.1 | 13.4 ± 0.1 | 0.79x | 0.78x | — |
| 4096 | 10 Users TTFT | 141.170 ± 1.370 s | 142.720 ± 0.490 s | 90.050 ± 0.890 s | 0.64x faster | 0.63x faster | — |

## Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (SHA-256: `6f85a640a97cf2bf...`)

### Benchmark Results: The Four Numbers

| Context | Actual Tokens | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |
|---|---|---|---|---|---|---|---|---|
| 512 | 506 | llama.cpp | 51.0 ± 0.1 | 92.5% | 262.0 ± 3.9 | 54.5 ± 0.5 | 61.7 ± 0.2 | 15.580 ± 3.090 s |
| 512 | 509 | peregrine | 59.4 ± 0.2 | 107.7% | 173.5 ± 4.4 | 52.9 ± 2.7 | 51.8 ± 0.6 | 20.950 ± 0.480 s |
| 512 | 509 | peregrine KREPACK=1 | 59.0 ± 1.9 | 154.9% | 211.5 ± 2.3 | 56.4 ± 2.4 | 62.7 ± 2.0 | 15.160 ± 1.080 s |
| 4096 | 4073 | peregrine | 44.2 ± 0.5 | 84.3% | 125.4 ± 1.3 | 7.1 ± 0.1 | 7.3 ± 0.0 | 204.570 ± 0.640 s |
| 4096 | 4075 | peregrine KREPACK=1 | 43.3 ± 0.3 | 83.8% | 144.6 ± 0.9 | 8.3 ± 0.1 | 8.4 ± 0.0 | 177.500 ± 0.410 s |
| 4096 | 4080 | llama.cpp | 32.4 ± 0.6 | 57.6% | 189.0 ± 8.4 | 9.1 ± 0.1 | 8.6 ± 0.0 | 143.970 ± 0.090 s |

### Head-to-Head Comparison

| Context | Metric | Peregrine | Peregrine KREPACK=1 | Competitor | Speedup | KREPACK speedup | Peregrine % Ceiling |
|---|---|---|---|---|---|---|---|
| 512 | Single Decode | 59.4 ± 0.2 | 59.0 ± 1.9 | 51.0 ± 0.1 (llama.cpp) | 1.16x | 1.16x | 107.7% |
| 512 | Prompt (Prefill) | 173.5 ± 4.4 | 211.5 ± 2.3 | 262.0 ± 3.9 | 0.66x | 0.81x | — |
| 512 | 4 Users Throughput | 52.9 ± 2.7 | 56.4 ± 2.4 | 54.5 ± 0.5 | 0.97x | 1.04x | — |
| 512 | 10 Users Throughput | 51.8 ± 0.6 | 62.7 ± 2.0 | 61.7 ± 0.2 | 0.84x | 1.02x | — |
| 512 | 10 Users TTFT | 20.950 ± 0.480 s | 15.160 ± 1.080 s | 15.580 ± 3.090 s | 0.74x faster | 1.03x faster | — |
| 4096 | Single Decode | 44.2 ± 0.5 | 43.3 ± 0.3 | 32.4 ± 0.6 (llama.cpp) | 1.36x | 1.33x | 84.3% |
| 4096 | Prompt (Prefill) | 125.4 ± 1.3 | 144.6 ± 0.9 | 189.0 ± 8.4 | 0.66x | 0.77x | — |
| 4096 | 4 Users Throughput | 7.1 ± 0.1 | 8.3 ± 0.1 | 9.1 ± 0.1 | 0.78x | 0.91x | — |
| 4096 | 10 Users Throughput | 7.3 ± 0.0 | 8.4 ± 0.0 | 8.6 ± 0.0 | 0.85x | 0.98x | — |
| 4096 | 10 Users TTFT | 204.570 ± 0.640 s | 177.500 ± 0.410 s | 143.970 ± 0.090 s | 0.70x faster | 0.81x faster | — |

## 2026-11-07 vs 2026-11-08

Medians, same harness, same files, same thread count. `P` is Peregrine with default
options, `P KREPACK` the same binary with `PEREGRINE_KREPACK=1`.

### `qwen2.5-0.5b-instruct-q4_k_m.gguf`

| Ctx | Metric | 11-07 P | 11-07 llama | 11-08 P | 11-08 P KREPACK | 11-08 llama |
|---|---|---|---|---|---|---|
| 512 | Decode tok/s | 66.5 | 70.8 | 68.1 (+2.5%) | 67.8 (+2.0%) | 75.8 (+7.1%) |
| 512 | Prefill tok/s | 271.1 | 419.4 | 270.6 (-0.2%) | 263.5 (-2.8%) | 435.2 (+3.8%) |
| 512 | 4 users tok/s | 71.1 | 79.6 | 70.6 (-0.7%) | 80.5 (+13.2%) | 81.8 (+2.7%) |
| 512 | 10 users tok/s | 80.0 | 93.8 | 78.0 (-2.6%) | 77.0 (-3.9%) | 93.4 (-0.4%) |
| 512 | 10 users TTFT s | 8.96 | 11.06 | 12.16 (+35.7%) | 14.19 (+58.4%) | 9.54 (-13.7%) |
| 4096 | Decode tok/s | 53.0 | 46.1 | 51.6 (-2.7%) | 50.5 (-4.8%) | 48.8 (+5.7%) |
| 4096 | Prefill tok/s | 200.7 | 269.9 | 184.9 (-7.9%) | 180.5 (-10.1%) | 279.4 (+3.5%) |
| 4096 | 4 users tok/s | 12.1 | 14.1 | 10.3 (-14.6%) | 10.2 (-15.6%) | 14.2 (+0.9%) |
| 4096 | 10 users tok/s | 12.0 | 13.4 | 10.5 (-12.4%) | 10.4 (-13.5%) | 13.4 (-0.3%) |
| 4096 | 10 users TTFT s | 121.31 | 91.03 | 141.17 (+16.4%) | 142.72 (+17.6%) | 90.05 (-1.1%) |

### `Llama-3.2-1B-Instruct-Q4_K_M.gguf`

| Ctx | Metric | 11-07 P | 11-07 llama | 11-08 P | 11-08 P KREPACK | 11-08 llama |
|---|---|---|---|---|---|---|
| 512 | Decode tok/s | 61.4 | 49.4 | 59.4 (-3.3%) | 59.0 (-3.8%) | 51.0 (+3.3%) |
| 512 | Prefill tok/s | 147.0 | 252.6 | 173.5 (+18.0%) | 211.5 (+43.9%) | 262.0 (+3.7%) |
| 512 | 4 users tok/s | 51.3 | 53.5 | 52.9 (+3.2%) | 56.4 (+10.0%) | 54.5 (+1.8%) |
| 512 | 10 users tok/s | 52.0 | 62.5 | 51.8 (-0.4%) | 62.7 (+20.5%) | 61.7 (-1.3%) |
| 512 | 10 users TTFT s | 18.29 | 15.19 | 20.95 (+14.5%) | 15.16 (-17.1%) | 15.58 (+2.6%) |
| 4096 | Decode tok/s | 45.5 | 31.9 | 44.2 (-2.7%) | 43.3 (-4.8%) | 32.4 (+1.8%) |
| 4096 | Prefill tok/s | 116.8 | 187.9 | 125.4 (+7.4%) | 144.6 (+23.9%) | 189.0 (+0.6%) |
| 4096 | 4 users tok/s | 7.1 | 9.2 | 7.1 (-0.1%) | 8.3 (+17.0%) | 9.1 (-1.2%) |
| 4096 | 10 users tok/s | 7.0 | 8.4 | 7.3 (+4.0%) | 8.4 (+19.6%) | 8.6 (+1.3%) |
| 4096 | 10 users TTFT s | 206.95 | 142.45 | 204.57 (-1.2%) | 177.50 (-14.2%) | 143.97 (+1.1%) |

## KREPACK=1 effect (same binary, same day)

| Model | Ctx | Decode | Prefill | 4 users | 10 users | 10 users TTFT |
|---|---|---|---|---|---|---|
| qwen2.5-0.5B | 512 | -0.5% | -2.6% | +14.1% | -1.3% | +16.7% (worse) |
| qwen2.5-0.5B | 4096 | -2.1% | -2.4% | -1.3% | -1.2% | +1.1% (worse) |
| Llama-3.2-1B | 512 | -0.6% | **+21.9%** | +6.7% | **+21.0%** | **-27.6%** (better) |
| Llama-3.2-1B | 4096 | -2.1% | **+15.3%** | +17.2% | +15.0% | **-13.2%** (better) |

## How the runs were split

The full harness sweep exceeds the 10-minute command budget on this host, so each
invocation covered one (model, context) pair. The 512-token pairs ran both engines in one
invocation (interleaved ABAB rounds, as in 2026-11-07); the 4096-token pairs ran one engine
per invocation. The four KREPACK=1 passes are Peregrine only and are documented by their
filenames (`*-peregrine-krepack-*.json`). Every file carries its own full machine record.

| File | Model | Ctx | Engine | Env |
|---|---|---|---|---|
| `2026-11-08-101658-*` | qwen 0.5B | 512 | peregrine + llama.cpp | default |
| `2026-11-08-103638-peregrine-*` | qwen 0.5B | 4096 | peregrine | default |
| `2026-11-08-105133-llama.cpp-*` | qwen 0.5B | 4096 | llama.cpp | — |
| `2026-11-08-105929-*` | 1B | 512 | peregrine + llama.cpp | default |
| `2026-11-08-112717-peregrine-*` | 1B | 4096 | peregrine | default |
| `2026-11-08-115036-llama.cpp-*` | 1B | 4096 | llama.cpp | — |
| `2026-11-08-115334-peregrine-krepack-*` | qwen 0.5B | 512 | peregrine | KREPACK=1 |
| `2026-11-08-121304-peregrine-krepack-*` | qwen 0.5B | 4096 | peregrine | KREPACK=1 |
| `2026-11-08-121645-peregrine-krepack-*` | 1B | 512 | peregrine | KREPACK=1 |
| `2026-11-08-124103-peregrine-krepack-*` | 1B | 4096 | peregrine | KREPACK=1 |
