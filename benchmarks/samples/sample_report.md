# Benchmark Report (Synthetic Smoke Sample)

> **Note**: This is a synthetic smoke sample generated using a tiny synthetic GGUF model for CI/development verification, not real model weights.

**Date**: 2026-11-07T00:28:25.023172+00:00
**Model**: `llama-F16.gguf` (SHA-256: `6dd66b5b17b64a32...`)

## Machine Record

| Property | Value |
|---|---|
| Host / CPU | Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz |
| CPU Features | fma, avx, avx2 |
| Memory Bandwidth Ceiling | **12.22 GB/s** (Peregrine telemetry probe) |
| Compute Ceiling | 269.6 GFLOP/s (Ridge: 22.1) |
| Kernel / OS | 6.8.0-147-generic (Linux) |
| CPU Governor | unknown (VM or fixed frequency) |
| Benchmark Threads | 8 |

## Engines Tested

| Engine | Version / Commit | Build Details |
|---|---|---|
| peregrine | `0.1.0` (`78b47c06`) | cargo build --release -p peregrine |

## Benchmark Results: The Four Numbers

Per `docs/targets.md`: single-user decode tok/s, prompt tok/s, total tok/s at 4 and 10 users, and first-token time at 10 users; medians reported with spread (± half-range).

| Context | Actual Tokens | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |
|---|---|---|---|---|---|---|---|---|
| 512 | 504 | peregrine | 29.3 | 0.6% | 962.3 | 23.2 | 23.2 | 3.930 s |
