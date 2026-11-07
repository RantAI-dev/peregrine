# Benchmark Report

**Date**: 2026-11-07T00:07:13.779170+00:00
**Model**: `llama-F16.gguf` (SHA-256: `6dd66b5b17b64a32...`)

## Machine Record

| Property | Value |
|---|---|
| Host / CPU | Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz |
| CPU Features | fma, avx, avx2 |
| Memory Bandwidth Ceiling | **18.32 GB/s** (Peregrine telemetry probe) |
| Compute Ceiling | 605.5 GFLOP/s (Ridge: 33.1) |
| Kernel / OS | 6.8.0-147-generic (Linux) |
| CPU Governor | unknown (VM or fixed frequency) |
| Benchmark Threads | 8 |

## Engines Tested

| Engine | Version / Commit | Build Details |
|---|---|---|
| peregrine | `0.1.0` (`62da8600`) | cargo build --release -p peregrine |

## Benchmark Results: The Four Numbers

Per `docs/targets.md`: single-user decode tok/s, prompt tok/s, total tok/s at 4 and 10 users, and first-token time at 10 users; medians reported with spread (± half-range).

| Context | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |
|---|---|---|---|---|---|---|---|
| 512 | peregrine | 4704.4 | 66.0% | 4318.2 | 598.3 | 588.1 | 0.160 s |
