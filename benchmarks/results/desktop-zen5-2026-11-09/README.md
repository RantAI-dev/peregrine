# Desktop baseline after the November 8-9 speed-ups (roadmap S3.2)

A same-host re-measurement of the desktop matrix after the week's kernel and scheduling work:
repacked k-quant tiles on AVX-512 (#108), AVX-512 repacked Q4_K/Q5_0 kernels (#111), the
prefill-attention rewrites (#113, #115), one attention region for all decoding users (#114),
the Q6_K and prefill-GEMM changes (#118, #120) and the AVX-512 attention mix (#121). The previous
desktop baseline is [`../desktop-zen5-2026-11-08-fa-off/`](../desktop-zen5-2026-11-08-fa-off/).

- Date: 2026-11-09 (runs 18:51-19:52 local, UTC+7).
- Peregrine: `origin/main` at `b88fcd7` (`perf: AVX-512 attention kernels (#121)`), `cargo build
  --release -p peregrine`, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`. The measured tree was that
  commit plus a test-only golden file change; the built binary is byte-identical to the one built
  from `b88fcd7` (SHA-256 `5cba085372b724fd0f153443b605831efe98283ba9f2663bfd125542351da73d`).
- llama.cpp: v0.9.0 (`cd959ba`), the harness cache build (CMake Release, AVX2/FMA CPU), driven
  with `--llama-fa auto` and `--llama-fa off` in two passes.
- Models, the two files the 2026-11-08 baseline used plus two new files:
  - `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807,694,464 bytes,
    `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`
  - `qwen2.5-0.5b-instruct-q4_k_m.gguf`, 491,400,032 bytes,
    `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
  - `SmolLM2-360M-Instruct.Q4_K_M.gguf`, 270,590,880 bytes,
    `2fa3f013dcdd7b99f9b237717fa0b12d75bbb89984cc1274be1471a465bac9c2` (bartowski)
  - `google_gemma-3-1b-it-Q4_K_M.gguf`, 806,058,496 bytes,
    `12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d` (bartowski)
- Settings: 12 threads (one per physical core), 256 greedy decode tokens per request, server mode,
  3 interleaved rounds per engine and flash-attention setting, prefix-cache defeat by a
  per-request nonce, prompt lengths calibrated per engine and model. Every measured prompt is
  within 5% of its target.

## The machine

| Property | Value |
|---|---|
| CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M (Zen 5), family 26, model 36 |
| Topology | 1 socket, 12 physical cores, 2 threads per core, 24 logical CPUs |
| SIMD | AVX2, FMA, AVX-512F/DQ/BW/VL/IFMA/VNNI, AVX-VNNI, AVX512-BF16, GFNI, VAES |
| RAM | 11,763 MiB total, 3,072 MiB swap |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2 (x86_64), glibc 2.39 |
| Virtualization | WSL2 under the Microsoft hypervisor, full virtualization (AMD-V) |
| CPU governor | unknown (VM or fixed frequency) - WSL2 does not expose it |
| Benchmark threads | 12 |
| Streaming-bandwidth probe | 32.4-52.6 GB/s across the invocations (38.7 GB/s in the first) |

The probe has ranged 29-53 GB/s across sessions on this host, so the `% of ceiling` column is not
carried over and the tok/s numbers are the comparable ones. The manifests record
`machine.simd_tier: x86_64-avx2-avx512-vnni-gfni-vbmi`; `peregrine tier` prints the same kernel
tier, so the engine ran AVX-512.

## Server default

llama.cpp v0.9.0's own default is `-fa auto`, which resolves to flash attention **on** on the CPU
backend. Both servers are started with 12 threads explicitly (`-t 12`), one per physical core;
12 is also the harness's setting for every row, so no engine runs at its own thread default. The `-fa auto` pass below is that default; the `-fa off` pass is the alternative. Each
ratio uses whichever setting measured faster for that row (lower for TTFT), as the harness policy
asks. The default is the faster setting for every prompt and multi-user row here; `-fa off` wins
some single-user decode rows, and for SmolLM2 360M and gemma-3-1b-it at 512 it is the decode
competitor.

## How the runs were split

The host keeps each command under ten minutes, so the sweep is one invocation per
(model, context/user-count slice, flash-attention setting) with both engines interleaved in the
same invocation where possible:

- 512 tokens, 1 and 10 users: Peregrine and llama.cpp in one invocation per setting.
- 512 tokens, 4 users: Peregrine and llama.cpp in one invocation per setting.
- 4096 tokens, 1 user: Peregrine and llama.cpp in one invocation per setting.
- 4096 tokens, 4 users: **not measured** (not attempted in this run or the 2026-11-08 one).
- 4096 tokens, 10 users: **not completed**. One round of the cheapest combination (Qwen 0.5B,
  llama.cpp `-fa auto`, `--runs 1`) took 3m43 and served 13.4 tok/s in total; the three-round row
  plus prompt calibration does not fit the 10-minute cap, and the 2026-11-08 run reached the same
  result on both models. The rows are recorded as not completed rather than estimated.

The raw manifests are beside this file; each carries the engine build, the machine record and
every individual sample.

One data-quality note: one llama.cpp single-user TTFT sample on Qwen 0.5B at 512 was negative
(-1.66 s), a timestamp artifact of the streaming client on a warm server. Single-user TTFT is not
published; the 10-user TTFT median for that engine and setting is unaffected (its three samples
are positive).

## Results

Medians of 3 rounds, +- half-range. The ratio is Peregrine / the faster llama.cpp setting for that
row, so above 1 means Peregrine is faster. For TTFT, below 1 means Peregrine's first token arrives
sooner.

### Llama 3.2 1B Instruct Q4_K_M

| Context | Users | Metric | Peregrine (`b88fcd7`) | llama.cpp default (`-fa auto`) | llama.cpp `-fa off` | Faster llama.cpp | Ratio |
|---|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 60.64 ± 0.74 | 46.41 ± 1.22 | 44.83 ± 1.01 | default | **1.31x** |
| 512 | 1 | Prompt tok/s | 272.59 ± 7.22 | 252.66 ± 0.98 | 232.29 ± 9.37 | default | **1.08x** |
| 512 | 4 | Aggregate tok/s | 70.32 ± 9.33 | 60.30 ± 4.51 | 41.30 ± 0.45 | default | **1.17x** |
| 512 | 10 | Aggregate tok/s | 83.56 ± 0.79 | 61.75 ± 2.09 | 36.14 ± 0.85 | default | **1.35x** |
| 512 | 10 | TTFT (s) | 13.39 ± 1.36 | 15.54 ± 3.37 | 22.70 ± 0.34 | default | **0.86x** |
| 4096 | 1 | Decode tok/s | 48.49 ± 0.27 | 30.35 ± 0.25 | 29.78 ± 0.12 | default | **1.60x** |
| 4096 | 1 | Prompt tok/s | 195.30 ± 1.55 | 174.30 ± 1.02 | 163.02 ± 6.60 | default | **1.12x** |
| 4096 | 4 | Aggregate tok/s | not measured | not measured | not measured | - | - |
| 4096 | 10 | Aggregate tok/s | not completed | not completed | not completed | - | - |

### Qwen 2.5 0.5B Instruct Q4_K_M

| Context | Users | Metric | Peregrine (`b88fcd7`) | llama.cpp default (`-fa auto`) | llama.cpp `-fa off` | Faster llama.cpp | Ratio |
|---|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 95.53 ± 0.50 | 69.48 ± 1.59 | 64.24 ± 2.16 | default | **1.37x** |
| 512 | 1 | Prompt tok/s | 479.29 ± 6.77 | 423.79 ± 4.30 | 390.43 ± 13.03 | default | **1.13x** |
| 512 | 4 | Aggregate tok/s | 108.42 ± 22.28 | 96.03 ± 10.71 | 58.71 ± 6.01 | default | **1.13x** |
| 512 | 10 | Aggregate tok/s | 121.66 ± 0.19 | 93.57 ± 0.87 | 54.37 ± 1.81 | default | **1.30x** |
| 512 | 10 | TTFT (s) | 8.13 ± 0.05 | 11.04 ± 1.34 | 12.65 ± 2.54 | default | **0.74x** |
| 4096 | 1 | Decode tok/s | 71.87 ± 1.16 | 45.80 ± 0.75 | 45.70 ± 0.91 | default | **1.57x** |
| 4096 | 1 | Prompt tok/s | 355.05 ± 3.96 | 272.43 ± 2.36 | 270.96 ± 7.44 | default | **1.30x** |
| 4096 | 4 | Aggregate tok/s | not measured | not measured | not measured | - | - |
| 4096 | 10 | Aggregate tok/s | not completed | not completed | not completed | - | - |

### SmolLM2 360M Instruct Q4_K_M

| Context | Users | Metric | Peregrine (`b88fcd7`) | llama.cpp default (`-fa auto`) | llama.cpp `-fa off` | Faster llama.cpp | Ratio |
|---|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 102.92 ± 6.80 | 70.32 ± 12.35 | 76.97 ± 3.05 | `-fa off` | **1.34x** |
| 512 | 1 | Prompt tok/s | 451.12 ± 16.78 | 436.31 ± 7.74 | 431.78 ± 8.72 | default | **1.03x** |
| 512 | 4 | Aggregate tok/s | 117.10 ± 4.92 | 84.96 ± 8.68 | 59.57 ± 1.34 | default | **1.38x** |
| 512 | 10 | Aggregate tok/s | 146.55 ± 6.50 | 102.40 ± 1.97 | 50.39 ± 0.46 | default | **1.43x** |
| 512 | 10 | TTFT (s) | 8.08 ± 0.26 | 9.45 ± 0.12 | 15.77 ± 0.35 | default | **0.86x** |
| 4096 | 1 | Decode tok/s | 70.83 ± 3.01 | 41.91 ± 0.40 | 41.68 ± 0.34 | default | **1.69x** |
| 4096 | 1 | Prompt tok/s | 292.86 ± 4.12 | 258.96 ± 2.13 | 263.87 ± 7.41 | `-fa off` | **1.11x** |
| 4096 | 4 | Aggregate tok/s | not measured | not measured | not measured | - | - |
| 4096 | 10 | Aggregate tok/s | not completed | not completed | not completed | - | - |

### gemma-3-1b-it Q4_K_M

| Context | Users | Metric | Peregrine (`b88fcd7`) | llama.cpp default (`-fa auto`) | llama.cpp `-fa off` | Faster llama.cpp | Ratio |
|---|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 59.87 ± 2.60 | 42.87 ± 2.26 | 44.26 ± 0.06 | `-fa off` | **1.35x** |
| 512 | 1 | Prompt tok/s | 284.86 ± 10.67 | 257.59 ± 4.02 | 258.10 ± 3.93 | `-fa off` | **1.10x** |
| 512 | 4 | Aggregate tok/s | 63.35 ± 0.28 | 49.59 ± 0.49 | 45.47 ± 0.58 | default | **1.28x** |
| 512 | 10 | Aggregate tok/s | 69.31 ± 0.31 | 54.14 ± 0.57 | 45.36 ± 0.19 | default | **1.28x** |
| 512 | 10 | TTFT (s) | 14.47 ± 0.09 | 20.58 ± 0.35 | 24.48 ± 0.12 | default | **0.70x** |
| 4096 | 1 | Decode tok/s | 46.30 ± 0.65 | 38.38 ± 1.52 | 40.97 ± 0.74 | `-fa off` | **1.13x** |
| 4096 | 1 | Prompt tok/s | 279.59 ± 2.39 | 233.82 ± 6.70 | 247.90 ± 2.05 | `-fa off` | **1.13x** |
| 4096 | 4 | Aggregate tok/s | not measured | not measured | not measured | - | - |
| 4096 | 10 | Aggregate tok/s | not completed | not completed | not completed | - | - |

Every completed throughput row is at or above 1.0x. The narrowest is SmolLM2 360M prompt
processing at 512 tokens (1.03x); TTFT is below 1.0 on every row (Peregrine's first token arrives
sooner), from 0.70x on gemma-3-1b-it to 0.86x on Llama 3.2 1B and SmolLM2 360M.

## What moved against the 2026-11-08 build

Peregrine against itself, same host, same files, 2026-11-08 `e48aea5` -> 2026-11-09 `b88fcd7`
(the 2026-11-08 numbers are from the `-fa off` folder's Peregrine rows):

| Model | Context | Metric | 2026-11-08 | 2026-11-09 | Change |
|---|---|---|---|---|---|
| Llama 3.2 1B | 512 | Decode tok/s | 62.63 | 60.64 | -3.2% |
| Llama 3.2 1B | 512 | Prompt tok/s | 183.76 | 272.59 | +48.3% |
| Llama 3.2 1B | 512 | 4-user tok/s | 51.62 | 70.32 | +36.2% |
| Llama 3.2 1B | 512 | 10-user tok/s | 53.40 | 83.56 | +56.5% |
| Llama 3.2 1B | 4096 | Decode tok/s | 47.03 | 48.49 | +3.1% |
| Llama 3.2 1B | 4096 | Prompt tok/s | 130.87 | 195.30 | +49.2% |
| Qwen 2.5 0.5B | 512 | Decode tok/s | 69.96 | 95.53 | +36.5% |
| Qwen 2.5 0.5B | 512 | Prompt tok/s | 321.90 | 479.29 | +48.9% |
| Qwen 2.5 0.5B | 512 | 4-user tok/s | 78.27 | 108.42 | +38.5% |
| Qwen 2.5 0.5B | 512 | 10-user tok/s | 83.35 | 121.66 | +46.0% |
| Qwen 2.5 0.5B | 4096 | Decode tok/s | 54.87 | 71.87 | +31.0% |
| Qwen 2.5 0.5B | 4096 | Prompt tok/s | 215.09 | 355.05 | +65.1% |

The 10-user TTFT also fell: Llama 3.2 1B 19.66 -> 13.39 s (-32%), Qwen 2.5 0.5B 12.12 -> 8.13 s
(-33%). No per-PR A/B was run for this report, so the table is the end-to-end change from all of
#108, #111, #113, #114, #115, #118, #120 and #121 together, plus same-day host drift.

The Llama 1B file has no Q5_0 tensors (it is Q4_K/Q6_K), so its prompt gain is the repacked Q4_K
path: repacked tiles on by default on AVX-512 (#108), the AVX-512 repacked Q4_K/Q5_0 kernels
(#111), the eight-row prefill tile (#120) and the Q6_K register accumulators (#118). The Qwen and
SmolLM2 files are mostly Q5_0 and gain the same way. Multi-user throughput is the one attention
region for all decoding users (#114); the Qwen 0.5B decode gain also picks up the AVX-512 attention
mix (#121) and the prefill-attention rewrites (#113, #115).

## What was checked, and what was not

- Checked: all four model files hash to the values above; every measured prompt is within 5% of
  its target; 3 samples per phase per engine and per flash-attention setting; the `-fa auto` and
  `-fa off` settings are recorded in every llama.cpp manifest.
- Checked: no completed row is below 1.0x on any throughput metric.
- Not measured: 4 users at 4096 (not attempted), 16k context, other quantisations. 10 users at
  4096 does not complete inside the 10-minute cap on this host (one round already takes 3m43 for
  the cheapest combination; see above).
