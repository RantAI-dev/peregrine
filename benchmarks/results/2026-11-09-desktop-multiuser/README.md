# Desktop multi-user re-check, 2026-11-09

Origin/main at `071dfad` (#109, F16-direct attention; which includes #111's AVX-512
Q4_K/Q5_0 tiles) versus this branch. Same host (Ryzen AI 9 HX 370, 12 cores / 24 threads,
AVX-512, WSL2, 32.95 GB/s streaming, 360 GFLOP/s FMA), `benchmarks/run.py --mode server`,
`--runs 3` (median ± half-range), greedy decode, 256 tokens per request, `ignore_eos`.

Two pool widths are recorded because the server's default is `available_parallelism()`
(24 here) and the tuned single-row pool is the physical cores (12):

- `before/512`, `after/512`: `--threads 12` (what the earlier comparisons used).
- `before/512-threads24`, `after/512-threads24`: `--threads 24`, the engine's default on
  this host. This is where the per-user attention fork–joins cost the most.
- `before/4096`: three one-round invocations per engine (one full run does not fit in a
  ten-minute command); samples are combined by `summarize.py`.
- `before/512-fa-off`: llama.cpp with `-fa off`, measured alongside the base; the llama.cpp
  binary is identical in both arms.

## Flash attention setting (llama.cpp v0.9.0)

`-fa auto` is llama.cpp's default, resolves to flash attention **on** on the CPU backend,
and is the faster setting on this host at every row. It is the published comparison:

| Row | `-fa auto` (on) | `-fa off` |
|---|---|---|
| 512, 1B, 10 users, aggregate tok/s | **60.5** | 35.6 |
| 512, 1B, 10 users, decode aggregate tok/s | **107.4** | 62.9 |
| 512, 0.5B, 10 users, aggregate tok/s | **92.8** | 56.9 |
| 512, 0.5B, 10 users, decode aggregate tok/s | **163.2** | 97.0 |
| 4096, 1B, 10 users, first token (s) | **144.2** | not finished after 590 s of prefill |

## 24-thread pool (the server default here), 512 context, Llama-3.2-1B (`6f85a640…`)

| Row | before | after | llama.cpp | before ratio | after ratio |
|---|---|---|---|---|---|
| 1 user decode tok/s | 67.0 ± 0.4 | 65.9 ± 0.6 | 32.1 / 27.8 | 2.09x | 2.37x |
| 4 users aggregate tok/s | 53.9 ± 6.9 | 61.7 ± 2.0 | 68.6 / 65.4 | 0.79x | 0.94x |
| 10 users aggregate tok/s | 63.2 ± 1.9 | **76.8 ± 0.3** | 78.0 / 78.0 | 0.81x | **0.98x** |
| 10 users decode aggregate tok/s | 119.4 ± 7.8 | **157.4 ± 5.5** | 122.8 / 118.9 | 0.97x | 1.32x |
| 10 users median TTFT (s) | 11.3 ± 1.0 | 9.5 ± 0.1 | 11.2 / 8.8 | 1.01x | 1.09x |

Ten-user aggregate rises 21.5 % and decode aggregate 31.8 %. Single-user decode uses the
12-thread narrow pool and does not move (the 1.6 % is inside the before range).

llama.cpp's single-user row is much worse at 24 threads because it has no narrow pool;
that row is not a like-for-like claim, it is recorded because one `--threads` sets both
engines.

## 12-thread pool (physical cores), 512 context

**Llama-3.2-1B** (`6f85a640…`)

| Row | before | after | llama.cpp | before ratio | after ratio |
|---|---|---|---|---|---|
| 1 user decode tok/s | 66.2 ± 0.7 | 66.6 ± 1.4 | 51.8 / 50.3 | 1.28x | 1.32x |
| 4 users aggregate tok/s | 69.4 ± 3.2 | 75.5 ± 5.1 | 58.9 / 59.7 | 1.18x | 1.27x |
| 10 users aggregate tok/s | 76.4 ± 0.1 | 76.5 ± 0.8 | 60.5 / 58.6 | 1.26x | 1.31x |
| 10 users decode aggregate tok/s | 306.7 ± 25.3 | 308.4 ± 36.9 | 107.4 / 107.5 | 2.86x | 2.87x |
| 10 users median TTFT (s) | 13.9 ± 0.6 | 13.8 ± 0.6 | 16.3 / 16.3 | 0.85x | 0.85x |

**Qwen2.5-0.5B** (`74a4da8c…`)

| Row | before | after | llama.cpp | before ratio | after ratio |
|---|---|---|---|---|---|
| 1 user decode tok/s | 100.7 ± 0.5 | 99.5 ± 0.5 | 73.6 / 71.9 | 1.37x | 1.38x |
| 4 users aggregate tok/s | 118.3 ± 7.0 | 102.3 ± 0.9 | 90.6 / 82.1 | 1.31x | 1.25x |
| 10 users aggregate tok/s | 111.3 ± 2.8 | 121.9 ± 1.2 | 92.8 / 91.6 | 1.20x | 1.33x |
| 10 users decode aggregate tok/s | 414.3 ± 2.9 | 427.5 ± 61.6 | 163.2 / 149.5 | 2.54x | 2.86x |
| 10 users median TTFT (s) | 8.8 ± 0.6 | 8.7 ± 0.1 | 10.5 / 9.3 | 0.84x | 0.94x |

At 12 threads the ten-user row is already at llama.cpp's ceiling on this engine (76 takes
the physical-core bandwidth/compute limit); the attention batching is neutral there
(attention 18.3 -> 18.4 ms of a 64 ms step). The 0.5B four-user row falls while both
engines' own numbers move (llama.cpp 90.6 -> 82.1, the engine 118.3 -> 102.3); it is
drift dominated, and nothing in this change touches that row's kernel.

## 4096 context, Llama-3.2-1B, 10 users, 12 threads

| Row | before | after | llama.cpp | before ratio | after ratio |
|---|---|---|---|---|---|
| aggregate tok/s | 9.4 ± 0.1 | 9.5 ± 0.1 | 8.4 / 8.4 | 1.12x | 1.13x |
| median TTFT (s) | 158.3 ± 1.4 | 152.3 ± 0.7 | 144.2 ± 0.1 | 1.10x | 1.06x |

The 64-token pieces for prompts over 1024 tokens are the whole of that TTFT move: with
the 16-token default the same base engine measures 158.3 s.

## Profile: pure 10-row decode step, 512 positions, Llama-3.2-1B

`PEREGRINE_PROF2=full`, engine thread, full 32-step windows in which all ten users decode
(`rows/step 10.0`), per step:

| Bucket | 24 threads before | 24 threads after | 12 threads before | 12 threads after |
|---|---|---|---|---|
| attention | 16.4-17.7 | 15.1-15.7 | 18.0-18.8 | 18.1-19.2 |
| matmul qkv | 6.0-6.8 | 4.6-5.1 | 3.0-3.1 | 3.0-3.1 |
| matmul wo | 6.8-7.9 | 4.9-5.4 | 2.5 | 2.5 |
| matmul gate/up | 11.9-12.7 | 10.5-10.8 | 12.7-13.0 | 12.1-12.3 |
| matmul down | 10.4-11.1 | 9.1-9.2 | 10.0-10.1 | 10.0-10.1 |
| output head | 7.3-7.5 | 7.5 | 11.2-11.6 | 11.3-11.5 |
| quantise (serial) | 2.3-2.4 | 2.1-2.2 | 1.6 | 1.7 |
| rope + kv write + residual | 4.0-6.1 | 3.3-3.8 | 2.3 | 2.3 |
| **step wall** | **82.4-89.8** | **69.1-72.2** | **64.2-65.7** | **63.3-65.5** |

At 24 threads the step lost 10 fork–joins per layer (barrier wait itself 32-39 ->
23-26 ms/step). At 12 threads the 10 per-user calls already fit one wave, so collapsing
them is neutral. The 4096-prefill profile (64-row steps) still has attention at
125-168 ms of a 370-420 ms step: those are AVX2 `scores`/`mix` kernels on an AVX-512 host,
which is the next lever (a numerics change for the avx512 tier, so its own PR).

## Files

- `before/512`, `after/512`: 12-thread run sets (fa auto).
- `before/512-threads24`, `after/512-threads24`: 24-thread run sets (fa auto).
- `before/512-fa-off`: llama.cpp `-fa off`.
- `before/4096`, `after/4096`: three one-round run sets per engine.
- `summarize.py`: rebuilds these medians from the JSON files.
