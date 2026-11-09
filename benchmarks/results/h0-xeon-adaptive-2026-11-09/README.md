# H0 (Xeon E5-2680 v4): adaptive spin-wait, 2026-11-09

An interleaved A/B of the adaptive spin-wait against `main` at `9bcf524`, on H0, a shared virtual
machine. The point of the change is that the 1 ms spin window is right on a quiet host and wrong on
an oversubscribed one, so both conditions were measured, plus a deliberately heavier one.

- Date: 2026-11-09, 11:00-12:50 UTC.
- Host: Intel Xeon E5-2680 v4 @ 2.40 GHz under KVM, 8 logical CPUs, 1 thread/core, 11 GiB RAM,
  Linux (Ubuntu). The host is shared: co-tenants ran other work throughout, and the 1-minute load
  average ranged from 4.3 to 17.2 during these runs. Every raw file's `-meta.txt` records the load
  average at each run's end.
- Pinning: `taskset -c 4-7` for the benchmark, the CPU hogs and llama.cpp; 4 worker threads
  (`peregrine bench --threads 4`, llama.cpp `-t 4`).
- Peregrine: `main` at `9bcf524` plus `--threads` (`peregrine-base`), against this branch
  (`peregrine-branch`), both `cargo build --release`, `rustc 1.99.1`. The runs predate the rebase
  onto `e31499c`; main moved afterwards (Q5_0 repack, AVX-512 attention), which changes absolute
  kernel speed but not this A/B, because both binaries came from the same parent.
- llama.cpp: v0.9.0 (`cd959ba`), CMake Release, `GGML_NATIVE=ON`, `-p 512 -n 64 -d 512 -r 2`.
- Models (bartowski Q4_K_M, SHA-256 of the 1B in the meta):
  - `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807694464 bytes,
    `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`;
  - `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, 397808192 bytes,
    `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`;
  - `google_gemma-3-1b-it-Q4_K_M.gguf`, 806058496 bytes.
- `peregrine bench -m <model> --prompt-tokens 512 --tokens 64 --threads 4 --json --no-profile`.
  Prefill is 512 tokens; decode is 64 greedy tokens at the 512-token context.
- 5 rounds per engine, alternating which engine runs first each round (A,B,B,A,...), medians below.
  The heavy condition has 4 rounds; llama.cpp there is one repetition.
- Conditions: `quiet` = no hog of ours; `loaded` = four ALU busy loops pinned to 4-7 (two runnable
  threads per core); `heavy` = eight (three per core).

## Results

Medians. `branch/base` is the median of the five per-round ratios, so it is not the ratio of the
medians. Raw JSON is in `raw/`.

### Quiet (no hog; co-tenant load 4-13)

| Model | Engine | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| Llama-3.2-1B | base | 83.5 | 16.67 |
| | branch | 84.1 | 14.89 |
| | `PEREGRINE_SPIN_US=0` | 73.5 | 15.29 |
| | llama.cpp | 95.7 | 12.39 |
| | **branch/base** | **1.008x** | **0.909x** |
| Qwen2.5-0.5B | base | 106.6 | 22.64 |
| | branch | 109.7 | 24.55 |
| | `PEREGRINE_SPIN_US=0` | 114.1 | 25.63 |
| | llama.cpp | 126.7 | 27.52 |
| | **branch/base** | **1.076x** | **1.083x** |
| Gemma-3-1B | base | 52.9 | 13.46 |
| | branch | 53.4 | 12.72 |
| | `PEREGRINE_SPIN_US=0` | 56.1 | 14.96 |
| | llama.cpp | 78.4 | 20.55 |
| | **branch/base** | **1.011x** | **1.101x** |

The quiet condition was not truly quiet (co-tenant load average 4-13), and the round-to-round spread
reflects it. The one sub-1 decode result (Llama-3.2-1B) did not reproduce: a separate 5-round A/B of
the same branch with adaptation on and off (`PEREGRINE_ADAPTIVE_SPIN=0`) gave a median decode ratio
of **1.063x** (prefill **1.000x**). A post-run check of the same signal the pool reads (`/tmp/probe_ratio`) on
these four pinned cores measured 0.6-3.7 % run-queue delay, far below the 35 % first step; on a
host in that state the adaptive branch keeps the configured 1000 µs window. A 200,000-join, 4-worker fork-join
microbenchmark measured **1.05 µs/join on both builds** (medians of five interleaved runs). The
sampling itself is one `/proc/thread-self/schedstat` read per 8 ms per thread, measured at 7.4 µs,
i.e. under 0.1 % of a 4-thread pool's time.

### Loaded (four CPU hogs on the same four cores; two runnable threads per core)

| Model | Engine | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| Llama-3.2-1B | base | 17.5 | 2.14 |
| | branch | 25.7 | 6.14 |
| | `PEREGRINE_SPIN_US=0` | 29.4 | 5.26 |
| | llama.cpp | 50.2 | 7.87 |
| | **branch/base** | **1.479x** | **2.876x** |
| Qwen2.5-0.5B | base | 15.0 | 2.54 |
| | branch | 31.3 | 7.49 |
| | `PEREGRINE_SPIN_US=0` | 32.5 | 8.24 |
| | llama.cpp | 34.7 | 0.53 |
| | **branch/base** | **2.092x** | **2.665x** |
| Gemma-3-1B | base | 22.7 | 5.91 |
| | branch | 17.3 | 3.72 |
| | `PEREGRINE_SPIN_US=0` | 18.4 | 3.78 |
| | llama.cpp | 20.9 | 2.49 |
| | **branch/base** | **0.804x** | **0.628x** |

Fixed-window sweeps on the same host and hogs (interleaved, 2-3 rounds) show why the ladder is
shaped the way it is. With four hogs, Llama-3.2-1B prefill/decode medians were 20.9/3.79 at 1000 µs,
35.5/6.59 at 500 and 26.6/6.05 at 0; with eight, 13.8/1.47 at 1000, 16.2/2.41 at 500 and 19.7/5.08
at 0. Gemma-3-1B under the same four hogs preferred 0 (15.4/3.52) to 500 (10.7/1.90) and 1000
(8.4/1.64) while the host's co-tenant load was higher. That model dependence is the reason the
decision is measured at run time rather than fixed per machine.

### Heavy (eight CPU hogs; three runnable threads per core), Llama-3.2-1B

| Engine | Prefill tok/s | Decode tok/s | Rounds |
|---|---:|---:|---:|
| base (1 ms) | 12.1 | 1.60 | 4 |
| branch (adaptive) | 23.2 | 5.65 | 4 |
| `PEREGRINE_SPIN_US=0` | 22.3 | 4.88 | 1 |
| llama.cpp | 28.9 | 0.41 | 1 |

This is the failure the change exists for: at three runnable threads per core the fixed window
roughly halves prefill and quarters decode, and the adaptive pool finds the zero-spin answer on its
own. The llama.cpp decode figure is one repetition taken while co-tenant load was 12-17; it is
included for completeness, not as a stable comparison.

## Reading the numbers

- Absolute figures are much lower than the quiet-host baselines in `docs/benchmark.md` because this
  session's H0 was continuously shared. Compare same-day ratios, not tok/s across days.
- The adaptive pool is not faster than the best fixed window for a *known* condition; it is close to
  it without knowing the condition in advance.
- Nothing here changes tokens: scheduling only. The native and scalar golden runs under
  `PEREGRINE_GOLDEN=require` passed unchanged.
