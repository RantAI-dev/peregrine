# Build profiles and thread pinning (roadmap S3.9)

- Date: 2026-11-10
- Host: AMD Ryzen AI 9 HX 370 (12 cores / 24 SMT threads, AVX-512 + VNNI + GFNI + VBMI), 12 GB,
  Ubuntu 24.04 under WSL2 (Microsoft hypervisor, kernel 6.18.33.2), `rustc 1.99.1` (the pinned
  toolchain), cargo with sccache and mold.
- Question: does the release binary get faster with fat LTO, one codegen unit and PGO, and does pinning
  the pool's threads to physical cores help decode and prefill here?

The short answer: **no to both, as a default.** Fat LTO buys 29 % of the binary and nothing on the
stopwatch; PGO moves decode by +1 to +2 % and costs Qwen2.5 prefill 10–20 %; pinning is worth +1 to +4 %
on a warm prefill and −6 to −9 % on the first prefill in a process. The release profile is unchanged,
`dist` stays the packaging profile, and pinning is available as `PEREGRINE_AFFINITY=1`.

## Method

Binary variants from the same tree (`origin/main`, `1c93345`), each built with
`cargo build --locked -p peregrine -p peregrine-server -j 12` on that host:

| arm | profile / flags | build (wall) | `peregrine` | `peregrine-server` |
| --- | --- | --- | --- | --- |
| `release` | the tree's `[profile.release]` (opt-level 3, cgu 16) | 29 s | 7 087 896 B | 6 617 448 B |
| `dist` | `lto = "fat"`, `codegen-units = 1` | 1 m 42 s | 5 002 400 B (−29.4 %) | 4 695 008 B (−29.1 %) |
| `dist` + `panic = "abort"` | as `dist` plus abort | 1 m 30 s | 4 340 896 B (−38.7 %) | 4 089 784 B (−38.2 %) |
| `dist` + PGO | `-Cprofile-use=pgo.profdata` | 1 m 32 s | 4 570 264 B (−35.5 %) | 4 421 496 B (−36.7 %) |

Build times are warm-cache (sccache); the unoptimised-library work is identical in all four, so the
ratios are what matters: fat LTO plus one codegen unit costs about 3.5× the link/codegen time.

Runtime ISA dispatch is untouched: no `target-cpu=native` in any shipped profile. A `native` build was
measured for reference only (5 210 032 B) and is not reported as a candidate.

The PGO profile was trained with an instrumented `dist` build (`-Cprofile-generate`) over
`peregrine bench --prompt-tokens 512 --tokens 32` on the three models below plus a four-slot serve run,
then merged with `llvm-profdata merge` (37 547 functions, 60 billion counts). Note that instrumentation
is *very* expensive for the k-quant kernels: Llama-3.2-1B's 512-token prefill runs at 9.5 tok/s
instrumented against ~330 tok/s plain, so the training run is a ~5 minute job and its timing is not
representative — only its counts are.

Models (SHA-256 of the exact files):

| model | file | sha256 |
| --- | --- | --- |
| Llama-3.2-1B-Instruct-Q4_K_M | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | `6f85a640a97cf2bf…` |
| Qwen2.5-0.5B-Instruct-Q4_K_M | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | `6eb923e7d26e9cea…` |
| Ternary-Bonsai-2-2B-PQ2_0 | `Ternary-Bonsai-2-2B-PQ2_0.gguf` | `d5f01a4ecaf49239…` |

The Qwen file above is the one in `models/`. The desktop benchmark directory names a file with the
same name and size but different bytes (`74a4da8c…`), which also misses the stored AVX-512 golden, so
the two are not interchangeable in a golden comparison.

Protocol: `peregrine bench`, one process per sample, arms rotated every round, medians of per-round
paired ratios against `release`. Two workloads: 512-token prompt / 48 decode tokens (9 rounds) and
4096-token prompt / 32 decode tokens (6 rounds). Defaults otherwise: 24 workers, 12 for the one-row
(decode) pool, prefill chunk 64.

The 512-token prefill of a *single* process is the noisiest number on this host: repeated runs of one
binary span ±15 %, because the first prefill in a process pays the KV/scratch first touch. The 4096
workload and decode are tight (±3 %). A second, low-noise instrument was therefore used as well: a
probe that loads the model once and repeats the bench workload seven times in-process, at 24 workers
(4 processes per arm) and at 12 workers (3 processes per arm), dropping the cold first pass.

## Speed

`peregrine bench`, 24 workers, 512 / 48 — ratio to `release` in brackets:

| model | release | dist | dist + PGO |
| --- | --- | --- | --- |
| Llama-3.2-1B | 325.9 / 68.69 | 348.7 / 68.32 (×1.070 / ×0.995) | 318.4 / 68.88 (×0.977 / ×1.003) |
| Qwen2.5-0.5B | 475.9 / 103.66 | 485.6 / 104.38 (×1.020 / ×1.007) | 383.1 / 100.76 (×0.805 / ×0.972) |
| Bonsai-2-2B | 272.9 / 76.80 | 270.1 / 76.54 (×0.990 / ×0.997) | 282.0 / 79.08 (×1.033 / ×1.030) |

`peregrine bench`, 24 workers, 4096 / 32:

| model | release | dist | dist + PGO |
| --- | --- | --- | --- |
| Llama-3.2-1B | 269.2 / 54.63 | 276.5 / 54.77 (×1.027 / ×1.003) | 269.2 / 55.62 (×1.000 / ×1.018) |
| Qwen2.5-0.5B | 445.8 / 81.39 | 422.7 / 81.27 (×0.948 / ×0.999) | 400.7 / 83.49 (×0.899 / ×1.026) |
| Bonsai-2-2B | 301.3 / 65.56 | 306.1 / 66.11 (×1.016 / ×1.008) | 296.6 / 71.11 (×0.985 / ×1.085) |

(prefill tok/s / decode tok/s; the arms are `release`, fat LTO, and fat LTO with PGO.)

Warm in-process probe, 24 workers — prefill / decode, ratio to `release`:

| model | dist | dist + PGO |
| --- | --- | --- |
| Llama-3.2-1B | ×1.006 / ×1.000 | ×0.975 / ×1.008 |
| Qwen2.5-0.5B | ×0.981 / ×1.004 | ×0.835 / ×1.015 |
| Bonsai-2-2B | ×1.017 / ×0.999 | ×0.944 / ×1.016 |

Warm in-process probe, 12 workers (no SMT in play):

| model | dist | dist + PGO |
| --- | --- | --- |
| Llama-3.2-1B | ×1.011 / ×1.014 | ×0.978 / ×1.004 |
| Qwen2.5-0.5B | ×0.997 / ×1.011 | ×0.963 / ×1.023 |
| Bonsai-2-2B | ×0.998 / ×0.994 | ×0.989 / ×0.982 |

Reading: the hot loops are already AVX-512 intrinsics in leaf functions, and every model above is
memory-bound at these thread counts, so cross-crate inlining and profile-guided layout have little to
rearrange. `dist` is neutral — the only number that could be called a win (the 512-token Llama prefill
at ×1.070) is the noisy metric and does not reproduce at 4096 or warm. PGO is a mixed bag that is
consistently *bad* for Qwen2.5 prefill (×0.805, ×0.899, ×0.835, ×0.963 in four independent
measurements) and mildly good for decode (+1 to +2 % median, +8.5 % on Bonsai at 4096). Rejected: a
build path that needs a two-stage build, a five-minute instrumented run and a per-toolchain profile
file, to trade a Qwen prompt regression for a ~1 % decode gain, is not worth its maintenance.

`panic = "abort"` on top of `dist` was measured on the 512 workload only (3 rounds): prefill 338 vs 334
tok/s and decode 66.7 vs 68.6 tok/s — no speed signal, −13 % more binary. Rejected because it turns a
panic in the server into a process abort for no gain.

Goldens: `PEREGRINE_GOLDEN=require` on every arm gives identical output for Llama-3.2-1B and
Ternary-Bonsai-2-2B, and recording the goldens with each arm gives byte-identical `.golden` files for
all three models — LTO and PGO do not reassociate anything. (Qwen2.5 mismatches the *stored* AVX-512
golden on every arm, including an untouched `release` build and before any of this work: the stored
file was recorded from the other Qwen2.5-0.5B file, `74a4da8c…`.)

## Thread pinning

`PEREGRINE_AFFINITY=1` pins each pool worker, and the caller (worker 0), to a logical CPU chosen from
`/proc/cpuinfo`'s `(physical id, core id)` topology: worker `w` of a pool `n` wide takes core
`w % cores` and, once the pool is wider than the core count, the `w / cores`-th thread of that core.
A pool as wide as the physical cores therefore puts one worker per core; the 24-worker pool fills the
SMT siblings in a fixed order. Off by default; on Linux only; no other platform changes.

Same protocol (off vs on, one binary), `peregrine bench`:

| workload | Llama-3.2-1B | Qwen2.5-0.5B | Bonsai-2-2B |
| --- | --- | --- | --- |
| 512 / 48, cold single prefill, 9 rounds: prefill | ×0.913 | ×0.925 | ×0.943 |
| 512 / 48: decode | ×1.004 | ×1.002 | ×1.029 |
| 4096 / 32, 4 rounds: prefill | ×0.991 | ×1.018 | ×1.014 |
| 4096 / 32: decode | ×0.998 | ×1.009 | ×1.018 |

Warm in-process probe, 24 workers, two runs of 3–4 processes (prefill / decode):

| model | run 1 | run 2 |
| --- | --- | --- |
| Llama-3.2-1B | ×1.040 / ×0.995 | ×1.024 / ×1.020 |
| Qwen2.5-0.5B | ×1.026 / ×1.003 | ×1.124 / ×1.021 |
| Bonsai-2-2B | ×1.044 / ×1.015 | ×1.012 / ×0.998 |

So: pinning helps a warm prefill by 1–4 % (12 % on Qwen in one run — the run-to-run spread on the
512-token prefill is wide enough that only the sign is trustworthy), does nothing for decode, and
*loses* 6–9 % on the first 512-token prefill a process runs. The last one is the number
`peregrine bench` reports, so the option is off by default and documented as a server knob, not a
benchmark knob. The mechanism is assumed, not verified: with every worker fixed to one CPU, the first
pass's page faults for the KV cache and scratch buffers are all served by the pinned workers, while
the scheduler can spread them when the threads are free to move.

On a single-socket host, first-touch placement does not matter for these working sets: every pool
thread touches its own output rows and the weights are read-only and shared. On a multi-socket host it
does, and this prototype is wrong there — it ignores NUMA nodes, so a worker can be pinned to a core
whose memory is remote from the model's pages. A NUMA-aware mapping (one core set per node, memory
bound with `numactl --membind`) is the next step if a multi-socket machine joins the fleet.

## Decisions

1. `[profile.release]` stays as it is (opt-level 3, `codegen-units = 16`): no measured speed win from
   fat LTO, and 3.5× the build time is a real cost to every `cargo xtest` and every CI job.
2. `[profile.dist]` stays the packaging profile: same speed, 29 % smaller binaries, and release
   artifacts are built once.
3. PGO is not adopted. The recipe is below so the next attempt starts from the numbers.
4. `PEREGRINE_AFFINITY=1` ships as an off-by-default knob with the caveats above.

## Recipe (PGO, for the next attempt)

```sh
# 1. instrument
rustup component add llvm-tools
RUSTFLAGS="-Cprofile-generate=$PWD/pgo-data" \
  cargo build --profile dist -p peregrine -p peregrine-server
# 2. train: the workload above, plus a four-slot serve run (a server killed with
#    SIGTERM writes no profile, so drive the engine in-process or shut it down cleanly)
mkdir -p pgo-data
for m in Llama-3.2-1B-Instruct-Q4_K_M Qwen2.5-0.5B-Instruct-Q4_K_M \
         bonsai2-2b/Ternary-Bonsai-2-2B-PQ2_0; do
  target/dist/peregrine bench -m models/$m.gguf --prompt-tokens 512 --tokens 32
done
# 3. merge, then rebuild with the profile
llvm-profdata merge -o pgo.profdata pgo-data/*.profraw
RUSTFLAGS="-Cprofile-use=$PWD/pgo.profdata" \
  cargo build --profile dist -p peregrine -p peregrine-server
```

`llvm-profdata` comes from `llvm-tools` for the toolchain in `rust-toolchain.toml`; the profile is
toolchain- and target-specific, so it is a build artifact and is not committed.
