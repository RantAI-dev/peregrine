# Benchmarking

A speed claim is a measurement someone else can repeat. This page says how to make one.
The automated benchmark harness in `benchmarks/` (roadmap item **S3.1**) is the source for all
`docs/targets.md` performance numbers from now on.

## The benchmark harness: `benchmarks/run.py`

The top-level harness orchestrates pinned engine builds, runs interleaved rounds, measures the four headline numbers across 512, 4k, and 16k contexts, records the host's hardware ceilings via Peregrine's telemetry probe, and generates Markdown comparison tables conforming to `docs/targets.md`.

### Running benchmarks

```sh
# Full benchmark: interleaved Peregrine vs llama.cpp (3 repeat runs, 8 threads)
python3 benchmarks/run.py \
    --models /path/to/model.gguf \
    --engines peregrine,llama_cpp \
    --contexts 512,4096,16384 \
    --runs 3 \
    --threads 8 \
    --output-dir benchmarks/results

# Quick smoke test on an AVX2 host using a synthetic model (no external weights needed)
python3 benchmarks/run.py --smoke --output-dir benchmarks/results

# Dry-run: inspect build and execution commands without running
python3 benchmarks/run.py --models /path/to/model.gguf --engines peregrine,llama_cpp --dry-run
```

### Generating reports

Convert raw JSON manifests into Markdown tables:

```sh
python3 benchmarks/report.py benchmarks/results/*.json --output report.md
cat report.md
```

## Rules

1. **Same file, same machine** for every engine and both sides of an A/B.
2. **Interleave.** Run A, B, A, B, ... at least 3 times each; never all of A then all of B. Report the median
   and the spread.
3. **Quiet host.** Close other work. On laptops, plug in and fix the power profile. Note anything running.
4. **Each engine at its best thread count**, found by sweeping, not assumed.
5. **Report against the ceiling.** Decode as a fraction of measured sustained bandwidth (Peregrine prints
   bytes per token); prompts and batches as a fraction of the int8 MAC rate.
6. **Record everything:** commit, `rustc -V`, build profile, `EngineOptions` (printed at startup), CPU,
   RAM, OS, competitor versions and their flags.
7. **Publish losses too.**

## Build for benchmarks

```sh
cargo build --release -p peregrine-server     # the profile published numbers use
```

The `quick` profile is for daily work only: it rebuilds about 11× faster and decodes about 0.5% slower
(`benchmarks/results/2026-11-06-h2-quick-profile/`). `dist` (fat LTO) is a candidate under evaluation (S3.9).

## The four numbers

Per model, at 512, 4k and 16k context:

1. decode tok/s, 1 user;
2. prompt tok/s;
3. total tok/s at 4 and 10 users;
4. first-token time at 10 users.

## Single-user benchmarking: `peregrine bench`

Run a clean single-user throughput and bytes/token measurement:

```sh
./target/release/peregrine bench -m path/to/model.gguf [--tokens N] [--prompt-tokens P] [--json]
```

- Prefills a fixed synthetic prompt of `P` tokens (default: 32).
- Decodes `N` greedy tokens (default: 32).
- Prints prefill tok/s, decode tok/s, bytes/token (from telemetry), and the effective options.
- `--json` formats results as a single JSON object for automation.
- Automatically checks for and applies the default tuning profile for the model and host CPU (or `--profile <path>`, `--no-profile`).

## Startup calibration

When no tuning profile exists (`peregrine tune`) and parameters are unpinned, Peregrine dynamically calibrates `decode_threads` and `spin_us` at startup (< 200 ms budget):
- Sweeps candidate decode threads up to physical cores for memory bandwidth saturation, stopping early when bandwidth gains drop below 5%.
- Measures condvar wake-up latency over 50 barrier cycles with 0 µs spin, setting `spin_us = 2 * L_wake` clamped to [50, 2000] µs.
- Aborts to safe defaults (`physical_cores()`, 1000 µs spin) if variance (CV) across repetitions exceeds 15% (noisy host).
- Can be disabled via `PEREGRINE_CALIBRATE=0`.
- Reported in startup logs, `/metrics` Prometheus gauges, and `--json` benchmark outputs under `"config"`.

## Model and machine tuning: `peregrine tune`

Model- and machine-specific tuning is data written by `peregrine tune`, never code (ADR 0005):

```sh
./target/release/peregrine tune -m path/to/model.gguf [--budget-s S] [--out path/to/profile]
```

- Evaluates candidates in isolated child processes (`bench --json --no-profile`) over interleaved rounds to measure the median decode tok/s and spread.
- Performs coordinate search over output-neutral execution knobs (`decode_threads`, `spin_us`, `prefetch_bytes`; the list is `options::TUNABLE`). Activation-layout switches (`PEREGRINE_TQ_SPAN`, `PEREGRINE_TB_SPAN`) change tokens and are never tuned or read from a profile, keeping a change only when it beats the current best by more than the measured noise.
- Respects the time budget (default: 60s), prints a table of tested candidates, and writes a tuning profile.
- Profiles record engine version, CPU model, CPU feature tier, and model file size/hash; any mismatch at runtime ignores the profile and falls back to safe defaults.

## Where results go

Curated results, with the raw JSON and the exact commands, go in `benchmarks/results/<date>-<host>-<topic>/`.
