# Benchmarking

A speed claim is a measurement someone else can repeat. This page says how to make one. The full
harness is roadmap item S3.1; until it exists, follow the same method by hand.

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

## Model and machine tuning: `peregrine tune`

Model- and machine-specific tuning is data written by `peregrine tune`, never code (ADR 0005):

```sh
./target/release/peregrine tune -m path/to/model.gguf [--budget-s S] [--out path/to/profile]
```

- Evaluates candidates in isolated child processes (`bench --json --no-profile`) over interleaved rounds to measure the median decode tok/s and spread.
- Performs coordinate search over output-neutral execution knobs (`decode_threads`, `spin_us`, `prefetch_bytes`, `tb_span`, `tq_span`), keeping a change only when it beats the current best by more than the measured noise.
- Respects the time budget (default: 60s), prints a table of tested candidates, and writes a tuning profile.
- Profiles record engine version, CPU model, CPU feature tier, and model file size/hash; any mismatch at runtime ignores the profile and falls back to safe defaults.

## Where results go

Curated results, with the raw JSON and the exact commands, go in `benchmarks/results/<date>-<host>-<topic>/`.
