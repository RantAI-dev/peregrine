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

The `quick` profile is for daily work only. `dist` (fat LTO) is a candidate under evaluation (S3.9).

## The four numbers

Per model, at 512, 4k and 16k context:

1. decode tok/s, 1 user;
2. prompt tok/s;
3. total tok/s at 4 and 10 users;
4. first-token time at 10 users.

## Where results go

Curated results, with the raw JSON and the exact commands, go in `benchmarks/results/<date>-<host>-<topic>/`.
