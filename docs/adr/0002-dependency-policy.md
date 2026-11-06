# 0002: Own the hot path; audited crates for the cold path

- Status: Accepted
- Date: 2026-11-06

## Context

The engine's speed and correctness live in code that runs per token: kernels, memory layout, threading,
the cache, sampling, the tokenizer and the scheduler. No library has kernels for PQ2_0 or PTQ1_0, and
general frameworks (BLAS, oneDNN, ggml, rayon) hide the thread and barrier counts and the bytes moved,
which this engine measures and reports. The server, by contrast, does HTTP, JSON, CLI and logging once per
request, where a mature crate is safer than our own code (HTTP parsing is a classic source of bugs).

## Decision

- `peregrine-core` and the future `peregrine-kernels` have no runtime dependency besides `memmap2`.
  Adding one needs an ADR that shows why owning the code is worse.
- `peregrine-server` and `peregrine-bench` may use well-maintained crates for cold-path work.
- `cargo deny` runs in CI: permissive licenses only, crates.io only, RustSec advisories are errors, and
  rayon / BLAS crates are banned.

## Consequences

Full control and visibility of every per-token cycle, at the cost of writing kernels ourselves, which is
the project's purpose anyway.
