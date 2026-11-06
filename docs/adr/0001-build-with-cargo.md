# 0001: Build with cargo; blaze rejected for now

- Status: Accepted
- Date: 2026-11-06

## Context

Blaze (indexable-inc) promises much faster Rust builds from a daemon that keeps the workspace compiled.
We tested blaze 0.20261030.1 on the AVX2 host (H2, bare-metal Linux) against this engine:

- The engine does not build unchanged: blaze has no lowering for the LLVM intrinsic behind
  `_mm512_gf2p8affine_epi64_epi8` (GFNI), in any profile including `--release`.
- With that intrinsic rewritten as inline assembly in a scratch branch, every target compiled and linked,
  in about 20 s from a partly warm store.
- **Blaze then computed wrong results**: `_mm_packs_epi32`, `_mm256_packs_epi32`, `_mm256_packs_epi16` and
  `_mm256_packus_epi16` returned partly garbage lanes in a 20-line probe, so 6 of 112 engine tests failed
  (activation quantisation) in debug and release, while all 112 pass under cargo.
- Cargo on the same host: 4.8 s (library, debug, cold), 10.8 s (tests, debug, cold), 46.1 s (tests, release,
  cold), 66.8 s (server, release, cold), 19.6 s (server, release, after a one-line edit). Blaze's edit-and-test
  gain in debug was about 1.5 s.
- On WSL2, blaze's build-script sandbox also fails unless `BLAZE_SANDBOX=off`.

## Decision

Build, test and release with cargo. Do not depend on blaze in scripts or CI.

To shorten the edit loop within cargo, add a `quick` profile (release optimisation, 256 codegen units,
incremental) for daily work, and keep `release` unchanged for published numbers.

## Consequences

- Correct builds with the toolchain everyone has; no wrong-code risk in kernels.
- Measured (S0.4, `benchmarks/results/2026-11-06-h2-quick-profile/`): after a one-line code edit, `quick`
  rebuilds the server in 2.6 s against 29.6 s for `release`, and decodes about 0.5% slower.
- Revisit when a blaze release supports GFNI and passes the full test suite, including the pack probe.
