# Testing

```sh
cargo xtest       # = cargo test --workspace --release --locked
cargo xclippy     # = clippy on every target, warnings are errors
```

Tests run in release mode: kernels are tested as they ship, and debug-mode SIMD tests are slow.

## Three kinds of test

1. **Kernel tests** (unit tests in the kernel modules, `tests/simd.rs`, `tests/tern.rs`): each fast kernel
   against its scalar reference. They need no model files and always run.
2. **Synthetic model tests** (roadmap S0.2): tiny random GGUFs for every architecture and format, compared
   against the exact reference forward. They will run in CI on every kernel tier.
3. **Real model tests** (`tests/engine.rs`, `batch.rs`, `bonsai.rs`, `oracle.rs`, `dspark.rs`, ...): real
   GGUF files, checking tokens against reference outputs. They skip, with a message, when the file is
   missing. They are the release gate.

## Where tests look for model files

| File | Location |
|---|---|
| SmolLM2-360M-Instruct Q4_K_M, Qwen3-4B Q4_K_M | `$PEREGRINE_MODEL`, or `~/models/` |
| Qwen3-1.7B Q4_K_M | `models/` at the repository root |
| Bonsai 2 9B PQ2_0 | `$PEREGRINE_BONSAI9`, or `models/bonsai2-9b/` |
| Bonsai 2 4B PQ2_0 with MTP | `$PEREGRINE_BONSAI4_MTP`, or `models/bonsai2-4b/` |

`models/` is git-ignored. These locations will be unified under one variable as part of S0.1.

## CPU tiers

Kernels are chosen by the CPU's features at run time, so a test run only covers the tier of the machine it
runs on. Until the forced-tier switch (S0.3) exists:

- the AVX2 tier is covered on an AVX2-only host (H2 in `docs/targets.md`);
- the AVX-512 tiers are covered on an AVX-512 host (H1, Zen 4/5 or recent Xeon).

CI runs x86-64 Linux, aarch64 Linux and aarch64 macOS.
