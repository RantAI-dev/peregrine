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

Kernels are chosen at run time from the CPU's features, capped by `PEREGRINE_ISA` (`scalar`, `avx2`,
`avx512`; default `native`). Every feature check goes through `cpu_has!` (`crates/peregrine-core/src/cpu.rs`),
so a cap reaches every kernel.

```sh
PEREGRINE_ISA=scalar cargo xtest    # portable code only
PEREGRINE_ISA=avx2   cargo xtest    # the AVX2 tier, also on an AVX-512 machine
```

Goldens are stored per effective tier (`cpu::tier_name()`), so a forced run is compared with that tier's
goldens. Two machines must agree on a shared tier: the AVX-512 host forced to `avx2` reproduces the AVX2
host's goldens.

CI runs the native tier on x86-64 Linux, aarch64 Linux and aarch64 macOS, and the `scalar` and `avx2` tiers
on x86-64 Linux.
