# Structure rules

How the code is divided, and the rules that keep breadth cheap: a new weight format, architecture or CPU
family should touch one place. The current state and the target layout are in `ARCHITECTURE.md`.

## Layers (ADR 0005)

```
peregrine           facade · CLI (run, serve, bench, tune, inspect) · HTTP server
      │ uses
peregrine-models    architectures, one module per family, each behind a feature · MTP heads
      │ uses
peregrine-core      runtime (options, pool, cache, sampler, scheduler, drafters, telemetry)
                    · shared layer blocks · `Architecture` interface · GGUF reader · tokenizer
      │ uses
peregrine-kernels   (format × ISA) kernels · scalar references · repacking · kernel table · all SIMD unsafe

peregrine-bench     harness and experiments; depends on the others, nothing depends on it
```

1. **Dependencies point down only.** `kernels` knows nothing about models; `core` knows nothing about a
   specific architecture or about HTTP; nothing depends on `bench`.
2. **The engine crates take no new runtime dependency without an ADR.** The facade's server and CLI and
   `bench` may use well-tested crates for cold-path work (HTTP, JSON, CLI, logging).
3. **A new crate needs a proven boundary.** Start as a module; split when something outside needs it alone.
4. **Specialise by CPU type, weight format, architecture or tensor shape, never by a named model.**
   Model- and machine-specific tuning is a profile file from `peregrine tune`, not code.
5. **CPU tiers are chosen at run time** by the kernel table, never by a Cargo feature in the default build.
6. **Optional parts are Cargo features** (`arch-*`, `fmt-*`, `mtp`, `dspark`, `littlebit`, `server`);
   the default build enables everything, and CI checks each feature alone (`cargo hack --each-feature`).

## Adding things

| To add | Touch | Never |
|---|---|---|
| A weight format | `gguf::Format`, one kernel module in `peregrine-kernels` with its scalar reference, the kernel-table entry, its `fmt-*` feature, a synthetic test model | model or architecture code |
| A CPU tier (e.g. AVX-VNNI-256, NEON) | kernels for each format that benefits, the table's feature detection, a CI job (native or emulated) | model code; scattered `is_x86_feature_detected!` |
| An architecture | one module in `peregrine-models` implementing `Architecture` from the shared layer blocks, its `arch-*` feature, a synthetic test model, a check against llama.cpp | the kernels; another architecture's forward |
| A tokenizer or template | `tokenizer.rs` / `template.rs` and their tests against llama.cpp's token output | the model |
| A speculation method | one implementation of the drafter interface (roadmap S1.4) | the engine's scheduling loop |
| An engine switch | a field in `EngineOptions` with a doc comment and a default | a new `std::env::var` read |
| A tuned value for one machine or model | a field in the tuning profile, measured by `peregrine tune` | a hard-coded constant or a check on the model's name |

### Choosing features for a new format or architecture

- **Weight formats (`peregrine-kernels`)**:
  - K-quants (Q2_K..Q6_K) belong to `fmt-kquant`.
  - Ternary formats (PQ2_0, PTQ1_0) belong to `fmt-ternary`.
  - Importance-matrix non-linear quants (IQ4_NL, IQ4_XS) belong to `fmt-iq`.
  - Legacy block formats (Q4_0, Q5_0, Q8_0) belong to `fmt-legacy`.
  - Unquantized floats (F32, F16, BF16) are core formats and always enabled (no feature required).
  - A novel format family (e.g. MXFP, NVFP4) gets a new `fmt-<family>` feature declared in `peregrine-kernels` and passed through `peregrine-core`, `peregrine-models`, and `peregrine`.
  - Update `Format::required_feature()` and `Format::is_supported()` in `peregrine-kernels/src/format.rs` so disabled formats are refused at load time with the feature name.

- **Architectures (`peregrine-models`)**:
  - Dense transformers get an `arch-<family>` feature (e.g. `arch-llama` for Llama, `arch-qwen` for Qwen2/Qwen3, `arch-gemma` for Gemma). If models share architecture internals but differ in identity, split by name if users would build them independently.
  - Hybrid or novel architectures get dedicated features (e.g. `arch-qwen35`, `arch-moe`).
  - Declare the feature in `peregrine-models` and pass through `peregrine`.
  - Gate the family registration in `FAMILIES` and `ARCHS` under `#[cfg(feature = "arch-<family>")]`.
  - Update `check_compatible_with` in `peregrine-core/src/lib.rs` and `Model::load_with` in `peregrine-core/src/model.rs` to refuse disabled architectures with `architecture ... is not enabled in this build (recompile with feature 'arch-<name>')`.

- **Support table (`README.md`)**:
  - After adding a format/arch, run `peregrine support --markdown` and paste between `<!-- support:begin -->` and `<!-- support:end -->` in `README.md`. CI validates this in the lint job.

## Files and modules

1. A module owns one job and says so in its `//!` header.
2. Files over ~1,500 lines are a smell; split by responsibility (loader, state, forward), not alphabetically.
3. Architecture-specific code lives in `peregrine-models/src/<family>`; shared blocks in `peregrine-core` (after S1.2).
4. Experiments and one-off benchmarks go in `peregrine-bench`, not in `examples/` of a library.

## Repository

```
crates/           workspace crates (all share [workspace.package] and [workspace.lints])
docs/             product, targets, roadmap, adr/, dev/, reference/
benchmarks/       harness scripts and curated results (results/), never scratch output
.github/          CI, issue forms, PR template, Dependabot
```

- Versions of every dependency live in the root `Cargo.toml` (`[workspace.dependencies]`).
- Machine-specific settings (linker, sccache, jobs) live in each developer's `~/.cargo/config.toml`, not in
  the repo's `.cargo/config.toml`.
- Decisions that are expensive to reverse get an ADR in `docs/adr/`.
