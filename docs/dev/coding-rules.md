# Coding rules

Rules for code in this repository. They exist because of failures this engine has had or that engines like
it commonly have. CI enforces what it can (`cargo xclippy`, `cargo fmt`, `cargo deny`); reviewers enforce
the rest. When a rule has to be broken, say so in a comment that gives the reason.

## 1. Correctness before speed

1. **Every fast kernel has a scalar reference** in the same module, and a test comparing them on random
   inputs, edge values (zero blocks, ±max, halfway rounding cases) and odd lengths. Integer kernels match
   bit for bit; floating-point kernels state their tolerance in the test.
2. **No change may alter greedy tokens or KL** against the exact path unless the PR says so, explains why,
   and adds an accuracy report. Lossy features are off by default.
3. **Determinism.** Sampling depends only on (seed, turn, position). Never introduce randomness or
   reductions whose result depends on thread count, batch size or timing.
4. **Prove a speedup before keeping it.** A hot-path change comes with an interleaved A/B
   (`docs/dev/benchmarking.md`) showing the gain on at least one reference host and no loss on another.

## 2. `unsafe`

1. `unsafe` belongs in kernel modules (today `simd.rs`, `tern.rs`, `quant.rs`, `vmath.rs`, `hadamard.rs`;
   after roadmap S1.1, the `peregrine-kernels` crate). New `unsafe` elsewhere needs a reviewer's explicit OK.
2. Every new `unsafe` block has a `// SAFETY:` comment naming the invariant that makes it sound: pointer
   validity, length, alignment, CPU feature, or exclusive access. Existing blocks gain one as their module
   moves into the kernels crate; the `undocumented_unsafe_blocks` lint is then turned on.
3. Functions with `#[target_feature]` are `unsafe fn` and are called only after the runtime feature check
   (through the kernel table once it exists). Never call a feature-gated function behind a compile-time
   `cfg` alone.
4. Sharing a buffer across pool workers uses one helper that hands out disjoint ranges, never raw
   pointers copied into closures (roadmap S1.3). Aliasing bugs have happened here before.

## 3. Performance hygiene

1. No allocation in the per-token path. Scratch space lives in the slot, the pool or a reused buffer.
2. No `std::sync::Mutex` or channel on the per-token path of the engine; the pool's barrier is the
   synchronisation point.
3. Do not rely on the optimiser for inlining in kernels: mark small kernel helpers `#[inline(always)]`
   and check the A/B. (A compiler update once stopped inlining `array::from_fn` in a kernel and cost 4×.)
4. Memory layout is part of the kernel. Repacked layouts are documented beside the code that writes them.
5. Prefer fixed-size chunks (`chunks_exact`, arrays) in kernels so bounds checks disappear; verify with the
   benchmark, not by assumption.

## 4. Errors and configuration

1. Library code returns `Result<_, String>` or a typed error today; it does not panic on bad input files.
   A model the engine cannot run fails at load with a message that names the first unsupported thing
   (architecture, format, tokenizer, template).
2. `panic!` / `unwrap` only for broken internal invariants, never for user input or file contents.
3. Configuration enters through `EngineOptions` (roadmap S0.1). Do not add new `std::env::var` reads in
   engine code; until S0.1 lands, any new switch is listed in the PR.

## 5. Style

1. `cargo fmt` (max width 100) and `cargo clippy -D warnings` with the workspace lints. Crate-level
   `#![allow]`s are kept short and each has a reason.
2. Names say what a thing is: `rows`, `cols`, `n_ctx`, not `a`, `b`, `n`. Units go in names where they are
   not obvious (`_bytes`, `_us`, `_tok_s`).
3. Comments explain **why**, especially for numbers (tile sizes, thresholds, spin windows): where they came
   from and how they were measured. Remove comments that only restate the code.
4. Public items have a doc comment. Modules start with a `//!` paragraph saying what they own.

## 6. Tests

1. A bug fix comes with a test that fails before the fix.
2. Tests that need a real model file skip with a message when it is missing; they never fail for that.
3. Kernel tests run in release mode in CI (`cargo xtest`).
4. Tests do not depend on the machine's thread count, timing or CPU tier unless that is what they test,
   and then they force the tier explicitly.
