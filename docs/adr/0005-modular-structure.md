# 0005: Modular structure by layer, features for optional parts, runtime CPU dispatch

- Status: Accepted (planning; implementation not started)
- Date: 2026-11-06

## Context

The engine specialises on four levels. Today they are mixed inside the model code:

| Level | Today | Problem |
|---|---|---|
| CPU type (ISA) | AVX2 / AVX-512 VNNI / GFNI / VBMI kernel tiers, chosen at run time | 14 feature checks inside `model.rs` and `model/hybrid.rs` |
| Weight format | separate kernels for PQ2_0, PTQ1_0, Q4_K, Q6_K, Q8_0, ... | 9 format branches inside model code |
| Architecture | dense path (llama, qwen2, qwen3) vs `hybrid.rs` (qwen35), MTP, Hadamard rotation | 17 `if` branches; the hybrid forward is a separate 1.6k-line path |
| Specific model | no code for a named model; tuned constants (tile spans, spin window) measured on one host | defaults may be wrong on other machines |

Breadth (more formats, architectures, CPU families) costs edits in several places while levels 1-3 stay mixed.
Embedders (such as RantAI products) want small builds with only their model family; end users want one
binary that runs fast on any CPU.

## Decision

### 1. Specialisation rule

- Optimise for a **CPU type, a weight format, an architecture or a tensor shape**, never for a named model.
  Shape-specialised kernels (for example, head size divisible by 64) are chosen by the kernel table and
  benefit every model with that shape.
- Model- and machine-specific **tuning is data, not code**: `peregrine tune` measures tile sizes, thread
  count and spin window on the host for a model and writes a profile file. Safe defaults always apply
  without one.

### 2. Crates: four to start

```
peregrine           facade: re-exports, feature pass-through, the CLI (`peregrine run|serve|bench|tune|inspect`)
                    and the HTTP server (module or binary target)
peregrine-models    architectures, one module per family, each behind a feature; MTP heads
peregrine-core      runtime: EngineOptions, pool, cache, prefix store, sampler, scheduler (Engine), drafter
                    interface, telemetry, shared layer blocks, the `Architecture` interface;
                    GGUF / safetensors readers and the tokenizer + templates as modules
peregrine-kernels   (format × ISA) kernels, scalar references, repacking, the kernel table; all SIMD `unsafe`
```

Plus `peregrine-bench` (harness and experiments), which nothing depends on.

Dependencies point down only: kernels ← core ← models ← peregrine. More crates are created only when a
boundary has a proven need: the tokenizer or GGUF reader become their own crates if someone needs them alone.

### 3. What is modular, and how

| Axis | Mechanism |
|---|---|
| Architectures | Cargo features in `peregrine-models` (`arch-llama`, `arch-qwen`, `arch-qwen35`, `arch-gemma`, `arch-moe`, ...) behind one `Architecture` interface |
| Weight formats | Cargo features in `peregrine-kernels` (`fmt-kquant`, `fmt-ternary`, `fmt-iq`, ...) |
| Extras | Cargo features: `mtp`, `dspark`, `littlebit`, `server` |
| **CPU types** | **not a feature**: one binary carries every tier, selected at run time by the kernel table. An optional feature may strip tiers for small embedded builds |
| Model / machine tuning | profile files from `peregrine tune` |

Default features build everything, so the official binary needs no choices. Embedders use
`default-features = false` and list what they need.

### 4. Guards

| Risk | Guard |
|---|---|
| Lost inlining across crate boundaries (a missed inline once cost 4× decode) | The kernel table dispatches at whole-kernel granularity, where an indirect call is negligible; small cross-crate helpers are `#[inline]`; every code move carries a before/after benchmark |
| Feature combinations that do not build | CI runs `cargo hack --each-feature` plus the full default build |
| API lock-in | The `Architecture` interface and feature names are unstable until 1.0; third-party architectures are a later goal |
| Stale or foreign tuning profiles | A profile records CPU model and features, Peregrine version and the model file's hash; any mismatch ignores it and uses safe defaults |
| Behaviour change during refactor | Each step keeps greedy tokens and KL identical (`ARCHITECTURE.md` invariants), checked by the synthetic and real-model tests |

## Alternatives considered

- **A general tensor-graph library (the ggml approach).** Architectures become graphs of generic ops, so new
  models are cheap. Rejected for now: the hand-fused forwards save memory traffic and make bit-identical
  output straightforward, and those are the product. Shared layer blocks recover most of the reuse.
  Revisit if the number of architectures passes about ten.
- **Runtime-generated kernels per CPU and shape (oneDNN, TVM).** Highest ceiling, much more complexity.
  Not now; it could plug into the kernel table later if hand-written kernels reach their limit.
- **Eight fine-grained crates from the start.** Rejected: boundaries not yet proven, and the overhead in
  manifests, versions and build time comes before the benefit.
- **Compile-time ISA selection (one build per CPU).** Rejected: users do not know their CPU features, and
  one download must run well everywhere.

## Consequences

- Adding a format, CPU tier or architecture touches one place (see `docs/dev/structure-rules.md`).
- `unsafe` SIMD code is isolated in one crate, which is easier to review, fuzz and test against references.
- Some upfront refactoring cost (roadmap Stage 1, about 5-6 person-weeks) before breadth work.
- Feature flags add a CI dimension, covered by `cargo hack`.

## Order of implementation (to be discussed before starting)

1. `peregrine-kernels` with the kernel table: removes the CPU checks and format branches from model code.
2. The `Architecture` interface and shared layer blocks in `peregrine-core`; dense and hybrid forwards
   rewritten as two families in `peregrine-models`.
3. Features, the facade crate, the CLI.
4. `peregrine tune` and profile files.
