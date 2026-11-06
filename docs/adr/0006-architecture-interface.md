# 0006: The `Architecture` interface and shared layer blocks

- Status: Accepted
- Date: 2026-11-06

## Context

Roadmap S1.2 splits the model code (`model.rs`, 2.3k lines; `model/hybrid.rs`, 1.6k; `model/mtp.rs`) into
shared layer blocks plus one module per architecture family, first inside `peregrine-core` (S1.2a), then in
a `peregrine-models` crate (S1.2b, ADR 0005).

What the rest of the engine needs from a model is narrow. `Engine`, `Session` and the server use `cfg`, `tk`,
`rope_factors`, `forward` / `forward_rows`, `new_cache`, `is_hybrid`, `has_mtp` / `mtp_*`, `lb_scratch` /
`lb_summary`, `name`, `gguf`, `weight_bytes`, plus `State` and `Caches`. DSpark and suffix speculation use
none of it directly. One piece of architecture knowledge sits in the runtime: `Session` turns off the
rotated int8 KV cache for Qwen2 (`Engine` does not).

Constraints:

- **Bit-identical output.** The golden tests hash every prefill logit's bits per kernel tier. A refactor may
  not change one floating-point operation's order. Several look alike and are not: `head_rmsnorm` sums
  serially in f64 while `rmsnorm_into` uses the two-accumulator `sumsq_f64`; `(x·inv)·w` versus
  `x *= inv·w`; three different dot products; SwiGLU split points (the AVX2 `swiglu` ends with a scalar
  `exp`, so where the rows split changes bits); attention planning depends on the batch size, so the
  hybrid's one-row attention calls and MTP's last-row call must keep their shape.
- **No allocation per token** (invariant 4), no new dependency, no speed loss: release builds have no LTO
  and 16 codegen units, so small helpers that cross a crate boundary must be `#[inline]`.
- After S1.2b, `peregrine-core` cannot name the families.

## Decision

### Dispatch: a trait object behind `Model`

`Model` stays a concrete type in `peregrine-core` with its current methods, so callers do not change. It
holds the weights and a `Box<dyn Architecture>`. The cost is one indirect call per forward pass.

Rejected: a generic `Engine<A: Architecture>` (the server picks the family at run time, so it needs type
erasure anyway, and Engine, Session and the server would be compiled once per family); an enum of families
at the boundary (core could not name the variants after S1.2b).

```rust
pub trait Architecture: Send + Sync + 'static {
    fn forward_rows(&self, w: &Weights, cfg: &Config, st: &mut State, caches: &mut Caches<'_>,
                    pool: &Pool, tokens: &[u32], positions: &[usize], want_logits: &[usize],
                    stats: &mut StepStats) -> Result<(), String>;
    fn recurrent(&self) -> Option<RecurrentShape> { None }      // new_cache, is_hybrid
    fn act_layout(&self) -> ActLayout { ActLayout::default() }  // activation quantisation the kernels expect
    fn new_scratch(&self, cfg: &Config, max_batch: usize) -> Option<Box<dyn Any + Send>> { None }
    fn session_kv_rotation(&self) -> bool { true }              // Qwen2: false (Session only, as today)
    fn linears_mut(&mut self, f: &mut dyn FnMut(&mut Linear) -> Result<(), String>) -> Result<(), String>;
    fn linear_groups(&self, f: &mut dyn FnMut(&[&Linear]));     // the forward's fusion groups (LittleBit scratch)
    fn mtp(&self) -> Option<&dyn MtpHead> { None }
}

pub trait MtpHead: Send + Sync {
    fn new_state(&self, w: &Weights, cfg: &Config, max_ctx: usize, threads: usize) -> Result<MtpState, String>;
    fn forward(&self, w: &Weights, cfg: &Config, ms: &mut MtpState, pool: &Pool, tokens: &[u32],
               hidden: &[f32], positions: &[usize], draft: bool, stats: &mut StepStats)
               -> Result<Option<u32>, String>;
}

pub trait Family: Sync {
    fn names(&self) -> &'static [&'static str];                 // `general.architecture` values
    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String>;
}
```

`Model::load` picks the family from a table of `&dyn Family` (in core during S1.2a, in `peregrine-models`
after S1.2b); `check_compatible` reads the same table. The input checks every forward shares stay in
`Model::forward_rows`, before the call into the family.

### Shared blocks: free functions over slices

`peregrine-core/src/blocks/` holds the layer computations both families use: `linear` (the matmul groups,
LittleBit's arm), `norm`, `rope`, `attn` (KV write, attention runs), `ffn`, `head`, `embed`, `rotate`
(Hadamard), `recurrent` (Gated DeltaNet steps). They take slices, row counts and scratch, never `&mut State`
or `&Model`. Where two families compute "the same" thing in a different float order, both versions stay,
named for what they do; unifying them is a separate PR that says it changes output.

`blocks/par.rs` holds every raw-pointer wrapper used to hand worker threads disjoint output rows (today
`OutPtr`, `SharedBuf`, `SharedActs`, `SharedScratch`, `SharedMl`, `RowPtrs`, `SnapPtrs`) behind a few
generic `par_rows`-style drivers. S1.3 then replaces their insides with one checked helper without touching
the families.

### State

Scratch the families need lives in `State`, sized once: the shared buffers as today, attention scratch
grouped as `AttnScratch`, LittleBit's as `LbScratch`, and the family's own (`HyScratch` today) as an opaque
`Box<dyn Any + Send>` from `new_scratch`, taken by split borrow, never by `mem::take`. `Model::new_state`
builds it and applies `rope_factors` and the LittleBit sizes, so a caller can no longer forget either.

### Order (each step keeps every golden bit-identical; steps 5 and 7 carry a decode A/B)

| Step | What |
|---|---|
| 0 | More synthetic coverage first: qwen2, rope scaling, batching invariance through `Engine`, LittleBit, MTP |
| 1 | No allocation per token (a counting-allocator test enforces it); 1b: the Gated DeltaNet head kernel moves to `peregrine-kernels` |
| 2 | The traits, `Weights`, `blocks/` and `par.rs` skeleton, scratch grouping (no behaviour change) |
| 3 | Blocks extracted from `model.rs` (3a) and `hybrid.rs` (3b), with `#[inline]` forwarding shims |
| 4 | `families/dense` (4a) and `families/hybrid` with MTP (4b) implement the traits on the blocks |
| 5 | Shims removed; `model.rs` becomes the facade; family table dispatch |
| 6 | `Config` loses architecture knowledge (`Arch` moves into the dense family); `Model::new_state` |
| 7 | `peregrine-models` crate: the families move (S1.2b) |
| 8 | Loading wired through `peregrine-models` (`open`, server, bench, tests) |

## Consequences

- A new architecture is a new family module built from blocks, plus one table entry.
- The model code loses its `unsafe` except inside `blocks/par.rs` (and S1.3 removes most of that).
- `Config` and `State` stop being architecture-specific; the Qwen2 Session/Engine asymmetry becomes an
  explicit trait method instead of an `arch ==` test. Making the two agree changes output and is its own decision.
- Costs: one `dyn` call per forward and one downcast of the family scratch per forward (both negligible);
  about 2 person-weeks; the interface is unstable until 1.0 (ADR 0005).
- Revisit if a family needs per-layer dispatch through the trait (it should use blocks instead), or if
  the trait object shows in a profile.
