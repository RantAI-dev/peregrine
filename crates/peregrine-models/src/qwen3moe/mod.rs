//! The `qwen3moe` family: a mixture of experts with Qwen3's attention (S2.5, ADR 0006).
//!
//! A `qwen3moe` file is a Qwen3 model whose feed-forward is a bank of experts: the router
//! (`ffn_gate_inp`) scores every expert for every row, the best few are chosen, and their
//! feed-forward networks are added together with the router's weights. The attention is
//! Qwen3's — a per-head RMSNorm on the queries and keys, RoPE, grouped-query attention — so
//! the family reuses the dense family's attention path unchanged.
//!
//! The routing and the expert loop are the shared block
//! ([`peregrine_core::blocks::moe`]); this module is the family around it: the loader, the
//! configuration, the forward pass and the scratch.
//!
//! The design note is `docs/design/moe.md`.

pub mod forward;
pub mod load;

use std::any::Any;

pub use forward::MoeState;
pub use load::MoeCfg;

use peregrine_core::arch::{Architecture, Family, Loaded};
use peregrine_core::gguf::{Gguf, TensorInfo};
use peregrine_core::model::{Caches, Config, Linear, State, StepStats, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;

use crate::dense::layer::AttnWeights;

/// One layer: Qwen3's attention and a bank of experts.
pub(crate) struct MoeLayer {
    /// The attention weights: the norm, the projections and the per-head q/k norms.
    pub(crate) attn: AttnWeights,
    /// The feed-forward's norm, the one the router sees.
    pub(crate) ffn_norm: TensorInfo,
    /// The router: `[d_model, n_expert]`, one row of scores per expert.
    pub(crate) router: TensorInfo,
    /// The gate projections, one 2D view per expert: `[d_model, n_ff_exp]`.
    pub(crate) gate: Vec<TensorInfo>,
    /// The up projections, one 2D view per expert: `[d_model, n_ff_exp]`.
    pub(crate) up: Vec<TensorInfo>,
    /// The down projections, one 2D view per expert: `[n_ff_exp, d_model]`.
    pub(crate) down: Vec<TensorInfo>,
}

/// A loaded `qwen3moe` model.
pub struct Qwen3Moe {
    pub(crate) cfg: MoeCfg,
    pub(crate) layers: Vec<MoeLayer>,
    pub(crate) tok_embd: TensorInfo,
    pub(crate) out_norm: TensorInfo,
    pub(crate) head: Linear,
}

/// The family: the `qwen3moe` architecture.
pub struct Qwen3MoeFamily;

impl Family for Qwen3MoeFamily {
    fn names(&self) -> &'static [&'static str] {
        &["qwen3moe"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        load::load(g, tk)
    }
}

impl Architecture for Qwen3Moe {
    fn new_scratch(&self, cfg: &Config, max_batch: usize) -> Option<Box<dyn Any + Send + Sync>> {
        Some(Box::new(MoeState::new(&self.cfg, cfg.d_model, max_batch)))
    }

    fn linears_mut(
        &mut self,
        _f: &mut dyn FnMut(&mut Linear) -> Result<(), String>,
    ) -> Result<(), String> {
        // A LittleBit sidecar must not touch the expert tensors, so this family reports no
        // linears at all: there is nothing in it a sidecar may replace.
        Ok(())
    }

    fn linear_groups(&self, _f: &mut dyn FnMut(&[&Linear])) {
        // No LittleBit linear groups: the experts are never repacked.
    }

    fn forward_rows(
        &self,
        w: &Weights,
        cfg: &Config,
        st: &mut State,
        caches: &mut Caches<'_>,
        pool: &Pool,
        tokens: &[u32],
        positions: &[usize],
        want_logits: &[usize],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        self.forward_moe(
            w,
            cfg,
            st,
            caches,
            pool,
            tokens,
            positions,
            want_logits,
            stats,
        )
    }
}
