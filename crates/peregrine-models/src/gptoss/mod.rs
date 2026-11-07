//! The `gpt-oss` family: OpenAI's mixture-of-experts model (S2.5, ADR 0006).
//!
//! A `gpt-oss` file is a transformer whose feed-forward is a bank of experts and
//! whose attention is unlike any other model here:
//!
//! * **attention sinks** — a learned logit per query head, added to the softmax's
//!   denominator without a value of its own (`src/models/openai-moe.cpp:42`,
//!   `ggml/src/ggml-cpu/ops.cpp:5883-5894`);
//! * **alternating sliding-window and full attention** — every other layer sees
//!   only the last `attention.sliding_window` positions
//!   (`src/models/openai-moe.cpp:6-9`);
//! * **biases** on the attention's output projection, on the router and on each
//!   expert's three projections (`src/models/openai-moe.cpp:49-54`);
//! * **YaRN** rotary scaling (`src/models/openai-moe.cpp:11-13`).
//!
//! The experts themselves are a bank like the `qwen3moe` family's — the same
//! three-dimensional tensors, read as two-dimensional views — but three things
//! differ: the router's weights are a softmax over the *selected* experts rather
//! than over all of them, the activation is OpenAI's `swiglu_oai`, and every
//! projection carries a bias.
//!
//! The routing and the expert loop are the shared block
//! ([`peregrine_core::blocks::moe`]); this module is the family around it: the
//! loader, the configuration, the forward pass and the scratch.
//!
//! The design note is `docs/design/moe.md`.

pub mod forward;
pub mod load;

use std::any::Any;

pub use forward::GptOssState;
pub use load::GptOssCfg;

use peregrine_core::arch::{ActLayout, Architecture, Family, Loaded};
use peregrine_core::blocks::moe::ExpertBias;
use peregrine_core::gguf::Format;
use peregrine_core::gguf::{Gguf, TensorInfo};
use peregrine_core::model::{Caches, Config, Linear, State, StepStats, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;

/// One layer: the attention, the norm the experts see, and the bank of experts.
pub(crate) struct GptOssLayer {
    /// The attention's norm.
    pub(crate) attn_norm: TensorInfo,
    /// The query projection.
    pub(crate) wq: Linear,
    /// The key projection.
    pub(crate) wk: Linear,
    /// The value projection.
    pub(crate) wv: Linear,
    /// The attention's output projection.
    pub(crate) wo: Linear,
    /// The attention output projection's bias (`attn_output.bias`).
    pub(crate) wo_bias: Option<Vec<f32>>,
    /// The attention sinks: one learned logit per query head.
    pub(crate) sinks: Option<Vec<f32>>,
    /// The norm the mixture of experts sees (`post_attention_norm`).
    pub(crate) post_norm: TensorInfo,
    /// The router: `[d_model, n_expert]`, one row of scores per expert.
    pub(crate) router: TensorInfo,
    /// The router's bias, one value per expert.
    pub(crate) router_bias: Option<Vec<f32>>,
    /// The gate projections, one 2D view per expert: `[d_model, n_ff_exp]`.
    pub(crate) gate: Vec<TensorInfo>,
    /// The up projections, one 2D view per expert: `[d_model, n_ff_exp]`.
    pub(crate) up: Vec<TensorInfo>,
    /// The down projections, one 2D view per expert: `[n_ff_exp, d_model]`.
    pub(crate) down: Vec<TensorInfo>,
    /// The experts' biases, one set per expert.
    pub(crate) expert_bias: Option<Vec<ExpertBias>>,
}

/// A loaded `gpt-oss` model.
pub struct GptOss {
    pub(crate) cfg: GptOssCfg,
    pub(crate) layers: Vec<GptOssLayer>,
    pub(crate) tok_embd: TensorInfo,
    pub(crate) out_norm: TensorInfo,
    pub(crate) head: Linear,
}

/// The family: the `gpt-oss` architecture.
pub struct GptOssFamily;

impl Family for GptOssFamily {
    fn names(&self) -> &'static [&'static str] {
        &["gpt-oss"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        load::load(g, tk)
    }
}

impl Architecture for GptOss {
    /// F32/F16/BF16 matrices multiply the f32 activations, which the activation rows must then keep.
    fn act_layout(&self) -> ActLayout {
        let float = |t: &TensorInfo| matches!(t.kind, Format::F32 | Format::F16 | Format::BF16);
        let lin = |l: &Linear| l.dense().is_some_and(float);
        let keep_f32 = lin(&self.head)
            || self.layers.iter().any(|l| {
                [&l.wq, &l.wk, &l.wv, &l.wo].into_iter().any(lin)
                    || float(&l.router)
                    || l.gate.iter().chain(&l.up).chain(&l.down).any(float)
            });
        ActLayout {
            keep_f32,
            ..ActLayout::default()
        }
    }

    fn new_scratch(&self, cfg: &Config, max_batch: usize) -> Option<Box<dyn Any + Send + Sync>> {
        Some(Box::new(GptOssState::new(
            &self.cfg,
            cfg.d_model,
            max_batch,
            self.act_layout().keep_f32,
        )))
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
        self.forward_gpt_oss(
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
