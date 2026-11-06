//! Qwen3.5-style hybrid models: PrismML's Bonsai 2 (GGUF architecture `qwen35`).

pub mod forward;
pub mod load;
pub mod mtp;

use std::any::Any;

pub use forward::{profile, HyScratch};
pub use load::HybridCfg;

use peregrine_core::arch::{ActLayout, Architecture, Family, Loaded, MtpHead, RecurrentShape};
use peregrine_core::blocks::rotate::Rotation;
use peregrine_core::gguf::{Gguf, TensorInfo};
use peregrine_core::model::{Caches, Config, Linear, State, StepStats, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;

pub struct Qwen35 {
    pub(crate) cfg: HybridCfg,
    pub(crate) layers: Vec<load::HLayer>,
    pub(crate) rot: Rotation,
    pub(crate) v_grouped: bool,
    pub(crate) tok_embd: TensorInfo,
    pub(crate) out_norm: TensorInfo,
    pub(crate) head: TensorInfo,
    pub(crate) mtp: Option<mtp::Mtp>,
}

pub struct Qwen35Family;

impl Family for Qwen35Family {
    fn names(&self) -> &'static [&'static str] {
        &["qwen35"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        load::load(g, tk)
    }
}

impl Architecture for Qwen35 {
    fn recurrent(&self) -> Option<RecurrentShape> {
        Some(RecurrentShape {
            n_layers: self.cfg.n_gdn,
            state_floats: self.cfg.state_floats(),
            conv_floats: self.cfg.conv_floats(),
        })
    }

    fn act_layout(&self) -> ActLayout {
        ActLayout {
            ternary: true,
            ptq1: self.cfg.ptq1 && peregrine_kernels::table::MatKernels::get().ptq1_layout(),
            tb_span: self.cfg.tb_span,
        }
    }

    fn new_scratch(&self, cfg: &Config, max_batch: usize) -> Option<Box<dyn Any + Send + Sync>> {
        Some(Box::new(HyScratch::new(cfg, &self.cfg, max_batch)))
    }

    fn session_kv_rotation(&self) -> bool {
        true
    }

    fn linears_mut(
        &mut self,
        _f: &mut dyn FnMut(&mut Linear) -> Result<(), String>,
    ) -> Result<(), String> {
        // Hybrid models do not use the dense Linear abstraction; reporting no linears
        // prevents a sidecar naming `output.weight` on an untied hybrid from swapping the head.
        Ok(())
    }

    fn linear_groups(&self, _f: &mut dyn FnMut(&[&Linear])) {
        // No LittleBit linear groups for hybrid architectures.
    }

    fn mtp(&self) -> Option<&dyn MtpHead> {
        self.mtp.as_ref().map(|m| m as &dyn MtpHead)
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
        self.forward_hybrid(
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
