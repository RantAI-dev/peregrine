//! The `qwen3moe` forward pass and its scratch (S2.5).
//!
//! The layer is the dense family's attention, unchanged, followed by the mixture-of-experts
//! block: the router scores every expert for the row, the best few are chosen, and their
//! feed-forward networks are added together with the router's weights, in slot order.
//!
//! The block's buffers live in a [`MoeState`], sized once per state, so a decode step
//! allocates nothing (invariant 4).

use std::time::Instant;

use crate::dense::layer;
use crate::qwen3moe::{load::MoeCfg, Qwen3Moe};
use peregrine_core::arch::Architecture;
use peregrine_core::blocks::moe::{moe_ffn, MoeScratch, MoeWeights};
use peregrine_core::model::{Caches, Config, State, StepStats, Weights};
use peregrine_core::pool::Pool;

/// The family's scratch: the mixture-of-experts block's buffers.
///
/// Sized once, from the state's batch width, and reused by every layer and every step.
pub struct MoeState {
    scratch: MoeScratch,
    /// The rows the scratch was sized for.
    rows: usize,
    /// Whether the expert activations keep their f32 values (float expert matrices).
    keep_f32: bool,
}

impl MoeState {
    /// The scratch for a state that will run batches of `rows` rows.
    pub fn new(cfg: &MoeCfg, d_model: usize, rows: usize, keep_f32: bool) -> MoeState {
        let mut scratch =
            MoeScratch::new(rows, cfg.n_expert, cfg.n_expert_used, d_model, cfg.n_ff_exp);
        scratch.set_keep_f32(keep_f32);
        MoeState {
            scratch,
            rows,
            keep_f32,
        }
    }

    /// The scratch for `nb` rows, grown when the state was built for fewer.
    fn ensure(&mut self, cfg: &MoeCfg, d_model: usize, nb: usize) -> &mut MoeScratch {
        if nb > self.rows {
            self.scratch =
                MoeScratch::new(nb, cfg.n_expert, cfg.n_expert_used, d_model, cfg.n_ff_exp);
            self.scratch.set_keep_f32(self.keep_f32);
            self.rows = nb;
        }
        &mut self.scratch
    }
}

impl Qwen3Moe {
    /// The forward pass: embed, then every layer's attention and experts, then the head.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_moe(
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
        let nb = tokens.len();
        let t_fwd = Instant::now();
        for (i, &t) in tokens.iter().enumerate() {
            peregrine_core::blocks::embed::embed(
                &self.tok_embd,
                w,
                t,
                &mut st.x[i * cfg.d_model..(i + 1) * cfg.d_model],
                stats,
            );
        }

        // The scratch, sized once. A state built by `State::new` (not `Model::new_state`)
        // has none, so it gets one here.
        if st.fam.is_none() {
            st.fam = Some(Box::new(MoeState::new(
                &self.cfg,
                cfg.d_model,
                st.max_batch,
                self.act_layout().keep_f32,
            )));
        }

        for (li, l) in self.layers.iter().enumerate() {
            // ---- attention: Qwen3's, the dense family's own path ----
            layer::attention_rows(w, cfg, st, caches, pool, &l.attn, li, nb, positions, stats);

            // ---- the experts ----
            // The router sees the normed residual; the block gathers the rows that chose each
            // expert, runs that expert's three matrices over them, and adds the results up in
            // slot order, so a row's result cannot depend on the batch it was in.
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.ffn_norm,
                cfg.d_model,
                cfg.eps,
                &st.x,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );
            {
                let state = st
                    .fam
                    .as_mut()
                    .and_then(|f| f.downcast_mut::<MoeState>())
                    .ok_or("the state has no mixture-of-experts scratch")?;
                let scratch = state.ensure(&self.cfg, cfg.d_model, nb);
                let mw = MoeWeights {
                    router: &l.router,
                    gate: &l.gate,
                    up: &l.up,
                    down: &l.down,
                    n_used: self.cfg.n_expert_used,
                    renorm: self.cfg.renorm,
                    w_scale: self.cfg.w_scale,
                };
                moe_ffn(
                    w,
                    &mw,
                    &st.xb[..nb * cfg.d_model],
                    &mut st.xb2[..nb * cfg.d_model],
                    nb,
                    scratch,
                    pool,
                    stats,
                );
            }
            for i in 0..nb * cfg.d_model {
                st.x[i] += st.xb2[i];
            }
            layer::copy_tap(st, cfg, li, nb);
        }

        // ---- output head ----
        if !want_logits.is_empty() {
            layer::head_rows(
                w,
                cfg,
                st,
                &self.out_norm,
                &self.head,
                nb,
                want_logits,
                pool,
                stats,
            )?;
        }
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }
}
