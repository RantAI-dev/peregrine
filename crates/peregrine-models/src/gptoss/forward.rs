//! The `gpt-oss` forward pass and its scratch (S2.5).
//!
//! The layer is llama.cpp v0.9.0's (`src/models/openai-moe.cpp:75-152`):
//!
//! 1. the attention's norm, the queries, keys and values, the rotation, the
//!    attention itself (with its sinks, and with the sliding window on every other
//!    layer), and the output projection with its bias;
//! 2. the residual, and the norm the experts see (`attn_post_norm`);
//! 3. the mixture of experts: the router scores every expert, the best few are
//!    chosen, a softmax over their scores gives their weights, and their
//!    feed-forward networks are added together with those weights, in slot order.
//!
//! The block's buffers live in a [`GptOssState`], sized once per state, so a decode
//! step allocates nothing (invariant 4).

use std::time::Instant;

use crate::gptoss::{load::GptOssCfg, GptOss};
use peregrine_core::arch::Architecture;
use peregrine_core::blocks::linear::LinJob;
use peregrine_core::blocks::moe::{moe_ffn, MoeAct, MoeGating, MoeScratch, MoeWeights};
use peregrine_core::blocks::par::Disjoint;
use peregrine_core::model::{Caches, Config, State, StepStats, Weights};
use peregrine_core::pool::Pool;

/// The family's scratch: the mixture-of-experts block's buffers.
///
/// Sized once, from the state's batch width, and reused by every layer and every step.
pub struct GptOssState {
    scratch: MoeScratch,
    /// The rows the scratch was sized for.
    rows: usize,
    /// Whether the expert activations keep their f32 values (float expert matrices).
    keep_f32: bool,
}

impl GptOssState {
    /// The scratch for a state that will run batches of `rows` rows.
    pub fn new(cfg: &GptOssCfg, d_model: usize, rows: usize, keep_f32: bool) -> GptOssState {
        let mut scratch =
            MoeScratch::new(rows, cfg.n_expert, cfg.n_expert_used, d_model, cfg.n_ff_exp);
        scratch.set_keep_f32(keep_f32);
        GptOssState {
            scratch,
            rows,
            keep_f32,
        }
    }

    /// The scratch for `nb` rows, grown when the state was built for fewer.
    fn ensure(&mut self, cfg: &GptOssCfg, d_model: usize, nb: usize) -> &mut MoeScratch {
        if nb > self.rows {
            self.scratch =
                MoeScratch::new(nb, cfg.n_expert, cfg.n_expert_used, d_model, cfg.n_ff_exp);
            self.scratch.set_keep_f32(self.keep_f32);
            self.rows = nb;
        }
        &mut self.scratch
    }
}

impl GptOss {
    /// The forward pass: embed, then every layer's attention and experts, then the head.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_gpt_oss(
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
        let dm = cfg.d_model;
        let qd = cfg.q_dim();
        let t_fwd = Instant::now();
        for (i, &t) in tokens.iter().enumerate() {
            peregrine_core::blocks::embed::embed(
                &self.tok_embd,
                w,
                t,
                &mut st.x[i * dm..(i + 1) * dm],
                stats,
            );
        }

        // The scratch, sized once. A state built by `State::new` (not `Model::new_state`)
        // has none, so it gets one here.
        if st.fam.is_none() {
            st.fam = Some(Box::new(GptOssState::new(
                &self.cfg,
                dm,
                st.max_batch,
                self.act_layout().keep_f32,
            )));
        }

        for (li, l) in self.layers.iter().enumerate() {
            // ---- the attention's norm ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.attn_norm,
                dm,
                cfg.eps,
                &st.x,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );
            // ---- the queries, keys and values ----
            {
                let lbx = Disjoint::new(&mut st.lb_x);
                let lbt = Disjoint::new(&mut st.lb_t);
                let acts = &st.acts[..nb];
                let xs: &[f32] = &st.xb;
                let jobs = [
                    LinJob {
                        lin: &l.wq,
                        out: Disjoint::new(&mut st.q),
                    },
                    LinJob {
                        lin: &l.wk,
                        out: Disjoint::new(&mut st.k),
                    },
                    LinJob {
                        lin: &l.wv,
                        out: Disjoint::new(&mut st.v),
                    },
                ];
                peregrine_core::blocks::linear::linear_group(
                    w,
                    w.lb(),
                    &jobs,
                    acts,
                    xs,
                    dm,
                    nb,
                    lbx,
                    lbt,
                    pool,
                    stats,
                    &mut st.kq,
                );
            }
            // ---- the rotation (no biases, no per-head norms) ----
            peregrine_core::blocks::attn::prepare_rows(
                cfg, None, None, None, None, None, &st.rope, &mut st.q, &mut st.k, &mut st.v, nb,
                positions, pool, stats,
            );
            peregrine_core::blocks::attn::write_kv_rows(
                caches,
                li,
                &st.k,
                &st.v,
                nb,
                positions,
                cfg.n_kv_head,
                cfg.head_dim,
            );
            // ---- the attention itself: the sinks, and the sliding window on every
            //      other layer ----
            let window = if self.cfg.is_swa[li] {
                Some(self.cfg.sliding_window)
            } else {
                None
            };
            let t_attn = Instant::now();
            for (r0, len, c) in caches.runs(nb) {
                peregrine_core::blocks::attn::attend_scaled_window(
                    cfg,
                    &mut st.attn,
                    &st.q,
                    &mut st.xb,
                    caches.cache(c),
                    pool,
                    li,
                    len,
                    positions[r0],
                    r0,
                    window,
                    None,
                    l.sinks.as_deref(),
                    stats,
                );
            }
            stats.ns_attn += t_attn.elapsed().as_nanos() as u64;
            // ---- the attention's output, projected back into the residual ----
            for b in 0..nb {
                st.acts[b].fill(&st.xb[b * qd..(b + 1) * qd]);
            }
            {
                let lbx = Disjoint::new(&mut st.lb_x);
                let lbt = Disjoint::new(&mut st.lb_t);
                let acts = &st.acts[..nb];
                let jobs = [LinJob {
                    lin: &l.wo,
                    out: Disjoint::new(&mut st.xb2),
                }];
                peregrine_core::blocks::linear::linear_group(
                    w,
                    w.lb(),
                    &jobs,
                    acts,
                    &st.xb,
                    qd,
                    nb,
                    lbx,
                    lbt,
                    pool,
                    stats,
                    &mut st.kq,
                );
            }
            // The attention output's bias (`src/llama-graph.cpp:3217-3219`).
            if let Some(bias) = &l.wo_bias {
                for b in 0..nb {
                    for (x, &v) in st.xb2[b * dm..(b + 1) * dm].iter_mut().zip(bias) {
                        *x += v;
                    }
                }
            }
            for i in 0..nb * dm {
                st.x[i] += st.xb2[i];
            }

            // ---- the norm the experts see, and the experts ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.post_norm,
                dm,
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
                    .and_then(|f| f.downcast_mut::<GptOssState>())
                    .ok_or("the state has no mixture-of-experts scratch")?;
                let scratch = state.ensure(&self.cfg, dm, nb);
                let mw = MoeWeights {
                    router: &l.router,
                    gate: &l.gate,
                    up: &l.up,
                    down: &l.down,
                    n_used: self.cfg.n_expert_used,
                    // gpt-oss does not renormalise the selected experts' weights: the
                    // softmax over the selection is the normalisation
                    // (`src/models/openai-moe.cpp:139`, `norm_w` is false).
                    renorm: false,
                    w_scale: self.cfg.w_scale,
                    // The best few by score, then a softmax over them
                    // (`src/models/openai-moe.cpp:141`).
                    gating: MoeGating::SoftmaxWeight,
                    // OpenAI's `swiglu_oai` (`src/models/openai-moe.cpp:139`).
                    act: MoeAct::SwiGLUOai {
                        alpha: self.cfg.swiglu_alpha,
                        limit: self.cfg.swiglu_limit,
                    },
                    router_bias: l.router_bias.as_deref(),
                    expert_bias: l.expert_bias.as_deref(),
                };
                moe_ffn(
                    w,
                    &mw,
                    &st.xb[..nb * dm],
                    &mut st.xb2[..nb * dm],
                    nb,
                    scratch,
                    pool,
                    stats,
                );
            }
            for i in 0..nb * dm {
                st.x[i] += st.xb2[i];
            }
            crate::layer::copy_tap(st, cfg, li, nb);
        }

        // ---- the output head ----
        crate::layer::write_hidden(w, cfg, st, &self.out_norm, nb);
        if !want_logits.is_empty() {
            crate::layer::head_rows(
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
