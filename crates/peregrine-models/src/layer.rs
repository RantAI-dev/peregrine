//! The layer pieces the dense and the mixture-of-experts families share (ADR 0006).
//!
//! A Qwen3-style layer — the norm, the Q/K/V projections, the per-head q/k RMSNorm, RoPE, the
//! KV write, the attention itself and the output projection — is the same code in the dense
//! family and in the `qwen3moe` family, and it has to stay the same code: both must produce
//! the same bits for the same weights. So it lives here, moved out of the dense forward
//! unchanged, and both families call it.
//!
//! Nothing here decides anything. The weights come from the caller, the arithmetic is the
//! blocks' (`blocks::norm`, `blocks::attn`, `blocks::linear`), and the order is the dense
//! family's.

use peregrine_core::blocks::linear::LinJob;
use peregrine_core::blocks::par::{for_row_chunks, for_row_chunks_2, Disjoint, Sched};
use peregrine_core::blocks::ROW_PARALLEL;
use peregrine_core::gguf::TensorInfo;
use peregrine_core::model::{Config, Linear, State, StepStats, Weights};
use peregrine_core::pool::Pool;

/// The attention weights of one layer.
///
/// The biases are Qwen2's (`None` for Qwen3 and for the mixture of experts), the per-head
/// q/k norms are Qwen3's (`None` for Llama and Qwen2).
#[cfg(any(
    feature = "arch-llama",
    feature = "arch-qwen",
    feature = "arch-qwen3moe",
    feature = "arch-phi3"
))]
pub(crate) struct AttnWeights {
    pub(crate) attn_norm: TensorInfo,
    pub(crate) wq: Linear,
    pub(crate) wk: Linear,
    pub(crate) wv: Linear,
    pub(crate) bq: Option<Vec<f32>>,
    pub(crate) bk: Option<Vec<f32>>,
    pub(crate) bv: Option<Vec<f32>>,
    pub(crate) q_norm: Option<Vec<f32>>,
    pub(crate) k_norm: Option<Vec<f32>>,
    pub(crate) wo: Linear,
}

/// One layer's attention: the norm, the projections, the per-head q/k norms and RoPE, the KV
/// write, the attention, the output projection, and the residual.
///
/// The rows are `nb` wide: `d_model` for the residual, `q_dim` for the attention's output.
#[cfg(any(
    feature = "arch-llama",
    feature = "arch-qwen",
    feature = "arch-qwen3moe",
    feature = "arch-phi3"
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_rows(
    w: &Weights,
    cfg: &Config,
    st: &mut State,
    caches: &mut peregrine_core::model::Caches<'_>,
    pool: &Pool,
    l: &AttnWeights,
    li: usize,
    nb: usize,
    positions: &[usize],
    stats: &mut StepStats,
) {
    // ---- the normed residual ----
    peregrine_core::blocks::norm::norm_rows(
        w,
        &l.attn_norm,
        cfg.d_model,
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
        peregrine_core::prof::time(peregrine_core::prof::Slot::Qkv, || {
            peregrine_core::blocks::linear::linear_group(
                w,
                w.lb(),
                &jobs,
                acts,
                xs,
                cfg.d_model,
                nb,
                lbx,
                lbt,
                pool,
                stats,
                &mut st.kq,
            )
        });
    }
    // ---- the biases, the per-head q/k norms and RoPE ----
    peregrine_core::prof::time(peregrine_core::prof::Slot::Rope, || {
        peregrine_core::blocks::attn::prepare_rows(
            cfg,
            l.bq.as_deref(),
            l.bk.as_deref(),
            l.bv.as_deref(),
            l.q_norm.as_deref(),
            l.k_norm.as_deref(),
            &st.rope,
            &mut st.q,
            &mut st.k,
            &mut st.v,
            nb,
            positions,
            pool,
            stats,
        );
    });
    let qd = cfg.q_dim();
    peregrine_core::prof::time(peregrine_core::prof::Slot::KvWrite, || {
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
    });

    let t_attn = std::time::Instant::now();
    peregrine_core::prof::time(peregrine_core::prof::Slot::Attn, || {
        peregrine_core::blocks::attn::attend_mapped(
            cfg,
            &mut st.attn,
            &st.q,
            &mut st.xb,
            caches,
            pool,
            li,
            nb,
            positions,
            stats,
        );
    });
    stats.ns_attn += t_attn.elapsed().as_nanos() as u64;

    // ---- the attention's output, projected back into the residual ----
    {
        let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
        peregrine_core::prof::time(peregrine_core::prof::Slot::Quant, || {
            for_row_chunks_2(
                pool,
                parallel,
                &mut st.xb,
                qd,
                &mut st.acts,
                1,
                nb,
                Sched::Static,
                |_wk, rows, xb, acts| {
                    for i in 0..rows.len() {
                        acts[i].fill(&xb[i * qd..(i + 1) * qd]);
                    }
                },
            );
        });
        if parallel {
            stats.barriers += 1;
        }
    }
    {
        let lbx = Disjoint::new(&mut st.lb_x);
        let lbt = Disjoint::new(&mut st.lb_t);
        let acts = &st.acts[..nb];
        let jobs = [LinJob {
            lin: &l.wo,
            out: Disjoint::new(&mut st.xb2),
        }];
        peregrine_core::prof::time(peregrine_core::prof::Slot::Wo, || {
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
            )
        });
    }
    {
        let xb2: &[f32] = &st.xb2;
        let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
        peregrine_core::prof::time(peregrine_core::prof::Slot::Resid, || {
            for_row_chunks(
                pool,
                parallel,
                &mut st.x,
                cfg.d_model,
                nb,
                Sched::Static,
                |_wk, rows, xc| {
                    let base = rows.start * cfg.d_model;
                    for (i, x) in xc.iter_mut().enumerate() {
                        *x += xb2[base + i];
                    }
                },
            );
        });
        if parallel {
            stats.barriers += 1;
        }
    }
}

/// The output head: the final norm and the projection into the vocabulary, for the rows the
/// caller asked for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn head_rows(
    w: &Weights,
    cfg: &Config,
    st: &mut State,
    out_norm: &TensorInfo,
    head: &Linear,
    nb: usize,
    want_logits: &[usize],
    pool: &Pool,
    stats: &mut StepStats,
) -> Result<(), String> {
    let nl = want_logits.len();
    if nl > st.acts.len() || nl * cfg.n_vocab > st.logits.len() {
        return Err(format!(
            "logits requested for {nl} rows; scratch holds {} activations and {} logit rows",
            st.acts.len(),
            st.logits.len() / cfg.n_vocab
        ));
    }
    for (i, &b) in want_logits.iter().enumerate() {
        if b >= nb {
            return Err(format!("logits requested for row {b} of a {nb}-row batch"));
        }
        let (xs, xe) = (b * cfg.d_model, (b + 1) * cfg.d_model);
        let (os, oe) = (i * cfg.d_model, (i + 1) * cfg.d_model);
        let (xv, xbv) = (&st.x, &mut st.xb);
        peregrine_core::prof::time(peregrine_core::prof::Slot::Norm, || {
            peregrine_core::blocks::norm::rmsnorm(
                w,
                out_norm,
                cfg.eps,
                &xv[xs..xe],
                &mut xbv[os..oe],
                stats,
            )
        });
        peregrine_core::prof::time(peregrine_core::prof::Slot::Quant, || {
            st.acts[i].fill(&st.xb[os..oe])
        });
    }
    let before = stats.ns_matmul;
    if let Some(ti) = head.dense() {
        // S4.9d: a label row reads only its token ids. The borrows are the state's
        // own fields, which the head touches and nothing else does at once.
        let State {
            acts,
            logits,
            label_rows,
            label_ids,
            label_out,
            kq,
            ..
        } = &mut *st;
        peregrine_core::blocks::linear::output_rows(
            w,
            ti,
            &acts[..nl],
            want_logits,
            logits,
            label_rows,
            label_ids,
            label_out,
            pool,
            stats,
            kq,
        );
    } else {
        // A LittleBit head has no single-row kernel here: compute every row and
        // gather the label ids out of it (the same values, just not the saving).
        let lbx = Disjoint::new(&mut st.lb_x);
        let lbt = Disjoint::new(&mut st.lb_t);
        let acts = &st.acts[..nl];
        let jobs = [LinJob {
            lin: head,
            out: Disjoint::new(&mut st.logits),
        }];
        peregrine_core::prof::time(peregrine_core::prof::Slot::Head, || {
            peregrine_core::blocks::linear::linear_group(
                w,
                w.lb(),
                &jobs,
                acts,
                &st.xb,
                cfg.d_model,
                nl,
                lbx,
                lbt,
                pool,
                stats,
                &mut st.kq,
            )
        });
        for &(row, start, count) in &st.label_rows {
            let base = row * cfg.n_vocab;
            for (k, &id) in st.label_ids[start..start + count].iter().enumerate() {
                st.label_out[start + k] = st.logits[base + id as usize];
            }
        }
    }
    stats.ns_head += stats.ns_matmul - before;
    stats.ns_matmul = before;
    Ok(())
}

/// Write every row's final normed hidden state into `State::hidden` when the caller asked for
/// it (`State::want_hidden`): the input an MTP block reads and the rows an embedding pools.
///
/// This is exactly llama.cpp's `result_norm` tensor (`src/models/llama.cpp`: `cur =
/// build_norm(cur, output_norm, ...)` over every row of the batch), so a vector pooled from
/// these rows is pooled from the same values llama.cpp pools. The norm is the same call the
/// output head makes, and it does not change `x` or the logits: asking for hidden states
/// leaves another slot's bits untouched. No bytes are accounted here, as with the hybrid
/// family's own write (the head accounts the norm it reads).
pub(crate) fn write_hidden(
    w: &Weights,
    cfg: &Config,
    st: &mut State,
    out_norm: &TensorInfo,
    nb: usize,
) {
    if !st.want_hidden {
        return;
    }
    let d = cfg.d_model;
    if st.hidden.len() < nb * d {
        st.hidden.resize(nb * d, 0.0);
    }
    for b in 0..nb {
        let (xs, xe) = (b * d, (b + 1) * d);
        peregrine_core::blocks::norm::rmsnorm_into(
            w,
            out_norm,
            cfg.eps,
            &st.x[xs..xe],
            &mut st.hidden[xs..xe],
        );
    }
}

/// Copy the residual stream after layer `li` into the tap buffer, when the caller asked for
/// that layer (what a DSpark drafter conditions on).
pub(crate) fn copy_tap(st: &mut State, cfg: &Config, li: usize, nb: usize) {
    if let Some(t) = st.tap_layers.iter().position(|&x| x == li) {
        let (d, nt) = (cfg.d_model, st.tap_layers.len());
        for b in 0..nb {
            let dst = (b * nt + t) * d;
            st.tap[dst..dst + d].copy_from_slice(&st.x[b * d..(b + 1) * d]);
        }
    }
}
