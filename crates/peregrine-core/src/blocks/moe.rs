//! The mixture-of-experts block: the router and the expert feed-forward network.
//!
//! A MoE layer is a router matrix and a bank of experts. For every row the router scores all the
//! experts, the best `k` are chosen, and the row's output is the weighted sum of those experts'
//! feed-forward networks. This module owns the routing arithmetic (llama.cpp v0.9.0's
//! `build_moe_ffn`, `src/llama-graph.cpp`) and the expert loop; the matmuls are the ordinary dense
//! blocks (`linear::matmul_group`, `ffn::swiglu_rows`), so a mixture of experts needs no new kernel.
//!
//! Two properties matter beyond the arithmetic:
//!
//! * **A row never sees the batch.** The router is per row, the experts are gathered per row, and
//!   the weighted sum runs in slot order, so a row's result is the same whether it is decoded alone
//!   or beside seventeen others (invariant 1 in `ARCHITECTURE.md`).
//! * **Nothing is allocated per token.** [`MoeScratch`] is sized once and reused; every buffer the
//!   block needs lives in it.
//!
//! Roadmap S2.5: the `qwen3moe` family (step 3) is the first caller. Nothing calls this yet.

// Nothing in this module is called yet: the `qwen3moe` family (S2.5 step 3) is the first caller,
// and until it lands every item here is dead code. The allow goes with that step.
#![allow(dead_code)]

use crate::blocks::ffn::swiglu_rows;
use crate::blocks::linear::{matmul_group, MatJob};
use crate::blocks::par::OutPtr;
use crate::gguf::TensorInfo;
use crate::model::{StepStats, Weights};
use crate::pool::Pool;
use crate::quant::QuantAct;

/// The smallest normal f16 (`2^-14`). llama.cpp clamps the sum of the selected experts' weights to
/// it, so a row whose experts all score near zero cannot divide by zero
/// (`src/llama-graph.cpp:2381`).
const MIN_F16_NORMAL: f32 = 6.103_515_6e-5;

/// The router's choice for one row: which experts, and how much each one counts.
///
/// This is llama.cpp v0.9.0's gating, step for step (`src/llama-graph.cpp`):
///
/// 1. `logits = ffn_gate_inp · x` — the caller does that matmul and hands the logits over
///    (`src/llama-graph.cpp:2261`).
/// 2. `probs = softmax(logits)` over **all** the experts, in f32: max-subtract, `expf`, sum,
///    divide (`src/llama-graph.cpp:2279`; ggml's `ggml_compute_forward_soft_max_f32`,
///    `ggml/src/ggml-cpu/ops.cpp:5797`).
/// 3. the best `k` by probability (`src/llama-graph.cpp:2345`), with `w_i = probs[selected_i]` in
///    that order (`src/llama-graph.cpp:2359`).
/// 4. with `renorm`, the weights are divided by their own sum, which is clamped to the smallest
///    f16 normal (`src/llama-graph.cpp:2381-2385`).
///
/// **Ties.** ggml's argsort compares values only and sorts with `std::sort`, which is not stable
/// (`ggml/src/ggml-cpu/ops.cpp:8698-8746`), so llama.cpp's choice among equal router scores is
/// implementation defined. Peregrine fixes one: **probability descending, then expert index
/// ascending**. The rule is per row and never looks at the batch, which is what invariant 1 needs
/// (`docs/design/moe.md` §1).
///
/// ggml's vector `exp` is a polynomial approximation worth about an ulp
/// (`ggml/src/ggml-cpu/vec.h`), so a probability can differ from llama.cpp's in the last bit.
/// The selection is unaffected by that, except at a tie, which the rule above settles anyway.
///
/// `out_ids` and `out_w` are the caller's buffers and must hold at least `k` entries, so the decode
/// path allocates nothing.
pub(crate) fn route(
    router_logits: &[f32],
    k: usize,
    renorm: bool,
    out_ids: &mut [u32],
    out_w: &mut [f32],
) {
    let n = router_logits.len();
    assert!(
        k > 0 && k <= n,
        "a router must choose between 1 and {n} experts, not {k}"
    );
    assert!(
        out_ids.len() >= k && out_w.len() >= k,
        "the route buffers hold fewer than {k} entries"
    );

    // 2. the softmax, in ggml's order: the maximum, then the exponentials, then the sum.
    let mut max = f32::NEG_INFINITY;
    for &l in router_logits {
        if l > max {
            max = l;
        }
    }
    // ggml sums the exponentials in double (`ggml_vec_soft_max_f32`,
    // `ggml/src/ggml-cpu/vec.cpp:625`), divides in double and then scales the f32 exponentials by
    // that quotient (`ggml/src/ggml-cpu/ops.cpp:5888-5896`).
    let mut sum = 0.0f64;
    for &l in router_logits {
        sum += f64::from((l - max).exp());
    }
    let inv = (1.0 / sum) as f32;

    // 3. the best k, probability descending and expert index ascending on a tie. The list is
    //    kept sorted in the caller's buffers, so no scratch is needed.
    let mut n_sel = 0usize;
    for (i, &l) in router_logits.iter().enumerate() {
        let p = (l - max).exp() * inv;
        let id = i as u32;
        // The first entry that is worse than this one; equal probabilities keep the lower index
        // first, because `better` is false for an equal probability and a higher index.
        let mut pos = 0usize;
        while pos < n_sel && !better(p, id, out_w[pos], out_ids[pos]) {
            pos += 1;
        }
        if pos >= k {
            continue;
        }
        // Shift the tail right, dropping the last entry when the list is full.
        let mut j = n_sel.min(k - 1);
        while j > pos {
            out_w[j] = out_w[j - 1];
            out_ids[j] = out_ids[j - 1];
            j -= 1;
        }
        out_w[pos] = p;
        out_ids[pos] = id;
        if n_sel < k {
            n_sel += 1;
        }
    }

    // 4. the renormalisation, over the selected experts in slot order.
    if renorm {
        // ggml's `ggml_sum_rows` sums the row in double and rounds once
        // (`ggml_vec_sum_f32`, `ggml/src/ggml-cpu/vec.h:1634`).
        let mut s = 0.0f64;
        for w in out_w.iter().take(n_sel) {
            s += f64::from(*w);
        }
        // The clamp to the smallest f16 normal (`src/llama-graph.cpp:2381`), then the division.
        let s = (s as f32).max(MIN_F16_NORMAL);
        for w in out_w.iter_mut().take(n_sel) {
            *w /= s;
        }
    }
}

/// Whether `(p_a, id_a)` outranks `(p_b, id_b)`: a higher probability, or the same probability and
/// the lower expert index.
#[inline(always)]
fn better(p_a: f32, id_a: u32, p_b: f32, id_b: u32) -> bool {
    p_a > p_b || (p_a == p_b && id_a < id_b)
}

/// The 2D matrix of one expert of a 3D expert tensor.
///
/// GGUF stores `dims[0]` contiguous, so llama.cpp's expert tensors (`[d_in, d_out, n_expert]`,
/// `src/llama-arch.cpp:535-538`) are expert-major: expert `e` is a contiguous run of `d_out` rows
/// of `d_in` weights — exactly a 2D `[d_in, d_out]` matrix in the layout the matmul group wants
/// (`blocks/linear.rs`). `Gguf::bytes` and `matmul_group` read the view as they read any other
/// matrix, so an expert needs no new kernel.
///
/// The expert's weights must be a whole number of blocks, or the view would not start on a block
/// boundary; a file that fails this is refused (the caller turns the error into a load failure).
/// The view is never repacked into the tile store: the LittleBit sidecar must not touch the expert
/// tensors.
pub(crate) fn expert_view(t: &TensorInfo, e: usize) -> Result<TensorInfo, String> {
    if t.dims.len() != 3 {
        return Err(format!(
            "{}: an expert tensor has three dimensions, not {}",
            t.name,
            t.dims.len()
        ));
    }
    let n_expert = t.dims[2] as usize;
    if e >= n_expert {
        return Err(format!("{}: expert {e} of {n_expert}", t.name));
    }
    let (cols, rows) = (t.dims[0], t.dims[1]);
    let per_expert = cols * rows;
    let block = t.kind.block() as u64;
    if per_expert % block != 0 {
        return Err(format!(
            "{}: an expert holds {per_expert} weights, not a whole number of {} blocks",
            t.name,
            t.kind.label()
        ));
    }
    let bytes = per_expert / block * t.kind.block_bytes() as u64;
    Ok(TensorInfo {
        name: format!("{}[{e}]", t.name),
        dims: vec![cols, rows],
        kind: t.kind,
        offset: t.offset + e as u64 * bytes,
        tile_off: None,
    })
}

/// The weights of one mixture-of-experts layer: the router and the per-expert matrices.
///
/// The three expert banks are the 2D views of the file's 3D tensors, one per expert
/// ([`expert_view`]), so the block reads an expert's weights as an ordinary matrix.
pub(crate) struct MoeWeights<'a> {
    /// The router: `[d_model, n_expert]`, one row of scores per expert.
    pub(crate) router: &'a TensorInfo,
    /// The gate projections, one 2D view per expert: `[d_model, n_ff]`.
    pub(crate) gate: &'a [TensorInfo],
    /// The up projections, one 2D view per expert: `[d_model, n_ff]`.
    pub(crate) up: &'a [TensorInfo],
    /// The down projections, one 2D view per expert: `[n_ff, d_model]`.
    pub(crate) down: &'a [TensorInfo],
    /// How many experts a row uses.
    pub(crate) n_used: usize,
    /// Whether the selected experts' weights are renormalised over the selection (qwen3moe: yes,
    /// `src/models/qwen3moe.cpp:144`).
    pub(crate) renorm: bool,
}

/// The buffers a mixture-of-experts layer needs, sized once.
///
/// Every buffer is written before it is read and reused by the next layer, so a decode step
/// allocates nothing (invariant 4). The sizes follow the layer's geometry: `nb` rows, `n_expert`
/// experts, `n_used` of them per row, `d_model` wide rows and `n_ff` wide expert intermediates.
pub(crate) struct MoeScratch {
    /// The router's logits: `[nb][n_expert]`.
    router: Vec<f32>,
    /// The selected experts and their weights: `[nb][n_used]`.
    ids: Vec<u32>,
    weights: Vec<f32>,
    /// The rows that chose each expert, and the slot each of those rows' selections sits in: a
    /// counting sort of `ids`, so an expert's rows are in row order.
    group: Vec<u32>,
    group_slot: Vec<u32>,
    /// Where each expert's group starts and ends: `n_expert + 1` entries.
    group_off: Vec<u32>,
    /// The cursor each expert's group is filled at: `n_expert` entries.
    group_at: Vec<u32>,
    /// The gathered rows of one expert's group: `[nb][d_model]`.
    gathered: Vec<f32>,
    /// One expert's gate and up projections: `[nb][n_ff]`. The SwiGLU takes the gate in place,
    /// so the buffer ends up holding the feed-forward's intermediate.
    gate: Vec<f32>,
    up: Vec<f32>,
    /// One expert's down projection: `[nb][d_model]`.
    down: Vec<f32>,
    /// Every row's expert outputs, in slot order: `[nb][n_used][d_model]`. The weighted sum reads
    /// them in that order, so the result cannot depend on the order the experts ran in.
    slots: Vec<f32>,
    /// The quantised activations: the rows (width `d_model`) and the expert intermediates
    /// (width `n_ff`).
    acts: Vec<QuantAct>,
    acts_ffn: Vec<QuantAct>,
}

impl MoeScratch {
    /// Size the scratch for a layer of this geometry.
    pub(crate) fn new(
        nb: usize,
        n_expert: usize,
        n_used: usize,
        d_model: usize,
        n_ff: usize,
    ) -> MoeScratch {
        MoeScratch {
            router: vec![0.0; nb * n_expert],
            ids: vec![0; nb * n_used],
            weights: vec![0.0; nb * n_used],
            group: vec![0; nb * n_used],
            group_slot: vec![0; nb * n_used],
            group_off: vec![0; n_expert + 1],
            group_at: vec![0; n_expert],
            gathered: vec![0.0; nb * d_model],
            gate: vec![0.0; nb * n_ff],
            up: vec![0.0; nb * n_ff],
            down: vec![0.0; nb * d_model],
            slots: vec![0.0; nb * n_used * d_model],
            acts: (0..nb).map(|_| QuantAct::with_capacity(d_model)).collect(),
            acts_ffn: (0..nb).map(|_| QuantAct::with_capacity(n_ff)).collect(),
        }
    }
}

/// The mixture-of-experts block: route every row, run the experts it chose, and add up their
/// weighted outputs.
///
/// The plan (`docs/design/moe.md` §3):
///
/// 1. the router matmul — one row of scores per row of `x`;
/// 2. [`route`] per row;
/// 3. the rows are grouped by expert (a counting sort), each expert's rows are gathered into a
///    contiguous buffer, and the expert's three matrices run over that group, so an expert's
///    weights are read once for all the rows that chose it;
/// 4. every row's expert outputs are kept **in slot order** and summed with the router's weights
///    in that order, so a row's result cannot depend on which other rows shared the batch
///    (invariant 1).
///
/// `out` is written, not accumulated: the caller adds the residual. The rows of `x` and `out` are
/// `d_model` wide.
///
/// The layer's geometry must fit the engine's activation rules: `d_model` and `n_ff` are multiples
/// of 32 (the activation block) and of the weights' own block size (the matmul's row width). The
/// loader checks that before it builds the views.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn(
    w: &Weights,
    mw: &MoeWeights<'_>,
    x: &[f32],
    out: &mut [f32],
    nb: usize,
    scratch: &mut MoeScratch,
    pool: &Pool,
    st: &mut StepStats,
) {
    let n_expert = mw.gate.len();
    let k = mw.n_used;
    let d_model = mw.router.dims[0] as usize;
    let n_ff = mw.gate[0].dims[1] as usize;
    assert_eq!(mw.up.len(), n_expert, "the expert banks disagree");
    assert_eq!(mw.down.len(), n_expert, "the expert banks disagree");
    assert_eq!(x.len(), nb * d_model, "the rows are not d_model wide");
    assert_eq!(out.len(), nb * d_model, "the rows are not d_model wide");
    assert!(nb > 0, "a mixture of experts needs at least one row");
    assert!(
        k <= n_expert,
        "a row cannot use more experts than there are"
    );

    // 1. the router. The rows are quantised once, as they are for every other matmul.
    for b in 0..nb {
        scratch.acts[b].fill(&x[b * d_model..(b + 1) * d_model]);
    }
    {
        let jobs = [MatJob {
            w: mw.router,
            out: OutPtr::new(&mut scratch.router),
            rows: None,
        }];
        matmul_group(w, &jobs, &scratch.acts[..nb], nb, pool, st);
    }

    // 2. the routing, one row at a time: the router never sees the batch.
    for b in 0..nb {
        route(
            &scratch.router[b * n_expert..(b + 1) * n_expert],
            k,
            mw.renorm,
            &mut scratch.ids[b * k..(b + 1) * k],
            &mut scratch.weights[b * k..(b + 1) * k],
        );
    }

    // 3. the rows that chose each expert, in row order: a counting sort over the selections.
    let off = &mut scratch.group_off;
    off[..=n_expert].fill(0);
    for b in 0..nb {
        for s in 0..k {
            off[scratch.ids[b * k + s] as usize + 1] += 1;
        }
    }
    for e in 0..n_expert {
        off[e + 1] += off[e];
    }
    scratch.group_at[..n_expert].copy_from_slice(&off[..n_expert]);
    for b in 0..nb {
        for s in 0..k {
            let e = scratch.ids[b * k + s] as usize;
            let p = scratch.group_at[e] as usize;
            scratch.group_at[e] += 1;
            scratch.group[p] = b as u32;
            scratch.group_slot[p] = s as u32;
        }
    }

    // 4. the experts, one at a time: each reads its weights once for all the rows that chose it.
    for e in 0..n_expert {
        let (a, z) = (off[e] as usize, off[e + 1] as usize);
        let g = z - a;
        if g == 0 {
            continue;
        }
        // Gather the group's rows. The copy is exact, so a row's activation is the same
        // whatever group it lands in.
        for j in 0..g {
            let b = scratch.group[a + j] as usize;
            scratch.gathered[j * d_model..(j + 1) * d_model]
                .copy_from_slice(&x[b * d_model..(b + 1) * d_model]);
        }
        for j in 0..g {
            scratch.acts[j].fill(&scratch.gathered[j * d_model..(j + 1) * d_model]);
        }
        // The gate and up projections, sharing one activation.
        {
            let jobs = [
                MatJob {
                    w: &mw.gate[e],
                    out: OutPtr::new(&mut scratch.gate),
                    rows: None,
                },
                MatJob {
                    w: &mw.up[e],
                    out: OutPtr::new(&mut scratch.up),
                    rows: None,
                },
            ];
            matmul_group(w, &jobs, &scratch.acts[..g], g, pool, st);
        }
        // SwiGLU: silu(gate) ⊙ up, quantised for the down projection. The gate is taken in
        // place, as the dense block does it.
        swiglu_rows(
            n_ff,
            &mut scratch.gate[..g * n_ff],
            &scratch.up[..g * n_ff],
            &mut scratch.acts_ffn[..g],
            g,
            pool,
            st,
        );
        // The down projection.
        {
            let jobs = [MatJob {
                w: &mw.down[e],
                out: OutPtr::new(&mut scratch.down),
                rows: None,
            }];
            matmul_group(w, &jobs, &scratch.acts_ffn[..g], g, pool, st);
        }
        // Scatter the group's outputs into the slot each row's selection occupies.
        for j in 0..g {
            let b = scratch.group[a + j] as usize;
            let s = scratch.group_slot[a + j] as usize;
            let dst = (b * k + s) * d_model;
            scratch.slots[dst..dst + d_model]
                .copy_from_slice(&scratch.down[j * d_model..(j + 1) * d_model]);
        }
    }

    // 5. the weighted sum, in slot order: llama.cpp adds the experts' outputs in the order the
    //    router chose them (`src/llama-graph.cpp:2607-2614`), and so do we.
    for b in 0..nb {
        for (d, o) in out[b * d_model..(b + 1) * d_model].iter_mut().enumerate() {
            let mut acc = scratch.weights[b * k] * scratch.slots[b * k * d_model + d];
            for s in 1..k {
                acc += scratch.weights[b * k + s] * scratch.slots[(b * k + s) * d_model + d];
            }
            *o = acc;
        }
    }
}
