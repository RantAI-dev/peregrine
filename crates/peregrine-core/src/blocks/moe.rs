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
pub fn route(
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
pub fn expert_view(t: &TensorInfo, e: usize) -> Result<TensorInfo, String> {
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
pub struct MoeWeights<'a> {
    /// The router: `[d_model, n_expert]`, one row of scores per expert.
    pub router: &'a TensorInfo,
    /// The gate projections, one 2D view per expert: `[d_model, n_ff]`.
    pub gate: &'a [TensorInfo],
    /// The up projections, one 2D view per expert: `[d_model, n_ff]`.
    pub up: &'a [TensorInfo],
    /// The down projections, one 2D view per expert: `[n_ff, d_model]`.
    pub down: &'a [TensorInfo],
    /// How many experts a row uses.
    pub n_used: usize,
    /// Whether the selected experts' weights are renormalised over the selection (qwen3moe: yes,
    /// `src/models/qwen3moe.cpp:144`).
    pub renorm: bool,
    /// The scale on the selected experts' weights (`expert_weights_scale`), with llama.cpp's
    /// meaning: **a scale of 0 or 1 is not applied** (`src/llama-graph.cpp:2390`), any other
    /// value multiplies the weights. Every qwen3moe file carries 0, which means "no scaling" —
    /// not "scale everything by zero".
    pub w_scale: f32,
}

/// The buffers a mixture-of-experts layer needs, sized once.
///
/// Every buffer is written before it is read and reused by the next layer, so a decode step
/// allocates nothing (invariant 4). The sizes follow the layer's geometry: `nb` rows, `n_expert`
/// experts, `n_used` of them per row, `d_model` wide rows and `n_ff` wide expert intermediates.
pub struct MoeScratch {
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
    pub fn new(
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

impl MoeScratch {
    /// Keep each activation row's f32 values for F32/F16/BF16 expert matrices (see `QuantAct::keep_f32`).
    pub fn set_keep_f32(&mut self, on: bool) {
        for a in self.acts.iter_mut().chain(self.acts_ffn.iter_mut()) {
            a.keep_f32 = on;
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
pub fn moe_ffn(
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

    // 2b. the scale on the weights, with llama.cpp's rule: a scale of 0 or 1 is not applied
    //     (`src/llama-graph.cpp:2390`), so the 0 every qwen3moe file carries means "no
    //     scaling". The scale lands after the renormalisation, as it does there.
    if mw.w_scale != 0.0 && mw.w_scale != 1.0 {
        for b in 0..nb {
            for w in scratch.weights[b * k..(b + 1) * k].iter_mut() {
                *w *= mw.w_scale;
            }
        }
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

#[cfg(test)]
#[cfg_attr(not(feature = "fmt-legacy"), allow(dead_code))]
mod tests {
    use super::*;
    use crate::gguf::{Format, Gguf};
    use crate::model::TileStore;
    use crate::pool::Pool;
    use crate::quant::{dequant_row, row_bytes, QuantAct};
    use std::path::PathBuf;

    /// A deterministic generator: the tests must not depend on the machine or the clock.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Rng {
            Rng(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (self.next_u32() as f32 / u32::MAX as f32)
        }
    }

    // ------------------------------------------------------------------ the routing --

    /// A straightforward reference for the router: softmax over every expert, sort by
    /// probability, renormalise.
    fn route_reference(logits: &[f32], k: usize, renorm: bool) -> (Vec<u32>, Vec<f32>) {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f64 = probs.iter().map(|p| f64::from(*p)).sum();
        let inv = (1.0 / sum) as f32;
        for p in probs.iter_mut() {
            *p *= inv;
        }
        let mut idx: Vec<u32> = (0..probs.len() as u32).collect();
        idx.sort_by(|&a, &b| {
            probs[b as usize]
                .partial_cmp(&probs[a as usize])
                .unwrap()
                .then_with(|| a.cmp(&b))
        });
        let ids: Vec<u32> = idx[..k].to_vec();
        let mut ws: Vec<f32> = ids.iter().map(|&i| probs[i as usize]).collect();
        if renorm {
            let s: f64 = ws.iter().map(|w| f64::from(*w)).sum();
            let s = (s as f32).max(MIN_F16_NORMAL);
            for w in ws.iter_mut() {
                *w /= s;
            }
        }
        (ids, ws)
    }

    #[test]
    fn route_matches_a_hand_computed_case() {
        // Four experts, the best two. The softmax over all four is
        // [1, e, e², e³] / (1 + e + e² + e³) with the maximum subtracted first.
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        let mut ids = [0u32; 2];
        let mut w = [0.0f32; 2];
        route(&logits, 2, true, &mut ids, &mut w);
        assert_eq!(ids, [3, 2], "the two best experts, best first");
        let e = std::f64::consts::E;
        let sum = 1.0 + e + e * e + e * e * e;
        let p3 = (e * e * e / sum) as f32;
        let p2 = (e * e / sum) as f32;
        // The renormalisation divides by the sum of the two.
        let s = p3 + p2;
        assert!((w[0] - p3 / s).abs() < 1e-6, "{} vs {}", w[0], p3 / s);
        assert!((w[1] - p2 / s).abs() < 1e-6, "{} vs {}", w[1], p2 / s);
        // Without the renormalisation the weights are the raw probabilities.
        let mut ids = [0u32; 2];
        let mut w = [0.0f32; 2];
        route(&logits, 2, false, &mut ids, &mut w);
        assert_eq!(ids, [3, 2]);
        assert!((w[0] - p3).abs() < 1e-6, "{}", w[0]);
        assert!((w[1] - p2).abs() < 1e-6, "{}", w[1]);
    }

    #[test]
    fn route_matches_the_scalar_reference_on_random_logits() {
        let mut rng = Rng::new(0x5eed);
        for trial in 0..200 {
            let n = 1 + (rng.next_u32() % 24) as usize;
            let k = 1 + (rng.next_u32() as usize % n);
            // Some trials repeat a logit, so the selection has ties to break.
            let mut logits: Vec<f32> = (0..n).map(|_| rng.range(-4.0, 4.0)).collect();
            if trial % 3 == 0 && n > 2 {
                logits[1] = logits[0];
                logits[n - 1] = logits[0];
            }
            let mut ids = vec![0u32; k];
            let mut w = vec![0.0f32; k];
            route(&logits, k, true, &mut ids, &mut w);
            let (rid, rw) = route_reference(&logits, k, true);
            assert_eq!(ids, rid, "trial {trial}: {logits:?}");
            for i in 0..k {
                assert!(
                    (w[i] - rw[i]).abs() < 1e-6,
                    "trial {trial}: weight {i} is {} vs {}",
                    w[i],
                    rw[i]
                );
            }
        }
    }

    #[test]
    fn route_breaks_a_tie_by_the_lower_expert_index() {
        // Three experts score exactly the same; the fourth is worse.
        let logits = [1.0f32, 1.0, 1.0, 0.5];
        let mut ids = [0u32; 2];
        let mut w = [0.0f32; 2];
        route(&logits, 2, true, &mut ids, &mut w);
        assert_eq!(ids, [0, 1], "the lower index wins the tie");
        // The two tied experts get the same weight.
        assert_eq!(w[0], w[1]);
        // A tie at the top of a mixed row.
        let logits = [0.5f32, 2.0, 2.0, 1.0];
        route(&logits, 2, true, &mut ids, &mut w);
        assert_eq!(ids, [1, 2]);
        // Three of four tie: the three lowest indices win.
        let logits = [1.0f32, 1.0, 1.0, 1.0];
        let mut ids3 = [0u32; 3];
        let mut w3 = [0.0f32; 3];
        route(&logits, 3, true, &mut ids3, &mut w3);
        assert_eq!(ids3, [0, 1, 2]);
    }

    // -------------------------------------------------------------- the expert views --

    #[test]
    fn the_expert_view_is_the_experts_own_matrix() {
        // A 3D expert tensor: [64 columns, 32 rows, 4 experts], in Q8_0.
        let t = TensorInfo {
            name: "blk.0.ffn_gate_exps.weight".into(),
            dims: vec![64, 32, 4],
            kind: Format::Q8_0,
            offset: 1024,
            tile_off: None,
        };
        let v = expert_view(&t, 2).unwrap();
        assert_eq!(v.dims, vec![64, 32], "a 2D matrix");
        assert_eq!(v.kind, Format::Q8_0);
        // One expert is 64 × 32 weights: 2048 of them, 64 Q8_0 blocks of 34 bytes.
        assert_eq!(v.offset, 1024 + 2 * 64 * 34);
        assert_eq!(v.bytes(), 64 * 34);
        assert!(
            v.tile_off.is_none(),
            "an expert is never repacked into tiles"
        );
        // A view of a matrix that is not three-dimensional, or an expert that does not exist,
        // is refused by name.
        let mut m = t.clone();
        m.dims = vec![64, 32];
        assert!(expert_view(&m, 0).unwrap_err().contains("three dimensions"));
        assert!(expert_view(&t, 4).unwrap_err().contains("expert 4 of 4"));
        // A tensor whose experts are not a whole number of blocks is refused too.
        let mut bad = t.clone();
        bad.dims = vec![3, 1, 4];
        assert!(expert_view(&bad, 0).unwrap_err().contains("blocks"));
    }

    // -------------------------------------------------------------- the whole block --

    /// The ggml type id of a format, as the tensor directory stores it.
    fn ggml_type_id(f: Format) -> u32 {
        match f {
            Format::F32 => 0,
            Format::F16 => 1,
            Format::Q4_0 => 2,
            Format::Q5_0 => 6,
            Format::Q8_0 => 8,
            Format::Q4_K => 12,
            Format::Q5_K => 13,
            Format::Q6_K => 14,
            Format::IQ4_NL => 20,
            Format::IQ4_XS => 23,
            Format::BF16 => 30,
            Format::MXFP4 => 39,
            Format::PQ2_0 => 142,
            Format::PTQ1_0 => 143,
        }
    }

    /// Write a GGUF file with these tensors and no metadata: enough for the block's own tests.
    fn write_gguf(path: &PathBuf, tensors: &[(&str, Vec<u64>, Format, Vec<u8>)]) {
        let mut data: Vec<u8> = Vec::new();
        let mut offsets: Vec<u64> = Vec::new();
        for (_, _, _, bytes) in tensors {
            while data.len() % 32 != 0 {
                data.push(0);
            }
            offsets.push(data.len() as u64);
            data.extend_from_slice(bytes);
        }
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        for ((name, dims, kind, _), off) in tensors.iter().zip(&offsets) {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&ggml_type_id(*kind).to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
        }
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&data);
        std::fs::write(path, out).expect("write the test model");
    }

    /// A row of weights in Q8_0: one f16 scale per 32 weights.
    fn q8_0_row(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 32 * 34);
        for blk in x.chunks_exact(32) {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&crate::quant::f32_to_f16(d).to_le_bytes());
            for &v in blk {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                out.push(q as u8);
            }
        }
        out
    }

    /// A tiny mixture-of-experts layer: the router and four experts, in Q8_0.
    struct Tiny {
        w: Weights,
        router: TensorInfo,
        gate: Vec<TensorInfo>,
        up: Vec<TensorInfo>,
        down: Vec<TensorInfo>,
        d_model: usize,
        n_ff: usize,
        n_expert: usize,
        path: PathBuf,
    }

    impl Tiny {
        /// Build the layer on disk and open it. The router's rows are one-hot, so a row's experts
        /// are decided by its first `n_expert` entries and are far apart in the score.
        fn build(rng: &mut Rng, d_model: usize, n_ff: usize, n_expert: usize) -> Tiny {
            // The name carries the generator's state, so two tests never write the same file.
            let path = std::env::temp_dir().join(format!(
                "peregrine-moe-{}-{d_model}-{n_ff}-{n_expert}-{}.gguf",
                std::process::id(),
                rng.0
            ));
            let mut tensors: Vec<(&str, Vec<u64>, Format, Vec<u8>)> = Vec::new();
            // the router: one row per expert, one-hot so the scores are far apart
            let mut router_data = Vec::new();
            for e in 0..n_expert {
                let mut row = vec![0.0f32; d_model];
                row[e] = 4.0;
                router_data.extend_from_slice(&q8_0_row(&row));
            }
            tensors.push((
                "router",
                vec![d_model as u64, n_expert as u64],
                Format::Q8_0,
                router_data,
            ));
            // The experts: gate and up are [d_model, n_ff] each, down is [n_ff, d_model]. The
            // expert index runs slowest, as llama.cpp's tensors do, so the file holds every
            // expert's rows one after another.
            let mut gate_data = Vec::new();
            let mut up_data = Vec::new();
            let mut down_data = Vec::new();
            for _ in 0..n_expert {
                let g: Vec<Vec<f32>> = (0..n_ff)
                    .map(|_| (0..d_model).map(|_| rng.range(-0.5, 0.5)).collect())
                    .collect();
                let u: Vec<Vec<f32>> = (0..n_ff)
                    .map(|_| (0..d_model).map(|_| rng.range(-0.5, 0.5)).collect())
                    .collect();
                let d: Vec<Vec<f32>> = (0..d_model)
                    .map(|_| (0..n_ff).map(|_| rng.range(-0.5, 0.5)).collect())
                    .collect();
                let flat = |rows: &Vec<Vec<f32>>| -> Vec<u8> {
                    rows.iter().flat_map(|r| q8_0_row(r)).collect()
                };
                gate_data.extend_from_slice(&flat(&g));
                up_data.extend_from_slice(&flat(&u));
                down_data.extend_from_slice(&flat(&d));
            }
            tensors.push((
                "gate",
                vec![d_model as u64, n_ff as u64, n_expert as u64],
                Format::Q8_0,
                gate_data,
            ));
            tensors.push((
                "up",
                vec![d_model as u64, n_ff as u64, n_expert as u64],
                Format::Q8_0,
                up_data,
            ));
            tensors.push((
                "down",
                vec![n_ff as u64, d_model as u64, n_expert as u64],
                Format::Q8_0,
                down_data,
            ));
            write_gguf(&path, &tensors);
            let g = Gguf::open(&path).expect("open the test model");
            let router = g.tensor("router").unwrap().clone();
            let gate = (0..n_expert)
                .map(|e| expert_view(g.tensor("gate").unwrap(), e).unwrap())
                .collect();
            let up = (0..n_expert)
                .map(|e| expert_view(g.tensor("up").unwrap(), e).unwrap())
                .collect();
            let down = (0..n_expert)
                .map(|e| expert_view(g.tensor("down").unwrap(), e).unwrap())
                .collect();
            Tiny {
                // The weights of a file with no repacked tiles and no LittleBit sidecar: the same
                // three fields `Model::load_with` fills in. `Weights` has no constructor of its own,
                // so this literal names its fields; a new field means adding it here too.
                w: Weights {
                    g,
                    tiles: TileStore::default(),
                    #[cfg(feature = "littlebit")]
                    lb: None,
                },
                router,
                gate,
                up,
                down,
                d_model,
                n_ff,
                n_expert,
                path,
            }
        }

        /// The weights the block reads.
        fn weights(&self, n_used: usize) -> MoeWeights<'_> {
            MoeWeights {
                router: &self.router,
                gate: &self.gate,
                up: &self.up,
                down: &self.down,
                n_used,
                renorm: true,
                w_scale: 0.0,
            }
        }
    }

    impl Drop for Tiny {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// The batch: the first `n_expert` entries of every row pick the experts (a rotation of
    /// {-1, -1/2, 1/2, 1}, so the rows route differently), the rest is noise.
    fn rows(rng: &mut Rng, nb: usize, d_model: usize, n_expert: usize) -> Vec<f32> {
        let pattern = [-1.0f32, -0.5, 0.5, 1.0];
        let mut x = vec![0.0f32; nb * d_model];
        for b in 0..nb {
            for e in 0..n_expert {
                x[b * d_model + e] = pattern[(e + b) % n_expert];
            }
            for j in n_expert..d_model {
                x[b * d_model + j] = rng.range(-0.5, 0.5);
            }
        }
        // One row ties two experts exactly: the same value in the same activation block, so the
        // router scores them identically and the tie rule decides.
        x[3 * d_model] = 1.0;
        x[3 * d_model + 1] = 1.0;
        x[3 * d_model + 2] = -1.0;
        x[3 * d_model + 3] = -1.0;
        x
    }

    /// A matrix's rows, dequantised.
    fn dequant_matrix(t: &TensorInfo, bytes: &[u8], rows: usize, cols: usize) -> Vec<Vec<f32>> {
        let rb = row_bytes(t.kind, cols);
        let mut out = Vec::with_capacity(rows);
        let mut buf = vec![0.0f32; cols];
        for r in 0..rows {
            dequant_row(t.kind, &bytes[r * rb..(r + 1) * rb], &mut buf);
            out.push(buf.clone());
        }
        out
    }

    /// The activation as the matmul sees it: the int8 codes and their scales.
    fn dequant_act(a: &QuantAct, n: usize) -> Vec<f32> {
        (0..n).map(|i| a.q[i] as f32 * a.d[i / 32]).collect()
    }

    /// The naive dense reference: every selected expert dequantised, f32 products, f64
    /// accumulation.
    ///
    /// The activations are the engine's own int8 activations, dequantised, because that is what
    /// the engine's matmuls multiply. The difference from the block is then the order of the
    /// arithmetic and nothing else — the routing, the grouping and the weighted sum are what the
    /// reference checks.
    fn reference(tiny: &Tiny, x: &[f32], nb: usize, k: usize) -> Vec<f32> {
        let d_model = tiny.d_model;
        let n_ff = tiny.n_ff;
        let n_expert = tiny.n_expert;
        let router_bytes = tiny.w.g.bytes(&tiny.router);
        let router_rows = dequant_matrix(&tiny.router, router_bytes, n_expert, d_model);
        let mut out = vec![0.0f32; nb * d_model];
        for b in 0..nb {
            let mut act = QuantAct::with_capacity(d_model);
            act.fill(&x[b * d_model..(b + 1) * d_model]);
            let xq = dequant_act(&act, d_model);
            // the router's logits
            let logits: Vec<f32> = router_rows
                .iter()
                .map(|r| {
                    r.iter()
                        .zip(&xq)
                        .map(|(w, a)| *w as f64 * *a as f64)
                        .sum::<f64>() as f32
                })
                .collect();
            let (ids, ws) = route_reference(&logits, k, true);
            let mut acc = vec![0.0f64; d_model];
            for (s, &e) in ids.iter().enumerate() {
                let e = e as usize;
                let g_rows =
                    dequant_matrix(&tiny.gate[e], tiny.w.g.bytes(&tiny.gate[e]), n_ff, d_model);
                let u_rows =
                    dequant_matrix(&tiny.up[e], tiny.w.g.bytes(&tiny.up[e]), n_ff, d_model);
                let d_rows =
                    dequant_matrix(&tiny.down[e], tiny.w.g.bytes(&tiny.down[e]), d_model, n_ff);
                // the gate and up projections, and the SwiGLU between them
                let mut hb = vec![0.0f32; n_ff];
                for r in 0..n_ff {
                    let gv: f64 = g_rows[r]
                        .iter()
                        .zip(&xq)
                        .map(|(w, a)| *w as f64 * *a as f64)
                        .sum();
                    let uv: f64 = u_rows[r]
                        .iter()
                        .zip(&xq)
                        .map(|(w, a)| *w as f64 * *a as f64)
                        .sum();
                    hb[r] = ((gv / (1.0 + (-gv).exp())) * uv) as f32;
                }
                // the feed-forward's intermediate is quantised, as it is in the engine
                let mut hact = QuantAct::with_capacity(n_ff);
                hact.fill(&hb);
                let hq = dequant_act(&hact, n_ff);
                // the down projection, and the row's share of it
                for r in 0..d_model {
                    let y: f64 = d_rows[r]
                        .iter()
                        .zip(&hq)
                        .map(|(w, a)| *w as f64 * *a as f64)
                        .sum();
                    acc[r] += ws[s] as f64 * y;
                }
            }
            for r in 0..d_model {
                out[b * d_model + r] = acc[r] as f32;
            }
        }
        out
    }

    /// The block's answer for a batch of rows.
    fn run(tiny: &Tiny, x: &[f32], nb: usize, k: usize, threads: usize) -> Vec<f32> {
        let pool = Pool::new(threads);
        let mut scratch = MoeScratch::new(nb, tiny.n_expert, k, tiny.d_model, tiny.n_ff);
        let mut out = vec![0.0f32; nb * tiny.d_model];
        let mut st = StepStats::default();
        moe_ffn(
            &tiny.w,
            &tiny.weights(k),
            x,
            &mut out,
            nb,
            &mut scratch,
            &pool,
            &mut st,
        );
        out
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_block_matches_the_dense_reference() {
        let mut rng = Rng::new(0x1234_5678);
        let (d_model, n_ff, n_expert, k) = (64usize, 32usize, 4usize, 2usize);
        let tiny = Tiny::build(&mut rng, d_model, n_ff, n_expert);
        let nb = 17;
        let x = rows(&mut rng, nb, d_model, n_expert);
        let got = run(&tiny, &x, nb, k, 4);
        let want = reference(&tiny, &x, nb, k);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "the reference produced nothing");
        // The tolerance: the block accumulates in f32 where the reference accumulates in f64,
        // and the feed-forward's intermediate is quantised to int8 in both — a code that lands on
        // the other side of a rounding boundary moves one element of one expert by about 0.07%.
        let tol = 5e-3 * scale;
        let mut worst = 0.0f32;
        for i in 0..want.len() {
            let d = (got[i] - want[i]).abs();
            worst = worst.max(d);
            assert!(
                d <= tol,
                "row {} element {}: {} vs {} (difference {d}, tolerance {tol})",
                i / d_model,
                i % d_model,
                got[i],
                want[i]
            );
        }
        assert!(worst < tol);
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn a_row_does_not_depend_on_the_batch_it_is_in() {
        let mut rng = Rng::new(0x9876_5432);
        let (d_model, n_ff, n_expert, k) = (64usize, 32usize, 4usize, 2usize);
        let tiny = Tiny::build(&mut rng, d_model, n_ff, n_expert);
        let nb = 17;
        let x = rows(&mut rng, nb, d_model, n_expert);
        let batched = run(&tiny, &x, nb, k, 4);
        // The rows really do route differently, or the test would prove nothing.
        let mut routings = std::collections::HashSet::new();
        {
            let pool = Pool::new(1);
            let mut scratch = MoeScratch::new(nb, n_expert, k, d_model, n_ff);
            let mut out = vec![0.0f32; nb * d_model];
            let mut st = StepStats::default();
            moe_ffn(
                &tiny.w,
                &tiny.weights(k),
                &x,
                &mut out,
                nb,
                &mut scratch,
                &pool,
                &mut st,
            );
            for b in 0..nb {
                let ids: Vec<u32> = scratch.ids[b * k..(b + 1) * k].to_vec();
                routings.insert(ids);
            }
            // Row three ties two experts exactly, and the tie goes to the lower index.
            assert_eq!(
                &scratch.ids[3 * k..4 * k],
                &[0, 1],
                "the tie did not go to the lower expert index"
            );
        }
        assert!(routings.len() > 1, "every row routed the same way");
        // Every row, alone, is bit for bit what it was in the batch.
        for b in 0..nb {
            let alone = run(&tiny, &x[b * d_model..(b + 1) * d_model], 1, k, 4);
            assert_eq!(
                alone,
                batched[b * d_model..(b + 1) * d_model].to_vec(),
                "row {b} changed with the batch"
            );
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_block_does_not_depend_on_the_thread_count() {
        let mut rng = Rng::new(0xfeed_face);
        let (d_model, n_ff, n_expert, k) = (64usize, 32usize, 4usize, 2usize);
        let tiny = Tiny::build(&mut rng, d_model, n_ff, n_expert);
        let nb = 5;
        let x = rows(&mut rng, nb, d_model, n_expert);
        let one = run(&tiny, &x, nb, k, 1);
        let four = run(&tiny, &x, nb, k, 4);
        assert_eq!(one, four, "the thread count changed the answer");
    }
}
