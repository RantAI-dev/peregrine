//! Feed-forward network (SwiGLU) block.

use crate::blocks::par::{SharedActs, SharedBuf};
use crate::blocks::ROW_PARALLEL;
use crate::model::StepStats;
use crate::pool::{share, Pool};

/// SwiGLU — `hb = silu(gate) ⊙ up` — and its quantisation, row-parallel like
/// `norm_rows`. The activation is `vmath::swiglu`, ggml's vector form, so the
/// feed-forward nonlinearity is llama.cpp's to the last rounding of its `exp`.
pub(crate) fn swiglu_rows(
    n_ff: usize,
    hb: &mut [f32],
    up: &[f32],
    acts: SharedActs,
    nb: usize,
    pool: &Pool,
    stats: &mut StepStats,
) {
    let f = n_ff;
    let hb_buf = SharedBuf(hb.as_mut_ptr());
    let body = |wk: usize, nw: usize| {
        let (lo, hi) = share(nb, wk, nw);
        for b in lo..hi {
            // SAFETY: as `norm_rows`.
            let g = unsafe { hb_buf.slice(b * f, f) };
            crate::vmath::swiglu(g, &up[b * f..(b + 1) * f]);
            unsafe { acts.get(b) }.fill(g);
        }
    };
    if nb >= ROW_PARALLEL && pool.width() > 1 {
        pool.run(body);
        stats.barriers += 1;
    } else if nb == 1 && pool.width() > 1 && f >= 1024 {
        // Decode: one row, but thousands of `exp`s on the calling thread while
        // the workers spin — a tenth of a SmolLM2 step at four threads.
        // The activation is split across the pool for the gate (same arithmetic,
        // element by element), then quantised once, vectorised, on this thread.
        let g = SharedBuf(hb.as_mut_ptr());
        pool.run(|wk, nw| {
            let (lo, hi) = share(f, wk, nw);
            // SAFETY: `share` gives each worker a disjoint range of the row.
            let gs = unsafe { g.slice(lo, hi - lo) };
            crate::vmath::swiglu(gs, &up[lo..hi]);
        });
        stats.barriers += 1;
        unsafe { acts.get(0) }.fill(&hb[..f]);
    } else {
        body(0, 1);
    }
}
