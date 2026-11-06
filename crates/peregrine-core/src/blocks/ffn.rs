//! Feed-forward network (SwiGLU) block.

use crate::blocks::par::{for_row_chunks, for_row_chunks_2, Sched};
use crate::blocks::ROW_PARALLEL;
use crate::model::StepStats;
use crate::pool::Pool;
use crate::quant::QuantAct;

/// SwiGLU — `hb = silu(gate) ⊙ up` — and its quantisation, row-parallel like
/// `norm_rows`. The activation is `vmath::swiglu`, ggml's vector form, so the
/// feed-forward nonlinearity is llama.cpp's to the last rounding of its `exp`.
pub fn swiglu_rows(
    n_ff: usize,
    hb: &mut [f32],
    up: &[f32],
    acts: &mut [QuantAct],
    nb: usize,
    pool: &Pool,
    stats: &mut StepStats,
) {
    let f = n_ff;
    if nb >= ROW_PARALLEL && pool.width() > 1 {
        for_row_chunks_2(
            pool,
            true,
            hb,
            f,
            acts,
            1,
            nb,
            Sched::Static,
            |_wk, rows, hb_chunk, acts_chunk| {
                for (i, b) in rows.enumerate() {
                    let g = &mut hb_chunk[i * f..(i + 1) * f];
                    crate::vmath::swiglu(g, &up[b * f..(b + 1) * f]);
                    acts_chunk[i].fill(g);
                }
            },
        );
        stats.barriers += 1;
    } else if nb == 1 && pool.width() > 1 && f >= 1024 {
        // Decode: one row, but thousands of `exp`s on the calling thread while
        // the workers spin — a tenth of a SmolLM2 step at four threads.
        // The activation is split across the pool for the gate (same arithmetic,
        // element by element), then quantised once, vectorised, on this thread.
        for_row_chunks(
            pool,
            true,
            &mut hb[..f],
            1,
            f,
            Sched::Static,
            |_wk, range, gs| {
                crate::vmath::swiglu(gs, &up[range]);
            },
        );
        stats.barriers += 1;
        acts[0].fill(&hb[..f]);
    } else {
        for_row_chunks_2(
            pool,
            false,
            hb,
            f,
            acts,
            1,
            nb,
            Sched::Static,
            |_wk, rows, hb_chunk, acts_chunk| {
                for (i, b) in rows.enumerate() {
                    let g = &mut hb_chunk[i * f..(i + 1) * f];
                    crate::vmath::swiglu(g, &up[b * f..(b + 1) * f]);
                    acts_chunk[i].fill(g);
                }
            },
        );
    }
}
