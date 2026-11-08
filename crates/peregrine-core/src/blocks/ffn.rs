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
        let a = &mut acts[0];
        // SwiGLU and the quantisation in one pass. The quantise is per 32-block, so a
        // worker's contiguous block range produces exactly the bytes one call would;
        // the down matmul then starts from a ready activation without the caller first
        // spending the serial quantise on it. Fall back to the two-step form when the
        // activation has a layout this split cannot write (ternary, or a width that is
        // not whole 256-blocks).
        let fused = !a.ternary && f.is_multiple_of(256) && f / 256 >= pool.width();
        if fused {
            a.prepare(f);
            // Each worker writes only its own 256-aligned block range of `a`.
            struct ActPtr(*mut QuantAct);
            // SAFETY: the ranges partition [0, f/256) by construction (`share`), so no
            // two workers write the same field element.
            unsafe impl Sync for ActPtr {}
            impl ActPtr {
                /// The activation this worker's range belongs to.
                ///
                /// # Safety
                /// The caller must write only its own block range.
                unsafe fn get(&self) -> *mut QuantAct {
                    self.0
                }
            }
            let ptr = ActPtr(a as *mut QuantAct);
            let out = crate::blocks::par::Disjoint::new(&mut hb[..f]);
            crate::prof::time(crate::prof::Slot::Swiglu, || {
                pool.run(move |wk, nw| {
                    let (b0, b1) = crate::pool::share(f / 256, wk, nw);
                    let (lo, hi) = (b0 * 256, b1 * 256);
                    // SAFETY: `lo..hi` is this worker's exclusive share of the row.
                    let g = unsafe { out.chunk(lo, hi - lo) };
                    crate::vmath::swiglu(g, &up[lo..hi]);
                    // SAFETY: the same exclusive range, 256-aligned, in `ActPtr`.
                    unsafe { (*ptr.get()).fill_range(g, lo) };
                });
            });
            stats.barriers += 1;
        } else {
            crate::prof::time(crate::prof::Slot::Swiglu, || {
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
            });
            stats.barriers += 1;
            crate::prof::time(crate::prof::Slot::Quant, || acts[0].fill(&hb[..f]));
        }
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

/// OpenAI's `swiglu_oai` — the gated activation gpt-oss's experts use:
/// `hb = x / (1 + e^(α·(−x))) ⊙ (y + 1)` with `x = min(gate, limit)` and
/// `y = clamp(up, −limit, limit)` — and its quantisation, row-parallel like
/// `swiglu_rows`.
///
/// llama.cpp v0.9.0's `LLM_FFN_SWIGLU_OAI_MOE` (`src/llama-graph.cpp:2519-2526`,
/// `ggml/src/ggml-cpu/ops.cpp:3388-3450`); the alpha and the limit are its
/// constants (`1.702` and `7.0`).
#[allow(clippy::too_many_arguments)]
pub fn swiglu_oai_rows(
    n_ff: usize,
    hb: &mut [f32],
    up: &[f32],
    acts: &mut [QuantAct],
    nb: usize,
    alpha: f32,
    limit: f32,
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
                    crate::vmath::swiglu_oai(g, &up[b * f..(b + 1) * f], alpha, limit);
                    acts_chunk[i].fill(g);
                }
            },
        );
        stats.barriers += 1;
    } else if nb == 1 && pool.width() > 1 && f >= 1024 {
        // Decode: one row, but thousands of exponentials on the calling thread
        // while the workers spin.
        for_row_chunks(
            pool,
            true,
            &mut hb[..f],
            1,
            f,
            Sched::Static,
            |_wk, range, gs| {
                crate::vmath::swiglu_oai(gs, &up[range], alpha, limit);
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
                    crate::vmath::swiglu_oai(g, &up[b * f..(b + 1) * f], alpha, limit);
                    acts_chunk[i].fill(g);
                }
            },
        );
    }
}

/// GeGLU — `hb = gelu_tanh(gate) ⊙ up` — and its quantisation, row-parallel like
/// `norm_rows`.
pub fn geglu_rows(
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
                    crate::vmath::geglu(g, &up[b * f..(b + 1) * f]);
                    acts_chunk[i].fill(g);
                }
            },
        );
        stats.barriers += 1;
    } else if nb == 1 && pool.width() > 1 && f >= 1024 {
        for_row_chunks(
            pool,
            true,
            &mut hb[..f],
            1,
            f,
            Sched::Static,
            |_wk, range, gs| {
                crate::vmath::geglu(gs, &up[range]);
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
                    crate::vmath::geglu(g, &up[b * f..(b + 1) * f]);
                    acts_chunk[i].fill(g);
                }
            },
        );
    }
}
