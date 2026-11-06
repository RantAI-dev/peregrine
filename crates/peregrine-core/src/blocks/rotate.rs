//! The block Hadamard rotation of the hybrid path (Qwen3.5-style models).
//!
//! The ternary weights were rotated by a block Hadamard transform with explicit signs. The engine applies the
//! rotation to every matmul input — the signs first, then the transform, `H·(s ⊙ x)` — and, inverted, to each
//! embedding row: the transform first, then the signs, `s ⊙ H·x`. The two are not the same operation and not
//! in the same order, so they are two functions here rather than one with a flag.
//!
//! The rotation's data — the sign vector of every input width and the block size — is the [`Rotation`] struct
//! the hybrid family owns; the functions below take it, slices and scratch, never the model.

use crate::blocks::par::{for_row_chunks_2, for_row_chunks_3, Sched};
use crate::hadamard::fwht;
use crate::model::StepStats;
use crate::pool::Pool;
use crate::quant::QuantAct;

/// A per-row step fused into a rotation's pass: `(row index, row)`.
pub type RowPrep<'a> = dyn Fn(usize, &mut [f32]) + Sync + 'a;

/// The Hadamard rotation's data: the sign vector of every input width and the block size.
#[derive(Clone)]
pub struct Rotation {
    /// `signs / √block` per input width.
    signs: Vec<(usize, Vec<f32>)>,
    /// The Hadamard block size (a power of two).
    block: usize,
}

/// The tiled→grouped v-head permutation's geometry: the value heads of a Gated DeltaNet output are permuted
/// from the tiled order to the grouped one before the rotation.
#[derive(Clone, Copy)]
pub struct VPerm {
    /// The width of one v-head's slice.
    pub dv: usize,
    /// The number of key heads the value heads are grouped over.
    pub n_k_heads: usize,
    /// The number of value heads.
    pub n_v_heads: usize,
}

#[inline(always)]
fn apply_rotation(r: &mut [f32], x: &[f32], ss: &[f32], perm: Option<VPerm>, blk: usize) {
    if let Some(p) = perm {
        let (hd, nk, rep) = (p.dv, p.n_k_heads, p.n_v_heads / p.n_k_heads);
        for rr in 0..rep {
            for gg in 0..nk {
                let s = (rr * nk + gg) * hd;
                let d = (gg * rep + rr) * hd;
                r[d..d + hd].copy_from_slice(&x[s..s + hd]);
            }
        }
        for (v, s) in r.iter_mut().zip(ss) {
            *v *= s;
        }
    } else {
        for ((v, &xi), s) in r.iter_mut().zip(x.iter()).zip(ss) {
            *v = xi * s;
        }
    }
    for c in r.chunks_exact_mut(blk) {
        fwht(c);
    }
}

impl Rotation {
    /// The rotation of a model: `signs` holds the sign vector of every input width (already divided by
    /// `√block`), `block` the transform's block size.
    pub fn new(signs: Vec<(usize, Vec<f32>)>, block: usize) -> Rotation {
        Rotation { signs, block }
    }

    /// The sign vector of an input width.
    #[inline]
    pub fn signs_for(&self, width: usize) -> Result<&[f32], String> {
        self.signs
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, s)| s.as_slice())
            .ok_or_else(|| format!("no Hadamard sign vector for input width {width}"))
    }

    /// The embedding's inverse rotation: the block Hadamard transform, then the signs (`s ⊙ H·row`).
    ///
    /// The row must already be dequantised. This is *not* the matmul input's rotation
    /// ([`Rotation::rotate_rows_pre`]), which applies the signs first: the two orders are not interchangeable.
    #[inline]
    pub fn rotate_embedding(&self, row: &mut [f32], signs: &[f32]) {
        for ch in row.chunks_exact_mut(self.block) {
            fwht(ch);
        }
        for (v, s) in row.iter_mut().zip(signs) {
            *v *= s;
        }
    }

    /// `x` (row `b` of `src`, width `w`) → optional tiled→grouped v-head permutation → signs → blockwise FWHT,
    /// into `rot`, then quantised into `acts[b]`. Rows are independent and split across the pool.
    ///
    /// `src` must hold `nb` rows of `w` floats that nothing else touches during the call.
    /// `rot` is the scratch the rotated rows are written to, `rot_w` its row stride.
    ///
    /// `parallel` says whether the rows are worth splitting across the pool: the caller decides, because the
    /// threshold belongs to the model.
    #[allow(clippy::too_many_arguments)]
    pub fn rotate_rows(
        &self,
        rot: &mut [f32],
        rot_w: usize,
        acts: &mut [QuantAct],
        src: &[f32],
        w: usize,
        nb: usize,
        perm: Option<VPerm>,
        parallel: bool,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        let ss = self.signs_for(w).expect("sign vector checked at load");
        let blk = self.block;
        for_row_chunks_2(
            pool,
            parallel,
            rot,
            rot_w,
            acts,
            1,
            nb,
            Sched::Dynamic(1),
            |_wk, rows, r_chunk, a_chunk| {
                for (i, b) in rows.enumerate() {
                    let r = &mut r_chunk[i * rot_w..i * rot_w + w];
                    let x = &src[b * w..(b + 1) * w];
                    apply_rotation(r, x, ss, perm, blk);
                    a_chunk[i].fill(r);
                }
            },
        );
        if parallel {
            stats.barriers += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rotate_rows_pre(
        &self,
        rot: &mut [f32],
        rot_w: usize,
        acts: &mut [QuantAct],
        src: &mut [f32],
        w: usize,
        nb: usize,
        perm: Option<VPerm>,
        pre: Option<&RowPrep>,
        parallel: bool,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        let ss = self.signs_for(w).expect("sign vector checked at load");
        let blk = self.block;
        for_row_chunks_3(
            pool,
            parallel,
            rot,
            rot_w,
            acts,
            1,
            src,
            w,
            nb,
            Sched::Dynamic(1),
            |_wk, rows, r_chunk, a_chunk, x_chunk| {
                for (i, b) in rows.enumerate() {
                    let x_row = &mut x_chunk[i * w..(i + 1) * w];
                    if let Some(f) = pre {
                        f(b, x_row);
                    }
                    let r = &mut r_chunk[i * rot_w..i * rot_w + w];
                    apply_rotation(r, x_row, ss, perm, blk);
                    a_chunk[i].fill(r);
                }
            },
        );
        if parallel {
            stats.barriers += 1;
        }
    }
}
