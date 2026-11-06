//! The Gated DeltaNet steps of the hybrid path (Qwen3.5-style models).
//!
//! The linear-attention layers of a hybrid model run, per layer: the β and α projections against the unrotated
//! normed rows, the causal conv over the query, key and value, the SiLU with the q/k L2 norms and the gates, the
//! delta rule over the value heads, and the gated norm of the output. Each is a free function over slices and
//! scratch here; the family's `gdn` calls them in order.
//!
//! Look-alike helpers are kept apart on purpose (ADR 0006): the L2 norm below scales by `(1/n.max(1e-6))·scale`
//! and the gated norm by `x *= inv·w`, neither of which is the norm of the dense path.

use crate::blocks::par::{RowPtrs, SharedBuf, SnapPtrs};
use crate::model::StepStats;
use crate::pool::Pool;
use std::sync::atomic::{AtomicUsize, Ordering as AO};

/// The Gated DeltaNet geometry of a layer: the head counts and widths its steps need.
#[derive(Clone, Copy)]
pub(crate) struct RecurrentCfg {
    /// The key (query) heads; the value heads are grouped over them.
    pub n_k_heads: usize,
    /// The value heads; each holds its own state.
    pub n_v_heads: usize,
    /// The state's width (the query's and the key's width).
    pub dk: usize,
    /// The value row's width.
    pub dv: usize,
    /// The conv's taps, the current token included.
    pub d_conv: usize,
}

impl RecurrentCfg {
    /// The width of the concatenated query heads.
    #[inline]
    pub(crate) fn key_dim(&self) -> usize {
        self.n_k_heads * self.dk
    }
    /// The width of the concatenated value heads.
    #[inline]
    pub(crate) fn value_dim(&self) -> usize {
        self.n_v_heads * self.dv
    }
    /// The width of the conv's input: the query, the key and the value, concatenated.
    #[inline]
    pub(crate) fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// Floats of the delta-rule state per sequence per layer.
    #[inline]
    pub(crate) fn state_floats(&self) -> usize {
        self.n_v_heads * self.dv * self.dk
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// β and α: `[2·n_v][d]` f32 against the unrotated normed rows, split across the pool by output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn beta_alpha(
    out: &mut [f32],
    ab_w: &[f32],
    xb: &[f32],
    d: usize,
    nv: usize,
    nb: usize,
    pool: &Pool,
    stats: &mut StepStats,
) {
    let out = SharedBuf(out.as_mut_ptr());
    let n_out = 2 * nv;
    let next = AtomicUsize::new(0);
    pool.run(|_wk, _nw| loop {
        let lo = next.fetch_add(4, AO::Relaxed);
        if lo >= n_out {
            break;
        }
        for r in lo..(lo + 4).min(n_out) {
            let w = &ab_w[r * d..(r + 1) * d];
            for b in 0..nb {
                // SAFETY: output (b, r) belongs to the worker claiming r.
                let v = crate::vmath::dot_f32(w, &xb[b * d..(b + 1) * d]);
                unsafe { out.slice(b * n_out + r, 1)[0] = v };
            }
        }
    });
    stats.barriers += 1;
    stats.bytes_weights += (ab_w.len() * 4) as u64;
    stats.macs += (ab_w.len() * nb) as u64;
}

/// The causal conv: channels are independent, so a worker takes a channel range and runs every row through it in
/// order (a shared cache's history advances one token per row). The history is a ring of `taps` planes of
/// `conv_dim` channels: the input at position p sits in plane p mod taps, so the oldest of the window ending at
/// `pos` is plane pos mod taps, the one this token overwrites after the sum.
///
/// `big` holds the layer's query, key and value rows (the conv's input), `big_w` its row stride; the conv
/// writes its result over them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn causal_conv(
    big: &mut [f32],
    big_w: usize,
    conv_w: &[f32],
    rp: RowPtrs,
    spp: SnapPtrs,
    cfg: &RecurrentCfg,
    nb: usize,
    positions: &[usize],
    pool: &Pool,
    stats: &mut StepStats,
) {
    let cd = cfg.conv_dim();
    let taps = cfg.d_conv - 1;
    let bigp = SharedBuf(big.as_mut_ptr());
    let wcur = &conv_w[taps * cd..(taps + 1) * cd];
    const C: usize = 256;
    let next = AtomicUsize::new(0);
    pool.run(|_wk, _nw| {
        let mut acc = [0.0f32; C];
        loop {
            let c0 = next.fetch_add(C, AO::Relaxed);
            if c0 >= cd {
                break;
            }
            let c1 = (c0 + C).min(cd);
            for b in 0..nb {
                // SAFETY: channels c0..c1 of every row's buffer and history are this worker's.
                let (_, hp) = unsafe { rp.at(b) };
                let pos = positions[b];
                let y = unsafe { bigp.slice(b * big_w + c0, c1 - c0) };
                for (i, ch) in (c0..c1).enumerate() {
                    acc[i] = y[i] * wcur[ch];
                }
                for t in 0..taps {
                    let plane = (pos + t) % taps;
                    let hist =
                        unsafe { std::slice::from_raw_parts(hp.add(plane * cd + c0), c1 - c0) };
                    let wt = &conv_w[t * cd + c0..t * cd + c1];
                    for i in 0..c1 - c0 {
                        acc[i] += hist[i] * wt[i];
                    }
                }
                let slot = pos % taps;
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(hp.add(slot * cd + c0), c1 - c0) };
                dst.copy_from_slice(y);
                y.copy_from_slice(&acc[..c1 - c0]);
                // A speculative verification's snapshot: this range of the history after its row.
                if let Some((sp, _, sconv)) = unsafe { spp.at(b) } {
                    if pos == sp {
                        for t in 0..taps {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    hp.add(t * cd + c0),
                                    sconv.add(t * cd + c0),
                                    c1 - c0,
                                )
                            };
                        }
                    }
                }
            }
        }
    });
    stats.barriers += 1;
}

/// SiLU, the q/k L2 norms and the gate maths, per row (rows are independent now).
///
/// The query and the key are scaled by their own L2 norm (`1/n.max(1e-6)`, then the query's `1/√dk`); the
/// gates turn α into the state's decay. `ab` holds `[β (n_v) | α (n_v)]` per row and comes back as
/// `[β | decay]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn silu_norm_gates(
    big: &mut [f32],
    big_w: usize,
    ab: &mut [f32],
    dt_bias: &[f32],
    a_log: &[f32],
    cfg: &RecurrentCfg,
    nb: usize,
    parallel: bool,
    pool: &Pool,
    stats: &mut StepStats,
) {
    let (nv, nk, dk) = (cfg.n_v_heads, cfg.n_k_heads, cfg.dk);
    let (kd, cd) = (cfg.key_dim(), cfg.conv_dim());
    let q_scale = 1.0 / (dk as f32).sqrt();
    let bigp = SharedBuf(big.as_mut_ptr());
    let abp = SharedBuf(ab.as_mut_ptr());
    let next = AtomicUsize::new(0);
    let body = |_wk: usize, _nw: usize| loop {
        let b = next.fetch_add(1, AO::Relaxed);
        if b >= nb {
            break;
        }
        // SAFETY: row b of `big` and of `ab` is claimed by this worker.
        let y = unsafe { bigp.slice(b * big_w, cd) };
        crate::vmath::silu(y);
        for (base, scale) in [(0usize, q_scale), (kd, 1.0)] {
            for h in 0..nk {
                let v = &mut y[base + h * dk..base + (h + 1) * dk];
                let n = crate::vmath::sumsq_f64(v).sqrt() as f32;
                let inv = 1.0 / n.max(1e-6) * scale;
                for x in v.iter_mut() {
                    *x *= inv;
                }
            }
        }
        let ab = unsafe { abp.slice(b * 2 * nv, 2 * nv) };
        for h in 0..nv {
            let beta = sigmoid(ab[h]);
            let g = a_log[h] * softplus(ab[nv + h] + dt_bias[h]);
            ab[h] = beta;
            ab[nv + h] = g.exp();
        }
    };
    if parallel {
        pool.run(body);
        stats.barriers += 1;
    } else {
        body(0, 1);
    }
}

/// The delta rule: tasks are v-heads; a task runs its head through every row in order.
///
/// `shared` says whether every row writes to one cache (then the state is read once, not once per row); it
/// only moves the byte accounting.
#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_rule(
    big: &[f32],
    big_w: usize,
    abv: &[f32],
    out: &mut [f32],
    rp: RowPtrs,
    spp: SnapPtrs,
    cfg: &RecurrentCfg,
    nb: usize,
    positions: &[usize],
    shared: bool,
    pool: &Pool,
    stats: &mut StepStats,
) {
    let (nv, nk, dk, dv) = (cfg.n_v_heads, cfg.n_k_heads, cfg.dk, cfg.dv);
    let (kd, vd) = (cfg.key_dim(), cfg.value_dim());
    let out = SharedBuf(out.as_mut_ptr());
    // Two tasks per head (each half of the state's rows), claimed dynamically so fast cores take more.
    const SPLIT: usize = 2;
    let next = AtomicUsize::new(0);
    pool.run(|_wk, _nw| loop {
        let task = next.fetch_add(1, AO::Relaxed);
        if task >= nv * SPLIT {
            break;
        }
        let (hv, part) = (task / SPLIT, task % SPLIT);
        let (j0, j1) = (part * dv / SPLIT, (part + 1) * dv / SPLIT);
        for b in 0..nb {
            // SAFETY: `ptrs` outlives the call; rows j0..j1 of head `hv` of row `b`'s state are this task's.
            let (sp, _) = unsafe { rp.at(b) };
            let s = unsafe {
                std::slice::from_raw_parts_mut(sp.add((hv * dv + j0) * dk), (j1 - j0) * dk)
            };
            let y = &big[b * big_w..];
            let kh = hv % nk;
            let q = &y[kh * dk..(kh + 1) * dk];
            let k = &y[kd + kh * dk..kd + (kh + 1) * dk];
            let v = &y[2 * kd + hv * dv..2 * kd + (hv + 1) * dv];
            let beta = abv[b * 2 * nv + hv];
            let decay = abv[b * 2 * nv + nv + hv];
            let o = unsafe { out.slice(b * vd + hv * dv + j0, j1 - j0) };
            crate::delta::delta_head(s, q, k, &v[j0..j1], beta, decay, o, dk);
            // A speculative verification's snapshot: this task's slice of the state after its row.
            if let Some((sp, sstate, _)) = unsafe { spp.at(b) } {
                if positions[b] == sp {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            s.as_ptr(),
                            sstate.add((hv * dv + j0) * dk),
                            s.len(),
                        )
                    };
                }
            }
        }
    });
    stats.barriers += 1;
    stats.bytes_kv += (2 * cfg.state_floats() * 4 * if shared { 1 } else { nb }) as u64;
    stats.macs += (4 * nv * dv * dk * nb) as u64;
}

/// Gated RMSNorm of one row of the GDN output, per head: rms(o_h)·w ⊙ silu(z_h).
pub(crate) fn gated_norm_row(
    o: &mut [f32],
    z: &[f32],
    norm_w: &[f32],
    nv: usize,
    dv: usize,
    eps: f32,
) {
    for h in 0..nv {
        let oh = &mut o[h * dv..(h + 1) * dv];
        let zh = &z[h * dv..(h + 1) * dv];
        let ss = crate::vmath::sumsq_f64(oh) / dv as f64;
        let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
        for i in 0..dv {
            oh[i] *= inv * norm_w[i];
        }
        crate::vmath::mul_silu(oh, zh);
    }
}

/// Every row's (state, conv) pointers for a recurrent layer, taken once per cache (rows of one cache share
/// them): re-borrowing a cache per row would invalidate the pointers taken before.
///
/// `runs` are the runs of consecutive rows sharing a cache (`(first row, rows, cache)`), `ptrs` the cache's
/// pointers for the layer being processed. `out` is reused scratch — a decode step runs this four times per
/// layer and must not allocate.
pub(crate) fn rec_ptrs(
    runs: &[(usize, usize, usize)],
    mut ptrs: impl FnMut(usize) -> (*mut f32, *mut f32),
    out: &mut [(*mut f32, *mut f32)],
) {
    for &(r0, len, ci) in runs {
        let p = ptrs(ci);
        for slot in &mut out[r0..r0 + len] {
            *slot = p;
        }
    }
}

/// Every row's snapshot request for a recurrent layer, taken once per cache: the position asked for and the
/// snapshot's state and conv buffers, if the cache asked for one.
pub(crate) fn snap_ptrs(
    runs: &[(usize, usize, usize)],
    mut ptrs: impl FnMut(usize) -> Option<(usize, *mut f32, *mut f32)>,
    out: &mut [Option<(usize, *mut f32, *mut f32)>],
) {
    for &(r0, len, ci) in runs {
        let p = ptrs(ci);
        for slot in &mut out[r0..r0 + len] {
            *slot = p;
        }
    }
}
