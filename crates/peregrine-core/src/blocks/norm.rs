//! Root-mean-square normalisation blocks.

use crate::blocks::par::{SharedActs, SharedBuf};
use crate::blocks::ROW_PARALLEL;
use crate::gguf::{f16, Format, TensorInfo};
use crate::model::{StepStats, Weights};
use crate::pool::{share, Pool};

#[inline]
fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
}

/// RMSNorm of one attention head in place, against its `head_dim` weights (Qwen3's
/// q/k norm). Summed in f64 like `rmsnorm_into`, so both norms round the same way.
#[inline]
pub(crate) fn head_rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let ss = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
    for (v, g) in x.iter_mut().zip(w) {
        *v = *v * inv * g;
    }
}

pub(crate) fn rmsnorm_into(w: &Weights, weight: &TensorInfo, eps: f32, x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let wb = w.bytes(weight);
    let ss = crate::vmath::sumsq_f64(x) / n as f64;
    let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
    match weight.kind {
        Format::F32 => {
            // read the weight as f32 in place when it is aligned (GGUF aligns tensor data), so the loop
            // vectorises; the byte-by-byte decode was most of a norm's time.
            // SAFETY: any 4 bytes are a valid f32; `align_to` only yields a middle of whole, aligned elements.
            let (pre, wf, _) = unsafe { wb.align_to::<f32>() };
            if pre.is_empty() && wf.len() >= n {
                for ((o, &xi), &wi) in out[..n].iter_mut().zip(&x[..n]).zip(&wf[..n]) {
                    *o = xi * inv * wi;
                }
            } else {
                for i in 0..n {
                    out[i] = x[i] * inv * f32_at(wb, i);
                }
            }
        }
        Format::F16 => {
            for i in 0..n {
                out[i] = x[i] * inv * f16(u16::from_le_bytes([wb[i * 2], wb[i * 2 + 1]]));
            }
        }
        other => {
            // A quantised norm vector would be a model we cannot read correctly;
            // refuse rather than approximate.
            panic!("norm weight in {} is not supported", other.label());
        }
    }
}

pub(crate) fn rmsnorm(
    w: &Weights,
    weight: &TensorInfo,
    eps: f32,
    x: &[f32],
    out: &mut [f32],
    st: &mut StepStats,
) {
    rmsnorm_into(w, weight, eps, x, out);
    st.bytes_weights += weight.bytes();
}

/// For every batch row: `xb[b] = rmsnorm(x[b])`, then quantise it into `acts[b]`.
///
/// Rows are independent, so with a batch (prefill, speculative verification) they
/// are split across the pool. This was a fifth of prefill time on one thread.
#[allow(clippy::too_many_arguments)]
pub(crate) fn norm_rows(
    w: &Weights,
    weight: &TensorInfo,
    d_model: usize,
    eps: f32,
    x: &[f32],
    xb: SharedBuf,
    acts: SharedActs,
    nb: usize,
    pool: &Pool,
    stats: &mut StepStats,
) {
    let d = d_model;
    let body = |wk: usize, nw: usize| {
        let (lo, hi) = share(nb, wk, nw);
        for b in lo..hi {
            // SAFETY: row `b` of `xb` and `acts[b]` belong to this worker alone.
            let out = unsafe { xb.slice(b * d, d) };
            rmsnorm_into(w, weight, eps, &x[b * d..(b + 1) * d], out);
            unsafe { acts.get(b) }.fill(out);
        }
    };
    if nb >= ROW_PARALLEL && pool.width() > 1 {
        pool.run(body);
        stats.barriers += 1;
    } else {
        body(0, 1);
    }
    // Once per call, like the matmuls: the vector crosses the memory interface
    // once however many rows reuse it.
    stats.bytes_weights += weight.bytes();
}
