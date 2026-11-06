//! Embedding table lookup block.

use crate::gguf::TensorInfo;
use crate::model::{StepStats, Weights};
use crate::quant::row_bytes;

/// One row of the embedding table, dequantised. The only place a weight is
/// decoded into a buffer — it is a single row of 960, not a tensor.
#[inline]
pub(crate) fn embed(
    tok_embd: &TensorInfo,
    w: &Weights,
    token: u32,
    out: &mut [f32],
    st: &mut StepStats,
) {
    let t = tok_embd;
    let n = t.dims[0] as usize;
    let rb = row_bytes(t.kind, n);
    let all = w.bytes(t);
    let off = token as usize * rb;
    crate::quant::dequant_row(t.kind, &all[off..off + rb], &mut out[..n]);
    st.bytes_weights += rb as u64;
}
