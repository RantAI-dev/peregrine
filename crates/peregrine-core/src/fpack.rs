//! The load-time float pack (S4.9f): the F32, F16 and BF16 matrices of a model
//! file, rewritten into 16-output-row panels.
//!
//! The packed GEMM of S4.9f multiplies sixteen output rows at once, so the weights
//! of sixteen rows have to sit side by side; the layout is
//! [`peregrine_kernels::fpack`]'s. This module is the load-time half: it picks the
//! matrices, lays the panels out and fills the store.
//!
//! # Which matrices
//!
//! Every F32, F16 and BF16 matrix of the file, except:
//!
//! * the token embedding, which is gathered row by row and never multiplied. When
//!   the embedding *is* the output head (a tied-head model) the packed panels
//!   would be a second full copy of it — 311 MB on the 0.6B f16 encoder — for a
//!   matmul whose bytes are the same either way and whose decode step is bound by
//!   those bytes, not by the multiplies. It keeps the per-row path;
//! * a matrix that is not two-dimensional — unless the family asks for its expert
//!   banks ([`layout_with`]'s `experts`, S4.8b): a three-dimensional
//!   `[d_in, d_out, n_expert]` tensor is then packed **per expert**, each expert a
//!   two-dimensional matrix of its own. Every other mixture-of-experts family keeps
//!   the per-row path and its exact bytes.
//!
//! # The store
//!
//! The panels of one matrix sit together, so a matrix's packed bytes are one
//! contiguous run. The store is filled by reading the file rather than through the
//! mapping (`Gguf::read_data`), so the matrix's pages are never faulted in and the
//! model does not cost twice what it should — the same rule the k-quant repack
//! follows. What it costs is one copy of every packed matrix's bytes, in the
//! file's own element type: the same size as the file's copy, plus the zero
//! padding of each matrix's last panel (at most 15 rows).

use crate::gguf::{Format, Gguf, TensorInfo};
use crate::model::FpStore;

/// The output rows one panel holds.
pub const PANEL: usize = peregrine_kernels::fpack::PANEL;

/// The token embedding: gathered row by row, never multiplied, so it is not packed.
const EMBEDDING: &str = "token_embd.weight";

/// What a Q8_0 matrix is dequantised into, when a family asks for it.
///
/// `Float` is the exact form (`scale × code` in f32) the small `bert` encoders take
/// (S4.9m); `Half` is the same value written in f16 panels, half the bytes, for a
/// family whose weights are too large for the f32 form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q8Panels {
    /// Keep the file's Q8_0 bytes (the int8 kernels).
    Keep,
    /// Dequantise into f32 panels.
    Float,
    /// Dequantise into f16 panels.
    Half,
}

impl Q8Panels {
    /// The panel format a converted Q8_0 matrix is written in.
    fn kind(self) -> Option<Format> {
        match self {
            Q8Panels::Keep => None,
            Q8Panels::Float => Some(Format::F32),
            Q8Panels::Half => Some(Format::F16),
        }
    }
}

/// One matrix's place in the float store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FpEntry {
    /// The tensor's name in the file. A three-dimensional expert bank writes one
    /// entry per expert under this name; `expert` says which one.
    pub name: String,
    /// The expert of a three-dimensional bank this entry is, `None` for a matrix.
    pub expert: Option<usize>,
    /// The matrix's format in the panel store: one of the three float formats.
    pub kind: Format,
    /// The matrix's format in the file. Different from `kind` only for a Q8_0 matrix
    /// dequantised into f32 panels at load (S4.9m).
    pub source: Format,
    /// Where the matrix's panels start in the store.
    pub offset: usize,
    /// How many bytes they take (whole panels, the last one zero-padded).
    pub len: usize,
    /// The matrix's output rows.
    pub rows: usize,
    /// The matrix's input width.
    pub cols: usize,
}

impl FpEntry {
    /// The panels the matrix holds.
    pub fn panels(&self) -> usize {
        self.rows.div_ceil(PANEL)
    }
}

/// Whether a tensor takes the packed float layout.
pub fn eligible(t: &TensorInfo) -> bool {
    matches!(t.kind, Format::F32 | Format::F16 | Format::BF16)
        && t.dims.len() == 2
        && t.name != EMBEDDING
        && t.dims[0] > 0
        && t.dims[1] > 0
}

/// Whether one expert of a three-dimensional bank can be packed as a matrix of its
/// own: a float format, three dimensions, a whole number of blocks per expert.
pub fn expert_eligible(t: &TensorInfo) -> bool {
    matches!(t.kind, Format::F32 | Format::F16 | Format::BF16)
        && t.dims.len() == 3
        && t.dims[0] > 0
        && t.dims[1] > 0
        && t.dims[2] > 0
        && (t.dims[0] * t.dims[1]).is_multiple_of(t.kind.block() as u64)
}

/// Whether a Q8_0 matrix can be dequantised into f32 panels (S4.9m): two-dimensional,
/// not the gathered embedding, and a whole number of 32-weight blocks.
pub fn convertible(t: &TensorInfo) -> bool {
    t.kind == Format::Q8_0
        && t.dims.len() == 2
        && t.name != EMBEDDING
        && t.dims[0] > 0
        && t.dims[1] > 0
        && t.dims[0].is_multiple_of(32)
}

/// Whether one expert of a three-dimensional Q8_0 bank can be dequantised into f32
/// panels: the same rules, one dimension up.
pub fn expert_convertible(t: &TensorInfo) -> bool {
    t.kind == Format::Q8_0
        && t.dims.len() == 3
        && t.dims[0] > 0
        && t.dims[1] > 0
        && t.dims[2] > 0
        && t.dims[0].is_multiple_of(32)
}

/// The matrices the store holds, in store order, without conversion.
pub fn layout(g: &Gguf) -> Vec<FpEntry> {
    layout_with(g, Q8Panels::Keep, false)
}

/// The matrices the store holds, in store order; `q8_as_f32` adds every
/// [`convertible`] Q8_0 matrix, its panels written in F32 (S4.9m), and `experts`
/// adds every three-dimensional float bank as one matrix per expert (S4.8b).
///
/// The order is the file's own: the matrices are laid out by their offset in the
/// data section, so a file always produces the same store.
pub fn layout_with(g: &Gguf, q8: Q8Panels, experts: bool) -> Vec<FpEntry> {
    let convert = q8.kind();
    let mut mats: Vec<(&String, &TensorInfo, Option<usize>)> = Vec::new();
    for (name, t) in &g.tensors {
        if eligible(t) || (convert.is_some() && convertible(t)) {
            mats.push((name, t, None));
        } else if experts && (expert_eligible(t) || (convert.is_some() && expert_convertible(t))) {
            for e in 0..t.dims[2] as usize {
                mats.push((name, t, Some(e)));
            }
        }
    }
    mats.sort_by_key(|(name, t, e)| (t.offset, name.as_str(), *e));
    let mut entries = Vec::with_capacity(mats.len());
    let mut at = 0usize;
    for (name, t, expert) in mats {
        // A bank's expert is `[dims[0], dims[1]]`; the bank itself is a matrix too, but
        // the families that use one read it through its experts.
        let (rows, cols) = (t.dims[1] as usize, t.dims[0] as usize);
        let kind = match (convertible(t), expert_convertible(t), convert) {
            (true, _, Some(f)) | (_, true, Some(f)) => f,
            _ => t.kind,
        };
        let len = peregrine_kernels::fpack::packed_len(kind, rows, cols);
        entries.push(FpEntry {
            name: name.clone(),
            expert,
            kind,
            source: t.kind,
            offset: at,
            len,
            rows,
            cols,
        });
        at += len;
    }
    entries
}

/// Pack a model file's float matrices into the store, and record in the file's
/// tensor directory where each matrix's packed bytes are (`TensorInfo::fp_off`).
///
/// The families read the file after this, so every matrix they hold carries the
/// offset of its packed panels.
pub fn repack(g: &mut Gguf) -> FpStore {
    repack_with(g, Q8Panels::Keep, false)
}

/// [`repack`], converting every [`convertible`] Q8_0 matrix into f32 panels when
/// `q8_as_f32` is set (S4.9m), and packing every three-dimensional float bank per
/// expert when `experts` is set (S4.8b). The tensor keeps its file format and gains
/// [`TensorInfo::fp_kind`], so a reader of the file's bytes is unaffected while the
/// float path sees an f32 panel.
///
/// An expert bank's tensor carries the **first expert's** offset: a family that
/// holds 2D views of the bank ([`crate::blocks::moe::expert_view`]) computes each
/// view's offset from it, one packed expert apart.
pub fn repack_with(g: &mut Gguf, q8: Q8Panels, experts: bool) -> FpStore {
    let entries = layout_with(g, q8, experts);
    let total: usize = entries.iter().map(|e| e.len).sum();
    let mut store = FpStore::new(total);
    {
        let dst = store.bytes_mut();
        for e in &entries {
            let t = g
                .tensor(&e.name)
                .expect("the matrix is in the file's own directory");
            pack_matrix(g, e, t, &mut dst[e.offset..e.offset + e.len]);
        }
    }
    apply(&mut g.tensors, &entries);
    store
}

/// Record in the tensor directory where each packed matrix sits, for a store that is
/// already in memory or was mapped from the pack cache.
///
/// An expert bank carries the **first** expert's offset: a family that holds 2D views
/// of the bank computes each view's own offset from it, one packed expert apart
/// ([`crate::blocks::moe::expert_view`]).
pub fn apply(tensors: &mut std::collections::HashMap<String, TensorInfo>, entries: &[FpEntry]) {
    for e in entries {
        let Some(t) = tensors.get_mut(&e.name) else {
            continue;
        };
        if e.expert.is_none_or(|x| x == 0) {
            t.fp_off = Some(e.offset);
            t.fp_kind = (e.kind != e.source).then_some(e.kind);
        }
    }
}

/// Write one matrix's panels, reading the file in large sequential chunks.
///
/// One `read_at` per panel is a syscall every 32 KB; a chunk of up to 64 MiB keeps
/// the reads sequential and the syscall count small, and the pack then works on
/// memory already in cache. The chunk is whole panels, so a panel never straddles one.
fn pack_matrix(file: &Gguf, e: &FpEntry, t: &TensorInfo, dst: &mut [u8]) {
    const CHUNK_BYTES: usize = 64 << 20;
    let pb = peregrine_kernels::fpack::panel_bytes(e.kind, e.cols);
    // The bytes the file holds per row, and the bytes one row takes in the store.
    let rb = crate::quant::row_bytes(e.source, e.cols);
    let out_row = e.cols * peregrine_kernels::fpack::elem_bytes(e.kind);
    let per_chunk = (CHUNK_BYTES / rb.max(1)).max(PANEL).div_ceil(PANEL) * PANEL;
    let mut buf = vec![0u8; per_chunk.min(e.rows) * rb];
    // A Q8_0 row decoded to f32, and the F32 bytes the panel writer reads; only used
    // when the matrix is converted, where one panel's rows are decoded at a time.
    let mut decoded = vec![0.0f32; PANEL * e.cols];
    let mut bytes = vec![0u8; PANEL * out_row];
    // One expert of a bank is a contiguous run of its rows; the bank's own offset is
    // the first expert's.
    let base = t.offset as usize + e.expert.unwrap_or(0) * e.rows * rb;
    let mut first = 0usize;
    while first < e.rows {
        let n = (e.rows - first).min(per_chunk);
        let at = base + first * rb;
        file.read_data(at, &mut buf[..n * rb])
            .expect("the matrix's bytes are in the file");
        for local in (0..n).step_by(PANEL) {
            let p = (first + local) / PANEL;
            let k = (n - local).min(PANEL);
            if e.source == e.kind {
                let refs: Vec<&[u8]> = (0..k)
                    .map(|r| &buf[(local + r) * rb..(local + r + 1) * rb])
                    .collect();
                peregrine_kernels::fpack::pack_panel(
                    e.kind,
                    &refs,
                    e.cols,
                    &mut dst[p * pb..(p + 1) * pb],
                );
            } else {
                // Dequantise this panel's rows (`scale × code` is exact in f32) and
                // write them in the panel writer's row form: f32, or f16 for a family
                // that keeps half the bytes.
                for r in 0..k {
                    let row = &buf[(local + r) * rb..(local + r + 1) * rb];
                    peregrine_kernels::quant::dequant_row(
                        e.source,
                        row,
                        &mut decoded[r * e.cols..(r + 1) * e.cols],
                    );
                }
                if e.kind == Format::F16 {
                    for (i, &v) in decoded[..k * e.cols].iter().enumerate() {
                        let h = crate::quant::f32_to_f16(v);
                        bytes[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
                    }
                } else {
                    for (i, &v) in decoded[..k * e.cols].iter().enumerate() {
                        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
                let refs: Vec<&[u8]> = (0..k)
                    .map(|r| &bytes[r * out_row..(r + 1) * out_row])
                    .collect();
                peregrine_kernels::fpack::pack_panel(
                    e.kind,
                    &refs,
                    e.cols,
                    &mut dst[p * pb..(p + 1) * pb],
                );
            }
        }
        first += n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_store_holds_every_float_matrix_but_the_embedding() {
        let ti = |name: &str, kind, dims: Vec<u64>| TensorInfo {
            name: name.into(),
            dims,
            kind,
            offset: 0,
            tile_off: None,
            q8_off: None,
            kq_off: None,
            fp_off: None,
            fp_kind: None,
        };
        assert!(eligible(&ti(
            "blk.0.attn_q.weight",
            Format::F16,
            vec![64, 32]
        )));
        assert!(eligible(&ti(
            "blk.0.attn_q.weight",
            Format::F32,
            vec![64, 32]
        )));
        assert!(eligible(&ti(
            "blk.0.attn_q.weight",
            Format::BF16,
            vec![64, 32]
        )));
        assert!(!eligible(&ti(
            "token_embd.weight",
            Format::F16,
            vec![64, 32]
        )));
        assert!(!eligible(&ti(
            "blk.0.attn_norm.weight",
            Format::F32,
            vec![64]
        )));
        assert!(!eligible(&ti(
            "blk.0.ffn_gate_exps.weight",
            Format::F16,
            vec![64, 32, 4]
        )));
        assert!(!eligible(&ti(
            "blk.0.attn_q.weight",
            Format::Q8_0,
            vec![64, 32]
        )));
    }

    #[test]
    fn the_packed_length_rounds_up_to_whole_panels() {
        let e = FpEntry {
            name: "w".into(),
            expert: None,
            kind: Format::F16,
            source: Format::F16,
            offset: 0,
            len: peregrine_kernels::fpack::packed_len(Format::F16, 17, 8),
            rows: 17,
            cols: 8,
        };
        assert_eq!(e.panels(), 2);
        assert_eq!(e.len, 2 * 8 * 16 * 2);
    }
}
