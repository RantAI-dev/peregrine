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
//! * a matrix that is not two-dimensional (the mixture-of-experts' experts are
//!   three-dimensional and are not packed yet).
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

/// One matrix's place in the float store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FpEntry {
    /// The tensor's name in the file.
    pub name: String,
    /// The matrix's format: one of the three float formats.
    pub kind: Format,
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

/// The matrices the store holds, in store order.
///
/// The order is the file's own: the matrices are laid out by their offset in the
/// data section, so a file always produces the same store.
pub fn layout(g: &Gguf) -> Vec<FpEntry> {
    let mut mats: Vec<(&String, &TensorInfo)> =
        g.tensors.iter().filter(|(_, t)| eligible(t)).collect();
    mats.sort_by_key(|(name, t)| (t.offset, name.as_str()));
    let mut entries = Vec::with_capacity(mats.len());
    let mut at = 0usize;
    for (name, t) in mats {
        let rows = t.dims[1] as usize;
        let cols = t.dims[0] as usize;
        let len = peregrine_kernels::fpack::packed_len(t.kind, rows, cols);
        entries.push(FpEntry {
            name: name.clone(),
            kind: t.kind,
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
    let entries = layout(g);
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
    for e in &entries {
        if let Some(t) = g.tensors.get_mut(&e.name) {
            t.fp_off = Some(e.offset);
        }
    }
    store
}

/// Write one matrix's panels, reading the file in large sequential chunks.
///
/// One `read_at` per panel is a syscall every 32 KB; a chunk of up to 64 MiB keeps
/// the reads sequential and the syscall count small, and the pack then works on
/// memory already in cache. The chunk is whole panels, so a panel never straddles one.
fn pack_matrix(file: &Gguf, e: &FpEntry, t: &TensorInfo, dst: &mut [u8]) {
    const CHUNK_BYTES: usize = 64 << 20;
    let pb = peregrine_kernels::fpack::panel_bytes(e.kind, e.cols);
    let rb = e.cols * peregrine_kernels::fpack::elem_bytes(e.kind);
    let per_chunk = (CHUNK_BYTES / rb.max(1)).max(PANEL).div_ceil(PANEL) * PANEL;
    let mut buf = vec![0u8; per_chunk.min(e.rows) * rb];
    let mut first = 0usize;
    while first < e.rows {
        let n = (e.rows - first).min(per_chunk);
        let at = t.offset as usize + first * rb;
        file.read_data(at, &mut buf[..n * rb])
            .expect("the matrix's bytes are in the file");
        for local in (0..n).step_by(PANEL) {
            let p = (first + local) / PANEL;
            let k = (n - local).min(PANEL);
            let refs: Vec<&[u8]> = (0..k)
                .map(|r| &buf[(local + r) * rb..(local + r + 1) * rb])
                .collect();
            peregrine_kernels::fpack::pack_panel(
                e.kind,
                &refs,
                e.cols,
                &mut dst[p * pb..(p + 1) * pb],
            );
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
            kq_off: None,
            fp_off: None,
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
            kind: Format::F16,
            offset: 0,
            len: peregrine_kernels::fpack::packed_len(Format::F16, 17, 8),
            rows: 17,
            cols: 8,
        };
        assert_eq!(e.panels(), 2);
        assert_eq!(e.len, 2 * 8 * 16 * 2);
    }
}
