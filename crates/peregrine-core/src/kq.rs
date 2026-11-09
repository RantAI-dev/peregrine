//! The load-time k-quant repack (S3.4, step 1): the Q4_K and Q6_K matrices of a
//! model file, rewritten into eight-row tiles, and the Q5_0 and Q8_0 matrices of a
//! model whose shape forces the legacy format (the small-model fixes).
//!
//! The repacked GEMM of S3.4 multiplies eight output rows at once, so the weights of
//! eight rows have to sit side by side; the layout is [`peregrine_kernels::kq`]'s.
//! This module is the load-time half: it picks the matrices, lays the tiles out and
//! fills the store.
//!
//! # Which matrices
//!
//! Every Q4_K, Q6_K, Q5_0 and Q8_0 matrix of the file, except:
//!
//! * the token embedding, which is gathered row by row and never multiplied, so the
//!   repacked layout would mean nothing to it — unless the embedding *is* the output
//!   head (a tied-head model), which is the one place a Q8_0 embedding is read as a
//!   matrix; then the file's bytes still feed the gather and the store holds the
//!   head's tiles;
//! * a matrix that is not two-dimensional (the mixture-of-experts' experts are
//!   three-dimensional and are not repacked yet).
//!
//! The tied-head copy is limited to Q8_0: it costs the embedding's bytes again in
//! resident memory, and the Q8_0 head is the one the decode profile showed below the
//! memory-bandwidth limit. The other formats' tied heads keep the row kernel they
//! have today.
//!
//! # Which rows
//!
//! A tile holds eight rows, so a matrix's rows are repacked eight at a time and the
//! rows past the last whole group of eight stay on the file's own bytes: they are
//! multiplied by today's kernels, which is also what keeps a row's result dependent
//! on its index alone.
//!
//! # The store
//!
//! The tiles of one matrix sit together, group by group and super-block by
//! super-block, so a matrix's repacked bytes are one contiguous run and a group of
//! eight rows is one contiguous run inside it.

use crate::gguf::{Format, Gguf, TensorInfo};
use crate::model::KqStore;

/// The rows one tile holds.
pub const TILE_ROWS: usize = 8;

/// The token embedding: gathered row by row, never multiplied, so it is not repacked.
const EMBEDDING: &str = "token_embd.weight";

/// The bytes one repacked tile of the given format takes, or `None` for the formats
/// that have no repacked layout.
pub fn tile_size(kind: Format) -> Option<usize> {
    match kind {
        Format::Q4_K => Some(peregrine_kernels::kq::Q4K8::SIZE),
        Format::Q6_K => Some(peregrine_kernels::kq::Q6K8::SIZE),
        Format::Q5_0 => Some(peregrine_kernels::kq::Q5_08::SIZE),
        Format::Q8_0 => Some(peregrine_kernels::kq::Q8_08::SIZE),
        _ => None,
    }
}

/// One matrix's place in the k-quant store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KqEntry {
    /// The tensor's name in the file.
    pub name: String,
    /// The matrix's format: one of the four the store repacks.
    pub kind: Format,
    /// Where the matrix's repacked bytes start in the store.
    pub offset: usize,
    /// How many bytes they take.
    pub len: usize,
    /// The rows the store holds: a whole number of groups of eight. The rows past
    /// them stay on the file's bytes.
    pub rows: usize,
    /// The matrix's input width.
    pub cols: usize,
}

impl KqEntry {
    /// The super-blocks a row of this matrix holds.
    pub fn super_blocks(&self) -> usize {
        self.cols / self.kind.block()
    }
}

/// The matrices the k-quant store holds, in store order.
///
/// The order is the file's own: the matrices are laid out by their offset in the
/// data section, so a file always produces the same store.
pub fn layout(g: &Gguf) -> Vec<KqEntry> {
    // A file with no `output.weight` uses its token embedding as the output head.
    // The embedding is normally left out — it is gathered, not multiplied — but a
    // tied Q8_0 head is a matrix the repacked kernel should read, so it is added.
    // The file's bytes still serve the gather: the store is a second copy of the
    // same weights, reordered.
    let tied_q8_head = !g.tensors.contains_key("output.weight");
    let mut mats: Vec<(&String, &TensorInfo)> = g
        .tensors
        .iter()
        .filter(|(name, t)| {
            eligible(t) || (tied_q8_head && name.as_str() == EMBEDDING && t.kind == Format::Q8_0)
        })
        .collect();
    mats.sort_by_key(|(name, t)| (t.offset, name.as_str()));
    let mut entries = Vec::with_capacity(mats.len());
    let mut at = 0usize;
    for (name, t) in mats {
        let rows = (t.dims[1] as usize / TILE_ROWS) * TILE_ROWS;
        let cols = t.dims[0] as usize;
        let n_sb = cols / t.kind.block();
        let len = (rows / TILE_ROWS) * n_sb * tile_size(t.kind).unwrap_or(0);
        entries.push(KqEntry {
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

/// Whether a tensor is one of the matrices the store takes.
fn eligible(t: &TensorInfo) -> bool {
    tile_size(t.kind).is_some()
        && t.dims.len() == 2
        && t.name != EMBEDDING
        && t.dims[1] as usize >= TILE_ROWS
        && t.dims[0] as usize % t.kind.block() == 0
}

/// Repack a model file's k-quant matrices into the store, and record in the file's
/// tensor directory where each matrix's repacked bytes are (`TensorInfo::kq_off`).
///
/// The families read the file after this, so every matrix they hold carries the
/// offset of its repacked bytes.
pub fn repack(g: &mut Gguf) -> KqStore {
    let entries = layout(g);
    let total: usize = entries.iter().map(|e| e.len).sum();
    let mut store = KqStore::new(total);
    {
        let dst = store.bytes_mut();
        for e in &entries {
            let t = g
                .tensor(&e.name)
                .expect("the matrix is in the file's own directory");
            repack_matrix(g, e, t, &mut dst[e.offset..e.offset + e.len]);
        }
    }
    for e in &entries {
        if let Some(t) = g.tensors.get_mut(&e.name) {
            t.kq_off = Some(e.offset);
        }
    }
    store
}

/// Write one matrix's repacked tiles.
///
/// The matrix's rows are read from the file, a group of eight at a time, rather
/// than through the engine's mapping: the mapping's pages for them are then
/// never faulted in, and the model does not cost twice what it should (S3.4,
/// decision 2).
fn repack_matrix(file: &Gguf, e: &KqEntry, t: &TensorInfo, dst: &mut [u8]) {
    let tile = match tile_size(e.kind) {
        Some(t) => t,
        None => return,
    };
    let blk = e.kind.block_bytes();
    let n_sb = e.super_blocks();
    let rb = n_sb * blk;
    let groups = e.rows / TILE_ROWS;
    let mut rows = vec![0u8; TILE_ROWS * rb];
    for g in 0..groups {
        // The group's eight rows are consecutive in the file.
        let at = t.offset as usize + g * TILE_ROWS * rb;
        file.read_data(at, &mut rows)
            .expect("the matrix's bytes are in the file");
        for s in 0..n_sb {
            let at = (g * n_sb + s) * tile;
            let mut blocks: [&[u8]; TILE_ROWS] = [&[]; TILE_ROWS];
            for (r, b) in blocks.iter_mut().enumerate() {
                *b = &rows[r * rb + s * blk..r * rb + s * blk + blk];
            }
            let out = &mut dst[at..at + tile];
            match e.kind {
                Format::Q4_K => peregrine_kernels::kq::repack_q4k8(&blocks, out),
                Format::Q5_0 => peregrine_kernels::kq::repack_q5_08(&blocks, out),
                Format::Q8_0 => peregrine_kernels::kq::repack_q8_08(&blocks, out),
                _ => peregrine_kernels::kq::repack_q6k8(&blocks, out),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tile_sizes_are_the_layouts_own() {
        assert_eq!(tile_size(Format::Q4_K), Some(1152));
        assert_eq!(tile_size(Format::Q6_K), Some(1680));
        assert_eq!(tile_size(Format::Q5_0), Some(176));
        assert_eq!(tile_size(Format::Q8_0), Some(272));
        assert_eq!(tile_size(Format::F16), None);
        assert_eq!(tile_size(Format::Q4_0), None);
    }
}
