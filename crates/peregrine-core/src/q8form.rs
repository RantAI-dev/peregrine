//! The resident 8-bit prompt form (S5.4b): a per-matrix copy of the Q4_K and Q6_K
//! weights re-rounded onto one 8-bit grid per 256-weight super-block, built at load
//! when `--fast-prompt` is on.
//!
//! The form is `peregrine_kernels::kq8`'s layout: eight output rows interleaved per
//! super-block, one f32 scale and one biased byte per weight. The kernel that
//! multiplies it is the opt-in fast prompt mode's; the design and its measurements
//! are `docs/design/s5.4-fast-prompt.md`. This module is the load-time half: it
//! picks the matrices, lays the runs of eight rows out and fills the store with the
//! scalar twin (`build_q4k`, `build_q6k`), which is the definition (ADR 0005).
//!
//! # Which matrices
//!
//! Every two-dimensional Q4_K and Q6_K matrix except the token embedding, which is
//! gathered and never multiplied. Q5_0, Q5_K, Q8_0, the ternary formats and the
//! float formats have no form in this step: their prompt rows keep the exact
//! kernels (Q5_0's own line is S5.4c's call).
//!
//! # Which rows
//!
//! A run holds eight rows, as the kernel does. The rows past the last whole run
//! stay on the file's bytes and are multiplied by the exact kernel, so a row's
//! result still depends only on its index — and the form's bytes are never read
//! for them.
//!
//! # The store
//!
//! The runs of one matrix sit together, so a matrix's form is one contiguous run
//! and a run of eight rows inside it is too. The store is anonymous memory with
//! the huge-page request the packed stores use (`PEREGRINE_TILE_HUGEPAGES=0` turns
//! it off).
//!
//! Unlike the k-quant tiles and the float panels, the form is not written to the
//! pack cache: its entry would need a third section and a new file header version
//! (S4.9e's format has room for two stores), and the build is cheap enough — the
//! scalar twin converts 1.4–2.0 G weights a second, so 0.75 G weights of the 1B
//! cost about half a second — that the first build of this step leaves the cache
//! alone. Adding the section is a follow-up.

use crate::gguf::{Format, Gguf, TensorInfo};
use crate::kq::TILE_ROWS;
use crate::model::KqStore;
use peregrine_kernels::kq8::Q8K_SIZE;

/// The token embedding: gathered row by row, never multiplied, so it has no form.
const EMBEDDING: &str = "token_embd.weight";

/// One matrix's place in the form store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q8Entry {
    /// The tensor's name in the file.
    pub name: String,
    /// The matrix's format: Q4_K or Q6_K.
    pub kind: Format,
    /// Where the matrix's form starts in the store.
    pub offset: usize,
    /// How many bytes it takes.
    pub len: usize,
    /// The rows the form holds: a whole number of runs of eight.
    pub rows: usize,
    /// The matrix's input width.
    pub cols: usize,
}

impl Q8Entry {
    /// The 256-weight super-blocks of one row.
    pub fn super_blocks(&self) -> usize {
        self.cols / 256
    }
}

/// Whether a tensor gets a form in this step.
fn eligible(t: &TensorInfo) -> bool {
    matches!(t.kind, Format::Q4_K | Format::Q6_K)
        && t.dims.len() == 2
        && t.name != EMBEDDING
        && t.dims[1] as usize >= TILE_ROWS
        && t.dims[0] as usize % 256 == 0
}

/// The matrices the form store holds, in store order (the file's tensor order, so
/// a file always produces the same store).
pub fn layout(g: &Gguf) -> Vec<Q8Entry> {
    let mut mats: Vec<(&String, &TensorInfo)> =
        g.tensors.iter().filter(|(_, t)| eligible(t)).collect();
    mats.sort_by_key(|(name, t)| (t.offset, name.as_str()));
    let mut entries = Vec::with_capacity(mats.len());
    let mut at = 0usize;
    for (name, t) in mats {
        let rows = (t.dims[1] as usize / TILE_ROWS) * TILE_ROWS;
        let cols = t.dims[0] as usize;
        let n_sb = cols / 256;
        let len = (rows / TILE_ROWS) * n_sb * Q8K_SIZE;
        entries.push(Q8Entry {
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

/// Build every matrix's form and record in the file's tensor directory where it
/// sits (`TensorInfo::q8_off`).
///
/// The weights are read from the file's own bytes rather than from the repacked
/// k-quant store: the form does not depend on `PEREGRINE_KREPACK`, and a scalar
/// tier (where the repack is off by default) still gets the mode.
pub fn build(g: &mut Gguf, entries: &[Q8Entry]) -> KqStore {
    let total: usize = entries.iter().map(|e| e.len).sum();
    let mut store = KqStore::new(total);
    {
        let dst = store.bytes_mut();
        for e in entries {
            let t = g
                .tensor(&e.name)
                .expect("the matrix is in the file's own directory");
            build_matrix(g, e, t, &mut dst[e.offset..e.offset + e.len]);
        }
    }
    for e in entries {
        if let Some(t) = g.tensors.get_mut(&e.name) {
            t.q8_off = Some(e.offset);
        }
    }
    store
}

/// Fill one matrix's form, a run of eight rows at a time, from the file's bytes.
///
/// The eight rows' blocks are repacked into a one-super-block k-quant tile (the
/// layout the scalar twin reads) and the twin builds the form from it. The file's
/// mapping is read in 8-row groups, so a page is faulted exactly when its weights
/// are used.
fn build_matrix(file: &Gguf, e: &Q8Entry, t: &TensorInfo, dst: &mut [u8]) {
    let blk = e.kind.block_bytes();
    let n_sb = e.super_blocks();
    let rb = n_sb * blk;
    let groups = e.rows / TILE_ROWS;
    let tile_bytes = match e.kind {
        Format::Q4_K => peregrine_kernels::kq::Q4K8::SIZE,
        _ => peregrine_kernels::kq::Q6K8::SIZE,
    };
    let mk = peregrine_kernels::table::MatKernels::get();
    let mut rows = vec![0u8; TILE_ROWS * rb];
    let mut tile = vec![0u8; n_sb * tile_bytes];
    for gr in 0..groups {
        let at = t.offset as usize + gr * TILE_ROWS * rb;
        file.read_data(at, &mut rows)
            .expect("the matrix's bytes are in the file");
        for sb in 0..n_sb {
            let mut blocks: [&[u8]; TILE_ROWS] = [&[]; TILE_ROWS];
            for (r, b) in blocks.iter_mut().enumerate() {
                *b = &rows[r * rb + sb * blk..r * rb + sb * blk + blk];
            }
            let out = &mut tile[sb * tile_bytes..(sb + 1) * tile_bytes];
            match e.kind {
                Format::Q4_K => peregrine_kernels::kq::repack_q4k8(&blocks, out),
                _ => peregrine_kernels::kq::repack_q6k8(&blocks, out),
            }
        }
        let out = &mut dst[gr * n_sb * Q8K_SIZE..(gr + 1) * n_sb * Q8K_SIZE];
        mk.q8k_build(e.kind, &tile, n_sb, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_two_dimensional_q4k_and_q6k_matrices_other_than_the_embedding_are_taken() {
        let t = |name: &str, kind: Format, dims: Vec<u64>| TensorInfo {
            name: name.into(),
            dims,
            kind,
            offset: 0,
            tile_off: None,
            kq_off: None,
            fp_off: None,
            fp_kind: None,
            q8_off: None,
        };
        assert!(eligible(&t("a", Format::Q4_K, vec![256, 8])));
        assert!(eligible(&t("b", Format::Q6_K, vec![512, 16])));
        assert!(!eligible(&t(
            "token_embd.weight",
            Format::Q6_K,
            vec![256, 8]
        )));
        assert!(!eligible(&t("c", Format::Q5_0, vec![256, 8])));
        assert!(!eligible(&t("d", Format::Q8_0, vec![256, 8])));
        assert!(!eligible(&t("e", Format::Q4_K, vec![256, 4])));
        assert!(!eligible(&t("f", Format::Q4_K, vec![128, 8])));
        assert!(!eligible(&t("g", Format::F16, vec![256, 8])));
    }
}
