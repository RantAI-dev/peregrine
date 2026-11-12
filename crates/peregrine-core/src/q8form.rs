//! The resident 8-bit prompt form (S5.4b): a per-matrix copy of the Q4_K, Q6_K and
//! (since S5.4f, where the tier's table says so) Q5_0 weights re-rounded onto one
//! 8-bit grid per 256-weight super-block, built at load when `--fast-prompt` is on.
//!
//! The form is `peregrine_kernels::kq8`'s layout: eight output rows interleaved per
//! super-block, one f32 scale and one biased byte per weight. The kernel that
//! multiplies it is the opt-in fast prompt mode's; the design and its measurements
//! are `docs/design/s5.4-fast-prompt.md`. This module is the load-time half: it
//! picks the matrices, lays the runs of eight rows out and fills the store with the
//! scalar twin (`build_q4k`, `build_q6k`, `build_q5_0`), which is the definition
//! (ADR 0005).
//!
//! # Which matrices
//!
//! Every two-dimensional matrix of a format the tier's table selects
//! ([`peregrine_kernels::table::MatKernels::q8k_uses_form`]), except the token
//! embedding, which is gathered and never multiplied. On AVX2 that is Q6_K only;
//! on AVX-512 Q4_K and Q6_K (unchanged by S5.4f); on a tier without a SIMD body,
//! nothing, so the mode computes exactly what the exact engine computes. Q5_0 is
//! in the form's set: aarch64's `dotprod`/`i8mm` caps select it (S5.4h, §5h),
//! no x86 tier does (level with AVX2's exact kernel at 2048 weights and 0.88×
//! at Qwen 2.5's 896-wide rows, §5g).
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
use peregrine_kernels::table::MatKernels;

/// The token embedding: gathered row by row, never multiplied, so it has no form.
const EMBEDDING: &str = "token_embd.weight";

/// One matrix's place in the form store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q8Entry {
    /// The tensor's name in the file.
    pub name: String,
    /// The matrix's format: Q4_K, Q6_K or Q5_0.
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
    ///
    /// Q4_K and Q6_K rows are always whole super-blocks; a Q5_0 row may stop
    /// inside one (Qwen 2.5 0.5B's 896-wide rows), and the form's last super-block
    /// is then short — its missing codes are zero.
    pub fn super_blocks(&self) -> usize {
        self.cols.div_ceil(256)
    }
}

/// Whether a tensor's format has a form on this tier (S5.4f's table), and its
/// shape allows one.
///
/// Q4_K and Q6_K matrices must be whole 256-weight super-blocks wide (their exact
/// kernels are). Q5_0's rows are multiples of its 32-weight block, and a row that
/// stops inside a super-block keeps a short last one.
fn eligible(t: &TensorInfo) -> bool {
    if t.dims.len() != 2
        || t.name == EMBEDDING
        || (t.dims[1] as usize) < TILE_ROWS
        || !MatKernels::get().q8k_uses_form(t.kind)
    {
        return false;
    }
    let cols = t.dims[0] as usize;
    match t.kind {
        Format::Q4_K | Format::Q6_K => cols % 256 == 0,
        #[cfg(feature = "fmt-legacy")]
        Format::Q5_0 => cols % 32 == 0,
        _ => false,
    }
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
        let n_sb = cols.div_ceil(256);
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
    // A Q5_0 matrix whose row stops inside its last 256-weight super-block (Qwen
    // 2.5's 896-wide rows) is multiplied against a padded activation buffer, which
    // `QuantAct::fill` builds only when it knows the form will read it (S5.4h).
    if entries
        .iter()
        .any(|e| e.kind == Format::Q5_0 && !e.cols.is_multiple_of(256))
    {
        peregrine_kernels::kq8::note_padded_q5_form();
    }
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
    let rb = (e.cols / e.kind.block()) * blk;
    let groups = e.rows / TILE_ROWS;
    let mk = MatKernels::get();
    let mut rows = vec![0u8; TILE_ROWS * rb];
    match e.kind {
        Format::Q4_K | Format::Q6_K => {
            let tile_bytes = match e.kind {
                Format::Q4_K => peregrine_kernels::kq::Q4K8::SIZE,
                _ => peregrine_kernels::kq::Q6K8::SIZE,
            };
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
        #[cfg(feature = "fmt-legacy")]
        Format::Q5_0 => {
            // One Q5_0 block is 32 weights and the tile holds one block per row, so
            // the tile count is the row's block count, not its super-block count.
            let n_blk = e.cols / 32;
            let tile_bytes = peregrine_kernels::kq::Q5_08::SIZE;
            let mut tile = vec![0u8; n_blk * tile_bytes];
            for gr in 0..groups {
                let at = t.offset as usize + gr * TILE_ROWS * rb;
                file.read_data(at, &mut rows)
                    .expect("the matrix's bytes are in the file");
                for b in 0..n_blk {
                    let mut blocks: [&[u8]; TILE_ROWS] = [&[]; TILE_ROWS];
                    for (r, out) in blocks.iter_mut().enumerate() {
                        *out = &rows[r * rb + b * blk..r * rb + b * blk + blk];
                    }
                    let out = &mut tile[b * tile_bytes..(b + 1) * tile_bytes];
                    peregrine_kernels::kq::repack_q5_08(&blocks, out);
                }
                let out = &mut dst[gr * n_sb * Q8K_SIZE..(gr + 1) * n_sb * Q8K_SIZE];
                mk.q8k_build(Format::Q5_0, &tile, n_blk, out);
            }
        }
        _ => unreachable!("the layout holds only formats with a form"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_tier_formats_other_than_the_embedding_are_taken() {
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
        let mk = MatKernels::get();
        assert_eq!(
            eligible(&t("a", Format::Q4_K, vec![256, 8])),
            mk.q8k_uses_form(Format::Q4_K)
        );
        assert_eq!(
            eligible(&t("b", Format::Q6_K, vec![512, 16])),
            mk.q8k_uses_form(Format::Q6_K)
        );
        // Q5_0 joins the set when the tier's table says so; its rows need not be a
        // whole number of 256-weight super-blocks (Qwen 2.5's are 896 wide).
        assert_eq!(
            eligible(&t("c", Format::Q5_0, vec![256, 8])),
            mk.q8k_uses_form(Format::Q5_0)
        );
        assert_eq!(
            eligible(&t("c896", Format::Q5_0, vec![896, 8])),
            mk.q8k_uses_form(Format::Q5_0)
        );
        assert!(!eligible(&t(
            "token_embd.weight",
            Format::Q6_K,
            vec![256, 8]
        )));
        assert!(!eligible(&t("d", Format::Q8_0, vec![256, 8])));
        assert!(!eligible(&t("e", Format::Q4_K, vec![256, 4])));
        assert!(!eligible(&t("f", Format::Q4_K, vec![128, 8])));
        assert!(!eligible(&t("g", Format::F16, vec![256, 8])));
    }

    #[test]
    fn a_q5_0_row_that_stops_inside_a_super_block_gets_a_short_last_one() {
        let e = Q8Entry {
            name: "x".into(),
            kind: Format::Q5_0,
            offset: 0,
            len: 4 * Q8K_SIZE,
            rows: 8,
            cols: 896,
        };
        assert_eq!(e.super_blocks(), 4);
        let e = Q8Entry { cols: 1024, ..e };
        assert_eq!(e.super_blocks(), 4);
        let e = Q8Entry { cols: 4864, ..e };
        assert_eq!(e.super_blocks(), 19);
    }
}
