//! The repacked k-quant layout: eight output rows of a matrix interleaved into one tile.
//!
//! The repacked GEMM of S3.4 (`docs/design/s3.4.md`) multiplies eight output rows at
//! once, so the weights of eight rows have to sit side by side: the kernel loads one
//! 32-byte chunk that holds four rows' worth of codes and pairs it with four
//! activation rows. This module defines that layout for the two k-quant formats the
//! engine reads in bulk — Q4_K and Q6_K — and the two functions that move between the
//! file's layout and the repacked one.
//!
//! # The Q4_K tile
//!
//! One tile holds eight Q4_K super-blocks: one per row, at the same position in the
//! row. Its bytes are
//!
//! ```text
//! 0    d      the eight rows' f16 super-block scales      16 bytes
//! 16   dmin   the eight rows' f16 super-block min scales  16 bytes
//! 32   sc     the eight rows' 6-bit scales, unpacked      64 bytes
//! 96   m      the eight rows' 6-bit mins, unpacked        64 bytes
//! 160  qs     the eight rows' nibbles                   1024 bytes
//! ```
//!
//! The scales are *unpacked*: the file packs eight six-bit scales and eight six-bit
//! mins into twelve bytes, and the kernel would have to undo that packing in its
//! inner loop. Unpacking once, at load, costs 32 bytes per tile (2.8 % of the Q4_K
//! weights) and leaves the kernel with a byte load and a broadcast per sub-block.
//!
//! `sc` and `m` are indexed `sc[sub_block * 8 + row]`: the sub-block's scale for all
//! eight rows, then the next sub-block's. `qs` is sixteen windows of eight bytes per
//! row, side by side: `qs[window * 64 + row * 8 + k]` is row `row`'s byte `k` of
//! window `window`. A 32-byte load at `qs + window * 64` therefore holds rows 0 to 3's
//! eight bytes, and one at `qs + window * 64 + 32` holds rows 4 to 7's.
//!
//! # The Q6_K tile
//!
//! ```text
//! 0    d      the eight rows' f16 super-block scales      16 bytes
//! 16   sc     the eight rows' 8-bit scales, unpacked      128 bytes
//! 144  ql     the eight rows' low nibbles                1024 bytes
//! 1168 qh     the eight rows' high bits                  512 bytes
//! ```
//!
//! Q6_K's codes are six bits wide and stored as four bits in `ql` and two in `qh`, so
//! the tile keeps them in the file's own encoding: expanding them to bytes would cost
//! 30 % more memory than the format itself. `sc` is indexed `sc[sub * 8 + row]`, with
//! one scale per sixteen weights.
//!
//! `ql` and `qh` are *chunked*: four weights of one group at a time, the eight rows
//! side by side. `ql[chunk * 32 + row * 4 + i]` is row `row`'s low-nibble byte for the
//! `i`-th weight of the chunk, and `qh[chunk * 32 + row * 4 + i]` the same weight's
//! high bits. A 32-byte load is then the eight rows' four codes of one chunk, one row
//! per 32-bit lane — the shape a `vpmaddubsw` wants, with no shuffle in between.
//!
//! The chunks follow the format's own grouping: a super-block is two halves of 128
//! weights, each four groups of 32, each two runs of 16. The half's first 32 `ql`
//! bytes are the groups 0 and 2's (the low nibbles and the high nibbles of the same
//! bytes), the next 32 the groups 1 and 3's; the half's 32 `qh` bytes hold all four
//! groups' high bits, two bits each. So the tile holds every byte of the file's own
//! bytes exactly once: the chunked order is a permutation, not a copy.
//!
//! # Sizes
//!
//! A Q4_K tile is 1184 bytes against 8 × 144 = 1152 bytes of the file's bytes: the
//! 2.8 % the unpacked scales cost. A Q6_K tile is 1680 bytes against 8 × 210 = 1680
//! bytes: exactly the same, because Q6_K's scales are already one byte each.

use std::mem::{offset_of, size_of};

use crate::format::f16;
use crate::quant::QuantAct;

/// The bytes one Q4_K super-block takes (ggml's `block_q4_K`).
pub const Q4K_BLOCK: usize = 144;
/// The bytes one Q6_K super-block takes (ggml's `block_q6_K`).
pub const Q6K_BLOCK: usize = 210;

/// The rows one interleaved activation group holds.
pub const ACT_GROUP: usize = 4;

/// Eight Q4_K super-blocks — eight output rows — interleaved for the repacked GEMM.
///
/// The byte offsets are the ones the file's own layout implies: the scales first
/// (unpacked, one byte each), then the codes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q4K8 {
    /// The eight rows' super-block scales, as the f16 bit patterns the file holds.
    pub d: [u16; 8],
    /// The eight rows' super-block min scales, as f16 bit patterns.
    pub dmin: [u16; 8],
    /// The eight rows' six-bit sub-block scales, unpacked: `sc[sub_block * 8 + row]`.
    pub sc: [u8; 64],
    /// The eight rows' six-bit sub-block mins, unpacked: `m[sub_block * 8 + row]`.
    pub m: [u8; 64],
    /// The eight rows' nibbles: `qs[window * 64 + row * 8 + k]`.
    pub qs: [u8; 1024],
}

/// Eight Q6_K super-blocks — eight output rows — interleaved for the repacked GEMM.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q6K8 {
    /// The eight rows' super-block scales, as the f16 bit patterns the file holds.
    pub d: [u16; 8],
    /// The eight rows' eight-bit scales, one per sixteen weights: `sc[sub * 8 + row]`.
    pub sc: [i8; 128],
    /// The eight rows' low nibbles, chunk by chunk: `ql[chunk * 32 + row * 4 + i]`.
    pub ql: [u8; 1024],
    /// The eight rows' high bits, chunk by chunk: `qh[chunk * 32 + row * 4 + i]`.
    pub qh: [u8; 512],
}

impl Q4K8 {
    /// The tile's size in bytes.
    pub const SIZE: usize = size_of::<Q4K8>();
    /// The bytes the eight super-blocks it holds take in the file.
    pub const SOURCE: usize = 8 * Q4K_BLOCK;
}

impl Q6K8 {
    /// The tile's size in bytes.
    pub const SIZE: usize = size_of::<Q6K8>();
    /// The bytes the eight super-blocks it holds take in the file.
    pub const SOURCE: usize = 8 * Q6K_BLOCK;
}

// The byte offsets of the fields, taken from the layout itself: the tile's bytes and
// the struct's fields cannot drift apart.
const Q4_D: usize = offset_of!(Q4K8, d);
const Q4_DMIN: usize = offset_of!(Q4K8, dmin);
const Q4_SC: usize = offset_of!(Q4K8, sc);
const Q4_M: usize = offset_of!(Q4K8, m);
const Q4_QS: usize = offset_of!(Q4K8, qs);
const Q6_D: usize = offset_of!(Q6K8, d);
const Q6_SC: usize = offset_of!(Q6K8, sc);
const Q6_QL: usize = offset_of!(Q6K8, ql);
const Q6_QH: usize = offset_of!(Q6K8, qh);

/// The byte offset of the eight rows' super-block scales in a Q4_K tile.
pub const fn q4_d() -> usize {
    Q4_D
}
/// The byte offset of the eight rows' super-block min scales in a Q4_K tile.
pub const fn q4_dmin() -> usize {
    Q4_DMIN
}
/// The byte offset of the eight rows' six-bit sub-block scales in a Q4_K tile.
pub const fn q4_sc() -> usize {
    Q4_SC
}
/// The byte offset of the eight rows' six-bit sub-block mins in a Q4_K tile.
pub const fn q4_m() -> usize {
    Q4_M
}
/// The byte offset of the eight rows' nibbles in a Q4_K tile.
pub const fn q4_qs() -> usize {
    Q4_QS
}
/// The byte offset of the eight rows' super-block scales in a Q6_K tile.
pub const fn q6_d() -> usize {
    Q6_D
}
/// The byte offset of the eight rows' eight-bit scales in a Q6_K tile.
pub const fn q6_sc() -> usize {
    Q6_SC
}
/// The byte offset of the eight rows' low nibbles in a Q6_K tile.
pub const fn q6_ql() -> usize {
    Q6_QL
}
/// The byte offset of the eight rows' high bits in a Q6_K tile.
pub const fn q6_qh() -> usize {
    Q6_QH
}

/// The bytes one Q6_K chunk holds: four weights of one group, of eight rows.
pub const Q6_CHUNK: usize = 32;

/// The eight rows' six-bit scales and mins, unpacked from a Q4_K super-block's twelve
/// scale bytes.
///
/// This is ggml's `get_scale_min_k4`, one field at a time: the first four scales and
/// mins live in the low six bits of eight bytes, the last four are split over the
/// high two bits of the first eight bytes and the low four bits of the last four.
fn q4k_scales(q: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..4 {
        sc[j] = q[j] & 63;
        m[j] = q[j + 4] & 63;
    }
    for j in 4..8 {
        sc[j] = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
    (sc, m)
}

/// The inverse of [`q4k_scales`]: eight six-bit scales and mins back into the twelve
/// bytes a Q4_K super-block holds them in.
///
/// Every bit of the twelve bytes carries a field, so this is a bijection: the round
/// trip gives the file's bytes back exactly, whatever they were.
fn q4k_pack_scales(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
    let mut q = [0u8; 12];
    for i in 0..4 {
        // q[i]: the low six bits are sc[i], the high two the top of sc[i + 4].
        q[i] = (sc[i] & 63) | (((sc[i + 4] >> 4) & 3) << 6);
        // q[i + 4]: the low six bits are m[i], the high two the top of m[i + 4].
        q[i + 4] = (m[i] & 63) | (((m[i + 4] >> 4) & 3) << 6);
        // q[i + 8]: the low four bits are the bottom of sc[i + 4], the high four the
        // bottom of m[i + 4].
        q[i + 8] = (sc[i + 4] & 0x0F) | ((m[i + 4] & 0x0F) << 4);
    }
    q
}

/// Repack eight Q4_K super-blocks — one per row — into one tile.
///
/// The blocks are given one by one because a matrix's rows are not adjacent: the
/// eighth row of a tile is a whole row away from the first. `dst` takes
/// [`Q4K8::SIZE`] bytes.
///
/// # Panics
///
/// If `src` does not hold eight blocks of [`Q4K_BLOCK`] bytes, or `dst` is shorter
/// than a tile.
pub fn repack_q4k8(src: &[&[u8]], dst: &mut [u8]) {
    assert_eq!(src.len(), 8, "a Q4_K tile holds eight rows");
    assert!(dst.len() >= Q4K8::SIZE);
    for (r, b) in src.iter().enumerate() {
        assert!(b.len() >= Q4K_BLOCK, "a Q4_K super-block is 144 bytes");
        dst[Q4_D + r * 2..Q4_D + r * 2 + 2].copy_from_slice(&b[0..2]);
        dst[Q4_DMIN + r * 2..Q4_DMIN + r * 2 + 2].copy_from_slice(&b[2..4]);
        let (sc, m) = q4k_scales(&b[4..16]);
        for j in 0..8 {
            dst[Q4_SC + j * 8 + r] = sc[j];
            dst[Q4_M + j * 8 + r] = m[j];
        }
        for c in 0..16 {
            let at = Q4_QS + c * 64 + r * 8;
            dst[at..at + 8].copy_from_slice(&b[16 + c * 8..16 + c * 8 + 8]);
        }
    }
}

/// The inverse of [`repack_q4k8`]: one tile back into eight Q4_K super-blocks.
///
/// The blocks are written in order: row `r`'s block at `r * 144`.
///
/// # Panics
///
/// If `src` is shorter than a tile, or `dst` is shorter than
/// [`Q4K8::SOURCE`] bytes.
pub fn unrepack_q4k8(src: &[u8], dst: &mut [u8]) {
    assert!(src.len() >= Q4K8::SIZE);
    assert!(dst.len() >= Q4K8::SOURCE);
    for r in 0..8 {
        let b = &mut dst[r * Q4K_BLOCK..r * Q4K_BLOCK + Q4K_BLOCK];
        b[0..2].copy_from_slice(&src[Q4_D + r * 2..Q4_D + r * 2 + 2]);
        b[2..4].copy_from_slice(&src[Q4_DMIN + r * 2..Q4_DMIN + r * 2 + 2]);
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            sc[j] = src[Q4_SC + j * 8 + r];
            m[j] = src[Q4_M + j * 8 + r];
        }
        b[4..16].copy_from_slice(&q4k_pack_scales(&sc, &m));
        for c in 0..16 {
            let at = 16 + c * 8;
            b[at..at + 8].copy_from_slice(&src[Q4_QS + c * 64 + r * 8..Q4_QS + c * 64 + r * 8 + 8]);
        }
    }
}

/// Repack eight Q6_K super-blocks — one per row — into one tile.
///
/// # Panics
///
/// If `src` does not hold eight blocks of [`Q6K_BLOCK`] bytes, or `dst` is shorter
/// than a tile.
pub fn repack_q6k8(src: &[&[u8]], dst: &mut [u8]) {
    assert_eq!(src.len(), 8, "a Q6_K tile holds eight rows");
    assert!(dst.len() >= Q6K8::SIZE);
    for (r, b) in src.iter().enumerate() {
        assert!(b.len() >= Q6K_BLOCK, "a Q6_K super-block is 210 bytes");
        dst[Q6_D + r * 2..Q6_D + r * 2 + 2].copy_from_slice(&b[208..210]);
        for s in 0..16 {
            dst[Q6_SC + s * 8 + r] = b[192 + s];
        }
        // The codes, chunk by chunk: four weights of one group, the eight rows
        // side by side. The half's first 32 low-nibble bytes are the groups 0
        // and 2's, the next 32 the groups 1 and 3's; the half's 32 high-bit
        // bytes hold all four groups' two bits.
        for c in 0..2 {
            for cc in 0..8 {
                let lo = Q6_QL + (c * 16 + cc) * Q6_CHUNK + r * 4;
                dst[lo..lo + 4].copy_from_slice(&b[c * 64 + cc * 4..c * 64 + cc * 4 + 4]);
                let hi = Q6_QL + (c * 16 + 8 + cc) * Q6_CHUNK + r * 4;
                dst[hi..hi + 4].copy_from_slice(&b[c * 64 + 32 + cc * 4..c * 64 + 32 + cc * 4 + 4]);
                let qh = Q6_QH + (c * 8 + cc) * Q6_CHUNK + r * 4;
                dst[qh..qh + 4]
                    .copy_from_slice(&b[128 + c * 32 + cc * 4..128 + c * 32 + cc * 4 + 4]);
            }
        }
    }
}

/// The inverse of [`repack_q6k8`]: one tile back into eight Q6_K super-blocks.
///
/// The blocks are written in order: row `r`'s block at `r * 210`.
///
/// # Panics
///
/// If `src` is shorter than a tile, or `dst` is shorter than
/// [`Q6K8::SOURCE`] bytes.
pub fn unrepack_q6k8(src: &[u8], dst: &mut [u8]) {
    assert!(src.len() >= Q6K8::SIZE);
    assert!(dst.len() >= Q6K8::SOURCE);
    for r in 0..8 {
        let b = &mut dst[r * Q6K_BLOCK..r * Q6K_BLOCK + Q6K_BLOCK];
        // The scales sit one sub-block apart: the tile holds them for all eight
        // rows side by side, the file holds one row's sixteen in a row.
        for s in 0..16 {
            b[192 + s] = src[Q6_SC + s * 8 + r];
        }
        b[208..210].copy_from_slice(&src[Q6_D + r * 2..Q6_D + r * 2 + 2]);
        for c in 0..2 {
            for cc in 0..8 {
                let lo = Q6_QL + (c * 16 + cc) * Q6_CHUNK + r * 4;
                b[c * 64 + cc * 4..c * 64 + cc * 4 + 4].copy_from_slice(&src[lo..lo + 4]);
                let hi = Q6_QL + (c * 16 + 8 + cc) * Q6_CHUNK + r * 4;
                b[c * 64 + 32 + cc * 4..c * 64 + 32 + cc * 4 + 4].copy_from_slice(&src[hi..hi + 4]);
                let qh = Q6_QH + (c * 8 + cc) * Q6_CHUNK + r * 4;
                b[128 + c * 32 + cc * 4..128 + c * 32 + cc * 4 + 4]
                    .copy_from_slice(&src[qh..qh + 4]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The interleaved activation
// ---------------------------------------------------------------------------

/// The k-quant interleaved activation: four rows side by side.
///
/// The repacked GEMM multiplies eight output rows at once and reads four
/// activation rows at a time, so the four rows' codes have to sit side by side:
/// one 32-byte group holds one window of eight codes of each of the four rows.
/// This is llama.cpp's `block_q8_Kx4` layout, and it is what the kernel's loads
/// expect — the four rows of a window share a cache line, which the plain
/// per-row layout does not.
///
/// The buffer is filled once per layer, from the activation that was quantised
/// for that layer's matmul group, and read by every output row of the group: the
/// quantisation is not repeated per matrix.
///
/// # The layout
///
/// For a group of four rows and one super-block of 256 weights:
///
/// ```text
/// q[(sb · 32 + j) · 32 + t · 8 + k]   row t's code for the weight sb·256 + 8j + k
/// ```
///
/// so the 32 groups of a super-block each hold eight codes of the four rows,
/// side by side, and one 32-byte load is one window of all four rows.
///
/// The scales and the run sums are kept per row rather than interleaved: the
/// kernel reads one row's scale and one row's run sums at a time, and the
/// interleaving would only spread them out.
#[derive(Default, Clone)]
pub struct KqActs {
    /// The interleaved codes: `groups · super_blocks · 1024` bytes.
    pub q: Vec<i8>,
    /// The per-256 scale of every row: `groups · super_blocks · 4`.
    pub d: Vec<f32>,
    /// The per-16-weight run sum of every row: `groups · super_blocks · 4 · 16`.
    ///
    /// Two formats need the activation's sums, at two granularities: the Q4_K
    /// min term is the row's mins against the sum over a 32-weight sub-block
    /// (two runs), and the Q6_K offset is the row's scales against the sum
    /// over a 16-weight run. Both are exact integers — a run's sum is at most
    /// 2 032 and a sub-block's at most 4 064, both inside the float's exact
    /// integer range — so the sums are kept per run, as floats, and the
    /// sub-block's sum is the sum of two of them.
    pub bs: Vec<f32>,
    /// The rows filled.
    pub rows: usize,
    /// The activation width.
    pub n: usize,
}

impl KqActs {
    /// The super-blocks one activation holds.
    pub fn super_blocks(&self) -> usize {
        self.n / 256
    }

    /// The groups of four rows the buffer holds.
    pub fn groups(&self) -> usize {
        self.rows.div_ceil(ACT_GROUP)
    }

    /// The interleaved activation of `acts`, four rows at a time.
    ///
    /// A batch whose width is not a multiple of four is padded with zero codes:
    /// a zero code contributes nothing to any dot product, and the padded rows'
    /// results are discarded.
    ///
    /// # Panics
    ///
    /// If the activations are not a whole number of super-blocks, or do not carry
    /// the per-256 layout a k-quant matmul needs.
    pub fn fill(&mut self, acts: &[QuantAct]) {
        assert!(!acts.is_empty(), "no activation to interleave");
        let n = acts[0].n;
        assert!(
            n % 256 == 0,
            "a k-quant activation is a whole number of super-blocks"
        );
        assert!(
            acts.iter().all(|a| a.n == n && !a.qk.is_empty()),
            "the activations are not in the per-256 layout"
        );
        let n_sb = n / 256;
        let groups = acts.len().div_ceil(ACT_GROUP);
        self.rows = acts.len();
        self.n = n;
        self.q.clear();
        self.q.resize(groups * n_sb * 1024, 0);
        self.d.clear();
        self.d.resize(groups * n_sb * ACT_GROUP, 0.0);
        self.bs.clear();
        self.bs.resize(groups * n_sb * ACT_GROUP * 16, 0.0);
        for g in 0..groups {
            for sb in 0..n_sb {
                let base = (g * n_sb + sb) * 1024;
                for j in 0..32 {
                    let dst = base + j * 32;
                    for t in 0..ACT_GROUP {
                        let b = g * ACT_GROUP + t;
                        if b >= acts.len() {
                            continue;
                        }
                        let src = &acts[b].qk[sb * 256 + j * 8..sb * 256 + j * 8 + 8];
                        self.q[dst + t * 8..dst + t * 8 + 8].copy_from_slice(src);
                    }
                }
                let dbase = (g * n_sb + sb) * ACT_GROUP;
                let bbase = (g * n_sb + sb) * ACT_GROUP * 16;
                for t in 0..ACT_GROUP {
                    let b = g * ACT_GROUP + t;
                    if b >= acts.len() {
                        continue;
                    }
                    self.d[dbase + t] = acts[b].dk[sb];
                    // The per-16 run sums: the Q6_K offset is applied against
                    // them, and the Q4_K min term against two of them.
                    let rs = &acts[b].bsums[sb * 16..sb * 16 + 16];
                    for j in 0..16 {
                        self.bs[bbase + t * 16 + j] = rs[j] as f32;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The scalar reference for the repacked layout
// ---------------------------------------------------------------------------

/// The value of one row of a repacked Q4_K tile against one activation row.
///
/// This is the specification the AVX2 kernel is checked against, and it is the
/// same real number the ordinary Q4_K kernel computes:
///
/// ```text
/// value = Σ_sb dk[sb] · ( d[sb] · S − dmin[sb] · M )
/// ```
///
/// with `S` the integer sum of the scaled products of the super-block and `M`
/// the integer sum of the scaled activation sums (the min term). Both are exact
/// integers, so the only difference from the ordinary kernel is where the eight
/// lanes are reduced: the ordinary kernel reduces them in float at the very end,
/// the repacked one reduces them in int32 inside the super-block.
///
/// # Panics
///
/// If the tile is shorter than `n_sb` tiles, or the activation does not carry
/// the per-256 layout.
pub fn dot_q4k8_ref(tile: &[u8], n_sb: usize, row: usize, act: &QuantAct) -> f32 {
    assert!(row < 8, "a tile holds eight rows");
    assert!(
        tile.len() >= n_sb * Q4K8::SIZE,
        "the tile is shorter than its super-blocks"
    );
    assert!(
        !act.qk.is_empty(),
        "a k-quant row needs the per-256 activation"
    );
    let mut acc = 0.0f32;
    for sb in 0..n_sb {
        let t = &tile[sb * Q4K8::SIZE..(sb + 1) * Q4K8::SIZE];
        let d = f16(u16::from_le_bytes([
            t[Q4_D + row * 2],
            t[Q4_D + row * 2 + 1],
        ]));
        let dmin = f16(u16::from_le_bytes([
            t[Q4_DMIN + row * 2],
            t[Q4_DMIN + row * 2 + 1],
        ]));
        let aq = &act.qk[sb * 256..sb * 256 + 256];
        let rs = &act.bsums[sb * 16..sb * 16 + 16];
        let mut sumi = 0i64;
        let mut summ = 0i64;
        for j in 0..8 {
            let sc = t[Q4_SC + j * 8 + row] as i64;
            let mn = t[Q4_M + j * 8 + row] as i64;
            // The sub-block's 32 weights: the low nibbles of the row's bytes
            // 32·(j/2) .. +32 when j is even, the high nibbles when it is odd.
            let qbase = (j / 2) * 32;
            let mut s = 0i64;
            for l in 0..32 {
                let byte = t[Q4_QS + (qbase + l) / 8 * 64 + row * 8 + (qbase + l) % 8];
                let code = if j % 2 == 0 {
                    (byte & 0x0F) as i64
                } else {
                    (byte >> 4) as i64
                };
                s += code * aq[j * 32 + l] as i64;
            }
            sumi += sc * s;
            summ += mn * (rs[2 * j] as i64 + rs[2 * j + 1] as i64);
        }
        // The same expression the kernel folds: the products rounded first, the
        // difference and the accumulation in one rounding each.
        let x = (-dmin).mul_add(summ as f32, d * sumi as f32);
        acc = x.mul_add(act.dk[sb], acc);
    }
    acc
}

/// The value of one row of a repacked Q6_K tile against one activation row.
///
/// This is the specification the AVX2 kernel is checked against, and it is the
/// same real number the ordinary Q6_K kernel computes:
///
/// ```text
/// value = Σ_sb dk[sb] · ( d[sb] · S − 32 · Σ_runs scale[run] · bsum[run] )
/// ```
///
/// with `S` the integer sum of the scaled products of the super-block. The
/// `−32` is the format's offset: the codes are `0..63` and the weights they
/// stand for are `−32..31`, so the offset is taken off the activation's run
/// sums, exactly as the ordinary kernel does it — in the integer domain, where
/// it is exact.
///
/// # Panics
///
/// If the tile is shorter than `n_sb` tiles, or the activation does not carry
/// the per-256 layout.
pub fn dot_q6k8_ref(tile: &[u8], n_sb: usize, row: usize, act: &QuantAct) -> f32 {
    assert!(row < 8, "a tile holds eight rows");
    assert!(
        tile.len() >= n_sb * Q6K8::SIZE,
        "the tile is shorter than its super-blocks"
    );
    assert!(
        !act.qk.is_empty(),
        "a k-quant row needs the per-256 activation"
    );
    let mut acc = 0.0f32;
    for sb in 0..n_sb {
        let t = &tile[sb * Q6K8::SIZE..(sb + 1) * Q6K8::SIZE];
        let d = f16(u16::from_le_bytes([
            t[Q6_D + row * 2],
            t[Q6_D + row * 2 + 1],
        ]));
        let aq = &act.qk[sb * 256..sb * 256 + 256];
        let rs = &act.bsums[sb * 16..sb * 16 + 16];
        let mut sumi = 0i64;
        let mut off = 0i64;
        // Two halves of 128 weights, four groups of 32, two runs of 16.
        for c in 0..2 {
            for g in 0..4 {
                for run in 0..2 {
                    let sub = c * 8 + g * 2 + run;
                    let scale = t[Q6_SC + sub * 8 + row] as i8 as i64;
                    // The run's sixteen weights.
                    let mut s = 0i64;
                    for l in 0..16 {
                        let w = c * 128 + g * 32 + run * 16 + l;
                        // The chunk of four weights the weight is in, and the
                        // byte inside it.
                        let cc = (run * 16 + l) / 4;
                        let i = (run * 16 + l) % 4;
                        let ql = t[Q6_QL + (c * 16 + (g & 1) * 8 + cc) * Q6_CHUNK + row * 4 + i];
                        let qh = t[Q6_QH + (c * 8 + cc) * Q6_CHUNK + row * 4 + i];
                        // The groups 0 and 1 take the low nibble of the byte,
                        // the groups 2 and 3 the high one; the high two bits
                        // are the group's own two of the byte's four fields.
                        let low = if g < 2 { ql & 0x0F } else { ql >> 4 };
                        let high = (qh >> (2 * g)) & 3;
                        let code = (low | (high << 4)) as i64;
                        s += code * aq[w] as i64;
                    }
                    sumi += scale * s;
                    off += 32 * scale * rs[sub] as i64;
                }
            }
        }
        // The same expression the kernel folds: the integer sums first, the
        // scale and the accumulation in one rounding each.
        let v = (sumi - off) as f32;
        acc = v.mul_add(d * act.dk[sb], acc);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A seeded generator, so a failure is reproducible.
    struct Rng(u64);

    impl Rng {
        fn next_u8(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 24) as u8
        }

        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next_u8()).collect()
        }
    }

    /// Eight blocks of the given size, as the tile's rows.
    fn group(src: &[u8]) -> Vec<&[u8]> {
        src.chunks_exact(144).take(8).collect()
    }

    fn group6(src: &[u8]) -> Vec<&[u8]> {
        src.chunks_exact(210).take(8).collect()
    }

    #[test]
    fn the_tiles_have_the_sizes_the_layout_says() {
        assert_eq!(Q4K8::SIZE, 1184);
        assert_eq!(Q4K8::SOURCE, 1152);
        assert_eq!(Q6K8::SIZE, 1680);
        assert_eq!(Q6K8::SOURCE, 1680);
    }

    #[test]
    fn a_q4_k_tile_gives_the_blocks_back() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for case in 0..8 {
            // The edge cases: all zeros, all ones, and the scale bytes at their
            // extremes, where the six-bit fields are 0 and 63.
            let src: Vec<u8> = match case {
                0 => vec![0u8; 8 * Q4K_BLOCK],
                1 => vec![0xffu8; 8 * Q4K_BLOCK],
                2 => {
                    let mut v = vec![0u8; 8 * Q4K_BLOCK];
                    for b in v.iter_mut().skip(4).step_by(12) {
                        *b = 0x3f;
                    }
                    v
                }
                3 => {
                    let mut v = vec![0u8; 8 * Q4K_BLOCK];
                    for b in v.iter_mut().skip(4).step_by(12) {
                        *b = 0xc0;
                    }
                    v
                }
                _ => rng.bytes(8 * Q4K_BLOCK),
            };
            let mut tile = vec![0u8; Q4K8::SIZE];
            repack_q4k8(&group(&src), &mut tile);
            let mut back = vec![0u8; Q4K8::SOURCE];
            unrepack_q4k8(&tile, &mut back);
            assert_eq!(&back[..], &src[..], "case {case}");
        }
    }

    #[test]
    fn a_q6_k_tile_gives_the_blocks_back() {
        let mut rng = Rng(0x0fed_cba9_8765_4321);
        for case in 0..4 {
            let src: Vec<u8> = match case {
                0 => vec![0u8; 8 * Q6K_BLOCK],
                1 => vec![0xffu8; 8 * Q6K_BLOCK],
                2 => {
                    // The scales at both ends of their range.
                    let mut v = vec![0u8; 8 * Q6K_BLOCK];
                    for (i, b) in v[192..208].iter_mut().enumerate() {
                        *b = if i % 2 == 0 { 0x80 } else { 0x7f };
                    }
                    v
                }
                _ => rng.bytes(8 * Q6K_BLOCK),
            };
            let mut tile = vec![0u8; Q6K8::SIZE];
            repack_q6k8(&group6(&src), &mut tile);
            let mut back = vec![0u8; Q6K8::SOURCE];
            unrepack_q6k8(&tile, &mut back);
            assert_eq!(&back[..], &src[..], "case {case}");
        }
    }

    #[test]
    fn the_unpacked_scales_are_the_ones_ggml_reads() {
        // The file packs eight six-bit scales and eight six-bit mins into twelve
        // bytes; the tile holds them one byte each. The packing has to be ggml's,
        // or the kernel would multiply by the wrong scale.
        let q: Vec<u8> = (0..12).map(|i| (i * 17 + 3) as u8).collect();
        let (sc, m) = q4k_scales(&q);
        for j in 0..4 {
            assert_eq!(sc[j], q[j] & 63);
            assert_eq!(m[j], q[j + 4] & 63);
        }
        for j in 4..8 {
            assert_eq!(sc[j], (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            assert_eq!(m[j], (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
        // And the packing is the inverse.
        assert_eq!(q4k_pack_scales(&sc, &m).to_vec(), q);
    }

    #[test]
    fn the_tile_holds_each_row_where_the_layout_says() {
        let mut rng = Rng(0xfeed_face_dead_beef);
        let src = rng.bytes(8 * Q4K_BLOCK);
        let mut tile = vec![0u8; Q4K8::SIZE];
        repack_q4k8(&group(&src), &mut tile);
        for r in 0..8 {
            // The scales.
            assert_eq!(
                &tile[Q4_D + r * 2..Q4_D + r * 2 + 2],
                &src[r * 144..r * 144 + 2]
            );
            assert_eq!(
                &tile[Q4_DMIN + r * 2..Q4_DMIN + r * 2 + 2],
                &src[r * 144 + 2..r * 144 + 4]
            );
            // The codes: window by window, the row's own bytes.
            for c in 0..16 {
                let at = Q4_QS + c * 64 + r * 8;
                assert_eq!(
                    &tile[at..at + 8],
                    &src[r * 144 + 16 + c * 8..r * 144 + 16 + c * 8 + 8]
                );
            }
        }
    }

    /// The Q6_K tile holds the codes chunk by chunk: four weights of one group
    /// at a time, the eight rows side by side.
    #[test]
    fn the_q6_k_tile_holds_its_chunks_where_the_layout_says() {
        let mut rng = Rng(0x0BAD_C0DE_F00D_BEEF);
        let src = rng.bytes(8 * Q6K_BLOCK);
        let mut tile = vec![0u8; Q6K8::SIZE];
        repack_q6k8(&group6(&src), &mut tile);
        for r in 0..8 {
            let b = &src[r * Q6K_BLOCK..(r + 1) * Q6K_BLOCK];
            // The scale and the super-block scale.
            assert_eq!(&tile[Q6_D + r * 2..Q6_D + r * 2 + 2], &b[208..210]);
            for s in 0..16 {
                assert_eq!(tile[Q6_SC + s * 8 + r], b[192 + s]);
            }
            // The codes: the half's first thirty-two low-nibble bytes are the
            // groups 0 and 2's, the next thirty-two the groups 1 and 3's; the
            // half's thirty-two high-bit bytes hold all four groups' two bits.
            for c in 0..2 {
                for cc in 0..8 {
                    let lo = Q6_QL + (c * 16 + cc) * Q6_CHUNK + r * 4;
                    assert_eq!(
                        &tile[lo..lo + 4],
                        &b[c * 64 + cc * 4..c * 64 + cc * 4 + 4],
                        "half {c} chunk {cc}"
                    );
                    let hi = Q6_QL + (c * 16 + 8 + cc) * Q6_CHUNK + r * 4;
                    assert_eq!(
                        &tile[hi..hi + 4],
                        &b[c * 64 + 32 + cc * 4..c * 64 + 32 + cc * 4 + 4],
                        "half {c} chunk {cc}"
                    );
                    let qh = Q6_QH + (c * 8 + cc) * Q6_CHUNK + r * 4;
                    assert_eq!(
                        &tile[qh..qh + 4],
                        &b[128 + c * 32 + cc * 4..128 + c * 32 + cc * 4 + 4],
                        "half {c} chunk {cc}"
                    );
                }
            }
        }
    }
}
