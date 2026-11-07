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
//! 32   scm    the eight rows' 6-bit scales and mins       96 bytes
//! 128  qs     the eight rows' nibbles                   1024 bytes
//! ```
//!
//! The scales and mins are *packed*: `scm[sub_block * 12 ..][..6]` holds the eight
//! rows' six-bit scales for that sub-block and `.. + 6` their mins. The tile is then
//! exactly the 8 × 144 = 1152 bytes of the file's own blocks. An unpacked tile would
//! be 32 bytes larger, and decode streams every byte of every tile once per token —
//! the 2.8 % would come straight off the decode rate, which the design note's
//! decision 9 does not allow. The kernel undoes the packing once per super-block,
//! before its window loop (`kqsimd::unpack_scales`), so the inner loop still reads a
//! ready scale and min vector per sub-block.
//!
//! `qs` is sixteen windows of eight bytes per row, side by side:
//! `qs[window * 64 + row * 8 + k]` is row `row`'s byte `k` of window `window`. A
//! 32-byte load at `qs + window * 64` therefore holds rows 0 to 3's eight bytes, and
//! one at `qs + window * 64 + 32` holds rows 4 to 7's.
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
//! one scale per sixteen weights, and `ql` and `qh` are windowed like `qs` above.
//!
//! # Sizes
//!
//! A Q4_K tile is 1152 bytes against 8 × 144 = 1152 bytes of the file's bytes, and a
//! Q6_K tile is 1680 against 8 × 210 = 1680: both are exactly the size of the blocks
//! they hold, so the repacked store is size-neutral and a decode step streams the
//! same bytes either way.

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
/// The byte offsets are the ones the file's own layout implies: the scales first,
/// then the codes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q4K8 {
    /// The eight rows' super-block scales, as the f16 bit patterns the file holds.
    pub d: [u16; 8],
    /// The eight rows' super-block min scales, as f16 bit patterns.
    pub dmin: [u16; 8],
    /// The eight rows' six-bit scales and mins, packed as the file packs them:
    /// `scm[sub_block * 12 ..][..6]` holds the eight rows' scales, `.. + 6` their
    /// mins, eight six-bit fields each.
    pub scm: [u8; 96],
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
    /// The eight rows' low nibbles: `ql[window * 64 + row * 8 + k]`.
    pub ql: [u8; 1024],
    /// The eight rows' high bits: `qh[window * 64 + row * 8 + k]`.
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
const Q4_SCM: usize = offset_of!(Q4K8, scm);
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
/// The byte offset of the eight rows' six-bit sub-block scales and mins in a
/// Q4_K tile.
pub const fn q4_scm() -> usize {
    Q4_SCM
}
/// The bytes one sub-block's packed scales and mins take: six for the eight
/// rows' scales, six for their mins.
pub const Q4_SCM_STRIDE: usize = 12;
/// The byte offset of the eight rows' nibbles in a Q4_K tile.
pub const fn q4_qs() -> usize {
    Q4_QS
}

/// The six-bit scale of row `row` for sub-block `sub` of a repacked Q4_K tile.
pub fn tile_q4k_scale(t: &[u8], sub: usize, row: usize) -> u8 {
    let at = Q4_SCM + sub * Q4_SCM_STRIDE;
    let v = u64::from_le_bytes(t[at..at + 8].try_into().unwrap());
    ((v >> (6 * row)) & 63) as u8
}

/// The six-bit min of row `row` for sub-block `sub` of a repacked Q4_K tile.
pub fn tile_q4k_min(t: &[u8], sub: usize, row: usize) -> u8 {
    let at = Q4_SCM + sub * Q4_SCM_STRIDE + 6;
    let v = u64::from_le_bytes(t[at..at + 8].try_into().unwrap());
    ((v >> (6 * row)) & 63) as u8
}

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
    // The scales and mins are packed one field per row per sub-block: the tile
    // is then exactly the size of the eight blocks it holds, so a decode step
    // streams the same bytes as the file's own layout does.
    let mut sc = [0u64; 8];
    let mut m = [0u64; 8];
    for (r, b) in src.iter().enumerate() {
        assert!(b.len() >= Q4K_BLOCK, "a Q4_K super-block is 144 bytes");
        dst[Q4_D + r * 2..Q4_D + r * 2 + 2].copy_from_slice(&b[0..2]);
        dst[Q4_DMIN + r * 2..Q4_DMIN + r * 2 + 2].copy_from_slice(&b[2..4]);
        let (s, mn) = q4k_scales(&b[4..16]);
        for j in 0..8 {
            sc[j] |= (s[j] as u64) << (6 * r);
            m[j] |= (mn[j] as u64) << (6 * r);
        }
        for c in 0..16 {
            let at = Q4_QS + c * 64 + r * 8;
            dst[at..at + 8].copy_from_slice(&b[16 + c * 8..16 + c * 8 + 8]);
        }
    }
    for j in 0..8 {
        let at = Q4_SCM + j * Q4_SCM_STRIDE;
        dst[at..at + 6].copy_from_slice(&sc[j].to_le_bytes()[..6]);
        dst[at + 6..at + 12].copy_from_slice(&m[j].to_le_bytes()[..6]);
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
            sc[j] = tile_q4k_scale(src, j, r);
            m[j] = tile_q4k_min(src, j, r);
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
        for c in 0..16 {
            let at = Q6_QL + c * 64 + r * 8;
            dst[at..at + 8].copy_from_slice(&b[c * 8..c * 8 + 8]);
        }
        for c in 0..8 {
            let at = Q6_QH + c * 64 + r * 8;
            dst[at..at + 8].copy_from_slice(&b[128 + c * 8..128 + c * 8 + 8]);
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
        for c in 0..16 {
            let at = Q6_QL + c * 64 + r * 8;
            b[c * 8..c * 8 + 8].copy_from_slice(&src[at..at + 8]);
        }
        for c in 0..8 {
            let at = Q6_QH + c * 64 + r * 8;
            b[128 + c * 8..128 + c * 8 + 8].copy_from_slice(&src[at..at + 8]);
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
    /// The per-32-weight sum of every row: `groups · super_blocks · 4 · 8`.
    ///
    /// The min term of the Q4_K format is the row's mins against the
    /// activation's sums, one per sub-block — and every product and partial sum
    /// of it is an integer below 2^24 (63 · 4064 · 8 = 2 048 256), so it is
    /// exact in either domain and the kernel folds it in int32 lanes.
    pub bs: Vec<i32>,
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
        self.bs.resize(groups * n_sb * ACT_GROUP * 8, 0);
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
                let bbase = (g * n_sb + sb) * ACT_GROUP * 8;
                for t in 0..ACT_GROUP {
                    let b = g * ACT_GROUP + t;
                    if b >= acts.len() {
                        continue;
                    }
                    self.d[dbase + t] = acts[b].dk[sb];
                    // The per-32-weight sums, from the per-16 run sums.
                    let rs = &acts[b].bsums[sb * 16..sb * 16 + 16];
                    for j in 0..8 {
                        self.bs[bbase + t * 8 + j] = rs[2 * j] as i32 + rs[2 * j + 1] as i32;
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
            let sc = tile_q4k_scale(t, j, row) as i64;
            let mn = tile_q4k_min(t, j, row) as i64;
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
        assert_eq!(Q4K8::SIZE, 1152);
        assert_eq!(Q4K8::SOURCE, 1152);
        assert_eq!(Q6K8::SIZE, 1680);
        assert_eq!(Q6K8::SOURCE, 1680);
        // Size-neutral: the store replaces the file's bytes, it does not add to
        // them. Decode streams every byte of the store once per token.
        assert_eq!(Q4K8::SIZE, Q4K8::SOURCE);
        assert_eq!(Q6K8::SIZE, Q6K8::SOURCE);
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
            // The six-bit scales and mins, at the row's six bits of each
            // sub-block's packed fields.
            let (sc, m) = q4k_scales(&src[r * 144 + 4..r * 144 + 16]);
            for j in 0..8 {
                assert_eq!(tile_q4k_scale(&tile, j, r), sc[j], "scale {j} of row {r}");
                assert_eq!(tile_q4k_min(&tile, j, r), m[j], "min {j} of row {r}");
            }
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
}
