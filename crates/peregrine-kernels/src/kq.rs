//! The repacked k-quant layout: eight output rows of a matrix interleaved into one tile.
//!
//! The repacked GEMM of S3.4 (`docs/design/s3.4.md`) multiplies eight output rows at
//! once, so the weights of eight rows have to sit side by side: the kernel loads one
//! 32-byte chunk that holds four rows' worth of codes and pairs it with four
//! activation rows. This module defines that layout for the k-quant formats the engine
//! reads in bulk — Q4_K, Q5_K and Q6_K — and the two functions that move between the
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
//! # The Q5_K tile
//!
//! ```text
//! 0    d      the eight rows' f16 super-block scales      16 bytes
//! 16   dmin   the eight rows' f16 super-block min scales  16 bytes
//! 32   scm    the eight rows' 6-bit scales and mins       96 bytes
//! 128  qh     the eight rows' fifth bits                 256 bytes
//! 384  qs     the eight rows' nibbles                   1024 bytes
//! ```
//!
//! Q5_K is Q4_K plus a fifth bit per weight. `d`, `dmin`, `scm` and `qs` are the
//! Q4_K tile's, at the same offsets; `qh` holds the fifth bits grouped by
//! sub-block and eight-weight window, one byte per row:
//! `qh[(sub * 4 + window) * 8 + row]`, bit `k` is the fifth bit of weight
//! `sub * 32 + window * 8 + k`. The file stores the bits the other way round —
//! bit `sub` of byte `l` — so the repack transposes them once, at load, and the
//! kernel's spread is one shuffle and one bit test per four weights. The tile is
//! 1408 bytes, exactly 8 × 176, so it is size-neutral like the others.
//!
//! # The Q5_0 tile
//!
//! ```text
//! 0    d      the eight rows' f16 block scales       16 bytes
//! 16   qh     the eight rows' fifth bits             32 bytes
//! 48   qs     the eight rows' nibbles               128 bytes
//! ```
//!
//! A Q5_0 block is only 32 weights, so the tile is built per block rather than per
//! super-block. The block's 32 weights are four logical eight-weight groups:
//! `h = 0` and `1` are the low nibbles of the row's `qs` bytes `0..8` and `8..16`
//! (weights `0..16`), and `h = 2` and `3` are the high nibbles of the same bytes
//! (weights `16..32`). The tile stores each of the sixteen `qs` bytes once —
//! `qs[(h % 2) * 64 + row * 8 + k]` — so two logical groups share a physical
//! window, and the kernel loads it once and reads it as low and then high
//! nibbles. `qh[h * 8 + row]` holds row `row`'s eight fifth bits for the logical
//! group `h` (bit `k` is weight `8h + k`).
//!
//! The fifth bits are stored *by weight group* — one byte per row per group —
//! rather than in the file's per-row 32-bit word. That is what lets the kernel
//! spread eight rows' bits with two `vpshufb`s per group instead of one full
//! 32-bit expansion per row: the byte a row needs is already the byte the shuffle
//! selects.
//!
//! # The Q8_0 tile
//!
//! ```text
//! 0    d      the eight rows' f16 block scales       16 bytes
//! 16   qs     the eight rows' signed codes           256 bytes
//! ```
//!
//! Q8_0's block is 32 weights, like Q5_0's, so the tile is built per block. The
//! codes are stored window by window: `qs[window * 64 + row * 8 + k]` is row
//! `row`'s code for weight `8·window + k`, one eight-byte group per row. A
//! 32-byte load at `qs + window * 64` therefore holds the four low rows' eight
//! codes each, and one at `+ 32` the four high rows' — the same shape the Q5_0
//! kernel reads, without the nibble unpacking or the fifth-bit spread.
//!
//! The interesting consequence is streaming, not arithmetic: the file's layout
//! puts the eight rows of a tile a whole row apart, so the eight-row row-major
//! GEMV reads eight separate streams. The tile puts them side by side, and a
//! decode step streams one contiguous run instead (the same fix as Q5_0's).
//!
//! # Sizes
//!
//! A Q4_K tile is 1152 bytes against 8 × 144 = 1152 bytes of the file's bytes, a
//! Q5_K tile is 1408 against 8 × 176 = 1408, a Q6_K tile is 1680 against
//! 8 × 210 = 1680, a Q5_0 tile is 176 against 8 × 22 = 176, and a Q8_0 tile is 272
//! against 8 × 34 = 272: every one is exactly the size of the blocks it holds, so the
//! repacked store is size-neutral and a decode step streams the same bytes either way.

use std::mem::{offset_of, size_of};

use crate::format::f16;
use crate::quant::QuantAct;

/// The bytes one Q4_K super-block takes (ggml's `block_q4_K`).
pub const Q4K_BLOCK: usize = 144;
/// The bytes one Q5_K super-block takes (ggml's `block_q5_K`).
pub const Q5K_BLOCK: usize = 176;
/// The bytes one Q6_K super-block takes (ggml's `block_q6_K`).
pub const Q6K_BLOCK: usize = 210;
/// The bytes one Q5_0 block takes (ggml's `block_q5_0`).
pub const Q5_0_BLOCK: usize = 22;
/// The weights one Q5_0 block holds.
pub const Q5_0_WEIGHTS: usize = 32;
/// The eight-weight groups one Q5_0 block splits into.
pub const Q5_0_WINDOWS: usize = 4;
/// The bytes one Q8_0 block takes (ggml's `block_q8_0`).
pub const Q8_0_BLOCK: usize = 34;
/// The weights one Q8_0 block holds.
pub const Q8_0_WEIGHTS: usize = 32;
/// The eight-weight windows one Q8_0 block splits into.
pub const Q8_0_WINDOWS: usize = 4;

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

/// Eight Q5_K super-blocks — eight output rows — interleaved for the repacked GEMM.
///
/// The scales, mins and nibbles are the Q4_K tile's; `qh` adds the fifth bits,
/// grouped by sub-block and eight-weight window (`qh[(sub * 4 + window) * 8 + row]`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q5K8 {
    /// The eight rows' super-block scales, as the f16 bit patterns the file holds.
    pub d: [u16; 8],
    /// The eight rows' super-block min scales, as f16 bit patterns.
    pub dmin: [u16; 8],
    /// The eight rows' six-bit scales and mins, packed as the file packs them:
    /// `scm[sub_block * 12 ..][..6]` holds the eight rows' scales, `.. + 6` their
    /// mins, eight six-bit fields each.
    pub scm: [u8; 96],
    /// The eight rows' fifth bits: `qh[(sub * 4 + window) * 8 + row]`, bit `k` is
    /// weight `sub * 32 + window * 8 + k`.
    pub qh: [u8; 256],
    /// The eight rows' nibbles: `qs[window * 64 + row * 8 + k]`.
    pub qs: [u8; 1024],
}

/// Eight Q5_0 blocks — eight output rows — interleaved for the repacked GEMM.
///
/// Q5_0's block is 32 weights, so the tile holds one block per row rather than a
/// whole 256-weight super-block. The fifth bits are grouped by eight-weight
/// window (`qh[h * 8 + row]`), which is what makes the kernel's spread cheap.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q5_08 {
    /// The eight rows' block scales, as the f16 bit patterns the file holds.
    pub d: [u16; 8],
    /// The eight rows' fifth bits, by eight-weight group:
    /// `qh[group * 8 + row]`, bit `k` is weight `8·group + k`.
    pub qh: [u8; 32],
    /// The eight rows' nibbles, two physical windows of eight bytes per row:
    /// `qs[window * 64 + row * 8 + k]`. Logical groups `window` and `window + 2`
    /// are the low and high nibbles of the same bytes.
    pub qs: [u8; 128],
}

/// Eight Q8_0 blocks — eight output rows — interleaved for the repacked GEMM.
///
/// Q8_0's block is 32 weights, so the tile holds one block per row. The codes are
/// signed bytes, one eight-weight group per row per window.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q8_08 {
    /// The eight rows' block scales, as the f16 bit patterns the file holds.
    pub d: [u16; 8],
    /// The eight rows' signed codes: `qs[window * 64 + row * 8 + k]`.
    pub qs: [i8; 256],
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

impl Q5K8 {
    /// The tile's size in bytes.
    pub const SIZE: usize = size_of::<Q5K8>();
    /// The bytes the eight super-blocks it holds take in the file.
    pub const SOURCE: usize = 8 * Q5K_BLOCK;
}

impl Q5_08 {
    /// The tile's size in bytes.
    pub const SIZE: usize = size_of::<Q5_08>();
    /// The bytes the eight blocks it holds take in the file.
    pub const SOURCE: usize = 8 * Q5_0_BLOCK;
}

impl Q8_08 {
    /// The tile's size in bytes.
    pub const SIZE: usize = size_of::<Q8_08>();
    /// The bytes the eight blocks it holds take in the file.
    pub const SOURCE: usize = 8 * Q8_0_BLOCK;
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
const Q5K_D: usize = offset_of!(Q5K8, d);
const Q5K_DMIN: usize = offset_of!(Q5K8, dmin);
const Q5K_SCM: usize = offset_of!(Q5K8, scm);
const Q5K_QH: usize = offset_of!(Q5K8, qh);
const Q5K_QS: usize = offset_of!(Q5K8, qs);
const Q5_D: usize = offset_of!(Q5_08, d);
const Q5_QH: usize = offset_of!(Q5_08, qh);
const Q5_QS: usize = offset_of!(Q5_08, qs);
const Q8_D: usize = offset_of!(Q8_08, d);
const Q8_QS: usize = offset_of!(Q8_08, qs);

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

/// The byte offset of the eight rows' block scales in a Q5_0 tile.
pub const fn q5_d() -> usize {
    Q5_D
}
/// The byte offset of the eight rows' fifth bits in a Q5_0 tile.
pub const fn q5_qh() -> usize {
    Q5_QH
}
/// The byte offset of the eight rows' nibbles in a Q5_0 tile.
pub const fn q5_qs() -> usize {
    Q5_QS
}
/// The bytes one row's nibbles take in one window of a Q5_0 tile.
pub const Q5_QS_ROW: usize = 8;
/// The bytes one window of a Q5_0 tile takes.
pub const Q5_WINDOW: usize = 64;

/// The byte offset of the eight rows' block scales in a Q8_0 tile.
pub const fn q8_d() -> usize {
    Q8_D
}
/// The byte offset of the eight rows' signed codes in a Q8_0 tile.
pub const fn q8_qs() -> usize {
    Q8_QS
}
/// The bytes one row's codes take in one window of a Q8_0 tile.
pub const Q8_QS_ROW: usize = 8;
/// The bytes one window of a Q8_0 tile takes.
pub const Q8_WINDOW: usize = 64;
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

/// The byte offset of the eight rows' super-block scales in a Q5_K tile.
pub const fn q5k_d() -> usize {
    Q5K_D
}
/// The byte offset of the eight rows' super-block min scales in a Q5_K tile.
pub const fn q5k_dmin() -> usize {
    Q5K_DMIN
}
/// The byte offset of the eight rows' six-bit sub-block scales and mins in a
/// Q5_K tile.
pub const fn q5k_scm() -> usize {
    Q5K_SCM
}
/// The byte offset of the eight rows' fifth bits in a Q5_K tile.
pub const fn q5k_qh() -> usize {
    Q5K_QH
}
/// The byte offset of the eight rows' nibbles in a Q5_K tile.
pub const fn q5k_qs() -> usize {
    Q5K_QS
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

/// Repack eight Q5_K super-blocks — one per row — into one tile.
///
/// The fifth bits are transposed at load: the file stores bit `sub` of byte `l`,
/// the tile stores bit `l % 8` of the byte for sub-block `sub` and window `l / 8`.
///
/// # Panics
///
/// If `src` does not hold eight blocks of [`Q5K_BLOCK`] bytes, or `dst` is shorter
/// than a tile.
pub fn repack_q5k8(src: &[&[u8]], dst: &mut [u8]) {
    assert_eq!(src.len(), 8, "a Q5_K tile holds eight rows");
    assert!(dst.len() >= Q5K8::SIZE);
    let mut sc = [0u64; 8];
    let mut m = [0u64; 8];
    for (r, b) in src.iter().enumerate() {
        assert!(b.len() >= Q5K_BLOCK, "a Q5_K super-block is 176 bytes");
        dst[Q5K_D + r * 2..Q5K_D + r * 2 + 2].copy_from_slice(&b[0..2]);
        dst[Q5K_DMIN + r * 2..Q5K_DMIN + r * 2 + 2].copy_from_slice(&b[2..4]);
        let (s, mn) = q4k_scales(&b[4..16]);
        for j in 0..8 {
            sc[j] |= (s[j] as u64) << (6 * r);
            m[j] |= (mn[j] as u64) << (6 * r);
        }
        for sub in 0..8 {
            for w in 0..4 {
                let mut v = 0u8;
                for k in 0..8 {
                    if (b[16 + w * 8 + k] >> sub) & 1 != 0 {
                        v |= 1 << k;
                    }
                }
                dst[Q5K_QH + (sub * 4 + w) * 8 + r] = v;
            }
        }
        for c in 0..16 {
            let at = Q5K_QS + c * 64 + r * 8;
            dst[at..at + 8].copy_from_slice(&b[48 + c * 8..48 + c * 8 + 8]);
        }
    }
    for j in 0..8 {
        let at = Q5K_SCM + j * Q4_SCM_STRIDE;
        dst[at..at + 6].copy_from_slice(&sc[j].to_le_bytes()[..6]);
        dst[at + 6..at + 12].copy_from_slice(&m[j].to_le_bytes()[..6]);
    }
}

/// The inverse of [`repack_q5k8`]: one tile back into eight Q5_K super-blocks.
///
/// # Panics
///
/// If `src` is shorter than a tile, or `dst` is shorter than
/// [`Q5K8::SOURCE`] bytes.
pub fn unrepack_q5k8(src: &[u8], dst: &mut [u8]) {
    assert!(src.len() >= Q5K8::SIZE);
    assert!(dst.len() >= Q5K8::SOURCE);
    for r in 0..8 {
        let b = &mut dst[r * Q5K_BLOCK..r * Q5K_BLOCK + Q5K_BLOCK];
        b[0..2].copy_from_slice(&src[Q5K_D + r * 2..Q5K_D + r * 2 + 2]);
        b[2..4].copy_from_slice(&src[Q5K_DMIN + r * 2..Q5K_DMIN + r * 2 + 2]);
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            sc[j] = tile_q4k_scale(src, j, r);
            m[j] = tile_q4k_min(src, j, r);
        }
        b[4..16].copy_from_slice(&q4k_pack_scales(&sc, &m));
        for l in 0..32 {
            let mut v = 0u8;
            for sub in 0..8 {
                let qh = src[Q5K_QH + (sub * 4 + l / 8) * 8 + r];
                v |= ((qh >> (l % 8)) & 1) << sub;
            }
            b[16 + l] = v;
        }
        for c in 0..16 {
            let at = 48 + c * 8;
            b[at..at + 8]
                .copy_from_slice(&src[Q5K_QS + c * 64 + r * 8..Q5K_QS + c * 64 + r * 8 + 8]);
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

/// Repack eight Q5_0 blocks — one per row — into one tile.
///
/// # Panics
///
/// If `src` does not hold eight blocks of [`Q5_0_BLOCK`] bytes, or `dst` is
/// shorter than a tile.
pub fn repack_q5_08(src: &[&[u8]], dst: &mut [u8]) {
    assert_eq!(src.len(), 8, "a Q5_0 tile holds eight rows");
    assert!(dst.len() >= Q5_08::SIZE);
    for (r, b) in src.iter().enumerate() {
        assert!(b.len() >= Q5_0_BLOCK, "a Q5_0 block is 22 bytes");
        dst[Q5_D + r * 2..Q5_D + r * 2 + 2].copy_from_slice(&b[0..2]);
        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
        for h in 0..Q5_0_WINDOWS {
            // The group's eight bits, one byte for this row.
            dst[Q5_QH + h * 8 + r] = (qh >> (8 * h)) as u8;
        }
        // The sixteen nibble bytes, once each: the two physical windows are the
        // file's `qs[0..8]` and `qs[8..16]`.
        for w in 0..2 {
            for k in 0..8 {
                dst[Q5_QS + w * Q5_WINDOW + r * Q5_QS_ROW + k] = b[6 + w * 8 + k];
            }
        }
    }
}

/// The inverse of [`repack_q5_08`]: one tile back into eight Q5_0 blocks.
///
/// The blocks are written in order: row `r`'s block at `r * 22`.
///
/// # Panics
///
/// If `src` is shorter than a tile, or `dst` is shorter than
/// [`Q5_08::SOURCE`] bytes.
pub fn unrepack_q5_08(src: &[u8], dst: &mut [u8]) {
    assert!(src.len() >= Q5_08::SIZE);
    assert!(dst.len() >= Q5_08::SOURCE);
    for r in 0..8 {
        let b = &mut dst[r * Q5_0_BLOCK..r * Q5_0_BLOCK + Q5_0_BLOCK];
        b[0..2].copy_from_slice(&src[Q5_D + r * 2..Q5_D + r * 2 + 2]);
        let mut qh = 0u32;
        for h in 0..Q5_0_WINDOWS {
            qh |= (src[Q5_QH + h * 8 + r] as u32) << (8 * h);
        }
        b[2..6].copy_from_slice(&qh.to_le_bytes());
        for w in 0..2 {
            for k in 0..8 {
                b[6 + w * 8 + k] = src[Q5_QS + w * Q5_WINDOW + r * Q5_QS_ROW + k];
            }
        }
    }
}

/// Repack eight Q8_0 blocks — one per row — into one tile.
///
/// # Panics
///
/// If `src` does not hold eight blocks of [`Q8_0_BLOCK`] bytes, or `dst` is
/// shorter than a tile.
pub fn repack_q8_08(src: &[&[u8]], dst: &mut [u8]) {
    assert_eq!(src.len(), 8, "a Q8_0 tile holds eight rows");
    assert!(dst.len() >= Q8_08::SIZE);
    for (r, b) in src.iter().enumerate() {
        assert!(b.len() >= Q8_0_BLOCK, "a Q8_0 block is 34 bytes");
        dst[Q8_D + r * 2..Q8_D + r * 2 + 2].copy_from_slice(&b[0..2]);
        for w in 0..Q8_0_WINDOWS {
            for k in 0..8 {
                dst[Q8_QS + w * Q8_WINDOW + r * Q8_QS_ROW + k] = b[2 + w * 8 + k];
            }
        }
    }
}

/// The inverse of [`repack_q8_08`]: one tile back into eight Q8_0 blocks.
///
/// The blocks are written in order: row `r`'s block at `r * 34`.
///
/// # Panics
///
/// If `src` is shorter than a tile, or `dst` is shorter than
/// [`Q8_08::SOURCE`] bytes.
pub fn unrepack_q8_08(src: &[u8], dst: &mut [u8]) {
    assert!(src.len() >= Q8_08::SIZE);
    assert!(dst.len() >= Q8_08::SOURCE);
    for r in 0..8 {
        let b = &mut dst[r * Q8_0_BLOCK..r * Q8_0_BLOCK + Q8_0_BLOCK];
        b[0..2].copy_from_slice(&src[Q8_D + r * 2..Q8_D + r * 2 + 2]);
        for w in 0..Q8_0_WINDOWS {
            for k in 0..8 {
                b[2 + w * 8 + k] = src[Q8_QS + w * Q8_WINDOW + r * Q8_QS_ROW + k];
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
    /// The per-32-weight sum of every row: `groups · super_blocks · 4 · 8`.
    ///
    /// The min term of the Q4_K format is the row's mins against the
    /// activation's sums, one per sub-block — and every product and partial sum
    /// of it is an integer below 2^24 (63 · 4064 · 8 = 2 048 256), so it is
    /// exact in either domain and the kernel folds it in int32 lanes.
    pub bs: Vec<i32>,
    /// The per-16-weight sum of every row: `groups · super_blocks · 4 · 16`.
    ///
    /// Q6_K changes its scale every sixteen weights, and its offset is applied
    /// once per run, so the tile kernel needs the finer partition that
    /// [`QuantAct::bsums`] already holds. The Q4_K kernels keep using `bs`.
    pub bs16: Vec<i32>,
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
        self.bs16.clear();
        self.bs16.resize(groups * n_sb * ACT_GROUP * 16, 0);
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
                let bbase16 = (g * n_sb + sb) * ACT_GROUP * 16;
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
                    // The per-16 run sums themselves, for Q6_K's offset and scale.
                    for (j, &s) in rs.iter().enumerate() {
                        self.bs16[bbase16 + t * 16 + j] = s as i32;
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

/// The value of one row of a repacked Q5_K tile against one activation row.
///
/// Q5_K is Q4_K plus a fifth bit per weight, so the expression is Q4_K's with the
/// codes widened to `0..31`:
///
/// ```text
/// value = Σ_sb dk[sb] · ( d[sb] · S − dmin[sb] · M )
/// ```
///
/// with `S = Σ_sub sc·(Σ_l code·q)` and `M = Σ_sub m·(Σ_l q)` the exact integer
/// sums of the super-block. The fifth bits are the tile's, the nibbles are the
/// file's, and the fold is [`dot_q4k8_ref`]'s, so the AVX2 kernel is checked
/// against the same real number the Q4_K tile computes.
///
/// # Panics
///
/// If the tile is shorter than `n_sb` tiles, or the activation does not carry
/// the per-256 layout.
pub fn dot_q5k8_ref(tile: &[u8], n_sb: usize, row: usize, act: &QuantAct) -> f32 {
    assert!(row < 8, "a tile holds eight rows");
    assert!(
        tile.len() >= n_sb * Q5K8::SIZE,
        "the tile is shorter than its super-blocks"
    );
    assert!(
        !act.qk.is_empty(),
        "a k-quant row needs the per-256 activation"
    );
    let mut acc = 0.0f32;
    for sb in 0..n_sb {
        let t = &tile[sb * Q5K8::SIZE..(sb + 1) * Q5K8::SIZE];
        let d = f16(u16::from_le_bytes([
            t[Q5K_D + row * 2],
            t[Q5K_D + row * 2 + 1],
        ]));
        let dmin = f16(u16::from_le_bytes([
            t[Q5K_DMIN + row * 2],
            t[Q5K_DMIN + row * 2 + 1],
        ]));
        let aq = &act.qk[sb * 256..sb * 256 + 256];
        let rs = &act.bsums[sb * 16..sb * 16 + 16];
        let mut sumi = 0i64;
        let mut summ = 0i64;
        for sub in 0..8 {
            let sc = tile_q4k_scale(t, sub, row) as i64;
            let mn = tile_q4k_min(t, sub, row) as i64;
            // The sub-block's 32 weights: the low nibbles of the row's bytes
            // 32·(sub/2) .. +32 when sub is even, the high nibbles when it is odd.
            let base = (sub / 2) * 32;
            let mut s = 0i64;
            for l in 0..32 {
                let at = Q5K_QS + (base + l) / 8 * 64 + row * 8 + (base + l) % 8;
                let nib = if sub % 2 == 0 {
                    (t[at] & 0x0F) as i64
                } else {
                    (t[at] >> 4) as i64
                };
                let qh = t[Q5K_QH + (sub * 4 + l / 8) * 8 + row];
                let hi = if (qh >> (l % 8)) & 1 != 0 { 16 } else { 0 };
                s += (nib | hi) * aq[sub * 32 + l] as i64;
            }
            sumi += sc * s;
            summ += mn * (rs[2 * sub] as i64 + rs[2 * sub + 1] as i64);
        }
        // The same expression the kernel folds.
        let x = (-dmin).mul_add(summ as f32, d * sumi as f32);
        acc = x.mul_add(act.dk[sb], acc);
    }
    acc
}

/// The value of one row of a repacked Q5_0 tile against one activation row.
///
/// This is the specification the AVX2 kernel is checked against. Q5_0 has one
/// scale per 32-weight block, so the fold is per block:
///
/// ```text
/// value = Σ_b (d[b] · dk[b]) · ( Σ_j code_j·q_j − 16·Σ_j q_j )
/// ```
///
/// with the integer sums exact. The expression is the kernel's, so the two agree
/// bit for bit; the existing scalar kernel uses the same real number with a
/// non-fused multiply-add, which differs by at most one rounding per block.
///
/// # Panics
///
/// If the tile is shorter than `n_blk` tiles, or the activation does not carry
/// the per-32 layout.
pub fn dot_q5_08_ref(tile: &[u8], n_blk: usize, row: usize, act: &QuantAct) -> f32 {
    assert!(row < 8, "a tile holds eight rows");
    assert!(
        tile.len() >= n_blk * Q5_08::SIZE,
        "the tile is shorter than its blocks"
    );
    assert!(
        !act.q.is_empty() && act.q.len() >= n_blk * Q5_0_WEIGHTS,
        "a Q5_0 row needs the per-32 activation"
    );
    let mut acc = 0.0f32;
    for t in 0..n_blk {
        let b = &tile[t * Q5_08::SIZE..(t + 1) * Q5_08::SIZE];
        let d = f16(u16::from_le_bytes([
            b[Q5_D + row * 2],
            b[Q5_D + row * 2 + 1],
        ]));
        let mut s = 0i64;
        for h in 0..Q5_0_WINDOWS {
            let qh = b[Q5_QH + h * 8 + row];
            let qs = Q5_QS + (h % 2) * Q5_WINDOW + row * Q5_QS_ROW;
            for k in 0..8 {
                let nib = if h < 2 {
                    b[qs + k] & 0x0F
                } else {
                    b[qs + k] >> 4
                };
                let bit = (qh >> k) & 1;
                let code = (nib + 16 * bit) as i64;
                s += code * act.q[t * Q5_0_WEIGHTS + h * 8 + k] as i64;
            }
        }
        let x = (s - 16 * act.sums[t] as i64) as f32;
        acc = (d * act.d[t]).mul_add(x, acc);
    }
    acc
}

/// The value of one row of a repacked Q8_0 tile against one activation row.
///
/// This is the specification the AVX2 and AVX-512 kernels are checked against.
/// Q8_0 has one scale per 32-weight block and no offset, so the fold is per block:
///
/// ```text
/// value = Σ_b (d[b] · dk[b]) · ( Σ_j code_j·q_j )
/// ```
///
/// with the integer sum exact and the accumulate an FMA, the expression the
/// kernels run. The ordinary scalar Q8_0 kernel computes the same real number
/// with a non-fused multiply-add, which differs by at most one rounding per block.
///
/// # Panics
///
/// If the tile is shorter than `n_blk` tiles, or the activation does not carry
/// the per-32 layout.
pub fn dot_q8_08_ref(tile: &[u8], n_blk: usize, row: usize, act: &QuantAct) -> f32 {
    assert!(row < 8, "a tile holds eight rows");
    assert!(
        tile.len() >= n_blk * Q8_08::SIZE,
        "the tile is shorter than its blocks"
    );
    assert!(
        !act.q.is_empty() && act.q.len() >= n_blk * Q8_0_WEIGHTS,
        "a Q8_0 row needs the per-32 activation"
    );
    let mut acc = 0.0f32;
    for t in 0..n_blk {
        let b = &tile[t * Q8_08::SIZE..(t + 1) * Q8_08::SIZE];
        let d = f16(u16::from_le_bytes([
            b[Q8_D + row * 2],
            b[Q8_D + row * 2 + 1],
        ]));
        let mut s = 0i64;
        for w in 0..Q8_0_WINDOWS {
            let qs = Q8_QS + w * Q8_WINDOW + row * Q8_QS_ROW;
            for k in 0..8 {
                s += b[qs + k] as i8 as i64 * act.q[t * Q8_0_WEIGHTS + w * 8 + k] as i64;
            }
        }
        acc = (d * act.d[t]).mul_add(s as f32, acc);
    }
    acc
}

/// The value of one row of a repacked Q6_K tile against one activation row.
///
/// This is the specification the NEON tile kernel is checked against, and it is exactly
/// [`crate::quant::dot_scalar`]'s Q6_K expression: the six-bit codes are unscaled
/// integers, the run's unsigned offset is applied once per sixteen weights against the
/// activation's run sum, the eight-bit scales and the super-block scale are applied in
/// the reference's order, and the integer sums are exact.
///
/// # Panics
///
/// If the tile is shorter than `n_sb` tiles, or the activation does not carry the per-256
/// layout.
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
        let mut sumi = 0i32;
        for c in 0..2 {
            let obase = sb * 256 + c * 128;
            for g in 0..4 {
                let nib_at = if g % 2 == 0 { 0 } else { 32 };
                let high = g >= 2;
                for hf in 0..2 {
                    let o = obase + g * 32 + hf * 16;
                    let mut s = 0i32;
                    for l in 0..16 {
                        let li = hf * 16 + l;
                        let ql =
                            Q6_QL + (c * 8 + (nib_at + li) / 8) * 64 + row * 8 + (nib_at + li) % 8;
                        let qh = Q6_QH + (c * 4 + li / 8) * 64 + row * 8 + li % 8;
                        let nib = if high { t[ql] >> 4 } else { t[ql] & 0x0F };
                        let code = (nib | (((t[qh] >> (2 * g)) & 3) << 4)) as i32;
                        s += code * act.qk[o + l] as i32;
                    }
                    let scale = t[Q6_SC + (c * 8 + 2 * g + hf) * 8 + row] as i8 as i32;
                    let bias = 32 * act.bsums[o / 16] as i32;
                    sumi += scale * (s - bias);
                }
            }
        }
        acc += d * act.dk[sb] * sumi as f32;
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
        assert_eq!(Q5K8::SIZE, 1408);
        assert_eq!(Q5K8::SOURCE, 1408);
        assert_eq!(Q6K8::SIZE, 1680);
        assert_eq!(Q6K8::SOURCE, 1680);
        assert_eq!(Q5_08::SIZE, 176);
        assert_eq!(Q5_08::SOURCE, 176);
        assert_eq!(Q8_08::SIZE, 272);
        assert_eq!(Q8_08::SOURCE, 272);
        // Size-neutral: the store replaces the file's bytes, it does not add to
        // them. Decode streams every byte of the store once per token.
        assert_eq!(Q4K8::SIZE, Q4K8::SOURCE);
        assert_eq!(Q5K8::SIZE, Q5K8::SOURCE);
        assert_eq!(Q6K8::SIZE, Q6K8::SOURCE);
        assert_eq!(Q5_08::SIZE, Q5_08::SOURCE);
        assert_eq!(Q8_08::SIZE, Q8_08::SOURCE);
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

    fn group5(src: &[u8]) -> Vec<&[u8]> {
        src.chunks_exact(Q5_0_BLOCK).take(8).collect()
    }

    #[test]
    fn a_q5_0_tile_gives_the_blocks_back() {
        let mut rng = Rng(0x0bad_f00d_5a5a_1234);
        for case in 0..8 {
            // All zeros, all ones, both fifth-bit extremes, and random bytes.
            let src: Vec<u8> = match case {
                0 => vec![0u8; 8 * Q5_0_BLOCK],
                1 => vec![0xffu8; 8 * Q5_0_BLOCK],
                2 => {
                    let mut v = vec![0u8; 8 * Q5_0_BLOCK];
                    for r in 0..8 {
                        v[r * Q5_0_BLOCK + 2..r * Q5_0_BLOCK + 6].copy_from_slice(&[0, 0, 0, 0]);
                    }
                    v
                }
                3 => {
                    let mut v = vec![0u8; 8 * Q5_0_BLOCK];
                    for r in 0..8 {
                        v[r * Q5_0_BLOCK + 2..r * Q5_0_BLOCK + 6]
                            .copy_from_slice(&[0xff, 0xff, 0xff, 0xff]);
                    }
                    v
                }
                _ => rng.bytes(8 * Q5_0_BLOCK),
            };
            let mut tile = vec![0u8; Q5_08::SIZE];
            repack_q5_08(&group5(&src), &mut tile);
            let mut back = vec![0u8; Q5_08::SOURCE];
            unrepack_q5_08(&tile, &mut back);
            assert_eq!(&back[..], &src[..], "case {case}");
        }
    }

    #[test]
    fn the_q5_0_tile_holds_each_row_where_the_layout_says() {
        let mut rng = Rng(0xfeed_5a5a_dead_0001);
        let src = rng.bytes(8 * Q5_0_BLOCK);
        let mut tile = vec![0u8; Q5_08::SIZE];
        repack_q5_08(&group5(&src), &mut tile);
        for r in 0..8 {
            assert_eq!(
                &tile[Q5_D + r * 2..Q5_D + r * 2 + 2],
                &src[r * Q5_0_BLOCK..r * Q5_0_BLOCK + 2]
            );
            let qh = u32::from_le_bytes([
                src[r * Q5_0_BLOCK + 2],
                src[r * Q5_0_BLOCK + 3],
                src[r * Q5_0_BLOCK + 4],
                src[r * Q5_0_BLOCK + 5],
            ]);
            for h in 0..Q5_0_WINDOWS {
                assert_eq!(
                    tile[Q5_QH + h * 8 + r],
                    (qh >> (8 * h)) as u8,
                    "row {r} group {h}'s fifth bits"
                );
            }
            for w in 0..2 {
                for k in 0..8 {
                    assert_eq!(
                        tile[Q5_QS + w * Q5_WINDOW + r * Q5_QS_ROW + k],
                        src[r * Q5_0_BLOCK + 6 + w * 8 + k],
                        "row {r} window {w} byte {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_q5_k_tile_gives_the_blocks_back() {
        let mut rng = Rng(0x5a5a_1c0f_0bad_9001);
        for case in 0..8 {
            let src: Vec<u8> = match case {
                0 => vec![0u8; 8 * Q5K_BLOCK],
                1 => vec![0xffu8; 8 * Q5K_BLOCK],
                2 => {
                    // The fifth bits all set, the six-bit fields at 63.
                    let mut v = vec![0u8; 8 * Q5K_BLOCK];
                    for r in 0..8 {
                        for b in v[r * Q5K_BLOCK + 16..r * Q5K_BLOCK + 48].iter_mut() {
                            *b = 0xff;
                        }
                        for b in v[r * Q5K_BLOCK + 4..r * Q5K_BLOCK + 16].iter_mut() {
                            *b = 0x3f;
                        }
                    }
                    v
                }
                3 => {
                    // The fifth bits clear, the mins at 63.
                    let mut v = vec![0u8; 8 * Q5K_BLOCK];
                    for r in 0..8 {
                        for b in v[r * Q5K_BLOCK + 4..r * Q5K_BLOCK + 16].iter_mut() {
                            *b = 0xc0;
                        }
                    }
                    v
                }
                _ => rng.bytes(8 * Q5K_BLOCK),
            };
            let mut tile = vec![0u8; Q5K8::SIZE];
            let refs: Vec<&[u8]> = src.chunks_exact(Q5K_BLOCK).take(8).collect();
            repack_q5k8(&refs, &mut tile);
            let mut back = vec![0u8; Q5K8::SOURCE];
            unrepack_q5k8(&tile, &mut back);
            assert_eq!(&back[..], &src[..], "case {case}");
        }
    }

    #[test]
    fn the_q5_k_tile_holds_each_row_where_the_layout_says() {
        let mut rng = Rng(0xfeed_5c5c_dead_0003);
        let src = rng.bytes(8 * Q5K_BLOCK);
        let mut tile = vec![0u8; Q5K8::SIZE];
        let refs: Vec<&[u8]> = src.chunks_exact(Q5K_BLOCK).take(8).collect();
        repack_q5k8(&refs, &mut tile);
        for r in 0..8 {
            assert_eq!(
                &tile[Q5K_D + r * 2..Q5K_D + r * 2 + 2],
                &src[r * Q5K_BLOCK..r * Q5K_BLOCK + 2]
            );
            assert_eq!(
                &tile[Q5K_DMIN + r * 2..Q5K_DMIN + r * 2 + 2],
                &src[r * Q5K_BLOCK + 2..r * Q5K_BLOCK + 4]
            );
            for sub in 0..8 {
                for w in 0..4 {
                    let mut want = 0u8;
                    for k in 0..8 {
                        if (src[r * Q5K_BLOCK + 16 + w * 8 + k] >> sub) & 1 != 0 {
                            want |= 1 << k;
                        }
                    }
                    assert_eq!(
                        tile[Q5K_QH + (sub * 4 + w) * 8 + r],
                        want,
                        "row {r} sub-block {sub} window {w}"
                    );
                }
            }
            for c in 0..16 {
                let at = Q5K_QS + c * 64 + r * 8;
                assert_eq!(
                    &tile[at..at + 8],
                    &src[r * Q5K_BLOCK + 48 + c * 8..r * Q5K_BLOCK + 48 + c * 8 + 8]
                );
            }
        }
    }

    fn group8(src: &[u8]) -> Vec<&[u8]> {
        src.chunks_exact(Q8_0_BLOCK).take(8).collect()
    }

    #[test]
    fn a_q8_0_tile_gives_the_blocks_back() {
        let mut rng = Rng(0x0bad_f00d_8080_1234);
        for case in 0..8 {
            // All zeros, all 0xFF, the signed extremes, and random bytes.
            let src: Vec<u8> = match case {
                0 => vec![0u8; 8 * Q8_0_BLOCK],
                1 => vec![0xffu8; 8 * Q8_0_BLOCK],
                2 => {
                    let mut v = vec![0u8; 8 * Q8_0_BLOCK];
                    for r in 0..8 {
                        for k in 2..Q8_0_BLOCK {
                            v[r * Q8_0_BLOCK + k] = 0x80;
                        }
                    }
                    v
                }
                3 => {
                    let mut v = vec![0u8; 8 * Q8_0_BLOCK];
                    for r in 0..8 {
                        for k in 2..Q8_0_BLOCK {
                            v[r * Q8_0_BLOCK + k] = 0x7f;
                        }
                    }
                    v
                }
                _ => rng.bytes(8 * Q8_0_BLOCK),
            };
            let mut tile = vec![0u8; Q8_08::SIZE];
            repack_q8_08(&group8(&src), &mut tile);
            let mut back = vec![0u8; Q8_08::SOURCE];
            unrepack_q8_08(&tile, &mut back);
            assert_eq!(&back[..], &src[..], "case {case}");
        }
    }

    #[test]
    fn the_q8_0_tile_holds_each_row_where_the_layout_says() {
        let mut rng = Rng(0xfeed_8080_dead_0002);
        let src = rng.bytes(8 * Q8_0_BLOCK);
        let mut tile = vec![0u8; Q8_08::SIZE];
        repack_q8_08(&group8(&src), &mut tile);
        for r in 0..8 {
            assert_eq!(
                &tile[Q8_D + r * 2..Q8_D + r * 2 + 2],
                &src[r * Q8_0_BLOCK..r * Q8_0_BLOCK + 2]
            );
            for w in 0..4 {
                for k in 0..8 {
                    assert_eq!(
                        tile[Q8_QS + w * Q8_WINDOW + r * Q8_QS_ROW + k] as u8,
                        src[r * Q8_0_BLOCK + 2 + w * 8 + k],
                        "row {r} window {w} byte {k}"
                    );
                }
            }
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
