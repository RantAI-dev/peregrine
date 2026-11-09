//! The repacked k-quant kernels (S3.4): the AVX2 and AVX-512 GEMM and GEMV over
//! the eight-row tiles of [`crate::kq::Q4K8`], [`crate::kq::Q5_08`] and
//! [`crate::kq::Q8_08`].
//!
//! # What the repack buys
//!
//! The tile holds eight output rows side by side, so one pass over the weights
//! serves eight outputs. The activation is read in four rows at a time, four
//! rows side by side ([`crate::kq::KqActs`]), and one `vpmaddubsw` then computes
//! eight distinct (row, activation) partial products of four bytes each — the
//! `_sp1`/`_sp2` shuffle trick of the design note (`docs/design/s3.4.md` §2.3).
//!
//! The consequence that matters is not the instruction count but the *shape* of
//! the arithmetic: the eight lanes of the accumulator are eight different
//! outputs, so the eight sub-block sums of a super-block are reduced in int32
//! inside the super-block and the float fold happens once per super-block
//! instead of once per (row, activation) pair. The old kernel's 64-float
//! accumulator spill disappears with it.
//!
//! # The fold, and why the two paths agree
//!
//! Both paths compute, for every (output row, activation row),
//!
//! ```text
//! value = Σ_sb dk[sb] · ( d[sb] · S − dmin[sb] · M )
//! ```
//!
//! with `S` and `M` exact integers and the same sequential FMA chain over the
//! super-blocks. A row computed alone (the GEMV, decode) and the same row
//! computed in a batch (the GEMM, prefill) therefore agree bit for bit — the
//! batch-invariance requirement of the design note's decisions 8 and 9.
//!
//! # The int16 stage, and the AVX-512 kernel
//!
//! The AVX2 kernel's int16 accumulator holds sixteen products per lane: eight
//! `vpmaddubsw` results, two products each. The bound is `16 · 15 · 127 =
//! 30 480`, inside int16's 32 767 — the same bound the design note works out,
//! and the test `the_int16_stage_has_headroom` in this file asserts it.
//!
//! The AVX-512 kernel (`q4k8_gemm_zmm`, `q5_08_gemm_zmm`, ...) has no int16
//! stage at all: `vpdpbusd` accumulates four products per int32 lane, and the
//! scale is applied with an exact int32 multiply on a different execution unit
//! from the byte dot product. It computes the same integers and folds them the
//! same way, so a row has the same bits as on the AVX2 tier — the test
//! `the_avx512_kernel_agrees_with_the_avx2_one_bit_for_bit` asserts it at every
//! batch width.

#![cfg(target_arch = "x86_64")]
#![cfg_attr(not(feature = "fmt-kquant"), allow(dead_code))]

use crate::kq::{KqActs, ACT_GROUP, Q4K8, Q5_08, Q8_08};
use crate::quant::QuantAct;
use std::arch::x86_64::*;

/// The rows one tile holds.
const ROWS: usize = 8;
/// The windows of a super-block: sixteen groups of eight weights per row.
const WINDOWS: usize = 16;
/// The sub-blocks of a super-block.
const SUBS: usize = 8;

/// The byte pairs the six-bit unpack reads: value `i` of a packed group lives at
/// bits `6i`, so it is the low bits of byte `6i/8` and the high bits of the byte
/// after it.
static PAIRS: [u8; 32] = {
    let mut t = [0x80u8; 32];
    let mut i = 0;
    while i < 8 {
        let k = 6 * i / 8;
        t[4 * i] = k as u8;
        t[4 * i + 1] = (k + 1) as u8;
        i += 1;
    }
    t
};

/// The shift each six-bit field of a packed group needs, after the byte pair is
/// in a 32-bit lane.
static SHIFTS: [i32; 8] = [0, 6, 4, 2, 0, 6, 4, 2];

/// The byte shuffle that takes a window's two rows out of their 128-bit lane.
///
/// The lane holds two rows' eight bytes; the shuffle brings the first four
/// bytes of each row to the front and leaves the second four behind, so the
/// four-byte groups of the eight rows end up one per 32-bit lane.
static TRANSPOSE: [u8; 32] = {
    let mut t = [0u8; 32];
    let mut lane = 0;
    while lane < 2 {
        let base = lane * 16;
        let mut k = 0;
        while k < 4 {
            t[base + k] = k as u8;
            t[base + 4 + k] = (8 + k) as u8;
            t[base + 8 + k] = (4 + k) as u8;
            t[base + 12 + k] = (12 + k) as u8;
            k += 1;
        }
        lane += 1;
    }
    t
};

/// The tile's eight rows in the kernel's lane order.
///
/// The transpose's unpack puts the first row of each half next to each other,
/// so the lanes come out in this order rather than the tile's. The scale and
/// min shuffles and the output write follow it.
const LANES: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// The dword permutation that puts the tile's rows in the lane order.
static LANEPERM: [i32; 8] = {
    let mut t = [0i32; 8];
    let mut k = 0;
    while k < 8 {
        t[k] = LANES[k] as i32;
        k += 1;
    }
    t
};

/// Where the repacked kernel reads an activation row's codes from.
///
/// The GEMM reads the interleaved layout — four rows side by side — and the
/// GEMV the plain per-row one. Both hold the same codes, so both give the same
/// results; only the way the four-byte groups are addressed differs.
pub trait ActSrc {
    /// The pointers to the four four-byte code groups one window pairs with:
    /// the low nibbles' two groups (the weights `8w .. 8w+4` and `8w+4 ..
    /// 8w+8`) and the high nibbles' two (the weights `8w+32 .. 8w+40`).
    ///
    /// # Safety
    ///
    /// The pointers must be valid for four bytes each.
    unsafe fn quads(&self, row: usize, sb: usize, w: usize) -> [*const u8; 4];
    /// The per-super-block scale of the row.
    fn scale(&self, row: usize, sb: usize) -> f32;
    /// The activation's sum over the sub-block `j` of the super-block.
    ///
    /// The min term of the Q4_K format: the row's six-bit mins against these
    /// sums. The sum is exact in either domain — a sub-block's sum is at most
    /// 4064, and the super-block's min term at most 2 048 256, so the int32
    /// accumulation and the float's exact-integer range agree bit for bit
    /// (`the_min_term_is_exact_in_int32_and_in_float` in this file).
    fn min_sum(&self, row: usize, sb: usize, j: usize) -> i32;
}

/// The interleaved activation: four rows side by side.
pub struct Interleaved<'a> {
    /// The buffer.
    pub acts: &'a KqActs,
}

impl ActSrc for Interleaved<'_> {
    #[inline(always)]
    unsafe fn quads(&self, row: usize, sb: usize, w: usize) -> [*const u8; 4] {
        let g = row / ACT_GROUP;
        let t = row % ACT_GROUP;
        let n_sb = self.acts.super_blocks();
        // The window's low nibbles are the weights 64·(w/4) + 8·(w%4) .. +8, and
        // its high nibbles the same 32 weights further along: the two halves of
        // a window's eight bytes are a whole sub-block apart.
        let j_lo = (w / 4) * 8 + (w % 4);
        let j_hi = j_lo + 4;
        let base = (g * n_sb + sb) * 1024;
        let lo = self.acts.q.as_ptr().add(base + j_lo * 32 + t * 8);
        let hi = self.acts.q.as_ptr().add(base + j_hi * 32 + t * 8);
        [
            lo as *const u8,
            lo.add(4) as *const u8,
            hi as *const u8,
            hi.add(4) as *const u8,
        ]
    }

    #[inline(always)]
    fn scale(&self, row: usize, sb: usize) -> f32 {
        let g = row / ACT_GROUP;
        let t = row % ACT_GROUP;
        let n_sb = self.acts.super_blocks();
        self.acts.d[(g * n_sb + sb) * ACT_GROUP + t]
    }

    #[inline(always)]
    fn min_sum(&self, row: usize, sb: usize, j: usize) -> i32 {
        let g = row / ACT_GROUP;
        let t = row % ACT_GROUP;
        let n_sb = self.acts.super_blocks();
        self.acts.bs[(g * n_sb + sb) * ACT_GROUP * 8 + t * 8 + j]
    }
}

/// The plain per-row activation: one row at a time.
pub struct Plain<'a> {
    /// The rows.
    pub acts: &'a [QuantAct],
}

impl ActSrc for Plain<'_> {
    #[inline(always)]
    unsafe fn quads(&self, row: usize, sb: usize, w: usize) -> [*const u8; 4] {
        // The window's low nibbles are the weights 64·(w/4) + 8·(w%4) .. +8, and
        // its high nibbles the same 32 weights further along.
        let lo = sb * 256 + (w / 4) * 64 + (w % 4) * 8;
        let hi = lo + 32;
        let p = self.acts[row].qk.as_ptr();
        [
            p.add(lo) as *const u8,
            p.add(lo + 4) as *const u8,
            p.add(hi) as *const u8,
            p.add(hi + 4) as *const u8,
        ]
    }

    #[inline(always)]
    fn scale(&self, row: usize, sb: usize) -> f32 {
        self.acts[row].dk[sb]
    }

    #[inline(always)]
    fn min_sum(&self, row: usize, sb: usize, j: usize) -> i32 {
        // The per-16 run sums, added into the sub-block's sum.
        let rs = &self.acts[row].bsums[sb * 16..sb * 16 + 16];
        rs[2 * j] as i32 + rs[2 * j + 1] as i32
    }
}

/// The four bytes at `p`, broadcast to the whole register.
///
/// The activation operand of a `vpmaddubsw`: one four-byte group of the
/// activation, repeated eight times so that it lines up with the eight rows of
/// the weight operand.
#[inline(always)]
unsafe fn bcast4(p: *const u8) -> __m256i {
    _mm256_set1_epi32(std::ptr::read_unaligned(p as *const i32))
}

/// The rows' codes of one window, in the two four-byte groups the window's
/// eight bytes split into.
///
/// The first vector holds the rows' first four bytes (the weights `8w .. w+4`),
/// the second their last four (the weights `8w+4 .. 8w+8`).
#[inline(always)]
unsafe fn window_groups(qs: *const u8, w: usize) -> (__m256i, __m256i) {
    // The window's eight rows sit side by side: rows 0..3 in the first 32
    // bytes, rows 4..7 in the next.
    let a = _mm256_loadu_si256(qs.add(w * 64) as *const __m256i);
    let b = _mm256_loadu_si256(qs.add(w * 64 + 32) as *const __m256i);
    let ctl = _mm256_loadu_si256(TRANSPOSE.as_ptr() as *const __m256i);
    let lo = _mm256_shuffle_epi8(a, ctl);
    let hi = _mm256_shuffle_epi8(b, ctl);
    // The two halves' rows have to be interleaved: the unpack puts the first
    // row of each half next to each other, which is the lane order.
    // The lane selectors: the low lane's source and the high lane's, each
    // 0 = the first operand's low lane, 1 = its high lane, 2 = the second
    // operand's low lane, 3 = its high lane.
    let x = _mm256_permute2x128_si256(lo, hi, 0x10);
    let y = _mm256_permute2x128_si256(lo, hi, 0x32);
    (_mm256_unpacklo_epi32(x, y), _mm256_unpackhi_epi32(x, y))
}

/// The eight six-bit values packed into the six bytes at `p`, one per 32-bit
/// lane, in the order the bytes hold them.
///
/// The value `i` sits at bits `6i`, so it is the low bits of byte `6i/8`
/// shifted down by `6i % 8`, with the next byte supplying what shifts in. A
/// byte-pair gather puts both source bytes in each 32-bit lane, one variable
/// shift extracts the field, and the mask drops the neighbours.
#[inline(always)]
unsafe fn unpack6(p: *const u8) -> __m256i {
    let src = _mm256_broadcastsi128_si256(_mm_loadu_si128(p as *const __m128i));
    let pairs = _mm256_shuffle_epi8(src, _mm256_loadu_si256(PAIRS.as_ptr() as *const __m256i));
    let v = _mm256_srlv_epi32(pairs, _mm256_loadu_si256(SHIFTS.as_ptr() as *const __m256i));
    _mm256_and_si256(v, _mm256_set1_epi32(63))
}

/// The eight rows' six-bit scales and mins of a whole tile, in the lane order.
///
/// The store keeps them packed — a tile is then the size of the file's own
/// bytes, which is what a decode step streams — so the kernel undoes the packing
/// once per super-block, before its windows: `sv[j]` is sub-block `j`'s scale
/// for the eight rows, each twice (the scale madd gives one lane per row), and
/// `mv[j]` is its eight mins as int32.
#[inline(always)]
unsafe fn unpack_scales(t: *const u8) -> ([__m256i; SUBS], [__m256i; SUBS]) {
    let lane = _mm256_loadu_si256(LANEPERM.as_ptr() as *const __m256i);
    let mut sv = [_mm256_setzero_si256(); SUBS];
    let mut mv = [_mm256_setzero_si256(); SUBS];
    for j in 0..SUBS {
        let at = t.add(crate::kq::q4_scm() + j * crate::kq::Q4_SCM_STRIDE);
        let sc = _mm256_permutevar8x32_epi32(unpack6(at), lane);
        let m = _mm256_permutevar8x32_epi32(unpack6(at.add(6)), lane);
        // Each row's scale twice, in the two 16-bit lanes of its accumulator lane.
        let p = _mm256_packs_epi32(sc, sc);
        sv[j] = _mm256_unpacklo_epi16(p, p);
        mv[j] = m;
    }
    (sv, mv)
}

/// The float fold of one super-block: `x = d·S − dmin·M`, `acc += x·dk`.
///
/// The same expression the ordinary Q4_K kernel folds (`simd.rs`'s `q4k_fold`),
/// so the two paths compute the same real number.
#[inline(always)]
unsafe fn fold(
    acc: __m256,
    sumi: __m256i,
    minterm: __m256,
    dv: __m256,
    dminv: __m256,
    dk: f32,
) -> __m256 {
    let x = _mm256_fnmadd_ps(dminv, minterm, _mm256_mul_ps(dv, _mm256_cvtepi32_ps(sumi)));
    _mm256_fmadd_ps(x, _mm256_set1_ps(dk), acc)
}

/// The most activation rows one pass over a tile decodes its weights for.
///
/// The activation side of the kernel is four rows per interleaved group and the
/// weight side is eight output rows, so eight activation rows (two groups) are
/// the widest block whose per-row state — one integer sum and one float
/// accumulator, two registers — still leaves room for the decoded window and
/// the scale/min vectors. Sixteen rows would need thirty-two accumulator
/// registers against the sixteen an AVX2 core has. Eight halves the number of
/// times a tile's weights are decoded per prefill pass, which is the decode the
/// four-row block repeated once per group.
const MAX_GROUP_ROWS: usize = 2 * ACT_GROUP;

/// The scale and min vectors of one super-block, in the kernel's lane order.
#[inline(always)]
unsafe fn q4k8_scale(
    t: *const u8,
    lane: __m256i,
) -> (__m256, __m256, [__m256i; SUBS], [__m256i; SUBS]) {
    // The scales and the mins come in the tile's row order; the fold wants
    // the lane order.
    let dv = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(
        _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(
            t.add(crate::kq::q4_d()) as *const __m128i
        ))),
        lane,
    ));
    let dminv = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(
        _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(
            t.add(crate::kq::q4_dmin()) as *const __m128i,
        ))),
        lane,
    ));
    let (sv, mv) = unpack_scales(t);
    (dv, dminv, sv, mv)
}

/// The mins packed into the even 16-bit lanes, zeros in the odd, so the min term
/// is one `vpmaddwd` per sub-block per row instead of a 32-bit multiply.
///
/// Only the sixteen-row block builds these: a decode row would spend more on
/// packing them than the multiply they save.
#[inline(always)]
unsafe fn q4k8_mins16(mv: &[__m256i; SUBS]) -> [__m256i; SUBS] {
    let zero = _mm256_setzero_si256();
    let mut mv16 = [_mm256_setzero_si256(); SUBS];
    for j in 0..SUBS {
        // `packs` puts the eight mins in both halves of each 128-bit lane; the
        // unpack then leaves each min in an even 16-bit lane.
        let p = _mm256_packs_epi32(mv[j], mv[j]);
        mv16[j] = _mm256_unpacklo_epi16(p, zero);
    }
    mv16
}

/// A window's four code vectors: its low nibbles in two four-byte groups and
/// its high nibbles in the same two.
type WindowCodes = (__m256i, __m256i, __m256i, __m256i);

/// Where a super-block's window codes come from.
///
/// The GEMV decodes a window just before using it; the sixteen-row GEMM decodes
/// the whole super-block once and hands the same vectors to both row groups.
trait CodeSrc {
    /// # Safety
    ///
    /// `w` must be a window of the super-block the source was built for.
    unsafe fn window(&self, w: usize) -> WindowCodes;
}

/// The window decode, done on demand: two loads and the nibble masks.
struct InlineCodes {
    t: *const u8,
}

impl CodeSrc for InlineCodes {
    #[inline(always)]
    unsafe fn window(&self, w: usize) -> WindowCodes {
        let lownib = _mm256_set1_epi8(0x0F);
        let (g0, g1) = window_groups(self.t.add(crate::kq::q4_qs()), w);
        (
            _mm256_and_si256(g0, lownib),
            _mm256_and_si256(g1, lownib),
            _mm256_and_si256(_mm256_srli_epi16(g0, 4), lownib),
            _mm256_and_si256(_mm256_srli_epi16(g1, 4), lownib),
        )
    }
}

/// The sixteen windows of a super-block's codes, decoded once.
#[inline(always)]
unsafe fn q4k8_codes(t: *const u8) -> [WindowCodes; WINDOWS] {
    let src = InlineCodes { t };
    let mut codes = [(
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
    ); WINDOWS];
    for w in 0..WINDOWS {
        codes[w] = src.window(w);
    }
    codes
}

/// The codes [`q4k8_codes`] decoded.
struct DecodedCodes<'a> {
    codes: &'a [WindowCodes; WINDOWS],
}

impl CodeSrc for DecodedCodes<'_> {
    #[inline(always)]
    unsafe fn window(&self, w: usize) -> WindowCodes {
        self.codes[w]
    }
}

/// One super-block of a tile against `N` activation rows, folded into `facc`.
///
/// The window and scale work does not change with the batch or with where the
/// codes come from: the sixteen-row block hands the same `codes` to both row
/// groups, the GEMV and the narrow groups decode each window in place.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn q4k8_sb<C: CodeSrc, S: ActSrc, const N: usize>(
    codes: &C,
    sb: usize,
    src: &S,
    row0: usize,
    dv: __m256,
    dminv: __m256,
    sv: &[__m256i; SUBS],
    mv: &[__m256i; SUBS],
    facc: &mut [__m256; N],
) {
    let mut sumi = [_mm256_setzero_si256(); N];
    // A window's low nibbles are the weights of sub-block 2g and its high
    // nibbles sub-block 2g+1, so one decode of the window feeds both. The
    // weights of a super-block are unpacked once, not once per sub-block.
    for g in 0..WINDOWS / 4 {
        let mut lo16 = [_mm256_setzero_si256(); N];
        let mut hi16 = [_mm256_setzero_si256(); N];
        for k in 0..4 {
            let w = g * 4 + k;
            let (vlo0, vlo1, vhi0, vhi1) = codes.window(w);
            for u in 0..N {
                let q = src.quads(row0 + u, sb, w);
                let (x1, x2) = (bcast4(q[0]), bcast4(q[1]));
                let (y1, y2) = (bcast4(q[2]), bcast4(q[3]));
                // Two `vpmaddubsw` per nibble half: one for each four-byte
                // group of the window. Their int16 lanes are the rows' two
                // products each, the rows in the lanes' order. Eight
                // `vpmaddubsw` reach one lane before the scale madd, which is
                // the int16 headroom the test below pins.
                let p = _mm256_add_epi16(
                    _mm256_maddubs_epi16(vlo0, x1),
                    _mm256_maddubs_epi16(vlo1, x2),
                );
                lo16[u] = _mm256_add_epi16(lo16[u], p);
                let p = _mm256_add_epi16(
                    _mm256_maddubs_epi16(vhi0, y1),
                    _mm256_maddubs_epi16(vhi1, y2),
                );
                hi16[u] = _mm256_add_epi16(hi16[u], p);
            }
        }
        // The scale madd: the int16 sums become the int32 sums of the
        // super-block, one lane per row.
        let (svlo, svhi) = (sv[2 * g], sv[2 * g + 1]);
        for u in 0..N {
            sumi[u] = _mm256_add_epi32(sumi[u], _mm256_madd_epi16(svlo, lo16[u]));
            sumi[u] = _mm256_add_epi32(sumi[u], _mm256_madd_epi16(svhi, hi16[u]));
        }
    }
    // The min term: the rows' six-bit mins against the activation's
    // per-sub-block sums. Every product is at most 63 · 4064, and every
    // partial sum at most 2 048 256, so all of them are integers the f32
    // can hold exactly, and the int32 accumulation and the float chain
    // round to the same bits.
    for u in 0..N {
        let mut mt = _mm256_setzero_si256();
        for j in 0..SUBS {
            mt = _mm256_add_epi32(
                mt,
                _mm256_mullo_epi32(mv[j], _mm256_set1_epi32(src.min_sum(row0 + u, sb, j))),
            );
        }
        facc[u] = fold(
            facc[u],
            sumi[u],
            _mm256_cvtepi32_ps(mt),
            dv,
            dminv,
            src.scale(row0 + u, sb),
        );
    }
}

/// The eight lanes are the eight output rows, in the lane order.
#[inline(always)]
unsafe fn q4k8_write<const N: usize>(facc: &[__m256; N], nvalid: usize, out: &mut [f32]) {
    let mut lane = [0f32; ROWS];
    for u in 0..nvalid {
        _mm256_storeu_ps(lane.as_mut_ptr(), facc[u]);
        for k in 0..ROWS {
            out[u * ROWS + LANES[k]] = lane[k];
        }
    }
}

/// Eight output rows of a repacked Q4_K tile against `N` activation rows.
///
/// The tile is `n_sb` super-blocks of eight rows; `out` receives
/// `out[b · 8 + r]` for the rows `0 .. nvalid` of the group whose first row is
/// `row0` of the activation.
///
/// `N` is a const generic rather than a parameter so that the activation loop
/// unrolls: with a runtime bound the compiler kept the loop rolled and branched
/// on the parity of the sub-block inside it (measured: the single-row GEMV cost
/// more than the row-at-a-time kernel it replaced). The rows are independent —
/// row `b`'s float chain does not involve any other row — so the instantiations
/// compute the same bits as each other.
///
/// # Safety
///
/// The caller must have checked that the CPU has AVX2, FMA and F16C. The tile
/// must hold `n_sb` tiles, and the activation must carry the per-256 layout.
/// Rows past the activation's own count must be readable (the interleaved buffer
/// pads a group with zero codes); the GEMM only ever asks for them with `N`
/// covering a padded group, whose results are discarded.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4k8_group<S: ActSrc, const N: usize>(
    tile: &[u8],
    n_sb: usize,
    src: &S,
    row0: usize,
    nvalid: usize,
    out: &mut [f32],
) {
    assert!(
        (1..=MAX_GROUP_ROWS).contains(&N),
        "one to eight activation rows"
    );
    assert!(nvalid > 0 && nvalid <= N, "the valid rows fit the group");
    assert!(
        out.len() >= nvalid * ROWS,
        "the output holds eight rows per row"
    );
    let lane = _mm256_loadu_si256(LANEPERM.as_ptr() as *const __m256i);
    let mut facc = [_mm256_setzero_ps(); N];
    for sb in 0..n_sb {
        let t = tile.as_ptr().add(sb * Q4K8::SIZE);
        let (dv, dminv, sv, mv) = q4k8_scale(t, lane);
        let codes = InlineCodes { t };
        q4k8_sb(&codes, sb, src, row0, dv, dminv, &sv, &mv, &mut facc);
    }
    q4k8_write(&facc, nvalid, &mut out[row0 * ROWS..]);
}

/// Sixteen output rows (two groups of eight) of a repacked Q4_K tile against
/// sixteen activation rows.
///
/// The prefill case: a prompt chunk is sixteen rows, and the two groups share
/// the super-block's scale and min unpack. The per-row arithmetic is
/// `q4k8_group`'s exactly — the seconds group's `row0` is just eight rows on —
/// so every row's bits are what the one-group kernel gives it.
///
/// # Safety
///
/// As [`q4k8_group`], with `nvalid <= 16`.
#[inline(never)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4k8_pair<S: ActSrc>(
    tile: &[u8],
    n_sb: usize,
    src: &S,
    row0: usize,
    nvalid: usize,
    out: &mut [f32],
) {
    assert!(
        (9..=16).contains(&nvalid),
        "a pair covers nine to sixteen rows"
    );
    assert!(
        out.len() >= nvalid * ROWS,
        "the output holds eight rows per row"
    );
    let lane = _mm256_loadu_si256(LANEPERM.as_ptr() as *const __m256i);
    let mut fa = [_mm256_setzero_ps(); 8];
    let mut fb = [_mm256_setzero_ps(); 8];
    for sb in 0..n_sb {
        let t = tile.as_ptr().add(sb * Q4K8::SIZE);
        let (dv, dminv, sv, mv) = q4k8_scale(t, lane);
        let mv16 = q4k8_mins16(&mv);
        let decoded = q4k8_codes(t);
        let codes = DecodedCodes { codes: &decoded };
        q4k8_sb(&codes, sb, src, row0, dv, dminv, &sv, &mv16, &mut fa);
        q4k8_sb(&codes, sb, src, row0 + 8, dv, dminv, &sv, &mv16, &mut fb);
    }
    q4k8_write(&fa, nvalid.min(8), out);
    q4k8_write(&fb, nvalid - 8, &mut out[8 * ROWS..]);
}

/// Eight output rows of a repacked Q4_K tile against a batch of activations.
///
/// The interleaved activation carries four rows at a time, so the batch is
/// walked in groups of four; a group with fewer rows left is padded with zero
/// codes, whose results are discarded.
///
/// # Safety
///
/// The caller must have checked that the CPU has AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q4k8_gemm(tile: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    let src = Interleaved { acts };
    let stride = MAX_GROUP_ROWS;
    // A prompt chunk is sixteen rows: one pass per tile for both groups.
    let mut b0 = 0;
    while nb - b0 >= 2 * stride {
        q4k8_pair(tile, n_sb, &src, b0, 2 * stride, &mut out[b0 * ROWS..]);
        b0 += 2 * stride;
    }
    // The batch's groups, not the buffer's: a caller may pass a narrower `nb`
    // than the interleaved buffer holds.
    let groups = (nb - b0).div_ceil(stride);
    for g in 0..groups {
        let b0 = b0 + g * stride;
        let n = (nb - b0).min(stride);
        // A group of eight rows (two interleaved groups) is the common case; the
        // last group of a batch whose width is not a multiple of eight is narrow.
        match n {
            8 => q4k8_group::<_, 8>(tile, n_sb, &src, b0, n, out),
            7 => q4k8_group::<_, 7>(tile, n_sb, &src, b0, n, out),
            6 => q4k8_group::<_, 6>(tile, n_sb, &src, b0, n, out),
            5 => q4k8_group::<_, 5>(tile, n_sb, &src, b0, n, out),
            4 => q4k8_group::<_, 4>(tile, n_sb, &src, b0, n, out),
            3 => q4k8_group::<_, 3>(tile, n_sb, &src, b0, n, out),
            2 => q4k8_group::<_, 2>(tile, n_sb, &src, b0, n, out),
            1 => q4k8_group::<_, 1>(tile, n_sb, &src, b0, n, out),
            0 => break,
            _ => unreachable!("a group holds eight activation rows"),
        }
    }
}

/// Eight output rows of a repacked Q4_K tile against one to three activations.
///
/// The decode path: the activation is read in the plain per-row layout, one row
/// at a time, and the result is the same as the batched path's.
///
/// # Safety
///
/// The caller must have checked that the CPU has AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q4k8_gemv(tile: &[u8], n_sb: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!((1..=3).contains(&nb), "the GEMV takes one to three rows");
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    // The rows together: the weight side is then unpacked once for all of them.
    let src = Plain { acts };
    match nb {
        1 => q4k8_group::<_, 1>(tile, n_sb, &src, 0, nb, out),
        2 => q4k8_group::<_, 2>(tile, n_sb, &src, 0, nb, out),
        _ => q4k8_group::<_, 3>(tile, n_sb, &src, 0, nb, out),
    }
}

// ---------------------------------------------------------------------------
// The AVX-512 kernel (S3.4 follow-up): the same eight-row tiles, 64 bytes at a time
// ---------------------------------------------------------------------------
//
// A ZMM holds a whole window — eight rows' eight bytes — so the AVX2 kernel's
// transpose and lane permutation disappear: the tile's window is already eight
// rows side by side. `vpdpbusd` then accumulates four byte products per int32
// lane, and the eight rows fall out in natural order (two int32 lanes per row),
// with no int16 stage at all: there is no saturation question, because every
// partial sum is an int32 (`the_avx512_int32_stage_has_headroom` asserts the
// bound).
//
// The arithmetic is kept identical to the AVX2 kernel, not merely close. The
// scale multiply is exact int32 either way, the min term is exact int32 either
// way, and the float fold is the same expression with the same operands in the
// same order, so a row has the same bits on the AVX-512 tier as on the AVX2 one
// and the AVX-512 goldens do not move.

/// Whether the AVX-512 repacked kernel can run: 512-bit integer byte/dword
/// arithmetic, VNNI's byte dot product, and the AVX2 baseline the fold needs.
#[inline]
pub fn have_zmm512() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::cpu_has!("avx512f")
            && crate::cpu_has!("avx512bw")
            && crate::cpu_has!("avx512vl")
            && crate::cpu_has!("avx512vnni")
            && crate::cpu_has!("fma")
            && crate::cpu_has!("f16c")
    })
}

/// The eight bytes at `p`, broadcast to the whole register.
///
/// The activation operand of a ZMM `vpdpbusd`: one window's eight codes of an
/// activation row, repeated eight times so that it lines up with the eight rows
/// of the weight operand.
#[inline(always)]
unsafe fn bcast8(p: *const u8) -> __m512i {
    _mm512_set1_epi64(std::ptr::read_unaligned(p as *const i64))
}

/// The index that duplicates each of eight int32 values, one per 32-bit lane
/// pair: the scale operand of the int32 multiply, two lanes per row.
static DUP2: [i32; 16] = {
    let mut t = [0i32; 16];
    let mut i = 0;
    while i < 16 {
        t[i] = (i / 2) as i32;
        i += 1;
    }
    t
};

/// The eight rows' six-bit scale of sub-block `j`, twice per row, as int32.
///
/// `vpdpbusd` leaves two int32 partials per row; multiplying each by the same
/// scale and adding the pair is exact, so the scale never needs the int16 stage
/// the AVX2 kernel's `vpmaddwd` uses.
#[inline(always)]
unsafe fn scale32(t: *const u8, j: usize) -> __m512i {
    let sc = unpack6(t.add(crate::kq::q4_scm() + j * crate::kq::Q4_SCM_STRIDE));
    _mm512_permutexvar_epi32(
        _mm512_loadu_si512(DUP2.as_ptr() as *const __m512i),
        _mm512_zextsi256_si512(sc),
    )
}

/// The even int32 lanes of `v + swap_adjacent(v)`, in the low eight lanes.
///
/// `sumi` carries two int32 partials per row (the two halves of a sub-block);
/// the fold wants their exact sum, one lane per row.
static EVEN: [i32; 16] = {
    let mut t = [0i32; 16];
    let mut i = 0;
    while i < 8 {
        t[i] = (2 * i) as i32;
        i += 1;
    }
    t
};

/// The exact sum of the two int32 partials of each row, in the low eight lanes.
#[inline(always)]
unsafe fn compact8(v: __m512i) -> __m256i {
    let s = _mm512_add_epi32(v, _mm512_shuffle_epi32(v, 0xB1));
    _mm512_castsi512_si256(_mm512_permutexvar_epi32(
        _mm512_loadu_si512(EVEN.as_ptr() as *const __m512i),
        s,
    ))
}

/// Eight output rows of a repacked Q4_K tile against `N` activation rows, on the
/// AVX-512 tier.
///
/// # Safety
///
/// The caller must have checked [`have_zmm512`]. The tile must hold `n_sb`
/// tiles, and the activation must carry the per-256 layout.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")]
unsafe fn q4k8_group_zmm<S: ActSrc, const N: usize>(
    tile: &[u8],
    n_sb: usize,
    src: &S,
    row0: usize,
    out: &mut [f32],
) {
    assert!((1..=ACT_GROUP).contains(&N), "one to four activation rows");
    assert!(out.len() >= N * ROWS, "the output holds eight rows per row");
    let lownib = _mm512_set1_epi8(0x0F);
    let mut facc = [_mm256_setzero_ps(); N];
    for sb in 0..n_sb {
        let t = tile.as_ptr().add(sb * Q4K8::SIZE);
        // The tile's rows are in natural order, so no lane permutation is needed.
        let dv = _mm256_cvtph_ps(_mm_loadu_si128(t.add(crate::kq::q4_d()) as *const __m128i));
        let dminv = _mm256_cvtph_ps(_mm_loadu_si128(
            t.add(crate::kq::q4_dmin()) as *const __m128i
        ));
        let mut sumi = [_mm512_setzero_si512(); N];
        let mut mt = [_mm256_setzero_si256(); N];
        // One window feeds two sub-blocks: its low nibbles are the weights of
        // sub-block 2g and its high nibbles sub-block 2g+1.
        for g in 0..WINDOWS / 4 {
            let mut lo32 = [_mm512_setzero_si512(); N];
            let mut hi32 = [_mm512_setzero_si512(); N];
            for k in 0..4 {
                let w = g * 4 + k;
                let v = _mm512_loadu_si512(t.add(crate::kq::q4_qs() + w * 64) as *const __m512i);
                let vlo = _mm512_and_si512(v, lownib);
                let vhi = _mm512_and_si512(_mm512_srli_epi16(v, 4), lownib);
                for u in 0..N {
                    let q = src.quads(row0 + u, sb, w);
                    // The low nibbles pair with the activation's first eight
                    // codes of the window, the high nibbles with the second
                    // eight (32 weights further along the row). VNNI's byte dot
                    // product accumulates straight into int32: no int16 stage,
                    // no saturation, and four products per lane per window.
                    lo32[u] = _mm512_dpbusd_epi32(lo32[u], vlo, bcast8(q[0]));
                    hi32[u] = _mm512_dpbusd_epi32(hi32[u], vhi, bcast8(q[2]));
                }
            }
            // The scale multiply, on the int32 partials: each of a row's two
            // partials is multiplied by the same six-bit scale and added, which
            // is exact. The int32 multiply runs on a different port from the
            // byte dot product, so it does not compete with it.
            let svlo = scale32(t, 2 * g);
            let svhi = scale32(t, 2 * g + 1);
            for u in 0..N {
                sumi[u] = _mm512_add_epi32(sumi[u], _mm512_mullo_epi32(lo32[u], svlo));
                sumi[u] = _mm512_add_epi32(sumi[u], _mm512_mullo_epi32(hi32[u], svhi));
            }
        }
        // The min term: the rows' six-bit mins against the activation's
        // per-sub-block sums. Every product is at most 63 · 4064 and every
        // partial sum at most 2 048 256, so the int32 chain and the float chain
        // the fold runs round to the same bits.
        for j in 0..SUBS {
            let mv = unpack6(t.add(crate::kq::q4_scm() + j * crate::kq::Q4_SCM_STRIDE + 6));
            for u in 0..N {
                mt[u] = _mm256_add_epi32(
                    mt[u],
                    _mm256_mullo_epi32(mv, _mm256_set1_epi32(src.min_sum(row0 + u, sb, j))),
                );
            }
        }
        for u in 0..N {
            facc[u] = fold(
                facc[u],
                compact8(sumi[u]),
                _mm256_cvtepi32_ps(mt[u]),
                dv,
                dminv,
                src.scale(row0 + u, sb),
            );
        }
    }
    // The eight lanes are the eight output rows, in the tile's own order.
    let mut lane = [0f32; ROWS];
    for u in 0..N {
        _mm256_storeu_ps(lane.as_mut_ptr(), facc[u]);
        out[(row0 + u) * ROWS..(row0 + u) * ROWS + ROWS].copy_from_slice(&lane);
    }
}

/// Eight output rows of a repacked Q4_K tile against a batch of activations, on
/// the AVX-512 tier. Same groups and tail handling as [`q4k8_gemm`].
///
/// # Safety
///
/// The caller must have checked [`have_zmm512`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")]
pub unsafe fn q4k8_gemm_zmm(tile: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    let src = Interleaved { acts };
    let groups = nb.div_ceil(ACT_GROUP);
    for g in 0..groups {
        let b0 = g * ACT_GROUP;
        let n = (nb - b0).min(ACT_GROUP);
        match n {
            4 => q4k8_group_zmm::<_, 4>(tile, n_sb, &src, b0, out),
            3 => q4k8_group_zmm::<_, 3>(tile, n_sb, &src, b0, out),
            2 => q4k8_group_zmm::<_, 2>(tile, n_sb, &src, b0, out),
            1 => q4k8_group_zmm::<_, 1>(tile, n_sb, &src, b0, out),
            0 => break,
            _ => unreachable!("a group holds four activation rows"),
        }
    }
}

/// Eight output rows of a repacked Q4_K tile against one to three activations,
/// on the AVX-512 tier. Same plain activation and the same fold as the GEMM.
///
/// # Safety
///
/// The caller must have checked [`have_zmm512`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")]
pub unsafe fn q4k8_gemv_zmm(
    tile: &[u8],
    n_sb: usize,
    acts: &[QuantAct],
    nb: usize,
    out: &mut [f32],
) {
    assert!((1..=3).contains(&nb), "the GEMV takes one to three rows");
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    let src = Plain { acts };
    match nb {
        1 => q4k8_group_zmm::<_, 1>(tile, n_sb, &src, 0, out),
        2 => q4k8_group_zmm::<_, 2>(tile, n_sb, &src, 0, out),
        _ => q4k8_group_zmm::<_, 3>(tile, n_sb, &src, 0, out),
    }
}

// ---------------------------------------------------------------------------
// The repacked Q5_0 kernels
// ---------------------------------------------------------------------------

/// The lane order `vphaddd` leaves the eight row sums in.
///
/// `vphaddd(s_lo, s_hi)` sums the two int32 lanes of each row: lanes 0..3 are
/// rows 0, 1 of the low half and 0, 1 of the high half, lanes 4..7 the other
/// four. `Q5_LANES[k]` is the natural row of sum lane `k`.
#[cfg(feature = "fmt-legacy")]
static Q5_LANES: [i32; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

/// The bit each of a window's eight fifth-bits is tested against.
#[cfg(feature = "fmt-legacy")]
static Q5_BITS: [u8; 32] = {
    let mut t = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        t[i] = 1u8 << (i % 8);
        i += 1;
    }
    t
};

/// The shuffle that spreads the low four rows' fifth-bit bytes across a vector.
///
/// Output byte `r * 8 + k` takes source byte `r`, so the row's eight-bit mask is
/// repeated eight times and the `Q5_BITS` test below reads one bit per byte.
#[cfg(feature = "fmt-legacy")]
static Q5_SPREAD_LO: [u8; 32] = {
    let mut t = [0u8; 32];
    let mut r = 0;
    while r < 4 {
        let mut k = 0;
        while k < 8 {
            t[r * 8 + k] = r as u8;
            k += 1;
        }
        r += 1;
    }
    t
};

/// The same for the high four rows.
#[cfg(feature = "fmt-legacy")]
static Q5_SPREAD_HI: [u8; 32] = {
    let mut t = [0u8; 32];
    let mut r = 0;
    while r < 4 {
        let mut k = 0;
        while k < 8 {
            t[r * 8 + k] = (r + 4) as u8;
            k += 1;
        }
        r += 1;
    }
    t
};

/// Eight output rows of a repacked Q5_0 tile against `N` activation rows.
///
/// One Q5_0 block is 32 weights, so the eight-row tile is one block per row: the
/// weight side is four eight-weight windows, each a 32-byte vector holding four
/// rows' eight nibbles. The fifth bits are spread with two `vpshufb`s per window
/// (`Q5_SPREAD_LO`/`_HI`) and OR'd into the nibbles to make the 5-bit codes.
///
/// The eight `vpmaddubsw` of a block accumulate in int16 and the block's row
/// sums come out in int32 with one `vphaddd`; the scale is per block, so the
/// float fold happens once per block per activation, with the same expression
/// the scalar reference uses. That is what keeps a row's bits independent of the
/// batch: there is only one path, and `N` only says how many independent rows
/// ride along.
///
/// # Safety
///
/// The caller must have checked that the CPU has AVX2, FMA and F16C. The tile
/// must hold `n_blk` tiles, and the activation must carry the per-32 layout.
#[target_feature(enable = "avx2,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
unsafe fn q5_08_group<const N: usize>(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    out: &mut [f32],
) {
    assert!((1..=4).contains(&N), "one to four activation rows");
    assert!(out.len() >= N * ROWS, "the output holds eight rows per row");
    let lownib = _mm256_set1_epi8(0x0F);
    let bits = _mm256_loadu_si256(Q5_BITS.as_ptr() as *const __m256i);
    let ten = _mm256_set1_epi8(0x10);
    let ctl_lo = _mm256_loadu_si256(Q5_SPREAD_LO.as_ptr() as *const __m256i);
    let ctl_hi = _mm256_loadu_si256(Q5_SPREAD_HI.as_ptr() as *const __m256i);
    let lanes = _mm256_loadu_si256(Q5_LANES.as_ptr() as *const __m256i);
    let ones = _mm256_set1_epi16(1);
    let mut facc = [_mm256_setzero_ps(); N];
    for t in 0..n_blk {
        let b = tile.as_ptr().add(t * Q5_08::SIZE);
        // The eight scales, in the lane order the `vphaddd` leaves the sums in.
        let dv = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(
            _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(
                b.add(crate::kq::q5_d()) as *const __m128i
            ))),
            lanes,
        ));
        let mut acc_lo = [_mm256_setzero_si256(); N];
        let mut acc_hi = [_mm256_setzero_si256(); N];
        // The tile stores each of the sixteen nibble bytes once, so two
        // logical eight-weight groups share a physical window: the kernel loads
        // the window's two 32-byte halves once and reads them as low and then
        // high nibbles. Logical group `h = w + 2·half`.
        for w in 0..2 {
            let a = _mm256_loadu_si256(
                b.add(crate::kq::q5_qs() + w * crate::kq::Q5_WINDOW) as *const __m256i
            );
            let c = _mm256_loadu_si256(
                b.add(crate::kq::q5_qs() + w * crate::kq::Q5_WINDOW + 32) as *const __m256i
            );
            for half in 0..2 {
                let h = w + 2 * half;
                // The eight rows' fifth bits for this group, one byte each,
                // spread to one byte per weight per row and turned into 0x10/0.
                let mask = _mm256_set1_epi64x(
                    (b.add(crate::kq::q5_qh() + h * 8) as *const u64).read_unaligned() as i64,
                );
                let hi_lo = _mm256_and_si256(
                    _mm256_cmpeq_epi8(
                        _mm256_and_si256(_mm256_shuffle_epi8(mask, ctl_lo), bits),
                        bits,
                    ),
                    ten,
                );
                let hi_hi = _mm256_and_si256(
                    _mm256_cmpeq_epi8(
                        _mm256_and_si256(_mm256_shuffle_epi8(mask, ctl_hi), bits),
                        bits,
                    ),
                    ten,
                );
                let (nlo, nhi) = if half == 0 {
                    (_mm256_and_si256(a, lownib), _mm256_and_si256(c, lownib))
                } else {
                    (
                        _mm256_and_si256(_mm256_srli_epi16(a, 4), lownib),
                        _mm256_and_si256(_mm256_srli_epi16(c, 4), lownib),
                    )
                };
                let clo = _mm256_or_si256(nlo, hi_lo);
                let chi = _mm256_or_si256(nhi, hi_hi);
                for u in 0..N {
                    let av = _mm256_set1_epi64x(
                        (acts[u].q.as_ptr().add(t * crate::kq::Q5_0_WEIGHTS + h * 8) as *const u64)
                            .read_unaligned() as i64,
                    );
                    // Four groups of two products per int16 lane: 8 · 31 · 127 =
                    // 31 496, inside int16 (`the_q5_0_int16_stage_has_headroom`).
                    acc_lo[u] = _mm256_add_epi16(acc_lo[u], _mm256_maddubs_epi16(clo, av));
                    acc_hi[u] = _mm256_add_epi16(acc_hi[u], _mm256_maddubs_epi16(chi, av));
                }
            }
        }
        for u in 0..N {
            let s = _mm256_hadd_epi32(
                _mm256_madd_epi16(acc_lo[u], ones),
                _mm256_madd_epi16(acc_hi[u], ones),
            );
            // The bias identity: codes are 0..31 with offset 16, so subtract
            // 16·Σq once per block, exactly, in int32.
            let sum = _mm256_set1_epi32(*acts[u].sums.get_unchecked(t));
            let x = _mm256_sub_epi32(s, _mm256_slli_epi32(sum, 4));
            let dv_dk = _mm256_mul_ps(dv, _mm256_set1_ps(*acts[u].d.get_unchecked(t)));
            facc[u] = _mm256_fmadd_ps(dv_dk, _mm256_cvtepi32_ps(x), facc[u]);
        }
    }
    let mut lane = [0f32; ROWS];
    for u in 0..N {
        _mm256_storeu_ps(lane.as_mut_ptr(), facc[u]);
        for k in 0..ROWS {
            out[u * ROWS + Q5_LANES[k] as usize] = lane[k];
        }
    }
}

/// Eight output rows of a repacked Q5_0 tile against `nb` activation rows.
///
/// The activation is the ordinary per-32 `QuantAct` layout, so there is no
/// interleave to build and the GEMV and the GEMM read the same bytes. Both use
/// `q5_08_group`, so a row's result is bit-identical at every batch width.
///
/// # Safety
///
/// The caller must have checked that the CPU has AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
pub unsafe fn q5_08_gemm(tile: &[u8], n_blk: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!(
        !acts.is_empty() && nb <= acts.len(),
        "more activations than rows"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    let mut b = 0;
    while b + 4 <= nb {
        q5_08_group::<4>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]);
        b += 4;
    }
    match nb - b {
        3 => q5_08_group::<3>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        2 => q5_08_group::<2>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        1 => q5_08_group::<1>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        _ => {}
    }
}

/// Eight output rows of a repacked Q5_0 tile against one to three activations.
///
/// The decode path. It is the batched kernel with a narrow batch — the same
/// arithmetic, so the GEMV and the GEMM agree bit for bit.
///
/// # Safety
///
/// The caller must have checked that the CPU has AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
pub unsafe fn q5_08_gemv(tile: &[u8], n_blk: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!((1..=3).contains(&nb), "the GEMV takes one to three rows");
    q5_08_gemm(tile, n_blk, acts, nb, out)
}

/// Eight output rows of a repacked Q8_0 tile against `N` activation rows.
///
/// One Q8_0 block is 32 weights, so the tile is one block per row: two 32-byte
/// vectors hold the eight rows' signed codes for a window (four rows each), one
/// eight-byte group per row. `vpmaddubsw` is unsigned × signed, so the weights go
/// in as `|w|` and the sign moves onto the activation; the two int32 partials a
/// row gets per window accumulate in int32, and the block's row sums fall out of
/// one `vphaddd` in the lane order [`Q5_LANES`] names.
///
/// The eight int32 lane sums are exact and the per-block float fold is the same
/// expression the scalar reference uses, so the GEMV and the GEMM agree bit for
/// bit and a row's bytes do not depend on the batch width.
///
/// # Safety
///
/// The caller must have checked AVX2, FMA and F16C. The tile must hold `n_blk`
/// tiles, and the activation must carry the per-32 layout.
#[target_feature(enable = "avx2,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
unsafe fn q8_08_group<const N: usize>(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    out: &mut [f32],
) {
    assert!((1..=4).contains(&N), "one to four activation rows");
    assert!(out.len() >= N * ROWS, "the output holds eight rows per row");
    let ones = _mm256_set1_epi16(1);
    let lanes = _mm256_loadu_si256(Q5_LANES.as_ptr() as *const __m256i);
    let pf = crate::simd::rows8_prefetch();
    let mut facc = [_mm256_setzero_ps(); N];
    for t in 0..n_blk {
        let b = tile.as_ptr().add(t * Q8_08::SIZE);
        crate::simd::prefetch_ahead(b, pf, 5);
        // The eight scales, in the lane order the `vphaddd` leaves the sums in.
        let dv = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(
            _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(
                b.add(crate::kq::q8_d()) as *const __m128i
            ))),
            lanes,
        ));
        let mut acc_lo = [_mm256_setzero_si256(); N];
        let mut acc_hi = [_mm256_setzero_si256(); N];
        for w in 0..crate::kq::Q8_0_WINDOWS {
            let lo = _mm256_loadu_si256(
                b.add(crate::kq::q8_qs() + w * crate::kq::Q8_WINDOW) as *const __m256i
            );
            let hi = _mm256_loadu_si256(
                b.add(crate::kq::q8_qs() + w * crate::kq::Q8_WINDOW + 32) as *const __m256i
            );
            // `VPMADDUBSW` multiplies unsigned bytes by signed ones: the weights
            // go in as |w| and their sign moves onto the activation.
            let absw_lo = _mm256_sign_epi8(lo, lo);
            let absw_hi = _mm256_sign_epi8(hi, hi);
            for u in 0..N {
                let av = _mm256_set1_epi64x(
                    (acts[u].q.as_ptr().add(t * crate::kq::Q8_0_WEIGHTS + w * 8) as *const i64)
                        .read_unaligned(),
                );
                let sa_lo = _mm256_sign_epi8(av, lo);
                let sa_hi = _mm256_sign_epi8(av, hi);
                // Two products per int16 lane, four weights per int32 partial.
                acc_lo[u] = _mm256_add_epi32(
                    acc_lo[u],
                    _mm256_madd_epi16(_mm256_maddubs_epi16(absw_lo, sa_lo), ones),
                );
                acc_hi[u] = _mm256_add_epi32(
                    acc_hi[u],
                    _mm256_madd_epi16(_mm256_maddubs_epi16(absw_hi, sa_hi), ones),
                );
            }
        }
        for u in 0..N {
            let s = _mm256_hadd_epi32(acc_lo[u], acc_hi[u]);
            let dv_dk = _mm256_mul_ps(dv, _mm256_set1_ps(*acts[u].d.get_unchecked(t)));
            facc[u] = _mm256_fmadd_ps(dv_dk, _mm256_cvtepi32_ps(s), facc[u]);
        }
    }
    let mut lane = [0f32; ROWS];
    for u in 0..N {
        _mm256_storeu_ps(lane.as_mut_ptr(), facc[u]);
        for k in 0..ROWS {
            out[u * ROWS + Q5_LANES[k] as usize] = lane[k];
        }
    }
}

/// Eight output rows of a repacked Q8_0 tile against `nb` activation rows.
///
/// The activation is the ordinary per-32 `QuantAct` layout, so there is no
/// interleave to build and the GEMV and the GEMM read the same bytes. Both use
/// `q8_08_group`, so a row's result is bit-identical at every batch width.
///
/// # Safety
///
/// The caller must have checked AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
pub unsafe fn q8_08_gemm(tile: &[u8], n_blk: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!(
        !acts.is_empty() && nb <= acts.len(),
        "more activations than rows"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    let mut b = 0;
    while b + 4 <= nb {
        q8_08_group::<4>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]);
        b += 4;
    }
    match nb - b {
        3 => q8_08_group::<3>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        2 => q8_08_group::<2>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        1 => q8_08_group::<1>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        _ => {}
    }
}

/// Eight output rows of a repacked Q8_0 tile against one to three activations.
///
/// The decode path. It is the batched kernel with a narrow batch — the same
/// arithmetic, so the GEMV and the GEMM agree bit for bit.
///
/// # Safety
///
/// The caller must have checked AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
pub unsafe fn q8_08_gemv(tile: &[u8], n_blk: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!((1..=3).contains(&nb), "the GEMV takes one to three rows");
    q8_08_gemm(tile, n_blk, acts, nb, out)
}

/// The shuffle that spreads the eight rows' fifth-bit bytes across a ZMM.
///
/// Output byte `r * 8 + k` takes source byte `r` of the 128-bit lane, and every
/// lane of the broadcast holds the same eight bytes, so one control serves all
/// eight rows.
#[cfg(feature = "fmt-legacy")]
static Q5_SPREAD_ZMM: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut lane = 0;
    while lane < 4 {
        let mut k = 0;
        while k < 8 {
            t[lane * 16 + k] = (2 * lane) as u8;
            t[lane * 16 + 8 + k] = (2 * lane + 1) as u8;
            k += 1;
        }
        lane += 1;
    }
    t
};

/// Eight output rows of a repacked Q5_0 tile against `N` activation rows, on the
/// AVX-512 tier.
///
/// One Q5_0 block is 32 weights, so the eight-row tile is one block per row: the
/// weight side is two 64-byte windows, each holding eight rows' eight nibbles.
/// The fifth bits are spread with one `vpshufb` per group and OR'd into the
/// nibbles to make the 5-bit codes, and `vpdpbusd` accumulates the four groups
/// of a block straight into int32: no int16 stage, no saturation.
///
/// # Safety
///
/// The caller must have checked [`have_zmm512`]. The tile must hold `n_blk`
/// tiles, and the activation must carry the per-32 layout.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
unsafe fn q5_08_group_zmm<const N: usize>(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    out: &mut [f32],
) {
    assert!((1..=4).contains(&N), "one to four activation rows");
    assert!(out.len() >= N * ROWS, "the output holds eight rows per row");
    let lownib = _mm512_set1_epi8(0x0F);
    let ten = _mm512_set1_epi8(0x10);
    let ctl = _mm512_loadu_si512(Q5_SPREAD_ZMM.as_ptr() as *const __m512i);
    // The bit each of a group's eight weights is tested against: byte `k` of a
    // row carries weight `8·group + k`, so it is tested against `1 << k`.
    let bitmask = _mm512_set1_epi64(0x8040_2010_0804_0201u64 as i64);
    let mut facc = [_mm256_setzero_ps(); N];
    for t in 0..n_blk {
        let b = tile.as_ptr().add(t * Q5_08::SIZE);
        // The tile's rows are in natural order: no lane permutation is needed.
        let dv = _mm256_cvtph_ps(_mm_loadu_si128(b.add(crate::kq::q5_d()) as *const __m128i));
        let mut acc = [_mm512_setzero_si512(); N];
        for w in 0..2 {
            let v = _mm512_loadu_si512(
                b.add(crate::kq::q5_qs() + w * crate::kq::Q5_WINDOW) as *const __m512i
            );
            let vlo = _mm512_and_si512(v, lownib);
            let vhi = _mm512_and_si512(_mm512_srli_epi16(v, 4), lownib);
            for half in 0..2 {
                let h = w + 2 * half;
                // The group's eight fifth bits, one byte per row, spread to one
                // byte per weight per row and turned into 0x10 or 0.
                let mask = _mm512_set1_epi64(
                    (b.add(crate::kq::q5_qh() + h * 8) as *const u64).read_unaligned() as i64,
                );
                let hi = _mm512_maskz_mov_epi8(
                    _mm512_cmpeq_epi8_mask(
                        _mm512_and_si512(_mm512_shuffle_epi8(mask, ctl), bitmask),
                        bitmask,
                    ),
                    ten,
                );
                let codes = if half == 0 {
                    _mm512_or_si512(vlo, hi)
                } else {
                    _mm512_or_si512(vhi, hi)
                };
                for u in 0..N {
                    let av =
                        bcast8(acts[u].q.as_ptr().add(t * crate::kq::Q5_0_WEIGHTS + h * 8)
                            as *const u8);
                    acc[u] = _mm512_dpbusd_epi32(acc[u], codes, av);
                }
            }
        }
        for u in 0..N {
            // The bias identity: codes are 0..31 with offset 16, so subtract
            // 16·Σq once per block, exactly, in int32.
            let s = compact8(acc[u]);
            let sum = _mm256_set1_epi32(*acts[u].sums.get_unchecked(t));
            let x = _mm256_sub_epi32(s, _mm256_slli_epi32(sum, 4));
            let dv_dk = _mm256_mul_ps(dv, _mm256_set1_ps(*acts[u].d.get_unchecked(t)));
            facc[u] = _mm256_fmadd_ps(dv_dk, _mm256_cvtepi32_ps(x), facc[u]);
        }
    }
    let mut lane = [0f32; ROWS];
    for u in 0..N {
        _mm256_storeu_ps(lane.as_mut_ptr(), facc[u]);
        out[u * ROWS..u * ROWS + ROWS].copy_from_slice(&lane);
    }
}

/// Eight output rows of a repacked Q5_0 tile against `nb` activation rows, on
/// the AVX-512 tier.
///
/// # Safety
///
/// The caller must have checked [`have_zmm512`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
pub unsafe fn q5_08_gemm_zmm(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    nb: usize,
    out: &mut [f32],
) {
    assert!(
        !acts.is_empty() && nb <= acts.len(),
        "more activations than rows"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    let mut b = 0;
    while b + 4 <= nb {
        q5_08_group_zmm::<4>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]);
        b += 4;
    }
    match nb - b {
        3 => q5_08_group_zmm::<3>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        2 => q5_08_group_zmm::<2>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        1 => q5_08_group_zmm::<1>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        _ => {}
    }
}

/// Eight output rows of a repacked Q5_0 tile against one to three activations,
/// on the AVX-512 tier.
///
/// # Safety
///
/// The caller must have checked [`have_zmm512`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
pub unsafe fn q5_08_gemv_zmm(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    nb: usize,
    out: &mut [f32],
) {
    assert!((1..=3).contains(&nb), "the GEMV takes one to three rows");
    q5_08_gemm_zmm(tile, n_blk, acts, nb, out)
}

/// Eight output rows of a repacked Q8_0 tile against `N` activation rows, on the
/// AVX-512 tier.
///
/// One Q8_0 block is 32 weights, so the tile is one block per row: a 64-byte
/// window holds the eight rows' eight signed codes, and `vpdpbusd` accumulates
/// four products per int32 lane straight into a zeroed accumulator. The signed
/// weights are handled by taking `|w|` as the unsigned operand and negating the
/// activation where the weight was negative; a zero weight contributes nothing
/// whatever the activation does. The two int32 partials a row gets are summed
/// exactly (`compact8`) and the per-block float fold is the AVX2 kernel's
/// expression, so the two tiers agree bit for bit.
///
/// # Safety
///
/// The caller must have checked [`have_zmm512`]. The tile must hold `n_blk`
/// tiles, and the activation must carry the per-32 layout.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
unsafe fn q8_08_group_zmm<const N: usize>(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    out: &mut [f32],
) {
    assert!((1..=4).contains(&N), "one to four activation rows");
    assert!(out.len() >= N * ROWS, "the output holds eight rows per row");
    let zero = _mm512_setzero_si512();
    let pf = crate::simd::rows8_prefetch();
    let mut facc = [_mm256_setzero_ps(); N];
    for t in 0..n_blk {
        let b = tile.as_ptr().add(t * Q8_08::SIZE);
        crate::simd::prefetch_ahead(b, pf, 5);
        // The tile's rows are in natural order: no lane permutation is needed.
        let dv = _mm256_cvtph_ps(_mm_loadu_si128(b.add(crate::kq::q8_d()) as *const __m128i));
        let mut acc = [_mm512_setzero_si512(); N];
        for w in 0..crate::kq::Q8_0_WINDOWS {
            let v = _mm512_loadu_si512(
                b.add(crate::kq::q8_qs() + w * crate::kq::Q8_WINDOW) as *const __m512i
            );
            let absw = _mm512_abs_epi8(v);
            let neg = _mm512_cmpgt_epi8_mask(zero, v);
            for u in 0..N {
                let av = bcast8(
                    acts[u].q.as_ptr().add(t * crate::kq::Q8_0_WEIGHTS + w * 8) as *const u8
                );
                let sa = _mm512_mask_sub_epi8(av, neg, zero, av);
                acc[u] = _mm512_dpbusd_epi32(acc[u], absw, sa);
            }
        }
        for u in 0..N {
            let s = compact8(acc[u]);
            let dv_dk = _mm256_mul_ps(dv, _mm256_set1_ps(*acts[u].d.get_unchecked(t)));
            facc[u] = _mm256_fmadd_ps(dv_dk, _mm256_cvtepi32_ps(s), facc[u]);
        }
    }
    let mut lane = [0f32; ROWS];
    for u in 0..N {
        _mm256_storeu_ps(lane.as_mut_ptr(), facc[u]);
        out[u * ROWS..u * ROWS + ROWS].copy_from_slice(&lane);
    }
}

/// Eight output rows of a repacked Q8_0 tile against `nb` activation rows, on the
/// AVX-512 tier.
///
/// # Safety
///
/// The caller must have checked [`have_zmm512`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
pub unsafe fn q8_08_gemm_zmm(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    nb: usize,
    out: &mut [f32],
) {
    assert!(
        !acts.is_empty() && nb <= acts.len(),
        "more activations than rows"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    let mut b = 0;
    while b + 4 <= nb {
        q8_08_group_zmm::<4>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]);
        b += 4;
    }
    match nb - b {
        3 => q8_08_group_zmm::<3>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        2 => q8_08_group_zmm::<2>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        1 => q8_08_group_zmm::<1>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        _ => {}
    }
}

/// Eight output rows of a repacked Q8_0 tile against one to three activations,
/// on the AVX-512 tier.
///
/// # Safety
///
/// The caller must have checked [`have_zmm512`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")]
#[cfg(feature = "fmt-legacy")]
pub unsafe fn q8_08_gemv_zmm(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    nb: usize,
    out: &mut [f32],
) {
    assert!((1..=3).contains(&nb), "the GEMV takes one to three rows");
    q8_08_gemm_zmm(tile, n_blk, acts, nb, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A seeded generator, so a failure is reproducible.
    pub struct Rng(pub u64);

    impl Rng {
        pub fn next_u8(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 24) as u8
        }

        pub fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next_u8()).collect()
        }
    }

    /// One Q4_K super-block with sane scales and random codes.
    pub fn q4k_block(rng: &mut Rng) -> Vec<u8> {
        let mut b = rng.bytes(144);
        // The two f16 scales: small positive numbers, so the products stay in
        // the range the float fold can hold.
        b[0..2].copy_from_slice(&crate::quant::f32_to_f16(0.01).to_le_bytes());
        b[2..4].copy_from_slice(&crate::quant::f32_to_f16(0.005).to_le_bytes());
        // The six-bit scales and mins: the file's own packing, so the tile's
        // unpacked ones are the values the format means.
        for i in 0..12 {
            b[4 + i] = rng.next_u8() & 0x3F;
        }
        b
    }

    /// A tile of eight rows and `n_sb` super-blocks.
    pub fn q4k8_tile(rng: &mut Rng, n_sb: usize) -> Vec<u8> {
        let mut tile = vec![0u8; n_sb * Q4K8::SIZE];
        for sb in 0..n_sb {
            let rows: Vec<Vec<u8>> = (0..8).map(|_| q4k_block(rng)).collect();
            let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
            crate::kq::repack_q4k8(&refs, &mut tile[sb * Q4K8::SIZE..(sb + 1) * Q4K8::SIZE]);
        }
        tile
    }

    /// One Q5_0 block with a sane scale and random codes.
    #[cfg(feature = "fmt-legacy")]
    pub fn q5_0_block(rng: &mut Rng) -> Vec<u8> {
        let mut b = rng.bytes(crate::kq::Q5_0_BLOCK);
        // A small positive scale, so the products stay in a sane float range.
        b[0..2].copy_from_slice(&crate::quant::f32_to_f16(0.01).to_le_bytes());
        b
    }

    /// A tile of eight rows and `n_blk` Q5_0 blocks.
    #[cfg(feature = "fmt-legacy")]
    pub fn q5_08_tile(rng: &mut Rng, n_blk: usize) -> Vec<u8> {
        let mut tile = vec![0u8; n_blk * Q5_08::SIZE];
        for t in 0..n_blk {
            let rows: Vec<Vec<u8>> = (0..8).map(|_| q5_0_block(rng)).collect();
            let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
            crate::kq::repack_q5_08(&refs, &mut tile[t * Q5_08::SIZE..(t + 1) * Q5_08::SIZE]);
        }
        tile
    }

    /// One Q8_0 block with a sane scale and random signed codes.
    #[cfg(feature = "fmt-legacy")]
    pub fn q8_0_block(rng: &mut Rng) -> Vec<u8> {
        let mut b = rng.bytes(crate::kq::Q8_0_BLOCK);
        // A small positive scale, so the products stay in a sane float range.
        b[0..2].copy_from_slice(&crate::quant::f32_to_f16(0.01).to_le_bytes());
        b
    }

    /// A tile of eight rows and `n_blk` Q8_0 blocks.
    #[cfg(feature = "fmt-legacy")]
    pub fn q8_08_tile(rng: &mut Rng, n_blk: usize) -> Vec<u8> {
        let mut tile = vec![0u8; n_blk * Q8_08::SIZE];
        for t in 0..n_blk {
            let rows: Vec<Vec<u8>> = (0..8).map(|_| q8_0_block(rng)).collect();
            let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
            crate::kq::repack_q8_08(&refs, &mut tile[t * Q8_08::SIZE..(t + 1) * Q8_08::SIZE]);
        }
        tile
    }

    /// An activation of `rows` rows and `n` weights.
    pub fn acts(rng: &mut Rng, rows: usize, n: usize) -> Vec<QuantAct> {
        (0..rows)
            .map(|_| {
                let x: Vec<f32> = (0..n)
                    .map(|_| (rng.next_u8() as f32 / 127.0 - 1.0) * 3.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n);
                a.fill(&x);
                a
            })
            .collect()
    }

    /// The distance between two floats, in units in the last place.
    pub fn ulps(a: f32, b: f32) -> i64 {
        if a == b {
            return 0;
        }
        if a.is_nan() || b.is_nan() {
            return i64::MAX;
        }
        let key = |x: f32| -> i64 {
            let bits = x.to_bits() as i64;
            if bits < 0 {
                i64::MIN - bits
            } else {
                bits
            }
        };
        (key(a) - key(b)).abs()
    }

    /// The int16 stage's bound: eight `vpmaddubsw` reach an int16 lane before
    /// the scale madd, i.e. sixteen products of a 15 code and a 127 activation.
    ///
    /// The doc comment above `q4k8_group` rests on this: if the bound were over
    /// int16's range, `vpaddw` would wrap and the kernel would be wrong for some
    /// inputs.
    #[test]
    fn the_int16_stage_has_headroom() {
        let bound = 16 * 15 * 127;
        assert!(bound <= i16::MAX as i32, "the int16 lanes would wrap");
        assert!(-bound >= i16::MIN as i32);
        if !crate::simd::have_batched() {
            return;
        }
        unsafe {
            // The weights at their largest code (15) against the activation at
            // the largest magnitude in both directions, eight madds deep.
            let codes = _mm256_set1_epi8(0x0F);
            for a in [127i8, -127] {
                let acts = _mm256_set1_epi8(a);
                let one = _mm256_maddubs_epi16(codes, acts);
                let mut acc = _mm256_setzero_si256();
                for _ in 0..8 {
                    acc = _mm256_add_epi16(acc, one);
                }
                let mut lanes = [0i16; 16];
                _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
                let want = 16 * 15 * a as i32;
                assert!(
                    lanes.iter().all(|&l| l as i32 == want),
                    "the int16 lane is {lanes:?}, not {want}"
                );
            }
        }
    }

    /// The min term is the same value in int32 and in the float chain the
    /// kernel used to run: every product and partial sum is an integer below
    /// 2^24, so neither rounds.
    #[test]
    fn the_min_term_is_exact_in_int32_and_in_float() {
        let mut rng = Rng(0x00D1_5EA5_E000_0001);
        for _ in 0..2000 {
            let mut int = 0i32;
            let mut chain = 0.0f32;
            for _ in 0..8 {
                // A six-bit min and a sub-block sum of a 32-code run (at most
                // 4064 in magnitude).
                let m = (rng.next_u8() & 0x3F) as i32;
                let s = (rng.next_u8() as i32 - 128) * 32;
                int += m * s;
                // `mul_add` is the kernel's `vfmadd` step, in order.
                chain = (m as f32).mul_add(s as f32, chain);
            }
            assert!(int.abs() <= 1 << 24, "the min term left the exact range");
            assert_eq!(
                int as f32, chain,
                "the float chain rounds where the int32 does not"
            );
        }
    }

    /// Every batch width, against the scalar reference: one activation row, a
    /// group of four, and the narrow group a batch whose width is not a multiple
    /// of four ends with — and every width of activation, which is the
    /// super-block count the store can hold.
    #[test]
    fn every_batch_width_agrees_with_the_reference() {
        if !crate::simd::have_batched() {
            return;
        }
        for n_sb in [1usize, 3, 4, 10, 32] {
            let n = n_sb * 256;
            let mut rng = Rng(0x5EED_1234_ABCD_9876 ^ n_sb as u64);
            let tile = q4k8_tile(&mut rng, n_sb);
            let batch = acts(&mut rng, 8, n);
            for nb in 1..=8usize {
                // The interleaved layout is what a batch of four or more uses;
                // a narrower batch reads the plain one.
                let mut kq = KqActs::default();
                kq.fill(&batch[..nb]);
                let mut got = vec![0.0f32; nb * 8];
                unsafe {
                    if nb < 4 {
                        q4k8_gemv(&tile, n_sb, &batch[..nb], nb, &mut got)
                    } else {
                        q4k8_gemm(&tile, n_sb, &kq, nb, &mut got)
                    }
                }
                for b in 0..nb {
                    for r in 0..8 {
                        let want = crate::kq::dot_q4k8_ref(&tile, n_sb, r, &batch[b]);
                        assert!(
                            ulps(got[b * 8 + r], want) <= 1,
                            "n_sb {n_sb} nb {nb}: row {r} of batch {b} is {} not {want}",
                            got[b * 8 + r]
                        );
                    }
                }
            }
        }
    }

    /// The repacked GEMM against the scalar reference, and the GEMV against the
    /// GEMM: the two paths have to agree bit for bit, because a row's result
    /// must not depend on the batch it ran in.
    #[test]
    fn the_repacked_kernels_match_the_reference_and_each_other() {
        if !crate::simd::have_batched() {
            return;
        }
        let n_sb = 4;
        let n = n_sb * 256;
        let mut rng = Rng(0x5EED_1234_ABCD_9876);
        let tile = q4k8_tile(&mut rng, n_sb);

        // The batch: four rows for the GEMM, and the same rows one at a time
        // for the GEMV.
        let batch = acts(&mut rng, 4, n);
        let mut kq = KqActs::default();
        kq.fill(&batch);
        let mut gemm = vec![0.0f32; 4 * 8];
        unsafe { q4k8_gemm(&tile, n_sb, &kq, 4, &mut gemm) };
        for b in 0..4 {
            for r in 0..8 {
                let want = crate::kq::dot_q4k8_ref(&tile, n_sb, r, &batch[b]);
                let got = gemm[b * 8 + r];
                assert!(
                    ulps(got, want) <= 1,
                    "row {r} of the batch {b}: {got} against {want}"
                );
            }
        }

        // The GEMV, one to three rows at a time.
        for nb in 1..=3 {
            let mut gemv = vec![0.0f32; 3 * 8];
            unsafe { q4k8_gemv(&tile, n_sb, &batch[..nb], nb, &mut gemv) };
            for b in 0..nb {
                for r in 0..8 {
                    assert_eq!(
                        gemv[b * 8 + r].to_bits(),
                        gemm[b * 8 + r].to_bits(),
                        "the GEMV and the GEMM disagree on row {r} of the batch {b}"
                    );
                }
            }
        }
    }

    /// The interleaved activation holds the codes the plain layout holds, four
    /// rows side by side.
    #[test]
    fn the_interleaved_activation_holds_the_same_codes() {
        let n = 512;
        let mut rng = Rng(0x0BAD_C0DE_1234_5678);
        let batch = acts(&mut rng, 6, n);
        let mut kq = KqActs::default();
        kq.fill(&batch);
        assert_eq!(kq.rows, 6);
        assert_eq!(kq.n, n);
        // Six rows: two groups, the second with two rows short.
        assert_eq!(kq.groups(), 2);
        let n_sb = n / 256;
        for g in 0..2 {
            for sb in 0..n_sb {
                for j in 0..32 {
                    for t in 0..4 {
                        let b = g * 4 + t;
                        if b >= 6 {
                            continue;
                        }
                        let at = (g * n_sb + sb) * 1024 + j * 32 + t * 8;
                        let want = &batch[b].qk[sb * 256 + j * 8..sb * 256 + j * 8 + 8];
                        assert_eq!(
                            &kq.q[at..at + 8],
                            want,
                            "group {g} super-block {sb} window {j}"
                        );
                    }
                }
            }
        }
    }

    /// The AVX-512 kernel against the AVX2 one and the scalar reference: the
    /// same bits, at every batch width, because the fold is the same expression.
    #[test]
    fn the_avx512_kernel_agrees_with_the_avx2_one_bit_for_bit() {
        if !crate::simd::have_batched() || !have_zmm512() {
            return;
        }
        for n_sb in [1usize, 3, 8, 32] {
            let n = n_sb * 256;
            let mut rng = Rng(0x5EED_512A_1234_0001 ^ n_sb as u64);
            let tile = q4k8_tile(&mut rng, n_sb);
            let batch = acts(&mut rng, 20, n);
            let mut kq = KqActs::default();
            kq.fill(&batch);
            for nb in 1..=20usize {
                let mut avx2 = vec![0.0f32; nb * 8];
                let mut zmm = vec![0.0f32; nb * 8];
                unsafe {
                    if nb < 4 {
                        q4k8_gemv(&tile, n_sb, &batch[..nb], nb, &mut avx2);
                        q4k8_gemv_zmm(&tile, n_sb, &batch[..nb], nb, &mut zmm);
                    } else {
                        q4k8_gemm(&tile, n_sb, &kq, nb, &mut avx2);
                        q4k8_gemm_zmm(&tile, n_sb, &kq, nb, &mut zmm);
                    }
                }
                for i in 0..nb * 8 {
                    assert_eq!(
                        avx2[i].to_bits(),
                        zmm[i].to_bits(),
                        "n_sb {n_sb} nb {nb} element {i}: {} against {}",
                        zmm[i],
                        avx2[i]
                    );
                }
            }
        }
    }

    /// A mid-batch start: rows 5..9 as their own batch agree with the wide run,
    /// which is what the dynamic row claiming relies on.
    #[test]
    fn the_avx512_kernel_has_no_cross_row_state() {
        if !crate::simd::have_batched() || !have_zmm512() {
            return;
        }
        let n_sb = 4;
        let n = n_sb * 256;
        let mut rng = Rng(0x5EED_512A_ABCD_0002);
        let tile = q4k8_tile(&mut rng, n_sb);
        let batch = acts(&mut rng, 12, n);
        let mut kq = KqActs::default();
        kq.fill(&batch);
        let mut wide = vec![0.0f32; 12 * 8];
        unsafe { q4k8_gemm_zmm(&tile, n_sb, &kq, 12, &mut wide) };
        for (start, nb) in [(0usize, 4usize), (5, 4), (7, 5), (9, 3)] {
            // A worker's range can begin inside a batch: the caller rebuilds the
            // interleaved buffer from its slice (the GEMV reads the plain rows
            // directly).
            let mut sub = KqActs::default();
            sub.fill(&batch[start..]);
            let mut part = vec![0.0f32; nb * 8];
            unsafe {
                if nb < 4 {
                    q4k8_gemv_zmm(&tile, n_sb, &batch[start..], nb, &mut part)
                } else {
                    q4k8_gemm_zmm(&tile, n_sb, &sub, nb, &mut part)
                }
            }
            for b in 0..nb {
                for r in 0..8 {
                    assert_eq!(
                        part[b * 8 + r].to_bits(),
                        wide[(b + start) * 8 + r].to_bits(),
                        "start {start} nb {nb}: row {r} of batch {b}"
                    );
                }
            }
        }
    }

    /// The AVX-512 int32 stage's bound: with `vpdpbusd` the partial sums are
    /// int32, so the only question is whether a lane can overflow.
    ///
    /// A sub-block's int32 lane holds sixteen products of a 15 code and a 127
    /// activation (four windows, four products each): 16 · 15 · 127 = 30 480,
    /// times the six-bit scale 63 gives 1 920 240, and the eight sub-blocks of a
    /// super-block sum to 15 361 920 — 0.7 % of int32's range. No saturation is
    /// possible, unlike the int16 stage the AVX2 kernel needs.
    #[test]
    fn the_avx512_int32_stage_has_headroom() {
        let lane = 16 * 15 * 127;
        let scaled = lane * 63;
        let super_block = 8 * scaled;
        assert!(super_block < i32::MAX as i64, "the int32 lane would wrap");
        // The min term: 8 sub-blocks of 63 · 4064.
        assert!(8 * 63 * 4064 < i32::MAX as i64);
        // And the float conversion the fold does is exact for the int32 sums.
        assert!(super_block < 1 << 24, "the fold would round an integer");
        if !crate::simd::have_batched() || !have_zmm512() {
            return;
        }
        unsafe {
            // The largest code against the largest activation, sixteen products
            // deep, in both directions.
            let codes = _mm512_set1_epi8(0x0F);
            for a in [127i8, -127] {
                let av = _mm512_set1_epi8(a);
                let one = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes, av);
                let mut acc = _mm512_setzero_si512();
                for _ in 0..4 {
                    acc = _mm512_add_epi32(acc, one);
                }
                let mut lanes = [0i32; 16];
                _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, acc);
                let want = 16 * 15 * a as i32;
                assert!(
                    lanes.iter().all(|&l| l == want),
                    "the int32 lane is {lanes:?}, not {want}"
                );
            }
        }
    }

    /// The AVX-512 Q5_0 kernel against the AVX2 one and the reference: the same
    /// bits, at every batch width.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_avx512_q5_0_kernel_agrees_with_the_avx2_one_bit_for_bit() {
        if !crate::simd::have_batched() || !have_zmm512() {
            return;
        }
        for n_blk in [1usize, 3, 4, 10, 28] {
            let n = n_blk * 32;
            let mut rng = Rng(0x5EED_512A_5A5A_0003 ^ n_blk as u64);
            let tile = q5_08_tile(&mut rng, n_blk);
            let batch = acts(&mut rng, 20, n);
            for nb in 1..=20usize {
                let mut avx2 = vec![0.0f32; nb * 8];
                let mut zmm = vec![0.0f32; nb * 8];
                unsafe {
                    if nb < 4 {
                        q5_08_gemv(&tile, n_blk, &batch[..nb], nb, &mut avx2);
                        q5_08_gemv_zmm(&tile, n_blk, &batch[..nb], nb, &mut zmm);
                    } else {
                        q5_08_gemm(&tile, n_blk, &batch[..nb], nb, &mut avx2);
                        q5_08_gemm_zmm(&tile, n_blk, &batch[..nb], nb, &mut zmm);
                    }
                }
                for b in 0..nb {
                    for r in 0..8 {
                        assert_eq!(
                            avx2[b * 8 + r].to_bits(),
                            zmm[b * 8 + r].to_bits(),
                            "n_blk {n_blk} nb {nb}: row {r} of batch {b} is {} not {}",
                            zmm[b * 8 + r],
                            avx2[b * 8 + r]
                        );
                        let want = crate::kq::dot_q5_08_ref(&tile, n_blk, r, &batch[b]);
                        assert!(
                            ulps(zmm[b * 8 + r], want) <= 1,
                            "the reference disagrees: {} against {want}",
                            zmm[b * 8 + r]
                        );
                    }
                }
            }
        }
    }

    /// The AVX-512 Q5_0 kernel has no cross-row state: a batch that starts in
    /// the middle of a group agrees with the wide run.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_avx512_q5_0_kernel_has_no_cross_row_state() {
        if !crate::simd::have_batched() || !have_zmm512() {
            return;
        }
        let n_blk = 28;
        let n = n_blk * 32;
        let mut rng = Rng(0x5EED_512A_5A5A_0004);
        let tile = q5_08_tile(&mut rng, n_blk);
        let batch = acts(&mut rng, 12, n);
        let mut wide = vec![0.0f32; 12 * 8];
        unsafe { q5_08_gemm_zmm(&tile, n_blk, &batch, 12, &mut wide) };
        for (start, nb) in [(0usize, 4usize), (5, 4), (7, 5), (9, 3)] {
            let mut part = vec![0.0f32; nb * 8];
            unsafe {
                if nb < 4 {
                    q5_08_gemv_zmm(&tile, n_blk, &batch[start..], nb, &mut part)
                } else {
                    q5_08_gemm_zmm(&tile, n_blk, &batch[start..], nb, &mut part)
                }
            }
            for b in 0..nb {
                for r in 0..8 {
                    assert_eq!(
                        part[b * 8 + r].to_bits(),
                        wide[(b + start) * 8 + r].to_bits(),
                        "start {start} nb {nb}: row {r} of batch {b}"
                    );
                }
            }
        }
    }

    /// The AVX-512 Q5_0 int32 stage's bound: four byte products per lane per
    /// block, and a block is 32 weights — at most 32 · 31 · 127 = 125 984, and
    /// the code is at most 31, so nothing can overflow.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_avx512_q5_0_int32_stage_has_headroom() {
        let bound = 32 * 31 * 127;
        assert!(bound < i32::MAX as i64);
        assert!(bound < 1 << 24, "the fold would round an integer");
        if !crate::simd::have_batched() || !have_zmm512() {
            return;
        }
        unsafe {
            let codes = _mm512_set1_epi8(31);
            for a in [127i8, -127] {
                let av = _mm512_set1_epi8(a);
                let one = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes, av);
                let mut acc = _mm512_setzero_si512();
                for _ in 0..4 {
                    acc = _mm512_add_epi32(acc, one);
                }
                let mut lanes = [0i32; 16];
                _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, acc);
                let want = 16 * 31 * a as i32;
                assert!(
                    lanes.iter().all(|&l| l == want),
                    "the int32 lane is {lanes:?}, not {want}"
                );
            }
        }
    }

    #[test]
    fn the_transpose_control_takes_two_rows_out_of_a_lane() {
        // The lane holds two rows' eight bytes; the control brings the first
        // four bytes of each row to the front. The control is checked here as
        // the bytes it is, because the kernel's correctness rests on it.
        for k in 0..4 {
            assert_eq!(TRANSPOSE[k], k as u8, "the first row's byte {k}");
            assert_eq!(TRANSPOSE[4 + k], (8 + k) as u8, "the second row's byte {k}");
            assert_eq!(
                TRANSPOSE[8 + k],
                (4 + k) as u8,
                "the first row's byte {}",
                4 + k
            );
            assert_eq!(
                TRANSPOSE[12 + k],
                (12 + k) as u8,
                "the second row's byte {}",
                4 + k
            );
        }
        // The high lane's control is the same: each lane holds two rows in the
        // same shape.
        assert_eq!(&TRANSPOSE[0..16], &TRANSPOSE[16..32]);
        // The six-bit unpack reads byte `6i/8` and its neighbour, and shifts by
        // `6i % 8`: the field never crosses more than two bytes.
        for i in 0..8 {
            let k = 6 * i / 8;
            assert_eq!(PAIRS[4 * i], k as u8, "the low byte of value {i}");
            assert_eq!(
                PAIRS[4 * i + 1],
                (k + 1) as u8,
                "the high byte of value {i}"
            );
            assert_eq!(SHIFTS[i], (6 * i % 8) as i32, "the shift of value {i}");
        }
    }

    /// The packed scales round-trip through the kernel's unpack exactly.
    #[test]
    fn the_packed_scales_unpack_to_the_values_the_tile_holds() {
        if !crate::simd::have_batched() {
            return;
        }
        let mut rng = Rng(0x005C_A1E5_0000_0001);
        for _ in 0..64 {
            let mut tile = vec![0u8; Q4K8::SIZE];
            // Every six bits of every field a different value, so a shift of one
            // bit anywhere shows up.
            for j in 0..SUBS {
                let mut sc = 0u64;
                let mut m = 0u64;
                for r in 0..8 {
                    sc |= ((rng.next_u8() & 63) as u64) << (6 * r);
                    m |= ((rng.next_u8() & 63) as u64) << (6 * r);
                }
                let at = crate::kq::q4_scm() + j * crate::kq::Q4_SCM_STRIDE;
                tile[at..at + 6].copy_from_slice(&sc.to_le_bytes()[..6]);
                tile[at + 6..at + 12].copy_from_slice(&m.to_le_bytes()[..6]);
                for r in 0..8 {
                    assert_eq!(
                        crate::kq::tile_q4k_scale(&tile, j, r) as u64,
                        (sc >> (6 * r)) & 63
                    );
                    assert_eq!(
                        crate::kq::tile_q4k_min(&tile, j, r) as u64,
                        (m >> (6 * r)) & 63
                    );
                }
            }
            let (sv, mv) = unsafe { unpack_scales(tile.as_ptr()) };
            for j in 0..SUBS {
                let mut sl = [0i16; 16];
                let mut ml = [0i32; 8];
                unsafe {
                    _mm256_storeu_si256(sl.as_mut_ptr() as *mut __m256i, sv[j]);
                    _mm256_storeu_si256(ml.as_mut_ptr() as *mut __m256i, mv[j]);
                }
                for k in 0..8 {
                    let want = crate::kq::tile_q4k_scale(&tile, j, LANES[k]) as i16;
                    assert_eq!(sl[2 * k], want, "scale of lane {k} of sub-block {j}");
                    assert_eq!(sl[2 * k + 1], want, "the scale twice");
                    assert_eq!(
                        ml[k],
                        crate::kq::tile_q4k_min(&tile, j, LANES[k]) as i32,
                        "min of lane {k} of sub-block {j}"
                    );
                }
            }
        }
    }

    /// The Q5_0 int16 stage's bound: four `vpmaddubsw` reach an int16 lane
    /// before the widening, i.e. eight products of a 31 code and a 127
    /// activation.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_q5_0_int16_stage_has_headroom() {
        let bound = 8 * 31 * 127;
        assert!(bound <= i16::MAX as i32, "the int16 lanes would wrap");
        assert!(-bound >= i16::MIN as i32);
        if !crate::simd::have_batched() {
            return;
        }
        unsafe {
            // The weights at their largest code (31) against the activation at
            // the largest magnitude in both directions, four madds deep.
            let codes = _mm256_set1_epi8(31);
            for a in [127i8, -127] {
                let av = _mm256_set1_epi8(a);
                let one = _mm256_maddubs_epi16(codes, av);
                let mut acc = _mm256_setzero_si256();
                for _ in 0..4 {
                    acc = _mm256_add_epi16(acc, one);
                }
                let mut lanes = [0i16; 16];
                _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
                let want = 8 * 31 * a as i32;
                assert!(
                    lanes.iter().all(|&l| l as i32 == want),
                    "the int16 lane is {lanes:?}, not {want}"
                );
            }
        }
    }

    /// Every batch width, against the scalar reference — and every width of
    /// activation, so the block count the store can hold is covered too.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn every_q5_0_batch_width_agrees_with_the_reference() {
        if !crate::simd::have_batched() {
            return;
        }
        for n_blk in [1usize, 3, 4, 10, 28, 32] {
            let n = n_blk * 32;
            let mut rng = Rng(0x5EED_5A5A_1234_9876 ^ n_blk as u64);
            let tile = q5_08_tile(&mut rng, n_blk);
            let batch = acts(&mut rng, 20, n);
            for nb in 1..=20usize {
                let mut got = vec![0.0f32; nb * 8];
                unsafe {
                    if nb < 4 {
                        q5_08_gemv(&tile, n_blk, &batch[..nb], nb, &mut got)
                    } else {
                        q5_08_gemm(&tile, n_blk, &batch[..nb], nb, &mut got)
                    }
                }
                for b in 0..nb {
                    for r in 0..8 {
                        let want = crate::kq::dot_q5_08_ref(&tile, n_blk, r, &batch[b]);
                        assert!(
                            ulps(got[b * 8 + r], want) <= 1,
                            "n_blk {n_blk} nb {nb}: row {r} of batch {b} is {} not {want}",
                            got[b * 8 + r]
                        );
                    }
                }
            }
        }
    }

    /// The GEMV and the GEMM agree bit for bit, including a row that starts in
    /// the middle of a group (the linear layer computes whole tiles, so a
    /// worker's range can begin inside one).
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_q5_0_gemv_and_gemm_agree_bit_for_bit() {
        if !crate::simd::have_batched() {
            return;
        }
        let n_blk = 28;
        let n = n_blk * 32;
        let mut rng = Rng(0x5EED_5A5A_ABCD_0001);
        let tile = q5_08_tile(&mut rng, n_blk);
        let batch = acts(&mut rng, 12, n);

        let mut gemm = vec![0.0f32; 12 * 8];
        unsafe { q5_08_gemm(&tile, n_blk, &batch, 12, &mut gemm) };
        for nb in 1..=3 {
            let mut gemv = vec![0.0f32; 3 * 8];
            unsafe { q5_08_gemv(&tile, n_blk, &batch[..nb], nb, &mut gemv) };
            for b in 0..nb {
                for r in 0..8 {
                    assert_eq!(
                        gemv[b * 8 + r].to_bits(),
                        gemm[b * 8 + r].to_bits(),
                        "the GEMV and the GEMM disagree on row {r} of batch {b}"
                    );
                }
            }
        }
        // A mid-batch start: rows 5..9 as their own batch agree with the wide
        // run. The kernel has no cross-row state, so this is the property the
        // dynamic row claiming relies on.
        let mut part = vec![0.0f32; 4 * 8];
        unsafe { q5_08_gemm(&tile, n_blk, &batch[5..], 4, &mut part) };
        for b in 0..4 {
            for r in 0..8 {
                assert_eq!(
                    part[b * 8 + r].to_bits(),
                    gemm[(b + 5) * 8 + r].to_bits(),
                    "the mid-batch start disagrees on row {r} of batch {b}"
                );
            }
        }
    }

    /// The repacked tile, unrepacked, is the same weights as the file: the
    /// ordinary scalar kernel and the repacked one compute the same real
    /// number, to the last bit or two.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_q5_0_tile_matches_the_ordinary_scalar_kernel() {
        if !crate::simd::have_batched() {
            return;
        }
        let n_blk = 4;
        let n = n_blk * 32;
        let mut rng = Rng(0x5EED_5A5A_0DDF_0001);
        let tile = q5_08_tile(&mut rng, n_blk);
        let batch = acts(&mut rng, 2, n);
        // The file's bytes, eight rows, for the scalar kernel.
        let mut blocks = vec![0u8; 8 * n_blk * crate::kq::Q5_0_BLOCK];
        for t in 0..n_blk {
            let mut back = vec![0u8; Q5_08::SOURCE];
            crate::kq::unrepack_q5_08(&tile[t * Q5_08::SIZE..(t + 1) * Q5_08::SIZE], &mut back);
            for r in 0..8 {
                let at = (r * n_blk + t) * crate::kq::Q5_0_BLOCK;
                blocks[at..at + crate::kq::Q5_0_BLOCK]
                    .copy_from_slice(&back[r * crate::kq::Q5_0_BLOCK..][..crate::kq::Q5_0_BLOCK]);
            }
        }
        let mut got = vec![0.0f32; 2 * 8];
        unsafe { q5_08_gemm(&tile, n_blk, &batch, 2, &mut got) };
        for b in 0..2 {
            for r in 0..8 {
                let row =
                    &blocks[r * n_blk * crate::kq::Q5_0_BLOCK..][..n_blk * crate::kq::Q5_0_BLOCK];
                let want = crate::quant::dot_scalar(crate::format::Format::Q5_0, row, &batch[b]);
                // The ordinary kernel sums a non-fused `d·dk·s` per block; the
                // tile folds with an FMA. Four blocks of cancellation put the
                // two real numbers at most a handful of ULP apart.
                assert!(
                    ulps(got[b * 8 + r], want) <= 16,
                    "row {r} of batch {b}: {} against the scalar kernel's {want}",
                    got[b * 8 + r]
                );
            }
        }
    }

    /// The Q8_0 pair stage's bound: `vpmaddubsw` sums two products of an 8-bit
    /// magnitude (at most 128, the `|−128|` code) and a 127 activation, which is
    /// 32 512 — inside int16. The int32 partial then holds four such pairs, and
    /// a block's lane sum at most 32 · 128 · 127 = 520 192, below the 2^24 where
    /// the float conversion would round.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_q8_0_int16_stage_and_the_fold_have_headroom() {
        let pair = 2 * 128 * 127;
        assert!(pair <= i16::MAX as i32, "the int16 pair lane would wrap");
        let block = 32 * 128 * 127;
        assert!(block < 1 << 24, "the fold would round an integer");
        assert!(4 * pair < i32::MAX);
        if !crate::simd::have_batched() {
            return;
        }
        unsafe {
            // The weight at its largest magnitude (|−128|) against the activation
            // at the largest magnitude in both directions, four windows deep.
            let codes = _mm256_set1_epi8(-128);
            let absw = _mm256_sign_epi8(codes, codes);
            for a in [127i8, -127] {
                let av = _mm256_set1_epi8(a);
                let sa = _mm256_sign_epi8(av, codes);
                let one = _mm256_maddubs_epi16(absw, sa);
                let mut lanes = [0i16; 16];
                _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, one);
                // `sign_epi8(av, codes)` moves the weight's sign onto the
                // activation, so both activation signs give the same magnitude.
                let want = 2 * 128 * 127;
                assert!(
                    lanes.iter().all(|&l| l.unsigned_abs() as i32 == want),
                    "the int16 lane is {lanes:?}, not ±{want}"
                );
            }
        }
    }

    /// The AVX-512 Q8_0 int32 stage: four products per lane, two lanes per row,
    /// four windows per block — at most 32 · 128 · 127 = 520 192, exact in int32
    /// and in the float the fold converts it to.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_avx512_q8_0_int32_stage_has_headroom() {
        let bound = 32 * 128 * 127;
        assert!(bound < i32::MAX as i64);
        assert!(bound < 1 << 24, "the fold would round an integer");
        if !crate::simd::have_batched() || !have_zmm512() {
            return;
        }
        unsafe {
            let codes = _mm512_set1_epi8(-128);
            let absw = _mm512_abs_epi8(codes);
            for a in [127i8, -127] {
                let av = _mm512_set1_epi8(a);
                let sa = _mm512_mask_sub_epi8(
                    av,
                    _mm512_cmpgt_epi8_mask(_mm512_setzero_si512(), codes),
                    _mm512_setzero_si512(),
                    av,
                );
                let one = _mm512_dpbusd_epi32(_mm512_setzero_si512(), absw, sa);
                let mut acc = _mm512_setzero_si512();
                for _ in 0..4 {
                    acc = _mm512_add_epi32(acc, one);
                }
                let mut lanes = [0i32; 16];
                _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, acc);
                let want = 16 * 128 * a as i32;
                assert!(
                    lanes.iter().all(|&l| l == want),
                    "the int32 lane is {lanes:?}, not {want}"
                );
            }
        }
    }

    /// Every batch width, against the scalar reference — and every width of
    /// activation, so the block count the store can hold is covered too.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn every_q8_0_batch_width_agrees_with_the_reference() {
        if !crate::simd::have_batched() {
            return;
        }
        for n_blk in [1usize, 3, 4, 10, 28, 32, 36] {
            let n = n_blk * 32;
            let mut rng = Rng(0x5EED_8080_1234_9876 ^ n_blk as u64);
            let tile = q8_08_tile(&mut rng, n_blk);
            let batch = acts(&mut rng, 20, n);
            for nb in 1..=20usize {
                let mut got = vec![0.0f32; nb * 8];
                unsafe {
                    if nb < 4 {
                        q8_08_gemv(&tile, n_blk, &batch[..nb], nb, &mut got)
                    } else {
                        q8_08_gemm(&tile, n_blk, &batch[..nb], nb, &mut got)
                    }
                }
                for b in 0..nb {
                    for r in 0..8 {
                        let want = crate::kq::dot_q8_08_ref(&tile, n_blk, r, &batch[b]);
                        assert!(
                            ulps(got[b * 8 + r], want) <= 1,
                            "n_blk {n_blk} nb {nb}: row {r} of batch {b} is {} not {want}",
                            got[b * 8 + r]
                        );
                    }
                }
            }
        }
    }

    /// The GEMV and the GEMM agree bit for bit, including a row that starts in
    /// the middle of a group (the linear layer computes whole tiles, so a
    /// worker's range can begin inside one).
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_q8_0_gemv_and_gemm_agree_bit_for_bit() {
        if !crate::simd::have_batched() {
            return;
        }
        let n_blk = 36;
        let n = n_blk * 32;
        let mut rng = Rng(0x5EED_8080_ABCD_0001);
        let tile = q8_08_tile(&mut rng, n_blk);
        let batch = acts(&mut rng, 12, n);

        let mut gemm = vec![0.0f32; 12 * 8];
        unsafe { q8_08_gemm(&tile, n_blk, &batch, 12, &mut gemm) };
        for nb in 1..=3 {
            let mut gemv = vec![0.0f32; 3 * 8];
            unsafe { q8_08_gemv(&tile, n_blk, &batch[..nb], nb, &mut gemv) };
            for b in 0..nb {
                for r in 0..8 {
                    assert_eq!(
                        gemv[b * 8 + r].to_bits(),
                        gemm[b * 8 + r].to_bits(),
                        "the GEMV and the GEMM disagree on row {r} of batch {b}"
                    );
                }
            }
        }
        let mut part = vec![0.0f32; 4 * 8];
        unsafe { q8_08_gemm(&tile, n_blk, &batch[5..], 4, &mut part) };
        for b in 0..4 {
            for r in 0..8 {
                assert_eq!(
                    part[b * 8 + r].to_bits(),
                    gemm[(b + 5) * 8 + r].to_bits(),
                    "the mid-batch start disagrees on row {r} of batch {b}"
                );
            }
        }
    }

    /// The AVX-512 Q8_0 kernel against the AVX2 one and the reference: the same
    /// bits, at every batch width.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_avx512_q8_0_kernel_agrees_with_the_avx2_one_bit_for_bit() {
        if !crate::simd::have_batched() || !have_zmm512() {
            return;
        }
        for n_blk in [1usize, 3, 4, 10, 28, 36] {
            let n = n_blk * 32;
            let mut rng = Rng(0x5EED_512A_8080_0005 ^ n_blk as u64);
            let tile = q8_08_tile(&mut rng, n_blk);
            let batch = acts(&mut rng, 20, n);
            for nb in 1..=20usize {
                let mut avx2 = vec![0.0f32; nb * 8];
                let mut zmm = vec![0.0f32; nb * 8];
                unsafe {
                    if nb < 4 {
                        q8_08_gemv(&tile, n_blk, &batch[..nb], nb, &mut avx2);
                        q8_08_gemv_zmm(&tile, n_blk, &batch[..nb], nb, &mut zmm);
                    } else {
                        q8_08_gemm(&tile, n_blk, &batch[..nb], nb, &mut avx2);
                        q8_08_gemm_zmm(&tile, n_blk, &batch[..nb], nb, &mut zmm);
                    }
                }
                for b in 0..nb {
                    for r in 0..8 {
                        assert_eq!(
                            avx2[b * 8 + r].to_bits(),
                            zmm[b * 8 + r].to_bits(),
                            "n_blk {n_blk} nb {nb}: row {r} of batch {b} is {} not {}",
                            zmm[b * 8 + r],
                            avx2[b * 8 + r]
                        );
                        let want = crate::kq::dot_q8_08_ref(&tile, n_blk, r, &batch[b]);
                        assert!(
                            ulps(zmm[b * 8 + r], want) <= 1,
                            "the reference disagrees: {} against {want}",
                            zmm[b * 8 + r]
                        );
                    }
                }
            }
        }
    }

    /// The AVX-512 Q8_0 kernel has no cross-row state: a batch that starts in
    /// the middle of a group agrees with the wide run.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_avx512_q8_0_kernel_has_no_cross_row_state() {
        if !crate::simd::have_batched() || !have_zmm512() {
            return;
        }
        let n_blk = 36;
        let n = n_blk * 32;
        let mut rng = Rng(0x5EED_512A_8080_0006);
        let tile = q8_08_tile(&mut rng, n_blk);
        let batch = acts(&mut rng, 12, n);
        let mut wide = vec![0.0f32; 12 * 8];
        unsafe { q8_08_gemm_zmm(&tile, n_blk, &batch, 12, &mut wide) };
        for (start, nb) in [(0usize, 4usize), (5, 4), (7, 5), (9, 3)] {
            let mut part = vec![0.0f32; nb * 8];
            unsafe {
                if nb < 4 {
                    q8_08_gemv_zmm(&tile, n_blk, &batch[start..], nb, &mut part)
                } else {
                    q8_08_gemm_zmm(&tile, n_blk, &batch[start..], nb, &mut part)
                }
            }
            for b in 0..nb {
                for r in 0..8 {
                    assert_eq!(
                        part[b * 8 + r].to_bits(),
                        wide[(b + start) * 8 + r].to_bits(),
                        "start {start} nb {nb}: row {r} of batch {b}"
                    );
                }
            }
        }
    }

    /// The repacked tile, unrepacked, is the same weights as the file: the
    /// ordinary scalar kernel and the repacked one compute the same real
    /// number, to the last bit or two.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn the_q8_0_tile_matches_the_ordinary_scalar_kernel() {
        if !crate::simd::have_batched() {
            return;
        }
        let n_blk = 4;
        let n = n_blk * 32;
        let mut rng = Rng(0x5EED_8080_0DDF_0001);
        let tile = q8_08_tile(&mut rng, n_blk);
        let batch = acts(&mut rng, 2, n);
        // The file's bytes, eight rows, for the scalar kernel.
        let mut blocks = vec![0u8; 8 * n_blk * crate::kq::Q8_0_BLOCK];
        for t in 0..n_blk {
            let mut back = vec![0u8; Q8_08::SOURCE];
            crate::kq::unrepack_q8_08(&tile[t * Q8_08::SIZE..(t + 1) * Q8_08::SIZE], &mut back);
            for r in 0..8 {
                let at = (r * n_blk + t) * crate::kq::Q8_0_BLOCK;
                blocks[at..at + crate::kq::Q8_0_BLOCK]
                    .copy_from_slice(&back[r * crate::kq::Q8_0_BLOCK..][..crate::kq::Q8_0_BLOCK]);
            }
        }
        let mut got = vec![0.0f32; 2 * 8];
        unsafe { q8_08_gemm(&tile, n_blk, &batch, 2, &mut got) };
        for b in 0..2 {
            for r in 0..8 {
                let row =
                    &blocks[r * n_blk * crate::kq::Q8_0_BLOCK..][..n_blk * crate::kq::Q8_0_BLOCK];
                let want = crate::quant::dot_scalar(crate::format::Format::Q8_0, row, &batch[b]);
                // The ordinary kernel sums a non-fused `d·dk·s` per block; the
                // tile folds with an FMA. Four blocks of cancellation put the
                // two real numbers at most a handful of ULP apart.
                assert!(
                    ulps(got[b * 8 + r], want) <= 16,
                    "row {r} of batch {b}: {} against the scalar kernel's {want}",
                    got[b * 8 + r]
                );
            }
        }
    }
}
