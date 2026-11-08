//! The repacked Q4_K kernels (S3.4, step 2): the AVX2 GEMM and GEMV over the
//! eight-row tiles of [`crate::kq::Q4K8`].
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
//! # The int16 stage
//!
//! The int16 accumulator holds sixteen products per lane: eight `vpmaddubsw`
//! results, two products each. The bound is `16 · 15 · 127 = 30 480`, inside
//! int16's 32 767 — the same bound the design note works out, and the test
//! `the_int16_stage_has_headroom` in this file asserts it.

#![cfg(target_arch = "x86_64")]
#![cfg_attr(not(feature = "fmt-kquant"), allow(dead_code))]

use crate::kq::{KqActs, ACT_GROUP, Q4K8, Q5_08};
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

/// Eight output rows of a repacked Q4_K tile against `N` activation rows.
///
/// The tile is `n_sb` super-blocks of eight rows; `out` receives
/// `out[b · 8 + r]` for the rows `row0 .. row0 + N` of the activation.
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
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4k8_group<S: ActSrc, const N: usize>(
    tile: &[u8],
    n_sb: usize,
    src: &S,
    row0: usize,
    out: &mut [f32],
) {
    assert!((1..=ACT_GROUP).contains(&N), "one to four activation rows");
    assert!(out.len() >= N * ROWS, "the output holds eight rows per row");
    let lownib = _mm256_set1_epi8(0x0F);
    let mut facc = [_mm256_setzero_ps(); N];
    for sb in 0..n_sb {
        let t = tile.as_ptr().add(sb * Q4K8::SIZE);
        // The scales and the mins come in the tile's row order; the fold wants
        // the lane order.
        let dv = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(
            _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(
                t.add(crate::kq::q4_d()) as *const __m128i
            ))),
            _mm256_loadu_si256(LANEPERM.as_ptr() as *const __m256i),
        ));
        let dminv = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(
            _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(
                t.add(crate::kq::q4_dmin()) as *const __m128i,
            ))),
            _mm256_loadu_si256(LANEPERM.as_ptr() as *const __m256i),
        ));
        let (sv, mv) = unpack_scales(t);
        let mut sumi = [_mm256_setzero_si256(); N];
        // A window's low nibbles are the weights of sub-block 2g and its high
        // nibbles sub-block 2g+1, so one decode of the window feeds both. The
        // weights of a super-block are unpacked once, not once per sub-block.
        for g in 0..WINDOWS / 4 {
            let mut lo16 = [_mm256_setzero_si256(); N];
            let mut hi16 = [_mm256_setzero_si256(); N];
            for k in 0..4 {
                let w = g * 4 + k;
                let (g0, g1) = window_groups(t.add(crate::kq::q4_qs()), w);
                let vlo = (_mm256_and_si256(g0, lownib), _mm256_and_si256(g1, lownib));
                let vhi = (
                    _mm256_and_si256(_mm256_srli_epi16(g0, 4), lownib),
                    _mm256_and_si256(_mm256_srli_epi16(g1, 4), lownib),
                );
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
                        _mm256_maddubs_epi16(vlo.0, x1),
                        _mm256_maddubs_epi16(vlo.1, x2),
                    );
                    lo16[u] = _mm256_add_epi16(lo16[u], p);
                    let p = _mm256_add_epi16(
                        _mm256_maddubs_epi16(vhi.0, y1),
                        _mm256_maddubs_epi16(vhi.1, y2),
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
    // The eight lanes are the eight output rows, in the lane order.
    let mut lane = [0f32; ROWS];
    for u in 0..N {
        _mm256_storeu_ps(lane.as_mut_ptr(), facc[u]);
        for k in 0..ROWS {
            out[(row0 + u) * ROWS + LANES[k]] = lane[k];
        }
    }
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
    let groups = acts.groups();
    for g in 0..groups {
        let b0 = g * ACT_GROUP;
        let n = (nb - b0).min(ACT_GROUP);
        // A group of four rows is the common case; the last group of a batch
        // whose width is not a multiple of four is the only narrow one.
        match n {
            4 => q4k8_group::<_, 4>(tile, n_sb, &src, b0, out),
            3 => q4k8_group::<_, 3>(tile, n_sb, &src, b0, out),
            2 => q4k8_group::<_, 2>(tile, n_sb, &src, b0, out),
            1 => q4k8_group::<_, 1>(tile, n_sb, &src, b0, out),
            0 => break,
            _ => unreachable!("a group holds four activation rows"),
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
        1 => q4k8_group::<_, 1>(tile, n_sb, &src, 0, out),
        2 => q4k8_group::<_, 2>(tile, n_sb, &src, 0, out),
        _ => q4k8_group::<_, 3>(tile, n_sb, &src, 0, out),
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
                assert!(
                    ulps(got[b * 8 + r], want) <= 4,
                    "row {r} of batch {b}: {} against the scalar kernel's {want}",
                    got[b * 8 + r]
                );
            }
        }
    }
}
