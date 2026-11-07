//! The repacked k-quant kernels (S3.4, steps 2 and 3): the AVX2 GEMM and GEMV
//! over the eight-row tiles of [`crate::kq::Q4K8`] and [`crate::kq::Q6K8`].
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
//!
//! Q6_K's codes are six bits wide, and the same bound would be `16 · 63 · 127 =
//! 128 016` — far outside int16. The Q6_K kernel therefore widens to int32
//! after every *second* `vpmaddubsw`: four products per lane, at most
//! `4 · 63 · 127 = 32 004`, which fits with 763 to spare. The test
//! `the_int16_stage_has_headroom` asserts that bound too.

#![cfg(target_arch = "x86_64")]
#![cfg_attr(not(feature = "fmt-kquant"), allow(dead_code))]

use crate::kq::{KqActs, ACT_GROUP, Q4K8, Q6K8};
use crate::quant::QuantAct;
use std::arch::x86_64::*;

/// The rows one tile holds.
const ROWS: usize = 8;
/// The sub-blocks of a super-block.
const SUBS: usize = 8;

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

/// The byte shuffle that pairs the eight rows' scales: each scale twice, so
/// that the sixteen lanes read `sc0, sc0, sc4, sc4, …` in the lane order.
///
/// The scale madd adds the lanes in pairs, and the int16 accumulator holds a
/// row's products in the lanes `2r` and `2r+1`, so the scale of row `r` has to
/// be in both.
static DUP16: [u8; 32] = {
    let mut t = [0u8; 32];
    let mut k = 0;
    while k < 16 {
        // The bytes 0..15 hold the lanes 0..3 of the accumulator's pairs and
        // the bytes 16..31 the lanes 4..7.
        t[k] = (2 * LANES[k / 4] + (k % 2)) as u8;
        t[k + 16] = (16 + 2 * LANES[k / 4 + 4] + (k % 2)) as u8;
        k += 1;
    }
    t
};

/// The byte shuffle that puts the eight rows' mins in the lane order.
static MINROW: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut k = 0;
    while k < 8 {
        t[k] = LANES[k] as u8;
        t[k + 8] = LANES[k] as u8;
        k += 1;
    }
    t
};

/// The byte shuffle that gives each of the eight rows' scales two lanes.
///
/// The scale madd adds the lanes in pairs, and the int16 accumulator holds a
/// row's products in the lanes `2r` and `2r+1`, so the scale of row `r` has to
/// be in both. The low half of the register holds the first four rows and the
/// high half the last four, so the high half takes the *second* four scales.
static DUP8: [u8; 32] = {
    let mut t = [0u8; 32];
    let mut k = 0;
    while k < 16 {
        // The scale is a 16-bit lane: both its bytes go to both copies.
        t[k] = (k / 4 * 2 + k % 2) as u8;
        t[16 + k] = (k / 4 * 2 + k % 2 + 8) as u8;
        k += 1;
    }
    t
};

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
    /// sums. The sum is exact in float — a sub-block's sum is at most 4064 —
    /// so the two paths agree on it.
    fn min_sum(&self, row: usize, sb: usize, j: usize) -> f32;
    /// The four activation bytes that one Q6_K chunk of four weights pairs
    /// with: the window `win` of the super-block, the four-byte group `grp`
    /// inside it.
    ///
    /// # Safety
    ///
    /// The pointer must be valid for four bytes.
    unsafe fn quad(&self, row: usize, sb: usize, win: usize, grp: usize) -> *const u8;
    /// The activation's sum over the sixteen weights of run `run` of the
    /// super-block.
    ///
    /// The offset term of the Q6_K format: the row's scale against this sum.
    /// The sum is exact in float — a run's sum is at most 2032 — so the two
    /// paths agree on it.
    fn run_sum(&self, row: usize, sb: usize, run: usize) -> f32;
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
    fn min_sum(&self, row: usize, sb: usize, j: usize) -> f32 {
        let g = row / ACT_GROUP;
        let t = row % ACT_GROUP;
        let n_sb = self.acts.super_blocks();
        // A sub-block is two runs of sixteen.
        let base = (g * n_sb + sb) * ACT_GROUP * 16 + t * 16;
        self.acts.bs[base + 2 * j] + self.acts.bs[base + 2 * j + 1]
    }

    #[inline(always)]
    unsafe fn quad(&self, row: usize, sb: usize, win: usize, grp: usize) -> *const u8 {
        let g = row / ACT_GROUP;
        let t = row % ACT_GROUP;
        let n_sb = self.acts.super_blocks();
        let base = (g * n_sb + sb) * 1024;
        self.acts.q.as_ptr().add(base + win * 32 + t * 8 + grp * 4) as *const u8
    }

    #[inline(always)]
    fn run_sum(&self, row: usize, sb: usize, run: usize) -> f32 {
        let g = row / ACT_GROUP;
        let t = row % ACT_GROUP;
        let n_sb = self.acts.super_blocks();
        self.acts.bs[(g * n_sb + sb) * ACT_GROUP * 16 + t * 16 + run]
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
    fn min_sum(&self, row: usize, sb: usize, j: usize) -> f32 {
        // The per-16 run sums, added into the sub-block's sum.
        let rs = &self.acts[row].bsums[sb * 16..sb * 16 + 16];
        (rs[2 * j] as i32 + rs[2 * j + 1] as i32) as f32
    }

    #[inline(always)]
    unsafe fn quad(&self, row: usize, sb: usize, win: usize, grp: usize) -> *const u8 {
        self.acts[row].qk.as_ptr().add(sb * 256 + win * 8 + grp * 4) as *const u8
    }

    #[inline(always)]
    fn run_sum(&self, row: usize, sb: usize, run: usize) -> f32 {
        self.acts[row].bsums[sb * 16 + run] as f32
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

/// The eight rows' six-bit scales of one sub-block, as sixteen 16-bit lanes:
/// each row's scale twice, so the scale madd gives one lane per row.
#[inline(always)]
unsafe fn scale_vec(sc: *const u8) -> __m256i {
    let w = _mm256_cvtepu8_epi16(_mm_loadl_epi64(sc as *const __m128i));
    // The eight scales are in the low half; the madd reads sixteen lanes, so
    // the high half has to hold them again.
    let w = _mm256_permute4x64_epi64::<0x44>(w);
    _mm256_shuffle_epi8(w, _mm256_loadu_si256(DUP16.as_ptr() as *const __m256i))
}

/// The eight rows' six-bit mins of one sub-block, as floats in the lane order.
///
/// The mins are at most 63, so the widen and the convert are exact.
#[inline(always)]
unsafe fn min_f32(m: *const u8) -> __m256 {
    let b = _mm_shuffle_epi8(
        _mm_loadl_epi64(m as *const __m128i),
        _mm_loadu_si128(MINROW.as_ptr() as *const __m128i),
    );
    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b))
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

/// Eight output rows of a repacked Q4_K tile against `nrows` activation rows.
///
/// The tile is `n_sb` super-blocks of eight rows; `out` receives
/// `out[b · 8 + r]` for the rows `row0 .. row0 + nrows` of the activation.
///
/// # Safety
///
/// The caller must have checked that the CPU has AVX2, FMA and F16C. The tile
/// must hold `n_sb` tiles, and the activation must carry the per-256 layout.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4k8_group<S: ActSrc>(
    tile: &[u8],
    n_sb: usize,
    src: &S,
    row0: usize,
    nrows: usize,
    out: &mut [f32],
) {
    debug_assert!((1..=ACT_GROUP).contains(&nrows));
    let lownib = _mm256_set1_epi8(0x0F);
    let mut facc = [_mm256_setzero_ps(); ACT_GROUP];
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
        let mut sumi = [_mm256_setzero_si256(); ACT_GROUP];
        for j in 0..SUBS {
            // A sub-block is 32 weights: the low nibbles of four windows when it
            // is even, the high nibbles of the same four windows when it is odd.
            let wbase = (j / 2) * 4;
            let high = j % 2 == 1;
            let mut i16 = [_mm256_setzero_si256(); ACT_GROUP];
            for k in 0..4 {
                let w = wbase + k;
                let (g0, g1) = window_groups(t.add(crate::kq::q4_qs()), w);
                let (v1, v2) = if high {
                    (
                        _mm256_and_si256(_mm256_srli_epi16(g0, 4), lownib),
                        _mm256_and_si256(_mm256_srli_epi16(g1, 4), lownib),
                    )
                } else {
                    (_mm256_and_si256(g0, lownib), _mm256_and_si256(g1, lownib))
                };
                for u in 0..nrows {
                    let q = src.quads(row0 + u, sb, w);
                    let a1 = bcast4(q[0]);
                    let a2 = bcast4(q[1]);
                    let (x1, x2) = if high {
                        (bcast4(q[2]), bcast4(q[3]))
                    } else {
                        (a1, a2)
                    };
                    // Two `vpmaddubsw`: one for each four-byte group of the
                    // window. Their int16 lanes are the rows' two products each,
                    // the rows in the lanes' order.
                    let p = _mm256_add_epi16(
                        _mm256_maddubs_epi16(v1, x1),
                        _mm256_maddubs_epi16(v2, x2),
                    );
                    i16[u] = _mm256_add_epi16(i16[u], p);
                }
            }
            // The scale madd: the int16 sums become the int32 sums of the
            // super-block, one lane per row.
            let sv = scale_vec(t.add(crate::kq::q4_sc() + j * 8));
            for u in 0..nrows {
                sumi[u] = _mm256_add_epi32(sumi[u], _mm256_madd_epi16(sv, i16[u]));
            }
        }
        // The min term: the rows' six-bit mins against the activation's
        // per-sub-block sums. It is exact in float — a sub-block's sum is at
        // most 4064 and the super-block's min term at most 2 048 256 — so the
        // two paths fold the same number.
        for u in 0..nrows {
            let mut minterm = _mm256_setzero_ps();
            for j in 0..8 {
                let mv = min_f32(t.add(crate::kq::q4_m()).add(j * 8));
                minterm =
                    _mm256_fmadd_ps(mv, _mm256_set1_ps(src.min_sum(row0 + u, sb, j)), minterm);
            }
            facc[u] = fold(
                facc[u],
                sumi[u],
                minterm,
                dv,
                dminv,
                src.scale(row0 + u, sb),
            );
        }
    }
    // The eight lanes are the eight output rows, in the lane order.
    let mut lane = [0f32; ROWS];
    for u in 0..nrows {
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
    let src = Interleaved { acts };
    let groups = acts.groups();
    for g in 0..groups {
        let b0 = g * ACT_GROUP;
        let n = (nb - b0).min(ACT_GROUP);
        q4k8_group(tile, n_sb, &src, b0, n, out);
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
    // The rows together: the weight side is then unpacked once for all of them.
    let src = Plain { acts };
    q4k8_group(tile, n_sb, &src, 0, nb, out);
}

/// The eight rows' eight-bit scales of one run, as sixteen 16-bit lanes: each
/// row's scale twice, so the scale madd gives one lane per row.
///
/// The scales are signed — a Q6_K scale is an `i8` — so the widen is signed.
#[inline(always)]
unsafe fn scale_vec6(sc: *const u8) -> __m256i {
    let v = _mm256_cvtepi8_epi16(_mm_loadl_epi64(sc as *const __m128i));
    // The low half holds the eight scales; each has to be in two lanes.
    let v = _mm256_permute4x64_epi64::<0x44>(v);
    _mm256_shuffle_epi8(v, _mm256_loadu_si256(DUP8.as_ptr() as *const __m256i))
}

/// The eight rows' codes of one four-weight chunk, as thirty-two bytes: each
/// row's four codes in one 32-bit lane, in the row order.
///
/// The codes are six bits wide: four in the low nibbles and two in the high
/// bits. The low nibbles are the byte's low half for the groups 0 and 1 and
/// its high half for the groups 2 and 3; the two high bits are the group's own
/// two of the byte's four fields, so the group's shift is the same for every
/// row and every weight of the group.
#[inline(always)]
unsafe fn q6_codes(ql: *const u8, qh: *const u8, g: usize) -> __m256i {
    let qlv = _mm256_loadu_si256(ql as *const __m256i);
    let qhv = _mm256_loadu_si256(qh as *const __m256i);
    let low = if g < 2 {
        _mm256_and_si256(qlv, _mm256_set1_epi8(0x0F))
    } else {
        _mm256_and_si256(_mm256_srli_epi16(qlv, 4), _mm256_set1_epi8(0x0F))
    };
    // The shift is per byte, and the 16-bit shift is not: the bits that come in
    // from the next byte land above the two wanted, and the mask takes them
    // away again. The group's shift is a constant, so the four cases are
    // spelled out.
    let hi = match g {
        0 => _mm256_and_si256(qhv, _mm256_set1_epi8(0x03)),
        1 => _mm256_and_si256(_mm256_srli_epi16(qhv, 2), _mm256_set1_epi8(0x03)),
        2 => _mm256_and_si256(_mm256_srli_epi16(qhv, 4), _mm256_set1_epi8(0x03)),
        _ => _mm256_and_si256(_mm256_srli_epi16(qhv, 6), _mm256_set1_epi8(0x03)),
    };
    // The two bits move up to the code's high nibble: the mask leaves them in
    // the byte's low two bits, and the shift by four puts them where the code
    // wants them.
    _mm256_or_si256(low, _mm256_slli_epi16(hi, 4))
}

/// Eight output rows of a repacked Q6_K tile against `nrows` activation rows.
///
/// The tile is `n_sb` super-blocks of eight rows; `out` receives
/// `out[b · 8 + r]` for the rows `row0 .. row0 + nrows` of the activation.
///
/// # Safety
///
/// The caller must have checked that the CPU has AVX2, FMA and F16C. The tile
/// must hold `n_sb` tiles, and the activation must carry the per-256 layout.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q6k8_group<S: ActSrc>(
    tile: &[u8],
    n_sb: usize,
    src: &S,
    row0: usize,
    nrows: usize,
    out: &mut [f32],
) {
    debug_assert!((1..=ACT_GROUP).contains(&nrows));
    let mut facc = [_mm256_setzero_ps(); ACT_GROUP];
    for sb in 0..n_sb {
        let t = tile.as_ptr().add(sb * Q6K8::SIZE);
        // The eight rows' super-block scales, in the row order.
        let dv = _mm256_cvtph_ps(_mm_loadu_si128(t.add(crate::kq::q6_d()) as *const __m128i));
        let mut sumi = [_mm256_setzero_si256(); ACT_GROUP];
        let mut off = [_mm256_setzero_si256(); ACT_GROUP];
        // Two halves of 128 weights, four groups of 32, two runs of 16.
        for half in 0..2 {
            for g in 0..4 {
                for run in 0..2 {
                    let sub = half * 8 + g * 2 + run;
                    let sv = scale_vec6(t.add(crate::kq::q6_sc() + sub * 8));
                    // The offset: 32 on every code, against the run's sum.
                    // The sum is exact, so the whole term is exact in the
                    // integer lanes, as the ordinary kernel has it.
                    for u in 0..nrows {
                        let b = src.run_sum(row0 + u, sb, sub) as i16;
                        let bz =
                            _mm256_blend_epi16(_mm256_set1_epi16(b), _mm256_setzero_si256(), 0xAA);
                        off[u] = _mm256_add_epi32(
                            off[u],
                            _mm256_slli_epi32(_mm256_madd_epi16(sv, bz), 5),
                        );
                    }
                    // The run's sixteen weights: four chunks of four, in two
                    // pairs. A pair's two `vpmaddubsw` stay inside the int16
                    // bound — four products per lane, at most 32 004 — and
                    // the scale madd widens them to int32.
                    for pair in 0..2 {
                        let cc = run * 4 + pair * 2;
                        let c0 = q6_codes(
                            t.add(crate::kq::q6_ql() + (half * 16 + (g & 1) * 8 + cc) * 32),
                            t.add(crate::kq::q6_qh() + (half * 8 + cc) * 32),
                            g,
                        );
                        let c1 = q6_codes(
                            t.add(crate::kq::q6_ql() + (half * 16 + (g & 1) * 8 + cc + 1) * 32),
                            t.add(crate::kq::q6_qh() + (half * 8 + cc + 1) * 32),
                            g,
                        );
                        let win = half * 16 + g * 4 + cc / 2;
                        for u in 0..nrows {
                            let a0 = bcast4(src.quad(row0 + u, sb, win, 0));
                            let a1 = bcast4(src.quad(row0 + u, sb, win, 1));
                            let i16 = _mm256_add_epi16(
                                _mm256_maddubs_epi16(c0, a0),
                                _mm256_maddubs_epi16(c1, a1),
                            );
                            sumi[u] = _mm256_add_epi32(sumi[u], _mm256_madd_epi16(sv, i16));
                        }
                    }
                }
            }
        }
        // The fold: the integer sums become the float value, one FMA per
        // super-block.
        for u in 0..nrows {
            let v = _mm256_sub_epi32(sumi[u], off[u]);
            let dk = _mm256_set1_ps(src.scale(row0 + u, sb));
            facc[u] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), _mm256_mul_ps(dv, dk), facc[u]);
        }
    }
    // The eight lanes are the eight output rows.
    let mut lane = [0f32; ROWS];
    for u in 0..nrows {
        _mm256_storeu_ps(lane.as_mut_ptr(), facc[u]);
        for k in 0..ROWS {
            out[(row0 + u) * ROWS + k] = lane[k];
        }
    }
}

/// Eight output rows of a repacked Q6_K tile against a batch of activations.
///
/// The interleaved activation carries four rows at a time, so the batch is
/// walked in groups of four; a group with fewer rows left is padded with zero
/// codes, whose results are discarded.
///
/// # Safety
///
/// The caller must have checked that the CPU has AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q6k8_gemm(tile: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    let src = Interleaved { acts };
    let groups = acts.groups();
    for g in 0..groups {
        let b0 = g * ACT_GROUP;
        let n = (nb - b0).min(ACT_GROUP);
        q6k8_group(tile, n_sb, &src, b0, n, out);
    }
}

/// Eight output rows of a repacked Q6_K tile against one to three activations.
///
/// The decode path: the activation is read in the plain per-row layout, one row
/// at a time, and the result is the same as the batched path's.
///
/// # Safety
///
/// The caller must have checked that the CPU has AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q6k8_gemv(tile: &[u8], n_sb: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!((1..=3).contains(&nb), "the GEMV takes one to three rows");
    let src = Plain { acts };
    q6k8_group(tile, n_sb, &src, 0, nb, out);
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

    /// One Q6_K super-block with sane scales and random codes.
    pub fn q6k_block(rng: &mut Rng) -> Vec<u8> {
        let mut b = rng.bytes(210);
        // The f16 super-block scale: a small positive number, so the products
        // stay in the range the float fold can hold.
        b[208..210].copy_from_slice(&crate::quant::f32_to_f16(0.01).to_le_bytes());
        // The sixteen int8 scales, over the whole range but not extreme.
        for s in 0..16 {
            b[192 + s] = (rng.next_u8() as i8).wrapping_mul(3).wrapping_div(4) as u8;
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

    /// A Q6_K tile of eight rows and `n_sb` super-blocks.
    pub fn q6k8_tile(rng: &mut Rng, n_sb: usize) -> Vec<u8> {
        let mut tile = vec![0u8; n_sb * Q6K8::SIZE];
        for sb in 0..n_sb {
            let rows: Vec<Vec<u8>> = (0..8).map(|_| q6k_block(rng)).collect();
            let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
            crate::kq::repack_q6k8(&refs, &mut tile[sb * Q6K8::SIZE..(sb + 1) * Q6K8::SIZE]);
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

    /// The Q6_K repacked GEMM against the scalar reference, and the GEMV against
    /// the GEMM: the two paths have to agree bit for bit, because a row's
    /// result must not depend on the batch it ran in.
    #[test]
    fn the_repacked_q6_k_kernels_match_the_reference_and_each_other() {
        if !crate::simd::have_batched() {
            return;
        }
        let n_sb = 4;
        let n = n_sb * 256;
        let mut rng = Rng(0xC0DE_0BAD_5EED_1234);
        let tile = q6k8_tile(&mut rng, n_sb);

        // The batch: four rows for the GEMM, and the same rows one at a time
        // for the GEMV.
        let batch = acts(&mut rng, 4, n);
        let mut kq = KqActs::default();
        kq.fill(&batch);
        let mut gemm = vec![0.0f32; 4 * 8];
        unsafe { q6k8_gemm(&tile, n_sb, &kq, 4, &mut gemm) };
        for b in 0..4 {
            for r in 0..8 {
                let want = crate::kq::dot_q6k8_ref(&tile, n_sb, r, &batch[b]);
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
            unsafe { q6k8_gemv(&tile, n_sb, &batch[..nb], nb, &mut gemv) };
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

    /// The repacked Q6_K reference against the ordinary per-row Q6_K kernel:
    /// the repacked layout has to compute the same real number as the format's
    /// own reading of the same bytes.
    #[test]
    fn the_repacked_q6_k_reference_agrees_with_the_ordinary_kernel() {
        let n_sb = 3;
        let n = n_sb * 256;
        let mut rng = Rng(0x1234_ABCD_9876_5432);
        let tile = q6k8_tile(&mut rng, n_sb);
        let batch = acts(&mut rng, 1, n);
        // The file's own bytes, so the ordinary kernel can read them. The tile
        // holds one super-block of eight rows, so each super-block is its own
        // set of blocks.
        let mut file = vec![0u8; 8 * 210];
        for r in 0..8 {
            for sb in 0..n_sb {
                crate::kq::unrepack_q6k8(&tile[sb * Q6K8::SIZE..(sb + 1) * Q6K8::SIZE], &mut file);
                // The ordinary kernel reads one super-block at a time, so it
                // gets an activation holding only that super-block.
                let mut a = batch[0].clone();
                a.n = 256;
                a.qk = batch[0].qk[sb * 256..sb * 256 + 256].to_vec();
                a.bsums = batch[0].bsums[sb * 16..sb * 16 + 16].to_vec();
                a.dk = vec![batch[0].dk[sb]];
                let blk = &file[r * 210..(r + 1) * 210];
                let want = crate::quant::dot_scalar(crate::format::Format::Q6_K, blk, &a) as f64;
                let got = crate::kq::dot_q6k8_ref(
                    &tile[sb * Q6K8::SIZE..(sb + 1) * Q6K8::SIZE],
                    1,
                    r,
                    &a,
                ) as f64;
                let scale = want.abs().max(1.0);
                assert!(
                    (got - want).abs() / scale < 1e-5,
                    "row {r} super-block {sb}: the repacked reference {got} against the ordinary {want}"
                );
            }
        }
    }

    /// The int16 stage of the repacked kernels has to stay inside int16.
    ///
    /// The Q4_K kernel sums eight `vpmaddubsw` results in int16 before the
    /// scale madd widens them: sixteen products per lane, at most
    /// `16 · 15 · 127 = 30 480`. The Q6_K kernel widens after every *second*
    /// `vpmaddubsw`, so its lanes hold four products: at most
    /// `4 · 63 · 127 = 32 004`. Both are inside int16's 32 767.
    #[test]
    fn the_int16_stage_has_headroom() {
        // Q4_K: sixteen products of a code of at most 15 and an activation of
        // at most 127.
        assert!(
            16 * 15 * 127 < i16::MAX as i32,
            "the Q4_K stage would saturate"
        );
        // Q6_K: four products of a code of at most 63 and an activation of at
        // most 127. The codes reach 63 and the quantiser clamps the
        // activations to ±127, so this is the bound by construction.
        assert!(
            4 * 63 * 127 < i16::MAX as i32,
            "the Q6_K stage would saturate"
        );
        // The int32 the scale madd widens to: the scale is at most 127 and
        // the lane holds four products.
        assert!(
            4 * 63 * 127 * 127 < i64::from(i32::MAX),
            "the Q6_K scale madd would overflow"
        );
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
        // The scale control pairs the rows' scales: each twice, in the lane
        // order.
        for k in 0..16 {
            assert_eq!(
                DUP16[k],
                (2 * LANES[k / 4] + (k % 2)) as u8,
                "the scale lane {k}"
            );
            assert_eq!(
                DUP16[k + 16],
                (16 + 2 * LANES[k / 4 + 4] + (k % 2)) as u8,
                "the scale lane {}",
                k + 16
            );
        }
        // The min control puts the rows' mins in the lane order.
        for k in 0..8 {
            assert_eq!(MINROW[k], LANES[k] as u8, "the min {k}");
        }
    }
}
