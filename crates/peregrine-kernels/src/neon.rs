//! aarch64 NEON kernels (roadmap S2.8).
//!
//! The ARM twin of `simd.rs` and `kqsimd.rs`: decode and prefill kernels for the formats whose
//! arithmetic is an int8 dot product, built on `sdot` (`asimddp`) and the eight-row tiles of S3.4.
//! The scalar references in `quant.rs` and `kq.rs` are the specification; this file only makes them
//! fast, and `docs/design/s2.8-neon.md` records the tier, the format order and the invariance
//! argument.
//!
//! # What `sdot` decides
//!
//! `vdotq_s32` multiplies sixteen signed bytes by sixteen signed bytes and adds four products into
//! each of four int32 lanes. That is exactly the shape of every dot product here: the weight codes
//! (Q8_0's signed bytes, Q4_K's 0..15 nibbles) against the int8 activations. There is no int16
//! stage and no saturation question — the lanes are int32, and every integer this file forms is
//! four orders of magnitude below the overflow point — so the kernels inherit the AVX-512 variant's
//! freedom rather than the AVX2 one's int16 bound (`docs/design/s3.4.md` §4.1).
//!
//! # Bit identity
//!
//! Integer sums are exact and associative, so the only thing a vector kernel can change is the
//! order of the float fold. Each kernel keeps the fold of the reference it replaces, operand for
//! operand:
//!
//! * the per-row Q8_0 and Q4_K kernels evaluate `quant::dot_scalar`'s expression in its order and
//!   are **bit-identical** to it;
//! * the repacked Q4_K kernels evaluate `kq::dot_q4k8_ref`'s `mul_add` chain and are **bit-identical**
//!   to it, hence to the AVX2 tier's tile kernels.
//!
//! The tests at the bottom of this file assert `to_bits()` equality, not a tolerance. That is what
//! keeps a row's result independent of batch width, chunk, pass width, thread count and
//! speculation (`docs/design/s2.8-neon.md` §5).
//!
//! # S4.9f: the packed float GEMM
//!
//! The packed float GEMM (S4.9f) keeps its kernels beside their scalar twin in `fpack.rs`,
//! the way the AVX2 and AVX-512 ones do; its NEON module is there too (`fcvtl` widens f16,
//! a high-half shift widens bf16, and the accumulator is one `vfmaq_f32` chain per output).
//! This file remains the home of the int8 NEON kernels above.

#![cfg(target_arch = "aarch64")]
#![cfg_attr(
    not(all(feature = "fmt-kquant", feature = "fmt-legacy")),
    allow(dead_code)
)]

use crate::format::{f16, Format};
use crate::kq::{
    tile_q4k_min, tile_q4k_scale, KqActs, ACT_GROUP, Q4K8, Q5_08, Q5_0_WEIGHTS, Q6K8, Q8_08,
    Q8_0_WEIGHTS,
};
use crate::quant::QuantAct;
use std::arch::aarch64::*;

/// The eight rows a repacked tile holds.
const ROWS: usize = 8;
/// The byte offset of a Q4_K tile's nibbles (from `kq::q4_qs`).
#[inline(always)]
fn q4_qs() -> usize {
    crate::kq::q4_qs()
}
/// The byte offset of a Q4_K tile's super-block scales.
#[inline(always)]
fn q4_d() -> usize {
    crate::kq::q4_d()
}
/// The byte offset of a Q4_K tile's super-block min scales.
#[inline(always)]
fn q4_dmin() -> usize {
    crate::kq::q4_dmin()
}

/// An f16 field of a byte slice, little endian.
#[inline(always)]
fn f16_at(b: &[u8], at: usize) -> f32 {
    f16(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// An f16 read through a raw pointer.
///
/// # Safety
/// `p` must be readable for two bytes.
#[inline(always)]
unsafe fn f16_ptr(p: *const u8) -> f32 {
    f16(u16::from_le_bytes([*p, *p.add(1)]))
}

/// `sdot`: four int32 lanes of four signed byte products each.
///
/// The intrinsic (`vdotq_s32`) is stable only since Rust 1.98 and the workspace's MSRV is 1.96, so
/// the one instruction is written out. The macro expands inside the caller, whose
/// `#[target_feature(enable = "dotprod")]` is what makes the assembler accept it; a function would
/// have to carry the attribute too and could then not be `#[inline(always)]` into it.
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = acc;
    std::arch::asm!(
        "sdot {r:v}.4s, {a:v}.16b, {b:v}.16b",
        r = inout(vreg) r,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    r
}

/// The sum of an `i32x4`'s four lanes.
///
/// `vaddvq_s32` is also newer than the MSRV, and the lane extracts are the older spelling of the
/// same three additions.
#[inline(always)]
unsafe fn hsum_s32x4(v: int32x4_t) -> i32 {
    #[allow(clippy::incompatible_msrv)]
    {
        vaddvq_s32(v)
    }
}

/// 32 signed codes against 32 signed activations, one int32 sum.
///
/// The two `sdot`s are one asm block so the accumulator stays in one register: as two separate
/// blocks the compiler cannot coalesce them and copies the vector between them (measured: 25
/// against 40 GMAC/s on this host).
///
/// # Safety
/// The CPU must have `asimddp`. Both pairs of vectors are read whole.
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn dot32_sum(x0: int8x16_t, x1: int8x16_t, a0: int8x16_t, a1: int8x16_t) -> i32 {
    let mut t = vdupq_n_s32(0);
    std::arch::asm!(
        "sdot {t:v}.4s, {x0:v}.16b, {a0:v}.16b",
        "sdot {t:v}.4s, {x1:v}.16b, {a1:v}.16b",
        t = inout(vreg) t,
        x0 = in(vreg) x0,
        x1 = in(vreg) x1,
        a0 = in(vreg) a0,
        a1 = in(vreg) a1,
        options(pure, nomem, nostack),
    );
    hsum_s32x4(t)
}

/// 32 signed codes against 32 signed activations, one int32 sum.
///
/// # Safety
/// The CPU must have `asimddp`. Both pointers must be readable for 32 bytes.
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn dot32_s8(w: *const i8, a0: int8x16_t, a1: int8x16_t) -> i32 {
    dot32_sum(vld1q_s8(w), vld1q_s8(w.add(16)), a0, a1)
}

// ---------------------------------------------------------------------------
// Activation quantisation and the small float helpers (S2.8 F6)
// ---------------------------------------------------------------------------

/// The largest absolute value of `x` (0 for an empty slice), the same fold the scalar
/// loop makes.
///
/// # Safety
/// The CPU must have NEON.
#[target_feature(enable = "neon")]
pub unsafe fn absmax(x: &[f32]) -> f32 {
    let mut m = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= x.len() {
        m = vmaxq_f32(m, vabsq_f32(vld1q_f32(x.as_ptr().add(i))));
        i += 4;
    }
    let mut lanes = [0.0f32; 4];
    vst1q_f32(lanes.as_mut_ptr(), m);
    let mut r = lanes.iter().fold(0.0f32, |a, &b| a.max(b));
    for &v in &x[i..] {
        r = r.max(v.abs());
    }
    r
}

/// `out[i] = clamp(round(x[i] · inv), ±127)` for 32 values.
///
/// `round` is half away from zero; it is rebuilt as `trunc(|y|) + (frac ≥ 0.5)` with the
/// sign put back, exactly the scalar loop's and the AVX2 kernel's construction, so the
/// codes are bit-identical to both.
///
/// # Safety
/// The CPU must have NEON; `x` must point at 32 readable floats and `out` at 32
/// writable bytes.
#[target_feature(enable = "neon")]
pub unsafe fn quantize32(x: *const f32, inv: f32, out: *mut i8) {
    let vinv = vdupq_n_f32(inv);
    let half = vdupq_n_f32(0.5);
    let zero = vdupq_n_f32(0.0);
    let hi = vdupq_n_s32(127);
    let lo = vdupq_n_s32(-127);
    let one = vdupq_n_s32(1);
    for k in (0..32).step_by(4) {
        let y = vmulq_f32(vld1q_f32(x.add(k)), vinv);
        let ay = vabsq_f32(y);
        let t = vcvtq_s32_f32(ay);
        let tf = vcvtq_f32_s32(t);
        // `frac >= 0.5`, as a 0/1 integer, then the sign and the clamp.
        let up = vandq_s32(
            vreinterpretq_s32_u32(vcgeq_f32(vsubq_f32(ay, tf), half)),
            one,
        );
        let mut r = vaddq_s32(t, up);
        r = vbslq_s32(vcltq_f32(y, zero), vnegq_s32(r), r);
        r = vminq_s32(vmaxq_s32(r, lo), hi);
        let mut vals = [0i32; 4];
        vst1q_s32(vals.as_mut_ptr(), r);
        for (j, &v) in vals.iter().enumerate() {
            *out.add(k + j) = v as i8;
        }
    }
}

// ---------------------------------------------------------------------------
// Q8_0: a block is one f16 scale and 32 signed codes
// ---------------------------------------------------------------------------

/// One Q8_0 row against one activation: [`crate::quant::dot_scalar`]'s expression, vectorised.
///
/// # Safety
/// The CPU must have `asimddp`; `w` must be `row_bytes(Q8_0, a.n)` long.
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q8_0(w: &[u8], a: &QuantAct) -> f32 {
    debug_assert_eq!(w.len(), crate::quant::row_bytes(Format::Q8_0, a.n));
    let mut acc = 0.0f32;
    for (b, blk) in w.chunks_exact(34).enumerate() {
        let a0 = vld1q_s8(a.q.as_ptr().add(b * 32));
        let a1 = vld1q_s8(a.q.as_ptr().add(b * 32 + 16));
        let s = dot32_s8(blk.as_ptr().add(2) as *const i8, a0, a1);
        // The scalar reference's expression, in its order.
        acc += f16_at(blk, 0) * a.d[b] * s as f32;
    }
    acc
}

/// Eight Q8_0 rows against one activation: the activation block is loaded once per block.
///
/// Every row's block loop is [`dot_q8_0`]'s, so the result is bit-identical to the one-row kernel.
///
/// # Safety
/// The CPU must have `asimddp`; `w` must hold eight rows of `rb` bytes and `out` eight floats.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q8_0_rows8(w: &[u8], rb: usize, a: &QuantAct, out: &mut [f32]) {
    assert!(rb % 34 == 0, "a Q8_0 row is a whole number of blocks");
    assert!(w.len() >= 8 * rb, "the kernel takes eight rows");
    assert!(out.len() >= 8, "the output holds eight rows");
    let mut acc = [0.0f32; 8];
    for blk in 0..rb / 34 {
        let a0 = vld1q_s8(a.q.as_ptr().add(blk * 32));
        let a1 = vld1q_s8(a.q.as_ptr().add(blk * 32 + 16));
        let ad = a.d[blk];
        // The row loop is unrolled by hand: an `asm!` block makes the loop unroller treat the body
        // as a call, so the eight rows of a block would otherwise stay a small non-unrolled loop and
        // the accumulators would round-trip through the stack (measured: 25 against 40 GMAC/s).
        macro_rules! row {
            ($r:literal) => {{
                let p = w.as_ptr().add($r * rb + blk * 34);
                let wd = f16_ptr(p);
                let s = dot32_s8(p.add(2) as *const i8, a0, a1);
                acc[$r] += wd * ad * s as f32;
            }};
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
        row!(4);
        row!(5);
        row!(6);
        row!(7);
    }
    out[..8].copy_from_slice(&acc);
}

/// Four Q8_0 rows against a batch of activations of the same width.
///
/// The weight block is loaded once per four activation rows; each (row, activation) pair keeps the
/// one-row kernel's block order and fold, so a row's bits do not depend on the batch.
///
/// # Safety
/// The CPU must have `asimddp`; `w` must be one row of `row_bytes(Q8_0, acts[0].n)` and `out` one
/// float per activation.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q8_0_batch(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    assert!(!acts.is_empty(), "no activation");
    assert!(out.len() >= acts.len(), "the output holds fewer rows");
    let n = acts[0].n;
    assert_eq!(w.len(), crate::quant::row_bytes(Format::Q8_0, n));
    let nb = acts.len();
    let mut b0 = 0usize;
    while b0 < nb {
        let t = (nb - b0).min(4);
        let mut acc = [0.0f32; 4];
        for blk in 0..n / 32 {
            let wd = f16_at(&w[blk * 34..], 0);
            let w0 = vld1q_s8(w.as_ptr().add(blk * 34 + 2) as *const i8);
            let w1 = vld1q_s8(w.as_ptr().add(blk * 34 + 18) as *const i8);
            for u in 0..t {
                let a = &acts[b0 + u];
                let a0 = vld1q_s8(a.q.as_ptr().add(blk * 32));
                let a1 = vld1q_s8(a.q.as_ptr().add(blk * 32 + 16));
                let s = dot32_sum(w0, w1, a0, a1);
                acc[u] += wd * a.d[blk] * s as f32;
            }
        }
        out[b0..b0 + t].copy_from_slice(&acc[..t]);
        b0 += t;
    }
}

// ---------------------------------------------------------------------------
// Q8_0's repacked tile (S3.4 follow-up, the #122 layout): eight rows per block
// ---------------------------------------------------------------------------

/// One block of a repacked Q8_0 tile against `N` activation rows, folded into `facc`.
///
/// The tile's window is two output rows of eight signed codes — one `sdot` operand — and the four
/// windows of a block are summed in int32; the fold is `kq::dot_q8_08_ref`'s `(d·dk).mul_add(s,
/// acc)`. The GEMV and the GEMM are this one function, so a row's bits do not depend on the batch.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q8_08_group<const N: usize>(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    out: &mut [f32],
) {
    assert!((1..=4).contains(&N), "one to four activation rows");
    assert!(
        out.len() >= N * ROWS,
        "the output holds eight rows per activation"
    );
    let mut facc = [[vdupq_n_f32(0.0); 2]; N];
    for t in 0..n_blk {
        let b = tile.as_ptr().add(t * Q8_08::SIZE);
        // [activation][window][row pair]
        let mut acc = [[[vdupq_n_s32(0); 4]; 4]; N];
        for w in 0..4 {
            let mut av = [vdupq_n_s8(0); N];
            for (u, a) in av.iter_mut().enumerate() {
                let p = acts[u].q.as_ptr().add(t * Q8_0_WEIGHTS + w * 8);
                *a = vcombine_s8(vld1_s8(p), vld1_s8(p));
            }
            for pair in 0..4 {
                let codes = vld1q_s8(
                    b.add(crate::kq::q8_qs() + w * crate::kq::Q8_WINDOW + pair * 16) as *const i8,
                );
                for u in 0..N {
                    acc[u][w][pair] = sdot(acc[u][w][pair], codes, av[u]);
                }
            }
        }
        let dv = rows_f16(tile, t * Q8_08::SIZE + crate::kq::q8_d());
        for u in 0..N {
            let mut pairs = [vdupq_n_s32(0); 4];
            for (pair, p) in pairs.iter_mut().enumerate() {
                let mut s = acc[u][0][pair];
                s = vaddq_s32(s, acc[u][1][pair]);
                s = vaddq_s32(s, acc[u][2][pair]);
                s = vaddq_s32(s, acc[u][3][pair]);
                *p = s;
            }
            let s = rows_of(&pairs);
            let dk = vdupq_n_f32(acts[u].d[t]);
            for h in 0..2 {
                // `dot_q8_08_ref`'s fold, in its order: `(d·dk)` first, then one fused multiply-add.
                let dd = vmulq_f32(dv[h], dk);
                facc[u][h] = vfmaq_f32(facc[u][h], dd, vcvtq_f32_s32(s[h]));
            }
        }
    }
    for u in 0..N {
        vst1q_f32(out.as_mut_ptr().add(u * ROWS), facc[u][0]);
        vst1q_f32(out.as_mut_ptr().add(u * ROWS + 4), facc[u][1]);
    }
}

/// Eight output rows of a repacked Q8_0 tile against a batch of activations.
///
/// # Safety
/// The CPU must have `asimddp`; the tile must hold `n_blk` blocks, `acts` the rows the caller
/// says, and `out` `nb · 8` floats.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q8_08_gemm(tile: &[u8], n_blk: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    #[cfg(test)]
    Q8_08_SDOT_CALLS.with(|c| c.set(c.get() + 1));
    assert!(
        !acts.is_empty() && nb <= acts.len(),
        "more rows than activations"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    assert!(
        tile.len() >= n_blk * Q8_08::SIZE,
        "the tile is shorter than its blocks"
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
/// # Safety
/// As [`q8_08_gemm`].
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q8_08_gemv(tile: &[u8], n_blk: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!(
        (1..=3).contains(&nb),
        "the Q8_0 GEMV takes one to three rows"
    );
    q8_08_gemm(tile, n_blk, acts, nb, out);
}

// Test-only counts of the two Q8_0 tile bodies' calls, so the dispatch test can see
// which one the tier's table takes. `thread_local` keeps a concurrently running test
// out of the count. Both are compiled only into the crate's own tests.
#[cfg(test)]
thread_local! {
    pub(crate) static Q8_08_I8MM_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static Q8_08_SDOT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// One block of a repacked Q8_0 tile against four activation rows (two pairs) on
/// `smmla`, folded onto `facc[activation_pair][row_pair]`.
///
/// Q8_0's block is one f16 scale and 32 signed codes per row, so the tile's window —
/// two output rows of eight codes — is `smmla`'s first operand as it stands, and two
/// activation rows' eight codes, joined, are its second. One instruction therefore
/// forms the 32 products of two output rows by two activations, and four instructions
/// cover the eight output rows of a window. A block's four windows share one scale,
/// so they accumulate into one exact int32 sum; the fold is
/// [`kq::dot_q8_08_ref`]'s `(d·dk)` then one fused multiply-add, block by block in
/// the tile's order, so a row is bit-identical to the `sdot` body in [`q8_08_group`].
#[inline]
#[target_feature(enable = "neon,i8mm")]
unsafe fn q8_08_group_i8mm(tile: &[u8], n_blk: usize, acts: &[QuantAct], out: &mut [f32]) {
    // [activation pair][row pair]; the lanes are [r0·a0, r1·a0, r0·a1, r1·a1].
    let mut facc = [[vdupq_n_f32(0.0); 4]; 2];
    for t in 0..n_blk {
        let b = tile.as_ptr().add(t * Q8_08::SIZE);
        let mut acc = [[vdupq_n_s32(0); 4]; 2];
        for w in 0..4 {
            // Each column of `smmla`'s second operand is one activation's eight codes.
            let mut bv = [vdupq_n_s8(0); 2];
            for (q, av) in bv.iter_mut().enumerate() {
                let p0 = acts[2 * q].q.as_ptr().add(t * Q8_0_WEIGHTS + w * 8);
                let p1 = acts[2 * q + 1].q.as_ptr().add(t * Q8_0_WEIGHTS + w * 8);
                *av = vcombine_s8(vld1_s8(p0), vld1_s8(p1));
            }
            for pair in 0..4 {
                let codes = vld1q_s8(
                    b.add(crate::kq::q8_qs() + w * crate::kq::Q8_WINDOW + pair * 16) as *const i8,
                );
                for q in 0..2 {
                    acc[q][pair] = smmla(acc[q][pair], codes, bv[q]);
                }
            }
        }
        let dv = rows_f16(tile, t * Q8_08::SIZE + crate::kq::q8_d());
        let mut d8 = [0.0f32; 8];
        vst1q_f32(d8.as_mut_ptr(), dv[0]);
        vst1q_f32(d8.as_mut_ptr().add(4), dv[1]);
        for q in 0..2 {
            let (u0, u1) = (2 * q, 2 * q + 1);
            let dk = pair_f32(acts[u0].d[t], acts[u1].d[t]);
            for pair in 0..4 {
                // `dot_q8_08_ref`'s fold, in its order: `(d·dk)` first, then one fused
                // multiply-add. The lanes line up with the accumulator's.
                let dp = quad_f32(d8[2 * pair], d8[2 * pair + 1]);
                let dd = vmulq_f32(dp, dk);
                facc[q][pair] = vfmaq_f32(facc[q][pair], dd, vcvtq_f32_s32(acc[q][pair]));
            }
        }
    }
    for q in 0..2 {
        for pair in 0..4 {
            let mut lanes = [0.0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), facc[q][pair]);
            let a = 2 * q;
            out[a * ROWS + 2 * pair] = lanes[0];
            out[(a + 1) * ROWS + 2 * pair] = lanes[1];
            out[a * ROWS + 2 * pair + 1] = lanes[2];
            out[(a + 1) * ROWS + 2 * pair + 1] = lanes[3];
        }
    }
}

/// Eight output rows of a repacked Q8_0 tile against a batch of activations, on
/// `smmla` for whole groups of four and on the `sdot` body for the 1–3-row tail.
///
/// # Safety
/// The CPU must have `asimd`, `asimddp` and `i8mm`; the tile must hold `n_blk` blocks
/// and `out` must hold `nb · 8` floats.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn q8_08_gemm_i8mm(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    nb: usize,
    out: &mut [f32],
) {
    #[cfg(test)]
    Q8_08_I8MM_CALLS.with(|c| c.set(c.get() + 1));
    assert!(
        !acts.is_empty() && nb <= acts.len(),
        "more rows than activations"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    assert!(
        tile.len() >= n_blk * Q8_08::SIZE,
        "the tile is shorter than its blocks"
    );
    let mut b = 0;
    while b + ACT_GROUP <= nb {
        q8_08_group_i8mm(tile, n_blk, &acts[b..], &mut out[b * ROWS..]);
        b += ACT_GROUP;
    }
    match nb - b {
        3 => q8_08_group::<3>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        2 => q8_08_group::<2>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        1 => q8_08_group::<1>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Q5_0: a block is one f16 scale, four fifth-bit bytes and sixteen nibbles
// ---------------------------------------------------------------------------

/// The shift counts that spread a replicated `qh` byte into one bit per byte.
const Q5_NEG_SHIFT: [i8; 16] = [0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7];

/// An index vector that copies source bytes `a` and `b` eight times each.
const fn spread_idx(a: u8, b: u8) -> [u8; 16] {
    let mut t = [0u8; 16];
    let mut k = 0;
    while k < 8 {
        t[k] = a;
        t[8 + k] = b;
        k += 1;
    }
    t
}

/// The two rows of a Q5_0 tile pair `p` (rows `2p`, `2p+1`) as one byte each, spread.
const Q5_PAIR_IDX: [[u8; 16]; 4] = [
    spread_idx(0, 1),
    spread_idx(2, 3),
    spread_idx(4, 5),
    spread_idx(6, 7),
];

/// The two `qh` bytes of one sixteen-weight half of a Q5_0 block, spread.
const Q5_HALF_IDX: [[u8; 16]; 2] = [spread_idx(0, 1), spread_idx(2, 3)];

/// The eight fifth bits of the two source bytes `idx` names, as one `0x10`/`0` byte each.
///
/// Every byte of `table` holds eight bits; byte `k` of the result is bit `k` of the
/// selected source byte. The shift is per byte (`vshl`), the mask keeps the one bit,
/// and the final shift puts it at the `16` the format's offset wants.
#[inline(always)]
unsafe fn spread_bits(table: uint8x16_t, idx: &[u8; 16]) -> uint8x16_t {
    let replicated = vqtbl1q_u8(table, vld1q_u8(idx.as_ptr()));
    let shifted = vshlq_u8(replicated, vld1q_s8(Q5_NEG_SHIFT.as_ptr()));
    vshlq_n_u8(vandq_u8(shifted, vdupq_n_u8(1)), 4)
}

/// One Q5_0 row against one activation: [`crate::quant::dot_scalar`]'s expression, vectorised.
///
/// The file's sixteen nibble bytes are the low nibbles of weights `0..16` and the high
/// nibbles of `16..32`; the two `sdot`s are one per half, with the fifth bits OR'd into
/// the codes. The integer sum and the `−16·Σa` bias are exact, so the result is
/// **bit-identical** to the scalar reference.
///
/// # Safety
/// The CPU must have `asimddp`; `w` must be `row_bytes(Q5_0, a.n)` long.
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q5_0(w: &[u8], a: &QuantAct) -> f32 {
    debug_assert_eq!(w.len(), crate::quant::row_bytes(Format::Q5_0, a.n));
    let lownib = vdupq_n_u8(0x0F);
    let mut acc = 0.0f32;
    for (b, blk) in w.chunks_exact(22).enumerate() {
        let p = blk.as_ptr();
        let qs = vld1q_u8(p.add(6));
        let qh = vld1q_u8(p.add(2));
        let lo = vorrq_u8(vandq_u8(qs, lownib), spread_bits(qh, &Q5_HALF_IDX[0]));
        let hi = vorrq_u8(vshrq_n_u8(qs, 4), spread_bits(qh, &Q5_HALF_IDX[1]));
        let a0 = vld1q_s8(a.q.as_ptr().add(b * 32));
        let a1 = vld1q_s8(a.q.as_ptr().add(b * 32 + 16));
        let s = dot32_sum(vreinterpretq_s8_u8(lo), vreinterpretq_s8_u8(hi), a0, a1);
        // The scalar reference's expression, in its order: the bias in integers,
        // then two multiplies and the accumulation.
        acc += f16_at(blk, 0) * a.d[b] * (s - 16 * a.sums[b]) as f32;
    }
    acc
}

/// Eight Q5_0 rows against one activation: the activation halves are loaded once per block.
///
/// Every row's block loop is [`dot_q5_0`]'s, so the result is bit-identical to the
/// one-row kernel.
///
/// # Safety
/// The CPU must have `asimddp`; `w` must hold eight rows of `rb` bytes and `out` eight floats.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q5_0_rows8(w: &[u8], rb: usize, a: &QuantAct, out: &mut [f32]) {
    assert!(rb % 22 == 0, "a Q5_0 row is a whole number of blocks");
    assert!(w.len() >= 8 * rb, "the kernel takes eight rows");
    assert!(out.len() >= 8, "the output holds eight rows");
    let lownib = vdupq_n_u8(0x0F);
    let mut acc = [0.0f32; 8];
    for blk in 0..rb / 22 {
        let a0 = vld1q_s8(a.q.as_ptr().add(blk * 32));
        let a1 = vld1q_s8(a.q.as_ptr().add(blk * 32 + 16));
        let bias = 16 * a.sums[blk];
        macro_rules! row {
            ($r:literal) => {{
                let p = w.as_ptr().add($r * rb + blk * 22);
                let qs = vld1q_u8(p.add(6));
                let qh = vld1q_u8(p.add(2));
                let lo = vorrq_u8(vandq_u8(qs, lownib), spread_bits(qh, &Q5_HALF_IDX[0]));
                let hi = vorrq_u8(vshrq_n_u8(qs, 4), spread_bits(qh, &Q5_HALF_IDX[1]));
                let s = dot32_sum(vreinterpretq_s8_u8(lo), vreinterpretq_s8_u8(hi), a0, a1);
                acc[$r] += f16_ptr(p) * a.d[blk] * (s - bias) as f32;
            }};
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
        row!(4);
        row!(5);
        row!(6);
        row!(7);
    }
    out[..8].copy_from_slice(&acc);
}

/// One super-block of a repacked Q5_0 tile against `N` activations, folded into `facc`.
///
/// The tile's 64-byte window holds two rows of eight weights per sixteen-byte load, so
/// the same `sdot` that a Q4_K window uses computes two output rows: lanes 0–1 are row
/// `2p`, lanes 2–3 row `2p+1`. Windows 0 and 1 are the low nibbles (weights 0–15) and
/// their high nibbles (16–31) of the same bytes, so both of the block's halves are
/// computed from the same loads, as in [`dot_q5_0`].
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q5_08_group<const N: usize>(
    tile: &[u8],
    n_blk: usize,
    acts: &[QuantAct],
    out: &mut [f32],
) {
    assert!((1..=4).contains(&N), "one to four activation rows");
    assert!(
        out.len() >= N * ROWS,
        "the output holds eight rows per activation"
    );
    let lownib = vdupq_n_u8(0x0F);
    let mut facc = [[vdupq_n_f32(0.0); 2]; N];
    for t in 0..n_blk {
        let b = tile.as_ptr().add(t * Q5_08::SIZE);
        // [activation][logical group][row pair]
        let mut acc = [[[vdupq_n_s32(0); 4]; 4]; N];
        for h in 0..4 {
            let w = h % 2;
            let high = h >= 2;
            let mut av = [vdupq_n_s8(0); N];
            for u in 0..N {
                let p = acts[u].q.as_ptr().add(t * Q5_0_WEIGHTS + h * 8);
                av[u] = vcombine_s8(vld1_s8(p), vld1_s8(p));
            }
            let table = vld1q_u8(b.add(crate::kq::q5_qh() + h * 8));
            for pair in 0..4 {
                let v = vld1q_u8(b.add(crate::kq::q5_qs() + w * crate::kq::Q5_WINDOW + pair * 16));
                let nib = if high {
                    vshrq_n_u8(v, 4)
                } else {
                    vandq_u8(v, lownib)
                };
                let codes =
                    vreinterpretq_s8_u8(vorrq_u8(nib, spread_bits(table, &Q5_PAIR_IDX[pair])));
                for u in 0..N {
                    acc[u][h][pair] = sdot(acc[u][h][pair], codes, av[u]);
                }
            }
        }
        let dv = rows_f16(tile, t * Q5_08::SIZE + crate::kq::q5_d());
        for u in 0..N {
            let mut pairs = [vdupq_n_s32(0); 4];
            for (pair, p) in pairs.iter_mut().enumerate() {
                let mut s = acc[u][0][pair];
                s = vaddq_s32(s, acc[u][1][pair]);
                s = vaddq_s32(s, acc[u][2][pair]);
                s = vaddq_s32(s, acc[u][3][pair]);
                *p = s;
            }
            let s = rows_of(&pairs);
            let bias = vdupq_n_s32(16 * acts[u].sums[t]);
            let dk = vdupq_n_f32(acts[u].d[t]);
            for h in 0..2 {
                // `dot_q5_08_ref`'s fold, in its order: `(d·dk)` first, then one
                // fused multiply-add per block.
                let dd = vmulq_f32(dv[h], dk);
                let x = vcvtq_f32_s32(vsubq_s32(s[h], bias));
                facc[u][h] = vfmaq_f32(facc[u][h], dd, x);
            }
        }
    }
    for u in 0..N {
        vst1q_f32(out.as_mut_ptr().add(u * ROWS), facc[u][0]);
        vst1q_f32(out.as_mut_ptr().add(u * ROWS + 4), facc[u][1]);
    }
}

/// Eight output rows of a repacked Q5_0 tile against a batch of activations.
///
/// Q5_0's activation is the ordinary per-32 layout, so the batched and single-row
/// paths read the same bytes and there is one code path, as on x86.
///
/// # Safety
/// The CPU must have `asimddp`; the tile must hold `n_blk` blocks and `out` must hold
/// `nb · 8` floats.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q5_08_gemm(tile: &[u8], n_blk: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!(
        !acts.is_empty() && nb <= acts.len(),
        "more activations than rows"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    assert!(
        tile.len() >= n_blk * Q5_08::SIZE,
        "the tile is shorter than its blocks"
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
/// # Safety
/// As [`q5_08_gemm`].
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q5_08_gemv(tile: &[u8], n_blk: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!(
        (1..=3).contains(&nb),
        "the Q5_0 GEMV takes one to three rows"
    );
    q5_08_gemm(tile, n_blk, acts, nb, out);
}

// ---------------------------------------------------------------------------
// Q4_K: the per-row kernel, against the per-256 activation layout
// ---------------------------------------------------------------------------

/// One Q4_K row against one activation: [`crate::quant::dot_scalar`]'s expression, vectorised.
///
/// The super-block's eight sub-block products are summed in int32 with their 6-bit scales, the min
/// term in int32 with the activation's run sums, and the two are folded in the reference's order.
///
/// # Safety
/// The CPU must have `asimddp`; `w` must be `row_bytes(Q4_K, a.n)` long, the activation must carry
/// the per-256 layout, and the width must be a whole number of super-blocks.
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k(w: &[u8], a: &QuantAct) -> f32 {
    debug_assert_eq!(w.len(), crate::quant::row_bytes(Format::Q4_K, a.n));
    assert!(
        !a.qk.is_empty(),
        "a k-quant row needs a multiple-of-256 activation"
    );
    let mut acc = 0.0f32;
    for (sb, blk) in w.chunks_exact(144).enumerate() {
        let d = f16_at(blk, 0);
        let dmin = f16_at(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let aq = a.qk.as_ptr().add(sb * 256);
        let bs = &a.bsums[sb * 16..sb * 16 + 16];
        let mut sumi = 0i32;
        let mut summ = 0i32;
        // Unrolled for the same reason as the eight-row blocks of the Q8_0 kernel: with the
        // `asm!` in the body the loop would not unroll, and `j % 2` would stay a branch.
        macro_rules! sub {
            ($j:literal) => {{
                let (sc, m) = q4k_scale_min($j, scales);
                let p = qs.as_ptr().add(($j / 2) * 32);
                let a0 = vld1q_s8(aq.add($j * 32));
                let a1 = vld1q_s8(aq.add($j * 32 + 16));
                let s = if $j % 2 == 0 {
                    let lo_mask = vdupq_n_u8(0x0F);
                    dot32_sum(
                        vreinterpretq_s8_u8(vandq_u8(vld1q_u8(p), lo_mask)),
                        vreinterpretq_s8_u8(vandq_u8(vld1q_u8(p.add(16)), lo_mask)),
                        a0,
                        a1,
                    )
                } else {
                    dot32_sum(
                        vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(p), 4)),
                        vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(p.add(16)), 4)),
                        a0,
                        a1,
                    )
                };
                sumi += sc as i32 * s;
                summ += m as i32 * (bs[2 * $j] as i32 + bs[2 * $j + 1] as i32);
            }};
        }
        sub!(0);
        sub!(1);
        sub!(2);
        sub!(3);
        sub!(4);
        sub!(5);
        sub!(6);
        sub!(7);
        // The reference's arithmetic, in its order: two multiplies, a subtract, one multiply-add.
        acc += a.dk[sb] * (d * sumi as f32 - dmin * summ as f32);
    }
    acc
}

/// ggml's `get_scale_min_k4`: the six-bit scale and min of sub-block `j`.
#[inline(always)]
fn q4k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

// ---------------------------------------------------------------------------
// Q6_K: a super-block of six-bit codes, eight-bit scales and one f16 scale
// ---------------------------------------------------------------------------

/// 16 signed codes against 16 signed activations, one int32 sum.
///
/// # Safety
/// The CPU must have `asimddp`. Both vectors are read whole.
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn dot16_sum(x: int8x16_t, a: int8x16_t) -> i32 {
    let mut t = vdupq_n_s32(0);
    std::arch::asm!(
        "sdot {t:v}.4s, {x:v}.16b, {a:v}.16b",
        t = inout(vreg) t,
        x = in(vreg) x,
        a = in(vreg) a,
        options(pure, nomem, nostack),
    );
    hsum_s32x4(t)
}

/// One Q6_K row against one activation: [`crate::quant::dot_scalar`]'s expression.
///
/// The 6-bit codes are four bits in `ql` and two in `qh`; the two low bits are selected
/// by the group (0–3) and OR'd into the nibble at bit 4. The offset is 32, applied once
/// per sixteen-weight run from the activation's run sums, exactly as the reference does.
///
/// # Safety
/// The CPU must have `asimddp`; `w` must be `row_bytes(Q6_K, a.n)` long and the
/// activation must carry the per-256 layout.
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q6_k(w: &[u8], a: &QuantAct) -> f32 {
    debug_assert_eq!(w.len(), crate::quant::row_bytes(Format::Q6_K, a.n));
    assert!(
        !a.qk.is_empty(),
        "a k-quant row needs a multiple-of-256 activation"
    );
    let lownib = vdupq_n_u8(0x0F);
    let three = vdupq_n_u8(3);
    let mut acc = 0.0f32;
    for (sb, blk) in w.chunks_exact(210).enumerate() {
        let p = blk.as_ptr();
        let d = f16_at(blk, 208);
        let mut sumi = 0i32;
        for c in 0..2 {
            let ql = p.add(c * 64);
            let qh = p.add(128 + c * 32);
            let sc = p.add(192 + c * 8) as *const i8;
            let obase = sb * 256 + c * 128;
            for g in 0..4 {
                let nib_at = if g % 2 == 0 { 0 } else { 32 };
                let high = g >= 2;
                for hf in 0..2 {
                    let o = obase + g * 32 + hf * 16;
                    let qs = vld1q_u8(ql.add(nib_at + hf * 16));
                    let nib = if high {
                        vshrq_n_u8(qs, 4)
                    } else {
                        vandq_u8(qs, lownib)
                    };
                    let bits = vandq_u8(
                        vshlq_u8(vld1q_u8(qh.add(hf * 16)), vdupq_n_s8(-2 * g as i8)),
                        three,
                    );
                    let codes = vorrq_u8(nib, vshlq_n_u8(bits, 4));
                    let s = dot16_sum(vreinterpretq_s8_u8(codes), vld1q_s8(a.qk.as_ptr().add(o)));
                    let scale = *sc.add(2 * g + hf) as i32;
                    let bias = 32 * a.bsums[o / 16] as i32;
                    sumi += scale * (s - bias);
                }
            }
        }
        // The reference's expression, in its order: two multiplies and an add.
        acc += d * a.dk[sb] * sumi as f32;
    }
    acc
}

// ---------------------------------------------------------------------------
// The repacked eight-row tile: GEMV and GEMM
// ---------------------------------------------------------------------------

/// Where one activation row of a tile multiply reads its window, run sums and scale.
///
/// The GEMM reads the interleaved four-row buffer and the GEMV the plain per-row layout; the
/// arithmetic does not know which, so the fold and the row's bits are the same either way.
trait ActSrc {
    /// The eight activations of window `w` of super-block `sb`, duplicated into both halves.
    ///
    /// # Safety
    /// The CPU must have `asimddp`; `row` must be a row the source holds.
    unsafe fn act8(&self, row: usize, sb: usize, w: usize) -> int8x16_t;

    /// The sum of the 32 activations of sub-block `j` of super-block `sb`.
    ///
    /// # Safety
    /// As [`ActSrc::act8`].
    unsafe fn sum32(&self, row: usize, sb: usize, j: usize) -> i32;

    /// The sum of the 16 activations of run `run` (0–15) of super-block `sb`.
    ///
    /// # Safety
    /// As [`ActSrc::act8`].
    unsafe fn sum16(&self, row: usize, sb: usize, run: usize) -> i32;

    /// The activation's per-256 scale.
    ///
    /// # Safety
    /// As [`ActSrc::act8`].
    unsafe fn dk(&self, row: usize, sb: usize) -> f32;
}

/// The plain per-row activation of the decode path.
struct Plain<'a> {
    acts: &'a [QuantAct],
}

impl ActSrc for Plain<'_> {
    #[inline(always)]
    unsafe fn act8(&self, row: usize, sb: usize, w: usize) -> int8x16_t {
        let p = self.acts[row].qk.as_ptr().add(sb * 256 + w * 8);
        let v = vld1_s8(p);
        vcombine_s8(v, v)
    }

    #[inline(always)]
    unsafe fn sum32(&self, row: usize, sb: usize, j: usize) -> i32 {
        let bs = &self.acts[row].bsums[sb * 16..sb * 16 + 16];
        bs[2 * j] as i32 + bs[2 * j + 1] as i32
    }

    #[inline(always)]
    unsafe fn sum16(&self, row: usize, sb: usize, run: usize) -> i32 {
        self.acts[row].bsums[sb * 16 + run] as i32
    }

    #[inline(always)]
    unsafe fn dk(&self, row: usize, sb: usize) -> f32 {
        self.acts[row].dk[sb]
    }
}

/// The interleaved four-row activation of the prefill path.
struct Interleaved<'a> {
    acts: &'a KqActs,
}

impl ActSrc for Interleaved<'_> {
    #[inline(always)]
    unsafe fn act8(&self, row: usize, sb: usize, w: usize) -> int8x16_t {
        let g = row / ACT_GROUP;
        let t = row % ACT_GROUP;
        let base = (g * self.acts.super_blocks() + sb) * 1024 + w * 32 + t * 8;
        let v = vld1_s8(self.acts.q.as_ptr().add(base));
        vcombine_s8(v, v)
    }

    #[inline(always)]
    unsafe fn sum32(&self, row: usize, sb: usize, j: usize) -> i32 {
        let g = row / ACT_GROUP;
        let t = row % ACT_GROUP;
        self.acts.bs[(g * self.acts.super_blocks() + sb) * 32 + t * 8 + j]
    }

    #[inline(always)]
    unsafe fn sum16(&self, row: usize, sb: usize, run: usize) -> i32 {
        let g = row / ACT_GROUP;
        let t = row % ACT_GROUP;
        self.acts.bs16[(g * self.acts.super_blocks() + sb) * ACT_GROUP * 16 + t * 16 + run]
    }

    #[inline(always)]
    unsafe fn dk(&self, row: usize, sb: usize) -> f32 {
        let g = row / ACT_GROUP;
        let t = row % ACT_GROUP;
        self.acts.d[(g * self.acts.super_blocks() + sb) * ACT_GROUP + t]
    }
}

/// The eight rows' f16 fields of one super-block, widened to two `f32x4`s.
#[inline(always)]
unsafe fn rows_f16(t: &[u8], at: usize) -> [float32x4_t; 2] {
    let mut v = [0.0f32; 8];
    for (r, x) in v.iter_mut().enumerate() {
        *x = f16_at(t, at + r * 2);
    }
    [vld1q_f32(v.as_ptr()), vld1q_f32(v.as_ptr().add(4))]
}

/// The eight rows' six-bit scales or mins of sub-block `j`, widened to two `i32x4`s.
#[inline(always)]
unsafe fn six_bit(t: &[u8], j: usize, mins: bool) -> [int32x4_t; 2] {
    let mut v = [0u8; 8];
    for (r, x) in v.iter_mut().enumerate() {
        *x = if mins {
            tile_q4k_min(t, j, r)
        } else {
            tile_q4k_scale(t, j, r)
        };
    }
    let u = vmovl_u8(vld1_u8(v.as_ptr()));
    [
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u))),
    ]
}

/// One super-block of a tile against `N` activation rows, folded into `facc`.
///
/// The window decode is shared by both nibble halves — a window's low nibbles are sub-block `2g`,
/// its high nibbles `2g+1` — and by all `N` activation rows. The four row-pairs' int32 sums are
/// widened to the eight rows, scaled per sub-block, and the super-block's one float fold is
/// `kq::dot_q4k8_ref`'s `mul_add` chain, so the result is bit-identical to it.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4k8_sb<S: ActSrc, const N: usize>(
    t: &[u8],
    sb: usize,
    src: &S,
    row0: usize,
    dv: [float32x4_t; 2],
    dminv: [float32x4_t; 2],
    facc: &mut [[float32x4_t; 2]; N],
) {
    let qs = t.as_ptr().add(q4_qs());
    let lo_mask = vdupq_n_u8(0x0F);
    let mut sumi = [[vdupq_n_s32(0); 2]; N];
    let mut summ = [[vdupq_n_s32(0); 2]; N];
    for g in 0..4 {
        let j0 = 2 * g;
        let j1 = j0 + 1;
        // [activation row][sub-block][row pair]
        let mut acc = [[[vdupq_n_s32(0); 4]; 2]; N];
        for w in 0..4 {
            let mut act = [[vdupq_n_s8(0); 2]; N];
            for u in 0..N {
                act[u][0] = src.act8(row0 + u, sb, j0 * 4 + w);
                act[u][1] = src.act8(row0 + u, sb, j1 * 4 + w);
            }
            let base = (4 * g + w) * 64;
            for p in 0..4 {
                let v = vld1q_u8(qs.add(base + p * 16));
                let lo = vreinterpretq_s8_u8(vandq_u8(v, lo_mask));
                let hi = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
                for u in 0..N {
                    acc[u][0][p] = sdot(acc[u][0][p], lo, act[u][0]);
                    acc[u][1][p] = sdot(acc[u][1][p], hi, act[u][1]);
                }
            }
        }
        let sc0 = six_bit(t, j0, false);
        let sc1 = six_bit(t, j1, false);
        let mn0 = six_bit(t, j0, true);
        let mn1 = six_bit(t, j1, true);
        for u in 0..N {
            let s0 = rows_of(&acc[u][0]);
            let s1 = rows_of(&acc[u][1]);
            sumi[u][0] = vmlaq_s32(sumi[u][0], sc0[0], s0[0]);
            sumi[u][1] = vmlaq_s32(sumi[u][1], sc0[1], s0[1]);
            sumi[u][0] = vmlaq_s32(sumi[u][0], sc1[0], s1[0]);
            sumi[u][1] = vmlaq_s32(sumi[u][1], sc1[1], s1[1]);
            let a0 = vdupq_n_s32(src.sum32(row0 + u, sb, j0));
            let a1 = vdupq_n_s32(src.sum32(row0 + u, sb, j1));
            summ[u][0] = vmlaq_s32(summ[u][0], mn0[0], a0);
            summ[u][1] = vmlaq_s32(summ[u][1], mn0[1], a0);
            summ[u][0] = vmlaq_s32(summ[u][0], mn1[0], a1);
            summ[u][1] = vmlaq_s32(summ[u][1], mn1[1], a1);
        }
    }
    for u in 0..N {
        let dk = vdupq_n_f32(src.dk(row0 + u, sb));
        for h in 0..2 {
            let sf = vcvtq_f32_s32(sumi[u][h]);
            let mf = vcvtq_f32_s32(summ[u][h]);
            // (-dmin)·summ + d·sumi, one rounding for the product, one for the fused add.
            let x = vfmsq_f32(vmulq_f32(dv[h], sf), dminv[h], mf);
            facc[u][h] = vfmaq_f32(facc[u][h], x, dk);
        }
    }
}

/// The eight rows' sums from the four row-pair vectors (rows 2p, 2p+1 in each pair's two lanes).
#[inline(always)]
unsafe fn rows_of(acc: &[int32x4_t; 4]) -> [int32x4_t; 2] {
    let p0 = vpadd_s32(vget_low_s32(acc[0]), vget_high_s32(acc[0]));
    let p1 = vpadd_s32(vget_low_s32(acc[1]), vget_high_s32(acc[1]));
    let p2 = vpadd_s32(vget_low_s32(acc[2]), vget_high_s32(acc[2]));
    let p3 = vpadd_s32(vget_low_s32(acc[3]), vget_high_s32(acc[3]));
    [vcombine_s32(p0, p1), vcombine_s32(p2, p3)]
}

/// Eight output rows of a repacked Q4_K tile against `N` activation rows.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4k8_group<S: ActSrc, const N: usize>(
    tile: &[u8],
    n_sb: usize,
    src: &S,
    row0: usize,
    out: &mut [f32],
) {
    assert!((1..=4).contains(&N), "one to four activation rows");
    assert!(
        out.len() >= N * ROWS,
        "the output holds eight rows per activation"
    );
    let mut facc = [[vdupq_n_f32(0.0); 2]; N];
    for sb in 0..n_sb {
        let t = &tile[sb * Q4K8::SIZE..];
        q4k8_sb::<S, N>(
            t,
            sb,
            src,
            row0,
            rows_f16(t, q4_d()),
            rows_f16(t, q4_dmin()),
            &mut facc,
        );
    }
    for u in 0..N {
        vst1q_f32(out.as_mut_ptr().add(u * ROWS), facc[u][0]);
        vst1q_f32(out.as_mut_ptr().add(u * ROWS + 4), facc[u][1]);
    }
}

/// Eight output rows of a repacked Q4_K tile against a batch of activations.
///
/// # Safety
/// The CPU must have `asimddp`. The tile must hold `n_sb` super-blocks, the activation must carry
/// the per-256 layout, and `out` must hold `nb · 8` floats.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q4k8_gemm(tile: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    assert!(
        tile.len() >= n_sb * Q4K8::SIZE,
        "the tile is shorter than its super-blocks"
    );
    let src = Interleaved { acts };
    let mut b0 = 0;
    while nb - b0 >= 4 {
        q4k8_group::<_, 4>(tile, n_sb, &src, b0, &mut out[b0 * ROWS..]);
        b0 += 4;
    }
    while b0 < nb {
        q4k8_group::<_, 1>(tile, n_sb, &src, b0, &mut out[b0 * ROWS..]);
        b0 += 1;
    }
}

/// Eight output rows of a repacked Q4_K tile against one to three activations.
///
/// # Safety
/// As [`q4k8_gemm`], with one to three activations in the plain per-row layout.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q4k8_gemv(tile: &[u8], n_sb: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!((1..=3).contains(&nb), "the GEMV takes one to three rows");
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    assert!(
        tile.len() >= n_sb * Q4K8::SIZE,
        "the tile is shorter than its super-blocks"
    );
    let src = Plain { acts };
    for b in 0..nb {
        q4k8_group::<_, 1>(tile, n_sb, &src, b, &mut out[b * ROWS..]);
    }
}

// ---------------------------------------------------------------------------
// The i8mm tier: the repacked tile GEMM on `smmla` (S2.8 F2)
// ---------------------------------------------------------------------------

/// `smmla`: the 2×8 signed tile in `a` times the 8×2 signed tile in `b`, accumulated
/// into a 2×2 int32 result whose lanes are `[r0·c0, r1·c0, r0·c1, r1·c1]`.
///
/// The intrinsic is newer than the MSRV, so the instruction is written out, as `sdot`
/// is; the macro expands into the caller's `#[target_feature(enable = "i8mm")]`.
///
/// # Safety
/// The CPU must have `i8mm`.
#[inline(always)]
unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut t = acc;
    std::arch::asm!(
        "smmla {t:v}.4s, {a:v}.16b, {b:v}.16b",
        t = inout(vreg) t,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    t
}

/// `[a, b, a, b]`, the per-output-row scale or min of a row pair, duplicated across the
/// two activations the `smmla` result holds.
#[inline(always)]
unsafe fn pair_i32(a: i32, b: i32) -> int32x4_t {
    let v = [a, b, a, b];
    vld1q_s32(v.as_ptr())
}

/// `[a, b, a, b]` for a float field.
#[inline(always)]
unsafe fn pair_f32(a: f32, b: f32) -> float32x4_t {
    let v = [a, b, a, b];
    vld1q_f32(v.as_ptr())
}

/// `[a, a, b, b]`, one value per output row, duplicated across the two activations.
#[inline(always)]
unsafe fn quad_i32(a: i32, b: i32) -> int32x4_t {
    let v = [a, a, b, b];
    vld1q_s32(v.as_ptr())
}

/// `[a, a, b, b]` for a float field.
#[inline(always)]
unsafe fn quad_f32(a: f32, b: f32) -> float32x4_t {
    let v = [a, a, b, b];
    vld1q_f32(v.as_ptr())
}

/// One super-block of a repacked Q4_K tile against four activation rows (two pairs) on
/// `smmla`, folded onto `facc[activation_pair][row_pair]`.
///
/// The tile's window is two output rows of eight codes, which is `smmla`'s first
/// operand; two activation rows' eight codes are contiguous in the interleaved buffer,
/// which is its second. One instruction therefore forms the 32 products of two output
/// rows by two activations, and four instructions cover the eight output rows. The
/// integers are exact and the scale/min application and the float fold keep
/// [`kq::dot_q4k8_ref`]'s order, so a row is bit-identical to the `sdot` GEMM/GEMV.
#[inline]
#[target_feature(enable = "neon,i8mm")]
unsafe fn q4k8_sb_i8mm(
    t: &[u8],
    sb: usize,
    acts: &KqActs,
    group: usize,
    d: &[f32; 8],
    dmin: &[f32; 8],
    facc: &mut [[float32x4_t; 4]; 2],
) {
    let src = Interleaved { acts };
    let lo_mask = vdupq_n_u8(0x0F);
    let qs = t.as_ptr().add(q4_qs());
    let n_sb = acts.super_blocks();
    // [sub-block j][activation pair][row pair], widened to the eight rows below.
    let mut sumi = [[vdupq_n_s32(0); 4]; 2];
    let mut summ = [[vdupq_n_s32(0); 4]; 2];
    for j in 0..8 {
        let hi = j % 2 == 1;
        let g = j / 2;
        let mut acc = [[vdupq_n_s32(0); 4]; 2];
        for w in 0..4 {
            let window = 4 * g + w;
            let at = (group * n_sb + sb) * 1024 + (j * 4 + w) * 32;
            let b0 = vld1q_s8(acts.q.as_ptr().add(at));
            let b1 = vld1q_s8(acts.q.as_ptr().add(at + 16));
            for p in 0..4 {
                let v = vld1q_u8(qs.add(window * 64 + p * 16));
                let codes = if hi {
                    vreinterpretq_s8_u8(vshrq_n_u8(v, 4))
                } else {
                    vreinterpretq_s8_u8(vandq_u8(v, lo_mask))
                };
                acc[0][p] = smmla(acc[0][p], codes, b0);
                acc[1][p] = smmla(acc[1][p], codes, b1);
            }
        }
        let sc = six_bit(t, j, false);
        let mn = six_bit(t, j, true);
        let mut sc_l = [[0i32; 4]; 2];
        let mut mn_l = [[0i32; 4]; 2];
        vst1q_s32(sc_l[0].as_mut_ptr(), sc[0]);
        vst1q_s32(sc_l[1].as_mut_ptr(), sc[1]);
        vst1q_s32(mn_l[0].as_mut_ptr(), mn[0]);
        vst1q_s32(mn_l[1].as_mut_ptr(), mn[1]);
        for q in 0..2 {
            let (a0, a1) = (group * ACT_GROUP + 2 * q, group * ACT_GROUP + 2 * q + 1);
            let bv = [
                src.sum32(a0, sb, j),
                src.sum32(a1, sb, j),
                src.sum32(a0, sb, j),
                src.sum32(a1, sb, j),
            ];
            let bv = vld1q_s32(bv.as_ptr());
            for p in 0..4 {
                let (sr, sl_, mr, ml_) = if p < 2 {
                    (
                        sc_l[0][2 * p],
                        sc_l[0][2 * p + 1],
                        mn_l[0][2 * p],
                        mn_l[0][2 * p + 1],
                    )
                } else {
                    (
                        sc_l[1][2 * p - 4],
                        sc_l[1][2 * p - 3],
                        mn_l[1][2 * p - 4],
                        mn_l[1][2 * p - 3],
                    )
                };
                let sp = quad_i32(sr, sl_);
                let mp = quad_i32(mr, ml_);
                sumi[q][p] = vmlaq_s32(sumi[q][p], sp, acc[q][p]);
                summ[q][p] = vmlaq_s32(summ[q][p], mp, bv);
            }
        }
    }
    for q in 0..2 {
        let (a0, a1) = (group * ACT_GROUP + 2 * q, group * ACT_GROUP + 2 * q + 1);
        let dk = {
            let v = [
                src.dk(a0, sb),
                src.dk(a1, sb),
                src.dk(a0, sb),
                src.dk(a1, sb),
            ];
            vld1q_f32(v.as_ptr())
        };
        for p in 0..4 {
            // The accumulator's lanes are [row0·act0, row0·act1, row1·act0, row1·act1].
            let (r, l) = (2 * p, 2 * p + 1);
            let dp = quad_f32(d[r], d[l]);
            let dminp = quad_f32(dmin[r], dmin[l]);
            let sf = vcvtq_f32_s32(sumi[q][p]);
            let mf = vcvtq_f32_s32(summ[q][p]);
            // (-dmin)·summ + d·sumi, one rounding for the product, one for the fused add.
            let x = vfmsq_f32(vmulq_f32(dp, sf), dminp, mf);
            facc[q][p] = vfmaq_f32(facc[q][p], x, dk);
        }
    }
}

/// The i8mm preview: eight output rows of a repacked Q4_K tile against four
/// activations, on `smmla`.
///
/// Callers pass one interleaved activation group (`nb` is a multiple of four and
/// four-aligned); the `sdot` kernel keeps the 1–3-row tail and the GEMV.
///
/// # Safety
/// The CPU must have `asimd`, `asimddp` and `i8mm`; the tile must hold `n_sb`
/// super-blocks and `out` must hold `nb · 8` floats.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn q4k8_gemm_i8mm(tile: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    assert!(
        tile.len() >= n_sb * Q4K8::SIZE,
        "the tile is shorter than its super-blocks"
    );
    let src = Interleaved { acts };
    let mut b0 = 0;
    while b0 + ACT_GROUP <= nb {
        let group = b0 / ACT_GROUP;
        let mut facc = [[vdupq_n_f32(0.0); 4]; 2];
        for sb in 0..n_sb {
            let t = &tile[sb * Q4K8::SIZE..];
            let dv = rows_f16(t, q4_d());
            let dmv = rows_f16(t, q4_dmin());
            let mut d8 = [0.0f32; 8];
            let mut dm8 = [0.0f32; 8];
            vst1q_f32(d8.as_mut_ptr(), dv[0]);
            vst1q_f32(d8.as_mut_ptr().add(4), dv[1]);
            vst1q_f32(dm8.as_mut_ptr(), dmv[0]);
            vst1q_f32(dm8.as_mut_ptr().add(4), dmv[1]);
            q4k8_sb_i8mm(t, sb, acts, group, &d8, &dm8, &mut facc);
        }
        for q in 0..2 {
            for p in 0..4 {
                let mut lanes = [0.0f32; 4];
                vst1q_f32(lanes.as_mut_ptr(), facc[q][p]);
                let a = b0 + 2 * q;
                out[a * ROWS + 2 * p] = lanes[0];
                out[(a + 1) * ROWS + 2 * p] = lanes[1];
                out[a * ROWS + 2 * p + 1] = lanes[2];
                out[(a + 1) * ROWS + 2 * p + 1] = lanes[3];
            }
        }
        b0 += ACT_GROUP;
    }
    while b0 < nb {
        q4k8_group::<_, 1>(tile, n_sb, &src, b0, &mut out[b0 * ROWS..]);
        b0 += 1;
    }
}

/// Eight output rows of a repacked Q5_0 tile against four activations on `smmla`.
///
/// Q5_0's activation is the ordinary per-32 layout, so the two activation rows of an
/// `smmla` operand are two eight-byte loads joined; the tile's window is already two
/// output rows of eight codes. The block's four logical groups accumulate into one
/// int32 vector per (activation pair, row pair); the bias and the `(d·dk)` fold keep
/// [`kq::dot_q5_08_ref`]'s order, so a row is bit-identical to the `sdot` GEMV.
///
/// # Safety
/// The CPU must have `asimd`, `asimddp` and `i8mm`; `acts` holds at least four rows and
/// `out` four times eight floats.
#[target_feature(enable = "neon,i8mm")]
unsafe fn q5_08_group_i8mm(tile: &[u8], n_blk: usize, acts: &[QuantAct], out: &mut [f32]) {
    let lownib = vdupq_n_u8(0x0F);
    let mut facc = [[vdupq_n_f32(0.0); 4]; 2];
    for t in 0..n_blk {
        let b = tile.as_ptr().add(t * Q5_08::SIZE);
        let mut acc = [[vdupq_n_s32(0); 4]; 2];
        for h in 0..4 {
            let w = h % 2;
            let high = h >= 2;
            let mut bv = [vdupq_n_s8(0); 2];
            for (q, av) in bv.iter_mut().enumerate() {
                let p0 = acts[2 * q].q.as_ptr().add(t * Q5_0_WEIGHTS + h * 8);
                let p1 = acts[2 * q + 1].q.as_ptr().add(t * Q5_0_WEIGHTS + h * 8);
                *av = vcombine_s8(vld1_s8(p0), vld1_s8(p1));
            }
            let table = vld1q_u8(b.add(crate::kq::q5_qh() + h * 8));
            for pair in 0..4 {
                let v = vld1q_u8(b.add(crate::kq::q5_qs() + w * crate::kq::Q5_WINDOW + pair * 16));
                let nib = if high {
                    vshrq_n_u8(v, 4)
                } else {
                    vandq_u8(v, lownib)
                };
                let codes =
                    vreinterpretq_s8_u8(vorrq_u8(nib, spread_bits(table, &Q5_PAIR_IDX[pair])));
                for q in 0..2 {
                    acc[q][pair] = smmla(acc[q][pair], codes, bv[q]);
                }
            }
        }
        let dv = rows_f16(tile, t * Q5_08::SIZE + crate::kq::q5_d());
        let mut d8 = [0.0f32; 8];
        vst1q_f32(d8.as_mut_ptr(), dv[0]);
        vst1q_f32(d8.as_mut_ptr().add(4), dv[1]);
        for q in 0..2 {
            let (u0, u1) = (2 * q, 2 * q + 1);
            let bias = pair_i32(16 * acts[u0].sums[t], 16 * acts[u1].sums[t]);
            let dk = pair_f32(acts[u0].d[t], acts[u1].d[t]);
            for pair in 0..4 {
                let s = acc[q][pair];
                let x = vcvtq_f32_s32(vsubq_s32(s, bias));
                let dp = quad_f32(d8[2 * pair], d8[2 * pair + 1]);
                let dd = vmulq_f32(dp, dk);
                facc[q][pair] = vfmaq_f32(facc[q][pair], dd, x);
            }
        }
    }
    for q in 0..2 {
        for pair in 0..4 {
            let mut lanes = [0.0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), facc[q][pair]);
            let a = 2 * q;
            out[a * ROWS + 2 * pair] = lanes[0];
            out[(a + 1) * ROWS + 2 * pair] = lanes[1];
            out[a * ROWS + 2 * pair + 1] = lanes[2];
            out[(a + 1) * ROWS + 2 * pair + 1] = lanes[3];
        }
    }
}

/// Eight output rows of a repacked Q5_0 tile against a batch of activations, on
/// `smmla` for whole groups of four and on the `sdot` kernel for the 1–3-row tail.
///
/// # Safety
/// The CPU must have `asimd`, `asimddp` and `i8mm`; the tile must hold `n_blk` blocks
/// and `out` must hold `nb · 8` floats.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn q5_08_gemm_i8mm(
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
    assert!(
        tile.len() >= n_blk * Q5_08::SIZE,
        "the tile is shorter than its blocks"
    );
    let mut b = 0;
    while b + 4 <= nb {
        q5_08_group_i8mm(tile, n_blk, &acts[b..], &mut out[b * ROWS..]);
        b += 4;
    }
    match nb - b {
        3 => q5_08_group::<3>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        2 => q5_08_group::<2>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        1 => q5_08_group::<1>(tile, n_blk, &acts[b..], &mut out[b * ROWS..]),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// The repacked Q6_K tile: GEMV and GEMM
// ---------------------------------------------------------------------------

/// The eight rows' eight-bit scales of one sixteen-weight run, widened to two `i32x4`s.
#[inline(always)]
unsafe fn rows_i8(t: &[u8], at: usize) -> [int32x4_t; 2] {
    let v = vmovl_s8(vld1_s8(t.as_ptr().add(at) as *const i8));
    [vmovl_s16(vget_low_s16(v)), vmovl_s16(vget_high_s16(v))]
}

/// One super-block of a repacked Q6_K tile against `N` activation rows, folded onto `facc`.
///
/// The tile interleaves the eight rows exactly as the Q4_K tile does, so one sixteen-byte
/// load is two output rows of eight codes: lanes 0–1 are row `2p`, lanes 2–3 row `2p+1`.
/// Each run of sixteen weights spans two such windows (their `sdot`s share one pair of
/// accumulators), and the run's scale and offset are applied to the eight row sums in
/// int32 before the super-block's one float fold. Every integer is the reference's, so
/// the result is bit-identical to [`crate::quant::dot_scalar`].
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q6k8_sb<S: ActSrc, const N: usize>(
    t: &[u8],
    sb: usize,
    src: &S,
    row0: usize,
    dv: [float32x4_t; 2],
    facc: &mut [[float32x4_t; 2]; N],
) {
    let lownib = vdupq_n_u8(0x0F);
    let three = vdupq_n_u8(3);
    let mut sumi = [[vdupq_n_s32(0); 2]; N];
    for c in 0..2 {
        for k in 0..4 {
            let hf = k % 2;
            let g_lo = if k < 2 { 0 } else { 1 };
            let g_hi = g_lo + 2;
            let mut acc = [[[vdupq_n_s32(0); 4]; 2]; N];
            for ww in 0..2 {
                let qi = c * 8 + 2 * k + ww;
                let hi_qh = c * 4 + (2 * k) % 4 + ww;
                let mut av = [[vdupq_n_s8(0); 2]; N];
                for u in 0..N {
                    // The run's eight-weight half in the super-block's window order.
                    let w_lo = 16 * c + 4 * g_lo + 2 * hf + ww;
                    let w_hi = 16 * c + 4 * g_hi + 2 * hf + ww;
                    av[u][0] = src.act8(row0 + u, sb, w_lo);
                    av[u][1] = src.act8(row0 + u, sb, w_hi);
                }
                for pair in 0..4 {
                    // The qh window is row-major: the 16 bytes at `pair · 16` are the
                    // two rows' high bits, one byte per weight, so the code vector's
                    // halves already line up with the `sdot` operands.
                    let raw = vld1q_u8(t.as_ptr().add(crate::kq::q6_qh() + hi_qh * 64 + pair * 16));
                    let bits_lo = vshlq_n_u8(
                        vandq_u8(vshlq_u8(raw, vdupq_n_s8(-2 * g_lo as i8)), three),
                        4,
                    );
                    let bits_hi = vshlq_n_u8(
                        vandq_u8(vshlq_u8(raw, vdupq_n_s8(-2 * g_hi as i8)), three),
                        4,
                    );
                    let v = vld1q_u8(t.as_ptr().add(crate::kq::q6_ql() + qi * 64 + pair * 16));
                    let lo_nib = vandq_u8(v, lownib);
                    let hi_nib = vshrq_n_u8(v, 4);
                    let codes_lo = vreinterpretq_s8_u8(vorrq_u8(lo_nib, bits_lo));
                    let codes_hi = vreinterpretq_s8_u8(vorrq_u8(hi_nib, bits_hi));
                    for u in 0..N {
                        acc[u][0][pair] = sdot(acc[u][0][pair], codes_lo, av[u][0]);
                        acc[u][1][pair] = sdot(acc[u][1][pair], codes_hi, av[u][1]);
                    }
                }
            }
            let run_lo = c * 8 + 2 * g_lo + hf;
            let run_hi = c * 8 + 2 * g_hi + hf;
            let sc_lo = rows_i8(t, crate::kq::q6_sc() + run_lo * 8);
            let sc_hi = rows_i8(t, crate::kq::q6_sc() + run_hi * 8);
            for u in 0..N {
                let s_lo = rows_of(&acc[u][0]);
                let s_hi = rows_of(&acc[u][1]);
                let bias_lo = vdupq_n_s32(32 * src.sum16(row0 + u, sb, run_lo));
                let bias_hi = vdupq_n_s32(32 * src.sum16(row0 + u, sb, run_hi));
                for h in 0..2 {
                    sumi[u][h] = vmlaq_s32(sumi[u][h], sc_lo[h], vsubq_s32(s_lo[h], bias_lo));
                    sumi[u][h] = vmlaq_s32(sumi[u][h], sc_hi[h], vsubq_s32(s_hi[h], bias_hi));
                }
            }
        }
    }
    for u in 0..N {
        let dk = vdupq_n_f32(src.dk(row0 + u, sb));
        for h in 0..2 {
            // `dot_q6k8_ref`'s fold, in its order: two multiplies, then an add.
            let x = vmulq_f32(vmulq_f32(dv[h], dk), vcvtq_f32_s32(sumi[u][h]));
            facc[u][h] = vaddq_f32(facc[u][h], x);
        }
    }
}

/// Eight output rows of a repacked Q6_K tile against `N` activation rows.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q6k8_group<S: ActSrc, const N: usize>(
    tile: &[u8],
    n_sb: usize,
    src: &S,
    row0: usize,
    out: &mut [f32],
) {
    assert!((1..=4).contains(&N), "one to four activation rows");
    assert!(
        out.len() >= N * ROWS,
        "the output holds eight rows per activation"
    );
    let mut facc = [[vdupq_n_f32(0.0); 2]; N];
    for sb in 0..n_sb {
        let t = &tile[sb * Q6K8::SIZE..];
        q6k8_sb::<S, N>(t, sb, src, row0, rows_f16(t, crate::kq::q6_d()), &mut facc);
    }
    for u in 0..N {
        vst1q_f32(out.as_mut_ptr().add(u * ROWS), facc[u][0]);
        vst1q_f32(out.as_mut_ptr().add(u * ROWS + 4), facc[u][1]);
    }
}

/// Eight output rows of a repacked Q6_K tile against a batch of interleaved activations.
///
/// # Safety
/// The CPU must have `asimddp`; the tile must hold `n_sb` super-blocks and `out` must
/// hold `nb · 8` floats.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q6k8_gemm(tile: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    assert!(
        tile.len() >= n_sb * Q6K8::SIZE,
        "the tile is shorter than its super-blocks"
    );
    let src = Interleaved { acts };
    let mut b0 = 0;
    while nb - b0 >= 4 {
        q6k8_group::<_, 4>(tile, n_sb, &src, b0, &mut out[b0 * ROWS..]);
        b0 += 4;
    }
    while b0 < nb {
        q6k8_group::<_, 1>(tile, n_sb, &src, b0, &mut out[b0 * ROWS..]);
        b0 += 1;
    }
}

/// Eight output rows of a repacked Q6_K tile against one to three plain activations.
///
/// # Safety
/// As [`q6k8_gemm`].
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q6k8_gemv(tile: &[u8], n_sb: usize, acts: &[QuantAct], nb: usize, out: &mut [f32]) {
    assert!(
        (1..=3).contains(&nb),
        "the Q6_K GEMV takes one to three rows"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    assert!(
        tile.len() >= n_sb * Q6K8::SIZE,
        "the tile is shorter than its super-blocks"
    );
    let src = Plain { acts };
    for b in 0..nb {
        q6k8_group::<_, 1>(tile, n_sb, &src, b, &mut out[b * ROWS..]);
    }
}

/// One super-block of a repacked Q6_K tile against four activation rows (two pairs) on
/// `smmla`, folded onto `facc[activation_pair][row_pair]`.
///
/// Each sixteen-weight run is two eight-weight windows, so a run's integer sum is two
/// `smmla`s per (activation pair, row pair); the run's int8 scale and its
/// `32·bsum` offset are applied to the eight row sums in int32 before the super-block's
/// one float fold, in [`kq::dot_q6k8_ref`]'s order, so a row is bit-identical to the
/// `sdot` GEMV.
#[inline]
#[target_feature(enable = "neon,i8mm")]
unsafe fn q6k8_sb_i8mm(
    t: &[u8],
    sb: usize,
    acts: &KqActs,
    group: usize,
    d: &[f32; 8],
    facc: &mut [[float32x4_t; 4]; 2],
) {
    let src = Interleaved { acts };
    let lownib = vdupq_n_u8(0x0F);
    let three = vdupq_n_u8(3);
    let n_sb = acts.super_blocks();
    let mut sumi = [[vdupq_n_s32(0); 4]; 2];
    for c in 0..2 {
        for k in 0..4 {
            let hf = k % 2;
            let g_lo = if k < 2 { 0 } else { 1 };
            for half in 0..2 {
                let g = g_lo + 2 * half;
                let run = c * 8 + 2 * g + hf;
                let w0 = 16 * c + 4 * g + 2 * hf;
                let mut acc = [[vdupq_n_s32(0); 4]; 2];
                for ww in 0..2 {
                    let qi = c * 8 + 2 * k + ww;
                    let hi_qh = c * 4 + (2 * k) % 4 + ww;
                    let at = (group * n_sb + sb) * 1024 + (w0 + ww) * 32;
                    let b0 = vld1q_s8(acts.q.as_ptr().add(at));
                    let b1 = vld1q_s8(acts.q.as_ptr().add(at + 16));
                    for pair in 0..4 {
                        let raw =
                            vld1q_u8(t.as_ptr().add(crate::kq::q6_qh() + hi_qh * 64 + pair * 16));
                        let bits =
                            vshlq_n_u8(vandq_u8(vshlq_u8(raw, vdupq_n_s8(-2 * g as i8)), three), 4);
                        let v = vld1q_u8(t.as_ptr().add(crate::kq::q6_ql() + qi * 64 + pair * 16));
                        let nib = if half == 1 {
                            vshrq_n_u8(v, 4)
                        } else {
                            vandq_u8(v, lownib)
                        };
                        let codes = vreinterpretq_s8_u8(vorrq_u8(nib, bits));
                        acc[0][pair] = smmla(acc[0][pair], codes, b0);
                        acc[1][pair] = smmla(acc[1][pair], codes, b1);
                    }
                }
                let sc = rows_i8(t, crate::kq::q6_sc() + run * 8);
                let mut sc_l = [[0i32; 4]; 2];
                vst1q_s32(sc_l[0].as_mut_ptr(), sc[0]);
                vst1q_s32(sc_l[1].as_mut_ptr(), sc[1]);
                for q in 0..2 {
                    let (a0, a1) = (group * ACT_GROUP + 2 * q, group * ACT_GROUP + 2 * q + 1);
                    let bias = pair_i32(32 * src.sum16(a0, sb, run), 32 * src.sum16(a1, sb, run));
                    for pair in 0..4 {
                        let (r, l) = (2 * pair, 2 * pair + 1);
                        let (sr, sl_) = if pair < 2 {
                            (sc_l[0][r], sc_l[0][l])
                        } else {
                            (sc_l[1][r - 4], sc_l[1][l - 4])
                        };
                        let sp = quad_i32(sr, sl_);
                        sumi[q][pair] = vmlaq_s32(sumi[q][pair], sp, vsubq_s32(acc[q][pair], bias));
                    }
                }
            }
        }
    }
    for q in 0..2 {
        let (a0, a1) = (group * ACT_GROUP + 2 * q, group * ACT_GROUP + 2 * q + 1);
        let dk = pair_f32(src.dk(a0, sb), src.dk(a1, sb));
        for pair in 0..4 {
            let (r, l) = (2 * pair, 2 * pair + 1);
            let dp = quad_f32(d[r], d[l]);
            // `dot_q6k8_ref`'s fold, in its order: two multiplies, then an add.
            let x = vmulq_f32(vmulq_f32(dp, dk), vcvtq_f32_s32(sumi[q][pair]));
            facc[q][pair] = vaddq_f32(facc[q][pair], x);
        }
    }
}

/// Eight output rows of a repacked Q6_K tile against a batch of activations, on
/// `smmla` for whole groups of four and on the `sdot` kernel for the 1–3-row tail.
///
/// # Safety
/// The CPU must have `asimd`, `asimddp` and `i8mm`; the tile must hold `n_sb`
/// super-blocks and `out` must hold `nb · 8` floats.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn q6k8_gemm_i8mm(tile: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    assert!(
        tile.len() >= n_sb * Q6K8::SIZE,
        "the tile is shorter than its super-blocks"
    );
    let src = Interleaved { acts };
    let mut b0 = 0;
    while b0 + 4 <= nb {
        let group = b0 / ACT_GROUP;
        let mut facc = [[vdupq_n_f32(0.0); 4]; 2];
        for sb in 0..n_sb {
            let t = &tile[sb * Q6K8::SIZE..];
            let dv = rows_f16(t, crate::kq::q6_d());
            let mut d8 = [0.0f32; 8];
            vst1q_f32(d8.as_mut_ptr(), dv[0]);
            vst1q_f32(d8.as_mut_ptr().add(4), dv[1]);
            q6k8_sb_i8mm(t, sb, acts, group, &d8, &mut facc);
        }
        for q in 0..2 {
            for pair in 0..4 {
                let mut lanes = [0.0f32; 4];
                vst1q_f32(lanes.as_mut_ptr(), facc[q][pair]);
                let a = b0 + 2 * q;
                out[a * ROWS + 2 * pair] = lanes[0];
                out[(a + 1) * ROWS + 2 * pair] = lanes[1];
                out[a * ROWS + 2 * pair + 1] = lanes[2];
                out[(a + 1) * ROWS + 2 * pair + 1] = lanes[3];
            }
        }
        b0 += ACT_GROUP;
    }
    while b0 < nb {
        q6k8_group::<_, 1>(tile, n_sb, &src, b0, &mut out[b0 * ROWS..]);
        b0 += 1;
    }
}

// ---------------------------------------------------------------------------
// PQ2_0: Bonsai 2's 2-bit ternary, one f16 scale per 128 weights
// ---------------------------------------------------------------------------

/// The four code planes of one 128-weight group, each split into its two 16-byte
/// halves: `[p0 lo, p0 hi, p1 lo, p1 hi, ...]`, where plane `s` holds the codes of
/// weights `4k + s` in byte `k`.
///
/// `sdot` sees the unsigned codes as signed bytes (they are 0..3), so a plane
/// against an activation plane is `Σ code·a` in four int32 lanes — lane `l` is the
/// 16-value block `8g + l` of the group. The `− Σ a` offset the x86 kernels apply
/// with `tls` is applied in the float fold below, in the reference's own order.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn pq2_planes(wq0: uint8x16_t, wq1: uint8x16_t) -> [int8x16_t; 8] {
    let m3 = vdupq_n_u8(3);
    macro_rules! p {
        (0) => {{
            (
                vreinterpretq_s8_u8(vandq_u8(wq0, m3)),
                vreinterpretq_s8_u8(vandq_u8(wq1, m3)),
            )
        }};
        ($s:literal) => {{
            (
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<$s>(wq0), m3)),
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<$s>(wq1), m3)),
            )
        }};
    }
    let (p0l, p0h) = p!(0);
    let (p1l, p1h) = p!(2);
    let (p2l, p2h) = p!(4);
    let (p3l, p3h) = p!(6);
    [p0l, p0h, p1l, p1h, p2l, p2h, p3l, p3h]
}

/// One group's four activation planes, in [`pq2_planes`]'s order.
///
/// # Safety
/// `ap` must point at `quant::tp_off(g)` of an activation whose `n` covers group `g`.
#[inline(always)]
unsafe fn pq2_acts(ap: *const i8) -> [int8x16_t; 8] {
    [
        vld1q_s8(ap),
        vld1q_s8(ap.add(16)),
        vld1q_s8(ap.add(64)),
        vld1q_s8(ap.add(80)),
        vld1q_s8(ap.add(128)),
        vld1q_s8(ap.add(144)),
        vld1q_s8(ap.add(192)),
        vld1q_s8(ap.add(208)),
    ]
}

/// The eight 16-value code sums of one group: `lo` holds blocks 0..4, `hi` 4..8.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn pq2_sums(planes: &[int8x16_t; 8], acts: &[int8x16_t; 8]) -> (int32x4_t, int32x4_t) {
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    lo = sdot(lo, planes[0], acts[0]);
    hi = sdot(hi, planes[1], acts[1]);
    lo = sdot(lo, planes[2], acts[2]);
    hi = sdot(hi, planes[3], acts[3]);
    lo = sdot(lo, planes[4], acts[4]);
    hi = sdot(hi, planes[5], acts[5]);
    lo = sdot(lo, planes[6], acts[6]);
    hi = sdot(hi, planes[7], acts[7]);
    (lo, hi)
}

/// One group's eight integer sums folded into the per-lane float accumulators:
/// lane `u` of `facc[0]` (blocks 0..4) and `facc[1]` (4..8) takes
/// `d·td[8g+u]·(s_u − tls[8g+u])`.
///
/// The fold is per lane, so the group's eight values never form a serial chain; and
/// because every path (two-row GEMV, four-activation GEMM, one row) runs this same
/// fold over the same lanes in the same group order, a row's bits do not depend on
/// the pass width or the batch.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn pq2_accum(
    facc: &mut [float32x4_t; 2],
    lo: int32x4_t,
    hi: int32x4_t,
    a: &QuantAct,
    g: usize,
    d: f32,
) {
    let dv = vdupq_n_f32(d);
    let tdl = vld1q_f32(a.td.as_ptr().add(8 * g));
    let tdh = vld1q_f32(a.td.as_ptr().add(8 * g + 4));
    let tll = vld1q_s32(a.tls.as_ptr().add(8 * g));
    let tlh = vld1q_s32(a.tls.as_ptr().add(8 * g + 4));
    let l = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(lo, tll)), tdl), dv);
    let h = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(hi, tlh)), tdh), dv);
    facc[0] = vaddq_f32(facc[0], l);
    facc[1] = vaddq_f32(facc[1], h);
}

/// The two accumulators' sixteen lanes, summed in order.
#[inline(always)]
unsafe fn pq2_finish(facc: &[float32x4_t; 2]) -> f32 {
    hsum_f32x4(vaddq_f32(facc[0], facc[1]))
}

/// The four lanes of a vector added in order.
#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    #[allow(clippy::incompatible_msrv)]
    {
        vaddvq_f32(v)
    }
}

/// One PQ2_0 row against one activation: eight `sdot`s per 128-weight group.
///
/// # Safety
/// The CPU must have `asimddp`; `w` must be `row_bytes(PQ2_0, a.n)` long and the
/// activation must be in the ternary layout.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_pq2(w: &[u8], a: &QuantAct) -> f32 {
    debug_assert_eq!(w.len(), crate::quant::row_bytes(Format::PQ2_0, a.n));
    let mut facc = [vdupq_n_f32(0.0); 2];
    for g in 0..w.len() / 34 {
        let blk = w.as_ptr().add(g * 34);
        let planes = pq2_planes(vld1q_u8(blk.add(2)), vld1q_u8(blk.add(18)));
        let acts = pq2_acts(a.tp.as_ptr().add(crate::quant::tp_off(g)));
        let (lo, hi) = pq2_sums(&planes, &acts);
        pq2_accum(&mut facc, lo, hi, a, g, f16_ptr(blk));
    }
    pq2_finish(&facc)
}

/// Two PQ2_0 rows against one activation: the activation planes are loaded once
/// and both rows' chains interleave.
///
/// Each row accumulates group by group exactly as [`dot_pq2`] does, so a row's bits
/// do not depend on how many rows shared the pass.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn pq2_two_rows(w0: *const u8, w1: *const u8, ng: usize, a: &QuantAct) -> (f32, f32) {
    let mut f0 = [vdupq_n_f32(0.0); 2];
    let mut f1 = [vdupq_n_f32(0.0); 2];
    for g in 0..ng {
        let (b0, b1) = (w0.add(g * 34), w1.add(g * 34));
        let acts = pq2_acts(a.tp.as_ptr().add(crate::quant::tp_off(g)));
        let p0 = pq2_planes(vld1q_u8(b0.add(2)), vld1q_u8(b0.add(18)));
        let p1 = pq2_planes(vld1q_u8(b1.add(2)), vld1q_u8(b1.add(18)));
        let (l0, h0) = pq2_sums(&p0, &acts);
        let (l1, h1) = pq2_sums(&p1, &acts);
        pq2_accum(&mut f0, l0, h0, a, g, f16_ptr(b0));
        pq2_accum(&mut f1, l1, h1, a, g, f16_ptr(b1));
    }
    (pq2_finish(&f0), pq2_finish(&f1))
}

/// One PQ2_0 row against `T` activations: the row's code planes are decoded once
/// per group and meet every activation's planes.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn pq2_batch<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let mut facc = [[vdupq_n_f32(0.0); 2]; T];
    for g in 0..w.len() / 34 {
        let blk = w.as_ptr().add(g * 34);
        let planes = pq2_planes(vld1q_u8(blk.add(2)), vld1q_u8(blk.add(18)));
        let d = f16_ptr(blk);
        for u in 0..T {
            let av = pq2_acts(acts[u].tp.as_ptr().add(crate::quant::tp_off(g)));
            let (lo, hi) = pq2_sums(&planes, &av);
            pq2_accum(&mut facc[u], lo, hi, &acts[u], g, d);
        }
    }
    for u in 0..T {
        out[u] = pq2_finish(&facc[u]);
    }
}

/// One PQ2_0 row against every activation, into `out[b]`.
///
/// Every (row, activation) pair runs [`dot_pq2`]'s group loop, so the result is
/// bit-identical to it at any batch width.
///
/// # Safety
/// The CPU must have `asimddp`; `w` must be one row of `row_bytes(PQ2_0, acts[0].n)`;
/// the activations must be in the ternary layout.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_batch_pq2(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    assert!(!acts.is_empty(), "no activation");
    assert!(out.len() >= acts.len(), "the output holds fewer rows");
    assert_eq!(w.len(), crate::quant::row_bytes(Format::PQ2_0, acts[0].n));
    let nb = acts.len();
    let mut b = 0;
    while b < nb {
        let t = (nb - b).min(4);
        let (a, o) = (&acts[b..b + t], &mut out[b..b + t]);
        match t {
            4 => pq2_batch::<4>(w, a, o),
            3 => pq2_batch::<3>(w, a, o),
            2 => pq2_batch::<2>(w, a, o),
            _ => pq2_batch::<1>(w, a, o),
        }
        b += t;
    }
}

/// `rows` PQ2_0 rows of `rb` bytes against every activation, into `out[r·nb + b]`.
///
/// A decode step (one activation) runs two rows per pass through `pq2_two_rows`;
/// a prefill runs one row against up to four activations through `pq2_batch`.
/// Both are [`dot_pq2`]'s arithmetic group by group, so decode and prefill agree bit
/// for bit on every row, and no row depends on the pass width.
///
/// # Safety
/// The CPU must have `asimddp`; `w.len() >= rows·rb`, `rb` is a whole number of
/// 34-byte groups, `out.len() >= rows·acts.len()`, and the activations are in the
/// ternary layout with the row's width.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_rows_pq2(w: &[u8], rb: usize, rows: usize, acts: &[QuantAct], out: &mut [f32]) {
    assert!(rb % 34 == 0, "a PQ2_0 row is a whole number of groups");
    assert!(!acts.is_empty(), "no activation");
    assert!(w.len() >= rows * rb, "the rows are shorter than advertised");
    assert!(
        out.len() >= rows * acts.len(),
        "the output holds fewer rows"
    );
    let nb = acts.len();
    if nb == 1 {
        let a = &acts[0];
        let ng = rb / 34;
        let mut r = 0;
        while r + 2 <= rows {
            let (v0, v1) =
                pq2_two_rows(w.as_ptr().add(r * rb), w.as_ptr().add((r + 1) * rb), ng, a);
            out[r] = v0;
            out[r + 1] = v1;
            r += 2;
        }
        if r < rows {
            out[r] = dot_pq2(&w[r * rb..(r + 1) * rb], a);
        }
    } else {
        for r in 0..rows {
            dot_batch_pq2(
                &w[r * rb..(r + 1) * rb],
                acts,
                &mut out[r * nb..(r + 1) * nb],
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// One row in any format NEON has a kernel for, or `None` when it has none.
///
/// # Safety
/// The CPU must have `asimddp`.
#[inline]
pub unsafe fn dot(fmt: Format, w: &[u8], a: &QuantAct) -> Option<f32> {
    match fmt {
        #[cfg(feature = "fmt-ternary")]
        Format::PQ2_0 => Some(dot_pq2(w, a)),
        #[cfg(feature = "fmt-legacy")]
        Format::Q8_0 => Some(dot_q8_0(w, a)),
        #[cfg(feature = "fmt-legacy")]
        Format::Q5_0 => Some(dot_q5_0(w, a)),
        #[cfg(feature = "fmt-kquant")]
        Format::Q4_K => Some(dot_q4_k(w, a)),
        #[cfg(feature = "fmt-kquant")]
        Format::Q6_K => Some(dot_q6_k(w, a)),
        _ => None,
    }
}

/// One row against every activation, for the formats with a batched NEON kernel.
///
/// # Safety
/// The CPU must have `asimddp`.
#[inline]
pub unsafe fn dot_batch(fmt: Format, w: &[u8], acts: &[QuantAct], out: &mut [f32]) -> bool {
    match fmt {
        #[cfg(feature = "fmt-ternary")]
        Format::PQ2_0 => {
            dot_batch_pq2(w, acts, out);
            true
        }
        #[cfg(feature = "fmt-legacy")]
        Format::Q8_0 => {
            q8_0_batch(w, acts, out);
            true
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// The 8-bit prompt form (S5.4d): the NEON bodies of the build and the multiply
// ---------------------------------------------------------------------------
//
// `crate::kq8` defines the form and its scalar twin; these are the aarch64 bodies
// of both, asserted bit for bit equal to the twin at every width. The arithmetic
// is the fork's (`docs/design/s5.4-fast-prompt.md` §2):
//
//   * build: the block's 256 weights as f32 with the format's own formula, the
//     largest magnitude, `dq = m/127`, `id = 1/dq`, `round_to_nearest_even(w·id)`;
//   * multiply: `sdot` (or `smmla`) over the signed codes, one `d·dk` multiply and
//     one fused multiply-add per (row, super-block).
//
// The form stores `code + 128` so that x86's `vpdpbusd` can take the weights as
// its unsigned operand. `sdot` and `smmla` are signed, so this file undoes the
// bias with one `eor 0x80` per 16-byte load and computes `Σ code·a` directly; the
// float epilogue is then the twin's own expression in its own order:
// `x = d·dk`, `acc = (s as f32).mul_add(x, acc)`. The integer sums are exact in
// any grouping, so the bias and the grouping cannot move a bit.

/// The eight signed codes of one row's eight-weight group, with the form's +128
/// bias undone (stored `(code + 128) as u8`, XOR 0x80, reinterpreted signed).
#[inline(always)]
unsafe fn q8k_codes(p: *const u8) -> int8x16_t {
    let stored = vld1q_u8(p);
    vreinterpretq_s8_u8(veorq_u8(stored, vdupq_n_u8(0x80)))
}

/// The eight rows' f32 scales of one super-block of the form (the form stores
/// them as f32, unlike the file's f16 tile scales).
#[inline(always)]
unsafe fn q8k_scales(t: &[u8]) -> [float32x4_t; 2] {
    let p = t.as_ptr().add(crate::kq8::Q8K_D) as *const f32;
    [vld1q_f32(p), vld1q_f32(p.add(4))]
}

/// One super-block of the 8-bit form against `N` activation rows, on `sdot`.
///
/// The form is eight output rows interleaved per super-block; a 16-byte load is
/// two rows of eight codes, which is one `sdot` operand. The activation's eight
/// codes for a row are duplicated to the operand's other half, so the four lanes
/// are `[r0 low, r0 high, r1 low, r1 high]` and [`rows_of`] folds them to eight
/// rows. All 32 groups accumulate into one exact int32 sum per row; the epilogue
/// is the twin's.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q8k_sb_sdot<const N: usize>(
    t: &[u8],
    n_sb: usize,
    acts: &KqActs,
    sb: usize,
    row0: usize,
    facc: &mut [[float32x4_t; 2]; N],
) {
    let mut acc = [[vdupq_n_s32(0); 4]; N];
    for g in 0..crate::kq8::Q8K_GROUPS {
        let base = t.as_ptr().add(g * 64);
        let c0 = q8k_codes(base);
        let c1 = q8k_codes(base.add(16));
        let c2 = q8k_codes(base.add(32));
        let c3 = q8k_codes(base.add(48));
        for u in 0..N {
            let row = row0 + u;
            let ag = row / ACT_GROUP;
            let at = ag * n_sb * 1024 + sb * 1024 + g * 32 + (row % ACT_GROUP) * 8;
            let a = vld1_s8(acts.q.as_ptr().add(at));
            let av = vcombine_s8(a, a);
            acc[u][0] = sdot(acc[u][0], c0, av);
            acc[u][1] = sdot(acc[u][1], c1, av);
            acc[u][2] = sdot(acc[u][2], c2, av);
            acc[u][3] = sdot(acc[u][3], c3, av);
        }
    }
    let dv = q8k_scales(t);
    for u in 0..N {
        let row = row0 + u;
        let at = ((row / ACT_GROUP) * n_sb + sb) * ACT_GROUP + row % ACT_GROUP;
        let dk = vdupq_n_f32(acts.d[at]);
        let s = rows_of(&acc[u]);
        for h in 0..2 {
            let x = vmulq_f32(dv[h], dk);
            facc[u][h] = vfmaq_f32(facc[u][h], vcvtq_f32_s32(s[h]), x);
        }
    }
}

/// Eight output rows of an 8-bit form against `N` activation rows, on `sdot`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q8k_group_sdot<const N: usize>(
    form: &[u8],
    n_sb: usize,
    acts: &KqActs,
    row0: usize,
    out: &mut [f32],
) {
    assert!((1..=4).contains(&N), "one to four activation rows");
    assert!(
        out.len() >= N * ROWS,
        "the output holds eight rows per activation"
    );
    let mut facc = [[vdupq_n_f32(0.0); 2]; N];
    for sb in 0..n_sb {
        let t = &form[sb * crate::kq8::Q8K_SIZE..];
        q8k_sb_sdot::<N>(t, n_sb, acts, sb, row0, &mut facc);
    }
    for u in 0..N {
        vst1q_f32(out.as_mut_ptr().add(u * ROWS), facc[u][0]);
        vst1q_f32(out.as_mut_ptr().add(u * ROWS + 4), facc[u][1]);
    }
}

// Test-only counts of the two 8-bit bodies' calls, so the dispatch test can see
// which one the tier's entry point takes. `thread_local` keeps a concurrently
// running test out of the count.
#[cfg(test)]
thread_local! {
    pub(crate) static Q8K_SDOT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static Q8K_I8MM_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Eight output rows of an 8-bit form against a batch of activations, on `sdot`.
///
/// # Safety
/// The CPU must have `asimddp`; the form must hold `n_sb` super-blocks, the
/// activation at least as many and at least `nb` rows, and `out` `nb · 8` floats.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q8k_gemm_sdot(form: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    #[cfg(test)]
    Q8K_SDOT_CALLS.with(|c| c.set(c.get() + 1));
    assert!(
        form.len() >= n_sb * crate::kq8::Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(
        acts.super_blocks() >= n_sb,
        "the activation is shorter than the weights"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    let mut b0 = 0;
    while nb - b0 >= ACT_GROUP {
        q8k_group_sdot::<4>(form, n_sb, acts, b0, &mut out[b0 * ROWS..]);
        b0 += ACT_GROUP;
    }
    match nb - b0 {
        3 => q8k_group_sdot::<3>(form, n_sb, acts, b0, &mut out[b0 * ROWS..]),
        2 => q8k_group_sdot::<2>(form, n_sb, acts, b0, &mut out[b0 * ROWS..]),
        1 => q8k_group_sdot::<1>(form, n_sb, acts, b0, &mut out[b0 * ROWS..]),
        _ => {}
    }
}

/// Eight output rows of an 8-bit form against four activation rows (two pairs)
/// on `smmla`, folded onto `facc[activation pair][row pair]`.
///
/// The form's window — two output rows of eight codes — is `smmla`'s first
/// operand as it stands (after the bias is undone), and two activation rows'
/// eight codes joined are its second. One instruction forms the 32 products of
/// two output rows by two activations, and four cover the eight output rows. The
/// 32 groups accumulate into one exact int32 sum per (row pair, activation pair);
/// the fold is the twin's, super-block by super-block in the form's order, so a
/// row is bit-identical to the `sdot` body.
#[inline]
#[target_feature(enable = "neon,i8mm")]
unsafe fn q8k_group_i8mm(form: &[u8], n_sb: usize, acts: &KqActs, b0: usize, out: &mut [f32]) {
    assert!(b0 % ACT_GROUP == 0, "an i8mm group starts at four rows");
    // [activation pair][row pair]; the lanes are [r0·a0, r0·a1, r1·a0, r1·a1].
    let mut facc = [[vdupq_n_f32(0.0); 4]; 2];
    for sb in 0..n_sb {
        let t = &form[sb * crate::kq8::Q8K_SIZE..];
        let dv = q8k_scales(t);
        let mut d8 = [0.0f32; 8];
        vst1q_f32(d8.as_mut_ptr(), dv[0]);
        vst1q_f32(d8.as_mut_ptr().add(4), dv[1]);
        let mut acc = [[vdupq_n_s32(0); 4]; 2];
        for g in 0..crate::kq8::Q8K_GROUPS {
            let base = t.as_ptr().add(g * 64);
            let codes = [
                q8k_codes(base),
                q8k_codes(base.add(16)),
                q8k_codes(base.add(32)),
                q8k_codes(base.add(48)),
            ];
            for q in 0..2 {
                let a0 = b0 + 2 * q;
                let at = (a0 / ACT_GROUP) * n_sb * 1024 + sb * 1024 + g * 32 + (a0 % ACT_GROUP) * 8;
                let lo = vld1_s8(acts.q.as_ptr().add(at));
                let hi = vld1_s8(acts.q.as_ptr().add(at + 8));
                let bv = vcombine_s8(lo, hi);
                for p in 0..4 {
                    acc[q][p] = smmla(acc[q][p], codes[p], bv);
                }
            }
        }
        for q in 0..2 {
            let a0 = b0 + 2 * q;
            let a1 = a0 + 1;
            let at0 = ((a0 / ACT_GROUP) * n_sb + sb) * ACT_GROUP + a0 % ACT_GROUP;
            let at1 = ((a1 / ACT_GROUP) * n_sb + sb) * ACT_GROUP + a1 % ACT_GROUP;
            let dk = pair_f32(acts.d[at0], acts.d[at1]);
            for p in 0..4 {
                // The lanes are [row 2p · a0, row 2p · a1, row 2p+1 · a0, row 2p+1 · a1].
                let dp = quad_f32(d8[2 * p], d8[2 * p + 1]);
                let x = vmulq_f32(dp, dk);
                facc[q][p] = vfmaq_f32(facc[q][p], vcvtq_f32_s32(acc[q][p]), x);
            }
        }
    }
    for q in 0..2 {
        for p in 0..4 {
            let mut lanes = [0.0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), facc[q][p]);
            let a = 2 * q;
            out[a * ROWS + 2 * p] = lanes[0];
            out[(a + 1) * ROWS + 2 * p] = lanes[1];
            out[a * ROWS + 2 * p + 1] = lanes[2];
            out[(a + 1) * ROWS + 2 * p + 1] = lanes[3];
        }
    }
}

/// Eight output rows of an 8-bit form against a batch of activations, on `smmla`
/// for whole groups of four and on the `sdot` body for the 1–3-row tail.
///
/// # Safety
/// The CPU must have `asimd`, `asimddp` and `i8mm`; the form must hold `n_sb`
/// super-blocks, the activation at least as many and at least `nb` rows, and
/// `out` `nb · 8` floats.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn q8k_gemm_i8mm(form: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    #[cfg(test)]
    Q8K_I8MM_CALLS.with(|c| c.set(c.get() + 1));
    assert!(
        form.len() >= n_sb * crate::kq8::Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(
        acts.super_blocks() >= n_sb,
        "the activation is shorter than the weights"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    let mut b0 = 0;
    while nb - b0 >= ACT_GROUP {
        q8k_group_i8mm(form, n_sb, acts, b0, &mut out[b0 * ROWS..]);
        b0 += ACT_GROUP;
    }
    match nb - b0 {
        3 => q8k_group_sdot::<3>(form, n_sb, acts, b0, &mut out[b0 * ROWS..]),
        2 => q8k_group_sdot::<2>(form, n_sb, acts, b0, &mut out[b0 * ROWS..]),
        1 => q8k_group_sdot::<1>(form, n_sb, acts, b0, &mut out[b0 * ROWS..]),
        _ => {}
    }
}

/// Eight output rows of an 8-bit form against a batch of activations, on the
/// widest body the CPU tier allows: `smmla` on `neon-i8mm`, `sdot` on
/// `neon-dotprod`, and the scalar twin on `neon` or `scalar`.
///
/// This is the tier entry point the engine's dispatch (`MatKernels::q8k_gemm`)
/// calls on aarch64 when the CPU has `dotprod` (S5.4e); the S5.4d tests call it
/// directly.
///
/// # Safety
/// The form must hold `n_sb` super-blocks, the activation at least as many and at
/// least `nb` rows, and `out` `nb · 8` floats. The CPU's own tier is checked
/// before a vector body is entered.
pub unsafe fn q8k_gemm(form: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    assert!(
        form.len() >= n_sb * crate::kq8::Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(
        acts.super_blocks() >= n_sb,
        "the activation is shorter than the weights"
    );
    assert!(
        out.len() >= nb * ROWS,
        "the output holds fewer than eight rows per activation"
    );
    if crate::cpu::have_i8mm() {
        q8k_gemm_i8mm(form, n_sb, acts, nb, out);
    } else if crate::cpu::have_dotprod() {
        q8k_gemm_sdot(form, n_sb, acts, nb, out);
    } else {
        crate::kq8::gemm_q8k(form, n_sb, acts, nb, out);
    }
}

/// The f32 weights of one block of eight rows, built from a Q4_K or Q5_0 tile.
///
/// This is the twin's `q4k_weights`/`q5_0_weights` for one row, vectorised: a
/// 32-weight sub-block is four eight-byte nibble loads, the codes widen to f32,
/// and each weight is one fused multiply-add of an exact integer product with
/// the sub-block's min term.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn q8k_weights_q4k(t: &[u8], row: usize, w: &mut [f32; crate::kq8::Q8K_WEIGHTS]) {
    let d = f16_ptr(t.as_ptr().add(q4_d() + row * 2));
    let dmin = f16_ptr(t.as_ptr().add(q4_dmin() + row * 2));
    let dv = vdupq_n_f32(d);
    let lo_mask = vdup_n_u8(0x0F);
    for sub in 0..8 {
        let ld = tile_q4k_scale(t, sub, row) as i32;
        let lm = tile_q4k_min(t, sub, row) as i32;
        // The fork's `vm = dmin * (-1)`, then `mm = vm * Lm`.
        let mm = vdupq_n_f32((-dmin) * lm as f32);
        let ldv = vdupq_n_f32(ld as f32);
        let base = q4_qs() + (sub / 2) * 4 * 64 + row * 8;
        let p = t.as_ptr().add(base);
        let g0 = vld1_u8(p);
        let g1 = vld1_u8(p.add(64));
        let g2 = vld1_u8(p.add(128));
        let g3 = vld1_u8(p.add(192));
        let mut nib = [0u8; 32];
        vst1_u8(nib.as_mut_ptr(), g0);
        vst1_u8(nib.as_mut_ptr().add(8), g1);
        vst1_u8(nib.as_mut_ptr().add(16), g2);
        vst1_u8(nib.as_mut_ptr().add(24), g3);
        for chunk in 0..4 {
            let v = vld1_u8(nib.as_ptr().add(chunk * 8));
            let q = if sub % 2 == 0 {
                vand_u8(v, lo_mask)
            } else {
                vshr_n_u8::<4>(v)
            };
            let wide = vmovl_u8(q);
            let lo = vmovl_u16(vget_low_u16(wide));
            let hi = vmovl_u16(vget_high_u16(wide));
            // `(q·ld) as f32` is exact: both are small integers, and so is the product.
            let f0 = vmulq_f32(vcvtq_f32_u32(lo), ldv);
            let f1 = vmulq_f32(vcvtq_f32_u32(hi), ldv);
            let at = sub * 32 + chunk * 8;
            vst1q_f32(w.as_mut_ptr().add(at), vfmaq_f32(mm, f0, dv));
            vst1q_f32(w.as_mut_ptr().add(at + 4), vfmaq_f32(mm, f1, dv));
        }
    }
}

/// The 256 integer products `(q − 32)·L` of one Q6_K row, vectorised.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn q8k_products_q6k(t: &[u8], row: usize, p: &mut [i32; crate::kq8::Q8K_WEIGHTS]) {
    for run in 0..16 {
        let c = run / 8;
        let g = (run % 8) / 2;
        let hf = run % 2;
        let nib_at = if g % 2 == 0 { 0 } else { 32 };
        let high = g >= 2;
        let ql = crate::kq::q6_ql() + (c * 8 + nib_at / 8 + hf * 2) * 64 + row * 8;
        let qh = crate::kq::q6_qh() + (c * 4 + hf * 2) * 64 + row * 8;
        let l = vcombine_u8(
            vld1_u8(t.as_ptr().add(ql)),
            vld1_u8(t.as_ptr().add(ql + 64)),
        );
        let nib = if high {
            vandq_u8(vshrq_n_u8::<4>(l), vdupq_n_u8(0x0F))
        } else {
            vandq_u8(l, vdupq_n_u8(0x0F))
        };
        let hb = vcombine_u8(
            vld1_u8(t.as_ptr().add(qh)),
            vld1_u8(t.as_ptr().add(qh + 64)),
        );
        let shifted = vshlq_u16(vreinterpretq_u16_u8(hb), vdupq_n_s16(-((2 * g) as i16)));
        let bits = vandq_u8(vreinterpretq_u8_u16(shifted), vdupq_n_u8(0x03));
        let code = vaddq_u8(nib, vshlq_n_u8::<4>(bits));
        let sc = t[crate::kq::q6_sc() + run * 8 + row] as i8 as i32;
        let scv = vdupq_n_s32(sc);
        let wide_lo = vmovl_u8(vget_low_u8(code));
        let wide_hi = vmovl_u8(vget_high_u8(code));
        let lo = vmovl_u16(vget_low_u16(wide_lo));
        let hi = vmovl_u16(vget_high_u16(wide_lo));
        let hi2 = vmovl_u16(vget_low_u16(wide_hi));
        let hi3 = vmovl_u16(vget_high_u16(wide_hi));
        let c0 = vsubq_s32(vreinterpretq_s32_u32(lo), vdupq_n_s32(32));
        let c1 = vsubq_s32(vreinterpretq_s32_u32(hi), vdupq_n_s32(32));
        let c2 = vsubq_s32(vreinterpretq_s32_u32(hi2), vdupq_n_s32(32));
        let c3 = vsubq_s32(vreinterpretq_s32_u32(hi3), vdupq_n_s32(32));
        vst1q_s32(p.as_mut_ptr().add(run * 16), vmulq_s32(c0, scv));
        vst1q_s32(p.as_mut_ptr().add(run * 16 + 4), vmulq_s32(c1, scv));
        vst1q_s32(p.as_mut_ptr().add(run * 16 + 8), vmulq_s32(c2, scv));
        vst1q_s32(p.as_mut_ptr().add(run * 16 + 12), vmulq_s32(c3, scv));
    }
}

/// The requantisation half of the build: `w`'s 256 floats, `id = 1/dq`, and the
/// form's +128 bias, written into one (row, super-block).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn q8k_requant_store_f32(
    form: &mut [u8],
    n_sb: usize,
    sb: usize,
    row: usize,
    w: &[f32; crate::kq8::Q8K_WEIGHTS],
) -> f32 {
    assert!(
        form.len() >= n_sb * crate::kq8::Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    let mut vmax = vdupq_n_f32(0.0);
    for i in 0..crate::kq8::Q8K_WEIGHTS / 4 {
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(w.as_ptr().add(i * 4))));
    }
    let mut lanes = [0.0f32; 4];
    vst1q_f32(lanes.as_mut_ptr(), vmax);
    let m = lanes.iter().fold(0.0f32, |a, &x| a.max(x));
    let dq = m / 127.0f32;
    let id = if dq != 0.0 { 1.0f32 / dq } else { 0.0 };
    let idv = vdupq_n_f32(id);
    let bias = vdup_n_u8(0x80);
    let out = form.as_mut_ptr();
    for gi in 0..crate::kq8::Q8K_GROUPS {
        let a = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(w.as_ptr().add(gi * 8)), idv));
        let b = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(w.as_ptr().add(gi * 8 + 4)), idv));
        let c = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
        let bytes = veor_u8(vreinterpret_u8_s8(vqmovn_s16(c)), bias);
        vst1_u8(out.add(crate::kq8::code_at(sb, gi, row, 0)), bytes);
    }
    let at = crate::kq8::scale_at(sb, row);
    std::ptr::write_unaligned(out.add(at) as *mut f32, dq);
    dq
}

/// The 8-bit form of `n_sb` repacked Q4_K super-blocks, on NEON.
///
/// The weights are the fork's: `fma((q·Ld) as f32, d, (−dmin)·Lm)` per weight.
///
/// # Safety
/// The tile must hold `n_sb` super-blocks and `form` `n_sb · Q8K_SIZE` bytes.
#[target_feature(enable = "neon")]
pub unsafe fn q8k_build_q4k_neon(tile: &[u8], n_sb: usize, form: &mut [u8]) {
    assert!(
        tile.len() >= n_sb * Q4K8::SIZE,
        "the tile holds fewer than n_sb super-blocks"
    );
    assert!(
        form.len() >= n_sb * crate::kq8::Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    let mut w = [0.0f32; crate::kq8::Q8K_WEIGHTS];
    for sb in 0..n_sb {
        let t = &tile[sb * Q4K8::SIZE..(sb + 1) * Q4K8::SIZE];
        for row in 0..ROWS {
            q8k_weights_q4k(t, row, &mut w);
            q8k_requant_store_f32(form, n_sb, sb, row, &w);
        }
    }
}

/// The 8-bit form of `n_sb` repacked Q6_K super-blocks, on NEON.
///
/// Q6_K's own arithmetic is the int16 product `(q − 32)·L`; the form's scale is
/// `d · max|p| / 127` (or exactly `d` in the fork's exact case), so the products
/// are what is requantised.
///
/// # Safety
/// The tile must hold `n_sb` super-blocks and `form` `n_sb · Q8K_SIZE` bytes.
#[target_feature(enable = "neon")]
pub unsafe fn q8k_build_q6k_neon(tile: &[u8], n_sb: usize, form: &mut [u8]) {
    assert!(
        tile.len() >= n_sb * Q6K8::SIZE,
        "the tile holds fewer than n_sb super-blocks"
    );
    assert!(
        form.len() >= n_sb * crate::kq8::Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    let bias = vdup_n_u8(0x80);
    for sb in 0..n_sb {
        let t = &tile[sb * Q6K8::SIZE..(sb + 1) * Q6K8::SIZE];
        for row in 0..ROWS {
            let d = f16_ptr(t.as_ptr().add(crate::kq::q6_d() + row * 2));
            let mut p = [0i32; crate::kq8::Q8K_WEIGHTS];
            q8k_products_q6k(t, row, &mut p);
            // `m = max |p|`, as the twin's int loop.
            let mut vmax = vdupq_n_s32(0);
            for i in 0..crate::kq8::Q8K_WEIGHTS / 4 {
                vmax = vmaxq_s32(vmax, vabsq_s32(vld1q_s32(p.as_ptr().add(i * 4))));
            }
            let mut lanes = [0i32; 4];
            vst1q_s32(lanes.as_mut_ptr(), vmax);
            let m = lanes.iter().fold(0i32, |a, &x| a.max(x));
            let dnew = m as f32 * (1.0f32 / 127.0f32);
            // The fork's exact case: `dnew < 1` keeps the products and the scale `d`.
            let exact = dnew < 1.0;
            let id = if exact {
                0.0
            } else if dnew.abs() > 1e-9 {
                1.0f32 / dnew
            } else {
                0.0
            };
            let scale = if exact { d } else { d * dnew };
            let idv = vdupq_n_f32(id);
            let out = form.as_mut_ptr();
            for gi in 0..crate::kq8::Q8K_GROUPS {
                let a = vld1q_s32(p.as_ptr().add(gi * 8));
                let b = vld1q_s32(p.as_ptr().add(gi * 8 + 4));
                let (ca, cb) = if exact {
                    (a, b)
                } else {
                    (
                        vcvtnq_s32_f32(vmulq_f32(vcvtq_f32_s32(a), idv)),
                        vcvtnq_s32_f32(vmulq_f32(vcvtq_f32_s32(b), idv)),
                    )
                };
                let c = vcombine_s16(vqmovn_s32(ca), vqmovn_s32(cb));
                let bytes = veor_u8(vreinterpret_u8_s8(vqmovn_s16(c)), bias);
                vst1_u8(out.add(crate::kq8::code_at(sb, gi, row, 0)), bytes);
            }
            let at = crate::kq8::scale_at(sb, row);
            std::ptr::write_unaligned(out.add(at) as *mut f32, scale);
        }
    }
}

/// The 8-bit form of `n_blk` repacked Q5_0 blocks, eight to a super-block, on
/// NEON.
///
/// Q5_0's own formula is `w = d · (q − 16)`; the requantisation that follows is
/// Q4_K's, so the same kernel multiplies with the result.
///
/// # Safety
/// The tile must hold `n_blk` blocks and `form` the `n_blk / 8` super-blocks it
/// needs.
#[cfg(feature = "fmt-legacy")]
#[target_feature(enable = "neon")]
pub unsafe fn q8k_build_q5_0_neon(tile: &[u8], n_blk: usize, form: &mut [u8]) {
    assert!(
        tile.len() >= n_blk * Q5_08::SIZE,
        "the tile holds fewer than n_blk blocks"
    );
    let n_sb = n_blk.div_ceil(crate::kq8::Q8K_WEIGHTS / Q5_0_WEIGHTS);
    assert!(
        form.len() >= n_sb * crate::kq8::Q8K_SIZE,
        "the form holds fewer than n_blk blocks"
    );
    let lo_mask = vdup_n_u8(0x0F);
    let bit = vld1_u8([1u8, 2, 4, 8, 16, 32, 64, 128].as_ptr());
    let fifth = vdup_n_u8(16);
    for sb in 0..n_sb {
        let b0 = sb * 8;
        let nb = (n_blk - b0).min(8);
        let t = &tile[b0 * Q5_08::SIZE..(b0 + nb) * Q5_08::SIZE];
        for row in 0..ROWS {
            let mut w = [0.0f32; crate::kq8::Q8K_WEIGHTS];
            for b in 0..nb {
                let at = b * Q5_08::SIZE;
                let d = f16_ptr(t.as_ptr().add(at + crate::kq::q5_d() + row * 2));
                let dv = vdupq_n_f32(d);
                for h in 0..4 {
                    let qs = t.as_ptr().add(
                        at + crate::kq::q5_qs()
                            + (h % 2) * crate::kq::Q5_WINDOW
                            + row * crate::kq::Q5_QS_ROW,
                    );
                    let v = vld1_u8(qs);
                    let nib = if h < 2 {
                        vand_u8(v, lo_mask)
                    } else {
                        vshr_n_u8::<4>(v)
                    };
                    let qh = vdup_n_u8(*t.as_ptr().add(at + crate::kq::q5_qh() + h * 8 + row));
                    let set = vceq_u8(vand_u8(qh, bit), bit);
                    let code = vorr_u8(nib, vand_u8(set, fifth));
                    let wide = vmovl_u8(code);
                    let lo = vmovl_u16(vget_low_u16(wide));
                    let hi = vmovl_u16(vget_high_u16(wide));
                    let c0 = vsubq_s32(vreinterpretq_s32_u32(lo), vdupq_n_s32(16));
                    let c1 = vsubq_s32(vreinterpretq_s32_u32(hi), vdupq_n_s32(16));
                    let at_w = b * 32 + h * 8;
                    vst1q_f32(w.as_mut_ptr().add(at_w), vmulq_f32(vcvtq_f32_s32(c0), dv));
                    vst1q_f32(
                        w.as_mut_ptr().add(at_w + 4),
                        vmulq_f32(vcvtq_f32_s32(c1), dv),
                    );
                }
            }
            q8k_requant_store_f32(form, n_sb, sb, row, &w);
        }
    }
}

// ---------------------------------------------------------------------------
// The attention value mix's row pair (S4.9s)
// ---------------------------------------------------------------------------
//
// `attn.rs`'s `mix_f16` walks one row's positions outermost and its dimensions
// innermost: for every key it loads the row's output, multiplies the stored half
// by the softmax weight, adds, and stores the output back. At a two-thousand-token
// text that reloads and re-stores the whole output row once per key, and a second
// row sharing the same KV head reads the same V halves again. The pair body below
// turns the loop inside out: one eight-dimension slice of both outputs stays in
// registers while the keys walk ascending, and each stored V half-vector is
// widened once and used by both rows.
//
// The arithmetic is the single-row body's, element for element: `α·o` first (when
// α is not one), then `o + p_j·v_j` per ascending key, each a separate multiply and
// a separate add in the same order, so no output bit can move. Only the load and
// the widening are shared.

/// `o = α·o + Σ_j p_j · v_j` for two rows over one F16 V tile: the scalar
/// definition the NEON body is held to.
///
/// `p0` and `p1` must be the same length; `v` holds at least `p0.len() · hd`
/// halves.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn mix_f16_pair_scalar(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    v: &[u8],
    hd: usize,
) {
    assert_eq!(p0.len(), p1.len(), "a pair mixes the same positions");
    assert!(v.len() >= p0.len() * hd * 2, "the V tile is shorter than p");
    if alpha0 != 1.0 {
        o0[..hd].iter_mut().for_each(|x| *x *= alpha0);
    }
    if alpha1 != 1.0 {
        o1[..hd].iter_mut().for_each(|x| *x *= alpha1);
    }
    for (j, (&w0, &w1)) in p0.iter().zip(p1).enumerate() {
        let row = j * hd;
        for d in 0..hd {
            let h = u16::from_le_bytes([v[(row + d) * 2], v[(row + d) * 2 + 1]]);
            let value = f16(h);
            o0[d] += w0 * value;
            o1[d] += w1 * value;
        }
    }
}

/// The NEON body of [`mix_f16_pair_scalar`]: both rows' outputs stay in
/// registers eight dimensions at a time while the keys walk ascending, and each
/// stored V half-vector is loaded and widened once for both rows.
///
/// # Safety
/// The CPU must have NEON; `p0` and `p1` must be the same length, `o0` and `o1`
/// at least `hd` floats, and `v` at least `p0.len() · hd · 2` bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mix_f16_pair(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    v: &[u8],
    hd: usize,
) {
    use std::arch::aarch64::*;
    assert_eq!(p0.len(), p1.len(), "a pair mixes the same positions");
    assert!(v.len() >= p0.len() * hd * 2, "the V tile is shorter than p");
    assert!(
        o0.len() >= hd && o1.len() >= hd,
        "an output row is shorter than hd"
    );
    if alpha0 != 1.0 {
        for d in 0..hd {
            *o0.get_unchecked_mut(d) *= alpha0;
        }
    }
    if alpha1 != 1.0 {
        for d in 0..hd {
            *o1.get_unchecked_mut(d) *= alpha1;
        }
    }
    let vp = v.as_ptr();
    let mut d = 0;
    while d + 8 <= hd {
        let mut a0 = [
            vld1q_f32(o0.as_ptr().add(d)),
            vld1q_f32(o0.as_ptr().add(d + 4)),
        ];
        let mut a1 = [
            vld1q_f32(o1.as_ptr().add(d)),
            vld1q_f32(o1.as_ptr().add(d + 4)),
        ];
        for (j, (&w0, &w1)) in p0.iter().zip(p1).enumerate() {
            let at = (j * hd + d) * 2;
            let v0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(at) as *const u16)));
            let v1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(at + 8) as *const u16)));
            let wv0 = vdupq_n_f32(w0);
            a0[0] = vaddq_f32(a0[0], vmulq_f32(wv0, v0));
            a0[1] = vaddq_f32(a0[1], vmulq_f32(wv0, v1));
            let wv1 = vdupq_n_f32(w1);
            a1[0] = vaddq_f32(a1[0], vmulq_f32(wv1, v0));
            a1[1] = vaddq_f32(a1[1], vmulq_f32(wv1, v1));
        }
        vst1q_f32(o0.as_mut_ptr().add(d), a0[0]);
        vst1q_f32(o0.as_mut_ptr().add(d + 4), a0[1]);
        vst1q_f32(o1.as_mut_ptr().add(d), a1[0]);
        vst1q_f32(o1.as_mut_ptr().add(d + 4), a1[1]);
        d += 8;
    }
    while d < hd {
        let mut x0 = *o0.get_unchecked(d);
        let mut x1 = *o1.get_unchecked(d);
        for (j, (&w0, &w1)) in p0.iter().zip(p1).enumerate() {
            let at = (j * hd + d) * 2;
            let h = u16::from_le_bytes([*vp.add(at), *vp.add(at + 1)]);
            let value = f16(h);
            x0 += w0 * value;
            x1 += w1 * value;
        }
        *o0.get_unchecked_mut(d) = x0;
        *o1.get_unchecked_mut(d) = x1;
        d += 1;
    }
}

// ---------------------------------------------------------------------------
// The attention tile's four query rows (S4.9t)
// ---------------------------------------------------------------------------
//
// Every query row of a prefill block scores and mixes the same tile of positions,
// and the row pair could only share the loads between two of them. The two bodies
// below take **four** rows over one tile. Neither changes any arithmetic: each row
// keeps its own accumulators, its own reduction tree and its own ascending walk over
// the keys; only the loads and widenings of K and V are shared, four rows at a time.
//
// Why four, and why it pays:
//
// * The score body keeps two four-lane accumulators per row (dimensions `i..i+4` and
//   `i+4..i+8` of each ascending eight-wide block), so it is one chain of `fma` per
//   accumulator: a row-pair has eight chains in flight and a four-row tile sixteen,
//   which covers the core's four vector FMA pipes at the dependency latency. Two
//   keys per pass — eight keys need thirty-two accumulators and the compiler spills,
//   measured 24.7 GMAC/s against this body's 40.
// * The mix body keeps two accumulators per row over an eight-dimension slice of the
//   output, so it too is chain-bound: a row-pair has four chains (28 GMAC/s), four
//   rows have eight (55). Sixteen chains (four rows over sixteen dimensions) measure
//   the same 46–55, so eight is the knee.

/// The canonical aarch64 score tree in scalar Rust: two four-lane accumulators
/// (dimensions `0..4` and `4..8` of each ascending eight-wide block), reduced as
/// `((a0+a1)+(a2+a3)) + ((b0+b1)+(b2+b3))`. This is `attn.rs`'s `canonical_score`
/// written against a widened F16 tile, and the definition the NEON body below is
/// held to.
#[cfg(target_arch = "aarch64")]
fn score_tree_scalar(q: &[f32], k: &[f32]) -> f32 {
    let mut a0 = [0.0f32; 4];
    let mut a1 = [0.0f32; 4];
    let mut i = 0;
    while i + 8 <= q.len() {
        for l in 0..4 {
            a0[l] = q[i + l].mul_add(k[i + l], a0[l]);
            a1[l] = q[i + 4 + l].mul_add(k[i + 4 + l], a1[l]);
        }
        i += 8;
    }
    ((a0[0] + a0[1]) + (a0[2] + a0[3])) + ((a1[0] + a1[1]) + (a1[2] + a1[3]))
}

/// [`scores_f16_tile4`]'s scalar definition: every row's own canonical tree over the
/// same stored halves.
///
/// `q` holds four query rows of `hd` floats; `k` a tile of `vis` keys of `hd` halves;
/// `s` holds four score rows, row `r` at `s[r · stride ..]`.
#[cfg(target_arch = "aarch64")]
pub fn scores_f16_tile4_scalar(
    q: &[f32],
    k: &[u8],
    vis: usize,
    hd: usize,
    scale: f32,
    s: &mut [f32],
    stride: usize,
) {
    assert!(q.len() >= 4 * hd, "four query rows are needed");
    assert!(
        k.len() >= vis * hd * 2,
        "the K tile is shorter than vis · hd"
    );
    assert!(
        s.len() >= 3 * stride + vis,
        "four score rows need 3 · stride + vis floats"
    );
    let halves: Vec<f32> = (0..vis * hd)
        .map(|i| f16(u16::from_le_bytes([k[i * 2], k[i * 2 + 1]])))
        .collect();
    for r in 0..4 {
        for j in 0..vis {
            s[r * stride + j] =
                score_tree_scalar(&q[r * hd..(r + 1) * hd], &halves[j * hd..][..hd]) * scale;
        }
    }
}

/// Four query rows scored against one F16 K tile, two keys per pass.
///
/// Each row's accumulators, reduction tree and ascending key order are its own, so
/// every score is the single-row kernel's bit for bit; only the loaded and widened
/// key is shared.
///
/// # Safety
/// The CPU must have NEON; `q` must hold four rows of `hd` floats, `k` at least
/// `vis · hd · 2` bytes, `s` four rows of `vis` floats at `stride`, and `hd` must be a
/// multiple of eight.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn scores_f16_tile4(
    q: &[f32],
    k: &[u8],
    vis: usize,
    hd: usize,
    scale: f32,
    s: &mut [f32],
    stride: usize,
) {
    use std::arch::aarch64::*;
    assert!(q.len() >= 4 * hd, "four query rows are needed");
    assert!(
        k.len() >= vis * hd * 2,
        "the K tile is shorter than vis · hd"
    );
    assert!(
        s.len() >= 3 * stride + vis,
        "four score rows need 3 · stride + vis floats"
    );
    assert!(
        hd.is_multiple_of(8),
        "the tile body walks eight dims at a time"
    );
    let qs = [
        q.as_ptr(),
        q.as_ptr().add(hd),
        q.as_ptr().add(2 * hd),
        q.as_ptr().add(3 * hd),
    ];
    let sc = vdupq_n_f32(scale);
    let kp = k.as_ptr();
    let mut j = 0;
    while j + 2 <= vis {
        // One accumulator pair per row and key: sixteen independent chains.
        let mut a = [[[vdupq_n_f32(0.0); 2]; 4]; 2];
        let mut i = 0;
        while i < hd {
            let qv = [
                (vld1q_f32(qs[0].add(i)), vld1q_f32(qs[0].add(i + 4))),
                (vld1q_f32(qs[1].add(i)), vld1q_f32(qs[1].add(i + 4))),
                (vld1q_f32(qs[2].add(i)), vld1q_f32(qs[2].add(i + 4))),
                (vld1q_f32(qs[3].add(i)), vld1q_f32(qs[3].add(i + 4))),
            ];
            for (u, keyslot) in a.iter_mut().enumerate() {
                let at = ((j + u) * hd + i) * 2;
                // SAFETY: `k` holds vis · hd halves and j + u < vis, i + 8 <= hd.
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                for (r, qrow) in qv.iter().enumerate() {
                    keyslot[r][0] = vfmaq_f32(keyslot[r][0], qrow.0, k0);
                    keyslot[r][1] = vfmaq_f32(keyslot[r][1], qrow.1, k1);
                }
            }
            i += 8;
        }
        for (u, keyslot) in a.iter().enumerate() {
            for (r, slot) in keyslot.iter().enumerate() {
                let v = vaddq_f32(hsum4_pairwise(slot[0]), hsum4_pairwise(slot[1]));
                *s.get_unchecked_mut(r * stride + j + u) = vgetq_lane_f32(vmulq_f32(v, sc), 0);
            }
        }
        j += 2;
    }
    // One key at a time: the same tree, so a key's score does not depend on the
    // block boundaries.
    while j < vis {
        for (r, qp) in qs.iter().enumerate() {
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i < hd {
                let at = ((j * hd) + i) * 2;
                // SAFETY: k holds vis · hd halves, i + 8 <= hd.
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                a0 = vfmaq_f32(a0, vld1q_f32(qp.add(i)), k0);
                a1 = vfmaq_f32(a1, vld1q_f32(qp.add(i + 4)), k1);
                i += 8;
            }
            let v = vaddq_f32(hsum4_pairwise(a0), hsum4_pairwise(a1));
            *s.get_unchecked_mut(r * stride + j) = vgetq_lane_f32(vmulq_f32(v, sc), 0);
        }
        j += 1;
    }
}

/// `((v0+v1)+(v2+v3))` in every lane: the score tree's four-lane reduction.
#[inline(always)]
unsafe fn hsum4_pairwise(v: float32x4_t) -> float32x4_t {
    let t = vpaddq_f32(v, v);
    vpaddq_f32(t, t)
}

/// [`mix_f16_quad`]'s scalar definition: four rows' own `α·o + Σ_j p_j · v_j`, one
/// separate multiply and one separate add per ascending key.
///
/// `o` holds four output rows of `hd` floats; `p` four weight rows of `n` weights, row
/// `r` at `p[r · p_stride ..]`; `v` the tile's halves.
#[cfg(target_arch = "aarch64")]
pub fn mix_f16_quad_scalar(
    o: &mut [f32],
    alphas: &[f32; 4],
    p: &[f32],
    p_stride: usize,
    n: usize,
    v: &[u8],
    hd: usize,
) {
    assert!(o.len() >= 4 * hd, "four output rows are needed");
    assert!(
        p.len() >= 3 * p_stride + n,
        "four score rows of n floats are needed"
    );
    assert!(v.len() >= n * hd * 2, "the V tile is shorter than n");
    for r in 0..4 {
        if alphas[r] != 1.0 {
            o[r * hd..(r + 1) * hd]
                .iter_mut()
                .for_each(|x| *x *= alphas[r]);
        }
    }
    for j in 0..n {
        for d in 0..hd {
            let at = (j * hd + d) * 2;
            let value = f16(u16::from_le_bytes([v[at], v[at + 1]]));
            for r in 0..4 {
                o[r * hd + d] += p[r * p_stride + j] * value;
            }
        }
    }
}

/// Four rows mixed over one F16 V tile: each row's output slice stays in registers
/// while the keys walk ascending, and each stored V half-vector is widened once for
/// all four rows.
///
/// The arithmetic is [`mix_f16_quad_scalar`]'s element for element — a separate
/// multiply and a separate add per key, in ascending key order, with `α·o` first — so
/// a row's bits are the single-row body's. All four rows mix the same `n` positions:
/// a row cut at another row's position would take an extra soft-max step, which moves
/// its denominator's summation, so the caller groups rows that share their context.
///
/// # Safety
/// The CPU must have NEON; `o` must hold four rows of `hd` floats, `p` four rows of
/// `n` weights at `p_stride`, `v` at least `n · hd · 2` bytes, and `hd` a multiple of
/// eight.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mix_f16_quad(
    o: &mut [f32],
    alphas: &[f32; 4],
    p: &[f32],
    p_stride: usize,
    n: usize,
    v: &[u8],
    hd: usize,
) {
    use std::arch::aarch64::*;
    assert!(o.len() >= 4 * hd, "four output rows are needed");
    assert!(
        p.len() >= 3 * p_stride + n,
        "four score rows of n floats are needed"
    );
    assert!(v.len() >= n * hd * 2, "the V tile is shorter than n");
    assert!(
        hd.is_multiple_of(8),
        "the tile body walks eight dims at a time"
    );
    for r in 0..4 {
        if alphas[r] != 1.0 {
            let mut d = 0;
            while d < hd {
                let at = r * hd + d;
                vst1q_f32(
                    o.as_mut_ptr().add(at),
                    vmulq_f32(vld1q_f32(o.as_ptr().add(at)), vdupq_n_f32(alphas[r])),
                );
                d += 4;
            }
        }
    }
    let vp = v.as_ptr();
    let mut d = 0;
    while d + 8 <= hd {
        let mut a = [[vdupq_n_f32(0.0); 2]; 4];
        for (r, acc) in a.iter_mut().enumerate() {
            acc[0] = vld1q_f32(o.as_ptr().add(r * hd + d));
            acc[1] = vld1q_f32(o.as_ptr().add(r * hd + d + 4));
        }
        for j in 0..n {
            let at = (j * hd + d) * 2;
            // SAFETY: `v` holds n · hd halves and d + 8 <= hd.
            let v0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(at) as *const u16)));
            let v1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(at + 8) as *const u16)));
            for (r, acc) in a.iter_mut().enumerate() {
                let w = vdupq_n_f32(*p.get_unchecked(r * p_stride + j));
                acc[0] = vaddq_f32(acc[0], vmulq_f32(w, v0));
                acc[1] = vaddq_f32(acc[1], vmulq_f32(w, v1));
            }
        }
        for (r, acc) in a.iter().enumerate() {
            vst1q_f32(o.as_mut_ptr().add(r * hd + d), acc[0]);
            vst1q_f32(o.as_mut_ptr().add(r * hd + d + 4), acc[1]);
        }
        d += 8;
    }
    // A head width that is not a multiple of eight never reaches this body (the
    // caller's gate), but the tail keeps the definition total.
    while d < hd {
        for r in 0..4 {
            let mut x = *o.get_unchecked(r * hd + d);
            for j in 0..n {
                let at = (j * hd + d) * 2;
                let value = f16(u16::from_le_bytes([*vp.add(at), *vp.add(at + 1)]));
                x += p[r * p_stride + j] * value;
            }
            *o.get_unchecked_mut(r * hd + d) = x;
        }
        d += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kq::{
        dot_q4k8_ref, dot_q5_08_ref, dot_q6k8_ref, dot_q8_08_ref, repack_q4k8, repack_q5_08,
        repack_q6k8, repack_q8_08, Q4K8, Q5_08, Q6K8, Q8_08,
    };

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

        fn f32(&mut self) -> f32 {
            let u = (self.next_u8() as u32) << 16 | (self.next_u8() as u32) << 8;
            u as f32 / 0x00FF_FFFF as f32 * 6.0 - 3.0
        }
    }

    fn activations(rng: &mut Rng, n: usize) -> QuantAct {
        let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        let mut a = QuantAct::with_capacity(n);
        a.fill(&x);
        a
    }

    /// A Q8_0 row of `n` weights with plausible scales.
    fn q8_0_row(rng: &mut Rng, n: usize) -> Vec<u8> {
        let mut w = rng.bytes(n / 32 * 34);
        for b in 0..n / 32 {
            let d = crate::quant::f32_to_f16(0.001 + (b as f32) * 0.0001);
            w[b * 34..b * 34 + 2].copy_from_slice(&d.to_le_bytes());
        }
        w
    }

    /// A Q8_0 row of `n_blk` blocks: the half-precision scale `d`, codes from `code`.
    fn q8_0_row_from(n_blk: usize, d: f32, code: fn(usize, usize) -> i8) -> Vec<u8> {
        let mut w = Vec::with_capacity(n_blk * 34);
        for t in 0..n_blk {
            w.extend_from_slice(&crate::quant::f32_to_f16(d).to_le_bytes());
            w.extend((0..32).map(|k| code(t, k) as u8));
        }
        w
    }

    /// An activation whose codes are all `code` and whose 32-block scales are `d`.
    fn q8_act(n: usize, code: i8, d: f32) -> QuantAct {
        q8_act_codes(&vec![code; n], d)
    }

    /// An activation from a code list, one 32-block scale `d`.
    fn q8_act_codes(codes: &[i8], d: f32) -> QuantAct {
        let mut a = QuantAct::with_capacity(codes.len());
        a.n = codes.len();
        a.q = codes.to_vec();
        a.d = vec![d; codes.len() / 32];
        a.sums = codes
            .chunks(32)
            .map(|c| c.iter().map(|&x| x as i32).sum())
            .collect();
        a
    }

    /// The row counts the Q8_0 i8mm body is checked at: every small count, the group
    /// boundaries, and the batch sizes the encoders actually use.
    fn q8_i8mm_row_counts() -> Vec<usize> {
        let mut v: Vec<usize> = (1..=40).collect();
        v.extend([63, 64, 65, 127, 128, 129, 511, 512, 513]);
        v
    }

    /// A Q4_K row of `n` weights: random codes, real-looking 6-bit scales and mins.
    fn q4_k_row(rng: &mut Rng, n: usize) -> Vec<u8> {
        let mut w = Vec::with_capacity(n / 256 * 144);
        for sb in 0..n / 256 {
            let d = crate::quant::f32_to_f16(0.002 + (sb as f32) * 0.0001);
            let dmin = crate::quant::f32_to_f16(0.001);
            w.extend_from_slice(&d.to_le_bytes());
            w.extend_from_slice(&dmin.to_le_bytes());
            // Twelve scale bytes, every bit of which is a field.
            w.extend_from_slice(&rng.bytes(12));
            w.extend_from_slice(&rng.bytes(128));
        }
        w
    }

    #[test]
    fn quantisation_matches_the_scalar_loop() {
        if !crate::cpu::have_neon() {
            eprintln!("skip: no NEON on this CPU");
            return;
        }
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for trial in 0..2000 {
            // Magnitudes around and inside the scale, so the half-way rule and the
            // clamp both get exercised.
            let inv: f32 = match trial % 4 {
                0 => 1.0,
                1 => 127.0,
                2 => 0.5,
                _ => 3.7,
            };
            let x: Vec<f32> = (0..32)
                .map(|_| (rng.f32() * inv.recip()) * if trial % 3 == 0 { 1.3 } else { 1.0 })
                .collect();
            let mut want = [0i8; 32];
            for (j, &v) in x.iter().enumerate() {
                want[j] = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
            let mut got = [0i8; 32];
            unsafe { quantize32(x.as_ptr(), inv, got.as_mut_ptr()) };
            assert_eq!(got, want, "trial {trial} inv {inv}");
            // And the absmax fold.
            let m = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert_eq!(
                unsafe { absmax(&x) }.to_bits(),
                m.to_bits(),
                "trial {trial}"
            );
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q8_0_matches_the_scalar_reference_bit_for_bit() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for n in [32usize, 64, 896, 4864] {
            let w = q8_0_row(&mut rng, n);
            let a = activations(&mut rng, n);
            let want = crate::quant::dot_scalar(Format::Q8_0, &w, &a);
            let got = unsafe { dot_q8_0(&w, &a) };
            assert_eq!(got.to_bits(), want.to_bits(), "n={n}: {got} vs {want}");
        }
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn q4_k_matches_the_scalar_reference_bit_for_bit() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        for n in [256usize, 512, 4864] {
            let w = q4_k_row(&mut rng, n);
            let a = activations(&mut rng, n);
            let want = crate::quant::dot_scalar(Format::Q4_K, &w, &a);
            let got = unsafe { dot_q4_k(&w, &a) };
            assert_eq!(got.to_bits(), want.to_bits(), "n={n}: {got} vs {want}");
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q8_0_batch_is_bit_identical_at_every_width() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let n = 896;
        let w = q8_0_row(&mut rng, n);
        let acts: Vec<QuantAct> = (0..16).map(|_| activations(&mut rng, n)).collect();
        let mut base = vec![0.0f32; acts.len()];
        for (b, a) in acts.iter().enumerate() {
            base[b] = unsafe { dot_q8_0(&w, a) };
        }
        for nb in [1usize, 2, 3, 4, 5, 7, 8, 16] {
            let mut got = vec![0.0f32; nb];
            assert!(unsafe { dot_batch(Format::Q8_0, &w, &acts[..nb], &mut got) });
            for b in 0..nb {
                assert_eq!(
                    got[b].to_bits(),
                    base[b].to_bits(),
                    "nb={nb} act {b}: {} vs {}",
                    got[b],
                    base[b]
                );
            }
        }
        // The eight-row decode kernel against the one-row kernel.
        let mut rows = Vec::with_capacity(8 * n / 32 * 34);
        let mut starts = [0usize; 8];
        for r in 0..8 {
            starts[r] = rows.len();
            rows.extend_from_slice(&q8_0_row(&mut rng, n));
        }
        let rb = n / 32 * 34;
        let mut eight = vec![0.0f32; 8];
        unsafe { q8_0_rows8(&rows, rb, &acts[0], &mut eight) };
        for r in 0..8 {
            let want = unsafe { dot_q8_0(&rows[starts[r]..starts[r] + rb], &acts[0]) };
            assert_eq!(eight[r].to_bits(), want.to_bits(), "row {r}");
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q8_08_gemv_and_gemm_agree_with_the_tile_reference() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0x0808_abcd_1234_5678);
        let n_blk = 4usize;
        let n = n_blk * 32;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q8_0_row(&mut rng, n)).collect();
        let mut tile = vec![0u8; n_blk * Q8_08::SIZE];
        for t in 0..n_blk {
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[t * 34..][..34]).collect();
            repack_q8_08(&src, &mut tile[t * Q8_08::SIZE..]);
        }
        let acts: Vec<QuantAct> = (0..8).map(|_| activations(&mut rng, n)).collect();
        for nb in 1..=8 {
            let mut got = vec![0.0f32; nb * 8];
            if nb <= 3 {
                unsafe { q8_08_gemv(&tile, n_blk, &acts[..nb], nb, &mut got) };
            } else {
                unsafe { q8_08_gemm(&tile, n_blk, &acts[..nb], nb, &mut got) };
            }
            for b in 0..nb {
                for r in 0..8 {
                    let want = dot_q8_08_ref(&tile, n_blk, r, &acts[b]);
                    assert_eq!(
                        got[b * 8 + r].to_bits(),
                        want.to_bits(),
                        "nb={nb} b={b} r={r}: {} vs {want}",
                        got[b * 8 + r]
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn q4k8_gemv_and_gemm_agree_with_the_reference_at_every_width() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0x5150_abcd_1234_5678);
        let n_sb = 2usize;
        let n = n_sb * 256;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q4_k_row(&mut rng, n)).collect();
        let mut tile = vec![0u8; n_sb * Q4K8::SIZE];
        for sb in 0..n_sb {
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[sb * 144..][..144]).collect();
            repack_q4k8(&src, &mut tile[sb * Q4K8::SIZE..]);
        }
        let acts: Vec<QuantAct> = (0..8).map(|_| activations(&mut rng, n)).collect();
        // The specification: the tile reference, row by row.
        let mut want = vec![0.0f32; 8 * 8];
        for b in 0..8 {
            for r in 0..8 {
                want[b * 8 + r] = dot_q4k8_ref(&tile, n_sb, r, &acts[b]);
            }
        }
        // The GEMV (1-3 rows) and the GEMM (4 and up) at every width the engine uses.
        for nb in 1..=8 {
            let mut got = vec![0.0f32; nb * 8];
            if nb <= 3 {
                unsafe { q4k8_gemv(&tile, n_sb, &acts[..nb], nb, &mut got) };
            } else {
                let mut kq = KqActs::default();
                kq.fill(&acts[..nb]);
                unsafe { q4k8_gemm(&tile, n_sb, &kq, nb, &mut got) };
            }
            for i in 0..nb * 8 {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "nb={nb} value {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q5_08_i8mm_gemm_is_bit_identical_to_the_sdot_gemv() {
        if !crate::cpu::have_i8mm() {
            eprintln!("skip: no i8mm on this CPU");
            return;
        }
        let mut rng = Rng(0x0f0f_1a2b_3c4d_5e6f);
        let n = 512usize;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q5_0_row(&mut rng, n)).collect();
        let n_blk = n / 32;
        let mut tile = vec![0u8; n_blk * Q5_08::SIZE];
        for t in 0..n_blk {
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[t * 22..][..22]).collect();
            repack_q5_08(&src, &mut tile[t * Q5_08::SIZE..]);
        }
        let acts: Vec<QuantAct> = (0..8).map(|_| activations(&mut rng, n)).collect();
        for nb in [4usize, 8] {
            let mut got = vec![0.0f32; nb * 8];
            unsafe { q5_08_gemm_i8mm(&tile, n_blk, &acts[..nb], nb, &mut got) };
            for b in 0..nb {
                for r in 0..8 {
                    let want = dot_q5_08_ref(&tile, n_blk, r, &acts[b]);
                    assert_eq!(
                        got[b * 8 + r].to_bits(),
                        want.to_bits(),
                        "nb={nb} b={b} r={r}: {} vs {want}",
                        got[b * 8 + r]
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q8_08_i8mm_gemm_is_bit_identical_to_the_sdot_gemm() {
        if !crate::cpu::have_i8mm() {
            eprintln!("skip: no i8mm on this CPU");
            return;
        }
        let mut rng = Rng(0x0808_1a2b_3c4d_5e6f);
        // Widths that are and are not multiples of the block: 32, 64, 96, 160 weights.
        for n_blk in [1usize, 2, 3, 5] {
            let n = n_blk * 32;
            let rows: Vec<Vec<u8>> = (0..8).map(|_| q8_0_row(&mut rng, n)).collect();
            let mut tile = vec![0u8; n_blk * Q8_08::SIZE];
            for t in 0..n_blk {
                let src: Vec<&[u8]> = rows.iter().map(|r| &r[t * 34..][..34]).collect();
                repack_q8_08(&src, &mut tile[t * Q8_08::SIZE..]);
            }
            let acts: Vec<QuantAct> = (0..513).map(|_| activations(&mut rng, n)).collect();
            for nb in q8_i8mm_row_counts() {
                let mut got = vec![0.0f32; nb * 8];
                let mut want = vec![0.0f32; nb * 8];
                unsafe {
                    q8_08_gemm_i8mm(&tile, n_blk, &acts[..nb], nb, &mut got);
                    q8_08_gemm(&tile, n_blk, &acts[..nb], nb, &mut want);
                }
                for b in 0..nb {
                    for r in 0..8 {
                        let i = b * 8 + r;
                        assert_eq!(
                            got[i].to_bits(),
                            want[i].to_bits(),
                            "the i8mm body and the sdot body disagree: n_blk={n_blk} nb={nb} b={b} r={r}: {} vs {}",
                            got[i],
                            want[i]
                        );
                        // And the specification, `quant::dot_scalar`'s Q8_0 expression.
                        let spec = dot_q8_08_ref(&tile, n_blk, r, &acts[b]);
                        assert_eq!(
                            got[i].to_bits(),
                            spec.to_bits(),
                            "the i8mm body and the tile reference disagree: n_blk={n_blk} nb={nb} b={b} r={r}: {} vs {spec}",
                            got[i]
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q8_08_i8mm_gemm_holds_the_extremes() {
        if !crate::cpu::have_i8mm() {
            eprintln!("skip: no i8mm on this CPU");
            return;
        }
        let n_blk = 2usize;
        let n = n_blk * 32;
        // Row r carries case r: all -128, all 127, alternating, random codes behind a
        // zero scale, and the two alternating patterns in between.
        let cases: [fn(usize, usize) -> i8; 8] = [
            |_, _| -128,
            |_, _| 127,
            |_, k| if k % 2 == 0 { -128 } else { 127 },
            |_, k| if k % 2 == 0 { 127 } else { -128 },
            |t, k| if (t + k) % 3 == 0 { -128 } else { 127 },
            |_, k| (k as i8).wrapping_mul(37).wrapping_sub(64),
            |t, _| if t % 2 == 0 { -128 } else { 127 },
            |_, _| 0,
        ];
        let rows: Vec<Vec<u8>> = cases
            .iter()
            .enumerate()
            .map(|(r, f)| {
                let d = if r == 3 { 0.0 } else { 0.02 + r as f32 * 0.001 };
                q8_0_row_from(n_blk, d, *f)
            })
            .collect();
        let mut tile = vec![0u8; n_blk * Q8_08::SIZE];
        for t in 0..n_blk {
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[t * 34..][..34]).collect();
            repack_q8_08(&src, &mut tile[t * Q8_08::SIZE..]);
        }
        // Activations at the int8 limits, and one row with every code at the two ends.
        let acts: Vec<QuantAct> = vec![
            q8_act(n, -128, 1.0),
            q8_act(n, 127, 1.0 / 127.0),
            q8_act(n, -128, 0.0),
            q8_act_codes(
                &(0..n)
                    .map(|i| if i % 2 == 0 { -128i8 } else { 127 })
                    .collect::<Vec<i8>>(),
                0.5,
            ),
        ];
        for nb in q8_i8mm_row_counts() {
            let acts: Vec<QuantAct> = (0..nb).map(|i| acts[i % acts.len()].clone()).collect();
            let mut got = vec![0.0f32; nb * 8];
            let mut want = vec![0.0f32; nb * 8];
            unsafe {
                q8_08_gemm_i8mm(&tile, n_blk, &acts, nb, &mut got);
                q8_08_gemm(&tile, n_blk, &acts, nb, &mut want);
            }
            for i in 0..nb * 8 {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "n_blk={n_blk} nb={nb} value {i}: {} vs {}",
                    got[i],
                    want[i]
                );
                let r = i % 8;
                let b = i / 8;
                let spec = dot_q8_08_ref(&tile, n_blk, r, &acts[b]);
                assert_eq!(
                    got[i].to_bits(),
                    spec.to_bits(),
                    "n_blk={n_blk} nb={nb} value {i}: {} vs {spec}",
                    got[i]
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn q6k8_i8mm_gemm_is_bit_identical_to_the_sdot_gemv() {
        if !crate::cpu::have_i8mm() {
            eprintln!("skip: no i8mm on this CPU");
            return;
        }
        let mut rng = Rng(0x6a6a_1a2b_3c4d_5e6f);
        let n_sb = 2usize;
        let n = n_sb * 256;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q6_k_row(&mut rng, n)).collect();
        let mut tile = vec![0u8; n_sb * Q6K8::SIZE];
        for sb in 0..n_sb {
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[sb * 210..][..210]).collect();
            repack_q6k8(&src, &mut tile[sb * Q6K8::SIZE..]);
        }
        let acts: Vec<QuantAct> = (0..8).map(|_| activations(&mut rng, n)).collect();
        for nb in [4usize, 8] {
            let mut kq = KqActs::default();
            kq.fill(&acts[..nb]);
            let mut got = vec![0.0f32; nb * 8];
            unsafe { q6k8_gemm_i8mm(&tile, n_sb, &kq, nb, &mut got) };
            for b in 0..nb {
                for r in 0..8 {
                    let want = dot_q6k8_ref(&tile, n_sb, r, &acts[b]);
                    assert_eq!(
                        got[b * 8 + r].to_bits(),
                        want.to_bits(),
                        "nb={nb} b={b} r={r}: {} vs {want}",
                        got[b * 8 + r]
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn q4k8_i8mm_gemm_is_bit_identical_to_the_sdot_gemv() {
        if !crate::cpu::have_i8mm() {
            eprintln!("skip: no i8mm on this CPU");
            return;
        }
        let mut rng = Rng(0x5150_1a2b_3c4d_5e6f);
        let n_sb = 2usize;
        let n = n_sb * 256;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q4_k_row(&mut rng, n)).collect();
        let mut tile = vec![0u8; n_sb * Q4K8::SIZE];
        for sb in 0..n_sb {
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[sb * 144..][..144]).collect();
            repack_q4k8(&src, &mut tile[sb * Q4K8::SIZE..]);
        }
        let acts: Vec<QuantAct> = (0..12).map(|_| activations(&mut rng, n)).collect();
        for nb in [4usize, 8, 12] {
            let mut kq = KqActs::default();
            kq.fill(&acts[..nb]);
            let mut got = vec![0.0f32; nb * 8];
            unsafe { q4k8_gemm_i8mm(&tile, n_sb, &kq, nb, &mut got) };
            for b in 0..nb {
                for r in 0..8 {
                    let want = dot_q4k8_ref(&tile, n_sb, r, &acts[b]);
                    assert_eq!(
                        got[b * 8 + r].to_bits(),
                        want.to_bits(),
                        "nb={nb} b={b} r={r}: {} vs {want}",
                        got[b * 8 + r]
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn the_repacked_tile_is_still_the_scalar_expression() {
        // A row of the tile, computed by the per-row kernel on the file's bytes, is *not* expected
        // to be bit-identical (the fold differs by design, `docs/design/s3.4.md` §4.2) — but it is
        // the same real number to a few ulps, which catches a wrong lane.
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0xfeed_face_0bad_c0de);
        let n = 512usize;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q4_k_row(&mut rng, n)).collect();
        let mut tile = vec![0u8; 2 * Q4K8::SIZE];
        for sb in 0..2 {
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[sb * 144..][..144]).collect();
            repack_q4k8(&src, &mut tile[sb * Q4K8::SIZE..]);
        }
        let a = activations(&mut rng, n);
        let mut out = vec![0.0f32; 8];
        unsafe { q4k8_gemv(&tile, 2, std::slice::from_ref(&a), 1, &mut out) };
        for r in 0..8 {
            let row = unsafe { dot_q4_k(&rows[r], &a) };
            let scale = row.abs().max(out[r].abs()).max(1.0);
            assert!(
                (out[r] - row).abs() <= 1e-4 * scale,
                "row {r}: tile {} vs per-row {row}",
                out[r]
            );
        }
    }

    /// A Q5_0 row of `n` weights with plausible scales and random codes.
    fn q5_0_row(rng: &mut Rng, n: usize) -> Vec<u8> {
        let mut w = rng.bytes(n / 32 * 22);
        for b in 0..n / 32 {
            let d = crate::quant::f32_to_f16(0.001 + (b as f32) * 0.0001);
            w[b * 22..b * 22 + 2].copy_from_slice(&d.to_le_bytes());
        }
        w
    }

    /// A Q6_K row of `n` weights: random codes, scales and a real-looking scale.
    fn q6_k_row(rng: &mut Rng, n: usize) -> Vec<u8> {
        let mut w = Vec::with_capacity(n / 256 * 210);
        for sb in 0..n / 256 {
            w.extend_from_slice(&rng.bytes(192));
            // Sixteen signed eight-bit scales: every byte is a valid scale.
            w.extend_from_slice(&rng.bytes(16));
            let d = crate::quant::f32_to_f16(0.002 + (sb as f32) * 0.0001);
            w.extend_from_slice(&d.to_le_bytes());
        }
        w
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q5_0_matches_the_scalar_reference_bit_for_bit() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for n in [32usize, 64, 896, 4864] {
            let w = q5_0_row(&mut rng, n);
            let a = activations(&mut rng, n);
            let want = crate::quant::dot_scalar(Format::Q5_0, &w, &a);
            let got = unsafe { dot_q5_0(&w, &a) };
            assert_eq!(got.to_bits(), want.to_bits(), "n={n}: {got} vs {want}");
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q5_0_rows8_and_the_tile_agree_with_the_reference_at_every_width() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0x8bad_f00d_dead_10cc);
        let n = 896usize;
        let n_blk = n / 32;
        // Eight file rows, the tiles repacked from them, and the one-row kernel.
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q5_0_row(&mut rng, n)).collect();
        let mut tile = vec![0u8; n_blk * Q5_08::SIZE];
        for t in 0..n_blk {
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[t * 22..][..22]).collect();
            repack_q5_08(&src, &mut tile[t * Q5_08::SIZE..]);
        }
        // The decode rows-eight kernel against the one-row kernel.
        let acts: Vec<QuantAct> = (0..8).map(|_| activations(&mut rng, n)).collect();
        let rb = n_blk * 22;
        let mut eight = vec![0.0f32; 8];
        unsafe { q5_0_rows8(&rows.concat(), rb, &acts[0], &mut eight) };
        for r in 0..8 {
            let want = unsafe { dot_q5_0(&rows[r], &acts[0]) };
            assert_eq!(eight[r].to_bits(), want.to_bits(), "rows8 row {r}");
        }
        // The tile GEMV and GEMM against the tile reference, at every width.
        let mut want = vec![0.0f32; 8 * 8];
        for b in 0..8 {
            for r in 0..8 {
                want[b * 8 + r] = dot_q5_08_ref(&tile, n_blk, r, &acts[b]);
            }
        }
        for nb in 1..=8 {
            let mut got = vec![0.0f32; nb * 8];
            if nb <= 3 {
                unsafe { q5_08_gemv(&tile, n_blk, &acts[..nb], nb, &mut got) };
            } else {
                unsafe { q5_08_gemm(&tile, n_blk, &acts[..nb], nb, &mut got) };
            }
            for i in 0..nb * 8 {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "nb={nb} value {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn q6_k_matches_the_scalar_reference_bit_for_bit() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0x1319_8a2e_0370_7344);
        for n in [256usize, 512, 4864] {
            let w = q6_k_row(&mut rng, n);
            let a = activations(&mut rng, n);
            let want = crate::quant::dot_scalar(Format::Q6_K, &w, &a);
            let got = unsafe { dot_q6_k(&w, &a) };
            assert_eq!(got.to_bits(), want.to_bits(), "n={n}: {got} vs {want}");
        }
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn q6k8_gemv_and_gemm_agree_with_the_reference_at_every_width() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0xa409_3822_299f_31d0);
        let n_sb = 2usize;
        let n = n_sb * 256;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q6_k_row(&mut rng, n)).collect();
        let mut tile = vec![0u8; n_sb * Q6K8::SIZE];
        for sb in 0..n_sb {
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[sb * 210..][..210]).collect();
            repack_q6k8(&src, &mut tile[sb * Q6K8::SIZE..]);
        }
        let acts: Vec<QuantAct> = (0..8).map(|_| activations(&mut rng, n)).collect();
        let mut want = vec![0.0f32; 8 * 8];
        for b in 0..8 {
            for r in 0..8 {
                want[b * 8 + r] = dot_q6k8_ref(&tile, n_sb, r, &acts[b]);
                // The tile reference is exactly the scalar expression, so the tier can
                // move to the tile without moving a bit.
                let scalar = crate::quant::dot_scalar(Format::Q6_K, &rows[r], &acts[b]);
                assert_eq!(
                    want[b * 8 + r].to_bits(),
                    scalar.to_bits(),
                    "row {r} act {b}: tile ref {} vs scalar {scalar}",
                    want[b * 8 + r]
                );
            }
        }
        for nb in 1..=8 {
            let mut got = vec![0.0f32; nb * 8];
            if nb <= 3 {
                unsafe { q6k8_gemv(&tile, n_sb, &acts[..nb], nb, &mut got) };
            } else {
                let mut kq = KqActs::default();
                kq.fill(&acts[..nb]);
                unsafe { q6k8_gemm(&tile, n_sb, &kq, nb, &mut got) };
            }
            for i in 0..nb * 8 {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "nb={nb} value {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn q6k8_matches_the_scalar_on_the_file_bytes_row_by_row() {
        // The repacked kernel against the file's own bytes, through the scalar
        // reference: Q6_K's fold is per super-block, so unlike Q4_K's tile there is
        // no last-bit difference to allow here.
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0x082e_fa98_ec4e_6c89);
        let n = 512usize;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q6_k_row(&mut rng, n)).collect();
        let mut tile = vec![0u8; 2 * Q6K8::SIZE];
        for sb in 0..2 {
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[sb * 210..][..210]).collect();
            repack_q6k8(&src, &mut tile[sb * Q6K8::SIZE..]);
        }
        let a = activations(&mut rng, n);
        let mut out = vec![0.0f32; 8];
        unsafe { q6k8_gemv(&tile, 2, std::slice::from_ref(&a), 1, &mut out) };
        for r in 0..8 {
            let row = crate::quant::dot_scalar(Format::Q6_K, &rows[r], &a);
            assert_eq!(out[r].to_bits(), row.to_bits(), "row {r}");
        }
    }

    // -----------------------------------------------------------------------
    // PQ2_0 (Bonsai 2's ternary format)
    // -----------------------------------------------------------------------

    /// A PQ2_0 row of `n` weights: one f16 scale per 128, ternary codes 0..2.
    #[cfg(feature = "fmt-ternary")]
    fn pq2_row(rng: &mut Rng, n: usize) -> Vec<u8> {
        let mut w = Vec::with_capacity(n / 128 * 34);
        for _ in 0..n / 128 {
            let d = 0.01 + (rng.next_u8() as f32) * 0.0004;
            w.extend_from_slice(&crate::quant::f32_to_f16(d).to_le_bytes());
            for _ in 0..32 {
                let mut b = 0u8;
                for s in 0..4 {
                    b |= (rng.next_u8() % 3) << (2 * s);
                }
                w.push(b);
            }
        }
        w
    }

    /// An activation in the ternary layout.
    #[cfg(feature = "fmt-ternary")]
    fn pq2_activations(rng: &mut Rng, n: usize) -> QuantAct {
        let x: Vec<f32> = (0..n).map(|_| rng.f32() * 3.0).collect();
        let mut a = QuantAct::with_capacity(n);
        a.ternary = true;
        a.fill(&x);
        a
    }

    #[test]
    #[cfg(feature = "fmt-ternary")]
    fn pq2_matches_the_scalar_reference() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for n in [128usize, 384, 4096, 12288] {
            let w = pq2_row(&mut rng, n);
            for k in 1..=9 {
                let acts: Vec<QuantAct> = (0..k).map(|_| pq2_activations(&mut rng, n)).collect();
                let mut got = vec![0.0f32; k];
                unsafe { dot_batch_pq2(&w, &acts, &mut got) };
                for (i, a) in acts.iter().enumerate() {
                    let want = crate::quant::dot_scalar(Format::PQ2_0, &w, a);
                    let tol = 1e-4 * want.abs().max(1.0);
                    assert!(
                        (got[i] - want).abs() <= tol,
                        "n={n} k={k} row {i}: {} vs {want}",
                        got[i]
                    );
                }
            }
        }
    }

    /// Every pass width gives a row the same bits: the two-row decode GEMV against
    /// the one-row kernel, and the batched GEMM against the single-activation one.
    #[test]
    #[cfg(feature = "fmt-ternary")]
    fn pq2_rows_are_independent_of_the_pass_width() {
        if !crate::cpu::have_dotprod() {
            eprintln!("skip: no asimddp on this CPU");
            return;
        }
        let mut rng = Rng(0xa11c_e5a1_1ce5_a11c);
        for n in [128usize, 4096] {
            for rows in 1..=11usize {
                let w: Vec<u8> = (0..rows).flat_map(|_| pq2_row(&mut rng, n)).collect();
                let rb = n / 128 * 34;
                let a = pq2_activations(&mut rng, n);
                let mut got = vec![0.0f32; rows];
                unsafe { dot_rows_pq2(&w, rb, rows, std::slice::from_ref(&a), &mut got) };
                for row in 0..rows {
                    let want = unsafe { dot_pq2(&w[row * rb..(row + 1) * rb], &a) };
                    assert_eq!(
                        got[row].to_bits(),
                        want.to_bits(),
                        "n={n} rows={rows} row={row}"
                    );
                }
            }
        }
        // GEMV (one activation) against GEMM (four) on the same row.
        let n = 4096usize;
        let rows = 6usize;
        let w: Vec<u8> = (0..rows).flat_map(|_| pq2_row(&mut rng, n)).collect();
        let rb = n / 128 * 34;
        let acts: Vec<QuantAct> = (0..4).map(|_| pq2_activations(&mut rng, n)).collect();
        let mut gemm = vec![0.0f32; rows * 4];
        unsafe { dot_rows_pq2(&w, rb, rows, &acts, &mut gemm) };
        for b in 0..4 {
            let mut gemv = vec![0.0f32; rows];
            unsafe { dot_rows_pq2(&w, rb, rows, &acts[b..b + 1], &mut gemv) };
            for r in 0..rows {
                assert_eq!(
                    gemm[r * 4 + b].to_bits(),
                    gemv[r].to_bits(),
                    "act {b} row {r}: decode and prefill disagree"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // The 8-bit prompt form (S5.4d): the NEON bodies against the twin
    // -----------------------------------------------------------------------

    /// The NEON build of the 8-bit form is the twin's, byte for byte, on random
    /// tiles and on every edge tile: all-zero, one huge weight, the six-bit
    /// limits, exact halves, Q6_K's exact case and one just outside it.
    #[test]
    fn the_eight_bit_build_is_bit_identical_to_the_twin_on_neon() {
        if !crate::cpu::have_neon() {
            return;
        }
        let mut rng = crate::kq8::tiles::Rng(0x8A8A_1111);
        let q4: Vec<(Vec<u8>, usize)> = vec![
            (crate::kq8::tiles::zero_tile(), 1),
            (crate::kq8::tiles::huge_tile(), 1),
            (crate::kq8::tiles::limits_tile(), 1),
            (crate::kq8::tiles::tie_tile(), 1),
            (crate::kq8::tiles::q4k_tile(&mut rng, 1), 1),
            (crate::kq8::tiles::q4k_tile(&mut rng, 3), 3),
        ];
        for (tile, n_sb) in q4.iter() {
            let mut want = vec![0u8; n_sb * crate::kq8::Q8K_SIZE];
            let mut got = vec![0u8; n_sb * crate::kq8::Q8K_SIZE];
            crate::kq8::build_q4k(tile, *n_sb, &mut want);
            unsafe { q8k_build_q4k_neon(tile, *n_sb, &mut got) };
            assert_eq!(want, got, "the Q4_K build differs on {n_sb} super-blocks");
        }
        let q6: Vec<(Vec<u8>, usize)> = vec![
            (crate::kq8::tiles::exact_q6k_tile(), 1),
            (crate::kq8::tiles::outside_q6k_tile(), 1),
            (crate::kq8::tiles::q6k_tile(&mut rng, 1), 1),
            (crate::kq8::tiles::q6k_tile(&mut rng, 3), 3),
        ];
        for (tile, n_sb) in q6.iter() {
            let mut want = vec![0u8; n_sb * crate::kq8::Q8K_SIZE];
            let mut got = vec![0u8; n_sb * crate::kq8::Q8K_SIZE];
            crate::kq8::build_q6k(tile, *n_sb, &mut want);
            unsafe { q8k_build_q6k_neon(tile, *n_sb, &mut got) };
            assert_eq!(want, got, "the Q6_K build differs on {n_sb} super-blocks");
        }
        #[cfg(feature = "fmt-legacy")]
        {
            for n_blk in [8usize, 16, 21] {
                let tile = crate::kq8::tiles::q5_08_tile(&mut rng, n_blk);
                let n_sb = n_blk.div_ceil(8);
                let mut want = vec![0u8; n_sb * crate::kq8::Q8K_SIZE];
                let mut got = vec![0u8; n_sb * crate::kq8::Q8K_SIZE];
                crate::kq8::build_q5_0(&tile, n_blk, &mut want);
                unsafe { q8k_build_q5_0_neon(&tile, n_blk, &mut got) };
                assert_eq!(want, got, "the Q5_0 build differs on {n_blk} blocks");
            }
        }
    }

    /// The NEON multiply is the twin's, bit for bit, at every activation width:
    /// every width from one to forty, the rows either side of the four-row group,
    /// and 63/64/65, 127/128/129 and 511/512/513.
    ///
    /// The Q4_K random form, the tie tile and the Q6_K block just outside the exact
    /// case all carry codes at the int8 limits; the Q5_0 form is built from random
    /// blocks through the same construction.
    #[test]
    fn the_eight_bit_kernel_is_bit_identical_to_the_twin_on_neon() {
        if !crate::cpu::have_dotprod() {
            return;
        }
        let mut rng = Rng(0x8A8A_2222);
        let n_sb = 2;
        let n = n_sb * 256;
        let mut forms: Vec<Vec<u8>> = Vec::new();
        let mut trng = crate::kq8::tiles::Rng(0x8A8A_4444);
        let tile = crate::kq8::tiles::q4k_tile(&mut trng, n_sb);
        let mut f = vec![0u8; crate::kq8::form_size(n_sb)];
        crate::kq8::build_q4k(&tile, n_sb, &mut f);
        forms.push(f);
        let tie = crate::kq8::tiles::tie_tile();
        let mut f = vec![0u8; crate::kq8::form_size(1)];
        crate::kq8::build_q4k(&tie, 1, &mut f);
        forms.push(f);
        let out6 = crate::kq8::tiles::outside_q6k_tile();
        let mut f = vec![0u8; crate::kq8::form_size(1)];
        crate::kq8::build_q6k(&out6, 1, &mut f);
        forms.push(f);
        #[cfg(feature = "fmt-legacy")]
        {
            let n_blk = 16;
            let tile5 = crate::kq8::tiles::q5_08_tile(&mut trng, n_blk);
            let mut f = vec![0u8; (n_blk / 8) * crate::kq8::Q8K_SIZE];
            crate::kq8::build_q5_0(&tile5, n_blk, &mut f);
            forms.push(f);
        }
        let mut widths: Vec<usize> = (1..=40).collect();
        widths.extend_from_slice(&[63, 64, 65, 127, 128, 129, 511, 512, 513]);
        let many: Vec<QuantAct> = (0..513).map(|_| activations(&mut rng, n)).collect();
        for &nb in widths.iter() {
            let mut kq = KqActs::default();
            kq.fill(&many[..nb]);
            for form in forms.iter() {
                let sb = form.len() / crate::kq8::Q8K_SIZE;
                let mut got = vec![0.0f32; nb * 8];
                unsafe { q8k_gemm_sdot(form, sb, &kq, nb, &mut got) };
                for b in 0..nb {
                    for r in 0..8 {
                        let want = crate::kq8::dot_q8k(form, sb, r, &kq, b);
                        assert_eq!(
                            got[b * 8 + r].to_bits(),
                            want.to_bits(),
                            "sdot, width {nb}, row {r}, activation {b}: {} against {want}",
                            got[b * 8 + r]
                        );
                    }
                }
                if crate::cpu::have_i8mm() {
                    let mut got = vec![0.0f32; nb * 8];
                    unsafe { q8k_gemm_i8mm(form, sb, &kq, nb, &mut got) };
                    for b in 0..nb {
                        for r in 0..8 {
                            let want = crate::kq8::dot_q8k(form, sb, r, &kq, b);
                            assert_eq!(
                                got[b * 8 + r].to_bits(),
                                want.to_bits(),
                                "i8mm, width {nb}, row {r}, activation {b}: {} against {want}",
                                got[b * 8 + r]
                            );
                        }
                    }
                }
            }
        }
    }

    /// The tier entry point takes `smmla` on `neon-i8mm` and `sdot` on
    /// `neon-dotprod`; running the tests under `PEREGRINE_ISA` shows each cap
    /// picking its own body.
    #[test]
    fn each_tier_takes_its_own_eight_bit_body() {
        if !crate::cpu::have_dotprod() {
            return;
        }
        let mut rng = Rng(0x8A8A_5555);
        let n_sb = 2;
        let n = n_sb * 256;
        let tile = crate::kq8::tiles::q4k_tile(&mut crate::kq8::tiles::Rng(0x8A8A_6666), n_sb);
        let mut form = vec![0u8; crate::kq8::form_size(n_sb)];
        crate::kq8::build_q4k(&tile, n_sb, &mut form);
        let acts: Vec<QuantAct> = (0..4).map(|_| activations(&mut rng, n)).collect();
        let mut kq = KqActs::default();
        kq.fill(&acts);
        let mut out = vec![0.0f32; 4 * 8];
        Q8K_SDOT_CALLS.with(|c| c.set(0));
        Q8K_I8MM_CALLS.with(|c| c.set(0));
        unsafe { q8k_gemm(&form, n_sb, &kq, 4, &mut out) };
        if crate::cpu::have_i8mm() {
            assert_eq!(
                Q8K_I8MM_CALLS.with(|c| c.get()),
                1,
                "the i8mm tier took another body"
            );
            assert_eq!(
                Q8K_SDOT_CALLS.with(|c| c.get()),
                0,
                "the i8mm tier fell back to sdot"
            );
        } else {
            assert_eq!(
                Q8K_SDOT_CALLS.with(|c| c.get()),
                1,
                "the dotprod tier took another body"
            );
            assert_eq!(
                Q8K_I8MM_CALLS.with(|c| c.get()),
                0,
                "the dotprod tier took the i8mm body"
            );
        }
        // The i8mm body's 1–3-row tail stays on sdot, so the result is still the
        // twin's (checked above at every width); here only the full group is counted.
        let _ = &mut out;
    }

    /// S5.4e: the engine's dispatch (`MatKernels::q8k_gemm`/`q8k_build`, the
    /// table's entry points) takes this file's body and builder exactly where
    /// the table says this tier has them, and the bits stay the twin's.
    #[test]
    fn the_engine_dispatch_takes_this_files_bodies() {
        let mk = crate::table::MatKernels::get();
        assert_eq!(
            mk.q8k_simd(),
            crate::cpu::have_dotprod(),
            "q8k_simd must say what the tier's multiply does"
        );
        let mut rng = Rng(0x8A8A_9999);
        let n_sb = 2;
        let n = n_sb * 256;
        let tile = crate::kq8::tiles::q4k_tile(&mut crate::kq8::tiles::Rng(0x8A8A_AAAA), n_sb);
        let mut form = vec![0u8; crate::kq8::form_size(n_sb)];
        let mut twin = vec![0u8; crate::kq8::form_size(n_sb)];
        crate::kq8::build_q4k(&tile, n_sb, &mut twin);
        mk.q8k_build(Format::Q4_K, &tile, n_sb, &mut form);
        assert_eq!(form, twin, "the dispatch's builder is not the twin's bytes");
        let acts: Vec<QuantAct> = (0..4).map(|_| activations(&mut rng, n)).collect();
        let mut kq = KqActs::default();
        kq.fill(&acts);
        let mut out = vec![0.0f32; 4 * 8];
        Q8K_SDOT_CALLS.with(|c| c.set(0));
        Q8K_I8MM_CALLS.with(|c| c.set(0));
        mk.q8k_gemm(&form, n_sb, &kq, 4, &mut out);
        let (sdot, i8mm) = (
            Q8K_SDOT_CALLS.with(|c| c.get()),
            Q8K_I8MM_CALLS.with(|c| c.get()),
        );
        if crate::cpu::have_dotprod() {
            assert_eq!(
                (sdot, i8mm),
                if crate::cpu::have_i8mm() {
                    (0, 1)
                } else {
                    (1, 0)
                },
                "the dispatch took another body than the tier's own"
            );
            assert!(
                mk.q8k_body().is_some_and(|b| b.starts_with("NEON")),
                "a tier with dotprod must report a NEON body"
            );
        } else {
            assert_eq!(
                (sdot, i8mm),
                (0, 0),
                "a tier without dotprod must stay on the scalar twin"
            );
        }
        for b in 0..4 {
            for r in 0..8 {
                let want = crate::kq8::dot_q8k(&form, n_sb, r, &kq, b);
                assert_eq!(
                    out[b * 8 + r].to_bits(),
                    want.to_bits(),
                    "the dispatch, row {r}, activation {b}"
                );
            }
        }
    }
}
