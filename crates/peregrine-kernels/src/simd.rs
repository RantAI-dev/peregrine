//! AVX2 fused dequantise-and-dot.
//!
//! These are the kernels that decide whether the engine's roofline claims apply to
//! itself. With the scalar path the measured compute roof on the reference machine
//! is 28.5 GFLOP/s against 30.6 GB/s of bandwidth — a ridge point of 0.93 FLOP/byte,
//! *below* this workload's 6.2 FLOP/byte. A scalar decode step is therefore
//! compute-bound, and every statement it makes about memory bandwidth describes our
//! own arithmetic instead. Reaching the vector units moves the compute roof by more
//! than an order of magnitude and puts the workload back on the bandwidth slope
//! where the model says it belongs.
//!
//! # The instruction that shapes everything
//!
//! `VPMADDUBSW` multiplies **unsigned** bytes by **signed** bytes, summing adjacent
//! pairs into 16-bit lanes. Every quantised format here stores unsigned codes, so
//! the codes go in the unsigned operand and the int8 activation in the signed one,
//! with the format's offset applied afterwards via the bias identity in `quant.rs`.
//! That is why the kernels never reconstruct a signed weight.
//!
//! Saturation is the trap. `VPMADDUBSW` saturates its 16-bit result, and it sums two
//! products, so the largest code times the largest activation must stay under 16384
//! to be safe. Q6_K has the widest codes at 63, and `63 × 127 × 2 = 16002`, which
//! fits — but only because activations are clamped to ±127 rather than ±128 when
//! they are quantised. That clamp in `QuantAct::fill` is load-bearing here.
//!
//! # Correctness
//!
//! Every kernel has a scalar twin in `quant.rs`, and `tests/simd.rs` asserts they
//! agree on the real weight file (acceptance criterion A4). The scalar path is the
//! specification; this file is an optimisation of it and is never the reference.
//!
//! # Float formats (F32, F16, BF16)
//!
//! Float weights run on the full `f32` activation (`QuantAct::xf`) rather than the
//! int8-quantised codes. Decoding uses native float load for F32, F16C (`_mm256_cvtph_ps`)
//! for F16, and integer shift-left-16 for BF16. Accumulation uses 8 vector lanes with
//! uncontracted multiply-add (`_mm256_mul_ps` followed by `_mm256_add_ps`), matching the
//! scalar twin bit for bit.

#![cfg(target_arch = "x86_64")]
#![cfg_attr(
    not(all(
        feature = "fmt-kquant",
        feature = "fmt-ternary",
        feature = "fmt-iq",
        feature = "fmt-mxfp4",
        feature = "fmt-legacy"
    )),
    allow(dead_code)
)]

use crate::format::{f16, Format};
use crate::quant::QuantAct;
use std::arch::x86_64::*;

/// Whether the AVX2 path may run. `is_x86_feature_detected!` caches internally, so
/// this is a load of a relaxed atomic after the first call.
#[inline]
pub fn have_avx2() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::cpu_has!("avx2"))
}

#[inline(always)]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    hsum_i32_128(_mm_add_epi32(lo, hi))
}

#[inline(always)]
unsafe fn hsum_i32_128(v: __m128i) -> i32 {
    let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_01_00_01));
    _mm_cvtsi128_si32(s)
}

/// `Σ codes·acts` for 32 unsigned codes against 32 signed activations.
#[inline(always)]
unsafe fn dot32_u8_i8(codes: __m256i, acts: __m256i) -> i32 {
    let prod = _mm256_maddubs_epi16(codes, acts); // 16 × i16
    hsum_i32(_mm256_madd_epi16(prod, _mm256_set1_epi16(1)))
}

#[inline(always)]
unsafe fn acts32(a: &QuantAct, block: usize) -> __m256i {
    _mm256_loadu_si256(a.q.as_ptr().add(block * 32) as *const __m256i)
}

#[target_feature(enable = "avx2")]
unsafe fn dot_q8_0(w: &[u8], a: &QuantAct) -> f32 {
    let mut acc = 0.0f32;
    for (b, blk) in w.chunks_exact(34).enumerate() {
        let wq = _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i);
        let aq = acts32(a, b);
        // Q8_0 codes are signed, so the unsigned operand must be |w| and the sign
        // moved onto the activation. `_mm256_sign_epi8(x, y)` negates x where y is
        // negative and zeroes it where y is zero — exactly the two moves needed.
        let absw = _mm256_sign_epi8(wq, wq);
        let sa = _mm256_sign_epi8(aq, wq);
        let s = dot32_u8_i8(absw, sa);
        acc += f16(u16::from_le_bytes([blk[0], blk[1]])) * a.d[b] * s as f32;
    }
    acc
}

/// Spread the 32 fifth-bits of a Q5_0 block into one `0x10`-or-`0` byte per weight.
///
/// Byte `i` of the result must carry bit `i` of `qh`. `_mm256_shuffle_epi8` works
/// inside 128-bit lanes, so the broadcast `qh` is gathered per lane: bytes 0..8 take
/// `qh` byte 0, 8..16 take byte 1, and the upper lane takes bytes 2 and 3.
#[inline(always)]
unsafe fn q5_high_bits(qh: u32) -> __m256i {
    let spread = _mm256_set_epi8(
        3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, // upper lane: bytes 3,2
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, // lower lane: bytes 1,0
    );
    let bit = _mm256_set_epi8(
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1,
        -128, 64, 32, 16, 8, 4, 2, 1,
    );
    let v = _mm256_shuffle_epi8(_mm256_set1_epi32(qh as i32), spread);
    let masked = _mm256_and_si256(v, bit);
    // cmpeq against the bit pattern gives 0xFF exactly where the bit was set; a
    // non-zero test would be wrong because the masked byte is the bit's own value.
    let set = _mm256_cmpeq_epi8(masked, bit);
    _mm256_and_si256(set, _mm256_set1_epi8(0x10))
}

#[target_feature(enable = "avx2")]
unsafe fn dot_q4_0(w: &[u8], a: &QuantAct) -> f32 {
    let lownib = _mm256_set1_epi8(0x0F);
    let mut acc = 0.0f32;
    for (b, blk) in w.chunks_exact(18).enumerate() {
        let qs = _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i);
        // Weights 0..16 are the low nibbles, 16..32 the high nibbles — the same
        // interleaving the scalar twin spells out.
        let lo = _mm_and_si128(qs, _mm256_castsi256_si128(lownib));
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), _mm256_castsi256_si128(lownib));
        let codes = _mm256_set_m128i(hi, lo);
        let s = dot32_u8_i8(codes, acts32(a, b));
        // Codes are [0,15] with offset 8: one correction for the whole block.
        acc += f16(u16::from_le_bytes([blk[0], blk[1]])) * a.d[b] * (s - 8 * a.sums[b]) as f32;
    }
    acc
}

#[target_feature(enable = "avx2")]
unsafe fn dot_q5_0(w: &[u8], a: &QuantAct) -> f32 {
    let lownib = _mm256_set1_epi8(0x0F);
    let mut acc = 0.0f32;
    for (b, blk) in w.chunks_exact(22).enumerate() {
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = _mm_loadu_si128(blk.as_ptr().add(6) as *const __m128i);
        // Weights 0..16 are the low nibbles, 16..32 the high nibbles — the same
        // interleaving the scalar twin spells out.
        let lo = _mm_and_si128(qs, _mm256_castsi256_si128(lownib));
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), _mm256_castsi256_si128(lownib));
        let nib = _mm256_set_m128i(hi, lo);
        let codes = _mm256_or_si256(nib, q5_high_bits(qh));
        let s = dot32_u8_i8(codes, acts32(a, b));
        // Codes are [0,31] with offset 16: one correction for the whole block.
        acc += f16(u16::from_le_bytes([blk[0], blk[1]])) * a.d[b] * (s - 16 * a.sums[b]) as f32;
    }
    acc
}

/// Signed 8-bit pairwise multiply-add into 16-bit lanes via `VPMADDUBSW`.
///
/// `VPMADDUBSW` multiplies unsigned bytes by signed bytes. To multiply signed `w` by
/// signed `a`, we take `|w|` as the unsigned operand via `sign_epi8(w, w)` and move
/// the sign of `w` onto `a` via `sign_epi8(a, w)`. Because `|w| · (sign(w) · a) = w · a`,
/// this computes the exact signed pairwise products.
#[inline(always)]
unsafe fn mul_add_epi8(w: __m256i, a: __m256i) -> __m256i {
    let absw = _mm256_sign_epi8(w, w);
    let sa = _mm256_sign_epi8(a, w);
    _mm256_maddubs_epi16(absw, sa)
}

#[target_feature(enable = "avx2")]
unsafe fn dot_iq4_nl(w: &[u8], a: &QuantAct) -> f32 {
    let lownib = _mm_set1_epi8(0x0F);
    let table = _mm_loadu_si128(crate::quant::KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let mut acc = 0.0f32;
    for (b, blk) in w.chunks_exact(18).enumerate() {
        let qs = _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i);
        let lo = _mm_and_si128(qs, lownib);
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), lownib);
        let w_lo = _mm_shuffle_epi8(table, lo);
        let w_hi = _mm_shuffle_epi8(table, hi);
        let w32 = _mm256_set_m128i(w_hi, w_lo);
        let aq = acts32(a, b);
        let prod = mul_add_epi8(w32, aq);
        let s = hsum_i32(_mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
        acc += f16(u16::from_le_bytes([blk[0], blk[1]])) * a.d[b] * s as f32;
    }
    acc
}

#[target_feature(enable = "avx2")]
unsafe fn dot_iq4_xs(w: &[u8], a: &QuantAct) -> f32 {
    debug_assert!(!a.qk.is_empty(), "k-quant rows need the per-256 activation");
    let lownib = _mm_set1_epi8(0x0F);
    let table = _mm_loadu_si128(crate::quant::KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let mut acc = 0.0f32;
    for (sb, blk) in w.chunks_exact(136).enumerate() {
        let d = f16(u16::from_le_bytes([blk[0], blk[1]]));
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = blk.as_ptr().add(8);
        let mut sumi = _mm256_setzero_si256();
        for ib in 0..8 {
            let q4bits = _mm_loadu_si128(qs.add(ib * 16) as *const __m128i);
            let lo = _mm_and_si128(q4bits, lownib);
            let hi = _mm_and_si128(_mm_srli_epi16(q4bits, 4), lownib);
            let w_lo = _mm_shuffle_epi8(table, lo);
            let w_hi = _mm_shuffle_epi8(table, hi);
            let w32 = _mm256_set_m128i(w_hi, w_lo);
            let aq = _mm256_loadu_si256(a.qk.as_ptr().add(sb * 256 + ib * 32) as *const __m256i);
            let p16 = mul_add_epi8(w32, aq);
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as i16
                | ((((scales_h >> (2 * ib)) & 3) << 4) as i16);
            let scale = ls - 32;
            let p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(scale));
            sumi = _mm256_add_epi32(sumi, p32);
        }
        let sumi_scalar = hsum_i32(sumi);
        acc += a.dk[sb] * (d * sumi_scalar as f32);
    }
    acc
}

/// # Safety
/// Caller must ensure AVX2 is available. `w` must have at least 17 bytes per 32-element block
/// and `a` must hold corresponding activations.
#[target_feature(enable = "avx2")]
unsafe fn dot_mxfp4(w: &[u8], a: &QuantAct) -> f32 {
    let lownib = _mm_set1_epi8(0x0F);
    let table = _mm_loadu_si128(crate::quant::KVALUES_MXFP4.as_ptr() as *const __m128i);
    let mut acc = 0.0f32;
    for (b, blk) in w.chunks_exact(17).enumerate() {
        let qs = _mm_loadu_si128(blk.as_ptr().add(1) as *const __m128i);
        let lo = _mm_and_si128(qs, lownib);
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), lownib);
        let w_lo = _mm_shuffle_epi8(table, lo);
        let w_hi = _mm_shuffle_epi8(table, hi);
        let w32 = _mm256_set_m128i(w_hi, w_lo);
        let aq = acts32(a, b);
        let prod = mul_add_epi8(w32, aq);
        let s = hsum_i32(_mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
        acc += crate::format::e8m0_to_f32_half(blk[0]) * a.d[b] * s as f32;
    }
    acc
}

/// The byte-wise reference `unpack_q4k_scales` is tested against.
#[cfg(test)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// All eight Q4_K scales and mins of a super-block at once: `get_scale_min_k4` for
/// `j = 0..8`, done as four 32-bit word operations (ggml's `kmask` unpacking) instead
/// of eight branchy byte extractions. Bytes 0..8 of the result are the scales, 8..16
/// the mins.
#[inline(always)]
fn unpack_q4k_scales(q: &[u8]) -> [u8; 16] {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let w = |i: usize| u32::from_le_bytes([q[i], q[i + 1], q[i + 2], q[i + 3]]);
    let (u0, u1, u2) = (w(0), w(4), w(8));
    let m_hi = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
    let m_lo = u1 & KMASK1;
    let s_hi = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
    let s_lo = u0 & KMASK1;
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&s_lo.to_le_bytes());
    out[4..8].copy_from_slice(&s_hi.to_le_bytes());
    out[8..12].copy_from_slice(&m_lo.to_le_bytes());
    out[12..16].copy_from_slice(&m_hi.to_le_bytes());
    out
}

/// The eight Q4_K scales as broadcast 16-bit vectors — `_mm256_set1_epi16(sc[k])`
/// for `k = 0..8` — from one widening load and one byte shuffle each, instead of a
/// byte extract, a GPR→vector move and a broadcast (two port-5 µops) per scale.
#[inline(always)]
unsafe fn q4k_scale_vecs(sm: &[u8; 16]) -> [__m256i; 8] {
    // Lanes 0..8 hold the scales, 8..16 the mins; duplicate the scale half.
    let s16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(sm.as_ptr() as *const __m128i));
    let sc = _mm256_permute4x64_epi64::<0x44>(s16);
    // The controls are a static table, loaded: built with `_mm256_setr_epi8` from
    // loop-computed bytes they were assembled byte by byte on every super-block
    // (measured: the Q4_K kernel ran 7× slower).
    let ctl = BCAST16.as_ptr() as *const __m256i;
    // spelled out, not `std::array::from_fn`. Whether LLVM inlines `from_fn` into the kernel depends on
    // how much unrelated code the crate holds; when it does not, every super-block pays a call and an array
    // returned through memory (the Q4_K kernel ran 7x slower once the batching engine was added to the crate).
    let t = |k: usize| _mm256_shuffle_epi8(sc, _mm256_loadu_si256(ctl.add(k)));
    [t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7)]
}

/// `pshufb` controls that broadcast 16-bit lane `k` of each 128-bit half to the
/// whole half, `k = 0..8`.
static BCAST16: [[u8; 32]; 8] = {
    let mut t = [[0u8; 32]; 8];
    let mut k = 0;
    while k < 8 {
        let mut i = 0;
        while i < 32 {
            t[k][i] = (2 * k + i % 2) as u8;
            i += 1;
        }
        k += 1;
    }
    t
};

/// Q6_K's controls: index `c·4 + g` broadcasts 16-bit lane `2g` to the low half and
/// lane `2g + 1` to the high half (both halves of the source hold the same 8 lanes).
static Q6K_SPLIT16: [[u8; 32]; 4] = {
    let mut t = [[0u8; 32]; 4];
    let mut g = 0;
    while g < 4 {
        let mut i = 0;
        while i < 32 {
            let lane = if i < 16 { 2 * g } else { 2 * g + 1 };
            t[g][i] = (2 * lane + i % 2) as u8;
            i += 1;
        }
        g += 1;
    }
    t
};

/// The eight Q4_K mins as sixteen 16-bit lanes, each repeated for the two 16-weight
/// runs of its 32-weight sub-block, so one `madd_epi16` against an activation's sixteen
/// run sums yields the min term as eight i32 partial sums with no `hadd`.
/// Built once per super-block and shared by every activation.
#[inline(always)]
unsafe fn q4k_mins16(mins: &[u8]) -> __m256i {
    let m = _mm_loadl_epi64(mins.as_ptr() as *const __m128i);
    _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(m, m))
}

/// One activation's super-block result, entirely in vector form:
/// `Σ w·a = dk·(d·Σ sc·q·a − dmin·Σ m·a)`. The scalar version reduced the min term to
/// a scalar per super-block per activation — a horizontal sum, an int→float convert and
/// a scalar multiply-add, about a dozen instructions against the eight the dot product
/// itself costs — and kept a second accumulator for it. Here both stay as eight lanes
/// and the one horizontal reduction happens once per row.
#[inline(always)]
unsafe fn q4k_fold(
    acc: __m256,
    sumi: __m256i,
    mins16: __m256i,
    bsums: *const i16,
    dv: __m256,
    dminv: __m256,
    dk: f32,
) -> __m256 {
    let minterm = _mm256_madd_epi16(mins16, _mm256_loadu_si256(bsums as *const __m256i));
    let x = _mm256_fnmadd_ps(
        dminv,
        _mm256_cvtepi32_ps(minterm),
        _mm256_mul_ps(dv, _mm256_cvtepi32_ps(sumi)),
    );
    _mm256_fmadd_ps(x, _mm256_set1_ps(dk), acc)
}

/// # Safety
/// Caller must have checked `have_avx2()`. `w` must be exactly `row_bytes(fmt, a.n)`
/// long and `a` must hold `a.n` codes; both hold for every call from `matmul`.
pub unsafe fn dot(fmt: Format, w: &[u8], a: &QuantAct) -> f32 {
    match fmt {
        #[cfg(feature = "fmt-legacy")]
        Format::Q8_0 => dot_q8_0(w, a),
        #[cfg(feature = "fmt-legacy")]
        Format::Q4_0 => dot_q4_0(w, a),
        #[cfg(feature = "fmt-legacy")]
        Format::Q5_0 => dot_q5_0(w, a),
        // The k-quants run on the per-256 tiles at width one: integer
        // accumulation across the super-block, one float scale per 256.
        #[cfg(feature = "fmt-kquant")]
        Format::Q2_K => {
            let mut o = [0.0f32];
            tile_q2_k::<1>(w, std::slice::from_ref(a), &mut o);
            o[0]
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q3_K => {
            let mut o = [0.0f32];
            tile_q3_k::<1>(w, std::slice::from_ref(a), &mut o);
            o[0]
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q4_K => {
            let mut o = [0.0f32];
            tile_q4_k::<1>(w, std::slice::from_ref(a), &mut o);
            o[0]
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q5_K => {
            let mut o = [0.0f32];
            tile_q5_k::<1>(w, std::slice::from_ref(a), &mut o);
            o[0]
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q6_K => {
            let mut o = [0.0f32];
            tile_q6_k::<1>(w, std::slice::from_ref(a), &mut o);
            o[0]
        }
        #[cfg(feature = "fmt-iq")]
        Format::IQ4_NL => dot_iq4_nl(w, a),
        #[cfg(feature = "fmt-iq")]
        Format::IQ4_XS => dot_iq4_xs(w, a),
        #[cfg(feature = "fmt-mxfp4")]
        Format::MXFP4 => dot_mxfp4(w, a),
        Format::F32 => dot_f32(w, a),
        Format::F16 => dot_f16(w, a),
        Format::BF16 => dot_bf16(w, a),
        #[cfg(feature = "fmt-ternary")]
        Format::PQ2_0 | Format::PTQ1_0 => {
            let mut o = [0.0f32];
            dot_batch(fmt, w, std::slice::from_ref(a), &mut o);
            o[0]
        }
        #[allow(unreachable_patterns)]
        _ => panic!("format {} disabled in this build", fmt.label()),
    }
}

// ---------------------------------------------------------------------------
// Batched kernels: one weight row, many activations.
//
// The kernels above are one row against one activation, and they end every 32-weight
// block with a horizontal sum — about eight shuffles and adds to turn a vector of
// partial products into a scalar, per block, per row. Two things follow from that
// shape and both were measured before this file changed:
//
//   * decode spent more instructions reducing than multiplying, and
//   * prefill, which hands `matmul` a batch of up to 64 activations, called the
//     one-activation kernel 64 times per row — so the weight row was *fetched* once
//     (the byte accounting is right) but *decoded* 64 times. A 42-token prompt took
//     1.08 s, about the same per token as a decode step, when reading the weights
//     once was supposed to be what made prefill cheap.
//
// These kernels fix both. The weight block is decoded once into a register, then
// reused against up to `TILE` activations; partial products stay in vector
// accumulators (`cvt` + `fmadd` per block instead of a horizontal sum) and are
// reduced once per row. The per-format bias terms (the offset of Q5_0/Q6_K, the min
// of Q4_K) accumulate separately in scalars and are subtracted at the end, which is
// the same bias identity as `quant.rs`, evaluated in float instead of per block in
// integers — algebraically identical, differing only in rounding.
//
// `dot` above stays: it is the fallback on a CPU with AVX2 but no FMA, and the
// second implementation `tests/simd.rs` checks these against.
// ---------------------------------------------------------------------------

/// Activations processed per pass over a weight row. Four keeps every accumulator
/// and the decoded block in the sixteen vector registers; wider tiles spill.
pub const TILE: usize = 4;

/// Whether the batched kernels can run. They use fused multiply-add and the F16C
/// scale conversion (every AVX2 CPU since Haswell has both).
#[inline]
pub fn have_batched() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::cpu_has!("avx2") && crate::cpu_has!("fma") && crate::cpu_has!("f16c"))
}

/// The sum of the eight lanes of `v`.
///
/// # Safety
/// AVX must be available (it is wherever an `__m256` exists in a caller compiled for AVX2).
#[inline(always)]
pub unsafe fn hsum_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0b01));
    _mm_cvtss_f32(s)
}

/// `Σ codes·acts − offset·Σacts` as eight exact i32 lane sums, converted to f32.
/// The lane layout matches `QuantAct::lane_sums`.
#[inline(always)]
unsafe fn madd_offset_ps<const SHIFT: i32>(
    codes: __m256i,
    acts: __m256i,
    lane_sums: *const i32,
) -> __m256 {
    let prod = _mm256_maddubs_epi16(codes, acts);
    let s = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
    let ls = _mm256_loadu_si256(lane_sums as *const __m256i);
    // The offset is a power of two (16 for Q5_0): a shift, not `VPMULLD` — which is
    // two micro-ops with ten cycles of latency, once per block per activation.
    _mm256_cvtepi32_ps(_mm256_sub_epi32(s, _mm256_slli_epi32(ls, SHIFT)))
}

/// `Σ codes·acts` as eight f32 partial sums, with no horizontal reduction.
#[inline(always)]
unsafe fn madd_ps(codes: __m256i, acts: __m256i) -> __m256 {
    let prod = _mm256_maddubs_epi16(codes, acts);
    _mm256_cvtepi32_ps(_mm256_madd_epi16(prod, _mm256_set1_epi16(1)))
}

#[cfg(test)]
#[target_feature(enable = "f16c")]
unsafe fn f16c_one(h: u16) -> f32 {
    f16_at(&h.to_le_bytes(), 0)
}

/// A block's f16 scale, converted by the F16C unit (`VCVTPH2PS`): two instructions
/// in place of `gguf::f16`'s branchy bit surgery, once per 32 weights in Q8_0/Q5_0.
/// The conversion is exact either way (every f16 is an f32), so the result is the
/// same bits — `f16c_matches_the_reference_conversion` checks every one of the 65,536.
#[inline(always)]
unsafe fn f16_at(b: &[u8], at: usize) -> f32 {
    debug_assert!(at + 2 <= b.len());
    let h = u16::from_le_bytes([*b.get_unchecked(at), *b.get_unchecked(at + 1)]);
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)))
}

/// `f16_at` for a scale with two more bytes of the same block after it: one 32-bit
/// load (`vmovd`) straight into the vector unit instead of two byte loads, a merge
/// and `vpinsrw` (a port-5 µop). Lane 0 is the same conversion of the same bits.
#[inline(always)]
unsafe fn f16_at4(p: *const u8) -> f32 {
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
        (p as *const i32).read_unaligned(),
    )))
}

/// How far ahead of the block being decoded the kernels ask for weight bytes.
///
/// Decode streams every weight once per token, and the L2 streamer that would fetch
/// ahead stops at each 4 KB page: without a hint, every page of weights costs a full
/// DRAM round trip that the core spends waiting (the hot instructions in a
/// profile were the loads, and one core streamed 7.4 GB/s against llama.cpp's 10.7).
/// `PEREGRINE_PF` (bytes, 0 = off) is the A/B switch it was tuned with.
fn prefetch_distance() -> usize {
    static PF: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PF.get_or_init(|| crate::options::get().prefetch_bytes)
}
// The default distance, 1024 bytes, is `options::PF_DEFAULT`.

/// The lookahead the eight-row kernels use.
///
/// The row kernels read one stream, and `PEREGRINE_PF`'s default (1 KB) was tuned on
/// that shape. Eight rows are eight streams a page apart, and at 1 KB each stream is
/// only hinted within the current page: the hardware streamer stops at a page boundary,
/// so every page still costs a DRAM round trip. 4 KB — one page — keeps all eight fed.
/// Measured on the 4-core Xeon (Qwen2.5-0.5B Q4_K_M, 512 context): decode 36.4 → 38.5
/// tok/s, prefill 124.0 → 126.9. `PEREGRINE_PF=0` still turns prefetching off.
pub(crate) fn rows8_prefetch() -> usize {
    match prefetch_distance() {
        0 => 0,
        pf => pf.max(4096),
    }
}

/// The lookahead the multi-row PQ2_0 decode kernel uses: two streams, so a page-size
/// hint (`PEREGRINE_PF`'s 1 KB default is tuned for the single-stream row kernels and
/// is too short here). `PEREGRINE_PF=0` still turns prefetching off.
pub(crate) fn pq2_rows_prefetch() -> usize {
    match prefetch_distance() {
        0 => 0,
        pf => pf.max(2048),
    }
}

/// Hint `lines` cache lines starting `pf` bytes past `p`. A prefetch never faults, so
/// running past the end of the tensor is harmless; `wrapping_add` keeps the pointer
/// arithmetic defined there.
#[inline(always)]
pub(crate) unsafe fn prefetch_ahead(p: *const u8, pf: usize, lines: usize) {
    if pf == 0 {
        return;
    }
    for l in 0..lines {
        _mm_prefetch(p.wrapping_add(pf + l * 64) as *const i8, _MM_HINT_T0);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q8_0<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let mut acc = [_mm256_setzero_ps(); T];
    for (b, blk) in w.chunks_exact(34).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 1);
        let wq = _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i);
        let absw = _mm256_sign_epi8(wq, wq);
        let dw = f16_at(blk, 0);
        for t in 0..T {
            let a = &acts[t];
            let sa = _mm256_sign_epi8(acts32(a, b), wq);
            acc[t] = _mm256_fmadd_ps(madd_ps(absw, sa), _mm256_set1_ps(dw * a.d[b]), acc[t]);
        }
    }
    for t in 0..T {
        out[t] = hsum_ps(acc[t]);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q4_0<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let lownib = _mm256_set1_epi8(0x0F);
    // Low lane keeps the bytes, high lane shifts each 64-bit half right by 4: after
    // the mask that is `qs & 0xF` in lanes 0..16 and `(qs >> 4) & 0xF` in 16..32 —
    // the `_mm256_set_m128i(hi, lo)` pair, from one broadcast load and no port-5
    // lane insert.
    let nib_shift = _mm256_set_epi64x(4, 4, 0, 0);
    let nblk = w.len() / 18;
    for a in &acts[..T] {
        assert!(a.d.len() >= nblk && a.q.len() >= nblk * 32 && a.lane_sums.len() >= nblk * 8);
    }
    let mut acc = [_mm256_setzero_ps(); T];
    for (b, blk) in w.chunks_exact(18).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 1);
        let qs =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i));
        let codes = _mm256_and_si256(_mm256_srlv_epi64(qs, nib_shift), lownib);
        let dw = f16_at4(blk.as_ptr());
        for t in 0..T {
            let a = &acts[t];
            // Codes are [0,15] with offset 8, removed exactly inside the lanes.
            let v = madd_offset_ps::<3>(codes, acts32(a, b), a.lane_sums.as_ptr().add(b * 8));
            acc[t] = _mm256_fmadd_ps(v, _mm256_set1_ps(dw * *a.d.get_unchecked(b)), acc[t]);
        }
    }
    for t in 0..T {
        out[t] = hsum_ps(acc[t]);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q5_0<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let lownib = _mm256_set1_epi8(0x0F);
    // Low lane keeps the bytes, high lane shifts each 64-bit half right by 4: after
    // the mask that is `qs & 0xF` in lanes 0..16 and `(qs >> 4) & 0xF` in 16..32 —
    // the `_mm256_set_m128i(hi, lo)` pair, from one broadcast load and no port-5
    // lane insert.
    let nib_shift = _mm256_set_epi64x(4, 4, 0, 0);
    let nblk = w.len() / 22;
    for a in &acts[..T] {
        assert!(a.d.len() >= nblk && a.q.len() >= nblk * 32 && a.lane_sums.len() >= nblk * 8);
    }
    let mut acc = [_mm256_setzero_ps(); T];
    for (b, blk) in w.chunks_exact(22).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 1);
        let qh = (blk.as_ptr().add(2) as *const u32).read_unaligned();
        let qs =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(blk.as_ptr().add(6) as *const __m128i));
        let nib = _mm256_and_si256(_mm256_srlv_epi64(qs, nib_shift), lownib);
        let codes = _mm256_or_si256(nib, q5_high_bits(u32::from_le(qh)));
        let dw = f16_at4(blk.as_ptr());
        for t in 0..T {
            let a = &acts[t];
            // Codes are [0,31] with offset 16, removed exactly inside the lanes.
            let v = madd_offset_ps::<4>(codes, acts32(a, b), a.lane_sums.as_ptr().add(b * 8));
            acc[t] = _mm256_fmadd_ps(v, _mm256_set1_ps(dw * *a.d.get_unchecked(b)), acc[t]);
        }
    }
    for t in 0..T {
        out[t] = hsum_ps(acc[t]);
    }
}

/// Eight consecutive Q8_0 rows against one activation: the decode GEMV.
///
/// Q8_0 rows are 34 bytes per 32 weights and the activation block is 32, so the
/// activation load is a third of the traffic; loading it once for eight rows leaves
/// only the per-row sign moves and multiply-adds. Each row runs the same operations
/// `tile_q8_0::<1>` runs, block by block, so its bits do not depend on the group.
///
/// `w` holds eight consecutive rows of `rb` bytes (34 per 32 weights); `out[r]` is row
/// `r`'s result.
///
/// # Safety
/// The caller must have checked AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q8_0_rows8(w: &[u8], rb: usize, a: &QuantAct, out: &mut [f32]) {
    const ROWS: usize = 8;
    assert_eq!(rb % 34, 0, "a Q8_0 row is a whole number of 34-byte blocks");
    assert!(w.len() >= ROWS * rb, "the weight bytes hold eight rows");
    assert!(out.len() >= ROWS, "the output holds eight rows");
    let nblk = rb / 34;
    assert!(
        a.d.len() >= nblk && a.q.len() >= nblk * 32,
        "the activation is narrower than a row"
    );
    let pf = rows8_prefetch();
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); ROWS];
    for b in 0..nblk {
        let act = _mm256_loadu_si256(a.q.as_ptr().add(b * 32) as *const __m256i);
        let d = *a.d.get_unchecked(b);
        for r in 0..ROWS {
            let blk = w.as_ptr().add(r * rb + b * 34);
            prefetch_ahead(blk, pf, 1);
            let wq = _mm256_loadu_si256(blk.add(2) as *const __m256i);
            let absw = _mm256_sign_epi8(wq, wq);
            let sa = _mm256_sign_epi8(act, wq);
            let dw = f16_at4(blk);
            let q = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(absw, sa), ones));
            acc[r] = _mm256_fmadd_ps(q, _mm256_set1_ps(dw * d), acc[r]);
        }
    }
    for r in 0..ROWS {
        out[r] = hsum_ps(acc[r]);
    }
}

/// Eight consecutive Q5_0 rows against one activation: the decode GEMV.
///
/// A Q5_0 row is only 22 bytes per 32 weights, so the activation block (32 bytes) and
/// its lane sums (32 bytes) are more traffic than the weights themselves, and the
/// per-row kernel reloads both once per row. Here they are loaded once per block for
/// eight rows; every row still runs exactly the operations `tile_q5_0::<1>` runs, block
/// by block, so a row's bits do not depend on how many rows shared the pass.
///
/// `w` holds eight consecutive rows of `rb` bytes (22 per 32 weights); `out[r]` is row
/// `r`'s result.
///
/// # Safety
/// The caller must have checked AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q5_0_rows8(w: &[u8], rb: usize, a: &QuantAct, out: &mut [f32]) {
    const ROWS: usize = 8;
    assert_eq!(rb % 22, 0, "a Q5_0 row is a whole number of 22-byte blocks");
    assert!(w.len() >= ROWS * rb, "the weight bytes hold eight rows");
    assert!(out.len() >= ROWS, "the output holds eight rows");
    let nblk = rb / 22;
    assert!(
        a.d.len() >= nblk && a.q.len() >= nblk * 32 && a.lane_sums.len() >= nblk * 8,
        "the activation is narrower than a row"
    );
    let pf = rows8_prefetch();
    let lownib = _mm256_set1_epi8(0x0F);
    let nib_shift = _mm256_set_epi64x(4, 4, 0, 0);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); ROWS];
    for b in 0..nblk {
        let act = _mm256_loadu_si256(a.q.as_ptr().add(b * 32) as *const __m256i);
        let lanes = _mm256_loadu_si256(a.lane_sums.as_ptr().add(b * 8) as *const __m256i);
        let d = *a.d.get_unchecked(b);
        for r in 0..ROWS {
            let blk = w.as_ptr().add(r * rb + b * 22);
            prefetch_ahead(blk, pf, 1);
            let qh = (blk.add(2) as *const u32).read_unaligned();
            let qs = _mm256_broadcastsi128_si256(_mm_loadu_si128(blk.add(6) as *const __m128i));
            let nib = _mm256_and_si256(_mm256_srlv_epi64(qs, nib_shift), lownib);
            let codes = _mm256_or_si256(nib, q5_high_bits(u32::from_le(qh)));
            let dw = f16_at4(blk);
            let s = _mm256_madd_epi16(_mm256_maddubs_epi16(codes, act), ones);
            // The same `madd_offset_ps::<4>` the per-row kernel runs.
            let v = _mm256_cvtepi32_ps(_mm256_sub_epi32(s, _mm256_slli_epi32(lanes, 4)));
            acc[r] = _mm256_fmadd_ps(v, _mm256_set1_ps(dw * d), acc[r]);
        }
    }
    for r in 0..ROWS {
        out[r] = hsum_ps(acc[r]);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q4_k<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    debug_assert!(
        acts.iter().all(|a| !a.qk.is_empty()),
        "k-quant rows need the per-256 activation"
    );
    let lownib = _mm256_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); T];
    for (sb, blk) in w.chunks_exact(144).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 3);
        let dv = _mm256_set1_ps(f16_at(blk, 0));
        let dminv = _mm256_set1_ps(f16_at(blk, 2));
        let qs = blk.as_ptr().add(16);
        let sm = unpack_q4k_scales(&blk[4..16]);
        let scv = q4k_scale_vecs(&sm);
        let mins16 = q4k_mins16(&sm[8..16]);
        let mut sumi = [_mm256_setzero_si256(); T];
        // Each 32 bytes of codes hold two sub-blocks: the low nibbles are sub-block
        // 2j, the high nibbles 2j+1. Both dot products are scaled by their 6-bit
        // scale in 16-bit lanes and summed in 32-bit lanes; nothing leaves the
        // integer domain until the super-block is done.
        for j in 0..4 {
            let raw = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
            let q4l = _mm256_and_si256(raw, lownib);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(raw, 4), lownib);
            let (scl, sch) = (scv[2 * j], scv[2 * j + 1]);
            for t in 0..T {
                let base = acts[t].qk.as_ptr().add(sb * 256 + j * 64);
                let q8l = _mm256_loadu_si256(base as *const __m256i);
                let q8h = _mm256_loadu_si256(base.add(32) as *const __m256i);
                let pl = _mm256_madd_epi16(_mm256_maddubs_epi16(q4l, q8l), scl);
                let ph = _mm256_madd_epi16(_mm256_maddubs_epi16(q4h, q8h), sch);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(pl, ph));
            }
        }
        for t in 0..T {
            // w = d·sc·q − dmin·m: the min term is the run sums against the mins.
            let a = &acts[t];
            acc[t] = q4k_fold(
                acc[t],
                sumi[t],
                mins16,
                a.bsums.as_ptr().add(sb * 16),
                dv,
                dminv,
                a.dk[sb],
            );
        }
    }
    for t in 0..T {
        out[t] = hsum_ps(acc[t]);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q5_k<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    debug_assert!(
        acts.iter().all(|a| !a.qk.is_empty()),
        "k-quant rows need the per-256 activation"
    );
    let lownib = _mm256_set1_epi8(0x0F);
    let bit16 = _mm256_set1_epi8(0x10);
    let mut acc = [_mm256_setzero_ps(); T];
    for (sb, blk) in w.chunks_exact(176).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 3);
        let dv = _mm256_set1_ps(f16_at(blk, 0));
        let dminv = _mm256_set1_ps(f16_at(blk, 2));
        let sm = unpack_q4k_scales(&blk[4..16]);
        let scv = q4k_scale_vecs(&sm);
        let mins16 = q4k_mins16(&sm[8..16]);
        let qh = _mm256_loadu_si256(blk.as_ptr().add(16) as *const __m256i);
        let qs = blk.as_ptr().add(48);
        let mut sumi = [_mm256_setzero_si256(); T];
        // Each 32 bytes of codes hold two sub-blocks: the low nibbles are sub-block
        // 2j, the high nibbles 2j+1. The fifth bit for sub-block is comes from bit is
        // of qh. Both dot products are scaled by their 6-bit scale in 16-bit lanes and
        // summed in 32-bit lanes; nothing leaves the integer domain until the super-block is done.
        for j in 0..4 {
            let raw = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
            let q4l = _mm256_and_si256(raw, lownib);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(raw, 4), lownib);

            let mask_l = _mm256_set1_epi8((1u8 << (2 * j)) as i8);
            let set_l = _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask_l), mask_l);
            let hi_l = _mm256_and_si256(set_l, bit16);
            let q5l = _mm256_or_si256(q4l, hi_l);

            let mask_h = _mm256_set1_epi8((1u8 << (2 * j + 1)) as i8);
            let set_h = _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask_h), mask_h);
            let hi_h = _mm256_and_si256(set_h, bit16);
            let q5h = _mm256_or_si256(q4h, hi_h);

            let (scl, sch) = (scv[2 * j], scv[2 * j + 1]);
            for t in 0..T {
                let base = acts[t].qk.as_ptr().add(sb * 256 + j * 64);
                let q8l = _mm256_loadu_si256(base as *const __m256i);
                let q8h = _mm256_loadu_si256(base.add(32) as *const __m256i);
                let pl = _mm256_madd_epi16(_mm256_maddubs_epi16(q5l, q8l), scl);
                let ph = _mm256_madd_epi16(_mm256_maddubs_epi16(q5h, q8h), sch);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(pl, ph));
            }
        }
        for t in 0..T {
            // w = d·sc·q − dmin·m: the min term is the run sums against the mins.
            let a = &acts[t];
            acc[t] = q4k_fold(
                acc[t],
                sumi[t],
                mins16,
                a.bsums.as_ptr().add(sb * 16),
                dv,
                dminv,
                a.dk[sb],
            );
        }
    }
    for t in 0..T {
        out[t] = hsum_ps(acc[t]);
    }
}

/// The eight run-scale vectors of a Q6_K super-block, index `c·4 + g`: 16-bit lanes
/// 0..8 hold scale `c·8 + 2g`, lanes 8..16 scale `c·8 + 2g + 1` — the values
/// `_mm256_set_m128i(_mm_set1_epi16(..), _mm_set1_epi16(..))` produced, built with
/// one byte shuffle each instead of five port-5 µops through the integer registers
/// (which made the Q6_K kernel port-5 bound). `sc16` is the 16 scales as i16 lanes.
#[inline(always)]
unsafe fn q6k_scales(sc16: __m256i) -> [__m256i; 8] {
    // Both 128-bit halves of `half[c]` hold scales c·8 .. c·8+8 (16 bytes).
    let half = [
        _mm256_permute4x64_epi64::<0x44>(sc16),
        _mm256_permute4x64_epi64::<0xEE>(sc16),
    ];
    let ctl = Q6K_SPLIT16.as_ptr() as *const __m256i;
    // spelled out for the same reason as `q4k_scale_vecs`.
    let t = |i: usize| _mm256_shuffle_epi8(half[i / 4], _mm256_loadu_si256(ctl.add(i % 4)));
    [t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7)]
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q6_k<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    debug_assert!(
        acts.iter().all(|a| !a.qk.is_empty()),
        "k-quant rows need the per-256 activation"
    );
    let lownib = _mm256_set1_epi8(0x0F);
    let two = _mm256_set1_epi8(0x03);
    let mut acc = [_mm256_setzero_ps(); T];
    for (sb, blk) in w.chunks_exact(210).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 4);
        let d = f16_at(blk, 208);
        // The 16 run scales as 16-bit lanes, for the offset term below.
        let sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(blk.as_ptr().add(192) as *const __m128i));
        let scales = q6k_scales(sc16);
        let mut sumi = [_mm256_setzero_si256(); T];
        for c in 0..2 {
            let ql = blk.as_ptr().add(c * 64);
            let qh = _mm256_loadu_si256(blk.as_ptr().add(128 + c * 32) as *const __m256i);
            let ql0 = _mm256_loadu_si256(ql as *const __m256i);
            let ql1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
            for g in 0..4 {
                let (base, hbits) = match g {
                    0 => (ql0, _mm256_and_si256(qh, two)),
                    1 => (ql1, _mm256_and_si256(_mm256_srli_epi16(qh, 2), two)),
                    2 => (ql0, _mm256_and_si256(_mm256_srli_epi16(qh, 4), two)),
                    _ => (ql1, _mm256_and_si256(_mm256_srli_epi16(qh, 6), two)),
                };
                let low = if g < 2 {
                    _mm256_and_si256(base, lownib)
                } else {
                    _mm256_and_si256(_mm256_srli_epi16(base, 4), lownib)
                };
                let codes = _mm256_or_si256(low, _mm256_slli_epi16(hbits, 4));
                // Lanes 0..8 of the 16-bit products cover the first 16 weights,
                // lanes 8..16 the next 16; each run has its own scale.
                let scale = scales[c * 4 + g];
                let o = sb * 256 + c * 128 + g * 32;
                for t in 0..T {
                    let q8 = _mm256_loadu_si256(acts[t].qk.as_ptr().add(o) as *const __m256i);
                    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(codes, q8), scale);
                    sumi[t] = _mm256_add_epi32(sumi[t], p);
                }
            }
        }
        for t in 0..T {
            let a = &acts[t];
            // Offset 32 on every code: subtract 32·Σ(scale_run · Σqk_run), exact in integers.
            let bs = _mm256_loadu_si256(a.bsums.as_ptr().add(sb * 16) as *const __m256i);
            let off = _mm256_slli_epi32(_mm256_madd_epi16(sc16, bs), 5);
            let v = _mm256_sub_epi32(sumi[t], off);
            acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(d * a.dk[sb]), acc[t]);
        }
    }
    for t in 0..T {
        out[t] = hsum_ps(acc[t]);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_iq4_nl<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let lownib = _mm_set1_epi8(0x0F);
    let table = _mm_loadu_si128(crate::quant::KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let nblk = w.len() / 18;
    for a in &acts[..T] {
        assert!(a.d.len() >= nblk && a.q.len() >= nblk * 32);
    }
    let mut acc = [0.0f32; T];
    for (b, blk) in w.chunks_exact(18).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 1);
        let qs = _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i);
        let lo = _mm_and_si128(qs, lownib);
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), lownib);
        let w_lo = _mm_shuffle_epi8(table, lo);
        let w_hi = _mm_shuffle_epi8(table, hi);
        let w32 = _mm256_set_m128i(w_hi, w_lo);
        let dw = f16_at(blk, 0);
        for t in 0..T {
            let a = &acts[t];
            let aq = acts32(a, b);
            let prod = mul_add_epi8(w32, aq);
            let s = hsum_i32(_mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
            acc[t] += dw * *a.d.get_unchecked(b) * s as f32;
        }
    }
    out[..T].copy_from_slice(&acc[..T]);
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_iq4_xs<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    debug_assert!(
        acts[..T].iter().all(|a| !a.qk.is_empty()),
        "k-quant rows need the per-256 activation"
    );
    let lownib = _mm_set1_epi8(0x0F);
    let table = _mm_loadu_si128(crate::quant::KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let mut acc = [0.0f32; T];
    for (sb, blk) in w.chunks_exact(136).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 3);
        let d = f16_at(blk, 0);
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = blk.as_ptr().add(8);

        let mut w32 = [_mm256_setzero_si256(); 8];
        let mut scv = [_mm256_setzero_si256(); 8];
        for ib in 0..8 {
            let q4bits = _mm_loadu_si128(qs.add(ib * 16) as *const __m128i);
            let lo = _mm_and_si128(q4bits, lownib);
            let hi = _mm_and_si128(_mm_srli_epi16(q4bits, 4), lownib);
            let w_lo = _mm_shuffle_epi8(table, lo);
            let w_hi = _mm_shuffle_epi8(table, hi);
            w32[ib] = _mm256_set_m128i(w_hi, w_lo);
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as i16
                | ((((scales_h >> (2 * ib)) & 3) << 4) as i16);
            scv[ib] = _mm256_set1_epi16(ls - 32);
        }

        for t in 0..T {
            let mut sumi = _mm256_setzero_si256();
            let base = acts[t].qk.as_ptr().add(sb * 256);
            for ib in 0..8 {
                let aq = _mm256_loadu_si256(base.add(ib * 32) as *const __m256i);
                let p16 = mul_add_epi8(w32[ib], aq);
                let p32 = _mm256_madd_epi16(p16, scv[ib]);
                sumi = _mm256_add_epi32(sumi, p32);
            }
            let sumi_scalar = hsum_i32(sumi);
            acc[t] += acts[t].dk[sb] * (d * sumi_scalar as f32);
        }
    }
    out[..T].copy_from_slice(&acc[..T]);
}

/// # Safety
/// Caller must ensure AVX2, FMA, and F16C are available. Length preconditions:
/// `w` must have at least `17 * nblk` bytes and every activation in `acts[..T]`
/// must hold at least `nblk * 32` values.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_mxfp4<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let lownib = _mm_set1_epi8(0x0F);
    let table = _mm_loadu_si128(crate::quant::KVALUES_MXFP4.as_ptr() as *const __m128i);
    let nblk = w.len() / 17;
    for a in &acts[..T] {
        assert!(a.d.len() >= nblk && a.q.len() >= nblk * 32);
    }
    let mut acc = [0.0f32; T];
    for (b, blk) in w.chunks_exact(17).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 1);
        let qs = _mm_loadu_si128(blk.as_ptr().add(1) as *const __m128i);
        let lo = _mm_and_si128(qs, lownib);
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), lownib);
        let w_lo = _mm_shuffle_epi8(table, lo);
        let w_hi = _mm_shuffle_epi8(table, hi);
        let w32 = _mm256_set_m128i(w_hi, w_lo);
        let dw = crate::format::e8m0_to_f32_half(blk[0]);
        for t in 0..T {
            let a = &acts[t];
            let aq = acts32(a, b);
            let prod = mul_add_epi8(w32, aq);
            let s = hsum_i32(_mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
            acc[t] += dw * *a.d.get_unchecked(b) * s as f32;
        }
    }
    out[..T].copy_from_slice(&acc[..T]);
}

/// F32 weight dot product against `T` activations in f32 (`QuantAct::xf`).
///
/// # Safety
/// AVX2 must be available. `w` must contain at least `acts[0].n * 4` bytes.
/// Each `acts[t].xf` must contain at least `acts[0].n` floats. `out` must have at least `T` elements.
#[target_feature(enable = "avx2")]
unsafe fn tile_f32<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    assert!(
        acts[..T].iter().all(|a| a.keep_f32),
        "float matrix matmul reached with keep_f32 off on QuantAct"
    );
    let pf = prefetch_distance();
    let n = acts[0].n;
    for a in &acts[..T] {
        debug_assert_eq!(a.n, n);
        debug_assert!(a.xf.len() >= n);
    }
    debug_assert!(w.len() >= n * 4);
    debug_assert!(out.len() >= T);

    let n8 = (n / 8) * 8;
    let mut acc = [_mm256_setzero_ps(); T];
    let w_ptr = w.as_ptr() as *const f32;
    let mut i = 0;
    while i < n8 {
        prefetch_ahead(w.as_ptr().add(i * 4), pf, 1);
        let wv = _mm256_loadu_ps(w_ptr.add(i));
        for t in 0..T {
            let xv = _mm256_loadu_ps(acts[t].xf.as_ptr().add(i));
            // IEEE-754 multiply followed by add without FMA contraction, matching scalar dot_f32_ref.
            acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(wv, xv));
        }
        i += 8;
    }
    let rem = n - n8;
    for t in 0..T {
        if rem == 0 {
            out[t] = hsum_ps(acc[t]);
        } else {
            let mut acc_arr = [0.0f32; 8];
            _mm256_storeu_ps(acc_arr.as_mut_ptr(), acc[t]);
            for j in 0..rem {
                let idx = n8 + j;
                let wv = f32::from_le_bytes(w[idx * 4..idx * 4 + 4].try_into().unwrap());
                let xv = *acts[t].xf.get_unchecked(idx);
                acc_arr[j] += wv * xv;
            }
            out[t] = crate::quant::hsum_8(acc_arr);
        }
    }
}

/// F16 weight dot product against `T` activations in f32 (`QuantAct::xf`), using F16C conversion.
///
/// # Safety
/// AVX2 and F16C must be available. `w` must contain at least `acts[0].n * 2` bytes.
/// Each `acts[t].xf` must contain at least `acts[0].n` floats. `out` must have at least `T` elements.
#[target_feature(enable = "avx2,f16c")]
unsafe fn tile_f16<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    assert!(
        acts[..T].iter().all(|a| a.keep_f32),
        "float matrix matmul reached with keep_f32 off on QuantAct"
    );
    let pf = prefetch_distance();
    let n = acts[0].n;
    for a in &acts[..T] {
        debug_assert_eq!(a.n, n);
        debug_assert!(a.xf.len() >= n);
    }
    debug_assert!(w.len() >= n * 2);
    debug_assert!(out.len() >= T);

    let n8 = (n / 8) * 8;
    let mut acc = [_mm256_setzero_ps(); T];
    let w_ptr = w.as_ptr() as *const u16;
    let mut i = 0;
    while i < n8 {
        prefetch_ahead(w.as_ptr().add(i * 2), pf, 1);
        let wh = _mm_loadu_si128(w_ptr.add(i) as *const __m128i);
        let wv = _mm256_cvtph_ps(wh);
        for t in 0..T {
            let xv = _mm256_loadu_ps(acts[t].xf.as_ptr().add(i));
            acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(wv, xv));
        }
        i += 8;
    }
    let rem = n - n8;
    for t in 0..T {
        if rem == 0 {
            out[t] = hsum_ps(acc[t]);
        } else {
            let mut acc_arr = [0.0f32; 8];
            _mm256_storeu_ps(acc_arr.as_mut_ptr(), acc[t]);
            for j in 0..rem {
                let idx = n8 + j;
                let h = u16::from_le_bytes([w[idx * 2], w[idx * 2 + 1]]);
                let wv = f16(h);
                let xv = *acts[t].xf.get_unchecked(idx);
                acc_arr[j] += wv * xv;
            }
            out[t] = crate::quant::hsum_8(acc_arr);
        }
    }
}

/// BF16 weight dot product against `T` activations in f32 (`QuantAct::xf`), using shift-left-16 conversion.
///
/// # Safety
/// AVX2 must be available. `w` must contain at least `acts[0].n * 2` bytes.
/// Each `acts[t].xf` must contain at least `acts[0].n` floats. `out` must have at least `T` elements.
#[target_feature(enable = "avx2")]
unsafe fn tile_bf16<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    assert!(
        acts[..T].iter().all(|a| a.keep_f32),
        "float matrix matmul reached with keep_f32 off on QuantAct"
    );
    let pf = prefetch_distance();
    let n = acts[0].n;
    for a in &acts[..T] {
        debug_assert_eq!(a.n, n);
        debug_assert!(a.xf.len() >= n);
    }
    debug_assert!(w.len() >= n * 2);
    debug_assert!(out.len() >= T);

    let n8 = (n / 8) * 8;
    let mut acc = [_mm256_setzero_ps(); T];
    let w_ptr = w.as_ptr() as *const u16;
    let mut i = 0;
    while i < n8 {
        prefetch_ahead(w.as_ptr().add(i * 2), pf, 1);
        let wb = _mm_loadu_si128(w_ptr.add(i) as *const __m128i);
        let raw32 = _mm256_cvtepu16_epi32(wb);
        let wv = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(raw32));
        for t in 0..T {
            let xv = _mm256_loadu_ps(acts[t].xf.as_ptr().add(i));
            acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(wv, xv));
        }
        i += 8;
    }
    let rem = n - n8;
    for t in 0..T {
        if rem == 0 {
            out[t] = hsum_ps(acc[t]);
        } else {
            let mut acc_arr = [0.0f32; 8];
            _mm256_storeu_ps(acc_arr.as_mut_ptr(), acc[t]);
            for j in 0..rem {
                let idx = n8 + j;
                let bits = (u16::from_le_bytes([w[idx * 2], w[idx * 2 + 1]]) as u32) << 16;
                let wv = f32::from_bits(bits);
                let xv = *acts[t].xf.get_unchecked(idx);
                acc_arr[j] += wv * xv;
            }
            out[t] = crate::quant::hsum_8(acc_arr);
        }
    }
}

/// Single-row F32 dot product.
///
/// # Safety
/// AVX2 must be available. `w` must contain `a.n * 4` bytes and `a.xf` must contain `a.n` floats.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_f32(w: &[u8], a: &QuantAct) -> f32 {
    let mut o = [0.0f32];
    // SAFETY: caller guaranteed AVX2 and length preconditions.
    tile_f32::<1>(w, std::slice::from_ref(a), &mut o);
    o[0]
}

/// Single-row F16 dot product with F16C conversion.
///
/// # Safety
/// AVX2 and F16C must be available. `w` must contain `a.n * 2` bytes and `a.xf` must contain `a.n` floats.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_f16(w: &[u8], a: &QuantAct) -> f32 {
    let mut o = [0.0f32];
    // SAFETY: caller guaranteed AVX2, F16C and length preconditions.
    tile_f16::<1>(w, std::slice::from_ref(a), &mut o);
    o[0]
}

/// Single-row BF16 dot product with shift-left-16 conversion.
///
/// # Safety
/// AVX2 must be available. `w` must contain `a.n * 2` bytes and `a.xf` must contain `a.n` floats.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_bf16(w: &[u8], a: &QuantAct) -> f32 {
    let mut o = [0.0f32];
    // SAFETY: caller guaranteed AVX2 and length preconditions.
    tile_bf16::<1>(w, std::slice::from_ref(a), &mut o);
    o[0]
}

const Q3K_SHUF: [[u8; 32]; 4] = [
    [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3,
    ],
    [
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7,
    ],
    [
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11,
    ],
    [
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14,
        15, 14, 15, 14, 15, 14, 15, 14, 15,
    ],
];

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q2_k<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    debug_assert!(
        acts.iter().all(|a| !a.qk.is_empty()),
        "k-quant rows need the per-256 activation"
    );
    // SAFETY: caller guarantees AVX2 and FMA are supported.
    let m3 = _mm256_set1_epi8(3);
    let m4 = _mm_set1_epi8(0x0F);
    // SAFETY: Q3K_SHUF has 32 bytes per entry; reading 256-bit vector is in-bounds.
    let shuf_masks = [
        _mm256_loadu_si256(Q3K_SHUF[0].as_ptr() as *const __m256i),
        _mm256_loadu_si256(Q3K_SHUF[1].as_ptr() as *const __m256i),
        _mm256_loadu_si256(Q3K_SHUF[2].as_ptr() as *const __m256i),
        _mm256_loadu_si256(Q3K_SHUF[3].as_ptr() as *const __m256i),
    ];
    let mut acc = [0.0f32; T];
    for (sb, blk) in w.chunks_exact(84).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 2);
        let d = f16_at(blk, 80);
        let dmin = f16_at(blk, 82);

        // SAFETY: blk is 84 bytes; scales are at 0..16.
        let mins_and_scales = _mm_loadu_si128(blk.as_ptr() as *const __m128i);
        let scales8 = _mm_and_si128(mins_and_scales, m4);
        let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        let mins16 = _mm256_cvtepi8_epi16(mins8);

        let all_scales = _mm256_cvtepi8_epi16(scales8);
        let l_scales = _mm256_extracti128_si256(all_scales, 0);
        let h_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            _mm256_set_m128i(l_scales, l_scales),
            _mm256_set_m128i(h_scales, h_scales),
        ];

        let qs = blk.as_ptr().add(16);

        // Unpack 2-bit weight codes once for all activations in this tile.
        let mut q2_unpacked = [[_mm256_setzero_si256(); 4]; 2];
        for j in 0..2 {
            // SAFETY: qs has 64 bytes; j * 32 + 32 <= 64.
            let q2bits = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
            q2_unpacked[j][0] = _mm256_and_si256(q2bits, m3);
            q2_unpacked[j][1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            q2_unpacked[j][2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            q2_unpacked[j][3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);
        }

        for t in 0..T {
            let mut sumi = _mm256_setzero_si256();
            let base = acts[t].qk.as_ptr().add(sb * 256);
            for j in 0..2 {
                let sc = scales[j];
                for k in 0..4 {
                    // SAFETY: base has 256 elements; j * 128 + k * 32 + 32 <= 256.
                    let q8 = _mm256_loadu_si256(base.add(j * 128 + k * 32) as *const __m256i);
                    let p16 = _mm256_maddubs_epi16(q2_unpacked[j][k], q8);
                    let p32 = _mm256_madd_epi16(_mm256_shuffle_epi8(sc, shuf_masks[k]), p16);
                    sumi = _mm256_add_epi32(sumi, p32);
                }
            }
            let sumi_scalar = hsum_i32(sumi);
            let bsums_ptr = acts[t].bsums.as_ptr().add(sb * 16);
            // SAFETY: bsums has 16 elements per super-block.
            let prod = _mm256_madd_epi16(mins16, _mm256_loadu_si256(bsums_ptr as *const __m256i));
            let summ_scalar = hsum_i32(prod);
            acc[t] += acts[t].dk[sb] * (d * sumi_scalar as f32 - dmin * summ_scalar as f32);
        }
    }
    out[..T].copy_from_slice(&acc[..T]);
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_q3_k<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    debug_assert!(
        acts.iter().all(|a| !a.qk.is_empty()),
        "k-quant rows need the per-256 activation"
    );
    // SAFETY: caller guarantees AVX2 and FMA are supported.
    let m3 = _mm256_set1_epi8(3);
    let zero = _mm256_setzero_si256();
    let four = _mm256_set1_epi8(4);
    let m32 = _mm_set1_epi8(32);
    // SAFETY: Q3K_SHUF has 32 bytes per entry; reading 256-bit vector is in-bounds.
    let shuf_masks = [
        _mm256_loadu_si256(Q3K_SHUF[0].as_ptr() as *const __m256i),
        _mm256_loadu_si256(Q3K_SHUF[1].as_ptr() as *const __m256i),
        _mm256_loadu_si256(Q3K_SHUF[2].as_ptr() as *const __m256i),
        _mm256_loadu_si256(Q3K_SHUF[3].as_ptr() as *const __m256i),
    ];
    let mut acc = [0.0f32; T];
    for (sb, blk) in w.chunks_exact(110).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 3);
        let d = f16_at(blk, 108);

        // SAFETY: blk has 110 bytes; scales are at 96..108.
        let a0 = (blk.as_ptr().add(96) as *const u32).read_unaligned();
        let a1 = (blk.as_ptr().add(100) as *const u32).read_unaligned();
        let a2 = (blk.as_ptr().add(104) as *const u32).read_unaligned();
        let (a0, a1, a2) = (u32::from_le(a0), u32::from_le(a1), u32::from_le(a2));
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;
        let s0 = (a0 & KMASK2) | ((a2 & KMASK1) << 4);
        let s1 = (a1 & KMASK2) | (((a2 >> 2) & KMASK1) << 4);
        let s2 = ((a0 >> 4) & KMASK2) | (((a2 >> 4) & KMASK1) << 4);
        let s3 = ((a1 >> 4) & KMASK2) | (((a2 >> 6) & KMASK1) << 4);
        let scales128 = _mm_setr_epi32(s0 as i32, s1 as i32, s2 as i32, s3 as i32);
        let scales128 = _mm_sub_epi8(scales128, m32);
        let all_scales = _mm256_cvtepi8_epi16(scales128);
        let l_scales = _mm256_extracti128_si256(all_scales, 0);
        let h_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            _mm256_set_m128i(l_scales, l_scales),
            _mm256_set_m128i(h_scales, h_scales),
        ];

        // SAFETY: blk has 110 bytes; hmask is 32 bytes at offset 0.
        let hbits = _mm256_loadu_si256(blk.as_ptr() as *const __m256i);
        let qs = blk.as_ptr().add(32);

        // Precompute low 2 bits and high bit masks once for all activations in this tile.
        let mut q3l_all = [[_mm256_setzero_si256(); 4]; 2];
        let mut q3h_all = [[_mm256_setzero_si256(); 4]; 2];
        for j in 0..2 {
            // SAFETY: qs has 64 bytes; j * 32 + 32 <= 64.
            let q3bits = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
            q3l_all[j][0] = _mm256_and_si256(q3bits, m3);
            q3l_all[j][1] = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            q3l_all[j][2] = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            q3l_all[j][3] = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);

            for k in 0..4 {
                let bit = j * 4 + k;
                let mask = _mm256_set1_epi8((1u8 << bit) as i8);
                let not_set = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, mask), zero);
                q3h_all[j][k] = _mm256_and_si256(not_set, four);
            }
        }

        for t in 0..T {
            let mut sumi = _mm256_setzero_si256();
            let base = acts[t].qk.as_ptr().add(sb * 256);
            for j in 0..2 {
                let sc = scales[j];
                for k in 0..4 {
                    // SAFETY: base has 256 elements; j * 128 + k * 32 + 32 <= 256.
                    let q8 = _mm256_loadu_si256(base.add(j * 128 + k * 32) as *const __m256i);
                    let q8s = _mm256_maddubs_epi16(q3h_all[j][k], q8);
                    let p16 = _mm256_maddubs_epi16(q3l_all[j][k], q8);
                    let p16 = _mm256_sub_epi16(p16, q8s);
                    let p32 = _mm256_madd_epi16(_mm256_shuffle_epi8(sc, shuf_masks[k]), p16);
                    sumi = _mm256_add_epi32(sumi, p32);
                }
            }
            let sumi_scalar = hsum_i32(sumi);
            acc[t] += acts[t].dk[sb] * (d * sumi_scalar as f32);
        }
    }
    out[..T].copy_from_slice(&acc[..T]);
}

/// `PEREGRINE_NO_WIDE=1` keeps the tiled kernels for every batch width — the A/B
/// switch the wide kernels were measured against (kbench, and a real prefill).
#[inline]
fn wide_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::options::get().wide)
}

/// Widest batch the wide kernels take in one call; `model::MAX_BATCH` must not exceed it.
pub const MAX_WIDE: usize = 128;

/// The per-activation half of a Q4_K super-block: `sumi[u] += Σ_j (q4l_j·a_{2j})·sc_{2j} + (q4h_j·a_{2j+1})·sc_{2j+1}`
/// for `T` activations against codes already unpacked by the caller.
#[inline(always)]
unsafe fn q4k_acts<const T: usize>(
    q4l: &[__m256i; 4],
    q4h: &[__m256i; 4],
    sc: &[u8; 8],
    acts: &[QuantAct],
    sb: usize,
    sumi: &mut [__m256i; 4],
) {
    for j in 0..4 {
        let scl = _mm256_set1_epi16(sc[2 * j] as i16);
        let sch = _mm256_set1_epi16(sc[2 * j + 1] as i16);
        for t in 0..T {
            let base = acts[t].qk.as_ptr().add(sb * 256 + j * 64);
            let q8l = _mm256_loadu_si256(base as *const __m256i);
            let q8h = _mm256_loadu_si256(base.add(32) as *const __m256i);
            let pl = _mm256_madd_epi16(_mm256_maddubs_epi16(q4l[j], q8l), scl);
            let ph = _mm256_madd_epi16(_mm256_maddubs_epi16(q4h[j], q8h), sch);
            sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(pl, ph));
        }
    }
}

/// Q4_K against a *wide* batch (a prompt): the super-block's scales, mins and
/// nibbles are unpacked once and run against every activation, where the tiled
/// kernel unpacks them again for each tile of four. The per-activation work is
/// the same eight integer ops; what this removes is the repeated row decoding,
/// which at a 51-token prompt was done thirteen times per super-block. The
/// accumulators live in a stack array (`MAX_WIDE` vectors), touched once per
/// super-block, so register pressure is the tile's, not the batch's.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn wide_q4_k(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let n = acts.len();
    debug_assert!(n <= MAX_WIDE && acts.iter().all(|a| !a.qk.is_empty()));
    let lownib = _mm256_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); MAX_WIDE];
    for (sb, blk) in w.chunks_exact(144).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 3);
        let dv = _mm256_set1_ps(f16_at(blk, 0));
        let dminv = _mm256_set1_ps(f16_at(blk, 2));
        let qs = blk.as_ptr().add(16);
        let sm = unpack_q4k_scales(&blk[4..16]);
        let mut sc = [0u8; 8];
        sc.copy_from_slice(&sm[0..8]);
        let mins16 = q4k_mins16(&sm[8..16]);
        let mut q4l = [_mm256_setzero_si256(); 4];
        let mut q4h = [_mm256_setzero_si256(); 4];
        for j in 0..4 {
            let raw = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
            q4l[j] = _mm256_and_si256(raw, lownib);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(raw, 4), lownib);
        }
        let mut t = 0;
        while t < n {
            let tw = (n - t).min(TILE);
            let mut sumi = [_mm256_setzero_si256(); 4];
            match tw {
                4 => q4k_acts::<4>(&q4l, &q4h, &sc, &acts[t..], sb, &mut sumi),
                3 => q4k_acts::<3>(&q4l, &q4h, &sc, &acts[t..], sb, &mut sumi),
                2 => q4k_acts::<2>(&q4l, &q4h, &sc, &acts[t..], sb, &mut sumi),
                _ => q4k_acts::<1>(&q4l, &q4h, &sc, &acts[t..], sb, &mut sumi),
            }
            for u in 0..tw {
                let a = &acts[t + u];
                acc[t + u] = q4k_fold(
                    acc[t + u],
                    sumi[u],
                    mins16,
                    a.bsums.as_ptr().add(sb * 16),
                    dv,
                    dminv,
                    a.dk[sb],
                );
            }
            t += tw;
        }
    }
    for b in 0..n {
        out[b] = hsum_ps(acc[b]);
    }
}

/// Q5_K against a *wide* batch; see `wide_q4_k`.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn wide_q5_k(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let n = acts.len();
    debug_assert!(n <= MAX_WIDE && acts.iter().all(|a| !a.qk.is_empty()));
    let lownib = _mm256_set1_epi8(0x0F);
    let bit16 = _mm256_set1_epi8(0x10);
    let mut acc = [_mm256_setzero_ps(); MAX_WIDE];
    for (sb, blk) in w.chunks_exact(176).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 3);
        let dv = _mm256_set1_ps(f16_at(blk, 0));
        let dminv = _mm256_set1_ps(f16_at(blk, 2));
        let sm = unpack_q4k_scales(&blk[4..16]);
        let mut sc = [0u8; 8];
        sc.copy_from_slice(&sm[0..8]);
        let mins16 = q4k_mins16(&sm[8..16]);
        let qh = _mm256_loadu_si256(blk.as_ptr().add(16) as *const __m256i);
        let qs = blk.as_ptr().add(48);
        let mut q5l = [_mm256_setzero_si256(); 4];
        let mut q5h = [_mm256_setzero_si256(); 4];
        for j in 0..4 {
            let raw = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
            let q4l = _mm256_and_si256(raw, lownib);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(raw, 4), lownib);

            let mask_l = _mm256_set1_epi8((1u8 << (2 * j)) as i8);
            let set_l = _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask_l), mask_l);
            let hi_l = _mm256_and_si256(set_l, bit16);
            q5l[j] = _mm256_or_si256(q4l, hi_l);

            let mask_h = _mm256_set1_epi8((1u8 << (2 * j + 1)) as i8);
            let set_h = _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask_h), mask_h);
            let hi_h = _mm256_and_si256(set_h, bit16);
            q5h[j] = _mm256_or_si256(q4h, hi_h);
        }
        let mut t = 0;
        while t < n {
            let tw = (n - t).min(TILE);
            let mut sumi = [_mm256_setzero_si256(); 4];
            match tw {
                4 => q4k_acts::<4>(&q5l, &q5h, &sc, &acts[t..], sb, &mut sumi),
                3 => q4k_acts::<3>(&q5l, &q5h, &sc, &acts[t..], sb, &mut sumi),
                2 => q4k_acts::<2>(&q5l, &q5h, &sc, &acts[t..], sb, &mut sumi),
                _ => q4k_acts::<1>(&q5l, &q5h, &sc, &acts[t..], sb, &mut sumi),
            }
            for u in 0..tw {
                let a = &acts[t + u];
                acc[t + u] = q4k_fold(
                    acc[t + u],
                    sumi[u],
                    mins16,
                    a.bsums.as_ptr().add(sb * 16),
                    dv,
                    dminv,
                    a.dk[sb],
                );
            }
            t += tw;
        }
    }
    for b in 0..n {
        out[b] = hsum_ps(acc[b]);
    }
}

/// The per-activation half of a Q6_K super-block against its eight unpacked code
/// vectors (`c·4 + g`), each run-scaled.
#[inline(always)]
unsafe fn q6k_acts<const T: usize>(
    codes: &[__m256i; 8],
    scales: &[__m256i; 8],
    acts: &[QuantAct],
    sb: usize,
    sumi: &mut [__m256i],
) {
    for c in 0..2 {
        for g in 0..4 {
            let scale = scales[c * 4 + g];
            let o = sb * 256 + c * 128 + g * 32;
            for t in 0..T {
                let q8 = _mm256_loadu_si256(acts[t].qk.as_ptr().add(o) as *const __m256i);
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(codes[c * 4 + g], q8), scale);
                sumi[t] = _mm256_add_epi32(sumi[t], p);
            }
        }
    }
}

/// Q6_K against a wide batch, eight activations at a time with their accumulators
/// in registers.
///
/// A super-block's codes are decoded once and reused by the whole group; each row
/// then keeps its eight integer lane sums and its eight f32 lanes in registers
/// across the block loop, where a `MAX_WIDE`-long array of accumulators would be
/// loaded and stored once per block per row. The arithmetic is unchanged — the same
/// `maddubs`/`madd` lane sums, the same per-block `fma`, the same lane-wise float
/// accumulation and final horizontal sum — so a row's bits do not depend on the
/// window it lands in (the kernel tests compare every window against the scalar
/// reference). Eight is the widest group whose accumulators still fit: measured on
/// H0 at 16 activations, 6052 -> 5278 ns per 8192-wide row, and it removes the
/// width cliff the array form had above 16 (32 activations: 16354 -> 11528 ns).
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn wide_q6_k(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let n = acts.len();
    debug_assert!(n <= MAX_WIDE && acts.iter().all(|a| !a.qk.is_empty()));
    let lownib = _mm256_set1_epi8(0x0F);
    let two = _mm256_set1_epi8(0x03);
    let mut base = 0;
    while base < n {
        let tw = (n - base).min(8);
        let acts = &acts[base..base + tw];
        let mut acc = [_mm256_setzero_ps(); 8];
        for (sb, blk) in w.chunks_exact(210).enumerate() {
            prefetch_ahead(blk.as_ptr(), pf, 4);
            let d = f16_at(blk, 208);
            let sc16 =
                _mm256_cvtepi8_epi16(_mm_loadu_si128(blk.as_ptr().add(192) as *const __m128i));
            let scales = q6k_scales(sc16);
            let mut codes = [_mm256_setzero_si256(); 8];
            for c in 0..2 {
                let ql = blk.as_ptr().add(c * 64);
                let qh = _mm256_loadu_si256(blk.as_ptr().add(128 + c * 32) as *const __m256i);
                let ql0 = _mm256_loadu_si256(ql as *const __m256i);
                let ql1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
                for g in 0..4 {
                    let (b, hbits) = match g {
                        0 => (ql0, _mm256_and_si256(qh, two)),
                        1 => (ql1, _mm256_and_si256(_mm256_srli_epi16(qh, 2), two)),
                        2 => (ql0, _mm256_and_si256(_mm256_srli_epi16(qh, 4), two)),
                        _ => (ql1, _mm256_and_si256(_mm256_srli_epi16(qh, 6), two)),
                    };
                    let low = if g < 2 {
                        _mm256_and_si256(b, lownib)
                    } else {
                        _mm256_and_si256(_mm256_srli_epi16(b, 4), lownib)
                    };
                    codes[c * 4 + g] = _mm256_or_si256(low, _mm256_slli_epi16(hbits, 4));
                }
            }
            let mut sumi = [_mm256_setzero_si256(); 8];
            match tw {
                8 => q6k_acts::<8>(&codes, &scales, acts, sb, &mut sumi),
                7 => q6k_acts::<7>(&codes, &scales, acts, sb, &mut sumi),
                6 => q6k_acts::<6>(&codes, &scales, acts, sb, &mut sumi),
                5 => q6k_acts::<5>(&codes, &scales, acts, sb, &mut sumi),
                4 => q6k_acts::<4>(&codes, &scales, acts, sb, &mut sumi),
                3 => q6k_acts::<3>(&codes, &scales, acts, sb, &mut sumi),
                2 => q6k_acts::<2>(&codes, &scales, acts, sb, &mut sumi),
                _ => q6k_acts::<1>(&codes, &scales, acts, sb, &mut sumi),
            }
            for t in 0..tw {
                let a = &acts[t];
                let bs = _mm256_loadu_si256(a.bsums.as_ptr().add(sb * 16) as *const __m256i);
                let off = _mm256_slli_epi32(_mm256_madd_epi16(sc16, bs), 5);
                let v = _mm256_sub_epi32(sumi[t], off);
                acc[t] =
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(d * a.dk[sb]), acc[t]);
            }
        }
        for t in 0..tw {
            out[base + t] = hsum_ps(acc[t]);
        }
        base += tw;
    }
}

/// Activations one `wide_q5_0` window carries.
///
/// Sixteen is where the accumulator array still fits the core's L1 beside the
/// activation rows. Decoding once for a whole 64-token prefill chunk is cheaper on
/// paper but spills: measured on the 0.5B's 4864x896 shape at eight threads, a
/// single 64-wide pass gives ~98 G MAC/s against the tiled kernel's ~104, while
/// 16-wide windows give ~114.5 (12: 111, 20: 112, 24: 111, 32: 100). The window
/// also keeps the working set independent of `PREFILL_CHUNK`, so widening the
/// chunk cannot turn the kernel into a regression.
const Q5_0_WINDOW: usize = 16;

/// Q5_0 against a *wide* batch; see `wide_q4_k`.
///
/// A window of sixteen activations is run against each decoded block at once,
/// where the tiled kernel expands the block again for every group of four. The
/// per-activation arithmetic is the tiled kernel's — the same integer lane sums,
/// then one fused multiply-add per block into a vector accumulator, reduced once
/// per row — so a row's result is bit-identical at every batch width, and
/// independent of where the windows fall. The fifth bit is folded into the codes
/// as `codes − 16`, which carries the block's bias in the weight and removes the
/// per-activation lane-sum load, shift and subtract (exactly: `|c|·sign(c)·a` in
/// `vmaddubsw` is `c·a` for these ranges).
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn wide_q5_0(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let n = acts.len();
    let mut t = 0;
    while t < n {
        let tw = (n - t).min(Q5_0_WINDOW);
        wide_q5_0_group(w, &acts[t..t + tw], &mut out[t..t + tw]);
        t += tw;
    }
}

/// One window of `wide_q5_0`: all of `acts` against each block decoded once.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn wide_q5_0_group(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let pf = prefetch_distance();
    let n = acts.len();
    debug_assert!(n <= Q5_0_WINDOW);
    let lownib = _mm256_set1_epi8(0x0F);
    let nib_shift = _mm256_set_epi64x(4, 4, 0, 0);
    let sixteen = _mm256_set1_epi8(16);
    let nblk = w.len() / 22;
    for a in acts {
        assert!(a.d.len() >= nblk && a.q.len() >= nblk * 32);
    }
    let mut acc = [_mm256_setzero_ps(); Q5_0_WINDOW];
    for (b, blk) in w.chunks_exact(22).enumerate() {
        prefetch_ahead(blk.as_ptr(), pf, 1);
        let qh = (blk.as_ptr().add(2) as *const u32).read_unaligned();
        let qs =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(blk.as_ptr().add(6) as *const __m128i));
        let nib = _mm256_and_si256(_mm256_srlv_epi64(qs, nib_shift), lownib);
        // `codes − 16` makes the unsigned code a signed one, so the block's bias is
        // carried in the weights and the per-activation lane sums are not needed.
        // The exact integer product is `madd_offset_ps::<4>`'s.
        let codes = _mm256_sub_epi8(
            _mm256_or_si256(nib, q5_high_bits(u32::from_le(qh))),
            sixteen,
        );
        let absw = _mm256_sign_epi8(codes, codes);
        let dw = f16_at4(blk.as_ptr());
        for (t, a) in acts.iter().enumerate() {
            let sa = _mm256_sign_epi8(acts32(a, b), codes);
            let v = madd_ps(absw, sa);
            // SAFETY: `n` is at most `Q5_0_WINDOW` (checked above).
            let acc_t = acc.get_unchecked_mut(t);
            *acc_t = _mm256_fmadd_ps(v, _mm256_set1_ps(dw * *a.d.get_unchecked(b)), *acc_t);
        }
    }
    for b in 0..n {
        out[b] = hsum_ps(acc[b]);
    }
}

/// One weight row against `acts.len()` activations, writing `out[b]` for each.
///
/// # Safety
/// Caller must have checked `have_batched()`. `w` must be exactly
/// `row_bytes(fmt, a.n)` long for every activation `a`, and `out.len() >= acts.len()`.
pub unsafe fn dot_batch(fmt: Format, w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    #[allow(unused_macros)]
    macro_rules! run {
        ($f:ident) => {{
            let mut b = 0;
            while b + TILE <= acts.len() {
                $f::<4>(w, &acts[b..b + 4], &mut out[b..b + 4]);
                b += TILE;
            }
            match acts.len() - b {
                3 => $f::<3>(w, &acts[b..b + 3], &mut out[b..b + 3]),
                2 => $f::<2>(w, &acts[b..b + 2], &mut out[b..b + 2]),
                1 => $f::<1>(w, &acts[b..b + 1], &mut out[b..b + 1]),
                _ => {}
            }
        }};
    }
    match fmt {
        // A prompt-sized batch decodes the row once for all of it; decode and the
        // speculative block (≤ 4) keep the tiled kernels, so their figures are unchanged.
        #[cfg(feature = "fmt-legacy")]
        Format::Q5_0 if acts.len() > TILE && acts.len() <= MAX_WIDE && wide_enabled() => {
            wide_q5_0(w, acts, out)
        }
        #[cfg(feature = "fmt-legacy")]
        Format::Q8_0 => run!(tile_q8_0),
        #[cfg(feature = "fmt-legacy")]
        Format::Q4_0 => run!(tile_q4_0),
        #[cfg(feature = "fmt-legacy")]
        Format::Q5_0 => run!(tile_q5_0),
        // A prompt-sized batch decodes the row once for all of it; decode and the
        // speculative block (≤ 4) keep the tiled kernels, so their figures are unchanged.
        #[cfg(feature = "fmt-kquant")]
        Format::Q4_K if acts.len() > TILE && acts.len() <= MAX_WIDE && wide_enabled() => {
            wide_q4_k(w, acts, out)
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q5_K if acts.len() > TILE && acts.len() <= MAX_WIDE && wide_enabled() => {
            wide_q5_k(w, acts, out)
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q6_K if acts.len() > TILE && acts.len() <= MAX_WIDE && wide_enabled() => {
            wide_q6_k(w, acts, out)
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q4_K => run!(tile_q4_k),
        #[cfg(feature = "fmt-kquant")]
        Format::Q5_K => run!(tile_q5_k),
        #[cfg(feature = "fmt-kquant")]
        Format::Q6_K => run!(tile_q6_k),
        #[cfg(feature = "fmt-kquant")]
        Format::Q2_K => run!(tile_q2_k),
        #[cfg(feature = "fmt-kquant")]
        Format::Q3_K => run!(tile_q3_k),
        #[cfg(feature = "fmt-iq")]
        Format::IQ4_NL => run!(tile_iq4_nl),
        #[cfg(feature = "fmt-iq")]
        Format::IQ4_XS => run!(tile_iq4_xs),
        #[cfg(feature = "fmt-mxfp4")]
        Format::MXFP4 => run!(tile_mxfp4),
        Format::F32 => run!(tile_f32),
        Format::F16 => run!(tile_f16),
        Format::BF16 => run!(tile_bf16),
        #[cfg(feature = "fmt-ternary")]
        Format::PQ2_0 => crate::tern::dot_batch_pq2(w, acts, out),
        #[cfg(feature = "fmt-ternary")]
        Format::PTQ1_0 => crate::tern::dot_batch_ptq1(w, acts, out),
        #[allow(unreachable_patterns)]
        _ => panic!("format {} disabled in this build", fmt.label()),
    }
}

/// `|x|` maximum over a slice, eight lanes at a time. `max` is exact and order-free
/// for finite values, so this equals the scalar fold.
///
/// # Safety
/// AVX2 must be available (`have_avx2`).
#[target_feature(enable = "avx2")]
pub unsafe fn absmax(x: &[f32]) -> f32 {
    let sign = _mm256_set1_ps(-0.0);
    let mut m = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= x.len() {
        m = _mm256_max_ps(
            m,
            _mm256_andnot_ps(sign, _mm256_loadu_ps(x.as_ptr().add(i))),
        );
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), m);
    let mut r = lanes.iter().fold(0.0f32, |a, &b| a.max(b));
    for &v in &x[i..] {
        r = r.max(v.abs());
    }
    r
}

/// `out[i] = clamp(round(x[i] · inv), ±127)` for 32 values — `QuantAct::fill`'s
/// inner loop. Rust's `round` goes half away from zero, which no single AVX rounding
/// mode does; it is rebuilt exactly as `trunc(|y|) + (frac ≥ 0.5)` (the fraction of a
/// float below 2²³ is exact), with the sign put back. Bit-identical to the scalar
/// loop for every finite input (`vector_quantisation_matches_the_scalar_loop`).
///
/// # Safety
/// AVX2 must be available; `x` must point at 32 readable floats and `out` at 32
/// writable bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn quantize32(x: *const f32, inv: f32, out: *mut i8) {
    let vinv = _mm256_set1_ps(inv);
    let sign = _mm256_set1_ps(-0.0);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let hi = _mm256_set1_ps(127.0);
    let lo = _mm256_set1_ps(-127.0);
    let mut iv = [_mm256_setzero_si256(); 4];
    for (k, v) in iv.iter_mut().enumerate() {
        let y = _mm256_mul_ps(_mm256_loadu_ps(x.add(8 * k)), vinv);
        let ay = _mm256_andnot_ps(sign, y);
        let t = _mm256_round_ps(ay, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let up = _mm256_and_ps(_mm256_cmp_ps(_mm256_sub_ps(ay, t), half, _CMP_GE_OQ), one);
        let r = _mm256_or_ps(_mm256_add_ps(t, up), _mm256_and_ps(y, sign));
        *v = _mm256_cvttps_epi32(_mm256_min_ps(_mm256_max_ps(r, lo), hi));
    }
    // i32 → i16 → i8 packs interleave the 128-bit lanes; the permute restores order.
    let a = _mm256_packs_epi32(iv[0], iv[1]);
    let b = _mm256_packs_epi32(iv[2], iv[3]);
    let c = _mm256_packs_epi16(a, b);
    let c = _mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256(out as *mut __m256i, c);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The eight-row Q5_0 decode kernel against the one-row kernel: every row must
    /// come out bit for bit the same, or a row's result would depend on how many rows
    /// the matmul happened to run beside it.
    /// The eight-row Q8_0 decode kernel against the one-row kernel, bit for bit.
    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q8_0_rows8_matches_the_one_row_kernel_exactly() {
        if !have_batched() {
            return;
        }
        let mut s = 0x0bad_c0de_1234_5678u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let rb = 34 * 20;
        let mut w = vec![0u8; 8 * rb];
        for r in 0..8 {
            for b in 0..20 {
                let blk = &mut w[r * rb + b * 34..r * rb + (b + 1) * 34];
                for x in blk.iter_mut() {
                    *x = next() as u8;
                }
                blk[..2].copy_from_slice(&crate::quant::f32_to_f16(0.01).to_le_bytes());
            }
        }
        let x: Vec<f32> = (0..rb / 34 * 32)
            .map(|_| (next() as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * 3.0)
            .collect();
        let mut a = QuantAct::with_capacity(x.len());
        a.fill(&x);
        let mut got = [0.0f32; 8];
        // SAFETY: AVX2, FMA and F16C checked above; lengths are the kernel's asserts.
        unsafe { q8_0_rows8(&w, rb, &a, &mut got) };
        for r in 0..8 {
            let mut want = [0.0f32; 1];
            // SAFETY: as above.
            unsafe {
                tile_q8_0::<1>(
                    &w[r * rb..(r + 1) * rb],
                    std::slice::from_ref(&a),
                    &mut want,
                )
            };
            assert_eq!(
                got[r].to_bits(),
                want[0].to_bits(),
                "row {r}: {} vs {}",
                got[r],
                want[0]
            );
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q5_0_rows8_matches_the_one_row_kernel_exactly() {
        if !have_batched() {
            return;
        }
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let rb = 22 * 20;
        let mut w = vec![0u8; 8 * rb];
        for r in 0..8 {
            for b in 0..20 {
                let blk = &mut w[r * rb + b * 22..r * rb + (b + 1) * 22];
                for x in blk.iter_mut() {
                    *x = next() as u8;
                }
                // A real f16 scale (a small positive one), not random bits.
                blk[..2].copy_from_slice(&crate::quant::f32_to_f16(0.01).to_le_bytes());
            }
        }
        let x: Vec<f32> = (0..rb / 22 * 32)
            .map(|_| (next() as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * 3.0)
            .collect();
        let mut a = QuantAct::with_capacity(x.len());
        a.fill(&x);
        let mut got = [0.0f32; 8];
        // SAFETY: AVX2, FMA and F16C checked above; the lengths are the ones the
        // kernel asserts.
        unsafe { q5_0_rows8(&w, rb, &a, &mut got) };
        for r in 0..8 {
            let mut want = [0.0f32; 1];
            // SAFETY: as above.
            unsafe {
                tile_q5_0::<1>(
                    &w[r * rb..(r + 1) * rb],
                    std::slice::from_ref(&a),
                    &mut want,
                )
            };
            assert_eq!(
                got[r].to_bits(),
                want[0].to_bits(),
                "row {r}: {} vs {}",
                got[r],
                want[0]
            );
        }
    }

    #[test]
    fn the_saturation_headroom_is_real() {
        // VPMADDUBSW sums two products into a saturating i16. The widest case in
        // this file is Q6_K: codes reach 63 and activations are clamped to ±127.
        // If `QuantAct::fill` ever allowed -128, this margin disappears and the
        // kernel would silently saturate on large activations.
        assert!(63 * 127 * 2 < i16::MAX as i32, "Q6_K would saturate");
        assert!(31 * 127 * 2 < i16::MAX as i32, "Q5_0 would saturate");
        assert!(31 * 127 * 2 < i16::MAX as i32, "Q5_K would saturate");
        assert!(15 * 127 * 2 < i16::MAX as i32, "Q4_K would saturate");
        assert!(15 * 127 * 2 < i16::MAX as i32, "Q4_0 would saturate");
        assert!(3 * 127 * 2 < i16::MAX as i32, "Q2_K would saturate");
        assert!(4 * 127 * 2 < i16::MAX as i32, "Q3_K would saturate");
        assert!(
            127 * 127 * 2 < i16::MAX as i32,
            "IQ4_NL / IQ4_XS would saturate"
        );
        assert!(12 * 127 * 2 < i16::MAX as i32, "MXFP4 would saturate");
        // Q8_0 is the tight one. Its codes are signed, and `_mm256_sign_epi8` maps a
        // code of -128 to unsigned 128 (negation overflows back to itself), so the
        // unsigned operand reaches 128 and not 127.
        assert!(128 * 127 * 2 < i16::MAX as i32, "Q8_0 would saturate");
        // An activation of -128 against that same 128 would reach exactly 32768,
        // one past i16::MAX, and saturate. That is why `QuantAct::fill` clamps to
        // ±127 rather than to the full i8 range — the clamp is load-bearing here,
        // not defensive.
        assert!(
            128 * 128 * 2 > i16::MAX as i32,
            "the ±127 activation clamp is load-bearing"
        );
    }

    /// The word-wise unpacking and the vector min term against the byte-wise
    /// reference, on random scale bytes and run sums (including the i16 extremes a
    /// 256-block of ±127 can produce).
    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn q4k_scale_unpacking_and_min_term_match_the_reference() {
        if !have_batched() {
            return;
        }
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..20_000 {
            let q: Vec<u8> = (0..12).map(|_| (rnd() >> 32) as u8).collect();
            let sm = unpack_q4k_scales(&q);
            let mut bs = [0i16; 16];
            for v in bs.iter_mut() {
                *v = ((rnd() >> 40) as i32 % 2033 - 1016) as i16;
            }
            let mut want = 0i32;
            for j in 0..8 {
                let (sc, m) = scale_min_k4(j, &q);
                assert_eq!((sm[j], sm[8 + j]), (sc, m), "j={j} q={q:?}");
                want += m as i32 * (bs[2 * j] as i32 + bs[2 * j + 1] as i32);
            }
            let got = unsafe {
                let lanes = _mm256_madd_epi16(
                    q4k_mins16(&sm[8..16]),
                    _mm256_loadu_si256(bs.as_ptr() as *const __m256i),
                );
                hsum_i32(lanes)
            };
            assert_eq!(got, want);
        }
    }

    /// Every code and every sum of the vector `fill` against the scalar loop, on
    /// random activations plus the cases that break a naive rounding: exact halves,
    /// the float just below a half, values past the clamp, zeros and a zero block.
    #[test]
    fn vector_quantisation_matches_the_scalar_loop() {
        if !have_avx2() {
            return;
        }
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for trial in 0..400 {
            let n = if trial % 2 == 0 { 2560 } else { 960 };
            let mut x: Vec<f32> = (0..n)
                .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0)
                .collect();
            if trial % 5 == 0 {
                for v in x[..32].iter_mut() {
                    *v = 0.0;
                }
            }
            for (i, v) in x.iter_mut().enumerate().skip(40).step_by(97) {
                // Values that land exactly on, or one ulp under, a half after scaling.
                *v = match i % 4 {
                    0 => 0.5,
                    1 => -2.5,
                    2 => f32::from_bits(0.5f32.to_bits() - 1),
                    _ => -127.5,
                };
            }
            let mut a = QuantAct::with_capacity(n);
            a.fill(&x);
            let mut b = QuantAct::with_capacity(n);
            b.fill_scalar(&x);
            assert_eq!(a.q, b.q, "trial {trial}");
            assert_eq!(a.d, b.d);
            assert_eq!(
                (&a.sums, &a.sums16, &a.lane_sums),
                (&b.sums, &b.sums16, &b.lane_sums)
            );
            assert_eq!((&a.qk, &a.dk, &a.bsums), (&b.qk, &b.dk, &b.bsums));
        }
    }

    #[test]
    fn f16c_matches_the_reference_conversion() {
        if !have_batched() {
            return;
        }
        for h in 0..=u16::MAX {
            let want = f16(h);
            let got = unsafe { f16c_one(h) };
            // NaN payloads aside, the two must be the same bits.
            if want.is_nan() {
                assert!(got.is_nan(), "{h:#06x}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "{h:#06x}");
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    #[allow(clippy::needless_range_loop)]
    fn q5_high_bit_expansion_maps_bit_i_to_byte_i() {
        if !have_avx2() {
            return;
        }
        for qh in [
            0u32,
            0xFFFF_FFFF,
            0x0000_0001,
            0x8000_0000,
            0xDEAD_BEEF,
            0x5555_5555,
        ] {
            let mut got = [0u8; 32];
            unsafe {
                _mm256_storeu_si256(got.as_mut_ptr() as *mut __m256i, q5_high_bits(qh));
            }
            for i in 0..32 {
                let want = if (qh >> i) & 1 == 1 { 0x10 } else { 0x00 };
                assert_eq!(got[i], want, "qh={qh:#010x} bit {i}");
            }
        }
    }

    #[cfg_attr(
        not(all(
            feature = "fmt-kquant",
            feature = "fmt-ternary",
            feature = "fmt-iq",
            feature = "fmt-mxfp4",
            feature = "fmt-legacy"
        )),
        allow(unused_variables, dead_code)
    )]
    fn sane_scales(fmt: Format, w: &mut [u8]) {
        let fix = |b: &mut [u8], at: usize| b[at + 1] = (b[at + 1] & 0x0F) | 0x30;
        let sz = fmt.block_bytes();
        for blk in w.chunks_exact_mut(sz) {
            match fmt {
                Format::Q4_0 | Format::Q8_0 | Format::Q5_0 | Format::IQ4_NL => fix(blk, 0),
                Format::Q2_K => {
                    fix(blk, 80);
                    fix(blk, 82);
                }
                Format::Q3_K => fix(blk, 108),
                Format::Q4_K | Format::Q5_K => {
                    fix(blk, 0);
                    fix(blk, 2);
                }
                Format::IQ4_XS => fix(blk, 0),
                Format::MXFP4 => blk[0] = (blk[0] % 30) + 115,
                _ => {}
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn avx2_dot_matches_dot_scalar_bit_for_bit_q4_0_and_q5_0() {
        if !have_avx2() {
            return;
        }
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        // Test random rows, odd block counts (32 = 1 blk, 96 = 3 blks, 160 = 5 blks),
        // plus zero blocks and max codes.
        for fmt in [Format::Q4_0, Format::Q5_0] {
            for n in [32, 96, 160, 256, 960] {
                let rb = crate::quant::row_bytes(fmt, n);
                // 1. Random rows
                for trial in 0..10 {
                    let mut w: Vec<u8> = (0..rb).map(|_| (rnd() >> 32) as u8).collect();
                    sane_scales(fmt, &mut w);
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() was checked above; w has row_bytes(fmt, n) and a has length n.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} trial {trial} n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                }

                // 2. Zero blocks (scale = 0 and codes = 0)
                {
                    let w = vec![0u8; rb];
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked, w has row_bytes(fmt, n) and a has length n.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} zero blocks n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                }

                // 3. Max codes (all nibbles 0xF, qh 0xFFFFFFFF for Q5_0)
                {
                    let mut w = vec![0xFFu8; rb];
                    sane_scales(fmt, &mut w);
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked, w has row_bytes(fmt, n) and a has length n.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} max codes n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                }

                // 4. Min codes (all nibbles 0x0, qh 0x00000000 for Q5_0)
                {
                    let mut w = vec![0x00u8; rb];
                    sane_scales(fmt, &mut w);
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked, w has row_bytes(fmt, n) and a has length n.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} min codes n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn avx2_dot_matches_dot_scalar_q5_k() {
        if !have_avx2() || !have_batched() {
            return;
        }
        let fmt = Format::Q5_K;
        let mut s = 0x5555_AAAA_1234_5678u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        for n in [256, 512, 1024, 2560] {
            let rb = crate::quant::row_bytes(fmt, n);

            // 1. Random rows: single-row and batched
            for trial in 0..10 {
                let mut w: Vec<u8> = (0..rb).map(|_| (rnd() >> 32) as u8).collect();
                sane_scales(fmt, &mut w);
                let x: Vec<f32> = (0..n)
                    .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n);
                a.fill(&x);
                // SAFETY: have_avx2() checked; w has row_bytes(fmt, n) and a has length n.
                let got = unsafe { dot(fmt, &w, &a) };
                let want = crate::quant::dot_scalar(fmt, &w, &a);
                let tol = 2e-4 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tol,
                    "Q5_K trial {trial} n={n}: avx2 {got} vs scalar {want} (diff {})",
                    (got - want).abs()
                );

                // Batched kernels across widths 1..=5
                for nb in 1..=5usize {
                    let acts: Vec<QuantAct> = (0..nb)
                        .map(|_i| {
                            let mut act = QuantAct::with_capacity(n);
                            let xi: Vec<f32> = (0..n)
                                .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
                                .collect();
                            act.fill(&xi);
                            act
                        })
                        .collect();
                    let mut bgot = vec![0.0f32; nb];
                    // SAFETY: have_batched() checked; lengths match preconditions.
                    unsafe { dot_batch(fmt, &w, &acts, &mut bgot) };
                    for (b, act) in acts.iter().enumerate() {
                        let bwant = crate::quant::dot_scalar(fmt, &w, act);
                        let btol = 2e-4 * bwant.abs().max(1.0);
                        assert!(
                            (bgot[b] - bwant).abs() <= btol,
                            "Q5_K batched nb={nb} act {b}: got {} vs scalar {bwant}",
                            bgot[b]
                        );
                    }
                }
            }

            // 2. Zero blocks (scales = 0, mins = 0, codes = 0)
            {
                let w = vec![0u8; rb];
                let x: Vec<f32> = (0..n)
                    .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n);
                a.fill(&x);
                // SAFETY: have_avx2() checked.
                let got = unsafe { dot(fmt, &w, &a) };
                let want = crate::quant::dot_scalar(fmt, &w, &a);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "Q5_K zero blocks n={n}: avx2 {got} vs scalar {want}"
                );
                let mut bgot = [0.0f32; 1];
                // SAFETY: have_batched() checked.
                unsafe { dot_batch(fmt, &w, std::slice::from_ref(&a), &mut bgot) };
                assert_eq!(
                    bgot[0].to_bits(),
                    want.to_bits(),
                    "Q5_K batched zero blocks n={n}: got {} vs scalar {want}",
                    bgot[0]
                );
            }

            // 3. Max codes (all 0xFF bytes)
            {
                let mut w = vec![0xFFu8; rb];
                sane_scales(fmt, &mut w);
                let x: Vec<f32> = (0..n)
                    .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n);
                a.fill(&x);
                // SAFETY: have_avx2() checked.
                let got = unsafe { dot(fmt, &w, &a) };
                let want = crate::quant::dot_scalar(fmt, &w, &a);
                let tol = 2e-4 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tol,
                    "Q5_K max codes n={n}: avx2 {got} vs scalar {want}"
                );
            }
        }
    }

    /// The wide Q6_K kernel takes its activations eight at a time; a row's bits must
    /// be the scalar reference's whatever window it lands in. The widths cover whole
    /// groups and every ragged tail around them.
    #[test]
    fn wide_q6_k_matches_one_row_at_every_width() {
        if !have_avx2() || !have_batched() || !wide_enabled() {
            return;
        }
        let fmt = Format::Q6_K;
        let mut s = 0x6B6B_1234_9876_5401u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let n = 1024;
        let rb = crate::quant::row_bytes(fmt, n);
        for trial in 0..5 {
            let mut w: Vec<u8> = (0..rb).map(|_| (rnd() >> 32) as u8).collect();
            sane_scales(fmt, &mut w);
            let acts: Vec<QuantAct> = (0..32)
                .map(|_| {
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    a
                })
                .collect();
            for &nb in &[5usize, 6, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32] {
                let mut got = vec![0.0f32; nb];
                // SAFETY: have_avx2()/have_batched() checked; lengths match preconditions.
                unsafe { dot_batch(fmt, &w, &acts[..nb], &mut got) };
                for b in 0..nb {
                    // SAFETY: as above; `acts[b]` has length n.
                    let want = unsafe { dot(fmt, &w, &acts[b]) };
                    assert_eq!(
                        got[b].to_bits(),
                        want.to_bits(),
                        "Q6_K trial {trial} row {b}/{nb}: wide {} vs one row {want}",
                        got[b]
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-iq")]
    fn avx2_dot_matches_dot_scalar_bit_for_bit_iq4_nl_and_iq4_xs() {
        if !have_avx2() || !have_batched() {
            return;
        }
        let mut s = 0x9876_5432_10FE_DCBAu64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        for fmt in [Format::IQ4_NL, Format::IQ4_XS] {
            let lens: &[usize] = if fmt == Format::IQ4_NL {
                &[32, 96, 160, 256, 960]
            } else {
                &[256, 512, 1024, 2560]
            };
            for &n in lens {
                let rb = crate::quant::row_bytes(fmt, n);

                // 1. Random rows: single-row and batched
                for trial in 0..10 {
                    let mut w: Vec<u8> = (0..rb).map(|_| (rnd() >> 32) as u8).collect();
                    sane_scales(fmt, &mut w);
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked; w has row_bytes(fmt, n) and a has length n.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} trial {trial} n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );

                    // Batched kernels across widths 1..=5
                    for nb in 1..=5usize {
                        let acts: Vec<QuantAct> = (0..nb)
                            .map(|_| {
                                let mut act = QuantAct::with_capacity(n);
                                let xi: Vec<f32> = (0..n)
                                    .map(|_| {
                                        ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0
                                    })
                                    .collect();
                                act.fill(&xi);
                                act
                            })
                            .collect();
                        let mut bgot = vec![0.0f32; nb];
                        // SAFETY: have_batched() checked; lengths match preconditions.
                        unsafe { dot_batch(fmt, &w, &acts, &mut bgot) };
                        for (b, act) in acts.iter().enumerate() {
                            let bwant = crate::quant::dot_scalar(fmt, &w, act);
                            assert_eq!(
                                bgot[b].to_bits(),
                                bwant.to_bits(),
                                "{:?} batched nb={nb} act {b} n={n}: got {} vs scalar {bwant}",
                                fmt.label(),
                                bgot[b]
                            );
                        }
                    }
                }

                // 2. Zero blocks (scale = 0, codes = 0)
                {
                    let w = vec![0u8; rb];
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} zero blocks n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                    let mut bgot = [0.0f32; 1];
                    // SAFETY: have_batched() checked.
                    unsafe { dot_batch(fmt, &w, std::slice::from_ref(&a), &mut bgot) };
                    assert_eq!(
                        bgot[0].to_bits(),
                        want.to_bits(),
                        "{:?} batched zero blocks n={n}: got {} vs scalar {want}",
                        fmt.label(),
                        bgot[0]
                    );
                }

                // 3. Max codes (all 0xFF bytes)
                {
                    let mut w = vec![0xFFu8; rb];
                    sane_scales(fmt, &mut w);
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} max codes n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                    let mut bgot = [0.0f32; 1];
                    // SAFETY: have_batched() checked.
                    unsafe { dot_batch(fmt, &w, std::slice::from_ref(&a), &mut bgot) };
                    assert_eq!(
                        bgot[0].to_bits(),
                        want.to_bits(),
                        "{:?} batched max codes n={n}: got {} vs scalar {want}",
                        fmt.label(),
                        bgot[0]
                    );
                }

                // 4. Min codes (all 0x00 bytes) with sane scales
                {
                    let mut w = vec![0x00u8; rb];
                    sane_scales(fmt, &mut w);
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} min codes n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                    let mut bgot = [0.0f32; 1];
                    // SAFETY: have_batched() checked.
                    unsafe { dot_batch(fmt, &w, std::slice::from_ref(&a), &mut bgot) };
                    assert_eq!(
                        bgot[0].to_bits(),
                        want.to_bits(),
                        "{:?} batched min codes n={n}: got {} vs scalar {want}",
                        fmt.label(),
                        bgot[0]
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "fmt-mxfp4")]
    fn avx2_dot_matches_dot_scalar_bit_for_bit_mxfp4() {
        if !have_avx2() || !have_batched() {
            return;
        }
        let mut s = 0xABCD_EF01_2345_6789u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        let fmt = Format::MXFP4;
        let lens: &[usize] = &[32, 64, 96, 128, 160, 256, 512, 960];
        for &n in lens {
            let rb = crate::quant::row_bytes(fmt, n);

            // 1. Random rows: single-row and batched
            for trial in 0..10 {
                let mut w: Vec<u8> = (0..rb).map(|_| (rnd() >> 32) as u8).collect();
                sane_scales(fmt, &mut w);
                let x: Vec<f32> = (0..n)
                    .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n);
                a.fill(&x);
                // SAFETY: have_avx2() checked; w has row_bytes(fmt, n) and a has length n.
                let got = unsafe { dot(fmt, &w, &a) };
                let want = crate::quant::dot_scalar(fmt, &w, &a);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{:?} trial {trial} n={n}: avx2 {got} vs scalar {want}",
                    fmt.label()
                );

                // Batched kernels across widths 1..=5
                for nb in 1..=5usize {
                    let acts: Vec<QuantAct> = (0..nb)
                        .map(|_| {
                            let mut act = QuantAct::with_capacity(n);
                            let xi: Vec<f32> = (0..n)
                                .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
                                .collect();
                            act.fill(&xi);
                            act
                        })
                        .collect();
                    let mut bgot = vec![0.0f32; nb];
                    // SAFETY: have_batched() checked; lengths match preconditions.
                    unsafe { dot_batch(fmt, &w, &acts, &mut bgot) };
                    for (b, act) in acts.iter().enumerate() {
                        let bwant = crate::quant::dot_scalar(fmt, &w, act);
                        assert_eq!(
                            bgot[b].to_bits(),
                            bwant.to_bits(),
                            "{:?} batched nb={nb} act {b} n={n}: got {} vs scalar {bwant}",
                            fmt.label(),
                            bgot[b]
                        );
                    }
                }
            }

            // 2. Zero blocks (scale = 0, codes = 0)
            {
                let w = vec![0u8; rb];
                let x: Vec<f32> = (0..n)
                    .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n);
                a.fill(&x);
                // SAFETY: have_avx2() checked.
                let got = unsafe { dot(fmt, &w, &a) };
                let want = crate::quant::dot_scalar(fmt, &w, &a);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{:?} zero blocks n={n}: avx2 {got} vs scalar {want}",
                    fmt.label()
                );
                let mut bgot = [0.0f32; 1];
                // SAFETY: have_batched() checked.
                unsafe { dot_batch(fmt, &w, std::slice::from_ref(&a), &mut bgot) };
                assert_eq!(
                    bgot[0].to_bits(),
                    want.to_bits(),
                    "{:?} batched zero blocks n={n}: got {} vs scalar {want}",
                    fmt.label(),
                    bgot[0]
                );
            }

            // 3. Max codes (all 0xFF bytes)
            {
                let mut w = vec![0xFFu8; rb];
                sane_scales(fmt, &mut w);
                let x: Vec<f32> = (0..n)
                    .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n);
                a.fill(&x);
                // SAFETY: have_avx2() checked.
                let got = unsafe { dot(fmt, &w, &a) };
                let want = crate::quant::dot_scalar(fmt, &w, &a);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{:?} max codes n={n}: avx2 {got} vs scalar {want}",
                    fmt.label()
                );
                let mut bgot = [0.0f32; 1];
                // SAFETY: have_batched() checked.
                unsafe { dot_batch(fmt, &w, std::slice::from_ref(&a), &mut bgot) };
                assert_eq!(
                    bgot[0].to_bits(),
                    want.to_bits(),
                    "{:?} batched max codes n={n}: got {} vs scalar {want}",
                    fmt.label(),
                    bgot[0]
                );
            }

            // 4. Extreme exponents (e8m0 edge values: 0, 1, 2, 127, 128, 254, 255)
            for &edge_e in &[0u8, 1, 2, 127, 128, 254, 255] {
                let mut w: Vec<u8> = (0..rb).map(|_| (rnd() >> 32) as u8).collect();
                for blk in w.chunks_exact_mut(17) {
                    blk[0] = edge_e;
                }
                let x: Vec<f32> = (0..n)
                    .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n);
                a.fill(&x);
                // SAFETY: have_avx2() checked.
                let got = unsafe { dot(fmt, &w, &a) };
                let want = crate::quant::dot_scalar(fmt, &w, &a);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{:?} edge e={edge_e} n={n}: avx2 {got} vs scalar {want}",
                    fmt.label()
                );
                let mut bgot = [0.0f32; 1];
                // SAFETY: have_batched() checked.
                unsafe { dot_batch(fmt, &w, std::slice::from_ref(&a), &mut bgot) };
                assert_eq!(
                    bgot[0].to_bits(),
                    want.to_bits(),
                    "{:?} batched edge e={edge_e} n={n}: got {} vs scalar {want}",
                    fmt.label(),
                    bgot[0]
                );
            }
        }
    }

    #[test]
    fn avx2_dot_matches_dot_scalar_bit_for_bit_float_formats() {
        if !have_avx2() || !have_batched() {
            return;
        }

        let mut seed = 0x51E2_F321_u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let lengths = [1, 3, 7, 8, 9, 15, 16, 32, 33, 64, 97, 128, 255, 256, 960];
        let formats = [Format::F32, Format::F16, Format::BF16];

        for fmt in formats {
            for &n in &lengths {
                let gen_w = |rnd: &mut dyn FnMut() -> u64| -> Vec<u8> {
                    let mut w = Vec::with_capacity(crate::quant::row_bytes(fmt, n));
                    for _ in 0..n {
                        match fmt {
                            Format::F32 => {
                                let v = ((rnd() >> 32) as f32 / (1u64 << 32) as f32 - 0.5) * 10.0;
                                w.extend_from_slice(&v.to_le_bytes());
                            }
                            Format::F16 => {
                                // Exponent <= 30 to stay finite (no inf/nan)
                                let h =
                                    ((rnd() >> 48) as u16 & 0x7bff) | ((rnd() >> 63) as u16) << 15;
                                w.extend_from_slice(&h.to_le_bytes());
                            }
                            Format::BF16 => {
                                // Exponent <= 254 to stay finite
                                let mut raw = (rnd() >> 48) as u16;
                                if (raw & 0x7f80) == 0x7f80 {
                                    raw &= 0x7f7f;
                                }
                                w.extend_from_slice(&raw.to_le_bytes());
                            }
                            _ => unreachable!(),
                        }
                    }
                    w
                };

                // 1. Random trials (single-row and batched)
                for trial in 0..5 {
                    let w = gen_w(&mut rnd);
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 32) as f32 / (1u64 << 32) as f32 - 0.5) * 6.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.n = n;
                    a.keep_f32 = true;
                    a.xf.clear();
                    a.xf.extend_from_slice(&x);

                    // Single-row
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} trial {trial} n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );

                    // Batched (nb = 1, 2, 3, 4, 7)
                    for nb in [1, 2, 3, 4, 7] {
                        let acts: Vec<QuantAct> = (0..nb)
                            .map(|_| {
                                let xi: Vec<f32> = (0..n)
                                    .map(|_| {
                                        ((rnd() >> 32) as f32 / (1u64 << 32) as f32 - 0.5) * 6.0
                                    })
                                    .collect();
                                let mut act = QuantAct::with_capacity(n);
                                act.n = n;
                                act.keep_f32 = true;
                                act.xf.clear();
                                act.xf.extend_from_slice(&xi);
                                act
                            })
                            .collect();
                        let mut bgot = vec![0.0f32; nb];
                        unsafe { dot_batch(fmt, &w, &acts, &mut bgot) };
                        for b in 0..nb {
                            let bwant = crate::quant::dot_scalar(fmt, &w, &acts[b]);
                            assert_eq!(
                                bgot[b].to_bits(),
                                bwant.to_bits(),
                                "{:?} trial {trial} n={n} b={b}/{nb}: avx2 {} vs scalar {bwant}",
                                fmt.label(),
                                bgot[b]
                            );
                        }
                    }
                }

                // 2. Extremes (subnormal/inf/nan-free large and small floats)
                {
                    let mut w = Vec::with_capacity(crate::quant::row_bytes(fmt, n));
                    for i in 0..n {
                        let sign: f32 = if i % 2 == 0 { 1.0 } else { -1.0 };
                        let mag: f32 = if i % 4 == 0 { 1000.0 } else { 0.001 };
                        let v: f32 = sign * mag;
                        match fmt {
                            Format::F32 => w.extend_from_slice(&v.to_le_bytes()),
                            Format::F16 => {
                                w.extend_from_slice(&crate::quant::f32_to_f16(v).to_le_bytes())
                            }
                            Format::BF16 => {
                                let bits = (v.to_bits() >> 16) as u16;
                                w.extend_from_slice(&bits.to_le_bytes());
                            }
                            _ => unreachable!(),
                        }
                    }
                    let x: Vec<f32> = (0..n)
                        .map(|i| if i % 3 == 0 { 500.0 } else { -0.005 })
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.n = n;
                    a.keep_f32 = true;
                    a.xf.clear();
                    a.xf.extend_from_slice(&x);
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} extremes n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                }

                // 3. Zeroes
                {
                    let w = vec![0u8; crate::quant::row_bytes(fmt, n)];
                    let x = vec![1.0f32; n];
                    let mut a = QuantAct::with_capacity(n);
                    a.n = n;
                    a.keep_f32 = true;
                    a.xf.clear();
                    a.xf.extend_from_slice(&x);
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} zeroes n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "float matrix matmul reached with keep_f32 off on QuantAct")]
    fn simd_float_matmul_panics_when_keep_f32_is_off() {
        if !have_avx2() {
            panic!("float matrix matmul reached with keep_f32 off on QuantAct");
        }
        let mut a = QuantAct::with_capacity(32);
        a.n = 32;
        let w = [0u8; 128];
        unsafe { dot(Format::F32, &w, &a) };
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn avx2_dot_matches_dot_scalar_bit_for_bit_q2_k_and_q3_k() {
        if !have_avx2() || !have_batched() {
            return;
        }
        let mut s = 0xABCD_EF01_2345_6789u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        for fmt in [Format::Q2_K, Format::Q3_K] {
            for n in [256, 512, 1024, 2560] {
                let rb = crate::quant::row_bytes(fmt, n);

                // 1. Random rows: single-row and batched across widths 1..=5
                for trial in 0..10 {
                    let mut w: Vec<u8> = (0..rb).map(|_| (rnd() >> 32) as u8).collect();
                    sane_scales(fmt, &mut w);
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked; w has row_bytes(fmt, n) and a has length n.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} trial {trial} n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );

                    // Batched kernels across widths 1..=5
                    for nb in 1..=5usize {
                        let acts: Vec<QuantAct> = (0..nb)
                            .map(|_| {
                                let mut act = QuantAct::with_capacity(n);
                                let xi: Vec<f32> = (0..n)
                                    .map(|_| {
                                        ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0
                                    })
                                    .collect();
                                act.fill(&xi);
                                act
                            })
                            .collect();
                        let mut bgot = vec![0.0f32; nb];
                        // SAFETY: have_batched() checked; lengths match preconditions.
                        unsafe { dot_batch(fmt, &w, &acts, &mut bgot) };
                        for (b, act) in acts.iter().enumerate() {
                            let bwant = crate::quant::dot_scalar(fmt, &w, act);
                            assert_eq!(
                                bgot[b].to_bits(),
                                bwant.to_bits(),
                                "{:?} batched nb={nb} act {b} n={n}: got {} vs scalar {bwant}",
                                fmt.label(),
                                bgot[b]
                            );
                        }
                    }
                }

                // 2. Zero blocks (scale = 0, codes = 0)
                {
                    let w = vec![0u8; rb];
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} zero blocks n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                    let mut bgot = [0.0f32; 1];
                    // SAFETY: have_batched() checked.
                    unsafe { dot_batch(fmt, &w, std::slice::from_ref(&a), &mut bgot) };
                    assert_eq!(
                        bgot[0].to_bits(),
                        want.to_bits(),
                        "{:?} batched zero blocks n={n}: got {} vs scalar {want}",
                        fmt.label(),
                        bgot[0]
                    );
                }

                // 3. Max codes (all 0xFF bytes)
                {
                    let mut w = vec![0xFFu8; rb];
                    sane_scales(fmt, &mut w);
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} max codes n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                    let mut bgot = [0.0f32; 1];
                    // SAFETY: have_batched() checked.
                    unsafe { dot_batch(fmt, &w, std::slice::from_ref(&a), &mut bgot) };
                    assert_eq!(
                        bgot[0].to_bits(),
                        want.to_bits(),
                        "{:?} batched max codes n={n}: got {} vs scalar {want}",
                        fmt.label(),
                        bgot[0]
                    );
                }

                // 4. Min codes (all 0x00 bytes) with sane scales
                {
                    let mut w = vec![0x00u8; rb];
                    sane_scales(fmt, &mut w);
                    let x: Vec<f32> = (0..n)
                        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
                        .collect();
                    let mut a = QuantAct::with_capacity(n);
                    a.fill(&x);
                    // SAFETY: have_avx2() checked.
                    let got = unsafe { dot(fmt, &w, &a) };
                    let want = crate::quant::dot_scalar(fmt, &w, &a);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{:?} min codes n={n}: avx2 {got} vs scalar {want}",
                        fmt.label()
                    );
                    let mut bgot = [0.0f32; 1];
                    // SAFETY: have_batched() checked.
                    unsafe { dot_batch(fmt, &w, std::slice::from_ref(&a), &mut bgot) };
                    assert_eq!(
                        bgot[0].to_bits(),
                        want.to_bits(),
                        "{:?} batched min codes n={n}: got {} vs scalar {want}",
                        fmt.label(),
                        bgot[0]
                    );
                }
            }
        }
    }
}
