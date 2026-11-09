//! Vector transcendental math: `exp`, softmax and SwiGLU.
//!
//! Before this module every `exp` in the engine was a scalar libm call — one per
//! attention score, one per SwiGLU element, one per vocabulary entry when sampling.
//! At 4 096 positions that is ~34 µs per (head, layer) of softmax alone on this CPU,
//! more than the dot products it normalises, and it grows with context.
//!
//! The vector `exp` is ggml's (`ggml-cpu/vec.h`, `ggml_v_expf`, itself adapted from
//! Arm's optimized-routines): max error 1.45 ulp + 0.5, flushing above 88.38 to
//! infinity and below −103.97 to zero. Porting llama.cpp's own routine rather than
//! writing another polynomial is deliberate — llama.cpp is this engine's correctness
//! oracle, and matching its softmax and SwiGLU arithmetic keeps the two engines'
//! outputs as close as the rest of the pipeline allows. The softmax sum accumulates
//! each eight-wide chunk into an `f64`, as `ggml_vec_soft_max_f32` does.
//!
//! Every function has a scalar path with identical semantics (tails and non-AVX2
//! machines), and the tests pin the vector path against it.
//!
//! # The forward path does not call the C library
//!
//! The transcendentals the forward path needs are built here, from IEEE basic
//! operations alone: [`exp_scalar`] is the scalar twin of [`exp8`] (bit for bit), and
//! [`tanh`] is built on it. A libm call is not reproducible across C libraries — glibc
//! 2.40 replaced `tanhf` with a correctly rounded version, and the same kernel then
//! produced different activations, different logits and different golden files on
//! glibc 2.39 and 2.43 (issue #64). A transcendental built here is fixed by its
//! definition, not by the machine's libc, so the engine's output depends on the kernel
//! tier alone.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline]
fn vector_ok() -> bool {
    crate::cpu_has!("avx2") && crate::cpu_has!("fma")
}

/// Eight `exp`s at once — ggml's `ggml_v_expf`, transcribed.
///
/// # Safety
/// AVX2 and FMA must be available.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp8(x: __m256) -> __m256 {
    let r = _mm256_set1_ps(f32::from_bits(0x4b40_0000)); // 0x1.8p23
    let z = _mm256_fmadd_ps(x, _mm256_set1_ps(f32::from_bits(0x3fb8_aa3b)), r); // log2(e)
    let n = _mm256_sub_ps(z, r);
    let b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(f32::from_bits(0x35bf_be8e)), // 0x1.7f7d1cp-20
        _mm256_fnmadd_ps(n, _mm256_set1_ps(f32::from_bits(0x3f31_7200)), x), // 0x1.62e4p-1
    );
    let e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    let k = _mm256_castsi256_ps(_mm256_add_epi32(
        e,
        _mm256_castps_si256(_mm256_set1_ps(1.0)),
    ));
    let c = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
        _mm256_set1_ps(126.0),
        _CMP_GT_OQ,
    ));
    let u = _mm256_mul_ps(b, b);
    let j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32::from_bits(0x3c07_2010)),
                b,
                _mm256_set1_ps(f32::from_bits(0x3d2b_9f17)),
            ),
            u,
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32::from_bits(0x3e2a_af33)),
                b,
                _mm256_set1_ps(f32::from_bits(0x3eff_fedb)),
            ),
        ),
        u,
        _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(0x3f7f_fff6)), b),
    );
    if _mm256_movemask_ps(_mm256_castsi256_ps(c)) == 0 {
        return _mm256_fmadd_ps(j, k, k);
    }
    let g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x8200_0000u32 as i32),
    );
    let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f00_0000)));
    let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    let d = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
        _mm256_set1_ps(192.0),
        _CMP_GT_OQ,
    ));
    _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(
                    _mm256_castsi256_ps(c),
                    _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1),
                ),
                _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)),
            ),
        ),
    )
}

/// Sixteen `exp`s at once — the same polynomial as [`exp8`], lane for lane.
///
/// The constants, the FMA order and the denormal handling are identical to the
/// eight-lane routine, so lane `i` of the result has the same bits as [`exp8`]'s
/// lane `i` had it seen the same input. The softmax sum keeps the eight-lane
/// grouping on top (see `exp_shift_sum_avx512`), so widening only the bank
/// of exponentials cannot move a softmax denominator.
///
/// # Safety
/// AVX-512F and FMA must be available.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f,fma")]
unsafe fn exp16(x: __m512) -> __m512 {
    let r = _mm512_set1_ps(f32::from_bits(0x4b40_0000)); // 0x1.8p23
    let z = _mm512_fmadd_ps(x, _mm512_set1_ps(f32::from_bits(0x3fb8_aa3b)), r); // log2(e)
    let n = _mm512_sub_ps(z, r);
    let b = _mm512_fnmadd_ps(
        n,
        _mm512_set1_ps(f32::from_bits(0x35bf_be8e)), // 0x1.7f7d1cp-20
        _mm512_fnmadd_ps(n, _mm512_set1_ps(f32::from_bits(0x3f31_7200)), x), // 0x1.62e4p-1
    );
    let e = _mm512_slli_epi32(_mm512_castps_si512(z), 23);
    let k = _mm512_castsi512_ps(_mm512_add_epi32(
        e,
        _mm512_castps_si512(_mm512_set1_ps(1.0)),
    ));
    let c = _mm512_cmp_ps_mask(
        _mm512_andnot_ps(_mm512_set1_ps(-0.0), n),
        _mm512_set1_ps(126.0),
        _CMP_GT_OQ,
    );
    let u = _mm512_mul_ps(b, b);
    let j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(
            _mm512_fmadd_ps(
                _mm512_set1_ps(f32::from_bits(0x3c07_2010)),
                b,
                _mm512_set1_ps(f32::from_bits(0x3d2b_9f17)),
            ),
            u,
            _mm512_fmadd_ps(
                _mm512_set1_ps(f32::from_bits(0x3e2a_af33)),
                b,
                _mm512_set1_ps(f32::from_bits(0x3eff_fedb)),
            ),
        ),
        u,
        _mm512_mul_ps(_mm512_set1_ps(f32::from_bits(0x3f7f_fff6)), b),
    );
    if c == 0 {
        return _mm512_fmadd_ps(j, k, k);
    }
    let c0 = _mm512_cmp_ps_mask(n, _mm512_setzero_ps(), _CMP_LE_OQ);
    let g = _mm512_and_si512(
        _mm512_maskz_set1_epi32(c0, -1),
        _mm512_set1_epi32(0x8200_0000u32 as i32),
    );
    let s1 = _mm512_castsi512_ps(_mm512_add_epi32(g, _mm512_set1_epi32(0x7f00_0000)));
    let s2 = _mm512_castsi512_ps(_mm512_sub_epi32(e, g));
    let d = _mm512_cmp_ps_mask(
        _mm512_andnot_ps(_mm512_set1_ps(-0.0), n),
        _mm512_set1_ps(192.0),
        _CMP_GT_OQ,
    );
    let cm = _mm512_maskz_set1_epi32(c, -1);
    let dm = _mm512_maskz_set1_epi32(d, -1);
    _mm512_or_ps(
        _mm512_castsi512_ps(_mm512_and_si512(
            dm,
            _mm512_castps_si512(_mm512_mul_ps(s1, s1)),
        )),
        _mm512_castsi512_ps(_mm512_andnot_si512(
            dm,
            _mm512_or_si512(
                _mm512_and_si512(
                    cm,
                    _mm512_castps_si512(_mm512_mul_ps(_mm512_fmadd_ps(s2, j, s2), s1)),
                ),
                _mm512_andnot_si512(cm, _mm512_castps_si512(_mm512_fmadd_ps(k, j, k))),
            ),
        )),
    )
}

/// `exp8`'s magic constant: `0x1.8p23`. Adding it to `x·log2(e)` keeps the integer part of
/// `x·log2(e)` in the low bits of the sum's mantissa, which is where `2ⁿ` is read from.
const EXP_BIAS: f32 = f32::from_bits(0x4b40_0000);
/// `log2(e)`.
const EXP_LOG2E: f32 = f32::from_bits(0x3fb8_aa3b);
/// `ln 2`, high part (`0x1.62e4p-1`).
const EXP_LN2_HI: f32 = f32::from_bits(0x3f31_7200);
/// `ln 2`, low part (`0x1.7f7d1cp-20`).
const EXP_LN2_LO: f32 = f32::from_bits(0x35bf_be8e);
/// `exp8`'s polynomial: the coefficient of `b⁵`.
const EXP_P0: f32 = f32::from_bits(0x3c07_2010);
/// The coefficient of `b⁴`.
const EXP_P1: f32 = f32::from_bits(0x3d2b_9f17);
/// The coefficient of `b³`.
const EXP_P2: f32 = f32::from_bits(0x3e2a_af33);
/// The coefficient of `b²`.
const EXP_P3: f32 = f32::from_bits(0x3eff_fedb);
/// The coefficient of `b`.
const EXP_P4: f32 = f32::from_bits(0x3f7f_fff6);

/// The pieces of [`exp8`]'s argument reduction: `n` (the exponent, `round(x·log2 e)`), `j`
/// (the polynomial, `exp(b) − 1` for `b = x − n·ln 2`) and `e` (the bit pattern `2ⁿ` is
/// built from).
#[derive(Clone, Copy)]
struct ExpParts {
    /// `round(x·log2 e)`, exactly: the sum that carries it is exact.
    n: f32,
    /// `exp(b) − 1`, to the polynomial's accuracy.
    j: f32,
    /// `n` shifted into the exponent field; `2ⁿ` is this plus the exponent bias.
    e: u32,
}

/// [`exp8`]'s range reduction and polynomial, in scalar: the same constants, the same
/// operation order, the same fused multiply-adds.
///
/// `#[inline(always)]`, so that the FMA-targeted wrappers below (`tanh_fma`, `geglu_fma`)
/// compile their `mul_add`s to the hardware instruction instead of a call to `fmaf`: the
/// inlined body inherits the caller's target features.
#[inline(always)]
fn exp_parts(x: f32) -> ExpParts {
    // `z = x·log2(e) + 0x1.8p23`; the sum's low mantissa bits hold `x·log2(e)`'s integer
    // part, and `n = z − 0x1.8p23` recovers it exactly (the subtraction is exact).
    let z = x.mul_add(EXP_LOG2E, EXP_BIAS);
    let n = z - EXP_BIAS;
    // `b = x − n·ln 2`, with `ln 2` split in two so that the products stay exact enough.
    let t = (-n).mul_add(EXP_LN2_HI, x);
    let b = (-n).mul_add(EXP_LN2_LO, t);
    // `e`: the low bits of `z`'s mantissa are `n` modulo 512, and the exponent field of a
    // float is bits 23..30 — hence the shift by 23.
    let e = z.to_bits() << 23;
    // The polynomial `j ≈ exp(b) − 1`, in `u = b²`.
    let u = b * b;
    let a = EXP_P0.mul_add(b, EXP_P1);
    let c = EXP_P2.mul_add(b, EXP_P3);
    let d = a.mul_add(u, c);
    let j = d.mul_add(u, EXP_P4 * b);
    ExpParts { n, j, e }
}

/// `|x| > limit`, with `NaN` answering false — the comparison the vector masks make
/// (`_CMP_GT_OQ` is false for a `NaN`).
#[inline]
fn greater(x: f32, limit: f32) -> bool {
    matches!(
        x.abs().partial_cmp(&limit),
        Some(std::cmp::Ordering::Greater)
    )
}

/// `exp` — the scalar twin of [`exp8`]: the same constants, the same operation order and the
/// same overflow and underflow handling, so the two agree bit for bit on every input
/// (the test sweeps them against each other).
///
/// It is built from IEEE basic operations alone, so unlike `f32::exp` its result does not
/// depend on the C library — glibc 2.40 replaced `expf`'s neighbours with correctly rounded
/// versions, and the engine's outputs must not move when the host's libc does (issue #64).
///
/// The fused multiply-adds are the IEEE-754 fused operation: the hardware instruction where
/// the target has one, a correctly rounded `fmaf` where it does not. Either way the result
/// is the one the standard prescribes, so it does not vary with the C library either.
///
/// `#[inline(always)]`, so that the FMA-targeted wrapper (`tanh_fma`) compiles these
/// `mul_add`s to the hardware instruction.
#[inline(always)]
pub fn exp_scalar(x: f32) -> f32 {
    let p = exp_parts(x);
    // `2ⁿ`, from `n` in the exponent field.
    let k = f32::from_bits(p.e.wrapping_add(0x3f80_0000));
    if !greater(p.n, 126.0) {
        // `2ⁿ·(1 + j)`, the ordinary path.
        return k.mul_add(p.j, k);
    }
    // `|n| > 126`: `2ⁿ` itself is out of range, so the exponent is carried by the scaling
    // factors instead. This is `exp8`'s masked path, one lane at a time.
    let g: u32 = if p.n <= 0.0 { 0x8200_0000 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(p.e.wrapping_sub(g));
    if greater(p.n, 192.0) {
        // Underflow to zero below, overflow to infinity above.
        s1 * s1
    } else {
        s2.mul_add(p.j, s2) * s1
    }
}

/// The largest `|x|` [`tanh`] handles with [`exp_scalar`]: from `|x| ≈ 8.96` the rounding of
/// `1 − 2/(e^{2x} + 1)` is already `1.0`, so the branch saturates there and `exp` is never
/// asked for an argument it would flush.
const TANH_SATURATE: f32 = 9.0;

/// Where [`tanh`]'s polynomial hands over to [`exp_scalar`]. Below it the polynomial is the
/// more accurate of the two, above it `exp`'s own error is.
const TANH_POLY_LIMIT: f32 = 0.55;

/// `tanh(x)/x` as a polynomial in `x²`, for `|x| ≤ 0.55`.
///
/// A least-squares fit on 600 Chebyshev nodes with a relative-error criterion (the `f64`
/// fit's own error is 9.6e-10, well under the `f32` evaluation's rounding), rounded to
/// `f32`.
const TANH_POLY: [f32; 6] = [
    f32::from_bits(0x3f80_0000), // 1
    f32::from_bits(0xbeaa_aaa3), // −1/3
    f32::from_bits(0x3e08_8637), // 2/15
    f32::from_bits(0xbd5c_8907), // −17/315
    f32::from_bits(0x3cac_52cd), // 62/2835
    f32::from_bits(0xbbcc_2668), // −1382/155925
];

/// `tanh(x)` for `|x| ≤ 0.55`: the polynomial above, times `x`.
///
/// `#[inline(always)]`, so that the FMA-targeted wrapper (`tanh_fma`) compiles these
/// `mul_add`s to the hardware instruction.
#[inline(always)]
fn tanh_poly(x: f32) -> f32 {
    let u = x * x;
    let p = TANH_POLY[5].mul_add(u, TANH_POLY[4]);
    let p = p.mul_add(u, TANH_POLY[3]);
    let p = p.mul_add(u, TANH_POLY[2]);
    let p = p.mul_add(u, TANH_POLY[1]);
    let p = p.mul_add(u, TANH_POLY[0]);
    x * p
}

/// `tanh` — deterministic, and built only from IEEE basic operations.
///
/// `f32::tanh` is the C library's `tanhf`, and glibc 2.40 replaced it with a correctly
/// rounded version: the same model then produced different activations and different logits
/// on glibc 2.39 and 2.43, and the Gemma 3 goldens could not satisfy both (issue #64).
/// This one is fixed by its definition instead of by the host's libc.
///
/// * `|x| ≤ 0.55`: an odd polynomial in `x²`, which keeps the relative accuracy where
///   `tanh(x) ≈ x` and a cancellation would lose it.
/// * `0.55 < |x| < 9`: `1 − 2/(e^{2|x|} + 1)`, on [`exp_scalar`]. The subtraction cancels
///   only where the result is not small, and the sign is restored afterwards, so the
///   function is exactly odd.
/// * `|x| ≥ 9`: `±1`, where `f32` rounds `tanh` to one anyway.
///
/// # Accuracy
///
/// Measured against the `f64` reference `tanh` over a dense sweep of the `f32` range
/// (−12…12 in steps of 1.3e-4, plus every 997th `f32` in [0, 1)): **max 1.72 ulp**, at
/// `|x| ≈ 0.61`, where `exp`'s own error is the largest. The two branches differ by
/// 1.3 ulp where they meet, and `tanh` is odd to the bit.
///
/// # The FMA path
///
/// On a CPU with the FMA instruction the same arithmetic runs inside a
/// `#[target_feature(enable = "fma")]` function (`tanh_fma`), where the `mul_add`s
/// become `vfmadd` instead of a call to the C library's `fmaf`. Both are the IEEE fused
/// multiply-add, so the result is the same bit for bit — the instruction only removes
/// the call. The tier cap (`PEREGRINE_ISA`) decides: `scalar` keeps the portable path.
pub fn tanh(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::have_fma() {
            // SAFETY: FMA checked.
            return unsafe { tanh_fma(x) };
        }
    }
    tanh_scalar(x)
}

/// The portable scalar `tanh`: the arithmetic [`tanh`] describes.
///
/// `#[inline(always)]`, so that the FMA-targeted wrappers (`tanh_fma`, `geglu_fma`)
/// compile the `mul_add`s to the hardware instruction: an inlined body inherits the
/// caller's target features.
#[inline(always)]
fn tanh_scalar(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let a = x.abs();
    let t = if a >= TANH_SATURATE {
        1.0
    } else if a <= TANH_POLY_LIMIT {
        tanh_poly(a)
    } else {
        // `tanh(x) = 1 − 2/(e^{2x} + 1)`.
        let e = exp_scalar(2.0 * a);
        1.0 - 2.0 / (e + 1.0)
    };
    if x.is_sign_negative() {
        -t
    } else {
        t
    }
}

/// [`tanh_scalar`] compiled for a CPU with the FMA instruction.
///
/// # Safety
///
/// FMA must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn tanh_fma(x: f32) -> f32 {
    tanh_scalar(x)
}

/// `x[i] = exp(x[i] − shift)` in place; returns `Σ x[i]` after the update.
///
/// The softmax numerator and normaliser in one pass. The caller divides.
pub fn exp_shift_sum(x: &mut [f32], shift: f32) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            if crate::cpu_has!("avx512f") {
                // SAFETY: features checked.
                return unsafe { exp_shift_sum_avx512(x, shift) };
            }
            // SAFETY: features checked.
            return unsafe { exp_shift_sum_avx2(x, shift) };
        }
    }
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = (*v - shift).exp();
        sum += *v as f64;
    }
    sum
}

/// Sixteen lanes per pass, but the accumulator keeps [`exp_shift_sum_avx2`]'s
/// eight-value grouping: each half is horizontally summed with the same
/// [`crate::simd::hsum_ps`] and added to the `f64` in the same order. `exp16` is
/// `exp8` lane for lane, so the denominator is bit-identical to the AVX2 tier's.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn exp_shift_sum_avx512(x: &mut [f32], shift: f32) -> f64 {
    let s = _mm512_set1_ps(shift);
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 16 <= x.len() {
        let v = exp16(_mm512_sub_ps(_mm512_loadu_ps(x.as_ptr().add(i)), s));
        _mm512_storeu_ps(x.as_mut_ptr().add(i), v);
        sum += crate::simd::hsum_ps(_mm512_castps512_ps256(v)) as f64;
        sum += crate::simd::hsum_ps(_mm512_extractf32x8_ps(v, 1)) as f64;
        i += 16;
    }
    // Keep the AVX2 tier's eight-value grouping for the last block too.
    if i + 8 <= x.len() {
        let v = exp8(_mm256_sub_ps(
            _mm256_loadu_ps(x.as_ptr().add(i)),
            _mm512_castps512_ps256(s),
        ));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), v);
        sum += crate::simd::hsum_ps(v) as f64;
        i += 8;
    }
    while i < x.len() {
        x[i] = (x[i] - shift).exp();
        sum += x[i] as f64;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_shift_sum_avx2(x: &mut [f32], shift: f32) -> f64 {
    let s = _mm256_set1_ps(shift);
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 8 <= x.len() {
        let v = exp8(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), s));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), v);
        sum += crate::simd::hsum_ps(v) as f64;
        i += 8;
    }
    while i < x.len() {
        x[i] = (x[i] - shift).exp();
        sum += x[i] as f64;
        i += 1;
    }
    sum
}

/// `x[i] = exp((x[i] − shift) · scale)` in place; returns the sum. Sampling's
/// temperature form: `scale = 1/T`.
pub fn exp_scaled_sum(x: &mut [f32], shift: f32, scale: f32) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: features checked.
            return unsafe { exp_scaled_sum_avx2(x, shift, scale) };
        }
    }
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = ((*v - shift) * scale).exp();
        sum += *v as f64;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_scaled_sum_avx2(x: &mut [f32], shift: f32, scale: f32) -> f64 {
    let s = _mm256_set1_ps(shift);
    let k = _mm256_set1_ps(scale);
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 8 <= x.len() {
        let v = exp8(_mm256_mul_ps(
            _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), s),
            k,
        ));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), v);
        sum += crate::simd::hsum_ps(v) as f64;
        i += 8;
    }
    while i < x.len() {
        x[i] = ((x[i] - shift) * scale).exp();
        sum += x[i] as f64;
        i += 1;
    }
    sum
}

/// The largest element (`−∞` for an empty slice).
pub fn max(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() && x.len() >= 16 {
            if crate::cpu_has!("avx512f") {
                // SAFETY: features checked.
                return unsafe { max_avx512(x) };
            }
            // SAFETY: features checked.
            return unsafe { max_avx2(x) };
        }
    }
    x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Sixteen lanes per pass. Max is order-free, so the result is the AVX2 tier's.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn max_avx512(x: &[f32]) -> f32 {
    let mut m = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 16 <= x.len() {
        m = _mm512_max_ps(m, _mm512_loadu_ps(x.as_ptr().add(i)));
        i += 16;
    }
    let mut lanes = [0.0f32; 16];
    _mm512_storeu_ps(lanes.as_mut_ptr(), m);
    let mut r = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    while i < x.len() {
        r = r.max(x[i]);
        i += 1;
    }
    r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn max_avx2(x: &[f32]) -> f32 {
    let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= x.len() {
        m = _mm256_max_ps(m, _mm256_loadu_ps(x.as_ptr().add(i)));
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), m);
    let mut r = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    while i < x.len() {
        r = r.max(x[i]);
        i += 1;
    }
    r
}

/// SwiGLU in place: `gate[i] = silu(gate[i]) · up[i]`, `silu(x) = x / (1 + e^−x)` —
/// ggml's `ggml_vec_swiglu_f32`, so the feed-forward activation matches llama.cpp's.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: features checked; lengths equal.
            return unsafe { swiglu_avx2(gate, up) };
        }
    }
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = (*g / (1.0 + (-*g).exp())) * u;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_avx2(gate: &mut [f32], up: &[f32]) {
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= gate.len() {
        let x = _mm256_loadu_ps(gate.as_ptr().add(i));
        let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
        let silu = _mm256_div_ps(x, _mm256_add_ps(one, e));
        _mm256_storeu_ps(
            gate.as_mut_ptr().add(i),
            _mm256_mul_ps(silu, _mm256_loadu_ps(up.as_ptr().add(i))),
        );
        i += 8;
    }
    // The tail goes through `exp8` too, in a padded vector, rather than through the
    // scalar path: a caller that splits a row across workers must get the same bits
    // for an element however the row was cut. `libm`'s `expf` and `exp8` differ, so
    // the old scalar tail made a row's result depend on where the split fell (and
    // the one-row path in `swiglu_rows` splits a row into unequal shares).
    if i < gate.len() {
        let rem = gate.len() - i;
        let mut g = [0.0f32; 8];
        let mut u = [0.0f32; 8];
        g[..rem].copy_from_slice(&gate[i..]);
        u[..rem].copy_from_slice(&up[i..]);
        let x = _mm256_loadu_ps(g.as_ptr());
        let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
        let silu = _mm256_div_ps(x, _mm256_add_ps(one, e));
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(
            out.as_mut_ptr(),
            _mm256_mul_ps(silu, _mm256_loadu_ps(u.as_ptr())),
        );
        gate[i..].copy_from_slice(&out[..rem]);
    }
}

/// OpenAI's `swiglu_oai`, the gated activation gpt-oss's experts use.
///
/// `x = min(gate, limit)`, `y = clamp(up, −limit, limit)` and
/// `out = x / (1 + e^(α·(−x))) · (y + 1)`.
///
/// llama.cpp v0.9.0 computes it in `ggml_compute_forward_swiglu_oai_f32`
/// (`ggml/src/ggml-cpu/ops.cpp:3388-3450`): the gate is clamped *above* only
/// (`std::min`), the up is clamped on both sides, and the exponential is the
/// library's `expf`. This routine uses the engine's own `exp` — ggml's vector
/// polynomial, whose scalar twin is bit for bit the same — so the result does
/// not move when the host's C library does (issue #64), and the vector and the
/// scalar paths agree bit for bit.
pub fn swiglu_oai(gate: &mut [f32], up: &[f32], alpha: f32, limit: f32) {
    debug_assert_eq!(gate.len(), up.len());
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: features checked; lengths equal.
            return unsafe { swiglu_oai_avx2(gate, up, alpha, limit) };
        }
    }
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = swiglu_oai_one(*g, u, alpha, limit);
    }
}

/// One element of [`swiglu_oai`], in the order llama.cpp's kernel uses.
#[inline(always)]
fn swiglu_oai_one(g: f32, u: f32, alpha: f32, limit: f32) -> f32 {
    // `std::min(gate, limit)`: the limit wins only when it is the smaller one.
    let x = if limit < g { limit } else { g };
    // `std::clamp(up, -limit, limit)`.
    let y = if u < -limit {
        -limit
    } else if limit < u {
        limit
    } else {
        u
    };
    let glu = x / (1.0 + exp_scalar(alpha * -x));
    glu * (y + 1.0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_oai_avx2(gate: &mut [f32], up: &[f32], alpha: f32, limit: f32) {
    let one = _mm256_set1_ps(1.0);
    let av = _mm256_set1_ps(alpha);
    let lv = _mm256_set1_ps(limit);
    let nlv = _mm256_set1_ps(-limit);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= gate.len() {
        let g = _mm256_loadu_ps(gate.as_ptr().add(i));
        let u = _mm256_loadu_ps(up.as_ptr().add(i));
        // `min(gate, limit)` and `clamp(up, -limit, limit)`, with the operands in
        // the order that keeps the NaN behaviour of the scalar path.
        let x = _mm256_min_ps(lv, g);
        let y = _mm256_max_ps(nlv, _mm256_min_ps(lv, u));
        let e = exp8(_mm256_mul_ps(av, _mm256_sub_ps(zero, x)));
        let glu = _mm256_div_ps(x, _mm256_add_ps(one, e));
        _mm256_storeu_ps(
            gate.as_mut_ptr().add(i),
            _mm256_mul_ps(glu, _mm256_add_ps(y, one)),
        );
        i += 8;
    }
    while i < gate.len() {
        let g = gate[i];
        gate[i] = swiglu_oai_one(g, up[i], alpha, limit);
        i += 1;
    }
}

const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876;
const GELU_COEF_A: f32 = 0.044715;
/// GELU (tanh approximation):
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
/// Matching ggml's `ggml_gelu_f32`, with [`tanh`] instead of the C library's `tanhf`:
/// ggml's own `tanhf` moves with the host's libc (issue #64), and this engine's outputs
/// must not.
///
/// On a CPU with the FMA instruction the same arithmetic runs inside a
/// `#[target_feature(enable = "fma")]` function (`gelu_tanh_fma`); the result is the
/// same bit for bit (see [`tanh`]).
pub fn gelu_tanh(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::have_fma() {
            // SAFETY: FMA checked.
            return unsafe { gelu_tanh_fma(x) };
        }
    }
    gelu_tanh_scalar(x)
}

/// The portable scalar GELU: the arithmetic [`gelu_tanh`] describes.
///
/// `#[inline(always)]`, so that the FMA-targeted wrappers (`gelu_tanh_fma`,
/// `geglu_fma`) compile the `mul_add`s of [`tanh_scalar`] to the hardware instruction.
#[inline(always)]
fn gelu_tanh_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + tanh_scalar(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)))
}

/// [`gelu_tanh_scalar`] compiled for a CPU with the FMA instruction.
///
/// # Safety
///
/// FMA must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn gelu_tanh_fma(x: f32) -> f32 {
    gelu_tanh_scalar(x)
}

/// GeGLU in place: `gate[i] = gelu_tanh(gate[i]) · up[i]` — matching llama.cpp's `LLM_FFN_GELU`.
///
/// Scalar, and deliberately so: it has no vector path, so there is no second arithmetic to
/// keep in step with [`gelu_tanh`]'s. A vector GeGLU may only be added if it evaluates the
/// same expressions lane-wise, so that it is bit for bit the scalar result.
///
/// # The FMA path
///
/// The scalar arithmetic is compiled twice: once for a plain x86-64 target, where every
/// `mul_add` of [`tanh`] is a call to the C library's `fmaf`, and once inside a
/// `#[target_feature(enable = "fma")]` function (`geglu_fma`), where they are the
/// hardware's `vfmadd`. The choice is made once per call, from the CPU's features under
/// the tier cap (`PEREGRINE_ISA`; `scalar` keeps the portable path).
///
/// The two are the same arithmetic: an IEEE fused multiply-add has one result, and both
/// the instruction and the library's `fmaf` produce it. So the GeGLU output does not
/// depend on which path ran — the test pins that over a dense sweep.
pub fn geglu(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::have_fma() {
            // SAFETY: FMA checked.
            return unsafe { geglu_fma(gate, up) };
        }
    }
    geglu_scalar(gate, up);
}

/// The GeGLU loop, written once: the portable path, and the body of the FMA path.
///
/// `#[inline(always)]`, so that `geglu_fma` compiles the `mul_add`s of [`tanh_scalar`]
/// to the hardware instruction: an inlined body inherits the caller's target features.
#[inline(always)]
fn geglu_scalar(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = gelu_tanh_scalar(*g) * u;
    }
}

/// The same loop, compiled for a CPU with the FMA instruction.
///
/// # Safety
///
/// FMA must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn geglu_fma(gate: &mut [f32], up: &[f32]) {
    geglu_scalar(gate, up);
}

// `Σ x²` accumulated in f64, four lanes at a time (the norms' sums; was a serial f64 loop).
pub fn sumsq_f64(x: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() && x.len() % 8 == 0 {
            // SAFETY: features checked; length a multiple of 8.
            return unsafe { sumsq_f64_avx2(x) };
        }
    }
    x.iter().map(|v| (*v as f64) * (*v as f64)).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sumsq_f64_avx2(x: &[f32]) -> f64 {
    let (mut a, mut b) = (_mm256_setzero_pd(), _mm256_setzero_pd());
    let mut i = 0;
    while i < x.len() {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
        let hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
        a = _mm256_fmadd_pd(lo, lo, a);
        b = _mm256_fmadd_pd(hi, hi, b);
        i += 8;
    }
    let s = _mm256_add_pd(a, b);
    let mut t = [0.0f64; 4];
    _mm256_storeu_pd(t.as_mut_ptr(), s);
    t[0] + t[1] + t[2] + t[3]
}

/// `Σ a·b` in f32, eight lanes (two accumulators).
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() && a.len() % 16 == 0 {
            // SAFETY: features checked; equal lengths, a multiple of 16.
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0;
    while i < a.len() {
        s0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            s0,
        );
        s1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            s1,
        );
        i += 16;
    }
    crate::simd::hsum_ps(_mm256_add_ps(s0, s1))
}

/// `x[i] *= sigmoid(z[i])`.
pub fn mul_sigmoid(x: &mut [f32], z: &[f32]) {
    debug_assert_eq!(x.len(), z.len());
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: features checked; equal lengths.
            return unsafe { mul_sigmoid_avx2(x, z) };
        }
    }
    for (v, &g) in x.iter_mut().zip(z) {
        *v /= 1.0 + (-g).exp();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_sigmoid_avx2(x: &mut [f32], z: &[f32]) {
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= x.len() {
        let e = exp8(_mm256_sub_ps(
            _mm256_setzero_ps(),
            _mm256_loadu_ps(z.as_ptr().add(i)),
        ));
        let v = _mm256_div_ps(_mm256_loadu_ps(x.as_ptr().add(i)), _mm256_add_ps(one, e));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), v);
        i += 8;
    }
    while i < x.len() {
        x[i] /= 1.0 + (-z[i]).exp();
        i += 1;
    }
}

// `x[i] = silu(x[i])`, and `o[i] *= silu(z[i])`, on the same `exp8` as `swiglu`.
pub fn silu(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: features checked.
            return unsafe { mul_silu_avx2(x, None) };
        }
    }
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

pub fn mul_silu(o: &mut [f32], z: &[f32]) {
    debug_assert_eq!(o.len(), z.len());
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: features checked; lengths equal.
            return unsafe { mul_silu_avx2(o, Some(z)) };
        }
    }
    for (v, &g) in o.iter_mut().zip(z) {
        *v *= g / (1.0 + (-g).exp());
    }
}

/// With `z`: `x[i] *= silu(z[i])`; without: `x[i] = silu(x[i])`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_silu_avx2(x: &mut [f32], z: Option<&[f32]>) {
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= x.len() {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let g = match z {
            Some(z) => _mm256_loadu_ps(z.as_ptr().add(i)),
            None => xv,
        };
        let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), g));
        let s = _mm256_div_ps(g, _mm256_add_ps(one, e));
        let r = if z.is_some() { _mm256_mul_ps(xv, s) } else { s };
        _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
        i += 8;
    }
    while i < x.len() {
        let g = z.map_or(x[i], |z| z[i]);
        let s = g / (1.0 + (-g).exp());
        x[i] = if z.is_some() { x[i] * s } else { s };
        i += 1;
    }
}

/// In-place orthonormal Walsh–Hadamard transform with a fixed sign flip first
/// (`x ← H·D·x / √n`). `n` must be a power of two. Its own inverse is
/// [`hadamard_inv`] (`x ← D·H·x / √n`), because `H/√n` is symmetric orthonormal and
/// `D` is a ±1 diagonal.
///
/// This is the rotation uzu applies before quantising its int8 KV cache
/// (`backends/cpu/kernel/attention/prepare_int8.rs`): it spreads an outlier channel
/// over every channel of the vector, so one per-vector absmax scale no longer has to
/// stretch to cover it, and the other channels keep their resolution. Dot products
/// are unchanged by an orthonormal rotation, so `q·k` can be taken entirely in the
/// rotated space once `q` is rotated the same way.
pub fn hadamard(x: &mut [f32]) {
    debug_assert!(x.len().is_power_of_two());
    for (i, v) in x.iter_mut().enumerate() {
        if sign_bit(i) {
            *v = -*v;
        }
    }
    fwht(x);
}

/// The inverse of [`hadamard`].
pub fn hadamard_inv(x: &mut [f32]) {
    debug_assert!(x.len().is_power_of_two());
    fwht(x);
    for (i, v) in x.iter_mut().enumerate() {
        if sign_bit(i) {
            *v = -*v;
        }
    }
}

/// A fixed pseudo-random sign pattern (a 64-bit constant, repeated): any pattern
/// that is not structured along the channel axis does the job, and a fixed one keeps
/// the cache deterministic across runs.
#[inline]
fn sign_bit(i: usize) -> bool {
    const SIGNS: u64 = 0x9E37_79B9_7F4A_7C15;
    (SIGNS >> (i & 63)) & 1 == 1
}

fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prng(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
            })
            .collect()
    }

    /// The gap between `x` and the next `f32` above it: the error's unit.
    fn ulp(x: f32) -> f64 {
        let a = x.abs();
        if a == 0.0 {
            // The smallest subnormal.
            return f64::from(f32::from_bits(1));
        }
        f64::from(f32::from_bits(a.to_bits() + 1) - a)
    }

    /// The error of [`tanh`] against the `f64` reference, in ulp of the result.
    fn tanh_err(x: f32) -> f64 {
        let got = tanh(x);
        let want = (x as f64).tanh();
        (got as f64 - want).abs() / ulp(got)
    }

    /// Eight `exp`s through the AVX2 path.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp8_lanes(x: &[f32; 8]) -> [f32; 8] {
        let v = exp8(_mm256_loadu_ps(x.as_ptr()));
        let mut out = [0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), v);
        out
    }

    /// The scalar twin must be the vector routine, bit for bit: it is the same arithmetic
    /// written once, and the two have to agree on every input, including the ones that
    /// overflow, underflow and land on a NaN.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp_scalar_is_exp8_bit_for_bit() {
        if !vector_ok() {
            eprintln!("skipped: no AVX2 and FMA");
            return;
        }
        let mut xs: Vec<f32> = Vec::new();
        let mut x = -120.0f32;
        while x <= 120.0 {
            xs.push(x);
            x += 0.001_7;
        }
        for v in [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            87.0,
            88.0,
            88.5,
            89.0,
            100.0,
            -87.0,
            -88.0,
            -103.0,
            -104.0,
            -104.5,
            -200.0,
            126.0,
            -126.0,
            -127.0,
            -193.0,
            193.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            xs.push(v);
        }
        let mut lane = [0f32; 8];
        for chunk in xs.chunks(8) {
            lane[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: AVX2 and FMA are available (checked above).
            let got = unsafe { exp8_lanes(&lane) };
            for (i, &x) in chunk.iter().enumerate() {
                assert_eq!(
                    got[i].to_bits(),
                    exp_scalar(x).to_bits(),
                    "exp({x}): vector {got:?}, scalar {}",
                    exp_scalar(x)
                );
            }
        }
    }

    /// The sixteen-lane `exp` must be the eight-lane one, lane for lane: same
    /// constants, same FMA order. Only the bank width changes.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp16_is_exp8_bit_for_bit() {
        if !crate::cpu_has!("avx512f") || !vector_ok() {
            eprintln!("skipped: no AVX-512F or AVX2+FMA");
            return;
        }
        let xs = prng(17, 16_000, -120.0, 120.0);
        for chunk in xs.chunks(16) {
            let mut lane = [0f32; 16];
            lane[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: AVX2, FMA and AVX-512F are available (checked above).
            let got = unsafe {
                let v = exp16(_mm512_loadu_ps(lane.as_ptr()));
                let mut out = [0f32; 16];
                _mm512_storeu_ps(out.as_mut_ptr(), v);
                out
            };
            for (i, &x) in chunk.iter().enumerate() {
                assert_eq!(got[i].to_bits(), exp_scalar(x).to_bits(), "exp({x})");
            }
        }
    }

    /// The widened softmax pass keeps the eight-lane sum grouping, so its
    /// denominator is the AVX2 tier's bit for bit — the widening alone cannot move
    /// a golden.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp_shift_sum_avx512_matches_avx2_bit_for_bit() {
        if !crate::cpu_has!("avx512f") || !vector_ok() {
            eprintln!("skipped: no AVX-512F or AVX2+FMA");
            return;
        }
        for n in [1usize, 7, 8, 15, 16, 17, 31, 64, 1001] {
            let xs = prng(19 + n as u64, n, -50.0, 50.0);
            let mut a = xs.clone();
            let mut b = xs;
            // SAFETY: features checked above.
            let (sa, sb) = unsafe {
                (
                    exp_shift_sum_avx2(&mut a, 3.25),
                    exp_shift_sum_avx512(&mut b, 3.25),
                )
            };
            assert_eq!(sa.to_bits(), sb.to_bits(), "sum at n {n}");
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "values at n {n}"
            );
        }
    }

    /// The twin is also an `exp`: the vector routine's accuracy is ggml's, and the scalar
    /// one inherits it.
    #[test]
    fn exp_scalar_is_within_two_ulp_of_libm() {
        let xs = prng(13, 4099, -90.0, 20.0);
        for x in xs {
            let want = x.exp();
            let got = exp_scalar(x);
            let err = (got as f64 - want as f64).abs();
            let u = err / ulp(got);
            assert!(u <= 2.0, "exp({x}) = {got}, libm {want}, {u} ulp");
        }
    }

    /// The accuracy the doc comment claims, measured: max 1.72 ulp over the sweep.
    #[test]
    fn tanh_is_within_a_few_ulp_of_the_f64_reference() {
        let mut worst = 0.0f64;
        let mut worst_x = 0.0f32;
        let mut x = -12.0f32;
        while x <= 12.0 {
            let e = tanh_err(x);
            if e > worst {
                worst = e;
                worst_x = x;
            }
            x += 0.000_13;
        }
        // Every 997th `f32` in [0, 1): the polynomial branch, where the relative error is
        // what matters.
        let mut bits = 0u32;
        while bits < 0x3f80_0000 {
            let x = f32::from_bits(bits);
            let e = tanh_err(x);
            if e > worst {
                worst = e;
                worst_x = x;
            }
            bits += 997;
        }
        assert!(
            worst <= 3.0,
            "tanh's max error is {worst:.2} ulp at x={worst_x}"
        );
        eprintln!("tanh: max error {worst:.2} ulp at x={worst_x:e}");
    }

    /// The sign is restored, not computed: `tanh` is odd to the bit.
    #[test]
    fn tanh_is_odd_to_the_bit() {
        let mut x = -12.0f32;
        while x <= 12.0 {
            assert_eq!(
                tanh(-x).to_bits(),
                (-tanh(x)).to_bits(),
                "tanh({x}) is not odd"
            );
            x += 0.000_7;
        }
        for x in [
            1e-30f32,
            1e-10,
            0.5,
            3.0,
            9.0,
            1e10,
            f32::MAX,
            f32::INFINITY,
        ] {
            assert_eq!(tanh(-x).to_bits(), (-tanh(x)).to_bits(), "tanh({x})");
        }
    }

    /// `tanh` saturates, and never produces a NaN or an infinity, whatever it is given.
    #[test]
    fn tanh_saturates_and_stays_finite() {
        for x in [9.0f32, 10.0, 20.0, 1e10, 1e30, f32::MAX, f32::INFINITY] {
            assert_eq!(tanh(x), 1.0, "tanh({x})");
            assert_eq!(tanh(-x), -1.0, "tanh(-{x})");
        }
        for x in [1e-30f32, 1e-20, 1e-10] {
            assert_eq!(tanh(x), x, "tanh({x})");
            assert_eq!(tanh(-x), -x, "tanh(-{x})");
        }
        assert!(tanh(f32::NAN).is_nan());
        for x in [
            -1e30f32, -1e10, -20.0, -9.0, -1.0, -0.5, 0.0, 0.5, 1.0, 9.0, 20.0, 1e10, 1e30,
        ] {
            assert!(tanh(x).is_finite(), "tanh({x}) is not finite");
        }
    }

    /// GELU is the reason `tanh` exists here: it must stay finite for every input the
    /// feed-forward can produce, and it must grow with the activation on the positive side.
    #[test]
    fn gelu_tanh_is_finite_and_monotone_above_zero() {
        let mut x = -20.0f32;
        let mut last = gelu_tanh(x);
        while x <= 20.0 {
            let g = gelu_tanh(x);
            assert!(g.is_finite(), "gelu_tanh({x}) is {g}");
            if x >= 0.0 {
                assert!(g >= last, "gelu_tanh({x}) = {g} is below {last}");
            }
            last = g;
            x += 0.001;
        }
    }

    #[test]
    fn vector_exp_is_within_two_ulp_of_libm_over_the_softmax_range() {
        // Softmax inputs are max-shifted, so ≤ 0; the far tail underflows to 0 in both.
        let xs = prng(7, 4099, -90.0, 0.0);
        let mut v = xs.clone();
        let s = exp_shift_sum(&mut v, 0.0);
        let mut want_sum = 0.0f64;
        for (i, (&x, &got)) in xs.iter().zip(&v).enumerate() {
            let want = x.exp();
            want_sum += want as f64;
            let ulp = (want.to_bits() as i64 - got.to_bits() as i64).abs();
            assert!(
                ulp <= 2 || (want - got).abs() < 1e-37,
                "exp({x}) = {got}, libm {want}, {ulp} ulp at {i}"
            );
        }
        assert!((s - want_sum).abs() <= 1e-5 * want_sum.max(1e-30));
    }

    #[test]
    fn exp_handles_overflow_and_underflow_like_ggml() {
        let mut v = vec![100.0f32, -200.0, 0.0, 88.0, -87.0, -104.5, 1.0, -1.0, 50.0];
        exp_shift_sum(&mut v, 0.0);
        assert!(v[0].is_infinite());
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 1.0);
        assert!((v[3] / 88.0f32.exp() - 1.0).abs() < 1e-6);
        assert!((v[6] / 1.0f32.exp() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn scaled_exp_is_exp_of_the_scaled_shift() {
        let xs = prng(11, 1003, -30.0, 4.0);
        let m = max(&xs);
        let mut a = xs.clone();
        let k = 1.0f32 / 0.7;
        let sa = exp_scaled_sum(&mut a, m, k);
        // The same rounded argument the kernel forms: `exp` multiplies an argument's
        // relative error by its magnitude, so `/0.7` vs `·(1/0.7)` alone is ~40 ulp.
        let mut b: Vec<f32> = xs.iter().map(|x| (x - m) * k).collect();
        let sb = exp_shift_sum(&mut b, 0.0);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() <= 2e-6 * y.max(1e-30), "{x} vs {y}");
        }
        assert!((sa - sb).abs() < 1e-4 * sb);
    }

    #[test]
    fn max_matches_the_scalar_fold() {
        for n in [0, 1, 7, 8, 9, 31, 1000] {
            let xs = prng(n as u64 + 3, n, -5.0, 5.0);
            assert_eq!(
                max(&xs),
                xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                "n = {n}"
            );
        }
    }

    #[test]
    fn swiglu_matches_the_scalar_definition() {
        let g0 = prng(5, 1539, -12.0, 12.0);
        let up = prng(9, 1539, -3.0, 3.0);
        let mut g = g0.clone();
        swiglu(&mut g, &up);
        for i in 0..g0.len() {
            let want = (g0[i] / (1.0 + (-g0[i]).exp())) * up[i];
            assert!(
                (g[i] - want).abs() <= 1e-6 * want.abs().max(1e-3),
                "{i}: {} vs {want}",
                g[i]
            );
        }
    }

    /// An element's `swiglu` bits must not depend on where the row was split.
    ///
    /// The one-row decode path splits a feed-forward row into unequal shares across
    /// the pool (`swiglu_rows`, `Sched::Static`), while a batched step hands every
    /// worker a whole row. Before the tail was moved onto `exp8`, the narrow path
    /// evaluated each share's last 1..7 elements with `libm`'s `expf` and the batched
    /// path evaluated them with `exp8`, so the same row came out different bits
    /// depending on the batch it ran in — the batch-invariance break the repacked
    /// Q4_K fold then surfaced. The scratch is dirtied by an eight-row call first, so
    /// a split can never pass by reading a leftover lane.
    ///
    /// It runs on every target: the scalar path must hold the same property.
    #[test]
    fn a_rows_swiglu_does_not_depend_on_the_split() {
        let n = 1024;
        let g0 = prng(29, n, -12.0, 12.0);
        let up = prng(31, n, -3.0, 3.0);
        let mut want = g0.clone();
        swiglu(&mut want, &up);

        let mut scratch_g = vec![0.0f32; 8 * n];
        let mut scratch_u = vec![0.0f32; 8 * n];
        for r in 0..8 {
            scratch_g[r * n..(r + 1) * n].copy_from_slice(&prng(100 + r as u64, n, -12.0, 12.0));
            scratch_u[r * n..(r + 1) * n].copy_from_slice(&prng(200 + r as u64, n, -3.0, 3.0));
        }
        // The wider call dirties the whole buffer.
        swiglu(&mut scratch_g, &scratch_u);

        // Widths one to eight, as a row split across that many workers.
        for w in 1..=8usize {
            scratch_g[..n].copy_from_slice(&g0);
            scratch_u[..n].copy_from_slice(&up);
            for k in 0..w {
                let (lo, hi) = (n * k / w, n * (k + 1) / w);
                swiglu(&mut scratch_g[lo..hi], &scratch_u[lo..hi]);
            }
            for i in 0..n {
                assert_eq!(
                    scratch_g[i].to_bits(),
                    want[i].to_bits(),
                    "width {w}, element {i}: {} vs {}",
                    scratch_g[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn geglu_matches_the_scalar_definition() {
        let g0 = prng(5, 1539, -12.0, 12.0);
        let up = prng(9, 1539, -3.0, 3.0);
        let mut g = g0.clone();
        geglu(&mut g, &up);
        for i in 0..g0.len() {
            let want = gelu_tanh(g0[i]) * up[i];
            assert_eq!(g[i], want, "{i}: {} vs {want}", g[i]);
        }
    }

    /// The `swiglu_oai` of llama.cpp v0.9.0 (`ggml/src/ggml-cpu/ops.cpp:3388-3450`):
    /// the gate is clamped above only, the up on both sides, and the gate's
    /// exponential carries the alpha.
    #[test]
    fn swiglu_oai_matches_llama_cpp_s_formula() {
        let (alpha, limit) = (1.702f32, 7.0f32);
        let g0 = prng(11, 1539, -12.0, 12.0);
        let up = prng(13, 1539, -9.0, 9.0);
        let mut g = g0.clone();
        swiglu_oai(&mut g, &up, alpha, limit);
        for i in 0..g0.len() {
            let x = g0[i].min(limit);
            let y = up[i].clamp(-limit, limit);
            let want = (x / (1.0 + (alpha * -x).exp())) * (y + 1.0);
            assert!(
                (g[i] - want).abs() <= 1e-5 * want.abs().max(1e-3),
                "{i}: {} vs {want}",
                g[i]
            );
        }
    }

    /// The vector path and the portable one must agree bit for bit: they are the
    /// same arithmetic, and the engine's output must not depend on the tier.
    #[test]
    fn swiglu_oai_vector_and_scalar_agree_bit_for_bit() {
        let (alpha, limit) = (1.702f32, 7.0f32);
        let g0 = prng(17, 1539, -12.0, 12.0);
        let up = prng(19, 1539, -9.0, 9.0);
        let mut vec = g0.clone();
        swiglu_oai(&mut vec, &up, alpha, limit);
        let mut scalar = g0.clone();
        for (g, &u) in scalar.iter_mut().zip(&up) {
            *g = swiglu_oai_one(*g, u, alpha, limit);
        }
        assert_eq!(vec, scalar);
    }

    /// The FMA path and the portable one must agree bit for bit, over a dense sweep of
    /// the argument.
    ///
    /// They are the same arithmetic: an IEEE fused multiply-add has one result, and the
    /// hardware's `vfmadd` and the C library's `fmaf` both produce it. If they ever
    /// differed, the engine's output would depend on the CPU it ran on — the one thing
    /// the kernel tiers must not decide.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_fma_path_agrees_with_the_portable_one_bit_for_bit() {
        if !crate::cpu::have_fma() {
            eprintln!("skipped: no FMA");
            return;
        }
        // A dense sweep across the whole range `tanh` handles, so that every branch is
        // walked many times: the polynomial's, the exponential's, and the saturation's.
        let mut xs: Vec<f32> = Vec::new();
        let mut x = -24.0f32;
        while x <= 24.0 {
            xs.push(x);
            x += 0.000_37;
        }
        // Every 997th `f32` in [0, 1): the polynomial branch, at the resolution of the
        // format rather than of the sweep.
        let mut bits = 0u32;
        while bits < 0x3f80_0000 {
            xs.push(f32::from_bits(bits));
            bits += 997;
        }
        // The corners: the branch boundaries, the extremes, the denormals and the NaN.
        for v in [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            TANH_POLY_LIMIT,
            -TANH_POLY_LIMIT,
            0.54999995,
            TANH_SATURATE,
            -TANH_SATURATE,
            8.999999,
            9.000001,
            1e-30,
            -1e-30,
            1e-45,
            -1e-45,
            1e10,
            -1e10,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            xs.push(v);
        }
        // `tanh` itself: the two paths must not differ.
        for &x in &xs {
            // SAFETY: FMA is available (checked above).
            let got = unsafe { tanh_fma(x) };
            assert_eq!(
                got.to_bits(),
                tanh_scalar(x).to_bits(),
                "tanh({x}): FMA {got:?}, portable {:?}",
                tanh_scalar(x)
            );
        }
        // And the GeGLU that uses it, over the same sweep.
        for u in [1.0f32, -2.5, 0.125, f32::MIN_POSITIVE] {
            let up = vec![u; xs.len()];
            let mut a = xs.clone();
            let mut b = xs.clone();
            geglu_scalar(&mut a, &up);
            // SAFETY: FMA is available (checked above).
            unsafe { geglu_fma(&mut b, &up) };
            for i in 0..a.len() {
                assert_eq!(
                    a[i].to_bits(),
                    b[i].to_bits(),
                    "geglu({}, {u}): FMA {:?}, portable {:?}",
                    xs[i],
                    b[i],
                    a[i]
                );
            }
        }
    }

    #[test]
    fn hadamard_is_orthonormal_and_inverts_exactly_enough() {
        for n in [64usize, 128, 256] {
            let x = prng(n as u64, n, -4.0, 4.0);
            let y = prng(n as u64 + 1, n, -4.0, 4.0);
            let dot = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(p, q)| (*p as f64) * (*q as f64))
                    .sum::<f64>()
            };
            let (mut hx, mut hy) = (x.clone(), y.clone());
            hadamard(&mut hx);
            hadamard(&mut hy);
            assert!(
                (dot(&hx, &hy) - dot(&x, &y)).abs() < 1e-3,
                "dot products survive the rotation"
            );
            hadamard_inv(&mut hx);
            for (a, b) in hx.iter().zip(&x) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn hadamard_spreads_an_outlier_channel() {
        // One channel 100× the rest — the case per-vector int8 handles worst.
        let mut x = vec![0.01f32; 128];
        x[17] = 1.0;
        let amax_before = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let mean_before = x.iter().map(|v| v.abs()).sum::<f32>() / 128.0;
        hadamard(&mut x);
        let amax_after = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let mean_after = x.iter().map(|v| v.abs()).sum::<f32>() / 128.0;
        assert!(amax_before / mean_before > 30.0);
        assert!(
            amax_after / mean_after < 2.0,
            "peak-to-mean {} after rotation",
            amax_after / mean_after
        );
    }
}
