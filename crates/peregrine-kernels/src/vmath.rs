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
#[inline]
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
#[inline]
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
pub fn tanh(x: f32) -> f32 {
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

/// `x[i] = exp(x[i] − shift)` in place; returns `Σ x[i]` after the update.
///
/// The softmax numerator and normaliser in one pass. The caller divides.
pub fn exp_shift_sum(x: &mut [f32], shift: f32) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
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
            // SAFETY: features checked.
            return unsafe { max_avx2(x) };
        }
    }
    x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
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
    while i < gate.len() {
        let g = gate[i];
        gate[i] = (g / (1.0 + (-g).exp())) * up[i];
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
#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + tanh(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)))
}

/// GeGLU in place: `gate[i] = gelu_tanh(gate[i]) · up[i]` — matching llama.cpp's `LLM_FFN_GELU`.
///
/// Scalar, and deliberately so: it has no vector path, so there is no second arithmetic to
/// keep in step with [`gelu_tanh`]'s. A vector GeGLU may only be added if it evaluates the
/// same expressions lane-wise, so that it is bit for bit the scalar result.
pub fn geglu(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = gelu_tanh(*g) * u;
    }
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
