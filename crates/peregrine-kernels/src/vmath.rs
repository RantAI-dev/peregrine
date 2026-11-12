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
//! operations alone: [`exp_scalar`] is the scalar twin of `exp8` (bit for bit), and
//! [`tanh`] is built on it. A libm call is not reproducible across C libraries — glibc
//! 2.40 replaced `tanhf` with a correctly rounded version, and the same kernel then
//! produced different activations, different logits and different golden files on
//! glibc 2.39 and 2.43 (issue #64). A transcendental built here is fixed by its
//! definition, not by the machine's libc, so the engine's output depends on the kernel
//! tier alone.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
#[inline]
fn vector_ok() -> bool {
    crate::cpu_has!("avx2") && crate::cpu_has!("fma")
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn vector_ok() -> bool {
    crate::cpu::have_neon()
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

/// Four `exp`s at once — [`exp8`]'s constants, operation order and fused multiply-adds,
/// four lanes wide. The result is [`exp_scalar`]'s bit for bit.
///
/// The `|n| > 126` path (underflow, overflow) is rare enough that the four lanes take
/// the scalar twin instead of carrying the masked sequence: each lane's result depends
/// on its input alone either way, so a row's bits cannot depend on its neighbours.
///
/// # Safety
/// The CPU must have NEON.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn exp4(x: float32x4_t) -> float32x4_t {
    let r = vdupq_n_f32(f32::from_bits(0x4b40_0000)); // 0x1.8p23
    let z = vfmaq_f32(r, x, vdupq_n_f32(EXP_LOG2E));
    let n = vsubq_f32(z, r);
    let neg = vnegq_f32(n);
    let b = vfmaq_f32(
        vfmaq_f32(x, neg, vdupq_n_f32(EXP_LN2_HI)),
        neg,
        vdupq_n_f32(EXP_LN2_LO),
    );
    let mut lanes = [0.0f32; 4];
    vst1q_f32(lanes.as_mut_ptr(), vabsq_f32(n));
    if lanes[0] > 126.0 || lanes[1] > 126.0 || lanes[2] > 126.0 || lanes[3] > 126.0 {
        vst1q_f32(lanes.as_mut_ptr(), x);
        for v in lanes.iter_mut() {
            *v = exp_scalar(*v);
        }
        return vld1q_f32(lanes.as_ptr());
    }
    let e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
    let u = vmulq_f32(b, b);
    let d = vfmaq_f32(vdupq_n_f32(EXP_P1), vdupq_n_f32(EXP_P0), b);
    let c = vfmaq_f32(vdupq_n_f32(EXP_P3), vdupq_n_f32(EXP_P2), b);
    let d = vfmaq_f32(c, d, u);
    let j = vfmaq_f32(vmulq_f32(vdupq_n_f32(EXP_P4), b), d, u);
    vfmaq_f32(k, j, k)
}

/// [`exp4`] of `−x`: the sigmoid's exponential.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn exp4_neg(x: float32x4_t) -> float32x4_t {
    exp4(vnegq_f32(x))
}

/// `x_i = exp(x_i − shift)` in place, in the scalar loop's index order.
///
/// # Safety
/// The CPU must have NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp_shift_sum_neon(x: &mut [f32], shift: f32) -> f64 {
    let s = vdupq_n_f32(shift);
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 4 <= x.len() {
        let v = exp4(vsubq_f32(vld1q_f32(x.as_ptr().add(i)), s));
        vst1q_f32(x.as_mut_ptr().add(i), v);
        let mut lanes = [0.0f32; 4];
        vst1q_f32(lanes.as_mut_ptr(), v);
        for v in lanes {
            sum += v as f64;
        }
        i += 4;
    }
    while i < x.len() {
        x[i] = exp_scalar(x[i] - shift);
        sum += x[i] as f64;
        i += 1;
    }
    sum
}

/// `x_i = exp((x_i − shift) · scale)` in place, in the scalar loop's index order.
///
/// # Safety
/// The CPU must have NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp_scaled_sum_neon(x: &mut [f32], shift: f32, scale: f32) -> f64 {
    let s = vdupq_n_f32(shift);
    let k = vdupq_n_f32(scale);
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 4 <= x.len() {
        let v = exp4(vmulq_f32(vsubq_f32(vld1q_f32(x.as_ptr().add(i)), s), k));
        vst1q_f32(x.as_mut_ptr().add(i), v);
        let mut lanes = [0.0f32; 4];
        vst1q_f32(lanes.as_mut_ptr(), v);
        for v in lanes {
            sum += v as f64;
        }
        i += 4;
    }
    while i < x.len() {
        x[i] = exp_scalar((x[i] - shift) * scale);
        sum += x[i] as f64;
        i += 1;
    }
    sum
}

/// [`swiglu`], four lanes at a time on [`exp4`]; the padded tail keeps a split row's
/// bits the same as an unsplit one.
///
/// # Safety
/// The CPU must have NEON; the slices have equal length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn swiglu_neon(gate: &mut [f32], up: &[f32]) {
    let one = vdupq_n_f32(1.0);
    let mut i = 0;
    while i + 4 <= gate.len() {
        let x = vld1q_f32(gate.as_ptr().add(i));
        let silu = vdivq_f32(x, vaddq_f32(one, exp4_neg(x)));
        vst1q_f32(
            gate.as_mut_ptr().add(i),
            vmulq_f32(silu, vld1q_f32(up.as_ptr().add(i))),
        );
        i += 4;
    }
    if i < gate.len() {
        let rem = gate.len() - i;
        let mut g = [0.0f32; 4];
        let mut u = [0.0f32; 4];
        g[..rem].copy_from_slice(&gate[i..]);
        u[..rem].copy_from_slice(&up[i..]);
        let x = vld1q_f32(g.as_ptr());
        let silu = vdivq_f32(x, vaddq_f32(one, exp4_neg(x)));
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), vmulq_f32(silu, vld1q_f32(u.as_ptr())));
        gate[i..].copy_from_slice(&out[..rem]);
    }
}

/// [`swiglu_oai`], four lanes at a time on [`exp4`].
///
/// # Safety
/// The CPU must have NEON; the slices have equal length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn swiglu_oai_neon(gate: &mut [f32], up: &[f32], alpha: f32, limit: f32) {
    let one = vdupq_n_f32(1.0);
    let av = vdupq_n_f32(alpha);
    let lv = vdupq_n_f32(limit);
    let nlv = vdupq_n_f32(-limit);
    let mut i = 0;
    while i + 4 <= gate.len() {
        let g = vld1q_f32(gate.as_ptr().add(i));
        let u = vld1q_f32(up.as_ptr().add(i));
        // `min(gate, limit)`: the limit wins only when it is the smaller one.
        let x = vbslq_f32(vcltq_f32(lv, g), lv, g);
        // `clamp(up, -limit, limit)`, in the scalar branches' order.
        let y = vbslq_f32(vcltq_f32(u, nlv), nlv, vbslq_f32(vcltq_f32(lv, u), lv, u));
        let glu = vdivq_f32(x, vaddq_f32(one, exp4(vmulq_f32(av, vnegq_f32(x)))));
        vst1q_f32(gate.as_mut_ptr().add(i), vmulq_f32(glu, vaddq_f32(y, one)));
        i += 4;
    }
    while i < gate.len() {
        gate[i] = swiglu_oai_one(gate[i], up[i], alpha, limit);
        i += 1;
    }
}

/// [`mul_sigmoid`], four lanes at a time on [`exp4`].
///
/// # Safety
/// The CPU must have NEON; the slices have equal length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mul_sigmoid_neon(x: &mut [f32], z: &[f32]) {
    let one = vdupq_n_f32(1.0);
    let mut i = 0;
    while i + 4 <= x.len() {
        let e = exp4_neg(vld1q_f32(z.as_ptr().add(i)));
        vst1q_f32(
            x.as_mut_ptr().add(i),
            vdivq_f32(vld1q_f32(x.as_ptr().add(i)), vaddq_f32(one, e)),
        );
        i += 4;
    }
    while i < x.len() {
        x[i] /= 1.0 + exp_scalar(-z[i]);
        i += 1;
    }
}

/// [`silu`]/[`mul_silu`], four lanes at a time on [`exp4`].
///
/// # Safety
/// The CPU must have NEON; with `z`, the slices have equal length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mul_silu_neon(x: &mut [f32], z: Option<&[f32]>) {
    let one = vdupq_n_f32(1.0);
    let mut i = 0;
    while i + 4 <= x.len() {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let g = match z {
            Some(z) => vld1q_f32(z.as_ptr().add(i)),
            None => xv,
        };
        let s = vdivq_f32(g, vaddq_f32(one, exp4_neg(g)));
        let r = if z.is_some() { vmulq_f32(xv, s) } else { s };
        vst1q_f32(x.as_mut_ptr().add(i), r);
        i += 4;
    }
    while i < x.len() {
        let g = z.map_or(x[i], |z| z[i]);
        let s = g / (1.0 + exp_scalar(-g));
        x[i] = if z.is_some() { x[i] * s } else { s };
        i += 1;
    }
}

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

/// `exp` — the scalar twin of `exp8`: the same constants, the same operation order and the
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
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_neon() {
            // SAFETY: NEON checked above.
            return unsafe { exp_shift_sum_neon(x, shift) };
        }
    }
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = det_exp(*v - shift);
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
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_neon() {
            // SAFETY: NEON checked above.
            return unsafe { exp_scaled_sum_neon(x, shift, scale) };
        }
    }
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = det_exp((*v - shift) * scale);
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
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_neon() {
            // SAFETY: NEON checked above.
            return unsafe { swiglu_neon(gate, up) };
        }
    }
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = (*g / (1.0 + det_exp(-*g))) * u;
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
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_neon() {
            // SAFETY: NEON checked above.
            return unsafe { swiglu_oai_neon(gate, up, alpha, limit) };
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

/// GeGLU with the **exact** GELU: `gate[i] = gelu_erf(gate[i]) · up[i]`.
///
/// This is the ModernBERT family's feed-forward activation (`LLM_FFN_GEGLU_ERF`,
/// `src/llama-graph.cpp`'s `ggml_geglu_erf`): the first half of the `ffn_up` row is
/// activated and multiplies the second half. llama.cpp evaluates
/// `0.5·x·(1 + erff(x/√2))·g` left to right; [`gelu_erf`] is `(0.5·x)·(1 + erf(x/√2))`
/// and the final multiply is the same one, so the bits match for the same `erf`.
///
/// On x86-64 with AVX2 and FMA, sixteen elements are evaluated at once by the
/// same GELU body [`gelu_erf_rows`] uses; the product with `up` is
/// the one `f32` multiply the scalar loop does, so the result is bit for bit the
/// scalar one. The tail takes the scalar [`gelu_erf`], which is itself bit for bit
/// the vector `erf` (the test pins that too). On aarch64 with NEON, eight elements are
/// evaluated at once: the same erf chain as [`gelu_erf_rows`]' NEON body, then one
/// multiply by `up`.
pub fn geglu_erf(gate: &mut [f32], up: &[f32]) {
    assert_eq!(
        gate.len(),
        up.len(),
        "geglu_erf: gate and up must be the same length, got {} and {}",
        gate.len(),
        up.len()
    );
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 and FMA checked; the lengths are equal.
            unsafe { geglu_erf_avx2(gate, up) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if vector_ok() {
            // SAFETY: NEON checked, and the lengths were asserted above.
            unsafe { geglu_erf_neon(gate, up) };
            return;
        }
    }
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = gelu_erf(*g) * u;
    }
}

/// The AVX2 [`geglu_erf`]: sixteen elements per step.
///
/// # Safety
///
/// AVX2 and FMA must be available, and `gate` and `up` must be the same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn geglu_erf_avx2(gate: &mut [f32], up: &[f32]) {
    let n = gate.len();
    let p = gate.as_mut_ptr();
    let u = up.as_ptr();
    let mut i = 0;
    while i + 16 <= n {
        // The GELU writes the gate block, then the one multiply the definition's
        // last step is.
        gelu_erf_16(p.add(i));
        let g0 = _mm256_loadu_ps(p.add(i));
        let g1 = _mm256_loadu_ps(p.add(i + 8));
        _mm256_storeu_ps(p.add(i), _mm256_mul_ps(g0, _mm256_loadu_ps(u.add(i))));
        _mm256_storeu_ps(
            p.add(i + 8),
            _mm256_mul_ps(g1, _mm256_loadu_ps(u.add(i + 8))),
        );
        i += 16;
    }
    while i < n {
        *p.add(i) = gelu_erf(*p.add(i)) * *u.add(i);
        i += 1;
    }
}

/// The NEON [`geglu_erf`]: eight elements per step, through [`gelu_erf_8_neon`] and
/// one `vmulq_f32`, so each lane is the scalar definition's arithmetic.
///
/// # Safety
///
/// NEON must be available, and `gate` and `up` must have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn geglu_erf_neon(gate: &mut [f32], up: &[f32]) {
    let n = gate.len();
    let g = gate.as_mut_ptr();
    let u = up.as_ptr();
    let mut i = 0;
    while i + 8 <= n {
        gelu_erf_8_neon(g.add(i));
        let a0 = vld1q_f32(g.add(i));
        let a1 = vld1q_f32(g.add(i + 4));
        let u0 = vld1q_f32(u.add(i));
        let u1 = vld1q_f32(u.add(i + 4));
        vst1q_f32(g.add(i), vmulq_f32(a0, u0));
        vst1q_f32(g.add(i + 4), vmulq_f32(a1, u1));
        i += 8;
    }
    while i < n {
        *g.add(i) = gelu_erf(*g.add(i)) * *u.add(i);
        i += 1;
    }
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

// ---------------------------------------------------------------------------
// The BERT family's element-wise kernels (S4.6a): erf, the exact GELU, LayerNorm
// ---------------------------------------------------------------------------
//
// These three are *definitions*, not implementations of an unspecified thing:
// the scalar reference forward in `crates/peregrine-models/tests/support/`
// shares exactly them, so an F32 encoder reproduces one vector bit for bit. The
// order of every operation is stated where it is used, and nothing here calls
// the C library (the module's rule): a tier's bits depend on this file only.
//
// The vector bodies below (AVX2 on x86-64, NEON on aarch64) evaluate exactly
// these sequences, lane for lane: a vector body is bit-identical to its scalar
// twin, which is why the definitions are written as a fixed sequence of IEEE
// operations rather than as "a polynomial approximation": the reduction order of
// `layer_norm` and the term counts of `erf` are part of the result.

/// `2/√π`, the series' prefactor (`0x3ff20dd750429b6d`).
const ERF_TWO_OVER_SQRT_PI: f64 = f64::from_bits(0x3ff2_0dd7_5042_9b6d);
/// `√π` (`0x3ffc5bf891b4ef6a`), the asymptotic `erfc`'s denominator.
const ERF_SQRT_PI: f64 = f64::from_bits(0x3ffc_5bf8_91b4_ef6a);

/// `(−1)^k / (k!·(2k+1))`: the Maclaurin series of `erf`,
///
/// ```text
/// erf(x) = (2/√π) · Σ_k (−1)^k x^(2k+1) / (k! (2k+1))
///        = (2/√π) · x · P(x²),
/// ```
///
/// truncated at a term count chosen per range (see [`erf_series`]).
const ERF_SERIES: [f64; 37] = [
    f64::from_bits(0x3ff0_0000_0000_0000), // 1
    f64::from_bits(0xbfd5_5555_5555_5555), // −1/3
    f64::from_bits(0x3fb9_9999_9999_999a), // 1/10
    f64::from_bits(0xbf98_6186_1861_8618), // −1/42
    f64::from_bits(0x3f72_f684_bda1_2f68), // 1/216
    f64::from_bits(0xbf48_d301_8d30_18d3), // −1/1320
    f64::from_bits(0x3f1c_01c0_1c01_c01c), // 1/9360
    f64::from_bits(0xbeeb_bd77_9334_ef0b), // −1/75600
    f64::from_bits(0x3eb8_7a00_187a_0018), // 1/685440
    f64::from_bits(0xbe83_777c_5556_8ccd), // −1/6894720
    f64::from_bits(0x3e4c_2e30_5487_0b38), // 1/76204800
    f64::from_bits(0xbe12_b673_10aa_9f3a), // −1/918086400
    f64::from_bits(0x3dd6_f448_e13e_85e1), // 1/11975040000
    f64::from_bits(0xbd9a_289e_e7e4_0f74), // −1/168129561600
    f64::from_bits(0x3d5b_d577_e658_d020), // 1/2528270412800
    f64::from_bits(0xbd1b_c625_0fb1_4231), // −1/40537905408000
    f64::from_bits(0x3cda_173a_167f_ba4d), // 1/690752066304000
    f64::from_bits(0xbc97_271c_be58_63ec), // −1/12449059983360000
    f64::from_bits(0x3c53_77c2_110f_2083), // 1/236887827111936000
    f64::from_bits(0xbc0f_1b40_73b3_4a68), // …
    f64::from_bits(0x3bc7_abd7_2258_fb6e),
    f64::from_bits(0xbb81_3246_abce_1bdd),
    f64::from_bits(0x3b37_e6b8_1382_cd42),
    f64::from_bits(0xbaef_d6be_bd65_107a),
    f64::from_bits(0x3aa4_5c0a_838e_fe59),
    f64::from_bits(0xba59_09c9_de3a_31c5),
    f64::from_bits(0x3a0d_a746_0554_e5da),
    f64::from_bits(0xb9c0_eef3_0fa1_0d2c),
    f64::from_bits(0x3972_ac65_385f_79ac),
    f64::from_bits(0xb923_e81b_b570_1ac5),
    f64::from_bits(0x38d4_899f_cdef_0a8e),
    f64::from_bits(0xb884_86ee_a20c_2656),
    f64::from_bits(0x3833_e53d_efc4_e233),
    f64::from_bits(0xb7e2_b778_acc3_dee0),
    f64::from_bits(0x3791_1ae8_1077_a49e),
    f64::from_bits(0xb73e_6597_092c_cbf1),
    f64::from_bits(0x36ea_4776_7f2a_3c33),
];

/// `(2/√π)·x·P(x²)` with the first `terms` coefficients of [`ERF_SERIES`], in `f64`.
///
/// Horner's rule from the last term down, which is the arithmetic the definition
/// names: `terms` multiply-adds from the top coefficient. The truncation counts
/// below are the smallest that keep the error under 3e-9 on each range (measured
/// from `[0, 1]` in steps of 1e-6 against the oracle in the tests):
///
/// | range | terms | first omitted term's worst | measured |
/// |---|---|---|---|
/// | `|x| ≤ 1` | 10 | `x¹⁹/19!` | 1.14e-9 |
/// | `≤ 2` | 22 | at `x = 2` | 1.13e-10 |
/// | `≤ 3` | 36 | at `x = 3` | 5.40e-10 |
#[inline(always)]
fn erf_series(x: f64, terms: usize) -> f64 {
    let u = x * x;
    let mut p = ERF_SERIES[terms - 1];
    let mut k = terms - 1;
    while k > 0 {
        k -= 1;
        p = p.mul_add(u, ERF_SERIES[k]);
    }
    x * ERF_TWO_OVER_SQRT_PI * p
}

/// `erfc(x)` for `3 < x < 4.6`: the asymptotic series
///
/// ```text
/// erfc(x) = exp(−x²)/(x·√π) · Σ_{k<8} (−1)^k (2k−1)!! / (2x²)^k
/// ```
///
/// eight terms, where the terms are smallest. Measured absolute error of
/// `1 − erfc` on `(3, 4.6)`: 2.26e-9 at `x = 3`, 2.5e-12 at `x = 3.5`.
fn erfc_tail(a: f64) -> f64 {
    let z = 2.0 * a * a;
    let mut t = 1.0f64;
    let mut sum = 1.0f64;
    let mut n = 1.0f64;
    while n < 8.0 {
        t *= -(2.0 * n - 1.0) / z;
        sum += t;
        n += 1.0;
    }
    exp_f64(-a * a) / (a * ERF_SQRT_PI) * sum
}

/// `erf(x)` — the exact error function the BERT family's GELU is defined with.
///
/// `x` is `f32`; every intermediate is `f64` and the result is rounded to `f32`
/// once, at the end. Range splitting, all on `a = |x|`:
///
/// * `a ≤ 1`: 10 terms of the Maclaurin series (`erf_series`);
/// * `1 < a ≤ 2`: 22 terms of the same series;
/// * `2 < a ≤ 3`: 36 terms;
/// * `3 < a < 4.6`: `1 − erfc_tail(a)`, the asymptotic `erfc`;
/// * `a ≥ 4.6`: `1` (`erfc(4.6) = 1.7e-10`, a thousandth of an `f32` ulp of 1).
///
/// Measured against `libm`'s `f64` `erf` over 1 000 001 points in `[−6, 6]`
/// (`erf_agrees_with_an_independent_oracle`): **max absolute error 5.44e-8, max
/// 1 ulp** of the `f32` result — the error is entirely the final rounding
/// wherever `erf` is near ±1, and the series is ~1e-15 relative near 0.
///
/// # The FMA split
///
/// Like [`gelu_tanh`], the arithmetic is compiled twice: once for a plain
/// x86-64 target (`fmaf` from the C library, one IEEE result) and once inside a
/// `#[target_feature(enable = "fma")]` function (the hardware instruction, the
/// same IEEE result), so every kernel tier produces the same bits.
pub fn erf(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::have_fma() {
            // SAFETY: FMA checked.
            return unsafe { erf_fma(x) };
        }
    }
    erf_scalar(x)
}

/// The portable [`erf`]; see it for the range split and the measured error.
#[inline(always)]
fn erf_scalar(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let a = x.abs() as f64;
    let y = if a <= 1.0 {
        erf_series(a, 10)
    } else if a <= 2.0 {
        erf_series(a, 22)
    } else if a <= 3.0 {
        erf_series(a, 36)
    } else if a < 4.6 {
        1.0 - erfc_tail(a)
    } else {
        1.0
    };
    (y as f32).copysign(x)
}

/// [`erf_scalar`] compiled for a CPU with the FMA instruction.
///
/// # Safety
///
/// FMA must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn erf_fma(x: f32) -> f32 {
    erf_scalar(x)
}

/// `1/√2`, the GELU's `erf` argument scale (`0x3f3504f3`).
const GELU_INV_SQRT_2: f32 = f32::from_bits(0x3f35_04f3);

/// The exact GELU: `gelu(x) = 0.5·x·(1 + erf(x/√2))`.
///
/// This is the BERT family's activation, and the model's own arithmetic: a
/// BERT checkpoint trains with `torch.nn.functional.gelu` (the erf form), not
/// with the `tanh` approximation [`gelu_tanh`] evaluates. The order is fixed:
/// `t = x · (1/√2)`, `e = erf(t)`, then `(0.5·x) · (1 + e)` — three `f32`
/// roundings plus [`erf`]'s own.
pub fn gelu_erf(x: f32) -> f32 {
    let t = x * GELU_INV_SQRT_2;
    (0.5 * x) * (1.0 + erf(t))
}

/// [`gelu_erf`] over a row, four elements at a time so the four `erf` chains
/// interleave (each element's arithmetic and its bits are unchanged).
///
/// On x86-64 with AVX2 and FMA, sixteen elements are evaluated at once: four
/// `f64` lanes each; on aarch64 with NEON, eight elements, two `f64` lanes each.
/// The per-element arithmetic is the scalar definition's, in
/// the same order — three Maclaurin polynomials (10, 22 and 36 terms) are
/// evaluated for the lanes that need them and selected by `|x|`, the `erfc`
/// tail is evaluated by the same scalar `erfc_tail` for the rare lane in
/// `(3, 4.6)`, and a NaN keeps its own bits. The tests pin the two paths equal
/// on every input class.
pub fn gelu_erf_rows(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 and FMA checked.
            unsafe { gelu_erf_rows_avx2(x) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if vector_ok() {
            // SAFETY: NEON checked.
            unsafe { gelu_erf_rows_neon(x) };
            return;
        }
    }
    gelu_erf_rows_scalar(x);
}

/// The portable [`gelu_erf_rows`]: the scalar twin, bit for bit.
#[inline]
fn gelu_erf_rows_scalar(x: &mut [f32]) {
    let mut i = 0;
    while i + 4 <= x.len() {
        let a = gelu_erf(x[i]);
        let b = gelu_erf(x[i + 1]);
        let c = gelu_erf(x[i + 2]);
        let d = gelu_erf(x[i + 3]);
        x[i] = a;
        x[i + 1] = b;
        x[i + 2] = c;
        x[i + 3] = d;
        i += 4;
    }
    while i < x.len() {
        x[i] = gelu_erf(x[i]);
        i += 1;
    }
}

/// The AVX2 [`gelu_erf_rows`]: sixteen elements per step.
///
/// # Safety
///
/// AVX2 and FMA must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_erf_rows_avx2(x: &mut [f32]) {
    let n = x.len();
    let p = x.as_mut_ptr();
    let mut i = 0;
    while i + 32 <= n {
        gelu_erf_16(p.add(i));
        gelu_erf_16(p.add(i + 16));
        i += 32;
    }
    while i + 16 <= n {
        gelu_erf_16(p.add(i));
        i += 16;
    }
    while i < n {
        *p.add(i) = gelu_erf(*p.add(i));
        i += 1;
    }
}

/// Sixteen GELUs: four `f64` lanes per `erf`, and the four chains interleaved so
/// the Horner loops' latency hides. Every lane does the scalar definition's
/// operations in the scalar definition's order.
///
/// # Safety
///
/// AVX2 and FMA must be available, and `p` must point at sixteen writable floats.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_erf_16(p: *mut f32) {
    let x0 = _mm256_loadu_ps(p);
    let x1 = _mm256_loadu_ps(p.add(8));
    let inv = _mm256_set1_ps(GELU_INV_SQRT_2);
    let t0 = _mm256_mul_ps(x0, inv);
    let t1 = _mm256_mul_ps(x1, inv);
    let tv = [
        _mm256_castps256_ps128(t0),
        _mm256_extractf128_ps(t0, 1),
        _mm256_castps256_ps128(t1),
        _mm256_extractf128_ps(t1, 1),
    ];
    let sign = _mm_set1_ps(-0.0);
    let one = _mm256_set1_pd(1.0);
    let three = _mm256_set1_pd(3.0);
    let mut a = [_mm256_setzero_pd(); 4];
    let mut u = [_mm256_setzero_pd(); 4];
    let mut p10 = [_mm256_set1_pd(ERF_SERIES[9]); 4];
    let mut p22 = [_mm256_set1_pd(ERF_SERIES[21]); 4];
    let mut p36 = [_mm256_set1_pd(ERF_SERIES[35]); 4];
    for g in 0..4 {
        // `x.abs() as f64`: the f32 absolute value, then the exact widening.
        a[g] = _mm256_cvtps_pd(_mm_andnot_ps(sign, tv[g]));
        u[g] = _mm256_mul_pd(a[g], a[g]);
    }
    // The term count is per lane, but a group of four lanes is usually inside one
    // range: on the encoder's feed-forward outputs almost every element has
    // `|x/√2| ≤ 2`. So the whole sixteen-element block takes the shortest path all
    // of its lanes can take. The three polynomials are still the definition's:
    // a shorter path skips only terms that no lane in the block would use.
    let mut any_gt1 = 0u32;
    let mut any_gt2 = 0u32;
    let two = _mm256_set1_pd(2.0);
    for g in 0..4 {
        any_gt1 |= _mm256_movemask_pd(_mm256_cmp_pd(a[g], one, _CMP_GT_OQ)) as u32;
        any_gt2 |= _mm256_movemask_pd(_mm256_cmp_pd(a[g], two, _CMP_GT_OQ)) as u32;
    }
    if any_gt1 == 0 {
        // Every lane is `a ≤ 1`: 10 terms.
        let mut k = 8usize;
        loop {
            let c = _mm256_set1_pd(ERF_SERIES[k]);
            for g in 0..4 {
                p10[g] = _mm256_fmadd_pd(p10[g], u[g], c);
            }
            if k == 0 {
                break;
            }
            k -= 1;
        }
        p22 = p10;
        p36 = p10;
    } else if any_gt2 == 0 {
        // Every lane is `a ≤ 2`: 22 and 10 terms, no 36-term chain.
        let mut k = 20usize;
        loop {
            let c = _mm256_set1_pd(ERF_SERIES[k]);
            for g in 0..4 {
                p22[g] = _mm256_fmadd_pd(p22[g], u[g], c);
                if k <= 8 {
                    p10[g] = _mm256_fmadd_pd(p10[g], u[g], c);
                }
            }
            if k == 0 {
                break;
            }
            k -= 1;
        }
        p36 = p22;
    } else {
        // The full split: 36, 22 and 10 terms.
        let mut k = 34usize;
        while k >= 21 {
            let c = _mm256_set1_pd(ERF_SERIES[k]);
            for g in 0..4 {
                p36[g] = _mm256_fmadd_pd(p36[g], u[g], c);
            }
            k -= 1;
        }
        while k > 8 {
            let c = _mm256_set1_pd(ERF_SERIES[k]);
            for g in 0..4 {
                p36[g] = _mm256_fmadd_pd(p36[g], u[g], c);
                p22[g] = _mm256_fmadd_pd(p22[g], u[g], c);
            }
            k -= 1;
        }
        loop {
            let c = _mm256_set1_pd(ERF_SERIES[k]);
            for g in 0..4 {
                p36[g] = _mm256_fmadd_pd(p36[g], u[g], c);
                p22[g] = _mm256_fmadd_pd(p22[g], u[g], c);
                p10[g] = _mm256_fmadd_pd(p10[g], u[g], c);
            }
            if k == 0 {
                break;
            }
            k -= 1;
        }
    }
    let pref = _mm256_set1_pd(ERF_TWO_OVER_SQRT_PI);
    let mut ef = [_mm_setzero_ps(); 4];
    for g in 0..4 {
        let le1 = _mm256_cmp_pd(a[g], one, _CMP_LE_OQ);
        let gt2 = _mm256_cmp_pd(a[g], _mm256_set1_pd(2.0), _CMP_GT_OQ);
        // `x * (2/√π) * P(x²)`, left to right as the definition writes it.
        let two_over = _mm256_mul_pd(a[g], pref);
        let s10 = _mm256_mul_pd(two_over, p10[g]);
        let s22 = _mm256_mul_pd(two_over, p22[g]);
        let s36 = _mm256_mul_pd(two_over, p36[g]);
        // a ≤ 1 takes 10 terms, 1 < a ≤ 2 takes 22, a > 2 takes 36.
        let mut y = _mm256_blendv_pd(s22, s36, gt2);
        y = _mm256_blendv_pd(y, s10, le1);
        // `3 < a < 4.6`: the scalar asymptotic `erfc`; `a ≥ 4.6`: exactly 1.
        let tail = _mm256_cmp_pd(a[g], three, _CMP_GT_OQ);
        if _mm256_movemask_pd(tail) != 0 {
            let mut av = [0.0f64; 4];
            let mut yv = [0.0f64; 4];
            _mm256_storeu_pd(av.as_mut_ptr(), a[g]);
            _mm256_storeu_pd(yv.as_mut_ptr(), y);
            for j in 0..4 {
                if av[j] > 3.0 {
                    yv[j] = if av[j] < 4.6 {
                        1.0 - erfc_tail(av[j])
                    } else {
                        1.0
                    };
                }
            }
            y = _mm256_loadu_pd(yv.as_ptr());
        }
        // Round once, then copy the sign of `t`; a NaN keeps its own bits
        // (the scalar `erf` returns its NaN argument unchanged).
        let mut e = _mm256_cvtpd_ps(y);
        let nan = _mm_cmp_ps(tv[g], tv[g], _CMP_UNORD_Q);
        e = _mm_blendv_ps(e, tv[g], nan);
        ef[g] = _mm_or_ps(e, _mm_and_ps(tv[g], sign));
    }
    let e01 = _mm256_set_m128(ef[1], ef[0]);
    let e23 = _mm256_set_m128(ef[3], ef[2]);
    let half = _mm256_set1_ps(0.5);
    let onef = _mm256_set1_ps(1.0);
    let r0 = _mm256_mul_ps(_mm256_mul_ps(half, x0), _mm256_add_ps(onef, e01));
    let r1 = _mm256_mul_ps(_mm256_mul_ps(half, x1), _mm256_add_ps(onef, e23));
    _mm256_storeu_ps(p, r0);
    _mm256_storeu_ps(p.add(8), r1);
}

/// [`layer_norm_row`] over `nb` rows of `dim` floats — the batch form the encoder
/// calls once per norm. A row's bits are [`layer_norm_row`]'s, whatever the batch.
///
/// On x86-64 with AVX2, eight rows are normalised at a time: each row is one lane
/// of an `f64` vector. On aarch64 with NEON it is still eight rows, one lane pair
/// of each `float64x2_t`, so its sum and its variance run in index order with one
/// accumulator per row, and the mean and the inverse scale round once. The
/// scale-and-shift is then the row's `f32` lanes with the same three unfused
/// operations. Both paths are pinned equal by the tests.
///
/// The bits are equal for every finite row. A row that holds an infinity or a NaN
/// produces a NaN on both paths, but the payload of that NaN is not part of the
/// definition: narrowing a NaN from `f64` to `f32` (`cvtsd2ss` against
/// `vcvtpd2ps`) may keep different payload bits. The encoder only feeds finite
/// rows here.
pub fn layer_norm_rows(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    dim: usize,
    nb: usize,
    out: &mut [f32],
) {
    let need = nb
        .checked_mul(dim)
        .expect("layer_norm_rows: row count times width overflows usize");
    assert!(
        x.len() >= need && out.len() >= need,
        "layer_norm_rows: {nb} rows of {dim} need {need} floats, got {} in and {} out",
        x.len(),
        out.len()
    );
    assert!(
        weight.len() >= dim && bias.len() >= dim,
        "layer_norm_rows: a {dim}-wide row needs {dim} weights and biases, got {} and {}",
        weight.len(),
        bias.len()
    );
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 and FMA checked, and the lengths were asserted above.
            unsafe { layer_norm_rows_avx2(x, weight, bias, eps, dim, nb, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if vector_ok() {
            // SAFETY: NEON checked, and the lengths were asserted above.
            unsafe { layer_norm_rows_neon(x, weight, bias, eps, dim, nb, out) };
            return;
        }
    }
    for r in 0..nb {
        layer_norm_row(
            &x[r * dim..(r + 1) * dim],
            weight,
            bias,
            eps,
            &mut out[r * dim..(r + 1) * dim],
        );
    }
}

/// The AVX2 [`layer_norm_rows`]: eight rows per step, one `f64` lane per row.
///
/// # Safety
///
/// AVX2 and FMA must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn layer_norm_rows_avx2(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    dim: usize,
    nb: usize,
    out: &mut [f32],
) {
    let mut r = 0;
    while r + 8 <= nb {
        layer_norm_8_rows(x, weight, bias, eps, dim, r, out);
        r += 8;
    }
    while r < nb {
        layer_norm_row(
            &x[r * dim..(r + 1) * dim],
            weight,
            bias,
            eps,
            &mut out[r * dim..(r + 1) * dim],
        );
        r += 1;
    }
}

/// Eight rows of [`layer_norm_row`], with the reduction walked in index order in
/// eight independent `f64` lanes.
///
/// # Safety
///
/// AVX2 must be available, `r0 + 8 <= nb`, and the slices must cover `nb * dim`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn layer_norm_8_rows(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    dim: usize,
    r0: usize,
    out: &mut [f32],
) {
    let (mean, inv) = layer_norm_8_rows_stats(x, eps, dim, r0);
    for j in 0..8 {
        let mb = _mm256_set1_ps(mean[j]);
        let ib = _mm256_set1_ps(inv[j]);
        let src = x.as_ptr().add((r0 + j) * dim);
        let dst = out.as_mut_ptr().add((r0 + j) * dim);
        let mut k = 0;
        while k + 8 <= dim {
            let xv = _mm256_loadu_ps(src.add(k));
            let t = _mm256_mul_ps(_mm256_sub_ps(xv, mb), ib);
            let wv = _mm256_loadu_ps(weight.as_ptr().add(k));
            let bv = _mm256_loadu_ps(bias.as_ptr().add(k));
            _mm256_storeu_ps(dst.add(k), _mm256_add_ps(_mm256_mul_ps(t, wv), bv));
            k += 8;
        }
        while k < dim {
            let t = (*src.add(k) - mean[j]) * inv[j];
            *dst.add(k) = t * *weight.get_unchecked(k) + *bias.get_unchecked(k);
            k += 1;
        }
    }
}

/// The mean and inverse scale of eight rows: the reduction of
/// [`layer_norm_row`], walked in index order in eight independent `f64` lanes.
///
/// # Safety
///
/// AVX2 must be available, `r0 + 8 <= nb`, and `x` must cover `(r0 + 8) * dim`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn layer_norm_8_rows_stats(
    x: &[f32],
    eps: f32,
    dim: usize,
    r0: usize,
) -> ([f32; 8], [f32; 8]) {
    let base = x.as_ptr().add(r0 * dim);
    let load = |i: usize, lane: usize| *base.add(lane * dim + i) as f64;
    let mut s0 = _mm256_setzero_pd();
    let mut s1 = _mm256_setzero_pd();
    let mut i = 0;
    while i < dim {
        s0 = _mm256_add_pd(
            s0,
            _mm256_set_pd(load(i, 3), load(i, 2), load(i, 1), load(i, 0)),
        );
        s1 = _mm256_add_pd(
            s1,
            _mm256_set_pd(load(i, 7), load(i, 6), load(i, 5), load(i, 4)),
        );
        i += 1;
    }
    let nf = _mm256_set1_pd(dim as f64);
    let m0 = _mm256_div_pd(s0, nf);
    let m1 = _mm256_div_pd(s1, nf);
    // The variance uses the unrounded `m`, and `d·d + var` is a mul and an add,
    // not a fused multiply-add: the definition's two roundings.
    let mut v0 = _mm256_setzero_pd();
    let mut v1 = _mm256_setzero_pd();
    i = 0;
    while i < dim {
        let d0 = _mm256_sub_pd(
            _mm256_set_pd(load(i, 3), load(i, 2), load(i, 1), load(i, 0)),
            m0,
        );
        let d1 = _mm256_sub_pd(
            _mm256_set_pd(load(i, 7), load(i, 6), load(i, 5), load(i, 4)),
            m1,
        );
        v0 = _mm256_add_pd(v0, _mm256_mul_pd(d0, d0));
        v1 = _mm256_add_pd(v1, _mm256_mul_pd(d1, d1));
        i += 1;
    }
    let eps = _mm256_set1_pd(eps as f64);
    let one = _mm256_set1_pd(1.0);
    let inv0 = _mm256_div_pd(
        one,
        _mm256_sqrt_pd(_mm256_add_pd(_mm256_div_pd(v0, nf), eps)),
    );
    let inv1 = _mm256_div_pd(
        one,
        _mm256_sqrt_pd(_mm256_add_pd(_mm256_div_pd(v1, nf), eps)),
    );
    let mut mean = [0.0f32; 8];
    let mut inv = [0.0f32; 8];
    _mm_storeu_ps(mean.as_mut_ptr(), _mm256_cvtpd_ps(m0));
    _mm_storeu_ps(mean.as_mut_ptr().add(4), _mm256_cvtpd_ps(m1));
    _mm_storeu_ps(inv.as_mut_ptr(), _mm256_cvtpd_ps(inv0));
    _mm_storeu_ps(inv.as_mut_ptr().add(4), _mm256_cvtpd_ps(inv1));
    (mean, inv)
}

/// The NEON [`gelu_erf_rows`]: eight elements per step, two `f64` lanes per `erf`
/// (a lane is an element). Every lane does the scalar definition's operations in the
/// scalar definition's order, and a block takes the shortest polynomial path all of
/// its lanes can take, exactly as the AVX2 body does.
///
/// # Safety
///
/// NEON must be available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gelu_erf_rows_neon(x: &mut [f32]) {
    let n = x.len();
    let p = x.as_mut_ptr();
    let mut i = 0;
    while i + 8 <= n {
        gelu_erf_8_neon(p.add(i));
        i += 8;
    }
    while i < n {
        *p.add(i) = gelu_erf(*p.add(i));
        i += 1;
    }
}

/// Eight GELUs: four `float64x2_t` groups, each holding one `erf`'s two lanes.
///
/// # Safety
///
/// NEON must be available, and `p` must point at eight writable floats.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn gelu_erf_8_neon(p: *mut f32) {
    let x0 = vld1q_f32(p);
    let x1 = vld1q_f32(p.add(4));
    let inv = vdupq_n_f32(GELU_INV_SQRT_2);
    let t0 = vmulq_f32(x0, inv);
    let t1 = vmulq_f32(x1, inv);
    let tv = [
        vget_low_f32(t0),
        vget_high_f32(t0),
        vget_low_f32(t1),
        vget_high_f32(t1),
    ];
    // `x.abs() as f64`: the f32 sign bit cleared (a bit operation, so a NaN keeps its
    // payload), then the exact widening.
    let abs_bits = vdup_n_u32(0x7fff_ffff);
    let mut a = [vdupq_n_f64(0.0); 4];
    let mut u = [vdupq_n_f64(0.0); 4];
    for g in 0..4 {
        let av = vreinterpret_f32_u32(vand_u32(vreinterpret_u32_f32(tv[g]), abs_bits));
        a[g] = vcvt_f64_f32(av);
        u[g] = vmulq_f64(a[g], a[g]);
    }
    let mut p10 = [vdupq_n_f64(ERF_SERIES[9]); 4];
    let mut p22 = [vdupq_n_f64(ERF_SERIES[21]); 4];
    let mut p36 = [vdupq_n_f64(ERF_SERIES[35]); 4];
    let mut any_gt1 = false;
    let mut any_gt2 = false;
    for g in 0..4 {
        let mut v = [0.0f64; 2];
        vst1q_f64(v.as_mut_ptr(), a[g]);
        any_gt1 |= v[0] > 1.0 || v[1] > 1.0;
        any_gt2 |= v[0] > 2.0 || v[1] > 2.0;
    }
    if !any_gt1 {
        // Every lane is `a ≤ 1`: 10 terms.
        let mut k = 8usize;
        loop {
            let c = vdupq_n_f64(ERF_SERIES[k]);
            for g in 0..4 {
                p10[g] = vfmaq_f64(c, p10[g], u[g]);
            }
            if k == 0 {
                break;
            }
            k -= 1;
        }
        p22 = p10;
        p36 = p10;
    } else if !any_gt2 {
        // Every lane is `a ≤ 2`: 22 and 10 terms, no 36-term chain.
        let mut k = 20usize;
        loop {
            let c = vdupq_n_f64(ERF_SERIES[k]);
            for g in 0..4 {
                p22[g] = vfmaq_f64(c, p22[g], u[g]);
                if k <= 8 {
                    p10[g] = vfmaq_f64(c, p10[g], u[g]);
                }
            }
            if k == 0 {
                break;
            }
            k -= 1;
        }
        p36 = p22;
    } else {
        // The full split: 36, 22 and 10 terms.
        let mut k = 34usize;
        while k >= 21 {
            let c = vdupq_n_f64(ERF_SERIES[k]);
            for g in 0..4 {
                p36[g] = vfmaq_f64(c, p36[g], u[g]);
            }
            k -= 1;
        }
        while k > 8 {
            let c = vdupq_n_f64(ERF_SERIES[k]);
            for g in 0..4 {
                p36[g] = vfmaq_f64(c, p36[g], u[g]);
                p22[g] = vfmaq_f64(c, p22[g], u[g]);
            }
            k -= 1;
        }
        loop {
            let c = vdupq_n_f64(ERF_SERIES[k]);
            for g in 0..4 {
                p36[g] = vfmaq_f64(c, p36[g], u[g]);
                p22[g] = vfmaq_f64(c, p22[g], u[g]);
                p10[g] = vfmaq_f64(c, p10[g], u[g]);
            }
            if k == 0 {
                break;
            }
            k -= 1;
        }
    }
    let pref = vdupq_n_f64(ERF_TWO_OVER_SQRT_PI);
    let one = vdupq_n_f64(1.0);
    let two = vdupq_n_f64(2.0);
    let mut ef = [vdup_n_f32(0.0); 4];
    for g in 0..4 {
        let le1 = vcleq_f64(a[g], one);
        let gt2 = vcgtq_f64(a[g], two);
        // `x * (2/√π) * P(x²)`, left to right as the definition writes it.
        let two_over = vmulq_f64(a[g], pref);
        let s10 = vmulq_f64(two_over, p10[g]);
        let s22 = vmulq_f64(two_over, p22[g]);
        let s36 = vmulq_f64(two_over, p36[g]);
        // a ≤ 1 takes 10 terms, 1 < a ≤ 2 takes 22, a > 2 takes 36.
        let mut y = vbslq_f64(gt2, s36, s22);
        y = vbslq_f64(le1, s10, y);
        // `3 < a < 4.6`: the scalar asymptotic `erfc`; `a ≥ 4.6`: exactly 1.
        let mut av = [0.0f64; 2];
        vst1q_f64(av.as_mut_ptr(), a[g]);
        if av[0] > 3.0 || av[1] > 3.0 {
            let mut yv = [0.0f64; 2];
            vst1q_f64(yv.as_mut_ptr(), y);
            for j in 0..2 {
                if av[j] > 3.0 {
                    yv[j] = if av[j] < 4.6 {
                        1.0 - erfc_tail(av[j])
                    } else {
                        1.0
                    };
                }
            }
            y = vld1q_f64(yv.as_ptr());
        }
        // Round once, then copy the sign of `t`; a NaN keeps its own bits
        // (the scalar `erf` returns its NaN argument unchanged).
        let mut e = vcvt_f32_f64(y);
        let nan = vmvn_u32(vceq_f32(tv[g], tv[g]));
        e = vbsl_f32(nan, tv[g], e);
        let sign = vdup_n_u32(0x8000_0000);
        let bits = vreinterpret_u32_f32(e);
        let tbits = vand_u32(vreinterpret_u32_f32(tv[g]), sign);
        ef[g] = vreinterpret_f32_u32(vorr_u32(bits, tbits));
    }
    let e01 = vcombine_f32(ef[0], ef[1]);
    let e23 = vcombine_f32(ef[2], ef[3]);
    let half = vdupq_n_f32(0.5);
    let onef = vdupq_n_f32(1.0);
    let r0 = vmulq_f32(vmulq_f32(half, x0), vaddq_f32(onef, e01));
    let r1 = vmulq_f32(vmulq_f32(half, x1), vaddq_f32(onef, e23));
    vst1q_f32(p, r0);
    vst1q_f32(p.add(4), r1);
}

/// The NEON [`layer_norm_rows`]: eight rows per step, one `float64x2_t` lane pair per
/// two rows, so each row still adds into a single `f64` accumulator in index order.
///
/// # Safety
///
/// NEON must be available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn layer_norm_rows_neon(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    dim: usize,
    nb: usize,
    out: &mut [f32],
) {
    let mut r = 0;
    while r + 8 <= nb {
        layer_norm_8_rows_neon(x, weight, bias, eps, dim, r, out);
        r += 8;
    }
    while r < nb {
        layer_norm_row(
            &x[r * dim..(r + 1) * dim],
            weight,
            bias,
            eps,
            &mut out[r * dim..(r + 1) * dim],
        );
        r += 1;
    }
}

/// Eight rows of [`layer_norm_row`], with the reduction walked in index order in
/// four `f64` lane pairs.
///
/// # Safety
///
/// NEON must be available, `r0 + 8 <= nb`, and the slices must cover `nb * dim`.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn layer_norm_8_rows_neon(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    dim: usize,
    r0: usize,
    out: &mut [f32],
) {
    let base = x.as_ptr().add(r0 * dim);
    let load = |i: usize, lane: usize| *base.add(lane * dim + i);
    let pair = |i: usize, g: usize| {
        vcombine_f64(
            vdup_n_f64(load(i, 2 * g) as f64),
            vdup_n_f64(load(i, 2 * g + 1) as f64),
        )
    };
    let mut s = [vdupq_n_f64(0.0); 4];
    let mut i = 0;
    while i < dim {
        for (g, sg) in s.iter_mut().enumerate() {
            *sg = vaddq_f64(*sg, pair(i, g));
        }
        i += 1;
    }
    let nf = vdupq_n_f64(dim as f64);
    let mut m = [vdupq_n_f64(0.0); 4];
    for (g, mg) in m.iter_mut().enumerate() {
        *mg = vdivq_f64(s[g], nf);
    }
    // The variance uses the unrounded `m`, and `d·d + var` is a mul and an add,
    // not a fused multiply-add: the definition's two roundings.
    let mut var = [vdupq_n_f64(0.0); 4];
    i = 0;
    while i < dim {
        for (g, vg) in var.iter_mut().enumerate() {
            let d = vsubq_f64(pair(i, g), m[g]);
            *vg = vaddq_f64(*vg, vmulq_f64(d, d));
        }
        i += 1;
    }
    let eps = vdupq_n_f64(eps as f64);
    let one = vdupq_n_f64(1.0);
    let mut inv = [vdupq_n_f64(0.0); 4];
    for (g, ig) in inv.iter_mut().enumerate() {
        *ig = vdivq_f64(one, vsqrtq_f64(vaddq_f64(vdivq_f64(var[g], nf), eps)));
    }
    let mut mean = [0.0f32; 8];
    let mut scale = [0.0f32; 8];
    for g in 0..4 {
        vst1_f32(mean.as_mut_ptr().add(2 * g), vcvt_f32_f64(m[g]));
        vst1_f32(scale.as_mut_ptr().add(2 * g), vcvt_f32_f64(inv[g]));
    }
    for j in 0..8 {
        let mb = vdupq_n_f32(mean[j]);
        let ib = vdupq_n_f32(scale[j]);
        let src = x.as_ptr().add((r0 + j) * dim);
        let dst = out.as_mut_ptr().add((r0 + j) * dim);
        let mut k = 0;
        while k + 4 <= dim {
            let xv = vld1q_f32(src.add(k));
            let t = vmulq_f32(vsubq_f32(xv, mb), ib);
            let wv = vld1q_f32(weight.as_ptr().add(k));
            let bv = vld1q_f32(bias.as_ptr().add(k));
            vst1q_f32(dst.add(k), vaddq_f32(vmulq_f32(t, wv), bv));
            k += 4;
        }
        while k < dim {
            let t = (*src.add(k) - mean[j]) * scale[j];
            *dst.add(k) = t * *weight.get_unchecked(k) + *bias.get_unchecked(k);
            k += 1;
        }
    }
}

/// LayerNorm of one row, with a defined reduction order.
///
/// ```text
/// m   = (Σ_{i<n} x_i) / n                       f64, index order, one rounding
/// v   = (Σ_{i<n} (x_i − m)²) / n                f64, index order, m unrounded
/// inv = 1 / √(v + eps)                          f64, rounded to f32 once
/// out_i = ((x_i − m_f32) · inv) · w_i + b_i      f32, in that order
/// ```
///
/// The mean and the variance are two independent walks over the row in index
/// order with one `f64` accumulator each (a vector version must add into a
/// single accumulator in index order, not into per-lane partials, or the bits
/// move). `m` is rounded to `f32` once for the shift, while the variance uses
/// the unrounded `m`, which is the two-pass form and the more accurate one.
///
/// The scale-and-shift is three plain `f32` operations, deliberately not fused:
/// `mul` then `add`, so a machine with FMA produces the same bits.
///
/// `eps` is the file's `attention.layer_norm_epsilon`, added before the square
/// root, as llama.cpp's `ggml_norm` does.
///
/// [`layer_norm_rows`] has the AVX2 body: eight rows at a time, one `f64` lane per
/// row, so each row still adds into a single accumulator in index order.
pub fn layer_norm_row(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    debug_assert!(out.len() >= n && weight.len() >= n && bias.len() >= n);
    let mut sum = 0.0f64;
    for &v in &x[..n] {
        sum += v as f64;
    }
    let nf = n as f64;
    let mean64 = sum / nf;
    let mean = mean64 as f32;
    let mut var = 0.0f64;
    for &v in &x[..n] {
        let d = v as f64 - mean64;
        var += d * d;
    }
    let inv = (1.0 / (var / nf + eps as f64).sqrt()) as f32;
    for i in 0..n {
        let t = (x[i] - mean) * inv;
        out[i] = t * weight[i] + bias[i];
    }
}

/// [`layer_norm_row`] with **no bias**, the ModernBERT family's LayerNorm.
///
/// llama.cpp's `build_norm` with a weight and no bias multiplies the centred,
/// scaled row by the weight alone (`ggml_norm` then `ggml_mul`; `src/llama-graph.cpp`).
/// The arithmetic is [`layer_norm_row`]'s without the final add:
///
/// ```text
/// out_i = ((x_i − m_f32) · inv) · w_i             f32, in that order
/// ```
///
/// The absent `+ 0` is not cosmetic: the add turns a `-0.0` product into `+0.0`,
/// so the bias form would give a different bit for a row of negative zeros. This
/// form keeps llama.cpp's `mul` result.
pub fn layer_norm_row_nobias(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    debug_assert!(out.len() >= n && weight.len() >= n);
    let mut sum = 0.0f64;
    for &v in &x[..n] {
        sum += v as f64;
    }
    let nf = n as f64;
    let mean64 = sum / nf;
    let mean = mean64 as f32;
    let mut var = 0.0f64;
    for &v in &x[..n] {
        let d = v as f64 - mean64;
        var += d * d;
    }
    let inv = (1.0 / (var / nf + eps as f64).sqrt()) as f32;
    for i in 0..n {
        out[i] = ((x[i] - mean) * inv) * weight[i];
    }
}

/// [`layer_norm_row_nobias`] over `nb` rows of `dim` floats.
///
/// On x86-64 with AVX2, eight rows are normalised at a time exactly as
/// [`layer_norm_rows`] does it — each row is one `f64` lane of the reduction, the
/// mean and inverse scale round once, and the scale-and-shift is the row's `f32`
/// lanes — but the last step is one multiply by the weight and no add, so a `-0.0`
/// stays a `-0.0`. A row's bits are [`layer_norm_row_nobias`]'s, whatever the batch.
/// On aarch64 with NEON, eight rows are normalised at a time exactly as
/// [`layer_norm_rows`]' NEON body does, and the final step is one multiply by the
/// weight with no add.
pub fn layer_norm_rows_nobias(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    dim: usize,
    nb: usize,
    out: &mut [f32],
) {
    let need = nb
        .checked_mul(dim)
        .expect("layer_norm_rows_nobias: row count times width overflows usize");
    assert!(
        x.len() >= need && out.len() >= need,
        "layer_norm_rows_nobias: {nb} rows of {dim} need {need} floats, got {} in and {} out",
        x.len(),
        out.len()
    );
    assert!(
        weight.len() >= dim,
        "layer_norm_rows_nobias: a {dim}-wide row needs {dim} weights, got {}",
        weight.len()
    );
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 and FMA checked, and the lengths were asserted above.
            unsafe { layer_norm_rows_nobias_avx2(x, weight, eps, dim, nb, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if vector_ok() {
            // SAFETY: NEON checked, and the lengths were asserted above.
            unsafe { layer_norm_rows_nobias_neon(x, weight, eps, dim, nb, out) };
            return;
        }
    }
    for r in 0..nb {
        layer_norm_row_nobias(
            &x[r * dim..(r + 1) * dim],
            weight,
            eps,
            &mut out[r * dim..(r + 1) * dim],
        );
    }
}

/// The AVX2 [`layer_norm_rows_nobias`]: eight rows per step, one `f64` lane per row.
///
/// # Safety
///
/// AVX2 and FMA must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn layer_norm_rows_nobias_avx2(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    dim: usize,
    nb: usize,
    out: &mut [f32],
) {
    let mut r = 0;
    while r + 8 <= nb {
        layer_norm_8_rows_nobias(x, weight, eps, dim, r, out);
        r += 8;
    }
    while r < nb {
        layer_norm_row_nobias(
            &x[r * dim..(r + 1) * dim],
            weight,
            eps,
            &mut out[r * dim..(r + 1) * dim],
        );
        r += 1;
    }
}

/// Eight rows of [`layer_norm_row_nobias`]: [`layer_norm_8_rows_stats`]'s eight
/// lanes, then the one weight multiply the definition ends with — no add.
///
/// # Safety
///
/// AVX2 must be available, `r0 + 8 <= nb`, and the slices must cover `nb * dim`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn layer_norm_8_rows_nobias(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    dim: usize,
    r0: usize,
    out: &mut [f32],
) {
    let (mean, inv) = layer_norm_8_rows_stats(x, eps, dim, r0);
    for j in 0..8 {
        let mb = _mm256_set1_ps(mean[j]);
        let ib = _mm256_set1_ps(inv[j]);
        let src = x.as_ptr().add((r0 + j) * dim);
        let dst = out.as_mut_ptr().add((r0 + j) * dim);
        let mut k = 0;
        while k + 8 <= dim {
            let xv = _mm256_loadu_ps(src.add(k));
            let t = _mm256_mul_ps(_mm256_sub_ps(xv, mb), ib);
            let wv = _mm256_loadu_ps(weight.as_ptr().add(k));
            // One multiply: a `+ 0.0` here would turn a `-0.0` product into `+0.0`.
            _mm256_storeu_ps(dst.add(k), _mm256_mul_ps(t, wv));
            k += 8;
        }
        while k < dim {
            let t = (*src.add(k) - mean[j]) * inv[j];
            *dst.add(k) = t * *weight.get_unchecked(k);
            k += 1;
        }
    }
}

/// The NEON [`layer_norm_rows_nobias`]: eight rows per step, one `float64x2_t` lane
/// pair per two rows, exactly as [`layer_norm_rows_neon`], with the final add gone.
///
/// # Safety
///
/// NEON must be available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn layer_norm_rows_nobias_neon(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    dim: usize,
    nb: usize,
    out: &mut [f32],
) {
    let mut r = 0;
    while r + 8 <= nb {
        layer_norm_8_rows_nobias_neon(x, weight, eps, dim, r, out);
        r += 8;
    }
    while r < nb {
        layer_norm_row_nobias(
            &x[r * dim..(r + 1) * dim],
            weight,
            eps,
            &mut out[r * dim..(r + 1) * dim],
        );
        r += 1;
    }
}

/// Eight rows of [`layer_norm_row_nobias`], walked in index order in four `f64` lane
/// pairs, then one `f32` multiply by the weight per element.
///
/// # Safety
///
/// NEON must be available, `r0 + 8 <= nb`, and the slices must cover `nb * dim`.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn layer_norm_8_rows_nobias_neon(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    dim: usize,
    r0: usize,
    out: &mut [f32],
) {
    let base = x.as_ptr().add(r0 * dim);
    let load = |i: usize, lane: usize| *base.add(lane * dim + i);
    let pair = |i: usize, g: usize| {
        vcombine_f64(
            vdup_n_f64(load(i, 2 * g) as f64),
            vdup_n_f64(load(i, 2 * g + 1) as f64),
        )
    };
    let mut s = [vdupq_n_f64(0.0); 4];
    let mut i = 0;
    while i < dim {
        for (g, sg) in s.iter_mut().enumerate() {
            *sg = vaddq_f64(*sg, pair(i, g));
        }
        i += 1;
    }
    let nf = vdupq_n_f64(dim as f64);
    let mut m = [vdupq_n_f64(0.0); 4];
    for (g, mg) in m.iter_mut().enumerate() {
        *mg = vdivq_f64(s[g], nf);
    }
    // The variance uses the unrounded `m`, and `d·d + var` is a mul and an add,
    // not a fused multiply-add: the definition's two roundings.
    let mut var = [vdupq_n_f64(0.0); 4];
    i = 0;
    while i < dim {
        for (g, vg) in var.iter_mut().enumerate() {
            let d = vsubq_f64(pair(i, g), m[g]);
            *vg = vaddq_f64(*vg, vmulq_f64(d, d));
        }
        i += 1;
    }
    let eps = vdupq_n_f64(eps as f64);
    let one = vdupq_n_f64(1.0);
    let mut inv = [vdupq_n_f64(0.0); 4];
    for (g, ig) in inv.iter_mut().enumerate() {
        *ig = vdivq_f64(one, vsqrtq_f64(vaddq_f64(vdivq_f64(var[g], nf), eps)));
    }
    let mut mean = [0.0f32; 8];
    let mut scale = [0.0f32; 8];
    for g in 0..4 {
        vst1_f32(mean.as_mut_ptr().add(2 * g), vcvt_f32_f64(m[g]));
        vst1_f32(scale.as_mut_ptr().add(2 * g), vcvt_f32_f64(inv[g]));
    }
    for j in 0..8 {
        let mb = vdupq_n_f32(mean[j]);
        let ib = vdupq_n_f32(scale[j]);
        let src = x.as_ptr().add((r0 + j) * dim);
        let dst = out.as_mut_ptr().add((r0 + j) * dim);
        let mut k = 0;
        while k + 4 <= dim {
            let xv = vld1q_f32(src.add(k));
            let t = vmulq_f32(vsubq_f32(xv, mb), ib);
            let wv = vld1q_f32(weight.as_ptr().add(k));
            vst1q_f32(dst.add(k), vmulq_f32(t, wv));
            k += 4;
        }
        while k < dim {
            let t = (*src.add(k) - mean[j]) * scale[j];
            *dst.add(k) = t * *weight.get_unchecked(k);
            k += 1;
        }
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
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_neon() {
            // SAFETY: NEON checked above.
            return unsafe { mul_sigmoid_neon(x, z) };
        }
    }
    for (v, &g) in x.iter_mut().zip(z) {
        *v /= 1.0 + det_exp(-g);
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
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_neon() {
            // SAFETY: NEON checked above.
            return unsafe { mul_silu_neon(x, None) };
        }
    }
    for v in x.iter_mut() {
        *v /= 1.0 + det_exp(-*v);
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
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_neon() {
            // SAFETY: NEON checked above.
            return unsafe { mul_silu_neon(o, Some(z)) };
        }
    }
    for (v, &g) in o.iter_mut().zip(z) {
        *v *= g / (1.0 + det_exp(-g));
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

// ---------------------------------------------------------------------------
// Deterministic f64 transcendentals, for aarch64 and every non-Linux system
// ---------------------------------------------------------------------------
//
// libm's `sin`, `cos`, `pow` and `log` are not one function across C libraries:
// glibc and Apple's libm round their last bits differently. The RoPE tables are
// built from them, every attention head reads those tables, and the difference is
// visible in the logits — which is why #140 had to give macOS its own golden
// directory. The aarch64 tiers use the implementations below instead: they are
// built from IEEE basic operations only, so one golden set holds on Linux and
// macOS alike. Windows is the same (S7.2): it has no golden history to protect and
// its Universal C Runtime is another `libm`, so every non-Linux build uses the
// engine's own functions. Only Linux x86-64 keeps the C library: its goldens were
// recorded against glibc, and the [`det_*`] dispatchers below leave those bits
// alone there.

/// `ln 2`, high part: 32 significant bits, so `k·ln2_hi` is exact for every `k`
/// the `exp` reduction produces.
const DET_LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
/// `ln 2` minus `DET_LN2_HI`.
const DET_LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);
/// `log2(e)`: the `exp` reduction counts whole multiples of `ln 2`.
const DET_LOG2E: f64 = f64::from_bits(0x3ff7_1547_652b_82fe);
/// `2/π`: the sine's quadrant count.
#[cfg(any(target_arch = "aarch64", not(target_os = "linux"), test))]
const DET_FRAC_2_PI: f64 = f64::from_bits(0x3fe4_5f30_6dc9_c883);
/// `π/2`, first part: the low mantissa bits are zero, so `n·PIO2_1` is exact for
/// `|n| < 2²⁰` and the subtraction that reduces the angle is exact as well.
#[cfg(any(target_arch = "aarch64", not(target_os = "linux"), test))]
const DET_PIO2_1: f64 = f64::from_bits(0x3ff9_21fb_5440_0000);
/// `π/2 − DET_PIO2_1`, to `f64`.
#[cfg(any(target_arch = "aarch64", not(target_os = "linux"), test))]
const DET_PIO2_2: f64 = f64::from_bits(0x3dd0_b461_1a62_6331);
/// The rest of `π/2` after `DET_PIO2_2`.
#[cfg(any(target_arch = "aarch64", not(target_os = "linux"), test))]
const DET_PIO2_3: f64 = f64::from_bits(0x3a71_701b_839a_2520);

/// `2^k` for an integer `k`, subnormals included.
fn pow2(k: i32) -> f64 {
    if (-1022..=1023).contains(&k) {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else if k > 1023 {
        f64::INFINITY
    } else if k < -1074 {
        0.0
    } else {
        // Subnormal: `2^k = 2^(k+54) · 2^-54`.
        f64::from_bits(((k + 54 + 1023) as u64) << 52) * f64::from_bits(969u64 << 52)
    }
}

/// `exp(x)`: the reduction `x = k·ln2 + r`, then a degree-14 series on `r`.
///
/// Pure IEEE arithmetic, so it is one function on every tier and every C library.
/// The BERT family's `erf` reads it (the tail of `erfc` is `exp(−x²)`); the
/// [`det_exp`] dispatcher below deliberately keeps libm on Linux x86-64, because
/// the logits goldens recorded there were made against it.
fn exp_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.78 {
        return f64::INFINITY;
    }
    if x < -745.2 {
        return 0.0;
    }
    let k = (x * DET_LOG2E).round();
    let r = (x - k * DET_LN2_HI) - k * DET_LN2_LO;
    // `1 + r + r²/2! + … + r¹⁴/14!`, Horner. `|r| ≤ ln2/2`.
    //
    // One Horner step per line, not a single nested expression: rustfmt's
    // line breaking doubles in cost with every level of nested parentheses,
    // so a fourteen-deep expression takes hours to format.
    let mut p = 1.0 / 87178291200.0;
    p = p * r + 1.0 / 6227020800.0;
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    p * pow2(k as i32)
}

/// `ln(x)`: split `x = m·2^e` with `m ∈ [√2/2, √2]`, then the `atanh` series in
/// `s = (m−1)/(m+1)`, whose `s²` is at most 0.03, so eleven terms reach `f64`.
#[cfg(any(target_arch = "aarch64", not(target_os = "linux"), test))]
fn ln_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return if x > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    let mut a = x;
    let mut e = 0i32;
    if a < f64::MIN_POSITIVE {
        a *= pow2(54);
        e -= 54;
    }
    let bits = a.to_bits();
    e += ((bits >> 52) & 0x7ff) as i32 - 1023;
    a = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if a > std::f64::consts::SQRT_2 {
        a *= 0.5;
        e += 1;
    }
    let s = (a - 1.0) / (a + 1.0);
    let u = s * s;
    // `1 + u/3 + u²/5 + … + u¹⁰/21`, Horner.
    let mut p = 1.0 / 21.0;
    for n in (0..10).rev() {
        p = p * u + 1.0 / (2 * n + 1) as f64;
    }
    2.0 * s * p + e as f64 * DET_LN2_HI + e as f64 * DET_LN2_LO
}

/// `(sin x, cos x)`: Cody–Waite reduction to `|r| ≤ π/4` and the Taylor series.
///
/// The three parts of `π/2` keep `x = n·π/2 + r` accurate out to the largest RoPE
/// angle the engine can build (a few million radians at most); beyond `10⁶`
/// radians, which no table reaches, libm keeps the job.
#[cfg(any(target_arch = "aarch64", not(target_os = "linux"), test))]
fn sin_cos_f64(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    if x.abs() > 1.0e6 {
        return (x.sin(), x.cos());
    }
    let n = (x * DET_FRAC_2_PI).round();
    let mut r = x - n * DET_PIO2_1;
    r -= n * DET_PIO2_2;
    r -= n * DET_PIO2_3;
    let u = r * r;
    // `sin r = r·(1 − u/6 + … + u⁶/13!)`, Horner.
    let mut sp = 1.0 / 6227020800.0;
    sp = sp * u - 1.0 / 39916800.0;
    sp = sp * u + 1.0 / 362880.0;
    sp = sp * u - 1.0 / 5040.0;
    sp = sp * u + 1.0 / 120.0;
    sp = sp * u - 1.0 / 6.0;
    sp = sp * u + 1.0;
    let s = r * sp;
    // `cos r = 1 − u/2 + … + u⁶/12!`, Horner.
    let mut cp = 1.0 / 479001600.0;
    cp = cp * u - 1.0 / 3628800.0;
    cp = cp * u + 1.0 / 40320.0;
    cp = cp * u - 1.0 / 720.0;
    cp = cp * u + 1.0 / 24.0;
    cp = cp * u - 0.5;
    cp = cp * u + 1.0;
    match (n as i64) & 3 {
        0 => (s, cp),
        1 => (cp, -s),
        2 => (-s, -cp),
        _ => (-cp, s),
    }
}

/// `base^exp = exp(exp·ln base)` for `base > 0`; the libm handles the rest.
#[cfg(any(target_arch = "aarch64", not(target_os = "linux"), test))]
fn pow_f64(base: f64, exp: f64) -> f64 {
    if exp == 0.0 {
        return 1.0;
    }
    if exp == 1.0 {
        return base;
    }
    if base > 0.0 && base.is_finite() && exp.is_finite() {
        exp_f64(exp * ln_f64(base))
    } else {
        base.powf(exp)
    }
}

/// `exp` on the forward path.
///
/// On aarch64 and off Linux (Windows x86-64): [`exp_scalar`], the engine's own
/// exponential. On Linux x86-64: the C library's `expf`, so those golden tiers do
/// not move (macOS is aarch64; Windows compares against its own goldens, S7.4).
#[inline]
pub fn det_exp(x: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", not(target_os = "linux")))]
    {
        if x.is_nan() {
            return x;
        }
        if x == f32::INFINITY {
            return x;
        }
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        exp_scalar(x)
    }
    #[cfg(all(not(target_arch = "aarch64"), target_os = "linux"))]
    {
        x.exp()
    }
}

/// `ln` on the forward path; see [`det_exp`] for the platform split.
#[inline]
pub fn det_ln(x: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", not(target_os = "linux")))]
    {
        ln_f64(x as f64) as f32
    }
    #[cfg(all(not(target_arch = "aarch64"), target_os = "linux"))]
    {
        x.ln()
    }
}

/// `ln` of an `f64` on the forward path; see [`det_exp`] for the platform split.
#[inline]
pub fn det_ln_f64(x: f64) -> f64 {
    #[cfg(any(target_arch = "aarch64", not(target_os = "linux")))]
    {
        ln_f64(x)
    }
    #[cfg(all(not(target_arch = "aarch64"), target_os = "linux"))]
    {
        x.ln()
    }
}

/// `tanh` on the forward path: the engine's own [`tanh`] on every platform.
///
/// Unlike its neighbours this one has no platform split. The C library's `tanhf` is the
/// function glibc 2.40 changed, so on x86 the same file gave different bits on two Linux
/// distributions; the functions that still call the C library on Linux x86-64 are ones
/// whose results have not moved between glibc versions.
#[inline]
pub fn det_tanh(x: f32) -> f32 {
    tanh(x)
}

/// `(sin x, cos x)`; see [`det_exp`] for the platform split.
#[inline]
pub fn det_sin_cos(x: f64) -> (f64, f64) {
    #[cfg(any(target_arch = "aarch64", not(target_os = "linux")))]
    {
        sin_cos_f64(x)
    }
    #[cfg(all(not(target_arch = "aarch64"), target_os = "linux"))]
    {
        (x.sin(), x.cos())
    }
}

/// `base^exp` for `f64`; see [`det_exp`] for the platform split.
#[inline]
pub fn det_pow_f64(base: f64, exp: f64) -> f64 {
    #[cfg(any(target_arch = "aarch64", not(target_os = "linux")))]
    {
        pow_f64(base, exp)
    }
    #[cfg(all(not(target_arch = "aarch64"), target_os = "linux"))]
    {
        base.powf(exp)
    }
}

/// `base^exp` for `f32`; see [`det_exp`] for the platform split.
#[inline]
pub fn det_pow_f32(base: f32, exp: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", not(target_os = "linux")))]
    {
        pow_f64(base as f64, exp as f64) as f32
    }
    #[cfg(all(not(target_arch = "aarch64"), target_os = "linux"))]
    {
        base.powf(exp)
    }
}

/// `√(a² + b²)` without the intermediate overflow; see [`det_exp`] for the platform
/// split. The sidecar builder's eigensolver uses it, and a sidecar built at test time
/// feeds a golden, so it is on the numeric path too.
#[inline]
pub fn det_hypot(a: f64, b: f64) -> f64 {
    #[cfg(any(target_arch = "aarch64", not(target_os = "linux")))]
    {
        let (big, small) = if a.abs() >= b.abs() {
            (a.abs(), b.abs())
        } else {
            (b.abs(), a.abs())
        };
        if big == 0.0 {
            return 0.0;
        }
        let r = small / big;
        big * (1.0 + r * r).sqrt()
    }
    #[cfg(all(not(target_arch = "aarch64"), target_os = "linux"))]
    {
        a.hypot(b)
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

    /// The aarch64 vector `exp` is the scalar twin's bits, lane for lane.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_exp4_is_exp_scalar_bit_for_bit() {
        if !crate::cpu::have_neon() {
            eprintln!("skipped: no NEON");
            return;
        }
        let xs = prng(0xA4, 20_000, -130.0, 130.0);
        for chunk in xs.chunks(4) {
            let mut lane = [0.0f32; 4];
            lane[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: NEON checked above.
            let got = unsafe {
                let mut out = [0.0f32; 4];
                vst1q_f32(out.as_mut_ptr(), exp4(vld1q_f32(lane.as_ptr())));
                out
            };
            for (i, &x) in chunk.iter().enumerate() {
                assert_eq!(got[i].to_bits(), exp_scalar(x).to_bits(), "exp({x})");
            }
        }
    }

    /// The NEON activation/softmax vectors against the scalar definition, including
    /// tails and a row split across calls (a worker's block range must not change a
    /// row's bits).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_vmath_matches_the_scalar_definition() {
        if !crate::cpu::have_neon() {
            eprintln!("skipped: no NEON");
            return;
        }
        let xs = prng(0xB7, 1001, -20.0, 20.0);
        for n in [1usize, 2, 3, 4, 5, 7, 8, 15, 16, 17, 64, 1001] {
            let g = &xs[..n];
            let u = prng(0xB8 + n as u64, n, -3.0, 3.0);

            let mut a = g.to_vec();
            // SAFETY: NEON checked above.
            unsafe { swiglu_neon(&mut a, &u) };
            for i in 0..n {
                let want = (g[i] / (1.0 + exp_scalar(-g[i]))) * u[i];
                assert_eq!(a[i].to_bits(), want.to_bits(), "swiglu n={n} i={i}");
            }
            // A split at every offset gives the same row.
            for cut in 1..n {
                let mut s = g.to_vec();
                unsafe {
                    swiglu_neon(&mut s[..cut], &u[..cut]);
                    swiglu_neon(&mut s[cut..], &u[cut..]);
                }
                assert_eq!(s, a, "swiglu split n={n} cut={cut}");
            }

            let mut a = g.to_vec();
            silu(&mut a);
            for i in 0..n {
                let want = g[i] / (1.0 + exp_scalar(-g[i]));
                assert_eq!(a[i].to_bits(), want.to_bits(), "silu n={n} i={i}");
            }

            let mut a = u.clone();
            mul_silu(&mut a, g);
            for i in 0..n {
                let want = u[i] * (g[i] / (1.0 + exp_scalar(-g[i])));
                assert_eq!(a[i].to_bits(), want.to_bits(), "mul_silu n={n} i={i}");
            }

            let mut a = u.clone();
            mul_sigmoid(&mut a, g);
            for i in 0..n {
                let want = u[i] / (1.0 + exp_scalar(-g[i]));
                assert_eq!(a[i].to_bits(), want.to_bits(), "mul_sigmoid n={n} i={i}");
            }

            let mut a = g.to_vec();
            swiglu_oai(&mut a, &u, 1.702, 7.0);
            for i in 0..n {
                let want = swiglu_oai_one(g[i], u[i], 1.702, 7.0);
                assert_eq!(a[i].to_bits(), want.to_bits(), "swiglu_oai n={n} i={i}");
            }

            let mut a = g.to_vec();
            let sa = exp_shift_sum(&mut a, 1.25);
            let mut want = 0.0f64;
            for i in 0..n {
                let v = exp_scalar(g[i] - 1.25);
                assert_eq!(a[i].to_bits(), v.to_bits(), "exp_shift n={n} i={i}");
                want += v as f64;
            }
            assert_eq!(sa.to_bits(), want.to_bits(), "exp_shift sum n={n}");

            let mut a = g.to_vec();
            let sa = exp_scaled_sum(&mut a, -0.5, 0.37);
            let mut want = 0.0f64;
            for i in 0..n {
                let v = exp_scalar((g[i] + 0.5) * 0.37);
                assert_eq!(a[i].to_bits(), v.to_bits(), "exp_scaled n={n} i={i}");
                want += v as f64;
            }
            assert_eq!(sa.to_bits(), want.to_bits(), "exp_scaled sum n={n}");

            let want = g.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(max(g).to_bits(), want.to_bits(), "max n={n}");
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

    /// The deterministic f64 transcendentals are libm-accurate. Their results are
    /// cast to `f32` for the RoPE tables, so a relative error of 1e-12 is four
    /// orders of magnitude below the format's resolution.
    #[test]
    fn deterministic_f64_transcendentals_match_libm() {
        // sin/cos across the angles a small-context table builds ...
        let mut x = -300.0f64;
        while x <= 300.0 {
            let (s, c) = sin_cos_f64(x);
            assert!((s - x.sin()).abs() < 1e-12, "sin({x}): {s} vs {}", x.sin());
            assert!((c - x.cos()).abs() < 1e-12, "cos({x}): {c} vs {}", x.cos());
            x += 0.017_3;
        }
        // ... and the largest angles a long-context table reaches.
        for &x in &[4095.7f64, 16_384.3, 131_071.9, -65_535.5, 524_287.25] {
            let (s, c) = sin_cos_f64(x);
            assert!((s - x.sin()).abs() < 1e-11, "sin({x}): {s} vs {}", x.sin());
            assert!((c - x.cos()).abs() < 1e-11, "cos({x}): {c} vs {}", x.cos());
        }
        // pow: the RoPE bases, every exponent a table uses.
        for &base in &[10_000.0f64, 150_000.0, 500_000.0, 1_000_000.0] {
            for i in 0..128 {
                let e = -((2 * i) as f64) / 128.0;
                let got = pow_f64(base, e);
                let want = base.powf(e);
                assert!(
                    ((got - want) / want).abs() < 1e-12,
                    "{base}^{e}: {got} vs {want}"
                );
            }
        }
        // exp and ln over the range the reduction handles.
        let mut x = -700.0f64;
        while x <= 700.0 {
            let e = exp_f64(x);
            let want = x.exp();
            assert!(((e - want) / want).abs() < 1e-12, "exp({x}): {e} vs {want}");
            x += 3.7;
        }
        let mut x = 1e-6f64;
        while x < 1e6 {
            let l = ln_f64(x);
            let want = x.ln();
            assert!(
                (l - want).abs() < 1e-13 * want.abs().max(1.0),
                "ln({x}): {l} vs {want}"
            );
            x *= 1.37;
        }
        // The corners the forward path actually meets.
        assert_eq!(det_exp(f32::NEG_INFINITY), 0.0);
        assert_eq!(det_exp(f32::INFINITY), f32::INFINITY);
        assert!(det_exp(f32::NAN).is_nan());
        assert_eq!(ln_f64(1.0), 0.0);
        assert_eq!(ln_f64(0.0), f64::NEG_INFINITY);
        assert!(ln_f64(-1.0).is_nan());
        assert_eq!(pow_f64(2.0, 0.0), 1.0);
        assert_eq!(pow_f64(2.0, 1.0), 2.0);
    }

    /// An oracle for `erf` that shares no arithmetic with the implementation:
    ///
    /// ```text
    /// erf(x) = (2/√π) · exp(−x²) · Σ_n (2x²)^n·x / (1·3·5···(2n+1))
    /// ```
    ///
    /// every term positive, so there is no cancellation at any `x`; it is summed
    /// until a term is below `1e-17` of the sum. `exp` is the C library's, not the
    /// module's, so the check does not share that implementation either.
    fn erf_oracle(x: f64) -> f64 {
        if x == 0.0 {
            return 0.0;
        }
        let x2 = 2.0 * x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 1.0f64;
        loop {
            term *= x2 / (2.0 * n + 1.0);
            sum += term;
            if term.abs() <= sum.abs() * 1e-18 {
                break;
            }
            n += 1.0;
            assert!(n < 10_000.0);
        }
        2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp() * sum
    }

    fn ulp_distance(a: f32, b: f32) -> u32 {
        a.to_bits().abs_diff(b.to_bits())
    }

    #[test]
    fn erf_agrees_with_an_independent_oracle() {
        let mut max_abs = 0.0f64;
        let mut max_ulp = 0u32;
        let mut at = 0.0f64;
        let n = 400_000;
        for i in 0..=n {
            let x = -6.0 + 12.0 * i as f64 / n as f64;
            let want = erf_oracle(x);
            let got = erf(x as f32) as f64;
            let e = (got - want).abs();
            if e > max_abs {
                max_abs = e;
                at = x;
            }
            max_ulp = max_ulp.max(ulp_distance(erf(x as f32), want as f32));
        }
        eprintln!("erf: max abs {max_abs:.3e} at {at}, max {max_ulp} ulp");
        // The definition's stated bound: one f32 ulp of the result, everywhere.
        // (Measured: 5.44e-8 at |erf| ≈ 0.5 — that is the final rounding, not the
        // series; the series itself is under 3e-9 absolute.)
        assert!(max_abs <= 6e-8, "max abs error {max_abs:.3e} at {at}");
        assert!(max_ulp <= 1, "max {max_ulp} ulp at {at}");
        // The corners, and the odd symmetry.
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(6.0), 1.0);
        assert_eq!(erf(-6.0), -1.0);
        assert_eq!(erf(f32::INFINITY), 1.0);
        assert_eq!(erf(f32::NEG_INFINITY), -1.0);
        assert!(erf(f32::NAN).is_nan());
        for i in 0..2000 {
            let x = -5.0 + 10.0 * i as f32 / 2000.0;
            assert_eq!(erf(-x), -erf(x), "odd symmetry at {x}");
        }
    }

    #[test]
    fn gelu_erf_is_half_x_one_plus_erf_of_x_over_root_two() {
        let mut max_abs = 0.0f64;
        let mut at = 0.0f64;
        // The rows path must be the single-element function element by element.
        let mut row: Vec<f32> = (0..1001)
            .map(|i| -12.0 + 24.0 * i as f32 / 1000.0)
            .collect();
        let single: Vec<f32> = row.iter().map(|&x| gelu_erf(x)).collect();
        gelu_erf_rows(&mut row);
        assert_eq!(
            row.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            single.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the row loop must not change an element's bits"
        );
        for (i, &x) in row.iter().enumerate() {
            let xf = -12.0 + 24.0 * i as f64 / 1000.0;
            let want = 0.5 * xf * (1.0 + erf_oracle(xf / 2.0f64.sqrt()));
            let e = (x as f64 - want).abs();
            if e > max_abs {
                max_abs = e;
                at = xf;
            }
        }
        eprintln!("gelu_erf: max abs {max_abs:.3e} at {at}");
        // Three f32 roundings of a value up to 12: under three ulps of the result,
        // which at the measured point (|gelu| ≈ 5) is 1.4e-6.
        assert!(max_abs < 2e-6, "max abs error {max_abs:.3e} at {at}");
        // Saturation: the two tails are exactly x and 0 at f32 precision.
        assert_eq!(gelu_erf(0.0), 0.0);
        assert_eq!(gelu_erf(12.0), 12.0);
        assert_eq!(gelu_erf(-12.0), 0.0);
    }

    #[test]
    fn layer_norm_row_is_the_defined_two_pass_norm() {
        // A hand case: mean 2.5, population variance 1.25.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32; 4];
        let b = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        layer_norm_row(&x, &w, &b, 1e-5, &mut out);
        // The scale is one f32 rounding of `1/sqrt(var + eps)`, then `(x − m)·inv`
        // in f32 — the definition's order.
        let inv = (1.0 / (1.25f64 + 1e-5).sqrt()) as f32;
        for (i, &xi) in x.iter().enumerate() {
            let want = (xi - 2.5) * inv;
            assert_eq!(out[i].to_bits(), want.to_bits(), "{i}");
        }
        // The affine part is applied in f32 after the scale, as the definition says.
        let w = [2.0f32, -1.0, 0.5, 3.0];
        let b = [0.25f32, -0.5, 1.0, 0.0];
        let inv = (1.0 / 1.25f64.sqrt()) as f32;
        layer_norm_row(&x, &w, &b, 0.0, &mut out);
        for i in 0..4 {
            let want = (x[i] - 2.5) * inv * w[i] + b[i];
            assert_eq!(out[i].to_bits(), want.to_bits(), "{i}");
        }
        // A row with a constant value is all bias.
        let flat = [7.0f32; 3];
        let w = [1.0f32; 3];
        let b = [0.0f32; 3];
        let mut out = [1.0f32; 3];
        layer_norm_row(&flat, &w, &b, 1e-5, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    fn next_bits(f: f32, up: bool) -> f32 {
        let b = f.to_bits();
        let b = if up {
            if f >= 0.0 {
                b + 1
            } else {
                b - 1
            }
        } else if f >= 0.0 {
            b - 1
        } else {
            b + 1
        };
        f32::from_bits(b)
    }

    /// The classes every `erf` branch is defined on, and their neighbours.
    fn erf_boundary_inputs() -> Vec<f32> {
        let mut v = vec![
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            3.0,
            -3.0,
            4.6,
            -4.6,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_1234), // qNaN, payload
            f32::from_bits(0xffc0_5678),
            f32::from_bits(0x7f80_0001), // sNaN
            f32::from_bits(0xff80_0002),
        ];
        // The GELU's argument boundaries sit at x = ±k·√2 for erf's k ∈ {1,2,3,4.6}.
        for k in [1.0f32, 2.0, 3.0, 4.6] {
            for x in [
                k * std::f32::consts::SQRT_2,
                -(k * std::f32::consts::SQRT_2),
            ] {
                v.push(x);
                v.push(next_bits(x, true));
                v.push(next_bits(x, false));
            }
        }
        v
    }

    #[test]
    fn gelu_erf_rows_matches_the_scalar_twin() {
        // 1 000 000 random bit patterns: every exponent, both signs, subnormals,
        // infinities and NaNs (the classes the definition splits on).
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rand = Vec::with_capacity(1_000_000);
        for _ in 0..1_000_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            rand.push(f32::from_bits((s >> 32) as u32));
        }
        let mut got = rand.clone();
        gelu_erf_rows(&mut got);
        for (i, (&x, &g)) in rand.iter().zip(&got).enumerate() {
            assert_eq!(
                g.to_bits(),
                gelu_erf(x).to_bits(),
                "random input {i}: {x:?} ({:#010x}) became {g:?}, scalar {:?}",
                x.to_bits(),
                gelu_erf(x)
            );
        }
        // The same values through the scalar row loop (the twin), too.
        let mut scalar = rand.clone();
        gelu_erf_rows_scalar(&mut scalar);
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            scalar.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the vector and scalar row loops disagree"
        );
        // The branch boundaries, their neighbours, ±0, subnormals and NaN payloads.
        let edges = erf_boundary_inputs();
        let mut row = edges.clone();
        gelu_erf_rows(&mut row);
        for (i, (&x, &g)) in edges.iter().zip(&row).enumerate() {
            assert_eq!(g.to_bits(), gelu_erf(x).to_bits(), "edge {i}: {x:?}");
        }
        // Row lengths 1..=40 exercise every remainder of the 16-wide body.
        for n in 1..=40usize {
            let mut row: Vec<f32> = rand[..n].to_vec();
            gelu_erf_rows(&mut row);
            for (i, &g) in row.iter().enumerate() {
                assert_eq!(
                    g.to_bits(),
                    gelu_erf(rand[i]).to_bits(),
                    "len {n}, index {i}"
                );
            }
            // The curated edges at every remainder too.
            let mut row: Vec<f32> = (0..n).map(|i| edges[i % edges.len()]).collect();
            let want: Vec<f32> = row.iter().map(|&x| gelu_erf(x)).collect();
            gelu_erf_rows(&mut row);
            assert_eq!(
                row.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "len {n} on the edges"
            );
        }
    }

    #[test]
    fn layer_norm_rows_matches_the_scalar_twin() {
        // The comparison the definition promises: every finite result is the same
        // bits; a NaN in (or made by an infinity) is a NaN on both paths, and its
        // payload is not part of the definition — `cvtsd2ss` and `vcvtpd2ps` may
        // pick different payload bits when they narrow a NaN.
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        let mut s = 0x1234_5678_9abc_def1u64;
        let mut rand = Vec::with_capacity(1_000_000 + 512);
        for _ in 0..1_000_000 + 512 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            rand.push(f32::from_bits((s >> 32) as u32));
        }
        let weight = prng(3, 512, -2.0, 2.0);
        let bias = prng(4, 512, -1.0, 1.0);
        // Every remainder: 40 widths, each with two batches of rows.
        for dim in 1..=40usize {
            let nb = 16;
            let x = &rand[..nb * dim];
            let w = &weight[..dim];
            let b = &bias[..dim];
            let mut got = vec![0.0f32; nb * dim];
            let mut want = vec![0.0f32; nb * dim];
            layer_norm_rows(x, w, b, 1e-12, dim, nb, &mut got);
            for r in 0..nb {
                layer_norm_row(
                    &x[r * dim..(r + 1) * dim],
                    w,
                    b,
                    1e-12,
                    &mut want[r * dim..(r + 1) * dim],
                );
            }
            for i in 0..nb * dim {
                assert!(
                    same(got[i], want[i]),
                    "dim {dim}, row {}: {:#010x} vs {:#010x}",
                    i / dim,
                    got[i].to_bits(),
                    want[i].to_bits()
                );
            }
        }
        // One million values in rows of 13 (a remainder of the eight-row body) and
        // of 40 (also a remainder, and a multiple of eight).
        for dim in [13usize, 40] {
            let nb = (1_000_000 + 511) / dim;
            let x = &rand[..nb * dim];
            let w = &weight[..dim];
            let b = &bias[..dim];
            let mut got = vec![0.0f32; nb * dim];
            layer_norm_rows(x, w, b, 1e-5, dim, nb, &mut got);
            for r in 0..nb {
                let mut want = vec![0.0f32; dim];
                layer_norm_row(&x[r * dim..(r + 1) * dim], w, b, 1e-5, &mut want);
                for i in 0..dim {
                    assert!(
                        same(got[r * dim + i], want[i]),
                        "dim {dim}, row {r}, index {i}: {:#010x} vs {:#010x}",
                        got[r * dim + i].to_bits(),
                        want[i].to_bits()
                    );
                }
            }
        }
        // The classes: a flat row (all bias), ±0 and subnormals, huge values, and
        // rows holding infinities and NaNs (the bits must agree).
        let cases: Vec<Vec<f32>> = vec![
            vec![0.0; 8],
            vec![-0.0; 8],
            vec![f32::from_bits(1); 8],
            vec![-f32::from_bits(2); 8],
            vec![1e30; 8],
            vec![-1e30, 1e30, 0.0, 1.0, -1.0, 1e-30, 0.0, 2.0],
            vec![f32::INFINITY; 8],
            vec![
                1.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                0.0,
                1.0,
                -1.0,
                2.0,
            ],
            vec![f32::from_bits(0x7fc0_1234); 8],
        ];
        for (c, row) in cases.iter().enumerate() {
            let dim = row.len();
            let w = vec![1.0f32; dim];
            let b = vec![0.0f32; dim];
            let mut x = row.clone();
            x.extend_from_slice(row); // two rows: the eight-row body needs a tail path here
            let mut got = vec![0.0f32; 2 * dim];
            let mut want = vec![0.0f32; 2 * dim];
            layer_norm_rows(&x, &w, &b, 1e-12, dim, 2, &mut got);
            for r in 0..2 {
                layer_norm_row(
                    &x[r * dim..(r + 1) * dim],
                    &w,
                    &b,
                    1e-12,
                    &mut want[r * dim..(r + 1) * dim],
                );
            }
            for i in 0..2 * dim {
                assert!(
                    same(got[i], want[i]),
                    "case {c}, index {i}: {:#010x} vs {:#010x}",
                    got[i].to_bits(),
                    want[i].to_bits()
                );
            }
        }
    }

    /// The no-bias LayerNorm is the bias form with the add removed, bit for bit
    /// for every row whose products are not zeros; where the product is a zero the
    /// two differ in the sign, and this form keeps the product's sign.
    #[test]
    fn layer_norm_rows_nobias_is_the_bias_form_without_the_add() {
        let dim = 37usize;
        let nb = 11usize;
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let x: Vec<f32> = (0..nb * dim)
            .map(|_| (next() as i32 as f32) / (1u32 << 20) as f32)
            .collect();
        let w: Vec<f32> = (0..dim)
            .map(|_| 0.5 + (next() % 1000) as f32 / 500.0)
            .collect();
        let zero_bias = vec![0.0f32; dim];
        for eps in [1e-12f32, 1e-5] {
            let mut got = vec![0.0f32; nb * dim];
            layer_norm_rows_nobias(&x, &w, eps, dim, nb, &mut got);
            let mut want = vec![0.0f32; nb * dim];
            layer_norm_rows(&x, &w, &zero_bias, eps, dim, nb, &mut want);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "eps {eps}"
            );
        }
        // A row of negative zeros: centring gives -0.0 and the weight is positive,
        // so the product is -0.0; the bias form's `+ 0.0` turns it into +0.0.
        let x = vec![-0.0f32; dim];
        let mut got = vec![0.0f32; dim];
        let mut biased = vec![0.0f32; dim];
        layer_norm_rows_nobias(&x, &w, 1e-5, dim, 1, &mut got);
        layer_norm_rows(&x, &w, &zero_bias, 1e-5, dim, 1, &mut biased);
        assert!(got.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()));
        assert!(biased.iter().all(|v| v.to_bits() == 0.0f32.to_bits()));
    }

    /// `geglu_erf` is exactly `gelu_erf` on the first half times the second half.
    #[test]
    fn geglu_erf_is_the_exact_gelu_times_the_second_half() {
        let n = 4097usize;
        let mut state = 0x1319_8a2e_0370_7344u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let values: Vec<f32> = (0..n)
            .map(|_| ((next() % 20_001) as f32 - 10_000.0) / 500.0)
            .collect();
        let ups: Vec<f32> = (0..n)
            .map(|_| ((next() % 20_001) as f32 - 10_000.0) / 3000.0)
            .collect();
        let mut got = values.clone();
        geglu_erf(&mut got, &ups);
        for i in 0..n {
            let want = gelu_erf(values[i]) * ups[i];
            assert_eq!(got[i].to_bits(), want.to_bits(), "index {i}");
        }
    }

    /// Every `geglu_erf` element is the scalar `gelu_erf` times `up`, bit for bit:
    /// the AVX2 body's sixteen lanes, its scalar tail, and every class of input.
    #[test]
    fn geglu_erf_matches_the_scalar_twin() {
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        // `erf_boundary_inputs` has ±0, subnormals, the branch boundaries, ±inf and
        // NaNs with payloads; add the large magnitudes and the GELU's own range.
        let mut edges = erf_boundary_inputs();
        edges.extend([
            1e30f32,
            -1e30,
            1e-30,
            -1e-30,
            12.0,
            -12.0,
            f32::from_bits(0x007f_ffff),
            f32::from_bits(0x807f_ffff),
        ]);
        let sweep = prng(11, 2304, -12.0, 12.0);
        let lens = [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 33, 768, 1152, 2304];
        for &n in &lens {
            // The edges first, so every block covers the branch boundaries, then the
            // dense sweep.
            let gate: Vec<f32> = (0..n)
                .map(|i| if i < edges.len() { edges[i] } else { sweep[i] })
                .collect();
            let up: Vec<f32> = (0..n)
                .map(|i| {
                    if i < edges.len() {
                        -edges[edges.len() - 1 - i]
                    } else {
                        sweep[(i * 7 + 3) % sweep.len()]
                    }
                })
                .collect();
            let want: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| gelu_erf(g) * u)
                .collect();
            let mut got = gate.clone();
            geglu_erf(&mut got, &up);
            assert_eq!(got.len(), want.len());
            for i in 0..n {
                assert!(
                    same(got[i], want[i]),
                    "len {n}, index {i}: {:#010x} vs {:#010x}",
                    got[i].to_bits(),
                    want[i].to_bits()
                );
            }
        }
        // 200 000 values of [-12, 12]: the sweep the acceptance names, carried by the
        // sixteen-wide body and one tail rather than by the edges.
        let n = 200_000usize;
        let gate = prng(12, n, -12.0, 12.0);
        let up = prng(13, n, -12.0, 12.0);
        let want: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| gelu_erf(g) * u)
            .collect();
        let mut got = gate.clone();
        geglu_erf(&mut got, &up);
        for i in 0..n {
            assert!(
                same(got[i], want[i]),
                "sweep {i}: {:#010x} vs {:#010x}",
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
    }

    /// The edge rows again, but nine of them at once: eight go through the widest row
    /// body the tier has and the ninth through the row tail, and each must still be the
    /// scalar twin's row. (One row alone only ever reaches the tail.)
    #[test]
    fn layer_norm_rows_nobias_edge_rows_match_inside_a_batch() {
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        let rows: Vec<Vec<f32>> = vec![
            vec![1e30; 13],
            vec![
                -1e30, 1e30, 0.0, -0.0, 1.0, -1.0, 1e-30, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ],
            vec![f32::INFINITY; 13],
            vec![f32::from_bits(1); 13],
            vec![-f32::from_bits(2); 13],
            vec![
                1.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                0.0,
                -0.0,
                1.0,
                -1.0,
                2.0,
                1e30,
                -1e30,
                3.0,
                4.0,
            ],
            vec![f32::from_bits(0x7fc0_1234); 13],
            vec![-0.0; 13],
            vec![
                0.25, -0.5, 1.5, -2.5, 3.5, -4.5, 5.5, -6.5, 7.5, -8.5, 9.5, -10.5, 11.5,
            ],
        ];
        let dim = 13usize;
        let nb = rows.len();
        assert_eq!(nb, 9);
        let weights = prng(31, dim, -2.0, 2.0);
        // Every rotation, so each edge row sits in every lane of the eight-row body
        // and takes its turn as the tail row.
        for shift in 0..nb {
            let mut x = Vec::with_capacity(nb * dim);
            for r in 0..nb {
                x.extend_from_slice(&rows[(r + shift) % nb]);
            }
            let mut got = vec![0.0f32; nb * dim];
            layer_norm_rows_nobias(&x, &weights, 1e-12, dim, nb, &mut got);
            for r in 0..nb {
                let mut want = vec![0.0f32; dim];
                layer_norm_row_nobias(&x[r * dim..(r + 1) * dim], &weights, 1e-12, &mut want);
                for i in 0..dim {
                    assert!(
                        same(got[r * dim + i], want[i]),
                        "shift {shift}, row {r}, index {i}: {:#010x} vs {:#010x}",
                        got[r * dim + i].to_bits(),
                        want[i].to_bits()
                    );
                }
            }
        }
    }

    /// Every `layer_norm_rows_nobias` row is the scalar twin's row, bit for bit:
    /// the eight-row AVX2 body, its scalar row tail, and widths on both sides of
    /// eight.
    #[test]
    fn layer_norm_rows_nobias_matches_the_scalar_twin() {
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        let weights = prng(21, 2304, -2.0, 2.0);
        let dims = [1usize, 7, 8, 9, 15, 16, 17, 31, 33, 37, 40, 768, 1152, 2304];
        for (c, &dim) in dims.iter().enumerate() {
            let w = &weights[..dim];
            for &nb in &[0usize, 1, 2, 7, 8, 9, 16, 17] {
                let x = prng(22 + c as u64, nb * dim, -12.0, 12.0);
                let mut got = vec![0.0f32; nb * dim];
                let mut want = vec![0.0f32; nb * dim];
                layer_norm_rows_nobias(&x, w, 1e-12, dim, nb, &mut got);
                for r in 0..nb {
                    layer_norm_row_nobias(
                        &x[r * dim..(r + 1) * dim],
                        w,
                        1e-12,
                        &mut want[r * dim..(r + 1) * dim],
                    );
                }
                for i in 0..nb * dim {
                    assert!(
                        same(got[i], want[i]),
                        "dim {dim}, nb {nb}, row {}: {:#010x} vs {:#010x}",
                        i / dim,
                        got[i].to_bits(),
                        want[i].to_bits()
                    );
                }
            }
        }
        // A row of negative zeros keeps the product's sign: the vector body's one
        // multiply, and no `+ 0.0`. The weights are positive, as in the bias form's
        // test, so the product is `-0.0`.
        let dim = 37usize;
        let nb = 9usize;
        let w = &weights[..dim];
        let w_pos: Vec<f32> = w.iter().map(|v| v.abs() + 0.5).collect();
        let x = vec![-0.0f32; nb * dim];
        let mut got = vec![0.0f32; nb * dim];
        layer_norm_rows_nobias(&x, &w_pos, 1e-5, dim, nb, &mut got);
        assert!(
            got.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()),
            "a row of -0.0 must stay -0.0"
        );
        // A constant row is all zeros.
        let x = vec![7.0f32; nb * dim];
        let mut got = vec![1.0f32; nb * dim];
        layer_norm_rows_nobias(&x, &w_pos, 1e-5, dim, nb, &mut got);
        assert!(got.iter().all(|&v| v == 0.0));
        // Huge magnitudes, infinities, NaNs and their payloads: the same comparison
        // the bias form's test makes (a NaN is a NaN on both paths; its payload is
        // not part of the definition).
        let cases: Vec<Vec<f32>> = vec![
            vec![1e30; 13],
            vec![
                -1e30, 1e30, 0.0, -0.0, 1.0, -1.0, 1e-30, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ],
            vec![f32::INFINITY; 13],
            vec![f32::from_bits(1); 13],
            vec![-f32::from_bits(2); 13],
            vec![
                1.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                0.0,
                -0.0,
                1.0,
                -1.0,
                2.0,
                1e30,
                -1e30,
                3.0,
                4.0,
            ],
            vec![f32::from_bits(0x7fc0_1234); 13],
        ];
        for (c, row) in cases.iter().enumerate() {
            let dim = row.len();
            let w = &weights[..dim];
            let x = row.clone();
            let mut got = vec![0.0f32; dim];
            let mut want = vec![0.0f32; dim];
            layer_norm_rows_nobias(&x, w, 1e-12, dim, 1, &mut got);
            layer_norm_row_nobias(&x, w, 1e-12, &mut want);
            for i in 0..dim {
                assert!(
                    same(got[i], want[i]),
                    "case {c}, index {i}: {:#010x} vs {:#010x}",
                    got[i].to_bits(),
                    want[i].to_bits()
                );
            }
        }
    }

    /// `NaN` compares equal to `NaN` (its payload is not part of a definition);
    /// every other result must match bit for bit.
    fn same_bits(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// The input classes the element-wise kernels must keep: both zeros, the
    /// subnormals on either side of zero, the `f32` extremes, `±1e30`, the
    /// infinities and qNaN/sNaN payloads.
    fn elementwise_edges() -> Vec<f32> {
        let mut v = erf_boundary_inputs();
        v.extend([1e30f32, -1e30, 1e30, -1e30]);
        v
    }

    /// One gate row and one up row: 100 000 pseudo-random values in `[-12, 12]`
    /// (the range the GELU is asked for), with the edge classes scattered through
    /// both ends.
    fn elementwise_sweep() -> (Vec<f32>, Vec<f32>) {
        let n = 100_000usize;
        let mut gate = prng(0x51ed_270b, n, -12.0, 12.0);
        let mut up = prng(0x0bad_c0de, n, -12.0, 12.0);
        let edges = elementwise_edges();
        for (i, &e) in edges.iter().enumerate() {
            gate[i] = e;
            up[i] = edges[(i * 5 + 1) % edges.len()];
            gate[n - 1 - i] = edges[(i * 3 + 2) % edges.len()];
            up[n - 1 - i] = e;
        }
        (gate, up)
    }

    fn check_geglu(gate: &[f32], up: &[f32], what: &str, run: impl Fn(&mut [f32], &[f32])) {
        let mut got = gate.to_vec();
        run(&mut got, up);
        for i in 0..gate.len() {
            let want = gelu_erf(gate[i]) * up[i];
            assert!(
                same_bits(got[i], want),
                "{what}: index {i} of {}: gate {:?} ({:#010x}), up {:?} ({:#010x}): \
                 got {:#010x}, scalar {:#010x}",
                gate.len(),
                gate[i],
                gate[i].to_bits(),
                up[i],
                up[i].to_bits(),
                got[i].to_bits(),
                want.to_bits()
            );
        }
    }

    /// The two degenerate lengths and every remainder of the eight-wide body.
    const ELEMENTWISE_LENGTHS: [usize; 15] =
        [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 768, 1152, 2304];

    #[test]
    fn geglu_erf_dispatch_matches_the_scalar_loop_on_edges_and_a_sweep() {
        let (gate, up) = elementwise_sweep();
        check_geglu(&gate, &up, "the 100 000-value sweep", geglu_erf);
        for n in ELEMENTWISE_LENGTHS {
            check_geglu(&gate[..n], &up[..n], &format!("length {n}"), geglu_erf);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn geglu_erf_neon_is_bit_equal_to_the_scalar_loop() {
        if !vector_ok() {
            eprintln!("geglu_erf NEON: no NEON under this tier cap; nothing to compare");
            return;
        }
        let (gate, up) = elementwise_sweep();
        check_geglu(&gate, &up, "the 100 000-value sweep", |g, u| unsafe {
            geglu_erf_neon(g, u)
        });
        for n in ELEMENTWISE_LENGTHS {
            check_geglu(
                &gate[..n],
                &up[..n],
                &format!("length {n}"),
                |g, u| unsafe { geglu_erf_neon(g, u) },
            );
        }
    }

    fn check_layer_norm_nobias(
        x: &[f32],
        weight: &[f32],
        eps: f32,
        dim: usize,
        nb: usize,
        what: &str,
        run: impl Fn(&[f32], &[f32], f32, usize, usize, &mut [f32]),
    ) {
        let mut got = vec![0.0f32; nb * dim];
        run(x, weight, eps, dim, nb, &mut got);
        let mut want = vec![0.0f32; nb * dim];
        for r in 0..nb {
            layer_norm_row_nobias(
                &x[r * dim..(r + 1) * dim],
                weight,
                eps,
                &mut want[r * dim..(r + 1) * dim],
            );
        }
        for i in 0..nb * dim {
            assert!(
                same_bits(got[i], want[i]),
                "{what}: dim {dim}, row {}, index {}: got {:#010x}, scalar {:#010x}",
                i / dim.max(1),
                i % dim.max(1),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
    }

    /// Every width the two bodies special-case, several row counts (the eight-row
    /// body and its tail), and the rows whose bits the definition distinguishes:
    /// all `-0.0`, a constant, `±1e30`, subnormals, infinities and NaN payloads.
    /// One case of the no-bias norm: a label, the input, the weight, `eps`, the
    /// row count and the width.
    type LayerNormCase = (String, Vec<f32>, Vec<f32>, f32, usize, usize);

    fn layer_norm_nobias_cases() -> Vec<LayerNormCase> {
        let pool = prng(0x2545_f491, 11 * 2304, -17.0, 17.0);
        let w = prng(0xdead_beef, 2304, 0.25, 2.0);
        let mut cases = Vec::new();
        for dim in ELEMENTWISE_LENGTHS {
            for nb in [1usize, 3, 10] {
                for eps in [1e-5f32, 1e-12] {
                    cases.push((
                        format!("dim {dim}, {nb} rows, eps {eps}"),
                        pool[..nb * dim].to_vec(),
                        w[..dim].to_vec(),
                        eps,
                        nb,
                        dim,
                    ));
                }
            }
        }
        let rows: Vec<(String, Vec<f32>)> = vec![
            ("a row of all -0.0".into(), vec![-0.0f32; 768]),
            ("a constant row".into(), vec![7.0f32; 768]),
            ("a row of 1e30".into(), vec![1e30f32; 768]),
            ("a row of subnormals".into(), vec![f32::from_bits(1); 768]),
            (
                "a row holding both infinities".into(),
                [f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0].repeat(192),
            ),
            (
                "a row holding a NaN payload".into(),
                vec![f32::from_bits(0x7fc0_1234); 768],
            ),
            (
                "a 37-wide dim (not a multiple of 4)".into(),
                prng(0x37, 37, -12.0, 12.0),
            ),
            (
                "a 1151-wide dim (not a multiple of 4)".into(),
                prng(0x1151, 1151, -12.0, 12.0),
            ),
        ];
        for (label, row) in rows {
            let dim = row.len();
            let mut x = Vec::with_capacity(10 * dim);
            for _ in 0..10 {
                x.extend_from_slice(&row);
            }
            cases.push((label, x, w[..dim].to_vec(), 1e-12, 10, dim));
        }
        cases
    }

    #[test]
    fn layer_norm_rows_nobias_dispatch_matches_the_scalar_rows() {
        for (what, x, w, eps, nb, dim) in layer_norm_nobias_cases() {
            check_layer_norm_nobias(&x, &w, eps, dim, nb, &what, layer_norm_rows_nobias);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn layer_norm_rows_nobias_neon_is_bit_equal_to_the_scalar_twin() {
        if !vector_ok() {
            eprintln!("layer_norm_rows_nobias NEON: no NEON under this tier cap");
            return;
        }
        for (what, x, w, eps, nb, dim) in layer_norm_nobias_cases() {
            check_layer_norm_nobias(
                &x,
                &w,
                eps,
                dim,
                nb,
                &what,
                |x, w, eps, dim, nb, out| unsafe {
                    layer_norm_rows_nobias_neon(x, w, eps, dim, nb, out)
                },
            );
        }
    }
}
