//! the normalised Sylvester–Walsh–Hadamard transform Bonsai 2's ternary weights were rotated by.
//!
//! PrismML folds a block-diagonal rotation `R = H·diag(s)/√n` (or its transpose, see `Rotation`) into every ternary
//! weight matrix, so the engine must rotate each activation by the same transform before the matmul. One 4096-wide
//! transform is 12 butterfly passes over the vector (49 152 adds), negligible beside the matmul it feeds, but it is
//! on the serial path of every layer, so the wide strides are vectorised.

/// In-place unnormalised fast Walsh–Hadamard transform of `x` (length a power of two), Sylvester (natural) order:
/// `y[i] = Σ_j (−1)^{popcount(i & j)} x[j]`.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::have_batched() && n >= 16 {
            // SAFETY: AVX2 checked; n is a power of two ≥ 16.
            unsafe { fwht_avx2(x) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_neon() && n >= 16 {
            // SAFETY: NEON checked; n is a power of two ≥ 16.
            unsafe { fwht_neon(x) };
            return;
        }
    }
    fwht_scalar(x);
}

/// `x[i] *= signs[i]` for every element, the rotation's sign step.
///
/// The multiply is exact and elementwise, so the vector path is bit-identical to the scalar one.
pub fn mul_signs(x: &mut [f32], signs: &[f32]) {
    debug_assert!(x.len() <= signs.len());
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_neon() && x.len() >= 16 {
            // SAFETY: NEON checked; `signs` covers `x`.
            unsafe { mul_signs_neon(x, signs) };
            return;
        }
    }
    for (v, s) in x.iter_mut().zip(signs) {
        *v *= s;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mul_signs_neon(x: &mut [f32], signs: &[f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let p = x.as_mut_ptr();
    let s = signs.as_ptr();
    let mut i = 0;
    while i + 16 <= n {
        for u in 0..4 {
            let at = i + 4 * u;
            vst1q_f32(
                p.add(at),
                vmulq_f32(vld1q_f32(p.add(at)), vld1q_f32(s.add(at))),
            );
        }
        i += 16;
    }
    while i + 4 <= n {
        vst1q_f32(
            p.add(i),
            vmulq_f32(vld1q_f32(p.add(i)), vld1q_f32(s.add(i))),
        );
        i += 4;
    }
    for j in i..n {
        *p.add(j) *= *s.add(j);
    }
}

/// NEON twin of [`fwht_avx2`]: strides 1, 2, 4 and 8 in registers over sixteen floats, then the wider
/// strides in place. Every element is the same `a + b` / `a − b` the scalar loop computes, so the
/// result is bit-identical to it.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fwht_neon(x: &mut [f32]) {
    use std::arch::aarch64::*;

    // Stride 1 inside each four-lane vector: out = [v0+v1, v0−v1, v2+v3, v2−v3].
    #[inline(always)]
    unsafe fn s1(v: float32x4_t) -> float32x4_t {
        let t = vrev64q_f32(v);
        vtrn1q_f32(vaddq_f32(v, t), vsubq_f32(v, t))
    }
    // Stride 2: out = [v0+v2, v1+v3, v0−v2, v1−v3].
    #[inline(always)]
    unsafe fn s2(v: float32x4_t) -> float32x4_t {
        let t = vextq_f32::<2>(v, v);
        vcombine_f32(vget_low_f32(vaddq_f32(v, t)), vget_low_f32(vsubq_f32(v, t)))
    }

    let n = x.len();
    let p = x.as_mut_ptr();
    let mut i = 0;
    while i + 16 <= n {
        let (mut v0, mut v1) = (s1(vld1q_f32(p.add(i))), s1(vld1q_f32(p.add(i + 4))));
        let (mut v2, mut v3) = (s1(vld1q_f32(p.add(i + 8))), s1(vld1q_f32(p.add(i + 12))));
        v0 = s2(v0);
        v1 = s2(v1);
        v2 = s2(v2);
        v3 = s2(v3);
        // Stride 4 across the two halves, then stride 8 across the quarters.
        let (a0, a1) = (vaddq_f32(v0, v1), vsubq_f32(v0, v1));
        let (a2, a3) = (vaddq_f32(v2, v3), vsubq_f32(v2, v3));
        vst1q_f32(p.add(i), vaddq_f32(a0, a2));
        vst1q_f32(p.add(i + 4), vaddq_f32(a1, a3));
        vst1q_f32(p.add(i + 8), vsubq_f32(a0, a2));
        vst1q_f32(p.add(i + 12), vsubq_f32(a1, a3));
        i += 16;
    }
    let mut h = 16;
    while h < n {
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < i + h {
                let a = vld1q_f32(p.add(j));
                let b = vld1q_f32(p.add(j + h));
                vst1q_f32(p.add(j), vaddq_f32(a, b));
                vst1q_f32(p.add(j + h), vsubq_f32(a, b));
                j += 4;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

pub fn fwht_scalar(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fwht_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let p = x.as_mut_ptr();
    // Strides 1, 2, 4 inside each group of eight, in register: out = swap(x) + x ⊙ sign, where `swap` exchanges
    // the partners of the stride and `sign` is +1 on the lower partner, −1 on the upper (exact: one add each).
    let s1 = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    let s2 = _mm256_setr_ps(1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0);
    let s4 = _mm256_setr_ps(1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0);
    let mut i = 0;
    while i < n {
        let mut v = _mm256_loadu_ps(p.add(i));
        v = _mm256_fmadd_ps(v, s1, _mm256_permute_ps::<0b10_11_00_01>(v));
        v = _mm256_fmadd_ps(v, s2, _mm256_permute_ps::<0b01_00_11_10>(v));
        v = _mm256_fmadd_ps(v, s4, _mm256_permute2f128_ps::<0x01>(v, v));
        _mm256_storeu_ps(p.add(i), v);
        i += 8;
    }
    let mut h = 8;
    while h < n {
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < i + h {
                let a = _mm256_loadu_ps(p.add(j));
                let b = _mm256_loadu_ps(p.add(j + h));
                _mm256_storeu_ps(p.add(j), _mm256_add_ps(a, b));
                _mm256_storeu_ps(p.add(j + h), _mm256_sub_ps(a, b));
                j += 8;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_matches_definition_and_vector_path() {
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 11) as f32) - 5.0).collect();
        let mut y = x.clone();
        fwht_scalar(&mut y);
        for i in 0..n {
            let want: f32 = (0..n)
                .map(|j| {
                    if ((i & j) as u32).count_ones() % 2 == 0 {
                        x[j]
                    } else {
                        -x[j]
                    }
                })
                .sum();
            assert_eq!(y[i], want);
        }
        let big: Vec<f32> = (0..4096)
            .map(|i| ((i * 7919 % 101) as f32) / 50.0 - 1.0)
            .collect();
        let (mut a, mut b) = (big.clone(), big.clone());
        fwht_scalar(&mut a);
        fwht(&mut b);
        for i in 0..4096 {
            assert!((a[i] - b[i]).abs() < 1e-3, "{i}: {} vs {}", a[i], b[i]);
        }
    }

    /// The vector path performs the scalar loop's own adds and subtracts, so it must
    /// reproduce them bit for bit at every block size.
    #[test]
    fn fwht_vector_path_is_bit_identical() {
        for n in [16usize, 64, 256, 1024, 4096] {
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 7919 % 101) as f32) / 50.0 - 1.0)
                .collect();
            let (mut want, mut got) = (x.clone(), x.clone());
            fwht_scalar(&mut want);
            fwht(&mut got);
            for i in 0..n {
                assert_eq!(got[i].to_bits(), want[i].to_bits(), "n={n} i={i}");
            }
        }
    }

    #[test]
    fn mul_signs_matches_the_scalar_loop() {
        let n = 4096;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let signs: Vec<f32> = (0..n)
            .map(|i| {
                if (i * 7) % 3 == 0 {
                    -0.015625
                } else {
                    0.015625
                }
            })
            .collect();
        let mut want = x.clone();
        for (v, s) in want.iter_mut().zip(&signs) {
            *v *= s;
        }
        let mut got = x;
        mul_signs(&mut got, &signs);
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
