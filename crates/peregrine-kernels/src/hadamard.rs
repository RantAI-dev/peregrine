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
    fwht_scalar(x);
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
}
