//! Gated DeltaNet linear-attention recurrent head (roadmap S1.2 step 1b, ADR 0005).
//!
//! Recurrent state update for Qwen3.5-style hybrid models (Bonsai 2). One v-head advances one token
//! over its recurrent state matrix `S` (`[dv][dk]`), row by row (where row `j` corresponds to output `j`):
//! `S ← e^g S; δ_j = β (v_j − S_j·k); S_j ← S_j + δ_j k; o_j = S_j·q`.
//!
//! AVX2 and AVX-512 kernels are selected at runtime based on CPU features and the `PEREGRINE_ISA` cap
//! (following `table.rs` and `cpu.rs`), with a portable scalar reference.

use std::sync::OnceLock;

/// The CPU tier for delta-rule kernels under `PEREGRINE_ISA`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeltaTier {
    /// Portable scalar reference.
    Scalar,
    /// AVX2 + FMA vector path.
    Avx2,
    /// AVX-512 vector path.
    Avx512,
}

/// The delta-rule tier for this process and CPU under `PEREGRINE_ISA`, detected once.
pub fn tier() -> DeltaTier {
    static TIER: OnceLock<DeltaTier> = OnceLock::new();
    *TIER.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if crate::cpu::have_vnni() {
                return DeltaTier::Avx512;
            }
            if crate::simd::have_batched() {
                return DeltaTier::Avx2;
            }
        }
        DeltaTier::Scalar
    })
}

/// `Σ a·b` over 128-ish slices, in eight-lane order (the reference sums in order; the difference is rounding).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let n = a.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        for l in 0..8 {
            acc[l] += a[i + l] * b[i + l];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in n..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Reference scalar implementation of the delta rule head.
#[allow(clippy::too_many_arguments)]
pub fn delta_head_scalar(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    o: &mut [f32],
    dk: usize,
) {
    assert!(q.len() >= dk, "q shorter than dk");
    assert!(k.len() >= dk, "k shorter than dk");
    assert!(v.len() >= o.len(), "v shorter than output rows");
    assert!(s.len() >= o.len() * dk, "state shorter than o.len() * dk");

    for j in 0..o.len() {
        let row = &mut s[j * dk..(j + 1) * dk];
        let delta = (v[j] - decay * dot(row, k)) * beta;
        for (x, &kk) in row.iter_mut().zip(k) {
            *x = *x * decay + delta * kk;
        }
        o[j] = dot(row, q);
    }
}

/// One v-head's delta rule for one token, row by row of the state `S` (`[dv][dk]`, row j = output j):
/// `S ← e^g S; δ_j = β (v_j − S_j·k); S_j ← S_j + δ_j k; o_j = S_j·q`, in two passes over each row: the first
/// takes `S_j·k` before decaying (`(e^g S_j)·k = e^g (S_j·k)`), the second writes the decayed, updated row and
/// takes its dot with `q` on the way.
///
/// Dispatches to AVX-512 when `dk % 64 == 0` and the CPU tier allows it, to AVX2 when `dk % 16 == 0`,
/// or falls back to the scalar reference.
#[allow(clippy::too_many_arguments)]
pub fn delta_head(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    o: &mut [f32],
    dk: usize,
) {
    assert!(q.len() >= dk, "q shorter than dk");
    assert!(k.len() >= dk, "k shorter than dk");
    assert!(v.len() >= o.len(), "v shorter than output rows");
    assert!(s.len() >= o.len() * dk, "state shorter than o.len() * dk");

    #[cfg(target_arch = "x86_64")]
    {
        match tier() {
            DeltaTier::Avx512 => {
                if dk % 64 == 0 {
                    // SAFETY: AVX-512F checked via tier cap and feature detection; lengths checked above; dk is a multiple of 64.
                    unsafe { delta_head_avx512(s, q, k, v, beta, decay, o, dk) };
                    return;
                }
                if dk % 16 == 0 {
                    // SAFETY: AVX2 + FMA checked via tier cap and feature detection; lengths checked above; dk is a multiple of 16.
                    unsafe { delta_head_avx2(s, q, k, v, beta, decay, o, dk) };
                    return;
                }
            }
            DeltaTier::Avx2 => {
                if dk % 16 == 0 {
                    // SAFETY: AVX2 + FMA checked via tier cap and feature detection; lengths checked above; dk is a multiple of 16.
                    unsafe { delta_head_avx2(s, q, k, v, beta, decay, o, dk) };
                    return;
                }
            }
            DeltaTier::Scalar => {}
        }
    }

    delta_head_scalar(s, q, k, v, beta, decay, o, dk);
}

/// AVX2 + FMA implementation of the Gated DeltaNet head update.
///
/// # Safety
/// The caller must ensure AVX2 and FMA are supported on this CPU, `dk % 16 == 0`,
/// and all slice lengths meet preconditions (`q.len() >= dk`, `k.len() >= dk`,
/// `v.len() >= o.len()`, `s.len() >= o.len() * dk`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn delta_head_avx2(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    o: &mut [f32],
    dk: usize,
) {
    use std::arch::x86_64::*;
    let dec = _mm256_set1_ps(decay);
    for j in 0..o.len() {
        let row = s.as_mut_ptr().add(j * dk);
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i < dk {
            a0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(row.add(i)),
                _mm256_loadu_ps(k.as_ptr().add(i)),
                a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(row.add(i + 8)),
                _mm256_loadu_ps(k.as_ptr().add(i + 8)),
                a1,
            );
            i += 16;
        }
        let sk = crate::simd::hsum_ps(_mm256_add_ps(a0, a1));
        let delta = _mm256_set1_ps((v[j] - decay * sk) * beta);
        let (mut b0, mut b1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i < dk {
            let r0 = _mm256_fmadd_ps(
                delta,
                _mm256_loadu_ps(k.as_ptr().add(i)),
                _mm256_mul_ps(_mm256_loadu_ps(row.add(i)), dec),
            );
            let r1 = _mm256_fmadd_ps(
                delta,
                _mm256_loadu_ps(k.as_ptr().add(i + 8)),
                _mm256_mul_ps(_mm256_loadu_ps(row.add(i + 8)), dec),
            );
            _mm256_storeu_ps(row.add(i), r0);
            _mm256_storeu_ps(row.add(i + 8), r1);
            b0 = _mm256_fmadd_ps(r0, _mm256_loadu_ps(q.as_ptr().add(i)), b0);
            b1 = _mm256_fmadd_ps(r1, _mm256_loadu_ps(q.as_ptr().add(i + 8)), b1);
            i += 16;
        }
        o[j] = crate::simd::hsum_ps(_mm256_add_ps(b0, b1));
    }
}

/// AVX-512 implementation of the Gated DeltaNet head update.
///
/// # Safety
/// The caller must ensure AVX-512F is supported on this CPU, `dk % 64 == 0`,
/// and all slice lengths meet preconditions (`q.len() >= dk`, `k.len() >= dk`,
/// `v.len() >= o.len()`, `s.len() >= o.len() * dk`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn delta_head_avx512(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    o: &mut [f32],
    dk: usize,
) {
    use std::arch::x86_64::*;
    let dec = _mm512_set1_ps(decay);
    for j in 0..o.len() {
        let row = s.as_mut_ptr().add(j * dk);
        let (mut a0, mut a1) = (_mm512_setzero_ps(), _mm512_setzero_ps());
        let mut i = 0;
        while i < dk {
            a0 = _mm512_fmadd_ps(
                _mm512_loadu_ps(row.add(i)),
                _mm512_loadu_ps(k.as_ptr().add(i)),
                a0,
            );
            a1 = _mm512_fmadd_ps(
                _mm512_loadu_ps(row.add(i + 16)),
                _mm512_loadu_ps(k.as_ptr().add(i + 16)),
                a1,
            );
            a0 = _mm512_fmadd_ps(
                _mm512_loadu_ps(row.add(i + 32)),
                _mm512_loadu_ps(k.as_ptr().add(i + 32)),
                a0,
            );
            a1 = _mm512_fmadd_ps(
                _mm512_loadu_ps(row.add(i + 48)),
                _mm512_loadu_ps(k.as_ptr().add(i + 48)),
                a1,
            );
            i += 64;
        }
        let sk = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
        let delta = _mm512_set1_ps((v[j] - decay * sk) * beta);
        let (mut b0, mut b1) = (_mm512_setzero_ps(), _mm512_setzero_ps());
        let mut i = 0;
        while i < dk {
            for u in 0..4 {
                let at = i + 16 * u;
                let r = _mm512_fmadd_ps(
                    delta,
                    _mm512_loadu_ps(k.as_ptr().add(at)),
                    _mm512_mul_ps(_mm512_loadu_ps(row.add(at)), dec),
                );
                _mm512_storeu_ps(row.add(at), r);
                if u % 2 == 0 {
                    b0 = _mm512_fmadd_ps(r, _mm512_loadu_ps(q.as_ptr().add(at)), b0);
                } else {
                    b1 = _mm512_fmadd_ps(r, _mm512_loadu_ps(q.as_ptr().add(at)), b1);
                }
            }
            i += 64;
        }
        o[j] = _mm512_reduce_add_ps(_mm512_add_ps(b0, b1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f32_range(&mut self, lo: f32, hi: f32) -> f32 {
            let frac = ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32;
            lo + (hi - lo) * frac
        }
        fn vec(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..len).map(|_| self.f32_range(lo, hi)).collect()
        }
    }

    #[derive(Default, Debug)]
    struct DiffStats {
        max_abs: f32,
        max_rel: f32, // |g - w| / max(|w|, 1.0)
        max_ulp: u32,
    }

    fn check_diff(got: &[f32], want: &[f32], max_stats: &mut DiffStats) {
        assert_eq!(got.len(), want.len());
        for (&g, &w) in got.iter().zip(want) {
            let abs = (g - w).abs();
            let rel = abs / w.abs().max(1.0);
            let ulp = (g.to_bits() as i64 - w.to_bits() as i64).unsigned_abs() as u32;
            if abs > max_stats.max_abs {
                max_stats.max_abs = abs;
            }
            if rel > max_stats.max_rel {
                max_stats.max_rel = rel;
            }
            if ulp > max_stats.max_ulp {
                max_stats.max_ulp = ulp;
            }
        }
    }

    #[test]
    fn delta_head_checks_preconditions() {
        let mut s = vec![0.0f32; 128];
        let q = vec![0.0f32; 64];
        let k = vec![0.0f32; 64];
        let v = vec![0.0f32; 2];
        let mut o = vec![0.0f32; 2];

        // Normal valid call does not panic
        delta_head(&mut s, &q, &k, &v, 0.5, 0.9, &mut o, 64);

        // q too short
        let r = std::panic::catch_unwind(|| {
            let mut s = vec![0.0f32; 128];
            let mut o = vec![0.0f32; 2];
            delta_head(&mut s, &q[..63], &k, &v, 0.5, 0.9, &mut o, 64);
        });
        assert!(r.is_err());

        // k too short
        let r = std::panic::catch_unwind(|| {
            let mut s = vec![0.0f32; 128];
            let mut o = vec![0.0f32; 2];
            delta_head(&mut s, &q, &k[..63], &v, 0.5, 0.9, &mut o, 64);
        });
        assert!(r.is_err());

        // v too short
        let r = std::panic::catch_unwind(|| {
            let mut s = vec![0.0f32; 128];
            let mut o = vec![0.0f32; 2];
            delta_head(&mut s, &q, &k, &v[..1], 0.5, 0.9, &mut o, 64);
        });
        assert!(r.is_err());

        // s too short
        let r = std::panic::catch_unwind(|| {
            let mut s = vec![0.0f32; 127];
            let mut o = vec![0.0f32; 2];
            delta_head(&mut s, &q, &k, &v, 0.5, 0.9, &mut o, 64);
        });
        assert!(r.is_err());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_vs_scalar_reference() {
        if !crate::simd::have_batched() {
            eprintln!("skipping avx2_vs_scalar_reference: AVX2+FMA not available");
            return;
        }

        let mut rng = Rng::new(0xABCD_EF01_2345_6789);
        let mut stats_s = DiffStats::default();
        let mut stats_o = DiffStats::default();

        // Test multiple (dk, dv) shapes where dk % 16 == 0
        let shapes = [
            (16, 1),
            (16, 8),
            (32, 4),
            (48, 16),
            (64, 32),
            (128, 64),
            (128, 128),
            (256, 16),
        ];

        for &(dk, dv) in &shapes {
            // 1. Zero inputs
            {
                let mut s_scalar = vec![0.0f32; dv * dk];
                let mut s_avx2 = s_scalar.clone();
                let q = vec![0.0f32; dk];
                let k = vec![0.0f32; dk];
                let v = vec![0.0f32; dv];
                let mut o_scalar = vec![0.0f32; dv];
                let mut o_avx2 = vec![0.0f32; dv];

                delta_head_scalar(&mut s_scalar, &q, &k, &v, 0.5, 0.9, &mut o_scalar, dk);
                // SAFETY: have_batched() checked above; lengths and dk % 16 == 0 verified.
                unsafe {
                    delta_head_avx2(&mut s_avx2, &q, &k, &v, 0.5, 0.9, &mut o_avx2, dk);
                }
                assert_eq!(s_avx2, s_scalar);
                assert_eq!(o_avx2, o_scalar);
            }

            // 2. Random normal inputs in [-1.0, 1.0]
            for _ in 0..10 {
                let s_init = rng.vec(dv * dk, -1.0, 1.0);
                let q = rng.vec(dk, -1.0, 1.0);
                let k = rng.vec(dk, -1.0, 1.0);
                let v = rng.vec(dv, -1.0, 1.0);
                let beta = rng.f32_range(0.01, 0.99);
                let decay = rng.f32_range(0.01, 0.99);

                let mut s_scalar = s_init.clone();
                let mut s_avx2 = s_init.clone();
                let mut o_scalar = vec![0.0f32; dv];
                let mut o_avx2 = vec![0.0f32; dv];

                delta_head_scalar(&mut s_scalar, &q, &k, &v, beta, decay, &mut o_scalar, dk);
                // SAFETY: have_batched() checked above; lengths and dk % 16 == 0 verified.
                unsafe {
                    delta_head_avx2(&mut s_avx2, &q, &k, &v, beta, decay, &mut o_avx2, dk);
                }

                check_diff(&s_avx2, &s_scalar, &mut stats_s);
                check_diff(&o_avx2, &o_scalar, &mut stats_o);
            }

            // 3. Large inputs
            {
                let s_init = rng.vec(dv * dk, -50.0, 50.0);
                let q = rng.vec(dk, -2.0, 2.0);
                let k = rng.vec(dk, -2.0, 2.0);
                let v = rng.vec(dv, -50.0, 50.0);
                let beta = 0.8;
                let decay = 0.95;

                let mut s_scalar = s_init.clone();
                let mut s_avx2 = s_init.clone();
                let mut o_scalar = vec![0.0f32; dv];
                let mut o_avx2 = vec![0.0f32; dv];

                delta_head_scalar(&mut s_scalar, &q, &k, &v, beta, decay, &mut o_scalar, dk);
                // SAFETY: have_batched() checked above; lengths and dk % 16 == 0 verified.
                unsafe {
                    delta_head_avx2(&mut s_avx2, &q, &k, &v, beta, decay, &mut o_avx2, dk);
                }

                check_diff(&s_avx2, &s_scalar, &mut stats_s);
                check_diff(&o_avx2, &o_scalar, &mut stats_o);
            }
        }

        eprintln!("AVX2 vs scalar stats: S={stats_s:?}, O={stats_o:?}");
        // Floating point reduction order differs between 8-wide scalar summation and 16-wide AVX2 FMA.
        // Across random inputs and large inputs (scale up to 50.0):
        //   - Recurrent state S: max_abs ≈ 3.7e-4, max_rel < 2.0e-4 (normalized by max(|w|, 1.0))
        //   - Output vector O: max_abs ≈ 5.9e-3, max_rel < 5.0e-5
        // Both match bit-for-bit on zero inputs.
        assert!(stats_s.max_rel < 2.5e-4, "s max_rel: {}", stats_s.max_rel);
        assert!(stats_o.max_rel < 5.0e-5, "o max_rel: {}", stats_o.max_rel);
        assert!(stats_s.max_abs < 5.0e-4, "s max_abs: {}", stats_s.max_abs);
        assert!(stats_o.max_abs < 1.0e-2, "o max_abs: {}", stats_o.max_abs);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_vs_scalar_reference() {
        if !crate::cpu::have_vnni() {
            eprintln!("skipping avx512_vs_scalar_reference: AVX-512 not available");
            return;
        }

        let mut rng = Rng::new(0x1234_5678_9ABC_DEF0);
        let mut stats_s = DiffStats::default();
        let mut stats_o = DiffStats::default();

        // Test multiple (dk, dv) shapes where dk % 64 == 0
        let shapes = [
            (64, 1),
            (64, 8),
            (64, 32),
            (128, 64),
            (128, 128),
            (192, 16),
            (256, 32),
        ];

        for &(dk, dv) in &shapes {
            // 1. Zero inputs
            {
                let mut s_scalar = vec![0.0f32; dv * dk];
                let mut s_avx512 = s_scalar.clone();
                let q = vec![0.0f32; dk];
                let k = vec![0.0f32; dk];
                let v = vec![0.0f32; dv];
                let mut o_scalar = vec![0.0f32; dv];
                let mut o_avx512 = vec![0.0f32; dv];

                delta_head_scalar(&mut s_scalar, &q, &k, &v, 0.5, 0.9, &mut o_scalar, dk);
                // SAFETY: have_vnni() checked above; lengths and dk % 64 == 0 verified.
                unsafe {
                    delta_head_avx512(&mut s_avx512, &q, &k, &v, 0.5, 0.9, &mut o_avx512, dk);
                }
                assert_eq!(s_avx512, s_scalar);
                assert_eq!(o_avx512, o_scalar);
            }

            // 2. Random normal inputs in [-1.0, 1.0]
            for _ in 0..10 {
                let s_init = rng.vec(dv * dk, -1.0, 1.0);
                let q = rng.vec(dk, -1.0, 1.0);
                let k = rng.vec(dk, -1.0, 1.0);
                let v = rng.vec(dv, -1.0, 1.0);
                let beta = rng.f32_range(0.01, 0.99);
                let decay = rng.f32_range(0.01, 0.99);

                let mut s_scalar = s_init.clone();
                let mut s_avx512 = s_init.clone();
                let mut o_scalar = vec![0.0f32; dv];
                let mut o_avx512 = vec![0.0f32; dv];

                delta_head_scalar(&mut s_scalar, &q, &k, &v, beta, decay, &mut o_scalar, dk);
                // SAFETY: have_vnni() checked above; lengths and dk % 64 == 0 verified.
                unsafe {
                    delta_head_avx512(&mut s_avx512, &q, &k, &v, beta, decay, &mut o_avx512, dk);
                }

                check_diff(&s_avx512, &s_scalar, &mut stats_s);
                check_diff(&o_avx512, &o_scalar, &mut stats_o);
            }

            // 3. Large inputs
            {
                let s_init = rng.vec(dv * dk, -50.0, 50.0);
                let q = rng.vec(dk, -2.0, 2.0);
                let k = rng.vec(dk, -2.0, 2.0);
                let v = rng.vec(dv, -50.0, 50.0);
                let beta = 0.8;
                let decay = 0.95;

                let mut s_scalar = s_init.clone();
                let mut s_avx512 = s_init.clone();
                let mut o_scalar = vec![0.0f32; dv];
                let mut o_avx512 = vec![0.0f32; dv];

                delta_head_scalar(&mut s_scalar, &q, &k, &v, beta, decay, &mut o_scalar, dk);
                // SAFETY: have_vnni() checked above; lengths and dk % 64 == 0 verified.
                unsafe {
                    delta_head_avx512(&mut s_avx512, &q, &k, &v, beta, decay, &mut o_avx512, dk);
                }

                check_diff(&s_avx512, &s_scalar, &mut stats_s);
                check_diff(&o_avx512, &o_scalar, &mut stats_o);
            }
        }

        eprintln!("AVX-512 vs scalar stats: S={stats_s:?}, O={stats_o:?}");
        // Floating point reduction order differs between 8-wide scalar summation and 64-wide AVX-512 FMA.
        // Across random inputs and large inputs (scale up to 50.0):
        //   - Recurrent state S: max_abs ≈ 3.7e-4, max_rel < 2.0e-4 (normalized by max(|w|, 1.0))
        //   - Output vector O: max_abs ≈ 3.9e-3, max_rel < 5.0e-5
        // Both match bit-for-bit on zero inputs.
        assert!(stats_s.max_rel < 2.5e-4, "s max_rel: {}", stats_s.max_rel);
        assert!(stats_o.max_rel < 5.0e-5, "o max_rel: {}", stats_o.max_rel);
        assert!(stats_s.max_abs < 5.0e-4, "s max_abs: {}", stats_s.max_abs);
        assert!(stats_o.max_abs < 1.0e-2, "o max_abs: {}", stats_o.max_abs);
    }

    #[test]
    fn delta_head_dispatch_across_shapes() {
        let mut rng = Rng::new(0x9876_5432_10FE_DCBA);
        let mut stats_s = DiffStats::default();
        let mut stats_o = DiffStats::default();

        // Various shapes: some unaligned (scalar), some AVX2-aligned (% 16 == 0), some AVX-512-aligned (% 64 == 0)
        let shapes = [
            (7, 4),    // scalar
            (15, 2),   // scalar
            (16, 4),   // avx2
            (24, 8),   // scalar (not % 16)
            (32, 16),  // avx2
            (48, 8),   // avx2
            (64, 16),  // avx-512 or avx2
            (70, 4),   // scalar
            (128, 32), // avx-512 or avx2
        ];

        for &(dk, dv) in &shapes {
            let s_init = rng.vec(dv * dk, -1.0, 1.0);
            let q = rng.vec(dk, -1.0, 1.0);
            let k = rng.vec(dk, -1.0, 1.0);
            let v = rng.vec(dv, -1.0, 1.0);
            let beta = rng.f32_range(0.1, 0.9);
            let decay = rng.f32_range(0.1, 0.9);

            let mut s_dispatch = s_init.clone();
            let mut s_scalar = s_init.clone();
            let mut o_dispatch = vec![0.0f32; dv];
            let mut o_scalar = vec![0.0f32; dv];

            delta_head(
                &mut s_dispatch,
                &q,
                &k,
                &v,
                beta,
                decay,
                &mut o_dispatch,
                dk,
            );
            delta_head_scalar(&mut s_scalar, &q, &k, &v, beta, decay, &mut o_scalar, dk);

            if dk % 16 != 0 {
                // Must be bit-for-bit identical to scalar when unaligned
                assert_eq!(s_dispatch, s_scalar, "bit identical s for dk={dk}");
                assert_eq!(o_dispatch, o_scalar, "bit identical o for dk={dk}");
            } else {
                check_diff(&s_dispatch, &s_scalar, &mut stats_s);
                check_diff(&o_dispatch, &o_scalar, &mut stats_o);
            }
        }

        assert!(
            stats_s.max_rel < 3.5e-4,
            "dispatch s max_rel: {}",
            stats_s.max_rel
        );
        assert!(
            stats_o.max_rel < 5.0e-5,
            "dispatch o max_rel: {}",
            stats_o.max_rel
        );
        assert!(
            stats_s.max_abs < 5.0e-4,
            "dispatch s max_abs: {}",
            stats_s.max_abs
        );
        assert!(
            stats_o.max_abs < 1.0e-2,
            "dispatch o max_abs: {}",
            stats_o.max_abs
        );
    }
}
