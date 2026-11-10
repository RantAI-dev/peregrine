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
    /// aarch64 NEON vector path.
    Neon,
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
        #[cfg(target_arch = "aarch64")]
        {
            if crate::cpu::have_neon() {
                return DeltaTier::Neon;
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
            DeltaTier::Neon | DeltaTier::Scalar => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if tier() == DeltaTier::Neon && dk % 8 == 0 {
            // SAFETY: NEON checked by `tier()`; lengths checked above; dk is a multiple of 8.
            unsafe { delta_head_neon(s, q, k, v, beta, decay, o, dk) };
            return;
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

/// NEON implementation of the Gated DeltaNet head update: the same two passes over
/// each state row the AVX2 kernel makes, four floats at a time.
///
/// # Safety
/// The caller must ensure NEON is supported on this CPU, `dk % 8 == 0`, and all slice
/// lengths meet the preconditions of [`delta_head`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn delta_head_neon(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    o: &mut [f32],
    dk: usize,
) {
    use std::arch::aarch64::*;
    let dec = vdupq_n_f32(decay);
    for j in 0..o.len() {
        let row = s.as_mut_ptr().add(j * dk);
        let (mut a0, mut a1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut i = 0;
        while i < dk {
            a0 = vfmaq_f32(a0, vld1q_f32(row.add(i)), vld1q_f32(k.as_ptr().add(i)));
            a1 = vfmaq_f32(
                a1,
                vld1q_f32(row.add(i + 4)),
                vld1q_f32(k.as_ptr().add(i + 4)),
            );
            i += 8;
        }
        let sk = hsum_f32x4(vaddq_f32(a0, a1));
        let delta = vdupq_n_f32((v[j] - decay * sk) * beta);
        let (mut b0, mut b1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut i = 0;
        while i < dk {
            let r0 = vfmaq_f32(
                vmulq_f32(vld1q_f32(row.add(i)), dec),
                delta,
                vld1q_f32(k.as_ptr().add(i)),
            );
            let r1 = vfmaq_f32(
                vmulq_f32(vld1q_f32(row.add(i + 4)), dec),
                delta,
                vld1q_f32(k.as_ptr().add(i + 4)),
            );
            vst1q_f32(row.add(i), r0);
            vst1q_f32(row.add(i + 4), r1);
            b0 = vfmaq_f32(b0, r0, vld1q_f32(q.as_ptr().add(i)));
            b1 = vfmaq_f32(b1, r1, vld1q_f32(q.as_ptr().add(i + 4)));
            i += 8;
        }
        o[j] = hsum_f32x4(vaddq_f32(b0, b1));
    }
}

/// The four lanes of a vector added in order.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn hsum_f32x4(v: std::arch::aarch64::float32x4_t) -> f32 {
    #[allow(clippy::incompatible_msrv)]
    {
        std::arch::aarch64::vaddvq_f32(v)
    }
}

/// The causal conv's inner accumulate for one row of one channel range:
/// `y[i] = y[i]·wcur[i] + Σ_t hist[t][i]·wt[t][i]`, in the scalar loop's order.
///
/// The history and the taps are slices because the ring's plane and the tap row are
/// chosen per position; `hist[t]` and `wt[t]` are the same channel range. Every
/// element is one multiply and one add per tap, so the vector path is bit-identical
/// to the scalar one.
pub fn causal_conv_row(y: &mut [f32], wcur: &[f32], hist: &[&[f32]], wt: &[&[f32]]) {
    debug_assert_eq!(y.len(), wcur.len());
    debug_assert!(hist.len() == wt.len());
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_neon() {
            // SAFETY: NEON checked; every slice is at least `y.len()` long.
            unsafe { causal_conv_row_neon(y, wcur, hist, wt) };
            return;
        }
    }
    for (i, out) in y.iter_mut().enumerate() {
        let mut acc = *out * wcur[i];
        for (h, w) in hist.iter().zip(wt) {
            acc += h[i] * w[i];
        }
        *out = acc;
    }
}

/// # Safety
/// The CPU must have NEON; every `hist[t]` and `wt[t]` must be at least `y.len()` long.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn causal_conv_row_neon(y: &mut [f32], wcur: &[f32], hist: &[&[f32]], wt: &[&[f32]]) {
    use std::arch::aarch64::*;
    let n = y.len();
    let p = y.as_mut_ptr();
    let mut i = 0;
    while i + 16 <= n {
        for u in 0..4 {
            let at = i + 4 * u;
            let mut acc = vmulq_f32(vld1q_f32(p.add(at)), vld1q_f32(wcur.as_ptr().add(at)));
            for (h, w) in hist.iter().zip(wt) {
                acc = vaddq_f32(
                    acc,
                    vmulq_f32(vld1q_f32(h.as_ptr().add(at)), vld1q_f32(w.as_ptr().add(at))),
                );
            }
            vst1q_f32(p.add(at), acc);
        }
        i += 16;
    }
    while i + 4 <= n {
        let mut acc = vmulq_f32(vld1q_f32(p.add(i)), vld1q_f32(wcur.as_ptr().add(i)));
        for (h, w) in hist.iter().zip(wt) {
            acc = vaddq_f32(
                acc,
                vmulq_f32(vld1q_f32(h.as_ptr().add(i)), vld1q_f32(w.as_ptr().add(i))),
            );
        }
        vst1q_f32(p.add(i), acc);
        i += 4;
    }
    for j in i..n {
        let mut acc = *p.add(j) * *wcur.get_unchecked(j);
        for (h, w) in hist.iter().zip(wt) {
            acc += *h.get_unchecked(j) * *w.get_unchecked(j);
        }
        *p.add(j) = acc;
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

            let vectorized = tier() != DeltaTier::Scalar
                && if cfg!(target_arch = "aarch64") {
                    dk % 8 == 0
                } else {
                    dk % 16 == 0
                };
            if !vectorized {
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

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_vs_scalar_reference() {
        if !crate::cpu::have_neon() {
            eprintln!("skipping neon_vs_scalar_reference: NEON not available");
            return;
        }
        let mut rng = Rng::new(0x0BAD_F00D_1234_5678);
        let mut stats_s = DiffStats::default();
        let mut stats_o = DiffStats::default();
        for &(dk, dv) in &[(8usize, 1usize), (128, 32), (128, 128), (256, 16)] {
            // Zero inputs are exact: the vector kernel makes the same adds of zeros.
            {
                let mut s_scalar = vec![0.0f32; dv * dk];
                let mut s_neon = s_scalar.clone();
                let q = vec![0.0f32; dk];
                let k = vec![0.0f32; dk];
                let v = vec![0.0f32; dv];
                let mut o_scalar = vec![0.0f32; dv];
                let mut o_neon = vec![0.0f32; dv];
                delta_head_scalar(&mut s_scalar, &q, &k, &v, 0.5, 0.9, &mut o_scalar, dk);
                unsafe { delta_head_neon(&mut s_neon, &q, &k, &v, 0.5, 0.9, &mut o_neon, dk) };
                assert_eq!(s_neon, s_scalar);
                assert_eq!(o_neon, o_scalar);
            }
            for _ in 0..8 {
                let s_init = rng.vec(dv * dk, -1.0, 1.0);
                let q = rng.vec(dk, -1.0, 1.0);
                let k = rng.vec(dk, -1.0, 1.0);
                let v = rng.vec(dv, -1.0, 1.0);
                let beta = rng.f32_range(0.01, 0.99);
                let decay = rng.f32_range(0.01, 0.99);
                let mut s_scalar = s_init.clone();
                let mut s_neon = s_init.clone();
                let mut o_scalar = vec![0.0f32; dv];
                let mut o_neon = vec![0.0f32; dv];
                delta_head_scalar(&mut s_scalar, &q, &k, &v, beta, decay, &mut o_scalar, dk);
                unsafe { delta_head_neon(&mut s_neon, &q, &k, &v, beta, decay, &mut o_neon, dk) };
                check_diff(&s_neon, &s_scalar, &mut stats_s);
                check_diff(&o_neon, &o_scalar, &mut stats_o);
            }
        }
        eprintln!("NEON vs scalar stats: S={stats_s:?}, O={stats_o:?}");
        assert!(stats_s.max_rel < 2.5e-4, "s max_rel: {}", stats_s.max_rel);
        assert!(stats_o.max_rel < 5.0e-5, "o max_rel: {}", stats_o.max_rel);
    }

    /// The conv's inner accumulate is one multiply and one add per tap, so the vector
    /// path must be bit-identical to the scalar loop.
    #[test]
    fn causal_conv_row_is_bit_identical() {
        let n = 8192;
        let mut rng = Rng::new(0xC0DE_1234_5678_9ABC);
        let mut y: Vec<f32> = rng.vec(n, -2.0, 2.0);
        let wcur = rng.vec(n, -2.0, 2.0);
        let h: Vec<Vec<f32>> = (0..3).map(|_| rng.vec(n, -2.0, 2.0)).collect();
        let w: Vec<Vec<f32>> = (0..3).map(|_| rng.vec(n, -2.0, 2.0)).collect();
        let hist: Vec<&[f32]> = h.iter().map(|v| v.as_slice()).collect();
        let wt: Vec<&[f32]> = w.iter().map(|v| v.as_slice()).collect();
        let mut scalar = y.clone();
        for (i, out) in scalar.iter_mut().enumerate() {
            let mut acc = *out * wcur[i];
            for (hh, ww) in hist.iter().zip(&wt) {
                acc += hh[i] * ww[i];
            }
            *out = acc;
        }
        causal_conv_row(&mut y, &wcur, &hist, &wt);
        assert_eq!(
            y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
