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
            if crate::tern::have_vnni() {
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

