//! kernels for PrismML's ternary formats (Bonsai 2): PQ2_0 and PTQ1_0, group size 128.
//!
//! Both store ternary weights {−1, 0, +1} as unsigned codes {0, 1, 2} with one f16 scale per 128 weights, so
//! the bias identity of `quant.rs` applies with offset 1: `Σ (c − 1)·a = Σ c·a − Σ a`. The activation side is
//! quantised once per matmul into `QuantAct`'s per-128 layout (`fill_ternary`).
//!
//! # PQ2_0: no shuffles
//!
//! Weight `j` of a group sits in bits `2(j % 4)` of byte `j / 4`. Shifting the 32 code bytes right by `2s` and
//! masking gives bit-plane `s`: the codes of weights `4k + s` for `k = 0..32`. `QuantAct::tp` stores the
//! activations permuted the same way (`tp[32s + k] = a[4k + s]`), so each plane meets 32 contiguous activation
//! bytes and a group is four shift–mask–multiply-accumulate steps. Lane `j` of `VPDPBUSD` (and of the AVX2
//! `VPMADDUBSW`+`VPMADDWD` pair) sums bytes `4j..4j+4` of each plane, i.e. activations `16j..16j+16` over the four
//! planes; `QuantAct::tls` holds exactly those sums, so the offset is subtracted in the integer lanes.
//!
//! # PTQ1_0
//!
//! Five trits per byte in base 3. The trit planes of the 16-byte and 8-byte sections are contiguous runs of the
//! natural order, so the kernel decodes planes into 32-byte vectors against the natural-order `tq`.
//!
//! # Batching
//!
//! A weight group is decoded once and used against every activation of the tile, which is what makes prefill and
//! several users in one pass cheaper than the same tokens one at a time.

#![cfg(target_arch = "x86_64")]

use crate::quant::QuantAct;
use std::arch::x86_64::*;

/// Activations per pass over a weight row. Sixteen vector registers (AVX2) hold 8 accumulators, the 4 planes and
/// temporaries; AVX-512's 32 hold them with room.
pub const TILE: usize = 8;

#[inline]
pub fn have_vnni() -> bool {
    crate::cpu_has!("avx512vnni") && crate::cpu_has!("avx512vl")
}

#[inline(always)]
unsafe fn f16_at(p: *const u8) -> f32 {
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
        u16::from_le_bytes([*p, *p.add(1)]) as i32,
    )))
}

#[inline(always)]
unsafe fn hsum(v: __m256) -> f32 {
    crate::simd::hsum_ps(v)
}

/// One row of PQ2_0 against `T` activations (VNNI).
#[target_feature(enable = "avx2,fma,f16c,avx512vnni,avx512vl")]
unsafe fn pq2_vnni<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let m3 = _mm256_set1_epi8(3);
    let mut acc = [_mm256_setzero_ps(); T];
    let ng = w.len() / 34;
    let wp = w.as_ptr();
    for g in 0..ng {
        let blk = wp.add(g * 34);
        let wq = _mm256_loadu_si256(blk.add(2) as *const __m256i);
        let c0 = _mm256_and_si256(wq, m3);
        let c1 = _mm256_and_si256(_mm256_srli_epi16(wq, 2), m3);
        let c2 = _mm256_and_si256(_mm256_srli_epi16(wq, 4), m3);
        let c3 = _mm256_and_si256(_mm256_srli_epi16(wq, 6), m3);
        let dw = f16_at(blk);
        for t in 0..T {
            let a = acts.get_unchecked(t);
            let ap = a.tp.as_ptr().add(crate::quant::tp_off(g)) as *const __m256i;
            // Two independent chains: VPDPBUSD's latency, not its throughput, bounded the one-chain form.
            let ls = _mm256_loadu_si256(a.tls.as_ptr().add(g * 8) as *const __m256i);
            let mut s0 = _mm256_dpbusd_epi32(
                _mm256_sub_epi32(_mm256_setzero_si256(), ls),
                c0,
                _mm256_loadu_si256(ap),
            );
            let mut s1 =
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), c1, _mm256_loadu_si256(ap.add(2)));
            s0 = _mm256_dpbusd_epi32(s0, c2, _mm256_loadu_si256(ap.add(4)));
            s1 = _mm256_dpbusd_epi32(s1, c3, _mm256_loadu_si256(ap.add(6)));
            let s = _mm256_add_epi32(s0, s1);
            let sc = _mm256_mul_ps(
                _mm256_set1_ps(dw),
                _mm256_loadu_ps(a.td.as_ptr().add(g * 8)),
            );
            acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sc, acc[t]);
        }
    }
    for t in 0..T {
        out[t] = hsum(acc[t]);
    }
}

#[inline(always)]
unsafe fn dot32_avx2(codes: __m256i, a: __m256i, ones: __m256i) -> __m256i {
    _mm256_madd_epi16(_mm256_maddubs_epi16(codes, a), ones)
}

/// One row of PQ2_0 against `T` activations (AVX2: `VPMADDUBSW` cannot saturate here, 2·2·127 = 508).
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn pq2_avx2<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let m3 = _mm256_set1_epi8(3);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); T];
    let ng = w.len() / 34;
    let wp = w.as_ptr();
    for g in 0..ng {
        let blk = wp.add(g * 34);
        let wq = _mm256_loadu_si256(blk.add(2) as *const __m256i);
        let c0 = _mm256_and_si256(wq, m3);
        let c1 = _mm256_and_si256(_mm256_srli_epi16(wq, 2), m3);
        let c2 = _mm256_and_si256(_mm256_srli_epi16(wq, 4), m3);
        let c3 = _mm256_and_si256(_mm256_srli_epi16(wq, 6), m3);
        let dw = f16_at(blk);
        for t in 0..T {
            let a = acts.get_unchecked(t);
            let ap = a.tp.as_ptr().add(crate::quant::tp_off(g)) as *const __m256i;
            let mut s = dot32_avx2(c0, _mm256_loadu_si256(ap), ones);
            s = _mm256_add_epi32(s, dot32_avx2(c1, _mm256_loadu_si256(ap.add(2)), ones));
            s = _mm256_add_epi32(s, dot32_avx2(c2, _mm256_loadu_si256(ap.add(4)), ones));
            s = _mm256_add_epi32(s, dot32_avx2(c3, _mm256_loadu_si256(ap.add(6)), ones));
            let ls = _mm256_loadu_si256(a.tls.as_ptr().add(g * 8) as *const __m256i);
            s = _mm256_sub_epi32(s, ls);
            let sc = _mm256_mul_ps(
                _mm256_set1_ps(dw),
                _mm256_loadu_ps(a.td.as_ptr().add(g * 8)),
            );
            acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sc, acc[t]);
        }
    }
    for t in 0..T {
        out[t] = hsum(acc[t]);
    }
}

/// The GFNI affine matrix that moves bits `2s, 2s+1` of every byte to bits 0, 1 (and clears the rest): row `i`
/// of the matrix is byte `7 − i` of the qword.
const fn plane_matrix(s: u32) -> i64 {
    (((1u64 << (2 * s)) << 56) | ((1u64 << (2 * s + 1)) << 48)) as i64
}

/// One row of PQ2_0 against `T` activations, two groups per 512-bit operation: plane extraction by one GFNI affine
/// per plane (instead of shift + mask), VNNI on 64 bytes. A trailing odd group goes through the 256-bit path.
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,gfni")]
unsafe fn pq2_zmm<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let m = [
        _mm512_set1_epi64(plane_matrix(0)),
        _mm512_set1_epi64(plane_matrix(1)),
        _mm512_set1_epi64(plane_matrix(2)),
        _mm512_set1_epi64(plane_matrix(3)),
    ];
    // Lanes 0..8 take scale 0, lanes 8..16 scale 1.
    let pick = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    let mut acc = [_mm512_setzero_ps(); T];
    let ng = w.len() / 34;
    let wp = w.as_ptr();
    let mut g = 0;
    while g + 2 <= ng {
        let b0 = wp.add(g * 34);
        let b1 = b0.add(34);
        let wq = _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm256_loadu_si256(b0.add(2) as *const __m256i)),
            _mm256_loadu_si256(b1.add(2) as *const __m256i),
        );
        let c0 = _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[0]);
        let c1 = _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[1]);
        let c2 = _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[2]);
        let c3 = _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[3]);
        // [d0 ×8, d1 ×8]: each group's weight scale across its half.
        let dws = _mm_cvtph_ps(_mm_cvtsi32_si128(
            (u16::from_le_bytes([*b0, *b0.add(1)]) as u32
                | ((u16::from_le_bytes([*b1, *b1.add(1)]) as u32) << 16)) as i32,
        ));
        let dwv = _mm512_permutexvar_ps(pick, _mm512_castps128_ps512(dws));
        for t in 0..T {
            let a = acts.get_unchecked(t);
            let ap = a.tp.as_ptr().add(crate::quant::tp_off(g)) as *const __m512i;
            let ls = _mm512_loadu_si512(a.tls.as_ptr().add(g * 8) as *const __m512i);
            let mut s0 = _mm512_dpbusd_epi32(
                _mm512_sub_epi32(_mm512_setzero_si512(), ls),
                c0,
                _mm512_loadu_si512(ap),
            );
            let mut s1 =
                _mm512_dpbusd_epi32(_mm512_setzero_si512(), c1, _mm512_loadu_si512(ap.add(1)));
            s0 = _mm512_dpbusd_epi32(s0, c2, _mm512_loadu_si512(ap.add(2)));
            s1 = _mm512_dpbusd_epi32(s1, c3, _mm512_loadu_si512(ap.add(3)));
            let s = _mm512_add_epi32(s0, s1);
            let sc = _mm512_mul_ps(dwv, _mm512_loadu_ps(a.td.as_ptr().add(g * 8)));
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), sc, acc[t]);
        }
        g += 2;
    }
    let mut tail = [0.0f32; T];
    if g < ng {
        // The odd last group, as a one-group row.
        let one = std::slice::from_raw_parts(wp.add(g * 34), 34);
        let _ = one;
        for t in 0..T {
            let a = acts.get_unchecked(t);
            let blk = wp.add(g * 34);
            let row = std::slice::from_raw_parts(blk, 34);
            tail[t] = crate::quant::dot_scalar_pq2_group(row, a, g);
        }
    }
    for t in 0..T {
        out[t] = _mm512_reduce_add_ps(acc[t]) + tail[t];
    }
}

#[inline]
pub fn have_zmm() -> bool {
    have_vnni() && crate::cpu_has!("avx512bw") && crate::cpu_has!("gfni") && zmm_enabled()
}

/// `PEREGRINE_ZMM=0` keeps the 256-bit kernels (an A/B switch).
fn zmm_enabled() -> bool {
    use std::sync::OnceLock;
    static Z: OnceLock<bool> = OnceLock::new();
    *Z.get_or_init(|| crate::options::get().zmm)
}

/// The 128 trits of a PTQ1_0 group as unsigned codes {0, 1, 2} in natural order, four 32-byte vectors.
#[target_feature(enable = "avx2")]
unsafe fn ptq1_codes(blk: *const u8) -> [__m256i; 4] {
    // Trit n of byte b: ((b · 3^n mod 256) · 3) >> 8, computed in 16-bit lanes.
    let lo8 = _mm256_set1_epi16(0xFF);
    let three = _mm256_set1_epi16(3);
    let plane = |b16: __m256i, p3: i16| -> __m256i {
        let m = _mm256_and_si256(_mm256_mullo_epi16(b16, _mm256_set1_epi16(p3)), lo8);
        _mm256_srli_epi16(_mm256_mullo_epi16(m, three), 8)
    };
    // 16 bytes → 16 u16 lanes; planes 0..5 give values 16n..16n+16.
    let b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(blk as *const __m128i));
    let p: [__m256i; 5] = [
        plane(b0, 1),
        plane(b0, 3),
        plane(b0, 9),
        plane(b0, 27),
        plane(b0, 81),
    ];
    // Pack two 16-lane planes into 32 bytes, in order (packus interleaves 128-bit halves; permute fixes it).
    let pack = |a: __m256i, b: __m256i| -> __m256i {
        _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0b11_01_10_00)
    };
    let v0 = pack(p[0], p[1]); // 0..32
    let v1 = pack(p[2], p[3]); // 32..64
                               // 8-byte section: planes give values 80 + 8n .. +8. Lanes 0..8 hold the 8 bytes (rest zero).
    let b1 = _mm256_cvtepu8_epi16(_mm_loadl_epi64(blk.add(16) as *const __m128i));
    let q: [__m256i; 5] = [
        plane(b1, 1),
        plane(b1, 3),
        plane(b1, 9),
        plane(b1, 27),
        plane(b1, 81),
    ];
    // qh: 2 bytes, 4 planes: values 120 + 2n + h.
    let bh = _mm256_cvtepu8_epi16(_mm_cvtsi32_si128(
        u16::from_le_bytes([*blk.add(24), *blk.add(25)]) as i32,
    ));
    let r: [__m256i; 4] = [plane(bh, 1), plane(bh, 3), plane(bh, 9), plane(bh, 27)];
    // Assemble 64..128 through a byte buffer: this group's tail is a quarter of the work, so clarity wins here.
    let mut buf = [0u8; 64];
    let mut tmp = [0u16; 16];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, p[4]);
    for i in 0..16 {
        buf[i] = tmp[i] as u8;
    }
    for n in 0..5 {
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, q[n]);
        for m in 0..8 {
            buf[16 + 8 * n + m] = tmp[m] as u8;
        }
    }
    for n in 0..4 {
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, r[n]);
        buf[56 + 2 * n] = tmp[0] as u8;
        buf[57 + 2 * n] = tmp[1] as u8;
    }
    [
        v0,
        v1,
        _mm256_loadu_si256(buf.as_ptr() as *const __m256i),
        _mm256_loadu_si256(buf.as_ptr().add(32) as *const __m256i),
    ]
}

/// One row of PTQ1_0 against `T` activations.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn ptq1_avx2<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); T];
    let ng = w.len() / 28;
    let wp = w.as_ptr();
    for g in 0..ng {
        let blk = wp.add(g * 28);
        let c = ptq1_codes(blk);
        let dw = f16_at(blk.add(26));
        let dwv = _mm256_set1_ps(dw);
        for t in 0..T {
            let a = acts.get_unchecked(t);
            let ap = a.tq.as_ptr().add(g * 128) as *const __m256i;
            let td = a.td.as_ptr().add(g * 8);
            // Chunk ch covers values 32ch..32ch+32: lanes 0..4 are block 2ch, lanes 4..8 block 2ch+1.
            let mut f = _mm256_setzero_ps();
            for ch in 0..4 {
                let s = dot32_avx2(c[ch], _mm256_loadu_si256(ap.add(ch)), ones);
                let sc = _mm256_set_m128(
                    _mm_set1_ps(*td.add(2 * ch + 1)),
                    _mm_set1_ps(*td.add(2 * ch)),
                );
                f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sc, f);
            }
            // Offset: Σ_u td[u]·Σq_u over the group's eight blocks.
            let ls = _mm256_cvtepi32_ps(_mm256_loadu_si256(
                a.tls.as_ptr().add(g * 8) as *const __m256i
            ));
            f = _mm256_fnmadd_ps(ls, _mm256_loadu_ps(td), f);
            acc[t] = _mm256_fmadd_ps(f, dwv, acc[t]);
        }
    }
    for t in 0..T {
        out[t] = hsum(acc[t]);
    }
}

macro_rules! tiled {
    ($f:ident, $w:expr, $acts:expr, $out:expr) => {{
        let (w, acts, out) = ($w, $acts, $out);
        let mut b = 0;
        while b + TILE <= acts.len() {
            $f::<TILE>(w, &acts[b..b + TILE], &mut out[b..b + TILE]);
            b += TILE;
        }
        let r = &acts[b..];
        let o = &mut out[b..];
        match r.len() {
            7 => $f::<7>(w, r, o),
            6 => $f::<6>(w, r, o),
            5 => $f::<5>(w, r, o),
            4 => $f::<4>(w, r, o),
            3 => $f::<3>(w, r, o),
            2 => $f::<2>(w, r, o),
            1 => $f::<1>(w, r, o),
            _ => {}
        }
    }};
}

/// PQ2_0 row against every activation.
///
/// # Safety
/// AVX2, FMA and F16C must be available; `w` is one row of `acts[i].n` weights and every activation was filled
/// with `fill_ternary`.
pub unsafe fn dot_batch_pq2(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    if have_zmm() {
        tiled!(pq2_zmm, w, acts, out)
    } else if have_vnni() {
        tiled!(pq2_vnni, w, acts, out)
    } else {
        tiled!(pq2_avx2, w, acts, out)
    }
}

/// PTQ1_0 row against every activation.
///
/// # Safety
/// As `dot_batch_pq2`.
pub unsafe fn dot_batch_ptq1(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    if acts.first().is_some_and(|a| a.ptq1) {
        if have_ptq1_zmm() {
            tiled!(ptq1_zmm, w, acts, out)
        } else {
            for (b, a) in acts.iter().enumerate() {
                out[b] = crate::quant::dot_scalar(crate::gguf::Format::PTQ1_0, w, a);
            }
        }
        return;
    }
    tiled!(ptq1_avx2, w, acts, out)
}

/// The AVX2 PQ2_0 path regardless of VNNI, for the tests. Safety as `dot_batch_pq2`.
#[doc(hidden)]
pub unsafe fn dot_batch_pq2_avx2(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    tiled!(pq2_avx2, w, acts, out)
}

/// The 256-bit VNNI PQ2_0 path regardless of the 512-bit one, for the tests. Safety as `dot_batch_pq2`.
#[doc(hidden)]
pub unsafe fn dot_batch_pq2_vnni256(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    tiled!(pq2_vnni, w, acts, out)
}

/// The ternary quantiser, eight lanes at a time: one absmax scale per 16 values, codes rounded half away from
/// zero (as `f32::round`), clamped to ±127, and each block's code sum.
///
/// # Safety
/// AVX2 must be available; `q.len() == x.len()`, `d.len() == sums.len() == x.len() / 16`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn quantize16_avx2(x: &[f32], q: &mut [i8], d: &mut [f32], sums: &mut [i32]) {
    let sign = _mm256_set1_ps(-0.0);
    let half = _mm256_set1_ps(0.5);
    for u in 0..x.len() / 16 {
        let p = x.as_ptr().add(u * 16);
        let (v0, v1) = (_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8)));
        let m = _mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1));
        let m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let amax = _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1)));
        let dd = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = _mm256_set1_ps(if dd > 0.0 { 1.0 / dd } else { 0.0 });
        // round half away from zero: trunc(|v·inv| + 0.5) with the sign put back.
        let r = |v: __m256| {
            let y = _mm256_mul_ps(v, inv);
            let a = _mm256_round_ps(
                _mm256_add_ps(_mm256_andnot_ps(sign, y), half),
                _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC,
            );
            _mm256_cvttps_epi32(_mm256_min_ps(
                _mm256_or_ps(a, _mm256_and_ps(sign, y)),
                _mm256_set1_ps(127.0),
            ))
        };
        let (i0, i1) = (r(v0), r(v1));
        let s = _mm256_add_epi32(i0, i1);
        let s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        let s2 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0b01_00_11_10));
        let s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0b10_11_00_01));
        sums[u] = _mm_cvtsi128_si32(s1);
        // i32 → i8: pack to i16 then i8 (values are within ±127, no saturation).
        let w16 = _mm256_packs_epi32(i0, i1); // lanes interleaved by 128-bit half
        let w16 = _mm256_permute4x64_epi64(w16, 0b11_01_10_00);
        let b8 = _mm_packs_epi16(
            _mm256_castsi256_si128(w16),
            _mm256_extracti128_si256(w16, 1),
        );
        _mm_storeu_si128(q.as_mut_ptr().add(u * 16) as *mut __m128i, b8);
        d[u] = dd;
    }
}

/// Two PQ2_0 rows against one activation, 512-bit: each activation load serves both rows, and the two rows'
/// chains interleave. The decode GEMV's inner loop.
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,gfni")]
unsafe fn pq2_zmm_2rows(w0: &[u8], w1: &[u8], a: &QuantAct) -> (f32, f32) {
    let m = [
        _mm512_set1_epi64(plane_matrix(0)),
        _mm512_set1_epi64(plane_matrix(1)),
        _mm512_set1_epi64(plane_matrix(2)),
        _mm512_set1_epi64(plane_matrix(3)),
    ];
    let pick = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    let (mut acc0, mut acc1) = (_mm512_setzero_ps(), _mm512_setzero_ps());
    let ng = w0.len() / 34;
    let (p0, p1) = (w0.as_ptr(), w1.as_ptr());
    let mut g = 0;
    while g + 2 <= ng {
        let ap = a.tp.as_ptr().add(crate::quant::tp_off(g)) as *const __m512i;
        let (a0, a1, a2, a3) = (
            _mm512_loadu_si512(ap),
            _mm512_loadu_si512(ap.add(1)),
            _mm512_loadu_si512(ap.add(2)),
            _mm512_loadu_si512(ap.add(3)),
        );
        let nls = _mm512_sub_epi32(
            _mm512_setzero_si512(),
            _mm512_loadu_si512(a.tls.as_ptr().add(g * 8) as *const __m512i),
        );
        let td = _mm512_loadu_ps(a.td.as_ptr().add(g * 8));
        macro_rules! row {
            ($p:expr, $acc:ident) => {{
                let b0 = $p.add(g * 34);
                let b1 = b0.add(34);
                let wq = _mm512_inserti64x4::<1>(
                    _mm512_castsi256_si512(_mm256_loadu_si256(b0.add(2) as *const __m256i)),
                    _mm256_loadu_si256(b1.add(2) as *const __m256i),
                );
                let mut s0 =
                    _mm512_dpbusd_epi32(nls, _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[0]), a0);
                let mut s1 = _mm512_dpbusd_epi32(
                    _mm512_setzero_si512(),
                    _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[1]),
                    a1,
                );
                s0 = _mm512_dpbusd_epi32(s0, _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[2]), a2);
                s1 = _mm512_dpbusd_epi32(s1, _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[3]), a3);
                let dws = _mm_cvtph_ps(_mm_cvtsi32_si128(
                    (u16::from_le_bytes([*b0, *b0.add(1)]) as u32
                        | ((u16::from_le_bytes([*b1, *b1.add(1)]) as u32) << 16))
                        as i32,
                ));
                let sc =
                    _mm512_mul_ps(_mm512_permutexvar_ps(pick, _mm512_castps128_ps512(dws)), td);
                $acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(s0, s1)), sc, $acc);
            }};
        }
        row!(p0, acc0);
        row!(p1, acc1);
        g += 2;
    }
    let mut r0 = _mm512_reduce_add_ps(acc0);
    let mut r1 = _mm512_reduce_add_ps(acc1);
    if g < ng {
        r0 += crate::quant::dot_scalar_pq2_group(
            std::slice::from_raw_parts(p0.add(g * 34), 34),
            a,
            g,
        );
        r1 += crate::quant::dot_scalar_pq2_group(
            std::slice::from_raw_parts(p1.add(g * 34), 34),
            a,
            g,
        );
    }
    (r0, r1)
}

/// Two PQ2_0 rows against `T` activations, 512-bit: the batched kernel (prefill, several users). Each pair of
/// groups is decoded once per row; each activation's planes, offsets and scales are loaded once for both rows.
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,gfni")]
unsafe fn pq2_zmm_2r<const T: usize>(
    w0: &[u8],
    w1: &[u8],
    acts: &[QuantAct],
    o0: &mut [f32],
    o1: &mut [f32],
) {
    let m = [
        _mm512_set1_epi64(plane_matrix(0)),
        _mm512_set1_epi64(plane_matrix(1)),
        _mm512_set1_epi64(plane_matrix(2)),
        _mm512_set1_epi64(plane_matrix(3)),
    ];
    let pick = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    let mut acc0 = [_mm512_setzero_ps(); T];
    let mut acc1 = [_mm512_setzero_ps(); T];
    let ng = w0.len() / 34;
    let (p0, p1) = (w0.as_ptr(), w1.as_ptr());
    let decode = |p: *const u8, g: usize| -> ([__m512i; 4], __m512) {
        let b0 = p.add(g * 34);
        let b1 = b0.add(34);
        let wq = _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm256_loadu_si256(b0.add(2) as *const __m256i)),
            _mm256_loadu_si256(b1.add(2) as *const __m256i),
        );
        let dws = _mm_cvtph_ps(_mm_cvtsi32_si128(
            (u16::from_le_bytes([*b0, *b0.add(1)]) as u32
                | ((u16::from_le_bytes([*b1, *b1.add(1)]) as u32) << 16)) as i32,
        ));
        (
            [
                _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[0]),
                _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[1]),
                _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[2]),
                _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[3]),
            ],
            _mm512_permutexvar_ps(pick, _mm512_castps128_ps512(dws)),
        )
    };
    let mut g = 0;
    while g + 2 <= ng {
        let (c0, d0) = decode(p0, g);
        let (c1, d1) = decode(p1, g);
        for t in 0..T {
            let a = acts.get_unchecked(t);
            let ap = a.tp.as_ptr().add(crate::quant::tp_off(g)) as *const __m512i;
            let (a0, a1, a2, a3) = (
                _mm512_loadu_si512(ap),
                _mm512_loadu_si512(ap.add(1)),
                _mm512_loadu_si512(ap.add(2)),
                _mm512_loadu_si512(ap.add(3)),
            );
            let nls = _mm512_sub_epi32(
                _mm512_setzero_si512(),
                _mm512_loadu_si512(a.tls.as_ptr().add(g * 8) as *const __m512i),
            );
            let td = _mm512_loadu_ps(a.td.as_ptr().add(g * 8));
            let mut s0 = _mm512_dpbusd_epi32(nls, c0[0], a0);
            let mut u0 = _mm512_dpbusd_epi32(nls, c1[0], a0);
            s0 = _mm512_dpbusd_epi32(s0, c0[1], a1);
            u0 = _mm512_dpbusd_epi32(u0, c1[1], a1);
            s0 = _mm512_dpbusd_epi32(s0, c0[2], a2);
            u0 = _mm512_dpbusd_epi32(u0, c1[2], a2);
            s0 = _mm512_dpbusd_epi32(s0, c0[3], a3);
            u0 = _mm512_dpbusd_epi32(u0, c1[3], a3);
            acc0[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s0), _mm512_mul_ps(d0, td), acc0[t]);
            acc1[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(u0), _mm512_mul_ps(d1, td), acc1[t]);
        }
        g += 2;
    }
    for t in 0..T {
        let a = acts.get_unchecked(t);
        let (mut r0, mut r1) = (_mm512_reduce_add_ps(acc0[t]), _mm512_reduce_add_ps(acc1[t]));
        if g < ng {
            r0 += crate::quant::dot_scalar_pq2_group(
                std::slice::from_raw_parts(p0.add(g * 34), 34),
                a,
                g,
            );
            r1 += crate::quant::dot_scalar_pq2_group(
                std::slice::from_raw_parts(p1.add(g * 34), 34),
                a,
                g,
            );
        }
        o0[t] = r0;
        o1[t] = r1;
    }
}

#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,gfni")]
unsafe fn pq2_rows_zmm(w: &[u8], rb: usize, rows: usize, acts: &[QuantAct], out: &mut [f32]) {
    let nb = acts.len();
    if nb == 1 {
        let a = &acts[0];
        let mut r = 0;
        while r + 2 <= rows {
            let (x, y) = pq2_zmm_2rows(&w[r * rb..(r + 1) * rb], &w[(r + 1) * rb..(r + 2) * rb], a);
            out[r] = x;
            out[r + 1] = y;
            r += 2;
        }
        if r < rows {
            pq2_zmm::<1>(&w[r * rb..(r + 1) * rb], acts, &mut out[r..r + 1]);
        }
        return;
    }
    // Pairs of rows through the two-row kernel, tiles of BT activations; an odd last row on its own.
    const BT: usize = 6;
    let mut r = 0;
    let mut t0 = [0.0f32; BT];
    let mut t1 = [0.0f32; BT];
    while r + 2 <= rows {
        let (w0, w1) = (&w[r * rb..(r + 1) * rb], &w[(r + 1) * rb..(r + 2) * rb]);
        let mut b = 0;
        while b < nb {
            let k = BT.min(nb - b);
            let a = &acts[b..b + k];
            match k {
                6 => pq2_zmm_2r::<6>(w0, w1, a, &mut t0, &mut t1),
                5 => pq2_zmm_2r::<5>(w0, w1, a, &mut t0, &mut t1),
                4 => pq2_zmm_2r::<4>(w0, w1, a, &mut t0, &mut t1),
                3 => pq2_zmm_2r::<3>(w0, w1, a, &mut t0, &mut t1),
                2 => pq2_zmm_2r::<2>(w0, w1, a, &mut t0, &mut t1),
                _ => pq2_zmm_2r::<1>(w0, w1, a, &mut t0, &mut t1),
            }
            out[r * nb + b..r * nb + b + k].copy_from_slice(&t0[..k]);
            out[(r + 1) * nb + b..(r + 1) * nb + b + k].copy_from_slice(&t1[..k]);
            b += k;
        }
        r += 2;
    }
    if r < rows {
        let row = &w[r * rb..(r + 1) * rb];
        let o = &mut out[r * nb..(r + 1) * nb];
        tiled!(pq2_zmm, row, acts, o)
    }
}

/// Rows `0..rows` of a PQ2_0 matrix (`rb` bytes each, starting at `w`) against every activation, into
/// `out[r·nb + b]`. One call per chunk of rows, so the dispatch is paid once per chunk rather than per row.
///
/// # Safety
/// As `dot_batch_pq2`; `w.len() >= rows·rb` and `out.len() >= rows·acts.len()`.
pub unsafe fn dot_rows_pq2(w: &[u8], rb: usize, rows: usize, acts: &[QuantAct], out: &mut [f32]) {
    if have_zmm() {
        pq2_rows_zmm(w, rb, rows, acts, out);
    } else {
        let nb = acts.len();
        for r in 0..rows {
            dot_batch_pq2(
                &w[r * rb..(r + 1) * rb],
                acts,
                &mut out[r * nb..(r + 1) * nb],
            );
        }
    }
}

#[inline]
pub fn have_vbmi() -> bool {
    crate::cpu_has!("avx512vbmi") && crate::cpu_has!("avx512bw")
}

/// `tp[tp_off(g) + 64s + k] = tq[128g + 4k + s]` for every group, two byte permutes per group.
///
/// # Safety
/// AVX-512 VBMI and BW must be available; `tq.len()` is a multiple of 128 and `tp` holds its padded layout.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx2")]
pub unsafe fn permute_planes_vbmi(tq: &[i8], tp: &mut [i8]) {
    // Output bytes 0..32 are plane `lo`, 32..64 plane `lo + 1`: index 4k + s into the group's 128 bytes.
    let idx = |lo: u8| -> __m512i {
        let mut v = [0u8; 64];
        for j in 0..64 {
            let (s, k) = (lo + (j / 32) as u8, (j % 32) as u8);
            v[j] = 4 * k + s;
        }
        _mm512_loadu_si512(v.as_ptr() as *const __m512i)
    };
    let (i01, i23) = (idx(0), idx(2));
    for g in 0..tq.len() / 128 {
        let src = tq.as_ptr().add(g * 128) as *const __m512i;
        let (z0, z1) = (_mm512_loadu_si512(src), _mm512_loadu_si512(src.add(1)));
        let p01 = _mm512_permutex2var_epi8(z0, i01, z1);
        let p23 = _mm512_permutex2var_epi8(z0, i23, z1);
        let d = tp.as_mut_ptr().add(crate::quant::tp_off(g));
        _mm256_storeu_si256(d as *mut __m256i, _mm512_castsi512_si256(p01));
        _mm256_storeu_si256(
            d.add(64) as *mut __m256i,
            _mm512_extracti64x4_epi64::<1>(p01),
        );
        _mm256_storeu_si256(d.add(128) as *mut __m256i, _mm512_castsi512_si256(p23));
        _mm256_storeu_si256(
            d.add(192) as *mut __m256i,
            _mm512_extracti64x4_epi64::<1>(p23),
        );
    }
}

/// One PTQ1_0 row against `T` activations in the PTQ1 layout (`QuantAct::ptq1`), two groups per pass, 512-bit.
///
/// A pair's 56 bytes are loaded once and spread by three byte permutes into the three source patterns (the
/// 16-byte runs; the 8-byte runs twice; the 8-byte runs and the `qh` pattern), zero-extended to 16 bits. Trit n of
/// a byte v is `mulhi(mullo(v, 3ⁿ·256), 3)`: the low multiply keeps `(v·3ⁿ mod 256)·256`, the high one returns
/// `(3·that) >> 16`. Round r yields the trits of values 16r..16r+16 of both groups, against the activations'
/// round r through `VPDPWSSD`; all eight rounds accumulate in one i32 vector per activation (lanes 0..8 group 0,
/// 8..16 group 1), offset-initialised, and scaled once.
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
unsafe fn ptq1_zmm<const T: usize>(w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    // Byte-permute indices into the 56-byte pair (block 0 at 0, block 1 at 28).
    let mk = |f: &dyn Fn(usize) -> usize| -> __m256i {
        let mut v = [0u8; 32];
        for (j, b) in v.iter_mut().enumerate() {
            *b = f(j) as u8;
        }
        _mm256_loadu_si256(v.as_ptr() as *const __m256i)
    };
    let ia = mk(&|j| if j < 16 { j } else { 28 + j - 16 });
    let ib = mk(&|j| {
        let (gi, m) = (j / 16, j % 8);
        28 * gi + 16 + m
    });
    let ic = mk(&|j| {
        let (gi, half, m) = (j / 16, (j / 8) % 2, j % 8);
        if half == 0 {
            28 * gi + 16 + m
        } else {
            28 * gi + 24 + (m % 2)
        }
    });
    let p3 = |n: u32| (3i32.pow(n) * 256) as i16;
    let pn: [__m512i; 5] = [
        _mm512_set1_epi16(p3(0)),
        _mm512_set1_epi16(p3(1)),
        _mm512_set1_epi16(p3(2)),
        _mm512_set1_epi16(p3(3)),
        _mm512_set1_epi16(p3(4)),
    ];
    // Per-lane multipliers for the 8-byte rounds and the last round (lanes in units of 16-bit).
    let lanes = |f: &dyn Fn(usize) -> u32| -> __m512i {
        let mut v = [0i16; 32];
        for (j, x) in v.iter_mut().enumerate() {
            *x = p3(f(j));
        }
        _mm512_loadu_si512(v.as_ptr() as *const __m512i)
    };
    let mb01 = lanes(&|j| ((j / 8) % 2) as u32);
    let mb23 = lanes(&|j| 2 + ((j / 8) % 2) as u32);
    let mbc = lanes(&|j| {
        if (j / 8) % 2 == 0 {
            4
        } else {
            ((j % 8) / 2) as u32
        }
    });
    let three = _mm512_set1_epi16(3);
    let pick = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    let mut acc = [_mm512_setzero_ps(); T];
    let np = w.len() / 56;
    for p in 0..np {
        let src =
            _mm512_maskz_loadu_epi8(0x00FF_FFFF_FFFF_FFFF, w.as_ptr().add(p * 56) as *const i8);
        let take = |idx: __m256i| -> __m512i {
            _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutexvar_epi8(
                _mm512_castsi256_si512(idx),
                src,
            )))
        };
        let (va, vb, vc) = (take(ia), take(ib), take(ic));
        let tr = |v: __m512i, m: __m512i| _mm512_mulhi_epu16(_mm512_mullo_epi16(v, m), three);
        let t: [__m512i; 8] = [
            tr(va, pn[0]),
            tr(va, pn[1]),
            tr(va, pn[2]),
            tr(va, pn[3]),
            tr(va, pn[4]),
            tr(vb, mb01),
            tr(vb, mb23),
            tr(vc, mbc),
        ];
        let b0 = w.as_ptr().add(p * 56);
        let dws = _mm_cvtph_ps(_mm_cvtsi32_si128(
            (u16::from_le_bytes([*b0.add(26), *b0.add(27)]) as u32
                | ((u16::from_le_bytes([*b0.add(54), *b0.add(55)]) as u32) << 16))
                as i32,
        ));
        let dwv = _mm512_permutexvar_ps(pick, _mm512_castps128_ps512(dws));
        for ti in 0..T {
            let a = acts.get_unchecked(ti);
            let ap = a.t1a.as_ptr().add(p * 256) as *const __m512i;
            // Two chains of four: one chain of eight was bound by VPDPWSSD's latency.
            let mut s0 = _mm512_loadu_si512(a.t1init.as_ptr().add(p * 16) as *const __m512i);
            let mut s1 = _mm512_setzero_si512();
            for r in 0..4 {
                s0 = _mm512_dpwssd_epi32(s0, t[2 * r], _mm512_loadu_si512(ap.add(2 * r)));
                s1 = _mm512_dpwssd_epi32(s1, t[2 * r + 1], _mm512_loadu_si512(ap.add(2 * r + 1)));
            }
            let s = _mm512_add_epi32(s0, s1);
            let sc = _mm512_mul_ps(dwv, _mm512_loadu_ps(a.t1sc.as_ptr().add(p * 16)));
            acc[ti] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), sc, acc[ti]);
        }
    }
    for ti in 0..T {
        out[ti] = _mm512_reduce_add_ps(acc[ti]);
    }
}

#[inline]
pub fn have_ptq1_zmm() -> bool {
    have_vnni() && have_vbmi() && zmm_enabled()
}

#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
unsafe fn ptq1_rows_zmm(w: &[u8], rb: usize, rows: usize, acts: &[QuantAct], out: &mut [f32]) {
    let nb = acts.len();
    for r in 0..rows {
        let row = &w[r * rb..(r + 1) * rb];
        let o = &mut out[r * nb..(r + 1) * nb];
        tiled!(ptq1_zmm, row, acts, o)
    }
}

/// Rows of a PTQ1_0 matrix against activations in the PTQ1 layout, into `out[r·nb + b]`.
///
/// # Safety
/// `have_ptq1_zmm()`; every activation filled with `ptq1` set; row width a multiple of 256.
pub unsafe fn dot_rows_ptq1(w: &[u8], rb: usize, rows: usize, acts: &[QuantAct], out: &mut [f32]) {
    ptq1_rows_zmm(w, rb, rows, acts, out)
}

/// The ternary quantiser with one scale per `span` values (a multiple of 16): codes as `quantize16_avx2`, the span
/// scale written into each 16-block's `d` slot and each 16-block's code sum into `sums`.
///
/// # Safety
/// AVX2 must be available; `q.len() == x.len()`, `d.len() == sums.len() == x.len() / 16`, `span % 16 == 0`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn quantize_span_avx2(
    x: &[f32],
    span: usize,
    q: &mut [i8],
    d: &mut [f32],
    sums: &mut [i32],
) {
    let sign = _mm256_set1_ps(-0.0);
    let half = _mm256_set1_ps(0.5);
    for c in 0..x.len() / span {
        let p = x.as_ptr().add(c * span);
        let mut m = _mm256_setzero_ps();
        let mut i = 0;
        while i < span {
            m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(p.add(i))));
            i += 8;
        }
        let m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let amax = _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1)));
        let dd = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = _mm256_set1_ps(if dd > 0.0 { 1.0 / dd } else { 0.0 });
        let r = |v: __m256| {
            let y = _mm256_mul_ps(v, inv);
            let a = _mm256_round_ps(
                _mm256_add_ps(_mm256_andnot_ps(sign, y), half),
                _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC,
            );
            _mm256_cvttps_epi32(_mm256_min_ps(
                _mm256_or_ps(a, _mm256_and_ps(sign, y)),
                _mm256_set1_ps(127.0),
            ))
        };
        for u in 0..span / 16 {
            let pu = p.add(u * 16);
            let (i0, i1) = (r(_mm256_loadu_ps(pu)), r(_mm256_loadu_ps(pu.add(8))));
            let s = _mm256_add_epi32(i0, i1);
            let s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
            let s2 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0b01_00_11_10));
            let s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0b10_11_00_01));
            let blk = c * span / 16 + u;
            sums[blk] = _mm_cvtsi128_si32(s1);
            let w16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0b11_01_10_00);
            let b8 = _mm_packs_epi16(
                _mm256_castsi256_si128(w16),
                _mm256_extracti128_si256(w16, 1),
            );
            _mm_storeu_si128(q.as_mut_ptr().add(blk * 16) as *mut __m128i, b8);
            d[blk] = dd;
        }
    }
}

/// Bytes of one 16-row tile per 128-group: 8 quads × 16 rows × 4 code bytes, then 16 f16 scales.
pub const TILE_GROUP_BYTES: usize = 8 * 64 + 32;

/// Repack `rows` PQ2_0 rows (`rb` bytes each, `rows % 16 == 0`) into row-interleaved tiles: per tile and group,
/// quad q holds code bytes 4q..4q+4 of the 16 rows side by side (one 64-byte load), then the 16 scales. A
/// reordering of the same bytes (lossless).
pub fn repack_tiles(w: &[u8], rows: usize, rb: usize, out: &mut [u8]) {
    debug_assert!(rows % 16 == 0 && out.len() == rows * rb && rb % 34 == 0);
    let ng = rb / 34;
    let mut o = 0;
    for tile in 0..rows / 16 {
        for g in 0..ng {
            for q in 0..8 {
                for r in 0..16 {
                    let src = (tile * 16 + r) * rb + g * 34 + 2 + 4 * q;
                    out[o..o + 4].copy_from_slice(&w[src..src + 4]);
                    o += 4;
                }
            }
            for r in 0..16 {
                let src = (tile * 16 + r) * rb + g * 34;
                out[o..o + 2].copy_from_slice(&w[src..src + 2]);
                o += 2;
            }
        }
    }
}

/// One 16-row tile against `T` activations (broadcast layout, `SPAN` values per activation scale), into
/// `out[r·stride + t]`. The 32-bit lanes are the 16 rows: each `VPDPBUSD` multiplies 4 code bytes of every row
/// (one GFNI-extracted plane) by the same 4 activation bytes, broadcast from memory, so an accumulator lane runs a
/// whole span of one row and the float epilogue is paid once per span rather than once per 16 values.
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,gfni")]
unsafe fn tile16<const T: usize, const SPAN: usize>(
    wt: *const u8,
    ng: usize,
    acts: &[QuantAct],
    out: *mut f32,
    stride: usize,
) {
    let m = [
        _mm512_set1_epi64(plane_matrix(0)),
        _mm512_set1_epi64(plane_matrix(1)),
        _mm512_set1_epi64(plane_matrix(2)),
        _mm512_set1_epi64(plane_matrix(3)),
    ];
    let mut accf = [_mm512_setzero_ps(); T];
    let qps = SPAN / 16;
    for g in 0..ng {
        let base = wt.add(g * TILE_GROUP_BYTES);
        let dw = _mm512_cvtph_ps(_mm256_loadu_si256(base.add(512) as *const __m256i));
        let mut q0 = 0;
        while q0 < 8 {
            let sp = (g * 128 + q0 * 16) / SPAN;
            let mut acc = [_mm512_setzero_si512(); T];
            for t in 0..T {
                acc[t] = _mm512_set1_epi32(*acts.get_unchecked(t).tbn.get_unchecked(sp));
            }
            for q in q0..q0 + qps {
                let wq = _mm512_loadu_si512(base.add(q * 64) as *const __m512i);
                for s in 0..4 {
                    let c = _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[s]);
                    for t in 0..T {
                        let bp = acts
                            .get_unchecked(t)
                            .tb
                            .as_ptr()
                            .add(g * 128 + q * 16 + s * 4)
                            as *const i32;
                        acc[t] =
                            _mm512_dpbusd_epi32(acc[t], c, _mm512_set1_epi32(bp.read_unaligned()));
                    }
                }
            }
            for t in 0..T {
                let sc = _mm512_mul_ps(
                    dw,
                    _mm512_set1_ps(*acts.get_unchecked(t).tbd.get_unchecked(sp)),
                );
                accf[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[t]), sc, accf[t]);
            }
            q0 += qps;
        }
    }
    let mut buf = [0.0f32; 16];
    for t in 0..T {
        _mm512_storeu_ps(buf.as_mut_ptr(), accf[t]);
        for r in 0..16 {
            *out.add(r * stride + t) = buf[r];
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,gfni")]
unsafe fn tiles_span<const SPAN: usize>(
    wt: &[u8],
    ng: usize,
    ntiles: usize,
    acts: &[QuantAct],
    out: &mut [f32],
) {
    let nb = acts.len();
    for tile in 0..ntiles {
        let w = wt.as_ptr().add(tile * ng * TILE_GROUP_BYTES);
        let o = out.as_mut_ptr().add(tile * 16 * nb);
        let mut b = 0;
        while b < nb {
            let k = (nb - b).min(8);
            let a = &acts[b..b + k];
            let ob = o.add(b);
            match k {
                8 => tile16::<8, SPAN>(w, ng, a, ob, nb),
                7 => tile16::<7, SPAN>(w, ng, a, ob, nb),
                6 => tile16::<6, SPAN>(w, ng, a, ob, nb),
                5 => tile16::<5, SPAN>(w, ng, a, ob, nb),
                4 => tile16::<4, SPAN>(w, ng, a, ob, nb),
                3 => tile16::<3, SPAN>(w, ng, a, ob, nb),
                2 => tile16::<2, SPAN>(w, ng, a, ob, nb),
                _ => tile16::<1, SPAN>(w, ng, a, ob, nb),
            }
            b += k;
        }
    }
}

/// `ntiles` tiles of a repacked PQ2_0 matrix (`ng` groups per row, starting at `wt`) against every activation, into
/// `out[r·nb + b]` for the tiles' 16·ntiles rows.
///
/// # Safety
/// `have_zmm()`; every activation filled with `tb_span` set (32, 64 or 128, the same for all); `wt` holds the
/// tiles; `out.len() >= 16·ntiles·acts.len()`.
pub unsafe fn dot_tiles_pq2(
    wt: &[u8],
    ng: usize,
    ntiles: usize,
    acts: &[QuantAct],
    out: &mut [f32],
) {
    match acts[0].tb_span {
        128 => tiles_span::<128>(wt, ng, ntiles, acts, out),
        64 => tiles_span::<64>(wt, ng, ntiles, acts, out),
        _ => tiles_span::<32>(wt, ng, ntiles, acts, out),
    }
}

/// `out[16k + 4s + i] = q[16k + 4i + s]` (a 4×4 byte transpose in every 16 bytes), 64 bytes per shuffle.
///
/// # Safety
/// AVX-512 BW must be available; `q.len() == out.len()`, a multiple of 64.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn transpose4x4_bytes(q: &[i8], out: &mut [i8]) {
    let idx = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
    ));
    let mut i = 0;
    while i < q.len() {
        let v = _mm512_loadu_si512(q.as_ptr().add(i) as *const __m512i);
        _mm512_storeu_si512(
            out.as_mut_ptr().add(i) as *mut __m512i,
            _mm512_shuffle_epi8(v, idx),
        );
        i += 64;
    }
}
