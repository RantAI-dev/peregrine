// which part bounds the batched PQ2_0 kernel? Variants of the 2-row × T kernel with one piece removed.
#![allow(clippy::all)]
use peregrine_core::gguf::Format;
use peregrine_core::quant::{f32_to_f16, row_bytes, tp_off, QuantAct};
use std::arch::x86_64::*;
use std::time::Instant;

const fn pm(s: u32) -> i64 {
    (((1u64 << (2 * s)) << 56) | ((1u64 << (2 * s + 1)) << 48)) as i64
}

#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,gfni")]
unsafe fn k2r<const T: usize, const V: u32>(
    w0: &[u8],
    w1: &[u8],
    acts: &[QuantAct],
    o: &mut [f32],
) {
    let m = [
        _mm512_set1_epi64(pm(0)),
        _mm512_set1_epi64(pm(1)),
        _mm512_set1_epi64(pm(2)),
        _mm512_set1_epi64(pm(3)),
    ];
    let pick = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    let mut acc0 = [_mm512_setzero_ps(); T];
    let mut acc1 = [_mm512_setzero_ps(); T];
    let mut iacc0 = [_mm512_setzero_si512(); T];
    let mut iacc1 = [_mm512_setzero_si512(); T];
    let ng = w0.len() / 34;
    let fixed_a = _mm512_loadu_si512(acts[0].tp.as_ptr() as *const __m512i);
    let mut c0 = [_mm512_set1_epi8(1); 4];
    let mut c1 = [_mm512_set1_epi8(2); 4];
    let mut g = 0;
    while g + 2 <= ng {
        let mut d0 = _mm512_set1_ps(1.0);
        let mut d1 = _mm512_set1_ps(1.0);
        if V != 3 {
            for (p, c, d) in [
                (w0.as_ptr(), &mut c0, &mut d0),
                (w1.as_ptr(), &mut c1, &mut d1),
            ] {
                let b0 = p.add(g * 34);
                let b1 = b0.add(34);
                let wq = _mm512_inserti64x4::<1>(
                    _mm512_castsi256_si512(_mm256_loadu_si256(b0.add(2) as *const __m256i)),
                    _mm256_loadu_si256(b1.add(2) as *const __m256i),
                );
                for s in 0..4 {
                    c[s] = _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[s]);
                }
                let dws = _mm_cvtph_ps(_mm_cvtsi32_si128(
                    (u16::from_le_bytes([*b0, *b0.add(1)]) as u32
                        | ((u16::from_le_bytes([*b1, *b1.add(1)]) as u32) << 16))
                        as i32,
                ));
                *d = _mm512_permutexvar_ps(pick, _mm512_castps128_ps512(dws));
            }
        }
        for t in 0..T {
            let a = acts.get_unchecked(t);
            let ap = a.tp.as_ptr().add(tp_off(g)) as *const __m512i;
            let (a0, a1, a2, a3) = if V == 2 {
                (fixed_a, fixed_a, fixed_a, fixed_a)
            } else {
                (
                    _mm512_loadu_si512(ap),
                    _mm512_loadu_si512(ap.add(1)),
                    _mm512_loadu_si512(ap.add(2)),
                    _mm512_loadu_si512(ap.add(3)),
                )
            };
            if V == 1 {
                let mut s = iacc0[t];
                let mut u = iacc1[t];
                s = _mm512_dpbusd_epi32(s, c0[0], a0);
                u = _mm512_dpbusd_epi32(u, c1[0], a0);
                s = _mm512_dpbusd_epi32(s, c0[1], a1);
                u = _mm512_dpbusd_epi32(u, c1[1], a1);
                s = _mm512_dpbusd_epi32(s, c0[2], a2);
                u = _mm512_dpbusd_epi32(u, c1[2], a2);
                s = _mm512_dpbusd_epi32(s, c0[3], a3);
                u = _mm512_dpbusd_epi32(u, c1[3], a3);
                iacc0[t] = s;
                iacc1[t] = u;
                continue;
            }
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
        o[t] = _mm512_reduce_add_ps(acc0[t])
            + _mm512_reduce_add_ps(acc1[t])
            + _mm512_reduce_add_epi32(_mm512_add_epi32(iacc0[t], iacc1[t])) as f32;
    }
}

/// Row-interleaved tiles: per (16-row tile, group): 8 quads × [16 rows × 4 code bytes] (512 B), then 16 f16 scales.
fn repack16(w: &[u8], rows: usize, rb: usize) -> Vec<u8> {
    let ng = rb / 34;
    let mut out = vec![0u8; w.len()];
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
    out
}

/// Activation side for the tiled kernel: per group, per quad q, per plane s, the 4 bytes
/// [a[16q+s], a[16q+4+s], a[16q+8+s], a[16q+12+s]] (broadcast to every row lane). One scale per SPAN values.
struct TAct {
    b: Vec<i8>,
    d: Vec<f32>,
    nsum: Vec<i32>,
}
fn tact(x: &[f32], span: usize) -> TAct {
    let n = x.len();
    let mut q = vec![0i8; n];
    let mut d = vec![0f32; n / span];
    let mut nsum = vec![0i32; n / span];
    for (c, blk) in x.chunks_exact(span).enumerate() {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let dd = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = if dd > 0.0 { 1.0 / dd } else { 0.0 };
        let mut sm = 0;
        for (i, &v) in blk.iter().enumerate() {
            let c8 = (v * inv).round().clamp(-127.0, 127.0) as i8;
            q[c * span + i] = c8;
            sm += c8 as i32;
        }
        d[c] = dd;
        nsum[c] = -sm;
    }
    let mut b = vec![0i8; n];
    for g in 0..n / 128 {
        for qq in 0..8 {
            for s in 0..4 {
                for i in 0..4 {
                    b[g * 128 + qq * 16 + s * 4 + i] = q[g * 128 + 16 * qq + 4 * i + s];
                }
            }
        }
    }
    TAct { b, d, nsum }
}

/// 16 rows × T activations over one tile; SPAN values share an activation scale.
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,gfni")]
unsafe fn tile16<const T: usize, const SPAN: usize>(
    wt: &[u8],
    ng: usize,
    acts: &[TAct],
    out: &mut [[f32; 16]],
) {
    let m = [
        _mm512_set1_epi64(pm(0)),
        _mm512_set1_epi64(pm(1)),
        _mm512_set1_epi64(pm(2)),
        _mm512_set1_epi64(pm(3)),
    ];
    let mut accf = [_mm512_setzero_ps(); T];
    let qps = SPAN / 16; // quads per span
    for g in 0..ng {
        let base = wt.as_ptr().add(g * 544);
        let dw = _mm512_cvtph_ps(_mm256_loadu_si256(base.add(512) as *const __m256i));
        let mut q0 = 0;
        while q0 < 8 {
            let sp = (g * 128 + q0 * 16) / SPAN;
            let mut acc = [_mm512_setzero_si512(); T];
            for t in 0..T {
                acc[t] = _mm512_set1_epi32(*acts.get_unchecked(t).nsum.get_unchecked(sp));
            }
            for q in q0..q0 + qps {
                let wq = _mm512_loadu_si512(base.add(q * 64) as *const __m512i);
                for s in 0..4 {
                    let c = _mm512_gf2p8affine_epi64_epi8::<0>(wq, m[s]);
                    for t in 0..T {
                        let a = acts.get_unchecked(t);
                        let bv = _mm512_set1_epi32(
                            (a.b.as_ptr().add(g * 128 + q * 16 + s * 4) as *const i32)
                                .read_unaligned(),
                        );
                        acc[t] = _mm512_dpbusd_epi32(acc[t], c, bv);
                    }
                }
            }
            for t in 0..T {
                let sc = _mm512_mul_ps(
                    dw,
                    _mm512_set1_ps(*acts.get_unchecked(t).d.get_unchecked(sp)),
                );
                accf[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[t]), sc, accf[t]);
            }
            q0 += qps;
        }
    }
    for t in 0..T {
        _mm512_storeu_ps(out[t].as_mut_ptr(), accf[t]);
    }
}

fn main() {
    let (rows, n) = (2048usize, 4096usize);
    let rb = row_bytes(Format::PQ2_0, n);
    let mut w = vec![0u8; rows * rb];
    let mut s = 1u64;
    for b in w.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = (s & 0x55) as u8;
    }
    for blk in w.chunks_exact_mut(34) {
        blk[..2].copy_from_slice(&f32_to_f16(0.02).to_le_bytes());
    }
    const T: usize = 6;
    let acts: Vec<QuantAct> = (0..T)
        .map(|k| {
            let x: Vec<f32> = (0..n).map(|i| ((i * 31 + k) % 17) as f32 - 8.0).collect();
            let mut a = QuantAct::with_capacity(n);
            a.ternary = true;
            a.fill(&x);
            a
        })
        .collect();
    let mut o = [0.0f32; T];
    let reps = 20;
    macro_rules! run {
        ($name:expr, $v:expr) => {{
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..reps {
                    let mut r = 0;
                    while r + 2 <= rows {
                        unsafe {
                            k2r::<T, $v>(
                                &w[r * rb..(r + 1) * rb],
                                &w[(r + 1) * rb..(r + 2) * rb],
                                &acts,
                                &mut o,
                            )
                        };
                        r += 2;
                    }
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            std::hint::black_box(&o);
            println!(
                "{:34} {:6.1} GMAC/s",
                $name,
                (reps * rows * n * T) as f64 / best / 1e9
            );
        }};
    }
    run!("current (2 rows x 6 acts)", 0);
    run!("no float epilogue", 1);
    run!("no activation loads", 2);
    run!("no weight decode", 3);

    // The row-interleaved tile kernel, against the current one at several widths, and its result against the scalar spec.
    let wt = repack16(&w, rows, rb);
    let ng = rb / 34;
    let xs: Vec<Vec<f32>> = (0..16)
        .map(|k| {
            (0..n)
                .map(|i| ((i * 31 + k * 7) % 17) as f32 - 8.0 + 0.01 * k as f32)
                .collect()
        })
        .collect();
    macro_rules! tiled {
        ($t:expr, $span:expr) => {{
            let ta: Vec<TAct> = xs[..$t].iter().map(|x| tact(x, $span)).collect();
            let mut out = vec![[0f32; 16]; $t];
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t0 = Instant::now();
                for _ in 0..reps {
                    for tile in 0..rows / 16 {
                        unsafe { tile16::<$t, $span>(&wt[tile * 16 * rb..], ng, &ta, &mut out) };
                    }
                }
                best = best.min(t0.elapsed().as_secs_f64());
            }
            // check row 0..16 of tile 0 against a float reference of the same quantised operands
            let mut maxrel = 0f64;
            for t in 0..$t {
                for r in 0..16 {
                    let mut y = vec![0f32; n];
                    peregrine_core::quant::dequant_row(
                        Format::PQ2_0,
                        &w[r * rb..(r + 1) * rb],
                        &mut y,
                    );
                    let mut want = 0f64;
                    for i in 0..n {
                        let sp = i / $span;
                        let qv = {
                            let blk = &xs[t][sp * $span..(sp + 1) * $span];
                            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                            (xs[t][i] * (127.0 / amax)).round().clamp(-127.0, 127.0)
                                * (amax / 127.0)
                        };
                        want += y[i] as f64 * qv as f64;
                    }
                    unsafe { tile16::<$t, $span>(&wt[..], ng, &ta, &mut out) };
                    let got = out[t][r] as f64;
                    maxrel = maxrel.max((got - want).abs() / want.abs().max(1.0));
                }
            }
            println!(
                "tile16 T={:2} span={:3}             {:6.1} GMAC/s   (max rel err {:.1e})",
                $t,
                $span,
                (reps * rows * n * $t) as f64 / best / 1e9,
                maxrel
            );
        }};
    }
    tiled!(1, 16);
    tiled!(8, 16);
    tiled!(1, 32);
    tiled!(4, 32);
    tiled!(8, 32);
    tiled!(12, 32);
    tiled!(16, 32);
    tiled!(8, 64);
    tiled!(8, 128);
    tiled!(16, 128);
}
