//! Stage benchmark for the decoder attention tile: scores, softmax and the value mix.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example attnstage -- [reps]
//! ```
//!
//! One thread (pin it outside). Every body is timed over the shapes the
//! Qwen3-Embedding prefill sees (tiles of 64 keys, 16 rows per KV head) and reported as
//! GMAC/s (kernels) or ns/element (softmax stages).

use std::hint::black_box;
use std::time::Instant;

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    for &hd in &[128usize, 64] {
        for &keys in &[64usize, 32] {
            println!("--- hd {hd}, {keys} keys");
            mix_suite(hd, keys, reps);
            score_suite(hd, keys, reps);
        }
    }
    softmax_suite(reps);
}

fn mix_suite(hd: usize, keys: usize, reps: usize) {
    let o = vec![0.0f32; 2 * hd];
    let p: Vec<f32> = (0..keys).map(|j| 1.0 / (j + 1) as f32).collect();
    let v: Vec<u16> = (0..keys * hd)
        .map(|i| half((i % 37) as f32 * 0.01))
        .collect();
    let vb: Vec<u8> = v.iter().flat_map(|h| h.to_le_bytes()).collect();
    let mac = 2.0 * keys as f64 * hd as f64;

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let mut o0 = o.clone();
        let mut o1 = o.clone();
        for _ in 0..reps {
            // SAFETY: NEON is aarch64's baseline; slices cover the tile.
            unsafe {
                peregrine_kernels::neon::mix_f16_pair(&mut o0, &mut o1, 1.0, 1.0, &p, &p, &vb, hd)
            };
            o0[0] = 0.0;
            o1[0] = 0.0;
        }
        black_box(&o0);
        black_box(&o1);
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "mix_pair_8   {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        mac / best / 1e9
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let mut o0 = o.clone();
        let mut o1 = o.clone();
        for _ in 0..reps {
            unsafe { mix_pair_16(&mut o0, &mut o1, 1.0, 1.0, &p, &p, &vb, hd) };
            o0[0] = 0.0;
            o1[0] = 0.0;
        }
        black_box(&o0);
        black_box(&o1);
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "mix_pair_16  {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        mac / best / 1e9
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let mut o0 = o.clone();
        let mut o1 = o.clone();
        for _ in 0..reps {
            unsafe { mix_pair_32(&mut o0, &mut o1, &p, &p, &vb, hd) };
            o0[0] = 0.0;
            o1[0] = 0.0;
        }
        black_box(&o0);
        black_box(&o1);
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "mix_pair_32  {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        mac / best / 1e9
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let mut o0 = o.clone();
        let mut o1 = o.clone();
        let mut o2 = o.clone();
        let mut o3 = o.clone();
        for _ in 0..reps {
            unsafe {
                mix_quad_8(
                    &mut o0, &mut o1, &mut o2, &mut o3, 1.0, &p, &p, &p, &p, &vb, hd,
                )
            };
            o0[0] = 0.0;
        }
        black_box((&o0, &o1, &o2, &o3));
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "mix_quad_8   {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        2.0 * mac / best / 1e9
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let mut o0 = o.clone();
        let mut o1 = o.clone();
        let mut o2 = o.clone();
        let mut o3 = o.clone();
        for _ in 0..reps {
            unsafe { mix_quad_16(&mut o0, &mut o1, &mut o2, &mut o3, &p, &p, &p, &p, &vb, hd) };
            o0[0] = 0.0;
        }
        black_box((&o0, &o1, &o2, &o3));
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "mix_quad_16  {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        2.0 * mac / best / 1e9
    );

    for &rows in &[4usize, 8, 16] {
        let mut pp = vec![0.0f32; rows * keys];
        for r in 0..rows {
            pp[r * keys..(r + 1) * keys].copy_from_slice(&p);
        }
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            let mut os = vec![0.0f32; rows * hd];
            for _ in 0..reps {
                unsafe { mix_oct_8(&mut os, &pp, keys, &vb, hd, rows) };
                os[0] = 0.0;
            }
            black_box(&os);
            best = best.min(t.elapsed().as_secs_f64() / reps as f64);
        }
        println!(
            "mix_rows{rows}   {:.1} us, {:.1} GMAC/s",
            best * 1e6,
            2.0 * rows as f64 * keys as f64 * hd as f64 / best / 1e9
        );
    }
}

fn score_suite(hd: usize, keys: usize, reps: usize) {
    let q: Vec<f32> = (0..4 * hd)
        .map(|i| ((i * 13) % 29) as f32 * 0.01 - 0.14)
        .collect();
    let kk: Vec<u16> = (0..keys * hd)
        .map(|i| half((i % 23) as f32 * 0.01))
        .collect();
    let kb: Vec<u8> = kk.iter().flat_map(|h| h.to_le_bytes()).collect();
    let scale = 1.0 / (hd as f32).sqrt();
    let mut s = vec![0.0f32; 4 * keys];
    let mac2 = 2.0 * keys as f64 * hd as f64;
    let mac4 = 4.0 * keys as f64 * hd as f64;

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let (mut s0, mut s1) = (vec![0.0f32; keys], vec![0.0f32; keys]);
        for _ in 0..reps {
            unsafe { scores_pair_2k(&q, &kb, keys, hd, scale, &mut s0, &mut s1) };
            s0[0] = 0.0;
            s1[0] = 0.0;
        }
        black_box((&s0, &s1));
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "scores 2r2k  {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        mac2 / best / 1e9
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let (mut s0, mut s1) = (vec![0.0f32; keys], vec![0.0f32; keys]);
        for _ in 0..reps {
            unsafe { scores_pair_4k(&q, &kb, keys, hd, scale, &mut s0, &mut s1) };
            s0[0] = 0.0;
            s1[0] = 0.0;
        }
        black_box((&s0, &s1));
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "scores 2r4k  {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        mac2 / best / 1e9
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let (mut s0, mut s1) = (vec![0.0f32; keys], vec![0.0f32; keys]);
        for _ in 0..reps {
            unsafe { scores_pair_8k(&q, &kb, keys, hd, scale, &mut s0, &mut s1) };
            s0[0] = 0.0;
            s1[0] = 0.0;
        }
        black_box((&s0, &s1));
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "scores 2r8k  {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        mac2 / best / 1e9
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..reps {
            let (a, b) = s.split_at_mut(2 * keys);
            unsafe { scores_4r(&q, &kb, keys, hd, scale, a, b) };
            s[0] = 0.0;
        }
        black_box(&s);
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "scores 4r2k  {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        mac4 / best / 1e9
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..reps {
            let (a, b) = s.split_at_mut(2 * keys);
            unsafe { scores_4r4k(&q, &kb, keys, hd, scale, a, b) };
            s[0] = 0.0;
        }
        black_box(&s);
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "scores 4r4k  {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        mac4 / best / 1e9
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let (mut s0, mut s1) = (vec![0.0f32; keys], vec![0.0f32; keys]);
        for _ in 0..reps {
            unsafe { scores_solo_8k(&q, &kb, keys, hd, scale, &mut s0) };
            s0[0] = 0.0;
        }
        black_box((&s0, &s1));
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "scores 1r8k  {:.1} us, {:.1} GMAC/s",
        best * 1e6,
        keys as f64 * hd as f64 / best / 1e9
    );
}

fn softmax_suite(reps: usize) {
    let rows = 16usize;
    let keys = 64usize;
    let stride = peregrine_core::attn::BLOCK;
    let base: Vec<f32> = (0..rows * stride)
        .map(|i| ((i * 37) % 101) as f32 * 0.02 - 1.0)
        .collect();
    let mut tile = base.clone();

    let mut best = f64::INFINITY;
    let mut acc = 0.0f32;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..reps {
            for r in 0..rows {
                let s = &mut tile[r * stride..r * stride + keys];
                s[reps % keys] = (acc * 1e-9).sin();
                acc = acc.max(peregrine_core::vmath::max(s));
                black_box(&mut tile[r * stride]);
            }
        }
        best = best.min(t.elapsed().as_secs_f64() / (reps * rows) as f64);
    }
    println!(
        "max          per row of {keys}: {:.1} ns ({:.3} ns/element) [{acc}]",
        best * 1e9,
        best * 1e9 / keys as f64
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..reps {
            for r in 0..rows {
                let s = &mut tile[r * stride..r * stride + keys];
                let mx = peregrine_core::vmath::max(s);
                acc += peregrine_core::vmath::exp_shift_sum(s, mx) as f32;
                black_box(&mut tile[r * stride]);
            }
        }
        best = best.min(t.elapsed().as_secs_f64() / (reps * rows) as f64);
    }
    println!(
        "max+exp_sum  per row of {keys}: {:.1} ns ({:.3} ns/element)",
        best * 1e9,
        best * 1e9 / keys as f64
    );

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..reps {
            for b in 0..rows / 4 {
                let mut mx = [0.0f32; 4];
                for r in 0..4 {
                    let s = &tile[(b * 4 + r) * stride..(b * 4 + r) * stride + keys];
                    mx[r] = peregrine_core::vmath::max(s);
                }
                let mut sums = [0.0f64; 4];
                peregrine_kernels::bidir::exp_rows4_sums(
                    &mut tile[b * 4 * stride..(b * 4 + 4) * stride],
                    stride,
                    keys,
                    mx,
                    &mut sums,
                );
                acc += (sums[0] + sums[1] + sums[2] + sums[3]) as f32;
                black_box(&mut tile[b * 4 * stride]);
            }
        }
        best = best.min(t.elapsed().as_secs_f64() / (reps * rows) as f64);
    }
    println!(
        "rows4 exp    per row of {keys}: {:.1} ns ({:.3} ns/element) [{acc}]",
        best * 1e9,
        best * 1e9 / keys as f64
    );
}

#[inline]
fn half(x: f32) -> u16 {
    peregrine_kernels::quant::f32_to_f16(x)
}

#[inline(always)]
unsafe fn hsum4(v: std::arch::aarch64::float32x4_t) -> f32 {
    use std::arch::aarch64::*;
    let t = vpaddq_f32(v, v);
    let t = vpaddq_f32(t, t);
    vgetq_lane_f32(t, 0)
}

// The macro above does not share the widened key between the rows the way the real
// body does; the concrete bodies below are written out.

#[target_feature(enable = "neon")]
unsafe fn scores_pair_2k(
    q: &[f32],
    k: &[u8],
    keys: usize,
    hd: usize,
    scale: f32,
    s0: &mut [f32],
    s1: &mut [f32],
) {
    use std::arch::aarch64::*;
    let (q0p, q1p, kp) = (q.as_ptr(), q.as_ptr().add(hd), k.as_ptr());
    let mut j = 0;
    while j + 2 <= keys {
        let mut a0 = [[vdupq_n_f32(0.0); 2]; 2];
        let mut a1 = [[vdupq_n_f32(0.0); 2]; 2];
        let mut i = 0;
        while i < hd {
            let q00 = vld1q_f32(q0p.add(i));
            let q01 = vld1q_f32(q0p.add(i + 4));
            let q10 = vld1q_f32(q1p.add(i));
            let q11 = vld1q_f32(q1p.add(i + 4));
            for u in 0..2 {
                let at = ((j + u) * hd + i) * 2;
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                a0[u][0] = vfmaq_f32(a0[u][0], q00, k0);
                a0[u][1] = vfmaq_f32(a0[u][1], q01, k1);
                a1[u][0] = vfmaq_f32(a1[u][0], q10, k0);
                a1[u][1] = vfmaq_f32(a1[u][1], q11, k1);
            }
            i += 8;
        }
        for u in 0..2 {
            s0[j + u] = (hsum4(a0[u][0]) + hsum4(a0[u][1])) * scale;
            s1[j + u] = (hsum4(a1[u][0]) + hsum4(a1[u][1])) * scale;
        }
        j += 2;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scores_pair_4k(
    q: &[f32],
    k: &[u8],
    keys: usize,
    hd: usize,
    scale: f32,
    s0: &mut [f32],
    s1: &mut [f32],
) {
    use std::arch::aarch64::*;
    let (q0p, q1p, kp) = (q.as_ptr(), q.as_ptr().add(hd), k.as_ptr());
    let mut j = 0;
    while j + 4 <= keys {
        let mut a0 = [[vdupq_n_f32(0.0); 2]; 4];
        let mut a1 = [[vdupq_n_f32(0.0); 2]; 4];
        let mut i = 0;
        while i < hd {
            let q00 = vld1q_f32(q0p.add(i));
            let q01 = vld1q_f32(q0p.add(i + 4));
            let q10 = vld1q_f32(q1p.add(i));
            let q11 = vld1q_f32(q1p.add(i + 4));
            for u in 0..4 {
                let at = ((j + u) * hd + i) * 2;
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                a0[u][0] = vfmaq_f32(a0[u][0], q00, k0);
                a0[u][1] = vfmaq_f32(a0[u][1], q01, k1);
                a1[u][0] = vfmaq_f32(a1[u][0], q10, k0);
                a1[u][1] = vfmaq_f32(a1[u][1], q11, k1);
            }
            i += 8;
        }
        for u in 0..4 {
            s0[j + u] = (hsum4(a0[u][0]) + hsum4(a0[u][1])) * scale;
            s1[j + u] = (hsum4(a1[u][0]) + hsum4(a1[u][1])) * scale;
        }
        j += 4;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scores_pair_8k(
    q: &[f32],
    k: &[u8],
    keys: usize,
    hd: usize,
    scale: f32,
    s0: &mut [f32],
    s1: &mut [f32],
) {
    use std::arch::aarch64::*;
    let (q0p, q1p, kp) = (q.as_ptr(), q.as_ptr().add(hd), k.as_ptr());
    let mut j = 0;
    while j + 8 <= keys {
        let mut a0 = [[vdupq_n_f32(0.0); 2]; 8];
        let mut a1 = [[vdupq_n_f32(0.0); 2]; 8];
        let mut i = 0;
        while i < hd {
            let q00 = vld1q_f32(q0p.add(i));
            let q01 = vld1q_f32(q0p.add(i + 4));
            let q10 = vld1q_f32(q1p.add(i));
            let q11 = vld1q_f32(q1p.add(i + 4));
            for u in 0..8 {
                let at = ((j + u) * hd + i) * 2;
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                a0[u][0] = vfmaq_f32(a0[u][0], q00, k0);
                a0[u][1] = vfmaq_f32(a0[u][1], q01, k1);
                a1[u][0] = vfmaq_f32(a1[u][0], q10, k0);
                a1[u][1] = vfmaq_f32(a1[u][1], q11, k1);
            }
            i += 8;
        }
        for u in 0..8 {
            s0[j + u] = (hsum4(a0[u][0]) + hsum4(a0[u][1])) * scale;
            s1[j + u] = (hsum4(a1[u][0]) + hsum4(a1[u][1])) * scale;
        }
        j += 8;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scores_solo_8k(q: &[f32], k: &[u8], keys: usize, hd: usize, scale: f32, s0: &mut [f32]) {
    use std::arch::aarch64::*;
    let (q0p, kp) = (q.as_ptr(), k.as_ptr());
    let mut j = 0;
    while j + 8 <= keys {
        let mut a0 = [[vdupq_n_f32(0.0); 2]; 8];
        let mut i = 0;
        while i < hd {
            let q00 = vld1q_f32(q0p.add(i));
            let q01 = vld1q_f32(q0p.add(i + 4));
            for u in 0..8 {
                let at = ((j + u) * hd + i) * 2;
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                a0[u][0] = vfmaq_f32(a0[u][0], q00, k0);
                a0[u][1] = vfmaq_f32(a0[u][1], q01, k1);
            }
            i += 8;
        }
        for u in 0..8 {
            s0[j + u] = (hsum4(a0[u][0]) + hsum4(a0[u][1])) * scale;
        }
        j += 8;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scores_4r(
    q: &[f32],
    k: &[u8],
    keys: usize,
    hd: usize,
    scale: f32,
    s0: &mut [f32],
    s1: &mut [f32],
) {
    use std::arch::aarch64::*;
    let qs = [
        q.as_ptr(),
        q.as_ptr().add(hd),
        q.as_ptr().add(2 * hd),
        q.as_ptr().add(3 * hd),
    ];
    let kp = k.as_ptr();
    let mut j = 0;
    while j + 2 <= keys {
        let mut a = [[[vdupq_n_f32(0.0); 2]; 2]; 4];
        let mut i = 0;
        while i < hd {
            let qv = [
                (vld1q_f32(qs[0].add(i)), vld1q_f32(qs[0].add(i + 4))),
                (vld1q_f32(qs[1].add(i)), vld1q_f32(qs[1].add(i + 4))),
                (vld1q_f32(qs[2].add(i)), vld1q_f32(qs[2].add(i + 4))),
                (vld1q_f32(qs[3].add(i)), vld1q_f32(qs[3].add(i + 4))),
            ];
            for u in 0..2 {
                let at = ((j + u) * hd + i) * 2;
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                for r in 0..4 {
                    a[r][u][0] = vfmaq_f32(a[r][u][0], qv[r].0, k0);
                    a[r][u][1] = vfmaq_f32(a[r][u][1], qv[r].1, k1);
                }
            }
            i += 8;
        }
        for u in 0..2 {
            s0[j + u] = (hsum4(a[0][u][0]) + hsum4(a[0][u][1])) * scale;
            s1[j + u] = (hsum4(a[1][u][0]) + hsum4(a[1][u][1])) * scale;
            s0[keys + j + u] = (hsum4(a[2][u][0]) + hsum4(a[2][u][1])) * scale;
            s1[keys + j + u] = (hsum4(a[3][u][0]) + hsum4(a[3][u][1])) * scale;
        }
        j += 2;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scores_4r4k(
    q: &[f32],
    k: &[u8],
    keys: usize,
    hd: usize,
    scale: f32,
    s0: &mut [f32],
    s1: &mut [f32],
) {
    use std::arch::aarch64::*;
    let qs = [
        q.as_ptr(),
        q.as_ptr().add(hd),
        q.as_ptr().add(2 * hd),
        q.as_ptr().add(3 * hd),
    ];
    let kp = k.as_ptr();
    let mut j = 0;
    while j + 4 <= keys {
        let mut a = [[[vdupq_n_f32(0.0); 2]; 4]; 4];
        let mut i = 0;
        while i < hd {
            let qv = [
                (vld1q_f32(qs[0].add(i)), vld1q_f32(qs[0].add(i + 4))),
                (vld1q_f32(qs[1].add(i)), vld1q_f32(qs[1].add(i + 4))),
                (vld1q_f32(qs[2].add(i)), vld1q_f32(qs[2].add(i + 4))),
                (vld1q_f32(qs[3].add(i)), vld1q_f32(qs[3].add(i + 4))),
            ];
            for u in 0..4 {
                let at = ((j + u) * hd + i) * 2;
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                for r in 0..2 {
                    a[r][u][0] = vfmaq_f32(a[r][u][0], qv[r].0, k0);
                    a[r][u][1] = vfmaq_f32(a[r][u][1], qv[r].1, k1);
                }
                for r in 2..4 {
                    a[r][u][0] = vfmaq_f32(a[r][u][0], qv[r].0, k0);
                    a[r][u][1] = vfmaq_f32(a[r][u][1], qv[r].1, k1);
                }
            }
            i += 8;
        }
        for u in 0..4 {
            s0[j + u] = (hsum4(a[0][u][0]) + hsum4(a[0][u][1])) * scale;
            s1[j + u] = (hsum4(a[1][u][0]) + hsum4(a[1][u][1])) * scale;
            s0[keys + j + u] = (hsum4(a[2][u][0]) + hsum4(a[2][u][1])) * scale;
            s1[keys + j + u] = (hsum4(a[3][u][0]) + hsum4(a[3][u][1])) * scale;
        }
        j += 4;
    }
}

/// Sixteen-dimension slices: four accumulator chains per row instead of two.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_pair_16(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    v: &[u8],
    hd: usize,
) {
    use std::arch::aarch64::*;
    let vp = v.as_ptr();
    let mut d = 0;
    while d + 16 <= hd {
        let mut a0 = [
            vld1q_f32(o0.as_ptr().add(d)),
            vld1q_f32(o0.as_ptr().add(d + 4)),
            vld1q_f32(o0.as_ptr().add(d + 8)),
            vld1q_f32(o0.as_ptr().add(d + 12)),
        ];
        let mut a1 = [
            vld1q_f32(o1.as_ptr().add(d)),
            vld1q_f32(o1.as_ptr().add(d + 4)),
            vld1q_f32(o1.as_ptr().add(d + 8)),
            vld1q_f32(o1.as_ptr().add(d + 12)),
        ];
        for (j, (&w0, &w1)) in p0.iter().zip(p1).enumerate() {
            let at = (j * hd + d) * 2;
            let mut vv = [vdupq_n_f32(0.0); 4];
            for (q, vq) in vv.iter_mut().enumerate() {
                *vq = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(
                    vp.add(at + q * 8) as *const u16
                )));
            }
            let wv0 = vdupq_n_f32(w0);
            let wv1 = vdupq_n_f32(w1);
            for q in 0..4 {
                a0[q] = vaddq_f32(a0[q], vmulq_f32(wv0, vv[q]));
                a1[q] = vaddq_f32(a1[q], vmulq_f32(wv1, vv[q]));
            }
        }
        for q in 0..4 {
            vst1q_f32(o0.as_mut_ptr().add(d + q * 4), a0[q]);
            vst1q_f32(o1.as_mut_ptr().add(d + q * 4), a1[q]);
        }
        d += 16;
    }
    let _ = (alpha0, alpha1);
}

/// Four rows over one V tile, two accumulator chains per row per slice.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_quad_8(
    o0: &mut [f32],
    o1: &mut [f32],
    o2: &mut [f32],
    o3: &mut [f32],
    alpha: f32,
    p0: &[f32],
    p1: &[f32],
    p2: &[f32],
    p3: &[f32],
    v: &[u8],
    hd: usize,
) {
    use std::arch::aarch64::*;
    let vp = v.as_ptr();
    let mut d = 0;
    while d + 8 <= hd {
        let mut a = [
            vld1q_f32(o0.as_ptr().add(d)),
            vld1q_f32(o0.as_ptr().add(d + 4)),
            vld1q_f32(o1.as_ptr().add(d)),
            vld1q_f32(o1.as_ptr().add(d + 4)),
            vld1q_f32(o2.as_ptr().add(d)),
            vld1q_f32(o2.as_ptr().add(d + 4)),
            vld1q_f32(o3.as_ptr().add(d)),
            vld1q_f32(o3.as_ptr().add(d + 4)),
        ];
        for (j, (((&w0, &w1), &w2), &w3)) in p0.iter().zip(p1).zip(p2).zip(p3).enumerate() {
            let at = (j * hd + d) * 2;
            let v0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(at) as *const u16)));
            let v1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(at + 8) as *const u16)));
            let w = vdupq_n_f32(w0);
            for q in 0..2 {
                a[q] = vaddq_f32(a[q], vmulq_f32(w, if q == 0 { v0 } else { v1 }));
            }
            let w = vdupq_n_f32(w1);
            for q in 0..2 {
                a[2 + q] = vaddq_f32(a[2 + q], vmulq_f32(w, if q == 0 { v0 } else { v1 }));
            }
            let w = vdupq_n_f32(w2);
            for q in 0..2 {
                a[4 + q] = vaddq_f32(a[4 + q], vmulq_f32(w, if q == 0 { v0 } else { v1 }));
            }
            let w = vdupq_n_f32(w3);
            for q in 0..2 {
                a[6 + q] = vaddq_f32(a[6 + q], vmulq_f32(w, if q == 0 { v0 } else { v1 }));
            }
        }
        for q in 0..2 {
            vst1q_f32(o0.as_mut_ptr().add(d + q * 4), a[q]);
            vst1q_f32(o1.as_mut_ptr().add(d + q * 4), a[2 + q]);
            vst1q_f32(o2.as_mut_ptr().add(d + q * 4), a[4 + q]);
            vst1q_f32(o3.as_mut_ptr().add(d + q * 4), a[6 + q]);
        }
        d += 8;
    }
    let _ = alpha;
}

/// Four rows, sixteen-dimension slices: sixteen chains.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_quad_16(
    o0: &mut [f32],
    o1: &mut [f32],
    o2: &mut [f32],
    o3: &mut [f32],
    p0: &[f32],
    p1: &[f32],
    p2: &[f32],
    p3: &[f32],
    v: &[u8],
    hd: usize,
) {
    use std::arch::aarch64::*;
    let vp = v.as_ptr();
    let mut d = 0;
    while d + 16 <= hd {
        let mut a = [[vdupq_n_f32(0.0); 4]; 4];
        for (r, o) in [&*o0, &*o1, &*o2, &*o3].iter().enumerate() {
            for q in 0..4 {
                a[r][q] = vld1q_f32(o.as_ptr().add(d + q * 4));
            }
        }
        for (j, (((&w0, &w1), &w2), &w3)) in p0.iter().zip(p1).zip(p2).zip(p3).enumerate() {
            let at = (j * hd + d) * 2;
            let mut vv = [vdupq_n_f32(0.0); 4];
            for (q, vq) in vv.iter_mut().enumerate() {
                *vq = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(
                    vp.add(at + q * 8) as *const u16
                )));
            }
            for (r, w) in [w0, w1, w2, w3].iter().enumerate() {
                let wv = vdupq_n_f32(*w);
                for q in 0..4 {
                    a[r][q] = vaddq_f32(a[r][q], vmulq_f32(wv, vv[q]));
                }
            }
        }
        for (r, o) in [&mut *o0, &mut *o1, &mut *o2, &mut *o3]
            .iter_mut()
            .enumerate()
        {
            for q in 0..4 {
                vst1q_f32(o.as_mut_ptr().add(d + q * 4), a[r][q]);
            }
        }
        d += 16;
    }
}

/// Eight rows, eight-dimension slices: sixteen chains.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_oct_8(o: &mut [f32], p: &[f32], p_stride: usize, v: &[u8], hd: usize, rows: usize) {
    use std::arch::aarch64::*;
    let vp = v.as_ptr();
    let mut d = 0;
    while d + 8 <= hd {
        let mut a = [[vdupq_n_f32(0.0); 2]; 16];
        for r in 0..rows {
            a[r][0] = vld1q_f32(o.as_ptr().add(r * hd + d));
            a[r][1] = vld1q_f32(o.as_ptr().add(r * hd + d + 4));
        }
        for j in 0..p_stride {
            let at = (j * hd + d) * 2;
            let v0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(at) as *const u16)));
            let v1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(at + 8) as *const u16)));
            for r in 0..rows {
                let w = vdupq_n_f32(p[r * p_stride + j]);
                a[r][0] = vaddq_f32(a[r][0], vmulq_f32(w, v0));
                a[r][1] = vaddq_f32(a[r][1], vmulq_f32(w, v1));
            }
        }
        for r in 0..rows {
            vst1q_f32(o.as_mut_ptr().add(r * hd + d), a[r][0]);
            vst1q_f32(o.as_mut_ptr().add(r * hd + d + 4), a[r][1]);
        }
        d += 8;
    }
}

/// Two rows over one V tile with a 32-dimension accumulator slice.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_pair_32(o0: &mut [f32], o1: &mut [f32], p0: &[f32], p1: &[f32], v: &[u8], hd: usize) {
    use std::arch::aarch64::*;
    let vp = v.as_ptr();
    let mut d = 0;
    while d + 32 <= hd {
        let mut a0 = [vdupq_n_f32(0.0); 8];
        let mut a1 = [vdupq_n_f32(0.0); 8];
        for q in 0..8 {
            a0[q] = vld1q_f32(o0.as_ptr().add(d + q * 4));
            a1[q] = vld1q_f32(o1.as_ptr().add(d + q * 4));
        }
        for (j, (&w0, &w1)) in p0.iter().zip(p1).enumerate() {
            let at = (j * hd + d) * 2;
            let mut vv = [vdupq_n_f32(0.0); 8];
            for (q, vq) in vv.iter_mut().enumerate() {
                *vq = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(
                    vp.add(at + q * 8) as *const u16
                )));
            }
            let wv0 = vdupq_n_f32(w0);
            let wv1 = vdupq_n_f32(w1);
            for q in 0..8 {
                a0[q] = vaddq_f32(a0[q], vmulq_f32(wv0, vv[q]));
                a1[q] = vaddq_f32(a1[q], vmulq_f32(wv1, vv[q]));
            }
        }
        for q in 0..8 {
            vst1q_f32(o0.as_mut_ptr().add(d + q * 4), a0[q]);
            vst1q_f32(o1.as_mut_ptr().add(d + q * 4), a1[q]);
        }
        d += 32;
    }
}
