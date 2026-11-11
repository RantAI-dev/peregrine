//! The packed float GEMM against the float tile it replaces, and against int8.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example fpbench            # every shape
//! PEREGRINE_ISA=avx2 cargo run --release -p peregrine-bench --example fpbench
//! ```
//!
//! One thread. For each (n_in, n_out) and activation-row count it reports GMAC/s
//! for: the old float path (`simd::dot_batch`: one weight row against four
//! activations, no FMA, a horizontal sum per output), the packed kernel
//! (`fpack::gemm`: `PEREGRINE_ISA` picks AVX2 or AVX-512), the repacked Q8_0 tile on
//! the same shape, and a register-only FMA chain as this core's peak.

use peregrine_kernels::cpu::Tier;
use peregrine_kernels::fpack;
use peregrine_kernels::kqsimd;
use peregrine_kernels::quant::{f32_to_f16, row_bytes, QuantAct};
use peregrine_kernels::table::MatKernels;
use peregrine_kernels::Format;
use std::time::Instant;

const SHAPES: [(usize, usize); 5] = [
    (384, 384),
    (384, 1536),
    (1024, 1024),
    (1024, 3072),
    (2048, 8192),
];
const ROWS: [usize; 5] = [1, 6, 16, 64, 128];
const REP_SECS: f64 = 0.25;

fn prng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn value(next: &mut impl FnMut() -> u64) -> f32 {
    let u = next();
    ((u >> 40) as f32 / (1u64 << 24) as f32) * 3.0 - 1.5
}

/// One weight matrix in the file's row layout.
fn weights(fmt: Format, n_out: usize, n_in: usize, seed: u64) -> Vec<u8> {
    let eb = fpack::elem_bytes(fmt);
    let mut next = prng(seed);
    let mut w = vec![0u8; n_out * n_in * eb];
    for r in 0..n_out {
        for k in 0..n_in {
            let v = value(&mut next);
            let at = (r * n_in + k) * eb;
            match fmt {
                Format::F32 => w[at..at + 4].copy_from_slice(&v.to_le_bytes()),
                Format::F16 => w[at..at + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes()),
                _ => w[at..at + 2].copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes()),
            }
        }
    }
    w
}

fn packed(fmt: Format, w: &[u8], n_out: usize, n_in: usize) -> Vec<u8> {
    let rb = n_in * fpack::elem_bytes(fmt);
    let pb = fpack::panel_bytes(fmt, n_in);
    let mut p = vec![0u8; fpack::packed_len(fmt, n_out, n_in)];
    for panel in 0..n_out.div_ceil(fpack::PANEL) {
        let lo = panel * fpack::PANEL;
        let hi = (lo + fpack::PANEL).min(n_out);
        let rows: Vec<&[u8]> = (lo..hi).map(|r| &w[r * rb..(r + 1) * rb]).collect();
        fpack::pack_panel(fmt, &rows, n_in, &mut p[panel * pb..(panel + 1) * pb]);
    }
    p
}

fn acts(rows: usize, n: usize, seed: u64) -> Vec<QuantAct> {
    let mut next = prng(seed);
    (0..rows)
        .map(|_| {
            let x: Vec<f32> = (0..n).map(|_| value(&mut next)).collect();
            let mut a = QuantAct::with_capacity(n);
            a.keep_f32 = true;
            a.fill(&x);
            a
        })
        .collect()
}

/// A Q8_0 matrix, repacked eight rows at a time.
fn q8_weights(n_out: usize, n_in: usize, seed: u64) -> Vec<u8> {
    let nb = n_in / 32;
    let mut next = prng(seed);
    let mut out = vec![0u8; (n_out / 8) * nb * 272];
    for t in 0..n_out / 8 {
        for s in 0..nb {
            let blocks: Vec<Vec<u8>> = (0..8)
                .map(|_| {
                    let mut b = vec![0u8; 34];
                    let d = ((next() >> 40) as f32 / (1u64 << 24) as f32) * 0.02;
                    b[0..2].copy_from_slice(&f32_to_f16(d.max(1e-4)).to_le_bytes());
                    for x in b[2..].iter_mut() {
                        *x = next() as u8;
                    }
                    b
                })
                .collect();
            let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
            peregrine_kernels::kq::repack_q8_08(&refs, &mut out[(t * nb + s) * 272..][..272]);
        }
    }
    out
}

/// `reps` seconds of `f`, returning GMAC/s.
fn run(macs: u128, mut f: impl FnMut()) -> f64 {
    // Warm up, then time enough repetitions that a 1 ms slice is noise.
    f();
    let t0 = Instant::now();
    let mut n = 0u64;
    loop {
        f();
        n += 1;
        if t0.elapsed().as_secs_f64() >= 0.03 {
            break;
        }
    }
    let per = t0.elapsed().as_secs_f64() / n as f64;
    let target = (REP_SECS / per).max(1.0) as u64;
    let t0 = Instant::now();
    for _ in 0..target {
        f();
    }
    let secs = t0.elapsed().as_secs_f64();
    macs as f64 * target as f64 / secs / 1e9
}

/// This core's peak: twelve independent FMA chains in registers.
fn fma_peak() -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn peak(rounds: usize) -> f32 {
            use std::arch::x86_64::*;
            let mut a = [_mm256_set1_ps(1.000_001); 12];
            let b = std::hint::black_box(_mm256_set1_ps(0.999_999));
            let c = _mm256_set1_ps(1.0e-7);
            for _ in 0..rounds {
                for x in a.iter_mut() {
                    *x = _mm256_fmadd_ps(*x, b, c);
                }
            }
            let mut s = _mm256_setzero_ps();
            for x in a {
                s = _mm256_add_ps(s, x);
            }
            let mut l = [0.0f32; 8];
            _mm256_storeu_ps(l.as_mut_ptr(), s);
            l.iter().sum()
        }
        const ROUNDS: usize = 2000;
        run((12 * 8 * ROUNDS) as u128, || {
            std::hint::black_box(unsafe { peak(ROUNDS) });
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    0.0
}

fn main() {
    let tier = peregrine_kernels::cpu::tier_name();
    println!("tier {tier}, one thread, GMAC/s (1 GMAC = 10^9 multiply-adds)");
    let peak = fma_peak();
    println!("FMA peak (12 ymm chains): {peak:.1}");
    let q8 = MatKernels::get().kq_tiles();
    for (n_in, n_out) in SHAPES {
        println!("\n== n_in {n_in}, n_out {n_out}");
        println!(
            "{:>4}  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>8}",
            "rows", "f32 old", "f32 new", "f16 old", "f16 new", "bf16 old", "bf16 new", "int8 q8_0"
        );
        for nb in ROWS {
            let xs = acts(nb, n_in, 0xabc ^ nb as u64);
            let o = format_row(n_in, n_out, nb, &xs, q8);
            println!(
                "{:>4}  {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1}  {:>8.1}",
                nb, o[0], o[1], o[2], o[3], o[4], o[5], o[6]
            );
        }
    }
    let _ = Tier::Avx2;
}

/// `[old f32, new f32, old f16, new f16, old bf16, new bf16, int8]` GMAC/s.
fn format_row(n_in: usize, n_out: usize, nb: usize, xs: &[QuantAct], q8: bool) -> [f64; 7] {
    let macs = (n_out as u128) * (n_in as u128) * (nb as u128);
    let w32 = weights(Format::F32, n_out, n_in, 11);
    let w16 = weights(Format::F16, n_out, n_in, 12);
    let wbf = weights(Format::BF16, n_out, n_in, 14);
    let p32 = packed(Format::F32, &w32, n_out, n_in);
    let p16 = packed(Format::F16, &w16, n_out, n_in);
    let pbf = packed(Format::BF16, &wbf, n_out, n_in);
    let panels = n_out.div_ceil(fpack::PANEL);
    let mut out = vec![0.0f32; panels * fpack::PANEL * nb];

    // The old path: one weight row against up to four activations, exactly what
    // `MatPath::PerRow` runs (the batched float tile takes four at a time).
    let mut old = vec![0.0f32; nb];
    let old32 = run(macs, || {
        let rb = n_in * 4;
        for r in 0..n_out {
            peregrine_kernels::quant::dot_batch(
                Format::F32,
                &w32[r * rb..(r + 1) * rb],
                xs,
                &mut old,
            );
        }
        std::hint::black_box(&old);
    });
    let old16 = run(macs, || {
        let rb = n_in * 2;
        for r in 0..n_out {
            peregrine_kernels::quant::dot_batch(
                Format::F16,
                &w16[r * rb..(r + 1) * rb],
                xs,
                &mut old,
            );
        }
        std::hint::black_box(&old);
    });
    let new32 = run(macs, || {
        fpack::gemm(Format::F32, &p32, n_in, panels, xs, None, &mut out);
        std::hint::black_box(&out);
    });
    let new16 = run(macs, || {
        fpack::gemm(Format::F16, &p16, n_in, panels, xs, None, &mut out);
        std::hint::black_box(&out);
    });
    let oldbf = run(macs, || {
        let rb = n_in * 2;
        for r in 0..n_out {
            peregrine_kernels::quant::dot_batch(
                Format::BF16,
                &wbf[r * rb..(r + 1) * rb],
                xs,
                &mut old,
            );
        }
        std::hint::black_box(&old);
    });
    let newbf = run(macs, || {
        fpack::gemm(Format::BF16, &pbf, n_in, panels, xs, None, &mut out);
        std::hint::black_box(&out);
    });
    let int8 = if q8 && n_in.is_multiple_of(32) && n_out.is_multiple_of(8) {
        let wq = q8_weights(n_out, n_in, 13);
        let nblk = n_in / 32;
        let mut q = vec![0.0f32; 8 * nb];
        let tile = nblk * 272;
        run((n_out as u128) * (n_in as u128) * (nb as u128), || {
            for t in 0..n_out / 8 {
                unsafe {
                    kqsimd::q8_08_gemm(&wq[t * tile..(t + 1) * tile], nblk, xs, nb, &mut q);
                }
            }
            std::hint::black_box(&q);
        })
    } else {
        0.0
    };
    let _ = row_bytes(Format::F32, n_in);
    [old32, new32, old16, new16, oldbf, newbf, int8]
}
