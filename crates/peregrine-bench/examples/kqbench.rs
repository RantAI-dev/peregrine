//! Repacked k-quant kernel micro-benchmark: the AVX2 and AVX-512 kernels on one tile.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example kqbench -- [n_sb] [reps]
//! ```
//!
//! One thread, a synthetic eight-row tile and synthetic activations, alternating
//! the two kernels so a busy machine moves both. Reports ns per (tile, activation
//! row, super-block) for the GEMV (1 row) and the GEMM (4 and 64 rows), and then the
//! row kernel against the tile kernel at 1, 16, 64 and 128 activation rows (S5's
//! widths: a decode row, the prefill chunk, and the widths beyond it).

#[cfg(target_arch = "x86_64")]
use peregrine_kernels::kq::{
    repack_q4k8, repack_q5_08, repack_q5k8, repack_q6k8, KqActs, Q4K8, Q4K_BLOCK, Q5K8, Q5_08,
    Q6K8, Q6K_BLOCK,
};
#[cfg(target_arch = "x86_64")]
use peregrine_kernels::kqsimd::{
    q4k8_gemm, q4k8_gemm_zmm, q4k8_gemm_zmm_wide, q4k8_gemv, q4k8_gemv_zmm, q5_08_gemm,
    q5_08_gemm_zmm, q5k8_gemm, q5k8_gemv, q6k8_gemm, q6k8_gemm_zmm, q6k8_gemv,
};
#[cfg(target_arch = "x86_64")]
use peregrine_kernels::quant::QuantAct;
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
fn prng(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn tile(n_sb: usize, seed: u64) -> Vec<u8> {
    let mut out = vec![0u8; n_sb * Q4K8::SIZE];
    for sb in 0..n_sb {
        let blocks: Vec<Vec<u8>> = (0..8)
            .map(|r| {
                let mut b = prng(seed ^ (sb * 8 + r) as u64, 144);
                b[0..2].copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes());
                b[2..4].copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.005).to_le_bytes());
                b
            })
            .collect();
        let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
        repack_q4k8(&refs, &mut out[sb * Q4K8::SIZE..(sb + 1) * Q4K8::SIZE]);
    }
    out
}

#[cfg(target_arch = "x86_64")]
fn tile5(n_blk: usize, seed: u64) -> Vec<u8> {
    let mut out = vec![0u8; n_blk * Q5_08::SIZE];
    for t in 0..n_blk {
        let blocks: Vec<Vec<u8>> = (0..8)
            .map(|r| {
                let mut b = prng(seed ^ (t * 8 + r) as u64, 22);
                b[0..2].copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes());
                b
            })
            .collect();
        let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
        repack_q5_08(&refs, &mut out[t * Q5_08::SIZE..(t + 1) * Q5_08::SIZE]);
    }
    out
}

#[cfg(target_arch = "x86_64")]
fn acts(rows: usize, n: usize, seed: u64) -> Vec<QuantAct> {
    (0..rows)
        .map(|r| {
            let mut s = seed ^ ((r as u64) << 32) | 1;
            let x: Vec<f32> = (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 32) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * 3.0
                })
                .collect();
            let mut a = QuantAct::with_capacity(n);
            a.fill(&x);
            a
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn best(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[0]
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("kqbench benchmarks the x86-only AVX2/AVX-512 k-quant kernels; nothing to run on this architecture");
}

#[cfg(target_arch = "x86_64")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n_sb: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(8);
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    if threads > 1 {
        // The same L1-resident kernel on `threads` cores at once: does the aggregate scale?
        let t0 = Instant::now();
        let hs: Vec<_> = (0..threads)
            .map(|k| {
                std::thread::spawn(move || {
                    let n = n_sb * 256;
                    let t = tile(n_sb, 0x5EED_1234 + k as u64);
                    let batch = acts(64, n, 0xC0DE_5678 + k as u64);
                    let mut kq = KqActs::default();
                    kq.fill(&batch);
                    let mut out = vec![0.0f32; 64 * 8];
                    let zmm = peregrine_kernels::kqsimd::have_zmm512();
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        unsafe {
                            if zmm {
                                q4k8_gemm_zmm(&t, n_sb, &kq, 64, &mut out)
                            } else {
                                q4k8_gemm(&t, n_sb, &kq, 64, &mut out)
                            }
                        }
                    }
                    t0.elapsed().as_secs_f64()
                })
            })
            .collect();
        let secs: Vec<f64> = hs.into_iter().map(|h| h.join().unwrap()).collect();
        let wall = t0.elapsed().as_secs_f64();
        let best = secs.iter().cloned().fold(f64::MAX, f64::min);
        println!(
            "gemm(64) x{threads}: best thread {:.1} GMAC/s/core, aggregate {:.1} GMAC/s, wall {wall:.2}s",
            2048.0 * 64.0 * n_sb as f64 * reps as f64 / best / 1e9,
            2048.0 * 64.0 * n_sb as f64 * reps as f64 * threads as f64 / best / 1e9
        );
        return;
    }
    let n = n_sb * 256;
    let t = tile(n_sb, 0x5EED_1234);
    let batch = acts(128, n, 0xC0DE_5678);
    let mut kq = KqActs::default();
    kq.fill(&batch);
    // The width sweeps below go to 128 activation rows, so the output holds them.
    let mut out = vec![0.0f32; 128 * 8];
    if !peregrine_kernels::simd::have_batched() {
        eprintln!("no AVX2 on this machine");
        return;
    }
    // Warm up.
    unsafe {
        q4k8_gemm(&t, n_sb, &kq, 64, &mut out);
        q4k8_gemv(&t, n_sb, &batch[..1], 1, &mut out);
    }
    let mut g4 = Vec::new();
    let mut g64 = Vec::new();
    let mut v1 = Vec::new();
    let mut z4 = Vec::new();
    let mut z64 = Vec::new();
    let mut z1 = Vec::new();
    let zmm = peregrine_kernels::kqsimd::have_zmm512();
    for _ in 0..5 {
        let t0 = Instant::now();
        for _ in 0..reps {
            unsafe { q4k8_gemm(&t, n_sb, &kq, 4, &mut out) };
        }
        g4.push(t0.elapsed().as_secs_f64());
        let t0 = Instant::now();
        for _ in 0..reps {
            unsafe { q4k8_gemm(&t, n_sb, &kq, 64, &mut out) };
        }
        g64.push(t0.elapsed().as_secs_f64());
        let t0 = Instant::now();
        for _ in 0..reps {
            unsafe { q4k8_gemv(&t, n_sb, &batch[..1], 1, &mut out) };
        }
        v1.push(t0.elapsed().as_secs_f64());
        if zmm {
            let t0 = Instant::now();
            for _ in 0..reps {
                unsafe { q4k8_gemm_zmm(&t, n_sb, &kq, 4, &mut out) };
            }
            z4.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                unsafe { q4k8_gemm_zmm(&t, n_sb, &kq, 64, &mut out) };
            }
            z64.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                unsafe { q4k8_gemv_zmm(&t, n_sb, &batch[..1], 1, &mut out) };
            }
            z1.push(t0.elapsed().as_secs_f64());
        }
    }
    let per = |secs: f64, rows: usize| secs * 1e9 / (reps as f64 * rows as f64 * n_sb as f64);
    for (name, v, rows) in [
        ("gemm(4)", &mut g4, 4usize),
        ("gemm(64)", &mut g64, 64),
        ("gemv(1)", &mut v1, 1),
    ] {
        let ns = per(best(v), rows);
        // One (tile, activation row, super-block) is 8 rows x 256 weights = 2048 MACs.
        println!(
            "{name}: {ns:.2} ns/(tile*act*sb)  {:.1} GMAC/s/core",
            2048.0 / ns
        );
    }
    // The widths the prefill study asks for (S5): the row kernel against the tile kernel on
    // the same eight rows, at 1, 16, 64 and 128 activation rows. The tile `t` above is the
    // repacked store, so the row kernel needs the file's own bytes; they are rebuilt here.
    {
        let rb = n_sb * 144;
        let mut rows = vec![0u8; 8 * rb];
        for r in 0..8 {
            for sb in 0..n_sb {
                let at = r * rb + sb * 144;
                let b = &mut rows[at..at + 144];
                b.copy_from_slice(&prng(0xA5A5_1234 ^ (r * 97 + sb) as u64, 144));
                b[0..2].copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes());
                b[2..4].copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.005).to_le_bytes());
            }
        }
        let mut tile = vec![0u8; n_sb * Q4K8::SIZE];
        for sb in 0..n_sb {
            let mut blocks: [&[u8]; 8] = [&[]; 8];
            for (r, b) in blocks.iter_mut().enumerate() {
                *b = &rows[r * rb + sb * 144..r * rb + sb * 144 + 144];
            }
            repack_q4k8(&blocks, &mut tile[sb * Q4K8::SIZE..(sb + 1) * Q4K8::SIZE]);
        }
        let mut o = vec![0.0f32; 8 * 128];
        for &nb in &[1usize, 4, 16, 64, 128] {
            let mut row = Vec::new();
            let mut tl = Vec::new();
            let mut tz = Vec::new();
            for _ in 0..5 {
                let t0 = Instant::now();
                for _ in 0..reps {
                    for r in 0..8 {
                        let rw = &rows[r * rb..(r + 1) * rb];
                        unsafe {
                            peregrine_kernels::simd::dot_batch(
                                peregrine_kernels::format::Format::Q4_K,
                                rw,
                                &batch[..nb],
                                &mut o[r * nb..(r + 1) * nb],
                            )
                        };
                    }
                }
                row.push(t0.elapsed().as_secs_f64());
                let t0 = Instant::now();
                for _ in 0..reps {
                    unsafe {
                        if nb < 4 {
                            q4k8_gemv(&tile, n_sb, &batch[..nb], nb, &mut o)
                        } else {
                            q4k8_gemm(&tile, n_sb, &kq, nb, &mut o)
                        }
                    }
                }
                tl.push(t0.elapsed().as_secs_f64());
                if zmm {
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        unsafe {
                            if nb < 4 {
                                q4k8_gemv_zmm(&tile, n_sb, &batch[..nb], nb, &mut o)
                            } else {
                                q4k8_gemm_zmm(&tile, n_sb, &kq, nb, &mut o)
                            }
                        }
                    }
                    tz.push(t0.elapsed().as_secs_f64());
                }
            }
            let macs = 2048.0 * n_sb as f64 * nb as f64 * reps as f64;
            let g = |v: &mut Vec<f64>| macs / best(v) / 1e9;
            print!(
                "q4k {nb:>3} acts: row {:6.1}  tile(avx2) {:6.1}",
                g(&mut row),
                g(&mut tl)
            );
            if zmm {
                print!("  tile(zmm) {:6.1}", g(&mut tz));
            }
            println!("  GMAC/s/core");
        }
    }

    if zmm {
        for (name, v, rows) in [
            ("zmm gemm(4)", &mut z4, 4usize),
            ("zmm gemm(64)", &mut z64, 64),
            ("zmm gemv(1)", &mut z1, 1),
        ] {
            let ns = per(best(v), rows);
            println!(
                "{name}: {ns:.2} ns/(tile*act*sb)  {:.1} GMAC/s/core",
                2048.0 / ns
            );
        }
    }

    if std::env::var("KQBENCH_LAYER").is_ok() {
        layer_mode(n_sb, reps);
    }

    // Q5_0: a block is 32 weights per row, so one (tile, act, block) is 8 x 32 = 256 MACs.
    let n_blk = n / 32;
    let t5 = tile5(n_blk, 0x5EED_5A5A);
    unsafe { q5_08_gemm(&t5, n_blk, &batch[..4], 4, &mut out) };
    for &nb in &[1usize, 4, 16, 64, 128] {
        let mut q = Vec::new();
        let mut qz = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..reps {
                unsafe { q5_08_gemm(&t5, n_blk, &batch[..nb], nb, &mut out) };
            }
            q.push(t0.elapsed().as_secs_f64());
            if zmm {
                let t0 = Instant::now();
                for _ in 0..reps {
                    unsafe { q5_08_gemm_zmm(&t5, n_blk, &batch[..nb], nb, &mut out) };
                }
                qz.push(t0.elapsed().as_secs_f64());
            }
        }
        let ns = per(best(&mut q), nb) / 8.0; // per block: one block is 1/8 of a super-block
        if zmm {
            let zns = per(best(&mut qz), nb) / 8.0;
            println!(
                "q5 {nb:>3} acts: avx2 {:.1} G  zmm {:.1} G MAC/s/core  ({:+.0}%)",
                256.0 / ns,
                256.0 / zns,
                (ns / zns - 1.0) * 100.0
            );
        } else {
            println!("q5 {nb:>3} acts: avx2 {:.1} G MAC/s/core", 256.0 / ns);
        }
    }

    // Q6_K and Q5_K (S4.9i): the repacked tile against the row-at-a-time path,
    // at the widths that decide the GEMV/GEMM split. One (tile, act, sb) is
    // eight rows x 256 weights = 2048 MACs.
    for (fmt, ts, block, label) in [
        (
            peregrine_kernels::format::Format::Q6_K,
            Q6K8::SIZE,
            210usize,
            "q6k",
        ),
        (
            peregrine_kernels::format::Format::Q5_K,
            Q5K8::SIZE,
            176usize,
            "q5k",
        ),
    ] {
        let rb = n_sb * block;
        let mut rows = vec![0u8; 8 * rb];
        for r in 0..8 {
            for sb in 0..n_sb {
                let at = r * rb + sb * block;
                let b = &mut rows[at..at + block];
                b.copy_from_slice(&prng(0x9E37_79B9 ^ (r * 131 + sb) as u64, block));
                if fmt == peregrine_kernels::format::Format::Q5_K {
                    b[0..2]
                        .copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes());
                    b[2..4].copy_from_slice(
                        &peregrine_kernels::quant::f32_to_f16(0.005).to_le_bytes(),
                    );
                    for i in 0..12 {
                        b[4 + i] &= 0x3F;
                    }
                } else {
                    b[208..210]
                        .copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes());
                    for i in 0..16 {
                        b[192 + i] &= 0x3F;
                    }
                }
            }
        }
        let mut tile = vec![0u8; n_sb * ts];
        for sb in 0..n_sb {
            let mut blocks: [&[u8]; 8] = [&[]; 8];
            for (r, b) in blocks.iter_mut().enumerate() {
                *b = &rows[r * rb + sb * block..r * rb + sb * block + block];
            }
            let out = &mut tile[sb * ts..(sb + 1) * ts];
            if fmt == peregrine_kernels::format::Format::Q6_K {
                repack_q6k8(&blocks, out)
            } else {
                repack_q5k8(&blocks, out)
            }
        }
        for &nb in &[1usize, 4, 16, 64, 128] {
            let mut old = Vec::new();
            let mut new = Vec::new();
            let mut wide = Vec::new();
            let mut o = vec![0.0f32; 8 * nb];
            for _ in 0..5 {
                let t0 = Instant::now();
                for _ in 0..reps {
                    for r in 0..8 {
                        let row = &rows[r * rb..(r + 1) * rb];
                        unsafe {
                            peregrine_kernels::simd::dot_batch(
                                fmt,
                                row,
                                &batch[..nb],
                                &mut o[r * nb..(r + 1) * nb],
                            )
                        };
                    }
                }
                old.push(t0.elapsed().as_secs_f64());
                let t0 = Instant::now();
                for _ in 0..reps {
                    unsafe {
                        if nb < 4 {
                            if fmt == peregrine_kernels::format::Format::Q6_K {
                                q6k8_gemv(&tile, n_sb, &batch[..nb], nb, &mut o)
                            } else {
                                q5k8_gemv(&tile, n_sb, &batch[..nb], nb, &mut o)
                            }
                        } else if fmt == peregrine_kernels::format::Format::Q6_K {
                            q6k8_gemm(&tile, n_sb, &kq, nb, &mut o)
                        } else {
                            q5k8_gemm(&tile, n_sb, &kq, nb, &mut o)
                        }
                    }
                }
                new.push(t0.elapsed().as_secs_f64());
                if zmm && fmt == peregrine_kernels::format::Format::Q6_K {
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        unsafe { q6k8_gemm_zmm(&tile, n_sb, &kq, nb, &mut o) };
                    }
                    wide.push(t0.elapsed().as_secs_f64());
                }
            }
            let macs = 8.0 * 256.0 * n_sb as f64 * nb as f64 * reps as f64;
            let old_g = macs / best(&mut old) / 1e9;
            let new_g = macs / best(&mut new) / 1e9;
            if wide.is_empty() {
                println!(
                    "{label} {nb:>3} acts: row {old_g:6.1} G  tile {new_g:6.1} G MAC/s/core  ({:+.0}%)",
                    (new_g / old_g - 1.0) * 100.0
                );
            } else {
                let wide_g = macs / best(&mut wide) / 1e9;
                println!(
                    "{label} {nb:>3} acts: row {old_g:6.1} G  tile {new_g:6.1} G  zmm {wide_g:6.1} G MAC/s/core  (zmm {:+.0}%)",
                    (wide_g / new_g - 1.0) * 100.0
                );
            }
        }
    }
}

/// One real layer's worth of tiles, streamed: `n_tiles` eight-row tiles is the 8192 output
/// rows of a gate/up matrix, and the activation rows sweep 8, 64 and 512 — the last is the
/// fork's ubatch, which Peregrine's engine never asks for (its chunk stops at 128) but
/// which is the width the comparison is about.
///
/// This exists for the S5.1 reconciliation (docs/design/s5-prefill-avx512.md §10): the
/// fork's kernel probe runs an 8192-row matrix, so a single L1 tile would not be the same
/// experiment. One call here is `n_tiles * 8 rows * (n_sb*256) weights * nb activations`
/// multiply-accumulates, the same count the fork's probe uses.
#[cfg(target_arch = "x86_64")]
fn layer_mode(n_sb: usize, reps: usize) {
    let n_tiles = 1024; // 8192 output rows / 8
    let n = n_sb * 256;
    let batch = acts(512, n, 0xC0DE_5678);
    let mut kq = KqActs::default();
    kq.fill(&batch);

    // Q4_K: 8 rows x n_sb super-blocks per tile, the same bytes a file holds.
    let mut tiles = vec![0u8; n_tiles * n_sb * Q4K8::SIZE];
    for t in 0..n_tiles {
        for sb in 0..n_sb {
            let mut blocks: [Vec<u8>; 8] = Default::default();
            for (r, b) in blocks.iter_mut().enumerate() {
                *b = prng(0x5EED_0000 + (t * 8 + r) as u64 * 31 + sb as u64, Q4K_BLOCK);
                b[0..2].copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes());
                b[2..4].copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.005).to_le_bytes());
            }
            let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
            let at = (t * n_sb + sb) * Q4K8::SIZE;
            repack_q4k8(&refs, &mut tiles[at..at + Q4K8::SIZE]);
        }
    }
    let mut tiles6 = vec![0u8; n_tiles * n_sb * Q6K8::SIZE];
    for t in 0..n_tiles {
        for sb in 0..n_sb {
            let mut blocks: [Vec<u8>; 8] = Default::default();
            for (r, b) in blocks.iter_mut().enumerate() {
                *b = prng(0x6EED_0000 + (t * 8 + r) as u64 * 37 + sb as u64, Q6K_BLOCK);
                b[208..210]
                    .copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes());
            }
            let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
            let at = (t * n_sb + sb) * Q6K8::SIZE;
            repack_q6k8(&refs, &mut tiles6[at..at + Q6K8::SIZE]);
        }
    }

    // Q5_0: a block is 32 weights per row, so a tile is 8 rows x n_blk blocks.
    let n_blk = n / 32;
    let mut tiles5 = vec![0u8; n_tiles * n_blk * Q5_08::SIZE];
    for t in 0..n_tiles {
        for blk in 0..n_blk {
            let mut blocks: [Vec<u8>; 8] = Default::default();
            for (r, b) in blocks.iter_mut().enumerate() {
                *b = prng(0x50ED_0000 + (t * 8 + r) as u64 * 41 + blk as u64, 22);
                b[0..2].copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes());
            }
            let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
            let at = (t * n_blk + blk) * Q5_08::SIZE;
            repack_q5_08(&refs, &mut tiles5[at..at + Q5_08::SIZE]);
        }
    }

    let zmm = peregrine_kernels::kqsimd::have_zmm512();
    for &nb in &[8usize, 64, 512] {
        let mut o = vec![0.0f32; nb * 8];
        let macs = (n_tiles * 8 * n * nb * reps) as f64;
        let mut q4 = Vec::new();
        let mut q6 = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let at = t * n_sb * Q4K8::SIZE;
                    unsafe {
                        if zmm {
                            q4k8_gemm_zmm(&tiles[at..at + n_sb * Q4K8::SIZE], n_sb, &kq, nb, &mut o)
                        } else {
                            q4k8_gemm(&tiles[at..at + n_sb * Q4K8::SIZE], n_sb, &kq, nb, &mut o)
                        }
                    }
                }
            }
            q4.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let at = t * n_sb * Q6K8::SIZE;
                    unsafe {
                        if zmm {
                            q6k8_gemm_zmm(
                                &tiles6[at..at + n_sb * Q6K8::SIZE],
                                n_sb,
                                &kq,
                                nb,
                                &mut o,
                            )
                        } else {
                            q6k8_gemm(&tiles6[at..at + n_sb * Q6K8::SIZE], n_sb, &kq, nb, &mut o)
                        }
                    }
                }
            }
            q6.push(t0.elapsed().as_secs_f64());
        }
        let mut q5 = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let at = t * n_blk * Q5_08::SIZE;
                    unsafe {
                        if zmm {
                            q5_08_gemm_zmm(
                                &tiles5[at..at + n_blk * Q5_08::SIZE],
                                n_blk,
                                &batch[..nb],
                                nb,
                                &mut o,
                            )
                        } else {
                            q5_08_gemm(
                                &tiles5[at..at + n_blk * Q5_08::SIZE],
                                n_blk,
                                &batch[..nb],
                                nb,
                                &mut o,
                            )
                        }
                    }
                }
            }
            q5.push(t0.elapsed().as_secs_f64());
        }
        println!(
            "layer 8192 rows x {n} weights, {nb:>3} acts ({n_tiles} tiles): q4k {:6.1}  q6k {:6.1}  q5_0 {:6.1}  GMAC/s/core ({})",
            macs / best(&mut q4) / 1e9,
            macs / best(&mut q6) / 1e9,
            macs / best(&mut q5) / 1e9,
            if zmm { "zmm" } else { "avx2" }
        );
    }

    if zmm && std::env::var("KQBENCH_WIDE").is_ok() {
        wide_layer(n_sb, n_tiles, &kq, &tiles);
    }
}

/// The widened-codes wide pass (S5.2): the same tile, its nibbles copied into a
/// byte buffer once, then `vpdpbusd` over the whole pass — against today's
/// kernel at the widths a wide pass would run.
///
/// ```text
/// KQBENCH_LAYER=1 KQBENCH_WIDE=1 taskset -c 4 kqbench 8 40 1
/// ```
///
/// Each width is checked bit for bit against `q4k8_gemm_zmm` before it is timed,
/// because that is the whole claim: the widening may not change one bit.
#[cfg(target_arch = "x86_64")]
fn wide_layer(n_sb: usize, n_tiles: usize, kq: &KqActs, tiles: &[u8]) {
    use peregrine_kernels::kqsimd::Q4K8_WIDE;
    let mut scratch = vec![0u8; n_sb * Q4K8_WIDE];
    let n = n_sb * 256;
    let macs = (n_tiles * 8 * n) as f64;
    println!("--- wide pass: nibbles widened to bytes once per tile, then 16/32/64 row groups ---");
    for &nb in &[8usize, 64, 128, 256, 512] {
        let reps = (320 / nb).max(3);
        let mut a = vec![0.0f32; nb * 8];
        let mut b = vec![0.0f32; nb * 8];
        for t in 0..n_tiles {
            let at = t * n_sb * Q4K8::SIZE;
            let tl = &tiles[at..at + n_sb * Q4K8::SIZE];
            unsafe {
                q4k8_gemm_zmm(tl, n_sb, kq, nb, &mut a);
                q4k8_gemm_zmm_wide(tl, n_sb, kq, nb, &mut b, &mut scratch);
            }
            assert_eq!(
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "the wide pass changed a bit at {nb} activation rows"
            );
        }
        let mut wide = Vec::new();
        let mut narrow = Vec::new();
        let mut widen = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let at = t * n_sb * Q4K8::SIZE;
                    unsafe {
                        q4k8_gemm_zmm(&tiles[at..at + n_sb * Q4K8::SIZE], n_sb, kq, nb, &mut a)
                    }
                }
            }
            narrow.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let at = t * n_sb * Q4K8::SIZE;
                    unsafe {
                        q4k8_gemm_zmm_wide(
                            &tiles[at..at + n_sb * Q4K8::SIZE],
                            n_sb,
                            kq,
                            nb,
                            &mut b,
                            &mut scratch,
                        )
                    }
                }
            }
            wide.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let at = t * n_sb * Q4K8::SIZE;
                    unsafe {
                        peregrine_kernels::kqsimd::q4k8_widen(
                            &tiles[at..at + n_sb * Q4K8::SIZE],
                            n_sb,
                            &mut scratch,
                        )
                    }
                }
            }
            widen.push(t0.elapsed().as_secs_f64());
        }
        let g = |v: &mut Vec<f64>| macs * nb as f64 * reps as f64 / best(v) / 1e9;
        println!(
            "q4k wide {nb:>3} acts: today {:6.1}  widened {:6.1}  ({:+.1}%)  widening alone {:.0} GMAC/s  GMAC/s/core",
            g(&mut narrow),
            g(&mut wide),
            (g(&mut wide) / g(&mut narrow) - 1.0) * 100.0,
            // The widening moves no multiply-accumulate, so its rate is the
            // weight bytes it copies per second instead.
            (n_tiles * 8 * n * reps) as f64 / best(&mut widen) / 1e9,
        );
    }
}
