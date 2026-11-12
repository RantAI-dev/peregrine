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
        if std::env::var("KQBENCH_PROMPT").is_ok() {
            prompt_kernel_table(n_sb, threads);
        }
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

    if std::env::var("KQBENCH_PROMPT").is_ok() {
        prompt_mode(n_sb, threads);
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

/// The 8-bit prompt form against the exact kernels (S5.4a): one real layer's worth of
/// tiles (2048 × 8192, 1024 eight-row tiles, 4 MB streamed — and the same tile over
/// and over, L1-resident), at the widths a prompt pass runs at.
///
/// Four numbers per (format, width):
///
///  * `exact`: today's kernel over the file's own tiles (`q4k8_gemm_zmm`,
///    `q6k8_gemm_zmm`, `q5_08_gemm_zmm`);
///  * `resident`: `q8k_gemm_zmm` over a form built once, outside the timing;
///  * `per-pass`: the same form built inside the pass, once per tile, timed with the
///    multiply;
///  * `build`: the conversion alone, reported as weights per second.
///
/// ```text
/// KQBENCH_PROMPT=1 taskset -c 20-23 cargo run --release -p peregrine-bench \
///     --example kqbench -- 8 40 1
/// ```
///
/// Before each width is timed, the resident form is rebuilt inside the pass and
/// compared with the one built outside: the two paths must agree, or the table
/// would be comparing a different arithmetic.
#[cfg(target_arch = "x86_64")]
fn prompt_mode(n_sb: usize, threads: usize) {
    use peregrine_kernels::kq8::{build_q4k, build_q5_0, build_q6k, form_size, Q8K_SIZE};
    use peregrine_kernels::kqsimd::{
        q4k8_gemm_zmm, q5_08_gemm_zmm, q6k8_gemm_zmm, q8k_build_q4k_avx2, q8k_build_q4k_zmm,
        q8k_build_q5_0_zmm, q8k_build_q6k_avx2, q8k_build_q6k_zmm, q8k_gemm_avx2, q8k_gemm_zmm,
    };
    let n_tiles = 1024; // 8192 output rows
    let n = n_sb * 256;
    let batch = acts(512, n, 0xC0DE_5678);
    let mut kq = KqActs::default();
    kq.fill(&batch);
    let zmm = peregrine_kernels::kqsimd::have_zmm512();
    let avx2 = peregrine_kernels::kqsimd::have_avx2();
    if !avx2 {
        eprintln!("the 8-bit form has no SIMD body on this machine; nothing to measure here");
        return;
    }
    // One name for the 8-bit body under test, so every row below measures the same
    // tier's kernel (S5.4d added the AVX2 one).
    macro_rules! q8k_build {
        ($kind:ident, $tile:expr, $n_sb:expr, $form:expr) => {
            unsafe {
                if zmm {
                    match stringify!($kind) {
                        "q4k" => q8k_build_q4k_zmm($tile, $n_sb, $form),
                        "q6k" => q8k_build_q6k_zmm($tile, $n_sb, $form),
                        _ => q8k_build_q5_0_zmm($tile, $n_sb, $form),
                    }
                } else {
                    match stringify!($kind) {
                        "q4k" => q8k_build_q4k_avx2($tile, $n_sb, $form),
                        "q6k" => q8k_build_q6k_avx2($tile, $n_sb, $form),
                        // S5.4f gives Q5_0 the AVX2 builder too.
                        _ => {
                            peregrine_kernels::kqsimd::q8k_build_q5_0_avx2($tile, $n_sb * 8, $form)
                        }
                    }
                }
            }
        };
    }
    macro_rules! q8k_gemm {
        ($form:expr, $n_sb:expr, $kq:expr, $nb:expr, $out:expr) => {
            unsafe {
                if zmm {
                    q8k_gemm_zmm($form, $n_sb, $kq, $nb, $out)
                } else {
                    q8k_gemm_avx2($form, $n_sb, $kq, $nb, $out)
                }
            }
        };
    }
    let n_blk5 = n / 32;

    // The exact tiles (the engine's own store) and the forms built from them.
    let mut tiles4 = vec![0u8; n_tiles * n_sb * Q4K8::SIZE];
    let mut tiles6 = vec![0u8; n_tiles * n_sb * Q6K8::SIZE];
    let mut tiles5 = vec![0u8; n_tiles * n_blk5 * Q5_08::SIZE];
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
            repack_q4k8(&refs, &mut tiles4[at..at + Q4K8::SIZE]);
        }
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
        for blk in 0..n_blk5 {
            let mut blocks: [Vec<u8>; 8] = Default::default();
            for (r, b) in blocks.iter_mut().enumerate() {
                *b = prng(0x50ED_0000 + (t * 8 + r) as u64 * 41 + blk as u64, 22);
                b[0..2].copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes());
            }
            let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
            let at = (t * n_blk5 + blk) * Q5_08::SIZE;
            repack_q5_08(&refs, &mut tiles5[at..at + Q5_08::SIZE]);
        }
    }
    // The resident forms, built once.
    let stride4 = n_sb * Q8K_SIZE;
    let stride6 = n_sb * Q8K_SIZE;
    let stride5 = n_sb * Q8K_SIZE;
    let mut forms4 = vec![0u8; n_tiles * form_size(n_sb)];
    let mut forms6 = vec![0u8; n_tiles * form_size(n_sb)];
    let mut forms5 = vec![0u8; n_tiles * form_size(n_sb)];
    let mut scratch = vec![0u8; form_size(n_sb)];
    for t in 0..n_tiles {
        let a = t * n_sb * Q4K8::SIZE;
        q8k_build!(
            q4k,
            &tiles4[a..a + n_sb * Q4K8::SIZE],
            n_sb,
            &mut forms4[t * stride4..(t + 1) * stride4]
        );
        let a = t * n_sb * Q6K8::SIZE;
        q8k_build!(
            q6k,
            &tiles6[a..a + n_sb * Q6K8::SIZE],
            n_sb,
            &mut forms6[t * stride6..(t + 1) * stride6]
        );
        let a = t * n_blk5 * Q5_08::SIZE;
        q8k_build!(
            q5_0,
            &tiles5[a..a + n_blk5 * Q5_08::SIZE],
            n_sb,
            &mut forms5[t * stride5..(t + 1) * stride5]
        );
    }
    // The forms are the twin's, built from the same tiles (this also checks that the
    // resident store is what the per-pass path will produce).
    for (i, tile) in [0usize, 7, 999].iter().enumerate() {
        let mut want = vec![0u8; form_size(n_sb)];
        let a = tile * n_sb * Q4K8::SIZE;
        build_q4k(&tiles4[a..a + n_sb * Q4K8::SIZE], n_sb, &mut want);
        assert_eq!(
            want,
            &forms4[tile * stride4..(tile + 1) * stride4],
            "the Q4_K form {i} is not the twin's"
        );
        let a = tile * n_sb * Q6K8::SIZE;
        build_q6k(&tiles6[a..a + n_sb * Q6K8::SIZE], n_sb, &mut want);
        assert_eq!(
            want,
            &forms6[tile * stride6..(tile + 1) * stride6],
            "the Q6_K form {i} is not the twin's"
        );
        let a = tile * n_blk5 * Q5_08::SIZE;
        build_q5_0(&tiles5[a..a + n_blk5 * Q5_08::SIZE], n_blk5, &mut want);
        assert_eq!(
            want,
            &forms5[tile * stride5..(tile + 1) * stride5],
            "the Q5_0 form {i} is not the twin's"
        );
    }
    println!(
        "--- the 8-bit prompt form (S5.4a): {n} weights x {} output rows, one thread; the columns are \
         'exact' (the file's tiles), 'resident' (the form built once), 'per-pass' (the form built per tile \
         inside the pass) and 'build' (the conversion alone) ---",
        8 * n_tiles
    );
    println!(
        "--- the form's bytes per weight: {:.5} (a Q4_K file holds {:.4}, a Q6_K file {:.4}) ---",
        Q8K_SIZE as f64 / 2048.0,
        Q4K_BLOCK as f64 / 256.0,
        Q6K_BLOCK as f64 / 256.0
    );
    for &nb in &[1usize, 8, 16, 64, 128, 512] {
        let reps = (200 / nb).max(1);
        let mut o = vec![0.0f32; nb * 8];
        let macs = (n_tiles * 8 * n * nb * reps) as f64;
        let mut rows = Vec::new();
        // Q4_K.
        let mut exact = Vec::new();
        let mut resident = Vec::new();
        let mut per_pass = Vec::new();
        let mut build_only = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * n_sb * Q4K8::SIZE;
                    unsafe {
                        if zmm {
                            q4k8_gemm_zmm(&tiles4[a..a + n_sb * Q4K8::SIZE], n_sb, &kq, nb, &mut o)
                        } else {
                            q4k8_gemm(&tiles4[a..a + n_sb * Q4K8::SIZE], n_sb, &kq, nb, &mut o)
                        }
                    };
                }
            }
            exact.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * stride4;
                    q8k_gemm!(&forms4[a..a + stride4], n_sb, &kq, nb, &mut o);
                }
            }
            resident.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * n_sb * Q4K8::SIZE;
                    q8k_build!(q4k, &tiles4[a..a + n_sb * Q4K8::SIZE], n_sb, &mut scratch);
                    q8k_gemm!(&scratch, n_sb, &kq, nb, &mut o);
                }
            }
            per_pass.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * n_sb * Q4K8::SIZE;
                    q8k_build!(q4k, &tiles4[a..a + n_sb * Q4K8::SIZE], n_sb, &mut scratch);
                }
            }
            build_only.push(t0.elapsed().as_secs_f64());
        }
        let g = |v: &mut Vec<f64>| macs / best(v) / 1e9;
        let w = |v: &mut Vec<f64>| (n_tiles * 8 * n * reps) as f64 / best(v) / 1e9;
        rows.push((
            "q4k",
            g(&mut exact),
            g(&mut resident),
            g(&mut per_pass),
            w(&mut build_only),
        ));
        // Q6_K.
        let mut exact = Vec::new();
        let mut resident = Vec::new();
        let mut per_pass = Vec::new();
        let mut build_only = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * n_sb * Q6K8::SIZE;
                    unsafe {
                        if zmm {
                            q6k8_gemm_zmm(&tiles6[a..a + n_sb * Q6K8::SIZE], n_sb, &kq, nb, &mut o)
                        } else {
                            q6k8_gemm(&tiles6[a..a + n_sb * Q6K8::SIZE], n_sb, &kq, nb, &mut o)
                        }
                    };
                }
            }
            exact.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * stride6;
                    q8k_gemm!(&forms6[a..a + stride6], n_sb, &kq, nb, &mut o);
                }
            }
            resident.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * n_sb * Q6K8::SIZE;
                    q8k_build!(q6k, &tiles6[a..a + n_sb * Q6K8::SIZE], n_sb, &mut scratch);
                    q8k_gemm!(&scratch, n_sb, &kq, nb, &mut o);
                }
            }
            per_pass.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * n_sb * Q6K8::SIZE;
                    q8k_build!(q6k, &tiles6[a..a + n_sb * Q6K8::SIZE], n_sb, &mut scratch);
                }
            }
            build_only.push(t0.elapsed().as_secs_f64());
        }
        rows.push((
            "q6k",
            g(&mut exact),
            g(&mut resident),
            g(&mut per_pass),
            w(&mut build_only),
        ));
        // Q5_0: the same form, built from the Q5_0 tiles.
        let mut exact = Vec::new();
        let mut resident = Vec::new();
        let mut per_pass = Vec::new();
        let mut build_only = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * n_blk5 * Q5_08::SIZE;
                    unsafe {
                        if zmm {
                            q5_08_gemm_zmm(
                                &tiles5[a..a + n_blk5 * Q5_08::SIZE],
                                n_blk5,
                                &batch[..nb],
                                nb,
                                &mut o,
                            )
                        } else {
                            q5_08_gemm(
                                &tiles5[a..a + n_blk5 * Q5_08::SIZE],
                                n_blk5,
                                &batch[..nb],
                                nb,
                                &mut o,
                            )
                        }
                    };
                }
            }
            exact.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * stride5;
                    q8k_gemm!(&forms5[a..a + stride5], n_sb, &kq, nb, &mut o);
                }
            }
            resident.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * n_blk5 * Q5_08::SIZE;
                    q8k_build!(
                        q5_0,
                        &tiles5[a..a + n_blk5 * Q5_08::SIZE],
                        n_sb,
                        &mut scratch
                    );
                    q8k_gemm!(&scratch, n_sb, &kq, nb, &mut o);
                }
            }
            per_pass.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..n_tiles {
                    let a = t * n_blk5 * Q5_08::SIZE;
                    q8k_build!(
                        q5_0,
                        &tiles5[a..a + n_blk5 * Q5_08::SIZE],
                        n_sb,
                        &mut scratch
                    );
                }
            }
            build_only.push(t0.elapsed().as_secs_f64());
        }
        rows.push((
            "q5_0",
            g(&mut exact),
            g(&mut resident),
            g(&mut per_pass),
            w(&mut build_only),
        ));
        for (label, exact, resident, per_pass, build) in rows {
            println!(
                "{label:>4} {nb:>3} acts: exact {exact:6.1}  resident {resident:6.1} ({:+.0}%)  \
                 per-pass {per_pass:6.1} ({:+.0}%)  build {build:5.1} G weights/s  GMAC/s/core",
                (resident / exact - 1.0) * 100.0,
                (per_pass / exact - 1.0) * 100.0,
            );
        }
    }
    prompt_kernel_table(n_sb, threads.max(1));
}

/// The kernel table S5.4d asks for: the exact wide kernel, the 8-bit form's scalar
/// twin and the tier's 8-bit SIMD body, in GMAC/s, per format, at one and at
/// `threads` cores. One "row" of work is 8 output rows of a tile over `n_sb`
/// super-blocks against 128 activation rows — a prompt chunk.
///
/// Each thread holds its own copy of the tile (the engine splits *output* rows
/// across workers, and a shared L1 tile would measure cache sharing rather than the
/// kernel), so the multi-thread column is the aggregate rate of `threads`
/// independent copies.
#[cfg(target_arch = "x86_64")]
fn prompt_kernel_table(n_sb: usize, threads: usize) {
    use peregrine_kernels::kq8::Q8K_SIZE;
    let n = n_sb * 256;
    let reps = 40usize;
    let zmm = peregrine_kernels::kqsimd::have_zmm512();
    let n_blk5 = n_sb * 8;
    println!(
        "--- kernel table: {n} weights x 8 output rows, at {{8, 64, 128, 512}} activation rows; \
         GMAC/s aggregate over {{1, {threads}}} cores (S5.4f's table a) ---"
    );
    for fmt in ["q4k", "q6k", "q5_0"] {
        // The exact tile, as the engine's store lays it out.
        let tile_bytes = match fmt {
            "q4k" => Q4K8::SIZE,
            "q6k" => Q6K8::SIZE,
            _ => Q5_08::SIZE,
        };
        let blocks_per_tile = if fmt == "q5_0" { n_blk5 } else { n_sb };
        let mut tile = vec![0u8; blocks_per_tile * tile_bytes];
        for sb in 0..blocks_per_tile {
            let block = match fmt {
                "q4k" => Q4K_BLOCK,
                "q6k" => Q6K_BLOCK,
                _ => 22,
            };
            let mut blocks: [Vec<u8>; 8] = Default::default();
            for (r, b) in blocks.iter_mut().enumerate() {
                let mut v = prng(0x5EED_0000 + (r * 31 + sb) as u64, block);
                match fmt {
                    "q4k" => {
                        v[0..2].copy_from_slice(
                            &peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes(),
                        );
                        v[2..4].copy_from_slice(
                            &peregrine_kernels::quant::f32_to_f16(0.005).to_le_bytes(),
                        );
                    }
                    "q6k" => v[208..210]
                        .copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.01).to_le_bytes()),
                    _ => v[0..2]
                        .copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.02).to_le_bytes()),
                }
                *b = v;
            }
            let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
            let out = &mut tile[sb * tile_bytes..(sb + 1) * tile_bytes];
            match fmt {
                "q4k" => repack_q4k8(&refs, out),
                "q6k" => repack_q6k8(&refs, out),
                _ => repack_q5_08(&refs, out),
            }
        }
        // The 8-bit form of the same tile.
        let mut form = vec![0u8; n_sb * Q8K_SIZE];
        unsafe {
            match fmt {
                "q4k" if zmm => {
                    peregrine_kernels::kqsimd::q8k_build_q4k_zmm(&tile, n_sb, &mut form)
                }
                "q4k" => peregrine_kernels::kqsimd::q8k_build_q4k_avx2(&tile, n_sb, &mut form),
                "q6k" if zmm => {
                    peregrine_kernels::kqsimd::q8k_build_q6k_zmm(&tile, n_sb, &mut form)
                }
                "q6k" => peregrine_kernels::kqsimd::q8k_build_q6k_avx2(&tile, n_sb, &mut form),
                _ if zmm => peregrine_kernels::kqsimd::q8k_build_q5_0_zmm(&tile, n_blk5, &mut form),
                _ => peregrine_kernels::kqsimd::q8k_build_q5_0_avx2(&tile, n_blk5, &mut form),
            }
        }
        for &nb in &[8usize, 64, 128, 512] {
            let batch = acts(nb, n, 0xC0DE_5678);
            let mut kq = KqActs::default();
            kq.fill(&batch);
            let macs = 2048.0 * n_sb as f64 * nb as f64;
            for (body, ncores) in [
                ("exact", 1usize),
                ("form", 1),
                ("exact", threads),
                ("form", threads),
            ] {
                let hs: Vec<_> = (0..ncores)
                    .map(|_| {
                        let tile = tile.clone();
                        let form = form.clone();
                        let kq = kq.clone();
                        let batch = batch.clone();
                        std::thread::spawn(move || {
                            let mut o = vec![0.0f32; nb * 8];
                            // The warm-up run faults the pages and lifts the clock; the
                            // best of four rounds is what the table reports.
                            let mut best = f64::MAX;
                            for _round in 0..4 {
                                let t0 = Instant::now();
                                for _ in 0..reps {
                                    unsafe {
                                        match (body, fmt) {
                                            ("exact", "q4k") => {
                                                if zmm {
                                                    q4k8_gemm_zmm(&tile, n_sb, &kq, nb, &mut o)
                                                } else {
                                                    q4k8_gemm(&tile, n_sb, &kq, nb, &mut o)
                                                }
                                            }
                                            ("exact", "q6k") => {
                                                if zmm {
                                                    q6k8_gemm_zmm(&tile, n_sb, &kq, nb, &mut o)
                                                } else {
                                                    q6k8_gemm(&tile, n_sb, &kq, nb, &mut o)
                                                }
                                            }
                                            ("exact", _) => {
                                                if zmm {
                                                    q5_08_gemm_zmm(
                                                        &tile,
                                                        n_blk5,
                                                        &batch[..nb],
                                                        nb,
                                                        &mut o,
                                                    )
                                                } else {
                                                    q5_08_gemm(
                                                        &tile,
                                                        n_blk5,
                                                        &batch[..nb],
                                                        nb,
                                                        &mut o,
                                                    )
                                                }
                                            }
                                            ("form", _) => {
                                                if zmm {
                                                    peregrine_kernels::kqsimd::q8k_gemm_zmm(
                                                        &form, n_sb, &kq, nb, &mut o,
                                                    )
                                                } else {
                                                    peregrine_kernels::kqsimd::q8k_gemm_avx2(
                                                        &form, n_sb, &kq, nb, &mut o,
                                                    )
                                                }
                                            }
                                            _ => unreachable!(),
                                        }
                                    }
                                }
                                best = best.min(t0.elapsed().as_secs_f64());
                            }
                            best
                        })
                    })
                    .collect();
                let wall = hs
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .fold(0.0f64, f64::max);
                println!(
                    "{fmt:>4} {nb:>3} acts {body:>5} x{ncores}: {:6.1} GMAC/s",
                    macs * reps as f64 * ncores as f64 / wall / 1e9
                );
            }
        }
    }
    if std::env::var("KQBENCH_Q5_SHAPE").is_ok() {
        q5_row_shape(threads);
    }
}

/// Q5_0 at the shape Qwen 2.5 0.5B's Q4_K_M file has: 896 weights = 28 blocks =
/// three and a half 256-weight super-blocks. The form needs a padded fourth
/// super-block, so this is the cell that decides whether Q5_0 joins the AVX2
/// tier's table (S5.4f; it does not: 0.88× the exact kernel).
///
/// The activation is 1024 wide, so the exact kernel reads its row's first 896
/// codes and the form reads all four super-blocks. GMAC/s counts the useful 896.
#[cfg(target_arch = "x86_64")]
fn q5_row_shape(threads: usize) {
    use peregrine_kernels::kq8::{build_q5_0, Q8K_SIZE};
    let n_blk = 28usize;
    let n_sb = n_blk.div_ceil(8);
    let mut tile = vec![0u8; n_blk * Q5_08::SIZE];
    for b in 0..n_blk {
        let mut rows: Vec<Vec<u8>> = (0..8)
            .map(|r| {
                let mut v = prng(0x5EED_0000 + (r * 31 + b) as u64, 22);
                v[0..2].copy_from_slice(&peregrine_kernels::quant::f32_to_f16(0.02).to_le_bytes());
                v
            })
            .collect();
        let refs: Vec<&[u8]> = rows.iter_mut().map(|v| v.as_slice()).collect();
        repack_q5_08(&refs, &mut tile[b * Q5_08::SIZE..(b + 1) * Q5_08::SIZE]);
    }
    let mut form = vec![0u8; n_sb * Q8K_SIZE];
    build_q5_0(&tile, n_blk, &mut form);
    let n = n_sb * 256;
    let batch: Vec<QuantAct> = (0..512)
        .map(|i| {
            let mut a = QuantAct::with_capacity(n);
            let x: Vec<f32> = prng(0xC0DE_0000 + i as u64, n)
                .iter()
                .map(|&b| b as f32 / 64.0 - 2.0)
                .collect();
            a.fill(&x);
            a
        })
        .collect();
    let mut kq = KqActs::default();
    kq.fill(&batch);
    println!(
        "--- Q5_0 at Qwen 2.5's 896-wide row (28 blocks, padded to {n_sb} super-blocks); \
         GMAC/s of useful weights, aggregate over {{1, {threads}}} cores ---"
    );
    for &nb in &[8usize, 64, 128, 512] {
        for (body, ncores) in [
            ("exact", 1usize),
            ("form", 1),
            ("exact", threads),
            ("form", threads),
        ] {
            let mut best = f64::MAX;
            for _ in 0..4 {
                let hs: Vec<_> = (0..ncores)
                    .map(|_| {
                        let tile = tile.clone();
                        let form = form.clone();
                        let kq = kq.clone();
                        let batch = batch.clone();
                        std::thread::spawn(move || {
                            let mut o = vec![0.0f32; nb * 8];
                            let t0 = Instant::now();
                            for _ in 0..40 {
                                unsafe {
                                    if body == "exact" {
                                        q5_08_gemm(&tile, n_blk, &batch[..nb], nb, &mut o)
                                    } else {
                                        peregrine_kernels::kqsimd::q8k_gemm_avx2(
                                            &form, n_sb, &kq, nb, &mut o,
                                        )
                                    }
                                }
                            }
                            t0.elapsed().as_secs_f64()
                        })
                    })
                    .collect();
                let wall = hs
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .fold(f64::MAX, f64::min);
                best = best.min(wall);
            }
            let macs = 896.0 * 8.0 * nb as f64 * 40.0;
            println!(
                "q5_0 {nb:>3} acts {body:>5} x{ncores}: {:6.1} GMAC/s (useful)",
                macs * ncores as f64 / best / 1e9
            );
        }
    }
}
