//! aarch64 NEON kernel micro-benchmark (S2.8): the `sdot` kernels against the scalar references.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example neonbench -- [n] [rows] [reps]
//! ```
//!
//! One thread, synthetic weights and activations, L1/L2-resident. Reports GMAC/s per core for the
//! per-row Q8_0 and Q4_K kernels, the eight-row Q8_0 decode kernel, and the repacked Q4_K tile
//! GEMV and GEMM, each beside the scalar reference it replaces. The ratio is what `docs/design/
//! s2.8-neon.md` §7.3 asks for: it separates "the kernel is faster" from "the model got faster".

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("neonbench is an aarch64 benchmark");
}

#[cfg(target_arch = "aarch64")]
fn main() {
    use peregrine_kernels::format::Format;
    use peregrine_kernels::kq::{
        dot_q4k8_ref, dot_q5_08_ref, repack_q4k8, repack_q5_08, KqActs, Q4K8, Q5_08,
    };
    use peregrine_kernels::quant::{dot_scalar, QuantAct};
    use std::hint::black_box;
    use std::time::Instant;

    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let n = args.first().copied().unwrap_or(2048).max(256);
    let rows = args.get(1).copied().unwrap_or(256).max(8);
    let reps = args.get(2).copied().unwrap_or(200).max(1);
    if !peregrine_kernels::cpu::have_dotprod() {
        eprintln!("this CPU has no asimddp; nothing to measure");
        return;
    }
    let n = n / 256 * 256;
    let n_sb = n / 256;

    let mut s = 0x243f_6a88_85a3_08d3u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 24) as u8
    };
    let x: Vec<f32> = (0..n)
        .map(|_| (next() as f32 / 255.0) * 2.0 - 1.0)
        .collect();
    let mut act = QuantAct::with_capacity(n);
    act.fill(&x);

    // A Q8_0 row and eight of them.
    let q8_row: Vec<u8> = {
        let mut w = Vec::with_capacity(n / 32 * 34);
        for _ in 0..n / 32 {
            w.extend_from_slice(&0x2c00u16.to_le_bytes()); // ~0.0625
            for _ in 0..32 {
                w.push(next());
            }
        }
        w
    };
    let q8_rows: Vec<u8> = (0..8).flat_map(|_| q8_row.iter().copied()).collect();

    // A Q4_K row and a tile of eight.
    let mut q4_rows: Vec<Vec<u8>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut w = Vec::with_capacity(n_sb * 144);
        for _ in 0..n_sb {
            w.extend_from_slice(&0x2000u16.to_le_bytes()); // d
            w.extend_from_slice(&0x1800u16.to_le_bytes()); // dmin
            for _ in 0..12 {
                w.push(next());
            }
            for _ in 0..128 {
                w.push(next());
            }
        }
        q4_rows.push(w);
    }
    let mut tile = vec![0u8; n_sb * Q4K8::SIZE];
    for sb in 0..n_sb {
        let src: Vec<&[u8]> = q4_rows.iter().map(|r| &r[sb * 144..][..144]).collect();
        repack_q4k8(&src, &mut tile[sb * Q4K8::SIZE..]);
    }
    // Eight Q5_0 rows and their tile, then eight Q6_K rows and theirs.
    let q5_rows: Vec<Vec<u8>> = (0..8)
        .map(|_| {
            let mut w = Vec::with_capacity(n / 32 * 22);
            for _ in 0..n / 32 {
                w.extend_from_slice(&0x2c00u16.to_le_bytes());
                for _ in 0..20 {
                    w.push(next());
                }
            }
            w
        })
        .collect();
    let n_blk = n / 32;
    let mut q5_tile = vec![0u8; n_blk * Q5_08::SIZE];
    for t in 0..n_blk {
        let src: Vec<&[u8]> = q5_rows.iter().map(|r| &r[t * 22..][..22]).collect();
        repack_q5_08(&src, &mut q5_tile[t * Q5_08::SIZE..]);
    }
    let acts: Vec<QuantAct> = (0..16)
        .map(|i| {
            let y: Vec<f32> = (0..n)
                .map(|j| ((i * 977 + j) as f32 * 0.001).sin())
                .collect();
            let mut a = QuantAct::with_capacity(n);
            a.fill(&y);
            a
        })
        .collect();
    let mut kq = KqActs::default();
    kq.fill(&acts[..16]);

    // Each closure performs `rowdots` output-row dot products (one activation row each); the
    // GMAC/s figure is `rowdots · n / time`, so the shapes are comparable.
    let bench = |name: &str, rowdots: usize, f: &mut dyn FnMut()| {
        f();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let ns = t.elapsed().as_nanos() as f64 / reps as f64;
        let macs = (rowdots * n) as f64;
        println!("{name:<34} {:>10.1} ns  {:>8.1} GMAC/s", ns, macs / ns);
    };

    let mut out = vec![0.0f32; 1024];
    println!(
        "n={n} rows={rows} reps={reps} tier={}",
        peregrine_kernels::cpu::tier_name()
    );

    bench("Q8_0 scalar, 1 row", rows, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows {
            acc += dot_scalar(Format::Q8_0, &q8_row, &act);
        }
        black_box(acc);
    });
    bench("Q8_0 sdot, 1 row", rows, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows {
            acc += unsafe { peregrine_kernels::neon::dot_q8_0(&q8_row, &act) };
        }
        black_box(acc);
    });
    bench("Q8_0 sdot rows8 (decode)", (rows / 8) * 8, &mut || {
        for _ in 0..rows / 8 {
            unsafe { peregrine_kernels::neon::q8_0_rows8(&q8_rows, q8_row.len(), &act, &mut out) };
        }
        black_box(out[0]);
    });
    bench("Q8_0 sdot batch-4", (rows / 4) * 4, &mut || {
        for _ in 0..rows / 4 {
            unsafe {
                peregrine_kernels::neon::q8_0_batch(&q8_row, &acts[..4], &mut out);
            }
        }
        black_box(out[0]);
    });

    bench("Q4_K scalar, 1 row", rows, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows {
            acc += dot_scalar(Format::Q4_K, &q4_rows[0], &act);
        }
        black_box(acc);
    });
    bench("Q4_K sdot, 1 row", rows, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows {
            acc += unsafe { peregrine_kernels::neon::dot_q4_k(&q4_rows[0], &act) };
        }
        black_box(acc);
    });

    bench("Q4_K tile ref, GEMV", (rows / 8) * 8, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows / 8 {
            for r in 0..8 {
                acc += dot_q4k8_ref(&tile, n_sb, r, &act);
            }
        }
        black_box(acc);
    });
    bench("Q4_K tile sdot, GEMV", (rows / 8) * 8, &mut || {
        for _ in 0..rows / 8 {
            unsafe {
                peregrine_kernels::neon::q4k8_gemv(
                    &tile,
                    n_sb,
                    std::slice::from_ref(&act),
                    1,
                    &mut out,
                )
            };
        }
        black_box(out[0]);
    });
    bench("Q4_K tile sdot, GEMM 4", (rows / 8) * 32, &mut || {
        for _ in 0..rows / 8 {
            unsafe { peregrine_kernels::neon::q4k8_gemm(&tile, n_sb, &kq, 4, &mut out) };
        }
        black_box(out[0]);
    });
    bench("Q4_K tile sdot, GEMM 16", (rows / 16) * 128, &mut || {
        for _ in 0..rows / 16 {
            unsafe { peregrine_kernels::neon::q4k8_gemm(&tile, n_sb, &kq, 16, &mut out) };
        }
        black_box(out[0]);
    });

    bench("Q5_0 scalar, 1 row", rows, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows {
            acc += dot_scalar(Format::Q5_0, &q5_rows[0], &act);
        }
        black_box(acc);
    });
    bench("Q5_0 sdot, 1 row", rows, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows {
            acc += unsafe { peregrine_kernels::neon::dot_q5_0(&q5_rows[0], &act) };
        }
        black_box(acc);
    });
    bench("Q5_0 sdot rows8 (decode)", (rows / 8) * 8, &mut || {
        for _ in 0..rows / 8 {
            unsafe {
                peregrine_kernels::neon::q5_0_rows8(&q5_rows.concat(), n_blk * 22, &act, &mut out)
            };
        }
        black_box(out[0]);
    });
    bench("Q5_0 tile ref, GEMV", (rows / 8) * 8, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows / 8 {
            for r in 0..8 {
                acc += dot_q5_08_ref(&q5_tile, n_blk, r, &act);
            }
        }
        black_box(acc);
    });
    bench("Q5_0 tile sdot, GEMV", (rows / 8) * 8, &mut || {
        for _ in 0..rows / 8 {
            unsafe {
                peregrine_kernels::neon::q5_08_gemv(
                    &q5_tile,
                    n_blk,
                    std::slice::from_ref(&act),
                    1,
                    &mut out,
                )
            };
        }
        black_box(out[0]);
    });
    bench("Q5_0 tile sdot, GEMM 4", (rows / 8) * 32, &mut || {
        for _ in 0..rows / 8 {
            unsafe { peregrine_kernels::neon::q5_08_gemm(&q5_tile, n_blk, &acts, 4, &mut out) };
        }
        black_box(out[0]);
    });
}
