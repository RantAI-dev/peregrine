//! aarch64 NEON kernel micro-benchmark (S2.8): the `sdot` kernels against the scalar references.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example neonbench -- [n] [rows] [reps]
//! cargo run --release -p peregrine-bench --example neonbench -- table [n_sb] [reps] [threads] [whole|row896|both]
//! ```
//!
//! One thread, synthetic weights and activations, L1/L2-resident. Reports GMAC/s per core for the
//! per-row Q8_0 and Q4_K kernels, the eight-row Q8_0 decode kernel, and the repacked Q4_K tile
//! GEMV and GEMM, each beside the scalar reference it replaces. The ratio is what `docs/design/
//! s2.8-neon.md` §7.3 asks for: it separates "the kernel is faster" from "the model got faster".
//!
//! The second form is S5.4h's decision table: the *exact* tile kernel against the *8-bit form*,
//! for Q4_K, Q6_K and Q5_0, at 8/64/128/512 activation rows, at one and `threads` workers, on
//! the whole-block shape and on Qwen 2.5's 896-wide row (three and a half super-blocks, so the
//! form pads a fourth). It goes through `MatKernels`' own entry points, so each tier's table
//! selects the body it would in the engine: run it under `PEREGRINE_ISA=neon-dotprod` and
//! `PEREGRINE_ISA=neon-i8mm` to get the two aarch64 rows of the table.

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("neonbench is an aarch64 benchmark");
}

#[cfg(target_arch = "aarch64")]
fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().is_some_and(|a| a == "table") {
        form_kernel_table(&raw[1..]);
        return;
    }
    use peregrine_kernels::format::Format;
    use peregrine_kernels::kq::{
        dot_q4k8_ref, dot_q5_08_ref, dot_q6k8_ref, repack_q4k8, repack_q5_08, repack_q6k8,
        repack_q8_08, KqActs, Q4K8, Q5_08, Q6K8, Q8_08,
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
    let q6_rows: Vec<Vec<u8>> = (0..8)
        .map(|_| {
            let mut w = Vec::with_capacity(n_sb * 210);
            for _ in 0..n_sb {
                for _ in 0..192 {
                    w.push(next());
                }
                for _ in 0..16 {
                    // Small signed scales, so the products stay in the range a real
                    // model's do.
                    w.push(next() % 17);
                }
                w.extend_from_slice(&0x2000u16.to_le_bytes());
            }
            w
        })
        .collect();
    let mut q6_tile = vec![0u8; n_sb * Q6K8::SIZE];
    for sb in 0..n_sb {
        let src: Vec<&[u8]> = q6_rows.iter().map(|r| &r[sb * 210..][..210]).collect();
        repack_q6k8(&src, &mut q6_tile[sb * Q6K8::SIZE..]);
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

    // Room for a 512-row batch of the eight-row tile (512 · 8 floats).
    let mut out = vec![0.0f32; 8192];
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

    // The repacked Q8_0 tile (eight rows per block): its `sdot` body against the
    // `smmla` body S4.9q adds, at the batch sizes the tile GEMM sees. One activation
    // row is the GEMV path; 512 is a full encoder batch.
    let mut q8_tile = vec![0u8; n_blk * Q8_08::SIZE];
    for t in 0..n_blk {
        let src: Vec<&[u8]> = (0..8).map(|_| &q8_row[t * 34..][..34]).collect();
        repack_q8_08(&src, &mut q8_tile[t * Q8_08::SIZE..]);
    }
    let wide: Vec<QuantAct> = (0..512)
        .map(|i| {
            let y: Vec<f32> = (0..n)
                .map(|j| (((i * 331 + j) as f32) * 0.0013).sin())
                .collect();
            let mut a = QuantAct::with_capacity(n);
            a.fill(&y);
            a
        })
        .collect();
    for nb in [1usize, 8, 64, 512] {
        bench(&format!("Q8_0 tile sdot, GEMM {nb}"), 8 * nb, &mut || {
            unsafe {
                peregrine_kernels::neon::q8_08_gemm(&q8_tile, n_blk, &wide[..nb], nb, &mut out)
            };
            black_box(out[0]);
        });
    }
    if peregrine_kernels::cpu::have_i8mm() {
        for nb in [1usize, 8, 64, 512] {
            bench(&format!("Q8_0 tile i8mm, GEMM {nb}"), 8 * nb, &mut || {
                unsafe {
                    peregrine_kernels::neon::q8_08_gemm_i8mm(
                        &q8_tile,
                        n_blk,
                        &wide[..nb],
                        nb,
                        &mut out,
                    )
                };
                black_box(out[0]);
            });
        }
    }

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
    if peregrine_kernels::cpu::have_i8mm() {
        bench("Q4_K tile i8mm, GEMM 4", (rows / 8) * 32, &mut || {
            for _ in 0..rows / 8 {
                unsafe { peregrine_kernels::neon::q4k8_gemm_i8mm(&tile, n_sb, &kq, 4, &mut out) };
            }
            black_box(out[0]);
        });
        bench("Q4_K tile i8mm, GEMM 16", (rows / 16) * 128, &mut || {
            for _ in 0..rows / 16 {
                unsafe { peregrine_kernels::neon::q4k8_gemm_i8mm(&tile, n_sb, &kq, 16, &mut out) };
            }
            black_box(out[0]);
        });
    }

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
    if peregrine_kernels::cpu::have_i8mm() {
        bench("Q5_0 tile i8mm, GEMM 4", (rows / 8) * 32, &mut || {
            for _ in 0..rows / 8 {
                unsafe {
                    peregrine_kernels::neon::q5_08_gemm_i8mm(&q5_tile, n_blk, &acts, 4, &mut out)
                };
            }
            black_box(out[0]);
        });
    }

    bench("Q6_K scalar, 1 row", rows, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows {
            acc += dot_scalar(Format::Q6_K, &q6_rows[0], &act);
        }
        black_box(acc);
    });
    bench("Q6_K sdot, 1 row", rows, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows {
            acc += unsafe { peregrine_kernels::neon::dot_q6_k(&q6_rows[0], &act) };
        }
        black_box(acc);
    });
    bench("Q6_K tile ref, GEMV", (rows / 8) * 8, &mut || {
        let mut acc = 0.0f32;
        for _ in 0..rows / 8 {
            for r in 0..8 {
                acc += dot_q6k8_ref(&q6_tile, n_sb, r, &act);
            }
        }
        black_box(acc);
    });
    bench("Q6_K tile sdot, GEMV", (rows / 8) * 8, &mut || {
        for _ in 0..rows / 8 {
            unsafe {
                peregrine_kernels::neon::q6k8_gemv(
                    &q6_tile,
                    n_sb,
                    std::slice::from_ref(&act),
                    1,
                    &mut out,
                )
            };
        }
        black_box(out[0]);
    });
    bench("Q6_K tile sdot, GEMM 4", (rows / 8) * 32, &mut || {
        for _ in 0..rows / 8 {
            unsafe { peregrine_kernels::neon::q6k8_gemm(&q6_tile, n_sb, &kq, 4, &mut out) };
        }
        black_box(out[0]);
    });
    bench("Q6_K tile sdot, GEMM 16", (rows / 16) * 128, &mut || {
        for _ in 0..rows / 16 {
            unsafe { peregrine_kernels::neon::q6k8_gemm(&q6_tile, n_sb, &kq, 16, &mut out) };
        }
        black_box(out[0]);
    });
    if peregrine_kernels::cpu::have_i8mm() {
        bench("Q6_K tile i8mm, GEMM 4", (rows / 8) * 32, &mut || {
            for _ in 0..rows / 8 {
                unsafe {
                    peregrine_kernels::neon::q6k8_gemm_i8mm(&q6_tile, n_sb, &kq, 4, &mut out)
                };
            }
            black_box(out[0]);
        });
        bench("Q6_K tile i8mm, GEMM 16", (rows / 16) * 128, &mut || {
            for _ in 0..rows / 16 {
                unsafe {
                    peregrine_kernels::neon::q6k8_gemm_i8mm(&q6_tile, n_sb, &kq, 16, &mut out)
                };
            }
            black_box(out[0]);
        });
    }

    // The 8-bit prompt form (S5.4d): the resident form against the exact tile
    // kernels, on `sdot` and on `smmla`, at the widths the design note lists.
    // `n_tiles` is `rows / 8`, capped at 256: at `rows=2048` and `n=8192` the
    // 17 MB of forms stream from L3/DRAM on every iteration, which is the
    // 2048 × 8192 shape; at `n=4096, rows=512` the same loop is L2/L3-resident.
    {
        use peregrine_kernels::kq8;
        let n_tiles = (rows / 8).clamp(1, 256);
        let mut form4 = vec![0u8; n_sb * kq8::Q8K_SIZE];
        let mut form6 = vec![0u8; n_sb * kq8::Q8K_SIZE];
        let n_sb5 = n_blk.div_ceil(8);
        let mut form5 = vec![0u8; n_sb5 * kq8::Q8K_SIZE];
        unsafe {
            peregrine_kernels::neon::q8k_build_q4k_neon(&tile, n_sb, &mut form4);
            peregrine_kernels::neon::q8k_build_q6k_neon(&q6_tile, n_sb, &mut form6);
            peregrine_kernels::neon::q8k_build_q5_0_neon(&q5_tile, n_blk, &mut form5);
        }
        let forms4 = vec![form4.clone(); n_tiles];
        let forms6 = vec![form6.clone(); n_tiles];
        let forms5 = vec![form5.clone(); n_tiles];
        let tiles4 = vec![tile; n_tiles];
        let tiles6 = vec![q6_tile; n_tiles];
        let tiles5 = vec![q5_tile; n_tiles];
        let mut kq512 = KqActs::default();
        kq512.fill(&wide);
        let macs = |nb: usize| 8 * n_tiles * nb;
        println!(
            "8-bit form: n={n} n_tiles={n_tiles} ({} KiB of Q4_K form per iteration)",
            n_tiles * form4.len() / 1024
        );
        for nb in [1usize, 8, 16, 64, 128, 512] {
            bench(&format!("Q4_K exact sdot, {nb}"), macs(nb), &mut || {
                for t in tiles4.iter() {
                    unsafe { peregrine_kernels::neon::q4k8_gemm(t, n_sb, &kq512, nb, &mut out) };
                }
                black_box(out[0]);
            });
            bench(&format!("Q4_K form sdot, {nb}"), macs(nb), &mut || {
                for f in forms4.iter() {
                    unsafe {
                        peregrine_kernels::neon::q8k_gemm_sdot(f, n_sb, &kq512, nb, &mut out)
                    };
                }
                black_box(out[0]);
            });
            bench(&format!("Q6_K exact sdot, {nb}"), macs(nb), &mut || {
                for t in tiles6.iter() {
                    unsafe { peregrine_kernels::neon::q6k8_gemm(t, n_sb, &kq512, nb, &mut out) };
                }
                black_box(out[0]);
            });
            bench(&format!("Q6_K form sdot, {nb}"), macs(nb), &mut || {
                for f in forms6.iter() {
                    unsafe {
                        peregrine_kernels::neon::q8k_gemm_sdot(f, n_sb, &kq512, nb, &mut out)
                    };
                }
                black_box(out[0]);
            });
            bench(&format!("Q5_0 exact sdot, {nb}"), macs(nb), &mut || {
                for t in tiles5.iter() {
                    unsafe {
                        peregrine_kernels::neon::q5_08_gemm(t, n_blk, &wide[..nb], nb, &mut out)
                    };
                }
                black_box(out[0]);
            });
            bench(&format!("Q5_0 form sdot, {nb}"), macs(nb), &mut || {
                for f in forms5.iter() {
                    unsafe {
                        peregrine_kernels::neon::q8k_gemm_sdot(f, n_sb5, &kq512, nb, &mut out)
                    };
                }
                black_box(out[0]);
            });
            if peregrine_kernels::cpu::have_i8mm() {
                bench(&format!("Q4_K exact i8mm, {nb}"), macs(nb), &mut || {
                    for t in tiles4.iter() {
                        unsafe {
                            peregrine_kernels::neon::q4k8_gemm_i8mm(t, n_sb, &kq512, nb, &mut out)
                        };
                    }
                    black_box(out[0]);
                });
                bench(&format!("Q4_K form i8mm, {nb}"), macs(nb), &mut || {
                    for f in forms4.iter() {
                        unsafe {
                            peregrine_kernels::neon::q8k_gemm_i8mm(f, n_sb, &kq512, nb, &mut out)
                        };
                    }
                    black_box(out[0]);
                });
                bench(&format!("Q6_K exact i8mm, {nb}"), macs(nb), &mut || {
                    for t in tiles6.iter() {
                        unsafe {
                            peregrine_kernels::neon::q6k8_gemm_i8mm(t, n_sb, &kq512, nb, &mut out)
                        };
                    }
                    black_box(out[0]);
                });
                bench(&format!("Q6_K form i8mm, {nb}"), macs(nb), &mut || {
                    for f in forms6.iter() {
                        unsafe {
                            peregrine_kernels::neon::q8k_gemm_i8mm(f, n_sb, &kq512, nb, &mut out)
                        };
                    }
                    black_box(out[0]);
                });
                bench(&format!("Q5_0 exact i8mm, {nb}"), macs(nb), &mut || {
                    for t in tiles5.iter() {
                        unsafe {
                            peregrine_kernels::neon::q5_08_gemm_i8mm(
                                t,
                                n_blk,
                                &wide[..nb],
                                nb,
                                &mut out,
                            )
                        };
                    }
                    black_box(out[0]);
                });
                bench(&format!("Q5_0 form i8mm, {nb}"), macs(nb), &mut || {
                    for f in forms5.iter() {
                        unsafe {
                            peregrine_kernels::neon::q8k_gemm_i8mm(f, n_sb5, &kq512, nb, &mut out)
                        };
                    }
                    black_box(out[0]);
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// S5.4h: the table that decides which formats take the 8-bit form on this tier
// ---------------------------------------------------------------------------

/// The exact tile kernel against the 8-bit form, per format and tier, at the
/// widths a prompt pass has (8, 64, 128 and 512 activation rows), on the
/// whole-block shape and on Qwen 2.5's 896-wide row.
///
/// One "row" of work is eight output rows of a tile over `n_sb` 256-weight
/// super-blocks (a Q5_0 tile is `8·n_sb` 32-weight blocks) against `nb`
/// activation rows, so the wide-pass ratio is the kernel's own and not the
/// model's. Both sides go through `MatKernels`, so the exact side is the body the
/// engine runs on this cap and the form side is the body S5.4h is deciding about.
///
/// Each worker holds its own copy of the tile and the form (the engine splits
/// *output* rows across workers), so the multi-worker column is the aggregate
/// rate of `threads` independent copies.
#[cfg(target_arch = "aarch64")]
fn form_kernel_table(args: &[String]) {
    use peregrine_kernels::format::Format;
    use peregrine_kernels::kq::{
        repack_q4k8, repack_q5_08, repack_q6k8, KqActs, Q4K8, Q5_08, Q6K8,
    };
    use peregrine_kernels::kq8::Q8K_SIZE;
    use peregrine_kernels::quant::{f32_to_f16, QuantAct};
    use peregrine_kernels::table::MatKernels;
    use std::time::Instant;

    let num = |i: usize, d: usize| -> usize {
        args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d).max(1)
    };
    let n_sb = num(0, 8);
    let reps = num(1, 20);
    let threads = num(2, 10);
    let which = args.get(3).map(String::as_str).unwrap_or("both");
    let mk = MatKernels::get();
    if !peregrine_kernels::cpu::have_dotprod() {
        eprintln!("this CPU (or PEREGRINE_ISA cap) has no asimddp; nothing to measure");
        return;
    }
    println!(
        "S5.4h form table: tier={} body={} n_sb={n_sb} reps={reps} threads={threads} shape={which}",
        peregrine_kernels::cpu::tier_name(),
        mk.q8k_body().unwrap_or("none")
    );
    // The shapes: a whole 2048-weight row (`n_sb` super-blocks), and Qwen 2.5's
    // 896-weight row — 28 Q5_0 blocks, which the form rounds up to four
    // super-blocks, so a quarter of its codes are padding.
    let mut shapes: Vec<(&str, usize, usize, usize, bool)> = Vec::new();
    if which == "whole" || which == "both" {
        shapes.push(("whole", n_sb * 256, n_sb, n_sb * 8, false));
    }
    if which == "row896" || which == "both" {
        shapes.push(("row896", 896, 4, 28, true));
    }
    let mut prng = {
        let mut s = 0x243f_6a88_85a3_08d3u64;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        }
    };
    for &(shape, n, form_sb, n_blk, _padded) in shapes.iter() {
        // The activation buffer: 896 real weights are padded to 1024 codes, and the
        // fill of the padded 256-block leaves the missing codes zero.
        let n_act = form_sb * 256;
        for fmt in ["q4k", "q6k", "q5_0"] {
            // Q4_K and Q6_K rows are whole 256-weight super-blocks by format; only
            // Q5_0 has a 896-wide row that stops inside one.
            if shape == "row896" && fmt != "q5_0" {
                continue;
            }
            // The exact tile, eight rows, as the engine's store lays it out.
            let (tile, blocks) = match fmt {
                "q4k" => {
                    let rows: Vec<Vec<u8>> = (0..8)
                        .map(|_| {
                            let mut w = Vec::new();
                            for _ in 0..n_sb {
                                w.extend_from_slice(&f32_to_f16(0.01).to_le_bytes());
                                w.extend_from_slice(&f32_to_f16(0.005).to_le_bytes());
                                for _ in 0..12 {
                                    w.push(prng());
                                }
                                for _ in 0..128 {
                                    w.push(prng());
                                }
                            }
                            w
                        })
                        .collect();
                    let mut tile = vec![0u8; n_sb * Q4K8::SIZE];
                    for sb in 0..n_sb {
                        let src: Vec<&[u8]> = rows.iter().map(|r| &r[sb * 144..][..144]).collect();
                        repack_q4k8(&src, &mut tile[sb * Q4K8::SIZE..]);
                    }
                    (tile, n_sb)
                }
                "q6k" => {
                    let rows: Vec<Vec<u8>> = (0..8)
                        .map(|_| {
                            let mut w = Vec::new();
                            for _ in 0..n_sb {
                                for _ in 0..192 {
                                    w.push(prng());
                                }
                                for _ in 0..16 {
                                    w.push(prng() % 17);
                                }
                                w.extend_from_slice(&f32_to_f16(0.01).to_le_bytes());
                            }
                            w
                        })
                        .collect();
                    let mut tile = vec![0u8; n_sb * Q6K8::SIZE];
                    for sb in 0..n_sb {
                        let src: Vec<&[u8]> = rows.iter().map(|r| &r[sb * 210..][..210]).collect();
                        repack_q6k8(&src, &mut tile[sb * Q6K8::SIZE..]);
                    }
                    (tile, n_sb)
                }
                _ => {
                    let rows: Vec<Vec<u8>> = (0..8)
                        .map(|_| {
                            let mut w = Vec::new();
                            for _ in 0..n_blk {
                                w.extend_from_slice(&f32_to_f16(0.02).to_le_bytes());
                                for _ in 0..20 {
                                    w.push(prng());
                                }
                            }
                            w
                        })
                        .collect();
                    let mut tile = vec![0u8; n_blk * Q5_08::SIZE];
                    for b in 0..n_blk {
                        let src: Vec<&[u8]> = rows.iter().map(|r| &r[b * 22..][..22]).collect();
                        repack_q5_08(&src, &mut tile[b * Q5_08::SIZE..]);
                    }
                    (tile, n_blk)
                }
            };
            // The form of the same tile. `n_blocks` is the super-block count for
            // Q4_K/Q6_K and the 32-weight block count for Q5_0, as `q8k_build` takes it.
            let kind = match fmt {
                "q4k" => Format::Q4_K,
                "q6k" => Format::Q6_K,
                _ => Format::Q5_0,
            };
            let mut form = vec![0u8; form_sb * Q8K_SIZE];
            mk.q8k_build(kind, &tile, blocks, &mut form);
            let useful = n;
            for &nb in &[8usize, 64, 128, 512] {
                let acts: Vec<QuantAct> = (0..nb)
                    .map(|i| {
                        let y: Vec<f32> = (0..n_act)
                            .map(|j| (((i * 331 + j) as f32) * 0.0013).sin())
                            .collect();
                        let mut a = QuantAct::with_capacity(n_act);
                        a.fill(&y);
                        a
                    })
                    .collect();
                let mut kq = KqActs::default();
                kq.fill(&acts);
                let macs = 8.0 * useful as f64 * nb as f64;
                let mut rates = Vec::new();
                for body in ["exact", "form"] {
                    let mut best = f64::MAX;
                    for _round in 0..3 {
                        let hs: Vec<_> = (0..threads)
                            .map(|_| {
                                let tile = tile.clone();
                                let form = form.clone();
                                let kq = kq.clone();
                                let acts = acts.clone();
                                std::thread::spawn(move || {
                                    let mut o = vec![0.0f32; nb * 8];
                                    // The table's own entry points: the exact
                                    // tile kernel the engine runs on this cap, and
                                    // the form's body. Both check their bounds.
                                    let run = |o: &mut [f32]| {
                                        let o = &mut o[..nb * 8];
                                        if body == "exact" {
                                            match fmt {
                                                "q4k" => mk.kq_tiles_q4k(&tile, n_sb, &kq, nb, o),
                                                "q6k" => mk.kq_tiles_q6k(&tile, n_sb, &kq, nb, o),
                                                _ => mk.kq_tiles_q5_0(
                                                    &tile,
                                                    n_blk,
                                                    &acts[..nb],
                                                    nb,
                                                    o,
                                                ),
                                            }
                                        } else {
                                            mk.q8k_gemm(&form, form_sb, &kq, nb, o);
                                        }
                                    };
                                    run(&mut o);
                                    let mut b = f64::MAX;
                                    for _ in 0..2 {
                                        let t0 = Instant::now();
                                        for _ in 0..reps {
                                            run(&mut o);
                                        }
                                        b = b.min(t0.elapsed().as_secs_f64());
                                    }
                                    b
                                })
                            })
                            .collect();
                        let wall = hs
                            .into_iter()
                            .map(|h| h.join().unwrap())
                            .fold(0.0f64, f64::max);
                        best = best.min(wall);
                    }
                    rates.push(macs * reps as f64 * threads as f64 / best / 1e9);
                }
                let (exact, form) = (rates[0], rates[1]);
                println!(
                    "{shape:>6} {fmt:>4} {nb:>3} acts  exact {exact:6.1}  form {form:6.1}  \
                     form/exact {:5.2}x  ({threads} worker{})",
                    form / exact,
                    if threads == 1 { "" } else { "s" }
                );
            }
        }
    }
}
