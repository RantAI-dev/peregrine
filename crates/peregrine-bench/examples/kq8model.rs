//! The 8-bit prompt form on a real model (S5.4a): its re-rounding error, and the
//! rows a C harness hands to the fork's own conversion for the conformance check.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example kq8model -- errors <model.gguf>
//! cargo run --release -p peregrine-bench --example kq8model -- dump <model.gguf> \
//!     <tensor substring> <groups of 16 rows> <prefix>
//! ```
//!
//! `errors` samples the Q4_K, Q6_K and Q5_0 tensors and reports the two numbers
//! the design note asks for — the worst and the mean re-rounding error relative to
//! the super-block's largest weight — the share of blocks that land exactly on the
//! grid, and what a resident form would cost.
//!
//! `dump` writes two files: `<prefix>-rows.bin`, the file's own blocks of the
//! chosen rows, and `<prefix>-peregrine.bin`, the form this crate builds from
//! them, in the canonical layout below. `forkconv.c` in the results directory
//! reads the first and writes the second from ik_llama.cpp's own
//! `iqk_convert_kquants_q8X_r8`, so `cmp` compares the two engines' codes and
//! scales byte for byte.
//!
//! ```text
//! rows.bin      "KQ8R" | u32 fmt | u32 n_sb | u32 n_groups | u32 row_bytes | rows
//! peregrine.bin "KQ8C" | u32 version=1 | u32 fmt | u32 n_sb | u32 n_groups |
//!               per group: per row (0..16): per super-block: f32 scale, 256 i8 codes
//! ```
//!
//! `fmt` is 0 for Q4_K and 1 for Q6_K.

use peregrine_core::gguf::Gguf;
use peregrine_kernels::format::Format;
use peregrine_kernels::kq::{repack_q4k8, repack_q5_08, repack_q6k8, Q4K8, Q5_08, Q6K8};
use peregrine_kernels::kq8::{
    build_q4k, build_q5_0, build_q6k, code_at, form_size, q4k_weights, q5_0_weights, q6k_products,
    scale_at, Q8K_SIZE, Q8K_WEIGHTS,
};

/// The first `n` bytes of a tensor's rows `r0 .. r0+nrows`, as one vec.
fn rows_block(
    gguf: &Gguf,
    t: &peregrine_core::gguf::TensorInfo,
    r0: usize,
    nrows: usize,
) -> Vec<u8> {
    let cols = t.dims[0] as usize;
    let rb = peregrine_kernels::quant::row_bytes(t.kind, cols);
    let all = gguf.bytes(t);
    let mut out = vec![0u8; nrows * rb];
    out.copy_from_slice(&all[r0 * rb..(r0 + nrows) * rb]);
    out
}

/// Build one tile of eight rows for super-block `sb` out of eight rows' blocks.
fn tile8(
    rows: &[u8],
    rb: usize,
    sb: usize,
    block: usize,
    repack: fn(&[&[u8]], &mut [u8]),
    size: usize,
) -> Vec<u8> {
    let mut blocks: [&[u8]; 8] = [&[]; 8];
    for (r, b) in blocks.iter_mut().enumerate() {
        *b = &rows[r * rb + sb * block..r * rb + sb * block + block];
    }
    let mut tile = vec![0u8; size];
    repack(&blocks, &mut tile);
    tile
}

/// One block's f32 weights, the re-rounded ones and the block's largest magnitude.
struct ErrStats {
    blocks: usize,
    weights: usize,
    sum_ratio: f64,
    lo: f64,
    hi: f64,
    worst: f64,
    exact: usize,
    zero: usize,
}

impl ErrStats {
    fn new() -> ErrStats {
        ErrStats {
            blocks: 0,
            weights: 0,
            sum_ratio: 0.0,
            lo: 1.0,
            hi: 0.0,
            worst: 0.0,
            exact: 0,
            zero: 0,
        }
    }

    /// `w` the block's f32 weights, `top` its largest magnitude, `scale` and
    /// `codes` the form's.
    fn add(&mut self, w: &[f32], top: f32, scale: f32, codes: &[i8; Q8K_WEIGHTS]) {
        self.blocks += 1;
        if top == 0.0 {
            self.zero += 1;
            return;
        }
        let mut sum = 0.0f64;
        for (i, &c) in codes.iter().enumerate() {
            let e = (w[i] - c as f32 * scale).abs() as f64;
            sum += e;
            if e == 0.0 {
                self.exact += 1;
            }
            self.worst = self.worst.max(e / top as f64);
            self.weights += 1;
        }
        let r = sum / w.len() as f64 / top as f64;
        self.sum_ratio += r;
        self.lo = self.lo.min(r);
        self.hi = self.hi.max(r);
    }

    fn print(&self, label: &str) {
        println!(
            "{label}: blocks {}  weights {}  all-zero blocks {}",
            self.blocks, self.weights, self.zero
        );
        if self.blocks == 0 {
            return;
        }
        println!(
            "  mean of the per-block mean |error| / block max: {:.5}   (blocks {:.5} .. {:.5})",
            self.sum_ratio / self.blocks as f64,
            self.lo,
            self.hi
        );
        println!(
            "  worst single weight / block max: {:.5}   (bound 0.5/127 = {:.5})",
            self.worst,
            0.5 / 127.0
        );
        println!(
            "  weights exactly on the grid: {:.2} %",
            100.0 * self.exact as f64 / self.weights.max(1) as f64
        );
    }
}

fn errors(path: &str) {
    let gguf = Gguf::open(std::path::Path::new(path)).expect("the model opens");
    let stride = 16;
    let mut q4 = ErrStats::new();
    let mut q6 = ErrStats::new();
    let mut q5 = ErrStats::new();
    let mut q6_exact_blocks = 0usize;
    let mut form_bytes = 0u64;
    let mut file_bytes = 0u64;
    let mut weights = 0u64;
    let mut tensors = 0usize;
    let mut names: Vec<&String> = gguf.tensors.keys().collect();
    names.sort();
    for name in names {
        let t = &gguf.tensors[name];
        let cols = t.dims[0] as usize;
        if !matches!(t.kind, Format::Q4_K | Format::Q6_K | Format::Q5_0) {
            continue;
        }
        // The k-quants are 256 weights to a super-block by definition; Q5_0's
        // blocks are 32 and a row can end in a short super-block.
        if !cols.is_multiple_of(32) || (t.kind != Format::Q5_0 && !cols.is_multiple_of(Q8K_WEIGHTS))
        {
            continue;
        }
        let rows: usize = t.dims[1..].iter().map(|&d| d as usize).product();
        if rows < 8 {
            continue;
        }
        let n_blk = cols / 32;
        let n_sb = match t.kind {
            Format::Q5_0 => n_blk.div_ceil(8),
            _ => cols / Q8K_WEIGHTS,
        };
        let block = t.kind.block_bytes();
        let geom5 = match t.kind {
            Format::Q4_K => Geometry {
                fmt: 0,
                block,
                size: Q4K8::SIZE,
                repack: repack_q4k8,
            },
            Format::Q6_K => Geometry {
                fmt: 1,
                block,
                size: Q6K8::SIZE,
                repack: repack_q6k8,
            },
            _ => Geometry {
                fmt: 2,
                block,
                size: Q5_08::SIZE,
                repack: repack_q5_08,
            },
        };
        let (size, repack, fmt) = (geom5.size, geom5.repack, geom5.fmt);
        tensors += 1;
        file_bytes += t.bytes();
        form_bytes += (rows / 8 * n_sb * Q8K_SIZE) as u64;
        weights += t.numel();
        let rb = peregrine_kernels::quant::row_bytes(t.kind, cols);
        // Sample: every `stride`-th eight-row group and every `stride`-th block.
        let mut r0 = 0;
        while r0 + 8 <= rows {
            let rows8 = rows_block(&gguf, t, r0, 8);
            // Q5_0's tile is one block per row, so the whole row group's blocks are
            // repacked once and a super-block is a slice of that.
            let mut tile5 = Vec::new();
            if fmt == 2 {
                tile5 = vec![0u8; n_blk * Q5_08::SIZE];
                for b in 0..n_blk {
                    let mut blocks: [&[u8]; 8] = [&[]; 8];
                    for (r, dst) in blocks.iter_mut().enumerate() {
                        *dst = &rows8[r * rb + b * block..r * rb + (b + 1) * block];
                    }
                    repack_q5_08(&blocks, &mut tile5[b * Q5_08::SIZE..(b + 1) * Q5_08::SIZE]);
                }
            }
            let mut sb = 0;
            while sb < n_sb {
                let (tile, n_blocks) = if fmt == 2 {
                    let b0 = sb * 8;
                    let nb = (n_blk - b0).min(8);
                    (
                        tile5[b0 * Q5_08::SIZE..(b0 + nb) * Q5_08::SIZE].to_vec(),
                        nb,
                    )
                } else {
                    (tile8(&rows8, rb, sb, block, repack, size), 8)
                };
                let mut form = vec![0u8; form_size(1)];
                match fmt {
                    0 => build_q4k(&tile, 1, &mut form),
                    1 => build_q6k(&tile, 1, &mut form),
                    _ => build_q5_0(&tile, n_blocks, &mut form),
                }
                for row in 0..8 {
                    let scale = f32::from_le_bytes(
                        form[scale_at(0, row)..scale_at(0, row) + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let mut codes = [0i8; Q8K_WEIGHTS];
                    for (i, c) in codes.iter_mut().enumerate() {
                        *c = (form[code_at(0, i / 8, row, i % 8)] as i32 - 128) as i8;
                    }
                    match fmt {
                        0 => {
                            let w = q4k_weights(&tile, row);
                            let top = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                            q4.add(&w, top, scale, &codes);
                        }
                        1 => {
                            let p = q6k_products(&tile, row);
                            let d = peregrine_kernels::f16(u16::from_le_bytes([
                                tile[peregrine_kernels::kq::q6_d() + row * 2],
                                tile[peregrine_kernels::kq::q6_d() + row * 2 + 1],
                            ]));
                            let w: Vec<f32> = p.iter().map(|&x| x as f32 * d).collect();
                            let top = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                            let m = p.iter().fold(0i32, |a, &x| a.max(x.abs()));
                            if (m as f32 * (1.0f32 / 127.0f32)) < 1.0 {
                                q6_exact_blocks += 1;
                            }
                            q6.add(&w, top, scale, &codes);
                        }
                        _ => {
                            let w = q5_0_weights(&tile, n_blocks, row);
                            let top = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                            q5.add(&w, top, scale, &codes);
                        }
                    }
                }
                sb += stride;
            }
            r0 += 8 * stride;
        }
    }
    println!("model {path}");
    println!(
        "tensors {}  weights {}  file bytes {} ({:.4} B/weight)  resident form {} ({:.4} B/weight, {:.3} GiB)",
        tensors,
        weights,
        file_bytes,
        file_bytes as f64 / weights as f64,
        form_bytes,
        form_bytes as f64 / weights as f64,
        form_bytes as f64 / (1u64 << 30) as f64
    );
    println!("sampling: every {stride}th eight-row group and every {stride}th super-block");
    println!();
    q4.print("Q4_K");
    q6.print("Q6_K");
    println!(
        "Q6_K blocks in the exact case: {} of {} sampled ({:.1} %)",
        q6_exact_blocks,
        q6.blocks,
        100.0 * q6_exact_blocks as f64 / q6.blocks.max(1) as f64
    );
    q5.print("Q5_0");
    println!();
    println!("The grid step is max/127, so the bound is half a step, max/254 = 0.3937 %;");
    println!("'block max' is the largest |w| in that row's 256-weight super-block, the number");
    println!("the form divides by 127. Every ratio above is relative to that maximum.");
}

/// How one format's tile is shaped: the file's block bytes, the repacked tile's
/// size, and the repack itself.
#[derive(Clone, Copy)]
struct Geometry {
    fmt: u8,
    block: usize,
    size: usize,
    repack: fn(&[&[u8]], &mut [u8]),
}

/// The canonical per-row codes and scales of a form built from `rows`.
fn canonical(geom: Geometry, rows: &[u8], rb: usize, n_sb: usize, n_groups: usize) -> Vec<u8> {
    let Geometry {
        fmt,
        block,
        size,
        repack,
    } = geom;
    let mut out = Vec::new();
    out.extend_from_slice(b"KQ8C");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(fmt as u32).to_le_bytes());
    out.extend_from_slice(&(n_sb as u32).to_le_bytes());
    out.extend_from_slice(&(n_groups as u32).to_le_bytes());
    for g in 0..n_groups {
        // Two eight-row tiles per group of sixteen.
        let mut forms = Vec::new();
        for half in 0..2 {
            let base = (g * 16 + half * 8) * rb;
            for sb in 0..n_sb {
                let tile = tile8(&rows[base..], rb, sb, block, repack, size);
                let mut form = vec![0u8; form_size(n_sb)];
                match fmt {
                    0 => build_q4k(&tile, 1, &mut form),
                    _ => build_q6k(&tile, 1, &mut form),
                }
                forms.push(form);
            }
        }
        for row in 0..16 {
            let half = row / 8;
            let r = row % 8;
            for sb in 0..n_sb {
                let form = &forms[half * n_sb + sb];
                out.extend_from_slice(&form[scale_at(0, r)..scale_at(0, r) + 4]);
                for i in 0..Q8K_WEIGHTS {
                    // The canonical form holds the signed code; the form itself holds
                    // it biased by 128 for `vpdpbusd`'s unsigned operand.
                    out.push((form[code_at(0, i / 8, r, i % 8)] as i32 - 128) as i8 as u8);
                }
            }
        }
    }
    out
}

fn dump(path: &str, want: &str, groups: usize, prefix: &str) {
    let gguf = Gguf::open(std::path::Path::new(path)).expect("the model opens");
    let mut names: Vec<&String> = gguf.tensors.keys().collect();
    names.sort();
    let name = names
        .iter()
        .find(|n| {
            n.contains(want) && {
                let t = &gguf.tensors[*n as &str];
                t.kind == Format::Q4_K || t.kind == Format::Q6_K
            }
        })
        .unwrap_or_else(|| panic!("no Q4_K or Q6_K tensor matches {want:?}"));
    let t = gguf.tensors[*name].clone();
    let cols = t.dims[0] as usize;
    let n_sb = cols / Q8K_WEIGHTS;
    let rows: usize = t.dims[1..].iter().map(|&d| d as usize).product();
    assert!(rows >= 16 * groups, "the tensor has fewer than 16 rows");
    let rows_bytes = rows_block(&gguf, &t, 0, 16 * groups);
    let rb = peregrine_kernels::quant::row_bytes(t.kind, cols);
    let geom = match t.kind {
        Format::Q4_K => Geometry {
            fmt: 0,
            block: 144,
            size: Q4K8::SIZE,
            repack: repack_q4k8,
        },
        _ => Geometry {
            fmt: 1,
            block: 210,
            size: Q6K8::SIZE,
            repack: repack_q6k8,
        },
    };
    let fmt = geom.fmt;
    let mut out = Vec::new();
    out.extend_from_slice(b"KQ8R");
    out.extend_from_slice(&(fmt as u32).to_le_bytes());
    out.extend_from_slice(&(n_sb as u32).to_le_bytes());
    out.extend_from_slice(&(groups as u32).to_le_bytes());
    out.extend_from_slice(&(rb as u32).to_le_bytes());
    out.extend_from_slice(&rows_bytes);
    std::fs::write(format!("{prefix}-rows.bin"), &out).expect("rows.bin writes");
    let can = canonical(geom, &rows_bytes, rb, n_sb, groups);
    std::fs::write(format!("{prefix}-peregrine.bin"), &can).expect("peregrine.bin writes");
    println!(
        "tensor {name} ({:?}), rows {} of {}, {} super-blocks, {} groups of 16 rows written",
        t.kind,
        16 * groups,
        rows,
        n_sb,
        groups
    );
}

/// Per-weight RMS and max error of the two 8-bit forms against the file's own
/// dequantised weights (S5.4i): one scale per 256 weights (`crate::kq8`, the
/// fast prompt mode's form) and one per 32 (`crate::p32`).
///
/// Every error is relative to the **super-block's** largest weight, the denominator
/// S5.4a's requant tables use, so the two forms are directly comparable; the max is
/// also given as a multiple of half the step the form can take (`max/254` per 256
/// weights, `max/254` per 32 with that block's own max).
fn p32_errors(path: &str) {
    use peregrine_kernels::kq8::{build_q4k, build_q6k, code_at, scale_at, Q8K_SIZE};
    use peregrine_kernels::p32::{build_tile, code_at as p32_at, P32_D, P32_ROWS, P32_SIZE};
    let gguf = Gguf::open(std::path::Path::new(path)).expect("the model opens");
    let mut names: Vec<&String> = gguf.tensors.keys().collect();
    names.sort();
    // [0] = per 256 weights, [1] = per 32 weights.
    // [form][format]: 0 = per 256, 1 = per 32; format 0 = Q4_K, 1 = Q6_K.
    let mut n = [[0u64; 2]; 2];
    let mut sq = [[0.0f64; 2]; 2];
    let mut hi = [[0.0f64; 2]; 2];
    let mut hi_abs = [[0.0f64; 2]; 2];
    let mut mean = [[0.0f64; 2]; 2];
    let mut tensors = 0usize;
    for name in names {
        let t = &gguf.tensors[name];
        let cols = t.dims[0] as usize;
        if !matches!(t.kind, Format::Q4_K | Format::Q6_K) || !cols.is_multiple_of(Q8K_WEIGHTS) {
            continue;
        }
        let rows: usize = t.dims[1..].iter().map(|&d| d as usize).product();
        if rows < P32_ROWS {
            continue;
        }
        tensors += 1;
        let n_sb = cols / Q8K_WEIGHTS;
        let block = t.kind.block_bytes();
        type Repack = fn(&[&[u8]], &mut [u8]);
        let (size, repack, fmt): (usize, Repack, usize) = match t.kind {
            Format::Q4_K => (Q4K8::SIZE, repack_q4k8, 0usize),
            _ => (Q6K8::SIZE, repack_q6k8, 1usize),
        };
        // Every 16th super-block of every tensor, and all eight rows of its tile.
        let mut sb = 0usize;
        while sb < n_sb {
            let row0 = 0usize; // row 0 of the tensor; the tile is rows row0..row0+8
            let tile = tile8(gguf.bytes(t), row_bytes_of(t), sb, block, repack, size);
            let mut w: [[f32; Q8K_WEIGHTS]; P32_ROWS] = [[0.0; Q8K_WEIGHTS]; P32_ROWS];
            for (r, wr) in w.iter_mut().enumerate() {
                let one = if fmt == 0 {
                    q4k_weights(&tile, r)
                } else {
                    peregrine_kernels::kq8::q6k_weights(&tile, r)
                };
                *wr = one;
            }
            let mut form = vec![0u8; Q8K_SIZE];
            if fmt == 0 {
                build_q4k(&tile, 1, &mut form)
            } else {
                build_q6k(&tile, 1, &mut form)
            }
            let mut refs: [&[f32]; P32_ROWS] = [&[]; P32_ROWS];
            for (r, x) in refs.iter_mut().enumerate() {
                *x = w[r].as_slice();
            }
            let mut form32 = vec![0u8; P32_SIZE * 8];
            build_tile(&refs, &mut form32);
            for (r, wr) in w.iter().enumerate() {
                let top = wr.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                if top == 0.0 {
                    continue;
                }
                let d256 = f32::from_le_bytes(
                    form[scale_at(0, r)..scale_at(0, r) + 4].try_into().unwrap(),
                );
                for (i, &x) in wr.iter().enumerate() {
                    let g = i / 8;
                    let k = i % 8;
                    let c = form[code_at(0, g, r, k)] as i32 - 128;
                    let e = (x - c as f32 * d256).abs() as f64;
                    n[0][fmt] += 1;
                    sq[0][fmt] += e * e;
                    mean[0][fmt] += e / top as f64;
                    hi[0][fmt] = hi[0][fmt].max(e / top as f64);
                    hi_abs[0][fmt] = hi_abs[0][fmt].max(e);
                }
                for blk in 0..8 {
                    let unit = &form32[blk * P32_SIZE..(blk + 1) * P32_SIZE];
                    let d = f32::from_le_bytes(
                        unit[P32_D + r * 4..P32_D + r * 4 + 4].try_into().unwrap(),
                    );
                    for j in 0..4 {
                        for k in 0..8 {
                            let c = unit[p32_at(j, r, k)] as i32 - 128;
                            let x = wr[blk * 32 + j * 8 + k];
                            let e = (x - c as f32 * d).abs() as f64;
                            n[1][fmt] += 1;
                            sq[1][fmt] += e * e;
                            mean[1][fmt] += e / top as f64;
                            hi[1][fmt] = hi[1][fmt].max(e / top as f64);
                            hi_abs[1][fmt] = hi_abs[1][fmt].max(e);
                        }
                    }
                }
            }
            let _ = row0;
            sb += 16;
        }
    }
    println!("{path}: {tensors} Q4_K/Q6_K tensors, every 16th super-block, all eight rows");
    for (i, label) in ["per 256 weights", "per 32 weights"].iter().enumerate() {
        for (f, fmt) in ["Q4_K", "Q6_K"].iter().enumerate() {
            println!(
                "  {label:>16} {fmt:>4}: weights {:>7}  RMS {:.6e}  max {:.6e}  \
                 (mean / super-block max {:.5}, max / super-block max {:.5})",
                n[i][f],
                (sq[i][f] / n[i][f] as f64).sqrt(),
                hi_abs[i][f],
                mean[i][f] / n[i][f] as f64,
                hi[i][f]
            );
        }
    }
}

fn row_bytes_of(t: &peregrine_core::gguf::TensorInfo) -> usize {
    let cols = t.dims[0] as usize;
    peregrine_kernels::quant::row_bytes(t.kind, cols)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("errors") => errors(&args[1]),
        Some("p32") => p32_errors(&args[1]),
        Some("dump") => dump(
            &args[1],
            &args[2],
            args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4),
            &args[4],
        ),
        _ => eprintln!(
            "usage: kq8model errors <model.gguf>\n       kq8model p32 <model.gguf>\n       \
             kq8model dump <model.gguf> <tensor> <groups> <prefix>"
        ),
    }
}
