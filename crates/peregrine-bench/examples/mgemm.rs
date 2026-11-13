//! The packed float GEMM's rate against the activation-row count, at 1 and 4 threads.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example mgemm
//! cargo run --release -p peregrine-bench --example mgemm -- --threads 1
//! cargo run --release -p peregrine-bench --example mgemm -- --threads 1,4 --reps 3
//! ```
//!
//! The question this answers is S4.9v's: the two new encoder families' 128-token pass is
//! 89–90 % this kernel and runs at 180–181 GMAC/s against ONNX Runtime's MLAS at 202–206
//! on 128 rows (level at 512). The engine calls the kernel in **units of two panels
//! against a row block** (`peregrine_core::blocks::linear`'s `FP_UNIT_PANELS`),
//! partitioned across the pool, so this bench does the same: the panels are cut into
//! `--unit`-panel units, the units are shared out over a persistent pool, and each unit
//! is one `gemm_strided` call writing the projection's row-major output (`--contiguous`
//! writes a private block instead, for the store layout's cost).
//!
//! For each (n_in, n_out) — the encoders' four shapes and four crossed ones — and each
//! row count it prints GMAC/s for F32 and F16 at one and four threads, the call's own
//! time in ms, the fraction of this core's measured roof (S4.9p: 259.6 GMAC/s on four
//! threads, 64.9 on one; the register-only FMA peak is measured too), and what the
//! engine pays before the kernel: the activation copy (`QuantAct::fill`).

use peregrine_kernels::fpack;
use peregrine_kernels::quant::QuantAct;
use peregrine_kernels::Format;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

/// The projection shapes: nomic-embed-text v1.5 / jina-embeddings-v2-base-en (d_model 768,
/// feed-forward 3 072) and the two small BERT encoders (384, 1 536), plus the transposes a
/// feed-forward's down projection has (`n_in` 3 072, `n_in` 2 304) and the crossed ones.
const SHAPES: [(usize, usize); 8] = [
    (384, 384),
    (384, 1536),
    (768, 768),
    (768, 3072),
    (384, 3072),
    (768, 1536),
    (3072, 768),
    (2304, 768),
];
const ROWS: [usize; 8] = [16, 32, 64, 96, 128, 192, 256, 512];
/// The engine's unit: two panels (peregrine-core `blocks::linear::FP_UNIT_PANELS`).
const UNIT_PANELS: usize = 2;

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
                _ => w[at..at + 2]
                    .copy_from_slice(&peregrine_kernels::quant::f32_to_f16(v).to_le_bytes()),
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

/// One unit of work: [`UNIT_PANELS`] panels against `nb` activation rows.
#[derive(Clone, Copy)]
struct Unit {
    first_panel: usize,
}

/// Run `reps` calls of every unit over a persistent `threads`-wide pool, returning the
/// minimum per-call time in seconds.
///
/// The pool spins rather than parks: `std::sync::Barrier` futex-waits, and at 128 rows a
/// call is 0.1 ms, so the wake-up latency would be most of the measurement (the engine's
/// own pool spins for the same reason). Each worker waits on an epoch counter and reports
/// completion on a counter the caller drains.
#[allow(clippy::too_many_arguments)]
fn timed(
    fmt: Format,
    packed: &[u8],
    n_in: usize,
    panels: usize,
    xs: &[QuantAct],
    threads: usize,
    reps: usize,
    wide_stride: bool,
    unit_panels: usize,
) -> f64 {
    let pb = fpack::panel_bytes(fmt, n_in);
    let rows = xs.len();
    let n_out = panels * fpack::PANEL;
    let units: Vec<Unit> = (0..panels.div_ceil(unit_panels))
        .map(|u| Unit {
            first_panel: u * unit_panels,
        })
        .collect();
    let next = AtomicUsize::new(0);
    let epoch = AtomicUsize::new(0);
    let finished = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    // The engine writes the unit's columns of the projection's row-major output
    // (`out_stride = n_out`); a private block is contiguous. Both are timed.
    let out_stride = if wide_stride {
        n_out
    } else {
        unit_panels * fpack::PANEL
    };
    let mut times = Vec::new();
    // The caller is one of the workers, as in the engine's pool: `threads` threads on
    // `threads` CPUs, never `threads + 1` (a spinning caller would preempt a worker).
    let helpers = threads.saturating_sub(1);
    std::thread::scope(|s| {
        for t in 0..helpers {
            let (packed, xs, units, next, epoch, finished, stop) =
                (packed, xs, &units, &next, &epoch, &finished, &stop);
            s.spawn(move || {
                let mut buf = vec![
                    0.0f32;
                    if wide_stride {
                        rows * n_out
                    } else {
                        (rows - 1) * out_stride + out_stride
                    }
                ];
                let mut seen = 0usize;
                loop {
                    while epoch.load(Ordering::Acquire) == seen {
                        if stop.load(Ordering::Acquire) {
                            return;
                        }
                        std::hint::spin_loop();
                    }
                    seen = epoch.load(Ordering::Acquire);
                    run_units(
                        fmt,
                        packed,
                        n_in,
                        panels,
                        xs,
                        units,
                        next,
                        unit_panels,
                        pb,
                        &mut buf,
                        wide_stride,
                        rows,
                        n_out,
                        out_stride,
                    );
                    let _ = t;
                    finished.fetch_add(1, Ordering::Release);
                }
            });
        }
        let mut buf = vec![
            0.0f32;
            if wide_stride {
                rows * n_out
            } else {
                (rows - 1) * out_stride + out_stride
            }
        ];
        for r in 0..reps {
            next.store(0, Ordering::Relaxed);
            let t0 = Instant::now();
            epoch.fetch_add(1, Ordering::Release);
            run_units(
                fmt,
                packed,
                n_in,
                panels,
                xs,
                &units,
                &next,
                unit_panels,
                pb,
                &mut buf,
                wide_stride,
                rows,
                n_out,
                out_stride,
            );
            while finished.load(Ordering::Acquire) < helpers * (r + 1) {
                std::hint::spin_loop();
            }
            times.push(t0.elapsed().as_secs_f64());
        }
        stop.store(true, Ordering::Release);
        epoch.fetch_add(1, Ordering::Release);
    });
    // The minimum: the kernel's own cost, without a co-tenant's interruption.
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[0]
}

/// One worker's pass over the unit list (dynamic, one unit at a time).
#[allow(clippy::too_many_arguments)]
fn run_units(
    fmt: Format,
    packed: &[u8],
    n_in: usize,
    panels: usize,
    xs: &[QuantAct],
    units: &[Unit],
    next: &AtomicUsize,
    unit_panels: usize,
    pb: usize,
    buf: &mut [f32],
    wide_stride: bool,
    rows: usize,
    n_out: usize,
    out_stride: usize,
) {
    let mut u = next.fetch_add(1, Ordering::Relaxed);
    while u < units.len() {
        let first = units[u].first_panel;
        let pn = unit_panels.min(panels - first);
        let off = first * pb;
        fpack::gemm_strided(
            fmt,
            &packed[off..],
            n_in,
            pn,
            xs,
            None,
            &mut buf[..if wide_stride {
                rows * n_out
            } else {
                (rows - 1) * out_stride + pn * fpack::PANEL
            }],
            out_stride,
        );
        u = next.fetch_add(1, Ordering::Relaxed);
    }
}

/// This core's peak: twelve independent FMA chains in registers.
fn fma_peak(threads: usize) -> f64 {
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
        let one = {
            let t0 = Instant::now();
            let mut reps = 0u64;
            while t0.elapsed().as_secs_f64() < 0.2 {
                std::hint::black_box(unsafe { peak(ROUNDS) });
                reps += 1;
            }
            (12 * 8 * ROUNDS) as f64 * reps as f64 / t0.elapsed().as_secs_f64() / 1e9
        };
        one * threads as f64
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = threads;
        0.0
    }
}

fn main() {
    let mut threads = vec![1usize, 4];
    let mut reps = 15usize;
    let mut stride = true;
    let mut unit = UNIT_PANELS;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                threads = args[i + 1]
                    .split(',')
                    .filter_map(|t| t.parse().ok())
                    .collect();
                i += 2;
            }
            "--unit" => {
                unit = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--contiguous" => {
                stride = false;
                i += 1;
            }
            "--reps" => {
                reps = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let tier = peregrine_kernels::cpu::tier_name();
    println!("tier {tier}, GMAC/s (1 GMAC = 10^9 multiply-adds), {reps} timed calls per cell");
    let peak4 = fma_peak(4);
    let peak1 = fma_peak(1);
    println!("FMA roof: one thread {peak1:.1}, four threads {peak4:.1}");
    for (n_in, n_out) in SHAPES {
        let panels = n_out.div_ceil(fpack::PANEL);
        println!("\n== n_in {n_in}, n_out {n_out} ({panels} panels)");
        let w32 = weights(Format::F32, n_out, n_in, 31);
        let w16 = weights(Format::F16, n_out, n_in, 32);
        let p32 = packed(Format::F32, &w32, n_out, n_in);
        let p16 = packed(Format::F16, &w16, n_out, n_in);
        println!(
            "{:>4} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9}",
            "rows", "f32 1t", "f32 4t", "f16 1t", "f16 4t", "f16 4t %", "ms f16 4t", "fill ms"
        );
        for nb in ROWS {
            let xs = acts(nb, n_in, 0x5eed ^ nb as u64);
            let macs = (n_out as f64) * (n_in as f64) * (nb as f64);
            let mut v = [0.0f64; 4];
            for (ti, &t) in threads.iter().take(2).enumerate() {
                v[ti * 2] =
                    macs / timed(Format::F32, &p32, n_in, panels, &xs, t, reps, stride, unit) / 1e9;
                v[ti * 2 + 1] =
                    macs / timed(Format::F16, &p16, n_in, panels, &xs, t, reps, stride, unit) / 1e9;
            }
            let four = if threads.contains(&4) { v[3] } else { 0.0 };
            let cell = |x: f64| {
                if threads.contains(&4) {
                    format!("{x:>10.1}")
                } else {
                    format!("{:>10}", "—")
                }
            };
            let pct = four / if threads.contains(&4) { peak4 } else { peak1 } * 100.0;
            let (pct_s, ms) = if four > 0.0 {
                (
                    format!("{:>8.1}%", pct),
                    format!("{:>9.3}", macs / four / 1e6),
                )
            } else {
                (format!("{:>9}", "—"), format!("{:>9}", "—"))
            };
            // What the engine pays before the kernel: `linear_rows` fills the block's
            // `QuantAct`s from the f32 rows, on the calling thread.
            let slab = vec![0.5f32; nb * n_in];
            let fill = {
                let mut a = acts(nb, n_in, 1);
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    for (i, x) in a.iter_mut().enumerate() {
                        x.fill(&slab[i * n_in..(i + 1) * n_in]);
                    }
                    std::hint::black_box(&a);
                    best = best.min(t0.elapsed().as_secs_f64());
                }
                best * 1e3
            };
            println!(
                "{:>4} {:>10.1} {:>10} {:>10.1} {:>10} {:>9} {:>9} {:>9.3}",
                nb,
                v[0],
                cell(v[2]),
                v[1],
                cell(v[3]),
                pct_s,
                ms,
                fill
            );
        }
    }
}
