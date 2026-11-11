//! The encoder attention strip's kernels (roadmap S4.9g, part 3).
//!
//! ```text
//! cargo run --release -p peregrine-bench --example bidirbench -- [reps] [score]
//! ```
//!
//! A second argument prints only the score-kernel table (the cheap A/B when a shape
//! changes).
//!
//! One text, three query heads on one KV head (GQA 3:1, EmbeddingGemma's shape), a
//! global layer: the table is *per thread*, which is what the engine runs on each
//! worker. Four things are printed:
//!
//! 1. the FMA peak of this core, measured the way S4.9f did — twelve independent
//!    register chains, no memory in the loop — so every rate below has a denominator;
//! 2. the score kernel at every row-block shape, as score pairs per second and as a
//!    fraction of the peak (one score pair is `head_dim` multiply-adds), plus the
//!    old row kernel on the lane panel as the control;
//! 3. the whole attention stage — panel build, scores, max+exp+sum, value mix — with
//!    the time each stage takes and its share, one thread;
//! 4. the same stage on four threads, splitting the strips and the panel's tiles, as
//!    the engine does, with the aggregate rate and its fraction of four times the
//!    single-core peak (this host is shared, so the scaling is not four).

use peregrine_kernels::bidir as k;
use std::time::Instant;

/// The row-block shapes the tiled score kernel instantiates.
const ROWS_TILES: [usize; 5] = [1, 2, 4, 6, 8];

const N_HEAD: usize = 3;
/// EmbeddingGemma's ratio: three query heads on one KV head.
const GQA: usize = 3;

/// This core's FMA peak: twelve independent `ymm` chains, ten rounds of a few thousand
/// iterations (the shape S4.9f measured against).
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
        const ROUNDS: usize = 20_000;
        let macs = (12 * 8 * ROUNDS) as f64;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            std::hint::black_box(unsafe { peak(ROUNDS) });
            best = best.min(t.elapsed().as_secs_f64());
        }
        macs / best / 1e9
    }
    #[cfg(target_arch = "aarch64")]
    {
        #[target_feature(enable = "neon")]
        unsafe fn peak(rounds: usize) -> f32 {
            use std::arch::aarch64::*;
            // Twenty-four chains: on this core twelve are latency-bound, and a
            // different start per chain stops the compiler merging identical ones.
            let mut a = [vdupq_n_f32(0.0); 24];
            for (i, x) in a.iter_mut().enumerate() {
                *x = vdupq_n_f32(1.000_001 + i as f32 * 1.0e-7);
            }
            let b = std::hint::black_box(vdupq_n_f32(0.999_999));
            let c = vdupq_n_f32(1.0e-7);
            for _ in 0..rounds {
                for x in a.iter_mut() {
                    *x = vfmaq_f32(*x, b, c);
                }
            }
            let mut s = vdupq_n_f32(0.0);
            for x in a {
                s = vaddq_f32(s, x);
            }
            let mut l = [0.0f32; 4];
            vst1q_f32(l.as_mut_ptr(), s);
            l.iter().sum()
        }
        const ROUNDS: usize = 20_000;
        let macs = (24 * 4 * ROUNDS) as f64;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            std::hint::black_box(unsafe { peak(ROUNDS) });
            best = best.min(t.elapsed().as_secs_f64());
        }
        macs / best / 1e9
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    0.0
}

/// Nanoseconds per call, the minimum over `reps` (the quietest sample).
fn ns_per(f: &mut dyn FnMut(), reps: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e9
}

/// One text's queries and its one KV head.
struct Text {
    q: Vec<f32>,
    kv: Vec<f32>,
    out: Vec<f32>,
    panel: Vec<f32>,
    row_panel: Vec<f32>,
    scores: Vec<f32>,
    inv: Vec<f32>,
}

impl Text {
    fn new(n: usize, hd: usize) -> Text {
        let mut x = 0x1234_5678u32;
        let mut next = || {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 8) as f32 / 16_777_216.0).mul_add(2.0, -1.0) * 0.01
        };
        let strip = strip_rows(n);
        Text {
            q: (0..n * N_HEAD * hd).map(|_| next()).collect(),
            kv: (0..n * hd).map(|_| next()).collect(),
            out: vec![0.0f32; n * N_HEAD * hd],
            panel: vec![0.0f32; hd * n],
            row_panel: vec![0.0f32; hd * n],
            scores: vec![0.0f32; strip * n],
            inv: vec![0.0f32; k::STRIP_MAX],
        }
    }
}

/// Strip rows for a text of `n` (the engine's rule).
fn strip_rows(n: usize) -> usize {
    (k::STRIP_AREA / n.max(1)).clamp(1, k::STRIP_MAX)
}

/// The score kernel alone over a whole text, strips of `strip` rows, keys in blocks of
/// [`k::SCORE_BLOCK`], at the engine's row block (`rows_tile`).
fn score_pass(t: &mut Text, n: usize, hd: usize, strip: usize, rows_tile: usize, scale: f32) {
    let qd = N_HEAD * hd;
    let mut r0 = 0;
    while r0 < n {
        let rows = strip.min(n - r0);
        for h in 0..N_HEAD {
            let qs = &t.q[r0 * qd + h * hd..];
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + k::SCORE_BLOCK).min(n);
                k::scores_tile_shaped(
                    rows_tile,
                    k::TILE_KEYS,
                    qs,
                    qd,
                    &t.panel,
                    n,
                    hd,
                    rows,
                    j0,
                    j1,
                    scale,
                    &mut t.scores,
                    n,
                );
                j0 = j1;
            }
        }
        r0 += rows;
    }
}

/// The old strip's score kernel on the lane panel: one row at a time.
fn score_pass_row(t: &mut Text, n: usize, hd: usize, strip: usize, scale: f32) {
    let qd = N_HEAD * hd;
    let mut r0 = 0;
    while r0 < n {
        let rows = strip.min(n - r0);
        for h in 0..N_HEAD {
            for r in 0..rows {
                let mut j0 = 0;
                while j0 < n {
                    let j1 = (j0 + k::SCORE_BLOCK).min(n);
                    k::scores_row(
                        &t.q[(r0 + r) * qd + h * hd..],
                        &t.row_panel,
                        n,
                        hd,
                        j0,
                        j1,
                        scale,
                        &mut t.scores[r * n..(r + 1) * n],
                    );
                    j0 = j1;
                }
            }
        }
        r0 += rows;
    }
}

/// The whole attention stage as `blocks/bidir.rs` drives it, one thread: returns
/// `(panel, scores, softmax, mix)` in ns.
#[allow(clippy::too_many_arguments)]
fn attend_pass(
    t: &mut Text,
    n: usize,
    hd: usize,
    strip: usize,
    rows_tile: usize,
    mix_block: usize,
    scale: f32,
    reps: usize,
) -> [f64; 4] {
    let qd = N_HEAD * hd;
    let kvd = hd;
    let mut panel_ns = f64::INFINITY;
    let mut score_ns = 0.0f64;
    let mut soft_ns = 0.0f64;
    let mut mix_ns = 0.0f64;
    for _ in 0..reps {
        panel_ns = panel_ns.min(ns_per(
            &mut || k::key_panel(&t.kv, kvd, 0, hd, n, &mut t.panel),
            reps,
        ));
        let mut r0 = 0;
        while r0 < n {
            let rows = strip.min(n - r0);
            for h in 0..N_HEAD {
                let t0 = Instant::now();
                let qs = &t.q[r0 * qd + h * hd..];
                let mut j0 = 0;
                while j0 < n {
                    let j1 = (j0 + k::SCORE_BLOCK).min(n);
                    k::scores_tile_shaped(
                        rows_tile,
                        k::TILE_KEYS,
                        qs,
                        qd,
                        &t.panel,
                        n,
                        hd,
                        rows,
                        j0,
                        j1,
                        scale,
                        &mut t.scores,
                        n,
                    );
                    j0 = j1;
                }
                score_ns += t0.elapsed().as_secs_f64() * 1e9;
                let t0 = Instant::now();
                for r in 0..rows {
                    let s = &mut t.scores[r * n..(r + 1) * n];
                    let max = k::max_row(s, 0, n);
                    let l = k::exp_row(s, 0, n, max);
                    t.inv[r] = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
                }
                soft_ns += t0.elapsed().as_secs_f64() * 1e9;
                let t0 = Instant::now();
                k::mix_rows_shaped(
                    mix_block,
                    &t.scores,
                    n,
                    r0,
                    rows,
                    n,
                    None,
                    &t.kv,
                    kvd,
                    0,
                    hd,
                    &t.inv,
                    &mut t.out[r0 * qd..],
                    qd,
                    h * hd,
                );
                mix_ns += t0.elapsed().as_secs_f64() * 1e9;
            }
            r0 += rows;
        }
    }
    [
        panel_ns / reps as f64,
        score_ns / reps as f64,
        soft_ns / reps as f64,
        mix_ns / reps as f64,
    ]
}

/// The whole attention stage on `threads` workers, as the driver runs it: the panel is
/// built once (serially here — the engine splits its tiles, which is one to two percent
/// of the stage), then the strips are split across the workers, each with its own score
/// strip and its own output rows. Returns the wall time in ns, the best of `reps`.
#[allow(clippy::too_many_arguments)]
fn attend_pass_threads(
    t: &mut Text,
    n: usize,
    hd: usize,
    strip: usize,
    rows_tile: usize,
    mix_block: usize,
    scale: f32,
    threads: usize,
    reps: usize,
) -> f64 {
    let qd = N_HEAD * hd;
    let kvd = hd;
    let strips = n.div_ceil(strip);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        k::key_panel(&t.kv, kvd, 0, hd, n, &mut t.panel);
        let Text {
            q, kv, panel, out, ..
        } = t;
        let q = &*q;
        let kv = &*kv;
        let panel = &*panel;
        let out = &mut out[..];
        let mut bufs: Vec<(Vec<f32>, Vec<f32>)> = (0..threads)
            .map(|_| (vec![0.0f32; strip * n], vec![0.0f32; strip]))
            .collect();
        let mut outs: Vec<&mut [f32]> = Vec::with_capacity(threads);
        let mut rest = out;
        for w in 0..threads {
            let (wa, wb) = share(strips, w, threads);
            let rows = (wb.saturating_sub(wa) * strip).min(n.saturating_sub(wa * strip));
            let (a, b) = rest.split_at_mut(rows * qd);
            outs.push(a);
            rest = b;
        }
        std::thread::scope(|s| {
            for (w, ((buf, inv), o)) in bufs.iter_mut().zip(outs).enumerate() {
                let (wa, wb) = share(strips, w, threads);
                let buf = buf.as_mut_slice();
                let inv = inv.as_mut_slice();
                s.spawn(move || {
                    for si in wa..wb {
                        let r0 = si * strip;
                        let rows = strip.min(n - r0);
                        for h in 0..N_HEAD {
                            let qs = &q[r0 * qd + h * hd..];
                            let mut j0 = 0;
                            while j0 < n {
                                let j1 = (j0 + k::SCORE_BLOCK).min(n);
                                k::scores_tile_shaped(
                                    rows_tile,
                                    k::TILE_KEYS,
                                    qs,
                                    qd,
                                    panel,
                                    n,
                                    hd,
                                    rows,
                                    j0,
                                    j1,
                                    scale,
                                    buf,
                                    n,
                                );
                                j0 = j1;
                            }
                            for r in 0..rows {
                                let s = &mut buf[r * n..(r + 1) * n];
                                let max = k::max_row(s, 0, n);
                                let l = k::exp_row(s, 0, n, max);
                                inv[r] = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
                            }
                            k::mix_rows_shaped(
                                mix_block,
                                buf,
                                n,
                                r0,
                                rows,
                                n,
                                None,
                                kv,
                                kvd,
                                0,
                                hd,
                                inv,
                                o,
                                qd,
                                h * hd,
                            );
                        }
                    }
                });
            }
        });
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best * 1e9
}

/// `[lo, hi)` of `total` split into `nw` shares, as the pool does.
fn share(total: usize, w: usize, nw: usize) -> (usize, usize) {
    let lo = total * w / nw;
    let hi = total * (w + 1) / nw;
    (lo, hi)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(3);
    let score_only = args.iter().any(|a| a == "score");
    let peak = fma_peak();
    println!(
        "one thread, minimum-of-{reps}; FMA peak (independent vector chains) {peak:.1} GMAC/s"
    );
    println!(
        "one score pair = head_dim MACs; % peak = pairs*head_dim / time / peak. {N_HEAD} query heads on one KV head (GQA {GQA}:1)."
    );

    // Only the engine's shape at the longest lengths: one pass there is seconds.
    let lens: [usize; 4] = [512, 2048, 4096, 8192];

    println!("\n== score kernel only, Mpairs/s per thread / (% of the FMA peak)");
    for &hd in &[64usize, 128, 256] {
        println!("\n-- head_dim {hd}");
        print!("{:>6}", "n");
        for rt in ROWS_TILES {
            print!(" {:>13}", format!("{rt}x16"));
        }
        println!(" {:>13}", "row (old)");
        for &n in &lens {
            let mut t = Text::new(n, hd);
            let strip = strip_rows(n);
            let scale = 1.0 / (hd as f32).sqrt();
            k::key_panel(&t.kv, hd, 0, hd, n, &mut t.panel);
            k::key_panel_rows(&t.kv, hd, 0, hd, n, 0, n, &mut t.row_panel);
            let pairs = (n * n * N_HEAD) as f64;
            let flops = pairs * hd as f64;
            let reps = if n >= 4096 { 1 } else { reps };
            print!("{n:>6}");
            for rt in ROWS_TILES {
                let ns = ns_per(&mut || score_pass(&mut t, n, hd, strip, rt, scale), reps);
                print!(
                    " {:>6.1}/{:>4.0}%",
                    pairs / (ns * 1e-9) / 1e6,
                    100.0 * flops / (ns * 1e-9) / 1e9 / peak
                );
            }
            let ns = ns_per(&mut || score_pass_row(&mut t, n, hd, strip, scale), reps);
            print!(
                " {:>6.1}/{:>4.0}%",
                pairs / (ns * 1e-9) / 1e6,
                100.0 * flops / (ns * 1e-9) / 1e9 / peak
            );
            println!();
        }
    }

    if score_only {
        return;
    }
    println!("\n== the whole attention stage, one thread (engine shape 6x16)");
    for &hd in &[64usize, 128, 256] {
        println!("\n-- head_dim {hd}");
        println!(
            "{:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7}",
            "n", "strip", "Mp/s", "%pk", "ms", "panel", "scores", "maxexp", "mix", "share"
        );
        for &n in &lens {
            let mut t = Text::new(n, hd);
            let strip = strip_rows(n);
            let scale = 1.0 / (hd as f32).sqrt();
            let reps = if n >= 4096 { 1 } else { reps };
            let s = attend_pass(
                &mut t,
                n,
                hd,
                strip,
                k::TILE_ROWS,
                k::MIX_BLOCK,
                scale,
                reps,
            );
            let pairs = (n * n * N_HEAD) as f64;
            let tot = s.iter().sum::<f64>();
            let flops = pairs * hd as f64;
            println!(
                "{n:>6} {strip:>6} {:>8.1} {:>7.0}% {:>8.3} {:>6.0}% {:>6.0}% {:>6.0}% {:>6.0}% {:>6.0}%",
                pairs / (tot * 1e-9) / 1e6,
                100.0 * flops / (tot * 1e-9) / 1e9 / peak,
                tot * 1e-6,
                100.0 * s[0] / tot,
                100.0 * s[1] / tot,
                100.0 * s[2] / tot,
                100.0 * s[3] / tot,
                100.0 * s[1] / tot + 100.0 * s[2] / tot + 100.0 * s[3] / tot,
            );
        }
    }

    println!("\n== the whole attention stage, four threads, one text");
    for &hd in &[64usize, 128, 256] {
        println!("\n-- head_dim {hd}");
        println!(
            "{:>6} {:>6} {:>10} {:>8} {:>9} {:>8} {:>7}",
            "n", "strip", "Mp/s", "%4pk", "ms", "speedup", "load"
        );
        let load = std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|s| s.split_whitespace().next().map(str::to_string))
            .unwrap_or_else(|| "?".into());
        for &n in &lens {
            let mut t = Text::new(n, hd);
            let strip = strip_rows(n);
            let scale = 1.0 / (hd as f32).sqrt();
            let reps = if n >= 4096 { 1 } else { reps };
            let one = attend_pass(
                &mut t,
                n,
                hd,
                strip,
                k::TILE_ROWS,
                k::MIX_BLOCK,
                scale,
                reps,
            )
            .iter()
            .sum::<f64>();
            let four = attend_pass_threads(
                &mut t,
                n,
                hd,
                strip,
                k::TILE_ROWS,
                k::MIX_BLOCK,
                scale,
                4,
                reps,
            );
            let pairs = (n * n * N_HEAD) as f64;
            let flops = pairs * hd as f64;
            println!(
                "{n:>6} {strip:>6} {:>10.1} {:>7.0}% {:>9.3} {:>7.2}x {:>7}",
                pairs / (four * 1e-9) / 1e6,
                100.0 * flops / (four * 1e-9) / 1e9 / (4.0 * peak),
                four * 1e-6,
                one / four,
                load,
            );
        }
    }
}
