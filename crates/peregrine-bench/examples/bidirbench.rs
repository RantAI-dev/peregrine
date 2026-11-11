//! The encoder attention strip's score tile against the row kernel it replaces
//! (roadmap S4.9g, part 2).
//!
//! ```text
//! cargo run --release -p peregrine-bench --example bidirbench -- [reps] [score]
//! ```
//!
//! A second argument prints only the score-kernel table (the cheap A/B when a shape
//! changes).
//!
//! One text, three query heads on one KV head (GQA 3:1, EmbeddingGemma's shape), a
//! global layer, one thread: the table is *per thread*, which is what the engine runs
//! on each worker. Three things are printed:
//!
//! 1. the FMA peak of this core, measured the way S4.9f did — twelve independent
//!    register chains, no memory in the loop — so every rate below has a denominator;
//! 2. the score kernel alone at every tile shape, as score pairs per second and as a
//!    fraction of the peak (one score pair is `head_dim` multiply-adds);
//! 3. the whole attention stage — panel build, scores, max+exp+sum, value mix — the
//!    old strip (one row at a time, four-row mix, scalar max fold) against the new
//!    tile, with the time each stage takes on both sides.

use peregrine_kernels::bidir as k;
use std::time::Instant;

/// The tile shapes the kernel instantiates, from one row to two rows of keys.
const SHAPES: [(usize, usize); 8] = [
    (1, 8),
    (1, 16),
    (2, 16),
    (4, 16),
    (6, 16),
    (8, 16),
    (6, 24),
    (4, 32),
];

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
    #[cfg(not(target_arch = "x86_64"))]
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
        Text {
            q: (0..n * N_HEAD * hd).map(|_| next()).collect(),
            kv: (0..n * hd).map(|_| next()).collect(),
            out: vec![0.0f32; n * N_HEAD * hd],
            panel: vec![0.0f32; hd * n],
            scores: vec![0.0f32; 32 * n],
            inv: vec![0.0f32; 32],
        }
    }
}

/// Strip rows for a text of `n` (the engine's rule).
fn strip_rows(n: usize) -> usize {
    (k::STRIP_AREA / n.max(1)).clamp(1, k::STRIP_MAX)
}

/// The score kernel alone over a whole text: `shape` rows × keys at a time, strips of
/// `strip` rows, keys in blocks of [`k::SCORE_BLOCK`].
fn score_pass(t: &mut Text, n: usize, hd: usize, strip: usize, shape: (usize, usize), scale: f32) {
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
                    shape.0,
                    shape.1,
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

/// The old strip's score kernel: one row at a time, [`k::scores_row`].
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
                        &t.panel,
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

/// The whole attention stage, as `blocks/bidir.rs` drives it. `tile` of `None` is the
/// old strip: one row at a time, four-row mix, scalar max fold; `Some(shape)` is the
/// tile, six-row mix, vector max. Returns `(panel, scores, softmax, mix)` in ns.
#[allow(clippy::too_many_arguments)]
fn attend_pass(
    t: &mut Text,
    n: usize,
    hd: usize,
    strip: usize,
    tile: Option<(usize, usize)>,
    mix_block: usize,
    scale: f32,
    reps: usize,
) -> [f64; 4] {
    let qd = N_HEAD * hd;
    let kvd = hd;
    let mut stage = [0.0f64; 4];
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
                match tile {
                    None => {
                        for r in 0..rows {
                            let mut j0 = 0;
                            while j0 < n {
                                let j1 = (j0 + k::SCORE_BLOCK).min(n);
                                k::scores_row(
                                    &t.q[(r0 + r) * qd + h * hd..],
                                    &t.panel,
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
                    Some(shape) => {
                        let qs = &t.q[r0 * qd + h * hd..];
                        let mut j0 = 0;
                        while j0 < n {
                            let j1 = (j0 + k::SCORE_BLOCK).min(n);
                            k::scores_tile_shaped(
                                shape.0,
                                shape.1,
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
                }
                score_ns += t0.elapsed().as_secs_f64() * 1e9;
                let t0 = Instant::now();
                for r in 0..rows {
                    let s = &mut t.scores[r * n..(r + 1) * n];
                    let max = match tile {
                        None => s.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                        Some(_) => k::max_row(s, 0, n),
                    };
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
    stage[0] = panel_ns / reps as f64;
    stage[1] = score_ns / reps as f64;
    stage[2] = soft_ns / reps as f64;
    stage[3] = mix_ns / reps as f64;
    stage
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(3);
    let score_only = args.iter().any(|a| a == "score");
    let peak = fma_peak();
    println!("one thread, median-of-{reps} minimums; FMA peak (12 ymm chains) {peak:.1} GMAC/s");
    println!(
        "one score pair = head_dim MACs; % peak = pairs*head_dim / time / peak. {N_HEAD} query heads on one KV head (GQA {GQA}:1)."
    );

    println!("\n== score kernel only, Mpairs/s per thread / (% of the FMA peak)");
    for &hd in &[32usize, 64, 256] {
        println!("\n-- head_dim {hd}");
        print!("{:>6}", "n");
        for (rt, kt) in SHAPES {
            print!(" {:>13}", format!("{rt}x{kt}"));
        }
        println!(" {:>13}", "row (old)");
        for &n in &[16usize, 128, 512, 2048] {
            let mut t = Text::new(n, hd);
            let strip = strip_rows(n);
            let scale = 1.0 / (hd as f32).sqrt();
            k::key_panel(&t.kv, hd, 0, hd, n, &mut t.panel);
            let pairs = (n * n * N_HEAD) as f64;
            let flops = pairs * hd as f64;
            print!("{n:>6}");
            let mut row = Vec::new();
            for (rt, kt) in SHAPES {
                let ns = ns_per(
                    &mut || score_pass(&mut t, n, hd, strip, (rt, kt), scale),
                    reps,
                );
                row.push(Some(ns));
            }
            row.push(Some(ns_per(
                &mut || score_pass_row(&mut t, n, hd, strip, scale),
                reps,
            )));
            for ns in row {
                let ns = ns.unwrap();
                print!(
                    " {:>6.1}/{:>4.0}%",
                    pairs / (ns * 1e-9) / 1e6,
                    100.0 * flops / (ns * 1e-9) / 1e9 / peak
                );
            }
            println!();
        }
    }

    if score_only {
        return;
    }
    println!("\n== the whole attention stage: old strip vs new tile (one thread)");
    for &hd in &[32usize, 64, 256] {
        println!("\n-- head_dim {hd}");
        println!(
            "{:>6} {:>6} {:>9} {:>7} {:>9} {:>7} {:>8} {:>8} {:>8} {:>7} {:>6}",
            "n",
            "strip",
            "old Mp/s",
            "old %pk",
            "new Mp/s",
            "new %pk",
            "old ms",
            "new ms",
            "mix12 ms",
            "spd",
            "old/new"
        );
        for &n in &[16usize, 128, 512, 2048] {
            let mut t = Text::new(n, hd);
            let strip = strip_rows(n);
            let scale = 1.0 / (hd as f32).sqrt();
            let old = attend_pass(&mut t, n, hd, strip, None, 4, scale, reps);
            let new = attend_pass(
                &mut t,
                n,
                hd,
                strip,
                Some((k::TILE_ROWS, k::TILE_KEYS)),
                k::MIX_BLOCK,
                scale,
                reps,
            );
            let wide = attend_pass(
                &mut t,
                n,
                hd,
                strip,
                Some((k::TILE_ROWS, k::TILE_KEYS)),
                12,
                scale,
                reps,
            );
            let pairs = (n * n * N_HEAD) as f64;
            let tot = |s: &[f64; 4]| s.iter().sum::<f64>();
            let flops = pairs * hd as f64;
            println!(
                "{n:>6} {strip:>7} {:>9.1} {:>7.0}% {:>9.1} {:>7.0}% {:>8.3} {:>8.3} {:>8.3} {:>7.2} {:>5.0}/{:<4.0}",
                pairs / (tot(&old) * 1e-9) / 1e6,
                100.0 * flops / (tot(&old) * 1e-9) / 1e9 / peak,
                pairs / (tot(&new) * 1e-9) / 1e6,
                100.0 * flops / (tot(&new) * 1e-9) / 1e9 / peak,
                tot(&old) * 1e-6,
                tot(&new) * 1e-6,
                tot(&wide) * 1e-6,
                tot(&old) / tot(&new),
                100.0 * old[1] / tot(&old),
                100.0 * new[1] / tot(&new),
            );
        }
    }

    println!("\n== where the new tile's time goes (share of the stage)");
    for &hd in &[32usize, 64, 256] {
        println!("-- head_dim {hd}");
        println!(
            "{:>6} {:>9} {:>9} {:>9} {:>9}",
            "n", "panel", "scores", "max+exp", "mix"
        );
        for &n in &[16usize, 128, 512, 2048] {
            let mut t = Text::new(n, hd);
            let strip = strip_rows(n);
            let scale = 1.0 / (hd as f32).sqrt();
            let s = attend_pass(
                &mut t,
                n,
                hd,
                strip,
                Some((k::TILE_ROWS, k::TILE_KEYS)),
                k::MIX_BLOCK,
                scale,
                reps,
            );
            let tot = s.iter().sum::<f64>();
            println!(
                "{n:>6} {:>8.0}% {:>8.0}% {:>8.0}% {:>8.0}%",
                100.0 * s[0] / tot,
                100.0 * s[1] / tot,
                100.0 * s[2] / tot,
                100.0 * s[3] / tot
            );
        }
    }
}
