//! The encoder attention strip against the loop it replaces (roadmap S4.9g).
//!
//! ```text
//! cargo run --release -p peregrine-bench --example bidirbench -- [head_dim] [threads] [reps]
//! ```
//!
//! One text, three query heads on one KV head (GQA 3:1, EmbeddingGemma's shape),
//! global layer, no window: the old scalar loop (`dot` in key order, the softmax with
//! the C library's `exp`, the value mix in key order — the code that was in
//! `blocks/bidir.rs`) against the strip (`peregrine_kernels::bidir`: panel, 32-key
//! score blocks, `exp8`, four-row value mix). Both on one thread, so the table is
//! scores per second *per thread*; the engine runs the strip over every worker.
//!
//! `head_dim` defaults to 256, EmbeddingGemma's actual head width; 32 and 64 are also
//! useful reference points because they are the BERT family's.

use peregrine_kernels::bidir as k;
use std::time::Instant;

/// The loop as it was: one strictly ordered f32 dot per (row, key), `l += e` in f64,
/// `libm`'s `exp`, and the value mix in key order.
fn old_attention(
    n: usize,
    hd: usize,
    n_head: usize,
    gqa: usize,
    q: &[f32],
    kv: &[f32],
    out: &mut [f32],
) {
    let qd = n_head * hd;
    let kvd = (n_head / gqa) * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    for r in 0..n {
        for h in 0..n_head {
            let kh = h / gqa;
            let mut scores = vec![0.0f32; n];
            let mut max = f32::NEG_INFINITY;
            for (j, s) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[r * qd + h * hd + d] * kv[j * kvd + kh * hd + d];
                }
                *s = dot * scale;
                max = max.max(*s);
            }
            let mut l = 0.0f64;
            for s in scores.iter_mut() {
                let e = (*s - max).exp();
                l += e as f64;
                *s = e;
            }
            let inv = (1.0 / l) as f32;
            let mut acc = vec![0.0f32; hd];
            for (j, &e) in scores.iter().enumerate() {
                for d in 0..hd {
                    acc[d] += e * kv[j * kvd + kh * hd + d];
                }
            }
            for d in 0..hd {
                out[r * qd + h * hd + d] = acc[d] * inv;
            }
        }
    }
}

/// The strip: the head's keys once, then strips of query rows scored against 32-key
/// panel blocks, soft-maxed and mixed while hot. Single-threaded, the schedule
/// `blocks/bidir.rs` runs on each worker.
fn strip_attention(
    n: usize,
    hd: usize,
    n_head: usize,
    gqa: usize,
    q: &[f32],
    kv: &[f32],
    out: &mut [f32],
) {
    let qd = n_head * hd;
    let n_kv = n_head / gqa;
    let kvd = n_kv * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let strip = (k::STRIP_AREA / n).clamp(1, k::STRIP_MAX);
    let mut panel = vec![0.0f32; hd * n];
    let mut scores = vec![0.0f32; strip * n];
    let mut inv = vec![0.0f32; strip];
    for kh in 0..n_kv {
        k::key_panel(kv, kvd, kh * hd, hd, n, &mut panel);
        let mut r0 = 0;
        while r0 < n {
            let rows = strip.min(n - r0);
            for h in kh * gqa..(kh + 1) * gqa {
                // Key blocks, as `blocks/bidir.rs` takes them: the block's panel slice
                // stays in cache while the strip's rows walk it.
                let mut j0 = 0;
                while j0 < n {
                    let j1 = (j0 + k::SCORE_BLOCK).min(n);
                    for r in 0..rows {
                        let s = &mut scores[r * n..(r + 1) * n];
                        k::scores_row(
                            &q[(r0 + r) * qd + h * hd..],
                            &panel,
                            n,
                            hd,
                            j0,
                            j1,
                            scale,
                            s,
                        );
                    }
                    j0 = j1;
                }
                for r in 0..rows {
                    let s = &mut scores[r * n..(r + 1) * n];
                    let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let l = k::exp_row(s, 0, n, max);
                    inv[r] = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
                }
                k::mix_rows(
                    &scores,
                    n,
                    r0,
                    rows,
                    n,
                    None,
                    kv,
                    kvd,
                    kh * hd,
                    hd,
                    &inv,
                    &mut out[r0 * qd..],
                    qd,
                    h * hd,
                );
            }
            r0 += rows;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let hd: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(256);
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let n_head = 3;
    let gqa = 3;
    let qd = n_head * hd;
    let kvd = hd;
    println!("head_dim {hd}, {n_head} query heads on one KV head, one thread, median of {reps}");
    println!(
        "{:>6} {:>12} {:>12} {:>8} {:>12} {:>12} {:>8}",
        "n", "old pairs/s", "old GFLOP/s", "old ms", "new pairs/s", "new GFLOP/s", "new ms"
    );
    for n in [16usize, 128, 512, 2048] {
        let q: Vec<f32> = (0..n * qd)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.002)
            .collect();
        let kv: Vec<f32> = (0..n * kvd)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.002)
            .collect();
        let mut a = vec![0.0f32; n * qd];
        let mut b = vec![0.0f32; n * qd];
        let mut old_ms = f64::INFINITY;
        let mut new_ms = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            old_attention(n, hd, n_head, gqa, &q, &kv, &mut a);
            old_ms = old_ms.min(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            strip_attention(n, hd, n_head, gqa, &q, &kv, &mut b);
            new_ms = new_ms.min(t.elapsed().as_secs_f64() * 1e3);
        }
        // Both compute the same attention (the arithmetic differs in the last bits):
        // a large disagreement would mean the bench is timing two different things.
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "the two loops disagree by {worst} at n = {n}");
        let pairs = (n * n * n_head) as f64;
        let flops = pairs * hd as f64 * 4.0; // dot + mix, two ops each
        println!(
            "{n:>6} {:>12.0} {:>12.2} {:>8.3} {:>12.0} {:>12.2} {:>8.3}   (max |old-new| {worst:.1e})",
            pairs / (old_ms * 1e-3),
            1e-9 * flops / (old_ms * 1e-3),
            old_ms,
            pairs / (new_ms * 1e-3),
            1e-9 * flops / (new_ms * 1e-3),
            new_ms
        );
    }
}
