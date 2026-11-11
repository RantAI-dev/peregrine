//! Bidirectional attention over whole texts (design §2.2, roadmap S4.4).
//!
//! An encoder's attention is one `n × n` block per text: every row of a text scores
//! against every position of that text, under the family's mask rule. There is no KV
//! cache: the keys and values this layer produced are read directly, in f32, and
//! nothing is rounded to f16. Texts are ragged — the batch is a list of rows with the
//! text boundaries beside it — so a text's arithmetic depends only on its own length,
//! never on what else shares the pass (design §2.3).
//!
//! The mask rule transcribed from llama.cpp v0.9.0 (`src/llama-hparams.h`,
//! `is_masked_swa`): a sliding-window layer masks a key when `|p1 - p0| > n_swa / 2`
//! (symmetric window, integer division); a global layer masks nothing. `n_swa` is the
//! file's `attention.sliding_window`.

use crate::blocks::par::{for_row_chunks, Disjoint, Sched};
use crate::blocks::ROW_PARALLEL;
use crate::model::{Config, StepStats};
use crate::pool::Pool;

/// The rows of one text in the pass, and where its rows begin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextSpan {
    /// The text's first row in the batch.
    pub row0: usize,
    /// The text's length in tokens.
    pub len: usize,
}

/// One row of bidirectional attention: `out[row][h][d] = softmax(q·kᵀ·scale)·v`.
///
/// `q`, `k` and `v` hold one row per token of the concatenated batch, at the model's
/// `q_dim` / `kv_dim` strides. `out` has `q_dim` per row. `half_window` is the
/// symmetric sliding window (`Some(n_swa / 2)`) or `None` for a global layer.
///
/// The score, softmax and value accumulation are f32 in a fixed order — key `0..len`,
/// head dimension `0..head_dim` — so a row's bits are the same at any thread count and
/// in any batch. The sum of exponentials is the one f64 accumulator here, because the
/// reference forward in the tests divides by it in f64; both sides use the same rule.
#[allow(clippy::too_many_arguments)]
pub fn attend_texts(
    cfg: &Config,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    spans: &[TextSpan],
    half_window: Option<usize>,
    scale: f32,
    // One score row per worker (`AttnScratch::bidir_scores`).
    scores: &mut [f32],
    pool: &Pool,
    stats: &mut StepStats,
) {
    let c = cfg;
    let hd = c.head_dim;
    let qd = c.q_dim();
    let kvd = c.kv_dim();
    let gqa = c.gqa();
    let nb = spans.iter().map(|s| s.len).sum::<usize>();
    if nb == 0 {
        return;
    }
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    let max_len = spans.iter().map(|s| s.len).max().unwrap_or(0);
    let t0 = std::time::Instant::now();
    let score_buf = Disjoint::new(scores);
    let body = |wk: usize, rows: std::ops::Range<usize>, out_chunk: &mut [f32]| {
        // The spans are in order and contiguous, and `rows` is one contiguous range:
        // advance a cursor instead of building a row -> text table per layer.
        let first = rows.start;
        let mut si = spans
            .iter()
            .position(|s| first >= s.row0 && first < s.row0 + s.len)
            .expect("every batch row belongs to a text");
        let mut span = spans[si];
        // SAFETY: worker `wk` alone uses score row `wk` (the pool has as many workers as
        // `AttnScratch` has score rows).
        let scores = unsafe { score_buf.chunk(wk * max_len, max_len) };
        for (i, gr) in rows.enumerate() {
            while gr >= span.row0 + span.len {
                si += 1;
                span = spans[si];
            }
            let local = gr - span.row0;
            let o_row = &mut out_chunk[i * qd..(i + 1) * qd];
            for h in 0..c.n_head {
                let kh = h / gqa;
                let q_head = &q[gr * qd + h * hd..gr * qd + (h + 1) * hd];
                let base = span.row0 * kvd + kh * hd;
                let mut max = f32::NEG_INFINITY;
                for j in 0..span.len {
                    let s = if half_window.is_some_and(|half| local.abs_diff(j) > half) {
                        f32::NEG_INFINITY
                    } else {
                        let k_head = &k[base + j * kvd..base + j * kvd + hd];
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += q_head[d] * k_head[d];
                        }
                        dot * scale
                    };
                    scores[j] = s;
                    if s > max {
                        max = s;
                    }
                }
                let mut l = 0.0f64;
                let o_head = &mut o_row[h * hd..(h + 1) * hd];
                o_head.fill(0.0);
                for (j, &s) in scores[..span.len].iter().enumerate() {
                    if s == f32::NEG_INFINITY {
                        continue;
                    }
                    let e = (s - max).exp() as f64;
                    l += e;
                    let v_head = &v[base + j * kvd..base + j * kvd + hd];
                    let e = e as f32;
                    for d in 0..hd {
                        o_head[d] += e * v_head[d];
                    }
                }
                let inv = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
                for x in o_head.iter_mut() {
                    *x *= inv;
                }
            }
        }
    };
    for_row_chunks(
        pool,
        parallel,
        out,
        qd,
        nb,
        Sched::Static,
        |wk, rows, out_chunk| body(wk, rows, out_chunk),
    );
    if parallel {
        stats.barriers += 1;
    }
    stats.ns_attn += t0.elapsed().as_nanos() as u64;
    // The arithmetic, counted rather than estimated: every allowed (query, key) pair
    // costs a dot product and a value mix, once per query head. A mask skips the pairs
    // it masks, as llama.cpp does.
    for s in spans {
        let n = s.len as u64;
        let pairs = match half_window {
            // |i - j| <= half: n + 2 * sum_{d=1..half} (n - d), when the window is
            // narrower than the text; otherwise every pair.
            Some(half) => {
                let h = (half as u64).min(n.saturating_sub(1));
                n * n - (n - h - 1) * (n - h)
            }
            None => n * n,
        };
        stats.macs += pairs * (c.n_head as u64) * (hd as u64) * 2;
    }
}
