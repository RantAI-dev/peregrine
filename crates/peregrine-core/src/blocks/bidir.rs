//! Bidirectional attention over whole texts (design §2.2, roadmap S4.4; strip S4.9g).
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
//!
//! # The strip (S4.9g)
//!
//! The kernels are `peregrine_kernels::bidir`; this file is the schedule around them.
//! Per KV head and per text the head's keys are transposed once into a panel
//! (`[head_dim][n]`, keys along the lanes — a query head of a GQA group shares it),
//! and then the text's query rows are taken in strips: a strip's scores are
//! `strip × n` floats, kept in the worker's own buffer, and the value mix reads the
//! pass's values in place. A strip is scored, soft-maxed and mixed while it is hot;
//! no `n × n` matrix exists for a whole text.
//!
//! Parallelism is over `(text, strip)` units, one worker at a time per unit; the panel
//! build is a fork-join of its own per KV head. A unit's rows are its own output rows,
//! so a row's bits never depend on the strip it fell in, the strip size, the thread or
//! the text's neighbours (the kernel tests pin all four).

use crate::blocks::attn::BidirScratch;
use crate::blocks::par::{for_row_chunks, Disjoint, Sched};
use crate::blocks::ROW_PARALLEL;
use crate::model::{Config, StepStats};
use crate::pool::Pool;
use peregrine_kernels::bidir as k;

/// The rows of one text in the pass, and where its rows begin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextSpan {
    /// The text's first row in the batch.
    pub row0: usize,
    /// The text's length in tokens.
    pub len: usize,
}

/// Query rows per strip: the largest strip whose scores fit the per-worker budget,
/// capped at [`k::STRIP_MAX`]. A 512-token text takes 32 rows (a 64 KiB score buffer,
/// L2); a 2 000-token text 16 (128 KiB); a 16-token text 32 (2 KiB).
///
/// When the pass has fewer strips than workers the strip is halved down to 1 row, so a
/// single long text still fills the pool. The strip size changes only how much of a
/// head's panel is re-read, never a row's bits (the kernel tests hold strips of 1, 7,
/// 16, 32 and `n` to the same output).
pub fn strip_rows(max_len: usize, strips_wanted: usize, width: usize) -> usize {
    let mut strip = (k::STRIP_AREA / max_len.max(1)).clamp(1, k::STRIP_MAX);
    while strip > 1 && strips_wanted < width {
        strip /= 2;
    }
    strip
}

/// One layer's bidirectional attention over every text of the pass.
///
/// `q`, `k` and `v` hold one row per token of the concatenated batch, at the model's
/// `q_dim` / `kv_dim` strides. `out` has `q_dim` per row. `half_window` is the
/// symmetric sliding window (`Some(n_swa / 2)`) or `None` for a global layer. `scale`
/// multiplies each score; the families fold `1/sqrt(head_dim)` into the queries
/// instead and pass `1.0`, as llama.cpp does.
///
/// The arithmetic is the one `peregrine_kernels::bidir` defines: a score is one
/// sequential FMA chain over the head dimension, the exponentials are `exp8` /
/// `exp_scalar`, the softmax denominator is an `f64` sum in ascending key order and the
/// value mix accumulates in ascending key order too. A row's bits are the same at any
/// thread count, in any batch and on any tier.
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
    scratch: &mut BidirScratch,
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
    let max_len = spans.iter().map(|s| s.len).max().unwrap_or(0);
    // The strip the budget allows, then the count of strips it makes of this pass: a
    // pass with fewer strips than workers halves the strip so the pool is used.
    let widest = (k::STRIP_AREA / max_len.max(1)).clamp(1, k::STRIP_MAX);
    let wanted = spans.iter().map(|s| s.len.div_ceil(widest)).sum::<usize>();
    let strip = strip_rows(max_len, wanted, pool.width());
    let area = strip * max_len;
    debug_assert!(area <= k::STRIP_AREA);
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    let t0 = std::time::Instant::now();
    let panel = Disjoint::new(&mut scratch.panel);
    let strips_buf = Disjoint::new(&mut scratch.strips);

    for kh in 0..c.n_kv_head {
        // ---- the head's keys, once, for every query head that reads them ----
        if parallel {
            pool.run(|wk, nw| {
                let (rlo, rhi) = crate::pool::share(nb, wk, nw);
                build_panels(spans, rlo..rhi, k, kvd, kh * hd, hd, &panel);
            });
            stats.barriers += 1;
        } else {
            build_panels(spans, 0..nb, k, kvd, kh * hd, hd, &panel);
        }

        // ---- strips of query rows ----
        let h_lo = kh * gqa;
        let h_hi = h_lo + gqa;
        for_row_chunks(
            pool,
            parallel,
            out,
            qd,
            nb,
            Sched::Static,
            |wk, rows, out_chunk| {
                // SAFETY: worker `wk` owns slot `wk` for the whole call: its strip's
                // scores first, then the strip's softmax denominators.
                let slot = unsafe { strips_buf.chunk(wk * k::STRIP_SLOT, k::STRIP_SLOT) };
                let (buf, inv) = slot.split_at_mut(area);
                let mut si = spans
                    .iter()
                    .position(|s| rows.start >= s.row0 && rows.start < s.row0 + s.len)
                    .expect("every batch row belongs to a text");
                let mut g = rows.start;
                while g < rows.end {
                    while spans[si].row0 + spans[si].len <= g {
                        si += 1;
                    }
                    let span = spans[si];
                    let end = rows.end.min(span.row0 + span.len);
                    let n = span.len;
                    // SAFETY: the panel is written only in the fork-join above, before
                    // this one, and read here by every worker.
                    let p = unsafe { panel.slice_ref(span.row0 * hd, n * hd) };
                    let v_text = &v[span.row0 * kvd..(span.row0 + span.len) * kvd];
                    let mut g0 = g;
                    while g0 < end {
                        let cnt = strip.min(end - g0);
                        let local0 = g0 - span.row0;
                        let rows_out = &mut out_chunk[(g0 - rows.start) * qd..];
                        for h in h_lo..h_hi {
                            // scores, softmax and denominator, a strip at a time
                            for r in 0..cnt {
                                let i = local0 + r;
                                let (lo, hi) = k::band(i, n, half_window);
                                let s = &mut buf[r * n..(r + 1) * n];
                                k::scores_row(
                                    &q[(span.row0 + i) * qd + h * hd..],
                                    p,
                                    n,
                                    hd,
                                    lo,
                                    hi,
                                    scale,
                                    s,
                                );
                                let max =
                                    s[lo..hi].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                                let l = k::exp_row(s, lo, hi, max);
                                inv[r] = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
                            }
                            k::mix_rows(
                                buf,
                                n,
                                local0,
                                cnt,
                                n,
                                half_window,
                                v_text,
                                kvd,
                                kh * hd,
                                hd,
                                inv,
                                rows_out,
                                qd,
                                h * hd,
                            );
                        }
                        g0 += cnt;
                    }
                    g = end;
                }
            },
        );
        if parallel {
            stats.barriers += 1;
        }
    }

    stats.ns_attn += t0.elapsed().as_nanos() as u64;
    // The arithmetic, counted rather than estimated: every pair inside a row's window
    // costs a dot product and a value mix, once per query head. The band is exact, so
    // the count is what the kernels do (less the vector tails).
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

/// Transpose the keys of every text whose rows fall in `rows` into its panel region.
///
/// `panel[row0 * hd + d * len + j] = k[(row0 + j) * kvd + base + d]`. Tiles of 16
/// keys × 16 head dimensions: each source line then serves sixteen head dimensions and
/// each destination line is written whole.
fn build_panels(
    spans: &[TextSpan],
    rows: std::ops::Range<usize>,
    k: &[f32],
    kvd: usize,
    base: usize,
    hd: usize,
    panel: &Disjoint<'_, f32>,
) {
    const T: usize = 16;
    for span in spans {
        let j0 = rows.start.max(span.row0);
        let j1 = rows.end.min(span.row0 + span.len);
        if j0 >= j1 {
            continue;
        }
        let n = span.len;
        let mut ja = j0 - span.row0;
        let jb_end = j1 - span.row0;
        while ja < jb_end {
            let jb = (ja + T).min(jb_end);
            let mut ka = 0;
            while ka < hd {
                let kb = (ka + T).min(hd);
                for d in ka..kb {
                    let off = span.row0 * hd + d * n + ja;
                    // SAFETY: `off..off + (jb - ja)` is this worker's own slice of the
                    // panel: rows `rows` are its disjoint share and `d` is unique.
                    let dst = unsafe { panel.chunk(off, jb - ja) };
                    for (t, o) in dst.iter_mut().enumerate() {
                        *o = k[(span.row0 + ja + t) * kvd + base + d];
                    }
                }
                ka = kb;
            }
            ja = jb;
        }
    }
}
