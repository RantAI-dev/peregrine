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
//! Per KV head and per text the head's keys are transposed once into a panel (16-key
//! key-major tiles on x86_64 and aarch64, keys along the lanes elsewhere — a query head
//! of a GQA group shares it), and then the text's query rows are taken in strips: a
//! strip's scores are `strip × n` floats, kept in the worker's own buffer, and the
//! value mix reads the pass's values in place. A strip is scored, soft-maxed and mixed
//! while it is hot; no `n × n` matrix exists for a whole text.
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

/// Query rows per strip: the widest strip whose scores fit the per-worker budget
/// ([`k::STRIP_AREA`]), at most [`k::STRIP_MAX`]. A 512-token text takes 32 rows
/// (a 64 KiB score buffer, L2); a 2 000-token text 16 (128 KiB); a 16-token text 16
/// (1 KiB, and one strip for the whole text — a short text is not cut up).
///
/// A pass with fewer strips than workers halves the strip down to [`STRIP_MIN`] so one
/// long text still fills the pool; the work is then split by rows anyway, so a text is
/// never fanned out pointlessly. The strip size changes only how much of a head's panel
/// is re-read, never a row's bits (the kernel tests hold strips of 1, 7, 16, 32 and `n`
/// to the same output).
pub const STRIP_MIN: usize = 16;

/// Rows per strip for a pass whose longest text is `max_len`, before the fan-out rule.
pub fn strip_rows(max_len: usize) -> usize {
    (k::STRIP_AREA / max_len.max(1)).clamp(1, k::STRIP_MAX)
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
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    // The budget's strip, then: a pass with fewer strips than workers halves it (never
    // below `STRIP_MIN`) so one long text still fills the pool.
    let mut strip = strip_rows(max_len);
    if parallel {
        while strip > STRIP_MIN {
            let units: usize = spans.iter().map(|s| s.len.div_ceil(strip)).sum();
            if units >= pool.width() {
                break;
            }
            strip /= 2;
        }
    }
    let area = strip * max_len;
    debug_assert!(area <= k::STRIP_AREA);
    // The panel build is a copy, not arithmetic: below a few tens of KiB of it a second
    // fork-join costs more than it saves (a 16-token text's panel is 4 Ki copies).
    let panel_par = parallel && nb * hd >= 32 * 1024;
    let t0 = std::time::Instant::now();
    let panel = Disjoint::new(&mut scratch.panel);
    let strips_buf = Disjoint::new(&mut scratch.strips);

    for kh in 0..c.n_kv_head {
        // ---- the head's keys, once, for every query head that reads them ----
        if panel_par {
            pool.run(|wk, nw| {
                build_panel_head(spans, wk, nw, k, kvd, kh * hd, hd, &panel);
            });
            stats.barriers += 1;
        } else {
            build_panel_head(spans, 0, 1, k, kvd, kh * hd, hd, &panel);
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
                            // Scores in blocks of keys: the block's panel slice stays
                            // in cache while every tile of the strip walks it. Rows of
                            // a strip have overlapping windows, so a block is the
                            // *union* of their bands, and the kernel scores the whole
                            // block for every row: the keys outside a particular row's
                            // band are written to that row's buffer and never read
                            // (the max, the exponentials and the mix take the row's own
                            // `band`). On a global layer the union is the text.
                            let u_lo = k::band(local0, n, half_window).0;
                            let u_hi = k::band(local0 + cnt - 1, n, half_window).1;
                            let qs = &q[(span.row0 + local0) * qd + h * hd..];
                            let mut j0 = u_lo;
                            while j0 < u_hi {
                                let j1 = (j0 + k::SCORE_BLOCK).min(u_hi);
                                k::scores_tile(qs, qd, p, n, hd, cnt, j0, j1, scale, buf, n);
                                j0 = j1;
                            }
                            // Then the row's maximum, weights and denominator, once per
                            // row over its exact band.
                            for r in 0..cnt {
                                let (lo, hi) = k::band(local0 + r, n, half_window);
                                let s = &mut buf[r * n..(r + 1) * n];
                                let max = k::max_row(s, lo, hi);
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

/// Transpose the keys of the head's texts into the panel, split across the workers.
///
/// On x86_64 and aarch64 the panel is key-major (S4.9g part 3): a text's panel is a
/// list of [`k::PANEL_TILE`]-key tiles, each `head_dim × 16` floats, followed by one
/// row-major tile for the `len % 16` keys that do not fill one. The fork-join splits
/// each text's *tiles*, so every tile is written by one worker and the copy needs no
/// other lock. Elsewhere the lane panel of part 2 is built in 16×16 tiles, split by key
/// rows.
#[allow(clippy::too_many_arguments)]
fn build_panel_head(
    spans: &[TextSpan],
    wk: usize,
    nw: usize,
    keys: &[f32],
    kvd: usize,
    base: usize,
    hd: usize,
    panel: &Disjoint<'_, f32>,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        for span in spans {
            let n = span.len;
            let nt = k::panel_tiles(n);
            let (t0, t1) = crate::pool::share(nt, wk, nw);
            for t in t0..t1 {
                let (off, len) = k::panel_tile_span(n, hd, t);
                // SAFETY: tile `t` of this text is this worker's own share of the panel:
                // rows are `span.row0 * hd` and the tiles of a text do not overlap.
                let dst = unsafe { panel.chunk(span.row0 * hd + off, len) };
                k::key_panel_tile(&keys[span.row0 * kvd..], kvd, base, hd, n, t, dst);
            }
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        const T: usize = 16;
        let nb: usize = spans.iter().map(|s| s.len).sum();
        let (rlo, rhi) = crate::pool::share(nb, wk, nw);
        for span in spans {
            let j0 = rlo.max(span.row0);
            let j1 = rhi.min(span.row0 + span.len);
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
                        // SAFETY: `off..off + (jb - ja)` is this worker's own slice of
                        // the panel: rows are its disjoint share and `d` is unique.
                        let dst = unsafe { panel.chunk(off, jb - ja) };
                        for (t, o) in dst.iter_mut().enumerate() {
                            *o = keys[(span.row0 + ja + t) * kvd + base + d];
                        }
                    }
                    ka = kb;
                }
                ja = jb;
            }
        }
    }
}
