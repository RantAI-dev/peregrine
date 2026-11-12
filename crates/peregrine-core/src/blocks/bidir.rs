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
/// symmetric sliding window (`Some(n_swa / 2)`) or `None` for a global layer. `alibi`
/// is the per-query-head slope of an additive `slope · (-|i - j|)` score bias
/// ([`peregrine_kernels::bidir::alibi_slopes`]), or `None` for a family without one.
/// `scale` multiplies each score. Most families fold `1/sqrt(head_dim)` into the
/// queries instead and pass `1.0`; `jina-bert-v2` passes the scale so that ALiBi's
/// bias is added after it, which is llama.cpp's order.
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
    alibi: Option<&[f32]>,
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
    if let Some(slopes) = alibi {
        // Checked here, not only in the kernel: a short slope list would make a head
        // read another model's value (or panic inside the strip).
        assert!(
            slopes.len() >= c.n_head,
            "ALiBi needs a slope per query head ({}), got {}",
            c.n_head,
            slopes.len()
        );
    }
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
                            // row over its exact band. A global layer's rows share their
                            // band, so four of them take one batched exponential and
                            // four interleaved `f64` denominators (S4.9r): each row keeps
                            // its own ascending fold, and the four chains keep the add
                            // units busy instead of waiting on one chain's latency.
                            // ALiBi first, for every row of the strip: llama.cpp's
                            // soft-max scales the score, then adds
                            // `slope · (−|p_q − p_k|)`, before the maximum — so the
                            // bias is in place before either path below reads a row.
                            if let Some(slopes) = alibi {
                                for ra in 0..cnt {
                                    let (lo, hi) = k::band(local0 + ra, n, half_window);
                                    let s = &mut buf[ra * n..(ra + 1) * n];
                                    k::alibi_row(s, lo, hi, local0 + ra, slopes[h]);
                                }
                            }
                            let mut r = 0usize;
                            if half_window.is_none() {
                                while r + 4 <= cnt {
                                    let mut mx = [0.0f32; 4];
                                    for g in 0..4 {
                                        mx[g] =
                                            k::max_row(&buf[(r + g) * n..(r + g + 1) * n], 0, n);
                                    }
                                    let mut sums = [0.0f64; 4];
                                    k::exp_rows4_sums(&mut buf[r * n..], n, n, mx, &mut sums);
                                    for g in 0..4 {
                                        inv[r + g] = if sums[g] > 0.0 {
                                            (1.0 / sums[g]) as f32
                                        } else {
                                            0.0
                                        };
                                    }
                                    r += 4;
                                }
                            }
                            for rr in r..cnt {
                                let (lo, hi) = k::band(local0 + rr, n, half_window);
                                let s = &mut buf[rr * n..(rr + 1) * n];
                                let max = k::max_row(s, lo, hi);
                                let l = k::exp_row(s, lo, hi, max);
                                inv[rr] = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
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

#[cfg(test)]
mod alibi_tests {
    //! The ALiBi path of [`super::attend_texts`], held to a row-by-row reference.

    use super::*;

    /// The engine's attention for one text, written out: one sequential FMA chain per
    /// `(row, key)` for the score, `scale`, then ALiBi's `slope·(−|i−j|)`, the maximum,
    /// the exponentials (`vmath::exp_scalar`, the block's own), an ascending `f64` sum
    /// and an ascending value mix — the arithmetic `peregrine_kernels::bidir` defines.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        cfg: &Config,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n: usize,
        slopes: &[f32],
        scale: f32,
    ) -> Vec<f32> {
        let hd = cfg.head_dim;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let mut out = vec![0.0f32; n * qd];
        let mut s = vec![0.0f32; n];
        for h in 0..cfg.n_head {
            for i in 0..n {
                for (j, o) in s.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for d in 0..hd {
                        acc = q[i * qd + h * hd + d].mul_add(k[j * kvd + h * hd + d], acc);
                    }
                    *o = acc * scale;
                }
                for (j, o) in s.iter_mut().enumerate() {
                    *o += slopes[h] * -(j.abs_diff(i) as f32);
                }
                let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut e = vec![0.0f32; n];
                for j in 0..n {
                    e[j] = crate::vmath::exp_scalar(s[j] - max);
                }
                let mut sum = 0.0f64;
                for &x in &e {
                    sum += x as f64;
                }
                let inv = (1.0f64 / sum) as f32;
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for (j, &ej) in e.iter().enumerate() {
                        acc = ej.mul_add(v[j * kvd + h * hd + d], acc);
                    }
                    out[i * qd + h * hd + d] = acc * inv;
                }
            }
        }
        out
    }

    fn cfg() -> Config {
        Config {
            arch_name: "jina-bert-v2",
            rope_neox: false,
            n_layer: 1,
            n_kv_layer: 1,
            n_head: 4,
            n_kv_head: 4,
            head_dim: 8,
            d_model: 32,
            n_ff: 64,
            n_vocab: 10,
            rope_dim: 0,
            eps: 1e-12,
            rope_base: 10_000.0,
            ctx_train: 128,
            rope_yarn: None,
        }
    }

    fn prng(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / 16_777_216.0).mul_add(2.0, -1.0)
            })
            .collect()
    }

    #[test]
    fn alibi_attention_is_the_row_by_row_reference_at_every_strip_and_thread_count() {
        let cfg = cfg();
        let n = 19;
        let q = prng(1, n * cfg.q_dim());
        let k = prng(2, n * cfg.kv_dim());
        let v = prng(3, n * cfg.kv_dim());
        let slopes = peregrine_kernels::bidir::alibi_slopes(cfg.n_head, 8.0);
        let want = naive(&cfg, &q, &k, &v, n, &slopes, 0.5);
        let spans = [TextSpan { row0: 0, len: n }];
        let mut got = vec![0.0f32; n * cfg.q_dim()];
        for threads in [1usize, 3, 4] {
            let pool = Pool::new(threads);
            let mut scratch = BidirScratch::default();
            scratch.resize(n, cfg.head_dim, threads);
            let mut stats = StepStats::default();
            got.iter_mut().for_each(|x| *x = 0.0);
            attend_texts(
                &cfg,
                &q,
                &k,
                &v,
                &mut got,
                &spans,
                None,
                Some(&slopes),
                0.5,
                &mut scratch,
                &pool,
                &mut stats,
            );
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "threads {threads}, index {i}");
            }
        }
    }

    #[test]
    fn a_batch_does_not_change_a_text_with_alibi() {
        let cfg = cfg();
        let a = 11;
        let b = 7;
        let n = a + b;
        let q = prng(4, n * cfg.q_dim());
        let k = prng(5, n * cfg.kv_dim());
        let v = prng(6, n * cfg.kv_dim());
        let slopes = peregrine_kernels::bidir::alibi_slopes(cfg.n_head, 8.0);
        let pool = Pool::new(2);
        let mut stats = StepStats::default();

        let mut run = |span_rows: &[usize]| -> Vec<Vec<f32>> {
            let mut out = vec![0.0f32; n * cfg.q_dim()];
            let mut spans = Vec::new();
            let mut row = 0;
            for &len in span_rows {
                spans.push(TextSpan { row0: row, len });
                row += len;
            }
            let mut scratch = BidirScratch::default();
            scratch.resize(n, cfg.head_dim, 2);
            attend_texts(
                &cfg,
                &q,
                &k,
                &v,
                &mut out,
                &spans,
                None,
                Some(&slopes),
                0.5,
                &mut scratch,
                &pool,
                &mut stats,
            );
            span_rows
                .iter()
                .zip(&spans)
                .map(|(&len, s)| out[s.row0 * cfg.q_dim()..(s.row0 + len) * cfg.q_dim()].to_vec())
                .collect()
        };

        // The two texts together, then the first alone from the same rows.
        let both = run(&[a, b]);
        let alone = run(&[a]);
        assert_eq!(
            both[0], alone[0],
            "a text's rows must not see its neighbour"
        );
    }
}
