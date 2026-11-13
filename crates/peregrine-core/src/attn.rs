//! Attention over the KV cache: grouped by KV head, with an online softmax.
//!
//! # What changed, and why (uzu)
//!
//! The previous loop ran one task per (batch row, query head). Each task scanned
//! every cached position of its KV head, widening the stored f16 or int8 key to f32
//! as it went, kept the whole score row, took a scalar `exp` per score, and scanned
//! the values again. With grouped-query attention — 4 query heads per KV head on
//! Qwen3-4B and Llama 3.2, 3 on SmolLM2 — that re-read and re-widened the same K/V
//! bytes once per query head, and during prefill or verification once per batch row
//! as well.
//!
//! uzu's Metal attention (`attention/gemm_grouped/attention.metal`, the two-pass
//! kernel and its `combine`) does it the other way round, and this is that design on
//! a CPU:
//!
//! * **One task per KV head** (and, when there is enough work, a block of batch rows
//!   or a chunk of positions). A block of `BLOCK` positions is decoded once into an
//!   f32 tile and every query row that shares the KV head — all `gqa` heads of every
//!   batch row in the task — is scored against it. K/V bytes are read once per task.
//! * **Online softmax** (flash attention): each row keeps a running max `m`, sum `l`
//!   and output `o`, rescaled when a block raises the max, so no `n_ctx`-long score
//!   row is ever stored and the value mix happens block by block while the decoded
//!   tile is hot in L1.
//! * **Split positions when the heads do not fill the pool** (flash-decoding): a
//!   model with two KV heads (Qwen2.5-0.5B) or five (SmolLM2) cannot keep four cores
//!   busy with one task per head at decode. The positions are then split into chunks;
//!   each chunk writes its partial `(o, m, l)` and a combine step merges them with
//!   `exp(m_i − m*)` weights — uzu's `combine.metal`.
//! * The scores use a transposed horizontal add instead of one reduction per key,
//!   and the exponentials are `vmath`'s vector `exp`.
//! * **Decode reads F16 K/V directly.** A one-row call (decode, or a chunk of one)
//!   consumes the stored halves in registers with `vcvtph2ps`; no f32 tile is
//!   written. The score pass is register-blocked four keys deep, two query rows share
//!   each loaded K vector, and the value mix keeps its accumulators in registers while
//!   two rows share each loaded V vector. A decode head group (seven query heads per
//!   KV head on Qwen2.5-0.5B) therefore reads the cache once per two query heads.
//! * **Prefill widens the block once** below ~1k positions of context: the tile is
//!   small enough to sit in cache, and one widening amortises over every query row
//!   that reads it. Past that the cache no longer fits, so the rows read the stored
//!   halves in registers instead (two rows share each loaded K and V vector), which
//!   halves the bytes they pull. Both paths use the same per-key reduction tree, so
//!   which one a row takes cannot change its bits.
//! * **The value mix widens to sixteen lanes on AVX-512.** `vcvtph2ps zmm` reads
//!   sixteen stored halves per instruction and 512-bit fused multiply-adds consume
//!   them; the mix is conversion-bound, so on the Ryzen AI 9 HX 370 the pair kernel
//!   is ~27% faster than the AVX2 one at 4096 positions. Scores keep the AVX2 tree on
//!   every tier: a zmm score row needs a 16-lane horizontal reduction, which measured
//!   30–70% *slower* than the AVX2 reduction at head width 64, so the tier keeps one
//!   score tree and every path agrees bit for bit with AVX2. The softmax's 16-wide
//!   `exp` keeps the eight-lane sum grouping, so it too is the AVX2 result bit for
//!   bit; the mix is per-element and matches for the same reason. Attention's output
//!   is therefore the AVX2 tier's, and this tier's goldens move only where another
//!   kernel (the repacked k-quant matmuls) does.
//! * **A sliding window is only the visible range.** A row with a window sees its
//!   last `window` positions; the position chunks are the even cut of that range, and
//!   every tile walk starts at it, so a windowed layer reads its window rather than
//!   the whole context. A window may start inside a tile: the fast kernels offset
//!   into the cache's halves at the same position the widened tile would start at.
//! * **A row's partition is its own.** The position chunks a row is cut into are a
//!   function of that row's visible context and the model shape alone — never of the
//!   other rows in the call, the call width, the prefill chunk or the pool. A
//!   single-row call spreads the chunks over tasks; a multi-row call spreads the
//!   rows and walks each row's chunks inside its task, in the same order. The merge
//!   is not order-exact against a different chunk count, so this is what makes
//!   prefill, speculative verification and plain decode agree bit for bit.
//!
//! The arithmetic is the same attention: identical scores, a softmax that differs
//! only in float summation order, and the tests pin it against a naive reference.

use crate::kv::KvCache;

/// Positions decoded and scored per step. 64 × 128 floats is a 32 KB tile each for
/// K and V — inside L2 with room to spare, and long enough to amortise the rescale.
pub const BLOCK: usize = 64;

/// The shortest visible context at which a prefill tile reads the F16 cache directly.
///
/// Below it a row's K/V fits in the core's private caches, so halving the bytes the
/// tile reads buys less than the per-row `vcvtph2ps` costs — the direct path measured
/// about 1% slower on 512-position prompts. The crossover is around a thousand
/// positions (five alternating rounds on H0), so shorter contexts keep the widened
/// tile while long ones read the halves in registers.
#[cfg(target_arch = "x86_64")]
const DIRECT_MIN_VIS: usize = 1024;

/// Most position chunks per KV head, which bounds the partial-result buffer.
pub const MAX_CHUNKS: usize = 8;

/// The number of tasks a call with split positions aims for.
///
/// A constant, not the pool width. [`combine_row`] is mathematically exact but not
/// bit-exact against a different chunk count: the merge weighs each partial by
/// `exp(m_i − m*)`, whereas the unsplit pass applies the online softmax's successive
/// `α` rescales, so the two round differently. A chunk count that followed the pool
/// would therefore make a prompt's bits depend on the number of threads the engine
/// happened to start, and the thread-invariance golden compares exactly that. So the
/// chunk count is a function of the context length and the model shape only.
///
/// Twenty-four is two tasks per worker on the twelve-core reference host, which is
/// the target the old worker-derived plan aimed for; an eight-worker host (bluefin)
/// runs three per worker, which costs a little tail imbalance but keeps the bits.
const TARGET_TASKS: usize = 24;

/// Most query rows (batch rows × `gqa`) whose chunk partials are held at once.
///
/// It bounds both buffers: a single-row position-split call's shared partials, and
/// the per-worker partials [`run_rows`] keeps while a multi-row task walks its rows'
/// chunks. A wider task processes sub-blocks of this many rows, so the arithmetic of
/// a row never depends on which sub-block it landed in.
pub const MAX_PARTIAL_ROWS: usize = 64;

/// The geometry of one attention call.
#[derive(Clone, Copy, Debug)]
pub struct Shape<'a> {
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Batch rows: their positions are `start_pos..start_pos + nb`, and row `b` sees
    /// cache positions `0..=start_pos + b` (causal).
    pub nb: usize,
    pub start_pos: usize,
    pub scale: f32,
    /// Sliding window attention: row `b` sees at most the last `window` positions
    /// (`(pos + 1 - window)..=pos`). `None` for full attention.
    pub window: Option<usize>,
    /// The attention sinks: one learned logit per query head, added to the
    /// softmax's denominator (`src/models/openai-moe.cpp:42`, gpt-oss).
    pub sinks: Option<&'a [f32]>,
    /// Which batch rows are **fast prompt rows** (S5.4j): one flag per row of this
    /// call, `None` when no row of it is a prompt row of the opt-in fast prompt mode
    /// or when this tier keeps the exact order. A flagged row computes its attention
    /// in the fast order `fast` documents (its own scores' tree, its own denominator
    /// fold) — which is lossy rounding-wise and is why only the mode's prompt rows
    /// take it. Every other row, every read-out and every tier without
    /// the flag keeps the exact engine's attention bit for bit.
    pub fast: Option<&'a [bool]>,
}

/// Whether batch row `b` of this call is a fast prompt row (S5.4j).
#[inline]
pub(crate) fn fast_row(sh: &Shape<'_>, b: usize) -> bool {
    sh.fast.is_some_and(|f| f.get(b).copied().unwrap_or(false))
}

impl<'a> Shape<'a> {
    pub fn gqa(&self) -> usize {
        self.n_head / self.n_kv_head
    }
    fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
}

/// How a call is divided: `row_blocks × pos_chunks` tasks per KV head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub parallel: bool,
    pub row_blocks: usize,
    pub pos_chunks: usize,
}

impl Plan {
    pub fn tasks(&self, sh: &Shape) -> usize {
        sh.n_kv_head * self.row_blocks * self.pos_chunks
    }
}

/// Divide a call among `workers`. Rows are split first (no combine needed), then
/// positions, into chunks of at least `BLOCK` positions and enough of them to reach
/// `TARGET_TASKS` tasks for the call.
///
/// A row's arithmetic must depend only on that row's own visible context and the
/// model shape — never on the other rows in the call, the call width (`nb`), the
/// prefill chunk or the pool. The position partition therefore is not a property of
/// the call at all: `row_chunks` gives every row its own chunk count from its own
/// visible context (a sliding window is the last `window` positions), and the cut is
/// the even one over that range. A single-row call splits those chunks across tasks
/// (flash decoding); a multi-row call splits rows across tasks and each task walks
/// every row's chunks serially with the same boundaries and the same merge order (see
/// [`run_rows`]). Both paths compute the same partials and call [`combine_row`] over
/// them in the same order, so a row's bits are the same either way.
///
/// `row_blocks` may follow the pool: each task computes whole rows, and a row's
/// arithmetic does not depend on how many rows share its task. `pos_chunks` for a
/// single-row call is `row_chunks`' count, also pool-independent.
///
/// A fork–join costs ~0.5 µs (measured; the 20 µs this used to assume kept attention
/// on one core, with three spinning, until ~270 positions), so only tiny calls stay
/// serial, and mid-sized ones fork one task per KV head — the very computation the
/// serial loop does per head, so the result is bit-identical.
pub fn plan(sh: &Shape<'_>, workers: usize) -> Plan {
    let avg_ctx = sh.start_pos + sh.nb.div_ceil(2);
    let work = sh.nb * sh.n_head * avg_ctx * sh.head_dim * 2;
    let ctx = sh.start_pos + sh.nb;
    if work < 1 << 13 {
        return Plan {
            parallel: false,
            row_blocks: 1,
            pos_chunks: 1,
        };
    }
    // Only a call that is one row can split positions across tasks. A multi-row call
    // splits rows and lets each row walk its own canonical chunks inside the task.
    // A fast prompt row always walks whole `BLOCK`-key tiles (S5.4j), so its partition
    // is never cut by the pool.
    let chunks = if sh.nb > 1 || sh.nb * sh.gqa() > MAX_PARTIAL_ROWS || fast_row(sh, 0) {
        1
    } else {
        row_chunks(sh, ctx)
    };
    if workers <= 1 {
        // The same partition, run serially: the bits must not change with the pool.
        return Plan {
            parallel: false,
            row_blocks: 1,
            pos_chunks: chunks,
        };
    }
    if work < 1 << 19 {
        return Plan {
            parallel: sh.n_kv_head > 1,
            row_blocks: 1,
            pos_chunks: 1,
        };
    }
    let per_head = (2 * workers).div_ceil(sh.n_kv_head);
    // Rows carry the rest of the target; when positions already fill it, one block
    // of rows is enough (and the old plan's row/position split is reproduced).
    let row_blocks = per_head.div_ceil(chunks).min(sh.nb).max(1);
    Plan {
        parallel: true,
        row_blocks,
        pos_chunks: chunks,
    }
}

/// One row's position partition: the number of chunks *that row's* context is cut
/// into, from its context length and the model shape alone.
///
/// The chunk count is the task target spread over the KV heads, capped so a chunk
/// holds at least one decode tile (`BLOCK`) and by `MAX_CHUNKS` (the partial buffer).
/// Only the visible positions count as work: a sliding window can leave a 4096-token
/// row with just its last 512 positions. The boundaries are the even cut of that
/// visible range, which is what the single-row decode path uses too; they need not
/// land on the `BLOCK` grid (see [`run_rows`]). The work gate is per row, so a short
/// row is never split just because a wide call is long.
fn row_chunks(sh: &Shape<'_>, n_ctx: usize) -> usize {
    // Only the visible positions are work: a sliding window can leave a 4096-token
    // row with nothing but its last 512 positions to read and score.
    let visible = n_ctx - visible_start(sh, n_ctx);
    let work = sh.n_head * visible * sh.head_dim * 2;
    if work < 1 << 19 {
        return 1;
    }
    let by_ctx = (visible / BLOCK).max(1);
    let by_heads = TARGET_TASKS.div_ceil(sh.n_kv_head);
    by_heads.min(by_ctx).clamp(1, MAX_CHUNKS)
}

/// The first position a row with `n_ctx` positions can see: the whole context, or,
/// with a sliding window, only its last `window` positions.
fn visible_start(sh: &Shape<'_>, n_ctx: usize) -> usize {
    match sh.window {
        Some(w) => n_ctx.saturating_sub(w),
        None => 0,
    }
}

/// One task: KV head `kh`, batch rows `b0..b1`, cache positions `p0..p1`.
#[derive(Clone, Copy, Debug)]
pub struct Task {
    pub kh: usize,
    pub b0: usize,
    pub b1: usize,
    pub p0: usize,
    pub p1: usize,
}

/// Task `i` of a plan, in KV-head-major order.
pub fn task(sh: &Shape<'_>, p: &Plan, i: usize) -> Task {
    let pi = i % p.pos_chunks;
    let ri = (i / p.pos_chunks) % p.row_blocks;
    let kh = i / (p.pos_chunks * p.row_blocks);
    let (b0, b1) = (ri * sh.nb / p.row_blocks, (ri + 1) * sh.nb / p.row_blocks);
    let ctx = sh.start_pos + sh.nb;
    // A single row partitions its visible positions — the even cut of the window when
    // there is one, not of the context, so a windowed layer's tasks split the work it
    // actually has. A multi-row task keeps the whole-context marker and walks each
    // row's own visible chunks inside it ([`run_rows`]).
    let (p0, p1) = if sh.nb == 1 {
        let v0 = visible_start(sh, ctx);
        let visible = ctx - v0;
        (
            v0 + pi * visible / p.pos_chunks,
            v0 + (pi + 1) * visible / p.pos_chunks,
        )
    } else {
        (0, ctx)
    };
    Task { kh, b0, b1, p0, p1 }
}

/// One worker's scratch, sized for the largest task it can be given.
pub struct Scratch {
    kf: Vec<f32>,
    vf: Vec<f32>,
    s: Vec<f32>,
    q: Vec<f32>,
    pub o: Vec<f32>,
    pub m: Vec<f32>,
    pub l: Vec<f64>,
    /// One tile's raw scores per query row, `row · BLOCK` floats, so every row of a
    /// multi-row task can be scored against the shared K tile in one pass, and the
    /// tile can be widened again as V before the mixes consume the scores.
    s_tile: Vec<f32>,
    /// One row's per-chunk partials while [`run_rows`] walks its chunks: `MAX_CHUNKS`
    /// contiguous `(o, m, l)` triples per row, for at most `MAX_PARTIAL_ROWS` rows.
    partial_o: Vec<f32>,
    partial_m: Vec<f32>,
    partial_l: Vec<f64>,
}

impl Scratch {
    pub fn new(max_rows: usize, head_dim: usize) -> Scratch {
        Scratch {
            kf: vec![0.0; BLOCK * head_dim],
            vf: vec![0.0; BLOCK * head_dim],
            s: vec![0.0; BLOCK],
            q: vec![0.0; max_rows * head_dim],
            o: vec![0.0; max_rows * head_dim],
            m: vec![0.0; max_rows],
            l: vec![0.0; max_rows],
            s_tile: vec![0.0; MAX_PARTIAL_ROWS * BLOCK],
            partial_o: vec![0.0; MAX_PARTIAL_ROWS * MAX_CHUNKS * head_dim],
            partial_m: vec![0.0; MAX_PARTIAL_ROWS * MAX_CHUNKS],
            partial_l: vec![0.0; MAX_PARTIAL_ROWS * MAX_CHUNKS],
        }
    }
}

/// Accumulate `(o, m, l)` for the task's rows — row `r = (b − b0)·gqa + g` is query
/// head `kh·gqa + g` of batch row `b` — over its positions. `q` is the call's query
/// buffer, `[nb][n_head][head_dim]`, after RoPE. When the cache is rotated, the
/// queries are rotated here into scratch; the cache's values stay rotated in `o`.
pub fn run(kv: &KvCache, layer: usize, sh: &Shape<'_>, q: &[f32], t: &Task, sc: &mut Scratch) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if t.b1 - t.b0 == 1 && f16_direct(kv, sh) {
        return run_f16(kv, layer, sh, q, t, sc);
    }
    let hd = sh.head_dim;
    let gqa = sh.gqa();
    let qd = sh.q_dim();
    let rows = (t.b1 - t.b0) * gqa;
    for bi in 0..t.b1 - t.b0 {
        for g in 0..gqa {
            let h = t.kh * gqa + g;
            let src = &q[(t.b0 + bi) * qd + h * hd..][..hd];
            let dst = &mut sc.q[(bi * gqa + g) * hd..][..hd];
            dst.copy_from_slice(src);
            if kv.rotated() {
                crate::vmath::hadamard(dst);
            }
        }
    }
    sc.o[..rows * hd].fill(0.0);
    sc.m[..rows].fill(f32::NEG_INFINITY);
    sc.l[..rows].fill(0.0);

    // No row of this task sees past `start_pos + b1 − 1`.
    let p_end = t.p1.min(sh.start_pos + t.b1);
    // A sliding window keeps every row from seeing further back than its own
    // `n_ctx − window`; no row of the task sees further back than its first row's.
    // Tiles entirely before it contribute nothing, so they are not decoded at all —
    // this is the read a windowed layer would otherwise do over the whole context.
    let p_min_task = match sh.window {
        Some(w) => (sh.start_pos + t.b0 + 1).saturating_sub(w),
        None => 0,
    };
    // Tiles stay on the global `BLOCK` grid: a chunk may start inside one, and the
    // multi-row walker ([`run_rows`]) then scores the same tile for every row. Only
    // the part from `t.p0` on is read, so a chunk is never decoded twice.
    let mut blk = t.p0.max(p_min_task / BLOCK * BLOCK);
    while blk < p_end {
        let bn = ((blk / BLOCK + 1) * BLOCK - blk).min(p_end - blk);
        let tdec = crate::prof::detail().then(std::time::Instant::now);
        kv.decode_block(true, layer, t.kh, blk, bn, &mut sc.kf);
        kv.decode_block(false, layer, t.kh, blk, bn, &mut sc.vf);
        if let Some(t) = tdec {
            crate::prof::add_shared(crate::prof::Slot::AttnDecode, t.elapsed().as_nanos() as u64);
        }
        for bi in 0..t.b1 - t.b0 {
            let n_ctx = sh.start_pos + t.b0 + bi + 1;
            if n_ctx <= blk {
                continue;
            }
            let p_min = match sh.window {
                Some(w) => n_ctx.saturating_sub(w),
                None => 0,
            };
            if blk + bn <= p_min {
                continue;
            }
            let j_start = p_min.saturating_sub(blk);
            let vis = bn.min(n_ctx - blk);
            if j_start >= vis {
                continue;
            }
            let count = vis - j_start;
            for g in 0..gqa {
                let r = bi * gqa + g;
                let s = &mut sc.s[..count];
                let ts = crate::prof::detail().then(std::time::Instant::now);
                scores(
                    &sc.q[r * hd..][..hd],
                    &sc.kf[j_start * hd..],
                    count,
                    hd,
                    sh.scale,
                    s,
                );
                if let Some(t) = ts {
                    crate::prof::add_shared(
                        crate::prof::Slot::AttnScores,
                        t.elapsed().as_nanos() as u64,
                    );
                }
                let m_old = sc.m[r];
                let tsm = crate::prof::detail().then(std::time::Instant::now);
                let m_new = m_old.max(crate::vmath::max(s));
                let psum = crate::vmath::exp_shift_sum(s, m_new);
                // exp(−∞) = 0 on the first block: nothing to rescale.
                let alpha = crate::vmath::det_exp(m_old - m_new);
                sc.l[r] = sc.l[r] * alpha as f64 + psum;
                if let Some(t) = tsm {
                    crate::prof::add_shared(
                        crate::prof::Slot::AttnSoftmax,
                        t.elapsed().as_nanos() as u64,
                    );
                }
                let tm = crate::prof::detail().then(std::time::Instant::now);
                mix(
                    &mut sc.o[r * hd..][..hd],
                    alpha,
                    s,
                    &sc.vf[j_start * hd..],
                    hd,
                );
                if let Some(t) = tm {
                    crate::prof::add_shared(
                        crate::prof::Slot::AttnMix,
                        t.elapsed().as_nanos() as u64,
                    );
                }
                sc.m[r] = m_new;
            }
        }
        blk += bn;
    }
}

/// [`run`] for one or more query rows of one KV head when the cache is f16 and the
/// half-precision tile can be consumed in registers ([`f16_direct`]).
///
/// Every row of the task sees the same positions (one batch row), so the tile is
/// scored for all of them — two rows share each decoded K vector — then each row runs
/// its own online-softmax update, and the value mix shares each decoded V vector
/// between two rows. The per-row arithmetic is the same as the scalar fallback's: a
/// score is a dot over ascending dimension blocks, the softmax rescales at the same
/// tile boundaries, and the mix accumulates ascending positions. Only the loaded
/// operands are shared.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn run_f16(kv: &KvCache, layer: usize, sh: &Shape<'_>, q: &[f32], t: &Task, sc: &mut Scratch) {
    let hd = sh.head_dim;
    let gqa = sh.gqa();
    let qd = sh.q_dim();
    let rows = (t.b1 - t.b0) * gqa;
    for bi in 0..t.b1 - t.b0 {
        for g in 0..gqa {
            let h = t.kh * gqa + g;
            let src = &q[(t.b0 + bi) * qd + h * hd..][..hd];
            sc.q[(bi * gqa + g) * hd..][..hd].copy_from_slice(src);
        }
    }
    sc.o[..rows * hd].fill(0.0);
    sc.m[..rows].fill(f32::NEG_INFINITY);
    sc.l[..rows].fill(0.0);

    // One batch row, so every query row of this task has the same context and the
    // same sliding-window start.
    let n_ctx = sh.start_pos + t.b0 + 1;
    let p_end = t.p1.min(n_ctx);
    let p_min = match sh.window {
        Some(w) => n_ctx.saturating_sub(w),
        None => 0,
    };
    // The first tile a row can see; anything before it contributes no partial and is
    // not decoded. Starting on the global `BLOCK` grid keeps the tile boundaries — and
    // therefore every score's reduction tree — exactly the widened path's.
    let mut blk = t.p0.max(p_min / BLOCK * BLOCK);
    while blk < p_end {
        let bn = ((blk / BLOCK + 1) * BLOCK - blk).min(p_end - blk);
        let vis_end = (blk + bn).min(n_ctx);
        let start = blk.max(p_min);
        // A tile wholly before the window still advances the tile loop; it has no
        // visible position and must not be read.
        if start >= vis_end {
            blk += bn;
            continue;
        }
        // A window can begin inside a tile, so the visible keys and values start at
        // `j_start` in the cache's own halves.
        let j_start = start - blk;
        let vis = vis_end - start;
        let kb = &kv.block_bytes(true, layer, t.kh, blk, bn)[j_start * hd * 2..];
        let vb = &kv.block_bytes(false, layer, t.kh, blk, bn)[j_start * hd * 2..];
        if vis > 0 {
            let ts = crate::prof::detail().then(std::time::Instant::now);
            for r in (0..rows).step_by(2) {
                if r + 1 < rows {
                    let (s_all, tail) = sc.s_tile.split_at_mut((r + 1) * BLOCK);
                    let s = &mut s_all[r * BLOCK..];
                    let s1 = &mut tail[..BLOCK];
                    // SAFETY: AVX2+FMA+F16C checked; the tile holds `vis · hd` halves.
                    unsafe {
                        scores_f16_pair(
                            &sc.q[r * hd..],
                            &sc.q[(r + 1) * hd..],
                            kb,
                            vis,
                            vis,
                            hd,
                            sh.scale,
                            s,
                            s1,
                        )
                    };
                } else {
                    let s = &mut sc.s_tile[r * BLOCK..(r + 1) * BLOCK];
                    // SAFETY: features checked; the tile holds `vis · hd` halves.
                    unsafe { scores_f16(&sc.q[r * hd..], kb, vis, hd, sh.scale, s) };
                }
            }
            if let Some(t) = ts {
                crate::prof::add_shared(
                    crate::prof::Slot::AttnScores,
                    t.elapsed().as_nanos() as u64,
                );
            }
            // One online-softmax step per row, as in the widened path.
            let mut alpha = [0.0f32; MAX_PARTIAL_ROWS];
            let tsm = crate::prof::detail().then(std::time::Instant::now);
            for r in 0..rows {
                let s = &mut sc.s_tile[r * BLOCK..r * BLOCK + vis];
                let m_old = sc.m[r];
                let m_new = m_old.max(crate::vmath::max(s));
                let psum = crate::vmath::exp_shift_sum(s, m_new);
                alpha[r] = crate::vmath::det_exp(m_old - m_new);
                sc.l[r] = sc.l[r] * alpha[r] as f64 + psum;
                sc.m[r] = m_new;
            }
            if let Some(t) = tsm {
                crate::prof::add_shared(
                    crate::prof::Slot::AttnSoftmax,
                    t.elapsed().as_nanos() as u64,
                );
            }
            let tm = crate::prof::detail().then(std::time::Instant::now);
            #[cfg(target_arch = "x86_64")]
            for r in (0..rows).step_by(2) {
                let p = &sc.s_tile[r * BLOCK..r * BLOCK + vis];
                if r + 1 < rows {
                    let (lo, hi) = sc.o.split_at_mut((r + 1) * hd);
                    let p1 = &sc.s_tile[(r + 1) * BLOCK..(r + 1) * BLOCK + vis];
                    // A tile in `run_f16` never crosses a row chunk (`run` is one
                    // chunk per task), so the end list holds one entry and no partial
                    // is banked inside the tile.
                    let one = [usize::MAX];
                    // SAFETY: features checked; `vb` holds `vis · hd` halves.
                    unsafe {
                        mix_f16_pair(
                            &mut lo[r * hd..],
                            &mut hi[..hd],
                            alpha[r],
                            alpha[r + 1],
                            p,
                            p1,
                            vis,
                            vis,
                            blk,
                            0,
                            0,
                            &one,
                            &one,
                            vb,
                            hd,
                            &mut [],
                            &mut [],
                        )
                    };
                } else {
                    // SAFETY: features checked; `vb` holds `vis · hd` halves.
                    unsafe { mix_f16(&mut sc.o[r * hd..], alpha[r], p, vb, hd) };
                }
            }
            // On aarch64 the mix already reads the halves in registers; each row is
            // mixed on its own (the per-element work has no K/V sharing to gain).
            #[cfg(target_arch = "aarch64")]
            for r in 0..rows {
                let p = &sc.s_tile[r * BLOCK..r * BLOCK + vis];
                // SAFETY: NEON checked; `vb` holds `vis · hd` halves.
                unsafe { mix_f16(&mut sc.o[r * hd..(r + 1) * hd], alpha[r], p, vb, hd) };
            }
            if let Some(t) = tm {
                crate::prof::add_shared(crate::prof::Slot::AttnMix, t.elapsed().as_nanos() as u64);
            }
        }
        blk += bn;
    }
}

/// Where a multi-row task's finished rows go.
///
/// A task owns the `gqa` query heads of one KV head in rows `b0..b1`, and each of
/// those `head_dim`-wide slices is strided by the full query width in the output
/// buffer. Handing out one slice per (row, head) lets the caller claim exactly the
/// bytes the task writes: the disjoint checker in `blocks::par` then sees per-head
/// ranges instead of whole rows, which the tasks of different KV heads share.
pub trait RowSink {
    /// The output slice for query head `head` of the task's `row` (0-based).
    fn head(&mut self, row: usize, head: usize) -> &mut [f32];
}

/// [`run`] for a multi-row task: every row walks its own canonical position chunks
/// (`row_chunks`) and the chunks are merged by [`combine_row`] in chunk order.
///
/// This is how a prefill chunk, a speculative verify or a mixed step reaches the same
/// bits as the same row decoded alone. A task covers whole rows (that split is free),
/// so the chunks cannot be separate tasks — but walking them here, inside the task,
/// and merging them in the same order is exactly what the single-row split path does
/// across tasks: the same chunk boundaries, the same tiles inside each chunk, the same
/// `parts[0..k]` passed to [`combine_row`]. A boundary may fall inside a `BLOCK` tile
/// (the even cut moves by one position with the row's context); the tile is decoded
/// once and scored in the two pieces each chunk would have read on its own.
///
/// Rows are processed in blocks of at most `MAX_PARTIAL_ROWS` so the per-row partial
/// buffer stays bounded; a sub-block reads its K/V again, but the arithmetic of a row
/// does not depend on which sub-block it landed in.
///
/// # The query-row × position tile
///
/// A prefill task carries tens of query rows over the same KV head, and each of them
/// scores the same tile of positions. The tile is therefore widened once and consumed
/// by every row (`run_range`): keys first, scored for all rows, then values, mixed
/// into all rows. Two rows share each loaded K vector (`scores_pair`) — the loads
/// are shared, the accumulators, the chunk boundaries and the merge order are not, so
/// a row's bits stay a function of its own query and context alone.
///
/// `out` receives only this task's query heads (`kh·gqa..kh·gqa+gqa`) of its rows.
pub fn run_rows<O: RowSink>(
    kv: &KvCache,
    layer: usize,
    sh: &Shape<'_>,
    q: &[f32],
    t: &Task,
    sc: &mut Scratch,
    out: &mut O,
) {
    debug_assert_eq!(t.p0, 0, "run_rows expects a whole-context task");
    let gqa = sh.gqa();
    let rows = (t.b1 - t.b0) * gqa;
    let mut r0 = 0;
    while r0 < rows {
        let r1 = (r0 + MAX_PARTIAL_ROWS).min(rows);
        // A sub-block of one row kind is walked by that kind's path: a fast prompt row
        // (S5.4j) has its own pieces — the absolute `BLOCK`-key tiles — and shares
        // them with the fast rows beside it; every other row keeps the exact engine's
        // path, bit for bit. Splitting a mixed block into runs of one kind keeps a
        // row's bits the row's own: neither path reads a neighbour's partition.
        let mut r = r0;
        while r < r1 {
            let fast = fast_row_of(sh, t, r);
            let mut e = r + 1;
            while e < r1 && fast_row_of(sh, t, e) == fast {
                e += 1;
            }
            if fast {
                run_range_fast(kv, layer, sh, q, t, r, e, sc, out);
            } else {
                run_range(kv, layer, sh, q, t, r, e, sc, out);
            }
            r = e;
        }
        r0 = r1;
    }
}

/// Whether task row `r` (a query head of one batch row) is a fast prompt row.
#[inline]
fn fast_row_of(sh: &Shape<'_>, t: &Task, r: usize) -> bool {
    fast_row(sh, t.b0 + r / sh.gqa())
}

/// [`run_rows`]'s fast branch: one sub-block of **fast prompt rows** (S5.4j).
///
/// The rows walk their own pieces — the absolute `BLOCK`-key tiles (and a sliding
/// window's own start) — so there is no position chunking and each row leaves exactly
/// one `(o, m, l)` part. Up to [`fast::FAST_ROWS`] rows that see the same keys of a
/// tile share the pass over K and V; each row's score, denominator and mix stay its
/// own arithmetic, in the order `fast`'s definition writes down.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_range_fast<O: RowSink>(
    kv: &KvCache,
    layer: usize,
    sh: &Shape<'_>,
    q: &[f32],
    t: &Task,
    r0: usize,
    r1: usize,
    sc: &mut Scratch,
    out: &mut O,
) {
    fast_walk(kv, layer, sh, q, t, r0, r1, sc);
    let hd = sh.head_dim;
    let gqa = sh.gqa();
    let sub = r1 - r0;
    let rotated = kv.rotated();
    for lr in 0..sub {
        let r = r0 + lr;
        let (bi, g) = (r / gqa, r % gqa);
        let h = t.kh * gqa + g;
        let part: [(&[f32], f32, f64); 1] = [(&sc.o[lr * hd..(lr + 1) * hd], sc.m[lr], sc.l[lr])];
        let sink = sh.sinks.map(|s| s[h]);
        combine_row(&part, sink, rotated, out.head(bi, h));
    }
}

/// One fast row's attention as one task: the fast walk, leaving `(o, m, l)` in `sc`
/// for the caller's [`finish_row`].
pub fn run_fast(kv: &KvCache, layer: usize, sh: &Shape<'_>, q: &[f32], t: &Task, sc: &mut Scratch) {
    let rows = (t.b1 - t.b0) * sh.gqa();
    fast_walk(kv, layer, sh, q, t, 0, rows, sc);
}

/// The walk both fast entry points share: fills `sc.q`, `sc.o`, `sc.m` and `sc.l` for
/// the task's rows `r0..r1`.
#[allow(clippy::too_many_arguments)]
fn fast_walk(
    kv: &KvCache,
    layer: usize,
    sh: &Shape<'_>,
    q: &[f32],
    t: &Task,
    r0: usize,
    r1: usize,
    sc: &mut Scratch,
) {
    let hd = sh.head_dim;
    let gqa = sh.gqa();
    let qd = sh.q_dim();
    let kh = t.kh;
    let sub = r1 - r0;
    debug_assert!(sub <= MAX_PARTIAL_ROWS, "a fast walk takes one sub-block");
    for lr in 0..sub {
        let r = r0 + lr;
        let (bi, g) = (r / gqa, r % gqa);
        let h = kh * gqa + g;
        let src = &q[(t.b0 + bi) * qd + h * hd..][..hd];
        sc.q[lr * hd..][..hd].copy_from_slice(src);
        if kv.rotated() {
            crate::vmath::hadamard(&mut sc.q[lr * hd..][..hd]);
        }
    }
    sc.o[..sub * hd].fill(0.0);
    sc.m[..sub].fill(f32::NEG_INFINITY);
    sc.l[..sub].fill(0.0);
    let mut ctx_of = [0usize; MAX_PARTIAL_ROWS];
    let mut max_ctx = 0;
    for lr in 0..sub {
        let b = t.b0 + (r0 + lr) / gqa;
        ctx_of[lr] = sh.start_pos + b + 1;
        max_ctx = max_ctx.max(ctx_of[lr]);
    }
    let p_min_task = match sh.window {
        Some(w) => (sh.start_pos + t.b0 + 1).saturating_sub(w),
        None => 0,
    };
    let p_end = t.p1.min(max_ctx);
    let mut blk = t.p0.max(p_min_task / BLOCK * BLOCK);
    let mut alpha = [1.0f32; MAX_PARTIAL_ROWS];
    // On aarch64 a F16 cache is read in the kernels' registers: the conversion is
    // exact, so the fast order's pieces see the same keys and values for half the
    // bytes the widened tile would stream (S5.4n).
    #[cfg(target_arch = "aarch64")]
    let direct = f16_direct(kv, sh);
    #[cfg(not(target_arch = "aarch64"))]
    let direct = false;
    while blk < p_end {
        let bn = ((blk / BLOCK + 1) * BLOCK - blk).min(p_end - blk);
        // 1. The keys: score every group of rows that sees this tile, and fold each
        //    row's own soft-max over its own piece.
        let k16 = direct.then(|| kv.block_bytes(true, layer, kh, blk, bn));
        if k16.is_none() {
            kv.decode_block(true, layer, kh, blk, bn, &mut sc.kf);
        }
        let mut lr = 0;
        while lr < sub {
            let (start, len0) = vis_range(sh, ctx_of[lr], blk, bn);
            if len0 == 0 {
                lr += 1;
                continue;
            }
            let mut rows = [0usize; fast::FAST_ROWS];
            let mut ends = [0usize; fast::FAST_ROWS];
            rows[0] = lr;
            ends[0] = start + len0;
            let mut n = 1;
            let mut nxt = lr + 1;
            while n < fast::FAST_ROWS && nxt < sub {
                let (s2, l2) = vis_range(sh, ctx_of[nxt], blk, bn);
                if s2 != start || l2 == 0 {
                    break;
                }
                rows[n] = nxt;
                ends[n] = s2 + l2;
                n += 1;
                nxt += 1;
            }
            let j0 = start - blk;
            // The scores, level by level: a level ends where the shortest row of the
            // group ends, so a row that sees the tile from the same key onward is
            // scored by the same pass.
            let mut lev = 0usize;
            loop {
                let mut act = [0usize; fast::FAST_ROWS];
                let mut na = 0;
                let mut e = usize::MAX;
                for r in 0..n {
                    if ends[r] > start + lev {
                        act[na] = rows[r];
                        na += 1;
                        e = e.min(ends[r] - start);
                    }
                }
                if na == 0 {
                    break;
                }
                let pad = act[0];
                for a in act.iter_mut().skip(na) {
                    *a = pad;
                }
                fast::scores_rows(
                    sc,
                    &act,
                    na,
                    j0 + lev,
                    e - lev,
                    hd,
                    sh.scale,
                    k16.map(|k| &k[(j0 + lev) * hd * 2..]),
                );
                lev = e;
            }
            for r in 0..n {
                let lr = rows[r];
                let len = ends[r] - start;
                let (m_new, psum, a) = fast::denom_row(sc, lr, j0, len, hd);
                sc.l[lr] = sc.l[lr] * a as f64 + psum;
                sc.m[lr] = m_new;
                alpha[lr] = a;
            }
            lr = nxt;
        }
        // 2. The values: the same tiles, the same groups, mixed ascending.
        let v16 = direct.then(|| kv.block_bytes(false, layer, kh, blk, bn));
        if v16.is_none() {
            kv.decode_block(false, layer, kh, blk, bn, &mut sc.kf);
        }
        let mut lr = 0;
        while lr < sub {
            let (start, len0) = vis_range(sh, ctx_of[lr], blk, bn);
            if len0 == 0 {
                lr += 1;
                continue;
            }
            let mut rows = [0usize; fast::FAST_ROWS];
            let mut ends = [0usize; fast::FAST_ROWS];
            rows[0] = lr;
            ends[0] = start + len0;
            let mut n = 1;
            let mut nxt = lr + 1;
            while n < fast::FAST_ROWS && nxt < sub {
                let (s2, l2) = vis_range(sh, ctx_of[nxt], blk, bn);
                if s2 != start || l2 == 0 {
                    break;
                }
                rows[n] = nxt;
                ends[n] = s2 + l2;
                n += 1;
                nxt += 1;
            }
            let j0 = start - blk;
            let mut lev = 0usize;
            loop {
                let mut act = [0usize; fast::FAST_ROWS];
                let mut na = 0;
                let mut e = usize::MAX;
                for r in 0..n {
                    if ends[r] > start + lev {
                        act[na] = rows[r];
                        na += 1;
                        e = e.min(ends[r] - start);
                    }
                }
                if na == 0 {
                    break;
                }
                let pad = act[0];
                for a in act.iter_mut().skip(na) {
                    *a = pad;
                }
                fast::mix_rows(
                    sc,
                    &act,
                    na,
                    &alpha,
                    j0 + lev,
                    e - lev,
                    hd,
                    lev == 0,
                    v16.map(|v| &v[(j0 + lev) * hd * 2..]),
                );
                lev = e;
            }
            lr = nxt;
        }
        blk += bn;
    }
}

/// One row's visible part of the tile at `blk..blk + bn`: `(start, len)`, where
/// `start` is the absolute position of the first visible score.
///
/// A row sees up to its own context (`blk` may lie past it, for a shorter row of the
/// same call) and, with a sliding window, no further back than `n_ctx − window`.
fn vis_range(sh: &Shape<'_>, n_ctx: usize, blk: usize, bn: usize) -> (usize, usize) {
    if n_ctx <= blk {
        return (blk, 0);
    }
    let p_min = match sh.window {
        Some(w) => n_ctx.saturating_sub(w),
        None => 0,
    };
    let start = blk.max(p_min);
    let end = (blk + bn).min(n_ctx);
    (start, end.saturating_sub(start))
}

/// One sub-block `r0..r1` of [`run_rows`]' query rows.
///
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_range<O: RowSink>(
    kv: &KvCache,
    layer: usize,
    sh: &Shape<'_>,
    q: &[f32],
    t: &Task,
    r0: usize,
    r1: usize,
    sc: &mut Scratch,
    out: &mut O,
) {
    let hd = sh.head_dim;
    let gqa = sh.gqa();
    let qd = sh.q_dim();
    let kh = t.kh;
    let sub = r1 - r0;

    for lr in 0..sub {
        let r = r0 + lr;
        let (bi, g) = (r / gqa, r % gqa);
        let h = kh * gqa + g;
        let src = &q[(t.b0 + bi) * qd + h * hd..][..hd];
        let dst = &mut sc.q[lr * hd..][..hd];
        dst.copy_from_slice(src);
        if kv.rotated() {
            crate::vmath::hadamard(dst);
        }
    }
    sc.o[..sub * hd].fill(0.0);
    sc.m[..sub].fill(f32::NEG_INFINITY);
    sc.l[..sub].fill(0.0);

    let mut count = [1usize; MAX_PARTIAL_ROWS];
    let mut cur = [0usize; MAX_PARTIAL_ROWS];
    let mut ctx_of = [0usize; MAX_PARTIAL_ROWS];
    let mut max_ctx = 0;
    // The inner timers accumulate here and reach the shared counters once per call. They
    // wrap a whole tile's score pass and a whole tile's mix pass, not a row: at 4096
    // positions a per-row timer costs as much as the 64-key dot it times (one timer per
    // row per tile measured 25 % of the pass), and a tile is the unit the work comes in.
    // The mix bucket contains the softmax's max and exponential, which run per row
    // between the mix's pieces in the same loop.
    let (mut ns_scores, mut ns_mix, mut ns_dec) = (0u64, 0u64, 0u64);
    for lr in 0..sub {
        let b = t.b0 + (r0 + lr) / gqa;
        let n_ctx = sh.start_pos + b + 1;
        ctx_of[lr] = n_ctx;
        count[lr] = row_chunks(sh, n_ctx);
        max_ctx = max_ctx.max(n_ctx);
    }

    // Two rows can share one pass over a widened tile with the register-blocked
    // score kernel, ten percent short of doubling the FMAs it issues.
    let pairs = {
        #[cfg(target_arch = "x86_64")]
        {
            hd.is_multiple_of(8) && crate::cpu_has!("avx2") && crate::cpu_has!("fma")
        }
        // On aarch64 the shared pass exists for the f16-direct tile; the widened
        // tile keeps the one-row kernel (an f32 cache is the rare path).
        #[cfg(target_arch = "aarch64")]
        {
            hd.is_multiple_of(8) && f16_direct(kv, sh)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = hd;
            false
        }
    };
    let p_end = t.p1.min(max_ctx);
    // The cache's own f16 halves can be read in the kernels' registers, as the
    // single-row path does: the widened f32 tile is then never built, and the rows
    // read half the bytes. That only pays once a row's visible context is long
    // enough that the extra `vcvtph2ps` work is cheaper than the bytes it saves
    // (`DIRECT_MIN_VIS`); shorter prefill keeps the widened tile. On aarch64 the
    // widening itself is element-at-a-time, so the halves are always read in the
    // kernel when the cache is f16. The per-row arithmetic is unchanged either way —
    // the f16 kernels are the same FMAs in the same order (their test pins them to
    // the widened ones bit for bit), so a row keeps its bits whichever path it takes.
    // A sliding window caps what any row can see, so the threshold is judged on the
    // visible span, not the whole context: a 512-position window keeps the widened
    // tile at any context length on x86 (aarch64 has no threshold).
    let direct = {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let vis_span = match sh.window {
                Some(w) => max_ctx.min(w),
                None => max_ctx,
            };
            prefill_direct(kv, sh, vis_span)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    };
    // No row of the task sees further back than the first row's window start; tiles
    // before it contribute nothing, so they are never decoded.
    let p_min_task = match sh.window {
        Some(w) => (sh.start_pos + t.b0 + 1).saturating_sub(w),
        None => 0,
    };
    let mut blk = t.p0.max(p_min_task / BLOCK * BLOCK);
    while blk < p_end {
        let bn = ((blk / BLOCK + 1) * BLOCK - blk).min(p_end - blk);
        // The tile is walked in two phases over the same widened buffer: keys first,
        // scored for every row, then values, mixed into every row's accumulator. One
        // resident tile the size of `bn × hd` floats plus the task's banked scores is
        // what keeps the row loop in L1; widening both K and V at once does not fit,
        // and every row then reads both tiles from L2.
        let k16 = direct.then(|| kv.block_bytes(true, layer, kh, blk, bn));
        if k16.is_none() {
            let td = crate::prof::detail().then(std::time::Instant::now);
            kv.decode_block(true, layer, kh, blk, bn, &mut sc.kf);
            if let Some(t) = td {
                ns_dec += t.elapsed().as_nanos() as u64;
            }
        }
        let ts = crate::prof::detail().then(std::time::Instant::now);
        let mut lr = 0;
        while lr < sub {
            let (start, len) = vis_range(sh, ctx_of[lr], blk, bn);
            // A second row can ride the same pass when it sees the tile from the same
            // position: rows of one batch row always do, and different batch rows do
            // when their windows happen to start together.
            //
            // On aarch64's F16-direct tile four rows can, when all four see exactly the
            // same positions: the four-row body keeps sixteen accumulation chains in
            // flight where the pair keeps eight, and it reads the K tile once for four
            // rows instead of once per pair (S4.9t).
            let mut rows = 1usize;
            let mut len1 = 0usize;
            if pairs {
                if four_rows_see_the_same(sh, &ctx_of, lr, sub, blk, bn) {
                    rows = 4;
                } else if lr + 1 < sub && vis_range(sh, ctx_of[lr + 1], blk, bn).0 == start {
                    rows = 2;
                    len1 = vis_range(sh, ctx_of[lr + 1], blk, bn).1;
                }
            }
            if len + len1 > 0 {
                score_tile(
                    sc,
                    pairs,
                    lr,
                    rows,
                    len1,
                    start - blk,
                    len,
                    hd,
                    sh.scale,
                    k16,
                );
            }
            lr += rows;
        }
        if let Some(t) = ts {
            ns_scores += t.elapsed().as_nanos() as u64;
        }
        // The values replace the keys in the same buffer; each row's own chunk
        // boundaries cut its banked scores into the same pieces as before. Within the
        // tile `pos` walks this row's chunks: usually once, twice when a chunk
        // boundary falls inside the tile.
        let v16 = direct.then(|| kv.block_bytes(false, layer, kh, blk, bn));
        if v16.is_none() {
            let td = crate::prof::detail().then(std::time::Instant::now);
            kv.decode_block(false, layer, kh, blk, bn, &mut sc.kf);
            if let Some(t) = td {
                ns_dec += t.elapsed().as_nanos() as u64;
            }
        }
        let tm = crate::prof::detail().then(std::time::Instant::now);
        // Two rows of one KV head that see the same positions are mixed in one pass
        // over V, with each row's own accumulators and its own ascending positions —
        // the sharing cannot move a bit, and the second row's FMAs ride loads the first
        // row already paid for. On x86 both tiles have the pair body; on aarch64 the
        // F16-direct tile has one (S4.9s); the other architectures mix a row at a time.
        #[cfg(target_arch = "x86_64")]
        let pair_mix = pairs;
        #[cfg(target_arch = "aarch64")]
        let pair_mix = v16.is_some() && pairs;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let pair_mix = false;
        // Four rows over one V tile is an aarch64 body (S4.9t).
        #[cfg(target_arch = "aarch64")]
        let quad_mix = v16.is_some() && pairs;
        #[cfg(not(target_arch = "aarch64"))]
        let quad_mix = false;
        let mut lr = 0;
        while lr < sub {
            let n_ctx = ctx_of[lr];
            if n_ctx <= blk {
                lr += 1;
                continue;
            }
            let (vis_start, vis_len) = vis_range(sh, n_ctx, blk, bn);
            let vis_end = vis_start + vis_len;
            // The row's canonical chunks are the even cut of its visible positions.
            let v0 = visible_start(sh, n_ctx);
            let visible = n_ctx - v0;
            // A pair needs the same visible range, hence the same chunk boundaries.
            // On aarch64's F16-direct tile four rows can ride one pass over V when all
            // four agree on the context and so on the chunks (S4.9t).
            let two =
                pair_mix && lr + 1 < sub && ctx_of[lr + 1] == n_ctx && count[lr + 1] == count[lr];
            // Four rows can ride one pass over V when all four see the same context, so
            // their pieces and their chunk boundaries coincide: cutting a row anywhere
            // but its own boundary would give its soft-max an extra step and move a bit.
            let four = quad_mix
                && lr + 3 < sub
                && (1..4).all(|d| ctx_of[lr + d] == n_ctx && count[lr + d] == count[lr]);
            let mut pos = blk;
            while pos < vis_end {
                let ci = cur[lr];
                let chunk_end = v0 + (ci + 1) * visible / count[lr];
                let end = vis_end.min(chunk_end);
                let start = pos.max(vis_start);
                if start < end {
                    let j_start = start - blk;
                    let len = end - start;
                    let base = start - vis_start;
                    let mut alpha = [1.0f32; 4];
                    let rows = if four {
                        4
                    } else if two {
                        2
                    } else {
                        1
                    };
                    softmax_rows(sc, lr, base, len, rows, &mut alpha);
                    match v16 {
                        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                        Some(v) => {
                            let halves = &v[j_start * hd * 2..];
                            // A piece never crosses a chunk boundary (it is cut at it),
                            // so the pair body has nothing to bank and the caller does.
                            #[cfg(target_arch = "aarch64")]
                            if four {
                                let s = &sc.s_tile[lr * BLOCK + base..];
                                // SAFETY: the F16 cache is direct (`f16_direct`); `v`
                                // holds the tile's halves and `s` four rows of scores at
                                // `BLOCK`.
                                unsafe {
                                    peregrine_kernels::neon::mix_f16_quad(
                                        &mut sc.o[lr * hd..],
                                        &alpha,
                                        s,
                                        BLOCK,
                                        len,
                                        halves,
                                        hd,
                                    )
                                };
                            } else if two {
                                let s0 = &sc.s_tile[lr * BLOCK + base..][..len];
                                let s1 = &sc.s_tile[(lr + 1) * BLOCK + base..][..len];
                                let (lo, hi) = sc.o.split_at_mut((lr + 1) * hd);
                                // SAFETY: NEON is this target's baseline (`f16_direct`
                                // also checked it); `v` holds the tile's halves and each
                                // `s` its row's scores for the positions mixed.
                                unsafe {
                                    peregrine_kernels::neon::mix_f16_pair(
                                        &mut lo[lr * hd..],
                                        &mut hi[..hd],
                                        alpha[0],
                                        alpha[1],
                                        s0,
                                        s1,
                                        halves,
                                        hd,
                                    )
                                };
                            } else {
                                let s = &sc.s_tile[lr * BLOCK + base..][..len];
                                // SAFETY: F16 directness checked; `v` holds the tile's
                                // halves and `s` the row's scores for the positions mixed.
                                unsafe {
                                    mix_f16(&mut sc.o[lr * hd..][..hd], alpha[0], s, halves, hd)
                                }
                            }
                            #[cfg(target_arch = "x86_64")]
                            if two {
                                let s0 = &sc.s_tile[lr * BLOCK + base..][..len];
                                let s1 = &sc.s_tile[(lr + 1) * BLOCK + base..][..len];
                                let (lo, hi) = sc.o.split_at_mut((lr + 1) * hd);
                                // SAFETY: the F16 cache is direct (`f16_direct`), `v`
                                // holds the tile's halves and each `s` its row's scores.
                                unsafe {
                                    mix_f16_pair_same(
                                        &mut lo[lr * hd..],
                                        &mut hi[..hd],
                                        alpha[0],
                                        alpha[1],
                                        s0,
                                        s1,
                                        halves,
                                        hd,
                                    )
                                };
                            } else {
                                let s = &sc.s_tile[lr * BLOCK + base..][..len];
                                // SAFETY: F16 directness checked; `v` holds the tile's
                                // halves and `s` the row's scores for the positions mixed.
                                unsafe {
                                    mix_f16(&mut sc.o[lr * hd..][..hd], alpha[0], s, halves, hd)
                                }
                            }
                            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                            {
                                let s = &sc.s_tile[lr * BLOCK + base..][..len];
                                // SAFETY: F16 directness checked; `v` holds the tile's
                                // halves and `s` the row's scores for the positions mixed.
                                unsafe {
                                    mix_f16(&mut sc.o[lr * hd..][..hd], alpha[0], s, halves, hd)
                                }
                            }
                        }
                        _ => {
                            // The widened tile has the same two-row body as the F16
                            // one: the loaded f32 value feeds both rows' FMAs.
                            #[cfg(target_arch = "x86_64")]
                            if two {
                                let s0 = &sc.s_tile[lr * BLOCK + base..][..len];
                                let s1 = &sc.s_tile[(lr + 1) * BLOCK + base..][..len];
                                let (lo, hi) = sc.o.split_at_mut((lr + 1) * hd);
                                // SAFETY: the widened tile `kf` holds the values and
                                // each `s` its row's scores for the positions mixed.
                                unsafe {
                                    mix_pair_same(
                                        &mut lo[lr * hd..],
                                        &mut hi[..hd],
                                        alpha[0],
                                        alpha[1],
                                        s0,
                                        s1,
                                        &sc.kf[j_start * hd..],
                                        hd,
                                    )
                                };
                            } else {
                                let s = &sc.s_tile[lr * BLOCK + base..][..len];
                                mix(
                                    &mut sc.o[lr * hd..][..hd],
                                    alpha[0],
                                    s,
                                    &sc.kf[j_start * hd..],
                                    hd,
                                );
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                let s = &sc.s_tile[lr * BLOCK + base..][..len];
                                mix(
                                    &mut sc.o[lr * hd..][..hd],
                                    alpha[0],
                                    s,
                                    &sc.kf[j_start * hd..],
                                    hd,
                                );
                            }
                        }
                    }
                }
                pos = end;
                if end == chunk_end && ci + 1 < count[lr] {
                    // The chunk is complete: bank it and start the next one fresh,
                    // exactly as a task of its own would have.
                    let banked = if four {
                        4
                    } else if two {
                        2
                    } else {
                        1
                    };
                    for &row in &[lr, lr + 1, lr + 2, lr + 3][..banked] {
                        save_part(sc, row, ci, hd);
                        sc.o[row * hd..(row + 1) * hd].fill(0.0);
                        sc.m[row] = f32::NEG_INFINITY;
                        sc.l[row] = 0.0;
                        cur[row] = ci + 1;
                    }
                }
            }
            lr += if four {
                4
            } else if two {
                2
            } else {
                1
            };
        }
        if let Some(t) = tm {
            ns_mix += t.elapsed().as_nanos() as u64;
        }
        blk += bn;
    }
    for lr in 0..sub {
        save_part(sc, lr, cur[lr], hd);
    }

    let rotated = kv.rotated();
    for lr in 0..sub {
        let r = r0 + lr;
        let (bi, g) = (r / gqa, r % gqa);
        let h = kh * gqa + g;
        let mut parts = [(&[][..], f32::NEG_INFINITY, 0.0f64); MAX_CHUNKS];
        for (ci, p) in parts.iter_mut().enumerate().take(count[lr]) {
            let slot = lr * MAX_CHUNKS + ci;
            *p = (
                &sc.partial_o[slot * hd..(slot + 1) * hd],
                sc.partial_m[slot],
                sc.partial_l[slot],
            );
        }
        let sink = sh.sinks.map(|s| s[h]);
        combine_row(&parts[..count[lr]], sink, rotated, out.head(bi, h));
    }
    if crate::prof::detail() {
        crate::prof::add_shared(crate::prof::Slot::AttnScores, ns_scores);
        crate::prof::add_shared(crate::prof::Slot::AttnMix, ns_mix);
        crate::prof::add_shared(crate::prof::Slot::AttnDecode, ns_dec);
    }
}

/// One row of one KV head at its own position, as one task: the row is cut into its
/// own canonical position chunks, each chunk is computed by [`run`] (so a one-row chunk
/// takes the F16-direct kernel when the cache allows it), and the chunks are merged by
/// [`combine_row`] in chunk order.
///
/// This is exactly what [`attend_scaled_window`]'s split path does for a solo row — the
/// same chunk boundaries, the same per-chunk kernel choice, the same merge order — so a
/// row's bits in a shared call are its solo bits, while the tasks of all the decoding
/// users still share one parallel region.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_row<O: RowSink>(
    kv: &KvCache,
    layer: usize,
    sh: &Shape<'_>,
    q_row: &[f32],
    row: usize,
    kh: usize,
    pos: usize,
    sc: &mut Scratch,
    out: &mut O,
) {
    let hd = sh.head_dim;
    let gqa = sh.gqa();
    let n_ctx = pos + 1;
    // The row alone, at its own position: the shape a solo decode would see.
    let one = Shape {
        nb: 1,
        start_pos: pos,
        // This row's own flag: `sh.fast` is indexed by batch row.
        fast: sh.fast.map(|f| &f[row..row + 1]),
        ..*sh
    };
    // A fast prompt row's pieces are the absolute key tiles, never position chunks
    // (S5.4j): one walk over its whole visible range, exactly as [`plan`] gives a solo
    // fast row one chunk. Cutting it here would make its bits depend on whether the row
    // was alone in its cache in this pass.
    let count = if fast_row(&one, 0) {
        1
    } else {
        row_chunks(sh, n_ctx)
    };
    // The same visible-position cut [`task`] makes for a solo row.
    let v0 = visible_start(&one, n_ctx);
    let visible = n_ctx - v0;
    for ci in 0..count {
        let t = Task {
            kh,
            b0: 0,
            b1: 1,
            p0: v0 + ci * visible / count,
            p1: v0 + (ci + 1) * visible / count,
        };
        if fast_row(&one, 0) {
            run_fast(kv, layer, &one, q_row, &t, sc);
        } else {
            run(kv, layer, &one, q_row, &t, sc);
        }
        for g in 0..gqa {
            save_part(sc, g, ci, hd);
        }
    }
    for g in 0..gqa {
        let h = kh * gqa + g;
        let mut parts = [(&[][..], f32::NEG_INFINITY, 0.0f64); MAX_CHUNKS];
        for (ci, p) in parts.iter_mut().enumerate().take(count) {
            let slot = g * MAX_CHUNKS + ci;
            *p = (
                &sc.partial_o[slot * hd..(slot + 1) * hd],
                sc.partial_m[slot],
                sc.partial_l[slot],
            );
        }
        let sink = sh.sinks.map(|s| s[h]);
        combine_row(&parts[..count], sink, kv.rotated(), out.head(0, h));
    }
}

/// Whether four rows of the task — `lr .. lr + 4` — see exactly the same part of the
/// tile, so one pass can score all four (`scores_f16_tile4`).
///
/// The four-row body shares only the loaded key; each row's tree and accumulators are
/// its own, so which rows a pass groups cannot move a bit. It needs all four to have
/// the same visible range because the kernel writes four equal-length rows.
fn four_rows_see_the_same(
    sh: &Shape<'_>,
    ctx_of: &[usize; MAX_PARTIAL_ROWS],
    lr: usize,
    sub: usize,
    blk: usize,
    bn: usize,
) -> bool {
    if !cfg!(target_arch = "aarch64") || lr + 4 > sub {
        return false;
    }
    let want = vis_range(sh, ctx_of[lr], blk, bn);
    (1..4).all(|d| vis_range(sh, ctx_of[lr + d], blk, bn) == want)
}

/// Score `len` positions of the tile for `rows` rows of the task's scratch, starting
/// at row `lr` and tile position `j_start`. A two-row pass may see different lengths
/// (`len` for the first row, `len1` for the second); a four-row pass needs all four
/// equal.
///
/// The rows' score rows land in their own `s_tile` slices. Only the loaded K vectors
/// are shared; each row's accumulator and reduction order are its own, so the
/// grouping cannot move a bit.
#[allow(clippy::too_many_arguments)]
fn score_tile(
    sc: &mut Scratch,
    pairs: bool,
    lr: usize,
    rows: usize,
    len1: usize,
    j_start: usize,
    len: usize,
    hd: usize,
    scale: f32,
    // The tile's f16 halves in the cache, when the f16 kernels may read them; `None`
    // means the widened f32 tile in `sc.kf`.
    k16: Option<&[u8]>,
) {
    // The four-row pass reads the tile's own halves and four rows of the scratch; it
    // is handled before the row-wise slices are taken so all four rows are one slice.
    #[cfg(target_arch = "aarch64")]
    if rows == 4 {
        let k = k16.expect("the four-row tile needs the F16 cache");
        let k = &k[j_start * hd * 2..];
        // SAFETY: the tier's F16 features are checked (`f16_direct`) and `hd % 8 == 0`
        // is part of `pairs`; `k` holds the tile's keys and `s_tile` four rows at `BLOCK`.
        unsafe {
            peregrine_kernels::neon::scores_f16_tile4(
                &sc.q[lr * hd..],
                k,
                len,
                hd,
                scale,
                &mut sc.s_tile[lr * BLOCK..],
                BLOCK,
            )
        };
        return;
    }
    let Scratch { kf, q, s_tile, .. } = sc;
    let (lo, mid) = s_tile.split_at_mut((lr + 1) * BLOCK);
    let s0 = &mut lo[lr * BLOCK..lr * BLOCK + len];
    let s1 = &mut mid[..len1];
    if let Some(k) = k16 {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            // A window can start inside the tile, so the visible keys begin at
            // `j_start` in the cache's own halves — the same position the widened
            // tile's `kf[j_start * hd..]` starts at.
            let k = &k[j_start * hd * 2..];
            if rows == 2 && pairs {
                // SAFETY: the tier's F16/FMA features are checked (`f16_direct`);
                // `k` holds `max(len, len1)` keys.
                unsafe {
                    scores_f16_pair(
                        &q[lr * hd..],
                        &q[(lr + 1) * hd..],
                        k,
                        len,
                        len1,
                        hd,
                        scale,
                        s0,
                        s1,
                    )
                };
                return;
            }
            // SAFETY: F16 directness checked; `k` holds the tile's keys.
            unsafe {
                scores_f16(&q[lr * hd..][..hd], k, len, hd, scale, s0);
                for r in 1..rows {
                    scores_f16(
                        &q[(lr + r) * hd..][..hd],
                        k,
                        len,
                        hd,
                        scale,
                        &mut s_tile[(lr + r) * BLOCK..(lr + r) * BLOCK + len],
                    );
                }
            }
            return;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = k;
    }
    let kf = &kf[j_start * hd..];
    #[cfg(target_arch = "x86_64")]
    if rows == 2 && pairs {
        // SAFETY: AVX2+FMA checked; `kf` holds the rest of the tile and each `s` row
        // holds `len`/`len1` floats.
        unsafe {
            scores_pair(
                &q[lr * hd..],
                &q[(lr + 1) * hd..],
                kf,
                len,
                len1,
                hd,
                scale,
                s0,
                s1,
            )
        };
        return;
    }
    let _ = (pairs, len1);
    scores(&q[lr * hd..][..hd], kf, len, hd, scale, s0);
    for r in 1..rows {
        scores(
            &q[(lr + r) * hd..][..hd],
            kf,
            len,
            hd,
            scale,
            &mut s_tile[(lr + r) * BLOCK..(lr + r) * BLOCK + len],
        );
    }
}

/// The online softmax of `rows` consecutive rows of the task over the same piece of
/// the tile: each row's running maximum, the exponentials in place, its ascending
/// `f64` denominator and its rescale factor `α`.
///
/// With four rows the four denominators advance interleaved (`exp_rows4_sums`, S4.9r's
/// encoder body); each row still folds its own exponentials in ascending key order
/// into its own `f64`, so no row's bits depend on the rows beside it.
///
/// The four-row body is aarch64's only. There `exp_rows4_sums` and
/// [`crate::vmath::exp_shift_sum`] define the same denominator (one ascending `f64`
/// fold per row). On x86-64 `exp_shift_sum` folds eight-value groups, a different
/// defined order, so the two are not interchangeable and four rows there are four
/// single rows.
fn softmax_rows(
    sc: &mut Scratch,
    lr: usize,
    base: usize,
    len: usize,
    rows: usize,
    alpha: &mut [f32; 4],
) {
    #[cfg(target_arch = "aarch64")]
    if rows == 4 {
        let mut m_new = [0.0f32; 4];
        let mut sums = [0.0f64; 4];
        for (r, m) in m_new.iter_mut().enumerate() {
            let row = lr + r;
            let s = &sc.s_tile[row * BLOCK + base..row * BLOCK + base + len];
            *m = sc.m[row].max(crate::vmath::max(s));
        }
        peregrine_kernels::bidir::exp_rows4_sums(
            &mut sc.s_tile[lr * BLOCK + base..],
            BLOCK,
            len,
            m_new,
            &mut sums,
        );
        for r in 0..4 {
            let row = lr + r;
            alpha[r] = crate::vmath::det_exp(sc.m[row] - m_new[r]);
            sc.l[row] = sc.l[row] * alpha[r] as f64 + sums[r];
            sc.m[row] = m_new[r];
        }
        return;
    }
    for (r, row) in (lr..lr + rows).enumerate() {
        let s = &mut sc.s_tile[row * BLOCK + base..row * BLOCK + base + len];
        let m_old = sc.m[row];
        let m_new = m_old.max(crate::vmath::max(s));
        let psum = crate::vmath::exp_shift_sum(s, m_new);
        alpha[r] = crate::vmath::det_exp(m_old - m_new);
        sc.l[row] = sc.l[row] * alpha[r] as f64 + psum;
        sc.m[row] = m_new;
    }
}

/// Store row `lr`'s current `(o, m, l)` as its partial for chunk `ci`.
fn save_part(sc: &mut Scratch, lr: usize, ci: usize, hd: usize) {
    let slot = lr * MAX_CHUNKS + ci;
    sc.partial_o[slot * hd..(slot + 1) * hd].copy_from_slice(&sc.o[lr * hd..(lr + 1) * hd]);
    sc.partial_m[slot] = sc.m[lr];
    sc.partial_l[slot] = sc.l[lr];
}

/// `out = o / l`, un-rotated when the cache is rotated. A row that saw no position
/// (impossible for a full-range task; possible for a position chunk past it) yields
/// zeros, which a combine weighs at zero anyway.
///
/// With an attention sink, the denominator also carries the sink's weight:
/// `out = o / (l + exp(sink − m))`. That is exactly what ggml computes: the sink
/// is a logit that joins the softmax without a value of its own
/// (`ggml/src/ggml-cpu/ops.cpp:5883-5894`, `ggml_soft_max_add_sinks`), so the
/// denominator gains `exp(sink − max)` while the numerator keeps the values'
/// weights. The maximum used here is the one over the positions, not the one over
/// the positions *and* the sink: dividing the numerator and the denominator by
/// `exp(m − max)` leaves the ratio unchanged, so the two agree whichever is larger.
pub fn finish_row(o: &[f32], l: f64, m: f32, sink: Option<f32>, rotated: bool, out: &mut [f32]) {
    if l <= 0.0 {
        out.fill(0.0);
        return;
    }
    let denom = match sink {
        Some(s) => l + f64::from(peregrine_kernels::vmath::exp_scalar(s - m)),
        None => l,
    };
    if denom <= 0.0 {
        out.fill(0.0);
        return;
    }
    let inv = (1.0 / denom) as f32;
    for (d, &x) in out.iter_mut().zip(o) {
        *d = x * inv;
    }
    if rotated {
        crate::vmath::hadamard_inv(out);
    }
}

/// Merge partial `(o_i, m_i, l_i)` results of one row (one per position chunk) into
/// its output — uzu's `combine.metal`: weights `exp(m_i − m*)`.
///
/// The attention sink, if the layer has one, is added to the merged denominator
/// once — after the merge, as ggml adds it to the whole row's softmax at once.
pub fn combine_row(
    parts: &[(&[f32], f32, f64)],
    sink: Option<f32>,
    rotated: bool,
    out: &mut [f32],
) {
    let m_star = parts
        .iter()
        .filter(|p| p.2 > 0.0)
        .map(|p| p.1)
        .fold(f32::NEG_INFINITY, f32::max);
    out.fill(0.0);
    let mut l = 0.0f64;
    for &(o, m, li) in parts {
        if li <= 0.0 {
            continue;
        }
        let w = crate::vmath::det_exp(m - m_star);
        l += li * w as f64;
        for (d, &x) in out.iter_mut().zip(o) {
            *d += x * w;
        }
    }
    let denom = match sink {
        Some(s) => l + f64::from(peregrine_kernels::vmath::exp_scalar(s - m_star)),
        None => l,
    };
    if denom <= 0.0 {
        out.fill(0.0);
        return;
    }
    let inv = (1.0 / denom) as f32;
    out.iter_mut().for_each(|d| *d *= inv);
    if rotated {
        crate::vmath::hadamard_inv(out);
    }
}

/// Whether the F16 cache can be consumed directly by the register-blocked kernels.
///
/// On x86 F16C widens the stored halves in registers and AVX2+FMA do the arithmetic;
/// on aarch64 `vcvt_f32_f16` widens and the baseline NEON FMA does it. The widening
/// itself is exact, so the same product sees the same bits on either path. A sliding
/// window only moves where the visible range starts: the decode path skips every tile
/// before it and offsets into the tile that holds it, so the same kernels read the
/// same positions the widened path would.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn f16_direct(kv: &KvCache, sh: &Shape<'_>) -> bool {
    let common = kv.prec() == crate::kv::KvPrec::F16 && sh.head_dim.is_multiple_of(8);
    #[cfg(target_arch = "x86_64")]
    {
        common && crate::cpu_has!("avx2") && crate::cpu_has!("fma") && crate::cpu_has!("f16c")
    }
    #[cfg(target_arch = "aarch64")]
    {
        common && peregrine_kernels::cpu::have_neon()
    }
}

/// Whether a prefill task whose longest row sees `vis` positions reads the F16 cache
/// directly instead of widening a tile (the x86 `DIRECT_MIN_VIS` crossover).
///
/// The threshold is an x86 trade (a vector `vcvtph2ps` against the bytes saved). On
/// aarch64 widening is element-at-a-time, so reading the halves in the kernel wins at
/// every length.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn prefill_direct(kv: &KvCache, sh: &Shape<'_>, vis: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        vis >= DIRECT_MIN_VIS && f16_direct(kv, sh)
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = vis;
        f16_direct(kv, sh)
    }
}

/// Whether the sixteen-lane value-mix kernels can run for this head width.
///
/// Only the mix is widened. The score kernels keep the AVX2 tree on every tier: a
/// 512-bit score row needs a 512-bit horizontal reduction, and on this host
/// (Ryzen AI 9 HX 370) that costs more than the halved FMA and load count saves —
/// measured at every visible length, the zmm score kernels were 30–70% slower than
/// the AVX2 ones at head width 64. Keeping one score tree everywhere also makes a
/// row's score bits independent of the path by construction. The mix is
/// per-element and conversion-bound, so widening it cannot move a bit either; the
/// tier's `mix` golden is the AVX2 one.
#[cfg(target_arch = "x86_64")]
#[inline]
fn avx512_attn(head_dim: usize) -> bool {
    head_dim.is_multiple_of(16)
        && crate::cpu_has!("avx512f")
        && crate::cpu_has!("avx512dq")
        && crate::cpu_has!("avx2")
        && crate::cpu_has!("fma")
        && crate::cpu_has!("f16c")
}

/// The fast prompt mode's attention order (S5.4j).
///
/// The mode is lossy by definition, and its prompt rows already have their own bits
/// (the 8-bit weight form). A *different summation order of the same sums* moves a
/// result by rounding only — orders of magnitude below the form's error — so a prompt
/// row of the mode may score its keys, fold its soft-max denominator and mix its
/// values in a tiled order that shares each loaded K and V vector between up to
/// `FAST_ROWS` query rows. Generated rows, embeddings, rerank, decisions and every
/// pass with the option off keep the exact engine's attention, bit for bit.
///
/// # The definition
///
/// A fast prompt row's attention is defined by these three pieces, and by nothing
/// else. Each is implemented twice: `scores_twin`, `mix_twin` and
/// `denom_twin` are the definition (plain loops, ADR 0005), and the `*_zmm`
/// bodies must equal them bit for bit at every length (the tests sweep the grid
/// S5.2b used). The **order is a function of the row's own visible key range alone**:
/// pieces are the `BLOCK`-key tiles on the absolute position grid (and a sliding
/// window's own start), never a prefill chunk's or a task's boundaries, so a row's
/// bits cannot depend on the batch, the chunk, the pass budget or the pool.
///
/// 1. **Scores.** `head_dim` is walked in ascending 16-lane blocks; lane `l` keeps
///    one `f32` accumulator and takes one fused multiply-add per block,
///    `acc[l] = fma(q[l], k[l], acc[l])` (dimensions past `head_dim` count as a zero
///    product). The 16 lanes are then reduced by
///    `T16(v) = T8(v[l] + v[l+8])`, where
///    `T8(w) = ((w0+w1)+(w2+w3)) + ((w4+w5)+(w6+w7))`, and the result is multiplied
///    by the soft-max scale (a separate `f32` multiply, as the exact kernels do).
/// 2. **Soft-max.** The maximum is over the piece and is order-free. Each exponential
///    is the engine's own ([`crate::vmath::det_exp`], which is `exp8`'s scalar twin
///    lane for lane). The denominator folds the piece in **16-key groups from the
///    piece's first key, ascending**: a group's exponentials are reduced with `T16`
///    in `f32`, and the group sums are added into the row's `f64` denominator in
///    ascending order. The online rescale is the engine's (`l = l·α + Σ`,
///    `α = exp(m_old − m_new)`), and the row's pieces are the tiles above, so the
///    number of rescales is a function of the row's range too.
/// 3. **Value mix.** Per element, one `f32` chain in ascending key order,
///    `o = fma(p_j, v_j, α·o)`. This is the exact tier's own mix formula; what the
///    fast order changes here is only that up to `FAST_ROWS` rows share one pass
///    over V (each row keeps its own chain, so the sharing moves no bit).
///
/// Nothing else changes: the causal and sliding-window ranges, the sinks, the
/// rotation, the head and KV-head geometry and the merge of the pieces' `(o, m, l)`
/// are the exact engine's. A fast row is never cut into position chunks by the pool
/// ([`plan`] gives it one), because its pieces are the tiles.
pub mod fast {
    use super::*;

    /// Query rows one fast body shares a pass over K or V between.
    ///
    /// Four rows × four keys is the widest score tile that keeps its sixteen 16-lane
    /// accumulators, the four query vectors and the four key vectors in the ZMM file at
    /// once; the same four rows fill sixteen accumulators over a 64-dimension value
    /// block, so one pass over V serves all four rows' mixes.
    pub const FAST_ROWS: usize = 4;

    /// Lanes of the fast order's score and denominator tree.
    const LANES: usize = 16;

    /// Whether this tier computes a fast prompt row's attention in the fast order.
    ///
    /// A per-tier property, like the form's formats: the AVX-512 tier yes; AVX2,
    /// aarch64 and scalar no (their fast-mode goldens must not move), so the mode
    /// behaves there exactly as it did before S5.4j.
    #[inline]
    pub fn tier() -> bool {
        peregrine_kernels::table::MatKernels::get().fast_attn()
    }

    /// Whether the 16-lane bodies can run: the tier, and a head width they cover.
    #[inline]
    pub(super) fn body_ok(hd: usize) -> bool {
        hd.is_multiple_of(LANES) && tier()
    }

    /// `T8`: the eight-lane tree `((0+1)+(2+3)) + ((4+5)+(6+7))`.
    #[inline]
    fn t8(w: &[f32; 8]) -> f32 {
        ((w[0] + w[1]) + (w[2] + w[3])) + ((w[4] + w[5]) + (w[6] + w[7]))
    }

    /// `T16(v) = T8(v[l] + v[l+8])`.
    #[inline]
    fn t16(v: &[f32; LANES]) -> f32 {
        let mut h = [0.0f32; 8];
        for (l, x) in h.iter_mut().enumerate() {
            *x = v[l] + v[l + 8];
        }
        t8(&h)
    }

    /// The definition's score of `vis` keys of one row: `s[j] = 16-lane tree · scale`.
    pub fn scores_twin(q: &[f32], kf: &[f32], vis: usize, hd: usize, scale: f32, s: &mut [f32]) {
        for (j, sj) in s.iter_mut().enumerate().take(vis) {
            let k = &kf[j * hd..(j + 1) * hd];
            let mut acc = [0.0f32; LANES];
            let mut d = 0;
            while d < hd {
                let n = (hd - d).min(LANES);
                for (l, a) in acc.iter_mut().enumerate() {
                    // Dimensions past `head_dim` are a zero product, and neither
                    // operand is read there.
                    if l < n {
                        *a = q[d + l].mul_add(k[d + l], *a);
                    }
                }
                d += LANES;
            }
            *sj = t16(&acc) * scale;
        }
    }

    /// The definition's value mix of one row, `o = α·o + Σ_j p_j·v_j` ascending.
    pub fn mix_twin(o: &mut [f32], alpha: f32, p: &[f32], vf: &[f32], hd: usize) {
        for (d, od) in o.iter_mut().enumerate().take(hd) {
            let mut acc = alpha * *od;
            for (j, &w) in p.iter().enumerate() {
                acc = w.mul_add(vf[j * hd + d], acc);
            }
            *od = acc;
        }
    }

    /// The definition's denominator of one piece: `(m_new, Σ, α)`.
    pub fn denom_twin(s: &mut [f32], m_old: f32) -> (f32, f64, f32) {
        let mut m_new = m_old;
        for &x in s.iter() {
            m_new = m_new.max(x);
        }
        let mut sum = 0.0f64;
        let mut g = 0;
        while g < s.len() {
            let n = (s.len() - g).min(LANES);
            let mut v = [0.0f32; LANES];
            for (l, vl) in v.iter_mut().enumerate() {
                let x = if l < n {
                    // The engine's own exponential (`exp8`'s scalar twin), which is
                    // what the 16-lane body's `exp16` computes lane for lane.
                    let e = peregrine_kernels::vmath::exp_scalar(s[g + l] - m_new);
                    s[g + l] = e;
                    e
                } else {
                    0.0
                };
                *vl = x;
            }
            sum += t16(&v) as f64;
            g += LANES;
        }
        (m_new, sum, crate::vmath::det_exp(m_old - m_new))
    }

    /// [`scores_twin`] on sixteen lanes, [`FAST_ROWS`] rows × four keys at a time.
    ///
    /// # Safety
    /// AVX-512F/DQ and FMA must be available; `hd` must be a multiple of 16; `q` must
    /// hold `hd` floats at every `rows[r] · hd`, `kf` the level's keys at `hd`, and
    /// `s` `len` floats at every `rows[r] · sstride`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq,fma")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn scores_zmm(
        q: *const f32,
        hd: usize,
        kf: *const f32,
        rows: &[usize; FAST_ROWS],
        nrows: usize,
        len: usize,
        scale: f32,
        s: *mut f32,
        sstride: usize,
        sofs: usize,
    ) {
        use std::arch::x86_64::*;
        assert!(
            hd.is_multiple_of(LANES),
            "the fast body takes whole 16-lane blocks"
        );
        let nblk = hd / LANES;
        let sc = _mm512_set1_ps(scale);
        let mut j = 0;
        // Four keys per pass: sixteen accumulators (four rows × four keys) stay in the
        // ZMM file beside the four query vectors of the block in flight, and four keys'
        // trees share one reduction.
        while j + 4 <= len {
            let mut a = [[_mm512_setzero_ps(); 4]; FAST_ROWS];
            for b in 0..nblk {
                let d = b * LANES;
                let mut qv = [_mm512_setzero_ps(); FAST_ROWS];
                for (r, qr) in qv.iter_mut().enumerate() {
                    if r < nrows {
                        *qr = _mm512_loadu_ps(q.add(rows[r] * hd + d));
                    }
                }
                let mut kv = [_mm512_setzero_ps(); 4];
                for (u, k) in kv.iter_mut().enumerate() {
                    *k = _mm512_loadu_ps(kf.add((j + u) * hd + d));
                }
                for r in 0..FAST_ROWS {
                    for u in 0..4 {
                        a[r][u] = _mm512_fmadd_ps(qv[r], kv[u], a[r][u]);
                    }
                }
            }
            for r in 0..nrows {
                let sums = hsum16x4(a[r]);
                let dst = s.add(rows[r] * sstride + sofs + j);
                for (u, t) in sums.iter().enumerate() {
                    *dst.add(u) = *t * scale;
                }
            }
            j += 4;
        }
        while j < len {
            let mut a = [_mm512_setzero_ps(); FAST_ROWS];
            for b in 0..nblk {
                let d = b * LANES;
                let k = _mm512_loadu_ps(kf.add(j * hd + d));
                for (r, ar) in a.iter_mut().enumerate() {
                    if r < nrows {
                        *ar = _mm512_fmadd_ps(_mm512_loadu_ps(q.add(rows[r] * hd + d)), k, *ar);
                    }
                }
            }
            for r in 0..nrows {
                *s.add(rows[r] * sstride + sofs + j) = hsum16(a[r]) * scale;
            }
            j += 1;
        }
        let _ = sc;
    }

    /// Four [`hsum16`]s at once: each accumulator is folded to eight lanes, then the
    /// four folds are reduced by the transposed `hadd` tree, whose lanes are
    /// `T8(v[l] + v[l+8])` of each — the definition's `T16`, four times, bit for bit.
    ///
    /// # Safety
    /// AVX-512F/DQ and AVX2 must be available.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn hsum16x4(v: [std::arch::x86_64::__m512; 4]) -> [f32; 4] {
        use std::arch::x86_64::*;
        let fold = |x: __m512| -> __m256 {
            _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1))
        };
        let s0 = fold(v[0]);
        let s1 = fold(v[1]);
        let s2 = fold(v[2]);
        let s3 = fold(v[3]);
        let t0 = _mm256_hadd_ps(s0, s1);
        let t1 = _mm256_hadd_ps(s2, s3);
        let u = _mm256_hadd_ps(t0, t1);
        let out = _mm_add_ps(_mm256_castps256_ps128(u), _mm256_extractf128_ps(u, 1));
        let mut r = [0.0f32; 4];
        _mm_storeu_ps(r.as_mut_ptr(), out);
        r
    }

    /// `T16` of sixteen lanes in registers: the low half plus the high half, then
    /// [`crate::attn`]'s eight-lane tree.
    ///
    /// # Safety
    /// AVX-512F must be available.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn hsum16(v: std::arch::x86_64::__m512) -> f32 {
        use std::arch::x86_64::*;
        let lo = _mm512_castps512_ps256(v);
        let hi = _mm512_extractf32x8_ps(v, 1);
        hsum_grouped_ps(_mm256_add_ps(lo, hi))
    }

    /// [`mix_twin`] sixteen lanes at a time, [`FAST_ROWS`] rows over one pass of V.
    ///
    /// # Safety
    /// AVX-512F/DQ and FMA must be available; `hd` a multiple of 16; `v` `len · hd`
    /// floats; `p` `len` floats at every `rows[r] · pstride`; `o` `hd` floats at every
    /// `rows[r] · hd`. The caller pads `rows` with a valid row, so every read is in
    /// bounds; only the first `nrows` are written back.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq,fma")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn mix_zmm(
        o: *mut f32,
        hd: usize,
        alpha: *const f32,
        rows: &[usize; FAST_ROWS],
        nrows: usize,
        p: *const f32,
        pstride: usize,
        v: *const f32,
        len: usize,
    ) {
        use std::arch::x86_64::*;
        assert!(
            hd.is_multiple_of(LANES),
            "the fast body takes whole 16-lane blocks"
        );
        // Sixty-four dimensions at a time: sixteen accumulators (four rows × four lane
        // blocks), so each position's four value vectors are loaded once and each row's
        // weight is broadcast once, for sixteen fused multiply-adds. Every loop below
        // has a constant trip count, which is what keeps the accumulators in the
        // register file.
        let mut d0 = 0;
        while d0 + 64 <= hd {
            let mut acc = [[_mm512_setzero_ps(); 4]; FAST_ROWS];
            for (r, ar) in acc.iter_mut().enumerate() {
                let al = _mm512_set1_ps(*alpha.add(rows[r]));
                for (l, a) in ar.iter_mut().enumerate() {
                    *a = _mm512_mul_ps(_mm512_loadu_ps(o.add(rows[r] * hd + d0 + l * LANES)), al);
                }
            }
            for j in 0..len {
                let row = v.add(j * hd + d0);
                let mut vv = [_mm512_setzero_ps(); 4];
                for (l, x) in vv.iter_mut().enumerate() {
                    *x = _mm512_loadu_ps(row.add(l * LANES));
                }
                for (r, ar) in acc.iter_mut().enumerate() {
                    let w = _mm512_set1_ps(*p.add(rows[r] * pstride + j));
                    for (l, a) in ar.iter_mut().enumerate() {
                        *a = _mm512_fmadd_ps(w, vv[l], *a);
                    }
                }
            }
            for (r, ar) in acc.iter().enumerate().take(nrows) {
                for (l, a) in ar.iter().enumerate() {
                    _mm512_storeu_ps(o.add(rows[r] * hd + d0 + l * LANES), *a);
                }
            }
            d0 += 64;
        }
        // The head's last lane blocks (80 and 16 wide heads are not a multiple of 64).
        while d0 < hd {
            let mut acc = [_mm512_setzero_ps(); FAST_ROWS];
            for (r, a) in acc.iter_mut().enumerate() {
                let al = _mm512_set1_ps(*alpha.add(rows[r]));
                *a = _mm512_mul_ps(_mm512_loadu_ps(o.add(rows[r] * hd + d0)), al);
            }
            for j in 0..len {
                let vv = _mm512_loadu_ps(v.add(j * hd + d0));
                for (r, a) in acc.iter_mut().enumerate() {
                    *a = _mm512_fmadd_ps(_mm512_set1_ps(*p.add(rows[r] * pstride + j)), vv, *a);
                }
            }
            for (r, a) in acc.iter().enumerate().take(nrows) {
                _mm512_storeu_ps(o.add(rows[r] * hd + d0), *a);
            }
            d0 += LANES;
        }
    }

    /// [`denom_twin`] sixteen keys at a time.
    ///
    /// # Safety
    /// AVX-512F/DQ and FMA must be available and `s` readable for `s.len()` floats.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq,fma")]
    pub(super) unsafe fn denom_zmm(s: &mut [f32], m_old: f32) -> (f32, f64, f32) {
        use std::arch::x86_64::*;
        let m_new = crate::vmath::max(s).max(m_old);
        let sh = _mm512_set1_ps(m_new);
        let mut sum = 0.0f64;
        let mut g = 0;
        while g < s.len() {
            let n = s.len() - g;
            let (v, valid) = if n >= LANES {
                let x = _mm512_sub_ps(_mm512_loadu_ps(s.as_ptr().add(g)), sh);
                (peregrine_kernels::vmath::exp16(x), u16::MAX)
            } else {
                let mask = (1u16 << n) - 1;
                let x = _mm512_maskz_loadu_ps(mask, s.as_ptr().add(g));
                (
                    _mm512_maskz_mov_ps(
                        mask,
                        peregrine_kernels::vmath::exp16(_mm512_sub_ps(x, sh)),
                    ),
                    mask,
                )
            };
            _mm512_mask_storeu_ps(s.as_mut_ptr().add(g), valid, v);
            sum += hsum16(v) as f64;
            g += LANES;
        }
        (m_new, sum, crate::vmath::det_exp(m_old - m_new))
    }

    /// [`scores_twin`] on sixteen NEON lanes, [`FAST_ROWS`] rows sharing one pass over
    /// the key: lane `l` of the four `f32x4` accumulators takes dimensions
    /// `l, l+16, …` and the reduction is `T16 = T8(v[l] + v[l+8])`, bit for bit.
    ///
    /// # Safety
    /// The CPU must have NEON; `hd` must be a multiple of 16; `q` must hold `hd`
    /// floats at every `rows[r] · hd`, `kf` the level's keys at `hd`, and `s` `len`
    /// floats at every `rows[r] · sstride`.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn scores_neon(
        q: *const f32,
        hd: usize,
        kf: *const f32,
        rows: &[usize; FAST_ROWS],
        nrows: usize,
        len: usize,
        scale: f32,
        s: *mut f32,
        sstride: usize,
        sofs: usize,
    ) {
        use std::arch::aarch64::*;
        assert!(
            hd.is_multiple_of(LANES),
            "the fast body takes whole 16-lane blocks"
        );
        // One key per pass, up to four rows' sixteen lane accumulators in flight: the
        // key vector is loaded once and every row's four FMAs ride it.
        let mut j = 0;
        while j < len {
            let kp = kf.add(j * hd);
            let mut a = [[vdupq_n_f32(0.0); 4]; FAST_ROWS];
            let mut d = 0;
            while d < hd {
                let k0 = vld1q_f32(kp.add(d));
                let k1 = vld1q_f32(kp.add(d + 4));
                let k2 = vld1q_f32(kp.add(d + 8));
                let k3 = vld1q_f32(kp.add(d + 12));
                for r in 0..nrows {
                    let qp = q.add(rows[r] * hd + d);
                    a[r][0] = vfmaq_f32(a[r][0], vld1q_f32(qp), k0);
                    a[r][1] = vfmaq_f32(a[r][1], vld1q_f32(qp.add(4)), k1);
                    a[r][2] = vfmaq_f32(a[r][2], vld1q_f32(qp.add(8)), k2);
                    a[r][3] = vfmaq_f32(a[r][3], vld1q_f32(qp.add(12)), k3);
                }
                d += LANES;
            }
            for r in 0..nrows {
                // T16: the low half plus the high half of the sixteen lanes, then the
                // eight-lane tree — the definition's association, not a wider sum.
                let lo = vaddq_f32(a[r][0], a[r][2]);
                let hi = vaddq_f32(a[r][1], a[r][3]);
                let t = hsum4_neon(lo) + hsum4_neon(hi);
                *s.add(rows[r] * sstride + sofs + j) = t * scale;
            }
            j += 1;
        }
    }

    /// [`mix_twin`] on sixteen NEON lanes, [`FAST_ROWS`] rows over one pass of V.
    ///
    /// # Safety
    /// The CPU must have NEON; `hd` a multiple of 16; `v` `len · hd` floats; `p`
    /// `len` floats at every `rows[r] · pstride`; `o` `hd` floats at every
    /// `rows[r] · hd`. The caller pads `rows` with a valid row, so every read is in
    /// bounds; only the first `nrows` are written back.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn mix_neon(
        o: *mut f32,
        hd: usize,
        alpha: *const f32,
        rows: &[usize; FAST_ROWS],
        nrows: usize,
        p: *const f32,
        pstride: usize,
        v: *const f32,
        len: usize,
    ) {
        use std::arch::aarch64::*;
        assert!(
            hd.is_multiple_of(LANES),
            "the fast body takes whole 16-lane blocks"
        );
        let mut d0 = 0;
        while d0 < hd {
            let mut acc = [[vdupq_n_f32(0.0); 4]; FAST_ROWS];
            for r in 0..FAST_ROWS {
                let al = vdupq_n_f32(*alpha.add(rows[r]));
                for (l, a) in acc[r].iter_mut().enumerate() {
                    *a = vmulq_f32(vld1q_f32(o.add(rows[r] * hd + d0 + l * 4)), al);
                }
            }
            // One ascending chain per element, exactly as the definition writes it; the
            // four value vectors are loaded once for every row.
            for j in 0..len {
                let vp = v.add(j * hd + d0);
                let v0 = vld1q_f32(vp);
                let v1 = vld1q_f32(vp.add(4));
                let v2 = vld1q_f32(vp.add(8));
                let v3 = vld1q_f32(vp.add(12));
                for r in 0..FAST_ROWS {
                    let w = vdupq_n_f32(*p.add(rows[r] * pstride + j));
                    acc[r][0] = vfmaq_f32(acc[r][0], w, v0);
                    acc[r][1] = vfmaq_f32(acc[r][1], w, v1);
                    acc[r][2] = vfmaq_f32(acc[r][2], w, v2);
                    acc[r][3] = vfmaq_f32(acc[r][3], w, v3);
                }
            }
            for r in 0..nrows {
                for (l, a) in acc[r].iter().enumerate() {
                    vst1q_f32(o.add(rows[r] * hd + d0 + l * 4), *a);
                }
            }
            d0 += LANES;
        }
    }

    /// [`denom_twin`] on NEON: [`peregrine_kernels::vmath::exp4`] (its own `exp`s
    /// lane for lane) and the `T16` group sum in `f32`, the groups added to the row's
    /// `f64` in ascending order.
    ///
    /// # Safety
    /// The CPU must have NEON and `s` must be readable for `s.len()` floats.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn denom_neon(s: &mut [f32], m_old: f32) -> (f32, f64, f32) {
        use std::arch::aarch64::*;
        let m_new = crate::vmath::max(s).max(m_old);
        let sh = vdupq_n_f32(m_new);
        let mut sum = 0.0f64;
        let mut g = 0;
        while g < s.len() {
            let n = s.len() - g;
            let (v0, v1, v2, v3) = if n >= LANES {
                let v0 = peregrine_kernels::vmath::exp4(vsubq_f32(
                    vld1q_f32(s.as_ptr().add(g)),
                    sh,
                ));
                let v1 = peregrine_kernels::vmath::exp4(vsubq_f32(
                    vld1q_f32(s.as_ptr().add(g + 4)),
                    sh,
                ));
                let v2 = peregrine_kernels::vmath::exp4(vsubq_f32(
                    vld1q_f32(s.as_ptr().add(g + 8)),
                    sh,
                ));
                let v3 = peregrine_kernels::vmath::exp4(vsubq_f32(
                    vld1q_f32(s.as_ptr().add(g + 12)),
                    sh,
                ));
                vst1q_f32(s.as_mut_ptr().add(g), v0);
                vst1q_f32(s.as_mut_ptr().add(g + 4), v1);
                vst1q_f32(s.as_mut_ptr().add(g + 8), v2);
                vst1q_f32(s.as_mut_ptr().add(g + 12), v3);
                (v0, v1, v2, v3)
            } else {
                // The piece's last group: lanes past its end are a zero term, as in
                // the twin, so `exp` must not be applied to them.
                let mut x = [0.0f32; LANES];
                for (l, xl) in x.iter_mut().enumerate() {
                    if l < n {
                        let e = peregrine_kernels::vmath::exp_scalar(s[g + l] - m_new);
                        s[g + l] = e;
                        *xl = e;
                    }
                }
                (
                    vld1q_f32(x.as_ptr()),
                    vld1q_f32(x.as_ptr().add(4)),
                    vld1q_f32(x.as_ptr().add(8)),
                    vld1q_f32(x.as_ptr().add(12)),
                )
            };
            let lo = vaddq_f32(v0, v2);
            let hi = vaddq_f32(v1, v3);
            sum += (hsum4_neon(lo) + hsum4_neon(hi)) as f64;
            g += LANES;
        }
        (m_new, sum, crate::vmath::det_exp(m_old - m_new))
    }

    /// [`scores_neon`] on the cache's own F16 halves: `vcvt_f32_f16` in registers,
    /// then the same accumulators and `T16`. The conversion is exact, so this is the
    /// widened body's bits — reading the halves only halves the bytes.
    ///
    /// # Safety
    /// The CPU must have NEON; `hd` must be a multiple of 16; `q` must hold `hd`
    /// floats at every `rows[r] · hd`, `k` the level's keys as F16 at `hd`, and `s`
    /// `len` floats at every `rows[r] · sstride`.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn scores_f16_neon(
        q: *const f32,
        hd: usize,
        k: *const u8,
        rows: &[usize; FAST_ROWS],
        nrows: usize,
        len: usize,
        scale: f32,
        s: *mut f32,
        sstride: usize,
        sofs: usize,
    ) {
        use std::arch::aarch64::*;
        assert!(
            hd.is_multiple_of(LANES),
            "the fast body takes whole 16-lane blocks"
        );
        let mut j = 0;
        while j < len {
            let kp = k.add(j * hd * 2);
            let mut a = [[vdupq_n_f32(0.0); 4]; FAST_ROWS];
            let mut d = 0;
            while d < hd {
                let at = d * 2;
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                let k2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 16) as *const u16)));
                let k3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 24) as *const u16)));
                for r in 0..nrows {
                    let qp = q.add(rows[r] * hd + d);
                    a[r][0] = vfmaq_f32(a[r][0], vld1q_f32(qp), k0);
                    a[r][1] = vfmaq_f32(a[r][1], vld1q_f32(qp.add(4)), k1);
                    a[r][2] = vfmaq_f32(a[r][2], vld1q_f32(qp.add(8)), k2);
                    a[r][3] = vfmaq_f32(a[r][3], vld1q_f32(qp.add(12)), k3);
                }
                d += LANES;
            }
            for r in 0..nrows {
                let lo = vaddq_f32(a[r][0], a[r][2]);
                let hi = vaddq_f32(a[r][1], a[r][3]);
                let t = hsum4_neon(lo) + hsum4_neon(hi);
                *s.add(rows[r] * sstride + sofs + j) = t * scale;
            }
            j += 1;
        }
    }

    /// [`mix_neon`] on the cache's own F16 halves: each position's four value vectors
    /// are converted once and feed every row's chain, so four rows share one
    /// conversion per element — the exact path converts once per row.
    ///
    /// # Safety
    /// The CPU must have NEON; `hd` a multiple of 16; `v` `len · hd` halves; `p`
    /// `len` floats at every `rows[r] · pstride`; `o` `hd` floats at every
    /// `rows[r] · hd`. The caller pads `rows` with a valid row; only the first
    /// `nrows` are written back.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn mix_f16_neon(
        o: *mut f32,
        hd: usize,
        alpha: *const f32,
        rows: &[usize; FAST_ROWS],
        nrows: usize,
        p: *const f32,
        pstride: usize,
        v: *const u8,
        len: usize,
    ) {
        use std::arch::aarch64::*;
        assert!(
            hd.is_multiple_of(LANES),
            "the fast body takes whole 16-lane blocks"
        );
        let mut d0 = 0;
        while d0 < hd {
            let mut acc = [[vdupq_n_f32(0.0); 4]; FAST_ROWS];
            for r in 0..FAST_ROWS {
                let al = vdupq_n_f32(*alpha.add(rows[r]));
                for (l, a) in acc[r].iter_mut().enumerate() {
                    *a = vmulq_f32(vld1q_f32(o.add(rows[r] * hd + d0 + l * 4)), al);
                }
            }
            for j in 0..len {
                let vp = v.add((j * hd + d0) * 2);
                let v0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp as *const u16)));
                let v1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(8) as *const u16)));
                let v2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(16) as *const u16)));
                let v3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(vp.add(24) as *const u16)));
                for r in 0..FAST_ROWS {
                    let w = vdupq_n_f32(*p.add(rows[r] * pstride + j));
                    acc[r][0] = vfmaq_f32(acc[r][0], w, v0);
                    acc[r][1] = vfmaq_f32(acc[r][1], w, v1);
                    acc[r][2] = vfmaq_f32(acc[r][2], w, v2);
                    acc[r][3] = vfmaq_f32(acc[r][3], w, v3);
                }
            }
            for r in 0..nrows {
                for (l, a) in acc[r].iter().enumerate() {
                    vst1q_f32(o.add(rows[r] * hd + d0 + l * 4), *a);
                }
            }
            d0 += LANES;
        }
    }

    /// The scores of up to [`FAST_ROWS`] rows that see the same keys of one tile.
    ///
    /// `k16` is the tile's F16 halves at the first visible key when the cache can be
    /// read directly (`f16_direct`); the conversion is exact, so the halves are only
    /// fewer bytes, never different bits.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn scores_rows(
        sc: &mut Scratch,
        rows: &[usize; FAST_ROWS],
        nrows: usize,
        j: usize,
        len: usize,
        hd: usize,
        scale: f32,
        k16: Option<&[u8]>,
    ) {
        let _ = k16;
        if body_ok(hd) {
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: as `scores_row`; `rows` are distinct scratch rows inside
                // `MAX_PARTIAL_ROWS` and `s_tile` holds one `BLOCK` row each; `k16`
                // holds the tile's halves and the widened tile its floats.
                unsafe {
                    scores_zmm(
                        sc.q.as_ptr(),
                        hd,
                        sc.kf.as_ptr().add(j * hd),
                        rows,
                        nrows,
                        len,
                        scale,
                        sc.s_tile.as_mut_ptr(),
                        BLOCK,
                        j,
                    )
                };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: NEON is this target's baseline and `body_ok` checked the
                // tier; `rows` are distinct scratch rows inside `MAX_PARTIAL_ROWS` and
                // `s_tile` holds one `BLOCK` row each; `k16` is the tile's halves.
                unsafe {
                    match k16 {
                        Some(k) => scores_f16_neon(
                            sc.q.as_ptr(),
                            hd,
                            k.as_ptr(),
                            rows,
                            nrows,
                            len,
                            scale,
                            sc.s_tile.as_mut_ptr(),
                            BLOCK,
                            j,
                        ),
                        None => scores_neon(
                            sc.q.as_ptr(),
                            hd,
                            sc.kf.as_ptr().add(j * hd),
                            rows,
                            nrows,
                            len,
                            scale,
                            sc.s_tile.as_mut_ptr(),
                            BLOCK,
                            j,
                        ),
                    }
                };
                return;
            }
        }
        for &lr in &rows[..nrows] {
            scores_twin(
                &sc.q[lr * hd..][..hd],
                &sc.kf[j * hd..],
                len,
                hd,
                scale,
                &mut sc.s_tile[lr * BLOCK + j..lr * BLOCK + j + len],
            );
        }
    }

    /// The mix of up to [`FAST_ROWS`] rows over the same keys of one tile.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mix_rows(
        sc: &mut Scratch,
        rows: &[usize; FAST_ROWS],
        nrows: usize,
        alpha: &[f32; MAX_PARTIAL_ROWS],
        j: usize,
        len: usize,
        hd: usize,
        first: bool,
        v16: Option<&[u8]>,
    ) {
        let _ = v16;
        if body_ok(hd) {
            #[cfg(target_arch = "x86_64")]
            {
                let one = [1.0f32; MAX_PARTIAL_ROWS];
                let al = if first { alpha.as_ptr() } else { one.as_ptr() };
                // SAFETY: as `scores_rows`; `al` is indexed by the same scratch rows,
                // `p` is the row's own scores of this piece, and `v` the tile's values.
                unsafe {
                    mix_zmm(
                        sc.o.as_mut_ptr(),
                        hd,
                        al,
                        rows,
                        nrows,
                        sc.s_tile.as_ptr().add(j),
                        BLOCK,
                        sc.kf.as_ptr().add(j * hd),
                        len,
                    )
                };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            {
                let one = [1.0f32; MAX_PARTIAL_ROWS];
                let al = if first { alpha.as_ptr() } else { one.as_ptr() };
                // SAFETY: as `scores_rows`; `al` is indexed by the same scratch rows,
                // `p` is the row's own scores of this piece, and `v16` the tile's
                // halves (or `kf` its widened floats).
                unsafe {
                    match v16 {
                        Some(v) => mix_f16_neon(
                            sc.o.as_mut_ptr(),
                            hd,
                            al,
                            rows,
                            nrows,
                            sc.s_tile.as_ptr().add(j),
                            BLOCK,
                            v.as_ptr(),
                            len,
                        ),
                        None => mix_neon(
                            sc.o.as_mut_ptr(),
                            hd,
                            al,
                            rows,
                            nrows,
                            sc.s_tile.as_ptr().add(j),
                            BLOCK,
                            sc.kf.as_ptr().add(j * hd),
                            len,
                        ),
                    }
                };
                return;
            }
        }
        for &lr in &rows[..nrows] {
            let a = if first { alpha[lr] } else { 1.0 };
            mix_twin(
                &mut sc.o[lr * hd..][..hd],
                a,
                &sc.s_tile[lr * BLOCK + j..lr * BLOCK + j + len],
                &sc.kf[j * hd..],
                hd,
            );
        }
    }

    /// One row's soft-max over a piece: the tier's body, or the definition.
    pub(super) fn denom_row(
        sc: &mut Scratch,
        lr: usize,
        j: usize,
        len: usize,
        hd: usize,
    ) -> (f32, f64, f32) {
        let m_old = sc.m[lr];
        if body_ok(hd) {
            #[cfg(target_arch = "x86_64")]
            {
                let s = &mut sc.s_tile[lr * BLOCK + j..lr * BLOCK + j + len];
                // SAFETY: the tier's features are checked (`body_ok`) and `s` is this
                // row's own piece of `s_tile`.
                return unsafe { denom_zmm(s, m_old) };
            }
            #[cfg(target_arch = "aarch64")]
            {
                let s = &mut sc.s_tile[lr * BLOCK + j..lr * BLOCK + j + len];
                // SAFETY: NEON is this target's baseline and `body_ok` checked the
                // tier; `s` is this row's own piece of `s_tile`.
                return unsafe { denom_neon(s, m_old) };
            }
        }
        denom_twin(&mut sc.s_tile[lr * BLOCK + j..lr * BLOCK + j + len], m_old)
    }
} // pub mod fast

/// The per-key reduction tree of the four-key score block: `((d0+d1)+(d2+d3)) +
/// ((d4+d5)+(d6+d7))`. A tail key reduced with this tree gets the same bits as one
/// that sat in a block, so a key's score does not depend on the block boundaries —
/// which is what lets positions be grouped by how many rows share them.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_grouped_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let t = _mm256_hadd_ps(v, v);
    let t = _mm256_hadd_ps(t, t);
    let lo = _mm256_castps256_ps128(t);
    let hi = _mm256_extractf128_ps(t, 1);
    _mm_cvtss_f32(_mm_add_ss(lo, hi))
}

/// `s[j] = scale · q·k_j` with K read from the cache as f16 and widened in registers.
///
/// Four keys per pass, so two query rows can share each loaded K vector in
/// [`scores_f16_pair`]; the per-key dot order is the widened kernel's (dimensions
/// ascending in eight-lane blocks).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn scores_f16(q: &[f32], k: &[u8], vis: usize, hd: usize, scale: f32, s: &mut [f32]) {
    use std::arch::x86_64::*;
    let qp = q.as_ptr();
    let kp = k.as_ptr();
    let sc = _mm_set1_ps(scale);
    let mut j = 0;
    while j + 4 <= vis {
        let mut a = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i < hd {
            let qv = _mm256_loadu_ps(qp.add(i));
            for (u, acc) in a.iter_mut().enumerate() {
                let kv = _mm256_cvtph_ps(_mm_loadu_si128(
                    kp.add(((j + u) * hd + i) * 2) as *const __m128i
                ));
                *acc = _mm256_fmadd_ps(qv, kv, *acc);
            }
            i += 8;
        }
        let t0 = _mm256_hadd_ps(a[0], a[1]);
        let t1 = _mm256_hadd_ps(a[2], a[3]);
        let u0 = _mm256_hadd_ps(t0, t1);
        let lo = _mm256_castps256_ps128(u0);
        let hi = _mm256_extractf128_ps(u0, 1);
        _mm_storeu_ps(s.as_mut_ptr().add(j), _mm_mul_ps(_mm_add_ps(lo, hi), sc));
        j += 4;
    }
    while j < vis {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < hd {
            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(qp.add(i)),
                _mm256_cvtph_ps(_mm_loadu_si128(kp.add((j * hd + i) * 2) as *const __m128i)),
                acc,
            );
            i += 8;
        }
        s[j] = hsum_grouped_ps(acc) * scale;
        j += 1;
    }
}

/// Two query rows scored from one decoded K vector.
///
/// Each row's accumulator takes the same `fma(q_row, k, acc)` in the same order as
/// [`scores_f16`], so the sharing cannot change a bit. Rows may have different visible
/// lengths; beyond a row's own length the K is read but its score is not stored.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn scores_f16_pair(
    q0: &[f32],
    q1: &[f32],
    k: &[u8],
    vis0: usize,
    vis1: usize,
    hd: usize,
    scale: f32,
    s0: &mut [f32],
    s1: &mut [f32],
) {
    use std::arch::x86_64::*;
    let vis = vis0.max(vis1);
    let common = vis0.min(vis1);
    let (q0p, q1p, kp) = (q0.as_ptr(), q1.as_ptr(), k.as_ptr());
    let sc = _mm_set1_ps(scale);
    let mut j = 0;
    while j + 4 <= common {
        let mut a0 = [_mm256_setzero_ps(); 4];
        let mut a1 = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i < hd {
            let qv0 = _mm256_loadu_ps(q0p.add(i));
            let qv1 = _mm256_loadu_ps(q1p.add(i));
            for u in 0..4 {
                let kv = _mm256_cvtph_ps(_mm_loadu_si128(
                    kp.add(((j + u) * hd + i) * 2) as *const __m128i
                ));
                a0[u] = _mm256_fmadd_ps(qv0, kv, a0[u]);
                a1[u] = _mm256_fmadd_ps(qv1, kv, a1[u]);
            }
            i += 8;
        }
        let t0 = _mm256_hadd_ps(a0[0], a0[1]);
        let t1 = _mm256_hadd_ps(a0[2], a0[3]);
        let u0 = _mm256_hadd_ps(t0, t1);
        let lo = _mm256_castps256_ps128(u0);
        let hi = _mm256_extractf128_ps(u0, 1);
        if j + 4 <= vis0 {
            _mm_storeu_ps(s0.as_mut_ptr().add(j), _mm_mul_ps(_mm_add_ps(lo, hi), sc));
        }
        let t0 = _mm256_hadd_ps(a1[0], a1[1]);
        let t1 = _mm256_hadd_ps(a1[2], a1[3]);
        let u0 = _mm256_hadd_ps(t0, t1);
        let lo = _mm256_castps256_ps128(u0);
        let hi = _mm256_extractf128_ps(u0, 1);
        if j + 4 <= vis1 {
            _mm_storeu_ps(s1.as_mut_ptr().add(j), _mm_mul_ps(_mm_add_ps(lo, hi), sc));
        }
        j += 4;
    }
    while j < vis {
        if j < vis0 {
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i < hd {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(q0p.add(i)),
                    _mm256_cvtph_ps(_mm_loadu_si128(kp.add((j * hd + i) * 2) as *const __m128i)),
                    acc,
                );
                i += 8;
            }
            s0[j] = hsum_grouped_ps(acc) * scale;
        }
        if j < vis1 {
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i < hd {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(q1p.add(i)),
                    _mm256_cvtph_ps(_mm_loadu_si128(kp.add((j * hd + i) * 2) as *const __m128i)),
                    acc,
                );
                i += 8;
            }
            s1[j] = hsum_grouped_ps(acc) * scale;
        }
        j += 1;
    }
}

/// `o = α·o + Σ_j p_j · v_j` with V read from the cache as f16 and widened in
/// registers. The output accumulators stay in registers across the whole block of
/// positions; `j` accumulates ascending exactly as the widened kernel does.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn mix_f16_avx2(o: &mut [f32], alpha: f32, p: &[f32], v: &[u8], hd: usize) {
    use std::arch::x86_64::*;
    let al = _mm256_set1_ps(alpha);
    let vp = v.as_ptr();
    let mut d0 = 0;
    while d0 + 32 <= hd {
        let mut acc = [_mm256_setzero_ps(); 4];
        for (l, a) in acc.iter_mut().enumerate() {
            *a = _mm256_mul_ps(_mm256_loadu_ps(o.as_ptr().add(d0 + l * 8)), al);
        }
        for (j, &w) in p.iter().enumerate() {
            let wv = _mm256_set1_ps(w);
            for (l, a) in acc.iter_mut().enumerate() {
                *a = _mm256_fmadd_ps(
                    wv,
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        vp.add((j * hd + d0 + l * 8) * 2) as *const __m128i
                    )),
                    *a,
                );
            }
        }
        for (l, a) in acc.iter().enumerate() {
            _mm256_storeu_ps(o.as_mut_ptr().add(d0 + l * 8), *a);
        }
        d0 += 32;
    }
    while d0 < hd {
        let mut acc = _mm256_mul_ps(_mm256_loadu_ps(o.as_ptr().add(d0)), al);
        for (j, &w) in p.iter().enumerate() {
            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(w),
                _mm256_cvtph_ps(_mm_loadu_si128(vp.add((j * hd + d0) * 2) as *const __m128i)),
                acc,
            );
        }
        _mm256_storeu_ps(o.as_mut_ptr().add(d0), acc);
        d0 += 8;
    }
}

/// Two rows' value mixes sharing each decoded V vector, with per-row chunk-boundary
/// resets. `ends[ci]` is the absolute end of chunk `ci` (see [`row_chunks`]); when a
/// row's position scan reaches one, its accumulator is banked into `part` at that
/// chunk's slot and restarts at zero — the same bank-and-restart [`run`] does per
/// piece, in the same position order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_f16_pair_avx2(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    vis0: usize,
    vis1: usize,
    blk: usize,
    cur0: usize,
    cur1: usize,
    ends0: &[usize],
    ends1: &[usize],
    v: &[u8],
    hd: usize,
    part0: &mut [f32],
    part1: &mut [f32],
) {
    use std::arch::x86_64::*;
    let vis = vis0.max(vis1);
    if vis == 0 {
        return;
    }
    let vp = v.as_ptr();
    let (al0, al1) = (_mm256_set1_ps(alpha0), _mm256_set1_ps(alpha1));
    let mut d0 = 0;
    while d0 < hd {
        let lanes = (hd - d0).min(32) / 8;
        let mut a0 = [_mm256_setzero_ps(); 4];
        let mut a1 = [_mm256_setzero_ps(); 4];
        if vis0 > 0 {
            for (l, a) in a0.iter_mut().enumerate().take(lanes) {
                *a = _mm256_mul_ps(_mm256_loadu_ps(o0.as_ptr().add(d0 + l * 8)), al0);
            }
        }
        if vis1 > 0 {
            for (l, a) in a1.iter_mut().enumerate().take(lanes) {
                *a = _mm256_mul_ps(_mm256_loadu_ps(o1.as_ptr().add(d0 + l * 8)), al1);
            }
        }
        let (mut c0, mut c1) = (cur0, cur1);
        for j in 0..vis {
            if j < vis0 {
                let wv = _mm256_set1_ps(p0[j]);
                for (l, a) in a0.iter_mut().enumerate().take(lanes) {
                    *a = _mm256_fmadd_ps(
                        wv,
                        _mm256_cvtph_ps(_mm_loadu_si128(
                            vp.add((j * hd + d0 + l * 8) * 2) as *const __m128i
                        )),
                        *a,
                    );
                }
                while c0 + 1 < ends0.len() && blk + j + 1 >= ends0[c0] {
                    for (l, a) in a0.iter().enumerate().take(lanes) {
                        _mm256_storeu_ps(part0.as_mut_ptr().add(c0 * hd + d0 + l * 8), *a);
                    }
                    a0 = [_mm256_setzero_ps(); 4];
                    c0 += 1;
                }
            }
            if j < vis1 {
                let wv = _mm256_set1_ps(p1[j]);
                for (l, a) in a1.iter_mut().enumerate().take(lanes) {
                    *a = _mm256_fmadd_ps(
                        wv,
                        _mm256_cvtph_ps(_mm_loadu_si128(
                            vp.add((j * hd + d0 + l * 8) * 2) as *const __m128i
                        )),
                        *a,
                    );
                }
                while c1 + 1 < ends1.len() && blk + j + 1 >= ends1[c1] {
                    for (l, a) in a1.iter().enumerate().take(lanes) {
                        _mm256_storeu_ps(part1.as_mut_ptr().add(c1 * hd + d0 + l * 8), *a);
                    }
                    a1 = [_mm256_setzero_ps(); 4];
                    c1 += 1;
                }
            }
        }
        if vis0 > 0 {
            for (l, a) in a0.iter().enumerate().take(lanes) {
                _mm256_storeu_ps(o0.as_mut_ptr().add(d0 + l * 8), *a);
            }
        }
        if vis1 > 0 {
            for (l, a) in a1.iter().enumerate().take(lanes) {
                _mm256_storeu_ps(o1.as_mut_ptr().add(d0 + l * 8), *a);
            }
        }
        d0 += lanes * 8;
    }
}

// ---------------------------------------------------------------------------
// The AVX-512 tier's sixteen-lane kernels.
//
// Every path on the tier — solo decode, the prefill tile, and a mapped
// multi-user row — takes these, and the f32-tile and F16-direct variants are the
// same arithmetic with the widening moved into the load. A row's bits therefore
// depend on neither the path nor the width.

/// `o = α·o + Σ_j p_j · v_j` over a widened f32 tile, sixteen lanes at a time.
/// The accumulators stay in registers across the positions; `j` accumulates
/// ascending, exactly as [`mix_avx2`] does.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2,fma,f16c")]
unsafe fn mix_avx512(o: &mut [f32], alpha: f32, p: &[f32], vf: &[f32], hd: usize) {
    use std::arch::x86_64::*;
    let al = _mm512_set1_ps(alpha);
    let vp = vf.as_ptr();
    let op = o.as_mut_ptr();
    let mut d0 = 0;
    while d0 + 64 <= hd {
        let mut acc = [_mm512_setzero_ps(); 4];
        for (l, a) in acc.iter_mut().enumerate() {
            *a = _mm512_mul_ps(_mm512_loadu_ps(op.add(d0 + l * 16)), al);
        }
        for (j, &w) in p.iter().enumerate() {
            let wv = _mm512_set1_ps(w);
            for (l, a) in acc.iter_mut().enumerate() {
                *a = _mm512_fmadd_ps(wv, _mm512_loadu_ps(vp.add(j * hd + d0 + l * 16)), *a);
            }
        }
        for (l, a) in acc.iter().enumerate() {
            _mm512_storeu_ps(op.add(d0 + l * 16), *a);
        }
        d0 += 64;
    }
    while d0 < hd {
        let mut acc = _mm512_mul_ps(_mm512_loadu_ps(op.add(d0)), al);
        for (j, &w) in p.iter().enumerate() {
            acc = _mm512_fmadd_ps(_mm512_set1_ps(w), _mm512_loadu_ps(vp.add(j * hd + d0)), acc);
        }
        _mm512_storeu_ps(op.add(d0), acc);
        d0 += 16;
    }
}

/// [`mix_avx512`] with V read as f16 halves; same operands and order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2,fma,f16c")]
unsafe fn mix_f16_avx512(o: &mut [f32], alpha: f32, p: &[f32], v: &[u8], hd: usize) {
    use std::arch::x86_64::*;
    let al = _mm512_set1_ps(alpha);
    let vp = v.as_ptr();
    let op = o.as_mut_ptr();
    let mut d0 = 0;
    while d0 + 64 <= hd {
        let mut acc = [_mm512_setzero_ps(); 4];
        for (l, a) in acc.iter_mut().enumerate() {
            *a = _mm512_mul_ps(_mm512_loadu_ps(op.add(d0 + l * 16)), al);
        }
        for (j, &w) in p.iter().enumerate() {
            let wv = _mm512_set1_ps(w);
            for (l, a) in acc.iter_mut().enumerate() {
                *a = _mm512_fmadd_ps(
                    wv,
                    _mm512_cvtph_ps(_mm256_loadu_si256(
                        vp.add((j * hd + d0 + l * 16) * 2) as *const __m256i
                    )),
                    *a,
                );
            }
        }
        for (l, a) in acc.iter().enumerate() {
            _mm512_storeu_ps(op.add(d0 + l * 16), *a);
        }
        d0 += 64;
    }
    while d0 < hd {
        let mut acc = _mm512_mul_ps(_mm512_loadu_ps(op.add(d0)), al);
        for (j, &w) in p.iter().enumerate() {
            acc = _mm512_fmadd_ps(
                _mm512_set1_ps(w),
                _mm512_cvtph_ps(_mm256_loadu_si256(
                    vp.add((j * hd + d0) * 2) as *const __m256i
                )),
                acc,
            );
        }
        _mm512_storeu_ps(op.add(d0), acc);
        d0 += 16;
    }
}

/// Two rows' sixteen-lane value mixes with the per-row chunk-boundary resets of
/// [`mix_f16_pair_avx2`]. Each output element sees the same ascending-position
/// `α·o + p_j·v_j` as every other mix kernel on the tier.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_f16_pair_avx512(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    vis0: usize,
    vis1: usize,
    blk: usize,
    cur0: usize,
    cur1: usize,
    ends0: &[usize],
    ends1: &[usize],
    v: &[u8],
    hd: usize,
    part0: &mut [f32],
    part1: &mut [f32],
) {
    use std::arch::x86_64::*;
    let vis = vis0.max(vis1);
    if vis == 0 {
        return;
    }
    let vp = v.as_ptr();
    let (al0, al1) = (_mm512_set1_ps(alpha0), _mm512_set1_ps(alpha1));
    let mut d0 = 0;
    while d0 < hd {
        let lanes = (hd - d0).min(64) / 16;
        let mut a0 = [_mm512_setzero_ps(); 4];
        let mut a1 = [_mm512_setzero_ps(); 4];
        if vis0 > 0 {
            for (l, a) in a0.iter_mut().enumerate().take(lanes) {
                *a = _mm512_mul_ps(_mm512_loadu_ps(o0.as_ptr().add(d0 + l * 16)), al0);
            }
        }
        if vis1 > 0 {
            for (l, a) in a1.iter_mut().enumerate().take(lanes) {
                *a = _mm512_mul_ps(_mm512_loadu_ps(o1.as_ptr().add(d0 + l * 16)), al1);
            }
        }
        let (mut c0, mut c1) = (cur0, cur1);
        for j in 0..vis {
            if j < vis0 {
                let wv = _mm512_set1_ps(p0[j]);
                for (l, a) in a0.iter_mut().enumerate().take(lanes) {
                    *a = _mm512_fmadd_ps(
                        wv,
                        _mm512_cvtph_ps(_mm256_loadu_si256(
                            vp.add((j * hd + d0 + l * 16) * 2) as *const __m256i
                        )),
                        *a,
                    );
                }
                while c0 + 1 < ends0.len() && blk + j + 1 >= ends0[c0] {
                    for (l, a) in a0.iter().enumerate().take(lanes) {
                        _mm512_storeu_ps(part0.as_mut_ptr().add(c0 * hd + d0 + l * 16), *a);
                    }
                    a0 = [_mm512_setzero_ps(); 4];
                    c0 += 1;
                }
            }
            if j < vis1 {
                let wv = _mm512_set1_ps(p1[j]);
                for (l, a) in a1.iter_mut().enumerate().take(lanes) {
                    *a = _mm512_fmadd_ps(
                        wv,
                        _mm512_cvtph_ps(_mm256_loadu_si256(
                            vp.add((j * hd + d0 + l * 16) * 2) as *const __m256i
                        )),
                        *a,
                    );
                }
                while c1 + 1 < ends1.len() && blk + j + 1 >= ends1[c1] {
                    for (l, a) in a1.iter().enumerate().take(lanes) {
                        _mm512_storeu_ps(part1.as_mut_ptr().add(c1 * hd + d0 + l * 16), *a);
                    }
                    a1 = [_mm512_setzero_ps(); 4];
                    c1 += 1;
                }
            }
        }
        if vis0 > 0 {
            for (l, a) in a0.iter().enumerate().take(lanes) {
                _mm512_storeu_ps(o0.as_mut_ptr().add(d0 + l * 16), *a);
            }
        }
        if vis1 > 0 {
            for (l, a) in a1.iter().enumerate().take(lanes) {
                _mm512_storeu_ps(o1.as_mut_ptr().add(d0 + l * 16), *a);
            }
        }
        d0 += lanes * 16;
    }
}

/// [`mix_f16_avx512`] or [`mix_f16_avx2`], by the tier.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn mix_f16(o: &mut [f32], alpha: f32, p: &[f32], v: &[u8], hd: usize) {
    if avx512_attn(hd) {
        return mix_f16_avx512(o, alpha, p, v, hd);
    }
    mix_f16_avx2(o, alpha, p, v, hd)
}

/// [`mix_f16_pair_avx512`] or [`mix_f16_pair_avx2`], by the tier.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_f16_pair(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    vis0: usize,
    vis1: usize,
    blk: usize,
    cur0: usize,
    cur1: usize,
    ends0: &[usize],
    ends1: &[usize],
    v: &[u8],
    hd: usize,
    part0: &mut [f32],
    part1: &mut [f32],
) {
    if avx512_attn(hd) {
        return mix_f16_pair_avx512(
            o0, o1, alpha0, alpha1, p0, p1, vis0, vis1, blk, cur0, cur1, ends0, ends1, v, hd,
            part0, part1,
        );
    }
    mix_f16_pair_avx2(
        o0, o1, alpha0, alpha1, p0, p1, vis0, vis1, blk, cur0, cur1, ends0, ends1, v, hd, part0,
        part1,
    )
}

/// Two rows of one KV head mixed over the same positions in one pass over V.
///
/// Both rows see exactly the same positions (they are the `gqa` query heads of one
/// batch row), so the F16 load and its widening are shared: one `vcvtph2ps` per 16
/// dimensions feeds both rows' fused multiply-adds. Each row keeps its own
/// accumulators and adds `p_j · v_j` in ascending position order, the same operands
/// and the same order as [`mix_f16`], so the sharing cannot change a bit.
///
/// The generic [`mix_f16_pair`] handles rows of different lengths and banks partial
/// results inside the pass; this body is for the prefill tile, whose rows share the
/// range and whose chunk boundaries the caller cuts the pass at, and it measures
/// 1.4-1.7× the two solo mixes (the generic body re-loads V per row and branches per
/// position, which is why it is not used here).
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_f16_pair_same(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    v: &[u8],
    hd: usize,
) {
    if avx512_attn(hd) {
        return mix_f16_pair_same_zmm(o0, o1, alpha0, alpha1, p0, p1, v, hd);
    }
    if hd.is_multiple_of(8) && crate::cpu_has!("avx2") && crate::cpu_has!("fma") {
        return mix_f16_pair_same_ymm(o0, o1, alpha0, alpha1, p0, p1, v, hd);
    }
    mix_f16(o0, alpha0, p0, v, hd);
    mix_f16(o1, alpha1, p1, v, hd);
}

/// [`mix_f16_pair_same`] sixteen lanes at a time.
///
/// # Safety
/// The CPU must have AVX-512F/DQ, AVX2, FMA and F16C, `hd` must be a multiple of 16,
/// `p0` and `p1` the same length, and `v` at least `p0.len() · hd` halves.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_f16_pair_same_zmm(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    v: &[u8],
    hd: usize,
) {
    use std::arch::x86_64::*;
    let vis = p0.len();
    if vis == 0 {
        return;
    }
    let vp = v.as_ptr();
    let (al0, al1) = (_mm512_set1_ps(alpha0), _mm512_set1_ps(alpha1));
    let mut d0 = 0;
    while d0 < hd {
        let lanes = (hd - d0).min(64) / 16;
        let mut a0 = [_mm512_setzero_ps(); 4];
        let mut a1 = [_mm512_setzero_ps(); 4];
        for l in 0..lanes {
            a0[l] = _mm512_mul_ps(_mm512_loadu_ps(o0.as_ptr().add(d0 + l * 16)), al0);
            a1[l] = _mm512_mul_ps(_mm512_loadu_ps(o1.as_ptr().add(d0 + l * 16)), al1);
        }
        for (j, (&w0, &w1)) in p0.iter().zip(p1).enumerate() {
            let (w0, w1) = (_mm512_set1_ps(w0), _mm512_set1_ps(w1));
            let row = vp.add((j * hd + d0) * 2);
            for (l, (a0, a1)) in a0.iter_mut().zip(a1.iter_mut()).enumerate().take(lanes) {
                let vv = _mm512_cvtph_ps(_mm256_loadu_si256(row.add(l * 32) as *const __m256i));
                *a0 = _mm512_fmadd_ps(w0, vv, *a0);
                *a1 = _mm512_fmadd_ps(w1, vv, *a1);
            }
        }
        for l in 0..lanes {
            _mm512_storeu_ps(o0.as_mut_ptr().add(d0 + l * 16), a0[l]);
            _mm512_storeu_ps(o1.as_mut_ptr().add(d0 + l * 16), a1[l]);
        }
        d0 += 64;
    }
}

/// [`mix_f16_pair_same`] eight lanes at a time.
///
/// # Safety
/// The CPU must have AVX2, FMA and F16C, `hd` must be a multiple of 8, `p0` and `p1`
/// the same length, and `v` at least `p0.len() · hd` halves.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_f16_pair_same_ymm(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    v: &[u8],
    hd: usize,
) {
    use std::arch::x86_64::*;
    let vis = p0.len();
    if vis == 0 {
        return;
    }
    let vp = v.as_ptr();
    let (al0, al1) = (_mm256_set1_ps(alpha0), _mm256_set1_ps(alpha1));
    let mut d0 = 0;
    while d0 < hd {
        let lanes = (hd - d0).min(32) / 8;
        let mut a0 = [_mm256_setzero_ps(); 4];
        let mut a1 = [_mm256_setzero_ps(); 4];
        for l in 0..lanes {
            a0[l] = _mm256_mul_ps(_mm256_loadu_ps(o0.as_ptr().add(d0 + l * 8)), al0);
            a1[l] = _mm256_mul_ps(_mm256_loadu_ps(o1.as_ptr().add(d0 + l * 8)), al1);
        }
        for (j, (&w0, &w1)) in p0.iter().zip(p1).enumerate() {
            let (w0, w1) = (_mm256_set1_ps(w0), _mm256_set1_ps(w1));
            let row = vp.add((j * hd + d0) * 2);
            for (l, (a0, a1)) in a0.iter_mut().zip(a1.iter_mut()).enumerate().take(lanes) {
                let vv = _mm256_cvtph_ps(_mm_loadu_si128(row.add(l * 16) as *const __m128i));
                *a0 = _mm256_fmadd_ps(w0, vv, *a0);
                *a1 = _mm256_fmadd_ps(w1, vv, *a1);
            }
        }
        for l in 0..lanes {
            _mm256_storeu_ps(o0.as_mut_ptr().add(d0 + l * 8), a0[l]);
            _mm256_storeu_ps(o1.as_mut_ptr().add(d0 + l * 8), a1[l]);
        }
        d0 += 32;
    }
}

/// Two rows of one KV head mixed over the same positions in one pass over a
/// **widened** (f32) V tile.
///
/// This is [`mix_f16_pair_same`] for the tile below [`DIRECT_MIN_VIS`], whose values
/// are decoded into f32 once per tile: both rows read the same f32 vector, each row
/// keeps its own accumulators and adds `p_j · v_j` ascending in `j`, so the sharing
/// cannot change a bit.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_pair_same(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    vf: &[f32],
    hd: usize,
) {
    if avx512_attn(hd) {
        return mix_pair_same_zmm(o0, o1, alpha0, alpha1, p0, p1, vf, hd);
    }
    if hd.is_multiple_of(8) && crate::cpu_has!("avx2") && crate::cpu_has!("fma") {
        return mix_pair_same_ymm(o0, o1, alpha0, alpha1, p0, p1, vf, hd);
    }
    mix(o0, alpha0, p0, vf, hd);
    mix(o1, alpha1, p1, vf, hd);
}

/// [`mix_pair_same`] sixteen lanes at a time.
///
/// # Safety
/// The CPU must have AVX-512F/DQ, AVX2 and FMA, `hd` must be a multiple of 16,
/// `p0` and `p1` the same length, and `vf` at least `p0.len() · hd` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_pair_same_zmm(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    vf: &[f32],
    hd: usize,
) {
    use std::arch::x86_64::*;
    let vis = p0.len();
    if vis == 0 {
        return;
    }
    let vp = vf.as_ptr();
    let (al0, al1) = (_mm512_set1_ps(alpha0), _mm512_set1_ps(alpha1));
    let mut d0 = 0;
    while d0 < hd {
        let lanes = (hd - d0).min(64) / 16;
        let mut a0 = [_mm512_setzero_ps(); 4];
        let mut a1 = [_mm512_setzero_ps(); 4];
        for l in 0..lanes {
            a0[l] = _mm512_mul_ps(_mm512_loadu_ps(o0.as_ptr().add(d0 + l * 16)), al0);
            a1[l] = _mm512_mul_ps(_mm512_loadu_ps(o1.as_ptr().add(d0 + l * 16)), al1);
        }
        for (j, (&w0, &w1)) in p0.iter().zip(p1).enumerate() {
            let (w0, w1) = (_mm512_set1_ps(w0), _mm512_set1_ps(w1));
            let row = vp.add(j * hd + d0);
            for l in 0..lanes {
                let vv = _mm512_loadu_ps(row.add(l * 16));
                a0[l] = _mm512_fmadd_ps(w0, vv, a0[l]);
                a1[l] = _mm512_fmadd_ps(w1, vv, a1[l]);
            }
        }
        for l in 0..lanes {
            _mm512_storeu_ps(o0.as_mut_ptr().add(d0 + l * 16), a0[l]);
            _mm512_storeu_ps(o1.as_mut_ptr().add(d0 + l * 16), a1[l]);
        }
        d0 += lanes * 16;
    }
}

/// [`mix_pair_same`] eight lanes at a time, four lanes per row in flight: the two
/// rows' eight accumulators fit in registers together with the loaded value.
///
/// # Safety
/// The CPU must have AVX2 and FMA, `hd` must be a multiple of 8, `p0` and `p1` the
/// same length, and `vf` at least `p0.len() · hd` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_pair_same_ymm(
    o0: &mut [f32],
    o1: &mut [f32],
    alpha0: f32,
    alpha1: f32,
    p0: &[f32],
    p1: &[f32],
    vf: &[f32],
    hd: usize,
) {
    use std::arch::x86_64::*;
    let vis = p0.len();
    if vis == 0 {
        return;
    }
    let vp = vf.as_ptr();
    let (al0, al1) = (_mm256_set1_ps(alpha0), _mm256_set1_ps(alpha1));
    let mut d0 = 0;
    while d0 < hd {
        let lanes = (hd - d0).min(32) / 8;
        let mut a0 = [_mm256_setzero_ps(); 4];
        let mut a1 = [_mm256_setzero_ps(); 4];
        for l in 0..lanes {
            a0[l] = _mm256_mul_ps(_mm256_loadu_ps(o0.as_ptr().add(d0 + l * 8)), al0);
            a1[l] = _mm256_mul_ps(_mm256_loadu_ps(o1.as_ptr().add(d0 + l * 8)), al1);
        }
        for (j, (&w0, &w1)) in p0.iter().zip(p1).enumerate() {
            let (w0, w1) = (_mm256_set1_ps(w0), _mm256_set1_ps(w1));
            let row = vp.add(j * hd + d0);
            for l in 0..lanes {
                let vv = _mm256_loadu_ps(row.add(l * 8));
                a0[l] = _mm256_fmadd_ps(w0, vv, a0[l]);
                a1[l] = _mm256_fmadd_ps(w1, vv, a1[l]);
            }
        }
        for l in 0..lanes {
            _mm256_storeu_ps(o0.as_mut_ptr().add(d0 + l * 8), a0[l]);
            _mm256_storeu_ps(o1.as_mut_ptr().add(d0 + l * 8), a1[l]);
        }
        d0 += lanes * 8;
    }
}

/// Two query rows scored from one widened K tile.
///
/// Four keys per pass and four accumulators per row, so the loaded K vectors are
/// shared and each row adds them in the same order as [`scores`] does. `s0`/`s1` hold
/// `vis0`/`vis1` scores of the same starting position. Groups are taken only while
/// both rows see four whole keys; the shorter row's last keys, and the longer row's
/// keys past them, are scored one at a time with the same tree. The per-key reduction
/// is the widened kernel's `((d0+d1)+(d2+d3))+((d4+d5)+(d6+d7))`, so pairing changes
/// no bit, whatever the two visible lengths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn scores_pair(
    q0: &[f32],
    q1: &[f32],
    kf: &[f32],
    vis0: usize,
    vis1: usize,
    hd: usize,
    scale: f32,
    s0: &mut [f32],
    s1: &mut [f32],
) {
    use std::arch::x86_64::*;
    let (q0p, q1p, kp) = (q0.as_ptr(), q1.as_ptr(), kf.as_ptr());
    let sc = _mm256_set1_ps(scale);
    let vis = vis0.max(vis1);
    // Groups only where both rows see the whole group; a row's keys past the other
    // row's last group are scored one at a time below (same reduction tree).
    let common = vis0.min(vis1);
    let mut j = 0;
    while j + 4 <= common {
        let mut a0 = [_mm256_setzero_ps(); 4];
        let mut a1 = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i < hd {
            let qv0 = _mm256_loadu_ps(q0p.add(i));
            let qv1 = _mm256_loadu_ps(q1p.add(i));
            let mut kv = [_mm256_setzero_ps(); 4];
            for (u, k) in kv.iter_mut().enumerate() {
                *k = _mm256_loadu_ps(kp.add((j + u) * hd + i));
            }
            for u in 0..4 {
                a0[u] = _mm256_fmadd_ps(qv0, kv[u], a0[u]);
                a1[u] = _mm256_fmadd_ps(qv1, kv[u], a1[u]);
            }
            i += 8;
        }
        let t0 = _mm256_hadd_ps(a0[0], a0[1]);
        let t1 = _mm256_hadd_ps(a0[2], a0[3]);
        let u0 = _mm256_hadd_ps(t0, t1);
        let lo = _mm256_castps256_ps128(u0);
        let hi = _mm256_extractf128_ps(u0, 1);
        _mm_storeu_ps(
            s0.as_mut_ptr().add(j),
            _mm_mul_ps(_mm_add_ps(lo, hi), _mm256_castps256_ps128(sc)),
        );
        let t0 = _mm256_hadd_ps(a1[0], a1[1]);
        let t1 = _mm256_hadd_ps(a1[2], a1[3]);
        let u0 = _mm256_hadd_ps(t0, t1);
        let lo = _mm256_castps256_ps128(u0);
        let hi = _mm256_extractf128_ps(u0, 1);
        _mm_storeu_ps(
            s1.as_mut_ptr().add(j),
            _mm_mul_ps(_mm_add_ps(lo, hi), _mm256_castps256_ps128(sc)),
        );
        j += 4;
    }
    while j < vis {
        if j < vis0 {
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i < hd {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(q0p.add(i)),
                    _mm256_loadu_ps(kp.add(j * hd + i)),
                    acc,
                );
                i += 8;
            }
            s0[j] = hsum_grouped_ps(acc) * scale;
        }
        if j < vis1 {
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i < hd {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(q1p.add(i)),
                    _mm256_loadu_ps(kp.add(j * hd + i)),
                    acc,
                );
                i += 8;
            }
            s1[j] = hsum_grouped_ps(acc) * scale;
        }
        j += 1;
    }
}
fn scores(q: &[f32], kf: &[f32], vis: usize, hd: usize, scale: f32, s: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if hd.is_multiple_of(8) && crate::cpu_has!("avx2") && crate::cpu_has!("fma") {
            // SAFETY: features checked; `kf` holds at least `vis · hd` floats.
            return unsafe { scores_avx2(q, kf, vis, hd, scale, s) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if hd.is_multiple_of(8) && peregrine_kernels::cpu::have_neon() {
            // SAFETY: NEON checked; `kf` holds at least `vis · hd` floats.
            return unsafe { scores_neon(q, kf, vis, hd, scale, s) };
        }
    }
    for (j, sj) in s.iter_mut().enumerate().take(vis) {
        let k = &kf[j * hd..(j + 1) * hd];
        *sj = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
    }
}

/// The NEON value mix (S2.8 F5): four dimensions at a time, the keys in the scalar
/// loop's order and each accumulator updated with a multiply and then an add, so the
/// result is bit-identical to [`mix`]'s scalar loop. Only the lanes change.
///
/// # Safety
/// The CPU must have NEON; `vf` holds at least `p.len() · hd` floats.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mix_neon(o: &mut [f32], alpha: f32, p: &[f32], vf: &[f32], hd: usize) {
    use std::arch::aarch64::*;
    if alpha != 1.0 {
        o.iter_mut().for_each(|x| *x *= alpha);
    }
    for (j, &w) in p.iter().enumerate() {
        let v = &vf[j * hd..(j + 1) * hd];
        let wv = vdupq_n_f32(w);
        let mut d = 0;
        while d + 4 <= hd {
            let ov = vld1q_f32(o.as_ptr().add(d));
            let vv = vld1q_f32(v.as_ptr().add(d));
            vst1q_f32(o.as_mut_ptr().add(d), vaddq_f32(ov, vmulq_f32(wv, vv)));
            d += 4;
        }
        while d < hd {
            o[d] += w * v[d];
            d += 1;
        }
    }
}

/// The canonical aarch64 attention-score reduction (S2.8 F5).
///
/// One key's score is accumulated into two `f32x4` accumulators — dimensions
/// `0..4` and `4..8` modulo 8 — walking the head ascending, then reduced as
/// `((d0+d1)+(d2+d3)) + ((d4+d5)+(d6+d7))`. That is the AVX2 tier's
/// `hsum_grouped_ps` tree with four lanes per half: the same association of the same
/// products, and the same fused multiply-add, so a key scored on this host has the
/// bits it would have on the AVX2 tier. It is defined once, here, and every aarch64
/// path uses it: the single-row decode, the prefill tile and the multi-user mapped
/// path all reach `scores`/`scores_f16`, so a row's scores never depend on which one
/// ran. A key's score is independent of the number of keys scored in one pass, which
/// is why grouping (four keys at a time, then one at a time) cannot move a bit. The
/// aarch64 goldens were re-recorded when this order replaced the scalar sum.
///
/// # Safety
/// The CPU must have NEON; `vis · hd` floats (or halves) must be readable.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn hsum4_neon(v: std::arch::aarch64::float32x4_t) -> f32 {
    use std::arch::aarch64::*;
    let t = vpaddq_f32(v, v);
    let t = vpaddq_f32(t, t);
    vgetq_lane_f32(t, 0)
}

/// [`scores`]'s dot, four keys per pass, on NEON.
///
/// # Safety
/// The CPU must have NEON; `kf` holds at least `vis · hd` floats and `s` `vis`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scores_neon(q: &[f32], kf: &[f32], vis: usize, hd: usize, scale: f32, s: &mut [f32]) {
    use std::arch::aarch64::*;
    let qp = q.as_ptr();
    let kp = kf.as_ptr();
    let mut j = 0;
    while j + 4 <= vis {
        let mut a0 = [vdupq_n_f32(0.0); 4];
        let mut a1 = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i < hd {
            let q0 = vld1q_f32(qp.add(i));
            let q1 = vld1q_f32(qp.add(i + 4));
            for u in 0..4 {
                let k0 = vld1q_f32(kp.add((j + u) * hd + i));
                let k1 = vld1q_f32(kp.add((j + u) * hd + i + 4));
                a0[u] = vfmaq_f32(a0[u], q0, k0);
                a1[u] = vfmaq_f32(a1[u], q1, k1);
            }
            i += 8;
        }
        for u in 0..4 {
            s[j + u] = (hsum4_neon(a0[u]) + hsum4_neon(a1[u])) * scale;
        }
        j += 4;
    }
    while j < vis {
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i < hd {
            a0 = vfmaq_f32(a0, vld1q_f32(qp.add(i)), vld1q_f32(kp.add(j * hd + i)));
            a1 = vfmaq_f32(
                a1,
                vld1q_f32(qp.add(i + 4)),
                vld1q_f32(kp.add(j * hd + i + 4)),
            );
            i += 8;
        }
        s[j] = (hsum4_neon(a0) + hsum4_neon(a1)) * scale;
        j += 1;
    }
}

/// [`scores`] on a F16 cache tile: `vcvt_f32_f16` in registers, then the same tree as
/// [`scores_neon`]. The conversion is exact, so this is bit-identical to widening the
/// tile first — the test pins that — and it halves the bytes the tile reads.
///
/// # Safety
/// The CPU must have NEON; `k` holds at least `vis · hd · 2` bytes and `s` `vis`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scores_f16(q: &[f32], k: &[u8], vis: usize, hd: usize, scale: f32, s: &mut [f32]) {
    use std::arch::aarch64::*;
    let qp = q.as_ptr();
    let kp = k.as_ptr();
    let mut j = 0;
    while j + 4 <= vis {
        let mut a0 = [vdupq_n_f32(0.0); 4];
        let mut a1 = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i < hd {
            let q0 = vld1q_f32(qp.add(i));
            let q1 = vld1q_f32(qp.add(i + 4));
            for u in 0..4 {
                let at = ((j + u) * hd + i) * 2;
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                a0[u] = vfmaq_f32(a0[u], q0, k0);
                a1[u] = vfmaq_f32(a1[u], q1, k1);
            }
            i += 8;
        }
        for u in 0..4 {
            s[j + u] = (hsum4_neon(a0[u]) + hsum4_neon(a1[u])) * scale;
        }
        j += 4;
    }
    while j < vis {
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i < hd {
            let at = ((j * hd) + i) * 2;
            a0 = vfmaq_f32(
                a0,
                vld1q_f32(qp.add(i)),
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16))),
            );
            a1 = vfmaq_f32(
                a1,
                vld1q_f32(qp.add(i + 4)),
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16))),
            );
            i += 8;
        }
        s[j] = (hsum4_neon(a0) + hsum4_neon(a1)) * scale;
        j += 1;
    }
}

/// Two query rows scored from one F16 K vector, two keys per pass.
///
/// Each row runs the canonical reduction on its own accumulators; only the loaded and
/// widened key is shared, so the sharing cannot move a bit. Rows may have different
/// visible lengths; beyond a row's own length the key is loaded but its score is not
/// stored.
///
/// # Safety
/// The CPU must have NEON; `k` holds `max(vis0, vis1) · hd · 2` bytes and each `s`
/// row `vis0`/`vis1`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn scores_f16_pair(
    q0: &[f32],
    q1: &[f32],
    k: &[u8],
    vis0: usize,
    vis1: usize,
    hd: usize,
    scale: f32,
    s0: &mut [f32],
    s1: &mut [f32],
) {
    use std::arch::aarch64::*;
    let (q0p, q1p, kp) = (q0.as_ptr(), q1.as_ptr(), k.as_ptr());
    let vis = vis0.max(vis1);
    let common = vis0.min(vis1);
    let mut j = 0;
    while j + 2 <= common {
        let mut a00 = [vdupq_n_f32(0.0); 2];
        let mut a01 = [vdupq_n_f32(0.0); 2];
        let mut a10 = [vdupq_n_f32(0.0); 2];
        let mut a11 = [vdupq_n_f32(0.0); 2];
        let mut i = 0;
        while i < hd {
            let q00 = vld1q_f32(q0p.add(i));
            let q01 = vld1q_f32(q0p.add(i + 4));
            let q10 = vld1q_f32(q1p.add(i));
            let q11 = vld1q_f32(q1p.add(i + 4));
            for u in 0..2 {
                let at = ((j + u) * hd + i) * 2;
                let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16)));
                let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16)));
                a00[u] = vfmaq_f32(a00[u], q00, k0);
                a01[u] = vfmaq_f32(a01[u], q01, k1);
                a10[u] = vfmaq_f32(a10[u], q10, k0);
                a11[u] = vfmaq_f32(a11[u], q11, k1);
            }
            i += 8;
        }
        for u in 0..2 {
            s0[j + u] = (hsum4_neon(a00[u]) + hsum4_neon(a01[u])) * scale;
            s1[j + u] = (hsum4_neon(a10[u]) + hsum4_neon(a11[u])) * scale;
        }
        j += 2;
    }
    while j < vis {
        for (r, qp) in [(0usize, q0p), (1usize, q1p)] {
            let own = if r == 0 { vis0 } else { vis1 };
            if j >= own {
                continue;
            }
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i < hd {
                let at = ((j * hd) + i) * 2;
                a0 = vfmaq_f32(
                    a0,
                    vld1q_f32(qp.add(i)),
                    vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at) as *const u16))),
                );
                a1 = vfmaq_f32(
                    a1,
                    vld1q_f32(qp.add(i + 4)),
                    vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kp.add(at + 8) as *const u16))),
                );
                i += 8;
            }
            let score = (hsum4_neon(a0) + hsum4_neon(a1)) * scale;
            if r == 0 {
                s0[j] = score;
            } else {
                s1[j] = score;
            }
        }
        j += 1;
    }
}

/// [`mix`] on a F16 V tile: `vcvt_f32_f16` in registers and the scalar loop's
/// multiply-then-add per element, in the same ascending-position order, so the result
/// is bit-identical to [`mix_neon`] on a widened tile.
///
/// # Safety
/// The CPU must have NEON; `v` holds at least `p.len() · hd · 2` bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mix_f16(o: &mut [f32], alpha: f32, p: &[f32], v: &[u8], hd: usize) {
    use std::arch::aarch64::*;
    if alpha != 1.0 {
        o.iter_mut().for_each(|x| *x *= alpha);
    }
    let vp = v.as_ptr();
    for (j, &w) in p.iter().enumerate() {
        let wv = vdupq_n_f32(w);
        let row = j * hd;
        let mut d = 0;
        while d + 8 <= hd {
            let v0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(
                vp.add((row + d) * 2) as *const u16
            )));
            let v1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(
                vp.add((row + d + 4) * 2) as *const u16
            )));
            let o0 = vld1q_f32(o.as_ptr().add(d));
            let o1 = vld1q_f32(o.as_ptr().add(d + 4));
            vst1q_f32(o.as_mut_ptr().add(d), vaddq_f32(o0, vmulq_f32(wv, v0)));
            vst1q_f32(o.as_mut_ptr().add(d + 4), vaddq_f32(o1, vmulq_f32(wv, v1)));
            d += 8;
        }
        while d < hd {
            let h = u16::from_le_bytes([*vp.add((row + d) * 2), *vp.add((row + d) * 2 + 1)]);
            o[d] += w * peregrine_kernels::format::f16(h);
            d += 1;
        }
    }
}

/// `o = α·o + Σ_j p_j · v_j` over the first `p.len()` values of the decoded tile.
fn mix(o: &mut [f32], alpha: f32, p: &[f32], vf: &[f32], hd: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_attn(hd) {
            // SAFETY: AVX-512F+FMA checked; `vf` holds at least `p.len() · hd` floats.
            return unsafe { mix_avx512(o, alpha, p, vf, hd) };
        }
        if hd.is_multiple_of(8) && crate::cpu_has!("avx2") && crate::cpu_has!("fma") {
            // SAFETY: features checked; `vf` holds at least `p.len() · hd` floats.
            return unsafe { mix_avx2(o, alpha, p, vf, hd) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if hd.is_multiple_of(4) && crate::cpu::have_neon() {
            // SAFETY: NEON checked; `vf` holds at least `p.len() · hd` floats.
            return unsafe { mix_neon(o, alpha, p, vf, hd) };
        }
    }
    if alpha != 1.0 {
        o.iter_mut().for_each(|x| *x *= alpha);
    }
    for (j, &w) in p.iter().enumerate() {
        for (x, &v) in o.iter_mut().zip(&vf[j * hd..(j + 1) * hd]) {
            *x += w * v;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scores_avx2(q: &[f32], kf: &[f32], vis: usize, hd: usize, scale: f32, s: &mut [f32]) {
    use std::arch::x86_64::*;
    let qp = q.as_ptr();
    let kp = kf.as_ptr();
    let sc = _mm256_set1_ps(scale);
    let mut j = 0;
    // Eight keys per pass: eight accumulators, then one transposed reduction gives
    // all eight scores in one register — instead of eight horizontal sums.
    while j + 8 <= vis {
        let mut a = [_mm256_setzero_ps(); 8];
        let mut i = 0;
        while i < hd {
            let qv = _mm256_loadu_ps(qp.add(i));
            for (u, acc) in a.iter_mut().enumerate() {
                *acc = _mm256_fmadd_ps(qv, _mm256_loadu_ps(kp.add((j + u) * hd + i)), *acc);
            }
            i += 8;
        }
        let t0 = _mm256_hadd_ps(a[0], a[1]);
        let t1 = _mm256_hadd_ps(a[2], a[3]);
        let t2 = _mm256_hadd_ps(a[4], a[5]);
        let t3 = _mm256_hadd_ps(a[6], a[7]);
        let u0 = _mm256_hadd_ps(t0, t1);
        let u1 = _mm256_hadd_ps(t2, t3);
        let lo = _mm256_permute2f128_ps(u0, u1, 0x20);
        let hi = _mm256_permute2f128_ps(u0, u1, 0x31);
        _mm256_storeu_ps(
            s.as_mut_ptr().add(j),
            _mm256_mul_ps(_mm256_add_ps(lo, hi), sc),
        );
        j += 8;
    }
    while j < vis {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < hd {
            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(qp.add(i)),
                _mm256_loadu_ps(kp.add(j * hd + i)),
                acc,
            );
            i += 8;
        }
        s[j] = hsum_grouped_ps(acc) * scale;
        j += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mix_avx2(o: &mut [f32], alpha: f32, p: &[f32], vf: &[f32], hd: usize) {
    use std::arch::x86_64::*;
    let al = _mm256_set1_ps(alpha);
    let vp = vf.as_ptr();
    let op = o.as_mut_ptr();
    // Sixty-four dimensions at a time: eight accumulators stay in registers for the
    // whole block of values, and each position's eight value vectors are loaded into
    // registers before they are used, so the loads do not sit in the FMAs' shadow.
    let mut d0 = 0;
    while d0 + 64 <= hd {
        let mut acc = [_mm256_setzero_ps(); 8];
        for l in 0..8 {
            acc[l] = _mm256_mul_ps(_mm256_loadu_ps(op.add(d0 + l * 8)), al);
        }
        for (j, &w) in p.iter().enumerate() {
            let wv = _mm256_set1_ps(w);
            let row = vp.add(j * hd + d0);
            let mut v = [_mm256_setzero_ps(); 8];
            for l in 0..8 {
                v[l] = _mm256_loadu_ps(row.add(l * 8));
            }
            for l in 0..8 {
                acc[l] = _mm256_fmadd_ps(wv, v[l], acc[l]);
            }
        }
        for l in 0..8 {
            _mm256_storeu_ps(op.add(d0 + l * 8), acc[l]);
        }
        d0 += 64;
    }
    while d0 < hd {
        let mut acc = _mm256_mul_ps(_mm256_loadu_ps(op.add(d0)), al);
        for (j, &w) in p.iter().enumerate() {
            acc = _mm256_fmadd_ps(_mm256_set1_ps(w), _mm256_loadu_ps(vp.add(j * hd + d0)), acc);
        }
        _mm256_storeu_ps(op.add(d0), acc);
        d0 += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::KvPrec;

    fn prng(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// The NEON value mix against the scalar loop, bit for bit, at the head widths the
    /// gate models use and at every lane remainder.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_mix_is_bit_identical_to_the_scalar_mix() {
        if !crate::cpu::have_neon() {
            eprintln!("skip: no NEON");
            return;
        }
        for hd in [4usize, 7, 8, 64, 80, 100, 128] {
            for vis in [1usize, 3, 7, 64] {
                let p: Vec<f32> = prng(0x51 + vis as u64, vis)
                    .iter()
                    .map(|x| x.abs())
                    .collect();
                let vf = prng(0x77 + hd as u64, vis * hd);
                let o = prng(0x99 + hd as u64, hd);
                for alpha in [1.0f32, 0.37] {
                    let mut got = o.clone();
                    mix(&mut got, alpha, &p, &vf, hd);
                    let mut want = o.clone();
                    if alpha != 1.0 {
                        want.iter_mut().for_each(|x| *x *= alpha);
                    }
                    for (j, &w) in p.iter().enumerate() {
                        for (x, &v) in want.iter_mut().zip(&vf[j * hd..(j + 1) * hd]) {
                            *x += w * v;
                        }
                    }
                    for i in 0..hd {
                        assert_eq!(
                            got[i].to_bits(),
                            want[i].to_bits(),
                            "hd={hd} vis={vis} alpha={alpha} i={i}"
                        );
                    }
                }
            }
        }
    }

    /// The canonical aarch64 score tree, in scalar Rust: two four-lane accumulators
    /// (dimensions 0..4 and 4..8 modulo 8) over ascending eight-wide blocks, reduced
    /// as `((d0+d1)+(d2+d3)) + ((d4+d5)+(d6+d7))`. This is the specification the NEON
    /// kernels below are checked against; it is also the AVX2 tier's
    /// `hsum_grouped_ps` tree with four lanes per half.
    #[cfg(target_arch = "aarch64")]
    fn canonical_score(q: &[f32], k: &[f32]) -> f32 {
        let mut a0 = [0.0f32; 4];
        let mut a1 = [0.0f32; 4];
        let mut i = 0;
        while i + 8 <= q.len() {
            for l in 0..4 {
                a0[l] = q[i + l].mul_add(k[i + l], a0[l]);
                a1[l] = q[i + 4 + l].mul_add(k[i + 4 + l], a1[l]);
            }
            i += 8;
        }
        ((a0[0] + a0[1]) + (a0[2] + a0[3])) + ((a1[0] + a1[1]) + (a1[2] + a1[3]))
    }

    /// The dispatch layer's `scores` (which takes the NEON kernel on this host) is the
    /// canonical tree, at every visible length and every head width the models use.
    /// The group/tail split must not move a bit — that is what lets decode, a prefill
    /// tile and the mapped path share one order.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_scores_are_the_canonical_tree() {
        if !crate::cpu::have_neon() {
            eprintln!("skip: no NEON");
            return;
        }
        for hd in [8usize, 16, 64, 80, 128] {
            for vis in [1usize, 3, 4, 5, 7, 8, 13] {
                let q = prng(0x11 + hd as u64, hd);
                let kf = prng(0x22 + vis as u64, vis * hd);
                let scale = 0.125f32;
                let mut got = vec![0.0f32; vis];
                scores(&q, &kf, vis, hd, scale, &mut got);
                for j in 0..vis {
                    let key = &kf[j * hd..(j + 1) * hd];
                    let want = canonical_score(&q, key) * scale;
                    assert_eq!(got[j].to_bits(), want.to_bits(), "hd={hd} vis={vis} j={j}");
                }
            }
        }
    }

    /// A F16 key, widened in the kernel, scores exactly as the widened tile does.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_f16_scores_equal_the_widened_scores() {
        if !crate::cpu::have_neon() {
            eprintln!("skip: no NEON");
            return;
        }
        for hd in [8usize, 64, 128] {
            for vis in [1usize, 3, 4, 9] {
                let q = prng(0x33 + hd as u64, hd);
                let wide = prng(0x44 + vis as u64, vis * hd);
                let mut half = Vec::with_capacity(wide.len() * 2);
                for x in &wide {
                    half.extend_from_slice(&crate::gguf::to_f16(*x).to_le_bytes());
                }
                let widened: Vec<f32> = half
                    .chunks_exact(2)
                    .map(|b| peregrine_kernels::format::f16(u16::from_le_bytes([b[0], b[1]])))
                    .collect();
                let scale = 0.25f32;
                let mut want = vec![0.0f32; vis];
                let mut got = vec![0.0f32; vis];
                unsafe {
                    scores_neon(&q, &widened, vis, hd, scale, &mut want);
                    scores_f16(&q, &half, vis, hd, scale, &mut got);
                }
                for j in 0..vis {
                    assert_eq!(
                        got[j].to_bits(),
                        want[j].to_bits(),
                        "hd={hd} vis={vis} j={j}"
                    );
                }
            }
        }
    }

    /// The paired kernel (`run_f16`'s two query heads over one K tile, and the prefill
    /// tile's paired rows) is each row's single-row score, bit for bit, including when
    /// the two rows see different lengths.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn the_paired_f16_score_is_each_rows_own_score() {
        if !crate::cpu::have_neon() {
            eprintln!("skip: no NEON");
            return;
        }
        for hd in [8usize, 64] {
            let vis0 = 9usize;
            let vis1 = 13usize;
            let q0 = prng(0x55 + hd as u64, hd);
            let q1 = prng(0x66 + hd as u64, hd);
            let wide = prng(0x77, vis1 * hd);
            let mut half = Vec::with_capacity(wide.len() * 2);
            for x in &wide {
                half.extend_from_slice(&crate::gguf::to_f16(*x).to_le_bytes());
            }
            let scale = 0.125f32;
            let (mut s0, mut s1) = (vec![0.0f32; vis0], vec![0.0f32; vis1]);
            unsafe {
                scores_f16_pair(&q0, &q1, &half, vis0, vis1, hd, scale, &mut s0, &mut s1);
            }
            for (v, q, s) in [(vis0, &q0, &s0), (vis1, &q1, &s1)] {
                let mut want = vec![0.0f32; v];
                unsafe { scores_f16(q, &half, v, hd, scale, &mut want) };
                for j in 0..v {
                    assert_eq!(s[j].to_bits(), want[j].to_bits(), "hd={hd} vis={v} j={j}");
                }
            }
        }
    }

    /// The F16 value mix against the widened mix, bit for bit.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_mix_f16_is_bit_identical_to_the_widened_mix() {
        if !crate::cpu::have_neon() {
            eprintln!("skip: no NEON");
            return;
        }
        for hd in [8usize, 64, 128] {
            for vis in [1usize, 3, 7] {
                let p: Vec<f32> = prng(0x88 + vis as u64, vis)
                    .iter()
                    .map(|x| x.abs())
                    .collect();
                let vf = prng(0x99, vis * hd);
                let mut half = Vec::with_capacity(vf.len() * 2);
                for x in &vf {
                    half.extend_from_slice(&crate::gguf::to_f16(*x).to_le_bytes());
                }
                // The widened tile holds exactly the halves' values.
                let widened: Vec<f32> = half
                    .chunks_exact(2)
                    .map(|b| peregrine_kernels::format::f16(u16::from_le_bytes([b[0], b[1]])))
                    .collect();
                let o = prng(0xaa + hd as u64, hd);
                for alpha in [1.0f32, 0.37] {
                    let mut want = o.clone();
                    mix(&mut want, alpha, &p, &widened, hd);
                    let mut got = o.clone();
                    unsafe { mix_f16(&mut got, alpha, &p, &half, hd) };
                    for i in 0..hd {
                        assert_eq!(
                            got[i].to_bits(),
                            want[i].to_bits(),
                            "hd={hd} vis={vis} alpha={alpha} i={i}"
                        );
                    }
                }
            }
        }
    }

    /// The cache, the queries, and the keys and values exactly as written.
    type Setup = (KvCache, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>);

    /// A filled cache and queries for one layer; returns the inputs as written.
    fn setup(prec: KvPrec, sh: &Shape, layers: usize) -> Setup {
        let hd = sh.head_dim;
        let ctx = sh.start_pos + sh.nb;
        let mut kv = KvCache::new(prec, layers, sh.n_kv_head, hd, ctx + 3);
        let (mut ks, mut vs) = (Vec::new(), Vec::new());
        for p in 0..ctx {
            for h in 0..sh.n_kv_head {
                let mut k = prng((p * 31 + h * 7 + 1) as u64, hd);
                // An outlier channel, as real keys have.
                k[3] *= 6.0;
                let v = prng((p * 17 + h * 5 + 2) as u64, hd);
                kv.put(layers - 1, h, p, &k, &v);
                ks.push(k);
                vs.push(v);
            }
        }
        kv.advance(ctx);
        let q = prng(99, sh.nb * sh.n_head * hd)
            .iter()
            .map(|x| x * 2.0)
            .collect();
        (kv, q, ks, vs)
    }

    /// Naive attention in f64 from the f32 inputs: the specification.
    fn reference(sh: &Shape, q: &[f32], ks: &[Vec<f32>], vs: &[Vec<f32>]) -> Vec<f32> {
        let hd = sh.head_dim;
        let mut out = vec![0.0f32; sh.nb * sh.n_head * hd];
        for b in 0..sh.nb {
            let n_ctx = sh.start_pos + b + 1;
            let p_min = match sh.window {
                Some(w) => n_ctx.saturating_sub(w),
                None => 0,
            };
            for h in 0..sh.n_head {
                let kh = h / sh.gqa();
                let qv = &q[(b * sh.n_head + h) * hd..][..hd];
                let sc: Vec<f64> = (p_min..n_ctx)
                    .map(|p| {
                        let k = &ks[p * sh.n_kv_head + kh];
                        qv.iter()
                            .zip(k)
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum::<f64>()
                            * sh.scale as f64
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for d in 0..hd {
                    let acc: f64 = (p_min..n_ctx)
                        .enumerate()
                        .map(|(idx, p)| e[idx] * vs[p * sh.n_kv_head + kh][d] as f64)
                        .sum();
                    out[(b * sh.n_head + h) * hd + d] = (acc / z) as f32;
                }
            }
        }
        out
    }

    /// [`RowSink`] over a plain row-major buffer, for the sequential plan simulation.
    struct SliceOut<'a> {
        out: &'a mut [f32],
        qd: usize,
        hd: usize,
    }

    impl RowSink for SliceOut<'_> {
        fn head(&mut self, row: usize, head: usize) -> &mut [f32] {
            &mut self.out[row * self.qd + head * self.hd..][..self.hd]
        }
    }

    /// Run a call the way the model does: every task of the plan, then finish or
    /// combine. Workers are simulated sequentially; the partition is what is tested.
    fn grouped(kv: &KvCache, layer: usize, sh: &Shape, q: &[f32], plan: Plan) -> Vec<f32> {
        let hd = sh.head_dim;
        let gqa = sh.gqa();
        let qd = sh.q_dim();
        let mut out = vec![0.0f32; sh.nb * sh.n_head * hd];
        let mut sc = Scratch::new(sh.nb * gqa, hd);
        if plan.pos_chunks > 1 {
            // Flash decoding: a task per position chunk, merged afterwards.
            let mut parts: Vec<Vec<(Vec<f32>, f32, f64)>> = vec![Vec::new(); sh.nb * sh.n_head];
            for i in 0..plan.tasks(sh) {
                let t = task(sh, &plan, i);
                run(kv, layer, sh, q, &t, &mut sc);
                for bi in 0..t.b1 - t.b0 {
                    for g in 0..gqa {
                        let r = bi * gqa + g;
                        let h = t.kh * gqa + g;
                        let row = (t.b0 + bi) * sh.n_head + h;
                        parts[row].push((sc.o[r * hd..(r + 1) * hd].to_vec(), sc.m[r], sc.l[r]));
                    }
                }
            }
            for (row, ps) in parts.iter().enumerate() {
                let refs: Vec<(&[f32], f32, f64)> =
                    ps.iter().map(|(o, m, l)| (&o[..], *m, *l)).collect();
                // The sink of the row's query head.
                let sink = sh.sinks.map(|s| s[row % sh.n_head]);
                combine_row(
                    &refs,
                    sink,
                    kv.rotated(),
                    &mut out[row * hd..(row + 1) * hd],
                );
            }
        } else if sh.nb > 1 {
            // Rows split, each row's chunks walk inside its task.
            for i in 0..plan.tasks(sh) {
                let t = task(sh, &plan, i);
                let mut sink = SliceOut {
                    out: &mut out[t.b0 * qd..t.b1 * qd],
                    qd,
                    hd,
                };
                run_rows(kv, layer, sh, q, &t, &mut sc, &mut sink);
            }
        } else {
            for i in 0..plan.tasks(sh) {
                let t = task(sh, &plan, i);
                run(kv, layer, sh, q, &t, &mut sc);
                for g in 0..gqa {
                    let h = t.kh * gqa + g;
                    let sink = sh.sinks.map(|s| s[h]);
                    finish_row(
                        &sc.o[g * hd..(g + 1) * hd],
                        sc.l[g],
                        sc.m[g],
                        sink,
                        kv.rotated(),
                        &mut out[h * hd..(h + 1) * hd],
                    );
                }
            }
        }
        out
    }

    /// The plans a call can take: serial, the canonical plan at a few pool widths,
    /// and one forced alternate split to exercise a different chunk count.
    fn test_plans(sh: &Shape) -> Vec<Plan> {
        let ctx = sh.start_pos + sh.nb;
        let mut v = vec![Plan {
            parallel: false,
            row_blocks: 1,
            pos_chunks: 1,
        }];
        for w in [1usize, 4, 8] {
            v.push(plan(sh, w));
        }
        if sh.nb == 1 {
            let k = row_chunks(sh, ctx);
            if k > 2 {
                // A different chunk count, to prove the merge is canonical.
                v.push(Plan {
                    parallel: true,
                    row_blocks: 1,
                    pos_chunks: k - 1,
                });
            }
        } else {
            v.push(Plan {
                parallel: true,
                row_blocks: sh.nb.clamp(1, 3),
                pos_chunks: 1,
            });
        }
        v
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / y.abs().max(0.05))
            .fold(0.0, f32::max)
    }

    #[test]
    fn grouped_attention_matches_naive_attention_for_every_partition() {
        // (n_head, n_kv_head, head_dim, nb, start_pos): decode, verification, prefill,
        // GQA 1/3/4, a width that is not a multiple of eight, and contexts spanning
        // several blocks with a ragged tail.
        let shapes = [
            (8, 2, 64, 1, 0),
            (12, 4, 64, 1, 300),
            (15, 5, 64, 3, 129),
            (8, 8, 32, 5, 70),
            (32, 8, 128, 1, 700),
            (16, 4, 128, 13, 520),
            (6, 2, 24, 7, 40),
            (12, 3, 64, 64, 0),
            (8, 2, 64, 33, 200),
        ];
        for &(n_head, n_kv_head, head_dim, nb, start_pos) in &shapes {
            let sh = Shape {
                n_head,
                n_kv_head,
                head_dim,
                nb,
                start_pos,
                scale: 1.0 / (head_dim as f32).sqrt(),
                window: None,
                sinks: None,
                fast: None,
            };
            let (kv, q, ks, vs) = setup(KvPrec::F16, &sh, 2);
            // The cache holds f16; feed the reference what it actually stores.
            let rnd = |x: &Vec<f32>| {
                x.iter()
                    .map(|v| crate::gguf::f16(crate::gguf::to_f16(*v)))
                    .collect::<Vec<f32>>()
            };
            let ks16: Vec<Vec<f32>> = ks.iter().map(rnd).collect();
            let vs16: Vec<Vec<f32>> = vs.iter().map(rnd).collect();
            let want = reference(&sh, &q, &ks16, &vs16);
            for plan in test_plans(&sh) {
                let got = grouped(&kv, 1, &sh, &q, plan);
                let e = max_rel(&got, &want);
                assert!(e < 2e-4, "{sh:?} {plan:?}: max relative error {e}");
            }
        }
    }

    /// The f32 cache: attention over full-precision keys and values must be the
    /// exact computation, to the tolerance of the f32 arithmetic itself.
    ///
    /// The reference is computed in f64 from the same f32 inputs, so the only
    /// difference is the summation order and the online softmax. The tolerance is
    /// 2e-5 of the larger of the two values (the same scale the f16 test uses),
    /// and the f32 cache must be far closer to the reference than the f16 one is.
    #[test]
    fn f32_cache_attention_matches_the_exact_f32_reference() {
        let shapes = [
            (8, 2, 64, 1, 0),
            (12, 4, 64, 1, 300),
            (15, 5, 64, 3, 129),
            (16, 4, 128, 2, 300),
            (6, 2, 24, 7, 40),
        ];
        for &(n_head, n_kv_head, head_dim, nb, start_pos) in &shapes {
            let sh = Shape {
                n_head,
                n_kv_head,
                head_dim,
                nb,
                start_pos,
                scale: 1.0 / (head_dim as f32).sqrt(),
                window: None,
                sinks: None,
                fast: None,
            };
            let (kv, q, ks, vs) = setup(KvPrec::F32, &sh, 1);
            assert!(!kv.rotated());
            // The cache holds the f32 inputs exactly, so the reference is the
            // specification of the same computation.
            let want = reference(&sh, &q, &ks, &vs);
            for plan in test_plans(&sh) {
                let got = grouped(&kv, 0, &sh, &q, plan);
                let e = max_rel(&got, &want);
                assert!(e < 2e-5, "{sh:?} {plan:?}: max relative error {e}");
            }
        }
    }

    /// The point of the f32 cache: the same attention over an f16 cache is
    /// measurably further from the exact computation.
    #[test]
    fn the_f16_cache_is_further_from_the_exact_attention_than_the_f32_one() {
        let sh = Shape {
            n_head: 16,
            n_kv_head: 4,
            head_dim: 128,
            nb: 2,
            start_pos: 300,
            scale: 1.0 / 128f32.sqrt(),
            window: None,
            sinks: None,
            fast: None,
        };
        let (_, q, ks, vs) = setup(KvPrec::F32, &sh, 1);
        let want = reference(&sh, &q, &ks, &vs);
        let rms = |a: &[f32], b: &[f32]| {
            (a.iter()
                .zip(b)
                .map(|(x, y)| ((x - y) * (x - y)) as f64)
                .sum::<f64>()
                / a.len() as f64)
                .sqrt()
        };
        let (kv32, q32, _, _) = setup(KvPrec::F32, &sh, 1);
        let e32 = rms(&grouped(&kv32, 0, &sh, &q32, plan(&sh, 4)), &want);
        let (kv16, q16, _, _) = setup(KvPrec::F16, &sh, 1);
        let e16 = rms(&grouped(&kv16, 0, &sh, &q16, plan(&sh, 4)), &want);
        assert!(e32 < 1e-6, "f32 cache rms error {e32}");
        assert!(
            e16 > 10.0 * e32,
            "f16 cache rms error {e16} is not above the f32 one"
        );
    }

    /// Rotated int8: the same attention (within int8 noise), and closer to it than
    /// the unrotated int8 cache when the keys have an outlier channel.
    #[test]
    fn rotated_int8_attention_tracks_the_exact_attention() {
        let sh = Shape {
            n_head: 16,
            n_kv_head: 4,
            head_dim: 128,
            nb: 2,
            start_pos: 300,
            scale: 1.0 / 128f32.sqrt(),
            window: None,
            sinks: None,
            fast: None,
        };
        let (kv, q, ks, vs) = setup(KvPrec::I8, &sh, 1);
        assert!(kv.rotated());
        let want = reference(&sh, &q, &ks, &vs);
        let got = grouped(&kv, 0, &sh, &q, plan(&sh, 4));
        let rms = |a: &[f32], b: &[f32]| {
            (a.iter()
                .zip(b)
                .map(|(x, y)| ((x - y) * (x - y)) as f64)
                .sum::<f64>()
                / a.len() as f64)
                .sqrt()
        };
        let e = rms(&got, &want);
        assert!(e < 5e-3, "rotated int8 rms error {e}");
    }

    #[test]
    fn sliding_window_attention_matches_naive_reference() {
        let sh = Shape {
            n_head: 8,
            n_kv_head: 4,
            head_dim: 64,
            nb: 8,
            start_pos: 20,
            scale: 1.0 / 8.0,
            window: Some(6),
            sinks: None,
            fast: None,
        };
        let (kv, q, ks, vs) = setup(KvPrec::F16, &sh, 1);
        let want = reference(&sh, &q, &ks, &vs);
        let p = plan(&sh, 4);
        let got = grouped(&kv, 0, &sh, &q, p);
        assert!(max_rel(&got, &want) < 0.005);
    }

    /// The attention sinks: a learned logit per query head that joins the softmax
    /// without a value of its own (`ggml/src/ggml-cpu/ops.cpp:5883-5894`).
    ///
    /// The sink's weight goes to the denominator only, so a sink of −∞ leaves the
    /// attention unchanged and a large sink shrinks it towards zero.
    #[test]
    fn the_attention_sinks_join_the_softmax() {
        let sh = Shape {
            n_head: 4,
            n_kv_head: 2,
            head_dim: 32,
            nb: 2,
            start_pos: 5,
            scale: 0.25,
            window: None,
            sinks: None,
            fast: None,
        };
        let (kv, q, ks, vs) = setup(KvPrec::F32, &sh, 1);
        let hd = sh.head_dim;
        let gqa = sh.gqa();

        // The reference: the sink's weight is added to the denominator, and the
        // numerator keeps the values' own weights.
        let reference = |sinks: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; sh.nb * sh.n_head * hd];
            for b in 0..sh.nb {
                let n_ctx = sh.start_pos + b + 1;
                for h in 0..sh.n_head {
                    let kh = h / gqa;
                    let qv = &q[(b * sh.n_head + h) * hd..][..hd];
                    let sc: Vec<f64> = (0..n_ctx)
                        .map(|p| {
                            let k = &ks[p * sh.n_kv_head + kh];
                            qv.iter()
                                .zip(k)
                                .map(|(a, b)| *a as f64 * *b as f64)
                                .sum::<f64>()
                                * sh.scale as f64
                        })
                        .collect();
                    let sink = sinks[h] as f64;
                    let m = sc.iter().cloned().fold(sink, f64::max);
                    let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                    let z: f64 = e.iter().sum::<f64>() + (sink - m).exp();
                    for d in 0..hd {
                        let acc: f64 = (0..n_ctx)
                            .enumerate()
                            .map(|(j, p)| e[j] * vs[p * sh.n_kv_head + kh][d] as f64)
                            .sum();
                        out[(b * sh.n_head + h) * hd + d] = (acc / z) as f32;
                    }
                }
            }
            out
        };

        // No sink: the ordinary attention.
        let plain = grouped(&kv, 0, &sh, &q, plan(&sh, 4));
        let want = reference(&[f32::NEG_INFINITY; 4]);
        assert!(
            max_rel(&plain, &want) < 2e-5,
            "the sinkless attention changed"
        );

        // With sinks: the same scores, a larger denominator.
        let sinks = [-2.0f32, 0.5, 1.5, 3.0];
        let sh = Shape {
            sinks: Some(&sinks),
            fast: None,
            ..sh
        };
        let got = grouped(&kv, 0, &sh, &q, plan(&sh, 4));
        let want = reference(&sinks);
        assert!(
            max_rel(&got, &want) < 2e-5,
            "the sinks changed the attention"
        );

        // A sink that is very large leaves almost nothing for the values.
        let big = [50.0f32, 50.0, 50.0, 50.0];
        let sh = Shape {
            sinks: Some(&big),
            fast: None,
            ..sh
        };
        let got = grouped(&kv, 0, &sh, &q, plan(&sh, 4));
        assert!(
            got.iter().all(|v| v.abs() < 1e-3),
            "a large sink did not shrink the output: {got:?}"
        );
    }

    #[test]
    fn the_plan_covers_every_row_and_position_exactly_once() {
        for &(n_kv_head, nb, start_pos, workers) in &[
            (8usize, 1usize, 4000usize, 4usize),
            (2, 1, 3000, 4),
            (5, 1, 900, 4),
            (8, 64, 0, 4),
            (2, 13, 600, 4),
            (4, 64, 1000, 3),
        ] {
            let sh = Shape {
                n_head: n_kv_head * 4,
                n_kv_head,
                head_dim: 64,
                nb,
                start_pos,
                scale: 0.1,
                window: None,
                sinks: None,
                fast: None,
            };
            let p = plan(&sh, workers);
            let ctx = start_pos + nb;
            let mut seen = vec![0u32; n_kv_head * nb * ctx];
            for i in 0..p.tasks(&sh) {
                let t = task(&sh, &p, i);
                for b in t.b0..t.b1 {
                    for pos in t.p0..t.p1 {
                        seen[(t.kh * nb + b) * ctx + pos] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "{sh:?} {p:?}");
            if p.parallel {
                assert!(p.tasks(&sh) >= workers, "{sh:?} {p:?} leaves workers idle");
            }
        }
    }

    /// A windowed single row assigns only its visible positions, each exactly once,
    /// and never reads a tile before them.
    #[test]
    fn a_windowed_plan_covers_every_visible_position_exactly_once() {
        let sh = Shape {
            n_head: 14,
            n_kv_head: 2,
            head_dim: 64,
            nb: 1,
            start_pos: 4095,
            scale: 0.1,
            window: Some(320),
            sinks: None,
            fast: None,
        };
        let p = plan(&sh, 4);
        assert!(p.pos_chunks > 1, "{p:?}");
        let ctx = sh.start_pos + sh.nb;
        let v0 = visible_start(&sh, ctx);
        let mut seen = vec![0u32; ctx];
        for i in 0..p.tasks(&sh) {
            let t = task(&sh, &p, i);
            // Positions are per KV head; the first head's tasks must cover the row.
            if t.kh != 0 {
                continue;
            }
            for pos in t.p0..t.p1 {
                seen[pos] += 1;
            }
        }
        assert!(seen[..v0].iter().all(|&c| c == 0), "read before the window");
        assert!(
            seen[v0..].iter().all(|&c| c == 1),
            "visible positions were not covered exactly once"
        );
    }

    /// The invariant this whole partition exists for: the same cache row, the same
    /// query, the same bits — whether it is one row in the call or one of many.
    ///
    /// Each row of a 6-row "verify" is compared, bit for bit, with the same row
    /// decoded alone: same cache, same query, same context length. Before the
    /// per-row partition the two took different `pos_chunks` (1 vs 4 here) and
    /// differed in almost every output element.
    #[test]
    fn a_rows_attention_does_not_depend_on_the_batch_width() {
        // (n_head, n_kv_head, head_dim, start_pos, nb). The second is Qwen3-1.7B's
        // head geometry at a 129-token prompt: the last prefill chunk is one row, and
        // it clears the split gate while the same row in a 64-row chunk does not.
        for &(n_head, n_kv_head, hd, start_pos, nb) in &[
            (8usize, 2usize, 128usize, 300usize, 6usize),
            (16, 8, 128, 124, 6),
        ] {
            let base = Shape {
                n_head,
                n_kv_head,
                head_dim: hd,
                nb,
                start_pos,
                scale: 1.0 / (hd as f32).sqrt(),
                window: None,
                sinks: None,
                fast: None,
            };
            let (kv, q, _, _) = setup(KvPrec::F32, &base, 1);
            let batch = grouped(&kv, 0, &base, &q, plan(&base, 4));
            let qd = base.q_dim();
            for b in 0..nb {
                let solo = Shape {
                    nb: 1,
                    start_pos: base.start_pos + b,
                    ..base
                };
                let got = grouped(&kv, 0, &solo, &q[b * qd..], plan(&solo, 4));
                let want = &batch[b * n_head * hd..(b + 1) * n_head * hd];
                assert_eq!(
                    got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "row {b} changed when batched ({n_head}/{n_kv_head}/{hd} at {start_pos})"
                );
            }
        }
    }

    /// S3.6 step 1: the position chunks at short context, and the rule that they never
    /// follow the pool.
    #[test]
    fn the_position_split_depends_on_the_context_and_shape_not_the_pool() {
        // Qwen2.5-0.5B: fourteen query heads over two KV heads, 64-wide.
        let short = |start_pos: usize| Shape {
            n_head: 14,
            n_kv_head: 2,
            head_dim: 64,
            nb: 1,
            start_pos,
            scale: 0.125,
            window: None,
            sinks: None,
            fast: None,
        };
        // At 512 positions the old plan gave a 12-worker pool four tasks; the split
        // now uses the whole pool without needing a thread count to decide it.
        let p = plan(&short(511), 12);
        assert_eq!(p.pos_chunks, 8, "{p:?}");
        assert_eq!(p.tasks(&short(511)), 16, "{p:?}");
        assert_eq!(plan(&short(4095), 12).pos_chunks, 8);
        // Below the task-count gate there is nothing to split.
        assert_eq!(plan(&short(255), 12).pos_chunks, 1);

        // Every pool width sees the same chunk count at the same context.
        for pos in [0, 63, 127, 255, 511, 1023, 4095] {
            let want = plan(&short(pos), 7).pos_chunks;
            for workers in 1..=64 {
                assert_eq!(
                    plan(&short(pos), workers).pos_chunks,
                    want,
                    "workers {workers} at position {pos}"
                );
            }
        }

        // A multi-row call splits rows, not positions: the prefill chunk must not
        // change a row's partition.
        for nb in [2, 3, 16, 64] {
            let mut sh = short(511);
            sh.nb = nb;
            assert_eq!(plan(&sh, 12).pos_chunks, 1, "nb {nb}");
        }

        // The 1B's eight KV heads reach three chunks — the old plan's worker-derived
        // target — now that the 64-position floor lets them.
        let one_b = Shape {
            n_head: 32,
            n_kv_head: 8,
            head_dim: 64,
            nb: 1,
            start_pos: 511,
            scale: 0.125,
            window: None,
            sinks: None,
            fast: None,
        };
        assert_eq!(plan(&one_b, 12).pos_chunks, 3);
        assert_eq!(plan(&one_b, 12).tasks(&one_b), 24);
        assert_eq!(plan(&one_b, 4).tasks(&one_b), 24);

        // The cut is the even one. A block-aligned cut at 512 positions gives
        // 192/192/128, whose longest chunk is a straggler on the critical path;
        // the even cut keeps every task within one position of its siblings.
        let p = plan(&one_b, 12);
        let lens: Vec<usize> = (0..p.pos_chunks)
            .map(|i| {
                let t = task(&one_b, &p, i);
                t.p1 - t.p0
            })
            .collect();
        assert!(
            lens.iter().max().unwrap() - lens.iter().min().unwrap() <= 1,
            "{lens:?}"
        );
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// One pass of an `n`-position prompt in pieces of `chunk` rows on a pool of
    /// `threads`, the way the engine reads it.
    fn read_prompt(
        cfg: &crate::model::Config,
        kv: &KvCache,
        q: &[f32],
        n: usize,
        chunk: usize,
        threads: usize,
    ) -> Vec<f32> {
        use crate::blocks::attn::{attend, AttnScratch};
        use crate::pool::Pool;
        use crate::state::StepStats;
        let pool = Pool::new(threads);
        let mut scratch = AttnScratch::new(cfg, chunk, threads);
        let mut stats = StepStats::default();
        let qd = cfg.q_dim();
        let mut out = vec![0.0f32; n * qd];
        let mut start = 0;
        while start < n {
            let nb = chunk.min(n - start);
            let mut xb = vec![0.0f32; nb * qd];
            attend(
                cfg,
                &mut scratch,
                &q[start * qd..],
                &mut xb,
                kv,
                &pool,
                0,
                nb,
                start,
                0,
                &mut stats,
            );
            out[start * qd..(start + nb) * qd].copy_from_slice(&xb);
            start += nb;
        }
        out
    }

    /// The hard rule of the prefill tiling: a row's output bits are a function of its
    /// query, its own context and the model shape alone — never of the prompt chunk it
    /// was read in or of the pool width. Read every row of a 512-position context as a
    /// solo decode (one row per call: the F16-direct path) and then as prompt pieces of
    /// 3, 16 and 64 rows on pools of 1, 2, 4 and 8 workers (the tiled prefill path,
    /// where `gqa` rows share every loaded K and V vector). All must agree bit for bit.
    ///
    /// Three geometries: Llama-3.2's 8×2/64 (four rows per KV head, so every pair is
    /// within one batch row), a 9×3/64 (three rows per head, so a pair straddles two
    /// batch rows with different context lengths), and Qwen2.5-0.5B's 14×2/64 (seven).
    #[test]
    fn prefill_tiles_are_bit_identical_to_a_solo_decode() {
        use crate::model::Config;

        let n = 512usize;

        let mut failures = Vec::new();
        for &(n_head, n_kv_head, hd) in &[(8usize, 2usize, 64usize), (9, 3, 64), (14, 2, 64)] {
            let cfg = Config {
                arch_name: "test",
                rope_neox: false,
                n_layer: 1,
                n_kv_layer: 1,
                n_head,
                n_kv_head,
                head_dim: hd,
                d_model: n_head * hd,
                n_ff: 4 * n_head * hd,
                n_vocab: 32,
                rope_dim: hd,
                eps: 1e-5,
                rope_base: 10_000.0,
                ctx_train: 1024,
                rope_yarn: None,
            };
            let mut kv = KvCache::new(KvPrec::F16, 1, n_kv_head, hd, n + 1);
            for p in 0..n {
                for h in 0..n_kv_head {
                    let k = prng((p * 31 + h * 7 + 1) as u64, hd);
                    let v = prng((p * 17 + h * 5 + 2) as u64, hd);
                    kv.put(0, h, p, &k, &v);
                }
            }
            kv.advance(n);
            let qd = cfg.q_dim();
            let q = prng(99, n * qd);
            // The reference is the decode path: one row, on the widest pool.
            let want = read_prompt(&cfg, &kv, &q, n, 1, 4);
            for &chunk in &[3usize, 16, 64] {
                for &threads in &[1usize, 2, 4, 8] {
                    let got = read_prompt(&cfg, &kv, &q, n, chunk, threads);
                    if let Some(row) = (0..n).find(|&row| {
                        !same_bits(
                            &got[row * qd..(row + 1) * qd],
                            &want[row * qd..(row + 1) * qd],
                        )
                    }) {
                        let at = (0..qd)
                            .find(|&d| got[row * qd + d].to_bits() != want[row * qd + d].to_bits())
                            .unwrap();
                        failures.push(format!(
                            "{n_head}x{n_kv_head}/{hd}: chunk {chunk}, {threads} threads: row {row} differs at {at}"
                        ));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "the prompt tiling changed a row:\n{}",
            failures.join("\n")
        );
    }

    /// The prefill F16-direct switch has a context threshold ([`DIRECT_MIN_VIS`]):
    /// below it a task widens the tile, at and above it the task reads the cache's
    /// halves. Crossing the line must not move a row's bits.
    ///
    /// A `DIRECT_MIN_VIS + 1`-position context is read as prompt pieces of 16 rows —
    /// the pieces that end below the line widen, the ones that end at or above it go
    /// direct, so both sides and the boundary itself appear in one pass — and then
    /// row by row, which is the decode path and always direct. Qwen2.5-0.5B's
    /// 14×2/64 geometry is used, the one the direct kernels exist for.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn the_direct_prefill_threshold_does_not_move_a_bit() {
        use crate::model::Config;

        let n = DIRECT_MIN_VIS + 1;
        let (n_head, n_kv_head, hd) = (14usize, 2usize, 64usize);
        let cfg = Config {
            arch_name: "test",
            rope_neox: false,
            n_layer: 1,
            n_kv_layer: 1,
            n_head,
            n_kv_head,
            head_dim: hd,
            d_model: n_head * hd,
            n_ff: 4 * n_head * hd,
            n_vocab: 32,
            rope_dim: hd,
            eps: 1e-5,
            rope_base: 10_000.0,
            ctx_train: 2 * DIRECT_MIN_VIS,
            rope_yarn: None,
        };
        let mut kv = KvCache::new(KvPrec::F16, 1, n_kv_head, hd, n + 1);
        for p in 0..n {
            for h in 0..n_kv_head {
                let k = prng((p * 31 + h * 7 + 1) as u64, hd);
                let v = prng((p * 17 + h * 5 + 2) as u64, hd);
                kv.put(0, h, p, &k, &v);
            }
        }
        kv.advance(n);
        let qd = cfg.q_dim();
        let q = prng(99, n * qd);

        // Row by row is the always-direct decode path; pieces of 16 cross the line.
        let want = read_prompt(&cfg, &kv, &q, n, 1, 4);
        let got = read_prompt(&cfg, &kv, &q, n, 16, 4);
        let row = (0..n).find(|&row| {
            !same_bits(
                &got[row * qd..(row + 1) * qd],
                &want[row * qd..(row + 1) * qd],
            )
        });
        assert!(row.is_none(), "the switch moved row {:?}", row);

        // And the switch itself: off one position below the line, on at it.
        if crate::cpu_has!("avx2") && crate::cpu_has!("fma") && crate::cpu_has!("f16c") {
            let sh = Shape {
                n_head,
                n_kv_head,
                head_dim: hd,
                nb: 1,
                start_pos: 0,
                scale: 1.0 / 8.0,
                window: None,
                sinks: None,
                fast: None,
            };
            assert!(
                !prefill_direct(&kv, &sh, DIRECT_MIN_VIS - 1),
                "the direct path was taken below the threshold"
            );
            assert!(
                prefill_direct(&kv, &sh, DIRECT_MIN_VIS),
                "the direct path was not taken at the threshold"
            );
        }
    }

    /// The hard rule of the F16-direct path: a row's output bits are a function of
    /// that row's query, context and the model shape alone — not of the other rows in
    /// the call, the prefill chunk width, the worker count or which worker runs it.
    ///
    /// Qwen2.5-0.5B's geometry (14 query heads / 2 KV heads, 64-wide) is used because
    /// it is the model this path was written for: seven query heads share a KV head,
    /// so the row-pair kernel never degenerates to one row.
    #[test]
    fn a_f16_rows_bits_are_identical_alone_in_a_chunk_and_at_any_pool_width() {
        let base = Shape {
            n_head: 14,
            n_kv_head: 2,
            head_dim: 64,
            nb: 16,
            start_pos: 400,
            scale: 1.0 / 8.0,
            window: None,
            sinks: None,
            fast: None,
        };
        let hd = base.head_dim;
        let qd = base.q_dim();
        let (kv, q, _, _) = setup(KvPrec::F16, &base, 1);
        let pool = [1usize, 2, 4, 8];
        // The batch of 16, at each width: the table every row is checked against.
        let batched: Vec<Vec<f32>> = pool
            .iter()
            .map(|&w| grouped(&kv, 0, &base, &q, plan(&base, w)))
            .collect();
        for w in 1..batched.len() {
            for b in 0..base.nb {
                let lo = b * base.n_head * hd;
                let hi = (b + 1) * base.n_head * hd;
                assert!(
                    same_bits(&batched[w][lo..hi], &batched[0][lo..hi]),
                    "row {b} moved with the pool width {}",
                    pool[w]
                );
            }
        }
        for (wi, batch) in batched.iter().enumerate() {
            for b in 0..base.nb {
                // The same row read alone: its context ends where its batch row's does.
                let solo = Shape {
                    nb: 1,
                    start_pos: base.start_pos + b,
                    ..base
                };
                let got = grouped(&kv, 0, &solo, &q[b * qd..], plan(&solo, pool[wi]));
                let lo = b * base.n_head * hd;
                assert!(
                    same_bits(&got, &batch[lo..(b + 1) * base.n_head * hd]),
                    "row {b} changed in a chunk of 16 at pool width {}",
                    pool[wi]
                );
                // ... and in a chunk of 3, where the row sharing changes entirely.
                if b + 3 <= base.nb {
                    let small = Shape {
                        nb: 3,
                        start_pos: base.start_pos + b,
                        ..base
                    };
                    let got = grouped(&kv, 0, &small, &q[b * qd..], plan(&small, pool[wi]));
                    assert!(
                        same_bits(&got[..base.n_head * hd], &batch[lo..lo + base.n_head * hd]),
                        "row {b} changed in a chunk of 3 at pool width {}",
                        pool[wi]
                    );
                }
            }
        }
    }

    /// The sixteen-lane value mix is the eight-lane mix widened: each output
    /// element sees `α·o` then `p_j·v_j` over ascending positions, in the same
    /// order and with the same operands (F16C's widening is exact). The tier's
    /// goldens therefore keep the AVX2 tier's bits for attention.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn the_sixteen_lane_mix_is_the_eight_lane_mix_bit_for_bit() {
        if !avx512_attn(64) {
            return;
        }
        let hd = 64;
        let n = 17;
        let mut c = KvCache::new(KvPrec::F16, 1, 1, hd, n + 2);
        for p in 0..n {
            let k = prng((p * 13 + 3) as u64, hd);
            let v = prng((p * 29 + 5) as u64, hd);
            c.put(0, 0, p, &k, &v);
        }
        c.advance(n);
        let vb = c.block_bytes(false, 0, 0, 0, n);
        let mut vf = vec![0.0f32; n * hd];
        c.decode_block(false, 0, 0, 0, n, &mut vf);
        let p: Vec<f32> = prng(91, n).iter().map(|x| x.abs() + 0.01).collect();
        let one = [usize::MAX];
        for vis in 1..=n {
            let o = prng(123, hd);
            let (mut a, mut b) = (o.clone(), o.clone());
            // SAFETY: AVX2+FMA+F16C and AVX-512 checked.
            unsafe {
                mix_f16_avx2(&mut a, 0.75, &p[..vis], vb, hd);
                mix_f16_avx512(&mut b, 0.75, &p[..vis], vb, hd);
            }
            assert!(same_bits(&a, &b), "mix at vis {vis}");
            let (mut a, mut b) = (o.clone(), o.clone());
            mix(&mut a, 0.75, &p[..vis], &vf, hd);
            // SAFETY: checked above.
            unsafe { mix_avx512(&mut b, 0.75, &p[..vis], &vf, hd) };
            assert!(same_bits(&a, &b), "widened mix at vis {vis}");
        }
        // The pair kernel shares each V read between two rows; each row's own
        // accumulation still has the single kernel's order.
        let mut failures = Vec::new();
        for vis0 in 1..=n {
            for vis1 in 1..=n {
                let o0: Vec<f32> = prng(123, hd);
                let o1: Vec<f32> = prng(124, hd);
                let (mut x0, mut x1) = (o0.clone(), o1.clone());
                let (mut y0, mut y1) = (o0.clone(), o1.clone());
                unsafe {
                    mix(&mut x0, 0.75, &p[..vis0], &vf, hd);
                    mix(&mut x1, 0.75, &p[..vis1], &vf, hd);
                    mix_f16_pair_avx512(
                        &mut y0,
                        &mut y1,
                        0.75,
                        0.75,
                        &p[..vis0],
                        &p[..vis1],
                        vis0,
                        vis1,
                        0,
                        0,
                        0,
                        &one,
                        &one,
                        vb,
                        hd,
                        &mut [],
                        &mut [],
                    );
                }
                if !(same_bits(&x0, &y0) && same_bits(&x1, &y1)) {
                    failures.push(format!("pair mix {vis0} {vis1}"));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    /// The prefill tile's two-row value mix is each row's own mix: same operands, same
    /// ascending positions, per-row accumulators — only the loaded V vector and its
    /// widening are shared. Checked against the single-row kernel for every visible
    /// length the task names, the head widths the models use (64, 128) and one that
    /// takes the eight-lane body (72, not a multiple of sixteen), on random scores and
    /// on the edge values a softmax can produce (zeros, a whole row of equal weights,
    /// tiny and huge alphas).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn the_prefill_pair_mix_is_each_rows_own_mix() {
        if !(crate::cpu_has!("avx2") && crate::cpu_has!("fma") && crate::cpu_has!("f16c")) {
            return;
        }
        let mut vis: Vec<usize> = (1..=70).collect();
        vis.extend([127, 128, 129, 511, 512, 513]);
        for hd in [64usize, 128, 72, 80, 16] {
            for &n in &vis {
                let mut c = KvCache::new(KvPrec::F16, 1, 1, hd, n + 2);
                for p in 0..n {
                    let k = prng((p * 13 + 3) as u64 + hd as u64, hd);
                    let v = prng((p * 29 + 5) as u64 + hd as u64, hd);
                    c.put(0, 0, p, &k, &v);
                }
                c.advance(n);
                let vb = c.block_bytes(false, 0, 0, 0, n);
                let p0: Vec<f32> = prng(91 + hd as u64, n)
                    .iter()
                    .map(|x| x.abs() + 0.01)
                    .collect();
                let p1: Vec<f32> = prng(92 + hd as u64, n)
                    .iter()
                    .map(|x| (x * 3.0).abs() + 0.001)
                    .collect();
                let cases: [(&[f32], &[f32], f32, f32); 5] = [
                    (&p0, &p1, 0.75, 0.25),
                    (&p0, &p1, 1.0, 1.0),
                    (&p0, &p1, 0.0, 0.0),
                    (&p0[..], &p0[..], 1.0, 0.5),
                    (&p0, &p1, -0.0, 3.5),
                ];
                for (a, b, al0, al1) in cases {
                    let o0 = prng(123 + hd as u64, hd);
                    let o1 = prng(124 + hd as u64, hd);
                    let (mut x0, mut x1) = (o0.clone(), o1.clone());
                    let (mut y0, mut y1) = (o0.clone(), o1.clone());
                    // SAFETY: AVX2+FMA+F16C checked; `vb` holds `n · hd` halves.
                    unsafe {
                        mix_f16(&mut x0, al0, a, vb, hd);
                        mix_f16(&mut x1, al1, b, vb, hd);
                        mix_f16_pair_same(&mut y0, &mut y1, al0, al1, a, b, vb, hd);
                    }
                    assert!(same_bits(&x0, &y0), "row 0 at hd {hd} vis {n}");
                    assert!(same_bits(&x1, &y1), "row 1 at hd {hd} vis {n}");
                }
            }
        }
        // The long lengths the task names, at the two widths a real model uses.
        for hd in [64usize, 128] {
            for n in [4095usize, 4096] {
                let mut c = KvCache::new(KvPrec::F16, 1, 1, hd, n + 2);
                for p in (0..n).step_by(7) {
                    let k = prng((p * 13 + 3) as u64, hd);
                    let v = prng((p * 29 + 5) as u64, hd);
                    c.put(0, 0, p, &k, &v);
                }
                c.advance(n);
                let vb = c.block_bytes(false, 0, 0, 0, n);
                let p0: Vec<f32> = (0..n).map(|i| ((i * 37 % 251) as f32).recip()).collect();
                let p1: Vec<f32> = (0..n).map(|i| ((i * 53 % 199) as f32).recip()).collect();
                let o0 = prng(7, hd);
                let o1 = prng(8, hd);
                let (mut x0, mut x1) = (o0.clone(), o1.clone());
                let (mut y0, mut y1) = (o0.clone(), o1.clone());
                // SAFETY: AVX2+FMA+F16C checked.
                unsafe {
                    mix_f16(&mut x0, 0.5, &p0, vb, hd);
                    mix_f16(&mut x1, 0.5, &p1, vb, hd);
                    mix_f16_pair_same(&mut y0, &mut y1, 0.5, 0.5, &p0, &p1, vb, hd);
                }
                assert!(same_bits(&x0, &y0), "long row 0 at hd {hd} vis {n}");
                assert!(same_bits(&x1, &y1), "long row 1 at hd {hd} vis {n}");
            }
        }
    }

    /// S4.9s: the aarch64 two-row mix is each row's own single-row mix — same
    /// operands, same ascending positions, per-row accumulators in registers. Only
    /// the loaded V half-vector and its widening are shared, and the output stays in
    /// a register while the keys walk instead of being reloaded per key.
    ///
    /// The same coverage as the x86 pair test: every visible length 1..=70,
    /// 127/128/129 and 511/512/513, the widths the models use (64, 128), widths that
    /// take the eight-lane body and a scalar tail (72, 80, 16), and the edge alphas
    /// a softmax can produce.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn the_neon_pair_mix_is_each_rows_own_mix() {
        let mut vis: Vec<usize> = (1..=70).collect();
        vis.extend([127, 128, 129, 511, 512, 513]);
        for hd in [64usize, 128, 72, 80, 16] {
            for &n in &vis {
                let mut c = KvCache::new(KvPrec::F16, 1, 1, hd, n + 2);
                for p in 0..n {
                    let k = prng((p * 13 + 3) as u64 + hd as u64, hd);
                    let v = prng((p * 29 + 5) as u64 + hd as u64, hd);
                    c.put(0, 0, p, &k, &v);
                }
                c.advance(n);
                let vb = c.block_bytes(false, 0, 0, 0, n);
                let p0: Vec<f32> = prng(91 + hd as u64, n)
                    .iter()
                    .map(|x| x.abs() + 0.01)
                    .collect();
                let p1: Vec<f32> = prng(92 + hd as u64, n)
                    .iter()
                    .map(|x| (x * 3.0).abs() + 0.001)
                    .collect();
                let cases: [(&[f32], &[f32], f32, f32); 5] = [
                    (&p0, &p1, 0.75, 0.25),
                    (&p0, &p1, 1.0, 1.0),
                    (&p0, &p1, 0.0, 0.0),
                    (&p0[..], &p0[..], 1.0, 0.5),
                    (&p0, &p1, -0.0, 3.5),
                ];
                for (a, b, al0, al1) in cases {
                    let o0 = prng(123 + hd as u64, hd);
                    let o1 = prng(124 + hd as u64, hd);
                    let (mut x0, mut x1) = (o0.clone(), o1.clone());
                    let (mut y0, mut y1) = (o0.clone(), o1.clone());
                    // SAFETY: NEON is this target's baseline; `vb` holds `n · hd` halves.
                    unsafe {
                        mix_f16(&mut x0, al0, a, vb, hd);
                        mix_f16(&mut x1, al1, b, vb, hd);
                        peregrine_kernels::neon::mix_f16_pair(
                            &mut y0, &mut y1, al0, al1, a, b, vb, hd,
                        );
                        // The scalar twin of the pair kernel agrees too.
                        let (mut z0, mut z1) = (o0.clone(), o1.clone());
                        peregrine_kernels::neon::mix_f16_pair_scalar(
                            &mut z0, &mut z1, al0, al1, a, b, vb, hd,
                        );
                        assert!(same_bits(&z0, &y0), "twin row 0 at hd {hd} vis {n}");
                        assert!(same_bits(&z1, &y1), "twin row 1 at hd {hd} vis {n}");
                    }
                    assert!(same_bits(&x0, &y0), "row 0 at hd {hd} vis {n}");
                    assert!(same_bits(&x1, &y1), "row 1 at hd {hd} vis {n}");
                }
            }
        }
    }

    /// The four-row score tile (S4.9t) is each row's own single-row score, bit for
    /// bit: only the loaded and widened key is shared. Tested over every visible
    /// length 1..=70 and 127/128/129, 511/512/513 — at a packed stride and a padded
    /// one — for head dims 64, 128, 80, 72 and 16, on random keys and on the values a
    /// saturated softmax produces, against both the single-row kernel and the scalar
    /// twin.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn the_four_row_score_tile_is_each_rows_own_score() {
        if !crate::cpu::have_neon() {
            eprintln!("skip: no NEON");
            return;
        }
        let mut vis: Vec<usize> = (1..=70).collect();
        vis.extend([127, 128, 129, 511, 512, 513]);
        for hd in [64usize, 128, 80, 72, 16] {
            for &n in &vis {
                let q: Vec<f32> = prng(0x91 + (hd * 977 + n) as u64, 4 * hd);
                let edge = n % 3 == 0;
                let wide: Vec<f32> = (0..n * hd)
                    .map(|i| {
                        if edge {
                            // The values a saturated softmax leaves in a K tile: signed
                            // zeros, the smallest subnormal, ±1 and the format's maximum.
                            const EDGE: [f32; 8] =
                                [0.0, -0.0, 6.0e-8, 1.0, -1.0, 65504.0, -65504.0, 0.5];
                            EDGE[(i * 7 + i / 5) % EDGE.len()]
                        } else {
                            prng(0x37 + (i * 13) as u64, 1)[0]
                        }
                    })
                    .collect();
                let mut half = Vec::with_capacity(wide.len() * 2);
                for x in &wide {
                    half.extend_from_slice(&crate::gguf::to_f16(*x).to_le_bytes());
                }
                let scale = 0.08838835f32;
                for stride in [n, n + 13] {
                    let mut got = vec![0.0f32; 3 * stride + n];
                    let mut twin = got.clone();
                    // SAFETY: NEON is this target's baseline; the slices hold four rows.
                    unsafe {
                        peregrine_kernels::neon::scores_f16_tile4(
                            &q, &half, n, hd, scale, &mut got, stride,
                        );
                        peregrine_kernels::neon::scores_f16_tile4_scalar(
                            &q, &half, n, hd, scale, &mut twin, stride,
                        );
                    }
                    for r in 0..4 {
                        let mut want = vec![0.0f32; n];
                        // SAFETY: `half` holds n · hd halves.
                        unsafe {
                            scores_f16(&q[r * hd..(r + 1) * hd], &half, n, hd, scale, &mut want)
                        };
                        for j in 0..n {
                            assert_eq!(
                                got[r * stride + j].to_bits(),
                                want[j].to_bits(),
                                "hd={hd} vis={n} stride={stride} row={r} key={j}"
                            );
                            assert_eq!(
                                got[r * stride + j].to_bits(),
                                twin[r * stride + j].to_bits(),
                                "twin hd={hd} vis={n} stride={stride} row={r} key={j}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The four-row value mix is each row's own mix: its own accumulator, its own α
    /// and its own ascending multiply-then-add, with only the loaded and widened V
    /// half shared. Held to the single-row kernel (`mix_f16`), the pair body and the
    /// scalar twin on the same length and head-width grid as the score tile, with the
    /// alphas a softmax produces (0, 1, −0.0, 3.5) and weights that include zeros.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn the_four_row_mix_is_each_rows_own_mix() {
        if !crate::cpu::have_neon() {
            eprintln!("skip: no NEON");
            return;
        }
        let mut vis: Vec<usize> = (1..=70).collect();
        vis.extend([127, 128, 129, 511, 512, 513]);
        for hd in [64usize, 128, 80, 72, 16] {
            for &n in &vis {
                let mut c = KvCache::new(KvPrec::F16, 1, 1, hd, n + 2);
                for p in 0..n {
                    let k = prng((p * 13 + 3) as u64 + hd as u64, hd);
                    let v = prng((p * 29 + 5) as u64 + hd as u64, hd);
                    c.put(0, 0, p, &k, &v);
                }
                c.advance(n);
                let vb = c.block_bytes(false, 0, 0, 0, n);
                let pads: Vec<f32> = prng(0xa1 + (hd * 31 + n) as u64, 4 * n)
                    .iter()
                    .map(|x| {
                        if n % 3 == 0 {
                            x.abs() * 0.5
                        } else {
                            x.abs() + 0.01
                        }
                    })
                    .collect();
                let cases: [(f32, f32, f32, f32); 5] = [
                    (0.75, 0.25, 1.0, 0.5),
                    (1.0, 1.0, 1.0, 1.0),
                    (0.0, 0.0, 1.0, 1.0),
                    (1.0, 0.5, 1.0, 1.0),
                    (-0.0, 3.5, 1.0, 1.0),
                ];
                for (al0, al1, al2, al3) in cases {
                    let alphas = [al0, al1, al2, al3];
                    {
                        let o: Vec<f32> = prng(0xb2 + (hd * 71 + n) as u64, 4 * hd);
                        let mut got = o.clone();
                        // SAFETY: NEON is this target's baseline; `vb` holds n · hd halves
                        // and `pads` four rows at stride n.
                        unsafe {
                            peregrine_kernels::neon::mix_f16_quad(
                                &mut got, &alphas, &pads, n, n, vb, hd,
                            );
                        }
                        let mut twin = o.clone();
                        peregrine_kernels::neon::mix_f16_quad_scalar(
                            &mut twin, &alphas, &pads, n, n, vb, hd,
                        );
                        for r in 0..4 {
                            let mut want = o[r * hd..(r + 1) * hd].to_vec();
                            // SAFETY: `vb` holds n · hd halves and `want` one row.
                            unsafe {
                                mix_f16(&mut want, alphas[r], &pads[r * n..(r + 1) * n], vb, hd)
                            };
                            for d in 0..hd {
                                assert_eq!(
                                    got[r * hd + d].to_bits(),
                                    want[d].to_bits(),
                                    "hd={hd} vis={n} row={r} d={d} alpha={:?}",
                                    alphas
                                );
                                assert_eq!(
                                    got[r * hd + d].to_bits(),
                                    twin[r * hd + d].to_bits(),
                                    "twin hd={hd} vis={n} row={r} d={d} alpha={:?}",
                                    alphas
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The four-row online soft-max (S4.9t) is each row's own: the same running
    /// maximum, the same exponentials in the tile, and the same ascending `f64`
    /// denominator as row-at-a-time; only the four add chains are interleaved.
    ///
    /// Checked twice: the kernel on every length 1..=70 and 127/128/129, 511/512/513 at
    /// a padded stride, and the tile walk itself at every piece length up to `BLOCK`.
    #[test]
    fn the_four_row_softmax_is_each_rows_own_softmax() {
        // The two kernels agree where they share a defined order: aarch64 (see
        // `softmax_rows`). On x86-64 the decoder's soft-max folds eight-value groups.
        #[cfg(target_arch = "aarch64")]
        {
            let mut vis: Vec<usize> = (1..=70).collect();
            vis.extend([127, 128, 129, 511, 512, 513]);
            for &n in &vis {
                for edge in [false, true] {
                    let stride = n + 5;
                    let raw: Vec<f32> = (0..4 * stride)
                        .map(|i| {
                            let x = prng(0x5e + (i * 7) as u64, 1)[0] * 3.0;
                            if edge {
                                const EDGE: [f32; 8] =
                                    [0.0, -0.0, 3.5, -3.5, -88.0, -1.0e30, 88.0, 0.5];
                                if i % 11 == 0 {
                                    EDGE[i % EDGE.len()]
                                } else {
                                    x
                                }
                            } else {
                                x
                            }
                        })
                        .collect();
                    let maxes = [
                        crate::vmath::max(&raw[..n]) + 0.25,
                        crate::vmath::max(&raw[stride..stride + n]) - 0.5,
                        crate::vmath::max(&raw[2 * stride..2 * stride + n]),
                        crate::vmath::max(&raw[3 * stride..3 * stride + n]) + 1.0,
                    ];
                    let mut four = raw.clone();
                    let mut sums4 = [0.0f64; 4];
                    peregrine_kernels::bidir::exp_rows4_sums(
                        &mut four, stride, n, maxes, &mut sums4,
                    );
                    for r in 0..4 {
                        let mut one = raw.clone();
                        let sum = crate::vmath::exp_shift_sum(
                            &mut one[r * stride..r * stride + n],
                            maxes[r],
                        );
                        assert_eq!(
                            sum.to_bits(),
                            sums4[r].to_bits(),
                            "row {r} denominator at n={n} edge={edge}"
                        );
                        for j in 0..n {
                            assert_eq!(
                                four[r * stride + j].to_bits(),
                                one[r * stride + j].to_bits(),
                                "row {r} key {j} at n={n} edge={edge}"
                            );
                        }
                    }
                }
            }
        }

        // The tile walk: four rows at `BLOCK` stride against the same rows one at a
        // time, with the prior `m`/`l` a previous tile leaves.
        let mut sc = Scratch::new(4, 16);
        for n in 1..=BLOCK {
            for edge in [false, true] {
                let fill: Vec<f32> = (0..MAX_PARTIAL_ROWS * BLOCK)
                    .map(|i| {
                        let x = prng(0x77 + (i * 13) as u64, 1)[0] * 5.0;
                        if edge && i % 17 == 0 {
                            -88.0
                        } else {
                            x
                        }
                    })
                    .collect();
                let m0 = [-2.0f32, 1.5, f32::NEG_INFINITY, 0.25];
                let l0 = [3.5f64, 0.0, 1.0e-30, 7.25];

                sc.s_tile.copy_from_slice(&fill);
                sc.m[..4].copy_from_slice(&m0);
                sc.l[..4].copy_from_slice(&l0);
                let mut four_alpha = [1.0f32; 4];
                softmax_rows(&mut sc, 0, 0, n, 4, &mut four_alpha);
                let four_tile = sc.s_tile.clone();
                let (four_m, four_l) = (sc.m[..4].to_vec(), sc.l[..4].to_vec());

                sc.s_tile.copy_from_slice(&fill);
                sc.m[..4].copy_from_slice(&m0);
                sc.l[..4].copy_from_slice(&l0);
                let mut one_alpha = [1.0f32; 4];
                for r in 0..4 {
                    let mut a = [1.0f32; 4];
                    softmax_rows(&mut sc, r, 0, n, 1, &mut a);
                    one_alpha[r] = a[0];
                }
                for r in 0..4 {
                    assert_eq!(
                        four_alpha[r].to_bits(),
                        one_alpha[r].to_bits(),
                        "row {r} alpha at n={n} edge={edge}"
                    );
                    assert_eq!(
                        four_m[r].to_bits(),
                        sc.m[r].to_bits(),
                        "row {r} max at n={n} edge={edge}"
                    );
                    assert_eq!(
                        four_l[r].to_bits(),
                        sc.l[r].to_bits(),
                        "row {r} l at n={n} edge={edge}"
                    );
                    for j in 0..n {
                        assert_eq!(
                            four_tile[r * BLOCK + j].to_bits(),
                            sc.s_tile[r * BLOCK + j].to_bits(),
                            "row {r} key {j} at n={n} edge={edge}"
                        );
                    }
                }
            }
        }
    }

    /// The widened tile's two-row value mix is each row's own mix: same f32 values,
    /// same ascending positions, per-row accumulators — only the loaded value is
    /// shared. Checked against the single-row kernel on the same grid as the F16 pair
    /// body: every visible length 1…70 and 127/128/129, 511/512/513, 4095/4096; head
    /// dims 64 and 128 (sixteen-lane body), 80 and 72 (eight-lane body) and 16; on
    /// random scores and the edge values a softmax can produce.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn the_prefill_f32_pair_mix_is_each_rows_own_mix() {
        if !(crate::cpu_has!("avx2") && crate::cpu_has!("fma")) {
            return;
        }
        let mut vis: Vec<usize> = (1..=70).collect();
        vis.extend([127, 128, 129, 511, 512, 513]);
        for hd in [64usize, 128, 80, 72, 16] {
            for &n in &vis {
                let vf: Vec<f32> = prng(0x77 + hd as u64, n * hd);
                let p0: Vec<f32> = prng(91 + hd as u64, n)
                    .iter()
                    .map(|x| x.abs() + 0.01)
                    .collect();
                let p1: Vec<f32> = prng(92 + hd as u64, n)
                    .iter()
                    .map(|x| (x * 3.0).abs() + 0.001)
                    .collect();
                let cases: [(&[f32], &[f32], f32, f32); 5] = [
                    (&p0, &p1, 0.75, 0.25),
                    (&p0, &p1, 1.0, 1.0),
                    (&p0, &p1, 0.0, 0.0),
                    (&p0[..], &p0[..], 1.0, 0.5),
                    (&p0, &p1, -0.0, 3.5),
                ];
                for (a, b, al0, al1) in cases {
                    let o0 = prng(123 + hd as u64, hd);
                    let o1 = prng(124 + hd as u64, hd);
                    let (mut x0, mut x1) = (o0.clone(), o1.clone());
                    let (mut y0, mut y1) = (o0.clone(), o1.clone());
                    // SAFETY: AVX2+FMA checked; `vf` holds `n · hd` floats.
                    unsafe {
                        mix(&mut x0, al0, a, &vf, hd);
                        mix(&mut x1, al1, b, &vf, hd);
                        mix_pair_same(&mut y0, &mut y1, al0, al1, a, b, &vf, hd);
                    }
                    assert!(same_bits(&x0, &y0), "row 0 at hd {hd} vis {n}");
                    assert!(same_bits(&x1, &y1), "row 1 at hd {hd} vis {n}");
                }
            }
        }
        // The long lengths the task names, at the two widths a real model uses.
        for hd in [64usize, 128] {
            for n in [4095usize, 4096] {
                let vf: Vec<f32> = prng(0x55 + hd as u64, n * hd);
                let p0: Vec<f32> = (0..n).map(|i| ((i * 37 % 251) as f32).recip()).collect();
                let p1: Vec<f32> = (0..n).map(|i| ((i * 53 % 199) as f32).recip()).collect();
                let o0 = prng(7, hd);
                let o1 = prng(8, hd);
                let (mut x0, mut x1) = (o0.clone(), o1.clone());
                let (mut y0, mut y1) = (o0.clone(), o1.clone());
                // SAFETY: AVX2+FMA checked.
                unsafe {
                    mix(&mut x0, 0.5, &p0, &vf, hd);
                    mix(&mut x1, 0.5, &p1, &vf, hd);
                    mix_pair_same(&mut y0, &mut y1, 0.5, 0.5, &p0, &p1, &vf, hd);
                }
                assert!(same_bits(&x0, &y0), "long row 0 at hd {hd} vis {n}");
                assert!(same_bits(&x1, &y1), "long row 1 at hd {hd} vis {n}");
            }
        }
    }

    /// The hard rule with a sliding window: a row's bits are its query, its own
    /// context and the window's shape alone — not the other rows in the call, the
    /// prefill chunk width or the pool width.
    ///
    /// The window starts mid-tile and its visible positions are split across the
    /// canonical chunks, so this exercises both sides of the window change at once:
    /// the F16-direct decode path (which now takes windowed layers, skipping every
    /// tile before the window) and the widened prefill-tile path must agree bit for
    /// bit, and a row read alone must match the same row in a chunk of 16 or 3.
    #[test]
    fn a_windowed_rows_bits_are_identical_alone_in_a_chunk_and_at_any_pool_width() {
        let base = Shape {
            n_head: 14,
            n_kv_head: 2,
            head_dim: 64,
            nb: 16,
            start_pos: 400,
            scale: 1.0 / 8.0,
            window: Some(320),
            sinks: None,
            fast: None,
        };
        // The window leaves the longest rows 320 visible positions: five chunks of
        // 64, and a start that is not on the `BLOCK` grid.
        assert_eq!(row_chunks(&base, base.start_pos + base.nb), 5);
        assert_ne!(visible_start(&base, base.start_pos + base.nb) % BLOCK, 0);
        let hd = base.head_dim;
        let qd = base.q_dim();
        let (kv, q, _, _) = setup(KvPrec::F16, &base, 1);
        let pool = [1usize, 2, 4, 8];
        let batched: Vec<Vec<f32>> = pool
            .iter()
            .map(|&w| grouped(&kv, 0, &base, &q, plan(&base, w)))
            .collect();
        for w in 1..batched.len() {
            for b in 0..base.nb {
                let lo = b * base.n_head * hd;
                let hi = (b + 1) * base.n_head * hd;
                assert!(
                    same_bits(&batched[w][lo..hi], &batched[0][lo..hi]),
                    "row {b} moved with the pool width {}",
                    pool[w]
                );
            }
        }
        for (wi, batch) in batched.iter().enumerate() {
            for b in 0..base.nb {
                let solo = Shape {
                    nb: 1,
                    start_pos: base.start_pos + b,
                    ..base
                };
                let got = grouped(&kv, 0, &solo, &q[b * qd..], plan(&solo, pool[wi]));
                let lo = b * base.n_head * hd;
                assert!(
                    same_bits(&got, &batch[lo..(b + 1) * base.n_head * hd]),
                    "windowed row {b} changed in a chunk of 16 at pool width {}",
                    pool[wi]
                );
                if b + 3 <= base.nb {
                    let small = Shape {
                        nb: 3,
                        start_pos: base.start_pos + b,
                        ..base
                    };
                    let got = grouped(&kv, 0, &small, &q[b * qd..], plan(&small, pool[wi]));
                    assert!(
                        same_bits(&got[..base.n_head * hd], &batch[lo..lo + base.n_head * hd]),
                        "windowed row {b} changed in a chunk of 3 at pool width {}",
                        pool[wi]
                    );
                }
            }
        }
    }

    /// A window long enough to keep the prefill tile on the F16-direct path
    /// ([`DIRECT_MIN_VIS`]) must still agree with a solo decode, including when its
    /// start falls inside a tile (the offset the fast kernels take).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn a_windowed_prefill_direct_tile_is_bit_identical_to_a_solo_decode() {
        let base = Shape {
            n_head: 14,
            n_kv_head: 2,
            head_dim: 64,
            nb: 6,
            start_pos: 1300,
            scale: 1.0 / 8.0,
            window: Some(1200),
            sinks: None,
            fast: None,
        };
        let hd = base.head_dim;
        let qd = base.q_dim();
        let (kv, q, _, _) = setup(KvPrec::F16, &base, 1);
        if crate::cpu_has!("avx2") && crate::cpu_has!("fma") && crate::cpu_has!("f16c") {
            assert!(
                prefill_direct(&kv, &base, 1300 + base.nb),
                "the test needs the prefill direct path"
            );
        }
        assert_ne!(
            visible_start(&base, base.start_pos + base.nb) % BLOCK,
            0,
            "the window must start inside a tile"
        );
        let batch = grouped(&kv, 0, &base, &q, plan(&base, 4));
        for b in 0..base.nb {
            let solo = Shape {
                nb: 1,
                start_pos: base.start_pos + b,
                ..base
            };
            let got = grouped(&kv, 0, &solo, &q[b * qd..], plan(&solo, 4));
            let lo = b * base.n_head * hd;
            assert!(
                same_bits(&got, &batch[lo..(b + 1) * base.n_head * hd]),
                "windowed direct row {b} changed between prefill and decode"
            );
        }
    }

    /// The direct kernels are the widened kernels with the widening moved into the
    /// load: same operands (F16C is exact), same per-key dot order, same per-dimension
    /// accumulation order. With whole blocks (no tail, where the widened kernel picks
    /// a different horizontal sum) the results must be bit-identical.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn f16_direct_kernels_match_the_widened_kernels_bit_for_bit() {
        if !(crate::cpu_has!("avx2") && crate::cpu_has!("fma") && crate::cpu_has!("f16c")) {
            return;
        }
        let hd = 64;
        let n = 17;
        let mut c = KvCache::new(KvPrec::F16, 1, 1, hd, n + 2);
        for p in 0..n {
            let k = prng((p * 13 + 3) as u64, hd);
            let v = prng((p * 29 + 5) as u64, hd);
            c.put(0, 0, p, &k, &v);
        }
        c.advance(n);
        let q = prng(77, hd);
        let mut kf = vec![0.0f32; n * hd];
        c.decode_block(true, 0, 0, 0, n, &mut kf);
        // Every visible length, including the tails of the score blocks: the direct
        // path must agree with the widened path at every one, not only on whole
        // blocks, because a prefill tile can end anywhere.
        for vis in 1..=n {
            let (mut wide, mut direct) = (vec![0.0f32; vis], vec![0.0f32; vis]);
            scores(&q, &kf, vis, hd, 0.125, &mut wide);
            // SAFETY: features checked above; `block_bytes` holds `n · hd` halves.
            unsafe {
                scores_f16(
                    &q,
                    c.block_bytes(true, 0, 0, 0, n),
                    vis,
                    hd,
                    0.125,
                    &mut direct,
                )
            };
            assert!(
                same_bits(&wide, &direct),
                "scores at vis {vis}: {wide:?} vs {direct:?}"
            );
        }

        let mut vf = vec![0.0f32; n * hd];
        c.decode_block(false, 0, 0, 0, n, &mut vf);
        let p: Vec<f32> = prng(91, n).iter().map(|x| x.abs() + 0.01).collect();
        let o0 = prng(123, hd);
        for vis in 1..=n {
            let (mut a, mut b) = (o0.clone(), o0.clone());
            mix(&mut a, 0.75, &p[..vis], &vf, hd);
            // SAFETY: features checked above; `block_bytes` holds `n · hd` halves.
            unsafe {
                mix_f16(
                    &mut b,
                    0.75,
                    &p[..vis],
                    c.block_bytes(false, 0, 0, 0, n),
                    hd,
                )
            };
            assert!(same_bits(&a, &b), "mix at vis {vis}");
        }
    }
    // ---- S5.4j: the fast prompt mode's attention order ----

    /// The lengths S5.2b's pair-mix test swept, plus the ones the mode's pieces take.
    ///
    /// Both grids are only read by the three bit-identity tests, which exist only on
    /// the architectures whose SIMD bodies they pin (the twin is portable; the bodies
    /// are not), so they are gated with them.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const FAST_LENS: [usize; 24] = [
        1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 63, 64, 65, 70, 127, 128, 129, 511, 512, 513, 4095,
        4096, 4097,
    ];

    /// The head widths the gate models use and the ones a non-multiple of the lane
    /// width produces.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const FAST_DIMS: [usize; 5] = [64, 128, 80, 72, 16];

    /// The tier's fast score body, one call shape for both architectures' tests.
    ///
    /// # Safety
    /// As [`fast::scores_zmm`] / [`fast::scores_neon`]: the tier is checked by the
    /// caller and `hd` is a multiple of 16.
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn fast_scores_body(
        q: *const f32,
        hd: usize,
        kf: *const f32,
        rows: &[usize; fast::FAST_ROWS],
        nrows: usize,
        len: usize,
        scale: f32,
        s: *mut f32,
        sstride: usize,
        sofs: usize,
    ) {
        // SAFETY: the caller checked the tier.
        unsafe { fast::scores_zmm(q, hd, kf, rows, nrows, len, scale, s, sstride, sofs) }
    }

    /// The tier's fast score body, one call shape for both architectures' tests.
    ///
    /// # Safety
    /// As [`fast::scores_neon`]: the tier is checked by the caller and `hd` is a
    /// multiple of 16.
    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn fast_scores_body(
        q: *const f32,
        hd: usize,
        kf: *const f32,
        rows: &[usize; fast::FAST_ROWS],
        nrows: usize,
        len: usize,
        scale: f32,
        s: *mut f32,
        sstride: usize,
        sofs: usize,
    ) {
        // SAFETY: the caller checked the tier.
        unsafe { fast::scores_neon(q, hd, kf, rows, nrows, len, scale, s, sstride, sofs) }
    }

    /// The tier's fast value-mix body, one call shape for both architectures' tests.
    ///
    /// # Safety
    /// As [`fast::mix_zmm`] / [`fast::mix_neon`]: the tier is checked by the caller
    /// and `hd` is a multiple of 16.
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn fast_mix_body(
        o: *mut f32,
        hd: usize,
        alpha: *const f32,
        rows: &[usize; fast::FAST_ROWS],
        nrows: usize,
        p: *const f32,
        pstride: usize,
        v: *const f32,
        len: usize,
    ) {
        // SAFETY: the caller checked the tier.
        unsafe { fast::mix_zmm(o, hd, alpha, rows, nrows, p, pstride, v, len) }
    }

    /// The tier's fast value-mix body, one call shape for both architectures' tests.
    ///
    /// # Safety
    /// As [`fast::mix_neon`]: the tier is checked by the caller and `hd` is a
    /// multiple of 16.
    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn fast_mix_body(
        o: *mut f32,
        hd: usize,
        alpha: *const f32,
        rows: &[usize; fast::FAST_ROWS],
        nrows: usize,
        p: *const f32,
        pstride: usize,
        v: *const f32,
        len: usize,
    ) {
        // SAFETY: the caller checked the tier.
        unsafe { fast::mix_neon(o, hd, alpha, rows, nrows, p, pstride, v, len) }
    }

    /// The tier's fast soft-max body, one call shape for both architectures' tests.
    ///
    /// # Safety
    /// The tier is checked by the caller; `s` is the row's own piece.
    #[cfg(target_arch = "x86_64")]
    unsafe fn fast_denom_body(s: &mut [f32], m_old: f32) -> (f32, f64, f32) {
        // SAFETY: the caller checked the tier.
        unsafe { fast::denom_zmm(s, m_old) }
    }

    /// The tier's fast soft-max body, one call shape for both architectures' tests.
    ///
    /// # Safety
    /// The tier is checked by the caller; `s` is the row's own piece.
    #[cfg(target_arch = "aarch64")]
    unsafe fn fast_denom_body(s: &mut [f32], m_old: f32) -> (f32, f64, f32) {
        // SAFETY: the caller checked the tier.
        unsafe { fast::denom_neon(s, m_old) }
    }

    /// The fast score body against the fast order's own definition, bit for bit.
    ///
    /// The identity is defined on the architectures whose SIMD bodies carry it (the
    /// twin is portable, the bodies are not): AVX-512 and aarch64's NEON tier, and
    /// `fast::tier()` is the tier table's verdict. Every other tier runs the twin.
    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn the_fast_score_body_is_the_twin() {
        if !fast::tier() {
            eprintln!("skip: this tier keeps the exact attention for prompt rows");
            return;
        }
        let mut failures = 0usize;
        for hd in FAST_DIMS {
            // The body takes whole 16-lane blocks; a width like 72 runs the twin (the
            // dispatch's `body_ok` is the same test).
            if !hd.is_multiple_of(16) {
                continue;
            }
            for &len in &FAST_LENS {
                for (seed, edge) in [(7u64, false), (11, true)] {
                    let q: Vec<f32> = prng(seed + hd as u64, fast::FAST_ROWS * hd);
                    let mut kf = prng(seed + 3 + hd as u64, len * hd);
                    if edge {
                        // The values a soft-max's edge cases produce: wide score spread.
                        for (i, x) in kf.iter_mut().enumerate() {
                            *x = if i % 97 == 0 { 30.0 } else { *x * 0.01 };
                        }
                    }
                    let mut want = vec![0.0f32; fast::FAST_ROWS * len];
                    for r in 0..fast::FAST_ROWS {
                        fast::scores_twin(
                            &q[r * hd..][..hd],
                            &kf,
                            len,
                            hd,
                            0.125,
                            &mut want[r * len..(r + 1) * len],
                        );
                    }
                    let rows: [usize; fast::FAST_ROWS] = std::array::from_fn(|r| r);
                    for nrows in [1usize, 3, fast::FAST_ROWS] {
                        let mut got = vec![0.0f32; fast::FAST_ROWS * len];
                        // SAFETY: the tier's features are checked by `fast::tier()`;
                        // `hd % 16 == 0` for every width but 72's tail, which the twin
                        // takes.
                        unsafe {
                            fast_scores_body(
                                q.as_ptr(),
                                hd,
                                kf.as_ptr(),
                                &rows,
                                nrows,
                                len,
                                0.125,
                                got.as_mut_ptr(),
                                len,
                                0,
                            )
                        };
                        for r in 0..nrows {
                            for j in 0..len {
                                let at = r * len + j;
                                if got[at].to_bits() != want[at].to_bits() {
                                    failures += 1;
                                    if failures < 4 {
                                        eprintln!(
                                            "score hd {hd} len {len} row {r} key {j}: {:e} vs {:e}",
                                            got[at], want[at]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            failures, 0,
            "{failures} fast scores differ from the definition"
        );
    }

    /// The fast value-mix body against the fast order's definition, bit for bit.
    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn the_fast_mix_body_is_the_twin() {
        if !fast::tier() {
            eprintln!("skip: this tier keeps the exact attention for prompt rows");
            return;
        }
        let mut failures = 0usize;
        for hd in FAST_DIMS {
            if !hd.is_multiple_of(16) {
                continue;
            }
            for &len in &FAST_LENS {
                for (seed, alpha) in [(13u64, 1.0f32), (17, 0.37), (19, 0.0)] {
                    let vf = prng(seed + hd as u64, len * hd);
                    let p: Vec<f32> = prng(seed + 5, fast::FAST_ROWS * len)
                        .iter()
                        .map(|x| x.abs())
                        .collect();
                    let o0 = prng(seed + 9, fast::FAST_ROWS * hd);
                    let alphas = [alpha; MAX_PARTIAL_ROWS];
                    let rows: [usize; fast::FAST_ROWS] = std::array::from_fn(|r| r);
                    for nrows in [1usize, 3, fast::FAST_ROWS] {
                        let mut got = o0.clone();
                        let mut want = o0.clone();
                        for r in 0..nrows {
                            fast::mix_twin(
                                &mut want[r * hd..(r + 1) * hd],
                                alphas[r],
                                &p[r * len..(r + 1) * len],
                                &vf,
                                hd,
                            );
                        }
                        // SAFETY: the tier's features are checked; `hd % 16 == 0` for
                        // every width the body takes (72 is the twin's, and the loop's
                        // guard agrees).
                        unsafe {
                            fast_mix_body(
                                got.as_mut_ptr(),
                                hd,
                                alphas.as_ptr(),
                                &rows,
                                nrows,
                                p.as_ptr(),
                                len,
                                vf.as_ptr(),
                                len,
                            )
                        };
                        for r in 0..nrows {
                            for d in 0..hd {
                                let at = r * hd + d;
                                if got[at].to_bits() != want[at].to_bits() {
                                    failures += 1;
                                    if failures < 4 {
                                        eprintln!(
                                            "mix hd {hd} len {len} alpha {alpha} row {r} d {d}: {:e} vs {:e}",
                                            got[at], want[at]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            failures, 0,
            "{failures} fast mixes differ from the definition"
        );
    }

    /// The fast soft-max body against the fast order's definition, bit for bit.
    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn the_fast_denominator_body_is_the_twin() {
        if !fast::tier() {
            eprintln!("skip: this tier keeps the exact attention for prompt rows");
            return;
        }
        let mut failures = 0usize;
        for _hd in [16usize, 64] {
            for &len in &FAST_LENS {
                for (seed, m_old, edge) in [
                    (23u64, f32::NEG_INFINITY, false),
                    (29, f32::NEG_INFINITY, true),
                    (31, 0.75, false),
                    (37, -4.0, true),
                ] {
                    let mut s = prng(seed + len as u64, len);
                    if edge {
                        // All equal (a flat distribution), and one huge logit, so the
                        // exponentials underflow and saturate as they can in a soft-max.
                        for (i, x) in s.iter_mut().enumerate() {
                            *x = match i % 61 {
                                0 => 100.0,
                                1 => -100.0,
                                _ => 0.0,
                            };
                        }
                    }
                    let mut want = s.clone();
                    let (wm, ws, wa) = fast::denom_twin(&mut want, m_old);
                    let mut got = s.clone();
                    // SAFETY: the tier's features are checked; `s` is the row's own
                    // piece.
                    let (gm, gs, ga) = unsafe { fast_denom_body(&mut got, m_old) };
                    if wm.to_bits() != gm.to_bits()
                        || ws.to_bits() != gs.to_bits()
                        || wa.to_bits() != ga.to_bits()
                    {
                        failures += 1;
                        if failures < 4 {
                            eprintln!("denom len {len} seed {seed} m_old {m_old}: {wm} {ws:e} {wa} vs {gm} {gs:e} {ga}");
                        }
                    }
                    for (a, b) in want.iter().zip(&got) {
                        if a.to_bits() != b.to_bits() {
                            failures += 1;
                            if failures < 4 {
                                eprintln!("denom len {len}: exp {a:e} vs {b:e}");
                            }
                            break;
                        }
                    }
                }
            }
        }
        assert_eq!(
            failures, 0,
            "{failures} fast denominators differ from the definition"
        );
    }

    /// The fast order's F16-direct bodies against the widened ones, bit for bit.
    ///
    /// `vcvt_f32_f16` is exact, so reading the cache's halves in the kernel — the
    /// aarch64 fast path (S5.4n) — cannot move a row's bits; it only halves the bytes
    /// the tile streams. The grid is the twin tests' (every piece length, every head
    /// width the bodies cover, one, three and four rows).
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn the_fast_f16_bodies_are_the_widened_ones() {
        if !fast::tier() {
            eprintln!("skip: this tier keeps the exact attention for prompt rows");
            return;
        }
        let mut failures = 0usize;
        for hd in FAST_DIMS {
            if !hd.is_multiple_of(16) {
                continue;
            }
            for &len in &FAST_LENS {
                let vf: Vec<f32> = prng(0x41 + hd as u64, len * hd)
                    .iter()
                    .map(|x| x * 0.5)
                    .collect();
                let halves: Vec<u8> = vf
                    .iter()
                    .flat_map(|x| peregrine_kernels::quant::f32_to_f16(*x).to_le_bytes())
                    .collect();
                let widened: Vec<f32> = halves
                    .chunks_exact(2)
                    .map(|b| peregrine_kernels::format::f16(u16::from_le_bytes([b[0], b[1]])))
                    .collect();
                let q: Vec<f32> = prng(0x43 + hd as u64, fast::FAST_ROWS * hd);
                let p: Vec<f32> = prng(0x47 + len as u64, fast::FAST_ROWS * len)
                    .iter()
                    .map(|x| x.abs())
                    .collect();
                let o0 = prng(0x49, fast::FAST_ROWS * hd);
                let alphas = [0.75f32; MAX_PARTIAL_ROWS];
                let rows: [usize; fast::FAST_ROWS] = std::array::from_fn(|r| r);
                for nrows in [1usize, 3, fast::FAST_ROWS] {
                    let mut want_s = vec![0.0f32; fast::FAST_ROWS * len];
                    let mut got_s = vec![0.0f32; fast::FAST_ROWS * len];
                    let mut want_m = o0.clone();
                    let mut got_m = o0.clone();
                    // SAFETY: NEON is this target's baseline and `fast::tier()` checked
                    // the tier; every buffer is `FAST_ROWS · hd`/`· len` long.
                    unsafe {
                        fast::scores_neon(
                            q.as_ptr(),
                            hd,
                            widened.as_ptr(),
                            &rows,
                            nrows,
                            len,
                            0.125,
                            want_s.as_mut_ptr(),
                            len,
                            0,
                        );
                        fast::scores_f16_neon(
                            q.as_ptr(),
                            hd,
                            halves.as_ptr(),
                            &rows,
                            nrows,
                            len,
                            0.125,
                            got_s.as_mut_ptr(),
                            len,
                            0,
                        );
                        fast::mix_neon(
                            want_m.as_mut_ptr(),
                            hd,
                            alphas.as_ptr(),
                            &rows,
                            nrows,
                            p.as_ptr(),
                            len,
                            widened.as_ptr(),
                            len,
                        );
                        fast::mix_f16_neon(
                            got_m.as_mut_ptr(),
                            hd,
                            alphas.as_ptr(),
                            &rows,
                            nrows,
                            p.as_ptr(),
                            len,
                            halves.as_ptr(),
                            len,
                        );
                    }
                    for at in 0..fast::FAST_ROWS * len {
                        if want_s[at].to_bits() != got_s[at].to_bits() {
                            failures += 1;
                        }
                    }
                    for at in 0..fast::FAST_ROWS * hd {
                        if want_m[at].to_bits() != got_m[at].to_bits() {
                            failures += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(
            failures, 0,
            "{failures} F16-direct fast values differ from the widened bodies"
        );
    }

    /// A test cache and query for the end-to-end fast-attention tests.
    fn fast_attend_case(cfg: &crate::model::Config, n: usize) -> (KvCache, Vec<f32>) {
        let hd = cfg.head_dim;
        let mut kv = KvCache::new(KvPrec::F16, 1, cfg.n_kv_head, hd, n + cfg.n_head);
        for p in 0..n {
            for h in 0..cfg.n_kv_head {
                let k = prng((p * 31 + h * 7 + 1) as u64, hd);
                let v = prng((p * 17 + h * 5 + 2) as u64, hd);
                kv.put(0, h, p, &k, &v);
            }
        }
        kv.advance(n);
        let q = prng(0x5a5a, cfg.n_head * cfg.n_head * hd);
        (kv, q)
    }

    /// One prompt of `nb` rows over `n` cached positions, every row's flag `fast`.
    fn fast_attend(
        cfg: &crate::model::Config,
        kv: &KvCache,
        q: &[f32],
        n: usize,
        nb: usize,
        fast: bool,
        threads: usize,
    ) -> Vec<f32> {
        use crate::blocks::attn::{attend_marked, AttnScratch};
        use crate::pool::Pool;
        let pool = Pool::new(threads);
        let mut scratch = AttnScratch::new(cfg, nb.max(1), threads);
        let mut stats = crate::state::StepStats::default();
        let mut xb = vec![0.0f32; nb * cfg.q_dim()];
        let flags = vec![fast; nb];
        attend_marked(
            cfg,
            &mut scratch,
            q,
            &mut xb,
            kv,
            &pool,
            0,
            nb,
            n,
            0,
            Some(&flags),
            &mut stats,
        );
        xb
    }

    /// The fast order is what a marked prompt row gets, and it is not the exact order.
    #[test]
    fn a_marked_prompt_row_takes_the_fast_attention_order() {
        use crate::model::Config;
        let cfg = Config {
            arch_name: "test",
            rope_neox: false,
            n_layer: 1,
            n_kv_layer: 1,
            n_head: 8,
            n_kv_head: 2,
            head_dim: 64,
            d_model: 8 * 64,
            n_ff: 4 * 8 * 64,
            n_vocab: 32,
            rope_dim: 64,
            eps: 1e-5,
            rope_base: 10_000.0,
            ctx_train: 1024,
            rope_yarn: None,
        };
        let n = 96usize;
        let (kv, q) = fast_attend_case(&cfg, n + 8);
        let exact = fast_attend(&cfg, &kv, &q, n, 8, false, 2);
        let fast = fast_attend(&cfg, &kv, &q, n, 8, true, 2);
        if fast::tier() {
            assert_ne!(
                exact.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                fast.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "a marked row got the exact order: the fast path is not being taken"
            );
            // A different order of the same sums: rounding only.
            for (a, b) in exact.iter().zip(&fast) {
                assert!((a - b).abs() <= 1e-4 * a.abs().max(1.0), "{a} vs {b}");
            }
        } else {
            assert_eq!(
                exact.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                fast.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "a tier without the fast order moved a row's bits"
            );
        }
        // A second pair, so the check is not one call's luck.
        if fast::tier() {
            let mixed = [
                fast_attend(&cfg, &kv, &q, n, 8, false, 2),
                fast_attend(&cfg, &kv, &q, n, 8, true, 2),
            ];
            assert_ne!(
                mixed[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                mixed[1].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "the two modes agree, so the flags did nothing"
            );
        }
    }

    /// A fast row's bits do not depend on how many rows share its pass: the same row at
    /// the same position is its own result at the head of a 1-, 2-, 4- and 8-row call,
    /// at 1 and 4 threads. This is the group-count half of §5k's invariance rule (the
    /// chunk and thread half is `fast_mode_is_prompt_invariant` in
    /// `crates/peregrine-models/tests/fast_prompt.rs`).
    #[test]
    fn a_fast_rows_attention_is_its_own_at_every_group_size() {
        use crate::model::Config;
        let cfg = Config {
            arch_name: "test",
            rope_neox: false,
            n_layer: 1,
            n_kv_layer: 1,
            n_head: 8,
            n_kv_head: 2,
            head_dim: 64,
            d_model: 8 * 64,
            n_ff: 4 * 8 * 64,
            n_vocab: 32,
            rope_dim: 64,
            eps: 1e-5,
            rope_base: 10_000.0,
            ctx_train: 1024,
            rope_yarn: None,
        };
        let n = 200usize;
        let (kv, q) = fast_attend_case(&cfg, n + 8);
        let qd = cfg.q_dim();
        let first = fast_attend(&cfg, &kv, &q, n, 1, true, 1);
        for nb in [2usize, 4, 8] {
            for threads in [1usize, 4] {
                let got = fast_attend(&cfg, &kv, &q, n, nb, true, threads);
                assert!(
                    first[..qd]
                        .iter()
                        .zip(&got[..qd])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "the first row moved at {nb} rows, {threads} threads"
                );
            }
        }
    }
}

