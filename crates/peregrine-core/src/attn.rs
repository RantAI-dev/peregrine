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
//! * **Prefill widens the block once** and every query row of the task reads that one
//!   decoded f32 tile. With tens of rows sharing a block, one widening amortises over
//!   them; re-converting per row pair would do more `vcvtph2ps` work than it saves.
//!   Both paths use the same per-key reduction tree, so which one a row takes cannot
//!   change its bits.
//! * **A row's partition is its own.** The position chunks a row is cut into are a
//!   function of that row's context length and the model shape alone — never of the
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
/// A row's arithmetic must depend only on that row's own context length and the model
/// shape — never on the other rows in the call, the call width (`nb`), the prefill
/// chunk or the pool. The position partition therefore is not a property of the call
/// at all: `row_chunks` gives every row its own chunk count from its own context,
/// and the cut is the even one. A single-row call splits those chunks across tasks
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
    let chunks = if sh.nb > 1 || sh.nb * sh.gqa() > MAX_PARTIAL_ROWS {
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
/// The boundaries are the even cut `i·n_ctx/k`, which is what the single-row decode
/// path uses too; they need not land on the `BLOCK` grid (see [`run_rows`]). The work
/// gate is per row, so a short row is never split just because a wide call is long.
fn row_chunks(sh: &Shape<'_>, n_ctx: usize) -> usize {
    let work = sh.n_head * n_ctx * sh.head_dim * 2;
    if work < 1 << 19 {
        return 1;
    }
    let by_ctx = (n_ctx / BLOCK).max(1);
    let by_heads = TARGET_TASKS.div_ceil(sh.n_kv_head);
    by_heads.min(by_ctx).clamp(1, MAX_CHUNKS)
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
    let (p0, p1) = (pi * ctx / p.pos_chunks, (pi + 1) * ctx / p.pos_chunks);
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
    /// One tile's raw scores per query row for the F16-direct path: `row · BLOCK`
    /// floats, so every row can be scored against the shared K block before its own
    /// online-softmax pass consumes them.
    #[cfg(target_arch = "x86_64")]
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
            #[cfg(target_arch = "x86_64")]
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
    #[cfg(target_arch = "x86_64")]
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
    // Tiles stay on the global `BLOCK` grid: a chunk may start inside one, and the
    // multi-row walker ([`run_rows`]) then scores the same tile for every row. Only
    // the part from `t.p0` on is read, so a chunk is never decoded twice.
    let mut blk = t.p0;
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
                let alpha = (m_old - m_new).exp();
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
#[cfg(target_arch = "x86_64")]
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

    let p_end = t.p1.min(sh.start_pos + t.b1);
    let mut blk = t.p0;
    while blk < p_end {
        let bn = ((blk / BLOCK + 1) * BLOCK - blk).min(p_end - blk);
        let kb = kv.block_bytes(true, layer, t.kh, blk, bn);
        let vb = kv.block_bytes(false, layer, t.kh, blk, bn);
        let vis = bn.min(sh.start_pos + t.b0 + 1 - blk);
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
                alpha[r] = (m_old - m_new).exp();
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
        run_range(kv, layer, sh, q, t, r0, r1, sc, out);
        r0 = r1;
    }
}

/// One sub-block `r0..r1` of [`run_rows`]' query rows.
#[allow(clippy::too_many_arguments)]
fn run_range<O: RowSink>(
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
    for lr in 0..sub {
        let b = t.b0 + (r0 + lr) / gqa;
        let n_ctx = sh.start_pos + b + 1;
        ctx_of[lr] = n_ctx;
        count[lr] = row_chunks(sh, n_ctx);
        max_ctx = max_ctx.max(n_ctx);
    }

    let p_end = t.p1.min(max_ctx);
    let mut blk = t.p0;
    while blk < p_end {
        let bn = BLOCK.min(p_end - blk);
        kv.decode_block(true, layer, kh, blk, bn, &mut sc.kf);
        kv.decode_block(false, layer, kh, blk, bn, &mut sc.vf);
        for lr in 0..sub {
            let n_ctx = ctx_of[lr];
            if n_ctx <= blk {
                continue;
            }
            let p_min = match sh.window {
                Some(w) => n_ctx.saturating_sub(w),
                None => 0,
            };
            let vis_end = (blk + bn).min(n_ctx);
            // Within the tile, `pos` walks this row's chunks: usually once, twice
            // when a chunk boundary falls inside the tile.
            let mut pos = blk;
            while pos < vis_end {
                let ci = cur[lr];
                // The row's chunks are `[i·n_ctx/k, (i+1)·n_ctx/k)`.
                let chunk_end = (ci + 1) * n_ctx / count[lr];
                let end = vis_end.min(chunk_end);
                let start = pos.max(p_min);
                if start < end {
                    let j_start = start - blk;
                    let len = end - start;
                    let s = &mut sc.s[..len];
                    scores(
                        &sc.q[lr * hd..][..hd],
                        &sc.kf[j_start * hd..],
                        len,
                        hd,
                        sh.scale,
                        s,
                    );
                    let m_old = sc.m[lr];
                    let m_new = m_old.max(crate::vmath::max(s));
                    let psum = crate::vmath::exp_shift_sum(s, m_new);
                    let alpha = (m_old - m_new).exp();
                    sc.l[lr] = sc.l[lr] * alpha as f64 + psum;
                    mix(
                        &mut sc.o[lr * hd..][..hd],
                        alpha,
                        s,
                        &sc.vf[j_start * hd..],
                        hd,
                    );
                    sc.m[lr] = m_new;
                }
                pos = end;
                if end == chunk_end && ci + 1 < count[lr] {
                    // The chunk is complete: bank it and start the next one fresh,
                    // exactly as a task of its own would have.
                    save_part(sc, lr, ci, hd);
                    sc.o[lr * hd..(lr + 1) * hd].fill(0.0);
                    sc.m[lr] = f32::NEG_INFINITY;
                    sc.l[lr] = 0.0;
                    cur[lr] = ci + 1;
                }
            }
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
        let w = (m - m_star).exp();
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
/// F16C widens the stored halves in registers; AVX2+FMA do the arithmetic. The
/// sliding-window path keeps the widened tile because its visible range starts
/// mid-tile, and it is not the case this path exists for.
#[cfg(target_arch = "x86_64")]
fn f16_direct(kv: &KvCache, sh: &Shape<'_>) -> bool {
    kv.prec() == crate::kv::KvPrec::F16
        && sh.window.is_none()
        && sh.head_dim.is_multiple_of(8)
        && crate::cpu_has!("avx2")
        && crate::cpu_has!("fma")
        && crate::cpu_has!("f16c")
}

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
unsafe fn mix_f16(o: &mut [f32], alpha: f32, p: &[f32], v: &[u8], hd: usize) {
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

fn scores(q: &[f32], kf: &[f32], vis: usize, hd: usize, scale: f32, s: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if hd.is_multiple_of(8) && crate::cpu_has!("avx2") && crate::cpu_has!("fma") {
            // SAFETY: features checked; `kf` holds at least `vis · hd` floats.
            return unsafe { scores_avx2(q, kf, vis, hd, scale, s) };
        }
    }
    for (j, sj) in s.iter_mut().enumerate().take(vis) {
        let k = &kf[j * hd..(j + 1) * hd];
        *sj = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
    }
}

/// `o = α·o + Σ_j p_j · v_j` over the first `p.len()` values of the decoded tile.
fn mix(o: &mut [f32], alpha: f32, p: &[f32], vf: &[f32], hd: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if hd.is_multiple_of(8) && crate::cpu_has!("avx2") && crate::cpu_has!("fma") {
            // SAFETY: features checked; `vf` holds at least `p.len() · hd` floats.
            return unsafe { mix_avx2(o, alpha, p, vf, hd) };
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
    // Sixty-four dimensions at a time: eight accumulators stay in registers for the
    // whole block of values.
    let mut d0 = 0;
    while d0 < hd {
        let lanes = (hd - d0).min(64) / 8;
        let mut acc = [_mm256_setzero_ps(); 8];
        for (l, a) in acc.iter_mut().enumerate().take(lanes) {
            *a = _mm256_mul_ps(_mm256_loadu_ps(o.as_ptr().add(d0 + l * 8)), al);
        }
        for (j, &w) in p.iter().enumerate() {
            let wv = _mm256_set1_ps(w);
            let row = vp.add(j * hd + d0);
            for (l, a) in acc.iter_mut().enumerate().take(lanes) {
                *a = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row.add(l * 8)), *a);
            }
        }
        for (l, a) in acc.iter().enumerate().take(lanes) {
            _mm256_storeu_ps(o.as_mut_ptr().add(d0 + l * 8), *a);
        }
        d0 += lanes * 8;
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
}
