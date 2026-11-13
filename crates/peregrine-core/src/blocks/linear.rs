//! Fused linear transformation blocks (dense GEMM groups and LittleBit).

use crate::blocks::par::{Disjoint, OutPtr, SharedBuf};
use crate::gguf::{Format, TensorInfo};
use crate::lbq::Slb;
use crate::model::{Linear, StepStats, Weights};
use crate::pool::{share, Pool};
use crate::quant::{row_bytes, QuantAct};
use crate::table::{MatKernels, MatPath};
use std::time::Instant;

/// The rows one repacked k-quant tile holds.
const KQ_TILE_ROWS: usize = 8;

/// The output panels one unit of packed-float work covers (S4.9f). Two are what
/// the AVX-512 tile multiplies at once; on AVX2 the kernel runs them in turn.
const FP_UNIT_PANELS: usize = 2;

/// Largest batch one matmul call accepts. A stack array of this many scratch floats
/// holds one row's results; `PEREGRINE_PREFILL_CHUNK` and `draft_len + 1` are both far below it.
///
/// S5.4b raises it from 128 to 512 so the fast prompt mode can measure the widest
/// useful prompt chunk (64 to 512): a wider pass is what reads the weights once for
/// more rows, and the driver split it into 128-row blocks before. A row's bits do not
/// depend on the batch (ADR 0003), so this moves time, never a number: with the option
/// off the default chunk is 64 and no pass changes.
pub const MAX_BATCH: usize = 512;

/// The rows one matmul call takes by default. Every batch the engine built before
/// S5.4b was split at this width; the fast prompt mode is what asks for more, and
/// widening a pass is the whole point of its wide chunk, so only a marked pass goes
/// past it. With the option off the split is exactly today's.
pub const DEFAULT_BATCH: usize = 128;

/// whether matmul rows are handed out in chunks from a counter (default) or in equal shares
/// (`PEREGRINE_DYNAMIC=0`, the static partition).
fn dynamic_rows() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| crate::options::get().dynamic_rows)
}

/// The transposed output of one matrix of a group: `rows` floats per activation.
///
/// A repacked tile's block is activation-major — `blk[b · nrows + i]`, row `i` of
/// the tile for activation `b` — and the output is transposed, `out[(b0 + b) ·
/// rows + row]`. The old write-out walked the rows first, which is one
/// single-float store per (row, activation), each `rows` floats from the next
/// (32 KiB on the 1B at chunk 128): every store its own cache line, so a wider
/// pass paid for it. Turned around the same data is contiguous on both sides —
/// one activation's tile is `nrows` consecutive floats in the block *and* in the
/// output — so each activation is one slice copy and the tile block stays in L1.
///
/// The packed-float and per-row paths keep the scatter: their blocks are
/// row-major (`blk[i · nb + b]`), so their activation-major output would need a
/// transpose, which is a job for a kernel that writes the treated layout.
struct Untiler<'a, 'b> {
    out: &'b OutPtr<'a>,
    /// The first activation of the group's output block.
    b0: usize,
    /// The floats between two activations of the output.
    rows: usize,
    /// The activations in the block.
    nb: usize,
}

impl Untiler<'_, '_> {
    /// Copy rows `lo..hi` of `base_row`'s tile from `blk` into every activation.
    ///
    /// # Safety
    ///
    /// Every row in `lo..hi` of the range must belong to this worker, as it did
    /// for the scatter this replaces.
    #[inline]
    unsafe fn untile(&self, base_row: usize, blk: &[f32], nrows: usize, lo: usize, hi: usize) {
        for b in 0..self.nb {
            let src = &blk[b * nrows + lo..b * nrows + hi];
            let dst = self
                .out
                .chunk((self.b0 + b) * self.rows + base_row + lo, hi - lo);
            dst.copy_from_slice(src);
        }
    }
}

/// One matrix of a fused matmul group, and where its output goes.
#[derive(Clone, Copy)]
pub struct MatJob<'a> {
    pub w: &'a TensorInfo,
    pub out: OutPtr<'a>,
    /// Output rows to compute, when fewer than the matrix's full `dims[1]` (the MTP draft head reads only the
    /// first part of the vocabulary). Kept here rather than in a cloned `TensorInfo`, whose `String` and `dims`
    /// would allocate on every draft step.
    pub rows: Option<usize>,
}

impl<'a> MatJob<'a> {
    /// The number of output rows this job computes.
    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.rows.unwrap_or(self.w.dims[1] as usize)
    }
}

/// The bytes one activation row holds: its codes, its scales and its run sums.
///
/// The activation traffic is counted in bytes the same way the weight traffic
/// is: what the matrix product reads of the activation, once per pass over it.
fn act_bytes(a: &QuantAct) -> usize {
    a.q.len()
        + a.qk.len()
        + a.tq.len()
        + 4 * (a.d.len() + a.dk.len() + a.td.len() + a.xf.len())
        + 2 * (a.sums.len() + a.bsums.len() + a.tls.len())
}

/// The format of a matrix's packed float panels: the file's format, or F32 for a
/// Q8_0 matrix dequantised into f32 panels at load (S4.9m).
#[inline]
fn panel_fmt(t: &TensorInfo) -> Format {
    t.fp_kind.unwrap_or(t.kind)
}

/// One matrix of a fused group, dense or LittleBit. `linear_group` dispatches.
#[derive(Clone, Copy)]
pub struct LinJob<'a> {
    pub lin: &'a Linear,
    pub out: OutPtr<'a>,
}

/// Several matrices that share one activation — Q, K and V; gate and up — in a
/// single fork–join. Three barriers become one, and the workers run through the
/// concatenated row space, so the split stays balanced even though K and V have a
/// third of Q's rows. The byte, MAC and barrier accounting is unchanged except
/// that the barrier count honestly falls.
pub fn matmul_group(
    w: &Weights,
    jobs: &[MatJob],
    acts: &[QuantAct],
    nb: usize,
    pool: &Pool,
    st: &mut StepStats,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    assert!(
        !acts.is_empty() && nb <= acts.len(),
        "batch of {nb} exceeds the activations it was given"
    );
    // A pass may be wider than one kernel call (S4.9b: a non-decoding pass carries up to
    // `PEREGRINE_PASS_ROWS` rows). The kernel's row limit is a property of its stack scratch,
    // not of the result: every output element is a function of one activation row and one
    // weight row, so the batch is split into blocks of `MAX_BATCH` rows and the results are
    // the same bits. The weight traffic is counted once per matrix, not once per block.
    let t0 = Instant::now();
    // The weight traffic is counted once per matrix, however many blocks the batch is split
    // into: the bytes crossed the memory interface once per block, but a pass's cost is reported
    // per matrix with the format's row size, as before.
    for j in jobs {
        let n = j.w.dims[0] as usize;
        st.bytes_weights += (j.rows() * row_bytes(j.w.kind, n)) as u64;
    }
    // A pass can carry prompt rows and generated rows: the fast prompt mode computes
    // the prompt rows from the resident 8-bit form and the generated rows with the
    // exact kernels (S5.4b). A row's kernel depends on its own kind and on nothing
    // else, so the batch is split into runs of one kind and each run is multiplied
    // as it would be alone. With no mixed pass — the default engine, a decode step,
    // a prompt with no decoder beside it — the split makes one call, as before.
    let has_form = jobs.iter().any(|j| j.w.q8_off.is_some());
    let split =
        has_form && acts[..nb].iter().any(|a| a.prompt) && acts[..nb].iter().any(|a| !a.prompt);
    let mut b0 = 0usize;
    while b0 < nb {
        let run_end = if split {
            let kind = acts[b0].prompt;
            let mut e = b0 + 1;
            while e < nb && acts[e].prompt == kind {
                e += 1;
            }
            e
        } else {
            nb
        };
        // A fast prompt pass reads the weights once for up to `MAX_BATCH` rows; every
        // other pass keeps the 128-row split it had before S5.4b. A group with no
        // resident form — every format outside the tier's table — keeps that split
        // too, so the mode runs the exact engine's own kernel calls (S5.4f).
        let cap = if acts[b0].prompt && has_form {
            MAX_BATCH
        } else {
            DEFAULT_BATCH
        };
        let mut p = b0;
        while p < run_end {
            let n = cap.min(run_end - p);
            matmul_block(w, jobs, &acts[p..p + n], n, p, pool, st, kq);
            p += n;
        }
        b0 = run_end;
    }
    st.ns_matmul += t0.elapsed().as_nanos() as u64;
}

/// The form's interleaved activation, filled on the pool (S5.4o).
///
/// [`KqActs::fill_with`] is the caller's own work: every output row of the group reads
/// the whole interleaved buffer, and each group of four activation rows is an
/// independent transpose, so a worker can write the groups it owns and no byte of the
/// buffer depends on who wrote it. Measured at 512 tokens and twelve threads, the
/// serial fill was **6 %** of a fast pass (78 ms of 1.23 s, 112 MiB of interleaving —
/// S5.4j's estimate of 24 MiB was low by the size of the `down` group's 32
/// super-blocks), which is worth the one join it costs; below ~256 KiB of interleaved
/// codes the join costs more than the copy, and the serial fill is kept.
fn fill_form(pool: &Pool, kq: &mut peregrine_kernels::kq::KqActs, acts: &[QuantAct], sum256: bool) {
    use peregrine_kernels::kq::{fill_group, ACT_GROUP};
    kq.size_for(acts, sum256, false);
    let groups = acts.len().div_ceil(ACT_GROUP);
    let n_sb = kq.n_sb;
    if pool.width() < 2 || groups < 2 || groups * n_sb < 256 {
        for g in 0..groups {
            fill_group(
                &mut kq.q[g * n_sb * 1024..(g + 1) * n_sb * 1024],
                &mut kq.d[g * n_sb * ACT_GROUP..(g + 1) * n_sb * ACT_GROUP],
                &mut kq.bs[g * n_sb * ACT_GROUP * 8..(g + 1) * n_sb * ACT_GROUP * 8],
                &mut kq.bs16[g * n_sb * ACT_GROUP * 16..(g + 1) * n_sb * ACT_GROUP * 16],
                &mut kq.bsp[g * n_sb * ACT_GROUP * 4..(g + 1) * n_sb * ACT_GROUP * 4],
                sum256.then(|| &mut kq.sum256[g * n_sb * ACT_GROUP..(g + 1) * n_sb * ACT_GROUP]),
                acts,
                g,
                n_sb,
            );
        }
        return;
    }
    let q = Disjoint::new(kq.q.as_mut_slice());
    let d = Disjoint::new(kq.d.as_mut_slice());
    let bs = Disjoint::new(kq.bs.as_mut_slice());
    let bs16 = Disjoint::new(kq.bs16.as_mut_slice());
    let bsp = Disjoint::new(kq.bsp.as_mut_slice());
    let sums = Disjoint::new(kq.sum256.as_mut_slice());
    pool.run(|wk, nw| {
        let (g0, g1) = share(groups, wk, nw);
        for g in g0..g1 {
            // SAFETY: the groups of one fill are disjoint in every buffer, and worker
            // `wk` writes only the groups `[g0, g1)` it was given. `sums` is only sized
            // (and only read) when the 8-bit form asked for the whole-block sums.
            let (q, d, bs, bs16, bsp) = unsafe {
                (
                    q.chunk(g * n_sb * 1024, n_sb * 1024),
                    d.chunk(g * n_sb * ACT_GROUP, n_sb * ACT_GROUP),
                    bs.chunk(g * n_sb * ACT_GROUP * 8, n_sb * ACT_GROUP * 8),
                    bs16.chunk(g * n_sb * ACT_GROUP * 16, n_sb * ACT_GROUP * 16),
                    bsp.chunk(g * n_sb * ACT_GROUP * 4, n_sb * ACT_GROUP * 4),
                )
            };
            let sums =
                sum256.then(|| unsafe { sums.chunk(g * n_sb * ACT_GROUP, n_sb * ACT_GROUP) });
            fill_group(q, d, bs, bs16, bsp, sums, acts, g, n_sb);
        }
    });
}

/// [`matmul_group`] for one block of at most `MAX_BATCH` activation rows (a fast prompt
/// pass's width; the default is [`DEFAULT_BATCH`]), starting at batch row
/// `b0` of the caller's batch. `b0` only shifts where the results are written: a row's value is
/// a function of the activation row and the weight row alone.
#[allow(clippy::too_many_arguments)]
fn matmul_block(
    w: &Weights,
    jobs: &[MatJob],
    acts: &[QuantAct],
    nb: usize,
    b0: usize,
    pool: &Pool,
    st: &mut StepStats,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    assert!(
        nb <= MAX_BATCH,
        "batch of {nb} exceeds the matmul scratch of {MAX_BATCH}"
    );
    let total: usize = jobs.iter().map(|j| j.rows()).sum();
    let acts = &acts[..nb];
    // Every activation row of this block carries the same kind (matmul_group splits
    // a mixed pass into runs): true is a prompt row the fast mode may re-round.
    let fast_row = acts[0].prompt;
    // Each matrix's kernel is chosen once, from the kernel table: by format, storage (tiles or rows),
    // activation layout and CPU. Tiled matrices are worked in whole 16-row tiles, so every range below
    // starts and ends on one.
    let mk = MatKernels::get();
    let act_ptq1 = acts[0].ptq1;
    // At most 4 matrices per group (q/k/v, gate/up): a fixed array, no allocation per call.
    assert!(jobs.len() <= 4, "a matmul group has at most 4 matrices");
    let mut paths = [MatPath::PerRow; 4];
    for (p, j) in paths.iter_mut().zip(jobs) {
        *p = mk.path(
            panel_fmt(j.w),
            j.w.tile_off.is_some(),
            j.w.kq_off.is_some(),
            j.w.fp_off.is_some(),
            act_ptq1,
        );
    }
    let paths = &paths[..jobs.len()];
    let tiled = paths.contains(&MatPath::Tiles);
    // The interleaved activation is built once for the group, from the
    // activation the group shares: every output row of the group reads it, so
    // building it per matrix would multiply the work by the group's size. Only
    // the k-quant tiles need it — the Q5_0 and Q8_0 exact tiles read the per-32
    // layout the activation already has — and its fill requires a whole number of
    // 256-weight super-blocks, which a Q5_0 exact matrix's 896-wide rows are not.
    let kq_fill = jobs.iter().zip(paths).any(|(j, &p)| {
        p == MatPath::KqTiles && matches!(j.w.kind, Format::Q4_K | Format::Q5_K | Format::Q6_K)
    });
    // The 8-bit form reads the interleaved activation at every width, even one row
    // (S5.4b); the exact k-quant kernels only need it from four rows up. A form
    // matrix whose row stops inside its last 256-weight super-block (Q5_0's
    // 896-wide rows) needs a padded buffer of its own, so the exact kernels keep
    // the unpadded one they had (S5.4f).
    let q8_plain = acts[0].prompt
        && jobs
            .iter()
            .any(|j| j.w.q8_off.is_some() && (j.w.dims[0] as usize).is_multiple_of(256));
    let q8_pad = acts[0].prompt
        && jobs
            .iter()
            .any(|j| j.w.q8_off.is_some() && !(j.w.dims[0] as usize).is_multiple_of(256));
    if (kq_fill && nb >= 4) || q8_plain {
        fill_form(pool, kq, &acts[..nb], q8_plain);
    }
    let mut kq_pad = peregrine_kernels::kq::KqActs::default();
    if q8_pad {
        kq_pad.fill_padded(&acts[..nb], true);
    }
    // The rows `lo..hi` of the concatenated row space.
    let rows_range = |lo: usize, hi: usize, tmp: &mut [f32; MAX_BATCH]| {
        let mut off = 0usize;
        for (j, &path) in jobs.iter().zip(paths) {
            let n = j.w.dims[0] as usize;
            let rows = j.rows();
            let (a, z) = (lo.max(off), hi.min(off + rows));
            if a >= z {
                off += rows;
                continue;
            }
            let rb = row_bytes(j.w.kind, n);
            let (mut r, rz) = (a - off, z - off);
            let tf = crate::prof::detail().then(Instant::now);
            match path {
                MatPath::Tiles => {
                    let toff = j.w.tile_off.expect("a tiled path has a tile offset");
                    let ng = n / 128;
                    let tb = 16 * rb;
                    const TPC: usize = 2; // tiles per call: 32 rows × MAX_BATCH outputs fit `blk`
                    let mut blk = [0.0f32; 32 * MAX_BATCH];
                    debug_assert!(r % 16 == 0 && rz % 16 == 0);
                    while r < rz {
                        let nt = TPC.min((rz - r) / 16);
                        let w_tiles = &w.tiles()[toff + (r / 16) * tb..toff + (r / 16 + nt) * tb];
                        mk.tiles_pq2(w_tiles, ng, nt, acts, &mut blk[..16 * nt * nb]);
                        for i in 0..16 * nt {
                            for b in 0..nb {
                                // SAFETY: row `r + i` of this job belongs to this worker (tile-aligned ranges).
                                unsafe { j.out.set((b0 + b) * rows + r + i, blk[i * nb + b]) };
                            }
                        }
                        r += 16 * nt;
                    }
                }
                MatPath::RowsPq2 | MatPath::RowsPtq1 => {
                    // A block of rows per kernel call (the dispatch is paid per block, not per row).
                    let bytes = w.bytes(j.w);
                    const BLK: usize = 32;
                    let mut blk = [0.0f32; BLK * MAX_BATCH];
                    while r < rz {
                        let k = BLK.min(rz - r).min(MAX_BATCH * BLK / nb.max(1));
                        let (wr, o) = (&bytes[r * rb..(r + k) * rb], &mut blk[..k * nb]);
                        if path == MatPath::RowsPtq1 {
                            mk.rows_ptq1(wr, rb, k, acts, o)
                        } else {
                            mk.rows_pq2(wr, rb, k, acts, o)
                        }
                        for i in 0..k {
                            for b in 0..nb {
                                // SAFETY: row `r + i` of this job belongs to this worker.
                                unsafe { j.out.set((b0 + b) * rows + r + i, blk[i * nb + b]) };
                            }
                        }
                        r += k;
                    }
                }
                MatPath::KqTiles => {
                    // The repacked eight-row tiles (S3.4). The store holds the
                    // rows in whole groups of eight; the rows past the last
                    // whole group keep today's kernel, so a row's result
                    // depends only on its index.
                    let kq_off = j.w.kq_off.expect("the repacked path has a store offset");
                    let n_sb = n / j.w.kind.block();
                    let tile_bytes = match j.w.kind {
                        Format::Q4_K => peregrine_kernels::kq::Q4K8::SIZE,
                        Format::Q5_K => peregrine_kernels::kq::Q5K8::SIZE,
                        Format::Q5_0 => peregrine_kernels::kq::Q5_08::SIZE,
                        Format::Q8_0 => peregrine_kernels::kq::Q8_08::SIZE,
                        Format::Q6_K => peregrine_kernels::kq::Q6K8::SIZE,
                        _ => unreachable!("a repacked path has a repacked format"),
                    };
                    let store = w.kq();
                    let rows = j.rows();
                    // The fast prompt mode's resident form, when this matrix has one
                    // and the pass's rows are prompt rows (S5.4b; Q5_0 joins where the
                    // tier's table selects it, S5.4f).
                    let q8_off = if fast_row { j.w.q8_off } else { None };
                    let repacked = (rows / KQ_TILE_ROWS) * KQ_TILE_ROWS;
                    // The tiles that cover the rows of this range. A tile is
                    // computed whole, so a range that starts inside one costs
                    // the whole tile and keeps the rows it does not want.
                    let lo = r.min(repacked);
                    let hi = rz.min(repacked);
                    if lo < hi {
                        let mut blk = [0.0f32; KQ_TILE_ROWS * MAX_BATCH];
                        let first = lo / KQ_TILE_ROWS;
                        let last = hi.div_ceil(KQ_TILE_ROWS);
                        for t in first..last {
                            let at = kq_off + t * n_sb * tile_bytes;
                            let tile = &store[at..at + n_sb * tile_bytes];
                            let out = &mut blk[..KQ_TILE_ROWS * nb];
                            if let Some(qo) = q8_off {
                                let form_sb = n.div_ceil(256);
                                let q = qo + t * form_sb * peregrine_kernels::kq8::Q8K_SIZE;
                                let form =
                                    &w.q8()[q..q + form_sb * peregrine_kernels::kq8::Q8K_SIZE];
                                // A row whose width is not a whole number of
                                // super-blocks reads the padded activation buffer.
                                let fa: &peregrine_kernels::kq::KqActs =
                                    if n % 256 == 0 { kq } else { &kq_pad };
                                mk.q8k_gemm(form, form_sb, fa, nb, out);
                            } else {
                                match j.w.kind {
                                    Format::Q5_0 => {
                                        // The Q5_0 tile reads the per-32 activation
                                        // layout, the same one decode uses.
                                        if nb >= 4 {
                                            mk.kq_tiles_q5_0(tile, n_sb, &acts[..nb], nb, out);
                                        } else {
                                            mk.kq_tiles_q5_0_gemv(tile, n_sb, &acts[..nb], nb, out);
                                        }
                                    }
                                    Format::Q8_0 => {
                                        // Q8_0 also reads the per-32 layout: no
                                        // interleaved activation to build.
                                        if nb >= 4 {
                                            mk.kq_tiles_q8_0(tile, n_sb, &acts[..nb], nb, out);
                                        } else {
                                            mk.kq_tiles_q8_0_gemv(tile, n_sb, &acts[..nb], nb, out);
                                        }
                                    }
                                    Format::Q6_K => {
                                        if nb >= 4 {
                                            mk.kq_tiles_q6k(tile, n_sb, kq, nb, out);
                                        } else {
                                            mk.kq_tiles_q6k_gemv(tile, n_sb, &acts[..nb], nb, out);
                                        }
                                    }
                                    Format::Q5_K => {
                                        if nb >= 4 {
                                            mk.kq_tiles_q5k(tile, n_sb, kq, nb, out);
                                        } else {
                                            mk.kq_tiles_q5k_gemv(tile, n_sb, &acts[..nb], nb, out);
                                        }
                                    }
                                    _ => {
                                        if nb >= 4 {
                                            mk.kq_tiles_q4k(tile, n_sb, kq, nb, out);
                                        } else {
                                            mk.kq_tiles_q4k_gemv(tile, n_sb, &acts[..nb], nb, out);
                                        }
                                    }
                                }
                            }
                            // Only the tile's rows this worker owns: a range can
                            // start or end inside a tile, and the other rows of
                            // it belong to another worker (or to nobody).
                            let lo = r.max(t * KQ_TILE_ROWS) - t * KQ_TILE_ROWS;
                            let hi = rz.min((t + 1) * KQ_TILE_ROWS) - t * KQ_TILE_ROWS;
                            let u = Untiler {
                                out: &j.out,
                                b0,
                                rows,
                                nb,
                            };
                            // SAFETY: those rows of this job belong to this
                            // worker (a `share` range, or a chunk claimed from
                            // the counter), and the tile's rows are the job's.
                            unsafe { u.untile(t * KQ_TILE_ROWS, out, KQ_TILE_ROWS, lo, hi) };
                        }
                    }
                    // The rows past the last whole group of eight: today's
                    // kernel, on the file's own bytes.
                    let tail = r.max(repacked);
                    if tail < rz {
                        let bytes = w.bytes(j.w);
                        for rr in tail..rz {
                            mk.row(
                                j.w.kind,
                                &bytes[rr * rb..(rr + 1) * rb],
                                acts,
                                &mut tmp[..nb],
                            );
                            for (b, &v) in tmp[..nb].iter().enumerate() {
                                // SAFETY: the row of the concatenated row space
                                // is taken by exactly one worker.
                                unsafe { j.out.set((b0 + b) * rows + rr, v) };
                            }
                        }
                    }
                }
                MatPath::FloatPanels => {
                    // S4.9f: whole panels of the packed float store. A range that
                    // starts or ends inside a panel computes the panel and keeps
                    // the rows it asked for; a row's bits do not depend on that.
                    let off = j.w.fp_off.expect("a packed float path has a store offset");
                    let fmt = panel_fmt(j.w);
                    let pb = peregrine_kernels::fpack::panel_bytes(fmt, n);
                    let store = w.fp();
                    let mut blk = [0.0f32; 16 * MAX_BATCH];
                    let first = r / 16;
                    let last = rz.div_ceil(16);
                    for p in first..last {
                        let at = off + p * pb;
                        mk.fp_panels_gemm(
                            fmt,
                            &store[at..at + pb],
                            n,
                            1,
                            acts,
                            None,
                            &mut blk[..16 * nb],
                        );
                        for i in 0..16 {
                            let row = p * 16 + i;
                            if row < r || row >= rz {
                                continue;
                            }
                            for b in 0..nb {
                                // SAFETY: the row belongs to this worker (a `share`
                                // range, or a chunk claimed from the counter).
                                unsafe { j.out.set((b0 + b) * rows + row, blk[b * 16 + i]) };
                            }
                        }
                    }
                }
                MatPath::PerRow => {
                    let bytes = w.bytes(j.w);
                    let mut r = r;
                    // A prompt row of a matrix with a resident form (S5.4b; Q5_0
                    // joins where the tier's table selects it, S5.4f). This path is
                    // the Q6_K prompt kernel on x86 (its tile GEMV loses there) and
                    // every form format on a tier whose k-quant tiles are off; the
                    // form's own kernel is 8-row, so whole runs are multiplied from
                    // it and a short tail stays exact.
                    if let Some(q8_off) = j.w.q8_off.filter(|_| fast_row) {
                        let form_sb = n.div_ceil(256);
                        let rows = j.rows();
                        let repacked = (rows / KQ_TILE_ROWS) * KQ_TILE_ROWS;
                        let lo = r.min(repacked);
                        let hi = rz.min(repacked);
                        if lo < hi {
                            let mut blk = [0.0f32; KQ_TILE_ROWS * MAX_BATCH];
                            let first = lo / KQ_TILE_ROWS;
                            let last = hi.div_ceil(KQ_TILE_ROWS);
                            for t in first..last {
                                let q = q8_off + t * form_sb * peregrine_kernels::kq8::Q8K_SIZE;
                                let form =
                                    &w.q8()[q..q + form_sb * peregrine_kernels::kq8::Q8K_SIZE];
                                let out = &mut blk[..KQ_TILE_ROWS * nb];
                                let fa: &peregrine_kernels::kq::KqActs =
                                    if n % 256 == 0 { kq } else { &kq_pad };
                                mk.q8k_gemm(form, form_sb, fa, nb, out);
                                let lo = r.max(t * KQ_TILE_ROWS) - t * KQ_TILE_ROWS;
                                let hi = rz.min((t + 1) * KQ_TILE_ROWS) - t * KQ_TILE_ROWS;
                                let u = Untiler {
                                    out: &j.out,
                                    b0,
                                    rows,
                                    nb,
                                };
                                // SAFETY: those rows of this job belong to this
                                // worker, and the tile's rows are the job's.
                                unsafe { u.untile(t * KQ_TILE_ROWS, out, KQ_TILE_ROWS, lo, hi) };
                            }
                        }
                        let tail = r.max(repacked);
                        if tail < rz {
                            for rr in tail..rz {
                                mk.row(
                                    j.w.kind,
                                    &bytes[rr * rb..(rr + 1) * rb],
                                    acts,
                                    &mut tmp[..nb],
                                );
                                for (b, &v) in tmp[..nb].iter().enumerate() {
                                    // SAFETY: the row belongs to this worker.
                                    unsafe { j.out.set((b0 + b) * rows + rr, v) };
                                }
                            }
                        }
                        r = rz;
                    }
                    // A decode step (one activation) runs a Q5_0/Q8_0 row eight at a
                    // time: the activation block (and, for Q5_0, its lane sums) is then
                    // loaded once for eight rows instead of once per row. Each row is
                    // still accumulated block by block exactly as the one-row kernel
                    // does, so the bits do not depend on the batch (a prefill uses
                    // `dot_batch` unchanged).
                    let mut eight = [0.0f32; 8];
                    if nb == 1 && mk.rows8(j.w.kind) {
                        while r + 8 <= rz {
                            mk.row8(
                                j.w.kind,
                                &bytes[r * rb..(r + 8) * rb],
                                rb,
                                &acts[0],
                                &mut eight,
                            );
                            for (i, &v) in eight.iter().enumerate() {
                                // SAFETY: row `r + i` is in this worker's range.
                                unsafe { j.out.set(b0 + r + i, v) };
                            }
                            r += 8;
                        }
                    }
                    for r in r..rz {
                        mk.row(j.w.kind, &bytes[r * rb..(r + 1) * rb], acts, &mut tmp[..nb]);
                        for (b, &v) in tmp[..nb].iter().enumerate() {
                            // SAFETY: each row of the concatenated row space is taken by exactly one worker
                            // (a `share` range, or a chunk claimed from the counter), so `(job, r)` — hence the
                            // index `b*rows + r` within this job's output — is written once.
                            unsafe { j.out.set((b0 + b) * rows + r, v) };
                        }
                    }
                }
            }
            if let Some(t) = tf {
                crate::prof::add_shared(
                    crate::prof::format_slot(j.w.kind),
                    t.elapsed().as_nanos() as u64,
                );
            }
            off += rows;
        }
    };
    // S4.9f: a group that is entirely packed float panels is worked in (panel,
    // activation-row block) units — the units the kernel itself would have used, so
    // distributing them changes neither the work nor a bit of the result. Every
    // other group (and a mixed one) keeps the row partition above.
    if paths.iter().all(|&p| p == MatPath::FloatPanels) {
        fp_group(w, jobs, acts, nb, b0, pool, st);
        return;
    }
    // on a CPU with fast and slow cores (Zen 5 + Zen 5c) equal shares make every matmul wait for the
    // slowest core. Workers instead claim chunks of rows from a counter, so a fast core takes more of them.
    // Which worker computes a row never changes the row's value, so the output is identical either way.
    if dynamic_rows() && pool.width() > 1 {
        let mut chunk = (total / (pool.width() * 8)).clamp(8, 256);
        if tiled {
            chunk = chunk.div_ceil(16) * 16;
        }
        if pool.claiming() {
            // The pool is stalled on a worker the host has taken away: the join waits
            // only for the ranges in flight, not for that worker to come back.
            pool.run_claimed(total, chunk, |_wk, _nw, lo, hi| {
                let mut tmp = [0.0f32; MAX_BATCH];
                rows_range(lo, hi, &mut tmp);
            });
        } else {
            let next = std::sync::atomic::AtomicUsize::new(0);
            pool.run(|_wk, _nw| {
                let mut tmp = [0.0f32; MAX_BATCH];
                loop {
                    let lo = next.fetch_add(chunk, std::sync::atomic::Ordering::Relaxed);
                    if lo >= total {
                        break;
                    }
                    rows_range(lo, (lo + chunk).min(total), &mut tmp);
                }
            });
        }
    } else {
        pool.run(|wk, nw| {
            let (lo, hi) = if tiled {
                let (a, b) = share(total / 16, wk, nw);
                (a * 16, b * 16)
            } else {
                share(total, wk, nw)
            };
            let mut tmp = [0.0f32; MAX_BATCH];
            rows_range(lo, hi, &mut tmp);
        });
    }
    // The activation's traffic, per block; the weight traffic is counted once by
    // `matmul_group`, and the MACs here sum over blocks to the batch's whole.
    for (j, &path) in jobs.iter().zip(paths) {
        let n = j.w.dims[0] as usize;
        let rows = j.rows();
        st.macs += (rows * n * nb) as u64;
        // The activation's traffic: how many times the activation crossed the
        // memory interface, and how many bytes it was. The repacked tile reads
        // it once per eight output rows, the ordinary path once per row, so
        // this is where the design note's claim (a 32× cut) is measured.
        let per_row = act_bytes(&acts[0]);
        let reads = match path {
            MatPath::KqTiles => rows.div_ceil(KQ_TILE_ROWS),
            MatPath::Tiles => rows.div_ceil(16),
            // One pass over the activation per 16-row panel and L2-sized block.
            MatPath::FloatPanels => {
                rows.div_ceil(16) * nb.div_ceil(peregrine_kernels::fpack::row_block(n))
            }
            _ => rows,
        };
        st.bytes_acts += (reads * nb * per_row) as u64;
    }
    st.barriers += 1;
}

/// The activation rows one packed-float unit covers.
///
/// At most [`peregrine_kernels::fpack::row_block`] (half of L2), and few enough rows
/// that a matrix with too few panels still fills the pool: a group of one panel pair
/// (a router, the MTP head's slice) would otherwise be one unit, one thread. The
/// block only changes how often a panel is read, never a result.
fn fp_row_block(n_in: usize, nb: usize, per: usize, width: usize) -> usize {
    let want = (width / per.max(1)).max(1);
    nb.div_ceil(want)
        .min(peregrine_kernels::fpack::row_block(n_in))
        .max(1)
}

/// The packed float panels of a group, one work unit at a time (S4.9f).
///
/// A unit is [`FP_UNIT_PANELS`] panels against one [`row_block`](peregrine_kernels::fpack::row_block)
/// of activation rows; the units of every job are concatenated and partitioned
/// exactly as `matmul_group`'s row space is, so nothing about who computed a unit
/// can reach the result. Threading: with few activation rows there is one block and
/// the units are panels (a weight stream per worker); with many there are several
/// blocks and the batch is split too, which is the same work the kernel would do
/// serially and the same total weight traffic.
fn fp_group(
    w: &Weights,
    jobs: &[MatJob],
    acts: &[QuantAct],
    nb: usize,
    b_off: usize,
    pool: &Pool,
    st: &mut StepStats,
) {
    let store = w.fp();
    // Units per job, the running base of each job's units, and the block size each
    // job's units were laid out with (the accounting below reads the same numbers).
    let mut bases = [0usize; 4];
    let mut per_of = [0usize; 4];
    let mut bs_of = [0usize; 4];
    let mut total = 0usize;
    for (ji, j) in jobs.iter().enumerate() {
        bases[ji] = total;
        let n = j.w.dims[0] as usize;
        let panels = j.rows().div_ceil(16);
        let per = panels.div_ceil(FP_UNIT_PANELS).max(1);
        let bs = fp_row_block(n, nb, per, pool.width());
        per_of[ji] = per;
        bs_of[ji] = bs;
        total += per * nb.div_ceil(bs);
    }
    let unit = |u: usize| {
        // The job whose units hold `u`.
        let ji = bases[..jobs.len()]
            .iter()
            .rposition(|&b| b <= u)
            .expect("a unit belongs to a job");
        let j = &jobs[ji];
        let n = j.w.dims[0] as usize;
        let rows = j.rows();
        let panels = rows.div_ceil(16);
        let bs = bs_of[ji];
        let per = per_of[ji];
        let local = u - bases[ji];
        let first_panel = (local % per) * FP_UNIT_PANELS;
        let b0 = (local / per) * bs;
        let pn = FP_UNIT_PANELS.min(panels - first_panel);
        let bn = bs.min(nb - b0);
        let pb = peregrine_kernels::fpack::panel_bytes(panel_fmt(j.w), n);
        let off = j.w.fp_off.expect("a packed float path has a store offset") + first_panel * pb;
        let at = off + pn * pb;
        let tf = crate::prof::detail().then(Instant::now);
        let pan = pn * 16;
        let first_row = first_panel * 16;
        // The direct store is used for the engine's own block (b_off == 0): a wider
        // pass that the matmul driver split, and a shape whose rows are not whole
        // panels, go through the block and the copy.
        if b_off == 0 && first_row + pan <= rows {
            // The kernel writes straight into the projection's row-major output:
            // activation row `b` lands at `out[b · rows ..][..pn · 16]`, the rows this
            // unit owns. No temporary block, no copy (S4.9m).
            // SAFETY: unit `u` is computed by exactly one worker, and the window holds
            // the unit's own rows of every activation row; the other rows of the same
            // output belong to other units and are written by them alone.
            let win = unsafe {
                j.out
                    .window_mut(b0 * rows + first_row, (bn - 1) * rows + pan)
            };
            MatKernels::get().fp_panels_gemm_strided(
                panel_fmt(j.w),
                &store[off..at],
                n,
                pn,
                &acts[b0..b0 + bn],
                None,
                win,
                rows,
            );
        } else {
            // A matrix whose output rows are not a whole number of panels: the
            // kernel's padded lanes have no row in the output, so this last panel
            // goes through a block and only the rows that exist are stored.
            let mut blk = [0.0f32; FP_UNIT_PANELS * 16 * MAX_BATCH];
            let out = &mut blk[..pn * 16 * bn];
            MatKernels::get().fp_panels_gemm(
                panel_fmt(j.w),
                &store[off..at],
                n,
                pn,
                &acts[b0..b0 + bn],
                None,
                out,
            );
            for i in 0..pan {
                let row = first_row + i;
                if row >= rows {
                    break;
                }
                for b in 0..bn {
                    // SAFETY: unit `u` is computed by exactly one worker.
                    unsafe { j.out.set((b_off + b0 + b) * rows + row, out[b * pan + i]) };
                }
            }
        }
        if let Some(t) = tf {
            crate::prof::add_shared(
                crate::prof::format_slot(j.w.kind),
                t.elapsed().as_nanos() as u64,
            );
        }
    };
    if dynamic_rows() && pool.width() > 1 {
        if pool.claiming() {
            pool.run_claimed(total, 1, |_wk, _nw, lo, hi| {
                for u in lo..hi {
                    unit(u);
                }
            });
        } else {
            let next = std::sync::atomic::AtomicUsize::new(0);
            pool.run(|_wk, _nw| loop {
                let u = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if u >= total {
                    break;
                }
                unit(u);
            });
        }
    } else {
        pool.run(|wk, nw| {
            let (lo, hi) = share(total, wk, nw);
            for u in lo..hi {
                unit(u);
            }
        });
    }
    // The same accounting `matmul_group` does, for this block: the multiply count
    // and the activation's traffic (one pass per 16-row panel and L2-sized block).
    // The weight bytes are counted once per matrix by `matmul_group`.
    for (ji, j) in jobs.iter().enumerate() {
        let n = j.w.dims[0] as usize;
        let rows = j.rows();
        st.macs += (rows * n * nb) as u64;
        let blocks = nb.div_ceil(bs_of[ji].max(1));
        st.bytes_acts += (rows.div_ceil(16) * blocks * nb * act_bytes(&acts[0])) as u64;
    }
    st.barriers += 1;
}

/// One value per requested token id of one activation row, with the arithmetic the
/// full-row head uses for that weight row on this tier (S4.9d).
///
/// A row is computed by the same kernel the full path would run it in: the per-row
/// kernel (`quant::dot_batch`, or the eight-row Q5_0/Q8_0 one when the pass has a
/// single activation) for a matrix stored as rows; the eight-row k-quant tile, the
/// sixteen-row ternary tile or the packed float panel that contains it, taking the
/// one output. Every one of those kernels computes each output independently of its
/// neighbours, so the value is the full row's entry bit for bit. `acts` is the whole
/// pass's activation batch; `b` is taken in the `MAX_BATCH`-row block the matmul
/// driver would have handed the kernel.
#[allow(clippy::too_many_arguments)]
fn one_row_outputs(
    w: &Weights,
    head: &TensorInfo,
    acts: &[QuantAct],
    b: usize,
    ids: &[u32],
    out: &mut [f32],
    st: &mut StepStats,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    let n = head.dims[0] as usize;
    let n_out = head.dims[1] as usize;
    let rb = row_bytes(head.kind, n);
    // The matmul driver splits a wide pass into blocks of at most `DEFAULT_BATCH`
    // activation rows (a label row is never a fast row), and a block's kernel call is
    // the one the full path makes: take the block that holds this row, and the row's
    // place in it.
    let b0 = (b / DEFAULT_BATCH) * DEFAULT_BATCH;
    let nb = DEFAULT_BATCH.min(acts.len() - b0);
    let block = &acts[b0..b0 + nb];
    let b = b - b0;
    let mk = MatKernels::get();
    let path = mk.path(
        head.kind,
        head.tile_off.is_some(),
        head.kq_off.is_some(),
        head.fp_off.is_some(),
        block[0].ptq1,
    );
    // The batched k-quant tiles read the interleaved activation, which the full
    // path fills once per block (S3.4).
    if nb >= 4
        && matches!(head.kind, Format::Q4_K | Format::Q6_K)
        && head.kq_off.is_some()
        && mk.kq_tiles_for(head.kind)
    {
        // A label row is never a fast row: no whole-block sums are built (S5.4b).
        kq.fill_with(block, false);
    }
    let bytes = w.bytes(head);
    let mut blk = [0.0f32; 16 * MAX_BATCH];
    let mut tmp = [0.0f32; MAX_BATCH];
    let mut eight = [0.0f32; 8];
    for (k, &id) in ids.iter().enumerate() {
        let r = id as usize;
        assert!(r < n_out, "label {r} outside the head's {n_out} rows");
        let (v, read) = match path {
            MatPath::Tiles => {
                // The 16-row ternary tile that holds row `r`; its outputs are the
                // same whatever tile count the call covers.
                let toff = head.tile_off.expect("a tiled path has a tile offset");
                let tb = 16 * rb;
                let p = r / 16;
                mk.tiles_pq2(
                    &w.tiles()[toff + p * tb..toff + (p + 1) * tb],
                    n / 128,
                    1,
                    block,
                    &mut blk[..16 * nb],
                );
                (blk[(r % 16) * nb + b], 16)
            }
            MatPath::RowsPq2 | MatPath::RowsPtq1 => {
                let row = &bytes[r * rb..(r + 1) * rb];
                if path == MatPath::RowsPtq1 {
                    mk.rows_ptq1(row, rb, 1, block, &mut tmp[..nb]);
                } else {
                    mk.rows_pq2(row, rb, 1, block, &mut tmp[..nb]);
                }
                (tmp[b], 1)
            }
            MatPath::KqTiles => {
                let kq_off = head.kq_off.expect("the repacked path has a store offset");
                let n_sb = n / head.kind.block();
                let tile_bytes = match head.kind {
                    Format::Q4_K => peregrine_kernels::kq::Q4K8::SIZE,
                    Format::Q5_K => peregrine_kernels::kq::Q5K8::SIZE,
                    Format::Q5_0 => peregrine_kernels::kq::Q5_08::SIZE,
                    Format::Q8_0 => peregrine_kernels::kq::Q8_08::SIZE,
                    Format::Q6_K => peregrine_kernels::kq::Q6K8::SIZE,
                    _ => unreachable!("a repacked path has a repacked format"),
                };
                let repacked = (n_out / KQ_TILE_ROWS) * KQ_TILE_ROWS;
                if r >= repacked {
                    // Past the last whole group of eight: the file's own row kernel.
                    mk.row(
                        head.kind,
                        &bytes[r * rb..(r + 1) * rb],
                        block,
                        &mut tmp[..nb],
                    );
                    (tmp[b], 1)
                } else {
                    let t = r / KQ_TILE_ROWS;
                    let at = kq_off + t * n_sb * tile_bytes;
                    let tile = &w.kq()[at..at + n_sb * tile_bytes];
                    match head.kind {
                        Format::Q5_0 => {
                            if nb >= 4 {
                                mk.kq_tiles_q5_0(tile, n_sb, block, nb, &mut blk[..8 * nb])
                            } else {
                                mk.kq_tiles_q5_0_gemv(tile, n_sb, block, nb, &mut blk[..8 * nb])
                            }
                        }
                        Format::Q8_0 => {
                            if nb >= 4 {
                                mk.kq_tiles_q8_0(tile, n_sb, block, nb, &mut blk[..8 * nb])
                            } else {
                                mk.kq_tiles_q8_0_gemv(tile, n_sb, block, nb, &mut blk[..8 * nb])
                            }
                        }
                        Format::Q6_K => {
                            if nb >= 4 {
                                mk.kq_tiles_q6k(tile, n_sb, kq, nb, &mut blk[..8 * nb])
                            } else {
                                mk.kq_tiles_q6k_gemv(tile, n_sb, block, nb, &mut blk[..8 * nb])
                            }
                        }
                        Format::Q5_K => {
                            if nb >= 4 {
                                mk.kq_tiles_q5k(tile, n_sb, kq, nb, &mut blk[..8 * nb])
                            } else {
                                mk.kq_tiles_q5k_gemv(tile, n_sb, block, nb, &mut blk[..8 * nb])
                            }
                        }
                        _ => {
                            if nb >= 4 {
                                mk.kq_tiles_q4k(tile, n_sb, kq, nb, &mut blk[..8 * nb])
                            } else {
                                mk.kq_tiles_q4k_gemv(tile, n_sb, block, nb, &mut blk[..8 * nb])
                            }
                        }
                    }
                    (blk[b * KQ_TILE_ROWS + (r % KQ_TILE_ROWS)], KQ_TILE_ROWS)
                }
            }
            MatPath::FloatPanels => {
                let off = head.fp_off.expect("a packed float path has a store offset");
                let fmt = panel_fmt(head);
                let pb = peregrine_kernels::fpack::panel_bytes(fmt, n);
                let p = r / 16;
                mk.fp_panels_gemm(
                    fmt,
                    &w.fp()[off + p * pb..off + (p + 1) * pb],
                    n,
                    1,
                    block,
                    None,
                    &mut blk[..16 * nb],
                );
                (blk[b * 16 + (r % 16)], 16)
            }
            MatPath::PerRow => {
                if nb == 1 && mk.rows8(head.kind) {
                    let start = (r / 8) * 8;
                    if start + 8 <= n_out {
                        mk.row8(
                            head.kind,
                            &bytes[start * rb..(start + 8) * rb],
                            rb,
                            &block[0],
                            &mut eight,
                        );
                        (eight[r - start], 8)
                    } else {
                        mk.row(
                            head.kind,
                            &bytes[r * rb..(r + 1) * rb],
                            block,
                            &mut tmp[..nb],
                        );
                        (tmp[0], 1)
                    }
                } else {
                    mk.row(
                        head.kind,
                        &bytes[r * rb..(r + 1) * rb],
                        block,
                        &mut tmp[..nb],
                    );
                    (tmp[b], 1)
                }
            }
        };
        out[k] = v;
        st.bytes_weights += (read * rb) as u64;
        st.macs += (read * n * nb) as u64;
    }
}

/// The output head's rows for one pass (S4.9d).
///
/// `acts` holds one quantised activation per entry of `want_logits`, normed and
/// quantised by the caller exactly as the full head does it. Every row not named in
/// `label_rows` is written whole into `logits[row · n_out..]` through the same group
/// kernel as before. A label row — `(index into want_logits, start in label_ids,
/// count)` — is evaluated only at its token ids and written to `label_out`.
#[allow(clippy::too_many_arguments)]
pub fn output_rows(
    w: &Weights,
    head: &TensorInfo,
    acts: &[QuantAct],
    want_logits: &[usize],
    logits: &mut [f32],
    label_rows: &[(usize, usize, usize)],
    label_ids: &[u32],
    label_out: &mut [f32],
    pool: &Pool,
    st: &mut StepStats,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    let n_out = head.dims[1] as usize;
    let nl = want_logits.len();
    debug_assert!(acts.len() >= nl);
    // Every contiguous run of rows that wants the whole vocabulary is one group
    // call; its output starts at that run's first logit row.
    let mut i = 0usize;
    while i < nl {
        if label_rows.iter().any(|r| r.0 == i) {
            i += 1;
            continue;
        }
        let start = i;
        while i < nl && !label_rows.iter().any(|r| r.0 == i) {
            i += 1;
        }
        let job = MatJob {
            w: head,
            out: Disjoint::new(&mut logits[start * n_out..i * n_out]),
            rows: None,
        };
        matmul_group(w, &[job], &acts[start..i], i - start, pool, st, kq);
    }
    if label_rows.is_empty() {
        return;
    }
    let t0 = Instant::now();
    for &(row, start, count) in label_rows {
        debug_assert!(row < nl && start + count <= label_ids.len());
        one_row_outputs(
            w,
            head,
            acts,
            // `acts` is one row per `want_logits` entry, so the label row's
            // activation is at its position in that list, not its batch row.
            row,
            &label_ids[start..start + count],
            &mut label_out[start..start + count],
            st,
            kq,
        );
    }
    st.ns_matmul += t0.elapsed().as_nanos() as u64;
}

/// Several matrices that share one activation, each dense or LittleBit.
///
/// Dense members go through `matmul_group` unchanged (one barrier); LittleBit
/// members run as two barriers — every stage's latent product, then every
/// output row — via `lb_group`. `xs` is the same activation `acts` was
/// quantised from: the LittleBit path works on floats (its per-column scale
/// `g` is applied before the signs), while the dense path keeps its int8 form.
#[cfg_attr(not(feature = "littlebit"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub fn linear_group(
    w: &Weights,
    slb: Option<&Slb>,
    jobs: &[LinJob],
    acts: &[QuantAct],
    xs: &[f32],
    xstride: usize,
    nb: usize,
    lbx: SharedBuf<'_>,
    lbt: SharedBuf<'_>,
    pool: &Pool,
    st: &mut StepStats,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    debug_assert!(jobs.len() <= 4);
    let mut dense: [Option<MatJob>; 4] = [None; 4];
    #[cfg(feature = "littlebit")]
    let mut lb: [(usize, OutPtr<'_>); 4] = [(0, jobs[0].out); 4];
    let mut nd = 0usize;
    #[cfg(feature = "littlebit")]
    let mut nl = 0usize;
    for j in jobs {
        if let Some(w) = j.lin.dense() {
            dense[nd] = Some(MatJob {
                w,
                out: j.out,
                rows: None,
            });
            nd += 1;
        } else {
            #[cfg(feature = "littlebit")]
            if let Linear::Lb(ti) = j.lin {
                lb[nl] = (*ti, j.out);
                nl += 1;
            }
        }
    }
    if nd > 0 {
        let dj: [MatJob; 4] = std::array::from_fn(|i| dense[i.min(nd - 1)].unwrap());
        matmul_group(w, &dj[..nd], acts, nb, pool, st, kq);
    }
    #[cfg(feature = "littlebit")]
    if nl > 0 {
        lb_group(slb, &lb[..nl], xs, xstride, nb, lbx, lbt, pool, st);
    }
}

/// The LittleBit members of a fused group: `y = Σ_s h ⊙ (U·(ℓ ⊙ (Vᵀ·(g ⊙ x))))`.
///
/// Three phases. A serial prologue scales each stage's activation copy
/// (`x′ = g ⊙ x` — microseconds against the products). Barrier one runs every
/// latent row of every (job, stage) — `t[j] = Σ ±x′` — batched, so a bit row is
/// read once for the whole batch. A serial seam applies `ℓ` while transposing
/// the latents to row-per-batch order, into the (now dead) `x′` buffer. Barrier
/// two runs every output row, summing its stages: `out = Σ_s h_i·(Σ ±t′)`.
/// Weight traffic counted: the bit matrices plus the decoded f32 scales, once
/// per call — like the dense path, independent of batch size.
#[cfg(feature = "littlebit")]
#[allow(clippy::too_many_arguments)]
pub fn lb_group(
    slb: Option<&Slb>,
    jobs: &[(usize, OutPtr<'_>)],
    xs: &[f32],
    xstride: usize,
    nb: usize,
    lbx: SharedBuf<'_>,
    lbt: SharedBuf<'_>,
    pool: &Pool,
    st: &mut StepStats,
) {
    assert!(
        !xs.is_empty() && nb <= xs.len() / xstride.max(1),
        "batch of {nb} exceeds the activations it was given"
    );
    let t0 = Instant::now();
    let slb0 = slb.expect("a LittleBit job without a sidecar");
    // A wide pass is split into blocks of at most `MAX_BATCH` rows, like the dense path: a row's
    // value is a function of its activation row and the weights alone. The weight traffic is
    // counted once, in the loop after the blocks.
    let mut b0 = 0usize;
    while b0 < nb {
        let n = DEFAULT_BATCH.min(nb - b0);
        lb_block(
            slb0,
            jobs,
            &xs[b0 * xstride..],
            xstride,
            n,
            b0,
            lbx,
            lbt,
            pool,
            st,
        );
        b0 += n;
    }
    for &(ti, _) in jobs {
        st.bytes_weights += slb0.tensors[ti].bytes_per_use;
    }
    st.ns_matmul += t0.elapsed().as_nanos() as u64;
}

/// [`lb_group`] for one block of at most `DEFAULT_BATCH` activation rows starting at batch row `b0`.
#[cfg(feature = "littlebit")]
#[allow(clippy::too_many_arguments)]
fn lb_block(
    slb: &Slb,
    jobs: &[(usize, OutPtr<'_>)],
    xs: &[f32],
    xstride: usize,
    nb: usize,
    b0: usize,
    lbx: SharedBuf<'_>,
    lbt: SharedBuf<'_>,
    pool: &Pool,
    st: &mut StepStats,
) {
    assert!(
        nb <= MAX_BATCH,
        "batch of {nb} exceeds the matmul scratch of {MAX_BATCH}"
    );
    assert!(
        nb <= xs.len() / xstride.max(1),
        "batch of {nb} exceeds the activations it was given"
    );
    #[derive(Clone, Copy, Default)]
    struct Slot {
        ti: usize,
        job: usize,
        stage: usize,
        x_off: usize,
        t_off: usize,
    }
    // Four fused jobs (the most `linear_group` takes) × the eight stages the
    // container allows.
    const MAX_SLOTS: usize = 4 * 8;
    let mut slots = [Slot::default(); MAX_SLOTS];
    let mut ns = 0usize;
    let (mut x_off, mut t_off) = (0usize, 0usize);
    let (mut lat_rows, mut out_rows) = (0usize, 0usize);
    for (ji, &(ti, _)) in jobs.iter().enumerate() {
        let t = &slb.tensors[ti];
        for si in 0..t.stages.len() {
            assert!(
                ns < MAX_SLOTS,
                "too many LittleBit stages in one fused group"
            );
            slots[ns] = Slot {
                ti,
                job: ji,
                stage: si,
                x_off,
                t_off,
            };
            ns += 1;
            // The x′ region is reused for ℓ⊙t after stage one; r_pad ≤ d_in is
            // a loader invariant, so the reuse always fits.
            x_off += nb * t.d_in;
            t_off += nb * t.r_pad;
            lat_rows += t.r_pad;
        }
        out_rows += t.d_out;
    }
    let slots = &slots[..ns];
    let i8p = crate::lbq::i8_enabled();
    for s in slots {
        let t = &slb.tensors[s.ti];
        let g = &t.stages[s.stage].g;
        // SAFETY: each slot's x′ region is disjoint by construction, and no
        // worker runs while this serial prologue writes it.
        let xp = unsafe { lbx.chunk(s.x_off, nb * t.d_in) };
        for b in 0..nb {
            let src = &xs[b * xstride..b * xstride + t.d_in];
            if i8p {
                // Packed int8 row in the space the f32 row would take (lbq.rs).
                crate::lbq::quantize_i8_row(&mut xp[b * t.d_in..], t.d_in, |i| src[i] * g[i]);
                continue;
            }
            let dst = &mut xp[b * t.d_in..(b + 1) * t.d_in];
            for (d, (&x, &gv)) in dst.iter_mut().zip(src.iter().zip(g)) {
                *d = x * gv;
            }
        }
    }
    pool.run(|wk, nw| {
        let (lo, hi) = share(lat_rows, wk, nw);
        let mut tmp = [0.0f32; MAX_BATCH];
        let mut base = 0usize;
        for s in slots {
            let t = &slb.tensors[s.ti];
            let (a, z) = (lo.max(base), hi.min(base + t.r_pad));
            if a < z {
                let v = slb.v_bits(t, s.stage);
                let vrb = t.d_in / 8;
                // SAFETY: x′ was fully written before the barrier and only read
                // here; latent rows a..z belong to this worker alone.
                let xp = unsafe { lbx.slice_ref(s.x_off, nb * t.d_in) };
                for j in (a - base)..(z - base) {
                    let row = &v[j * vrb..(j + 1) * vrb];
                    if i8p {
                        crate::lbq::lb_dot_i8_batch(row, t.d_in, xp, t.d_in, nb, &mut tmp[..nb]);
                    } else {
                        crate::lbq::lb_dot_batch(row, t.d_in, xp, t.d_in, nb, &mut tmp[..nb]);
                    }
                    // SAFETY: slot `s` latent row `j` belongs to this worker alone.
                    unsafe { lbt.chunk(s.t_off + j * nb, nb) }.copy_from_slice(&tmp[..nb]);
                }
            }
            base += t.r_pad;
        }
    });
    st.barriers += 1;
    for s in slots {
        let t = &slb.tensors[s.ti];
        let l = &t.stages[s.stage].l;
        // SAFETY: serial seam between the barriers; regions as above.
        let src = unsafe { lbt.slice_ref(s.t_off, nb * t.r_pad) };
        let dst = unsafe { lbx.chunk(s.x_off, nb * t.r_pad) };
        for b in 0..nb {
            if i8p {
                crate::lbq::quantize_i8_row(&mut dst[b * t.r_pad..], t.r_pad, |j| {
                    l[j] * src[j * nb + b]
                });
                continue;
            }
            for (j, &lv) in l.iter().enumerate() {
                dst[b * t.r_pad + j] = lv * src[j * nb + b];
            }
        }
    }
    pool.run(|wk, nw| {
        let (lo, hi) = share(out_rows, wk, nw);
        let mut tmp = [0.0f32; MAX_BATCH];
        let mut acc = [0.0f32; MAX_BATCH];
        let mut base = 0usize;
        for (ji, &(ti, out)) in jobs.iter().enumerate() {
            let t = &slb.tensors[ti];
            let (a, z) = (lo.max(base), hi.min(base + t.d_out));
            if a < z {
                let urb = t.r_pad / 8;
                // Per-job stage setup, once per worker rather than once per row:
                // the row loop below runs a few hundred nanoseconds per row.
                let mut st2: [(&[u8], &[f32], &[f32]); 8] = [(&[], &[], &[]); 8];
                let mut ns2 = 0usize;
                for s in slots.iter().filter(|s| s.job == ji) {
                    assert!(
                        ns2 < st2.len(),
                        "more than 8 LittleBit stages in one tensor"
                    );
                    // SAFETY: ℓ⊙t was fully written in the seam; read-only here.
                    let tp = unsafe { lbx.slice_ref(s.x_off, nb * t.r_pad) };
                    st2[ns2] = (slb.u_bits(t, s.stage), tp, &t.stages[s.stage].h[..]);
                    ns2 += 1;
                }
                for i in (a - base)..(z - base) {
                    acc[..nb].fill(0.0);
                    for &(u, tp, hs) in &st2[..ns2] {
                        let row = &u[i * urb..(i + 1) * urb];
                        if i8p {
                            crate::lbq::lb_dot_i8_batch(
                                row,
                                t.r_pad,
                                tp,
                                t.r_pad,
                                nb,
                                &mut tmp[..nb],
                            );
                        } else {
                            crate::lbq::lb_dot_batch(row, t.r_pad, tp, t.r_pad, nb, &mut tmp[..nb]);
                        }
                        let h = hs[i];
                        for (av, &tv) in acc[..nb].iter_mut().zip(&tmp[..nb]) {
                            *av += h * tv;
                        }
                    }
                    for (b, &v) in acc[..nb].iter().enumerate() {
                        // SAFETY: output row `i` of job `ji` belongs to this worker.
                        unsafe { out.set((b0 + b) * t.d_out + i, v) };
                    }
                }
            }
            base += t.d_out;
        }
    });
    st.barriers += 1;
    for &(ti, _) in jobs {
        let t = &slb.tensors[ti];
        st.macs += t.macs_per_row() * nb as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The form's activation, filled on the pool, is the serial fill's bytes (S5.4o).
    ///
    /// The fill is the one step that moved onto the pool, and a group's bytes depend on
    /// its own four activation rows only, so who writes a group cannot reach another
    /// group — but the buffers are shared, so the property is worth pinning: 71 rows is
    /// not a whole number of groups (the last one is padded with zero codes) and the
    /// shape is past the size gate the parallel path needs.
    #[test]
    fn the_parallel_form_fill_is_the_serial_fill() {
        let (n, rows) = (4096usize, 71usize);
        let mut acts: Vec<QuantAct> = (0..rows)
            .map(|b| {
                let x: Vec<f32> = (0..n)
                    .map(|i| (((b * 37 + i * 13) % 251) as f32 - 125.0) / 64.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n);
                a.fill(&x);
                a
            })
            .collect();
        acts[0].prompt = true;
        for sum256 in [true, false] {
            let mut serial = peregrine_kernels::kq::KqActs::default();
            serial.fill_with(&acts, sum256);
            let mut parallel = peregrine_kernels::kq::KqActs::default();
            let pool = Pool::new(4);
            fill_form(&pool, &mut parallel, &acts, sum256);
            assert_eq!(serial.q, parallel.q, "codes, sum256={sum256}");
            assert_eq!(serial.d, parallel.d, "scales, sum256={sum256}");
            assert_eq!(serial.bs, parallel.bs, "run sums, sum256={sum256}");
            assert_eq!(serial.bs16, parallel.bs16, "16-run sums, sum256={sum256}");
            assert_eq!(serial.bsp, parallel.bsp, "paired sums, sum256={sum256}");
            assert_eq!(
                serial.sum256, parallel.sum256,
                "whole-block sums, sum256={sum256}"
            );
        }
    }
}
