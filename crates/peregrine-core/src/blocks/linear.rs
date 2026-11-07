//! Fused linear transformation blocks (dense GEMM groups and LittleBit).

use crate::blocks::par::{OutPtr, SharedBuf};
use crate::gguf::TensorInfo;
use crate::lbq::Slb;
use crate::model::{Linear, StepStats, Weights};
use crate::pool::{share, Pool};
use crate::quant::{row_bytes, QuantAct};
use crate::table::{MatKernels, MatPath};
use std::time::Instant;

/// The rows one repacked k-quant tile holds.
const KQ_TILE_ROWS: usize = 8;

/// Largest batch one matmul call accepts. A stack array of this many scratch floats
/// holds one row's results; `PREFILL_CHUNK` and `draft_len + 1` are both far below it.
pub const MAX_BATCH: usize = 128;

/// whether matmul rows are handed out in chunks from a counter (default) or in equal shares
/// (`PEREGRINE_DYNAMIC=0`, the static partition).
fn dynamic_rows() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| crate::options::get().dynamic_rows)
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
        + 4 * (a.d.len() + a.dk.len() + a.td.len())
        + 2 * (a.sums.len() + a.bsums.len() + a.tls.len())
}

/// One matrix of a fused group, dense or LittleBit. `linear_group` dispatches.
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
        nb <= MAX_BATCH,
        "batch of {nb} exceeds the matmul scratch of {MAX_BATCH}"
    );
    let t0 = Instant::now();
    let total: usize = jobs.iter().map(|j| j.rows()).sum();
    let acts = &acts[..nb];
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
            j.w.kind,
            j.w.tile_off.is_some(),
            j.w.kq_off.is_some(),
            act_ptq1,
        );
    }
    let paths = &paths[..jobs.len()];
    let tiled = paths.contains(&MatPath::Tiles);
    let kq_tiled = paths.contains(&MatPath::KqTiles);
    // The interleaved activation is built once for the group, from the
    // activation the group shares: every output row of the group reads it, so
    // building it per matrix would multiply the work by the group's size.
    if kq_tiled && nb >= 4 {
        kq.fill(&acts[..nb]);
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
                                unsafe { j.out.set(b * rows + r + i, blk[i * nb + b]) };
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
                                unsafe { j.out.set(b * rows + r + i, blk[i * nb + b]) };
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
                    let tile_bytes = peregrine_kernels::kq::Q4K8::SIZE;
                    let store = w.kq();
                    let rows = j.rows();
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
                            if nb >= 4 {
                                mk.kq_tiles_q4k(tile, n_sb, kq, nb, out);
                            } else {
                                mk.kq_tiles_q4k_gemv(tile, n_sb, &acts[..nb], nb, out);
                            }
                            for i in 0..KQ_TILE_ROWS {
                                let row = t * KQ_TILE_ROWS + i;
                                if row < r || row >= rz {
                                    continue;
                                }
                                for b in 0..nb {
                                    // SAFETY: the row belongs to this worker (a
                                    // `share` range, or a chunk claimed from
                                    // the counter), and the tile's rows are the
                                    // job's rows.
                                    unsafe { j.out.set(b * rows + row, out[b * KQ_TILE_ROWS + i]) };
                                }
                            }
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
                                unsafe { j.out.set(b * rows + rr, v) };
                            }
                        }
                    }
                }
                MatPath::PerRow => {
                    let bytes = w.bytes(j.w);
                    for r in r..rz {
                        mk.row(j.w.kind, &bytes[r * rb..(r + 1) * rb], acts, &mut tmp[..nb]);
                        for (b, &v) in tmp[..nb].iter().enumerate() {
                            // SAFETY: each row of the concatenated row space is taken by exactly one worker
                            // (a `share` range, or a chunk claimed from the counter), so `(job, r)` — hence the
                            // index `b*rows + r` within this job's output — is written once.
                            unsafe { j.out.set(b * rows + r, v) };
                        }
                    }
                }
            }
            off += rows;
        }
    };
    // on a CPU with fast and slow cores (Zen 5 + Zen 5c) equal shares make every matmul wait for the
    // slowest core. Workers instead claim chunks of rows from a counter, so a fast core takes more of them.
    // Which worker computes a row never changes the row's value, so the output is identical either way.
    if dynamic_rows() && pool.width() > 1 {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut chunk = (total / (pool.width() * 8)).clamp(8, 256);
        if tiled {
            chunk = chunk.div_ceil(16) * 16;
        }
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
    // Counted once per matrix: the bytes crossed the memory interface once,
    // however many activations reused them.
    for (j, &path) in jobs.iter().zip(paths) {
        let n = j.w.dims[0] as usize;
        let rows = j.rows();
        st.bytes_weights += (rows * row_bytes(j.w.kind, n)) as u64;
        st.macs += (rows * n * nb) as u64;
        // The activation's traffic: how many times the activation crossed the
        // memory interface, and how many bytes it was. The repacked tile reads
        // it once per eight output rows, the ordinary path once per row, so
        // this is where the design note's claim (a 32× cut) is measured.
        let per_row = act_bytes(&acts[0]);
        let reads = match path {
            MatPath::KqTiles => rows.div_ceil(KQ_TILE_ROWS),
            MatPath::Tiles => rows.div_ceil(16),
            _ => rows,
        };
        st.bytes_acts += (reads * nb * per_row) as u64;
    }
    st.barriers += 1;
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
        nb <= MAX_BATCH,
        "batch of {nb} exceeds the matmul scratch of {MAX_BATCH}"
    );
    let t0 = Instant::now();
    let slb = slb.expect("a LittleBit job without a sidecar");
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
                        unsafe { out.set(b * t.d_out + i, v) };
                    }
                }
            }
            base += t.d_out;
        }
    });
    st.barriers += 1;
    for &(ti, _) in jobs {
        let t = &slb.tensors[ti];
        st.bytes_weights += t.bytes_per_use;
        st.macs += t.macs_per_row() * nb as u64;
    }
    st.ns_matmul += t0.elapsed().as_nanos() as u64;
}
