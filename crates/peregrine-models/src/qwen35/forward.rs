//! Qwen3.5 hybrid forward pass and scratch buffers (ADR 0006).

use std::sync::atomic::{AtomicU64, Ordering as AO};
use std::time::Instant;

use crate::qwen35::load::{HybridCfg, Mixer};
use crate::qwen35::Qwen35;
use peregrine_core::blocks;
use peregrine_core::blocks::linear::MatJob;
use peregrine_core::blocks::norm::head_rmsnorm;
use peregrine_core::blocks::par::{Disjoint, RecScratch, RowPtrs, SnapPtrs};
use peregrine_core::blocks::recurrent;
use peregrine_core::blocks::rotate::{RowPrep, VPerm};
use peregrine_core::blocks::ROW_PARALLEL;
use peregrine_core::gguf::TensorInfo;
use peregrine_core::model::{Caches, Config, State, StepStats, Weights};
use peregrine_core::pool::Pool;
use peregrine_kernels::quant::QuantAct;
use peregrine_kernels::{table, vmath};

// Profiling buckets (`PEREGRINE_PROF=1`): wall nanoseconds per part of the hybrid pass, read by `profile()`.
const PROF_NAMES: [&str; 16] = [
    "embed",
    "norm",
    "rotate",
    "matmul",
    "gdn_ab",
    "gdn_conv",
    "gdn_delta",
    "gdn_gnorm",
    "attn",
    "ffn_act",
    "mm<8MB ns",
    "mm<8MB bytes",
    "mm8-20MB ns",
    "mm8-20MB bytes",
    "mm>20MB ns",
    "mm>20MB bytes",
];
static PROF: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];

fn prof_on() -> bool {
    use std::sync::OnceLock;
    static P: OnceLock<bool> = OnceLock::new();
    *P.get_or_init(|| peregrine_core::options::get().prof)
}

struct Tick(Option<Instant>, usize);

impl Tick {
    fn new(b: usize) -> Tick {
        Tick(prof_on().then(Instant::now), b)
    }
}

impl Drop for Tick {
    fn drop(&mut self) {
        if let Some(t) = self.0 {
            PROF[self.1].fetch_add(t.elapsed().as_nanos() as u64, AO::Relaxed);
        }
    }
}

/// The profiling buckets since the last call, `(name, ns)`, and reset them.
pub fn profile() -> Vec<(&'static str, u64)> {
    PROF_NAMES
        .iter()
        .zip(PROF.iter())
        .map(|(n, v)| (*n, v.swap(0, AO::Relaxed)))
        .collect()
}

/// Scratch for the hybrid pass, beside `State`'s own buffers.
pub struct HyScratch {
    /// `attn_q` output (q and gate interleaved per head) or the GDN `attn_qkv` output, per row.
    pub(crate) big: Vec<f32>,
    pub(crate) big_w: usize,
    /// GDN `z`, or the attention gate, per row.
    pub(crate) z: Vec<f32>,
    pub(crate) z_w: usize,
    /// GDN per row: `[β (n_v) | α (n_v)]`, then `[β | decay]` after the gate maths.
    pub(crate) ab: Vec<f32>,
    /// Rotated matmul inputs, per row.
    pub(crate) rot: Vec<f32>,
    pub(crate) rot_w: usize,
    /// GDN output per row, `value_dim`.
    pub(crate) o: Vec<f32>,
    /// Runs of rows sharing one cache, and the caches already seen, reused every call.
    pub(crate) runs: Vec<(usize, usize, usize)>,
    pub(crate) seen: Vec<usize>,
    /// Recurrent-state pointers for the delta-rule and conv workers, reused every layer.
    pub(crate) ptrs: RecScratch,
}

impl HyScratch {
    pub fn new(c: &Config, h: &HybridCfg, b: usize) -> HyScratch {
        let big_w = (2 * c.q_dim()).max(h.conv_dim());
        let z_w = h.value_dim().max(c.q_dim());
        let rot_w = c.n_ff.max(c.d_model).max(h.value_dim()).max(c.q_dim());
        HyScratch {
            big: vec![0.0; b * big_w],
            big_w,
            z: vec![0.0; b * z_w],
            z_w,
            ab: vec![0.0; b * 2 * h.n_v_heads],
            rot: vec![0.0; b * rot_w],
            rot_w,
            o: vec![0.0; b * h.value_dim()],
            runs: Vec::with_capacity(b),
            seen: Vec::with_capacity(b),
            ptrs: RecScratch {
                rec: vec![(std::ptr::null_mut(), std::ptr::null_mut()); b],
                snap: vec![None; b],
            },
        }
    }
}

/// Runs of at most this many rows of one cache attend one row at a time.
const SHORT_RUN: usize = 4;

impl Qwen35 {
    /// The sign vector of an input width, when the file has a rotation.
    #[inline]
    pub(crate) fn signs_for(&self, width: usize) -> Result<Option<&[f32]>, String> {
        match &self.rot {
            Some(r) => r.signs_for(width).map(Some),
            None => Ok(None),
        }
    }

    #[inline]
    pub(crate) fn vperm(&self, perm: bool) -> Option<VPerm> {
        perm.then_some(VPerm {
            dv: self.cfg.dv,
            n_k_heads: self.cfg.n_k_heads,
            n_v_heads: self.cfg.n_v_heads,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rotate_rows(
        &self,
        hs: &mut HyScratch,
        acts: &mut [QuantAct],
        src: &[f32],
        w: usize,
        nb: usize,
        perm: bool,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        let _t = Tick::new(2);
        let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
        match &self.rot {
            Some(rot) => rot.rotate_rows(
                &mut hs.rot,
                hs.rot_w,
                acts,
                src,
                w,
                nb,
                self.vperm(perm),
                parallel,
                pool,
                stats,
            ),
            // No rotation: the row is quantised as it is.
            None => blocks::rotate::fill_rows(acts, src, w, nb, parallel, pool, stats),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rotate_rows_pre(
        &self,
        hs: &mut HyScratch,
        acts: &mut [QuantAct],
        src: &mut [f32],
        w: usize,
        nb: usize,
        perm: bool,
        pre: Option<&RowPrep>,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        let _t = Tick::new(2);
        let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
        match &self.rot {
            Some(rot) => rot.rotate_rows_pre(
                &mut hs.rot,
                hs.rot_w,
                acts,
                src,
                w,
                nb,
                self.vperm(perm),
                pre,
                parallel,
                pool,
                stats,
            ),
            // No rotation: the row is prepared and quantised as it is.
            None => blocks::rotate::fill_rows_pre(acts, src, w, nb, pre, parallel, pool, stats),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mm(
        &self,
        w: &Weights,
        jobs: &[(&TensorInfo, Disjoint<'_, f32>)],
        acts: &[QuantAct],
        nb: usize,
        pool: &Pool,
        stats: &mut StepStats,
        kq: &mut peregrine_kernels::kq::KqActs,
    ) {
        assert!(!jobs.is_empty() && jobs.len() <= 4);
        let mj: [MatJob; 4] = std::array::from_fn(|i| {
            let (wi, o) = jobs[i.min(jobs.len() - 1)];
            MatJob {
                w: wi,
                out: o,
                rows: None,
            }
        });
        let mj = &mj[..jobs.len()];
        self.mm_jobs(w, mj, acts, nb, pool, stats, kq);
    }

    #[allow(clippy::too_many_arguments)]
    fn mm_jobs(
        &self,
        w: &Weights,
        mj: &[MatJob],
        acts: &[QuantAct],
        nb: usize,
        pool: &Pool,
        stats: &mut StepStats,
        kq: &mut peregrine_kernels::kq::KqActs,
    ) {
        let _t = Tick::new(3);
        let t0 = prof_on().then(Instant::now);
        blocks::linear::matmul_group(w, mj, acts, nb, pool, stats, kq);
        if let Some(t0) = t0 {
            let bytes: u64 = mj.iter().map(|j| j.w.bytes()).sum();
            let k = if bytes < 8 << 20 {
                10
            } else if bytes < 20 << 20 {
                12
            } else {
                14
            };
            PROF[k].fetch_add(t0.elapsed().as_nanos() as u64, AO::Relaxed);
            PROF[k + 1].fetch_add(bytes, AO::Relaxed);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_hybrid(
        &self,
        w: &Weights,
        c: &Config,
        st8: &mut State,
        caches: &mut Caches,
        pool: &Pool,
        tokens: &[u32],
        positions: &[usize],
        want_logits: &[usize],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let t_fwd = Instant::now();
        let hc = self.cfg;
        let nb = tokens.len();
        let d = c.d_model;
        let State {
            x,
            xb,
            xb2,
            hb,
            hb2,
            q,
            k,
            v,
            attn,
            rope,
            acts,
            kq,
            logits,
            fam,
            want_hidden,
            hidden,
            ..
        } = st8;
        if fam.is_none() {
            *fam = Some(Box::new(HyScratch::new(c, &self.cfg, st8.max_batch)));
        }
        for a in acts.iter_mut().take(nb) {
            a.ternary = self.cfg.ternary;
            a.ptq1 = self.cfg.ptq1 && table::MatKernels::get().ptq1_layout();
            a.tb_span = self.cfg.tb_span;
        }
        let hs = fam
            .as_mut()
            .and_then(|f| f.downcast_mut::<HyScratch>())
            .ok_or("state has no hybrid scratch")?;
        for b in 0..nb {
            if !caches.get(b).has_recurrent() {
                return Err("the cache has no recurrent state (KvCache::with_recurrent)".into());
            }
        }
        {
            caches.runs_into(nb, &mut hs.runs);
            let seen = &mut hs.seen;
            seen.clear();
            for &(r0, len, ci) in &hs.runs {
                let ok = !seen.contains(&ci)
                    && positions[r0] == caches.cache(ci).filled()
                    && (1..len).all(|k| positions[r0 + k] == positions[r0] + k);
                if !ok {
                    return Err(format!(
                        "rows {r0}..{} do not continue their recurrent cache in order (cache holds {}, row starts at {})",
                        r0 + len,
                        caches.cache(ci).filled(),
                        positions[r0]
                    ));
                }
                seen.push(ci);
            }
        }

        // ---- embedding: dequantised row, then the inverse rotation s ⊙ H·row ----
        let te = Tick::new(0);
        let se = self.signs_for(d)?;
        for (i, &t) in tokens.iter().enumerate() {
            let row = &mut x[i * d..(i + 1) * d];
            blocks::embed::embed(&self.tok_embd, w, t, row, stats);
            // A file with no rotation uses its embedding rows as they are.
            if let (Some(rot), Some(se)) = (&self.rot, se) {
                rot.rotate_embedding(row, se);
            }
        }
        drop(te);

        for l in &self.layers {
            // ---- mixer ----
            {
                let nw: &TensorInfo = &l.attn_norm;
                let pre = |b: usize, row: &mut [f32]| {
                    blocks::norm::rmsnorm_into(w, nw, c.eps, &x[b * d..(b + 1) * d], row)
                };
                self.rotate_rows_pre(hs, acts, xb, d, nb, false, Some(&pre), pool, stats);
                stats.bytes_weights += l.attn_norm.bytes();
            }
            match &l.mix {
                Mixer::Gdn {
                    qkv,
                    gate,
                    ab,
                    conv,
                    dt_bias,
                    a,
                    norm,
                    out,
                    idx,
                } => {
                    self.mm(
                        w,
                        &[
                            (qkv, Disjoint::new(&mut hs.big)),
                            (gate, Disjoint::new(&mut hs.z)),
                        ],
                        &acts[..nb],
                        nb,
                        pool,
                        stats,
                        kq,
                    );
                    restride(&mut hs.big, hc.conv_dim(), hs.big_w, nb);
                    restride(&mut hs.z, hc.value_dim(), hs.z_w, nb);
                    self.gdn(
                        c, hs, xb, caches, ab, conv, dt_bias, a, *idx, positions, nb, pool, stats,
                    );
                    let vd = hc.value_dim();
                    let mut o = std::mem::take(&mut hs.o);
                    let z = std::mem::take(&mut hs.z);
                    {
                        let zw = hs.z_w;
                        let zr: &[f32] = &z;
                        let pre = |b: usize, row: &mut [f32]| {
                            recurrent::gated_norm_row(
                                row,
                                &zr[b * zw..b * zw + vd],
                                norm,
                                hc.n_v_heads,
                                hc.dv,
                                c.eps,
                            )
                        };
                        self.rotate_rows_pre(
                            hs,
                            acts,
                            &mut o,
                            vd,
                            nb,
                            self.v_grouped,
                            Some(&pre),
                            pool,
                            stats,
                        );
                    }
                    hs.o = o;
                    hs.z = z;
                    self.mm(
                        w,
                        &[(out, Disjoint::new(xb2))],
                        &acts[..nb],
                        nb,
                        pool,
                        stats,
                        kq,
                    );
                }
                Mixer::Attn {
                    q: mq,
                    k: mk,
                    v: mv,
                    q_norm,
                    k_norm,
                    o,
                    idx,
                } => {
                    let (qd, kvd, hd) = (c.q_dim(), c.kv_dim(), c.head_dim);
                    self.mm(
                        w,
                        &[
                            (mq, Disjoint::new(&mut hs.big)),
                            (mk, Disjoint::new(k)),
                            (mv, Disjoint::new(v)),
                        ],
                        &acts[..nb],
                        nb,
                        pool,
                        stats,
                        kq,
                    );
                    let ta = Tick::new(8);
                    {
                        let big: &[f32] = &hs.big;
                        let zw = hs.z_w;
                        let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
                        blocks::par::for_row_chunks_3(
                            pool,
                            parallel,
                            q,
                            qd,
                            k,
                            kvd,
                            &mut hs.z,
                            zw,
                            nb,
                            blocks::par::Sched::Dynamic(1),
                            |_wk, rows, q_chunk, k_chunk, z_chunk| {
                                for (i, b) in rows.enumerate() {
                                    let q_row = &mut q_chunk[i * qd..(i + 1) * qd];
                                    let k_row = &mut k_chunk[i * kvd..(i + 1) * kvd];
                                    let z_row = &mut z_chunk[i * zw..i * zw + qd];
                                    for h in 0..c.n_head {
                                        let src = b * 2 * qd + h * 2 * hd;
                                        q_row[h * hd..(h + 1) * hd]
                                            .copy_from_slice(&big[src..src + hd]);
                                        z_row[h * hd..(h + 1) * hd]
                                            .copy_from_slice(&big[src + hd..src + 2 * hd]);
                                    }
                                    for row in q_row.chunks_exact_mut(hd) {
                                        head_rmsnorm(row, q_norm, c.eps);
                                    }
                                    for row in k_row.chunks_exact_mut(hd) {
                                        head_rmsnorm(row, k_norm, c.eps);
                                    }
                                    blocks::rope::rope(
                                        rope,
                                        q_row,
                                        c.n_head,
                                        positions[b],
                                        c.head_dim,
                                        c.rope_dim,
                                        c.rope_neox,
                                    );
                                    blocks::rope::rope(
                                        rope,
                                        k_row,
                                        c.n_kv_head,
                                        positions[b],
                                        c.head_dim,
                                        c.rope_dim,
                                        c.rope_neox,
                                    );
                                }
                            },
                        );
                        if parallel {
                            stats.barriers += 1;
                        }
                    }
                    for b in 0..nb {
                        let kv_cache = caches.get_mut(b);
                        for h in 0..c.n_kv_head {
                            let ks = b * kvd + h * hd;
                            kv_cache.put(*idx, h, positions[b], &k[ks..ks + hd], &v[ks..ks + hd]);
                        }
                    }
                    for b in 0..nb {
                        caches.get_mut(b).advance(positions[b] + 1);
                    }
                    let t_attn = Instant::now();
                    for (r0, len, ci) in caches.runs(nb) {
                        if len <= SHORT_RUN {
                            for k_idx in 0..len {
                                blocks::attn::attend(
                                    c,
                                    attn,
                                    q,
                                    xb,
                                    caches.cache(ci),
                                    pool,
                                    *idx,
                                    1,
                                    positions[r0 + k_idx],
                                    r0 + k_idx,
                                    stats,
                                );
                            }
                        } else {
                            blocks::attn::attend(
                                c,
                                attn,
                                q,
                                xb,
                                caches.cache(ci),
                                pool,
                                *idx,
                                len,
                                positions[r0],
                                r0,
                                stats,
                            );
                        }
                    }
                    stats.ns_attn += t_attn.elapsed().as_nanos() as u64;
                    drop(ta);
                    {
                        let z = std::mem::take(&mut hs.z);
                        let zw = hs.z_w;
                        let zr: &[f32] = &z;
                        let pre = |b: usize, row: &mut [f32]| {
                            vmath::mul_sigmoid(row, &zr[b * zw..b * zw + qd])
                        };
                        self.rotate_rows_pre(hs, acts, xb, qd, nb, false, Some(&pre), pool, stats);
                        hs.z = z;
                    }
                    self.mm(
                        w,
                        &[(o, Disjoint::new(xb2))],
                        &acts[..nb],
                        nb,
                        pool,
                        stats,
                        kq,
                    );
                }
            }
            for i in 0..nb * d {
                x[i] += xb2[i];
            }

            // ---- feed-forward ----
            {
                let nw: &TensorInfo = &l.post_norm;
                let pre = |b: usize, row: &mut [f32]| {
                    blocks::norm::rmsnorm_into(w, nw, c.eps, &x[b * d..(b + 1) * d], row)
                };
                self.rotate_rows_pre(hs, acts, xb, d, nb, false, Some(&pre), pool, stats);
                stats.bytes_weights += l.post_norm.bytes();
            }
            self.mm(
                w,
                &[(&l.gate, Disjoint::new(hb)), (&l.up, Disjoint::new(hb2))],
                &acts[..nb],
                nb,
                pool,
                stats,
                kq,
            );
            let f = c.n_ff;
            if nb >= ROW_PARALLEL {
                let up: &[f32] = hb2;
                let pre = |b: usize, row: &mut [f32]| vmath::swiglu(row, &up[b * f..(b + 1) * f]);
                self.rotate_rows_pre(hs, acts, hb, f, nb, false, Some(&pre), pool, stats);
            } else {
                let tf = Tick::new(9);
                {
                    let total = nb * f;
                    let up: &[f32] = hb2;
                    blocks::par::for_row_chunks(
                        pool,
                        true,
                        &mut hb[..total],
                        1,
                        total,
                        blocks::par::Sched::StaticAligned(8),
                        |_wk, range, gs| {
                            vmath::swiglu(gs, &up[range]);
                        },
                    );
                    stats.barriers += 1;
                }
                drop(tf);
                let hb_taken = std::mem::take(hb);
                self.rotate_rows(hs, acts, &hb_taken, f, nb, false, pool, stats);
                *hb = hb_taken;
            }
            self.mm(
                w,
                &[(&l.down, Disjoint::new(xb2))],
                &acts[..nb],
                nb,
                pool,
                stats,
                kq,
            );
            for i in 0..nb * d {
                x[i] += xb2[i];
            }
        }

        if *want_hidden {
            hidden.resize(nb * d, 0.0);
            for b in 0..nb {
                let (xv, hv) = (&*x, &mut **hidden);
                blocks::norm::rmsnorm_into(
                    w,
                    &self.out_norm,
                    c.eps,
                    &xv[b * d..(b + 1) * d],
                    &mut hv[b * d..(b + 1) * d],
                );
            }
        }
        // ---- output head ----
        if !want_logits.is_empty() {
            let nl = want_logits.len();
            if nl * c.n_vocab > logits.len() {
                return Err(format!(
                    "logits requested for {nl} rows; scratch holds fewer"
                ));
            }
            for (i, &b) in want_logits.iter().enumerate() {
                if b >= nb {
                    return Err(format!("logits requested for row {b} of a {nb}-row batch"));
                }
                let (xv, xbv) = (&*x, &mut **xb);
                blocks::norm::rmsnorm_into(
                    w,
                    &self.out_norm,
                    c.eps,
                    &xv[b * d..(b + 1) * d],
                    &mut xbv[i * d..(i + 1) * d],
                );
            }
            stats.bytes_weights += self.out_norm.bytes();
            let xb_taken = std::mem::take(xb);
            self.rotate_rows(hs, acts, &xb_taken, d, nl, false, pool, stats);
            *xb = xb_taken;
            let before = stats.ns_matmul;
            self.mm(
                w,
                &[(&self.head, Disjoint::new(logits))],
                &acts[..nl],
                nl,
                pool,
                stats,
                kq,
            );
            stats.ns_head += stats.ns_matmul - before;
            stats.ns_matmul = before;
        }
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn(
        &self,
        c: &Config,
        hs: &mut HyScratch,
        xb: &[f32],
        caches: &mut Caches,
        ab_w: &[f32],
        conv_w: &[f32],
        dt_bias: &[f32],
        a_log: &[f32],
        idx: usize,
        positions: &[usize],
        nb: usize,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        let rc = self.cfg.recurrent();
        let (d, nv) = (c.d_model, rc.n_v_heads);

        {
            let _t = Tick::new(4);
            recurrent::beta_alpha(&mut hs.ab, ab_w, xb, d, nv, nb, pool, stats);
        }

        caches.runs_into(nb, &mut hs.runs);
        recurrent::rec_ptrs(
            &hs.runs,
            |ci| caches.cache_mut(ci).recurrent_ptrs(idx),
            &mut hs.ptrs.rec,
        );
        recurrent::snap_ptrs(
            &hs.runs,
            |ci| caches.cache_mut(ci).snapshot_ptrs(idx),
            &mut hs.ptrs.snap,
        );
        let (rp, spp) = (
            RowPtrs(hs.ptrs.rec.as_ptr()),
            SnapPtrs(hs.ptrs.snap.as_ptr()),
        );

        let tc = Tick::new(5);
        recurrent::causal_conv(
            &mut hs.big,
            hs.big_w,
            conv_w,
            rp,
            spp,
            &rc,
            nb,
            positions,
            pool,
            stats,
        );
        recurrent::silu_norm_gates(
            &mut hs.big,
            hs.big_w,
            &mut hs.ab,
            dt_bias,
            a_log,
            &rc,
            nb,
            nb >= ROW_PARALLEL && pool.width() > 1,
            pool,
            stats,
        );
        drop(tc);

        let td = Tick::new(6);
        recurrent::delta_rule(
            &hs.big,
            hs.big_w,
            &hs.ab,
            &mut hs.o,
            rp,
            spp,
            &rc,
            nb,
            positions,
            matches!(caches, Caches::Shared(_)),
            pool,
            stats,
        );
        drop(td);
    }
}

fn restride(buf: &mut [f32], from: usize, to: usize, nb: usize) {
    if from == to {
        return;
    }
    for b in (0..nb).rev() {
        buf.copy_within(b * from..(b + 1) * from, b * to);
    }
}
