//! The Phi-3 and Phi-4 model family (ADR 0005, ADR 0006).

use std::any::Any;
use std::time::Instant;

use peregrine_core::arch::{ActLayout, Architecture, Family, Loaded};
use peregrine_core::blocks::linear::LinJob;
use peregrine_core::blocks::par::Disjoint;
use peregrine_core::gguf::{Format, Gguf, TensorInfo};
use peregrine_core::model::{Caches, Config, Linear, State, StepStats, TileStore, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;

pub(crate) enum AttnProjections {
    Fused(Linear),
    Separate { wq: Linear, wk: Linear, wv: Linear },
}

pub(crate) enum FfnProjections {
    Fused(Linear),
    Separate { gate: Linear, up: Linear },
}

pub(crate) struct Phi3Layer {
    pub(crate) attn_norm: TensorInfo,
    pub(crate) attn: AttnProjections,
    pub(crate) wo: Linear,
    pub(crate) ffn_norm: TensorInfo,
    pub(crate) ffn: FfnProjections,
    pub(crate) down: Linear,
}

pub struct Phi3 {
    pub(crate) tok_embd: TensorInfo,
    pub(crate) out_norm: TensorInfo,
    pub(crate) head: Linear,
    pub(crate) layers: Vec<Phi3Layer>,
}

#[derive(Default)]
pub struct Phi3Scratch {
    pub qkv: Vec<f32>,
    pub ffn_up: Vec<f32>,
}

impl Phi3Scratch {
    pub fn new(cfg: &Config, max_batch: usize) -> Self {
        let qkv_dim = cfg.q_dim() + 2 * cfg.kv_dim();
        Self {
            qkv: vec![0.0; max_batch * qkv_dim],
            ffn_up: vec![0.0; max_batch * 2 * cfg.n_ff],
        }
    }

    pub fn ensure(&mut self, cfg: &Config, nb: usize) {
        let qkv_dim = cfg.q_dim() + 2 * cfg.kv_dim();
        let needed_qkv = nb * qkv_dim;
        if self.qkv.len() < needed_qkv {
            self.qkv.resize(needed_qkv, 0.0);
        }
        let needed_ffn = nb * 2 * cfg.n_ff;
        if self.ffn_up.len() < needed_ffn {
            self.ffn_up.resize(needed_ffn, 0.0);
        }
    }
}

#[cfg(feature = "arch-phi3")]
pub struct Phi3Family;

#[cfg(feature = "arch-phi3")]
impl Family for Phi3Family {
    fn names(&self) -> &'static [&'static str] {
        &["phi3"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        Phi3::load(g, tk)
    }
}

impl Phi3 {
    pub fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        let arch_name = g.str("general.architecture")?;
        if arch_name != "phi3" {
            return Err(format!(
                "expected architecture \"phi3\", found {arch_name:?}"
            ));
        }

        // LongRoPE 128k factors: refuse explicitly if present
        if g.tensor("rope_factors_long.weight").is_ok()
            || g.tensor("rope_factors_short.weight").is_ok()
            || g.tensor("blk.0.rope_factors_long.weight").is_ok()
            || g.tensor("blk.0.rope_factors_short.weight").is_ok()
        {
            return Err(
                "LongRoPE rope_factors_long/short is not supported (Phi-3 128k)".to_string(),
            );
        }

        let k = |suffix: &str| format!("phi3.{suffix}");
        let n_layer = g.u32(&k("block_count"))? as usize;
        let d_model = g.u32(&k("embedding_length"))? as usize;
        let n_head = g.u32(&k("attention.head_count"))? as usize;
        let n_kv_head = g
            .u32(&k("attention.head_count_kv"))
            .unwrap_or(n_head as u32) as usize;
        let n_ff = g.u32(&k("feed_forward_length"))? as usize;

        if n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            return Err(format!(
                "implausible head configuration: {n_head} query heads over {n_kv_head} kv heads"
            ));
        }

        let head_dim = g
            .u32(&k("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(d_model / n_head);
        let rope_dim = g.u32(&k("rope.dimension_count")).unwrap_or(head_dim as u32) as usize;
        if rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
            return Err(format!(
                "implausible rope dimension {rope_dim} for head dim {head_dim}"
            ));
        }
        for (what, n) in [("embedding length", d_model), ("feed-forward length", n_ff)] {
            if !n.is_multiple_of(32) {
                return Err(format!("{what} {n} is not a multiple of 32"));
            }
        }
        if !(head_dim * n_head).is_multiple_of(32) {
            return Err(format!(
                "query width {} is not a multiple of 32",
                head_dim * n_head
            ));
        }

        // Sliding window: Phi-3-mini 4k has one; Phi-4 has none.
        // In llama.cpp (src/models/phi3.cpp), SWA is explicitly disabled:
        // `if (found_swa && hparams.n_swa > 0) { hparams.swa_type = LLAMA_SWA_TYPE_NONE; hparams.n_swa = 0; }`
        // We follow llama.cpp and do not apply sliding window mask.
        let _ = g.u32(&k("attention.sliding_window")).unwrap_or(0);

        let rope_base = g.f32(&k("rope.freq_base")).unwrap_or(10_000.0);
        let ctx_train = g.u32(&k("context_length")).unwrap_or(2048) as usize;
        let eps = g
            .f32(&k("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        let cfg = Config {
            arch_name: "phi3",
            rope_neox: true,
            n_layer,
            n_kv_layer: n_layer,
            n_head,
            n_kv_head,
            head_dim,
            d_model,
            n_ff,
            n_vocab: g.u32(&k("vocab_size")).unwrap_or(0) as usize,
            rope_dim,
            eps,
            rope_base,
            ctx_train,
        };

        let get = |n: &str| -> Result<TensorInfo, String> { g.tensor(n).cloned() };
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let attn_norm = get(&format!("blk.{i}.attn_norm.weight"))?;
            let attn = if let Ok(t) = get(&format!("blk.{i}.attn_qkv.weight")) {
                AttnProjections::Fused(Linear::Dense(t))
            } else {
                let wq = Linear::Dense(get(&format!("blk.{i}.attn_q.weight"))?);
                let wk = Linear::Dense(get(&format!("blk.{i}.attn_k.weight"))?);
                let wv = Linear::Dense(get(&format!("blk.{i}.attn_v.weight"))?);
                AttnProjections::Separate { wq, wk, wv }
            };
            let wo = Linear::Dense(get(&format!("blk.{i}.attn_output.weight"))?);
            let ffn_norm = get(&format!("blk.{i}.ffn_norm.weight"))?;
            let ffn = if let Ok(t) = get(&format!("blk.{i}.ffn_up.weight")) {
                if t.dims.get(1).copied().unwrap_or(0) as usize == 2 * n_ff {
                    FfnProjections::Fused(Linear::Dense(t))
                } else if let Ok(gate) = get(&format!("blk.{i}.ffn_gate.weight")) {
                    FfnProjections::Separate {
                        gate: Linear::Dense(gate),
                        up: Linear::Dense(t),
                    }
                } else {
                    FfnProjections::Fused(Linear::Dense(t))
                }
            } else if let Ok(gate) = get(&format!("blk.{i}.ffn_gate.weight")) {
                let up = get(&format!("blk.{i}.ffn_up.weight"))?;
                FfnProjections::Separate {
                    gate: Linear::Dense(gate),
                    up: Linear::Dense(up),
                }
            } else {
                return Err(format!("blk.{i}.ffn_up.weight not found"));
            };
            let down = Linear::Dense(get(&format!("blk.{i}.ffn_down.weight"))?);
            layers.push(Phi3Layer {
                attn_norm,
                attn,
                wo,
                ffn_norm,
                ffn,
                down,
            });
        }

        let tok_embd = get("token_embd.weight")?;
        let out_norm = get("output_norm.weight")?;
        let head = Linear::Dense(
            g.tensor("output.weight")
                .ok()
                .cloned()
                .unwrap_or_else(|| tok_embd.clone()),
        );

        let mut cfg = cfg;
        if cfg.n_vocab == 0 {
            cfg.n_vocab = tk.n_vocab;
        }
        if cfg.n_vocab != tk.n_vocab {
            return Err(format!(
                "metadata says {} vocabulary entries but the tokenizer has {}",
                cfg.n_vocab, tk.n_vocab
            ));
        }

        let shape = |t: &TensorInfo, d_in: usize, d_out: usize| -> Result<(), String> {
            let (a, b) = (
                t.dims.first().copied().unwrap_or(0) as usize,
                t.dims.get(1).copied().unwrap_or(1) as usize,
            );
            if t.dims.len() > 2 || a != d_in || b != d_out {
                return Err(format!(
                    "{} is {:?}, the metadata implies [{d_in}, {d_out}]",
                    t.name, t.dims
                ));
            }
            Ok(())
        };

        let (dm, qd, kd, ff) = (cfg.d_model, cfg.q_dim(), cfg.kv_dim(), cfg.n_ff);
        let qkv_d = qd + 2 * kd;
        for l in &layers {
            shape(&l.attn_norm, dm, 1)?;
            match &l.attn {
                AttnProjections::Fused(qkv) => {
                    if let Some(t) = qkv.dense() {
                        shape(t, dm, qkv_d)?;
                    }
                }
                AttnProjections::Separate { wq, wk, wv } => {
                    for (lin, d_out) in [(wq, qd), (wk, kd), (wv, kd)] {
                        if let Some(t) = lin.dense() {
                            shape(t, dm, d_out)?;
                        }
                    }
                }
            }
            if let Some(t) = l.wo.dense() {
                shape(t, qd, dm)?;
            }
            shape(&l.ffn_norm, dm, 1)?;
            match &l.ffn {
                FfnProjections::Fused(up) => {
                    if let Some(t) = up.dense() {
                        shape(t, dm, 2 * ff)?;
                    }
                }
                FfnProjections::Separate { gate, up } => {
                    for lin in [gate, up] {
                        if let Some(t) = lin.dense() {
                            shape(t, dm, ff)?;
                        }
                    }
                }
            }
            if let Some(t) = l.down.dense() {
                shape(t, ff, dm)?;
            }
        }
        shape(&tok_embd, dm, cfg.n_vocab)?;
        shape(&out_norm, dm, 1)?;
        if let Some(t) = head.dense() {
            shape(t, dm, cfg.n_vocab)?;
        }

        Ok(Loaded {
            cfg,
            arch: Box::new(Phi3 {
                tok_embd,
                out_norm,
                head,
                layers,
            }),
            rope_factors: None,
            tiles: TileStore::default(),
        })
    }
}

impl Architecture for Phi3 {
    fn act_layout(&self) -> ActLayout {
        let mut keep_f32 = false;
        self.linear_groups(&mut |lins| {
            for lin in lins {
                if let Some(ti) = lin.dense() {
                    if matches!(ti.kind, Format::F32 | Format::F16 | Format::BF16) {
                        keep_f32 = true;
                    }
                }
            }
        });
        ActLayout {
            keep_f32,
            ..ActLayout::default()
        }
    }

    fn new_scratch(&self, cfg: &Config, max_batch: usize) -> Option<Box<dyn Any + Send + Sync>> {
        Some(Box::new(Phi3Scratch::new(cfg, max_batch)))
    }

    fn linears_mut(
        &mut self,
        f: &mut dyn FnMut(&mut Linear) -> Result<(), String>,
    ) -> Result<(), String> {
        for l in &mut self.layers {
            match &mut l.attn {
                AttnProjections::Fused(qkv) => f(qkv)?,
                AttnProjections::Separate { wq, wk, wv } => {
                    f(wq)?;
                    f(wk)?;
                    f(wv)?;
                }
            }
            f(&mut l.wo)?;
            match &mut l.ffn {
                FfnProjections::Fused(up) => f(up)?,
                FfnProjections::Separate { gate, up } => {
                    f(gate)?;
                    f(up)?;
                }
            }
            f(&mut l.down)?;
        }
        f(&mut self.head)
    }

    fn linear_groups(&self, f: &mut dyn FnMut(&[&Linear])) {
        for l in &self.layers {
            match &l.attn {
                AttnProjections::Fused(qkv) => f(&[qkv]),
                AttnProjections::Separate { wq, wk, wv } => f(&[wq, wk, wv]),
            }
            f(&[&l.wo]);
            match &l.ffn {
                FfnProjections::Fused(up) => f(&[up]),
                FfnProjections::Separate { gate, up } => f(&[gate, up]),
            }
            f(&[&l.down]);
        }
        f(&[&self.head]);
    }

    fn forward_rows(
        &self,
        w: &Weights,
        cfg: &Config,
        st: &mut State,
        caches: &mut Caches<'_>,
        pool: &Pool,
        tokens: &[u32],
        positions: &[usize],
        want_logits: &[usize],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let nb = tokens.len();
        if nb == 0 {
            return Ok(());
        }
        let t_fwd = Instant::now();
        let dm = cfg.d_model;
        let qd = cfg.q_dim();
        let kd = cfg.kv_dim();
        let qkv_dim = qd + 2 * kd;

        if st.fam.is_none() {
            st.fam = Some(Box::new(Phi3Scratch::new(cfg, st.max_batch.max(nb))));
        }
        {
            let scratch = st
                .fam
                .as_mut()
                .and_then(|f| f.downcast_mut::<Phi3Scratch>())
                .ok_or_else(|| "missing Phi3 scratch in state".to_string())?;
            scratch.ensure(cfg, nb);
        }

        for (i, &t) in tokens.iter().enumerate() {
            peregrine_core::blocks::embed::embed(
                &self.tok_embd,
                w,
                t,
                &mut st.x[i * dm..(i + 1) * dm],
                stats,
            );
        }

        for (li, l) in self.layers.iter().enumerate() {
            // ---- attention norm ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.attn_norm,
                dm,
                cfg.eps,
                &st.x,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );

            // ---- attention projections (Q, K, V) ----
            match &l.attn {
                AttnProjections::Fused(qkv) => {
                    let scratch = st
                        .fam
                        .as_mut()
                        .and_then(|f| f.downcast_mut::<Phi3Scratch>())
                        .unwrap();
                    let lbx = Disjoint::new(&mut st.lb_x);
                    let lbt = Disjoint::new(&mut st.lb_t);
                    let acts = &st.acts[..nb];
                    let xs = &st.xb;
                    let jobs = [LinJob {
                        lin: qkv,
                        out: Disjoint::new(&mut scratch.qkv[..nb * qkv_dim]),
                    }];
                    peregrine_core::blocks::linear::linear_group(
                        w,
                        w.lb(),
                        &jobs,
                        acts,
                        xs,
                        dm,
                        nb,
                        lbx,
                        lbt,
                        pool,
                        stats,
                    );
                    let scratch = st
                        .fam
                        .as_ref()
                        .and_then(|f| f.downcast_ref::<Phi3Scratch>())
                        .unwrap();
                    for b in 0..nb {
                        let base = b * qkv_dim;
                        st.q[b * qd..(b + 1) * qd].copy_from_slice(&scratch.qkv[base..base + qd]);
                        st.k[b * kd..(b + 1) * kd]
                            .copy_from_slice(&scratch.qkv[base + qd..base + qd + kd]);
                        st.v[b * kd..(b + 1) * kd]
                            .copy_from_slice(&scratch.qkv[base + qd + kd..base + qd + 2 * kd]);
                    }
                }
                AttnProjections::Separate { wq, wk, wv } => {
                    let lbx = Disjoint::new(&mut st.lb_x);
                    let lbt = Disjoint::new(&mut st.lb_t);
                    let acts = &st.acts[..nb];
                    let xs = &st.xb;
                    let jobs = [
                        LinJob {
                            lin: wq,
                            out: Disjoint::new(&mut st.q),
                        },
                        LinJob {
                            lin: wk,
                            out: Disjoint::new(&mut st.k),
                        },
                        LinJob {
                            lin: wv,
                            out: Disjoint::new(&mut st.v),
                        },
                    ];
                    peregrine_core::blocks::linear::linear_group(
                        w,
                        w.lb(),
                        &jobs,
                        acts,
                        xs,
                        dm,
                        nb,
                        lbx,
                        lbt,
                        pool,
                        stats,
                    );
                }
            }

            // ---- prepare rows (RoPE) ----
            peregrine_core::blocks::attn::prepare_rows(
                cfg, None, None, None, None, None, &st.rope, &mut st.q, &mut st.k, &mut st.v, nb,
                positions, pool, stats,
            );

            // ---- KV cache write ----
            peregrine_core::blocks::attn::write_kv_rows(
                caches,
                li,
                &st.k,
                &st.v,
                nb,
                positions,
                cfg.n_kv_head,
                cfg.head_dim,
            );

            // ---- attention runs ----
            let t_attn = Instant::now();
            for (r0, len, c) in caches.runs(nb) {
                peregrine_core::blocks::attn::attend(
                    cfg,
                    &mut st.attn,
                    &st.q,
                    &mut st.xb,
                    caches.cache(c),
                    pool,
                    li,
                    len,
                    positions[r0],
                    r0,
                    stats,
                );
            }
            stats.ns_attn += t_attn.elapsed().as_nanos() as u64;

            // ---- attention output projection ----
            for b in 0..nb {
                st.acts[b].fill(&st.xb[b * qd..(b + 1) * qd]);
            }
            {
                let lbx = Disjoint::new(&mut st.lb_x);
                let lbt = Disjoint::new(&mut st.lb_t);
                let acts = &st.acts[..nb];
                let jobs = [LinJob {
                    lin: &l.wo,
                    out: Disjoint::new(&mut st.xb2),
                }];
                peregrine_core::blocks::linear::linear_group(
                    w,
                    w.lb(),
                    &jobs,
                    acts,
                    &st.xb,
                    qd,
                    nb,
                    lbx,
                    lbt,
                    pool,
                    stats,
                );
            }
            for i in 0..nb * dm {
                st.x[i] += st.xb2[i];
            }

            // ---- FFN norm ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.ffn_norm,
                dm,
                cfg.eps,
                &st.x,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );

            // ---- FFN up/gate projection ----
            match &l.ffn {
                FfnProjections::Fused(up) => {
                    let scratch = st
                        .fam
                        .as_mut()
                        .and_then(|f| f.downcast_mut::<Phi3Scratch>())
                        .unwrap();
                    let lbx = Disjoint::new(&mut st.lb_x);
                    let lbt = Disjoint::new(&mut st.lb_t);
                    let acts = &st.acts[..nb];
                    let jobs = [LinJob {
                        lin: up,
                        out: Disjoint::new(&mut scratch.ffn_up[..nb * 2 * cfg.n_ff]),
                    }];
                    peregrine_core::blocks::linear::linear_group(
                        w,
                        w.lb(),
                        &jobs,
                        acts,
                        &st.xb,
                        dm,
                        nb,
                        lbx,
                        lbt,
                        pool,
                        stats,
                    );
                    let scratch = st
                        .fam
                        .as_ref()
                        .and_then(|f| f.downcast_ref::<Phi3Scratch>())
                        .unwrap();
                    for b in 0..nb {
                        let base = b * 2 * cfg.n_ff;
                        st.hb[b * cfg.n_ff..(b + 1) * cfg.n_ff]
                            .copy_from_slice(&scratch.ffn_up[base..base + cfg.n_ff]);
                        st.hb2[b * cfg.n_ff..(b + 1) * cfg.n_ff]
                            .copy_from_slice(&scratch.ffn_up[base + cfg.n_ff..base + 2 * cfg.n_ff]);
                    }
                }
                FfnProjections::Separate { gate, up } => {
                    let lbx = Disjoint::new(&mut st.lb_x);
                    let lbt = Disjoint::new(&mut st.lb_t);
                    let acts = &st.acts[..nb];
                    let jobs = [
                        LinJob {
                            lin: gate,
                            out: Disjoint::new(&mut st.hb),
                        },
                        LinJob {
                            lin: up,
                            out: Disjoint::new(&mut st.hb2),
                        },
                    ];
                    peregrine_core::blocks::linear::linear_group(
                        w,
                        w.lb(),
                        &jobs,
                        acts,
                        &st.xb,
                        dm,
                        nb,
                        lbx,
                        lbt,
                        pool,
                        stats,
                    );
                }
            }

            // ---- SwiGLU: hb = silu(hb) ⊙ hb2 ----
            peregrine_core::blocks::ffn::swiglu_rows(
                cfg.n_ff,
                &mut st.hb,
                &st.hb2,
                &mut st.acts,
                nb,
                pool,
                stats,
            );

            // ---- FFN down projection ----
            {
                let lbx = Disjoint::new(&mut st.lb_x);
                let lbt = Disjoint::new(&mut st.lb_t);
                let acts = &st.acts[..nb];
                let jobs = [LinJob {
                    lin: &l.down,
                    out: Disjoint::new(&mut st.xb2),
                }];
                peregrine_core::blocks::linear::linear_group(
                    w,
                    w.lb(),
                    &jobs,
                    acts,
                    &st.hb,
                    cfg.n_ff,
                    nb,
                    lbx,
                    lbt,
                    pool,
                    stats,
                );
            }
            for i in 0..nb * dm {
                st.x[i] += st.xb2[i];
            }

            if let Some(t) = st.tap_layers.iter().position(|&x| x == li) {
                let nt = st.tap_layers.len();
                for b in 0..nb {
                    let (xs, xe) = (b * dm, (b + 1) * dm);
                    let ts = (b * nt + t) * dm;
                    st.tap[ts..ts + dm].copy_from_slice(&st.x[xs..xe]);
                }
            }
        }

        // ---- Output head ----
        let nl = want_logits.len();
        if nl == 0 {
            stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
            return Ok(());
        }
        if nl > st.acts.len() || nl * cfg.n_vocab > st.logits.len() {
            return Err(format!(
                "logits requested for {nl} rows; scratch holds {} activations and {} logit rows",
                st.acts.len(),
                st.logits.len() / cfg.n_vocab
            ));
        }
        for (i, &b) in want_logits.iter().enumerate() {
            if b >= nb {
                return Err(format!("logits requested for row {b} of a {nb}-row batch"));
            }
            let (xs, xe) = (b * dm, (b + 1) * dm);
            let (os, oe) = (i * dm, (i + 1) * dm);
            let (xv, xbv) = (&st.x, &mut st.xb);
            peregrine_core::blocks::norm::rmsnorm(
                w,
                &self.out_norm,
                cfg.eps,
                &xv[xs..xe],
                &mut xbv[os..oe],
                stats,
            );
            st.acts[i].fill(&st.xb[os..oe]);
        }
        let before = stats.ns_matmul;
        {
            let lbx = Disjoint::new(&mut st.lb_x);
            let lbt = Disjoint::new(&mut st.lb_t);
            let acts = &st.acts[..nl];
            let jobs = [LinJob {
                lin: &self.head,
                out: Disjoint::new(&mut st.logits),
            }];
            peregrine_core::blocks::linear::linear_group(
                w,
                w.lb(),
                &jobs,
                acts,
                &st.xb,
                dm,
                nl,
                lbx,
                lbt,
                pool,
                stats,
            );
        }
        stats.ns_head += stats.ns_matmul - before;
        stats.ns_matmul = before;

        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }
}
