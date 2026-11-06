//! The Gemma 3 model family (ADR 0006).

use std::sync::RwLock;
use std::time::Instant;

use peregrine_core::arch::{Architecture, Family, Loaded};
use peregrine_core::blocks::linear::LinJob;
use peregrine_core::blocks::par::Disjoint;
use peregrine_core::gguf::{Gguf, TensorInfo};
use peregrine_core::model::{Caches, Config, Linear, State, StepStats, TileStore, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;

pub(crate) struct Gemma3Layer {
    pub(crate) attn_norm: TensorInfo,
    pub(crate) wq: Linear,
    pub(crate) wk: Linear,
    pub(crate) wv: Linear,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) wo: Linear,
    pub(crate) attn_post_norm: TensorInfo,
    pub(crate) ffn_norm: TensorInfo,
    pub(crate) gate: Linear,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
    pub(crate) ffn_post_norm: TensorInfo,
}

/// Gemma 3's attention scale, as llama.cpp v0.9.0 sets `f_attention_scale` (`src/models/gemma3.cpp`): the 27B
/// (the only size with 62 layers) uses `1/sqrt(n_embd / n_head)` with an *integer* division
/// (`query_pre_attn_scalar = d_model / n_head`), every other size `1/sqrt(head_dim)`. llama.cpp loads any layer
/// count and takes the second branch for an unknown one; so does this.
pub fn attention_scale(n_layer: usize, d_model: usize, n_head: usize, head_dim: usize) -> f32 {
    if n_layer == 62 {
        1.0 / ((d_model / n_head) as f32).sqrt()
    } else {
        1.0 / (head_dim as f32).sqrt()
    }
}

pub struct Gemma3 {
    pub(crate) tok_embd: TensorInfo,
    pub(crate) out_norm: TensorInfo,
    pub(crate) head: Linear,
    pub(crate) layers: Vec<Gemma3Layer>,
    pub(crate) sliding_window: usize,
    pub(crate) is_swa: Vec<bool>,
    pub(crate) rope_swa: RwLock<Vec<f32>>,
    pub(crate) rope_base_swa: f32,
    pub(crate) final_logit_softcapping: Option<f32>,
    pub(crate) attn_scale: f32,
}

#[cfg(feature = "arch-gemma3")]
pub struct Gemma3Family;

#[cfg(feature = "arch-gemma3")]
impl Family for Gemma3Family {
    fn names(&self) -> &'static [&'static str] {
        &["gemma3"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        Gemma3::load(g, tk)
    }
}

impl Gemma3 {
    fn ensure_rope_swa_len(&self, cfg: &Config, needed_ctx: usize) {
        let half = cfg.rope_dim / 2;
        let needed_floats = needed_ctx * half * 2;
        {
            let current = self.rope_swa.read().unwrap();
            if current.len() >= needed_floats {
                return;
            }
        }
        let mut current = self.rope_swa.write().unwrap();
        if current.len() < needed_floats {
            let target_ctx = needed_ctx.max(current.len() / (half * 2) * 2);
            *current = peregrine_core::blocks::rope::build_rope_table(
                cfg.rope_dim,
                self.rope_base_swa,
                target_ctx,
                None,
            );
        }
    }

    pub fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        let arch_name = g.str("general.architecture")?;
        if arch_name != "gemma3" {
            return Err(format!(
                "expected architecture \"gemma3\", found {arch_name:?}"
            ));
        }
        let k = |suffix: &str| format!("gemma3.{suffix}");
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

        let attn_scale = attention_scale(n_layer, d_model, n_head, head_dim);

        let rope_scaling_type = g.str(&k("rope.scaling.type")).ok();
        let rope_scaling_factor = g.f32(&k("rope.scaling.factor")).ok();
        let mut rope_factors: Option<Vec<f32>> = match (rope_scaling_type, rope_scaling_factor) {
            (Some("none"), _) => None,
            (Some("linear"), Some(factor)) if factor > 0.0 => Some(vec![factor; rope_dim / 2]),
            (None, Some(factor)) if factor > 0.0 => Some(vec![factor; rope_dim / 2]),
            (Some(other), _) => {
                return Err(format!("unsupported Gemma 3 rope scaling type {other:?}"));
            }
            _ => None,
        };
        if let Ok(t) = g.tensor("rope_freqs.weight") {
            let n = t.numel() as usize;
            if n != rope_dim / 2 {
                return Err(format!(
                    "rope_freqs.weight has {n} entries, expected {}",
                    rope_dim / 2
                ));
            }
            let mut f = vec![0.0f32; n];
            peregrine_kernels::quant::dequant_row(t.kind, g.bytes(t), &mut f);
            if let Some(base) = &rope_factors {
                for (x, b) in f.iter_mut().zip(base) {
                    *x *= b;
                }
            }
            rope_factors = Some(f);
        }

        let sliding_window = g.u32(&k("attention.sliding_window")).unwrap_or(4096) as usize;

        // Sliding window pattern:
        // In llama.cpp, if sliding_window is found and > 0, default pattern is 6 (5 local, 1 global).
        // It can also be specified as an explicit bool/u32 array or scalar pattern.
        let is_swa: Vec<bool> = if sliding_window > 0 {
            if let Ok(pattern) = g.u32(&k("attention.sliding_window_pattern")) {
                let p = pattern as usize;
                if p == 0 {
                    vec![true; n_layer]
                } else {
                    (0..n_layer).map(|il| (il % p) < (p - 1)).collect()
                }
            } else {
                // Default pattern 6 in Gemma 3: il % 6 < 5 is sliding-window; il % 6 == 5 is global
                (0..n_layer).map(|il| (il % 6) < 5).collect()
            }
        } else {
            vec![false; n_layer]
        };

        let rope_base = g.f32(&k("rope.freq_base")).unwrap_or(1_000_000.0);
        let rope_base_swa = g.f32(&k("rope.freq_base_swa")).unwrap_or(10_000.0);
        let ctx_train = g.u32(&k("context_length")).unwrap_or(2048) as usize;
        let eps = g
            .f32(&k("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        let final_logit_softcapping = g
            .f32(&k("final_logit_softcapping"))
            .ok()
            .filter(|&c| c > 0.0);

        let cfg = Config {
            arch_name: "gemma3",
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
        let head_norm = |n: &str| -> Result<Vec<f32>, String> {
            let t = get(n)?;
            if t.numel() as usize != head_dim {
                return Err(format!(
                    "{n} has {} entries, expected the head dim {head_dim}",
                    t.numel()
                ));
            }
            let mut v = vec![0.0f32; head_dim];
            peregrine_kernels::quant::dequant_row(t.kind, g.bytes(&t), &mut v);
            Ok(v)
        };

        let get_norm = |name1: &str, name2: &str| -> Result<TensorInfo, String> {
            g.tensor(name1)
                .or_else(|_| g.tensor(name2))
                .cloned()
                .map_err(|e| e.to_string())
        };

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layers.push(Gemma3Layer {
                attn_norm: get(&format!("blk.{i}.attn_norm.weight"))?,
                wq: Linear::Dense(get(&format!("blk.{i}.attn_q.weight"))?),
                wk: Linear::Dense(get(&format!("blk.{i}.attn_k.weight"))?),
                wv: Linear::Dense(get(&format!("blk.{i}.attn_v.weight"))?),
                q_norm: head_norm(&format!("blk.{i}.attn_q_norm.weight"))?,
                k_norm: head_norm(&format!("blk.{i}.attn_k_norm.weight"))?,
                wo: Linear::Dense(get(&format!("blk.{i}.attn_output.weight"))?),
                attn_post_norm: get_norm(
                    &format!("blk.{i}.post_attention_norm.weight"),
                    &format!("blk.{i}.attn_post_norm.weight"),
                )?,
                ffn_norm: get(&format!("blk.{i}.ffn_norm.weight"))?,
                gate: Linear::Dense(get(&format!("blk.{i}.ffn_gate.weight"))?),
                up: Linear::Dense(get(&format!("blk.{i}.ffn_up.weight"))?),
                down: Linear::Dense(get(&format!("blk.{i}.ffn_down.weight"))?),
                ffn_post_norm: get_norm(
                    &format!("blk.{i}.post_ffw_norm.weight"),
                    &format!("blk.{i}.ffn_post_norm.weight"),
                )?,
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
        for l in &layers {
            for (lin, d_in, d_out) in [
                (&l.wq, dm, qd),
                (&l.wk, dm, kd),
                (&l.wv, dm, kd),
                (&l.wo, qd, dm),
                (&l.gate, dm, ff),
                (&l.up, dm, ff),
                (&l.down, ff, dm),
            ] {
                if let Some(t) = lin.dense() {
                    shape(t, d_in, d_out)?;
                }
            }
            shape(&l.attn_norm, dm, 1)?;
            shape(&l.attn_post_norm, dm, 1)?;
            shape(&l.ffn_norm, dm, 1)?;
            shape(&l.ffn_post_norm, dm, 1)?;
        }
        shape(&tok_embd, dm, cfg.n_vocab)?;
        shape(&out_norm, dm, 1)?;
        if let Some(t) = head.dense() {
            shape(t, dm, cfg.n_vocab)?;
        }

        let initial_swa_ctx = ctx_train.max(2048);
        let rope_swa = peregrine_core::blocks::rope::build_rope_table(
            rope_dim,
            rope_base_swa,
            initial_swa_ctx,
            None,
        );

        Ok(Loaded {
            cfg,
            arch: Box::new(Gemma3 {
                tok_embd,
                out_norm,
                head,
                layers,
                sliding_window,
                is_swa,
                rope_swa: RwLock::new(rope_swa),
                rope_base_swa,
                final_logit_softcapping,
                attn_scale,
            }),
            rope_factors,
            tiles: TileStore::default(),
        })
    }
}

impl Architecture for Gemma3 {
    fn session_kv_rotation(&self) -> bool {
        true
    }

    fn linears_mut(
        &mut self,
        f: &mut dyn FnMut(&mut Linear) -> Result<(), String>,
    ) -> Result<(), String> {
        for l in &mut self.layers {
            for lin in [
                &mut l.wq,
                &mut l.wk,
                &mut l.wv,
                &mut l.wo,
                &mut l.gate,
                &mut l.up,
                &mut l.down,
            ] {
                f(lin)?;
            }
        }
        if let Some(ti) = self.head.dense() {
            if ti.name == "output.weight" {
                f(&mut self.head)?;
            }
        }
        Ok(())
    }

    fn linear_groups(&self, f: &mut dyn FnMut(&[&Linear])) {
        for l in &self.layers {
            f(&[&l.wq, &l.wk, &l.wv]);
            f(&[&l.wo]);
            f(&[&l.gate, &l.up]);
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
        let t_fwd = Instant::now();

        // 1. Embedding lookup scaled by sqrt(d_model)
        for (i, &t) in tokens.iter().enumerate() {
            peregrine_core::blocks::embed::embed(
                &self.tok_embd,
                w,
                t,
                &mut st.x[i * cfg.d_model..(i + 1) * cfg.d_model],
                stats,
            );
        }
        let embed_scale = (cfg.d_model as f32).sqrt();
        for x in &mut st.x[..nb * cfg.d_model] {
            *x *= embed_scale;
        }

        // Ensure rope_swa is long enough for all requested positions
        let max_pos = positions.iter().copied().max().unwrap_or(0);
        self.ensure_rope_swa_len(cfg, max_pos + 1);
        let rope_swa_guard = self.rope_swa.read().unwrap();
        let rope_swa: &[f32] = &rope_swa_guard;

        let qd = cfg.q_dim();
        let dm = cfg.d_model;
        let ff = cfg.n_ff;

        for (li, l) in self.layers.iter().enumerate() {
            let is_swa = self.is_swa[li];
            let rope_table = if is_swa { rope_swa } else { &st.rope };
            // KV cache for local layers stays full-length for now (memory, not correctness)
            let window = if is_swa {
                Some(self.sliding_window)
            } else {
                None
            };

            // ---- attention pre-norm ----
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

            // ---- attention QKV projections ----
            {
                let lbx = Disjoint::new(&mut st.lb_x);
                let lbt = Disjoint::new(&mut st.lb_t);
                let acts = &st.acts[..nb];
                let xs: &[f32] = &st.xb;
                let jobs = [
                    LinJob {
                        lin: &l.wq,
                        out: Disjoint::new(&mut st.q),
                    },
                    LinJob {
                        lin: &l.wk,
                        out: Disjoint::new(&mut st.k),
                    },
                    LinJob {
                        lin: &l.wv,
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

            // ---- prepare rows (per-head q/k RMSNorm + RoPE) ----
            peregrine_core::blocks::attn::prepare_rows(
                cfg,
                None,
                None,
                None,
                Some(&l.q_norm),
                Some(&l.k_norm),
                rope_table,
                &mut st.q,
                &mut st.k,
                &mut st.v,
                nb,
                positions,
                pool,
                stats,
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

            // ---- Attend ----
            let t_attn = Instant::now();
            for (r0, len, c) in caches.runs(nb) {
                peregrine_core::blocks::attn::attend_scaled_window(
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
                    window,
                    Some(self.attn_scale),
                    stats,
                );
            }
            stats.ns_attn += t_attn.elapsed().as_nanos() as u64;

            // ---- Output projection wo ----
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

            // ---- Attention post-norm and residual add ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.attn_post_norm,
                dm,
                cfg.eps,
                &st.xb2,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );
            for i in 0..nb * dm {
                st.x[i] += st.xb[i];
            }

            // ---- feed-forward pre-norm ----
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

            // ---- Gate and Up projections ----
            {
                let lbx = Disjoint::new(&mut st.lb_x);
                let lbt = Disjoint::new(&mut st.lb_t);
                let acts = &st.acts[..nb];
                let jobs = [
                    LinJob {
                        lin: &l.gate,
                        out: Disjoint::new(&mut st.hb),
                    },
                    LinJob {
                        lin: &l.up,
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

            // ---- GeGLU activation ----
            peregrine_core::blocks::ffn::geglu_rows(
                ff,
                &mut st.hb,
                &st.hb2,
                &mut st.acts,
                nb,
                pool,
                stats,
            );

            // ---- Down projection ----
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
                    ff,
                    nb,
                    lbx,
                    lbt,
                    pool,
                    stats,
                );
            }

            // ---- Feed-forward post-norm and residual add ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.ffn_post_norm,
                dm,
                cfg.eps,
                &st.xb2,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );
            for i in 0..nb * dm {
                st.x[i] += st.xb[i];
            }

            if let Some(t) = st.tap_layers.iter().position(|&x| x == li) {
                let nt = st.tap_layers.len();
                for b in 0..nb {
                    let dst = (b * nt + t) * dm;
                    st.tap[dst..dst + dm].copy_from_slice(&st.x[b * dm..(b + 1) * dm]);
                }
            }
        }

        // ---- output head ----
        if want_logits.is_empty() {
            stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
            return Ok(());
        }
        let nl = want_logits.len();
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

        if let Some(cap) = self.final_logit_softcapping {
            let inv_cap = 1.0 / cap;
            for logit in &mut st.logits[..nl * cfg.n_vocab] {
                *logit = cap * (*logit * inv_cap).tanh();
            }
        }

        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }
}
