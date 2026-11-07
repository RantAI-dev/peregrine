//! The dense transformer families: Llama, Qwen2 and Qwen3 (ADR 0006).

pub(crate) mod layer;

use std::time::Instant;

use peregrine_core::arch::{ActLayout, Architecture, Family, Loaded};
use peregrine_core::blocks::linear::LinJob;
use peregrine_core::blocks::par::Disjoint;
use peregrine_core::gguf::{Format, Gguf, TensorInfo};
use peregrine_core::model::{Caches, Config, Linear, State, StepStats, TileStore, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;

use layer::AttnWeights;

/// The dense transformer architectures implemented by Peregrine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    #[cfg(feature = "arch-llama")]
    Llama,
    #[cfg(feature = "arch-qwen")]
    Qwen2,
    /// Qwen3: no attention biases, a per-head RMSNorm on queries and keys before
    /// rotation, and a head width independent of the model width (Qwen3-4B runs 32
    /// heads of 128 over a 2560-wide residual).
    #[cfg(feature = "arch-qwen")]
    Qwen3,
    #[cfg(feature = "arch-phi3")]
    Phi3,
}

impl Arch {
    pub fn from_name(name: &str) -> Option<Arch> {
        match name {
            #[cfg(feature = "arch-llama")]
            "llama" => Some(Arch::Llama),
            #[cfg(feature = "arch-qwen")]
            "qwen2" => Some(Arch::Qwen2),
            #[cfg(feature = "arch-qwen")]
            "qwen3" => Some(Arch::Qwen3),
            #[cfg(feature = "arch-phi3")]
            "phi3" => Some(Arch::Phi3),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "arch-llama")]
            Arch::Llama => "llama",
            #[cfg(feature = "arch-qwen")]
            Arch::Qwen2 => "qwen2",
            #[cfg(feature = "arch-qwen")]
            Arch::Qwen3 => "qwen3",
            #[cfg(feature = "arch-phi3")]
            Arch::Phi3 => "phi3",
        }
    }

    pub fn key_prefix(self) -> &'static str {
        self.name()
    }

    pub fn is_llama(self) -> bool {
        match self {
            #[cfg(feature = "arch-llama")]
            Arch::Llama => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    pub fn is_qwen2(self) -> bool {
        match self {
            #[cfg(feature = "arch-qwen")]
            Arch::Qwen2 => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    pub fn is_qwen3(self) -> bool {
        match self {
            #[cfg(feature = "arch-qwen")]
            Arch::Qwen3 => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    pub fn is_phi3(self) -> bool {
        match self {
            #[cfg(feature = "arch-phi3")]
            Arch::Phi3 => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

pub(crate) struct DenseLayer {
    pub(crate) attn: AttnWeights,
    pub(crate) ffn_norm: TensorInfo,
    pub(crate) gate: Linear,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
}

pub struct Dense {
    pub(crate) tok_embd: TensorInfo,
    pub(crate) out_norm: TensorInfo,
    pub(crate) head: Linear,
    pub(crate) layers: Vec<DenseLayer>,
    pub(crate) arch: Arch,
}

#[cfg(feature = "arch-llama")]
pub struct LlamaFamily;

#[cfg(feature = "arch-llama")]
impl Family for LlamaFamily {
    fn names(&self) -> &'static [&'static str] {
        &["llama"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        Dense::load(g, tk)
    }
}

#[cfg(feature = "arch-qwen")]
pub struct QwenFamily;

#[cfg(feature = "arch-qwen")]
impl Family for QwenFamily {
    fn names(&self) -> &'static [&'static str] {
        &["qwen2", "qwen3"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        Dense::load(g, tk)
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
        Dense::load(g, tk)
    }
}

pub struct DenseFamily;

impl Family for DenseFamily {
    fn names(&self) -> &'static [&'static str] {
        &[
            #[cfg(feature = "arch-llama")]
            "llama",
            #[cfg(feature = "arch-qwen")]
            "qwen2",
            #[cfg(feature = "arch-qwen")]
            "qwen3",
            #[cfg(feature = "arch-phi3")]
            "phi3",
        ]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        Dense::load(g, tk)
    }
}

impl Dense {
    pub fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        let arch_name = g.str("general.architecture")?;
        let arch = Arch::from_name(arch_name).ok_or_else(|| {
            match arch_name {
                "llama" => "architecture \"llama\" is not enabled in this build (recompile with feature 'arch-llama')".to_string(),
                "qwen2" | "qwen3" => format!("architecture {arch_name:?} is not enabled in this build (recompile with feature 'arch-qwen')"),
                "phi3" => "architecture \"phi3\" is not enabled in this build (recompile with feature 'arch-phi3')".to_string(),
                _ => format!(
                    "Peregrine implements {}; this model is {arch_name:?}",
                    crate::ARCHS.join(" and ")
                ),
            }
        })?;
        if arch.is_phi3() {
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
        }
        let k = |suffix: &str| format!("{}.{suffix}", arch.key_prefix());
        let n_layer = g.u32(&k("block_count"))? as usize;
        let d_model = g.u32(&k("embedding_length"))? as usize;
        let n_head = g.u32(&k("attention.head_count"))? as usize;
        let n_kv_head = g
            .u32(&k("attention.head_count_kv"))
            .unwrap_or(n_head as u32) as usize;
        let n_ff = g.u32(&k("feed_forward_length"))? as usize;
        if let Ok(t) = g.str(&k("rope.scaling.type")) {
            if t != "none" && t != "linear" {
                return Err(format!("rope scaling {t:?} is not implemented"));
            }
        }
        if n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            return Err(format!(
                "implausible head configuration: {n_head} query heads over {n_kv_head} kv heads"
            ));
        }
        let head_dim = g
            .u32(&k("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(d_model / n_head);
        let is_qwen3 = arch.is_qwen3();
        if !is_qwen3 && !arch.is_phi3() && head_dim * n_head != d_model {
            return Err(format!(
                "head dim {head_dim} × {n_head} heads ≠ embedding length {d_model}"
            ));
        }
        if !(head_dim * n_head).is_multiple_of(32) {
            return Err(format!(
                "query width {} is not a multiple of 32",
                head_dim * n_head
            ));
        }
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

        // Sliding window: Phi-3-mini 4k has one; Phi-4 has none.
        // In llama.cpp (src/models/phi3.cpp), SWA is explicitly disabled:
        // `if (found_swa && hparams.n_swa > 0) { hparams.swa_type = LLAMA_SWA_TYPE_NONE; hparams.n_swa = 0; }`
        // We follow llama.cpp and do not apply sliding window mask.
        let _ = g.u32(&k("attention.sliding_window")).unwrap_or(0);

        let rope_scale = g
            .f32(&k("rope.scaling.factor"))
            .ok()
            .filter(|f| *f > 0.0)
            .map(|f| 1.0 / f);
        let cfg = Config {
            arch_name: arch.name(),
            rope_neox: !arch.is_llama(),
            n_layer,
            n_kv_layer: n_layer,
            n_head,
            n_kv_head,
            head_dim,
            d_model,
            n_ff,
            n_vocab: g.u32(&k("vocab_size")).unwrap_or(0) as usize,
            rope_dim,
            eps: g
                .f32(&k("attention.layer_norm_rms_epsilon"))
                .unwrap_or(if arch.is_phi3() { 1e-6 } else { 1e-5 }),
            rope_base: g.f32(&k("rope.freq_base")).unwrap_or(10000.0),
            ctx_train: g.u32(&k("context_length")).unwrap_or(2048) as usize,
        };
        let mut rope_factors: Option<Vec<f32>> = rope_scale.map(|s| vec![1.0 / s; rope_dim / 2]);
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

        let get = |n: &str| -> Result<TensorInfo, String> { g.tensor(n).cloned() };
        let bias = |n: &str| -> Result<Option<Vec<f32>>, String> {
            if arch.is_qwen2() {
                let t = get(n)?;
                let mut v = vec![0.0f32; t.numel() as usize];
                peregrine_kernels::quant::dequant_row(t.kind, g.bytes(&t), &mut v);
                Ok(Some(v))
            } else {
                Ok(None)
            }
        };
        let head_norm = |n: &str| -> Result<Option<Vec<f32>>, String> {
            if !arch.is_qwen3() {
                return Ok(None);
            }
            let t = get(n)?;
            if t.numel() as usize != head_dim {
                return Err(format!(
                    "{n} has {} entries, expected the head dim {head_dim}",
                    t.numel()
                ));
            }
            let mut v = vec![0.0f32; head_dim];
            peregrine_kernels::quant::dequant_row(t.kind, g.bytes(&t), &mut v);
            Ok(Some(v))
        };
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let (wq, wk, wv) = if let Ok(wq_t) = get(&format!("blk.{i}.attn_q.weight")) {
                let wk_t = get(&format!("blk.{i}.attn_k.weight"))?;
                let wv_t = get(&format!("blk.{i}.attn_v.weight"))?;
                (
                    Linear::Dense(wq_t),
                    Linear::Dense(wk_t),
                    Linear::Dense(wv_t),
                )
            } else if let Ok(qkv) = get(&format!("blk.{i}.attn_qkv.weight")) {
                let qd = cfg.q_dim();
                let kd = cfg.kv_dim();
                let wq_ti = qkv.rows(0, qd)?;
                let wk_ti = qkv.rows(qd, kd)?;
                let wv_ti = qkv.rows(qd + kd, kd)?;
                (
                    Linear::Dense(wq_ti),
                    Linear::Dense(wk_ti),
                    Linear::Dense(wv_ti),
                )
            } else {
                return Err(format!(
                    "blk.{i}.attn_q.weight or blk.{i}.attn_qkv.weight not found"
                ));
            };
            let (gate, up) = if let Ok(gate_t) = get(&format!("blk.{i}.ffn_gate.weight")) {
                let up_t = get(&format!("blk.{i}.ffn_up.weight"))?;
                (Linear::Dense(gate_t), Linear::Dense(up_t))
            } else {
                let ffn_up = get(&format!("blk.{i}.ffn_up.weight"))?;
                let gate_ti = ffn_up.rows(0, n_ff)?;
                let up_ti = ffn_up.rows(n_ff, n_ff)?;
                (Linear::Dense(gate_ti), Linear::Dense(up_ti))
            };
            layers.push(DenseLayer {
                attn: AttnWeights {
                    attn_norm: get(&format!("blk.{i}.attn_norm.weight"))?,
                    wq,
                    wk,
                    wv,
                    bq: bias(&format!("blk.{i}.attn_q.bias"))?,
                    bk: bias(&format!("blk.{i}.attn_k.bias"))?,
                    bv: bias(&format!("blk.{i}.attn_v.bias"))?,
                    q_norm: head_norm(&format!("blk.{i}.attn_q_norm.weight"))?,
                    k_norm: head_norm(&format!("blk.{i}.attn_k_norm.weight"))?,
                    wo: Linear::Dense(get(&format!("blk.{i}.attn_output.weight"))?),
                },
                ffn_norm: get(&format!("blk.{i}.ffn_norm.weight"))?,
                gate,
                up,
                down: Linear::Dense(get(&format!("blk.{i}.ffn_down.weight"))?),
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
                (&l.attn.wq, dm, qd),
                (&l.attn.wk, dm, kd),
                (&l.attn.wv, dm, kd),
                (&l.attn.wo, qd, dm),
                (&l.gate, dm, ff),
                (&l.up, dm, ff),
                (&l.down, ff, dm),
            ] {
                if let Some(t) = lin.dense() {
                    shape(t, d_in, d_out)?;
                }
            }
            shape(&l.attn.attn_norm, dm, 1)?;
            shape(&l.ffn_norm, dm, 1)?;
        }
        shape(&tok_embd, dm, cfg.n_vocab)?;
        shape(&out_norm, dm, 1)?;
        if let Some(t) = head.dense() {
            shape(t, dm, cfg.n_vocab)?;
        }

        Ok(Loaded {
            cfg,
            arch: Box::new(Dense {
                tok_embd,
                out_norm,
                head,
                layers,
                arch,
            }),
            rope_factors,
            tiles: TileStore::default(),
        })
    }
}

impl Architecture for Dense {
    fn session_kv_rotation(&self) -> bool {
        !self.arch.is_qwen2()
    }

    fn linears_mut(
        &mut self,
        f: &mut dyn FnMut(&mut Linear) -> Result<(), String>,
    ) -> Result<(), String> {
        for l in &mut self.layers {
            for lin in [
                &mut l.attn.wq,
                &mut l.attn.wk,
                &mut l.attn.wv,
                &mut l.attn.wo,
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
            f(&[&l.attn.wq, &l.attn.wk, &l.attn.wv]);
            f(&[&l.attn.wo]);
            f(&[&l.gate, &l.up]);
            f(&[&l.down]);
        }
        f(&[&self.head]);
    }

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
        for (i, &t) in tokens.iter().enumerate() {
            peregrine_core::blocks::embed::embed(
                &self.tok_embd,
                w,
                t,
                &mut st.x[i * cfg.d_model..(i + 1) * cfg.d_model],
                stats,
            );
        }

        for (li, l) in self.layers.iter().enumerate() {
            // ---- attention ----
            layer::attention_rows(w, cfg, st, caches, pool, &l.attn, li, nb, positions, stats);

            // ---- feed-forward ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.ffn_norm,
                cfg.d_model,
                cfg.eps,
                &st.x,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );
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
                    cfg.d_model,
                    nb,
                    lbx,
                    lbt,
                    pool,
                    stats,
                );
            }
            peregrine_core::blocks::ffn::swiglu_rows(
                cfg.n_ff,
                &mut st.hb,
                &st.hb2,
                &mut st.acts,
                nb,
                pool,
                stats,
            );
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
            for i in 0..nb * cfg.d_model {
                st.x[i] += st.xb2[i];
            }
            layer::copy_tap(st, cfg, li, nb);
        }

        // ---- output head ----
        if !want_logits.is_empty() {
            layer::head_rows(
                w,
                cfg,
                st,
                &self.out_norm,
                &self.head,
                nb,
                want_logits,
                pool,
                stats,
            )?;
        }
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }
}
