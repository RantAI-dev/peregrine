//! The dense transformer families: Llama, Qwen2, Qwen3, Phi-3 and the `mistral3`
//! bidirectional embedder (ADR 0006, roadmap S4.14).

use std::time::Instant;

use peregrine_core::arch::{ActLayout, Architecture, Family, Loaded};
use peregrine_core::blocks::linear::LinJob;
use peregrine_core::blocks::par::Disjoint;
use peregrine_core::gguf::{Format, Gguf, TensorInfo};
use peregrine_core::model::{Caches, Config, Linear, State, StepStats, TileStore, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;

#[cfg(feature = "arch-mistral3")]
use crate::encoder::{linear_rows, linear_rows_shared, RowsOut};
use crate::layer;
use crate::layer::AttnWeights;

/// llama.cpp's defaults for the YaRN ramp (`src/llama-hparams.h`): `yarn_beta_fast` 32 and
/// `yarn_beta_slow` 1.
const YARN_BETA_FAST: f32 = 32.0;
const YARN_BETA_SLOW: f32 = 1.0;

/// The longest text a `mistral3` embedder runs (roadmap S4.14).
///
/// The file says `context_length 262144`, but Nemotron 3.5 Embed 1B is trained for 32768
/// tokens. The engine sizes a bidirectional pass's scratch to its context, so this is the
/// cap: a longer text is refused by the engine's own "does not fit in N positions" error
/// instead of being attempted (or allocating for 262144 rows).
pub const MISTRAL3_MAX_TEXT: usize = 32768;

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
    /// Mistral 3 as an encoder (`mistral3`): NVIDIA Nemotron 3.5 Embed 1B. Query heads
    /// wider than the residual (24 × 128 over 2048), grouped key/value heads, YaRN
    /// rotary scaling, an optional per-row attention temperature, non-causal attention
    /// and mean pooling (roadmap S4.14).
    #[cfg(feature = "arch-mistral3")]
    Mistral3,
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
            #[cfg(feature = "arch-mistral3")]
            "mistral3" => Some(Arch::Mistral3),
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
            #[cfg(feature = "arch-mistral3")]
            Arch::Mistral3 => "mistral3",
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

    pub fn is_mistral3(self) -> bool {
        match self {
            #[cfg(feature = "arch-mistral3")]
            Arch::Mistral3 => true,
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
    /// Mistral 3's attention temperature (`attention.temperature_scale`): the factor
    /// applied per query row is `ln(floor(pos / floor) + 1) * scale + 1`, where `floor`
    /// is `rope.scaling.original_context_length` (llama.cpp's
    /// `llm_graph_input_attn_temp::set_input`, `src/llama-graph.cpp:298-323`).
    #[cfg_attr(not(feature = "arch-mistral3"), allow(dead_code))]
    pub(crate) attn_temp: Option<AttnTemp>,
}

/// The non-`pow` attention temperature of llama.cpp v0.9.0 (`src/llama-graph.cpp:304-312`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnTemp {
    /// The file's `attention.temperature_scale`.
    pub scale: f32,
    /// The divisor: `rope.scaling.original_context_length`.
    pub floor: usize,
}

impl AttnTemp {
    /// The scalar a query row at `pos` is multiplied by.
    ///
    /// `ln` is [`peregrine_core::vmath::det_ln`]: the log is the crate's deterministic one,
    /// so a vector does not depend on the platform's libm. Everything else is one f32
    /// multiply and add, as in llama.cpp. Below `floor` the scalar is exactly 1.0, and
    /// multiplying by 1.0 changes no bits.
    pub fn at(&self, pos: usize) -> f32 {
        peregrine_core::vmath::det_ln((pos / self.floor) as f32 + 1.0) * self.scale + 1.0
    }
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

/// `mistral3` as an embedder: NVIDIA Nemotron 3.5 Embed 1B (roadmap S4.14).
///
/// A `mistral3` file whose `attention.causal` is not `false` is a Mistral 3 *chat*
/// model: it is refused by name, because the causal path is not implemented.
#[cfg(feature = "arch-mistral3")]
pub struct Mistral3Family;

#[cfg(feature = "arch-mistral3")]
impl Family for Mistral3Family {
    fn names(&self) -> &'static [&'static str] {
        &["mistral3"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        Dense::load(g, tk)
    }

    fn embedding_only(&self) -> bool {
        true
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
                "mistral3" => "architecture \"mistral3\" is not enabled in this build (recompile with feature 'arch-mistral3')".to_string(),
                _ => format!(
                    "Peregrine implements {}; this model is {arch_name:?}",
                    crate::ARCHS.join(" and ")
                ),
            }
        })?;
        if arch.is_mistral3() {
            // Only the bidirectionally-trained embedder runs here. A causal file is a
            // Mistral 3 chat model, whose graph this family does not implement.
            if g.bool(&format!("{arch_name}.attention.causal")) != Some(false) {
                return Err(format!(
                    "{arch_name}: this is a causal Mistral 3 model (attention.causal is not \
                     false); Mistral 3 chat models are not supported yet — only the \
                     bidirectional `mistral3` embedders (Nemotron 3.5 Embed) are"
                ));
            }
            // Tensors only the audio/chat variants of the family carry. Refusing them by
            // name is better than running the model without them.
            for (name, what) in [
                ("ffn_norm_ada_down.weight", "an ada RMSNorm"),
                ("ffn_norm_ada_up.weight", "an ada RMSNorm"),
                ("blk.0.ffn_norm_ada_down.weight", "an ada RMSNorm"),
                ("blk.0.ffn_norm_ada_up.weight", "an ada RMSNorm"),
                ("blk.0.attn_q.bias", "attention biases"),
                ("blk.0.attn_k.bias", "attention biases"),
                ("blk.0.attn_v.bias", "attention biases"),
                ("blk.0.attn_output.bias", "attention biases"),
                ("blk.0.ffn_gate.bias", "feed-forward biases"),
                ("blk.0.ffn_up.bias", "feed-forward biases"),
                ("blk.0.ffn_down.bias", "feed-forward biases"),
            ] {
                if g.tensor(name).is_ok() {
                    return Err(format!(
                        "{arch_name}: {what} ({name}) are not implemented for this family"
                    ));
                }
            }
        }
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
        let is_mistral3 = arch.is_mistral3();
        if let Ok(t) = g.str(&k("rope.scaling.type")) {
            if t != "none" && t != "linear" && !(t == "yarn" && is_mistral3) {
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
        if !is_qwen3 && !arch.is_phi3() && !is_mistral3 && head_dim * n_head != d_model {
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

        let rope_base = g.f32(&k("rope.freq_base")).unwrap_or(10000.0);
        // A `mistral3` embedder ships `context_length 262144`, but the model's real limit
        // is 32768 tokens: the engine sizes the pass's whole-text scratch to its context,
        // so the larger number would only be memory. Longer texts are refused by the
        // engine's own "does not fit in N positions" check.
        let ctx_train =
            (g.u32(&k("context_length")).unwrap_or(2048) as usize).min(if is_mistral3 {
                MISTRAL3_MAX_TEXT
            } else {
                usize::MAX
            });
        let rope_yarn = if is_mistral3 && g.str(&k("rope.scaling.type")) == Ok("yarn") {
            let factor = g.f32(&k("rope.scaling.factor")).map_err(|_| {
                "mistral3: the file asks for YaRN rope scaling but carries no \
                 rope.scaling.factor"
                    .to_string()
            })?;
            if factor <= 0.0 || factor.is_nan() {
                return Err(format!("mistral3.rope.scaling.factor = {factor}"));
            }
            let n_ctx_orig = g
                .u32(&k("rope.scaling.original_context_length"))
                .unwrap_or(4096) as usize;
            let ext_factor = g
                .f32(&k("rope.scaling.yarn_ext_factor"))
                .map(|v| if v < 0.0 { 1.0 } else { v })
                .unwrap_or(1.0);
            let attn_factor = g.f32(&k("rope.scaling.attn_factor")).unwrap_or(1.0);
            let log_mul = g.f32(&k("rope.scaling.yarn_log_multiplier")).unwrap_or(0.0);
            let beta_fast = g
                .f32(&k("rope.scaling.yarn_beta_fast"))
                .unwrap_or(YARN_BETA_FAST);
            let beta_slow = g
                .f32(&k("rope.scaling.yarn_beta_slow"))
                .unwrap_or(YARN_BETA_SLOW);
            Some(peregrine_core::blocks::rope::yarn_params(
                rope_dim,
                rope_base,
                factor,
                n_ctx_orig,
                ext_factor,
                attn_factor,
                log_mul,
                beta_fast,
                beta_slow,
            ))
        } else {
            None
        };
        let rope_scale = if rope_yarn.is_some() {
            None
        } else {
            g.f32(&k("rope.scaling.factor"))
                .ok()
                .filter(|f| *f > 0.0)
                .map(|f| 1.0 / f)
        };
        let attn_temp = if is_mistral3 {
            let scale = g.f32(&k("attention.temperature_scale")).unwrap_or(0.0);
            if scale == 0.0 {
                None
            } else {
                // llama.cpp takes the divisor from `n_ctx_orig_yarn` and refuses a zero
                // one (`src/models/mistral3.cpp:13-21`).
                let floor = g
                    .u32(&k("rope.scaling.original_context_length"))
                    .unwrap_or(0) as usize;
                if floor == 0 {
                    return Err("mistral3: attention.temperature_scale needs a non-zero \
                         rope.scaling.original_context_length"
                        .to_string());
                }
                Some(AttnTemp { scale, floor })
            }
        } else {
            None
        };
        let cfg = Config {
            arch_name: arch.name(),
            rope_neox: !arch.is_llama() && !is_mistral3,
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
            rope_base,
            ctx_train,
            rope_yarn,
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
                attn_temp,
            }),
            rope_factors,
            tiles: TileStore::default(),
        })
    }
}

/// `mistral3`'s bidirectional pass (roadmap S4.14).
#[cfg(feature = "arch-mistral3")]
impl Dense {
    /// The transcription of llama.cpp v0.9.0's `mistral3` graph
    /// (`src/models/mistral3.cpp:139-269`) with non-causal attention, as NVIDIA's
    /// Nemotron 3.5 Embed 1B file asks for it (`attention.causal = false`):
    ///
    /// - the token embedding is read as it is (no `sqrt(n_embd)` scale — llama.cpp's
    ///   `build_inp_embd` applies none here);
    /// - every layer is `attn_norm` → q/k/v → RoPE (the YaRN table) → the attention
    ///   temperature (`Q` per row multiplied by `ln(floor(pos / n_ctx_orig) + 1)·0.1 + 1`)
    ///   → non-causal attention with `kq_scale = 1/sqrt(head_dim)` → `attn_output` →
    ///   residual → `ffn_norm` → gate/up → SwiGLU → down → residual;
    /// - `output_norm` is `result_norm`, which is pooled per text (the file's
    ///   `pooling_type = 1`, mean), and the caller truncates and normalises.
    ///
    /// There is no mask and no sliding window: every row of a text sees every row of its
    /// own text. Keys and values stay f32 in the pass; no cache is written.
    #[allow(clippy::too_many_arguments)]
    fn forward_texts_embedding(
        &self,
        w: &Weights,
        cfg: &Config,
        st: &mut State,
        pool: &Pool,
        texts: &[&[u32]],
        wants: &[peregrine_core::arch::TextWant],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let t_fwd = Instant::now();
        let dm = cfg.d_model;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let ff = cfg.n_ff;
        let nb: usize = texts.iter().map(|t| t.len()).sum();

        // The batch's positions: every text starts at 0. Both buffers are taken out of
        // the state (and put back at the end) so the loop can borrow the rest of `st`
        // mutably: a bidirectional pass must not allocate.
        let mut pos_buf = std::mem::take(&mut st.positions);
        let mut spans = std::mem::take(&mut st.spans);
        if pos_buf.len() < nb {
            pos_buf.resize(nb, 0);
        }
        spans.clear();
        let mut row = 0usize;
        for t in texts {
            spans.push(peregrine_core::blocks::bidir::TextSpan {
                row0: row,
                len: t.len(),
            });
            for i in 0..t.len() {
                pos_buf[row] = i;
                row += 1;
            }
        }
        let positions = &pos_buf[..nb];

        // The embedding lookup. `mistral3` scales nothing here.
        let mut row = 0usize;
        for t in texts {
            for &tok in t.iter() {
                peregrine_core::blocks::embed::embed(
                    &self.tok_embd,
                    w,
                    tok,
                    &mut st.x[row * dm..(row + 1) * dm],
                    stats,
                );
                row += 1;
            }
        }

        // llama.cpp's `kq_scale` for this family: `f_attention_scale` is 0, so
        // `1/sqrt(head_dim)` (`src/models/mistral3.cpp:167`).
        let kq_scale = 1.0 / (cfg.head_dim as f32).sqrt();

        for l in self.layers.iter() {
            // ---- attention pre-norm ----
            peregrine_core::prof::time(peregrine_core::prof::Slot::Norm, || {
                peregrine_core::blocks::norm::norm_rows(
                    w,
                    &l.attn.attn_norm,
                    dm,
                    cfg.eps,
                    &st.x,
                    &mut st.xb,
                    &mut st.acts,
                    nb,
                    pool,
                    stats,
                )
            });

            // ---- QKV projections: one quantisation and one fork-join for the three
            // matrices, which read the same normed rows (S4.9a) ----
            {
                let mut jobs = [
                    RowsOut {
                        lin: &l.attn.wq,
                        out: Disjoint::new(&mut st.q),
                        out_dim: qd,
                    },
                    RowsOut {
                        lin: &l.attn.wk,
                        out: Disjoint::new(&mut st.k),
                        out_dim: kvd,
                    },
                    RowsOut {
                        lin: &l.attn.wv,
                        out: Disjoint::new(&mut st.v),
                        out_dim: kvd,
                    },
                ];
                peregrine_core::prof::time(peregrine_core::prof::Slot::Qkv, || {
                    linear_rows_shared(
                        w,
                        pool,
                        stats,
                        &mut jobs,
                        &st.acts,
                        &mut st.lb_x,
                        &mut st.lb_t,
                        &st.xb,
                        dm,
                        nb,
                        &mut st.kq,
                    )
                });
            }

            // ---- RoPE on Q and K. There is no per-head norm and no bias ----
            peregrine_core::prof::time(peregrine_core::prof::Slot::Rope, || {
                peregrine_core::blocks::attn::prepare_rows(
                    cfg, None, None, None, None, None, &st.rope, &mut st.q, &mut st.k, &mut st.v,
                    nb, positions, pool, stats,
                )
            });

            // ---- the attention temperature, per query row ----
            if let Some(t) = &self.attn_temp {
                let parallel = nb >= peregrine_core::blocks::ROW_PARALLEL && pool.width() > 1;
                peregrine_core::blocks::par::for_row_chunks(
                    pool,
                    parallel,
                    &mut st.q,
                    qd,
                    nb,
                    peregrine_core::blocks::par::Sched::Static,
                    |_wk, rows, q| {
                        for (i, r) in rows.enumerate() {
                            if let Some(&pos) = positions.get(r) {
                                let s = t.at(pos);
                                for x in &mut q[i * qd..(i + 1) * qd] {
                                    *x *= s;
                                }
                            }
                        }
                    },
                );
                if parallel {
                    stats.barriers += 1;
                }
            }

            // ---- attention: one n × n block per text, no cache ----
            let t_attn = Instant::now();
            peregrine_core::prof::time(peregrine_core::prof::Slot::Attn, || {
                peregrine_core::blocks::bidir::attend_texts(
                    cfg,
                    &st.q,
                    &st.k,
                    &st.v,
                    &mut st.xb,
                    &spans,
                    None,
                    None,
                    kq_scale,
                    &mut st.attn.bidir,
                    pool,
                    stats,
                )
            });
            stats.ns_attn += t_attn.elapsed().as_nanos() as u64;

            // ---- output projection ----
            peregrine_core::prof::time(peregrine_core::prof::Slot::Wo, || {
                linear_rows(
                    w,
                    pool,
                    stats,
                    &l.attn.wo,
                    &mut st.acts,
                    &mut st.lb_x,
                    &mut st.lb_t,
                    &st.xb,
                    qd,
                    nb,
                    &mut st.xb2,
                    dm,
                    &mut st.kq,
                )
            });
            for i in 0..nb * dm {
                st.x[i] += st.xb2[i];
            }

            // ---- feed-forward ----
            peregrine_core::prof::time(peregrine_core::prof::Slot::Norm, || {
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
                )
            });
            // Gate and up read the same normed rows: one quantisation, one fork-join.
            {
                let mut jobs = [
                    RowsOut {
                        lin: &l.gate,
                        out: Disjoint::new(&mut st.hb),
                        out_dim: ff,
                    },
                    RowsOut {
                        lin: &l.up,
                        out: Disjoint::new(&mut st.hb2),
                        out_dim: ff,
                    },
                ];
                peregrine_core::prof::time(peregrine_core::prof::Slot::GateUp, || {
                    linear_rows_shared(
                        w,
                        pool,
                        stats,
                        &mut jobs,
                        &st.acts,
                        &mut st.lb_x,
                        &mut st.lb_t,
                        &st.xb,
                        dm,
                        nb,
                        &mut st.kq,
                    )
                });
            }
            peregrine_core::prof::time(peregrine_core::prof::Slot::Swiglu, || {
                peregrine_core::blocks::ffn::swiglu_rows(
                    ff,
                    &mut st.hb,
                    &st.hb2,
                    &mut st.acts,
                    nb,
                    pool,
                    stats,
                )
            });
            // The activation of `swiglu_rows` is already quantised: the down projection
            // reads it as it is instead of filling the same rows again (S4.9a).
            {
                let mut jobs = [RowsOut {
                    lin: &l.down,
                    out: Disjoint::new(&mut st.xb2),
                    out_dim: dm,
                }];
                peregrine_core::prof::time(peregrine_core::prof::Slot::Down, || {
                    linear_rows_shared(
                        w,
                        pool,
                        stats,
                        &mut jobs,
                        &st.acts,
                        &mut st.lb_x,
                        &mut st.lb_t,
                        &st.hb,
                        ff,
                        nb,
                        &mut st.kq,
                    )
                });
            }
            for i in 0..nb * dm {
                st.x[i] += st.xb2[i];
            }
        }

        // ---- output_norm: llama.cpp's `result_norm`, the rows pooling reads ----
        let saved_hidden = st.want_hidden;
        st.want_hidden = true;
        layer::write_hidden(w, cfg, st, &self.out_norm, nb);
        st.want_hidden = saved_hidden;

        // ---- pooling, text by text ----
        if st.pool_sum.len() < dm {
            st.pool_sum.resize(dm, 0.0);
        }
        for (t, text) in texts.iter().enumerate() {
            let row0 = spans[t].row0;
            let n = text.len();
            let pooling = match wants[t] {
                peregrine_core::arch::TextWant::Pooled(p) => p,
                peregrine_core::arch::TextWant::Head => {
                    return Err(format!("{} has no classifier head", cfg.arch_name))
                }
            };
            peregrine_core::embed::pool_rows_into(
                &st.hidden[row0 * dm..(row0 + n) * dm],
                dm,
                pooling,
                &mut st.pool_sum,
                &mut st.xb2[..dm],
            );
            // The vector this text's slot reads: the first row of the text, so the
            // engine needs no buffer of its own.
            st.hidden[row0 * dm..row0 * dm + dm].copy_from_slice(&st.xb2[..dm]);
        }

        // Put the caller's buffers back with the capacity they had.
        st.positions = pos_buf;
        st.spans = spans;
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
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
        let mut quantize = false;
        self.linear_groups(&mut |lins| {
            for lin in lins {
                match lin.dense() {
                    Some(ti) => {
                        if matches!(ti.kind, Format::F32 | Format::F16 | Format::BF16) {
                            keep_f32 = true;
                        } else {
                            quantize = true;
                        }
                    }
                    None => quantize = true,
                }
            }
        });
        ActLayout {
            keep_f32,
            quantize,
            ..ActLayout::default()
        }
    }

    fn causality(&self) -> peregrine_core::arch::Causality {
        if self.arch.is_mistral3() {
            peregrine_core::arch::Causality::Bidirectional
        } else {
            peregrine_core::arch::Causality::Causal
        }
    }

    fn forward_texts(
        &self,
        _w: &Weights,
        _cfg: &Config,
        _st: &mut State,
        _pool: &Pool,
        _texts: &[&[u32]],
        _wants: &[peregrine_core::arch::TextWant],
        _stats: &mut StepStats,
    ) -> Result<(), String> {
        #[cfg(feature = "arch-mistral3")]
        if self.arch.is_mistral3() {
            return self.forward_texts_embedding(_w, _cfg, _st, _pool, _texts, _wants, _stats);
        }
        Err(format!(
            "{} has no bidirectional forward pass",
            _cfg.arch_name
        ))
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
                peregrine_core::prof::time(peregrine_core::prof::Slot::GateUp, || {
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
                        &mut st.kq,
                    )
                });
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
                peregrine_core::prof::time(peregrine_core::prof::Slot::Down, || {
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
                        &mut st.kq,
                    )
                });
            }
            {
                let xb2: &[f32] = &st.xb2;
                let parallel = nb >= peregrine_core::blocks::ROW_PARALLEL && pool.width() > 1;
                peregrine_core::prof::time(peregrine_core::prof::Slot::Resid, || {
                    peregrine_core::blocks::par::for_row_chunks(
                        pool,
                        parallel,
                        &mut st.x,
                        cfg.d_model,
                        nb,
                        peregrine_core::blocks::par::Sched::Static,
                        |_wk, rows, xc| {
                            let base = rows.start * cfg.d_model;
                            for (i, x) in xc.iter_mut().enumerate() {
                                *x += xb2[base + i];
                            }
                        },
                    );
                });
                if parallel {
                    stats.barriers += 1;
                }
            }
            layer::copy_tap(st, cfg, li, nb);
        }

        // ---- output head ----
        layer::write_hidden(w, cfg, st, &self.out_norm, nb);
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
