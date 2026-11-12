//! The `modern-bert` family: the pre-norm encoder with alternating local and
//! global attention (roadmap S4.7).
//!
//! `architecture = "modern-bert"` is the 2024 encoder ModernBERT-base and -large,
//! gte-modernbert-base, IBM Granite Embedding R2 and mmBERT ship as (mmBERT's own
//! vocabulary needs the `mmbert` metaspace pre-tokenizer, which is refused by name
//! here). It differs
//! from [`crate::bert`] in four places: the blocks are **pre-norm** (two LayerNorms
//! with a weight and no bias, and the first block's attention norm is the
//! identity), attention is **RoPE** with one base for the local layers and one for
//! the global ones, every third layer from the first is **global** and the others
//! see a symmetric sliding window (`attention.sliding_window / 2` either side),
//! and the feed-forward is a **gated pair**: `ffn_up` is `2·n_ff` wide, the first
//! half is activated and multiplies the second.
//!
//! Every detail here is transcribed from llama.cpp **v0.9.0 (`cd959ba`)**:
//! `src/models/modern-bert.cpp` (the graph, the tensor list, the window pattern),
//! `src/llama-model.cpp` (the rope type, the SWA pattern, `f_norm_eps`),
//! `src/llama-hparams.h` (`is_masked_swa`) and `conversion/bert.py` (the
//! `modern-bert` class and its vocabulary). [`docs/reference/modern-bert.md`]
//! holds the reading, the tensor table and the measured accuracy.
//!
//! [`docs/reference/modern-bert.md`]: ../../../../docs/reference/modern-bert.md

use std::time::Instant;

use peregrine_core::arch::{ActLayout, Architecture, Causality, Family, Loaded};
use peregrine_core::blocks::par::{for_row_chunks_2, for_row_chunks_3, Sched};
use peregrine_core::blocks::ROW_PARALLEL;
use peregrine_core::gguf::{Gguf, TensorInfo};
use peregrine_core::model::{Config, Linear, State, StepStats, TileStore, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;
use peregrine_core::vmath;

use crate::encoder::linear_rows;

/// The family, registered as `modern-bert`.
pub struct ModernBertFamily;

impl Family for ModernBertFamily {
    fn names(&self) -> &'static [&'static str] {
        &["modern-bert"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        ModernBert::load(g, tk)
    }

    /// A `modern-bert` file is an encoder: it has no decode path, so the engine
    /// refuses chat and label requests on it by name (design §3).
    fn embedding_only(&self) -> bool {
        true
    }
}

/// The feed-forward's gated activation (`llm_ffn_op_type` in llama.cpp).
///
/// `modern-bert.hidden_activation` picks it. The converter writes `gelu` for
/// ModernBERT itself (the exact, erf GELU) and for Granite Embedding R2; some Granite
/// derivatives say `silu`. The `gelu_new`-style tanh names map to `GeGLU` as they do
/// in llama.cpp's `LLM_FFN_OP_TYPES_FROM_STRING`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GatedAct {
    /// `LLM_FFN_GEGLU_ERF`: the first half through the exact GELU.
    GegluErf,
    /// `LLM_FFN_GEGLU`: the first half through the tanh approximation.
    Geglu,
    /// `LLM_FFN_SWIGLU`: the first half through SiLU.
    Swiglu,
}

/// The classifier head of a `modern-bert` file (roadmap S4.7b), transcribed from
/// `llm_graph_context::build_pooling`'s `LLAMA_POOLING_TYPE_RANK` branch
/// (`src/llama-graph.cpp:4185-4220`) and `llama_model_modern_bert::load_arch_tensors`
/// (`src/models/modern-bert.cpp:90-93`):
///
/// ```text
/// h = pooled(x)                    // mean (the default), or row 0
/// h = cls.weight · h               // no bias: llama.cpp creates no `cls.bias` here
/// h = act(h)                       // classifier.activation, GELU_ERF by default
/// h = layer_norm(h, cls.norm.w)    // weight only, no bias
/// y = cls.output.weight · h + cls.output.bias
/// ```
///
/// The order is Transformers' `ModernBertPredictionHead`: dense → activation → norm →
/// decoder. There is no softmax for this family: `/v1/rerank`'s score is `y[0]`, the raw
/// logit, and a labelled file's scores are the caller's softmax.
pub(crate) struct ModernBertHead {
    /// `cls.weight`: the dense layer. Absent on a one-layer head (`cls.output` alone).
    pub(crate) dense: Option<Linear>,
    /// `cls.norm.weight`: the head's LayerNorm, weight only (`build_norm` is given a
    /// NULL bias, `src/llama-graph.cpp:4205-4207`).
    pub(crate) norm_w: Option<Vec<f32>>,
    /// `cls.output.weight` (+ `.bias`): the projection to the file's outputs.
    pub(crate) out: Option<Linear>,
    pub(crate) out_b: Vec<f32>,
    pub(crate) act: HeadAct,
    /// `classifier.pooling_type == 1` (mean) instead of row 0; the default is mean
    /// (`src/models/modern-bert.cpp:26-28`).
    pub(crate) mean: bool,
    /// The width of one output row: `cls.output`'s rows, or `d_model` without one.
    pub(crate) n_out: usize,
    /// `classifier.output_labels`, in file order. Empty when the file has none.
    pub(crate) labels: Vec<String>,
}

/// `classifier.activation`: llama.cpp's `LLM_CLS_ACT_TYPES_FROM_STRING`
/// (`src/llama-model.cpp:1209-1213`). Unlike the `bert` family, a `modern-bert` file
/// without the key gets the **erf** GELU, Transformers' default for this head
/// (`src/models/modern-bert.cpp:31-35`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeadAct {
    Tanh,
    Gelu,
    Silu,
}

/// One pre-norm encoder layer.
pub(crate) struct ModernBertLayer {
    /// `attn_qkv`: **one** matrix with the query, key and value rows concatenated
    /// ([`Config::q_dim`], then two [`Config::kv_dim`] blocks). No bias.
    pub(crate) wqkv: Linear,
    pub(crate) wo: Linear,
    /// `attn_norm`: the pre-attention LayerNorm. llama.cpp creates it
    /// `TENSOR_NOT_REQUIRED` for layer 0, and the graph uses the identity when the
    /// tensor is absent.
    pub(crate) attn_norm: Option<Vec<f32>>,
    /// `ffn_norm`: the pre-feed-forward LayerNorm.
    pub(crate) ffn_norm_w: Vec<f32>,
    /// `ffn_up`: `2·n_ff` output rows, the gated pair.
    pub(crate) up: Linear,
    pub(crate) down: Linear,
    /// The layer's attention is over a symmetric window (`is_swa_impl`).
    pub(crate) is_swa: bool,
}

/// A loaded `modern-bert` model.
pub struct ModernBert {
    pub(crate) tok_embd: TensorInfo,
    pub(crate) tok_norm_w: Vec<f32>,
    pub(crate) layers: Vec<ModernBertLayer>,
    pub(crate) out_norm_w: Vec<f32>,
    pub(crate) act: GatedAct,
    /// The classifier head, when the file carries one (S4.7b).
    pub(crate) head: Option<ModernBertHead>,
    /// `attention.sliding_window`: the full window width; a local layer masks a
    /// pair when `|p1 - p0| > window / 2` (`LLAMA_SWA_TYPE_SYMMETRIC`).
    pub(crate) window: usize,
    /// The local layers' RoPE table (`rope.freq_base_swa`).
    pub(crate) rope_local: std::sync::RwLock<Vec<f32>>,
    pub(crate) rope_base_local: f32,
    /// `context_length`: the longest text, in tokens.
    pub(crate) ctx: usize,
    /// Whether any matrix reads the per-256 activation layout (S4.9a).
    pub(crate) k_layout: bool,
}

/// The family's scratch, sized once by [`Architecture::new_scratch`]: the fused
/// q/k/v projection's output, `3·d_model` wide per row.
pub(crate) struct ModernBertScratch {
    /// One buffer, used twice per layer: the fused q/k/v output
    /// (`q_dim + 2·kv_dim` per row) in the attention half, then `ffn_up`'s
    /// `2·n_ff`-wide gated pair in the feed-forward half. They never overlap, so
    /// the wider of the two sizes the buffer.
    buf: Vec<f32>,
    width: usize,
}

impl ModernBertScratch {
    fn new(max_batch: usize, width: usize) -> ModernBertScratch {
        ModernBertScratch {
            buf: vec![0.0; max_batch * width],
            width,
        }
    }

    fn ensure(&mut self, nb: usize) {
        if self.buf.len() < nb * self.width {
            self.buf.resize(nb * self.width, 0.0);
        }
    }
}

/// A vector weight (a norm's weight), dequantised.
fn vec_f32(g: &Gguf, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let t = g.tensor(name)?;
    if t.numel() as usize != n {
        return Err(format!("{name} has {} entries, expected {n}", t.numel()));
    }
    let mut v = vec![0.0f32; n];
    peregrine_kernels::quant::dequant_row(t.kind, g.bytes(t), &mut v);
    Ok(v)
}

impl ModernBert {
    #[allow(clippy::too_many_lines)]
    fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        let arch = g.str("general.architecture")?;
        if arch != "modern-bert" {
            return Err(format!(
                "expected architecture \"modern-bert\", found {arch:?} — bert, nomic-bert and \
                 jina-bert are other families (roadmap S4.6, S4.7, S4.8)"
            ));
        }
        let k = |suffix: &str| format!("modern-bert.{suffix}");
        let n_layer = g.u32(&k("block_count"))? as usize;
        let d_model = g.u32(&k("embedding_length"))? as usize;
        let n_head = g.u32(&k("attention.head_count"))? as usize;
        let n_kv_head = g
            .u32(&k("attention.head_count_kv"))
            .unwrap_or(n_head as u32) as usize;
        let n_ff = g.u32(&k("feed_forward_length"))? as usize;
        let head_dim = g
            .u32(&k("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(d_model / n_head);
        if n_layer == 0 || n_ff == 0 || n_head == 0 || n_kv_head == 0 {
            return Err(format!(
                "implausible shape: {n_layer} layers, {n_head}/{n_kv_head} heads, {n_ff} \
                 feed-forward"
            ));
        }
        // llama.cpp creates the fused `attn_qkv` as `[n_embd, 3·n_embd]`: a modern-bert
        // file has no separate query width and no grouped-query attention.
        if head_dim * n_head != d_model || head_dim * n_kv_head != d_model {
            return Err(format!(
                "a modern-bert attention head is d_model/n_head wide and keys and values are \
                 the same width as queries: {n_head} heads of {head_dim}, {n_kv_head} kv heads, \
                 {d_model} wide"
            ));
        }
        let qd = n_head * head_dim;
        let kvd = n_kv_head * head_dim;
        let qkv_width = qd + 2 * kvd;

        let ctx = g.u32(&k("context_length")).map_err(|_| {
            "modern-bert model needs modern-bert.context_length (its longest text)".to_string()
        })? as usize;
        if ctx == 0 {
            return Err("modern-bert.context_length is 0".into());
        }
        let eps = g.f32(&k("attention.layer_norm_epsilon")).unwrap_or(1e-5);

        // A decision checkpoint (`ModernBertDecisionModel`: Laya, Julia-1) appends its
        // scorer's pre-norm blocks to `block_count` (`src/models/modern-bert.cpp:38-52`,
        // `conversion/bert.py`'s `_load_decision_hparams`). Reading them as encoder
        // layers would compute a shape the file's head never saw, so the file is refused
        // by name until S4.12.
        match g.u32(&k("decision.block_count")) {
            Ok(n) if n > 0 => {
                return Err(format!(
                    "{}decision.block_count is {n}: a modern-bert decision model's head blocks \
                     are not read yet (roadmap S4.12)",
                    k("")
                ))
            }
            _ => {}
        }

        // The window and its pattern: llama.cpp reads `attention.sliding_window`
        // and, when it is positive, `load_swa_pattern(ml, 3, dense_first = true)`
        // (`src/models/modern-bert.cpp`). `il % pattern != 0` is a local layer, so
        // layer 0 and every third layer after it are global; a pattern of 0 makes
        // every layer local. A file without the window key has no local layers.
        let window = g.u32(&k("attention.sliding_window")).unwrap_or(0) as usize;
        let pattern = g
            .u32(&k("attention.sliding_window_pattern"))
            .map(|v| v as usize)
            .unwrap_or(3);
        let is_swa: Vec<bool> = if window > 0 {
            (0..n_layer)
                .map(|il| pattern == 0 || il % pattern != 0)
                .collect()
        } else {
            vec![false; n_layer]
        };

        // RoPE: NEOX pairing over `rope.dimension_count` (the head width by
        // default), base `rope.freq_base` for the global layers and
        // `rope.freq_base_swa` for the local ones. llama.cpp's two defaults are
        // 10 000 each, and a file that sets only the global key leaves the local
        // layers at 10 000 — the pair is not inherited.
        let rope_dim = g
            .u32(&k("rope.dimension_count"))
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        if rope_dim == 0 || rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
            return Err(format!(
                "rope.dimension_count is {rope_dim}, expected an even number up to the head \
                 width {head_dim}"
            ));
        }
        let rope_base = g.f32(&k("rope.freq_base")).unwrap_or(10_000.0);
        let rope_base_local = g.f32(&k("rope.freq_base_swa")).unwrap_or(10_000.0);

        let act = match g.str(&k("hidden_activation")) {
            Ok(name) => match name {
                "gelu" | "gelu_python" => GatedAct::GegluErf,
                "gelu_new" | "gelu_fast" | "gelu_accurate" | "gelu_pyTorch_tanh"
                | "gelu_pytorch_tanh" | "gelu_python_tanh" | "geglu" => GatedAct::Geglu,
                "silu" | "swish" | "swiglu" => GatedAct::Swiglu,
                other => {
                    return Err(format!(
                        "modern-bert.hidden_activation {other:?} is not a gated activation \
                         Peregrine reads (gelu, the gelu_new family, silu)"
                    ))
                }
            },
            Err(_) => GatedAct::GegluErf,
        };

        // The vocabulary must be one the tokenizer can read: a `modern-bert` file
        // is GPT-2 byte-level BPE with the `modern-bert` pre-tokenizer, and the
        // mmBERT vocabulary's `mmbert` pre-tokenizer is not implemented.
        let n_vocab = g.u32(&k("vocab_size")).map(|v| v as usize).ok();
        if let Some(n) = n_vocab {
            if n != tk.n_vocab {
                return Err(format!(
                    "metadata says {n} vocabulary entries but the tokenizer has {}",
                    tk.n_vocab
                ));
            }
        }

        let tok_embd = g.tensor("token_embd.weight")?.clone();
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
        shape(&tok_embd, d_model, tk.n_vocab)?;
        let tok_norm_w = vec_f32(g, "token_embd_norm.weight", d_model)?;
        let out_norm_w = vec_f32(g, "output_norm.weight", d_model)?;

        let mut layers = Vec::with_capacity(n_layer);
        for (i, &swa) in is_swa.iter().enumerate() {
            let p = |s: &str| format!("blk.{i}.{s}");
            let wqkv = Linear::Dense(g.tensor(&p("attn_qkv.weight"))?.clone());
            let wo = Linear::Dense(g.tensor(&p("attn_output.weight"))?.clone());
            let up = Linear::Dense(g.tensor(&p("ffn_up.weight"))?.clone());
            let down = Linear::Dense(g.tensor(&p("ffn_down.weight"))?.clone());
            for (lin, d_in, d_out, what) in [
                (&wqkv, d_model, qkv_width, "attn_qkv"),
                (&wo, qd, d_model, "attn_output"),
                (&up, d_model, 2 * n_ff, "ffn_up"),
                (&down, n_ff, d_model, "ffn_down"),
            ] {
                if let Some(t) = lin.dense() {
                    shape(t, d_in, d_out).map_err(|e| format!("blk.{i}.{what}: {e}"))?;
                }
            }
            let attn_norm = match g.tensor(&p("attn_norm.weight")) {
                Ok(_) => Some(vec_f32(g, &p("attn_norm.weight"), d_model)?),
                Err(_) if i == 0 => None,
                Err(_) => return Err(format!("blk.{i}.attn_norm.weight is missing")),
            };
            let ffn_norm_w = vec_f32(g, &p("ffn_norm.weight"), d_model)?;
            layers.push(ModernBertLayer {
                wqkv,
                wo,
                attn_norm,
                ffn_norm_w,
                up,
                down,
                is_swa: swa,
            });
        }

        // The classifier head (S4.7b): `cls`, `cls.norm`, `cls.output` and
        // `classifier.output_labels`. A file that carries them serves `/v1/rerank` and
        // `/v1/classify` from them; its embeddings still use `pooling_type`.
        let head = read_head(g, d_model)?;

        let cfg = Config {
            arch_name: "modern-bert",
            // `llama_model_rope_type` lists MODERN_BERT in the NEOX group.
            rope_neox: true,
            n_layer,
            n_kv_layer: n_layer,
            n_head,
            n_kv_head,
            head_dim,
            d_model,
            n_ff,
            n_vocab: tk.n_vocab,
            rope_dim,
            eps,
            rope_base,
            ctx_train: ctx,
            rope_yarn: None,
        };

        let mut linears: Vec<&Linear> = Vec::new();
        for l in &layers {
            linears.extend([&l.wqkv, &l.wo, &l.up, &l.down]);
        }
        if let Some(h) = &head {
            // The head's matrices read the same activation rows as a layer's, so they
            // count for the layout decision too (a Q8_0 `cls.weight` needs the int8 rows).
            linears.extend(h.dense.iter().chain(h.out.iter()));
        }
        let k_layout = linears
            .iter()
            .filter_map(|lin| lin.dense())
            .any(|t| t.kind.reads_k_layout());

        let rope_local = peregrine_core::blocks::rope::build_rope_table(
            rope_dim,
            rope_base_local,
            ctx.max(1),
            None,
        );

        Ok(Loaded {
            cfg,
            arch: Box::new(ModernBert {
                tok_embd,
                tok_norm_w,
                layers,
                out_norm_w,
                act,
                head,
                window,
                rope_local: std::sync::RwLock::new(rope_local),
                rope_base_local,
                ctx,
                k_layout,
            }),
            rope_factors: None,
            tiles: TileStore::default(),
        })
    }
}

/// Read and check the optional classifier head (S4.7b). `Ok(None)` when the file has
/// none; a malformed head is an error naming the tensor, because llama.cpp would abort
/// on it.
///
/// llama.cpp creates `cls.weight`, `cls.norm.weight`, `cls.output.weight` and
/// `cls.output.bias` `TENSOR_NOT_REQUIRED` for this family (`src/models/modern-bert.cpp:90-93`)
/// and sizes the output by `classifier.output_labels` when the file has them, by 1
/// otherwise (`src/llama-model.cpp:1635-1638`). It creates **no** `cls.bias` and no
/// `cls.norm.bias` for a sequence-classification file — the dense layer and the head
/// norm are bias-less — so a file that carries those tensors is read without them, as
/// llama.cpp reads it.
fn read_head(g: &Gguf, d_model: usize) -> Result<Option<ModernBertHead>, String> {
    let arch = "modern-bert";
    let labels = g
        .arr_str(&format!("{arch}.classifier.output_labels"))
        .map(<[String]>::to_vec)
        .unwrap_or_default();
    // The converters write a one-output `cls.output.weight` 1-d, as they write
    // `token_types.weight`; llama.cpp's tensor shape is `{n_embd, 1}` either way.
    let mat2 = |name: &str| -> Option<TensorInfo> {
        g.tensor(name).ok().map(|t| {
            let mut t = t.clone();
            if t.dims.len() == 1 {
                t.dims.push(1);
            }
            t
        })
    };
    let dense_t = mat2("cls.weight");
    let out_t = mat2("cls.output.weight");
    let norm_t = g.tensor("cls.norm.weight").ok().cloned();
    if dense_t.is_none() && out_t.is_none() && norm_t.is_none() && labels.is_empty() {
        return Ok(None);
    }
    if dense_t.is_none() && out_t.is_none() {
        let named = if norm_t.is_some() {
            "cls.norm.weight"
        } else {
            "classifier.output_labels"
        };
        return Err(format!(
            "modern-bert classifier head: {named} is present but the file carries neither \
             cls.weight nor cls.output.weight, so there is nothing to apply it to"
        ));
    }
    // llama.cpp sizes `cls.output` by the labels when the file has them, and by 1
    // otherwise; a mismatch is the metadata's error.
    let n_out = if labels.is_empty() {
        out_t.as_ref().map_or(d_model, |t| t.dims[1] as usize)
    } else {
        labels.len()
    };
    let dense = match dense_t {
        Some(t) => {
            shape_of(&t, d_model, d_model)
                .map_err(|e| format!("the classifier head's cls.weight: {e}"))?;
            Some(Linear::Dense(t))
        }
        None => None,
    };
    let out = match out_t {
        Some(t) => {
            let want = if labels.is_empty() {
                "one output".to_string()
            } else {
                format!("{} outputs (its labels)", labels.len())
            };
            if t.dims[0] as usize != d_model || t.dims[1] as usize != n_out {
                return Err(format!(
                    "cls.output.weight is {:?}, expected [{d_model}, {n_out}] for {want}",
                    t.dims
                ));
            }
            Some(Linear::Dense(t))
        }
        None => None,
    };
    if out.is_none() && !labels.is_empty() {
        return Err(format!(
            "modern-bert.classifier.output_labels names {} outputs but the file carries no \
             cls.output.weight to produce them",
            labels.len()
        ));
    }
    let out_b = if out.is_some() {
        match g.tensor("cls.output.bias") {
            Ok(t) => {
                if t.numel() as usize != n_out {
                    return Err(format!(
                        "cls.output.bias has {} entries, expected {n_out}",
                        t.numel()
                    ));
                }
                vec_f32(g, "cls.output.bias", n_out)?
            }
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    };
    let norm_w = match &norm_t {
        Some(t) => {
            shape_of(t, d_model, 1).map_err(|e| format!("cls.norm.weight: {e}"))?;
            Some(vec_f32(g, "cls.norm.weight", d_model)?)
        }
        None => None,
    };
    let act = match g.str(&format!("{arch}.classifier.activation")) {
        Ok("gelu") => HeadAct::Gelu,
        Ok("silu") => HeadAct::Silu,
        Ok("tanh") => HeadAct::Tanh,
        Ok(other) => {
            return Err(format!(
                "{arch}.classifier.activation {other:?} is not one llama.cpp reads \
                 (gelu, silu, tanh)"
            ))
        }
        // `src/models/modern-bert.cpp:31-35`: the file's key, else GELU_ERF.
        Err(_) => HeadAct::Gelu,
    };
    // `classifier.pooling_type`: 1 means mean, anything else is row 0, and the absent
    // key means mean for this family (`src/models/modern-bert.cpp:26-28`).
    let mean = g
        .u32(&format!("{arch}.classifier.pooling_type"))
        .map_or(true, |v| v == 1);
    Ok(Some(ModernBertHead {
        dense,
        norm_w,
        out,
        out_b,
        act,
        mean,
        n_out,
        labels,
    }))
}

/// A tensor's `[d_in, d_out]` shape, with a 1-d tensor read as one output row.
fn shape_of(t: &TensorInfo, d_in: usize, d_out: usize) -> Result<(), String> {
    let a = t.dims.first().copied().unwrap_or(0) as usize;
    let b = t.dims.get(1).copied().unwrap_or(1) as usize;
    if t.dims.len() > 2 || a != d_in || b != d_out {
        return Err(format!(
            "{} is {:?}, expected [{d_in}, {d_out}]",
            t.name, t.dims
        ));
    }
    Ok(())
}

/// Every row of a weight-less LayerNorm, into `out` (`layer_norm_rows_nobias`).
///
/// Rows are independent, so a batch is split across the pool; a row's bits are
/// the scalar kernel's whatever the split.
#[allow(clippy::too_many_arguments)]
fn norm_rows_nobias(
    pool: &Pool,
    stats: &mut StepStats,
    x: &[f32],
    w: &[f32],
    eps: f32,
    dim: usize,
    nb: usize,
    out: &mut [f32],
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    peregrine_core::blocks::par::for_row_chunks(
        pool,
        parallel,
        out,
        dim,
        nb,
        Sched::Static,
        |_wk, rows, oc| {
            let n = rows.len();
            vmath::layer_norm_rows_nobias(&x[rows.start * dim..rows.end * dim], w, eps, dim, n, oc);
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

/// The gated feed-forward's activation, in place in `hb`'s first half, then
/// compacted into `hb2` (one `n_ff` row per token) for the down projection.
///
/// `hb` holds `2·n_ff` per row: the first half is activated, the second half
/// multiplies it (`ggml_geglu_erf` / `ggml_geglu` / `ggml_swiglu`). Rows are split
/// across the pool; a row's bits are the row's own.
fn gated_compact(
    pool: &Pool,
    stats: &mut StepStats,
    hb: &mut [f32],
    hb2: &mut [f32],
    act: GatedAct,
    n_ff: usize,
    nb: usize,
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    for_row_chunks_2(
        pool,
        parallel,
        hb,
        2 * n_ff,
        hb2,
        n_ff,
        nb,
        Sched::Static,
        |_wk, rows, hc, oc| {
            for (i, _r) in rows.enumerate() {
                let row = &mut hc[i * 2 * n_ff..(i + 1) * 2 * n_ff];
                let (gate, up) = row.split_at_mut(n_ff);
                match act {
                    GatedAct::GegluErf => vmath::geglu_erf(gate, up),
                    GatedAct::Geglu => vmath::geglu(gate, up),
                    GatedAct::Swiglu => vmath::swiglu(gate, up),
                }
                oc[i * n_ff..(i + 1) * n_ff].copy_from_slice(gate);
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

impl ModernBert {
    /// Grow the local layers' RoPE table to `needed` positions. It is built for
    /// the file's context at load, so a warm pass does not allocate.
    fn ensure_rope_local(&self, cfg: &Config, needed: usize) {
        let half = cfg.rope_dim / 2;
        let needed_floats = needed * half * 2;
        if self.rope_local.read().unwrap().len() >= needed_floats {
            return;
        }
        let mut current = self.rope_local.write().unwrap();
        if current.len() < needed_floats {
            let target = needed.max(current.len() / (half * 2).max(1));
            *current = peregrine_core::blocks::rope::build_rope_table(
                cfg.rope_dim,
                self.rope_base_local,
                target,
                None,
            );
        }
    }
}

impl Architecture for ModernBert {
    fn causality(&self) -> Causality {
        Causality::Bidirectional
    }

    fn head_width(&self) -> usize {
        self.head.as_ref().map_or(0, |h| h.n_out)
    }

    fn head_labels(&self) -> &[String] {
        self.head.as_ref().map_or(&[], |h| h.labels.as_slice())
    }

    fn act_layout(&self) -> ActLayout {
        let mut keep_f32 = false;
        let mut quantize = false;
        self.linear_groups(&mut |lins| {
            for lin in lins {
                match lin.dense() {
                    Some(ti) => {
                        if ti.fp_kind.is_some()
                            || matches!(
                                ti.kind,
                                peregrine_core::gguf::Format::F32
                                    | peregrine_core::gguf::Format::F16
                                    | peregrine_core::gguf::Format::BF16
                            )
                        {
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

    fn new_scratch(
        &self,
        cfg: &Config,
        max_batch: usize,
    ) -> Option<Box<dyn std::any::Any + Send + Sync>> {
        Some(Box::new(ModernBertScratch::new(
            max_batch,
            (cfg.q_dim() + 2 * cfg.kv_dim()).max(2 * cfg.n_ff),
        )))
    }

    fn forward_rows(
        &self,
        _w: &Weights,
        cfg: &Config,
        _st: &mut State,
        _caches: &mut peregrine_core::model::Caches<'_>,
        _pool: &Pool,
        _tokens: &[u32],
        _positions: &[usize],
        _want_logits: &[usize],
        _stats: &mut StepStats,
    ) -> Result<(), String> {
        Err(format!(
            "a {} model has no causal forward; the engine must use forward_texts",
            cfg.arch_name
        ))
    }

    fn linears_mut(
        &mut self,
        f: &mut dyn FnMut(&mut Linear) -> Result<(), String>,
    ) -> Result<(), String> {
        for l in &mut self.layers {
            for lin in [&mut l.wqkv, &mut l.wo, &mut l.up, &mut l.down] {
                f(lin)?;
            }
        }
        if let Some(h) = &mut self.head {
            for lin in [h.dense.as_mut(), h.out.as_mut()].into_iter().flatten() {
                f(lin)?;
            }
        }
        Ok(())
    }

    fn linear_groups(&self, f: &mut dyn FnMut(&[&Linear])) {
        for l in &self.layers {
            f(&[&l.wqkv]);
            f(&[&l.wo]);
            f(&[&l.up]);
            f(&[&l.down]);
        }
        if let Some(h) = &self.head {
            for lin in [h.dense.as_ref(), h.out.as_ref()].into_iter().flatten() {
                f(&[lin]);
            }
        }
    }

    /// One bidirectional pass over whole texts (design §2.2).
    ///
    /// The transcribed llama.cpp graph (`src/models/modern-bert.cpp`), per text
    /// and in order:
    ///
    /// 1. `x = token_embd[token]` (no position or token-type table);
    /// 2. `token_embd_norm`: LayerNorm with a weight and no bias;
    /// 3. per layer, pre-norm:
    ///    a. `attn_norm` (the identity on layer 0 when its tensor is absent), then
    ///    the fused `attn_qkv` (+ biases in a decision head only — never here);
    ///    b. RoPE (NEOX, the local base on a local layer) on q and k, then
    ///    attention over the whole text at `1/√head_dim`; a local layer masks
    ///    `|p1 - p0| > window / 2`, a global one masks nothing;
    ///    c. `attn_output`; `x = x + that`; `ffn_norm`;
    ///    d. `ffn_up` (the gated pair), the first half's activation times the
    ///    second, `ffn_down`; `x = x + that`;
    /// 4. `output_norm`: LayerNorm with a weight and no bias;
    /// 5. the read-out, per text: pooling of those rows by the caller's type (then the
    ///    engine's truncation and L2 normalisation), or — for a `TextWant::Head` text —
    ///    the same rows through the classifier head (mean or row 0, `cls`, the head's
    ///    activation, `cls.norm`, `cls.output`), into `st.head_out`.
    ///
    /// No layer reads a cache, nothing is chunked, and a text's rows are computed
    /// exactly as they would be alone, so its vector is bit-identical in any batch.
    fn forward_texts(
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
        let ff = cfg.n_ff;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let qkv_width = qd + 2 * kvd;
        let nb: usize = texts.iter().map(|t| t.len()).sum();
        if nb == 0 {
            stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
            return Ok(());
        }
        for (i, t) in texts.iter().enumerate() {
            if t.len() > self.ctx {
                return Err(format!(
                    "text {i} has {} tokens; the modern-bert model's context_length is {}",
                    t.len(),
                    self.ctx
                ));
            }
        }
        for a in st.acts.iter_mut() {
            a.set_k_layout(self.k_layout);
        }
        if st.hidden.len() < nb * dm {
            st.hidden.resize(nb * dm, 0.0);
        }

        // The batch's spans and per-row positions: every text starts at 0.
        let mut spans = std::mem::take(&mut st.spans);
        spans.clear();
        let mut row = 0usize;
        for t in texts {
            spans.push(peregrine_core::blocks::bidir::TextSpan {
                row0: row,
                len: t.len(),
            });
            row += t.len();
        }
        let mut pos = std::mem::take(&mut st.positions);
        if pos.len() < nb {
            pos.resize(nb, 0);
        }
        for (t, s) in texts.iter().zip(&spans) {
            for i in 0..t.len() {
                pos[s.row0 + i] = i;
            }
        }

        // ---- embeddings: the token rows alone ----
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
        // `token_embd_norm` feeds the first block's fused q/k/v.
        norm_rows_nobias(
            pool,
            stats,
            &st.x,
            &self.tok_norm_w,
            cfg.eps,
            dm,
            nb,
            &mut st.xb,
        );
        st.x[..nb * dm].copy_from_slice(&st.xb[..nb * dm]);

        let max_pos = pos[..nb].iter().copied().max().unwrap_or(0);
        self.ensure_rope_local(cfg, max_pos + 1);
        let rope_local = self.rope_local.read().unwrap();
        let rope_local: &[f32] = &rope_local;

        // The fused q/k/v output, sized once. A state built by `State::new` (not
        // `Model::new_state`) has no family scratch, so it gets one here.
        if st.fam.is_none() {
            st.fam = Some(Box::new(ModernBertScratch::new(
                st.max_batch,
                qkv_width.max(2 * ff),
            )));
        }
        let scratch = st
            .fam
            .as_mut()
            .and_then(|f| f.downcast_mut::<ModernBertScratch>())
            .ok_or("the state has no modern-bert scratch")?;
        scratch.ensure(nb);

        for l in &self.layers {
            // ---- attention pre-norm: the identity on layer 0 without the tensor ----
            let normed = match &l.attn_norm {
                Some(nw) => {
                    norm_rows_nobias(pool, stats, &st.x, nw, cfg.eps, dm, nb, &mut st.xb);
                    true
                }
                None => false,
            };

            // ---- fused q/k/v: one weight pass, one quantisation per input row ----
            {
                let input: &[f32] = if normed {
                    &st.xb[..nb * dm]
                } else {
                    &st.x[..nb * dm]
                };
                linear_rows(
                    w,
                    pool,
                    stats,
                    &l.wqkv,
                    &mut st.acts,
                    &mut st.lb_x,
                    &mut st.lb_t,
                    input,
                    dm,
                    nb,
                    &mut scratch.buf,
                    qkv_width,
                    &mut st.kq,
                );
            }

            // ---- split the fused rows into q, k and v ----
            {
                let qkv = &scratch.buf[..nb * qkv_width];
                let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
                for_row_chunks_3(
                    pool,
                    parallel,
                    &mut st.q,
                    qd,
                    &mut st.k,
                    kvd,
                    &mut st.v,
                    kvd,
                    nb,
                    Sched::Static,
                    |_wk, rows, qc, kc, vc| {
                        for (i, r) in rows.enumerate() {
                            let src = &qkv[r * qkv_width..(r + 1) * qkv_width];
                            qc[i * qd..(i + 1) * qd].copy_from_slice(&src[..qd]);
                            kc[i * kvd..(i + 1) * kvd].copy_from_slice(&src[qd..qd + kvd]);
                            vc[i * kvd..(i + 1) * kvd].copy_from_slice(&src[qd + kvd..]);
                        }
                    },
                );
                if parallel {
                    stats.barriers += 1;
                }
            }

            // ---- RoPE: the local base on a local layer, the global one otherwise ----
            let rope_table = if l.is_swa { rope_local } else { &st.rope };
            peregrine_core::blocks::attn::prepare_rows(
                cfg,
                None,
                None,
                None,
                None,
                None,
                rope_table,
                &mut st.q,
                &mut st.k,
                &mut st.v,
                nb,
                &pos[..nb],
                pool,
                stats,
            );
            // llama.cpp passes `kq_scale = 1/√head_dim` to the attention; folding it
            // into Q is the same product and keeps the block's interface.
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            for q in &mut st.q[..nb * qd] {
                *q *= scale;
            }

            // ---- attention: one n × n block per text, the window on local layers ----
            let half_window = if l.is_swa {
                Some(self.window / 2)
            } else {
                None
            };
            peregrine_core::blocks::bidir::attend_texts(
                cfg,
                &st.q,
                &st.k,
                &st.v,
                &mut st.xb,
                &spans,
                half_window,
                None,
                1.0,
                &mut st.attn.bidir,
                pool,
                stats,
            );

            // ---- attn_output, residual: `ffn_inp = attn_out + x` ----
            linear_rows(
                w,
                pool,
                stats,
                &l.wo,
                &mut st.acts,
                &mut st.lb_x,
                &mut st.lb_t,
                &st.xb,
                qd,
                nb,
                &mut st.xb2,
                dm,
                &mut st.kq,
            );
            for i in 0..nb * dm {
                st.xb2[i] += st.x[i];
            }

            // ---- ffn_norm, the gated pair, ffn_down, the second residual ----
            norm_rows_nobias(
                pool,
                stats,
                &st.xb2,
                &l.ffn_norm_w,
                cfg.eps,
                dm,
                nb,
                &mut st.xb,
            );
            {
                // `ffn_up` is `2·n_ff` wide: one weight pass over the normed rows.
                linear_rows(
                    w,
                    pool,
                    stats,
                    &l.up,
                    &mut st.acts,
                    &mut st.lb_x,
                    &mut st.lb_t,
                    &st.xb,
                    dm,
                    nb,
                    &mut scratch.buf,
                    2 * ff,
                    &mut st.kq,
                );
            }
            gated_compact(pool, stats, &mut scratch.buf, &mut st.hb2, self.act, ff, nb);
            linear_rows(
                w,
                pool,
                stats,
                &l.down,
                &mut st.acts,
                &mut st.lb_x,
                &mut st.lb_t,
                &st.hb2,
                ff,
                nb,
                &mut st.xb,
                dm,
                &mut st.kq,
            );
            for i in 0..nb * dm {
                st.x[i] = st.xb[i] + st.xb2[i];
            }
        }

        // ---- output_norm: the rows pooling reads ----
        norm_rows_nobias(
            pool,
            stats,
            &st.x,
            &self.out_norm_w,
            cfg.eps,
            dm,
            nb,
            &mut st.xb,
        );
        // The read-outs draw on both `st.xb` (as the head's and the pooler's working row)
        // and these rows, so keep one exact copy of them in `st.x` — the pre-norm rows
        // there are spent. A copy is exact, so a vector's bits do not change.
        st.x[..nb * dm].copy_from_slice(&st.xb[..nb * dm]);

        // ---- read-outs, text by text (the vector or the head row this text's slot reads) ----
        if st.pool_sum.len() < dm {
            st.pool_sum.resize(dm, 0.0);
        }
        let hw = self.head.as_ref().map_or(0, |h| h.n_out);
        for (i, (t, s)) in texts.iter().zip(&spans).enumerate() {
            let row0 = s.row0;
            match wants[i] {
                peregrine_core::arch::TextWant::Pooled(pooling) => {
                    peregrine_core::embed::pool_rows_into(
                        &st.x[row0 * dm..(row0 + t.len()) * dm],
                        dm,
                        pooling,
                        &mut st.pool_sum,
                        &mut st.xb2[..dm],
                    );
                    st.hidden[row0 * dm..row0 * dm + dm].copy_from_slice(&st.xb2[..dm]);
                }
                peregrine_core::arch::TextWant::Head => {
                    let h = self.head.as_ref().ok_or_else(|| {
                        format!("the {} model has no classifier head", cfg.arch_name)
                    })?;
                    head_row(w, pool, st, h, cfg.eps, row0, t.len(), dm, hw, stats)?;
                }
            }
        }
        st.positions = pos;
        st.spans = spans;
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }
}

/// One text's classifier head row (S4.7b), into `st.head_out[row0 * hw ..]`.
///
/// The input is the text's `output_norm` rows (`st.x`), reduced by the head's
/// `classifier.pooling_type` — the mean, unless the file names another type, or row 0 —
/// and then exactly `build_pooling`'s rank branch for this family
/// (`src/llama-graph.cpp:4185-4220`): `cls` (no bias) → `classifier.activation` →
/// `cls.norm` (weight only) → `cls.output` (+ bias). A head's output does not depend on
/// the other texts of the pass: every input is that text's own row and each projection
/// is one row, quantised on its own.
#[allow(clippy::too_many_arguments)]
fn head_row(
    w: &Weights,
    pool: &Pool,
    st: &mut State,
    h: &ModernBertHead,
    eps: f32,
    row0: usize,
    len: usize,
    dm: usize,
    hw: usize,
    stats: &mut StepStats,
) -> Result<(), String> {
    // The pooled row, into `st.xb[..dm]` (the head's input row).
    if h.mean {
        st.pool_sum[..dm].fill(0.0);
        for r in st.x[row0 * dm..(row0 + len) * dm].chunks_exact(dm) {
            for (s, &x) in st.pool_sum[..dm].iter_mut().zip(r) {
                *s += x as f64;
            }
        }
        let inv = 1.0 / len.max(1) as f64;
        for (o, &s) in st.xb[..dm].iter_mut().zip(&st.pool_sum[..dm]) {
            *o = (s * inv) as f32;
        }
    } else {
        st.xb[..dm].copy_from_slice(&st.x[row0 * dm..row0 * dm + dm]);
    }

    if let Some(dense) = &h.dense {
        // `cls.weight · h`: the dense layer is bias-less for this family — llama.cpp
        // creates no `cls.bias` for a sequence-classification file
        // (`src/models/modern-bert.cpp:92`).
        linear_rows(
            w,
            pool,
            stats,
            dense,
            &mut st.acts,
            &mut st.lb_x,
            &mut st.lb_t,
            &st.xb[..dm],
            dm,
            1,
            &mut st.xb2[..dm],
            dm,
            &mut st.kq,
        );
        // The head's activation (`hparams.act_cls`), in place.
        match h.act {
            // The engine's own `tanh`, not the C library's: `tanhf` changed its rounding in
            // glibc 2.40, and a score may not depend on the host's libc (issue #64).
            HeadAct::Tanh => {
                for v in st.xb2[..dm].iter_mut() {
                    *v = vmath::tanh(*v);
                }
            }
            HeadAct::Gelu => vmath::gelu_erf_rows(&mut st.xb2[..dm]),
            HeadAct::Silu => {
                for v in st.xb2[..dm].iter_mut() {
                    *v *= 1.0 / (1.0 + (-*v).exp());
                }
            }
        }
        // `cls.norm`: `build_norm(cur, cls_norm, NULL, LLM_NORM, -1)`, the same
        // weight-less LayerNorm as the encoder's blocks.
        match &h.norm_w {
            Some(nw) => {
                vmath::layer_norm_rows_nobias(&st.xb2[..dm], nw, eps, dm, 1, &mut st.xb[..dm]);
            }
            None => st.xb[..dm].copy_from_slice(&st.xb2[..dm]),
        }
    }

    match &h.out {
        // `cls.output.weight · h + cls.output.bias`: the file's outputs.
        Some(out) => {
            linear_rows(
                w,
                pool,
                stats,
                out,
                &mut st.acts,
                &mut st.lb_x,
                &mut st.lb_t,
                &st.xb[..dm],
                dm,
                1,
                &mut st.head_out[row0 * hw..row0 * hw + hw],
                hw,
                &mut st.kq,
            );
            // A bias of zeros is skipped, as the encoder's residuals skip it, so a `-0.0`
            // from the projection stays.
            if !h.out_b.iter().all(|&b| b == 0.0) {
                for (o, &b) in st.head_out[row0 * hw..row0 * hw + hw]
                    .iter_mut()
                    .zip(&h.out_b)
                {
                    *o += b;
                }
            }
        }
        // A head without `cls.output` *is* its dense layer: `d_model` values out.
        None => st.head_out[row0 * hw..row0 * hw + hw].copy_from_slice(&st.xb[..hw]),
    }
    Ok(())
}
