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

        // The classifier head. llama.cpp reads `cls` (dense), `cls.norm`,
        // `cls.output` and `classifier.output_labels`; a file that carries them
        // still serves embeddings when it declares a pooling type, and Peregrine
        // does not serve `classify`/`rerank` for this family yet (S4.6b is not on
        // main), so the tensors are read, checked and named in one line.
        read_classifier(g, d_model, &k)?;

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

/// Read and check the optional classifier head. `Ok(())` whether or not it is
/// there; a malformed head is an error, because llama.cpp would abort on it.
///
/// The head is not served: classify and rerank arrive with the BERT family's head
/// (S4.6b), and until then a file that carries one gets one line of output and its
/// embeddings, which use `pooling_type` exactly as an embedder's file does.
fn read_classifier(g: &Gguf, d_model: usize, k: &impl Fn(&str) -> String) -> Result<(), String> {
    let names = [
        "cls.weight",
        "cls.bias",
        "cls.norm.weight",
        "cls.norm.bias",
        "cls.output.weight",
        "cls.output.bias",
    ];
    let present: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| g.tensor(n).is_ok())
        .collect();
    let labels = g
        .arr_str(&k("classifier.output_labels"))
        .map(|v| v.to_vec())
        .unwrap_or_default();
    if present.is_empty() && labels.is_empty() {
        return Ok(());
    }
    let n_out = match g.tensor("cls.output.weight") {
        Ok(t) => t.dims.get(1).copied().unwrap_or(0) as usize,
        Err(_) => labels.len(),
    };
    if n_out == 0 {
        return Err(
            "modern-bert classifier head: neither classifier.output_labels nor cls.output.weight \
             says how many outputs it has"
                .into(),
        );
    }
    if !labels.is_empty() && labels.len() != n_out {
        return Err(format!(
            "modern-bert classifier head: {} labels for {n_out} outputs",
            labels.len()
        ));
    }
    let shapes: [(&str, usize, usize); 4] = [
        ("cls.weight", d_model, d_model),
        ("cls.bias", d_model, 1),
        ("cls.norm.weight", d_model, 1),
        ("cls.output.weight", d_model, n_out),
    ];
    for (name, d_in, d_out) in shapes {
        if let Ok(t) = g.tensor(name) {
            let a = t.dims.first().copied().unwrap_or(0) as usize;
            let b = t.dims.get(1).copied().unwrap_or(1) as usize;
            if t.dims.len() > 2 || a != d_in || b != d_out {
                return Err(format!(
                    "{name} is {:?}, expected [{d_in}, {d_out}]",
                    t.dims
                ));
            }
        }
    }
    eprintln!(
        "info: modern-bert classifier head ({} labels) loaded but not served: classify and \
         rerank are not available for this family yet, embeddings only",
        labels.len().max(n_out)
    );
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
        Ok(())
    }

    fn linear_groups(&self, f: &mut dyn FnMut(&[&Linear])) {
        for l in &self.layers {
            f(&[&l.wqkv]);
            f(&[&l.wo]);
            f(&[&l.up]);
            f(&[&l.down]);
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
    /// 5. pooling (the caller's type) of the final rows, then the engine's
    ///    truncation and L2 normalisation.
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
        poolings: &[peregrine_core::embed::Pooling],
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

        // ---- pooling, text by text (the vector this text's slot reads) ----
        if st.pool_sum.len() < dm {
            st.pool_sum.resize(dm, 0.0);
        }
        for (i, (t, s)) in texts.iter().zip(&spans).enumerate() {
            let row0 = s.row0;
            peregrine_core::embed::pool_rows_into(
                &st.xb[row0 * dm..(row0 + t.len()) * dm],
                dm,
                poolings[i],
                &mut st.pool_sum,
                &mut st.xb2[..dm],
            );
            st.hidden[row0 * dm..row0 * dm + dm].copy_from_slice(&st.xb2[..dm]);
        }
        st.positions = pos;
        st.spans = spans;
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }
}
