//! The `nomic-bert` family: the post-norm encoder nomic-embed-text v1.5 ships as
//! (roadmap S4.8a).
//!
//! `architecture = "nomic-bert"` is Nomic AI's BERT-shaped encoder
//! (`nomic-ai/nomic-embed-text-v1.5`). It differs from [`crate::bert`] in five
//! places, all read from llama.cpp v0.9.0 (`cd959ba`):
//!
//! - the token embeddings add the **token-type row 0** and the token embedding
//!   LayerNorm has a weight and a bias, as `bert`'s does (`src/models/bert.cpp`);
//! - **no position table**: positions enter as **RoPE** on q and k (NeoX pairing,
//!   base `nomic-bert.rope.freq_base`, 1000 in the published files) — the graph
//!   branches on `LLM_ARCH_NOMIC_BERT` for the rotation (`src/models/bert.cpp:132`);
//! - the query/key/value projection is **fused** (`attn_qkv`, q rows then k then
//!   v, no bias in the published files), and `attn_output` has no bias either;
//! - the feed-forward is **SwiGLU**: `ffn_gate` through SiLU times `ffn_up`, then
//!   `ffn_down` — the `else` branch of llama.cpp's feed-forward does this with the
//!   biases NULL, so no bias is added (`src/models/bert.cpp:184-198`);
//! - it is **post-norm** like `bert`: `attn_output_norm` and `layer_output_norm`
//!   are LayerNorms with weight and bias, and the last layer's
//!   `layer_output_norm` output is what pooling reads (there is no final norm).
//!
//! A `nomic-bert` file is an encoder: the engine refuses chat and label requests on
//! it by name. The pooling is the file's own `<arch>.pooling_type` (mean, 1, for
//! nomic-embed-text v1.5), and `/v1/embeddings` truncates to `dimensions` and
//! normalises after it, as for every encoder.
//!
//! [`docs/reference/nomic-bert.md`]: ../../../../docs/reference/nomic-bert.md

use std::time::Instant;

use peregrine_core::arch::{ActLayout, Architecture, Causality, Family, Loaded};
use peregrine_core::blocks::par::{for_row_chunks_2, for_row_chunks_3, Sched};
use peregrine_core::blocks::ROW_PARALLEL;
use peregrine_core::gguf::{Gguf, TensorInfo};
use peregrine_core::model::{Config, Linear, State, StepStats, TileStore, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;

use crate::encoder::{linear_rows, linear_rows_shared, RowsOut};

/// The family, registered as `nomic-bert`.
pub struct NomicBertFamily;

impl Family for NomicBertFamily {
    fn names(&self) -> &'static [&'static str] {
        &["nomic-bert"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        NomicBert::load(g, tk)
    }

    /// A `nomic-bert` file is an encoder: no decode path, so the engine refuses
    /// chat and label requests on it by name (design §3).
    fn embedding_only(&self) -> bool {
        true
    }
}

/// One post-norm encoder layer.
pub(crate) struct NomicBertLayer {
    /// `attn_qkv`: the fused query/key/value matrix. The row layout is q (d_model),
    /// then k, then v.
    pub(crate) wqkv: Linear,
    pub(crate) wo: Linear,
    /// The fused projection's bias, when the file has one (the MoE variant does;
    /// nomic-embed-text v1.5 does not). `q_dim + 2·kv_dim` floats.
    pub(crate) bqkv: Vec<f32>,
    /// `attn_output.bias`, optional in llama.cpp (`TENSOR_NOT_REQUIRED`).
    pub(crate) bo: Vec<f32>,
    /// `attn_output_norm`: the post-attention LayerNorm, weight and bias.
    pub(crate) attn_norm_w: Vec<f32>,
    pub(crate) attn_norm_b: Vec<f32>,
    /// The SwiGLU pair: `silu(ffn_gate · x) ⊙ (ffn_up · x)`.
    pub(crate) gate: Linear,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
    /// `layer_output_norm`: the post-FFN LayerNorm, weight and bias.
    pub(crate) out_norm_w: Vec<f32>,
    pub(crate) out_norm_b: Vec<f32>,
}

/// A loaded `nomic-bert` model.
pub struct NomicBert {
    pub(crate) tok_embd: TensorInfo,
    /// `token_types.weight`, row 0 only: llama.cpp hardcodes token type 0
    /// ("Sentence A") for every token.
    pub(crate) type_row0: Option<Vec<f32>>,
    pub(crate) tok_norm_w: Vec<f32>,
    pub(crate) tok_norm_b: Vec<f32>,
    pub(crate) layers: Vec<NomicBertLayer>,
    /// `context_length`: the longest text, in tokens.
    pub(crate) ctx: usize,
    /// Whether any matrix reads the per-256 activation layout (S4.9a).
    pub(crate) k_layout: bool,
}

/// The family's scratch, sized once by [`Architecture::new_scratch`]: the fused
/// q/k/v projection's output, `3·d_model` wide per row.
pub(crate) struct NomicBertScratch {
    /// The fused q/k/v output in the attention half, then `ffn_up`'s rows in the
    /// feed-forward half; the wider of the two sizes the buffer.
    buf: Vec<f32>,
    width: usize,
}

impl NomicBertScratch {
    fn new(max_batch: usize, width: usize) -> NomicBertScratch {
        NomicBertScratch {
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

/// A vector weight (a norm's weight or bias), dequantised.
fn vec_f32(g: &Gguf, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let t = g.tensor(name)?;
    if t.numel() as usize != n {
        return Err(format!("{name} has {} entries, expected {n}", t.numel()));
    }
    let mut v = vec![0.0f32; n];
    peregrine_kernels::quant::dequant_row(t.kind, g.bytes(t), &mut v);
    Ok(v)
}

impl NomicBert {
    #[allow(clippy::too_many_lines)]
    fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        let arch = g.str("general.architecture")?;
        if arch != "nomic-bert" {
            return Err(format!(
                "expected architecture \"nomic-bert\", found {arch:?} — nomic-bert-moe is not \
                 supported yet (roadmap S4.8b); bert and modern-bert are other families"
            ));
        }
        let k = |suffix: &str| format!("nomic-bert.{suffix}");
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
        // llama.cpp creates the fused `attn_qkv` as `[n_embd, 3·n_embd]`: a nomic-bert
        // file has no separate query width and no grouped-query attention.
        if head_dim * n_head != d_model || head_dim * n_kv_head != d_model {
            return Err(format!(
                "a nomic-bert attention head is d_model/n_head wide and keys and values are \
                 the same width as queries: {n_head} heads of {head_dim}, {n_kv_head} kv heads, \
                 {d_model} wide"
            ));
        }
        let qd = n_head * head_dim;
        let kvd = n_kv_head * head_dim;
        let qkv_width = qd + 2 * kvd;

        let ctx = g.u32(&k("context_length")).map_err(|_| {
            "nomic-bert model needs nomic-bert.context_length (its longest text)".to_string()
        })? as usize;
        if ctx == 0 {
            return Err("nomic-bert.context_length is 0".into());
        }
        let eps = g.f32(&k("attention.layer_norm_epsilon")).unwrap_or(1e-5);
        // NEOX pairing over `rope.dimension_count` (the head width by default):
        // `llama_model_rope_type` lists NOMIC_BERT in the offset-by-`n_rot/2` group,
        // and `hparams.n_rot_full` defaults to the head width.
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

        // llama.cpp's loader throws when the file does not define a token type
        // count; the graph then adds row 0 of `token_types.weight` to every token.
        let n_token_types = g.u32("tokenizer.ggml.token_type_count").map_err(|_| {
            "nomic-bert model needs to define token type count \
             (tokenizer.ggml.token_type_count)"
                .to_string()
        })? as usize;
        if n_token_types == 0 {
            return Err("nomic-bert model needs to define token type count".into());
        }

        let tok_embd = g.tensor("token_embd.weight")?.clone();
        let n_vocab = g.u32(&k("vocab_size")).map(|v| v as usize).ok();
        if let Some(n) = n_vocab {
            if n != tk.n_vocab {
                return Err(format!(
                    "metadata says {n} vocabulary entries but the tokenizer has {}",
                    tk.n_vocab
                ));
            }
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
        shape(&tok_embd, d_model, tk.n_vocab)?;

        let type_row0 = match g.tensor("token_types.weight") {
            Ok(t) => {
                let cols = t.dims.first().copied().unwrap_or(0) as usize;
                // A single token type is written as a 1-d `[n_embd]` tensor.
                let rows = t.dims.get(1).copied().unwrap_or(1) as usize;
                if cols != d_model || rows != n_token_types || t.dims.len() > 2 {
                    return Err(format!(
                        "token_types.weight is {:?}, expected [{d_model}, {n_token_types}]",
                        t.dims
                    ));
                }
                let rb = peregrine_core::quant::row_bytes(t.kind, cols);
                let mut v = vec![0.0f32; cols];
                peregrine_kernels::quant::dequant_row(t.kind, &g.bytes(t)[..rb], &mut v);
                Some(v)
            }
            Err(_) => None,
        };

        let tok_norm_w = vec_f32(g, "token_embd_norm.weight", d_model)?;
        let tok_norm_b = vec_f32(g, "token_embd_norm.bias", d_model)?;

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let wqkv = Linear::Dense(g.tensor(&p("attn_qkv.weight"))?.clone());
            let wo = Linear::Dense(g.tensor(&p("attn_output.weight"))?.clone());
            let gate = Linear::Dense(g.tensor(&p("ffn_gate.weight"))?.clone());
            let up = Linear::Dense(g.tensor(&p("ffn_up.weight"))?.clone());
            let down = Linear::Dense(g.tensor(&p("ffn_down.weight"))?.clone());
            for (lin, d_in, d_out, what) in [
                (&wqkv, d_model, qkv_width, "attn_qkv"),
                (&wo, qd, d_model, "attn_output"),
                (&gate, d_model, n_ff, "ffn_gate"),
                (&up, d_model, n_ff, "ffn_up"),
                (&down, n_ff, d_model, "ffn_down"),
            ] {
                if let Some(t) = lin.dense() {
                    shape(t, d_in, d_out).map_err(|e| format!("blk.{i}.{what}: {e}"))?;
                }
            }
            // Biases are `TENSOR_NOT_REQUIRED` in llama.cpp and absent from
            // nomic-embed-text v1.5's files (the converter asserts it). The MoE
            // variant, which is not served yet, carries them.
            let bqkv = match g.tensor(&p("attn_qkv.bias")) {
                Ok(_) => vec_f32(g, &p("attn_qkv.bias"), qkv_width)?,
                Err(_) => vec![0.0; qkv_width],
            };
            let bo = match g.tensor(&p("attn_output.bias")) {
                Ok(_) => vec_f32(g, &p("attn_output.bias"), d_model)?,
                Err(_) => vec![0.0; d_model],
            };
            layers.push(NomicBertLayer {
                wqkv,
                wo,
                bqkv,
                bo,
                attn_norm_w: vec_f32(g, &p("attn_output_norm.weight"), d_model)?,
                attn_norm_b: vec_f32(g, &p("attn_output_norm.bias"), d_model)?,
                gate,
                up,
                down,
                out_norm_w: vec_f32(g, &p("layer_output_norm.weight"), d_model)?,
                out_norm_b: vec_f32(g, &p("layer_output_norm.bias"), d_model)?,
            });
        }

        let cfg = Config {
            arch_name: "nomic-bert",
            // `llama_model_rope_type` lists NOMIC_BERT in the NeoX group.
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
            linears.extend([&l.wqkv, &l.wo, &l.gate, &l.up, &l.down]);
        }
        let k_layout = linears
            .iter()
            .filter_map(|lin| lin.dense())
            .any(|t| t.kind.reads_k_layout());

        Ok(Loaded {
            cfg,
            arch: Box::new(NomicBert {
                tok_embd,
                type_row0,
                tok_norm_w,
                tok_norm_b,
                layers,
                ctx,
                k_layout,
            }),
            rope_factors: None,
            tiles: TileStore::default(),
        })
    }
}

/// LayerNorm every row of the batch into `out` (weight and bias), then fill each
/// row's activation while the row is still in cache (`acts[r] = out row r`).
#[allow(clippy::too_many_arguments)]
fn norm_rows_fill(
    pool: &Pool,
    stats: &mut StepStats,
    x: &mut [f32],
    w: &[f32],
    b: &[f32],
    eps: f32,
    dim: usize,
    nb: usize,
    out: &mut [f32],
    acts: &mut [peregrine_core::quant::QuantAct],
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    for_row_chunks_3(
        pool,
        parallel,
        x,
        dim,
        out,
        dim,
        acts,
        1,
        nb,
        Sched::Static,
        |_wk, _rows, xc, oc, ac| {
            let n = oc.len() / dim;
            peregrine_core::vmath::layer_norm_rows(xc, w, b, eps, dim, n, oc);
            for (i, a) in ac.iter_mut().enumerate() {
                a.fill(&oc[i * dim..(i + 1) * dim]);
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

/// The SwiGLU pair: `out = silu(gate) ⊙ up`, one row per token.
fn swiglu_compact(
    pool: &Pool,
    stats: &mut StepStats,
    gate: &mut [f32],
    up: &[f32],
    out: &mut [f32],
    n_ff: usize,
    nb: usize,
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    for_row_chunks_2(
        pool,
        parallel,
        gate,
        n_ff,
        out,
        n_ff,
        nb,
        Sched::Static,
        |_wk, rows, gc, oc| {
            for (i, r) in rows.enumerate() {
                let g = &mut gc[i * n_ff..(i + 1) * n_ff];
                let u = &up[r * n_ff..(r + 1) * n_ff];
                peregrine_core::vmath::swiglu(g, u);
                oc[i * n_ff..(i + 1) * n_ff].copy_from_slice(g);
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

/// `q[row][i] *= scale` over the pool (S4.9u). Rows are independent, so which thread
/// scales a row cannot change its bits; the multiply is the serial loop's own.
fn qscale_rows_par(
    pool: &Pool,
    stats: &mut StepStats,
    q: &mut [f32],
    qd: usize,
    nb: usize,
    scale: f32,
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    peregrine_core::blocks::par::for_row_chunks(
        pool,
        parallel,
        q,
        qd,
        nb,
        Sched::Static,
        |_wk, _rows, qc| {
            for v in qc.iter_mut() {
                *v *= scale;
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

/// `out[row][i] += residual[row][i]` over the pool (S4.9u): the layer's second
/// residual reads the same row on the same worker, and the add is the serial loop's.
fn add_residual_par(
    pool: &Pool,
    stats: &mut StepStats,
    out: &mut [f32],
    residual: &mut [f32],
    dim: usize,
    nb: usize,
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    for_row_chunks_2(
        pool,
        parallel,
        out,
        dim,
        residual,
        dim,
        nb,
        Sched::Static,
        |_wk, _rows, oc, rc| {
            for (o, &r) in oc.iter_mut().zip(rc.iter()) {
                *o += r;
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

/// `out[row][i] = (out[row][i] + bias[i]) + residual[row][i]` over the pool
/// (S4.9u). One row's bias walk and residual walk stay on one worker; the operation
/// order is the serial walk's, and an all-zero bias still skips so a `-0.0` stays.
fn add_bias_residual_par(
    pool: &Pool,
    stats: &mut StepStats,
    out: &mut [f32],
    residual: &mut [f32],
    bias: &[f32],
    dim: usize,
    nb: usize,
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    let has_bias = !bias.iter().all(|&b| b == 0.0);
    for_row_chunks_2(
        pool,
        parallel,
        out,
        dim,
        residual,
        dim,
        nb,
        Sched::Static,
        |_wk, _rows, oc, rc| {
            if has_bias {
                for (row, res) in oc.chunks_mut(dim).zip(rc.chunks_exact(dim)) {
                    for ((o, &b), &r) in row.iter_mut().zip(bias).zip(res) {
                        *o = (*o + b) + r;
                    }
                }
            } else {
                for (o, &r) in oc.iter_mut().zip(rc.iter()) {
                    *o += r;
                }
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

impl Architecture for NomicBert {
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
        Some(Box::new(NomicBertScratch::new(
            max_batch,
            (cfg.q_dim() + 2 * cfg.kv_dim()).max(cfg.n_ff),
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
            for lin in [&mut l.wqkv, &mut l.wo, &mut l.gate, &mut l.up, &mut l.down] {
                f(lin)?;
            }
        }
        Ok(())
    }

    fn linear_groups(&self, f: &mut dyn FnMut(&[&Linear])) {
        for l in &self.layers {
            f(&[&l.wqkv]);
            f(&[&l.wo]);
            // The gate and up projections read the same rows, so they are one group.
            f(&[&l.gate, &l.up]);
            f(&[&l.down]);
        }
    }

    /// One bidirectional pass over whole texts (design §2.2).
    ///
    /// The transcribed llama.cpp graph (`src/models/bert.cpp`, the
    /// `LLM_ARCH_NOMIC_BERT` branches), per text and in order:
    ///
    /// 1. `x = token_embd[token] + token_types[0]` (no position table);
    /// 2. `token_embd_norm`: LayerNorm with weight and bias;
    /// 3. per layer, post-norm:
    ///    a. the fused `attn_qkv` (+ bias when present), split into q/k/v;
    ///    b. RoPE (NeoX pairing, `rope.freq_base`) on q and k, then attention over
    ///    the whole text at `1/√head_dim`;
    ///    c. `attn_output` (+ bias); `x = x + that`; `attn_output_norm`;
    ///    d. `ffn_gate` through SiLU times `ffn_up`, then `ffn_down`; `x = x + that`;
    ///    `layer_output_norm` → the next layer's input;
    /// 4. pooling (the caller's type) of the last layer's rows, then the engine's
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
        wants: &[peregrine_core::arch::TextWant],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let t_fwd = Instant::now();
        if wants
            .iter()
            .any(|w| matches!(w, peregrine_core::arch::TextWant::Head))
        {
            return Err(format!(
                "the {} model has no classifier head",
                cfg.arch_name
            ));
        }
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
                    "text {i} has {} tokens; the nomic-bert model's context_length is {}",
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

        // ---- embeddings: the token rows and token type 0 ----
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
                if let Some(t0) = &self.type_row0 {
                    for (x, &b) in st.x[row * dm..(row + 1) * dm].iter_mut().zip(t0) {
                        *x += b;
                    }
                }
                row += 1;
            }
        }
        // `token_embd_norm` feeds the first block's fused q/k/v.
        norm_rows_fill(
            pool,
            stats,
            &mut st.x,
            &self.tok_norm_w,
            &self.tok_norm_b,
            cfg.eps,
            dm,
            nb,
            &mut st.xb,
            &mut st.acts,
        );
        st.x[..nb * dm].copy_from_slice(&st.xb[..nb * dm]);

        if st.fam.is_none() {
            st.fam = Some(Box::new(NomicBertScratch::new(
                st.max_batch,
                qkv_width.max(ff),
            )));
        }
        let scratch = st
            .fam
            .as_mut()
            .and_then(|f| f.downcast_mut::<NomicBertScratch>())
            .ok_or("the state has no nomic-bert scratch")?;
        scratch.ensure(nb);

        for l in &self.layers {
            // ---- the fused q/k/v projection, one weight pass ----
            {
                let mut jobs = [RowsOut {
                    lin: &l.wqkv,
                    out: peregrine_core::blocks::par::Disjoint::new(&mut scratch.buf),
                    out_dim: qkv_width,
                }];
                linear_rows_shared(
                    w,
                    pool,
                    stats,
                    &mut jobs,
                    &st.acts,
                    &mut st.lb_x,
                    &mut st.lb_t,
                    &st.x,
                    dm,
                    nb,
                    &mut st.kq,
                );
            }

            // ---- split the fused rows into q, k and v, then the fused bias ----
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

            // ---- RoPE (NeoX pairing) on q and k ----
            // The fused bias is applied inside the same fork-join, row by row, before
            // the rotation (S4.9u): `q += bq` (and k, v) then `rope(q)` is the order
            // the serial bias walk ran in, and a row's bits do not depend on which
            // worker owns it.
            let has_b = l.bqkv.iter().any(|&b| b != 0.0);
            let (bq, bk, bv) = if has_b {
                (
                    Some(&l.bqkv[..qd]),
                    Some(&l.bqkv[qd..qd + kvd]),
                    Some(&l.bqkv[qd + kvd..]),
                )
            } else {
                (None, None, None)
            };
            peregrine_core::blocks::attn::prepare_rows(
                cfg,
                bq,
                bk,
                bv,
                None,
                None,
                &st.rope,
                &mut st.q,
                &mut st.k,
                &mut st.v,
                nb,
                &pos[..nb],
                pool,
                stats,
            );
            // llama.cpp passes `kq_scale = 1/√head_dim` to the attention; folding it
            // into Q is the same product and keeps the block's interface. The scale
            // runs after the rotation (the product's own rounding order), on every
            // row: rows are independent, so the pool may split them.
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            qscale_rows_par(pool, stats, &mut st.q[..nb * qd], qd, nb, scale);

            // ---- attention: one n × n block per text, no window ----
            peregrine_core::blocks::bidir::attend_texts(
                cfg,
                &st.q,
                &st.k,
                &st.v,
                &mut st.xb,
                &spans,
                None,
                None,
                1.0,
                &mut st.attn.bidir,
                pool,
                stats,
            );

            // ---- attn_output (+ bias), residual, attn_output_norm ----
            // `linear_rows` fills the activation rows from its input, which is the
            // attention output here (the last fill was the q/k/v input).
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
            add_bias_residual_par(
                pool,
                stats,
                &mut st.xb2[..nb * dm],
                &mut st.x[..nb * dm],
                &l.bo,
                dm,
                nb,
            );
            norm_rows_fill(
                pool,
                stats,
                &mut st.xb2,
                &l.attn_norm_w,
                &l.attn_norm_b,
                cfg.eps,
                dm,
                nb,
                &mut st.xb,
                &mut st.acts,
            );

            // ---- the SwiGLU pair: gate and up read the same normed rows ----
            {
                let mut jobs = [
                    RowsOut {
                        lin: &l.gate,
                        out: peregrine_core::blocks::par::Disjoint::new(&mut st.hb),
                        out_dim: ff,
                    },
                    RowsOut {
                        lin: &l.up,
                        out: peregrine_core::blocks::par::Disjoint::new(&mut scratch.buf),
                        out_dim: ff,
                    },
                ];
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
                );
            }
            swiglu_compact(
                pool,
                stats,
                &mut st.hb,
                &scratch.buf[..nb * ff],
                &mut st.hb2,
                ff,
                nb,
            );

            // ---- ffn_down, the second residual, layer_output_norm ----
            {
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
                    &mut st.xb2,
                    dm,
                    &mut st.kq,
                );
            }
            add_residual_par(
                pool,
                stats,
                &mut st.xb2[..nb * dm],
                &mut st.xb[..nb * dm],
                dm,
                nb,
            );
            norm_rows_fill(
                pool,
                stats,
                &mut st.xb2,
                &l.out_norm_w,
                &l.out_norm_b,
                cfg.eps,
                dm,
                nb,
                &mut st.x,
                &mut st.acts,
            );
        }

        // ---- pooling, text by text (the vector this text's slot reads) ----
        if st.pool_sum.len() < dm {
            st.pool_sum.resize(dm, 0.0);
        }
        for (i, (t, s)) in texts.iter().zip(&spans).enumerate() {
            let row0 = s.row0;
            peregrine_core::embed::pool_rows_into(
                &st.x[row0 * dm..(row0 + t.len()) * dm],
                dm,
                match wants[i] {
                    peregrine_core::arch::TextWant::Pooled(p) => p,
                    peregrine_core::arch::TextWant::Head => unreachable!("refused above"),
                },
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
