//! The `nomic-bert` family and its mixture-of-experts sibling (roadmap S4.8a, S4.8b).
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
//! `architecture = "nomic-bert-moe"` is the **same graph** in llama.cpp — the same
//! file, `src/models/bert.cpp`, with two branch points (`LLM_ARCH_NOMIC_BERT_MOE`):
//!
//! - every layer whose index satisfies `il % moe_every_n_layers == 1` (the odd
//!   layers of nomic-embed-text-v2-moe, whose file says 2) replaces the feed-forward
//!   with a **mixture of experts**: a `ffn_gate_inp` router scores `expert_count`
//!   experts, a softmax over all of them gives the weights, the best
//!   `expert_used_count` are run, each as a **plain GELU** `ffn_down_exps` of
//!   `gelu(ffn_up_exps)` (no gate matrix), and their outputs are summed in the
//!   router's order (`src/models/bert.cpp:171-184`, `src/llama-graph.cpp:2227-2633`);
//! - every other layer is a **plain GELU** feed-forward, not the SwiGLU pair:
//!   `ffn_down · gelu(ffn_up · x + b_up) + b_down` (`src/models/bert.cpp:185-192`).
//!
//! Its file also carries the projection biases (`attn_qkv.bias`, `attn_output.bias`)
//! and the two FFN biases, and its tokenizer is the XLM-R SentencePiece-unigram
//! vocabulary (`tokenizer.ggml.model = "t5"`), which [`peregrine_core::tokenizer`]
//! already serves for `bge-m3`.
//!
//! A file of either family is an encoder: the engine refuses chat and label requests
//! on it by name. The pooling is the file's own `<arch>.pooling_type` (mean, 1, for
//! both published models), and `/v1/embeddings` truncates to `dimensions` and
//! normalises after it, as for every encoder.
//!
//! The two families share this one file because the graph is the same file in the
//! reference, and the attention half, the tensor list, the RoPE, the LayerNorms and
//! the pooling are identical: a layer's feed-forward is the only branch, so it is an
//! enum (`NomicFfn`) rather than a second copy of the pass. The public reading is
//! [`docs/reference/nomic-bert.md`] and [`docs/reference/nomic-bert-moe.md`].
//!
//! [`docs/reference/nomic-bert.md`]: ../../../../docs/reference/nomic-bert.md
//! [`docs/reference/nomic-bert-moe.md`]: ../../../../docs/reference/nomic-bert-moe.md

use std::time::Instant;

use peregrine_core::arch::{ActLayout, Architecture, Causality, Family, Loaded};
use peregrine_core::blocks::moe::{
    expert_view, moe_ffn, MoeAct, MoeGating, MoeScratch, MoeWeights,
};
use peregrine_core::blocks::par::{for_row_chunks_2, for_row_chunks_3, Sched};
use peregrine_core::blocks::ROW_PARALLEL;
use peregrine_core::gguf::{Format, Gguf, TensorInfo};
use peregrine_core::model::{Config, Linear, State, StepStats, TileStore, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;

use crate::encoder::{linear_rows, linear_rows_shared, RowsOut};

/// The `nomic-bert` family, registered as `nomic-bert`.
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

/// The `nomic-bert-moe` family, registered as `nomic-bert-moe` (S4.8b).
///
/// It is the same graph with a mixture-of-experts feed-forward on every
/// `moe_every_n_layers`-th layer (the odd ones when the interval is 2) and a plain
/// GELU feed-forward on the rest (`src/models/bert.cpp:171-192`); the loader reads
/// both shapes, so this family does not need a second pass.
pub struct NomicBertMoeFamily;

impl Family for NomicBertMoeFamily {
    fn names(&self) -> &'static [&'static str] {
        &["nomic-bert-moe"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        NomicBert::load(g, tk)
    }

    /// A `nomic-bert-moe` file is an encoder too: the engine refuses chat and label
    /// requests on it by name.
    fn embedding_only(&self) -> bool {
        true
    }

    /// The Q8_0 file is dequantised into f16 panels at load (S4.8b).
    ///
    /// This family's arithmetic is more sensitive to the int8 activation path than a
    /// dense encoder's: a mixture of experts makes a discrete choice per row, and the
    /// *mean* pooling of a bidirectional pass turns an error shared by many rows into a
    /// large one in the pooled vector (the reference note has the measurements). f16
    /// panels hold the whole model in 566 MB — half the f32 form the small `bert`
    /// encoders take — and remove the int8 activations, so what is left of the file's
    /// distance to the F16 conversion is its weights' quantisation and no more.
    fn q8_panels(&self) -> peregrine_core::fpack::Q8Panels {
        peregrine_core::fpack::Q8Panels::Half
    }

    /// The expert banks take the float pack, one matrix per expert (S4.8b).
    ///
    /// A mixture-of-experts layer runs two of eight experts per row, and the rows a
    /// pass routes to one expert are a contiguous group, so the experts are ordinary
    /// GEMMs of `n_ff` rows — the same shape as the plain layer's feed-forward. Left
    /// unpacked they are the only matmuls of the family that cannot reach the packed
    /// float kernel; packed per expert they take the same path as everything else.
    /// Only this family opts in: every other mixture's bytes stay where they were.
    fn packs_experts(&self) -> bool {
        true
    }
}

/// One layer's feed-forward: the branch that separates the two families.
pub(crate) enum NomicFfn {
    /// `nomic-bert`: `silu(ffn_gate · x) ⊙ (ffn_up · x)`, then `ffn_down`
    /// (`src/models/bert.cpp:202-208`, the `else` branch, every bias NULL).
    SwiGlu {
        gate: Linear,
        up: Linear,
        down: Linear,
    },
    /// `nomic-bert-moe` on a plain layer: `ffn_down · gelu(ffn_up · x + b_up) + b_down`
    /// (`src/models/bert.cpp:185-192`, `LLM_FFN_GELU` and `LLM_FFN_SEQ`).
    Gelu {
        up: Linear,
        down: Linear,
        /// `ffn_up.bias` / `ffn_down.bias`, `TENSOR_NOT_REQUIRED` in llama.cpp
        /// (`src/models/nomic-bert-moe.cpp:43-45`); the published file has both.
        up_b: Option<Vec<f32>>,
        down_b: Option<Vec<f32>>,
    },
    /// `nomic-bert-moe` on a mixture-of-experts layer: the router and the expert
    /// banks, as 2D views of the file's 3D tensors (`src/models/bert.cpp:171-184`,
    /// `src/models/nomic-bert-moe.cpp:37-40`).
    Moe {
        /// `ffn_gate_inp.weight`: `[d_model, n_expert]`.
        router: TensorInfo,
        /// `ffn_up_exps.weight`: one `[d_model, n_ff]` view per expert.
        up_exps: Vec<TensorInfo>,
        /// `ffn_down_exps.weight`: one `[n_ff, d_model]` view per expert.
        down_exps: Vec<TensorInfo>,
    },
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
    /// The layer's feed-forward: the family's one branch.
    pub(crate) ffn: NomicFfn,
    /// `layer_output_norm`: the post-FFN LayerNorm, weight and bias.
    pub(crate) out_norm_w: Vec<f32>,
    pub(crate) out_norm_b: Vec<f32>,
}

/// The mixture of experts' geometry, when a `nomic-bert-moe` file has MoE layers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeCfg {
    /// `expert_count`: the experts the router chooses from.
    pub(crate) n_expert: usize,
    /// `expert_used_count`: the experts one row uses.
    pub(crate) n_used: usize,
}

/// A loaded `nomic-bert` or `nomic-bert-moe` model.
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
    /// The mixture of experts, when the file has any MoE layer.
    pub(crate) moe: Option<MoeCfg>,
    /// Whether the MoE block's activations keep their f32 values: true when every
    /// expert matrix is F32/F16/BF16 (`MoeScratch::set_keep_f32`).
    pub(crate) moe_keep_f32: bool,
    /// Whether any matrix reads the per-256 activation layout (S4.9a).
    pub(crate) k_layout: bool,
}

/// The family's scratch, sized once by [`Architecture::new_scratch`]: the fused
/// q/k/v projection's output, `3·d_model` wide per row, and the mixture-of-experts
/// block's buffers when the file has MoE layers.
pub(crate) struct NomicBertScratch {
    /// The fused q/k/v output in the attention half, then `ffn_up`'s rows in the
    /// feed-forward half; the wider of the two sizes the buffer.
    buf: Vec<f32>,
    width: usize,
    moe: Option<MoeScratch>,
}

impl NomicBertScratch {
    fn new(
        cfg: &Config,
        max_batch: usize,
        width: usize,
        moe: Option<MoeCfg>,
        moe_keep_f32: bool,
    ) -> NomicBertScratch {
        let moe = moe.map(|m| {
            let mut scratch =
                MoeScratch::new(max_batch, m.n_expert, m.n_used, cfg.d_model, cfg.n_ff);
            scratch.set_keep_f32(moe_keep_f32);
            scratch
        });
        NomicBertScratch {
            buf: vec![0.0; max_batch * width],
            width,
            moe,
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

/// An optional vector weight: `None` when llama.cpp marks it `TENSOR_NOT_REQUIRED`
/// and the file has no tensor.
fn opt_vec_f32(g: &Gguf, name: &str, n: usize) -> Result<Option<Vec<f32>>, String> {
    match g.tensor(name) {
        Ok(_) => Ok(Some(vec_f32(g, name, n)?)),
        Err(_) => Ok(None),
    }
}

/// The `n_expert` 2D views of one 3D expert bank, checked against the metadata.
fn expert_bank(
    g: &Gguf,
    name: &str,
    d_in: usize,
    d_out: usize,
    n_expert: usize,
) -> Result<Vec<TensorInfo>, String> {
    let t = g.tensor(name)?;
    if t.dims.len() != 3
        || t.dims[0] as usize != d_in
        || t.dims[1] as usize != d_out
        || t.dims[2] as usize != n_expert
    {
        return Err(format!(
            "{name} is {:?}, the metadata implies [{d_in}, {d_out}, {n_expert}]",
            t.dims
        ));
    }
    (0..n_expert)
        .map(|e| expert_view(t, e).map_err(|err| format!("{name}: {err}")))
        .collect()
}

/// Whether a matrix is read as floats (F32/F16/BF16, or a packed float form).
fn float_matrix(t: &TensorInfo) -> bool {
    t.fp_kind.is_some() || matches!(t.kind, Format::F32 | Format::F16 | Format::BF16)
}

impl NomicBert {
    #[allow(clippy::too_many_lines)]
    fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        let arch = g.str("general.architecture")?;
        let is_moe_arch = match arch {
            "nomic-bert" => false,
            "nomic-bert-moe" => true,
            other => {
                return Err(format!(
                    "expected architecture \"nomic-bert\" or \"nomic-bert-moe\", found \
                     {other:?} — bert, modern-bert and jina-bert-v2 are other families"
                ))
            }
        };
        let k = |suffix: &str| format!("{arch}.{suffix}");
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
        if n_layer == 0 || n_ff == 0 || n_head == 0 || n_kv_head == 0 || d_model == 0 {
            return Err(format!(
                "implausible shape: {n_layer} layers, {n_head}/{n_kv_head} heads, {n_ff} \
                 feed-forward"
            ));
        }
        // llama.cpp creates the fused `attn_qkv` as `[n_embd, 3·n_embd]`: a nomic-bert
        // file has no separate query width and no grouped-query attention.
        if head_dim * n_head != d_model || head_dim * n_kv_head != d_model {
            return Err(format!(
                "a {arch} attention head is d_model/n_head wide and keys and values are \
                 the same width as queries: {n_head} heads of {head_dim}, {n_kv_head} kv heads, \
                 {d_model} wide"
            ));
        }
        let qd = n_head * head_dim;
        let kvd = n_kv_head * head_dim;
        let qkv_width = qd + 2 * kvd;

        let ctx = g
            .u32(&k("context_length"))
            .map_err(|_| format!("{arch} model needs {arch}.context_length (its longest text)"))?
            as usize;
        if ctx == 0 {
            return Err(format!("{arch}.context_length is 0"));
        }
        let eps = g.f32(&k("attention.layer_norm_epsilon")).unwrap_or(1e-5);
        // NEOX pairing over `rope.dimension_count` (the head width by default):
        // `llama_model_rope_type` lists NOMIC_BERT and NOMIC_BERT_MOE in the
        // offset-by-`n_rot/2` group, and `hparams.n_rot_full` defaults to the head width.
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
            format!(
                "{arch} model needs to define token type count \
                 (tokenizer.ggml.token_type_count)"
            )
        })? as usize;
        if n_token_types == 0 {
            return Err(format!("{arch} model needs to define token type count"));
        }

        // The mixture of experts' geometry. llama.cpp reads
        // `moe_every_n_layers` (default 0, which means no MoE layer) and puts the
        // experts on `il % moe_every_n_layers == 1` (`src/models/bert.cpp:171`).
        let moe_every = if is_moe_arch {
            g.u32(&k("moe_every_n_layers")).unwrap_or(0) as usize
        } else {
            0
        };
        let is_moe_layer = |i: usize| moe_every > 0 && i % moe_every == 1;
        let moe = if (0..n_layer).any(is_moe_layer) {
            let n_expert = g
                .u32(&k("expert_count"))
                .map_err(|_| format!("{arch}: a MoE layer needs {arch}.expert_count"))?
                as usize;
            if n_expert == 0 {
                return Err(format!("{arch}.expert_count is 0"));
            }
            let n_used = g
                .u32(&k("expert_used_count"))
                .map_err(|_| format!("{arch}: a MoE layer needs {arch}.expert_used_count"))?
                as usize;
            if n_used == 0 || n_used > n_expert {
                return Err(format!(
                    "{arch}.expert_used_count = {n_used}: a row uses 1..={n_expert} of \
                     {n_expert} experts"
                ));
            }
            // The MoE block quantises activations in 32-value blocks, as every
            // encoder width does (the qwen3moe loader checks the same widths).
            for (what, n) in [("embedding length", d_model), ("feed-forward length", n_ff)] {
                if !n.is_multiple_of(32) {
                    return Err(format!(
                        "{arch}: {what} {n} is not a multiple of 32, which the mixture of \
                         experts needs"
                    ));
                }
            }
            Some(MoeCfg { n_expert, n_used })
        } else {
            None
        };

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
            for (lin, d_in, d_out, what) in [
                (&wqkv, d_model, qkv_width, "attn_qkv"),
                (&wo, qd, d_model, "attn_output"),
            ] {
                if let Some(t) = lin.dense() {
                    shape(t, d_in, d_out).map_err(|e| format!("blk.{i}.{what}: {e}"))?;
                }
            }
            // Biases are `TENSOR_NOT_REQUIRED` in llama.cpp. nomic-embed-text v1.5's
            // files have none; the MoE variant has the attention ones.
            let bqkv = match g.tensor(&p("attn_qkv.bias")) {
                Ok(_) => vec_f32(g, &p("attn_qkv.bias"), qkv_width)?,
                Err(_) => vec![0.0; qkv_width],
            };
            let bo = match g.tensor(&p("attn_output.bias")) {
                Ok(_) => vec_f32(g, &p("attn_output.bias"), d_model)?,
                Err(_) => vec![0.0; d_model],
            };

            let ffn = if is_moe_layer(i) {
                let m = moe.expect("a MoE layer has a MoE config");
                let router = g.tensor(&p("ffn_gate_inp.weight"))?.clone();
                shape(&router, d_model, m.n_expert)
                    .map_err(|e| format!("blk.{i}.ffn_gate_inp: {e}"))?;
                let up_exps = expert_bank(g, &p("ffn_up_exps.weight"), d_model, n_ff, m.n_expert)
                    .map_err(|e| format!("blk.{i}: {e}"))?;
                let down_exps =
                    expert_bank(g, &p("ffn_down_exps.weight"), n_ff, d_model, m.n_expert)
                        .map_err(|e| format!("blk.{i}: {e}"))?;
                NomicFfn::Moe {
                    router,
                    up_exps,
                    down_exps,
                }
            } else if is_moe_arch {
                // The plain layers of the MoE file: a sequential GELU MLP with the
                // optional biases (`src/models/bert.cpp:185-192`).
                let up = Linear::Dense(g.tensor(&p("ffn_up.weight"))?.clone());
                let down = Linear::Dense(g.tensor(&p("ffn_down.weight"))?.clone());
                for (lin, d_in, d_out, what) in [
                    (&up, d_model, n_ff, "ffn_up"),
                    (&down, n_ff, d_model, "ffn_down"),
                ] {
                    if let Some(t) = lin.dense() {
                        shape(t, d_in, d_out).map_err(|e| format!("blk.{i}.{what}: {e}"))?;
                    }
                }
                NomicFfn::Gelu {
                    up,
                    down,
                    up_b: opt_vec_f32(g, &p("ffn_up.bias"), n_ff)?,
                    down_b: opt_vec_f32(g, &p("ffn_down.bias"), d_model)?,
                }
            } else {
                let gate = Linear::Dense(g.tensor(&p("ffn_gate.weight"))?.clone());
                let up = Linear::Dense(g.tensor(&p("ffn_up.weight"))?.clone());
                let down = Linear::Dense(g.tensor(&p("ffn_down.weight"))?.clone());
                for (lin, d_in, d_out, what) in [
                    (&gate, d_model, n_ff, "ffn_gate"),
                    (&up, d_model, n_ff, "ffn_up"),
                    (&down, n_ff, d_model, "ffn_down"),
                ] {
                    if let Some(t) = lin.dense() {
                        shape(t, d_in, d_out).map_err(|e| format!("blk.{i}.{what}: {e}"))?;
                    }
                }
                NomicFfn::SwiGlu { gate, up, down }
            };

            layers.push(NomicBertLayer {
                wqkv,
                wo,
                bqkv,
                bo,
                attn_norm_w: vec_f32(g, &p("attn_output_norm.weight"), d_model)?,
                attn_norm_b: vec_f32(g, &p("attn_output_norm.bias"), d_model)?,
                ffn,
                out_norm_w: vec_f32(g, &p("layer_output_norm.weight"), d_model)?,
                out_norm_b: vec_f32(g, &p("layer_output_norm.bias"), d_model)?,
            });
        }

        // `MoeScratch::set_keep_f32`: the block's activation rows keep their f32
        // values when any of its matrices is float — llama.cpp's router is F32 even in
        // the Q8_0 file, and the qwen3moe family passes its global `keep_f32` here.
        let mut moe_keep_f32 = false;
        for l in &layers {
            if let NomicFfn::Moe {
                router,
                up_exps,
                down_exps,
            } = &l.ffn
            {
                for t in std::iter::once(router).chain(up_exps).chain(down_exps) {
                    if float_matrix(t) {
                        moe_keep_f32 = true;
                    }
                }
            }
        }

        let cfg = Config {
            arch_name: if is_moe_arch {
                "nomic-bert-moe"
            } else {
                "nomic-bert"
            },
            // `llama_model_rope_type` lists both architectures in the NeoX group.
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
        let mut matrices: Vec<&TensorInfo> = Vec::new();
        for l in &layers {
            linears.extend([&l.wqkv, &l.wo]);
            match &l.ffn {
                NomicFfn::SwiGlu { gate, up, down } => {
                    linears.extend([gate, up, down]);
                }
                NomicFfn::Gelu { up, down, .. } => {
                    linears.extend([up, down]);
                }
                NomicFfn::Moe {
                    router,
                    up_exps,
                    down_exps,
                } => {
                    matrices.push(router);
                    matrices.extend(up_exps);
                    matrices.extend(down_exps);
                }
            }
        }
        let k_layout = linears
            .iter()
            .filter_map(|lin| lin.dense())
            .chain(matrices)
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
                moe,
                moe_keep_f32,
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

/// The plain GELU feed-forward of a `nomic-bert-moe` layer: `x = gelu(x)` on the
/// `ffn_up` output, in place, and the activated rows for `ffn_down`.
fn gelu_fill(
    pool: &Pool,
    stats: &mut StepStats,
    x: &mut [f32],
    dim: usize,
    nb: usize,
    acts: &mut [peregrine_core::quant::QuantAct],
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    for_row_chunks_2(
        pool,
        parallel,
        x,
        dim,
        acts,
        1,
        nb,
        Sched::Static,
        |_wk, _rows, xc, ac| {
            for (i, a) in ac.iter_mut().enumerate() {
                let row = &mut xc[i * dim..(i + 1) * dim];
                for v in row.iter_mut() {
                    *v = peregrine_core::vmath::gelu_tanh(*v);
                }
                a.fill(row);
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

/// `out[row][i] += bias[i]` for every row of a projection's output.
#[inline]
fn add_bias(out: &mut [f32], dim: usize, nb: usize, bias: &[f32]) {
    for row in out.chunks_mut(dim).take(nb) {
        for (o, &b) in row.iter_mut().zip(bias) {
            *o += b;
        }
    }
}

/// `out[row][i] = (out[row][i] + bias[i]) + residual[row][i]`.
#[inline]
fn add_bias_residual(out: &mut [f32], residual: &[f32], bias: &[f32], dim: usize) {
    if bias.iter().all(|&b| b == 0.0) {
        for (o, &r) in out.iter_mut().zip(residual) {
            *o += r;
        }
        return;
    }
    for (row, res) in out.chunks_mut(dim).zip(residual.chunks_exact(dim)) {
        for ((o, &b), &r) in row.iter_mut().zip(bias).zip(res) {
            *o = (*o + b) + r;
        }
    }
}

/// [`add_bias_residual`] with an optional bias: llama.cpp's graph has no bias add
/// at all when the tensor is absent, which is a different f32 operation from adding
/// zero (`-0.0 + 0.0` is `+0.0`).
#[inline]
fn add_opt_bias_residual(out: &mut [f32], residual: &[f32], bias: Option<&[f32]>, dim: usize) {
    match bias {
        Some(b) => add_bias_residual(out, residual, b, dim),
        None => {
            for (o, &r) in out.iter_mut().zip(residual) {
                *o += r;
            }
        }
    }
}

impl Architecture for NomicBert {
    fn causality(&self) -> Causality {
        Causality::Bidirectional
    }

    fn act_layout(&self) -> ActLayout {
        let mut keep_f32 = false;
        let mut quantize = false;
        let mut note = |float: bool| {
            if float {
                keep_f32 = true;
            } else {
                quantize = true;
            }
        };
        self.linear_groups(&mut |lins| {
            for lin in lins {
                match lin.dense() {
                    Some(ti) => note(float_matrix(ti)),
                    None => note(false),
                }
            }
        });
        // The mixture of experts: the router and the expert banks are matrices too,
        // and the block's activations follow them.
        for l in &self.layers {
            if let NomicFfn::Moe {
                router,
                up_exps,
                down_exps,
            } = &l.ffn
            {
                for t in std::iter::once(router).chain(up_exps).chain(down_exps) {
                    note(float_matrix(t));
                }
            }
        }
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
            cfg,
            max_batch,
            (cfg.q_dim() + 2 * cfg.kv_dim()).max(cfg.n_ff),
            self.moe,
            self.moe_keep_f32,
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
            for lin in [&mut l.wqkv, &mut l.wo] {
                f(lin)?;
            }
            match &mut l.ffn {
                NomicFfn::SwiGlu { gate, up, down } => {
                    for lin in [gate, up, down] {
                        f(lin)?;
                    }
                }
                NomicFfn::Gelu { up, down, .. } => {
                    for lin in [up, down] {
                        f(lin)?;
                    }
                }
                // A LittleBit sidecar must not touch the expert tensors (they are 3D
                // views), so the MoE layers report no linear.
                NomicFfn::Moe { .. } => {}
            }
        }
        Ok(())
    }

    fn linear_groups(&self, f: &mut dyn FnMut(&[&Linear])) {
        for l in &self.layers {
            f(&[&l.wqkv]);
            f(&[&l.wo]);
            match &l.ffn {
                // The gate and up projections read the same rows, so they are one group.
                NomicFfn::SwiGlu { gate, up, down } => {
                    f(&[gate, up]);
                    f(&[down]);
                }
                NomicFfn::Gelu { up, down, .. } => {
                    f(&[up]);
                    f(&[down]);
                }
                NomicFfn::Moe { .. } => {}
            }
        }
    }

    /// One bidirectional pass over whole texts (design §2.2).
    ///
    /// The transcribed llama.cpp graph (`src/models/bert.cpp`, the
    /// `LLM_ARCH_NOMIC_BERT` and `LLM_ARCH_NOMIC_BERT_MOE` branches), per text and
    /// in order:
    ///
    /// 1. `x = token_embd[token] + token_types[0]` (no position table);
    /// 2. `token_embd_norm`: LayerNorm with weight and bias;
    /// 3. per layer, post-norm:
    ///    a. the fused `attn_qkv` (+ bias when present), split into q/k/v;
    ///    b. RoPE (NeoX pairing, `rope.freq_base`) on q and k, then attention over
    ///    the whole text at `1/√head_dim`;
    ///    c. `attn_output` (+ bias); `x = x + that`; `attn_output_norm`;
    ///    d. the feed-forward, which is the family's branch: `nomic-bert`'s SwiGLU
    ///    pair (`ffn_gate` through SiLU times `ffn_up`, then `ffn_down`); on a plain
    ///    `nomic-bert-moe` layer `gelu(ffn_up · x + b_up)` through `ffn_down` (+
    ///    `b_down`); on a MoE layer the router's softmax over every expert, the best
    ///    `expert_used_count`, each through `ffn_up_exps` and GELU and `ffn_down_exps`,
    ///    weighted and summed in the router's order;
    ///    e. `x = x + that`; `layer_output_norm` → the next layer's input;
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
                    "text {i} has {} tokens; the {} model's context_length is {}",
                    t.len(),
                    cfg.arch_name,
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
                cfg,
                st.max_batch,
                qkv_width.max(ff),
                self.moe,
                self.moe_keep_f32,
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
                if l.bqkv.iter().any(|&b| b != 0.0) {
                    add_bias(&mut st.q[..nb * qd], qd, nb, &l.bqkv[..qd]);
                    add_bias(&mut st.k[..nb * kvd], kvd, nb, &l.bqkv[qd..qd + kvd]);
                    add_bias(&mut st.v[..nb * kvd], kvd, nb, &l.bqkv[qd + kvd..]);
                }
            }

            // ---- RoPE (NeoX pairing) on q and k ----
            peregrine_core::blocks::attn::prepare_rows(
                cfg,
                None,
                None,
                None,
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
            // into Q is the same product and keeps the block's interface.
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            for q in &mut st.q[..nb * qd] {
                *q *= scale;
            }

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
            add_bias_residual(&mut st.xb2[..nb * dm], &st.x[..nb * dm], &l.bo, dm);
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

            // ---- the feed-forward: the family's one branch ----
            match &l.ffn {
                NomicFfn::SwiGlu { gate, up, down } => {
                    // The gate and up projections read the same normed rows.
                    {
                        let mut jobs = [
                            RowsOut {
                                lin: gate,
                                out: peregrine_core::blocks::par::Disjoint::new(&mut st.hb),
                                out_dim: ff,
                            },
                            RowsOut {
                                lin: up,
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
                    linear_rows(
                        w,
                        pool,
                        stats,
                        down,
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
                    for i in 0..nb * dm {
                        st.xb2[i] += st.xb[i];
                    }
                }
                NomicFfn::Gelu {
                    up,
                    down,
                    up_b,
                    down_b,
                } => {
                    linear_rows(
                        w,
                        pool,
                        stats,
                        up,
                        &mut st.acts,
                        &mut st.lb_x,
                        &mut st.lb_t,
                        &st.xb,
                        dm,
                        nb,
                        &mut st.hb,
                        ff,
                        &mut st.kq,
                    );
                    if let Some(b) = up_b {
                        add_bias(&mut st.hb[..nb * ff], ff, nb, b);
                    }
                    gelu_fill(pool, stats, &mut st.hb, ff, nb, &mut st.acts);
                    linear_rows(
                        w,
                        pool,
                        stats,
                        down,
                        &mut st.acts,
                        &mut st.lb_x,
                        &mut st.lb_t,
                        &st.hb,
                        ff,
                        nb,
                        &mut st.xb2,
                        dm,
                        &mut st.kq,
                    );
                    add_opt_bias_residual(
                        &mut st.xb2[..nb * dm],
                        &st.xb[..nb * dm],
                        down_b.as_deref(),
                        dm,
                    );
                }
                NomicFfn::Moe {
                    router,
                    up_exps,
                    down_exps,
                } => {
                    // The mixture-of-experts block routes each row (a softmax over
                    // every expert, the best `expert_used_count`), runs the experts
                    // it chose and adds their weighted outputs in the router's
                    // order; the result is the layer's feed-forward output. The
                    // activation rows are the normed `ffn_inp`.
                    let moe_scratch = scratch
                        .moe
                        .as_mut()
                        .ok_or("the state has no mixture-of-experts scratch")?;
                    let mw = MoeWeights {
                        router,
                        // nomic-bert-moe's experts have no gate matrix: plain GELU.
                        gate: &[],
                        up: up_exps,
                        down: down_exps,
                        n_used: self.moe.expect("a MoE layer has a MoE config").n_used,
                        // `build_moe_ffn(cur, ..., norm_w = false, ...)`: the weights
                        // are the router's raw probabilities (`src/models/bert.cpp:173-183`).
                        renorm: false,
                        // `hparams.expert_weights_scale`, which is 0 in the published
                        // file, and a scale of 0 or 1 is not applied.
                        w_scale: 0.0,
                        gating: MoeGating::Softmax,
                        act: MoeAct::Gelu,
                        router_bias: None,
                        expert_bias: None,
                    };
                    moe_ffn(
                        w,
                        &mw,
                        &st.xb[..nb * dm],
                        &mut st.xb2[..nb * dm],
                        nb,
                        moe_scratch,
                        pool,
                        stats,
                    );
                    for i in 0..nb * dm {
                        st.xb2[i] += st.xb[i];
                    }
                }
            }

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
