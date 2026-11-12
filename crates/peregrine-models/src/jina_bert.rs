//! The `jina-bert-v2` family: Jina's ALiBi encoder with a GEGLU feed-forward
//! (roadmap S4.8c).
//!
//! `architecture = "jina-bert-v2"` is what jina-embeddings-v2-base-en and -small-en
//! ship as. It looks like [`crate::bert`] and is not: there is **no position table**
//! (ALiBi biases each score by `slope · (−|i − j|)`), the query and key matrices may
//! carry their own LayerNorm (not on the reference files), the block has **one**
//! LayerNorm between the attention residual and the feed-forward (plus an optional
//! second), the feed-forward is a **GEGLU pair** (`ffn_gate` and `ffn_up`, gelu of
//! the gate times the up; some conversions fuse them into one `2·n_ff` matrix), every
//! projection has a bias, and the layer's own LayerNorm is its output — there is no
//! final output norm.
//!
//! Every detail here is transcribed from llama.cpp **v0.9.0 (`cd959ba`)**:
//! `src/models/bert.cpp` (the graph, shared with `bert`/`nomic-bert`, branched for
//! this family), `src/models/jina-bert-v2.cpp` (the hparams and the tensor list),
//! `src/llama-model.cpp` (the ALiBi flag), `src/llama-graph.cpp` and
//! `ggml/src/ggml-cpu/ops.cpp` (the bias) and `conversion/bert.py` (the converter).
//! [`docs/reference/jina-bert.md`] holds the reading, the tensor table and the
//! measured accuracy.
//!
//! [`docs/reference/jina-bert.md`]: ../../../../docs/reference/jina-bert.md

use std::time::Instant;

use peregrine_core::arch::{ActLayout, Architecture, Causality, Family, Loaded};
use peregrine_core::blocks::par::{for_row_chunks_2, for_row_chunks_3, Sched};
use peregrine_core::blocks::ROW_PARALLEL;
use peregrine_core::gguf::{Gguf, TensorInfo};
use peregrine_core::model::{Config, Linear, State, StepStats, TileStore, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::quant::QuantAct;
use peregrine_core::tokenizer::Tokenizer;
use peregrine_core::vmath;

use crate::encoder::{linear_rows_shared, RowsOut};

/// The family, registered as `jina-bert-v2`.
pub struct JinaBertFamily;

impl Family for JinaBertFamily {
    fn names(&self) -> &'static [&'static str] {
        &["jina-bert-v2"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        JinaBert::load(g, tk)
    }

    /// A `jina-bert-v2` file is an encoder: it has no decode path, so the engine
    /// refuses chat and label requests on it by name (design §3).
    fn embedding_only(&self) -> bool {
        true
    }
}

/// One encoder layer, every tensor read by llama.cpp's `LLAMA_ARCH_JINA_BERT_V2`
/// branch of `src/models/bert.cpp` (`load_arch_tensors` in
/// `src/models/jina-bert-v2.cpp`, lines 25-60).
pub(crate) struct JinaBertLayer {
    pub(crate) wq: Linear,
    pub(crate) wk: Linear,
    pub(crate) wv: Linear,
    pub(crate) wo: Linear,
    /// The projections' biases. Q/K/V are `TENSOR_NOT_REQUIRED` in llama.cpp, so a
    /// file without them gets zeros; the output projection's bias is required.
    pub(crate) bq: Vec<f32>,
    pub(crate) bk: Vec<f32>,
    pub(crate) bv: Vec<f32>,
    pub(crate) bo: Vec<f32>,
    /// `attn_q_norm` / `attn_k_norm`: a LayerNorm over one token's whole query (or
    /// key) row, before attention (`src/models/bert.cpp:115-129`). Absent on both
    /// reference files.
    pub(crate) q_norm: Option<(Vec<f32>, Vec<f32>)>,
    pub(crate) k_norm: Option<(Vec<f32>, Vec<f32>)>,
    /// `attn_out_norm`: the LayerNorm after the attention residual.
    pub(crate) attn_out_norm_w: Vec<f32>,
    pub(crate) attn_out_norm_b: Vec<f32>,
    /// `attn_norm_2`: an optional second residual-plus-norm after `attn_out_norm`
    /// (`src/models/bert.cpp:162-165`).
    pub(crate) attn_norm_2: Option<(Vec<f32>, Vec<f32>)>,
    /// `ffn_gate` and `ffn_up`: the GEGLU pair, gelu of the gate times the up. When
    /// the conversion fused them, `gate` is `None` and `up` has `2·n_ff` rows (the
    /// first half is the gate, `src/models/bert.cpp:194-200`).
    pub(crate) gate: Option<Linear>,
    pub(crate) up: Linear,
    pub(crate) up_b: Vec<f32>,
    pub(crate) down: Linear,
    pub(crate) down_b: Vec<f32>,
    /// `layer_output_norm`: the layer's last LayerNorm, whose output is the next
    /// layer's input (there is no final output norm in this family).
    pub(crate) out_norm_w: Vec<f32>,
    pub(crate) out_norm_b: Vec<f32>,
}

/// A loaded `jina-bert-v2` model.
pub struct JinaBert {
    pub(crate) tok_embd: TensorInfo,
    /// `token_types.weight` row 0: llama.cpp hardcodes token type 0 ("Sentence A"),
    /// so only that row is read (`src/models/bert.cpp:90-94`).
    pub(crate) type_row0: Vec<f32>,
    pub(crate) tok_norm_w: Vec<f32>,
    pub(crate) tok_norm_b: Vec<f32>,
    pub(crate) layers: Vec<JinaBertLayer>,
    /// ALiBi's per-head slopes (`f_max_alibi_bias = 8`, `src/models/jina-bert-v2.cpp:5`).
    pub(crate) slopes: Vec<f32>,
    /// `context_length`: the longest text, in tokens.
    pub(crate) ctx: usize,
    /// Whether a layer's `ffn_up` carries the gate in its first half.
    pub(crate) fused_gate: bool,
    /// Whether any matrix reads the per-256 activation layout (S4.9a).
    pub(crate) k_layout: bool,
}

/// The family's scratch, sized once by [`Architecture::new_scratch`] and only when a
/// file fuses its GEGLU pair (the reference files do not): the `2·n_ff`-wide up
/// projection output.
pub(crate) struct JinaScratch {
    buf: Vec<f32>,
    width: usize,
}

impl JinaScratch {
    fn new(max_batch: usize, width: usize) -> JinaScratch {
        JinaScratch {
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

/// An optional vector weight: zeros when the tensor is absent (llama.cpp marks these
/// `TENSOR_NOT_REQUIRED`).
fn opt_vec_f32(g: &Gguf, name: &str, n: usize) -> Result<Vec<f32>, String> {
    match g.tensor(name) {
        Ok(_) => vec_f32(g, name, n),
        Err(_) => Ok(vec![0.0; n]),
    }
}

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

impl JinaBert {
    #[allow(clippy::too_many_lines)]
    fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        let arch = g.str("general.architecture")?;
        if arch != "jina-bert-v2" {
            return Err(format!(
                "expected architecture \"jina-bert-v2\", found {arch:?} — bert, nomic-bert and \
                 modern-bert are other families (roadmap S4.6-S4.8)"
            ));
        }
        let k = |suffix: &str| format!("jina-bert-v2.{suffix}");
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
                 feed-forward, {d_model} wide"
            ));
        }
        if head_dim == 0 {
            return Err(format!(
                "an attention head of {head_dim} is not a width this family can read"
            ));
        }
        let qd = n_head * head_dim;
        let kvd = n_kv_head * head_dim;
        if n_head % n_kv_head != 0 {
            return Err(format!(
                "{n_head} query heads do not divide into {n_kv_head} key/value heads"
            ));
        }
        let ctx = g.u32(&k("context_length")).map_err(|_| {
            "jina-bert-v2 model needs jina-bert-v2.context_length (its longest text)".to_string()
        })? as usize;
        if ctx == 0 {
            return Err("jina-bert-v2.context_length is 0".into());
        }
        let eps = g.f32(&k("attention.layer_norm_epsilon")).unwrap_or(1e-12);
        if !(eps.is_finite() && eps >= 0.0) {
            return Err(format!(
                "jina-bert-v2.attention.layer_norm_epsilon is {eps}"
            ));
        }

        // llama.cpp's jina loader requires the token-type table and adds its row 0 to
        // every token (`src/models/jina-bert-v2.cpp:18`, `src/models/bert.cpp:91-94`).
        let type_row0 = {
            let t = g.tensor("token_types.weight").map_err(|_| {
                "a jina-bert-v2 file needs token_types.weight: llama.cpp adds row 0 \
                 (token type 0) to every token embedding"
                    .to_string()
            })?;
            if t.dims.first().copied().unwrap_or(0) as usize != d_model {
                return Err(format!(
                    "token_types.weight is {:?}, expected {d_model} per row",
                    t.dims
                ));
            }
            let mut row = vec![0.0f32; d_model];
            peregrine_kernels::quant::dequant_row(t.kind, g.bytes(t), &mut row);
            row
        };

        let tok_embd = g.tensor("token_embd.weight")?.clone();
        {
            let n_vocab = g.u32(&k("vocab_size")).map(|v| v as usize).ok();
            if let Some(n) = n_vocab {
                if n != tk.n_vocab {
                    return Err(format!(
                        "metadata says {n} vocabulary entries but the tokenizer has {}",
                        tk.n_vocab
                    ));
                }
            }
            shape_of(&tok_embd, d_model, tk.n_vocab)
                .map_err(|e| format!("token_embd.weight: {e}"))?;
        }
        let tok_norm_w = vec_f32(g, "token_embd_norm.weight", d_model)?;
        let tok_norm_b = vec_f32(g, "token_embd_norm.bias", d_model)?;

        let mut layers = Vec::with_capacity(n_layer);
        let mut fused_gate = false;
        for i in 0..n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let wq = Linear::Dense(g.tensor(&p("attn_q.weight"))?.clone());
            let wk = Linear::Dense(g.tensor(&p("attn_k.weight"))?.clone());
            let wv = Linear::Dense(g.tensor(&p("attn_v.weight"))?.clone());
            let wo = Linear::Dense(g.tensor(&p("attn_output.weight"))?.clone());
            for (lin, d_in, d_out, what) in [
                (&wq, d_model, qd, "attn_q"),
                (&wk, d_model, kvd, "attn_k"),
                (&wv, d_model, kvd, "attn_v"),
                (&wo, qd, d_model, "attn_output"),
            ] {
                if let Some(t) = lin.dense() {
                    shape_of(t, d_in, d_out).map_err(|e| format!("blk.{i}.{what}: {e}"))?;
                }
            }

            let up_t = g.tensor(&p("ffn_up.weight"))?.clone();
            let up_rows = up_t.dims.get(1).copied().unwrap_or(1) as usize;
            let (gate, up_width) = match g.tensor(&p("ffn_gate.weight")) {
                Ok(t) => {
                    shape_of(t, d_model, n_ff).map_err(|e| format!("blk.{i}.ffn_gate: {e}"))?;
                    if up_rows != n_ff {
                        return Err(format!(
                            "blk.{i}.ffn_up.weight has {up_rows} rows; with a separate \
                             ffn_gate.weight it must have feed_forward_length ({n_ff})"
                        ));
                    }
                    (Some(Linear::Dense(t.clone())), n_ff)
                }
                Err(_) => {
                    // Without a gate tensor, `ffn_up` carries the gate in its first half
                    // (`src/models/bert.cpp:194`). Anything else is not a GEGLU pair.
                    if up_rows != 2 * n_ff {
                        return Err(format!(
                            "blk.{i} has no ffn_gate.weight and ffn_up.weight has {up_rows} \
                             rows; a fused GEGLU file needs {} (2 · feed_forward_length)",
                            2 * n_ff
                        ));
                    }
                    fused_gate = true;
                    (None, 2 * n_ff)
                }
            };
            shape_of(&up_t, d_model, up_width).map_err(|e| format!("blk.{i}.ffn_up: {e}"))?;
            let up = Linear::Dense(up_t);
            let down = Linear::Dense(g.tensor(&p("ffn_down.weight"))?.clone());
            if let Some(t) = down.dense() {
                shape_of(t, n_ff, d_model).map_err(|e| format!("blk.{i}.ffn_down: {e}"))?;
            }

            layers.push(JinaBertLayer {
                wq,
                wk,
                wv,
                wo,
                bq: opt_vec_f32(g, &p("attn_q.bias"), qd)?,
                bk: opt_vec_f32(g, &p("attn_k.bias"), kvd)?,
                bv: opt_vec_f32(g, &p("attn_v.bias"), kvd)?,
                bo: opt_vec_f32(g, &p("attn_output.bias"), d_model)?,
                q_norm: match g.tensor(&p("attn_q_norm.weight")) {
                    Ok(_) => Some((
                        vec_f32(g, &p("attn_q_norm.weight"), qd)?,
                        opt_vec_f32(g, &p("attn_q_norm.bias"), qd)?,
                    )),
                    Err(_) => None,
                },
                k_norm: match g.tensor(&p("attn_k_norm.weight")) {
                    Ok(_) => Some((
                        vec_f32(g, &p("attn_k_norm.weight"), kvd)?,
                        opt_vec_f32(g, &p("attn_k_norm.bias"), kvd)?,
                    )),
                    Err(_) => None,
                },
                attn_out_norm_w: vec_f32(g, &p("attn_output_norm.weight"), d_model)?,
                attn_out_norm_b: vec_f32(g, &p("attn_output_norm.bias"), d_model)?,
                attn_norm_2: match g.tensor(&p("attn_norm_2.weight")) {
                    Ok(_) => Some((
                        vec_f32(g, &p("attn_norm_2.weight"), d_model)?,
                        opt_vec_f32(g, &p("attn_norm_2.bias"), d_model)?,
                    )),
                    Err(_) => None,
                },
                gate,
                up,
                up_b: opt_vec_f32(g, &p("ffn_up.bias"), up_width)?,
                down,
                down_b: vec_f32(g, &p("ffn_down.bias"), d_model)?,
                out_norm_w: vec_f32(g, &p("layer_output_norm.weight"), d_model)?,
                out_norm_b: vec_f32(g, &p("layer_output_norm.bias"), d_model)?,
            });
        }

        let cfg = Config {
            arch_name: "jina-bert-v2",
            // This family has no RoPE: positions enter through ALiBi's bias only.
            rope_neox: false,
            n_layer,
            n_kv_layer: n_layer,
            n_head,
            n_kv_head,
            head_dim,
            d_model,
            n_ff,
            n_vocab: tk.n_vocab,
            rope_dim: 0,
            eps,
            rope_base: 10_000.0,
            ctx_train: ctx,
            rope_yarn: None,
        };

        let mut linears: Vec<&Linear> = Vec::new();
        for l in &layers {
            linears.extend([&l.wq, &l.wk, &l.wv, &l.wo, &l.up, &l.down]);
            if let Some(gt) = &l.gate {
                linears.push(gt);
            }
        }
        let k_layout = linears
            .iter()
            .filter_map(|lin| lin.dense())
            .any(|t| t.kind.reads_k_layout());

        Ok(Loaded {
            cfg,
            arch: Box::new(JinaBert {
                tok_embd,
                type_row0,
                tok_norm_w,
                tok_norm_b,
                layers,
                // `hparams.f_max_alibi_bias = 8` is the loader's constant for this
                // architecture, and any positive value turns ALiBi on.
                slopes: peregrine_kernels::bidir::alibi_slopes(n_head, 8.0),
                ctx,
                fused_gate,
                k_layout,
            }),
            rope_factors: None,
            tiles: TileStore::default(),
        })
    }
}

/// Every row of a LayerNorm (weight and bias) into `out`, and the row filled into the
/// activation store it feeds. Rows are independent, so the batch is split across the
/// pool; a row's bits are the per-row kernel's whatever the split (S4.9a, S4.9h).
#[allow(clippy::too_many_arguments)]
fn norm_fill(
    pool: &Pool,
    stats: &mut StepStats,
    x: &mut [f32],
    w: &[f32],
    b: &[f32],
    eps: f32,
    dim: usize,
    nb: usize,
    out: &mut [f32],
    acts: &mut [QuantAct],
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
            vmath::layer_norm_rows(xc, w, b, eps, dim, n, oc);
            for (i, a) in ac.iter_mut().enumerate() {
                a.fill(&oc[i * dim..(i + 1) * dim]);
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

/// Fill the activation rows from an f32 buffer (the attention output is the next
/// projection's input) without recomputing anything.
fn fill_acts(
    pool: &Pool,
    stats: &mut StepStats,
    x: &[f32],
    dim: usize,
    nb: usize,
    acts: &mut [QuantAct],
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    peregrine_core::blocks::par::for_row_chunks(
        pool,
        parallel,
        acts,
        1,
        nb,
        Sched::Static,
        |_wk, rows, ac| {
            for (i, a) in ac.iter_mut().enumerate() {
                let r = rows.start + i;
                a.fill(&x[r * dim..(r + 1) * dim]);
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

/// The GEGLU pair with separate matrices: `gate = gelu(gate) · up` in place, then the
/// activation rows for the down projection.
fn gelu_gate_fill(
    pool: &Pool,
    stats: &mut StepStats,
    gate: &mut [f32],
    up: &[f32],
    dim: usize,
    nb: usize,
    acts: &mut [QuantAct],
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    for_row_chunks_2(
        pool,
        parallel,
        gate,
        dim,
        acts,
        1,
        nb,
        Sched::Static,
        |_wk, rows, gc, ac| {
            for (i, a) in ac.iter_mut().enumerate() {
                let r = rows.start + i;
                let g = &mut gc[i * dim..(i + 1) * dim];
                vmath::geglu(g, &up[r * dim..(r + 1) * dim]);
                a.fill(g);
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

/// The fused GEGLU pair: `up` holds `2·n_ff` per row, the first half is activated and
/// multiplied by the second, and the `n_ff`-wide product goes to `hb` and the
/// activation rows (`ggml_geglu`, the `swapped = false` layout).
fn gelu_fused_fill(
    pool: &Pool,
    stats: &mut StepStats,
    up: &mut [f32],
    hb: &mut [f32],
    n_ff: usize,
    nb: usize,
    acts: &mut [QuantAct],
) {
    let parallel = nb >= ROW_PARALLEL && pool.width() > 1;
    for_row_chunks_3(
        pool,
        parallel,
        up,
        2 * n_ff,
        hb,
        n_ff,
        acts,
        1,
        nb,
        Sched::Static,
        |_wk, _rows, uc, hc, ac| {
            for (i, a) in ac.iter_mut().enumerate() {
                let row = &mut uc[i * 2 * n_ff..(i + 1) * 2 * n_ff];
                let (gate, up) = row.split_at_mut(n_ff);
                vmath::geglu(gate, up);
                hc[i * n_ff..(i + 1) * n_ff].copy_from_slice(gate);
                a.fill(gate);
            }
        },
    );
    if parallel {
        stats.barriers += 1;
    }
}

/// `out[row][i] += bias[i]` for every row, skipping an all-zero bias so a `-0.0` from
/// the projection stays.
fn add_bias(out: &mut [f32], dim: usize, nb: usize, bias: &[f32]) {
    if bias.iter().all(|&b| b == 0.0) {
        return;
    }
    for r in 0..nb {
        let row = &mut out[r * dim..(r + 1) * dim];
        for (o, &b) in row.iter_mut().zip(bias) {
            *o += b;
        }
    }
}

/// `out[row][i] = (out[row][i] + bias[i]) + residual[row][i]`, llama.cpp's bias add
/// followed by the residual add, in one walk.
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

impl Architecture for JinaBert {
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
        if self.fused_gate {
            Some(Box::new(JinaScratch::new(max_batch, 2 * cfg.n_ff)))
        } else {
            None
        }
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
            for lin in [
                &mut l.wq,
                &mut l.wk,
                &mut l.wv,
                &mut l.wo,
                &mut l.up,
                &mut l.down,
            ] {
                f(lin)?;
            }
            if let Some(gt) = &mut l.gate {
                f(gt)?;
            }
        }
        Ok(())
    }

    fn linear_groups(&self, f: &mut dyn FnMut(&[&Linear])) {
        for l in &self.layers {
            f(&[&l.wq]);
            f(&[&l.wk]);
            f(&[&l.wv]);
            f(&[&l.wo]);
            if let Some(gt) = &l.gate {
                f(&[gt]);
            }
            f(&[&l.up]);
            f(&[&l.down]);
        }
    }

    /// One bidirectional pass over whole texts (design §2.2).
    ///
    /// The transcribed llama.cpp graph (`src/models/bert.cpp:68-225` with the
    /// `LLM_ARCH_JINA_BERT_V2` branches):
    ///
    /// 1. `x = token_embd[token] + token_types[0]` (no position table);
    /// 2. `token_embd_norm`: LayerNorm with a weight and a bias;
    /// 3. per layer:
    ///    a. `attn_q/k/v` from `x` (+ biases), optionally with each token's whole
    ///    q/k row LayerNormed (`attn_q_norm` / `attn_k_norm`);
    ///    b. attention over the whole text at `1/√head_dim`, with ALiBi's
    ///    `slope · (−|i − j|)` added to every score before the softmax;
    ///    c. `attn_output` (+ bias): `x = that + x`; `attn_out_norm`;
    ///    d. if `attn_norm_2` exists: `x = that + x` again, then `attn_norm_2`;
    ///    e. `ffn_gate`/`ffn_up` (+ up bias), gelu of the gate times the up (or the
    ///    fused pair's halves), `ffn_down` (+ bias): `x = that + ffn_inp`;
    ///    f. `layer_output_norm` — the layer's output, and there is no final norm;
    /// 4. the read-out per text: pooling of those rows by the caller's type.
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
        let nb: usize = texts.iter().map(|t| t.len()).sum();
        if nb == 0 {
            stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
            return Ok(());
        }
        for (i, t) in texts.iter().enumerate() {
            if t.len() > self.ctx {
                return Err(format!(
                    "text {i} has {} tokens; the jina-bert-v2 model's context_length is {}",
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

        // The batch's spans and per-row positions: every text starts at 0 (the
        // positions are ALiBi's `i`, not a table index).
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

        // ---- embeddings: token + token type 0, then the embedding LayerNorm ----
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
                for (x, &b) in st.x[row * dm..(row + 1) * dm]
                    .iter_mut()
                    .zip(&self.type_row0)
                {
                    *x += b;
                }
                row += 1;
            }
        }
        norm_fill(
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

        // The fused GEGLU pair's scratch, when a file has one. A state built by
        // `State::new` (not `Model::new_state`) has no family scratch, so it gets one
        // here.
        if self.fused_gate && st.fam.is_none() {
            st.fam = Some(Box::new(JinaScratch::new(st.max_batch, 2 * ff)));
        }

        for l in &self.layers {
            // ---- q/k/v: one quantisation, one fork-join, three matrices ----
            {
                let mut jobs = [
                    RowsOut {
                        lin: &l.wq,
                        out: peregrine_core::blocks::par::Disjoint::new(&mut st.q),
                        out_dim: qd,
                    },
                    RowsOut {
                        lin: &l.wk,
                        out: peregrine_core::blocks::par::Disjoint::new(&mut st.k),
                        out_dim: kvd,
                    },
                    RowsOut {
                        lin: &l.wv,
                        out: peregrine_core::blocks::par::Disjoint::new(&mut st.v),
                        out_dim: kvd,
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
                    &st.x,
                    dm,
                    nb,
                    &mut st.kq,
                );
            }
            add_bias(&mut st.q, qd, nb, &l.bq);
            add_bias(&mut st.k, kvd, nb, &l.bk);
            add_bias(&mut st.v, kvd, nb, &l.bv);

            // ---- optional q/k LayerNorm over each token's whole row ----
            if let Some((qw, qb)) = &l.q_norm {
                vmath::layer_norm_rows(&st.q, qw, qb, cfg.eps, qd, nb, &mut st.xb);
                st.q[..nb * qd].copy_from_slice(&st.xb[..nb * qd]);
            }
            if let Some((kw, kb)) = &l.k_norm {
                vmath::layer_norm_rows(&st.k, kw, kb, cfg.eps, kvd, nb, &mut st.xb);
                st.k[..nb * kvd].copy_from_slice(&st.xb[..nb * kvd]);
            }

            // llama.cpp passes `kq_scale = 1/√head_dim` to the softmax, which applies
            // it before ALiBi's bias; the block takes the same scale.
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();

            // ---- attention: one n × n block per text, ALiBi's bias on every score ----
            peregrine_core::blocks::bidir::attend_texts(
                cfg,
                &st.q,
                &st.k,
                &st.v,
                &mut st.xb,
                &spans,
                None,
                Some(&self.slopes),
                scale,
                &mut st.attn.bidir,
                pool,
                stats,
            );

            // ---- attn_output (+ bias), residual, attn_out_norm ----
            fill_acts(pool, stats, &st.xb, qd, nb, &mut st.acts);
            {
                let mut jobs = [RowsOut {
                    lin: &l.wo,
                    out: peregrine_core::blocks::par::Disjoint::new(&mut st.xb2),
                    out_dim: dm,
                }];
                linear_rows_shared(
                    w,
                    pool,
                    stats,
                    &mut jobs,
                    &st.acts,
                    &mut st.lb_x,
                    &mut st.lb_t,
                    &st.xb,
                    qd,
                    nb,
                    &mut st.kq,
                );
            }
            add_bias_residual(&mut st.xb2[..nb * dm], &st.x[..nb * dm], &l.bo, dm);
            norm_fill(
                pool,
                stats,
                &mut st.xb2,
                &l.attn_out_norm_w,
                &l.attn_out_norm_b,
                cfg.eps,
                dm,
                nb,
                &mut st.xb,
                &mut st.acts,
            );
            if let Some((nw, nbias)) = &l.attn_norm_2 {
                for i in 0..nb * dm {
                    st.xb[i] += st.x[i];
                }
                norm_fill(
                    pool,
                    stats,
                    &mut st.xb,
                    nw,
                    nbias,
                    cfg.eps,
                    dm,
                    nb,
                    &mut st.xb2,
                    &mut st.acts,
                );
                st.xb[..nb * dm].copy_from_slice(&st.xb2[..nb * dm]);
            }

            // ---- the GEGLU pair, ffn_down, the second residual ----
            if let Some(gate) = &l.gate {
                let mut jobs = [
                    RowsOut {
                        lin: gate,
                        out: peregrine_core::blocks::par::Disjoint::new(&mut st.hb),
                        out_dim: ff,
                    },
                    RowsOut {
                        lin: &l.up,
                        out: peregrine_core::blocks::par::Disjoint::new(&mut st.hb2),
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
                // `ffn_up.bias` is added to the up matrix's own output (`hb2`); the
                // gate matrix has no bias in llama.cpp's loader for this family.
                add_bias(&mut st.hb2, ff, nb, &l.up_b);
                gelu_gate_fill(pool, stats, &mut st.hb, &st.hb2, ff, nb, &mut st.acts);
            } else {
                let scratch = st
                    .fam
                    .as_mut()
                    .and_then(|f| f.downcast_mut::<JinaScratch>())
                    .ok_or("the state has no jina-bert-v2 scratch")?;
                scratch.ensure(nb);
                {
                    let mut jobs = [RowsOut {
                        lin: &l.up,
                        out: peregrine_core::blocks::par::Disjoint::new(&mut scratch.buf),
                        out_dim: 2 * ff,
                    }];
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
                add_bias(&mut scratch.buf, 2 * ff, nb, &l.up_b);
                gelu_fused_fill(
                    pool,
                    stats,
                    &mut scratch.buf,
                    &mut st.hb,
                    ff,
                    nb,
                    &mut st.acts,
                );
            }

            {
                let mut jobs = [RowsOut {
                    lin: &l.down,
                    out: peregrine_core::blocks::par::Disjoint::new(&mut st.xb2),
                    out_dim: dm,
                }];
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
            // `x = down + ffn_inp`, where `ffn_inp` is the normed row in `st.xb`.
            add_bias_residual(&mut st.xb2[..nb * dm], &st.xb[..nb * dm], &l.down_b, dm);

            // `layer_output_norm` is the layer's output: this family has no final norm.
            norm_fill(
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

        // ---- read-outs, text by text ----
        if st.pool_sum.len() < dm {
            st.pool_sum.resize(dm, 0.0);
        }
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
                    return Err(format!(
                        "the {} model has no classifier head",
                        cfg.arch_name
                    ));
                }
            }
        }
        st.positions = pos;
        st.spans = spans;
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }
}
