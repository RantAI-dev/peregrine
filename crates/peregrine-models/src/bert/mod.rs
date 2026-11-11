//! The BERT family: the post-norm bidirectional encoder (roadmap S4.6a).
//!
//! `architecture = "bert"` in llama.cpp's table is the encoder every MiniLM, BGE,
//! E5, GTE and XLM-R (`bge-m3`, multilingual-e5) embedding file ships as: token,
//! position and token-type embeddings summed and LayerNormed once, then N layers
//! of post-norm blocks — q/k/v with biases, full bidirectional attention, the
//! output projection, a residual add and a LayerNorm; the feed-forward (up, the
//! exact GELU, down), a residual add and a second LayerNorm. Nothing is causal,
//! nothing is cached, and the layer's output *is* its normalised value: the next
//! layer applies no pre-norm.
//!
//! Every detail here is transcribed from llama.cpp **v0.9.0 (`cd959ba`)**:
//! `src/models/bert.cpp` (the graph), `src/llama-model.cpp` (the tensors and
//! `causal_attn`/`pooling_type`), `src/llama-graph.cpp` (`build_inp_pos`,
//! `build_attn_inp_no_cache`, `build_norm`) and `conversion/bert.py` (the token
//! types and the XLM-R position table). [`docs/reference/bert.md`] holds the
//! reading, the tensor table and the measured accuracy.
//!
//! [`docs/reference/bert.md`]: ../../../../docs/reference/bert.md

use std::time::Instant;

use peregrine_core::arch::{ActLayout, Architecture, Causality, Family, Loaded};
use peregrine_core::blocks::linear::LinJob;
use peregrine_core::blocks::par::{Disjoint, OutPtr};
use peregrine_core::gguf::{Gguf, TensorInfo};
use peregrine_core::model::{Caches, Config, Linear, State, StepStats, TileStore, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::quant::QuantAct;
use peregrine_core::tokenizer::Tokenizer;

/// The family, registered as `bert`.
pub struct BertFamily;

impl Family for BertFamily {
    fn names(&self) -> &'static [&'static str] {
        &["bert"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        Bert::load(g, tk)
    }

    /// A `bert` file is an embedding model: an encoder has no decode path, so the
    /// engine refuses chat and label requests on it by name (design §3).
    fn embedding_only(&self) -> bool {
        true
    }
}

/// One post-norm encoder layer.
pub(crate) struct BertLayer {
    pub(crate) wq: Linear,
    pub(crate) wk: Linear,
    pub(crate) wv: Linear,
    pub(crate) wo: Linear,
    /// Attention biases, `q_dim` / `kv_dim` / `kv_dim` / `d_model` floats. A file
    /// without them (llama.cpp marks them `TENSOR_NOT_REQUIRED`) gets zeros.
    pub(crate) bq: Vec<f32>,
    pub(crate) bk: Vec<f32>,
    pub(crate) bv: Vec<f32>,
    pub(crate) bo: Vec<f32>,
    /// `attn_output_norm`, the post-attention LayerNorm: weight *and* bias.
    pub(crate) attn_norm_w: Vec<f32>,
    pub(crate) attn_norm_b: Vec<f32>,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
    pub(crate) up_b: Vec<f32>,
    pub(crate) down_b: Vec<f32>,
    /// `layer_output_norm`, the post-FFN LayerNorm.
    pub(crate) out_norm_w: Vec<f32>,
    pub(crate) out_norm_b: Vec<f32>,
}

/// A loaded `bert` model.
pub struct Bert {
    pub(crate) tok_embd: TensorInfo,
    /// `token_types.weight`, row 0 only: llama.cpp hardcodes token type 0
    /// ("Sentence A") for every token, so the other rows are never read.
    pub(crate) type_row0: Option<Vec<f32>>,
    pub(crate) pos_embd: TensorInfo,
    pub(crate) tok_norm_w: Vec<f32>,
    pub(crate) tok_norm_b: Vec<f32>,
    pub(crate) layers: Vec<BertLayer>,
    /// Rows of the position table: the longest text the model can embed. This is
    /// `context_length`; a file whose metadata claims more is refused at load.
    pub(crate) n_pos: usize,
    /// Whether any matrix reads the per-256 activation layout (S4.9a).
    pub(crate) k_layout: bool,
}

impl Bert {
    #[allow(clippy::too_many_lines)]
    fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        let arch = g.str("general.architecture")?;
        if arch != "bert" {
            return Err(format!(
                "expected architecture \"bert\", found {arch:?} — nomic-bert, modern-bert and \
                 jina-bert are other families (roadmap S4.7, S4.8)"
            ));
        }
        let k = |suffix: &str| format!("bert.{suffix}");
        let n_layer = g.u32(&k("block_count"))? as usize;
        let d_model = g.u32(&k("embedding_length"))? as usize;
        let n_head = g.u32(&k("attention.head_count"))? as usize;
        let n_kv_head = g
            .u32(&k("attention.head_count_kv"))
            .unwrap_or(n_head as u32) as usize;
        let n_ff = g.u32(&k("feed_forward_length"))? as usize;
        // llama.cpp: `n_embd_head_k = n_embd / n_head`, unless the file names a
        // `head_dim` (conversion/base.py writes `attention.key_length` for those).
        let head_dim = g
            .u32(&k("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(d_model / n_head);
        if n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            return Err(format!(
                "implausible head configuration: {n_head} query heads over {n_kv_head} kv heads"
            ));
        }
        if head_dim * n_head != d_model {
            // A `bert` attention head is `d_model / n_head` wide: q, k and v are all
            // `n_embd` wide on both sides (create_tensor_qkv(layer, i, n_embd, n_embd,
            // n_embd_gqa, n_embd_gqa, 0) with `n_embd_gqa == n_embd`).
            return Err(format!(
                "the query width {} is not the embedding length {d_model}: a bert file has no \
                 separate query dimension",
                n_head * head_dim
            ));
        }
        if n_kv_head * head_dim != d_model {
            return Err(format!(
                "the key/value width {} is not the embedding length {d_model}",
                n_kv_head * head_dim
            ));
        }

        // The token type table is required by llama.cpp's loader (`n_token_types == 0`
        // throws); its row 0 is what the graph adds to every token.
        let n_token_types = g
            .u32("tokenizer.ggml.token_type_count")
            .map_err(|_| {
                "bert model needs to define token type count \
                 (tokenizer.ggml.token_type_count)"
                    .to_string()
            })? as usize;
        if n_token_types == 0 {
            return Err("bert model needs to define token type count".into());
        }

        let eps = g
            .f32(&k("attention.layer_norm_epsilon"))
            .unwrap_or(1e-5);

        // The position table is the text limit, and there is no offset: llama.cpp
        // indexes it with the sequence's own 0-based positions (build_inp_pos), and
        // the XLM-R converter chops `pad_token_id + 1` rows off the table it reads
        // (`conversion/bert.py`, `RobertaModel::modify_tensors`), so row `p` is
        // position `p` on both sides.
        let pos_embd = g.tensor("position_embd.weight")?.clone();
        let n_pos = pos_embd.dims.get(1).copied().unwrap_or(0) as usize;
        if n_pos == 0 || pos_embd.dims.first().copied().unwrap_or(0) as usize != d_model {
            return Err(format!(
                "position_embd.weight is {:?}, expected [{d_model}, positions]",
                pos_embd.dims
            ));
        }
        let mut ctx_train = g.u32(&k("context_length")).map_or(n_pos, |v| v as usize);
        if ctx_train > n_pos {
            return Err(format!(
                "{}.context_length is {ctx_train} but the position table holds {n_pos} \
                 positions: llama.cpp would read past its rows",
                arch
            ));
        }
        ctx_train = ctx_train.min(n_pos);

        let tok_embd = g.tensor("token_embd.weight")?.clone();
        let n_vocab_meta = g.u32(&k("vocab_size")).map(|v| v as usize);
        let n_vocab = n_vocab_meta.unwrap_or(tk.n_vocab);
        if n_vocab != tk.n_vocab {
            return Err(format!(
                "metadata says {n_vocab} vocabulary entries but the tokenizer has {}",
                tk.n_vocab
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
        shape(&tok_embd, d_model, n_vocab)?;

        // A vector weight (a norm's weight or bias, or a projection bias), dequantised:
        // these are F32 in every published file, and F16 would be read exactly anyway.
        let vec_f32 = |name: &str, n: usize| -> Result<Vec<f32>, String> {
            let t = g.tensor(name)?;
            if t.numel() as usize != n {
                return Err(format!("{name} has {} entries, expected {n}", t.numel()));
            }
            let mut v = vec![0.0f32; n];
            peregrine_kernels::quant::dequant_row(t.kind, g.bytes(t), &mut v);
            Ok(v)
        };
        // An optional bias: llama.cpp marks the projections' biases `TENSOR_NOT_REQUIRED`.
        let bias_or_zero = |name: &str, n: usize| -> Result<Vec<f32>, String> {
            match g.tensor(name) {
                Ok(_) => vec_f32(name, n),
                Err(_) => Ok(vec![0.0f32; n]),
            }
        };

        let type_row0 = match g.tensor("token_types.weight") {
            Ok(t) => {
                let cols = t.dims.first().copied().unwrap_or(0) as usize;
                let rows = t.dims.get(1).copied().unwrap_or(0) as usize;
                if cols != d_model || rows != n_token_types {
                    return Err(format!(
                        "token_types.weight is {:?}, expected [{d_model}, {n_token_types}]",
                        t.dims
                    ));
                }
                // Row 0 only: llama.cpp hardcodes token type 0 for every token, and
                // reads that row as a 1-d view (`ggml_view_1d(ctx0, type_embd, n_embd, 0)`).
                let rb = peregrine_core::quant::row_bytes(t.kind, cols);
                let mut v = vec![0.0f32; cols];
                peregrine_kernels::quant::dequant_row(t.kind, &g.bytes(t)[..rb], &mut v);
                Some(v)
            }
            Err(_) => None,
        };

        let qd = n_head * head_dim;
        let kvd = n_kv_head * head_dim;
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let wq = Linear::Dense(g.tensor(&p("attn_q.weight"))?.clone());
            let wk = Linear::Dense(g.tensor(&p("attn_k.weight"))?.clone());
            let wv = Linear::Dense(g.tensor(&p("attn_v.weight"))?.clone());
            let wo = Linear::Dense(g.tensor(&p("attn_output.weight"))?.clone());
            let up = Linear::Dense(g.tensor(&p("ffn_up.weight"))?.clone());
            let down = Linear::Dense(g.tensor(&p("ffn_down.weight"))?.clone());
            for (lin, d_in, d_out, what) in [
                (&wq, d_model, qd, "attn_q"),
                (&wk, d_model, kvd, "attn_k"),
                (&wv, d_model, kvd, "attn_v"),
                (&wo, qd, d_model, "attn_output"),
                (&up, d_model, n_ff, "ffn_up"),
                (&down, n_ff, d_model, "ffn_down"),
            ] {
                if let Some(t) = lin.dense() {
                    shape(t, d_in, d_out).map_err(|e| format!("blk.{i}.{what}: {e}"))?;
                }
            }
            layers.push(BertLayer {
                wq,
                wk,
                wv,
                wo,
                bq: bias_or_zero(&p("attn_q.bias"), qd)?,
                bk: bias_or_zero(&p("attn_k.bias"), kvd)?,
                bv: bias_or_zero(&p("attn_v.bias"), kvd)?,
                bo: bias_or_zero(&p("attn_output.bias"), d_model)?,
                attn_norm_w: vec_f32(&p("attn_output_norm.weight"), d_model)?,
                attn_norm_b: vec_f32(&p("attn_output_norm.bias"), d_model)?,
                up,
                down,
                up_b: bias_or_zero(&p("ffn_up.bias"), n_ff)?,
                down_b: bias_or_zero(&p("ffn_down.bias"), d_model)?,
                out_norm_w: vec_f32(&p("layer_output_norm.weight"), d_model)?,
                out_norm_b: vec_f32(&p("layer_output_norm.bias"), d_model)?,
            });
        }

        let tok_norm_w = vec_f32("token_embd_norm.weight", d_model)?;
        let tok_norm_b = vec_f32("token_embd_norm.bias", d_model)?;

        let cfg = Config {
            arch_name: "bert",
            rope_neox: false,
            n_layer,
            n_kv_layer: n_layer,
            n_head,
            n_kv_head,
            head_dim,
            d_model,
            n_ff,
            n_vocab,
            // No rotary positions: the encoder has a learned absolute table instead.
            rope_dim: 0,
            eps,
            rope_base: 10_000.0,
            ctx_train,
            rope_yarn: None,
        };

        let mut linears: Vec<&Linear> = Vec::new();
        for l in &layers {
            linears.extend([&l.wq, &l.wk, &l.wv, &l.wo, &l.up, &l.down]);
        }
        let k_layout = linears
            .iter()
            .filter_map(|lin| lin.dense())
            .any(|t| t.kind.reads_k_layout());

        Ok(Loaded {
            cfg,
            arch: Box::new(Bert {
                tok_embd,
                type_row0,
                pos_embd,
                tok_norm_w,
                tok_norm_b,
                layers,
                n_pos,
                k_layout,
            }),
            rope_factors: None,
            tiles: TileStore::default(),
        })
    }
}

/// One projection over many rows, split into matmul-sized chunks.
///
/// The shared GEMM accepts at most `MAX_BATCH` rows per call; an encoder forwards a
/// whole text (up to the model's position table) at once, so a layer's projection
/// runs in as many calls as it takes, one weight pass each. Rows of the batch are
/// contiguous, so the chunk is a plain row range and a row's bits do not depend on
/// where a chunk boundary fell.
#[allow(clippy::too_many_arguments)]
fn linear_rows(
    w: &Weights,
    pool: &Pool,
    stats: &mut StepStats,
    lin: &Linear,
    acts: &mut [QuantAct],
    lb_x: &mut [f32],
    lb_t: &mut [f32],
    input: &[f32],
    d_in: usize,
    nb: usize,
    out: &mut [f32],
    out_dim: usize,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    let step = peregrine_core::blocks::linear::MAX_BATCH;
    let mut r0 = 0;
    while r0 < nb {
        let r1 = (r0 + step).min(nb);
        let take = r1 - r0;
        for (i, a) in acts[r0..r1].iter_mut().enumerate() {
            a.fill(&input[(r0 + i) * d_in..(r0 + i + 1) * d_in]);
        }
        let lbx = Disjoint::new(lb_x);
        let lbt = Disjoint::new(lb_t);
        let jobs = [LinJob {
            lin,
            out: Disjoint::new(&mut out[r0 * out_dim..r1 * out_dim]),
        }];
        peregrine_core::blocks::linear::linear_group(
            w,
            w.lb(),
            &jobs,
            &acts[r0..r1],
            &input[r0 * d_in..r1 * d_in],
            d_in,
            take,
            lbx,
            lbt,
            pool,
            stats,
            kq,
        );
        r0 = r1;
    }
}

/// One or more projections that read the **same** activation rows, in matmul-sized
/// chunks, without quantising anything: the rows are already in `acts`, filled once
/// by the norm that produced them (S4.9a). q/k/v share one fill and one fork-join.
#[allow(clippy::too_many_arguments)]
fn linear_rows_shared(
    w: &Weights,
    pool: &Pool,
    stats: &mut StepStats,
    jobs: &mut [RowsOut<'_>],
    acts: &[QuantAct],
    lb_x: &mut [f32],
    lb_t: &mut [f32],
    input: &[f32],
    d_in: usize,
    nb: usize,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    debug_assert!(!jobs.is_empty() && jobs.len() <= 4);
    let njobs = jobs.len();
    let step = peregrine_core::blocks::linear::MAX_BATCH;
    let mut r0 = 0;
    while r0 < nb {
        let r1 = (r0 + step).min(nb);
        let take = r1 - r0;
        let mut lj = [LinJob {
            lin: jobs[0].lin,
            out: Disjoint::new(&mut []),
        }; 4];
        for i in 0..njobs {
            lj[i].lin = jobs[i].lin;
            let width = jobs[i].out_dim;
            // SAFETY: each job's output is a different buffer, and this chunk of it
            // belongs to this call alone.
            lj[i].out = Disjoint::new(unsafe { jobs[i].out.rows(r0..r1, width) });
        }
        let lbx = Disjoint::new(lb_x);
        let lbt = Disjoint::new(lb_t);
        peregrine_core::blocks::linear::linear_group(
            w,
            w.lb(),
            &lj[..njobs],
            &acts[r0..r1],
            &input[r0 * d_in..r1 * d_in],
            d_in,
            take,
            lbx,
            lbt,
            pool,
            stats,
            kq,
        );
        r0 = r1;
    }
}

/// One projection of [`linear_rows_shared`]: the matrix, its output buffer, and the
/// width of one output row. Every job's buffer must be a different one — the group
/// writes them in one fork-join.
struct RowsOut<'a> {
    lin: &'a Linear,
    out: OutPtr<'a>,
    out_dim: usize,
}

/// `out[row][i] += bias[i]` for every row of a projection's output.
#[inline]
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

/// LayerNorm every row of the batch into `out` (f32 only, no quantisation); the
/// per-row arithmetic is `vmath::layer_norm_row`'s, and shared with the reference.
#[inline]
fn layer_norm_rows(x: &[f32], w: &[f32], b: &[f32], eps: f32, dim: usize, nb: usize, out: &mut [f32]) {
    peregrine_core::vmath::layer_norm_rows(x, w, b, eps, dim, nb, out);
}

impl Architecture for Bert {
    fn causality(&self) -> Causality {
        Causality::Bidirectional
    }

    fn act_layout(&self) -> ActLayout {
        // A float (F32/F16/BF16) matrix's packed kernel reads the f32 activation
        // rows (`keep_f32`); an all-Q8_0 file never needs them (S4.9f, S4.9a).
        let mut keep_f32 = false;
        self.linear_groups(&mut |lins| {
            for lin in lins {
                if let Some(ti) = lin.dense() {
                    if matches!(
                        ti.kind,
                        peregrine_core::gguf::Format::F32
                            | peregrine_core::gguf::Format::F16
                            | peregrine_core::gguf::Format::BF16
                    ) {
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

    /// A `bert` model has no causal forward: a text is one bidirectional pass, and
    /// the engine must call [`Architecture::forward_texts`].
    fn forward_rows(
        &self,
        _w: &Weights,
        cfg: &Config,
        _st: &mut State,
        _caches: &mut Caches<'_>,
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
            for lin in [&mut l.wq, &mut l.wk, &mut l.wv, &mut l.wo, &mut l.up, &mut l.down] {
                f(lin)?;
            }
        }
        Ok(())
    }

    fn linear_groups(&self, f: &mut dyn FnMut(&[&Linear])) {
        for l in &self.layers {
            f(&[&l.wq, &l.wk, &l.wv]);
            f(&[&l.wo]);
            f(&[&l.up]);
            f(&[&l.down]);
        }
    }

    /// One bidirectional pass over whole texts (design §2.2).
    ///
    /// The transcribed llama.cpp graph (`src/models/bert.cpp`), per text and in
    /// order:
    ///
    /// 1. `x = token_embd[token] + token_types[0] + position_embd[p]`;
    /// 2. `token_embd_norm`: LayerNorm with weight and bias;
    /// 3. per layer, post-norm:
    ///    a. q/k/v (+ biases), with **no** pre-norm: the layer's input is already
    ///       the previous layer's normalised output;
    ///    b. attention over the whole text, `1/√head_dim` scale, no cache;
    ///    c. `attn_output` (+ bias); `x = x + that`; `attn_output_norm`;
    ///    d. `ffn_up` (+ bias); the exact GELU; `ffn_down` (+ bias);
    ///       `x = x + that`; `layer_output_norm` → the next layer's input;
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
        poolings: &[peregrine_core::embed::Pooling],
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
        // The position table is the limit, and it is also the engine's context; a
        // text past it would read rows the table does not have.
        for (i, t) in texts.iter().enumerate() {
            if t.len() > self.n_pos {
                return Err(format!(
                    "text {i} has {} tokens; the bert position table holds {} positions",
                    t.len(),
                    self.n_pos
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

        // ---- embeddings: token + token type (row 0) + absolute position ----
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
                peregrine_core::blocks::embed::embed(
                    &self.pos_embd,
                    w,
                    pos[row] as u32,
                    &mut st.xb2[..dm],
                    stats,
                );
                for (x, &p) in st.x[row * dm..(row + 1) * dm]
                    .iter_mut()
                    .zip(&st.xb2[..dm])
                {
                    *x += p;
                }
                row += 1;
            }
        }
        // `token_embd_norm` feeds the first block's q/k/v: norm into `st.x`, quantise once.
        {
            let mut xin = std::mem::take(&mut st.x);
            layer_norm_rows(
                &xin,
                &self.tok_norm_w,
                &self.tok_norm_b,
                cfg.eps,
                dm,
                nb,
                &mut st.xb,
            );
            xin[..nb * dm].copy_from_slice(&st.xb[..nb * dm]);
            st.x = xin;
        }
        for r in 0..nb {
            st.acts[r].fill(&st.x[r * dm..(r + 1) * dm]);
        }

        for l in &self.layers {
            // ---- q/k/v (+ biases): one quantisation and one fork-join for the three
            // matrices, which read the same normed rows (S4.9a) ----
            {
                let mut jobs = [
                    RowsOut {
                        lin: &l.wq,
                        out: Disjoint::new(&mut st.q),
                        out_dim: qd,
                    },
                    RowsOut {
                        lin: &l.wk,
                        out: Disjoint::new(&mut st.k),
                        out_dim: kvd,
                    },
                    RowsOut {
                        lin: &l.wv,
                        out: Disjoint::new(&mut st.v),
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
            // llama.cpp passes `kq_scale = 1/√head_dim`; scaling Q here and 1.0 there
            // is the same product, and keeps the bidirectional block's interface.
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            for q in &mut st.q[..nb * qd] {
                *q *= scale;
            }

            // ---- attention: one n × n block per text, no cache ----
            peregrine_core::blocks::bidir::attend_texts(
                cfg,
                &st.q,
                &st.k,
                &st.v,
                &mut st.xb,
                &spans,
                None,
                1.0,
                &mut st.attn.bidir_scores,
                pool,
                stats,
            );

            // ---- attn_output (+ bias), residual, attn_output_norm ----
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
            add_bias(&mut st.xb2, dm, nb, &l.bo);
            for i in 0..nb * dm {
                st.xb2[i] += st.x[i];
            }
            layer_norm_rows(&st.xb2, &l.attn_norm_w, &l.attn_norm_b, cfg.eps, dm, nb, &mut st.xb);
            // The normed rows feed ffn_up: one fill, one group.
            for r in 0..nb {
                st.acts[r].fill(&st.xb[r * dm..(r + 1) * dm]);
            }

            // ---- feed-forward: up (+ bias), the exact GELU, down (+ bias) ----
            {
                let mut jobs = [RowsOut {
                    lin: &l.up,
                    out: Disjoint::new(&mut st.hb),
                    out_dim: ff,
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
            add_bias(&mut st.hb, ff, nb, &l.up_b);
            peregrine_core::vmath::gelu_erf_rows(&mut st.hb[..nb * ff]);
            // The GELU output is the down projection's input: quantise it once here
            // rather than filling the same rows again inside the matmul (S4.9a).
            for r in 0..nb {
                st.acts[r].fill(&st.hb[r * ff..(r + 1) * ff]);
            }
            {
                let mut jobs = [RowsOut {
                    lin: &l.down,
                    out: Disjoint::new(&mut st.xb2),
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
                    &st.hb,
                    ff,
                    nb,
                    &mut st.kq,
                );
            }
            add_bias(&mut st.xb2, dm, nb, &l.down_b);
            for i in 0..nb * dm {
                st.xb2[i] += st.xb[i];
            }
            // The layer's output is its normalised value: the next layer applies no
            // pre-norm, so `st.x` and the activation rows are its input directly.
            layer_norm_rows(&st.xb2, &l.out_norm_w, &l.out_norm_b, cfg.eps, dm, nb, &mut st.x);
            for r in 0..nb {
                st.acts[r].fill(&st.x[r * dm..(r + 1) * dm]);
            }
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
