//! An exact scalar reference forward for `bert` (roadmap S4.6a).
//!
//! Written independently of `peregrine_models::bert::Bert::forward_texts`: the
//! graph transcribed from llama.cpp v0.9.0 (`src/models/bert.cpp`) is
//! reimplemented here layer by layer, on one text at a time, with plain loops.
//! It shares only the *defined* kernels with the engine — `vmath::gelu_erf` and
//! `vmath::layer_norm_row`, whose arithmetic and order are part of the result,
//! and the float matmul's FMA chain (`row_dot`, exactly `embed_ref`'s) — because
//! the test's claim is that the *forward graph and its order* are right, and
//! those primitives are what make an F32 model reproduce bit for bit.
//!
//! The arithmetic the reference owns, and which it must get right on its own:
//! the embedding sum (token + type row 0 + position), the bias adds after every
//! projection, the residuals and the `1/√head_dim` scale applied to Q. The
//! `n × n` attention itself is the engine's block (`blocks::bidir::attend_texts`),
//! called here for one text on one worker: since S4.9g it is a strip kernel whose
//! arithmetic is pinned by that kernel's own cross-tier and cross-thread tests, and
//! the block is shared the same way the float matmul chain is.

use peregrine_core::embed::Pooling;
use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::model::Model;
use peregrine_kernels::quant::{dequant_row, dot_f32_ref};
use peregrine_kernels::table::MatKernels;

/// One float matrix row times an activation row, in the arithmetic the engine
/// defines for this build (the same helper `embed_ref` uses): the packed panel
/// kernel's FMA chain, or `dot_f32_ref` for a matrix the store does not pack.
fn row_dot(row: &[u8], x: &[f32]) -> f32 {
    if MatKernels::get().fp_panels() {
        let mut acc = 0.0f32;
        for (k, &xv) in x.iter().enumerate() {
            let wv = f32::from_le_bytes(row[k * 4..k * 4 + 4].try_into().unwrap());
            acc = xv.mul_add(wv, acc);
        }
        acc
    } else {
        dot_f32_ref(Format::F32, row, x)
    }
}

/// One projection's weights, in the file's format.
///
/// F32 is the defined float matmul chain ([`row_dot`]); Q8_0 is the engine's
/// defined int8 path — the activation row quantised once by `QuantAct::fill`
/// (the same call the forward makes) and the weight row dotted by
/// `quant::dot_scalar`, the scalar twin of the tile kernel (integer sums are
/// exact, and the per-block `d·d_a·Σ` accumulation is in block order).
struct Mat {
    kind: Format,
    bytes: Vec<u8>,
    d_in: usize,
    d_out: usize,
    /// Whether the engine reads this matrix from the packed float panels (whose chain is
    /// `row_dot`'s fused one) or row by row (`dot_f32_ref`, a plain multiply and add).
    /// A tensor written 1-d — a classifier head's one-row `cls.output.weight` — is not a
    /// panel matrix (`fpack::eligible` wants two dimensions), so its chain differs.
    packed: bool,
}

impl Mat {
    fn row(&self, r: usize) -> &[u8] {
        let rb = match self.kind {
            Format::F32 => self.d_in * 4,
            Format::Q8_0 => peregrine_core::quant::row_bytes(Format::Q8_0, self.d_in),
            other => panic!("the bert reference has no {other:?} path"),
        };
        &self.bytes[r * rb..(r + 1) * rb]
    }
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        match self.kind {
            Format::F32 => {
                for (r, o) in out[..self.d_out].iter_mut().enumerate() {
                    *o = if self.packed {
                        row_dot(self.row(r), x)
                    } else {
                        dot_f32_ref(Format::F32, self.row(r), x)
                    };
                }
            }
            Format::Q8_0 => {
                let mut a = peregrine_core::quant::QuantAct::with_capacity(self.d_in);
                a.fill(x);
                for (r, o) in out[..self.d_out].iter_mut().enumerate() {
                    let row = self.row(r).to_vec();
                    *o = peregrine_kernels::quant::dot_scalar(Format::Q8_0, &row, &a);
                }
            }
            _ => unreachable!(),
        }
    }
}

fn mat(g: &Gguf, name: &str) -> Mat {
    let t = g.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let d_in = t.dims[0] as usize;
    let d_out = t.dims.get(1).copied().unwrap_or(1) as usize;
    // S4.9m: a small Q8_0 matrix dequantised into f32 panels at load reads as the
    // float path, on the dequantised rows (`scale × code`, exact in f32).
    if t.fp_kind.is_some() {
        let src_rb = peregrine_core::quant::row_bytes(t.kind, d_in);
        let mut bytes = vec![0u8; d_out * d_in * 4];
        let mut row = vec![0.0f32; d_in];
        for r in 0..d_out {
            let src = &g.bytes(t)[r * src_rb..(r + 1) * src_rb];
            dequant_row(t.kind, src, &mut row);
            for (i, &v) in row.iter().enumerate() {
                bytes[(r * d_in + i) * 4..(r * d_in + i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        return Mat {
            kind: Format::F32,
            bytes,
            d_in,
            d_out,
            packed: true,
        };
    }
    Mat {
        kind: t.kind,
        bytes: g.bytes(t).to_vec(),
        d_in,
        d_out,
        // A Q8_0 matrix the store did not convert keeps the int8 twin; a float matrix is
        // a panel matrix unless the file wrote it 1-d.
        packed: t.kind != Format::F32 || t.fp_off.is_some(),
    }
}

fn vec_f32(g: &Gguf, name: &str) -> Vec<f32> {
    let t = g.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let n = t.numel() as usize;
    let mut v = vec![0.0f32; n];
    dequant_row(t.kind, g.bytes(t), &mut v);
    v
}

/// One table row (the token and position embeddings), dequantised.
fn table_row(g: &Gguf, name: &str, r: usize, n: usize) -> Vec<f32> {
    let t = g.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let rb = peregrine_core::quant::row_bytes(t.kind, n);
    let mut v = vec![0.0f32; n];
    dequant_row(t.kind, &g.bytes(t)[r * rb..(r + 1) * rb], &mut v);
    v
}

/// The last layer's rows for one text: llama.cpp's `result_embd`.
pub fn hidden_rows(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let g = model.gguf();
    let cfg = &model.cfg;
    let dm = cfg.d_model;
    let qd = cfg.q_dim();
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let n = tokens.len();
    let eps = cfg.eps;
    let scale = 1.0 / (hd as f32).sqrt();

    // ---- embeddings: token + token type row 0 + absolute position ----
    let type0 = if g.tensor("token_types.weight").is_ok() {
        Some(table_row(g, "token_types.weight", 0, dm))
    } else {
        None
    };
    let mut x = vec![0.0f32; n * dm];
    for (r, &tok) in tokens.iter().enumerate() {
        let row = table_row(g, "token_embd.weight", tok as usize, dm);
        let pos = table_row(g, "position_embd.weight", r, dm);
        for d in 0..dm {
            // llama.cpp's order: `inpL = tok + type_row0`, then `embd_pos + inpL`.
            let mut v = row[d];
            if let Some(t0) = &type0 {
                v += t0[d];
            }
            x[r * dm + d] = pos[d] + v;
        }
    }
    // `token_embd_norm`, the embedding LayerNorm.
    let tok_norm_w = vec_f32(g, "token_embd_norm.weight");
    let tok_norm_b = vec_f32(g, "token_embd_norm.bias");
    {
        let mut out = vec![0.0f32; n * dm];
        peregrine_core::vmath::layer_norm_rows(&x, &tok_norm_w, &tok_norm_b, eps, dm, n, &mut out);
        x = out;
    }

    for il in 0..cfg.n_layer {
        let p = |s: &str| format!("blk.{il}.{s}");
        let wq = mat(g, &p("attn_q.weight"));
        let wk = mat(g, &p("attn_k.weight"));
        let wv = mat(g, &p("attn_v.weight"));
        let wo = mat(g, &p("attn_output.weight"));
        let bq = vec_f32(g, &p("attn_q.bias"));
        let bk = vec_f32(g, &p("attn_k.bias"));
        let bv = vec_f32(g, &p("attn_v.bias"));
        let bo = vec_f32(g, &p("attn_output.bias"));
        let attn_w = vec_f32(g, &p("attn_output_norm.weight"));
        let attn_b = vec_f32(g, &p("attn_output_norm.bias"));
        let up = mat(g, &p("ffn_up.weight"));
        let down = mat(g, &p("ffn_down.weight"));
        let up_b = vec_f32(g, &p("ffn_up.bias"));
        let down_b = vec_f32(g, &p("ffn_down.bias"));
        let out_w = vec_f32(g, &p("layer_output_norm.weight"));
        let out_b = vec_f32(g, &p("layer_output_norm.bias"));

        // No pre-norm: the layer's input is the previous layer's normalised output.
        let mut q = vec![0.0f32; n * qd];
        let mut k = vec![0.0f32; n * kvd];
        let mut v = vec![0.0f32; n * kvd];
        for r in 0..n {
            let xr = &x[r * dm..(r + 1) * dm];
            wq.apply(xr, &mut q[r * qd..(r + 1) * qd]);
            wk.apply(xr, &mut k[r * kvd..(r + 1) * kvd]);
            wv.apply(xr, &mut v[r * kvd..(r + 1) * kvd]);
            for d in 0..qd {
                q[r * qd + d] = (q[r * qd + d] + bq[d]) * scale;
            }
            for d in 0..kvd {
                k[r * kvd + d] += bk[d];
                v[r * kvd + d] += bv[d];
            }
        }
        // Attention: the engine's bidirectional block (S4.9g's strip kernel), one
        // text, one worker. It is a *shared* kernel like the matmul chain: its
        // arithmetic is pinned by S4.9g's own cross-tier and cross-thread tests, and
        // this reference owns the graph around it.
        let pool = peregrine_core::pool::Pool::new(1);
        let mut stats = peregrine_core::model::StepStats::default();
        let mut scratch = peregrine_core::blocks::attn::BidirScratch::default();
        scratch.resize(n, hd, 1);
        let span = [peregrine_core::blocks::bidir::TextSpan { row0: 0, len: n }];
        let mut attn = vec![0.0f32; n * qd];
        peregrine_core::blocks::bidir::attend_texts(
            cfg,
            &q,
            &k,
            &v,
            &mut attn,
            &span,
            None,
            1.0,
            &mut scratch,
            &pool,
            &mut stats,
        );
        // attn_output (+ bias), residual, attn_output_norm.
        let mut o = vec![0.0f32; n * dm];
        for r in 0..n {
            wo.apply(&attn[r * qd..(r + 1) * qd], &mut o[r * dm..(r + 1) * dm]);
            for d in 0..dm {
                o[r * dm + d] += bo[d];
            }
        }
        for i in 0..n * dm {
            o[i] += x[i];
        }
        let mut xnorm = vec![0.0f32; n * dm];
        peregrine_core::vmath::layer_norm_rows(&o, &attn_w, &attn_b, eps, dm, n, &mut xnorm);

        // Feed-forward: up (+ bias), the exact GELU, down (+ bias), residual, norm.
        let mut hb = vec![0.0f32; n * cfg.n_ff];
        for r in 0..n {
            up.apply(
                &xnorm[r * dm..(r + 1) * dm],
                &mut hb[r * cfg.n_ff..(r + 1) * cfg.n_ff],
            );
            for d in 0..cfg.n_ff {
                hb[r * cfg.n_ff + d] += up_b[d];
            }
        }
        peregrine_core::vmath::gelu_erf_rows(&mut hb);
        let mut o = vec![0.0f32; n * dm];
        for r in 0..n {
            down.apply(
                &hb[r * cfg.n_ff..(r + 1) * cfg.n_ff],
                &mut o[r * dm..(r + 1) * dm],
            );
            for d in 0..dm {
                o[r * dm + d] += down_b[d];
            }
        }
        for i in 0..n * dm {
            o[i] += xnorm[i];
        }
        let mut out = vec![0.0f32; n * dm];
        peregrine_core::vmath::layer_norm_rows(&o, &out_w, &out_b, eps, dm, n, &mut out);
        x = out;
    }
    x
}

/// The pooled vector: the last layer's rows reduced to one, before truncation and
/// L2 normalisation (the engine's `embed::finish`).
pub fn reference_embedding(model: &Model, tokens: &[u32], pooling: Pooling) -> Vec<f32> {
    let dm = model.cfg.d_model;
    let hidden = hidden_rows(model, tokens);
    let mut sum = vec![0.0f64; dm];
    let mut v = vec![0.0f32; dm];
    peregrine_core::embed::pool_rows_into(&hidden, dm, pooling, &mut sum, &mut v);
    v
}

/// One text's classifier-head row (roadmap S4.6b), the reference's own transcription of
/// llama.cpp's `build_pooling` rank branch (`src/llama-graph.cpp:4185-4226`):
///
/// ```text
/// h = row 0 of the last layer (or the mean, `bert.classifier.pooling_type = 1`)
/// if cls:      h = cls.weight · h + cls.bias; h = activation(h)   // tanh, gelu, silu
/// if cls.out:  y = cls.out.weight · h + cls.out.bias
/// ```
///
/// The matrices go through [`Mat`], the engine's defined float/int8 chain, so an F32 file's
/// head row is bit for bit the engine's.
pub fn reference_head(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let g = model.gguf();
    let dm = model.cfg.d_model;
    let hidden = hidden_rows(model, tokens);
    // `classifier.pooling_type = 1` is the mean; anything else is row 0.
    let mut h = if g.u32("bert.classifier.pooling_type") == Ok(1) {
        let mut sum = vec![0.0f64; dm];
        let mut v = vec![0.0f32; dm];
        peregrine_core::embed::pool_rows_into(&hidden, dm, Pooling::Mean, &mut sum, &mut v);
        v
    } else {
        hidden[..dm].to_vec()
    };
    let hw = head_width(g);
    if g.tensor("cls.weight").is_ok() {
        let w = mat(g, "cls.weight");
        let mut out = vec![0.0f32; dm];
        w.apply(&h, &mut out);
        if g.tensor("cls.bias").is_ok() {
            for (o, b) in out.iter_mut().zip(vec_f32(g, "cls.bias")) {
                *o += b;
            }
        }
        // `classifier.activation`: llama.cpp's default is TANH (`hparams.act_cls`).
        match g.str("bert.classifier.activation") {
            Ok("gelu") => peregrine_core::vmath::gelu_erf_rows(&mut out),
            Ok("silu") => {
                for v in out.iter_mut() {
                    *v *= 1.0 / (1.0 + (-*v).exp());
                }
            }
            _ => {
                // `ggml_vec_tanh_f32` is `tanhf`.
                for v in out.iter_mut() {
                    *v = v.tanh();
                }
            }
        }
        h = out;
    }
    if g.tensor("cls.output.weight").is_ok() {
        let w = mat(g, "cls.output.weight");
        let mut out = vec![0.0f32; hw];
        w.apply(&h, &mut out);
        if g.tensor("cls.output.bias").is_ok() {
            for (o, b) in out.iter_mut().zip(vec_f32(g, "cls.output.bias")) {
                *o += b;
            }
        }
        out
    } else {
        // A head without `cls.output` is its dense layer's activated output.
        h[..hw].to_vec()
    }
}

/// The width of a synthetic head's row: `cls.output`'s rows, or `d_model` without one.
fn head_width(g: &Gguf) -> usize {
    match g.tensor("cls.output.weight") {
        Ok(t) => t.dims.get(1).copied().unwrap_or(1) as usize,
        Err(_) => g.tensor("cls.weight").map_or(0, |t| t.dims[0] as usize),
    }
}
