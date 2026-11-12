//! An exact scalar reference forward for `nomic-bert` (roadmap S4.8a).
//!
//! Written independently of `peregrine_models::nomic_bert::NomicBert::forward_texts`:
//! the graph transcribed from llama.cpp v0.9.0 (`src/models/bert.cpp`, the
//! `LLM_ARCH_NOMIC_BERT` branches) is reimplemented here layer by layer, on one text
//! at a time, with plain loops. It shares only the *defined* kernels with the engine
//! — `vmath::layer_norm_row`, `vmath::swiglu`, `blocks::rope::rope` and the float
//! matmul's FMA chain (`row_dot`, exactly `bert_ref`'s) — because the test's claim is
//! that the *forward graph and its order* are right, and those primitives are what
//! make an F32 model reproduce bit for bit.
//!
//! The arithmetic the reference owns, and which it must get right on its own: which
//! tensors are read, the token type row, the fused `attn_qkv` split (q, then k, then
//! v), the optional fused bias, RoPE with NeoX pairing at `rope.freq_base`, the
//! `1/√head_dim` scale, the two residuals and their LayerNorms (weight *and* bias),
//! the SwiGLU pair (`silu(ffn_gate) ⊙ ffn_up`), and the absence of a final norm.
//! The `n × n` attention itself is the engine's block
//! (`blocks::bidir::attend_texts`), called here for one text on one worker, the same
//! way `bert_ref` and `modern_bert_ref` share it.

use peregrine_core::embed::Pooling;
use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::model::Model;
use peregrine_kernels::quant::{dequant_row, dot_f32_ref};
use peregrine_kernels::table::MatKernels;

/// One float matrix row times an activation row, in the arithmetic the engine
/// defines for this build (the same helper `bert_ref` uses).
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

/// One projection's weights, in the file's format (F32's defined float chain, or
/// the engine's int8 path with the activation row quantised once).
struct Mat {
    kind: Format,
    bytes: Vec<u8>,
    d_in: usize,
    d_out: usize,
}

impl Mat {
    fn row(&self, r: usize) -> &[u8] {
        let rb = match self.kind {
            Format::F32 => self.d_in * 4,
            Format::Q8_0 => peregrine_core::quant::row_bytes(Format::Q8_0, self.d_in),
            other => panic!("the nomic-bert reference has no {other:?} path"),
        };
        &self.bytes[r * rb..(r + 1) * rb]
    }
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        match self.kind {
            Format::F32 => {
                for (r, o) in out[..self.d_out].iter_mut().enumerate() {
                    *o = row_dot(self.row(r), x);
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
        };
    }
    Mat {
        kind: t.kind,
        bytes: g.bytes(t).to_vec(),
        d_in,
        d_out,
    }
}

fn vec_f32(g: &Gguf, name: &str) -> Vec<f32> {
    let t = g.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let n = t.numel() as usize;
    let mut v = vec![0.0f32; n];
    dequant_row(t.kind, g.bytes(t), &mut v);
    v
}

/// One table row (the token embedding or a token type), dequantised.
fn table_row(g: &Gguf, name: &str, r: usize, n: usize) -> Vec<f32> {
    let t = g.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let rb = peregrine_core::quant::row_bytes(t.kind, n);
    let mut v = vec![0.0f32; n];
    dequant_row(t.kind, &g.bytes(t)[r * rb..(r + 1) * rb], &mut v);
    v
}

/// Every row of a LayerNorm (weight and bias), one row at a time.
fn norm_row(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    peregrine_core::vmath::layer_norm_row(x, w, b, eps, &mut out);
    out
}

/// The last layer's rows for one text (the encoder's output; there is no final norm).
pub fn hidden_rows(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let g = model.gguf();
    let cfg = &model.cfg;
    let arch = "nomic-bert";
    let dm = cfg.d_model;
    let qd = cfg.q_dim();
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let ff = cfg.n_ff;
    let n = tokens.len();
    let eps = cfg.eps;
    let scale = 1.0 / (hd as f32).sqrt();
    let base = g.f32(&format!("{arch}.rope.freq_base")).unwrap_or(10_000.0);
    let table = peregrine_core::blocks::rope::build_rope_table(cfg.rope_dim, base, n.max(1), None);

    let mut x = vec![0.0f32; n * dm];
    let type0 = match g.tensor("token_types.weight") {
        Ok(_) => Some(table_row(g, "token_types.weight", 0, dm)),
        Err(_) => None,
    };
    for (r, &tok) in tokens.iter().enumerate() {
        let row = table_row(g, "token_embd.weight", tok as usize, dm);
        x[r * dm..(r + 1) * dm].copy_from_slice(&row);
        if let Some(t0) = &type0 {
            for (v, &b) in x[r * dm..(r + 1) * dm].iter_mut().zip(t0) {
                *v += b;
            }
        }
    }
    let tok_norm_w = vec_f32(g, "token_embd_norm.weight");
    let tok_norm_b = vec_f32(g, "token_embd_norm.bias");
    let mut normed = vec![0.0f32; n * dm];
    for r in 0..n {
        let row = norm_row(&x[r * dm..(r + 1) * dm], &tok_norm_w, &tok_norm_b, eps);
        normed[r * dm..(r + 1) * dm].copy_from_slice(&row);
    }
    x = normed;

    let pool = peregrine_core::pool::Pool::new(1);
    let mut stats = peregrine_core::model::StepStats::default();
    let mut scratch = peregrine_core::blocks::attn::BidirScratch::default();
    scratch.resize(n, hd, 1);
    let span = [peregrine_core::blocks::bidir::TextSpan { row0: 0, len: n }];

    for il in 0..cfg.n_layer {
        let p = |s: &str| format!("blk.{il}.{s}");
        let wqkv = mat(g, &p("attn_qkv.weight"));
        let wo = mat(g, &p("attn_output.weight"));
        let gate = mat(g, &p("ffn_gate.weight"));
        let up = mat(g, &p("ffn_up.weight"));
        let down = mat(g, &p("ffn_down.weight"));
        let bqkv = match g.tensor(&p("attn_qkv.bias")) {
            Ok(_) => Some(vec_f32(g, &p("attn_qkv.bias"))),
            Err(_) => None,
        };
        let bo = match g.tensor(&p("attn_output.bias")) {
            Ok(_) => Some(vec_f32(g, &p("attn_output.bias"))),
            Err(_) => None,
        };

        // The fused q/k/v matrix: q rows, then k, then v, one matmul per row, then
        // the fused bias (llama.cpp adds it before RoPE).
        let mut q = vec![0.0f32; n * qd];
        let mut k = vec![0.0f32; n * kvd];
        let mut v = vec![0.0f32; n * kvd];
        let mut fused = vec![0.0f32; qd + 2 * kvd];
        for r in 0..n {
            wqkv.apply(&x[r * dm..(r + 1) * dm], &mut fused[..qd + 2 * kvd]);
            if let Some(b) = &bqkv {
                for (o, &b) in fused[..qd].iter_mut().zip(&b[..qd]) {
                    *o += b;
                }
                for (o, &b) in fused[qd..qd + kvd].iter_mut().zip(&b[qd..qd + kvd]) {
                    *o += b;
                }
                for (o, &b) in fused[qd + kvd..].iter_mut().zip(&b[qd + kvd..]) {
                    *o += b;
                }
            }
            q[r * qd..(r + 1) * qd].copy_from_slice(&fused[..qd]);
            k[r * kvd..(r + 1) * kvd].copy_from_slice(&fused[qd..qd + kvd]);
            v[r * kvd..(r + 1) * kvd].copy_from_slice(&fused[qd + kvd..]);
        }

        // RoPE, NeoX pairing, on q and k only.
        for r in 0..n {
            peregrine_core::blocks::rope::rope(
                &table,
                &mut q[r * qd..(r + 1) * qd],
                cfg.n_head,
                r,
                hd,
                cfg.rope_dim,
                cfg.rope_neox,
            );
            peregrine_core::blocks::rope::rope(
                &table,
                &mut k[r * kvd..(r + 1) * kvd],
                cfg.n_kv_head,
                r,
                hd,
                cfg.rope_dim,
                cfg.rope_neox,
            );
            for d in 0..qd {
                q[r * qd + d] *= scale;
            }
        }

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

        // attn_output (+ bias), the residual, then attn_output_norm.
        let mut o = vec![0.0f32; n * dm];
        for r in 0..n {
            wo.apply(&attn[r * qd..(r + 1) * qd], &mut o[r * dm..(r + 1) * dm]);
            if let Some(b) = &bo {
                for (v, &b) in o[r * dm..(r + 1) * dm].iter_mut().zip(b) {
                    *v += b;
                }
            }
            for d in 0..dm {
                o[r * dm + d] += x[r * dm + d];
            }
        }
        let anw = vec_f32(g, &p("attn_output_norm.weight"));
        let anb = vec_f32(g, &p("attn_output_norm.bias"));
        let mut ffn_in = vec![0.0f32; n * dm];
        for r in 0..n {
            let row = norm_row(&o[r * dm..(r + 1) * dm], &anw, &anb, eps);
            ffn_in[r * dm..(r + 1) * dm].copy_from_slice(&row);
        }

        // The SwiGLU pair: `silu(ffn_gate · x) ⊙ (ffn_up · x)`, then `ffn_down`.
        let mut g_row = vec![0.0f32; ff];
        let mut u_row = vec![0.0f32; ff];
        let mut hb = vec![0.0f32; n * ff];
        for r in 0..n {
            gate.apply(&ffn_in[r * dm..(r + 1) * dm], &mut g_row);
            up.apply(&ffn_in[r * dm..(r + 1) * dm], &mut u_row);
            peregrine_core::vmath::swiglu(&mut g_row, &u_row);
            hb[r * ff..(r + 1) * ff].copy_from_slice(&g_row);
        }
        let mut down_out = vec![0.0f32; n * dm];
        let onw = vec_f32(g, &p("layer_output_norm.weight"));
        let onb = vec_f32(g, &p("layer_output_norm.bias"));
        for r in 0..n {
            down.apply(
                &hb[r * ff..(r + 1) * ff],
                &mut down_out[r * dm..(r + 1) * dm],
            );
            // `x = down + ffn_inp`, where `ffn_inp` is the normalised attention
            // output — llama.cpp's second `ggml_add`; then `layer_output_norm`.
            for d in 0..dm {
                down_out[r * dm + d] += ffn_in[r * dm + d];
            }
            let row = norm_row(&down_out[r * dm..(r + 1) * dm], &onw, &onb, eps);
            x[r * dm..(r + 1) * dm].copy_from_slice(&row);
        }
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
