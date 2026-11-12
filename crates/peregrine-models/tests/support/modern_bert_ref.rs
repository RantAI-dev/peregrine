//! An exact scalar reference forward for `modern-bert` (roadmap S4.7).
//!
//! Written independently of `peregrine_models::modern_bert::ModernBert::forward_texts`:
//! the graph transcribed from llama.cpp v0.9.0 (`src/models/modern-bert.cpp`) is
//! reimplemented here layer by layer, on one text at a time, with plain loops. It
//! shares only the *defined* kernels with the engine — `vmath::layer_norm_row_nobias`,
//! `vmath::geglu_erf` / `vmath::geglu` / `vmath::swiglu`, `blocks::rope::rope` and
//! the float matmul's FMA chain (`row_dot`, exactly `bert_ref`'s) — because the
//! test's claim is that the *forward graph and its order* are right, and those
//! primitives are what make an F32 model reproduce bit for bit.
//!
//! The arithmetic the reference owns, and which it must get right on its own:
//! which tensors are read, which layer is local and which global, the window's
//! half width, which RoPE base each layer uses, the identity `attn_norm` of layer
//! 0, the split of the fused `attn_qkv` rows (q, then k, then v), the `1/√head_dim`
//! scale, the two residuals and the gated feed-forward's order. The `n × n`
//! attention itself is the engine's block (`blocks::bidir::attend_texts`), called
//! here for one text on one worker, the same way `bert_ref` shares it.

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
            other => panic!("the modern-bert reference has no {other:?} path"),
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

/// One table row (the token embedding), dequantised.
fn table_row(g: &Gguf, name: &str, r: usize, n: usize) -> Vec<f32> {
    let t = g.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let rb = peregrine_core::quant::row_bytes(t.kind, n);
    let mut v = vec![0.0f32; n];
    dequant_row(t.kind, &g.bytes(t)[r * rb..(r + 1) * rb], &mut v);
    v
}

/// Every row of a weight-less LayerNorm, one row at a time.
fn norm_row(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    peregrine_core::vmath::layer_norm_row_nobias(x, w, eps, &mut out);
    out
}

/// The gated feed-forward's activation: the first half activated times the second.
fn gated(gate: &[f32], up: &[f32], act: &str) -> Vec<f32> {
    let mut g = gate.to_vec();
    match act {
        "gelu" | "gelu_python" => peregrine_core::vmath::geglu_erf(&mut g, up),
        "silu" | "swish" | "swiglu" => peregrine_core::vmath::swiglu(&mut g, up),
        _ => peregrine_core::vmath::geglu(&mut g, up),
    }
    g
}

/// The last layer's rows for one text, after `output_norm`.
pub fn hidden_rows(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let g = model.gguf();
    let cfg = &model.cfg;
    let arch = "modern-bert";
    let dm = cfg.d_model;
    let qd = cfg.q_dim();
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let ff = cfg.n_ff;
    let n = tokens.len();
    let eps = cfg.eps;
    let scale = 1.0 / (hd as f32).sqrt();

    let window = g
        .u32(&format!("{arch}.attention.sliding_window"))
        .unwrap_or(0) as usize;
    let pattern = g
        .u32(&format!("{arch}.attention.sliding_window_pattern"))
        .map(|v| v as usize)
        .unwrap_or(3);
    let is_swa: Vec<bool> = if window > 0 {
        (0..cfg.n_layer)
            .map(|il| pattern == 0 || il % pattern != 0)
            .collect()
    } else {
        vec![false; cfg.n_layer]
    };
    let base = g.f32(&format!("{arch}.rope.freq_base")).unwrap_or(10_000.0);
    let base_local = g
        .f32(&format!("{arch}.rope.freq_base_swa"))
        .unwrap_or(10_000.0);
    let act = g
        .str(&format!("{arch}.hidden_activation"))
        .unwrap_or("gelu");
    let table = peregrine_core::blocks::rope::build_rope_table(cfg.rope_dim, base, n.max(1), None);
    let table_local =
        peregrine_core::blocks::rope::build_rope_table(cfg.rope_dim, base_local, n.max(1), None);

    let mut x = vec![0.0f32; n * dm];
    for (r, &tok) in tokens.iter().enumerate() {
        let row = table_row(g, "token_embd.weight", tok as usize, dm);
        x[r * dm..(r + 1) * dm].copy_from_slice(&row);
    }
    let tok_norm = vec_f32(g, "token_embd_norm.weight");
    let mut out = vec![0.0f32; n * dm];
    for r in 0..n {
        let row = norm_row(&x[r * dm..(r + 1) * dm], &tok_norm, eps);
        out[r * dm..(r + 1) * dm].copy_from_slice(&row);
    }
    x = out;

    let pool = peregrine_core::pool::Pool::new(1);
    let mut stats = peregrine_core::model::StepStats::default();
    let mut scratch = peregrine_core::blocks::attn::BidirScratch::default();
    scratch.resize(n, hd, 1);
    let span = [peregrine_core::blocks::bidir::TextSpan { row0: 0, len: n }];

    for (il, &swa) in is_swa.iter().enumerate().take(cfg.n_layer) {
        let p = |s: &str| format!("blk.{il}.{s}");
        let wqkv = mat(g, &p("attn_qkv.weight"));
        let wo = mat(g, &p("attn_output.weight"));
        let up = mat(g, &p("ffn_up.weight"));
        let down = mat(g, &p("ffn_down.weight"));
        let ffn_norm = vec_f32(g, &p("ffn_norm.weight"));
        let attn_norm = if il == 0 && g.tensor(&p("attn_norm.weight")).is_err() {
            None
        } else {
            Some(vec_f32(g, &p("attn_norm.weight")))
        };

        // The pre-attention norm (the identity on layer 0 when its tensor is absent).
        let attn_in = match &attn_norm {
            Some(w) => {
                let mut o = vec![0.0f32; n * dm];
                for r in 0..n {
                    let row = norm_row(&x[r * dm..(r + 1) * dm], w, eps);
                    o[r * dm..(r + 1) * dm].copy_from_slice(&row);
                }
                o
            }
            None => x.clone(),
        };

        // The fused q/k/v matrix: q rows, then k, then v, one matmul per row.
        let mut q = vec![0.0f32; n * qd];
        let mut k = vec![0.0f32; n * kvd];
        let mut v = vec![0.0f32; n * kvd];
        let mut fused = vec![0.0f32; qd + 2 * kvd];
        for r in 0..n {
            wqkv.apply(&attn_in[r * dm..(r + 1) * dm], &mut fused[..qd + 2 * kvd]);
            q[r * qd..(r + 1) * qd].copy_from_slice(&fused[..qd]);
            k[r * kvd..(r + 1) * kvd].copy_from_slice(&fused[qd..qd + kvd]);
            v[r * kvd..(r + 1) * kvd].copy_from_slice(&fused[qd + kvd..]);
        }

        // RoPE, NEOX pairing, the local base on a local layer.
        let table_l = if swa { &table_local } else { &table };
        for r in 0..n {
            peregrine_core::blocks::rope::rope(
                table_l,
                &mut q[r * qd..(r + 1) * qd],
                cfg.n_head,
                r,
                hd,
                cfg.rope_dim,
                cfg.rope_neox,
            );
            peregrine_core::blocks::rope::rope(
                table_l,
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

        let half = if swa { Some((window / 2).max(1)) } else { None };
        let mut attn = vec![0.0f32; n * qd];
        peregrine_core::blocks::bidir::attend_texts(
            cfg,
            &q,
            &k,
            &v,
            &mut attn,
            &span,
            half,
            1.0,
            &mut scratch,
            &pool,
            &mut stats,
        );

        // attn_output, the first residual; then ffn_norm.
        let mut o = vec![0.0f32; n * dm];
        for r in 0..n {
            wo.apply(&attn[r * qd..(r + 1) * qd], &mut o[r * dm..(r + 1) * dm]);
            for d in 0..dm {
                o[r * dm + d] += x[r * dm + d];
            }
        }
        let mut ffn_in = vec![0.0f32; n * dm];
        for r in 0..n {
            let row = norm_row(&o[r * dm..(r + 1) * dm], &ffn_norm, eps);
            ffn_in[r * dm..(r + 1) * dm].copy_from_slice(&row);
        }

        // ffn_up is 2·n_ff wide: the first half is activated, then multiplies the
        // second; ffn_down reads the n_ff-wide product.
        let mut pair = vec![0.0f32; 2 * ff];
        let mut hb = vec![0.0f32; n * ff];
        for r in 0..n {
            up.apply(&ffn_in[r * dm..(r + 1) * dm], &mut pair);
            let g_row = gated(&pair[..ff], &pair[ff..], act);
            hb[r * ff..(r + 1) * ff].copy_from_slice(&g_row);
        }
        let mut down_out = vec![0.0f32; n * dm];
        for r in 0..n {
            down.apply(
                &hb[r * ff..(r + 1) * ff],
                &mut down_out[r * dm..(r + 1) * dm],
            );
            // `x = down + ffn_inp`, where `ffn_inp` is the *unnormalised* first
            // residual (`o` here), exactly llama.cpp's second `ggml_add`.
            for d in 0..dm {
                x[r * dm + d] = down_out[r * dm + d] + o[r * dm + d];
            }
        }
    }

    let out_norm = vec_f32(g, "output_norm.weight");
    let mut normed = vec![0.0f32; n * dm];
    for r in 0..n {
        let row = norm_row(&x[r * dm..(r + 1) * dm], &out_norm, eps);
        normed[r * dm..(r + 1) * dm].copy_from_slice(&row);
    }
    normed
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

/// The width of the head's row: `cls.output`'s rows, or `d_model` without one.
fn head_width(g: &Gguf) -> usize {
    match g.tensor("cls.output.weight") {
        Ok(t) => t.dims.get(1).copied().unwrap_or(1) as usize,
        Err(_) => g.tensor("cls.weight").map_or(0, |t| t.dims[0] as usize),
    }
}

/// One text's classifier-head row, exactly as llama.cpp's `build_pooling` computes it
/// for this family (`src/llama-graph.cpp:4185-4220`, roadmap S4.7b):
///
/// ```text
/// h = pooled(output_norm rows)     // mean unless classifier.pooling_type names another
///                                  // type; the key's absence means mean
/// if cls:  h = cls.weight · h      // no bias: llama.cpp creates no cls.bias here
///          h = act(h)              // classifier.activation, GELU_ERF when absent
///          h = norm(h, cls.norm.w) // only when the tensor is there
/// if cls.output: y = cls.output.weight · h + cls.output.bias
/// ```
///
/// The reference reads `cls.bias` and `cls.norm.bias` not at all, because llama.cpp
/// does not create them for a `modern-bert` classification file: a file that carries
/// them computes the same row as one that does not.
pub fn reference_head(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let g = model.gguf();
    let cfg = &model.cfg;
    let dm = cfg.d_model;
    let hidden = hidden_rows(model, tokens);
    let mean = g
        .u32("modern-bert.classifier.pooling_type")
        .map_or(true, |v| v == 1);
    let mut h = if mean {
        let mut sum = vec![0.0f64; dm];
        let mut v = vec![0.0f32; dm];
        peregrine_core::embed::pool_rows_into(&hidden, dm, Pooling::Mean, &mut sum, &mut v);
        v
    } else {
        hidden[..dm].to_vec()
    };
    if g.tensor("cls.weight").is_ok() {
        let w = mat(g, "cls.weight");
        let mut out = vec![0.0f32; dm];
        w.apply(&h, &mut out);
        match g.str("modern-bert.classifier.activation") {
            Ok("silu") => {
                for v in out.iter_mut() {
                    *v *= 1.0 / (1.0 + (-*v).exp());
                }
            }
            Ok("tanh") => {
                for v in out.iter_mut() {
                    *v = v.tanh();
                }
            }
            // The family's default is the exact (erf) GELU, not `bert`'s tanh.
            _ => peregrine_core::vmath::gelu_erf_rows(&mut out),
        }
        h = match g.tensor("cls.norm.weight") {
            Ok(_) => norm_row(&out, &vec_f32(g, "cls.norm.weight"), cfg.eps),
            Err(_) => out,
        };
    }
    let hw = head_width(g);
    match g.tensor("cls.output.weight") {
        Ok(_) => {
            let w = mat(g, "cls.output.weight");
            let mut out = vec![0.0f32; hw];
            w.apply(&h, &mut out);
            if g.tensor("cls.output.bias").is_ok() {
                for (o, b) in out.iter_mut().zip(vec_f32(g, "cls.output.bias")) {
                    *o += b;
                }
            }
            out
        }
        // A head without `cls.output` is its dense layer's activated, normed row.
        Err(_) => h[..hw].to_vec(),
    }
}
