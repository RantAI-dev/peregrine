//! An exact scalar reference forward for `jina-bert-v2` (roadmap S4.8c).
//!
//! Written independently of `peregrine_models::jina_bert::JinaBert::forward_texts`:
//! the graph transcribed from llama.cpp v0.9.0 (`src/models/bert.cpp`'s
//! `LLM_ARCH_JINA_BERT_V2` branches, `src/models/jina-bert-v2.cpp`) is reimplemented
//! here layer by layer, on one text at a time, with plain loops. It shares only the
//! *defined* kernels with the engine — the LayerNorm row, `vmath::geglu` and the
//! float matmul's FMA chain (`row_dot`) — because the test's claim is that the
//! *forward graph and its order* are right.
//!
//! The arithmetic the reference owns, and which it must get right on its own: the
//! token-type row, which tensors are read, the q/k norms' width (a token's whole
//! query row), ALiBi's slopes and where the bias enters (after the scale, before
//! the maximum), the optional second residual and norm, the GEGLU pair (separate or
//! fused into the first half of `ffn_up`), all the biases, and the fact that the
//! layer's `layer_output_norm` is the model's output — there is no final norm.
//! Attention is written out here rather than called from `blocks::bidir`, so the
//! engine's ALiBi kernel has an independent check.

use peregrine_core::embed::Pooling;
use peregrine_core::gguf::Gguf;
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
        dot_f32_ref(peregrine_core::gguf::Format::F32, row, x)
    }
}

/// One projection's weights, in the file's format (F32's defined float chain, or
/// the engine's int8 path with the activation row quantised once).
struct Mat {
    kind: peregrine_core::gguf::Format,
    bytes: Vec<u8>,
    d_in: usize,
    d_out: usize,
}

impl Mat {
    fn row(&self, r: usize) -> &[u8] {
        let rb = match self.kind {
            peregrine_core::gguf::Format::F32 => self.d_in * 4,
            peregrine_core::gguf::Format::Q8_0 => {
                peregrine_core::quant::row_bytes(peregrine_core::gguf::Format::Q8_0, self.d_in)
            }
            other => panic!("the jina-bert-v2 reference has no {other:?} path"),
        };
        &self.bytes[r * rb..(r + 1) * rb]
    }
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        match self.kind {
            peregrine_core::gguf::Format::F32 => {
                for (r, o) in out[..self.d_out].iter_mut().enumerate() {
                    *o = row_dot(self.row(r), x);
                }
            }
            peregrine_core::gguf::Format::Q8_0 => {
                let mut a = peregrine_core::quant::QuantAct::with_capacity(self.d_in);
                a.fill(x);
                for (r, o) in out[..self.d_out].iter_mut().enumerate() {
                    let row = self.row(r).to_vec();
                    *o = peregrine_kernels::quant::dot_scalar(
                        peregrine_core::gguf::Format::Q8_0,
                        &row,
                        &a,
                    );
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
            kind: peregrine_core::gguf::Format::F32,
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

fn zeros_or_vec(g: &Gguf, name: &str, n: usize) -> Vec<f32> {
    match g.tensor(name) {
        Ok(_) => vec_f32(g, name),
        Err(_) => vec![0.0; n],
    }
}

/// One table row (the token or token-type embedding), dequantised.
fn table_row(g: &Gguf, name: &str, r: usize, n: usize) -> Vec<f32> {
    let t = g.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let rb = peregrine_core::quant::row_bytes(t.kind, n);
    let mut v = vec![0.0f32; n];
    dequant_row(t.kind, &g.bytes(t)[r * rb..(r + 1) * rb], &mut v);
    v
}

/// llama.cpp's ALiBi slopes (`f_max_alibi_bias = 8`), computed here from the formula
/// so the engine's `bidir::alibi_slopes` has an independent check.
fn slopes(n_head: usize) -> Vec<f32> {
    let n_head_log2 = 1usize << (usize::BITS - 1 - n_head.leading_zeros());
    let l = n_head_log2 as f32;
    // The engine's platform-independent power, as the kernel uses: the reference is
    // independent in its formula and its loop, not in its C library.
    let pow = peregrine_core::vmath::det_pow_f32;
    let m0 = pow(2.0, -8.0 / l);
    let m1 = pow(2.0, -4.0 / l);
    (0..n_head)
        .map(|h| {
            if h < n_head_log2 {
                pow(m0, h as f32 + 1.0)
            } else {
                pow(m1, 2.0 * (h - n_head_log2) as f32 + 1.0)
            }
        })
        .collect()
}

/// One text's last-layer rows, exactly as llama.cpp's graph computes them.
pub fn hidden_rows(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let g = model.gguf();
    let cfg = &model.cfg;
    let dm = cfg.d_model;
    let qd = cfg.q_dim();
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let ff = cfg.n_ff;
    let n = tokens.len();
    let eps = cfg.eps;
    let scale = 1.0 / (hd as f32).sqrt();
    let slopes = slopes(cfg.n_head);

    // x = token_embd[token] + token_types[0]
    let type0 = table_row(g, "token_types.weight", 0, dm);
    let mut x = vec![0.0f32; n * dm];
    for (r, &tok) in tokens.iter().enumerate() {
        let row = table_row(g, "token_embd.weight", tok as usize, dm);
        for d in 0..dm {
            x[r * dm + d] = row[d] + type0[d];
        }
    }
    let tok_w = vec_f32(g, "token_embd_norm.weight");
    let tok_b = vec_f32(g, "token_embd_norm.bias");
    let mut out = vec![0.0f32; n * dm];
    for r in 0..n {
        peregrine_core::vmath::layer_norm_row(
            &x[r * dm..(r + 1) * dm],
            &tok_w,
            &tok_b,
            eps,
            &mut out[r * dm..(r + 1) * dm],
        );
    }
    x = out;

    let mut s = vec![0.0f32; n];
    let mut e = vec![0.0f32; n];
    for il in 0..cfg.n_layer {
        let p = |suffix: &str| format!("blk.{il}.{suffix}");
        let wq = mat(g, &p("attn_q.weight"));
        let wk = mat(g, &p("attn_k.weight"));
        let wv = mat(g, &p("attn_v.weight"));
        let wo = mat(g, &p("attn_output.weight"));
        let bq = vec_f32(g, &p("attn_q.bias"));
        let bk = vec_f32(g, &p("attn_k.bias"));
        let bv = vec_f32(g, &p("attn_v.bias"));
        let bo = vec_f32(g, &p("attn_output.bias"));

        // q/k/v, with the optional per-token LayerNorm over the whole row.
        let mut q = vec![0.0f32; n * qd];
        let mut k = vec![0.0f32; n * kvd];
        let mut v = vec![0.0f32; n * kvd];
        for r in 0..n {
            let xr = &x[r * dm..(r + 1) * dm];
            wq.apply(xr, &mut q[r * qd..(r + 1) * qd]);
            wk.apply(xr, &mut k[r * kvd..(r + 1) * kvd]);
            wv.apply(xr, &mut v[r * kvd..(r + 1) * kvd]);
            for d in 0..qd {
                q[r * qd + d] += bq[d];
            }
            for d in 0..kvd {
                k[r * kvd + d] += bk[d];
                v[r * kvd + d] += bv[d];
            }
        }
        if g.tensor(&p("attn_q_norm.weight")).is_ok() {
            let w = vec_f32(g, &p("attn_q_norm.weight"));
            let b = zeros_or_vec(g, &p("attn_q_norm.bias"), qd);
            let mut o = vec![0.0f32; qd];
            for r in 0..n {
                peregrine_core::vmath::layer_norm_row(
                    &q[r * qd..(r + 1) * qd],
                    &w,
                    &b,
                    eps,
                    &mut o,
                );
                q[r * qd..(r + 1) * qd].copy_from_slice(&o);
            }
        }
        if g.tensor(&p("attn_k_norm.weight")).is_ok() {
            let w = vec_f32(g, &p("attn_k_norm.weight"));
            let b = zeros_or_vec(g, &p("attn_k_norm.bias"), kvd);
            let mut o = vec![0.0f32; kvd];
            for r in 0..n {
                peregrine_core::vmath::layer_norm_row(
                    &k[r * kvd..(r + 1) * kvd],
                    &w,
                    &b,
                    eps,
                    &mut o,
                );
                k[r * kvd..(r + 1) * kvd].copy_from_slice(&o);
            }
        }

        // Attention: one n × n block. The scale is llama.cpp's `kq_scale`, applied to
        // the score before ALiBi's bias, as the engine's block does.
        let mut attn = vec![0.0f32; n * qd];
        for h in 0..cfg.n_head {
            for i in 0..n {
                for (j, o) in s.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for d in 0..hd {
                        acc = q[i * qd + h * hd + d].mul_add(k[j * kvd + h * hd + d], acc);
                    }
                    *o = acc * scale;
                }
                for (j, o) in s.iter_mut().enumerate() {
                    *o += slopes[h] * -(j.abs_diff(i) as f32);
                }
                let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for j in 0..n {
                    e[j] = peregrine_core::vmath::exp_scalar(s[j] - max);
                }
                let mut sum = 0.0f64;
                for &x in &e {
                    sum += x as f64;
                }
                let inv = (1.0f64 / sum) as f32;
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for j in 0..n {
                        acc = e[j].mul_add(v[j * kvd + h * hd + d], acc);
                    }
                    attn[i * qd + h * hd + d] = acc * inv;
                }
            }
        }

        // attn_output + bias, the residual, attn_output_norm.
        let mut o = vec![0.0f32; n * dm];
        let mut one = vec![0.0f32; dm];
        for r in 0..n {
            wo.apply(&attn[r * qd..(r + 1) * qd], &mut one);
            for d in 0..dm {
                o[r * dm + d] = (one[d] + bo[d]) + x[r * dm + d];
            }
        }
        let nw = vec_f32(g, &p("attn_output_norm.weight"));
        let nbias = vec_f32(g, &p("attn_output_norm.bias"));
        let mut normed = vec![0.0f32; n * dm];
        for r in 0..n {
            peregrine_core::vmath::layer_norm_row(
                &o[r * dm..(r + 1) * dm],
                &nw,
                &nbias,
                eps,
                &mut normed[r * dm..(r + 1) * dm],
            );
        }
        let mut ffn_inp = normed;
        if g.tensor(&p("attn_norm_2.weight")).is_ok() {
            let nw = vec_f32(g, &p("attn_norm_2.weight"));
            let nbias = zeros_or_vec(g, &p("attn_norm_2.bias"), dm);
            let mut o2 = vec![0.0f32; n * dm];
            for r in 0..n {
                for d in 0..dm {
                    o2[r * dm + d] = ffn_inp[r * dm + d] + x[r * dm + d];
                }
            }
            let mut n2 = vec![0.0f32; n * dm];
            for r in 0..n {
                peregrine_core::vmath::layer_norm_row(
                    &o2[r * dm..(r + 1) * dm],
                    &nw,
                    &nbias,
                    eps,
                    &mut n2[r * dm..(r + 1) * dm],
                );
            }
            ffn_inp = n2;
        }

        // The GEGLU pair: separate gate/up, or the fused first half.
        let up = mat(g, &p("ffn_up.weight"));
        let up_b = zeros_or_vec(g, &p("ffn_up.bias"), up.d_out);
        let mut gated = vec![0.0f32; n * ff];
        let mut pair = vec![0.0f32; up.d_out];
        for r in 0..n {
            up.apply(&ffn_inp[r * dm..(r + 1) * dm], &mut pair);
            for (x, &b) in pair.iter_mut().zip(&up_b) {
                *x += b;
            }
            if g.tensor(&p("ffn_gate.weight")).is_ok() {
                let gate = mat(g, &p("ffn_gate.weight"));
                let mut gv = vec![0.0f32; ff];
                gate.apply(&ffn_inp[r * dm..(r + 1) * dm], &mut gv);
                peregrine_core::vmath::geglu(&mut gv, &pair[..ff]);
                gated[r * ff..(r + 1) * ff].copy_from_slice(&gv);
            } else {
                let (gate, u) = pair.split_at_mut(ff);
                peregrine_core::vmath::geglu(gate, u);
                gated[r * ff..(r + 1) * ff].copy_from_slice(gate);
            }
        }
        let down = mat(g, &p("ffn_down.weight"));
        let down_b = vec_f32(g, &p("ffn_down.bias"));
        let mut o3 = vec![0.0f32; n * dm];
        let mut one = vec![0.0f32; dm];
        for r in 0..n {
            down.apply(&gated[r * ff..(r + 1) * ff], &mut one);
            for d in 0..dm {
                o3[r * dm + d] = (one[d] + down_b[d]) + ffn_inp[r * dm + d];
            }
        }
        let ow = vec_f32(g, &p("layer_output_norm.weight"));
        let ob = vec_f32(g, &p("layer_output_norm.bias"));
        let mut last = vec![0.0f32; n * dm];
        for r in 0..n {
            peregrine_core::vmath::layer_norm_row(
                &o3[r * dm..(r + 1) * dm],
                &ow,
                &ob,
                eps,
                &mut last[r * dm..(r + 1) * dm],
            );
        }
        x = last;
    }
    x
}

/// The pooled vector: the last layer's rows reduced to one, before truncation and L2
/// normalisation (the engine's `embed::finish`).
pub fn reference_embedding(model: &Model, tokens: &[u32], pooling: Pooling) -> Vec<f32> {
    let dm = model.cfg.d_model;
    let hidden = hidden_rows(model, tokens);
    let mut sum = vec![0.0f64; dm];
    let mut v = vec![0.0f32; dm];
    peregrine_core::embed::pool_rows_into(&hidden, dm, pooling, &mut sum, &mut v);
    v
}
