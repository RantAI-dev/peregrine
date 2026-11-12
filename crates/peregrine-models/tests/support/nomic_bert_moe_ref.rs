//! An exact scalar reference forward for `nomic-bert-moe` (roadmap S4.8b).
//!
//! Written independently of `peregrine_models::nomic_bert::NomicBert::forward_texts`:
//! the graph transcribed from llama.cpp v0.9.0 (`src/models/bert.cpp`, the
//! `LLM_ARCH_NOMIC_BERT_MOE` branches) is reimplemented here layer by layer, on one
//! text at a time, with plain loops. It shares only the *defined* kernels with the
//! engine — `vmath::layer_norm_row`, `vmath::swiglu`, `vmath::gelu_tanh`,
//! `blocks::rope::rope` and the float matmul's FMA chain (`row_dot`, exactly
//! `nomic_bert_ref`'s) — because the test's claim is that the *forward graph and its
//! order* are right.
//!
//! The arithmetic the reference owns, and which it must get right on its own: the
//! token type row, the fused `attn_qkv` split, the optional fused bias, RoPE with NeoX
//! pairing, the `1/√head_dim` scale, the two residuals and their LayerNorms, the plain
//! GELU feed-forward on the non-MoE layers (`(ffn_down · gelu(ffn_up · x + b_up)) +
//! b_down`), and the **mixture of experts** on the MoE layers: a softmax over every
//! expert written out longhand, the best `expert_used_count` with
//! probability-descending/expert-index-ascending ties, each chosen expert's
//! `ffn_down_exps · gelu(ffn_up_exps · x)` weighted by its probability, and the sum in
//! the router's order. The `n × n` attention is the engine's block
//! (`blocks::bidir::attend_texts`), called here for one text on one worker, as
//! `nomic_bert_ref` does.

use peregrine_core::embed::Pooling;
use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::model::Model;
use peregrine_kernels::quant::{dequant_row, dot_f32_ref};
use peregrine_kernels::table::MatKernels;
use peregrine_kernels::vmath::{det_exp, gelu_tanh};

/// One float matrix row times an activation row, in the arithmetic the engine
/// defines for this build (the same helper `nomic_bert_ref` uses).
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
    /// Whether the engine reads this matrix as a raw row (`mk.row`, i.e.
    /// `quant::dot_batch`) rather than through the float panels it packs for the
    /// model's declared linear groups. The expert banks and the router are 3D views
    /// or extra matrices the family does not report as linears, so they take the raw
    /// row kernel; the attention and plain feed-forward matrices are packed.
    raw_row: bool,
}

impl Mat {
    fn row(&self, r: usize) -> &[u8] {
        let rb = match self.kind {
            Format::F32 => self.d_in * 4,
            Format::Q8_0 => peregrine_core::quant::row_bytes(Format::Q8_0, self.d_in),
            other => panic!("the nomic-bert-moe reference has no {other:?} path"),
        };
        &self.bytes[r * rb..(r + 1) * rb]
    }
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        match self.kind {
            // A raw F32 row goes through the engine's per-row kernel, exactly as
            // `matmul_block`'s PerRow path calls it.
            Format::F32 if self.raw_row => {
                let mut a = peregrine_core::quant::QuantAct::with_capacity(self.d_in);
                a.keep_f32 = true;
                a.fill(x);
                let mut one = [0.0f32; 1];
                let rb = self.d_in * 4;
                for (r, o) in out[..self.d_out].iter_mut().enumerate() {
                    peregrine_kernels::quant::dot_batch(
                        Format::F32,
                        &self.bytes[r * rb..(r + 1) * rb],
                        std::slice::from_ref(&a),
                        &mut one,
                    );
                    *o = one[0];
                }
            }
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
            raw_row: false,
        };
    }
    Mat {
        kind: t.kind,
        bytes: g.bytes(t).to_vec(),
        d_in,
        d_out,
        raw_row: false,
    }
}

/// One expert's 2D view of a 3D bank, sliced by the reference's own arithmetic:
/// GGUF stores `dims[0]` contiguously and the expert index runs slowest, so expert
/// `e` is the contiguous run of `d_out` rows that starts `e · d_out` rows in.
fn expert_mat(g: &Gguf, name: &str, e: usize, d_in: usize, d_out: usize) -> Mat {
    let t = g.tensor(name).unwrap_or_else(|err| panic!("{name}: {err}"));
    assert_eq!(t.dims[0] as usize, d_in, "{name}");
    assert_eq!(t.dims[1] as usize, d_out, "{name}");
    let rb = match t.kind {
        Format::F32 => d_in * 4,
        Format::Q8_0 => peregrine_core::quant::row_bytes(Format::Q8_0, d_in),
        other => panic!("the nomic-bert-moe reference has no {other:?} expert path"),
    };
    let per = d_out * rb;
    Mat {
        kind: t.kind,
        bytes: g.bytes(t)[e * per..(e + 1) * per].to_vec(),
        d_in,
        d_out,
        raw_row: true,
    }
}

/// The router: a 2D float matrix, which the loader's float pack (`fpack::eligible`)
/// turns into panels like every other 2D float matrix, so its rows take the packed
/// chain (`row_dot`), not the raw-row kernel.
fn router_mat(g: &Gguf, name: &str) -> Mat {
    mat(g, name)
}

fn vec_f32(g: &Gguf, name: &str) -> Vec<f32> {
    let t = g.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let n = t.numel() as usize;
    let mut v = vec![0.0f32; n];
    dequant_row(t.kind, g.bytes(t), &mut v);
    v
}

fn opt_vec_f32(g: &Gguf, name: &str) -> Option<Vec<f32>> {
    match g.tensor(name) {
        Ok(_) => Some(vec_f32(g, name)),
        Err(_) => None,
    }
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

/// The router, written out longhand: a softmax over every expert (maximum subtracted,
/// the exponentials summed in f64, one f32 reciprocal), then the best `k` — probability
/// descending, expert index ascending on a tie — and their raw probabilities as the
/// weights (llama.cpp passes `norm_w = false`).
fn route_longhand(logits: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; logits.len()];
    let mut sum = 0.0f64;
    for (e, &l) in logits.iter().enumerate() {
        probs[e] = det_exp(l - max);
        sum += f64::from(probs[e]);
    }
    let inv = (1.0 / sum) as f32;
    for p in probs.iter_mut() {
        *p *= inv;
    }
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_by(|&a, &b| {
        probs[b as usize]
            .partial_cmp(&probs[a as usize])
            .unwrap()
            .then_with(|| a.cmp(&b))
    });
    order.truncate(k);
    let weights: Vec<f32> = order.iter().map(|&e| probs[e as usize]).collect();
    (order, weights)
}

/// The last layer's rows for one text (the encoder's output; there is no final norm).
pub fn hidden_rows(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let g = model.gguf();
    let cfg = &model.cfg;
    let arch = cfg.arch_name;
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

    let is_moe_arch = arch == "nomic-bert-moe";
    let moe_every = if is_moe_arch {
        g.u32(&format!("{arch}.moe_every_n_layers")).unwrap_or(0) as usize
    } else {
        0
    };
    let n_expert = g.u32(&format!("{arch}.expert_count")).unwrap_or(0) as usize;
    let n_used = g.u32(&format!("{arch}.expert_used_count")).unwrap_or(0) as usize;

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
        let bqkv = opt_vec_f32(g, &p("attn_qkv.bias"));
        let bo = opt_vec_f32(g, &p("attn_output.bias"));

        // The fused q/k/v matrix: q rows, then k, then v, one matmul per row, then
        // the fused bias (llama.cpp adds it before RoPE).
        let mut q = vec![0.0f32; n * qd];
        let mut k = vec![0.0f32; n * kvd];
        let mut v = vec![0.0f32; n * kvd];
        let mut fused = vec![0.0f32; qd + 2 * kvd];
        for r in 0..n {
            wqkv.apply(&x[r * dm..(r + 1) * dm], &mut fused[..qd + 2 * kvd]);
            if let Some(b) = &bqkv {
                for (o, &b) in fused.iter_mut().zip(b) {
                    *o += b;
                }
            }
            q[r * qd..(r + 1) * qd].copy_from_slice(&fused[..qd]);
            k[r * kvd..(r + 1) * kvd].copy_from_slice(&fused[qd..qd + kvd]);
            v[r * kvd..(r + 1) * kvd].copy_from_slice(&fused[qd + kvd..]);
        }

        // RoPE, NeoX pairing, on q and k only, then the attention scale folded into Q.
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

        // The feed-forward: llama.cpp's one arch branch.
        let mut ffn_out = vec![0.0f32; n * dm];
        let is_moe = is_moe_arch && moe_every > 0 && il % moe_every == 1;
        if is_moe {
            let router = router_mat(g, &p("ffn_gate_inp.weight"));
            let ups: Vec<Mat> = (0..n_expert)
                .map(|e| expert_mat(g, &p("ffn_up_exps.weight"), e, dm, ff))
                .collect();
            let downs: Vec<Mat> = (0..n_expert)
                .map(|e| expert_mat(g, &p("ffn_down_exps.weight"), e, ff, dm))
                .collect();
            let mut logits = vec![0.0f32; n_expert];
            let mut up_row = vec![0.0f32; ff];
            let mut down_row = vec![0.0f32; dm];
            for r in 0..n {
                router.apply(&ffn_in[r * dm..(r + 1) * dm], &mut logits);
                let (ids, weights) = route_longhand(&logits, n_used);
                let out = &mut ffn_out[r * dm..(r + 1) * dm];
                for (s, &e) in ids.iter().enumerate() {
                    let e = e as usize;
                    ups[e].apply(&ffn_in[r * dm..(r + 1) * dm], &mut up_row);
                    for u in up_row.iter_mut() {
                        *u = gelu_tanh(*u);
                    }
                    downs[e].apply(&up_row, &mut down_row);
                    // llama.cpp multiplies the expert's output by its probability and
                    // adds the slots in the router's order (`src/llama-graph.cpp:2596-2614`).
                    if s == 0 {
                        for d in 0..dm {
                            out[d] = weights[s] * down_row[d];
                        }
                    } else {
                        for d in 0..dm {
                            out[d] += weights[s] * down_row[d];
                        }
                    }
                }
                // The feed-forward's residual is added by the graph right after the
                // MoE output, exactly as for the plain branch.
                for d in 0..dm {
                    ffn_out[r * dm + d] += ffn_in[r * dm + d];
                }
            }
        } else if is_moe_arch {
            // The plain GELU layer: `down · gelu(up · x + b_up) + b_down`.
            let up = mat(g, &p("ffn_up.weight"));
            let down = mat(g, &p("ffn_down.weight"));
            let up_b = opt_vec_f32(g, &p("ffn_up.bias"));
            let down_b = opt_vec_f32(g, &p("ffn_down.bias"));
            let mut up_row = vec![0.0f32; ff];
            let mut down_row = vec![0.0f32; dm];
            for r in 0..n {
                up.apply(&ffn_in[r * dm..(r + 1) * dm], &mut up_row);
                if let Some(b) = &up_b {
                    for (u, &b) in up_row.iter_mut().zip(b) {
                        *u += b;
                    }
                }
                for u in up_row.iter_mut() {
                    *u = gelu_tanh(*u);
                }
                down.apply(&up_row, &mut down_row);
                for d in 0..dm {
                    let mut o = down_row[d];
                    if let Some(b) = &down_b {
                        o += b[d];
                    }
                    ffn_out[r * dm + d] = o + ffn_in[r * dm + d];
                }
            }
        } else {
            // `nomic-bert`'s SwiGLU pair.
            let gate = mat(g, &p("ffn_gate.weight"));
            let up = mat(g, &p("ffn_up.weight"));
            let down = mat(g, &p("ffn_down.weight"));
            let mut g_row = vec![0.0f32; ff];
            let mut u_row = vec![0.0f32; ff];
            let mut hb = vec![0.0f32; n * ff];
            for r in 0..n {
                gate.apply(&ffn_in[r * dm..(r + 1) * dm], &mut g_row);
                up.apply(&ffn_in[r * dm..(r + 1) * dm], &mut u_row);
                peregrine_core::vmath::swiglu(&mut g_row, &u_row);
                hb[r * ff..(r + 1) * ff].copy_from_slice(&g_row);
            }
            let mut down_out = vec![0.0f32; dm];
            for r in 0..n {
                down.apply(&hb[r * ff..(r + 1) * ff], &mut down_out);
                for d in 0..dm {
                    ffn_out[r * dm + d] = down_out[d] + ffn_in[r * dm + d];
                }
            }
        }

        let onw = vec_f32(g, &p("layer_output_norm.weight"));
        let onb = vec_f32(g, &p("layer_output_norm.bias"));
        for r in 0..n {
            let row = norm_row(&ffn_out[r * dm..(r + 1) * dm], &onw, &onb, eps);
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
