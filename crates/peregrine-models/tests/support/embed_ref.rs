//! An exact scalar reference forward for `gemma-embedding` (roadmap S4.4).
//!
//! Written independently of `peregrine_models::gemma3::Gemma3::forward_texts`: the
//! transcribed llama.cpp graph is reimplemented here layer by layer, on one text at a
//! time, with plain loops. It uses the same arithmetic primitives the engine uses —
//! `row_dot` for a matrix row (the packed kernel's FMA chain, or the old partial-sum
//! dot where the tier has no packed kernel), `sumsq_f64` for a norm, `gelu_tanh`, the
//! rotary table — because the test's claim is that the *forward graph and its order*
//! are right, and those primitives are what makes an F32 model reproduce bit for bit.
//! The synthetic reference model is F32; other formats are checked by the invariance
//! and golden tests.
//!
//! Attention is written out here as the *definition* the kernels implement (S4.9g): one
//! sequential FMA chain per score over the head dimension, `vmath::exp_scalar` for the
//! weights, an f64 denominator summed in ascending key order, and a value mix that
//! folds the keys in ascending order — all over the row's exact window.
//!
//! The mask rule is llama.cpp v0.9.0's symmetric window (`src/llama-hparams.h`,
//! `is_masked_swa`): `|p1 - p0| > n_swa / 2` is masked. Pooling is `pool_rows`' rule:
//! f64 over the text's rows in position order, one rounding.

use peregrine_core::embed::Pooling;
use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::model::Model;
use peregrine_kernels::quant::{dequant_row, dot_f32_ref};
use peregrine_kernels::table::MatKernels;

/// One float matrix row times an activation row, in the arithmetic the engine defines
/// for this build.
///
/// Where the tier runs the packed float kernel (S4.9f: any x86-64 tier unless
/// `PEREGRINE_FPACK=0`) a matrix row is one sequential `f32::mul_add` chain from
/// `+0.0`, which is `fpack::gemm_scalar` and, lane per output, the vector kernels;
/// it is reproduced here from the row's f32 words rather than called, so the graph
/// stays the thing under test. Where the kernel is not in the table (aarch64) the
/// engine multiplies a float row with eight partial sums and a horizontal fold,
/// which `dot_f32_ref` pins. The predicate is the kernel table's own, so the
/// reference and the engine cannot disagree about which path a tier takes.
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

struct Mat {
    bytes: Vec<u8>,
    d_in: usize,
    d_out: usize,
}

impl Mat {
    fn row(&self, r: usize) -> &[u8] {
        &self.bytes[r * self.d_in * 4..(r + 1) * self.d_in * 4]
    }
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        for (r, o) in out[..self.d_out].iter_mut().enumerate() {
            *o = row_dot(self.row(r), x);
        }
    }
}

fn mat(g: &Gguf, name: &str) -> Mat {
    let t = g.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(t.kind, Format::F32, "{name} is not F32");
    let d_in = t.dims[0] as usize;
    let d_out = t.dims.get(1).copied().unwrap_or(1) as usize;
    Mat {
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

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss = peregrine_core::vmath::sumsq_f64(x) / x.len() as f64;
    let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
    x.iter().zip(w).map(|(&xi, &wi)| xi * inv * wi).collect()
}

fn head_rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let ss = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
    for (v, g) in x.iter_mut().zip(w) {
        *v = *v * inv * g;
    }
}

/// The final normed hidden state of every row (llama.cpp's `result_norm`), in order.
pub fn hidden_rows(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let g = model.gguf();
    let cfg = &model.cfg;
    let dm = cfg.d_model;
    let hd = cfg.head_dim;
    let qd = cfg.q_dim();
    let kvd = cfg.kv_dim();
    let n = tokens.len();
    let arch = "gemma-embedding";

    let tok_embd = mat(g, "token_embd.weight");
    let out_norm = vec_f32(g, "output_norm.weight");
    let eps = cfg.eps;
    let scale = 1.0 / (hd as f32).sqrt();
    let window = g
        .u32(&format!("{arch}.attention.sliding_window"))
        .unwrap_or(0) as usize;
    let rope_base = g
        .f32(&format!("{arch}.rope.freq_base"))
        .unwrap_or(1_000_000.0);
    let rope_base_swa = g
        .f32(&format!("{arch}.rope.freq_base_swa"))
        .unwrap_or(10_000.0);
    let rope_global =
        peregrine_core::blocks::rope::build_rope_table(cfg.rope_dim, rope_base, n.max(1), None);
    let rope_swa =
        peregrine_core::blocks::rope::build_rope_table(cfg.rope_dim, rope_base_swa, n.max(1), None);
    let is_swa: Vec<bool> = if window > 0 {
        match g.u32(&format!("{arch}.attention.sliding_window_pattern")) {
            Ok(p) if p > 0 => (0..cfg.n_layer)
                .map(|il| (il % p as usize) < (p as usize - 1))
                .collect(),
            _ => (0..cfg.n_layer).map(|il| il % 6 < 5).collect(),
        }
    } else {
        vec![false; cfg.n_layer]
    };

    let mut x = vec![0.0f32; n * dm];
    // The embedding lookup scaled by sqrt(d_model).
    let emb_scale = (dm as f32).sqrt();
    for (r, &tok) in tokens.iter().enumerate() {
        let row = tok_embd.row(tok as usize);
        for d in 0..dm {
            x[r * dm + d] =
                f32::from_le_bytes(row[d * 4..d * 4 + 4].try_into().unwrap()) * emb_scale;
        }
    }

    for (il, &swa) in is_swa.iter().enumerate().take(cfg.n_layer) {
        let p = |s: &str| format!("blk.{il}.{s}");
        let attn_norm = vec_f32(g, &p("attn_norm.weight"));
        let wq = mat(g, &p("attn_q.weight"));
        let wk = mat(g, &p("attn_k.weight"));
        let wv = mat(g, &p("attn_v.weight"));
        let q_norm = vec_f32(g, &p("attn_q_norm.weight"));
        let k_norm = vec_f32(g, &p("attn_k_norm.weight"));
        let wo = mat(g, &p("attn_output.weight"));
        let attn_post_norm = vec_f32(g, &p("post_attention_norm.weight"));
        let ffn_norm = vec_f32(g, &p("ffn_norm.weight"));
        let gate = mat(g, &p("ffn_gate.weight"));
        let up = mat(g, &p("ffn_up.weight"));
        let down = mat(g, &p("ffn_down.weight"));
        let ffn_post_norm = vec_f32(g, &p("post_ffw_norm.weight"));

        let table: &[f32] = if swa { &rope_swa } else { &rope_global };
        let half = if swa { Some(window / 2) } else { None };

        let mut xnorm = vec![0.0f32; n * dm];
        for r in 0..n {
            let row = rmsnorm(&x[r * dm..(r + 1) * dm], &attn_norm, eps);
            xnorm[r * dm..(r + 1) * dm].copy_from_slice(&row);
        }
        let mut q = vec![0.0f32; n * qd];
        let mut k = vec![0.0f32; n * kvd];
        let mut v = vec![0.0f32; n * kvd];
        for r in 0..n {
            wq.apply(&xnorm[r * dm..(r + 1) * dm], &mut q[r * qd..(r + 1) * qd]);
            wk.apply(&xnorm[r * dm..(r + 1) * dm], &mut k[r * kvd..(r + 1) * kvd]);
            wv.apply(&xnorm[r * dm..(r + 1) * dm], &mut v[r * kvd..(r + 1) * kvd]);
        }
        for r in 0..n {
            for h in 0..cfg.n_head {
                head_rmsnorm(&mut q[r * qd + h * hd..r * qd + (h + 1) * hd], &q_norm, eps);
            }
            for h in 0..cfg.n_kv_head {
                head_rmsnorm(
                    &mut k[r * kvd + h * hd..r * kvd + (h + 1) * hd],
                    &k_norm,
                    eps,
                );
            }
            peregrine_core::blocks::rope::rope(
                table,
                &mut q[r * qd..(r + 1) * qd],
                cfg.n_head,
                r,
                hd,
                cfg.rope_dim,
                cfg.rope_neox,
            );
            peregrine_core::blocks::rope::rope(
                table,
                &mut k[r * kvd..(r + 1) * kvd],
                cfg.n_kv_head,
                r,
                hd,
                cfg.rope_dim,
                cfg.rope_neox,
            );
            for qv in &mut q[r * qd..(r + 1) * qd] {
                *qv *= scale;
            }
        }
        let gqa = cfg.gqa();
        let mut attn = vec![0.0f32; n * qd];
        for r in 0..n {
            for h in 0..cfg.n_head {
                let kh = h / gqa;
                // The arithmetic the kernels define (S4.9g,
                // `peregrine_kernels::bidir`): a score is one sequential FMA chain over
                // the head dimension, the weights are the engine's own `exp` (the
                // scalar twin of the vector one, bit-identical), the denominator is an
                // f64 sum in ascending key order, and the value mix accumulates in
                // ascending key order. Only the keys inside the row's window are
                // computed — the band, not a mask.
                let (lo, hi) = peregrine_kernels::bidir::band(r, n, half);
                let mut scores = vec![0.0f32; n];
                for (j, s) in scores.iter_mut().enumerate().take(hi).skip(lo) {
                    let mut acc = 0.0f32;
                    for d in 0..hd {
                        acc = q[r * qd + h * hd + d].mul_add(k[j * kvd + kh * hd + d], acc);
                    }
                    *s = acc;
                }
                let max = scores[lo..hi]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut l = 0.0f64;
                for s in scores[lo..hi].iter_mut() {
                    *s = peregrine_kernels::vmath::exp_scalar(*s - max);
                }
                for &e in &scores[lo..hi] {
                    l += e as f64;
                }
                let inv = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
                let mut acc = vec![0.0f32; hd];
                for (j, &e) in scores[lo..hi].iter().enumerate() {
                    let j = lo + j;
                    for d in 0..hd {
                        acc[d] = e.mul_add(v[j * kvd + kh * hd + d], acc[d]);
                    }
                }
                for d in 0..hd {
                    attn[r * qd + h * hd + d] = acc[d] * inv;
                }
            }
        }
        let mut o = vec![0.0f32; n * dm];
        for r in 0..n {
            wo.apply(&attn[r * qd..(r + 1) * qd], &mut o[r * dm..(r + 1) * dm]);
        }
        for r in 0..n {
            let normed = rmsnorm(&o[r * dm..(r + 1) * dm], &attn_post_norm, eps);
            for d in 0..dm {
                x[r * dm + d] += normed[d];
            }
        }

        let mut xnorm = vec![0.0f32; n * dm];
        for r in 0..n {
            let row = rmsnorm(&x[r * dm..(r + 1) * dm], &ffn_norm, eps);
            xnorm[r * dm..(r + 1) * dm].copy_from_slice(&row);
        }
        let mut hb = vec![0.0f32; n * cfg.n_ff];
        let mut hb2 = vec![0.0f32; n * cfg.n_ff];
        for r in 0..n {
            gate.apply(
                &xnorm[r * dm..(r + 1) * dm],
                &mut hb[r * cfg.n_ff..(r + 1) * cfg.n_ff],
            );
            up.apply(
                &xnorm[r * dm..(r + 1) * dm],
                &mut hb2[r * cfg.n_ff..(r + 1) * cfg.n_ff],
            );
        }
        for i in 0..n * cfg.n_ff {
            hb[i] = peregrine_kernels::vmath::gelu_tanh(hb[i]) * hb2[i];
        }
        let mut o = vec![0.0f32; n * dm];
        for r in 0..n {
            down.apply(
                &hb[r * cfg.n_ff..(r + 1) * cfg.n_ff],
                &mut o[r * dm..(r + 1) * dm],
            );
        }
        for r in 0..n {
            let normed = rmsnorm(&o[r * dm..(r + 1) * dm], &ffn_post_norm, eps);
            for d in 0..dm {
                x[r * dm + d] += normed[d];
            }
        }
    }

    let mut hidden = vec![0.0f32; n * dm];
    for r in 0..n {
        let row = rmsnorm(&x[r * dm..(r + 1) * dm], &out_norm, eps);
        hidden[r * dm..(r + 1) * dm].copy_from_slice(&row);
    }
    hidden
}

/// The pooled vector, then EmbeddingGemma's `dense_2` and `dense_3` — the whole answer
/// before truncation and normalisation.
pub fn reference_embedding(model: &Model, tokens: &[u32], pooling: Pooling) -> Vec<f32> {
    let g = model.gguf();
    let dm = model.cfg.d_model;
    let hidden = hidden_rows(model, tokens);
    let mut sum = vec![0.0f64; dm];
    let mut v = vec![0.0f32; dm];
    peregrine_core::embed::pool_rows_into(&hidden, dm, pooling, &mut sum, &mut v);

    if g.tensor("dense_2.weight").is_ok() {
        let d2 = mat(g, "dense_2.weight");
        let d3 = mat(g, "dense_3.weight");
        let mut mid = vec![0.0f32; d2.d_out];
        d2.apply(&v, &mut mid);
        let mut out = vec![0.0f32; d3.d_out];
        d3.apply(&mid, &mut out);
        return out;
    }
    v
}

/// `mistral3`'s final normed hidden states of one text (llama.cpp's `result_norm`), in
/// order — the reference for NVIDIA Nemotron 3.5 Embed 1B's graph (roadmap S4.14).
///
/// The graph is llama.cpp v0.9.0's `src/models/mistral3.cpp:139-269`, with the mask
/// removed (`attention.causal = false`): every row sees every row of its own text. It is
/// reimplemented here layer by layer with plain loops, over the same arithmetic the
/// engine's blocks define, as [`hidden_rows`] is for `gemma-embedding`:
///
/// - the embeddings are read unscaled;
/// - `attn_norm` → q/k/v → RoPE (YaRN) → the per-row attention temperature
///   (`ln(floor(pos / n_ctx_orig) + 1) · scale + 1`, `src/llama-graph.cpp:304-312`) →
///   attention with `kq_scale = 1/sqrt(head_dim)` → `attn_output` → residual;
/// - `ffn_norm` → gate/up → SwiGLU → down → residual;
/// - `output_norm` over every row.
///
/// The YaRN table is built by the same `build_rope_table_yarn` the engine's state uses;
/// the table's own arithmetic is pinned by `blocks::rope`'s tests, and the metadata that
/// feeds it is read here independently.
pub fn hidden_rows_mistral3(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let g = model.gguf();
    let cfg = &model.cfg;
    let dm = cfg.d_model;
    let hd = cfg.head_dim;
    let qd = cfg.q_dim();
    let kvd = cfg.kv_dim();
    let n = tokens.len();
    let arch = "mistral3";

    let tok_embd = mat(g, "token_embd.weight");
    let out_norm = vec_f32(g, "output_norm.weight");
    let eps = cfg.eps;
    let scale = 1.0 / (hd as f32).sqrt();

    let base = g.f32(&format!("{arch}.rope.freq_base")).unwrap_or(10_000.0);
    let factor = g
        .f32(&format!("{arch}.rope.scaling.factor"))
        .expect("mistral3 YaRN factor");
    let n_ctx_orig = g
        .u32(&format!("{arch}.rope.scaling.original_context_length"))
        .unwrap_or(4096) as usize;
    let beta_fast = g
        .f32(&format!("{arch}.rope.scaling.yarn_beta_fast"))
        .unwrap_or(32.0);
    let beta_slow = g
        .f32(&format!("{arch}.rope.scaling.yarn_beta_slow"))
        .unwrap_or(1.0);
    let log_mul = g
        .f32(&format!("{arch}.rope.scaling.yarn_log_multiplier"))
        .unwrap_or(0.0);
    let yarn = peregrine_core::blocks::rope::yarn_params(
        cfg.rope_dim,
        base,
        factor,
        n_ctx_orig,
        1.0,
        1.0,
        log_mul,
        beta_fast,
        beta_slow,
    );
    let table =
        peregrine_core::blocks::rope::build_rope_table_yarn(cfg.rope_dim, base, n.max(1), None, &yarn);
    let temp_scale = g
        .f32(&format!("{arch}.attention.temperature_scale"))
        .unwrap_or(0.0);

    let mut x = vec![0.0f32; n * dm];
    for (r, &tok) in tokens.iter().enumerate() {
        let row = tok_embd.row(tok as usize);
        for d in 0..dm {
            x[r * dm + d] = f32::from_le_bytes(row[d * 4..d * 4 + 4].try_into().unwrap());
        }
    }

    for il in 0..cfg.n_layer {
        let p = |s: &str| format!("blk.{il}.{s}");
        let attn_norm = vec_f32(g, &p("attn_norm.weight"));
        let wq = mat(g, &p("attn_q.weight"));
        let wk = mat(g, &p("attn_k.weight"));
        let wv = mat(g, &p("attn_v.weight"));
        let wo = mat(g, &p("attn_output.weight"));
        let ffn_norm = vec_f32(g, &p("ffn_norm.weight"));
        let gate = mat(g, &p("ffn_gate.weight"));
        let up = mat(g, &p("ffn_up.weight"));
        let down = mat(g, &p("ffn_down.weight"));

        let mut xnorm = vec![0.0f32; n * dm];
        for r in 0..n {
            let row = rmsnorm(&x[r * dm..(r + 1) * dm], &attn_norm, eps);
            xnorm[r * dm..(r + 1) * dm].copy_from_slice(&row);
        }
        let mut q = vec![0.0f32; n * qd];
        let mut k = vec![0.0f32; n * kvd];
        let mut v = vec![0.0f32; n * kvd];
        for r in 0..n {
            wq.apply(&xnorm[r * dm..(r + 1) * dm], &mut q[r * qd..(r + 1) * qd]);
            wk.apply(&xnorm[r * dm..(r + 1) * dm], &mut k[r * kvd..(r + 1) * kvd]);
            wv.apply(&xnorm[r * dm..(r + 1) * dm], &mut v[r * kvd..(r + 1) * kvd]);
        }
        for r in 0..n {
            // NORM rotary pairing (llama.cpp's `llama_model_rope_type` for `mistral3`).
            peregrine_core::blocks::rope::rope(
                &table,
                &mut q[r * qd..(r + 1) * qd],
                cfg.n_head,
                r,
                hd,
                cfg.rope_dim,
                false,
            );
            peregrine_core::blocks::rope::rope(
                &table,
                &mut k[r * kvd..(r + 1) * kvd],
                cfg.n_kv_head,
                r,
                hd,
                cfg.rope_dim,
                false,
            );
            // `llm_graph_input_attn_temp::set_input`: Q is multiplied per row, after RoPE.
            if temp_scale != 0.0 {
                let t = peregrine_core::vmath::det_ln(((r / n_ctx_orig) as f32) + 1.0)
                    * temp_scale
                    + 1.0;
                for x in &mut q[r * qd..(r + 1) * qd] {
                    *x *= t;
                }
            }
        }
        let gqa = cfg.gqa();
        let mut attn = vec![0.0f32; n * qd];
        for r in 0..n {
            for h in 0..cfg.n_head {
                let kh = h / gqa;
                let (lo, hi) = peregrine_kernels::bidir::band(r, n, None);
                let mut scores = vec![0.0f32; n];
                for (j, s) in scores.iter_mut().enumerate().take(hi).skip(lo) {
                    let mut acc = 0.0f32;
                    for d in 0..hd {
                        acc = q[r * qd + h * hd + d].mul_add(k[j * kvd + kh * hd + d], acc);
                    }
                    *s = acc * scale;
                }
                let max = scores[lo..hi]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut l = 0.0f64;
                for s in scores[lo..hi].iter_mut() {
                    *s = peregrine_kernels::vmath::exp_scalar(*s - max);
                }
                for &e in &scores[lo..hi] {
                    l += e as f64;
                }
                let inv = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
                let mut acc = vec![0.0f32; hd];
                for (j, &e) in scores[lo..hi].iter().enumerate() {
                    let j = lo + j;
                    for d in 0..hd {
                        acc[d] = e.mul_add(v[j * kvd + kh * hd + d], acc[d]);
                    }
                }
                for d in 0..hd {
                    attn[r * qd + h * hd + d] = acc[d] * inv;
                }
            }
        }
        let mut o = vec![0.0f32; n * dm];
        for r in 0..n {
            wo.apply(&attn[r * qd..(r + 1) * qd], &mut o[r * dm..(r + 1) * dm]);
        }
        for i in 0..n * dm {
            x[i] += o[i];
        }

        let mut xnorm = vec![0.0f32; n * dm];
        for r in 0..n {
            let row = rmsnorm(&x[r * dm..(r + 1) * dm], &ffn_norm, eps);
            xnorm[r * dm..(r + 1) * dm].copy_from_slice(&row);
        }
        let f = cfg.n_ff;
        let mut hb = vec![0.0f32; n * f];
        let mut hb2 = vec![0.0f32; n * f];
        for r in 0..n {
            gate.apply(&xnorm[r * dm..(r + 1) * dm], &mut hb[r * f..(r + 1) * f]);
            up.apply(&xnorm[r * dm..(r + 1) * dm], &mut hb2[r * f..(r + 1) * f]);
        }
        for r in 0..n {
            peregrine_kernels::vmath::swiglu(&mut hb[r * f..(r + 1) * f], &hb2[r * f..(r + 1) * f]);
        }
        let mut o = vec![0.0f32; n * dm];
        for r in 0..n {
            down.apply(&hb[r * f..(r + 1) * f], &mut o[r * dm..(r + 1) * dm]);
        }
        for i in 0..n * dm {
            x[i] += o[i];
        }
    }

    let mut hidden = vec![0.0f32; n * dm];
    for r in 0..n {
        let row = rmsnorm(&x[r * dm..(r + 1) * dm], &out_norm, eps);
        hidden[r * dm..(r + 1) * dm].copy_from_slice(&row);
    }
    hidden
}

/// The pooled embedding of one text through the `mistral3` reference graph.
pub fn reference_embedding_mistral3(model: &Model, tokens: &[u32], pooling: Pooling) -> Vec<f32> {
    let dm = model.cfg.d_model;
    let hidden = hidden_rows_mistral3(model, tokens);
    let mut sum = vec![0.0f64; dm];
    let mut v = vec![0.0f32; dm];
    peregrine_core::embed::pool_rows_into(&hidden, dm, pooling, &mut sum, &mut v);
    v
}
