//! The `gpt-oss` loader: the metadata, the per-expert views, and the refusals (S2.5).
//!
//! A `gpt-oss` file is what OpenAI's gpt-oss models are converted to. llama.cpp
//! v0.9.0 reads it in `src/models/openai-moe.cpp` and `src/llama-arch.cpp`:
//!
//! | key | meaning |
//! |---|---|
//! | `gpt-oss.expert_count` | how many experts the router chooses from (32) |
//! | `gpt-oss.expert_used_count` | how many of them one token uses (4) |
//! | `gpt-oss.expert_feed_forward_length` | the width of one expert's feed-forward (2880) |
//! | `gpt-oss.expert_weights_scale` | a scale on the selected experts' weights (0: none) |
//! | `gpt-oss.attention.sliding_window` | the sliding window (128) |
//! | `gpt-oss.rope.scaling.type` | `"yarn"` |
//! | `gpt-oss.rope.scaling.factor` | the YaRN factor (32) |
//! | `gpt-oss.rope.scaling.original_context_length` | the context YaRN was trained for (4096) |
//!
//! The tensors of a layer (`src/models/openai-moe.cpp:33-55`):
//!
//! | tensor | dims | what |
//! |---|---|---|
//! | `blk.i.attn_norm.weight` | `[d_model]` | the attention's norm |
//! | `blk.i.attn_q.weight` | `[d_model, n_head·head_dim]` | the queries |
//! | `blk.i.attn_k.weight` | `[d_model, n_kv_head·head_dim]` | the keys |
//! | `blk.i.attn_v.weight` | `[d_model, n_kv_head·head_dim]` | the values |
//! | `blk.i.attn_output.weight` | `[n_head·head_dim, d_model]` | the attention's output |
//! | `blk.i.attn_output.bias` | `[d_model]` | its bias |
//! | `blk.i.attn_sinks.weight` | `[n_head]` | the attention sinks |
//! | `blk.i.post_attention_norm.weight` | `[d_model]` | the norm the experts see |
//! | `blk.i.ffn_gate_inp.weight` | `[d_model, n_expert]` | the router |
//! | `blk.i.ffn_gate_inp.bias` | `[n_expert]` | the router's bias |
//! | `blk.i.ffn_gate_exps.weight` | `[d_model, n_ff_exp, n_expert]` | the experts' gates |
//! | `blk.i.ffn_gate_exps.bias` | `[n_ff_exp, n_expert]` | their biases |
//! | `blk.i.ffn_up_exps.weight` | `[d_model, n_ff_exp, n_expert]` | the experts' ups |
//! | `blk.i.ffn_up_exps.bias` | `[n_ff_exp, n_expert]` | their biases |
//! | `blk.i.ffn_down_exps.weight` | `[n_ff_exp, d_model, n_expert]` | the experts' downs |
//! | `blk.i.ffn_down_exps.bias` | `[d_model, n_expert]` | their biases |
//!
//! The gate and the up are **separate** tensors, not one fused one: upstream's
//! converter splits the checkpoint's fused `gate_up_proj` into the two
//! (`conversion/gpt_oss.py:106-121`).

use crate::gptoss::GptOssLayer;
use peregrine_core::arch::Loaded;
use peregrine_core::blocks::moe::ExpertBias;
use peregrine_core::blocks::rope::yarn_params;
use peregrine_core::gguf::{Gguf, TensorInfo, Value};
use peregrine_core::model::{Config, Linear, TileStore};
use peregrine_core::tokenizer::Tokenizer;
use peregrine_kernels::quant::dequant_row;

/// The geometry of a `gpt-oss` model.
#[derive(Debug, Clone)]
pub struct GptOssCfg {
    /// How many experts the router chooses from (`expert_count`).
    pub n_expert: usize,
    /// How many of them one row uses (`expert_used_count`).
    pub n_expert_used: usize,
    /// The width of one expert's feed-forward (`expert_feed_forward_length`).
    pub n_ff_exp: usize,
    /// The scale on the selected experts' weights (`expert_weights_scale`), with
    /// llama.cpp's meaning: a scale of 0 or 1 is not applied
    /// (`src/llama-graph.cpp:2390`).
    pub w_scale: f32,
    /// The sliding window (`attention.sliding_window`): the layers that use it see
    /// at most the last `sliding_window` positions.
    pub sliding_window: usize,
    /// Which layers use the sliding window: every other one, as llama.cpp's
    /// `load_swa_pattern(ml, 2)` sets it (`src/models/openai-moe.cpp:9`).
    pub is_swa: Vec<bool>,
    /// The alpha of the experts' `swiglu_oai` (`src/llama-graph.cpp:2522`).
    pub swiglu_alpha: f32,
    /// The limit of the experts' `swiglu_oai` (`src/llama-graph.cpp:2523`).
    pub swiglu_limit: f32,
}

/// The alpha and the limit of OpenAI's `swiglu_oai`, as llama.cpp v0.9.0's
/// `LLM_FFN_SWIGLU_OAI_MOE` case sets them (`src/llama-graph.cpp:2519-2526`).
const SWIGLU_OAI_ALPHA: f32 = 1.702;
const SWIGLU_OAI_LIMIT: f32 = 7.0;

/// llama.cpp's default YaRN parameters (`src/llama-hparams.h:175-178`).
const YARN_BETA_FAST: f32 = 32.0;
const YARN_BETA_SLOW: f32 = 1.0;

/// Read a bias tensor's rows: `n` rows of `width` values each, the row index
/// running slowest (GGUF stores `dims[0]` contiguous).
fn bias_rows(g: &Gguf, t: &TensorInfo, n: usize, width: usize) -> Result<Vec<Vec<f32>>, String> {
    if t.dims.len() != 2 {
        return Err(format!(
            "{}: a bias has two dimensions, not {}",
            t.name,
            t.dims.len()
        ));
    }
    if t.dims[0] as usize != width || t.dims[1] as usize != n {
        return Err(format!(
            "{}: the file holds {:?}, expected [{width}, {n}]",
            t.name, t.dims
        ));
    }
    let block = t.kind.block();
    if !width.is_multiple_of(block) {
        return Err(format!(
            "{}: a row of {width} values is not a whole number of {} blocks",
            t.name,
            t.kind.label()
        ));
    }
    let rb = peregrine_kernels::quant::row_bytes(t.kind, width);
    let bytes = g.bytes(t);
    let mut out = Vec::with_capacity(n);
    let mut row = vec![0.0f32; width];
    for e in 0..n {
        dequant_row(t.kind, &bytes[e * rb..(e + 1) * rb], &mut row);
        out.push(row.clone());
    }
    Ok(out)
}

/// Read a flat bias: one row of `n` values.
fn bias_flat(g: &Gguf, t: &TensorInfo, n: usize) -> Result<Vec<f32>, String> {
    if t.dims.len() != 1 || t.dims[0] as usize != n {
        return Err(format!(
            "{}: the file holds {:?}, expected [{n}]",
            t.name, t.dims
        ));
    }
    let mut out = vec![0.0f32; n];
    dequant_row(t.kind, g.bytes(t), &mut out);
    Ok(out)
}

pub(crate) fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
    let arch_name = g.str("general.architecture")?;
    if arch_name != "gpt-oss" {
        return Err(format!(
            "the gpt-oss family loads gpt-oss files, not {arch_name:?}"
        ));
    }
    let k = |s: &str| format!("gpt-oss.{s}");
    let n_layer = g.u32(&k("block_count"))? as usize;
    let d_model = g.u32(&k("embedding_length"))? as usize;
    let n_head = g.u32(&k("attention.head_count"))? as usize;
    let n_kv_head = g
        .u32(&k("attention.head_count_kv"))
        .unwrap_or(n_head as u32) as usize;
    let head_dim = g
        .u32(&k("attention.key_length"))
        .map(|v| v as usize)
        .unwrap_or(d_model / n_head);
    let rope_dim = g.u32(&k("rope.dimension_count")).unwrap_or(head_dim as u32) as usize;

    // The experts' geometry.
    let n_expert = g.u32(&k("expert_count"))? as usize;
    let n_expert_used = g.u32(&k("expert_used_count"))? as usize;
    let n_ff_exp = g.u32(&k("expert_feed_forward_length")).map_err(|_| {
        "gpt-oss: the file carries no expert_feed_forward_length; the experts' width cannot \
         be guessed"
            .to_string()
    })? as usize;
    let w_scale = g.f32(&k("expert_weights_scale")).unwrap_or(0.0);
    if w_scale != 0.0 && w_scale != 1.0 {
        return Err(format!(
            "gpt-oss.expert_weights_scale = {w_scale}: only 0 or 1 (no scaling) is supported"
        ));
    }

    // The sliding window, and the layers that use it. llama.cpp reads the window
    // (`src/models/openai-moe.cpp:6`) and the pattern with `load_swa_pattern(ml, 2)`
    // (`src/models/openai-moe.cpp:9`): every other layer is a sliding-window one
    // (`set_swa_pattern`, `src/llama-hparams.cpp:8-22`).
    let sliding_window = g.u32(&k("attention.sliding_window")).map_err(|_| {
        "gpt-oss: the file carries no attention.sliding_window; which layers use one cannot \
         be guessed"
            .to_string()
    })? as usize;
    if sliding_window == 0 {
        return Err("gpt-oss: attention.sliding_window is zero".into());
    }
    let is_swa = swa_pattern(g, &k("attention.sliding_window_pattern"), n_layer)?;

    if n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
        return Err(format!(
            "implausible head configuration: {n_head} query heads over {n_kv_head} kv heads"
        ));
    }
    if n_expert == 0 {
        return Err("gpt-oss: expert_count is zero".into());
    }
    if n_expert_used == 0 || n_expert_used > n_expert {
        return Err(format!(
            "gpt-oss: a row uses {n_expert_used} of {n_expert} experts"
        ));
    }
    // llama.cpp's queries and keys are `n_head · n_rot` wide
    // (`src/models/openai-moe.cpp:39`), and the engine's rotary span is the head's
    // width. A file whose rotary span is shorter would need a second head width.
    if rope_dim != head_dim {
        return Err(format!(
            "gpt-oss: the rotary span ({rope_dim}) is not the head width ({head_dim}); a partial \
             rotary attention is not implemented"
        ));
    }
    if !rope_dim.is_multiple_of(2) {
        return Err(format!("implausible rope dimension {rope_dim}"));
    }
    // The activations are quantised 32 values at a time, so every width the engine
    // carries has to be a whole number of those blocks.
    for (what, n) in [
        ("embedding length", d_model),
        ("expert feed-forward length", n_ff_exp),
    ] {
        if !n.is_multiple_of(32) {
            return Err(format!("{what} {n} is not a multiple of 32"));
        }
    }
    if !(head_dim * n_head).is_multiple_of(32) {
        return Err(format!(
            "query width {} is not a multiple of 32",
            head_dim * n_head
        ));
    }

    // The rotary frequencies. llama.cpp gives the sliding-window layers their own
    // base (`rope.freq_base_swa`), which defaults to the model's own
    // (`src/models/openai-moe.cpp:11-13`).
    let rope_base = g.f32(&k("rope.freq_base")).unwrap_or(150_000.0);
    if let Ok(swa_base) = g.f32(&k("rope.freq_base_swa")) {
        if swa_base != rope_base {
            return Err(format!(
                "gpt-oss.rope.freq_base_swa = {swa_base}: a different rotary base for the \
                 sliding-window layers is not implemented"
            ));
        }
    }
    // YaRN. The file says which scaling it wants; gpt-oss's is YaRN
    // (`src/models/openai-moe.cpp:11`, the converter writes `rope_type: "yarn"`).
    let rope_yarn = match g.str(&k("rope.scaling.type")) {
        Ok("none") => None,
        Ok("yarn") => {
            let factor = g.f32(&k("rope.scaling.factor")).map_err(|_| {
                "gpt-oss: the file asks for YaRN rope scaling but carries no \
                 rope.scaling.factor"
                    .to_string()
            })?;
            if factor <= 0.0 || factor.is_nan() {
                return Err(format!("gpt-oss.rope.scaling.factor = {factor}"));
            }
            let n_ctx_orig = g
                .u32(&k("rope.scaling.original_context_length"))
                .unwrap_or(4096) as usize;
            let ext_factor = g
                .f32(&k("rope.scaling.yarn_ext_factor"))
                .map(|v| if v < 0.0 { 1.0 } else { v })
                .unwrap_or(1.0);
            let attn_factor = g.f32(&k("rope.scaling.attn_factor")).unwrap_or(1.0);
            let log_mul = g.f32(&k("rope.scaling.yarn_log_multiplier")).unwrap_or(0.0);
            let beta_fast = g
                .f32(&k("rope.scaling.yarn_beta_fast"))
                .unwrap_or(YARN_BETA_FAST);
            let beta_slow = g
                .f32(&k("rope.scaling.yarn_beta_slow"))
                .unwrap_or(YARN_BETA_SLOW);
            Some(yarn_params(
                rope_dim,
                rope_base,
                factor,
                n_ctx_orig,
                ext_factor,
                attn_factor,
                log_mul,
                beta_fast,
                beta_slow,
            ))
        }
        Ok(other) => {
            return Err(format!(
                "gpt-oss: rope scaling {other:?} is not implemented"
            ));
        }
        Err(_) => None,
    };

    let cfg = Config {
        arch_name: "gpt-oss",
        // gpt-oss pairs the rotary frequencies NeoX-style: llama.cpp's
        // `llama_model_rope_type` returns `LLAMA_ROPE_TYPE_NEOX` for
        // `LLM_ARCH_OPENAI_MOE` (`src/llama-model.cpp:3740`).
        rope_neox: true,
        n_layer,
        n_kv_layer: n_layer,
        n_head,
        n_kv_head,
        head_dim,
        d_model,
        // The experts' feed-forward is the model's feed-forward.
        n_ff: n_ff_exp,
        n_vocab: g.u32(&k("vocab_size")).unwrap_or(0) as usize,
        rope_dim,
        eps: g
            .f32(&k("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5),
        rope_base,
        ctx_train: g.u32(&k("context_length")).unwrap_or(2048) as usize,
        rope_yarn,
    };

    let get = |n: &str| -> Result<TensorInfo, String> { g.tensor(n).cloned() };
    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let p = |t: &str| format!("blk.{i}.{t}");
        let gate_t = get(&p("ffn_gate_exps.weight"))?;
        let up_t = get(&p("ffn_up_exps.weight"))?;
        let down_t = get(&p("ffn_down_exps.weight"))?;
        // The experts are read as 2D matrices: one expert is a contiguous run of rows,
        // because GGUF stores `dims[0]` contiguous and the expert index runs slowest.
        let bank = |t: &TensorInfo| -> Result<Vec<TensorInfo>, String> {
            (0..n_expert)
                .map(|e| peregrine_core::blocks::moe::expert_view(t, e))
                .collect::<Result<Vec<_>, _>>()
        };
        for t in [&gate_t, &up_t, &down_t] {
            if t.dims.len() != 3 || t.dims[2] as usize != n_expert {
                return Err(format!(
                    "{}: the file holds {:?}, expected {n_expert} experts",
                    t.name, t.dims
                ));
            }
        }
        // A row of an expert's matrix is multiplied by an activation along it, so it
        // has to be a whole number of the format's blocks.
        for (t, row) in [(&gate_t, d_model), (&up_t, d_model), (&down_t, n_ff_exp)] {
            let block = t.kind.block();
            if !row.is_multiple_of(block) {
                return Err(format!(
                    "{}: a row of {row} weights is not a whole number of {} blocks",
                    t.name,
                    t.kind.label()
                ));
            }
        }
        // The biases. gpt-oss's experts carry one on each of their three
        // projections, and the router carries one of its own
        // (`src/models/openai-moe.cpp:49-54`); a file without them is refused, since
        // the feed-forward would then be a different function.
        let gate_b = bias_rows(g, &get(&p("ffn_gate_exps.bias"))?, n_expert, n_ff_exp)?;
        let up_b = bias_rows(g, &get(&p("ffn_up_exps.bias"))?, n_expert, n_ff_exp)?;
        let down_b = bias_rows(g, &get(&p("ffn_down_exps.bias"))?, n_expert, d_model)?;
        let expert_bias = Some(
            gate_b
                .into_iter()
                .zip(up_b)
                .zip(down_b)
                .map(|((gate, up), down)| ExpertBias { gate, up, down })
                .collect::<Vec<_>>(),
        );
        // The attention sinks: one learned logit per query head
        // (`src/models/openai-moe.cpp:42`). llama.cpp treats them as optional; a
        // file without them has no sinks.
        let sinks = match get(&p("attn_sinks.weight")) {
            Ok(t) => {
                if t.dims.len() != 1 || t.dims[0] as usize != n_head {
                    return Err(format!(
                        "{}: the file holds {:?}, expected [{n_head}]",
                        t.name, t.dims
                    ));
                }
                let mut v = vec![0.0f32; n_head];
                dequant_row(t.kind, g.bytes(&t), &mut v);
                Some(v)
            }
            Err(_) => None,
        };
        layers.push(GptOssLayer {
            attn_norm: get(&p("attn_norm.weight"))?,
            wq: Linear::Dense(get(&p("attn_q.weight"))?),
            wk: Linear::Dense(get(&p("attn_k.weight"))?),
            wv: Linear::Dense(get(&p("attn_v.weight"))?),
            wo: Linear::Dense(get(&p("attn_output.weight"))?),
            wo_bias: Some(bias_flat(g, &get(&p("attn_output.bias"))?, d_model)?),
            sinks,
            post_norm: get(&p("post_attention_norm.weight"))?,
            router: get(&p("ffn_gate_inp.weight"))?,
            router_bias: Some(bias_flat(g, &get(&p("ffn_gate_inp.bias"))?, n_expert)?),
            gate: bank(&gate_t)?,
            up: bank(&up_t)?,
            down: bank(&down_t)?,
            expert_bias,
        });
    }
    let tok_embd = get("token_embd.weight")?;
    let out_norm = get("output_norm.weight")?;
    let head = Linear::Dense(
        g.tensor("output.weight")
            .ok()
            .cloned()
            .unwrap_or_else(|| tok_embd.clone()),
    );

    let mut cfg = cfg;
    if cfg.n_vocab == 0 {
        cfg.n_vocab = tk.n_vocab;
    }
    if cfg.n_vocab != tk.n_vocab {
        return Err(format!(
            "metadata says {} vocabulary entries but the tokenizer has {}",
            cfg.n_vocab, tk.n_vocab
        ));
    }

    // The shapes the metadata implies.
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
    let (dm, qd, kd) = (cfg.d_model, cfg.q_dim(), cfg.kv_dim());
    for l in &layers {
        for (lin, d_in, d_out) in [
            (&l.wq, dm, qd),
            (&l.wk, dm, kd),
            (&l.wv, dm, kd),
            (&l.wo, qd, dm),
        ] {
            if let Some(t) = lin.dense() {
                shape(t, d_in, d_out)?;
            }
        }
        shape(&l.attn_norm, dm, 1)?;
        shape(&l.post_norm, dm, 1)?;
        shape(&l.router, dm, n_expert)?;
        // The three expert banks: `[d_model, n_ff_exp, n_expert]` and
        // `[n_ff_exp, d_model, n_expert]`.
        for (bank, d_in, d_out) in [
            (&l.gate, dm, n_ff_exp),
            (&l.up, dm, n_ff_exp),
            (&l.down, n_ff_exp, dm),
        ] {
            for e in bank {
                shape(e, d_in, d_out)?;
            }
        }
    }
    shape(&tok_embd, dm, cfg.n_vocab)?;
    shape(&out_norm, dm, 1)?;
    if let Some(t) = head.dense() {
        shape(t, dm, cfg.n_vocab)?;
    }

    let gpt_oss = crate::gptoss::GptOss {
        cfg: GptOssCfg {
            n_expert,
            n_expert_used,
            n_ff_exp,
            w_scale,
            sliding_window,
            is_swa,
            swiglu_alpha: SWIGLU_OAI_ALPHA,
            swiglu_limit: SWIGLU_OAI_LIMIT,
        },
        layers,
        tok_embd,
        out_norm,
        head,
    };
    Ok(Loaded {
        cfg,
        arch: Box::new(gpt_oss),
        rope_factors: None,
        tiles: TileStore::default(),
    })
}

/// Which layers use the sliding window.
///
/// llama.cpp's `load_swa_pattern` (`src/llama-model.cpp:4053-4061`) reads the
/// file's `attention.sliding_window_pattern` — an array of flags, or a scalar
/// period — and falls back to the period the architecture asks for (2 for gpt-oss,
/// `src/models/openai-moe.cpp:9`). The flags then follow
/// `set_swa_pattern` (`src/llama-hparams.cpp:8-22`): with a period of `p`, layer
/// `il` is a sliding-window one when `il % p < p − 1`.
fn swa_pattern(g: &Gguf, key: &str, n_layer: usize) -> Result<Vec<bool>, String> {
    match g.meta.get(key) {
        Some(Value::ArrI32(v)) => {
            if v.len() != n_layer {
                return Err(format!(
                    "{key}: the file holds {} flags, expected {n_layer}",
                    v.len()
                ));
            }
            Ok(v.iter().map(|&x| x != 0).collect())
        }
        Some(Value::ArrOther) => Err(format!(
            "{key}: the file holds an array this engine cannot read (a boolean one); \
             a gpt-oss file's sliding-window pattern is a period"
        )),
        Some(_) => {
            let p = g.u32(key)? as usize;
            Ok((0..n_layer)
                .map(|il| p == 0 || (il % p) < (p - 1))
                .collect())
        }
        None => Ok((0..n_layer).map(|il| il % 2 < 1).collect()),
    }
}
