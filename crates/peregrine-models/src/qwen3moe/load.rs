//! The `qwen3moe` loader: the metadata, the per-expert views, and the refusals (S2.5).
//!
//! A `qwen3moe` file is a Qwen3 model whose feed-forward is a bank of experts. llama.cpp
//! v0.9.0 reads it in `src/models/qwen3moe.cpp` and `src/llama-arch.cpp`:
//!
//! | key | meaning |
//! |---|---|
//! | `qwen3moe.expert_count` | how many experts the router chooses from (128 in Qwen3-30B-A3B) |
//! | `qwen3moe.expert_used_count` | how many of them one token uses (8) |
//! | `qwen3moe.expert_feed_forward_length` | the width of one expert's feed-forward (768) |
//! | `qwen3moe.expert_weights_scale` | a scale on the selected experts' weights (0: none) |
//!
//! The file also carries the *dense* `feed_forward_length` (6144), which the experts ignore:
//! llama.cpp reads it, and so does this loader, but the experts' own width is
//! `expert_feed_forward_length`. A file without that key is refused rather than guessed at.

use crate::dense::layer::AttnWeights;
use crate::qwen3moe::{MoeLayer, Qwen3Moe};
use peregrine_core::arch::Loaded;
use peregrine_core::blocks::moe::expert_view;
use peregrine_core::gguf::{Gguf, TensorInfo};
use peregrine_core::model::{Config, Linear, TileStore};
use peregrine_core::tokenizer::Tokenizer;
use peregrine_kernels::quant::dequant_row;

/// The mixture-of-experts geometry of a `qwen3moe` file.
#[derive(Debug, Clone, Copy)]
pub struct MoeCfg {
    /// How many experts the router chooses from (`expert_count`).
    pub n_expert: usize,
    /// How many of them one row uses (`expert_used_count`).
    pub n_expert_used: usize,
    /// The width of one expert's feed-forward (`expert_feed_forward_length`).
    pub n_ff_exp: usize,
    /// The scale on the selected experts' weights (`expert_weights_scale`), with llama.cpp's
    /// meaning: a scale of 0 or 1 is not applied (`src/llama-graph.cpp:2390`). Every
    /// qwen3moe file carries 0, which means "no scaling".
    pub w_scale: f32,
    /// Whether the selected experts' weights are renormalised over the selection. llama.cpp
    /// passes `true` for qwen3moe (`src/models/qwen3moe.cpp`), which is the HF card's
    /// `norm_topk_prob: true`.
    pub renorm: bool,
}

pub(crate) fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
    let arch_name = g.str("general.architecture")?;
    if arch_name != "qwen3moe" {
        return Err(format!(
            "the qwen3moe family loads qwen3moe files, not {arch_name:?}"
        ));
    }
    let k = |s: &str| format!("qwen3moe.{s}");
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

    // The experts' geometry. `expert_feed_forward_length` is the width the experts use; the
    // file's dense `feed_forward_length` is read but ignored, as in llama.cpp.
    let n_expert = g.u32(&k("expert_count"))? as usize;
    let n_expert_used = g.u32(&k("expert_used_count"))? as usize;
    let n_ff_exp = g.u32(&k("expert_feed_forward_length")).map_err(|_| {
        "qwen3moe: the file carries no expert_feed_forward_length; the experts' width cannot \
         be guessed (the dense feed_forward_length is not it)"
            .to_string()
    })? as usize;
    let w_scale = g.f32(&k("expert_weights_scale")).unwrap_or(0.0);
    // llama.cpp does not apply `expert_weights_scale` to qwen3moe, so a file with a real scale would
    // run differently here and there; every published qwen3moe file carries 0. Refuse instead of guessing.
    if w_scale != 0.0 && w_scale != 1.0 {
        return Err(format!(
            "qwen3moe.expert_weights_scale = {w_scale}: only 0 or 1 (no scaling) is supported"
        ));
    }

    if let Ok(t) = g.str(&k("rope.scaling.type")) {
        if t != "none" && t != "linear" {
            return Err(format!("rope scaling {t:?} is not implemented"));
        }
    }
    if n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
        return Err(format!(
            "implausible head configuration: {n_head} query heads over {n_kv_head} kv heads"
        ));
    }
    if n_expert == 0 {
        return Err("qwen3moe: expert_count is zero".into());
    }
    if n_expert_used == 0 || n_expert_used > n_expert {
        return Err(format!(
            "qwen3moe: a row uses {n_expert_used} of {n_expert} experts"
        ));
    }
    if rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
        return Err(format!(
            "implausible rope dimension {rope_dim} for head dim {head_dim}"
        ));
    }
    // The activations are quantised 32 values at a time, so every width the engine carries
    // has to be a whole number of those blocks.
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

    let rope_scale = g
        .f32(&k("rope.scaling.factor"))
        .ok()
        .filter(|f| *f > 0.0)
        .map(|f| 1.0 / f);
    let cfg = Config {
        arch_name: "qwen3moe",
        // Qwen3 pairs the rotary frequencies NeoX-style (`i` with `i + rope_dim/2`).
        rope_neox: true,
        n_layer,
        n_kv_layer: n_layer,
        n_head,
        n_kv_head,
        head_dim,
        d_model,
        // The experts' feed-forward is the model's feed-forward; the dense one the file also
        // carries is unused.
        n_ff: n_ff_exp,
        n_vocab: g.u32(&k("vocab_size")).unwrap_or(0) as usize,
        rope_dim,
        eps: g
            .f32(&k("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6),
        rope_base: g.f32(&k("rope.freq_base")).unwrap_or(10000.0),
        ctx_train: g.u32(&k("context_length")).unwrap_or(2048) as usize,
    };
    let mut rope_factors: Option<Vec<f32>> = rope_scale.map(|s| vec![1.0 / s; rope_dim / 2]);
    if let Ok(t) = g.tensor("rope_freqs.weight") {
        let n = t.numel() as usize;
        if n != rope_dim / 2 {
            return Err(format!(
                "rope_freqs.weight has {n} entries, expected {}",
                rope_dim / 2
            ));
        }
        let mut f = vec![0.0f32; n];
        dequant_row(t.kind, g.bytes(t), &mut f);
        if let Some(base) = &rope_factors {
            for (x, b) in f.iter_mut().zip(base) {
                *x *= b;
            }
        }
        rope_factors = Some(f);
    }

    let get = |n: &str| -> Result<TensorInfo, String> { g.tensor(n).cloned() };
    // Qwen3's per-head q/k RMSNorm: one weight per head dimension, and no attention biases.
    let head_norm = |n: &str| -> Result<Option<Vec<f32>>, String> {
        let t = get(n)?;
        if t.numel() as usize != head_dim {
            return Err(format!(
                "{n} has {} entries, expected the head dim {head_dim}",
                t.numel()
            ));
        }
        let mut v = vec![0.0f32; head_dim];
        dequant_row(t.kind, g.bytes(&t), &mut v);
        Ok(Some(v))
    };

    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let p = |t: &str| format!("blk.{i}.{t}");
        let gate_t = get(&p("ffn_gate_exps.weight"))?;
        let up_t = get(&p("ffn_up_exps.weight"))?;
        let down_t = get(&p("ffn_down_exps.weight"))?;
        // The experts are read as 2D matrices: one expert is a contiguous run of rows, because
        // GGUF stores `dims[0]` contiguous and the expert index runs slowest.
        let bank = |t: &TensorInfo| -> Result<Vec<TensorInfo>, String> {
            (0..n_expert)
                .map(|e| expert_view(t, e))
                .collect::<Result<Vec<_>, _>>()
        };
        // The file must hold as many experts as the metadata says.
        for t in [&gate_t, &up_t, &down_t] {
            if t.dims.len() != 3 || t.dims[2] as usize != n_expert {
                return Err(format!(
                    "{}: the file holds {:?}, expected {n_expert} experts",
                    t.name, t.dims
                ));
            }
        }
        // A row of an expert's matrix is multiplied by an activation along it, so it has to be
        // a whole number of the format's blocks.
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
        layers.push(MoeLayer {
            attn: AttnWeights {
                attn_norm: get(&p("attn_norm.weight"))?,
                wq: Linear::Dense(get(&p("attn_q.weight"))?),
                wk: Linear::Dense(get(&p("attn_k.weight"))?),
                wv: Linear::Dense(get(&p("attn_v.weight"))?),
                bq: None,
                bk: None,
                bv: None,
                q_norm: head_norm(&p("attn_q_norm.weight"))?,
                k_norm: head_norm(&p("attn_k_norm.weight"))?,
                wo: Linear::Dense(get(&p("attn_output.weight"))?),
            },
            ffn_norm: get(&p("ffn_norm.weight"))?,
            router: get(&p("ffn_gate_inp.weight"))?,
            gate: bank(&gate_t)?,
            up: bank(&up_t)?,
            down: bank(&down_t)?,
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
            (&l.attn.wq, dm, qd),
            (&l.attn.wk, dm, kd),
            (&l.attn.wv, dm, kd),
            (&l.attn.wo, qd, dm),
        ] {
            if let Linear::Dense(t) = lin {
                shape(t, d_in, d_out)?;
            }
        }
        shape(&l.attn.attn_norm, dm, 1)?;
        shape(&l.ffn_norm, dm, 1)?;
        shape(&l.router, dm, n_expert)?;
        // The three expert banks: `[d_model, n_ff_exp, n_expert]` and `[n_ff_exp, d_model,
        // n_expert]`.
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
    if let Linear::Dense(t) = &head {
        shape(t, dm, cfg.n_vocab)?;
    }

    let moe = Qwen3Moe {
        cfg: MoeCfg {
            n_expert,
            n_expert_used,
            n_ff_exp,
            w_scale,
            // llama.cpp renormalises the selected experts' weights for qwen3moe
            // (`src/models/qwen3moe.cpp`): the HF card's `norm_topk_prob: true`.
            renorm: true,
        },
        layers,
        tok_embd,
        out_norm,
        head,
    };
    Ok(Loaded {
        cfg,
        arch: Box::new(moe),
        rope_factors,
        tiles: TileStore::default(),
    })
}
