//! Qwen3.5 hybrid loader and configuration (ADR 0006).

use crate::arch::Loaded;
use crate::blocks::recurrent::RecurrentCfg;
use crate::blocks::rotate::Rotation;
use crate::families::qwen35::{mtp, Qwen35};
use crate::gguf::{Format, Gguf, TensorInfo};
use crate::model::{Config, TileStore};
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_imports))]
use crate::quant::row_bytes;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy)]
pub struct HybridCfg {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub dk: usize,
    pub dv: usize,
    pub d_conv: usize,
    pub interval: usize,
    pub had_block: usize,
    pub n_gdn: usize,
    pub n_attn: usize,
    /// Every ternary matmul is PTQ1_0 (else every one is PQ2_0): selects the activation layout.
    pub ptq1: bool,
    /// PQ2_0 matrices run from 16-row tiles (`tern::dot_tiles_pq2`) with one activation scale per `tb_span`
    /// values; 0 = the per-row kernels with one scale per 16.
    pub tb_span: usize,
}

impl HybridCfg {
    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.dk
    }
    pub fn value_dim(&self) -> usize {
        self.n_v_heads * self.dv
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// Floats of delta-rule state per sequence per linear-attention layer.
    pub fn state_floats(&self) -> usize {
        self.n_v_heads * self.dv * self.dk
    }
    /// Floats of conv input history per sequence per linear-attention layer: `(d_conv − 1)` taps per channel.
    pub fn conv_floats(&self) -> usize {
        (self.d_conv - 1) * self.conv_dim()
    }
    pub(crate) fn is_attn(&self, layer: usize) -> bool {
        (layer + 1) % self.interval == 0
    }
    /// The Gated DeltaNet geometry, for the recurrent block's steps.
    pub(crate) fn recurrent(&self) -> RecurrentCfg {
        RecurrentCfg {
            n_k_heads: self.n_k_heads,
            n_v_heads: self.n_v_heads,
            dk: self.dk,
            dv: self.dv,
            d_conv: self.d_conv,
        }
    }
}

pub(crate) enum Mixer {
    Gdn {
        qkv: TensorInfo,
        gate: TensorInfo,
        /// `ssm_beta` then `ssm_alpha`, `[2·n_v][d_model]` f32: small, unrotated, applied in f32.
        ab: Vec<f32>,
        /// Depthwise conv taps, `[d_conv][conv_dim]`; tap `d_conv − 1` multiplies the current token.
        conv: Vec<f32>,
        dt_bias: Vec<f32>,
        /// `−exp(A_log)` as stored.
        a: Vec<f32>,
        norm: Vec<f32>,
        out: TensorInfo,
        /// Index of this layer's recurrent state in the cache.
        idx: usize,
    },
    Attn {
        q: TensorInfo,
        k: TensorInfo,
        v: TensorInfo,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        o: TensorInfo,
        /// Index of this layer in the KV cache.
        idx: usize,
    },
}

pub(crate) struct HLayer {
    pub(crate) attn_norm: TensorInfo,
    pub(crate) mix: Mixer,
    pub(crate) post_norm: TensorInfo,
    pub(crate) gate: TensorInfo,
    pub(crate) up: TensorInfo,
    pub(crate) down: TensorInfo,
}

fn f32_vec(g: &Gguf, t: &TensorInfo) -> Vec<f32> {
    let mut v = vec![0.0f32; t.numel() as usize];
    crate::quant::dequant_row(t.kind, g.bytes(t), &mut v);
    v
}

pub(crate) fn load(g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
    let k = |s: &str| format!("qwen35.{s}");
    let n_nextn = g.u32(&k("nextn_predict_layers")).unwrap_or(0) as usize;
    let n_layer = (g.u32(&k("block_count"))? as usize)
        .checked_sub(n_nextn)
        .ok_or("nextn_predict_layers exceeds block_count")?;
    let d_model = g.u32(&k("embedding_length"))? as usize;
    let n_head = g.u32(&k("attention.head_count"))? as usize;
    let n_kv_head = g.u32(&k("attention.head_count_kv"))? as usize;
    let head_dim = g.u32(&k("attention.key_length"))? as usize;
    let n_ff = g.u32(&k("feed_forward_length"))? as usize;
    let rope_dim = g.u32(&k("rope.dimension_count"))? as usize;
    let interval = g.u32(&k("full_attention_interval"))? as usize;
    let hc0 = HybridCfg {
        n_k_heads: g.u32(&k("ssm.group_count"))? as usize,
        n_v_heads: g.u32(&k("ssm.time_step_rank"))? as usize,
        dk: g.u32(&k("ssm.state_size"))? as usize,
        dv: g.u32(&k("ssm.state_size"))? as usize,
        d_conv: g.u32(&k("ssm.conv_kernel"))? as usize,
        interval,
        had_block: g.u32("prism.hadamard.block_size")? as usize,
        n_gdn: 0,
        n_attn: 0,
        ptq1: false,
        tb_span: 0,
    };
    if hc0.value_dim() != g.u32(&k("ssm.inner_size"))? as usize {
        return Err("ssm.inner_size ≠ time_step_rank × state_size".into());
    }
    if g.str("prism.hadamard.transform")? != "normalized-sylvester-walsh-hadamard"
        || g.str("prism.hadamard.sign_mode")? != "explicit"
    {
        return Err("unsupported prism.hadamard transform or sign mode".into());
    }
    if !hc0.had_block.is_power_of_two()
        || n_head % n_kv_head != 0
        || hc0.n_v_heads % hc0.n_k_heads != 0
    {
        return Err("implausible hybrid geometry".into());
    }
    let cfg0 = Config {
        arch_name: "qwen35",
        rope_neox: true,
        n_layer,
        n_kv_layer: hc0.n_attn,
        n_head,
        n_kv_head,
        head_dim,
        d_model,
        n_ff,
        n_vocab: 0,
        rope_dim,
        eps: g
            .f32(&k("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6),
        rope_base: g.f32(&k("rope.freq_base")).unwrap_or(1e7),
        ctx_train: g.u32(&k("context_length")).unwrap_or(4096) as usize,
    };
    let widths = match g.meta.get("prism.hadamard.sign_widths") {
        Some(crate::gguf::Value::ArrI32(v)) => v.iter().map(|&x| x as usize).collect::<Vec<_>>(),
        _ => return Err("prism.hadamard.sign_widths missing".into()),
    };
    let vals = g.arr_i32("prism.hadamard.sign_values")?;
    if vals.len() != widths.iter().sum::<usize>() {
        return Err("prism.hadamard.sign_values length does not match sign_widths".into());
    }
    let scale = 1.0 / (hc0.had_block as f32).sqrt();
    let mut signs = Vec::new();
    let mut at = 0;
    for &w in &widths {
        if w % hc0.had_block != 0 {
            return Err(format!(
                "sign width {w} is not a multiple of the Hadamard block"
            ));
        }
        signs.push((
            w,
            vals[at..at + w].iter().map(|&s| s as f32 * scale).collect(),
        ));
        at += w;
    }
    let v_grouped = g.bool("prism.hadamard.gdn_v_grouped").unwrap_or(false);

    let get = |n: &str| -> Result<TensorInfo, String> { g.tensor(n).cloned() };
    let vecf = |n: &str| -> Result<Vec<f32>, String> { Ok(f32_vec(g, g.tensor(n)?)) };
    let ternary = |t: &TensorInfo| matches!(t.kind, Format::PQ2_0 | Format::PTQ1_0);
    let mut layers = Vec::with_capacity(n_layer);
    let (mut n_gdn, mut n_attn) = (0usize, 0usize);
    for i in 0..n_layer {
        let p = |s: &str| format!("blk.{i}.{s}");
        let mix = if hc0.is_attn(i) {
            n_attn += 1;
            Mixer::Attn {
                q: get(&p("attn_q.weight"))?,
                k: get(&p("attn_k.weight"))?,
                v: get(&p("attn_v.weight"))?,
                q_norm: vecf(&p("attn_q_norm.weight"))?,
                k_norm: vecf(&p("attn_k_norm.weight"))?,
                o: get(&p("attn_output.weight"))?,
                idx: n_attn - 1,
            }
        } else {
            n_gdn += 1;
            let mut ab = vecf(&p("ssm_beta.weight"))?;
            ab.extend(vecf(&p("ssm_alpha.weight"))?);
            Mixer::Gdn {
                qkv: get(&p("attn_qkv.weight"))?,
                gate: get(&p("attn_gate.weight"))?,
                ab,
                conv: {
                    let w = vecf(&p("ssm_conv1d.weight"))?;
                    let (kk, cd) = (hc0.d_conv, hc0.conv_dim());
                    let mut t = vec![0.0f32; kk * cd];
                    for ch in 0..cd {
                        for tap in 0..kk {
                            t[tap * cd + ch] = w[ch * kk + tap];
                        }
                    }
                    t
                },
                dt_bias: vecf(&p("ssm_dt.bias"))?,
                a: vecf(&p("ssm_a"))?,
                norm: vecf(&p("ssm_norm.weight"))?,
                out: get(&p("ssm_out.weight"))?,
                idx: n_gdn - 1,
            }
        };
        let l = HLayer {
            attn_norm: get(&p("attn_norm.weight"))?,
            mix,
            post_norm: get(&p("post_attention_norm.weight"))?,
            gate: get(&p("ffn_gate.weight"))?,
            up: get(&p("ffn_up.weight"))?,
            down: get(&p("ffn_down.weight"))?,
        };
        for t in [&l.gate, &l.up, &l.down] {
            if !ternary(t) {
                return Err(format!(
                    "{} is {}; the hybrid path runs ternary matmuls",
                    t.name,
                    t.kind.label()
                ));
            }
        }
        layers.push(l);
    }

    let mut kinds: Vec<Format> = Vec::new();
    for l in &layers {
        let mut ts: Vec<&TensorInfo> = vec![&l.gate, &l.up, &l.down];
        match &l.mix {
            Mixer::Gdn { qkv, gate, out, .. } => ts.extend([qkv, gate, out]),
            Mixer::Attn { q, k, v, o, .. } => ts.extend([q, k, v, o]),
        }
        for t in ts {
            if !kinds.contains(&t.kind) {
                kinds.push(t.kind);
            }
        }
    }

    let tied = g.bool("prism.hadamard.tied_output").unwrap_or(false);
    let head_name = if tied {
        "token_embd.weight"
    } else {
        "output.weight"
    };
    if tied && g.tensor("output.weight").is_ok() {
        return Err("prism.hadamard.tied_output with an output.weight present".into());
    }
    let output_kind = g.tensor(head_name)?.kind;
    if !kinds.contains(&output_kind) {
        kinds.push(output_kind);
    }
    let ptq1 = match kinds.as_slice() {
        [Format::PQ2_0] => false,
        [Format::PTQ1_0] => true,
        other => {
            return Err(format!(
            "the hybrid path needs one ternary format for every matmul; this file mixes {other:?}"
        ))
        }
    };

    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    let mut output = g.tensor(head_name)?.clone();
    #[allow(unused_mut)]
    let mut tiles = TileStore::default();
    #[allow(unused_mut)]
    let mut tb_span = 0usize;
    #[cfg(target_arch = "x86_64")]
    {
        let want = !ptq1 && crate::table::MatKernels::get().tiles() && crate::options::get().tiles;
        let mut mats: Vec<&mut TensorInfo> = Vec::new();
        for l in layers.iter_mut() {
            mats.extend([&mut l.gate, &mut l.up, &mut l.down]);
            match &mut l.mix {
                Mixer::Gdn { qkv, gate, out, .. } => mats.extend([qkv, gate, out]),
                Mixer::Attn { q, k, v, o, .. } => mats.extend([q, k, v, o]),
            }
        }
        mats.push(&mut output);
        let ok = mats
            .iter()
            .all(|t| t.dims.get(1).is_some_and(|r| r % 16 == 0) && t.dims[0] % 128 == 0);
        if want && ok {
            let total: usize = mats.iter().map(|t| t.bytes() as usize).sum();
            tiles = TileStore::new(total);
            let mut at = 0usize;
            for t in mats {
                let (n, rows) = (t.dims[0] as usize, t.dims[1] as usize);
                let rb = row_bytes(t.kind, n);
                let len = rows * rb;
                crate::tern::repack_tiles(
                    g.bytes(t),
                    rows,
                    rb,
                    &mut tiles.bytes_mut()[at..at + len],
                );
                t.tile_off = Some(at);
                at += len;
            }
            tb_span = crate::options::get().tb_span;
        }
    }
    let hc = HybridCfg {
        n_gdn,
        n_attn,
        ptq1,
        tb_span,
        ..hc0
    };
    let tok_embd = get("token_embd.weight")?;
    let out_norm = get("output_norm.weight")?;
    let cfg = Config {
        n_vocab: tk.n_vocab,
        n_kv_layer: hc.n_attn,
        ..cfg0
    };
    if tok_embd.dims.get(1).copied().unwrap_or(0) as usize != cfg.n_vocab {
        return Err("token_embd rows ≠ vocabulary".into());
    }
    let unit = if ptq1 { 256 } else { 128 };
    for (what, w) in [
        ("embedding", d_model),
        ("feed-forward", n_ff),
        ("value", hc.value_dim()),
        ("query", cfg.q_dim()),
    ] {
        if w % unit != 0 {
            return Err(format!("{what} width {w} is not a multiple of {unit}"));
        }
    }
    let rot = Rotation::new(signs, hc.had_block);
    for w in [d_model, n_ff, hc.value_dim(), cfg.q_dim()] {
        rot.signs_for(w)?;
    }
    let mtp = if n_nextn == 1 {
        let se = rot.signs_for(d_model)?;
        Some(mtp::Mtp::load(
            g, n_layer, &tok_embd, &output, &rot, se, &hc,
        )?)
    } else {
        None
    };
    let qwen35 = Qwen35 {
        cfg: hc,
        layers,
        rot,
        v_grouped,
        tok_embd,
        out_norm,
        head: output,
        mtp,
    };
    Ok(Loaded {
        cfg,
        arch: Box::new(qwen35),
        rope_factors: None,
        tiles,
    })
}
