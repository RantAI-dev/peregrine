#[cfg(feature = "mtp")]
use crate::qwen35::mtp;
use crate::qwen35::Qwen35;
use peregrine_core::arch::Loaded;
use peregrine_core::blocks::recurrent::RecurrentCfg;
use peregrine_core::blocks::rotate::Rotation;
use peregrine_core::gguf::{Format, Gguf, TensorInfo, Value};
use peregrine_core::model::{Config, TileStore};
#[cfg(target_arch = "x86_64")]
use peregrine_core::options;
use peregrine_core::tokenizer::Tokenizer;
use peregrine_kernels::quant::dequant_row;
#[allow(unused_imports)]
use peregrine_kernels::quant::row_bytes;
#[cfg(target_arch = "x86_64")]
use peregrine_kernels::table::MatKernels;
#[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
use peregrine_kernels::tern;

#[derive(Debug, Clone, Copy)]
pub struct HybridCfg {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub dk: usize,
    pub dv: usize,
    pub d_conv: usize,
    pub interval: usize,
    /// The Hadamard block size; 0 when the file has no rotation.
    pub had_block: usize,
    pub n_gdn: usize,
    pub n_attn: usize,
    /// Every matmul is a ternary format (else every one is an ordinary quantised matrix):
    /// selects the activation layout.
    pub ternary: bool,
    /// Every ternary matmul is PTQ1_0 (else every one is PQ2_0): selects the activation layout.
    pub ptq1: bool,
    /// PQ2_0 matrices run from 16-row tiles (`tern::dot_tiles_pq2`) with one activation scale per `tb_span`
    /// values; 0 = the per-row kernels with one scale per 16.
    pub tb_span: usize,
    /// Some matmul is F32/F16/BF16: the activations stay unquantised beside the codes.
    pub keep_f32: bool,
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
    dequant_row(t.kind, g.bytes(t), &mut v);
    v
}

/// The `prism.hadamard.*` keys the rotation is made of. A file that carries none of them
/// has no rotation: that is a plain Qwen 3.5 file, whose weights were not folded into a
/// Hadamard basis, and none is applied to it.
///
/// llama.cpp v0.9.0 switches on `prism.hadamard.version` alone
/// (`llama_model_base::load_hparams_hadamard`); the list here is wider, because a file
/// that carries the rotation's other keys without the version still has one.
const HADAMARD_KEYS: [&str; 6] = [
    "prism.hadamard.version",
    "prism.hadamard.block_size",
    "prism.hadamard.transform",
    "prism.hadamard.sign_mode",
    "prism.hadamard.sign_widths",
    "prism.hadamard.sign_values",
];

/// The file's rotation, if it has one.
///
/// A Bonsai 2 file carries the block Hadamard transform it folded its weights into, in
/// `prism.hadamard.*`: the block size, the transform's name, the sign mode and the sign
/// vector of every input width. A plain Qwen 3.5 file carries none of them, and the
/// engine applies no rotation to it — the weights are used as they are.
///
/// The keys are read as llama.cpp v0.9.0 reads them (`llama_model_base::load_hparams_hadamard`):
/// the transform's name and the sign mode must be the ones the engine implements, and the
/// sign vectors must cover every width the file's folded weights take.
fn load_rotation(g: &Gguf) -> Result<Option<Rotation>, String> {
    if !HADAMARD_KEYS.iter().any(|k| g.meta.contains_key(*k)) {
        return Ok(None);
    }
    let had_block = g.u32("prism.hadamard.block_size")? as usize;
    if g.str("prism.hadamard.transform")? != "normalized-sylvester-walsh-hadamard"
        || g.str("prism.hadamard.sign_mode")? != "explicit"
    {
        return Err("unsupported prism.hadamard transform or sign mode".into());
    }
    if !had_block.is_power_of_two() {
        return Err(format!("invalid prism.hadamard.block_size: {had_block}"));
    }
    let widths = match g.meta.get("prism.hadamard.sign_widths") {
        Some(Value::ArrI32(v)) => v.iter().map(|&x| x as usize).collect::<Vec<_>>(),
        _ => return Err("prism.hadamard.sign_widths missing".into()),
    };
    let vals = g.arr_i32("prism.hadamard.sign_values")?;
    if vals.len() != widths.iter().sum::<usize>() {
        return Err("prism.hadamard.sign_values length does not match sign_widths".into());
    }
    let scale = 1.0 / (had_block as f32).sqrt();
    let mut signs = Vec::new();
    let mut at = 0;
    for &w in &widths {
        if w % had_block != 0 {
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
    Ok(Some(Rotation::new(signs, had_block)))
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
    // The rotation, if the file has one: a Bonsai 2 file carries it, a plain Qwen 3.5
    // file does not, and none is applied to it.
    let rot = load_rotation(g)?;
    let had_block = rot.as_ref().map_or(0, |r| r.block());
    let hc0 = HybridCfg {
        n_k_heads: g.u32(&k("ssm.group_count"))? as usize,
        n_v_heads: g.u32(&k("ssm.time_step_rank"))? as usize,
        dk: g.u32(&k("ssm.state_size"))? as usize,
        dv: g.u32(&k("ssm.state_size"))? as usize,
        d_conv: g.u32(&k("ssm.conv_kernel"))? as usize,
        interval,
        had_block,
        n_gdn: 0,
        n_attn: 0,
        ternary: false,
        ptq1: false,
        tb_span: 0,
        keep_f32: false,
    };
    if hc0.value_dim() != g.u32(&k("ssm.inner_size"))? as usize {
        return Err("ssm.inner_size ≠ time_step_rank × state_size".into());
    }
    if n_head % n_kv_head != 0 || hc0.n_v_heads % hc0.n_k_heads != 0 {
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
        rope_yarn: None,
    };
    // The value heads are permuted from the grouped order to the tiled one only when the
    // file says so (`prism.hadamard.gdn_v_grouped`): a plain Qwen 3.5 file's converter
    // folds the permutation into the weights, so it needs none.
    let v_grouped = g.bool("prism.hadamard.gdn_v_grouped").unwrap_or(false);

    let get = |n: &str| -> Result<TensorInfo, String> { g.tensor(n).cloned() };
    let vecf = |n: &str| -> Result<Vec<f32>, String> { Ok(f32_vec(g, g.tensor(n)?)) };
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

    // The output head: `output.weight`, or the embedding table when the file ties them —
    // `prism.hadamard.tied_output` for a Bonsai 2 file, the absence of `output.weight` for
    // a plain one (llama.cpp v0.9.0 falls back to the embedding table the same way).
    let tied = g.bool("prism.hadamard.tied_output").unwrap_or(false);
    if tied && g.tensor("output.weight").is_ok() {
        return Err("prism.hadamard.tied_output with an output.weight present".into());
    }
    let head_name = if tied || g.tensor("output.weight").is_err() {
        "token_embd.weight"
    } else {
        "output.weight"
    };
    let output_kind = g.tensor(head_name)?.kind;
    if !kinds.contains(&output_kind) {
        kinds.push(output_kind);
    }
    // The formats: either every matmul is the same ternary format (a Bonsai 2 file, whose
    // weights were folded into a Hadamard basis and quantised to three levels), or none is
    // (a plain Qwen 3.5 file, whose matmuls are ordinary quantised matrices). A file that
    // mixes the two is refused: the activation layout is one or the other.
    let n_ternary = kinds
        .iter()
        .filter(|k| matches!(k, Format::PQ2_0 | Format::PTQ1_0))
        .count();
    let ternary = n_ternary == kinds.len();
    if n_ternary != 0 && !ternary {
        return Err(format!(
            "the file mixes ternary and ordinary weight formats: {kinds:?}"
        ));
    }
    let ptq1 = if ternary {
        match kinds.as_slice() {
            [Format::PQ2_0] => false,
            [Format::PTQ1_0] => true,
            other => {
                return Err(format!(
            "the hybrid path needs one ternary format for every matmul; this file mixes {other:?}"
        ))
            }
        }
    } else {
        false
    };
    // An F32, F16 or BF16 matrix multiplies the activations unquantised.
    let keep_f32 = kinds
        .iter()
        .any(|k| matches!(k, Format::F32 | Format::F16 | Format::BF16));

    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    let mut output = g.tensor(head_name)?.clone();
    #[allow(unused_mut)]
    let mut tiles = TileStore::default();
    #[allow(unused_mut)]
    let mut tb_span = 0usize;
    #[cfg(target_arch = "x86_64")]
    {
        let want = ternary && !ptq1 && MatKernels::get().tiles() && options::get().tiles;
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
            #[cfg(not(all(target_arch = "x86_64", feature = "fmt-ternary")))]
            {
                return Err(
                    "ternary format disabled in this build (recompile with feature 'fmt-ternary')"
                        .to_string(),
                );
            }
            #[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
            {
                let total: usize = mats.iter().map(|t| t.bytes() as usize).sum();
                tiles = TileStore::new(total);
                let mut at = 0usize;
                for t in mats {
                    let (n, rows) = (t.dims[0] as usize, t.dims[1] as usize);
                    let rb = row_bytes(t.kind, n);
                    let len = rows * rb;
                    tern::repack_tiles(g.bytes(t), rows, rb, &mut tiles.bytes_mut()[at..at + len]);
                    t.tile_off = Some(at);
                    at += len;
                }
                tb_span = options::get().tb_span;
            }
        }
    }
    let hc = HybridCfg {
        n_gdn,
        n_attn,
        ternary,
        ptq1,
        tb_span,
        keep_f32,
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
    // The widths the matmul inputs take: the ternary layout needs a multiple of its group
    // (128, or 256 for PTQ1_0's pairs of groups), the ordinary one a multiple of the
    // activation quantiser's block (32).
    let unit = if ternary {
        if ptq1 {
            256
        } else {
            128
        }
    } else {
        32
    };
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
    // A file with a rotation must carry a sign vector for every width its matmuls take.
    if let Some(r) = &rot {
        for w in [d_model, n_ff, hc.value_dim(), cfg.q_dim()] {
            r.signs_for(w)?;
        }
    }
    #[cfg(feature = "mtp")]
    let mtp = if n_nextn == 1 {
        let se = rot.as_ref().map(|r| r.signs_for(d_model)).transpose()?;
        Some(mtp::Mtp::load(
            g,
            n_layer,
            &tok_embd,
            &output,
            rot.as_ref(),
            se,
            &hc,
        )?)
    } else {
        None
    };
    #[cfg(not(feature = "mtp"))]
    if n_nextn > 0 {
        eprintln!("info: MTP blocks in model ignored (recompile with feature 'mtp')");
    }
    let qwen35 = Qwen35 {
        cfg: hc,
        layers,
        rot,
        v_grouped,
        tok_embd,
        out_norm,
        head: output,
        #[cfg(feature = "mtp")]
        mtp,
    };
    Ok(Loaded {
        cfg,
        arch: Box::new(qwen35),
        rope_factors: None,
        tiles,
    })
}
