//! Builders for tiny synthetic models: shapes small enough to run in milliseconds, large enough that
//! every block format and the batched kernel paths accept them.

use super::vocab::{self, TemplateKind, VOCAB_SIZE};
use super::weights::{generate_norm, generate_weights, Rng};
use super::writer::{GgufWriter, MetaValue};
use peregrine_core::gguf::Format;
use std::path::{Path, PathBuf};

/// A dense (llama-family) model's shape.
#[derive(Clone, Copy, Debug)]
pub struct Dense {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub ctx: usize,
    /// Write no `output.weight`: the file ties its output head to the token embedding.
    pub tied: bool,
}

impl Default for Dense {
    fn default() -> Self {
        // Every matrix row is a multiple of 256 (the K-quant block), heads are 32 wide.
        Dense {
            d_model: 256,
            n_layer: 2,
            n_head: 8,
            n_head_kv: 4,
            n_ff: 512,
            ctx: 512,
            tied: false,
        }
    }
}

/// How a dense llama file expresses its rotary frequencies.
#[derive(Clone, Copy, Debug)]
pub enum Rope {
    /// The metadata's base angle only.
    None,
    /// Llama 3.x's per-frequency factor table (`rope_freqs.weight`).
    Freqs,
    /// Linear scaling through `rope.scaling.type = "linear"` and `rope.scaling.factor`.
    Linear { factor: f32 },
}

/// Write a dense `arch` model ("llama", "qwen2" or "qwen3") whose matrices are all `fmt`, seeded by
/// `seed`. `qwen2` carries the attention biases and NeoX rotary pairing its loader requires.
pub fn write_dense(dir: &Path, arch: &str, fmt: Format, seed: u64, s: Dense) -> PathBuf {
    write_dense_impl(dir, arch, fmt, seed, s, Rope::None, false, None, &[])
}

/// A dense `arch` file that declares the pooling type an embedder was trained with
/// (`<arch>.pooling_type`: 1 mean, 3 last). The weights are the same as [`write_dense`]'s.
pub fn write_dense_pooled(
    dir: &Path,
    arch: &str,
    fmt: Format,
    seed: u64,
    s: Dense,
    pooling: u32,
) -> PathBuf {
    write_dense_impl(
        dir,
        arch,
        fmt,
        seed,
        s,
        Rope::None,
        false,
        Some(pooling),
        &[],
    )
}

/// The dense model whose drafts are accepted: every output-head row is identical, so all logits tie and
/// the argmax picks row 0 whatever the hidden state — the model always answers token 0. A corpus of
/// zeros is what the suffix index needs to find a match in (the same n-gram twice), and what makes the
/// draft it proposes the token the target draws.
pub fn write_dense_accepting(dir: &Path, arch: &str, fmt: Format, seed: u64, s: Dense) -> PathBuf {
    write_dense_impl(dir, arch, fmt, seed, s, Rope::None, true, None, &[])
}

/// A dense model carrying extra GGUF metadata (the decision keys of S4.3, or one deliberately
/// invalid entry for a validation test).
pub fn write_dense_extra(
    dir: &Path,
    arch: &str,
    fmt: Format,
    seed: u64,
    s: Dense,
    extra: &[(String, MetaValue)],
) -> PathBuf {
    write_dense_impl(dir, arch, fmt, seed, s, Rope::None, false, None, extra)
}

/// The decision metadata a synthetic decision model carries: `<arch>.decision.type`, the
/// temperatures, the score biases, the labels and the `systemone` template's bytes.
///
/// The template here is opaque fixture text, not a real prompt template: this stage only records
/// its presence and length.
#[derive(Clone, Default, Debug)]
pub struct DecisionSpec {
    pub type_name: Option<String>,
    pub temperatures: Vec<(String, f32)>,
    pub score_bias: Vec<((usize, usize), f32)>,
    pub labels: Option<Vec<String>>,
    /// The bytes stored under `tokenizer.chat_template.systemone`.
    pub template: Option<String>,
}

impl DecisionSpec {
    /// A spec for `type_name` with a short opaque template.
    pub fn new(type_name: &str) -> DecisionSpec {
        DecisionSpec {
            type_name: Some(type_name.to_string()),
            template: Some("State: {state}. Question: {question}".to_string()),
            ..Default::default()
        }
    }

    /// A spec for a real `openjev` model: the type, the temperatures the published file carries and
    /// the template the renderer was written from (as bytes, never printed). The template comes
    /// from [`peregrine_core::systemone::OPENJEV_TEMPLATE`], so the synthetic file passes the same
    /// load-time check a real file does.
    pub fn openjev() -> DecisionSpec {
        DecisionSpec {
            type_name: Some("openjev".to_string()),
            temperatures: vec![
                ("choice".to_string(), 0.85),
                ("score".to_string(), 0.85),
                ("noul".to_string(), 1.554_713),
            ],
            template: Some(peregrine_core::systemone::OPENJEV_TEMPLATE.to_string()),
            ..Default::default()
        }
    }

    /// The GGUF entries this spec writes for architecture `arch`.
    pub fn keys(&self, arch: &str) -> Vec<(String, MetaValue)> {
        let mut out = Vec::new();
        if let Some(t) = &self.type_name {
            out.push((format!("{arch}.decision.type"), MetaValue::Str(t.clone())));
        }
        for (name, value) in &self.temperatures {
            out.push((
                format!("{arch}.decision.temperature.{name}"),
                MetaValue::F32(*value),
            ));
        }
        for ((n, level), value) in &self.score_bias {
            out.push((
                format!("{arch}.decision.score_bias.{n}.{level}"),
                MetaValue::F32(*value),
            ));
        }
        if let Some(labels) = &self.labels {
            out.push((
                format!("{arch}.decision.labels"),
                MetaValue::ArrStr(labels.clone()),
            ));
        }
        if let Some(template) = &self.template {
            out.push((
                "tokenizer.chat_template.systemone".to_string(),
                MetaValue::Str(template.clone()),
            ));
        }
        out
    }
}

/// A llama file in F16 with the requested rotary-frequency treatment, for the rope-scaling loaders.
pub fn write_rope_dense(dir: &Path, rope: Rope, seed: u64) -> PathBuf {
    write_dense_impl(
        dir,
        "llama",
        Format::F16,
        seed,
        Dense::default(),
        rope,
        false,
        None,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn write_dense_impl(
    dir: &Path,
    arch: &str,
    fmt: Format,
    seed: u64,
    s: Dense,
    rope: Rope,
    accept: bool,
    pooling: Option<u32>,
    extra: &[(String, MetaValue)],
) -> PathBuf {
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    let head_dim = s.d_model / s.n_head;
    w.str("general.architecture", arch);
    w.str("general.name", &format!("synthetic-{arch}-{}", fmt.label()));
    let k = |x: &str| format!("{arch}.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("attention.head_count_kv"), s.n_head_kv as u32);
    w.u32(&k("feed_forward_length"), s.n_ff as u32);
    w.u32(&k("context_length"), s.ctx as u32);
    w.u32(&k("vocab_size"), VOCAB_SIZE as u32);
    w.u32(&k("rope.dimension_count"), head_dim as u32);
    if let Some(p) = pooling {
        w.u32(&k("pooling_type"), p);
    }
    w.f32(&k("rope.freq_base"), 10000.0);
    if let Rope::Linear { factor } = rope {
        w.str(&k("rope.scaling.type"), "linear");
        w.f32(&k("rope.scaling.factor"), factor);
    }
    w.f32(&k("attention.layer_norm_rms_epsilon"), 1e-6);
    // Extra metadata (the decision keys of S4.3, or a deliberately invalid one).
    for (key, value) in extra {
        w.add_meta(key, value.clone());
    }
    if arch == "qwen3" {
        w.u32(&k("attention.key_length"), head_dim as u32);
        w.u32(&k("attention.value_length"), head_dim as u32);
    }
    let (pre, tpl) = if arch == "llama" {
        ("llama-bpe", TemplateKind::Llama3)
    } else {
        ("qwen2", TemplateKind::Qwen3)
    };
    vocab::add_tokenizer_metadata(&mut w, pre);
    w.str("tokenizer.chat_template", vocab::chat_template(tpl));

    let kv = s.n_head_kv * head_dim;
    let mat = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64],
            fmt,
            generate_weights(fmt, cols * rows, rng),
        );
    };
    let norm = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        w.add_tensor(name, &[n as u64], Format::F32, generate_norm(n, rng));
    };
    mat(&mut w, &mut rng, "token_embd.weight", s.d_model, VOCAB_SIZE);
    for i in 0..s.n_layer {
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_norm.weight"),
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_q.weight"),
            s.d_model,
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_k.weight"),
            s.d_model,
            kv,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_v.weight"),
            s.d_model,
            kv,
        );
        if arch == "qwen2" {
            // Qwen2's attention biases: the loader requires all three and decodes
            // them once (`Model::load`'s `bias`).
            let bias = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
                let data: Vec<u8> = (0..n)
                    .flat_map(|_| rng.f32_range(-0.1, 0.1).to_le_bytes())
                    .collect();
                w.add_tensor(name, &[n as u64], Format::F32, data);
            };
            bias(&mut w, &mut rng, &format!("blk.{i}.attn_q.bias"), s.d_model);
            bias(&mut w, &mut rng, &format!("blk.{i}.attn_k.bias"), kv);
            bias(&mut w, &mut rng, &format!("blk.{i}.attn_v.bias"), kv);
        }
        if arch == "qwen3" {
            norm(
                &mut w,
                &mut rng,
                &format!("blk.{i}.attn_q_norm.weight"),
                head_dim,
            );
            norm(
                &mut w,
                &mut rng,
                &format!("blk.{i}.attn_k_norm.weight"),
                head_dim,
            );
        }
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_output.weight"),
            s.d_model,
            s.d_model,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_norm.weight"),
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_gate.weight"),
            s.d_model,
            s.n_ff,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_up.weight"),
            s.d_model,
            s.n_ff,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_down.weight"),
            s.n_ff,
            s.d_model,
        );
    }
    norm(&mut w, &mut rng, "output_norm.weight", s.d_model);
    // A tied file writes no `output.weight`: the loader uses `token_embd.weight`.
    if !s.tied {
        if accept {
            // Every output row is the same row, so every logit is the same value and
            // the argmax picks row 0 whatever the hidden state: the model always
            // answers token 0.
            let row = generate_weights(fmt, s.d_model, &mut rng);
            let mut data = Vec::with_capacity(row.len() * VOCAB_SIZE);
            for _ in 0..VOCAB_SIZE {
                data.extend_from_slice(&row);
            }
            w.add_tensor(
                "output.weight",
                &[s.d_model as u64, VOCAB_SIZE as u64],
                fmt,
                data,
            );
        } else {
            mat(&mut w, &mut rng, "output.weight", s.d_model, VOCAB_SIZE);
        }
    }
    if matches!(rope, Rope::Freqs) {
        // Llama 3.x stores one positive factor per rotary frequency; the loader divides
        // each angle by it. Kept near 1 so every angle stays finite.
        w.add_tensor(
            "rope_freqs.weight",
            &[(head_dim / 2) as u64],
            Format::F32,
            generate_norm(head_dim / 2, &mut rng),
        );
    }
    let name = match (accept, rope) {
        (true, _) => format!("{arch}-{}-accept", fmt.label()),
        (false, Rope::None) => format!(
            "{arch}-{}{}",
            match pooling {
                Some(p) => format!("{}-pool{p}", fmt.label()),
                None => fmt.label().to_string(),
            },
            if s.tied { "-tied" } else { "" }
        ),
        (false, Rope::Freqs) => format!("{arch}-rope-freqs-{}", fmt.label()),
        (false, Rope::Linear { .. }) => format!("{arch}-rope-linear-{}", fmt.label()),
    };
    let path = dir.join(format!("{name}.gguf"));
    w.write_to_file(&path).expect("write synthetic model");
    path
}

/// A Gemma 3 model's shape.
#[derive(Clone, Copy, Debug)]
pub struct Gemma3Config {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub ctx: usize,
    pub sliding_window: usize,
    pub rope: Rope,
}

impl Default for Gemma3Config {
    fn default() -> Self {
        Gemma3Config {
            d_model: 256,
            n_layer: 6,
            n_head: 8,
            n_head_kv: 4,
            n_ff: 512,
            ctx: 512,
            sliding_window: 16,
            rope: Rope::None,
        }
    }
}

/// Write a synthetic `gemma3` model whose matrices are all `fmt`, seeded by `seed`.
pub fn write_gemma3(dir: &Path, fmt: Format, seed: u64, s: Gemma3Config) -> PathBuf {
    write_gemma3_named(dir, fmt, seed, s, None)
}

/// Write a synthetic `gemma3` model with optional custom file stem / general.name.
pub fn write_gemma3_named(
    dir: &Path,
    fmt: Format,
    seed: u64,
    s: Gemma3Config,
    custom_name: Option<&str>,
) -> PathBuf {
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    let head_dim = s.d_model / s.n_head;
    w.str("general.architecture", "gemma3");
    let name = custom_name
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("synthetic-gemma3-{}", fmt.label()));
    w.str("general.name", &name);
    let k = |x: &str| format!("gemma3.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("attention.head_count_kv"), s.n_head_kv as u32);
    w.u32(&k("feed_forward_length"), s.n_ff as u32);
    w.u32(&k("context_length"), s.ctx as u32);
    w.u32(&k("vocab_size"), VOCAB_SIZE as u32);
    w.u32(&k("attention.key_length"), head_dim as u32);
    w.u32(&k("attention.value_length"), head_dim as u32);
    w.u32(&k("rope.dimension_count"), head_dim as u32);
    w.f32(&k("rope.freq_base"), 1_000_000.0);
    w.f32(&k("rope.freq_base_swa"), 10_000.0);
    w.u32(&k("attention.sliding_window"), s.sliding_window as u32);
    w.f32(&k("attention.layer_norm_rms_epsilon"), 1e-6);
    if let Rope::Linear { factor } = s.rope {
        w.str(&k("rope.scaling.type"), "linear");
        w.f32(&k("rope.scaling.factor"), factor);
    }

    vocab::add_gemma_tokenizer_metadata(&mut w);
    w.str(
        "tokenizer.chat_template",
        vocab::chat_template(TemplateKind::Gemma),
    );

    let kv = s.n_head_kv * head_dim;
    let mat = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64],
            fmt,
            generate_weights(fmt, cols * rows, rng),
        );
    };
    let norm = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        w.add_tensor(name, &[n as u64], Format::F32, generate_norm(n, rng));
    };

    mat(&mut w, &mut rng, "token_embd.weight", s.d_model, VOCAB_SIZE);
    for i in 0..s.n_layer {
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_norm.weight"),
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_q.weight"),
            s.d_model,
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_k.weight"),
            s.d_model,
            kv,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_v.weight"),
            s.d_model,
            kv,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_q_norm.weight"),
            head_dim,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_k_norm.weight"),
            head_dim,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_output.weight"),
            s.d_model,
            s.d_model,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.post_attention_norm.weight"),
            s.d_model,
        );

        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_norm.weight"),
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_gate.weight"),
            s.d_model,
            s.n_ff,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_up.weight"),
            s.d_model,
            s.n_ff,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_down.weight"),
            s.n_ff,
            s.d_model,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.post_ffw_norm.weight"),
            s.d_model,
        );
    }
    norm(&mut w, &mut rng, "output_norm.weight", s.d_model);
    // output.weight is intentionally omitted to verify tied word embeddings

    let file_stem = custom_name
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("gemma3-{}", fmt.label()));
    let path = dir.join(format!("{file_stem}.gguf"));
    w.write_to_file(&path).expect("write synthetic model");
    path
}

/// An EmbeddingGemma-shaped model: the Gemma 3 blocks, bidirectional, plus the two
/// sentence-transformers dense modules and mean pooling.
#[derive(Clone, Copy, Debug)]
pub struct GemmaEmbeddingConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub dense_width: usize,
    pub ctx: usize,
    pub sliding_window: usize,
    pub pooling_type: u32,
}

impl Default for GemmaEmbeddingConfig {
    fn default() -> Self {
        GemmaEmbeddingConfig {
            d_model: 128,
            n_layer: 6,
            n_head: 4,
            n_head_kv: 2,
            n_ff: 256,
            dense_width: 256,
            ctx: 256,
            sliding_window: 8,
            pooling_type: 1,
        }
    }
}

/// Write a synthetic `gemma-embedding` model (roadmap S4.4).
///
/// Six layers with llama.cpp's default sliding-window pattern (five local at window 8 and
/// one global), GQA (four query heads over two), per-head q/k norms, GeGLU, and the
/// `dense_2`/`dense_3` pair. There is no output head: the file never generates.
pub fn write_gemma_embedding(
    dir: &Path,
    fmt: Format,
    seed: u64,
    s: GemmaEmbeddingConfig,
) -> PathBuf {
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    let head_dim = s.d_model / s.n_head;
    w.str("general.architecture", "gemma-embedding");
    w.str(
        "general.name",
        &format!("synthetic-gemma-embedding-{}", fmt.label()),
    );
    let k = |x: &str| format!("gemma-embedding.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("attention.head_count_kv"), s.n_head_kv as u32);
    w.u32(&k("feed_forward_length"), s.n_ff as u32);
    w.u32(&k("context_length"), s.ctx as u32);
    w.u32(&k("vocab_size"), VOCAB_SIZE as u32);
    w.u32(&k("attention.key_length"), head_dim as u32);
    w.u32(&k("attention.value_length"), head_dim as u32);
    w.u32(&k("rope.dimension_count"), head_dim as u32);
    w.f32(&k("rope.freq_base"), 1_000_000.0);
    w.f32(&k("rope.freq_base_swa"), 10_000.0);
    w.u32(&k("attention.sliding_window"), s.sliding_window as u32);
    w.f32(&k("attention.layer_norm_rms_epsilon"), 1e-6);
    w.u32(&k("dense_2_feat_in"), s.d_model as u32);
    w.u32(&k("dense_2_feat_out"), s.dense_width as u32);
    w.u32(&k("dense_3_feat_in"), s.dense_width as u32);
    w.u32(&k("dense_3_feat_out"), s.d_model as u32);
    w.u32(&k("pooling_type"), s.pooling_type);

    vocab::add_gemma_tokenizer_metadata(&mut w);
    // No chat template: an embedding-only file has none, and the build says so.

    let kv = s.n_head_kv * head_dim;
    let mat = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64],
            fmt,
            generate_weights(fmt, cols * rows, rng),
        );
    };
    let norm = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        w.add_tensor(name, &[n as u64], Format::F32, generate_norm(n, rng));
    };

    mat(&mut w, &mut rng, "token_embd.weight", s.d_model, VOCAB_SIZE);
    for i in 0..s.n_layer {
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_norm.weight"),
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_q.weight"),
            s.d_model,
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_k.weight"),
            s.d_model,
            kv,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_v.weight"),
            s.d_model,
            kv,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_q_norm.weight"),
            head_dim,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_k_norm.weight"),
            head_dim,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_output.weight"),
            s.d_model,
            s.d_model,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.post_attention_norm.weight"),
            s.d_model,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_norm.weight"),
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_gate.weight"),
            s.d_model,
            s.n_ff,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_up.weight"),
            s.d_model,
            s.n_ff,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_down.weight"),
            s.n_ff,
            s.d_model,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.post_ffw_norm.weight"),
            s.d_model,
        );
    }
    norm(&mut w, &mut rng, "output_norm.weight", s.d_model);
    mat(&mut w, &mut rng, "dense_2.weight", s.d_model, s.dense_width);
    mat(&mut w, &mut rng, "dense_3.weight", s.dense_width, s.d_model);

    let path = dir.join(format!("gemma-embedding-{}.gguf", fmt.label()));
    w.write_to_file(&path).expect("write synthetic model");
    path
}

/// The shape of a synthetic `bert` encoder (roadmap S4.6a).
#[derive(Clone, Copy, Debug)]
pub struct BertConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub ctx: usize,
    /// `bert.pooling_type`: 2 is cls (BGE's default), 1 is mean (E5/GTE-style).
    pub pooling_type: u32,
    /// Whether the file carries the WordPiece (`bert`) or the unigram (`t5`, XLM-R)
    /// tokenizer. The weights are the same either way.
    pub unigram: bool,
    /// Token-type rows in `token_types.weight` (BERT's Sentence A/B is 2).
    pub n_token_types: usize,
}

impl Default for BertConfig {
    fn default() -> Self {
        BertConfig {
            d_model: 128,
            n_layer: 2,
            n_head: 4,
            n_ff: 256,
            ctx: 64,
            pooling_type: 2,
            unigram: false,
            n_token_types: 2,
        }
    }
}

/// Write a synthetic `bert` encoder: post-norm blocks, absolute positions, token
/// types, biases on every projection, LayerNorm (weight and bias) and the exact
/// GELU — the graph `src/models/bert.cpp` builds.
///
/// Every matrix is `fmt` (F32 or Q8_0, both multiples of the block size); the
/// norms and the biases are F32, as the converter writes them. `pooling_type` is
/// the file's own (`1` mean, `2` cls), so `Model::pooling` reports it.
pub fn write_bert(dir: &Path, fmt: Format, seed: u64, s: BertConfig) -> PathBuf {
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    w.str("general.architecture", "bert");
    w.str("general.name", &format!("synthetic-bert-{}", fmt.label()));
    let k = |x: &str| format!("bert.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("feed_forward_length"), s.n_ff as u32);
    w.u32(&k("context_length"), s.ctx as u32);
    w.u32(&k("vocab_size"), VOCAB_SIZE as u32);
    w.f32(&k("attention.layer_norm_epsilon"), 1e-5);
    w.u32(&k("pooling_type"), s.pooling_type);
    w.bool(&k("causal_attention"), false);

    if s.unigram {
        vocab::add_unigram_tokenizer_metadata(&mut w);
    } else {
        vocab::add_bert_tokenizer_metadata(&mut w);
    }

    let mat = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64],
            fmt,
            generate_weights(fmt, cols * rows, rng),
        );
    };
    let norm = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        w.add_tensor(name, &[n as u64], Format::F32, generate_norm(n, rng));
    };
    // A bias of its own scale, and never zero: a bias the engine forgot to add
    // must move the vector.
    let bias = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            let v = rng.f32_range(-0.05, 0.05);
            out.extend_from_slice(&v.to_le_bytes());
        }
        w.add_tensor(name, &[n as u64], Format::F32, out);
    };

    mat(&mut w, &mut rng, "token_embd.weight", s.d_model, VOCAB_SIZE);
    mat(
        &mut w,
        &mut rng,
        "token_types.weight",
        s.d_model,
        s.n_token_types,
    );
    mat(&mut w, &mut rng, "position_embd.weight", s.d_model, s.ctx);
    norm(&mut w, &mut rng, "token_embd_norm.weight", s.d_model);
    bias(&mut w, &mut rng, "token_embd_norm.bias", s.d_model);

    for i in 0..s.n_layer {
        let p = |x: &str| format!("blk.{i}.{x}");
        for (name, cols, rows) in [
            ("attn_q", s.d_model, s.d_model),
            ("attn_k", s.d_model, s.d_model),
            ("attn_v", s.d_model, s.d_model),
            ("attn_output", s.d_model, s.d_model),
            ("ffn_up", s.d_model, s.n_ff),
            ("ffn_down", s.n_ff, s.d_model),
        ] {
            mat(&mut w, &mut rng, &p(&format!("{name}.weight")), cols, rows);
        }
        for (name, n) in [
            ("attn_q.bias", s.d_model),
            ("attn_k.bias", s.d_model),
            ("attn_v.bias", s.d_model),
            ("attn_output.bias", s.d_model),
            ("ffn_up.bias", s.n_ff),
            ("ffn_down.bias", s.d_model),
        ] {
            bias(&mut w, &mut rng, &p(name), n);
        }
        for norm_name in ["attn_output_norm", "layer_output_norm"] {
            norm(&mut w, &mut rng, &p(&format!("{norm_name}.weight")), s.d_model);
            bias(&mut w, &mut rng, &p(&format!("{norm_name}.bias")), s.d_model);
        }
    }

    let name = if s.unigram {
        format!("bert-unigram-{}.gguf", fmt.label())
    } else {
        format!("bert-{}.gguf", fmt.label())
    };
    let path = dir.join(name);
    w.write_to_file(&path).expect("write synthetic model");
    path
}

/// Write a `qwen35` hybrid model (Gated DeltaNet layers, one gated-attention layer in every
/// `interval`, Hadamard-rotated ternary matmuls) with every matmul in `fmt` (PQ2_0 or PTQ1_0).
///
/// Geometry: d_model 256, FFN 512, 4 layers (attention at layer 3), attention 4 heads × 64 with 2 KV
/// heads, GDN 2 key heads and 4 value heads of width 64, Hadamard block 64. Every width is a multiple of
/// 256, which the PTQ1_0 activation layout needs, and every matrix has a multiple of 16 rows, which the
/// AVX-512 tile path needs.
///
/// With `mtp`, one NextN block follows the four main layers (`nextn_predict_layers = 1`, counted in
/// `block_count`), in Q8_0 as the 4B Bonsai file stores it. The main layers are written first from the
/// same RNG stream, so `mtp = false` files stay byte-identical.
pub fn write_hybrid(dir: &Path, fmt: Format, seed: u64, mtp: bool) -> PathBuf {
    write_hybrid_impl(dir, fmt, seed, mtp, false, true, &[])
}

/// A plain Qwen 3.5 file: the same hybrid model, without the Hadamard rotation and with
/// ordinary quantised matmuls (Q8_0, F16) instead of the ternary ones.
///
/// This is what upstream's converter writes for `Qwen/Qwen3.5-9B` (llama.cpp v0.9.0,
/// `conversion/qwen.py`, `Qwen3_5TextModel`): no `prism.hadamard.*` metadata at all, the
/// value heads already in the tiled order, and the MRoPE sections the loader requires.
pub fn write_hybrid_plain(dir: &Path, fmt: Format, seed: u64) -> PathBuf {
    write_hybrid_impl(dir, fmt, seed, false, false, false, &[])
}

/// A plain Qwen 3.5 file carrying extra GGUF metadata, for a decision model whose prompt renderer
/// is checked with the real `systemone` template (S4.3c).
pub fn write_hybrid_decision(
    dir: &Path,
    fmt: Format,
    seed: u64,
    extra: &[(String, MetaValue)],
) -> PathBuf {
    write_hybrid_impl(dir, fmt, seed, false, false, false, extra)
}

/// A plain Qwen 3.5 file with one MTP (NextN) block, as upstream's converter writes it
/// when the checkpoint carries one (`mtp_num_hidden_layers`).
pub fn write_hybrid_plain_mtp(dir: &Path, fmt: Format, seed: u64) -> PathBuf {
    write_hybrid_impl(dir, fmt, seed, true, false, false, &[])
}

/// The MTP variant whose drafts are accepted: every output-head row is identical, so all logits tie and
/// both the main model and the MTP block's shared head argmax to token 0; a draft then always matches what
/// the main model sampled. Random weights otherwise (the ordinary MTP model rejects every draft).
pub fn write_hybrid_accepting(dir: &Path, fmt: Format, seed: u64) -> PathBuf {
    write_hybrid_impl(dir, fmt, seed, true, true, true, &[])
}

#[allow(clippy::too_many_arguments)]
fn write_hybrid_impl(
    dir: &Path,
    fmt: Format,
    seed: u64,
    mtp: bool,
    accept: bool,
    rotated: bool,
    extra: &[(String, MetaValue)],
) -> PathBuf {
    if rotated {
        assert!(matches!(fmt, Format::PQ2_0 | Format::PTQ1_0));
    }
    let (d_model, n_ff, n_layer, interval) = (256usize, 512usize, 4usize, 4usize);
    let (n_head, n_kv, head_dim) = (4usize, 2usize, 64usize);
    // A rotated file rotates two thirds of its head (the Bonsai 2 shape); a plain one
    // rotates all of it, so that the MRoPE sections below cover every rotary pair.
    let rope_dim = if rotated { 32 } else { head_dim };
    let (n_k, n_v, ds, d_conv) = (2usize, 4usize, 64usize, 4usize);
    let (key_dim, value_dim) = (n_k * ds, n_v * ds);
    let conv_dim = 2 * key_dim + value_dim;
    let q_dim = n_head * head_dim;
    let had_block = 64usize;

    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    w.str("general.architecture", "qwen35");
    w.str("general.name", &format!("synthetic-qwen35-{}", fmt.label()));
    let k = |x: &str| format!("qwen35.{x}");
    w.u32(&k("block_count"), (n_layer + usize::from(mtp)) as u32);
    if mtp {
        w.u32(&k("nextn_predict_layers"), 1);
    }
    w.u32(&k("embedding_length"), d_model as u32);
    w.u32(&k("attention.head_count"), n_head as u32);
    w.u32(&k("attention.head_count_kv"), n_kv as u32);
    w.u32(&k("attention.key_length"), head_dim as u32);
    w.u32(&k("attention.value_length"), head_dim as u32);
    w.u32(&k("feed_forward_length"), n_ff as u32);
    w.u32(&k("rope.dimension_count"), rope_dim as u32);
    w.u32(&k("full_attention_interval"), interval as u32);
    w.u32(&k("ssm.group_count"), n_k as u32);
    w.u32(&k("ssm.time_step_rank"), n_v as u32);
    w.u32(&k("ssm.state_size"), ds as u32);
    w.u32(&k("ssm.conv_kernel"), d_conv as u32);
    w.u32(&k("ssm.inner_size"), value_dim as u32);
    w.u32(&k("context_length"), 512);
    w.f32(&k("rope.freq_base"), 1e7);
    w.f32(&k("attention.layer_norm_rms_epsilon"), 1e-6);
    if rotated {
        w.u32("prism.hadamard.block_size", had_block as u32);
        w.str(
            "prism.hadamard.transform",
            "normalized-sylvester-walsh-hadamard",
        );
        w.str("prism.hadamard.sign_mode", "explicit");
        // One sign vector per distinct matmul input width: d_model = value = query = 256, FFN = 512.
        let widths = [256i32, 512];
        let signs: Vec<i32> = (0..widths.iter().sum::<i32>())
            .map(|_| if rng.next_u32() & 1 == 0 { 1 } else { -1 })
            .collect();
        w.arr_i32("prism.hadamard.sign_widths", &widths);
        w.arr_i32("prism.hadamard.sign_values", &signs);
    } else {
        // The MRoPE sections a plain Qwen 3.5 file carries: llama.cpp's `qwen35` loader
        // requires them, and its converter writes the default when a checkpoint omits them
        // (conversion/qwen.py, `_Qwen35MRopeMixin`). The sum is `rope_dim / 2`, the number
        // of rotary pairs.
        w.arr_i32(&k("rope.dimension_sections"), &[11, 11, 10, 0]);
    }
    for (key, value) in extra {
        w.add_meta(key, value.clone());
    }
    vocab::add_tokenizer_metadata(&mut w, "qwen35");
    w.str(
        "tokenizer.chat_template",
        vocab::chat_template(TemplateKind::Qwen3),
    );

    let mat = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64],
            fmt,
            generate_weights(fmt, cols * rows, rng),
        );
    };
    let f32m = |w: &mut GgufWriter, rng: &mut Rng, name: &str, dims: &[u64], lo: f32, hi: f32| {
        let n: u64 = dims.iter().product();
        let data: Vec<u8> = (0..n)
            .flat_map(|_| rng.f32_range(lo, hi).to_le_bytes())
            .collect();
        w.add_tensor(name, dims, Format::F32, data);
    };
    let norm = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        w.add_tensor(name, &[n as u64], Format::F32, generate_norm(n, rng));
    };
    w.add_tensor(
        "token_embd.weight",
        &[d_model as u64, VOCAB_SIZE as u64],
        Format::F16,
        generate_weights(Format::F16, d_model * VOCAB_SIZE, &mut rng),
    );
    for i in 0..n_layer {
        let p = |s: &str| format!("blk.{i}.{s}");
        norm(&mut w, &mut rng, &p("attn_norm.weight"), d_model);
        if (i + 1) % interval == 0 {
            // Gated attention: q and its gate interleaved per head, so attn_q has 2 × q_dim rows.
            mat(&mut w, &mut rng, &p("attn_q.weight"), d_model, 2 * q_dim);
            mat(
                &mut w,
                &mut rng,
                &p("attn_k.weight"),
                d_model,
                n_kv * head_dim,
            );
            mat(
                &mut w,
                &mut rng,
                &p("attn_v.weight"),
                d_model,
                n_kv * head_dim,
            );
            norm(&mut w, &mut rng, &p("attn_q_norm.weight"), head_dim);
            norm(&mut w, &mut rng, &p("attn_k_norm.weight"), head_dim);
            mat(&mut w, &mut rng, &p("attn_output.weight"), q_dim, d_model);
        } else {
            mat(&mut w, &mut rng, &p("attn_qkv.weight"), d_model, conv_dim);
            mat(&mut w, &mut rng, &p("attn_gate.weight"), d_model, value_dim);
            f32m(
                &mut w,
                &mut rng,
                &p("ssm_beta.weight"),
                &[d_model as u64, n_v as u64],
                -0.05,
                0.05,
            );
            f32m(
                &mut w,
                &mut rng,
                &p("ssm_alpha.weight"),
                &[d_model as u64, n_v as u64],
                -0.05,
                0.05,
            );
            f32m(
                &mut w,
                &mut rng,
                &p("ssm_conv1d.weight"),
                &[d_conv as u64, conv_dim as u64],
                -0.3,
                0.3,
            );
            f32m(
                &mut w,
                &mut rng,
                &p("ssm_dt.bias"),
                &[n_v as u64],
                -0.5,
                0.5,
            );
            f32m(&mut w, &mut rng, &p("ssm_a"), &[n_v as u64], -1.0, -0.1);
            norm(&mut w, &mut rng, &p("ssm_norm.weight"), ds);
            mat(&mut w, &mut rng, &p("ssm_out.weight"), value_dim, d_model);
        }
        norm(&mut w, &mut rng, &p("post_attention_norm.weight"), d_model);
        mat(&mut w, &mut rng, &p("ffn_gate.weight"), d_model, n_ff);
        mat(&mut w, &mut rng, &p("ffn_up.weight"), d_model, n_ff);
        mat(&mut w, &mut rng, &p("ffn_down.weight"), n_ff, d_model);
    }
    norm(&mut w, &mut rng, "output_norm.weight", d_model);
    if accept {
        // Every output row is the same ternary row, so every logit is the same value and
        // argmax picks row 0 whatever the hidden state: both heads propose token 0, and a
        // draft always matches what the main model sampled.
        let row = generate_weights(fmt, d_model, &mut rng);
        let mut data = Vec::with_capacity(row.len() * VOCAB_SIZE);
        for _ in 0..VOCAB_SIZE {
            data.extend_from_slice(&row);
        }
        w.add_tensor(
            "output.weight",
            &[d_model as u64, VOCAB_SIZE as u64],
            fmt,
            data,
        );
    } else {
        mat(&mut w, &mut rng, "output.weight", d_model, VOCAB_SIZE);
    }
    if mtp {
        // One Qwen3.5 full-attention NextN block behind the input projection `[enorm(e) | hnorm(h)]`.
        // Its head is the model's own output head, reached through `shared_head_norm`.
        let il = n_layer;
        let p = |s: &str| format!("blk.{il}.{s}");
        let q8 = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
            w.add_tensor(
                name,
                &[cols as u64, rows as u64],
                Format::Q8_0,
                generate_weights(Format::Q8_0, cols * rows, rng),
            );
        };
        q8(
            &mut w,
            &mut rng,
            &p("nextn.eh_proj.weight"),
            2 * d_model,
            d_model,
        );
        norm(&mut w, &mut rng, &p("nextn.enorm.weight"), d_model);
        norm(&mut w, &mut rng, &p("nextn.hnorm.weight"), d_model);
        norm(&mut w, &mut rng, &p("attn_norm.weight"), d_model);
        q8(&mut w, &mut rng, &p("attn_q.weight"), d_model, 2 * q_dim);
        q8(
            &mut w,
            &mut rng,
            &p("attn_k.weight"),
            d_model,
            n_kv * head_dim,
        );
        q8(
            &mut w,
            &mut rng,
            &p("attn_v.weight"),
            d_model,
            n_kv * head_dim,
        );
        norm(&mut w, &mut rng, &p("attn_q_norm.weight"), head_dim);
        norm(&mut w, &mut rng, &p("attn_k_norm.weight"), head_dim);
        q8(&mut w, &mut rng, &p("attn_output.weight"), q_dim, d_model);
        norm(&mut w, &mut rng, &p("post_attention_norm.weight"), d_model);
        q8(&mut w, &mut rng, &p("ffn_gate.weight"), d_model, n_ff);
        q8(&mut w, &mut rng, &p("ffn_up.weight"), d_model, n_ff);
        q8(&mut w, &mut rng, &p("ffn_down.weight"), n_ff, d_model);
        norm(
            &mut w,
            &mut rng,
            &p("nextn.shared_head_norm.weight"),
            d_model,
        );
    }
    let name = if accept {
        format!("qwen35-{}-mtp-accept", fmt.label())
    } else if rotated {
        format!("qwen35-{}", fmt.label())
    } else {
        format!("qwen35-plain-{}", fmt.label())
    };
    let path = dir.join(format!("{name}.gguf"));
    w.write_to_file(&path).expect("write synthetic model");
    path
}

/// A `qwen3moe` model's shape: ordinary Qwen3 attention, but the FFN is a bank of experts.
#[derive(Clone, Copy, Debug)]
pub struct MoeShape {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// The *dense* FFN width the file also carries. llama.cpp reads it, but the experts ignore it
    /// (`expert_feed_forward_length` is what the experts use).
    pub n_ff: usize,
    /// How many experts the router chooses from.
    pub n_expert: usize,
    /// How many of them one token uses.
    pub n_expert_used: usize,
    /// `expert_feed_forward_length`: the width of one expert's FFN.
    pub expert_ff: usize,
    pub ctx: usize,
}

impl Default for MoeShape {
    fn default() -> Self {
        // Every matrix row is a multiple of 256 (the K-quant block), heads are 32 wide, and the
        // experts are 128 wide — a multiple of the 32-weight block of Q8_0.
        MoeShape {
            d_model: 256,
            n_layer: 2,
            n_head: 8,
            n_head_kv: 4,
            n_ff: 512,
            n_expert: 8,
            n_expert_used: 2,
            expert_ff: 128,
            ctx: 512,
        }
    }
}

/// Write a `qwen3moe` file — a mixture-of-experts model as llama.cpp v0.9.0 expects it
/// (`src/models/qwen3moe.cpp`, `src/llama-arch.cpp`).
///
/// The attention block is Qwen3's (per-head q/k RMSNorm, GQA); the FFN is a bank of experts:
///
/// | tensor | GGUF dims | what |
/// |---|---|---|
/// | `blk.i.ffn_gate_inp.weight` | `[d_model, n_expert]` | the router |
/// | `blk.i.ffn_gate_exps.weight` | `[d_model, expert_ff, n_expert]` | per expert, `d_model → expert_ff` |
/// | `blk.i.ffn_up_exps.weight` | `[d_model, expert_ff, n_expert]` | per expert, `d_model → expert_ff` |
/// | `blk.i.ffn_down_exps.weight` | `[expert_ff, d_model, n_expert]` | per expert, `expert_ff → d_model` |
///
/// The expert tensors are three-dimensional, and GGUF stores `dims[0]` contiguous, so the layout
/// is expert-major: expert `e` is a contiguous run of `dims[1]` rows of `dims[0]` weights — a
/// 2D `[in, out]` matrix in the layout the matmul wants, with no new kernel.
///
/// Every matrix is in `fmt`; the RMSNorm weights are F32, as in the dense models.
pub fn write_moe(dir: &Path, fmt: Format, seed: u64, s: MoeShape) -> PathBuf {
    // ggml multiplies a quantized matrix by an activation along `ne[0]`, so that row must be a
    // whole number of blocks. The default shape satisfies this for F16 and Q8_0; a K-quant would
    // need `d_model` and `expert_ff` to be multiples of 256.
    assert!(
        s.d_model.is_multiple_of(fmt.block()) && s.expert_ff.is_multiple_of(fmt.block()),
        "qwen3moe rows must be a multiple of the {} block {}",
        fmt.label(),
        fmt.block()
    );
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    let head_dim = s.d_model / s.n_head;
    w.str("general.architecture", "qwen3moe");
    w.str(
        "general.name",
        &format!("synthetic-qwen3moe-{}", fmt.label()),
    );
    let k = |x: &str| format!("qwen3moe.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("attention.head_count_kv"), s.n_head_kv as u32);
    w.u32(&k("attention.key_length"), head_dim as u32);
    w.u32(&k("attention.value_length"), head_dim as u32);
    // The dense FFN width: llama.cpp reads it, and the experts ignore it.
    w.u32(&k("feed_forward_length"), s.n_ff as u32);
    // The experts' own width, the router's bank and how many of them a token uses.
    w.u32(&k("expert_feed_forward_length"), s.expert_ff as u32);
    w.u32(&k("expert_count"), s.n_expert as u32);
    w.u32(&k("expert_used_count"), s.n_expert_used as u32);
    w.u32(&k("context_length"), s.ctx as u32);
    w.u32(&k("vocab_size"), VOCAB_SIZE as u32);
    w.u32(&k("rope.dimension_count"), head_dim as u32);
    w.f32(&k("rope.freq_base"), 1e6);
    w.f32(&k("attention.layer_norm_rms_epsilon"), 1e-6);
    // llama.cpp never reads this key for qwen3moe — its own value stays 0, which means "no
    // scaling" (`llama-graph.cpp`: a scale of 0 or 1 is not applied).
    w.f32(&k("expert_weights_scale"), 0.0);
    vocab::add_tokenizer_metadata(&mut w, "qwen2");
    w.str(
        "tokenizer.chat_template",
        vocab::chat_template(TemplateKind::Qwen3),
    );

    let kv = s.n_head_kv * head_dim;
    let mat = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64],
            fmt,
            generate_weights(fmt, cols * rows, rng),
        );
    };
    // One expert's weights: `cols` weights per row, `rows` rows, `n_expert` of them, the last
    // dimension running slowest — llama.cpp's own order.
    let exps = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64, s.n_expert as u64],
            fmt,
            generate_weights(fmt, cols * rows * s.n_expert, rng),
        );
    };
    let norm = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        w.add_tensor(name, &[n as u64], Format::F32, generate_norm(n, rng));
    };
    mat(&mut w, &mut rng, "token_embd.weight", s.d_model, VOCAB_SIZE);
    for i in 0..s.n_layer {
        let p = |t: &str| format!("blk.{i}.{t}");
        norm(&mut w, &mut rng, &p("attn_norm.weight"), s.d_model);
        mat(&mut w, &mut rng, &p("attn_q.weight"), s.d_model, s.d_model);
        mat(&mut w, &mut rng, &p("attn_k.weight"), s.d_model, kv);
        mat(&mut w, &mut rng, &p("attn_v.weight"), s.d_model, kv);
        norm(&mut w, &mut rng, &p("attn_q_norm.weight"), head_dim);
        norm(&mut w, &mut rng, &p("attn_k_norm.weight"), head_dim);
        mat(
            &mut w,
            &mut rng,
            &p("attn_output.weight"),
            s.d_model,
            s.d_model,
        );
        norm(&mut w, &mut rng, &p("ffn_norm.weight"), s.d_model);
        // The router: one row of `n_expert` weights per input.
        mat(
            &mut w,
            &mut rng,
            &p("ffn_gate_inp.weight"),
            s.d_model,
            s.n_expert,
        );
        // The experts themselves.
        exps(
            &mut w,
            &mut rng,
            &p("ffn_gate_exps.weight"),
            s.d_model,
            s.expert_ff,
        );
        exps(
            &mut w,
            &mut rng,
            &p("ffn_up_exps.weight"),
            s.d_model,
            s.expert_ff,
        );
        exps(
            &mut w,
            &mut rng,
            &p("ffn_down_exps.weight"),
            s.expert_ff,
            s.d_model,
        );
    }
    norm(&mut w, &mut rng, "output_norm.weight", s.d_model);
    mat(&mut w, &mut rng, "output.weight", s.d_model, VOCAB_SIZE);
    let path = dir.join(format!("qwen3moe-{}.gguf", fmt.label()));
    w.write_to_file(&path).expect("write synthetic model");
    path
}

/// A `gpt-oss` model's shape: alternating sliding-window and full attention, attention
/// sinks, YaRN rope scaling, and a bank of experts with biases.
#[derive(Clone, Copy, Debug)]
pub struct GptOssShape {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// How many experts the router chooses from.
    pub n_expert: usize,
    /// How many of them one token uses.
    pub n_expert_used: usize,
    /// `expert_feed_forward_length`: the width of one expert's feed-forward.
    pub expert_ff: usize,
    /// The sliding window (`attention.sliding_window`).
    pub sliding_window: usize,
    pub ctx: usize,
}

impl Default for GptOssShape {
    fn default() -> Self {
        // Every matrix row is a multiple of 32 (the activation block and MXFP4's
        // block), the heads are 32 wide, and the experts are 64 wide.
        GptOssShape {
            d_model: 256,
            n_layer: 4,
            n_head: 8,
            n_head_kv: 4,
            n_expert: 4,
            n_expert_used: 2,
            expert_ff: 64,
            sliding_window: 16,
            ctx: 512,
        }
    }
}

/// Write a synthetic `gpt-oss` file, the way llama.cpp v0.9.0 expects it
/// (`src/models/openai-moe.cpp`, `conversion/gpt_oss.py`).
///
/// The experts are in `fmt` (F16 or MXFP4, as the published files are) and the
/// attention in the same format — except for MXFP4, where the published files keep
/// the attention in f16 and quantise only the experts. The biases and the norms are
/// F32, as in every file.
///
/// | tensor | GGUF dims | what |
/// |---|---|---|
/// | `blk.i.attn_norm.weight` | `[d_model]` | the attention's norm |
/// | `blk.i.attn_q.weight` | `[d_model, n_head·head_dim]` | the queries |
/// | `blk.i.attn_k.weight` | `[d_model, n_kv_head·head_dim]` | the keys |
/// | `blk.i.attn_v.weight` | `[d_model, n_kv_head·head_dim]` | the values |
/// | `blk.i.attn_output.weight` | `[n_head·head_dim, d_model]` | the attention's output |
/// | `blk.i.attn_output.bias` | `[d_model]` | its bias |
/// | `blk.i.attn_sinks.weight` | `[n_head]` | the attention sinks |
/// | `blk.i.post_attention_norm.weight` | `[d_model]` | the norm the experts see |
/// | `blk.i.ffn_gate_inp.weight` | `[d_model, n_expert]` | the router |
/// | `blk.i.ffn_gate_inp.bias` | `[n_expert]` | the router's bias |
/// | `blk.i.ffn_gate_exps.weight` | `[d_model, expert_ff, n_expert]` | the experts' gates |
/// | `blk.i.ffn_gate_exps.bias` | `[expert_ff, n_expert]` | their biases |
/// | `blk.i.ffn_up_exps.weight` | `[d_model, expert_ff, n_expert]` | the experts' ups |
/// | `blk.i.ffn_up_exps.bias` | `[expert_ff, n_expert]` | their biases |
/// | `blk.i.ffn_down_exps.weight` | `[expert_ff, d_model, n_expert]` | the experts' downs |
/// | `blk.i.ffn_down_exps.bias` | `[d_model, n_expert]` | their biases |
pub fn write_gpt_oss(dir: &Path, fmt: Format, seed: u64, s: GptOssShape) -> PathBuf {
    write_gpt_oss_with_template(dir, fmt, seed, s, None)
}

/// The same file, optionally declaring a `tokenizer.chat_template`. The template
/// test writes gpt-oss's own Jinja template into it, so detection runs on a file
/// the way a real gpt-oss file does.
pub fn write_gpt_oss_with_template(
    dir: &Path,
    fmt: Format,
    seed: u64,
    s: GptOssShape,
    chat_template: Option<&str>,
) -> PathBuf {
    assert!(
        s.d_model.is_multiple_of(fmt.block()) && s.expert_ff.is_multiple_of(fmt.block()),
        "gpt-oss rows must be a multiple of the {} block {}",
        fmt.label(),
        fmt.block()
    );
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    let head_dim = s.d_model / s.n_head;
    w.str("general.architecture", "gpt-oss");
    w.str(
        "general.name",
        &format!("synthetic-gpt-oss-{}", fmt.label()),
    );
    let k = |x: &str| format!("gpt-oss.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("attention.head_count_kv"), s.n_head_kv as u32);
    w.u32(&k("attention.key_length"), head_dim as u32);
    w.u32(&k("attention.value_length"), head_dim as u32);
    w.u32(&k("attention.sliding_window"), s.sliding_window as u32);
    w.f32(&k("attention.layer_norm_rms_epsilon"), 1e-5);
    w.u32(&k("rope.dimension_count"), head_dim as u32);
    // The rotary base of gpt-oss's files: 150000.
    w.f32(&k("rope.freq_base"), 150_000.0);
    // YaRN: the factor and the context it was trained for.
    w.str(&k("rope.scaling.type"), "yarn");
    w.f32(&k("rope.scaling.factor"), 32.0);
    w.u32(&k("rope.scaling.original_context_length"), 4096);
    w.u32(&k("expert_count"), s.n_expert as u32);
    w.u32(&k("expert_used_count"), s.n_expert_used as u32);
    w.u32(&k("expert_feed_forward_length"), s.expert_ff as u32);
    // llama.cpp never applies this scale for gpt-oss: 0 means "no scaling".
    w.f32(&k("expert_weights_scale"), 0.0);
    w.u32(&k("context_length"), s.ctx as u32);
    w.u32(&k("vocab_size"), VOCAB_SIZE as u32);
    // The `o200k_harmony` pre-tokenizer.
    if chat_template.is_some() {
        // The template brings the Harmony control tokens with it; a check of the
        // declared template needs them in the vocabulary.
        vocab::add_gpt_oss_tokenizer_metadata(&mut w);
    } else {
        vocab::add_tokenizer_metadata(&mut w, "gpt-4o");
    }
    if let Some(t) = chat_template {
        w.str("tokenizer.chat_template", t);
    }

    let kv = s.n_head_kv * head_dim;
    let qd = s.n_head * head_dim;
    let mat = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64],
            fmt,
            generate_weights(fmt, cols * rows, rng),
        );
    };
    // The attention's matrices are in the file's format, except for MXFP4: the
    // published gpt-oss files keep the attention in f16 and quantise only the
    // experts.
    let attn_fmt = if fmt == Format::MXFP4 {
        Format::F16
    } else {
        fmt
    };
    let mat16 = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64],
            attn_fmt,
            generate_weights(attn_fmt, cols * rows, rng),
        );
    };
    let norm = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        w.add_tensor(name, &[n as u64], Format::F32, generate_norm(n, rng));
    };
    // A bias: F32, one value per unit, the rows running slowest.
    let bias = |w: &mut GgufWriter, rng: &mut Rng, name: &str, dims: &[u64]| {
        let n: u64 = dims.iter().product();
        let data: Vec<u8> = (0..n)
            .flat_map(|_| rng.f32_range(-0.05, 0.05).to_le_bytes())
            .collect();
        w.add_tensor(name, dims, Format::F32, data);
    };
    // One expert's weights: `cols` weights per row, `rows` rows, `n_expert` of them,
    // the last dimension running slowest — llama.cpp's own order.
    let exps = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64, s.n_expert as u64],
            fmt,
            generate_weights(fmt, cols * rows * s.n_expert, rng),
        );
    };

    mat16(&mut w, &mut rng, "token_embd.weight", s.d_model, VOCAB_SIZE);
    for i in 0..s.n_layer {
        let p = |t: &str| format!("blk.{i}.{t}");
        norm(&mut w, &mut rng, &p("attn_norm.weight"), s.d_model);
        mat16(&mut w, &mut rng, &p("attn_q.weight"), s.d_model, qd);
        mat16(&mut w, &mut rng, &p("attn_k.weight"), s.d_model, kv);
        mat16(&mut w, &mut rng, &p("attn_v.weight"), s.d_model, kv);
        mat16(&mut w, &mut rng, &p("attn_output.weight"), qd, s.d_model);
        bias(
            &mut w,
            &mut rng,
            &p("attn_output.bias"),
            &[s.d_model as u64],
        );
        // The attention sinks: one learned logit per query head.
        bias(
            &mut w,
            &mut rng,
            &p("attn_sinks.weight"),
            &[s.n_head as u64],
        );
        norm(
            &mut w,
            &mut rng,
            &p("post_attention_norm.weight"),
            s.d_model,
        );
        // The router.
        mat(
            &mut w,
            &mut rng,
            &p("ffn_gate_inp.weight"),
            s.d_model,
            s.n_expert,
        );
        bias(
            &mut w,
            &mut rng,
            &p("ffn_gate_inp.bias"),
            &[s.n_expert as u64],
        );
        // The experts.
        exps(
            &mut w,
            &mut rng,
            &p("ffn_gate_exps.weight"),
            s.d_model,
            s.expert_ff,
        );
        bias(
            &mut w,
            &mut rng,
            &p("ffn_gate_exps.bias"),
            &[s.expert_ff as u64, s.n_expert as u64],
        );
        exps(
            &mut w,
            &mut rng,
            &p("ffn_up_exps.weight"),
            s.d_model,
            s.expert_ff,
        );
        bias(
            &mut w,
            &mut rng,
            &p("ffn_up_exps.bias"),
            &[s.expert_ff as u64, s.n_expert as u64],
        );
        exps(
            &mut w,
            &mut rng,
            &p("ffn_down_exps.weight"),
            s.expert_ff,
            s.d_model,
        );
        bias(
            &mut w,
            &mut rng,
            &p("ffn_down_exps.bias"),
            &[s.d_model as u64, s.n_expert as u64],
        );
    }
    norm(&mut w, &mut rng, "output_norm.weight", s.d_model);
    mat16(&mut w, &mut rng, "output.weight", s.d_model, VOCAB_SIZE);
    let path = dir.join(format!("gpt-oss-{}.gguf", fmt.label()));
    w.write_to_file(&path).expect("write synthetic model");
    path
}

/// Write a synthetic `phi3` model (fused `attn_qkv`, fused `ffn_up` with SwiGLU, RMSNorm,
/// NeoX rotary pairing, no biases) whose matrices are all `fmt`, seeded by `seed`.
pub fn write_phi3(dir: &Path, fmt: Format, seed: u64, s: Dense) -> PathBuf {
    write_phi3_named(dir, fmt, seed, s, None)
}

/// Write a synthetic `phi3` model with optional custom file stem / general.name.
pub fn write_phi3_named(
    dir: &Path,
    fmt: Format,
    seed: u64,
    s: Dense,
    custom_name: Option<&str>,
) -> PathBuf {
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    let head_dim = s.d_model / s.n_head;
    w.str("general.architecture", "phi3");
    let name = custom_name
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("synthetic-phi3-{}", fmt.label()));
    w.str("general.name", &name);
    let k = |x: &str| format!("phi3.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("attention.head_count_kv"), s.n_head_kv as u32);
    w.u32(&k("feed_forward_length"), s.n_ff as u32);
    w.u32(&k("context_length"), s.ctx as u32);
    w.u32(&k("vocab_size"), VOCAB_SIZE as u32);
    w.u32(&k("attention.key_length"), head_dim as u32);
    w.u32(&k("attention.value_length"), head_dim as u32);
    w.u32(&k("rope.dimension_count"), head_dim as u32);
    w.f32(&k("rope.freq_base"), 10000.0);
    w.f32(&k("attention.layer_norm_rms_epsilon"), 1e-6);
    w.u32(&k("attention.sliding_window"), 0);

    vocab::add_tokenizer_metadata(&mut w, "llama-bpe");
    w.str(
        "tokenizer.chat_template",
        vocab::chat_template(TemplateKind::ChatMl),
    );

    let kv = s.n_head_kv * head_dim;
    let qkv = s.d_model + 2 * kv;
    let mat = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
        w.add_tensor(
            name,
            &[cols as u64, rows as u64],
            fmt,
            generate_weights(fmt, cols * rows, rng),
        );
    };
    let norm = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        w.add_tensor(name, &[n as u64], Format::F32, generate_norm(n, rng));
    };

    mat(&mut w, &mut rng, "token_embd.weight", s.d_model, VOCAB_SIZE);
    for i in 0..s.n_layer {
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_norm.weight"),
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_qkv.weight"),
            s.d_model,
            qkv,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_output.weight"),
            s.d_model,
            s.d_model,
        );
        norm(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_norm.weight"),
            s.d_model,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_up.weight"),
            s.d_model,
            2 * s.n_ff,
        );
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.ffn_down.weight"),
            s.n_ff,
            s.d_model,
        );
    }
    norm(&mut w, &mut rng, "output_norm.weight", s.d_model);
    mat(&mut w, &mut rng, "output.weight", s.d_model, VOCAB_SIZE);

    let file_name = format!("{name}.gguf");
    let path = dir.join(file_name);
    w.write_to_file(&path).expect("write synthetic phi3 model");
    path
}
