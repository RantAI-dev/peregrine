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
    write_dense_impl(dir, arch, fmt, seed, s, Rope::None, false, None, &[], None)
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
        None,
    )
}

/// The dense model whose drafts are accepted: every output-head row is identical, so all logits tie and
/// the argmax picks row 0 whatever the hidden state — the model always answers token 0. A corpus of
/// zeros is what the suffix index needs to find a match in (the same n-gram twice), and what makes the
/// draft it proposes the token the target draws.
pub fn write_dense_accepting(dir: &Path, arch: &str, fmt: Format, seed: u64, s: Dense) -> PathBuf {
    write_dense_impl(dir, arch, fmt, seed, s, Rope::None, true, None, &[], None)
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
    write_dense_impl(
        dir,
        arch,
        fmt,
        seed,
        s,
        Rope::None,
        false,
        None,
        extra,
        None,
    )
}

/// A dense `arch` file with a chosen `tokenizer.chat_template`. The weights and the
/// vocabulary are [`write_dense`]'s; only the template string differs, which is how a
/// synthetic file can carry a tool format's template.
pub fn write_dense_with_template(
    dir: &Path,
    arch: &str,
    fmt: Format,
    seed: u64,
    s: Dense,
    template: &str,
) -> PathBuf {
    write_dense_impl(
        dir,
        arch,
        fmt,
        seed,
        s,
        Rope::None,
        false,
        None,
        &[],
        Some(template),
    )
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
        None,
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
    chat_template: Option<&str>,
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
    w.str(
        "tokenizer.chat_template",
        chat_template.unwrap_or_else(|| vocab::chat_template(tpl)),
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

/// The classifier head of a synthetic `bert` file (roadmap S4.6b).
#[derive(Clone, Debug)]
pub struct BertHeadConfig {
    /// `cls.output`'s rows: 1 is a reranker's raw score, more is a classifier.
    pub n_out: usize,
    /// `classifier.output_labels`; empty writes no key (a reranker).
    pub labels: Vec<String>,
    /// `classifier.activation`; `None` writes no key, which llama.cpp reads as `tanh`.
    pub activation: Option<&'static str>,
    /// Write `cls.weight`/`cls.bias` (the dense layer). `false` writes only `cls.output`,
    /// the one-layer head (jina-reranker-tiny's shape).
    pub dense: bool,
    /// `classifier.pooling_type`: 1 writes mean, anything else names row 0.
    pub pooling_type: Option<u32>,
}

impl Default for BertHeadConfig {
    fn default() -> Self {
        BertHeadConfig {
            n_out: 1,
            labels: Vec::new(),
            activation: None,
            dense: true,
            pooling_type: None,
        }
    }
}

/// The shape of a synthetic `bert` encoder (roadmap S4.6a).
#[derive(Clone, Debug)]
pub struct BertConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub ctx: usize,
    /// `bert.pooling_type`: 2 is cls (BGE's default), 1 is mean (E5/GTE-style), 4 rank.
    pub pooling_type: u32,
    /// Whether the file carries the WordPiece (`bert`) or the unigram (`t5`, XLM-R)
    /// tokenizer. The weights are the same either way.
    pub unigram: bool,
    /// Token-type rows in `token_types.weight` (BERT's Sentence A/B is 2).
    pub n_token_types: usize,
    /// The classifier head, when the file has one (S4.6b).
    pub head: Option<BertHeadConfig>,
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
            head: None,
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
            norm(
                &mut w,
                &mut rng,
                &p(&format!("{norm_name}.weight")),
                s.d_model,
            );
            bias(
                &mut w,
                &mut rng,
                &p(&format!("{norm_name}.bias")),
                s.d_model,
            );
        }
    }

    if let Some(h) = &s.head {
        if h.dense {
            mat(&mut w, &mut rng, "cls.weight", s.d_model, s.d_model);
            bias(&mut w, &mut rng, "cls.bias", s.d_model);
        }
        if let Some(a) = h.activation {
            w.str(&k("classifier.activation"), a);
        }
        if let Some(p) = h.pooling_type {
            w.u32(&k("classifier.pooling_type"), p);
        }
        if !h.labels.is_empty() {
            w.arr_str(&k("classifier.output_labels"), &h.labels);
        }
        // llama.cpp's converter writes a one-row output weight 1-d, as it does for
        // `token_types.weight`; the loader accepts the missing trailing 1.
        let dims: Vec<u64> = if h.n_out == 1 {
            vec![s.d_model as u64]
        } else {
            vec![s.d_model as u64, h.n_out as u64]
        };
        w.add_tensor(
            "cls.output.weight",
            &dims,
            fmt,
            generate_weights(fmt, s.d_model * h.n_out, &mut rng),
        );
        bias(&mut w, &mut rng, "cls.output.bias", h.n_out);
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

/// The shape of a synthetic `modern-bert` encoder (roadmap S4.7).
///
/// The defaults have four layers with the converter's pattern 3: layer 0 and 3
/// are global and layers 1 and 2 see a symmetric window of 8 (`|i - j| <= 4`),
/// small enough that a 37-token test text is cut into several bands.
#[derive(Clone, Debug)]
pub struct ModernBertConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub ctx: usize,
    /// `attention.sliding_window`; 0 writes no window (every layer global).
    pub window: u32,
    /// `attention.sliding_window_pattern`; every `pattern`-th layer is global.
    pub pattern: u32,
    /// `rope.freq_base` (global layers) and `rope.freq_base_swa` (local layers).
    /// The two differ so a layer that reads the wrong table cannot pass.
    pub rope_base: f32,
    pub rope_base_local: f32,
    /// `modern-bert.hidden_activation`: `"gelu"` (the default, exact erf) or
    /// `"silu"` (Granite Embedding R2's shape).
    pub hidden_act: &'static str,
    /// `modern-bert.pooling_type`: 1 is mean, 2 is cls.
    pub pooling_type: u32,
    /// The classifier head (`cls`, `cls.norm`, `cls.output`,
    /// `classifier.output_labels`), when the file carries one (S4.7b).
    pub head: Option<ModernBertHeadConfig>,
}

/// The classifier head of a synthetic `modern-bert` file (roadmap S4.7b).
#[derive(Clone, Debug)]
pub struct ModernBertHeadConfig {
    /// `cls.output`'s rows: 1 is a reranker's raw score, more is a classifier.
    pub n_out: usize,
    /// `classifier.output_labels`; empty writes no key (a reranker).
    pub labels: Vec<String>,
    /// `classifier.activation`; `None` writes no key, which llama.cpp reads as the
    /// exact (erf) GELU for this family.
    pub activation: Option<&'static str>,
    /// Write `cls.weight`, the dense layer.
    pub dense: bool,
    /// Write `cls.norm.weight`, the head's LayerNorm.
    pub norm: bool,
    /// Write `cls.output.weight`/`.bias`, the projection to the file's outputs.
    pub out: bool,
    /// `classifier.pooling_type`: `None` writes no key (mean), 2 writes cls (row 0).
    pub pooling_type: Option<u32>,
    /// Write `cls.bias` and `cls.norm.bias`, which llama.cpp does **not** create for
    /// this family (`src/models/modern-bert.cpp:90-93`): a file that carries them is
    /// read without them.
    pub extra_biases: bool,
}

impl Default for ModernBertHeadConfig {
    fn default() -> Self {
        ModernBertHeadConfig {
            n_out: 1,
            labels: Vec::new(),
            activation: None,
            dense: true,
            norm: true,
            out: true,
            pooling_type: None,
            extra_biases: false,
        }
    }
}

impl Default for ModernBertConfig {
    fn default() -> Self {
        ModernBertConfig {
            d_model: 128,
            n_layer: 4,
            n_head: 4,
            n_ff: 192,
            ctx: 64,
            window: 8,
            pattern: 3,
            rope_base: 100_000.0,
            rope_base_local: 10_000.0,
            hidden_act: "gelu",
            pooling_type: 1,
            head: None,
        }
    }
}

/// Write a synthetic `modern-bert` encoder: pre-norm blocks whose first
/// attention norm is the identity, a fused `attn_qkv`, RoPE with one base for the
/// local layers and one for the global ones, the symmetric window, the gated
/// `ffn_up` (the first half activated) and LayerNorms with a weight and no bias —
/// the graph `src/models/modern-bert.cpp` builds.
///
/// Every matrix is `fmt` (F32 or Q8_0, both multiples of the block size); the
/// norms are F32, as the converter writes them.
pub fn write_modern_bert(dir: &Path, fmt: Format, seed: u64, s: ModernBertConfig) -> PathBuf {
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    let arch = "modern-bert";
    w.str("general.architecture", arch);
    w.str(
        "general.name",
        &format!("synthetic-modern-bert-{}", fmt.label()),
    );
    let k = |x: &str| format!("{arch}.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("feed_forward_length"), s.n_ff as u32);
    w.u32(&k("context_length"), s.ctx as u32);
    w.u32(&k("vocab_size"), VOCAB_SIZE as u32);
    w.f32(&k("attention.layer_norm_epsilon"), 1e-5);
    w.f32(&k("rope.freq_base"), s.rope_base);
    w.f32(&k("rope.freq_base_swa"), s.rope_base_local);
    w.u32(&k("rope.dimension_count"), (s.d_model / s.n_head) as u32);
    w.str(&k("hidden_activation"), s.hidden_act);
    w.u32(&k("pooling_type"), s.pooling_type);
    if s.window > 0 {
        w.u32(&k("attention.sliding_window"), s.window);
        w.u32(&k("attention.sliding_window_pattern"), s.pattern);
    }
    vocab::add_modern_bert_tokenizer_metadata(&mut w);

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
    let bias = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            let v = rng.f32_range(-0.05, 0.05);
            out.extend_from_slice(&v.to_le_bytes());
        }
        w.add_tensor(name, &[n as u64], Format::F32, out);
    };

    mat(&mut w, &mut rng, "token_embd.weight", s.d_model, VOCAB_SIZE);
    norm(&mut w, &mut rng, "token_embd_norm.weight", s.d_model);
    norm(&mut w, &mut rng, "output_norm.weight", s.d_model);

    for i in 0..s.n_layer {
        let p = |x: &str| format!("blk.{i}.{x}");
        // Layer 0's attention norm is the identity (llama.cpp's
        // `TENSOR_NOT_REQUIRED`), so it is not written.
        if i > 0 {
            norm(&mut w, &mut rng, &p("attn_norm.weight"), s.d_model);
        }
        for (name, cols, rows) in [
            ("attn_qkv", s.d_model, 3 * s.d_model),
            ("attn_output", s.d_model, s.d_model),
            ("ffn_up", s.d_model, 2 * s.n_ff),
            ("ffn_down", s.n_ff, s.d_model),
        ] {
            mat(&mut w, &mut rng, &p(&format!("{name}.weight")), cols, rows);
        }
        norm(&mut w, &mut rng, &p("ffn_norm.weight"), s.d_model);
    }

    if let Some(h) = &s.head {
        // `ModernBertForSequenceClassification`'s head, in llama.cpp's order: dense
        // (`cls`, no bias), activation, `cls.norm` (weight only), then `cls.output`.
        // `cls.bias` and `cls.norm.bias` exist in the transformers config only when
        // `classifier_bias` / `norm_bias` are on, and llama.cpp creates neither; a file
        // that carries them is read without them (`extra_biases` writes them so the
        // tests can say so).
        if h.dense {
            mat(&mut w, &mut rng, "cls.weight", s.d_model, s.d_model);
        }
        if h.norm {
            norm(&mut w, &mut rng, "cls.norm.weight", s.d_model);
        }
        if h.out {
            mat(&mut w, &mut rng, "cls.output.weight", s.d_model, h.n_out);
            bias(&mut w, &mut rng, "cls.output.bias", h.n_out);
        }
        if h.extra_biases {
            bias(&mut w, &mut rng, "cls.bias", s.d_model);
            bias(&mut w, &mut rng, "cls.norm.bias", s.d_model);
        }
        if !h.labels.is_empty() {
            w.arr_str(&k("classifier.output_labels"), &h.labels);
        }
        if let Some(a) = h.activation {
            w.str(&k("classifier.activation"), a);
        }
        if let Some(p) = h.pooling_type {
            w.u32(&k("classifier.pooling_type"), p);
        }
    }

    let path = dir.join(format!("modern-bert-{}.gguf", fmt.label()));
    w.write_to_file(&path).expect("write synthetic model");
    path
}

/// The shape of a synthetic `nomic-bert` encoder (roadmap S4.8a).
///
/// The defaults have four layers, a 64-position context and a SwiGLU feed-forward,
/// with nomic-embed-text v1.5's shape: a fused `attn_qkv`, RoPE base 1000, mean
/// pooling, no projection biases, and no final norm (the last `layer_output_norm`
/// is the encoder's output).
#[derive(Clone, Copy, Debug)]
pub struct NomicBertConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub ctx: usize,
    /// `attention.layer_norm_epsilon`: nomic-embed-text's own files say 1e-12.
    pub eps: f32,
    /// `rope.freq_base`: nomic-embed-text's own files say 1000.
    pub rope_base: f32,
    /// `nomic-bert.pooling_type`: 1 is mean, 2 is cls.
    pub pooling_type: u32,
    /// Write the optional q/k/v and `attn_output` biases (the MoE variant's shape,
    /// which this family reads even though nomic-embed-text v1.5 has none).
    pub biases: bool,
    /// Omit this tensor from the file, to test the loader's refusal by name.
    pub omit: Option<&'static str>,
}

impl Default for NomicBertConfig {
    fn default() -> Self {
        NomicBertConfig {
            d_model: 128,
            n_layer: 4,
            n_head: 4,
            n_ff: 256,
            ctx: 64,
            eps: 1e-12,
            rope_base: 1000.0,
            pooling_type: 1,
            biases: false,
            omit: None,
        }
    }
}

/// Write a synthetic `nomic-bert` encoder: token embeddings plus token type 0, a
/// LayerNorm with a weight and a bias, post-norm blocks with a fused `attn_qkv`,
/// NeoX RoPE, full bidirectional attention and a SwiGLU feed-forward, then the last
/// block's `layer_output_norm` — the rows pooling reads. There is no position table
/// and no final norm, exactly the tensor list of nomic-embed-text v1.5's files.
pub fn write_nomic_bert(dir: &Path, fmt: Format, seed: u64, s: NomicBertConfig) -> PathBuf {
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    let arch = "nomic-bert";
    w.str("general.architecture", arch);
    w.str(
        "general.name",
        &format!("synthetic-nomic-bert-{}", fmt.label()),
    );
    let k = |x: &str| format!("{arch}.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("feed_forward_length"), s.n_ff as u32);
    w.u32(&k("context_length"), s.ctx as u32);
    w.u32(&k("vocab_size"), vocab::BERT_VOCAB_SIZE as u32);
    w.u32(&k("attention.causal"), 0);
    w.f32(&k("attention.layer_norm_epsilon"), s.eps);
    w.f32(&k("rope.freq_base"), s.rope_base);
    w.u32(&k("rope.dimension_count"), (s.d_model / s.n_head) as u32);
    w.u32(&k("pooling_type"), s.pooling_type);
    vocab::add_bert_tokenizer_metadata(&mut w);

    let wanted = |name: &str| s.omit != Some(name);
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
    let bias = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            let v = rng.f32_range(-0.05, 0.05);
            out.extend_from_slice(&v.to_le_bytes());
        }
        w.add_tensor(name, &[n as u64], Format::F32, out);
    };

    if wanted("token_embd.weight") {
        mat(
            &mut w,
            &mut rng,
            "token_embd.weight",
            s.d_model,
            vocab::BERT_VOCAB_SIZE,
        );
    }
    if wanted("token_types.weight") {
        // One token type ("Sentence A"), which llama.cpp hardcodes for every token.
        mat(&mut w, &mut rng, "token_types.weight", s.d_model, 2);
    }
    for name in ["token_embd_norm.weight", "token_embd_norm.bias"] {
        if wanted(name) {
            norm(&mut w, &mut rng, name, s.d_model);
        }
    }

    for i in 0..s.n_layer {
        let p = |x: &str| format!("blk.{i}.{x}");
        for (name, cols, rows) in [
            ("attn_qkv", s.d_model, 3 * s.d_model),
            ("attn_output", s.d_model, s.d_model),
            ("ffn_gate", s.d_model, s.n_ff),
            ("ffn_up", s.d_model, s.n_ff),
            ("ffn_down", s.n_ff, s.d_model),
        ] {
            let name = p(&format!("{name}.weight"));
            if wanted(&name) {
                mat(&mut w, &mut rng, &name, cols, rows);
            }
        }
        for name in [
            p("attn_output_norm.weight"),
            p("attn_output_norm.bias"),
            p("layer_output_norm.weight"),
            p("layer_output_norm.bias"),
        ] {
            if wanted(&name) {
                norm(&mut w, &mut rng, &name, s.d_model);
            }
        }
        if s.biases {
            for (name, n) in [
                (p("attn_qkv.bias"), 3 * s.d_model),
                (p("attn_output.bias"), s.d_model),
            ] {
                if wanted(&name) {
                    bias(&mut w, &mut rng, &name, n);
                }
            }
        }
    }

    let path = dir.join(format!("nomic-bert-{}.gguf", fmt.label()));
    w.write_to_file(&path).expect("write synthetic model");
    path
}

/// The shape of a synthetic `jina-bert-v2` encoder (roadmap S4.8c).
///
/// The defaults are tiny: four layers of 128 with 4 heads and a 192-wide GEGLU pair, a
/// 64-token context, and the separate `ffn_gate`/`ffn_up` matrices the reference files
/// carry. `fused_gate` writes the fused `2·n_ff` layout the graph also accepts;
/// `qk_norm` and `attn_norm_2` write the optional branches; `token_types` and
/// `context_length` can be turned off to test the loader's refusals.
#[derive(Clone, Copy, Debug)]
pub struct JinaBertConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub ctx: usize,
    pub pooling_type: u32,
    pub fused_gate: bool,
    pub qk_norm: bool,
    pub attn_norm_2: bool,
    /// Write `ffn_up.bias` (the reference files do not; llama.cpp marks it optional).
    pub up_bias: bool,
    pub token_types: bool,
    pub context_length: bool,
    /// Override the `ffn_up` width, to make a file the loader must refuse.
    pub up_rows: Option<usize>,
    pub eps: f32,
}

impl Default for JinaBertConfig {
    fn default() -> Self {
        JinaBertConfig {
            d_model: 128,
            n_layer: 4,
            n_head: 4,
            n_ff: 192,
            ctx: 64,
            pooling_type: 1,
            fused_gate: false,
            qk_norm: false,
            attn_norm_2: false,
            up_bias: true,
            token_types: true,
            context_length: true,
            up_rows: None,
            eps: 1e-12,
        }
    }
}

/// Write a synthetic `jina-bert-v2` encoder: separate q/k/v with biases, no position
/// table (ALiBi's bias comes from `f_max_alibi_bias = 8`), the token-type embedding's
/// row 0, LayerNorms with a bias, the GEGLU pair (separate or fused), and the layer's
/// own `layer_output_norm` as its output — the graph `src/models/bert.cpp` builds for
/// `LLM_ARCH_JINA_BERT_V2`.
///
/// Every matrix is `fmt` (F32 or Q8_0, both multiples of the block size); the norms and
/// the biases are F32, as the converter writes them.
pub fn write_jina_bert(dir: &Path, fmt: Format, seed: u64, s: JinaBertConfig) -> PathBuf {
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    let arch = "jina-bert-v2";
    w.str("general.architecture", arch);
    w.str(
        "general.name",
        &format!("synthetic-jina-bert-v2-{}", fmt.label()),
    );
    let k = |x: &str| format!("{arch}.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("feed_forward_length"), s.n_ff as u32);
    if s.context_length {
        w.u32(&k("context_length"), s.ctx as u32);
    }
    w.u32(&k("vocab_size"), VOCAB_SIZE as u32);
    w.f32(&k("attention.layer_norm_epsilon"), s.eps);
    w.u32(&k("pooling_type"), s.pooling_type);
    w.bool(&k("attention.causal"), false);
    vocab::add_jina_bert_tokenizer_metadata(&mut w);

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
    // Never zero: a bias the engine forgot to add must move the vector.
    let bias = |w: &mut GgufWriter, rng: &mut Rng, name: &str, n: usize| {
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            let v = rng.f32_range(-0.05, 0.05);
            out.extend_from_slice(&v.to_le_bytes());
        }
        w.add_tensor(name, &[n as u64], Format::F32, out);
    };

    mat(&mut w, &mut rng, "token_embd.weight", s.d_model, VOCAB_SIZE);
    if s.token_types {
        mat(&mut w, &mut rng, "token_types.weight", s.d_model, 2);
    }
    norm(&mut w, &mut rng, "token_embd_norm.weight", s.d_model);
    bias(&mut w, &mut rng, "token_embd_norm.bias", s.d_model);

    let up_rows = s
        .up_rows
        .unwrap_or(if s.fused_gate { 2 * s.n_ff } else { s.n_ff });
    for i in 0..s.n_layer {
        let p = |x: &str| format!("blk.{i}.{x}");
        for (name, cols, rows) in [
            ("attn_q", s.d_model, s.d_model),
            ("attn_k", s.d_model, s.d_model),
            ("attn_v", s.d_model, s.d_model),
            ("attn_output", s.d_model, s.d_model),
        ] {
            mat(&mut w, &mut rng, &p(&format!("{name}.weight")), cols, rows);
            bias(&mut w, &mut rng, &p(&format!("{name}.bias")), rows);
        }
        if !s.fused_gate {
            mat(&mut w, &mut rng, &p("ffn_gate.weight"), s.d_model, s.n_ff);
        }
        mat(&mut w, &mut rng, &p("ffn_up.weight"), s.d_model, up_rows);
        if s.up_bias {
            bias(&mut w, &mut rng, &p("ffn_up.bias"), up_rows);
        }
        mat(&mut w, &mut rng, &p("ffn_down.weight"), s.n_ff, s.d_model);
        bias(&mut w, &mut rng, &p("ffn_down.bias"), s.d_model);
        norm(&mut w, &mut rng, &p("attn_output_norm.weight"), s.d_model);
        bias(&mut w, &mut rng, &p("attn_output_norm.bias"), s.d_model);
        if s.qk_norm {
            norm(&mut w, &mut rng, &p("attn_q_norm.weight"), s.d_model);
            bias(&mut w, &mut rng, &p("attn_q_norm.bias"), s.d_model);
            norm(&mut w, &mut rng, &p("attn_k_norm.weight"), s.d_model);
            bias(&mut w, &mut rng, &p("attn_k_norm.bias"), s.d_model);
        }
        if s.attn_norm_2 {
            norm(&mut w, &mut rng, &p("attn_norm_2.weight"), s.d_model);
            bias(&mut w, &mut rng, &p("attn_norm_2.bias"), s.d_model);
        }
        norm(&mut w, &mut rng, &p("layer_output_norm.weight"), s.d_model);
        bias(&mut w, &mut rng, &p("layer_output_norm.bias"), s.d_model);
    }

    let path = dir.join(format!("jina-bert-v2-{}-{seed}.gguf", fmt.label()));
    w.write_to_file(&path).expect("write synthetic model");
    path
}

/// The shape of a synthetic `mistral3` embedder (roadmap S4.14).
///
/// The default has query heads wider than the residual (6 × 32 over 128), two key/value
/// heads over six query heads, YaRN with a 32-token original context and a 4× factor,
/// and an attention temperature of 0.1: a text of 40 tokens sees positions at and above
/// the original context, so both the YaRN ramp and the temperature run.
#[derive(Clone, Copy, Debug)]
pub struct Mistral3Config {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// The head width (`attention.key_length`); `n_head * head_dim` may exceed `d_model`.
    pub head_dim: usize,
    pub n_ff: usize,
    pub ctx: usize,
    /// `rope.scaling.original_context_length`: the YaRN original context and the
    /// attention temperature's divisor.
    pub n_ctx_orig: usize,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub log_mul: f32,
    /// `attention.temperature_scale`; 0 writes the key as zero and disables the scaling.
    pub temp_scale: f32,
    pub pooling_type: u32,
    /// Write `attention.causal true` — a file the loader must refuse by name.
    pub causal: bool,
}

impl Default for Mistral3Config {
    fn default() -> Self {
        Mistral3Config {
            d_model: 128,
            n_layer: 3,
            n_head: 6,
            n_head_kv: 2,
            head_dim: 32,
            n_ff: 256,
            ctx: 256,
            n_ctx_orig: 32,
            factor: 4.0,
            beta_fast: 8.0,
            beta_slow: 1.0,
            log_mul: 1.0,
            temp_scale: 0.1,
            pooling_type: 1,
            causal: false,
        }
    }
}

/// Write a synthetic `mistral3` embedder: NVIDIA Nemotron 3.5 Embed 1B's shape, tiny.
///
/// There is no output head and no chat template: the file only produces embeddings.
pub fn write_mistral3(dir: &Path, fmt: Format, seed: u64, s: Mistral3Config) -> PathBuf {
    let mut rng = Rng::new(seed);
    let mut w = GgufWriter::new();
    w.str("general.architecture", "mistral3");
    w.str(
        "general.name",
        &format!("synthetic-mistral3-{}", fmt.label()),
    );
    let k = |x: &str| format!("mistral3.{x}");
    w.u32(&k("block_count"), s.n_layer as u32);
    w.u32(&k("embedding_length"), s.d_model as u32);
    w.u32(&k("attention.head_count"), s.n_head as u32);
    w.u32(&k("attention.head_count_kv"), s.n_head_kv as u32);
    w.u32(&k("feed_forward_length"), s.n_ff as u32);
    w.u32(&k("context_length"), s.ctx as u32);
    w.u32(&k("vocab_size"), VOCAB_SIZE as u32);
    w.u32(&k("attention.key_length"), s.head_dim as u32);
    w.u32(&k("attention.value_length"), s.head_dim as u32);
    w.u32(&k("rope.dimension_count"), s.head_dim as u32);
    w.f32(&k("rope.freq_base"), 10_000.0);
    w.str(&k("rope.scaling.type"), "yarn");
    w.f32(&k("rope.scaling.factor"), s.factor);
    w.u32(
        &k("rope.scaling.original_context_length"),
        s.n_ctx_orig as u32,
    );
    w.f32(&k("rope.scaling.yarn_beta_fast"), s.beta_fast);
    w.f32(&k("rope.scaling.yarn_beta_slow"), s.beta_slow);
    w.f32(&k("rope.scaling.yarn_log_multiplier"), s.log_mul);
    w.bool(&k("attention.causal"), s.causal);
    w.f32(&k("attention.temperature_scale"), s.temp_scale);
    w.f32(&k("attention.layer_norm_rms_epsilon"), 1e-5);
    w.u32(&k("pooling_type"), s.pooling_type);

    vocab::add_tokenizer_metadata(&mut w, "qwen2");
    // No chat template: an embedding-only file has none, and the build says so.

    let kv = s.n_head_kv * s.head_dim;
    let q = s.n_head * s.head_dim;
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
            q,
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
        mat(
            &mut w,
            &mut rng,
            &format!("blk.{i}.attn_output.weight"),
            q,
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

    let path = dir.join(format!("mistral3-{}.gguf", fmt.label()));
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
