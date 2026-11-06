//! Builders for tiny synthetic models: shapes small enough to run in milliseconds, large enough that
//! every block format and the batched kernel paths accept them.

use super::vocab::{self, TemplateKind, VOCAB_SIZE};
use super::weights::{generate_norm, generate_weights, Rng};
use super::writer::GgufWriter;
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
    write_dense_impl(dir, arch, fmt, seed, s, Rope::None)
}

/// A llama file in F16 with the requested rotary-frequency treatment, for the rope-scaling loaders.
pub fn write_rope_dense(dir: &Path, rope: Rope, seed: u64) -> PathBuf {
    write_dense_impl(dir, "llama", Format::F16, seed, Dense::default(), rope)
}

fn write_dense_impl(
    dir: &Path,
    arch: &str,
    fmt: Format,
    seed: u64,
    s: Dense,
    rope: Rope,
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
    w.f32(&k("rope.freq_base"), 10000.0);
    if let Rope::Linear { factor } = rope {
        w.str(&k("rope.scaling.type"), "linear");
        w.f32(&k("rope.scaling.factor"), factor);
    }
    w.f32(&k("attention.layer_norm_rms_epsilon"), 1e-6);
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
    mat(&mut w, &mut rng, "output.weight", s.d_model, VOCAB_SIZE);
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
    let name = match rope {
        Rope::None => format!("{arch}-{}", fmt.label()),
        Rope::Freqs => format!("{arch}-rope-freqs-{}", fmt.label()),
        Rope::Linear { .. } => format!("{arch}-rope-linear-{}", fmt.label()),
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
    write_hybrid_impl(dir, fmt, seed, mtp, false)
}

/// The MTP variant whose drafts are accepted: every output-head row is identical, so all logits tie and
/// both the main model and the MTP block's shared head argmax to token 0; a draft then always matches what
/// the main model sampled. Random weights otherwise (the ordinary MTP model rejects every draft).
pub fn write_hybrid_accepting(dir: &Path, fmt: Format, seed: u64) -> PathBuf {
    write_hybrid_impl(dir, fmt, seed, true, true)
}

fn write_hybrid_impl(dir: &Path, fmt: Format, seed: u64, mtp: bool, accept: bool) -> PathBuf {
    assert!(matches!(fmt, Format::PQ2_0 | Format::PTQ1_0));
    let (d_model, n_ff, n_layer, interval) = (256usize, 512usize, 4usize, 4usize);
    let (n_head, n_kv, head_dim, rope_dim) = (4usize, 2usize, 64usize, 32usize);
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
    vocab::add_tokenizer_metadata(&mut w, "qwen35");
    w.str(
        "tokenizer.chat_template",
        vocab::chat_template(TemplateKind::Qwen3),
    );

    let tern = |w: &mut GgufWriter, rng: &mut Rng, name: &str, cols: usize, rows: usize| {
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
            tern(&mut w, &mut rng, &p("attn_q.weight"), d_model, 2 * q_dim);
            tern(
                &mut w,
                &mut rng,
                &p("attn_k.weight"),
                d_model,
                n_kv * head_dim,
            );
            tern(
                &mut w,
                &mut rng,
                &p("attn_v.weight"),
                d_model,
                n_kv * head_dim,
            );
            norm(&mut w, &mut rng, &p("attn_q_norm.weight"), head_dim);
            norm(&mut w, &mut rng, &p("attn_k_norm.weight"), head_dim);
            tern(&mut w, &mut rng, &p("attn_output.weight"), q_dim, d_model);
        } else {
            tern(&mut w, &mut rng, &p("attn_qkv.weight"), d_model, conv_dim);
            tern(&mut w, &mut rng, &p("attn_gate.weight"), d_model, value_dim);
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
            tern(&mut w, &mut rng, &p("ssm_out.weight"), value_dim, d_model);
        }
        norm(&mut w, &mut rng, &p("post_attention_norm.weight"), d_model);
        tern(&mut w, &mut rng, &p("ffn_gate.weight"), d_model, n_ff);
        tern(&mut w, &mut rng, &p("ffn_up.weight"), d_model, n_ff);
        tern(&mut w, &mut rng, &p("ffn_down.weight"), n_ff, d_model);
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
        tern(&mut w, &mut rng, "output.weight", d_model, VOCAB_SIZE);
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
    } else {
        format!("qwen35-{}", fmt.label())
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
