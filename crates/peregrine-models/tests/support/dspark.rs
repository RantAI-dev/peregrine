//! A synthetic DSpark drafter: the two files a drafter directory holds.
//!
//! DeepSeek's drafters ship as safetensors (bf16) with a `config.json` beside
//! them, and nothing else in the engine reads them. Both formats are simple
//! enough to write in a test, so the drafter paths — the taps, the context cache,
//! the draft — can be exercised without a model file.
//!
//! The drafter here is random: it proposes nonsense, and the target rejects it.
//! What the tests need from it is that it *runs*.

use super::weights::Rng;
use std::path::{Path, PathBuf};

/// The drafter's shape. `hidden` is the target's width (the drafter reads the
/// target's hidden states), the rest is the drafter's own.
#[derive(Clone, Debug)]
pub struct DrafterShape {
    pub hidden: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub block_size: usize,
    pub mask_id: u32,
    /// The target's layers the drafter taps.
    pub target_layer_ids: Vec<usize>,
}

impl Default for DrafterShape {
    fn default() -> Self {
        DrafterShape {
            hidden: 256,
            n_layer: 1,
            n_head: 8,
            n_kv_head: 4,
            head_dim: 32,
            n_ff: 512,
            n_vocab: 320,
            block_size: 7,
            mask_id: 259,
            target_layer_ids: vec![0],
        }
    }
}

/// Write a drafter directory (`config.json` + `model.safetensors`) under `dir`,
/// and return it. The weights are random and small.
pub fn write_drafter(dir: &Path, shape: DrafterShape, seed: u64) -> PathBuf {
    let d = dir.join("dspark");
    std::fs::create_dir_all(&d).expect("create the drafter directory");
    let s = shape;
    let taps = s.target_layer_ids.len();
    let q_dim = s.n_head * s.head_dim;
    let kv_dim = s.n_kv_head * s.head_dim;
    let mut rng = Rng::new(seed);

    // The config the drafter's reader parses.
    let config = format!(
        "{{\"architectures\":[\"Qwen3DSparkModel\"],\
          \"hidden_size\":{h},\"num_hidden_layers\":{l},\"num_attention_heads\":{nh},\
          \"num_key_value_heads\":{nkv},\"head_dim\":{hd},\"intermediate_size\":{ff},\
          \"vocab_size\":{v},\"block_size\":{bs},\"mask_token_id\":{mask},\"markov_rank\":0,\
          \"rms_norm_eps\":1e-06,\"rope_parameters\":{{\"rope_theta\":1000000}},\
          \"target_layer_ids\":[{taps}],\"enable_confidence_head\":false,\
          \"confidence_head_with_markov\":false,\"layer_types\":[\"full_attention\"]}}",
        h = s.hidden,
        l = s.n_layer,
        nh = s.n_head,
        nkv = s.n_kv_head,
        hd = s.head_dim,
        ff = s.n_ff,
        v = s.n_vocab,
        bs = s.block_size,
        mask = s.mask_id,
        taps = s
            .target_layer_ids
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    std::fs::write(d.join("config.json"), config).expect("write the drafter config");

    // The weights: every matrix the reader names, in the `[out, in]` shape the
    // safetensors convention uses.
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    mat(
        &mut tensors,
        &mut rng,
        "embed_tokens.weight",
        s.n_vocab,
        s.hidden,
    );
    mat(
        &mut tensors,
        &mut rng,
        "fc.weight",
        s.hidden,
        taps * s.hidden,
    );
    vec1(&mut tensors, &mut rng, "hidden_norm.weight", s.hidden);
    vec1(&mut tensors, &mut rng, "norm.weight", s.hidden);
    for i in 0..s.n_layer {
        let p = |x: &str| format!("layers.{i}.{x}");
        vec1(
            &mut tensors,
            &mut rng,
            &p("input_layernorm.weight"),
            s.hidden,
        );
        vec1(
            &mut tensors,
            &mut rng,
            &p("post_attention_layernorm.weight"),
            s.hidden,
        );
        mat(
            &mut tensors,
            &mut rng,
            &p("self_attn.q_proj.weight"),
            q_dim,
            s.hidden,
        );
        mat(
            &mut tensors,
            &mut rng,
            &p("self_attn.k_proj.weight"),
            kv_dim,
            s.hidden,
        );
        mat(
            &mut tensors,
            &mut rng,
            &p("self_attn.v_proj.weight"),
            kv_dim,
            s.hidden,
        );
        mat(
            &mut tensors,
            &mut rng,
            &p("self_attn.o_proj.weight"),
            s.hidden,
            q_dim,
        );
        vec1(
            &mut tensors,
            &mut rng,
            &p("self_attn.q_norm.weight"),
            s.head_dim,
        );
        vec1(
            &mut tensors,
            &mut rng,
            &p("self_attn.k_norm.weight"),
            s.head_dim,
        );
        mat(
            &mut tensors,
            &mut rng,
            &p("mlp.gate_proj.weight"),
            s.n_ff,
            s.hidden,
        );
        mat(
            &mut tensors,
            &mut rng,
            &p("mlp.up_proj.weight"),
            s.n_ff,
            s.hidden,
        );
        mat(
            &mut tensors,
            &mut rng,
            &p("mlp.down_proj.weight"),
            s.hidden,
            s.n_ff,
        );
    }
    mat(
        &mut tensors,
        &mut rng,
        "lm_head.weight",
        s.n_vocab,
        s.hidden,
    );
    write_safetensors(&d.join("model.safetensors"), &tensors);
    d
}

/// A random matrix, `[rows, cols]`.
fn mat(
    tensors: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
    rng: &mut Rng,
    name: &str,
    rows: usize,
    cols: usize,
) {
    let w: Vec<f32> = (0..rows * cols)
        .map(|_| rng.f32_range(-0.04, 0.04))
        .collect();
    tensors.push((name.to_string(), vec![rows, cols], w));
}

/// A random norm, around 1.
fn vec1(tensors: &mut Vec<(String, Vec<usize>, Vec<f32>)>, rng: &mut Rng, name: &str, n: usize) {
    let w: Vec<f32> = (0..n).map(|_| rng.f32_range(0.95, 1.05)).collect();
    tensors.push((name.to_string(), vec![n], w));
}

/// Write a safetensors file: a little-endian header length, a JSON header, then
/// the tensors' bytes. Every tensor here is F32.
fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = String::from("{");
    let mut data: Vec<u8> = Vec::new();
    for (i, (name, shape, values)) in tensors.iter().enumerate() {
        let begin = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();
        if i > 0 {
            header.push(',');
        }
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{dims}],\"data_offsets\":[{begin},{end}]}}"
        ));
    }
    header.push('}');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    std::fs::write(path, out).expect("write the drafter weights");
}
