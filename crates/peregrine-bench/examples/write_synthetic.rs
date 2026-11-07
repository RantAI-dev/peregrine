//! Write a synthetic GGUF model for benchmarking and testing.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example write_synthetic -- <dir> [ctx] [arch] [fmt] [d_model] [n_layer] [n_ff]
//! ```
//!
//! `arch` is `llama` (the default), `gemma3` (the GeGLU model the `peregrine bench` numbers
//! for the GeGLU kernel are measured on) or `qwen35-plain` (the hybrid model without the
//! Hadamard rotation, with ordinary quantised matmuls). `fmt` is `f16` (the default),
//! `q8_0`, `q4_k` or `q6_k`; it applies to `qwen35-plain` and to `llama`.
//!
//! The last three arguments size the model: the embedding width, the number of layers
//! and the feed-forward width. The defaults are the tiny ones the tests use; a
//! benchmark wants a model big enough to matter (`d_model` and `n_ff` must be
//! multiples of 256, the k-quant block).

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use std::path::PathBuf;
use support::models::{write_dense, write_gemma3, write_hybrid_plain, Dense, Gemma3Config};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("peregrine-synthetic"));
    let ctx = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let arch = args.get(2).map(String::as_str).unwrap_or("llama");
    let fmt = match args.get(3).map(String::as_str) {
        Some("q8_0") => Format::Q8_0,
        Some("q4_k") => Format::Q4_K,
        Some("q6_k") => Format::Q6_K,
        _ => Format::F16,
    };
    let d_model: Option<usize> = args.get(4).and_then(|s| s.parse().ok());
    let n_layer: Option<usize> = args.get(5).and_then(|s| s.parse().ok());
    let n_ff: Option<usize> = args.get(6).and_then(|s| s.parse().ok());
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let path = match arch {
        // The seeds are the ones the synthetic test uses: `1000 + index` over
        // `[F16, Q8_0]`.
        "qwen35-plain" => {
            let seed = if fmt == Format::Q8_0 { 1001 } else { 1000 };
            write_hybrid_plain(&out_dir, fmt, seed)
        }
        "gemma3" => write_gemma3(
            &out_dir,
            Format::F16,
            42,
            Gemma3Config {
                d_model: 1024,
                n_layer: 8,
                n_head: 8,
                n_head_kv: 4,
                n_ff: 2816,
                ctx,
                sliding_window: 64,
                ..Gemma3Config::default()
            },
        ),
        _ => {
            // The size arguments, when given, make the model big enough to measure
            // on; without them the model is the tiny one the tests use.
            let mut shape = Dense {
                ctx,
                ..Dense::default()
            };
            if let Some(d_model) = d_model {
                shape.n_head = (d_model / 64).max(1);
                shape.n_head_kv = (d_model / 256).max(1);
                shape.d_model = d_model;
            }
            if let Some(n_layer) = n_layer {
                shape.n_layer = n_layer;
            }
            if let Some(n_ff) = n_ff {
                shape.n_ff = n_ff;
            }
            write_dense(&out_dir, "llama", fmt, 42, shape)
        }
    };
    println!("{}", path.display());
}
