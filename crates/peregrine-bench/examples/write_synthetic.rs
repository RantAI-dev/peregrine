//! Write a synthetic GGUF model for benchmarking and testing.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example write_synthetic -- <dir> [ctx] [arch]
//! ```
//!
//! `arch` is `llama` (the default) or `gemma3` — the latter writes the GeGLU model the
//! `peregrine bench` numbers for the GeGLU kernel are measured on.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use std::path::PathBuf;
use support::models::{write_dense, write_gemma3, Dense, Gemma3Config};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("peregrine-synthetic"));
    let ctx = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let arch = args.get(2).map(String::as_str).unwrap_or("llama");
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let path = match arch {
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
        _ => write_dense(
            &out_dir,
            "llama",
            Format::F16,
            42,
            Dense {
                ctx,
                ..Dense::default()
            },
        ),
    };
    println!("{}", path.display());
}
