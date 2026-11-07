//! Write a synthetic GGUF model for benchmarking and testing.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example write_synthetic -- <dir> [ctx] [arch] [fmt]
//! ```
//!
//! `arch` is `llama` (the default), `gemma3` (the GeGLU model the `peregrine bench` numbers
//! for the GeGLU kernel are measured on) or `qwen35-plain` (the hybrid model without the
//! Hadamard rotation, with ordinary quantised matmuls). `fmt` is `f16` (the default) or
//! `q8_0`; it applies to `qwen35-plain`.

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
        _ => Format::F16,
    };
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
