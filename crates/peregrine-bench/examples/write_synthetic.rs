//! Write a synthetic GGUF model for benchmarking and testing.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use std::path::PathBuf;
use support::models::{write_dense, Dense};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("peregrine-synthetic"));
    let ctx = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let shape = Dense {
        ctx,
        ..Dense::default()
    };
    let path = write_dense(&out_dir, "llama", Format::F16, 42, shape);
    println!("{}", path.display());
}
