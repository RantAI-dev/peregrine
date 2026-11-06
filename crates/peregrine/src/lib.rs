//! Peregrine: a CPU inference engine for language models.
//!
//! This is the facade crate an application depends on. It re-exports the engine and, with the default
//! `server` feature, the OpenAI-compatible server; the `peregrine` binary is built with the `cli` feature.
//!
//! ```toml
//! peregrine = { git = "https://github.com/RantAI-dev/peregrine", default-features = false }
//! ```
//!
//! Which architectures and formats are compiled in becomes a feature choice with roadmap S1.2.

/// The engine: model files, tokenizer, layers, cache, sampling, the batching `Engine`.
pub use peregrine_core as engine;
/// CPU kernels, the kernel tier cap and the weight formats.
pub use peregrine_kernels as kernels;
/// Model architecture implementations and loading.
pub use peregrine_models as models;
/// The OpenAI-compatible HTTP server.
#[cfg(feature = "server")]
pub use peregrine_server as server;

pub use peregrine_core::gguf::{Format, Gguf};
pub use peregrine_core::model::Model;
pub use peregrine_core::options::EngineOptions;
pub use peregrine_core::profile::Profile;
pub use peregrine_core::serve::Engine;

pub mod support;
pub use support::support;

#[cfg(feature = "cli")]
pub mod bench;
#[cfg(feature = "cli")]
pub mod tune;

/// A description of a model file: what it is, what it holds, and whether this build can run it.
pub fn inspect(path: &std::path::Path) -> Result<String, String> {
    use std::fmt::Write as _;
    let g = Gguf::open(path)?;
    let mut out = String::new();
    let arch = g.str("general.architecture").unwrap_or("unknown");
    let _ = writeln!(out, "file          {}", path.display());
    let _ = writeln!(
        out,
        "name          {}",
        g.str("general.name").unwrap_or("-")
    );
    let _ = writeln!(out, "architecture  {arch}");
    for key in [
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "context_length",
    ] {
        if let Ok(v) = g.u32(&format!("{arch}.{key}")) {
            let _ = writeln!(
                out,
                "{:<14}{v}",
                key.replace("_length", "").replace("_count", "s")
            );
        }
    }
    let tok_model = g.str("tokenizer.ggml.model").unwrap_or("-");
    let tok_display = match tok_model {
        "llama" => "llama (SPM)",
        other => other,
    };
    let _ = writeln!(out, "tokenizer     {tok_display}");
    let mut by_fmt: std::collections::BTreeMap<&str, (usize, u64)> = Default::default();
    for t in g.tensors.values() {
        let e = by_fmt.entry(t.kind.label()).or_default();
        e.0 += 1;
        e.1 += t.bytes();
    }
    let _ = writeln!(
        out,
        "weights       {:.1} MiB",
        g.total_weight_bytes() as f64 / (1u64 << 20) as f64
    );
    for (f, (n, b)) in &by_fmt {
        let _ = writeln!(
            out,
            "  {f:<8} {n:>4} tensors {:>10.1} MiB",
            *b as f64 / (1u64 << 20) as f64
        );
    }
    let _ = match peregrine_models::check_compatible(&g) {
        Ok(()) => writeln!(out, "supported     yes"),
        Err(e) => writeln!(out, "supported     no: {e}"),
    };
    Ok(out)
}

/// The CPU kernel tier and every engine option in effect.
pub fn describe_runtime() -> String {
    let o = peregrine_core::options::get();
    format!(
        "kernel tier   {}\nsimd          {}\noptions:\n{}",
        peregrine_kernels::cpu::tier_name(),
        peregrine_core::simd_name(),
        o.describe()
            .lines()
            .map(|l| format!("  {l}\n"))
            .collect::<String>()
    )
}
