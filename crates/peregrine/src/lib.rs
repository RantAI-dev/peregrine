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

pub mod fetch;
pub mod support;
pub use support::support;

#[cfg(feature = "cli")]
pub mod bench;
#[cfg(feature = "cli")]
pub mod decide;
#[cfg(feature = "cli")]
pub mod embed;
#[cfg(feature = "cli")]
pub mod run;
#[cfg(all(feature = "cli", feature = "server"))]
pub mod service;
#[cfg(feature = "cli")]
pub mod tune;

/// A description of a model file: what it is, what it holds, and whether this build can run it.
pub fn inspect(path: &std::path::Path) -> Result<String, String> {
    use std::fmt::Write as _;
    // `inspect` is a header reading, not a model load: it must not latch the process options or
    // copy the file into huge-page memory.
    let g = Gguf::open_plain(path)?;
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
    // What the file is for, when it says: `<arch>.pooling_type`, read as llama.cpp reads it
    // (0 none, 1 mean, 2 cls, 3 last, 4 rank).
    if let Ok(v) = g.u32(&format!("{arch}.pooling_type")) {
        let name = match v {
            0 => "none",
            1 => "mean",
            2 => "cls",
            3 => "last",
            4 => "rank",
            _ => "unknown",
        };
        let use_for = match v {
            1..=3 => " (embedding model)",
            4 => " (reranker/classifier)",
            _ => "",
        };
        let _ = writeln!(out, "pooling       {name} ({v}){use_for}");
    }
    // What a decision model declares (`docs/reference/systemone.md` §1.1): the type, how many
    // temperatures the file carries and the length of its `systemone` template. The template's
    // bytes are never printed.
    match peregrine_core::decision::DecisionInfo::from_gguf(&g) {
        Ok(Some(d)) => {
            let _ = writeln!(
                out,
                "decision      {} ({} temperatures, template {} bytes)",
                d.kind().name(),
                d.temperatures().len(),
                d.template_len()
            );
        }
        Ok(None) => {}
        Err(e) => {
            let _ = writeln!(out, "decision      invalid: {e}");
        }
    }
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
    // The pack cache (S4.9e): whether a load would map the packed weights instead
    // of packing them. `from_env` does not latch the process options.
    let opts = EngineOptions::from_env();
    let cache = peregrine_core::packcache::model_status(path, &opts);
    let _ = writeln!(out, "pack cache    {}", cache.describe());
    Ok(out)
}

/// The CPU kernel tier and every engine option in effect.
pub fn describe_runtime() -> String {
    let o = peregrine_core::options::get();
    // The prefill chunk's default depends on the pool, so the resolved value is printed beside the
    // option: `prefill_chunk=default` alone does not say what the next engine will use.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = o
        .prefill_chunk
        .unwrap_or_else(|| peregrine_core::options::prefill_chunk_default_for(threads));
    // The pass budget is per model (it depends on the model's width), so the default is printed
    // for a representative 1024-wide model beside the option.
    let pass = o.pass_rows.map_or_else(
        || {
            format!(
                "{} (default, for a Q8_0 model of d_model 1024; L2 and bytes-per-weight)",
                peregrine_core::options::pass_rows_default_for(1024, 1.06, false)
            )
        },
        |n| n.to_string(),
    );
    format!(
        "kernel tier   {}\nsimd features {}\nkq_repack     {}\nfpack         {}\npack cache    {}\nprefill chunk {chunk} (for {threads} workers)\npass rows     {pass}\noptions:\n{}",
        peregrine_kernels::cpu::tier_name(),
        peregrine_core::simd_name(),
        o.kq_repack_note(),
        o.fpack_note(),
        o.pack_cache_note(),
        o.describe()
            .lines()
            .map(|l| format!("  {l}\n"))
            .collect::<String>()
    )
}
