//! Peregrine: a CPU inference engine for language models that reports the bytes it moved.
//!
//! Single-stream decode is memory-bandwidth-bound by an order of magnitude, so the quantity that
//! matters is not tokens per second alone but bytes moved per token. Peregrine reports both, so every
//! run can be checked against the machine's own roofline instead of taken on faith.
//!
//! This crate is the engine: model loading (GGUF, safetensors), tokenizer and chat templates, kernels,
//! the thread pool, the cache, sampling, speculation and the batching scheduler (`serve::Engine`). It
//! knows nothing about HTTP; `peregrine-server` is the thin layer on top. The architecture is described
//! in `ARCHITECTURE.md` at the repository root.

// The workspace builds with `-D warnings`; these idioms are kept as they are.
// `chunks_exact_to_as_chunks` exists only in newer clippy (CI's stable is ahead of older local toolchains).
#![allow(
    unknown_lints,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_map_or,
    clippy::chunks_exact_to_as_chunks
)]

pub mod arch;
pub mod attn;
pub mod blocks;
pub mod calib;
pub mod decision;
pub mod draft;
#[cfg(feature = "dspark")]
pub mod dspark;
pub mod embed;
#[cfg(not(feature = "dspark"))]
pub mod dspark {
    pub struct Drafter {
        pub stats: Stats,
        pub cfg: Cfg,
    }
    pub struct Stats {
        pub bytes: u64,
    }
    pub struct Cfg {
        pub block_size: usize,
        pub logits_start: usize,
        pub target_layer_ids: Vec<usize>,
    }
    impl Drafter {
        pub fn update_context(&mut self, _taps: &[f32], _start: usize, _pool: &crate::pool::Pool) {}
        pub fn truncate(&mut self, _pos: usize) {}
        pub fn draft_with(
            &mut self,
            _next: u32,
            _kmax: usize,
            _pool: &crate::pool::Pool,
            mut _f: impl FnMut(usize, &[f32]) -> u32,
        ) -> (Vec<u32>, Vec<f32>) {
            (Vec::new(), Vec::new())
        }
    }
}
pub mod fpack;
pub mod gguf;
pub mod json;
pub mod kq;
pub mod kv;
#[cfg(feature = "littlebit")]
pub mod lbq;
#[cfg(not(feature = "littlebit"))]
pub mod lbq {
    pub struct Slb;
}
pub mod model;
pub mod options;
pub mod packcache;
pub mod pool;
pub mod prefix;
pub mod profile;
pub mod safetensors;
pub mod sample;
pub mod serve;
pub mod session;
pub mod state;
pub mod suffix;
pub mod systemone;
pub mod telem;
pub mod template;
pub mod tokenizer;
mod unicode_data;

pub use gguf::{Format, Gguf};
// The kernels live in `peregrine-kernels`; they keep their paths here.
pub use peregrine_kernels::cpu_has;
#[cfg(target_arch = "x86_64")]
pub use peregrine_kernels::simd;
#[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
pub use peregrine_kernels::tern;
pub use peregrine_kernels::{cpu, delta, format, hadamard, prof, quant, table, vmath};
pub use profile::Profile;
pub use session::{BenchReport, Session, SessionCfg};

/// Whether this engine can run the GGUF at `path`, read from its header alone.
///
/// `Ok(())` means it will load and chat correctly; `Err` names the first thing it
/// cannot do. This is the check a host runs *before* offering a
/// pairing, because every way of failing it — another architecture, another
/// tokenizer, another chat template — otherwise surfaces at the first token as an
/// error or, worse, as fluent text from a prompt the model never saw.
pub fn compatibility(path: &std::path::Path, families: &[&dyn arch::Family]) -> Result<(), String> {
    let g = Gguf::open_plain(path)?;
    check_compatible(&g, families)
}

pub fn check_compatible(g: &Gguf, families: &[&dyn arch::Family]) -> Result<(), String> {
    check_compatible_with(g, None, families)
}

/// The same check, with the chat template the caller chose (`--chat-template`).
///
/// A file that declares no template at all — or one whose template this engine
/// does not render — is refused with a message naming the option: the template
/// is the one thing the file cannot say, and the engine will not guess it.
pub fn check_compatible_with(
    g: &Gguf,
    template: Option<template::Template>,
    families: &[&dyn arch::Family],
) -> Result<(), String> {
    // Everything a check can decide from the file and the build's features lives here. Anything
    // that needs the process options (the kernel tier, the kernel table) belongs in the load path,
    // not in this pre-flight: a download happens before the command line has set the options.
    check_compatible_metadata(g, template, families)
}

/// The part of [`check_compatible_with`] that reads only the file's metadata and this build's
/// features: the architecture, the tensor formats, the tokenizer, the rope scaling and the chat
/// template. It never reads [`crate::options::get`] or the kernel table, so it is safe to run on a
/// partial `hf:` download before the command line has initialised the process options.
pub fn check_compatible_metadata(
    g: &Gguf,
    template: Option<template::Template>,
    families: &[&dyn arch::Family],
) -> Result<(), String> {
    let arch = g.str("general.architecture").unwrap_or("unknown");
    let family = families.iter().find(|f| f.names().contains(&arch));
    let Some(family) = family else {
        let feat = match arch {
            "llama" => Some("arch-llama"),
            "qwen2" | "qwen3" => Some("arch-qwen"),
            "qwen25" | "qwen35" => Some("arch-qwen35"),
            "gemma3" => Some("arch-gemma3"),
            "gemma-embedding" => Some("arch-gemma-embedding"),
            "qwen3moe" => Some("arch-qwen3moe"),
            "gpt-oss" => Some("arch-gpt-oss"),
            "phi3" => Some("arch-phi3"),
            _ => None,
        };
        if let Some(feat) = feat {
            return Err(format!(
                "architecture {arch:?} is not enabled in this build (recompile with feature '{feat}')"
            ));
        }
        let arch_names: Vec<&str> = families
            .iter()
            .flat_map(|f| f.names().iter().copied())
            .collect();
        return Err(format!(
            "architecture {arch:?} — Peregrine implements {}",
            arch_names.join(", ")
        ));
    };
    for t in g.tensors.values() {
        if !t.kind.is_supported() {
            let feat = t.kind.required_feature().unwrap_or("unknown");
            return Err(format!(
                "tensor '{}' uses format {} which is disabled in this build (enable feature '{feat}')",
                t.name,
                t.kind.label()
            ));
        }
    }
    let tok = g.str("tokenizer.ggml.model").unwrap_or("unknown");
    if !matches!(tok, "gpt2" | "llama" | "bert" | "t5") {
        return Err(format!(
            "tokenizer {tok:?} — Peregrine reads gpt2 (BPE), llama (SentencePiece BPE), \
             bert (WordPiece) and t5 (SentencePiece unigram) vocabularies"
        ));
    }
    if tok == "gpt2" {
        let pre = g.str("tokenizer.ggml.pre").unwrap_or("smollm");
        if tokenizer::PreType::from_name(pre).is_none() {
            return Err(format!(
                "pre-tokenizer {pre:?} — Peregrine implements {}",
                tokenizer::PRE_TYPES.join(", ")
            ));
        }
    }
    if let Ok(t) = g.str(&format!("{arch}.rope.scaling.type")) {
        // YaRN is gpt-oss's scaling; the families that read it check the rest.
        if t != "none" && t != "linear" && !(t == "yarn" && arch == "gpt-oss") {
            return Err(format!("rope scaling {t:?} is not implemented"));
        }
    }
    // An embedding-only file has no chat template and needs none: it is served on the
    // embeddings endpoint and refuses generation (design §3).
    if !family.embedding_only() {
        match template {
            // The caller named the template: it is used as it is, and the file's own
            // template (if any) is not read.
            Some(t) => t.tokens_present(g)?,
            None => {
                template::Template::detect(g).map_err(|e| {
                    format!(
                        "{e} — a file without a usable chat template needs --chat-template <name>"
                    )
                })?;
            }
        }
    }
    Ok(())
}

/// Which kernel path this build will actually take on this machine.
///
/// Reported in telemetry so that a slow run on a machine without AVX2 is never
/// mistaken for a slow engine — and so that a run claiming SIMD throughput can be
/// checked against whether SIMD was in fact available.
pub fn simd_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu_has!("avx2") {
            if crate::cpu_has!("fma") {
                return "avx2+fma";
            }
            return "avx2";
        }
        return "scalar";
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return "neon";
        }
        return "scalar";
    }
    #[allow(unreachable_code)]
    "scalar"
}

/// The kernel tier this process actually runs, as [`peregrine_kernels::cpu::tier_name`]
/// spells it: `x86_64-avx2-avx512-vnni-gfni-vbmi` and so on, after the
/// `PEREGRINE_ISA` cap.
///
/// This is the string manifests and `peregrine tier` report as the tier. It is
/// not the same question [`simd_name`] answers: that one names the instruction
/// sets compiled into the build, and it stops at AVX2 even where the kernels run
/// the AVX-512 tier.
pub fn simd_tier() -> String {
    cpu::tier_name()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "x86_64")]
    use crate::cpu::Tier;

    /// The manifest's tier is the kernel tier, not the instruction-set baseline:
    /// an AVX-512 host must not be labelled `avx2+fma` just because the build's
    /// `simd_name` stops there.
    #[test]
    fn the_reported_tier_is_the_kernel_tier() {
        assert_eq!(simd_tier(), cpu::tier_name());
        #[cfg(target_arch = "x86_64")]
        {
            // Every x86-64 tier name starts with the architecture and names a tier.
            assert!(simd_tier().starts_with("x86_64"), "{}", simd_tier());
            let detected = cpu::detected();
            let resolved = cpu::resolved();
            assert!(
                resolved <= detected,
                "the cap raised the tier: {resolved:?} > {detected:?}"
            );
            if resolved == Tier::Avx512 {
                assert!(simd_tier().contains("avx512"), "{}", simd_tier());
            }
        }
    }

    /// The two strings answer different questions, and the old field is kept for
    /// the tooling that reads it.
    #[test]
    fn the_simd_field_is_kept_beside_the_tier() {
        let m = telem::Machine::probe(1);
        assert_eq!(m.simd, simd_name());
        assert_eq!(m.simd_tier, simd_tier());
    }
}
