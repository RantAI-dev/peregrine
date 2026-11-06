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

pub mod attn;
mod blocks;
pub mod dspark;
pub mod gguf;
pub mod kv;
pub mod lbq;
pub mod model;
pub mod options;
pub mod pool;
pub mod prefix;
pub mod profile;
pub mod safetensors;
pub mod sample;
pub mod serve;
pub mod session;
pub mod suffix;
pub mod telem;
pub mod template;
pub mod tokenizer;

pub use gguf::{Format, Gguf};
// The kernels live in `peregrine-kernels`; they keep their paths here.
pub use peregrine_kernels::cpu_has;
pub use peregrine_kernels::{cpu, delta, format, hadamard, quant, table, vmath};
#[cfg(target_arch = "x86_64")]
pub use peregrine_kernels::{simd, tern};
pub use profile::Profile;
pub use session::{BenchReport, Session, SessionCfg};

/// Whether this engine can run the GGUF at `path`, read from its header alone.
///
/// `Ok(())` means it will load and chat correctly; `Err` names the first thing it
/// cannot do. This is the check a host runs *before* offering a
/// pairing, because every way of failing it — another architecture, another
/// tokenizer, another chat template — otherwise surfaces at the first token as an
/// error or, worse, as fluent text from a prompt the model never saw.
///
/// What is implemented, and therefore what this accepts:
/// * the `llama` and `qwen2` architectures (`model::ARCHS`);
/// * a `gpt2` byte-level BPE vocabulary with the `smollm`, `llama-bpe` or `qwen2`
///   pre-tokenizer (`tokenizer::PRE_TYPES`) — each transcribed from llama.cpp and
///   checked against `llama-tokenize`; a near-miss tokenizer produces confident
///   wrong text, so nothing else is approximated;
/// * a chat template (`template::TEMPLATES`), detected from the
///   file's own `tokenizer.chat_template`;
/// * rope scaling only as Llama 3's factor table or a linear factor;
/// * tensors in F32, F16, Q5_0, Q8_0, Q4_K or Q6_K (`Gguf::open` refuses others).
pub fn compatibility(path: &std::path::Path) -> Result<(), String> {
    let g = Gguf::open(path)?;
    check_compatible(&g)
}

pub fn check_compatible(g: &Gguf) -> Result<(), String> {
    check_compatible_with(g, None)
}

/// The same check, with the chat template the caller chose (`--chat-template`).
///
/// A file that declares no template at all — or one whose template this engine
/// does not render — is refused with a message naming the option: the template
/// is the one thing the file cannot say, and the engine will not guess it.
pub fn check_compatible_with(g: &Gguf, template: Option<template::Template>) -> Result<(), String> {
    let arch = g.str("general.architecture").unwrap_or("unknown");
    if model::Arch::from_name(arch).is_none() {
        return Err(format!(
            "architecture {arch:?} — Peregrine implements {}",
            model::ARCHS.join(", ")
        ));
    }
    let tok = g.str("tokenizer.ggml.model").unwrap_or("unknown");
    if tok != "gpt2" && tok != "llama" {
        return Err(format!(
            "tokenizer {tok:?} — Peregrine reads gpt2 (BPE) and llama (SentencePiece) vocabularies"
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
        if t != "none" && t != "linear" {
            return Err(format!("rope scaling {t:?} is not implemented"));
        }
    }
    match template {
        // The caller named the template: it is used as it is, and the file's own
        // template (if any) is not read.
        Some(t) => t.tokens_present(g)?,
        None => {
            template::Template::detect(g).map_err(|e| {
                format!("{e} — a file without a usable chat template needs --chat-template <name>")
            })?;
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
