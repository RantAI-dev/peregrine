//! Model architecture implementations for Peregrine (ADR 0005, ADR 0006).
//!
//! Architectures are implemented behind the [`peregrine_core::arch::Architecture`] trait from `peregrine-core`.
//! Loading model files into a [`Model`] or [`peregrine_core::session::Session`] is wired through this crate.

// The workspace builds with `-D warnings`; these idioms are kept as they are.
// `chunks_exact_to_as_chunks` exists only in newer clippy (CI's stable is ahead of older local toolchains).
#![allow(
    unknown_lints,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_map_or,
    clippy::chunks_exact_to_as_chunks
)]

pub mod dense;
pub mod qwen35;

pub use dense::DenseFamily;
pub use qwen35::Qwen35Family;

// Re-exports for consumers expecting `hybrid` and `mtp`
pub mod hybrid {
    pub use crate::qwen35::{profile, HyScratch, HybridCfg};
}
pub mod mtp {
    pub use crate::qwen35::mtp::Mtp;
    pub use peregrine_core::arch::{MtpHead, MtpState, MTP_MAX_ROWS};
}

use peregrine_core::arch::Family;
use peregrine_core::gguf::Gguf;
use peregrine_core::model::Model;
use peregrine_core::template::Template;
use std::path::Path;
use std::sync::Arc;

/// Architecture names this crate implements, for compatibility probe messages.
pub const ARCHS: &[&str] = &["llama", "qwen2", "qwen3", "qwen35"];

/// Table of model families implemented by Peregrine.
pub static FAMILIES: &[&dyn Family] = &[&DenseFamily, &Qwen35Family];

/// Load a model from a parsed GGUF file using the built-in model families.
pub fn load(g: Gguf) -> Result<Model, String> {
    Model::load_with(g, FAMILIES)
}

/// Check whether the GGUF model file is compatible with Peregrine's architectures.
pub fn check_compatible(g: &Gguf) -> Result<(), String> {
    peregrine_core::check_compatible(g, FAMILIES)
}

/// Check compatibility with a specific chosen chat template.
pub fn check_compatible_with(g: &Gguf, template: Option<Template>) -> Result<(), String> {
    peregrine_core::check_compatible_with(g, template, FAMILIES)
}

/// Check whether the model file at `path` can be loaded and run by Peregrine.
pub fn compatibility(path: &Path) -> Result<(), String> {
    peregrine_core::compatibility(path, FAMILIES)
}

/// Kept model caching helpers wired to Peregrine's built-in model families.
pub mod keep {
    use peregrine_core::model::Model;
    use peregrine_core::template::Template;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    /// The model at `path`: the kept one when it is the same file, else freshly loaded.
    pub fn open(path: &Path) -> Result<Arc<Model>, String> {
        open_with(path, true)
    }

    /// `open`, choosing whether a LittleBit sidecar beside the model is used.
    pub fn open_with(path: &Path, lb: bool) -> Result<Arc<Model>, String> {
        open_with_template(path, lb, None)
    }

    /// `open_with`, with the chat template the caller chose (`--chat-template`).
    pub fn open_with_template(
        path: &Path,
        lb: bool,
        template: Option<Template>,
    ) -> Result<Arc<Model>, String> {
        peregrine_core::model::keep::open_with_template(path, lb, template, crate::FAMILIES)
    }

    pub fn has(path: &Path) -> bool {
        peregrine_core::model::keep::has(path)
    }

    pub fn has_with(path: &Path, lb: bool) -> bool {
        peregrine_core::model::keep::has_with(path, lb)
    }

    pub fn status() -> Option<(PathBuf, usize, f64)> {
        peregrine_core::model::keep::status()
    }

    pub fn drop_kept() -> bool {
        peregrine_core::model::keep::drop_kept()
    }

    pub fn sweep() -> bool {
        peregrine_core::model::keep::sweep()
    }
}

pub fn open(path: &Path) -> Result<Arc<Model>, String> {
    keep::open(path)
}

pub fn open_with(path: &Path, lb: bool) -> Result<Arc<Model>, String> {
    keep::open_with(path, lb)
}

pub fn open_with_template(
    path: &Path,
    lb: bool,
    template: Option<Template>,
) -> Result<Arc<Model>, String> {
    keep::open_with_template(path, lb, template)
}

pub mod session {
    use peregrine_core::session::{Session, SessionCfg};
    use std::path::Path;

    /// Open a model at `path` and initialize a single-user `Session`.
    pub fn open(path: &Path, cfg: SessionCfg) -> Result<Session, String> {
        let model = crate::open_with_template(path, cfg.littlebit, cfg.template)?;
        Session::from_model(model, cfg)
    }
}

pub use session::open as open_session;
