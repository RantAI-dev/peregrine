//! Regression test for the first-download bug: the `hf:` pre-flight must not latch the engine
//! options before the command sets them.
//!
//! `options::init` is process-global, so this lives in its own integration-test binary: nothing
//! else in this process may read the options first.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::options::EngineOptions;
use std::path::PathBuf;
use support::models::{write_dense, Dense};

fn tmpdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-fetch-preflight-test-{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&d);
    d
}

#[test]
fn the_download_preflight_does_not_latch_the_engine_options() {
    let dir = tmpdir();
    let model = write_dense(&dir, "llama", Format::F16, 7, Dense::default());

    // The exact sequence `fetch::fetch` runs on a cold download: the header-only probe first,
    // then the complete file, each through the option-free reader and the metadata-only check.
    let header = Gguf::open_header_plain(&model).expect("the probe parses the header");
    peregrine_models::check_compatible_metadata(&header).expect("the build can run the model");
    let full = Gguf::open_plain(&model).expect("the complete file parses");
    peregrine_models::check_compatible_metadata(&full).expect("the build can run the model");

    // The pre-flight must leave the options unset: the command initialises them afterwards, from
    // its own `--opt` and the tuning profile.
    let wanted = EngineOptions::from_env();
    assert!(
        peregrine_core::options::init(wanted.clone()).is_ok(),
        "the download pre-flight read the engine options before the command set them"
    );
    assert_eq!(peregrine_core::options::get(), &wanted);

    let _ = std::fs::remove_dir_all(&dir);
}
