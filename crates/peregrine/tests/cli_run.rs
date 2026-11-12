//! `peregrine run` on a file that carries Qwen 3.5's chat template (roadmap S6.3c).
//!
//! That template opens the reasoning block at the end of the prompt, so the command shows
//! the opening tag itself and the model writes the rest. A synthetic model writes random
//! tokens, so these tests pin the prompt's two endings and the tag, not what a model says;
//! the prompt's bytes are pinned against llama.cpp in `peregrine-core` (`template.rs` and
//! `tests/tools_render.rs`).

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use support::models::{write_dense, write_dense_with_template, Dense};

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-cli-run-{}-{test}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

fn shape() -> Dense {
    Dense {
        ctx: 1024,
        ..Dense::default()
    }
}

/// `peregrine run --json` on `model`, greedy, four tokens.
fn run(model: &Path, extra: &[&str]) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_peregrine"))
        .args(["run", "-m", model.to_str().unwrap()])
        .args(["--max-tokens", "4", "--temperature", "0", "--threads", "2"])
        .args(["--no-profile", "--quiet", "--json"])
        .args(extra)
        .arg("Hello there")
        .output()
        .expect("run peregrine");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("one JSON object")
}

#[test]
fn a_qwen35_template_has_its_reasoning_block_opened_by_the_prompt() {
    let dir = tmpdir("qwen35");
    let model = write_dense_with_template(
        &dir,
        "qwen2",
        Format::F16,
        51,
        shape(),
        peregrine_core::tools::QWEN35_TEMPLATE,
    );

    let on = run(&model, &[]);
    let text = on["text"].as_str().unwrap();
    assert!(text.starts_with("<think>\n"), "{text:?}");
    assert!(text.len() > "<think>\n".len(), "{text:?}");

    // Thinking off: the template closes an empty block in the prompt, so the reply has no
    // tag to show and the prompt is longer by that block's closing.
    let off = run(&model, &["--no-thinking"]);
    let text = off["text"].as_str().unwrap();
    assert!(!text.starts_with("<think>"), "{text:?}");
    assert!(
        off["prompt_tokens"].as_u64().unwrap() > on["prompt_tokens"].as_u64().unwrap(),
        "{off} against {on}"
    );

    // The same request is the same reply.
    assert_eq!(run(&model, &[])["text"], on["text"]);
}

#[test]
fn another_template_is_printed_as_the_model_wrote_it() {
    let dir = tmpdir("plain");
    let model = write_dense(&dir, "qwen2", Format::F16, 52, shape());
    let text = run(&model, &[])["text"].as_str().unwrap().to_string();
    assert!(!text.starts_with("<think>"), "{text:?}");
}
