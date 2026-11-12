//! `peregrine run` on a file that carries Qwen 3.5's chat template (roadmap S6.3c).
//!
//! That template opens the reasoning block at the end of the prompt, so the command shows
//! the opening tag itself and the model writes the rest. A synthetic model writes random
//! tokens, so these tests pin the prompt's two endings and the tag, not what a model says;
//! the prompt's bytes are pinned against llama.cpp in `peregrine-core` (`template.rs` and
//! `tests/tools_render.rs`).

#[path = "../../peregrine-core/tests/support/mod.rs"]
mod check;
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

/// `peregrine run --json-schema`: the reply is a complete document of the schema, and the
/// same model with `--json-object` produces a prefix of an object.
#[test]
fn the_cli_can_constrain_a_reply_to_a_json_schema_or_an_object() {
    let dir = tmpdir("constrain");
    let model = write_dense(&dir, "qwen2", Format::F16, 53, shape());
    let schema_text = r#"{"type":"object","properties":{"answer":{"type":"integer","minimum":0,"maximum":1}},"required":["answer"],"additionalProperties":false}"#;
    let schema_path = dir.join("schema.json");
    std::fs::write(&schema_path, schema_text).unwrap();

    let out = run_custom(
        &model,
        &[
            "--json-schema",
            schema_path.to_str().unwrap(),
            "--max-tokens",
            "256",
            "--temperature",
            "0",
        ],
    );
    let text = out["text"].as_str().unwrap();
    assert_valid(text, schema_text);
    let value = check::parse(text).unwrap();
    assert!(matches!(value, check::Json::Object(_)), "{text:?}");

    let object = run_custom(
        &model,
        &["--json-object", "--max-tokens", "256", "--temperature", "0"],
    );
    let text = object["text"].as_str().unwrap();
    let c = peregrine_core::constrain::Constraint::json_object();
    let mut m = peregrine_core::constrain::Matcher::new(&c).unwrap();
    assert!(
        m.advance_bytes(text.as_bytes()).is_ok(),
        "the CLI reply is not a prefix of an object: {text:?}"
    );
}

#[test]
fn the_cli_refuses_an_unsupported_schema_keyword_and_a_thinking_model() {
    let dir = tmpdir("constrain-errors");
    let model = write_dense(&dir, "qwen2", Format::F16, 54, shape());
    let bad = dir.join("bad.json");
    std::fs::write(
        &bad,
        r#"{"type":"object","properties":{"x":{"type":"string","pattern":"a+"}}}"#,
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_peregrine"))
        .args(["run", "-m", model.to_str().unwrap(), "--no-profile"])
        .args(["--json-schema", bad.to_str().unwrap()])
        .arg("Hello there")
        .output()
        .expect("run peregrine");
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("pattern") && err.contains("#/properties/x"),
        "{err}"
    );

    let thinking = write_dense_with_template(
        &dir,
        "qwen2",
        Format::F16,
        55,
        shape(),
        peregrine_core::tools::QWEN35_TEMPLATE,
    );
    let good = dir.join("good.json");
    std::fs::write(&good, r#"{"type":"object"}"#).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_peregrine"))
        .args(["run", "-m", thinking.to_str().unwrap(), "--no-profile"])
        .args(["--json-schema", good.to_str().unwrap(), "--max-tokens", "8"])
        .arg("Hello there")
        .output()
        .expect("run peregrine");
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("response_format is not supported while the model is thinking"),
        "{err}"
    );

    // Thinking off serves it.
    let out = Command::new(env!("CARGO_BIN_EXE_peregrine"))
        .args(["run", "-m", thinking.to_str().unwrap(), "--no-profile"])
        .args([
            "--json-schema",
            good.to_str().unwrap(),
            "--max-tokens",
            "160",
            "--no-thinking",
            "--json",
            "--quiet",
        ])
        .arg("Hello there")
        .output()
        .expect("run peregrine");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `peregrine run` with only the output flags, so a test can set its own token budget.
fn run_custom(model: &Path, extra: &[&str]) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_peregrine"))
        .args(["run", "-m", model.to_str().unwrap(), "--no-profile"])
        .args(["--quiet", "--json", "--threads", "2"])
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

fn assert_valid(text: &str, schema_text: &str) {
    let value = check::parse(text).unwrap_or_else(|e| panic!("not JSON ({e}): {text:?}"));
    let schema = check::parse(schema_text).unwrap();
    check::validate(&value, &schema, &schema)
        .unwrap_or_else(|e| panic!("does not validate ({e}): {text:?}"));
}
