//! `peregrine embed` end to end on a synthetic embedding model (roadmap S4.1).
//!
//! The vector rules are pinned in `peregrine-models/tests/embeddings.rs`; this file pins the
//! command: one line of space-separated floats per text, `--dimensions`, `--json`, stdin, the
//! over-length error and `inspect`'s pooling line.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use support::models::{
    write_dense_pooled, write_gemma_embedding, write_jina_bert, write_modern_bert,
    write_nomic_bert, Dense, GemmaEmbeddingConfig, JinaBertConfig, ModernBertConfig,
    NomicBertConfig,
};

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-cli-embed-{}-{test}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_peregrine"))
}

/// The synthetic model's width (`peregrine-models/tests/support/models.rs`).
const WIDTH: usize = 256;

fn floats(line: &str) -> Vec<f32> {
    line.split_whitespace()
        .map(|x| x.parse::<f32>().expect("a float"))
        .collect()
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

fn embed(model: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin())
        .arg("embed")
        .arg("-m")
        .arg(model)
        .args(["--no-profile", "-t", "2"])
        .args(args)
        .output()
        .expect("run peregrine embed")
}

#[test]
fn embed_prints_one_normalised_vector_per_text_in_input_order() {
    let dir = tmpdir("lines");
    let model = write_dense_pooled(&dir, "llama", Format::F16, 21, Dense::default(), 3);
    let out = embed(&model, &["alpha", "a longer second text", "third"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3, "one line per text: {text:?}");
    for line in &lines {
        let v = floats(line);
        assert_eq!(v.len(), WIDTH);
        assert!((norm(&v) - 1.0).abs() < 1e-5, "normalised: {line:?}");
    }

    // The same text twice in one command gives the same bits, and its position in the batch
    // does not change it.
    let out = embed(&model, &["alpha", "a longer second text", "third", "alpha"]);
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], lines[3], "the same text has one vector");
}

#[test]
fn dimensions_truncate_then_normalise_and_json_has_the_api_shape() {
    let dir = tmpdir("json");
    let model = write_dense_pooled(&dir, "llama", Format::F16, 22, Dense::default(), 3);
    let out = embed(&model, &["--dimensions", "16", "--json", "hello", "world"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let body: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim()).expect("JSON");
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data");
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);
    let v: Vec<f32> = serde_json::from_value(data[0]["embedding"].clone()).expect("floats");
    assert_eq!(v.len(), 16, "dimensions before normalising");
    assert!((norm(&v) - 1.0).abs() < 1e-5);
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        body["usage"]["total_tokens"], body["usage"]["prompt_tokens"],
        "an embedding has no completion tokens"
    );
}

#[test]
fn stdin_lines_are_the_texts_when_none_are_given() {
    let dir = tmpdir("stdin");
    let model = write_dense_pooled(&dir, "llama", Format::F16, 23, Dense::default(), 3);
    let mut child = Command::new(bin())
        .arg("embed")
        .arg("-m")
        .arg(&model)
        .args(["--no-profile", "-t", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"one\ntwo\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{text:?}");
    for line in lines {
        assert_eq!(floats(line).len(), WIDTH);
    }
}

#[test]
fn an_over_length_text_is_an_error_naming_the_limit_unless_truncated() {
    let dir = tmpdir("length");
    let model = write_dense_pooled(&dir, "llama", Format::F16, 24, Dense::default(), 3);
    let long: String = "ab ".repeat(200);
    let out = embed(&model, &["-c", "32", &long]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("32 positions"), "stderr: {err}");
    assert!(err.contains("--truncate"), "names the opt-in: {err}");

    let out = embed(&model, &["-c", "32", "--truncate", &long]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        floats(String::from_utf8_lossy(&out.stdout).trim()).len(),
        WIDTH
    );
}

#[test]
fn inspect_names_the_pooling_type_and_a_model_without_one_is_refused() {
    let dir = tmpdir("inspect");
    let pooled = write_dense_pooled(&dir, "llama", Format::F16, 25, Dense::default(), 3);
    let out = Command::new(bin())
        .arg("inspect")
        .arg(&pooled)
        .output()
        .expect("run inspect");
    assert!(out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("pooling       last (3)"),
        "inspect: {}",
        String::from_utf8_lossy(&out.stdout)
    );

    // A file that declares no pooling type has no default vector: the command says which way
    // out (`--pooling`), and refuses the file's own "none".
    let plain = support::models::write_dense(&dir, "llama", Format::F16, 26, Dense::default());
    let out = embed(&plain, &["hello"]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("pooling_type"), "stderr: {err}");
    assert!(err.contains("--pooling"), "stderr: {err}");

    // The override is enough, and `cls` is refused for causal models.
    let out = embed(&plain, &["--pooling", "mean", "hello"]);
    assert!(out.status.success(), "{out:?}");
    let out = embed(&plain, &["--pooling", "cls", "hello"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cls"));
}

/// A bidirectional encoder (roadmap S4.4): `embed` works on it, `inspect` says what it is, and
/// `run` refuses to generate from it.
#[test]
fn an_encoder_embeds_and_refuses_to_generate() {
    let dir = tmpdir("encoder");
    let model = write_gemma_embedding(&dir, Format::F16, 27, GemmaEmbeddingConfig::default());
    let out = embed(&model, &["hello", "a longer second text"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    for line in &lines {
        let v = floats(line);
        assert_eq!(v.len(), 128);
        assert!((norm(&v) - 1.0).abs() < 1e-5);
    }
    // `cls` is legal for an encoder.
    let out = embed(&model, &["--pooling", "cls", "hello"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let out = Command::new(bin())
        .arg("inspect")
        .arg(&model)
        .output()
        .expect("run inspect");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("pooling       mean (1) (embedding model)"),
        "{text}"
    );

    let out = Command::new(bin())
        .args(["run", "-m"])
        .arg(&model)
        .args(["--no-profile", "-t", "2", "hello"])
        .output()
        .expect("run peregrine run");
    assert!(!out.status.success(), "a chat run on an encoder must fail");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("embedding model"), "stderr: {err}");
}

/// A synthetic `modern-bert` encoder (roadmap S4.7): `embed` works over the
/// GPT-2 vocabulary (which adds a BOS and an EOS), `inspect` prints the file's
/// mean pooling, and `run` refuses to generate.
#[test]
fn a_modern_bert_encoder_embeds_and_refuses_to_generate() {
    let dir = tmpdir("modern-bert");
    let model = write_modern_bert(&dir, Format::F16, 29, ModernBertConfig::default());
    let out = embed(&model, &["hello world", "the test of a paris city"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    for line in &lines {
        let v = floats(line);
        assert_eq!(v.len(), 128);
        assert!((norm(&v) - 1.0).abs() < 1e-5);
    }
    let out = Command::new(bin())
        .arg("inspect")
        .arg(&model)
        .output()
        .expect("run inspect");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("pooling       mean (1) (embedding model)"),
        "{text}"
    );
    let out = Command::new(bin())
        .args(["run", "-m"])
        .arg(&model)
        .args(["--no-profile", "-t", "2", "hello"])
        .output()
        .expect("run peregrine run");
    assert!(!out.status.success(), "a chat run on an encoder must fail");
    assert!(String::from_utf8_lossy(&out.stderr).contains("embedding model"));
}

/// A synthetic `nomic-bert` encoder (roadmap S4.8a): `embed` works over the
/// WordPiece vocabulary (`[CLS] … [SEP]`), `--dimensions` truncates then
/// normalises, `inspect` prints the file's mean pooling, and `run` refuses.
#[test]
fn a_nomic_bert_encoder_embeds_and_refuses_to_generate() {
    let dir = tmpdir("nomic-bert");
    let model = write_nomic_bert(&dir, Format::F16, 31, NomicBertConfig::default());
    let out = embed(&model, &["hello world", "the test of a paris city"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    for line in &lines {
        let v = floats(line);
        assert_eq!(v.len(), 128);
        assert!((norm(&v) - 1.0).abs() < 1e-5);
    }
    let out = embed(&model, &["--dimensions", "32", "--json", "hello"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let data: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    let v: Vec<f32> = serde_json::from_value(data["data"][0]["embedding"].clone()).unwrap();
    assert_eq!(v.len(), 32);
    assert!((norm(&v) - 1.0).abs() < 1e-5);
    let out = Command::new(bin())
        .arg("inspect")
        .arg(&model)
        .output()
        .expect("run inspect");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("pooling       mean (1) (embedding model)"),
        "{text}"
    );
    let out = Command::new(bin())
        .args(["run", "-m"])
        .arg(&model)
        .args(["--no-profile", "-t", "2", "hello"])
        .output()
        .expect("run peregrine run");
    assert!(!out.status.success(), "a chat run on an encoder must fail");
    assert!(String::from_utf8_lossy(&out.stderr).contains("embedding model"));
}

/// A synthetic `bert` encoder (roadmap S4.6a): `embed` works, `inspect` prints the
/// file's pooling and calls it an embedding model, and `run` refuses to generate.
#[test]
fn a_bert_encoder_embeds_and_refuses_to_generate() {
    let dir = tmpdir("bert");
    let model = support::models::write_bert(
        &dir,
        Format::F16,
        28,
        support::models::BertConfig::default(),
    );
    let out = embed(&model, &["hello world", "the test of a paris city"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    for line in &lines {
        let v = floats(line);
        assert_eq!(v.len(), 128);
        assert!((norm(&v) - 1.0).abs() < 1e-5);
    }
    let out = Command::new(bin())
        .arg("inspect")
        .arg(&model)
        .output()
        .expect("run inspect");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("pooling       cls (2) (embedding model)"),
        "{text}"
    );
    let out = Command::new(bin())
        .args(["run", "-m"])
        .arg(&model)
        .args(["--no-profile", "-t", "2", "hello"])
        .output()
        .expect("run peregrine run");
    assert!(!out.status.success(), "a chat run on an encoder must fail");
    assert!(String::from_utf8_lossy(&out.stderr).contains("embedding model"));
}

/// A synthetic `jina-bert-v2` encoder (roadmap S4.8c): `embed` works over the
/// WordPiece vocabulary (`[CLS] … [SEP]`), `inspect` prints the file's mean pooling,
/// and `run` refuses to generate.
#[test]
fn a_jina_bert_encoder_embeds_and_refuses_to_generate() {
    let dir = tmpdir("jina-bert");
    let model = write_jina_bert(&dir, Format::F16, 30, JinaBertConfig::default());
    let out = embed(&model, &["hello world", "the test of a paris city"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    for line in &lines {
        let v = floats(line);
        assert_eq!(v.len(), 128);
        assert!((norm(&v) - 1.0).abs() < 1e-5);
    }
    let out = Command::new(bin())
        .arg("inspect")
        .arg(&model)
        .output()
        .expect("run inspect");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("pooling       mean (1) (embedding model)"),
        "{text}"
    );
    let out = Command::new(bin())
        .args(["run", "-m"])
        .arg(&model)
        .args(["--no-profile", "-t", "2", "hello"])
        .output()
        .expect("run peregrine run");
    assert!(!out.status.success(), "a chat run on an encoder must fail");
    assert!(String::from_utf8_lossy(&out.stderr).contains("embedding model"));
}
