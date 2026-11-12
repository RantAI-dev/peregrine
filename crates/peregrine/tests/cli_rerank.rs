//! `peregrine rerank` end to end on a synthetic reranker (roadmap S4.6b).
//!
//! The score rules are pinned in `peregrine-models/tests/bert_heads.rs` and the endpoint in
//! `server_rerank.rs`; this file pins the command: one line per document, highest score
//! first, `--top-n`, `--json`, stdin, the over-length error and the refusal of a file
//! without a head.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use support::models::{write_bert, write_modern_bert, BertConfig, BertHeadConfig};

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-cli-rerank-{}-{test}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&d);
    d
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_peregrine"))
}

/// A synthetic reranker: one raw score, no labels.
fn reranker(dir: &Path) -> PathBuf {
    write_bert(
        dir,
        Format::F32,
        81,
        BertConfig {
            pooling_type: 4,
            head: Some(BertHeadConfig {
                n_out: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
    )
}

fn rerank(model: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin())
        .arg("rerank")
        .arg("-m")
        .arg(model)
        .args(["--no-profile", "-t", "2"])
        .args(args)
        .output()
        .expect("run peregrine rerank")
}

/// `(score, index, document)` per printed line.
fn lines(out: &str) -> Vec<(f32, usize, String)> {
    out.lines()
        .map(|l| {
            let mut parts = l.splitn(3, ' ');
            let score = parts.next().unwrap().parse::<f32>().expect("a score");
            let index = parts.next().unwrap().parse::<usize>().expect("an index");
            (score, index, parts.next().unwrap_or("").to_string())
        })
        .collect()
}

#[test]
fn rerank_prints_the_documents_highest_score_first() {
    let dir = tmpdir("lines");
    let model = reranker(&dir);
    let out = rerank(
        &model,
        &["--query", "hello world", "the cat", "hello world", "taxes"],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    let rows = lines(&text);
    assert_eq!(rows.len(), 3, "{text}");
    // Descending, every document once, and the text kept.
    let mut seen: Vec<usize> = rows.iter().map(|r| r.1).collect();
    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1, 2]);
    for w in rows.windows(2) {
        assert!(w[0].0 >= w[1].0, "not sorted: {text}");
    }
    for r in &rows {
        let want = ["the cat", "hello world", "taxes"][r.1];
        assert_eq!(r.2, want);
    }

    // `--top-n` keeps the first N lines; the same text alone keeps the same bits.
    let out = rerank(
        &model,
        &["--query", "hello world", "--top-n", "1", "hello world"],
    );
    assert!(out.status.success());
    let one = lines(&String::from_utf8(out.stdout).unwrap());
    assert_eq!(one.len(), 1);
    let alone = rows.iter().find(|r| r.1 == 1).unwrap().0;
    assert_eq!(one[0].0.to_bits(), alone.to_bits());

    // `--json` is the API's object, in the same order.
    let out = rerank(
        &model,
        &[
            "--query",
            "hello world",
            "--json",
            "the cat",
            "hello world",
            "taxes",
        ],
    );
    assert!(out.status.success());
    let body: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(body["object"], "list");
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["index"], rows[0].1);
    // The text line prints six decimals; the JSON carries the f32 itself.
    assert!(
        (results[0]["relevance_score"].as_f64().unwrap() as f32 - rows[0].0).abs() < 1e-6,
        "{} vs {}",
        results[0]["relevance_score"],
        rows[0].0
    );
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

#[test]
fn rerank_reads_stdin_and_refuses_what_it_cannot_do() {
    let dir = tmpdir("stdin");
    let model = reranker(&dir);
    let mut child = Command::new(bin())
        .arg("rerank")
        .arg("-m")
        .arg(&model)
        .args(["--no-profile", "-t", "2", "--query", "hello"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn peregrine rerank");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"one document\nanother document\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8(out.stdout).unwrap().lines().count(), 2);

    // A pair past the model's position table is refused by name; `--truncate` accepts it.
    let long = "hello ".repeat(200);
    let out = rerank(&model, &["--query", "hello", &long]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("64 positions") && err.contains("--truncate"),
        "{err}"
    );
    let out = rerank(&model, &["--query", "hello", "--truncate", &long]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // A file without a head is refused by name, pointing at `peregrine embed`.
    let plain = write_bert(&dir, Format::F32, 82, BertConfig::default());
    let out = rerank(&plain, &["--query", "hello", "a document"]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("classifier head") && err.contains("peregrine embed"),
        "{err}"
    );
}

/// The `modern-bert` family (roadmap S4.7b) reaches the same command without a change:
/// its head is mean-pooled and erf-GELU activated by default.
#[test]
fn rerank_works_on_a_modern_bert_reranker() {
    let dir = tmpdir("modern-bert");
    let path = write_modern_bert(
        &dir,
        Format::F32,
        82,
        support::models::ModernBertConfig {
            head: Some(support::models::ModernBertHeadConfig {
                n_out: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    let to = dir.join("mb-reranker.gguf");
    std::fs::rename(&path, &to).unwrap();
    let out = rerank(&to, &["--query", "hello world", "the cat", "hello world"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    let rows = lines(&text);
    assert_eq!(rows.len(), 2, "{text}");
    assert!(rows[0].0 >= rows[1].0);
    assert_eq!(rows[0].2, "the cat");
}
