//! The real `/v1/systemone` endpoint against llama.cpp v0.9.0's stored answers (roadmap S4.3c).
//!
//! `crates/peregrine-core/tests/fixtures/systemone-openjev.jsonl` holds 48 requests and the
//! answers llama.cpp gave on `tinyopenjev-for-testing-Q8_0.gguf`. With that file under `models/`
//! or `~/models/`, the real `peregrine serve` binary answers every request and this test compares
//! each answer with llama.cpp's (same option; probabilities, score and `noul` within 5e-3;
//! confidence within 1e-2) and `usage.input_tokens` with the fixture's token count. Without the
//! file it is ignored, and CI has no model files.
//!
//! Run it with:
//! `cargo test -p peregrine --release --test systemone_real -- --ignored --nocapture`.
//!
//! No request, prompt or fixture line is printed; a failure names the case number only.

mod common;

use peregrine_core::json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::thread;
use std::time::{Duration, Instant};

const FILE: &str = "tinyopenjev-for-testing-Q8_0.gguf";

/// Where model files live: `models/` in the repository, then `~/models/`.
fn model_file(name: &str) -> Option<PathBuf> {
    let mut dirs = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")];
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(Path::new(&home).join("models"));
    }
    dirs.into_iter().map(|d| d.join(name)).find(|p| p.exists())
}

struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn try_http(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(300))).ok();
    let payload = body.unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(request.as_bytes()).ok()?;
    let mut out = String::new();
    stream.read_to_string(&mut out).ok()?;
    let status = out
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body = out
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    Some((status, body))
}

fn serve(model: &Path) -> (Guard, u16) {
    let (child, port) = common::start(
        &std::env::temp_dir(),
        &[],
        &[
            "serve",
            "-m",
            model.to_str().unwrap(),
            "--host",
            "127.0.0.1",
            "--port",
            common::PORT,
            "--ctx",
            "4096",
            "--threads",
            "4",
            "--slots",
            "4",
            "--no-profile",
        ],
    );
    let guard = Guard(child);
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if matches!(try_http(port, "GET", "/health", None), Some((200, _))) {
            return (guard, port);
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!("the server did not answer /health");
}

fn number(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        other => panic!("not a number: {other:?}"),
    }
}

/// The largest difference between two `(key, probability)` objects, whose keys must match in
/// order.
fn probability_gap(ours: &Value, theirs: &Value) -> f64 {
    let ours = ours.as_object().expect("probabilities");
    let theirs = theirs.as_object().expect("probabilities");
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|((key, p), (their_key, their_p))| {
            assert_eq!(key, their_key);
            (number(p) - number(their_p)).abs()
        })
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "needs tinyopenjev-for-testing-Q8_0.gguf in models/ or ~/models/"]
fn the_endpoint_answers_the_fixture_like_llama_cpp() {
    let Some(path) = model_file(FILE) else {
        eprintln!("  SKIPPED: {FILE} not found under models/ or ~/models/");
        return;
    };
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../peregrine-core/tests/fixtures/systemone-openjev.jsonl");
    let fixture = std::fs::read_to_string(&fixture_path).expect("the fixture file");

    let (_guard, port) = serve(&path);
    let (mut worst_value, mut worst_confidence) = (0f64, 0f64);
    let mut cases = 0usize;
    for (n, line) in fixture.lines().enumerate() {
        let case = Value::parse(line).expect("a fixture line");
        let state = case.get("state").expect("state").clone();
        let question = case.get("question").expect("question").clone();
        let body = Value::Obj(vec![
            ("state".into(), state),
            ("questions".into(), Value::Obj(vec![("q".into(), question)])),
        ]);
        let (status, text) = try_http(port, "POST", "/v1/systemone", Some(&body.to_jinja_json()))
            .expect("the server answers");
        assert_eq!(status, 200, "case {}: status {status}", n + 1);
        let out = Value::parse(&text).expect("a response");
        let answer = out
            .get("answers")
            .and_then(|a| a.get("q"))
            .unwrap_or_else(|| panic!("case {}: no answer", n + 1));
        let want = case.get("answer").expect("llama.cpp's answer");

        match answer.get("type").and_then(Value::as_str) {
            Some("choice") => {
                assert_eq!(
                    answer.get("choice").and_then(Value::as_str),
                    want.get("choice").and_then(Value::as_str),
                    "case {}: a different option",
                    n + 1
                );
                worst_value = worst_value.max(probability_gap(
                    answer.get("probabilities").unwrap(),
                    want.get("probabilities").unwrap(),
                ));
                worst_confidence = worst_confidence.max(
                    (number(answer.get("confidence").unwrap())
                        - number(want.get("confidence").unwrap()))
                    .abs(),
                );
            }
            Some("score") => {
                worst_value = worst_value
                    .max(probability_gap(
                        answer.get("probabilities").unwrap(),
                        want.get("probabilities").unwrap(),
                    ))
                    .max(
                        (number(answer.get("score").unwrap()) - number(want.get("score").unwrap()))
                            .abs(),
                    );
                worst_confidence = worst_confidence.max(
                    (number(answer.get("confidence").unwrap())
                        - number(want.get("confidence").unwrap()))
                    .abs(),
                );
            }
            Some("noul") => {
                worst_value = worst_value.max(
                    (number(answer.get("noul").unwrap()) - number(want.get("noul").unwrap())).abs(),
                );
            }
            other => panic!("case {}: unexpected answer type {other:?}", n + 1),
        }

        let tokens = case
            .get("tokens")
            .and_then(Value::as_array)
            .expect("tokens");
        let usage = out.get("usage").expect("usage");
        assert_eq!(
            number(usage.get("input_tokens").unwrap()) as usize,
            tokens.len(),
            "case {}: usage.input_tokens",
            n + 1
        );
        assert_eq!(
            number(usage.get("output_tokens").unwrap()),
            0.0,
            "case {}: output_tokens",
            n + 1
        );
        cases += 1;
    }
    eprintln!(
        "  /v1/systemone vs llama.cpp, {cases} cases: values within {worst_value:.1e}, confidence within {worst_confidence:.1e}"
    );
    assert!(cases >= 40);
    assert!(worst_value < 5e-3, "values differ by {worst_value:e}");
    assert!(
        worst_confidence < 1e-2,
        "confidence differs by {worst_confidence:e}"
    );
}
