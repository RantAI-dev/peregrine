//! `peregrine decide` end to end on a synthetic decision model (roadmap S4.3c).
//!
//! The API's shapes are pinned in `server_systemone.rs`; this file pins the command: one line per
//! question, `--json`'s response shape, the three ways to pass a state and a questions object
//! (`--state`/`--questions`, `@file`, `-`), `--request`, and a chat model refusing the command.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use peregrine_core::json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use support::models::{write_hybrid_decision, write_hybrid_plain, DecisionSpec};

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-cli-decide-{}-{test}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_peregrine"))
}

fn model(dir: &Path) -> PathBuf {
    write_hybrid_decision(
        dir,
        Format::F16,
        51,
        &DecisionSpec::openjev().keys("qwen35"),
    )
}

fn decide(model: &Path, args: &[&str]) -> Output {
    Command::new(bin())
        .arg("decide")
        .arg("-m")
        .arg(model)
        .args(["--no-profile", "-t", "2", "-c", "512"])
        .args(args)
        .output()
        .expect("run peregrine decide")
}

fn decide_stdin(model: &Path, args: &[&str], stdin: &str) -> Output {
    let mut child = Command::new(bin())
        .arg("decide")
        .arg("-m")
        .arg(model)
        .args(["--no-profile", "-t", "2", "-c", "512"])
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn peregrine decide");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    child.wait_with_output().expect("wait")
}

const QUESTIONS: &str = r#"{"pick":{"type":"choice","instructions":"which one","criteria":{"zeta":null,"alpha":{"k":1}}},"rate":{"type":"score","instructions":"how much","criteria":["low","mid","high"]},"yes":{"type":"noul","instructions":"is it so?"}}"#;

#[test]
fn it_prints_one_line_per_question_in_request_order() {
    let dir = tmpdir("lines");
    let model = model(&dir);
    let out = decide(
        &model,
        &[
            "--state",
            "a plain state with a few words",
            "--questions",
            QUESTIONS,
        ],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3, "{text:?}");
    assert!(lines[0].starts_with("pick: "), "{text:?}");
    let choice = &lines[0]["pick: ".len()..];
    let (key, confidence) = choice
        .split_once(" (")
        .unwrap_or_else(|| panic!("no confidence: {choice:?}"));
    assert!(["zeta", "alpha"].contains(&key));
    assert!(confidence.ends_with(')'));
    confidence[..confidence.len() - 1].parse::<f64>().unwrap();
    let score = lines[1]
        .strip_prefix("rate: ")
        .unwrap_or_else(|| panic!("{text:?}"));
    assert!((0.0..=2.0).contains(&score.parse::<f64>().unwrap()));
    let noul = lines[2]
        .strip_prefix("yes: ")
        .unwrap_or_else(|| panic!("{text:?}"));
    assert!((0.0..=1.0).contains(&noul.parse::<f64>().unwrap()));
}

#[test]
fn json_is_the_api_shape_and_matches_the_one_question_run() {
    let dir = tmpdir("json");
    let model = model(&dir);
    let out = decide(
        &model,
        &[
            "--state",
            "a plain state with a few words",
            "--questions",
            QUESTIONS,
            "--json",
        ],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let body = Value::parse(&String::from_utf8_lossy(&out.stdout)).expect("JSON");
    let top: Vec<&str> = body
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(top, ["model", "answers", "usage"]);
    let answers = body.get("answers").unwrap();
    let ids: Vec<&str> = answers
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(ids, ["pick", "rate", "yes"]);
    let pick = answers.get("pick").unwrap();
    let keys: Vec<&str> = pick
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(keys, ["type", "choice", "probabilities", "confidence"]);
    let usage = body.get("usage").unwrap();
    assert!(
        match usage.get("input_tokens").unwrap() {
            Value::Int(i) => *i > 0,
            Value::Float(f) => *f > 0.0,
            _ => panic!("input_tokens"),
        },
        "usage: {usage:?}"
    );
    assert!(matches!(usage.get("output_tokens"), Some(Value::Int(0))));

    // The same request one question at a time: each answer is exactly the one in the batch, and
    // the batch's input_tokens is their sum.
    let mut alone = 0i64;
    for (id, one) in [
        (
            "pick",
            r#"{"pick":{"type":"choice","instructions":"which one","criteria":{"zeta":null,"alpha":{"k":1}}}}"#,
        ),
        (
            "rate",
            r#"{"rate":{"type":"score","instructions":"how much","criteria":["low","mid","high"]}}"#,
        ),
        (
            "yes",
            r#"{"yes":{"type":"noul","instructions":"is it so?"}}"#,
        ),
    ] {
        let out = decide(
            &model,
            &[
                "--state",
                "a plain state with a few words",
                "--questions",
                one,
                "--json",
            ],
        );
        assert!(
            out.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let body = Value::parse(&String::from_utf8_lossy(&out.stdout)).expect("JSON");
        assert_eq!(body.get("answers").unwrap().get(id), answers.get(id));
        alone += match body.get("usage").unwrap().get("input_tokens").unwrap() {
            Value::Int(i) => *i,
            other => panic!("input_tokens: {other:?}"),
        };
    }
    let batch = match usage.get("input_tokens").unwrap() {
        Value::Int(i) => *i,
        other => panic!("input_tokens: {other:?}"),
    };
    assert_eq!(batch, alone);
}

#[test]
fn state_questions_and_request_read_files_and_stdin() {
    let dir = tmpdir("sources");
    let model = model(&dir);
    let state_file = dir.join("state.txt");
    std::fs::write(&state_file, "a plain state with a few words").unwrap();
    let questions_file = dir.join("questions.json");
    std::fs::write(&questions_file, QUESTIONS).unwrap();
    let request_file = dir.join("request.json");
    std::fs::write(
        &request_file,
        format!(r#"{{"state":"a plain state with a few words","questions":{QUESTIONS}}}"#),
    )
    .unwrap();

    let direct = decide(
        &model,
        &[
            "--state",
            "a plain state with a few words",
            "--questions",
            QUESTIONS,
            "--json",
        ],
    );
    assert!(direct.status.success());
    let want = Value::parse(&String::from_utf8_lossy(&direct.stdout)).expect("JSON");

    let from_files = decide(
        &model,
        &[
            "--state",
            &format!("@{}", state_file.display()),
            "--questions",
            &format!("@{}", questions_file.display()),
            "--json",
        ],
    );
    assert!(
        from_files.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&from_files.stderr)
    );
    let got = Value::parse(&String::from_utf8_lossy(&from_files.stdout)).expect("JSON");
    assert_eq!(got.get("answers"), want.get("answers"));
    assert_eq!(got.get("usage"), want.get("usage"));

    let from_request = decide(
        &model,
        &[
            "--request",
            &format!("@{}", request_file.display()),
            "--json",
        ],
    );
    assert!(
        from_request.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&from_request.stderr)
    );
    let got = Value::parse(&String::from_utf8_lossy(&from_request.stdout)).expect("JSON");
    assert_eq!(got.get("answers"), want.get("answers"));

    // `-` is stdin, both for the state and for the whole request.
    let from_stdin = decide_stdin(
        &model,
        &["--state", "-", "--questions", QUESTIONS, "--json"],
        "a plain state with a few words",
    );
    assert!(
        from_stdin.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&from_stdin.stderr)
    );
    let got = Value::parse(&String::from_utf8_lossy(&from_stdin.stdout)).expect("JSON");
    assert_eq!(got.get("answers"), want.get("answers"));

    let request =
        format!(r#"{{"state":"a plain state with a few words","questions":{QUESTIONS}}}"#);
    let from_stdin = decide_stdin(&model, &["--request", "-", "--json"], &request);
    assert!(
        from_stdin.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&from_stdin.stderr)
    );
    let got = Value::parse(&String::from_utf8_lossy(&from_stdin.stdout)).expect("JSON");
    assert_eq!(got.get("answers"), want.get("answers"));
}

#[test]
fn a_chat_model_refuses_the_command() {
    let dir = tmpdir("refusal");
    let model = write_hybrid_plain(&dir, Format::F16, 53);
    let out = decide(
        &model,
        &[
            "--state",
            "a plain state",
            "--questions",
            r#"{"a":{"type":"noul","instructions":"is it so?"}}"#,
        ],
    );
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not a decision model"), "stderr: {stderr}");
}
