//! `POST /v1/systemone` end to end on synthetic decision models (roadmap S4.3c).
//!
//! The answer math is pinned in `peregrine-core` and the real-file parity in
//! `crates/peregrine-models/tests/systemone_openjev.rs`; this file pins the endpoint: the three
//! question types, the request's order, `usage`, every validation message, the 501s, a request
//! with more questions than slots, `/v1/models`, and a chat model refusing the route.
//!
//! A synthetic `qwen35` file carries `decision.type = openjev`, the real `systemone` template (from
//! `peregrine_core::systemone::OPENJEV_TEMPLATE`) and the published temperatures. Requests here
//! use plain words; no rendered prompt is printed.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use peregrine_core::json::Value;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use support::models::{write_hybrid_decision, write_hybrid_plain, DecisionSpec};

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-server-systemone-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// Kill the server even when an assertion fails first.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
}

/// One HTTP/1.1 exchange; `None` when the server is not listening yet.
fn try_http(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(60))).ok();
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

fn post(port: u16, body: &str) -> (u16, Value) {
    let (status, text) = try_http(port, "POST", "/v1/systemone", Some(body))
        .expect("the server is listening");
    let value = Value::parse(&text).unwrap_or(Value::Null);
    (status, value)
}

fn wait_for_health(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if matches!(try_http(port, "GET", "/health", None), Some((200, _))) {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("the server did not answer /health");
}

/// Serve one synthetic file; returns the guard and the port.
fn serve(model: &Path, slots: usize) -> (Guard, u16) {
    let port = free_port();
    let dir = model.parent().expect("a model directory");
    let log = dir.join(format!("serve-{port}.log"));
    let log_file = std::fs::File::create(&log).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_peregrine"))
        .args([
            "serve",
            "-m",
            model.to_str().unwrap(),
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--ctx",
            "512",
            "--threads",
            "2",
            "--slots",
            &slots.to_string(),
            "--no-profile",
        ])
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .spawn()
        .expect("spawn peregrine serve");
    let guard = Guard(child);
    wait_for_health(port);
    (guard, port)
}

fn decision_model(dir: &Path) -> PathBuf {
    write_hybrid_decision(
        dir,
        Format::F16,
        41,
        &DecisionSpec::openjev().keys("qwen35"),
    )
}

fn answer<'a>(body: &'a Value, id: &str) -> &'a Value {
    body.get("answers")
        .and_then(|a| a.get(id))
        .unwrap_or_else(|| panic!("no answer {id:?}"))
}

fn number(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        other => panic!("not a number: {other:?}"),
    }
}

fn keys(v: &Value) -> Vec<&str> {
    v.as_object()
        .expect("an object")
        .iter()
        .map(|(k, _)| k.as_str())
        .collect()
}

#[test]
fn three_question_types_come_back_in_request_order_with_usage() {
    let dir = tmpdir("three");
    let model = decision_model(&dir);
    let (_guard, port) = serve(&model, 4);

    let body = r#"{
        "state": "a plain state with a few words",
        "questions": {
            "pick": {"type": "choice", "instructions": "which one",
                     "criteria": {"zeta": null, "alpha": {"k": 1}, "mid": "middle"}},
            "rate": {"type": "score", "instructions": "how much",
                     "criteria": ["low", "mid", "high"]},
            "yes":  {"type": "noul", "instructions": "is it so?",
                     "criteria": {"true": "yes", "false": "no"}}
        }
    }"#;
    let (status, out) = post(port, body);
    assert_eq!(status, 200, "{out:?}");
    assert_eq!(out.get("model"), Some(&Value::Str("peregrine".into())));
    let top: Vec<&str> = keys(&out);
    assert_eq!(top, ["model", "answers", "usage"]);
    assert_eq!(keys(out.get("answers").unwrap()), ["pick", "rate", "yes"]);

    // A choice keeps the request's criteria order (not sorted), and the chosen key is one of them.
    let pick = answer(&out, "pick");
    assert_eq!(keys(pick), ["type", "choice", "probabilities", "confidence"]);
    assert_eq!(pick.get("type").and_then(Value::as_str), Some("choice"));
    let chosen = pick.get("choice").and_then(Value::as_str).unwrap();
    assert!(["zeta", "alpha", "mid"].contains(&chosen));
    let probabilities = pick.get("probabilities").unwrap();
    assert_eq!(keys(probabilities), ["zeta", "alpha", "mid"]);
    let sum: f64 = probabilities
        .as_object()
        .unwrap()
        .iter()
        .map(|(_, p)| number(p))
        .sum();
    assert!((sum - 1.0).abs() < 1e-9, "{sum}");
    let confidence = number(pick.get("confidence").unwrap());
    assert!((0.0..=1.0).contains(&confidence));

    // A score's legend and probabilities use the level keys and echo the descriptions as given.
    let rate = answer(&out, "rate");
    assert_eq!(
        keys(rate),
        ["type", "score", "legend", "probabilities", "confidence"]
    );
    let legend = rate.get("legend").unwrap();
    assert_eq!(keys(legend), ["0", "1", "2"]);
    assert_eq!(
        legend.get("0"),
        Some(&Value::Str("low".into())),
        "a string description stays a string"
    );
    assert_eq!(keys(rate.get("probabilities").unwrap()), ["0", "1", "2"]);
    let score = number(rate.get("score").unwrap());
    assert!((0.0..=2.0).contains(&score), "{score}");

    // A noul has its probability and no confidence.
    let yes = answer(&out, "yes");
    assert_eq!(keys(yes), ["type", "noul"]);
    let noul = number(yes.get("noul").unwrap());
    assert!((0.0..=1.0).contains(&noul), "{noul}");

    // usage is the sum of the three prompts' token counts; nothing is generated.
    let usage = out.get("usage").unwrap();
    assert_eq!(keys(usage), ["input_tokens", "output_tokens"]);
    assert!(number(usage.get("input_tokens").unwrap()) > 0.0);
    assert_eq!(number(usage.get("output_tokens").unwrap()), 0.0);

    // The same questions one at a time add up to the same token count.
    let mut alone = 0.0;
    for one in [
        r#"{"state":"a plain state with a few words","questions":{"pick":{"type":"choice","instructions":"which one","criteria":{"zeta":null,"alpha":{"k":1},"mid":"middle"}}}}"#,
        r#"{"state":"a plain state with a few words","questions":{"rate":{"type":"score","instructions":"how much","criteria":["low","mid","high"]}}}"#,
        r#"{"state":"a plain state with a few words","questions":{"yes":{"type":"noul","instructions":"is it so?","criteria":{"true":"yes","false":"no"}}}}"#,
    ] {
        let (status, body) = post(port, one);
        assert_eq!(status, 200, "{body:?}");
        alone += number(body.get("usage").unwrap().get("input_tokens").unwrap());
    }
    assert_eq!(number(usage.get("input_tokens").unwrap()), alone);
}

#[test]
fn a_question_is_identical_alone_and_beyond_the_slot_count() {
    let dir = tmpdir("queue");
    let model = decision_model(&dir);
    // Two slots and five questions: three wait in the engine's queue.
    let (_guard, port) = serve(&model, 2);

    let question = |i: usize| {
        format!(
            r#""q{i}":{{"type":"choice","instructions":"pick a word","criteria":{{"one{i}":null,"two{i}":null,"three{i}":null}}}}"#
        )
    };
    let questions: Vec<String> = (0..5).map(question).collect();
    let body = format!(
        r#"{{"state":"a plain state with a few words","questions":{{{}}}}}"#,
        questions.join(",")
    );
    let (status, all) = post(port, &body);
    assert_eq!(status, 200, "{all:?}");
    assert_eq!(
        keys(all.get("answers").unwrap()),
        ["q0", "q1", "q2", "q3", "q4"],
        "answers keep request order across the queue"
    );

    for i in 0..5 {
        let alone_body = format!(
            r#"{{"state":"a plain state with a few words","questions":{{{}}}}}"#,
            question(i)
        );
        let (status, alone) = post(port, &alone_body);
        assert_eq!(status, 200, "{alone:?}");
        assert_eq!(
            answer(&all, &format!("q{i}")),
            answer(&alone, &format!("q{i}")),
            "question {i} changed when it waited for a slot"
        );
    }
}

#[test]
fn every_validation_message_is_llama_cpps() {
    let dir = tmpdir("errors");
    let model = decision_model(&dir);
    let (_guard, port) = serve(&model, 4);

    let too_many: String = (0..53)
        .map(|i| format!("\"{i}\": null"))
        .collect::<Vec<_>>()
        .join(",");
    let cases: Vec<(String, String)> = vec![
        (
            r#"{"questions":{"a":{"type":"noul","instructions":"x"}}}"#.into(),
            "\"state\" must be provided".into(),
        ),
        (
            r#"{"state":"s"}"#.into(),
            "\"questions\" must be a non-empty object".into(),
        ),
        (
            r#"{"state":"s","questions":{}}"#.into(),
            "\"questions\" must be a non-empty object".into(),
        ),
        (
            r#"{"state":"s","questions":{"a":7}}"#.into(),
            "questions.a: must be an object".into(),
        ),
        (
            r#"{"state":"s","questions":{"a":{"type":"noul"}}}"#.into(),
            "questions.a: \"instructions\" must be provided".into(),
        ),
        (
            r#"{"state":"s","questions":{"a":{"type":"bad","instructions":"x"}}}"#.into(),
            "questions.a: \"type\" must be one of: choice, score, noul".into(),
        ),
        (
            r#"{"state":"s","questions":{"a":{"type":"choice","instructions":"x","criteria":{}}}}"#
                .into(),
            "questions.a: \"criteria\" must be a non-empty object".into(),
        ),
        (
            r#"{"state":"s","questions":{"a":{"type":"score","instructions":"x","criteria":[1]}}}"#
                .into(),
            "questions.a: \"criteria\" must be an array of 2 to 10 levels".into(),
        ),
        (
            r#"{"state":"s","questions":{"a":{"type":"noul","instructions":"x","criteria":[]}}}"#
                .into(),
            "questions.a: \"criteria\" must be an object".into(),
        ),
        (
            format!(
                r#"{{"state":"s","questions":{{"a":{{"type":"choice","instructions":"x","criteria":{{{too_many}}}}}}}}}"#
            ),
            "questions.a: too many options (53), this model supports at most 52".into(),
        ),
    ];
    for (body, want) in cases {
        let (status, out) = post(port, &body);
        assert_eq!(status, 400, "{want}: {out:?}");
        assert_eq!(
            out.get("error").and_then(|e| e.get("message")),
            Some(&Value::Str(want.clone())),
            "request: {}",
            body.len()
        );
    }

    // Not JSON.
    let (status, out) = post(port, "{");
    assert_eq!(status, 400);
    assert!(out.get("error").and_then(|e| e.get("message")).is_some());
}

#[test]
fn media_is_a_501_and_bad_media_is_still_a_400() {
    let dir = tmpdir("media");
    let model = decision_model(&dir);
    let (_guard, port) = serve(&model, 4);

    let question = r#""a":{"type":"noul","instructions":"x"}"#;
    for key in ["files", "images"] {
        let body = format!(
            r#"{{"state":"s","{key}":["data:image/png;base64,AAAA"],"questions":{{{question}}}}}"#
        );
        let (status, out) = post(port, &body);
        assert_eq!(status, 501, "{out:?}");
        assert_eq!(
            out.get("error").and_then(|e| e.get("message")),
            Some(&Value::Str(
                "This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`".into()
            ))
        );
        assert_eq!(
            out.get("error").and_then(|e| e.get("type")),
            Some(&Value::Str("not_supported_error".into()))
        );
    }
    // An empty array is no media.
    let body = format!(r#"{{"state":"s","files":[],"questions":{{{question}}}}}"#);
    assert_eq!(post(port, &body).0, 200);

    for (body, want) in [
        (
            format!(r#"{{"state":"s","files":"data:image/png;base64,AA","questions":{{{question}}}}}"#),
            "\"files\" must be an array",
        ),
        (
            format!(r#"{{"state":"s","images":["http://example.test/a.png"],"questions":{{{question}}}}}"#),
            "media must be data URLs (data:image/..., data:audio/... or data:video/...;base64,...)",
        ),
        (
            format!(r#"{{"state":"s","videos":["data:video/mp4;base64,AA"],"questions":{{{question}}}}}"#),
            "\"videos\" is not supported",
        ),
    ] {
        let (status, out) = post(port, &body);
        assert_eq!(status, 400, "{want}: {out:?}");
        assert_eq!(
            out.get("error").and_then(|e| e.get("message")),
            Some(&Value::Str(want.to_string()))
        );
    }
}

#[test]
fn a_chat_model_refuses_systemone_and_the_modalities_differ() {
    let dir = tmpdir("refusal");
    let plain_dir = dir.join("plain");
    std::fs::create_dir_all(&plain_dir).unwrap();
    let plain = write_hybrid_plain(&plain_dir, Format::F16, 43);
    let (guard, plain_port) = serve(&plain, 2);

    let (status, out) = post(
        plain_port,
        r#"{"state":"s","questions":{"a":{"type":"noul","instructions":"x"}}}"#,
    );
    assert_eq!(status, 501, "{out:?}");
    assert_eq!(
        out.get("error").and_then(|e| e.get("message")),
        Some(&Value::Str("This model is not a decision model".into()))
    );
    let (_, models) = try_http(plain_port, "GET", "/v1/models", None).unwrap();
    let models = Value::parse(&models).unwrap();
    let arch = models
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|d| d.first())
        .and_then(|m| m.get("architecture"))
        .expect("architecture");
    assert_eq!(
        arch.get("output_modalities"),
        Some(&Value::Arr(vec![Value::Str("text".into())]))
    );
    drop(guard);

    let decision_dir = dir.join("decision");
    std::fs::create_dir_all(&decision_dir).unwrap();
    let model = decision_model(&decision_dir);
    let (guard, port) = serve(&model, 2);
    let (_, models) = try_http(port, "GET", "/v1/models", None).unwrap();
    let models = Value::parse(&models).unwrap();
    let arch = models
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|d| d.first())
        .and_then(|m| m.get("architecture"))
        .expect("architecture");
    assert_eq!(
        arch.get("output_modalities"),
        Some(&Value::Arr(vec![Value::Str("decisions".into())]))
    );
    // A decision-only model serves no text generation.
    let (status, out) =
        try_http(port, "POST", "/v1/chat/completions", Some(r#"{"messages":[]}"#)).unwrap();
    assert_eq!(status, 501);
    let out = Value::parse(&out).unwrap();
    assert_eq!(
        out.get("error").and_then(|e| e.get("message")),
        Some(&Value::Str(
            "This model is a decision model without text generation, use /v1/systemone".into()
        ))
    );
    drop(guard);
}
