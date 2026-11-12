//! `response_format` on `/v1/chat/completions` (S6.6a), on synthetic models.
//!
//! Every reply that ends by itself is parsed and validated by an independent validator
//! (`peregrine-core/tests/support`); a truncated reply is reported as `length` and is a
//! prefix, as OpenAI documents.

#[path = "../../peregrine-core/tests/support/mod.rs"]
mod check;
#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

mod common;

use peregrine_core::gguf::Format;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::time::{Duration, Instant};
use support::models::{write_dense, write_dense_with_template, Dense};

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-server-constrain-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn shape() -> Dense {
    Dense {
        ctx: 1024,
        ..Dense::default()
    }
}

fn serve(name: &str, model: &Path) -> (Guard, u16) {
    let dir = tmpdir(name);
    let (child, port) = common::start(
        &dir,
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
            "1024",
            "--threads",
            "2",
            "--slots",
            "4",
            "--no-profile",
        ],
    );
    let guard = Guard(child);
    wait_for_health(port);
    (guard, port)
}

fn try_http(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&Value>,
    timeout: u64,
) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(timeout)))
        .ok();
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
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

fn wait_for_health(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if matches!(try_http(port, "GET", "/health", None, 5), Some((200, _))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("the server did not answer /health");
}

fn post(port: u16, body: &Value) -> (u16, Value) {
    let (status, text) = try_http(port, "POST", "/v1/chat/completions", Some(body), 120)
        .expect("the server is listening");
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}

/// One streaming exchange: the concatenated `content` deltas and the final finish reason.
fn post_stream(port: u16, body: &Value) -> (u16, String, String) {
    let (status, text) = try_http(port, "POST", "/v1/chat/completions", Some(body), 120)
        .expect("the server is listening");
    let mut content = String::new();
    let mut finish = String::new();
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(choice) = v["choices"].get(0) {
            if let Some(t) = choice["delta"]["content"].as_str() {
                content.push_str(t);
            }
            if let Some(f) = choice["finish_reason"].as_str() {
                finish = f.to_string();
            }
        }
    }
    (status, content, finish)
}

fn message(content: &str) -> Value {
    json!({"role": "user", "content": content})
}

fn chat(extra: Value) -> Value {
    let mut b = json!({"messages": [message("Give me an object.")]});
    if let Some(o) = extra.as_object() {
        for (k, v) in o {
            b[k] = v.clone();
        }
    }
    b
}

fn json_schema(schema: Value, strict: bool, max_tokens: usize) -> Value {
    chat(json!({
        "temperature": 0.0,
        "seed": 7,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": strict, "schema": schema}
        }
    }))
}

fn content_of(v: &Value) -> &str {
    v["choices"][0]["message"]["content"].as_str().unwrap_or("")
}

fn finish_of(v: &Value) -> &str {
    v["choices"][0]["finish_reason"].as_str().unwrap_or("")
}

fn assert_valid(text: &str, schema_text: &str) {
    let value =
        check::parse(text).unwrap_or_else(|e| panic!("the reply is not JSON ({e}): {text:?}"));
    let schema = check::parse(schema_text).expect("the schema parses");
    check::validate(&value, &schema, &schema)
        .unwrap_or_else(|e| panic!("the reply does not validate ({e}): {text:?}"));
}

const SMALL_SCHEMA: &str = r#"{"type":"object","properties":{"answer":{"type":"integer","minimum":0,"maximum":1}},"required":["answer"],"additionalProperties":false}"#;

#[test]
fn a_json_schema_reply_validates_and_streaming_is_the_same_text() {
    let dir = tmpdir("schema");
    let model = write_dense(&dir, "qwen2", Format::F16, 71, shape());
    let (_server, port) = serve("schema-serve", &model);
    let schema: Value = serde_json::from_str(SMALL_SCHEMA).unwrap();
    let (status, v) = post(port, &json_schema(schema.clone(), true, 128));
    assert_eq!(status, 200, "{v}");
    assert_eq!(finish_of(&v), "stop", "the reply should end itself: {v}");
    assert_valid(content_of(&v), SMALL_SCHEMA);

    let mut stream_body = json_schema(schema, true, 128);
    stream_body["stream"] = json!(true);
    let (status, streamed, finish) = post_stream(port, &stream_body);
    assert_eq!(status, 200);
    assert_eq!(finish, "stop");
    assert_eq!(
        streamed,
        content_of(&v),
        "streaming and non-streaming must agree"
    );
}

#[test]
fn json_object_replies_that_end_by_themselves_are_valid_objects() {
    let dir = tmpdir("object");
    // A seed whose random weights close the object within the token budget; the search is
    // deterministic, so a model that can do it always does.
    let mut found = None;
    for model_seed in 1..=24u64 {
        let model = write_dense(&dir, "qwen2", Format::F16, model_seed, shape());
        let (child, port) = common::start(
            &tmpdir(&format!("obj-{model_seed}")),
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
                "1024",
                "--threads",
                "2",
                "--slots",
                "1",
                "--no-profile",
            ],
        );
        let guard = Guard(child);
        wait_for_health(port);
        let body = chat(json!({
            "temperature": 0.0,
            "seed": 7,
            "max_tokens": 96,
            "response_format": {"type": "json_object"}
        }));
        let (status, v) = post(port, &body);
        assert_eq!(status, 200, "{v}");
        if finish_of(&v) == "stop" {
            assert_valid(content_of(&v), r#"{"type":"object"}"#);
            found = Some(content_of(&v).to_string());
            drop(guard);
            break;
        }
        assert_eq!(finish_of(&v), "length", "{v}");
        drop(guard);
    }
    assert!(
        found.is_some(),
        "no deterministic random model closed an object within the budget"
    );
}

#[test]
fn max_tokens_cuts_a_constrained_reply_and_reports_length() {
    let dir = tmpdir("truncate");
    let model = write_dense(&dir, "qwen2", Format::F16, 72, shape());
    let (_server, port) = serve("truncate-serve", &model);
    let schema: Value = serde_json::from_str(SMALL_SCHEMA).unwrap();
    let (status, v) = post(port, &json_schema(schema, true, 1));
    assert_eq!(status, 200, "{v}");
    assert_eq!(finish_of(&v), "length", "{v}");
    let text = content_of(&v);
    assert!(
        check::parse(text).is_err() || text.chars().count() <= 1,
        "a one-token cut is an incomplete prefix: {text:?}"
    );
}

#[test]
fn the_400s_are_openai_shaped_and_name_the_field() {
    let dir = tmpdir("errors");
    let model = write_dense(&dir, "qwen2", Format::F16, 73, shape());
    let (_server, port) = serve("errors-serve", &model);

    let (status, v) = post(port, &chat(json!({"response_format": {"type": "xml"}})));
    assert_eq!(status, 400, "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["param"], "response_format");

    let (status, v) = post(
        port,
        &json_schema(
            json!({"type": "object", "properties": {"x": {"type": "string", "pattern": "a+"}}}),
            false,
            32,
        ),
    );
    assert_eq!(status, 400, "{v}");
    assert_eq!(v["error"]["param"], "response_format");
    let message = v["error"]["message"].as_str().unwrap_or("");
    assert!(message.contains("pattern"), "{message}");
    assert!(message.contains("#/properties/x"), "{message}");

    let (status, v) = post(
        port,
        &chat(json!({
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "response_format": {"type": "json_object"}
        })),
    );
    assert_eq!(status, 400, "{v}");
    assert_eq!(v["error"]["param"], "response_format");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("response_format with tools is not supported yet"),
        "{v}"
    );
}

#[test]
fn an_unconstrained_request_is_unaffected_and_two_slots_do_not_interfere() {
    let dir = tmpdir("parallel");
    let model = write_dense(&dir, "qwen2", Format::F16, 74, shape());
    let (_server, port) = serve("parallel-serve", &model);

    let plain = chat(json!({"temperature": 0.0, "seed": 5, "max_tokens": 32}));
    let (status, baseline) = post(port, &plain);
    assert_eq!(status, 200, "{baseline}");

    let schema: Value = serde_json::from_str(SMALL_SCHEMA).unwrap();
    let constrained = json_schema(schema, true, 128);
    // Both are admitted before either is awaited: the engine runs them in one pass when
    // the scheduler can, and the plain slot must still emit the baseline's tokens.
    let plain1 = plain.clone();
    let constrained1 = constrained.clone();
    let (a, b) = std::thread::scope(|s| {
        let plain = s.spawn(move || post(port, &plain1));
        let constrained = s.spawn(move || post(port, &constrained1));
        (plain.join().unwrap(), constrained.join().unwrap())
    });
    assert_eq!(a.0, 200, "{}", a.1);
    assert_eq!(a.1["choices"], baseline["choices"]);
    assert_eq!(b.0, 200, "{}", b.1);
    assert_valid(content_of(&b.1), SMALL_SCHEMA);

    // The same plain request after a constrained one is still the same.
    let (_, again) = post(port, &plain);
    assert_eq!(again["choices"], baseline["choices"]);
}

#[test]
fn a_reasoning_model_is_refused_while_thinking_and_served_when_it_is_off() {
    let dir = tmpdir("reasoning");
    let model = write_dense_with_template(
        &dir,
        "qwen2",
        Format::F16,
        75,
        shape(),
        peregrine_core::tools::QWEN35_TEMPLATE,
    );
    let (_server, port) = serve("reasoning-serve", &model);
    let schema: Value = serde_json::from_str(SMALL_SCHEMA).unwrap();
    let (status, v) = post(port, &json_schema(schema.clone(), true, 64));
    assert_eq!(status, 400, "{v}");
    assert_eq!(v["error"]["param"], "response_format");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("response_format is not supported while the model is thinking"),
        "{v}"
    );

    let mut body = json_schema(schema, true, 128);
    body["chat_template_kwargs"] = json!({"enable_thinking": false});
    let (status, v) = post(port, &body);
    assert_eq!(status, 200, "{v}");
    assert_eq!(finish_of(&v), "stop", "{v}");
    assert_valid(content_of(&v), SMALL_SCHEMA);
}
