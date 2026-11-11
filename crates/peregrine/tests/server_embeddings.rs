//! `POST /v1/embeddings` end to end: the OpenAI shapes, `dimensions`, `base64`, the unknown-field
//! refusal and the over-length error (roadmap S4.1).
//!
//! A synthetic embedding model is served by the real binary on a free port; the endpoint is
//! driven over HTTP exactly as a client drives it.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use support::models::{write_dense_pooled, Dense};

const WIDTH: usize = 256;

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-server-embed-{}-{name}",
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
fn try_http(port: u16, method: &str, path: &str, body: Option<&Value>) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(60))).ok();
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

/// One HTTP/1.1 exchange; the server must be listening.
fn http(port: u16, method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
    try_http(port, method, path, body).expect("the server is listening")
}

fn post(port: u16, path: &str, body: Value) -> (u16, Value) {
    let (status, text) = http(port, "POST", path, Some(&body));
    let value = serde_json::from_str(&text).unwrap_or(Value::Null);
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

/// Serve a synthetic last-token-pooling llama file; returns (guard, port).
fn serve(dir: &Path) -> (Guard, u16) {
    let model = write_dense_pooled(dir, "llama", Format::F16, 31, Dense::default(), 3);
    let port = free_port();
    let log = dir.join("serve.log");
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
            "64",
            "--threads",
            "2",
            "--slots",
            "4",
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

fn floats(v: &Value) -> Vec<f32> {
    serde_json::from_value(v.clone()).expect("a float array")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

fn embedding(body: &Value, i: usize) -> Vec<f32> {
    floats(&body["data"][i]["embedding"])
}

/// Straightforward base64 decode for the test (`+`, `/`, `=` padding).
fn unbase64(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u8 {
        match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => 0,
        }
    };
    let mut out = Vec::new();
    for chunk in s.as_bytes().chunks(4) {
        let n = ((val(chunk[0]) as u32) << 18)
            | ((val(chunk[1]) as u32) << 12)
            | ((val(*chunk.get(2).unwrap_or(&b'=')) as u32) << 6)
            | val(*chunk.get(3).unwrap_or(&b'=')) as u32;
        out.push((n >> 16) as u8);
        if chunk.get(2).is_some_and(|&c| c != b'=') {
            out.push((n >> 8) as u8);
        }
        if chunk.get(3).is_some_and(|&c| c != b'=') {
            out.push(n as u8);
        }
    }
    out
}

#[test]
fn every_input_shape_returns_normalised_vectors_in_input_order() {
    let dir = tmpdir("shapes");
    let (_guard, port) = serve(&dir);

    // A string, with the rest of the OpenAI envelope.
    let (status, one) = post(
        port,
        "/v1/embeddings",
        json!({"input": "hello", "model": "my-alias"}),
    );
    assert_eq!(status, 200);
    assert_eq!(one["object"], "list");
    assert_eq!(one["model"], "my-alias");
    assert_eq!(one["data"].as_array().unwrap().len(), 1);
    assert_eq!(one["data"][0]["object"], "embedding");
    assert_eq!(one["data"][0]["index"], 0);
    let v = embedding(&one, 0);
    assert_eq!(v.len(), WIDTH);
    assert!((norm(&v) - 1.0).abs() < 1e-5, "L2-normalised by default");
    assert!(one["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(one["usage"]["total_tokens"], one["usage"]["prompt_tokens"]);

    // An array of strings, in input order and bit-identical to each text alone.
    let (status, many) = post(
        port,
        "/v1/embeddings",
        json!({"input": ["alpha", "a second, longer text", "gamma"]}),
    );
    assert_eq!(status, 200);
    assert_eq!(many["data"].as_array().unwrap().len(), 3);
    for i in 0..3 {
        assert_eq!(many["data"][i]["index"], i);
    }
    let (status, alone) = post(
        port,
        "/v1/embeddings",
        json!({"input": "a second, longer text"}),
    );
    assert_eq!(status, 200);
    assert_eq!(
        bits(&embedding(&many, 1)),
        bits(&embedding(&alone, 0)),
        "a text's vector does not depend on the batch it was in"
    );

    // An array of token ids is one input; an array of arrays is several.
    let (status, ids) = post(port, "/v1/embeddings", json!({"input": [11, 12, 13]}));
    assert_eq!(status, 200);
    assert_eq!(ids["data"].as_array().unwrap().len(), 1);
    let (status, arrays) = post(port, "/v1/embeddings", json!({"input": [[11], [12, 13]]}));
    assert_eq!(status, 200);
    assert_eq!(arrays["data"].as_array().unwrap().len(), 2);
    assert_eq!(embedding(&arrays, 0).len(), WIDTH);
}

#[test]
fn a_batch_bigger_than_the_slot_count_queues_and_keeps_input_order() {
    let dir = tmpdir("queue");
    let (_guard, port) = serve(&dir);

    // The server runs `--slots 4`; 12 texts is three waves, so eight of them wait in the
    // engine's queue and must be admitted as slots free up, not refused or dropped.
    let texts: Vec<String> = (0..12)
        .map(|i| format!("text number {i} with a few more words to make it {i} tokens long"))
        .collect();
    let (status, body) = post(port, "/v1/embeddings", json!({"input": &texts}));
    assert_eq!(status, 200, "{body}");
    let data = body["data"].as_array().expect("a data array");
    assert_eq!(data.len(), texts.len(), "every text of the batch came back");
    for (i, item) in data.iter().enumerate() {
        assert_eq!(
            item["index"].as_u64(),
            Some(i as u64),
            "results are in input order"
        );
    }

    // And each vector is the one the text gets alone: waiting for a slot changes no number.
    for (i, text) in texts.iter().enumerate() {
        let (status, alone) = post(port, "/v1/embeddings", json!({"input": text}));
        assert_eq!(status, 200);
        assert_eq!(
            bits(&embedding(&body, i)),
            bits(&embedding(&alone, 0)),
            "text {i} changed when it waited for a slot"
        );
    }
}

#[test]
fn dimensions_truncate_then_normalise_and_base64_is_little_endian_f32() {
    let dir = tmpdir("dimensions");
    let (_guard, port) = serve(&dir);

    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": "hello", "dimensions": 8}),
    );
    assert_eq!(status, 200);
    let v = embedding(&body, 0);
    assert_eq!(v.len(), 8);
    assert!((norm(&v) - 1.0).abs() < 1e-5);

    let (status, _) = post(
        port,
        "/v1/embeddings",
        json!({"input": "hello", "dimensions": 999}),
    );
    assert_eq!(status, 400, "dimensions larger than the width is refused");

    // base64 is the same vector's little-endian f32 bytes.
    let (status, plain) = post(port, "/v1/embeddings", json!({"input": "hello"}));
    assert_eq!(status, 200);
    let (status, encoded) = post(
        port,
        "/v1/embeddings",
        json!({"input": "hello", "encoding_format": "base64"}),
    );
    assert_eq!(status, 200);
    let text = encoded["data"][0]["embedding"]
        .as_str()
        .expect("a base64 string");
    let bytes = unbase64(text);
    assert_eq!(bytes.len(), WIDTH * 4);
    let decoded: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(bits(&decoded), bits(&embedding(&plain, 0)));
}

#[test]
fn unknown_fields_and_bad_formats_are_refused_and_over_length_names_the_limit() {
    let dir = tmpdir("errors");
    let (_guard, port) = serve(&dir);

    // Unknown fields are refused by name, as on the chat route.
    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": "hello", "pooling": "mean"}),
    );
    assert_eq!(status, 400);
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("pooling"), "message: {message}");

    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": "hello", "encoding_format": "int8"}),
    );
    assert_eq!(status, 400);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("float"));

    // The server was started with 64 positions: a longer text is the caller's error, and the
    // message names the limit and the opt-in.
    let long: Vec<u32> = (0..100).collect();
    let (status, body) = post(port, "/v1/embeddings", json!({"input": long}));
    assert_eq!(status, 400);
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("100 tokens"), "message: {message}");
    assert!(message.contains("64 positions"), "message: {message}");
    assert!(message.contains("truncate"), "message: {message}");

    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": long, "truncate": true}),
    );
    assert_eq!(status, 200);
    assert!((norm(&embedding(&body, 0)) - 1.0).abs() < 1e-5);
    assert_eq!(
        body["usage"]["prompt_tokens"].as_u64().unwrap(),
        64,
        "the truncated text is billed at the served limit"
    );
}
