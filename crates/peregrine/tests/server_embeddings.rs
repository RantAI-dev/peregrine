//! `POST /v1/embeddings` end to end: the OpenAI shapes, `dimensions`, `base64`, the unknown-field
//! refusal and the over-length error (roadmap S4.1).
//!
//! A synthetic embedding model is served by the real binary on a free port; the endpoint is
//! driven over HTTP exactly as a client drives it.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

mod common;

use peregrine_core::gguf::Format;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::thread;
use std::time::{Duration, Instant};
use support::models::{
    write_dense_pooled, write_gemma_embedding, write_jina_bert, write_modern_bert, Dense,
    GemmaEmbeddingConfig, JinaBertConfig, ModernBertConfig,
};

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
    serve_with(dir, 4)
}

/// [`serve`] with a chosen `--slots` (S4.9b: the chat-slot bound, not the embedding bound).
fn serve_with(dir: &Path, slots: usize) -> (Guard, u16) {
    serve_with_ctx(dir, slots, 64)
}

/// [`serve_with`] with a chosen context (the chat template's system block needs room).
fn serve_with_ctx(dir: &Path, slots: usize, ctx: usize) -> (Guard, u16) {
    let model = write_dense_pooled(dir, "llama", Format::F16, 31, Dense::default(), 3);
    let (child, port) = common::start(
        dir,
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
            &ctx.to_string(),
            "--threads",
            "2",
            "--slots",
            &slots.to_string(),
            "--no-profile",
        ],
    );
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
        json!({"input": "hello", "model": "peregrine"}),
    );
    assert_eq!(status, 200);
    assert_eq!(one["object"], "list");
    assert_eq!(one["model"], "peregrine");
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

/// A bidirectional encoder (roadmap S4.4): the embeddings route works, `dimensions` truncates
/// after the dense head, and chat is refused with a message that names the model's purpose.
#[test]
fn an_encoder_serves_embeddings_and_refuses_chat() {
    let dir = tmpdir("encoder");
    let model = write_gemma_embedding(&dir, Format::F16, 61, GemmaEmbeddingConfig::default());
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
            "128",
            "--threads",
            "2",
            "--slots",
            "4",
            "--no-profile",
        ],
    );
    let _guard = Guard(child);
    wait_for_health(port);

    let a: Vec<u32> = (11..48).collect();
    let b: Vec<u32> = (20..35).collect();
    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": [a.clone(), b.clone(), a.clone()]}),
    );
    assert_eq!(status, 200, "{body}");
    let width = 128usize;
    assert_eq!(embedding(&body, 0).len(), width);
    assert!((norm(&embedding(&body, 0)) - 1.0).abs() < 1e-5);
    assert_eq!(
        bits(&embedding(&body, 0)),
        bits(&embedding(&body, 2)),
        "the same text is the same vector twice in one request"
    );

    // Matryoshka: truncated, then normalised.
    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": [a.clone()], "dimensions": 64, "encoding_format": "base64"}),
    );
    assert_eq!(status, 200, "{body}");
    let v = floats(&json!(unbase64_f32(
        body["data"][0]["embedding"].as_str().unwrap()
    )));
    assert_eq!(v.len(), 64);
    assert!((norm(&v) - 1.0).abs() < 1e-5);

    let (status, body) = post(
        port,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "hi"}]}),
    );
    assert_eq!(status, 400, "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("embedding model"), "message: {message}");
    assert!(message.contains("cannot generate"), "message: {message}");

    // An over-length text names the encoder's context, not a cache's.
    let long: Vec<u32> = (0..200).collect();
    let (status, body) = post(port, "/v1/embeddings", json!({"input": long}));
    assert_eq!(status, 400, "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("200 tokens"), "message: {message}");
    assert!(message.contains("128 positions"), "message: {message}");
}

fn unbase64_f32(s: &str) -> Vec<f32> {
    let bytes = unbase64(s);
    bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// S4.9b: on a server started with `--slots 2`, a 128-text request is admitted whole and returns
/// 128 vectors in input order, each bit-identical to the same text alone. The texts do not take
/// the chat slots: `--slots` bounds chat only.
#[test]
fn a_128_text_request_is_admitted_whole_on_a_two_slot_server() {
    let dir = tmpdir("wide-batch");
    let (_guard, port) = serve_with(&dir, 2);
    let texts: Vec<String> = (0..128).map(|i| format!("text number {i}")).collect();
    let (status, body) = post(port, "/v1/embeddings", json!({"input": &texts}));
    assert_eq!(status, 200, "{body}");
    let data = body["data"].as_array().expect("a data array");
    assert_eq!(data.len(), 128, "every text of the batch came back");
    for (i, item) in data.iter().enumerate() {
        assert_eq!(item["index"].as_u64(), Some(i as u64), "input order");
    }
    // A sample of the batch is the text's own vector, alone.
    for i in [0usize, 1, 63, 126, 127] {
        let (status, alone) = post(port, "/v1/embeddings", json!({"input": texts[i]}));
        assert_eq!(status, 200);
        assert_eq!(
            bits(&embedding(&body, i)),
            bits(&embedding(&alone, 0)),
            "text {i} changed in a 128-text request"
        );
    }
}

/// S4.9b: a chat request is served while a 128-text embedding request runs on the same
/// one-chat-slot server. The embedding texts have their own admission, so they never consume the
/// chat slot or wait for it (and the chat never waits for the whole batch).
#[test]
fn a_chat_request_is_served_while_a_wide_embedding_request_runs() {
    let dir = tmpdir("chat-beside");
    let (_guard, port) = serve_with_ctx(&dir, 1, 256);
    let texts: Vec<String> = (0..128).map(|i| format!("text number {i}")).collect();
    let embedding = thread::spawn(move || post(port, "/v1/embeddings", json!({"input": texts})));
    let (status, chat) = post(
        port,
        "/v1/chat/completions",
        json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "temperature": 0,
        }),
    );
    assert_eq!(status, 200, "{chat}");
    assert!(chat["choices"][0]["message"]["content"].is_string());
    let (status, body) = embedding.join().expect("the embedding request thread");
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["data"].as_array().unwrap().len(), 128);
}

/// The ModernBERT family served (roadmap S4.7): a synthetic `modern-bert` file with
/// mean pooling answers `/v1/embeddings` for strings, refuses chat by name, and
/// names its context as the text limit.
#[test]
fn a_modern_bert_encoder_serves_embeddings_and_refuses_chat() {
    let dir = tmpdir("modern-bert");
    let model = write_modern_bert(&dir, Format::F16, 72, ModernBertConfig::default());
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
            "64",
            "--threads",
            "2",
            "--slots",
            "4",
            "--no-profile",
        ],
    );
    let _guard = Guard(child);
    wait_for_health(port);

    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": ["hello world", "the test of a paris city", "hello world"]}),
    );
    assert_eq!(status, 200, "{body}");
    for i in 0..3 {
        let v = embedding(&body, i);
        assert_eq!(v.len(), 128, "width");
        assert!((norm(&v) - 1.0).abs() < 1e-4, "normalised");
    }
    assert_eq!(
        bits(&embedding(&body, 0)),
        bits(&embedding(&body, 2)),
        "the same text has one vector"
    );

    let (status, body) = post(
        port,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
    );
    assert_eq!(status, 400, "{body}");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("embedding model"));

    let long: Vec<u32> = (5..90).collect();
    let (status, body) = post(port, "/v1/embeddings", json!({"input": long}));
    assert_eq!(status, 400, "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("64 positions"), "message: {message}");
    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": long, "truncate": true}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["usage"]["prompt_tokens"].as_u64().unwrap(), 64);
}

/// The BERT family served (roadmap S4.6a): a synthetic `bert` file with cls pooling
/// answers `/v1/embeddings` for strings (WordPiece `[CLS] … [SEP]`), refuses chat by
/// name, and names its position table as the text limit.
#[test]
fn a_bert_encoder_serves_embeddings_and_refuses_chat() {
    let dir = tmpdir("bert");
    let model = support::models::write_bert(
        &dir,
        Format::F16,
        71,
        support::models::BertConfig::default(),
    );
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
            "64",
            "--threads",
            "2",
            "--slots",
            "4",
            "--no-profile",
        ],
    );
    let _guard = Guard(child);
    wait_for_health(port);

    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": ["hello world", "the test of a paris city", "hello world"]}),
    );
    assert_eq!(status, 200, "{body}");
    // The file's width is 128; cls pooling is the file's own.
    for i in 0..3 {
        let v = embedding(&body, i);
        assert_eq!(v.len(), 128, "width");
        assert!((norm(&v) - 1.0).abs() < 1e-4, "normalised");
    }
    assert_eq!(
        bits(&embedding(&body, 0)),
        bits(&embedding(&body, 2)),
        "the same text has one vector"
    );
    // Matryoshka: truncate, then normalise.
    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": "hello", "dimensions": 32}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(embedding(&body, 0).len(), 32);

    // Chat on an encoder is a 400 that names what the model is.
    let (status, body) = post(
        port,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
    );
    assert_eq!(status, 400, "{body}");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("embedding model"));

    // The position table is 64 rows: a longer text is refused, and `truncate` opts in.
    let long: Vec<u32> = (5..90).collect();
    let (status, body) = post(port, "/v1/embeddings", json!({"input": long}));
    assert_eq!(status, 400, "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("64 positions"), "message: {message}");
    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": long, "truncate": true}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["usage"]["prompt_tokens"].as_u64().unwrap(), 64);
}

/// The `jina-bert-v2` family served (roadmap S4.8c): a synthetic file with mean
/// pooling answers `/v1/embeddings` for strings, refuses chat by name, and names its
/// `context_length` as the text limit.
#[test]
fn a_jina_bert_encoder_serves_embeddings_and_refuses_chat() {
    let dir = tmpdir("jina-bert");
    let model = write_jina_bert(&dir, Format::F16, 73, JinaBertConfig::default());
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
            "64",
            "--threads",
            "2",
            "--slots",
            "4",
            "--no-profile",
        ],
    );
    let _guard = Guard(child);
    wait_for_health(port);

    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": ["hello world", "the test of a paris city", "hello world"]}),
    );
    assert_eq!(status, 200, "{body}");
    for i in 0..3 {
        let v = embedding(&body, i);
        assert_eq!(v.len(), 128, "width");
        assert!((norm(&v) - 1.0).abs() < 1e-4, "normalised");
    }
    assert_eq!(
        bits(&embedding(&body, 0)),
        bits(&embedding(&body, 2)),
        "the same text has one vector"
    );

    let (status, body) = post(
        port,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
    );
    assert_eq!(status, 400, "{body}");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("embedding model"));

    let long: Vec<u32> = (5..90).collect();
    let (status, body) = post(port, "/v1/embeddings", json!({"input": long}));
    assert_eq!(status, 400, "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("64 positions"), "message: {message}");
    let (status, body) = post(
        port,
        "/v1/embeddings",
        json!({"input": long, "truncate": true}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["usage"]["prompt_tokens"].as_u64().unwrap(), 64);
}
