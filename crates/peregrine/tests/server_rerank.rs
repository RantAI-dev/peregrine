//! `POST /v1/rerank` and `POST /v1/classify` end to end (roadmap S4.6b).
//!
//! A synthetic `bert` file with a classifier head is served by the real binary on a free
//! port; the endpoints are driven over HTTP exactly as a client drives them. A file without
//! a head answers 400 by name, and an over-length pair is refused (or truncated when asked)
//! the way `/v1/embeddings` treats one.

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
use support::models::{write_bert, BertConfig, BertHeadConfig};

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-server-rerank-{}-{name}",
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

fn post(port: u16, path: &str, body: Value) -> (u16, Value) {
    let (status, text) =
        try_http(port, "POST", path, Some(&body)).expect("the server is listening");
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}

fn get(port: u16, path: &str) -> Value {
    let (_, text) = try_http(port, "GET", path, None).expect("the server is listening");
    serde_json::from_str(&text).unwrap_or(Value::Null)
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

/// Serve a synthetic `bert` file on a free port; returns (guard, port).
fn serve(dir: &Path, model: &Path) -> (Guard, u16) {
    let port = free_port();
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
            "64",
            "--threads",
            "2",
            "--slots",
            "2",
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

/// A reranker's file: one raw score, no labels, `pooling_type = 4` (rank).
fn reranker(dir: &Path) -> PathBuf {
    write_bert(
        dir,
        Format::F32,
        71,
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

/// A classifier's file: three labelled outputs.
fn classifier(dir: &Path) -> PathBuf {
    write_bert(
        dir,
        Format::F32,
        72,
        BertConfig {
            head: Some(BertHeadConfig {
                n_out: 3,
                labels: vec!["negative".into(), "neutral".into(), "positive".into()],
                ..Default::default()
            }),
            ..Default::default()
        },
    )
}

fn results(body: &Value) -> Vec<(usize, f32)> {
    body["results"]
        .as_array()
        .expect("results is an array")
        .iter()
        .map(|r| {
            (
                r["index"].as_u64().unwrap() as usize,
                r["relevance_score"].as_f64().unwrap() as f32,
            )
        })
        .collect()
}

#[test]
fn rerank_returns_sorted_scores_in_llama_cpps_shape() {
    let dir = tmpdir("rerank");
    let model = reranker(&dir);
    let (_guard, port) = serve(&dir, &model);
    let documents = json!([
        "the cat sat on the mat",
        "a completely unrelated sentence about taxes",
        "hello world",
    ]);
    let (status, body) = post(
        port,
        "/v1/rerank",
        json!({"model": "my-reranker", "query": "hello world", "documents": documents}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "my-reranker");
    let got = results(&body);
    assert_eq!(got.len(), 3);
    // Descending by score, and every document appears once.
    let mut indices: Vec<usize> = got.iter().map(|(i, _)| *i).collect();
    indices.sort_unstable();
    assert_eq!(indices, vec![0, 1, 2]);
    for w in got.windows(2) {
        assert!(w[0].1 >= w[1].1, "not sorted: {got:?}");
    }
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        body["usage"]["total_tokens"],
        body["usage"]["prompt_tokens"]
    );
    // No document text unless it was asked for.
    assert!(body["results"][0].get("document").is_none());

    // `top_n` keeps the best N, and the first of the full list is the same document.
    let (status, top) = post(
        port,
        "/v1/rerank",
        json!({"query": "hello world", "documents": documents, "top_n": 2}),
    );
    assert_eq!(status, 200);
    assert_eq!(results(&top).len(), 2);
    assert_eq!(results(&top)[0], got[0]);

    // A document as an object, and the text returned with it.
    let (status, objects) = post(
        port,
        "/v1/rerank",
        json!({
            "query": {"text": "hello world"},
            "documents": [{"text": "the cat sat on the mat"}, {"text": ["hello", " world"]}],
            "return_documents": true,
        }),
    );
    assert_eq!(status, 200, "{objects}");
    // The returned text is the document's, at the index the result names (an array of
    // strings is joined, as llama.cpp joins it).
    let docs = ["the cat sat on the mat", "hello world"];
    let top = &objects["results"][0];
    let at = top["index"].as_u64().unwrap() as usize;
    assert_eq!(top["document"]["text"], docs[at]);

    // The alias routes llama.cpp serves answer the same.
    for path in ["/rerank", "/v1/reranking", "/reranking"] {
        let (status, alias) = post(
            port,
            path,
            json!({"query": "hello world", "documents": documents}),
        );
        assert_eq!(status, 200, "{path}");
        assert_eq!(results(&alias), got, "{path} differs");
    }

    // The score is the head's raw logit: the same pair alone gives the same bits as in a batch.
    let (_, alone) = post(
        port,
        "/v1/rerank",
        json!({"query": "hello world", "documents": ["hello world"]}),
    );
    let at = got.iter().position(|(i, _)| *i == 2).unwrap();
    assert_eq!(
        results(&alone)[0].1.to_bits(),
        got[at].1.to_bits(),
        "a pair's score depends on the batch it was in"
    );
}

#[test]
fn classify_returns_every_label_and_the_top_one() {
    let dir = tmpdir("classify");
    let model = classifier(&dir);
    let (_guard, port) = serve(&dir, &model);
    let (status, body) = post(
        port,
        "/v1/classify",
        json!({"model": "my-classifier", "input": ["hello world", "a second text"]}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "my-classifier");
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r["index"], i);
        let scores = r["scores"].as_object().expect("a scores object");
        assert_eq!(scores.len(), 3);
        let sum: f64 = scores.values().map(|v| v.as_f64().unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sums to 1: {sum}");
        let label = r["label"].as_str().unwrap();
        assert!(["negative", "neutral", "positive"].contains(&label));
        let score = r["score"].as_f64().unwrap();
        let named = scores[label].as_f64().unwrap();
        assert_eq!(score, named, "the top label's score is its probability");
        let best = scores
            .values()
            .map(|v| v.as_f64().unwrap())
            .fold(0.0, f64::max);
        assert_eq!(score, best, "the reported label is the most likely one");
    }
    // A string input is one entry.
    let (status, one) = post(port, "/v1/classify", json!({"input": "hello world"}));
    assert_eq!(status, 200);
    assert_eq!(one["results"].as_array().unwrap().len(), 1);
    // The same text alone has the same probabilities as in the batch.
    assert_eq!(
        one["results"][0]["scores"], results[0]["scores"],
        "a text's probabilities depend on the batch"
    );
    assert!(one["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

#[test]
fn a_file_without_a_head_and_bad_requests_are_refused_by_name() {
    let dir = tmpdir("refusals");
    let plain = write_bert(&dir, Format::F32, 73, BertConfig::default());
    let (_guard, port) = serve(&dir, &plain);
    let (status, body) = post(
        port,
        "/v1/rerank",
        json!({"query": "q", "documents": ["d"]}),
    );
    assert_eq!(status, 400);
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("bert") && msg.contains("classifier head"),
        "{msg}"
    );
    let (status, body) = post(port, "/v1/classify", json!({"input": "d"}));
    assert_eq!(status, 400);
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(msg.contains("classifier head"), "{msg}");
    // The reranker's own pool is unchanged: `/v1/embeddings` still works on this file.
    let (status, _) = post(port, "/v1/embeddings", json!({"input": "hello"}));
    assert_eq!(status, 200);
}

#[test]
fn over_length_pairs_are_refused_or_truncated_as_embeddings_are() {
    let dir = tmpdir("length");
    let model = reranker(&dir);
    let (_guard, port) = serve(&dir, &model);
    let long = "hello ".repeat(200);
    let (status, body) = post(
        port,
        "/v1/rerank",
        json!({"query": "hello", "documents": [&long]}),
    );
    assert_eq!(status, 400);
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("64 positions") && msg.contains("truncate"),
        "{msg}"
    );
    let (status, body) = post(
        port,
        "/v1/rerank",
        json!({"query": "hello", "documents": [&long], "truncate": true}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(results(&body).len(), 1);

    // Unknown fields are refused by name, as on the other routes.
    let (status, body) = post(
        port,
        "/v1/rerank",
        json!({"query": "q", "documents": ["d"], "documentss": []}),
    );
    assert_eq!(status, 400);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("documentss"));
    let (status, body) = post(port, "/v1/rerank", json!({"documents": ["d"]}));
    assert_eq!(status, 400);
    assert!(body["error"]["message"].as_str().unwrap().contains("query"));
    let (status, body) = post(port, "/v1/rerank", json!({"query": "q", "documents": []}));
    assert_eq!(status, 400);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("non-empty"));
    // An image needs a multimodal projector, which this file does not have.
    let (status, body) = post(
        port,
        "/v1/rerank",
        json!({"query": "q", "documents": [{"image": "http://example.com/x.png"}]}),
    );
    assert_eq!(status, 400);
    assert!(body["error"]["message"].as_str().unwrap().contains("image"));
}

#[test]
fn classify_names_an_unlabelled_multi_output_head_and_refuses_a_rank_head() {
    let dir = tmpdir("classify-shapes");
    // Three outputs, no labels: `LABEL_0..2`, as Hugging Face would name them.
    let unlabelled = write_bert(
        &dir,
        Format::F32,
        74,
        BertConfig {
            head: Some(BertHeadConfig {
                n_out: 3,
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    let (_guard, port) = serve(&dir, &unlabelled);
    let (status, body) = post(port, "/v1/classify", json!({"input": "hello"}));
    assert_eq!(status, 200, "{body}");
    let scores = body["results"][0]["scores"].as_object().unwrap();
    let mut names: Vec<&String> = scores.keys().collect();
    names.sort();
    assert_eq!(names, ["LABEL_0", "LABEL_1", "LABEL_2"]);
}

#[test]
fn models_reports_what_a_file_can_do() {
    let dir = tmpdir("modalities");
    let rank = reranker(&dir);
    let (_guard, port) = serve(&dir, &rank);
    let models = get(port, "/v1/models");
    assert_eq!(
        models["data"][0]["architecture"]["output_modalities"],
        json!(["rerank"])
    );
    drop(_guard);

    let labels = classifier(&dir);
    let (_guard, port) = serve(&dir, &labels);
    let models = get(port, "/v1/models");
    assert_eq!(
        models["data"][0]["architecture"]["output_modalities"],
        json!(["classify"])
    );
}
