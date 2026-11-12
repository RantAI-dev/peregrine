//! Several models in one server (roadmap S4.10).
//!
//! `serve -m a -m b -m c -m d` serves a chat model, a causal embedder, a bidirectional
//! encoder and a decision model from one process. A request's `model` field selects the
//! model; a request without one goes to the first `-m`. Every answer must be bit-identical
//! to what a single-model server gives for the same file, even while the other models are
//! working, and the models take passes in turn rather than running two pools at once.
//!
//! The models are the synthetic files under `crates/peregrine-models/tests/support/`, so CI
//! needs no model files.
//!
//! One test per mechanism asks the server to drain: `SIGTERM` on Unix, the stop file on Windows
//! (S7.3). Both drain every model's in-flight request and then exit with status 0.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

mod common;

use peregrine_core::gguf::Format;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::thread;
use std::time::{Duration, Instant};
use support::models::{
    write_bert, write_dense, write_dense_pooled, write_gemma_embedding, write_hybrid_decision,
    BertConfig, BertHeadConfig, DecisionSpec, Dense, GemmaEmbeddingConfig,
};

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-server-multi-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
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
    stream.set_read_timeout(Some(Duration::from_secs(120))).ok();
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

fn http(port: u16, method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
    try_http(port, method, path, body).expect("the server is listening")
}

fn post(port: u16, path: &str, body: &Value) -> (u16, Value) {
    let (status, text) = http(port, "POST", path, Some(body));
    let value = serde_json::from_str(&text).unwrap_or(Value::Null);
    (status, value)
}

fn wait_for_health(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if matches!(try_http(port, "GET", "/health", None), Some((200, _))) {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("the server did not answer /health");
}

/// Serve the given files with the given ids, one `-m`/`--alias` pair each.
fn serve(dir: &Path, models: &[(&Path, &str)], threads: usize, ctx: usize) -> (Guard, u16) {
    let mut args: Vec<String> = vec![
        "serve".into(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        common::PORT.into(),
        "--ctx".into(),
        ctx.to_string(),
        "--threads".into(),
        threads.to_string(),
        "--slots".into(),
        "4".into(),
        "--no-profile".into(),
    ];
    for (path, id) in models {
        args.push("-m".into());
        args.push(path.to_str().unwrap().into());
        args.push("--alias".into());
        args.push((*id).into());
    }
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (child, port) = common::start(dir, &[], &args);
    let guard = Guard(child);
    wait_for_health(port);
    (guard, port)
}

/// The four synthetic files the multi-model server serves.
struct Files {
    chat: PathBuf,
    causal: PathBuf,
    encoder: PathBuf,
    decision: PathBuf,
}

fn files(dir: &Path) -> Files {
    let chat = dir.join("chat");
    let causal = dir.join("causal");
    let encoder = dir.join("encoder");
    let decision = dir.join("decision");
    for d in [&chat, &causal, &encoder, &decision] {
        std::fs::create_dir_all(d).unwrap();
    }
    Files {
        chat: write_dense(&chat, "llama", Format::F16, 11, Dense::default()),
        causal: write_dense_pooled(&causal, "llama", Format::F16, 12, Dense::default(), 3),
        encoder: write_gemma_embedding(&encoder, Format::F16, 13, GemmaEmbeddingConfig::default()),
        decision: write_hybrid_decision(
            &decision,
            Format::F16,
            14,
            &DecisionSpec::openjev().keys("qwen35"),
        ),
    }
}

fn multi(files: &Files, dir: &Path) -> (Guard, u16) {
    serve(
        dir,
        &[
            (&files.chat, "chat"),
            (&files.causal, "causal"),
            (&files.encoder, "encoder"),
            (&files.decision, "decide"),
        ],
        2,
        512,
    )
}

fn chat_body(model: &str, max_tokens: u64) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": "hello there"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 7,
    })
}

fn embedding_body(model: &str) -> Value {
    json!({
        "model": model,
        "input": ["hello", "a second text", "the third one"],
    })
}

fn decision_body(model: &str) -> Value {
    json!({
        "model": model,
        "state": "a plain state with a few words",
        "questions": {
            "pick": {"type": "choice", "instructions": "which one",
                     "criteria": {"zeta": null, "alpha": {"k": 1}, "mid": "middle"}},
            "rate": {"type": "score", "instructions": "how much",
                     "criteria": ["low", "mid", "high"]},
            "yes": {"type": "noul", "instructions": "is it so?",
                    "criteria": {"true": "yes", "false": "no"}}
        }
    })
}

fn content(body: &Value) -> String {
    body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("no content in {body}"))
        .to_string()
}

fn bits(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("an embedding array")
        .iter()
        .map(|x| (x.as_f64().expect("a float") as f32).to_bits())
        .collect()
}

/// The labelled metric value, or 0 when the line is absent.
fn metric(text: &str, name: &str, model: &str) -> u64 {
    let needle = format!("{name}{{model=\"{model}\"}}");
    text.lines()
        .find_map(|l| l.strip_prefix(&needle))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or_else(|| panic!("no {needle} line in /metrics:\n{text}"))
}

#[test]
fn models_are_routed_by_id_and_the_first_is_the_default() {
    let dir = tmpdir("routing");
    let f = files(&dir);
    let (_guard, port) = multi(&f, &dir);

    let (status, listed) = http(port, "GET", "/v1/models", None);
    assert_eq!(status, 200);
    let listed: Value = serde_json::from_str(&listed).unwrap();
    let ids: Vec<&str> = listed["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["chat", "causal", "encoder", "decide"]);
    // What each model can do, as a one-model server reports it.
    assert_eq!(
        listed["data"][0]["architecture"]["output_modalities"][0],
        "text"
    );
    assert_eq!(
        listed["data"][3]["architecture"]["output_modalities"][0],
        "decisions"
    );

    // No `model` goes to the first `-m`, and the answer is the one the id names.
    let (status, implicit) = post(port, "/v1/chat/completions", &chat_body("", 4));
    assert_eq!(status, 200);
    let (status, explicit) = post(port, "/v1/chat/completions", &chat_body("chat", 4));
    assert_eq!(status, 200);
    assert_eq!(content(&implicit), content(&explicit));

    // An unknown id is OpenAI's 404.
    let (status, unknown) = post(port, "/v1/chat/completions", &chat_body("nope", 4));
    assert_eq!(status, 404, "{unknown}");
    assert_eq!(unknown["error"]["type"], "invalid_request_error");
    assert_eq!(unknown["error"]["code"], "model_not_found");
    assert!(unknown["error"]["message"]
        .as_str()
        .unwrap()
        .contains("chat"));

    // Each route uses the model the request names.
    let (status, emb) = post(port, "/v1/embeddings", &embedding_body("causal"));
    assert_eq!(status, 200, "{emb}");
    assert_eq!(emb["data"].as_array().unwrap().len(), 3);
    let (status, emb) = post(port, "/v1/embeddings", &embedding_body("encoder"));
    assert_eq!(status, 200, "{emb}");
    assert_eq!(emb["data"].as_array().unwrap().len(), 3);
    let (status, dec) = post(port, "/v1/systemone", &decision_body("decide"));
    assert_eq!(status, 200, "{dec}");
    assert_eq!(dec["model"], "decide");

    // A capability the named model does not have is refused with the one-model message.
    let (status, wrong) = post(port, "/v1/chat/completions", &chat_body("encoder", 4));
    assert_eq!(status, 400);
    assert!(wrong["error"]["message"]
        .as_str()
        .unwrap()
        .contains("embedding model"));
    let (status, wrong) = post(port, "/v1/embeddings", &embedding_body("chat"));
    assert_eq!(status, 400);
    assert!(wrong["error"]["message"]
        .as_str()
        .unwrap()
        .contains("pooling_type"));
    let (status, wrong) = post(port, "/v1/embeddings", &embedding_body("decide"));
    assert_eq!(status, 400);
    assert!(wrong["error"]["message"]
        .as_str()
        .unwrap()
        .contains("pooling_type"));
    let (status, wrong) = post(port, "/v1/systemone", &decision_body("chat"));
    assert_eq!(status, 501);
    assert!(wrong["error"]["message"]
        .as_str()
        .unwrap()
        .contains("not a decision model"));
    let (status, _wrong) = post(port, "/v1/systemone", &decision_body("encoder"));
    assert_eq!(status, 501);

    // `/metrics` keeps the unlabelled lines for the first model and labels every model.
    let (status, text) = http(port, "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert!(text.contains("peregrine_requests_total "));
    assert!(text.contains("peregrine_model_requests_total{model=\"chat\"}"));
    assert!(text.contains("peregrine_model_requests_total{model=\"decide\"}"));
    assert!(text.contains("peregrine_pass_holders_max 1"));
}

#[test]
fn each_model_gives_the_same_bits_as_its_own_server() {
    let dir = tmpdir("bits");
    let f = files(&dir);
    let (guard, port) = multi(&f, &dir);

    // All four models busy at the same time: the answers must still be the ones a
    // single-model server gives, and every request must be served (no model starves).
    let (chat, causal, encoder, dec) = thread::scope(|s| {
        let a = s.spawn(|| post(port, "/v1/chat/completions", &chat_body("chat", 12)));
        let b = s.spawn(|| post(port, "/v1/embeddings", &embedding_body("causal")));
        let c = s.spawn(|| post(port, "/v1/embeddings", &embedding_body("encoder")));
        let d = s.spawn(|| post(port, "/v1/systemone", &decision_body("decide")));
        (
            a.join().unwrap(),
            b.join().unwrap(),
            c.join().unwrap(),
            d.join().unwrap(),
        )
    });
    drop(guard);

    for (status, body) in [&chat, &causal, &encoder, &dec] {
        assert_eq!(*status, 200, "{body}");
    }

    // One reference server per file, one at a time.
    let (chat_guard, chat_port) = serve(&dir, &[(&f.chat, "chat")], 2, 512);
    let (_, chat_ref) = post(chat_port, "/v1/chat/completions", &chat_body("chat", 12));
    drop(chat_guard);
    assert_eq!(content(&chat.1), content(&chat_ref), "chat bits changed");

    let (a_guard, a_port) = serve(&dir, &[(&f.causal, "causal")], 2, 512);
    let (_, causal_ref) = post(a_port, "/v1/embeddings", &embedding_body("causal"));
    drop(a_guard);
    for i in 0..3 {
        assert_eq!(
            bits(&causal.1["data"][i]["embedding"]),
            bits(&causal_ref["data"][i]["embedding"]),
            "causal embedding {i} changed"
        );
    }

    let (e_guard, e_port) = serve(&dir, &[(&f.encoder, "encoder")], 2, 512);
    let (_, encoder_ref) = post(e_port, "/v1/embeddings", &embedding_body("encoder"));
    drop(e_guard);
    for i in 0..3 {
        assert_eq!(
            bits(&encoder.1["data"][i]["embedding"]),
            bits(&encoder_ref["data"][i]["embedding"]),
            "encoder embedding {i} changed"
        );
    }

    let (d_guard, d_port) = serve(&dir, &[(&f.decision, "decide")], 2, 512);
    let (_, dec_ref) = post(d_port, "/v1/systemone", &decision_body("decide"));
    drop(d_guard);
    assert_eq!(dec.1["answers"], dec_ref["answers"], "decision changes");
    assert_eq!(dec.1["usage"], dec_ref["usage"], "decision usage changes");
}

/// A long chat prefill must not hold another model's request until it finishes: the
/// turn changes hands pass by pass, so the embedding is done within a few chat passes.
#[test]
fn a_long_chat_prefill_does_not_starve_an_embedding() {
    let dir = tmpdir("fairness");
    let f = files(&dir);
    let (_guard, port) = multi(&f, &dir);

    let long = "word ".repeat(60);
    let body = json!({
        "model": "chat",
        "messages": [{"role": "user", "content": long}],
        "max_tokens": 300,
        "temperature": 0,
    });
    let chat = thread::spawn(move || post(port, "/v1/chat/completions", &body));

    // The chat request is admitted and starts its prefill before the embedding arrives.
    thread::sleep(Duration::from_millis(100));
    let (_, before) = http(port, "GET", "/metrics", None);
    let chat_before = metric(&before, "peregrine_model_passes_total", "chat");

    let t = Instant::now();
    let (status, emb) = post(port, "/v1/embeddings", &embedding_body("causal"));
    assert_eq!(status, 200, "{emb}");
    let (_, after) = http(port, "GET", "/metrics", None);
    let chat_after = metric(&after, "peregrine_model_passes_total", "chat");
    let emb_steps = metric(&after, "peregrine_model_passes_total", "causal");

    // The embedding needed at least one pass of its own, and the chat model ran only a
    // handful meanwhile: it did not run its whole prefill and decode to completion first.
    assert!(emb_steps > 0);
    let chat_passes = chat_after.saturating_sub(chat_before);
    // 64 is far below the ~300 passes the chat request needs in total; the bound is loose
    // because the test machine is shared and the number of passes that fit in a scheduling
    // round is not fixed. The whole prefill and decode must not run first.
    assert!(
        chat_passes <= 64,
        "the embedding waited for {chat_passes} chat passes ({:.1} ms)",
        t.elapsed().as_secs_f64() * 1e3
    );

    let (status, body) = chat.join().unwrap();
    assert_eq!(status, 200, "the chat request never finished: {body}");
}

/// No two engines run a pass at once, so the process never has two thread pools working.
#[test]
fn no_two_engines_run_a_pass_at_once() {
    let dir = tmpdir("oversubscription");
    let f = files(&dir);
    let (_guard, port) = multi(&f, &dir);

    let (_, before) = http(port, "GET", "/metrics", None);
    let turns_before: u64 = before
        .lines()
        .find_map(|l| l.strip_prefix("peregrine_pass_turns_total "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap();

    for _ in 0..3 {
        thread::scope(|s| {
            s.spawn(|| post(port, "/v1/chat/completions", &chat_body("chat", 8)));
            s.spawn(|| post(port, "/v1/embeddings", &embedding_body("causal")));
            s.spawn(|| post(port, "/v1/embeddings", &embedding_body("encoder")));
            s.spawn(|| post(port, "/v1/systemone", &decision_body("decide")));
        });
    }

    let (_, after) = http(port, "GET", "/metrics", None);
    let turns_after: u64 = after
        .lines()
        .find_map(|l| l.strip_prefix("peregrine_pass_turns_total "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap();
    assert!(turns_after > turns_before, "no pass ran");
    let holders: u64 = after
        .lines()
        .find_map(|l| l.strip_prefix("peregrine_pass_holders_max "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap();
    assert_eq!(holders, 1, "two models ran a pass at the same time");
}

/// SIGTERM drains requests in flight on every model, then the process exits cleanly.
///
/// Unix only: Windows has no `SIGTERM`; its stop file (and the drain it triggers) is S7.3,
/// and Ctrl-C is the clean shutdown there today.
#[cfg(unix)]
#[test]
fn drain_covers_every_engine() {
    let dir = tmpdir("drain");
    let f = files(&dir);
    let (child, port) = common::start(
        &dir,
        &[("PEREGRINE_DRAIN_TIMEOUT_MS", "20000")],
        &[
            "serve",
            "-m",
            f.chat.to_str().unwrap(),
            "--alias",
            "chat",
            "-m",
            f.causal.to_str().unwrap(),
            "--alias",
            "causal",
            "-m",
            f.decision.to_str().unwrap(),
            "--alias",
            "decide",
            "--host",
            "127.0.0.1",
            "--port",
            common::PORT,
            "--ctx",
            "512",
            "--threads",
            "2",
            "--no-profile",
        ],
    );
    let mut guard = Guard(child);
    wait_for_health(port);

    let slow = json!({
        "model": "chat",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 200,
        "temperature": 0,
    });
    let requests = thread::scope(|s| {
        let a = s.spawn(|| post(port, "/v1/chat/completions", &slow));
        let b = s.spawn(|| post(port, "/v1/embeddings", &embedding_body("causal")));
        let c = s.spawn(|| post(port, "/v1/systemone", &decision_body("decide")));
        // Let the requests reach their engines, then ask the process to drain.
        thread::sleep(Duration::from_millis(300));
        let pid = guard.0.id();
        let sent = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .expect("run kill");
        assert!(sent.success(), "kill -TERM {pid} failed");
        (a.join().unwrap(), b.join().unwrap(), c.join().unwrap())
    });
    assert_eq!(requests.0 .0, 200, "{}", requests.0 .1);
    assert_eq!(requests.1 .0, 200, "{}", requests.1 .1);
    assert_eq!(requests.2 .0, 200, "{}", requests.2 .1);

    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = guard.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "the server did not exit after draining"
        );
        thread::sleep(Duration::from_millis(50));
    };
    assert!(status.success(), "the drained server exited with {status}");
}

/// The same drain on Windows, asked for through the stop file (S7.3): every model's in-flight
/// request finishes, a new connection is refused once the drain starts, and the process exits 0.
#[cfg(windows)]
#[test]
fn drain_covers_every_engine_through_the_stop_file() {
    let dir = tmpdir("drain-stopfile");
    let f = files(&dir);
    let state = dir.join("state");
    std::fs::create_dir_all(&state).unwrap();
    let state_s = state.to_str().unwrap().to_string();
    let (child, port) = common::start(
        &dir,
        &[
            ("PEREGRINE_DRAIN_TIMEOUT_MS", "20000"),
            ("PEREGRINE_STATE_DIR", &state_s),
        ],
        &[
            "serve",
            "-m",
            f.chat.to_str().unwrap(),
            "--alias",
            "chat",
            "-m",
            f.causal.to_str().unwrap(),
            "--alias",
            "causal",
            "-m",
            f.decision.to_str().unwrap(),
            "--alias",
            "decide",
            "--host",
            "127.0.0.1",
            "--port",
            common::PORT,
            "--ctx",
            "512",
            "--threads",
            "2",
            "--no-profile",
        ],
    );
    let mut guard = Guard(child);
    wait_for_health(port);

    let slow = json!({
        "model": "chat",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 200,
        "temperature": 0,
    });
    let stop = state.join(format!("serve-{port}.stop"));
    let (a, b, c) = thread::scope(|s| {
        let a = s.spawn(|| post(port, "/v1/chat/completions", &slow));
        let b = s.spawn(|| post(port, "/v1/embeddings", &embedding_body("causal")));
        let c = s.spawn(|| post(port, "/v1/systemone", &decision_body("decide")));
        // Let the requests reach their engines, then ask the process to drain.
        thread::sleep(Duration::from_millis(300));
        std::fs::write(&stop, b"stop\n").expect("write the stop file");
        // The server removes it as it starts the drain, then stops accepting.
        let deadline = Instant::now() + Duration::from_secs(10);
        while stop.exists() {
            assert!(
                Instant::now() < deadline,
                "the server never noticed the stop file"
            );
            thread::sleep(Duration::from_millis(20));
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while TcpStream::connect(("127.0.0.1", port)).is_ok() {
            assert!(
                Instant::now() < deadline,
                "a new connection was accepted while draining"
            );
            thread::sleep(Duration::from_millis(20));
        }
        (a.join().unwrap(), b.join().unwrap(), c.join().unwrap())
    });
    assert_eq!(a.0, 200, "{}", a.1);
    assert_eq!(b.0, 200, "{}", b.1);
    assert_eq!(c.0, 200, "{}", c.1);

    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = guard.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "the server did not exit after draining"
        );
        thread::sleep(Duration::from_millis(50));
    };
    assert!(status.success(), "the drained server exited with {status}");
    assert!(!stop.exists(), "the server must remove the stop file");
}
/// multi-model server names each file by its stem; two equal stems are refused.
#[test]
fn the_cli_takes_repeated_models_and_aliases() {
    let dir = tmpdir("cli");
    let f = files(&dir);

    // No aliases: the ids are the file stems, in `-m` order.
    let (child, port2) = common::start(
        &dir,
        &[],
        &[
            "serve",
            "-m",
            f.chat.to_str().unwrap(),
            "-m",
            f.causal.to_str().unwrap(),
            "--host",
            "127.0.0.1",
            "--port",
            common::PORT,
            "--ctx",
            "128",
            "--threads",
            "1",
            "--no-profile",
        ],
    );
    let guard = Guard(child);
    wait_for_health(port2);
    let (_, listed) = http(port2, "GET", "/v1/models", None);
    let listed: Value = serde_json::from_str(&listed).unwrap();
    let ids: Vec<&str> = listed["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&"llama-F16"));
    assert!(ids.contains(&"llama-F16-pool3"));
    drop(guard);

    // Two equal stems with no alias are refused, with a message that says what to do.
    let dup = dir.join("dup");
    std::fs::create_dir_all(&dup).unwrap();
    let a = write_dense(&dup, "llama", Format::F16, 21, Dense::default());
    let b_dir = dir.join("dup2");
    std::fs::create_dir_all(&b_dir).unwrap();
    let b = write_dense(&b_dir, "llama", Format::F16, 22, Dense::default());
    let out = Command::new(env!("CARGO_BIN_EXE_peregrine"))
        .args([
            "serve",
            "-m",
            a.to_str().unwrap(),
            "-m",
            b.to_str().unwrap(),
            "--host",
            "127.0.0.1",
            "--port",
            &common::free_port().to_string(),
            "--no-profile",
        ])
        .output()
        .expect("run peregrine serve");
    assert!(!out.status.success(), "two equal ids were accepted");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--alias"), "{stderr}");
}

/// A server with one model serves it whatever name a request sends: an OpenAI client is
/// configured with some model name, and with one model there is nothing to choose.
#[test]
fn a_one_model_server_accepts_any_model_name() {
    let dir = tmpdir("any-name");
    let chat = write_dense(&dir, "llama", Format::F16, 11, Dense::default());
    let (_guard, port) = serve(&dir, &[(&chat, "peregrine")], 2, 512);
    let (status, named) = post(port, "/v1/chat/completions", &chat_body("peregrine", 4));
    assert_eq!(status, 200, "{named}");
    for name in ["gpt-4o-mini", "definitely-not-the-served-model"] {
        let (status, other) = post(port, "/v1/chat/completions", &chat_body(name, 4));
        assert_eq!(status, 200, "{name}: {other}");
        assert_eq!(content(&other), content(&named), "{name}");
    }
}

/// `/v1/rerank` and `/v1/classify` go to the model the request names (S4.6b on a
/// several-models server), and each answer is the one that file gives alone.
#[test]
fn rerank_and_classify_are_routed_by_id() {
    let dir = tmpdir("heads");
    let dirs: Vec<PathBuf> = ["embed", "rank", "labels"]
        .iter()
        .map(|d| {
            let d = dir.join(d);
            std::fs::create_dir_all(&d).unwrap();
            d
        })
        .collect();
    let embed = write_dense_pooled(&dirs[0], "llama", Format::F16, 12, Dense::default(), 3);
    let rank = write_bert(
        &dirs[1],
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
    );
    let labels = write_bert(
        &dirs[2],
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
    );
    let rerank_body = |model: &str| {
        json!({
            "model": model,
            "query": "hello world",
            "documents": ["the cat sat on the mat", "hello world", "taxes"],
        })
    };
    let classify_body = |model: &str| json!({"model": model, "input": ["hello world", "a text"]});

    let (guard, port) = serve(
        &dir,
        &[(&embed, "embed"), (&rank, "rank"), (&labels, "labels")],
        2,
        128,
    );
    let (_, listed) = http(port, "GET", "/v1/models", None);
    let listed: Value = serde_json::from_str(&listed).unwrap();
    let modality = |i: usize| listed["data"][i]["architecture"]["output_modalities"][0].clone();
    assert_eq!(modality(1), "rerank");
    assert_eq!(modality(2), "classify");

    let (status, ranked) = post(port, "/v1/rerank", &rerank_body("rank"));
    assert_eq!(status, 200, "{ranked}");
    assert_eq!(ranked["model"], "rank");
    assert_eq!(ranked["results"].as_array().unwrap().len(), 3);
    let (status, classified) = post(port, "/v1/classify", &classify_body("labels"));
    assert_eq!(status, 200, "{classified}");
    assert_eq!(classified["model"], "labels");

    // The first `-m` is an embedder: a request that names no model, or names it, is told
    // that file has no head; an unknown id is a 404 on these routes too.
    for model in ["", "embed"] {
        let (status, refused) = post(port, "/v1/rerank", &rerank_body(model));
        assert_eq!(status, 400, "{model:?}: {refused}");
        assert!(
            refused["error"]["message"]
                .as_str()
                .unwrap()
                .contains("no classifier head"),
            "{refused}"
        );
    }
    for path in ["/v1/rerank", "/v1/classify"] {
        let (status, unknown) = post(port, path, &rerank_body("nope"));
        assert_eq!(status, 404, "{path}: {unknown}");
        assert_eq!(unknown["error"]["code"], "model_not_found");
    }
    drop(guard);

    // Each file alone gives the same scores.
    let (guard, port) = serve(&dir, &[(&rank, "rank")], 2, 128);
    let (_, alone) = post(port, "/v1/rerank", &rerank_body("rank"));
    drop(guard);
    assert_eq!(ranked["results"], alone["results"]);
    let (guard, port) = serve(&dir, &[(&labels, "labels")], 2, 128);
    let (_, alone) = post(port, "/v1/classify", &classify_body("labels"));
    drop(guard);
    assert_eq!(classified["results"], alone["results"]);
}

/// The duplicate-text embedding cache (S4.9k) is per model: the same text and the same
/// pooling on two different files must give each file's own vector, never the other's.
#[test]
fn one_models_embedding_is_never_served_for_another() {
    let dir = tmpdir("cache");
    let a_dir = dir.join("a");
    let b_dir = dir.join("b");
    std::fs::create_dir_all(&a_dir).unwrap();
    std::fs::create_dir_all(&b_dir).unwrap();
    // Same architecture, tokenizer and pooling; different weights. Their stems are equal,
    // so the server's ids must come from --alias.
    let a = write_dense_pooled(&a_dir, "llama", Format::F16, 71, Dense::default(), 3);
    let b = write_dense_pooled(&b_dir, "llama", Format::F16, 72, Dense::default(), 3);
    let body = json!({"input": ["the very same text"], "dimensions": 64});

    let (guard, port) = serve(&dir, &[(&a, "a"), (&b, "b")], 2, 128);
    let mut body_a = body.clone();
    body_a["model"] = "a".into();
    let mut body_b = body.clone();
    body_b["model"] = "b".into();
    let (status, first) = post(port, "/v1/embeddings", &body_a);
    assert_eq!(status, 200, "{first}");
    let (status, second) = post(port, "/v1/embeddings", &body_b);
    assert_eq!(status, 200, "{second}");
    drop(guard);

    let (a_guard, a_port) = serve(&dir, &[(&a, "a")], 2, 128);
    let (_, a_ref) = post(a_port, "/v1/embeddings", &body_a);
    drop(a_guard);
    let (b_guard, b_port) = serve(&dir, &[(&b, "b")], 2, 128);
    let (_, b_ref) = post(b_port, "/v1/embeddings", &body_b);
    drop(b_guard);

    assert_eq!(
        bits(&first["data"][0]["embedding"]),
        bits(&a_ref["data"][0]["embedding"]),
        "the first model's vector changed"
    );
    assert_eq!(
        bits(&second["data"][0]["embedding"]),
        bits(&b_ref["data"][0]["embedding"]),
        "the second model was served the first model's cached vector"
    );
    assert_ne!(
        bits(&first["data"][0]["embedding"]),
        bits(&second["data"][0]["embedding"]),
        "two different files produced the same vector, so the test proves nothing"
    );
}
