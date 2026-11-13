//! Tool calling on `POST /v1/chat/completions`, end to end (roadmap S6.3).
//!
//! Synthetic chat files are served by the real binary on a free port and driven over HTTP.
//! One file carries a chat template Peregrine has a tool format for (Qwen 2.5's, from
//! `peregrine_core::tools`), the other the ordinary synthetic template. A synthetic model
//! writes random tokens, so these tests pin the routing, the refusals and the response
//! shapes; what a call looks like in a reply is pinned by the endpoint's unit tests, and
//! the comparison with llama.cpp on a real file is in `docs/design/s6-tool-calling.md`.

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
use support::models::{write_dense, write_dense_with_template, Dense};

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-server-tools-{}-{name}",
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

fn post(port: u16, body: Value) -> (u16, Value) {
    let (status, text) = try_http(port, "POST", "/v1/chat/completions", Some(&body))
        .expect("the server is listening");
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
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

/// A context wide enough for a prompt that carries tool definitions.
const CTX: usize = 4096;

fn shape() -> Dense {
    Dense {
        ctx: CTX,
        ..Dense::default()
    }
}

fn serve(model: &Path, dir: &Path) -> (Guard, u16) {
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
            &CTX.to_string(),
            "--threads",
            "2",
            "--slots",
            "2",
            "--no-profile",
        ],
    );
    let guard = Guard(child);
    wait_for_health(port);
    (guard, port)
}

/// A file whose chat template has a tool format.
fn serve_tool_model(name: &str) -> (Guard, u16) {
    let dir = tmpdir(name);
    let model = write_dense_with_template(
        &dir,
        "qwen2",
        Format::F16,
        41,
        shape(),
        peregrine_core::tools::QWEN25_TEMPLATE,
    );
    serve(&model, &dir)
}

fn weather_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    })
}

fn tool_request(extra: Value) -> Value {
    let mut body = json!({
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [weather_tool()],
        "temperature": 0,
        "max_tokens": 4
    });
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    body
}

fn error_message(v: &Value) -> String {
    v["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[test]
fn a_request_with_tools_is_served_on_a_file_that_has_a_tool_format() {
    let (_server, port) = serve_tool_model("served");

    let (status, body) = post(port, tool_request(json!({})));
    assert_eq!(status, 200, "{body}");
    let choice = &body["choices"][0];
    assert_eq!(choice["message"]["role"], "assistant");
    let finish = choice["finish_reason"].as_str().unwrap();
    assert!(
        ["stop", "length", "tool_calls"].contains(&finish),
        "finish_reason {finish}"
    );
    // The usage counts the tool definitions: the prompt is far longer than the question.
    assert!(
        body["usage"]["prompt_tokens"].as_u64().unwrap() > 60,
        "{body}"
    );

    // The same request is the same reply: the tool path is deterministic at temperature 0.
    let (_, again) = post(port, tool_request(json!({})));
    assert_eq!(body["choices"], again["choices"]);

    // `tool_choice: "none"` and `"auto"` are served; `parallel_tool_calls` is read.
    for extra in [
        json!({"tool_choice": "none"}),
        json!({"tool_choice": "auto"}),
        json!({"parallel_tool_calls": false}),
    ] {
        let (status, body) = post(port, tool_request(extra.clone()));
        assert_eq!(status, 200, "{extra}: {body}");
    }

    // A conversation with a call and its result in the history is rendered and served.
    let (status, body) = post(
        port,
        json!({
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"}
            ],
            "tools": [weather_tool()],
            "temperature": 0,
            "max_tokens": 4
        }),
    );
    assert_eq!(status, 200, "{body}");

    // Streaming: server-sent chunks, each a chat.completion.chunk, then [DONE].
    let (status, text) = try_http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(&tool_request(json!({"stream": true}))),
    )
    .unwrap();
    assert_eq!(status, 200);
    assert!(text.contains("chat.completion.chunk"), "{text}");
    assert!(text.contains("[DONE]"), "{text}");

    // An ordinary request on the same file still takes the ordinary path.
    let plain = json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "max_tokens": 4
    });
    let (status, a) = post(port, plain.clone());
    assert_eq!(status, 200, "{a}");
    assert!(a["choices"][0]["message"].get("tool_calls").is_none());
    let (_, b) = post(port, plain);
    assert_eq!(a["choices"], b["choices"]);
}

/// A file that carries one of the texts of Qwen 3.5's template.
fn serve_qwen35_model(name: &str, seed: u64, template: &str) -> (Guard, u16) {
    let dir = tmpdir(name);
    let model = write_dense_with_template(&dir, "qwen2", Format::F16, seed, shape(), template);
    serve(&model, &dir)
}

fn text_of(v: &Value) -> String {
    v.as_str().unwrap_or_default().to_string()
}

/// Qwen 3.5's template opens the thinking block itself, so the reply begins as reasoning
/// and only the closing tag is the model's. A synthetic model never writes that tag: with
/// thinking on, everything it writes is reasoning; with thinking off, nothing is.
#[test]
fn a_template_that_opens_the_thinking_block_starts_the_reply_as_reasoning() {
    let (_server, port) = serve_qwen35_model(
        "qwen35-thinking",
        43,
        peregrine_core::tools::QWEN35_TEMPLATE,
    );
    let message = |extra: Value| {
        let (status, body) = post(port, tool_request(extra.clone()));
        assert_eq!(status, 200, "{extra}: {body}");
        body["choices"][0]["message"].clone()
    };

    // Thinking is on unless the request says otherwise.
    let on = message(json!({}));
    assert!(!text_of(&on["reasoning_content"]).is_empty(), "{on}");
    assert!(text_of(&on["content"]).is_empty(), "{on}");
    assert!(on.get("tool_calls").is_none(), "{on}");

    // Off, either way a client may say it: the same tokens are content.
    for extra in [
        json!({"chat_template_kwargs": {"enable_thinking": false}}),
        json!({"reasoning_effort": "none"}),
    ] {
        let off = message(extra.clone());
        assert!(
            text_of(&off["reasoning_content"]).is_empty(),
            "{extra}: {off}"
        );
        assert!(
            !text_of(&off["content"]).is_empty() || off.get("tool_calls").is_some(),
            "{extra}: {off}"
        );
    }

    // `reasoning_format: none` returns the reasoning span as content, and it is still not
    // read for calls.
    let inline = message(json!({"reasoning_format": "none"}));
    assert!(text_of(&inline["reasoning_content"]).is_empty(), "{inline}");
    // (The reasoning field is stripped at its start; the span as content is not.)
    assert_eq!(
        text_of(&inline["content"]).trim_start(),
        text_of(&on["reasoning_content"]).trim_start()
    );
    assert!(inline.get("tool_calls").is_none(), "{inline}");

    // The thinking switch changes the prompt: one more closing tag and two newlines.
    let usage = |extra: Value| {
        let (_, body) = post(port, tool_request(extra));
        body["usage"]["prompt_tokens"].as_u64().unwrap()
    };
    assert!(
        usage(json!({"chat_template_kwargs": {"enable_thinking": false}})) > usage(json!({})),
        "thinking off writes an empty block into the prompt"
    );

    // Streaming: the same split, as reasoning deltas.
    let (status, text) = try_http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(&tool_request(json!({"stream": true}))),
    )
    .unwrap();
    assert_eq!(status, 200);
    assert!(text.contains("reasoning_content"), "{text}");
    assert!(text.contains("[DONE]"), "{text}");
}

/// A file with Qwen 3.5's template has every chat request rendered by the template's own
/// rules, tools or not: without tools the prompt still ends inside the thinking block, so
/// the reply still starts as reasoning.
#[test]
fn a_request_without_tools_gets_the_same_template() {
    let (_server, port) =
        serve_qwen35_model("qwen35-plain", 45, peregrine_core::tools::QWEN35_TEMPLATE);
    let plain = |extra: Value| {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": "  Be brief.  "},
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0,
            "max_tokens": 4
        });
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        let (status, body) = post(port, body);
        assert_eq!(status, 200, "{body}");
        body
    };
    let on = plain(json!({}));
    let message = &on["choices"][0]["message"];
    assert!(
        !text_of(&message["reasoning_content"]).is_empty(),
        "{message}"
    );
    assert!(text_of(&message["content"]).is_empty(), "{message}");
    assert!(message.get("tool_calls").is_none(), "{message}");

    let off = plain(json!({"chat_template_kwargs": {"enable_thinking": false}}));
    let message = &off["choices"][0]["message"];
    assert!(
        text_of(&message["reasoning_content"]).is_empty(),
        "{message}"
    );
    assert!(!text_of(&message["content"]).is_empty(), "{message}");
    // The empty thinking block is part of the prompt when thinking is off.
    assert!(
        off["usage"]["prompt_tokens"].as_u64().unwrap()
            > on["usage"]["prompt_tokens"].as_u64().unwrap()
    );
    // The same request is the same reply.
    assert_eq!(plain(json!({}))["choices"], on["choices"]);

    // A request without tools is still refused for what it was always refused for.
    let (status, body) = post(
        port,
        json!({"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]}),
    );
    assert_eq!(status, 400, "{body}");
    assert!(error_message(&body).contains("user"), "{body}");
}

/// Qwen 3: the tools are served, the thinking switch changes the prompt, and the four
/// published wordings of the template give the same prompt and reply for an ordinary
/// request (they part only on corner cases pinned in `peregrine-core`).
#[test]
fn a_qwen3_template_serves_tools_under_every_wording() {
    let texts = std::iter::once(peregrine_core::tools::QWEN3_TEMPLATE).chain(
        peregrine_core::tools::QWEN3_VARIANTS
            .iter()
            .map(|(text, _)| *text),
    );
    let mut replies = Vec::new();
    for (i, template) in texts.enumerate() {
        let (_server, port) = serve_qwen35_model(&format!("qwen3-text-{i}"), 46, template);
        let (status, on) = post(port, tool_request(json!({})));
        assert_eq!(status, 200, "wording {i}: {on}");
        let (status, off) = post(
            port,
            tool_request(json!({"chat_template_kwargs": {"enable_thinking": false}})),
        );
        assert_eq!(status, 200, "wording {i}: {off}");
        // Thinking off writes the empty block into the prompt.
        assert!(
            off["usage"]["prompt_tokens"].as_u64().unwrap()
                > on["usage"]["prompt_tokens"].as_u64().unwrap(),
            "wording {i}"
        );
        // A request without tools is rendered by the same template.
        let (status, plain) = post(
            port,
            json!({
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hello"}
                ],
                "temperature": 0,
                "max_tokens": 4
            }),
        );
        assert_eq!(status, 200, "wording {i}: {plain}");
        replies.push((
            on["usage"]["prompt_tokens"].clone(),
            on["choices"].clone(),
            plain["usage"]["prompt_tokens"].clone(),
            plain["choices"].clone(),
        ));
    }
    assert_eq!(replies.len(), 4);
    for (i, reply) in replies.iter().enumerate() {
        assert_eq!(
            reply, &replies[0],
            "wording {i} is not served like the first"
        );
    }
}

/// The texts of Qwen 3.5's template that other publishers ship are the same format: the
/// same weights and the same request give the same prompt and the same reply.
#[test]
fn every_published_text_of_qwen35s_template_serves_tools() {
    let texts = std::iter::once(peregrine_core::tools::QWEN35_TEMPLATE)
        .chain(peregrine_core::tools::QWEN35_VARIANTS.iter().copied());
    let mut replies = Vec::new();
    for (i, template) in texts.enumerate() {
        let (_server, port) = serve_qwen35_model(&format!("qwen35-text-{i}"), 44, template);
        let (status, body) = post(port, tool_request(json!({})));
        assert_eq!(status, 200, "text {i}: {body}");
        replies.push((
            body["usage"]["prompt_tokens"].clone(),
            body["choices"].clone(),
        ));
    }
    assert_eq!(replies.len(), 4);
    for (i, reply) in replies.iter().enumerate() {
        assert_eq!(reply, &replies[0], "text {i} is not served like the first");
    }
}

#[test]
fn what_cannot_be_honoured_is_refused_by_name() {
    let (_server, port) = serve_tool_model("refusals");
    let refused = |extra: Value, word: &str| {
        let (status, body) = post(port, tool_request(extra.clone()));
        assert_eq!(status, 400, "{extra}: {body}");
        let message = error_message(&body);
        assert!(
            message.contains(word),
            "{extra}: {message:?} lacks {word:?}"
        );
    };
    refused(
        json!({"tool_choice": {"type": "function", "function": {"name": "get_stocks"}}}),
        "which is not one of the request's tools",
    );
    refused(json!({"tool_choice": "always"}), "tool_choice must be");
    refused(
        json!({"tools": [], "tool_choice": "required"}),
        "needs tools",
    );
    refused(json!({"functions": [{"name": "f"}]}), "functions");
    refused(json!({"function_call": "auto"}), "function_call");
    refused(json!({"tools": [{"type": "retrieval"}]}), "retrieval");
    refused(
        json!({"tools": [{"type": "function", "function": {}}]}),
        "name",
    );
    refused(json!({"parallel_tool_calls": "no"}), "parallel_tool_calls");
    refused(
        json!({"messages": [{"role": "developer", "content": "x"}, {"role": "user", "content": "y"}]}),
        "developer",
    );
    refused(
        json!({"messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": "{broken"}}
            ]},
            {"role": "tool", "content": "1"}
        ]}),
        "not JSON",
    );
}

/// A forced choice is a constraint on the sampler, so even a file of random weights
/// answers with calls that parse and fit the tool's schema (S6.6b). The tool's arguments
/// are a finite language here, so every reply closes well inside the token limit.
#[test]
fn a_forced_choice_answers_with_valid_calls_whatever_the_model_writes() {
    let (_server, port) = serve_tool_model("forced");
    let tools = json!([
        {"type": "function", "function": {"name": "set_unit", "description": "Set the unit",
            "parameters": {"type": "object",
                "properties": {"unit": {"enum": ["celsius", "fahrenheit"]}, "exact": {"type": "boolean"}},
                "required": ["unit"], "additionalProperties": false}}},
        {"type": "function", "function": {"name": "reset"}}
    ]);
    let check_call = |call: &Value| {
        assert_eq!(call["type"], "function", "{call}");
        let name = call["function"]["name"].as_str().expect("a name");
        let args: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap())
            .unwrap_or_else(|e| panic!("arguments are not JSON ({e}): {call}"));
        let args = args.as_object().expect("arguments are an object");
        match name {
            "set_unit" => {
                assert!(
                    matches!(args["unit"].as_str(), Some("celsius" | "fahrenheit")),
                    "{call}"
                );
                assert!(args.keys().all(|k| k == "unit" || k == "exact"), "{call}");
                assert!(args.get("exact").is_none_or(Value::is_boolean), "{call}");
            }
            "reset" => assert!(args.is_empty(), "{call}"),
            other => panic!("a call to {other:?}, which is not a tool of the request"),
        }
        name.to_string()
    };
    for seed in 0..6u64 {
        // A named function: exactly one call, to it, and nothing else in the message.
        for name in ["set_unit", "reset"] {
            let (status, body) = post(
                port,
                tool_request(json!({
                    "tools": tools, "seed": seed, "temperature": 1.0, "max_tokens": 200,
                    "tool_choice": {"type": "function", "function": {"name": name}}
                })),
            );
            assert_eq!(status, 200, "{body}");
            let choice = &body["choices"][0];
            assert_eq!(choice["finish_reason"], "tool_calls", "{body}");
            assert!(choice["message"]["content"].is_null(), "{body}");
            let calls = choice["message"]["tool_calls"].as_array().expect("calls");
            assert_eq!(calls.len(), 1, "{body}");
            assert_eq!(check_call(&calls[0]), name);
        }
        // `"required"` with one call allowed: one call to either tool.
        let (status, body) = post(
            port,
            tool_request(json!({
                "tools": tools, "seed": seed, "temperature": 1.0, "max_tokens": 200,
                "tool_choice": "required", "parallel_tool_calls": false
            })),
        );
        assert_eq!(status, 200, "{body}");
        let choice = &body["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "{body}");
        let calls = choice["message"]["tool_calls"].as_array().expect("calls");
        assert_eq!(calls.len(), 1, "{body}");
        check_call(&calls[0]);
        // `"required"` with parallel calls: at least one, each valid. A random model may
        // keep calling until the token limit; whatever it completed is still valid.
        let (status, body) = post(
            port,
            tool_request(json!({
                "tools": tools, "seed": seed, "temperature": 1.0, "max_tokens": 120,
                "tool_choice": "required"
            })),
        );
        assert_eq!(status, 200, "{body}");
        let choice = &body["choices"][0];
        let calls = choice["message"]["tool_calls"].as_array().expect("calls");
        assert!(!calls.is_empty(), "{body}");
        calls.iter().for_each(|c| {
            check_call(c);
        });
    }
}

/// Tools beside a `response_format`: the reply is calls or JSON of the response schema,
/// never anything else, on a file of random weights too (S6.6b).
#[test]
fn tools_beside_a_response_format_answer_with_calls_or_the_schemas_json() {
    let (_server, port) = serve_tool_model("either");
    let tools = json!([{"type": "function", "function": {"name": "reset"}}]);
    let format = json!({"type": "json_schema", "json_schema": {"name": "verdict", "schema": {
        "type": "object", "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"], "additionalProperties": false
    }}});
    let (mut called, mut answered) = (0, 0);
    for seed in 0..12u64 {
        let (status, body) = post(
            port,
            tool_request(json!({
                "tools": tools, "seed": seed, "temperature": 1.0, "max_tokens": 120,
                "response_format": format, "parallel_tool_calls": false
            })),
        );
        assert_eq!(status, 200, "{body}");
        let choice = &body["choices"][0];
        let message = &choice["message"];
        match message["tool_calls"].as_array() {
            Some(calls) if !calls.is_empty() => {
                called += 1;
                assert_eq!(calls.len(), 1, "{body}");
                assert_eq!(calls[0]["function"]["name"], "reset", "{body}");
                assert_eq!(calls[0]["function"]["arguments"], "{}", "{body}");
                assert!(message["content"].is_null(), "{body}");
                assert_eq!(choice["finish_reason"], "tool_calls", "{body}");
            }
            _ => {
                answered += 1;
                let content = message["content"].as_str().expect("content");
                let v: Value = serde_json::from_str(content)
                    .unwrap_or_else(|e| panic!("content is not JSON ({e}): {body}"));
                let o = v.as_object().expect("an object");
                assert_eq!(o.len(), 1, "{body}");
                assert!(o["ok"].is_boolean(), "{body}");
                assert_eq!(choice["finish_reason"], "stop", "{body}");
            }
        }
    }
    assert_eq!(called + answered, 12);
}

/// The XML call syntax (Qwen 3.5's template) under a forced choice, with thinking off so
/// the reply begins with its content: a file of random weights still answers with calls
/// that parse and fit the tool. The values here are finite languages (a boolean, an enum),
/// so every reply closes; a string value is free text, which a random model need not end.
#[test]
fn a_forced_choice_in_the_xml_syntax_answers_with_valid_calls() {
    let (_server, port) =
        serve_qwen35_model("forced-xml", 43, peregrine_core::tools::QWEN35_TEMPLATE);
    let tools = json!([
        {"type": "function", "function": {"name": "set_mode", "description": "Set the mode",
            "parameters": {"type": "object",
                "properties": {"mode": {"enum": ["fast", "exact"]}, "verbose": {"type": "boolean"},
                               "level": {"enum": [1, 2, 3]}},
                "required": ["mode", "verbose"]}}},
        {"type": "function", "function": {"name": "reset"}}
    ]);
    for seed in 0..6u64 {
        for choice in [
            json!("required"),
            json!({"type": "function", "function": {"name": "set_mode"}}),
        ] {
            let (status, body) = post(
                port,
                tool_request(json!({
                    "tools": tools, "seed": seed, "temperature": 1.0, "max_tokens": 200,
                    "tool_choice": choice, "parallel_tool_calls": false,
                    "chat_template_kwargs": {"enable_thinking": false}
                })),
            );
            assert_eq!(status, 200, "{body}");
            let c = &body["choices"][0];
            assert_eq!(c["finish_reason"], "tool_calls", "{body}");
            let calls = c["message"]["tool_calls"].as_array().expect("calls");
            assert_eq!(calls.len(), 1, "{body}");
            let name = calls[0]["function"]["name"].as_str().unwrap();
            let args: Value =
                serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap())
                    .unwrap_or_else(|e| panic!("arguments are not JSON ({e}): {body}"));
            match name {
                "set_mode" => {
                    assert!(
                        matches!(args["mode"].as_str(), Some("fast" | "exact")),
                        "{body}"
                    );
                    assert!(args["verbose"].is_boolean(), "{body}");
                    assert!(
                        args.get("level")
                            .is_none_or(|l| matches!(l.as_i64(), Some(1..=3))),
                        "{body}"
                    );
                }
                "reset" => {
                    assert!(
                        choice.is_string(),
                        "the named choice called another tool: {body}"
                    );
                    assert_eq!(args, json!({}), "{body}");
                }
                other => panic!("a call to {other:?}: {body}"),
            }
        }
    }
    // Thinking on: the forced call follows the model's reasoning. A random model never
    // closes its thinking, so the reply is reasoning up to the token limit; the request
    // is served, not refused.
    let (status, body) = post(
        port,
        tool_request(json!({"tools": tools, "tool_choice": "required", "max_tokens": 24})),
    );
    assert_eq!(status, 200, "{body}");
    let c = &body["choices"][0];
    assert_eq!(c["finish_reason"], "length", "{body}");
    assert!(
        c["message"]["tool_calls"]
            .as_array()
            .is_none_or(Vec::is_empty),
        "{body}"
    );
}

/// Llama 3.2's template (S6.4): a forced choice is one call in the `llama3` syntax, even
/// with parallel calls on, and the date a request pins is the one the prompt writes.
#[test]
fn a_forced_choice_on_llama3_answers_with_exactly_one_valid_call() {
    let (_server, port) =
        serve_qwen35_model("forced-llama3", 47, peregrine_core::tools::LLAMA32_TEMPLATE);
    // Enumerated values, so a random model's call ends within the token limit.
    let tools = json!([
        {"type": "function", "function": {"name": "set_unit", "description": "Set the unit",
            "parameters": {"type": "object",
                "properties": {"unit": {"enum": ["celsius", "fahrenheit"]}, "exact": {"type": "boolean"}},
                "required": ["unit"], "additionalProperties": false}}},
        {"type": "function", "function": {"name": "reset"}}
    ]);
    for seed in 0..6u64 {
        for choice in [
            json!("required"),
            json!({"type": "function", "function": {"name": "set_unit"}}),
        ] {
            let (status, body) = post(
                port,
                tool_request(json!({
                    "tools": tools, "seed": seed, "temperature": 1.0, "max_tokens": 120,
                    "tool_choice": choice, "parallel_tool_calls": true
                })),
            );
            assert_eq!(status, 200, "{body}");
            let c = &body["choices"][0];
            assert_eq!(c["finish_reason"], "tool_calls", "{body}");
            let calls = c["message"]["tool_calls"].as_array().expect("calls");
            assert_eq!(calls.len(), 1, "one call per reply: {body}");
            let name = calls[0]["function"]["name"].as_str().unwrap();
            let args: Value =
                serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap())
                    .unwrap_or_else(|e| panic!("arguments are not JSON ({e}): {body}"));
            match name {
                "set_unit" => {
                    assert!(
                        matches!(args["unit"].as_str(), Some("celsius" | "fahrenheit")),
                        "{body}"
                    );
                    assert!(args.get("exact").is_none_or(Value::is_boolean), "{body}");
                }
                "reset" => assert!(choice.is_string(), "the named choice called reset: {body}"),
                other => panic!("a call to {other:?}: {body}"),
            }
        }
    }
    // Without a forced choice the random model writes text, which is served as text.
    let (status, body) = post(
        port,
        tool_request(json!({"tools": tools, "max_tokens": 8,
            "chat_template_kwargs": {"date_string": "26 Jul 2024"}})),
    );
    assert_eq!(status, 200, "{body}");
}

#[test]
fn tools_on_a_file_without_a_tool_format_are_refused() {
    let dir = tmpdir("no-format");
    let model = write_dense(&dir, "qwen2", Format::F16, 42, shape());
    let (_server, port) = serve(&model, &dir);

    let (status, body) = post(port, tool_request(json!({})));
    assert_eq!(status, 400, "{body}");
    assert!(
        error_message(&body).contains("no tool format"),
        "{}",
        error_message(&body)
    );

    // A tool conversation without `tools` takes the tool path too, and is refused the same way.
    let (status, body) = post(
        port,
        json!({"messages": [
            {"role": "user", "content": "x"},
            {"role": "tool", "content": "1"}
        ]}),
    );
    assert_eq!(status, 400, "{body}");
    assert!(error_message(&body).contains("no tool format"));

    // An ordinary request is served as before.
    let (status, body) = post(
        port,
        json!({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 4}),
    );
    assert_eq!(status, 200, "{body}");
}
