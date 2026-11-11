//! The tool-calling renderers against prompts llama.cpp itself rendered (S6.2).
//!
//! `tests/fixtures/tools-*.jsonl` is written by `scripts/tool-fixtures.py`: each line is a
//! chat request (`messages`, and `tools` when the case has them), exactly as it was sent to
//! llama.cpp v0.9.0's `POST /apply-template`, and the prompt llama.cpp built for it from the
//! model's own template. This test rebuilds the prompt from the request alone and compares
//! it byte for byte. It needs no model file.

use peregrine_core::json::Value;
use peregrine_core::tools::{messages_from_json, render, tools_from_json, ToolFormat};

fn check(fixture: &str, format: ToolFormat) -> usize {
    let mut seen = 0;
    for (n, line) in fixture.lines().enumerate() {
        let case = Value::parse(line).unwrap_or_else(|e| panic!("line {}: {e}", n + 1));
        let name = case.get("name").and_then(Value::as_str).expect("name");
        let request = case
            .get("request")
            .and_then(Value::as_str)
            .expect("request");
        let request = Value::parse(request).unwrap_or_else(|e| panic!("{name}: {e}"));
        let tools = tools_from_json(request.get("tools")).unwrap_or_else(|e| panic!("{name}: {e}"));
        let messages = messages_from_json(request.get("messages").expect("messages"));
        match case.get("prompt").and_then(Value::as_str) {
            Some(want) => {
                let messages = messages.unwrap_or_else(|e| panic!("{name}: {e}"));
                let got =
                    render(format, &tools, &messages).unwrap_or_else(|e| panic!("{name}: {e}"));
                if got != want {
                    let i = got
                        .char_indices()
                        .zip(want.chars())
                        .find(|((_, a), b)| a != b)
                        .map(|((i, _), _)| i)
                        .unwrap_or(got.len().min(want.len()));
                    let lo = got[..i].char_indices().rev().nth(60).map_or(0, |(k, _)| k);
                    let end = |t: &str| {
                        t[i..]
                            .char_indices()
                            .nth(80)
                            .map_or(t.len(), |(k, _)| i + k)
                    };
                    panic!(
                        "{name}: the prompt differs from llama.cpp's at byte {i}\n  ours:  {:?}\n  theirs: {:?}",
                        &got[lo..end(&got)],
                        &want[lo.min(want.len())..end(want).min(want.len())]
                    );
                }
            }
            // A request llama.cpp refused must be refused here too.
            None => {
                let refused = match messages {
                    Err(_) => true,
                    Ok(m) => render(format, &tools, &m).is_err(),
                };
                assert!(refused, "{name}: llama.cpp refused this request");
            }
        }
        seen += 1;
    }
    seen
}

#[test]
fn qwen25_prompts_match_llama_cpp_byte_for_byte() {
    let n = check(
        include_str!("fixtures/tools-qwen25.jsonl"),
        ToolFormat::Hermes,
    );
    assert_eq!(n, 34, "every case of the fixture was compared");
    println!("tools (qwen 2.5, hermes): {n} cases agree with llama.cpp");
}
