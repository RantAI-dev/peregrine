//! The tool-calling renderers against prompts llama.cpp itself rendered (S6.2).
//!
//! `tests/fixtures/tools-*.jsonl` is written by `scripts/tool-fixtures.py`: each line is a
//! chat request (`messages`, and `tools` when the case has them), exactly as it was sent to
//! llama.cpp v0.9.0's `POST /apply-template`, and the prompt llama.cpp built for it from the
//! model's own template. This test rebuilds the prompt from the request alone and compares
//! it byte for byte. It needs no model file.

use peregrine_core::json::Value;
use peregrine_core::tools::{
    messages_from_json_for, render_with, tools_from_json, Qwen3Text, RenderOptions, ToolFormat,
    QWEN3_TEMPLATE, QWEN3_VARIANTS,
};

fn check(fixture: &str, format: ToolFormat) -> usize {
    check_each(fixture, &|_| format)
}

/// [`check`] with the format chosen per case (a fixture that mixes template wordings).
fn check_each(fixture: &str, format_of: &dyn Fn(&Value) -> ToolFormat) -> usize {
    let mut seen = 0;
    for (n, line) in fixture.lines().enumerate() {
        let case = Value::parse(line).unwrap_or_else(|e| panic!("line {}: {e}", n + 1));
        let format = format_of(&case);
        let name = case.get("name").and_then(Value::as_str).expect("name");
        let request = case
            .get("request")
            .and_then(Value::as_str)
            .expect("request");
        let request = Value::parse(request).unwrap_or_else(|e| panic!("{name}: {e}"));
        let tools = tools_from_json(request.get("tools")).unwrap_or_else(|e| panic!("{name}: {e}"));
        let messages = messages_from_json_for(format, request.get("messages").expect("messages"));
        // `chat_template_kwargs.enable_thinking: false` is how a request turns thinking off.
        let thinking = request
            .get("chat_template_kwargs")
            .and_then(|k| k.get("enable_thinking"))
            != Some(&Value::Bool(false));
        let options = RenderOptions { thinking };
        match case.get("prompt").and_then(Value::as_str) {
            Some(want) => {
                let messages = messages.unwrap_or_else(|e| panic!("{name}: {e}"));
                let got = render_with(format, &tools, &messages, options)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
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
                    Ok(m) => render_with(format, &tools, &m, options).is_err(),
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

#[test]
fn qwen35_prompts_match_llama_cpp_byte_for_byte() {
    let n = check(
        include_str!("fixtures/tools-qwen35.jsonl"),
        ToolFormat::QwenXml,
    );
    assert_eq!(n, 51, "every case of the fixture was compared");
    println!("tools (qwen 3.5 / bonsai 2, qwen-xml): {n} cases agree with llama.cpp");
}

#[test]
fn qwen3_prompts_match_llama_cpp_byte_for_byte() {
    // llama.cpp rendered these with the repository's template; the three other published
    // wordings gave the same bytes for every case (`tool-fixtures.py --check`), so each
    // wording's rules must reproduce them too.
    for text in [
        Qwen3Text::Upstream,
        Qwen3Text::QwenGguf,
        Qwen3Text::UnslothLarge,
    ] {
        let n = check(
            include_str!("fixtures/tools-qwen3.jsonl"),
            ToolFormat::Qwen3(text),
        );
        assert_eq!(n, 51, "every case of the fixture was compared ({text:?})");
    }
    println!("tools (qwen 3): 51 cases agree with llama.cpp under each wording's rules");
}

#[test]
fn qwen3_wordings_part_where_llama_cpp_parts_them() {
    // Eleven corner cases rendered by llama.cpp with each of the four published wordings
    // of the template: a `<tool_response>` block inside a user's text, thinking that is
    // only whitespace, messages that are not trimmed, a role the template passes over, a
    // continued assistant text. Each line names the wording it was rendered with.
    let format_of = |case: &Value| {
        let template = match case.get("template").and_then(Value::as_str) {
            Some("qwen3") => QWEN3_TEMPLATE,
            Some("qwen3-gguf") => QWEN3_VARIANTS[0].0,
            Some("qwen3-unsloth") => QWEN3_VARIANTS[1].0,
            Some("qwen3-unsloth-large") => QWEN3_VARIANTS[2].0,
            other => panic!("unknown template {other:?}"),
        };
        ToolFormat::for_template(template).expect("a recognised wording")
    };
    let n = check_each(include_str!("fixtures/tools-qwen3-edges.jsonl"), &format_of);
    assert_eq!(n, 44, "eleven cases under four wordings");
    println!("tools (qwen 3): {n} corner cases agree with llama.cpp, wording by wording");
}

/// Llama 3.2's template (S6.4). The fixture pins the template's clock with
/// `chat_template_kwargs.date_string`, which the renderer takes as its date.
#[test]
fn llama3_prompts_match_llama_cpp_byte_for_byte() {
    let mut seen = 0;
    for (n, line) in include_str!("fixtures/tools-llama3.jsonl")
        .lines()
        .enumerate()
    {
        let case = Value::parse(line).unwrap_or_else(|e| panic!("line {}: {e}", n + 1));
        let name = case.get("name").and_then(Value::as_str).expect("name");
        let request = case
            .get("request")
            .and_then(Value::as_str)
            .expect("request");
        let request = Value::parse(request).unwrap_or_else(|e| panic!("{name}: {e}"));
        let tools = tools_from_json(request.get("tools")).unwrap_or_else(|e| panic!("{name}: {e}"));
        let messages = messages_from_json_for(
            ToolFormat::Hermes,
            request.get("messages").expect("messages"),
        );
        let date = request
            .get("chat_template_kwargs")
            .and_then(|k| k.get("date_string"))
            .and_then(Value::as_str)
            .expect("the fixture pins the date");
        let rendered =
            messages.and_then(|m| peregrine_core::tools::render_llama3(&tools, &m, date));
        match case.get("prompt").and_then(Value::as_str) {
            Some(want) => {
                let got = rendered.unwrap_or_else(|e| panic!("{name}: {e}"));
                if got != want {
                    let i = got
                        .bytes()
                        .zip(want.bytes())
                        .position(|(a, b)| a != b)
                        .unwrap_or(got.len().min(want.len()));
                    let mut lo = i.saturating_sub(60);
                    while !got.is_char_boundary(lo) {
                        lo -= 1;
                    }
                    let end = |t: &str| {
                        let mut e = (i + 80).min(t.len());
                        while !t.is_char_boundary(e) {
                            e -= 1;
                        }
                        e
                    };
                    panic!(
                        "{name}: the prompt differs from llama.cpp's at byte {i}\n  ours:   {:?}\n  theirs: {:?}",
                        &got[lo..end(&got)],
                        &want[lo.min(want.len())..end(want)]
                    );
                }
            }
            None => assert!(rendered.is_err(), "{name}: llama.cpp refused this request"),
        }
        seen += 1;
    }
    assert_eq!(seen, 34);
}
