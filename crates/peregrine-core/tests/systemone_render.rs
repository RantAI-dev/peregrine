//! The System One prompt renderers against prompts llama.cpp itself rendered.
//!
//! `tests/fixtures/systemone-openjev.jsonl` was written by `scripts/systemone-fixtures.py`: each
//! line is a `/v1/systemone` request with one question, the prompt llama.cpp v0.9.0 rendered for
//! it, the token ids of that prompt and the server's answer. This test rebuilds the prompt from
//! the request alone and compares it byte for byte. It needs no model file.

use peregrine_core::json::Value;
use peregrine_core::systemone::{
    options_from_criteria, render_openjev, PromptQuestion, QuestionKind, OPENJEV_RULES,
};

fn kind_of(name: &str) -> QuestionKind {
    match name {
        "choice" => QuestionKind::Choice,
        "score" => QuestionKind::Score,
        "noul" => QuestionKind::Noul,
        other => panic!("unknown question type {other:?}"),
    }
}

#[test]
fn openjev_prompts_match_llama_cpp_byte_for_byte() {
    let fixture = include_str!("fixtures/systemone-openjev.jsonl");
    let mut seen = 0;
    for (n, line) in fixture.lines().enumerate() {
        let case = Value::parse(line).unwrap_or_else(|e| panic!("line {}: {e}", n + 1));
        let state = case.get("state").expect("state");
        let q = case.get("question").expect("question");
        let kind = kind_of(q.get("type").and_then(Value::as_str).expect("type"));
        let criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
        let question = PromptQuestion {
            kind,
            instructions: q.get("instructions").cloned().unwrap_or(Value::Null),
            options: options_from_criteria(kind, &criteria, OPENJEV_RULES)
                .unwrap_or_else(|e| panic!("line {}: {e}", n + 1)),
        };
        let want = case.get("prompt").and_then(Value::as_str).expect("prompt");
        let got = render_openjev(&question, state).expect("renders");
        assert_eq!(got, want, "line {}: {line}", n + 1);
        seen += 1;
    }
    assert!(seen >= 40, "only {seen} fixture cases");
}

#[test]
fn the_fixture_covers_every_question_type_and_the_option_limits() {
    let fixture = include_str!("fixtures/systemone-openjev.jsonl");
    let mut kinds = std::collections::BTreeSet::new();
    let mut most_options = 0;
    for line in fixture.lines() {
        let case = Value::parse(line).unwrap();
        let q = case.get("question").unwrap();
        kinds.insert(q.get("type").and_then(Value::as_str).unwrap().to_string());
        if let Some(c) = q.get("criteria").and_then(Value::as_object) {
            most_options = most_options.max(c.len());
        }
    }
    assert_eq!(kinds.len(), 3, "{kinds:?}");
    assert_eq!(most_options, 52);
}

/// Where model files live: `models/` in the repository, then `~/models/`.
fn model_file(name: &str) -> Option<std::path::PathBuf> {
    let mut dirs = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")];
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(std::path::Path::new(&home).join("models"));
    }
    dirs.into_iter().map(|d| d.join(name)).find(|p| p.exists())
}

/// With the model file present: the file's template is the one the renderer was written from,
/// and tokenising each rendered prompt the way llama.cpp does for a decision (no BOS added,
/// special tokens parsed) gives llama.cpp's token ids. Skipped when the file is absent.
#[test]
fn openjev_prompt_tokens_match_llama_cpp_on_the_model_file() {
    const FILE: &str = "tinyopenjev-for-testing-Q8_0.gguf";
    let Some(path) = model_file(FILE) else {
        eprintln!("  SKIPPED: openjev prompt tokens (model file {FILE} not found under models/ or ~/models/)");
        return;
    };
    let g = peregrine_core::gguf::Gguf::open_plain(&path).expect("the model file opens");
    let template = g
        .str("tokenizer.chat_template.systemone")
        .expect("the file has a systemone template");
    assert!(
        peregrine_core::systemone::is_openjev_template(template),
        "the file's systemone template is not the one the renderer was written from"
    );
    let tk = peregrine_core::tokenizer::Tokenizer::from_gguf(&g).expect("the tokenizer loads");
    let fixture = include_str!("fixtures/systemone-openjev.jsonl");
    for (n, line) in fixture.lines().enumerate() {
        let case = Value::parse(line).unwrap();
        let prompt = case.get("prompt").and_then(Value::as_str).unwrap();
        let want: Vec<u32> = case
            .get("tokens")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|t| match t {
                Value::Int(i) => *i as u32,
                other => panic!("token {other:?}"),
            })
            .collect();
        assert_eq!(tk.encode(prompt), want, "line {}: tokens differ", n + 1);
    }
}
