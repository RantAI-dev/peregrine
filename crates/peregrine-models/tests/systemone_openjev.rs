//! An `openjev` decision end to end against llama.cpp's answers (roadmap S4.3).
//!
//! `crates/peregrine-core/tests/fixtures/systemone-openjev.jsonl` holds 48 `/v1/systemone`
//! requests and the answers llama.cpp v0.9.0 gave on `tinyopenjev-for-testing-Q8_0.gguf`
//! (`scripts/systemone-fixtures.py`). With that file under `models/` or `~/models/`, this test
//! answers every request with Peregrine's own pieces — the prompt renderer, the tokenizer, a
//! labels slot in the engine, the answer math with the file's temperatures — and compares. Without
//! the file it prints `SKIPPED`; CI has no model files.
//!
//! The two engines do not compute the same bits (each quantises activations its own way), so the
//! comparison has a bound: probabilities, the expected score and the `noul` value within 5e-3 and
//! the confidence within 1e-2, with the chosen option identical. Measured on this file: 2.5e-3 at
//! most (1.6e-3 with an f32 cache), and Peregrine's own kernel paths differ from each other by
//! about 1e-3. The file is a 4-block cut of the model with a 64-wide state, which amplifies such
//! differences; a real model is expected to sit well inside the bound.

use std::sync::Arc;

use peregrine_core::decision::{format_answer, Answer, DecisionInfo, DecisionType};
use peregrine_core::json::Value;
use peregrine_core::kv::KvPrec;
use peregrine_core::serve::{Engine, SlotCfg};
use peregrine_core::systemone::{
    is_openjev_template, options_from_criteria, render_openjev, PromptQuestion, QuestionKind,
    OPENJEV_LETTERS, OPENJEV_RULES,
};
use peregrine_core::tokenizer::Tokenizer;

const FILE: &str = "tinyopenjev-for-testing-Q8_0.gguf";

/// Where model files live: `models/` in the repository, then `~/models/`.
fn model_file(name: &str) -> Option<std::path::PathBuf> {
    let mut dirs = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")];
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(std::path::Path::new(&home).join("models"));
    }
    dirs.into_iter().map(|d| d.join(name)).find(|p| p.exists())
}

fn number(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        other => panic!("not a number: {other:?}"),
    }
}

/// The largest difference between our `(key, probability)` list and llama.cpp's object, which
/// must list the same keys in the same order.
fn probability_gap(ours: &[(String, f64)], theirs: &Value) -> f64 {
    let theirs = theirs.as_object().expect("probabilities");
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|((key, p), (their_key, their_p))| {
            assert_eq!(key, their_key, "options are in a different order");
            (p - number(their_p)).abs()
        })
        .fold(0.0, f64::max)
}

#[test]
fn openjev_answers_match_llama_cpp_on_the_model_file() {
    let Some(path) = model_file(FILE) else {
        eprintln!(
            "  SKIPPED: openjev answers (model file {FILE} not found under models/ or ~/models/)"
        );
        return;
    };
    let g = peregrine_core::Gguf::open(&path).expect("the model file opens");
    let info = DecisionInfo::from_gguf(&g)
        .expect("valid decision metadata")
        .expect("a decision model");
    assert_eq!(info.kind(), &DecisionType::OpenJev);
    assert!(is_openjev_template(
        g.str("tokenizer.chat_template.systemone")
            .expect("the systemone template")
    ));
    let tk = Tokenizer::from_gguf(&g).expect("the tokenizer loads");
    // Option `i` is answered with letter `i`; each letter must be one token.
    let letters: Vec<u32> = OPENJEV_LETTERS
        .chars()
        .map(|c| {
            let ids = tk.encode(&c.to_string());
            assert_eq!(ids.len(), 1, "letter {c} is not one token");
            ids[0]
        })
        .collect();
    let model = Arc::new(peregrine_models::load(g).expect("the model loads"));

    let fixture = include_str!("../../peregrine-core/tests/fixtures/systemone-openjev.jsonl");
    let (mut worst_value, mut worst_confidence) = (0f64, 0f64);
    let mut cases = 0;
    for (n, line) in fixture.lines().enumerate() {
        let case = Value::parse(line).expect("a fixture line");
        let q = case.get("question").expect("question");
        let kind = match q.get("type").and_then(Value::as_str).expect("type") {
            "choice" => QuestionKind::Choice,
            "score" => QuestionKind::Score,
            "noul" => QuestionKind::Noul,
            other => panic!("unknown question type {other:?}"),
        };
        let criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
        let options = options_from_criteria(kind, &criteria, OPENJEV_RULES).expect("options");
        let prompt_question = PromptQuestion {
            kind,
            instructions: q.get("instructions").cloned().unwrap_or(Value::Null),
            options,
        };
        let prompt = render_openjev(&prompt_question, case.get("state").expect("state"))
            .expect("the prompt renders");

        // One labels slot: prefill the prompt, read the last row's logits at the option letters.
        let mut engine = Engine::new(model.clone(), 4, 4096, 1, KvPrec::F16, 0);
        engine
            .admit_labels(
                tk.encode(&prompt),
                SlotCfg::default(),
                letters[..prompt_question.options.len()].to_vec(),
            )
            .map_err(|(_, e)| e)
            .expect("the labels slot is admitted");
        let mut scores = None;
        while engine.active() > 0 {
            for emission in engine.step() {
                if emission.finish.is_some() {
                    scores = emission.label_logits.clone();
                }
            }
        }
        let scores = scores.expect("the slot returned its label logits");

        let answer = format_answer(
            info.kind(),
            &prompt_question,
            &[scores],
            info.temperatures(),
            info.score_bias(),
        )
        .expect("an answer");
        let want = case.get("answer").expect("llama.cpp's answer");
        match &answer {
            Answer::Choice {
                choice,
                probabilities,
                confidence,
            } => {
                assert_eq!(
                    Some(choice.as_str()),
                    want.get("choice").and_then(Value::as_str),
                    "case {}: a different option was chosen",
                    n + 1
                );
                worst_value = worst_value.max(probability_gap(
                    probabilities,
                    want.get("probabilities").unwrap(),
                ));
                worst_confidence = worst_confidence
                    .max((confidence - number(want.get("confidence").unwrap())).abs());
            }
            Answer::Score {
                score,
                probabilities,
                confidence,
                ..
            } => {
                worst_value = worst_value
                    .max(probability_gap(
                        probabilities,
                        want.get("probabilities").unwrap(),
                    ))
                    .max((score - number(want.get("score").unwrap())).abs());
                worst_confidence = worst_confidence
                    .max((confidence - number(want.get("confidence").unwrap())).abs());
            }
            Answer::Noul { noul } => {
                worst_value = worst_value.max((noul - number(want.get("noul").unwrap())).abs());
            }
        }
        cases += 1;
    }
    eprintln!(
        "  openjev vs llama.cpp, {cases} cases: values within {worst_value:.1e}, confidence within {worst_confidence:.1e}"
    );
    assert!(cases >= 40);
    assert!(worst_value < 5e-3, "values differ by {worst_value:e}");
    assert!(
        worst_confidence < 1e-2,
        "confidence differs by {worst_confidence:e}"
    );
}
