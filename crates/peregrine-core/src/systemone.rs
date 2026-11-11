//! System One prompts: the text a decision model reads for one question (roadmap S4.3).
//!
//! A native decision model carries a Jinja template, `tokenizer.chat_template.systemone`, and
//! llama.cpp renders one prompt per question from it (`server_decision_context::render`,
//! `tools/server/server-decision.cpp`, v0.9.0). Peregrine has no Jinja engine and does not gain
//! one (`docs/design/s4-embeddings-decisions.md`, decision 6): each supported decision type has a
//! renderer here, written from that type's template and checked byte for byte against prompts
//! llama.cpp itself rendered (`tests/systemone_render.rs`, `tests/fixtures/systemone-*.jsonl`).
//!
//! A renderer is only valid for the template it was written from, so each one comes with the
//! template text and a check: a file whose template differs is refused by the caller rather than
//! rendered approximately.
//!
//! The values a request supplies (`state`, the instructions, option descriptions) may be any
//! JSON; a string goes into the prompt as it is and anything else as the JSON text llama.cpp's
//! `tojson` writes ([`Value::to_jinja_json`]).

use crate::json::Value;

/// The three question types of the System One API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    /// One of several named options.
    Choice,
    /// A level on an ordered scale; the option keys are the level indices.
    Score,
    /// True or false; the option keys are `"true"` and `"false"`.
    Noul,
}

impl QuestionKind {
    /// The name the API and the templates use.
    pub fn name(self) -> &'static str {
        match self {
            QuestionKind::Choice => "choice",
            QuestionKind::Score => "score",
            QuestionKind::Noul => "noul",
        }
    }
}

/// One answer a question offers, as the template sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptOption {
    pub key: String,
    /// The description from the request's `criteria`; `Value::Null` when there is none.
    pub description: Value,
}

/// One question of a request, in the order its options are scored.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptQuestion {
    pub kind: QuestionKind,
    pub instructions: Value,
    pub options: Vec<PromptOption>,
}

impl PromptQuestion {
    /// A question whose options are `keys` with empty descriptions and no instructions, for the
    /// answer math and its tests. The endpoint builds a question from a request's `criteria`
    /// ([`options_from_criteria`]).
    pub fn new(kind: QuestionKind, keys: &[&str]) -> PromptQuestion {
        PromptQuestion {
            kind,
            instructions: Value::Null,
            options: keys
                .iter()
                .map(|k| PromptOption {
                    key: (*k).to_string(),
                    description: Value::Str(String::new()),
                })
                .collect(),
        }
    }
}

/// How a decision type orders a question's options (the flags `parse_questions` reads in
/// llama.cpp's `server-decision.cpp`). The order matters twice: it is the order the prompt lists
/// the options in and the order their scores are read in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OptionRules {
    /// A `noul` question's options are `[true, false]` instead of `[false, true]`.
    pub noul_true_first: bool,
    /// A `choice` question's options are sorted by key instead of kept in request order.
    pub choice_sorted: bool,
}

/// The option rules of the `openjev` type.
pub const OPENJEV_RULES: OptionRules = OptionRules {
    noul_true_first: true,
    choice_sorted: false,
};

/// The options of a question from its `criteria` (`Value::Null` when the request gave none), as
/// llama.cpp's `parse_questions` derives them:
///
/// - `choice`: a non-empty object, one option per key with its description, in request order;
/// - `score`: an array of 2 to 10 level descriptions, the keys are the level indices `"0"`, `"1"`…;
/// - `noul`: `false` then `true` (or the reverse, per `rules`), described by an optional object.
///
/// The errors are llama.cpp's messages, without the `questions.<id>: ` prefix the caller adds.
pub fn options_from_criteria(
    kind: QuestionKind,
    criteria: &Value,
    rules: OptionRules,
) -> Result<Vec<PromptOption>, String> {
    match kind {
        QuestionKind::Choice => {
            let pairs = criteria
                .as_object()
                .filter(|o| !o.is_empty())
                .ok_or("\"criteria\" must be a non-empty object")?;
            let mut options: Vec<PromptOption> = pairs
                .iter()
                .map(|(key, description)| PromptOption {
                    key: key.clone(),
                    description: description.clone(),
                })
                .collect();
            if rules.choice_sorted {
                options.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
            }
            Ok(options)
        }
        QuestionKind::Score => {
            let levels = criteria
                .as_array()
                .filter(|a| (2..=10).contains(&a.len()))
                .ok_or("\"criteria\" must be an array of 2 to 10 levels")?;
            Ok(levels
                .iter()
                .enumerate()
                .map(|(i, description)| PromptOption {
                    key: i.to_string(),
                    description: description.clone(),
                })
                .collect())
        }
        QuestionKind::Noul => {
            if !criteria.is_null() && criteria.as_object().is_none() {
                return Err("\"criteria\" must be an object".into());
            }
            let mut options: Vec<PromptOption> = ["false", "true"]
                .iter()
                .map(|key| PromptOption {
                    key: (*key).to_string(),
                    description: criteria.get(key).cloned().unwrap_or(Value::Null),
                })
                .collect();
            if rules.noul_true_first {
                options.swap(0, 1);
            }
            Ok(options)
        }
    }
}

/// A value in a prompt: a string as it is, anything else as JSON text. This is the template
/// expression `x if x is string else x | tojson`.
fn text(v: &Value) -> std::borrow::Cow<'_, str> {
    match v {
        Value::Str(s) => s.as_str().into(),
        other => other.to_jinja_json().into(),
    }
}

/// The `systemone` template of the `openjev` decision type, as llama.cpp's converter writes it
/// (`conversion/qwen.py`, `OpenJevModel`) and as the published files carry it.
pub const OPENJEV_TEMPLATE: &str = include_str!("systemone/openjev.jinja");

/// The option labels of `openjev`, in order: option `i` is shown as, and answered with, letter `i`.
pub const OPENJEV_LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Whether `template` is the one [`render_openjev`] was written from.
pub fn is_openjev_template(template: &str) -> bool {
    template == OPENJEV_TEMPLATE
}

/// The prompt of one `openjev` question about `state`, without images: exactly what llama.cpp
/// renders from [`OPENJEV_TEMPLATE`].
///
/// More options than there are letters is an error (llama.cpp refuses such a request before it
/// renders anything).
pub fn render_openjev(question: &PromptQuestion, state: &Value) -> Result<String, String> {
    let letters = OPENJEV_LETTERS.as_bytes();
    if question.options.len() > letters.len() {
        return Err(format!(
            "too many options ({}), this model supports at most {}",
            question.options.len(),
            letters.len()
        ));
    }
    let mut p = String::new();
    p.push_str("<|im_start|>user\nState:\n");
    p.push_str(&text(state));
    p.push_str("\n\nQuestion: ");
    p.push_str(&text(&question.instructions));
    if question.kind == QuestionKind::Score {
        p.push_str(" Rate along the ordered levels below (lowest first).");
    }
    p.push_str("\nOptions:\n");
    for (i, o) in question.options.iter().enumerate() {
        p.push('[');
        p.push(letters[i] as char);
        p.push_str("] ");
        let described = o.description.is_truthy();
        if question.kind != QuestionKind::Noul {
            p.push_str(&o.key);
            p.push_str(": ");
            if described {
                p.push_str(&text(&o.description));
            }
        } else if o.key == "true" {
            p.push_str("yes: ");
            if described {
                p.push_str(&text(&o.description));
            } else {
                p.push_str("The statement is true.");
            }
        } else {
            p.push_str("no: ");
            if described {
                p.push_str(&text(&o.description));
            } else {
                p.push_str("The statement is false.");
            }
        }
        p.push('\n');
    }
    p.push_str(
        "\nAnswer with the letter of the best option only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_template_is_the_one_the_renderer_was_written_from() {
        assert!(is_openjev_template(OPENJEV_TEMPLATE));
        assert!(!is_openjev_template(
            &OPENJEV_TEMPLATE.replace("State:", "Status:")
        ));
        // The pieces the renderer hard-codes are in the template it claims to follow.
        for piece in [
            OPENJEV_LETTERS,
            "State:",
            "Question: ",
            " Rate along the ordered levels below (lowest first).",
            "The statement is true.",
            "The statement is false.",
            "Answer with the letter of the best option only.",
        ] {
            assert!(OPENJEV_TEMPLATE.contains(piece), "{piece:?}");
        }
    }

    #[test]
    fn more_options_than_letters_is_refused() {
        let q = PromptQuestion {
            kind: QuestionKind::Choice,
            instructions: Value::Str("pick".into()),
            options: (0..53)
                .map(|i| PromptOption {
                    key: format!("k{i}"),
                    description: Value::Null,
                })
                .collect(),
        };
        assert!(render_openjev(&q, &Value::Str("s".into())).is_err());
    }
}
