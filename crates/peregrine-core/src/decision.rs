//! System One decision models (roadmap S4.3): the answer math and the file's decision metadata.
//!
//! This module is the part of `docs/design/s4-embeddings-decisions.md` §2.5 that does not render a
//! prompt: the arithmetic that turns one raw score per option into a `choice`, a `score` or a `noul`
//! answer, and the metadata a decision GGUF carries. Everything here is transcribed from llama.cpp
//! v0.9.0 (`tools/server/server-decision.cpp`); the note `docs/reference/systemone.md` §1 and §2 quotes
//! the code with line numbers, and each function below names the llama.cpp function it comes from.
//!
//! The prompts themselves, the chat template named `systemone` and the endpoint are deliberately not
//! here. The template is only stored, as an opaque string whose bytes are never printed.

use crate::gguf::{Gguf, Value};
use std::collections::HashMap;
use std::fmt;

/// The nine-point rating scale a Lev `noul` question is read over
/// (`DECISION_LEV_N_RATINGS`, `server-decision.cpp:20`).
pub const LEV_N_RATINGS: usize = 9;

/// The decision type a file declares (`<arch>.decision.type`).
///
/// The sixteen names are llama.cpp's `COMMON_DECISION_TYPE_NAMES` (`common/common.cpp:1187-1204`);
/// any other string is [`DecisionType::Unknown`], which llama.cpp refuses at load
/// (`server-decision.cpp:205-207`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionType {
    OpenJev,
    Lev,
    Kev,
    Nimble,
    Laya,
    Clef,
    PplxDecider,
    VllmDecision2,
    VllmD3,
    Lfm2D1,
    Lfm2D1Omni,
    GraniteSwitch,
    H2oLightning,
    OpenbmbMinicpmOcrJev,
    Lfm3D1,
    Lfm3D1Omni,
    /// A name llama.cpp does not know (`common_decision_type_from_string`'s fallback).
    Unknown(String),
}

impl DecisionType {
    /// The type a GGUF names (`common_decision_type_from_string`).
    pub fn from_name(name: &str) -> DecisionType {
        match name {
            "openjev" => DecisionType::OpenJev,
            "lev" => DecisionType::Lev,
            "kev" => DecisionType::Kev,
            "nimble" => DecisionType::Nimble,
            "laya" => DecisionType::Laya,
            "clef" => DecisionType::Clef,
            "pplx-decider" => DecisionType::PplxDecider,
            "vllm-decision2" => DecisionType::VllmDecision2,
            "vllm-d3" => DecisionType::VllmD3,
            "lfm2-d1" => DecisionType::Lfm2D1,
            "lfm2-d1-omni" => DecisionType::Lfm2D1Omni,
            "granite-switch" => DecisionType::GraniteSwitch,
            "h2o-lightning" => DecisionType::H2oLightning,
            "openbmb-minicpm-ocr-jev" => DecisionType::OpenbmbMinicpmOcrJev,
            "lfm3-d1" => DecisionType::Lfm3D1,
            "lfm3-d1-omni" => DecisionType::Lfm3D1Omni,
            other => DecisionType::Unknown(other.to_string()),
        }
    }

    /// The name the file uses.
    pub fn name(&self) -> &str {
        match self {
            DecisionType::OpenJev => "openjev",
            DecisionType::Lev => "lev",
            DecisionType::Kev => "kev",
            DecisionType::Nimble => "nimble",
            DecisionType::Laya => "laya",
            DecisionType::Clef => "clef",
            DecisionType::PplxDecider => "pplx-decider",
            DecisionType::VllmDecision2 => "vllm-decision2",
            DecisionType::VllmD3 => "vllm-d3",
            DecisionType::Lfm2D1 => "lfm2-d1",
            DecisionType::Lfm2D1Omni => "lfm2-d1-omni",
            DecisionType::GraniteSwitch => "granite-switch",
            DecisionType::H2oLightning => "h2o-lightning",
            DecisionType::OpenbmbMinicpmOcrJev => "openbmb-minicpm-ocr-jev",
            DecisionType::Lfm3D1 => "lfm3-d1",
            DecisionType::Lfm3D1Omni => "lfm3-d1-omni",
            DecisionType::Unknown(s) => s,
        }
    }

    /// Lev: the option-count temperature buckets are `small`/`mid`/`large`, a `noul` question is
    /// read over nine ratings, and a `choice` question is asked in both option orders.
    pub fn is_lev(&self) -> bool {
        matches!(self, DecisionType::Lev)
    }

    /// lfm3-d1 and lfm3-d1-omni: the score is the median level and confidence is `p_max`
    /// (`server-decision.cpp:1625-1628`, `1737-1747`).
    pub fn is_lfm3_d1(&self) -> bool {
        matches!(self, DecisionType::Lfm3D1 | DecisionType::Lfm3D1Omni)
    }

    /// The confidence formula this model's answers use (`get_decision_confidence`,
    /// `server-decision.cpp:1621-1649`).
    pub fn confidence_formula(&self) -> ConfidenceFormula {
        match self {
            DecisionType::VllmDecision2 | DecisionType::VllmD3 => ConfidenceFormula::Entropy,
            t if t.is_lfm3_d1() => ConfidenceFormula::MaxProbability,
            _ => ConfidenceFormula::Default,
        }
    }
}

/// A question's type (`SERVER_DECISION_QUESTION_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

impl QuestionType {
    /// The name the answer's `type` field carries (`decision_question_type_name`).
    pub fn name(self) -> &'static str {
        match self {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        }
    }
}

/// One option of a question: its key and the description the answer's legend echoes.
///
/// llama.cpp keeps a description as JSON (a string, object or array); the answer math only echoes it,
/// so a caller that has already rendered it as text passes that text. The engine's endpoint is not
/// part of this stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestionOption {
    pub key: String,
    pub description: String,
}

/// One question of a request, as far as the answer math needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question {
    pub kind: QuestionType,
    pub options: Vec<QuestionOption>,
}

impl Question {
    /// A question whose options are keys with empty descriptions.
    pub fn new(kind: QuestionType, keys: &[&str]) -> Question {
        Question {
            kind,
            options: keys
                .iter()
                .map(|k| QuestionOption {
                    key: (*k).to_string(),
                    description: String::new(),
                })
                .collect(),
        }
    }
}

/// The number of prompts one question is evaluated with (`n_variants`, `server-decision.cpp:952-958`):
/// a Lev `choice` question is asked in both option orders, everything else once.
pub fn n_variants(decision: &DecisionType, question: &Question) -> usize {
    if decision.is_lev() && question.kind == QuestionType::Choice && question.options.len() > 1 {
        2
    } else {
        1
    }
}

/// The number of raw scores one question is answered from (`n_outputs`,
/// `server-decision.cpp:960-965`): a Lev `noul` question is read over nine ratings.
pub fn n_outputs(decision: &DecisionType, question: &Question) -> usize {
    if decision.is_lev() && question.kind == QuestionType::Noul {
        LEV_N_RATINGS
    } else {
        question.options.len()
    }
}

/// The temperature one question's softmax divides by (`get_temperature`,
/// `server-decision.cpp:1572-1591`): `<type>.<bucket>`, then `<type>`, then `1.0`.
///
/// The bucket is by option count — `2`, `3_5`, `6_10`, `11` — except for a Lev model, whose
/// temperatures were fitted as `small` (≤ 8), `mid` (≤ 26) and `large`.
///
/// A request with media may override the result (`server-decision.cpp:1659-1664`); image input is
/// not supported in this stage, so that override is not here.
pub fn temperature(
    decision: &DecisionType,
    question: &Question,
    temperatures: &HashMap<String, f32>,
) -> f64 {
    let n = n_outputs(decision, question);
    let type_name = question.kind.name();
    let bucket = if decision.is_lev() {
        if n <= 8 {
            "small"
        } else if n <= 26 {
            "mid"
        } else {
            "large"
        }
    } else if n <= 2 {
        "2"
    } else if n <= 5 {
        "3_5"
    } else if n <= 10 {
        "6_10"
    } else {
        "11"
    };
    let by_bucket = format!("{type_name}.{bucket}");
    for name in [by_bucket.as_str(), type_name] {
        if let Some(&t) = temperatures.get(name) {
            return t as f64;
        }
    }
    1.0
}

/// The option probabilities of one question (`format_answer`'s softmax,
/// `server-decision.cpp:1665-1692`).
///
/// `n` is [`n_outputs`], `scores` one vector per variant, and `score_bias` the bias for a `score`
/// question with exactly `n` levels. Each variant's scores are softmaxed in `f64` — the bias added
/// first, the maximum subtracted, divided by `temperature` — and the variants are averaged with the
/// second one reversed (it was asked in the other option order).
pub fn probabilities(
    kind: QuestionType,
    n: usize,
    scores: &[Vec<f32>],
    score_bias: Option<&[f32]>,
    temperature: f64,
) -> Result<Vec<f64>, String> {
    let variants = scores.len();
    if variants == 0 || n == 0 {
        return Err("decision result has no scores".into());
    }
    let mut probs = vec![0.0f64; n];
    for (v, s) in scores.iter().enumerate() {
        if s.len() != n {
            return Err("decision result does not match the number of options".into());
        }
        // A joint head returns NaN if it could not use the decision order, and +inf would make the
        // softmax NaN (`server-decision.cpp:1672-1674`).
        if s.iter().any(|x| x.is_nan() || *x == f32::INFINITY) {
            return Err("the model could not evaluate the decision".into());
        }
        let mut logits: Vec<f64> = s.iter().map(|&x| x as f64).collect();
        if kind == QuestionType::Score {
            if let Some(bias) = score_bias {
                for (i, b) in bias.iter().enumerate().take(n) {
                    logits[i] += *b as f64;
                }
            }
        }
        let logit_max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut p = vec![0.0f64; n];
        let mut sum = 0.0f64;
        for i in 0..n {
            p[i] = ((logits[i] - logit_max) / temperature).exp();
            sum += p[i];
        }
        for i in 0..n {
            // The second variant is in the reverse order.
            let dst = if v == 0 { i } else { n - 1 - i };
            probs[dst] += p[i] / sum / variants as f64;
        }
    }
    Ok(probs)
}

/// The confidence formulas (`server-decision.cpp:1595-1649`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceFormula {
    /// The default: choice's distance from uniform, or score's distance from the mode
    /// (`decision_confidence_choice`, `decision_confidence_score`).
    Default,
    /// vllm-decision2 and vllm-d3: `clamp(1 - H(p)/ln(n), 0, 1)`
    /// (`decision_confidence_entropy`).
    Entropy,
    /// lfm3-d1 and lfm3-d1-omni: the probability of the answer, `p_max`.
    MaxProbability,
}

impl ConfidenceFormula {
    /// The formula [`DecisionType::confidence_formula`] selects for a model type.
    pub fn of(decision: &DecisionType) -> ConfidenceFormula {
        decision.confidence_formula()
    }
}

/// One question's confidence (`get_decision_confidence`, `server-decision.cpp:1621-1649`).
///
/// Only `choice` and `score` answers have one; a `noul` answer returns none, as llama.cpp does.
pub fn confidence(
    kind: QuestionType,
    probs: &[f64],
    formula: ConfidenceFormula,
) -> Result<f64, String> {
    match formula {
        ConfidenceFormula::Entropy => {
            if probs.len() < 2 {
                return Ok(1.0);
            }
            let mut entropy = 0.0;
            for &p in probs {
                if p > 0.0 {
                    entropy -= p * p.ln();
                }
            }
            Ok((1.0 - entropy / (probs.len() as f64).ln()).clamp(0.0, 1.0))
        }
        ConfidenceFormula::MaxProbability => {
            // llama.cpp takes p_max unconditionally; `probs` is never empty (n >= 1).
            Ok(probs.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        }
        ConfidenceFormula::Default => match kind {
            QuestionType::Choice => {
                if probs.len() < 2 {
                    return Ok(1.0);
                }
                let uniform = 1.0 / probs.len() as f64;
                let p_max = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                Ok(((p_max - uniform) / (1.0 - uniform)).max(0.0))
            }
            QuestionType::Score => {
                if probs.len() < 2 {
                    return Ok(1.0);
                }
                let n = probs.len();
                let mode = first_max(probs);
                // Mean distance to the mode, relative to the one of a uniform distribution
                // around its centre.
                let mut dist = 0.0;
                let mut dist_uniform = 0.0;
                for (i, &p) in probs.iter().enumerate() {
                    dist += p * (i as f64 - mode as f64).abs();
                    dist_uniform += (i as f64 - (n - 1) as f64 / 2.0).abs() / n as f64;
                }
                Ok((1.0 - dist / dist_uniform).max(0.0))
            }
            QuestionType::Noul => Err("no confidence for this question type".into()),
        },
    }
}

// `std::max_element`'s first maximum, which is what llama.cpp's answers use for ties.
fn first_max(probs: &[f64]) -> usize {
    let mut best = 0;
    for (i, &p) in probs.iter().enumerate().skip(1) {
        if p > probs[best] {
            best = i;
        }
    }
    best
}

/// One question's answer, with the fields llama.cpp's response has (`format_answer`,
/// `server-decision.cpp:1694-1746`; schema in `docs/reference/systemone.md` §1.3).
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// `{"type":"choice","choice":...,"probabilities":...,"confidence":...}`
    Choice {
        choice: String,
        /// `(key, probability)` in option order.
        probabilities: Vec<(String, f64)>,
        confidence: f64,
    },
    /// `{"type":"score","score":...,"legend":...,"probabilities":...,"confidence":...}`
    Score {
        score: f64,
        /// `(key, description)` in level order.
        legend: Vec<(String, String)>,
        probabilities: Vec<(String, f64)>,
        confidence: f64,
    },
    /// `{"type":"noul":...}` — no confidence.
    Noul { noul: f64 },
}

impl Answer {
    /// The answer's `type` field.
    pub fn type_name(&self) -> &'static str {
        match self {
            Answer::Choice { .. } => "choice",
            Answer::Score { .. } => "score",
            Answer::Noul { .. } => "noul",
        }
    }
}

/// One question's answer from its raw scores (`format_answer`, `server-decision.cpp:1651-1747`).
pub fn format_answer(
    decision: &DecisionType,
    question: &Question,
    scores: &[Vec<f32>],
    temperatures: &HashMap<String, f32>,
    score_bias: &HashMap<usize, Vec<f32>>,
) -> Result<Answer, String> {
    let n = n_outputs(decision, question);
    if scores.len() != n_variants(decision, question) {
        return Err("decision result does not match the number of variants".into());
    }
    let temp = temperature(decision, question, temperatures);
    let bias = if question.kind == QuestionType::Score {
        score_bias.get(&n).map(Vec::as_slice)
    } else {
        None
    };
    let probs = probabilities(question.kind, n, scores, bias, temp)?;

    if question.kind == QuestionType::Noul {
        if decision.is_lev() {
            // noul = sum_i p_i * i / (n - 1) over the nine-point rating scale
            // (`server-decision.cpp:1696-1704`).
            let mut expected = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                expected += p * i as f64 / (n - 1) as f64;
            }
            return Ok(Answer::Noul { noul: expected });
        }
        // noul = p("true") over the two options (`1705-1710`).
        for (i, option) in question.options.iter().enumerate() {
            if option.key == "true" {
                return Ok(Answer::Noul { noul: probs[i] });
            }
        }
        return Err("the noul question has no \"true\" option".into());
    }

    let probabilities: Vec<(String, f64)> = question
        .options
        .iter()
        .take(n)
        .zip(probs.iter().copied())
        .map(|(o, p)| (o.key.clone(), p))
        .collect();

    let formula = decision.confidence_formula();
    if question.kind == QuestionType::Choice {
        let best = first_max(&probs);
        Ok(Answer::Choice {
            choice: question.options[best].key.clone(),
            probabilities,
            confidence: confidence(QuestionType::Choice, &probs, formula)?,
        })
    } else {
        // score = sum_i i * p_i (`server-decision.cpp:1728-1745`).
        let mut expected = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            expected += i as f64 * p;
        }
        let legend: Vec<(String, String)> = question
            .options
            .iter()
            .take(n)
            .map(|o| (o.key.clone(), o.description.clone()))
            .collect();
        Ok(Answer::Score {
            score: expected,
            legend,
            probabilities,
            confidence: confidence(QuestionType::Score, &probs, formula)?,
        })
    }
}

/// The decision metadata a GGUF carries (`server_decision_context::init`,
/// `server-decision.cpp:41-230`), as far as the answer needs it.
///
/// The `systemone` chat template is stored as an opaque string: its bytes are never printed, and
/// `Debug` reports its length only.
pub struct DecisionInfo {
    kind: DecisionType,
    temperatures: HashMap<String, f32>,
    score_bias: HashMap<usize, Vec<f32>>,
    labels: Vec<String>,
    template: Option<String>,
}

/// The key llama.cpp looks a named chat template up under (`LLM_KV_TOKENIZER_CHAT_TEMPLATE` plus
/// the name, `llama-model.cpp:3881-3886`).
const SYSTEMONE_TEMPLATE: &str = "tokenizer.chat_template.systemone";

impl DecisionInfo {
    /// Read the decision metadata from a GGUF header, or `None` when the file declares no decision
    /// type (`common_get_decision_type`, `common/common.cpp:1216-1226`).
    ///
    /// The validations and messages are llama.cpp's: an unknown type, a non-positive temperature,
    /// a malformed score bias and a decision model without the `systemone` template are refused.
    pub fn from_gguf(g: &Gguf) -> Result<Option<DecisionInfo>, String> {
        let arch = match g.meta.get("general.architecture") {
            Some(Value::Str(s)) => s.clone(),
            _ => return Ok(None),
        };
        let type_key = format!("{arch}.decision.type");
        let type_name = match g.meta.get(&type_key) {
            None => return Ok(None),
            Some(Value::Str(s)) => s.clone(),
            Some(_) => return Err(format!("gguf: metadata {type_key:?} is not a string")),
        };
        let kind = DecisionType::from_name(&type_name);
        if let DecisionType::Unknown(name) = &kind {
            return Err(format!("unsupported decision model type: {name}"));
        }

        let template = match g.meta.get(SYSTEMONE_TEMPLATE) {
            Some(Value::Str(s)) => Some(s.clone()),
            Some(_) => {
                return Err(format!(
                    "gguf: metadata {SYSTEMONE_TEMPLATE:?} is not a string"
                ))
            }
            None => {
                return Err("decision model has no \"systemone\" template".into());
            }
        };

        let prefix_temp = format!("{arch}.decision.temperature.");
        let mut temperatures = HashMap::new();
        for (key, value) in &g.meta {
            let Some(name) = key.strip_prefix(&prefix_temp) else {
                continue;
            };
            let Some(t) = number(value) else {
                return Err(format!("invalid decision temperature: {key}"));
            };
            if t <= 0.0 {
                return Err(format!("invalid decision temperature: {key} = {t}"));
            }
            temperatures.insert(name.to_string(), t);
        }

        // "<n_levels>.<level>" -> value
        let prefix_bias = format!("{arch}.decision.score_bias.");
        let mut score_bias: HashMap<usize, Vec<f32>> = HashMap::new();
        for (key, value) in &g.meta {
            let Some(suffix) = key.strip_prefix(&prefix_bias) else {
                continue;
            };
            let parts: Vec<&str> = suffix.split('.').collect();
            let (n, level) = if parts.len() == 2 {
                (
                    parts[0].parse::<usize>().unwrap_or(0),
                    parts[1].parse::<usize>().unwrap_or(0),
                )
            } else {
                (0, 0)
            };
            if !(2..=255).contains(&n) || level >= n {
                return Err(format!("invalid decision score bias: {key}"));
            }
            let Some(b) = number(value) else {
                return Err(format!("invalid decision score bias: {key}"));
            };
            let bias = score_bias.entry(n).or_insert_with(|| vec![0.0; n]);
            bias[level] = b;
        }

        let labels = match g.meta.get(&format!("{arch}.decision.labels")) {
            Some(Value::ArrStr(v)) => v.clone(),
            Some(_) => {
                let key = format!("{arch}.decision.labels");
                return Err(format!("gguf: metadata {key:?} is not a string array"));
            }
            None => Vec::new(),
        };

        Ok(Some(DecisionInfo {
            kind,
            temperatures,
            score_bias,
            labels,
            template,
        }))
    }

    /// The model's decision type.
    pub fn kind(&self) -> &DecisionType {
        &self.kind
    }

    /// `<arch>.decision.temperature.*` by name (`choice`, `score`, `noul`, `<type>.<bucket>`, ...).
    pub fn temperatures(&self) -> &HashMap<String, f32> {
        &self.temperatures
    }

    /// `<arch>.decision.score_bias.<n_levels>.<level>` by number of levels.
    pub fn score_bias(&self) -> &HashMap<usize, Vec<f32>> {
        &self.score_bias
    }

    /// `<arch>.decision.labels`, when the file has them.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The length of the `systemone` template in bytes. The text itself stays inside: it is not
    /// rendered, printed or logged here.
    pub fn template_len(&self) -> usize {
        self.template.as_ref().map_or(0, String::len)
    }
}

impl fmt::Debug for DecisionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionInfo")
            .field("kind", &self.kind)
            .field("temperatures", &self.temperatures)
            .field("score_bias", &self.score_bias)
            .field("labels", &self.labels)
            .field("template_bytes", &self.template_len())
            .finish()
    }
}

// A metadata value as the number llama.cpp's `strtof`/`strtoul` would read out of its text.
fn number(v: &Value) -> Option<f32> {
    match v {
        Value::F32(x) => Some(*x),
        Value::U32(x) => Some(*x as f32),
        Value::I32(x) => Some(*x as f32),
        Value::U64(x) => Some(*x as f32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temps(pairs: &[(&str, f32)]) -> HashMap<String, f32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn question(kind: QuestionType, n: usize) -> Question {
        let keys: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        Question::new(kind, &refs)
    }

    #[test]
    fn uniform_scores_have_no_confidence_and_probabilities_sum_to_one() {
        for n in [2usize, 3, 5, 6, 10, 11] {
            let q = question(QuestionType::Choice, n);
            let scores = vec![vec![0.5f32; n]];
            let ans = format_answer(
                &DecisionType::OpenJev,
                &q,
                &scores,
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
            let Answer::Choice {
                probabilities,
                confidence,
                ..
            } = ans
            else {
                panic!("choice");
            };
            let sum: f64 = probabilities.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-12, "{n}: {sum}");
            assert!(confidence.abs() < 1e-12, "{n}: {confidence}");
        }
        // Score confidence is 0 for a uniform distribution too: the mode is the first level but
        // the distance is exactly the uniform one.
        let q = question(QuestionType::Score, 4);
        let ans = format_answer(
            &DecisionType::OpenJev,
            &q,
            &[vec![0.0; 4]],
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        let Answer::Score {
            score, confidence, ..
        } = ans
        else {
            panic!("score");
        };
        assert!((score - 1.5).abs() < 1e-12);
        assert!(confidence.abs() < 1e-12);
    }

    #[test]
    fn a_dominant_option_has_confidence_near_one() {
        let q = question(QuestionType::Choice, 3);
        let ans = format_answer(
            &DecisionType::OpenJev,
            &q,
            &[vec![100.0, 0.0, 0.0]],
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        let Answer::Choice {
            choice,
            probabilities,
            confidence,
        } = ans
        else {
            panic!("choice");
        };
        assert_eq!(choice, "0");
        assert!((probabilities[0].1 - 1.0).abs() < 1e-9);
        assert!(confidence > 0.999_999);
    }

    #[test]
    fn temperature_is_looked_up_by_bucket_then_type_then_one() {
        // Every bucket key is present, so the answer names the bucket the edge belongs to.
        let all = temps(&[
            ("choice.2", 0.2),
            ("choice.3_5", 0.35),
            ("choice.6_10", 0.6),
            ("choice.11", 1.1),
            ("choice", 9.0),
        ]);
        for (n, want) in [
            (1usize, 0.2f32),
            (2, 0.2),
            (3, 0.35),
            (5, 0.35),
            (6, 0.6),
            (10, 0.6),
            (11, 1.1),
        ] {
            let q = question(QuestionType::Choice, n);
            assert_eq!(
                temperature(&DecisionType::OpenJev, &q, &all),
                f64::from(want),
                "{n} options"
            );
        }
        // Without a bucket key the type's own key answers, and without that 1.0.
        let q = question(QuestionType::Choice, 3);
        assert_eq!(
            temperature(&DecisionType::OpenJev, &q, &temps(&[("choice", 0.25)])),
            0.25
        );
        // Lev uses the fitted small/mid/large buckets, at the edges 8/9 and 26/27.
        for (n, want) in [(8usize, 0.1f32), (9, 0.2), (26, 0.2), (27, 0.3)] {
            let q = question(QuestionType::Score, n);
            assert_eq!(
                temperature(
                    &DecisionType::Lev,
                    &q,
                    &temps(&[
                        ("score.small", 0.1),
                        ("score.mid", 0.2),
                        ("score.large", 0.3),
                    ]),
                ),
                f64::from(want),
                "{n} options"
            );
        }
        // Nothing configured: 1.0.
        let q = question(QuestionType::Choice, 3);
        assert_eq!(
            temperature(&DecisionType::OpenJev, &q, &HashMap::new()),
            1.0
        );
    }

    #[test]
    fn the_temperature_changes_the_probabilities_and_the_bias_only_score_questions() {
        let q = question(QuestionType::Choice, 2);
        let scores = vec![vec![0.0f32, 1.0]];
        let cold = format_answer(
            &DecisionType::OpenJev,
            &q,
            &scores,
            &temps(&[("choice", 0.5)]),
            &HashMap::new(),
        )
        .unwrap();
        let hot = format_answer(
            &DecisionType::OpenJev,
            &q,
            &scores,
            &temps(&[("choice", 4.0)]),
            &HashMap::new(),
        )
        .unwrap();
        let p = |a: &Answer| match a {
            Answer::Choice { probabilities, .. } => probabilities[1].1,
            _ => panic!("choice"),
        };
        assert!(p(&cold) > p(&hot));

        // The bias is added before the temperature, and only to a score question with that many
        // levels.
        let q = question(QuestionType::Score, 2);
        let biased = format_answer(
            &DecisionType::OpenJev,
            &q,
            &scores,
            &HashMap::new(),
            &[(2usize, vec![5.0, 0.0])].into_iter().collect(),
        )
        .unwrap();
        let Answer::Score { score, .. } = biased else {
            panic!("score")
        };
        assert!(score < 0.5, "the bias favours level 0: {score}");
        let choice = question(QuestionType::Choice, 2);
        let ignored = format_answer(
            &DecisionType::OpenJev,
            &choice,
            &scores,
            &HashMap::new(),
            &[(2usize, vec![5.0, 0.0])].into_iter().collect(),
        )
        .unwrap();
        let Answer::Choice { choice, .. } = ignored else {
            panic!("choice")
        };
        assert_eq!(choice, "1", "a bias does not touch a choice question");
    }

    #[test]
    fn a_second_variant_is_averaged_in_reverse() {
        let q = question(QuestionType::Choice, 2);
        let forward = vec![vec![2.0f32, 0.0]];
        // The same preference in both orders: the second is reversed when averaged.
        let both = vec![vec![2.0f32, 0.0], vec![2.0, 0.0]];
        let a = format_answer(
            &DecisionType::OpenJev,
            &q,
            &forward,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        let b = format_answer(
            &DecisionType::Lev,
            &q,
            &both,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        // Asking both orders cancels the order preference: the answer is uniform.
        let Answer::Choice {
            probabilities,
            confidence,
            ..
        } = b
        else {
            panic!("choice");
        };
        assert!((probabilities[0].1 - 0.5).abs() < 1e-12);
        assert!(confidence.abs() < 1e-12);
        let Answer::Choice {
            probabilities: one, ..
        } = a
        else {
            panic!("choice");
        };
        assert!(one[0].1 > 0.8);
    }

    #[test]
    fn lev_noul_reads_nine_ratings_and_a_plain_noul_reads_true() {
        let q = question(QuestionType::Noul, 2);
        let scores = vec![vec![0.0f32; LEV_N_RATINGS]];
        let ans = format_answer(
            &DecisionType::Lev,
            &q,
            &scores,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        let Answer::Noul { noul } = ans else {
            panic!("noul")
        };
        // Uniform over the nine ratings: expected rating 4, normalized to 4/8 = 0.5.
        assert!((noul - 0.5).abs() < 1e-12);

        let q = Question::new(QuestionType::Noul, &["false", "true"]);
        let ans = format_answer(
            &DecisionType::OpenJev,
            &q,
            &[vec![0.0f32, 10.0]],
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        let Answer::Noul { noul } = ans else {
            panic!("noul")
        };
        assert!(noul > 0.999);
    }

    #[test]
    fn confidence_formulas_follow_the_model_type() {
        let probs = [0.7, 0.2, 0.1];
        assert_eq!(
            ConfidenceFormula::of(&DecisionType::VllmD3),
            ConfidenceFormula::Entropy
        );
        assert_eq!(
            ConfidenceFormula::of(&DecisionType::Lfm3D1),
            ConfidenceFormula::MaxProbability
        );
        let entropy = confidence(QuestionType::Choice, &probs, ConfidenceFormula::Entropy).unwrap();
        assert!((0.0..=1.0).contains(&entropy));
        let max = confidence(
            QuestionType::Score,
            &probs,
            ConfidenceFormula::MaxProbability,
        )
        .unwrap();
        assert_eq!(max, 0.7);
        assert!(confidence(QuestionType::Noul, &probs, ConfidenceFormula::Default).is_err());
    }

    #[test]
    fn scores_of_the_wrong_shape_are_refused() {
        let q = question(QuestionType::Choice, 2);
        assert!(format_answer(
            &DecisionType::OpenJev,
            &q,
            &[vec![0.0f32, 0.0], vec![0.0, 0.0]],
            &HashMap::new(),
            &HashMap::new(),
        )
        .is_err());
        assert!(format_answer(
            &DecisionType::OpenJev,
            &q,
            &[vec![0.0f32]],
            &HashMap::new(),
            &HashMap::new(),
        )
        .is_err());
        assert!(format_answer(
            &DecisionType::OpenJev,
            &q,
            &[vec![f32::NAN, 0.0]],
            &HashMap::new(),
            &HashMap::new(),
        )
        .is_err());
        assert!(format_answer(
            &DecisionType::OpenJev,
            &q,
            &[vec![f32::INFINITY, 0.0]],
            &HashMap::new(),
            &HashMap::new(),
        )
        .is_err());
    }

    #[test]
    fn the_unknown_type_keeps_its_name() {
        let t = DecisionType::from_name("a-model-from-the-future");
        assert_eq!(t.name(), "a-model-from-the-future");
        assert_eq!(t, DecisionType::Unknown("a-model-from-the-future".into()));
        for name in [
            "openjev",
            "lev",
            "kev",
            "nimble",
            "laya",
            "clef",
            "pplx-decider",
            "vllm-decision2",
            "vllm-d3",
            "lfm2-d1",
            "lfm2-d1-omni",
            "granite-switch",
            "h2o-lightning",
            "openbmb-minicpm-ocr-jev",
            "lfm3-d1",
            "lfm3-d1-omni",
        ] {
            assert_eq!(DecisionType::from_name(name).name(), name);
        }
    }
}
