//! The answer math against an independent implementation (`roadmap S4.3`).
//!
//! `scripts/systemone-math.py` recomputes llama.cpp's `format_answer`, `get_temperature` and the
//! confidence helpers from `docs/reference/systemone.md` §2 in Python, writes ~40 random cases to
//! `tests/fixtures/systemone-math.jsonl`, and records the answers it gets. This test reads those
//! cases back and requires every number the Rust transcription produces to agree to 1e-12.
//!
//! `peregrine-core` has no JSON dependency (ADR 0002), so the little reader below parses exactly
//! the value types the fixture uses.

use peregrine_core::decision::{
    format_answer, Answer, DecisionType, Question, QuestionOption, QuestionType,
};
use peregrine_core::json::Value as JsonValue;
use std::collections::HashMap;

fn fixture() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/systemone-math.jsonl");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(items) => items
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("no key {key:?}")),
            _ => panic!("not an object"),
        }
    }
    fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            other => panic!("not a string: {other:?}"),
        }
    }
    fn num(&self) -> f64 {
        match self {
            Json::Num(x) => *x,
            other => panic!("not a number: {other:?}"),
        }
    }
    fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            other => panic!("not an array: {other:?}"),
        }
    }
    fn f32(&self) -> f32 {
        self.num() as f32
    }
    fn pairs(&self) -> Vec<(String, f64)> {
        self.arr()
            .iter()
            .map(|p| {
                let p = p.arr();
                (p[0].str().to_string(), p[1].num())
            })
            .collect()
    }
    fn text_pairs(&self) -> Vec<(String, String)> {
        self.arr()
            .iter()
            .map(|p| {
                let p = p.arr();
                (p[0].str().to_string(), p[1].str().to_string())
            })
            .collect()
    }
}

struct Parser<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Parser<'a> {
    fn parse(text: &'a str) -> Json {
        let mut p = Parser {
            b: text.as_bytes(),
            p: 0,
        };
        let v = p.value();
        p.ws();
        assert_eq!(p.p, p.b.len(), "trailing bytes in a fixture line");
        v
    }

    fn ws(&mut self) {
        while self.p < self.b.len() && (self.b[self.p] as char).is_ascii_whitespace() {
            self.p += 1;
        }
    }

    fn eat(&mut self, c: u8) {
        assert_eq!(self.b[self.p], c, "expected {:?}", c as char);
        self.p += 1;
    }

    fn value(&mut self) -> Json {
        self.ws();
        match self.b[self.p] {
            b'{' => {
                self.p += 1;
                let mut items = Vec::new();
                self.ws();
                if self.b[self.p] == b'}' {
                    self.p += 1;
                    return Json::Obj(items);
                }
                loop {
                    self.ws();
                    let k = self.string();
                    self.ws();
                    self.eat(b':');
                    let v = self.value();
                    items.push((k, v));
                    self.ws();
                    match self.b[self.p] {
                        b',' => self.p += 1,
                        b'}' => {
                            self.p += 1;
                            break;
                        }
                        other => panic!("bad object byte {:?}", other as char),
                    }
                }
                Json::Obj(items)
            }
            b'[' => {
                self.p += 1;
                let mut items = Vec::new();
                self.ws();
                if self.b[self.p] == b']' {
                    self.p += 1;
                    return Json::Arr(items);
                }
                loop {
                    items.push(self.value());
                    self.ws();
                    match self.b[self.p] {
                        b',' => self.p += 1,
                        b']' => {
                            self.p += 1;
                            break;
                        }
                        other => panic!("bad array byte {:?}", other as char),
                    }
                }
                Json::Arr(items)
            }
            b'"' => Json::Str(self.string()),
            b't' => {
                self.literal("true");
                Json::Bool(true)
            }
            b'f' => {
                self.literal("false");
                Json::Bool(false)
            }
            b'n' => {
                self.literal("null");
                Json::Null
            }
            _ => self.number(),
        }
    }

    fn literal(&mut self, word: &str) {
        assert!(self.b[self.p..].starts_with(word.as_bytes()));
        self.p += word.len();
    }

    fn string(&mut self) -> String {
        self.eat(b'"');
        let mut out = String::new();
        loop {
            match self.b[self.p] {
                b'"' => {
                    self.p += 1;
                    return out;
                }
                b'\\' => {
                    self.p += 1;
                    let c = self.b[self.p];
                    self.p += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = std::str::from_utf8(&self.b[self.p..self.p + 4]).unwrap();
                            let n = u32::from_str_radix(hex, 16).unwrap();
                            self.p += 4;
                            out.push(char::from_u32(n).unwrap());
                        }
                        other => panic!("bad escape {:?}", other as char),
                    }
                }
                c => {
                    // The fixture is UTF-8; a multi-byte character is copied through.
                    let start = self.p;
                    let len = utf8_len(c);
                    out.push_str(std::str::from_utf8(&self.b[start..start + len]).unwrap());
                    self.p += len;
                }
            }
        }
    }

    fn number(&mut self) -> Json {
        let start = self.p;
        if self.b[self.p] == b'-' {
            self.p += 1;
        }
        while self.p < self.b.len()
            && matches!(
                self.b[self.p],
                b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-'
            )
        {
            self.p += 1;
        }
        let text = std::str::from_utf8(&self.b[start..self.p]).unwrap();
        Json::Num(
            text.parse()
                .unwrap_or_else(|e| panic!("number {text:?}: {e}")),
        )
    }
}

fn utf8_len(first: u8) -> usize {
    match first {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

fn question(v: &Json) -> Question {
    let kind = match v.get("type").str() {
        "choice" => QuestionType::Choice,
        "score" => QuestionType::Score,
        "noul" => QuestionType::Noul,
        other => panic!("question type {other:?}"),
    };
    let keys = v.get("keys").arr();
    let descriptions = v.get("descriptions").arr();
    Question {
        kind,
        instructions: JsonValue::Null,
        options: keys
            .iter()
            .zip(descriptions)
            .map(|(k, d)| QuestionOption {
                key: k.str().to_string(),
                description: JsonValue::Str(d.str().to_string()),
            })
            .collect(),
    }
}

fn temperatures(v: &Json) -> HashMap<String, f32> {
    v.arr_or_obj()
        .into_iter()
        .map(|(k, x)| (k, x.f32()))
        .collect()
}

fn score_bias(v: &Json) -> HashMap<usize, Vec<f32>> {
    v.arr_or_obj()
        .into_iter()
        .map(|(k, x)| {
            (
                k.parse::<usize>().unwrap(),
                x.arr().iter().map(Json::f32).collect(),
            )
        })
        .collect()
}

trait Entries {
    fn arr_or_obj(&self) -> Vec<(String, &Json)>;
}
impl Entries for Json {
    fn arr_or_obj(&self) -> Vec<(String, &Json)> {
        match self {
            Json::Obj(items) => items.iter().map(|(k, v)| (k.clone(), v)).collect(),
            other => panic!("not an object: {other:?}"),
        }
    }
}

fn close(got: f64, want: f64, what: &str, line: usize) {
    assert!(
        (got - want).abs() <= 1e-12,
        "line {line}: {what}: got {got:?}, want {want:?} (difference {})",
        (got - want).abs()
    );
}

#[test]
fn the_answer_math_agrees_with_the_python_reference_to_1e_12() {
    let text = fixture();
    let mut lines = 0;
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        lines += 1;
        let case = Parser::parse(line);
        let decision = DecisionType::from_name(case.get("decision").str());
        let q = question(case.get("question"));
        let temps = temperatures(case.get("temperatures"));
        let bias = score_bias(case.get("score_bias"));
        let scores: Vec<Vec<f32>> = case
            .get("scores")
            .arr()
            .iter()
            .map(|row| row.arr().iter().map(Json::f32).collect())
            .collect();
        let got = format_answer(&decision, &q, &scores, &temps, &bias)
            .unwrap_or_else(|e| panic!("line {}: {e}", i + 1));
        let want = case.get("answer");
        assert_eq!(got.type_name(), want.get("type").str(), "line {}", i + 1);
        match &got {
            Answer::Choice {
                choice,
                probabilities,
                confidence,
            } => {
                assert_eq!(choice, want.get("choice").str(), "line {}", i + 1);
                compare_probs(probabilities, &want.get("probabilities").pairs(), i + 1);
                close(
                    *confidence,
                    want.get("confidence").num(),
                    "confidence",
                    i + 1,
                );
            }
            Answer::Score {
                score,
                legend,
                probabilities,
                confidence,
            } => {
                close(*score, want.get("score").num(), "score", i + 1);
                let want_legend = want.get("legend").text_pairs();
                assert_eq!(legend.len(), want_legend.len(), "line {}", i + 1);
                for (got, (key, description)) in legend.iter().zip(&want_legend) {
                    assert_eq!(&got.0, key, "line {}", i + 1);
                    assert_eq!(got.1, JsonValue::Str(description.clone()), "line {}", i + 1);
                }
                compare_probs(probabilities, &want.get("probabilities").pairs(), i + 1);
                close(
                    *confidence,
                    want.get("confidence").num(),
                    "confidence",
                    i + 1,
                );
            }
            Answer::Noul { noul } => close(*noul, want.get("noul").num(), "noul", i + 1),
        }
    }
    assert!(lines >= 40, "the fixture holds {lines} cases");
}

fn compare_probs(got: &[(String, f64)], want: &[(String, f64)], line: usize) {
    assert_eq!(got.len(), want.len(), "line {line}");
    for (i, ((gk, gp), (wk, wp))) in got.iter().zip(want).enumerate() {
        assert_eq!(gk, wk, "line {line}, probability {i}");
        close(*gp, *wp, "probability", line);
    }
    let sum: f64 = got.iter().map(|(_, p)| p).sum();
    close(sum, 1.0, "the probabilities sum", line);
}
