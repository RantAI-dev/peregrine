//! Constrained decoding: the matcher's corpus and schemas, the random-model property, and
//! the mask's agreement with brute force.
//!
//! The validator in `support` is independent of the matcher: the property test would fail on
//! any disagreement.

mod support;

use peregrine_core::constrain::{Constraint, Matcher, Part, Schema, Vocabulary};
use peregrine_core::json::Value as CoreValue;
use peregrine_core::sample::{sample_constrained, Penalties, Scratch};
use support::{parse as parse_json, validate};

fn core_schema(text: &str) -> Schema {
    Schema::new(CoreValue::parse(text).expect("a schema document"))
}

fn constraint(text: &str) -> Constraint {
    Constraint::from_schema(core_schema(text))
}

fn new_matcher(c: &Constraint) -> Matcher {
    Matcher::new(c).expect("the schema compiles")
}

/// Feed every byte; true when the text is an accepted, complete document.
fn complete(c: &Constraint, text: &str) -> bool {
    let mut m = new_matcher(c);
    if m.advance_bytes(text.as_bytes()).is_err() {
        return false;
    }
    m.can_end()
}

// --------------------------------------------------------------------------------------
// The JSON grammar corpus: valid documents, and invalid ones with the byte that kills them.
// --------------------------------------------------------------------------------------

#[test]
fn the_json_corpus_is_accepted_and_the_invalid_corpus_is_rejected_at_the_right_byte() {
    let any = constraint("{}");
    let valid = [
        "null",
        "true",
        "false",
        "0",
        "-0",
        "1",
        "-1",
        "1e2",
        "1E+2",
        "1e-10",
        "0.0",
        "-0.5",
        "123456789012345678901234567890",
        "\"\"",
        "\"a\"",
        "\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"",
        "\"\\u0000\\u001f\\u007f\"",
        "\"\\u00e9\"",
        "\"\\ud83d\\ude00\"",
        "\"é😀\"",
        "[]",
        "[1]",
        "[1,2,3]",
        "[[[[[1]]]]]",
        "{}",
        "{\"a\":1}",
        "{\"a\":{\"b\":[true,null,\"x\"]}}",
        "{\"a\":1,\"a\":2}",
        "[ 1 ,\n 2 ]",
        "{\n\"a\" : 1\n}",
        "[\n\n \t\t1]",
    ];
    for text in valid {
        assert!(complete(&any, text), "the corpus should accept {text:?}");
        assert!(
            parse_json(text).is_ok(),
            "the independent parser should accept {text:?}"
        );
    }
    let invalid: &[(&str, usize)] = &[
        ("{", 1),
        ("[", 1),
        ("[1,]", 3),
        ("{\"a\":1,}", 7),
        ("{\"a\" 1}", 5),
        ("{'a':1}", 1),
        ("01", 1),
        ("-", 1),
        ("1.", 2),
        (".5", 0),
        ("+1", 0),
        ("1e", 2),
        ("1e+", 3),
        ("NaN", 0),
        ("nul", 3),
        ("tru", 3),
        ("\"abc", 4),
        ("\"a\nb\"", 2),
        ("\"\\x\"", 2),
        ("\"\\ud83d\"", 7),
        ("\"\\udc00\"", 4),
        ("[1 2]", 3),
        ("{\"a\":1 \"b\":2}", 7),
        ("[]x", 2),
    ];
    for &(text, at) in invalid {
        let bytes = text.as_bytes();
        let mut m = new_matcher(&any);
        let mut rejected = None;
        for (i, &b) in bytes.iter().enumerate() {
            if !m.advance(b) {
                rejected = Some(i);
                break;
            }
        }
        let rejected = rejected.or(if m.can_end() { None } else { Some(bytes.len()) });
        assert_eq!(
            rejected,
            Some(at),
            "invalid {text:?} should be rejected at byte {at}"
        );
        assert!(
            parse_json(text).is_err(),
            "the independent parser should reject {text:?}"
        );
    }
    // Raw invalid UTF-8 bytes are rejected at the byte that makes the sequence invalid.
    let mut m = new_matcher(&any);
    assert!(m.advance(b'"'));
    assert!(m.advance(0xED));
    assert!(!m.advance(0xA0));
    let mut m = new_matcher(&any);
    assert!(m.advance(b'"'));
    assert!(!m.advance(0xC0));
}

// --------------------------------------------------------------------------------------
// Forty-plus schemas: every prefix of a valid instance, the whole instance, and invalid
// instances that the independent validator also refuses.
// --------------------------------------------------------------------------------------

type Case = (
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
);

const CASES: &[Case] = &[
    (
        r#"{"type":"string"}"#,
        &[r#""""#, r#""abc""#, r#""é😀""#, r#""\u00e9""#],
        &["\"a\nb\"", "1", "null"],
    ),
    (
        r#"{"type":"string","minLength":2,"maxLength":4}"#,
        &[r#""ab""#, r#""abcd""#, r#""éé""#, r#""😀😀""#],
        &[r#""a""#, r#""abcde""#],
    ),
    (
        r#"{"type":"string","maxLength":0}"#,
        &[r#""""#],
        &[r#""a""#],
    ),
    (
        r#"{"type":"integer"}"#,
        &["0", "-0", "42", "-7", "123456789012345678901234567890"],
        &["1.5", "1e2", "01", "-"],
    ),
    (
        r#"{"type":"number"}"#,
        &[
            "0",
            "-0",
            "1e-10",
            "0.0",
            "1.5e+3",
            "123456789012345678901234567890",
        ],
        &[".5", "1.", "01"],
    ),
    (
        r#"{"type":"integer","minimum":0,"maximum":5}"#,
        &["0", "1", "5"],
        &["-1", "6", "1.0"],
    ),
    (
        r#"{"type":"integer","minimum":-2,"maximum":-1}"#,
        &["-2", "-1"],
        &["0", "-3"],
    ),
    (
        r#"{"type":"integer","minimum":0}"#,
        &["0", "1", "123"],
        &["-1", "-0"],
    ),
    (
        r#"{"type":"integer","maximum":0}"#,
        &["0", "-0", "-1", "-123"],
        &["1"],
    ),
    (
        r#"{"type":"integer","exclusiveMinimum":0,"exclusiveMaximum":4}"#,
        &["1", "2", "3"],
        &["0", "4", "-1"],
    ),
    (
        r#"{"type":"integer","enum":[1,2,3]}"#,
        &["1", "2", "3"],
        &["0", "4", "1.0"],
    ),
    (r#"{"const":7}"#, &["7"], &["8", "7.0"]),
    (r#"{"const":"a\nb"}"#, &["\"a\\nb\""], &["\"a b\""]),
    (
        r#"{"enum":["a",null,true,2]}"#,
        &["\"a\"", "null", "true", "2"],
        &["\"b\"", "false", "3"],
    ),
    (r#"{"type":"boolean"}"#, &["true", "false"], &["1", "null"]),
    (r#"{"type":"null"}"#, &["null"], &["true", "0"]),
    (
        r#"{"type":["string","null"]}"#,
        &[r#""x""#, "null"],
        &["1", "true"],
    ),
    (
        r#"{"type":["integer","boolean"]}"#,
        &["0", "12", "true"],
        &[r#""x""#, "1.5"],
    ),
    (
        r#"{"type":"array"}"#,
        &["[]", "[1,\"a\",null]", "[[1]]"],
        &["{}", "1"],
    ),
    (
        r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3}"#,
        &["[1]", "[1,2,3]"],
        &["[]", "[1,2,3,4]", "[1,\"a\"]"],
    ),
    (
        r#"{"type":"array","prefixItems":[{"type":"string"},{"type":"integer"}],"items":{"type":"boolean"},"maxItems":4}"#,
        &[r#"["a",1]"#, r#"["a",1,true]"#, r#"["a",1,false,true]"#],
        &[r#"[1,"a"]"#, r#"["a",1,2]"#, r#"["a",1,true,true,true]"#],
    ),
    (r#"{"type":"array","maxItems":0}"#, &["[]"], &["[1]"]),
    (
        r#"{"type":"array","minItems":2,"maxItems":2,"items":{"type":"string"}}"#,
        &[r#"["a","b"]"#],
        &[r#"["a"]"#, r#"["a","b","c"]"#],
    ),
    (
        r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}"#,
        &[r#"{"a":1}"#],
        &["{}", r#"{"a":1,"b":2}"#, r#"{"b":2}"#],
    ),
    (
        r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"additionalProperties":false}"#,
        &["{}", r#"{"a":1}"#, r#"{"b":"x"}"#, r#"{"a":1,"b":"x"}"#],
        &[r#"{"b":"x","a":1}"#, r#"{"c":1}"#],
    ),
    (
        r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},"required":["c"],"additionalProperties":false}"#,
        &[r#"{"c":3}"#, r#"{"a":1,"c":3}"#, r#"{"a":1,"b":2,"c":3}"#],
        &[r#"{"a":1}"#, r#"{"c":3,"a":1}"#],
    ),
    (
        r#"{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":{"type":"string"}}"#,
        &[r#"{"a":1}"#, r#"{"a":1,"b":"x","c":""}"#],
        &[r#"{"a":1,"b":2}"#],
    ),
    (
        r#"{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":true}"#,
        &[r#"{"a":1}"#, r#"{"a":1,"z":true}"#],
        &[r#"{"a":"x"}"#],
    ),
    (
        r#"{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"integer"}},"required":["b"],"additionalProperties":false}},"required":["a"],"additionalProperties":false}"#,
        &[r#"{"a":{"b":1}}"#],
        &[r#"{"a":{}}"#, r#"{"a":{"b":1},"x":2}"#],
    ),
    (
        r#"{"type":"object","properties":{},"additionalProperties":false}"#,
        &["{}"],
        &[r#"{"a":1}"#],
    ),
    (
        r##"{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#"}}},"required":["v"],"additionalProperties":false}"##,
        &[
            r#"{"v":1}"#,
            r#"{"v":1,"kids":[]}"#,
            r#"{"v":1,"kids":[{"v":2},{"v":3,"kids":[{"v":4}]}]}"#,
        ],
        &[
            r#"{"v":1,"kids":[{}]}"#,
            r#"{"v":1,"kids":[{"v":2,"x":1}]}"#,
        ],
    ),
    (
        r##"{"$defs":{"node":{"type":"object","properties":{"value":{"type":"integer"},"next":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/node"}]}},"required":["value","next"],"additionalProperties":false}},"$ref":"#/$defs/node"}"##,
        &[
            r#"{"value":1,"next":null}"#,
            r#"{"value":1,"next":{"value":2,"next":null}}"#,
        ],
        &[r#"{"value":1}"#, r#"{"value":1,"next":{"value":2}}"#],
    ),
    (
        r##"{"definitions":{"box":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false}},"$ref":"#/definitions/box"}"##,
        &[r#"{"n":1}"#],
        &[r#"{"n":"x"}"#, "{}"],
    ),
    (
        r#"{"anyOf":[{"const":1},{"const":10}]}"#,
        &["1", "10"],
        &["0", "11", "2"],
    ),
    (
        r#"{"anyOf":[{"type":"string","maxLength":2},{"type":"string","minLength":3,"maxLength":4}]}"#,
        &[r#""""#, r#""ab""#, r#""abc""#, r#""abcd""#],
        &[r#""abcde""#],
    ),
    (
        r#"{"anyOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false},{"type":"array","items":{"type":"integer"},"maxItems":2}]}"#,
        &[r#"{"a":1}"#, "[1]", "[1,2]"],
        &[r#"{"b":1}"#, "[1,2,3]"],
    ),
    (
        r#"{"oneOf":[{"type":"integer"},{"type":"boolean"}]}"#,
        &["1", "true"],
        &[r#""x""#],
    ),
    (
        r#"{"allOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}]}"#,
        &[r#"{"a":1}"#],
        &["{}"],
    ),
    (
        r#"{"allOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},{"type":"object","properties":{"b":{"type":"string"}}}]}"#,
        &[r#"{"a":1}"#, r#"{"a":1,"b":"x"}"#],
        &[r#"{"b":"x"}"#, r#"{"a":"x"}"#],
    ),
    (
        r#"{"type":"object","properties":{"s":{"type":"string","minLength":1,"maxLength":3},"n":{"type":"integer","minimum":0,"maximum":2}},"required":["s","n"],"additionalProperties":false}"#,
        &[
            r#"{"s":"a","n":0}"#,
            r#"{"s":"abc","n":2}"#,
            r#"{"s":"é😀","n":1}"#,
        ],
        &[
            r#"{"s":"","n":1}"#,
            r#"{"s":"abcd","n":1}"#,
            r#"{"s":"a","n":3}"#,
        ],
    ),
    (
        r#"{"type":"object","properties":{"level1":{"type":"object","properties":{"level2":{"type":"array","items":{"type":"object","properties":{"x":{"type":"boolean"}},"required":["x"],"additionalProperties":false}}},"required":["level2"],"additionalProperties":false}},"required":["level1"],"additionalProperties":false}"#,
        &[r#"{"level1":{"level2":[{"x":true},{"x":false}]}}"#],
        &[r#"{"level1":{"level2":[{}]}}"#],
    ),
    (
        r#"{"type":"array","items":{"type":"array","items":{"type":"integer"},"maxItems":2},"maxItems":2}"#,
        &["[]", "[[1],[2]]", "[[],[1]]"],
        &["[[1,2,3]]", "[[1],[2],[3]]"],
    ),
    (
        r#"{"type":"object","additionalProperties":{"type":"integer"}}"#,
        &["{}", r#"{"a":1,"b":2}"#],
        &[r#"{"a":"x"}"#],
    ),
    (
        r#"{"type":"object"}"#,
        &["{}", r#"{"a":1,"b":[true]}"#],
        &["[]", "1"],
    ),
    (
        r#"{"type":"array","prefixItems":[{"const":"a"},{"const":"b"}],"items":{"type":"integer"},"minItems":2}"#,
        &[r#"["a","b"]"#, r#"["a","b",1,2]"#],
        &[r#"["a"]"#, r#"["b","a"]"#],
    ),
];

#[test]
fn every_schema_accepts_its_valid_instances_and_rejects_its_invalid_ones() {
    for (i, (schema_text, valid, invalid)) in CASES.iter().enumerate() {
        let c = constraint(schema_text);
        let root = parse_json(schema_text).expect("the schema parses");
        for text in *valid {
            let mut m = new_matcher(&c);
            let bytes = text.as_bytes();
            for (at, &b) in bytes.iter().enumerate() {
                assert!(
                    m.advance(b),
                    "schema {i} rejected the prefix of {text:?} at byte {at} ({schema_text})"
                );
            }
            assert!(
                m.can_end(),
                "schema {i} did not accept the complete instance {text:?} ({schema_text})"
            );
            let value = parse_json(text).expect("the independent parser accepts a valid instance");
            validate(&value, &root, &root)
                .unwrap_or_else(|e| panic!("schema {i} validator refused {text:?}: {e}"));
        }
        for text in *invalid {
            assert!(
                !complete(&c, text),
                "schema {i} accepted the invalid instance {text:?} ({schema_text})"
            );
        }
    }
}

// --------------------------------------------------------------------------------------
// Refusals: every unsupported keyword, with its pointer.
// --------------------------------------------------------------------------------------

#[test]
fn unsupported_keywords_are_refused_with_the_keyword_and_pointer() {
    let cases: &[(&str, &str, &str)] = &[
        (r#"{"type":"string","pattern":"a+"}"#, "pattern", "#"),
        (
            r#"{"type":"object","properties":{"x":{"type":"string","format":"date-time"}}}"#,
            "format",
            "#/properties/x",
        ),
        (
            r#"{"type":"object","patternProperties":{"^a":{"type":"integer"}},"type":"object"}"#,
            "patternProperties",
            "#",
        ),
        (r#"{"type":"array","uniqueItems":true}"#, "uniqueItems", "#"),
        (r#"{"type":"integer","multipleOf":2}"#, "multipleOf", "#"),
        (
            r#"{"type":"object","dependentRequired":{"a":["b"]}}"#,
            "dependentRequired",
            "#",
        ),
        (
            r#"{"type":"object","if":{"required":["a"]},"then":{"required":["b"]}}"#,
            "if",
            "#",
        ),
        (r#"{"type":"string","not":{"const":"a"}}"#, "not", "#"),
        (
            r#"{"type":"object","minProperties":1}"#,
            "minProperties",
            "#",
        ),
        (
            r#"{"type":"object","propertyNames":{"type":"string"}}"#,
            "propertyNames",
            "#",
        ),
        (
            r#"{"type":"array","contains":{"type":"integer"}}"#,
            "contains",
            "#",
        ),
        (
            r#"{"type":"object","unevaluatedProperties":false}"#,
            "unevaluatedProperties",
            "#",
        ),
        (
            r#"{"type":"string","minLength":1,"maxLength":2,"pattern":"x"}"#,
            "pattern",
            "#",
        ),
        (r#"{"type":"number","minimum":0}"#, "minimum", "#"),
        (r#"{"type":"integer","minimum":10}"#, "minimum", "#"),
        (r#"{"type":"integer","maximum":10}"#, "maximum", "#"),
        (
            r#"{"type":"integer","minimum":0,"maximum":2000}"#,
            "minimum",
            "#",
        ),
        (
            r#"{"type":"string","enum":["a"],"minLength":2}"#,
            "minLength",
            "#",
        ),
        (r#"{"properties":{"a":{"type":"integer"}}}"#, "type", "#"),
        (
            r#"{"type":"object","properties":{"a":{"$ref":"other.json#/x"}}}"#,
            "$ref",
            "#/properties/a",
        ),
    ];
    for &(text, keyword, pointer) in cases {
        let err = constraint(text)
            .compile()
            .expect_err("the schema should be refused");
        assert_eq!(err.keyword, keyword, "in {text}");
        assert_eq!(err.pointer, pointer, "in {text}");
        assert!(err.message().contains(keyword));
        assert!(err.message().contains(pointer));
    }
}

#[test]
fn a_nullable_repeat_is_refused() {
    let empty = Constraint::sequence(Vec::new());
    let c = Constraint::repeat(empty, 0, None);
    let err = c.compile().expect_err("a nullable repeat must not loop");
    assert_eq!(err.keyword, "repeat");
}

#[test]
fn a_repeat_with_max_below_min_is_refused() {
    let c = Constraint::repeat(
        Constraint::sequence(vec![Part::Literal("x".into())]),
        3,
        Some(2),
    );
    assert!(c.compile().is_err());
}

// --------------------------------------------------------------------------------------
// The combinators the tool wiring will use.
// --------------------------------------------------------------------------------------

#[test]
fn the_combinators_match_made_up_literals_byte_for_byte() {
    let a = Constraint::sequence(vec![
        Part::Literal("<<A>>".into()),
        Part::Json(core_schema(r#"{"type":"integer","minimum":0,"maximum":1}"#)),
        Part::Literal("<<B>>".into()),
    ]);
    let b = Constraint::sequence(vec![Part::Literal("<<C>>".into())]);
    let either = Constraint::any_of(vec![a.clone(), b.clone()]);
    assert!(complete(&either, "<<A>>0<<B>>"));
    assert!(complete(&either, "<<A>>1<<B>>"));
    assert!(complete(&either, "<<C>>"));
    assert!(!complete(&either, "<<A>>2<<B>>"));
    assert!(!complete(&either, "<<A>>0"));

    let repeated = Constraint::repeat(a, 1, Some(2));
    assert!(complete(&repeated, "<<A>>0<<B>>"));
    assert!(complete(&repeated, "<<A>>0<<B>><<A>>1<<B>>"));
    assert!(!complete(&repeated, "<<A>>0<<B>><<A>>1<<B>><<A>>0<<B>>"));
    assert!(!complete(&repeated, "<<A>>0<<B>><<A>>2<<B>>"));
}

#[test]
fn any_of_overlapping_prefixes_keeps_the_longer_alternative_alive() {
    let c = Constraint::any_of(vec![
        Constraint::sequence(vec![Part::Literal("<<A>>".into())]),
        Constraint::sequence(vec![Part::Literal("<<AB>>".into())]),
    ]);
    assert!(complete(&c, "<<A>>"));
    assert!(complete(&c, "<<AB>>"));
    let mut m = new_matcher(&c);
    assert!(m.advance_bytes(b"<<A").is_ok());
    assert!(m.accepts(b'>'));
    assert!(m.accepts(b'B'));
}

// --------------------------------------------------------------------------------------
// The property test: a random model over a synthetic vocabulary.
// --------------------------------------------------------------------------------------

/// A vocabulary with single bytes, multi-byte pieces, tokens that span structural
/// characters, a token that ends in the middle of a UTF-8 character, and one end token.
fn synthetic_vocab() -> (Vocabulary, Vec<u32>) {
    let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let extra: &[&[u8]] = &[
        b"true",
        b"false",
        b"null",
        b"\"abc\"",
        b"{\"",
        b"\": ",
        b"\":",
        b",\"",
        b"}]",
        b"]}",
        b"[]",
        b"{}",
        b"[{",
        b"\"a\"",
        b"123",
        b"0.5",
        b"1e",
        b"\\u",
        b"\\n",
        b"\xC3\xA9",
        b"\xF0\x9F\x98\x80",
        b"<<A>>",
        b"<<B>>",
        b"<|end|>",
    ];
    for e in extra {
        tokens.push(e.to_vec());
    }
    let end = (tokens.len() - 1) as u32;
    (Vocabulary::from_tokens(tokens), vec![end])
}

fn random_logits(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 6.0
        })
        .collect()
}

/// Run the sampler and the matcher together until the constraint ends, and return the text.
fn random_run(
    label: &str,
    c: &Constraint,
    vocab: &Vocabulary,
    ends: &[u32],
    seed: u64,
    max_steps: usize,
) -> String {
    let mut m = new_matcher(c);
    let mut sc = Scratch::default();
    let mut out = Vec::new();
    for step in 0..max_steps {
        if m.is_closed() {
            // Only end tokens are allowed once the top-level value closed.
            let allowed = vocab.allowed(&m, ends);
            assert_eq!(
                allowed.ids(),
                ends,
                "a closed state must allow only end tokens"
            );
            return String::from_utf8(out).expect("the reply is UTF-8");
        }
        let allowed = vocab.allowed(&m, ends);
        assert!(
            !allowed.is_empty(),
            "schema {label} seed {seed}: the matcher allowed no token at step {step}; state {m:?}"
        );
        let logits = random_logits(
            seed.wrapping_mul(1_000_003).wrapping_add(step as u64),
            vocab.len(),
        );
        let key = peregrine_core::sample::position_key(seed, 0, step);
        let pick = sample_constrained(
            &logits,
            1.0,
            1.0,
            0,
            0.0,
            &Penalties::default(),
            &[],
            &allowed,
            key,
            &mut sc,
        )
        .unwrap_or_else(|| panic!("seed {seed}: an empty allowed set was sampled at {step}"));
        assert!(
            allowed.contains(pick),
            "seed {seed}: the sampler chose {pick}, outside the mask"
        );
        if ends.contains(&pick) {
            assert!(
                m.can_end(),
                "seed {seed}: an end token at a non-accepting state"
            );
            return String::from_utf8(out).expect("the reply is UTF-8");
        }
        let bytes = vocab.token_bytes(pick).to_vec();
        assert!(
            !bytes.is_empty(),
            "seed {seed}: a content token with no bytes"
        );
        let mut check = m.clone();
        assert!(
            check.advance_bytes(&bytes).is_ok(),
            "schema {label} seed {seed}: the mask allowed token {pick} ({} bytes, {:?}) that the matcher rejects; state {m:?}",
            bytes.len(),
            String::from_utf8_lossy(&bytes)
        );
        assert!(m.advance_bytes(&bytes).is_ok());
        out.extend_from_slice(&bytes);
    }
    panic!(
        "schema {label} seed {seed}: no complete document within {max_steps} steps; text {:?} state {m:?}",
        String::from_utf8_lossy(&out)
    );
}

/// The 24 schemas the property test walks. All have a bounded language, so a random model
/// cannot wander in an unbounded string or array forever.
const PROPERTY_SCHEMAS: &[&str] = &[
    r#"{"type":"boolean"}"#,
    r#"{"type":"null"}"#,
    r#"{"const":1}"#,
    r#"{"enum":[1,2,"a",true,null]}"#,
    r#"{"type":"integer","minimum":0,"maximum":3}"#,
    r#"{"type":"integer","minimum":0}"#,
    r#"{"type":"number"}"#,
    r#"{"type":"string","maxLength":4}"#,
    r#"{"type":"string","minLength":1,"maxLength":2}"#,
    r#"{"type":"array","items":{"type":"integer","minimum":0,"maximum":2},"maxItems":3}"#,
    r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}"#,
    r#"{"type":"array","items":{"type":"string","maxLength":2},"maxItems":2}"#,
    r#"{"type":"array","prefixItems":[{"type":"string","maxLength":2},{"type":"integer","minimum":0,"maximum":2}],"maxItems":2}"#,
    r#"{"type":"object","properties":{},"additionalProperties":false}"#,
    r#"{"type":"object","properties":{"a":{"type":"integer","minimum":0,"maximum":2}},"required":["a"],"additionalProperties":false}"#,
    r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"}},"additionalProperties":false}"#,
    r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},"required":["c"],"additionalProperties":false}"#,
    r#"{"anyOf":[{"const":1},{"const":10}]}"#,
    r#"{"anyOf":[{"type":"string","maxLength":2},{"type":"integer","minimum":0,"maximum":1}]}"#,
    r#"{"oneOf":[{"type":"null"},{"type":"boolean"}]}"#,
    r#"{"type":["string","null"],"maxLength":3}"#,
    r#"{"type":"object","properties":{"x":{"type":"array","items":{"type":"integer"},"maxItems":2}},"required":["x"],"additionalProperties":false}"#,
    r##"{"type":"object","properties":{"t":{"$ref":"#/$defs/t"}},"required":["t"],"additionalProperties":false,"$defs":{"t":{"type":"object","properties":{"n":{"type":"integer","minimum":0,"maximum":1}},"required":["n"],"additionalProperties":false}}}"##,
    r#"{"type":"array","items":{"type":"array","items":{"type":"boolean"},"maxItems":1},"maxItems":2}"#,
];

#[test]
fn a_random_model_never_leaves_the_schema() {
    let (vocab, ends) = synthetic_vocab();
    let runs = 2000;
    for (i, schema_text) in PROPERTY_SCHEMAS.iter().enumerate() {
        let c = constraint(schema_text);
        let root = parse_json(schema_text).expect("the schema parses");
        for seed in 0..runs {
            let text = random_run(schema_text, &c, &vocab, &ends, seed as u64 + 1, 600);
            let value = parse_json(&text).unwrap_or_else(|e| {
                panic!("schema {i} seed {seed}: the reply is not JSON ({e}): {text:?}")
            });
            validate(&value, &root, &root).unwrap_or_else(|e| {
                panic!("schema {i} seed {seed}: the reply does not validate ({e}): {text:?}")
            });
        }
    }
}

// --------------------------------------------------------------------------------------
// The trie-walk mask equals the brute-force mask.
// --------------------------------------------------------------------------------------

#[test]
fn the_mask_equals_the_brute_force_mask_at_random_states() {
    let (vocab, ends) = synthetic_vocab();
    for schema_text in PROPERTY_SCHEMAS {
        let c = constraint(schema_text);
        for seed in 0..8u64 {
            let mut m = new_matcher(&c);
            let mut logits = random_logits(seed + 1000, vocab.len());
            let mut sc = Scratch::default();
            for step in 0..40 {
                let got = vocab.allowed(&m, &ends);
                let mut want = Vec::new();
                if m.is_closed() {
                    want.extend(ends.iter().copied());
                } else {
                    for id in 0..vocab.len() as u32 {
                        let bytes = vocab.token_bytes(id);
                        if bytes.is_empty() || ends.contains(&id) {
                            continue;
                        }
                        let mut next = m.clone();
                        if next.advance_bytes(bytes).is_ok() {
                            want.push(id);
                        }
                    }
                    if m.can_end() {
                        want.extend(ends.iter().copied());
                    }
                }
                want.sort_unstable();
                want.dedup();
                assert_eq!(
                    got.ids(),
                    want,
                    "schema {schema_text:?} seed {seed} step {step} mask disagreement"
                );
                if m.is_closed() || got.is_empty() {
                    break;
                }
                let key = peregrine_core::sample::position_key(seed + 1, 0, step);
                let Some(pick) = sample_constrained(
                    &logits,
                    1.0,
                    1.0,
                    0,
                    0.0,
                    &Penalties::default(),
                    &[],
                    &got,
                    key,
                    &mut sc,
                ) else {
                    break;
                };
                if ends.contains(&pick) {
                    break;
                }
                let bytes = vocab.token_bytes(pick).to_vec();
                if m.advance_bytes(&bytes).is_err() {
                    break;
                }
                logits = random_logits(
                    seed.wrapping_mul(31).wrapping_add(step as u64 + 7),
                    vocab.len(),
                );
            }
        }
    }
}

#[test]
fn a_vocabulary_of_identical_pieces_allows_every_id_that_shares_their_bytes() {
    // Two tokens with the same bytes must both be in the mask.
    let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    tokens.push(b"ab".to_vec());
    tokens.push(b"ab".to_vec());
    let vocab = Vocabulary::from_tokens(tokens);
    let c = constraint(r#"{"type":"string","maxLength":4}"#);
    let mut m = new_matcher(&c);
    assert!(m.advance(b'"'));
    let allowed = vocab.allowed(&m, &[999]);
    assert!(allowed.contains(256));
    assert!(allowed.contains(257));
}
