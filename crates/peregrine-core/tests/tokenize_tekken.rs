//! The Tekken pre-tokenizer must agree with llama.cpp token for token.
//!
//! NVIDIA's Nemotron 3.5 Embed 1B vocabulary is byte-level BPE with
//! `tokenizer.ggml.pre = "pixtral"`, which llama.cpp v0.9.0 maps to
//! `LLAMA_VOCAB_PRE_TYPE_TEKKEN` (`src/llama-vocab.cpp` lines 2619-2634) with
//! `ignore_merges`. Unlike every other pre-type, llama.cpp has no hand-written
//! splitter for it: the pattern runs through `std::regex` over a collapsed text
//! (`src/unicode.cpp`), so only ASCII case splits words and a non-ASCII letter counts
//! as both upper and lower. `Tokenizer` reproduces that evaluation; this test checks
//! it on 361 strings against ids llama.cpp produced, and the `nemotron-3.5-embed`
//! pre-type (the same pattern plus the zero-width-non-joiner rule) on 22 more.
//!
//! The 750 MB model file is not committed, so the test builds the tokenizer from
//! `models/nemotron/` or `~/models/nemotron/` and skips with a one-line notice when it
//! is absent (CI has none of these files). The splitter's own tests in `src/tokenizer.rs` run in
//! CI and carry one case per alternative of the pattern.
//!
//! The fixtures also carry the ids of the same string with `parse_special = false`,
//! where the vocabulary's 1000 control tokens are text rather than tokens.

use peregrine_core::gguf::Gguf;
use peregrine_core::tokenizer::{PreType, Tokenizer};
use std::path::PathBuf;

const ORACLE: &str = include_str!("fixtures/tokenize-golden-tekken.tsv");
const NEMOTRON: &str = include_str!("fixtures/tokenize-golden-tekken-nemotron.tsv");

/// Where model files live: `models/` and `models/nemotron/` in the repository, then
/// the same under `~/models/`.
fn find(file: &str) -> Option<PathBuf> {
    let mut dirs = Vec::new();
    for base in [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"),
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("models"),
    ] {
        dirs.push(base.clone());
        dirs.push(base.join("nemotron"));
    }
    dirs.into_iter().map(|d| d.join(file)).find(|p| p.is_file())
}

/// Undo the fixture escaping: `\n`, `\r`, `\t`, `\\` and `\xNN`, as bytes so that a
/// detokenization that is not valid UTF-8 still round-trips.
fn unescape(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'\\' || i + 1 >= b.len() {
            out.push(b[i]);
            i += 1;
            continue;
        }
        match b[i + 1] {
            b'n' => {
                out.push(b'\n');
                i += 2;
            }
            b'r' => {
                out.push(b'\r');
                i += 2;
            }
            b't' => {
                out.push(b'\t');
                i += 2;
            }
            b'\\' => {
                out.push(b'\\');
                i += 2;
            }
            b'x' if i + 3 < b.len() => {
                let hex = std::str::from_utf8(&b[i + 2..i + 4]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(v) => {
                        out.push(v);
                        i += 4;
                    }
                    Err(_) => {
                        out.push(b'\\');
                        i += 1;
                    }
                }
            }
            _ => {
                out.push(b'\\');
                i += 1;
            }
        }
    }
    out
}

fn as_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `[1, 2, 3]` (llama.cpp's printers use this shape).
fn parse_ids(s: &str) -> Vec<u32> {
    s.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.parse().expect("malformed id fixture"))
        .collect()
}

fn tokenizer() -> Option<Tokenizer> {
    let Some(path) = find("Nemotron-3.5-Embed-1B-Q4_K_M.gguf") else {
        eprintln!(
            "  SKIPPED: the Tekken oracle needs Nemotron-3.5-Embed-1B-Q4_K_M.gguf \
             under models/nemotron/ or ~/models/nemotron/"
        );
        return None;
    };
    let g = Gguf::open(&path).expect("open the model file");
    let tk = Tokenizer::from_gguf(&g).expect("build the tokenizer from the model file");
    assert_eq!(tk.n_vocab, 131072, "vocabulary size");
    // The file says `pixtral`; llama.cpp treats the name exactly like `tekken`.
    assert_eq!(tk.pre, PreType::Tekken);
    // The file overrides llama.cpp's `add_bos = true` default for this pre-type, and
    // has no EOS either, so an embedding input adds nothing around the text.
    assert!(!tk.add_bos);
    assert!(!tk.add_eos);
    assert_eq!((tk.bos, tk.eos), (1, 2));
    Some(tk)
}

/// Checks one fixture and returns how many strings it held.
fn check(tk: &Tokenizer, fixture: &str) -> usize {
    let mut cases = 0;
    let mut failures = Vec::new();
    for line in fixture
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let mut cols = line.split('\t');
        let bare = parse_ids(cols.next().expect("bare ids"));
        let special = parse_ids(cols.next().expect("special ids"));
        let text = as_text(&unescape(cols.next().expect("text")));
        let detok = as_text(&unescape(cols.next().expect("detokenization")));
        let unparsed = parse_ids(cols.next().expect("ids without special parsing"));

        let got = tk.encode(&text);
        if got != bare {
            failures.push(format!(
                "  {text:?}\n    llama.cpp   {bare:?}\n    peregrine   {got:?}"
            ));
        }
        // `add_special = true`: the file adds no BOS or EOS, so this is the bare
        // encoding, which the fixture also proves.
        let got_special = tk.encode_special(&text);
        if got_special != special {
            failures.push(format!(
                "  {text:?} with special tokens\n    llama.cpp   {special:?}\n    peregrine   {got_special:?}"
            ));
        }
        // `parse_special = false`: `[INST]`, `<s>` and friends are text.
        let got_unparsed = tk.encode_unparsed(&text);
        if got_unparsed != unparsed {
            failures.push(format!(
                "  {text:?} without parsing specials\n    llama.cpp   {unparsed:?}\n    peregrine   {got_unparsed:?}"
            ));
        }
        // llama.cpp's detokenizer, on the bare ids.
        let round = tk.decode(&bare);
        if round != detok {
            failures.push(format!(
                "  {text:?} decoded\n    llama.cpp   {detok:?}\n    peregrine   {round:?}"
            ));
        }
        cases += 1;
    }
    assert!(
        failures.is_empty(),
        "{} differences from llama.cpp over {cases} strings:\n{}",
        failures.len(),
        failures.join("\n")
    );
    cases
}

#[test]
fn encoding_matches_llama_cpp() {
    let Some(tk) = tokenizer() else {
        return;
    };
    let cases = check(&tk, ORACLE);
    assert!(cases >= 300, "fixture is too thin: {cases} cases");
    eprintln!("tekken (pixtral): {cases} strings match llama.cpp exactly");
}

#[test]
fn the_nemotron_pre_type_matches_llama_cpp() {
    let Some(mut tk) = tokenizer() else {
        return;
    };
    // The fixture was produced with `tokenizer.ggml.pre` overridden to
    // `nemotron-3.5-embed`, which the file itself does not say; switching the
    // pre-type is the same thing here.
    tk.pre = PreType::Nemotron35Embed;
    let cases = check(&tk, NEMOTRON);
    assert!(cases >= 20, "fixture is too thin: {cases} cases");
    eprintln!("tekken (nemotron-3.5-embed): {cases} strings match llama.cpp exactly");
}

#[test]
fn the_joiner_rule_changes_the_ids_it_should() {
    let Some(mut tk) = tokenizer() else {
        return;
    };
    // Special tokens are matched against the raw text before any pre-tokenizing
    // (`tokenizer_st_partition`), so the joiner rule cannot *create* a control token:
    // `[` + U+200C + `INST]` is plain text under both pre-types. What the rule changes
    // is the text that then reaches the splitter: with `pixtral` the joiner survives,
    // with `nemotron-3.5-embed` the text is `[INST]` and tokenizes exactly like the
    // spelling with no control token in it.
    let with_joiner = tk.encode("[\u{200c}INST]");
    let as_text = tk.encode_unparsed("[INST]");
    let untouched = tk.encode("a\u{200c}b");
    assert_ne!(with_joiner, as_text);
    tk.pre = PreType::Nemotron35Embed;
    assert_eq!(tk.encode("[\u{200c}INST]"), as_text);
    assert_eq!(tk.encode("<\u{200c}s>"), tk.encode_unparsed("<s>"));
    // Only after `[` or `<`: everywhere else the joiner is an ordinary character.
    assert_eq!(tk.encode("a\u{200c}b"), untouched);
}

#[test]
fn special_tokens_are_atomic_only_when_parsed() {
    let Some(tk) = tokenizer() else {
        return;
    };
    let inst = tk.id_of("[INST]").expect("[INST] in the vocabulary");
    assert!(tk.is_special(inst));
    assert_eq!(tk.encode("[INST]"), vec![inst]);
    assert_eq!(tk.encode_special("[INST]"), vec![inst]);
    // With `parse_special = false` the spelling is five words: `[`, `INST`, `]`.
    let plain = tk.encode_unparsed("[INST]");
    assert!(
        plain.len() > 1,
        "the control token was matched anyway: {plain:?}"
    );
    assert!(!plain.contains(&inst));
    // The user's text around a token is not swallowed.
    let ids = tk.encode("x[INST]y");
    let mut want = tk.encode("x");
    want.push(inst);
    want.extend(tk.encode("y"));
    assert_eq!(ids, want);
}
