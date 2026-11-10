//! The encoder tokenizers must agree with llama.cpp token for token.
//!
//! WordPiece (`tokenizer.ggml.model = "bert"`) and SentencePiece unigram (`"t5"`,
//! the T5 and XLM-R vocabularies) are checked here against llama.cpp v0.9.0
//! (commit cd959ba) on the vocabularies real embedding models ship. Every id in the
//! `.tsv` fixtures is what llama.cpp printed on that exact GGUF upload; the last
//! column is what llama.cpp's own detokenizer produced for the same ids.
//!
//! ```text
//! python3 scripts/gen-tokenizer-golden.py --help
//! ```
//!
//! The model files are not in the repository, so for the WordPiece vocabularies and
//! the 32k-piece T5 one the vocabulary each fixture needs is carried in a `.bin`
//! beside it (`Tokenizer::from_vocab_dump`), extracted from the GGUF by
//! `crates/peregrine-bench/examples/vocab_dump.rs` and holding exactly what
//! `Tokenizer::from_gguf` read out of the file: the pieces, their scores and types,
//! the precompiled charsmap, and the special-token rules. When the model file is
//! present the test also checks the fixture against the file itself.
//!
//! The XLM-R vocabulary is 250,002 pieces, which would make its `.bin` larger than
//! the rest of the repository's fixtures combined (4.1 MB), so its test builds the
//! tokenizer from `bge-m3-Q2_K.gguf` under `models/` or `~/models/` and skips with a
//! one-line notice when that file is absent. `PEREGRINE_GOLDEN=require` still skips
//! an absent model file (CI has none of these files), so this is a plain skip too.

use peregrine_core::gguf::Gguf;
use peregrine_core::tokenizer::Tokenizer;
use std::path::PathBuf;

struct Oracle {
    /// The fixture's name, used in messages.
    name: &'static str,
    /// The extracted vocabulary, when the test carries it in the repository.
    bin: Option<&'static [u8]>,
    /// llama.cpp's ids for the fixture strings.
    tsv: &'static str,
    /// llama.cpp's ids for the reranker's (query, document) pairs.
    pairs: &'static str,
    /// The GGUF the vocabulary came from, under `~/models/` or `models/`.
    model: &'static str,
    /// The file's vocabulary size, as the fixture asserts.
    n_vocab: usize,
}

const ORACLES: &[Oracle] = &[
    Oracle {
        name: "WordPiece (all-MiniLM-L6-v2, uncased)",
        bin: Some(include_bytes!("fixtures/tokenize-vocab-wpm-minilm.bin")),
        tsv: include_str!("fixtures/tokenize-golden-wpm-minilm.tsv"),
        pairs: include_str!("fixtures/tokenize-golden-wpm-minilm-pair.tsv"),
        model: "all-MiniLM-L6-v2-Q2_K.gguf",
        n_vocab: 30522,
    },
    Oracle {
        name: "WordPiece (KB-BERT swedish-cased, cased vocabulary)",
        bin: Some(include_bytes!("fixtures/tokenize-vocab-wpm-cased.bin")),
        tsv: include_str!("fixtures/tokenize-golden-wpm-cased.tsv"),
        pairs: include_str!("fixtures/tokenize-golden-wpm-cased-pair.tsv"),
        model: "bert-swedish-pos-q4_0.gguf",
        n_vocab: 50325,
    },
    Oracle {
        name: "SentencePiece unigram (flan-t5-small, the T5 vocabulary)",
        bin: Some(include_bytes!("fixtures/tokenize-vocab-ugm-t5.bin")),
        tsv: include_str!("fixtures/tokenize-golden-ugm-t5.tsv"),
        pairs: include_str!("fixtures/tokenize-golden-ugm-t5-pair.tsv"),
        model: "flan-t5-small-q4_0.gguf",
        n_vocab: 32128,
    },
    Oracle {
        name: "SentencePiece unigram (bge-m3, the XLM-R vocabulary)",
        bin: None,
        tsv: include_str!("fixtures/tokenize-golden-ugm-xlmr.tsv"),
        pairs: include_str!("fixtures/tokenize-golden-ugm-xlmr-pair.tsv"),
        model: "bge-m3-Q2_K.gguf",
        n_vocab: 250002,
    },
];

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

/// `[1, 2, 3]` (llama-tokenize and the oracle both print this shape).
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

/// The tokenizer for this fixture: from the carried `.bin`, or from the model file
/// when the vocabulary is too large to commit. `None` means the model file is absent
/// and the caller skips this vocabulary.
fn tokenizer(o: &Oracle) -> Option<Tokenizer> {
    let tk = if let Some(bin) = o.bin {
        Tokenizer::from_vocab_dump(bin).expect("load the vocabulary fixture")
    } else {
        let Some(path) = find(o.model) else {
            eprintln!(
                "  SKIPPED: {} (model file {} not found under models/ or ~/models/)",
                o.name, o.model
            );
            return None;
        };
        let g = Gguf::open(&path).expect("open the model file");
        Tokenizer::from_gguf(&g).expect("build the tokenizer from the model file")
    };
    assert_eq!(tk.n_vocab, o.n_vocab, "{}: vocabulary size", o.name);
    Some(tk)
}

/// Where model files live: `models/` in the repository, then `~/models/`.
fn find(file: &str) -> Option<PathBuf> {
    let mut dirs = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")];
    if let Ok(h) = std::env::var("HOME") {
        dirs.push(PathBuf::from(h).join("models"));
    }
    dirs.into_iter().map(|d| d.join(file)).find(|p| p.is_file())
}

#[test]
fn encoding_matches_llama_cpp() {
    for o in ORACLES {
        let Some(tk) = tokenizer(o) else {
            continue;
        };
        let mut cases = 0;
        let mut failures = Vec::new();
        for line in o
            .tsv
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let mut cols = line.split('\t');
            let bare = parse_ids(cols.next().expect("bare ids"));
            let special = parse_ids(cols.next().expect("special ids"));
            let text = as_text(&unescape(cols.next().expect("text")));
            let detok = as_text(&unescape(cols.next().expect("detokenization")));

            // The bare encoding: llama.cpp with `add_special = false`.
            let got = tk.encode(&text);
            if got != bare {
                failures.push(format!(
                    "  {:?}\n    llama.cpp   {:?}\n    peregrine   {:?}",
                    text, bare, got
                ));
            }
            // The embedding input: llama.cpp with `add_special = true`.
            let got_special = tk.encode_special(&text);
            if got_special != special {
                failures.push(format!(
                    "  {:?} with special tokens\n    llama.cpp   {:?}\n    peregrine   {:?}",
                    text, special, got_special
                ));
            }
            // llama.cpp's detokenizer, on the bare ids.
            let round = tk.decode(&bare);
            if round != detok {
                failures.push(format!(
                    "  {:?} decoded\n    llama.cpp   {:?}\n    peregrine   {:?}",
                    text, detok, round
                ));
            }
            cases += 1;
        }
        assert!(cases >= 40, "{}: fixture is too thin: {cases}", o.name);
        assert!(
            failures.is_empty(),
            "{}: {} differences from llama.cpp over {cases} strings:\n{}",
            o.name,
            failures.len(),
            failures.join("\n")
        );
        eprintln!("{}: {cases} strings match llama.cpp exactly", o.name);
    }
}

#[test]
fn rerank_pairs_match_llama_cpp() {
    for o in ORACLES {
        let Some(tk) = tokenizer(o) else {
            continue;
        };
        let mut cases = 0;
        for line in o
            .pairs
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let mut cols = line.split('\t');
            let query = as_text(&unescape(cols.next().expect("query")));
            let doc = as_text(&unescape(cols.next().expect("document")));
            let want = parse_ids(cols.next().expect("ids"));
            let got = tk.encode_pair(&query, &doc);
            assert_eq!(
                got, want,
                "{}: pair ({query:?}, {doc:?}) differs from llama.cpp's format_prompt_rerank",
                o.name
            );
            cases += 1;
        }
        assert!(cases >= 8, "{}: pair fixture is too thin: {cases}", o.name);
        eprintln!("{}: {cases} rerank pairs match llama.cpp exactly", o.name);
    }
}

/// The fixture must be the vocabulary the model file has: same size, same pieces,
/// same ids. Without this a fixture could drift from the file its goldens came from
/// and every other test here would still pass.
#[test]
fn fixtures_match_the_model_files_when_present() {
    let mut checked = 0;
    for o in ORACLES {
        // A vocabulary that is loaded from the model file has no fixture to compare.
        let Some(bin) = o.bin else {
            continue;
        };
        let Some(path) = find(o.model) else {
            continue;
        };
        let g = Gguf::open(&path).expect("open the model file");
        let from_file = Tokenizer::from_gguf(&g).expect("build the tokenizer from the file");
        let from_fixture = tokenizer(o).expect("the carried vocabulary fixture loads");
        assert_eq!(from_file.n_vocab, from_fixture.n_vocab, "{}", o.name);
        assert_eq!(
            from_file.export_vocab().expect("export"),
            bin,
            "{}: the fixture no longer matches {}",
            o.name,
            path.display()
        );
        for id in [
            0u32,
            1,
            2,
            100,
            (o.n_vocab / 2) as u32,
            o.n_vocab as u32 - 1,
        ] {
            assert_eq!(
                from_file.piece(id),
                from_fixture.piece(id),
                "{}: piece {id}",
                o.name
            );
        }
        checked += 1;
    }
    if checked == 0 {
        eprintln!("  SKIPPED: no encoder model files present for the fixture check");
    }
}

/// The edge cases the goldens cover, asserted on the tokenizers directly so that a
/// failure names the behaviour rather than a line number in a fixture.
#[test]
fn edge_cases() {
    let minilm = tokenizer(&ORACLES[0]).expect("the WordPiece fixture is carried");
    // A literal special token is matched atomically by `encode`, and is not matched
    // by the rerank path (`parse_special = false`).
    let cls = minilm.id_of("[CLS]").expect("[CLS] in the vocabulary");
    assert_eq!(minilm.encode("[CLS]"), vec![cls]);
    assert_ne!(minilm.encode_pair("[CLS]", "")[1], cls);
    // An empty string has no word to tokenize, only the special tokens.
    assert!(minilm.encode("").is_empty());
    assert_eq!(
        minilm.encode_special(""),
        vec![minilm.cls.unwrap(), minilm.sep.unwrap()]
    );
    // The cased vocabulary is still lower-cased, because llama.cpp's loader defaults
    // the normalizer to lowercase and no converter writes the key that would say
    // otherwise. An uppercase word therefore encodes exactly as its lowercase form.
    let cased = tokenizer(&ORACLES[1]).expect("the WordPiece fixture is carried");
    assert_eq!(
        cased.encode("Ångström ÖVERALLT"),
        cased.encode("ångström överallt")
    );
    // ...and the same is true of the uncased one.
    assert_eq!(
        minilm.encode("MiXeD CaSe WoRd"),
        minilm.encode("mixed case word")
    );

    // Unigram: whitespace becomes the escaped space, and repeated or leading spaces
    // collapse because the T5 file sets `remove_extra_whitespaces`. This vocabulary
    // is carried in the repository, unlike XLM-R's.
    let t5 = tokenizer(&ORACLES[2]).expect("the T5 fixture is carried");
    assert!(!t5.encode("hello").is_empty());
    assert_eq!(t5.encode("  hello \t "), t5.encode("hello"));
    assert!(t5.encode("   ").is_empty());
    // A 200-character word is matched piece by piece rather than dropped.
    let long = "a".repeat(200);
    assert!(!t5.encode(&long).is_empty());
    assert!(!minilm.encode(&long).is_empty());
}
