// The workspace lints with -D warnings; these idioms are kept as they are.
#![allow(clippy::ptr_arg)]
//! The SentencePiece tokenizer on real Llama 2 and Mistral files, against llama.cpp.
//!
//! The SentencePiece tokenizer was so far only checked against llama.cpp's
//! vocabulary-only test file (`tokenize_spm.rs`), which is a table of ids for a
//! synthetic vocabulary. That says the merge algorithm is right; it does not say
//! the tokenizer reads a real model's own vocabulary the way the model was
//! trained. This test runs the same tokenizer over the two SentencePiece
//! families Peregrine ships chat templates for — Llama 2 and Mistral 7B — on the
//! exact GGUF uploads the goldens and the chat path use, and compares every
//! token against `llama-tokenize` on the same file.
//!
//! ```text
//! llama-tokenize -m Llama-2-7b-chat-hf-Q4_K_M.gguf -f case.txt --ids --no-escape
//! ```
//!
//! Two columns are compared for every string:
//!
//! * `Tokenizer::encode` against llama.cpp's `--no-bos --ids` (the tokenizer
//!   itself adds no BOS), and
//! * `Tokenizer::encode_prompt` against llama.cpp's `--ids` (the prompt path
//!   adds exactly one BOS).
//!
//! The strings cover the cases that break a SentencePiece implementation:
//! leading, trailing and repeated spaces, newlines and tabs, digits, code,
//! punctuation, emoji, Chinese, Japanese, accented Latin, Greek and Cyrillic,
//! and the literal special-token texts `<s>`, `</s>` and `[INST]`.

use peregrine_core::gguf::Gguf;
use peregrine_core::tokenizer::Tokenizer;
use std::path::PathBuf;

/// One real model file and the fixture with llama.cpp's ids for it.
struct Oracle {
    /// The file name under `~/models/` (or `models/`).
    file: &'static str,
    /// The fixture with the ids llama.cpp produced on that exact file.
    tsv: &'static str,
    /// The vocabulary size the file has.
    n_vocab: usize,
}

const ORACLES: &[Oracle] = &[
    Oracle {
        file: "Llama-2-7b-chat-hf-Q4_K_M.gguf",
        tsv: include_str!("fixtures/tokenize-golden-llama2-spm.tsv"),
        n_vocab: 32000,
    },
    Oracle {
        file: "Mistral-7B-Instruct-v0.2.Q4_K_M.gguf",
        tsv: include_str!("fixtures/tokenize-golden-mistral-spm.tsv"),
        n_vocab: 32000,
    },
];

/// Where the model files live: `models/` in the repository, then `~/models/`.
fn find(file: &str) -> Option<PathBuf> {
    let mut dirs = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")];
    if let Ok(h) = std::env::var("HOME") {
        dirs.push(PathBuf::from(h).join("models"));
    }
    dirs.into_iter().map(|d| d.join(file)).find(|p| p.is_file())
}

/// Undo the escaping the fixture uses: `\n`, `\t`, `\r`, `\\` and `\u{...}`.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some('u') => {
                // `\u{XXXX}`: a code point written out, so that invisible
                // characters (a zero-width joiner, say) stay visible here.
                assert_eq!(it.next(), Some('{'), "malformed \\u escape in the fixture");
                let mut hex = String::new();
                for c in it.by_ref() {
                    if c == '}' {
                        break;
                    }
                    hex.push(c);
                }
                let cp = u32::from_str_radix(&hex, 16).expect("valid code point");
                out.push(char::from_u32(cp).expect("valid code point"));
            }
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// One fixture line: the name, the text, and llama.cpp's ids with and without BOS.
struct Case {
    name: String,
    text: String,
    no_bos: Vec<u32>,
    bos: Vec<u32>,
}

fn parse(tsv: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    for line in tsv
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let mut it = line.split('\t');
        let name = it.next().expect("name column").to_string();
        let text = it.next().expect("text column");
        let no_bos = it.next().expect("ids column");
        let bos = it.next().expect("ids column");
        let ids = |s: &str| -> Vec<u32> {
            s.split(',')
                .filter(|x| !x.trim().is_empty())
                .map(|x| x.trim().parse().expect("token id"))
                .collect()
        };
        cases.push(Case {
            name,
            text: unescape(text),
            no_bos: ids(no_bos),
            bos: ids(bos),
        });
    }
    cases
}

#[test]
fn spm_tokenization_matches_llama_cpp_on_real_files() {
    let mut checked = 0;
    for oracle in ORACLES {
        let Some(path) = find(oracle.file) else {
            eprintln!(
                "\n  SKIPPED: the SentencePiece oracle needs {} at ~/models/\n",
                oracle.file
            );
            continue;
        };
        let g = Gguf::open(&path).expect("open model");
        let tk = Tokenizer::from_gguf(&g).expect("build tokenizer");

        assert_eq!(
            tk.n_vocab, oracle.n_vocab,
            "{}: vocabulary size",
            oracle.file
        );
        assert_eq!(tk.pre.name(), "default", "{}: pre-tokenizer", oracle.file);
        assert_eq!(tk.n_byte_tokens(), 256, "{}: byte tokens", oracle.file);
        assert!(tk.add_bos, "{}: the file asks for a BOS", oracle.file);
        assert_eq!(tk.bos, 1, "{}: BOS id", oracle.file);
        assert_eq!(tk.eos, 2, "{}: EOS id", oracle.file);

        let cases = parse(oracle.tsv);
        assert!(
            cases.len() >= 20,
            "{}: the fixture is too thin",
            oracle.file
        );

        let mut failures = Vec::new();
        let mut double_bos = Vec::new();
        for c in &cases {
            // The tokenizer itself adds no BOS, so this is llama.cpp's `--no-bos`
            // answer, exactly.
            let got = tk.encode(&c.text);
            if got != c.no_bos {
                failures.push(format!(
                    "  [{}] {:?}\n    llama.cpp   --no-bos --ids  {:?}\n    peregrine   encode          {:?}",
                    c.name, c.text, c.no_bos, got
                ));
            }

            // The prompt path adds exactly one BOS. llama.cpp prepends the
            // model's BOS to every prompt *unconditionally* — including a prompt
            // that already begins with one, which it warns about ("the final
            // prompt starts with 2 BOS tokens"). Peregrine keeps one. The two
            // therefore agree everywhere except on a string that starts with the
            // BOS token itself, where llama.cpp has one BOS too many.
            let expected = if c.no_bos.first() == Some(&tk.bos) {
                double_bos.push(c.name.clone());
                c.no_bos.clone()
            } else {
                c.bos.clone()
            };
            let got = tk.encode_prompt(&c.text);
            if got != expected {
                failures.push(format!(
                    "  [{}] {:?}\n    expected                   {:?}\n    peregrine   encode_prompt   {:?}",
                    c.name, c.text, expected, got
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{}: {} of {} tokenisations differ from llama.cpp:\n{}",
            oracle.file,
            failures.len(),
            cases.len(),
            failures.join("\n")
        );
        eprintln!(
            "tokenizer ({}): {} strings, encode matches llama.cpp on all of them",
            oracle.file,
            cases.len()
        );
        if !double_bos.is_empty() {
            eprintln!(
                "tokenizer ({}): {} string(s) start with the BOS token; llama.cpp prepends a \
                 second BOS there and warns, Peregrine keeps exactly one (encode_prompt matches \
                 llama.cpp on the other {}): {}",
                oracle.file,
                double_bos.len(),
                cases.len() - double_bos.len(),
                double_bos.join(", ")
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 1 || !ORACLES.iter().any(|o| find(o.file).is_some()),
        "no SentencePiece oracle model present"
    );
}
