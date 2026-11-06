#![allow(clippy::ptr_arg)]
//! Tests for the SentencePiece (llama "spm") tokenizer against llama.cpp ground truth
//! and standalone unit tests on a synthetic SPM vocabulary.
//!
//! Ground truth files are fetched via `scripts/fetch-test-vocabs.sh` from llama.cpp
//! release tag `v0.9.0` (commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8`).

use peregrine_core::gguf::Gguf;
use peregrine_core::tokenizer::Tokenizer;
use std::path::PathBuf;

fn vocab_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/vocab/ggml-vocab-llama-spm.gguf")
}

/// Parse llama.cpp tokenizer test cases from `.inp` and `.out`.
///
/// Parsing care for llama.cpp's format:
/// 1. `.inp` contains test cases delimited by `\n__ggml_vocab_test__\n`. If the file
///    ends with the delimiter, no trailing empty case is added (matching
///    `test-tokenizer-0.cpp`).
/// 2. `.out` has exactly one line per test case. Empty input produces empty output,
///    which manifests as an empty line. Each non-empty line contains space-separated
///    integer token IDs, potentially surrounded by whitespace. Trimming the line and
///    returning an empty `Vec` for empty lines ensures correct parsing.
fn parse_test_cases(sraw: &str, sout: &str) -> Vec<(String, Vec<u32>)> {
    let sep = "\n__ggml_vocab_test__\n";
    let mut sinp = Vec::new();
    let mut pos = 0usize;
    while pos < sraw.len() {
        if let Some(next) = sraw[pos..].find(sep) {
            sinp.push(sraw[pos..pos + next].to_string());
            pos += next + sep.len();
        } else {
            sinp.push(sraw[pos..].to_string());
            break;
        }
    }

    let mut expected = Vec::new();
    for line in sout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            expected.push(Vec::new());
        } else {
            let toks: Vec<u32> = trimmed
                .split_whitespace()
                .map(|tok| tok.parse::<u32>().expect("valid token integer"))
                .collect();
            expected.push(toks);
        }
    }

    assert_eq!(
        sinp.len(),
        expected.len(),
        "input cases count ({}) differs from output lines count ({})",
        sinp.len(),
        expected.len()
    );

    sinp.into_iter().zip(expected).collect()
}

#[test]
fn llama_spm_encoding_and_decoding_match_llama_cpp() {
    let path = vocab_path();
    if !path.exists() {
        eprintln!(
            "\n  SKIPPED: SPM test vocab absent at {:?}\n  Run ./scripts/fetch-test-vocabs.sh to download.\n",
            path
        );
        return;
    }

    let inp_path = path.with_extension("gguf.inp");
    let out_path = path.with_extension("gguf.out");
    assert!(inp_path.exists(), "missing inp file at {:?}", inp_path);
    assert!(out_path.exists(), "missing out file at {:?}", out_path);

    let g = Gguf::open(&path).expect("open spm vocab model");
    let tk = Tokenizer::from_gguf(&g).expect("build spm tokenizer");

    assert_eq!(tk.n_vocab, 32000);
    assert_eq!(tk.bos, 1);
    assert_eq!(tk.eos, 2);
    assert_eq!(tk.n_byte_tokens(), 256);

    let sinp_str = std::fs::read_to_string(&inp_path).expect("read inp");
    let sout_str = std::fs::read_to_string(&out_path).expect("read out");
    let cases = parse_test_cases(&sinp_str, &sout_str);

    assert_eq!(
        cases.len(),
        46,
        "expected 46 standard llama.cpp SPM test cases"
    );

    let mut encode_passed = 0;
    let mut decode_passed = 0;
    let mut failures = Vec::new();

    for (i, (inp, want_ids)) in cases.iter().enumerate() {
        let got_ids = tk.encode(inp);
        if got_ids != *want_ids {
            failures.push(format!(
                "Case {i} encode mismatch:\n  input:    {inp:?}\n  expected: {want_ids:?}\n  got:      {got_ids:?}"
            ));
        } else {
            encode_passed += 1;
        }

        // Verify that decoding inverts encoding (and matches input text)
        let decoded = tk.decode(want_ids);
        if decoded != *inp {
            failures.push(format!(
                "Case {i} decode mismatch:\n  input:    {inp:?}\n  expected: {inp:?}\n  decoded:  {decoded:?}"
            ));
        } else {
            decode_passed += 1;
        }
    }

    assert!(
        failures.is_empty(),
        "{} failures in llama SPM test cases:\n{}",
        failures.len(),
        failures.join("\n")
    );

    eprintln!(
        "tokenizer (llama-spm): {} / {} test cases match llama.cpp exactly; {} / {} decode round-trips pass",
        encode_passed,
        cases.len(),
        decode_passed,
        cases.len()
    );
}

#[test]
fn synthetic_spm_vocab_coverage() {
    // A hand-built SPM vocabulary to verify SPM algorithms in CI without downloads:
    // - bigram priority merge by score
    // - byte fallback (<0xXX>)
    // - whitespace escaping (U+2581) and leading space prefix
    // - user-defined / control special tokens
    // - unknown tokens fallback
    let pieces = vec![
        "<unk>".to_string(),      // 0
        "<s>".to_string(),        // 1
        "</s>".to_string(),       // 2
        "[USER]".to_string(),     // 3
        "<0x61>".to_string(),     // 4 ('a')
        "<0x62>".to_string(),     // 5 ('b')
        "<0x63>".to_string(),     // 6 ('c')
        "<0x21>".to_string(),     // 7 ('!')
        "\u{2581}".to_string(),   // 8 (' ')
        "\u{2581}a".to_string(),  // 9
        "\u{2581}b".to_string(),  // 10
        "a".to_string(),          // 11
        "b".to_string(),          // 12
        "ab".to_string(),         // 13
        "\u{2581}ab".to_string(), // 14
        "c".to_string(),          // 15
    ];
    let scores = vec![
        0.0,  // 0
        0.0,  // 1
        0.0,  // 2
        0.0,  // 3
        0.0,  // 4
        0.0,  // 5
        0.0,  // 6
        0.0,  // 7
        -1.0, // 8 (' ')
        -0.5, // 9 (' a')
        -0.5, // 10 (' b')
        -0.8, // 11 ('a')
        -0.8, // 12 ('b')
        10.0, // 13 ('ab')
        20.0, // 14 (' ab')
        -0.8, // 15 ('c')
    ];
    let token_types = vec![
        2, // <unk> = UNKNOWN
        3, // <s> = CONTROL
        3, // </s> = CONTROL
        4, // [USER] = USER_DEFINED
        6, // <0x61> = BYTE
        6, // <0x62> = BYTE
        6, // <0x63> = BYTE
        6, // <0x21> = BYTE
        1, // normal
        1, // normal
        1, // normal
        1, // normal
        1, // normal
        1, // normal
        1, // normal
        1, // normal
    ];

    let tk = Tokenizer::from_spm_parts(pieces, scores, token_types, 1, 2, 0, true);

    // 1. Merge by score: "ab" with add_space_prefix=true becomes " ab" -> token 14
    assert_eq!(tk.encode("ab"), vec![14]);
    assert_eq!(tk.decode(&[14]), "ab"); // leading space stripped on decode

    // 2. Mid-sentence space: "ab ab" -> [14, 14]
    assert_eq!(tk.encode("ab ab"), vec![14, 14]);
    assert_eq!(tk.decode(&[14, 14]), "ab ab");

    // 3. Special tokens matched atomically
    assert_eq!(tk.encode("[USER]ab"), vec![3, 14]);
    assert_eq!(tk.decode(&[3, 14]), "[USER] ab");

    // 4. Byte fallback: '!' (0x21) is token 7
    assert_eq!(tk.encode("!"), vec![8, 7]);
    assert_eq!(tk.decode(&[8, 7]), "!");

    // 5. Raw bytes representation
    assert_eq!(tk.token_bytes(4), vec![b'a']);
    assert_eq!(tk.token_bytes(7), vec![b'!']);
    assert_eq!(tk.token_bytes(9), b" a".to_vec());
}
