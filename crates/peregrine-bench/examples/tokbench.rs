//! Tokenizer throughput: MB/s and tokens/s for WordPiece and SentencePiece unigram.
//!
//! Design §4.3 item 4: on a MiniLM-class model the tokenizer is a visible share of a
//! request, so it is written allocation-free per token and measured. This example
//! encodes a ~1 MB text (English, CJK, accents, punctuation, digits and emoji mixed,
//! which is what an embedding service actually receives) and reports the best of a
//! few runs, plus the token rate.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example tokbench
//! cargo run --release -p peregrine-bench --example tokbench -- path/to/model.gguf
//! ```
//!
//! With no argument it uses the two vocabularies the tokenizer tests carry
//! (`crates/peregrine-core/tests/fixtures/tokenize-vocab-*.bin`). A `.gguf` argument
//! is read through the real loader instead. The comparison number is
//! `llama-tokenize -m model.gguf -f text.txt --ids --no-bos` on the same text; see the
//! PR that added the tokenizers for the measurement on this machine.

use peregrine_core::gguf::Gguf;
use peregrine_core::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::time::Instant;

const TARGET_BYTES: usize = 1 << 20;

/// A paragraph of each kind the tokenizers treat differently.
const CORPUS: &[&str] = &[
    "Retrieval-augmented generation answers a question by first finding passages that \
     mention the entities in it. The retriever embeds the question and every passage, \
     ranks passages by cosine similarity, and hands the top few to the generator. ",
    "Normalisasi katalog produk kecantikan ParagonCorp; INCI: Aqua, Glycerin, \
     Niacinamide 5%, Sodium Hyaluronate 0.1%. SKU-12345 / batch#7 qty:  48. ",
    "café naïve — “quoted” über Ångström; The Quick Brown Fox Jumps Over The Lazy Dog. ",
    "你好，世界！这是一个测试。吾輩は猫である。名前はまだ無い。안녕하세요 세계! ",
    "emoji 🙂 🚀 👩‍💻 and family 👨‍👩‍👧‍👦; price €1,234.56 (≈ ±5%). ",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n",
];

fn corpus(len: usize) -> String {
    let mut s = String::with_capacity(len + 256);
    let mut i = 0;
    while s.len() < len {
        s.push_str(CORPUS[i % CORPUS.len()]);
        i += 1;
    }
    s
}

fn load(path: &Path) -> Tokenizer {
    let bytes = std::fs::read(path).expect("read the vocabulary");
    if bytes.starts_with(b"GGUF") {
        let g = Gguf::open(path).expect("open the model file");
        Tokenizer::from_gguf(&g).expect("build the tokenizer")
    } else {
        Tokenizer::from_vocab_dump(&bytes).expect("load the vocabulary fixture")
    }
}

fn bench(tk: &Tokenizer, text: &str, reps: usize) -> (f64, usize) {
    let mut out = Vec::with_capacity(text.len() / 3);
    let mut best = f64::INFINITY;
    let mut n_tokens = 0;
    for _ in 0..reps {
        out.clear();
        let t = Instant::now();
        tk.encode_into(text, &mut out);
        let dt = t.elapsed().as_secs_f64();
        best = best.min(dt);
        n_tokens = out.len();
    }
    (best, n_tokens)
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut args = std::env::args().skip(1).peekable();
    // `--text PATH` writes the benchmark text so the same bytes can be fed to
    // `llama-tokenize -f` for an apples-to-apples timing.
    if let Some(a) = args.peek() {
        if a == "--text" {
            args.next();
            let out = args.next().unwrap_or_else(|| "tokbench.txt".into());
            std::fs::write(&out, corpus(TARGET_BYTES)).expect("write the text");
            println!("wrote {out}");
            return;
        }
    }
    let paths: Vec<PathBuf> = args.map(PathBuf::from).collect();
    let paths = if paths.is_empty() {
        vec![
            root.join("crates/peregrine-core/tests/fixtures/tokenize-vocab-wpm-minilm.bin"),
            root.join("crates/peregrine-core/tests/fixtures/tokenize-vocab-ugm-t5.bin"),
        ]
    } else {
        paths
    };
    let text = corpus(TARGET_BYTES);
    let mb = text.len() as f64 / 1e6;
    let reps = 20;
    println!(
        "{:<32} {:>9} {:>10} {:>12} {:>10}",
        "vocabulary", "tokens", "MB/s", "tokens/s", "MB"
    );
    for path in paths {
        let tk = load(&path);
        let (dt, n_tokens) = bench(&tk, &text, reps);
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        println!(
            "{:<32} {:>9} {:>10.1} {:>12.0} {:>10.2}",
            name,
            n_tokens,
            mb / dt,
            n_tokens as f64 / dt,
            mb
        );
    }
}
