//! Writes the compact vocabulary fixture the tokenizer tests load.
//!
//! The encoder tokenizers (WordPiece, SentencePiece unigram) can only be checked
//! against llama.cpp on the vocabulary the model was trained with, and those
//! vocabularies are 30k to 250k pieces — carrying the model files in the repository
//! is not an option. This tool reads a GGUF file's vocabulary, builds the tokenizer
//! exactly as the engine does, and writes the pieces, scores, token types, charsmap
//! and special-token rules out in the format `Tokenizer::from_vocab_dump` reads:
//!
//! ```text
//! cargo run --release -p peregrine-bench --example vocab_dump -- \
//!     all-MiniLM-L6-v2-Q2_K.gguf \
//!     crates/peregrine-core/tests/fixtures/tokenize-vocab-minilm.bin
//! ```
//!
//! The binary is little-endian throughout:
//!
//! ```text
//!   0   "PGVOC1\0\0"                       magic
//!   8   u8   kind                          1 = WordPiece, 2 = unigram
//!   9   u8   flags                         bit 0 add_bos, 1 add_eos, 2 add_sep,
//!                                          3 add_space_prefix, 4 remove_extra_whitespaces,
//!                                          5 lowercase, 6 strip_accents
//!  10   i32  bos, eos, unk, sep, cls, mask  -1 for "absent"
//!  34   u32  n_pieces
//!  38   u32  charsmap_len
//!  42   ..   charsmap bytes
//!        then n_pieces records: u8 token type, u16 piece length, piece bytes, f32 score
//! ```

use peregrine_core::gguf::Gguf;
use peregrine_core::tokenizer::Tokenizer;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: vocab_dump MODEL.gguf OUT.bin");
        std::process::exit(2);
    }
    let path = Path::new(&args[1]);
    let g = Gguf::open(path).expect("open the model file");
    let tk = Tokenizer::from_gguf(&g).expect("build the tokenizer");
    let dump = tk.export_vocab().expect("export the vocabulary");
    std::fs::write(&args[2], &dump).expect("write the fixture");
    // Read it back: a fixture the loader cannot parse is worse than none.
    let back = Tokenizer::from_vocab_dump(&dump).expect("reload the fixture");
    assert_eq!(back.n_vocab, tk.n_vocab, "vocabulary size changed");
    println!(
        "{}: {} pieces, {} bytes -> {}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        tk.n_vocab,
        dump.len(),
        args[2]
    );
}
