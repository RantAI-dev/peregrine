//! Byte-level BPE tokenizer, built from the vocabulary embedded in the GGUF file.
//!
//! This is the second place (after the block layouts) where being *nearly* right
//! produces fluent, wrong output rather than an error. A tokenizer that splits
//! `"2026"` into one token where the model was trained on four still generates
//! confident English — it is just answering a different prompt than the user typed.
//! So this module is checked against `llama-tokenize`'s own output on real strings
//! (`tests/tokenize.rs`), not against expectation.
//!
//! # Pipeline
//!
//! ```text
//!   text ──▶ split out special tokens (matched atomically, longest first)
//!        ──▶ regex 1: every \p{N} becomes its own piece
//!        ──▶ regex 2: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
//!        ──▶ bytes ──▶ GPT-2 byte-to-char alphabet
//!        ──▶ BPE merges by rank
//!        ──▶ ids
//! ```
//!
//! The two regexes are the ones llama.cpp uses for `LLAMA_VOCAB_PRE_TYPE_SMOLLM`,
//! transcribed from `llama-vocab.cpp`. They are hand-scanned rather than run through
//! a regex engine: the crate has one dependency and this pattern is a fixed,
//! six-way alternation that a scanner expresses more directly than a regex would.
//!
//! More pre-tokenizers are implemented the same way, because the supported models'
//! files use them (Llama 3.2, Qwen 2.5/3.5, gpt-oss, Phi-4/DBRX, Mistral's Tekken):
//!
//! ```text
//!   llama-bpe: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
//!              | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//!   qwen2:     the same with \p{N} in place of \p{N}{1,3}
//! ```
//!
//! `split_llama3` mirrors `unicode_regex_split_custom_llama3` in llama.cpp's
//! `unicode.cpp` branch for branch, since that function — not a regex engine — is
//! what produced the golden fixtures. A model whose `tokenizer.ggml.pre` names
//! anything else is refused at load, never approximated.
//!
//! # Tekken (`tekken`, `pixtral`, `nemotron-3.5-embed`)
//!
//! Mistral's Tekken vocabulary (Nemotron 3.5 Embed among them) uses the case-sensitive
//! pattern llama.cpp carries at `src/llama-vocab.cpp` lines 425-431:
//!
//! ```text
//! [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+
//! [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*
//! \p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! This pre-type is special in one way the others are not: llama.cpp has **no
//! hand-written splitter for it**. It reaches `unicode_regex_split`'s generic path,
//! which collapses every non-ASCII code point to one byte standing for its Unicode
//! class (`src/unicode.cpp`, `unicode_regex_split`) and runs `std::regex` over that.
//! The pattern as written and the pattern as evaluated therefore differ: inside the
//! collapsed text every category class is one byte, so the `\p{Lu}` / `\p{Ll}` of the
//! original (which the comment above llama.cpp's rewrite quotes) would both match
//! every non-ASCII letter, and the rewrite's `[^a-z]` / `[^A-Z]` do the same. Only
//! ASCII case decides where a word splits. [`split_tekken`] reproduces the collapsed
//! classes from the same Unicode tables (`cpt_flags`) and the alternation in order
//! with regex backtracking, so the pieces are what `std::regex` produced — which is
//! what the goldens are checked against.
//!
//! `nemotron-3.5-embed` is that pattern plus one normalization applied before
//! splitting: every `[` or `<` followed by U+200C (zero-width non-joiner) drops the
//! joiner (`src/llama-vocab.cpp` lines 652-680). `tekken-num3` is the same pattern
//! with digits in right-aligned groups of three.
//!
//! # Known divergence
//!
//! `\p{L}` is approximated by `char::is_alphabetic()`, which is the Unicode
//! *Alphabetic* property: `\p{L}` plus `Nl` (letter-like numerals such as `Ⅻ`) and
//! `Other_Alphabetic` (some combining marks). For those few code points this
//! tokenizer takes the `\p{L}+` branch where llama.cpp takes `\p{N}+`. `\p{N}` maps
//! exactly onto `char::is_numeric()`. Rust's standard library does not expose
//! General_Category, and shipping a category table to close a gap this narrow would
//! cost more than it buys — but it is a real difference and it is recorded here
//! rather than discovered later. It applies to the SmolLM, Llama 3, Qwen and GPT-4o
//! splitters; the Tekken splitter reads the generated category tables (`cpt_flags`),
//! because llama.cpp evaluates that pattern over its own collapsed classes.
//!
//! # SPM (SentencePiece)
//!
//! SentencePiece BPE (`tokenizer.ggml.model = "llama"`), transcribed from
//! llama.cpp release tag `v0.9.0` (commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8`):
//! `src/llama-vocab.cpp` (`llm_tokenizer_spm`, `llm_tokenizer_spm_session`).
//!
//! The algorithm operates on symbol lists linked as a doubly-linked chain,
//! prioritising bigrams by vocabulary score (breaking ties by earlier position
//! in the input), resegmenting unmerged symbols, and falling back to `<0xXX>`
//! byte tokens for code points not in the vocabulary. Leading whitespace escaping
//! (` ` / U+2581) and space prefixing (`add_space_prefix`) match llama.cpp's
//! exact behavior.
//!
//! # Encoder models: WordPiece and SentencePiece unigram
//!
//! The two tokenizers encoder models ship (`bert` = WordPiece, `t5` = SentencePiece
//! unigram, the XLM-R / bge-m3 vocabulary) are transcribed from the same file:
//! `llm_tokenizer_wpm` / `llm_tokenizer_wpm_session` and `llm_tokenizer_ugm` /
//! `llm_tokenizer_ugm_session`, with the normalization tables in `src/unicode-data.cpp`
//! (see `unicode_data.rs`, generated by `scripts/gen-unicode-tables.py`).
//!
//! WordPiece's pre-tokenizer is `llm_tokenizer_wpm_session::preprocess`: lower-case
//! (and, unless the file says otherwise, strip accents with llama.cpp's one-pass
//! NFD map), split on whitespace, drop control characters and accent marks, and
//! make every punctuation mark, ASCII symbol and CJK ideograph a word of its own.
//! Each word is then matched greedily, longest token first, against the vocabulary
//! with a phantom `▁` in front of it — the converter stores WordPiece pieces that
//! way. A word no token covers becomes one unknown token. Note that the converter
//! writes no `tokenizer.ggml.normalizer.*` key, so llama.cpp lower-cases *every*
//! WordPiece vocabulary, cased ones included; that quirk is matched here.
//!
//! Unigram's normalization is SentencePiece's own: the file's `precompiled_charsmap`
//! (an XOR-compressed double-array prefix map, plus its replacement strings) rewrites
//! the text, whitespace becomes `▁`, and then a Viterbi search over the vocabulary
//! trie picks the highest-scoring segmentation, with one unknown-token penalty per
//! code point that no token covers (consecutive unknowns collapse). Scores are
//! `f32` in the file and accumulated in `f64`, as in llama.cpp.
//!
//! Special tokens are matched against the raw text *before* either tokenizer runs,
//! longest token first and in vocabulary order for equal lengths, exactly as
//! `llama_vocab::impl::tokenizer_st_partition` does; `encode`/`encode_special` parse
//! them (`parse_special`), while the reranker's query and document are tokenized with
//! `parse_special = false`, where only user-defined tokens are still matched.

use crate::gguf::Gguf;
use crate::unicode_data;
use std::collections::{BinaryHeap, HashMap};

/// GGUF token types.
#[allow(dead_code)]
const TYPE_NORMAL: i32 = 1;
const TYPE_UNKNOWN: i32 = 2;
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;
#[allow(dead_code)]
const TYPE_UNUSED: i32 = 5;
const TYPE_BYTE: i32 = 6;

/// Which pre-tokenizer regex set the vocabulary was trained with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreType {
    /// `smollm`: digits one at a time, then the GPT-2 alternation.
    SmolLm,
    /// `gpt2` / `gpt-2` / `modern-bert`: the GPT-2 alternation alone (contractions,
    /// ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`, `\s+(?!\S)`), with digits in
    /// one run (`src/llama-vocab.cpp`, `LLAMA_VOCAB_PRE_TYPE_GPT2`).
    Gpt2,
    /// `llama-bpe`: Llama 3's pattern, digits in groups of up to three.
    Llama3,
    /// `qwen2`: Llama 3's pattern with digits one at a time.
    Qwen2,
    /// `qwen35`: `qwen2` with combining marks counted as letters (Qwen3.5, Bonsai 2).
    Qwen35,
    /// `gpt-4o`: the `o200k_harmony` pattern gpt-oss's files carry — the
    /// case-sensitive word pattern, digits in groups of up to three, and
    /// contractions.
    Gpt4o,
    /// `dbrx`: same pattern as Llama 3 (Phi-4, DBRX).
    /// llama.cpp v0.9.0 src/llama-vocab.cpp lines 310-315 map LLM_PRE_TOKENIZER_TYPE_DBRX
    /// to the Llama 3 regex split pattern.
    Dbrx,
    /// `tekken` / `pixtral`: Mistral's case-sensitive pattern, digits one at a time.
    /// llama.cpp v0.9.0 maps both names to `LLAMA_VOCAB_PRE_TYPE_TEKKEN`
    /// (`src/llama-vocab.cpp` lines 2619-2634) with `ignore_merges`.
    Tekken,
    /// `tekken-num3`: `tekken` with digits in right-aligned groups of three
    /// (`src/llama-vocab.cpp` lines 432-437).
    TekkenNum3,
    /// `nemotron-3.5-embed`: the Tekken pattern plus the zero-width-non-joiner rule,
    /// with `ignore_merges` (`src/llama-vocab.cpp` lines 2764-2769).
    Nemotron35Embed,
    /// `default`: SentencePiece (SPM) or models without a custom BPE pre-tokenizer.
    Default,
}

impl PreType {
    pub fn from_name(name: &str) -> Option<PreType> {
        Some(match name {
            "smollm" => PreType::SmolLm,
            "gpt2" | "gpt-2" | "modern-bert" => PreType::Gpt2,
            "llama-bpe" => PreType::Llama3,
            "qwen2" => PreType::Qwen2,
            "qwen35" => PreType::Qwen35,
            "gpt-4o" => PreType::Gpt4o,
            "dbrx" => PreType::Dbrx,
            "tekken" | "pixtral" => PreType::Tekken,
            "tekken-num3" => PreType::TekkenNum3,
            "nemotron-3.5-embed" => PreType::Nemotron35Embed,
            "default" => PreType::Default,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            PreType::SmolLm => "smollm",
            PreType::Gpt2 => "gpt2",
            PreType::Llama3 => "llama-bpe",
            PreType::Qwen2 => "qwen2",
            PreType::Qwen35 => "qwen35",
            PreType::Gpt4o => "gpt-4o",
            PreType::Dbrx => "dbrx",
            PreType::Tekken => "tekken",
            PreType::TekkenNum3 => "tekken-num3",
            PreType::Nemotron35Embed => "nemotron-3.5-embed",
            PreType::Default => "default",
        }
    }
}

/// Pre-tokenizer names this crate implements, for the compatibility probe's message.
/// `pixtral` is a second name for `tekken` and `modern-bert` for `gpt2`, as they are
/// in llama.cpp.
pub const PRE_TYPES: &[&str] = &[
    "smollm",
    "gpt2",
    "modern-bert",
    "llama-bpe",
    "qwen2",
    "qwen35",
    "gpt-4o",
    "dbrx",
    "tekken",
    "pixtral",
    "tekken-num3",
    "nemotron-3.5-embed",
    "default",
];

struct BpeData {
    ranks: HashMap<(String, String), u32>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
}

struct SpmData {
    scores: Vec<f32>,
    #[allow(dead_code)]
    token_types: Vec<i32>,
    byte_tokens: [u32; 256],
    decoded_pieces: Vec<Vec<u8>>,
    #[allow(dead_code)]
    unk_id: u32,
    add_space_prefix: bool,
    remove_extra_whitespaces: bool,
}

/// A byte trie over a vocabulary, built once at load.
///
/// WordPiece asks "which token is the longest prefix of this word" and unigram asks
/// it at every code point; a hash map would need a fresh `String` per probe (the
/// allocation the hot path must not make), so both walk a trie instead. Children are
/// contiguous and sorted by byte, so a step is a binary search over at most 256
/// entries. Nodes are 10 bytes plus one edge pair each — for the 250k-token XLM-R
/// vocabulary, about 25 MB, which llama.cpp also pays (its trie is a `std::map` per
/// node, an order of magnitude more).
struct Trie {
    /// The token that ends at this node, or `u32::MAX`.
    token: Vec<u32>,
    /// Index of the node's first edge in [`Trie::edge_to`].
    first: Vec<u32>,
    /// How many children the node has.
    count: Vec<u16>,
    edge_byte: Vec<u8>,
    edge_to: Vec<u32>,
}

impl Trie {
    fn empty() -> Trie {
        Trie {
            token: vec![u32::MAX],
            first: vec![0],
            count: vec![0],
            edge_byte: Vec::new(),
            edge_to: Vec::new(),
        }
    }

    /// Builds the trie over `(piece, id)` pairs. For duplicate pieces the highest id
    /// wins, which is what `token_to_id[word] = i` in llama.cpp's loader leaves behind.
    fn build(items: &[(&str, u32)]) -> Trie {
        let mut order: Vec<u32> = (0..items.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            items[a as usize]
                .0
                .as_bytes()
                .cmp(items[b as usize].0.as_bytes())
                .then_with(|| items[b as usize].1.cmp(&items[a as usize].1))
        });
        let mut t = Trie::empty();
        if !order.is_empty() {
            t.build_range(items, &order, 0, 0);
        }
        t
    }

    fn build_range(&mut self, items: &[(&str, u32)], order: &[u32], depth: usize, node: u32) {
        let mut i = 0;
        if items[order[0] as usize].0.len() == depth {
            // A duplicate piece keeps the highest id: `order` is sorted by bytes and
            // then by descending id, and `token_to_id[word] = i` in llama.cpp's loader
            // leaves the last (highest) one behind.
            // An empty piece would make the root itself a token; llama.cpp renames
            // those to `[EMPTY_i]` at load. No real vocabulary has one.
            if depth > 0 {
                self.token[node as usize] = items[order[0] as usize].1;
            }
            while i < order.len() && items[order[i] as usize].0.len() == depth {
                i += 1;
            }
        }
        let base = self.edge_to.len() as u32;
        self.first[node as usize] = base;
        let (mut n, mut k) = (0u16, i);
        while k < order.len() {
            let b = items[order[k] as usize].0.as_bytes()[depth];
            self.edge_byte.push(b);
            self.edge_to.push(0);
            n += 1;
            k += 1;
            while k < order.len() && items[order[k] as usize].0.as_bytes()[depth] == b {
                k += 1;
            }
        }
        self.count[node as usize] = n;
        for c in 0..n as usize {
            let child = self.token.len() as u32;
            self.token.push(u32::MAX);
            self.first.push(0);
            self.count.push(0);
            self.edge_to[base as usize + c] = child;
        }
        let (mut k, mut c) = (i, 0usize);
        while k < order.len() {
            let b = items[order[k] as usize].0.as_bytes()[depth];
            let mut j = k + 1;
            while j < order.len() && items[order[j] as usize].0.as_bytes()[depth] == b {
                j += 1;
            }
            let child = self.edge_to[base as usize + c];
            self.build_range(items, &order[k..j], depth + 1, child);
            c += 1;
            k = j;
        }
    }

    #[inline]
    fn child(&self, node: u32, b: u8) -> Option<u32> {
        let first = self.first[node as usize] as usize;
        let n = self.count[node as usize] as usize;
        let bytes = &self.edge_byte[first..first + n];
        let i = bytes.binary_search(&b).ok()?;
        Some(self.edge_to[first + i])
    }

    #[inline]
    fn token(&self, node: u32) -> Option<u32> {
        let t = self.token[node as usize];
        (t != u32::MAX).then_some(t)
    }
}

/// WordPiece (`tokenizer.ggml.model = "bert"`), llama.cpp's `LLAMA_VOCAB_TYPE_WPM`.
struct WpmData {
    trie: Trie,
    token_types: Vec<i32>,
    unk_id: u32,
    lowercase: bool,
    strip_accents: bool,
}

/// SentencePiece unigram (`tokenizer.ggml.model = "t5"`), llama.cpp's
/// `LLAMA_VOCAB_TYPE_UGM`.
struct UgmData {
    trie: Trie,
    /// The user-defined tokens alone: `normalize_prefix` hands their text through
    /// untouched so the Viterbi search can match them as one piece (score 0).
    user_trie: Trie,
    scores: Vec<f32>,
    token_types: Vec<i32>,
    unk_id: u32,
    unknown_token_score: f32,
    add_space_prefix: bool,
    remove_extra_whitespaces: bool,
    charsmap: Option<Charsmap>,
}

/// SentencePiece's `precompiled_charsmap`: an XCDA (XOR-compressed compact double
/// array) prefix map plus the NUL-terminated replacement strings it points into.
struct Charsmap {
    xcda: Vec<u32>,
    replacements: Vec<u8>,
}

impl Charsmap {
    /// Parses the blob as llama.cpp's `llm_tokenizer_ugm` constructor does.
    fn parse(blob: &[u8]) -> Result<Charsmap, String> {
        if blob.len() < 4 {
            return Err("precompiled_charsmap is too small for its length header".into());
        }
        let size = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
        if size + 4 >= blob.len() {
            return Err(format!(
                "precompiled_charsmap: {} bytes of XCDA in a {} byte blob",
                size,
                blob.len()
            ));
        }
        let xcda = blob[4..4 + size]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok(Charsmap {
            xcda,
            replacements: blob[4 + size..].to_vec(),
        })
    }

    #[inline]
    fn node(&self, index: u32) -> Option<u32> {
        self.xcda.get(index as usize).copied()
    }

    /// `BASE`, the unsigned part shifted by the leaf bit (llama.cpp's `xcda_array_view`).
    fn base(&self, index: u32) -> Option<u32> {
        let n = self.node(index)?;
        Some((n >> 10) << ((n & (1 << 9)) >> 6))
    }

    fn lcheck(&self, index: u32) -> Option<u32> {
        Some(self.node(index)? & ((1 << 31) | 0xff))
    }

    fn leaf(&self, index: u32) -> Option<bool> {
        Some((self.node(index)? >> 8) & 1 == 1)
    }

    fn value(&self, index: u32) -> Option<u32> {
        Some(self.node(index)? & ((1 << 31) - 1))
    }
}

/// One fragment of the input: literal text, or a special token already matched.
enum Fragment<'a> {
    Raw(&'a str),
    Token(u32),
}

enum TokenizerAlgo {
    Bpe(BpeData),
    Spm(SpmData),
    Wpm(WpmData),
    Ugm(UgmData),
}

#[derive(Clone, Copy, Debug)]
struct Symbol {
    start: u32,
    len: u32,
    prev: i32,
    next: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Bigram {
    left: i32,
    right: i32,
    score: f32,
    size: u32,
}

impl Eq for Bigram {}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct SpmScratch {
    symbols: Vec<Symbol>,
    work_queue: BinaryHeap<Bigram>,
    todo: Vec<(u32, u32)>,
}

pub struct Tokenizer {
    pieces: Vec<String>,
    ids: HashMap<String, u32>,
    specials: Vec<(String, u32)>,
    /// `specials` indexed by first byte: the raw-text scan's lookup (S4.9c).
    special_index: SpecialIndex,
    /// The user-defined subset of `specials`, and its index: llama.cpp's
    /// `tokenizer_st_partition` still matches those when `parse_special` is false.
    user_specials: Vec<(String, u32)>,
    user_index: SpecialIndex,
    pub bos: u32,
    pub eos: u32,
    /// `tokenizer.ggml.add_bos_token`: whether a prompt starts with `bos`. Llama 3
    /// says yes and its template writes the token itself; SmolLM and Qwen say no.
    pub add_bos: bool,
    /// `tokenizer.ggml.add_eos_token`: whether an encoder input ends with `eos`
    /// (SentencePiece unigram does, WordPiece does not).
    pub add_eos: bool,
    /// `tokenizer.ggml.add_sep_token`: whether a pair puts `sep` between the two
    /// texts (BERT does).
    pub add_sep: bool,
    /// `tokenizer.ggml.seperator_token_id` (the key really is spelled that way).
    pub sep: Option<u32>,
    /// `tokenizer.ggml.cls_token_id`.
    pub cls: Option<u32>,
    /// `tokenizer.ggml.mask_token_id`.
    pub mask: Option<u32>,
    pub pre: PreType,
    pub n_vocab: usize,
    algo: TokenizerAlgo,
}

/// The token types of a vocabulary, as llama.cpp reads them. Absent means all normal;
/// a shorter list is a broken file and refused rather than padded.
fn token_types(g: &Gguf, pieces: &[String]) -> Result<Vec<i32>, String> {
    match g.arr_i32("tokenizer.ggml.token_type") {
        Ok(t) => {
            if t.len() < pieces.len() {
                return Err(format!(
                    "vocab has {} tokens but {} token types",
                    pieces.len(),
                    t.len()
                ));
            }
            Ok(t[..pieces.len()].to_vec())
        }
        Err(_) => Ok(vec![TYPE_NORMAL; pieces.len()]),
    }
}

/// The special tokens llama.cpp matches against the raw text before tokenizing:
/// control, user-defined and unknown ones, longest text first and lowest id first
/// for equal lengths.
fn collect_specials(pieces: &[String], types: &[i32]) -> Vec<(String, u32)> {
    let mut specials: Vec<(String, u32)> = pieces
        .iter()
        .zip(types.iter())
        .enumerate()
        .filter(|(_, (_, &t))| t == TYPE_CONTROL || t == TYPE_USER_DEFINED || t == TYPE_UNKNOWN)
        .map(|(i, (p, _))| (p.clone(), i as u32))
        .collect();
    specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
    specials
}

/// The user-defined tokens only: the ones `tokenizer_st_partition` keeps matching
/// when `parse_special` is false, in the same order as [`collect_specials`].
fn collect_user_specials(pieces: &[String], types: &[i32]) -> Vec<(String, u32)> {
    let mut specials: Vec<(String, u32)> = pieces
        .iter()
        .zip(types.iter())
        .enumerate()
        .filter(|(_, (_, &t))| t == TYPE_USER_DEFINED)
        .map(|(i, (p, _))| (p.clone(), i as u32))
        .collect();
    specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
    specials
}

/// A first-byte index over `specials`, so the raw-text scan costs one lookup per
/// byte instead of one `starts_with` per special per byte.
///
/// `collect_specials` sorts the list longest text first (then lowest id), and the
/// scan takes the first entry that matches at a position. [`Self::hit`] tries a
/// byte's candidates in that same order, so the token a position yields — and the
/// text between two tokens — is exactly what the per-special scan produced.
struct SpecialIndex {
    /// Bit `b % 64` of `first[b / 64]`: some special starts with byte `b`.
    first: [u64; 4],
    /// The `specials` indices whose text starts with byte `b`, in `specials` order.
    by_byte: [Vec<u32>; 256],
}

impl Default for SpecialIndex {
    fn default() -> Self {
        Self {
            first: [0; 4],
            by_byte: std::array::from_fn(|_| Vec::new()),
        }
    }
}

impl SpecialIndex {
    fn build(specials: &[(String, u32)]) -> SpecialIndex {
        let mut ix = SpecialIndex::default();
        for (i, (s, _)) in specials.iter().enumerate() {
            let Some(&b) = s.as_bytes().first() else {
                continue;
            };
            ix.first[(b / 64) as usize] |= 1u64 << (b % 64);
            ix.by_byte[b as usize].push(i as u32);
        }
        ix
    }

    /// Whether some special token's text starts with `b`.
    #[inline(always)]
    fn may_start(&self, b: u8) -> bool {
        self.first[(b / 64) as usize] & (1u64 << (b % 64)) != 0
    }

    /// The first special of `specials` that matches at `p`, as `(byte length, id)`.
    ///
    /// `p` must be a char boundary (the scan only ever stops on one).
    #[inline]
    fn hit(&self, text: &str, p: usize, specials: &[(String, u32)]) -> Option<(usize, u32)> {
        let b = *text.as_bytes().get(p)?;
        if !self.may_start(b) {
            return None;
        }
        for &i in &self.by_byte[b as usize] {
            let (s, id) = &specials[i as usize];
            if text[p..].starts_with(s.as_str()) {
                return Some((s.len(), *id));
            }
        }
        None
    }
}

/// The first position at or after `p` where one of `specials` starts, or the text's
/// length when there is none: the end of the plain run before the next special.
fn next_special_in(
    text: &str,
    p: usize,
    index: &SpecialIndex,
    specials: &[(String, u32)],
) -> usize {
    let bytes = text.as_bytes();
    let mut q = p;
    while q < bytes.len() {
        if index.may_start(bytes[q])
            && text.is_char_boundary(q)
            && index.hit(text, q, specials).is_some()
        {
            return q;
        }
        q += 1;
    }
    bytes.len()
}

/// A special-token id from the file, falling back to llama.cpp's per-type default.
///
/// llama.cpp disables a default that is outside the vocabulary and ignores an
/// out-of-range id in the file, keeping the default; both are reproduced here.
fn special_id(g: &Gguf, key: &str, default: Option<u32>, n_vocab: usize) -> Option<u32> {
    let mut id = default.filter(|&d| (d as usize) < n_vocab);
    if let Ok(v) = g.u32(key) {
        if (v as usize) < n_vocab {
            id = Some(v);
        }
    }
    id
}

/// The GPT-2 byte alphabet: every byte maps to a printable code point so that BPE
/// can operate on text while remaining byte-exact and total over arbitrary input.
fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut enc = ['\0'; 256];
    let mut taken = [false; 256];
    for (a, b) in [(0x21u32, 0x7Eu32), (0xA1, 0xAC), (0xAE, 0xFF)] {
        for c in a..=b {
            enc[c as usize] = char::from_u32(c).unwrap();
            taken[c as usize] = true;
        }
    }
    let mut n = 0u32;
    for b in 0..256usize {
        if !taken[b] {
            enc[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    let dec = enc.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
    (enc, dec)
}

/// The special-token configuration of an encoder vocabulary, from a file or a fixture.
struct VocabConfig {
    bos: Option<u32>,
    eos: Option<u32>,
    unk: Option<u32>,
    sep: Option<u32>,
    cls: Option<u32>,
    mask: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    add_sep: bool,
}

/// Assembles a WordPiece tokenizer from a vocabulary and its rules.
fn build_wpm(
    pieces: Vec<String>,
    ids: HashMap<String, u32>,
    types: Vec<i32>,
    cfg: VocabConfig,
    lowercase: bool,
    strip_accents: bool,
) -> Tokenizer {
    let specials = collect_specials(&pieces, &types);
    let user_specials = collect_user_specials(&pieces, &types);
    let items: Vec<(&str, u32)> = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_str(), i as u32))
        .collect();
    let trie = Trie::build(&items);
    drop(items);
    let n_vocab = pieces.len();
    Tokenizer {
        pieces,
        ids,
        special_index: SpecialIndex::build(&specials),
        specials,
        user_index: SpecialIndex::build(&user_specials),
        user_specials,
        bos: cfg.bos.unwrap_or(u32::MAX),
        eos: cfg.eos.unwrap_or(u32::MAX),
        add_bos: cfg.add_bos,
        add_eos: cfg.add_eos,
        add_sep: cfg.add_sep,
        sep: cfg.sep,
        cls: cfg.cls,
        mask: cfg.mask,
        pre: PreType::Default,
        n_vocab,
        algo: TokenizerAlgo::Wpm(WpmData {
            trie,
            token_types: types,
            unk_id: cfg.unk.unwrap_or(u32::MAX),
            lowercase,
            strip_accents,
        }),
    }
}

/// The normalization rules a unigram vocabulary carries in its file.
struct UgmRules {
    charsmap: Option<Charsmap>,
    add_space_prefix: bool,
    remove_extra_whitespaces: bool,
}

/// Assembles a SentencePiece-unigram tokenizer from a vocabulary and its rules.
fn build_ugm(
    pieces: Vec<String>,
    ids: HashMap<String, u32>,
    types: Vec<i32>,
    scores: Vec<f32>,
    cfg: VocabConfig,
    rules: UgmRules,
) -> Tokenizer {
    // llama.cpp: the unknown penalty is a flat 10 below the lowest *normal* score,
    // and the matcher holds normal, user-defined and unused pieces.
    let mut min_score = f32::MAX;
    let mut trie_items: Vec<(&str, u32)> = Vec::new();
    let mut user_items: Vec<(&str, u32)> = Vec::new();
    for (i, p) in pieces.iter().enumerate() {
        let t = types.get(i).copied().unwrap_or(TYPE_NORMAL);
        if t == TYPE_NORMAL {
            min_score = min_score.min(scores.get(i).copied().unwrap_or(0.0));
        }
        if matches!(t, TYPE_NORMAL | TYPE_USER_DEFINED | TYPE_UNUSED) {
            trie_items.push((p.as_str(), i as u32));
        }
        if t == TYPE_USER_DEFINED {
            user_items.push((p.as_str(), i as u32));
        }
    }
    let specials = collect_specials(&pieces, &types);
    let user_specials = collect_user_specials(&pieces, &types);
    let n_vocab = pieces.len();
    let trie = Trie::build(&trie_items);
    let user_trie = Trie::build(&user_items);
    drop(trie_items);
    drop(user_items);
    Tokenizer {
        pieces,
        ids,
        special_index: SpecialIndex::build(&specials),
        specials,
        user_index: SpecialIndex::build(&user_specials),
        user_specials,
        bos: cfg.bos.unwrap_or(u32::MAX),
        eos: cfg.eos.unwrap_or(u32::MAX),
        add_bos: cfg.add_bos,
        add_eos: cfg.add_eos,
        add_sep: cfg.add_sep,
        sep: cfg.sep,
        cls: cfg.cls,
        mask: cfg.mask,
        pre: PreType::Default,
        n_vocab,
        algo: TokenizerAlgo::Ugm(UgmData {
            trie,
            user_trie,
            scores,
            token_types: types,
            unk_id: cfg.unk.unwrap_or(u32::MAX),
            unknown_token_score: min_score - 10.0,
            add_space_prefix: rules.add_space_prefix,
            remove_extra_whitespaces: rules.remove_extra_whitespaces,
            charsmap: rules.charsmap,
        }),
    }
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        let model = g.str("tokenizer.ggml.model")?;
        if !matches!(model, "gpt2" | "llama" | "bert" | "t5") {
            return Err(format!(
                "tokenizer model {model:?} is not implemented — Peregrine reads gpt2 \
                 byte-level BPE, llama SentencePiece BPE, bert WordPiece and t5 \
                 SentencePiece unigram vocabularies"
            ));
        }
        let toks = g.arr_str("tokenizer.ggml.tokens")?;
        let pieces: Vec<String> = toks.to_vec();
        let mut ids = HashMap::with_capacity(pieces.len());
        for (i, p) in pieces.iter().enumerate() {
            // First id wins: a duplicate piece in the vocabulary is a model-file
            // problem, and preferring the lower id matches llama.cpp.
            ids.entry(p.clone()).or_insert(i as u32);
        }
        let n_vocab = pieces.len();

        if model == "bert" {
            let types = token_types(g, &pieces)?;
            let lowercase = g
                .bool("tokenizer.ggml.normalizer.lowercase")
                .unwrap_or(true);
            // llama.cpp defaults `strip_accents` to whatever `lowercase` is, and only
            // then lets the file override it.
            let strip_accents = g
                .bool("tokenizer.ggml.normalizer.strip_accents")
                .unwrap_or(lowercase);
            if g.bool("tokenizer.ggml.normalizer.nfc").unwrap_or(false) {
                // No converter writes this key; no shipped vocabulary needs it.
                return Err(
                    "tokenizer.ggml.normalizer.nfc is not implemented: llama.cpp applies \
                     Unicode NFC before WordPiece, and no llama.cpp-exported WordPiece file \
                     asks for it"
                        .into(),
                );
            }
            let cfg = VocabConfig {
                bos: special_id(g, "tokenizer.ggml.bos_token_id", Some(101), n_vocab),
                eos: special_id(g, "tokenizer.ggml.eos_token_id", None, n_vocab),
                unk: special_id(g, "tokenizer.ggml.unknown_token_id", Some(100), n_vocab),
                sep: special_id(g, "tokenizer.ggml.seperator_token_id", Some(102), n_vocab),
                cls: special_id(g, "tokenizer.ggml.cls_token_id", Some(101), n_vocab),
                mask: special_id(g, "tokenizer.ggml.mask_token_id", Some(103), n_vocab),
                add_bos: g.bool("tokenizer.ggml.add_bos_token").unwrap_or(true),
                add_eos: g.bool("tokenizer.ggml.add_eos_token").unwrap_or(false),
                add_sep: g.bool("tokenizer.ggml.add_sep_token").unwrap_or(true),
            };
            Ok(build_wpm(pieces, ids, types, cfg, lowercase, strip_accents))
        } else if model == "t5" {
            let types = token_types(g, &pieces)?;
            let scores: Vec<f32> = if let Ok(s) = g.arr_f32("tokenizer.ggml.scores") {
                s.to_vec()
            } else if let Ok(s) = g.arr_i32("tokenizer.ggml.scores") {
                s.iter().map(|&x| x as f32).collect()
            } else {
                vec![0.0; pieces.len()]
            };
            let charsmap = match g.arr_u8("tokenizer.ggml.precompiled_charsmap") {
                Ok(b) => Some(Charsmap::parse(b)?),
                Err(_) => None,
            };
            let cfg = VocabConfig {
                bos: special_id(g, "tokenizer.ggml.bos_token_id", None, n_vocab),
                eos: special_id(g, "tokenizer.ggml.eos_token_id", Some(1), n_vocab),
                unk: special_id(g, "tokenizer.ggml.unknown_token_id", Some(2), n_vocab),
                sep: special_id(g, "tokenizer.ggml.seperator_token_id", None, n_vocab),
                cls: special_id(g, "tokenizer.ggml.cls_token_id", None, n_vocab),
                mask: special_id(g, "tokenizer.ggml.mask_token_id", None, n_vocab),
                add_bos: g.bool("tokenizer.ggml.add_bos_token").unwrap_or(false),
                add_eos: g.bool("tokenizer.ggml.add_eos_token").unwrap_or(true),
                add_sep: g.bool("tokenizer.ggml.add_sep_token").unwrap_or(false),
            };
            Ok(build_ugm(
                pieces,
                ids,
                types,
                scores,
                cfg,
                UgmRules {
                    charsmap,
                    add_space_prefix: g.bool("tokenizer.ggml.add_space_prefix").unwrap_or(false),
                    remove_extra_whitespaces: g
                        .bool("tokenizer.ggml.remove_extra_whitespaces")
                        .unwrap_or(false),
                },
            ))
        } else if model == "gpt2" {
            let pre_name = g.str("tokenizer.ggml.pre").unwrap_or("smollm");
            let pre = PreType::from_name(pre_name).ok_or_else(|| {
                format!(
                    "pre-tokenizer {pre_name:?} is not implemented — Peregrine reads {}",
                    PRE_TYPES.join(", ")
                )
            })?;
            let types = g.arr_i32("tokenizer.ggml.token_type")?;
            if toks.len() != types.len() {
                return Err(format!(
                    "vocab has {} tokens but {} token types",
                    toks.len(),
                    types.len()
                ));
            }
            let merges = g.arr_str("tokenizer.ggml.merges")?;

            let mut ranks = HashMap::with_capacity(merges.len());
            for (rank, m) in merges.iter().enumerate() {
                let (l, r) = m
                    .split_once(' ')
                    .ok_or_else(|| format!("merge {rank} is not a pair: {m:?}"))?;
                ranks.insert((l.to_string(), r.to_string()), rank as u32);
            }

            // Control and user-defined tokens are matched against the raw text before
            // any splitting, so that `<|im_start|>` stays one token instead of being
            // shredded into punctuation by the regex.
            let mut specials: Vec<(String, u32)> = pieces
                .iter()
                .zip(types.iter())
                .enumerate()
                .filter(|(_, (_, &t))| t == TYPE_CONTROL || t == TYPE_USER_DEFINED)
                .map(|(i, (p, _))| (p.clone(), i as u32))
                .collect();
            // Longest first, so `<|im_start|>` is preferred over a shorter prefix.
            specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
            let user_specials = collect_user_specials(&pieces, types);

            let (byte_enc, byte_dec) = byte_alphabet();
            let n_vocab = pieces.len();
            Ok(Tokenizer {
                pieces,
                ids,
                special_index: SpecialIndex::build(&specials),
                specials,
                user_index: SpecialIndex::build(&user_specials),
                user_specials,
                bos: g.u32("tokenizer.ggml.bos_token_id").unwrap_or(1),
                eos: g.u32("tokenizer.ggml.eos_token_id").unwrap_or(2),
                // llama.cpp turns `add_bos` on for `llama-bpe` whatever the file says
                // (`llama-vocab.cpp`, the LLAMA3 pre-type branch), and for the Tekken
                // and Nemotron pre-types as a default the file's `add_bos_token` then
                // overrides — Nemotron 3.5 Embed says false, so it gets none.
                add_bos: g.bool("tokenizer.ggml.add_bos_token").unwrap_or(matches!(
                    pre,
                    PreType::Llama3
                        | PreType::Tekken
                        | PreType::TekkenNum3
                        | PreType::Nemotron35Embed
                )),
                add_eos: g.bool("tokenizer.ggml.add_eos_token").unwrap_or(false),
                add_sep: g.bool("tokenizer.ggml.add_sep_token").unwrap_or(false),
                sep: special_id(g, "tokenizer.ggml.seperator_token_id", None, n_vocab),
                cls: special_id(g, "tokenizer.ggml.cls_token_id", None, n_vocab),
                mask: special_id(g, "tokenizer.ggml.mask_token_id", None, n_vocab),
                pre,
                n_vocab,
                algo: TokenizerAlgo::Bpe(BpeData {
                    ranks,
                    byte_enc,
                    byte_dec,
                }),
            })
        } else {
            // llama (SentencePiece)
            let pre_name = g.str("tokenizer.ggml.pre").unwrap_or("default");
            let pre = PreType::from_name(pre_name).unwrap_or(PreType::Default);

            let types: Vec<i32> = if let Ok(t) = g.arr_i32("tokenizer.ggml.token_type") {
                if t.len() != pieces.len() {
                    return Err(format!(
                        "vocab has {} tokens but {} token types",
                        pieces.len(),
                        t.len()
                    ));
                }
                t.to_vec()
            } else {
                vec![TYPE_NORMAL; pieces.len()]
            };

            let scores: Vec<f32> = if let Ok(s) = g.arr_f32("tokenizer.ggml.scores") {
                s.to_vec()
            } else if let Ok(s) = g.arr_i32("tokenizer.ggml.scores") {
                s.iter().map(|&x| x as f32).collect()
            } else {
                vec![0.0; pieces.len()]
            };

            let unk_id = g.u32("tokenizer.ggml.unknown_token_id").unwrap_or(0);
            let bos = g.u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
            let eos = g.u32("tokenizer.ggml.eos_token_id").unwrap_or(2);
            let add_bos = g.bool("tokenizer.ggml.add_bos_token").unwrap_or(true);
            let add_eos = g.bool("tokenizer.ggml.add_eos_token").unwrap_or(false);
            let add_sep = g.bool("tokenizer.ggml.add_sep_token").unwrap_or(false);
            let add_space_prefix = g.bool("tokenizer.ggml.add_space_prefix").unwrap_or(true);
            let remove_extra_whitespaces = g
                .bool("tokenizer.ggml.remove_extra_whitespaces")
                .unwrap_or(false);

            let mut specials: Vec<(String, u32)> = pieces
                .iter()
                .zip(types.iter())
                .enumerate()
                .filter(|(_, (_, &t))| {
                    t == TYPE_CONTROL || t == TYPE_USER_DEFINED || t == TYPE_UNKNOWN
                })
                .map(|(i, (p, _))| (p.clone(), i as u32))
                .collect();
            specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

            let mut byte_tokens = [unk_id; 256];
            for b in 0..=255u8 {
                let hex_token = format!("<0x{:02X}>", b);
                if let Some(&id) = ids.get(&hex_token) {
                    byte_tokens[b as usize] = id;
                } else if let Some(&id) = ids.get((b as char).to_string().as_str()) {
                    byte_tokens[b as usize] = id;
                }
            }

            let decoded_pieces: Vec<Vec<u8>> = pieces
                .iter()
                .zip(types.iter())
                .map(|(p, &t)| {
                    if t == TYPE_BYTE
                        || (p.len() == 6
                            && p.starts_with("<0x")
                            && p.ends_with('>')
                            && u8::from_str_radix(&p[3..5], 16).is_ok())
                    {
                        vec![u8::from_str_radix(&p[3..5], 16).unwrap()]
                    } else if t == TYPE_CONTROL || t == TYPE_USER_DEFINED || t == TYPE_UNKNOWN {
                        p.as_bytes().to_vec()
                    } else {
                        p.replace('\u{2581}', " ").into_bytes()
                    }
                })
                .collect();

            let n_vocab = pieces.len();
            let user_specials = collect_user_specials(&pieces, &types);
            Ok(Tokenizer {
                pieces,
                ids,
                special_index: SpecialIndex::build(&specials),
                specials,
                user_index: SpecialIndex::build(&user_specials),
                user_specials,
                bos,
                eos,
                add_bos,
                add_eos,
                add_sep,
                sep: special_id(g, "tokenizer.ggml.seperator_token_id", None, n_vocab),
                cls: special_id(g, "tokenizer.ggml.cls_token_id", None, n_vocab),
                mask: special_id(g, "tokenizer.ggml.mask_token_id", None, n_vocab),
                pre,
                n_vocab,
                algo: TokenizerAlgo::Spm(SpmData {
                    scores,
                    token_types: types,
                    byte_tokens,
                    decoded_pieces,
                    unk_id,
                    add_space_prefix,
                    remove_extra_whitespaces,
                }),
            })
        }
    }

    /// Serializes a WordPiece or unigram tokenizer into the compact vocabulary fixture
    /// format the CI tests load (`tests/fixtures/tokenize-vocab-*.bin`).
    ///
    /// The fixture holds the vocabulary and the rules the loader derived from the file
    /// — pieces, token types, unigram scores, the precompiled charsmap and the special
    /// token ids and flags — so a test can build the exact tokenizer a model file gives
    /// without carrying the model file. `tests/tokenize_encoders.rs` checks that the
    /// fixture matches llama.cpp's ids and, when the file is present, the file itself.
    ///
    /// The generator is `crates/peregrine-bench/examples/vocab_dump.rs`:
    ///
    /// ```text
    /// cargo run --release -p peregrine-bench --example vocab_dump -- model.gguf out.bin
    /// ```
    pub fn export_vocab(&self) -> Result<Vec<u8>, String> {
        let (kind, types, scores, charsmap, add_space_prefix, remove_extra_whitespaces) =
            match &self.algo {
                TokenizerAlgo::Wpm(w) => (
                    1u8,
                    w.token_types.clone(),
                    Vec::new(),
                    Vec::new(),
                    false,
                    false,
                ),
                TokenizerAlgo::Ugm(u) => (
                    2u8,
                    u.token_types.clone(),
                    u.scores.clone(),
                    u.charsmap
                        .as_ref()
                        .map(|c| {
                            let mut blob =
                                Vec::with_capacity(4 + c.xcda.len() * 4 + c.replacements.len());
                            blob.extend_from_slice(&((c.xcda.len() * 4) as u32).to_le_bytes());
                            for v in &c.xcda {
                                blob.extend_from_slice(&v.to_le_bytes());
                            }
                            blob.extend_from_slice(&c.replacements);
                            blob
                        })
                        .unwrap_or_default(),
                    u.add_space_prefix,
                    u.remove_extra_whitespaces,
                ),
                _ => {
                    return Err(
                        "only WordPiece and SentencePiece-unigram vocabularies are exported".into(),
                    )
                }
            };
        let mut flags = 0u8;
        for (i, bit) in [
            self.add_bos,
            self.add_eos,
            self.add_sep,
            add_space_prefix,
            remove_extra_whitespaces,
        ]
        .into_iter()
        .enumerate()
        {
            if bit {
                flags |= 1 << i;
            }
        }
        let (unk, lowercase, strip_accents) = match &self.algo {
            TokenizerAlgo::Wpm(w) => (w.unk_id, w.lowercase, w.strip_accents),
            TokenizerAlgo::Ugm(u) => (u.unk_id, false, false),
            _ => (u32::MAX, false, false),
        };
        flags |= u8::from(lowercase) << 5;
        flags |= u8::from(strip_accents) << 6;
        let opt = |v: u32| if v == u32::MAX { -1i32 } else { v as i32 };
        let mut out = Vec::with_capacity(42 + charsmap.len() + self.pieces.len() * 9);
        out.extend_from_slice(b"PGVOC1\0\0");
        out.push(kind);
        out.push(flags);
        for v in [
            opt(self.bos),
            opt(self.eos),
            opt(unk),
            opt(self.sep.unwrap_or(u32::MAX)),
            opt(self.cls.unwrap_or(u32::MAX)),
            opt(self.mask.unwrap_or(u32::MAX)),
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.pieces.len() as u32).to_le_bytes());
        out.extend_from_slice(&(charsmap.len() as u32).to_le_bytes());
        out.extend_from_slice(&charsmap);
        for (i, p) in self.pieces.iter().enumerate() {
            let len = u8::try_from(p.len())
                .map_err(|_| format!("token {i} is {} bytes, too long for the fixture", p.len()))?;
            out.push(len);
            out.extend_from_slice(p.as_bytes());
        }
        if kind == 2 {
            for i in 0..self.pieces.len() {
                out.extend_from_slice(&scores.get(i).copied().unwrap_or(0.0).to_le_bytes());
            }
        }
        // Token types are written sparsely: a vocabulary is almost all normal tokens,
        // and the 250k-token XLM-R one carries a few control and unused ids.
        let odd: Vec<(u32, i32)> = types
            .iter()
            .enumerate()
            .filter(|(_, &t)| t != TYPE_NORMAL)
            .map(|(i, &t)| (i as u32, t))
            .collect();
        out.extend_from_slice(&(odd.len() as u32).to_le_bytes());
        for (i, t) in odd {
            out.extend_from_slice(&i.to_le_bytes());
            out.push(t as u8);
        }
        Ok(out)
    }

    /// Loads a vocabulary fixture written by [`Self::export_vocab`].
    ///
    /// This is the CI path for the encoder tokenizers: the extracted vocabularies are
    /// large enough that committing the model files is not an option and the goldens
    /// are only meaningful against the real vocabulary, so the fixture carries exactly
    /// what `from_gguf` read out of the file.
    pub fn from_vocab_dump(bytes: &[u8]) -> Result<Tokenizer, String> {
        fn take<'a>(b: &'a [u8], p: &mut usize, n: usize) -> Result<&'a [u8], String> {
            let end = p.checked_add(n).ok_or("vocab fixture: length overflow")?;
            let s = b.get(*p..end).ok_or("vocab fixture: truncated")?;
            *p = end;
            Ok(s)
        }
        if bytes.len() < 8 || &bytes[..6] != b"PGVOC1" {
            return Err("vocab fixture: bad magic".into());
        }
        let mut p = 8usize;
        let kind = take(bytes, &mut p, 1)?[0];
        let flags = take(bytes, &mut p, 1)?[0];
        let raw = take(bytes, &mut p, 24)?;
        let field = |i: usize| {
            let v = i32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            (v >= 0).then_some(v as u32)
        };
        let (bos, eos, unk, sep, cls, mask) =
            (field(0), field(1), field(2), field(3), field(4), field(5));
        let n_pieces = u32::from_le_bytes(take(bytes, &mut p, 4)?.try_into().unwrap()) as usize;
        let charsmap_len = u32::from_le_bytes(take(bytes, &mut p, 4)?.try_into().unwrap()) as usize;
        let charsmap = take(bytes, &mut p, charsmap_len)?.to_vec();
        let cfg = VocabConfig {
            bos,
            eos,
            unk,
            sep,
            cls,
            mask,
            add_bos: flags & 1 != 0,
            add_eos: flags & 2 != 0,
            add_sep: flags & 4 != 0,
        };
        let mut pieces = Vec::with_capacity(n_pieces);
        for _ in 0..n_pieces {
            let len = take(bytes, &mut p, 1)?[0] as usize;
            let piece = take(bytes, &mut p, len)?;
            pieces.push(String::from_utf8(piece.to_vec()).map_err(|_| "vocab fixture: bad UTF-8")?);
        }
        let mut scores = vec![0.0f32; n_pieces];
        if kind == 2 {
            for score in scores.iter_mut() {
                *score = f32::from_le_bytes(take(bytes, &mut p, 4)?.try_into().unwrap());
            }
        }
        let mut types = vec![TYPE_NORMAL; n_pieces];
        let n_odd = u32::from_le_bytes(take(bytes, &mut p, 4)?.try_into().unwrap()) as usize;
        for _ in 0..n_odd {
            let id = u32::from_le_bytes(take(bytes, &mut p, 4)?.try_into().unwrap()) as usize;
            let t = take(bytes, &mut p, 1)?[0] as i32;
            if id < n_pieces {
                types[id] = t;
            }
        }
        let mut ids = HashMap::with_capacity(pieces.len());
        for (i, piece) in pieces.iter().enumerate() {
            ids.entry(piece.clone()).or_insert(i as u32);
        }
        match kind {
            1 => Ok(build_wpm(
                pieces,
                ids,
                types,
                cfg,
                flags & 0x20 != 0,
                flags & 0x40 != 0,
            )),
            2 => {
                let charsmap = if charsmap.is_empty() {
                    None
                } else {
                    Some(Charsmap::parse(&charsmap)?)
                };
                Ok(build_ugm(
                    pieces,
                    ids,
                    types,
                    scores,
                    cfg,
                    UgmRules {
                        charsmap,
                        add_space_prefix: flags & 8 != 0,
                        remove_extra_whitespaces: flags & 16 != 0,
                    },
                ))
            }
            other => Err(format!("vocab fixture: unknown kind {other}")),
        }
    }

    /// Test helper: build an SPM tokenizer directly from raw parts without a GGUF file.
    pub fn from_spm_parts(
        pieces: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<i32>,
        bos: u32,
        eos: u32,
        unk_id: u32,
        add_space_prefix: bool,
    ) -> Tokenizer {
        let mut ids = HashMap::with_capacity(pieces.len());
        for (i, p) in pieces.iter().enumerate() {
            ids.entry(p.clone()).or_insert(i as u32);
        }
        let mut specials: Vec<(String, u32)> = pieces
            .iter()
            .zip(token_types.iter())
            .enumerate()
            .filter(|(_, (_, &t))| t == TYPE_CONTROL || t == TYPE_USER_DEFINED || t == TYPE_UNKNOWN)
            .map(|(i, (p, _))| (p.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let user_specials = collect_user_specials(&pieces, &token_types);

        let mut byte_tokens = [unk_id; 256];
        for b in 0..=255u8 {
            let hex_token = format!("<0x{:02X}>", b);
            if let Some(&id) = ids.get(&hex_token) {
                byte_tokens[b as usize] = id;
            } else if let Some(&id) = ids.get((b as char).to_string().as_str()) {
                byte_tokens[b as usize] = id;
            }
        }

        let decoded_pieces: Vec<Vec<u8>> = pieces
            .iter()
            .zip(token_types.iter())
            .map(|(p, &t)| {
                if t == TYPE_BYTE
                    || (p.len() == 6
                        && p.starts_with("<0x")
                        && p.ends_with('>')
                        && u8::from_str_radix(&p[3..5], 16).is_ok())
                {
                    vec![u8::from_str_radix(&p[3..5], 16).unwrap()]
                } else if t == TYPE_CONTROL || t == TYPE_USER_DEFINED || t == TYPE_UNKNOWN {
                    p.as_bytes().to_vec()
                } else {
                    p.replace('\u{2581}', " ").into_bytes()
                }
            })
            .collect();

        let n_vocab = pieces.len();
        Tokenizer {
            pieces,
            ids,
            special_index: SpecialIndex::build(&specials),
            specials,
            user_index: SpecialIndex::build(&user_specials),
            user_specials,
            bos,
            eos,
            add_bos: true,
            add_eos: false,
            add_sep: false,
            sep: None,
            cls: None,
            mask: None,
            pre: PreType::Default,
            n_vocab,
            algo: TokenizerAlgo::Spm(SpmData {
                scores,
                token_types,
                byte_tokens,
                decoded_pieces,
                unk_id,
                add_space_prefix,
                remove_extra_whitespaces: false,
            }),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    /// Encodes one text with llama.cpp's `parse_special = false`: the vocabulary's
    /// control-token spellings (`[INST]`, `<s>`) are ordinary text, not tokens.
    /// User-defined tokens are still matched, as `tokenizer_st_partition` does.
    ///
    /// [`Self::encode`] is this with `parse_special = true`, which is what the
    /// server's `/tokenize` does by default and what a chat prompt needs.
    pub fn encode_unparsed(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        match &self.algo {
            TokenizerAlgo::Bpe(bpe) => self.encode_bpe_into(text, bpe, false, &mut out),
            TokenizerAlgo::Spm(spm) => self.encode_spm_into(text, spm, false, &mut out),
            TokenizerAlgo::Wpm(wpm) => self.encode_wpm(text, wpm, false, &mut out),
            TokenizerAlgo::Ugm(ugm) => self.encode_ugm(text, ugm, false, &mut out),
        }
        out
    }

    /// Encodes a chat prompt into tokens, ensuring that the token stream has exactly
    /// one BOS token if `add_bos` is set.
    ///
    /// SentencePiece models usually add `<s>` / `<bos>` through the tokenizer
    /// (`add_bos`), and some templates also write it. If the encoded text already
    /// begins with `self.bos`, no duplicate is added; if `add_bos` is true and the
    /// stream does not begin with `self.bos`, `self.bos` is prepended.
    ///
    /// For an encoder vocabulary (WordPiece, SentencePiece unigram) this is
    /// [`Self::encode_special`]: there is no chat prompt and no template of the
    /// engine's own, so the file's rules for an input's special tokens are the
    /// prompt path.
    pub fn encode_prompt(&self, text: &str) -> Vec<u32> {
        if matches!(self.algo, TokenizerAlgo::Wpm(_) | TokenizerAlgo::Ugm(_)) {
            return self.encode_special(text);
        }
        let mut toks = self.encode(text);
        if self.add_bos && toks.first().copied() != Some(self.bos) {
            toks.insert(0, self.bos);
        }
        toks
    }

    /// Encodes one text the way llama.cpp does for an encoder input: with the
    /// file's special tokens around it (`add_special = true`).
    ///
    /// * WordPiece: `[CLS] text [SEP]` — llama.cpp's `LLAMA_VOCAB_TYPE_WPM` branch
    ///   pushes the BOS and SEP ids whatever `add_eos`/`add_sep` say, but only when
    ///   the caller asks for special tokens, which the tools gate on `add_bos`. A
    ///   BERT file whose `add_bos_token` is false therefore gets nothing.
    /// * SentencePiece unigram: `<s>` when `add_bos`, `</s>` when `add_eos`.
    /// * SentencePiece BPE and byte-level BPE (the generator tokenizers, which is what
    ///   a causal embedding model such as Qwen3-Embedding uses): `bos` when `add_bos`,
    ///   then the text, then `eos` when `add_eos` — llama.cpp's `add_special` prologue
    ///   and epilogue around the tokenizer itself. The ids are pushed unconditionally,
    ///   as llama.cpp pushes them: the text's own leading `bos` is a token, not a reason
    ///   to skip the prologue. (Chat prompts keep [`Self::encode_prompt`], which avoids
    ///   the duplicate a template-written BOS would otherwise cause.)
    pub fn encode_special(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        match &self.algo {
            TokenizerAlgo::Wpm(wpm) => {
                if self.add_bos {
                    if let Some(cls) = self.cls {
                        out.push(cls);
                    }
                    self.encode_wpm(text, wpm, true, &mut out);
                    if let Some(sep) = self.sep {
                        out.push(sep);
                    }
                } else {
                    self.encode_wpm(text, wpm, true, &mut out);
                }
            }
            TokenizerAlgo::Ugm(ugm) => {
                if self.add_bos && self.bos != u32::MAX {
                    out.push(self.bos);
                }
                self.encode_ugm(text, ugm, true, &mut out);
                if self.add_eos && self.eos != u32::MAX {
                    out.push(self.eos);
                }
            }
            _ => {
                let mut out = self.encode(text);
                if self.add_bos && self.bos != u32::MAX {
                    out.insert(0, self.bos);
                }
                if self.add_eos && self.eos != u32::MAX {
                    out.push(self.eos);
                }
                return out;
            }
        }
        out
    }

    /// Encodes a (query, document) pair the way llama.cpp's rerank path
    /// (`tools/server/server-common.cpp`, `format_prompt_rerank`) builds it when the
    /// file carries no `rerank` chat template:
    ///
    /// ```text
    /// [bos]  query  [eos]  [sep]  document  [eos]
    /// ```
    ///
    /// each part present only when the file's `add_bos` / `add_eos` / `add_sep` says
    /// so, and `eos` falling back to `sep` when the file has no EOS id. For a BERT
    /// reranker that is `[CLS] q [SEP] d [SEP]` (the file sets `add_eos`); for an
    /// XLM-R one, where EOS and SEP are both `</s>` and `add_sep` is set,
    /// `<s> q </s> </s> d </s>`.
    ///
    /// Both texts are tokenized with `parse_special = false`, as that path does: only
    /// user-defined tokens are still matched, control tokens are not. That is WordPiece,
    /// unigram and the byte-level BPE an encoder reranker carries (`modern-bert`, S4.7b).
    /// The generator tokenizers fall back to [`Self::encode`], which does match their
    /// special tokens — no generative reranker reaches this path (they are scored through
    /// the file's own `rerank` chat template instead).
    pub fn encode_pair(&self, query: &str, document: &str) -> Vec<u32> {
        let eos = if self.eos != u32::MAX {
            Some(self.eos)
        } else {
            self.sep
        };
        let mut query_tokens = Vec::new();
        let mut doc_tokens = Vec::new();
        match &self.algo {
            TokenizerAlgo::Wpm(wpm) => {
                self.encode_wpm(query, wpm, false, &mut query_tokens);
                self.encode_wpm(document, wpm, false, &mut doc_tokens);
            }
            TokenizerAlgo::Ugm(ugm) => {
                self.encode_ugm(query, ugm, false, &mut query_tokens);
                self.encode_ugm(document, ugm, false, &mut doc_tokens);
            }
            TokenizerAlgo::Bpe(bpe) => {
                // `modern-bert`'s GPT-2 vocabulary (S4.7b): `[SEP]` in a document is
                // four ordinary words here, as it is for llama.cpp's pair path.
                self.encode_bpe_into(query, bpe, false, &mut query_tokens);
                self.encode_bpe_into(document, bpe, false, &mut doc_tokens);
            }
            _ => {
                self.encode_into(query, &mut query_tokens);
                self.encode_into(document, &mut doc_tokens);
            }
        }
        let mut out = Vec::with_capacity(query_tokens.len() + doc_tokens.len() + 3);
        if self.add_bos && self.bos != u32::MAX {
            out.push(self.bos);
        }
        out.extend_from_slice(&query_tokens);
        if self.add_eos {
            out.extend(eos);
        }
        if self.add_sep {
            out.extend(self.sep);
        }
        out.extend_from_slice(&doc_tokens);
        if self.add_eos {
            out.extend(eos);
        }
        out
    }

    pub fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        match &self.algo {
            TokenizerAlgo::Bpe(bpe) => self.encode_bpe_into(text, bpe, true, out),
            TokenizerAlgo::Spm(spm) => self.encode_spm_into(text, spm, true, out),
            TokenizerAlgo::Wpm(wpm) => self.encode_wpm(text, wpm, true, out),
            TokenizerAlgo::Ugm(ugm) => self.encode_ugm(text, ugm, true, out),
        }
    }

    /// Byte-level BPE over the whole input. `parse_special` decides which specials
    /// are pulled out atomically: all of them, or only the user-defined ones.
    fn encode_bpe_into(&self, text: &str, bpe: &BpeData, parse_special: bool, out: &mut Vec<u32>) {
        let (specials, index) = if parse_special {
            (&self.specials, &self.special_index)
        } else {
            (&self.user_specials, &self.user_index)
        };
        let mut p = 0usize;
        while p < text.len() {
            // Atomic special tokens first.
            if let Some((len, id)) = index.hit(text, p, specials) {
                out.push(id);
                p += len;
                continue;
            }
            // Otherwise accumulate a plain run up to the next special, and split it.
            let q = next_special_in(text, p, index, specials);
            self.encode_plain_bpe(&text[p..q], bpe, out);
            p = q;
        }
    }

    fn encode_spm_into(&self, text: &str, spm: &SpmData, parse_special: bool, out: &mut Vec<u32>) {
        let (specials, index) = if parse_special {
            (&self.specials, &self.special_index)
        } else {
            (&self.user_specials, &self.user_index)
        };
        let mut p = 0usize;
        let mut is_prev_special = true;
        let mut scratch = SpmScratch::default();
        let mut buf = String::new();
        while p < text.len() {
            if let Some((len, id)) = index.hit(text, p, specials) {
                out.push(id);
                p += len;
                is_prev_special = true;
                continue;
            }
            let q = next_special_in(text, p, index, specials);
            self.encode_fragment_spm(
                &text[p..q],
                spm,
                is_prev_special,
                &mut buf,
                &mut scratch,
                out,
            );
            is_prev_special = false;
            p = q;
        }
    }

    /// `llama_vocab::impl::tokenizer_st_partition`: every special token's text is
    /// pulled out of the raw input, longest token first. Running each special in turn
    /// over what is left (rather than scanning left to right) is what decides
    /// overlapping matches — `ab` and `bc` in `abc` split as `a` + `bc`, not `ab` + `c`.
    fn partition<'a>(&self, text: &'a str, parse_special: bool, frags: &mut Vec<Fragment<'a>>) {
        frags.clear();
        frags.push(Fragment::Raw(text));
        for (special, id) in &self.specials {
            if special.is_empty() {
                continue;
            }
            if !parse_special && !self.is_user_defined(*id) {
                continue;
            }
            let mut i = 0;
            while i < frags.len() {
                let split = match frags[i] {
                    Fragment::Raw(s) => s.find(special.as_str()).map(|p| (p, s)),
                    Fragment::Token(_) => None,
                };
                if let Some((p, s)) = split {
                    frags[i] = Fragment::Raw(&s[..p]);
                    frags.insert(i + 1, Fragment::Token(*id));
                    frags.insert(i + 2, Fragment::Raw(&s[p + special.len()..]));
                    // Keep scanning from the remainder, so `[MASK][MASK]` finds both.
                    i += 2;
                } else {
                    i += 1;
                }
            }
        }
    }

    /// The token type of `id`, for the two encoder vocabularies.
    fn token_type(&self, id: u32) -> i32 {
        let ts = match &self.algo {
            TokenizerAlgo::Wpm(w) => &w.token_types,
            TokenizerAlgo::Ugm(u) => &u.token_types,
            TokenizerAlgo::Spm(s) => &s.token_types,
            TokenizerAlgo::Bpe(_) => return TYPE_NORMAL,
        };
        ts.get(id as usize).copied().unwrap_or(TYPE_NORMAL)
    }

    fn is_user_defined(&self, id: u32) -> bool {
        self.token_type(id) == TYPE_USER_DEFINED
    }

    /// WordPiece over the whole input: special tokens are matched first, then each
    /// remaining fragment is preprocessed and matched word by word.
    fn encode_wpm(&self, text: &str, wpm: &WpmData, parse_special: bool, out: &mut Vec<u32>) {
        let mut frags = Vec::new();
        self.partition(text, parse_special, &mut frags);
        let mut buf = String::with_capacity(text.len() + 16);
        let mut words: Vec<(usize, usize)> = Vec::new();
        for frag in &frags {
            match frag {
                Fragment::Token(id) => out.push(*id),
                Fragment::Raw(s) => {
                    words.clear();
                    buf.clear();
                    wpm_preprocess(s, wpm, &mut buf, &mut words);
                    for &(start, end) in &words {
                        self.wpm_word(&buf[start..end], wpm, out);
                    }
                }
            }
        }
    }

    /// One preprocessed word (its bytes already carry the phantom `▁`): the longest
    /// vocabulary token at each position, and one unknown token for the whole word if
    /// any position has no match.
    fn wpm_word(&self, word: &str, wpm: &WpmData, out: &mut Vec<u32>) {
        let bytes = word.as_bytes();
        let started = out.len();
        let mut pos = 0usize;
        while pos < bytes.len() {
            let mut node = 0u32;
            let mut best: Option<(u32, usize)> = None;
            for (i, &b) in bytes[pos..].iter().enumerate() {
                match wpm.trie.child(node, b) {
                    Some(n) => {
                        node = n;
                        if let Some(t) = wpm.trie.token(node) {
                            best = Some((t, pos + i + 1));
                        }
                    }
                    None => break,
                }
            }
            match best {
                Some((id, next)) => {
                    out.push(id);
                    pos = next;
                }
                None => {
                    out.truncate(started);
                    if wpm.unk_id != u32::MAX {
                        out.push(wpm.unk_id);
                    }
                    return;
                }
            }
        }
    }

    /// SentencePiece unigram (llama.cpp's `llm_tokenizer_ugm_session`): normalize each
    /// literal fragment, then find the highest-scoring segmentation of it.
    fn encode_ugm(&self, text: &str, ugm: &UgmData, parse_special: bool, out: &mut Vec<u32>) {
        let mut frags = Vec::new();
        self.partition(text, parse_special, &mut frags);
        let mut normalized: Vec<u8> = Vec::with_capacity(text.len() + 4);
        for frag in &frags {
            match frag {
                Fragment::Token(id) => out.push(*id),
                Fragment::Raw(s) => {
                    normalized.clear();
                    self.ugm_normalize(s, ugm, &mut normalized);
                    self.ugm_viterbi(&normalized, ugm, out);
                }
            }
        }
    }

    /// `llm_tokenizer_ugm_session::normalize`: rewrite the text through the file's
    /// charsmap (one prefix at a time), turn spaces into `▁`, and apply the file's
    /// leading-space and whitespace-collapsing rules.
    fn ugm_normalize(&self, input: &str, ugm: &UgmData, out: &mut Vec<u8>) {
        const SPACE: &[u8] = b"\xE2\x96\x81";
        let shall_prepend_space = ugm.add_space_prefix;
        // `treat_whitespace_as_suffix` is never set by any converter, so the suffix
        // half of llama.cpp's rule cannot fire.
        let shall_merge_spaces = ugm.remove_extra_whitespaces;
        let mut is_space_prepended = false;
        let mut processing_non_ws = false;
        let input = input.as_bytes();
        let mut offset = 0usize;
        while offset < input.len() {
            let (normalized, consumed) = self.ugm_normalize_prefix(input, offset, ugm);
            for &c in normalized {
                if c != b' ' {
                    if !processing_non_ws {
                        processing_non_ws = true;
                        if (shall_prepend_space && !is_space_prepended) || shall_merge_spaces {
                            out.extend_from_slice(SPACE);
                            is_space_prepended = true;
                        }
                    }
                    out.push(c);
                } else {
                    processing_non_ws = false;
                    if !shall_merge_spaces {
                        out.extend_from_slice(SPACE);
                    }
                }
            }
            offset += consumed;
        }
    }

    /// `llm_tokenizer_ugm_session::normalize_prefix`: the replacement for the longest
    /// charsmap prefix of `input[offset..]`, or the code point itself, or U+FFFD for an
    /// invalid byte. Always consumes at least one byte.
    fn ugm_normalize_prefix<'a>(
        &'a self,
        input: &'a [u8],
        offset: usize,
        ugm: &'a UgmData,
    ) -> (&'a [u8], usize) {
        const REPLACEMENT: &[u8] = b"\xEF\xBF\xBD";
        if offset >= input.len() {
            return (&input[input.len()..], 0);
        }
        // A user-defined token's text is handed through untouched, so the Viterbi
        // search can match it as one piece.
        let mut node = 0u32;
        let mut best: Option<(usize, usize)> = None;
        for (i, &b) in input[offset..].iter().enumerate() {
            match ugm.user_trie.child(node, b) {
                Some(n) => {
                    node = n;
                    if ugm.user_trie.token(node).is_some() {
                        best = Some((i + 1, 0));
                    }
                }
                None => break,
            }
        }
        if let Some((len, _)) = best {
            return (&input[offset..offset + len], len);
        }

        // The XCDA walk. Out-of-range node indices only happen in a corrupt charsmap;
        // llama.cpp throws there, this stops matching and hands the prefix through
        // unchanged, which keeps `encode` total.
        if let Some(charsmap) = &ugm.charsmap {
            let mut longest: Option<(usize, usize)> = None;
            if let Some(mut node) = charsmap.base(0) {
                for (i, &c) in input[offset..].iter().enumerate() {
                    if c == 0 {
                        break;
                    }
                    node ^= c as u32;
                    if charsmap.lcheck(node) != Some(c as u32) {
                        break;
                    }
                    let leaf = charsmap.leaf(node).unwrap_or(false);
                    let Some(base) = charsmap.base(node) else {
                        break;
                    };
                    node ^= base;
                    if leaf {
                        if let Some(v) = charsmap.value(node) {
                            longest = Some((i + 1, v as usize));
                        }
                    }
                }
            }
            if let Some((len, start)) = longest {
                if start < charsmap.replacements.len() {
                    let rest = &charsmap.replacements[start..];
                    let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
                    return (&rest[..end], len);
                }
            }
        }

        // A valid UTF-8 code point is passed through unchanged; an invalid byte becomes
        // U+FFFD and is consumed alone.
        let max = (input.len() - offset).min(4);
        for n in 1..=max {
            if let Ok(s) = std::str::from_utf8(&input[offset..offset + n]) {
                return (s.as_bytes(), n);
            }
        }
        (REPLACEMENT, 1)
    }

    /// The Viterbi search of `llm_tokenizer_ugm_session::tokenize`: one unknown token
    /// per code point the vocabulary cannot cover (with a flat penalty below the
    /// lowest normal-token score), consecutive unknowns merged, scores in `f64`.
    fn ugm_viterbi(&self, normalized: &[u8], ugm: &UgmData, out: &mut Vec<u32>) {
        if normalized.is_empty() {
            return;
        }
        let n = normalized.len();
        #[derive(Clone, Copy)]
        struct Best {
            token: u32,
            offset: usize,
            score: f64,
        }
        let mut best = vec![
            Best {
                token: ugm.unk_id,
                offset: 0,
                score: -f64::MAX,
            };
            n + 1
        ];
        best[0].score = 0.0;
        let unk = ugm.unk_id;
        let mut offset = 0usize;
        while offset < n {
            let units = (unicode_len_utf8(normalized[offset])).min(n - offset);
            let mut single_cp_found = false;
            let current = best[offset];
            let mut node = ugm.trie.child(0, normalized[offset]);
            let mut prefix = offset + 1;
            while prefix <= n {
                let Some(n0) = node else { break };
                if let Some(token) = ugm.trie.token(n0) {
                    if prefix - offset == units {
                        single_cp_found = true;
                    }
                    let score = if self.token_type(token) == TYPE_USER_DEFINED {
                        0.0
                    } else {
                        ugm.scores.get(token as usize).copied().unwrap_or(0.0) as f64
                    };
                    let challenger = current.score + score;
                    if challenger > best[prefix].score {
                        best[prefix] = Best {
                            token,
                            offset,
                            score: challenger,
                        };
                    }
                }
                node = if prefix < n {
                    ugm.trie.child(n0, normalized[prefix])
                } else {
                    None
                };
                prefix += 1;
            }
            if !single_cp_found {
                let challenger = current.score + ugm.unknown_token_score as f64;
                let next = offset + units;
                if challenger > best[next].score {
                    best[next] = Best {
                        token: unk,
                        offset,
                        score: challenger,
                    };
                }
            }
            offset += units;
        }

        let started = out.len();
        let mut is_prev_unknown = false;
        let mut at = n;
        loop {
            let t = best[at];
            let is_unknown = t.token == unk;
            if !(is_prev_unknown && is_unknown) {
                out.push(t.token);
            }
            if t.offset == 0 {
                break;
            }
            is_prev_unknown = is_unknown;
            at = t.offset;
        }
        out[started..].reverse();
    }

    fn encode_fragment_spm(
        &self,
        raw_piece: &str,
        spm: &SpmData,
        is_prev_special: bool,
        buf: &mut String,
        scratch: &mut SpmScratch,
        out: &mut Vec<u32>,
    ) {
        if raw_piece.is_empty() {
            return;
        }
        buf.clear();
        if spm.add_space_prefix && is_prev_special {
            buf.push('\u{2581}');
        }
        if spm.remove_extra_whitespaces {
            let mut in_ws = false;
            for ch in raw_piece.chars() {
                if ch == ' ' {
                    if !in_ws {
                        buf.push('\u{2581}');
                        in_ws = true;
                    }
                } else {
                    in_ws = false;
                    buf.push(ch);
                }
            }
        } else {
            for ch in raw_piece.chars() {
                if ch == ' ' {
                    buf.push('\u{2581}');
                } else {
                    buf.push(ch);
                }
            }
        }
        self.tokenize_spm_text(buf, spm, scratch, out);
    }

    fn tokenize_spm_text(
        &self,
        text: &str,
        spm: &SpmData,
        scratch: &mut SpmScratch,
        out: &mut Vec<u32>,
    ) {
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return;
        }
        scratch.symbols.clear();
        scratch.work_queue.clear();
        let mut rev_merge: HashMap<&str, (u32, u32)> = HashMap::with_capacity(32);

        let mut offs = 0usize;
        let mut idx = 0i32;
        while offs < bytes.len() {
            let len = unicode_len_utf8(bytes[offs]).min(bytes.len() - offs);
            let prev = idx - 1;
            let next = if offs + len == bytes.len() {
                -1
            } else {
                idx + 1
            };
            scratch.symbols.push(Symbol {
                start: offs as u32,
                len: len as u32,
                prev,
                next,
            });
            offs += len;
            idx += 1;
        }

        for i in 1..scratch.symbols.len() {
            try_add_bigram(
                (i - 1) as i32,
                i as i32,
                text,
                &self.ids,
                &spm.scores,
                scratch,
                &mut rev_merge,
            );
        }

        while let Some(bigram) = scratch.work_queue.pop() {
            let left_idx = bigram.left as usize;
            let right_idx = bigram.right as usize;
            let left_len = scratch.symbols[left_idx].len;
            let right_len = scratch.symbols[right_idx].len;

            if left_len == 0 || right_len == 0 || left_len + right_len != bigram.size {
                continue;
            }

            scratch.symbols[left_idx].len += right_len;
            scratch.symbols[right_idx].len = 0;

            let next = scratch.symbols[right_idx].next;
            scratch.symbols[left_idx].next = next;
            if next >= 0 {
                scratch.symbols[next as usize].prev = bigram.left;
            }

            let prev = scratch.symbols[left_idx].prev;
            try_add_bigram(
                prev,
                bigram.left,
                text,
                &self.ids,
                &spm.scores,
                scratch,
                &mut rev_merge,
            );
            try_add_bigram(
                bigram.left,
                next,
                text,
                &self.ids,
                &spm.scores,
                scratch,
                &mut rev_merge,
            );
        }

        scratch.todo.clear();
        let mut curr = if scratch.symbols.is_empty() { -1 } else { 0 };
        while curr != -1 {
            let sym = scratch.symbols[curr as usize];
            scratch.todo.push((sym.start, sym.len));
            while let Some((start, len)) = scratch.todo.pop() {
                let slice = &text[start as usize..(start + len) as usize];
                if let Some(&id) = self.ids.get(slice) {
                    out.push(id);
                    continue;
                }
                if let Some(&(left_len, right_len)) = rev_merge.get(slice) {
                    scratch.todo.push((start + left_len, right_len));
                    scratch.todo.push((start, left_len));
                    continue;
                }
                for &b in slice.as_bytes() {
                    out.push(spm.byte_tokens[b as usize]);
                }
            }
            curr = scratch.symbols[curr as usize].next;
        }
    }

    fn encode_plain_bpe(&self, text: &str, bpe: &BpeData, out: &mut Vec<u32>) {
        match self.pre {
            PreType::Gpt2 => {
                let mut p = 0usize;
                while p < text.len() {
                    let take = match_piece(&text[p..]).unwrap_or_else(|| {
                        text[p..].chars().next().map(|c| c.len_utf8()).unwrap_or(1)
                    });
                    self.bpe(&text[p..p + take], bpe, out);
                    p += take;
                }
            }
            PreType::SmolLm => {
                for chunk in split_numbers(text) {
                    let mut p = 0usize;
                    while p < chunk.len() {
                        let take = match_piece(&chunk[p..]).unwrap_or_else(|| {
                            // No alternative matched: emit one character, as llama.cpp's
                            // splitter does for text its patterns do not cover.
                            chunk[p..].chars().next().map(|c| c.len_utf8()).unwrap_or(1)
                        });
                        self.bpe(&chunk[p..p + take], bpe, out);
                        p += take;
                    }
                }
            }
            PreType::Llama3 | PreType::Dbrx => {
                for piece in split_llama3(text, 3) {
                    self.bpe(piece, bpe, out);
                }
            }
            PreType::Qwen2 => {
                for piece in split_llama3(text, 1) {
                    self.bpe(piece, bpe, out);
                }
            }
            PreType::Qwen35 => {
                for piece in split_with(text, 1, true) {
                    self.bpe(piece, bpe, out);
                }
            }
            PreType::Gpt4o => {
                for piece in split_gpt4o(text) {
                    self.bpe(piece, bpe, out);
                }
            }
            PreType::Tekken => {
                for piece in split_tekken(text) {
                    self.bpe(piece, bpe, out);
                }
            }
            PreType::TekkenNum3 => {
                for piece in split_tekken_num3(text) {
                    self.bpe(piece, bpe, out);
                }
            }
            PreType::Nemotron35Embed => {
                // `[` + U+200C and `<` + U+200C lose the joiner before splitting
                // (`src/llama-vocab.cpp` lines 652-680). Borrow the text as it is
                // when there is nothing to rewrite.
                if text.contains("[\u{200c}") || text.contains("<\u{200c}") {
                    let normalized = strip_zwnj_after_bracket(text);
                    for piece in split_tekken(&normalized) {
                        self.bpe(piece, bpe, out);
                    }
                } else {
                    for piece in split_tekken(text) {
                        self.bpe(piece, bpe, out);
                    }
                }
            }
            PreType::Default => {}
        }
    }

    /// Byte-level BPE over one pre-tokenized piece.
    fn bpe(&self, piece: &str, bpe: &BpeData, out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        // Into the byte alphabet: one symbol per *byte*, not per character.
        let mut syms: Vec<String> = piece
            .as_bytes()
            .iter()
            .map(|&b| bpe.byte_enc[b as usize].to_string())
            .collect();

        // `ignore_merges` (llama.cpp, Llama 3 and the Tekken family): a piece that is
        // itself a vocabulary entry is emitted whole. Llama 3's tiktoken vocabulary
        // holds tokens its merge list cannot reach, so running the merges would split
        // "Hello" into pieces the model never saw as a word; Tekken's vocabulary does
        // the same with multi-byte and rare tokens.
        if matches!(
            self.pre,
            PreType::Llama3 | PreType::Tekken | PreType::TekkenNum3 | PreType::Nemotron35Embed
        ) {
            let whole: String = syms.concat();
            if let Some(&id) = self.ids.get(&whole) {
                out.push(id);
                return;
            }
        }

        // Repeatedly merge the lowest-ranked adjacent pair. Pieces are short (a word
        // at most), so the quadratic scan is cheaper than maintaining a heap.
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&r) = bpe.ranks.get(&(syms[i].clone(), syms[i + 1].clone())) {
                    if best.is_none_or(|(_, br)| r < br) {
                        best = Some((i, r));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }

        for s in syms {
            match self.ids.get(&s) {
                Some(&id) => out.push(id),
                None => {
                    // Should be unreachable for a byte-level vocabulary, since every
                    // single-byte symbol is in it. Emit the bytes individually rather
                    // than dropping text, and never panic on user input.
                    for ch in s.chars() {
                        if let Some(&id) = self.ids.get(&ch.to_string()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Raw bytes of one token, before UTF-8 validation.
    ///
    /// For the SentencePiece-family vocabularies this is what llama.cpp's
    /// `token_to_piece` renders: the piece text with U+2581 turned back into a space,
    /// the literal text for control / user-defined / unknown tokens, and the single
    /// byte for a `<0xXX>` token.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        match &self.algo {
            TokenizerAlgo::Bpe(bpe) => match self.pieces.get(id as usize) {
                Some(p) => p
                    .chars()
                    .filter_map(|c| bpe.byte_dec.get(&c).copied())
                    .collect(),
                None => Vec::new(),
            },
            TokenizerAlgo::Spm(spm) => spm
                .decoded_pieces
                .get(id as usize)
                .cloned()
                .unwrap_or_default(),
            TokenizerAlgo::Wpm(_) | TokenizerAlgo::Ugm(_) => match self.pieces.get(id as usize) {
                Some(p) => {
                    let t = self.token_type(id);
                    if t == TYPE_BYTE
                        || (p.len() == 6
                            && p.starts_with("<0x")
                            && p.ends_with('>')
                            && u8::from_str_radix(&p[3..5], 16).is_ok())
                    {
                        vec![u8::from_str_radix(&p[3..5], 16).unwrap()]
                    } else if t == TYPE_CONTROL || t == TYPE_USER_DEFINED || t == TYPE_UNKNOWN {
                        p.as_bytes().to_vec()
                    } else {
                        p.replace('\u{2581}', " ").into_bytes()
                    }
                }
                None => Vec::new(),
            },
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        match &self.algo {
            TokenizerAlgo::Bpe(_) => {
                let mut bytes = Vec::new();
                for &id in ids {
                    bytes.extend_from_slice(&self.token_bytes(id));
                }
                String::from_utf8_lossy(&bytes).into_owned()
            }
            TokenizerAlgo::Spm(spm) => {
                let mut bytes = Vec::new();
                let mut remove_space = spm.add_space_prefix;
                for &id in ids {
                    let piece_bytes = self.token_bytes(id);
                    let slice = if remove_space && piece_bytes.first() == Some(&b' ') {
                        &piece_bytes[1..]
                    } else {
                        &piece_bytes[..]
                    };
                    remove_space = false;
                    bytes.extend_from_slice(slice);
                }
                String::from_utf8_lossy(&bytes).into_owned()
            }
            // `llama_vocab::impl::detokenize` with `remove_special = false`: the first
            // piece loses its leading space when the file adds one, and WordPiece then
            // runs the `clean_spaces` passes.
            TokenizerAlgo::Wpm(_) | TokenizerAlgo::Ugm(_) => {
                let mut bytes = Vec::new();
                let mut remove_space = self.add_space_prefix();
                for &id in ids {
                    let mut piece_bytes = self.token_bytes(id);
                    if remove_space && piece_bytes.first() == Some(&b' ') {
                        piece_bytes.remove(0);
                    }
                    remove_space = false;
                    bytes.extend_from_slice(&piece_bytes);
                }
                if matches!(self.algo, TokenizerAlgo::Wpm(_)) {
                    clean_spaces(&mut bytes);
                }
                String::from_utf8_lossy(&bytes).into_owned()
            }
        }
    }

    /// `tokenizer.ggml.add_space_prefix`, which the two encoder tokenizers read from
    /// their own records.
    fn add_space_prefix(&self) -> bool {
        match &self.algo {
            TokenizerAlgo::Spm(s) => s.add_space_prefix,
            TokenizerAlgo::Ugm(u) => u.add_space_prefix,
            _ => false,
        }
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.specials.iter().any(|(_, i)| *i == id)
    }

    /// Every control / user-defined / unknown token id, for a caller that tests the whole
    /// vocabulary (the constrained-decoding mask) in one pass.
    pub fn special_ids(&self) -> Vec<u32> {
        self.specials.iter().map(|&(_, id)| id).collect()
    }

    /// Every user-defined token id (GGUF token type 4): tokens a model's authors added
    /// with a text of their own, such as `<tool_call>` or `</think>` in Qwen's files.
    /// Unlike a control token, such a token is written into the reply as its text.
    pub fn user_defined_ids(&self) -> Vec<u32> {
        self.user_specials.iter().map(|&(_, id)| id).collect()
    }

    pub fn piece(&self, id: u32) -> &str {
        self.pieces
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Look up a token by its literal text, e.g. `"<|im_end|>"`.
    pub fn id_of(&self, literal: &str) -> Option<u32> {
        self.ids.get(literal).copied()
    }

    /// How many single-byte fallback tokens the vocabulary has. The WordPiece and
    /// unigram tokenizers have none — an uncovered code point becomes the unknown
    /// token, not a byte — so this is 0 for them.
    pub fn n_byte_tokens(&self) -> usize {
        match &self.algo {
            TokenizerAlgo::Bpe(bpe) => bpe.byte_dec.len(),
            TokenizerAlgo::Spm(_) => 256,
            TokenizerAlgo::Wpm(_) | TokenizerAlgo::Ugm(_) => 0,
        }
    }
}

#[inline]
fn unicode_len_utf8(b: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(b >> 4) as usize]
}

#[inline]
fn try_add_bigram<'a>(
    left: i32,
    right: i32,
    text: &'a str,
    ids: &HashMap<String, u32>,
    scores: &[f32],
    scratch: &mut SpmScratch,
    rev_merge: &mut HashMap<&'a str, (u32, u32)>,
) {
    if left < 0 || right < 0 {
        return;
    }
    let left_sym = scratch.symbols[left as usize];
    let right_sym = scratch.symbols[right as usize];
    let start = left_sym.start as usize;
    let size = left_sym.len + right_sym.len;
    let slice = &text[start..start + size as usize];
    if let Some(&id) = ids.get(slice) {
        let score = scores.get(id as usize).copied().unwrap_or(0.0);
        scratch.work_queue.push(Bigram {
            left,
            right,
            score,
            size,
        });
        rev_merge.insert(slice, (left_sym.len, right_sym.len));
    }
}

/// Streaming UTF-8 assembler.
///
/// A token is a run of *bytes*, and a multi-byte character is routinely split across
/// two tokens — emoji and accented Indonesian text do it constantly. Decoding each
/// token independently would emit a replacement character for the first half and
/// then another for the second, so the UI would show `��` where the model produced
/// `é`. This holds the incomplete tail back until its continuation arrives.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Feed one token's bytes; returns whatever is now a complete UTF-8 prefix.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let good = e.valid_up_to();
                // An *invalid* sequence will never complete, so flush it lossily
                // rather than blocking the stream forever. Only a truncated tail
                // (error_len() == None) is worth holding.
                let keep = match e.error_len() {
                    None => good,
                    Some(_) => self.pending.len(),
                };
                let out = String::from_utf8_lossy(&self.pending[..keep]).into_owned();
                self.pending.drain(..keep);
                out
            }
        }
    }

    /// Flush anything left at end of generation, lossily.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// The Unicode category flags of one code point, as llama.cpp's
/// `unicode_cpt_flags_from_cpt` builds them: the generated range table, plus the
/// `White_Space` set layered on top.
#[inline]
fn cpt_flags(cp: u32) -> u16 {
    let i = unicode_data::RANGES_FLAGS.partition_point(|&(start, _)| start <= cp);
    let mut flags = if i == 0 {
        unicode_data::UNDEFINED
    } else {
        unicode_data::RANGES_FLAGS[i - 1].1
    };
    if unicode_data::WHITESPACE_SET.binary_search(&cp).is_ok() {
        flags |= unicode_data::WHITESPACE;
    }
    flags
}

/// The simple lowercase mapping llama.cpp carries (`unicode_tolower`).
fn tolower(cp: u32) -> u32 {
    match unicode_data::LOWERCASE.binary_search_by_key(&cp, |&(from, _)| from) {
        Ok(i) => unicode_data::LOWERCASE[i].1,
        Err(_) => cp,
    }
}

/// llama.cpp's one-pass accent stripping: the base letter for a precomposed
/// character, itself for everything else.
fn strip_accent(cp: u32) -> u32 {
    let i = unicode_data::NFD_RANGES.partition_point(|&(first, _, _)| first <= cp);
    if i == 0 {
        return cp;
    }
    let (_, last, base) = unicode_data::NFD_RANGES[i - 1];
    if cp <= last {
        base
    } else {
        cp
    }
}

/// llama.cpp's `llm_tokenizer_wpm_session::is_chinese_char`. The ranges are copied
/// verbatim, including `0x2B920` where the reference Python tokenizer has `0x2B820`
/// — see the note in `llama-vocab.cpp`; the two disagree only for that block.
fn is_chinese_char(cp: u32) -> bool {
    (0x04E00..=0x09FFF).contains(&cp)
        || (0x03400..=0x04DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B920..=0x2CEAF).contains(&cp)
        || (0x0F900..=0x0FAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

/// `llm_tokenizer_wpm_session::preprocess`: the words WordPiece is applied to, each
/// already carrying the phantom `▁` the converter puts on every piece.
///
/// `buf` holds the words back to back and `words` their byte ranges. The transformations
/// are llama.cpp's: accent stripping (a one-pass map to the base letter) before
/// anything else, whitespace as the word separator, control characters and stray
/// accent marks dropped, lower-casing every code point, and one word per punctuation
/// mark, per ASCII symbol and per CJK ideograph.
fn wpm_preprocess(text: &str, wpm: &WpmData, buf: &mut String, words: &mut Vec<(usize, usize)>) {
    const SPACE: char = '\u{2581}';
    let marker = SPACE.len_utf8();
    let mut start = buf.len();
    buf.push(SPACE);
    for ch in text.chars() {
        let raw = ch as u32;
        let cp = if wpm.strip_accents {
            strip_accent(raw)
        } else {
            raw
        };
        let flags = cpt_flags(cp);
        if flags & unicode_data::WHITESPACE != 0 {
            if buf.len() > start + marker {
                words.push((start, buf.len()));
                start = buf.len();
                buf.push(SPACE);
            }
            continue;
        }
        if cp == 0 || cp == 0xFFFD || flags & unicode_data::CONTROL != 0 {
            continue;
        }
        if wpm.strip_accents && flags & unicode_data::ACCENT_MARK != 0 {
            continue;
        }
        let cp = if wpm.lowercase { tolower(cp) } else { cp };
        let s = char::from_u32(cp).unwrap_or(ch);
        if flags & unicode_data::PUNCTUATION != 0
            || (cp < 0x7F && flags & unicode_data::SYMBOL != 0)
            || is_chinese_char(cp)
        {
            if buf.len() > start + marker {
                words.push((start, buf.len()));
                start = buf.len();
                buf.push(SPACE);
            }
            let one = start;
            buf.push(s);
            words.push((one, buf.len()));
            start = buf.len();
            buf.push(SPACE);
        } else {
            buf.push(s);
        }
    }
    if buf.len() > start + marker {
        words.push((start, buf.len()));
    }
}

/// llama.cpp's `clean_spaces` post-pass in `llama_vocab::impl::detokenize` (WordPiece
/// only): drop the space before `? ! . ,`, the spaces around a lone apostrophe, and
/// the space before an `'s`/`'m`/`'re`/`'ve` contraction. The three passes are
/// transcribed with their in-place read/write indices, because pass two also zeroes
/// the byte it skips and pass three reads what pass two wrote.
fn clean_spaces(text: &mut Vec<u8>) {
    if text.is_empty() {
        return;
    }
    // Pass 1: " ?", " !", " .", " ,".
    let n = text.len();
    let mut total = 1usize;
    for i in 1..n {
        let x = text[i];
        if text[i - 1] == b' ' && matches!(x, b'?' | b'!' | b'.' | b',') {
            total -= 1;
        }
        text[total] = x;
        total += 1;
    }
    text.truncate(total);

    // Pass 2: strip a lone apostrophe between spaces (" ' ").
    let n = text.len();
    let mut total = 1usize;
    let mut i = 1usize;
    while i < n {
        let x = text[i];
        if x == b'\'' && i + 1 < n && text[i - 1] == b' ' && text[i + 1] == b' ' {
            total -= 1;
            i += 1;
            text[i] = 0;
        }
        text[total] = x;
        total += 1;
        i += 1;
    }
    text.truncate(total);

    // Pass 3: contractions keep no space before 's, 'm, 're, 've.
    let n = text.len();
    let mut total = 1usize;
    for i in 1..n {
        let x = text[i];
        if text[i - 1] == b' ' && x == b'\'' && i + 1 < n {
            let x1 = text[i + 1];
            let mut drop = false;
            if x1 == b's' || x1 == b'm' {
                drop = true;
            } else if i + 2 < n {
                let x2 = text[i + 2];
                if (x1 == b'r' && x2 == b'e') || (x1 == b'v' && x2 == b'e') {
                    drop = true;
                }
            }
            if drop {
                total -= 1;
            }
        }
        text[total] = x;
        total += 1;
    }
    text.truncate(total);
}

/// Regex 1: `\p{N}` — every numeric character becomes its own piece.
fn split_numbers(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        if ch.is_numeric() {
            if i > start {
                out.push(&s[start..i]);
            }
            out.push(&s[i..i + ch.len_utf8()]);
            start = i + ch.len_utf8();
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// Regex 2, as a scanner. Returns the byte length of the match at the start of
/// `rest`, trying the alternatives in the order the pattern lists them.
fn match_piece(rest: &str) -> Option<usize> {
    for c in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if rest.starts_with(c) {
            return Some(c.len());
        }
    }
    // The three ` ?X+` alternatives share an optional single leading space.
    let sp = usize::from(rest.starts_with(' '));
    let after = &rest[sp..];

    let run = |pred: fn(char) -> bool| -> usize {
        let mut n = 0;
        for ch in after.chars() {
            if pred(ch) {
                n += ch.len_utf8();
            } else {
                break;
            }
        }
        n
    };
    for pred in [
        (|c: char| c.is_alphabetic()) as fn(char) -> bool,
        |c: char| c.is_numeric(),
        |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric(),
    ] {
        let n = run(pred);
        if n > 0 {
            return Some(sp + n);
        }
    }

    // `\s+(?!\S)`. Greedy `\s+` with a negative lookahead backtracks by exactly one
    // whitespace character when a non-space follows — that character is left for the
    // next match, where it becomes the leading space of ` ?\p{L}+`. If the run
    // reaches the end of the piece there is nothing to give back.
    let mut ws = 0usize;
    for ch in rest.chars() {
        if ch.is_whitespace() {
            ws += ch.len_utf8();
        } else {
            break;
        }
    }
    if ws > 0 {
        if ws == rest.len() {
            return Some(ws);
        }
        let last = rest[..ws].chars().next_back().unwrap().len_utf8();
        if ws > last {
            return Some(ws - last);
        }
    }
    None
}

/// The Llama 3 family splitter, branch for branch after llama.cpp's
/// `unicode_regex_split_custom_llama3`. `max_digits` is 3 for `llama-bpe`
/// (`\p{N}{1,3}`) and 1 for `qwen2` (`\p{N}`); nothing else differs.
///
/// `\p{L}` and `\p{N}` are `char::is_alphabetic` / `char::is_numeric`, and `\s` is
/// `char::is_whitespace` — the same approximations as the SmolLM scanner, recorded
/// in the module docs.
pub fn split_llama3(s: &str, max_digits: usize) -> Vec<&str> {
    split_with(s, max_digits, false)
}

/// Combining marks (`\p{M}`) that `char::is_alphabetic` does not already cover: the
/// general-purpose combining blocks. Indic and other script-specific marks are mostly
/// `Other_Alphabetic`, hence already letters here. An approximation of this
/// splitter's Unicode classes, like the rest of them; the Tekken splitter uses the
/// generated tables (`cpt_flags`) instead, because llama.cpp's `std::regex` path does.
fn is_mark(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F | 0x0483..=0x0489 | 0x0591..=0x05BD | 0x0610..=0x061A | 0x064B..=0x065F
        | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0x302A..=0x302F | 0x3099..=0x309A
        | 0xFE00..=0xFE0F | 0xFE20..=0xFE2F | 0xE0100..=0xE01EF)
}

// `qwen35` (Qwen3.5 / Bonsai 2) is `qwen2` with `[\p{L}\p{M}]` wherever `qwen2` has `\p{L}`
// (llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_QWEN35); `marks` switches that in.
pub fn split_with(s: &str, max_digits: usize, marks: bool) -> Vec<&str> {
    let cps: Vec<(usize, char)> = s.char_indices().collect();
    let n = cps.len();
    let at = |i: usize| -> Option<char> { cps.get(i).map(|x| x.1) };
    let letter = |i: usize| at(i).is_some_and(|c| c.is_alphabetic() || (marks && is_mark(c)));
    let number = |i: usize| at(i).is_some_and(char::is_numeric);
    let white = |i: usize| at(i).is_some_and(char::is_whitespace);
    let newline = |i: usize| matches!(at(i), Some('\r') | Some('\n'));
    let byte = |i: usize| if i < n { cps[i].0 } else { s.len() };

    let mut out = Vec::new();
    let mut prev_end = 0usize;
    let mut pos = 0usize;
    macro_rules! add {
        ($end:expr) => {{
            let end: usize = $end;
            if end > prev_end {
                out.push(&s[byte(prev_end)..byte(end)]);
            }
            prev_end = end;
        }};
    }

    while pos < n {
        let c = cps[pos].1;

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == '\'' && pos + 1 < n {
            let c1 = at(pos + 1).map(|x| x.to_ascii_lowercase());
            if matches!(c1, Some('s') | Some('t') | Some('m') | Some('d')) {
                pos += 2;
                add!(pos);
                continue;
            }
            if pos + 2 < n {
                let c2 = at(pos + 2).map(|x| x.to_ascii_lowercase());
                if matches!(
                    (c1, c2),
                    (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l'))
                ) {
                    pos += 3;
                    add!(pos);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if !(c == '\r' || c == '\n' || number(pos)) && (letter(pos) || letter(pos + 1)) {
            pos += 1;
            while letter(pos) {
                pos += 1;
            }
            add!(pos);
            continue;
        }

        // \p{N}{1,max}
        if number(pos) {
            let mut ini = pos;
            while number(pos) {
                pos += 1;
                if pos - ini >= max_digits {
                    add!(pos);
                    ini = pos;
                }
            }
            add!(pos);
            continue;
        }

        // <space>?[^\s\p{L}\p{N}]+[\r\n]*
        let f2 = if c == ' ' { pos + 1 } else { pos };
        if at(f2).is_some() && !(white(f2) || letter(f2) || number(f2)) {
            pos = f2;
            while at(pos).is_some() && !(white(pos) || letter(pos) || number(pos)) {
                pos += 1;
            }
            while newline(pos) {
                pos += 1;
            }
            add!(pos);
            continue;
        }

        let mut num_ws = 0usize;
        let mut last_end_nl = 0usize;
        while white(pos + num_ws) {
            if newline(pos + num_ws) {
                last_end_nl = pos + num_ws + 1;
            }
            num_ws += 1;
        }
        // \s*[\r\n]+
        if last_end_nl > 0 {
            pos = last_end_nl;
            add!(pos);
            continue;
        }
        // \s+(?!\S)
        if num_ws > 1 && at(pos + num_ws).is_some() {
            pos += num_ws - 1;
            add!(pos);
            continue;
        }
        // \s+
        if num_ws > 0 {
            pos += num_ws;
            add!(pos);
            continue;
        }
        // no match: one character
        pos += 1;
        add!(pos);
    }
    out
}

/// A letter or a combining mark: llama.cpp's `[\p{L}\p{M}]`.
///
/// `\p{L}` is the Unicode *category* `L`, and Rust's `char::is_alphabetic` is the
/// `Alphabetic` property, which also covers the letter numbers (`Nl`) — the Roman
/// numerals among them. llama.cpp's collapsed regex calls a Roman numeral a
/// number, so this excludes it.
fn is_word_char(c: char) -> bool {
    (c.is_alphabetic() && !is_letter_number(c)) || is_mark(c)
}

/// The Unicode category `Nl`: the letters that are numbers.
fn is_letter_number(c: char) -> bool {
    matches!(c as u32,
        0x2160..=0x2182 | 0x218F | 0x3007 | 0x3021..=0x3029 | 0x3038..=0x303A
        | 0xA000..=0xA48C | 0xA490..=0xA4C6 | 0x12400..=0x1246E)
}

/// The GPT-4o pre-tokenizer's splitter, as llama.cpp v0.9.0's
/// `LLAMA_VOCAB_PRE_TYPE_GPT4O` regex (`src/llama-vocab.cpp:454-465`).
///
/// The regex is
///
/// ```text
/// [^\r\n\p{L}\p{N}]? A* B+ (?:'s|'t|'re|'ve|'m|'ll|'d)?
/// [^\r\n\p{L}\p{N}]? A+ B* (?:'s|'t|'re|'ve|'m|'ll|'d)?
/// \p{N}{1,3}
///  ?[^\s\p{L}\p{N}]+[\r\n/]*
/// \s*[\r\n]+
/// \s+(?!\S)
/// \s+
/// ```
///
/// with `A` a letter or a mark that is not a lowercase ASCII letter (`[^a-z]`
/// behind a `[\p{L}\p{M}]` lookahead) and `B` a letter or a mark that is not an
/// uppercase ASCII letter. This is the vocabulary gpt-oss's files carry
/// (`tokenizer.ggml.pre = "gpt-4o"`, the `o200k_harmony` one).
///
/// The scanner follows the regex branch for branch, backtracking where the regex
/// backtracks: the optional prefix gives back its character when the rest of the
/// word does not match, the `A*` gives back characters until a `B` follows, and
/// the whitespace alternatives give back one character where the negative
/// lookahead asks for it.
///
/// The Unicode classes are approximations, like the other splitters' here: `\p{L}`
/// is Rust's `Alphabetic` property without the letter numbers, `\p{M}` is the
/// general-purpose combining blocks, and `\s` is `char::is_whitespace`.
pub fn split_gpt4o(s: &str) -> Vec<&str> {
    let cps: Vec<(usize, char)> = s.char_indices().collect();
    let n = cps.len();
    let at = |i: usize| -> Option<char> { cps.get(i).map(|x| x.1) };
    let byte = |i: usize| -> usize {
        if i < n {
            cps[i].0
        } else {
            s.len()
        }
    };
    // `(?=[\p{L}\p{M}])([^a-z])`.
    let not_lower = |i: usize| at(i).is_some_and(|c| is_word_char(c) && !c.is_ascii_lowercase());
    // `(?=[\p{L}\p{M}])([^A-Z])`.
    let not_upper = |i: usize| at(i).is_some_and(|c| is_word_char(c) && !c.is_ascii_uppercase());
    // The number of consecutive `B` characters from `i`.
    let mut blen = vec![0usize; n + 1];
    for i in (0..n).rev() {
        blen[i] = if not_upper(i) { blen[i + 1] + 1 } else { 0 };
    }
    // The contraction that may follow a word: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`.
    // The `'ll` one is two letters from `{l, L}`, so `'LL` counts too.
    let contraction = |i: usize| -> usize {
        match at(i) {
            Some('\'') => match at(i + 1) {
                Some('s') | Some('S') | Some('t') | Some('T') | Some('m') | Some('M')
                | Some('d') | Some('D') => 2,
                Some('r') if matches!(at(i + 2), Some('e') | Some('E')) => 3,
                Some('v') if matches!(at(i + 2), Some('e') | Some('E')) => 3,
                Some('l' | 'L') if matches!(at(i + 2), Some('l') | Some('L')) => 3,
                _ => 0,
            },
            _ => 0,
        }
    };
    // The word pattern: the optional prefix, then `A* B+` or `A+ B*`, then the
    // contraction. The end index (exclusive) of the match, or `None`.
    let word = |start: usize| -> Option<usize> {
        // The prefix is greedy: the regex tries it first and gives it back only
        // when the rest of the word does not match.
        for prefix in [1usize, 0] {
            if prefix == 1 {
                let c = at(start)?;
                if c == '\r' || c == '\n' || c.is_alphabetic() || c.is_numeric() {
                    continue;
                }
            }
            let q = start + prefix;
            // The maximal `A` run.
            let mut a = 0usize;
            while q + a < n && not_lower(q + a) {
                a += 1;
            }
            // `A* B+`: the `A` run gives back characters until a `B` follows.
            for a2 in (0..=a).rev() {
                if blen[q + a2] > 0 {
                    let end = q + a2 + blen[q + a2];
                    return Some(end + contraction(end));
                }
            }
            // `A+ B*`: the `A` run is greedy and the `B` run may be empty.
            if a >= 1 {
                let end = q + a + blen[q + a];
                return Some(end + contraction(end));
            }
        }
        None
    };

    let mut out = Vec::new();
    let mut prev_end = 0usize;
    let mut pos = 0usize;
    macro_rules! add {
        ($end:expr) => {{
            let end: usize = $end;
            if end > prev_end {
                out.push(&s[byte(prev_end)..byte(end)]);
            }
            prev_end = end;
        }};
    }

    while pos < n {
        // The word pattern.
        if let Some(end) = word(pos) {
            pos = end;
            add!(pos);
            continue;
        }
        // \p{N}{1,3}
        if at(pos).is_some_and(char::is_numeric) {
            let mut k = 0usize;
            while k < 3 && pos + k < n && at(pos + k).is_some_and(char::is_numeric) {
                k += 1;
            }
            pos += k;
            add!(pos);
            continue;
        }
        // <space>?[^\s\p{L}\p{N}]+[\r\n/]*
        {
            let mut i = pos;
            if at(i) == Some(' ') {
                i += 1;
            }
            let mut k = i;
            while k < n
                && at(k)
                    .is_some_and(|c| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric())
            {
                k += 1;
            }
            if k > i {
                while k < n && matches!(at(k), Some('\r') | Some('\n') | Some('/')) {
                    k += 1;
                }
                pos = k;
                add!(pos);
                continue;
            }
        }
        // The whitespace run, and what the alternatives do with it.
        {
            let mut num_ws = 0usize;
            let mut last_end_nl = 0usize;
            while pos + num_ws < n && at(pos + num_ws).is_some_and(char::is_whitespace) {
                if matches!(at(pos + num_ws), Some('\r') | Some('\n')) {
                    last_end_nl = pos + num_ws + 1;
                }
                num_ws += 1;
            }
            if num_ws > 0 {
                // \s*[\r\n]+
                if last_end_nl > 0 {
                    pos = last_end_nl;
                    add!(pos);
                    continue;
                }
                // \s+(?!\S): the run gives back its last character when a
                // non-space follows.
                if num_ws > 1 && at(pos + num_ws).is_some() {
                    pos += num_ws - 1;
                    add!(pos);
                    continue;
                }
                // \s+
                pos += num_ws;
                add!(pos);
                continue;
            }
        }
        // No alternative matched: one character.
        pos += 1;
        add!(pos);
    }
    out
}

// The classes llama.cpp's collapsed text gives a code point for the Tekken pattern.
const TK_LETTER: u8 = 1 << 0;
const TK_MARK: u8 = 1 << 1;
const TK_NUMBER: u8 = 1 << 2;
const TK_SPACE: u8 = 1 << 3;
const TK_NEWLINE: u8 = 1 << 4;
const TK_LOWER: u8 = 1 << 5;
const TK_UPPER: u8 = 1 << 6;

/// One code point's classes as llama.cpp's `unicode_regex_split` sees them.
///
/// That function has no hand-written Tekken splitter: it collapses every code point
/// to one byte and runs `std::regex` over the result (`src/unicode.cpp`). ASCII stays
/// itself, and the collapsed regex spells `\p{L}` as `[A-Za-z]`, `\p{N}` as `[0-9]`;
/// every non-ASCII code point becomes one byte standing for its category, with
/// whitespace collapsing to a vertical tab. So the classes below are the collapsed
/// ones, from the same tables switch (`cpt_flags`) — not Rust's `is_alphabetic`,
/// which is a different property.
fn tekken_class(c: char) -> u8 {
    let cp = c as u32;
    if cp < 128 {
        let mut f = 0u8;
        if c.is_ascii_alphabetic() {
            f |= TK_LETTER;
        }
        if c.is_ascii_digit() {
            f |= TK_NUMBER;
        }
        // ECMAScript `\s` over a byte string: these six, and nothing else.
        if matches!(c, '\t' | '\n' | '\x0b' | '\x0c' | '\r' | ' ') {
            f |= TK_SPACE;
        }
        if matches!(c, '\r' | '\n') {
            f |= TK_NEWLINE;
        }
        if c.is_ascii_lowercase() {
            f |= TK_LOWER;
        }
        if c.is_ascii_uppercase() {
            f |= TK_UPPER;
        }
        return f;
    }
    let flags = cpt_flags(cp);
    let mut f = 0u8;
    if flags & unicode_data::LETTER != 0 {
        f |= TK_LETTER;
    }
    if flags & unicode_data::ACCENT_MARK != 0 {
        f |= TK_MARK;
    }
    if flags & unicode_data::NUMBER != 0 {
        f |= TK_NUMBER;
    }
    if flags & unicode_data::WHITESPACE != 0 {
        f |= TK_SPACE;
    }
    f
}

/// The Tekken splitter: `LLAMA_VOCAB_PRE_TYPE_TEKKEN` (`src/llama-vocab.cpp`
/// lines 425-431) as llama.cpp's generic `unicode_regex_split` evaluates it.
///
/// ```text
/// [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+
/// [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*
/// \p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// llama.cpp has no custom implementation of this pattern, so `std::regex` sees a
/// collapsed text in which only ASCII case is real case: a non-ASCII letter matches
/// both `[^a-z]` and `[^A-Z]`, so a Greek or Cyrillic word never splits on case and
/// only ASCII boundaries do. A word run is therefore "any letter or mark that is not
/// an ASCII lowercase letter" followed by one that is not an ASCII capital.
///
/// The alternation is tried in the pattern's order, and the greedy groups backtrack
/// as `std::regex` does: `U* L+` gives back letters until `L+` has one to start on,
/// which is why an all-caps run before a lowercase tail (`HTMLParser`) stays one
/// piece while `iPhone` splits after the first letter. No alternative fails for any
/// code point (letter, mark, number, whitespace, punctuation covers everything); the
/// one-character fallback mirrors llama.cpp's for the theoretically unreachable case.
pub fn split_tekken(s: &str) -> Vec<&str> {
    tekken_pieces(s, false)
}

/// `tekken-num3`: the same pattern with
/// `\p{N}{1,3}(?=(\p{N}{3})*(?!\p{N}))|\p{N}{1,2}` in place of `\p{N}`, grouping a
/// digit run in threes from the right (`src/llama-vocab.cpp` lines 432-437).
pub fn split_tekken_num3(s: &str) -> Vec<&str> {
    tekken_pieces(s, true)
}

fn tekken_pieces(s: &str, digits_in_threes: bool) -> Vec<&str> {
    let cps: Vec<(usize, char)> = s.char_indices().collect();
    let n = cps.len();
    let cls: Vec<u8> = cps.iter().map(|&(_, c)| tekken_class(c)).collect();
    let at = |i: usize| -> u8 { cls.get(i).copied().unwrap_or(0) };
    let has = |i: usize, bits: u8| at(i) & bits != 0;
    let word = |i: usize| has(i, TK_LETTER | TK_MARK);
    let upper = |i: usize| word(i) && !has(i, TK_LOWER);
    let lower = |i: usize| word(i) && !has(i, TK_UPPER);
    // `[^\r\n\p{L}\p{N}]`, the optional character that attaches to a word.
    let prefix = |i: usize| i < n && !has(i, TK_NEWLINE | TK_LETTER | TK_NUMBER);
    // `[^\s\p{L}\p{N}]`.
    let punctuation = |i: usize| i < n && !has(i, TK_SPACE | TK_LETTER | TK_NUMBER);
    let byte = |i: usize| if i < n { cps[i].0 } else { s.len() };

    // Alternative 1: `P? U* L+`, prefix first when there is one.
    let upper_then_lower = |p: usize| -> Option<usize> {
        let mut q = p;
        while upper(q) {
            q += 1;
        }
        // Greedy `U*` gives its last character back until `L+` can start.
        let mut e = q;
        loop {
            if e < n && lower(e) {
                let mut end = e;
                while lower(end) {
                    end += 1;
                }
                return Some(end);
            }
            if e == p {
                return None;
            }
            e -= 1;
        }
    };
    let alt1 = |pos: usize| -> Option<usize> {
        if prefix(pos) {
            if let Some(end) = upper_then_lower(pos + 1) {
                return Some(end);
            }
        }
        upper_then_lower(pos)
    };

    // Alternative 2: `P? U+ L*`.
    let upper_run = |p: usize| -> Option<usize> {
        if !upper(p) {
            return None;
        }
        let mut end = p;
        while upper(end) {
            end += 1;
        }
        while lower(end) {
            end += 1;
        }
        Some(end)
    };
    let alt2 = |pos: usize| -> Option<usize> {
        if prefix(pos) {
            if let Some(end) = upper_run(pos + 1) {
                return Some(end);
            }
        }
        upper_run(pos)
    };

    // Alternative 3: `\p{N}`, or the right-aligned groups of `tekken-num3`.
    let alt3 = |pos: usize| -> Option<usize> {
        if !has(pos, TK_NUMBER) {
            return None;
        }
        if !digits_in_threes {
            return Some(pos + 1);
        }
        let mut q = pos;
        while has(q, TK_NUMBER) {
            q += 1;
        }
        // `\p{N}{1,3}` with the lookahead "what follows is a multiple of three".
        Some(pos + (q - pos - 1) % 3 + 1)
    };

    // Alternative 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`.
    let alt4 = |pos: usize| -> Option<usize> {
        let space = cps[pos].1 == ' ' && punctuation(pos + 1);
        let start = if space { pos + 1 } else { pos };
        if !punctuation(start) {
            return None;
        }
        let mut end = start;
        while punctuation(end) {
            end += 1;
        }
        while end < n && matches!(cps[end].1, '\r' | '\n' | '/') {
            end += 1;
        }
        Some(end)
    };

    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < n {
        let mut end = alt1(pos)
            .or_else(|| alt2(pos))
            .or_else(|| alt3(pos))
            .or_else(|| alt4(pos));

        if end.is_none() {
            let mut ws = 0usize;
            while has(pos + ws, TK_SPACE) {
                ws += 1;
            }
            if ws > 0 {
                // `\s*[\r\n]+`: keep everything up to and including the run of
                // newlines around the whitespace run's *last* newline.
                let mut newline = None;
                for k in (0..ws).rev() {
                    if has(pos + k, TK_NEWLINE) {
                        newline = Some(pos + k);
                        break;
                    }
                }
                end = if let Some(r) = newline {
                    let mut e = r;
                    while has(e, TK_NEWLINE) {
                        e += 1;
                    }
                    Some(e)
                } else if ws > 1 && pos + ws < n {
                    // `\s+(?!\S)`: the run gives back one character when a
                    // non-space follows, so it can prefix the next word.
                    Some(pos + ws - 1)
                } else {
                    // `\s+`.
                    Some(pos + ws)
                };
            }
        }

        // No alternative matched: one character (llama.cpp's own fallback).
        let end = end.unwrap_or(pos + 1);
        out.push(&s[byte(pos)..byte(end)]);
        pos = end;
    }
    out
}

/// `nemotron-3.5-embed` only: llama.cpp rewrites `[` + U+200C and `<` + U+200C to
/// the bare bracket before splitting (`src/llama-vocab.cpp` lines 652-680). One
/// joiner per bracket goes, and the scan does not revisit a joiner it kept.
fn strip_zwnj_after_bracket(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev: Option<char> = None;
    for c in text.chars() {
        if !(c == '\u{200c}' && matches!(prev, Some('[') | Some('<'))) {
            out.push(c);
        }
        prev = Some(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPT-4o splitter, against llama.cpp v0.9.0's own
    /// `unicode_regex_split` with the `LLAMA_VOCAB_PRE_TYPE_GPT4O` regex
    /// (`src/llama-vocab.cpp:454-465`). The expectations are what llama.cpp
    /// produces, checked with its `std::regex` over its collapsed text.
    #[test]
    fn the_gpt4o_splitter_matches_llama_cpp() {
        // A word, and the space that belongs to the next one.
        assert_eq!(split_gpt4o("hello world"), vec!["hello", " world"]);
        // The case-sensitive pattern: a capitalised word, an all-caps one.
        assert_eq!(split_gpt4o("Hello World"), vec!["Hello", " World"]);
        assert_eq!(split_gpt4o("HELLO WORLD"), vec!["HELLO", " WORLD"]);
        // The contraction: the apostrophe is the word's prefix, and `'LL` is a
        // contraction of two letters from `{l, L}`.
        assert_eq!(split_gpt4o("it's"), vec!["it's"]);
        assert_eq!(split_gpt4o("I'LL don't"), vec!["I'LL", " don't"]);
        assert_eq!(split_gpt4o("don't"), vec!["don't"]);
        assert_eq!(split_gpt4o(" 'don't"), vec![" '", "don't"]);
        // Digits in groups of up to three.
        assert_eq!(split_gpt4o("12345"), vec!["123", "45"]);
        // Punctuation takes one optional space and the newlines after it.
        assert_eq!(split_gpt4o("x !?\n\ny"), vec!["x", " !?\n\n", "y"]);
        // A slash is punctuation, and each piece stops at the next letter.
        assert_eq!(split_gpt4o("x/y/z"), vec!["x", "/y", "/z"]);
        // Whitespace: a run of two before a non-space gives back exactly one.
        assert_eq!(split_gpt4o("a  b"), vec!["a", " ", " b"]);
        // A letter outside ASCII is a letter, and a Roman numeral is a number.
        assert_eq!(split_gpt4o("héllo"), vec!["héllo"]);
        assert_eq!(split_gpt4o("中文"), vec!["中文"]);
        assert_eq!(split_gpt4o("Ⅻfoo"), vec!["Ⅻ", "foo"]);
    }

    /// The Tekken splitter, against llama.cpp v0.9.0's own `unicode_regex_split` with
    /// the `LLAMA_VOCAB_PRE_TYPE_TEKKEN` pattern (`src/llama-vocab.cpp` lines 425-431).
    /// Unlike the other pre-types there is no hand-written implementation there, so
    /// every expectation below is what `std::regex` returned over llama.cpp's
    /// collapsed text — including the pieces that surprised: `HTMLParser` stays one
    /// piece because the first alternative's `U* L+` matches the whole run, while
    /// `iPhone` splits after the first letter.
    #[test]
    fn the_tekken_splitter_matches_llama_cpp() {
        // Alternative 1: `P? U* L+`. An all-caps run followed by a lowercase tail is
        // one piece, because the regex does not prefer the shorter `L+`.
        assert_eq!(split_tekken("HTMLParser"), vec!["HTMLParser"]);
        assert_eq!(split_tekken("HTTPServer"), vec!["HTTPServer"]);
        assert_eq!(split_tekken("ABc"), vec!["ABc"]);
        // Alternative 2: `P? U+ L*`. A lowercase first letter leaves `U+` to the
        // capitalised tail, which is where `camelCase` splits.
        assert_eq!(split_tekken("iPhone"), vec!["i", "Phone"]);
        assert_eq!(split_tekken("macOS"), vec!["mac", "OS"]);
        assert_eq!(split_tekken("abcDEF"), vec!["abc", "DEF"]);
        // The optional prefix: one non-letter character attaches to the word.
        assert_eq!(split_tekken(" hello"), vec![" hello"]);
        assert_eq!(split_tekken("(hello"), vec!["(hello"]);
        assert_eq!(split_tekken("-hello"), vec!["-hello"]);
        assert_eq!(split_tekken("\thello"), vec!["\thello"]);
        assert_eq!(split_tekken("1hello"), vec!["1", "hello"]);
        // A quote is not part of a word, but the space before it is its prefix.
        assert_eq!(split_tekken(" \"hello"), vec![" \"", "hello"]);
        assert_eq!(split_tekken("it's"), vec!["it", "'s"]);
        assert_eq!(split_tekken("don't"), vec!["don", "'t"]);
        assert_eq!(split_tekken("I'LL"), vec!["I", "'LL"]);
        // Alternative 3: `\p{N}`, one digit at a time.
        assert_eq!(split_tekken("1"), vec!["1"]);
        assert_eq!(split_tekken("42"), vec!["4", "2"]);
        assert_eq!(split_tekken("2026"), vec!["2", "0", "2", "6"]);
        assert_eq!(split_tekken("12345"), vec!["1", "2", "3", "4", "5"]);
        assert_eq!(split_tekken("3.14"), vec!["3", ".", "1", "4"]);
        // Alternative 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*` — punctuation takes the slash
        // and newlines that follow it.
        assert_eq!(split_tekken("a.../b"), vec!["a", ".../", "b"]);
        assert_eq!(split_tekken("a//b"), vec!["a", "//", "b"]);
        assert_eq!(split_tekken("///"), vec!["///"]);
        assert_eq!(split_tekken("x!?\n\ny"), vec!["x", "!?\n\n", "y"]);
        assert_eq!(split_tekken("x !?\n\ny"), vec!["x", " !?\n\n", "y"]);
        assert_eq!(split_tekken("a/\n/b"), vec!["a", "/\n/", "b"]);
        // Alternative 5: `\s*[\r\n]+` — a whitespace run ends at its last newline.
        assert_eq!(split_tekken("a\n\nb"), vec!["a", "\n\n", "b"]);
        assert_eq!(split_tekken("a\r\nb"), vec!["a", "\r\n", "b"]);
        assert_eq!(split_tekken("a\r\rb"), vec!["a", "\r\r", "b"]);
        assert_eq!(split_tekken("\n"), vec!["\n"]);
        assert_eq!(split_tekken("\n\n\n"), vec!["\n\n\n"]);
        assert_eq!(split_tekken(" \n "), vec![" \n", " "]);
        assert_eq!(split_tekken("a \n \n b"), vec!["a", " \n \n", " b"]);
        // Alternatives 6 and 7: `\s+(?!\S)` gives back one whitespace character when
        // a non-space follows, `\s+` keeps a trailing run whole.
        assert_eq!(split_tekken("a b"), vec!["a", " b"]);
        assert_eq!(split_tekken("a  b"), vec!["a", " ", " b"]);
        assert_eq!(split_tekken("  word"), vec![" ", " word"]);
        assert_eq!(split_tekken("a \tb"), vec!["a", " ", "\tb"]);
        assert_eq!(split_tekken("a\t b"), vec!["a", "\t", " b"]);
        assert_eq!(split_tekken("abc "), vec!["abc", " "]);
        assert_eq!(split_tekken("abc  "), vec!["abc", "  "]);
        assert_eq!(split_tekken("abc\t"), vec!["abc", "\t"]);
        assert_eq!(split_tekken("abc \t\n"), vec!["abc", " \t\n"]);
        // Only ASCII case splits: a non-ASCII letter is both `[^a-z]` and `[^A-Z]` in
        // llama.cpp's collapsed text, so Greek and Cyrillic runs stay whole.
        assert_eq!(split_tekken("Привет"), vec!["Привет"]);
        assert_eq!(split_tekken("Γειά"), vec!["Γειά"]);
        assert_eq!(split_tekken("中文"), vec!["中文"]);
        assert_eq!(split_tekken("café"), vec!["café"]);
        // A combining mark is a letter for this pattern and stays with its base.
        assert_eq!(split_tekken("e\u{301}"), vec!["e\u{301}"]);
        assert_eq!(split_tekken("x\u{301}y"), vec!["x\u{301}y"]);
        // Symbols are punctuation, and can prefix the word after them.
        assert_eq!(split_tekken("a🙂b"), vec!["a", "🙂b"]);
        assert_eq!(split_tekken("🙂 a"), vec!["🙂", " a"]);
        assert_eq!(split_tekken("🙂"), vec!["🙂"]);
        // The embedding prefixes: the colon is punctuation, the trailing space is a
        // run of its own because nothing follows it.
        assert_eq!(split_tekken("query: "), vec!["query", ":", " "]);
        assert_eq!(split_tekken("passage: "), vec!["passage", ":", " "]);
        assert_eq!(
            split_tekken("query: the capital"),
            vec!["query", ":", " the", " capital"]
        );
        // Empty input has no pieces; the tokenizer emits no ids for it.
        assert!(split_tekken("").is_empty());
    }

    /// `tekken-num3`: digits in right-aligned groups of three, the same in everything
    /// else (`src/llama-vocab.cpp` lines 432-437).
    #[test]
    fn the_tekken_num3_splitter_matches_llama_cpp() {
        assert_eq!(split_tekken_num3("1234567"), vec!["1", "234", "567"]);
        assert_eq!(split_tekken_num3("12345"), vec!["12", "345"]);
        assert_eq!(split_tekken_num3("1234"), vec!["1", "234"]);
        assert_eq!(split_tekken_num3("123"), vec!["123"]);
        assert_eq!(split_tekken_num3("12"), vec!["12"]);
        assert_eq!(split_tekken_num3("1"), vec!["1"]);
        assert_eq!(split_tekken_num3("0"), vec!["0"]);
        assert_eq!(split_tekken_num3("2026"), vec!["2", "026"]);
        assert_eq!(split_tekken_num3("12.345"), vec!["12", ".", "345"]);
        assert_eq!(
            split_tekken_num3("a1234567b"),
            vec!["a", "1", "234", "567", "b"]
        );
        assert_eq!(
            split_tekken_num3("1 22 333"),
            vec!["1", " ", "22", " ", "333"]
        );
        // Everything that is not a digit splits exactly as `tekken` does.
        assert_eq!(split_tekken_num3("HTMLParser"), vec!["HTMLParser"]);
        assert_eq!(split_tekken_num3("iPhone"), vec!["i", "Phone"]);
        assert_eq!(split_tekken_num3("a  b"), vec!["a", " ", " b"]);
    }

    /// `nemotron-3.5-embed` drops the zero-width non-joiner after `[` and `<` before
    /// splitting (`src/llama-vocab.cpp` lines 652-680); `pixtral`/`tekken` does not.
    #[test]
    fn the_tekken_joiner_rule_is_nemotron_only() {
        assert_eq!(strip_zwnj_after_bracket("[\u{200c}INST]"), "[INST]");
        assert_eq!(strip_zwnj_after_bracket("<\u{200c}s>"), "<s>");
        assert_eq!(strip_zwnj_after_bracket("x[\u{200c}y"), "x[y");
        // Not after a bracket: untouched.
        assert_eq!(strip_zwnj_after_bracket("a\u{200c}b"), "a\u{200c}b");
        assert_eq!(strip_zwnj_after_bracket("\u{200c}["), "\u{200c}[");
        // One joiner per bracket, and the scan does not revisit what it rewrote.
        assert_eq!(strip_zwnj_after_bracket("[\u{200c}\u{200c}"), "[\u{200c}");
        assert_eq!(
            strip_zwnj_after_bracket("[[\u{200c}\u{200c}x"),
            "[[\u{200c}x"
        );
        // Without the rule the joiner is a character of its own: llama.cpp splits
        // `[` + U+200C into a piece of its own and `INST` into another.
        assert_eq!(
            split_tekken("[\u{200c}INST]"),
            vec!["[\u{200c}", "INST", "]"]
        );
        // With the rule the text is `[INST]`, which the splitter also cuts apart —
        // the control token is matched before the splitter ever sees it.
        assert_eq!(
            split_tekken(&strip_zwnj_after_bracket("[\u{200c}INST]")),
            vec!["[INST", "]"]
        );
    }

    #[test]
    fn llama3_splitter_follows_llama_cpp_branch_for_branch() {
        // Digits: groups of three for Llama 3, singles for Qwen2.
        assert_eq!(split_llama3("12345", 3), vec!["123", "45"]);
        assert_eq!(split_llama3("12345", 1), vec!["1", "2", "3", "4", "5"]);
        // A leading space attaches to a letter run; before a digit it stands alone.
        assert_eq!(split_llama3("a b", 3), vec!["a", " b"]);
        assert_eq!(split_llama3("a 1", 3), vec!["a", " ", "1"]);
        // Contractions are case-insensitive and win over the letter run.
        assert_eq!(
            split_llama3("I'LL don't", 3),
            vec!["I", "'LL", " don", "'t"]
        );
        // Punctuation takes one optional space and trailing newlines.
        assert_eq!(
            split_llama3(
                "x !?

y", 3
            ),
            vec![
                "x", " !?

", "y"
            ]
        );
        // Whitespace ending in a newline is one piece; a trailing space goes on.
        assert_eq!(
            split_llama3(
                "a

  
 b", 3
            ),
            vec![
                "a", "

  
", " b"
            ]
        );
        // Run of spaces before a non-space gives back exactly one.
        assert_eq!(split_llama3("a    b", 3), vec!["a", "   ", " b"]);
        // Trailing whitespace is kept whole.
        assert_eq!(split_llama3("a   ", 3), vec!["a", "   "]);
        // Any non-newline, non-letter, non-digit char may prefix a letter run.
        assert_eq!(split_llama3("(abc", 3), vec!["(abc"]);
        assert_eq!(
            split_llama3(
                "
abc", 3
            ),
            vec![
                "
", "abc"
            ]
        );
        assert_eq!(split_llama3("", 3), Vec::<&str>::new());
    }

    /// A tiny WordPiece vocabulary in the shape the converter stores: `▁`-prefixed
    /// words, `##` continuations un-prefixed, the specials the BERT converter marks
    /// control, and one word longer than any token so the greedy split is exercised.
    fn toy_wpm() -> Tokenizer {
        let mut pieces = vec![
            "[PAD]".to_string(),
            "[UNK]".to_string(),
            "[CLS]".to_string(),
            "[SEP]".to_string(),
            "[MASK]".to_string(),
        ];
        pieces.extend(
            [
                "▁hello", "▁world", "▁he", "llo", "▁a", "▁test", "##ing", "▁12",
            ]
            .iter()
            .map(|p| p.to_string()),
        );
        let types = vec![3, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1];
        let mut ids = HashMap::new();
        for (i, p) in pieces.iter().enumerate() {
            ids.entry(p.clone()).or_insert(i as u32);
        }
        build_wpm(
            pieces,
            ids,
            types,
            VocabConfig {
                bos: Some(2),
                eos: None,
                unk: Some(1),
                sep: Some(3),
                cls: Some(2),
                mask: Some(4),
                add_bos: true,
                add_eos: false,
                add_sep: true,
            },
            true,
            true,
        )
    }

    #[test]
    fn wordpiece_prepares_words_like_llama_cpp() {
        let tk = toy_wpm();
        // Longest-match first, with `▁` as the word start.
        assert_eq!(tk.encode("hello world"), vec![5, 6]);
        assert_eq!(tk.encode("test"), vec![10]);
        // A word the vocabulary cannot cover whole becomes one unknown token.
        assert_eq!(tk.encode("xyz"), vec![1]);
        // Case folding and accent stripping.
        assert_eq!(tk.encode("HELLO"), vec![5]);
        assert_eq!(tk.encode("héllo"), vec![5]);
        // Punctuation is a word of its own, and so is a CJK ideograph.
        assert_eq!(tk.encode("hello!"), vec![5, 1]);
        // Numbers are one word, so the digit token matches.
        assert_eq!(tk.encode("12"), vec![12]);
    }

    #[test]
    fn wordpiece_special_tokens() {
        let tk = toy_wpm();
        // llama.cpp pulls special-token texts out of the raw input, longest first.
        assert_eq!(tk.encode("[CLS]hello[SEP]"), vec![2, 5, 3]);
        assert_eq!(tk.encode("[MASK][MASK]"), vec![4, 4]);
        // The embedding input wraps the text.
        assert_eq!(tk.encode_special("hello"), vec![2, 5, 3]);
        assert_eq!(tk.encode_prompt("hello"), vec![2, 5, 3]);
        // The rerank pair, where the query and document do not parse specials.
        assert_eq!(tk.encode_pair("hello", "world"), vec![2, 5, 3, 6]);
        // `[CLS]` in a rerank query is three unknown words, not the CLS token.
        assert_eq!(tk.encode_pair("[CLS]", ""), vec![2, 1, 1, 1, 3]);
    }

    #[test]
    fn unicode_tables_answer_like_llama_cpp() {
        use unicode_data::{ACCENT_MARK, CONTROL, PUNCTUATION, SYMBOL, WHITESPACE};
        // `\p{Z}` and `\p{C}` ranges, the white-space set, and the ASCII split the WPM
        // preprocessor relies on.
        assert!(cpt_flags(' ' as u32) & WHITESPACE != 0);
        assert!(cpt_flags('\u{00a0}' as u32) & WHITESPACE != 0);
        assert!(cpt_flags('\u{3000}' as u32) & WHITESPACE != 0);
        assert!(cpt_flags('\u{0007}' as u32) & CONTROL != 0);
        assert!(cpt_flags(',' as u32) & PUNCTUATION != 0);
        assert!(cpt_flags('\u{2014}' as u32) & PUNCTUATION != 0);
        assert!(cpt_flags('+' as u32) & SYMBOL != 0);
        assert!(cpt_flags('\u{0301}' as u32) & ACCENT_MARK != 0);
        assert_eq!(cpt_flags('a' as u32) & (PUNCTUATION | SYMBOL | CONTROL), 0);
        // The simple case mappings and the one-pass accent strip.
        assert_eq!(tolower('A' as u32), 'a' as u32);
        assert_eq!(tolower('İ' as u32), 'i' as u32);
        assert_eq!(tolower('a' as u32), 'a' as u32);
        assert_eq!(strip_accent('é' as u32), 'e' as u32);
        assert_eq!(strip_accent('Å' as u32), 'A' as u32);
        assert_eq!(strip_accent('ß' as u32), 'ß' as u32);
        assert_eq!(strip_accent('a' as u32), 'a' as u32);
        // The CJK ranges, including llama.cpp's `0x2B920` (the HF tokenizer's is
        // `0x2B820`) — copied, not corrected, because the goldens come from llama.cpp.
        assert!(is_chinese_char('中' as u32));
        assert!(is_chinese_char(0x3400));
        assert!(is_chinese_char(0x2B920));
        assert!(!is_chinese_char('あ' as u32));
    }

    #[test]
    fn trie_finds_the_longest_token() {
        let pieces = ["a", "ab", "abc", "bc"];
        let items: Vec<(&str, u32)> = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (*p, i as u32))
            .collect();
        let t = Trie::build(&items);
        let mut node = 0;
        let mut longest = None;
        for &b in b"abcd" {
            match t.child(node, b) {
                Some(n) => {
                    node = n;
                    if let Some(id) = t.token(node) {
                        longest = Some(id);
                    }
                }
                None => break,
            }
        }
        assert_eq!(longest, Some(2));
        assert_eq!(t.child(0, b'z'), None);
        // A duplicate piece keeps the highest id, as llama.cpp's `token_to_id` does.
        let dup: Vec<(&str, u32)> = vec![("x", 3), ("x", 7)];
        let t = Trie::build(&dup);
        assert_eq!(t.token(t.child(0, b'x').unwrap()), Some(7));
    }

    #[test]
    fn clean_spaces_does_what_llama_cpp_does() {
        let mut b = b" a , b".to_vec();
        clean_spaces(&mut b);
        assert_eq!(String::from_utf8(b).unwrap(), " a, b");
        // " ' " collapses to a single apostrophe with no spaces.
        let mut b = b" a ' b".to_vec();
        clean_spaces(&mut b);
        assert_eq!(String::from_utf8(b).unwrap(), " a'b");
        // Contractions: llama.cpp drops the space before 's, 'm, 're and 've, and
        // keeps it before 't, 'd and 'll (the `'ll` branch has its `total--`
        // commented out; the goldens are generated from that behaviour, so it is
        // matched rather than fixed).
        let mut b = b" it 's don 't we 'll they 're".to_vec();
        clean_spaces(&mut b);
        assert_eq!(String::from_utf8(b).unwrap(), " it's don 't we 'll they're");
        let mut empty: Vec<u8> = Vec::new();
        clean_spaces(&mut empty);
        assert!(empty.is_empty());
    }

    #[test]
    fn pre_types_round_trip_their_names() {
        for n in PRE_TYPES {
            let pre = PreType::from_name(n).unwrap_or_else(|| panic!("{n} is not accepted"));
            // Every accepted name maps to a type whose canonical name is also
            // accepted: `pixtral` is the second name of `tekken`, as in llama.cpp.
            assert!(PreType::from_name(pre.name()).is_some());
            assert_eq!(PreType::from_name(pre.name()).unwrap().name(), pre.name());
        }
        assert_eq!(PreType::from_name("pixtral"), Some(PreType::Tekken));
        assert_eq!(
            PreType::from_name("nemotron-3.5-embed"),
            Some(PreType::Nemotron35Embed)
        );
        // ModernBERT's vocabulary says `modern-bert`; it is the GPT-2 pattern.
        assert_eq!(PreType::from_name("modern-bert"), Some(PreType::Gpt2));
        assert_eq!(PreType::from_name("gpt-2"), Some(PreType::Gpt2));
    }

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let (enc, dec) = byte_alphabet();
        assert_eq!(
            dec.len(),
            256,
            "every byte must map to a distinct character"
        );
        for b in 0..256usize {
            assert_eq!(dec[&enc[b]], b as u8);
        }
        // The GPT-2 constants everyone recognises: space is Ġ, newline is Ċ.
        assert_eq!(enc[b' ' as usize], 'Ġ');
        assert_eq!(enc[b'\n' as usize], 'Ċ');
        assert_eq!(enc[b'!' as usize], '!');
    }

    #[test]
    fn digits_are_split_individually() {
        assert_eq!(split_numbers("abc"), vec!["abc"]);
        assert_eq!(split_numbers("a1b"), vec!["a", "1", "b"]);
        assert_eq!(split_numbers("2026"), vec!["2", "0", "2", "6"]);
        assert_eq!(split_numbers(" 5%"), vec![" ", "5", "%"]);
    }

    /// The GPT-2 pre-type is the plain alternation, with digits in one run
    /// (`LLAMA_VOCAB_PRE_TYPE_GPT2`; ModernBERT's `modern-bert` pre-tokenizer).
    #[test]
    fn the_gpt2_splitter_keeps_digit_runs_together() {
        fn split(s: &str) -> Vec<&str> {
            let mut out = Vec::new();
            let mut p = 0usize;
            while p < s.len() {
                let take = match_piece(&s[p..])
                    .unwrap_or_else(|| s[p..].chars().next().map(|c| c.len_utf8()).unwrap_or(1));
                out.push(&s[p..p + take]);
                p += take;
            }
            out
        }
        assert_eq!(split("Hello world!"), vec!["Hello", " world", "!"]);
        assert_eq!(split("it's 2026!"), vec!["it", "'s", " 2026", "!"]);
        assert_eq!(split("  x"), vec![" ", " x"]);
        assert_eq!(split("\n\nline"), vec!["\n", "\n", "line"]);
    }

    #[test]
    fn the_whitespace_lookahead_gives_back_exactly_one_character() {
        // "  abc": the greedy run backtracks so the final space joins "abc".
        assert_eq!(match_piece("  abc"), Some(1));
        // At end of piece there is nothing to give back.
        assert_eq!(match_piece("  "), Some(2));
        // A single space before a word cannot give anything back, so this
        // alternative fails and ` ?\p{L}+` takes it instead.
        assert_eq!(match_piece(" abc"), Some(4));
        // A lone newline followed by a word matches nothing and is emitted raw.
        assert_eq!(match_piece("\nline"), None);
        // Two newlines: one is given back.
        assert_eq!(match_piece("\n\nline"), Some(1));
    }

    #[test]
    fn contractions_win_over_the_letter_run() {
        assert_eq!(match_piece("'s rest"), Some(2));
        assert_eq!(match_piece("'ll go"), Some(3));
        // Not a listed contraction: falls through to the punctuation run.
        assert_eq!(match_piece("'x"), Some(1));
    }

    #[test]
    fn utf8_stream_holds_a_split_character_until_it_completes() {
        let mut s = Utf8Stream::default();
        // "é" is 0xC3 0xA9, routinely split across two tokens.
        assert_eq!(s.push(&[0xC3]), "");
        assert_eq!(s.push(&[0xA9]), "é");
        assert_eq!(s.push("ok".as_bytes()), "ok");
        // Genuinely invalid input must not block the stream forever.
        let mut s2 = Utf8Stream::default();
        assert_eq!(s2.push(&[0xFF]), "\u{fffd}");
        assert_eq!(s2.push(b"a"), "a");
    }

    #[test]
    fn utf8_stream_flushes_a_truncated_tail_at_the_end() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(&[0xE2, 0x82]), ""); // two thirds of "€"
        assert_eq!(s.finish(), "\u{fffd}");
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn encode_prompt_ensures_exactly_one_bos_when_add_bos_is_enabled() {
        let pieces = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "a".to_string(),
        ];
        let scores = vec![0.0, 0.0, 0.0, 1.0];
        let token_types = vec![2, 3, 3, 1];
        let mut tk = Tokenizer::from_spm_parts(pieces, scores, token_types, 1, 2, 0, false);
        // Without leading BOS in text, encode_prompt prepends BOS
        assert_eq!(tk.encode_prompt("a"), vec![1, 3]);
        // With leading BOS already present, no duplicate is added
        assert_eq!(tk.encode_prompt("<s>a"), vec![1, 3]);

        // When add_bos is false, no BOS is prepended
        tk.add_bos = false;
        assert_eq!(tk.encode_prompt("a"), vec![3]);
    }

    #[test]
    fn pretype_dbrx_matches_llama3_split() {
        assert_eq!(PreType::from_name("dbrx"), Some(PreType::Dbrx));
        assert_eq!(PreType::Dbrx.name(), "dbrx");
        // llama.cpp v0.9.0 src/llama-vocab.cpp lines 310-315 map DBRX to Llama 3 regex
        let text = "Hello 123 world!";
        let pieces: Vec<&str> = split_llama3(text, 3);
        assert_eq!(pieces, vec!["Hello", " ", "123", " world", "!"]);
    }

    /// A byte-level BPE over a toy vocabulary, with the given pre-tokenizer.
    ///
    /// The vocabulary holds every single byte and the merged pair `12`, so the
    /// difference between the pre-tokenizers shows up in the ids: the Llama 3
    /// pattern (and so `dbrx`) keeps "123" in one piece, which the merge can
    /// reach, while the SmolLM pattern splits the digits apart.
    fn toy_bpe(pre: PreType) -> Tokenizer {
        let (byte_enc, byte_dec) = byte_alphabet();
        let mut pieces: Vec<String> =
            vec!["<unk>".to_string(), "<s>".to_string(), "12".to_string()];
        for b in 0..=255u8 {
            pieces.push(byte_enc[b as usize].to_string());
        }
        let mut ids = HashMap::with_capacity(pieces.len());
        for (i, p) in pieces.iter().enumerate() {
            ids.entry(p.clone()).or_insert(i as u32);
        }
        let mut ranks = HashMap::new();
        ranks.insert(("1".to_string(), "2".to_string()), 0u32);
        Tokenizer {
            pieces,
            ids,
            specials: Vec::new(),
            special_index: SpecialIndex::default(),
            user_specials: Vec::new(),
            user_index: SpecialIndex::default(),
            bos: 1,
            eos: 1,
            add_bos: false,
            add_eos: false,
            add_sep: false,
            sep: None,
            cls: None,
            mask: None,
            pre,
            n_vocab: 258,
            algo: TokenizerAlgo::Bpe(BpeData {
                ranks,
                byte_enc,
                byte_dec,
            }),
        }
    }

    #[test]
    fn generator_special_tokens_are_the_llama_cpp_prologue_and_epilogue() {
        // A byte-level BPE (Qwen3-Embedding's tokenizer type): `add_special` is BOS
        // before the text and EOS after it, both from the file's flags.
        let mut bpe = toy_bpe(PreType::Qwen2);
        assert!(!bpe.add_bos && !bpe.add_eos);
        assert_eq!(bpe.encode_special("12"), bpe.encode("12"));
        bpe.add_eos = true;
        let mut with_eos = bpe.encode("12");
        with_eos.push(bpe.eos);
        assert_eq!(bpe.encode_special("12"), with_eos);
        bpe.add_bos = true;
        let mut both = vec![bpe.bos];
        both.extend(bpe.encode("12"));
        both.push(bpe.eos);
        assert_eq!(bpe.encode_special("12"), both);
        // The chat path still deduplicates a template-written BOS (and never appends EOS).
        let mut chat = vec![bpe.bos];
        chat.extend(bpe.encode("12"));
        assert_eq!(bpe.encode_prompt("12"), chat);

        // A SentencePiece file: `add_bos` is on, `add_eos` only when the file says so.
        let pieces = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "a".to_string(),
        ];
        let scores = vec![0.0, 0.0, 0.0, 1.0];
        let token_types = vec![2, 3, 3, 1];
        let mut spm = Tokenizer::from_spm_parts(pieces, scores, token_types, 1, 2, 0, false);
        assert!(!spm.add_eos);
        assert_eq!(spm.encode_special("a"), vec![1, 3]);
        spm.add_eos = true;
        assert_eq!(spm.encode_special("a"), vec![1, 3, 2]);
        assert_eq!(spm.encode_prompt("a"), vec![1, 3]);
    }

    #[test]
    fn bpe_pair_path_does_not_parse_control_tokens() {
        // The rerank pair of a `modern-bert` file (S4.7b): llama.cpp's
        // `format_prompt_rerank` tokenizes the query and the document with
        // `parse_special = false`, so a control-token spelling is ordinary text while a
        // user-defined one is still matched, and it assembles
        // `[bos] q [eos] [sep] d [eos]` from the file's own flags.
        let (byte_enc, byte_dec) = byte_alphabet();
        let mut pieces: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
        let mut types = vec![TYPE_UNKNOWN, TYPE_CONTROL, TYPE_CONTROL];
        for b in 0..=255u8 {
            pieces.push(byte_enc[b as usize].to_string());
            types.push(TYPE_NORMAL);
        }
        pieces.push("[SEP]".into());
        types.push(TYPE_CONTROL);
        pieces.push("[unused0]".into());
        types.push(TYPE_USER_DEFINED);
        let mut ids = HashMap::new();
        for (i, p) in pieces.iter().enumerate() {
            ids.entry(p.clone()).or_insert(i as u32);
        }
        let specials = collect_specials(&pieces, &types);
        let user_specials = collect_user_specials(&pieces, &types);
        let n_vocab = pieces.len();
        let tk = Tokenizer {
            pieces,
            ids,
            special_index: SpecialIndex::build(&specials),
            specials,
            user_index: SpecialIndex::build(&user_specials),
            user_specials,
            bos: 1,
            eos: 2,
            add_bos: true,
            add_eos: true,
            add_sep: true,
            sep: Some(2),
            cls: None,
            mask: None,
            pre: PreType::Gpt2,
            n_vocab,
            algo: TokenizerAlgo::Bpe(BpeData {
                ranks: HashMap::new(),
                byte_enc,
                byte_dec,
            }),
        };
        let byte = |c: char| tk.ids[&c.to_string()];
        // `encode` parses control tokens; the pair path does not.
        assert_eq!(tk.encode("[SEP]"), vec![259]);
        assert_eq!(
            tk.encode_pair("[SEP]", "[unused0]"),
            vec![
                1,
                byte('['),
                byte('S'),
                byte('E'),
                byte('P'),
                byte(']'),
                2,
                2,
                260,
                2
            ]
        );
    }

    #[test]
    fn dbrx_encodes_like_llama3() {
        // The `dbrx` pre-tokenizer is the Llama 3 pattern (llama.cpp v0.9.0
        // src/llama-vocab.cpp lines 310-315), so the two must agree id for id.
        let dbrx = toy_bpe(PreType::Dbrx);
        let llama3 = toy_bpe(PreType::Llama3);
        for text in ["123", "Hello 123 world!", "it's 2026!", "a1b2c3", " 12 "] {
            assert_eq!(
                dbrx.encode(text),
                llama3.encode(text),
                "dbrx and llama-bpe disagree on {text:?}"
            );
        }
        // And it is not the SmolLM pattern: the digits stay together under
        // `dbrx`, so the merge fires and the pair "12" is one token.
        let smollm = toy_bpe(PreType::SmolLm);
        let twelve = dbrx.ids.get("12").copied().unwrap();
        let three = dbrx.ids.get("3").copied().unwrap();
        assert_eq!(dbrx.encode("123"), vec![twelve, three]);
        assert_ne!(smollm.encode("123"), dbrx.encode("123"));
    }

    /// The raw-text scan S4.9c replaced, kept as the differential test's reference:
    /// every special's text tested at every position, in `specials` order, and the
    /// plain run between two specials found by the same per-byte test.
    fn encode_with_per_special_scan(tk: &Tokenizer, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        match &tk.algo {
            TokenizerAlgo::Bpe(bpe) => {
                let mut p = 0usize;
                while p < text.len() {
                    let mut hit = None;
                    for (s, id) in &tk.specials {
                        if !s.is_empty() && text[p..].starts_with(s.as_str()) {
                            hit = Some((s.len(), *id));
                            break;
                        }
                    }
                    if let Some((len, id)) = hit {
                        out.push(id);
                        p += len;
                        continue;
                    }
                    let mut q = p;
                    while q < text.len() {
                        if text.is_char_boundary(q)
                            && tk
                                .specials
                                .iter()
                                .any(|(s, _)| !s.is_empty() && text[q..].starts_with(s.as_str()))
                        {
                            break;
                        }
                        q += 1;
                        while q < text.len() && !text.is_char_boundary(q) {
                            q += 1;
                        }
                    }
                    tk.encode_plain_bpe(&text[p..q], bpe, &mut out);
                    p = q;
                }
            }
            TokenizerAlgo::Spm(spm) => {
                let mut p = 0usize;
                let mut is_prev_special = true;
                let mut scratch = SpmScratch::default();
                let mut buf = String::new();
                while p < text.len() {
                    let mut hit = None;
                    for (s, id) in &tk.specials {
                        if !s.is_empty() && text[p..].starts_with(s.as_str()) {
                            hit = Some((s.len(), *id));
                            break;
                        }
                    }
                    if let Some((len, id)) = hit {
                        out.push(id);
                        p += len;
                        is_prev_special = true;
                        continue;
                    }
                    let mut q = p;
                    while q < text.len() {
                        if text.is_char_boundary(q)
                            && tk
                                .specials
                                .iter()
                                .any(|(s, _)| !s.is_empty() && text[q..].starts_with(s.as_str()))
                        {
                            break;
                        }
                        q += 1;
                        while q < text.len() && !text.is_char_boundary(q) {
                            q += 1;
                        }
                    }
                    tk.encode_fragment_spm(
                        &text[p..q],
                        spm,
                        is_prev_special,
                        &mut buf,
                        &mut scratch,
                        &mut out,
                    );
                    is_prev_special = false;
                    p = q;
                }
            }
            _ => panic!("the differential test covers the BPE and SentencePiece scans only"),
        }
        out
    }

    /// A SentencePiece vocabulary whose specials overlap each other and are
    /// prefixes of one another (`abc`/`ab`/`bc`/`a`, `<s>`/`<s`, `\n\n`/`\n`).
    fn toy_spm_with_specials() -> Tokenizer {
        let specials = [
            "<s>", "</s>", "[MASK]", "<s", "abc", "ab", "bc", "a", "\n\n", "\n", "▁x",
        ];
        let normal = [
            "▁", "b", "c", "x", "y", " ", "é", "你", "s", ">", "[", "]", "M", "A", "S", "K", "<",
            "/",
        ];
        let mut pieces: Vec<String> = Vec::new();
        let mut types: Vec<i32> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        for s in specials {
            pieces.push(s.to_string());
            types.push(TYPE_CONTROL);
            scores.push(0.0);
        }
        pieces.push("<unk>".to_string());
        types.push(TYPE_UNKNOWN);
        scores.push(0.0);
        for (i, p) in normal.iter().enumerate() {
            pieces.push(p.to_string());
            types.push(TYPE_NORMAL);
            scores.push(1.0 + i as f32);
        }
        let unk = (pieces.len() - normal.len() - 1) as u32;
        Tokenizer::from_spm_parts(pieces, scores, types, 0, 1, unk, false)
    }

    /// A byte-level BPE with the same overlapping specials, and every byte in the
    /// vocabulary so a plain run always encodes.
    fn toy_bpe_with_specials() -> Tokenizer {
        let mut tk = toy_bpe(PreType::Qwen2);
        for s in [
            "<s>", "</s>", "[MASK]", "<s", "abc", "ab", "bc", "a", "\n\n", "\n", "▁x",
        ] {
            let id = tk.pieces.len() as u32;
            tk.pieces.push(s.to_string());
            tk.ids.insert(s.to_string(), id);
            tk.specials.push((s.to_string(), id));
        }
        tk.specials
            .sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        tk.special_index = SpecialIndex::build(&tk.specials);
        tk.n_vocab = tk.pieces.len();
        tk
    }

    /// The first-byte lookup against the per-special scan on random strings built
    /// from the specials' own texts, their prefixes and their overlaps (S4.9c).
    #[test]
    fn the_special_lookup_matches_the_per_special_scan() {
        let spm = toy_spm_with_specials();
        let bpe = toy_bpe_with_specials();
        // Each atom is a whole special, a prefix of one, or a byte that can begin
        // one; concatenating them produces the overlaps the scan must resolve.
        let mut atoms: Vec<String> = spm.specials.iter().map(|(s, _)| s.clone()).collect();
        for s in &spm.specials.clone() {
            for cut in 1..s.0.len() {
                if s.0.is_char_boundary(cut) {
                    atoms.push(s.0[..cut].to_string());
                }
            }
        }
        atoms.extend(
            [
                "a", "b", "c", "x", " ", "é", "你", "\n", "s", ">", "<", "[", "]", "M", "▁", "/",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
        let mut x: u64 = 0x9e3779b97f4a7c15;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut checked = 0usize;
        let mut with_special = 0usize;
        for _ in 0..4000 {
            let n = 1 + (next() % 8) as usize;
            let mut text = String::new();
            let mut saw_special = false;
            for _ in 0..n {
                let atom = &atoms[(next() % atoms.len() as u64) as usize];
                if atom.len() > 1 {
                    saw_special = true;
                }
                text.push_str(atom);
            }
            for tk in [&spm, &bpe] {
                let mut want = Vec::new();
                tk.encode_into(&text, &mut want);
                let got = encode_with_per_special_scan(tk, &text);
                assert_eq!(want, got, "special scan differs on {text:?}");
            }
            checked += 1;
            with_special += saw_special as usize;
        }
        assert_eq!(checked, 4000);
        assert!(
            with_special > 3000,
            "only {with_special} strings carried a special token's text"
        );
        // Every whole special text is one token of its own.
        for (s, id) in &spm.specials {
            let mut out = Vec::new();
            spm.encode_into(s, &mut out);
            assert_eq!(out, vec![*id], "{s:?} is not one token");
            let mut out = Vec::new();
            spm.encode_into(&format!("{s}{s}"), &mut out);
            assert!(!out.is_empty(), "{} twice encodes to nothing", s);
        }
    }
}
