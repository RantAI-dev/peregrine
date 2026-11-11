//! Minimal GPT-2 byte-level BPE vocabulary and chat template definitions.

use super::writer::GgufWriter;

pub const VOCAB_SIZE: usize = 320;
pub const TYPE_NORMAL: i32 = 1;
pub const TYPE_CONTROL: i32 = 3;

#[derive(Debug, Clone, Copy)]
pub enum TemplateKind {
    ChatMl,
    Qwen3,
    Llama3,
    Gemma,
}

/// The 256 GPT-2 byte pieces mapping each byte 0..=255 to a distinct printable char.
pub fn byte_alphabet() -> [char; 256] {
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
    enc
}

/// Builds the vocabulary tokens, token types, and merges.
pub fn build_vocab() -> (Vec<String>, Vec<i32>, Vec<String>) {
    let alpha = byte_alphabet();
    let mut tokens = Vec::with_capacity(VOCAB_SIZE);
    let mut types = Vec::with_capacity(VOCAB_SIZE);

    // 0..256: byte tokens
    for &c in &alpha {
        tokens.push(c.to_string());
        types.push(TYPE_NORMAL);
    }

    // 256..265: special tokens
    let specials = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<think>",
        "</think>",
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ];
    for &s in &specials {
        tokens.push(s.to_string());
        types.push(TYPE_CONTROL);
    }

    // Common BPE merges
    let space = alpha[b' ' as usize].to_string();
    let merge_pairs = [
        (space.clone(), alpha[b't' as usize].to_string()),
        (
            alpha[b't' as usize].to_string(),
            alpha[b'h' as usize].to_string(),
        ),
        (
            alpha[b'h' as usize].to_string(),
            alpha[b'e' as usize].to_string(),
        ),
        (
            alpha[b'i' as usize].to_string(),
            alpha[b'n' as usize].to_string(),
        ),
        (
            alpha[b'e' as usize].to_string(),
            alpha[b'r' as usize].to_string(),
        ),
        (
            alpha[b'a' as usize].to_string(),
            alpha[b'n' as usize].to_string(),
        ),
        (
            alpha[b'r' as usize].to_string(),
            alpha[b'e' as usize].to_string(),
        ),
        (
            alpha[b'o' as usize].to_string(),
            alpha[b'n' as usize].to_string(),
        ),
        (
            alpha[b'a' as usize].to_string(),
            alpha[b't' as usize].to_string(),
        ),
        (
            alpha[b'e' as usize].to_string(),
            alpha[b'n' as usize].to_string(),
        ),
    ];

    let mut merges = Vec::new();
    for (l, r) in &merge_pairs {
        merges.push(format!("{l} {r}"));
        tokens.push(format!("{l}{r}"));
        types.push(TYPE_NORMAL);
    }

    // Fill remaining tokens up to VOCAB_SIZE
    let curr = tokens.len();
    for i in curr..VOCAB_SIZE {
        tokens.push(format!("extra_token_{i}"));
        types.push(TYPE_NORMAL);
    }

    (tokens, types, merges)
}

/// Returns a chat template string that `Template::detect` recognises.
pub fn chat_template(kind: TemplateKind) -> &'static str {
    match kind {
        TemplateKind::ChatMl => {
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + \
             message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}\
             {{ '<|im_start|>assistant\n' }}{% endif %}"
        }
        TemplateKind::Qwen3 => {
            "{% if enable_thinking %}{% endif %}{% for message in messages %}{{'<|im_start|>' + \
             message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
        }
        TemplateKind::Llama3 => {
            "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + \
             '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}{% endfor %}"
        }
        TemplateKind::Gemma => {
            "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
        }
    }
}

/// Writes gpt-oss tokenizer metadata: the `o200k_harmony` pre-tokenizer, the
/// Harmony control tokens in the vocabulary, and `<|return|>` as EOS — the shape
/// the published gpt-oss files have.
pub fn add_gpt_oss_tokenizer_metadata(w: &mut GgufWriter) {
    let (mut tokens, mut types, merges) = build_vocab();
    let specials = [
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
    ];
    let first_extra = tokens
        .iter()
        .position(|t| t.starts_with("extra_token_"))
        .expect("the synthetic vocabulary has filler tokens");
    for (i, s) in specials.iter().enumerate() {
        tokens[first_extra + i] = (*s).to_string();
        types[first_extra + i] = TYPE_CONTROL;
    }
    let ret = (first_extra + specials.len() - 1) as u32;
    w.str("tokenizer.ggml.model", "gpt2");
    w.str("tokenizer.ggml.pre", "gpt-4o");
    w.arr_str("tokenizer.ggml.tokens", &tokens);
    w.arr_i32("tokenizer.ggml.token_type", &types);
    w.arr_str("tokenizer.ggml.merges", &merges);
    w.u32("tokenizer.ggml.bos_token_id", 258); // <|endoftext|>
    w.u32("tokenizer.ggml.eos_token_id", ret); // <|return|>
    w.u32("tokenizer.ggml.padding_token_id", 258);
    w.bool("tokenizer.ggml.add_bos_token", false);
}

/// Writes Gemma tokenizer metadata to a `GgufWriter`.
pub fn add_gemma_tokenizer_metadata(w: &mut GgufWriter) {
    let (mut tokens, types, merges) = build_vocab();
    tokens[VOCAB_SIZE - 2] = "<start_of_turn>".to_string();
    tokens[VOCAB_SIZE - 1] = "<end_of_turn>".to_string();
    w.str("tokenizer.ggml.model", "gpt2");
    w.str("tokenizer.ggml.pre", "qwen2");
    w.arr_str("tokenizer.ggml.tokens", &tokens);
    w.arr_i32("tokenizer.ggml.token_type", &types);
    w.arr_str("tokenizer.ggml.merges", &merges);
    w.u32("tokenizer.ggml.bos_token_id", (VOCAB_SIZE - 2) as u32);
    w.u32("tokenizer.ggml.eos_token_id", (VOCAB_SIZE - 1) as u32);
    w.u32("tokenizer.ggml.padding_token_id", 0);
    w.bool("tokenizer.ggml.add_bos_token", false);
}

/// Writes tokenizer metadata to a `GgufWriter`.
pub fn add_tokenizer_metadata(w: &mut GgufWriter, pre: &str) {    let (tokens, types, merges) = build_vocab();
    w.str("tokenizer.ggml.model", "gpt2");
    w.str("tokenizer.ggml.pre", pre);
    w.arr_str("tokenizer.ggml.tokens", &tokens);
    w.arr_i32("tokenizer.ggml.token_type", &types);
    w.arr_str("tokenizer.ggml.merges", &merges);
    w.u32("tokenizer.ggml.bos_token_id", 258); // <|endoftext|>
    w.u32("tokenizer.ggml.eos_token_id", 257); // <|im_end|>
    w.u32("tokenizer.ggml.padding_token_id", 258);
    w.bool("tokenizer.ggml.add_bos_token", false);
}

/// The vocabulary size of the synthetic encoders (`bert` and unigram): a multiple
/// of 32 so a Q8_0 token-embedding row quantises in whole blocks.
pub const BERT_VOCAB_SIZE: usize = 320;
pub const TYPE_UNKNOWN: i32 = 2;

/// A tiny WordPiece vocabulary in the shape llama.cpp's BERT converter stores:
/// `[PAD]`/`[UNK]`/`[CLS]`/`[SEP]`/`[MASK]` first, ordinary words with the `▁`
/// word-start the converter writes, `##` continuations, and punctuation as its
/// own word. The filler keeps the size at [`BERT_VOCAB_SIZE`].
pub fn bert_vocab() -> (Vec<String>, Vec<i32>) {
    let mut tokens: Vec<String> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let mut types = vec![TYPE_CONTROL, TYPE_UNKNOWN, TYPE_CONTROL, TYPE_CONTROL, TYPE_CONTROL];
    for word in [
        "▁the", "▁a", "▁test", "▁hello", "▁world", "▁text", "▁of", "▁and", "▁to", "▁in", "▁is",
        "▁for", "▁on", "▁it", "▁with", "▁as", "▁at", "▁this", "▁be", "▁by", "▁capitol", "▁france",
        "▁paris", "▁city", "▁what", "▁one", "▁two", "▁three",
    ] {
        tokens.push(word.to_string());
        types.push(TYPE_NORMAL);
    }
    for cont in ["##s", "##ing", "##ed", "##er", "##ly", "##tion", "##al", "##ness", "##est"] {
        tokens.push(cont.to_string());
        types.push(TYPE_NORMAL);
    }
    for punct in ["▁,", "▁.", "▁!", "▁?", "▁-", "▁'", "▁\"", "▁(", "▁)"] {
        tokens.push(punct.to_string());
        types.push(TYPE_NORMAL);
    }
    for d in 0..10 {
        tokens.push(format!("▁{d}"));
        types.push(TYPE_NORMAL);
    }
    let mut i = 0;
    while tokens.len() < BERT_VOCAB_SIZE {
        tokens.push(format!("▁filler{i}"));
        types.push(TYPE_NORMAL);
        i += 1;
    }
    (tokens, types)
}

/// Writes a `bert` WordPiece tokenizer: the vocabulary, its token types, and the
/// special-token ids llama.cpp's converter reads from the HF config. `add_bos`
/// is what makes [`peregrine_core::tokenizer::Tokenizer::encode_special`] wrap the
/// text in `[CLS] … [SEP]` (llama.cpp's WPM branch gates on it).
pub fn add_bert_tokenizer_metadata(w: &mut GgufWriter) {
    let (tokens, types) = bert_vocab();
    w.str("tokenizer.ggml.model", "bert");
    w.arr_str("tokenizer.ggml.tokens", &tokens);
    w.arr_i32("tokenizer.ggml.token_type", &types);
    w.u32("tokenizer.ggml.bos_token_id", 2); // [CLS]
    w.u32("tokenizer.ggml.unknown_token_id", 1); // [UNK]
    w.u32("tokenizer.ggml.seperator_token_id", 3); // [SEP]
    w.u32("tokenizer.ggml.cls_token_id", 2);
    w.u32("tokenizer.ggml.mask_token_id", 4);
    w.u32("tokenizer.ggml.padding_token_id", 0);
    w.bool("tokenizer.ggml.add_bos_token", true);
    w.bool("tokenizer.ggml.add_eos_token", false);
    w.u32("tokenizer.ggml.token_type_count", 2);
}

/// Writes a SentencePiece-unigram (`tokenizer.ggml.model = "t5"`) vocabulary, the
/// XLM-R shape: `<unk>`, `<s>`, `</s>`, `<pad>`, the `▁`-prefixed pieces above,
/// single characters so any string tokenises, and scores. `<s>`/`</s>` are the
/// file's BOS/EOS, both added (`add_bos`/`add_eos`), as an XLM-R file does.
pub fn add_unigram_tokenizer_metadata(w: &mut GgufWriter) {
    let mut tokens: Vec<String> = ["<unk>", "<s>", "</s>", "<pad>"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let mut types = vec![TYPE_UNKNOWN, TYPE_CONTROL, TYPE_CONTROL, TYPE_CONTROL];
    let mut scores = vec![0.0f32, 0.0, 0.0, 0.0];
    let add = |tokens: &mut Vec<String>, types: &mut Vec<i32>, scores: &mut Vec<f32>, t: &str, s: f32| {
        tokens.push(t.to_string());
        types.push(TYPE_NORMAL);
        scores.push(s);
    };
    for (i, word) in [
        "▁the", "▁a", "▁test", "▁hello", "▁world", "▁text", "▁of", "▁and", "▁to", "▁in", "▁is",
        "▁for", "▁on", "▁it", "▁with", "▁as", "▁at", "▁this", "▁be", "▁by", "▁capitol", "▁france",
        "▁paris", "▁city", "▁what", "▁one", "▁two", "▁three",
    ]
    .iter()
    .enumerate()
    {
        add(&mut tokens, &mut types, &mut scores, word, -2.0 - i as f32);
    }
    for (i, c) in "abcdefghijklmnopqrstuvwxyz"
        .chars()
        .chain("0123456789".chars())
        .chain([',', '.', '!', '?', '-', '\''].into_iter())
        .enumerate()
    {
        add(&mut tokens, &mut types, &mut scores, &format!("{c}"), -20.0 - i as f32);
    }
    let mut i = 0;
    while tokens.len() < BERT_VOCAB_SIZE {
        add(
            &mut tokens,
            &mut types,
            &mut scores,
            &format!("▁filler{i}"),
            -40.0 - i as f32,
        );
        i += 1;
    }
    w.str("tokenizer.ggml.model", "t5");
    w.arr_str("tokenizer.ggml.tokens", &tokens);
    w.arr_i32("tokenizer.ggml.token_type", &types);
    w.arr_f32("tokenizer.ggml.scores", &scores);
    w.u32("tokenizer.ggml.bos_token_id", 1); // <s>
    w.u32("tokenizer.ggml.eos_token_id", 2); // </s>
    w.u32("tokenizer.ggml.unknown_token_id", 0);
    w.u32("tokenizer.ggml.padding_token_id", 3);
    w.u32("tokenizer.ggml.token_type_count", 2);
    w.bool("tokenizer.ggml.add_bos_token", true);
    w.bool("tokenizer.ggml.add_eos_token", true);
    w.bool("tokenizer.ggml.add_space_prefix", true);
    w.bool("tokenizer.ggml.remove_extra_whitespaces", true);
}
