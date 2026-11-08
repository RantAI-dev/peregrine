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
pub fn add_tokenizer_metadata(w: &mut GgufWriter, pre: &str) {
    let (tokens, types, merges) = build_vocab();
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
