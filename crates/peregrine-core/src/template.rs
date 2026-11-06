//! Chat templates, rendered in Rust from the shape the GGUF's own Jinja template
//! describes.
//!
//! The engine does not run Jinja. It recognises templates supported by the engine —
//! ChatML (SmolLM2, Qwen2.5), Llama 3, Qwen3, Gemma, Mistral (v1 and v7), and Llama 2 —
//! from the markers in `tokenizer.chat_template`, and renders them so that the token
//! stream is the one llama.cpp's `--jinja` path produces for the same messages. That
//! equality is what makes a greedy answer from this engine comparable with llama.cpp's
//! on the same file: a prompt that differs by one newline is a different prompt.
//!
//! Anything else is refused at load (`detect`), never approximated with ChatML:
//! a model prompted in a template it was not trained on answers fluently and wrongly.
//!
//! ## BOS handling
//! SentencePiece models (Gemma, Mistral, Llama 2) usually add `<s>` / `<bos>` through the
//! tokenizer (`add_bos`), and some templates also write it. To match what llama.cpp's
//! `--jinja` path sends, the engine ensures that the token stream has exactly one BOS:
//! if the rendered prompt string already begins with BOS (e.g. Llama 3's `<|begin_of_text|>`),
//! no duplicate BOS is inserted; if `add_bos` is true on the tokenizer and the encoded
//! stream does not start with `bos`, `bos` is prepended at prompt encode time (`Tokenizer::encode_prompt`).

use crate::gguf::Gguf;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    /// `<|im_start|>role\n…<|im_end|>\n`
    ChatMl,
    /// `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>`
    Llama3,
    /// Qwen3's ChatML: no default system message, and a reasoning block the model
    /// opens itself — or, with thinking switched off, the template writes an empty
    /// `<think>\n\n</think>\n\n` at the start of the assistant turn.
    Qwen3,
    /// `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`
    /// Gemma 2/3 has no system role; the system text is prepended to the first user turn.
    Gemma,
    /// ` [INST] … [/INST]`, Mistral v1 spacing.
    MistralV1,
    /// `[SYSTEM_PROMPT] …[/SYSTEM_PROMPT][INST] …[/INST]`, Mistral v7 spacing.
    MistralV7,
    /// `[INST] <<SYS>>\n…\n<</SYS>>\n\n… [/INST]`, Llama 2 with system block.
    Llama2,
}

/// Template kinds this crate renders, for the compatibility probe's message.
pub const TEMPLATES: &[&str] = &[
    "chatml",
    "llama3",
    "qwen3",
    "gemma",
    "mistral-v1",
    "mistral-v7",
    "llama2",
];

impl Template {
    pub fn name(self) -> &'static str {
        match self {
            Template::ChatMl => "chatml",
            Template::Llama3 => "llama3",
            Template::Qwen3 => "qwen3",
            Template::Gemma => "gemma",
            Template::MistralV1 => "mistral-v1",
            Template::MistralV7 => "mistral-v7",
            Template::Llama2 => "llama2",
        }
    }

    /// Which template the file's own `tokenizer.chat_template` is, and whether the
    /// vocabulary carries the control tokens it needs. A missing template string is
    /// treated as ChatML only when the ChatML tokens are present.
    pub fn detect(g: &Gguf) -> Result<Template, String> {
        let toks = g.arr_str("tokenizer.ggml.tokens")?;
        let has = |t: &str| toks.iter().any(|x| x == t);
        let tpl = g.str("tokenizer.chat_template").unwrap_or("");
        let kind = Self::detect_from_template(tpl, has)?;
        for t in kind.required_tokens() {
            if !has(t) {
                return Err(format!(
                    "no {t} token — Peregrine renders the {} template",
                    kind.name()
                ));
            }
        }
        Ok(kind)
    }

    /// Detects the template variant from the Jinja template string and a token existence check.
    pub fn detect_from_template(tpl: &str, has: impl Fn(&str) -> bool) -> Result<Template, String> {
        if tpl.contains("<|start_header_id|>") {
            Ok(Template::Llama3)
        } else if tpl.contains("<|im_start|>") && tpl.contains("enable_thinking") {
            Ok(Template::Qwen3)
        } else if tpl.contains("<|im_start|>") || (tpl.is_empty() && has("<|im_start|>")) {
            Ok(Template::ChatMl)
        } else if tpl.contains("<start_of_turn>") || (tpl.is_empty() && has("<start_of_turn>")) {
            Ok(Template::Gemma)
        } else if tpl.contains("[INST]") {
            if tpl.contains("[SYSTEM_PROMPT]") {
                Ok(Template::MistralV7)
            } else if tpl.contains("' [INST] ' + system_message") {
                Ok(Template::MistralV1)
            } else if tpl.contains("[AVAILABLE_TOOLS]") || tpl.contains("[TOOL_CALLS]") {
                Err(format!(
                    "Mistral v3 / tool-calling chat templates are not supported — Peregrine renders {}",
                    TEMPLATES.join(", ")
                ))
            } else if tpl.contains("<<SYS>>") {
                Ok(Template::Llama2)
            } else {
                Err(format!(
                    "unsupported [INST] chat template variant — Peregrine renders {}",
                    TEMPLATES.join(", ")
                ))
            }
        } else if tpl.is_empty() {
            Err(format!(
                "chat template is missing or not recognised — Peregrine renders {}",
                TEMPLATES.join(", ")
            ))
        } else {
            Err(format!(
                "chat template is not recognised — Peregrine renders {}",
                TEMPLATES.join(", ")
            ))
        }
    }

    /// Detects the template variant purely from the Jinja template string.
    pub fn detect_from_str(tpl: &str) -> Result<Template, String> {
        Self::detect_from_template(tpl, |_| false)
    }

    pub fn required_tokens(self) -> &'static [&'static str] {
        match self {
            Template::ChatMl => &["<|im_start|>", "<|im_end|>"],
            Template::Qwen3 => &["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
            Template::Llama3 => &[
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ],
            Template::Gemma => &["<start_of_turn>", "<end_of_turn>"],
            Template::MistralV1 | Template::MistralV7 | Template::Llama2 => &["<s>", "</s>"],
        }
    }

    /// Token ids that end an assistant turn under this template, besides the
    /// tokenizer's own `eos`. Llama 3 instruct models stop on `<|eot_id|>` and may
    /// emit `<|eom_id|>`; `<|end_of_text|>` ends the document.
    pub fn stop_ids(self, tk: &Tokenizer) -> Vec<u32> {
        let names: &[&str] = match self {
            Template::ChatMl | Template::Qwen3 => &["<|im_end|>", "<|endoftext|>"],
            Template::Llama3 => &["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"],
            Template::Gemma => &["<end_of_turn>"],
            Template::MistralV1 | Template::MistralV7 | Template::Llama2 => &["</s>"],
        };
        let mut ids: Vec<u32> = names.iter().filter_map(|n| tk.id_of(n)).collect();
        if !ids.contains(&tk.eos) {
            ids.push(tk.eos);
        }
        ids
    }

    /// The system message the model's own template inserts when the caller gives
    /// none. SmolLM2 and Qwen2.5 each name themselves; Llama 3.2 inserts only its
    /// date header (rendered in `system_block`).
    pub fn default_system(self, model_name: &str) -> Option<&'static str> {
        match self {
            Template::ChatMl if model_name.to_ascii_lowercase().contains("qwen") => {
                Some("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
            }
            Template::ChatMl => {
                Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face")
            }
            Template::Llama3
            | Template::Qwen3
            | Template::Gemma
            | Template::MistralV1
            | Template::MistralV7
            | Template::Llama2 => None,
        }
    }

    /// What the template writes at the start of the assistant's reply before the
    /// model continues: Qwen3 with thinking off gets an empty reasoning block, which
    /// is how its own template (`enable_thinking=false`) switches reasoning off.
    pub fn assistant_open(self, thinking: bool) -> &'static str {
        match self {
            Template::Qwen3 if !thinking => "<think>\n\n</think>\n\n",
            _ => "",
        }
    }

    /// The text that opens a conversation: the system block, before any user turn.
    pub fn system_block(self, system: Option<&str>, today: &str) -> String {
        match self {
            Template::ChatMl | Template::Qwen3 => match system {
                Some(sys) => format!("<|im_start|>system\n{sys}<|im_end|>\n"),
                None => String::new(),
            },
            // Llama 3.2's template always writes the header with the knowledge
            // cut-off and today's date (`strftime_now("%d %b %Y")`), then the system
            // message if any — so an absent system message still costs these lines.
            Template::Llama3 => format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
                 Cutting Knowledge Date: December 2023\nToday Date: {today}\n\n{}<|eot_id|>",
                system.unwrap_or("")
            ),
            Template::Gemma => match system {
                Some(sys) => format!("<start_of_turn>user\n{}\n\n", sys.trim()),
                None => "<start_of_turn>user\n".to_string(),
            },
            Template::MistralV1 => match system {
                Some(sys) => format!(" [INST] {sys}\n\n"),
                None => " [INST] ".to_string(),
            },
            Template::MistralV7 => match system {
                Some(sys) => format!("[SYSTEM_PROMPT] {sys}[/SYSTEM_PROMPT]"),
                None => String::new(),
            },
            Template::Llama2 => match system {
                Some(sys) => format!("[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n"),
                None => "[INST] ".to_string(),
            },
        }
    }

    /// One user message followed by the assistant header the model continues from.
    pub fn user_turn(self, user: &str) -> String {
        match self {
            Template::ChatMl | Template::Qwen3 => {
                format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
            }
            Template::Llama3 => format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            Template::Gemma => {
                format!("{}<end_of_turn>\n<start_of_turn>model\n", user.trim())
            }
            Template::MistralV1 => {
                format!("{user} [/INST]")
            }
            Template::MistralV7 => {
                format!("[INST] {user}[/INST]")
            }
            Template::Llama2 => {
                format!("{user} [/INST]")
            }
        }
    }

    /// A completed assistant message, as it appears in a replayed transcript.
    pub fn assistant_turn(self, text: &str) -> String {
        match self {
            Template::ChatMl | Template::Qwen3 => format!("{text}<|im_end|>\n"),
            Template::Llama3 => format!("{text}<|eot_id|>"),
            Template::Gemma => format!("{}<end_of_turn>\n<start_of_turn>user\n", text.trim()),
            Template::MistralV1 => format!(" {text}</s> [INST] "),
            Template::MistralV7 => format!(" {text}</s>"),
            Template::Llama2 => format!("{text}</s><s>[INST] "),
        }
    }

    /// What closes the previous assistant message when a conversation continues
    /// incrementally: the end-of-turn token was sampled but never fed, so the cache
    /// holds the answer without its terminator.
    pub fn close_previous(self) -> &'static str {
        match self {
            Template::ChatMl | Template::Qwen3 => "<|im_end|>\n",
            Template::Llama3 => "<|eot_id|>",
            Template::Gemma => "<end_of_turn>\n<start_of_turn>user\n",
            Template::MistralV1 => "</s> [INST] ",
            Template::MistralV7 => "</s>",
            Template::Llama2 => "</s><s>[INST] ",
        }
    }
}

/// Today's date as Llama 3's template formats it (`%d %b %Y`, e.g. `14 Oct 2026`),
/// from the system clock, with no dependency: the civil-from-days algorithm.
pub fn today_llama3() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    const MON: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!("{d:02} {} {y}", MON[(m - 1) as usize])
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 → (year, month, day).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper rendering full conversation in the same structure serve::chat_prompt uses.
    fn render_chat_prompt(
        t: Template,
        system: Option<&str>,
        pairs: &[(&str, &str)],
        user: &str,
    ) -> String {
        let mut s = t.system_block(system, "14 Oct 2026");
        for (u, a) in pairs {
            s.push_str(&t.user_turn(u));
            s.push_str(&t.assistant_turn(a));
        }
        s.push_str(&t.user_turn(user));
        s.push_str(t.assistant_open(false));
        s
    }

    #[test]
    fn chatml_renders_the_smollm_transcript_shape() {
        let t = Template::ChatMl;
        let s = t.system_block(Some("S"), "x") + &t.user_turn("U");
        assert_eq!(s, "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(t.assistant_turn("A"), "A<|im_end|>\n");
        assert_eq!(t.close_previous(), "<|im_end|>\n");
        assert_eq!(t.system_block(None, "x"), "");
    }

    #[test]
    fn llama3_always_writes_the_date_header() {
        let t = Template::Llama3;
        let s = t.system_block(None, "14 Oct 2026");
        assert!(s.starts_with("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"));
        assert!(s.contains("Cutting Knowledge Date: December 2023\nToday Date: 14 Oct 2026\n\n"));
        assert!(s.ends_with("\n\n<|eot_id|>"), "{s:?}");
        let s = t.system_block(Some("Be terse."), "14 Oct 2026");
        assert!(
            s.ends_with("Today Date: 14 Oct 2026\n\nBe terse.<|eot_id|>"),
            "{s:?}"
        );
        assert_eq!(t.user_turn("U"),
            "<|start_header_id|>user<|end_header_id|>\n\nU<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        assert_eq!(t.close_previous(), "<|eot_id|>");
    }

    // Expected outputs from llama.cpp's test-chat-template.cpp, v0.9.0 (MIT-licensed).
    // Ground truth conversation:
    //   system: "You are a helpful assistant"
    //   user: "Hello"
    //   assistant: "Hi there"
    //   user: "Who are you"
    //   assistant: "   I am an assistant   "
    //   user: "Another question"
    const PAIRS: [(&str, &str); 2] = [
        ("Hello", "Hi there"),
        ("Who are you", "   I am an assistant   "),
    ];
    const FINAL_USER: &str = "Another question";
    const SYSTEM_MSG: &str = "You are a helpful assistant";

    #[test]
    fn gemma_renders_expected_output_matching_llama_cpp() {
        let t = Template::Gemma;
        let rendered = render_chat_prompt(t, Some(SYSTEM_MSG), &PAIRS, FINAL_USER);
        // expected_output from llama.cpp's test-chat-template.cpp, v0.9.0 ("google/gemma-7b-it")
        let expected = "<start_of_turn>user\n\
                        You are a helpful assistant\n\n\
                        Hello<end_of_turn>\n\
                        <start_of_turn>model\n\
                        Hi there<end_of_turn>\n\
                        <start_of_turn>user\n\
                        Who are you<end_of_turn>\n\
                        <start_of_turn>model\n\
                        I am an assistant<end_of_turn>\n\
                        <start_of_turn>user\n\
                        Another question<end_of_turn>\n\
                        <start_of_turn>model\n";
        assert_eq!(rendered, expected);

        // Turn without system message
        assert_eq!(
            t.system_block(None, "x") + &t.user_turn("Hello"),
            "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        );
        assert_eq!(
            t.assistant_turn("Hi there"),
            "Hi there<end_of_turn>\n<start_of_turn>user\n"
        );
        assert_eq!(t.close_previous(), "<end_of_turn>\n<start_of_turn>user\n");
        assert_eq!(t.assistant_open(false), "");
    }

    #[test]
    fn mistral_v1_renders_expected_output_matching_llama_cpp() {
        let t = Template::MistralV1;
        let rendered = render_chat_prompt(t, Some(SYSTEM_MSG), &PAIRS, FINAL_USER);
        // expected_output from llama.cpp's test-chat-template.cpp, v0.9.0
        // ("mistralai/Mistral-7B-Instruct-v0.2 (mistralai 'v1' template with a system prompt)")
        let expected = " [INST] You are a helpful assistant\n\n\
                        Hello [/INST] Hi there</s> [INST] Who are you [/INST]    \
                        I am an assistant   </s> [INST] Another question [/INST]";
        assert_eq!(rendered, expected);

        // Turn without system message
        assert_eq!(
            t.system_block(None, "x") + &t.user_turn("Hello"),
            " [INST] Hello [/INST]"
        );
        assert_eq!(t.assistant_turn("Hi there"), " Hi there</s> [INST] ");
        assert_eq!(t.close_previous(), "</s> [INST] ");
        assert_eq!(t.assistant_open(false), "");
    }

    #[test]
    fn mistral_v7_renders_expected_output_matching_llama_cpp() {
        let t = Template::MistralV7;
        let rendered = render_chat_prompt(t, Some(SYSTEM_MSG), &PAIRS, FINAL_USER);
        // expected_output from llama.cpp's test-chat-template.cpp, v0.9.0
        // ("mistralai/Mistral-Large-Instruct-2411 (mistralai 'v7' template)")
        let expected = "[SYSTEM_PROMPT] You are a helpful assistant[/SYSTEM_PROMPT]\
                        [INST] Hello[/INST] Hi there</s>\
                        [INST] Who are you[/INST]    I am an assistant   </s>\
                        [INST] Another question[/INST]";
        assert_eq!(rendered, expected);

        // Turn without system message
        assert_eq!(
            t.system_block(None, "x") + &t.user_turn("Hello"),
            "[INST] Hello[/INST]"
        );
        assert_eq!(t.assistant_turn("Hi there"), " Hi there</s>");
        assert_eq!(t.close_previous(), "</s>");
        assert_eq!(t.assistant_open(false), "");
    }

    #[test]
    fn llama2_renders_expected_output_matching_llama_cpp() {
        let t = Template::Llama2;
        let rendered = render_chat_prompt(t, Some(SYSTEM_MSG), &PAIRS, FINAL_USER);
        // expected_output from llama.cpp's test-chat-template.cpp, v0.9.0
        // ("TheBloke/FusionNet_34Bx2_MoE-AWQ")
        let expected = "[INST] <<SYS>>\n\
                        You are a helpful assistant\n\
                        <</SYS>>\n\n\
                        Hello [/INST]Hi there</s><s>[INST] Who are you [/INST]   \
                        I am an assistant   </s><s>[INST] Another question [/INST]";
        assert_eq!(rendered, expected);

        // Turn without system message
        assert_eq!(
            t.system_block(None, "x") + &t.user_turn("Hello"),
            "[INST] Hello [/INST]"
        );
        assert_eq!(t.assistant_turn("Hi there"), "Hi there</s><s>[INST] ");
        assert_eq!(t.close_previous(), "</s><s>[INST] ");
        assert_eq!(t.assistant_open(false), "");
    }

    #[test]
    fn detection_from_real_jinja_templates() {
        // Real templates from llama.cpp's test-chat-template.cpp, v0.9.0 (MIT-licensed):
        let gemma_tmpl = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}";
        assert_eq!(Template::detect_from_str(gemma_tmpl), Ok(Template::Gemma));

        let mistral_v1_tmpl = "{{ bos_token }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- if loop.first and system_message is defined %}{{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}{%- else %}{{- ' [INST] ' + message['content'] + ' [/INST]' }}{%- endif %}{%- elif message['role'] == 'assistant' %}{{- ' ' + message['content'] + eos_token}}{%- endif %}{%- endfor %}";
        assert_eq!(
            Template::detect_from_str(mistral_v1_tmpl),
            Ok(Template::MistralV1)
        );

        let mistral_v7_tmpl = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + '[/INST]' }}{% elif message['role'] == 'system' %}{{ '[SYSTEM_PROMPT] ' + message['content'] + '[/SYSTEM_PROMPT]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% else %}{{ raise_exception('Only user, system and assistant roles are supported!') }}{% endif %}{% endfor %}";
        assert_eq!(
            Template::detect_from_str(mistral_v7_tmpl),
            Ok(Template::MistralV7)
        );

        let llama2_tmpl = "{%- for idx in range(0, messages|length) -%}{%- if messages[idx]['role'] == 'user' -%}{%- if idx > 1 -%}{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}{%- else -%}{{- messages[idx]['content'] + ' [/INST]' -}}{%- endif -%}{% elif messages[idx]['role'] == 'system' %}{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}{%- elif messages[idx]['role'] == 'assistant' -%}{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}{% endif %}{% endfor %}";
        assert_eq!(Template::detect_from_str(llama2_tmpl), Ok(Template::Llama2));

        // Detection from vocab tokens when template string is missing/empty
        assert_eq!(
            Template::detect_from_template("", |t| t == "<start_of_turn>"),
            Ok(Template::Gemma)
        );
        assert_eq!(
            Template::detect_from_template("", |t| t == "<|im_start|>"),
            Ok(Template::ChatMl)
        );
    }

    #[test]
    fn unsupported_templates_are_refused() {
        // Mistral v3 / tool-calling template
        let v3_tmpl = "{{ bos_token }}[AVAILABLE_TOOLS][{\"type\": \"function\"}][/AVAILABLE_TOOLS][INST] Hello [/INST]";
        let err = Template::detect_from_str(v3_tmpl).unwrap_err();
        assert!(
            err.contains("Mistral v3 / tool-calling chat templates are not supported"),
            "{err}"
        );

        // Unknown [INST] variant
        let unknown_inst = "[INST] Hello [/INST]";
        let err = Template::detect_from_str(unknown_inst).unwrap_err();
        assert!(
            err.contains("unsupported [INST] chat template variant"),
            "{err}"
        );

        // Empty template with no recognisable tokens
        let err = Template::detect_from_template("", |_| false).unwrap_err();
        assert!(
            err.contains("chat template is missing or not recognised"),
            "{err}"
        );
    }

    #[test]
    fn default_system_follows_the_model_family() {
        assert!(Template::ChatMl
            .default_system("Qwen2.5 0.5B Instruct")
            .unwrap()
            .contains("Qwen"));
        assert!(Template::ChatMl
            .default_system("SmolLM2 360M Instruct")
            .unwrap()
            .contains("SmolLM"));
        assert!(Template::Llama3
            .default_system("Llama 3.2 1B Instruct")
            .is_none());
        assert!(Template::Gemma.default_system("Gemma 2 2B").is_none());
        assert!(Template::MistralV1.default_system("Mistral 7B").is_none());
        assert!(Template::MistralV7
            .default_system("Mistral Large")
            .is_none());
        assert!(Template::Llama2.default_system("Llama 2 7B").is_none());
    }

    #[test]
    fn civil_dates_are_right_across_leap_years_and_epoch() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        assert_eq!(civil_from_days(20_740), (2026, 10, 14));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        let today = today_llama3();
        assert_eq!(today.len(), 11, "{today:?}");
        assert!(
            today.ends_with("202")
                || today.ends_with("203")
                || today.chars().rev().take(4).all(|c| c.is_ascii_digit())
        );
    }
}
