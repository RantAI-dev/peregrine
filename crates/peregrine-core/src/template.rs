//! Chat templates, rendered in Rust from the shape the GGUF's own Jinja template
//! describes.
//!
//! The engine does not run Jinja. It recognises templates supported by the engine —
//! ChatML (SmolLM2, Qwen2.5), Llama 3, Qwen3, Gemma, Mistral (v0, v1 and v7),
//! Llama 2 (with and without a system block), and OpenAI's Harmony (gpt-oss) — from the markers in
//! `tokenizer.chat_template`, and renders them so that the token stream is the one
//! llama.cpp's `--jinja` path produces for the same messages. That equality is what
//! makes a greedy answer from this engine comparable with llama.cpp's on the same
//! file: a prompt that differs by one newline is a different prompt.
//!
//! Anything else is refused at load (`detect`), never approximated with ChatML:
//! a model prompted in a template it was not trained on answers fluently and wrongly.
//! A file that declares no template at all is refused too; the caller names the
//! template it was trained with (`--chat-template`, see [`Template::from_name`]),
//! and that choice is the only one the engine will make.
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
    /// Phi-4's ChatML: `<|im_start|>role<|im_sep|>…<|im_end|>`, with no newlines
    /// between the parts (llama.cpp's `phi4`).
    Phi4,
    /// `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>`
    Llama3,
    /// Qwen3's ChatML: no default system message, and a reasoning block the model
    /// opens itself — or, with thinking switched off, the template writes an empty
    /// `<think>\n\n</think>\n\n` at the start of the assistant turn.
    Qwen3,
    /// Qwen 3.5's ChatML, which Bonsai 2 files carry too: the template opens the
    /// reasoning block itself at the end of the prompt (`<think>\n`, or the empty
    /// block with thinking off), trims every message, and drops the thinking of
    /// earlier turns. Recognised by the template's exact text, one of the published
    /// ones (`crate::tools::ToolFormat::for_template`); its prompt is the tool
    /// module's renderer with no tools, which is checked against llama.cpp.
    Qwen35,
    /// `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`
    /// Gemma 2/3 has no system role; the system text is prepended to the first user turn.
    Gemma,
    /// ` [INST] … [/INST]`, Mistral v1 spacing.
    MistralV1,
    /// `[INST] … [/INST]`, the pre-update Mistral template: no system branch, and
    /// no space before the `[INST]` marker. The system message is folded into the
    /// first user turn, as llama.cpp folds it.
    MistralV0,
    /// `[SYSTEM_PROMPT] …[/SYSTEM_PROMPT][INST] …[/INST]`, Mistral v7 spacing.
    MistralV7,
    /// `[INST] <<SYS>>\n…\n<</SYS>>\n\n… [/INST]`, Llama 2 with a system block
    /// (llama.cpp's `llama2-sys-bos`).
    Llama2,
    /// `[INST] … [/INST]`, the plain Llama 2 format: no system block, the system
    /// message folded into the first user turn (llama.cpp's `llama2`).
    Llama2Plain,
    /// OpenAI's Harmony format, the one gpt-oss's own Jinja template renders:
    /// `<|start|>role<|message|>…<|end|>`, assistant channels
    /// (`<|channel|>final<|message|>`), a fixed system block, and a developer
    /// block for the caller's system message.
    Harmony,
}

/// Template kinds this crate renders, for the compatibility probe's message and
/// for `--chat-template`.
pub const TEMPLATES: &[&str] = &[
    "chatml",
    "phi4",
    "llama3",
    "qwen3",
    "qwen35",
    "gemma",
    "mistral-v0",
    "mistral-v1",
    "mistral-v7",
    "llama2",
    "llama2-sys-bos",
    "harmony",
];

impl Template {
    pub fn name(self) -> &'static str {
        match self {
            Template::ChatMl => "chatml",
            Template::Phi4 => "phi4",
            Template::Llama3 => "llama3",
            Template::Qwen3 => "qwen3",
            Template::Qwen35 => "qwen35",
            Template::Gemma => "gemma",
            Template::MistralV0 => "mistral-v0",
            Template::MistralV1 => "mistral-v1",
            Template::MistralV7 => "mistral-v7",
            Template::Llama2 => "llama2-sys-bos",
            Template::Llama2Plain => "llama2",
            Template::Harmony => "harmony",
        }
    }

    /// The template the caller named (`--chat-template`), by the name in
    /// [`TEMPLATES`]. `None` for a name this engine does not render.
    pub fn from_name(name: &str) -> Option<Template> {
        Some(match name {
            "chatml" => Template::ChatMl,
            "phi4" => Template::Phi4,
            "llama3" => Template::Llama3,
            "qwen3" => Template::Qwen3,
            "qwen35" => Template::Qwen35,
            "gemma" => Template::Gemma,
            "mistral-v0" => Template::MistralV0,
            "mistral-v1" => Template::MistralV1,
            "mistral-v7" => Template::MistralV7,
            "llama2" => Template::Llama2Plain,
            "llama2-sys-bos" => Template::Llama2,
            "harmony" => Template::Harmony,
            _ => return None,
        })
    }

    /// Which template the file's own `tokenizer.chat_template` is, and whether the
    /// vocabulary carries the control tokens it needs. A missing template string is
    /// treated as ChatML only when the ChatML tokens are present.
    pub fn detect(g: &Gguf) -> Result<Template, String> {
        let toks = g.arr_str("tokenizer.ggml.tokens")?;
        let has = |t: &str| toks.iter().any(|x| x == t);
        let tpl = g.str("tokenizer.chat_template").unwrap_or("");
        let kind = Self::detect_from_template(tpl, has)?;
        kind.check_tokens(&has)?;
        Ok(kind)
    }

    /// Whether the vocabulary carries the control tokens this template needs —
    /// the check a caller-chosen template (`--chat-template`) also needs.
    pub fn tokens_present(self, g: &Gguf) -> Result<(), String> {
        let toks = g.arr_str("tokenizer.ggml.tokens")?;
        self.check_tokens(&|t| toks.iter().any(|x| x == t))
    }

    fn check_tokens(self, has: &impl Fn(&str) -> bool) -> Result<(), String> {
        for t in self.required_tokens() {
            if !has(t) {
                return Err(format!(
                    "no {t} token — Peregrine renders the {} template",
                    self.name()
                ));
            }
        }
        Ok(())
    }

    /// Detects the template variant from the Jinja template string and a token existence check.
    pub fn detect_from_template(tpl: &str, has: impl Fn(&str) -> bool) -> Result<Template, String> {
        if tpl.contains("<|start_header_id|>") {
            Ok(Template::Llama3)
        } else if tpl.contains("<|start|>") && tpl.contains("<|channel|>") {
            // gpt-oss's own Jinja template. llama.cpp's heuristic detector reads the
            // same pair of markers (`src/llama-chat.cpp:222-223`, v0.9.0).
            Ok(Template::Harmony)
        } else if tpl.contains("<|im_start|>") && tpl.contains("<|im_sep|>") {
            // llama.cpp v0.9.0 src/llama-chat.cpp:99-101.
            Ok(Template::Phi4)
        } else if crate::tools::ToolFormat::for_template(tpl)
            == Some(crate::tools::ToolFormat::QwenXml)
        {
            // One of the published texts of Qwen 3.5's template, exactly. Any other
            // template with a thinking switch is read as Qwen 3's below.
            Ok(Template::Qwen35)
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
            } else if tpl.contains("'[INST] '")
                && tpl.contains("Only user and assistant roles are supported!")
            {
                // The pre-update Mistral template: `mistralai/Mistral-7B-Instruct-v0.2`
                // as every mirror converted it before Mistral's template update
                // (TheBloke's uploads among them). It has no system branch, and no
                // space before the `[INST]` marker.
                Ok(Template::MistralV0)
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
            Template::Phi4 => &["<|im_start|>", "<|im_sep|>", "<|im_end|>"],
            Template::Qwen3 | Template::Qwen35 => {
                &["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
            }
            Template::Llama3 => &[
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ],
            Template::Gemma => &["<start_of_turn>", "<end_of_turn>"],
            Template::MistralV0
            | Template::MistralV1
            | Template::MistralV7
            | Template::Llama2
            | Template::Llama2Plain => &["<s>", "</s>"],
            Template::Harmony => &[
                "<|start|>",
                "<|message|>",
                "<|end|>",
                "<|channel|>",
                "<|return|>",
            ],
        }
    }

    /// Token ids that end an assistant turn under this template, besides the
    /// tokenizer's own `eos`. Llama 3 instruct models stop on `<|eot_id|>` and may
    /// emit `<|eom_id|>`; `<|end_of_text|>` ends the document.
    pub fn stop_ids(self, tk: &Tokenizer) -> Vec<u32> {
        let names: &[&str] = match self {
            Template::ChatMl | Template::Phi4 | Template::Qwen3 | Template::Qwen35 => {
                &["<|im_end|>", "<|endoftext|>"]
            }
            Template::Llama3 => &["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"],
            Template::Gemma => &["<end_of_turn>"],
            Template::MistralV0
            | Template::MistralV1
            | Template::MistralV7
            | Template::Llama2
            | Template::Llama2Plain => &["</s>"],
            // A Harmony reply ends on `<|return|>` (the answer) or `<|call|>` (a tool
            // call). `<|end|>` is not a stop: it closes the analysis message, and the
            // final answer follows it in the same reply. llama.cpp v0.9.0 drops
            // `<|end|>` from the end-of-generation set when the vocabulary has
            // `<|return|>` and `<|call|>` (`src/llama-vocab.cpp:3345-3383`).
            Template::Harmony => &["<|return|>", "<|call|>"],
        };
        let mut ids: Vec<u32> = names.iter().filter_map(|n| tk.id_of(n)).collect();
        let eos_is_end = self == Template::Harmony && tk.id_of("<|end|>") == Some(tk.eos);
        if !ids.contains(&tk.eos) && !eos_is_end {
            ids.push(tk.eos);
        }
        ids
    }

    /// The system message the model's own template inserts when the caller gives
    /// none. SmolLM2 and Qwen2.5 each name themselves; Llama 3.2 inserts only its
    /// date header (rendered in `system_block`). A model that names neither gets
    /// none: its own template writes no system message, and one invented for it
    /// would be a stranger's sentence.
    pub fn default_system(self, model_name: &str) -> Option<&'static str> {
        match self {
            Template::ChatMl => {
                let name = model_name.to_ascii_lowercase();
                if name.contains("qwen") {
                    Some("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
                } else if name.contains("smollm") {
                    Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face")
                } else {
                    None
                }
            }
            Template::Llama3
            | Template::Phi4
            | Template::Qwen3
            | Template::Qwen35
            | Template::Gemma
            | Template::MistralV0
            | Template::MistralV1
            | Template::MistralV7
            | Template::Llama2
            | Template::Llama2Plain
            | Template::Harmony => None,
        }
    }

    /// What the template writes at the start of the assistant's reply before the
    /// model continues: Qwen3 with thinking off gets an empty reasoning block, which
    /// is how its own template (`enable_thinking=false`) switches reasoning off.
    pub fn assistant_open(self, thinking: bool) -> &'static str {
        match self {
            Template::Qwen3 if !thinking => "<think>\n\n</think>\n\n",
            // Qwen 3.5's template opens the block itself, and closes it at once when
            // thinking is off.
            Template::Qwen35 if thinking => "<think>\n",
            Template::Qwen35 => "<think>\n\n</think>\n\n",
            _ => "",
        }
    }

    /// Whether the prompt ends inside a reasoning block the template opened, so the
    /// model's reply begins with its reasoning and has no opening tag of its own.
    pub fn opens_reasoning(self, thinking: bool) -> bool {
        self == Template::Qwen35 && thinking
    }

    /// The text that opens a conversation: the system block, before any user turn.
    ///
    /// Empty for the templates with no system branch of their own: there the
    /// system message is folded into the first user turn (`system_fold`).
    pub fn system_block(self, system: Option<&str>, today: &str) -> String {
        match self {
            Template::ChatMl | Template::Qwen3 => match system {
                Some(sys) => format!("<|im_start|>system\n{sys}<|im_end|>\n"),
                None => String::new(),
            },
            // Qwen 3.5's template trims every message.
            Template::Qwen35 => match system {
                Some(sys) => format!(
                    "<|im_start|>system\n{}<|im_end|>\n",
                    crate::tools::jinja_trim(sys)
                ),
                None => String::new(),
            },
            Template::Phi4 => match system {
                Some(sys) => format!("<|im_start|>system<|im_sep|>{sys}<|im_end|>"),
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
            // The fixed system block the gpt-oss Jinja template always writes, then
            // the caller's system text as a developer message. The template renders
            // messages[0] with role system/developer through the developer block.
            Template::Harmony => {
                let date = harmony_date(today);
                let mut s = format!(
                    "<|start|>system<|message|>You are ChatGPT, a large language model trained \
                     by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: {date}\n\n\
                     Reasoning: medium\n\n# Valid channels: analysis, commentary, final. \
                     Channel must be included for every message.<|end|>"
                );
                if let Some(sys) = system {
                    s.push_str(&format!(
                        "<|start|>developer<|message|># Instructions\n\n{sys}\n\n<|end|>"
                    ));
                }
                s
            }
            // No system branch: the message is folded into the first user turn.
            Template::MistralV0 | Template::Llama2Plain => String::new(),
        }
    }

    /// The system message to fold into the first user turn, for the templates with
    /// no system branch of their own; `None` for the templates that write the
    /// system message themselves.
    ///
    /// This is what llama.cpp does with such a template (`workaround::
    /// system_message_not_supported` in v0.9.0's `common/chat.cpp`): the system
    /// text and the first user text are joined with a newline, inside the first
    /// user turn, because the Jinja template itself refuses a system role.
    pub fn system_fold(self, system: Option<&str>) -> Option<&str> {
        match self {
            Template::MistralV0 | Template::Llama2Plain => system,
            _ => None,
        }
    }

    /// One user message followed by the assistant header the model continues from.
    pub fn user_turn(self, user: &str) -> String {
        match self {
            Template::ChatMl | Template::Qwen3 => {
                format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
            }
            Template::Qwen35 => format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                crate::tools::jinja_trim(user)
            ),
            Template::Phi4 => {
                format!("<|im_start|>user<|im_sep|>{user}<|im_end|><|im_start|>assistant<|im_sep|>")
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
            Template::MistralV0 | Template::Llama2Plain => {
                format!("[INST] {user} [/INST]")
            }
            // The user message, then the generation prompt's role header. The model
            // writes its own `<|channel|>…<|message|>` after it.
            Template::Harmony => {
                format!("<|start|>user<|message|>{user}<|end|><|start|>assistant")
            }
        }
    }

    /// The final-channel text in a Harmony reply. gpt-oss writes an analysis channel
    /// before the final one; its own Jinja keeps only the final channel when an earlier
    /// assistant turn is replayed ("CoT is dropped during all previous turns", the
    /// template's `else` branch), so a replayed transcript must do the same.
    fn harmony_final_text(text: &str) -> &str {
        const FINAL: &str = "<|channel|>final<|message|>";
        let body = match text.rfind(FINAL) {
            Some(i) => &text[i + FINAL.len()..],
            // A reply with only an analysis channel has no final text to replay.
            None if text.contains("<|channel|>analysis<|message|>") => "",
            None => text,
        };
        let end = [body.find("<|return|>"), body.find("<|end|>")]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(body.len());
        &body[..end]
    }

    /// A completed assistant message, as it appears in a replayed transcript.
    pub fn assistant_turn(self, text: &str) -> String {
        match self {
            Template::ChatMl | Template::Qwen3 => format!("{text}<|im_end|>\n"),
            // An earlier turn is written without its thinking: what follows the last
            // closing tag, trimmed (the template's own split).
            Template::Qwen35 => {
                let text = crate::tools::jinja_trim(text);
                let answer = match text.rfind("</think>") {
                    Some(at) => text[at + "</think>".len()..].trim_start_matches('\n'),
                    None => text,
                };
                format!("{answer}<|im_end|>\n")
            }
            Template::Phi4 => format!("{text}<|im_end|>"),
            Template::Llama3 => format!("{text}<|eot_id|>"),
            Template::Gemma => format!("{}<end_of_turn>\n<start_of_turn>user\n", text.trim()),
            Template::MistralV1 => format!(" {text}</s> [INST] "),
            Template::MistralV7 => format!(" {text}</s>"),
            Template::Llama2 => format!("{text}</s><s>[INST] "),
            Template::MistralV0 | Template::Llama2Plain => format!("{text}</s>"),
            // The assistant header came from `user_turn`; a non-final answer closes
            // with `<|end|>`, as gpt-oss's template writes an earlier turn.
            Template::Harmony => format!(
                "<|channel|>final<|message|>{}<|end|>",
                Self::harmony_final_text(text)
            ),
        }
    }

    /// A final assistant message with no generation prompt after it: Harmony closes
    /// it with `<|return|>`, the token that ends a generation. Every other template
    /// writes the same terminator as [`Template::assistant_turn`].
    pub fn assistant_final(self, text: &str) -> String {
        match self {
            Template::Harmony => format!(
                "<|channel|>final<|message|>{}<|return|>",
                Self::harmony_final_text(text)
            ),
            _ => self.assistant_turn(text),
        }
    }

    /// Whether a continuing conversation must be replayed from its transcript rather
    /// than extended incrementally.
    ///
    /// gpt-oss's template drops every earlier assistant turn's analysis channel
    /// (keeping only the final one), so a prompt built by appending to the cache of
    /// the reply just generated would keep text the full render does not. Harmony
    /// therefore rebuilds the transcript; the other templates extend the cache.
    pub fn replays_transcript(self) -> bool {
        self == Template::Harmony
    }

    /// What closes the previous assistant message when a conversation continues
    /// incrementally: the end-of-turn token was sampled but never fed, so the cache
    /// holds the answer without its terminator.
    pub fn close_previous(self) -> &'static str {
        match self {
            Template::ChatMl | Template::Qwen3 | Template::Qwen35 => "<|im_end|>\n",
            Template::Phi4 => "<|im_end|>",
            Template::Llama3 => "<|eot_id|>",
            Template::Gemma => "<end_of_turn>\n<start_of_turn>user\n",
            Template::MistralV1 => "</s> [INST] ",
            Template::MistralV7 => "</s>",
            Template::Llama2 => "</s><s>[INST] ",
            Template::MistralV0 | Template::Llama2Plain => "</s>",
            // The generation ended on `<|return|>`, which is never fed back; the
            // replayed transcript closes the assistant message with `<|end|>`.
            Template::Harmony => "<|end|>",
        }
    }

    /// The prompt of one chat turn: the system block, the earlier exchanges, this
    /// turn's message, and the assistant's opening (plus whatever the assistant
    /// has already written, for a continuation).
    ///
    /// One builder for both paths, so a prompt served by the batching engine is
    /// the same string a single session would have built, token for token.
    pub fn render(
        self,
        system: Option<&str>,
        pairs: &[(String, String)],
        user: &str,
        today: &str,
        thinking: bool,
        assistant_prefix: Option<&str>,
    ) -> String {
        let mut s = self.system_block(system, today);
        // The system message, folded into the first user turn for the templates
        // with no system branch of their own (llama.cpp folds it the same way).
        let mut fold = self.system_fold(system);
        for (u, a) in pairs {
            let u = match fold.take() {
                Some(sys) => format!("{sys}\n{u}"),
                None => u.clone(),
            };
            s.push_str(&self.user_turn(&u));
            s.push_str(&self.assistant_turn(a));
        }
        let user = match fold.take() {
            Some(sys) => format!("{sys}\n{user}"),
            None => user.to_string(),
        };
        s.push_str(&self.user_turn(&user));
        s.push_str(self.assistant_open(thinking));
        if let Some(p) = assistant_prefix {
            s.push_str(p);
        }
        s
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

/// The same day in Harmony's format (`%Y-%m-%d`, e.g. `2026-10-14`). gpt-oss's
/// Jinja template calls `strftime_now("%Y-%m-%d")`; the engine's callers pass
/// [`today_llama3`]'s string, so the month name is mapped back to its number.
fn harmony_date(today: &str) -> String {
    let mut parts = today.split_whitespace();
    let (Some(d), Some(m), Some(y)) = (parts.next(), parts.next(), parts.next()) else {
        return today.to_string();
    };
    const MON: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let Some(month) = MON.iter().position(|x| x.eq_ignore_ascii_case(m)) else {
        return today.to_string();
    };
    let (Ok(day), Ok(year)) = (d.parse::<u32>(), y.parse::<i32>()) else {
        return today.to_string();
    };
    format!("{year:04}-{:02}-{day:02}", month + 1)
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
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(u, a)| (u.to_string(), a.to_string()))
            .collect();
        t.render(system, &pairs, user, "14 Oct 2026", false, None)
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

    /// Phi-4's format, from llama.cpp v0.9.0 src/llama-chat.cpp:345-352: the role
    /// and the text joined by `<|im_sep|>`, no newlines anywhere.
    #[test]
    fn phi4_renders_llama_cpps_strings() {
        let t = Template::Phi4;
        assert_eq!(
            render_chat_prompt(t, None, &[], "U"),
            "<|im_start|>user<|im_sep|>U<|im_end|><|im_start|>assistant<|im_sep|>"
        );
        assert_eq!(
            render_chat_prompt(t, Some("S"), &[], "U"),
            "<|im_start|>system<|im_sep|>S<|im_end|>\
             <|im_start|>user<|im_sep|>U<|im_end|><|im_start|>assistant<|im_sep|>"
        );
        assert_eq!(
            render_chat_prompt(t, Some("S"), &[("U1", "A1")], "U2"),
            "<|im_start|>system<|im_sep|>S<|im_end|>\
             <|im_start|>user<|im_sep|>U1<|im_end|><|im_start|>assistant<|im_sep|>A1<|im_end|>\
             <|im_start|>user<|im_sep|>U2<|im_end|><|im_start|>assistant<|im_sep|>"
        );
        // A continued conversation closes the answer exactly as a replayed one does.
        assert_eq!(t.close_previous(), t.assistant_turn(""));
        assert!(t.default_system("phi-4").is_none());
        assert_eq!(Template::from_name("phi4"), Some(t));
        // Phi-4's own Jinja template (microsoft/phi-4 tokenizer_config.json).
        let phi4_tmpl = "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}";
        assert_eq!(Template::detect_from_str(phi4_tmpl), Ok(Template::Phi4));
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

    /// The template `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` carries, read from
    /// the file's own `tokenizer.chat_template`: the pre-update Mistral template,
    /// with no system branch. The same string is in llama.cpp's
    /// test-chat-template.cpp, v0.9.0 (MIT-licensed), as
    /// "mistralai/Mistral-7B-Instruct-v0.2 (NOTE: Old pre-v1 without a system prompt)".
    const MISTRAL_V0_TMPL: &str = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";

    #[test]
    fn mistral_v0_renders_expected_output_matching_llama_cpp() {
        let t = Template::MistralV0;
        let rendered = render_chat_prompt(t, Some(SYSTEM_MSG), &PAIRS, FINAL_USER);
        // expected_output from llama.cpp's test-chat-template.cpp, v0.9.0
        // ("mistralai/Mistral-7B-Instruct-v0.2 (NOTE: Old pre-v1 without a system
        // prompt)"): the template has no system branch, so the system message is
        // folded into the first user turn, with a newline between the two.
        let expected = "[INST] You are a helpful assistant\n\
                        Hello [/INST]Hi there</s>[INST] Who are you [/INST]   \
                        I am an assistant   </s>[INST] Another question [/INST]";
        assert_eq!(rendered, expected);

        // The same conversation without a system message, as llama-server's
        // /apply-template renders it for TheBloke's file (v0.9.0, --jinja):
        // "[INST] Hello [/INST]Hi there</s>[INST] Another question [/INST]".
        assert_eq!(
            render_chat_prompt(t, None, &PAIRS, FINAL_USER),
            "[INST] Hello [/INST]Hi there</s>[INST] Who are you [/INST]   \
             I am an assistant   </s>[INST] Another question [/INST]"
        );

        // The template writes no system block of its own.
        assert_eq!(t.system_block(Some(SYSTEM_MSG), "x"), "");
        assert_eq!(t.user_turn("Hello"), "[INST] Hello [/INST]");
        assert_eq!(t.assistant_turn("Hi there"), "Hi there</s>");
        assert_eq!(t.close_previous(), "</s>");
        assert_eq!(t.assistant_open(false), "");
        assert_eq!(t.default_system("Mistral 7B"), None);
    }

    #[test]
    fn the_pre_update_mistral_template_is_detected() {
        assert_eq!(
            Template::detect_from_str(MISTRAL_V0_TMPL),
            Ok(Template::MistralV0)
        );
        // The updated template (the one `mistralai/Mistral-7B-Instruct-v0.2`
        // declares today) is still the v1 one.
        let mistral_v1_tmpl = "{{ bos_token }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- if loop.first and system_message is defined %}{{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}{%- else %}{{- ' [INST] ' + message['content'] + ' [/INST]' }}{%- endif %}{%- elif message['role'] == 'assistant' %}{{- ' ' + message['content'] + eos_token}}{%- endif %}{%- endfor %}";
        assert_eq!(
            Template::detect_from_str(mistral_v1_tmpl),
            Ok(Template::MistralV1)
        );
    }

    #[test]
    fn llama2_plain_renders_what_llama_servers_llama2_renders() {
        let t = Template::Llama2Plain;
        // llama-server v0.9.0's /apply-template with `--chat-template llama2`
        // (the built-in name, which is this format):
        //   "[INST] What is the capital of France? [/INST]" without a system
        //   message, and with one:
        //   "[INST] You are a helpful assistant.\nWhat is the capital of France? [/INST]"
        assert_eq!(
            t.render(
                Some(SYSTEM_MSG),
                &[],
                "What is the capital of France?",
                "x",
                false,
                None
            ),
            "[INST] You are a helpful assistant\nWhat is the capital of France? [/INST]"
        );
        assert_eq!(
            t.render(
                None,
                &[],
                "What is the capital of France?",
                "x",
                false,
                None
            ),
            "[INST] What is the capital of France? [/INST]"
        );
        // Two turns: "[INST] Hello [/INST]Hi there</s>[INST] Another question [/INST]"
        assert_eq!(
            render_chat_prompt(t, None, &[("Hello", "Hi there")], FINAL_USER),
            "[INST] Hello [/INST]Hi there</s>[INST] Another question [/INST]"
        );
    }

    #[test]
    fn harmony_drops_an_earlier_turns_analysis_channel_from_the_replay() {
        let t = Template::Harmony;
        // What the model generated in the earlier turn: analysis, then final.
        let generated = "<|channel|>analysis<|message|>SECRET REASONING<|end|>\
                         <|start|>assistant<|channel|>final<|message|>Hi there";
        // An earlier assistant turn keeps only its final channel, as the Jinja
        // template's "CoT is dropped during all previous turns" branch does.
        assert_eq!(
            t.assistant_turn(generated),
            "<|channel|>final<|message|>Hi there<|end|>"
        );
        assert_eq!(
            t.render(
                None,
                &[("Hello".into(), generated.into())],
                "Next",
                "14 Oct 2026",
                false,
                None
            ),
            format!(
                "{HARMONY_SYSTEM}<|start|>user<|message|>Hello<|end|>\
                 <|start|>assistant<|channel|>final<|message|>Hi there<|end|>\
                 <|start|>user<|message|>Next<|end|><|start|>assistant"
            )
        );
        assert!(!t
            .render(
                None,
                &[("Hello".into(), generated.into())],
                "Next",
                "14 Oct 2026",
                false,
                None
            )
            .contains("SECRET"));
        // The incremental path's closing token is the one a full render writes, so a
        // final-only history produces the same string token for token.
        let full = t.render(
            None,
            &[("Hello".into(), "Hi there".into())],
            "Next",
            "14 Oct 2026",
            false,
            None,
        );
        let incremental = format!(
            "{}{}{}{}",
            t.system_block(None, "14 Oct 2026"),
            t.user_turn("Hello"),
            t.assistant_turn("Hi there"),
            t.user_turn("Next")
        );
        assert_eq!(full, incremental);
        assert_eq!(t.close_previous(), "<|end|>");
        assert!(t.replays_transcript());
        assert!(!Template::Qwen3.replays_transcript());
    }

    /// gpt-oss's own chat template, from llama.cpp v0.9.0's
    /// `models/templates/openai-gpt-oss-120b.jinja`.
    const HARMONY_TMPL: &str = include_str!("../tests/fixtures/harmony.jinja");

    /// The fixed block Harmony writes before any conversation. The date is the one
    /// the test passes (`14 Oct 2026`), rendered as `%Y-%m-%d`.
    const HARMONY_SYSTEM: &str = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-10-14\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>";

    fn harmony_developer(sys: &str) -> String {
        format!("<|start|>developer<|message|># Instructions\n\n{sys}\n\n<|end|>")
    }

    #[test]
    fn harmony_renders_system_and_user_exactly_like_the_gpt_oss_template() {
        let t = Template::Harmony;
        let rendered = t.render(Some(SYSTEM_MSG), &[], "Hello", "14 Oct 2026", false, None);
        assert_eq!(
            rendered,
            format!(
                "{HARMONY_SYSTEM}{}<|start|>user<|message|>Hello<|end|><|start|>assistant",
                harmony_developer(SYSTEM_MSG)
            )
        );

        // No system message: Harmony still writes its fixed system block, and no
        // developer block.
        assert_eq!(
            t.render(None, &[], "Hello", "14 Oct 2026", false, None),
            format!("{HARMONY_SYSTEM}<|start|>user<|message|>Hello<|end|><|start|>assistant")
        );

        // The pieces the incremental session uses.
        assert_eq!(t.default_system("gpt-oss-20b"), None);
        assert_eq!(t.system_fold(Some(SYSTEM_MSG)), None);
        assert_eq!(
            t.user_turn("Hello"),
            "<|start|>user<|message|>Hello<|end|><|start|>assistant"
        );
        assert_eq!(
            t.assistant_turn("Hi there"),
            "<|channel|>final<|message|>Hi there<|end|>"
        );
        assert_eq!(t.close_previous(), "<|end|>");
        assert_eq!(t.assistant_open(false), "");
    }

    #[test]
    fn harmony_renders_a_multi_turn_conversation_exactly_like_the_gpt_oss_template() {
        let t = Template::Harmony;
        let rendered = render_chat_prompt(t, Some(SYSTEM_MSG), &PAIRS, FINAL_USER);
        let expected = format!(
            "{HARMONY_SYSTEM}{}\
             <|start|>user<|message|>Hello<|end|>\
             <|start|>assistant<|channel|>final<|message|>Hi there<|end|>\
             <|start|>user<|message|>Who are you<|end|>\
             <|start|>assistant<|channel|>final<|message|>   I am an assistant   <|end|>\
             <|start|>user<|message|>Another question<|end|><|start|>assistant",
            harmony_developer(SYSTEM_MSG)
        );
        assert_eq!(rendered, expected);
    }

    #[test]
    fn harmony_ends_a_final_assistant_message_with_return_without_a_generation_prompt() {
        let t = Template::Harmony;
        // The template's training form: a last assistant turn and
        // add_generation_prompt=false ends in `<|return|>`.
        let mut rendered = t.system_block(Some(SYSTEM_MSG), "14 Oct 2026");
        rendered.push_str(&t.user_turn("Hello"));
        rendered.push_str(&t.assistant_final("Hi there"));
        assert_eq!(
            rendered,
            format!(
                "{HARMONY_SYSTEM}{}\
                 <|start|>user<|message|>Hello<|end|>\
                 <|start|>assistant<|channel|>final<|message|>Hi there<|return|>",
                harmony_developer(SYSTEM_MSG)
            )
        );
        // With a generation prompt instead, the same transcript is replayed with
        // `<|end|>` and the assistant header opens the next turn.
        let with_prompt = render_chat_prompt(t, Some(SYSTEM_MSG), &[("Hello", "Hi there")], "Next");
        assert!(with_prompt.contains("<|channel|>final<|message|>Hi there<|end|>"));
        assert!(with_prompt.ends_with("<|start|>user<|message|>Next<|end|><|start|>assistant"));
        assert!(!with_prompt.contains("<|return|>"));
    }

    #[test]
    fn harmony_is_detected_from_the_gpt_oss_template() {
        assert_eq!(
            Template::detect_from_str(HARMONY_TMPL),
            Ok(Template::Harmony)
        );
        // llama.cpp detects the same pair of markers (`src/llama-chat.cpp:222-223`).
        assert_eq!(
            Template::detect_from_str("<|start|>assistant<|channel|>final<|message|>"),
            Ok(Template::Harmony)
        );
        assert_eq!(Template::from_name("harmony"), Some(Template::Harmony));
    }

    #[test]
    fn template_names_round_trip() {
        for name in TEMPLATES {
            let t = Template::from_name(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(t.name(), *name, "{name}");
        }
        assert_eq!(Template::from_name("llama2"), Some(Template::Llama2Plain));
        assert_eq!(
            Template::from_name("llama2-sys-bos"),
            Some(Template::Llama2)
        );
        assert_eq!(Template::from_name("mistral-v0"), Some(Template::MistralV0));
        assert_eq!(Template::from_name("nonesuch"), None);
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
        // A ChatML file that names neither family gets no system message of its own:
        // the SmolLM one would be a stranger's sentence in someone else's conversation.
        assert!(Template::ChatMl.default_system("Phi 4").is_none());
        assert!(Template::ChatMl
            .default_system("Llama 3.2 1B Instruct")
            .is_none());
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
    // --- Qwen 3.5's template (Bonsai 2 carries it too) --------------------------------

    /// The conversation as the tool module's renderer writes it, with no tools: the
    /// bytes llama.cpp renders (`tests/fixtures/tools-qwen35.jsonl`).
    fn qwen35_reference(
        system: Option<&str>,
        pairs: &[(String, String)],
        user: &str,
        thinking: bool,
    ) -> String {
        use crate::tools::{render_with, Msg, RenderOptions, ToolFormat};
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(Msg::System(sys.to_string()));
        }
        for (u, a) in pairs {
            messages.push(Msg::User(u.clone()));
            messages.push(Msg::Assistant {
                content: a.clone(),
                reasoning: None,
                tool_calls: Vec::new(),
            });
        }
        messages.push(Msg::User(user.to_string()));
        render_with(
            ToolFormat::QwenXml,
            &[],
            &messages,
            RenderOptions {
                thinking,
                date: None,
            },
        )
        .expect("a plain conversation renders")
    }

    #[test]
    fn qwen35_is_detected_by_its_exact_text_and_qwen3_by_the_rest() {
        assert_eq!(
            Template::detect_from_str(crate::tools::QWEN35_TEMPLATE),
            Ok(Template::Qwen35)
        );
        for variant in crate::tools::QWEN35_VARIANTS {
            assert_eq!(Template::detect_from_str(variant), Ok(Template::Qwen35));
        }
        // A template with a thinking switch that is not one of those texts is Qwen 3's.
        let other = crate::tools::QWEN35_TEMPLATE.replace("<function=", "<fn=");
        assert_eq!(Template::detect_from_str(&other), Ok(Template::Qwen3));
        assert_eq!(Template::from_name("qwen35"), Some(Template::Qwen35));
        assert_eq!(Template::Qwen35.name(), "qwen35");
        assert!(TEMPLATES.contains(&"qwen35"));
    }

    #[test]
    fn qwen35_prompts_are_the_tool_renderers_prompts() {
        let pair = |u: &str, a: &str| (u.to_string(), a.to_string());
        let histories: Vec<Vec<(String, String)>> = vec![
            vec![],
            vec![pair("Hi", "Hello!")],
            vec![pair("  Hi  ", "\n Hello! \n"), pair("And?", "More.")],
            // A reply as a session keeps it: the reasoning the model wrote after the
            // opened block, its closing tag, then the answer.
            vec![pair(
                "What time is it?",
                "Let me think.\n</think>\n\nIt is noon.",
            )],
            // The same with the opening tag, as a client might send it back.
            vec![pair(
                "What time is it?",
                "<think>\nLet me think.\n</think>\n\nIt is noon.",
            )],
        ];
        for system in [None, Some("Be brief."), Some("  padded system \n")] {
            for pairs in &histories {
                for user in ["Why?", "  spaces around  ", "line one\nline two\n"] {
                    for thinking in [true, false] {
                        let got = Template::Qwen35.render(system, pairs, user, "", thinking, None);
                        let want = qwen35_reference(system, pairs, user, thinking);
                        assert_eq!(
                            got, want,
                            "system {system:?}, {} earlier turns, user {user:?}, thinking {thinking}",
                            pairs.len()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn qwen35_opens_the_reasoning_block_itself() {
        let on = Template::Qwen35.render(None, &[], "Hi", "", true, None);
        assert!(on.ends_with("<|im_start|>assistant\n<think>\n"), "{on:?}");
        assert!(Template::Qwen35.opens_reasoning(true));
        let off = Template::Qwen35.render(None, &[], "Hi", "", false, None);
        assert!(
            off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "{off:?}"
        );
        assert!(!Template::Qwen35.opens_reasoning(false));
        // Qwen 3's template leaves the opening to the model.
        assert!(!Template::Qwen3.opens_reasoning(true));
        assert!(!Template::ChatMl.opens_reasoning(true));
        // A continuation follows the opened block.
        let continued = Template::Qwen35.render(None, &[], "Hi", "", true, Some("So far"));
        assert!(continued.ends_with("<think>\nSo far"), "{continued:?}");
    }
}
