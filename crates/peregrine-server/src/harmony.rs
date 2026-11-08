//! gpt-oss's Harmony reply, split into its analysis (reasoning) and final (answer) channels.
//!
//! gpt-oss continues from `<|start|>assistant` and writes
//!
//! ```text
//! <|channel|>analysis<|message|>REASONING<|end|><|start|>assistant<|channel|>final<|message|>ANSWER<|return|>
//! ```
//!
//! The analysis part may be absent, a `commentary` channel may precede it, and a
//! channel header may carry `<|constrain|>` and a recipient (`to=functions.…`).
//! llama.cpp v0.9.0 (commit `cd959ba`) parses the same grammar in
//! `common/parsers/gpt-oss.cpp:68-93`: `<|start|>assistant` opens a message,
//! `analysis` goes to reasoning (`:80-81`), `commentary` is a preamble and goes to
//! content (`:87`), and `final` is the answer (`:88`). The control tokens are
//! consumed, never returned, and a block may be split across streaming chunks: the
//! parser holds a trailing prefix of a control token until the next chunk decides
//! it.
//!
//! `reasoning_format` follows llama.cpp: `none` does not extract the analysis
//! channel (`common/parsers/gpt-oss.cpp:66,82-83`), so its text is returned as
//! content; `auto` and `deepseek` split it into `reasoning_content`
//! (`common/common.h:428-436`); `deepseek-legacy` also mirrors it into a streaming
//! `content` delta (`tools/server/server-task.cpp:78`). Unlike llama.cpp, whose
//! `none` mode leaves the raw channel markers in `message.content` (the parser tags
//! the whole matched region as content, `common/parsers/gpt-oss.cpp:83`), this
//! parser always strips the control tokens: neither field carries them.

/// How a reply's reasoning is presented, the values of llama.cpp's
/// `--reasoning-format` (`common/chat.cpp:1026-1041`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReasoningFormat {
    /// Leave the analysis channel's text in `content`; no `reasoning_content`.
    None,
    /// Split the analysis channel into `reasoning_content` (llama.cpp's default).
    Auto,
    /// The same split; the name llama.cpp also accepts for it.
    Deepseek,
    /// The split, plus the reasoning mirrored into streaming `content` deltas.
    DeepseekLegacy,
}

impl ReasoningFormat {
    /// The format named by a request or command line, or `None` for a name
    /// llama.cpp does not know.
    pub(crate) fn parse(name: &str) -> Option<ReasoningFormat> {
        Some(match name {
            "none" => ReasoningFormat::None,
            "auto" => ReasoningFormat::Auto,
            "deepseek" => ReasoningFormat::Deepseek,
            "deepseek-legacy" => ReasoningFormat::DeepseekLegacy,
            _ => return None,
        })
    }

    /// Whether the analysis channel is separated into `reasoning_content`.
    /// llama.cpp's `extract_reasoning` (`common/parsers/gpt-oss.cpp:66`).
    pub(crate) fn extracts(self) -> bool {
        !matches!(self, ReasoningFormat::None)
    }

    /// Whether reasoning is also a streaming `content` delta
    /// (`tools/server/server-task.cpp:78`).
    pub(crate) fn reasoning_in_content(self) -> bool {
        matches!(self, ReasoningFormat::DeepseekLegacy)
    }
}

/// Where a piece of the reply belongs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Channel {
    Reasoning,
    Content,
}

/// The control tokens that structure a Harmony reply. `<|return|>` and `<|call|>`
/// are stop tokens (their text is empty in an emission), but a parser that is fed
/// raw text must consume them too.
const MARKERS: &[&str] = &[
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|end|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
];

/// The part of the reply being read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Between messages: expecting `<|start|>`, `<|channel|>` or the end.
    AwaitChannel,
    /// After `<|start|>`: the role header (`assistant`, `assistant to=…`) runs to
    /// the next `<|channel|>`; it is control metadata, not text.
    Role,
    /// After `<|channel|>`: the channel header runs to `<|message|>`; it names the
    /// channel and may carry `<|constrain|>` and a recipient.
    Header,
    /// Inside a channel's message, up to the next `<|end|>`.
    Body(Channel),
}

/// An incremental Harmony reply parser: feed it each emitted token's text, then
/// flush it when the generation ends.
pub(crate) struct Harmony {
    state: State,
    pending: String,
}

impl Harmony {
    pub(crate) fn new() -> Harmony {
        Harmony {
            state: State::AwaitChannel,
            pending: String::new(),
        }
    }

    /// Read one emission's text and return the pieces of it that are text, with the
    /// channel each belongs to. Control tokens are consumed; a trailing partial
    /// control token is held for the next call.
    pub(crate) fn feed(&mut self, text: &str) -> Vec<(Channel, String)> {
        self.pending.push_str(text);
        self.drain(false)
    }

    /// The generation ended: flush anything buffered. A control token that was cut
    /// off (a `max_tokens` stop in the middle of one) is dropped rather than leaked;
    /// a reply cut off in the middle of the analysis is returned in full.
    pub(crate) fn finish(&mut self) -> Vec<(Channel, String)> {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> Vec<(Channel, String)> {
        let mut out = Vec::new();
        loop {
            match self.state {
                State::AwaitChannel => {
                    if let Some((at, marker)) = next_marker(&self.pending) {
                        // Text before the first channel is not part of the format;
                        // keep it as content rather than dropping the answer.
                        emit(&mut out, Channel::Content, &self.pending[..at]);
                        self.pending.drain(..at + marker.len());
                        match marker {
                            "<|start|>" => self.state = State::Role,
                            "<|channel|>" => self.state = State::Header,
                            _ => {}
                        }
                        continue;
                    }
                    if flush {
                        let hold = partial_marker(&self.pending);
                        let n = self.pending.len() - hold;
                        emit(&mut out, Channel::Content, &self.pending[..n]);
                        self.pending.clear();
                        return out;
                    }
                    let hold = partial_marker(&self.pending);
                    let n = self.pending.len() - hold;
                    emit(&mut out, Channel::Content, &self.pending[..n]);
                    self.pending.drain(..n);
                    return out;
                }
                State::Role => {
                    if let Some((at, marker)) = next_marker(&self.pending) {
                        self.pending.drain(..at + marker.len());
                        match marker {
                            "<|channel|>" => self.state = State::Header,
                            "<|end|>" | "<|return|>" | "<|call|>" => {
                                self.state = State::AwaitChannel
                            }
                            _ => {}
                        }
                        continue;
                    }
                    if flush {
                        self.pending.clear();
                    }
                    return out;
                }
                State::Header => {
                    if let Some(at) = self.pending.find("<|message|>") {
                        let header = self.pending[..at].to_ascii_lowercase();
                        self.pending.drain(..at + "<|message|>".len());
                        // `analysis` is reasoning; `commentary` is a preamble and
                        // `final` is the answer (llama.cpp v0.9.0,
                        // common/parsers/gpt-oss.cpp:80-88). A recipient
                        // (`to=…`) is a tool call, which this server does not
                        // support: keep its arguments visible as content.
                        let word = header.split_whitespace().next().unwrap_or("");
                        let channel = if word == "analysis" && !header.contains("to=") {
                            Channel::Reasoning
                        } else {
                            Channel::Content
                        };
                        self.state = State::Body(channel);
                        continue;
                    }
                    if flush {
                        // An unfinished channel header is control, not text.
                        self.pending.clear();
                    }
                    return out;
                }
                State::Body(channel) => {
                    if let Some((at, marker)) = next_marker(&self.pending) {
                        emit(&mut out, channel, &self.pending[..at]);
                        self.pending.drain(..at + marker.len());
                        match marker {
                            "<|channel|>" => self.state = State::Header,
                            "<|end|>" | "<|start|>" | "<|return|>" | "<|call|>" => {
                                self.state = State::AwaitChannel
                            }
                            // `<|message|>` or `<|constrain|>` outside a header is
                            // malformed; consume it and stay in the message.
                            _ => {}
                        }
                        continue;
                    }
                    let hold = partial_marker(&self.pending);
                    let n = self.pending.len() - hold;
                    emit(&mut out, channel, &self.pending[..n]);
                    self.pending.drain(..n);
                    if flush {
                        self.pending.clear();
                    }
                    return out;
                }
            }
        }
    }
}

/// The earliest control token in `s`, with its offset.
fn next_marker(s: &str) -> Option<(usize, &'static str)> {
    MARKERS
        .iter()
        .filter_map(|m| s.find(m).map(|at| (at, *m)))
        .min_by_key(|(at, m)| (*at, std::cmp::Reverse(m.len())))
}

/// The length of the longest suffix of `s` that is a prefix of a control token and
/// at least two bytes long (`<|`). Such a suffix may be completed by the next
/// emission, so it is held back; a lone `<` is ordinary text.
fn partial_marker(s: &str) -> usize {
    let mut hold = 0;
    for m in MARKERS {
        let max = (m.len() - 1).min(s.len());
        for n in (2..=max).rev() {
            if s.ends_with(&m[..n]) {
                hold = hold.max(n);
                break;
            }
        }
    }
    hold
}

fn emit(out: &mut Vec<(Channel, String)>, channel: Channel, text: &str) {
    if !text.is_empty() {
        out.push((channel, text.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::{Channel, Harmony, ReasoningFormat};

    /// Feed one string in chunks, then flush. Returns (reasoning, content).
    fn parse_chunks(chunks: &[&str]) -> (String, String) {
        let mut p = Harmony::new();
        let (mut reasoning, mut content) = (String::new(), String::new());
        for c in chunks {
            for (ch, t) in p.feed(c) {
                match ch {
                    Channel::Reasoning => reasoning.push_str(&t),
                    Channel::Content => content.push_str(&t),
                }
            }
        }
        for (ch, t) in p.finish() {
            match ch {
                Channel::Reasoning => reasoning.push_str(&t),
                Channel::Content => content.push_str(&t),
            }
        }
        (reasoning, content)
    }

    #[test]
    fn analysis_then_final_splits_the_two_channels() {
        let reply = "<|channel|>analysis<|message|>Let me think.<|end|>\
                     <|start|>assistant<|channel|>final<|message|>The answer is 42.";
        assert_eq!(
            parse_chunks(&[reply]),
            ("Let me think.".into(), "The answer is 42.".into())
        );
    }

    #[test]
    fn final_only_has_no_reasoning() {
        let reply = "<|channel|>final<|message|>Just the answer.";
        assert_eq!(
            parse_chunks(&[reply]),
            (String::new(), "Just the answer.".into())
        );
    }

    #[test]
    fn a_reply_cut_off_mid_analysis_returns_what_it_wrote() {
        // A max_tokens stop: no <|end|>, no final channel.
        let reply = "<|channel|>analysis<|message|>The first half of a thought";
        assert_eq!(
            parse_chunks(&[reply]),
            ("The first half of a thought".into(), String::new())
        );
    }

    #[test]
    fn control_tokens_split_across_streaming_chunks_never_leak() {
        // Every control token is cut in the middle of its text.
        let chunks = [
            "<|chan",
            "nel|>analy",
            "sis<|mess",
            "age|>Rea",
            "soning<|en",
            "d|><|sta",
            "rt|>assis",
            "tant<|chann",
            "el|>final<|me",
            "ssage|>Ans",
            "wer",
        ];
        assert_eq!(parse_chunks(&chunks), ("Reasoning".into(), "Answer".into()));
    }

    #[test]
    fn a_commentary_preamble_and_a_constraint_are_content() {
        // llama.cpp parses `<|channel|>commentary…` as a preamble in content
        // (common/parsers/gpt-oss.cpp:87), and a header may carry a constraint.
        let reply = "<|channel|>commentary to=assistant<|message|>Preface\
                     <|end|><|start|>assistant<|channel|>analysis<|message|>R<|end|>\
                     <|start|>assistant<|channel|>final json<|constrain|>json<|message|>A";
        assert_eq!(parse_chunks(&[reply]), ("R".into(), "PrefaceA".into()));
    }

    #[test]
    fn format_none_keeps_the_analysis_text_and_still_strips_the_markers() {
        // The decision lives in the server's mapping from Channel to field; the
        // parser's job in `none` mode is not to lose the analysis text.
        let (reasoning, content) = parse_chunks(&["<|channel|>analysis<|message|>R<|end|>"]);
        assert_eq!(reasoning, "R");
        assert!(content.is_empty());
        assert!(!ReasoningFormat::None.extracts());
        assert!(ReasoningFormat::Auto.extracts());
        assert!(ReasoningFormat::Deepseek.extracts());
        assert!(ReasoningFormat::DeepseekLegacy.extracts());
        assert!(ReasoningFormat::DeepseekLegacy.reasoning_in_content());
        assert!(!ReasoningFormat::Auto.reasoning_in_content());
    }

    #[test]
    fn an_incomplete_control_token_at_the_end_is_dropped() {
        // A max_tokens stop in the middle of `<|end|>`: the partial token is not
        // the answer, so it does not leak into content.
        let (reasoning, content) = parse_chunks(&["<|channel|>final<|message|>Answer<|en"]);
        assert_eq!(reasoning, "");
        assert_eq!(content, "Answer");
    }
}
