//! Tool calling: the tool formats (S6.2, `docs/design/s6-tool-calling.md`).
//!
//! A client sends tools with a chat request; the model may answer with calls to them.
//! For the engine that is text handling on both sides of the forward pass:
//!
//! * **rendering**: the tools, and earlier calls and their results, are written into the
//!   prompt where the model's own chat template puts them;
//! * **parsing**: the model's text is split into ordinary text and calls.
//!
//! Peregrine has no template engine, so each family of templates is a **tool format**: a
//! renderer and a parser written here and checked against llama.cpp. The renderer must give
//! the bytes llama.cpp v0.9.0's `POST /apply-template` gives for the same request on the
//! same template (`tests/tools_render.rs`, fixtures written by `scripts/tool-fixtures.py`).
//!
//! A template is recognised by comparing it with the copy kept under `tools/`: a file whose
//! template differs is not guessed at, and a request that carries tools is refused for it.
//!
//! # Formats
//!
//! * [`ToolFormat::Hermes`]: Qwen 2.5's ChatML template. Tool signatures are JSON lines
//!   inside `<tools>` in the system message; a call is `{"name": ..., "arguments": {...}}`
//!   inside `<tool_call>`; results come back inside `<tool_response>` in a user turn,
//!   consecutive results sharing one turn.

use crate::json::Value;

/// ChatML's turn markers, as the Qwen templates write them.
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// The system message Qwen 2.5's template writes when the conversation has none.
const QWEN25_DEFAULT_SYSTEM: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// Qwen 2.5's chat template, as its GGUF files carry it (`tokenizer.chat_template`).
pub const QWEN25_TEMPLATE: &str = include_str!("tools/qwen25.jinja");

/// A family of templates that writes tools and calls the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// Qwen 2.5: JSON calls inside `<tool_call>` (the "Hermes" convention).
    Hermes,
}

impl ToolFormat {
    /// The format of a file's chat template, or `None` when the template is not one whose
    /// tool section Peregrine reproduces. The comparison is exact (trailing whitespace
    /// aside): a variant is refused rather than rendered almost right.
    pub fn for_template(template: &str) -> Option<ToolFormat> {
        if template.trim_end() == QWEN25_TEMPLATE.trim_end() {
            return Some(ToolFormat::Hermes);
        }
        None
    }

    pub fn name(self) -> &'static str {
        match self {
            ToolFormat::Hermes => "hermes",
        }
    }
}

/// One call to a tool: its name and its arguments, a JSON object.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// One message of a conversation, in the shape the templates read.
#[derive(Debug, Clone, PartialEq)]
pub enum Msg {
    System(String),
    User(String),
    /// What the assistant wrote: text, calls, or both.
    Assistant {
        content: String,
        tool_calls: Vec<ToolCall>,
    },
    /// A tool's result, sent back by the client.
    Tool {
        content: String,
    },
}

/// A message's text: a string, `null` (an assistant message that only calls tools), or a
/// list of text parts, which llama.cpp joins with a newline for these templates.
fn content_of(m: &Value, i: usize) -> Result<String, String> {
    match m.get("content") {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::Str(s)) => Ok(s.clone()),
        Some(Value::Arr(parts)) => {
            let mut texts = Vec::with_capacity(parts.len());
            for (j, p) in parts.iter().enumerate() {
                match (p.get("type").and_then(Value::as_str), p.get("text")) {
                    (Some("text"), Some(Value::Str(t))) => texts.push(t.as_str()),
                    _ => {
                        return Err(format!(
                            "messages[{i}].content[{j}] is not a text part; only text is supported"
                        ))
                    }
                }
            }
            Ok(texts.join("\n"))
        }
        Some(_) => Err(format!(
            "messages[{i}].content must be a string or a list of text parts"
        )),
    }
}

/// The `tools` of a request, checked and put in the shape the templates are given.
///
/// llama.cpp does not hand a template the client's tool objects: it reads each one into
/// (name, description, parameters) and writes it back as `{"type": "function", "function":
/// {"name", "description", "parameters"}}` (`common_chat_tools_parse_oaicompat` and
/// `common_chat_tools_to_json_oaicompat`, `common/chat.cpp`). So a missing description
/// becomes `""`, missing parameters become `{}`, and any other key of the function
/// (`strict`, for one) is gone before the template sees it. `defer_loading: true`, on the
/// function or beside `type`, is the one extra key kept. The parameters keep their own key
/// order, because the templates write them out as JSON.
pub fn tools_from_json(tools: Option<&Value>) -> Result<Vec<Value>, String> {
    let Some(tools) = tools else {
        return Ok(Vec::new());
    };
    if tools.is_null() {
        return Ok(Vec::new());
    }
    let list = tools.as_array().ok_or("tools must be an array")?;
    let mut out = Vec::with_capacity(list.len());
    let mut seen: Vec<String> = Vec::new();
    for (i, t) in list.iter().enumerate() {
        if t.as_object().is_none() {
            return Err(format!("tools[{i}] must be an object"));
        }
        match t.get("type").and_then(Value::as_str) {
            Some("function") => {}
            Some(other) => {
                return Err(format!(
                    "tools[{i}].type {other:?} is not supported; only \"function\" is"
                ))
            }
            None => return Err(format!("tools[{i}].type must be \"function\"")),
        }
        let function = t
            .get("function")
            .ok_or_else(|| format!("tools[{i}].function is missing"))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("tools[{i}].function.name must be a string"))?;
        if name.is_empty() {
            return Err(format!("tools[{i}].function.name must not be empty"));
        }
        if seen.iter().any(|n| n == name) {
            return Err(format!("tools[{i}].function.name {name:?} is given twice"));
        }
        seen.push(name.to_string());
        let description = match function.get("description") {
            None => "",
            Some(Value::Str(d)) => d.as_str(),
            Some(_) => return Err(format!("tools[{i}].function.description must be a string")),
        };
        let parameters = function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| Value::Obj(Vec::new()));
        let defer = match function
            .get("defer_loading")
            .or_else(|| t.get("defer_loading"))
        {
            None => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => return Err(format!("tools[{i}].defer_loading must be a boolean")),
        };
        let mut f = vec![
            ("name".to_string(), Value::Str(name.to_string())),
            (
                "description".to_string(),
                Value::Str(description.to_string()),
            ),
            ("parameters".to_string(), parameters),
        ];
        if defer {
            f.push(("defer_loading".to_string(), Value::Bool(true)));
        }
        out.push(Value::Obj(vec![
            ("type".to_string(), Value::Str("function".to_string())),
            ("function".to_string(), Value::Obj(f)),
        ]));
    }
    Ok(out)
}

/// The names of a request's tools, in order.
pub fn tool_names(tools: &[Value]) -> Vec<String> {
    tools
        .iter()
        .filter_map(|t| t.get("function")?.get("name")?.as_str())
        .map(str::to_string)
        .collect()
}

/// The `messages` of a chat request as [`Msg`]s.
///
/// OpenAI's shapes: `system`, `user`, `assistant` (with optional `tool_calls`) and `tool`.
/// A call's `arguments` is a JSON object written as a string; an object is accepted too.
/// Anything else is refused by name, so a conversation is never rendered with a message
/// silently dropped.
pub fn messages_from_json(messages: &Value) -> Result<Vec<Msg>, String> {
    let list = messages.as_array().ok_or("messages must be an array")?;
    let mut out = Vec::with_capacity(list.len());
    for (i, m) in list.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("messages[{i}].role must be a string"))?;
        match role {
            "system" => out.push(Msg::System(content_of(m, i)?)),
            "user" => out.push(Msg::User(content_of(m, i)?)),
            "tool" => out.push(Msg::Tool {
                content: content_of(m, i)?,
            }),
            "assistant" => {
                let mut tool_calls = Vec::new();
                match m.get("tool_calls") {
                    None | Some(Value::Null) => {}
                    Some(calls) => {
                        let calls = calls
                            .as_array()
                            .ok_or_else(|| format!("messages[{i}].tool_calls must be an array"))?;
                        for (j, c) in calls.iter().enumerate() {
                            tool_calls.push(
                                call_from_json(c)
                                    .map_err(|e| format!("messages[{i}].tool_calls[{j}]: {e}"))?,
                            );
                        }
                    }
                }
                out.push(Msg::Assistant {
                    content: content_of(m, i)?,
                    tool_calls,
                });
            }
            other => {
                return Err(format!(
                    "messages[{i}].role {other:?} is not supported (system, user, assistant, tool)"
                ))
            }
        }
    }
    Ok(out)
}

fn call_from_json(c: &Value) -> Result<ToolCall, String> {
    // OpenAI nests the call under `function`; the templates accept it flat as well.
    let f = c.get("function").unwrap_or(c);
    let name = f
        .get("name")
        .and_then(Value::as_str)
        .ok_or("function.name must be a string")?;
    let arguments = match f.get("arguments") {
        None | Some(Value::Null) => Value::Obj(Vec::new()),
        Some(Value::Str(s)) => {
            Value::parse(s).map_err(|e| format!("function.arguments is not JSON: {e}"))?
        }
        Some(v) => v.clone(),
    };
    if arguments.as_object().is_none() {
        return Err("function.arguments must be a JSON object".into());
    }
    Ok(ToolCall {
        name: name.to_string(),
        arguments,
    })
}

/// The prompt for a conversation with tools, ending with the assistant's opening.
///
/// With no tools this is the template's ordinary prompt, so one renderer serves both. A
/// conversation that ends with an assistant message of text is a continuation, as in
/// llama.cpp: that text is left open for the model to go on from.
pub fn render(format: ToolFormat, tools: &[Value], messages: &[Msg]) -> Result<String, String> {
    match format {
        ToolFormat::Hermes => render_qwen25(tools, messages),
    }
}

/// Qwen 2.5's template, branch for branch (`tools/qwen25.jinja`).
fn render_qwen25(tools: &[Value], messages: &[Msg]) -> Result<String, String> {
    if messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    let mut s = String::new();
    let system = match &messages[0] {
        Msg::System(text) => text.as_str(),
        _ => QWEN25_DEFAULT_SYSTEM,
    };
    s.push_str(IM_START);
    s.push_str("system\n");
    s.push_str(system);
    if !tools.is_empty() {
        s.push_str(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\
             \n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        );
        for t in tools {
            s.push('\n');
            s.push_str(&t.to_jinja_json());
        }
        s.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function name and \
             arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": \
             <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
        );
    }
    s.push_str(IM_END);
    s.push('\n');

    let turn = |s: &mut String, role: &str, text: &str| {
        s.push_str(IM_START);
        s.push_str(role);
        s.push('\n');
        s.push_str(text);
        s.push_str(IM_END);
        s.push('\n');
    };
    // A last assistant message of text is continued, not closed.
    let (messages, prefill) = match messages.split_last() {
        Some((
            Msg::Assistant {
                content,
                tool_calls,
            },
            rest,
        )) if tool_calls.is_empty() => (rest, Some(content.as_str())),
        Some((Msg::Assistant { .. }, _)) => {
            return Err(
                "the conversation ends with an assistant tool call; send the tool's result".into(),
            )
        }
        _ => (messages, None),
    };
    for (i, m) in messages.iter().enumerate() {
        match m {
            // The first system message is the header above; a later one is a turn.
            Msg::System(text) => {
                if i > 0 {
                    turn(&mut s, "system", text);
                }
            }
            Msg::User(text) => turn(&mut s, "user", text),
            Msg::Assistant {
                content,
                tool_calls,
            } => {
                if tool_calls.is_empty() {
                    turn(&mut s, "assistant", content);
                    continue;
                }
                s.push_str(IM_START);
                s.push_str("assistant");
                if !content.is_empty() {
                    s.push('\n');
                    s.push_str(content);
                }
                for c in tool_calls {
                    s.push_str("\n<tool_call>\n{\"name\": \"");
                    s.push_str(&c.name);
                    s.push_str("\", \"arguments\": ");
                    s.push_str(&c.arguments.to_jinja_json());
                    s.push_str("}\n</tool_call>");
                }
                s.push_str(IM_END);
                s.push('\n');
            }
            // Consecutive results share one user turn.
            Msg::Tool { content } => {
                let prev_is_tool = i > 0 && matches!(messages[i - 1], Msg::Tool { .. });
                let next_is_tool = matches!(messages.get(i + 1), Some(Msg::Tool { .. }));
                if !prev_is_tool {
                    s.push_str(IM_START);
                    s.push_str("user");
                }
                s.push_str("\n<tool_response>\n");
                s.push_str(content);
                s.push_str("\n</tool_response>");
                if !next_is_tool {
                    s.push_str(IM_END);
                    s.push('\n');
                }
            }
        }
    }
    s.push_str(IM_START);
    s.push_str("assistant\n");
    if let Some(text) = prefill {
        s.push_str(text);
    }
    Ok(s)
}

/// What the parser finds in the model's text, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Ordinary text: the message's `content`.
    Text(String),
    /// A complete call. `arguments` is a JSON object as text; `index` counts the calls of
    /// this message from zero.
    Call {
        index: usize,
        name: String,
        arguments: String,
    },
}

const HERMES_OPEN: &str = "<tool_call>";
const HERMES_CLOSE: &str = "</tool_call>";

/// Splits a model's output into text and calls as it arrives.
///
/// Feed it the generated text piece by piece with [`CallParser::push`] and call
/// [`CallParser::finish`] at the end. Rules, for every format:
///
/// * Text is passed through as it comes, except a tail that could be the start of a call's
///   opening tag, which is held until it is known to be one or not: a client never sees
///   half a tag.
/// * Whitespace between text and a call, between calls and after the last call is dropped.
///   A message with no call is passed through unchanged, whitespace included.
/// * A call that does not parse (broken JSON, no name, a name that is not one of the
///   request's tools, arguments that are not an object) is given back as text, exactly as
///   the model wrote it. It is never an error.
/// * A call cut off by the end of the output is a call if what is there parses, and text
///   otherwise.
#[derive(Debug)]
pub struct CallParser {
    format: ToolFormat,
    names: Vec<String>,
    /// Text not yet decided: in text mode a possible tag prefix and trailing whitespace; in
    /// call mode the call's body so far.
    pending: String,
    in_call: bool,
    calls: usize,
    /// True after a call until other text is passed on: whitespace there is dropped.
    after_call: bool,
    /// The whitespace that stood before the call being read, given back with the call's
    /// text if it turns out not to be a call.
    ws_before_call: String,
}

impl CallParser {
    /// A parser for one message. `names` are the request's tool names; a call to any other
    /// name is text.
    pub fn new(format: ToolFormat, names: Vec<String>) -> CallParser {
        CallParser {
            format,
            names,
            pending: String::new(),
            in_call: false,
            calls: 0,
            after_call: false,
            ws_before_call: String::new(),
        }
    }

    /// How many calls have been found so far.
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Take the next piece of generated text; events that are now certain go to `out`.
    pub fn push(&mut self, text: &str, out: &mut Vec<Event>) {
        self.pending.push_str(text);
        match self.format {
            ToolFormat::Hermes => self.step_hermes(false, out),
        }
    }

    /// The output has ended: decide everything that was held back.
    pub fn finish(&mut self, out: &mut Vec<Event>) {
        match self.format {
            ToolFormat::Hermes => self.step_hermes(true, out),
        }
    }

    fn text(&mut self, text: &str, out: &mut Vec<Event>) {
        let text = if self.after_call {
            text.trim_start()
        } else {
            text
        };
        if text.is_empty() {
            return;
        }
        self.after_call = false;
        if let Some(Event::Text(last)) = out.last_mut() {
            last.push_str(text);
        } else {
            out.push(Event::Text(text.to_string()));
        }
    }

    fn step_hermes(&mut self, end: bool, out: &mut Vec<Event>) {
        loop {
            if self.in_call {
                let Some(close) = self.pending.find(HERMES_CLOSE) else {
                    if end {
                        // Cut off: a call if the body is whole, text otherwise.
                        let body = std::mem::take(&mut self.pending);
                        self.in_call = false;
                        let raw = format!("{HERMES_OPEN}{body}");
                        self.call_or_text(&body, raw, out);
                    }
                    return;
                };
                let body = self.pending[..close].to_string();
                self.pending.drain(..close + HERMES_CLOSE.len());
                self.in_call = false;
                let raw = format!("{HERMES_OPEN}{body}{HERMES_CLOSE}");
                self.call_or_text(&body, raw, out);
                continue;
            }
            if let Some(open) = self.pending.find(HERMES_OPEN) {
                // Text before a call loses its trailing whitespace; it is kept aside in
                // case this is not a call after all.
                let solid = self.pending[..open].trim_end().len();
                let before = self.pending[..solid].to_string();
                self.ws_before_call = self.pending[solid..open].to_string();
                self.pending.drain(..open + HERMES_OPEN.len());
                self.text(&before, out);
                self.in_call = true;
                continue;
            }
            if end {
                let rest = std::mem::take(&mut self.pending);
                // After a call, trailing whitespace is dropped; a message with no call
                // is passed through as written.
                let rest = if self.calls > 0 {
                    rest.trim_end().to_string()
                } else {
                    rest
                };
                self.text(&rest, out);
                return;
            }
            // Hold back what may still turn into a tag: a tail that is a prefix of the
            // opening tag, and the whitespace before it (dropped if a call follows).
            let hold_tag = longest_tag_prefix(&self.pending, HERMES_OPEN);
            let keep = self.pending.len() - hold_tag;
            let solid = self.pending[..keep].trim_end().len();
            if solid > 0 {
                let ready = self.pending[..solid].to_string();
                self.pending.drain(..solid);
                self.text(&ready, out);
            }
            return;
        }
    }

    /// One finished `<tool_call>` body: a call, or the model's own text (`raw`) with the
    /// whitespace that stood before it.
    fn call_or_text(&mut self, body: &str, raw: String, out: &mut Vec<Event>) {
        let ws = std::mem::take(&mut self.ws_before_call);
        match self.hermes_call(body) {
            Some(ev) => {
                out.push(ev);
                self.after_call = true;
            }
            None => {
                // Between a call and this text the whitespace stays dropped.
                let lead = if self.after_call { "" } else { ws.as_str() };
                self.after_call = false;
                self.text(&format!("{lead}{raw}"), out);
            }
        }
    }

    /// The call inside one `<tool_call>` body, if it is one.
    fn hermes_call(&mut self, body: &str) -> Option<Event> {
        let v = Value::parse(body.trim()).ok()?;
        v.as_object()?;
        let name = v.get("name")?.as_str()?;
        if !self.names.iter().any(|n| n == name) {
            return None;
        }
        let arguments = match v.get("arguments") {
            None | Some(Value::Null) => Value::Obj(Vec::new()),
            // Some models write the arguments as a JSON string of an object.
            Some(Value::Str(s)) => Value::parse(s).ok()?,
            Some(a) => a.clone(),
        };
        arguments.as_object()?;
        let index = self.calls;
        self.calls += 1;
        Some(Event::Call {
            index,
            name: name.to_string(),
            arguments: arguments.to_jinja_json(),
        })
    }
}

/// The length of the longest tail of `text` that is a proper prefix of `tag`.
fn longest_tag_prefix(text: &str, tag: &str) -> usize {
    let max = tag.len().saturating_sub(1).min(text.len());
    for n in (1..=max).rev() {
        if text.is_char_boundary(text.len() - n) && text.ends_with(&tag[..n]) {
            return n;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["get_weather".to_string(), "now".to_string()]
    }

    /// Parse `text` whole, and again in pieces of every size: the events must not depend
    /// on how the text was cut.
    fn parse(text: &str) -> Vec<Event> {
        let mut whole = Vec::new();
        let mut p = CallParser::new(ToolFormat::Hermes, names());
        p.push(text, &mut whole);
        p.finish(&mut whole);
        let chars: Vec<char> = text.chars().collect();
        for size in 1..=chars.len().clamp(1, 24) {
            let mut got = Vec::new();
            let mut p = CallParser::new(ToolFormat::Hermes, names());
            for piece in chars.chunks(size) {
                let piece: String = piece.iter().collect();
                p.push(&piece, &mut got);
            }
            p.finish(&mut got);
            // Pieces may split a text event; join neighbours before comparing.
            let mut joined: Vec<Event> = Vec::new();
            for ev in got {
                match (joined.last_mut(), ev) {
                    (Some(Event::Text(a)), Event::Text(b)) => a.push_str(&b),
                    (_, ev) => joined.push(ev),
                }
            }
            assert_eq!(joined, whole, "pieces of {size} characters");
        }
        whole
    }

    fn text(s: &str) -> Event {
        Event::Text(s.to_string())
    }

    fn call(index: usize, name: &str, arguments: &str) -> Event {
        Event::Call {
            index,
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    #[test]
    fn text_without_a_call_is_passed_through_unchanged() {
        assert_eq!(parse("Hello there.\n"), vec![text("Hello there.\n")]);
        assert_eq!(
            parse("  a < b, and <tool is not a tag\n\n"),
            vec![text("  a < b, and <tool is not a tag\n\n")]
        );
        assert_eq!(parse(""), vec![]);
    }

    #[test]
    fn one_call() {
        assert_eq!(
            parse("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"),
            vec![call(0, "get_weather", "{\"city\": \"Paris\"}")]
        );
    }

    #[test]
    fn text_then_calls_and_the_whitespace_between_is_dropped() {
        let out = parse(
            "Let me check.\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>\n",
        );
        assert_eq!(
            out,
            vec![
                text("Let me check."),
                call(0, "get_weather", "{\"city\": \"Paris\"}"),
                call(1, "now", "{}"),
            ]
        );
    }

    #[test]
    fn text_after_a_call_is_kept() {
        let out =
            parse("<tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>\n\nDone.\n");
        assert_eq!(out, vec![call(0, "now", "{}"), text("Done.")]);
    }

    #[test]
    fn a_broken_call_is_text() {
        // Broken JSON.
        let raw = "<tool_call>\n{\"name\": \"now\", \"arguments\": {\n</tool_call>";
        assert_eq!(parse(raw), vec![text(raw)]);
        // A tool the request did not offer.
        let raw = "<tool_call>\n{\"name\": \"rm_rf\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(parse(raw), vec![text(raw)]);
        // Arguments that are not an object.
        let raw = "<tool_call>\n{\"name\": \"now\", \"arguments\": [1]}\n</tool_call>";
        assert_eq!(parse(raw), vec![text(raw)]);
        // No name.
        let raw = "<tool_call>\n{\"arguments\": {}}\n</tool_call>";
        assert_eq!(parse(raw), vec![text(raw)]);
    }

    #[test]
    fn arguments_written_as_a_string_or_left_out() {
        assert_eq!(
            parse("<tool_call>{\"name\": \"get_weather\", \"arguments\": \"{\\\"city\\\": \\\"Rome\\\"}\"}</tool_call>"),
            vec![call(0, "get_weather", "{\"city\": \"Rome\"}")]
        );
        assert_eq!(
            parse("<tool_call>{\"name\": \"now\"}</tool_call>"),
            vec![call(0, "now", "{}")]
        );
    }

    #[test]
    fn a_call_cut_off_by_the_end_of_the_output() {
        // The body is whole: a call.
        assert_eq!(
            parse("<tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n"),
            vec![call(0, "now", "{}")]
        );
        // The body is not: text, as written.
        let raw = "Checking.\n<tool_call>\n{\"name\": \"get_weather\", \"argu";
        assert_eq!(parse(raw), vec![text(raw)]);
    }

    #[test]
    fn argument_order_and_values_are_kept() {
        let out = parse(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"zeta\": 1, \"alpha\": [true, null, 2.5], \"city\": \"São Paulo\"}}\n</tool_call>",
        );
        assert_eq!(
            out,
            vec![call(
                0,
                "get_weather",
                "{\"zeta\": 1, \"alpha\": [true, null, 2.5], \"city\": \"São Paulo\"}"
            )]
        );
    }

    #[test]
    fn the_template_is_recognised_exactly() {
        assert_eq!(
            ToolFormat::for_template(QWEN25_TEMPLATE),
            Some(ToolFormat::Hermes)
        );
        assert_eq!(
            ToolFormat::for_template(&format!("{QWEN25_TEMPLATE}\n")),
            Some(ToolFormat::Hermes)
        );
        assert_eq!(
            ToolFormat::for_template(&QWEN25_TEMPLATE.replace("# Tools", "# Tool")),
            None
        );
        assert_eq!(ToolFormat::for_template(""), None);
    }

    #[test]
    fn requests_are_checked_by_name() {
        let bad = |json: &str| messages_from_json(&Value::parse(json).unwrap()).unwrap_err();
        assert!(bad(r#"[{"role": "developer", "content": "x"}]"#).contains("developer"));
        assert!(
            bad(r#"[{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]"#)
                .contains("text part")
        );
        assert!(bad(
            r#"[{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "nope"}}]}]"#
        )
        .contains("not JSON"));
        assert!(bad(
            r#"[{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}]"#
        )
        .contains("JSON object"));
        let tools = |json: &str| tools_from_json(Some(&Value::parse(json).unwrap())).unwrap_err();
        assert!(tools(r#"[{"type": "retrieval"}]"#).contains("retrieval"));
        assert!(tools(r#"[{"type": "function", "function": {}}]"#).contains("name"));
        assert!(tools(
            r#"[{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "a"}}]"#
        )
        .contains("twice"));
        assert!(tools_from_json(None).unwrap().is_empty());
    }
}
