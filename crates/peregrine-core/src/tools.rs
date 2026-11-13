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
//! A template is recognised by comparing it with the copies kept under `tools/`: a file whose
//! template differs is not guessed at, and a request that carries tools is refused for it.
//! A format may keep several copies: publishers ship variants of one template whose text
//! differs and whose prompts do not, and each copy here has been rendered through llama.cpp
//! on the format's whole fixture set and found byte-identical to the others.
//!
//! # Formats
//!
//! * [`ToolFormat::Hermes`]: Qwen 2.5's ChatML template. Tool signatures are JSON lines
//!   inside `<tools>` in the system message; a call is `{"name": ..., "arguments": {...}}`
//!   inside `<tool_call>`; results come back inside `<tool_response>` in a user turn,
//!   consecutive results sharing one turn.
//! * [`ToolFormat::Qwen3`]: Qwen 3's template. Tools and calls are written as in `Hermes`
//!   (with no default system message), messages are written untrimmed, and the assistant's
//!   turns after the last user question carry their thinking block. The model opens its own
//!   thinking block; with thinking off the prompt ends with an empty one. The publishers'
//!   wordings of this template agree except on two corner rules ([`Qwen3Text`]).
//! * [`ToolFormat::QwenXml`]: Qwen 3.5's template, which Bonsai 2 files carry. The same
//!   `<tools>` list, but a call is `<function=name>` with one `<parameter=key>` block per
//!   argument inside `<tool_call>`, values written bare (a string as it is, `True`, `None`,
//!   JSON for lists and objects). The assistant's turns after the last user question carry
//!   their thinking block, and the prompt ends inside one, or after an empty one when
//!   thinking is off.

use crate::constrain::{Constraint, Part, Schema};
use crate::json::Value;

/// ChatML's turn markers, as the Qwen templates write them.
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// The system message Qwen 2.5's template writes when the conversation has none.
const QWEN25_DEFAULT_SYSTEM: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// Qwen 2.5's chat template, as its GGUF files carry it (`tokenizer.chat_template`).
pub const QWEN25_TEMPLATE: &str = include_str!("tools/qwen25.jinja");

/// Qwen 3.5's chat template, as Bonsai 2's GGUF files and Qwen's own 4B and 9B files
/// carry it: the model thinks unless `enable_thinking` is false.
pub const QWEN35_TEMPLATE: &str = include_str!("tools/qwen35.jinja");

/// The variants of Qwen 3.5's template that render the same prompts as [`QWEN35_TEMPLATE`].
///
/// * `qwen35-optin.jinja`: Qwen's own 0.8B and 2B files. The model thinks only when
///   `enable_thinking` is true. llama.cpp always defines `enable_thinking` (true unless the
///   request says otherwise), and so does this renderer, so the "not defined" branch that
///   tells the two templates apart is never taken.
/// * `qwen35-unsloth.jinja` and `qwen35-unsloth-optin.jinja`: unsloth's files, with either
///   thinking rule. They walk a call's arguments with `is mapping` and a key loop where
///   Qwen's template uses `|items`: the same text for an object, and arguments that are
///   not an object are refused before rendering ([`messages_from_json`]).
///
/// Each was rendered by llama.cpp v0.9.0 on every case of `tests/fixtures/tools-qwen35.jsonl`
/// and gave the same prompts and the same refusals (`scripts/tool-fixtures.py --check`).
pub const QWEN35_VARIANTS: [&str; 3] = [
    include_str!("tools/qwen35-optin.jinja"),
    include_str!("tools/qwen35-unsloth.jinja"),
    include_str!("tools/qwen35-unsloth-optin.jinja"),
];

/// Qwen 3's chat template as the model's Hugging Face repository has it, which is what
/// bartowski's and lmstudio-community's GGUF files carry.
pub const QWEN3_TEMPLATE: &str = include_str!("tools/qwen3.jinja");

/// The other published wordings of Qwen 3's template, each with the rules it follows:
/// Qwen's own GGUF files, unsloth's files up to 1.7B, and unsloth's larger files.
///
/// llama.cpp v0.9.0 renders every case of `tests/fixtures/tools-qwen3.jsonl` to the same
/// bytes with all four. They part on two corner cases, kept apart by [`Qwen3Text`] and
/// pinned per wording in `tests/fixtures/tools-qwen3-edges.jsonl`.
pub const QWEN3_VARIANTS: [(&str, Qwen3Text); 3] = [
    (include_str!("tools/qwen3-gguf.jinja"), Qwen3Text::QwenGguf),
    (
        include_str!("tools/qwen3-unsloth.jinja"),
        Qwen3Text::Upstream,
    ),
    (
        include_str!("tools/qwen3-unsloth-large.jinja"),
        Qwen3Text::UnslothLarge,
    ),
];

/// Which wording of Qwen 3's template a file carries, where the wordings disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3Text {
    /// The repository's template (bartowski, lmstudio-community, unsloth up to 1.7B).
    Upstream,
    /// Qwen's own GGUF files: a user message that contains a `<tool_response>` block
    /// anywhere is taken for a tool's result, not only one that is exactly such a block.
    QwenGguf,
    /// unsloth's 4B and larger files: thinking that is only whitespace counts as none.
    UnslothLarge,
}

/// The thinking block's tags in the Qwen 3 and 3.5 templates.
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// What Qwen 3.5's template writes after the tool list (`tools/qwen35.jinja`).
const QWEN35_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// A family of templates that writes tools and calls the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// Qwen 2.5: JSON calls inside `<tool_call>` (the "Hermes" convention).
    Hermes,
    /// Qwen 3: the same calls, and a thinking block the model opens itself.
    Qwen3(Qwen3Text),
    /// Qwen 3.5 and Bonsai 2: `<function=name>` and `<parameter=key>` blocks inside
    /// `<tool_call>`.
    QwenXml,
}

impl ToolFormat {
    /// The format of a file's chat template, or `None` when the template is not one whose
    /// tool section Peregrine reproduces. The comparison is exact (trailing whitespace
    /// aside): a variant is refused rather than rendered almost right.
    pub fn for_template(template: &str) -> Option<ToolFormat> {
        if template.trim_end() == QWEN25_TEMPLATE.trim_end() {
            return Some(ToolFormat::Hermes);
        }
        if template.trim_end() == QWEN3_TEMPLATE.trim_end() {
            return Some(ToolFormat::Qwen3(Qwen3Text::Upstream));
        }
        for (text, rules) in QWEN3_VARIANTS {
            if template.trim_end() == text.trim_end() {
                return Some(ToolFormat::Qwen3(rules));
            }
        }
        if template.trim_end() == QWEN35_TEMPLATE.trim_end()
            || QWEN35_VARIANTS
                .iter()
                .any(|v| template.trim_end() == v.trim_end())
        {
            return Some(ToolFormat::QwenXml);
        }
        None
    }

    /// Whether `prompt`, as [`render_with`] wrote it for this format, ends inside a thinking
    /// block the template opened for the model. The model's reply then begins with its
    /// reasoning and writes only the closing tag, so whoever reads the reply must start in
    /// the reasoning span.
    pub fn opens_reasoning(self, prompt: &str) -> bool {
        match self {
            ToolFormat::Hermes | ToolFormat::Qwen3(_) => false,
            ToolFormat::QwenXml => prompt.ends_with("<think>\n"),
        }
    }

    /// Whether the family's models think: their replies may carry a thinking block, and
    /// their templates take the thinking switch.
    pub fn thinks(self) -> bool {
        match self {
            ToolFormat::Hermes => false,
            ToolFormat::Qwen3(_) | ToolFormat::QwenXml => true,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ToolFormat::Hermes => "hermes",
            ToolFormat::Qwen3(_) => "qwen3",
            ToolFormat::QwenXml => "qwen-xml",
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
    /// What the assistant wrote: text, calls, or both, and its thinking when the client
    /// sends it back (`reasoning_content`).
    Assistant {
        content: String,
        reasoning: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    /// A tool's result, sent back by the client.
    Tool {
        content: String,
    },
    /// A message with a role the template has no branch for. Only a format whose template
    /// passes over such a message produces it (Qwen 3's); it writes nothing, but it still
    /// stands between its neighbours, as it does in the template's loop.
    Other,
}

/// A message's text: a string, `null` (an assistant message that only calls tools), or a
/// list of text parts joined by `sep`. llama.cpp joins the parts with a newline for a
/// template that reads content as a string (Qwen 2.5); Qwen 3.5's template reads the parts
/// itself and writes them one after another.
fn content_of(m: &Value, i: usize, sep: &str) -> Result<String, String> {
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
            Ok(texts.join(sep))
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

/// Which call a reply is required to make: OpenAI's `tool_choice` beyond `"auto"` and
/// `"none"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Forced {
    /// `"required"`: at least one call, to any of the request's tools.
    Any,
    /// A named function: exactly one call, to this tool.
    Named(String),
}

/// Where a reply's thinking stands when a constraint is laid over the reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reasoning {
    /// The reply begins with its content: the model does not think, or the prompt
    /// already closed an empty thinking block.
    None,
    /// The prompt opened the thinking block (Qwen 3.5 with thinking on): the reply begins
    /// with the model's reasoning and writes only the closing tag.
    Open,
    /// The model may open a thinking block itself (Qwen 3 with thinking on).
    Optional,
}

/// `content`, after the reply's thinking (S6.6b): the reasoning is free text up to the
/// closing tag, then the two newlines the templates put before the content, then the
/// constraint. The thinking is the model's own; only what follows it is held.
pub fn after_reasoning(reasoning: Reasoning, content: Constraint) -> Constraint {
    let close = || {
        vec![
            Part::Until(THINK_CLOSE.to_string()),
            Part::Literal("\n\n".to_string()),
        ]
    };
    match reasoning {
        Reasoning::None => content,
        Reasoning::Open => Constraint::concat(vec![Constraint::sequence(close()), content]),
        Reasoning::Optional => {
            let mut block = vec![Part::Literal(THINK_OPEN.to_string())];
            block.extend(close());
            Constraint::concat(vec![
                Constraint::repeat(Constraint::sequence(block), 0, Some(1)),
                content,
            ])
        }
    }
}

/// The constraint that makes a reply consist of tool calls and nothing else (S6.6b).
///
/// The reply is written in the format's own call syntax, byte for byte as its template
/// writes an earlier turn's calls, so what the sampler is held to is text the model was
/// trained on:
///
/// * the Hermes convention (Qwen 2.5, Qwen 3):
///   `<tool_call>\n{"name": "T", "arguments": A}\n</tool_call>`, where `A` is one JSON
///   value of the tool's own `parameters` schema;
/// * the XML convention (Qwen 3.5): `<tool_call>\n<function=T>\n`, one
///   `<parameter=K>\nV\n</parameter>\n` block per property in the schema's order (a
///   property that is not required may be left out), `</function>\n</tool_call>`. A
///   value is written bare, as the template writes it: a string as free text up to the
///   closing tag, a boolean as `True` or `False`, null as `None`, an `enum` as one of its
///   values, a number, list or object as JSON of the property's schema.
///
/// Calls are joined by one newline. The wrapper is literal, so the arguments are valid
/// for the tool by construction.
///
/// * [`Forced::Any`]: one call to any tool, and with `parallel` any number of further
///   calls (the model ends the reply when it has made the calls it wants).
/// * [`Forced::Named`]: exactly one call to that tool.
///
/// `tools` is [`tools_from_json`]'s list. An error names the tool it is about: a name
/// that is not in the list, parameters that are not an object schema, or a schema keyword
/// the constraint compiler refuses.
pub fn forced_calls(
    format: ToolFormat,
    tools: &[Value],
    forced: &Forced,
    parallel: bool,
) -> Result<Constraint, String> {
    if tools.is_empty() {
        return Err("a required or named tool_choice needs tools".into());
    }
    let mut calls = Vec::new();
    for (i, t) in tools.iter().enumerate() {
        let Some(f) = t.get("function") else { continue };
        let Some(name) = f.get("name").and_then(Value::as_str) else {
            continue;
        };
        if let Forced::Named(want) = forced {
            if want != name {
                continue;
            }
        }
        // The templates write a call's name as it is, between quotes or inside a tag; a
        // name that would need an escape there has no faithful literal.
        if name
            .chars()
            .any(|c| matches!(c, '"' | '\\' | '>' | '<') || c.is_control())
        {
            return Err(format!(
                "tools[{i}].function.name {name:?} cannot be the target of a required or named tool_choice: the call syntax cannot write it"
            ));
        }
        let arguments = argument_schema(f.get("parameters"), i)?;
        let call = match format {
            ToolFormat::Hermes | ToolFormat::Qwen3(_) => Constraint::sequence(vec![
                Part::Literal(format!(
                    "<tool_call>\n{{\"name\": \"{name}\", \"arguments\": "
                )),
                Part::Json(Schema::new(arguments)),
                Part::Literal("}\n</tool_call>".into()),
            ]),
            ToolFormat::QwenXml => xml_call(name, &arguments, i)?,
        };
        // Each tool's schema is checked on its own, so the error can name the tool.
        call.compile()
            .and_then(|_| crate::constrain::Matcher::new(&call).map(|_| ()))
            .map_err(|e| format!("tools[{i}].function.parameters: {}", e.message()))?;
        calls.push(call);
    }
    if calls.is_empty() {
        return Err(match forced {
            Forced::Named(want) => {
                format!("tool_choice names {want:?}, which is not one of the request's tools")
            }
            Forced::Any => "a required tool_choice needs tools".into(),
        });
    }
    let one = if calls.len() == 1 {
        calls.remove(0)
    } else {
        Constraint::any_of(calls)
    };
    let out = match forced {
        Forced::Any if parallel => Constraint::concat(vec![
            one.clone(),
            Constraint::repeat(
                Constraint::concat(vec![
                    Constraint::sequence(vec![Part::Literal("\n".into())]),
                    one,
                ]),
                0,
                None,
            ),
        ]),
        _ => one,
    };
    // The whole constraint can be too large to match even when every tool's own schema
    // is not (many tools); that is the request's doing, refused here.
    crate::constrain::Matcher::new(&out).map_err(|e| format!("tools: {}", e.message()))?;
    Ok(out)
}

/// One call in the XML convention: the function tag, a block per property, the closing
/// tags. `arguments` is the tool's object schema ([`argument_schema`]).
fn xml_call(name: &str, arguments: &Value, i: usize) -> Result<Constraint, String> {
    let required: Vec<&str> = arguments
        .get("required")
        .and_then(Value::as_array)
        .unwrap_or(&[])
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let mut parts = vec![Constraint::sequence(vec![Part::Literal(format!(
        "<tool_call>\n<function={name}>\n"
    ))])];
    let properties = arguments.get("properties").and_then(Value::as_object);
    for (key, schema) in properties.unwrap_or(&[]) {
        if key.is_empty()
            || key
                .chars()
                .any(|c| matches!(c, '>' | '<') || c.is_control())
        {
            return Err(format!(
                "tools[{i}].function.parameters: the property {key:?} cannot be written as a <parameter=…> tag"
            ));
        }
        let open = format!("<parameter={key}>\n");
        const CLOSE: &str = "\n</parameter>\n";
        let block = match xml_value(schema, arguments) {
            XmlValue::Text => {
                Constraint::sequence(vec![Part::Literal(open), Part::Until(CLOSE.to_string())])
            }
            XmlValue::Choices(values) => Constraint::any_of(
                values
                    .into_iter()
                    .map(|v| Constraint::sequence(vec![Part::Literal(format!("{open}{v}{CLOSE}"))]))
                    .collect(),
            ),
            XmlValue::Json(schema) => Constraint::sequence(vec![
                Part::Literal(open),
                Part::Json(Schema::new(schema)),
                Part::Literal(CLOSE.to_string()),
            ]),
        };
        if required.contains(&key.as_str()) {
            parts.push(block);
        } else {
            parts.push(Constraint::repeat(block, 0, Some(1)));
        }
    }
    parts.push(Constraint::sequence(vec![Part::Literal(
        "</function>\n</tool_call>".into(),
    )]));
    Ok(Constraint::concat(parts))
}

/// How one argument's value is held in the XML convention.
enum XmlValue {
    /// Free text up to the closing tag: a string, and any schema with no single bare
    /// spelling (the parser reads such a value as text too).
    Text,
    /// One of these exact texts: an `enum`'s or `const`'s values, a boolean, null.
    Choices(Vec<String>),
    /// One JSON value of this schema: a number, a list, an object.
    Json(Value),
}

/// The value form of a property, following how the template writes a value
/// ([`qwen35_value`]) and how [`CallParser`] reads one back.
fn xml_value(schema: &Value, root: &Value) -> XmlValue {
    let choices = |values: &[Value]| {
        XmlValue::Choices(values.iter().map(qwen35_value).collect::<Vec<String>>())
    };
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        if !values.is_empty() {
            return choices(values);
        }
    }
    if let Some(value) = schema.get("const") {
        return choices(std::slice::from_ref(value));
    }
    match schema.get("type") {
        Some(Value::Str(t)) => match t.as_str() {
            "boolean" => XmlValue::Choices(vec!["True".into(), "False".into()]),
            "null" => XmlValue::Choices(vec!["None".into()]),
            "integer" | "number" | "array" | "object" => match with_definitions(schema, root) {
                Some(schema) => XmlValue::Json(schema),
                None => XmlValue::Text,
            },
            _ => XmlValue::Text,
        },
        _ => XmlValue::Text,
    }
}

/// `schema` as a schema that stands on its own: a property's schema is compiled apart
/// from the tool's, so `$ref`s into the tool's `$defs` or `definitions` need those tables
/// beside it. `None` when it refers to the tool's root (`"$ref": "#"`), which has no
/// meaning once the property stands alone.
fn with_definitions(schema: &Value, root: &Value) -> Option<Value> {
    fn refers(v: &Value, to_root: &mut bool) -> bool {
        match v {
            Value::Obj(fields) => {
                let mut any = false;
                for (k, v) in fields {
                    if k == "$ref" {
                        any = true;
                        if v.as_str() == Some("#") {
                            *to_root = true;
                        }
                    }
                    any |= refers(v, to_root);
                }
                any
            }
            Value::Arr(items) => {
                let mut any = false;
                for v in items {
                    any |= refers(v, to_root);
                }
                any
            }
            _ => false,
        }
    }
    let mut to_root = false;
    if !refers(schema, &mut to_root) {
        return Some(schema.clone());
    }
    if to_root {
        return None;
    }
    let mut fields = schema.as_object()?.to_vec();
    for table in ["$defs", "definitions"] {
        if schema.get(table).is_none() {
            if let Some(defs) = root.get(table) {
                fields.push((table.to_string(), defs.clone()));
            }
        }
    }
    Some(Value::Obj(fields))
}

/// The schema a forced call's `arguments` must satisfy: the tool's `parameters`.
///
/// Arguments are a JSON object in every format, so the schema must describe one. Missing
/// or empty parameters mean the function takes none, and the arguments are `{}`. A schema
/// without `type` is read as an object schema; any other type is refused.
fn argument_schema(parameters: Option<&Value>, i: usize) -> Result<Value, String> {
    let none = || {
        Value::Obj(vec![
            ("type".to_string(), Value::Str("object".to_string())),
            ("properties".to_string(), Value::Obj(Vec::new())),
            ("additionalProperties".to_string(), Value::Bool(false)),
        ])
    };
    let Some(p) = parameters else {
        return Ok(none());
    };
    let Some(fields) = p.as_object() else {
        return Err(format!("tools[{i}].function.parameters must be an object"));
    };
    if fields.is_empty() {
        return Ok(none());
    }
    match p.get("type") {
        Some(Value::Str(t)) if t == "object" => Ok(p.clone()),
        Some(_) => Err(format!(
            "tools[{i}].function.parameters must be a schema of type \"object\""
        )),
        None => {
            let mut fields = fields.to_vec();
            fields.insert(0, ("type".to_string(), Value::Str("object".to_string())));
            Ok(Value::Obj(fields))
        }
    }
}

/// The `messages` of a chat request as [`Msg`]s.
///
/// OpenAI's shapes: `system`, `user`, `assistant` (with optional `tool_calls`) and `tool`.
/// A call's `arguments` is a JSON object written as a string; an object is accepted too.
/// Anything else is refused by name, so a conversation is never rendered with a message
/// silently dropped.
pub fn messages_from_json(messages: &Value) -> Result<Vec<Msg>, String> {
    messages_from_json_for(ToolFormat::Hermes, messages)
}

/// [`messages_from_json`] for a given format: the formats differ in how a list of text
/// parts becomes one text.
pub fn messages_from_json_for(format: ToolFormat, messages: &Value) -> Result<Vec<Msg>, String> {
    let sep = match format {
        ToolFormat::Hermes | ToolFormat::Qwen3(_) => "\n",
        ToolFormat::QwenXml => "",
    };
    let list = messages.as_array().ok_or("messages must be an array")?;
    let mut out = Vec::with_capacity(list.len());
    for (i, m) in list.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("messages[{i}].role must be a string"))?;
        match role {
            "system" => out.push(Msg::System(content_of(m, i, sep)?)),
            "user" => out.push(Msg::User(content_of(m, i, sep)?)),
            "tool" => out.push(Msg::Tool {
                content: content_of(m, i, sep)?,
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
                let reasoning = match m.get("reasoning_content") {
                    Some(Value::Str(r)) => Some(r.clone()),
                    _ => None,
                };
                out.push(Msg::Assistant {
                    content: content_of(m, i, sep)?,
                    reasoning,
                    tool_calls,
                });
            }
            // Qwen 3's template has no branch for another role and no refusal either.
            _ if matches!(format, ToolFormat::Qwen3(_)) => out.push(Msg::Other),
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
    render_with(format, tools, messages, RenderOptions::default())
}

/// What a request may choose about its prompt besides the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderOptions {
    /// Whether the model may think before it answers (`enable_thinking`). Formats without
    /// a thinking block ignore it.
    pub thinking: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions { thinking: true }
    }
}

/// [`render`] with options.
pub fn render_with(
    format: ToolFormat,
    tools: &[Value],
    messages: &[Msg],
    options: RenderOptions,
) -> Result<String, String> {
    match format {
        ToolFormat::Hermes => render_qwen25(tools, messages),
        ToolFormat::Qwen3(text) => render_qwen3(tools, messages, options.thinking, text),
        ToolFormat::QwenXml => render_qwen35(tools, messages, options.thinking),
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
                ..
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
            // Only Qwen 3's reader produces this.
            Msg::Other => {}
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
                ..
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

/// Jinja's `trim`: the whitespace C's `isspace` knows, from both ends.
pub(crate) fn jinja_trim(text: &str) -> &str {
    text.trim_matches(|c| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'))
}

/// Qwen 3's template, branch for branch (`tools/qwen3.jinja`).
///
/// Nothing is trimmed: a message is written as the client sent it. A message with a role
/// the template has no branch for writes nothing. `text` selects the two rules on which the
/// published wordings of the template differ.
fn render_qwen3(
    tools: &[Value],
    messages: &[Msg],
    thinking: bool,
    text: Qwen3Text,
) -> Result<String, String> {
    if messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    let mut s = String::new();
    let system = match &messages[0] {
        Msg::System(text) => Some(text.as_str()),
        _ => None,
    };
    if !tools.is_empty() {
        s.push_str(IM_START);
        s.push_str("system\n");
        if let Some(text) = system {
            s.push_str(text);
            s.push_str("\n\n");
        }
        s.push_str(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\
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
        s.push_str(IM_END);
        s.push('\n');
    } else if let Some(text) = system {
        s.push_str(IM_START);
        s.push_str("system\n");
        s.push_str(text);
        s.push_str(IM_END);
        s.push('\n');
    }

    // The last real question: the last user message that is not a tool's result written as
    // a user turn. With none, the template leaves the index at the last message, so no
    // assistant turn is "after" it.
    let is_result = |t: &str| match text {
        Qwen3Text::QwenGguf => t.contains("<tool_response>") && t.contains("</tool_response>"),
        Qwen3Text::Upstream | Qwen3Text::UnslothLarge => {
            t.starts_with("<tool_response>") && t.ends_with("</tool_response>")
        }
    };
    let last = messages.len() - 1;
    let last_query = messages
        .iter()
        .rposition(|m| matches!(m, Msg::User(t) if !is_result(t)))
        .unwrap_or(last);

    // A last assistant message of text is a continuation: it is written like any other and
    // left open, and no new assistant turn follows it.
    let continued = match messages.last() {
        Some(Msg::Assistant { tool_calls, .. }) if tool_calls.is_empty() => true,
        Some(Msg::Assistant { .. }) => {
            return Err(
                "the conversation ends with an assistant tool call; send the tool's result".into(),
            )
        }
        _ => false,
    };
    for (i, m) in messages.iter().enumerate() {
        match m {
            // The first system message is the header above; a later one is a turn.
            Msg::System(text) => {
                if i > 0 {
                    s.push_str(IM_START);
                    s.push_str("system\n");
                    s.push_str(text);
                    s.push_str(IM_END);
                    s.push('\n');
                }
            }
            Msg::User(text) => {
                s.push_str(IM_START);
                s.push_str("user\n");
                s.push_str(text);
                s.push_str(IM_END);
                s.push('\n');
            }
            Msg::Assistant {
                content,
                reasoning,
                tool_calls,
            } => {
                // The thinking the client sent back, or the block still inside the text.
                let (reasoning, content) = match reasoning {
                    Some(r) => (r.as_str(), content.as_str()),
                    None if content.contains(THINK_CLOSE) => {
                        let before = content.split(THINK_CLOSE).next().unwrap_or("");
                        let before = before.trim_end_matches('\n');
                        let inner = before.rsplit(THINK_OPEN).next().unwrap_or("");
                        let after = content.rsplit(THINK_CLOSE).next().unwrap_or("");
                        (
                            inner.trim_start_matches('\n'),
                            after.trim_start_matches('\n'),
                        )
                    }
                    None => ("", content.as_str()),
                };
                let has_reasoning = match text {
                    Qwen3Text::UnslothLarge => !jinja_trim(reasoning).is_empty(),
                    Qwen3Text::Upstream | Qwen3Text::QwenGguf => !reasoning.is_empty(),
                };
                s.push_str(IM_START);
                s.push_str("assistant\n");
                if i > last_query && (i == last || has_reasoning) {
                    s.push_str(THINK_OPEN);
                    s.push('\n');
                    s.push_str(reasoning.trim_matches('\n'));
                    s.push('\n');
                    s.push_str(THINK_CLOSE);
                    s.push_str("\n\n");
                    s.push_str(content.trim_start_matches('\n'));
                } else {
                    s.push_str(content);
                }
                for (j, c) in tool_calls.iter().enumerate() {
                    if j > 0 || !content.is_empty() {
                        s.push('\n');
                    }
                    s.push_str("<tool_call>\n{\"name\": \"");
                    s.push_str(&c.name);
                    s.push_str("\", \"arguments\": ");
                    s.push_str(&c.arguments.to_jinja_json());
                    s.push_str("}\n</tool_call>");
                }
                if !(continued && i == last) {
                    s.push_str(IM_END);
                    s.push('\n');
                }
            }
            // Consecutive results share one user turn.
            Msg::Tool { content } => {
                if i == 0 || !matches!(messages[i - 1], Msg::Tool { .. }) {
                    s.push_str(IM_START);
                    s.push_str("user");
                }
                s.push_str("\n<tool_response>\n");
                s.push_str(content);
                s.push_str("\n</tool_response>");
                if i == last || !matches!(messages[i + 1], Msg::Tool { .. }) {
                    s.push_str(IM_END);
                    s.push('\n');
                }
            }
            Msg::Other => {}
        }
    }
    if !continued {
        s.push_str(IM_START);
        s.push_str("assistant\n");
        if !thinking {
            s.push_str(THINK_OPEN);
            s.push_str("\n\n");
            s.push_str(THINK_CLOSE);
            s.push_str("\n\n");
        }
    }
    Ok(s)
}

/// One argument's value as Qwen 3.5's template writes it: a list or an object as JSON,
/// anything else through Jinja's `string` (a string as it is, `True`, `False`, `None`,
/// a number in Python's spelling).
fn qwen35_value(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => crate::json::float_repr(*f),
        Value::Arr(_) | Value::Obj(_) => v.to_jinja_json(),
    }
}

/// Qwen 3.5's template, branch for branch (`tools/qwen35.jinja`).
fn render_qwen35(tools: &[Value], messages: &[Msg], thinking: bool) -> Result<String, String> {
    if messages.is_empty() {
        return Err("No messages provided.".into());
    }
    let mut s = String::new();
    let system = match &messages[0] {
        Msg::System(text) => Some(jinja_trim(text)),
        _ => None,
    };
    if !tools.is_empty() {
        s.push_str(IM_START);
        s.push_str("system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for t in tools {
            s.push('\n');
            s.push_str(&t.to_jinja_json());
        }
        s.push_str("\n</tools>");
        s.push_str(QWEN35_INSTRUCTIONS);
        if let Some(text) = system.filter(|t| !t.is_empty()) {
            s.push_str("\n\n");
            s.push_str(text);
        }
        s.push_str(IM_END);
        s.push('\n');
    } else if let Some(text) = system {
        s.push_str(IM_START);
        s.push_str("system\n");
        s.push_str(text);
        s.push_str(IM_END);
        s.push('\n');
    }

    // The last real question: the last user message that is not a tool's result written
    // as a user turn. Assistant turns after it keep their thinking block.
    let last_query = messages.iter().rposition(|m| match m {
        Msg::User(text) => {
            let t = jinja_trim(text);
            !(t.starts_with("<tool_response>") && t.ends_with("</tool_response>"))
        }
        _ => false,
    });
    let Some(last_query) = last_query else {
        return Err("No user query found in messages.".into());
    };

    // A last assistant message of text is a continuation: it is written like any other and
    // left open, and no new assistant turn follows it.
    let continued = match messages.last() {
        Some(Msg::Assistant { tool_calls, .. }) if tool_calls.is_empty() => true,
        Some(Msg::Assistant { .. }) => {
            return Err(
                "the conversation ends with an assistant tool call; send the tool's result".into(),
            )
        }
        _ => false,
    };
    let last = messages.len() - 1;
    for (i, m) in messages.iter().enumerate() {
        match m {
            // Only Qwen 3's reader produces this.
            Msg::Other => {}
            Msg::System(_) => {
                if i > 0 {
                    return Err("System message must be at the beginning.".into());
                }
            }
            Msg::User(text) => {
                s.push_str(IM_START);
                s.push_str("user\n");
                s.push_str(jinja_trim(text));
                s.push_str(IM_END);
                s.push('\n');
            }
            Msg::Assistant {
                content,
                reasoning,
                tool_calls,
            } => {
                let mut content = jinja_trim(content);
                // The thinking the client sent back, or the block still inside the text.
                let reasoning = match reasoning {
                    Some(r) => r.as_str(),
                    None => match content.find(THINK_CLOSE) {
                        Some(_) => {
                            let before = content.split(THINK_CLOSE).next().unwrap_or("");
                            let before = before.trim_end_matches('\n');
                            let inner = before.rsplit(THINK_OPEN).next().unwrap_or("");
                            let after = content.rsplit(THINK_CLOSE).next().unwrap_or("");
                            content = after.trim_start_matches('\n');
                            inner.trim_start_matches('\n')
                        }
                        None => "",
                    },
                };
                let reasoning = jinja_trim(reasoning);
                s.push_str(IM_START);
                s.push_str("assistant\n");
                if i > last_query {
                    s.push_str(THINK_OPEN);
                    s.push('\n');
                    s.push_str(reasoning);
                    s.push('\n');
                    s.push_str(THINK_CLOSE);
                    s.push_str("\n\n");
                }
                s.push_str(content);
                for (j, c) in tool_calls.iter().enumerate() {
                    if j > 0 {
                        s.push('\n');
                    } else if !jinja_trim(content).is_empty() {
                        s.push_str("\n\n");
                    }
                    s.push_str("<tool_call>\n<function=");
                    s.push_str(&c.name);
                    s.push_str(">\n");
                    for (key, value) in c.arguments.as_object().unwrap_or(&[]) {
                        s.push_str("<parameter=");
                        s.push_str(key);
                        s.push_str(">\n");
                        s.push_str(&qwen35_value(value));
                        s.push_str("\n</parameter>\n");
                    }
                    s.push_str("</function>\n</tool_call>");
                }
                if !(continued && i == last) {
                    s.push_str(IM_END);
                    s.push('\n');
                }
            }
            // Consecutive results share one user turn. A result that is the first message
            // has no turn opening: the template writes none, and neither does this.
            Msg::Tool { content } => {
                if i > 0 && !matches!(messages[i - 1], Msg::Tool { .. }) {
                    s.push_str(IM_START);
                    s.push_str("user");
                }
                s.push_str("\n<tool_response>\n");
                s.push_str(jinja_trim(content));
                s.push_str("\n</tool_response>");
                if i == last || !matches!(messages[i + 1], Msg::Tool { .. }) {
                    s.push_str(IM_END);
                    s.push('\n');
                }
            }
        }
    }
    if !continued {
        s.push_str(IM_START);
        s.push_str("assistant\n");
        s.push_str(THINK_OPEN);
        s.push('\n');
        if !thinking {
            s.push('\n');
            s.push_str(THINK_CLOSE);
            s.push_str("\n\n");
        }
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
    /// Per tool, the parameters its schema declares as strings (only [`CallParser::with_tools`]
    /// knows them). The XML format writes values bare, so the schema says whether `5` is
    /// the number or the text.
    string_params: Vec<(String, Vec<String>)>,
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
            string_params: Vec::new(),
        }
    }

    /// A parser that knows the request's tools (as [`tools_from_json`] returns them): their
    /// names, and which parameters are strings. Formats that write values bare need the
    /// second; without it a value that reads as JSON is taken as JSON.
    pub fn with_tools(format: ToolFormat, tools: &[Value]) -> CallParser {
        let mut parser = CallParser::new(format, tool_names(tools));
        for t in tools {
            let Some(f) = t.get("function") else { continue };
            let Some(name) = f.get("name").and_then(Value::as_str) else {
                continue;
            };
            let mut strings = Vec::new();
            let properties = f.get("parameters").and_then(|p| p.get("properties"));
            for (key, schema) in properties.and_then(Value::as_object).unwrap_or(&[]) {
                let is_string = match schema.get("type") {
                    Some(Value::Str(t)) => t == "string",
                    // A union such as ["string", "null"]: text unless it reads as null.
                    Some(Value::Arr(ts)) => ts.iter().any(|t| t.as_str() == Some("string")),
                    _ => false,
                };
                if is_string {
                    strings.push(key.clone());
                }
            }
            parser.string_params.push((name.to_string(), strings));
        }
        parser
    }

    /// How many calls have been found so far.
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Take the next piece of generated text; events that are now certain go to `out`.
    pub fn push(&mut self, text: &str, out: &mut Vec<Event>) {
        self.pending.push_str(text);
        match self.format {
            // Both formats frame a call the same way; they differ in its body.
            ToolFormat::Hermes | ToolFormat::Qwen3(_) | ToolFormat::QwenXml => {
                self.step_hermes(false, out)
            }
        }
    }

    /// The output has ended: decide everything that was held back.
    pub fn finish(&mut self, out: &mut Vec<Event>) {
        match self.format {
            ToolFormat::Hermes | ToolFormat::Qwen3(_) | ToolFormat::QwenXml => {
                self.step_hermes(true, out)
            }
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
        let call = match self.format {
            ToolFormat::Hermes | ToolFormat::Qwen3(_) => self.hermes_call(body),
            ToolFormat::QwenXml => self.xml_call(body),
        };
        match call {
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

impl CallParser {
    /// The call inside one `<tool_call>` body of the XML format, if it is one:
    /// `<function=name>`, then `<parameter=key>` blocks, then `</function>`.
    fn xml_call(&mut self, body: &str) -> Option<Event> {
        let body = body.trim();
        let rest = body.strip_prefix("<function=")?;
        let (name, rest) = rest.split_once('>')?;
        let name = name.trim();
        if !self.names.iter().any(|n| n == name) {
            return None;
        }
        // The closing tag may be missing when the output was cut off after the last
        // parameter; what is there must still be whole parameters.
        let mut rest = rest.strip_suffix("</function>").unwrap_or(rest);
        let strings: &[String] = self
            .string_params
            .iter()
            .find(|(n, _)| n == name)
            .map_or(&[], |(_, s)| s.as_slice());
        let typed = self.string_params.iter().any(|(n, _)| n == name);
        let mut arguments: Vec<(String, Value)> = Vec::new();
        loop {
            rest = rest.trim_start();
            if rest.is_empty() {
                break;
            }
            let after = rest.strip_prefix("<parameter=")?;
            let (key, after) = after.split_once('>')?;
            let (raw, after) = after.split_once("</parameter>")?;
            // The template puts one newline on each side of a value.
            let raw = raw.strip_prefix('\n').unwrap_or(raw);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            let is_string = typed && strings.iter().any(|s| s == key);
            let value = if is_string {
                Value::Str(raw.to_string())
            } else {
                match raw.trim() {
                    // The template writes Python's spellings; a model trained on it does too.
                    "True" | "true" => Value::Bool(true),
                    "False" | "false" => Value::Bool(false),
                    "None" | "null" => Value::Null,
                    other => match Value::parse(other) {
                        Ok(v) if !matches!(v, Value::Str(_)) => v,
                        _ => Value::Str(raw.to_string()),
                    },
                }
            };
            // A repeated parameter keeps its first place and its last value, as JSON does.
            match arguments.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = value,
                None => arguments.push((key.to_string(), value)),
            }
            rest = after;
        }
        let index = self.calls;
        self.calls += 1;
        Some(Event::Call {
            index,
            name: name.to_string(),
            arguments: Value::Obj(arguments).to_jinja_json(),
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
        parse_with(text, || CallParser::new(ToolFormat::Hermes, names()))
    }

    fn parse_with(text: &str, parser: impl Fn() -> CallParser) -> Vec<Event> {
        let mut whole = Vec::new();
        let mut p = parser();
        p.push(text, &mut whole);
        p.finish(&mut whole);
        let chars: Vec<char> = text.chars().collect();
        for size in 1..=chars.len().clamp(1, 24) {
            let mut got = Vec::new();
            let mut p = parser();
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
    // --- the XML format (Qwen 3.5, Bonsai 2) ------------------------------------------

    fn xml_tools() -> Vec<Value> {
        let json = r#"[
            {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object",
              "properties": {"query": {"type": "string"}, "limit": {"type": "integer"},
                             "safe": {"type": "boolean"}, "site": {"type": ["string", "null"]},
                             "filters": {"type": "object"}}}}},
            {"type": "function", "function": {"name": "now"}}
        ]"#;
        tools_from_json(Some(&Value::parse(json).unwrap())).unwrap()
    }

    fn xml(text: &str) -> Vec<Event> {
        parse_with(text, || {
            CallParser::with_tools(ToolFormat::QwenXml, &xml_tools())
        })
    }

    #[test]
    fn an_xml_call_takes_its_types_from_the_schema() {
        let out = xml(
            "<tool_call>\n<function=web_search>\n<parameter=query>\n5\n</parameter>\n<parameter=limit>\n5\n</parameter>\n<parameter=safe>\nTrue\n</parameter>\n<parameter=site>\nexample.com\n</parameter>\n<parameter=filters>\n{\"lang\": [\"en\", \"de\"]}\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(
            out,
            vec![call(
                0,
                "web_search",
                "{\"query\": \"5\", \"limit\": 5, \"safe\": true, \"site\": \"example.com\", \"filters\": {\"lang\": [\"en\", \"de\"]}}"
            )]
        );
    }

    #[test]
    fn an_xml_string_keeps_its_lines_and_its_spaces() {
        let out = xml(
            "<tool_call>\n<function=web_search>\n<parameter=query>\nline one\n  line two  \n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(
            out,
            vec![call(
                0,
                "web_search",
                "{\"query\": \"line one\\n  line two  \"}"
            )]
        );
    }

    #[test]
    fn xml_text_then_two_calls_and_a_call_without_parameters() {
        let out = xml(
            "I will search, then check the time.\n\n<tool_call>\n<function=web_search>\n<parameter=query>\nrust\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=now>\n</function>\n</tool_call>",
        );
        assert_eq!(
            out,
            vec![
                text("I will search, then check the time."),
                call(0, "web_search", "{\"query\": \"rust\"}"),
                call(1, "now", "{}"),
            ]
        );
    }

    #[test]
    fn an_xml_value_with_no_schema_is_json_when_it_reads_as_json() {
        // `now` declares no parameters: numbers, booleans and `None` are taken as such,
        // anything else is text.
        let out = xml(
            "<tool_call>\n<function=now>\n<parameter=zone>\nUTC\n</parameter>\n<parameter=offset>\n-3\n</parameter>\n<parameter=dst>\nNone\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(
            out,
            vec![call(
                0,
                "now",
                "{\"zone\": \"UTC\", \"offset\": -3, \"dst\": null}"
            )]
        );
    }

    #[test]
    fn a_broken_xml_call_is_text() {
        // No function tag.
        let raw = "<tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(xml(raw), vec![text(raw)]);
        // A tool the request did not offer.
        let raw = "<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>";
        assert_eq!(xml(raw), vec![text(raw)]);
        // Text between the parameters.
        let raw = "<tool_call>\n<function=now>\nplease\n</function>\n</tool_call>";
        assert_eq!(xml(raw), vec![text(raw)]);
        // A parameter that is never closed.
        let raw = "<tool_call>\n<function=now>\n<parameter=zone>\nUTC\n</function>\n</tool_call>";
        assert_eq!(xml(raw), vec![text(raw)]);
    }

    #[test]
    fn an_xml_call_cut_off_by_the_end_of_the_output() {
        // Every parameter is whole: a call.
        assert_eq!(
            xml("<tool_call>\n<function=web_search>\n<parameter=query>\nrust\n</parameter>\n"),
            vec![call(0, "web_search", "{\"query\": \"rust\"}")]
        );
        // Cut inside a parameter: text, as written.
        let raw = "Searching.\n<tool_call>\n<function=web_search>\n<parameter=query>\nru";
        assert_eq!(xml(raw), vec![text(raw)]);
    }

    #[test]
    fn both_templates_are_recognised() {
        assert_eq!(
            ToolFormat::for_template(QWEN35_TEMPLATE),
            Some(ToolFormat::QwenXml)
        );
        assert_eq!(ToolFormat::QwenXml.name(), "qwen-xml");
        assert_ne!(QWEN25_TEMPLATE, QWEN35_TEMPLATE);
    }

    #[test]
    fn the_four_wordings_of_qwen3s_template_are_recognised_with_their_rules() {
        assert_eq!(
            ToolFormat::for_template(QWEN3_TEMPLATE),
            Some(ToolFormat::Qwen3(Qwen3Text::Upstream))
        );
        let rules: Vec<Qwen3Text> = QWEN3_VARIANTS
            .iter()
            .map(|(text, rules)| {
                assert_ne!(*text, QWEN3_TEMPLATE);
                assert_eq!(
                    ToolFormat::for_template(text),
                    Some(ToolFormat::Qwen3(*rules))
                );
                // A trailing newline is forgiven; an edited template is not.
                assert_eq!(
                    ToolFormat::for_template(&format!("{text}\n")),
                    Some(ToolFormat::Qwen3(*rules))
                );
                assert_eq!(
                    ToolFormat::for_template(&text.replace("# Tools", "# Tool")),
                    None
                );
                *rules
            })
            .collect();
        assert_eq!(
            rules,
            [
                Qwen3Text::QwenGguf,
                Qwen3Text::Upstream,
                Qwen3Text::UnslothLarge
            ]
        );
        let format = ToolFormat::Qwen3(Qwen3Text::Upstream);
        assert_eq!(format.name(), "qwen3");
        assert!(format.thinks() && ToolFormat::QwenXml.thinks() && !ToolFormat::Hermes.thinks());
        // The model opens its own thinking block: no prompt of this format ends inside one.
        let ask = vec![Msg::User("Hi".into())];
        for thinking in [true, false] {
            let prompt = render_with(format, &[], &ask, RenderOptions { thinking }).unwrap();
            assert!(!format.opens_reasoning(&prompt), "{prompt:?}");
        }
    }

    #[test]
    fn a_qwen3_call_is_read_as_json_and_a_role_without_a_branch_writes_nothing() {
        let format = ToolFormat::Qwen3(Qwen3Text::Upstream);
        let mut parser = CallParser::new(format, vec!["get_weather".into()]);
        let mut events = Vec::new();
        parser.push(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
            &mut events,
        );
        parser.finish(&mut events);
        assert_eq!(
            events,
            vec![call(0, "get_weather", "{\"city\": \"Paris\"}")]
        );

        // The reader keeps such a message's place and the renderer writes nothing for it;
        // the other formats refuse it by name.
        let body = Value::parse(
            r#"[{"role":"user","content":"a"},{"role":"critic","content":"x"},{"role":"user","content":"b"}]"#,
        )
        .unwrap();
        let messages = messages_from_json_for(format, &body).unwrap();
        assert_eq!(messages[1], Msg::Other);
        let prompt = render(format, &[], &messages).unwrap();
        assert!(
            !prompt.contains("critic") && !prompt.contains('x'),
            "{prompt:?}"
        );
        assert!(messages_from_json_for(ToolFormat::Hermes, &body).is_err());
        assert!(messages_from_json_for(ToolFormat::QwenXml, &body).is_err());
    }

    #[test]
    fn the_published_variants_of_qwen35s_template_are_the_same_format() {
        for (i, variant) in QWEN35_VARIANTS.iter().enumerate() {
            assert_ne!(*variant, QWEN35_TEMPLATE, "variant {i} is a copy");
            assert_eq!(
                ToolFormat::for_template(variant),
                Some(ToolFormat::QwenXml),
                "variant {i}"
            );
            // A GGUF writer may add a newline at the end; nothing else is forgiven.
            assert_eq!(
                ToolFormat::for_template(&format!("{variant}\n")),
                Some(ToolFormat::QwenXml)
            );
            assert_eq!(
                ToolFormat::for_template(&variant.replace("tool_call", "call")),
                None
            );
        }
        // The variants differ from one another too: three texts, not one text three times.
        assert_ne!(QWEN35_VARIANTS[0], QWEN35_VARIANTS[1]);
        assert_ne!(QWEN35_VARIANTS[0], QWEN35_VARIANTS[2]);
        assert_ne!(QWEN35_VARIANTS[1], QWEN35_VARIANTS[2]);
    }

    #[test]
    fn a_prompt_that_ends_inside_a_thinking_block_says_so() {
        let ask = vec![Msg::User("What time is it?".into())];
        let xml = ToolFormat::QwenXml;
        let on = render_with(xml, &[], &ask, RenderOptions { thinking: true }).unwrap();
        assert!(xml.opens_reasoning(&on), "{on:?}");
        // Thinking off: the template closes an empty block, so the reply starts as content.
        let off = render_with(xml, &[], &ask, RenderOptions { thinking: false }).unwrap();
        assert!(!xml.opens_reasoning(&off), "{off:?}");
        // A continued assistant text is left open where the client stopped, outside a block.
        let continued = vec![
            Msg::User("What time is it?".into()),
            Msg::Assistant {
                content: "It is".into(),
                reasoning: None,
                tool_calls: vec![],
            },
        ];
        let open = render(xml, &[], &continued).unwrap();
        assert!(!xml.opens_reasoning(&open), "{open:?}");
        // Qwen 2.5's template has no thinking block.
        let hermes = render(ToolFormat::Hermes, &[], &ask).unwrap();
        assert!(!ToolFormat::Hermes.opens_reasoning(&hermes));
    }

    #[test]
    fn thinking_left_inside_an_assistant_text_is_moved_into_its_block() {
        // A client that sends the model's whole output back, thinking block included, and
        // no `reasoning_content`: the template splits it, and so does the renderer.
        let whole = format!("{THINK_OPEN}\nCheck the time.\n{THINK_CLOSE}\n\nIt is noon.");
        let split = vec![
            Msg::User("What time is it?".into()),
            Msg::Assistant {
                content: "It is noon.".into(),
                reasoning: Some("Check the time.".into()),
                tool_calls: vec![],
            },
        ];
        let inline = vec![
            Msg::User("What time is it?".into()),
            Msg::Assistant {
                content: whole,
                reasoning: None,
                tool_calls: vec![],
            },
        ];
        let a = render(ToolFormat::QwenXml, &[], &split).unwrap();
        let b = render(ToolFormat::QwenXml, &[], &inline).unwrap();
        assert_eq!(a, b);
        assert!(a.ends_with("It is noon."), "a continuation is left open");
    }

    // ---- forced calls (S6.6b) ----

    fn forced_tools() -> Vec<Value> {
        let raw = Value::parse(
            r#"[
            {"type":"function","function":{"name":"get_weather","description":"w",
              "parameters":{"type":"object","properties":{"city":{"type":"string"},
                "unit":{"enum":["c","f"]}},"required":["city"],"additionalProperties":false}}},
            {"type":"function","function":{"name":"now"}}
        ]"#,
        )
        .unwrap();
        tools_from_json(Some(&raw)).unwrap()
    }

    /// Feed `text` to a fresh matcher of `c`: `Ok((can_end, closed))`, or the byte offset
    /// the constraint refused.
    fn feed(c: &Constraint, text: &str) -> Result<(bool, bool), usize> {
        let mut m = crate::constrain::Matcher::new(c).expect("the constraint starts");
        m.advance_bytes(text.as_bytes())?;
        Ok((m.can_end(), m.is_closed()))
    }

    const WEATHER: &str =
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    const NOW: &str = "<tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>";

    #[test]
    fn a_named_choice_is_exactly_one_call_to_that_tool() {
        let tools = forced_tools();
        for format in [ToolFormat::Hermes, ToolFormat::Qwen3(Qwen3Text::Upstream)] {
            let c =
                forced_calls(format, &tools, &Forced::Named("get_weather".into()), true).unwrap();
            // The call, whole: the reply may end there and nothing may follow it.
            assert_eq!(feed(&c, WEATHER), Ok((true, true)));
            // Not prose, not the other tool, not a second call.
            assert_eq!(feed(&c, "Sure"), Err(0));
            assert!(feed(&c, NOW).is_err());
            assert!(feed(&c, &format!("{WEATHER}\n{WEATHER}")).is_err());
            // The arguments are the tool's own schema: a missing required key, a key the
            // schema does not have and a value outside the enum are all refused.
            for bad in [
                "{\"unit\": \"c\"}",
                "{\"city\": \"Paris\", \"day\": 1}",
                "{\"city\": \"Paris\", \"unit\": \"k\"}",
            ] {
                let text = format!(
                    "<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {bad}}}\n</tool_call>"
                );
                assert!(feed(&c, &text).is_err(), "{bad}");
            }
            // A prefix is not an end.
            assert_eq!(feed(&c, &WEATHER[..WEATHER.len() - 1]), Ok((false, false)));
        }
    }

    #[test]
    fn a_required_choice_is_one_call_or_several_joined_by_a_newline() {
        let tools = forced_tools();
        let any = forced_calls(ToolFormat::Hermes, &tools, &Forced::Any, true).unwrap();
        // Either tool; after a call the reply may end or go on with another call.
        assert_eq!(feed(&any, WEATHER), Ok((true, false)));
        assert_eq!(feed(&any, NOW), Ok((true, false)));
        assert_eq!(
            feed(&any, &format!("{NOW}\n{WEATHER}\n{NOW}")),
            Ok((true, false))
        );
        // Between calls exactly one newline, and never text.
        assert!(feed(&any, &format!("{NOW}{WEATHER}")).is_err());
        assert!(feed(&any, &format!("{NOW}\n\n{WEATHER}")).is_err());
        assert!(feed(&any, &format!("{NOW}\nDone.")).is_err());
        assert_eq!(feed(&any, "I will"), Err(0));
        // A tool without parameters takes no arguments.
        assert!(feed(
            &any,
            "<tool_call>\n{\"name\": \"now\", \"arguments\": {\"tz\": 1}}\n</tool_call>"
        )
        .is_err());

        // `parallel_tool_calls: false`: exactly one.
        let one = forced_calls(ToolFormat::Hermes, &tools, &Forced::Any, false).unwrap();
        assert_eq!(feed(&one, NOW), Ok((true, true)));
        assert!(feed(&one, &format!("{NOW}\n{WEATHER}")).is_err());
    }

    #[test]
    fn what_a_forced_reply_says_is_what_the_parser_reads() {
        let tools = forced_tools();
        let any = forced_calls(ToolFormat::Hermes, &tools, &Forced::Any, true).unwrap();
        let reply = format!("{WEATHER}\n{NOW}");
        assert_eq!(feed(&any, &reply), Ok((true, false)));
        let events = parse_with(&reply, || {
            CallParser::with_tools(ToolFormat::Hermes, &tools)
        });
        assert_eq!(
            events,
            vec![
                call(0, "get_weather", "{\"city\": \"Paris\"}"),
                call(1, "now", "{}"),
            ]
        );
    }

    #[test]
    fn a_forced_choice_that_cannot_be_honoured_is_refused_by_name() {
        let tools = forced_tools();
        let err = |r: Result<Constraint, String>| r.expect_err("refused");
        assert!(err(forced_calls(
            ToolFormat::Hermes,
            &tools,
            &Forced::Named("nope".into()),
            true
        ))
        .contains("\"nope\", which is not one of the request's tools"));
        assert!(
            err(forced_calls(ToolFormat::Hermes, &[], &Forced::Any, true)).contains("needs tools")
        );
        let parse = |text: &str| tools_from_json(Some(&Value::parse(text).unwrap())).unwrap();
        // Arguments are an object: a schema of another type cannot be a tool's parameters.
        let scalar = parse(
            r#"[{"type":"function","function":{"name":"f","parameters":{"type":"string"}}}]"#,
        );
        assert!(err(forced_calls(
            ToolFormat::Hermes,
            &scalar,
            &Forced::Any,
            true
        ))
        .contains("tools[0].function.parameters must be a schema of type \"object\""));
        // A keyword the constraint compiler refuses names the tool it is in.
        let pattern = parse(
            r#"[{"type":"function","function":{"name":"ok","parameters":{"type":"object"}}},
                {"type":"function","function":{"name":"f","parameters":{"type":"object",
                 "properties":{"a":{"type":"string","pattern":"^a+$"}}}}}]"#,
        );
        let e = err(forced_calls(
            ToolFormat::Hermes,
            &pattern,
            &Forced::Any,
            true,
        ));
        assert!(e.starts_with("tools[1].function.parameters: "), "{e}");
        assert!(e.contains("pattern"), "{e}");
        // Naming the other tool does not trip over it.
        assert!(forced_calls(
            ToolFormat::Hermes,
            &pattern,
            &Forced::Named("ok".into()),
            true
        )
        .is_ok());
        // A schema without `type` is read as an object schema.
        let untyped = parse(
            r#"[{"type":"function","function":{"name":"f","parameters":{"properties":{"a":{"type":"integer"}},"required":["a"]}}}]"#,
        );
        let c = forced_calls(ToolFormat::Hermes, &untyped, &Forced::Any, false).unwrap();
        assert_eq!(
            feed(
                &c,
                "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 3}}\n</tool_call>"
            ),
            Ok((true, true))
        );
        assert!(feed(
            &c,
            "<tool_call>\n{\"name\": \"f\", \"arguments\": 3}\n</tool_call>"
        )
        .is_err());
    }

    // ---- forced calls in the XML convention, and after thinking (S6.6b) ----

    fn forced_xml_tools() -> Vec<Value> {
        let raw = Value::parse(
            r#"[
            {"type":"function","function":{"name":"web_search","description":"s",
              "parameters":{"type":"object","properties":{
                "query":{"type":"string"},
                "limit":{"type":"integer"},
                "safe":{"type":"boolean"},
                "site":{"enum":["docs","blog"]},
                "tags":{"type":"array","items":{"type":"string"}}},
              "required":["query","safe"]}}},
            {"type":"function","function":{"name":"now"}}
        ]"#,
        )
        .unwrap();
        tools_from_json(Some(&raw)).unwrap()
    }

    #[test]
    fn a_forced_call_in_the_xml_convention_is_the_templates_own_text() {
        let tools = forced_xml_tools();
        let c = forced_calls(
            ToolFormat::QwenXml,
            &tools,
            &Forced::Named("web_search".into()),
            true,
        )
        .unwrap();
        // Every property, in the schema's order, each value in its bare spelling.
        let full = "<tool_call>\n<function=web_search>\n<parameter=query>\nrust 2024\nedition\n</parameter>\n<parameter=limit>\n5\n</parameter>\n<parameter=safe>\nTrue\n</parameter>\n<parameter=site>\ndocs\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(feed(&c, full), Ok((true, true)));
        // Only the required ones.
        let least = "<tool_call>\n<function=web_search>\n<parameter=query>\nrust\n</parameter>\n<parameter=safe>\nFalse\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(feed(&c, least), Ok((true, true)));
        // What the constraint says is what the parser reads, and it reads the arguments
        // the schema describes.
        let events = parse_with(full, || CallParser::with_tools(ToolFormat::QwenXml, &tools));
        assert_eq!(
            events,
            vec![call(
                0,
                "web_search",
                "{\"query\": \"rust 2024\\nedition\", \"limit\": 5, \"safe\": true, \"site\": \"docs\", \"tags\": [\"a\", \"b\"]}"
            )]
        );
        // Refused: a required property left out, a property out of order, a value outside
        // its type or its enum, a property the schema does not have, JSON's own `true`.
        let with = |params: &str| {
            format!("<tool_call>\n<function=web_search>\n{params}</function>\n</tool_call>")
        };
        for bad in [
            "<parameter=query>\nrust\n</parameter>\n",
            "<parameter=safe>\nTrue\n</parameter>\n<parameter=query>\nrust\n</parameter>\n",
            "<parameter=query>\nrust\n</parameter>\n<parameter=limit>\nfive\n</parameter>\n<parameter=safe>\nTrue\n</parameter>\n",
            "<parameter=query>\nrust\n</parameter>\n<parameter=safe>\nTrue\n</parameter>\n<parameter=site>\nwiki\n</parameter>\n",
            "<parameter=query>\nrust\n</parameter>\n<parameter=safe>\nTrue\n</parameter>\n<parameter=lang>\nen\n</parameter>\n",
            "<parameter=query>\nrust\n</parameter>\n<parameter=safe>\ntrue\n</parameter>\n",
        ] {
            assert!(feed(&c, &with(bad)).is_err(), "{bad:?}");
        }
        // A tool without parameters, any tool, and several calls joined by a newline.
        let any = forced_calls(ToolFormat::QwenXml, &tools, &Forced::Any, true).unwrap();
        let now = "<tool_call>\n<function=now>\n</function>\n</tool_call>";
        assert_eq!(feed(&any, now), Ok((true, false)));
        assert_eq!(feed(&any, &format!("{least}\n{now}")), Ok((true, false)));
        assert!(feed(&any, &format!("{least}{now}")).is_err());
        assert_eq!(feed(&any, "Let me"), Err(0));
    }

    #[test]
    fn a_constraint_after_thinking_leaves_the_thinking_free() {
        let tools = forced_tools();
        let calls = || forced_calls(ToolFormat::Hermes, &tools, &Forced::Any, false).unwrap();
        // The prompt opened the block: reasoning, the closing tag, two newlines, the call.
        let open = after_reasoning(Reasoning::Open, calls());
        assert_eq!(
            feed(
                &open,
                &format!("They want <tool_call> weather.\n</think>\n\n{NOW}")
            ),
            Ok((true, true))
        );
        assert_eq!(feed(&open, &format!("</think>\n\n{NOW}")), Ok((true, true)));
        assert!(feed(&open, &format!("thinking</think>{NOW}")).is_err());
        assert!(feed(&open, "thinking</think>\n\nSure.").is_err());
        // The reply cannot end in the reasoning, however long it is.
        assert_eq!(feed(&open, "still thinking"), Ok((false, false)));
        // The model may open a block itself, or go straight to the call.
        let optional = after_reasoning(Reasoning::Optional, calls());
        assert_eq!(feed(&optional, NOW), Ok((true, true)));
        assert_eq!(
            feed(&optional, &format!("<think>\nhm\n</think>\n\n{NOW}")),
            Ok((true, true))
        );
        assert!(feed(&optional, &format!("hm</think>\n\n{NOW}")).is_err());
        assert!(feed(
            &optional,
            &format!("<think>a</think>\n\n<think>b</think>\n\n{NOW}")
        )
        .is_err());
        // No thinking: the constraint as it is.
        assert_eq!(
            feed(&after_reasoning(Reasoning::None, calls()), NOW),
            Ok((true, true))
        );
    }

    #[test]
    fn an_xml_value_follows_the_propertys_schema() {
        let parse = |text: &str| tools_from_json(Some(&Value::parse(text).unwrap())).unwrap();
        let tools = parse(
            r##"[{"type":"function","function":{"name":"f","parameters":{"type":"object",
              "$defs":{"point":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}},
              "properties":{
                "p":{"$ref":"#/$defs/point"},
                "o":{"type":"object","properties":{"at":{"$ref":"#/$defs/point"}},"required":["at"],"additionalProperties":false},
                "n":{"type":"null"},
                "c":{"const":3},
                "u":{"type":["string","null"]}},
              "required":["p","o","n","c","u"]}}}]"##,
        );
        let c = forced_calls(ToolFormat::QwenXml, &tools, &Forced::Any, false).unwrap();
        let with = |o: &str, n: &str, c: &str| {
            format!("<tool_call>\n<function=f>\n<parameter=p>\nanything at all\n</parameter>\n<parameter=o>\n{o}\n</parameter>\n<parameter=n>\n{n}\n</parameter>\n<parameter=c>\n{c}\n</parameter>\n<parameter=u>\nfree\n</parameter>\n</function>\n</tool_call>")
        };
        // A bare `$ref` and a union have no single bare spelling: free text. An object is
        // JSON of its schema, with the tool's definitions beside it.
        assert_eq!(
            feed(&c, &with("{\"at\": {\"x\": 1}}", "None", "3")),
            Ok((true, true))
        );
        assert!(feed(&c, &with("{\"at\": {\"x\": \"1\"}}", "None", "3")).is_err());
        assert!(feed(&c, &with("{\"at\": {\"x\": 1}}", "null", "3")).is_err());
        assert!(feed(&c, &with("{\"at\": {\"x\": 1}}", "None", "4")).is_err());
    }
}
