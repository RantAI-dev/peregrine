# S6 Design: tool calling

- Status: accepted
- Target: a new roadmap stage, S6 (items S6.1 to S6.7)
- Depends on: the chat endpoint (`/v1/chat/completions`), the chat templates (`template.rs`), the
  order-keeping JSON value (`json.rs`)

## 0. Summary

A client sends a list of tools with a chat request; the model may answer with one or more calls to
them instead of text; the client runs them and sends the results back. Peregrine has none of this
today: `tools` is not read, an assistant message with `tool_calls` and a `tool` message are not
understood, and constrained sampling is refused.

This stage adds OpenAI's tool-calling shapes to `/v1/chat/completions`, for the models that were
trained to call tools, with the same rule as every other model feature: the prompt Peregrine builds
is byte for byte the prompt llama.cpp builds from the model's own template, and what Peregrine
reads back out of the model's text is what llama.cpp reads.

## 1. What a tool call is, for the engine

Nothing in the forward pass changes. Tool calling is three pieces of text handling:

1. **Rendering.** The tools (name, description, JSON schema of the arguments) are written into the
   prompt where the model's template puts them. Earlier calls and their results in the conversation
   are written back in the model's own format.
2. **Parsing.** The model's output is split into ordinary text and calls. A call is a name and a
   JSON object of arguments.
3. **Constraining** (optional, S6.6). The sampler is restricted to tokens that keep the output a
   valid call with arguments that fit the tool's schema. This is the only piece that touches
   sampling.

## 2. The formats

Peregrine has no template engine (decision 6 of `s4-embeddings-decisions.md`): each model family
has a renderer written in Rust and checked against llama.cpp. Tool calling follows the same rule.
A **tool format** is a renderer and a parser for one family of templates:

| Format | Models | What the template does |
|---|---|---|
| `hermes` | Qwen 2.5 (and Qwen 3, with its thinking block) | Tool signatures as JSON lines inside `<tools>` in the system message; a call is `{"name": ..., "arguments": {...}}` inside `<tool_call>`; a result is sent back inside `<tool_response>` in a user turn |
| `qwen-xml` | Qwen 3.5 and Bonsai 2 | The same `<tools>` list with its own instructions; a call is `<function=name>` with one `<parameter=key>` block per argument inside `<tool_call>`, values written bare (`True`, `None`, JSON for lists and objects); assistant turns after the last question keep their thinking block; the prompt ends inside one, or after an empty one when thinking is off |
| `llama3` | Llama 3.1 / 3.2 | Tool definitions in the first user message; a call is one JSON object `{"name": ..., "parameters": {...}}`; a result is an `ipython` turn |
| `harmony` | gpt-oss | Tools as a TypeScript-like namespace in the developer message; a call is a message to `functions.<name>` on the commentary channel |
| `generic` | templates with no tool section (Gemma 3, Phi-4, Llama 2, Mistral v0.2) | A system instruction that describes the tools and asks for a JSON answer, as llama.cpp's fallback does |

The `hermes` and `qwen-xml` rows are taken from llama.cpp's renderings (Qwen 2.5's template and
the template Bonsai 2 files carry); the other rows describe the templates as read and are confirmed
against llama.cpp when their step is built. The first version of this table put Qwen 3.5 and
Bonsai 2 under `hermes`: their template writes calls differently, so they are a format of their own.

A file whose template is not one Peregrine recognises with tools is refused when a request carries
`tools`, by name, the way an unknown chat template is refused today. `generic` comes last because
it only works reliably with constrained sampling.

**Checked against llama.cpp.** `llama-server --jinja` renders any request through
`POST /apply-template`. For each format a fixture set of conversations (no tools, one tool, several
tools, nested schemas, calls and results in the history, several calls in one turn, a system
message or none, thinking on and off where the family has it) is rendered by llama.cpp v0.9.0 on the
model's own file, and Peregrine's renderer must give the same bytes. The parser is checked on
recorded model outputs and on written cases: text only, one call, several calls, text before a
call, a call cut off by the token limit, arguments that are not valid JSON.

## 3. API

`POST /v1/chat/completions`, OpenAI's shapes:

- Request: `tools` (a list of `{"type": "function", "function": {"name", "description",
  "parameters"}}`), `tool_choice` (`"auto"`, `"none"`, `"required"`, or a named function),
  `parallel_tool_calls`. Messages may be an assistant message with `tool_calls` and a `tool`
  message with `tool_call_id`.
- Response: `message.tool_calls` (each with an `id`, `type: "function"`, and `function.name` and
  `function.arguments`, the arguments as a JSON string), `finish_reason: "tool_calls"`. Text the
  model wrote before the call stays in `content`.
- Streaming: `delta.tool_calls` with an `index` per call; the first chunk of a call carries its
  `id` and `name`, the following ones carry pieces of `arguments`. Text that could be the start of a
  call is held back until it is known to be one or not, so a client never sees half a tag.
- `tool_choice: "none"` renders the tools as the template does and returns text only. `"required"`
  and a named function need the constraint of S6.6; until then they are refused by name rather than
  accepted and not honoured.
- The older `functions` and `function_call` fields are refused by name.
- A model's text that looks like a call but does not parse (broken JSON, an unknown tool name) is
  returned as text, never as a server error. With the constraint on, this cannot happen.

## 4. Constrained sampling (S6.6)

Small models often write almost-valid JSON. The fix is to allow, at each step, only the tokens that
keep the output inside the grammar: the call's wrapper, then a JSON value that fits the tool's
schema. llama.cpp does this with a grammar built from the schema, switched on when the model starts
a call.

Peregrine's form:

- A matcher over JSON driven by the schema (objects with `properties` and `required`, strings,
  numbers, integers, booleans, `null`, arrays, `enum`, nested), not a general grammar engine. What
  the matcher does not support in a schema is refused when the request arrives, by name.
- It gives the sampler a set of allowed tokens for the next step. The sampler masks the others and
  then samples as it does now, with the same position-keyed randomness.
- **Invariance.** The allowed set depends only on the text of that one request so far. A request's
  output is therefore still the same alone, in any batch, on any thread count.
- **Speculation.** A drafted token that the mask would not allow is rejected like any other wrong
  draft, so speculation stays lossless; if that costs more than it saves, speculation is turned off
  for a constrained request.
- With it, `tool_choice: "required"` and a named function are honoured, and `generic` becomes
  usable.

## 5. What does not change

- The forward pass, the kernels, the goldens.
- A request without `tools` builds exactly the prompt it builds today.
- Lossless by default: nothing here approximates the model.

## 6. Steps

| ID | Item | Needs |
|---|---|---|
| S6.1 | This design and the roadmap rows | |
| S6.2 | The tool module in `peregrine-core`: tool and call types, the `hermes` renderer over a message list (system, user, assistant with calls, tool results) and its parser (whole text and streaming), fixtures from llama.cpp on Qwen 2.5 0.5B; then `qwen-xml` for Qwen 3.5 / Bonsai 2 with its thinking blocks (S6.2b), and Qwen 3 | |
| S6.3 | The endpoint: request and response shapes of §3, streaming, the refusals, usage; `peregrine run` stays text only | S6.2 |
| S6.4 | `llama3`: renderer, parser, fixtures on Llama 3.2 1B | S6.3 |
| S6.5 | `harmony` for gpt-oss | S6.3 |
| S6.6 | Constrained sampling for calls: the schema matcher, the token mask, `required` and named choices | S6.3 |
| S6.7 | `generic` for templates without a tool section | S6.6 |

## 7. Accuracy gate

1. **Prompts**: identical bytes and token ids to llama.cpp's `/apply-template` on the fixture set,
   per format and per model file.
2. **Parsing**: on the same model outputs, the same content and the same calls (names, and
   arguments equal as JSON) as llama.cpp's server returns.
3. **End to end**: on a set of requests where the right answer is one known call (a weather tool, a
   calculator, a search), each listed model calls the right tool with arguments that parse, at a
   rate no lower than llama.cpp's on the same file and sampling settings. With S6.6, every call
   parses.
4. **Invariance**: a request with tools gives the same tokens alone and in a batch; with the
   constraint on, the same.

## 8. Decisions

1. **OpenAI's shapes on the existing chat endpoint**, as llama-server serves them. No second API.
2. **A format per template family, checked against llama.cpp; no template engine.** A template
   Peregrine does not know with tools is refused when tools are sent.
3. **Unhonoured options are refused**, not ignored: `required` and named choices wait for S6.6.
4. **A broken call is text**, never an error.
5. **The constraint is a schema matcher**, not a grammar language. A general `grammar` field stays
   unsupported.

## 9. S6.3 as built (the endpoint)

**Routing.** A chat request takes the tool path when it carries a non-empty `tools`, a `tool`
message, or an assistant message with `tool_calls`. The file's `tokenizer.chat_template` is matched
once at start (`ToolFormat::for_template`); a file with no tool format answers such a request with a
400 that names the model. The request body is read a second time with the order-keeping JSON,
because a tool's schema is written into the prompt in the client's own key order. Every other
request takes the path it took before and builds the same prompt.

**The reply.** Generated content goes through `CallParser`; its text events become `content` and
each call becomes a `tool_calls` entry with an id (`call_` and 24 hex digits, numbered from the
request's sequence number, so they are unique within a response without a random source).
`content` is `null` when the reply is only calls; `finish_reason` is `"tool_calls"` when there is
at least one. A stream sends each call as one chunk (`delta.tool_calls` with `index`, `id`, `type`
and the whole `function`). Reasoning text does not pass through the parser.

**Options.** `tool_choice`: `"auto"` or absent parses calls; `"none"` renders the tools and returns
text; `"required"` and a named function are refused (they need S6.6). `parallel_tool_calls: false`
cancels the generation when the first call is complete. `functions` and `function_call` are refused
by name, also on requests without tools.

**The call tags reach the parser.** In Qwen 2.5's vocabulary the opening and closing tags of a call
are single tokens that are not control tokens; the text the endpoint assembles contains them, so
nothing had to change in detokenisation.

**Against llama.cpp** (`scripts/tool-e2e.py`; `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, temperature 0,
three tools: weather, calculator, search; llama.cpp v0.9.0 `llama-server --jinja` on the same file):

| | Peregrine | llama.cpp |
|---|---|---|
| 12 requests where one known call is right: right tool called | 9 | 9 |
| 4 requests where no call is right: answered in text | 4 | 4 |
| replies identical to the other engine's (same calls, arguments equal as JSON, same text) | 15 of 16 | |
| two-step exchange: the tool's result sent back, answered in text | same sentence in both | |

The three wrong calls are the model's (a 0.5B model calls the weather tool for a power of two and
for a football result, and answers one search request in text) and are the same in both engines.
The one reply that differs is that text answer: the two engines' greedy tokens part at a near-tie
in the middle of the sentence, as their kernels' rounding differs; neither calls a tool there.

**Tests.** The endpoint's unit tests drive the reply builder with scripted pieces through the
parser (text only, one call, text then two calls, a broken call as text, an unknown tool as text, a
call split across pieces, `parallel_tool_calls: false`, stream chunk shapes, `finish_reason`).
`crates/peregrine/tests/server_tools.rs` serves two synthetic files with the real binary, one whose
template has a tool format and one whose template has none: a request with tools is served and is
deterministic, a tool conversation is rendered, the refusals answer 400 by name, and an ordinary
request is untouched.

**Not built.** Other formats (Llama 3, gpt-oss, the generic fallback); `required` and
named choices (S6.6); arguments streamed in pieces (a call arrives whole); tool calling in
`peregrine run`.

## 10. S6.3b as built (the `qwen-xml` format on the endpoint)

**Which files.** Bonsai 2 files and Qwen's own 4B and 9B files carry one text of Qwen 3.5's
template; Qwen's 0.8B and 2B files and unsloth's conversions carry three others. They differ in
two places: whether the thinking block is written "unless `enable_thinking` is false" or "only
when it is true", and whether a call's arguments are walked with `|items` or with a key loop.
Neither changes a prompt: llama.cpp always defines `enable_thinking` (true unless the request
says otherwise) and so does the renderer, and arguments that are not an object are refused before
rendering. llama.cpp v0.9.0 renders all 51 fixture cases to the same bytes with each of the four
texts (`scripts/tool-fixtures.py --check`), so `ToolFormat::for_template` accepts the four and
nothing else. The template of unsloth's `*-MTP` conversions (a `developer` role, two leading
system messages merged, no refusals) is a different one and is refused for tools.

**The request.** The tool path reads the messages the way this format's template does
(`messages_from_json_for`: the parts of a content list are joined without a separator) and passes
the thinking switch to the renderer: `chat_template_kwargs.enable_thinking` (default on), and
`reasoning_effort: "none"` switches it off, as on the ordinary path.

**The reply starts as reasoning.** With thinking on, this template opens the thinking block at
the end of the prompt, so the model writes its reasoning first and then only the closing tag.
`ToolFormat::opens_reasoning` says whether a rendered prompt ends that way; when it does, the
reply is read from inside the reasoning span: the text up to the closing tag is
`reasoning_content` and what follows goes through the call parser. Text inside the span never
reaches the parser, also under `reasoning_format: none` (where it is returned as `content`): a
call the model only thought about is not a call. A reasoning budget (`thinking_budget_tokens`)
continues the same turn with the reasoning so far and the closing tag appended to the prompt,
without a second opening tag. With thinking off the template writes an empty block and the reply
is content from its first token.

**Argument types.** The format writes values bare, so the parser is built with the request's
tools (`CallParser::with_tools`): a parameter the schema declares a string stays text (`5` is
`"5"`), anything else is read as JSON when it parses.

**Against llama.cpp** (`scripts/tool-e2e.py`; `Qwen_Qwen3.5-2B-Q4_K_M.gguf` from
`bartowski/Qwen_Qwen3.5-2B-GGUF`, temperature 0, the same three tools and 16 requests as §9;
llama.cpp v0.9.0 `llama-server --jinja` on the same file):

| | thinking off | thinking on |
|---|---|---|
| 12 requests where one known call is right: right tool called (Peregrine / llama.cpp) | 12 / 12 | 11 / 11 |
| 4 requests where no call is right: answered in text (Peregrine / llama.cpp) | 4 / 4 | 4 / 4 |
| calls identical to the other engine's (same tool, arguments equal as JSON) | 12 of 12 | 11 of 11 |
| whole replies identical (calls, text and reasoning) | 13 of 16 | 4 of 16 |
| two-step exchange: the tool's result sent back, answered in text | same sentence | same facts, wording differs |

The prompts are the same length in both engines (285 tokens for a request with one tool). The
one miss with thinking on is the model's and is the same in both: asked who won a football
tournament, it reasons that it knows and answers in text. Where whole replies differ, the calls
do not: the two engines' greedy tokens part at a near-tie somewhere in a free-text answer or in
the reasoning (about a hundred tokens of prose before each call), as their kernels' rounding
differs. `content` is `null` beside a call in Peregrine and `""` in llama.cpp, and Peregrine
always returns `reasoning_content` (empty when there is none); both are as §9 left them.

**Whitespace.** llama.cpp returns the reasoning stripped at both ends and starts the answer at its
first character after the closing tag; a reply cut off inside its reasoning keeps the whitespace
it ended with. A reply whose thinking block the template opened is returned the same way, in a
stream too: whitespace at the end of the reasoning is held back until more reasoning follows, and
dropped when the closing tag does.

**Requests without tools.** Qwen 3.5's template is not the ChatML the ordinary path writes: it
opens the thinking block, trims each message, and writes earlier turns' thinking back. On the 2B
file the ordinary path's prompt was two tokens shorter than llama.cpp's with thinking on, and the
model answered at once where llama.cpp's prompt made it reason first; with a padded system message
it was two tokens longer with thinking off. The renderer gives llama.cpp's bytes for conversations
without tools too (the fixtures include them), so a file with this template now has every chat
request rendered by it (`renders_every_chat`): the prompt lengths are llama.cpp's in the four
cases checked (thinking on and off, with and without a system message: 16, 25, 18 and 27 tokens)
and the model reasons as it does there. Such a request is validated as before (the last message
must be the user's) and nothing in its reply is read as a call. Qwen 2.5's template is the
ordinary ChatML with a default system message, so its requests without tools are unchanged.
`peregrine run` and the session take the same prompt from the `qwen35` chat template
(`Template::Qwen35`, S6.3c): the template is recognised by its exact text, its pieces (system
block, user turn, earlier assistant turns without their thinking, the opened block) are tested
equal to this renderer on 180 conversations, and the command shows the opening tag the prompt
wrote, so the output reads as a whole thinking block. On the two real files the prompt of
"Say hi in three words." is 16 tokens with thinking and 18 without, llama.cpp's counts. A
conversation continued in one session keeps the earlier turns' thinking in its cache, as it
does for Qwen 3 (the template drops it when a client resends the history); it is rendered
without it only when the oldest turns are dropped and the prompt is rebuilt.

**Bonsai 2** (`Ternary-Bonsai-2-2B-PQ2_0.gguf`, Peregrine only: llama.cpp does not load the
format; same requests):

| | thinking off | thinking on |
|---|---|---|
| 12 requests where one known call is right: right tool called | 12 | 11 |
| 4 requests where no call is right: answered in text | 4 | 4 |
| two-step exchange: the tool's result sent back | answered in text | answered in text |

The miss with thinking on is the same question as on Qwen 3.5 (the football result, answered
from memory, here correctly). Every call's arguments are valid for its tool's schema.

**Not built.** The template of unsloth's `*-MTP` conversions; tool calling in `peregrine run`.

**Tests.** `tools.rs`: the four texts are recognised and a changed one is not; a rendered prompt
says whether it ends inside a thinking block (on, off, a continued assistant text, the `hermes`
format). The endpoint's unit tests: the thinking switch changes the prompt's ending and
`reasoning_open`; a reply fed three characters at a time gives one call whose arguments have the
schema's types. `server_tools.rs` serves a synthetic file with the template: with thinking on
everything the model writes is `reasoning_content` and nothing is read for calls, with it off
(either spelling) the same tokens are content, `reasoning_format: none` returns the span as
content, a stream carries `reasoning_content` deltas, and the four texts of the template give the
same prompt length and the same reply for the same weights.

## 11. S6.2c as built (Qwen 3)

**The format.** `ToolFormat::Qwen3`: the tool list and the calls are Qwen 2.5's (`<tools>` in the
system message, a JSON object inside `<tool_call>`, results inside `<tool_response>` in a user
turn), without a default system message. What differs is everything around them: no message is
trimmed, a later system message is a turn of its own, a message with a role the template has no
branch for writes nothing (and still separates its neighbours), the assistant turns after the last
real user question carry their thinking block, and the prompt ends with the assistant's header,
or with an empty thinking block when thinking is off. The model opens its own thinking block, so
`opens_reasoning` is false and the reply is read as before (the opening tag starts the reasoning
span); the reasoning and the answer are returned stripped, as §10 describes. The call parser is
the `hermes` one.

**Four wordings, two rules apart.** The repository's template (bartowski and lmstudio-community
carry it), Qwen's own GGUF files, unsloth's files up to 1.7B and unsloth's larger files each ship
a different text. llama.cpp renders the 51 shared cases to the same bytes with all four, but two
corner rules differ, and a file is served by the rules of the text it carries (`Qwen3Text`):

| Wording | A user message that contains a `<tool_response>` block in the middle | Thinking that is only whitespace |
|---|---|---|
| repository, unsloth up to 1.7B | is a question | is thinking |
| Qwen's GGUF files | is a tool's result (the earlier turn keeps its thinking) | is thinking |
| unsloth 4B and larger | is a question | is none |

`tests/fixtures/tools-qwen3-edges.jsonl` holds eleven corner cases rendered by llama.cpp with
each wording (44 records), and the renderer reproduces all of them.

**Requests without tools** on a Qwen 3 file are rendered by the format too
(`renders_every_chat`), for the same reason as Qwen 3.5: the template's prompt is not quite the
built-in ChatML (untrimmed messages, later system messages as turns, earlier turns without their
thinking). `peregrine run` keeps the built-in `qwen3` template.

**Against llama.cpp** (`scripts/tool-e2e.py`; `Qwen_Qwen3-1.7B-Q4_K_M.gguf` from
`bartowski/Qwen_Qwen3-1.7B-GGUF`, temperature 0, the same 16 requests; llama.cpp v0.9.0
`llama-server --jinja`):

| | thinking off | thinking on |
|---|---|---|
| 12 requests where one known call is right: right tool called (Peregrine / llama.cpp) | 11 / 11 | 12 / 11 |
| 4 requests where no call is right: answered in text (Peregrine / llama.cpp) | 4 / 4 | 4 / 4 |
| requests where either engine calls: same tool | 11 of 11 | 11 of 12 |
| requests where either engine calls: same tool and same arguments | 11 of 11 | 10 of 12 |
| whole replies identical (calls, text and reasoning) | 13 of 16 | 0 of 16 |
| two-step exchange: the tool's result sent back, answered in text | same sentence | same facts, wording differs |

The prompts are the same length in both engines (305 tokens for a request with the three
tools). With thinking off the one miss is the model's and the same in both (asked whether it is
raining, it answers from nothing). With thinking on this model writes two to five hundred
characters of reasoning before every answer, and no two runs of that prose agree to the end: the
engines' greedy tokens part at a near-tie within the first few hundred characters of every
request (llama.cpp's own two runs of the same request parted at one: " me" at probability 0.52
against " to" at 0.49). The calls are what the prose leads to, and they agree on ten of twelve;
the two that differ are a search query worded differently (both valid) and the power of two, where
llama.cpp's reasoning ran to the token limit before it called and Peregrine's called.

**Tests.** `tools_render.rs`: the 51 shared cases under each wording's rules, and the 44 corner
records wording by wording. `tools.rs`: the four texts are recognised with their rules, a changed
one is not; a call is read as JSON; a role without a branch keeps its place and writes nothing.
The endpoint's unit tests: the thinking switch on this format's prompt, `reasoning_open` false, a
call routed. `server_tools.rs`: a synthetic file under each of the four wordings serves tools,
thinking off lengthens the prompt, a request without tools is rendered by the same template, and
the four give the same prompt lengths and replies.

**Not built.** Llama 3, gpt-oss, the generic fallback; `required` and named choices (S6.6).
