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

**Not built.** Other formats (Qwen 3, Qwen 3.5 / Bonsai 2, Llama 3, gpt-oss, the generic fallback);
`required` and named choices (S6.6); arguments streamed in pieces (a call arrives whole); tool
calling in `peregrine run`.
