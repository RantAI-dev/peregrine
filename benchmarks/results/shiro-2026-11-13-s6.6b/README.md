# S6.6b and S6.6c: forced tool calls, tools beside a response format, free text and marker tokens (2026-11-13)

What was measured for `docs/design/s6.6-constrained-decoding.md` §12 and §13. Host: the
shared 8-vCPU Xeon E5-2680 v4 VM (AVX2, 11 GB), under outside load during every run
(load average 7–14); llama.cpp is v0.9.0 (`cd959ba`) with `--jinja`. Every number here is
from the files in `raw/`; the scripts beside this file are the clients that produced them
(each is run against a server started on its own; `python3 -I <script> <port>`).

## 1. A forced call is valid, and is the model's own tokens

`forced.py`, Qwen 2.5 0.5B Instruct Q4_K_M, temperature 0 (`raw/forced-qwen25-0.5b.txt`):

| request | reply |
|---|---|
| `required`, a weather question | one call, `get_weather {"city": "Paris", "unit": "celsius"}`, 26 tokens |
| `required`, two things asked | two calls, weather then time, 50 tokens |
| `required`, `parallel_tool_calls: false` | one call, 27 tokens |
| `required`, "Say hello in one word." | two calls (the model has to call something) |
| a named function, on a question about another tool | one call to the named function |
| a named function that is not in `tools` | `400 tool_choice names "nope", which is not one of the request's tools` |

**Marker tokens.** Before S6.6c the same forced weather call took **31 tokens**, against 26
for the same call under `"auto"`: the mask excluded Qwen's `<tool_call>` and `</tool_call>`
tokens, so the tags were spelled out of ordinary tokens. With markers allowed where the
constraint spells them the forced call is **26 tokens**, the unforced call's count.

## 2. Thinking models, both call syntaxes

`forced2.py`, temperature 0 (`raw/forced-qwen3-qwen35.txt`):

| model | request | thinking | calls |
|---|---|---|---|
| Qwen 3 1.7B (hermes syntax) | `required`, thinking off | none | weather, then time |
| | a named function, thinking off | none | that function |
| | `required`, thinking on | 1 355 characters | weather, then time |
| Qwen 3.5 2B (XML syntax) | `required`, thinking off | none | weather, then time |
| | a named function, thinking off | none | that function |
| | `required`, thinking on | 338 characters | weather, then time |

`forced_stream.py` (`raw/stream-qwen3-1.7b.txt`): on Qwen 3 1.7B with thinking on, the
forced request and the unforced one give the same reply (534 characters of thinking, the
same call, 142 tokens), and each streams to the same thinking, call and finish reason as it
returns whole. When the model does what the constraint asks, the constraint changes nothing.

## 3. Tools beside a `response_format`

The reply is calls or JSON of the schema, never text. Which of the two is the model's
choice, and small models choose a call too readily (`raw/either-qwen25-0.5b.txt`, and the
"auto+response_format" rows of `raw/forced-qwen3-qwen35.txt`): on a question that needs no
tool and asks for a JSON object, Qwen 2.5 0.5B calls `get_weather {"city": "water"}` — and
it does exactly the same with the tools alone and **no constraint at all**. The constraint
does not create that call; it only removes the option of prose. With `tool_choice: "none"`
the same request answers `{"answer": "Yes", "confident": true}`.

## 4. Against llama.cpp

`forced_cmp.py`: ten questions, each with `"required"` and with a named function, 20 forced
requests a round, three interleaved rounds (`raw/forced-vs-llamacpp.txt`). A reply counts
when it carries at least one call; a call is valid when its arguments parse and fit the
tool's schema (checked by the script, not by either engine).

| | replies with a call | calls valid | decode tok/s, medians of the three rounds |
|---|---|---|---|
| Peregrine | 20 / 20 in every round | 22 / 22 | 35.8, 39.5, 38.9 |
| llama.cpp v0.9.0 | 16 / 20 in every round | 20 / 20 | 32.2, 32.8, 28.5 |

Peregrine ÷ llama.cpp on decode speed: **1.11, 1.20, 1.36** (round by round; the host was
loaded, so read it as "not slower"). llama.cpp's four replies without a call are the same
four in every round: "Hello!" and "Is it late in America/New_York?", under `"required"`
and under the named function, answered in prose.

## 5. The cost of the mask

`cargo run --release -p peregrine-bench --example constrain_bench --features arch-qwen,arch-llama -- <model> 4`
with `PEREGRINE_MICRO=1` (`raw/constrain_bench.txt`), Qwen 2.5's 151 936-token vocabulary:

| state | first call | the next request's first call | warm |
|---|---|---|---|
| free text | 64.8 ms | 14 µs | 3.0 µs |
| free text, part of the terminator matched | 65.4 ms | 15 µs | 4.3 µs |
| free text, inside a UTF-8 character | 71 µs | 24 µs | 3.9 µs |
| a literal that a marker token can write | 19 µs | 14 µs | 3.2 µs |
| the middle of a literal | 15 µs | 12 µs | 4.1 µs |
| the JSON value after a literal | 20 µs | 15 µs | 4.2 µs |

The 65 ms is the one walk of the whole vocabulary for a (terminator, matched bytes) pair;
the vocabulary keeps the result for every later request of the model. The JSON states of
S6.6a are unchanged (0.06–0.24 ms first, 3 µs warm), and vocabulary preparation is 192 ms
(median of five; S6.6a's budget is 200). Decode speed in the same run: unconstrained 44.1
tok/s, JSON object 44.6, the medium schema 49.3, the large one 42.9 (medians of five
interleaved runs on a loaded host: level).
