# OpenAI API compatibility

The server speaks the subset of the OpenAI API that chat clients use. `tests/compat/compat.py` is the
executable definition of that claim: it starts (or attaches to) a server and checks the HTTP shapes,
streaming, sampling controls, error bodies and concurrency, then prints a PASS/FAIL/DEVIATION/SKIP
line per check.

```sh
# Build the engine, then run the suite against a fresh server on SmolLM2-360M-Instruct:
cargo build --release -p peregrine
python3 tests/compat/compat.py \
  --model /path/to/SmolLM2-360M-Instruct.Q4_K_M.gguf \
  --peregrine-bin target/release/peregrine

# Qwen 3 has a reasoning channel; --thinking also checks the reasoning/answer split.
python3 tests/compat/compat.py --model /path/to/qwen3-1.7b-q4km.gguf --thinking

# Or point it at an already-running server (Peregrine or llama-server):
python3 tests/compat/compat.py --base-url http://127.0.0.1:8201 --engine llama-server
```

CI runs the SmolLM2 pass with the model's SHA-256 pinned (see `.github/workflows/ci.yml`); the
thinking-model pass needs a second file and is run by hand. `tests/compat/README.md` has the model
uploads and the llama-server comparison command.

## What is covered

| Check | |
|---|---|
| `GET /health`, `GET /v1/models` | liveness; a `list` of `model` objects with a string `id` |
| `POST /v1/chat/completions` | object/id/model/choices, `message.content`, `usage` that adds up, `finish_reason` |
| streaming | `chat.completion.chunk` objects, one id, deltas, one `finish_reason`, `usage` (with `stream_options.include_usage`), `data: [DONE]` |
| `POST /v1/completions` | called; reported as a deviation when absent |
| sampling | `temperature: 0` reproducible; same `temperature`/`top_p`/`seed` reproducible from the first request |
| `stop` | a string or up to four strings: the reply must end at the sequence, the sequence must not be returned, `finish_reason` is `"stop"`, and streaming must hold back partial matches (the string, array and SSE forms must agree) |
| `max_tokens` | exactly that many tokens with `ignore_eos`, `finish_reason: "length"` |
| `n > 1` | three choices indexed 0..2; `n` with `stream` is either supported or refused cleanly |
| `logprobs` | one entry per content token, numeric log-probability ≤ 0 |
| errors | malformed JSON, missing/empty/non-array `messages`, a last assistant turn, an unknown model, an unsupported sampler, `n > 16`, unknown path, wrong method — all with the right status and, where applicable, an OpenAI error object |
| concurrency | eight simultaneous requests all succeed, and the server is healthy afterwards |
| `/metrics` | the llama-server-compatible queue gauges |
| reasoning | with `--thinking`: `reasoning_content` and `content` are split, no `<think>` tags; `reasoning_format: "none"` puts the analysis in `content`; streaming splits the same way |

A DEVIATION is a known difference from the OpenAI specification (or from llama-server), not a
failure: the suite exits non-zero only on FAIL. The current deviations are below.

## Current deviations from the OpenAI specification

Verified with the suite on SmolLM2-360M-Instruct Q4_K_M and Qwen3-1.7B Q4_K_M (the uploads in
`docs/dev/testing.md`), November 2026.

| Difference | Effect |
|---|---|
| `POST /v1/completions` is not implemented | 404. OpenAI and llama-server serve it. Use `/v1/chat/completions` |
| `n > 1` with `stream: true` | 400. OpenAI and llama-server stream n choices |
| `n > 16` | 400. OpenAI documents no maximum |
| `logprobs` entries carry only `token` and `logprob` | OpenAI also returns `bytes` and `top_logprobs`; the request's `top_logprobs` is ignored |
| An unknown `model` id is accepted | The request is served by the one loaded model. The flag documentation says so; OpenAI returns 404 `model_not_found` |
| Streaming `usage` is always present | OpenAI only sends it with `stream_options.include_usage`. Peregrine attaches it to the chunk that carries `finish_reason`; OpenAI sends a separate final chunk with empty `choices` |
| `message.reasoning_content` is always present | Empty string when there is no reasoning; OpenAI has no such field |
| `timings` is always present | llama-server also adds it; OpenAI has none |
| `ignore_eos` is accepted | Non-standard (llama-server has the same field); off by default |
| Unsupported sampling fields (`typical_p`, `mirostat*`, `dry_*`, `xtc_*`, `top_n_sigma`, a non-empty `logit_bias`) are refused with 400 naming the field | Deliberate: silently sampling differently is worse than an error. At their no-op values they are accepted |
| `reasoning_format: "none"` consumes `<think>` tags | Peregrine never leaks control tokens into `content`; llama-server's `none` leaves the tags in `content` |
| A model object has no `created` timestamp | It has `id`, `object`, `owned_by` and an extra `weights` |

Everything else the suite checks matches: the response objects, SSE framing and `[DONE]`, the
`usage` arithmetic, `finish_reason` values, error statuses and the OpenAI error object (including a
malformed JSON body, which `fix: answer a malformed JSON body with the OpenAI error object` made
JSON), and eight-way concurrency.

## Against llama-server

Run on the same files on the same aarch64 host (llama.cpp 0.9.0-dev, commit `0c6dca2`,
`llama-server -m MODEL --jinja --metrics`):

- **Peregrine** on SmolLM2: 11 passed, 3 deviations, 1 skipped (the reasoning check needs a thinking
  model). On Qwen3-1.7B with `--thinking`: 12 passed, 3 deviations.
- **llama-server** on SmolLM2: 12 passed, 2 deviations. On Qwen3-1.7B with `--thinking`: 12 passed,
  3 deviations.

The differences the suite found in llama-server's favour:

- `/v1/completions` works there.
- `stop` works on both.
- `n > 1` streams there.
- `logprobs` entries carry `bytes` and `top_logprobs`.
- Its SSE stream follows OpenAI's two-chunk ending (finish, then a usage-only chunk).
- `reasoning_format: "none"` leaves `<think>` in `content` (the llama.cpp meaning of "none").

The differences the suite found in Peregrine's favour:

- `seed` reproduces from the first request. llama-server's first request after start differs and only
  later identical requests reproduce (a cold-slot sampler state, not the OpenAI contract, but worth
  knowing when replaying a trace).
- Peregrine always sends `usage` in a stream, so a client does not need `stream_options`.
- A malformed JSON body is answered with the OpenAI error object. llama-server returned 400 without a
  JSON content type in this run.

Two differences are deliberate and stay: a server with one model accepts any model id (there is
nothing to choose, as with llama-server; a server started with several `-m` answers an unknown id
with OpenAI's 404), and sampling controls the engine cannot honour are refused by name.
