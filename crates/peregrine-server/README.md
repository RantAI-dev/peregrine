# peregrine-server

One GGUF model served by the Peregrine engine behind the OpenAI chat API.

```sh
peregrine-server -m model.gguf [--port 8201] [--host 127.0.0.1] [-t THREADS] [-c CTX] [--slots N]
```

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | streamed (SSE) or not, with `usage` and llama-style `timings`; `n > 1` without streaming |
| `GET /v1/models` | the served model (also the health check) |
| `GET /health` | liveness |
| `GET /metrics` | Prometheus text: slots, requests, tokens, plus llama-server-compatible queue gauges |

Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, and the penalties `presence_penalty`,
`frequency_penalty`, `repeat_penalty` and `repeat_last_n` (llama.cpp semantics; the window includes the
prompt). A request with penalties is never speculated (MTP stays off for it), which keeps output lossless.
Sampling controls the engine does not implement (`typical_p`, `tfs_z`, `mirostat*`, `dry_*`, `xtc_*`,
`top_n_sigma`, a non-empty `logit_bias`) are refused with a 400 naming the field, unless sent at their
no-op value.

Each request becomes a slot in the engine's scheduler; all active slots share one weight pass per step.
The server binds to localhost and has no authentication (see `SECURITY.md`). Run `--help` for every flag.
