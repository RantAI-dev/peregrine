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

Each request becomes a slot in the engine's scheduler; all active slots share one weight pass per step.
The server binds to localhost and has no authentication (see `SECURITY.md`). Run `--help` for every flag.
