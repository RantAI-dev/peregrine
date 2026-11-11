# Quickstart

Run a model, call it. About five minutes: one binary (or one container), one small model, one
request — the same requests then work against any bigger model file.

You need Linux on x86-64 or aarch64 (macOS: build from source below), and about 300 MB of disk for
the example model.

## 1. Get Peregrine

### Install script

The script detects the architecture, downloads the matching release archive, verifies it against the
release's `SHA256SUMS`, and installs `peregrine` to `~/.local/bin` (or `--prefix DIR`):

```sh
curl -fsSL https://peregrine.rantai.dev/install.sh | sh
peregrine tier   # the kernel tier picked for your CPU (AVX-512 / AVX2 / NEON / scalar)
```

The vanity host redirects to `install.sh` in the newest GitHub release; the same script is in the
repository at `scripts/install.sh`, and works with `sh` as well as `bash`.

Pass `--version vX.Y.Z` to pin a release and `--prefix /usr/local` for a system-wide install;
`PEREGRINE_REPO` and `PEREGRINE_BASE_URL` point the script at a fork or a mirror.

### Release binary

Download the archive for your architecture from the
[releases page](https://github.com/RantAI-dev/peregrine/releases) and put the binary on your `PATH`:

```sh
VERSION=v0.1.2   # the newest release tag on the releases page
ARCH=$(uname -m) # x86_64 or aarch64
curl -fLO "https://github.com/RantAI-dev/peregrine/releases/download/$VERSION/peregrine-$VERSION-linux-$ARCH.tar.gz"
tar -xzf "peregrine-$VERSION-linux-$ARCH.tar.gz"
mkdir -p ~/.local/bin && mv "peregrine-$VERSION-linux-$ARCH/peregrine" ~/.local/bin/
peregrine tier   # the kernel tier picked for your CPU (AVX-512 / AVX2 / NEON / scalar)
```

The binary is built for the plain baseline of each architecture: AVX2, AVX-VNNI and AVX-512 are
picked at run time on x86-64, NEON variants on aarch64 — the same file runs on any CPU of its
architecture. `peregrine tier` prints what it picked.

### Docker

The repository ships a multi-stage `Dockerfile` (a small Debian image with the server, the model
mounted at run time):

```sh
git clone https://github.com/RantAI-dev/peregrine && cd peregrine
docker build -t peregrine-server .
docker run --rm -p 8201:8201 \
  -v "$HOME/models/SmolLM2-360M-Instruct.Q4_K_M.gguf:/models/model.gguf:ro" \
  peregrine-server
```

### From source

```sh
git clone https://github.com/RantAI-dev/peregrine && cd peregrine
cargo build --release
./target/release/peregrine tier
```

## 2. Get a model

Peregrine runs GGUF files: Llama, Qwen 2/3/3.5, Gemma 3, Phi-3/4, Qwen 3 MoE, gpt-oss — see the
support table in the [README](../README.md). Any file works; the quickstart uses
**SmolLM2-360M-Instruct (Q4_K_M, ~270 MB)** because it is small and still answers like a chat model.
Keep models wherever you like — this page uses `~/models`:

```sh
mkdir -p ~/models
curl -fL -o ~/models/SmolLM2-360M-Instruct.Q4_K_M.gguf \
  "https://huggingface.co/mradermacher/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct.Q4_K_M.gguf"
```

`peregrine inspect <file>` says what a file is and whether this build runs it — useful before
downloading something large, or when a file needs a feature this build left out.

Or skip the manual download: every `-m` accepts `hf:<namespace>/<repo>/<file>` (with an optional
`@<rev>`), and Peregrine fetches the file once into
`$XDG_CACHE_HOME/peregrine/models/<repo>/<file>` (`$PEREGRINE_CACHE_DIR` overrides it; the
repository keeps two repos' `model-Q4_K_M.gguf` apart). Only the GGUF header is
read first, so a model this build cannot run is refused before the weights are downloaded:
`hf:mradermacher/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct.Q4_K_M.gguf`. A download that is
interrupted resumes from its `.part` file on the next run.

The load-time packed weights are cached on disk too, under
`~/.cache/peregrine/packs/` (`$XDG_CACHE_HOME/peregrine/packs`, or `$PEREGRINE_CACHE_DIR/packs`).
The first load of a model writes them; later loads map that file read-only, so a one-shot `run`,
`embed` or `decide` does not repack, and two processes serving the same model share the packed pages.
`PEREGRINE_PACK_CACHE=<dir>` moves the directory and `PEREGRINE_PACK_CACHE=0` turns the cache off;
`PEREGRINE_PACK_CACHE_MB` sets its size cap (default 4096 MiB, least-recently-used packs are evicted).
`peregrine inspect <file>` and `peregrine tier -m <file>` report a model's cache and its size. To
clear the cache, delete the directory: `rm -rf ~/.cache/peregrine/packs`.

## 3. Run one prompt

No server and no client needed for a single reply — it streams to stdout:

```sh
peregrine run -m ~/models/SmolLM2-360M-Instruct.Q4_K_M.gguf "Say hello in one sentence."
```

`--max-tokens`, `--temperature` (0 is greedy and the default), `--top-p`, `--top-k`, `--min-p`,
`--seed`, `-c`, `-t`, `--system`, `--chat-template` and `--kv` work as they do in `serve`. With no
prompt argument it reads stdin, so pipes and scripts work:

```sh
echo "Name three colors." | peregrine run -m hf:mradermacher/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct.Q4_K_M.gguf
```

`--json` prints one object instead of streaming: the text, prompt and completion token counts,
prefill and decode tokens per second, time to first token and total wall time. `--quiet` drops the
timing line from stderr.

## 4. Serve it

```sh
peregrine serve -m ~/models/SmolLM2-360M-Instruct.Q4_K_M.gguf --port 8201
```

You should see the model, the detected kernel tier, the chat template and the listening address in
the log. When no tuning profile exists (`peregrine tune`) and no thread count or spin window is
pinned, the server also calibrates those two for this host in under 200 ms and prints what it chose;
`PEREGRINE_CALIBRATE=0` turns that off. The server speaks the OpenAI chat API (`/v1/chat/completions`,
streamed or not) plus `/v1/models`, `/health` and `/metrics`, and `/v1/embeddings` when the file is an
embedding model. One process serves one model, as with
llama-server; `-t` caps the threads, `-c` the context, and `--slots` the number of users served in one
weight pass.

## 5. Call it

With `curl` (the `model` field is accepted but ignored — one process is one model):

```sh
curl -s http://127.0.0.1:8201/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"peregrine","messages":[{"role":"user","content":"Say hello in one sentence."}]}'
```

Streamed, token by token (Server-Sent Events):

```sh
curl -sN http://127.0.0.1:8201/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"peregrine","messages":[{"role":"user","content":"Name three colors."}],"stream":true}'
```

With the OpenAI Python client (`pip install openai`) — point `base_url` at the server, the key can
be any string:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8201/v1", api_key="not-needed")

reply = client.chat.completions.create(
    model="peregrine",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(reply.choices[0].message.content)
print(reply.usage)  # prompt + completion token counts

stream = client.chat.completions.create(
    model="peregrine",
    messages=[{"role": "user", "content": "Name three colors."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Standard sampling parameters work: `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`,
`stop`, `presence_penalty`, `frequency_penalty`, `n`, `logprobs`. Reasoning models (Qwen 3, gpt-oss)
split their thinking into `reasoning_content`, per `--reasoning-format` and the request's
`reasoning_format`. Fields the engine does not implement are refused with a 400 rather than
silently ignored.

## 6. Embeddings

A causal model trained as an embedder — Qwen3-Embedding, for example — or a bidirectional encoder such
as EmbeddingGemma serves a vector per text from the same server and the same engine:

```sh
peregrine serve -m hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf --port 8201
curl -s http://127.0.0.1:8201/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["What is the capital of France?","Paris is the capital of France."],"model":"peregrine"}'
```

The response is the OpenAI shape: `{object:"list", data:[{object:"embedding", index, embedding}],
model, usage}`. `input` may be a string, an array of strings, an array of token ids or an array of
token-id arrays; `dimensions` truncates before normalising; `encoding_format` is `float` or `base64`.
The pooling type (`last` or `mean`) and the EOS token come from the file, and `peregrine inspect`
prints them. Vectors are L2-normalised and bit-identical whatever else is in the batch.

The route accepts exactly these fields, and refuses any other key with a 400 that names it (so an
option the server would otherwise ignore cannot change the vector silently):

| Field | Value |
|---|---|
| `input` | a string, an array of strings, an array of token ids, or an array of token-id arrays |
| `model` | any string; echoed back in the response |
| `dimensions` | an integer from 1 to the model's width; truncates (Matryoshka) before normalising |
| `encoding_format` | `"float"` (the default) or `"base64"` (little-endian `f32`) |
| `truncate` | a boolean, default `false`: truncate an over-length text instead of refusing it |
| `user` | any string; accepted for compatibility, unused |

Every text of a request is admitted at once, within the engine's budget for work that never
decodes: a pooled slot's cache is sized to its own text, not to the context, and the budget is
`PEREGRINE_WORK_CACHE_MB` (512 MiB by default). Embedding and decision work never takes a chat
slot (`--slots` bounds chat only), and the response lists the vectors in input order.

Two engine options matter for this work; `peregrine tier` prints the resolved pass budget, and
`serve --opt NAME=VALUE` sets either:

| Option | Default | Meaning |
|---|---|---|
| `PEREGRINE_PASS_ROWS` | from the machine and the file | rows one weight pass carries for embedding and decision slots. More rows means fewer passes but a compute-bound pass; the default keeps a pass at the measured ridge (16 for Q8_0, ~22 for F16, 32 for an encoder) |
| `PEREGRINE_WORK_CACHE_MB` | `512` | memory for the prompt-sized caches of slots that never decode, i.e. how many texts of a request are in flight at once |

Without a server, one process embeds every text it is given:

```sh
peregrine embed -m hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf \
  "What is the capital of France?"
peregrine embed -m model.gguf --json --dimensions 256 < texts.txt
```

A text longer than the context is an error naming the limit rather than a silent truncation; pass
`truncate: true` (API) or `--truncate` (CLI) to opt in.

EmbeddingGemma 300M (`gemma-embedding`) is a bidirectional encoder: a text is forwarded whole (it is
never chunked and no KV cache is written), mean-pooled, then passed through `dense_2`/`dense_3` and
normalised. It serves the same endpoints, and a chat request on it is a 400 that says it is an
embedding model:

```sh
peregrine embed -m hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf \
  "task: search result | query: What is the capital of France?"
```

The BERT family (`bert` in the file) is served the same way: all-MiniLM, bge-small/base/large,
e5, gte and the XLM-R models (bge-m3, multilingual-e5). Their blocks are post-norm with biases on
every projection, learned absolute positions and LayerNorm, and pooling is the file's `cls` or
`mean`. The text limit is the model's position table — 512 for MiniLM and BGE, 8192 for bge-m3 —
and an over-length text is a 400 that names it unless `truncate` (API) or `--truncate` (CLI) is
given:

```sh
peregrine embed -m hf:second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q8_0.gguf \
  "What is the capital of France?" "Paris is the capital of France."
peregrine serve -m hf:CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf --port 8202
```

## 7. Decisions (System One)

A native decision model answers questions about a state without generating text: each option is a
token whose logit the answer is read from. The `openjev` type is supported in this build, and the
endpoint shares the engine and weight passes with chat and embeddings:

```sh
peregrine serve -m hf:ggml-org/tinyopenjev-for-testing-gguf/tinyopenjev-for-testing-Q8_0.gguf --port 8201
curl -s http://127.0.0.1:8201/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
  "questions": {
    "route":   {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": null, "shipping": null, "technical": null}},
    "angry":   {"type": "noul", "instructions": "Is the customer angry?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today", "right now"]}
  }
}'
```

That model file is a 46 MB test-size model published for trying the endpoint. The body takes:

| Field | Value |
|---|---|
| `state` | the text, object or array the questions are about (required) |
| `questions` | a non-empty object of id → question, in request order (required) |
| `questions.<id>.type` | `"choice"`, `"score"` or `"noul"` |
| `questions.<id>.instructions` | the question's text, object or array (required) |
| `questions.<id>.criteria` | choice: a non-empty object of key → description, options in request order; score: 2–10 level descriptions, lowest first, keys `"0"`…; noul: an optional object describing `"false"`/`"true"` |
| `files`, `images` | a 501: image input is not supported yet |
| `videos` | a 400, as in llama.cpp |

The answer for each question is one of `{"type":"choice","choice":...,"probabilities":{...},
"confidence":...}`, `{"type":"score","score":...,"legend":{...},"probabilities":{...},
"confidence":...}` or `{"type":"noul","noul":...}`, in request order. `usage.input_tokens` is the
sum of the prompts' token counts and `output_tokens` is 0. A bad request is a 400 with llama.cpp's
message and the question it is about (`questions.<id>: ...`); a model that is not a decision model
is a 501. `GET /v1/models` reports `architecture.output_modalities` as `["decisions"]` for a
decision model and `["text"]` otherwise. More questions than free slots queue and still answer in
order.

Without a server:

```sh
peregrine decide -m model.gguf --state @message.txt --questions @questions.json
peregrine decide -m model.gguf --request @request.json --json
```

`--state` and `--questions` accept the value itself, `@file` or `-` (stdin); a state that is valid
JSON is that value, anything else a string. A file whose decision type or `systemone` template this
build does not render is refused at load, naming the type.

## 8. Keep it running

For a long-lived server, `--background` detaches it, logs to a file and records state:

```sh
peregrine serve -m ~/models/SmolLM2-360M-Instruct.Q4_K_M.gguf --port 8201 --background
peregrine status
peregrine stop
```

`peregrine upgrade` installs the newest release and replaces a running background server in place:
the new process binds the same port with `SO_REUSEPORT`, the old one drains its in-flight requests,
and no connection is refused. See [service.md](service.md) for the state file, logs, `--no-restart`,
systemd and the caveats.

After a local rebuild, `kill -HUP <pid>` does the same handoff to the binary on disk:
`--background` already enables `SO_REUSEPORT`, a foreground server needs `--reuse-port`.

## Where next

- [Running it as a service](service.md) — background mode, `status`/`stop`, zero-downtime
  `upgrade`, systemd.
- [`peregrine bench`](dev/benchmarking.md) — prefill/decode tok/s and bytes/token against your
  machine's bandwidth ceiling.
- [`peregrine tune`](dev/benchmarking.md) — search the execution knobs for this host and model,
  saved as a tuning profile.
- [`/metrics`](../ARCHITECTURE.md) — the engine's options and telemetry; the architecture map
  explains what the engine reports.
- [OpenAI compatibility](../tests/compat/README.md) — the suite that pins the API surface above.
