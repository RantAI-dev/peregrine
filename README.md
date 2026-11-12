<div align="center">

# Peregrine

**A CPU inference engine for language models, built to be the fastest on the hardware people already have.**

[![CI](https://github.com/RantAI-dev/peregrine/actions/workflows/ci.yml/badge.svg)](https://github.com/RantAI-dev/peregrine/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MSRV 1.96](https://img.shields.io/badge/rustc-1.96+-orange.svg)](Cargo.toml)

</div>

Peregrine runs GGUF models on CPUs, in pure Rust with one runtime dependency (`memmap2`). It reports the
**bytes it moved per token** beside tokens per second, so every number can be checked against the machine's
own memory-bandwidth ceiling.

## Install

Linux x86-64 and Linux aarch64 have release archives today. macOS on Apple Silicon and Windows on
x86-64 build and pass the whole suite on every change, and their archives and installers ship
**from the release after v0.2.0**; until then build either from source (`cargo build --release`).
On Linux and macOS the installer detects the architecture, downloads the matching release archive,
verifies it against `SHA256SUMS`, and installs `peregrine` to `~/.local/bin`:

```sh
curl -fsSL https://peregrine.rantai.dev/install.sh | sh
peregrine tier      # the SIMD tier picked for this CPU: AVX-512, AVX2, NEON, i8mm, or scalar
```

On Windows, from the release after v0.2.0, in PowerShell (no administrator rights; the installer
uses your user `Path`):

```powershell
irm https://peregrine.rantai.dev/install.ps1 | iex
peregrine tier      # the SIMD tier picked for this CPU: AVX-512, AVX2, or scalar
```

Pin a release or choose the install directory by passing arguments to `sh`:

```sh
curl -fsSL https://peregrine.rantai.dev/install.sh | sh -s -- --version v0.2.0
curl -fsSL https://peregrine.rantai.dev/install.sh | sh -s -- --prefix /usr/local
peregrine upgrade            # self-update in place to the newest release
```

On Windows pass `-Version vX.Y.Z` to the script and `-InstallDir DIR` to choose the directory.

The release archives on the [releases page](https://github.com/RantAI-dev/peregrine/releases) are
`peregrine-v0.2.0-linux-x86_64.tar.gz` and `peregrine-v0.2.0-linux-aarch64.tar.gz`; they contain
portable binaries built for the baseline of each architecture. AVX-512, AVX2, AVX-VNNI, ARM NEON and
i8mm are selected at run time, so one binary runs on every CPU of that architecture;
`peregrine tier` prints what it picked. From the release after v0.2.0 the page also carries
`peregrine-vX.Y.Z-macos-aarch64.tar.gz` and `peregrine-vX.Y.Z-windows-x86_64.zip`. The macOS archive
is ad-hoc signed, not notarised: installed with `curl` it runs as is; see the
[quickstart](docs/quickstart.md#macos-on-apple-silicon) for a browser download, and Intel Macs
build from source. The Windows build links the C runtime statically, so it needs no Visual C++
redistributable; it is unsigned (no Authenticode certificate yet), and the
[quickstart](docs/quickstart.md#windows-x86-64) says what SmartScreen shows for a browser download.

**Docker** builds the server image (see [Docker](docs/docker.md)):

```sh
docker build -t peregrine .
docker run --rm -p 8201:8201 -v /path/to/models:/models:ro peregrine
```

**From source** (Rust 1.96 or newer):

```sh
cargo build --release
./target/release/peregrine tier
```

## Run a model

`peregrine run` sends one prompt and streams the reply to the terminal — no server, no client:

```sh
peregrine run -m hf:mradermacher/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct.Q4_K_M.gguf \
  "Say hello in one sentence."
```

With no prompt argument it reads stdin, so pipes and scripts work:

```sh
echo "Say hello in one sentence." | peregrine run -m model.gguf
peregrine run -m model.gguf --json "Say hello in one sentence."   # one JSON object with text and timings
```

Common flags (all are listed by `peregrine run --help`):

| Flag | Meaning |
|---|---|
| `-c, --ctx N` | Context length in tokens, clamped to the model's training context (default 8192) |
| `-t, --threads N` | Worker threads (default: every logical CPU) |
| `--max-tokens N` | Tokens to generate at most (default 256) |
| `--temperature T` | Sampling temperature; `0` is greedy and deterministic (default 0) |
| `--system TEXT` | A system message for the chat template |
| `--chat-template NAME` | Render a named template when the file declares none |
| `--kv f16\|f32\|i8` | KV-cache precision (default f16) |
| `--fast-prompt` | Experimental and lossy: prompt tokens read the weights of the formats whose 8-bit form beats this CPU tier's exact kernel (AVX-512: Q4_K and Q6_K; AVX2: Q6_K; a tier with no SIMD body: none, so the option changes nothing) from an 8-bit copy built at load. Faster prompts; a reply can differ from the default engine's, and the accuracy report is not published yet (generated tokens keep the exact kernels; off by default) |
| `--no-thinking` | Answer without the reasoning block (reasoning models only) |
| `--speculate` | Lossless prompt-lookup speculation (off by default) |
| `--json` | Print one JSON object (text, token counts, prefill/decode tok/s, TTFT, wall time) |

### `hf:` models

Any command that takes `-m` — `run`, `serve`, `embed`, `inspect`, `bench` and `tune` — accepts
`hf:<namespace>/<repo>/<file>[@<rev>]`. The file is downloaded once into
`~/.cache/peregrine/models/<repo>/<file>` (or `$PEREGRINE_CACHE_DIR` / `$XDG_CACHE_HOME/peregrine`) and
reused afterwards. Partial downloads resume from a `.part` file; the repository and optional revision are
part of the cache path, so two repos that publish the same file name cannot collide.

Before the weights arrive, Peregrine downloads only the first 16 MiB of the GGUF and checks that this
build can run the file, so an unsupported model is refused early. `HF_TOKEN` is sent as a bearer token for
gated or private repositories.

## Serve the OpenAI API

Start a foreground server and call it:

```sh
peregrine serve -m model.gguf --port 8201
curl -s localhost:8201/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

The server speaks `/v1/chat/completions` (streamed or not), plus `/v1/models`, `/health` and `/metrics`.
Its `response_format` constrains the reply to valid JSON: `{"type": "json_object"}` or a
`{"type": "json_schema", "json_schema": {...}}` document (S6.6a; the supported keyword table is in
[docs/compat.md](docs/compat.md#structured-output-response_format)).

`-m` may be repeated: every file is a model in the same process, and a request's `model` field
selects one. The first `-m` is the default for a request that names no model. With one `-m` the id is
`peregrine`, as before; with several, each file is named by its stem unless `--alias` says otherwise
(one `--alias` per `-m`, in order):

```sh
peregrine serve \
  -m chat.gguf --alias chat \
  -m embed.gguf --alias embed \
  -m decide.gguf --alias decide \
  --port 8201
curl -s localhost:8201/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"embed","input":"What is the capital of France?"}'
```

Each model has its own engine, slots and admission, and `/v1/models` lists what each can do. The
models take weight passes in turn — one pass at a time for the whole process, first come first
served — so they share the cores instead of competing for them, and a request's output does not
change because another model is busy. An unknown `model` is a 404. Every model loads at start; there
is no unloading of idle models.

### Tool calling

Send OpenAI's `tools` with a chat request and the model may answer with `tool_calls`:

```sh
curl -s localhost:8201/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
}'
```

The reply's `message.tool_calls` holds the call (`get_weather` with `{"city": "Paris"}`) and
`finish_reason` is `"tool_calls"`. Run the tool, then send the conversation back with the assistant's
message and a `tool` message carrying the result, and the model answers in text. Supported today:
Qwen 2.5, Qwen 3, Qwen 3.5 and Bonsai 2 chat models; `tool_choice` `"auto"` and `"none"`;
`parallel_tool_calls`; streaming. A Qwen 3, Qwen 3.5 or Bonsai 2 model thinks before it answers, and its
thinking comes back in `reasoning_content`, never read for calls; send
`"chat_template_kwargs": {"enable_thinking": false}` (or `"reasoning_effort": "none"`) to switch
it off. Other model families, and `tool_choice: "required"` or a named function, are refused with
a 400 until they are built ([design](docs/design/s6-tool-calling.md)).

For a long-lived service, `--background` detaches the server, writes a lifecycle state file and a log,
and returns once it is listening:

```sh
peregrine serve -m model.gguf --port 8201 --background
peregrine status       # pid, version, URL, model, log path
peregrine stop         # drain in-flight requests, then exit
```

`peregrine upgrade` replaces the running background server: on Linux and macOS it starts the new
binary beside the old one (`SO_REUSEPORT`), waits for it to report ready, then drains and stops the
old process, so no connection is refused. On Windows it uses the stop file to drain the old server
and then starts the new one (a short gap), because Windows has neither `SIGTERM` nor
`SO_REUSEPORT`. `kill -HUP <pid>` does the zero-downtime handoff for a rebuilt binary on Unix. See
[Service & upgrades](docs/service.md).

## Embeddings

A causal model that was trained as an embedder (it declares `<arch>.pooling_type` in its GGUF)
serves `POST /v1/embeddings` beside chat, on the same engine and the same weight passes:

```sh
peregrine serve -m hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf --port 8201
curl -s localhost:8201/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["What is the capital of France?","Paris is the capital of France."],"model":"peregrine"}'
```

The body follows OpenAI: `input` is a string, an array of strings, an array of token ids or an array of
token-id arrays; `dimensions` truncates the vector (Matryoshka, then L2 normalisation); and
`encoding_format` is `float` (default) or `base64` (little-endian f32). The pooling type is the file's
own — `last` for Qwen3-Embedding, `mean` for the models trained that way — and vectors come back in
input order, bit-identical alone or in any batch. A text longer than `-c` is a 400 that names the
limit, unless the request sets `truncate: true`.

Bidirectional encoders are embedders too. EmbeddingGemma 300M (`gemma-embedding`) runs every text as
one whole bidirectional pass — no chunking, no KV cache — and applies its mean pooling, its
`dense_2`/`dense_3` head, and the Matryoshka truncation:

```sh
peregrine embed -m hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf \
  "task: search result | query: What is the capital of France?"
```

The BERT encoders — all-MiniLM, BGE, E5, GTE and the XLM-R models (bge-m3, multilingual-e5) —
are the same kind of pass with a different block: post-norm, biases on every projection, learned
absolute positions, token types, LayerNorm and the exact GELU, pooled by the file's `cls` or
`mean`. One file serves every model of the family, and a long text is refused against the
model's own position table (512 for MiniLM/BGE, 8192 for bge-m3):

```sh
peregrine embed -m hf:second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q8_0.gguf \
  "What is the capital of France?"
peregrine serve -m hf:CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf --port 8202
```

ModernBERT (`modern-bert`) is the newer encoder: pre-norm blocks, RoPE with one base for the local
layers and another for the global ones, every third layer global and the two between seeing a
128-token symmetric window, and a gated feed-forward. One file serves ModernBERT-base and -large,
gte-modernbert and IBM Granite Embedding R2; the file's `cls` or `mean` pooling is used, and a text
longer than its 8192-token context is refused:

```sh
peregrine embed -m hf:keisuke-miyako/gte-modernbert-base-gguf/gte-modernbert-base-Q8_0.gguf \
  "What is the capital of France?"
```

Nomic's nomic-embed-text v1.5 (`nomic-bert`) is a BERT-shaped post-norm encoder with a fused
q/k/v, NeoX RoPE instead of a position table, a SwiGLU feed-forward and mean pooling, over a
2048-token context. It was trained with task prefixes, and the client writes them: `search_document: `
before a passage, `search_query: ` before a question. `dimensions` (512, 256, 64) truncates the
Matryoshka vector before normalising:

```sh
peregrine embed -m hf:nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf \
  "search_document: Paris is the capital of France." \
  "search_query: What is the capital of France?"
```

NVIDIA's Nemotron 3.5 Embed 1B (`mistral3`, OpenMDW 1.1) is another encoder: the Mistral 3 blocks
without the causal mask, queried over 24 heads of 128, rotated with YaRN and mean-pooled to 2048
dimensions. It was trained with a role prefix on each text, so write `query: ` before a question and
`passage: ` before a document:

```sh
peregrine embed -m hf:W4D/Nemotron-3.5-Embed-1B-GGUF/Nemotron-3.5-Embed-1B-Q8_0.gguf \
  "query: What is the capital of France?" \
  "passage: Paris is the capital of France."
```

A BERT reranker is the same encoder with its classifier head. Every (query, document) pair is one
text — `[CLS] query [SEP] document` for a WordPiece vocabulary, `<s> query </s> document </s>` for
the XLM-R one — and the head scores it. The score is the head's raw logit, as llama.cpp returns
it; the documents come back sorted, highest first:

```sh
peregrine serve -m hf:gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q8_0.gguf --port 8203
curl -s localhost:8203/v1/rerank -H 'content-type: application/json' -d '{
  "query": "What is the capital of France?",
  "documents": ["Paris is the capital of France.", "Lyon is a city in France."],
  "top_n": 2
}'
```

The same file at `/rerank`, `/v1/reranking` and `/reranking`, a `peregrine rerank -m model.gguf
--query "..." "doc one" "doc two"` on the command line, and `/v1/classify` for a head with
`classifier.output_labels` (every label and its probability, the top one first). A file without a
head answers 400 by name.

Without a server:

```sh
peregrine embed -m model.gguf "What is the capital of France?"   # one line of floats per text
echo "one text per line" | peregrine embed -m model.gguf          # stdin
peregrine embed -m model.gguf --json --dimensions 256 "..."       # the API's JSON shape
peregrine rerank -m reranker.gguf --query "..." "doc one" "doc two" # scores, highest first
```

`peregrine inspect` prints the pooling type a file declares and whether it is an embedding model.

## Decisions (System One)

A native decision model (a file with `<arch>.decision.type` and a `systemone` template) answers
questions about a state, each answer read from the option letters' logits rather than generated as
text. The `openjev` type is supported in this build; `POST /v1/systemone` shares the engine and the
weight passes with chat and embeddings:

```sh
peregrine serve -m hf:ggml-org/tinyopenjev-for-testing-gguf/tinyopenjev-for-testing-Q8_0.gguf --port 8201
curl -s localhost:8201/v1/systemone -H 'content-type: application/json' -d '{
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

That file is a 46 MB test-size model, published for trying the endpoint; a production decision model
is larger. The body takes:

| Field | Value |
|---|---|
| `state` | the text, object or array the questions are about (required) |
| `questions` | a non-empty object of id → question, in request order (required) |
| `questions.<id>.type` | `"choice"`, `"score"` or `"noul"` |
| `questions.<id>.instructions` | the question's text, object or array (required) |
| `questions.<id>.criteria` | choice: a non-empty object of key → description; score: 2–10 level descriptions, lowest first; noul: an optional object describing `"false"`/`"true"` |
| `files`, `images` | refused with a 501: image input is not supported yet |
| `videos` | refused with a 400, as in llama.cpp |

Each question is answered as `{"type":"choice","choice":...,"probabilities":{...},"confidence":...}`,
`{"type":"score","score":...,"legend":{...},"probabilities":{...},"confidence":...}` or
`{"type":"noul","noul":...}`, in request order, with `usage.input_tokens` the sum of the prompts'
token counts. A validation error is a 400 naming the question (`questions.<id>: ...`) with
llama.cpp's message; a model that is not a decision model is a 501. `GET /v1/models` advertises
`architecture.output_modalities` as `["decisions"]` for a decision model and `["text"]` otherwise.
Requests with more questions than free slots queue and still answer in order.

Without a server:

```sh
peregrine decide -m hf:ggml-org/tinyopenjev-for-testing-gguf/tinyopenjev-for-testing-Q8_0.gguf \
  --state @message.txt --questions @questions.json          # one line per answer
peregrine decide -m model.gguf --request @request.json --json   # the API's response
```

`--state` and `--questions` also take `-` for stdin; a state that is valid JSON is used as that value,
anything else as a string.

## CLI reference

| Command | What it does |
|---|---|
| `peregrine run -m <MODEL> [PROMPT]...` | Run one prompt (or stdin) and stream the reply in the terminal (`--json` for structured output). |
| `peregrine serve -m <MODEL> [--port 8201] [--background]` | Start the OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/models`, `/health`, `/metrics`). |
| `peregrine status` | Show the running background server (PID, version, URL, model, log path). |
| `peregrine stop` | Gracefully drain in-flight requests and stop the background server. |
| `peregrine upgrade [--check] [--version TAG]` | Download, verify (`SHA256SUMS`), and atomically install the newest release with zero-downtime handoff (`SO_REUSEPORT`). |
| `peregrine embed -m <MODEL> [TEXT]...` | Embed texts (or stdin lines): one L2-normalised vector per text, the file's own pooling type (`--dimensions`, `--json`). |
| `peregrine rerank -m <MODEL> --query <Q> [DOC]...` | Rank documents (or stdin lines) by relevance to the query with a reranker's classifier head (`--top-n`, `--json`, `--truncate`). |
| `peregrine decide -m <MODEL> --state <TEXT> --questions <JSON>` | Answer questions about a state with a native decision model (System One): one line per answer, or the API JSON with `--json` (`--request` for a whole body). |
| `peregrine inspect <MODEL>` | Inspect GGUF metadata, architecture, quantization formats, pooling type, and check whether this build can run the file. |
| `peregrine bench -m <MODEL>` | Measure single-user prefill and greedy decode tok/s and memory bandwidth (bytes/token). |
| `peregrine tune -m <MODEL>` | Auto-tune thread counts and spin windows for this CPU + model and save a hardware profile. |
| `peregrine tier` | Print the detected CPU SIMD kernel tier (`x86_64-avx512`, `x86_64-avx2`, `aarch64-neon-i8mm`, etc.) and active engine options. |
| `peregrine support [--markdown]` | List all compiled-in architectures, weight formats, CPU tiers, tokenizers, and chat templates. |

## Performance

Against llama.cpp v0.9.0 for the Q4_K_M models and PrismML's llama.cpp fork for the ternary Bonsai 2
files, on the same file, machine and thread count (embedding models also against OpenVINO, ONNX Runtime,
fastembed and text-embeddings-inference):

![Where Peregrine is fastest](docs/img/bench-wins.svg)

- **Single-user decode up to 1.69× faster** on the 12-core desktop (SmolLM2 360M at 4096 context) and
  **1.07–1.34× faster** on the 4-core Ryzen 3, winning every completed decode row across Llama 3.2 1B,
  Qwen 2.5 0.5B, SmolLM2 360M and Gemma 3 1B. The lead grows with context because attention reads each
  cached key and value once per group of heads.
- **Up to 1.80× the throughput with 10 users at once** on the 4-core Ryzen 3 (1.47–1.73× at 512 context)
  and **1.28–1.43×** on the 12-core desktop.
- **Prompt processing 1.03–1.43× faster on x86 and up to 2.45× on ARM64**: 1.03–1.30× on the desktop,
  1.03–1.43× on the 4-core Ryzen 3, and 1.78–2.45× on ARM Cortex-X925 (`aarch64-neon-i8mm`).
- **Bonsai 2 (ternary): 1.4–2.3× the decode and 1.9–2.6× the prompt speed** of PrismML's fork across
  x86-64 (AVX-512 and AVX2) and ARM64 (NEON), and **1.8–2.8× with 10 users**.

- **Embeddings and decisions: fastest in 55 of 68 embedding cells and all nine decision cells** on the
  4-core Ryzen 3, against llama.cpp, OpenVINO, ONNX Runtime, fastembed and text-embeddings-inference at
  equal accuracy: bge-m3, EmbeddingGemma, Qwen3-Embedding and Nemotron 3.5 Embed are **1.18–1.62×**
  llama.cpp, and `/v1/systemone` decisions **1.33–1.84×**.

- **Where Peregrine is behind:** the ik_llama.cpp fork processes prompts about twice as fast on the
  AVX-512 desktop (Peregrine at 0.39–0.58×). Decode is level with it (0.89–1.06×) and Peregrine is ahead
  with 10 users (1.05–3.09×). For embeddings, OpenVINO and ONNX Runtime are faster on the two small BERT
  encoders at 128 and 512 tokens (Peregrine at 0.85–0.99×).

Full tables, charts, accuracy verification, and machine records are in [docs/benchmark.md](docs/benchmark.md).

## Why

- **Measured, not claimed.** Every result is reported as a fraction of the host's roofline, with the exact
  command, commit and options that produced it. Losses are published along with wins.
- **Lossless by default.** Speculation, batching and every kernel produce the same tokens as the exact
  reference path. Anything lossy is off by default and ships with its own accuracy report.
- **Many users, one weight pass.** A batching scheduler with chunked prefill, a shared prefix cache and
  position-keyed sampling: a user gets identical text alone or alongside others.
- **Owned hot path.** Kernels, thread pool, cache, sampler, tokenizer and scheduler are this project's
  code, so their speed and correctness are under its control.

## Supported models & hardware

<!-- support:begin -->
| | Supported now |
|---|---|
| Architectures | Llama, Qwen 2, Qwen 3, Qwen 3.5 / Bonsai 2 (Gated DeltaNet hybrid, MTP), Gemma 3, Gemma 3 embedding (EmbeddingGemma), BERT encoder (MiniLM, BGE, E5, GTE, bge-m3), ModernBERT encoder (ModernBERT, gte-modernbert, Granite Embedding R2), NomicBERT encoder (nomic-embed-text v1.5), Jina BERT v2 encoder (jina-embeddings-v2, ALiBi), Qwen 3 MoE, gpt-oss, Phi-3 / Phi-4, Mistral 3 embedding (Nemotron 3.5 Embed) |
| Weight formats | F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4, PQ2_0, PTQ1_0 |
| CPUs | x86-64 (AVX2, AVX-VNNI, AVX-512 VNNI/GFNI/VBMI), ARM64 (NEON, SDOT, i8mm); others through scalar code |
| Tokenizers | GPT-2 byte-level BPE (smollm, gpt2 / modern-bert, llama-bpe, qwen2, qwen35, gpt-4o, dbrx, tekken / pixtral, tekken-num3, nemotron-3.5-embed), SentencePiece BPE (llama), WordPiece (bert), SentencePiece unigram (t5) |
| Chat templates | read from the model file and rendered exactly as llama.cpp renders them: ChatML, Phi-4, Llama 3, Qwen 3, Qwen 3.5 (Bonsai 2), Gemma, Mistral (v0, v1, v7), Llama 2, OpenAI Harmony (gpt-oss) — or named with `--chat-template` for a file that declares none |
| Embeddings | causal decoders as embedders (`<arch>.pooling_type`: last or mean), and bidirectional encoders: BERT (MiniLM, BGE, E5, GTE, bge-m3: cls/mean), ModernBERT (ModernBERT, gte-modernbert, Granite Embedding R2: cls/mean, local and global attention), NomicBERT (nomic-embed-text v1.5: mean, RoPE, SwiGLU), Jina BERT v2 (jina-embeddings-v2: ALiBi, GEGLU, mean/cls), Gemma 3 (EmbeddingGemma: mean/cls with `dense_2`/`dense_3`) and Mistral 3 (Nemotron 3.5 Embed 1B: mean); `POST /v1/embeddings`, `peregrine embed` |
| Tool calling | OpenAI `tools`, `tool_choice` (`auto`, `none`) and `tool_calls` on `POST /v1/chat/completions`, streamed or not, for Qwen 2.5 (the `hermes` tool format), Qwen 3 (the `qwen3` format) and Qwen 3.5 and Bonsai 2 (the `qwen-xml` format), thinking on or off; the prompt is the one llama.cpp builds from the model's template |
| Rerank and classify | a `bert` or `modern-bert` file's classifier head (`cls`, `cls.norm`, `cls.output`, `classifier.output_labels`): `POST /v1/rerank` (and `/rerank`, `/v1/reranking`, `/reranking`), `POST /v1/classify`, `peregrine rerank`; bge-reranker-v2-m3, bge-reranker-base and gte-reranker-modernbert-base checked against llama.cpp v0.9.0 |
| Decision models | native decision models with a `systemone` template: the `openjev` type, `POST /v1/systemone`, `peregrine decide` |
<!-- support:end -->

Every supported family is verified against llama.cpp v0.9.0 on real model files for tokenizer parity,
rendered chat templates, greedy token streams, KL divergence, and downstream task accuracy
(GSM8K, MMLU, HumanEval+; see [docs/dev/accuracy.md](docs/dev/accuracy.md)).

## Building with fewer features

By default, Peregrine builds with all architectures, quantisation formats, and extras enabled. For embedded applications or smaller binaries, you can disable default features and pick only what you need:

```sh
# Dense Llama models only with K-quant formats:
cargo build --release -p peregrine --no-default-features --features cli,arch-llama,fmt-kquant

# Hybrid Qwen3.5 models with ternary formats:
cargo build --release -p peregrine --no-default-features --features cli,arch-qwen35,fmt-ternary
```

| Category | Features |
|---|---|
| Architectures | `arch-llama` (Llama), `arch-qwen` (Qwen 2/3), `arch-qwen35` (Qwen 3.5 / Bonsai 2), `arch-gemma3` (Gemma 3), `arch-gemma-embedding` (EmbeddingGemma), `arch-bert` (BERT encoders), `arch-modern-bert` (ModernBERT encoders), `arch-nomic-bert` (NomicBERT encoders), `arch-phi3` (Phi-3/4), `arch-qwen3moe` (Qwen 3 MoE), `arch-gptoss` (gpt-oss), `arch-mistral3` (Mistral 3 embedders) |
| Formats | `fmt-kquant` (Q2_K..Q6_K), `fmt-ternary` (PQ2_0, PTQ1_0), `fmt-iq` (IQ4_NL, IQ4_XS), `fmt-mxfp4` (MXFP4), `fmt-legacy` (Q4_0, Q5_0, Q8_0); F32/F16/BF16 always enabled |
| Extras | `littlebit` (LittleBit sidecars), `dspark` (speculative drafter), `mtp` (multi-token prediction head), `server` (OpenAI HTTP server), `cli` (CLI binary) |

## Documentation

| | |
|---|---|
| [Product](docs/product.md) | what Peregrine is, who it is for, what it will not do |
| [Quickstart](docs/quickstart.md) | a release binary or Docker, a model, and the first request |
| [Service & upgrades](docs/service.md) | background mode, state and log files, zero-downtime upgrades |
| [Docker](docs/docker.md) | the server image, volumes, ports and container flags |
| [Targets](docs/targets.md) | the speed, accuracy and support numbers we hold ourselves to |
| [Benchmarks](docs/benchmark.md) | Peregrine against llama.cpp, wins and losses, with charts |
| [Roadmap](docs/roadmap.md) | the plan, stage by stage |
| [Architecture](ARCHITECTURE.md) | a map of the code |
| [Contributing](CONTRIBUTING.md) | how to build, test, benchmark and send changes |
| [Decisions](docs/adr/README.md) | why things are the way they are |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions are accepted under the same license.
