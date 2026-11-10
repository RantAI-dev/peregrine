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
streamed or not) plus `/v1/models`, `/health` and `/metrics`. One process serves one model, as with
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

## 6. Keep it running

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
