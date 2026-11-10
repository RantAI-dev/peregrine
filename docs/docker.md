# Docker

The image runs the OpenAI-compatible server (`peregrine serve`) on a CPU, with no shell and no
root user: the runtime stage is Debian 12's glibc from distroless, and the process runs as
uid/gid 65532 (`nonroot`).

```sh
docker build -t peregrine .
docker run --rm -p 8201:8201 -v /path/to/models:/models:ro peregrine
curl -s localhost:8201/health
curl -s localhost:8201/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

The default command looks for `/models/model.gguf`. To serve another file, put the flags the
server takes after the image name — they are appended to `peregrine serve`:

```sh
docker run --rm -p 8201:8201 -v /path/to/models:/models:ro peregrine \
  -m /models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --host 0.0.0.0 --port 8201 -t 4
```

Mount the model directory read-only. The mounted file must be readable by uid 65532 (the default
`docker run -v` mount keeps the host's permissions, and a `0600` file is not readable). Either
`chmod o+r` the file, or run with your own uid:

```sh
docker run --rm -p 8201:8201 --user "$(id -u):$(id -g)" \
  -v /path/to/models:/models:ro peregrine -m /models/model.gguf
```

## The image

| | |
|---|---|
| Builder | `rust:1.99.1-slim-bookworm`, pinned by version and index digest, matching `rust-toolchain.toml` |
| Runtime | `gcr.io/distroless/cc-debian12:nonroot`, pinned by digest; no shell, no package manager |
| Binary | `peregrine` built with its default features (all architectures, all weight formats, the server) |
| User | `nonroot` (uid/gid 65532) |
| Volume | `/models` |
| Port | `8201` |
| Entrypoint | `/usr/local/bin/peregrine serve` |
| Default command | `-m /models/model.gguf --host 0.0.0.0 --port 8201` |

The binary is built with baseline x86-64/aarch64 code and picks AVX2/AVX-512 or NEON at run time,
so one image works across CPU generations. It is statically linked against nothing but the
system's glibc and `libgcc_s`, which the runtime image provides.

The container binds `0.0.0.0` so that publishing a port works; the server has no authentication
(see `SECURITY.md`). Publish it on a host interface you trust, or bind the host side to localhost
with `-p 127.0.0.1:8201:8201`.

## Server flags

Every flag is `peregrine serve`'s; `docker run peregrine --help` prints the full list. The ones
worth setting in a container:

| Flag | Default | Meaning |
|---|---|---|
| `-m, --model PATH` | `/models/model.gguf` (from the image's CMD) | The GGUF file to serve |
| `--host ADDR` | `127.0.0.1` | Bind address; the image's CMD sets `0.0.0.0` |
| `--port PORT` | `8201` | Listen port; `EXPOSE 8201` |
| `-t, --threads N` | available CPUs | Worker threads. The container sees the host's CPUs unless `--cpus` is set, so pass the number you want |
| `-c, --ctx N` | `8192` | Context length, clamped to the model's training context |
| `--slots N` | `4` | Requests served at once, in one shared weight pass |
| `--alias NAME` | `peregrine` | The id `/v1/models` reports (requests may name any id) |
| `--chat-template NAME` | the file's own | Needed only for a file that declares no template |
| `--reasoning-format F` | `auto` | `none`, `auto`/`deepseek`, or `deepseek-legacy` |
| `--kv F16\|F32\|I8` | f16 | KV-cache precision |
| `--prefix-cache-mb N` | `768` | Memory for cache snapshots of earlier prompts; `0` turns it off |
| `--slots N` | `4` | Users served at once |
| `--opt NAME=VALUE` | — | Any `PEREGRINE_<NAME>` engine option, e.g. `--opt MTP=0`; repeatable |
| `--profile PATH` / `--no-profile` | host profile | Tuning profile; containers usually have none, so the defaults apply |

## Other architectures

The image builds for the host architecture by default. To build both Linux targets on one machine:

```sh
docker buildx build --platform linux/amd64,linux/arm64 -t peregrine .
```

CI builds the image for `linux/amd64` on every change (build only; nothing is pushed), so the
`Dockerfile` is exercised even though the benchmarks run on other hosts.

## Checking the container

The server's own endpoints are the health check; distroless has no shell or `curl`:

```sh
curl -sf localhost:8201/health            # {"status":"ok"}
curl -sf localhost:8201/v1/models         # the served model
docker logs <container>                    # load line: file, threads, tier, ctx, slots
```

`/metrics` is Prometheus text with the queue gauges llama-server exports, so existing scrapes keep
working.
