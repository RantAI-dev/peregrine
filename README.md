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

## Performance

Against the fastest engine that runs the same file on the same machine and thread count —
llama.cpp v0.9.0 for the Q4_K_M models, and PrismML's llama.cpp fork for the ternary Bonsai 2 files:

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
| Architectures | Llama, Qwen 2, Qwen 3, Qwen 3.5 / Bonsai 2 (Gated DeltaNet hybrid, MTP), Gemma 3, Qwen 3 MoE, gpt-oss, Phi-3 / Phi-4 |
| Weight formats | F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4, PQ2_0, PTQ1_0 |
| CPUs | x86-64 (AVX2, AVX-VNNI, AVX-512 VNNI/GFNI/VBMI), ARM64 (NEON, SDOT, i8mm); others through scalar code |
| Tokenizers | GPT-2 byte-level BPE, SentencePiece (llama) |
| Chat templates | read from the model file and rendered exactly as llama.cpp renders them: ChatML, Phi-4, Llama 3, Qwen 3, Gemma, Mistral (v0, v1, v7), Llama 2, OpenAI Harmony (gpt-oss) — or named with `--chat-template` for a file that declares none |
<!-- support:end -->

Every supported family is verified against llama.cpp v0.9.0 on real model files for tokenizer parity,
rendered chat templates, greedy token streams, KL divergence, and downstream task accuracy
(GSM8K, MMLU, HumanEval+; see [docs/dev/accuracy.md](docs/dev/accuracy.md)).

## Quick start

See [docs/quickstart.md](docs/quickstart.md) for a complete walkthrough using a one-shot prompt,
a release binary, Docker, `curl`, or the OpenAI Python client. Any `-m` also accepts
`hf:<namespace>/<repo>/<file>`, which downloads the file once into the cache.

```sh
# install the binary (Linux x86-64 or aarch64), then run a model
curl -fsSL https://peregrine.rantai.dev/install.sh | sh
peregrine run -m hf:mradermacher/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct.Q4_K_M.gguf \
  "Say hello in one sentence."                    # one prompt, streamed, no server
peregrine inspect path/to/model.gguf      # what it is, and whether it runs
peregrine bench -m path/to/model.gguf        # measure prefill & decode tok/s and bytes/token
peregrine tune -m path/to/model.gguf         # tune execution knobs for host CPU and model
peregrine serve -m path/to/model.gguf --port 8201
curl -s localhost:8201/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

Or build from source: `cargo build --release`, then use `./target/release/peregrine`.

The server speaks the OpenAI chat API (`/v1/chat/completions`, streamed or not), plus `/v1/models`,
`/health` and `/metrics`. For a long-lived service, `peregrine serve ... --background` detaches it and
records lifecycle state; `peregrine status`, `stop` and `upgrade` manage it, and an upgrade replaces
the running server without refusing a connection (see [docs/service.md](docs/service.md)).

### Building with fewer features

By default, Peregrine builds with all architectures, quantisation formats, and extras enabled. For embedded applications or smaller binaries, you can disable default features and pick only what you need:

```sh
# Dense Llama models only with K-quant formats:
cargo build --release -p peregrine --no-default-features --features cli,arch-llama,fmt-kquant

# Hybrid Qwen3.5 models with ternary formats:
cargo build --release -p peregrine --no-default-features --features cli,arch-qwen35,fmt-ternary
```

| Category | Features |
|---|---|
| Architectures | `arch-llama` (Llama), `arch-qwen` (Qwen 2/3), `arch-qwen35` (Qwen 3.5 / Bonsai 2), `arch-gemma3` (Gemma 3), `arch-phi3` (Phi-3/4), `arch-qwen3moe` (Qwen 3 MoE), `arch-gptoss` (gpt-oss) |
| Formats | `fmt-kquant` (Q2_K..Q6_K), `fmt-ternary` (PQ2_0, PTQ1_0), `fmt-iq` (IQ4_NL, IQ4_XS), `fmt-mxfp4` (MXFP4), `fmt-legacy` (Q4_0, Q5_0, Q8_0); F32/F16/BF16 always enabled |
| Extras | `littlebit` (LittleBit sidecars), `dspark` (speculative drafter), `mtp` (multi-token prediction head), `server` (OpenAI HTTP server), `cli` (CLI binary) |

## Documentation

| | |
|---|---|
| [Product](docs/product.md) | what Peregrine is, who it is for, what it will not do |
| [Quickstart](docs/quickstart.md) | a release binary or Docker, a model, and the first request |
| [Targets](docs/targets.md) | the speed, accuracy and support numbers we hold ourselves to |
| [Benchmarks](docs/benchmark.md) | Peregrine against llama.cpp, wins and losses, with charts |
| [Roadmap](docs/roadmap.md) | the plan, stage by stage |
| [Architecture](ARCHITECTURE.md) | a map of the code |
| [Contributing](CONTRIBUTING.md) | how to build, test, benchmark and send changes |
| [Decisions](docs/adr/README.md) | why things are the way they are |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions are accepted under the same license.
