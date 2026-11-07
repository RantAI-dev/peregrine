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

> **Status: early.** The engine works and is measured, but the API, CLI and crate layout will change as the [roadmap](docs/roadmap.md) lands.

## Why

- **Measured, not claimed.** Every result is reported as a fraction of the host's roofline, with the exact
  command, commit and options that produced it. Losses are published along with wins.
- **Lossless by default.** Speculation, batching and every kernel produce the same tokens as the exact
  reference path. Anything lossy is off by default and ships with its own accuracy report.
- **Many users, one weight pass.** A batching scheduler with chunked prefill, a shared prefix cache and
  position-keyed sampling: a user gets identical text alone or alongside others.
- **Owned hot path.** Kernels, thread pool, cache, sampler, tokenizer and scheduler are this project's
  code, so their speed and correctness are under its control.

## What runs today

<!-- support:begin -->
| | Supported now |
|---|---|
| Architectures | Llama, Qwen 2, Qwen 3, Qwen 3.5 / Bonsai 2 (Gated DeltaNet hybrid, MTP), Gemma 3, Qwen 3 MoE, gpt-oss (synthetic check; real file pending), Phi-3 / Phi-4 (synthetic check; real file pending) |
| Weight formats | F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4, PQ2_0, PTQ1_0 |
| CPUs | x86-64 AVX2, AVX-VNNI (256-bit), AVX-512 (VNNI, GFNI, VBMI); others through scalar code |
| Tokenizers | GPT-2 byte-level BPE, SentencePiece (llama) |
| Chat templates | read from the model file and rendered exactly as llama.cpp renders them: ChatML, Llama 3, Qwen 3, Gemma, Mistral (v0, v1, v7), Llama 2, OpenAI Harmony (gpt-oss) — or named with `--chat-template` for a file that declares none |
<!-- support:end -->

### On the roadmap

- **Architectures**: Phi-4
- **CPUs**: AVX-VNNI-256, ARM NEON/SDOT
- **Chat templates**: more of llama.cpp's built-ins

Measured on a Ryzen AI 9 HX 370 (WSL2), Bonsai 2 9B: **23.5 tok/s single user, 28.7 with lossless MTP
speculation**, about 75 tok/s total at 10 users. The full table, method and competitors are in
[docs/targets.md](docs/targets.md).

Checked against llama.cpp v0.9.0 on the same file, for **Llama 3.2 1B Instruct** and **Qwen 2.5 0.5B
Instruct** (Q4_K_M): the tokenizer produces the same token ids, greedy decoding produces the same token
ids except where the top two logits are closer than the two implementations' own rounding difference,
and the chat path returns the same prompt token count and the same answer.

The same holds for the two SentencePiece families, **Llama 2 7B Chat** and **Mistral 7B Instruct
v0.2** (Q4_K_M): the tokenizer produces the same ids as llama.cpp's `llama-tokenize`, greedy decoding
produces the same ids except at one near-tie below the implementations' own logit spread, and the chat
path returns a byte-identical rendered prompt, the same prompt token count and the same answer.

The same holds for **Gemma 3 1B Instruct** (Q4_K_M): the tokenizer produces the same ids as
llama.cpp's `llama-tokenize` (22/22 strings), greedy decoding produces bit-identical token ids
across all 32 steps on all three prompts (including beyond the 512-token sliding window), and the
chat path returns a byte-identical rendered prompt, the same prompt token count and the same answer.

The same holds for **Qwen 3 MoE (Qwen3-30B-A3B)** (Q2_K): the tokenizer produces the same ids as
llama.cpp's `llama-tokenize` (22/22 strings), greedy decoding matches bit for bit on prompts 2 and 3
(32/32 steps) and through 26 tokens on prompt 1 (first divergence at a near-tie of 0.17 logit difference),
and the chat path returns matching answers with reasoning tokens.

The same holds for plain **Qwen 3.5 (Qwen3.5-2B)** (Q4_K_M): the tokenizer produces the same ids as
llama.cpp's `llama-tokenize` (22/22 strings), greedy decoding matches bit for bit on prompts 1, 2, and 4
(32/32 steps, including across a 602-token prompt) and through 27 tokens on prompt 3 (first divergence
at a near-tie of 0.045 logit difference), and the chat path returns matching answers with reasoning tokens.

## Quick start

```sh
git clone https://github.com/RantAI-dev/peregrine && cd peregrine
cargo build --release
./target/release/peregrine inspect path/to/model.gguf      # what it is, and whether it runs
./target/release/peregrine bench -m path/to/model.gguf        # measure prefill & decode tok/s and bytes/token
./target/release/peregrine tune -m path/to/model.gguf         # tune execution knobs for host CPU and model
./target/release/peregrine serve -m path/to/model.gguf --port 8201
curl -s localhost:8201/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

The server speaks the OpenAI chat API (`/v1/chat/completions`, streamed or not), plus `/v1/models`,
`/health` and `/metrics`.

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
| Architectures | `arch-llama` (Llama), `arch-qwen` (Qwen 2/3), `arch-qwen35` (Qwen 3.5 hybrid) |
| Formats | `fmt-kquant` (Q2_K..Q6_K), `fmt-ternary` (PQ2_0, PTQ1_0), `fmt-iq` (IQ4_NL, IQ4_XS), `fmt-legacy` (Q4_0, Q5_0, Q8_0); F32/F16/BF16 always enabled |
| Extras | `littlebit` (LittleBit sidecars), `dspark` (speculative drafter), `mtp` (multi-token prediction head), `server` (OpenAI HTTP server), `cli` (CLI binary) |

## Documentation

| | |
|---|---|
| [Product](docs/product.md) | what Peregrine is, who it is for, what it will not do |
| [Targets](docs/targets.md) | the speed, accuracy and support numbers we hold ourselves to |
| [Roadmap](docs/roadmap.md) | the plan, stage by stage |
| [Architecture](ARCHITECTURE.md) | a map of the code |
| [Contributing](CONTRIBUTING.md) | how to build, test, benchmark and send changes |
| [Decisions](docs/adr/README.md) | why things are the way they are |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions are accepted under the same license.
