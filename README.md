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

| | Supported now | On the roadmap |
|---|---|---|
| Architectures | Llama, Qwen 2, Qwen 3, Qwen 3.5 / Bonsai 2 (Gated DeltaNet hybrid, MTP) | Gemma 3, Phi-4, MoE (qwen3moe, gpt-oss) |
| Weight formats | F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, PQ2_0, PTQ1_0 | MXFP4 |
| CPUs | x86-64 AVX2, AVX-512 (VNNI, GFNI, VBMI); others through scalar code | AVX-VNNI-256, ARM NEON/SDOT |
| Tokenizers | GPT-2 byte-level BPE, SentencePiece (llama) | |
| Chat templates | read from the model file and rendered exactly as llama.cpp renders them: ChatML, Llama 3, Qwen 3, Gemma, Mistral (v0, v1, v7), Llama 2 — or named with `--chat-template` for a file that declares none | more of llama.cpp's built-ins |

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

## Quick start

```sh
git clone https://github.com/RantAI-dev/peregrine && cd peregrine
cargo build --release
./target/release/peregrine inspect path/to/model.gguf      # what it is, and whether it runs
./target/release/peregrine serve -m path/to/model.gguf --port 8201
curl -s localhost:8201/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

The server speaks the OpenAI chat API (`/v1/chat/completions`, streamed or not), plus `/v1/models`,
`/health` and `/metrics`.

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
