# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, minor versions may break things.

## [Unreleased]

### Added
- `peregrine-core`: CPU inference engine for GGUF models.
  - Architectures: Llama, Qwen 2, Qwen 3, Qwen 3.5 / Bonsai 2 (Gated DeltaNet hybrid).
  - Weight formats: F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q6_K, and the ternary PQ2_0 / PTQ1_0.
  - x86-64 kernels for AVX2 and AVX-512 (VNNI, GFNI, VBMI), chosen at run time; scalar code elsewhere.
  - Batching engine: many users per weight pass, chunked prefill, a shared prefix store, position-keyed
    sampling (identical tokens alone or batched).
  - Lossless MTP speculation for a single user; DSpark and prompt-lookup speculation.
  - Bytes-moved telemetry against the machine's bandwidth ceiling.
- `peregrine-server`: OpenAI-compatible chat API (streaming, `n > 1`, `logprobs`, `/metrics`).
- Golden-output tests per kernel tier.
- `peregrine-bench`: the micro-benchmarks and experiments, moved out of the engine library.
- Synthetic test models (llama, qwen3 in six formats; qwen35 hybrid in PQ2_0 / PTQ1_0) generated at test time,
  so CI runs every model code path and format without model files; the qwen35 hybrid also has an MTP (NextN)
  variant that runs lossless speculation with it on and off.
- Sampling penalties (`presence_penalty`, `frequency_penalty`, `repeat_penalty`, `repeat_last_n`, llama.cpp
  semantics); sampling fields the engine does not implement are refused with a 400 instead of ignored.
- `PEREGRINE_ISA` caps the kernel tier (`scalar`, `avx2`, `avx512`); CI tests the lower tiers.
- Project setup: CI on x86-64 / aarch64 Linux and macOS, MSRV and supply-chain checks, issue forms,
  governance, security policy, product, targets, roadmap, architecture, coding and structure rules, ADRs.
