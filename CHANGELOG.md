# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, minor versions may break things.

## [Unreleased]

### Fixed
- The prefill chunk's default now follows the worker pool. 16 is right on a pool as wide as the physical
  cores, but the engine builds `available_parallelism()` workers, which on an SMT machine is twice the
  core count. A 16-row pass is only 16 rows deep: its row-deep blocks (the rotation) give one row to 16
  workers and leave the other 8 spinning on the siblings of the cores doing the work, and every parallel
  region is short enough that its slowest worker is a large share of it. Measured on the reference
  machine at 24 workers, a forward pass carries about 20 ms that does not scale with rows, on both Bonsai
  sizes (2B and 9B), so a 512-token prompt pays it 32 times instead of 8. The default is now 64 when the
  pool is wider than the physical cores and stays 16 otherwise (`options::prefill_chunk_default_for`).
  At 24 workers, medians of four rounds: Bonsai-2B 304 prompt tok/s against 212 (+43 %), Bonsai-9B 90.7
  against 79.4 (+14 %), Llama-3.2-1B 225 against 189 (+19 %); decode is unchanged on all three. At 12
  workers 16 keeps winning (257 against 248, 69.0 against 64.9, 180 against 168), and the default stays
  16 there. `PEREGRINE_PREFILL_CHUNK` still overrides either.

### Added
- A tunable prefill chunk (`PEREGRINE_PREFILL_CHUNK`, default 16 on a pool as wide as the physical
  cores and 64 when the pool is wider; see the fix below): the prompt tokens one weight pass
  carries. The engine prefills a prompt in pieces of this size, and the piece is how the prompt is
  *read*, not what it produces: the same prompt, read in pieces of 16, 32, 64 or 128 tokens, gives the
  same KV cache, the same logits and the same tokens (`crates/peregrine-models/tests/prefill_chunk.rs`).
  The default comes from an A/B on the reference machine (12 threads, four rotating rounds): Llama-3.2-1B
  165.9 prompt tok/s against 154.3 at the old 64 (+7.5 %), Ternary-Bonsai-2-2B 239.2 against 227.0
  (+5.4 %), Qwen2.5-0.5B 257.3 against 270.7 (-5.0 %). The tenth user's first token does not get later
  (19.8 s against 21.8 s at 64). A smaller chunk reads the weights more often (18.9 GB against 4.9 GB for
  a 512-token prompt on the 1B model) and keeps the activations in cache; the prefill is compute-bound at
  that width, so the traffic is free.
- The `gpt-oss` model family (S2.5 step 5): OpenAI's mixture of experts — attention sinks (a learned
  per-head logit in the softmax), alternating sliding-window and full attention, biases on the attention
  output, the router and every expert projection, YaRN rope scaling, the router's softmax over the
  *selected* experts, and OpenAI's `swiglu_oai` activation. The `gpt-4o` (o200k) pre-tokenizer is
  transcribed from llama.cpp v0.9.0's regex. The synthetic F16 and MXFP4 files give the same greedy
  tokens as llama.cpp; the check against a real gpt-oss file is pending.
- OpenAI's Harmony chat template, which gpt-oss's own Jinja template renders (`--chat-template harmony`):
  the fixed system block, the developer block for the caller's system message, user and assistant
  turns with the `<|channel|>final<|message|>` header, and the generation prompt. Rendered strings are
  pinned against llama.cpp v0.9.0; the synthetic gpt-oss file now carries the model's Jinja template
  and is detected as `harmony`.
- The server splits gpt-oss's Harmony reply into `reasoning_content` (the `analysis` channel) and
  `content` (the `final` channel), non-streamed and as SSE deltas, with the control tokens consumed.
  A `commentary` preamble is content and a `<|constrain|>` header is accepted. `reasoning_format`
  (`none`, `auto`, `deepseek`, `deepseek-legacy`) and `--reasoning-format`, and `reasoning_effort:
  "none"`, map as llama-server maps them: `none` keeps the analysis text in `content`, `auto`/`deepseek`
  split it, and `deepseek-legacy` also mirrors reasoning into streaming content. Qwen3's `<think>` span
  obeys the same switch. Earlier assistant turns replay only their final channel, matching gpt-oss's
  Jinja, and a Harmony `Session` re-renders its transcript so the incremental path cannot retain an
  analysis the full render drops. The reasoning-budget forcing stays Qwen3-only: Harmony has no
  `</think>` to close.
- Support for plain Qwen 3.5 files (S2.2c): a `qwen35` file with no `prism.hadamard.*` metadata is loaded
  without a rotation, and its ordinary quantised matmuls (Q8_0, F16, ...) run through the dense kernels
  instead of the ternary ones. The synthetic plain file gives the same greedy tokens as llama.cpp v0.9.0;
  the check against a real Qwen 3.5 file is pending.
- Support for GGML's Q2_K and Q3_K weight formats (AVX2 single-row and batched kernels, scalar reference).
- Support for GGML's MXFP4 weight format (AVX2 single-row and batched kernels, scalar reference).
- Support for the Gemma 3 model family (S2.4a): interleaved sliding-window (local) and global layers, four RMSNorms per layer (pre/post attention, pre/post FFN), per-head q/k RMSNorms, GeGLU activation, embedding scaled by sqrt(d_model), tied output head, sliding-window attention restriction, and distinct RoPE frequency bases for local vs global layers matching llama.cpp v0.9.0.
- An opt-in f32 KV cache (`KvPrec::F32`, `PEREGRINE_KV=f32`, `peregrine serve --kv f32`, `kld --kv f32`), for a
  like-for-like comparison against an engine whose attention reads f16 operands; the f16 cache stays the default.
- `peregrine-bench` example `kld`: KL divergence and top-1 agreement against llama.cpp on real text (`docs/dev/accuracy.md`); first results in `docs/targets.md`.
- `peregrine bench`, `peregrine tune` and tuning profiles (S1.7).
- The `qwen3moe` mixture-of-experts family (S2.5 step 3): the synthetic models load, run, and give the
  same tokens alone and batched; the check against llama.cpp on a real file is step 4.
- Support for GGML's IQ4_NL and IQ4_XS weight formats (AVX2 single-row and batched kernels, scalar reference).
- Support for GGML's Q5_K weight format (AVX2 single-row and batched kernels, scalar reference).
- Support for Gemma, Mistral (v1, v7), and Llama 2 chat templates with exact llama.cpp parity (S2.3b).
- Support for the pre-update Mistral v0.2 chat template (no system branch), and `--chat-template <name>`
  for a model file that declares no chat template (S2.3c).
- Support for SentencePiece (llama) tokenizers transcribed from llama.cpp (S2.3a).
- Support for GGML's Q4_0 weight format (AVX2 and scalar kernels).
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
- The `peregrine` facade crate and command: `peregrine serve`, `peregrine inspect <model>`, `peregrine tier`.
- Synthetic test models (llama, qwen3 in six formats; qwen35 hybrid in PQ2_0 / PTQ1_0) generated at test time,
  so CI runs every model code path and format without model files; the qwen35 hybrid also has an MTP (NextN)
  variant that runs lossless speculation with it on and off.
- Synthetic coverage for Qwen2 attention biases, Llama 3.x and linear rope scaling, engine batching (shared
  prefixes, mixed prefill/decode, mapped and per-row caches), LittleBit sidecars and accepted MTP drafts.
- Synthetic coverage of what the session and a one-slot engine must agree on (ADR 0007): the same prompt
  through `Session` and a one-slot `Engine`, the taps a DSpark drafter needs, a second user arriving
  mid-speculation, and the prefix store with a drafter attached.
- `peregrine-core`: the `Drafter` interface (ADR 0007) — one way of proposing tokens, told what the target
  committed and what was rolled back — with wrappers over today's drafters (the suffix index, DeepSeek's
  DSpark head, the model's own MTP block). Unstable until 1.0.
- `peregrine-core`: one verification step in the engine — the slot's drafter proposes, the engine prices the
  proposal against the verify-cost model and verifies it against the target's own samples. MTP speculation
  runs through it, so the engine no longer knows what a draft is (ADR 0007, S1.4 step 3).
- `peregrine-core`: the engine speculates with the suffix index and with DeepSeek's DSpark head, through the
  same verification step the MTP block goes through. `SlotCfg` names the drafter a slot asks for (none, MTP,
  suffix, DSpark with its directory); the per-slot drafter is built when the slot is admitted, and the engine
  prices the proposal, cuts the accepted run where the turn ends, and parks the DSpark head's drafting when a
  round does not pay for itself (ADR 0007, S1.4 steps 4-5).
- `SlotCfg`: the sampling `turn` (mixed into the position key) and the KV rotation choice, so a conversation
  can be given the same settings it uses today.
- Sampling penalties (`presence_penalty`, `frequency_penalty`, `repeat_penalty`, `repeat_last_n`, llama.cpp
  semantics); sampling fields the engine does not implement are refused with a 400 instead of ignored.
- `PEREGRINE_ISA` caps the kernel tier (`scalar`, `avx2`, `avx512`); CI tests the lower tiers.
- Project setup: CI on x86-64 / aarch64 Linux and macOS, MSRV and supply-chain checks, issue forms,
  governance, security policy, product, targets, roadmap, architecture, coding and structure rules, ADRs.

### Changed
- The startup calibration is off by default: `PEREGRINE_CALIBRATE=1` turns it on (S2.6 decision 6).
  A wrong thread count costs far more than the tuned default it replaces, and the calibration is only
  switched on by default once it has passed A/Bs on two machines.
- `Session` is a conversation layer over a one-slot `Engine` (ADR 0007, S1.4): the chat history, the template,
  the prompt, the reuse of the positions the previous turn computed and the context shifting stay in the
  session; the generation, the speculation and the verification of the drafts are the engine's. Its public API
  is unchanged.
- `peregrine bench` measures the engine. The benchmark used to walk a prefill and a decode loop of its own,
  beside the runtime the server runs; it now admits a slot and lets the engine do the work, so the numbers it
  reports are the numbers the engine charges a request. The bytes a token moves are the same, and the JSON
  keys are unchanged.
- The GeGLU kernel's `tanh` runs on the CPU's FMA instruction where there is one: the scalar arithmetic is
  compiled a second time inside a `#[target_feature(enable = "fma")]` function, so the `mul_add`s become
  `vfmadd` instead of a call to the C library's `fmaf` (8.3x on the kernel, bit for bit the same result).
- F32/F16/BF16 matrices multiply f32 activations instead of int8-quantised ones (output change for those formats).

### Removed
- The last of the old runtime and the old speculation paths (ADR 0007, S1.4): `Session::probe_verify_curve`
  (the engine's verify-cost model measures those widths now, on the engine's own forward passes),
  `dspark::Drafter::draft`, `SuffixIndex::propose`, `Engine::slot_cost`, `DrafterChoice::Mtp` (an alias for
  `DrafterChoice::None`) and `Needs::history` (the engine never read it). The interfaces are unstable until 1.0.

### Fixed
- The startup calibration no longer stops its decode-thread sweep at the first candidate that fails to beat
  the previous best by 5%: one noisy high early sample froze it at one or two threads and the 9B decoded
  at 5.6 tok/s instead of 23.6. Every candidate is measured now, the default (the physical cores) stands
  unless a smaller count is at least as good (median bandwidth ≥ 98% of the default's, both measurements
  quiet), and the count never goes below half the physical cores.
- The A1 dequantisation oracle is re-cut from one pinned SmolLM2-360M-Instruct Q4_K_M file (size and
  FNV-1a checked at test time, SHA-256 in the failure message), so another upload fails loudly instead
  of looking like a kernel bug.
