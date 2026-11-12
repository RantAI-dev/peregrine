# Roadmap

Structure first, then breadth, then speed: breadth is cheap only once the structure is right, and speed
work needs the measurement harness to be trusted. Each item becomes a GitHub issue with the same ID;
estimates are in person-weeks (pw).

Every item keeps the invariants in `ARCHITECTURE.md`: identical tokens and the accuracy gate in
`docs/targets.md`.

## Stage 0: groundwork (~1.5 pw)

| ID | Item | pw |
|---|---|---|
| S0.1 | `EngineOptions`: every `PEREGRINE_*` switch parsed once (env, then CLI), printed at startup, reported in `/metrics` and benchmark JSON, passed down explicitly — done | 0.5 |
| S0.2 | Synthetic test models: tiny random llama, qwen3 and qwen35 (with MTP) GGUFs in every format, checked against the exact reference forward in CI — done | 0.5 |
| S0.3 | Forced kernel tier (`scalar` / `avx2` / `avx512`) through `EngineOptions`; Intel SDE job in CI for the AVX-512, VNNI and GFNI paths — done | 0.25 |
| S0.4 | Measure the `quick` profile against `release` (rebuild time, kernel speed); record the result — done | 0.1 |
| S0.5 | The AVX2-only host (H2) runs the full test suite on every release candidate — done: the release step and the gate are written down in `docs/dev/releasing.md` | 0.1 |
| S0.6 | Presence, frequency and repetition penalties; unknown sampling fields rejected instead of ignored — done | 0.5 |

## Stage 1: structure (~5-6 pw, ADR 0005; to be discussed before starting)

| ID | Item | pw | Needs |
|---|---|---|---|
| S1.1 | `peregrine-kernels`: the kernel table per (format, ISA) chosen once at load, a scalar reference beside every kernel, all SIMD `unsafe` moved here; removes the 14 CPU checks and 9 format branches from model code — done | 1.5 | S0.2, S0.3 |
| S1.2 | `Architecture` interface and shared layer blocks (norm, RoPE, attention, FFN, recurrent, head) in `peregrine-core`; dense and hybrid forwards rewritten as two families in `peregrine-models`; `model.rs` / `hybrid.rs` split — done (ADR 0006) | 2.0 | S1.1 |
| S1.3 | One checked helper that hands workers disjoint row ranges; `unsafe` leaves the model code — done | 0.5 | S1.2 |
| S1.4 | One runtime: `Session` becomes a one-slot `Engine`; one drafter interface for MTP, DSpark and suffix speculation — done (ADR 0007) | 1.5 | S1.2 |
| S1.5 | Features (`arch-*`, `fmt-*`, `mtp`, `dspark`, `littlebit`, `server`), the `peregrine` facade crate with the CLI and server, `cargo hack --each-feature` in CI — done | 0.5 | S1.2 |
| S1.6 | `peregrine-bench` for examples and the harness; stale docs fixed — done | 0.25 | |
| S1.7 | `peregrine tune` and tuning profiles (CPU, version and model hash recorded; mismatch falls back to defaults) — done | 0.5 | S1.1, S0.1 |

Guards on every step: greedy tokens and KL identical; a before/after benchmark for each code move (cross-crate
inlining); small cross-crate helpers marked `#[inline]`.

Exit: identical tokens and KL; synthetic models pass on every kernel tier and feature set; `ARCHITECTURE.md`
matches the code.

## Stage 2: breadth (~6-8 pw)

| ID | Item | pw |
|---|---|---|
| S2.1 | Formats: Q5_K, Q4_0, then IQ4_XS / IQ4_NL, then Q2_K / Q3_K; BF16 in matrix multiplies — done (also MXFP4) | 1.5 |
| S2.2 | Plain Qwen 3.5 (rotation optional); check Llama 3.x and Mistral on real files — done | 1.0 |
| S2.3 | SentencePiece tokenizer; chat templates rendered from the model file's own template — done | 1.0 |
| S2.4 | Gemma 3 (sliding-window attention, its norms) and Phi-4 — done | 1.0 |
| S2.5 | Mixture of experts: qwen3moe, then gpt-oss (MXFP4) — qwen3moe done; gpt-oss synthetic, its harmony template and the harmony reasoning split done, real-file check open | 1.5 |
| S2.6 | Every format on AVX2; AVX-VNNI-256 tier; startup calibration of thread count and spin window — every format has an AVX2 kernel; the VNNI-256 tier is in, its kernels on hold (#77); calibration open (#80) | 1.0 |
| S2.7 | Support table generated from the kernel table and the architecture registry, checked in CI — done | 0.25 |
| S2.8 | ARM NEON/SDOT/i8mm tier — done (`aarch64-neon`, `aarch64-neon-dotprod`, `aarch64-neon-i8mm`, plus Bonsai 2 on NEON, #127–#132, #137, #141) | 2-4 |

Each model is checked against llama.cpp on the same file before it is listed.

## Stage 3: speed and proof (~6-8 pw)

| ID | Item | pw |
|---|---|---|
| S3.1 | Benchmark harness (`benchmarks/`): one script per engine at pinned versions, the four numbers at three contexts, fraction-of-ceiling, raw JSON — done | 1.0 |
| S3.2 | Competitor baselines on H1, H2 and H3 — desktop (H1), bluefin (H2), and Spark ARM (H3) baselines done, incl. the Bonsai 2/PrismML comparisons (#116, #134, #141) and the 2026-11-09 re-measurement — done | 1.0 |
| S3.3 | Task-accuracy gate against the reference engine (GSM8K, HumanEval+, MMLU subset), batch and speculation invariance on 500+ prompts — KL against llama.cpp per model (`docs/targets.md`), task-accuracy gate and 520-prompt batch/speculation invariance passed on all three hosts (bluefin AVX2 #144, desktop AVX-512 #145, Spark ARM #142; `docs/dev/accuracy.md`, `benchmarks/accuracy/`) — done | 1.0 |
| S3.4 | Faster k-quant kernels (repacked row tiles, int8 dot products) — repacked store (#86), AVX2 Q4_K (#88), Q5_0 (#103–#105), AVX-512 Q4_K/Q5_0 (#108, #111), wide Q6_K (#118), wide Q5_K (#133), Q8_0 tiles (#122), Q5_K/Q6_K x86 tiles (S4.9i) — done | 1.5 |
| S3.5 | Batched ternary GEMM toward the int8 compute roof; prompt path (large tiles, chunked Gated DeltaNet scan) — prefill chunk tuning (`PEREGRINE_PREFILL_CHUNK`, #87, #92) and batched ternary GEMM (#138; chunked scan design in `docs/design/s3.5-delta-scan.md`) — done | 1.5 |
| S3.6 | Faster PTQ1_0 for the 27B | 0.5 |
| S3.7 | Long context: split-K attention per length; q8 KV as an opt-in with its own accuracy report — attention split into more tasks at short context (#94), F16-direct register-block attention (#109), query×position prefill tiles (#113), the F16 K/V prefill path (#115) and the AVX-512 attention mix (#121) done; q8 KV open | 1.0 |
| S3.8 | Serving: first token at 10 users, MTP at 2-3 users, server overhead — multi-user attention region (#114) and adaptive spin-wait on busy hosts (#125) — done | 1.0 |
| S3.9 | Build profile A/B (`dist`: fat LTO, one codegen unit, then PGO); thread pinning and first-touch placement — `dist` profile A/B (`docs/dev/build-profiles.md`) and `PEREGRINE_AFFINITY` thread pinning (#139) — done | 0.5 |
| S3.10 | Published report: method, raw data, wins and losses — published benchmark and accuracy report (`docs/benchmark.md`, `docs/dev/accuracy.md`, `benchmarks/results/`) — done | 0.5 |

## Stage 4: embeddings and decision models

Design: [`docs/design/s4-embeddings-decisions.md`](design/s4-embeddings-decisions.md); the decision
models are specified in [`docs/reference/systemone.md`](reference/systemone.md). Embedding models
(`/v1/embeddings`) and decision models: native System One models on `/v1/systemone` (typed `choice`,
`score` and `noul` questions about a state), rerankers and classifier heads, each one forward pass
with no decode.

| ID | Item | pw |
|---|---|---|
| S4.1 | Pooled outputs on causal models; `/v1/embeddings`, `peregrine embed`; Qwen3-Embedding against llama.cpp — done (#158) | 0.5 |
| S4.2 | Benchmark harness modes for embeddings, rerank and System One; competitor baselines (llama.cpp, text-embeddings-inference, fastembed, OpenVINO) on H1 and H2 | 1.0 |
| S4.3 | System One, part 1: `/v1/systemone`, the answer math, the label-logit types on Qwen 3.5 (OpenJev, Lev, Nimble, pplx-decider, H2O Lightning) | 1.5 |
| S4.4 | Bidirectional forward; `gemma-embedding` (EmbeddingGemma) — `Architecture::forward_texts` over whole texts with no KV cache, symmetric sliding-window attention, mean/cls pooling and the `dense_2`/`dense_3` head (#162); EmbeddingGemma 300M checked against llama.cpp v0.9.0 — done | 1.0 |
| S4.5 | WordPiece and SentencePiece-unigram tokenizers — done (#157) | 0.5 |
| S4.6 | `bert` family and the classifier head: MiniLM, bge, e5, bge-m3, bge-reranker; `/v1/classify` — part a, the family (post-norm blocks, absolute positions, token types, biases everywhere, LayerNorm, the exact GELU, cls/mean pooling; WordPiece and the XLM-R unigram vocabulary) with the defined `erf`/`gelu_erf`/`layer_norm` kernels: MiniLM-L6-v2, bge-small-en-v1.5 and bge-m3 checked against llama.cpp v0.9.0 and the fp32 PyTorch reference — done; part b, the head: `cls`/`cls.output` with the file's activation and labels, `Want::Head`, `/v1/rerank` (four routes) and `/v1/classify`, `peregrine rerank`, bge-reranker-v2-m3 and bge-reranker-base checked against llama.cpp v0.9.0 (pair ids identical, document order identical, scores within the measured Q8_0 bound) — done | 1.5 |
| S4.7 | `modern-bert` and labelled classifier heads (guards, routers) — part a, the family: pre-norm blocks with weight-only LayerNorms, fused q/k/v, RoPE with separate local/global bases, alternating 128-token local and global attention, the gated feed-forward (erf GeGLU, tanh GeGLU or SwiGLU), the `modern-bert` GPT-2 pre-tokenizer and cls/mean pooling; gte-modernbert-base and Granite Embedding R2 F16 and Q8_0 checked against llama.cpp v0.9.0, on 200 texts of 16–6976 tokens. Part b, the head: `cls`→erf GELU→`cls.norm`→`cls.output` over the mean (or row 0), served on `/v1/rerank` and `/v1/classify`, with the `[bos] q [eos] [sep] d [eos]` pair; gte-reranker-modernbert-base F16 and Q8_0 on 100 pairs up to 445 tokens, ids identical and F16 order identical — done | 1.0 |
| S4.8 | `nomic-bert`, `jina-bert-v2/v3` — part a, the `nomic-bert` family: token type 0, an embedding LayerNorm with a bias, post-norm blocks with a fused q/k/v, NeoX RoPE at base 1000, the SwiGLU feed-forward, mean pooling and no final norm; nomic-embed-text v1.5 F16 and Q8_0 checked against llama.cpp v0.9.0 on 200 texts of 20–2040 tokens (ids identical; F16 0.9999995 min, Q8_0 0.99830 mean vs the F16 reference, llama.cpp's own Q8_0 0.99830) — done; part b, `nomic-bert-moe` (nomic-embed-text-v2-moe) and `jina-bert-v2/v3` | 1.0 |
| S4.9 | State of the art for embeddings and decision models against every engine (OpenVINO, ONNX Runtime, llama.cpp): the plan and its steps S4.9a-l are in [`docs/design/s4.9-sota.md`](design/s4.9-sota.md): token-budget passes with no padding, the batch-invariant float GEMM, the encoder attention strip, state forking for decision models, threading for small work, and the proof run on three machines | 6.0 |
| S4.10 | Several models in one server: `-m` repeated, routed by the request's `model`, one engine per model and one weight pass at a time for the process | 1.0 |
| S4.11 | System One, part 2: Kev (pointer head) and vLLM D3 (non-causal Qwen 3.5) | 1.0 |
| S4.12 | System One, part 3: Laya and Julia-1 on `modern-bert` | 0.5 |
| S4.13 | Generative rerankers (Qwen3-Reranker) on `/v1/rerank` | 0.5 |
| S4.14 | `mistral3` embedders: Nemotron 3.5 Embed 1B and Nemotron 3 Embed 1B (text input), with the Tekken tokenizer | 1.0 |

Each model is checked against llama.cpp on the same file before it is listed; a text's output is
bit-identical alone and in any batch.

Ideas that are not scheduled yet are kept in the [backlog](backlog.md).

## Stage 5: prompt speed on AVX-512

The ik_llama.cpp fork processes prompts about twice as fast as Peregrine on the AVX-512 desktop
([benchmark.md](benchmark.md)). The study and the list of what to build are in
[`docs/design/s5-prefill-avx512.md`](design/s5-prefill-avx512.md).

| ID | Item | pw |
|---|---|---|
| S5.1 | The k-quant prefill kernels on AVX-512: the Q4_K and Q5_0 epilogue per group of rows, a 512-bit Q6_K tile, a write-out that makes a wide pass pay | 2.0 |
| S5.1a | A wide prefill pass when no slot is decoding: the chunk is chosen per pass, 64 wide on both measured tiers | 0.3 |
| S5.1b | The Q4_K and Q6_K kernels at ik_llama.cpp's instruction mix: measured, both are at the core's byte-dot roof (77–81 and 58–61 of 106–117 GMAC/s) and the fork is not ahead | 1.0 |
| S5.2a | Is ik_llama.cpp's fast wide pass exact? No: it requantises each 256-weight row-block onto one 8-bit grid (worst case 0.394 % of the block's largest weight), its `-ub 512` logits differ from `-ub 16` by 0.0011 mean KLD and 97.9 % top-1, and the same conversion done exactly — widen, keep the per-32 scale and the min — measures +0.2 % at 128 rows and +1 % at 512, so nothing was ported. Q5_0's gap to Q4_K is its 32-weight fold; two exact variants measured slower and were withdrawn | 1.5 |
| S5.2b | Where a prompt pass spends its time: attention is 8–10 % of a 512-token pass and 35–45 % of a 4096 one (the value mix twice the score stage), the matmul is the stage that stops scaling (7.1× of 12 against attention's 8–9×), and 6–10 % of a pass is on the calling thread. The prefill value mix now mixes two rows per V pass (+5.7 % at 4096, twelve threads, both models; the bits do not move) and runs at 16 % of the f32 roof instead of 13 %. A 512-bit score row stays unbuilt (a 16-lane reduction measured 30–70 % slower) and the 24-thread collapse is the pool, not a kernel | 1.5 |
| S5.3 | The same questions on AVX2 and NEON; ik_llama.cpp measured on bluefin and ARM | 1.0 |
| S5.4 | An opt-in fast prompt mode (`--fast-prompt`, off by default): prompt passes read Q4_K and Q6_K weights re-rounded to 8 bits per 256-weight block, the trade ik_llama.cpp makes by default; decode stays exact; ships with an accuracy report. Design: [`s5.4-fast-prompt.md`](design/s5.4-fast-prompt.md) | 3.0 |

## Stage 6: tool calling

OpenAI's tool-calling shapes on `/v1/chat/completions`, for the models trained to call tools. The
prompt is the one llama.cpp builds from the model's template, byte for byte. Design:
[`docs/design/s6-tool-calling.md`](design/s6-tool-calling.md).

| ID | Item | pw |
|---|---|---|
| S6.1 | Design | 0.2 |
| S6.2 | The tool module: the `hermes` format (Qwen 2.5) and the `qwen-xml` format (Qwen 3.5 / Bonsai 2), renderers and parsers, fixtures from llama.cpp — done for both in the module; and the `qwen3` format (Qwen 3: the same calls, thinking opened by the model, four published wordings of the template with their two differing rules) (S6.2c) | 1.0 |
| S6.3 | The endpoint: `tools`, `tool_choice`, `tool_calls` in requests and responses, streaming — done for the `hermes` format (Qwen 2.5): 15 of 16 replies identical to llama.cpp's on the same file; and for the `qwen-xml` format (Qwen 3.5, Bonsai 2), thinking on or off: every call identical to llama.cpp's on a Qwen 3.5 file (S6.3b). Chat without tools on these files is rendered by the same renderer in `serve`, and `peregrine run` takes the same prompt from the `qwen35` chat template (S6.3c). Qwen 3 is served too: on a 1.7B file the right tool is called on 11 of 12 requests with thinking off (llama.cpp 11) and 12 of 12 with it on (llama.cpp 11) | 1.0 |
| S6.4 | The `llama3` format (Llama 3.1 / 3.2) | 0.5 |
| S6.5 | The `harmony` format (gpt-oss) | 0.5 |
| S6.6 | Constrained sampling for calls: a JSON-schema matcher and a token mask; `tool_choice: "required"` and named choices | 2.0 |
| S6.7 | The `generic` format for templates without a tool section | 0.5 |

## Stage 7: macOS and Windows

Peregrine on the two other desktop systems, to the bar Linux is held to: CI on every PR, a release
archive, an installer, `upgrade`, the server's lifetime commands, goldens. The design is
[`docs/design/s7-platforms.md`](design/s7-platforms.md).

| ID | Item | pw |
|---|---|---|
| S7.1 | macOS on Apple Silicon: release job with a dry run, installer, `upgrade`, performance-core default, synthetic goldens, quickstart | 1.0 |
| S7.2 | Windows x86-64 builds and passes the suite in CI | 1.5 |
| S7.3 | Windows lifetime commands, `upgrade`, `install.ps1`, release job | 1.5 |
| S7.4 | Windows goldens, docs for both systems, a native Windows measurement on the desktop | 1.0 |
| S7.5 | A release that ships all four archives | 0.3 |

## Open decisions

1. ARM in Stage 2, or x86 only for now.
2. MoE (S2.5) in Stage 2, or later.
3. LittleBit: keep behind a feature flag, or remove.
4. H3: which quiet high-end machine produces the published numbers.
