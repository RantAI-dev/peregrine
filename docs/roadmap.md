# Roadmap

Structure first, then breadth, then speed: breadth is cheap only once the structure is right, and speed
work needs the measurement harness to be trusted. Each item becomes a GitHub issue with the same ID;
estimates are in person-weeks (pw).

Every item keeps the invariants in `ARCHITECTURE.md`: identical tokens and the accuracy gate in
`docs/targets.md`.

## Stage 0: groundwork (~1.5 pw)

| ID | Item | pw |
|---|---|---|
| S0.1 | `EngineOptions`: every `PEREGRINE_*` switch parsed once (env, then CLI), printed at startup, reported in `/metrics` and benchmark JSON, passed down explicitly | 0.5 |
| S0.2 | Synthetic test models: tiny random llama, qwen3 and qwen35 (with MTP) GGUFs in every format, checked against the exact reference forward in CI | 0.5 |
| S0.3 | Forced kernel tier (`scalar` / `avx2` / `avx512`) through `EngineOptions`; Intel SDE job in CI for the AVX-512, VNNI and GFNI paths | 0.25 |
| S0.4 | Measure the `quick` profile against `release` (rebuild time, kernel speed); record the result | 0.1 |
| S0.5 | The AVX2-only host (H2) runs the full test suite on every release candidate | 0.1 |
| S0.6 | Presence, frequency and repetition penalties; unknown sampling fields rejected instead of ignored | 0.5 |

## Stage 1: structure (~5-6 pw, ADR 0005; to be discussed before starting)

| ID | Item | pw | Needs |
|---|---|---|---|
| S1.1 | `peregrine-kernels`: the kernel table per (format, ISA) chosen once at load, a scalar reference beside every kernel, all SIMD `unsafe` moved here; removes the 14 CPU checks and 9 format branches from model code | 1.5 | S0.2, S0.3 |
| S1.2 | `Architecture` interface and shared layer blocks (norm, RoPE, attention, FFN, recurrent, head) in `peregrine-core`; dense and hybrid forwards rewritten as two families in `peregrine-models`; `model.rs` / `hybrid.rs` split | 2.0 | S1.1 |
| S1.3 | One checked helper that hands workers disjoint row ranges; `unsafe` leaves the model code | 0.5 | S1.2 |
| S1.4 | One runtime: `Session` becomes a one-slot `Engine`; one drafter interface for MTP, DSpark and suffix speculation — done (ADR 0007) | 1.5 | S1.2 |
| S1.5 | Features (`arch-*`, `fmt-*`, `mtp`, `dspark`, `littlebit`, `server`), the `peregrine` facade crate with the CLI and server, `cargo hack --each-feature` in CI | 0.5 | S1.2 |
| S1.6 | `peregrine-bench` for examples and the harness; stale docs fixed | 0.25 | |
| S1.7 | `peregrine tune` and tuning profiles (CPU, version and model hash recorded; mismatch falls back to defaults) | 0.5 | S1.1, S0.1 |

Guards on every step: greedy tokens and KL identical; a before/after benchmark for each code move (cross-crate
inlining); small cross-crate helpers marked `#[inline]`.

Exit: identical tokens and KL; synthetic models pass on every kernel tier and feature set; `ARCHITECTURE.md`
matches the code.

## Stage 2: breadth (~6-8 pw)

| ID | Item | pw |
|---|---|---|
| S2.1 | Formats: Q5_K, Q4_0, then IQ4_XS / IQ4_NL, then Q2_K / Q3_K; BF16 in matrix multiplies | 1.5 |
| S2.2 | Plain Qwen 3.5 (rotation optional); check Llama 3.x and Mistral on real files | 1.0 |
| S2.3 | SentencePiece tokenizer; chat templates rendered from the model file's own template | 1.0 |
| S2.4 | Gemma 3 (sliding-window attention, its norms) and Phi-4 | 1.0 |
| S2.5 | Mixture of experts: qwen3moe, then gpt-oss (MXFP4) | 1.5 |
| S2.6 | Every format on AVX2; AVX-VNNI-256 tier; startup calibration of thread count and spin window | 1.0 |
| S2.7 | Support table generated from the kernel table and the architecture registry, checked in CI | 0.25 |
| S2.8 | ARM NEON/SDOT tier (if in scope) | 2-4 |

Each model is checked against llama.cpp on the same file before it is listed.

## Stage 3: speed and proof (~6-8 pw)

| ID | Item | pw |
|---|---|---|
| S3.1 | Benchmark harness (`benchmarks/`): one script per engine at pinned versions, the four numbers at three contexts, fraction-of-ceiling, raw JSON | 1.0 |
| S3.2 | Competitor baselines on H1, H2 and H3 | 1.0 |
| S3.3 | Task-accuracy gate against the reference engine (GSM8K, HumanEval+, MMLU subset), batch and speculation invariance on 500+ prompts | 1.0 |
| S3.4 | Faster k-quant kernels (repacked row tiles, int8 dot products) | 1.5 |
| S3.5 | Batched ternary GEMM toward the int8 compute roof; prompt path (large tiles, chunked Gated DeltaNet scan) | 1.5 |
| S3.6 | Faster PTQ1_0 for the 27B | 0.5 |
| S3.7 | Long context: split-K attention per length; q8 KV as an opt-in with its own accuracy report | 1.0 |
| S3.8 | Serving: first token at 10 users, MTP at 2-3 users, server overhead | 1.0 |
| S3.9 | Build profile A/B (`dist`: fat LTO, one codegen unit, then PGO); thread pinning and first-touch placement | 0.5 |
| S3.10 | Published report: method, raw data, wins and losses | 0.5 |

## Open decisions

1. ARM in Stage 2, or x86 only for now.
2. MoE (S2.5) in Stage 2, or later.
3. LittleBit: keep behind a feature flag, or remove.
4. H3: which quiet high-end machine produces the published numbers.
