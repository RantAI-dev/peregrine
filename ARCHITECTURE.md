# Architecture

A map of the code for someone about to change it. It describes what is true **today** and, where the
roadmap changes it, says so. If this file and the code disagree, the code wins and this file gets a fix.

## Bird's eye view

```
            HTTP (OpenAI chat API)                 peregrine-server
                     │
                     ▼
   ┌──────────── serve::Engine ─────────────┐      peregrine-core
   │ slots · scheduler · chunked prefill    │
   │ prefix store · MTP speculation         │
   └──────────────────┬─────────────────────┘
                      │ one forward pass for every active row
                      ▼
   model::Model ── arch::Architecture ── peregrine-models (dense, qwen35, MTP)
                      │
     ┌────────────────┼──────────────────┬──────────────┐
     ▼                ▼                  ▼              ▼
  kernels         attention          cache (kv)      sampling
  simd/tern/      attn.rs            kv.rs,          sample.rs
  quant/vmath                        prefix.rs
     │
     ▼
  pool.rs: one persistent thread pool, spin-then-park barrier
```

A request becomes a **slot** in `serve::Engine`. Each engine step builds one batch of rows, from running
users' next tokens and chunks of new prompts, and runs **one forward pass** over it, so all users share a
single read of the weights. Decode for one user is bound by memory bandwidth; batches and prompts are
bound by compute. That split drives most design choices.

## Code map

### `crates/peregrine-kernels`: CPU kernels (no dependencies)

| Module | Role |
|---|---|
| `format.rs` | `Format`, the weight encodings, and the f16 conversion. |
| `cpu.rs` | Kernel tiers and `cpu_has!`: feature detection under the `PEREGRINE_ISA` cap. |
| `delta.rs` | Gated DeltaNet linear-attention recurrent head: safe dispatch, scalar reference, AVX2 and AVX-512 kernels. |
| `simd.rs`, `quant.rs`, `vmath.rs` | x86 kernels for k-quants and Q8_0, activation quantization, vector math. Each has a scalar twin it is tested against. |
| `tern.rs` | Ternary kernels for PQ2_0 / PTQ1_0 (AVX2, AVX-512 VNNI/GFNI/VBMI tiers). |
| `hadamard.rs` | Fast Walsh-Hadamard transform for rotated weights and the rotated int8 KV cache. |
| `options.rs` | The six switches kernel code reads, set from `EngineOptions`. |

`peregrine-core` re-exports these modules under their old paths (`peregrine_core::simd`, ...).

### `crates/peregrine-core`: the engine (a library; knows nothing about HTTP)

| Module | Role |
|---|---|
| `gguf.rs`, `safetensors.rs` | Read model files by memory map. `gguf::Format` lists every weight format the engine knows. |
| `tokenizer.rs`, `template.rs` | Byte-level BPE with the pre-tokenizers transcribed from llama.cpp; chat templates rendered in Rust so the token stream equals llama.cpp's `--jinja` output. |
| `arch.rs` | The `Architecture`, `MtpHead` and `Family` traits (ADR 0006). |
| `model.rs` | `Model`: facade delegating to the `Architecture` trait, `Model::new_state`, tile store for repacked weights, LittleBit accounting, and parameterized loading via family table. |
| `state.rs` | `State`, `Caches`, and `StepStats`: reusable runtime scratch, KV caches, and step telemetry accounting. |
| `model/mtp.rs` | Re-exports for the multi-token-prediction head traits. |
| `blocks/` | Pieces shared by the model families (ADR 0006), as free functions over slices: `linear` (matmul groups, LittleBit), `norm`, `ffn` (SwiGLU, GeGLU), `moe` (the router and the expert feed-forward), `rope`, `attn` (attention scratch, row prep, KV write, attention), `embed`; `rotate`, the block Hadamard rotation (the signs and the transform, in the two orders the hybrid needs); `recurrent`, the Gated DeltaNet steps; `par`, the `Disjoint<'a, T>` checked helper for disjoint worker writes, bounds-checked with debug overlap detection, plus partition drivers. |
| `attn.rs` | Grouped online-softmax attention with optional sliding-window restriction. |
| `kv.rs`, `prefix.rs` | One cache type for attention K/V and recurrent state, snapshots for speculation, and the prefix store that reuses a shared prompt beginning. |
| `sample.rs` | Position-keyed sampling: the random draw depends on (seed, turn, position) only, which makes batching and speculation reproducible. |
| `serve.rs` | `Engine`: slots, scheduling, mixed prefill/decode steps, MTP verification. The server runs this. |
| `session.rs`, `dspark.rs`, `suffix.rs` | `Session`, the older single-user runtime, with DSpark and suffix speculation. To be folded into `Engine` (roadmap S1.4). |
| `profile.rs` | Tuning profiles: read, write, mismatch detection and default paths (ADR 0005). |
| `pool.rs` | The thread pool and barrier. Thread count and barrier count are measured quantities, which is why there is no rayon. |
| `telem.rs` | Bytes moved, phases and machine ceilings: the numbers every benchmark reports. |
| `lbq.rs` | LittleBit 1-bit-class weights. Off by default (it fails accuracy without trained factors). |
| `tests/` | Integration tests for core infrastructure (tokenizers, quantization, SIMD, LBQ container). Model-loading integration tests live in `peregrine-models/tests/`. |

### `crates/peregrine-models`: model architectures and family implementations

| Module | Role |
|---|---|
| `dense/` | Dense transformer architectures: Llama, Qwen2, Qwen3, Phi-3, Phi-4 (fused tensors split at load). `dense/layer.rs` holds the attention path and the output head the dense and the MoE families share. |
| `gemma3/` | Gemma 3 architectures: interleaved sliding-window/global attention, 4 norms per layer, per-head q/k RMSNorm, GeGLU activation, tied output head. |
| `qwen35/` | Hybrid architectures: Gated DeltaNet linear attention + softmax attention, Hadamard rotation, MTP head. |
| `qwen3moe/` | Mixture of experts: Qwen3's attention and a bank of experts (the router and the expert feed-forward), over the shared `blocks::moe`. |
| `lib.rs` | `FAMILIES` table, model loading (`load`, `open`, `open_with`), compatibility checks (`check_compatible`, `check_compatible_with`, `compatibility`), and session opening (`session::open`). |
| `tests/` | Synthetic model generation, golden output regression tests, no-allocation tests, batching and speculative decoding integration tests. |

### `crates/peregrine-bench`: benchmarks and experiments

The micro-benchmarks and one-off experiments, as examples of this crate (`cargo run --release -p
peregrine-bench --example <name>`); most are x86-64 only. Nothing depends on it.

### `crates/peregrine`: the facade and the `peregrine` command

What applications depend on: re-exports the engine (`peregrine::engine`), the kernels (`peregrine::kernels`), the models (`peregrine::models`), and with the default
`server` feature, the server. The `peregrine` binary (`cli` feature) has `serve`, `inspect` (a model file's
architecture, formats and whether this build runs it), `bench` (single-user throughput and bytes/token
measurement), `tune` (coordinate search over tuning knobs) and `tier` (the kernel tier and every option).

### `crates/peregrine-server`: HTTP

A library (`run`, `Args`) and a thin binary. Maps the OpenAI chat request onto `Engine` slots, streams with SSE,
and exposes `/metrics`.

## Cargo Features (ADR 0005)

The engine provides fine-grained Cargo features across its layers so embedders can build minimal binaries for specific models or formats. Default features enable everything.

| Axis | Feature | Crate | Description |
|---|---|---|---|
| **Architectures** | `arch-llama` | `peregrine-models` | Dense Llama models (`llama`) |
| | `arch-qwen` | `peregrine-models` | Dense Qwen models (`qwen2`, `qwen3`) |
| | `arch-qwen35` | `peregrine-models` | Hybrid Gated DeltaNet + softmax attention (`qwen35`, `qwen25`) |
| **Formats** | `fmt-kquant` | `peregrine-kernels` | K-quants: Q2_K through Q6_K |
| | `fmt-ternary` | `peregrine-kernels` | Ternary formats: PQ2_0, PTQ1_0 |
| | `fmt-iq` | `peregrine-kernels` | Importance-matrix quants: IQ4_NL, IQ4_XS |
| | `fmt-legacy` | `peregrine-kernels` | Legacy formats: Q4_0, Q5_0, Q8_0 |
| **Extras** | `littlebit` | `peregrine-core` | LittleBit sidecar loading and 1-bit linear variant (`Linear::Lb`) |
| | `dspark` | `peregrine-core` | DSpark speculative drafting |
| | `mtp` | `peregrine-core` | Multi-token prediction speculation |
| | `server` | `peregrine` | OpenAI-compatible HTTP server (`peregrine-server`) |
| | `cli` | `peregrine` | CLI binary (`clap`, `serde_json`) |

Unused formats and architectures are stripped at compile time. Loading a model that requires a disabled feature produces an error naming the required feature to enable (e.g. `architecture "llama" is not enabled in this build (recompile with feature 'arch-llama')` or `tensor '...' uses format Q4_K which is disabled in this build (enable feature 'fmt-kquant')`). When LittleBit or MTP are disabled, sidecars or MTP blocks are safely ignored with a single log line.

## Invariants

These hold everywhere. A change that breaks one is a bug even if every test passes.

1. **Same tokens.** For a fixed seed, a user's tokens do not depend on batch size, on other users, on
   speculation being on or off, or on which CPU kernel tier ran.
2. **Every fast kernel has a scalar reference** and a test that compares them, bit for bit where integer
   math allows it and within a stated tolerance where floating-point order differs.
3. **Lossy is opt-in.** No default path loses accuracy against the exact computation beyond the measured
   KL budget in `docs/targets.md`.
4. **No hidden allocation per token** in the decode loop; scratch buffers live in the slot or the pool.
5. **The engine crate has one runtime dependency** (`memmap2`). New ones need an ADR (`docs/adr/0002`).

## Cross-cutting concerns

- **CPU dispatch.** Today, feature checks (`have_zmm`, `have_vnni`, `have_avx_vnni`, `have_batched`, ...) sit beside the
  layer code. Kernel tiers (`scalar`, `avx2`, `avx-vnni`, `avx512`) are capped by `PEREGRINE_ISA` and resolved
  once at load per (format, ISA) in `peregrine-kernels` together with all `unsafe` SIMD code (ADR 0005).
- **Configuration.** Engine switches are `PEREGRINE_*` environment variables read where used. The roadmap
  (S0.1) gathers them into one `EngineOptions` struct, printed at startup and recorded in benchmark output.
- **Testing.** Kernel tests run without models. Model tests need GGUF files (`docs/dev/testing.md`); small
  synthetic models for CI are roadmap item S0.2.

## Target layout (ADR 0005, roadmap Stage 1)

| Crate | Contents |
|---|---|
| `peregrine-kernels` | (format × ISA) kernels, scalar references, repacking, the kernel table; all SIMD `unsafe`; formats behind features |
| `peregrine-core` | runtime (options, pool, cache, sampler, scheduler, drafter interface, telemetry), shared layer blocks, the `Architecture` interface; GGUF reader and tokenizer as modules |
| `peregrine-models` | architectures, one module per family, each behind a feature; MTP heads |
| `peregrine` | facade crate, CLI (`run`, `serve`, `bench`, `tune`, `inspect`) and HTTP server |
| `peregrine-bench` | benchmark harness and experiments (nothing depends on it) |

CPU tiers are not compile-time choices: one binary carries all of them and the kernel table picks at run
time. Model- and machine-specific tuning lives in profile files written by `peregrine tune`, not in code.
