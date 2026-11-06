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
   model::Model ── arch::Architecture ── families/ (dense, qwen35, MTP)
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
| `arch.rs` | The `Architecture`, `MtpHead` and `Family` traits, and the family dispatch table (ADR 0006). |
| `families/` | Architecture implementations behind the `Architecture` trait: `dense` (Llama, Qwen2, Qwen3) and `qwen35` (Gated DeltaNet + attention, Hadamard rotation, MTP). |
| `model.rs` | `Model`: facade delegating to the `Architecture` trait, tile store for repacked weights, and LittleBit accounting. |
| `model/hybrid.rs` | Re-exports for Qwen 3.5 / Bonsai 2 (`families/qwen35`). |
| `model/mtp.rs` | Re-exports for the multi-token-prediction head (`families/qwen35/mtp.rs`). |
| `blocks/` | Pieces shared by the model families (ADR 0006), as free functions over slices: `linear` (matmul groups, LittleBit), `norm`, `ffn` (SwiGLU), `rope`, `attn` (attention scratch, row prep, KV write, attention), `embed`; `rotate`, the block Hadamard rotation (the signs and the transform, in the two orders the hybrid needs); `recurrent`, the Gated DeltaNet steps; `par`, the `Disjoint<'a, T>` checked helper for disjoint worker writes, bounds-checked with debug overlap detection, plus partition drivers. |
| `attn.rs` | Grouped online-softmax attention. |
| `kv.rs`, `prefix.rs` | One cache type for attention K/V and recurrent state, snapshots for speculation, and the prefix store that reuses a shared prompt beginning. |
| `sample.rs` | Position-keyed sampling: the random draw depends on (seed, turn, position) only, which makes batching and speculation reproducible. |
| `serve.rs` | `Engine`: slots, scheduling, mixed prefill/decode steps, MTP verification. The server runs this. |
| `session.rs`, `dspark.rs`, `suffix.rs` | `Session`, the older single-user runtime, with DSpark and suffix speculation. To be folded into `Engine` (roadmap S1.4). |
| `pool.rs` | The thread pool and barrier. Thread count and barrier count are measured quantities, which is why there is no rayon. |
| `telem.rs` | Bytes moved, phases and machine ceilings: the numbers every benchmark reports. |
| `lbq.rs` | LittleBit 1-bit-class weights. Off by default (it fails accuracy without trained factors). |
| `tests/` | Integration tests. Those that need real model files skip when the files are absent. |

### `crates/peregrine-bench`: benchmarks and experiments

The micro-benchmarks and one-off experiments, as examples of this crate (`cargo run --release -p
peregrine-bench --example <name>`); most are x86-64 only. Nothing depends on it.

### `crates/peregrine`: the facade and the `peregrine` command

What applications depend on: re-exports the engine (`peregrine::engine`), the kernels and, with the default
`server` feature, the server. The `peregrine` binary (`cli` feature) has `serve`, `inspect` (a model file's
architecture, formats and whether this build runs it) and `tier` (the kernel tier and every option).

### `crates/peregrine-server`: HTTP

A library (`run`, `Args`) and a thin binary. Maps the OpenAI chat request onto `Engine` slots, streams with SSE,
and exposes `/metrics`.

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

- **CPU dispatch.** Today, feature checks (`have_zmm`, `have_vnni`, `have_batched`, ...) sit beside the
  layer code. The roadmap (S1.1) replaces them with a kernel table chosen once at load per (format, ISA),
  moved into a `peregrine-kernels` crate together with all `unsafe` SIMD code (ADR 0005).
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
