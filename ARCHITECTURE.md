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
  pool.rs: one persistent thread pool, spin-then-park barrier; a server with
           several models shares it, and a PassTurn runs one engine step at a time
```

A request becomes a **slot** in `serve::Engine`. Each engine step builds one batch of rows, from running
users' next tokens and chunks of new prompts, and runs **one forward pass** over it, so all users share a
single read of the weights. Decode for one user is bound by memory bandwidth; batches and prompts are
bound by compute. That split drives most design choices.

A slot that never decodes — an embedding's pooled slot or a decision question's label slot — is a slot
like any other, with two differences. A pass fills up to a **row budget** (`PEREGRINE_PASS_ROWS`, from
the machine's L2 and the file's bytes per weight) with such slots' rows, taking whole short texts and
pieces of a long one from as many waiting slots as fit, instead of one prefill chunk in total; the
matmul driver splits a batch wider than its 128-row kernel scratch into blocks. Its **cache is sized to
its own prompt**, rounded to whole prefill chunks, not to the model's context, so a batch's memory is
its texts' tokens. The server admits them by a memory budget (`PEREGRINE_WORK_CACHE_MB`, 512 MiB) and a
slot cap, not by `--slots`, which bounds chat slots only: a wide embedding request never takes or waits
for a chat slot. A chat slot keeps its chunk and its share of a mixed pass (embedding rows in a pass that
carries decode rows are bounded to a quarter of a prefill chunk).

There is one runtime (ADR 0007). A conversation is the same engine with one slot: `Session` renders the
prompt, keeps the chat history and lends the slot the cache the conversation has filled so far.

### Two kinds of forward pass

Every decoder forwards rows causally and writes a KV cache: a prompt is read in chunks, a decode step is
one row per user, and the cache is what makes the next token cheap. A **bidirectional encoder** (`bert`:
MiniLM, BGE, E5, GTE, bge-m3; `modern-bert`: ModernBERT, gte-modernbert, Granite Embedding R2;
`gemma-embedding`: EmbeddingGemma; `mistral3`: Nemotron 3.5 Embed) is the other shape.
`Architecture::causality` says which a family is.
For an encoder the engine schedules **whole texts**, never chunks: `Architecture::forward_texts` takes
the batch's ragged texts, sends every row of every text through the layer projections in the same weight
passes, and computes attention as one `n × n` block per text. Keys and values stay f32 in the pass, no KV
cache is written, and a text's arithmetic depends only on its own length. A pass packs whole texts (never
pieces) up to the same row budget, so a batch of short texts fills it. Pooling, the model's dense head and
normalisation happen after the final `output_norm`; chat is refused with a 400. The encoder's slots
allocate no cache.

## Code map

### `crates/peregrine-kernels`: CPU kernels (no dependencies)

| Module | Role |
|---|---|
| `format.rs` | `Format`, the weight encodings, and the f16 conversion. |
| `cpu.rs` | Kernel tiers and `cpu_has!`: feature detection under the `PEREGRINE_ISA` cap. |
| `delta.rs` | Gated DeltaNet linear-attention recurrent head: safe dispatch, scalar reference, AVX2 and AVX-512 kernels. |
| `simd.rs`, `quant.rs`, `vmath.rs` | x86 kernels for k-quants and Q8_0, activation quantization, vector math. Each has a scalar twin it is tested against. |
| `fpack.rs` | The float GEMM (S4.9f): packing F32/F16/BF16 weights into 16-output-row panels, and the scalar, AVX2 (6×16) and AVX-512 (2×6×16) kernels plus the single-row GEMV. Every output is one sequential FMA chain, so the tier, tile and batch cannot change a bit. |
| `tern.rs` | Ternary kernels for PQ2_0 / PTQ1_0 (AVX2, AVX-512 VNNI/GFNI/VBMI tiers). |
| `hadamard.rs` | Fast Walsh-Hadamard transform for rotated weights and the rotated int8 KV cache. |
| `options.rs` | The kernel switches (`PEREGRINE_ISA`, `ZMM`, `NO_WIDE`, `PF`, `TQ_SPAN`, `EXACT`, `KREPACK`, `FPACK`), set from `EngineOptions`. |
| `packcache.rs` | The on-disk cache of the load-time packs (S4.9e): the key and version, the file format, atomic writes, header/body verification, the LRU size cap, and `peregrine inspect`/`tier -m`'s model status. |

`peregrine-core` re-exports these modules under their old paths (`peregrine_core::simd`, ...).

### `crates/peregrine-core`: the engine (a library; knows nothing about HTTP)

| Module | Role |
|---|---|
| `gguf.rs`, `safetensors.rs` | Read model files by memory map. `gguf::Format` lists every weight format the engine knows. |
| `tokenizer.rs`, `template.rs` | Byte-level BPE with the pre-tokenizers transcribed from llama.cpp; chat templates rendered in Rust so the token stream equals llama.cpp's `--jinja` output. |
| `systemone.rs`, `json.rs` | System One prompts: one renderer per decision type, written from the type's `systemone` template and checked byte for byte against prompts llama.cpp rendered (`openjev` so far); and the order-keeping JSON value whose text is the one llama.cpp's Jinja `tojson` writes. |
| `decision.rs` | The decision front end and answer math: `DecisionInfo` reads a file's decision metadata; `Decider` refuses a type or `systemone` template this build does not render, checks the option letters are single tokens, and turns a request into prompt questions (llama.cpp's `parse_questions` and its messages); `format_answer` turns one score vector per question into a choice, score or noul answer. |
| `arch.rs` | The `Architecture`, `MtpHead` and `Family` traits (ADR 0006). |
| `model.rs` | `Model`: facade delegating to the `Architecture` trait, `Model::new_state`, the tile/k-quant/float stores for repacked and packed weights, LittleBit accounting, and parameterized loading via family table. |
| `kq.rs`, `fpack.rs` | The load-time packs: k-quants into eight-row tiles (S3.4) and F32/F16/BF16 into 16-row panels (S4.9f), both filled by reading the file so the mapped pages stay cold. |
| `packcache.rs` | Where those two stores are cached on disk and mapped read-only on the next load (S4.9e): `~/.cache/peregrine/packs`, keyed by the model's size, mtime and header hash, the layout version, the packed tensors, the switches and the kernel tier; atomic writes, a sampled body checksum and an LRU cap. |
| `state.rs` | `State`, `Caches`, and `StepStats`: reusable runtime scratch, KV caches, and step telemetry accounting. |
| `model/mtp.rs` | Re-exports for the multi-token-prediction head traits. |
| `blocks/` | Pieces shared by the model families (ADR 0006), as free functions over slices: `linear` (matmul groups, LittleBit), `norm`, `ffn` (SwiGLU, GeGLU, OpenAI's `swiglu_oai`), `moe` (the router, the expert feed-forward, the expert biases), `rope` (the plain and the YaRN tables), `attn` (attention scratch, row prep, KV write, attention, the attention sinks), `bidir` (the encoder's per-text `n × n` attention and its symmetric sliding window), `embed`; `rotate`, the block Hadamard rotation (the signs and the transform, in the two orders the hybrid needs); `recurrent`, the Gated DeltaNet steps; `par`, the `Disjoint<'a, T>` checked helper for disjoint worker writes, bounds-checked with debug overlap detection, plus partition drivers. |
| `attn.rs` | Grouped online-softmax attention with optional sliding-window restriction. |
| `kv.rs`, `prefix.rs` | One cache type for attention K/V and recurrent state; snapshots that carry both (the recurrent state after exactly their length), used by speculation, by the prefix store and to fork a cache into another at a position (`KvCache::fork_from`); and the prefix store that reuses a shared prompt beginning. |
| `sample.rs` | Position-keyed sampling: the random draw depends on (seed, turn, position) only, which makes batching and speculation reproducible. |
| `serve.rs` | `Engine`: the one runtime — slots, scheduling, mixed prefill/decode steps, the non-decoding pass budget and admission, and the verification step every draft goes through. The server runs this. |
| `session.rs` | `Session`: the conversation layer over a one-slot `Engine` (ADR 0007) — the chat history, the template, the prompt, the reuse of the positions the previous turn computed, the context shifting, the per-turn telemetry. |
| `draft.rs`, `dspark.rs`, `suffix.rs` | The `Drafter` interface every speculation goes through, and the drafters behind it: the suffix index (`suffix.rs`), DeepSeek's DSpark head (`dspark.rs`), the model's own MTP block (`draft.rs`). |
| `profile.rs` | Tuning profiles: read, write, mismatch detection and default paths (ADR 0005). |
| `pool.rs` | The thread pool and barrier. Thread count and barrier count are measured quantities, which is why there is no rayon. |
| `telem.rs` | Bytes moved, phases and machine ceilings: the numbers every benchmark reports. |
| `lbq.rs` | LittleBit 1-bit-class weights. Off by default (it fails accuracy without trained factors). |
| `tests/` | Integration tests for core infrastructure (tokenizers, quantization, SIMD, LBQ container). Model-loading integration tests live in `peregrine-models/tests/`. |

### `crates/peregrine-models`: model architectures and family implementations

| Module | Role |
|---|---|
| `dense/` | Dense transformer architectures: Llama, Qwen2, Qwen3, Phi-3, Phi-4 (fused tensors split at load), and Mistral 3 as a bidirectional embedder (`mistral3`). `dense/layer.rs` holds the attention path and the output head the dense and the MoE families share. |
| `bert/` | The BERT encoder family (`bert`): post-norm blocks with biases on every projection, learned absolute positions, token types, LayerNorm and the exact GELU; bidirectional attention over whole texts, cls/mean pooling. MiniLM, BGE, E5, GTE and the XLM-R models (bge-m3, multilingual-e5). |
| `modern_bert.rs` | The ModernBERT family (`modern-bert`): pre-norm blocks whose norms have a weight and no bias, a fused q/k/v matrix, RoPE with separate local/global bases, every third layer global and the others on a symmetric 128-token window, the gated feed-forward (erf GeGLU, tanh GeGLU or SwiGLU), cls/mean pooling. ModernBERT-base/large, gte-modernbert, Granite Embedding R2; a classifier head is read and not served yet. |
| `gemma3/` | Gemma 3 and `gemma-embedding`: interleaved sliding-window/global attention, 4 norms per layer, per-head q/k RMSNorm, GeGLU activation, tied output head; the encoder is the same blocks bidirectional, with mean/cls pooling and the `dense_2`/`dense_3` head. |
| `qwen35/` | Hybrid architectures: Gated DeltaNet linear attention + softmax attention, Hadamard rotation, MTP head. |
| `qwen3moe/` | Mixture of experts: Qwen3's attention and a bank of experts (the router and the expert feed-forward), over the shared `blocks::moe`. |
| `gptoss/` | OpenAI's mixture of experts: attention sinks, alternating sliding-window and full attention, YaRN rope scaling, biases on every projection, and the `swiglu_oai` activation, over the shared `blocks::moe`. |
| `lib.rs` | `FAMILIES` table, model loading (`load`, `open`, `open_with`), compatibility checks (`check_compatible`, `check_compatible_with`, `compatibility`), and session opening (`session::open`). |
| `tests/` | Synthetic model generation, golden output regression tests, no-allocation tests, batching and speculative decoding integration tests. |

### `crates/peregrine-bench`: benchmarks and experiments

The micro-benchmarks and one-off experiments, as examples of this crate (`cargo run --release -p
peregrine-bench --example <name>`); most are x86-64 only. Nothing depends on it.

### `crates/peregrine`: the facade and the `peregrine` command

What applications depend on: re-exports the engine (`peregrine::engine`), the kernels (`peregrine::kernels`), the models (`peregrine::models`), and with the default
`server` feature, the server. The `peregrine` binary (`cli` feature) has `run` (one prompt, streamed to
stdout, no server), `serve`, `inspect` (a model file's architecture, formats and whether this build runs
it), `bench` (single-user throughput and bytes/token measurement), `tune` (coordinate search over tuning
knobs) and `tier` (the kernel tier and every option). Every command that takes `-m` also accepts
`hf:<namespace>/<repo>/<file>[@<rev>]`: `fetch.rs` reads the GGUF header before downloading the weights,
checks it against the build, and caches the file under `~/.cache/peregrine/models`.
`serve --background` detaches with `SO_REUSEPORT` and writes a lifecycle state file
(`crates/peregrine-server/src/lifecycle.rs`); `service.rs` starts it, `status`/`stop` read that file,
and `upgrade` installs a release and hands the port from the old process to the new one while both
listen, so in-flight requests drain instead of failing.

### `crates/peregrine-server`: HTTP

A library (`run`, `Args`) and a thin binary. Maps the OpenAI chat request onto `Engine` slots, streams with SSE,
and exposes `/metrics`. `/v1/embeddings`, `/v1/rerank` and `/v1/classify` submit one pooled or
head slot per text, and `/v1/systemone` one labels slot per input or
question through the same queue, so they share the weight pass with chat; the decision route's body is
parsed with `peregrine_core::json` to keep the request's order, and its answers are engine label logits
through `decision::format_answer`.

## Cargo Features (ADR 0005)

The engine provides fine-grained Cargo features across its layers so embedders can build minimal binaries for specific models or formats. Default features enable everything.

| Axis | Feature | Crate | Description |
|---|---|---|---|
| **Architectures** | `arch-llama` | `peregrine-models` | Dense Llama models (`llama`) |
| | `arch-qwen` | `peregrine-models` | Dense Qwen models (`qwen2`, `qwen3`) |
| | `arch-qwen35` | `peregrine-models` | Hybrid Gated DeltaNet + softmax attention (`qwen35`, `qwen25`) |
| | `arch-gemma3` | `peregrine-models` | Gemma 3 (`gemma3`) |
| | `arch-gemma-embedding` | `peregrine-models` | Gemma 3 bidirectional encoder (`gemma-embedding`, EmbeddingGemma) |
| | `arch-bert` | `peregrine-models` | BERT bidirectional encoder (`bert`: MiniLM, BGE, E5, GTE, bge-m3) |
| | `arch-modern-bert` | `peregrine-models` | ModernBERT bidirectional encoder (`modern-bert`: ModernBERT, gte-modernbert, Granite Embedding R2) |
| | `arch-qwen3moe` | `peregrine-models` | Mixture of experts (`qwen3moe`) |
| | `arch-gpt-oss` | `peregrine-models` | OpenAI's mixture of experts (`gpt-oss`) |
| | `arch-phi3` | `peregrine-models` | Phi-3 and Phi-4 (`phi3`) |
| | `arch-mistral3` | `peregrine-models` | Mistral 3 bidirectional embedders (`mistral3`, Nemotron 3.5 Embed) |
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
