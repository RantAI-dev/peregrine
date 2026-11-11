# S4 Design: embeddings and decision models

- Status: Accepted; revised 2026-11-10 (the decision part now follows the System One API, see §2.5)
- Date: 2026-11-10
- Target: a new roadmap stage, S4 (items S4.1 to S4.10)
- Inputs: `ARCHITECTURE.md` (the invariants), ADR 0003 (lossless by default), ADR 0006 (the
  architecture interface), ADR 0007 (one runtime), `docs/targets.md`
- Reference: llama.cpp **v0.9.0** (`cd959ba`). Its architecture table (`src/llama-arch.cpp`) lists
  `bert`, `modern-bert`, `nomic-bert`, `nomic-bert-moe`, `neo-bert`, `jina-bert-v2`, `jina-bert-v3`,
  `eurobert`, `gemma-embedding` and the keys `%s.pooling_type`, `%s.classifier.output_labels`,
  `%s.dense_2_feat_in/out`, `cls`, `cls.output`. Every model detail below marked *(transcribe)*
  must be read from llama.cpp at that tag by the implementing PR, not taken from this note.
  The decision models are specified in [`docs/reference/systemone.md`](../reference/systemone.md),
  a reading of llama.cpp's `tools/server/server-decision.cpp` with the code quoted.

## 0. Summary

Peregrine serves text generation. This stage adds the two other things a CPU box is asked to host
next to a chat model:

1. **Embedding models**: a text goes in, one vector comes out (`POST /v1/embeddings`).
2. **Decision models** ("System One": one fast forward pass, an answer instead of prose):
   - **System One** (`POST /v1/systemone`, the TypeSafe-compatible API llama.cpp serves): typed
     questions about a `state` (`choice` among options, `score` on an ordered scale, `noul` for
     yes/no), answered by a **native decision model**: OpenJev, Lev, Kev, Nimble, Laya, vLLM D3,
     H2O Lightning and the others in §1. These are files that carry `<arch>.decision.type` and a
     `systemone` prompt template; most of them are Qwen 3.5 models, which Peregrine already runs.
   - **rerank**: a query and N documents, a relevance score per document (`POST /v1/rerank`);
   - **classify**: a text, a score per label from a classifier head (`POST /v1/classify`), which is
     how encoder guard and router models ship.

   A chat model that classifies through a prompt is not a decision model (llama.cpp draws the same
   line); an earlier draft of this note had a `/v1/decide` endpoint for that, which is dropped.

The speed claim is the same one the rest of the project makes: on the same machine and the same
file, at least llama.cpp's throughput, and against the engines people actually deploy for this
(ONNX Runtime through text-embeddings-inference and fastembed, OpenVINO) at least their throughput
at equal accuracy. Nothing is claimed before it is measured (S4.2 measures first).

Why Peregrine should win here, and where the work is:

- An encoder pass and a decision are **prompt processing with no decode**. That is the path S3.4,
  S3.5 and S3.7 made fast: batched int8 GEMM over repacked tiles and tiled attention.
- **A ragged batch costs nothing.** The engine's batch is a list of rows, not a padded rectangle.
  Thirty-two texts of different lengths are one weight pass over exactly their tokens; engines that
  pad to the longest text in the batch pay for the padding in every matmul.
- **Decisions share their prompt.** The questions of one System One request are asked about the
  same `state`, and a reranker scores N documents against one instruction and one query. For the
  causal decision models the shared beginning is computed once and the questions fork from it.
- A decision needs **one row of outputs and no sampling**: the last prompt row's logits at a few
  label tokens (or a small head on its hidden state) are the answer. No decode step runs.

## 1. Models in scope, in order

| Step | Family | Files it opens | New code |
|---|---|---|---|
| S4.1 | decoders already supported, as embedders | Qwen3-Embedding 0.6B / 4B / 8B (`qwen3`, last-token pooling) | pooling, the endpoint; no model code |
| S4.3 | System One, label-logit types on `qwen35` | OpenJev (and its 46 MB test file), Lev 4B, Nimble 9B, pplx-decider, H2O Lightning 4B | `/v1/systemone`, the prompt renderers, the answer math; no model code |
| S4.11 | System One, small heads on `qwen35` | Kev 0.8B / 4B / 9B (`cls.output` pointer head); vLLM D3 edge / lite / nano / mini (non-causal attention blocks) | the head; the bidirectional mask on Qwen 3.5's attention blocks |
| S4.12 | System One on `modern-bert` | Laya (421M), Julia-1 (144M) | the two decision head blocks and the scorer, after S4.7 |
| S4.13 | generative rerankers | Qwen3-Reranker 0.6B / 4B (`pooling_type = rank`, a 2-row `cls.output` in place of the output head) | the rank read-out on a causal model, `/v1/rerank` |
| S4.4 | `gemma-embedding` | EmbeddingGemma 300M | bidirectional attention on the Gemma 3 family, mean pooling, the two dense layers |
| S4.5 | tokenizers | WordPiece (`bert`), SentencePiece unigram (`t5`, the XLM-R vocabulary) | two tokenizers |
| S4.6 | `bert` | all-MiniLM-L6/L12, bge-small/base/large-en-v1.5, e5, gte, bge-m3, multilingual-e5; bge-reranker-v2-m3 with its `cls` head | post-norm encoder, absolute positions, the classifier head |
| S4.7 | `modern-bert` | ModernBERT, gte-modernbert, mmBERT, Granite Embedding R2, and their fine-tuned classifiers (`classifier.output_labels`) | alternating local and global attention, the labelled head |
| S4.8 | `nomic-bert`, `nomic-bert-moe`, `jina-bert-v2/v3` | nomic-embed-text v1.5 / v2, jina-embeddings v2 / v3 | per family, on demand |
| S4.14 | `mistral3`, non-causal | Nemotron 3.5 Embed 1B and Nemotron 3 Embed 1B (the same text model; mean pooling, 2048 dimensions); the 8B sizes load by the same code | the Tekken pre-tokenizer, the Mistral 3 block in the dense family, a bidirectional pass for dense models |

Out of scope for this stage: `neo-bert`, `eurobert`, DeBERTa (Prompt Guard 2; llama.cpp does not
read it either), the ColBERT multi-vector output; image input for Nemotron 3.5 Embed (§2.6);
and of the decision types, the ones that need a
model family Peregrine does not have (`lfm2-d1`, `lfm3-d1`, `granite-switch`), the large joint head
(`clef`), vision (`minicpm-ocr-jev`, and image input for every type), and `vllm-decision2`, for
which no file in llama.cpp's format is published (`docs/reference/systemone.md` §5).

## 2. The runtime

### 2.1 One runtime stays one runtime (ADR 0007)

An embedding or decision request is a **slot** in `serve::Engine` like a chat request. It differs in
what it asks for at the end:

```rust
pub enum Want {
    Tokens,                         // today: sample and stream
    Pooled(Pooling),                // an embedding: Last | Mean | Cls, then the model's own head
    Labels,                         // a decision: the last row's logits at these label tokens
    Head,                           // a classifier head's row for the text (S4.6b)
}
```

The slot is admitted, prefilled in chunks beside whatever else is running, and finishes when its
last prompt row has been computed. It never enters decode. `State::hidden` already carries the
final hidden states for the MTP drafter (`want_hidden`); pooling reads the same rows.

### 2.2 Two kinds of forward pass

- **Causal** (every decoder; S4.1, S4.3): nothing changes in the forward pass. Rows are prefilled
  with the cache as today, so the prefix store, the chunked prefill and batching apply unchanged.
- **Bidirectional** (S4.4 onward): every row of a text attends to every position of that text, so a
  text cannot be chunked and has no reusable prefix. The whole text is one unit of work:
  1. all rows of the text go through each layer's projections in the shared GEMM (rows of several
     texts side by side, ragged);
  2. attention for a text is its own `n × n` block: K and V for the layer are computed for all `n`
     rows first, then every row scores against all `n` keys. It reads the f32 K and V directly; no
     cache is written and nothing is rounded to f16.

  `Architecture` gains one method, `fn attention(&self) -> Causality { Causal | Bidirectional }`,
  and the engine schedules whole texts for bidirectional models (admission by a token budget per
  step instead of the prefill chunk). `forward_rows` receives the text boundaries of the batch.

### 2.3 Invariance, restated for this stage

Invariant 1 extends as written: **a text's vector, scores or choice probabilities are bit-identical
whatever else is in the batch**, for any batch size, thread count and admission order. This is the
property that lets a server batch freely and is tested the same way as today's batch invariance
(alone, in a batch of 2, 7, 64; mixed with chat slots on a causal model).

It dictates two details: a text's attention partition is a function of its own length only (as in
`attn.rs` since #94), and pooling sums in a fixed order (position order, f64 accumulator for the
mean, one f32 rounding at the end).

### 2.4 Pooling and heads

- `Last` (causal embedders), `Mean` (attention-mask mean over the text's own tokens), `Cls` (row 0).
  The default comes from `<arch>.pooling_type`; a request may not override it to a type the model
  was not trained for unless asked with an explicit option.
- After pooling: the model's own dense layers if present (`dense_2`, `dense_3`: EmbeddingGemma),
  then optional truncation to `dimensions` (Matryoshka), then L2 normalisation. Truncate, then
  normalise.
- Classifier head *(transcribe per family)*: `cls` (dense + activation) and `cls.output`, giving one
  logit (a reranker) or one per label (`classifier.output_labels`).
  **As built (S4.6b)**: the reranker's `relevance_score` is that raw logit, because that is what
  llama.cpp returns (`send_rerank` reads `embd[0]`; the earlier note said sigmoid, which no
  llama.cpp path applies for `bert`); a classifier's scores are the softmax of the head's outputs,
  and the sigmoid only when the file has one output and names its label. The head's activation is
  the file's `classifier.activation` (tanh unless it says gelu or silu), and its pool is row 0
  unless `classifier.pooling_type = 1` (mean).

### 2.5 System One decisions

Everything here is transcribed from llama.cpp v0.9.0 and written down, with the code, in
`docs/reference/systemone.md`; the section numbers below are that note's.

**What a decision model is.** A GGUF with `<arch>.decision.type` (one of sixteen type names) and a
Jinja template `tokenizer.chat_template.systemone`, plus per-type extras: `decision.labels`,
`decision.temperature.<name>`, `decision.score_bias.<n>.<i>`, and for some types head tensors
(`cls.output`, `decision.*`). `peregrine inspect` and `/v1/models` report it
(`architecture.output_modalities` contains `"decisions"`).

**A request** is a `state` and a map of questions; each question has a type (`choice`, `score`,
`noul`), `instructions` and `criteria`. Questions are answered independently.

**One question, one pass** (two for a Lev `choice`, which is asked in both option orders):

1. *Render* the prompt from the type's template with the state, the instructions and the options.
   Peregrine has no Jinja engine and does not gain one: each supported type gets a renderer written
   in Rust, as the chat templates are, **checked byte for byte against llama.cpp's rendering** on a
   fixture set that covers strings, objects and arrays as state, null and empty descriptions, every
   question type and the option-count limits. The renderer is selected by `decision.type`; if the
   file's template text is not the one the renderer was transcribed from (compared by hash), the
   file is refused with a message naming the type, never rendered approximately.
2. *Tokenise* with no BOS added and special tokens parsed, as llama.cpp does, including its choice
   per type of whether text from the user is escaped (§3 of the note).
3. *Forward* the prompt. For the causal types the pass is a normal prefill.
4. *Read* one raw score per option:
   - label-logit types (openjev, lev, nimble, pplx-decider, h2o-lightning, vllm-d3): the last row's
     logit at option i's label token. Only those rows of the output head are needed, not the
     whole vocabulary: the head is evaluated for the label tokens alone.
   - kev: the dot product of the projected hidden state of the last token with that of each
     option's end token (`cls.output`).
   - laya: the scorer head's output at each option's marker token.
5. *Answer* (§2 of the note): `p = softmax((score + score_bias) / T)` over the options only, in
   f64, with `T` looked up by question type and option-count bucket; then `choice` and its
   confidence, the expected `score` level, or the `noul` probability, each by llama.cpp's formula.

**Sharing the state.** The questions of a request (and both variants of a Lev question) begin with
the same tokens up to the point where the question text starts. For causal types the engine
computes that prefix once and forks the cache for each question; for non-causal types (vllm-d3,
laya) nothing can be shared and each question is one whole pass. Whether a fork is bit-identical
to computing the question alone is part of §2.3 and is tested; if the cache's f16 rounding makes it
differ, the fork is made at f32 or not at all.

**The slot** asks for `Want::Labels { ids }` (or `Want::Head` for kev and laya) and finishes when
its last prompt row is computed, like a pooled slot.

**Generative rerankers** (S4.13) are the same mechanism with a fixed prompt: llama.cpp's converter
gives Qwen3-Reranker `pooling_type = rank`, labels `["yes", "no"]`, a `rerank` template, and a
2-row `cls.output` holding the output head's rows for the two tokens (the file may have no output
head at all). The score is the softmax of those two logits at the last row.

### 2.6 Mistral 3 embedders (S4.14)

NVIDIA's Nemotron 3.5 Embed 1B is a retrieval embedder for text and page images in 34 languages.
Its text model is a 16-layer Mistral 3 transformer run without the causal mask; llama.cpp v0.9.0
reads it as `general.architecture = mistral3` with `mistral3.attention.causal = false` and
`mistral3.pooling_type = 1` (mean). Its text model was taken from Nemotron 3 Embed 1B and kept frozen, so
one piece of code serves both. What the published files say (Q8_0 header, 146 tensors):

| | |
|---|---|
| Size | 16 blocks, width 2048, feed-forward 6144 (SiLU, gated), RMSNorm (ε 1e-5), no biases |
| Attention | 24 heads of 128 over 8 key/value heads: queries are 3072 wide, keys and values 1024 |
| Positions | RoPE over the whole head (128), base 1e6, YaRN scaling: factor 16, original context 16384, β 32 / 1, `yarn_log_multiplier` 1 |
| Attention temperature | `attention.temperature_scale = 0.1`: each query row is multiplied by `1 + 0.1 · ln(⌊pos / 16384⌋ + 1)` after rotation, which is exactly 1 below position 16384 |
| Output | final RMSNorm, mean over every token, L2 normalisation; 2048 numbers, and the first 1024 or 512 re-normalised (`dimensions`) |
| Vocabulary | 131072 tokens, byte-pair merges, pre-tokenizer `pixtral` (Mistral's Tekken); no BOS or EOS is added |
| Limits | 32768 tokens per text (the file's `context_length` is 262144) |

**What Peregrine needs.** The parts exist: a bidirectional pass over whole texts with no cache
(§2.2), mean pooling, YaRN positions (gpt-oss), heads wider than the model (Qwen3). Missing are the
Tekken pre-tokenizer, `mistral3` as a member of the dense family, and that family's
`forward_texts`. The temperature is implemented as written, not skipped for short texts, and tested
on a synthetic file whose original context is a few tokens.

**Input.** A text is embedded as given. The model was trained on `query: ` and `passage: `
prefixes, and the client writes them, as it does with llama.cpp. The model's own template also
puts a zero-width non-joiner after every `[` and `<` of user text so that text can never spell a
control token; llama.cpp's later pre-tokenizer name `nemotron-3.5-embed` removes the joiner again
before splitting. The files published so far were converted before that name existed and say
`pixtral`. Peregrine accepts `tekken`, `pixtral` and `nemotron-3.5-embed`, and gives the same ids
as llama.cpp for each.

**Not in this step.** Image input: the vision encoder (Pixtral, 24 layers) and its projector ship
as a separate `mmproj` file, and llama.cpp does not accept images on its embedding endpoint for
this model either. The causal Mistral 3 chat models (Ministral 3) would load by the same block
but need Mistral's chat template; they are not claimed here.

**Checks.** The gate of §5 on the real file in Q8_0, Q4_K_M and F16; goldens on every tier; speed
against llama.cpp on both x86 hosts at 16, 128, 512 and 2000 tokens, alone and batched.

## 3. API

Shapes follow what clients already speak; unknown fields are rejected, as in chat.

- `POST /v1/embeddings` (OpenAI): `input` (string, array of strings, array of token ids, array of
  arrays of token ids), `model`, `dimensions`, `encoding_format` (`float` | `base64`), and the
  usual `usage`. A text longer than the model's context is an error naming the limit, not a silent
  truncation; `truncate: true` opts into truncation.
- `POST /v1/rerank` (Jina / Cohere shape, also served at `/rerank`): `query`, `documents`, `top_n`,
  `return_documents`; `results: [{index, relevance_score}]` sorted by score.
- `POST /v1/classify`: `input` (string or array); `results: [{index, label, score, scores:
  {label: p}}]`, in input order — `label` is the most likely label, `score` its probability and
  `scores` every label's (the softmax of the head's outputs; a sigmoid when a one-output head names
  its label). A head with several outputs and no labels is served as `LABEL_0`, `LABEL_1`, …; a
  one-output head with no labels is a reranker and is refused here by name.
- `POST /v1/systemone` (TypeSafe / llama.cpp shape): `state` (string, object or array; a
  non-string is given to the model as JSON text), `questions` (id → `{type, instructions,
  criteria}`); response `{model, answers: {id: {...}}, usage: {input_tokens, output_tokens: 0}}`
  with `choice` + `probabilities` + `confidence`, `score` + `legend` + `probabilities` +
  `confidence`, or `noul`. Errors as llama.cpp gives them: 400 for an invalid request, 501 when the
  model is not a decision model or when `files` / `images` are sent (image input is not supported
  in this stage). The exact schema, limits and messages are in `docs/reference/systemone.md` §1.
- CLI: `peregrine embed`, `peregrine rerank`, `peregrine classify`, `peregrine decide` (a System
  One request: `--state`, and questions from flags or a JSON file), each with
  `-m` (and `hf:`), reading stdin when no text is given, `--json` for machine output.
  `peregrine inspect` says what a file is for (generate, embed, rerank, classify, decide).
- `serve` on an encoder file exposes only the endpoints the file supports; chat on an embedding
  model is a 400 that says so.

## 4. Speed

### 4.1 What is measured

`benchmarks/` gains an embedding mode: texts per second and tokens per second, for one text at a
time and for batches (1, 8, 32, 128), at token lengths 16, 128 and 512 (and 2048 where the model
allows), single client and 10 clients; p50 and p99 latency for one short text (the guard and
router case). Rerank: documents per second for 1 query × 32 documents of 128 and 512 tokens.
System One: questions per second and p50 latency for a 300-token state with 1, 3 and 10 questions
(choice with 3 and 20 options, score, noul), against `llama-server` on the same file.

Competitors, pinned: llama.cpp v0.9.0 (`llama-server --embedding`, `--reranking`),
text-embeddings-inference (CPU image, ONNX Runtime backend), fastembed (ONNX Runtime), and
sentence-transformers with the OpenVINO backend. Same model, same threads, same texts. Where a
competitor runs a different precision (ONNX int8, fp32) the table says so and the accuracy column
is what makes the row comparable.

### 4.2 Targets (to be confirmed by S4.2's baselines)

| Measure | Target |
|---|---|
| Throughput, any batch, vs llama.cpp on the same GGUF | ≥ 1.0× on every row, ≥ 1.5× on batches |
| Throughput vs the best of ONNX Runtime / OpenVINO at equal accuracy | ≥ 1.0× |
| p50 latency, one 16-token text, MiniLM-class model | ≤ the best competitor |
| System One on the same file, vs llama.cpp | ≥ 1.0× per question; ≥ 1.5× for a request with several questions (the state is computed once) |
| Rerank with a shared instruction and query | ≥ 1.5× llama.cpp |

### 4.3 Where the time will be, and the work it implies

1. **Weight matmuls** dominate for base-size models and up: the existing tiles. Encoder files are
   usually F32, F16 or Q8_0; the F32 and F16 GEMM must be register-blocked like the int8 ones (S4.9
   measures first; llama.cpp's f32 path is a tuned SGEMM).
2. **Attention** is `n²` per text and is the larger share for long texts on small models. The
   bidirectional block is a dense `n × n` score matrix: tile it query × key as #113 does, f32 in,
   no cache traffic.
3. **Small models** (MiniLM: 6 layers, d 384, 22 M parameters) are overhead-bound: the fork-join
   per matmul, the tokenizer, the HTTP path. One text on such a model should run on few threads,
   or one; the calibration that picks the decode pool picks this too. A batch amortises it.
4. **The tokenizer** is a visible share for MiniLM-class models: WordPiece and unigram are written
   allocation-free with the vocabulary in a trie or hash built at load.
5. **Exact-duplicate texts** in one request or across requests: an LRU keyed by the token ids
   returns the stored vector. Lossless, off the critical path, sized by an option, off when 0.

## 5. Accuracy gate

A model is listed when, on the same file:

1. **Tokenizer**: identical ids to llama.cpp's `llama-tokenize` on the fixture strings (the
   existing 22 plus accented, CJK, and long-word cases for WordPiece).
2. **Vectors**: cosine similarity to llama.cpp's embedding on 200 texts of mixed length, with the
   per-format bound set from measurement (the way KL is today), and ≥ 0.9999 against Peregrine's
   own exact scalar path. Both engines quantise activations differently, so the bound is stated
   against the truth as well: Peregrine must be no further from a full-precision reference (the
   same weights in F16, `dense_2`/`dense_3` present) than llama.cpp is, within 1e-4. Measured for
   `gemma-embedding` (S4.4, 200 texts, both engines at their Q8_0 path, reference = llama.cpp on
   the F16 conversion): llama.cpp Q8_0 vs the reference is min 0.99801 / mean 0.999473 at 768 d
   and min 0.99840 / mean 0.999555 at 256 d; Peregrine Q8_0 vs the reference is min 0.99875 /
   mean 0.999473 and min 0.99890 / mean 0.999552; Peregrine F16 vs the reference is min 0.99998.
   The per-format bound for this family is therefore **min 0.998 and mean 0.9995 at 768 d, min
   0.9985 and mean 0.9996 at 256 d** against llama.cpp's Q8_0, encoded with the reference vectors
   in `crates/peregrine-models/tests/fixtures/embeddinggemma/`. The design's original 0.999 was an
   F16-class bound: a Q8_0 engine's own distance from the unquantised model is about 0.998 at the
   minimum, for llama.cpp as much as for Peregrine. F16 models keep the 0.9999-class bound.
3. **Task**: an MTEB subset (retrieval: NFCorpus, SciFact; STS: STSBenchmark; one classification
   and one reranking set) within 0.5 point of llama.cpp and of the model's published F32 number.
4. **Decisions**: the rendered prompt and its token ids identical to llama.cpp's on the fixture
   set; System One probabilities, scores and noul values within 5e-3 and confidences within
   1e-2 of `llama-server`'s on the same file (measured on the tiny test file: 2.5e-3 at most,
   about the size of the difference between Peregrine's own kernel paths) (first on `ggml-org/tinyopenjev-for-testing-gguf`, the file
   llama.cpp's own tests use, then on each real model listed); the chosen option identical except
   at near-ties; classifier label argmax identical except at near-ties.
   **Rerank, as built (S4.6b)**: 1e-3 was an F16-class number and no Q8_0 encoder can hold it, by
   either engine. On 320 pairs the two engines' Q8_0 scores differ by 0.0107 mean / 0.186 max,
   while each engine's *own* Q8_0 scores are 0.0158 (llama.cpp, max 0.222) and 0.0174 (Peregrine,
   max 0.282) from its FP16 run: the quantisation moves the scores further than the engines differ.
   At the same precision (FP16 vs FP16) the difference is 0.0047 mean / 0.0688 max, and it is one
   thing — llama.cpp's GELU is the tanh approximation through an f16 table, ours is the exact erf
   form (ADR 0003) — because substituting llama.cpp's own GELU into our pass brings it to 0.00056
   mean / 0.0067 max. The gate for this family is therefore: pair token ids identical, document
   order identical outside near-ties (scores within 1e-2), and scores within the measured 2e-2
   mean, 0.3 max.
5. **Invariance** (§2.3): bit-identical alone and batched, on every kernel tier, with goldens.

Synthetic encoders (tiny random `bert`, `modern-bert`, `gemma-embedding` files with every head)
join the synthetic models in CI, checked against the exact reference forward.

## 6. Steps

Each is one PR unless noted, checked against llama.cpp on a real file before the model is listed.

| ID | Item | Needs |
|---|---|---|
| S4.1 | `Want::Pooled` on causal models: last / mean pooling, normalisation, `dimensions`; `/v1/embeddings`; `peregrine embed`; Qwen3-Embedding 0.6B against llama.cpp | |
| S4.2 | Benchmark harness embedding / rerank / System One modes and competitor baselines on bluefin and the desktop (llama.cpp, TEI, fastembed, OpenVINO); the targets in §4.2 confirmed or corrected | |
| S4.3 | System One, part 1: decision metadata, `/v1/systemone`, the answer math, `Want::Labels`, the state fork, the **openjev** renderer; parity with `llama-server` on the tinyopenjev test file. Then the other label-logit types, one renderer each: **lev** (two-variant choice, 9-point noul), **nimble**, **pplx-decider**, **h2o-lightning** | S4.1 |
| S4.4 | Bidirectional forward (§2.2) and `gemma-embedding`: EmbeddingGemma 300M | S4.1 |
| S4.5 | WordPiece and unigram tokenizers against `llama-tokenize` | |
| S4.6a | `bert` family: MiniLM, bge, e5, bge-m3 on `/v1/embeddings`, `peregrine embed`; the WordPiece and XLM-R vocabularies | S4.4, S4.5 |
| S4.6b | the `cls` head: `Want::Head`, bge-reranker-v2-m3 and bge-reranker-base on `/v1/rerank`, `/v1/classify`, `peregrine rerank` | S4.6a |
| S4.7 | `modern-bert` and labelled classifier heads | S4.6 |
| S4.8 | `nomic-bert` (+ MoE), `jina-bert-v2/v3` | S4.6 |
| S4.9 | Speed: F32 / F16 GEMM blocking, the `n × n` attention tiles, small-model thread choice, the duplicate cache; every §4.2 target met on both x86 hosts, then NEON | S4.2, S4.6 |
| S4.11 | System One, part 2: **kev** (the `cls.output` pointer head) and **vllm-d3** (non-causal Qwen 3.5 attention blocks, whole prompt in one pass) | S4.3, S4.4 |
| S4.12 | System One, part 3: **laya** and Julia-1 on `modern-bert` with the decision head | S4.7 |
| S4.13 | Generative rerankers: the rank read-out (`cls.output` on the last row), `/v1/rerank` and `peregrine rerank` with Qwen3-Reranker | S4.3 |
| S4.14 | `mistral3` embedders (§2.6): the Tekken pre-tokenizer against llama.cpp; the Mistral 3 block and `forward_texts` in the dense family; Nemotron 3.5 Embed 1B against llama.cpp on the real file (Q8_0, Q4_K_M, F16), goldens per tier, speed on both x86 hosts | S4.4 |
| S4.10 | Several models in one server (`serve -m a -m b`, routed by `model`): a chat model, an embedder and a decision model in one process sharing one pool | S4.3 |

## 7. Decisions

1. **"Decision model" means a native System One model** (§2.5), served on `/v1/systemone` with
   llama.cpp's request and response shapes; rerank and classify are the two related shapes beside
   it. A chat model prompted to classify is not one. Token-level outputs (NER, ColBERT) and image
   input are not in this stage.
2. **No lossy shortcut by default** (ADR 0003): a file runs in its own format; Peregrine does not
   re-quantise an F32 encoder to int8 at load unless an option asks for it and an accuracy report
   backs it.
3. **Encoders bypass the KV cache**; causal embedders and deciders keep it, because the prefix
   store is their main speed-up.
4. **Probabilities are over the options only**, scaled by the file's temperatures, exactly as
   llama.cpp computes them, so a client gets the same numbers from either engine.
6. **No Jinja engine.** A decision type is supported when its renderer is written and matches
   llama.cpp byte for byte on the fixtures; an unknown type or an unrecognised template is refused.
5. **Multi-model serving is last** (S4.10): every earlier step is useful with one model per
   process, and it needs a decision on how models share the thread pool.

## 8. S4.6b as built: the classifier head, rerank and classify

The second half of S4.6 is on top of the family: `crates/peregrine-models/src/bert/mod.rs` reads
the head (`cls`, `cls.bias`, `cls.output`, `classifier.output_labels`, `classifier.activation`,
`classifier.pooling_type`) and computes its row after the last layer for the texts that ask for
it; `Want::Head` is the slot, `Architecture::forward_texts` takes a `TextWant` per text
(`Pooled` or `Head`), and the row lands in `State::head_out`, sized once with the rest of the
scratch so a head pass allocates nothing. The head is a property of the tensors, not of
`<arch>.pooling_type`: llama.cpp's converter never writes `pooling_type = 4` for a `bert` file,
and llama-server is told `--reranking` instead.

Four decisions the note left open, now settled:

1. **`relevance_score` is the raw logit**, in llama.cpp's shape, at four paths (`/v1/rerank`,
   `/rerank`, `/v1/reranking`, `/reranking`); `return_documents` adds the text, and `top_n`
   defaults to every document. The TEI request shape (`texts`) is refused by name: it answers a
   bare array, a second response shape this step does not need.
2. **`/v1/classify`** answers one entry per input, in input order, as §3 now states.
3. **A pair longer than the position table** is refused with the limit, or truncated under
   `"truncate": true` — the embeddings rule, because a rerank pair is one text.
4. **A generative reranker's `rerank` chat template** is refused by name; it is a prompt, not a
   pair, and it is S4.13.

What was measured on bluefin (4 threads): 320 pairs in eight languages, ids identical and order
identical to llama.cpp for bge-reranker-v2-m3 Q8_0 and bge-reranker-base Q8_0 (the numbers in §5);
rerank throughput level with llama.cpp at 123-token pairs and 5-6 % ahead at 449-token ones
(`docs/reference/bert.md`). No published `bert` GGUF with `classifier.output_labels` turned up
(searches for DistilBERT/XLM-R classifiers found embedding conversions only, whose files carry no
`cls` tensors), so `/v1/classify` is covered end to end by the synthetic labelled file and the
server tests, not by a real-file gate.
