# S4 Design: embeddings and decision models

- Status: Proposed
- Date: 2026-11-10
- Target: a new roadmap stage, S4 (items S4.1 to S4.10)
- Inputs: `ARCHITECTURE.md` (the invariants), ADR 0003 (lossless by default), ADR 0006 (the
  architecture interface), ADR 0007 (one runtime), `docs/targets.md`
- Reference: llama.cpp **v0.9.0** (`cd959ba`). Its architecture table (`src/llama-arch.cpp`) lists
  `bert`, `modern-bert`, `nomic-bert`, `nomic-bert-moe`, `neo-bert`, `jina-bert-v2`, `jina-bert-v3`,
  `eurobert`, `gemma-embedding` and the keys `%s.pooling_type`, `%s.classifier.output_labels`,
  `%s.dense_2_feat_in/out`, `cls`, `cls.output`. Every model detail below marked *(transcribe)*
  must be read from llama.cpp at that tag by the implementing PR, not taken from this note.

## 0. Summary

Peregrine serves text generation. This stage adds the two other things a CPU box is asked to host
next to a chat model:

1. **Embedding models**: a text goes in, one vector comes out (`POST /v1/embeddings`).
2. **Decision models** ("system one": one fast forward pass, an answer instead of prose):
   - **rerank**: a query and N documents, a relevance score per document (`POST /v1/rerank`);
   - **classify**: a text, a score per label from a classifier head (`POST /v1/classify`), which is
     how guard, router and intent models ship;
   - **decide**: a prompt and a closed list of choices, the probability of each, from any
     generative model Peregrine already runs (`POST /v1/decide`). This covers generative guards
     (Llama Guard, Qwen3Guard), generative rerankers (Qwen3-Reranker) and yes/no or A/B/C routing.

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
- **Decisions share their prompt.** A reranker scores N documents against one instruction and one
  query; a guard or router runs one fixed system prompt on every request. The prefix store already
  reuses a shared beginning across slots, so the fixed part is computed once.
- A decision needs **one row of logits and no sampling**: the last prefill row's logits, restricted
  to the choice tokens, are the answer. No decode step runs.

## 1. Models in scope, in order

| Step | Family | Files it opens | New code |
|---|---|---|---|
| S4.1 | decoders already supported, as embedders | Qwen3-Embedding 0.6B / 4B / 8B (`qwen3`, last-token pooling) | pooling, the endpoint; no model code |
| S4.3 | decoders already supported, as deciders | Qwen3-Reranker, Llama Guard 3 1B, Qwen3Guard-Gen, any chat model | choice scoring, the endpoints |
| S4.4 | `gemma-embedding` | EmbeddingGemma 300M | bidirectional attention on the Gemma 3 family, mean pooling, the two dense layers |
| S4.5 | tokenizers | WordPiece (`bert`), SentencePiece unigram (`t5`, the XLM-R vocabulary) | two tokenizers |
| S4.6 | `bert` | all-MiniLM-L6/L12, bge-small/base/large-en-v1.5, e5, gte, bge-m3, multilingual-e5; bge-reranker-v2-m3 with its `cls` head | post-norm encoder, absolute positions, the classifier head |
| S4.7 | `modern-bert` | ModernBERT, gte-modernbert, mmBERT, Granite Embedding R2, and their fine-tuned classifiers (`classifier.output_labels`) | alternating local and global attention, the labelled head |
| S4.8 | `nomic-bert`, `nomic-bert-moe`, `jina-bert-v2/v3` | nomic-embed-text v1.5 / v2, jina-embeddings v2 / v3 | per family, on demand |

`neo-bert`, `eurobert`, DeBERTa (Prompt Guard 2; llama.cpp does not read it either) and the ColBERT
multi-vector output are out of scope for this stage.

## 2. The runtime

### 2.1 One runtime stays one runtime (ADR 0007)

An embedding or decision request is a **slot** in `serve::Engine` like a chat request. It differs in
what it asks for at the end:

```rust
pub enum Want {
    Tokens,                         // today: sample and stream
    Pooled(Pooling),                // an embedding: Last | Mean | Cls, then the model's own head
    Choice { ids: Vec<Vec<u32>> },  // a decision: log-probability of each choice
    Labels,                         // a classifier head's logits
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
  logit (a reranker; the score is the sigmoid) or one per label (`classifier.output_labels`; the
  scores are the softmax, or sigmoids when the file says multi-label).

### 2.5 Choice scoring (decide, generative rerank)

Input: a rendered prompt and `k` choices, each a token sequence.

- All choices one token long (yes/no, safe/unsafe, A/B/C): one prefill, read `k` logits from the
  last row, log-softmax over the **full vocabulary**, report each choice's probability and also
  the probabilities renormalised over the choices. No extra pass.
- Longer choices: the choices are `k` slots sharing the prompt through the prefix store; each scores
  its own continuation by teacher forcing (sum of token log-probabilities). One more batched pass
  of the continuation rows; the prompt is computed once.
- Generative rerankers: the prompt is the model's rerank template (`tokenizer.chat_template.rerank`
  when the file has one; Qwen3-Reranker's documented template otherwise *(transcribe)*), the score is
  `p(yes) / (p(yes) + p(no))`. Instruction and query come before the document in that template, so
  the prefix store shares them across the N documents.

## 3. API

Shapes follow what clients already speak; unknown fields are rejected, as in chat.

- `POST /v1/embeddings` (OpenAI): `input` (string, array of strings, array of token ids, array of
  arrays of token ids), `model`, `dimensions`, `encoding_format` (`float` | `base64`), and the
  usual `usage`. A text longer than the model's context is an error naming the limit, not a silent
  truncation; `truncate: true` opts into truncation.
- `POST /v1/rerank` (Jina / Cohere shape, also served at `/rerank`): `query`, `documents`, `top_n`,
  `return_documents`; `results: [{index, relevance_score}]` sorted by score.
- `POST /v1/classify`: `input` (string or array); `results: [{label, score, scores: {label: p}}]`.
- `POST /v1/decide`: `messages` (or `prompt`), `choices: [string]`;
  `{choice, index, probabilities: [..], logprobs: [..]}`.
- CLI: `peregrine embed`, `peregrine rerank`, `peregrine classify`, `peregrine decide`, each with
  `-m` (and `hf:`), reading stdin when no text is given, `--json` for machine output.
  `peregrine inspect` says what a file is for (generate, embed, rerank, classify).
- `serve` on an encoder file exposes only the endpoints the file supports; chat on an embedding
  model is a 400 that says so.

## 4. Speed

### 4.1 What is measured

`benchmarks/` gains an embedding mode: texts per second and tokens per second, for one text at a
time and for batches (1, 8, 32, 128), at token lengths 16, 128 and 512 (and 2048 where the model
allows), single client and 10 clients; p50 and p99 latency for one short text (the guard and
router case). Rerank: documents per second for 1 query × 32 documents of 128 and 512 tokens.
Decide: decisions per second and p50 latency with a 300-token fixed system prompt.

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
| Rerank and decide with a shared prompt | ≥ 2× llama.cpp (the prefix is computed once) |

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
2. **Vectors**: cosine similarity to llama.cpp's embedding ≥ 0.999 on 200 texts of mixed length
   (the engines quantise activations differently; the bound is set per format from measurement, the
   way KL is today), and ≥ 0.9999 against Peregrine's own exact scalar path.
3. **Task**: an MTEB subset (retrieval: NFCorpus, SciFact; STS: STSBenchmark; one classification
   and one reranking set) within 0.5 point of llama.cpp and of the model's published F32 number.
4. **Decisions**: rerank scores and choice probabilities within 1e-3 of llama.cpp's; label argmax
   identical on the fixture set except at near-ties.
5. **Invariance** (§2.3): bit-identical alone and batched, on every kernel tier, with goldens.

Synthetic encoders (tiny random `bert`, `modern-bert`, `gemma-embedding` files with every head)
join the synthetic models in CI, checked against the exact reference forward.

## 6. Steps

Each is one PR unless noted, checked against llama.cpp on a real file before the model is listed.

| ID | Item | Needs |
|---|---|---|
| S4.1 | `Want::Pooled` on causal models: last / mean pooling, normalisation, `dimensions`; `/v1/embeddings`; `peregrine embed`; Qwen3-Embedding 0.6B against llama.cpp | |
| S4.2 | Benchmark harness embedding / rerank / decide modes and competitor baselines on bluefin and the desktop (llama.cpp, TEI, fastembed, OpenVINO); the targets in §4.2 confirmed or corrected | |
| S4.3 | `Want::Choice`: `/v1/decide`, `/v1/rerank` on generative rerankers; `peregrine decide`, `peregrine rerank`; Qwen3-Reranker 0.6B and Llama Guard 3 1B against llama.cpp | S4.1 |
| S4.4 | Bidirectional forward (§2.2) and `gemma-embedding`: EmbeddingGemma 300M | S4.1 |
| S4.5 | WordPiece and unigram tokenizers against `llama-tokenize` | |
| S4.6 | `bert` family and the `cls` head: MiniLM, bge, e5, bge-m3; bge-reranker-v2-m3 on `/v1/rerank`; `/v1/classify` | S4.4, S4.5 |
| S4.7 | `modern-bert` and labelled classifier heads | S4.6 |
| S4.8 | `nomic-bert` (+ MoE), `jina-bert-v2/v3` | S4.6 |
| S4.9 | Speed: F32 / F16 GEMM blocking, the `n × n` attention tiles, small-model thread choice, the duplicate cache; every §4.2 target met on both x86 hosts, then NEON | S4.2, S4.6 |
| S4.10 | Several models in one server (`serve -m a -m b`, routed by `model`): a chat model, an embedder and a guard in one process sharing one pool | S4.3 |

## 7. Decisions

1. **"Decision model" means the four shapes in §0** (rerank, classify, decide, plus embeddings
   beside them). Token-level outputs (NER, ColBERT) are not in this stage.
2. **No lossy shortcut by default** (ADR 0003): a file runs in its own format; Peregrine does not
   re-quantise an F32 encoder to int8 at load unless an option asks for it and an accuracy report
   backs it.
3. **Encoders bypass the KV cache**; causal embedders and deciders keep it, because the prefix
   store is their main speed-up.
4. **Probabilities are over the full vocabulary**, with the choice-renormalised values beside
   them, so a caller can tell "confidently B" from "none of these".
5. **Multi-model serving is last** (S4.10): every earlier step is useful with one model per
   process, and it needs a decision on how models share the thread pool.
