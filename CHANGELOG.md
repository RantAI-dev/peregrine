# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, minor versions may break things.

## [Unreleased]

### Added

- **The `bert` family: MiniLM, BGE, E5, GTE and bge-m3 embeddings (roadmap S4.6, part a)**:
  `architecture = "bert"` is the post-norm encoder every encoder-only embedding file in
  llama.cpp's format ships as — token + token-type + learned absolute-position embeddings and one
  LayerNorm, then per layer q/k/v with biases, full bidirectional attention, the output
  projection, a residual and a LayerNorm; the feed-forward (up, the exact GELU, down), a
  residual and a second LayerNorm. There is no pre-norm: each layer's output *is* its normalised
  value. The graph, the tensors and the metadata are transcribed from llama.cpp v0.9.0
  (`src/models/bert.cpp`, `llama-model.cpp`, `llama-graph.cpp`, `conversion/bert.py`) in
  `docs/reference/bert.md`. The family is an embedding model: the engine schedules whole texts
  with no KV cache, `pooling_type` is the file's own (`cls` for BGE/bge-m3, `mean` for
  MiniLM/E5/GTE), and the position table is the text limit — the XLM-R converter chops the
  `pad+1` offset off the table, so positions are 0-based and `bge-m3`'s 8192 rows are 8192
  tokens. Every projection goes through the shared GEMM, so F32/F16 files use the packed float
  kernel and Q8_0 the int8 tiles, each distinct activation quantised once (S4.9a); biases are
  added after the matmul. New, defined kernels in `peregrine-kernels::vmath`: `erf` (the
  Maclaurin series with range splitting, in f64, measured at max 1 f32 ulp against an
  independent oracle over `[−6, 6]`), `gelu_erf` (the model's own exact GELU — llama.cpp
  computes it through an f16 lookup table) and `layer_norm_row(s)` (mean and variance in f64 in
  index order, one rounding of the scale, then the f32 scale-and-shift, unfused); the AVX2
  bodies are S4.9h's, and the definitions are what must not change. On 200 mixed-length texts
  (16–428 tokens) the token ids are identical to llama.cpp's on all 200 for every file, and
  against the **fp32 PyTorch reference** (sentence-transformers, `max_seq_length = 512`)
  Peregrine reaches cosine min/mean 0.9999996/0.9999998 on MiniLM F16, 0.9991371/0.9994906 on
  MiniLM Q8_0, 1.0000000/1.0000000 on bge-small F32, 0.9997426/0.9998402 on bge-small Q8_0 and
  0.9968921/0.9991315 on bge-m3 Q8_0 — better than llama.cpp's own Q8_0 in minimum and mean
  on MiniLM and bge-m3 and in mean on bge-small (its minimum there is 1.3e-5 lower, inside the
  1e-4 slack). A synthetic `bert` file (with
  the WordPiece vocabulary and a second with the XLM-R unigram one) is checked bit for bit
  against an independent scalar reference forward, is batch-invariant (2/7/64 texts, mixed
  lengths, 1/2/4/8 threads, both admission orders), allocates nothing in a whole-text pass, and
  has a golden on each pinned x86 tier; server and CLI tests cover the endpoint, `cls`/`mean`
  pooling, Matryoshka truncation and the chat refusal. `peregrine run` on a bidirectional
  encoder now names it as an embedding model instead of complaining about a missing chat
  template. The real-file fixtures are under `tests/fixtures/bert/` with an ignored test.
- **A decision computes only its label rows of the output head (S4.9d)**: a `Want::Labels` slot used
  to evaluate the whole vocabulary row of its last prompt position and gather the handful of token
  ids it reads. The head now evaluates those ids alone: one dot product of the already-quantised
  hidden row per label, through the same kernel the full path uses for that weight row and format on
  the current tier — the per-row kernel (or the eight-row Q5_0/Q8_0 one for a single activation),
  the eight-row k-quant tile, the sixteen-row ternary tile or the sixteen-row packed float panel,
  taking the one output from the tile or panel that contains the row. Every output of those kernels
  is independent of its neighbours, so a label entry is the full row's entry bit for bit; no format
  falls back, and only a LittleBit sidecar head keeps the full row. Chat, embeddings and speculation
  are untouched. Tested in every head format the synthetic generator writes (F32, F16, BF16, Q4_0,
  Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4, and the ternary PQ2_0/PTQ1_0 of a
  rotated hybrid), tied and untied, on `native`, `PEREGRINE_ISA=avx2` and `PEREGRINE_ISA=scalar`,
  with labels at the first, last and arbitrary vocabulary indices, duplicates, and 1, 3 and 255
  labels; the existing label invariance, prefix, fork and beside-other-slots tests are unchanged.
  On a 248,320-token vocabulary the head stage falls from 1.2 ms to ≤0.01 ms on one thread and from
  0.65 ms to ≤0.01 ms on four (tinyopenjev), and from 11-12 ms to 0.02 ms on Qwen3.5-2B Q4_K_M
  (four threads, repack off); `/v1/systemone` on a ~300-token state gains about 3-11% in
  questions/s at 1, 3 and 10 questions.

- **Token-budget passes and prompt-sized caches for embedding and decision work (S4.9b)**: a step
  now fills a weight pass up to a row budget (`PEREGRINE_PASS_ROWS`) with pooled (embedding) and
  label (decision) slots, taking whole short texts and pieces of a long one from as many waiting
  slots as fit, instead of carrying one prefill chunk in total: 32 short texts are 2 passes at a
  256-row budget rather than 32. The budget is derived from the machine and the file (the L2 and
  the file's bytes per weight; the measured ridge), overridable, and reported by `peregrine tier`.
  The matmul driver splits a batch wider than its 128-row kernel scratch into row blocks, so the
  budget is not capped. A bidirectional encoder packs whole texts up to the same budget.
  A pooled or label slot's cache — and each forked follower's — is sized to its own prompt,
  rounded to whole prefill chunks, not to the model's context (a 16-token text on a 0.6B model at
  context 8192: 896 MiB before, 1.75 MiB after; a chat slot keeps the context). Embedding texts and
  decision questions get their own admission: a memory budget for their prompt-sized caches
  (`PEREGRINE_WORK_CACHE_MB`, 512 MiB) and a slot cap, not `--slots`, so a 128-text request is
  admitted whole on a `--slots 2` server, never consumes a chat slot and never waits for one.
  Chat slots keep today's chunk and fairness rules; when a decode row shares a pass, embedding work
  is bounded to a quarter of a prefill chunk, so a chat's token rate beside two embedding streams
  is 1.4x lower instead of 7.4x (main). A text's vector and a question's label logits are
  bit-identical for budgets 1, 16, 64, 256 and the default, at 1/2/4 threads, alone and in batches,
  in decision groups and beside chat slots; measured on bluefin (tables and raw JSON in
  `benchmarks/results/bluefin-embed-2026-11-11/`).

- **A required hygiene job (`ci`)**: `scripts/hygiene.sh` fails a tracked credential file, a
  credential-shaped string in a tracked text file, and (when `HYGIENE_BANNED_PATTERN` is set) a banned
  name, on the tree and on every commit a pull request adds; `scripts/hygiene-selftest.sh` proves each
  check and `ci-ok` now requires the job.

- **Decision questions fork one evaluation of their shared state (S4.9j, part 1)**: a cache snapshot
  now carries a hybrid model's per-layer Gated DeltaNet state and convolution history
  (`KvSnapshot::rec`), restorable only at the snapshot's own length, and `KvCache::fork_from` copies
  everything a later row reads — attention K/V rows `0..p`, each recurrent layer's state after row
  `p−1`, the convolution ring — from one cache into another at position `p`, dense and hybrid alike.
  `Engine::admit_labels_group` admits a decision request's questions as one group: the longest common
  prompt beginning is forwarded **once** in the first question's slot, its state forked into every
  sibling at that position, and the tails (different lengths, several slots' rows in one pass) then
  share weight passes; a request wider than the engine is admitted in several groups, each forked
  from one evaluation (or a prefix-store restore) of the beginning. `/v1/systemone` and
  `peregrine decide` use it. A forked question is bit-identical to the same question computed alone:
  tested for groups of 2/3/10 with different tail lengths, the fork at 1 token, at a prefill chunk
  boundary, mid-chunk and one row before the end, in the f16 and f32 caches, at chunks 1/16/64 and
  1/2/4 threads on synthetic `llama` and `qwen35` files, and beside chat and embedding slots. Rows
  forwarded for a group are `prefix + Σ tail` (pinned by a count test). The prefix store keeps hybrid
  snapshots now, and a group's shared beginning is offered to it at the fork, so a repeated state (or
  a fixed system prompt) is evaluated once per process, bounded by the store's byte budget.
  `crates/peregrine-bench/examples/decide_fork.rs` measures the fork against admitting questions one
  by one on a real hybrid model. Also removes an accidentally committed `.env` (local credentials)
  and ignores it.

- **System One decisions end to end (S4.3c)**: `POST /v1/systemone` and `peregrine decide` serve
  native decision models of the `openjev` type. The request is the llama.cpp shape — a `state` and
  a non-empty `questions` object of `choice` (a non-empty criteria object, options in request
  order), `score` (2–10 levels, keys `"0"`… ) and `noul` (optional `false`/`true` descriptions,
  read `[true, false]`) — parsed with the order-keeping JSON value, so the order of the questions
  and of each criteria object is the order of the answers and options. Every question becomes one
  labels slot in the engine, so questions share weight passes; a request with more questions than
  slots queues the rest and still answers in order. The response is
  `{model, answers, usage:{input_tokens,
  output_tokens:0}}` with llama.cpp's answer fields (choice and probabilities, score with the
  legend as given and probabilities, noul) and round-tripping floats. Validation messages,
  statuses and the 400/501 bodies are llama.cpp's; media is a 501 (no image input yet) and a chat
  model answers the route with a 501. `GET /v1/models` reports `architecture.output_modalities`
  `["decisions"]` for a decision model and `["text"]` otherwise, and a decision-only model refuses
  chat. `peregrine decide -m <file|hf:ref> --state <text|@file|-> --questions <JSON|@file>` (or
  `--request @file`) prints one line per question or the API JSON with `--json`. A file whose
  decision type or `systemone` template this build does not render is refused at load, naming the
  type; the 52 option letters must each be one token.
- **System One decision-model groundwork (S4.3)**: the answer math, the decision metadata and the
  label read-out, with no endpoint yet. `peregrine_core::decision` transcribes llama.cpp v0.9.0's
  `format_answer`, `get_temperature` and confidence helpers in `f64` — `choice` (best option,
  probabilities, confidence), `score` (expected level, legend, probabilities, confidence) and `noul`
  (the probability of true; Lev's nine-point scale) — with the per-type confidence formulas
  (entropy for vllm-decision2/vllm-d3, `p_max` for lfm3-d1) behind an enum, and is checked against an
  independent Python implementation to 1e-12 on 45 random cases. `DecisionInfo::from_gguf` reads
  `<arch>.decision.type` (the sixteen known names), every `temperature`, every
  `score_bias.<n>.<i>`, `labels` and the presence and byte length of the named `systemone` template,
  with llama.cpp's validations and messages; `peregrine inspect` prints one `decision` line. The
  engine gains `Want::Labels` (`Engine::admit_labels`): a slot prefilled like a pooled one, never
  decoded, whose single emission carries its last prompt row's logits at the given token ids. A
  question after another with the same prompt prefix restores the cache from the prefix store and
  is bit-identical on dense models; a hybrid model's cache is not kept there (no recurrent state in
  a snapshot), so it recomputes. The label logits are bit-identical alone, after a sibling and in a
  batch, at chunks 1/16/64 and 1/2/4 threads on synthetic llama and qwen35 files, through both
  scheduling paths.
- **Groundwork for System One decision models** (roadmap S4.3): the prompt a native decision
  model reads. `peregrine_core::systemone` renders the `openjev` type's prompt for a `choice`,
  `score` or `noul` question about a state, and `peregrine_core::json` is the order-keeping JSON
  value that writes a non-string state, instruction or description exactly as llama.cpp's Jinja
  `tojson` does (key order kept, integers as integers, other numbers as Python's `repr`, llama.cpp's
  string escapes). There is no Jinja engine: the renderer is written from the model's template,
  refuses any other template, and is checked byte for byte against 48 prompts rendered by
  llama.cpp v0.9.0 itself; with the model file present the token ids match too, and the renderer,
  the tokenizer, a labels slot and the answer math together give llama.cpp's answers on all 48
  requests (the same option chosen, probabilities within 2.5e-3). No endpoint yet.
- **The bidirectional pass and `gemma-embedding`** (roadmap S4.4): EmbeddingGemma 300M as the
  first encoder. `Architecture::causality` and `Architecture::forward_texts` forward whole texts
  side by side — ragged rows, one `n × n` attention block per text, f32 keys and values with no
  KV cache — under llama.cpp's mask rule (a sliding layer is a symmetric `± n_swa/2` window, a
  global layer is unmasked). The engine admits only pooled slots for an encoder, takes whole
  texts up to the scratch budget, allocates no per-slot cache, and refuses chat with a 400 that
  says the model is an embedding model. `gemma-embedding` reuses the Gemma 3 blocks and adds
  llama.cpp's `dense_2`/`dense_3` pair, mean pooling over a text's own rows (f64 accumulator, one
  rounding) and `cls` pooling; `peregrine embed`, `peregrine inspect` and `POST /v1/embeddings`
  work unchanged. Synthetic encoders, an exact scalar reference forward, batch invariance and
  three kernel-tier goldens are in CI. Against llama.cpp's vectors on 200 texts, Peregrine's
  Q8_0 is no further from a full-precision F16 conversion of the same weights than llama.cpp's
  Q8_0 is (min 0.99875 vs 0.99801 at 768 d, means within 6e-7), and the per-format bound is
  measured and encoded with the reference vectors under `tests/fixtures/embeddinggemma/`.
- **WordPiece and SentencePiece-unigram tokenizers** (roadmap S4.5), the two vocabularies
  encoder models ship: `tokenizer.ggml.model = "bert"` (all-MiniLM, bge, e5, gte) and
  `"t5"` (T5 and the XLM-R vocabulary: bge-m3, multilingual-e5, bge-reranker). Both are
  transcribed from llama.cpp v0.9.0 (`llm_tokenizer_wpm`, `llm_tokenizer_ugm`), including
  WordPiece's lower-casing and one-pass accent stripping, its per-punctuation and
  per-CJK-ideograph word split, the `▁` word-start convention the converter stores, and
  the unknown-token fallback; and unigram's `precompiled_charsmap` normalisation, the
  Viterbi search over the vocabulary with its unknown-token penalty, and whitespace
  collapsing. The Unicode tables WordPiece needs are generated into
  `crates/peregrine-core/src/unicode_data.rs` by `scripts/gen-unicode-tables.py` (no new
  dependency; ADR 0002). The vocabulary is compiled into a trie at load, so
  `encode_into` allocates nothing per token: 12 MB/s and 3.4 M tokens/s on WordPiece,
  7 MB/s and 2.0 M tokens/s on the 250k-piece XLM-R vocabulary on the reference machine.
- `Tokenizer::encode_special` (one text with the file's special tokens as llama.cpp adds
  them for an embedding input: `[CLS] … [SEP]`, `<s> … </s>`, or the file's BOS/EOS
  prologue and epilogue for a byte-level BPE such as Qwen3-Embedding) and `Tokenizer::encode_pair`
  (a query and a document, formatted exactly like llama.cpp's `format_prompt_rerank`:
  `[CLS] q [SEP] d [SEP]` for BERT, `<s> q </s> d </s>` for XLM-R).
- Tokenizers are now checked against `llama-tokenize` on three real vocabularies in CI,
  without the model files: `tests/tokenize_encoders.rs` loads the extracted vocabulary
  (`Tokenizer::from_vocab_dump`) and compares 51 strings and 10 rerank pairs per
  vocabulary — accented Latin, combining marks, CJK, Korean, emoji and ZWJ sequences,
  200-character words, control characters, the literal `[CLS]`/`[SEP]`/`[MASK]`/<s>/
  </s> texts, and a 2000-character paragraph — plus llama.cpp's own detokenisation.
  The fixtures are generated by `scripts/gen-tokenizer-golden.py` and
  `crates/peregrine-bench/examples/vocab_dump.rs`. The 250,002-piece XLM-R
  vocabulary is too large to carry as a fixture (4.1 MB), so that vocabulary is
  checked when `bge-m3-Q2_K.gguf` is present under `models/` or `~/models/` and
  skipped with a notice otherwise.
- `crates/peregrine-bench/examples/tokbench.rs`: tokenizer MB/s and tokens/s.
- `peregrine support` and the README table list the two new tokenizers.
- **Embeddings on causal models (S4.1)**: `POST /v1/embeddings` and `peregrine embed`.
  A model file that declares `<arch>.pooling_type` — Qwen3-Embedding and the other decoder-based
  embedders — takes the OpenAI request shapes (a string, an array of strings, token ids, or arrays
  of token ids), `dimensions` (Matryoshka truncation before L2 normalisation) and `encoding_format`
  `float` or `base64`. Every text of one request is a slot in the same engine, prefilled beside
  chat requests and never decoded; vectors come back in input order and are bit-identical alone or
  in any batch. The pooling type (`last` or `mean`) and the appended EOS token are read from the
  file as llama.cpp reads them. `peregrine embed` prints one line of floats per text (or `--json`),
  reads stdin lines, and `peregrine inspect` reports the file's pooling type.
- `peregrine support` lists embeddings in the support table.
- **The float GEMM (roadmap S4.9f).** F32, F16 and BF16 matrices are packed at load into panels of
  sixteen output rows (`TensorInfo::fp_off`, a store of its own built by reading the file, so the
  matrix's mapped pages are never faulted in), and `MatPath::FloatPanels` multiplies them with a
  broadcast-form FMA kernel: 6×16 on AVX2 and 2×6×16 on the AVX-512 tier's extra registers, with a
  single-row GEMV for decode. Every output element is one sequential FMA chain over the input index,
  so a row's bits do not depend on the tile, the tail, the row block, the thread split or the vector
  width: AVX2, AVX-512 and the scalar `mul_add` reference agree bit for bit, and a float model gives
  the same tokens on the AVX2 and AVX-512 tiers. Measured on the Ryzen AI 9 HX 370, one thread: 25-27
  GMAC/s against the old float tile's 11-15 and the repacked Q8_0 tile's 29-36 (the FMA peak is 28).
  On Qwen3-Embedding 0.6B f16 (12 threads): prefill 159 → 237 tok/s, embedding throughput 1.5× at
  109 and 437 tokens and +55 % for 32-text batches; decode is unchanged. `PEREGRINE_FPACK=0` is the
  A/B switch back to the old tile. The load-time pack costs about 2 s for the 1.2 GB f16 file and
  +18 % resident memory until the store is cached (S4.9e); aarch64 keeps the old float path for now.
  The rest of the gap to llama.cpp on an f16 file is the **pass size**, not the kernel: at the
  128-row cap the two engines are within 2 % (290.1 against 289.6 tok/s on one 437-token text), and
  llama.cpp's 512-row ubatch is worth 1.08-1.13× — which is S4.9b's token-budget pass, not a kernel
  change. `/v1/embeddings` at 12 threads: f16 253.6 → 290.1 tok/s from chunk 16 to 128, Q8_0 flat
  (330.7 → 334.7, it was never weight-bound).

### Changed

- **The encoder's attention is a fused, cache-blocked strip with a defined arithmetic (S4.9g).**
  A whole text's bidirectional attention was a scalar, strict-order dot product per (query row,
  key), parallel over rows only — 35 % of an EmbeddingGemma pass at 128 tokens and 60 % at 512, and
  after S4.9a a larger share still. `peregrine_kernels::bidir` is the kernel now: one head's keys
  are transposed once per (text, KV head) into a panel with the keys along the vector lanes, so a
  lane holds one score and the head dimension folds inside it as one sequential FMA chain
  (`acc = fma(q[k], key[k], acc)`); scores are taken in 128-key blocks (an L1/L2-resident panel
  slice shared by the strip's rows); the window is an exact band, so a sliding layer computes only
  the keys inside each row's window; the weights go through `vmath::exp8` and its scalar twin; the
  denominator is an `f64` sum in ascending key order; and the value mix accumulates four query rows
  at a time in ascending key order, sharing one value load. A row's bits depend only on its own
  text: not on the strip it fell in, the strip size, the thread, the neighbours or the tier — the
  kernel tests hold the scalar twin and AVX2 equal on `to_bits` for head dims 16-256, n from 1 to
  600, six windows, GQA 1/2/4 and strips of 1/7/16/32/n, and the engine's batch/thread/admission
  matrix is bit-identical. `blocks/bidir.rs` schedules it (panel build in one fork-join per KV head,
  then row ranges re-chunked by text and strip of up to 32 rows), and `tests/support/embed_ref.rs`
  states the same arithmetic as the reference. Measured on H0 (Xeon E5-2680 v4 VM, 8 vCPU, AVX2, load
  average 4.5-7), EmbeddingGemma 300M Q8_0, one pass, 4 threads, median of 3: 128 tokens 257 → 195 ms
  (attention 98 → 17 ms, 38 % → 8.6 % of the pass), 512 tokens 2051 → 1177 ms (1386 → 429 ms,
  66 % → 33 %), 2000 tokens 11.8 → 5.0 s (9.0 → 2.0 s, 76 % → 39 %); 16 tokens is unchanged
  (26.5 → 26.0 ms). Through `/v1/embeddings` at 4 threads: 128-token text 2.35 → 3.45 texts/s,
  512-token 0.32 → 0.72, 2000-token 0.078 → 0.177, a batch of 32×128 2.40 → 3.50, 32×16 unchanged
  within the host's noise. `crates/peregrine-bench/examples/bidirbench.rs` reports the loop against
  the strip per thread: at EmbeddingGemma's head width (256) 3.0 → 21.4 M score-pairs/s at n = 128
  and 3.0 → 10.7 M at 512. The synthetic encoder's three tier goldens are re-recorded once (the FMA
  chain moves the vectors by about one ulp); attention is now bit-identical across tiers, and with
  the kernel forced to its scalar twin on an AVX2 machine the AVX2 golden still matches. The
  real-file accuracy gate still holds (mean 0.9994629 vs llama.cpp's 0.9994734 at 768 d, and our
  Q8_0 min against llama.cpp's Q8_0 improved from 0.99817 to 0.99910); its worst-text slack is
  widened from 1e-4 to 1.5e-3 because an int8-activation encoder path re-rolls a few texts by up to
  ~1.2e-3 when any arithmetic changes — the mean bound (1e-4) is the accuracy statement.
- **The encoder quantises each input once, and only the layouts its kernels read (S4.9a).**
  A Gemma 3 encoder layer filled the int8 activation of every projection separately, serially,
  while the matmuls ran in parallel: q/k/v read the same normed rows and gate/up read the same
  rows, so five of the fills were the same bytes again, and the two post-norms (whose output only
  a residual add reads) quantised for nothing. `linear_rows_shared` fills the group's activation
  once and runs the projections as one fork-join, as the causal families already did; the post
  norms use `norm_rows_f32`; the down projection reuses the activation `geglu_rows` produced. The
  per-256 activation layout (`qk`/`dk`/`bsums`, a second quantisation of the whole row) is built
  only when some matrix of the model reads it (`Format::reads_k_layout`), which a Q8_0/F16 model
  such as EmbeddingGemma never does. Every vector is bit-identical: the encoder's three tier
  goldens, the exact scalar reference forward and the batch-invariance tests are unchanged, and
  the Q8_0 model's vectors hash the same as `main`'s on all four measured text shapes. Measured on
  H0 (Xeon E5-2680 v4 VM, 8 vCPU, AVX2), EmbeddingGemma 300M Q8_0 through `/v1/embeddings`, median
  of five before/after rounds at 4 threads: 16-token text 21.7 → 33.9 texts/s, 128-token 2.49 →
  3.33, 512-token 0.37 → 0.46, a batch of 32×16 26.0 → 39.5, 32×128 2.06 → 3.49 (1.24-1.69×).
- **The tokenizer finds special tokens with a first-byte index instead of testing every special
  at every byte (S4.9c).** The SentencePiece and byte-level BPE paths ran `starts_with` for each
  of a vocabulary's control/user-defined/unknown tokens at every position — 163 of them on
  EmbeddingGemma, 87 µs per token. `SpecialIndex` keys them by first byte and tests a byte's
  candidates in `specials` order (longest text first, then lowest id), which is exactly the order
  the per-special scan took the first match in, so the ids do not change; the "next special"
  search is one bitmap test per byte. 87.9 → 1.17 µs per token on a 1 MB text (75×) with that
  vocabulary. A differential test keeps the old scan as its reference and compares 4000 random
  strings built from the specials' texts, their prefixes and their overlaps on the BPE and
  SentencePiece paths.

## [0.1.2] - 2026-11-10

### Fixed
- **The server no longer exits 30 seconds after it starts.** The drain timeout in `peregrine serve`
  wrapped the whole serving future instead of the shutdown, so a server nobody had asked to stop
  exited 30 s after it began listening. The timeout now starts only when a shutdown signal
  (`SIGTERM`, Ctrl-C, or a `SIGHUP`/`upgrade` handover) begins the drain; `serve`,
  `serve --background`, `status`, `stop` and `upgrade` otherwise behave as before.
- **The first `hf:` download no longer breaks the command that made it.** On a cold model cache,
  the download pre-flight opened the partial file through the normal GGUF reader, which read the
  engine options and latched them before the command could set its own: `peregrine run` exited with
  "engine options were already set or read", `peregrine serve` panicked on the same line, and
  `peregrine bench` silently measured with environment-default options instead of its tuning
  profile. The pre-flight now opens the file with a plain read-only map and runs only the
  metadata/build compatibility check, so it reads no process options; `run`, `serve`, `bench` and
  `tune` now fail loudly if their options cannot be initialised. A warm cache was unaffected.

## [0.1.1] - 2026-11-10

### Added
- **`peregrine serve --background`**, with `peregrine status` and `peregrine stop`: the server
  detaches into its own process group, logs to `serve-<port>.log` (or `--log-file`), and writes a
  lifecycle state file under `$PEREGRINE_STATE_DIR`, `$XDG_STATE_HOME/peregrine` or
  `~/.local/state/peregrine`. The command returns only after the server is listening (bounded by
  `--startup-timeout`, default 300 s), and a second background server is refused while one runs.
- **`peregrine upgrade`**: resolves the newest GitHub release (or `--version TAG`), downloads the
  `linux-x86_64`/`linux-aarch64` archive, verifies it against `SHA256SUMS`, runs the extracted
  binary's `--version` **before** replacing anything, and installs it with a copy-beside + atomic
  rename. `--check` reports without installing; `--base-url`/`PEREGRINE_RELEASE_BASE` point at a
  mirror; `--exe` chooses the target; `--no-restart` installs without touching the running server;
  cargo builds under `target/` are refused without `--force`.
- `scripts/install.sh` is POSIX `sh` (no `pipefail`, `#!/bin/sh`), so `curl ... | sh` works on hosts
  where `sh` is dash; it previously needed `bash`.
- **Vanity download host**: the public one-liner is now
  `https://peregrine.rantai.dev/install.sh`, a Cloudflare redirect to
  `releases/latest/download/install.sh` (the `get.rantai.dev/lakehouse` pattern); the files stay on
  GitHub. The release workflow uploads `install.sh` as a release asset, `peregrine upgrade` tries
  `peregrine.rantai.dev/releases/download` first and falls back to GitHub, and
  `PEREGRINE_RELEASE_BASE` / `PEREGRINE_RELEASE_LATEST` override both for mirrors and tests.
- **Zero-downtime replacement**: a background server binds with `SO_REUSEPORT` (also exposed as
  `--reuse-port`), so `upgrade` starts the new binary with the same `serve` arguments beside the old
  one, waits for the new lifecycle state, then sends the old pid `SIGTERM`. The old process stops
  accepting, drains in-flight requests (30 s cap) and exits, while the new one serves; nothing is
  refused during the handoff. Verified with 300 requests across the swap: zero failures.
- **`SIGHUP` self-handoff**: `kill -HUP <pid>` re-executes the binary on disk with the same `serve`
  arguments and `SO_REUSEPORT`, waits for the replacement to report ready, then drains — so a local
  rebuild, a new tuning profile or a replaced binary is picked up without refusing a connection. The
  inode guard makes a SIGHUP on an unchanged binary a no-op (no restart loop), and a failed
  replacement leaves the old server serving. A foreground server needs `--reuse-port`; background
  servers already have it. Verified with 200 requests across a SIGHUP swap: zero failures.
- **Graceful shutdown** on `SIGTERM` and `Ctrl-C`: stop accepting, drain, remove the state file, exit
  0. `peregrine stop` waits for the drain (40 s cap) and cleans a stale state file.
- `docs/service.md`: background mode, state/log locations, upgrade semantics, the `SO_REUSEPORT`
  requirement, and a systemd unit for boot-time start.
- **`peregrine run`**: one prompt in, one reply out, streamed to stdout, with no server and no
  client. It reuses the same session, chat template and sampling path as `serve`; `--json` prints a
  single object with the text, prompt/completion token counts, prefill and decode tokens per second,
  time to first token and wall time. With no prompt argument it reads stdin, so
  `echo "..." | peregrine run -m model.gguf` works. `--max-tokens`, `--temperature`, `--top-p`,
  `--top-k`, `--min-p`, `--seed`, `-c`, `-t`, `--system`, `--chat-template`, `--kv`, `--opt`,
  `--profile` and `--no-profile` mirror the server's knobs.
- **`hf:` model references**, accepted by every command that takes `-m` (`run`, `serve`, `inspect`,
  `bench`, `tune`): `hf:<namespace>/<repo>/<file>[@<rev>]` downloads the file once into
  `$PEREGRINE_CACHE_DIR`, `$XDG_CACHE_HOME/peregrine/models/<repo>/<file>` or
  `~/.cache/peregrine/models/<repo>/<file>` and reuses it afterwards. The repository (and a pinned
  revision) is part of the cache path, so two repos that both publish `model-Q4_K_M.gguf` cannot
  collide. Downloads resume from a `.part` file, and only the first 16 MiB are fetched
  before `Gguf::open_header` (new) runs the build's compatibility check — a model this build cannot
  run is refused before the weights arrive, and the partial file is removed. `PEREGRINE_HF_BASE`
  points the resolver at a mirror, and `HF_TOKEN` is sent as a bearer token for gated or private
  repositories.
- `scripts/install.sh`, the `curl | sh` installer: it detects `linux-x86_64` or `linux-aarch64`,
  downloads the release archive and its `SHA256SUMS`, verifies the checksum before writing anything,
  and installs to `~/.local/bin` (or `--prefix DIR`). The binary is renamed into place, so upgrading a
  running server never hits `ETXTBSY`. `--version TAG` pins a release. The quickstart's first step is
  now one command instead of "download this URL with `curl -fLO` and `mv` the binary yourself".
- `PEREGRINE_BASE_URL` overrides the download base for the installer, for mirrors and for testing the
  installer against a local release directory.

### Changed
- **The startup calibration now runs by default in `peregrine serve`**, when no tuning profile exists
  and neither `decode_threads` nor `spin_us` is pinned. It measures the host in under 200 ms, prints
  what it chose, and reports it through the existing `/metrics` gauges; `PEREGRINE_CALIBRATE=0` turns
  it off. `PEREGRINE_CALIBRATE` now accepts `0`/`off`/`false` and `1`/`on`/`true`, and any value
  other than the off-words turns it on. `peregrine bench` and `peregrine tune` still do not calibrate
  unless asked, so a published number keeps recording the settings it was handed (S2.6 decision 6).
- `docs/dev/benchmarking.md`, `docs/quickstart.md`, `docs/design/s2.6.md` and the README's quick
  start describe the installer and the new calibration default.

## [0.1.0] - 2026-11-10

### Added
- A release pipeline for the first release: `scripts/release/build.sh` builds a stripped, portable
  archive per target (`peregrine-vX.Y.Z-linux-x86_64.tar.gz` / `-linux-aarch64.tar.gz` with `LICENSE`/`NOTICE` and a SHA256SUMS
  line), and a `release` workflow on tags `vX.Y.Z` runs the full test suite on both Linux runners,
  asserts the x86-64 binary keeps the baseline ISA with AVX2/AVX-512 picked at run time (no
  `-C target-cpu=native`), and uploads the archives and checksums to the GitHub release with notes
  generated from PR labels. The release step itself — versioning, CHANGELOG conventions, the
  AVX2-host (H2) full-suite gate on every release candidate (S0.5) — is documented in
  `docs/dev/releasing.md`.
- **Bonsai 2 on the aarch64 NEON tier** (S2.8 F4): the `PQ2_0` ternary kernels, the block
  Hadamard transform, the Gated DeltaNet recurrent step and the causal conv. `PQ2_0` is 128
  ternary weights in 34 bytes; the kernel extracts the four code planes with shifts and masks
  and runs eight `sdot`s per group against the activation's plane layout, so lane `l` is the
  16-value sum of block `l` and the `−Σq` offset is the reference's own `tls` fold. Decode runs
  two output rows per pass; prefill decodes a row once and meets four activations with it. Both
  keep the fold per lane and per group, so a row's bits are the same in the one-row kernel, the
  two-row decode kernel and the four-activation prefill kernel (asserted with `to_bits()`), and
  MTP speculation still changes no token. The Hadamard FWHT does strides 1–8 in registers over
  sixteen floats and is bit-identical to the scalar transform; the sign multiply is a four-lane
  multiply; the conv's inner accumulate is elementwise and bit-identical. The delta step uses
  two four-lane accumulators over the state row (tolerance-tested, like the AVX2 kernel). An
  `smmla` GEMM was prototyped and measured at 1.42× the `sdot` GEMM at eight activations, but its
  block-before-scale fold would make a row's bits depend on the pass width, so the GEMM stays on
  `sdot` and the tier is `neon-dotprod` for this format. On the 2B file at ten X925 cores,
  prefill went 14.8 → 404 tok/s and decode 11.0 → 140 tok/s, and the 4B's MTP decode 64.8 →
  80.7 tok/s; see
  `docs/design/s2.8-bonsai2.md` and `benchmarks/results/spark-arm-2026-11-10-bonsai2/`.
- `PEREGRINE_AFFINITY=1` pins every pool worker, and the caller (worker 0), to a logical CPU chosen
  from `/proc/cpuinfo`'s `(physical id, core id)` topology: a pool as wide as the physical cores gets
  one worker per core, and a wider pool fills the SMT siblings in a fixed order. Off by default, Linux
  only. On the reference machine (24 workers over 12 cores) a warm 512-token prefill gains 2-4 % (medians
  of in-process repeats over 3-4 processes, two runs: Llama-3.2-1B x1.040/x1.024, Qwen2.5-0.5B
  x1.026/x1.124, Bonsai-2-2B x1.044/x1.012), decode is unchanged, and the *first* prefill a process runs
  loses 6-9 % (`peregrine bench`, 9 alternating rounds), so it is a server knob and not a default.
  First-touch placement does not matter for these working sets on one socket; the mapping is not
  NUMA-aware and would need one core set per node on a multi-socket host. See `docs/dev/build-profiles.md`
  for the build-profile A/B it was measured with (S3.9).
- NEON `vmath` and activation quantisation on aarch64 (S2.8 F5/F6). `exp` is now a four-lane
  `exp4` with `exp8`'s constants, operation order and fused multiply-adds (bit-identical to the
  scalar twin `exp_scalar`), used by softmax (`exp_shift_sum`, `exp_scaled_sum`), SwiGLU, SiLU,
  `mul_silu`, `mul_sigmoid` and `swiglu_oai`; the attention value mix is a four-dimension NEON loop
  that is bit-identical to the scalar one; and `QuantAct::fill` takes `absmax`/`quantize32` on
  NEON (also bit-identical, rebuilding `round` as `trunc(|y|) + (frac ≥ 0.5)`). Micro-benchmarks on
  the X925 core: SwiGLU 2.34x, SiLU 2.77x, softmax exp 2.32x, activation quantisation 2.32x over
  the same binary with `PEREGRINE_ISA=scalar`. This replaces libm's `expf` on the tier with the
  engine's deterministic `exp` (the x86 tier's, issue #64), so all `aarch64-neon-dotprod` goldens
  move in the last bits and were re-recorded; the scalar tier's goldens are unchanged.
- NEON kernels for **Q6_K** on aarch64 (S2.8 F3): the repacked `Q6K8` tile as a GEMM and a GEMV,
  plus a per-row `sdot` kernel. The six-bit codes are a nibble from `ql` and two bits from `qh`,
  expanded with a mask, a shift and an OR before each `sdot`; the run's offset is applied once per
  sixteen weights against the activation's run sum, and the per-super-block fold is the scalar
  reference's own expression. Unlike the Q5_0 tile, this one is **bit-identical to the scalar
  kernel**, so no golden was re-recorded (the Q6_K models are byte-for-byte unchanged). The
  interleaved activation gained `bs16`, the per-16 run sums the kernel needs. On the DGX Spark
  (X925, one thread) the tile reaches 34.9 GMAC/s in GEMV and 52.8 in the four-row GEMM, against
  5.8 for the scalar row; Llama-3.2-1B prefill on the X925 cluster goes 203 -> 394 tok/s and
  decode 68.5 -> 101.3 against the Q5_0-only tree, closing the last weight-format gap to llama.cpp.
- A NEON kernel for **Q5_0** on aarch64 (S2.8 F1): the repacked `Q5_08` tile on `sdot` (GEMM and
  GEMV), a per-row `sdot` kernel, and an eight-row decode kernel. The tile's sixteen nibble bytes
  are one `sdot` per half of the block, and the fifth bits are spread from the tile's per-window
  bytes with a per-byte shift; the integer sum and the `−16·Σa` bias are exact, and the fold is
  the tile reference's `(d·dk).mul_add`, so the tile, its GEMV and its GEMM agree bit for bit
  (tested at every width, and against the scalar reference for the per-row kernel). On this DGX
  Spark (10 × X925) the tile runs at 21.8 GMAC/s per core in GEMV and 34.5 in the four-row GEMM,
  against 10.8 for the scalar row. The aarch64 goldens for the Q5_0 models move in the last bits
  because the tile's fold is fused where the scalar per-row kernel's `+=` is not; they now equal
  the x86 tile reference by construction.
- A repacked eight-row store for **Q5_0** (`Q5_08`, 176 bytes = 8 × 22, size-neutral) and an AVX2
  GEMM/GEMV on it, for the small "Q4_K_M" files whose `d_model` is not a multiple of 256 and whose
  matrices llama.cpp therefore left at Q5_0. The tile groups the fifth bits by eight-weight window
  (one byte per row per window), so the activation keeps its ordinary per-32 layout: no interleave
  to build, no 256-alignment requirement, and the GEMV and GEMM are one code path, bit-identical per
  row. Gated by the existing `PEREGRINE_KREPACK` (on below the AVX-512 tier, off on it); rows past
  the last whole group of eight keep the per-row kernel. With the repack off, decode uses the
  eight-row Q5_0/Q8_0 GEMV above and Q8_0 does regardless (it has no repacked layout). On bluefin,
  5 alternating rounds against the tree #104 lands on: Qwen2.5-0.5B prefill 271.7 → 458.2 tok/s at
  512 and 206.5 → 299.8 at 4096, decode 71.6 → 83.6 and 58.8 → 65.5; SmolLM2-360M 299.1 → 436.5 /
  199.9 → 261.1 and 88.4 → 109.8 / 57.3 → 64.5; Llama-3.2-1B flat. Against llama.cpp v0.9.0 at its
  faster flash-attention setting per row, the small models are 1.43×/1.20× (0.5B prefill at 512
  and 4096) and 1.09×/1.27× (decode); the 360M is 1.27×/1.12× and 1.17×/1.42×.
- `PEREGRINE_PROF2=1` makes `peregrine bench` print a per-token phase breakdown (matmuls
  by projection, attention, norms, quantisation, RoPE, KV writes, residuals, SwiGLU,
  barrier wait); `=full` adds the attention subtimers. Off by default, no timers in the
  default path.
- An adaptive spin window (`PEREGRINE_ADAPTIVE_SPIN`, on by default). A pool thread samples its own
  run-queue delay from `/proc/thread-self/schedstat` every 8 ms; a smoothed reading of that delay is
  the gate. Below 35 % (quiet, and two runnable threads per core) the configured window is used
  exactly as before and no A/B runs. Above it the pool tests its own join wait: it holds the current
  window or parking for a few thousand joins, then measures one window on each and keeps the lower
  mean (with a 10 % margin, so it does not flap). This is outcome-based because the same run-queue
  reading maps to different winners on different days and models — a first version mapped the
  reading to a half-window middle rung, which lost to both the full window and parking and regressed
  Gemma-3-1B under load. `PEREGRINE_ADAPTIVE_SPIN=0` keeps the window fixed for measurement.
  `PEREGRINE_ADAPTIVE_LOG=1` prints the rung, the smoothed reading and each trial decision. On H0
  (Xeon E5-2680 v4, four threads pinned to four cores, shared) the medians of five alternating
  rounds with four CPU hogs: Llama-3.2-1B prefill/decode 13.2/1.98 with the default window,
  27.2/6.17 adaptive and 27.4/6.31 at `PEREGRINE_SPIN_US=0`; Gemma-3-1B 11.1/1.41, 28.4/6.07 and
  28.1/6.11. With eight hogs: Llama 10.3/1.43, 24.4/4.62, 21.6/5.20. Scheduling only: no kernel,
  quantisation or arithmetic changes.
- `peregrine bench --threads N` fixes the worker pool width and records it in the JSON `config`, so a
  run can be pinned to the same core count as a comparison engine. The default is unchanged.
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
- Batched ternary GEMM for prompts (S3.5). The AVX2 PQ2_0 prefill path ran one output row at a
  time through the whole K, so every row streamed the shared activation set from L2 again; only the
  groups inside one 8-activation window shared a decode. `dot_rows_pq2` now walks a 16-row tile over
  an 8-group K-block before the next block, so that block's activation window (8 activation rows ×
  8 groups × 192 B ≈ 12 KiB) stays in L1 across the tile, while the per-row order — ascending
  groups, the same four-plane 16-bit sum, the same integer offset, the same `d·td` scale and the
  same lane-wise fused multiply-add — is untouched. Eight activation accumulators are still held in
  registers per row, and the GEMV path is unchanged; batches smaller than a prefill chunk (the MTP
  verify batch) keep the per-row kernel, where all accumulators stay in registers and the tile's
  stack accumulators cost more than they save. A new test compares the GEMM against the GEMV
  bit for bit at widths 1..32, row counts 1..31 and K that is not a whole K-block. On bluefin, five
  alternating rounds against the parent tree (medians): Bonsai 2 2B prefill 130.6 → 154.0 tok/s at
  512 (+17.9 %) and 121.7 → 142.4 at 4096 (+17.0 %); Bonsai 2 4B 52.4 → 60.1 (+14.7 %) and
  48.6 → 57.7 (+18.8 %). Decode is flat (2B 61.0 → 60.8 tok/s, 4B 26.50 → 26.56, medians of five
  decode-only rounds), and the 4B MTP decode is flat (29.2 tok/s on both). The matmul bucket runs 230–254 GMAC/s, against 188–211 before, on four
  cores; a register-resident loop of the same 16-bit plane sum measures 606 GMAC/s and the same
  loop with the float fold 453, so the kernel is at 38–42 % of the raw int8 madd roof and ~55 % of
  the fold's arithmetic-only ceiling. PrismML's fork, same host and four threads: 53.6/52.9 tok/s
  (2B, 512/4096) and 17.8/17.5 (4B), i.e. Peregrine is 2.9×/2.7× and 3.4×/3.3×.
  The Gated DeltaNet scan was measured first: it is 3.0–3.4 % of the prefill wall on both Bonsai
  sizes (`docs/design/s3.5-delta-scan.md`), so a chunked scan is not in this PR.
- Q5_K prefill keeps eight activation accumulators in registers. `wide_q5_k` decoded a
  super-block once and then ran it against four activations at a time, folding each row
  into a `MAX_WIDE`-long stack array; it now walks eight at a time with the accumulators
  in registers, the shape #118 gave `wide_q6_k`, and the five-bit codes are still decoded
  once per super-block for the group. The arithmetic is unchanged — the same lane sums,
  the same per-block fused multiply-add and the same final horizontal sum — and a new
  test compares every window and ragged tail against the one-row kernel bit for bit.
  On bluefin (`PERGEGRINE_PROF2=full`, 512-token prompt) the Q5_K bucket runs
  140.5 → 171.9 GMAC/s, level with wide Q6_K's 183.4 and against the Q4_K tile's 282.5;
  its share of the prompt's wall falls from 30 % to 26 %. Five alternating rounds on
  Qwen3.5-2B Q4_K_M: prompt 139.4 → 148.9 tok/s at 512 (+6.8 %) and 132.4 → 139.3 at
  4096 (+5.2 %), decode flat; against llama.cpp v0.9.0 at its faster flash-attention
  setting the prompt rows move from 1.11× to 1.19× and 1.17×. Llama-3.2-1B (Q4_K/Q6_K)
  and Qwen3-1.7B (Q4_K/Q6_K) controls stay inside 1 %. The AVX2 real-model goldens for
  the three files this covers were re-recorded after the prefill-attention change
  (#115) that left them stale on main; the branch reproduces main's logits exactly on
  all of them.
- Attention's value mix widens to sixteen lanes on the AVX-512 tier. The mix is
  conversion-bound: it widens every stored half of V into f32 and accumulates it into
  the output, so `vcvtph2ps zmm` with 512-bit fused multiply-adds halves that work.
  Every path takes it — solo decode's F16-direct mix, the prefill tile's widened and
  F16-direct mixes, and a mapped multi-user row's chunked mix — and the pair kernel
  still shares each V read between two rows. The per-element order is the AVX2
  kernel's (`α·o`, then `p_j·v_j` over ascending positions) and F16C's widening is
  exact, so the widened mix is bit-identical to AVX2; the tier's real-model goldens
  for Llama-3.2-1B and Qwen2.5-0.5B came back unchanged. The softmax's `exp` is also
  sixteen wide (`exp16` is `exp8` lane for lane) with the eight-value f64 sum grouping
  kept, so its denominator is unchanged too. Attention-only at 4096 positions, one
  thread: 32.9 -> 29.9 ms/token (medians of three); at twelve threads the mix pass
  falls from ~2.55 to ~2.08 ms per token and attention from ~6.51 to ~5.93 ms.
  Whole-pass decode (medians of five alternating rounds, twelve threads): Llama-3.2-1B
  59.9 -> 60.9 tok/s at 512 and 47.9 -> 49.1 at 4096; Qwen2.5-0.5B 98.4 -> 99.8 at 512
  (nine rounds) and 71.7 -> 73.3 at 4096. Prompt processing is a wash inside the
  host's spread. Scores stay on the AVX2 tree on every tier: a zmm score row needs a
  sixteen-lane horizontal reduction and, measured at every visible length at head
  width 64, the zmm variants (including one that packs two keys into the two halves
  to keep the AVX2 reduction tree exactly) were 30–70 % slower. One score tree
  everywhere also keeps a row's score bits independent of its path by construction.
- Prefill attention reads the F16 K/V cache directly. The multi-row tile used to be
  widened f16 -> f32 before every score and every value mix; it now hands the cache's
  halves to the same register-blocked kernels the decode path reads, so a K or V tile
  is half the bytes and the widening pass is gone. The f16 kernels are the widened
  kernels' FMAs in the same order (`f16_direct_kernels_match_the_widened_kernels_bit_for_bit`),
  so a row's bits are unchanged: the real-model goldens for Llama-3.2-1B and
  Qwen2.5-0.5B do not move, nor do the synthetic ones at either tier. The path is
  taken once a row's visible context reaches 1024 positions; shorter prompts keep the
  widened tile, where the two measure the same. Prompt processing at 4096 tokens on
  four cores (H0, Llama-3.2-1B Q4_K_M): 75.9 -> 79.4 tok/s (medians of five
  alternating rounds); attention's share of the pass fell from 51.8 to 46.1 ms per
  16-token chunk.
- The repacked Q4_K prefill kernel takes eight activation rows per weight decode, and a
  sixteen-row prompt chunk shares one scale/min/code unpack between its two groups. The old
  kernel decoded a tile's scales, its mins and its sixteen windows once per group of four
  activations, so a 16-row chunk decoded every tile four times; the per-row arithmetic is
  untouched (the same `vpmaddubsw` pairs, the same int32 sum and float fold in the same order),
  so a row's bits do not move and the GEMV still agrees with the GEMM bit for bit. The
  sixteen-row block predecodes the super-block's window codes once and hands them to both
  groups, and the min term is one `vpmaddwd` per sub-block per row. On H0 (Xeon E5-2680 v4,
  AVX2) a 2048-wide row streamed from memory with sixteen activations went 32.4 -> 40.1 GMAC/s
  (33.1 -> 45.7 with the tile in L2); four pinned cores, on top of the Q6_K and attention
  changes already merged: Llama-3.2-1B Q4_K_M prefill 512 tokens 110.2 -> 120.1 tok/s and
  4096 83.2 -> 89.6, Qwen2.5-0.5B 232.4 -> 235.7 and 156.0 -> 161.5 (medians of five
  alternating rounds; Qwen is mostly Q5_0 and Q8_0, so little of it is this kernel). Decode is
  untouched: the GEMV is arithmetically the same and still agrees with the GEMM bit for bit.
  The prefill chunk is unchanged.
- The wide Q6_K prefill kernel keeps its activations' accumulators in registers. A
  super-block's codes are still decoded once, but they are now consumed by a group of
  eight activations whose eight integer lane sums and eight f32 accumulators never leave
  the registers: the old form kept a `MAX_WIDE`-long array in memory and reloaded it once
  per block per row. The arithmetic and its order are unchanged — the same `maddubs`/`madd`
  lane sums, the same per-block fused multiply-add, the same lane-wise float accumulation
  and final horizontal sum — so a row's bits do not move and do not depend on the width of
  the window it lands in. On H0 an 8192-wide row with sixteen activations went 6352 ->
  5486 ns (20.6 -> 23.9 GMAC/s), and the width cliff above sixteen activations is gone
  (32 activations: 17144 -> 11809 ns). Llama-3.2-1B Q4_K_M prefill at 4096 tokens on four
  cores: 76.1 -> 82.8 tok/s (medians of five alternating rounds).
- Prefill's remaining serial element-wise passes run in the pool. The attention output's
  quantisation and the two residual adds were loops on the calling thread between parallel
  regions; above one row they now go through the same row-parallel helper the norms and
  SwiGLU use. Element-wise and row-disjoint, so the bits do not move; decode keeps the
  serial form.
- Prefill attention now tiles over query rows as well as positions. A prefill task
  carries tens of query rows over one KV head, all scoring the same positions, so the
  64-position tile is widened once and walked in two phases over one buffer: keys
  first, scored for every row, then values, mixed into every row. Widening K and V
  together overflowed L1 and every row re-read both tiles from L2. Two rows share each
  loaded K vector in a register-blocked score kernel (their own accumulators, chunk
  boundaries and merge order are untouched), and the widened value mix now loads a
  position's eight value vectors into registers before the FMAs. A row's bits are
  unchanged: the tests read a 512-position context as a solo decode and as prompt
  pieces of 3, 16 and 64 rows on pools of 1, 2, 4 and 8 workers and require identical
  bits, for three attention geometries. On the 4-core Xeon (H0), attention-only
  prefill at 4096 fell from 16.6 s to 12.4 s on Llama-3.2-1B Q4_K_M and from 11.2 s to
  6.2 s on Qwen2.5-0.5B Q4_K_M. End-to-end prompt processing (medians of three
  alternating rounds) went 74.1 -> 79.9 tok/s (+7.8 %) and 130.7 -> 159.0 (+21.6 %) at
  4096, and 99.4 -> 103.5 (+4.1 %) and 221.0 -> 234.8 (+6.3 %) at 512. Decode does not
  move: attention-only decode at 4096 is 10.7-11.0 ms/token on both builds, and the
  end-to-end decode deltas of six alternating rounds have paired medians of -0.2 %
  and +3.1 % against a per-round spread of 20 %.
- Decode attention reads the F16 KV cache directly instead of widening a 64-position
  tile to f32 first. The stored halves are widened in registers (`vcvtph2ps`), the
  score pass is register-blocked four keys deep with two query rows sharing each
  loaded K vector, and the value mix keeps two rows' accumulators in registers while
  they share each loaded V vector. Prefill still widens a block once and lets every
  query row of the task read that tile: with tens of rows per block one widening
  amortises, whereas re-converting per row pair does not. Both paths use the same
  per-key reduction tree, so a row's bits are the same whether it is decoded alone,
  read in a prefill chunk, verified in a batch or run on a different pool width
  (`crates/peregrine-core/src/attn.rs`, `crates/peregrine-models/tests/attn_invariance.rs`).
  The tail key of a score block reduces with the same tree as a key inside one, so a
  position's score no longer depends on where its block ends; that is a one-time change
  to logits for contexts whose last tile ends mid-group, and the goldens are re-recorded.
  On the 4-core Xeon (H0), attention-only at 4096 context fell from 8.17 to 6.88
  ms/token on Qwen2.5-0.5B Q4_K_M and from 14.28 to 11.68 on Llama-3.2-1B Q4_K_M
  (medians of five alternating runs; decode attention is 25-30 % of a token, so the
  end-to-end token rate moved +0.4 % and +2.8 % respectively, within the host's
  noise). Prefill attention was unchanged (11.4-12.0 s and 16.8 to 16.4 s over a 4096
  token prompt), and its end-to-end rate was within noise.
- A decode step's attention is one parallel region for every decoding user, not one per
  user. Each user has a separate cache, so the dense layer used to call attention once
  per user per layer — ten fork-joins per layer at ten users, each splintering the same
  KV head's positions across the pool. The single-row runs now become one task per
  (KV head, row) in a single region; a row walks its own canonical position chunks with
  the same per-chunk kernel a solo decode chooses (the F16-direct one where the cache
  allows it) and merges them in the same order, so its arithmetic is its solo arithmetic
  and does not depend on the batch (a new test pins three users' rows against their solo
  decodes bit for bit). On the 12-core desktop with the server's default 24-thread pool
  (Llama-3.2-1B Q4_K_M, 512 positions, ten users) the step fell from 82-90 to 69-72 ms
  and ten-user aggregate throughput from 63.2 to 76.8 tok/s (+21.5 %; llama.cpp v0.9.0
  `-fa auto`: 78.0 tok/s), with decode-only aggregate 119.4 -> 157.4 (+31.8 %). On a
  12-thread pool the change is neutral (ten users 76.4 -> 76.5), because ten one-row
  calls already fit one wave of twelve workers.
- A prompt longer than 1024 tokens is prefilled in 64-token pieces instead of the tuned
  16. At 4096 positions with ten users the time to first token fell from 158.3 s to
  152.3 s (llama.cpp v0.9.0 `-fa auto`: 144.2 s), and single-user prefill of that prompt
  is no slower. The chunk changes how the prompt is read, not what it produces.
- `PEREGRINE_PROF2=1` now prints a periodic per-bucket breakdown of the server's engine
  thread to stderr (steps, rows per step, wall per step, matmul by phase, attention,
  quantise, barrier wait); `=full` adds the inner-loop timers.
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
- `stop` sequences are honoured by `/v1/chat/completions`. A request may send one string or up to
  four: generation ends when the decoded text contains one, the sequence itself is not returned, and
  `finish_reason` is `"stop"`. Streaming holds back the longest suffix that could still become a stop
  sequence, so a sequence split across tokens never leaks into a delta, and the match is per request,
  so one user's stop cannot truncate another's reply. Multi-byte UTF-8 is matched on character
  boundaries. Previously the field was accepted and ignored.
- A request body that is not valid JSON is now rejected with the OpenAI error object instead of
  axum's plain-text rejection. The status was already 400; a client that parses `error.message` can
  now read the reason. The compatibility suite checks the body as well as the code.
- The aarch64 numeric path no longer depends on the host's libm. `sin`, `cos`, `pow`, `log`,
  `exp` and `tanh` are now the engine's own (`vmath::det_*`) in the RoPE/YaRN table builder,
  attention's running-max rescale and row merge, MoE routing, the recurrent (Bonsai) gates,
  GeGLU/softcapping, sampling, and the sidecar builder's `hypot`; `exp_scalar` also replaces the
  remaining `expf` calls in the scalar softmax/SwiGLU fallbacks. glibc and Apple's libm round
  these differently, which is why the Linux-recorded aarch64 goldens did not reproduce on the
  macOS runner (#140); with the engine's implementations the aarch64 bits are fixed by the
  definition, not the C library. The `det_*` dispatchers call libm off aarch64, so no x86 golden
  moved. The aarch64 goldens were re-recorded: the `aarch64` scalar tier's move because its
  softmax/SwiGLU fallbacks were the last libm calls on that path, and the Qwen3-MoE and
  Qwen3.5 (Bonsai) entries of the `aarch64-neon*` tiers move in the last bits.
- Small-model decode no longer runs a Q5_0 or Q8_0 row at a time. A Q5_0 row is 22 bytes
  per 32 weights and a Q8_0 row 34, so the 32-byte activation block (and, for Q5_0, its
  32-byte lane sums) is more traffic than the weights; the per-row kernel reloaded both
  for every output row. The new eight-row kernel loads them once per block for eight
  rows, and each row still runs the per-row kernel's operations block by block, so the
  bits are unchanged (a row's result does not depend on the batch, tested bit for bit).
  On the 4-core Xeon (Qwen2.5-0.5B Q4_K_M) decode went 32.1 → 37.8 tok/s at 512 context
  and 27.1 → 29.0 at 4096 (medians of 5 and 3 alternating rounds). The eight-row kernels
  also prefetch one page ahead per row stream (`PEREGRINE_PF=0` still turns it off).
- The SwiGLU activation's quantisation now runs in the workers' SwiGLU pass instead of
  serially on the calling thread afterwards. A block's scale comes from that block's 32
  values alone, so a worker's contiguous block range writes exactly the bytes one call
  would; `QuantAct::prepare`/`fill_range` are tested against `fill` for any split. The
  decode quantise on the 0.5B fell from 0.73 ms to 0.17 ms per token.
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
- The startup calibration no longer stops its decode-thread sweep at the first candidate that fails to beat
  the previous best by 5%: one noisy high early sample froze it at one or two threads and the 9B decoded
  at 5.6 tok/s instead of 23.6. Every candidate is measured now, the default (the physical cores) stands
  unless a smaller count is at least as good (median bandwidth ≥ 98% of the default's, both measurements
  quiet), and the count never goes below half the physical cores.
- The A1 dequantisation oracle is re-cut from one pinned SmolLM2-360M-Instruct Q4_K_M file (size and
  FNV-1a checked at test time, SHA-256 in the failure message), so another upload fails loudly instead
  of looking like a kernel bug.
