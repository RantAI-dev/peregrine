# Testing

```sh
cargo xtest       # = cargo test --workspace --release --locked
cargo xclippy     # = clippy on every target, warnings are errors
```

Tests run in release mode: kernels are tested as they ship, and debug-mode SIMD tests are slow.
Hygiene: `scripts/hygiene.sh` checks tracked files for credential files and strings (and banned names when `HYGIENE_BANNED_PATTERN` is set); `scripts/hygiene-selftest.sh` tests it.

## Three kinds of test

1. **Kernel tests** (unit tests in the kernel modules, `tests/simd.rs`, `tests/tern.rs`): each fast kernel
   against its scalar reference. They need no model files and always run.
2. **Synthetic model tests** (`tests/synthetic.rs`): tiny random GGUFs written at test time by
   `tests/support/` — llama and qwen3 in F32, F16, Q8_0, Q5_0, Q4_0, Q4_K and Q6_K, qwen2 (attention biases,
   NeoX rotary pairing) in F16 and Q8_0, llama with a Llama 3.x `rope_freqs.weight` table or linear rope
   scaling, and the qwen35 hybrid (Gated DeltaNet, gated attention, Hadamard rotation) in PQ2_0 and
   PTQ1_0, plus that hybrid with one Q8_0 MTP (NextN) block whose speculation must be lossless and a
   variant whose drafts are actually accepted. The same hybrid is written there without the rotation —
   a plain Qwen 3.5 file, in F16 and Q8_0 — and llama.cpp v0.9.0 loads it and gives the same greedy
   tokens. Each must load, give finite logits, be bit-identical across
   runs and across 1, 2 and 4 threads, decode greedily the same way twice, and match its golden for the
   current tier (`tests/golden/synthetic/<tier>/`). A `qwen3moe` file (8 experts, 2 used, F16 and Q8_0
   experts) is written there too: the mixture-of-experts family runs it — it loads, gives finite logits,
   is bit-identical across runs and across 1, 2 and 4 threads, matches its golden, and gives the same
   tokens alone and batched through `Engine` — and llama.cpp v0.9.0 loads and runs the same file. A
   `gpt-oss` file (4 experts, 2 used, F16 and MXFP4 experts, attention sinks, alternating
   sliding-window and full attention, YaRN rope scaling) is written there too, and llama.cpp v0.9.0
   gives the same greedy tokens for both; a variant declares gpt-oss's own Harmony Jinja template and
   is detected as the `harmony` chat template. The same tests drive small models through `Engine`:
   prompts served alone must produce the same tokens when admitted together (a mixed prefill/decode step,
   both `Caches::Mapped` and, with `PEREGRINE_MIX=0`, `Caches::PerRow`), a shared opening must restore from
   the prefix store, and a LittleBit sidecar built in-test must give finite, deterministic logits. They
   need no model files, so they run in CI on every tier; the `x86_64-avx2` and scalar (`x86_64`) goldens
   were recorded on an Intel host and reproduced bit for bit on an AMD host. Every test writes into a
   directory of its own — `tmpdir("<test>")`, never a name two tests share: the tests of one binary run
   in parallel threads, and `write_to_file` truncates, so a second writer would pull the file out from
   under a test that has it memory-mapped (SIGBUS, most visibly on macOS).
   CI's pinned tiers (scalar, AVX2 and the AVX-512 tier under SDE) run with `PEREGRINE_GOLDEN=require`,
   so a new model or format must commit its goldens for all three; `PEREGRINE_GOLDEN=record` writes the
   missing ones (the AVX-512 ones on an AVX-512 machine). For the real-model goldens (`tests/golden.rs`), `require` fails a model file that is present but has no golden for the tier, and still skips a file that is absent. A golden pins one upload, byte for byte: a different upload of the same quant (the Qwen official `qwen2.5-0.5b-instruct-q4_k_m.gguf` is not bartowski's `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`) fails at the golden's `model … bytes` line, so check the SHA-256 listed here before reading a mismatch as a regression.
   `tests/runtime_parity.rs` pins what the session and a one-slot engine must agree on, since ADR 0007
   made the engine the only runtime: the same prompt through `Session` and through a one-slot `Engine`
   (greedy: the same tokens; sampled: the session counts conversation turns into the sampling key and the
   engine does not, and the two per-slot settings — the turn and the KV rotation — are what make them
   agree), the taps a DSpark drafter needs (which must not change a logit bit), a second user arriving
   while the first one speculates, and the prefix store, which is off while a drafter is attached. The
   drafter it uses is written in the test (`tests/support/dspark.rs`: a `config.json` and a safetensors
   file), so it needs no model files either.
   `tests/embeddings.rs` covers S4.1: a pooled slot's vector must equal the pooling of the exact
   reference forward's final hidden states (synthetic llama and qwen3 files, `last` and `mean`), and
   must be bit-identical alone, in batches of 2/7/64 of mixed lengths, at different chunks and thread
   counts, and beside a decoding chat slot. It also holds an ignored real-file test for
   Qwen3-Embedding 0.6B (looked up in `models/`, `~/models/` and the model cache):
   `cargo test --release -p peregrine-models --test embeddings -- --ignored`.
   S4.3's groundwork is tested without model files. `crates/peregrine-core/tests/systemone_math.rs`
   reads the 45 random cases `scripts/systemone-math.py` writes to `tests/fixtures/systemone-math.jsonl`
   (an independent Python implementation of llama.cpp's answer math) and requires every probability,
   score, confidence, `noul` and legend entry to agree to 1e-12; the module's unit tests pin the
   properties (uniform → confidence 0, dominant → confidence near 1, probabilities summing to 1, the
   temperature and score-bias effects, and the bucket edges at 2/3/5/6/10/11 options and 8/9/26/27
   for Lev). `tests/decision_meta.rs` writes synthetic llama files carrying the decision keys and
   checks the round trip and each refusal (unknown type, missing `systemone` template, a non-positive
   temperature, a malformed score bias, and `Decider`'s refusal of a type or template it does not
   render, while the real `openjev` pair loads with 52 single-token letters);
   `crates/peregrine/tests/inspect_decision.rs` pins the one
   `decision` line `peregrine inspect` prints and that the template's text never appears in it.
   `tests/labels.rs` covers `Want::Labels`: its emission's values are bit-identical to the same
   entries of the full logits row a reference forward computes for that position, alone, after a
   sibling question with the same prompt prefix (which the prefix store restores on dense `llama`
   files; a hybrid `qwen35` snapshot now carries its recurrent state and restores at that snapshot's
   own length) and in a batch of questions, at prefill chunks 1/16/64 and 1/2/4 threads, through both
   scheduling paths (the mixed engine and a `PEREGRINE_MIX=0` child). S4.9j's fork is pinned the same
   way: a question forked from a group's one evaluation of the shared beginning equals the same
   question computed alone, for groups of 2/3/10 with different tail lengths, the fork at 1, at a
   chunk boundary, mid-chunk and one row before the end, in the f16 and f32 caches, on both synthetic
   families; a count test pins rows forwarded at `prefix + Σ tail`, and a repeated-state test pins the
   store's restore across requests. The read-out computes the full row and gathers:
   the output head has no cheaper path that would stay bit-identical. The same file checks that a
   question's logits are bit-identical beside an embedding slot and a decoding chat slot, which is
   what one server serving all three endpoints asks the engine for.
   S4.3c, the decision endpoint, is tested without model files too. `decision.rs`'s unit tests pin
   `Decider::parse_request` to llama.cpp's messages (state, questions, instructions, type, criteria,
   the 52-option limit, media and `videos`), the request and option order, the answer field order and
   the ordered response text. `crates/peregrine/tests/server_systemone.rs` starts the built server on
   a synthetic `qwen35` file carrying `decision.type = openjev`, the real `systemone` template (from
   `peregrine_core::systemone::OPENJEV_TEMPLATE`) and the published temperatures: it covers the three
   question types, request order, `usage`, every validation message, the media 501s, a chat model
   refusing the route, a decision model refusing chat, `/v1/models`' modalities, and five questions
   on two slots (each answer identical alone). `crates/peregrine/tests/cli_decide.rs` pins
   `peregrine decide`'s one-line and `--json` output and the `@file`/`-`/`--request` sources. With
   `tinyopenjev-for-testing-Q8_0.gguf` in `~/models/`,
   `cargo test -p peregrine --release --test systemone_real -- --ignored` drives the real endpoint
   with the 48 fixture requests and compares every answer with llama.cpp's;
   `cargo test -p peregrine-models --release --test systemone_openjev` does the same through the
   pieces directly.
   S4.6a's `bert` family is covered the same way: a synthetic post-norm encoder with the WordPiece
   vocabulary (and a second with the XLM-R unigram one), the vector against `tests/support/bert_ref.rs`'s
   independent scalar forward (bit for bit for F32; for Q8_0 exactly on the scalar tier and to 1–2 ulp
   elsewhere, where the int8 tile fuses its block scale and the scalar twin does not), the invariance
   matrix, the position-table limit, the no-allocation pass, the server and CLI cases
   (`a_bert_encoder_*`), and `tests/golden/synthetic/<tier>/bert-Q8_0.golden` recorded on scalar, AVX2
   and AVX-512 under SDE. Its three ignored real-file tests use the 200-text fixtures under
   `tests/fixtures/bert/` (llama.cpp's ids and vectors, the fp32 PyTorch reference vectors) and skip,
   with a notice, when the GGUF files are absent. S4.7's `modern-bert` family is covered the same
   way: a synthetic four-layer file with a window narrower than the texts, two different RoPE bases,
   F32 and Q8_0 variants, the three gated activations and a classifier-headed variant; the vector
   against `tests/support/modern_bert_ref.rs`'s independent scalar forward (bit for bit for F32;
   for Q8_0 exactly on the scalar tier, and on SIMD tiers within the measured int8 distance, because
   a 1-ulp projection difference can flip an activation code in the next layer); the invariance
   matrix; the context limit; the no-allocation pass; the server and CLI cases
   (`a_modern_bert_encoder_*`); and
   `tests/golden/synthetic/<tier>/modern-bert-Q8_0.golden`, recorded on scalar, AVX2 and AVX-512
   under SDE (no aarch64 golden yet). Its ignored real-file test uses the 200 texts (16–6976 tokens)
   and llama.cpp's ids and vectors under `tests/fixtures/modern-bert/` for gte-modernbert-base and
   Granite Embedding R2, F16 and Q8_0, and skips a file that is absent.
   The same file covers S4.4: a synthetic bidirectional `gemma-embedding` file (mean/cls/last),
   the vector against an independent scalar reference forward in `tests/support/embed_ref.rs`,
   the invariance matrix (alone, batches of 2/7/64, mixed lengths, 1/2/4/8 threads, both
   admission orders), and the tier golden (`tests/golden/synthetic/<tier>/gemma-embedding-Q8_0.golden`,
   recorded on scalar, AVX2 and AVX-512 under SDE). There is no aarch64 golden yet: it will be
   recorded on the ARM host, the test treats a missing golden as a skip unless
   `PEREGRINE_GOLDEN=require`, and the aarch64 CI jobs do not set it. Its two ignored real-file
   tests are EmbeddingGemma 300M (batch invariance, and the 200-text cosine bound against
   llama.cpp's vectors in `tests/fixtures/embeddinggemma/`, which states the per-format bound
   from `docs/design/s4-embeddings-decisions.md` §5); `peregrine-tests` cover the server and CLI
   refusal of chat on an encoder. S4.14b adds the same for `mistral3` (Nemotron 3.5 Embed 1B): a
   synthetic non-causal file whose original context is 32 tokens, so the YaRN ramp and the
   attention temperature's steps run; the vector against its own reference forward; the invariance
   matrix; the tier golden
   (`tests/golden/synthetic/<tier>/mistral3-Q8_0.golden`, recorded on scalar, AVX2 and AVX-512
   under SDE, no aarch64 golden); no-allocation coverage; and an ignored real-file test that feeds
   the 64 token-id lists from `tests/fixtures/nemotron/` and checks the F16 and Q8_0 cosines
   against llama.cpp's vectors; a second ignored test encodes the fixture texts with the file's
   Tekken tokenizer (S4.14a), checks all 64 id lists against llama.cpp's, and checks that a text
   embedded by its ids and by its own encoding is the same vector bit for bit.
3. **Real model tests** (`tests/engine.rs`, `batch.rs`, `bonsai.rs`, `oracle.rs`, `dspark.rs`, ...): real
   GGUF files, checking tokens against reference outputs. They skip, with a message, when the file is
   missing. They are the release gate.

## The server's lifetime

`crates/peregrine/tests/serve_lifetime.rs` starts the built `peregrine serve` binary on a synthetic
model and checks that it is still serving (and answering a chat completion) after a drain timeout has
elapsed with no signal, then that `SIGTERM` stops it cleanly. `PEREGRINE_DRAIN_TIMEOUT_MS` overrides
the 30 s drain timeout for that test; it is for tests only and is not part of the user-facing CLI.
The unit tests around `serve_with_drain` in `crates/peregrine-server/src/lib.rs` cover the three
shutdown paths directly: no shutdown request (the server keeps serving), an idle drain (prompt exit),
and a drain with a streaming client holding the connection open (it ends at the timeout, not before).

## Where tests look for model files

| File | Location |
|---|---|
| SmolLM2-360M-Instruct Q4_K_M, Qwen3-4B Q4_K_M | `$PEREGRINE_MODEL`, or `~/models/` |
| Qwen3-1.7B Q4_K_M | `models/` at the repository root |
| Qwen2.5-0.5B-Instruct Q4_K_M, Llama-3.2-1B-Instruct Q4_K_M | `~/models/` |
| Llama-2-7b-chat-hf Q4_K_M, Mistral-7B-Instruct-v0.2 Q4_K_M | `~/models/` |
| Gemma 3 1B Instruct Q4_K_M (`google_gemma-3-1b-it-Q4_K_M.gguf`) | `~/models/` |
| Qwen3-30B-A3B Q2_K (`Qwen3-30B-A3B-Q2_K.gguf`) | `~/models/` |
| Qwen3.5-2B Q4_K_M (`Qwen3.5-2B-Q4_K_M.gguf`) | `~/models/` |
| tinyopenjev test model Q8_0 (`tinyopenjev-for-testing-Q8_0.gguf`, 46 MB) | `models/` or `~/models/` |
| Phi-4 14B Q4_K_M (`phi-4-Q4_K_M.gguf`) | `~/models/` |
| `bge-m3-Q2_K.gguf` (encoder-tokenizer XLM-R oracle; optional) | `models/` or `~/models/` |
| `Nemotron-3.5-Embed-1B-Q4_K_M.gguf` (Tekken tokenizer oracle; optional) | `models/nemotron/` or `~/models/nemotron/` |
| Bonsai 2 9B PQ2_0 | `$PEREGRINE_BONSAI9`, or `models/bonsai2-9b/` |
| Bonsai 2 2B PQ2_0 | `models/bonsai2-2b/` |
| Bonsai 2 4B PQ2_0 with MTP | `$PEREGRINE_BONSAI4_MTP`, or `models/bonsai2-4b/` |
| TheBloke's Mistral 7B Instruct v0.2 and Llama 2 7B Chat Q4_K_M | `~/models/` (see below) |

The Bonsai 2 files come from PrismML's Hugging Face repositories. The two the aarch64
goldens and the Bonsai benchmark use are:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `Ternary-Bonsai-2-2B-PQ2_0.gguf` | <https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/main/Ternary-Bonsai-2-2B-PQ2_0.gguf> | 517312992 | `d5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928` |
| `Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf` | <https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf/resolve/main/Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf> | 1274642176 | `0acf517de1dc3d85c403a60df6b8b07a4878ac174f368d4390be98406a4d6c0d` |


`models/` is git-ignored. These locations will be unified under one variable as part of S0.1.

The SmolLM2 A1 oracle fixture is cut from one exact file: mradermacher's
`SmolLM2-360M-Instruct.Q4_K_M.gguf` (SHA-256 `ade7aee1e5bd7c3b8f1fedafc61a1a0be31e8afa64e006607cd202323e2ccce0`),
and `tests/oracle.rs` refuses any other upload.

### The two tokenizer oracle files

`tests/tokenize.rs` checks the tokenizer of two more models against token ids taken from llama.cpp's
`llama-tokenize`. It needs both files in `~/models/`, under exactly these names:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | <https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf> | 397808192 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | <https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf> | 807694464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |

Both were checked against llama.cpp v0.9.0 (`cd959ba`) with `llama-tokenize --no-bos --ids`: all twelve
fixture strings tokenise to the same ids as the fixture files
`tests/fixtures/tokenize-golden-qwen2.tsv` and `tests/fixtures/tokenize-golden-llama3.tsv`. The two
models are also checked end to end against llama.cpp on the same files (greedy decoding, the chat
path and the goldens under `tests/golden/<tier>/`).

### The two chat-template files

The chat templates are checked against llama.cpp v0.9.0 on two files of TheBloke's, under exactly these
names in `~/models/`:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `thebloke-mistral-7b-instruct-v0.2.Q4_K_M.gguf` | <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf> | 4368439584 | `3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9` |
| `thebloke-llama-2-7b-chat.Q4_K_M.gguf` | <https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf> | 4081004224 | `08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa` |

The first carries the pre-update Mistral v0.2 template (no system branch), the second declares no chat
template at all: the first is served as it is, the second with `--chat-template llama2`. Both were compared
with llama.cpp v0.9.0 on the same files — the rendered prompt, its token ids and the answer.

### gpt-oss and the Harmony template

The synthetic gpt-oss file in `tests/synthetic.rs` is written with gpt-oss's own Jinja template in
`tokenizer.chat_template` (the fixture `crates/peregrine-core/tests/fixtures/harmony.jinja`, from llama.cpp
v0.9.0's `models/templates/openai-gpt-oss-120b.jinja`). `Template::detect` must choose `harmony` from it,
the way llama.cpp detects the `<|start|>` + `<|channel|>` pair (`src/llama-chat.cpp:222-223`, v0.9.0), and
the stop set must carry `<|return|>` (and `<|call|>`, when the vocabulary has it) but not `<|end|>`, which closes the analysis message before the answer (llama.cpp v0.9.0 drops it from the end-of-generation set, `src/llama-vocab.cpp:3345-3383`).
`template.rs`'s tests pin the rendered strings — system block, developer block, user and assistant turns,
and the generation prompt — for a system + user turn, a multi-turn conversation, and a final assistant
message with no generation prompt. They also pin that a replayed earlier assistant turn keeps only its
final channel: gpt-oss's Jinja drops every previous turn's analysis ("CoT is dropped during all previous
turns"), so `Template::assistant_turn` strips it and a Harmony `Session` re-renders its transcript instead
of appending to the cache of the reply just generated (whose analysis the replay does not contain). A real
gpt-oss file is still the open check.

The reasoning split lives in `crates/peregrine-server/src/harmony.rs`. It follows llama.cpp v0.9.0's
gpt-oss handler (`common/parsers/gpt-oss.cpp:66,68-93`): `<|start|>assistant` opens a message,
`<|channel|>analysis<|message|>` … `<|end|>` is the reasoning span, a `commentary` preamble is content, and
`<|channel|>final[ constraint]<|message|>` … is the answer. `reasoning_format` maps as llama.cpp does:
`none` leaves the analysis text in `content`, `auto`/`deepseek` split it into `reasoning_content`, and
`deepseek-legacy` also mirrors it into streaming content deltas (`common/parsers/gpt-oss.cpp:66,80-83`;
`common/common.h:428-436`; `tools/server/server-task.cpp:78`). Control tokens are consumed and never appear
in either field, including when a control token is split across streaming chunks. The parser's unit tests
cover analysis + final, final only, a reply cut off mid-analysis by `max_tokens`, and every control token
split across a chunk boundary; the server's tests cover the `reasoning_format`/`reasoning_effort` mapping
and that an earlier assistant turn's `reasoning_content` is not replayed, and `tests/synthetic.rs` runs a
gpt-oss `Session` with an analysis channel in its history against one with only the final text, comparing
the prompt and the tokens. There is no server integration test for gpt-oss: the synthetic model writers
live in `peregrine-models`' test support, which the server crate cannot import, and the server's `App` is
built inside `run`.

### The two SentencePiece oracle files

`tests/tokenize_spm_real.rs` checks the SentencePiece tokenizer on the two SentencePiece families
Peregrine renders chat templates for — Llama 2 and Mistral 7B — against token ids taken from
llama.cpp's `llama-tokenize` on the same uploads. It needs both files in `~/models/`, under exactly
these names:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `Llama-2-7b-chat-hf-Q4_K_M.gguf` | <https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/Llama-2-7b-chat-hf-Q4_K_M.gguf> | 4081005152 | `f475bcb77959f5721548bcc5122f4d5a0eb9d94ddb5b7fdce826bc1410ae0a8b` |
| `Mistral-7B-Instruct-v0.2.Q4_K_M.gguf` | <https://huggingface.co/mradermacher/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/Mistral-7B-Instruct-v0.2.Q4_K_M.gguf> | 4368441024 | `b2d7044c02b0ce9bce2ef8f37fb5bbab0190b774515473613400627b6f49f321` |

Both hashes match the ones Hugging Face reports for those uploads. The fixture files
`tests/fixtures/tokenize-golden-llama2-spm.tsv` and `tests/fixtures/tokenize-golden-mistral-spm.tsv`
hold, for 22 strings each, the ids `llama-tokenize --ids --no-escape` produced with and without
`--no-bos`; the test compares `Tokenizer::encode` against the third column and
`Tokenizer::encode_prompt` against the fourth.

Two strings in each fixture start with the literal `<s>`, and there the two implementations differ
by one token: llama.cpp prepends the model's BOS to every prompt *unconditionally* — so a prompt that
already begins with one gets two, which llama.cpp itself warns about — while `encode_prompt` keeps
exactly one. The test states that difference explicitly rather than hiding it.

Both models are also checked end to end against llama.cpp on the same files (greedy decoding, the
chat path and the goldens under `tests/golden/<tier>/`).

The upload matters: TheBloke's `Llama-2-7B-Chat-GGUF` and `Mistral-7B-Instruct-v0.2-GGUF` carry no
chat template and a template Peregrine does not render, and `peregrine inspect` refuses them with
"chat template is missing or not recognised" and "unsupported [INST] chat template variant". The two
files above were converted from the Hugging Face repositories with a `convert_hf_to_gguf.py` new
enough to carry the models' own templates.

### Tokenizer test vocabularies

`scripts/fetch-test-vocabs.sh` downloads the llama.cpp SPM test vocabulary (`models/ggml-vocab-llama-spm.gguf`)
and reference test vectors (`.inp` and `.out`) from pinned llama.cpp release files into `models/vocab/` (git-ignored),
validating pinned SHA-256 checksums. `crates/peregrine-core/tests/tokenize_spm.rs` tests these reference vectors
and skips with an informative message if the files are absent.

### The encoder tokenizers: three run without model files, XLM-R uses one

`tests/tokenize_encoders.rs` checks WordPiece (`tokenizer.ggml.model = "bert"`) and SentencePiece
unigram (`"t5"`) against llama.cpp v0.9.0 on four real vocabularies. Three need no model file: the
vocabulary is carried beside the goldens in a compact `.bin` holding exactly what `from_gguf` read out
of the file (pieces, scores, token types, the precompiled charsmap and the special-token rules), so
they run in CI on every tier.

| Vocabulary | Model file (not committed) | `.bin` | `.tsv` |
|---|---|---|---|
| WordPiece, uncased | `second-state/All-MiniLM-L6-v2-Embedding-GGUF` / `all-MiniLM-L6-v2-Q2_K.gguf` (`05276e01…5acd`) | 291 KB | 20 KB |
| WordPiece, cased | `lorinet3/kblab-bert-swedish-cased-pos-gguf` / `bert-swedish-pos-q4_0.gguf` (`bddde253…96f3`) | 482 KB | 20 KB |
| unigram, T5 | `fernandoruiz/flan-t5-small-Q4_0-GGUF` / `flan-t5-small-q4_0.gguf` (`407c1f1c…8058`) | 650 KB | 15 KB |
| unigram, XLM-R | `gpustack/bge-m3-GGUF` / `bge-m3-Q2_K.gguf` (`8b0dacc2…44d8`) | — (model file only) | 18 KB |

The T5 vocabulary is the standard 32,128-piece SentencePiece unigram vocabulary that
`t5-small` and `flan-t5-small` ship, with `</s>` as EOS, `<unk>` as unknown and the
`<extra_id_N>` sentinels. The XLM-R vocabulary is 250,002 pieces; carrying its `.bin`
would add 4.1 MB, larger than the rest of the repository's fixtures combined, so that
vocabulary is not committed. The XLM-R test builds its tokenizer from `bge-m3-Q2_K.gguf`
in `models/` or `~/models/` and skips with a one-line notice when the file is absent —
including under `PEREGRINE_GOLDEN=require`, which in this repository still skips a
missing model file (CI has none of these files). With the file present the test runs
the same 51 strings and 10 pairs as the other three.

Each `.tsv` holds, per string, the ids without special tokens, the ids with them, the escaped input
and llama.cpp's detokenisation of the bare ids; the `-pair.tsv` beside it holds ten (query, document)
pairs formatted by llama.cpp's own `format_prompt_rerank`. T5 has no rerank template, so the pair
file uses that function's generic path (`q </s> d </s>`). For T5, `llama-tokenize --ids` cannot add
EOS because the model has no BOS, so the generator runs with `--skip-cross-check`: the bare column
was cross-checked separately against `llama-tokenize --ids --no-bos` and matches on all 50 non-empty
strings, and the special column comes from `llama_tokenize(add_special=true)`.

To regenerate:

```text
python3 scripts/gen-tokenizer-golden.py --llama-tokenize .../llama-tokenize \
    --oracle /tmp/oracle --pair-oracle /tmp/pairoracle --model MODEL.gguf \
    --name minilm --out crates/peregrine-core/tests/fixtures/tokenize-golden-wpm-minilm.tsv \
    --provenance "repo / file" --sha256 HASH
cargo run --release -p peregrine-bench --example vocab_dump -- MODEL.gguf OUT.bin
```

`/tmp/oracle` and `/tmp/pairoracle` are `scripts/tokenizer-oracle.cpp` and
`scripts/tokenizer-pair-oracle.cpp` built against the same llama.cpp checkout; the first prints the
two id columns plus `llama_detokenize`'s output, the second calls `format_prompt_rerank` directly
rather than re-implementing it. When one of the three model files happens to be present, the test also
checks that its `.bin` is byte-identical to what the file yields. A 2,400-input differential fuzz
against `llama-tokenize`/the oracle on the T5 vocabulary (bare ids, ids with specials, and
detokenisation: 7,200 comparisons) found no divergence.

### The Tekken pre-tokenizer: Nemotron 3.5 Embed

Mistral's Tekken vocabulary is the one pre-type llama.cpp has no hand-written splitter for: it runs
the pattern through `std::regex` over a collapsed text, so only ASCII case splits words and a
non-ASCII letter counts as both upper and lower. NVIDIA's Nemotron 3.5 Embed 1B file
(`general.architecture = mistral3`, `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = pixtral`,
131,072 pieces and 269,443 merges) carries one, which is why Peregrine has `tekken`/`pixtral`,
`tekken-num3` and `nemotron-3.5-embed` pre-types.

`tests/tokenize_tekken.rs` checks 361 strings against ids llama.cpp v0.9.0 produced — every
alternative of the pattern, ASCII case boundaries (`HTMLParser`, `iPhone`), leading punctuation and
whitespace, single digits, punctuation runs before `/` and newlines, newline and CRLF runs, trailing
whitespace, Latin/Cyrillic/Greek/CJK/Arabic/Hebrew/Indic/Thai text, combining marks, emoji, long
words, contractions, the `query: ` / `passage: ` prefixes, literal `[INST]`/`<s>` spellings with
`parse_special` on and off, and empty or whitespace-only input. A second fixture holds the 22 strings
whose ids change under `nemotron-3.5-embed` (the ones containing U+200C), generated with that
pre-tokenizer's metadata override, and the test switches the pre-type to match.

That vocabulary is 131,072 pieces and 269,443 merges, so the `.bin` the encoder fixtures carry would
be larger than 2 MB; like XLM-R, the test builds the tokenizer from
`Nemotron-3.5-Embed-1B-Q4_K_M.gguf` in `models/nemotron/` or `~/models/nemotron/` and skips with a
one-line notice when it is absent (CI has none of these files). The splitter itself is tested in
`src/tokenizer.rs` with one case per alternative, and runs in CI with no model file.

To regenerate the fixtures:

```text
g++ -std=c++17 -O2 -I$LLAMA/include -I$LLAMA/common -I$LLAMA/ggml/include \
    -L$LLAMA/build/bin -Wl,-rpath,$LLAMA/build/bin scripts/tokenizer-oracle.cpp \
    -o /tmp/tekken-oracle -lllama -lggml
python3 scripts/gen-tekken-golden.py --oracle /tmp/tekken-oracle \
    --llama-tokenize $LLAMA/build/bin/llama-tokenize \
    --model nemotron/Nemotron-3.5-Embed-1B-Q4_K_M.gguf \
    --out crates/peregrine-core/tests/fixtures/tokenize-golden-tekken.tsv \
    --provenance "..." --sha256 HASH
python3 scripts/gen-tekken-golden.py --oracle /tmp/tekken-oracle --pre nemotron-3.5-embed \
    --model nemotron/Nemotron-3.5-Embed-1B-Q4_K_M.gguf \
    --out crates/peregrine-core/tests/fixtures/tokenize-golden-tekken-nemotron.tsv \
    --provenance "..." --sha256 HASH
```

The oracle prints a fifth column beside the four the encoder fixtures use — the ids with
`add_special = true` and `parse_special = false` — and accepts an optional `KEY=VALUE` metadata
override, because `llama-tokenize`'s own `--override-kv` never reaches the loader. In development the
splitter was also compared with llama.cpp's `unicode_regex_split` on the Tekken and `tekken-num3`
patterns over 10,538 strings (every three-character combination of the classes that matter, 4,000
random mixed-script strings, and runs of each class), with no divergence.

### Tool-calling prompts: fixtures rendered by llama.cpp

`crates/peregrine-core/tests/tools_render.rs` rebuilds the prompt of every case in
`tests/fixtures/tools-*.jsonl` from the request alone and compares it byte for byte with the prompt
llama.cpp built. It needs no model file and runs in CI. A case llama.cpp refused must be refused by
Peregrine too.

The fixtures come from llama.cpp itself: `scripts/tool-fixtures.py` sends each case of
`scripts/tool-cases.json` (messages and tools in OpenAI's shapes) to `POST /apply-template` of a
`llama-server --jinja` built at v0.9.0 (`cd959ba`) and stores the request text exactly as sent and
the prompt that came back. `--template <file>` renders with a template file in place of the model's
own, so a template can be exercised with any small chat model loaded. The Qwen 2.5 fixture was made
with `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (the pinned bartowski upload) and its own template, which
is the file `crates/peregrine-core/src/tools/qwen25.jinja`. The Qwen 3.5 fixture was made with the
same model loaded and `--template crates/peregrine-core/src/tools/qwen35.jinja --set qwen35`: that
file is the template of `Ternary-Bonsai-2-2B-PQ2_0.gguf`, which llama.cpp cannot load itself, and
the set adds the cases that only this family has (thinking on and off, thinking in the history).

### System One prompts: fixtures rendered by llama.cpp

`crates/peregrine-core/tests/systemone_render.rs` rebuilds the prompt of every case in
`tests/fixtures/systemone-openjev.jsonl` from the request alone and compares it byte for byte with
the prompt llama.cpp rendered. It needs no model file and runs in CI. Each fixture line also holds
the prompt's token ids and the server's answer; with `tinyopenjev-for-testing-Q8_0.gguf`
(`ggml-org/tinyopenjev-for-testing-gguf`, 46 MB, SHA-256 `bb451e6f…3b15a7`) under `models/` or
`~/models/`, a third test checks that the file's `systemone` template is the one the renderer was
written from and that Peregrine's tokenizer gives the same ids; without the file it prints
`SKIPPED`.

The fixtures come from llama.cpp itself: `scripts/systemone-fixtures.py` sends each case of
`scripts/systemone-cases.json` to a `llama-server` built at v0.9.0 (`cd959ba`) with
`scripts/systemone-dump.patch`, which makes the server append each rendered prompt and its tokens
to a file and changes nothing else. A new decision type gets its own cases, fixture and renderer
the same way.

`crates/peregrine-models/tests/systemone_openjev.rs` goes the whole way with the same model
file: it answers every fixture request with the renderer, the tokenizer, a labels slot and the
answer math, and compares with the answer llama.cpp gave (same option chosen; probabilities, score
and noul within 5e-3). It prints `SKIPPED` without the file.

### Gemma 3 1B Instruct

Gemma 3 1B is checked end to end against llama.cpp v0.9.0 on the bartowski Q4_K_M upload:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `google_gemma-3-1b-it-Q4_K_M.gguf` | <https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf> | 806058496 | `12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d` |

The tokenizer was checked across 22 test strings against `llama-tokenize --ids` and `llama-tokenize --ids --no-bos` (100% agreement, with exactly one BOS on the prompt path).
Greedy decoding (32 tokens, 6 threads) was checked against llama.cpp on three prompts, including a 713-token prompt that exercises the 512-token sliding window: all 32/32 tokens match llama.cpp bit for bit on all three prompts.
The chat path (`/apply-template` and `/v1/chat/completions`) matches byte for byte with and without a system message (folded into the first user turn).
The model is recorded in `crates/peregrine-models/tests/golden/<tier>/` and checked in `docs/targets.md` (KL divergence at `-c 512` and `-c 1024`).

### Qwen 3 MoE (Qwen3-30B-A3B)

Qwen3-30B-A3B is checked end to end against llama.cpp v0.9.0 on the unsloth Q2_K upload:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `Qwen3-30B-A3B-Q2_K.gguf` | <https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q2_K.gguf> | 11258610240 | `db3ce897ccc9e7d9dbf17fe083cae7880a2092aa473b45eba8b77715aa9ca170` |

The tokenizer was checked across 22 test strings against `llama-tokenize --ids --no-bos` (100% agreement, 22/22 matches).
Greedy decoding (32 tokens, 8 threads) was checked against llama.cpp on three prompts: prompt 2 and prompt 3 match bit for bit across all 32 steps (32/32), while prompt 1 matches through the first 26 tokens with the first divergence at token 26 being a near-tie between " Portugal" (Peregrine top-1, logit 31.92) and " the" (llama.cpp top-1, logit 31.75), a margin of 0.169.
The chat path (`/apply-template` and `/v1/chat/completions`) renders identical prompt templates (15 tokens without system, 26 tokens with system) and yields matching answers identifying Paris with reasoning tokens.
The model is recorded in `crates/peregrine-models/tests/golden/<tier>/` and checked in `docs/targets.md` (KL divergence at `-c 512`, 20 chunks).

### Plain Qwen 3.5 (Qwen3.5-2B)

Qwen3.5-2B is checked end to end against llama.cpp v0.9.0 on the Q4_K_M file:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `Qwen3.5-2B-Q4_K_M.gguf` | `~/models/Qwen3.5-2B-Q4_K_M.gguf` | 1280835840 | `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223` |

The file carries no `prism.hadamard.*` metadata and is verified by `peregrine inspect`.
The tokenizer was checked across 22 test strings against `llama-tokenize --ids --no-bos` (100% agreement, 22/22 matches).
Greedy decoding (32 tokens, 8 threads) was checked against llama.cpp across four prompts, including a 602-token prompt from wikitext-2: prompts 1, 2, and 4 match bit for bit across all 32 steps (32/32), while prompt 3 matches through the first 27 tokens with the first divergence at token 27 being a near-tie between " book" (Peregrine top-1, logit 14.92) and " printing" (llama.cpp top-1, logit 14.97, logprob margin 0.045).
The chat path (`/v1/chat/completions`) generates matching reasoning tokens with answers identifying Paris as the capital of France.
The model is recorded in `crates/peregrine-models/tests/golden/<tier>/` and checked in `docs/targets.md` (KL divergence at `-c 512`, 20 chunks).

### Phi-4 (Phi 4, 14B)

Phi-4 is checked end to end against llama.cpp v0.9.0 on the Q4_K_M file:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `phi-4-Q4_K_M.gguf` | `~/models/phi-4-Q4_K_M.gguf` | 9053114816 | `009aba717c09d4a35890c7d35eb59d54e1dba884c7c526e7197d9c13ab5911d9` |

The file declares the `phi3` architecture (40 blocks, embedding 5120, feed-forward 17920, context 16384) and a GPT-2 byte-level BPE vocabulary whose pre-tokenizer is `dbrx`. llama.cpp v0.9.0 splits that with the Llama 3 pattern (`src/llama-vocab.cpp`, lines 310-315), so `dbrx` is a Peregrine pre-type of its own now, and the ids are checked against `llama-tokenize` in `crates/peregrine-core/tests/fixtures/tokenize-golden-phi4.tsv`: 22 of 22 strings match exactly (`llama-tokenize --ids --no-bos`, 100% agreement).
Greedy decoding (32 tokens, 8 threads) was checked against llama.cpp on three prompts: all three match bit for bit across all 32 steps (32/32 each, 96/96 steps in all). The end-of-generation tokens are taken out of the running on both sides — exactly what `llama-server`'s `ignore_eos` does — so the comparison runs the full 32 steps instead of stopping at the model's own end token.
The chat path (`/v1/chat/completions`) answers "The capital of France is Paris.", like llama.cpp's `--jinja` server, on the same prompts: 14 tokens without a system message and 24 with one. The file's template is Phi-4's own ChatML variant (`<|im_start|>role<|im_sep|>text<|im_end|>`, no newlines), which Peregrine detects by `<|im_sep|>` as llama.cpp v0.9.0 does (`src/llama-chat.cpp:99-101`, rendered at `:345-352`).
The model is recorded in `crates/peregrine-models/tests/golden/<tier>/` and checked in `docs/targets.md` (KL divergence at `-c 512`, 20 chunks).

**The AVX2 golden needs re-recording.** The file holds 40 Q5_K tensors, and S4.9i changed the order
in which the AVX2 and AVX-512 Q5_K kernels add up a row (the result stays within one ulp of the
scalar reference). `tests/golden/x86_64-avx2/phi-4-Q4_K_M.golden` was recorded before that and no
longer matches. Loading the file takes about 9 GB of memory, more than any x86 host of the project
has free, so it could not be recorded again with the change: on an AVX2 machine that has the file,
`PEREGRINE_GOLDEN=require` fails for this model until someone records it on a larger machine
(`PEREGRINE_GOLDEN=record`). The other large files (Llama-2-7B, Mistral-7B, Qwen3-30B-A3B) hold no
Q5_K tensors and are not touched by that change.

## Accuracy against llama.cpp

`docs/dev/accuracy.md`: the KL-divergence measurement on real text (`peregrine-bench` example `kld`).

## CPU tiers

Kernels are chosen at run time from the CPU's features, capped by `PEREGRINE_ISA` (`scalar`, `avx2`,
`avx-vnni`, `avx512`; default `native`). Every feature check goes through `cpu_has!` (`crates/peregrine-kernels/src/cpu.rs`),
so a cap reaches every kernel.

```sh
PEREGRINE_ISA=scalar   cargo xtest    # portable code only
PEREGRINE_ISA=avx2     cargo xtest    # the AVX2 tier, also on an AVX-512 machine
PEREGRINE_ISA=avx-vnni cargo xtest    # the AVX-VNNI-256 tier, also on an AVX-512 machine
```

Goldens are stored per effective tier (`cpu::tier_name()`), so a forced run is compared with that tier's
goldens. Two machines must agree on a shared tier: the AVX-512 host forced to `avx2` reproduces the AVX2
host's goldens. The `x86_64-avx2-vnni256` tier aliases directly to `x86_64-avx2` goldens to check bit-identity.

CI runs the native tier on x86-64 Linux, aarch64 Linux and aarch64 macOS, and the `scalar` and `avx2` tiers
on x86-64 Linux.

GitHub's x86-64 runners have AVX2 but no AVX-512 or AVX-VNNI, so the SDE jobs run the kernel tests and the
synthetic models under Intel's Software Development Emulator:
- `sde64 -spr` (Sapphire Rapids: AVX-512 F/BW/VL, VNNI, GFNI, VBMI) fails unless the tier reported is `x86_64-avx2-avx512-vnni-gfni-vbmi`;
- `sde64 -adl` (Alder Lake: AVX-VNNI 256-bit, no AVX-512) fails unless the tier reported is `x86_64-avx2-vnni256`.

The jobs download SDE 10.13.1 from Intel's official mirror, check the pinned SHA-256, and cache the unpacked emulator.
SDE is free of charge for any type of use under the Intel Simplified Software License. Both jobs run with
`PEREGRINE_GOLDEN=require`, so every synthetic model must have a golden for its tier; the AVX-512 ones
live under `tests/golden/synthetic/x86_64-avx2-avx512-vnni-gfni-vbmi/` and are recorded under
`sde64 -spr`. The real-model goldens under `tests/golden/<tier>/` need the model files, which CI does
not have, so an AVX-512 host still records those itself.

## OpenAI API compatibility

`tests/compat/compat.py` starts the real server on SmolLM2-360M-Instruct and checks the HTTP contract:
`/v1/models`, chat completions non-stream and SSE, `stop`, `max_tokens`, `n`, `logprobs`, error
bodies, eight concurrent requests, and (with `--thinking` and a Qwen 3 file) the reasoning split. It
is standard-library Python and can point at a running llama-server with `--base-url` for a
comparison. CI runs the SmolLM2 pass; see `docs/compat.md` for the checks and the recorded
differences from the OpenAI specification and llama-server, and `tests/compat/README.md` for the
commands.
