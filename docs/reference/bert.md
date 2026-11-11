# `bert` (MiniLM, BGE, E5, GTE, bge-m3)

Reference note for roadmap **S4.6, part a** (the family) and **part b** (the classifier head,
`/v1/rerank` and `/v1/classify`), transcribed from **llama.cpp v0.9.0,
commit `cd959ba`**: `src/models/bert.cpp` (the graph), `src/llama-model.cpp` (the tensors,
`causal_attn`, `pooling_type`), `src/llama-graph.cpp` (`build_inp_pos`,
`build_attn_inp_no_cache`, `build_norm`, `build_ffn`), `src/llama-hparams.h`,
`src/llama-arch.cpp` and `conversion/bert.py` (token types, the XLM-R position table). Every
detail below was read from that source, not from the design note. Peregrine's implementation
is `crates/peregrine-models/src/bert/mod.rs` (`Bert::load`, `forward_texts`, `BertHead`,
`head_row`), the pooling in `crates/peregrine-core/src/embed.rs`, the head slot in
`crates/peregrine-core/src/serve.rs` (`Want::Head`) and the element-wise kernels in
`crates/peregrine-kernels/src/vmath.rs`.

## What it is

`architecture = "bert"` is the post-norm bidirectional encoder that every encoder-only
embedding model in llama.cpp's format ships as: all-MiniLM-L6/L12, bge-small/base/large-en-v1.5,
e5, gte, and the XLM-R based bge-m3 and multilingual-e5 (which differ only in width, depth and
vocabulary). The same graph with a classifier head (`cls`, `cls.output`) is the reranker /
classifier half of the step, S4.6b, which has its own section below.

Three functions are text-in, one vector-out: `<arch>.embedding_length` wide, L2-normalised.

## Metadata

All model keys are under the `bert.` prefix. `causal_attention` is written by the converter as
the key `attention.causal` (`conversion/bert.py` calls `add_causal_attention(False)`); llama.cpp
reads `LLM_KV_ATTENTION_CAUSAL` into `hparams.causal_attn`, and the graph's mask is built from
it. The reference files below are the ones the S4.6a accuracy gate used.

| Key | MiniLM-L6-v2 F16 | bge-small F32 | bge-m3 Q8_0 | Meaning |
|---|---|---|---|---|
| `block_count` | 6 | 12 | 24 | layers |
| `embedding_length` | 384 | 384 | 1024 | `d_model` |
| `feed_forward_length` | 1536 | 1536 | 4096 | FFN width |
| `attention.head_count` | 12 | 12 | 16 | query heads |
| `attention.head_count_kv` | absent | absent | absent | KV heads = heads (no GQA) |
| `attention.key_length` | absent | absent | absent | head width = `d_model / n_head` |
| `context_length` | 512 | 512 | 8192 | rows of the position table |
| `attention.layer_norm_epsilon` | 1e-12 | 1e-12 | 1e-5 | LayerNorm epsilon |
| `attention.causal` | false | false | false | no causal mask |
| `pooling_type` | 1 (mean) | 2 (cls) | 2 (cls) | see below |
| `vocab_size` | absent | absent | absent | the tokenizer's vocabulary |
| `tokenizer.ggml.model` | `bert` | `bert` | `t5` | WordPiece / SentencePiece unigram |
| `tokenizer.ggml.token_type_count` | 2 | 2 | 1 | rows of `token_types.weight` |

`pooling_type` is written by the converter from the sentence-transformers pooling module
(`conversion/base.py`, `_try_set_pooling_type`): `mean`, `cls` or `lasttoken` → 1, 2, 3. Most
BGE files are `cls`; MiniLM and the E5/GTE family are `mean`.

The reference files and their SHA-256:

| File | Repo | SHA-256 |
|---|---|---|
| `all-MiniLM-L6-v2-ggml-model-f16.gguf` | `second-state/All-MiniLM-L6-v2-Embedding-GGUF` | `797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662` |
| `all-MiniLM-L6-v2-Q8_0.gguf` | same | `263215c3cadd6e16740741a7624ab4cbb6c8e777688bd5331ecfbf5681c2f8ed` |
| `bge-small-en-v1.5-f32.gguf` | `CompendiumLabs/bge-small-en-v1.5-gguf` | `bf40c42ad7d89382e9ba7376d5c4b73f6b556cb541fab37aaa1da9c320149b65` |
| `bge-small-en-v1.5-q8_0.gguf` | same | `ec38e8da142596baa913124ae50550de284b6916bf59577ef2f0cb9660c2f514` |
| `bge-m3-Q8_0.gguf` | `gpustack/bge-m3-GGUF` | `950f4a8e5e19477a6d3c26d2f162233c20002c601f75e4b002e3239997821167` |

## Tensors

| Tensor | Shape | Notes |
|---|---|---|
| `token_embd.weight` | `[n_embd, n_vocab]` | |
| `token_types.weight` | `[n_embd, n_token_types]` | optional; only row 0 is ever read |
| `position_embd.weight` | `[n_embd, n_pos]` | learned absolute positions |
| `token_embd_norm.weight` / `.bias` | `[n_embd]` | the embedding LayerNorm |
| `blk.N.attn_q.weight` | `[n_embd, n_embd]` | |
| `blk.N.attn_q.bias` | `[n_embd]` | optional in llama.cpp, present in every file |
| `blk.N.attn_k.weight` / `.bias` | `[n_embd, n_embd]` | |
| `blk.N.attn_v.weight` / `.bias` | `[n_embd, n_embd]` | |
| `blk.N.attn_output.weight` / `.bias` | `[n_embd, n_embd]` | |
| `blk.N.attn_output_norm.weight` / `.bias` | `[n_embd]` | post-attention LayerNorm |
| `blk.N.ffn_up.weight` / `.bias` | `[n_embd, n_ff]` | |
| `blk.N.ffn_down.weight` / `.bias` | `[n_ff, n_embd]` | |
| `blk.N.layer_output_norm.weight` / `.bias` | `[n_embd]` | post-FFN LayerNorm |
| `cls.weight` / `cls.bias`, `cls.output.weight` / `.bias` | | the classifier head: see below |

Every projection has a bias, and every norm has a weight **and** a bias: this family is
LayerNorm, not RMSNorm. There is no output head and no final norm after the last layer — the
last layer's `layer_output_norm` output *is* `result_embd`, which pooling reads.

`n_token_types` comes from `tokenizer.ggml.token_type_count`; llama.cpp's loader throws
`"<arch> model needs to define token type count"` when it is 0, and the graph adds row 0 of
`token_types.weight` to every token ("Sentence A", hardcoded).

### The XLM-R position table

XLM-R's position embeddings start at `pad_token_id + 1` = 2 in the Hugging Face model.
llama.cpp's converter removes the offset by **chopping the table**
(`conversion/bert.py`, `RobertaModel::modify_tensors`: `data_torch = data_torch[self._position_offset:, :]`)
and decrementing `max_position_embeddings` by the same amount before writing `context_length`.
The graph therefore indexes the table with the sequence's own 0-based positions
(`build_inp_pos` is the plain position ids) and applies no offset. Peregrine does the same, so
for this family **the position table's row count is the text limit, with no offset**; a file
whose `context_length` exceeds the table is refused at load (llama.cpp would read past the
tensor), and a text longer than the table is refused by name (the server and the CLI compare
against `-c`, which is clamped to the file's context).

## The forward, in order

`llama_model_bert::graph` (`src/models/bert.cpp`), for one sequence of `n` tokens:

1. `embd_pos = position_embd[pos]`; `inpL = token_embd[token]`.
2. `inpL = inpL + token_types[0]` (a 1-d view of row 0).
3. `inpL = embd_pos + inpL` — in that order, which is the order Peregrine adds them in
   (an F32 reference that adds them in another order differs in the last bits).
4. `inpL = norm(inpL, token_embd_norm, token_embd_norm.bias)` — LayerNorm.
5. Per layer `il` (post-norm: there is **no** pre-attention norm, the layer input is the
   previous layer's normalised output):
   1. `Qcur, Kcur, Vcur = qkv(inpL)` with their biases; no RoPE, no q/k norm, no GQA
      (`n_embd_head = n_embd_head_v`, `n_head_kv = n_head`).
   2. `cur = attn(Qcur, Kcur, Vcur)` with `kq_scale = 1/sqrt(n_embd_head)`, then
      `attn_output` and its bias.
   3. `cur = cur + inpL` (the **unnormalised** layer input).
   4. `cur = norm(cur, attn_output_norm[.bias])`.
   5. `ffn_inp = cur`; `cur = ffn_up(cur) + bias`; `cur = gelu(cur)`;
      `cur = ffn_down(cur) + bias` (`LLM_FFN_GELU` with `LLM_FFN_SEQ`: one projection, the
      activation, then the other — not a gated pair).
   6. `cur = cur + ffn_inp`.
   7. `cur = norm(cur, layer_output_norm[.bias])` → the next layer's input.
6. `result_embd = cur` (the last `layer_output_norm`'s output).

Then pooling and normalisation (below).

### The mask

`build_attn_inp_no_cache` builds a full `n_tokens × n_tokens` mask from `cparams.causal_attn`,
which the file sets false. `fill_mask` masks pairs from different sequences and, when causal,
`p0 > p1`; with `causal = false` and no sliding window (`swa_type = NONE` for this arch) every
key of a text is visible to every query. No KV cache is written: the batch's own K/V rows are
read as float.

## Pooling and normalisation

`llama_model::build_graph` calls `build_pooling` with `hparams.pooling_type`:

- `MEAN` (1): the arithmetic mean of the text's rows;
- `CLS` (2): row 0 of the text;
- `LAST` (3): the last row.

Then `common_embd_normalize(..., 2)`: the sum of squares in `f64`, `sqrt`, `1/sum`, and each
component multiplied by that one `f32` factor. Peregrine's `embed::finish` is the same
arithmetic, after the caller's optional Matryoshka truncation. `pooling_type = 0` (none) has no
single-vector meaning for `/v1/embeddings`; a request may still name `cls`, `mean` or `last`
(it is a reduction over rows the model computed, not a weight).

## GELU and LayerNorm: Peregrine's definitions

llama.cpp v0.9.0's GELU is `ggml_gelu_f32`, which with `GGML_GELU_FP16` (its default) converts
the input to f16, looks up a 64 K-entry table of `0.5·x·(1+tanh(√(2/π)(x+0.044715x³)))`
evaluated at that input, and converts the result back (`ggml-cpu/vec.h`,
`ggml_vec_gelu_f32`, `ggml-cpu.c` builds `ggml_table_gelu_f16`). That is the **tanh**
approximation of GELU, computed through f16. A BERT checkpoint trains with the **exact**
function, `torch.nn.functional.gelu`, i.e. `x·Φ(x) = 0.5·x·(1+erf(x/√2))`. Peregrine
evaluates that exact form, so its activation is closer to the model's own arithmetic than
llama.cpp's; the measurements below show it (bge-small F32 against the fp32 PyTorch reference:
1.0000000, where llama.cpp is 0.9999963).

Two definitions live in `peregrine-kernels/src/vmath.rs` and are shared with the test
reference — their arithmetic and order are part of the result:
- **`erf(x)`** (`vmath::erf`). `x` is f32; every intermediate is f64 and the result is rounded
  to f32 once. Range split, all on `a = |x|`: 10 terms of the Maclaurin series
  `erf(x) = (2/√π)·x·Σ_k (−1)^k x^(2k)/(k!(2k+1))` for `a ≤ 1`, 22 terms for `1 < a ≤ 2`, 36
  terms for `2 < a ≤ 3`, and above 3 the eight-term asymptotic `erfc` (`erfc(3) = 2.2e-5`, and
  the series is at its smallest term there), with `±1` above 4.6. Measured against an
  independent oracle (the positive-term series
  `erf(x) = 2/√π·e^{−x²}·Σ (2x²)^n x/(1·3···(2n+1))`, summed in f64) over 400 001 points in
  `[−6, 6]`: **max absolute error 5.44e-8 (at |erf| ≈ 0.5) and max 1 ulp** of the f32 result — the error is the
  final rounding wherever erf is near ±1, and the series is ~1e-15 relative near 0. The test
  `erf_agrees_with_an_independent_oracle` pins it.
- **`gelu_erf(x)`** = `(0.5·x)·(1 + erf(x·(1/√2)))` in f32, in that order. Measured against
  the f64 definition over 1 001 points in `[−12, 12]`: max absolute error 1.2e-6 at
  `|gelu| ≈ 5` (three f32 roundings, ~2.5 ulp).
- **`layer_norm_row(x, w, b, eps, out)`**: `m = Σ x_i / n` and
  `v = Σ (x_i − m)² / n` each accumulated in `f64` in index order (one accumulator, no
  partials — a vector body must add into a single accumulator in index order or the bits
  move); `inv = 1/√(v + eps)` in f64, rounded to f32 once; then
  `out_i = ((x_i − m_f32)·inv)·w_i + b_i`, three plain f32 operations, deliberately not fused.
  `m` is rounded once for the shift and unrounded for the variance (the two-pass form).

The AVX2 bodies (S4.9h) reproduce these sequences lane for lane, so they are `to_bits` equal to
the scalar twins. `gelu_erf_rows` takes sixteen elements per step, four `f64` lanes per `erf`,
and evaluates the three polynomials above as three Horner chains selected per lane; a block of
sixteen takes the shortest path every lane in it can take. `layer_norm_rows` takes eight rows per
step, one `f64` lane per row, so a row's sums stay in index order, and does the scale-and-shift
eight `f32` lanes along the row. A `layer_norm_rows` row holding an infinity or a NaN produces a
NaN on both paths, but the NaN payload is not part of the definition (narrowing an `f64` NaN to
`f32` differs between `cvtsd2ss` and `vcvtpd2ps`); every finite row is bit-identical. aarch64
keeps the scalar twins and has a `// S4.9h:` note where the NEON bodies go.

## Accuracy: against llama.cpp and against the truth

Each engine's vectors are compared with the **fp32 PyTorch reference**
(`sentence-transformers`, CPU torch, `normalize_embeddings=True`, `max_seq_length = 512` — the
model cards' 256-token default would truncate the long texts and is not the comparison), on
200 mixed-length English texts of 16–428 tokens. Cosine of L2-normalised vectors:

| File | vs llama.cpp (min / mean) | vs fp32 reference, Peregrine (min / mean) | vs fp32 reference, llama.cpp (min / mean) |
|---|---|---|---|
| MiniLM F16 | 0.9999960 / 0.9999979 | **0.9999996 / 0.9999998** | 0.9999952 / 0.9999975 |
| MiniLM Q8_0 | 0.9997713 / 0.9999023 | **0.9991371 / 0.9994906** | 0.9991341 / 0.9994718 |
| bge-small F32 | 0.9999963 / 0.9999987 | **1.0000000 / 1.0000000** | 0.9999963 / 0.9999987 |
| bge-small Q8_0 | 0.9998677 / 0.9999233 | **0.9997426 / 0.9998402** | 0.9997560 / 0.9998390 |
| bge-m3 Q8_0 | 0.9990365 / 0.9997504 | **0.9968921 / 0.9991315** | 0.9966748 / 0.9991207 |

Token ids are identical to llama.cpp's (`/tokenize` with `add_special`, i.e. `[CLS] … [SEP]` or
`<s> … </s>`) on all 200 texts for every file: WordPiece and the XLM-R unigram vocabulary both.

The F16/F32 files are at the 0.9999-class bound. The Q8_0 files are quantised: their distance
from the unquantised model is the quantisation's, for both engines, and the meaningful
comparison is which engine is closer to the reference. Peregrine's *minimum* is above
llama.cpp's on MiniLM Q8_0 (+3.0e-6) and bge-m3 (+2.2e-4) and 1.3e-5 below it on bge-small,
and its mean is above llama.cpp's on all three (+1.9e-5, +1.2e-6, +1.1e-5). Per text, 3 of 200
texts on MiniLM Q8_0, none on bge-small and 12 on bge-m3 are more than 1e-4 *further* from the
reference than llama.cpp's (worst margins 2.1e-4 and 7.9e-4) — the two engines' int8 activation
arithmetic differs by that much on individual texts even though the aggregate is level or
better, which is why the S4 gate states the bound on the minimum and the mean.

The bge-small F32 row is the strongest statement about the graph: against the fp32 reference
Peregrine's vectors round to the reference's to the printed precision on every text, while
llama.cpp's GELU-through-f16 keeps it at 0.9999963.


## The classifier head (S4.6b)

A `bert` file may carry a classification head on top of the encoder. llama.cpp reads it from
four optional tensors and three optional keys, and only when the context's pooling type is
`rank` (`4`): the head is not applied to a `cls`-pooled embedding.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `bert.pooling_type` | u32 | absent = 0 (none) | `4` is `rank`: pool one row, then the head (`llama-arch.cpp:461`, read at `llama-model.cpp:1473`) |
| `bert.classifier.output_labels` | array of string | absent | the labels, which also size the head's output (`llama-model.cpp:1634-1637`) |
| `bert.classifier.activation` | string | absent = `tanh` | `gelu` (the exact erf form), `silu` or `tanh` (`llama-model.cpp:1475-1480`, `1209-1213`) |
| `bert.classifier.pooling_type` | u32 | absent = row 0 | `1` (mean) pools the mean before the head (`llama-graph.cpp:4187-4195`) |

The tensors (`src/models/bert.cpp:34-38`, all `TENSOR_NOT_REQUIRED`):

| Tensor | Shape | Notes |
|---|---|---|
| `cls.weight` | `[n_embd, n_embd]` | the head's dense layer (Hugging Face's `classifier.dense`) |
| `cls.bias` | `[n_embd]` | |
| `cls.output.weight` | `[n_embd, n_cls_out]` | the projection to the file's outputs; written **1-d** (`[n_embd]`) when `n_cls_out` is 1, as `bge-reranker-v2-m3-Q8_0.gguf` does |
| `cls.output.bias` | `[n_cls_out]` | |

`n_cls_out` is the number of labels when the file names them, and `1` otherwise
(`llama-hparams.h:271`). A file with `cls` but no `cls.output` (jina-reranker-v1-tiny-en) is a
one-layer head and its row is `n_embd` wide. `cls.norm` exists as a tensor name
(`llama-arch.cpp:603`) but `bert`'s loader does not create it — only `modern-bert` and `lfm2` do
— so a `bert` file that carries it loads with an "unused tensor" warning and the norm is **not**
applied; Peregrine ignores it too. `classifier.activation` is read for every architecture, and its
default is `GGML_UNARY_OP_TANH` (`llama-hparams.h:429`): a BERT reranker's head is
dense → **tanh** → output, which is what bge-reranker's checkpoint does.

The head's row, transcribed from `llm_graph_context::build_pooling`
(`llama-graph.cpp:4185-4226`), for one sequence:

1. pool row 0 of the text (`build_inp_cls`), or the mean of its rows when
   `bert.classifier.pooling_type = 1` (`4187-4195`);
2. if `cls`: `h = cls.weight · pooled + cls.bias`, then `h = activation(h)` (`4199-4204`);
   `cls.norm`, when the arch has it, follows the activation (`4205-4208`);
3. if `cls.output`: `y = cls.output.weight · h + cls.output.bias` (`4215-4220`);
4. a softmax **only** for Qwen 3 generative rerankers (`4222-4225`), never for `bert`.

The result is `result_embd_pooled` (`4233-4234`), and a `rank` context extracts `n_cls_out`
floats per sequence (`llama-context.cpp:1887-1901`).

**Which files declare `rank`.** llama.cpp's converter never writes `pooling_type = 4` for a
`bert` file: it writes what the sentence-transformers pooling module says (`conversion/base.py`,
`_try_set_pooling_type`), and the reranker uploads therefore carry **no** `bert.pooling_type` at
all (the bge-reranker files below have none). llama-server is told instead:
`--reranking` sets `embedding` and `pooling_type = RANK` (`common/arg.cpp:3472-3476`), and
`--pooling rank` does the same directly (`2264-2271`); the route refuses without it
(`server-context.cpp:7009-7011`). Peregrine therefore reads the head from the tensors — a file
with `cls` or `cls.output` has one — and serves `/v1/rerank` and `/v1/classify` from it, rather
than requiring a flag for what the file's tensors already say.

### `/v1/rerank`

The route is served at four paths, `/rerank`, `/reranking`, `/v1/rerank` and `/v1/reranking`
(`tools/server/server.cpp:195-198`), by one handler, `handle_rerank_impl`
(`tools/server/server-context.cpp:7007-7155`):

1. **The pair text.** One task per document (`7106-7119`), and each task's tokens are
   `format_prompt_rerank` (`tools/server/server-common.cpp:1973-2045`):
   - if the file carries a `rerank` chat template (a generative reranker, S4.13), the template is
     rendered with `{query}` and `{document}` and tokenized **with** special tokens
     (`1987-2012`);
   - otherwise the query and the document are tokenized **separately**, with
     `add_special = false, parse_special = false` (`2021-2022`), and joined as
     `[BOS] query [EOS] [SEP] document [EOS]`, where each bracket is present only when the
     vocabulary says so and `EOS` falls back to `SEP` when the file has no EOS (`2023-2042`).
     For a **BERT WordPiece** vocabulary that is `[CLS] query [SEP] document` (the file's BOS is
     `[CLS]`, its EOS is absent and `add_sep` is on: `llama-vocab.cpp:2785-2789`); for an
     **XLM-R unigram** vocabulary it is `<s> query </s> document </s>` (`add_bos` and `add_eos`,
     no `add_sep`). `Tokenizer::encode_pair` is this function, and
     `tests/fixtures/bert/rerank/*.ids` holds its output for 320 pairs.
2. **The score.** `send_rerank` reads the pooled tensor's first float,
   `res->score = embd[0]` (`server-context.cpp:2599-2626`): the **raw logit**, with no sigmoid
   and no normalisation. bge-reranker-v2-m3's `Paris is the capital of France.` against
   `What is the capital of France?` scores `8.6065` — a client that wants a probability applies
   its own sigmoid.
3. **The response** (`server-common.cpp:1613-1657`): `{"model", "object": "list", "usage":
   {"prompt_tokens", "total_tokens"}, "results": [{"index", "relevance_score"}]}`, sorted by
   score descending, cut to `top_n` (which defaults to every document), and `usage.prompt_tokens`
   is the sum of each pair's token count (`tokens_evaluated`, `server-task.cpp:1501-1512`). The
   TEI shape (`"texts"` in the request) returns a bare array of `{"index", "score"}`; Peregrine
   serves the Jina shape only and refuses `texts` by name.
4. **The request.** `query` and each document may be a string or an object with `text` (a
   string or an array of strings, joined); `image`/`video` need a multimodal projector
   (`7021-7060`); `top_n` must be non-negative (`7098-7103`).

`/v1/classify` has no llama.cpp counterpart: llama.cpp's `/v1/embeddings` returns the pooled row
of a `cls` file, not the head's outputs. Peregrine's shape is the design's
(`docs/design/s4-embeddings-decisions.md` §3): one entry per input, the softmax of the head's
outputs (a sigmoid when the file has one output and names it), every label and its probability,
and the most likely label.

## Accuracy of the head

`crates/peregrine-models/tests/rerank_real.rs` runs 320 (query, document) pairs — 40 queries of
eight documents, in English, Chinese, French, German, Spanish, Russian, Arabic and Japanese,
mixing 16- and 428-token documents — through both engines on the same files. The fixtures are
`crates/peregrine-models/tests/fixtures/bert/rerank/`.

| File | Pairs | Token ids | Score difference (mean / p95 / max) | Order |
|---|---|---|---|---|
| `bge-reranker-v2-m3-Q8_0.gguf` | 320 | identical (26,175 ids) | 0.0107 / 0.0588 / 0.186 | identical |
| `bge-reranker-base-q8_0.gguf` | 319 (one pair is past its 512-token table) | identical | 0.0103 / 0.0708 / 0.245 | identical |

"Identical" order means no two documents of a query swapped outside a near-tie, which the test
defines as scores within **1e-2** of each other in either engine.

The design's 1e-3 was an F16-class number and is **not** what a Q8_0 encoder can deliver, by
either engine. Measured on the same pairs:

| Comparison | mean abs | max abs |
|---|---|---|
| llama.cpp Q8_0 vs llama.cpp FP16 | 0.0158 | 0.222 |
| Peregrine Q8_0 vs Peregrine FP16 | 0.0174 | 0.282 |
| Peregrine Q8_0 vs llama.cpp Q8_0 | 0.0107 | 0.186 |
| Peregrine FP16 vs llama.cpp FP16 | 0.0047 | 0.0688 |
| Peregrine FP16 with llama.cpp's GELU substituted vs llama.cpp FP16 | 0.00056 | 0.0067 |

So each engine's own quantisation moves its scores twice as far as the two engines differ from
each other, and the residue at the same precision is one thing: llama.cpp's activation. Its GELU
is the tanh approximation evaluated through a 64 K-entry f16 table (`ggml-cpu/vec.h:999-1013`,
`ggml-cpu/ggml-cpu.c:3908`), whose f16 input rounding alone is ~5e-4 relative; this family
evaluates the exact `erf` form the checkpoints train with (ADR 0003, and the section above), and
that difference compounds through 24 layers into 5e-3 of a logit's score. Substituting
llama.cpp's own GELU into our pass brings the pair to 5.6e-4 mean (0.0067 max), which is the
matrix and attention arithmetic alone: the head is the reference's, bit for bit, on the synthetic
files. The gate for this family is therefore the measured one above, and the order is the property
that matters for a reranker.

## What the other BERT-like families do differently (not this step)

Read from the same source, one line each; `nomic-bert`/`nomic-bert-moe` and `jina-bert-v2/v3`
are roadmap S4.8 and `modern-bert` is S4.7:

- `nomic-bert`: pre-norm blocks with **RMSNorm** and SwiGLU instead of post-norm LayerNorm and
  GELU; q/k/v have no biases (or all of them, for the MoE variant); **RoPE** on q and k instead
  of a learned position table; attention has a symmetric sliding window; there are no
  `attn_output_norm`/`layer_output_norm` tensors (`src/models/nomic-bert.cpp`).
- `modern-bert`: pre-norm RMSNorm, alternating local (128-token symmetric) and global
  attention, **RoPE** with a local/global base, an ffn pair whose `ffn_up` may hold the gate
  half (`GeGLU`), and no biases (`src/models/modern-bert.cpp`).
- `jina-bert-v2`: a `bert` with a different FFN pair (`ffn_up` holding the gate half →
  `GeGLU`) and optional per-head q/k norms, still post-norm (`src/models/bert.cpp`, the
  `JINA_BERT_V2` branches).
- `jina-bert-v3`: pre-norm, RMSNorm, RoPE, no `attn_output_norm`, GELU FFN with biases
  (`src/models/bert.cpp`, the `JINA_BERT_V3` branch).

## Speed, and where it goes

`/v1/embeddings` on H0 (8 vCPU Xeon E5-2680 v4, AVX2+FMA, **shared host**: load average 6.6–10
during the runs), a text of 16/128/512 tokens repeated `batch` times, medians of three, llama.cpp
v0.9.0 on the same file and thread count (main's S4.9g attention strip is in both columns for
Peregrine — it landed before this branch was rebased):

texts/s, 4 threads:

| model | tokens | 1 | 8 | 32 | 128 | llama.cpp (1 / 8 / 32 / 128) |
|---|---|---|---|---|---|---|
| MiniLM-L6-v2 F16 | 16 | 161 | 187 | 206 | 135 | 102 / 185 / 217 / 159 |
| | 128 | 22.5 | 26.1 | 20.5 | 21.7 | 29.5 / 33.6 / 11.1 / 19.9 |
| | 512 | 5.30 | 5.12 | 2.81 | 4.27 | 6.41 / 6.82 / 3.21 / 3.62 |
| bge-small Q8_0 | 16 | 98.8 | 80.1 | 24.4 | 87.0 | 72.8 / 116 / 41.6 / 102 |
| | 128 | 13.4 | 10.9 | 11.2 | 12.3 | 14.2 / 14.4 / 15.0 / 12.1 |
| | 512 | 2.47 | 2.47 | 2.31 | 1.95 | 3.27 / 3.31 / 3.09 / 3.06 |

At 8 threads, bge-small Q8_0: 16 tokens 16.5 vs 4.3 texts/s (llama's 8-thread run on this host
was 10× slower than its 4-thread one, load 12 — the 8-thread llama column is not comparable), 128
tokens 6.8 vs 6.6, 512 tokens 1.24 vs 3.12. Single 16-token text p50: **7.2 ms** vs llama.cpp's
9.2 ms on MiniLM, 11.7 ms vs 16.2 ms on bge-small (4 threads).

So: ahead on single short texts (the p50 row matters for guard and router models), level at
batches of short texts, and behind at 128 and 512 tokens. The residue there is the matrix work
and attention's tail. S4.9h vectorised the exact GELU and the per-row LayerNorms (each with a
bit-identical scalar twin) and stopped a float file's activation fills from building int8
layouts nothing reads.

Temporary per-phase timers on one 420-token MiniLM text, warm passes, 4 threads. The S4.6a
column is the note's own measurement at about 170 ms; the S4.9h column is the minimum of 12 warm
passes on the same host at load average 6.6-6.8 (minimising the shared host's noise):

| phase | S4.6a ms (share) | S4.9h ms (share) |
|---|---|---|
| embedding lookup | 40 (23 %) with norm and fills | 1.2 (1 %) |
| embedding LayerNorm + fill | — | 0.2 (0 %) |
| attention (the S4.9g strip) | 44 (26 %) | 27.9 (29 %) |
| q/k/v | 17 (10 %) | 15.5 (16 %) |
| `attn_output` + bias + residual | 20 (12 %) with norm | 6.0 (6 %) |
| `attn_output_norm` + fill | — | 0.6 (1 %) |
| `ffn_up` | 25 (15 %) | 20.7 (21 %) |
| the exact GELU + fill | 16 (10 %) | 8.6 (9 %) |
| `ffn_down` + bias + residual | 8 (5 %) with norm | 15.7 (16 %) |
| `layer_output_norm` + fill | — | 0.6 (1 %) |
| pooling | 0.1 | 0.1 |
| **pass** | **~170** | **97.0** |

The S4.6a column's rows group the phases differently and the host's load differs, so only the
shares of the element-wise rows are directly comparable: the GELU, the LayerNorms and the fills
were about a third of the pass and are now about a tenth. The embedding phase's 40 ms was the
serial int8 quantisation `QuantAct::fill` performed for a file whose matrices are F16; a pure
float file now keeps only the f32 row copy, and the gather is 1 % of the pass.

### Rerank throughput (S4.6b)

`bge-reranker-v2-m3-Q8_0.gguf`, 4 threads, one query against N documents, medians of three,
both engines asked over HTTP on the same file, one engine at a time
(`scripts/rerank-bench.py`). Documents of about 123 and 449 tokens; the count in the table is
what the response's `usage` reports, per pair.

| tokens/pair | documents | Peregrine pairs/s | llama.cpp pairs/s | ratio |
|---|---|---|---|---|
| 123 | 8 | 3.28 | 3.34 | 0.98 |
| 123 | 32 | 3.30 | 3.33 | 0.99 |
| 123 | 100 | 3.32 | 3.29 | 1.01 |
| 449 | 8 | 0.83 | 0.79 | 1.05 |
| 449 | 32 | 0.83 | 0.78 | 1.06 |
| 449 | 100 | 0.83 | 0.78 | 1.06 |

Level at short pairs and 5-6 % ahead at long ones. The 1-2 % at 123 tokens is the pass budget,
not a kernel: the engine runs **one 123-token text per pass** at that size (the profiler's
`rows/step` is 123 and barrier wait is 4 % of the wall), while llama.cpp packs about sixteen pairs
into one 2048-token ubatch; with `PEREGRINE_PASS_ROWS=2048` the same run measures 3.36 pairs/s
against llama.cpp's 3.33. The pass budget is S4.9's to tune (the default was measured for
embeddings, where one text is the normal case), so this step leaves it alone and records the
number. The head itself is not in this: its two projections are 1.3 M MACs against the encoder's
78 G for a 123-token pair.

## What Peregrine does

- The classifier head (`Bert::load`, `BertHead`, `head_row`): `cls` → activation → `cls.output`,
  with the file's `classifier.activation` (tanh by default), its
  `classifier.pooling_type` (row 0, or the mean) and its `classifier.output_labels`. A 1-d
  `cls.output.weight` is accepted as a one-row matrix, which is how the reranker uploads write it.
  A head slot (`Want::Head`) is one `TextWant::Head` text of a bidirectional pass: the head runs
  after the last layer, on that text's rows, and writes `n_cls_out` floats into
  `State::head_out` at the text's first row. Every input it reads is the text's own, so the row is
  bit-identical in any batch (tested on the synthetic files and on the real file).
- `/v1/rerank` (four routes), `/v1/classify`, `peregrine rerank`, and `/v1/models`'
  `output_modalities` (`["rerank"]` for a one-output head, `["classify"]` for a labelled one).
  A file without a head answers 400 by name; a pair past the position table is refused with the
  limit, or truncated with `"truncate": true`, as `/v1/embeddings` does; a generative `rerank`
  chat template is refused by name (S4.13).
- `Bert::forward_texts` runs one whole text per unit (never chunked, no KV cache), with all
  texts of a pass side by side in the shared GEMM; a layer's activation is filled once per
  distinct input (the normed rows for q/k/v, the same rows for `ffn_up`, the GELU output for
  `ffn_down`) and the matrices that share it run in one fork-join (S4.9a). F32/F16 files go
  through the packed float GEMM (S4.9f), Q8_0 files through the int8 tiles; a pure F32/F16/BF16
  file does not build the int8 layouts at all (S4.9h), and the norm or GELU that produces a row
  fills it in the same pool task. Biases are added after the matmul, as llama.cpp does, with the
  attention output's bias and residual fused into the projection's chunk.
- The family refuses a text longer than the position table, naming it; chat and label requests
  on it are refused by the engine with the model's name (`is an embedding model …`).
- Pooling is the text's own rows in position order (`embed::pool_rows_into`), then `finish`'s
  truncation and L2 normalisation.
- The exact GELU and the defined LayerNorm above, with their AVX2 bodies and no f16 lookup
  table; the scalar twins are the reference's.
- Attention is the engine's shared bidirectional block (`blocks::bidir::attend_texts`, now the
  S4.9g strip kernel), the way the float matmul chain is shared: the exact scalar reference
  forward in the tests owns the graph around it and calls the block for the `n × n` attention, so
  a change to that kernel moves both sides together.
- Tests: a synthetic `bert` file in `tests/support/models.rs` (WordPiece and unigram variants,
  cls and mean pooling, F32 and Q8_0), the independent scalar reference in
  `tests/support/bert_ref.rs`, the invariance matrix, server/CLI tests, the no-allocation test
  and one golden per pinned x86 tier. The 200-text fixtures are under
  `crates/peregrine-models/tests/fixtures/bert/` with the ignored real-file test in
  `tests/embeddings.rs`.
