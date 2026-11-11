# `bert` (MiniLM, BGE, E5, GTE, bge-m3)

Reference note for roadmap **S4.6, part a**, transcribed from **llama.cpp v0.9.0,
commit `cd959ba`**: `src/models/bert.cpp` (the graph), `src/llama-model.cpp` (the tensors,
`causal_attn`, `pooling_type`), `src/llama-graph.cpp` (`build_inp_pos`,
`build_attn_inp_no_cache`, `build_norm`, `build_ffn`), `src/llama-hparams.h`,
`src/llama-arch.cpp` and `conversion/bert.py` (token types, the XLM-R position table). Every
detail below was read from that source, not from the design note. Peregrine's implementation
is `crates/peregrine-models/src/bert/mod.rs` (`Bert::load`, `forward_texts`), the pooling in
`crates/peregrine-core/src/embed.rs` and the element-wise kernels in
`crates/peregrine-kernels/src/vmath.rs`.

## What it is

`architecture = "bert"` is the post-norm bidirectional encoder that every encoder-only
embedding model in llama.cpp's format ships as: all-MiniLM-L6/L12, bge-small/base/large-en-v1.5,
e5, gte, and the XLM-R based bge-m3 and multilingual-e5 (which differ only in width, depth and
vocabulary). The same graph with a classifier head (`cls`, `cls.output`) is the reranker /
classifier half of the step and is **not** part of S4.6a.

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
| `cls.weight` / `cls.bias`, `cls.output.weight` / `.bias` | | the head: S4.6b, not read here |

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

There is no AVX2 body yet: `// S4.9h:` marks where it goes, and the note says its lane
arithmetic must reproduce these sequences exactly.

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

`/v1/embeddings` on H0 (8 vCPU Xeon E5-2680 v4, AVX2+FMA, **shared host**: load average 9–12
during the runs), text of ~16/128/512 tokens, medians of three, llama.cpp v0.9.0 on the same file
and thread count:

| model, threads | tokens | texts/s (ours vs llama.cpp) |
|---|---|---|
| MiniLM F16, 4 | 16 | 33.0 vs 22.0 |
| | 128 | 6.0 vs 17.8 |
| | 512 | 1.70 vs 5.37 |
| bge-small Q8_0, 4 | 16 | 39.7 vs 10.3 (llama's first requests were cold) |
| | 128 | 5.5 vs 10.3 |
| | 512 | 0.76 vs 2.21 |
| bge-small Q8_0, 8 | 16 | 16.5 vs 4.3 (p50 9.3 ms vs 11 ms at batch 1) |
| | 512 | 1.24 vs 3.12 |

Batches of 8/32/128 change the picture only a little: parity at 16 tokens (105/179/156 texts/s
vs llama's 133/182/161), 1.8× behind at 128 and 2.5–3.6× at 512, at every batch size. The gap is
**length**, not batch, which is the signature of the two `n²`/element-wise pieces this step does
not contain: the bidirectional attention strip (S4.9g) and the vector erf-GELU/LayerNorm (S4.9h).

Temporary per-phase timers on one 420-token MiniLM text (4 threads, 636 ms):

| phase | ms | share |
|---|---|---|
| attention + Q scale + bias adds | 348 | 55 % |
| q/k/v + embedding lookup + embedding LayerNorm | 124 | 19 % |
| `attn_output` + `attn_output_norm` | 64 | 10 % |
| `ffn_up` (+ bias) | 54 | 8 % |
| the scalar GELU (420 × 1536 × 6) | 30 | 5 % |
| `ffn_down` + `layer_output_norm` | 18 | 3 % |
| pooling | 0.1 | — |

Single-threaded the same text is 1321 ms with attention at 74 %. That is what the S4.9 design
predicted for this family ("35 % of the pass at 128 tokens, 60 % at 512" on the attention loop,
"a scalar erf would cost half the matrix time"), and the two kernels that remove it are S4.9g
and S4.9h.

## What Peregrine does

- `Bert::forward_texts` runs one whole text per unit (never chunked, no KV cache), with all
  texts of a pass side by side in the shared GEMM; a layer's activation is quantised once per
  distinct input (the normed rows for q/k/v, the same rows for `ffn_up`, the GELU output for
  `ffn_down`) and the matrices that share it run in one fork-join (S4.9a). F32/F16 files go
  through the packed float GEMM (S4.9f), Q8_0 files through the int8 tiles; biases are added
  after the matmul, as llama.cpp does.
- The family refuses a text longer than the position table, naming it; chat and label requests
  on it are refused by the engine with the model's name (`is an embedding model …`).
- Pooling is the text's own rows in position order (`embed::pool_rows_into`), then `finish`'s
  truncation and L2 normalisation.
- The exact GELU and the defined LayerNorm above; no f16 lookup table.
- Tests: a synthetic `bert` file in `tests/support/models.rs` (WordPiece and unigram variants,
  cls and mean pooling, F32 and Q8_0), the independent scalar reference in
  `tests/support/bert_ref.rs`, the invariance matrix, server/CLI tests, the no-allocation test
  and one golden per pinned x86 tier. The 200-text fixtures are under
  `crates/peregrine-models/tests/fixtures/bert/` with the ignored real-file test in
  `tests/embeddings.rs`.
