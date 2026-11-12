# `jina-bert-v2` (jina-embeddings-v2-base-en and -small-en)

Reference note for roadmap **S4.8c**, transcribed from **llama.cpp v0.9.0,
commit `cd959ba`**: `src/models/bert.cpp` (the graph, shared by `bert`, `nomic-bert`,
`jina-bert-v2` and the others, with a branch per family), `src/models/jina-bert-v2.cpp`
(the hparams and the tensor list), `src/llama-model.cpp` (the ALiBi flag, the rope type,
the head widths), `src/llama-graph.cpp` (`build_inp_embd`, `build_qkv`, `build_norm`,
`build_attn_inp_no_cache`, `build_attn`, `build_ffn`), `src/llama-hparams.h`,
`ggml/src/ggml-cpu/ops.cpp` and `ggml/src/ggml-cpu/vec.h` (the ALiBi slopes and the
soft-max, the tanh GELU), `conversion/bert.py` (`JinaBertV2Model`, `BertModel`) and
`gguf-py/gguf/tensor_mapping.py` (the names). Every statement below was read from that
source, not from the design note. Peregrine's implementation is
`crates/peregrine-models/src/jina_bert.rs` (`JinaBert::load`, `forward_texts`), the
ALiBi bias in `crates/peregrine-kernels/src/bidir.rs` (`alibi_slopes`, `alibi_row`,
applied by `crates/peregrine-core/src/blocks/bidir.rs`), and the pooling in
`crates/peregrine-core/src/embed.rs`.

## What it is

`architecture = "jina-bert-v2"` is Jina's encoder: jina-embeddings-v2-base-en
(137M) and -small-en (33M). Its block looks like BERT's and is not:

- **no position table and no RoPE.** Positions enter only through **ALiBi**, an
  additive `slope_h · (−|i − j|)` bias inside every attention score.
- the **token-type** embedding's row 0 is added to every token (token types are
  not a request input; the graph hardcodes "Sentence A").
- **one LayerNorm between the attention residual and the feed-forward**, with an
  optional second one (`attn_norm_2`) if the file carries it, and the layer's
  `layer_output_norm` **is** the layer's output — this family has no final output
  norm.
- the feed-forward is a **GEGLU pair**: `ffn_gate` and `ffn_up` are separate
  matrices and `gelu(gate · x) · (up · x)`; a conversion may fuse them into one
  `2·n_ff` matrix whose first half is the gate.
- every projection has a **bias** (the query/key/value ones are optional in
  llama.cpp's loader; the reference files carry them).
- optionally, a LayerNorm over each token's **whole query row** and **whole key
  row** before attention (`attn_q_norm`, `attn_k_norm`); neither reference file has
  them, jina-v2-code's conversion does.

The graph is `src/models/bert.cpp:68-225`; the family's branches are lines 78 (no
position embedding), 193-201 (the GEGLU pair) and the absence of the RoPE call at
131-139. The hparams and tensors are `src/models/jina-bert-v2.cpp:3-61`.

## Metadata

All model keys are under the `jina-bert-v2.` prefix. The reference files are the
S4.8c accuracy gate's.

| Key | jina-embeddings-v2-base-en | Meaning |
|---|---|---|
| `block_count` | 12 | layers |
| `embedding_length` | 768 | `d_model` |
| `feed_forward_length` | 3072 | `n_ff`; `ffn_gate` and `ffn_up` are both this wide |
| `attention.head_count` | 12 | heads; `head_dim = d_model/n_head` |
| `attention.head_count_kv` | absent | KV heads = query heads |
| `context_length` | 8192 | the longest text, in tokens |
| `attention.layer_norm_epsilon` | 1e-12 | every LayerNorm's epsilon |
| `attention.causal` | false | no causal mask |
| `pooling_type` | 1 (mean) | this file's embedding pooling |
| `vocab_size` | 30528 | the tokenizer's vocabulary |
| `tokenizer.ggml.model` | `bert` | WordPiece |
| `tokenizer.ggml.pre` | `jina-v2-en` | ignored by a WordPiece vocabulary |
| `tokenizer.ggml.token_type_count` | 2 | the `token_types` table's rows |
| `tokenizer.ggml.add_bos_token` / `add_eos_token` | true / true | `[CLS] … [SEP]` |

`attention.key_length` and `attention.value_length` are absent; llama.cpp then uses
`d_model/n_head` (`src/llama-model.cpp:1604-1611`). `f_max_alibi_bias` is **not** read from
the file: `llama_model_jina_bert_v2::load_arch_hparams` sets it to `8.0f` as a constant
(`src/models/jina-bert-v2.cpp:5`), and any positive value turns ALiBi on
(`src/llama-model.cpp:1652-1654`). The rope type is `NONE`
(`llama_model_rope_type`, `src/llama-model.cpp:3613`), and the architecture keeps no KV
cache (`src/llama-model.cpp:2891`).

The reference files and their SHA-256:

| File | Repo | SHA-256 |
|---|---|---|
| `jina-embeddings-v2-base-en-FP16.gguf` | `gpustack/jina-embeddings-v2-base-en-GGUF` | `8d29fa0c9b806ae5c165eaf1d8053820760c5d4a3b4f57dd203c426c4680c5f0` |
| `jina-embeddings-v2-base-en-Q8_0.gguf` | same | `123b13ec105de8e0d2becd7dca4f4c46cef3a535f6de9ff1b6ade3fbbf6f5184` |

## Tensors

| Tensor | Shape | Notes |
|---|---|---|
| `token_embd.weight` | `[n_embd, n_vocab]` | |
| `token_types.weight` | `[n_embd, 2]` | **row 0** is added to every token; required by the loader |
| `token_embd_norm.weight` / `.bias` | `[n_embd]` | LayerNorm with a bias |
| `blk.N.attn_q.weight` / `.bias` | `[n_embd, n_embd]` / `[n_embd]` | bias optional in the loader |
| `blk.N.attn_k.weight` / `.bias` | `[n_embd, n_embd]` / `[n_embd]` | |
| `blk.N.attn_v.weight` / `.bias` | `[n_embd, n_embd]` / `[n_embd]` | |
| `blk.N.attn_output.weight` / `.bias` | `[n_embd, n_embd]` / `[n_embd]` | bias required |
| `blk.N.attn_q_norm.weight` / `.bias` | `[n_embd]` / `[n_embd]` | optional; a token's whole query row |
| `blk.N.attn_k_norm.weight` / `.bias` | `[n_embd]` / `[n_embd]` | optional |
| `blk.N.attn_output_norm.weight` / `.bias` | `[n_embd]` | the post-attention LayerNorm |
| `blk.N.attn_norm_2.weight` / `.bias` | `[n_embd]` | optional second residual + norm |
| `blk.N.ffn_gate.weight` | `[n_embd, n_ff]` | optional: absent means the gate is the first half of `ffn_up` |
| `blk.N.ffn_up.weight` / `.bias` | `[n_embd, n_ff]` or `[n_embd, 2·n_ff]` / `[n_ff or 2·n_ff]` | bias optional |
| `blk.N.ffn_down.weight` / `.bias` | `[n_ff, n_embd]` / `[n_embd]` | bias required |
| `blk.N.layer_output_norm.weight` / `.bias` | `[n_embd]` | the layer's output |

`src/models/jina-bert-v2.cpp:17-59` creates exactly this list. There is **no
`output_norm`**: the last `layer_output_norm`'s rows are the model's output. The
reference files carry neither `attn_q_norm`/`attn_k_norm` nor `attn_norm_2` (the
converter maps them from `encoder.layer.N.attention.self.layer_norm_q/k` and
`encoder.layer.N.layer_norm_1`, names only jina-v2-code uses).

## The forward, in order

`llama_model_bert::graph` (`src/models/bert.cpp:68-225`) for
`LLM_ARCH_JINA_BERT_V2`, one sequence of `n` tokens:

1. `inpL = token_embd[token]` (`build_inp_embd`, line 88). `inp_pos` is `nullptr`
   because line 78 skips it, and there is no position add (line 95 is `BERT` only).
2. `inpL += type_embd[0]` — token type 0 for every token (lines 91-94).
3. `inpL = layer_norm(inpL, token_embd_norm.w, .b)` (line 101).
4. Per layer `il`:
   1. `Qcur, Kcur, Vcur = attn_q/k/v · cur (+ their biases)` (line 112;
      `build_qkv`, `src/llama-graph.cpp:1857-1889`; the fused `attn_qkv` layout is
      also accepted, the reference files have separate matrices).
   2. If `attn_q_norm` exists: reshape q to `(n_embd_head·n_head, n_tokens)`, i.e.
      **LayerNorm across one token's whole query**, then back (lines 115-121); the
      same for k (123-129). No RoPE: lines 131-139 apply it only to nomic-bert and
      jina-bert-v3.
   3. `cur = attention(Qcur, Kcur, Vcur)` at `kq_scale = 1/√head_dim` (lines
      145-148), with ALiBi's bias inside the soft-max (below).
   4. `cur = cur + inpL` (the layer's **unnormalised** input, line 157).
   5. `cur = layer_norm(cur, attn_output_norm)` (line 160).
   6. If `attn_norm_2` exists: `cur = cur + inpL` again, then
      `cur = layer_norm(cur, attn_norm_2)` (lines 162-165).
   7. `ffn_inp = cur` (line 167).
   8. The feed-forward (lines 193-201): if `ffn_gate` is absent and `ffn_up` is
      `2·n_ff` wide, the GEGLU is fused (`LLM_FFN_GEGLU`, parallel); otherwise
      `cur = gelu(ffn_gate · cur) · (ffn_up · cur + ffn_up.bias)`.
   9. `cur = ffn_down · cur + ffn_down.bias` (lines 196-200).
   10. `cur = cur + ffn_inp` (line 212).
   11. `cur = layer_norm(cur, layer_output_norm)` (line 215). That is the next
       layer's input (line 218).
5. `result_embd = inpL` (lines 221-224): the last layer's `layer_output_norm` rows.
   Pooling reads them (`<arch>.pooling_type = 1`, the mean over every token,
   specials included).

### ALiBi

The bias is inside `ggml_soft_max_ext`, not an additive tensor
(`build_attn_mha`, `src/llama-graph.cpp:3037-3103`). The no-cache mask is filled with
`-|p0 - p1|` when `hparams.use_alibi` (`src/llama-graph.cpp:609`) and `-inf` nowhere,
because attention is not causal. The CPU soft-max (`ggml/src/ggml-cpu/ops.cpp:5809-5900`)
then computes, per head `h`:

```text
n_head_log2 = 1 << floor(log2(n_head))
m0 = 2^(-max_bias / n_head_log2)              // max_bias = 8
m1 = 2^(-(max_bias/2) / n_head_log2)
slope_h = m0^(h+1)                            h <  n_head_log2
        = m1^(2·(h − n_head_log2) + 1)        h >= n_head_log2
score   = qk·(1/√head_dim) + slope_h·(−|i − j|)
```

The order is exact: `wp = qk · scale` first, **then** `wp += slope · mask`, then the
maximum, the exponentials and the `f64` sum. For 12 heads the slopes are 0.5, 0.25,
0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, then √½, √½³, √½⁵, √½⁷
(the two-group scheme of `ops.cpp:5831-5850`, not the single `2^(-8h/n_head)` formula).
For a single head the group is one head and the slope is `2^-8`.

### GEGLU

`ggml_geglu`/`ggml_glu_impl(..., GGML_GLU_OP_GEGLU, swapped = false)`
(`ggml/src/ggml.c:3062-3066`) acts on the **first** half of a row and multiplies the
second: `y = gelu(x_first) · x_second`. The gelu is the **tanh approximation**,
`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` (`ggml/src/ggml-cpu/vec.h:962-968`).
llama.cpp's build defines `GGML_GELU_FP16` (`vec.h:46`), so the f32 path rounds `x`
through an f16 lookup table first (`vec.h:999-1011`) — a coarser evaluation than
Peregrine's `vmath::geglu`, which is the same formula without the table (see
"What Peregrine does").

### Pooling

`<arch>.pooling_type` is read generically (`src/llama-model.cpp:1473`): 1 mean, 2 cls,
3 last. Both reference files say 1 (mean). The embeddings endpoint L2-normalises the
pooled vector (`common_embd_normalize(..., 2)`).

### The tokenizer

`tokenizer.ggml.model = "bert"` is llama.cpp's WordPiece path
(`LLAMA_VOCAB_TYPE_WPM`, `src/llama-vocab.cpp:2283-2296`): lower-case and
strip-accents on, the phantom-space vocabulary, `[CLS]` at the front and `[SEP]` at
the end when `add_bos_token` is set. The file's `tokenizer.ggml.pre = "jina-v2-en"`
has **no effect**: llama.cpp's pre-type switch does not list that name
(`src/llama-vocab.cpp:2530-2560`), and a WordPiece vocabulary does not use a
pre-tokenizer at all. llama.cpp sets `add_sep = true` for every `bert` vocabulary,
but that flag is only read by the rerank pair path, so a single text is
`[CLS] … [SEP]` (verified: `"hello world"` → `101, 7592, 2088, 102`).

## Accuracy: against llama.cpp and against the full-precision reference

The S4 gate, on the 200 texts in `crates/peregrine-models/tests/fixtures/jina-bert/`
(16 to 8180 tokens; the long ones reach the 8192-token context, so ALiBi's bias spans
its whole range), 4 threads, cosine of L2-normalised vectors. The Q8_0 reference is
llama.cpp on the **F16 conversion**, as the design states.

| File | ids | vs llama.cpp min / mean | vs the F16 reference min / mean | llama.cpp's own Q8_0 vs the reference min / mean |
|---|---|---|---|---|
| base-en F16 | identical | **0.9999999 / 1.0000000** | 0.9999999 / 1.0000000 | — |
| base-en Q8_0 | identical | 0.9998662 / 0.9999650 | **0.9993389 / 0.9996382** | 0.9992368 / 0.9996404 |

F16 is at the 0.9999-class bound. For Q8_0 the gate's clause is on the **mean**: Peregrine
is 2.2e-6 below llama.cpp's mean distance from the F16 reference, inside the 1e-4 band.
The Q8_0 file is also 0.9998662/0.9999650 from llama.cpp's own Q8_0 vectors — the two
int8 engines agree more closely with each other than either does with its own F16 run,
which is the same finding the `modern-bert` gate reported. The test bounds are min
0.9992 / mean 0.9996 for the Q8_0 distance to the reference, and the F16-class bound
0.9999/0.9999 against llama.cpp's F16.

The 8180-token text is the one that exercises the slopes across the whole bias range;
its vector is in the same class as the rest (the minimum above is not that text).

## Speed

`/v1/embeddings` on `jina-embeddings-v2-base-en-Q8_0.gguf`, both engines on the same
**shared 8-vCPU VM** (timings here are rough: other jobs were running), 4 threads,
embedding cache off (`--opt EMBED_CACHE_MB=0`), five runs after two warm-ups, medians;
the cell is one request. A text of N tokens; `32×16` is one request of sixteen
32-token texts:

| Cell | llama.cpp texts/s [spread] | Peregrine texts/s [spread] | Peregrine ÷ llama.cpp |
|---|---|---|---|
| 16 tokens | 22.55 [22.50–23.29] | **33.75** [32.31–36.71] | 1.50 |
| 128 tokens | 4.201 [4.122–4.248] | **4.654** [4.314–4.789] | 1.11 |
| 512 tokens | 0.857 [0.825–0.870] | **0.888** [0.873–0.912] | **1.04** (the worst cell) |
| 32 tokens × 16 | 16.01 [15.50–16.58] | **260.7** [259.4–264.0] | 16.3 |

The 512-token cell is the worst ratio and still ahead. The 16-text batch is where the
engines differ most: Peregrine puts all sixteen texts of the request through one
weight pass (512 rows, no padding); llama.cpp's server runs a non-causal batch one
sequence after another, so its 512-token request costs what five 32-token requests
cost it. A 512-token text is dominated by its own `n²` attention.

Stage split is not repeated here: the family has the same projection-heavy shape as
`modern-bert`, and the ALiBi bias is one extra `f32` multiply-add per score (less than
the exponential beside it).

## What Peregrine does, and where it differs on purpose

- `JinaBert::forward_texts` runs one whole text per unit (never chunked, no KV cache),
  with all texts of a pass side by side in the shared GEMM. Q/K/V are three matrices in
  one fork-join; the two GEGLU matrices likewise. The graph is exactly the transcription
  above.
- **The attention scale is the block's**, not folded into Q: `attend_texts` multiplies
  each score by `kq_scale` and *then* adds ALiBi's bias, which is llama.cpp's order. The
  other families fold the scale into Q for historical reasons; this one does not need to.
- The ALiBi bias is `peregrine_kernels::bidir::alibi_row`: `x[j] += slope·(−|j − qpos|)`
  on the row's own band, scalar, with the slope list built once at load by
  `alibi_slopes` (`f32::powf`, llama.cpp's own formula). `attend_texts` takes
  `Option<&[f32]>`; every other family passes `None`, so their goldens did not move.
- **The GELU is the tanh form** (`vmath::geglu`), matching `ggml_gelu_f32`, but without
  llama.cpp's f16 lookup table: ADR 0003 makes the engine's own definition the
  reference, and the table is a rounding, not the function. This is the same deliberate
  difference the `bert` reranker gate measured (0.00056 mean on its scores).
- The LayerNorms are `vmath::layer_norm_rows` (weight and bias): the mean and variance
  are `f64` walks in index order, rounded once, where llama.cpp sums in `f32`.
- The family refuses a text longer than `context_length`, naming it, and the engine
  refuses chat and label requests on it (`is an embedding model …`). It refuses a file
  without `token_types.weight`, a fused GEGLU whose `ffn_up` is not `2·n_ff` wide, a
  separate gate whose `ffn_up` is not `n_ff` wide, and a file without `context_length`
  by name.
- The tokenizer is the existing WordPiece reader. Its ids matched llama.cpp on all 200
  gate texts, including the 8180-token one.
- No head: this family has no `cls` tensors in either reference file, so
  `head_width()` is 0 and `/v1/rerank`/`/v1/classify` refuse it as they refuse a
  headless `bert` file.

## jina-bert-v3, and what it would take

`jina-bert-v3` (`src/models/jina-bert-v3.cpp`) is **not** a small step from v2 and
was not built. Its graph is the `nomic-bert` shape: RoPE (the branch at
`src/models/bert.cpp:132-139` includes it), no ALiBi, plain `ffn_up` + `ffn_down` with
a GELU (`src/models/bert.cpp:185-192`, the `BERT`/`NOMIC_BERT_MOE`/`JINA_BERT_V3`
branch), no `attn_norm_2`, `attn_q/k/v` biases optional, and 24 layers (558M). It shares
the token-type row and the layer's output norm with v2 but not its attention or its
feed-forward. On top of that it is a **LoRA-adapter family**: the converter
(`conversion/bert.py:469-570`, `XLMRobertaModel`) writes the base as an XLM-R WordPiece
file plus one `lora-<task>` GGUF per task (rank/alpha in the adapter's metadata, the
prompt prefix in `adapter.lora.prompt_prefix`), which llama.cpp applies at load. So a
v3 step needs (1) the nomic-bert block with RoPE and a plain GELU, which is PR #231's
work rather than this file, (2) the XLM-R WordPiece position offset, which the `bert`
family already reads, and (3) a LoRA application path (load an adapter file, add its
`lora_a · lora_b` into the base matrices or apply it per pass). None of that is a
one-file addition, and no v3 file is in the S4.8c gate; it stays in the roadmap's S4.8
row.
