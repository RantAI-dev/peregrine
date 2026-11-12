# `modern-bert` (ModernBERT, gte-modernbert, Granite Embedding R2)

Reference note for roadmap **S4.7**, transcribed from **llama.cpp v0.9.0,
commit `cd959ba`**: `src/models/modern-bert.cpp` (the graph, the tensor list, the
window pattern), `src/llama-model.cpp` (the rope type, the SWA pattern,
`f_norm_eps`, the pooling keys), `src/llama-hparams.h/.cpp` (`is_masked_swa`,
`set_swa_pattern`), `src/llama-graph.cpp` (`build_norm`, `build_qkv`,
`build_attn_inp_no_cache`, `build_ffn`, the classifier head), `src/llama-arch.cpp`
(the names), `conversion/bert.py` and `conversion/base.py` (the converter) and
`ggml/src/ggml.c` / `ggml/src/ggml-cpu/ops.cpp` (the gated activation). Every
statement below was read from that source, not from the design note. Peregrine's
implementation is `crates/peregrine-models/src/modern_bert.rs`
(`ModernBert::load`, `forward_texts`), the pooling in
`crates/peregrine-core/src/embed.rs`, and the element-wise kernels in
`crates/peregrine-kernels/src/vmath.rs` (the bias-less LayerNorm and the erf GeGLU
are new here; both are scalar, with tests).

## What it is

`architecture = "modern-bert"` is the encoder ModernBERT-base and -large,
gte-modernbert-base, IBM Granite Embedding R2 and mmBERT ship as
(`src/llama-arch.cpp:24`, `conversion/bert.py:598-651`). It is a pre-norm
transformer with two LayerNorms per block instead of BERT's post-norm layout, RoPE
instead of a learned position table, alternating local (sliding-window) and global
attention, and a gated feed-forward. A file with a classifier head
(`ModernBertForSequenceClassification`) carries `cls`, `cls.norm`, `cls.output` and
`classifier.output_labels`; S4.7b serves it on `/v1/rerank` and `/v1/classify` (see
"The classifier head" below).

## Metadata

All model keys are under the `modern-bert.` prefix. The reference files are the
S4.7 accuracy gate's.

| Key | gte-modernbert-base F16 | Granite Small English R2 Q8_0 | Meaning |
|---|---|---|---|
| `block_count` | 22 | 12 | layers |
| `embedding_length` | 768 | 384 | `d_model` |
| `feed_forward_length` | 1152 | 1536 | `n_ff` (the `ffn_up` output is `2·n_ff`) |
| `attention.head_count` | 12 | 12 | heads; `head_dim = d_model/n_head` |
| `attention.head_count_kv` | absent | absent | KV heads = query heads |
| `context_length` | 8192 | 8192 | the longest text, in tokens |
| `attention.layer_norm_epsilon` | 1e-5 | 1e-5 | LayerNorm epsilon |
| `attention.causal` | false | false | no causal mask |
| `attention.sliding_window` | 128 | 128 | local window width |
| `attention.sliding_window_pattern` | 3 | 3 | every 3rd layer is global |
| `rope.freq_base` | 160000 | 80000 | global layers' RoPE base |
| `rope.freq_base_swa` | 10000 | 10000 | local layers' RoPE base |
| `hidden_activation` | absent (`gelu`) | absent (`gelu`) | the gated activation |
| `pooling_type` | 2 (cls) | 2 (cls) | this file's embedding pooling |
| `vocab_size` | 50368 | 50368 | the tokenizer's vocabulary |
| `tokenizer.ggml.model` | `gpt2` | `gpt2` | byte-level BPE |
| `tokenizer.ggml.pre` | `modern-bert` | `modern-bert` | the GPT-2 pre-tokenizer |
| `tokenizer.ggml.add_bos_token` / `add_eos_token` | true / true | true / true | `<bos> … <eos>` |
| `tokenizer.ggml.add_sep_token` | false | false | stored; never added |

`rope.dimension_count` is absent in both files; llama.cpp then uses the head width
(`hparams.n_rot_full = n_embd_head_k_full`, `src/llama-model.cpp:1610`), and
`attention.key_length` is absent too (`n_embd_head_k = n_embd/n_head`, line 1606).
`rope.freq_base` is the global base and `freq_base_swa` the local one; when only the
first is present, llama.cpp's local default is still **10000**, not the global value
(`llama_hparams`' two defaults, `get_rope_freq_base` at `src/llama-model.cpp:2853-2854`).

The reference files and their SHA-256:

| File | Repo | SHA-256 |
|---|---|---|
| `gte-modernbert-base-F16.gguf` | `keisuke-miyako/gte-modernbert-base-gguf` | `2cda419ce09e87b1dd5294177a1c7d563e9536700e33652a9bb9e9c503c4e437` |
| `gte-modernbert-base-Q8_0.gguf` | same | `d3c637bb219c4900ef56f4fe2fde2cf0f03bed0de5476e0141e049e68bc93301` |
| `granite-embedding-small-english-r2.f16.gguf` | `mradermacher/granite-embedding-small-english-r2-GGUF` | `6f5f7c70199658f54547b35174b50eef635ccc6d5b8e171dc635c201af4b2806` |
| `granite-embedding-small-english-r2.Q8_0.gguf` | same | `d4b41f5d7db712806722103a1c6aba2f0fe99f77740501d6f313c8240641f145` |

## Tensors

| Tensor | Shape | Notes |
|---|---|---|
| `token_embd.weight` | `[n_embd, n_vocab]` | |
| `token_embd_norm.weight` | `[n_embd]` | the embedding LayerNorm; **no bias** |
| `blk.N.attn_norm.weight` | `[n_embd]` | optional on layer 0 (identity there) |
| `blk.N.attn_qkv.weight` | `[n_embd, 3·n_embd]` | fused: q rows, then k, then v |
| `blk.N.attn_output.weight` | `[n_embd, n_embd]` | |
| `blk.N.ffn_norm.weight` | `[n_embd]` | |
| `blk.N.ffn_up.weight` | `[n_embd, 2·n_ff]` | the gate's first half and the up second half |
| `blk.N.ffn_down.weight` | `[n_ff, n_embd]` | |
| `output_norm.weight` | `[n_embd]` | the final LayerNorm; no bias |
| `cls.weight` | `[n_embd, n_embd]` | optional classifier dense; **no bias** |
| `cls.norm.weight` | `[n_embd]` | optional classifier LayerNorm; **no bias** |
| `cls.output.weight` / `.bias` | `[n_embd, n_cls_out]` / `[n_cls_out]` | optional classifier projection (the weight is written 1-d when `n_cls_out = 1`) |

`src/models/modern-bert.cpp:64-93` creates exactly this list, with the projections
**without biases** and only layer 0's `attn_norm` marked `TENSOR_NOT_REQUIRED`. The head's
`cls` and `cls.norm` are created weight-only (lines 92-93): for a sequence-classification
file llama.cpp creates neither `cls.bias` nor `cls.norm.bias` — `cls_b` is created only in
the decision-model branch (line 125), and the head norm is built with a NULL bias
(`src/llama-graph.cpp:4205-4207`).
Every norm has a weight and no bias: llama.cpp's `build_norm` multiplies the
centred, scaled row by the weight alone (`src/llama-graph.cpp:1808-1821`), unlike
the `bert` family's weight-plus-bias LayerNorm.

## The forward, in order

`llama_model_modern_bert::graph` (`src/models/modern-bert.cpp:134-245`), for one
sequence of `n` tokens:

1. `inpL = token_embd[token]` (`build_inp_embd`, line 148 — no token types and no
   position table; positions are RoPE inputs only).
2. `inpL = norm(inpL, token_embd_norm)` — LayerNorm with a weight only (line 152).
3. Per encoder layer `il` (pre-norm):
   1. `cur = inpL`; if `attn_norm[il]` exists, `cur = norm(inpL, attn_norm)` (lines
      166-174). Layer 0's tensor is absent, so its attention input is `inpL`.
   2. `Qcur, Kcur, Vcur = qkv(cur)` from the fused matrix (line 177;
      `build_qkv`, `src/llama-graph.cpp:1857-1889`: the q view is rows `0..n_embd`,
      k the next `n_embd`, v the last `n_embd`; no bias in the encoder).
   3. RoPE on q and k only (NEOX pairing: `llama_model_rope_type` lists
      `MODERN_BERT` in the offset-by-`n_rot/2` group, `src/llama-model.cpp:3693`;
      the base is `get_rope_freq_base(cparams, il)`, the local key on a local layer,
      line 163).
   4. `cur = attn(Qcur, Kcur, Vcur)` with `kq_scale = 1/√n_embd_head` (lines
      197-200; `build_attn` with no cache).
   5. `ffn_inp = cur + inpL` — the residual is the **unnormalised** layer input
      (line 208).
   6. `cur = norm(ffn_inp, ffn_norm)` (lines 212-214).
   7. `cur = ffn(cur)` (lines 217-223): `ffn_up` is `2·n_ff` wide, the gated
      activation, then `ffn_down`, `LLM_FFN_SEQ`, no biases.
   8. `inpL = cur + ffn_inp` (line 226).
4. `result_embd = norm(inpL, output_norm)` (lines 232-237) — the rows pooling reads.

The feed-forward's activation is `LLM_FFN_GEGLU_ERF` unless the file's
`hidden_activation` maps to another gated op (`src/models/modern-bert.cpp:18-24`);
`llm_ffn_op_type_from_string` maps `gelu` and `gelu_python` to `GEGLU_ERF`, the
`gelu_new`-style names to the tanh `GEGLU`, and `silu` / `swish` / `swiglu` to
`SWIGLU` (`src/llama-model.cpp:1192-1206`). Granite Embedding R2's config says
`hidden_activation = "gelu"`, so both reference files take the erf path (Granite's
GGUF predates the key, and the default is the same).

`ggml_geglu_erf` acts on the **first** half of the `2·n_ff` row and multiplies by
the second: `ggml_glu_impl(..., GGML_GLU_OP_GEGLU_ERF, false)` (`ggml/src/ggml.c:3104-3108`),
and with `swapped = false` the code reads `src0 += 0` and `src1 += n` — gelu of the
first half times the second (`ggml/src/ggml-cpu/ops.cpp:3716-3734`), with
`y = 0.5·x·(1 + erff(x/√2))·g` (`ggml/src/ggml-cpu/vec.h:1584-1588`). `ggml_swiglu`
uses the same swapped=false layout (`ggml/src/ggml.c:3083-3087`).

### Local and global attention

`src/models/modern-bert.cpp:7-14` reads `attention.sliding_window`; when it is
positive, `swa_type` is `LLAMA_SWA_TYPE_SYMMETRIC` and
`load_swa_pattern(ml, 3, dense_first = true)` runs. That helper
(`src/llama-model.cpp:4053-4059`, `src/llama-hparams.cpp:8-22`) sets
`is_swa[il] = n_pattern == 0 || il % n_pattern != 0`: **layer 0 is global**, every
third layer after it global, the two between local. A file's
`attention.sliding_window_pattern` scalar overrides the 3 (array form is read by
llama.cpp's `get_arr`; the converter writes the scalar).

The mask is `llama_hparams::is_masked_swa`'s symmetric branch
(`src/llama-hparams.h:540-575`): a local layer masks keys with
`|p1 − p0| > n_swa / 2` (integer division). With `n_swa = 128` a token sees 64
either side. Global layers mask nothing; `attention.causal` is false, so no layer
masks the future. The no-cache attention input builds the `n × n` mask from exactly
these three rules (`src/llama-graph.cpp:576-616`, `build_attn_inp_no_cache` at
3138).

### Pooling and the classifier head

`<arch>.pooling_type` is read generically (`src/llama-model.cpp:1473`): `1` mean,
`2` cls, `3` last, `0` none. Both reference files say `2` (cls). The embeddings
endpoint L2-normalises after pooling (`common_embd_normalize(..., 2)`).

A classifier file has `classifier.output_labels` (the labels), `cls` (a dense
`n_embd × n_embd`), `cls.norm`, and `cls.output` (`n_embd × n_labels`). The head is
built in `build_pooling`'s `LLAMA_POOLING_TYPE_RANK` branch, which the server selects
with `--reranking` (`src/llama-graph.cpp:4185-4220`, reached from
`src/llama-model.cpp:3453`). S4.7b's transcription is below; a classifier file that
declares a `pooling_type` still serves embeddings from the encoder, exactly as
llama.cpp serves `/v1/embeddings` from it.

### The classifier head (S4.7b)

This is `build_pooling`'s `LLAMA_POOLING_TYPE_RANK` branch as it runs for this family
(`src/llama-graph.cpp:4185-4220`), every step read from that source:

```text
h = mean(x)                      // the text's output_norm rows, or row 0
h = cls.weight · h               // no bias
h = act_cls(h)                   // classifier.activation, GELU_ERF when absent
h = layer_norm(h, cls.norm.w)    // weight only, no bias; skipped when absent
y = cls.output.weight · h + cls.output.bias
score = y[0]                     // llama.cpp's /v1/rerank reads embd[0]
```

1. **The matrix the branch runs under.** The switch is on the *request's* pooling type,
   `cparams.pooling_type == LLAMA_POOLING_TYPE_RANK`, which `llama-server --reranking`
   sets. `<arch>.pooling_type` — the embeddings key read at
   `src/llama-model.cpp:1473` — has no part in the head. So the head is a property of
   the tensors, exactly as it is for the `bert` family (design §8).
2. **The pooled row.** `hparams.pooling_type_cls` is `classifier.pooling_type`; when the
   file has no such key it is set to `MEAN` for this family
   (`src/models/modern-bert.cpp:26-28`). `build_pooling` takes the mean when the value is
   `MEAN` and **row 0 otherwise** — there is no other branch
   (`src/llama-graph.cpp:4187-4194`). The mean is over *every* token of the sequence,
   specials included: `build_inp_mean` fills a one-hot row per sequence with `1/n`
   (`src/llama-graph.cpp:389-429`) and the graph computes
   `transpose(x) · mean` (`4187-4191`).
3. **The dense layer and its bias.** `if (cls) { cur = ggml_mul_mat(cls, cur); if (cls_b)
   add }` (`4199-4202`). For a sequence-classification `modern-bert` file `cls_b` is
   **null**: `llama_model_modern_bert::load_arch_tensors` creates `cls` weight-only
   (`src/models/modern-bert.cpp:92`), and creates `cls_b` only in the decision-model
   branch (line 125). A file that carries a `cls.bias` tensor is therefore read without
   it, and so is a `cls.norm.bias`: the head norm is built with a NULL bias
   (`build_norm(cur, cls_norm, NULL, LLM_NORM, -1)`, `4205-4207`). The converters do not
   write either for this family unless `classifier_bias` / `norm_bias` are on, and
   Transformers' `ModernBertConfig` defaults both to false.
4. **The activation.** `ggml_unary(cur, hparams.act_cls)` (`4204`). `act_cls` comes from
   `classifier.activation` through `LLM_CLS_ACT_TYPES_FROM_STRING` — `gelu` is
   `GGML_UNARY_OP_GELU_ERF`, `silu`, `tanh` (`src/llama-model.cpp:1209-1213`) — and when
   the key is absent this family's loader sets `GGML_UNARY_OP_GELU_ERF`
   (`src/models/modern-bert.cpp:31-35`). Note the difference from `bert`: that family's
   default is TANH.
5. **The head norm.** Inside the same `if (cls)`, after the activation:
   `build_norm(cur, cls_norm, NULL, LLM_NORM, -1)` (`4205-4207`) — the same weight-less
   LayerNorm the encoder's blocks use (above).
6. **The projection.** `if (cls_out) { cur = ggml_mul_mat(cls_out, cur); if (cls_out_b)
   add }` (`4215-4220`). `cls_out` is `{n_embd, n_cls_out}` (`src/models/modern-bert.cpp:90`);
   `n_cls_out` is the number of `classifier.output_labels`, or 1 when the file has none
   (`src/llama-model.cpp:1635-1638`, default at `src/llama-hparams.h:271`). The
   converters write the weight 1-d when that number is 1.
7. **No softmax.** The only softmax in the branch is for Qwen 3's generative reranker
   (`4222-4225`); a `modern-bert` score is the raw logit. `/v1/rerank`'s
   `relevance_score` is that number (`send_rerank` reads `embd[0]`,
   `tools/server/server-context.cpp:2599-2627`), and `/v1/classify`'s probabilities are
   the caller's softmax over the outputs.

The order is Transformers' `ModernBertPredictionHead`: dense → activation → norm →
decoder, which is also what the branch computes.

A **decision model** (`ModernBertDecisionModel`: Laya, Julia-1) appends scorer blocks to
`block_count` (`modern-bert.decision.block_count`) and then applies `cls`/`cls.norm` with
biases (`src/models/modern-bert.cpp:38-52`, `125-131`, `build_decision_head` at
`248-310`). That is S4.12; this step refuses such a file by name at load rather than
reading its head blocks as encoder layers.

The **pair** llama.cpp scores is built by `format_prompt_rerank` when the file has no
`rerank` chat template (`tools/server/server-common.cpp:2019-2046`): both halves are
tokenized with `add_special = false` and `parse_special = false` (`2021-2022`), then
`[bos] q [eos] [sep] d [eos]`, each of the three specials present only when the file's
`add_bos` / `add_eos` / `add_sep` says so, and `eos` falling back to `sep` when the file
has no EOS id (`2024-2041`). For the gte reranker (`add_sep = false`, BOS `[CLS]`, EOS
`[SEP]`) that is `[CLS] q [SEP] d [SEP]`. `parse_special = false` matters for this
vocabulary: `[CLS]` or `<|endoftext|>` inside a query is ordinary text, while the
`[unused*]` and `|||EMAIL_ADDRESS|||` user-defined tokens are still matched, as
`tokenizer_st_partition` does (`src/llama-vocab.cpp:3645-3658`).

### The tokenizer

`tokenizer.ggml.model = "gpt2"` and `tokenizer.ggml.pre = "modern-bert"`;
llama.cpp maps that pre-tokenizer to `LLAMA_VOCAB_PRE_TYPE_GPT2`
(`src/llama-vocab.cpp:2538`) — the GPT-2 alternation with digits in one run
(`src/llama-vocab.cpp:380-390`):

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
```

The vocabulary is byte-level BPE with `<bos>`/`<eos>` specials (gte's conversion spells
them `[CLS]`/`[SEP]`, ids 50281/50282); the converter writes `add_bos_token`,
`add_eos_token` and `add_sep_token` true (`conversion/bert.py:604-607`), and llama.cpp's
BPE tokenizer adds the BOS and the EOS around one text (`src/llama-vocab.cpp:3891-3900`).
`add_sep` is stored and used only by the rerank pair path above, so a text is
`[CLS] … [SEP]` and a pair is `[CLS] q [SEP] [SEP] d [SEP]` for a file that sets the key
(gte-reranker's conversion sets it false: `[CLS] q [SEP] d [SEP]`). Peregrine's
`PreType::Gpt2` is that pattern, and its ids match llama.cpp on all 200 gate texts for
both files and on all 100 rerank pairs.

mmBERT's vocabulary says `tokenizer.ggml.pre = "mmbert"`
(`src/llama-vocab.cpp:2550`), a metaspace pre-tokenizer whose words are split in
`tokenize()` and whose BPE is not byte-encoded (`src/llama-vocab.cpp:563-569`,
`686-707`). Peregrine refuses that name by name; the other models of the family
are byte-level GPT-2 BPE.

## Accuracy: against llama.cpp and against the full-precision reference

The S4 gate, on the 200 texts in `crates/peregrine-models/tests/fixtures/modern-bert/`
(16 to 6976 tokens; five past the 128-token window, so both mask rules run),
4 threads, cosine of L2-normalised vectors. The Q8_0 reference is llama.cpp on the
**F16 conversion**, as the design states.

| File | ids | vs llama.cpp min / mean | vs the F16 reference min / mean | llama.cpp's own Q8_0 vs the reference min / mean |
|---|---|---|---|---|
| gte-modernbert-base F16 | identical | **0.9999936 / 0.9999997** | 0.9999936 / 0.9999997 | — |
| gte-modernbert-base Q8_0 | identical | 0.9969122 / 0.9998028 | **0.9963540 / 0.9996125** | 0.9981270 / 0.9996284 |
| Granite R2 F16 | identical | **0.9999997 / 0.9999999** | 0.9999997 / 0.9999999 | — |
| Granite R2 Q8_0 | identical | 0.9999290 / 0.9999600 | **0.9998753 / 0.9999246** | 0.9998838 / 0.9999252 |

F16 is at the 0.9999-class bound (min 0.9999936). For Q8_0 the gate's clause is on
the **mean**: Peregrine is 1.6e-5 below llama.cpp's mean distance on gte and 6.5e-7
below it on Granite, both inside the 1e-4 band. The per-text minimum is not stable
either way: one 69-token gte text is 1.8e-3 further out than llama.cpp's (0.99635
against 0.99969), while another text sits closer. The cause is the int8 activation
quantiser: a 1-ulp difference between the two engines' projection kernels flips a
code now and then, and the next layer reads it. That is also why the per-text
minimum is printed in the ignored test and only the mean is asserted against
llama.cpp's. The measured test bounds are min 0.996 / mean 0.9996 for gte Q8_0 and
min 0.9998 / mean 0.9999 for Granite Q8_0.

### The classifier head (S4.7b)

`gte-reranker-modernbert-base` on the 100 pairs in
`crates/peregrine-models/tests/fixtures/modern-bert/rerank/` (36–445 tokens per pair,
51 of them past the 128-token window; ten documents per query), 4 threads, the raw
score's absolute difference. The F16 conversion is the full-precision reference, as
the gate states.

| Our file | Compared with | ids | mean abs | max abs | order |
|---|---|---|---|---|---|
| F16 | llama.cpp F16 | identical | **0.00067** | **0.00273** | 10/10 queries identical |
| Q8_0 | llama.cpp Q8_0 | identical | 0.01141 | 0.07430 | 6/10; the four swaps are documents ≤ 2.24e-2 apart in llama.cpp's own scores |
| Q8_0 | llama.cpp F16 (the reference) | identical | 0.02173 | 0.07970 | — |
| llama.cpp Q8_0 | llama.cpp F16 | — | 0.02310 | 0.09370 | 5/10 |

So the Q8_0 file's own head row is closer to llama.cpp's **F16** run than llama.cpp's
Q8_0 run is, and llama.cpp's two precisions disagree on order for 5 of the 10 queries
while Peregrine's Q8_0 and llama.cpp's Q8_0 disagree on 4. The swaps are the same
phenomenon as the embeddings' per-text minimum: the int8 activation code flips and two
documents whose true scores are a couple of hundredths apart trade places. The ignored
test therefore allows a 3e-2 near-tie on the Q8_0 comparisons and 1e-2 at F16 (where
the measured mean is 7e-4), and requires the ids and the F16 order exactly.

## Speed

Rerank, `/v1/rerank`, one request with eight pairs, `gte-reranker-modernbert-base-Q8_0.gguf`,
both engines at once on the same **shared 8-vCPU VM** (timings here are rough: other jobs
were running), 4 threads each, five alternating runs, medians; pairs/s:

| tokens per pair | llama.cpp | Peregrine | Peregrine ÷ llama.cpp |
|---|---|---|---|
| 128 (1016-token request) | 3.82 [3.48–3.98] | **4.35** [4.14–4.39] | 1.14 |
| 512 (4056-token request) | 0.73 [0.73–0.74] | **0.91** [0.89–0.92] | 1.24 |

The head itself is two row projections per pair and does not move these numbers; what
does is the encoder pass, where Peregrine is ahead as it is for embeddings.

`/v1/embeddings` on `gte-modernbert-base-F16.gguf` on H0 (8 vCPU Xeon E5-2680 v4,
AVX2+FMA, **shared host**: other jobs were running), 4 threads, one engine at a
time, alternating three runs per engine, medians. A text of N tokens repeated
`batch` times in one request; texts/s:

| tokens | 1 (Peregrine / llama.cpp) | 32 (Peregrine / llama.cpp) |
|---|---|---|
| 16 | 30.8 / 21.7 | 37.3 / 29.4 |
| 128 | 4.59 / 3.55 | 5.20 / 3.76 |
| 512 | 1.29 / 0.892 | 1.17 / 0.812 |
| 2000 | 0.255 / 0.180 | not measured (64 Ki-token request) |

Peregrine is ahead on every row measured. A 32-text request of 2000-token texts is
64 Ki tokens and takes minutes per run; it was left out rather than timed on the
shared host.

Stage split from the forward's own `StepStats` (best of three warm passes, one
text, 4 threads; the matmul bucket includes the caller's wait for its workers, so
the buckets are attribution, not a partition):

| tokens | wall ms | matmul | attention | other (norms, RoPE, residuals, quantisation) |
|---|---|---|---|---|
| 512 | 814 | 564 (69 %) | 88 (11 %) | 161 (20 %) |
| 2000 | 3814 | 2148 (56 %) | 1020 (27 %) | 646 (17 %) |

The matrices dominate, as the design's §4.3 says; attention grows to a quarter of
the pass at 2000 tokens, where the local layers still do the banded work. The
"other" share is the bias-less LayerNorm pairs, the gated activation and the
residual walks; S4.9h's vector kernels cover `gelu_erf_rows` and the bias LayerNorm
but not this family's no-bias form or its GeGLU, which are scalar today.

## What Peregrine does, and where it differs on purpose

- `ModernBert::forward_texts` runs one whole text per unit (never chunked, no KV
  cache), with all texts of a pass side by side in the shared GEMM. The fused
  `attn_qkv` is one projection, its rows are split into q/k/v, RoPE is applied with
  the layer's table, and attention is the shared `blocks::bidir::attend_texts`
  with `Some(window/2)` on a local layer and `None` on a global one.
- Q is scaled by `1/√head_dim` after RoPE and the attention block is called with
  scale 1; llama.cpp passes the scale to `build_attn`. It is the same product with
  one extra rounding, and the reference forward in the tests does the same.
- The bias-less LayerNorm is `vmath::layer_norm_rows_nobias`: the mean and the
  variance are two `f64` walks in index order (llama.cpp's `ggml_norm` sums in
  `f32` through `ggml_vec_sum_f32` and `ggml_vec_cvar_f32`), rounded once, and the
  final scale-and-shift is `(x − m)·inv·w` with no `+ 0` (the `+ 0` would turn a
  `−0.0` product into `+0.0`). The exact erf GELU is `vmath::geglu_erf`, and a
  tanh-GELU or SiLU file uses the existing `vmath::geglu` / `vmath::swiglu`. These
  two new kernels are **scalar** (S4.7 adds no SIMD); a vector body may only be
  added if it is `to_bits` equal to the scalar twin.
- The family refuses a text longer than `context_length`, naming it, and the
  engine refuses chat and label requests on it (`is an embedding model …`).
- **The head** (S4.7b) is `ModernBertHead` in `modern_bert.rs`. The loader reads
  `cls.weight`, `cls.norm.weight`, `cls.output.weight`, `cls.output.bias` and
  `classifier.output_labels`, with `classifier.activation` and
  `classifier.pooling_type`; it refuses by name a label count that disagrees with
  `cls.output`, an activation llama.cpp does not read, labels with no projection,
  and a `cls.norm` with nothing to apply it to, and it refuses a decision model
  (`modern-bert.decision.block_count > 0`) by name. `cls.bias` and `cls.norm.bias`
  are read without, exactly as llama.cpp reads them. A `Want::Head` text leaves its
  row in `State::head_out` at `row0 * head_width()`; the read-out start is the mean
  (the family's default) or row 0, and the two projections are one row each, so a
  row is bit-identical alone, in any batch, at any pass size and thread count, and a
  head pass allocates nothing. `head_width()` is `cls.output`'s rows (or `d_model`
  for a one-layer head), and `head_labels()` is the file's list, so `/v1/rerank`,
  `/v1/classify`, `/v1/models` and `peregrine rerank` serve the family with no
  endpoint change. The read-out reuses the pooled row's finding: it copies the
  `output_norm` rows into `st.x` (an exact copy) so the head's and the pooler's
  scratch rows do not fight over `st.xb`.
- **The rerank pair** is `format_prompt_rerank`'s: `Tokenizer::encode_pair` now
  tokenizes a byte-level BPE pair with `parse_special = false` and assembles
  `[bos] q [eos] [sep] d [eos]` from the file's flags. It matched llama.cpp on all
  100 fixture pairs.
- Tests: a synthetic `modern-bert` file (`tests/support/models.rs`) with four
  layers, a window of 8, both rope bases different, F32 and Q8_0 variants, the three
  activations and two headed variants (a rank head and a labelled one); the
  independent scalar reference (`tests/support/modern_bert_ref.rs`, including
  `reference_head`); the batch/thread invariance matrix; the no-allocation test;
  server and CLI tests; one golden per pinned x86 tier for the encoder
  (`tests/golden/synthetic/<tier>/modern-bert-Q8_0.golden`, with aarch64 ones recorded
  on the ARM host in S4.9n) and one for the heads (`modern-bert-heads-Q8_0.golden`:
  scalar, AVX2 and AVX-512 under SDE; no aarch64 one yet).
- The 200-text fixtures and the ignored real-file test live in
  `tests/fixtures/modern-bert/` and `tests/embeddings.rs`; the 100 rerank pairs live
  in `tests/fixtures/modern-bert/rerank/` and run from `tests/rerank_real.rs`.
