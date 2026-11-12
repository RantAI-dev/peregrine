# `nomic-bert-moe` (nomic-embed-text-v2-moe)

Reference note for roadmap **S4.8b**, transcribed from **llama.cpp v0.9.0, commit
`cd959ba`**: `src/models/bert.cpp` (the shared encoder graph and its
`LLM_ARCH_NOMIC_BERT_MOE` branches), `src/models/nomic-bert-moe.cpp` (the hparams and
the tensor list), `src/llama-model.cpp` (the arch dispatch, the rope type, the
pooling key), `src/llama-graph.cpp` (`build_inp_embd`, `build_qkv`,
`build_attn_inp_no_cache`, `build_attn`, `build_ffn`, `build_moe_ffn`,
`build_pooling`), `src/llama-vocab.cpp` (the SentencePiece-unigram tokenizer) and
`conversion/bert.py` (the `NomicBertModel` converter class). Every statement below
was read from that source, not from the design note. Peregrine's implementation is
`crates/peregrine-models/src/nomic_bert.rs` (`NomicBert::load`, `forward_texts`, the
`NomicFfn::Moe` branch), the routing is
`crates/peregrine-core/src/blocks/moe.rs`, the pooling is
`crates/peregrine-core/src/embed.rs`, and the scalar reference forward is
`crates/peregrine-models/tests/support/nomic_bert_moe_ref.rs`.

## What it is

`general.architecture = "nomic-bert-moe"` is Nomic AI's multilingual embedding model
(`llama-arch.cpp:26`). `nomic-embed-text-v2-moe` is the 475M-parameter member
(`src/models/nomic-bert-moe.cpp:7-11`): 12 post-norm encoder blocks, 768 wide, 12
heads of 64, **RoPE instead of a position table**, and a feed-forward that is a
**mixture of 8 experts on the odd layers** (2 experts per row) and a plain **GELU
MLP** on the even ones. It is the same `llama_model_bert::graph` as `nomic-bert`,
branched twice inside it (`src/models/bert.cpp:132`, `:171-192`).

The differences from [`nomic-bert`](nomic-bert.md) are the whole of this family:

| | `nomic-bert` (v1.5) | `nomic-bert-moe` (v2-moe) |
|---|---|---|
| Feed-forward, every layer | SwiGLU: `silu(ffn_gate) ⊙ ffn_up` → `ffn_down` | **alternating**: odd layers MoE, even layers `gelu(ffn_up) → ffn_down` |
| MoE layers | none | `ffn_gate_inp` router, `ffn_up_exps` / `ffn_down_exps`, 8 experts, top 2, **GELU**, raw softmax weights, summed in selection order |
| Biases | none in v1.5's files | `attn_qkv.bias`, `attn_output.bias`, `ffn_up.bias`, `ffn_down.bias` all present |
| RoPE base | 1000 | 10000 |
| Epsilon | 1e-12 | 1e-5 |
| Context | 2048 | 512 |
| Tokenizer | WordPiece (`bert`, 30 522) | **SentencePiece unigram** (`t5`, 250 048), the XLM-R vocabulary |

Everything else is the same: token embeddings plus token-type row 0, a `token_embd_norm`
LayerNorm with weight and bias, the fused `attn_qkv`, NeoX RoPE, full bidirectional
attention at `1/√head_dim`, `attn_output` and a residual, `attn_output_norm`,
`layer_output_norm` as the layer's output, and **no final norm**.

## Metadata

The reference files are Nomic AI's own conversions,
`nomic-ai/nomic-embed-text-v2-moe-GGUF`. All model keys are under the
`nomic-bert-moe.` prefix.

| Key | Value | Meaning |
|---|---|---|
| `block_count` | 12 | layers; the odd ones (1, 3, …, 11) are the MoE layers |
| `context_length` | 512 | the longest text, in tokens |
| `embedding_length` | 768 | `d_model` |
| `feed_forward_length` | 3072 | `n_ff`: the plain layer's width **and** each expert's width |
| `attention.head_count` | 12 | heads; `head_dim = 768/12 = 64` |
| `attention.head_count_kv` | absent | keys and values are as wide as queries (`n_embd_gqa = n_embd`, `src/models/nomic-bert-moe.cpp:29`) |
| `attention.layer_norm_epsilon` | 1e-5 | LayerNorm epsilon |
| `attention.causal` | false | no causal mask (`BertModel.set_gguf_parameters`) |
| `pooling_type` | 1 | mean |
| `rope.freq_base` | 10000 | `rotary_emb_base` |
| `rope.dimension_count` | absent | llama.cpp rotates the whole head (`n_rot_full = n_embd_head_k_full`, `src/llama-model.cpp:1601-1610`) |
| `moe_every_n_layers` | 2 | the MoE condition is `il % moe_every_n_layers == 1` (`src/models/bert.cpp:171`); 0 would mean no MoE layer at all |
| `expert_count` | 8 | the router's width |
| `expert_used_count` | 2 | the experts one row runs |
| `expert_weights_scale` | absent | `hparams.expert_weights_scale` stays 0, and `build_moe_ffn` does not apply a scale of 0 or 1 (`src/llama-graph.cpp:2580-2584`) |
| `vocab_size` | absent | the vocabulary size comes from the tokenizer: 250 048 |
| `tokenizer.ggml.model` | `t5` | SentencePiece unigram (`src/llama-vocab.cpp:2341-2342`) |
| `tokenizer.ggml.pre` | `default` | the DEFAULT pre-tokenizer; unigram needs none beyond its normalization |
| `tokenizer.ggml.add_space_prefix` / `remove_extra_whitespaces` | true / true | the metaspace prefix and whitespace cleanup the XLM-R vocabulary was trained with |
| `tokenizer.ggml.precompiled_charsmap` | 237 539 bytes | the XLM-R normalization table |
| `tokenizer.ggml.token_type_count` | 1 | llama.cpp's loader throws when it is 0 (`src/models/nomic-bert-moe.cpp:17-19`) |
| `tokenizer.ggml.bos_token_id` / `.eos_token_id` | 0 / 2 | `<s>` / `</s>`; both are added (`add_bos_token` and `add_eos_token` true) |

The reference files and their SHA-256:

| File | Repo | Bytes | SHA-256 |
|---|---|---|---|
| `nomic-embed-text-v2-moe.f16.gguf` | `nomic-ai/nomic-embed-text-v2-moe-GGUF` | 957 680 480 | `a5db3381f2e514d3490a3a31fe70eb1a65e95016c85c6c2c23223b810806594f` |
| `nomic-embed-text-v2-moe.Q8_0.gguf` | same | 512 225 120 | `06e7a7e594a26985523c18383aba4aad39fe6e14f08ffc6ab5b554e1ccdc3cff` |

## Tensors

The F16 file holds 142 tensors (the Q8_0 file the same list with Q8_0 matrices):

| Tensor | Shape (F16 file) | Notes |
|---|---|---|
| `token_embd.weight` | `[768, 250048]` | F16 |
| `token_types.weight` | `[768]` | one row, written 1-d; added to every token (`src/models/bert.cpp:91-94`) |
| `token_embd_norm.bias` / `.weight` | `[768]` | the embedding LayerNorm, weight **and** bias |
| `blk.N.attn_qkv.weight` | `[768, 2304]` | fused: q rows 0..768, k 768..1536, v 1536..2304 |
| `blk.N.attn_qkv.bias` | `[2304]` | added before RoPE |
| `blk.N.attn_output.weight` | `[768, 768]` | |
| `blk.N.attn_output.bias` | `[768]` | |
| `blk.N.attn_output_norm.bias` / `.weight` | `[768]` | the post-attention LayerNorm |
| `blk.N.layer_output_norm.bias` / `.weight` | `[768]` | the post-FFN LayerNorm; the last one **is** the encoder's output |
| `blk.N.ffn_up.weight` (even N) | `[768, 3072]` | the plain GELU layer |
| `blk.N.ffn_up.bias` (even N) | `[3072]` | added before the GELU |
| `blk.N.ffn_down.weight` (even N) | `[3072, 768]` | |
| `blk.N.ffn_down.bias` (even N) | `[768]` | |
| `blk.N.ffn_gate_inp.weight` (odd N) | `[768, 8]` | the router, **F32** even in the Q8_0 file |
| `blk.N.ffn_up_exps.weight` (odd N) | `[768, 3072, 8]` | expert-major: expert `e` is the contiguous run of 3072 rows |
| `blk.N.ffn_down_exps.weight` (odd N) | `[3072, 768, 8]` | the same layout |

The tensor list is `src/models/nomic-bert-moe.cpp:14-51`: `create_tensor_qkv` for the
fused projection (line 29), `ffn_up_exps` / `ffn_down_exps` / `ffn_gate_inp` for the
MoE layers (37-40), `ffn_up` / `ffn_down` and their optional biases for the plain
layers (42-45), and `layer_out_norm` for every layer (48-49). There are **no expert
biases** and no final `output_norm`.

The conversion's FFN naming is the same as v1.5's: HF's `fc11`/`fc12` map to
`FFN_UP`/`FFN_GATE`, but this family's plain branch does not read a gate at all —
`LLM_ARCH_NOMIC_BERT_MOE` takes the `LLM_FFN_GELU` + `LLM_FFN_SEQ` branch with
`gate == NULL` (`src/models/bert.cpp:185-192`).

## The forward, in order

`llama_model_bert::graph::graph` (`src/models/bert.cpp:68-225`), for one sequence of
`n` tokens:

1. `inpL = token_embd[token]` (`build_inp_embd`, line 88).
2. If the file has `token_types.weight`, `inpL += token_types[0]` — token types are
   hardcoded to zero, "Sentence A" (lines 91-94). `inp_pos` exists only for RoPE
   (lines 78-80): there is no position table.
3. `inpL = LayerNorm(inpL, token_embd_norm.weight, token_embd_norm.bias)` (line 101).
4. Per layer `il`, post-norm:
   1. `Qcur, Kcur, Vcur = qkv(inpL)` from the fused `attn_qkv` plus its bias (lines
      108-113; `build_qkv`, `src/llama-graph.cpp:1855-1894`), split q / k / v.
   2. `Qcur, Kcur = RoPE(Qcur, Kcur, inp_pos)` — the `LLM_ARCH_NOMIC_BERT_MOE`
      branch (lines 132-139). No q/k norm tensors exist (115-128 do nothing).
   3. `cur = attn(Q, K, V, wo, wo_b)` with `kq_scale = 1/√n_embd_head`, no cache:
      every position sees every position (lines 145-147, `build_attn_inp_no_cache`).
   4. `cur = cur + inpL`; `cur = LayerNorm(cur, attn_output_norm)` (lines 156-160).
      `attn_norm_2` does not exist for this arch (162-165).
   5. `ffn_inp = cur` (line 167).
   6. The feed-forward, one of two branches:
      * **MoE layer** (`il % moe_every_n_layers == 1`, lines 171-184):
        `cur = build_moe_ffn(cur, ffn_gate_inp, ffn_up_exps, NULL, ffn_down_exps,
        NULL, n_expert, n_expert_used, LLM_FFN_GELU, false, expert_weights_scale,
        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il)`;
      * **plain layer** (lines 185-192): `build_ffn` with `ffn_up` (+ `ffn_up.bias`),
        no gate, `LLM_FFN_GELU`, `LLM_FFN_SEQ`, `ffn_down` (+ `ffn_down.bias`) —
        `ffn_down · gelu(ffn_up · x + b_up) + b_down` (`src/llama-graph.cpp:1998-2004,
        2070-2077, 2155-2161`).
   7. `cur = cur + ffn_inp` (line 212); `cur = LayerNorm(cur, layer_output_norm)`
      (line 215).
   8. `inpL = cur`.
5. `res->t_embd = inpL` (lines 221-224): there is **no final norm**.

### The mixture of experts, step by step

`build_moe_ffn` (`src/llama-graph.cpp:2227-2633`, the full overload):

1. `logits = ffn_gate_inp · x`, `[n_expert, n_tokens]` (line 2261). There is no
   router bias (`gate_inp_b == nullptr`).
2. `probs = softmax(logits)` over **all** 8 experts, per row (line 2279; the CPU
   softmax is `ggml_vec_soft_max_f32`, `ggml/src/ggml-cpu/vec.h:1000-1019`):
   maximum subtracted, `exp` per element, summed in double, one reciprocal, scaled.
3. `selected = argsort_top_k(probs, 2)` (line 2345): `ggml_argsort` sorts
   **descending** (`src/ggml.c:5501-5513`); the first `n_expert_used` entries are
   the choice. ggml compares values only with `std::sort`
   (`ggml/src/ggml-cpu/ops.cpp:8698-8746`), so the order among exactly equal
   probabilities is implementation-defined; Peregrine fixes probability descending,
   then expert index ascending (the rule `blocks/moe.rs` already uses, and the only
   deterministic choice that does not look at the batch).
4. `weights = probs[selected]`, `[1, n_expert_used, n_tokens]` (line 2359). `norm_w`
   is false, so **the weights are not renormalised**, and `w_scale` is 0, so they
   are not scaled.
5. `up = ffn_up_exps[selected] · x` (lines 2436-2444), then **`gelu(up)`** — no gate
   bank (`LLM_FFN_GELU`, `has_gate == false`, lines 2511-2517).
6. `down = ffn_down_exps[selected] · gelu(up)` (line 2560), `[n_embd,
   n_expert_used, n_tokens]`.
7. `experts = down · weights` (line 2586), then the two slots are added **in the
   router's order**: `moe_out = slot[0]; moe_out += slot[1]` (lines 2600-2611). With
   `n_expert_used == 1` the result is made contiguous.
8. The graph adds `ffn_inp` afterwards (line 212).

### RoPE

`llama_model_rope_type` lists `LLM_ARCH_NOMIC_BERT_MOE` in the
offset-by-`n_rot/2` group (`src/llama-model.cpp:3688-3700`), i.e.
`LLAMA_ROPE_TYPE_NEOX`: element `i` of a rotary pair is paired with `i + n_rot/2`.
`n_rot` is the head width, 64, and the base is `rope.freq_base`, 10000; the file has
no scaling.

### Pooling

`<arch>.pooling_type` is read generically (`src/llama-model.cpp:1473`) and
`build_pooling`'s `LLAMA_POOLING_TYPE_MEAN` branch averages the final rows
(`src/llama-graph.cpp:4174-4180`). The file says `1` (mean). The server then
L2-normalises (`common_embd_normalize(..., 2)`).

### The tokenizer

`tokenizer.ggml.model = "t5"` selects llama.cpp's SentencePiece-unigram tokenizer
(`LLAMA_VOCAB_TYPE_UGM`, `src/llama-vocab.cpp:2341-2342`); the file carries no
`tokenizer.ggml.pre`, so the pre-tokenizer is DEFAULT and the vocabulary's own
normalization (the precompiled charsmap) does the work:

- normalize with the XLM-R charsmap (NFKC-like; the `▁` metaspace is the space);
- `add_space_prefix` prepends `▁` and `remove_extra_whitespaces` collapses runs;
- the unigram Viterbi picks the highest-scoring segmentation;
- `<s>` is pushed when `add_bos_token` is set and `</s>` when `add_eos_token` is set
  (`src/llama-vocab.cpp:3946-3960`); this file sets both, so an input is
  `<s> … </s>`.

Peregrine's tokenizer already served this vocabulary for `bge-m3` (the same XLM-R
250k unigram model), and its ids are identical to llama.cpp's on all 200 gate texts,
including 24 multilingual sentences (CJK, Arabic, Hebrew, Devanagari, Cyrillic,
Greek, Thai, Korean, Turkish, Vietnamese and the Latin-script languages).

## What the client writes, and `dimensions`

The model card's task prefixes are `search_query: `, `search_document: `,
`clustering: ` and `classification: `. **The client writes the prefix**, exactly as it
does with llama.cpp: the model does not add one, and Peregrine does not either. The
gate's fixtures are `search_document: ` + text.

Matryoshka: the request's `dimensions` field truncates the pooled vector to its first
N components **and then re-normalises** (the server's `embed::finish`, design §2.4).
Any N up to 768 is accepted.

## Accuracy: against llama.cpp and against the full-precision reference

Measured on the 200 texts in `crates/peregrine-models/tests/fixtures/nomic-bert-moe/`
(17 to 511 tokens: 156 `bert` paragraphs, 24 multilingual sentences and 20
concatenations trimmed to the 512-token context; every text prefixed with
`search_document: `), 4 threads, L2-normalised vectors. The Q8_0 reference is
llama.cpp on the **F16 conversion**, as the design states (§5).

| File | ids | vs llama.cpp min / mean | vs the F16 reference min / mean | llama.cpp's own Q8_0 vs the reference min / mean |
|---|---|---|---|---|
| nomic-embed-text-v2-moe F16 | identical | **0.9997738 / 0.9999977** | 0.9997738 / 0.9999977 | — |
| nomic-embed-text-v2-moe Q8_0 | identical | 0.9615146 / 0.9995664 | **0.9625625 / 0.9991114** | 0.9990825 / 0.9994075 |

**The S4 F16 clause (min ≥ 0.9999) is not met on this model, and this note records
that as a measured deviation rather than hiding it.** The mean is 0.9999977 — the
siblings' class — and 199 of the 200 texts are at 0.99996 or better. The minimum is
one 96-token text whose **last MoE layer's router** sits near a choice boundary: the
two engines' hidden states differ by about 0.2 % after ten layers (the ordinary f32
accumulation difference), and the discrete top-2 choice amplifies that into about
2 % on the final vector. Truncating the model layer by layer against llama.cpp shows
the continuous agreement is 0.9999999 after one layer and 0.999995 after eight; the
jump is the last layer's selection. This is a property of a mixture of experts, not
a stale bound: llama.cpp's own Q8_0 run, which differs from its F16 run by about the
same continuous amount, also has a worst text at 0.99908.

For Q8_0 the mean clause ("no further from the full-precision reference than
llama.cpp's own Q8_0, within 1e-4") is missed by **3.0e-4**: 0.9991114 against
0.9994075. The outliers are the longest texts (185: 0.9626, 102: 0.9890) where the
quantised run's larger hidden-state drift reaches the later MoE layers' near-ties;
short texts are at llama.cpp's own Q8_0 distance. Our Q8_0 arithmetic and llama.cpp's
agree per layer (their first layer is 0.9999909 and 0.9999910 from the F16 run), so
the gap is the routing's amplification, not a kernel error: the Q8_0 file measured
against Peregrine's own reference forward is within 1e-6 of the reference's per-row
arithmetic at every synthetic length and pooling.

## Speed

`peregrine serve`'s `/v1/embeddings` and `llama-server --embeddings` on both files,
4 threads, one engine at a time, embedding cache off for Peregrine
(`--opt EMBED_CACHE_MB=0`), two warm-up requests and five timed ones per cell, each
request with **fresh token ids** (and the 16-text cell with 16 different texts), the
median and (min..max). texts/s, ratio Peregrine ÷ llama.cpp (above 1 means Peregrine
is faster). The host is the shared 8-vCPU AVX2 VM the other notes use; the load
average was 2.39 before the run and 6.14 after, so these are rough.

| Case | Peregrine F16 | llama.cpp F16 | ratio | Peregrine Q8_0 | llama.cpp Q8_0 | ratio |
|---|---|---|---|---|---|---|
| 16 tokens, 1 text | 12.55 (0.0554..0.0924 s) | 12.65 (0.0659..0.1496 s) | 0.99 | 23.14 (0.0416..0.0492 s) | 19.09 (0.0490..0.0651 s) | **1.21** |
| 128 tokens, 1 text | 2.31 (0.4058..0.4905 s) | 2.78 (0.3399..0.5123 s) | 0.83 | 2.91 (0.3230..0.3831 s) | 3.31 (0.2867..0.3472 s) | 0.88 |
| 512 tokens, 1 text | 0.45 (1.8055..3.0532 s) | 0.72 (1.3626..1.4111 s) | 0.62 | 0.67 (1.3214..1.6617 s) | 0.80 (1.2473..1.3016 s) | 0.84 |
| 32 tokens, 16 texts | 9.04 (1.6067..2.4560 s) | 11.45 (1.3691..1.5767 s) | 0.79 | 11.52 (1.3154..1.4282 s) | 11.72 (1.1663..1.8604 s) | 0.98 |

Every cell below 1.0 is named: F16 at 16 tokens (0.99), 128 (0.83), 512 (0.62) and
the 16-text batch (0.79); Q8_0 at 128 (0.88), 512 (0.84) and the batch (0.98). Only
the 16-token Q8_0 cell is above llama.cpp. The cause is the float path: the
mixture-of-experts' **3D expert tensors are not in the float pack** (S4.9f packs the
2-dimensional F16/F32 matrices only, `fpack.rs:18-19`), so the F16 experts run the
raw per-row kernel while llama.cpp runs its tuned f16 GEMM; the Q8_0 experts reach
the batched int8 kernel and the Q8_0 rows are close to llama.cpp. Packing the 3D
expert banks per expert is the next step for this family's float speed, and it was
left for S4.9 rather than changing every MoE family's goldens in this step.

## What Peregrine does, and where it differs on purpose

- **One family file.** `NomicFfn` (SwiGLU, GELU, MoE) is the only branch: the
  attention half, the tensor list, RoPE, the LayerNorms and the pooling are shared
  with `nomic-bert`, because llama.cpp's graph is literally the same file with two
  branch points. `NomicBertFamily` and `NomicBertMoeFamily` are two registrations of
  that loader, behind `arch-nomic-bert` and `arch-nomic-bert-moe`.
- **The MoE block is the engine's.** `blocks::moe::moe_ffn` already routes with a
  softmax and top-k, groups rows by expert and sums in selection order; this step
  added the plain-GELU, gate-less expert shape (`MoeAct::Gelu`) and an empty gate
  bank. No new kernel: the expert views are 2D slices (`expert_view`) and the
  matmuls are `matmul_group`'s.
- **GELU is the function, not the f16 table.** `vmath::gelu_tanh` evaluates
  llama.cpp's tanh form exactly; llama.cpp's CPU `ggml_gelu` rounds the input and the
  output to f16 and reads a table (`ggml/src/ggml-cpu/vec.h:1000-1019`). This is the
  deliberate ADR 0003 difference `jina-bert-v2`'s GEGLU already states. Substituting
  the table in this family changes the F16 vectors in the eighth decimal only — the
  routing difference dominates.
- **The ties are fixed.** llama.cpp's `std::sort` may order exactly equal experts
  either way; Peregrine takes probability descending then expert index ascending,
  per row, as `docs/design/moe.md` §1 states.
- **The router is F32, as the file is.** Even the Q8_0 conversion keeps
  `ffn_gate_inp.weight` in F32; the block's activations keep their f32 rows
  (`MoeScratch::set_keep_f32`) so the router never sees an int8 row.
- **The context limit is the file's**, 512; a text longer than it is refused by
  name (or truncated under `truncate`). Chat and label slots are refused on the
  encoder by name.
- **Tests**: a synthetic `nomic-bert-moe` file (`tests/support/models.rs`) with four
  layers, four experts of which a row uses two, MoE on the odd layers, the
  attention and plain-FFN biases, the XLM-R unigram tokenizer, and malformed
  variants (wrong expert count, wrong router rows, a missing tensor); the
  independent scalar reference (`tests/support/nomic_bert_moe_ref.rs`) that writes
  the routing out longhand; the batch/thread/pass-budget invariance matrix on F32
  and Q8_0; the no-allocation pass; server and CLI cases; and one golden per pinned
  x86 tier (`tests/golden/synthetic/<tier>/nomic-bert-moe-Q8_0.golden`: scalar,
  AVX2 and the AVX-512 tier under SDE). The 200-text fixtures and the ignored
  real-file test live in `tests/fixtures/nomic-bert-moe/` and `tests/embeddings.rs`.
