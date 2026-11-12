# `nomic-bert` (nomic-embed-text v1.5)

Reference note for roadmap **S4.8a**, transcribed from **llama.cpp v0.9.0, commit
`cd959ba`**: `src/models/bert.cpp` (the shared encoder graph and its
`LLM_ARCH_NOMIC_BERT` branches), `src/models/nomic-bert.cpp` (the hparams and the
tensor list), `src/llama-model.cpp` (the arch dispatch, the rope type, the pooling
key), `src/llama-graph.cpp` (`build_inp_embd`, `build_qkv`, `build_attn_inp_no_cache`,
`build_attn`, `build_ffn`, `build_pooling`), `src/llama-vocab.cpp` (the WordPiece
tokenizer and its flags), `conversion/bert.py` (the `NomicBertModel` converter class)
and `gguf-py/gguf/tensor_mapping.py` (which HF tensor becomes which GGUF tensor).
Every statement below was read from that source, not from the design note. Peregrine's
implementation is `crates/peregrine-models/src/nomic_bert.rs`
(`NomicBert::load`, `forward_texts`); the pooling is
`crates/peregrine-core/src/embed.rs`; the scalar reference forward is
`crates/peregrine-models/tests/support/nomic_bert_ref.rs`.

## What it is

`general.architecture = "nomic-bert"` is Nomic AI's BERT-shaped encoder
(`llama-arch.cpp:25`). `nomic-embed-text-v1.5` is the 137M-parameter member: 12
post-norm encoder blocks, 768 wide, 12 heads of 64, a SwiGLU feed-forward of 3072,
**RoPE instead of a learned position table**, and mean pooling. It is the same graph
that serves `nomic-bert-moe` and the jina BERT variants, branched per arch inside
`llama_model_bert::graph` (`src/models/bert.cpp`); the MoE variant is not part of
this step (roadmap S4.8b) and is refused by name.

The differences from [`bert`](bert.md) are the ones the branch points select:

| | `bert` | `nomic-bert` |
|---|---|---|
| Positions | a learned `position_embd` table | **RoPE** on q and k, NeoX pairing, base `rope.freq_base` |
| Token type | row 0 of `token_types.weight` | row 0 of `token_types.weight` (the same) |
| q/k/v | three matrices | **one fused `attn_qkv`** (q rows, then k, then v) |
| Feed-forward | `ffn_up` → exact GELU → `ffn_down` | **SwiGLU**: `silu(ffn_gate) ⊙ ffn_up` → `ffn_down` |
| Biases | q/k/v/out/ffn all have them | none in v1.5's files (the MoE variant has them) |
| Output norm | none (`layer_output_norm` is the last) | none (the same) |

## Metadata

The reference file is nomic-ai's own conversion,
`nomic-ai/nomic-embed-text-v1.5-GGUF`. All model keys are under the `nomic-bert.`
prefix.

| Key | Value | Meaning |
|---|---|---|
| `block_count` | 12 | layers |
| `embedding_length` | 768 | `d_model` |
| `feed_forward_length` | 3072 | `n_ff`, the width of each half of the SwiGLU pair |
| `attention.head_count` | 12 | heads; `head_dim = 768/12 = 64` |
| `attention.head_count_kv` | absent | keys and values are as wide as queries |
| `context_length` | 2048 | the longest text, in tokens (the converter rewrites `n_positions` to `max_trained_positions`, `conversion/bert.py:336-342`) |
| `attention.layer_norm_epsilon` | 1e-12 | LayerNorm epsilon (`layer_norm_epsilon` in the HF config) |
| `attention.causal` | false | no causal mask (`BertModel.set_gguf_parameters` calls `add_causal_attention(False)`) |
| `rope.freq_base` | 1000 | `rotary_emb_base` |
| `rope.dimension_count` | absent | llama.cpp then rotates the whole head (`n_rot_full = n_embd_head_k_full`, `src/llama-model.cpp:1601-1608`) |
| `pooling_type` | 1 | mean |
| `vocab_size` | 30522 | the WordPiece vocabulary's size |
| `tokenizer.ggml.model` | `bert` | WordPiece |
| `tokenizer.ggml.pre` | absent | the DEFAULT pre-tokenizer, with `clean_spaces = true` (`src/llama-vocab.cpp:2780-2789`) |
| `tokenizer.ggml.token_type_count` | 2 | rows of `token_types.weight`; llama.cpp's loader throws when it is 0 (`src/models/bert.cpp:24-26`) |
| `tokenizer.ggml.add_bos_token` / `add_eos_token` | absent | WordPiece's tokenizer adds `[CLS]` and `[SEP]` whenever specials are on, whatever these say (`src/llama-vocab.cpp:3928-3950`) |

The reference files and their SHA-256:

| File | Repo | SHA-256 |
|---|---|---|
| `nomic-embed-text-v1.5.f16.gguf` | `nomic-ai/nomic-embed-text-v1.5-GGUF` | `f7af6f66802f4df86eda10fe9bbcfc75c39562bed48ef6ace719a251cf1c2fdb` |
| `nomic-embed-text-v1.5.Q8_0.gguf` | same | `3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7` |

## Tensors

`nomic-bert-Q8_0.gguf` holds 112 tensors, in this order (the F16 file is the same
list with F16 matrices):

| Tensor | Shape | Notes |
|---|---|---|
| `token_embd_norm.bias` / `.weight` | `[768]` | the embedding LayerNorm, **weight and bias** |
| `token_types.weight` | `[768, 2]` | row 0 is added to every token; row 1 is never read |
| `token_embd.weight` | `[768, 30522]` | |
| `blk.N.attn_qkv.weight` | `[768, 2304]` | fused: q rows 0..768, k 768..1536, v 1536..2304 |
| `blk.N.attn_output.weight` | `[768, 768]` | |
| `blk.N.ffn_up.weight` | `[768, 3072]` | HF's `fc11`: the **multiplied** half |
| `blk.N.ffn_gate.weight` | `[768, 3072]` | HF's `fc12`: the **activated** half |
| `blk.N.ffn_down.weight` | `[3072, 768]` | |
| `blk.N.attn_output_norm.bias` / `.weight` | `[768]` | the post-attention LayerNorm |
| `blk.N.layer_output_norm.bias` / `.weight` | `[768]` | the post-FFN LayerNorm; the last one **is** the encoder's output |

There is **no `output_norm`**, no position table and no projection bias in
v1.5's files: the converter asserts `qkv_proj_bias == mlp_fc1_bias == mlp_fc2_bias
== is_moe`, so only the MoE variant writes biases (`conversion/bert.py:344-358`).
llama.cpp marks `attn_output.bias`, `ffn_up.bias` and `ffn_down.bias`
`TENSOR_NOT_REQUIRED`, and the fused `attn_qkv.bias` likewise
(`create_tensor_qkv`, `src/llama-model.cpp:3994-4021`); Peregrine reads the q/k/v and
`attn_output` biases when they exist and, as llama.cpp's `else` feed-forward branch
passes NULL for the FFN biases, never adds the FFN ones for this arch.

The FFN naming is worth stating once because it is inverted from what the names
suggest: HF's `NomciBertGatedMLP` computes `y = fc11(x) * activation(fc12(x))`
(`nomic-bert-2048/modeling_hf_nomic_bert.py:1086-1088`), and
`gguf-py/gguf/tensor_mapping.py` maps `fc11 → FFN_UP` and `fc12 → FFN_GATE`
(the same file's `FFN_UP`/`FFN_GATE` lists). `build_ffn` with `LLM_FFN_SILU` and
`LLM_FFN_PAR` activates the gate output and multiplies it by the up output
(`src/llama-graph.cpp:2011-2064`), i.e. `silu(ffn_gate) ⊙ ffn_up =
silu(fc12) ⊙ fc11`, which is HF's expression.

## The forward, in order

`llama_model_bert::graph::graph` (`src/models/bert.cpp:73-224`), for one sequence of
`n` tokens:

1. `inpL = token_embd[token]` (`build_inp_embd`, line 88).
2. If the file has `token_types.weight`, `inpL += token_types[0]` — token types are
   hardcoded to zero, "Sentence A" (lines 91-94). There is **no position table**:
   `inp_pos` is built only to feed RoPE (line 79-81), and the `LLM_ARCH_BERT` branch
   that adds `position_embd` is skipped (lines 97-99).
3. `inpL = LayerNorm(inpL, token_embd_norm.weight, token_embd_norm.bias)`
   (line 101, `LLM_NORM`).
4. Per layer `il`, **post-norm** (there is no pre-attention norm):
   1. `Qcur, Kcur, Vcur = qkv(inpL)` from the fused `attn_qkv` (+ its bias if
      present), split q / k / v (lines 108-113; `build_qkv`,
      `src/llama-graph.cpp:1855-1894`).
   2. `Qcur, Kcur = RoPE(Qcur, Kcur, inp_pos)` — the `LLM_ARCH_NOMIC_BERT` branch
      (lines 130-139). No q/k norm tensors exist for this arch, so those branches
      do nothing (lines 115-128).
   3. `cur = attn(Qcur, Kcur, Vcur, wo, wo_b)` with `kq_scale = 1/√n_embd_head`
      (lines 145-147; no KV cache: `build_attn_inp_no_cache`, all positions
      visible).
   4. `cur = cur + inpL`; `cur = LayerNorm(cur, attn_output_norm)` (lines 156-160).
      `attn_norm_2` does not exist for this arch (lines 162-165), so no second
      residual/norm.
   5. `ffn_inp = cur` (line 167).
   6. `cur = ffn(cur)`: `silu(ffn_gate · cur) ⊙ (ffn_up · cur)`, then
      `ffn_down · that` — the **`else`** branch of the arch chain
      (lines 199-208), with every bias NULL.
   7. `cur = cur + ffn_inp`; `cur = LayerNorm(cur, layer_output_norm)` (lines
      212-215).
   8. `inpL = cur`.
5. `res->t_embd = inpL` (line 219-221). There is **no final norm**: the last
   layer's `layer_output_norm` output is what pooling reads.

### RoPE

`llama_model_rope_type` lists `LLM_ARCH_NOMIC_BERT` in the offset-by-`n_rot/2`
group (`src/llama-model.cpp:3690-3697`), i.e. `LLAMA_ROPE_TYPE_NEOX`: element `i` of
a rotary pair is paired with `i + n_rot/2`, not with `i + 1`. The HF config agrees:
`"rotary_emb_interleaved": false` (`conversion/bert.py:356-358` asserts it), and
HF's `apply_rotary_emb_torch(..., interleaved=False)` rotates the two halves.
`n_rot` is the head width (64) and the base is `rope.freq_base` (1000); there is no
rope scaling (`rotary_emb_scale_base` is asserted null by the converter).

### Pooling

`<arch>.pooling_type` is read generically (`src/llama-model.cpp:1473`) and
`build_pooling`'s `LLAMA_POOLING_TYPE_MEAN` branch averages the final rows
(`src/llama-graph.cpp:4174-4180`). nomic-embed-text v1.5's file says `1` (mean). The
server then L2-normalises (`common_embd_normalize(..., 2)`).

### The tokenizer

`tokenizer.ggml.model = "bert"` selects llama.cpp's WordPiece tokenizer
(`LLAMA_VOCAB_TYPE_WPM`); the file carries no `tokenizer.ggml.pre`, so the
pre-tokenizer is `LLAMA_VOCAB_PRE_TYPE_DEFAULT` with `clean_spaces = true`
(`src/llama-vocab.cpp:2780-2789`):

- split on Unicode whitespace;
- each punctuation character (and ASCII symbol, and CJK character) is its own word;
- a word is prefixed with the phantom `▁` the converter stores (`▁word`,
  `continuation` for `##` pieces — the converter rewrites the vocabulary in
  `BertModel.set_vocab`, `conversion/bert.py:40-67`);
- longest-match-first over the vocabulary, and a word with no full match becomes
  `[UNK]`;
- `[CLS]` is pushed before the text and `[SEP]` after it whenever specials are on
  (`src/llama-vocab.cpp:3928-3950`).

No lowercasing or accent stripping: llama.cpp reads those from
`tokenizer.ggml.normalizer.lowercase` / `.strip_accents` (`src/llama-vocab.cpp:3016-3020`),
and this file carries neither, so the vocabulary's lowercase form is used as-is.
Peregrine matches on all 200 gate texts (mixed lengths, punctuation, digits, URLs,
accents and CJK).

## What the client writes, and `dimensions`

nomic-embed-text v1.5 is trained with task prefixes: `search_document: ` for a
passage and `search_query: ` for a query (v2 also uses `classification: ` and
`clustering: `).
**The client writes the prefix**, exactly as it does with llama.cpp: the model does
not add one, and Peregrine does not either. The gate's fixtures are
`search_document: ` + text, and a server client does the same.

Matryoshka: the request's `dimensions` field truncates the pooled vector to its
first N components **and then re-normalises** (the server's existing
`embed::finish`, design §2.4). Nomic's model card recommends 512, 256 or 64
dimensions for v1.5; any N up to 768 is accepted.

## Accuracy: against llama.cpp and against the full-precision reference

The S4 gate on the 200 texts in `crates/peregrine-models/tests/fixtures/nomic-bert/`
(20 to 2040 tokens; the five long ones are concatenations close to 2048, so the
whole context runs), 4 threads, L2-normalised vectors, `search_document: ` in front
of every text. The Q8_0 reference is llama.cpp on the **F16 conversion**, as the
design states (§5).

| File | ids | vs llama.cpp min / mean | vs the F16 reference min / mean | llama.cpp's own Q8_0 vs the reference min / mean |
|---|---|---|---|---|
| nomic-embed-text v1.5 F16 | identical | **0.9999995 / 0.9999998** | 0.9999995 / 0.9999998 | — |
| nomic-embed-text v1.5 Q8_0 | identical | 0.9997719 / 0.9998660 | **0.9978348 / 0.9982994** | 0.9978020 / 0.9982981 |

F16 is at the 0.9999-class bound. For Q8_0 the gate's clause is on the **mean**:
Peregrine is 1.3e-6 *closer* to the F16 reference than llama.cpp's own Q8_0 is,
inside the 1e-4 band, and the per-text minimum is 3.3e-5 better too. This family's
Q8_0 conversion is further from its F16 conversion than a BERT encoder's is — a
0.9983 mean for both engines, not the 0.9996 the `modern-bert` files reach — so the
test's measured bound is min 0.9975 / mean 0.9982, not a family-generic one. The
cause is the Q8_0 weights themselves, not either engine's activation path: the two
engines' Q8_0 vectors agree to 0.99987 mean.

## Speed

`peregrine serve`'s `/v1/embeddings` and `llama-server --embeddings` on the F16 file,
4 threads, one engine at a time, embedding cache off for Peregrine
(`--opt EMBED_CACHE_MB=0`), 5 runs per cell, median and (min..max). One text of N
tokens, and a batch of 16 texts of 32 tokens in one request; texts/s, ratio
Peregrine ÷ llama.cpp (above 1 means Peregrine is faster). The host is H0, the
shared 8-vCPU Xeon E5-2680 v4 (AVX2+FMA) the other notes use; these timings are
rough because other jobs run on it.

| Case | Peregrine | llama.cpp | ratio |
|---|---|---|---|
| 16 tokens, 1 text | 34.5 (33.7..36.1) | 20.9 (20.0..21.6) | **1.65** |
| 128 tokens, 1 text | 5.70 (5.60..5.82) | 4.02 (3.90..4.04) | 1.42 |
| 512 tokens, 1 text | 1.31 (1.22..1.31) | 0.96 (0.94..0.97) | 1.36 |
| 32 tokens, 16 texts | 20.8 (20.4..21.2) | 15.9 (15.6..15.9) | **1.31** (worst) |

The inputs are token-id arrays, so the token count is exact; the two tokenizers are
the same, and the text path is not what separates the engines at these sizes. The
16-text batch is 512 rows and Peregrine's embedding scheduler runs 32 rows per pass
(the calibrated default), so the batch pays several weight passes; llama.cpp runs
the request as one ubatch. Even so Peregrine is 1.31× ahead on that cell, the worst
of the four, and 1.65× on the short single text, where per-request overhead
dominates.

## What Peregrine does, and where it differs on purpose

- `NomicBert::forward_texts` runs one whole text per unit (never chunked, no KV
  cache), with all texts of a pass side by side in the shared GEMM. The fused
  `attn_qkv` is one projection, its rows are split into q/k/v, `prepare_rows` adds
  the optional bias and applies RoPE with the state's table, and attention is the
  shared `blocks::bidir::attend_texts` with no window.
- Q is scaled by `1/√head_dim` after RoPE and the attention block is called with
  scale 1; llama.cpp passes the scale to `build_attn`. It is the same product with
  one extra rounding, and the reference forward in the tests does the same.
- The SwiGLU is `vmath::swiglu(ffn_gate, ffn_up)`, the same kernel Granite
  Embedding R2's file uses; the LayerNorms are `vmath::layer_norm_rows` (weight and
  bias, f64 mean/variance walks in index order, one f32 rounding), which the `bert`
  family already uses.
- The family refuses a text longer than `context_length`, naming it, and the engine
  refuses chat and label requests on it (`is an embedding model …`).
- Tests: a synthetic `nomic-bert` file (`tests/support/models.rs`) with four
  layers, RoPE base 1000, epsilon 1e-12, a with-bias variant, F32 and Q8_0; the
  independent scalar reference (`tests/support/nomic_bert_ref.rs`); the
  batch/thread invariance matrix; the no-allocation test; server and CLI tests; one
  golden per pinned x86 tier (`tests/golden/synthetic/<tier>/nomic-bert-Q8_0.golden`:
  scalar, AVX2 and the AVX-512 tier under SDE). The aarch64 goldens are not
  recorded on this x86 host.
- The 200-text fixtures and the ignored real-file test live in
  `tests/fixtures/nomic-bert/` and `tests/embeddings.rs`.
- `nomic-bert-moe` (nomic-embed-text-v2-moe) shares this family's graph in
  llama.cpp, but adds a mixture-of-experts feed-forward on every second layer and
  its XLM-R unigram tokenizer. It is not served in this step; a `nomic-bert-moe`
  file is refused by name.
