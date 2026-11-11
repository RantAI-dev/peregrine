# `gemma-embedding` (EmbeddingGemma 300M)

Reference note for roadmap S4.4, transcribed from **llama.cpp v0.9.0, commit `cd959ba`**:
`src/models/gemma-embedding.cpp`, `src/llama-model.cpp`, `src/llama-graph.cpp`,
`src/llama-hparams.h` and `src/llama-arch.cpp`. Every detail here was read from that
source, not from the design note. Peregrine's implementation is
`crates/peregrine-models/src/gemma3/mod.rs` (`Gemma3::load_impl`, `forward_texts`) and
`crates/peregrine-core/src/blocks/bidir.rs`.

## What it is

Google's EmbeddingGemma 300M, shipped as `architecture = "gemma-embedding"`: the Gemma 3
blocks as an encoder, mean-pooled, then two sentence-transformers dense projections and
L2 normalisation. The reference file used for the S4.4 gate is
`ggml-org/embeddinggemma-300M-GGUF`, `embeddinggemma-300M-Q8_0.gguf`
(SHA-256 `b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63`).

## Metadata

All keys are under the `gemma-embedding.` prefix.

| Key | Value on the reference file | Meaning |
|---|---|---|
| `block_count` | 24 | layers |
| `embedding_length` | 768 | `d_model` |
| `attention.head_count` | 3 | query heads |
| `attention.head_count_kv` | 1 | KV heads (GQA 3:1) |
| `attention.key_length` / `value_length` | 256 | head width |
| `feed_forward_length` | 1152 | FFN width |
| `context_length` | 2048 | training context |
| `attention.sliding_window` | 512 | `n_swa`; symmetric window, `n_swa / 2` each way |
| `attention.layer_norm_rms_epsilon` | 1e-6 | RMSNorm epsilon |
| `rope.freq_base` | 1e6 | global-layer RoPE base |
| `rope.freq_base_swa` | 1e4 | sliding-layer RoPE base |
| `dense_2_feat_in` | 768 | must equal `n_embd` |
| `dense_2_feat_out` | 3072 | |
| `dense_3_feat_in` | 3072 | |
| `dense_3_feat_out` | 768 | must equal `n_embd` |
| `pooling_type` | 1 | mean |
| `tokenizer.ggml.add_bos_token` | true | |
| `tokenizer.ggml.add_eos_token` | true | |

The architecture sets `hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC`,
`load_swa_pattern(ml, 6)` (the same six-layer pattern as Gemma 3: `il % 6 < 5` sliding,
`il % 6 == 5` global), `hparams.causal_attn = false`, and
`f_attention_scale = 1/sqrt(n_embd_head_k)` (no 27B special case).

## Tensors

| Tensor | Shape | Notes |
|---|---|---|
| `token_embd.weight` | `[n_embd, n_vocab]` | tied input/output |
| `output_norm.weight` | `[n_embd]` | `result_norm` |
| `blk.N.attn_norm.weight` | `[n_embd]` | |
| `blk.N.attn_q.weight` | `[n_embd, n_head*head_dim]` | |
| `blk.N.attn_k.weight` | `[n_embd, n_kv_head*head_dim]` | |
| `blk.N.attn_v.weight` | `[n_embd, n_kv_head*head_dim]` | |
| `blk.N.attn_q_norm.weight` | `[head_dim]` | per-head RMSNorm |
| `blk.N.attn_k_norm.weight` | `[head_dim]` | per-head RMSNorm |
| `blk.N.attn_output.weight` | `[n_head*head_dim, n_embd]` | |
| `blk.N.post_attention_norm.weight` | `[n_embd]` | |
| `blk.N.ffn_norm.weight` | `[n_embd]` | |
| `blk.N.ffn_gate.weight` / `ffn_up.weight` | `[n_embd, n_ff]` | GeGLU |
| `blk.N.ffn_down.weight` | `[n_ff, n_embd]` | |
| `blk.N.post_ffw_norm.weight` | `[n_embd]` | |
| `dense_2.weight` | `[n_embd, dense_2_feat_out]` | optional; no bias |
| `dense_3.weight` | `[dense_3_feat_in, n_embd]` | optional; no bias |

The loader (`llama_model_gemma_embedding::load_arch_tensors`) creates the dense tensors with
`TENSOR_NOT_REQUIRED` and **no bias tensor**; `llama_model.cpp` never sets
`dense_2_out_layers_b` for this architecture. There is no output head: the graph stops at
`result_norm`, and pooling is built from it.

## The forward, in order

Per layer (`llama_model_gemma_embedding::graph`):

1. `x = tok_embd[token] * sqrt(n_embd)`.
2. `attn_norm` RMSNorm.
3. Q/K/V projections. No biases.
4. Q per-head RMSNorm, then RoPE with `get_rope_freq_base(cparams, il)` (the SWA base on a
   sliding layer, the global base otherwise). Same for K.
5. `Q *= f_attention_scale` (`1/sqrt(head_dim)`), then attention with `kq_scale = 1`.
6. `attn_output`; `attn_post_norm`; residual add.
7. `ffn_norm`; gate and up; `LLM_FFN_GELU` with `LLM_FFN_PAR` (`gelu(gate) * up`); down;
   `post_ffw_norm`; residual add.

After the last layer: `output_norm` → `result_norm`. No logits, no LM head.

### The mask

`build_attn_inp_no_cache` builds a full `n_tokens × n_tokens` mask. `fill_mask` skips a pair
when it belongs to another sequence, when causal (`p0 > p1`) and when `is_masked_swa` says
so. `causal = cparams.causal_attn`, which is false for this architecture; for the sliding
mask, `causal_swa = causal_attn || non_causal_type == DENSE_ONLY`, also false. So:

- global layer: every key of the text is visible to every query;
- sliding layer (`LLAMA_SWA_TYPE_SYMMETRIC`): masked when
  `|p1 - p0| > n_swa / 2`, with integer division — for `n_swa = 512`, a symmetric ±256
  window.

No KV cache is written: `build_attn_inp_no_cache` pads K/V to 256 for the mask and reads
the batch's own keys and values. This is why an encoder text cannot be chunked.

## Pooling and the dense head

`llama_model::build_graph` calls `build_pooling` and then
`build_dense_out(dense_2, dense_2_b, dense_3)`:

1. `build_pooling` with `pooling_type = MEAN` computes `transpose(result_norm) · inp_mean`:
   the mean over the sequence's tokens. (`CLS`/`LAST` would take row 0/the last row;
   `RANK` would run the classifier head, which this family does not have.)
2. `build_dense_out` only runs when `cparams.embeddings` is set:
   `cur = dense_2 · cur`, then `cur = dense_3 · cur`. There is no bias and no activation
   between them (`dense_2_b` is null here).
3. The embedding example normalises with `common_embd_normalize(..., 2)`: sum of squares in
   **f64**, `sqrt`, `1/sum`, then each component `* norm` (one f32 rounding of the factor).
   Truncation to `dimensions` happens before that in the server/CLI, as Matryoshka requires.

So a text's vector is `normalise(dense_3 · dense_2 · mean_rows(output_norm(x)))`, and a
file without the two dense tensors (some converters omit them) is
`normalise(mean_rows(output_norm(x)))` — a different vector space, as the measurements in
the S4.4 PR show.

## Accuracy: against llama.cpp and against the truth

The Q8_0 upload is quantised: its weights differ from the unquantised model, so the meaningful
question is which engine is closer to the unquantised vectors, not which two engines agree. The
F32/F16/BF16 uploads on Hugging Face (`unsloth/embeddinggemma-300m-GGUF`,
`second-state/embeddinggemma-300m-GGUF`, `gaianet/embeddinggemma-300m-GGUF`) were converted
without `--sentence-transformers-dense-modules` and carry no `dense_2`/`dense_3`, so they cannot
be the reference for the full pipeline. The reference here is converted from the ungated mirror
of the original weights (`unsloth/embeddinggemma-300m`, commit `bfa3c84`) with llama.cpp
`cd959ba`:

```sh
python3 convert_hf_to_gguf.py <unsloth-embeddinggemma-300m> \
    --outtype f16 --sentence-transformers-dense-modules --outfile embeddinggemma-f16.gguf
```

That file has 316 tensors (including `dense_2` and `dense_3`), 621,867,232 bytes, SHA-256
`eeccb769249a2bac5e23b3e408c6ea17d016d6d8c273c3b5bdb23b7fdd955d44`; its tensors agree with the
Q8_0 upload's to cosine 0.99997, so the two are the same model at two precisions. The vectors
below are llama.cpp v0.9.0's (`llama-server --embedding --pooling mean`, one text per request)
on 200 mixed-length texts (16–451 tokens), cosine of L2-normalised vectors, 256 d =
truncate-then-normalise:

| Engine | Width | min | mean | max |
|---|---|---|---|---|
| llama.cpp Q8_0 | 768 | 0.998013 | 0.9994734 | 0.9996802 |
| Peregrine Q8_0 | 768 | 0.998747 | 0.9994728 | 0.9996988 |
| Peregrine F16 | 768 | 0.999983 | 0.9999997 | 0.9999999 |
| llama.cpp Q8_0 | 256 | 0.998398 | 0.9995551 | 0.9997291 |
| Peregrine Q8_0 | 256 | 0.998900 | 0.9995524 | 0.9997827 |
| Peregrine F16 | 256 | 0.999987 | 0.9999997 | 0.9999999 |

Against the F16 reference: Peregrine's Q8_0 minimum is 7.3e-4 **higher** than llama.cpp's at
768 d and 5.0e-4 higher at 256 d; the means differ by 6e-7 and 2.7e-6 (within the 1e-4 slack).
The same-engine F16 row (0.99998) shows the forward graph itself is right; the Q8_0 gap is the
two engines' int8 activation arithmetic, not the mask, the head or the pooling. Cross-engine
Q8_0 against Q8_0 on these texts is min 0.998168 / mean 0.999639 at 768 d (min 0.998583 /
mean 0.999701 at 256 d). The worst text at 768 d is a 64-token text (not the longest): both
engines' error to the reference peaks there, llama.cpp at 0.998013 and Peregrine at 0.999153.

The fixtures (`texts.txt`, `llama_q8_0.f32`, `llama_f16.f32`) and the ignored test that enforces
the bound are in `crates/peregrine-models/tests/fixtures/embeddinggemma/` and
`crates/peregrine-models/tests/embeddings.rs`.

## What Peregrine does

- `Gemma3::forward_texts` runs the projections over every row of every text in matmul-sized
  chunks (the shared GEMM takes at most 128 rows per call), keeps f32 Q/K/V for the whole
  pass, and computes attention one text at a time with the transcribed symmetric mask.
- Pooling is the text's own rows, in position order, with an f64 accumulator and one
  rounding (`embed::pool_rows_into`).
- `dense_2` then `dense_3`, both pure matmuls, then the caller's `dimensions` truncation and
  L2 normalisation (`embed::finish`).
- The engine schedules whole texts and allocates no KV cache for an encoder slot.
