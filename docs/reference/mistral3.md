# `mistral3` embedders (NVIDIA Nemotron 3.5 Embed 1B)

Reference note for roadmap S4.14, transcribed from **llama.cpp v0.9.0, commit `cd959ba`**:
`src/models/mistral3.cpp`, `src/llama-graph.cpp`, `src/llama-context.cpp`, `src/llama-hparams.h`.
Every detail here was read from that source. Peregrine's implementation is
`crates/peregrine-models/src/dense/mod.rs` (`Dense::load`, `forward_texts_embedding`),
`crates/peregrine-core/src/blocks/bidir.rs` and `crates/peregrine-models/src/encoder.rs`.

## What it is

NVIDIA's Nemotron 3.5 Embed 1B is a retrieval embedder for text and page images in 34 languages.
The text model is a 16-layer Mistral 3 transformer run **without the causal mask**: a GGUF with
`general.architecture = "mistral3"`, `mistral3.attention.causal = false` and
`mistral3.pooling_type = 1` (mean). The reference files are
`Nemotron-3.5-Embed-1B-{F16,Q8_0,Q4_K_M}.gguf` (SHA-256 in
`crates/peregrine-models/tests/fixtures/nemotron/README.md`). The model is published under the
OpenMDW 1.1 licence.

The same family's causal members are the Mistral 3 chat models (Ministral 3): they carry
`attention.causal = true` and are **not** supported here — Peregrine refuses them by name.
`mistral3` is a different architecture entry from `mistral4` (Mistral Large 4); the file has no
vision tensors in the text GGUF (the Pixtral vision encoder ships as a separate `mmproj` file).

## Metadata

All keys are under the `mistral3.` prefix.

| Key | Value on the Q8_0 file | Meaning |
|---|---|---|
| `block_count` | 16 | layers |
| `embedding_length` | 2048 | `d_model` |
| `feed_forward_length` | 6144 | gated SiLU feed-forward |
| `attention.head_count` | 24 | query heads |
| `attention.head_count_kv` | 8 | KV heads (GQA 3:1) |
| `attention.key_length` / `value_length` | 128 | head width |
| `attention.layer_norm_rms_epsilon` | 1e-5 | RMSNorm epsilon |
| `attention.causal` | false | the encoder marker |
| `attention.temperature_scale` | 0.1 | the per-row attention temperature |
| `rope.dimension_count` | 128 | the whole head is rotated |
| `rope.freq_base` | 1000000 | |
| `rope.scaling.type` | yarn | |
| `rope.scaling.factor` | 16 | |
| `rope.scaling.original_context_length` | 16384 | YaRN's original context **and** the temperature's divisor |
| `rope.scaling.yarn_beta_fast` / `yarn_beta_slow` | 32 / 1 | the ramp's ends |
| `rope.scaling.yarn_log_multiplier` | 1 | |
| `pooling_type` | 1 | mean |
| `vocab_size` | 131072 | Tekken tokenizer, pre-tokenizer `pixtral` |
| `context_length` | 262144 | **not** the model's limit: it is 32768 tokens |

`f_attention_scale` is 0 for this architecture, so llama.cpp's `kq_scale` is
`1/sqrt(head_dim)` (`src/models/mistral3.cpp:163`). `f_attn_temp_offset` is 0 and the temperature
uses the `log` branch, not `pow` (`src/models/mistral3.cpp:13-21`: `n_attn_temp_floor_scale =
n_ctx_orig_yarn`, and it refuses a zero).

The query width is `24 × 128 = 3072`, wider than the 2048-wide residual; keys and values are
`8 × 128 = 1024` wide. Like Qwen 3, the head width is independent of the model width.

## Tensors (146 total)

| Tensor | Shape | Notes |
|---|---|---|
| `token_embd.weight` | `[2048, 131072]` | no separate output head |
| `output_norm.weight` | `[2048]` | `result_norm` |
| `blk.N.attn_norm.weight` | `[2048]` | pre-norm |
| `blk.N.attn_q.weight` | `[2048, 3072]` | no bias |
| `blk.N.attn_k.weight` / `attn_v.weight` | `[2048, 1024]` | no bias |
| `blk.N.attn_output.weight` | `[3072, 2048]` | no bias |
| `blk.N.ffn_norm.weight` | `[2048]` | |
| `blk.N.ffn_gate.weight` / `ffn_up.weight` | `[2048, 6144]` | SwiGLU |
| `blk.N.ffn_down.weight` | `[6144, 2048]` | |

`llama_model_mistral3::load_arch_tensors` (`src/models/mistral3.cpp:31-127`) marks `output` as
`TENSOR_NOT_REQUIRED` and falls back to the token embedding; it also accepts optional attention and
feed-forward biases and `ffn_norm_ada_*` tensors for the audio/chat variants. The published embedder
has none of them, and Peregrine refuses a file that carries any rather than running it without.

## The forward, in order

Per layer (`llama_model_mistral3::graph`, `src/models/mistral3.cpp:130-290`):

1. `x = tok_embd[token]` — no `sqrt(n_embd)` scale (`build_inp_embd`, `src/llama-graph.cpp:2669`).
2. `attn_norm` RMSNorm.
3. Q/K/V projections, no biases.
4. `ggml_rope_ext` on Q and on K with the file's YaRN parameters (`src/llama-context.cpp:280-317`,
   `ggml_rope_yarn_corr_dims` in `ggml/src/ggml.c:4534`). The rope pairing is NORM (adjacent
   pairs), not NeoX (`llama_model_rope_type`, `src/llama-model.cpp:3667`).
5. When `attention.temperature_scale != 0`, **Q is multiplied per row** by
   `ln(floor((pos + 0) / n_ctx_orig) + 1) · 0.1 + 1` (`llm_graph_input_attn_temp::set_input`,
   `src/llama-graph.cpp:298-323`, the non-`pow` branch). This is exactly 1.0 for every position
   below 16384 and about 1.0693 from 16384 to 32767.
6. Attention with `kq_scale = 1/sqrt(head_dim)`; `attention.causal = false` makes the mask
   non-causal, so every row sees every position of its own text.
7. `attn_output`, residual add.
8. `ffn_norm`, gate and up, `LLM_FFN_SILU` with `LLM_FFN_PAR` (`silu(gate) * up`), down, residual
   add.

After the last layer: `output_norm` → `result_norm`. There are no logits and no LM head for the
embedder.

### Pooling

`llama_model::build_graph` calls `build_pooling` with the file's `pooling_type`; `MEAN` is
`transpose(result_norm) · inp_mean`, the mean over the text's own tokens. The embedding example then
normalises with `common_embd_normalize(..., 2)` (sum of squares in f64, one f32 rounding of the
factor). Truncation to `dimensions` happens before normalisation, as Matryoshka requires.

## What Peregrine does

- `Dense::forward_texts_embedding` is the same schedule as `gemma-embedding`: every row of every
  text goes through a layer's projections in the same weight passes (the shared GEMM, in
  matmul-sized chunks), Q/K/V stay f32 for the whole pass, and attention is one `n × n` block per
  text with no cache. A layer's activation is quantised once per distinct input — q/k/v read the
  same normed rows, gate/up the same, down reads the SwiGLU output.
- The attention temperature is applied as written per query row, with
  `peregrine_core::vmath::det_ln` so the scalar is the same on every platform. Multiplying by 1.0
  is skipped by the value being exactly 1.0, not by a branch on the position.
- YaRN uses `blocks::rope::yarn_params` and `build_rope_table_yarn`, the implementation gpt-oss
  already exercises (including `yarn_log_multiplier = 1`).
- `MISTRAL3_MAX_TEXT = 32768` caps `cfg.ctx_train`, so the pass scratch (about 96 KiB per row on
  this geometry: the residual, the two attention buffers, q/k/v, two feed-forward buffers and the
  hidden rows) is not sized for 262144 tokens. A text longer than the served context is refused by
  the engine's own "does not fit in N positions" error.

  Measured on this host (Q8_0, 4 threads, a shared machine under load). The scratch is allocated
  for the engine's context, but only the rows a pass uses are touched, so resident memory follows
  the texts, not `-c`: RSS after load was 1.45 GiB at `-c 4096`, 1.61 GiB at 8192, 1.93 GiB at
  16384 and 2.57 GiB at 32768 (the model is 1.22 GiB). An 8192-token text ran to a vector (452 s,
  18 tokens/s under that load); a 16384-token text did not finish inside a 9-minute command at the
  same load, so it was not measured, and 32768 was not attempted. A 33000-token input at
  `-c 32768` is refused before any compute:
  `input 0 is 33000 tokens; this model is served with 32768 positions`.
- Pooling is the text's own rows in position order, f64 accumulator, one rounding
  (`embed::pool_rows_into`), then the caller's `embed::finish` truncates and normalises.

## Accuracy

Against llama.cpp v0.9.0 on the same files, 64 texts of 7–1520 tokens, cosine of the normalised
vectors (`crates/peregrine-models/tests/fixtures/nemotron/README.md` has the full table):

| Width | Peregrine F16 vs llama.cpp F16 (min / mean) | Peregrine Q8_0 vs F16 (min / mean) | llama.cpp Q8_0 vs F16 (min / mean) |
|---|---|---|---|
| 2048 | 0.9999998 / 0.99999994 | 0.999775 / 0.9998594 | 0.999762 / 0.9998549 |
| 1024 | 0.9999996 / 0.99999992 | 0.999686 / 0.9998428 | 0.999680 / 0.9998384 |
| 512 | 0.9999992 / 0.99999989 | 0.999519 / 0.9998107 | 0.999507 / 0.9998062 |

Peregrine's and llama.cpp's Q8_0 vectors agree to 0.999944 (min) / 0.9999777 (mean) at 2048 d.
Q4_K_M, measured once with llama.cpp's Q4_K_M vectors (not committed): Peregrine 0.9866695 /
0.99241421 against the F16 reference, llama.cpp 0.9870541 / 0.99242838, and the two engines against
each other 0.9996871 / 0.99986461.
