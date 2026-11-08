# Mixture of experts in Peregrine (S2.5, design note)

Design only, no code. Scope: `qwen3moe` first, `gpt-oss` (MXFP4) later. References: `origin/main`
(7a0e182), branch `s1.2-4-families` (PR #39), branch `adr-0006-architecture` (PR #18), llama.cpp
v0.9.0 (`cd959ba`).

## 1. What a qwen3moe file contains, and how llama.cpp routes

`general.architecture = "qwen3moe"`; keys carry the `qwen3moe.` prefix: `expert_count` (128),
`expert_used_count` (8), `expert_feed_forward_length` (768) (`src/models/qwen3moe.cpp:4`,
`src/llama-model.cpp:1485-1487`). The file also carries the *dense* `feed_forward_length` (6144),
unused by the experts (`src/models/qwen3moe.cpp:50`).

Per layer `i` (`src/llama-arch.cpp:527,535-537`, `src/models/qwen3moe.cpp:40,52-54`), GGUF dims
fastest-varying first:

| tensor | dims | shape |
|---|---|---|
| `blk.i.ffn_gate_inp.weight` | `[2048, 128]` | the router: `d_model × n_expert` |
| `blk.i.ffn_gate_exps.weight` | `[2048, 768, 128]` | per expert: `d_model → n_ff_exp` |
| `blk.i.ffn_up_exps.weight` | `[2048, 768, 128]` | per expert: `d_model → n_ff_exp` |
| `blk.i.ffn_down_exps.weight` | `[768, 2048, 128]` | per expert: `n_ff_exp → d_model` |

The rest of the layer is ordinary Qwen3 (per-head q/k RMSNorm, GQA), which the dense path covers
(`crates/peregrine-core/src/model.rs:700-718`).

The expert tensors are 3D, but GGUF stores `dims[0]` contiguous, so the layout is expert-major:
expert `e` is a contiguous run of `dims[1]` rows of `dims[0]` weights — exactly a 2D `[in, out]`
matrix in the layout the matmul group wants (`crates/peregrine-core/src/blocks/linear.rs:83-96`).
**No new kernel is needed** (§3).

### Routing math, as llama.cpp computes it

Per row, with `x` the `ffn_norm` output:

1. `logits = ffn_gate_inp · x` → 128 values, no bias (`src/llama-graph.cpp:2261`).
2. `probs = softmax(logits)` over **all 128** experts, f32 (`llama-graph.cpp:2279`; ggml's softmax
   is max-subtract, exp, sum, divide — `ggml/src/ggml-cpu/ops.cpp:5797+`).
3. `selected = argsort_top_k(probs, 8)`, descending probability (`llama-graph.cpp:2345`;
   `ggml/src/ggml.c:5501-5517`: an argsort DESC plus a view of the first *k*); `w_i =
   probs[selected_i]` in that order.
4. Renormalise: `s = Σ_i w_i` in slot order, `s = max(s, 6.103515625e-5)` (the smallest f16
   normal, `llama-graph.cpp:2381`), `w_i /= s`. `norm_w = true` for qwen3moe
   (`src/models/qwen3moe.cpp:144`) — the HF card's `norm_topk_prob: true`. `expert_weights_scale`
   is 0 here (`llama-graph.cpp:2390`).
5. Per selected expert `y_e = down_e(silu(gate_e(x)) ⊙ up_e(x))`; the layer's output is
   `Σ_{i=0..7} w_i · y_{e_i}`, summed **in slot order** (`llama-graph.cpp:2434-2490,2560,2586`,
   the sum at `2601-2614`).

**Ties.** ggml's argsort compares values only (`ggml/src/ggml-cpu/ops.cpp:8697-8746`) and
`std::sort` is not stable, so llama.cpp's choice among equal router scores is implementation
defined. Peregrine must fix a rule: proposal **probability descending, expert index ascending**,
per row, so it cannot depend on the batch.

## 2. Where it goes in ADR 0006

- **A new family**, `families/qwen3moe.rs`, registered in `FAMILIES`
  (`crates/peregrine-core/src/arch.rs:124-127` on `s1.2-4-families`), behind the `arch-moe`
  feature ADR 0005 already names (`docs/adr/0005-modular-structure.md:53`). Not a dense-family
  option: the loader, the config and the scratch all differ.
- **A new shared block** `crates/peregrine-core/src/blocks/moe.rs` (`blocks/mod.rs:6-14`): the
  routing math and the expert loop, as `blocks/ffn.rs` holds the dense SwiGLU.
- **No new `Architecture` methods.** The trait already has everything: `forward_rows` (the
  family's forward), `new_scratch` (the MoE scratch — the qwen35 family's pattern,
  `families/qwen35/mod.rs:59-61`), `act_layout` (the default int8 layout is right: the expert
  weights are k-quants, so the activations use the per-256 q8 layout), `recurrent`/`mtp` (None),
  `linears_mut`/`linear_groups` (empty: the LittleBit sidecar must not touch the expert
  tensors), `session_kv_rotation` (true).
- **Config.** ADR 0006 step 6 takes the architecture out of `Config`, so the MoE geometry
  (`n_expert`, `n_expert_used`, `n_ff_exp`) lives in the family (`model.rs:86-123`).

## 3. The compute plan for a batch

### Decode (one row)

The router is one `matmul_group` job (`ffn_gate_inp`: 128 rows of 2048 weights, 147 KB at Q4_K);
softmax, top-k and renormalisation are 128+8 scalars per row. Then, for each of the 8 selected
experts, three matrices are read whole, once: `gate_exps[e]`, `up_exps[e]` (768×2048) and
`down_exps[e]` (2048×768) — 4.72 M weights per expert, 37.7 M per layer, **1.81 G per token over
48 layers** ≈1.0 GB at Q4_K: decode is memory-bound on the selected experts.

### Prefill and batched decode

Rows are grouped by expert: `R_e` = the rows that selected expert `e`. Expert `e`'s three matmuls
run over `R_e` as one batch, so its weights are read once for all of them. Implementation:
**gather → `matmul_group` → scatter**: `R_e`'s rows are copied into a contiguous buffer,
`matmul_group` runs unchanged with `nb = |R_e|`, the results are scattered back. The copies are
exact; teaching `matmul_group` a row list instead is the S3.5 GEMM work.

The expert outputs are summed **in slot order**: each slot's result goes to a per-slot buffer
`[slot][row][d_model]`, and a second pass adds slots 0..7 in order, as llama.cpp does.
Accumulating "as the workers finish" would make a row depend on the batch.

### Invariant 1

A row's result must not depend on which other rows share the batch. Four properties make that
true, and each needs a test:

1. **The router is per row**: the softmax reduction runs over the experts in index order and the
   top-k is a per-row selection with a fixed tie rule; neither sees the batch.
2. **The matmul is per activation.** `quant::dot_batch`
   (`crates/peregrine-kernels/src/quant.rs:1015`) gives each activation its own accumulator
   (`crates/peregrine-kernels/src/simd.rs:600-627`), and the tiled (`tile_q4_k`,
   `simd.rs:593-645`) and wide (`wide_q4_k`, `simd.rs:914-964`) kernels accumulate a given
   activation in the *same* order (integer within a super-block, one float fold per super-block).
   A row's dot product is therefore bit-identical whether its expert runs for 1 row or 40.
   **Re-verify per format**, by test.
3. **SwiGLU and the norms are elementwise** (`vmath::swiglu` computes eight lanes at a time with
   the same per-element arithmetic), so splitting rows across threads cannot change a value.
4. **The expert sum runs in slot order**, and the gather/scatter copies are exact.

Only *which* kernel the table picks changes with the batch (tiled up to four activations, wide
above — `simd.rs:1129-1157`); because the two agree bit for bit, that is safe, and the
batch-invariance test proves it.

### Kernel table

`MatKernels::path` (`crates/peregrine-kernels/src/table.rs:78-93`) takes (format, tiled,
activation layout). For the experts: the file's format, not tiled, int8 activations → `PerRow` →
`quant::dot_batch` → the existing per-format kernels. **Nothing to add to the table** for qwen3moe.

The 3D tensor is used through a **2D view**: expert `e` is
`TensorInfo { dims: [d_in, d_out], kind, offset: t.offset + e·expert_bytes }`
(`crates/peregrine-core/src/gguf.rs:34-62`; `gguf.rs:262-265` reads by offset), so `Gguf::bytes`
returns exactly that expert's bytes and `row_bytes(kind, d_in)` (`quant.rs:776`) is unchanged.
The loader must check `d_in % block == 0` (256 for the k-quants, 32 otherwise) and refuse a file
that fails.

## 4. Telemetry

`StepStats.bytes_weights` (`model.rs:127-179`) counts the weight bytes read; the MoE block adds
the experts it actually read (per expert `rows × row_bytes(kind, d_in)`), never the whole 3D
tensor. `macs` gets `Σ_e |R_e| · 3 · n_ff_exp · d_model`. `TokenTelem`
(`crates/peregrine-core/src/telem.rs:19-60`) attributes a step's bytes to each token as
`bytes/n_tokens` (`session.rs:1148`), so a ten-user step reports each user its honest share. The
number to publish: **bytes per token** for Qwen3-30B-A3B Q4_K_M ≈1.7 GB (1.0 GB experts, 0.5 GB
attention, 0.18 GB head) against 17 GB of file.

## 5. Formats

Common qwen3moe GGUFs (Qwen/Qwen3-30B-A3B-GGUF, unsloth/Qwen3-30B-A3B-GGUF): Q4_K_M 18.6 GB,
IQ4_XS 16.4, IQ4_NL 17.3, Q4_0 17.4, Q5_K_M 21.7, Q6_K 25.1, Q8_0 32.5, BF16 61.1, Q2_K 11.3,
UD-IQ1_S 9.0.

Peregrine has kernels for F16/BF16/Q5_0/Q8_0/Q4_K/Q6_K (plus Q4_0, Q5_K, IQ4_NL, IQ4_XS), so
**every qwen3moe file in a format we support is covered with no new kernel**. Missing: MXFP4
(gpt-oss, the second half of S2.5) needs a new `Format`, a dequantiser (e2m1 codes, an e8m0 scale
per 32-element group) and a dot kernel; gpt-oss also routes differently (32 experts, top-4,
renormalised). IQ1_S/IQ1_M, the smallest files, are not supported.

## 6. Step plan

| PR | What | Tests |
|---|---|---|
| 1 (0.3 pw) | `blocks/moe.rs`: the routing math as pure functions — softmax, top-k with the tie rule, renormalisation | a scalar reference beside it; the tie rule tested (equal scores → the lower index wins) |
| 2 (0.3 pw) | the loader: `families/qwen3moe.rs` reads the metadata, builds the per-expert 2D views, refuses a bad `d_in % block` or a missing `expert_feed_forward_length` | `tests/support/models.rs`: `write_moe(dir, fmt, seed)` writes a tiny model (d_model 256, 2 layers, 8 experts, top-2, n_ff_exp 128); a load test asserts the shapes |
| 3 (0.5 pw) | the forward: the expert loop (decode first) and the family forward on the shared blocks | golden logits (`tests/golden/synthetic/<tier>/`), bit-identical across runs and across 1/2/4 threads |
| 4 (0.3 pw) | the batched path: grouping by expert, gather/scatter, the slot-order sum | the **batch-invariance test** (a prompt alone and beside nine others gives identical tokens, as `tests/engine.rs` does for the dense models); one row's expert output bit-identical for `|R_e| = 1` and `40` |
| 5 (0.2 pw) | telemetry, the support table (S2.7), the docs | the byte count matches the hand-computed one |
| 6 (0.3 pw) | the real-file check against llama.cpp v0.9.0 on the same file (greedy tokens, the chat path) | skips when the file is absent (`docs/dev/testing.md:27-29`) |

## 7. Open questions

1. **Tie-breaking.** llama.cpp's choice among equal router scores is unspecified; proposal:
   (probability descending, expert index ascending). Accept a divergence on an exact tie, or
   match llama.cpp?
2. **The real-file check.** No qwen3moe GGUF in a format we support fits a 13 GB host: the
   smallest is IQ4_XS at 16.4 GB (the smallest file overall is unsloth's UD-IQ1_S at 9.0 GB, a
   format we lack). Options: (a) run it on H1 as a release gate; (b) pull Q2_K (11.3 GB) into
   S2.1 so a 13 GB host can run one; (c) list the architecture without the check, contradicting
   `docs/targets.md:73`.
3. **Family or dense option?** The design says a new family plus `arch-moe`; confirm before PR 2,
   it decides where the code lands.
4. **Scratch budget.** The slot-ordered sum needs `nb × k × d_model` floats (4 MB at nb = 64,
   k = 8, d_model = 2048) plus the per-expert FFN buffers (`nb × n_ff_exp` × 2). Accept, or
   shrink it?
5. **Two metadata edges.** `expert_weights_scale` is 0 for qwen3moe — refuse such a file, or
   implement the scale? And a file carries both the dense `feed_forward_length` and
   `expert_feed_forward_length`: confirm we use only the latter and refuse a file without it.
6. **gpt-oss in S2.5 or later?** MXFP4 is a new format and kernel; recommend a separate PR after
   qwen3moe.

## 8. What gpt-oss turned out to need (S2.5 step 5)

The design above is `qwen3moe`'s. gpt-oss is the second half of S2.5, and it shares the
expert bank but almost nothing else, so it is its own family (`crates/peregrine-models/src/gptoss/`,
behind `arch-gpt-oss`) over the same `blocks::moe`. What differs, and what each difference
cost:

| what | qwen3moe | gpt-oss | where |
|---|---|---|---|
| the router's weights | a softmax over **all** the experts, then the best `k`, then a renormalisation over the selection | the best `k` by score, then a softmax over **those** | `src/llama-graph.cpp:2285-2287, 2363-2367` |
| the experts' activation | `silu(gate) ⊙ up` | `min(gate, 7) / (1 + e^(1.702·−min(gate, 7))) ⊙ (clamp(up) + 1)` | `src/llama-graph.cpp:2519-2526`, `ggml/src/ggml-cpu/ops.cpp:3388-3450` |
| biases | none | on the router and on every expert projection | `src/models/openai-moe.cpp:49-54` |
| the attention | Qwen3's | with **sinks** (a learned logit per head in the softmax), with an output bias, and with the sliding window on every other layer | `src/models/openai-moe.cpp:6-9, 42, 49` |
| the rope | plain or linear | YaRN | `src/llama-context.cpp:276-317`, `ggml/src/ggml-cpu/ops.cpp:6164-6187` |

The MXFP4 format the roadmap expected is already in the kernels (S2.1), so no new kernel
was needed: the experts are read as the same 2D views as `qwen3moe`'s, in MXFP4.

The gaps left for later: the real-file check (a `gpt-oss-20b` file, on a machine that has
one) and the AVX-512 goldens. Harmony's `analysis` and `final` channels are split into
`reasoning_content` and `content` by the server's `harmony.rs` (the model file's own Jinja
template is rendered exactly; llama.cpp's built-in one is only a small rendering, but the
file's template is the one that matters).
