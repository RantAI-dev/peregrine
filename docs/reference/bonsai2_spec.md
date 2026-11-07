# Bonsai 2 (PrismML ternary Qwen3.5, GGUF arch `qwen35`): implementation spec

Reference: PrismML's llama.cpp fork, branch `prism`, commit `62c418ccd7aefee37997e94c2d9fac1eed7fc89b`.
All file:line citations are relative to that repo. All shapes use **ggml order `[ne0, ne1]`**: ne0 is the contiguous (input) dimension. A weight `[n_in, n_out]` therefore has `n_out` rows of `n_in` elements, and `y = W·x` means `y[r] = dot(row r, x)`.

Fixture: `benchmarks/bonsai2_9b_reference.json` (see §8).

---

## 0. TL;DR: the things most likely to become silent bugs

1. **Every ternary matmul except `ssm_alpha`/`ssm_beta` takes a Hadamard-rotated input.** Its input is `x' = H_blk · (signs ⊙ x)`, and the GGUF lists exactly which weights get this. The **token embedding table is stored rotated**: after the row lookup you must apply `e = signs ⊙ (H_blk · row)` (the inverse transform, in the opposite order). `output.weight` is a separate tensor and is also folded. Block size is 4096 for 9B and **1024 for 27B**. The rotation is block-diagonal over the input dimension, and there is **one sign vector per input width**.
2. **`ssm_out`'s input must be permuted from tiled to grouped v-head order *before* the signs and the Hadamard** (`prism.hadamard.gdn_v_grouped = true`). Nothing else gets that permutation.
3. **GDN k-heads are shared in tiled order**: v-head `h` uses k-head `h % n_k_heads`, *not* `h / rep`. The converter reordered the V heads to make this work. The full-attention GQA layers are the opposite: q-head `h` uses kv-head `h / (n_head/n_head_kv)`.
4. RMSNorm weights already contain the Qwen3.5 `(1+w)`, so use them as plain `x̂ * w`. The one exception is `ssm_norm` (the gated norm), which was **never** offset; it is also plain `x̂ * w`. Either way you add nothing.
5. `ssm_a` already holds `-exp(A_log)` (it is negative). Decay = `exp(ssm_a[h] * softplus(alpha + dt_bias[h]))`.
6. Full attention: `attn_q` outputs **512 values per head: `[q(256) | gate(256)]` interleaved per head**, not `[all q | all gates]`.
7. RoPE: IMROPE with sections [11,11,10,0]. For text, all three position components are equal, so this reduces **exactly** to plain **NEOX** RoPE over the **first 64 of 256** head dims, with theta = 1e7. Dims 64..255 pass through unchanged.
8. q/k of the GDN layer go through an **L2 norm** (`x / max(‖x‖, 1e-6)`), and q is then scaled by `1/sqrt(128)`.
9. PQ2_0 keeps its fp16 scale **at the start** of the 34-byte block. PTQ1_0 keeps it **at the end** of the 28-byte block, and the PTQ1_0 element order is non-trivial (§5).
10. No BOS token is added. Raw prompt "The capital of France is" tokenizes to `[760, 6511, 314, 9338, 369]`.

---

## 1. GGUF metadata

The GGUF parser was pure Python because numpy is not installed. Key dump for `models/bonsai2-9b/Ternary-Bonsai-2-9B-PQ2_0.gguf` (GGUF v3, 427 tensors):

| key | 9B value | 27B (PTQ1_0) | read by |
|---|---|---|---|
| general.architecture | `qwen35` | same | |
| general.file_type | 141 (PQ2_0 file), 143 (PTQ1_0 file) | 143 | |
| qwen35.block_count | 32 | 64 | n_layer |
| qwen35.context_length | 262144 | same | |
| qwen35.embedding_length | 4096 | 5120 | n_embd |
| qwen35.feed_forward_length | 12288 | 17408 | n_ff |
| qwen35.attention.head_count | 16 | 24 | n_head (full-attn layers) |
| qwen35.attention.head_count_kv | 4 | 4 | n_head_kv |
| qwen35.attention.key_length | 256 | 256 | head_dim k |
| qwen35.attention.value_length | 256 | 256 | head_dim v |
| qwen35.attention.layer_norm_rms_epsilon | 1e-6 (f32 9.99999997e-07) | same | **every** RMSNorm, the L2 norm eps |
| qwen35.rope.dimension_count | 64 | 64 | n_rot (partial rotary = 64/256) |
| qwen35.rope.freq_base | 1.0e7 | same | |
| qwen35.rope.dimension_sections | [11, 11, 10, 0] | same | IMROPE sections (sum 32 = n_rot/2) |
| qwen35.ssm.conv_kernel | 4 | 4 | d_conv |
| qwen35.ssm.state_size | 128 | 128 | head_k_dim = head_v_dim |
| qwen35.ssm.group_count | 16 | 16 | num_k_heads |
| qwen35.ssm.time_step_rank | 32 | 48 | num_v_heads |
| qwen35.ssm.inner_size | 4096 | 6144 | value_dim = 128·num_v_heads |
| qwen35.full_attention_interval | 4 | 4 | layer pattern |
| prism.hadamard.version | 1 | 1 | |
| prism.hadamard.block_size | **4096** | **1024** | |
| prism.hadamard.transform | `normalized-sylvester-walsh-hadamard` | same | |
| prism.hadamard.axis | `input-last-dimension` | same | |
| prism.hadamard.sign_mode | `explicit` | same | |
| prism.hadamard.sign_widths | [4096, 12288] | [5120, 6144, 17408] | |
| prism.hadamard.sign_values | int32[16384], all ±1 | int32[28672] | concatenation, in sign_widths order |
| prism.hadamard.weight_names | string[201] | string[401] | folded weights |
| prism.hadamard.inverse_weight_names | [`token_embd.weight`] | same | |
| prism.hadamard.gdn_v_grouped | true | true | |
| (prism.hadamard.tied_output) | absent → false | absent | |
| tokenizer.ggml.model | `gpt2` (byte-level BPE) | same | |
| tokenizer.ggml.pre | `qwen35` | same | |
| tokenizer.ggml.tokens / token_type | 248320 entries | same | |
| tokenizer.ggml.merges | 247587 | | |
| tokenizer.ggml.bos_token_id | 248044 (`<|endoftext|>`) | same | |
| tokenizer.ggml.eos_token_id | 248046 (`<|im_end|>`) | same | |
| tokenizer.ggml.padding_token_id | 248044 | same | |
| tokenizer.ggml.add_bos_token | **false** | false | |
| general.sampling.* | temp 1.0, top_p 0.95, top_k 20, min_p 0, penalty_repeat 1.0 | | defaults only |

The loader reads hparams in `src/models/qwen35.cpp:6-33` and Hadamard settings in `src/llama-model.cpp:2089-2254`. Where a key is absent the loader uses a default: `f_attention_scale` defaults to 0, which means `1/sqrt(head_dim)`, and rope scaling and YaRN are off (`freq_scale=1`, `ext_factor=0`, `attn_factor=1`).

**What the `prism.hadamard.*` keys mean** (`llama-model.cpp:2183-2184`):

> `// a folded weight W_f = W*D*H is the weight W with the +1/-1 signs D and the normalized block Hadamard H folded in`
> `// H*H = I and D*D = I, so W*x = W_f*(H*(D*x)): the graph applies D, then H, to the matmul input`

- `block_size`: the Hadamard size `B`. H_blk is block-diagonal with `n_in/B` copies of the B×B matrix `H[r][c] = (-1)^popcount(r & c) / sqrt(B)` (natural Sylvester order, symmetric; `llama-model.cpp:2342-2351`). The loader requires `n_in % B == 0`.
- `sign_widths` / `sign_values`: one ±1 vector per **input width**. The vector for a weight is looked up by `weight->ne[0]` (`llama-model.cpp:2364-2368`). All folded weights with the same input width share one sign vector. 9B: width 4096 covers everything except `ffn_down`, which uses 12288. 27B: 5120 (n_embd inputs), 6144 (`attn_output` and `ssm_out` inputs), 17408 (`ffn_down`).
- `weight_names`: the weights whose matmul input is transformed. 9B per layer:
  - linear-attention layers: `attn_qkv, attn_gate, ssm_out, ffn_gate, ffn_up, ffn_down`
  - full-attention layers: `attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down`
  - plus `output.weight`
  - **Not folded:** `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, and all norms. These take the *unrotated* input.
- `inverse_weight_names = [token_embd.weight]`: the embedding table rows are stored in the rotated ("latent") basis. After the lookup, apply `h = s ⊙ (H z)` (`llama-graph.cpp:2681-2689`).
- `gdn_v_grouped`: `ssm_out` was folded with its input in grouped v-head order, but the graph produces tiled order. The runtime permutes the activation (§4).
- `tied_output` (version 2) is absent here, so `output.weight` is its own folded tensor.

---

## 2. Tensors (9B PQ2_0 file; the PTQ1_0 file has the same tensors with type 143)

ggml type ids: **142 = PQ2_0**, **143 = PTQ1_0**, 0 = F32, 30 = BF16.

Global:

| name | shape [ne0, ne1] | type |
|---|---|---|
| token_embd.weight | [4096, 248320] | PQ2_0 (latent/rotated rows) |
| output_norm.weight | [4096] | F32 |
| output.weight | [4096, 248320] | PQ2_0 (folded, **not tied**) |

**Layer pattern:** `is_recr(i) = (i+1) % full_attention_interval != 0` (`qwen35.cpp:23`). Full attention is at layers **3, 7, 11, 15, 19, 23, 27, 31** (8 layers in 9B; 16 in 27B, at every i ≡ 3 mod 4). Every other layer is Gated DeltaNet (24 layers in 9B; 48 in 27B).

**Linear-attention (GDN) layer, e.g. blk.0:**

| name | shape | type | meaning |
|---|---|---|---|
| blk.0.attn_norm.weight | [4096] | F32 | input RMSNorm |
| blk.0.attn_qkv.weight | [4096, 8192] | PQ2_0 | in_proj_qkv: out = [q 2048 \| k 2048 \| v 4096] |
| blk.0.attn_gate.weight | [4096, 4096] | PQ2_0 | in_proj_z (gate z) |
| blk.0.ssm_beta.weight | [4096, 32] | F32 (27B: BF16) | in_proj_b |
| blk.0.ssm_alpha.weight | [4096, 32] | F32 (27B: BF16) | in_proj_a |
| blk.0.ssm_conv1d.weight | [4, 8192] | F32 | depthwise conv, ne0 = tap (no bias) |
| blk.0.ssm_dt.bias | [32] | F32 | dt_bias |
| blk.0.ssm_a | [32] | F32 | = -exp(A_log) (all negative; 9B blk0 range −0.146..−0.009) |
| blk.0.ssm_norm.weight | [128] | F32 | gated RMSNorm weight (per head_v_dim, shared over heads) |
| blk.0.ssm_out.weight | [4096, 4096] | PQ2_0 | out_proj |
| blk.0.post_attention_norm.weight | [4096] | F32 | pre-FFN RMSNorm |
| blk.0.ffn_gate.weight | [4096, 12288] | PQ2_0 | |
| blk.0.ffn_up.weight | [4096, 12288] | PQ2_0 | |
| blk.0.ffn_down.weight | [12288, 4096] | PQ2_0 | |

**Full-attention layer, e.g. blk.3:**

| name | shape | type | meaning |
|---|---|---|---|
| blk.3.attn_norm.weight | [4096] | F32 | |
| blk.3.attn_q.weight | [4096, 8192] | PQ2_0 | 16 heads × (256 q + 256 gate), **per-head interleaved** |
| blk.3.attn_k.weight | [4096, 1024] | PQ2_0 | 4 kv heads × 256 |
| blk.3.attn_v.weight | [4096, 1024] | PQ2_0 | |
| blk.3.attn_q_norm.weight | [256] | F32 | |
| blk.3.attn_k_norm.weight | [256] | F32 | |
| blk.3.attn_output.weight | [4096, 4096] | PQ2_0 | input 16·256 |
| blk.3.post_attention_norm.weight | [4096] | F32 | |
| blk.3.ffn_gate / ffn_up / ffn_down | as above | PQ2_0 | |

27B for comparison: attn_qkv [5120, 10240], attn_gate [5120, 6144], ssm_out [6144, 5120], ssm_a/dt [48], conv1d [4, 10240], attn_q [5120, 12288], attn_k/v [5120, 1024], attn_output [6144, 5120], ffn [5120 ↔ 17408].

The `-MTP-Q8_0` 9B file adds a NextN/MTP block as an extra layer. The main forward pass ignores it (`qwen35.cpp:200`), so skip it.

Measured norm magnitudes, which confirm that the `+1` is already folded in: attn_norm mean 1.04, output_norm mean 2.14, q_norm mean 1.35, ssm_norm mean 0.86. Converter evidence: `conversion/qwen.py:411-412`:
```python
elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
    data_torch = data_torch + 1
```
and `qwen.py:405-406`: `if name.endswith(".A_log"): data_torch = -torch.exp(data_torch)`.

---

## 3. Forward pass

Notation: `D = n_embd` (4096). `rms(x, w) = x / sqrt(mean(x²) + 1e-6) * w`, with the mean over the whole vector (accumulated in double: `ops.cpp:4243`). `matmul_f(W, x)` means a **folded** matmul: `W · hadamard_in(W, x)` (§4). `matmul(W, x)` means a plain matmul.

### 3.0 Top level (`qwen35.cpp:190-311`)
```
for each token t:  x_t = embed(tok_t)                # §4: s ⊙ H_blk·dequant(token_embd row)
for il in 0..n_layer-1:
    h   = rms(x, attn_norm[il])                                  # qwen35.cpp:206
    a   = GDN(h) if is_recr(il) else FullAttn(h)                 # :212-218
    x   = x + a                                                  # :226
    h2  = rms(x, post_attention_norm[il])                        # :233
    f   = matmul_f(ffn_down, silu(matmul_f(ffn_gate,h2)) * matmul_f(ffn_up,h2))   # :621-628 (LLM_FFN_SILU, LLM_FFN_PAR)
    x   = x + f                                                  # :241
y      = rms(x, output_norm)                                     # :287
logits = matmul_f(output, y)                                     # :310 (output.weight is folded)
```
Prefill and decode share this graph. Only the per-layer mixers keep state across tokens. llama.cpp computes logits only for the rows it needs (the last token).

### 3.1 Full-attention layer (`qwen35.cpp:343-422`)
`n_head=16, n_head_kv=4, hd=256, n_rot=64`.
```
QG = matmul_f(attn_q, h)      # [8192] ; per head hh: QG[hh*512 + 0..255] = q, QG[hh*512 + 256..511] = gate
K  = matmul_f(attn_k, h)      # [1024] = 4 heads × 256
V  = matmul_f(attn_v, h)      # [1024]
```
- Q is a view with head stride 512 and offset 0 (`:364-366`). The gate is the same view at offset 256 (`:378-381`), made contiguous to `[4096]`, giving `gate[hh*256 + d] = QG[hh*512 + 256 + d]`.
- `q_hh = rms(q_hh, attn_q_norm)` per head over 256 dims (`:370`). `k_g = rms(k_g, attn_k_norm)` (`:375`). **The gate is not normed.**
- RoPE is `ggml_rope_multi(..., n_rot=64, sections, IMROPE, freq_base=1e7, freq_scale=1)` on q and k (`:388-396`). V gets no RoPE.
  - **Text-only reduction (exact):** text positions are passed as `[p, p, p, 0]` (`llama-graph.cpp:229`). In `ggml_mrope_cache_init` (`ops.cpp:6338-6402`), every pair index `i = 0..31` (sector = i, since `sect_dims = 32`) maps to section t, h or w, never e. With `p_t = p_h = p_w = p`, every pair gets `theta_i = p · 1e7^(-2i/64)`, `i = 0..31` (`theta_scale = powf(freq_base, -2.0f/n_dims)`, `ops.cpp:6490`).
  - Pairing is NEOX (rotate-half) within the first 64 dims: `rotate_pairs(n_dims, n_dims/2, ...)` (`ops.cpp:6559`):
    `x0 = v[i], x1 = v[i+32]; v[i] = x0 cos − x1 sin; v[i+32] = x0 sin + x1 cos`, for i in 0..31.
  - Dims 64..255 are copied through unchanged. Position p = absolute token index starting at 0 (no BOS).
- Attention: scale `1/sqrt(256)` (`:405`), causal, softmax over all cached positions. **GQA is grouped:** q head `hh` uses kv head `hh / 4`, which is the ggml mul_mat / flash-attn broadcast `i02 = i12 / r2`.
  - The KV cache stores K after norm and RoPE, and V raw.
  - The reference stores K/V as **F16** (default cache type) and uses flash-attn on CPU. That is a small source of rounding; f32 is fine for you.
- `o = attn_out (16×256 = [4096]) * sigmoid(gate)`, elementwise in the same layout (`:412-415`).
- `out = matmul_f(attn_output, o)` (`:418`). The Hadamard input is `o`, width 4096.

### 3.2 Gated DeltaNet layer (`qwen35.cpp:424-617`, `delta-net-base.cpp`, CPU op `ops.cpp:11425-11597`)
9B: `Hk = 16` k-heads, `Hv = 32` v-heads, `dk = dv = 128`, `key_dim = 2048`, `value_dim = 4096`, `conv_dim = 8192`, `rep = Hv/Hk = 2`.
```
qkv  = matmul_f(attn_qkv,  h)          # [8192]  (:322)  layout [q: 16×128 | k: 16×128 | v: 32×128]
z    = matmul_f(attn_gate, h)          # [4096]  (:326)  32 heads × 128
b    = matmul  (ssm_beta,  h)          # [32]    (:447)  plain F32 matmul on the UNROTATED h
a    = matmul  (ssm_alpha, h)          # [32]    (:456)  plain, unrotated
beta = sigmoid(b)                                                   # :453 / ops.cpp:11544
g    = ssm_a[hv] * softplus(a[hv] + ssm_dt_bias[hv])                # :476-480 / ops.cpp:11545-11546
       # CPU raw-gate path: softplus(x) = x if x > 20 else log(1+exp(x)).  decay = exp(g) in (0,1]
```
**Causal conv1d, depthwise, kernel 4, no bias** (`build_conv_state` in `delta-net-base.cpp:451-525`; `ssm_conv` in `ops.cpp:10233-10284`):
```
conv_state: per layer per sequence, float [8192 channels][3 taps], oldest first. Zero at sequence start.
For channel c with window w = [state[c][0], state[c][1], state[c][2], qkv_t[c]]:
    y_t[c] = Σ_{i=0..3} w[i] * conv1d[c][i]   # conv1d memory: weight[c*4 + i]; i=3 multiplies the current token
    (multi-token: window slides over concat(state, x_0..x_{T-1}), i.e. the input for token t is cols t..t+3)
new conv_state[c] = last 3 inputs (raw qkv, pre-conv) of the concatenated sequence
y = silu(y)                                                          # qwen35.cpp:531
```
Split the conv output (`:541-557`): `q = y[0:2048]` (16×128), `k = y[2048:4096]` (16×128), `v = y[4096:8192]` (32×128, head-major, 128 contiguous per head).

L2 norm per 128-vector of q and of k (`:575`, `ops.cpp:4612`): `x = x * (1 / max(sqrt(Σx²), 1e-6))`. This is **not** the HF `rsqrt(Σx²+eps)` form; the source comments that the Prism contract keeps it this way.

**Head mapping (tiled):** v-head `hv` uses q/k head `hv % Hk` (`ops.cpp:11511-11512`: `iq1 = iv1 % neq1; ik1 = iv1 % nek1`). The unfused path uses `ggml_repeat_4d` (`:591`), which also tiles. The beta, alpha, dt_bias, ssm_a, z and v head index are all `hv` in the GGUF's (tiled) order, so you never need to undo the tiling, except for the `ssm_out` Hadamard input (§4).

**Delta-rule recurrence** per v-head `hv`, per token, sequentially. Prefill runs the same loop over tokens; the CPU fused op is the reference for both (`ops.cpp:11538-11583`):
```
S: float[128 (v index j)][128 (k index i)]   memory s[j*128 + i]   ("state is stored transposed: s_out[j*S_v + i] = S[i][j]")
qh = q[hv % 16], kh = k[hv % 16], vh = v[hv]
S      *= exp(g[hv])                                        # ops.cpp:11562
for j: delta[j] = (vh[j] - Σ_i s[j][i]*kh[i]) * beta[hv]    # :11566-11570
for j: s[j][i] += delta[j] * kh[i]   (all i)                # :11573-11575
for j: o[hv][j] = (Σ_i s[j][i]*qh[i]) / sqrt(128)           # :11578-11582  (scale = 1/sqrt(S_v))
```
That is, with `S` as v×k: `S ← e^g S; S ← S + β (v − S k) kᵀ; o = S q / √dk`.

State size per sequence per GDN layer: `32 × 128 × 128` f32 = 2 MiB (`n_embd_s = d_state·d_inner = 524288` floats). The 9B has 24 such layers, so 48 MiB per sequence. Conv state per layer = `3 × 8192` f32 (`n_embd_r = (d_conv−1)·(d_inner + 2·n_group·d_state) = 24576`). The state starts at zero.

**Output** (`:603-613`):
```
for each hv:  n = rms(o[hv] (128), ssm_norm)       # per-head RMSNorm, eps 1e-6, weight [128] shared
              out_h = silu(z[hv]) * n              # ggml_swiglu_split(gate=z, normalized) :340
final = concat_hv out_h  -> [4096] (tiled v-head order)
out   = matmul_f(ssm_out, final)                   # Hadamard input gets the tiled->grouped permutation, §4
```

### 3.3 Prefill vs decode
- The math is identical. Prefill runs the conv over `[state | x_0..x_{T-1}]` and the delta rule token by token (fused op, `cparams.fused_gdn_ch = true`, `llama-context.cpp:342`). Attention is causal within the batch plus over the cache.
- Snapshots/rollback (`n_rs_seq`) and replay are serving features. The default single-sequence path writes only the final state.

---

## 4. Hadamard rotation (activation side)

**Forward transform for a folded weight W** (`llama-graph.cpp:1684-1713` `build_hadamard_input`, `llama-impl.h:57-75` `llama_mul_mat_hadamard`):
```
def hadamard_in(W, x):              # x: f32 vector of length n_in = W.ne0
    if W is ssm_out and gdn_v_grouped:               # perm_hd=128, perm_nk=Hk=16, perm_rep=Hv/Hk=2 (27B: 3)
        # tiled index  (r*nk + g)*hd + d   ->  grouped index (g*rep + r)*hd + d
        y = empty(n_in)
        for r in range(rep): for g in range(nk): for d in range(hd):
            y[(g*rep + r)*hd + d] = x[(r*nk + g)*hd + d]
        x = y                                         # llama-graph.cpp:1699-1706 (reshape [hd,nk,rep], permute(0,2,1,3))
    x = x * signs[n_in]                               # :1708  sign vector chosen by input WIDTH
    for each consecutive block of B = block_size elements:
        x_blk = FWHT(x_blk) / sqrt(B)                 # :1710
    return x
```
- Applies to the input of **every** weight in `weight_names`: attn_qkv, attn_gate, attn_q, attn_k, attn_v, attn_output, ssm_out, ffn_gate, ffn_up, ffn_down, output. It does **not** apply to ssm_alpha or ssm_beta (F32/BF16).
- One transform per (input tensor, rotation) is computed and reused, e.g. q/k/v share one, and gate/up share one. This is safe because the sign vector depends only on the width.
- 9B ffn_down: n_in = 12288 → 3 independent 4096-blocks, with the 12288-long sign vector applied first. 27B: block 1024, so 5120 → 5 blocks, 6144 → 6, 17408 → 17.
- Activation quantization to Q8 (§5) happens **after** this transform.

**CPU FWHT** (`ops.cpp:12808-12880`, dispatched from `ggml-cpu.c:1309` when the mul_mat has `GGML_HINT_SRC0_IS_HADAMARD`). It scales first, then runs the standard in-place natural-order butterflies:
```c
for (j = 0; j < n; j++) dst_row[j] = src_row[j] * scale;          // scale = 1/sqrtf(n)   (:12843)
for (len = 1; len < n; len <<= 1)
  for (i = 0; i < n; i += 2*len)
    for (j = 0; j < len; j++) { u = dst[i+j]; v = dst[i+len+j]; dst[i+j] = u+v; dst[i+len+j] = u-v; }   (:12852-12862)
```
The result equals `H·x` with `H[r][c] = (−1)^popcount(r&c)/√n` (natural/Hadamard order, no bit-reversal), which is the matrix the loader builds. H is symmetric and orthonormal, so H = H⁻¹.

**Embedding (inverse) transform** (`llama-graph.cpp:2681-2689`, `2795-2803`): the order is reversed (H first, then signs):
```
row = dequant(token_embd row tok)        # 4096 floats (PQ2_0/PTQ1_0 dequant)
row = blockwise FWHT(row)/sqrt(B)
x0  = row * signs[n_embd]                # "h = s * (H z)"
```

**Sign vectors (9B):** `sign_values[0:4096]` is for width 4096 and `sign_values[4096:16384]` is for width 12288. The first 8 of the 4096-vector are `[-1,-1,-1,1,1,-1,-1,1]`. Read them from the GGUF. The KV array element type is INT32 (gguf type 5).

(The KV-cache "attn_rot" Hadamard in `llama-kv-cache.cpp` activates only for quantized KV cache types. It is off with the default f16 KV, so ignore it.)

### 4.1 A plain Qwen 3.5 file has no rotation at all

The rotation above is a PrismML/Bonsai 2 thing, not a Qwen 3.5 thing. Upstream's own `qwen35`
files — what `conversion/qwen.py` writes for `Qwen/Qwen3.5-9B` — carry none of the
`prism.hadamard.*` keys, and llama.cpp v0.9.0 reads them as "no rotation"
(`llama_model_base::load_hparams_hadamard`: the whole block is behind `prism.hadamard.version`,
which the plain files do not have). Their weights are used as they are, and so is their
embedding table.

Two more differences come with a plain file, and both are the converter's doing
(`conversion/qwen.py`, `_LinearAttentionVReorderBase`):

- The value heads are in the **tiled** order, not the grouped one: the converter folds the
  permutation into `attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`, `ssm_a`, `ssm_dt.bias`,
  `ssm_conv1d` and `ssm_out`. A Bonsai 2 file keeps the grouped order and says so with
  `prism.hadamard.gdn_v_grouped`, and the engine permutes the activation instead.
- The matmuls are ordinary quantised matrices (Q8_0, Q4_K, ...), not ternary ones, so the
  activation layout is the ordinary one (int8 codes with one scale per 32) rather than the
  ternary one.

The MRoPE sections (`qwen35.rope.dimension_sections`) are the one thing a plain file has and
Bonsai 2 files also carry: llama.cpp's `qwen35` loader requires them, and the converter writes
`[11, 11, 10, 0]` when a checkpoint omits them. For a text-only model they change nothing —
all four position streams are the same, so the interleaved MRoPE is the NeoX rope — but the
key has to be there.

---

## 5. PQ2_0 / PTQ1_0 formats and kernels

### PQ2_0 (type 142), 128 weights per block, 34 bytes (`ggml-common.h:199-207`)
```c
#define QK_PQ2_0 128
typedef struct {
    ggml_half d;               // offset 0: fp16 scale
    uint8_t qs[QK_PQ2_0 / 4];  // offset 2: 32 bytes, 2 bits/elem
} block_pq2_0;
```
Element `j` is `(qs[j/4] >> (2*(j%4))) & 3`, LSB-first within each byte, natural order. Codes map `00 → −1, 01 → 0, 10 → +1, 11 → +2`. The value is `(code − 1)·d`. Code 3 never appears in these models (checked on 60k sampled blocks): they are ternary. Reference dequant (`ggml-quants.c:494-512`):
```c
const float d = GGML_FP16_TO_FP32(x[i].d);
for (int j = 0; j < qk; ++j) {
    const uint8_t q = (x[i].qs[j/4] >> ((j % 4) * 2)) & 0x03;
    y[i*qk + j] = ((int)q - 1) * d;     // 00=-1, 01=0, 10=+1, 11=+2
}
```
The quantizer (`:113-145`) uses `d = amax` and `q = round(w/d)+1`, clamped to [0,3].

### PTQ1_0 (type 143), 128 weights per block, 28 bytes (`ggml-common.h:214-220`)
```c
typedef struct {
    uint8_t qs[24];  // offset 0:  5 trits/byte -> 120 values
    uint8_t qh[2];   // offset 24: 4 trits/byte -> 8 values
    ggml_half d;     // offset 26: fp16 scale  (AT THE END)
} block_ptq1_0;
```
Trit `n` (n = 0 is the most significant) of a byte `b` is `t = ((uint8_t)(b * 3^n) * 3) >> 8`, where the multiply wraps mod 256. The value is `(t − 1)·d`.

**Element order** (the stages are 32/16/8 bytes; with 24 qs bytes only 16 and 8 are used). From `ggml-quants.c:2256-2283`:
- elements `0..79`: element `16*n + m` = trit n of `qs[m]`, m = 0..15, n = 0..4
- elements `80..119`: element `80 + 8*n + m` = trit n of `qs[16+m]`, m = 0..7, n = 0..4
- elements `120..127`: element `120 + 2*n + h` = trit n of `qh[h]`, h = 0..1, n = 0..3

```c
const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
size_t j = 0;
for (size_t s = 0; s < 3; ++s) { const size_t c = ptq1_0_stages[s];   // {32,16,8}
    for (; j + c <= sizeof(x->qs); j += c)
        for (size_t n = 0; n < 5; ++n) for (size_t m = 0; m < c; ++m) {
            uint8_t q = x[i].qs[j + m] * pow3[n];
            int16_t xi = ((uint16_t) q * 3) >> 8;
            *y++ = (float) (xi - 1) * d; } }
for (size_t n = 0; n < 4; ++n) for (size_t h = 0; h < sizeof(x->qh); ++h) {
    uint8_t q = x[i].qh[h] * pow3[n];
    int16_t xi = ((uint16_t) q * 3) >> 8;
    *y++ = (float) (xi - 1) * d; }
```
The encoder (`:2206-2254`) packs `q = Σ trit·3^(4−n)` and stores `ceil(q·256/243)`. In `qh`, the 4-trit value is multiplied by 3 before scaling.

**The 9B PQ2_0 and PTQ1_0 files hold bit-identical dequantized weights**: 0 mismatches on 60600 sampled blocks across all 202 quantized tensors, and every F32 tensor is identical. So one test fixture serves both.

### Activation pairing on CPU (`ggml-cpu.c:241-256`)
- PQ2_0: `vec_dot = ggml_vec_dot_pq2_0_q8_K`, with `vec_dot_type = Q8_K`. The Q8_K block is 256 elements: `float d; int8 qs[256]; int16 bsums[16]`. Quantization uses `iscale = −127/max` (max = the signed value with the largest |x|), `qs = min(127, nearest_int(iscale·x))`, `d = 1/iscale` (`ggml-quants.c:2940-2977`). Because 128 < 256, one Q8_K block covers two PQ2_0 blocks: block `i` uses `y[i>>1].qs + 128*(i&1)`.
- **However, on AVX-512 VNNI CPUs (including the fixture machine)**, PQ2_0 weights are repacked (`repack.cpp:5460-5465`, `pq2_0_4x8_q8_0`, `tensor_traits<block_pq2_0, 8, 4, GGML_TYPE_Q8_0>`) and multiplied against **Q8_0** activations. Q8_0 is 32 elements with an fp16 `d = amax/127`, `qs = round(x/d)`.
- PTQ1_0: `vec_dot_ptq1_0_q8_0`, `vec_dot_type = Q8_0`.
- Generic dot, PQ2_0 × Q8_K (`quants.c:1468-1496`): `sumi += (code−1)·q8[...]` is exact in int32, then `sumf += d_w·d_a·sumi` once per 128-weight block.

**SIMD trick (all kernels):** keep the unsigned codes `c ∈ {0,1,2}` and use the u8×s8 VNNI `dpbusd`, computing `Σ(c−1)·y = dpbusd(c, y) − dpbusd(1, y)`. Then apply one float FMA per block with `d_w·d_a`.
- PTQ1_0 AVX-512 (`arch/x86/quants.c:648-703`):
  - Trits are decoded in-register: a per-16-bit-lane `mullo` by a power of 3 implements the mod-256 multiply (`ptq1_0_mul_u8`).
  - `trit = (v >= 86) + (v >= 171)` via mask compares (`ptq1_0_trit`). This equals `(3v)>>8`.
  - One shuffle arranges the qs/qh runs so that each 128-bit lane is one source run × one power of 3, so 64 weights decode per 512-bit vector.
  - `_mm512_dpbusd_epi32(t, y) − _mm512_dpbusd_epi32(ones, y)`.
- PQ2_0 AVX-512 GEMV (`arch/x86/repack.cpp:6439-6453`, `6852-6904`): 4 output rows are interleaved in 8-byte chunks. The 2-bit fields are spread to bytes with `cvtepu8_epi32` + shifts (by 0,6,12,18) + `and 3`. Then `dpbusd(codes, qa) − dpbusd(ones, qa)` and an FMA with `d_w·d_a` per 32-element Q8_0 sub-block.

You don't have to copy any of this. A correct f32 dequant-and-dot will match to within the kernel-to-kernel noise shown in §8.

---

## 6. Tokenizer and chat template

- `tokenizer.ggml.model = gpt2`: byte-level BPE with ranked merges, 248320 tokens (padding tokens at the end, `[PAD...]`, type 5).
- Pre-tokenizer `qwen35` (`src/llama-vocab.cpp:400-406`). The regex is:
  `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
  Note `\p{N}` matches a single digit, and letters include `\p{M}`. `clean_spaces = false`.
- BOS id 248044 `<|endoftext|>`, but `add_bos_token = false`, so **do not prepend it**. EOS id 248046 `<|im_end|>`. Also treat 248044 as end-of-generation. Other specials: 248045 `<|im_start|>`, 248068 `<think>`, 248069 `</think>`.
- Chat template (ChatML + thinking): `<|im_start|>{role}\n{content}<|im_end|>\n` for each turn. With `add_generation_prompt`, it appends `<|im_start|>assistant\n` and then:
  - `<think>\n` by default (thinking **on**), or
  - `<think>\n\n</think>\n\n` when `enable_thinking` is defined and false (thinking off).
  - Historical assistant turns after the last user query re-emit `<think>\n{reasoning}\n</think>\n\n{content}`. Earlier assistant turns drop the reasoning.
- Tools follow the Qwen3.5 XML format (`<tool_call><function=...><parameter=...>`).

---

## 7. Surprising or easy to get wrong (checklist)

1. **Embedding = inverse transform; matmul inputs = forward transform.** Inverse is `signs ⊙ H(row)`, i.e. H then signs. Forward is `H(signs ⊙ x)`, i.e. signs then H. If you skip either, the output is garbage, but plausible-looking garbage.
2. **The Hadamard block size is not n_embd in general** (27B: 1024, while n_embd is 5120). ffn_down in the 9B has 3 blocks. The sign vector spans the full width.
3. **ssm_alpha and ssm_beta read the un-rotated `h`.** They are F32 in the 9B file, BF16 in the 27B file.
4. **ssm_out tiled→grouped permutation** before signs and H (rep = 2 in 9B, 3 in 27B). Only `ssm_out` gets it. `attn_output` in the full-attention layers does not.
5. **GDN k-head sharing is `hv % Hk` (tiled).** Full-attention GQA sharing is `h / 4` (grouped). Don't mix them up.
6. **The q/gate split in attn_q is interleaved per head**: `[q0(256) g0(256) q1 g1 ...]`. Q gets q_norm and RoPE. The gate gets only sigmoid, applied after attention.
7. **RoPE covers 64 of 256 dims, NEOX pairs (i, i+32), theta 1e7.** IMROPE reduces to this for text. Don't apply the "normal" adjacent-pair RoPE.
8. **All RMSNorm weights are used as-is** (the +1 is pre-folded; ssm_norm never had one). The eps is 1e-6 everywhere and goes **inside** the sqrt: `1/sqrt(mean+eps)`.
9. **The L2 norm eps sits outside the sqrt as a max**: `1/max(‖x‖, 1e-6)`. q is additionally scaled by `1/sqrt(128)`: the CPU op scales the output, which is equivalent.
10. **Decay** = `exp(ssm_a · softplus(alpha + dt_bias))`, with `ssm_a < 0` already. dt_bias values reach 18.5, so use the softplus threshold `x > 20 → x` for parity.
11. **The order inside the delta rule matters**: decay S **first**, then compute `S·k` with the decayed S, then update, then read out with the **updated** S.
12. **Conv state holds raw (pre-conv, pre-SiLU) qkv inputs.** The conv has no bias. Tap index 3 multiplies the current token. SiLU comes after the conv, over all 8192 channels (q, k and v).
13. **The gated norm is `rms(o)*w * silu(z)`**: norm first, SiLU on z, multiply. It is per head with a 128-wide shared weight.
14. The 9B `output.weight` is **not** tied to token_embd. Both are PQ2_0. Both are in the Hadamard lists, but in opposite directions.
15. PTQ1_0's scale sits at byte 26 (the end). PQ2_0's scale sits at byte 0.
16. Q8_K's `d` can be negative (`iscale = −127/max_signed`). That is harmless if you mirror it; it doesn't matter if you use your own activation quantizer.
17. The full-attention layer's attention output width is `n_head·256` (4096 in 9B, **6144 in 27B ≠ n_embd**). Likewise the GDN value_dim is 6144 in 27B.
18. There is no BOS. Positions start at 0 on the first prompt token.

---

## 8. Reference fixture (`bonsai9b_ref.json`)

How it was produced: `llama-server -m Ternary-Bonsai-2-9B-PQ2_0.gguf --port 8398 -t 8 -c 2048 -np 1 -ngl 0` (CPU, AMD Ryzen AI 9 HX 370, AVX-512 VNNI), using `/completion` with `{"prompt":"The capital of France is","n_predict":16,"n_probs":5,"temperature":0,"cache_prompt":false}`.

The reported logprobs are `log_softmax` over the full vocab of the raw logits (`post_sampling_probs=false`).

- Prompt tokens: `[760 "The", 6511 " capital", 314 " of", 9338 " France", 369 " is"]`
- Greedy output ids: `[11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369]`, which is `" Paris.\nThe capital of France is Paris.\nThe capital of France is"`
- Step 0 top-5: ` Paris` 0.5883, ` not` 0.0650, ` a` 0.0458, ` in` 0.0270, ` located` 0.0205
- Re-running the same request gave bit-identical results.
- The PTQ1_0 file (identical weights, different kernel) gives the same 16 tokens. Probabilities differ by up to about 0.01 (step 0: 0.5894 vs 0.5883). It is stored under `ptq1_0_variant`.
- **Step 8 is a near-tie** (` Paris` 0.2605 vs ` not` 0.2600). A correct implementation can flip it. Judge parity by the top-5 probabilities (tolerance about 0.01) rather than by exact token equality at that step.
