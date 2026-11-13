# Targets

The numbers Peregrine holds itself to. A target is met only when the benchmark harness
(`docs/dev/benchmarking.md`) shows it on the named host, with the raw results committed under
`benchmarks/results/`. Until then a number here is a goal, not a claim.

## Reference hosts

| ID | Machine | CPU features | Role |
|---|---|---|---|
| H1 | Ryzen AI 9 HX 370 laptop, WSL2, 74-80 GB/s, ~1.85 GHz under load | AVX-512 VNNI/GFNI/VBMI | development; noisy, not for published numbers |
| H2 | "bluefin": Ryzen 3 PRO 4350G, bare-metal Linux, 8 threads, 13 GB | AVX2 only | AVX2 tier correctness and speed |
| H3 | quiet high-end x86 machine, bare metal | AVX-512 | **published numbers** (to be provided) |
| H4 | Intel 12th gen+ or Core Ultra laptop | AVX-VNNI-256, no AVX-512 | Intel client tier |
| H5 | Apple M-series or Graviton | NEON/SDOT | ARM tier, if ARM is in scope |

## 1. Accuracy: always on

Every change, every release.

| Measure | Target |
|---|---|
| Mean KL divergence against the exact f32 path, per gate model | ≤ 1e-4 (today: 6e-5 on Bonsai 2 9B) |
| Greedy tokens against the reference engine, gate prompts, first 256 tokens | identical |
| Batch invariance: one user alone vs with 9 others, 500 prompts, T=0 and T=0.7 | 100% identical |
| Speculation (MTP, DSpark, suffix) on vs off | 100% identical |
| Task scores (GSM8K, HumanEval+, MMLU subset) vs the reference engine on the same weights | within ±1 point |
| Default paths that lose accuracy | none |

**Distance to llama.cpp on real text** (not the budget above, which is against Peregrine's own exact path; method
in `docs/dev/accuracy.md`). 2026-11-07, bluefin (Ryzen 3 PRO 4350G, AVX2), branch `s3.3a-kl` (main at `bc4dda6`
plus the `kld` tool), llama.cpp v0.9.0 `cd959ba`; wikitext-2 test, `-c 512`, 40 chunks, 8 threads:

| Model (Q4_K_M) | Mean KLD | Median / 99% KLD | RMS Δp | Same top p | PPL Peregrine / llama.cpp |
|---|---|---|---|---|---|
| SmolLM2-360M-Instruct | 0.00398 ± 0.00006 | 0.0028 / 0.025 | 1.59 % | 96.1 % | 14.232 / 14.220 |
| Qwen2.5-0.5B-Instruct | 0.00345 ± 0.00005 | 0.0025 / 0.019 | 1.42 % | 96.5 % | 13.956 / 13.928 |
| Llama-3.2-1B-Instruct | 0.00248 ± 0.00003 | 0.0018 / 0.014 | 1.15 % | 97.2 % | 14.893 / 14.865 |
| Qwen3-1.7B | 0.01262 ± 0.00041 | 0.0054 / 0.105 | 3.16 % | 94.8 % | 15.819 / 15.590 |
| Llama-2-7B-Chat | 0.00171 ± 0.00004 | 0.0007 / 0.015 | 1.24 % | 98.1 % | 7.283 / 7.281 |
| Mistral-7B-Instruct-v0.2 | 0.00121 ± 0.00003 | 0.0006 / 0.010 | 1.01 % | 98.2 % | 6.245 / 6.237 |
| Gemma-3-1B-IT | 0.00726 ± 0.00032 | 0.0025 / 0.082 | 2.41 % | 96.1 % | 28.329 / 27.881 |
| Gemma-3-1B-IT (-c 1024) | 0.00799 ± 0.00085 | 0.0022 / 0.084 | 2.58 % | 96.3 % | 27.096 / 26.632 |
| Qwen3-30B-A3B | 0.00885 ± 0.00041 | 0.0040 / 0.077 | 2.98 % | 95.9 % | 10.450 / 10.425 |
| Qwen3.5-2B | 0.00417 ± 0.00006 | 0.0032 / 0.020 | 1.50 % | 96.3 % | 13.858 / 13.845 |
| Phi-4 (20 chunks) | 0.00149 ± 0.00004 | 0.0009 / 0.0096 | 1.08 % | 97.7 % | 6.648 / 6.652 |

Qwen3-1.7B is the outlier, and the cause is on llama.cpp's side: a layer-by-layer comparison (S3.3b) finds every
Qwen3-specific step (q/k head norms, NeoX RoPE at base 1e6, the tied head) equal to llama.cpp's within ~1e-7; the
first jump is at the attention, where llama.cpp's default F16 KV cache makes it compute Q·K and the value mix
with f16 operands. Qwen3's head norms make the raw scores large (up to ~350), so that rounding becomes a 0.3-0.4 %
error in every softmax weight. Against an exact f32 evaluation of the attention, Peregrine is 4.7e-6 away and
llama.cpp 1.6e-4 (1.8e-7 with an F32 KV cache). A like-for-like comparison needs both engines' attention in f32
(llama.cpp `--cache-type-k f32 --cache-type-v f32 --flash-attn off`). Bonsai (PQ2_0) is not listed: llama.cpp v0.9.0 refuses the file (`invalid ggml type 142`).

### Like-for-like: the same cache precision on both sides

The table above compares two engines that do the same computation *differently*: Peregrine keeps its queries in
f32 and rounds its cached keys and values to f16, and llama.cpp's cache is f16 too, so its attention reads f16
operands. The f32 KV cache (`KvPrec::F32`, `--kv f32`, `PEREGRINE_KV=f32`) removes that difference: both engines'
attention then reads f32 keys and values.

2026-11-07, bluefin (Ryzen 3 PRO 4350G, AVX2), branch `s3.3c-f32-kv` (main at `74656e4` plus the f32 cache),
llama.cpp v0.9.0 `cd959ba`; wikitext-2 test, `-c 512`, 40 chunks, 8 threads. llama.cpp:
`--cache-type-k <prec> --cache-type-v <prec> --flash-attn off`; Peregrine: `kld --kv <prec>`. Flash attention is
off on the llama.cpp side in both rows (its CPU kernels do support it, and `auto` picks it on — a two-chunk
SmolLM2 run gives PPL 13.7227 with it on and 13.7081 with it off, so the two are not the same computation; the
row above was measured with llama.cpp's defaults, that is with flash attention on).

| Model (Q4_K_M) | KV cache, both sides | Mean KLD | Median / 99% KLD | RMS Δp | Same top p | PPL Peregrine / llama.cpp |
|---|---|---|---|---|---|---|
| SmolLM2-360M-Instruct | f16 | 0.004078 ± 0.000066 | 0.0027 / 0.028 | 1.62 % | 96.03 % | 14.232 / 14.225 |
| SmolLM2-360M-Instruct | **f32** | 0.003975 ± 0.000063 | 0.0027 / 0.025 | 1.58 % | 96.28 % | 14.231 / 14.230 |
| Llama-3.2-1B-Instruct | f16 | 0.002514 ± 0.000032 | 0.0018 / 0.015 | 1.17 % | 97.20 % | 14.893 / 14.867 |
| Llama-3.2-1B-Instruct | **f32** | 0.002552 ± 0.000036 | 0.0018 / 0.016 | 1.26 % | 97.12 % | 14.904 / 14.869 |
| Qwen3-1.7B | f16 | 0.012714 ± 0.000306 | 0.0056 / 0.117 | 3.26 % | 94.96 % | 15.819 / 15.587 |
| Qwen3-1.7B | **f32** | 0.012283 ± 0.000402 | 0.0053 / 0.103 | 3.13 % | 94.96 % | 15.925 / 15.603 |

**The cache precision is not what separates the engines.** Putting both engines on f32 moves Qwen3-1.7B's mean
KLD by 3 % (0.012714 → 0.012283) and its same-top-p not at all; the other two models do not move either. Yet the
cache precision is not a small perturbation of one engine's output: llama.cpp compared against *itself* with the
other cache precision gives

| Model (Q4_K_M) | llama.cpp f16 vs f32, one engine | RMS Δp | Same top p |
|---|---|---|---|
| SmolLM2-360M-Instruct | 0.00399 ± 0.00006 | 1.59 % | 96.29 % |
| Llama-3.2-1B-Instruct | 0.00257 ± 0.00003 | 1.21 % | 97.22 % |
| Qwen3-1.7B | 0.01319 ± 0.00039 | 3.22 % | 94.70 % |

— as large as the whole distance between the two engines. The reason it does not show up in the comparison is that
it is a *common* perturbation: the two engines' keys and values are nearly the same, so they round to the same
halves, and the rounding moves both distributions the same way. The four combinations for Qwen3-1.7B say it
plainly:

| Peregrine cache | llama.cpp cache | Mean KLD | Same top p |
|---|---|---|---|
| f16 | f16 | 0.012714 ± 0.000306 | 94.96 % |
| f16 | f32 | 0.012823 ± 0.000332 | 94.74 % |
| f32 | f16 | 0.012852 ± 0.000295 | 94.72 % |
| f32 | f32 | 0.012283 ± 0.000402 | 94.96 % |

All four are within 5 % of each other. So the f16 KV cache is *not* what separates the engines, and S3.3b's
finding has to be read with that in mind: in the isolated attention the f16 cache is the only error there is
(Peregrine 4.7e-6 from the exact f32 attention, llama.cpp 1.6e-4), and the measurement above confirms it — the
cache precision moves llama.cpp's own end-to-end output by 0.0132 on Qwen3-1.7B. But the two engines' keys and
values are nearly the same, so they round to the same halves and the rounding moves *both* distributions the
same way: it is a common perturbation, not a divergence, and it cancels in the comparison. What is left is the
rest of the attention — llama.cpp quantises the activations to Q8_K before the Q4_K projections, while Peregrine
uses its own int8 layouts, and Qwen3's larger normed scores make that difference show up more than in the other
two families. The 7 B models were not re-measured here (each run is about four times the 1.7 B one); their numbers
above stand.

### The opt-in fast prompt mode (`--fast-prompt`)

Not the default engine: the mode is lossy and off by default, and this is the accuracy report ADR 0003 requires of
a feature that is (S5.4c, 2026-11-12, desktop H1, branch `s5.4c-fast-prompt-accuracy`, 12 threads; method and raw
output in `benchmarks/results/desktop-zen5-2026-11-12-s5.4c/`). Wikitext-2 test, 40 chunks of 512 tokens, each
chunk read as a prompt, 10 200 scored positions; "fast vs exact" is the same model and cache with the option on
against off, "against F16" is the unquantised file of the same upload:

| Model (Q4_K_M) | Fast vs exact: mean KLD | Median / 99 % | RMS Δp | Same top p | Exact vs F16 | Fast vs F16 | Share added |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 0.003514 ± 0.000049 | 0.002472 / 0.023960 | 1.41 % | 96.75 % | 0.038558 | 0.039006 | +1.16 % |
| Qwen2.5-0.5B-Instruct | 0.002645 ± 0.000053 | 0.001843 / 0.015386 | 1.23 % | 97.20 % | 0.032525 | 0.032439 | −0.26 % |
| SmolLM2-360M-Instruct | 0.004504 ± 0.000091 | 0.003078 / 0.029123 | 1.65 % | 95.80 % | 0.035133 | 0.035477 | +0.98 % |
| Gemma-3-1B-IT | 0.009219 ± 0.000452 | 0.002859 / 0.105939 | 2.70 % | 95.84 % | 0.087227 | 0.088221 | +1.14 % |
| Qwen3-1.7B | 0.017115 ± 0.000993 | 0.006523 / 0.148500 | 3.75 % | 93.91 % | 0.112264 | 0.111481 | −0.70 % |

The share is `(KL_fast_vs_F16 − KL_exact_vs_F16) / KL_exact_vs_F16`; two of the five are negative, which means the
addition is smaller than the run's own error, not an improvement. The design's ship bound was 0.0015 on the 1B
until S5.4j; 0.0035 was measured, and the fork's own `q8_k_r16` arithmetic measures 0.0034 on the same file and
corpus when its repacked path is on (`-rtr`), while the 0.000945 that the original bound was taken from is the
fork's *default* wide path, a finer `Q8_1`/`Q8_0` conversion (design §5c). **S5.4j re-baselines the bound to 0.0040
on the 1B, the same top token on ≥ 96 % of positions and no task score outside the paired test's noise, on the
project lead's recommendation; the owner's confirmation is pending, so the option stays labelled Experimental**
(design §4). S5.4j's own accuracy run, fast against exact on the same corpus and chunks, is 0.003464 on the 1B and
0.017336 on Qwen3 1.7B — 1.4 % below and 1.3 % above S5.4c's numbers, inside the runs' error (the attention order
moves rounding only). Task scores (GSM8K/MMLU/HumanEval+, `benchmarks/accuracy/`) move by at most +7.5 points at
n = 80, every paired McNemar p ≥ 0.15. Prompt speed is 1.26-1.46× the default engine's in the eight measured cells
(AVX-512; the per-tier format table of S5.4f keeps the mode to Q6_K on AVX2, where it is 1.05-1.08× the exact
engine on the 1B); the ratio against ik_llama.cpp is in `docs/benchmark.md`.

## 2. Speed

Measured per the method in section 4. Baselines are H1 today.

### Claim 1: fastest CPU engine for ternary models on x86

| Measure (Bonsai 2) | Today (H1) | Target |
|---|---|---|
| 9B decode, 1 user | 23.5 tok/s (28.7 with MTP) | ≥ 90% of the sustained-bandwidth ceiling, and above every competitor |
| 9B total throughput, 10 users (engine) | ~75 tok/s | ≥ 1.5× today |
| 9B prompt processing | ~80 tok/s | ≥ 1.5× today |
| 27B decode, 1 user | 3.6 tok/s | ≥ 6 tok/s (ceiling ~9) |
| Server throughput as a share of the engine, 10 users | 70-78% | ≥ 90% |
| First token at 10 users | 5.8 s | ≤ 3 s |

Competitors: PrismML's llama.cpp fork, bitnet.cpp / T-MAC (on converted weights if they cannot read PQ2_0).

### Claim 2: standard GGUF models

| Measure (Qwen 3 / Qwen 3.5 / Llama 3, Q4_K_M and Q8_0) | Target |
|---|---|
| Decode, 1 user | ≥ 85% of sustained bandwidth; ≥ llama.cpp; within 5% of ik_llama.cpp |
| Total throughput, 10 users | ≥ llama.cpp and ik_llama.cpp |
| Prompt processing | ≥ llama.cpp |
| MTP-capable models, 1 user | ≥ 1.25× without MTP, lossless |

Competitors: llama.cpp, ik_llama.cpp, OpenVINO GenAI, MNN; vLLM / SGLang CPU backends for many users.

### Long context

| Measure | Target |
|---|---|
| Decode at 16k context vs 512 | Bonsai 2: ≥ 80% of the 512 rate (only 1 layer in 4 keeps a KV cache) |
| Reported contexts | 512, 4k, 16k (32k where the model allows) |

## 3. Support

| Release | Architectures | Formats | CPUs | Tokenizers / templates |
|---|---|---|---|---|
| today | llama, qwen2, qwen3, qwen35, gemma3, qwen3moe, phi3 | F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q6_K, PQ2_0, PTQ1_0 | AVX2, AVX-512 | GPT-2 BPE; 3 built-in templates |
| v0.3 (Stage 2) | + Gemma 3 (real-file gate) | + Q4_0, Q5_K, IQ4_XS, IQ4_NL | + AVX-VNNI-256; all formats on AVX2 | + SentencePiece; the model file's own template |
| v0.4 | + gpt-oss | + Q2_K, Q3_K, MXFP4 | + ARM NEON/SDOT (if in scope) | |

A model is "supported" when it loads, passes the accuracy gate in section 1 against llama.cpp on the same
file, and appears in the generated support table.

## 4. How numbers are measured

- Same model file and machine for every engine; each engine at its own best thread count.
- Four numbers per model: decode tok/s at 1 user, prompt tok/s, total tok/s at 4 and 10 users, first-token
  time at 10 users; each at 512, 4k and 16k context.
- At least 3 interleaved repeats; the median is reported with the spread.
- Each number also as a fraction of the host's measured ceiling (bandwidth for decode, int8 MAC rate for
  prompts and batches).
- Commit, toolchain, options, competitor versions and raw JSON are published with the table, including
  every case where Peregrine loses.

## 5. Engineering

| Measure | Target |
|---|---|
| CI wall time | ≤ 15 min |
| Kernel tiers tested in CI (scalar, AVX2, AVX-512 via emulation, NEON) | all |
| `unsafe` outside the kernels crate | none (after Stage 1) |
| Runtime dependencies of the engine crate | 1 (`memmap2`) |
| Rebuild after a one-line edit, `quick` profile, H2 | ≤ 10 s (release today: 19.6 s) |
