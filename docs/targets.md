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

Qwen3-1.7B is the outlier (and Peregrine's perplexity on it is 1.5 % higher than llama.cpp's). On 8 chunks the
scalar kernels give 0.0136 against 0.0145 on AVX2 (`PEREGRINE_EXACT=1`, which affects only ternary weights, gives
the same as native), so the difference is not a SIMD kernel's: it is in the computation itself and is the first
thing to investigate under S3.3. Bonsai (PQ2_0) is not listed: llama.cpp v0.9.0 refuses the file (`invalid ggml type 142`).

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
| today | llama, qwen2, qwen3, qwen35 | F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q6_K, PQ2_0, PTQ1_0 | AVX2, AVX-512 | GPT-2 BPE; 3 built-in templates |
| v0.3 (Stage 2) | + plain Qwen 3.5, Gemma 3, Phi-4, qwen3moe | + Q4_0, Q5_K, IQ4_XS, IQ4_NL | + AVX-VNNI-256; all formats on AVX2 | + SentencePiece; the model file's own template |
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
