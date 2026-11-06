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
