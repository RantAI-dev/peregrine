# Benchmarks: Peregrine against llama.cpp

Where Peregrine stands against llama.cpp on the same machine, the same model file and the same thread
count, and, on the desktop, against the ik_llama.cpp fork. Wins and losses are both reported here. The raw data, method and machine records are in
[`benchmarks/results/`](../benchmarks/results/); the targets these numbers are held to are in
[targets.md](targets.md).

![Where Peregrine is fastest](img/bench-wins.svg)

## Summary

- **Single-user decode wins every completed throughput row.** On the 12-core desktop, Peregrine leads
  llama.cpp's faster flash-attention setting by **1.13× to 1.69×**: Llama 3.2 1B is 1.31× at 512 and
  1.60× at 4096, Qwen 2.5 0.5B is 1.37× and 1.57×, SmolLM2 360M is 1.34× and 1.69×, and gemma-3-1b-it is
  1.35× and 1.13×. On the 4-core Ryzen 3 (bluefin) all three models are ahead, from **1.07× to 1.34×**.
  The lead grows with context because attention reads each cached key and value once per group of heads.
- **Multi-user throughput is 1.28–1.80× faster.** With 10 concurrent users, Peregrine serves **1.28–1.43×**
  llama.cpp's aggregate throughput on the 12-core desktop (1.13–1.38× at 4 users) and **1.47–1.80×** on
  the 4-core Ryzen 3 (1.80× on SmolLM2 360M at 4096; 1.73× at 512).
- **Prompt processing is ahead on all three hosts.** Prompt throughput is **1.03–1.30×** on the 12-core
  desktop, **1.03–1.43×** on the 4-core Ryzen 3, and **1.78–2.45×** on the 20-core ARM Cortex-X925/A725
  (`aarch64-neon-i8mm`).
- **Bonsai 2 (ternary): 1.4–2.3× decode, 1.9–2.6× prompt, and 1.8–2.8× with 10 users** against PrismML's
  llama.cpp fork across x86-64 (AVX-512 and AVX2) and ARM64 (NEON). Upstream llama.cpp cannot load
  `PQ2_0` files, so PrismML's fork is the comparison engine for ternary models.

- **Against ik_llama.cpp, prompt processing is behind.** On the desktop the fork processes prompts about
  twice as fast (Peregrine at 0.39-0.58×); decode is level (0.89-1.06×) and Peregrine is ahead with 10
  users (1.05-3.09×). See [Against ik_llama.cpp](#against-ik_llamacpp-desktop-avx-512).

**Against llama.cpp, no completed throughput row is below 1.0×** on any host. The single remaining loss is time to first
token (TTFT) with 10 users on bluefin for Llama 3.2 1B at 4096 context (1.04×, Peregrine slower); at
512 context Peregrine is faster on every TTFT row (0.66–0.86× on bluefin, 0.70–0.86× on the desktop).

## Test machines

| | Desktop (H1) | Bluefin (H2) | DGX Spark (H3) |
|---|---|---|---|
| CPU | AMD Ryzen AI 9 HX 370 (Zen 5), 12C/24T | AMD Ryzen 3 PRO 4350G (Zen 2), 4C/8T | 10 × Cortex-X925 + 10 × Cortex-A725, 20C |
| SIMD tier | `x86_64-avx2-avx512-vnni-gfni-vbmi` | `x86_64-avx2` | `aarch64-neon-i8mm` / `aarch64-neon-dotprod` |
| Threads used | 12 (one per physical core) | 4 (one per physical core) | 10 (X925 cluster) and 20 (all cores) |
| OS | Linux (WSL2) | Fedora (bare metal) | Ubuntu Linux (bare metal) |
| Results | [`desktop-zen5-2026-11-09`](../benchmarks/results/desktop-zen5-2026-11-09/README.md) | [`bluefin-zen2-2026-11-08-q5`](../benchmarks/results/bluefin-zen2-2026-11-08-q5/README.md) | [`spark-arm-2026-11-09-i8mm`](../benchmarks/results/spark-arm-2026-11-09-i8mm/README.md) |

All comparisons run llama.cpp v0.9.0 (`cd959ba`) against Peregrine on the exact same GGUF files
(verified by SHA-256), with 256 greedy tokens per request, prompts calibrated to the target context,
three interleaved rounds, and the median reported. **Each ratio is measured against llama.cpp's faster
flash-attention setting (`-fa auto` or `-fa off`) for that specific row.**

## Single-user decode

![Single-user decode](img/bench-decode.svg)

| Machine | Model | Context | Peregrine (tok/s) | llama.cpp (tok/s) | Peregrine ÷ llama.cpp |
|---|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 512 | 60.6 | 46.4 | **1.31×** |
| Desktop | Llama 3.2 1B | 4096 | 48.5 | 30.4 | **1.60×** |
| Desktop | Qwen 2.5 0.5B | 512 | 95.5 | 69.5 | **1.37×** |
| Desktop | Qwen 2.5 0.5B | 4096 | 71.9 | 45.8 | **1.57×** |
| Desktop | SmolLM2 360M | 512 | 102.9 | 77.0 | **1.34×** |
| Desktop | SmolLM2 360M | 4096 | 70.8 | 41.9 | **1.69×** |
| Desktop | gemma-3-1b-it | 512 | 59.9 | 44.3 | **1.35×** |
| Desktop | gemma-3-1b-it | 4096 | 46.3 | 41.0 | **1.13×** |
| Bluefin | Llama 3.2 1B | 512 | 46.8 | 42.5 | **1.10×** |
| Bluefin | Llama 3.2 1B | 4096 | 36.8 | 29.3 | **1.26×** |
| Bluefin | Qwen 2.5 0.5B | 512 | 83.5 | 77.7 | **1.07×** |
| Bluefin | Qwen 2.5 0.5B | 4096 | 65.9 | 52.2 | **1.26×** |
| Bluefin | SmolLM2 360M | 512 | 108.9 | 94.5 | **1.15×** |
| Bluefin | SmolLM2 360M | 4096 | 63.8 | 47.5 | **1.34×** |

## Prompt processing

![Prompt processing](img/bench-prefill.svg)

| Machine | Model | Context | Peregrine (tok/s) | llama.cpp (tok/s) | Peregrine ÷ llama.cpp |
|---|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 512 | 272.6 | 252.7 | **1.08×** |
| Desktop | Llama 3.2 1B | 4096 | 195.3 | 174.3 | **1.12×** |
| Desktop | Qwen 2.5 0.5B | 512 | 479.3 | 423.8 | **1.13×** |
| Desktop | Qwen 2.5 0.5B | 4096 | 355.1 | 272.4 | **1.30×** |
| Desktop | SmolLM2 360M | 512 | 451.1 | 436.3 | **1.03×** |
| Desktop | SmolLM2 360M | 4096 | 292.9 | 263.9 | **1.11×** |
| Desktop | gemma-3-1b-it | 512 | 284.9 | 258.1 | **1.10×** |
| Desktop | gemma-3-1b-it | 4096 | 279.6 | 247.9 | **1.13×** |
| Bluefin | Llama 3.2 1B | 512 | 195.7 | 188.1 | **1.04×** |
| Bluefin | Llama 3.2 1B | 4096 | 145.0 | 141.4 | **1.03×** |
| Bluefin | Qwen 2.5 0.5B | 512 | 434.8 | 303.7 | **1.43×** |
| Bluefin | Qwen 2.5 0.5B | 4096 | 277.6 | 236.5 | **1.17×** |
| Bluefin | SmolLM2 360M | 512 | 443.2 | 318.9 | **1.39×** |
| Bluefin | SmolLM2 360M | 4096 | 228.5 | 217.1 | **1.05×** |

## Multi-user throughput

![Total throughput, 10 users at once](img/bench-users10.svg)

| Machine | Model | 4 users (tok/s) | 10 users (tok/s) | 10 users, Peregrine ÷ llama.cpp |
|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 70.3 vs 60.3 | 83.6 vs 61.8 | **1.35×** |
| Desktop | Qwen 2.5 0.5B | 108.4 vs 96.0 | 121.7 vs 93.6 | **1.30×** |
| Desktop | SmolLM2 360M | 117.1 vs 85.0 | 146.6 vs 102.4 | **1.43×** |
| Desktop | gemma-3-1b-it | 63.4 vs 49.6 | 69.3 vs 54.1 | **1.28×** |
| Bluefin | Llama 3.2 1B | 53.7 vs 39.6 | 55.5 vs 37.7 | **1.47×** |
| Bluefin | Qwen 2.5 0.5B | 104.3 vs 65.7 | 108.4 vs 68.4 | **1.59×** |
| Bluefin | SmolLM2 360M | 114.2 vs 66.2 | 116.8 vs 67.6 | **1.73×** |

Aggregate tokens per second across all concurrent users at a 512-token context. At a 4096-token context
on bluefin, Peregrine leads by **1.43×** (4 users) and **1.52×** (10 users) on Llama 3.2 1B, **1.35×**
and **1.40×** on Qwen 2.5 0.5B, and **1.55×** and **1.80×** on SmolLM2 360M.

## Against ik_llama.cpp (desktop, AVX-512)

ik_llama.cpp is the llama.cpp fork most often reported faster than upstream on x86. It runs the
same GGUF files, so it was measured on the desktop on 2026-11-11 with its best documented CPU
settings (an AVX-512 build, `-rtr` run-time repacking, its own flash attention), beside llama.cpp
and Peregrine in the same session. Ratios are Peregrine ÷ ik_llama.cpp; above 1 means Peregrine is
faster. Medians of three interleaved rounds.

| Model | Context | Decode (tok/s): Peregrine / ik | Ratio | Prompt (tok/s): Peregrine / ik | Ratio | 10 users (tok/s): Peregrine / ik | Ratio |
|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 512 | 56.6 / 54.6 | **1.04×** | 274.1 / 526.8 | 0.52× | 85.1 / 58.5 | **1.45×** |
| Llama 3.2 1B | 4096 | 40.6 / 42.3 | 0.96× | 208.4 / 359.0 | 0.58× | not completed | n/a |
| Qwen 2.5 0.5B | 512 | 93.9 / 91.9 | **1.02×** | 470.9 / 1116.9 | 0.42× | 150.4 / 100.4 | **1.50×** |
| Qwen 2.5 0.5B | 4096 | 65.3 / 69.1 | 0.95× | 379.1 / 662.9 | 0.57× | 20.2 / 6.5 | **3.09×** |
| SmolLM2 360M | 512 | 97.9 / 92.4 | **1.06×** | 426.5 / 1030.6 | 0.41× | 132.0 / 74.8 | **1.76×** |
| SmolLM2 360M | 4096 | 57.8 / 55.5 | **1.04×** | 304.5 / 556.0 | 0.55× | not measured | n/a |
| gemma-3-1b-it | 512 | 54.3 / 54.0 | **1.01×** | 254.3 / 644.7 | 0.39× | 76.3 / 72.7 | **1.05×** |
| gemma-3-1b-it | 4096 | 47.5 / 53.4 | 0.89× | 281.2 / 569.4 | 0.49× | not measured | n/a |

What this says:

- **Prompt processing: ik_llama.cpp is about twice as fast on every row** (Peregrine at 0.39-0.58×).
  This is Peregrine's largest measured loss. The fork repacks the k-quant weights at load (`-rtr`)
  for its own interleaved AVX-512 kernels. Where Peregrine's time goes in comparison has not been
  profiled yet.
- **Decode: level.** Peregrine is 1.01-1.06× at 512 context and 0.89-1.04× at 4096.
- **Many users: Peregrine is ahead on every completed row** (1.05-3.09× with 10 users), because it
  batches all users into one pass over the weights. At 4 users one row is behind (gemma-3-1b-it
  at 512 context, 0.85×).
- Time to first token with 10 users is behind on two 512-context rows (SmolLM2 1.09× and
  gemma-3-1b-it 1.34× of ik_llama.cpp's time) and ahead on the others.

Not measured: OpenVINO GenAI, the 4096-context multi-user rows for SmolLM2 and gemma-3-1b-it, and
ik_llama.cpp on bluefin and on ARM. The 4096-context multi-user rows are single rounds. Method,
settings and raw data: [`benchmarks/results/desktop-zen5-2026-11-11-ik/`](../benchmarks/results/desktop-zen5-2026-11-11-ik/README.md).
The other tables on this page were measured earlier and against llama.cpp only.

## Bonsai 2 (ternary) vs PrismML's fork

Bonsai 2's ternary models (`PQ2_0`, ggml type 142) are compared against
[PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (`62c418c`, branch `prism`).
"Fork best" is the faster of the fork's flash-attention settings (`-fa on` or `-fa off`) for each row.
Full manifests: [desktop](../benchmarks/results/2026-11-09-bonsai2-prism/) and
[bluefin](../benchmarks/results/bluefin-zen2-2026-11-08-bonsai2-prism/).

![Bonsai 2: Peregrine against PrismML's fork](img/bench-bonsai.svg)

### Desktop (12 cores, AVX-512)

| Model | Context | Users | Metric | Peregrine | Fork best | Ratio |
|---|---|---|---|---|---|---|
| Bonsai 2 2B PQ2_0 | 512 | 1 | Decode tok/s | 75.14 | 34.89 | **2.15×** |
| Bonsai 2 2B PQ2_0 | 512 | 1 | Prompt tok/s | 270.24 | 119.35 | **2.26×** |
| Bonsai 2 2B PQ2_0 | 512 | 10 | Aggregate tok/s | 75.77 | 29.56 | **2.56×** |
| Bonsai 2 2B PQ2_0 | 512 | 10 | TTFT (s) | 14.10 | 44.38 | **0.32×** |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Decode tok/s | 64.60 | 32.65 | **1.98×** |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Prompt tok/s | 247.30 | 116.35 | **2.13×** |
| Bonsai 2 9B PQ2_0 | 512 | 1 | Decode tok/s | 25.47 | 10.92 | **2.33×** |
| Bonsai 2 9B PQ2_0 | 512 | 1 | Prompt tok/s | 71.65 | 28.45 | **2.52×** |
| Bonsai 2 9B PQ2_0 | 512 | 10 | Aggregate tok/s | 21.66 | 7.87 | **2.75×** |
| Bonsai 2 9B PQ2_0 | 512 | 10 | TTFT (s) | 52.12 | 181.96 | **0.29×** |
| Bonsai 2 9B PQ2_0 | 4096 | 1 | Decode tok/s | 22.69 | 10.12 | **2.24×** |
| Bonsai 2 9B PQ2_0 | 4096 | 1 | Prompt tok/s | 63.04 | 27.85 | **2.26×** |

With lossless multi-token prediction (MTP) on the 9B `-MTP-Q8_0` model (512 context, 1 user), Peregrine
decodes at **28.80 tok/s** (1.15× over its non-speculative 25.14 tok/s, and **3.09×** against the fork's
9.33 tok/s with `--spec-type draft-mtp`).

### Bluefin (4 cores, AVX2)

| Model | Context | Users | Metric | Peregrine | Fork best | Ratio |
|---|---|---|---|---|---|---|
| Bonsai 2 2B PQ2_0 | 512 | 1 | Decode tok/s | 46.40 | 33.63 | **1.38×** |
| Bonsai 2 2B PQ2_0 | 512 | 1 | Prompt tok/s | 118.67 | 56.95 | **2.08×** |
| Bonsai 2 2B PQ2_0 | 512 | 10 | Aggregate tok/s | 33.69 | 19.08 | **1.77×** |
| Bonsai 2 2B PQ2_0 | 512 | 10 | TTFT (s) | 32.82 | 82.05 | **0.40×** |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Decode tok/s | 42.28 | 30.79 | **1.37×** |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Prompt tok/s | 109.04 | 58.07 | **1.88×** |
| Bonsai 2 4B PQ2_0 | 512 | 1 | Decode tok/s | 24.59 | 14.95 | **1.64×** |
| Bonsai 2 4B PQ2_0 | 512 | 1 | Prompt tok/s | 48.05 | 18.55 | **2.59×** |
| Bonsai 2 4B PQ2_0 | 512 | 10 | Aggregate tok/s | 14.12 | 6.37 | **2.22×** |
| Bonsai 2 4B PQ2_0 | 512 | 10 | TTFT (s) | 78.84 | 279.73 | **0.28×** |

## ARM64 NEON and i8mm (DGX Spark)

Measured on the 20-core DGX Spark (10 × Cortex-X925 + 10 × Cortex-A725, `aarch64-neon-i8mm` and
`aarch64-neon-dotprod`) at a 512-token context:

| Model | Threads | Metric | Peregrine (tok/s) | Competitor (tok/s) | Ratio |
|---|---|---|---|---|---|
| Llama 3.2 1B Q4_K_M | 10 (X925) | Prompt tok/s | 573.8 | 234.2 (llama.cpp) | **2.45×** |
| Llama 3.2 1B Q4_K_M | 10 (X925) | Decode tok/s | 112.9 | 99.0 (llama.cpp) | **1.14×** |
| Llama 3.2 1B Q4_K_M | 20 (all) | Prompt tok/s | 629.8 | 322.0 (llama.cpp) | **1.96×** |
| Llama 3.2 1B Q4_K_M | 20 (all) | Decode tok/s | 111.6 | 86.4 (llama.cpp) | **1.29×** |
| Qwen 2.5 0.5B Q4_K_M | 10 (X925) | Prompt tok/s | 935.3 | 524.5 (llama.cpp) | **1.78×** |
| Qwen 2.5 0.5B Q4_K_M | 10 (X925) | Decode tok/s | 199.9 | 194.2 (llama.cpp) | **1.03×** |
| Qwen 2.5 0.5B Q4_K_M | 20 (all) | Prompt tok/s | 886.9 | 547.5 (llama.cpp) | **1.62×** |
| Qwen 2.5 0.5B Q4_K_M | 20 (all) | Decode tok/s | 198.8 | 128.3 (llama.cpp) | **1.55×** |
| Bonsai 2 2B PQ2_0 | 10 (X925) | Prompt tok/s | 404.0 | 167.7 (PrismML fork) | **2.41×** |
| Bonsai 2 2B PQ2_0 | 10 (X925) | Decode tok/s | 140.4 | 88.7 (PrismML fork) | **1.58×** |
| Bonsai 2 2B PQ2_0 | 20 (all) | Prompt tok/s | 433.1 | 200.5 (PrismML fork) | **2.16×** |
| Bonsai 2 2B PQ2_0 | 20 (all) | Decode tok/s | 146.8 | 64.5 (PrismML fork) | **2.28×** |
| Bonsai 2 4B-MTP PQ2_0 | 10 (X925) | Decode tok/s (MTP on) | 80.7 | 50.8 (PrismML fork) | **1.59×** |
| Bonsai 2 4B-MTP PQ2_0 | 20 (all) | Prompt tok/s | 185.2 | 80.9 (PrismML fork) | **2.29×** |
| Bonsai 2 4B-MTP PQ2_0 | 20 (all) | Decode tok/s | 67.7 | 32.5 (PrismML fork) | **2.08×** |

Full details: [`spark-arm-2026-11-09-i8mm`](../benchmarks/results/spark-arm-2026-11-09-i8mm/README.md)
and [`spark-arm-2026-11-10-bonsai2`](../benchmarks/results/spark-arm-2026-11-10-bonsai2/README.md).

## Task accuracy and decoding invariance

Speed numbers are paired with the task-accuracy gate and decoding-invariance suite in
[`docs/dev/accuracy.md`](dev/accuracy.md) and [`benchmarks/accuracy/`](../benchmarks/accuracy/):

- **Task accuracy (GSM8K 250, MMLU 500, HumanEval+ 163):** all four gate models
  (Qwen 2.5 0.5B Instruct, Llama 3.2 1B Instruct, gemma-3-1b-it, Qwen 3 1.7B, Q4_K_M) pass every
  paired check against same-host llama.cpp (`|Δ| ≤ 4` percentage points, exact McNemar `p ≥ 0.12`) and
  the BF16 vLLM floor across all three hardware tiers — AVX2, AVX-512, and ARM64 NEON i8mm.
- **Batch and speculation invariance:** 520 prompts decoded alone, in concurrent batches of 8, and
  with `--speculate` produce **520/520 byte-identical replies** on every host and tier.

## Reproducing

```sh
python3 benchmarks/run.py --help     # run the benchmark harness on a model and engine
python3 benchmarks/charts.py         # regenerate the SVG charts in docs/img/
```
