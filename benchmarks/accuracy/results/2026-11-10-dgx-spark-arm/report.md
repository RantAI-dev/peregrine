# Task-accuracy gate

Generated 2026-11-09T21:00:31+00:00 from `2026-11-10-dgx-spark-arm/full, 2026-11-09-dgx-spark/full`.

Greedy decoding, fixed prompts (`prompts.py`), the same item set per task across engines.
Confidence intervals are Wilson 95%; paired differences use the discordant pairs only.

| run | engine | task | accuracy | n | 95% CI | mean s/item |
|---|---|---|---|---|---|---|
| gemma3-1b-peregrine | peregrine | GSM8K | 45.6% | 250 | 39.5-51.8% | 6.79 |
| gemma3-1b-peregrine | peregrine | HumanEval+ | 32.5% | 163 | 25.8-40.0% | 8.18 |
| gemma3-1b-peregrine | peregrine | MMLU | 40.4% | 500 | 36.2-44.8% | 1.86 |
| llama32-1b-peregrine | peregrine | GSM8K | 48.0% | 250 | 41.9-54.2% | 8.28 |
| llama32-1b-peregrine | peregrine | HumanEval+ | 31.9% | 163 | 25.2-39.4% | 5.57 |
| llama32-1b-peregrine | peregrine | MMLU | 28.8% | 500 | 25.0-32.9% | 1.45 |
| qwen25-05b-peregrine | peregrine | GSM8K | 40.0% | 250 | 34.1-46.2% | 7.86 |
| qwen25-05b-peregrine | peregrine | HumanEval+ | 34.4% | 163 | 27.5-41.9% | 4.07 |
| qwen25-05b-peregrine | peregrine | MMLU | 35.0% | 500 | 30.9-39.3% | 0.93 |
| qwen3-17b-peregrine | peregrine | GSM8K | 77.6% | 250 | 72.0-82.3% | 24.38 |
| qwen3-17b-peregrine | peregrine | HumanEval+ | 55.8% | 163 | 48.2-63.2% | 9.92 |
| qwen3-17b-peregrine | peregrine | MMLU | 45.4% | 500 | 41.1-49.8% | 4.88 |
| gemma3-1b-llama | llama_cpp | GSM8K | 49.2% | 250 | 43.1-55.4% | 2.18 |
| gemma3-1b-llama | llama_cpp | HumanEval+ | 31.9% | 163 | 25.2-39.4% | 2.31 |
| gemma3-1b-llama | llama_cpp | MMLU | 40.0% | 500 | 35.8-44.4% | 0.25 |
| gemma3-1b-vllm | vllm | GSM8K | 54.0% | 250 | 47.8-60.1% | 2.53 |
| gemma3-1b-vllm | vllm | HumanEval+ | 38.7% | 163 | 31.5-46.3% | 2.47 |
| gemma3-1b-vllm | vllm | MMLU | 39.0% | 500 | 34.8-43.3% | 0.11 |
| llama32-1b-llama | llama_cpp | GSM8K | 48.8% | 250 | 42.7-55.0% | 1.53 |
| llama32-1b-llama | llama_cpp | HumanEval+ | 30.7% | 163 | 24.1-38.1% | 1.39 |
| llama32-1b-llama | llama_cpp | MMLU | 29.6% | 500 | 25.8-33.7% | 0.19 |
| llama32-1b-vllm | vllm | GSM8K | 48.8% | 250 | 42.7-55.0% | 2.33 |
| llama32-1b-vllm | vllm | HumanEval+ | 33.1% | 163 | 26.4-40.7% | 1.89 |
| llama32-1b-vllm | vllm | MMLU | 30.4% | 500 | 26.5-34.6% | 0.16 |
| qwen25-05b-llama | llama_cpp | GSM8K | 40.4% | 250 | 34.5-46.6% | 1.64 |
| qwen25-05b-llama | llama_cpp | HumanEval+ | 32.5% | 163 | 25.8-40.0% | 1.30 |
| qwen25-05b-llama | llama_cpp | MMLU | 33.6% | 500 | 29.6-37.9% | 0.12 |
| qwen25-05b-vllm | vllm | GSM8K | 42.8% | 250 | 36.8-49.0% | 1.63 |
| qwen25-05b-vllm | vllm | HumanEval+ | 25.2% | 163 | 19.1-32.3% | 1.54 |
| qwen25-05b-vllm | vllm | MMLU | 36.2% | 500 | 32.1-40.5% | 0.07 |
| qwen3-17b-llama | llama_cpp | GSM8K | 78.0% | 250 | 72.5-82.7% | 2.88 |
| qwen3-17b-llama | llama_cpp | HumanEval+ | 56.4% | 163 | 48.8-63.8% | 2.26 |
| qwen3-17b-llama | llama_cpp | MMLU | 45.4% | 500 | 41.1-49.8% | 0.44 |
| qwen3-17b-vllm | vllm | GSM8K | 80.4% | 250 | 75.0-84.8% | 4.64 |
| qwen3-17b-vllm | vllm | HumanEval+ | 62.6% | 163 | 54.9-69.6% | 3.68 |
| qwen3-17b-vllm | vllm | MMLU | 47.0% | 500 | 42.7-51.4% | 0.38 |

## Paired differences

`Δ llama.cpp` is the accuracy difference on the items both runs answered, with the 95% CI of the paired difference and the exact McNemar p-value. `Δ ref` is the same against the full-precision reference. `n` is the number of paired items.

| run | task | n | Δ ref | Δ llama.cpp | paired 95% CI (pp) | McNemar p | gate |
|---|---|---|---|---|---|---|---|
| gemma3-1b-peregrine | GSM8K | 250 | -8.4 | -3.6 | [-7.6, +0.4] | 0.122 | pass |
| gemma3-1b-peregrine | HumanEval+ | 163 | -6.1 | +0.6 | [-2.1, +3.3] | 1.000 | pass |
| gemma3-1b-peregrine | MMLU | 500 | +1.4 | +0.4 | [-1.3, +2.1] | 0.815 | pass |
| llama32-1b-peregrine | GSM8K | 250 | -0.8 | -0.8 | [-5.1, +3.5] | 0.856 | pass |
| llama32-1b-peregrine | HumanEval+ | 163 | -1.2 | +1.2 | [-1.7, +4.2] | 0.688 | pass |
| llama32-1b-peregrine | MMLU | 500 | -1.6 | -0.8 | [-2.7, +1.1] | 0.541 | pass |
| qwen25-05b-peregrine | GSM8K | 250 | -2.8 | -0.4 | [-4.5, +3.7] | 1.000 | pass |
| qwen25-05b-peregrine | HumanEval+ | 163 | +9.2 | +1.8 | [-1.8, +5.4] | 0.508 | pass |
| qwen25-05b-peregrine | MMLU | 500 | -1.2 | +1.4 | [-0.5, +3.3] | 0.210 | pass |
| qwen3-17b-peregrine | GSM8K | 250 | -2.8 | -0.4 | [-4.6, +3.8] | 1.000 | pass |
| qwen3-17b-peregrine | HumanEval+ | 163 | -6.7 | -0.6 | [-4.9, +3.7] | 1.000 | pass |
| qwen3-17b-peregrine | MMLU | 500 | -1.6 | +0.0 | [-1.9, +1.9] | 1.000 | pass |

## Gate verdict

All 12 checks pass: within ±4 pp of llama.cpp (paired CI or non-significant McNemar) and no more than 10 points below the full-precision reference.

