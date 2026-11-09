# Task-accuracy gate

Generated 2026-11-09T17:15:51+00:00 from `full, subset`.

Greedy decoding, fixed prompts (`prompts.py`), the same item set per task across engines.
Confidence intervals are Wilson 95%; paired differences use the discordant pairs only.

| run | engine | task | accuracy | n | 95% CI | mean s/item |
|---|---|---|---|---|---|---|
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
| gemma3-1b-peregrine | peregrine | GSM8K | 55.0% | 60 | 42.5-66.9% | 25.79 |
| gemma3-1b-peregrine | peregrine | HumanEval+ | 25.0% | 60 | 15.8-37.2% | 55.34 |
| gemma3-1b-peregrine | peregrine | MMLU | 40.0% | 200 | 33.5-46.9% | 9.37 |
| llama32-1b-peregrine | peregrine | GSM8K | 42.7% | 150 | 35.0-50.7% | 28.14 |
| llama32-1b-peregrine | peregrine | HumanEval+ | 31.0% | 100 | 22.8-40.6% | 44.24 |
| llama32-1b-peregrine | peregrine | MMLU | 28.0% | 300 | 23.2-33.3% | 9.67 |
| qwen25-05b-peregrine | peregrine | GSM8K | 38.0% | 150 | 30.6-46.0% | 20.29 |
| qwen25-05b-peregrine | peregrine | HumanEval+ | 29.0% | 100 | 21.0-38.5% | 35.39 |
| qwen25-05b-peregrine | peregrine | MMLU | 32.0% | 300 | 27.0-37.5% | 4.20 |
| qwen3-17b-peregrine | peregrine | GSM8K | 75.0% | 40 | 59.8-85.8% | 47.31 |
| qwen3-17b-peregrine | peregrine | HumanEval+ | 47.5% | 40 | 32.9-62.5% | 47.91 |
| qwen3-17b-peregrine | peregrine | MMLU | 46.7% | 150 | 38.9-54.6% | 16.79 |

## Paired differences

`Δ llama.cpp` is the accuracy difference on the items both runs answered, with the 95% CI of the paired difference and the exact McNemar p-value. `Δ ref` is the same against the full-precision reference. `n` is the number of paired items.

| run | task | n | Δ ref | Δ llama.cpp | paired 95% CI (pp) | McNemar p | gate |
|---|---|---|---|---|---|---|---|
| gemma3-1b-peregrine | GSM8K | 60 | -8.3 | -3.3 | [-11.3, +4.6] | 0.688 | pass |
| gemma3-1b-peregrine | HumanEval+ | 60 | -8.3 | -1.7 | [-9.0, +5.6] | 1.000 | pass |
| gemma3-1b-peregrine | MMLU | 200 | +1.0 | +0.5 | [-2.1, +3.1] | 1.000 | pass |
| llama32-1b-peregrine | GSM8K | 150 | -4.7 | -3.3 | [-8.4, +1.7] | 0.302 | pass |
| llama32-1b-peregrine | HumanEval+ | 100 | +1.0 | +3.0 | [-0.3, +6.3] | 0.250 | pass |
| llama32-1b-peregrine | MMLU | 300 | -1.3 | -2.0 | [-4.4, +0.4] | 0.180 | pass |
| qwen25-05b-peregrine | GSM8K | 150 | -5.3 | -2.0 | [-6.3, +2.3] | 0.549 | pass |
| qwen25-05b-peregrine | HumanEval+ | 100 | +1.0 | +1.0 | [-4.2, +6.2] | 1.000 | pass |
| qwen25-05b-peregrine | MMLU | 300 | -3.0 | -2.0 | [-4.8, +0.8] | 0.238 | pass |
| qwen3-17b-peregrine | GSM8K | 40 | -7.5 | +5.0 | [-10.4, +20.4] | 0.754 | pass |
| qwen3-17b-peregrine | HumanEval+ | 40 | -2.5 | -2.5 | [-13.4, +8.4] | 1.000 | pass |
| qwen3-17b-peregrine | MMLU | 150 | -4.7 | -3.3 | [-7.6, +1.0] | 0.227 | pass |

## Gate verdict

All 12 checks pass: within ±4 pp of llama.cpp (paired CI or non-significant McNemar) and no more than 10 points below the full-precision reference.

