# Task-accuracy gate

Generated 2026-11-09T21:34:06+00:00 from `2026-11-09-dgx-spark, 2026-11-10-bluefin-x86`.

Greedy decoding, fixed prompts (`prompts.py`), the same item set per task across engines.
Confidence intervals are Wilson 95%; paired differences use the discordant pairs only.

| run | engine | task | accuracy | n | 95% CI | mean s/item |
|---|---|---|---|---|---|---|
| gemma3-1b-vllm | vllm | GSM8K | 54.0% | 250 | 47.8-60.1% | 2.53 |
| gemma3-1b-vllm | vllm | HumanEval+ | 38.7% | 163 | 31.5-46.3% | 2.47 |
| gemma3-1b-vllm | vllm | MMLU | 39.0% | 500 | 34.8-43.3% | 0.11 |
| qwen25-05b-vllm | vllm | GSM8K | 42.8% | 250 | 36.8-49.0% | 1.63 |
| qwen25-05b-vllm | vllm | HumanEval+ | 25.2% | 163 | 19.1-32.3% | 1.54 |
| qwen25-05b-vllm | vllm | MMLU | 36.2% | 500 | 32.1-40.5% | 0.07 |
| gemma3-1b-llama | llama_cpp | GSM8K | 47.2% | 250 | 41.1-53.4% | 26.47 |
| gemma3-1b-llama | llama_cpp | HumanEval+ | 32.5% | 163 | 25.8-40.0% | 30.51 |
| gemma3-1b-llama | llama_cpp | MMLU | 39.6% | 500 | 35.4-44.0% | 5.85 |
| gemma3-1b-peregrine | peregrine | GSM8K | 45.2% | 250 | 39.1-51.4% | 15.87 |
| gemma3-1b-peregrine | peregrine | HumanEval+ | 34.4% | 163 | 27.5-41.9% | 18.26 |
| gemma3-1b-peregrine | peregrine | MMLU | 39.8% | 500 | 35.6-44.2% | 3.74 |
| qwen25-05b-llama | llama_cpp | GSM8K | 43.6% | 250 | 37.6-49.8% | 16.80 |
| qwen25-05b-llama | llama_cpp | HumanEval+ | 31.3% | 163 | 24.7-38.8% | 15.23 |
| qwen25-05b-llama | llama_cpp | MMLU | 32.0% | 500 | 28.1-36.2% | 2.93 |
| qwen25-05b-peregrine | peregrine | GSM8K | 42.4% | 250 | 36.4-48.6% | 9.66 |
| qwen25-05b-peregrine | peregrine | HumanEval+ | 32.5% | 163 | 25.8-40.0% | 8.72 |
| qwen25-05b-peregrine | peregrine | MMLU | 33.6% | 500 | 29.6-37.9% | 1.80 |

## Paired differences

`Δ llama.cpp` is the accuracy difference on the items both runs answered, with the 95% CI of the paired difference and the exact McNemar p-value. `Δ ref` is the same against the full-precision reference. `n` is the number of paired items.

| run | task | n | Δ ref | Δ llama.cpp | paired 95% CI (pp) | McNemar p | gate |
|---|---|---|---|---|---|---|---|
| gemma3-1b-peregrine | GSM8K | 250 | -8.8 | -2.0 | [-6.4, +2.4] | 0.473 | pass |
| gemma3-1b-peregrine | HumanEval+ | 163 | -4.3 | +1.8 | [-2.1, +5.8] | 0.549 | pass |
| gemma3-1b-peregrine | MMLU | 500 | +0.8 | +0.2 | [-2.1, +2.5] | 1.000 | pass |
| qwen25-05b-peregrine | GSM8K | 250 | -0.4 | -1.2 | [-5.1, +2.7] | 0.690 | pass |
| qwen25-05b-peregrine | HumanEval+ | 163 | +7.4 | +1.2 | [-2.6, +5.0] | 0.754 | pass |
| qwen25-05b-peregrine | MMLU | 500 | -2.6 | +1.6 | [-0.6, +3.8] | 0.215 | pass |

## Gate verdict

All 6 checks pass: within ±4 pp of llama.cpp (paired CI or non-significant McNemar) and no more than 10 points below the full-precision reference.

