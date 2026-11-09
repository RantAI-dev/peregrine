# Bluefin: Q8_0 decode at the bandwidth limit (2026-11-09)

The Q8_0 matrices — Gemma 3 1B's 302M-element token embedding used as the tied output
head, its `attn_v`, Qwen 2.5 0.5B's tied head and `attn_v`, SmolLM2's, Qwen 3.5 2B's
`attn_v`, Bonsai 2 4B's MTP block — now go through a repacked eight-row tile like Q5_0
(#105) instead of the eight separate streams of `q8_0_rows8`.

Previously the Q8_0 matmul was 38-39 % of Gemma 3 1B's decode step and sustained
~26-27 GB/s against llama.cpp's ~33 GB/s (the coverage sweep, under load). The tile
holds eight rows per 32-weight block side by side (272 bytes, exactly the eight blocks
it replaces), so decode streams one contiguous run; the measured rate is now
**~41 GB/s**, the same as the Q5_0, Q6_K and Q4_K tile buckets on this host.

## Host and builds

| | |
|---|---|
| CPU | AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads, AVX2+FMA+F16C |
| RAM | 13.8 GiB + 7 GiB zram |
| Peregrine main | `origin/main` `0ef4096` (`perf: prefill attention reads F16 K/V directly (#115)`) |
| Peregrine branch | this PR (Q8_0 tile), same tree otherwise |
| Threads | decode 4 (physical cores), prefill pool 8 (default) |
| llama.cpp | v0.9.0-dev, build 1, commit `cd959ba`, `-t 4` |
| Load | quiet (1-minute load average < 0.5 at the start of the rounds) |

## Method

`peregrine bench --prompt-tokens {512,4096} --tokens 128` and
`llama-bench -p {512,4096} -n 0` / `-p 0 -n 128 -d {512,4096}`, `-t 4`, `-r 3`, `-fa 0`
and `-fa 1`. Five rounds alternate main and branch (AB, BA, AB, BA, AB); the tables
report medians. The llama.cpp column is its faster `-fa` setting for that row: `-fa 1`
for prefill and `-fa 0` for decode.

The raw per-round JSON is in this directory.

## Decode

| Model | Context | main | branch | llama.cpp | branch / main | branch / llama |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 3 1B | 512 | 41.66 | **43.73** | 41.86 | **1.050** | **1.045** |
| Gemma 3 1B | 4096 | 36.35 | **38.45** | 40.21 | **1.058** | 0.956 |
| Qwen 2.5 0.5B | 512 | 85.71 | **91.40** | 78.38 | **1.066** | **1.166** |
| Qwen 2.5 0.5B | 4096 | 69.57 | **73.09** | 51.96 | **1.051** | **1.407** |
| Llama 3.2 1B (control) | 512 | 47.43 | 46.89 | 42.12 | 0.989 | **1.113** |
| Llama 3.2 1B (control) | 4096 | 38.60 | 38.53 | 29.09 | 0.998 | **1.324** |

Tokens/s. The Llama control has no Q8_0 tensor, so it exercises none of this change;
its 512 row is a ~1 % run-to-run offset, and 4096 is flat.

Gemma 3 1B decode was 0.906× llama.cpp at 4096 in the sweep and is 0.956× here; at 512 it
is 1.045×. The remaining 4096 gap is attention (KV decode 2.4 ms/step), not the head:
the Q8_0 weight bucket is at the same rate as every other tile bucket.

## Prefill

| Model | Context | main | branch | llama.cpp | branch / main | branch / llama |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 3 1B | 512 | 265.63 | 266.96 | 165.06 | 1.005 | **1.617** |
| Gemma 3 1B | 4096 | 236.48 | 235.79 | 158.47 | 0.997 | **1.488** |
| Qwen 2.5 0.5B | 512 | 490.81 | 494.69 | 303.82 | 1.008 | **1.628** |
| Qwen 2.5 0.5B | 4096 | 343.16 | 345.04 | 235.25 | 1.005 | **1.467** |
| Llama 3.2 1B (control) | 512 | 233.16 | 233.21 | 191.46 | 1.000 | **1.218** |
| Llama 3.2 1B (control) | 4096 | 177.31 | 178.27 | 142.65 | 1.005 | **1.250** |

The Q8_0 tile is neutral on prefill: it streams the weights once per four activations,
like the Q5_0 tile, and the Q8_0 matrices are a small share of prompt processing.

## The Q8_0 bucket

`PEREGRINE_PROF2=full`, Gemma 3 1B, 128 decode steps, mean µs/step:

| Bucket | main (512) | branch (512) | branch (4096) |
|---|---:|---:|---:|
| Q8_0 matmul | 9 332 | 7 875 | 7 926 |
| Q5_0 matmul | 8 140 | 8 280 | 8 343 |
| Q6_K matmul | 2 030 | 2 042 | 2 039 |
| Q4_K matmul | 1 818 | 1 869 | 1 870 |
| **Q8_0 rate** (325.0 MB/step) | **34.8 GB/s** | **41.3 GB/s** | **41.0 GB/s** |

The Q8_0 bucket moves from the slowest of the four formats to the same bandwidth as the
others. Wall time at 512 goes from 24.1 to 22.9 ms/step.

## KL divergence against llama.cpp

`llama-perplexity --chunks 40`, `-c 512 -b 512`, wikitext-2-raw `wiki.test.raw`,
f32 KV cache on both sides (`--cache-type-k f32 --cache-type-v f32 --flash-attn off`
and `kld --kv f32`), 10 200 scored positions.

| Model | mean KLD main | mean KLD branch | same top p main | same top p branch |
|---|---:|---:|---:|---:|
| Gemma 3 1B | 0.006992 | 0.007608 | 96.686 % | 96.373 % |
| Qwen 2.5 0.5B | 0.003412 | 0.003412 | 96.294 % | 96.608 % |

Both are inside the sweep's distance from llama.cpp (the two engines' attention and
activation quantisation dominate). Gemma's mean KLD moves by 0.0006 (about 1.2 σ of the
reported error); Qwen's is unchanged and its top-1 agreement improves. The tile's fold
is the exact integer sum of a block, converted and scaled once per block — the same
expression the Q5_0 tile uses — so this is a re-rounding, not a new approximation.

## Notes

- No AVX-512 host was available: the 512-bit Q8_0 kernel is exercised by the bit-identity
  tests in CI under Intel SDE; no AVX-512 goldens exist to re-record (see the PR).
- Real-model AVX2 goldens move for the five models with Q8_0 matrices (Gemma 3 1B,
  Qwen 2.5 0.5B, SmolLM2 360M, Qwen 3.5 2B, Bonsai 2 4B MTP) and are re-recorded here;
  the synthetic Q8_0 goldens and the session fixture for the synthetic Q8_0 model move
  and are re-recorded too. Scalar has no repacked tiles, so its goldens are unchanged.
