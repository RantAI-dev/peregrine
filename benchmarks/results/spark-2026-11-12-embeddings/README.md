# DGX Spark embedding standings: Peregrine vs llama.cpp

The published embedding table compares Peregrine with every engine on x86. This run
measures the same `/v1/embeddings` path on the project's ARM host: EmbeddingGemma 300M
Q8_0, Qwen3-Embedding 0.6B in F16 and Q8_0, and Nemotron 3.5 Embed 1B Q8_0, against
llama.cpp v0.9.0, with the stage profile that explains each cell. Nothing in the engine
changed.

- Date: 2026-11-12, 00:00–02:30 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores (10 × Cortex-X925 +
  10 × Cortex-A725), 121 GB unified memory, Linux `6.17.0-1008-nvidia`, governor
  `performance`, kernel tier `aarch64-neon-i8mm`.
- Peregrine: `origin/main` at `3030172`, built from the bench branch's merge commit
  `a762a5c` (no source difference from `3030172`), `cargo build --release -p peregrine`,
  rustc 1.99.1. The chat baselines in
  [`spark-2026-11-11-ik`](../spark-2026-11-11-ik/README.md) were measured on `487f0a2`;
  this run rebuilt after `origin/main` gained the bert and mistral3 families.
- llama.cpp: v0.9.0, `cd959ba`, harness cache build,
  `CUDA_VISIBLE_DEVICES="" ... --embedding -ngl 0 -t 10 -np 32 -fa auto
  -b 2048 -ub 2048`.
- Host otherwise idle (`top` reported 98–99% idle; the 1-minute load average was the
  previous timed run decaying, so the harness's 0.5 load gate was relaxed). Memory
  available stayed at 34–36 GB.

## Method

`benchmarks/embed.py`, three interleaved rounds, one model process at a time, median of
the per-round `texts/s`. Every engine sees byte-identical texts for a model.

- **Cells**: 16 / 128 / 512 / 2000 tokens × batch 1 and 32, plus one mixed-length batch
  of 32 texts drawn from 8–512 tokens with the fixed seed `20261111` (3 565 tokens).
  32 texts per length. Batch 1 sends up to 20 distinct texts (60 s budget); batch 32 is
  one request of 32 texts. Context 2048 tokens per slot, so a 2 000-token text plus
  special tokens fits (`--ctx-tokens 2048`).
- **Threads**: 10 (the Cortex-X925 cluster), the main-table count of the chat run.
- **llama.cpp**: one slot per text, `-np 32`, `-c 2056 × 32`, `-b 2048 -ub 2048`. A
  non-causal model needs the whole prompt in one ubatch; the bluefin run established
  that. llama.cpp feeds one embedding task at a time even inside one request, so its
  batch cells are serial by design; this is the same behaviour the published bluefin
  tables record.
- **Texts**: built with each GGUF's own tokenizer through `llama-server /tokenize`, so
  no licence-gated Hugging Face tokenizer repo is needed. The windows are distinct and
  calibrated within 5%: 16-token targets came out 15–16, 128 targets 127–129, 512
  targets 524–529, 2000 targets 1999–2001; mixed 8–528. The harness records the
  tokenizer's token count for every text.

## Models

| Model | Bytes | SHA-256 |
|---|---|---|
| EmbeddingGemma 300M Q8_0 | 333,590,944 | `b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63` |
| Qwen3-Embedding 0.6B F16 | 1,197,629,632 | `421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340` |
| Qwen3-Embedding 0.6B Q8_0 | 639,150,592 | `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` |
| Nemotron 3.5 Embed 1B Q8_0 | 1,220,296,800 | `27c8e82c48828017c29f16fa87eb8958b19d3e35d8f25d73bf51b61efb2ec298` |

The Nemotron row is added to `benchmarks/embed_models.json` in this PR (mean pooling,
dim 2048). The Qwen F16 file is the same weights widened with `llama-quantize`; the Q8_0
file is the official Qwen GGUF.

## Results

`texts/s` is the median of three rounds; `ratio` is Peregrine ÷ llama.cpp, above 1.00 is
Peregrine ahead. `p50` is the batch-1 request latency.

### EmbeddingGemma 300M Q8_0

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine p50 (ms) | llama.cpp p50 (ms) |
|---|---|---|---|---|---|
| L16 b1 | 102.30 | 73.22 | **1.40×** | 10.6 | 13.3 |
| L16 b32 | 271.48 | 121.22 | **2.24×** | 117.4 | 263.5 |
| L128 b1 | 15.52 | 11.89 | **1.31×** | 71.0 | 82.9 |
| L128 b32 | 38.00 | 15.98 | **2.38×** | 841.5 | 2001.3 |
| L512 b1 | 3.72 | 3.24 | **1.15×** | 297.8 | 308.5 |
| L512 b32 | 8.91 | 3.55 | **2.51×** | 3591.1 | 9010.8 |
| L2000 b1 | 0.93 | 0.84 | **1.11×** | 1191.8 | 1190.6 |
| L2000 b32 | — | — | — | — | — |
| mixed b32 | 16.73 | 17.17 | 0.97× | 1911.9 | 1863.4 |

### Qwen3-Embedding 0.6B Q8_0

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine p50 (ms) | llama.cpp p50 (ms) |
|---|---|---|---|---|---|
| L16 b1 | 34.10 | 28.84 | **1.18×** | 32.2 | 34.4 |
| L16 b32 | 84.11 | 31.05 | **2.71×** | 379.1 | 1029.7 |
| L128 b1 | 4.46 | 4.23 | **1.05×** | 248.4 | 236.2 |
| L128 b32 | 10.52 | 4.32 | **2.44×** | 3040.6 | 7402.5 |
| L512 b1 | 0.99 | 1.00 | 0.99× | 1114.7 | 1000.4 |
| L512 b32 | 2.36 | 1.00 | **2.36×** | 13555.1 | 32067.4 |
| L2000 b1 | 0.213 | 0.217 | 0.98× | 5563.3 | 4606.6 |
| L2000 b32 | — | — | — | — | — |
| mixed b32 | 4.56 | 4.79 | 0.95× | 7021.0 | 6674.7 |

### Qwen3-Embedding 0.6B F16

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine p50 (ms) | llama.cpp p50 (ms) |
|---|---|---|---|---|---|
| L16 b1 | 41.87 | 27.61 | **1.52×** | 26.2 | 35.9 |
| L16 b32 | 94.73 | 29.51 | **3.21×** | 336.8 | 1083.5 |
| L128 b1 | 5.00 | 4.16 | **1.20×** | 222.0 | 240.2 |
| L128 b32 | 12.09 | 4.22 | **2.87×** | 2646.0 | 7578.0 |
| L512 b1 | 1.13 | 0.99 | **1.14×** | 979.7 | 1010.9 |
| L512 b32 | 2.69 | 0.99 | **2.72×** | 11913.8 | 32420.1 |
| L2000 b1 | 0.23 | 0.22 | **1.05×** | 5047.9 | 4541.0 |
| L2000 b32 | — | — | — | — | — |
| mixed b32 | 5.20 | 4.69 | **1.11×** | 6148.3 | 6826.8 |

### Nemotron 3.5 Embed 1B Q8_0

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine p50 (ms) | llama.cpp p50 (ms) |
|---|---|---|---|---|---|
| L16 b1 | 20.30 | 17.09 | **1.19×** | 54.7 | 58.2 |
| L16 b32 | 49.84 | 17.81 | **2.80×** | 640.8 | 1796.0 |
| L128 b1 | 2.65 | 2.37 | **1.12×** | 417.6 | 422.3 |
| L128 b32 | 6.38 | 2.38 | **2.68×** | 5014.7 | 13421.7 |
| L512 b1 | 0.64 | 0.56 | **1.14×** | 1740.7 | 1799.5 |
| L512 b32 | 1.53 | 0.56 | **2.73×** | 20864.6 | 57542.0 |
| L2000 b1 | 0.16 | 0.13 | **1.23×** | 7684.4 | 7933.9 |
| L2000 b32 | — | — | — | — | — |
| mixed b32 | 2.85 | 2.66 | **1.07×** | 11219.0 | 12042.6 |

## Cells where Peregrine is behind

Only four of the 32 measured cells are behind llama.cpp, all close to parity:

| Model | Cell | Peregrine | llama.cpp | Ratio | Stage that explains it |
|---|---|---|---|---|---|
| EmbeddingGemma 300M Q8_0 | mixed b32 | 16.73 | 17.17 | 0.97× | matmul-bound (80% of the pass); the mixed batch is dominated by short texts, so there is little to amortise, and llama.cpp's per-text loop is slightly faster |
| Qwen3-Embedding 0.6B Q8_0 | L512 b1 | 0.994 | 1.002 | 0.99× | parity inside a ±1% round spread; attention is 11% of the pass at 512 |
| Qwen3-Embedding 0.6B Q8_0 | L2000 b1 | 0.212 | 0.217 | 0.98× | attention is 31% of the pass at 2 000 tokens; llama.cpp's CPU flash attention is the faster long single-text attention path |
| Qwen3-Embedding 0.6B Q8_0 | mixed b32 | 4.56 | 4.79 | 0.95× | matmul-bound (81%), mixed short texts; the batch does not amortise enough to beat the serial path |

The F16 Qwen row is ahead in every cell, so the Q8_0 losses are not a model-level gap.
The batch-32 cells are Peregrine's largest wins (2.2–3.2×) because llama.cpp feeds one
embedding task at a time while Peregrine batches all 32 texts into shared weight passes.

## Stage profile per cell (Peregrine)

`crates/peregrine-bench/examples/cellprof` drives the same `Engine`, one request per
cell, and prints the `StepStats` split in milliseconds. Tokenisation is timed apart
(0.03–3.35 ms per text) and is not in these numbers. `matmul` includes the projections
and the feed-forward; `other` is norms, GELU, pooling and the quantisation fills.

### EmbeddingGemma 300M Q8_0

| Cell | matmul | attention | other | total |
|---|---|---|---|---|
| L16 b1 | 9.8 | 0.2 | 2.2 | 12.3 |
| L16 b32 | 242.8 | 5.1 | 58.0 | 306.0 |
| L128 b1 | 62.3 | 4.1 | 13.4 | 79.9 |
| L128 b32 | 1746.9 | 64.0 | 338.6 | 2149.4 |
| L512 b1 | 234.9 | 23.9 | 41.0 | 299.8 |
| L512 b32 | 7129.6 | 773.0 | 1287.4 | 9190.1 |
| L2000 b1 | 885.7 | 163.6 | 137.5 | 1186.8 |
| mixed b32 | 1519.6 | 101.7 | 288.2 | 1909.5 |

### Qwen3-Embedding 0.6B Q8_0 (causal)

| Cell | matmul | attention | other | total |
|---|---|---|---|---|
| L16 b1 | 27.7 | 0.8 | 3.4 | 31.9 |
| L16 b32 | 811.2 | 17.5 | 110.3 | 939.1 |
| L128 b1 | 203.6 | 9.0 | 27.8 | 240.4 |
| L128 b32 | 6515.9 | 293.0 | 897.6 | 7706.4 |
| L512 b1 | 839.8 | 121.9 | 116.0 | 1077.7 |
| L512 b32 | 27008.5 | 3945.5 | 3402.6 | 34356.7 |
| L2000 b1 | 3215.2 | 1647.9 | 460.0 | 5323.1 |
| mixed b32 | 5715.4 | 501.9 | 799.2 | 7016.5 |

### Nemotron 3.5 Embed 1B Q8_0 (non-causal)

| Cell | matmul | attention | other | total |
|---|---|---|---|---|
| L16 b1 | 46.7 | 1.1 | 2.8 | 50.5 |
| L16 b32 | 1494.3 | 28.2 | 89.9 | 1612.5 |
| L128 b1 | 377.3 | 15.0 | 15.8 | 408.0 |
| L128 b32 | 12002.3 | 474.0 | 503.7 | 12980.0 |
| L512 b1 | 1578.3 | 157.5 | 4.7 | 1740.4 |
| L512 b32 | 49962.7 | 5124.3 | 88.1 | 55175.1 |
| L2000 b1 | 6016.3 | 2781.7 | 0.0 | 7742.4 |
| mixed b32 | 10562.8 | 693.6 | 253.8 | 11510.2 |

The pattern is the same on all three encoders: **matmul dominates**, and the attention
share grows with text length — 11% at 512 and 31% at 2 000 tokens for Qwen, 36% at
2 000 for Nemotron, 14% at 2 000 for Gemma. That is where the only repeated loss sits:
the long single-text cells are the ones where llama.cpp's flash-attention path is
competitive or slightly faster, while the batched cells are matmul-bound and Peregrine
wins them by 2.2–3.2×.

## Skips and caveats

- **L2000 b32 is skipped for every row**: 32 × 2 000 = 64 000 tokens exceeds each
  model's 32 768-token batch cap (the padded attention buffers would not fit). This is
  the same skip the bluefin tables record.
- The mixed cells for Qwen Q8_0 and EmbeddingGemma are within 3–5% of llama.cpp; the
  per-round spread of the batch-1 cells is under 1%, so the small Qwen Q8_0 losses are
  not noise.
- llama.cpp is fed one task at a time by `embed.py` (one request per text inside a
  batch cell, as its server does); this is the documented competitor setting, not a
  harness bug. Peregrine batches, which is why the batch-32 ratios are so large.
- The 16-token texts include a few 15-token windows (≤6.3% short); the harness records
  the tokenizer's count and every engine sees the same texts.

## Files

- `README.md`: this report.
- `<model>-<engine>-<precision>-r{1,2,3}.json`: raw run sets with the machine record,
  engine build, per-cell times, latencies and token counts.
- `*-skips.json`: the per-row skip decisions (L2000 b32).
- `summarize.py`: medians across the three rounds for every cell, including the
  2 000-token and mixed cells that `embed_report.py`'s fixed grid does not print.
- `summary.json`: `embed_report.py`'s generated summary (16/128/512 grid).
