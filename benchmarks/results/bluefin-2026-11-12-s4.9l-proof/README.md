# The S4.9l proof run on bluefin: embeddings and decisions against every engine (2026-11-12)

The stage's claim is "the fastest CPU engine for embedding and decision models at equal
accuracy" (roadmap S4.9l, `docs/design/s4.9-sota.md`). This is the bluefin part of the proof
run: every model family Peregrine serves for embeddings and decisions, against every engine on
this host that can run it, with accuracy beside speed. These are the bluefin inputs to
`docs/benchmark.md`.

**Rerank was skipped, by the brief's own rule.** `feat: rerank and classify for the bert
family` is not on `origin/main` as of `2a5e9f6`; commit `4196296` exists only on
`origin/s4.6b-bert-heads`. The brief says to measure rerank only if that commit is on main, so
there is no rerank table here.

## Machine, build and models

| | |
|---|---|
| Host | bluefin, AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads, AVX2+FMA, 13 GB RAM |
| Kernel | 7.2.7-200.fc44.x86_64 |
| Threads | **4** (one per physical core) for every engine and every row |
| Measured commit | `origin/main` at `2a5e9f6149cca9467365b912a219ca38d4ef9b1c` |
| Toolchain | rustc 1.99.1; `CARGO_TARGET_DIR=/var/tmp/target-s49l cargo build --release -p peregrine` |
| Load | 1-minute average 3.6-4.8 during the runs, but 99% idle CPU between runs (`vmstat`): the average is held up by a stuck `find` in D state and `kswapd0`, not by compute. No other benchmark ran on this host. |

`origin/main` at `2a5e9f6` already contains the bert family (S4.6a), its element-wise kernels
and float path (S4.9h, S4.9m), encoder attention at long text lengths (S4.9g part 3), the exact
embedding result cache (S4.9k) and Nemotron 3.5 Embed 1B (S4.14b). No engine source was changed
for this PR: the measured binary is a plain release build of that commit.

Models: `all-MiniLM-L6-v2` (F16, Q8_0), `bge-small-en-v1.5` (F32, Q8_0), `bge-m3` (Q8_0),
`EmbeddingGemma-300M` (Q8_0), `Qwen3-Embedding-0.6B` (F16, Q8_0) and `Nemotron-3.5-Embed-1B`
(Q8_0), under `~/models/embed/` and `~/models/nemotron/`. The Nemotron GGUF was downloaded from
`W4D/Nemotron-3.5-Embed-1B-GGUF`, SHA-256
`27c8e82c48828017c29f16fa87eb8958b19d3e35d8f25d73bf51b61efb2ec298` verified. Its F16 file was
downloaded too, only for the repo's accuracy test (below); no F16 timing row was measured.
Decisions: `tinyopenjev-for-testing-Q8_0.gguf` on `/v1/systemone`.

## Engines and the settings they got

| Engine | Version | Invocation (4 threads) |
|---|---|---|
| Peregrine | `peregrine 0.1.2`, commit `2a5e9f6` | `peregrine serve -m <gguf> -t 4 -c <model max tokens> --slots 40 --no-profile`; timing runs export `PEREGRINE_EMBED_CACHE_MB=0`, the cache-on table does not (64 MiB default) |
| llama.cpp | v0.9.0-dev, commit `cd959ba` | `llama-server --embedding --pooling <file's> -t 4 -fa auto` plus, per model: MiniLM/bge-small `-c 16640 -b 1024 -ub 1024 -np 32`; bge-m3 `-c 16448 -b 2048 -ub 2048 -np 8`; EmbeddingGemma and Nemotron `-c 65792 -b 2048 -ub 2048 -np 32`; Qwen3 `-c 20560 -b 2048 -ub 2048 -np 10` |
| OpenVINO | `openvino 2026.4.1`, `sentence-transformers 6.2.0` | `servers/openvino_server.py`: `SentenceTransformer(backend="openvino")`, `INFERENCE_NUM_THREADS=4`, `PERFORMANCE_HINT=LATENCY` |
| ONNX Runtime | `onnxruntime 1.31.1` | `servers/ort_server.py`: plain ORT, `CPUExecutionProvider`, `intra_op_num_threads=4`, `ORT_ENABLE_ALL`; mean/cls pooling over `last_hidden_state` |
| fastembed | `fastembed 0.9.1` on ONNX Runtime 1.31.1 | `servers/fastembed_server.py`, `CPUExecutionProvider`, threads 4 |
| text-embeddings-inference | `1.9.4`, `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4` | `podman run --network host` with MKL/OMP/RAYON=4, `--dtype float32 --max-batch-tokens 16384` |

Python engines run from `/var/tmp/embed-venv`. The llama.cpp slot counts are the ones that make
one request of the largest measured batch fit: `-np 32` holds 32 x 512-token texts,
`-np 10` is what Qwen3's causal 2 000-token texts need on this host (32 slots x 2 056 tokens
would not fit in 13 GB), and bge-m3's row is the registry's `-np 8 -c 16448` because its
16 384-token batch cap equals 8 x 2 048. Batch 32 therefore runs in four waves on bge-m3 and
Qwen3 at 512 tokens; that is recorded, not hidden.

## Method

- **Cells**: 16 / 128 / 512 tokens x batch 1, 8 and 32, plus one mixed-length batch of 32 texts
  drawn from 8-512 tokens with the fixed seed `20261111` (3 516 tokens). bge-m3,
  EmbeddingGemma, Qwen3 and Nemotron also get 2 000 tokens (batch 1 and 8; `L2000_b32` is
  64 000 tokens, above every model's batch cap). MiniLM and bge-small stop at 512.
- **Rounds**: three interleaved rounds, one engine and one model process at a time; the table
  is the median of the three per-round `texts/s`.
- **Texts**: `benchmarks/embed_texts.py` with each model's own tokenizer, so every engine sees
  byte-identical texts for a model. Costs per text: MiniLM and bge-small 16 / 128 / 509-512
  tokens; bge-m3 15-16 / 127-129 / 505-519 / 1 998-2 002; EmbeddingGemma and Qwen3 16 / 128 /
  512 / 2 000; Nemotron 14-17 / 126-130 / 510-513 / 1 998-2 001. Nothing is truncated.
- **Nemotron's prefixes**: its model card embeds retrieval text with `passage: ` and queries
  with `query: `. Every timing text is a passage (`passage: ` prepended, token counts include
  it) and the mixed batch alternates `query: ` and `passage: `; `embed_texts.py` gained
  `--prefix` and `--mixed-prefixes` for this.
- **Peregrine's cache** is off for every timing cell (`PEREGRINE_EMBED_CACHE_MB=0`): every
  round repeats the same texts, and a cache hit is not a computation. One extra table per model
  repeats the same request with the default 64 MiB cache on, labelled as repeated traffic.
  That table is a ceiling, not a compute claim: the harness's warm-up embeds the first text of
  each length, so batch-1 cells are all hits after it, while batch cells average one cold pass
  and four warm ones.
- **Accuracy**: the harness's `--mode accuracy` against the fp32 PyTorch vectors for MiniLM,
  bge-small and bge-m3 (200 texts from `crates/peregrine-models/tests/fixtures/bert/`), the
  torch-fp32 reference for Qwen3, and the fp32 ONNX export for EmbeddingGemma (no torch
  reference for that licence-gated checkpoint is on disk). The repo's ignored real-file tests
  were also run for every model on this machine and their printed cosines are the second
  accuracy table.
- **Equal accuracy**: a competitor row is compared when its cosine mean is within 0.0005 of
  Peregrine's row and not below 0.999. The ratios are always Peregrine / the best other engine.
- **Percentages**: `cellprof` runs the same `Engine` over the same texts and reports
  `StepStats` (matmul, attention, other) plus the `PEREGRINE_PROF2` barrier wait. "other" is
  everything net of matmul and attention: LayerNorm, GELU, the activation quantisation, the
  token embedding and pooling. This build has no separate bert-phase timers, so a Q8_0
  activation fill appears inside "other" rather than in its own column.

## Results

Tables generated from `raw/`, `raw-cache/` and `profile/` by
`python benchmarks/proof_report.py --dir <this dir>`. `texts/s` is the median of three rounds;
p50 is the whole request; the profile is one `cellprof` pass per cell and is not throughput.

## all-MiniLM-L6-v2

### texts/s (median of three rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fastembed fp32 | 256.19 | 578.36 | 623.27 | 81.44 | 85.97 | 74.20 | 15.50 | 13.30 | 13.08 | 13.76 |
| fastembed int8 (0.982) | 299.87 | 595.80 | 713.95 | 87.93 | 95.10 | 87.17 | 15.01 | 14.01 | 13.91 | 14.65 |
| llama_cpp f16 | 240.60 | 408.50 | 432.10 | 56.45 | 55.31 | 55.61 | 13.10 | 11.98 | 12.14 | 58.80 |
| llama_cpp q8_0 | 266.45 | 458.82 | 486.17 | 52.48 | 62.91 | 63.23 | 14.79 | 13.36 | 13.52 | 66.17 |
| onnxruntime fp32 | 263.58 | 587.13 | 609.80 | 87.93 | 91.66 | 73.67 | 15.78 | 13.32 | 13.17 | 13.53 |
| onnxruntime int8 (0.982) | 307.38 | 604.12 | 718.38 | 90.08 | 95.30 | 85.88 | 15.10 | 14.04 | 14.00 | 14.40 |
| openvino fp32 | 197.07 | 533.56 | 653.18 | 78.17 | 98.91 | 95.09 | 20.87 | 20.92 | 20.60 | 21.96 |
| peregrine f16 | 383.57 | 643.34 | 689.25 | 86.15 | 93.01 | 93.79 | 17.83 | 18.36 | 18.63 | 97.51 |
| peregrine q8_0 | 341.90 | 665.49 | 719.41 | 89.52 | 96.73 | 97.15 | 18.40 | 19.26 | 19.32 | 100.41 |
| tei fp32 | 151.50 | 210.16 | 239.01 | 34.79 | 34.86 | 37.33 | 4.66 | 4.31 | 4.14 | 9.79 |

### p50 latency, batch 1 (ms)

| engine / file | L16 b1 | L128 b1 | L512 b1 |
|---|---:|---:|---:|
| fastembed fp32 | 3.7 | 11.6 | 63.4 |
| fastembed int8 | 3.2 | 11.3 | 66.7 |
| llama_cpp f16 | 4.0 | 17.4 | 75.8 |
| llama_cpp q8_0 | 3.5 | 15.4 | 67.1 |
| onnxruntime fp32 | 3.6 | 11.3 | 63.0 |
| onnxruntime int8 | 3.1 | 11.0 | 66.1 |
| openvino fp32 | 4.9 | 12.7 | 47.5 |
| peregrine f16 | 2.4 | 11.5 | 54.8 |
| peregrine q8_0 | 2.8 | 11.0 | 53.0 |
| tei fp32 | 6.1 | 27.9 | 218.8 |

### Peregrine against the best equal-accuracy engine

| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |
|---|---:|---:|---|---:|
| L512 b1 | 17.83 | 20.87 | openvino fp32 | **0.85×** |
| L512 b8 | 18.36 | 20.92 | openvino fp32 | **0.88×** |
| L512 b32 | 18.63 | 20.60 | openvino fp32 | **0.90×** |
| L128 b8 | 93.01 | 98.91 | openvino fp32 | **0.94×** |
| L128 b1 | 86.15 | 87.93 | onnxruntime fp32 | **0.98×** |
| L128 b32 | 93.79 | 95.09 | openvino fp32 | **0.99×** |
| L16 b32 | 689.25 | 653.18 | openvino fp32 | **1.06×** |
| L16 b8 | 643.34 | 587.13 | onnxruntime fp32 | **1.10×** |
| L16 b1 | 383.57 | 266.45 | llama_cpp q8_0 | **1.44×** |
| mixed b32 | 97.51 | 66.17 | llama_cpp q8_0 | **1.47×** |

### Accuracy

| engine / file | cosine mean | cosine min | reference |
|---|---:|---:|---|
| fastembed fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| fastembed int8 | 0.981557 | 0.924134 | torch-fp32, 200 texts |
| llama_cpp f16 | 0.999998 | 0.999995 | torch-fp32, 200 texts |
| llama_cpp q8_0 | 0.999474 | 0.999124 | torch-fp32, 200 texts |
| onnxruntime fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| onnxruntime int8 | 0.981557 | 0.924134 | torch-fp32, 200 texts |
| openvino fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| peregrine f16 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| peregrine q8_0 | 0.999764 | 0.999521 | torch-fp32, 200 texts |
| tei fp32 | 0.999998 | 0.999995 | torch-fp32, 200 texts |
| peregrine f16 | 1.000000 | 1.000000 | real-file test, fp32 PyTorch |
| peregrine q8_0 | 0.999764 | 0.999521 | real-file test, fp32 PyTorch |

### Peregrine stage profile

| cell | request ms | tokenise ms | matmul | attention | other | barrier wait | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| L16 b1 | 2.4 | 0.02 | 74.1% | 6.5% | 19.4% | 3.7% | 115 |
| L16 b8 | 12.2 | 0.08 | 75.9% | 3.7% | 20.4% | 2.6% | 115 |
| L16 b32 | 46.1 | 0.26 | 76.2% | 4.3% | 19.5% | 2.9% | 460 |
| L128 b1 | 11.5 | 0.10 | 72.4% | 10.1% | 17.6% | 3.3% | 115 |
| L128 b8 | 85.8 | 0.40 | 72.3% | 10.5% | 17.3% | 2.5% | 920 |
| L128 b32 | 340.9 | 1.44 | 72.3% | 10.5% | 17.2% | 2.6% | 3680 |
| L512 b1 | 54.8 | 0.25 | 60.2% | 23.2% | 16.7% | 2.5% | 187 |
| L512 b8 | 435.6 | 1.44 | 59.4% | 24.3% | 16.3% | 2.2% | 1496 |
| L512 b32 | 1717.2 | 5.41 | 59.4% | 24.4% | 16.3% | 2.2% | 5984 |
| mixed b32 | 327.8 | 1.28 | 66.2% | 16.9% | 16.8% | 2.6% | 2727 |

### Peregrine, cache on (repeated texts)

| cell | cache off texts/s | cache on, repeated texts/s | speed-up |
|---|---:|---:|---:|
| L16 b1 | 383.57 | 1317.32 | 3.4× |
| L16 b8 | 643.34 | 2258.78 | 3.5× |
| L16 b32 | 689.25 | 2550.01 | 3.7× |
| L128 b1 | 86.15 | 1301.89 | 15.1× |
| L128 b8 | 93.01 | 539.63 | 5.8× |
| L128 b32 | 93.79 | 553.83 | 5.9× |
| L512 b1 | 17.83 | 1058.47 | 59.4× |
| L512 b8 | 18.36 | 117.75 | 6.4× |
| L512 b32 | 18.63 | 119.46 | 6.4× |
| mixed b32 | 97.51 | 187.61 | 1.9× |

## bge-small-en-v1.5

### texts/s (median of three rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fastembed fp32 | 172.23 | 350.73 | 407.20 | 49.11 | 48.63 | 39.53 | 8.36 | 7.25 | 7.27 | 7.43 |
| fastembed int8 (0.973) | 176.95 | 283.86 | 404.86 | 48.33 | 49.47 | 47.13 | 8.44 | 8.42 | 8.58 | 8.49 |
| llama_cpp f32 | 150.71 | 312.52 | 331.22 | 42.65 | 40.36 | 40.46 | 9.12 | 8.23 | 8.29 | 41.22 |
| llama_cpp q8_0 | 154.59 | 260.75 | 267.89 | 35.17 | 32.16 | 32.25 | 7.59 | 6.83 | 6.88 | 33.84 |
| onnxruntime fp32 | 176.67 | 352.29 | 407.01 | 49.59 | 48.31 | 39.27 | 8.34 | 7.19 | 7.25 | 7.41 |
| onnxruntime int8 (0.973) | 196.50 | 281.68 | 380.35 | 47.41 | 48.03 | 47.57 | 8.52 | 8.41 | 8.49 | 8.43 |
| openvino fp32 | 141.57 | 344.03 | 402.18 | 45.93 | 53.70 | 50.80 | 11.22 | 10.97 | 10.88 | 11.21 |
| peregrine f32 | 214.53 | 378.30 | 393.25 | 47.39 | 49.44 | 48.94 | 9.58 | 9.68 | 9.65 | 50.95 |
| peregrine q8_0 | 214.39 | 375.48 | 389.89 | 47.18 | 48.78 | 49.34 | 9.56 | 9.47 | 9.61 | 50.86 |
| tei fp32 | 73.89 | 109.51 | 109.67 | 18.87 | 19.08 | 19.07 | 2.28 | 2.21 | 2.15 | 4.91 |

### p50 latency, batch 1 (ms)

| engine / file | L16 b1 | L128 b1 | L512 b1 |
|---|---:|---:|---:|
| fastembed fp32 | 5.6 | 20.2 | 119.7 |
| fastembed int8 | 6.2 | 20.6 | 118.0 |
| llama_cpp f32 | 6.5 | 23.3 | 108.3 |
| llama_cpp q8_0 | 5.8 | 28.3 | 130.9 |
| onnxruntime fp32 | 5.5 | 20.1 | 119.9 |
| onnxruntime int8 | 4.4 | 20.5 | 117.5 |
| openvino fp32 | 6.9 | 21.6 | 88.7 |
| peregrine f32 | 4.5 | 21.0 | 104.0 |
| peregrine q8_0 | 4.5 | 21.1 | 104.4 |
| tei fp32 | 13.7 | 52.4 | 439.1 |

### Peregrine against the best equal-accuracy engine

| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |
|---|---:|---:|---|---:|
| L512 b1 | 9.56 | 11.22 | openvino fp32 | **0.85×** |
| L512 b8 | 9.47 | 10.97 | openvino fp32 | **0.86×** |
| L512 b32 | 9.61 | 10.88 | openvino fp32 | **0.88×** |
| L128 b8 | 48.78 | 53.70 | openvino fp32 | **0.91×** |
| L128 b1 | 47.18 | 49.59 | onnxruntime fp32 | **0.95×** |
| L16 b32 | 389.89 | 407.20 | fastembed fp32 | **0.96×** |
| L128 b32 | 49.34 | 50.80 | openvino fp32 | **0.97×** |
| L16 b8 | 375.48 | 352.29 | onnxruntime fp32 | **1.07×** |
| L16 b1 | 214.39 | 176.67 | onnxruntime fp32 | **1.21×** |
| mixed b32 | 50.86 | 41.22 | llama_cpp f32 | **1.23×** |

### Accuracy

| engine / file | cosine mean | cosine min | reference |
|---|---:|---:|---|
| fastembed fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| fastembed int8 | 0.972794 | 0.950791 | torch-fp32, 200 texts |
| llama_cpp f32 | 0.999999 | 0.999996 | torch-fp32, 200 texts |
| llama_cpp q8_0 | 0.999840 | 0.999740 | torch-fp32, 200 texts |
| onnxruntime fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| onnxruntime int8 | 0.972794 | 0.950791 | torch-fp32, 200 texts |
| openvino fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| peregrine f32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| peregrine q8_0 | 0.999928 | 0.999879 | torch-fp32, 200 texts |
| tei fp32 | 0.999999 | 0.999996 | torch-fp32, 200 texts |
| peregrine f32 | 1.000000 | 1.000000 | real-file test, fp32 PyTorch |
| peregrine q8_0 | 0.999928 | 0.999879 | real-file test, fp32 PyTorch |

### Peregrine stage profile

| cell | request ms | tokenise ms | matmul | attention | other | barrier wait | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| L16 b1 | 4.5 | 0.02 | 78.4% | 4.8% | 16.9% | 6.1% | 229 |
| L16 b8 | 21.1 | 0.08 | 75.9% | 4.2% | 19.9% | 3.0% | 229 |
| L16 b32 | 81.8 | 0.26 | 75.8% | 4.4% | 19.8% | 3.2% | 916 |
| L128 b1 | 21.1 | 0.08 | 71.0% | 11.1% | 17.9% | 3.0% | 229 |
| L128 b8 | 163.8 | 0.42 | 71.0% | 11.0% | 18.0% | 2.9% | 1832 |
| L128 b32 | 648.0 | 1.47 | 70.8% | 11.0% | 18.1% | 3.1% | 7328 |
| L512 b1 | 104.4 | 0.26 | 58.4% | 24.7% | 16.9% | 2.6% | 373 |
| L512 b8 | 844.3 | 1.45 | 58.0% | 25.2% | 16.8% | 2.0% | 2984 |
| L512 b32 | 3329.1 | 5.47 | 58.0% | 25.2% | 16.8% | 2.1% | 11936 |
| mixed b32 | 628.8 | 1.28 | 65.0% | 17.5% | 17.4% | 2.6% | 5433 |

### Peregrine, cache on (repeated texts)

| cell | cache off texts/s | cache on, repeated texts/s | speed-up |
|---|---:|---:|---:|
| L16 b1 | 214.39 | 1364.11 | 6.4× |
| L16 b8 | 375.48 | 1623.69 | 4.3× |
| L16 b32 | 389.89 | 1802.83 | 4.6× |
| L128 b1 | 47.18 | 1336.20 | 28.3× |
| L128 b8 | 48.78 | 304.16 | 6.2× |
| L128 b32 | 49.34 | 308.09 | 6.2× |
| L512 b1 | 9.56 | 1075.60 | 112.5× |
| L512 b8 | 9.47 | 62.65 | 6.6× |
| L512 b32 | 9.61 | 63.61 | 6.6× |
| mixed b32 | 50.86 | 104.33 | 2.1× |

## bge-m3

### texts/s (median of three rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | L2000 b1 | L2000 b8 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama_cpp q8_0 | 20.54 | 24.51 | 25.32 | 3.21 | 2.69 | 2.65 | 0.69 | 0.56 | 0.56 | 0.12 | 0.12 | 3.02 |
| peregrine q8_0 | 28.75 | 30.66 | 30.92 | 3.80 | 3.81 | 3.79 | 0.86 | 0.85 | 0.85 | 0.16 | 0.16 | 4.20 |

### p50 latency, batch 1 (ms)

| engine / file | L16 b1 | L128 b1 | L512 b1 | L2000 b1 |
|---|---:|---:|---:|---:|
| llama_cpp q8_0 | 47.2 | 310.9 | 1448.9 | 8326.7 |
| peregrine q8_0 | 34.0 | 262.7 | 1166.6 | 6170.2 |

### Peregrine against the best equal-accuracy engine

| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |
|---|---:|---:|---|---:|
| L128 b1 | 3.80 | 3.21 | llama_cpp q8_0 | **1.18×** |
| L16 b32 | 30.92 | 25.32 | llama_cpp q8_0 | **1.22×** |
| L512 b1 | 0.86 | 0.69 | llama_cpp q8_0 | **1.24×** |
| L16 b8 | 30.66 | 24.51 | llama_cpp q8_0 | **1.25×** |
| L2000 b1 | 0.16 | 0.12 | llama_cpp q8_0 | **1.35×** |
| L2000 b8 | 0.16 | 0.12 | llama_cpp q8_0 | **1.35×** |
| mixed b32 | 4.20 | 3.02 | llama_cpp q8_0 | **1.39×** |
| L16 b1 | 28.75 | 20.54 | llama_cpp q8_0 | **1.40×** |
| L128 b8 | 3.81 | 2.69 | llama_cpp q8_0 | **1.42×** |
| L128 b32 | 3.79 | 2.65 | llama_cpp q8_0 | **1.43×** |
| L512 b32 | 0.85 | 0.56 | llama_cpp q8_0 | **1.51×** |
| L512 b8 | 0.85 | 0.56 | llama_cpp q8_0 | **1.53×** |

### Accuracy

| engine / file | cosine mean | cosine min | reference |
|---|---:|---:|---|
| llama_cpp q8_0 | 0.999122 | 0.996776 | torch-fp32, 200 texts |
| peregrine q8_0 | 0.999139 | 0.996888 | torch-fp32, 200 texts |
| peregrine q8_0 | 0.999139 | 0.996888 | real-file test, fp32 PyTorch |

### Peregrine stage profile

| cell | request ms | tokenise ms | matmul | attention | other | barrier wait | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| L16 b1 | 34.0 | 0.04 | 80.3% | 3.1% | 16.6% | 3.6% | 553 |
| L16 b8 | 260.7 | 0.16 | 65.2% | 2.3% | 32.6% | 2.3% | 2765 |
| L16 b32 | 1034.5 | 0.42 | 76.0% | 2.5% | 21.5% | 2.1% | 9401 |
| L128 b1 | 262.7 | 0.17 | 77.1% | 5.1% | 17.8% | 1.5% | 553 |
| L128 b8 | 2096.5 | 0.66 | 77.4% | 5.2% | 17.5% | 1.3% | 4424 |
| L128 b32 | 8445.9 | 2.05 | 78.4% | 5.3% | 16.3% | 1.0% | 17792 |
| L512 b1 | 1166.6 | 0.47 | 69.8% | 13.6% | 16.7% | 1.5% | 1225 |
| L512 b8 | 9354.5 | 2.24 | 70.4% | 13.5% | 16.1% | 0.9% | 9800 |
| L512 b32 | 37462.3 | 7.81 | 70.4% | 13.6% | 16.0% | 1.0% | 39296 |
| L2000 b1 | 6170.2 | 1.29 | 51.2% | 36.5% | 12.4% | 1.0% | 2377 |
| L2000 b8 | 49518.0 | 8.11 | 51.1% | 36.6% | 12.3% | 0.9% | 19016 |
| mixed b32 | 7619.0 | 1.98 | 74.7% | 9.1% | 16.2% | 1.1% | 18053 |

### Peregrine, cache on (repeated texts)

| cell | cache off texts/s | cache on, repeated texts/s | speed-up |
|---|---:|---:|---:|
| L16 b1 | 28.75 | 931.01 | 32.4× |
| L16 b8 | 30.66 | 177.64 | 5.8× |
| L16 b32 | 30.92 | 191.85 | 6.2× |
| L128 b1 | 3.80 | 874.55 | 229.9× |
| L128 b8 | 3.81 | 25.12 | 6.6× |
| L128 b32 | 3.79 | 25.18 | 6.6× |
| L512 b1 | 0.86 | 669.59 | 781.3× |
| L512 b8 | 0.85 | 5.70 | 6.7× |
| L512 b32 | 0.85 | 5.69 | 6.7× |
| L2000 b1 | 0.16 | 402.95 | 2487.4× |
| L2000 b8 | 0.16 | 1.09 | 6.7× |
| mixed b32 | 4.20 | 8.72 | 2.1× |

## EmbeddingGemma 300M

### texts/s (median of three rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | L2000 b1 | L2000 b8 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama_cpp q8_0 (0.999) | 51.86 | 63.00 | 65.63 | 7.70 | 8.10 | 7.26 | 1.84 | 1.67 | 1.66 | 0.38 | 0.38 | 8.02 |
| peregrine q8_0 (0.999) | 78.20 | 86.63 | 87.23 | 10.70 | 10.90 | 10.81 | 2.39 | 2.41 | 2.42 | 0.54 | 0.55 | 11.96 |

### p50 latency, batch 1 (ms)

| engine / file | L16 b1 | L128 b1 | L512 b1 | L2000 b1 |
|---|---:|---:|---:|---:|
| llama_cpp q8_0 | 19.0 | 129.6 | 543.9 | 2603.1 |
| peregrine q8_0 | 12.6 | 92.8 | 418.6 | 1848.1 |

### Peregrine against the best equal-accuracy engine

| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |
|---|---:|---:|---|---:|
| L512 b1 | 2.39 | 1.84 | llama_cpp q8_0 | **1.30×** |
| L16 b32 | 87.23 | 65.63 | llama_cpp q8_0 | **1.33×** |
| L128 b8 | 10.90 | 8.10 | llama_cpp q8_0 | **1.35×** |
| L16 b8 | 86.63 | 63.00 | llama_cpp q8_0 | **1.37×** |
| L128 b1 | 10.70 | 7.70 | llama_cpp q8_0 | **1.39×** |
| L2000 b1 | 0.54 | 0.38 | llama_cpp q8_0 | **1.41×** |
| L2000 b8 | 0.55 | 0.38 | llama_cpp q8_0 | **1.42×** |
| L512 b8 | 2.41 | 1.67 | llama_cpp q8_0 | **1.45×** |
| L512 b32 | 2.42 | 1.66 | llama_cpp q8_0 | **1.46×** |
| L128 b32 | 10.81 | 7.26 | llama_cpp q8_0 | **1.49×** |
| mixed b32 | 11.96 | 8.02 | llama_cpp q8_0 | **1.49×** |
| L16 b1 | 78.20 | 51.86 | llama_cpp q8_0 | **1.51×** |

### Accuracy

| engine / file | cosine mean | cosine min | reference |
|---|---:|---:|---|
| llama_cpp q8_0 | 0.998746 | 0.991447 | onnx-fp32, 200 texts |
| peregrine q8_0 | 0.998763 | 0.991512 | onnx-fp32, 200 texts |
| peregrine q8_0 | 0.999463 | 0.997989 | real-file test, llama.cpp F16 |

### Peregrine stage profile

| cell | request ms | tokenise ms | matmul | attention | other | barrier wait | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| L16 b1 | 12.6 | 0.10 | 81.3% | 1.5% | 17.2% | 4.3% | 266 |
| L16 b8 | 92.1 | 0.44 | 75.5% | 1.4% | 23.1% | 2.0% | 1072 |
| L16 b32 | 366.4 | 1.34 | 78.4% | 1.4% | 20.2% | 2.0% | 4288 |
| L128 b1 | 92.8 | 0.48 | 79.1% | 4.1% | 16.9% | 1.9% | 290 |
| L128 b8 | 733.7 | 2.46 | 78.4% | 4.0% | 17.6% | 1.2% | 2320 |
| L128 b32 | 2958.7 | 8.58 | 79.1% | 4.1% | 16.9% | 1.2% | 9280 |
| L512 b1 | 418.6 | 1.54 | 70.9% | 12.5% | 16.6% | 4.2% | 578 |
| L512 b8 | 3318.0 | 9.00 | 70.9% | 12.5% | 16.7% | 3.7% | 4624 |
| L512 b32 | 13200.7 | 33.66 | 71.0% | 12.5% | 16.6% | 3.8% | 18496 |
| L2000 b1 | 1848.1 | 4.43 | 62.5% | 20.6% | 16.8% | 4.1% | 1730 |
| L2000 b8 | 14675.0 | 32.22 | 62.2% | 20.9% | 16.9% | 3.8% | 13840 |
| mixed b32 | 2675.3 | 7.62 | 74.8% | 8.1% | 17.0% | 2.3% | 9184 |

### Peregrine, cache on (repeated texts)

| cell | cache off texts/s | cache on, repeated texts/s | speed-up |
|---|---:|---:|---:|
| L16 b1 | 78.20 | 1002.93 | 12.8× |
| L16 b8 | 86.63 | 458.12 | 5.3× |
| L16 b32 | 87.23 | 484.69 | 5.6× |
| L128 b1 | 10.70 | 771.52 | 72.1× |
| L128 b8 | 10.90 | 69.48 | 6.4× |
| L128 b32 | 10.81 | 69.14 | 6.4× |
| L512 b1 | 2.39 | 398.16 | 166.7× |
| L512 b8 | 2.41 | 15.87 | 6.6× |
| L512 b32 | 2.42 | 15.82 | 6.5× |
| L2000 b1 | 0.54 | 180.51 | 333.7× |
| L2000 b8 | 0.55 | 3.60 | 6.6× |
| mixed b32 | 11.96 | 24.57 | 2.1× |

## Qwen3-Embedding 0.6B

### texts/s (median of three rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | L2000 b1 | L2000 b8 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama_cpp f16 | 12.75 | 12.97 | 13.03 | 1.82 | 1.82 | 1.82 | 0.43 | 0.43 | 0.43 | 0.08 | 0.09 | 2.05 |
| llama_cpp q8_0 | 14.88 | 15.92 | 15.95 | 2.21 | 2.21 | 2.21 | 0.50 | 0.50 | 0.50 | 0.08 | 0.08 | 2.46 |
| peregrine f16 | 20.90 | 20.67 | 21.46 | 2.56 | 2.63 | 2.59 | 0.61 | 0.61 | 0.61 | 0.12 | 0.12 | 2.93 |
| peregrine q8_0 | 22.35 | 23.02 | 22.97 | 2.84 | 2.81 | 2.77 | 0.65 | 0.65 | 0.65 | 0.13 | 0.13 | 3.15 |

### p50 latency, batch 1 (ms)

| engine / file | L16 b1 | L128 b1 | L512 b1 | L2000 b1 |
|---|---:|---:|---:|---:|
| llama_cpp f16 | 77.7 | 549.1 | 2330.6 | 11889.9 |
| llama_cpp q8_0 | 63.7 | 452.4 | 1997.0 | 12274.9 |
| peregrine f16 | 47.6 | 390.7 | 1626.5 | 8306.6 |
| peregrine q8_0 | 43.8 | 352.2 | 1531.5 | 7903.9 |

### Peregrine against the best equal-accuracy engine

| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |
|---|---:|---:|---|---:|
| L128 b32 | 2.77 | 2.21 | llama_cpp q8_0 | **1.25×** |
| L128 b8 | 2.81 | 2.21 | llama_cpp q8_0 | **1.28×** |
| mixed b32 | 3.15 | 2.46 | llama_cpp q8_0 | **1.28×** |
| L128 b1 | 2.84 | 2.21 | llama_cpp q8_0 | **1.28×** |
| L512 b8 | 0.65 | 0.50 | llama_cpp q8_0 | **1.28×** |
| L512 b1 | 0.65 | 0.50 | llama_cpp q8_0 | **1.30×** |
| L512 b32 | 0.65 | 0.50 | llama_cpp q8_0 | **1.31×** |
| L16 b32 | 22.97 | 15.95 | llama_cpp q8_0 | **1.44×** |
| L16 b8 | 23.02 | 15.92 | llama_cpp q8_0 | **1.45×** |
| L2000 b8 | 0.13 | 0.09 | llama_cpp f16 | **1.48×** |
| L16 b1 | 22.35 | 14.88 | llama_cpp q8_0 | **1.50×** |
| L2000 b1 | 0.13 | 0.08 | llama_cpp f16 | **1.51×** |

### Accuracy

| engine / file | cosine mean | cosine min | reference |
|---|---:|---:|---|
| llama_cpp f16 | 0.999682 | 0.999484 | torch-fp32, 200 texts |
| llama_cpp q8_0 | 0.999319 | 0.998576 | torch-fp32, 200 texts |
| peregrine f16 | 0.999682 | 0.999483 | torch-fp32, 200 texts |
| peregrine q8_0 | 0.999313 | 0.998573 | torch-fp32, 200 texts |

### Peregrine stage profile

| cell | request ms | tokenise ms | matmul | attention | other | barrier wait | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| L16 b1 | 43.8 | 0.08 | 90.0% | 2.1% | 7.8% | 2.6% | 336 |
| L16 b8 | 347.3 | 0.46 | 88.0% | 2.1% | 9.8% | 2.5% | 2688 |
| L16 b32 | 1392.4 | 1.38 | 88.3% | 2.1% | 9.6% | 2.2% | 10752 |
| L128 b1 | 352.2 | 0.44 | 86.0% | 4.5% | 9.5% | 2.1% | 2688 |
| L128 b8 | 2843.0 | 2.35 | 85.7% | 4.5% | 9.8% | 2.3% | 21504 |
| L128 b32 | 11549.6 | 8.48 | 85.4% | 4.5% | 10.1% | 2.9% | 86016 |
| L512 b1 | 1531.5 | 1.76 | 78.7% | 12.0% | 9.3% | 2.2% | 10752 |
| L512 b8 | 12375.2 | 8.87 | 78.6% | 12.2% | 9.2% | 2.2% | 86016 |
| L512 b32 | 48925.1 | 34.14 | 79.7% | 12.3% | 8.0% | 2.4% | 344064 |
| L2000 b1 | 7903.9 | 4.60 | 60.5% | 32.3% | 7.2% | 1.8% | 42000 |
| L2000 b8 | 63674.7 | 33.62 | 60.3% | 32.3% | 7.3% | 2.2% | 336000 |
| mixed b32 | 10166.3 | 7.42 | 82.5% | 7.9% | 9.6% | 2.3% | 74284 |

### Peregrine, cache on (repeated texts)

| cell | cache off texts/s | cache on, repeated texts/s | speed-up |
|---|---:|---:|---:|
| L16 b1 | 22.35 | 908.06 | 40.6× |
| L16 b8 | 23.02 | 137.55 | 6.0× |
| L16 b32 | 22.97 | 142.49 | 6.2× |
| L128 b1 | 2.84 | 724.66 | 255.4× |
| L128 b8 | 2.81 | 18.52 | 6.6× |
| L128 b32 | 2.77 | 18.40 | 6.6× |
| L512 b1 | 0.65 | 392.51 | 601.1× |
| L512 b8 | 0.65 | 4.21 | 6.5× |
| L512 b32 | 0.65 | 4.34 | 6.6× |
| L2000 b1 | 0.13 | 165.74 | 1305.1× |
| L2000 b8 | 0.13 | 0.84 | 6.7× |
| mixed b32 | 3.15 | 6.51 | 2.1× |

## Nemotron 3.5 Embed 1B

### texts/s (median of three rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | L2000 b1 | L2000 b8 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama_cpp q8_0 | 8.43 | 8.70 | 8.77 | 1.17 | 1.19 | 1.18 | 0.27 | 0.27 | 0.27 | 0.05 | 0.05 | 1.30 |
| peregrine q8_0 | 11.49 | 12.33 | 12.88 | 1.59 | 1.59 | 1.58 | 0.37 | 0.37 | 0.37 | 0.08 | 0.08 | 1.80 |

### p50 latency, batch 1 (ms)

| engine / file | L16 b1 | L128 b1 | L512 b1 | L2000 b1 |
|---|---:|---:|---:|---:|
| llama_cpp q8_0 | 113.8 | 857.9 | 3776.9 | 21456.4 |
| peregrine q8_0 | 80.3 | 628.2 | 2691.1 | 13133.2 |

### Peregrine against the best equal-accuracy engine

| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |
|---|---:|---:|---|---:|
| L128 b32 | 1.58 | 1.18 | llama_cpp q8_0 | **1.34×** |
| L128 b8 | 1.59 | 1.19 | llama_cpp q8_0 | **1.34×** |
| L16 b1 | 11.49 | 8.43 | llama_cpp q8_0 | **1.36×** |
| L128 b1 | 1.59 | 1.17 | llama_cpp q8_0 | **1.37×** |
| mixed b32 | 1.80 | 1.30 | llama_cpp q8_0 | **1.38×** |
| L512 b8 | 0.37 | 0.27 | llama_cpp q8_0 | **1.40×** |
| L512 b32 | 0.37 | 0.27 | llama_cpp q8_0 | **1.40×** |
| L512 b1 | 0.37 | 0.27 | llama_cpp q8_0 | **1.40×** |
| L16 b8 | 12.33 | 8.70 | llama_cpp q8_0 | **1.42×** |
| L16 b32 | 12.88 | 8.77 | llama_cpp q8_0 | **1.47×** |
| L2000 b1 | 0.08 | 0.05 | llama_cpp q8_0 | **1.62×** |
| L2000 b8 | 0.08 | 0.05 | llama_cpp q8_0 | **1.62×** |

### Accuracy

| engine / file | cosine mean | cosine min | reference |
|---|---:|---:|---|
| peregrine f16 | 1.000000 | 1.000000 | real-file test, llama.cpp F16 |
| peregrine q8_0 | 0.999859 | 0.999775 | real-file test, llama.cpp F16 |

### Peregrine stage profile

| cell | request ms | tokenise ms | matmul | attention | other | barrier wait | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| L16 b1 | 80.3 | 0.04 | 93.0% | 1.8% | 5.1% | 0.7% | 272 |
| L16 b8 | 648.3 | 0.14 | 93.0% | 2.0% | 5.0% | 0.8% | 1088 |
| L16 b32 | 2483.6 | 0.45 | 93.1% | 2.0% | 5.0% | 0.5% | 4352 |
| L128 b1 | 628.2 | 0.14 | 91.2% | 5.8% | 3.0% | 0.3% | 272 |
| L128 b8 | 5031.9 | 0.66 | 91.4% | 5.5% | 3.1% | 0.3% | 2176 |
| L128 b32 | 20204.6 | 2.08 | 91.4% | 5.4% | 3.2% | 0.4% | 8704 |
| L512 b1 | 2691.1 | 0.48 | 85.2% | 17.2% | 0.0% | 0.4% | 592 |
| L512 b8 | 21517.1 | 2.15 | 85.1% | 17.3% | 0.0% | 0.4% | 4736 |
| L512 b32 | 86179.8 | 7.78 | 85.2% | 17.3% | 0.0% | 0.4% | 18944 |
| L2000 b1 | 13133.2 | 1.14 | 66.2% | 57.2% | 0.0% | 0.4% | 1360 |
| L2000 b8 | 105260.7 | 7.52 | 66.7% | 56.3% | 0.0% | 0.4% | 10880 |
| mixed b32 | 17789.7 | 1.86 | 88.6% | 10.5% | 0.8% | 0.6% | 9488 |

### Peregrine, cache on (repeated texts)

| cell | cache off texts/s | cache on, repeated texts/s | speed-up |
|---|---:|---:|---:|
| L16 b1 | 11.49 | 652.34 | 56.8× |
| L16 b8 | 12.33 | 80.20 | 6.5× |
| L16 b32 | 12.88 | 81.41 | 6.3× |
| L128 b1 | 1.59 | 648.58 | 407.7× |
| L128 b8 | 1.59 | 10.48 | 6.6× |
| L128 b32 | 1.58 | 10.49 | 6.6× |
| L512 b1 | 0.37 | 516.48 | 1388.4× |
| L512 b8 | 0.37 | 2.47 | 6.6× |
| L512 b32 | 0.37 | 2.48 | 6.7× |
| L2000 b1 | 0.08 | 336.46 | 4427.1× |
| L2000 b8 | 0.08 | 0.51 | 6.7× |
| mixed b32 | 1.80 | 3.58 | 2.0× |

## Decisions, /v1/systemone

| state tokens | questions | peregrine q/s | llama.cpp q/s | ratio |
|---:|---:|---:|---:|---:|
| 64 | 1 | 179.22 | 97.45 | **1.84×** |
| 64 | 3 | 243.89 | 161.25 | **1.51×** |
| 64 | 10 | 310.09 | 203.20 | **1.53×** |
| 300 | 1 | 64.72 | 46.42 | **1.39×** |
| 300 | 3 | 116.37 | 86.49 | **1.35×** |
| 300 | 10 | 198.80 | 132.96 | **1.50×** |
| 1000 | 1 | 20.72 | 15.58 | **1.33×** |
| 1000 | 3 | 48.75 | 35.71 | **1.37×** |
| 1000 | 10 | 95.31 | 67.64 | **1.41×** |

## Accuracy, harness

| model | engine / file | cosine mean | cosine min | reference |
|---|---|---:|---:|---|
| minilm | fastembed fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| minilm | fastembed int8 | 0.981557 | 0.924134 | torch-fp32, 200 texts |
| minilm | llama_cpp f16 | 0.999998 | 0.999995 | torch-fp32, 200 texts |
| minilm | llama_cpp q8_0 | 0.999474 | 0.999124 | torch-fp32, 200 texts |
| minilm | onnxruntime fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| minilm | onnxruntime int8 | 0.981557 | 0.924134 | torch-fp32, 200 texts |
| minilm | openvino fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| minilm | peregrine f16 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| minilm | peregrine q8_0 | 0.999764 | 0.999521 | torch-fp32, 200 texts |
| minilm | tei fp32 | 0.999998 | 0.999995 | torch-fp32, 200 texts |
| bge-small | fastembed fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| bge-small | fastembed int8 | 0.972794 | 0.950791 | torch-fp32, 200 texts |
| bge-small | llama_cpp f32 | 0.999999 | 0.999996 | torch-fp32, 200 texts |
| bge-small | llama_cpp q8_0 | 0.999840 | 0.999740 | torch-fp32, 200 texts |
| bge-small | onnxruntime fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| bge-small | onnxruntime int8 | 0.972794 | 0.950791 | torch-fp32, 200 texts |
| bge-small | openvino fp32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| bge-small | peregrine f32 | 1.000000 | 1.000000 | torch-fp32, 200 texts |
| bge-small | peregrine q8_0 | 0.999928 | 0.999879 | torch-fp32, 200 texts |
| bge-small | tei fp32 | 0.999999 | 0.999996 | torch-fp32, 200 texts |
| bge-m3 | llama_cpp q8_0 | 0.999122 | 0.996776 | torch-fp32, 200 texts |
| bge-m3 | peregrine q8_0 | 0.999139 | 0.996888 | torch-fp32, 200 texts |
| embeddinggemma | llama_cpp q8_0 | 0.998746 | 0.991447 | onnx-fp32, 200 texts |
| embeddinggemma | peregrine q8_0 | 0.998763 | 0.991512 | onnx-fp32, 200 texts |
| qwen3-embed-0.6b | llama_cpp f16 | 0.999682 | 0.999484 | torch-fp32, 200 texts |
| qwen3-embed-0.6b | llama_cpp q8_0 | 0.999319 | 0.998576 | torch-fp32, 200 texts |
| qwen3-embed-0.6b | peregrine f16 | 0.999682 | 0.999483 | torch-fp32, 200 texts |
| qwen3-embed-0.6b | peregrine q8_0 | 0.999313 | 0.998573 | torch-fp32, 200 texts |

## Accuracy, real-file tests

| model | check | cosine min | cosine mean |
|---|---|---:|---:|
| minilm | f16: peregrine vs llama.cpp | 0.9999960 | 0.9999979 |
| minilm | f16: peregrine vs fp32 PyTorch | 0.9999996 | 0.9999998 |
| minilm | f16: llama.cpp vs fp32 PyTorch | 0.9999952 | 0.9999975 |
| minilm | q8_0: peregrine vs llama.cpp | 0.9995759 | 0.9997501 |
| minilm | q8_0: peregrine vs fp32 PyTorch | 0.9995207 | 0.9997637 |
| minilm | q8_0: llama.cpp vs fp32 PyTorch | 0.9991341 | 0.9994718 |
| bge-small | f32: peregrine vs llama.cpp | 0.9999963 | 0.9999987 |
| bge-small | f32: peregrine vs fp32 PyTorch | 1.0000000 | 1.0000000 |
| bge-small | f32: llama.cpp vs fp32 PyTorch | 0.9999963 | 0.9999987 |
| bge-small | q8_0: peregrine vs llama.cpp | 0.9998713 | 0.9999097 |
| bge-small | q8_0: peregrine vs fp32 PyTorch | 0.9998793 | 0.9999283 |
| bge-small | q8_0: llama.cpp vs fp32 PyTorch | 0.9997560 | 0.9998390 |
| bge-m3 | q8_0: peregrine vs llama.cpp | 0.9992233 | 0.9997594 |
| bge-m3 | q8_0: peregrine vs fp32 PyTorch | 0.9968882 | 0.9991393 |
| bge-m3 | q8_0: llama.cpp vs fp32 PyTorch | 0.9966748 | 0.9991207 |
| embeddinggemma | q8_0: peregrine vs llama.cpp Q8_0 (768-d) | 0.9990990 | 0.9996390 |
| embeddinggemma | q8_0: peregrine vs llama.cpp F16 (768-d) | 0.9979890 | 0.9994629 |
| embeddinggemma | q8_0: llama.cpp Q8_0 vs llama.cpp F16 (768-d) | 0.9980130 | 0.9994734 |
| embeddinggemma | q8_0: peregrine vs llama.cpp Q8_0 (256-d) | 0.9991660 | 0.9997002 |
| embeddinggemma | q8_0: peregrine vs llama.cpp F16 (256-d) | 0.9982180 | 0.9995477 |
| embeddinggemma | q8_0: llama.cpp Q8_0 vs llama.cpp F16 (256-d) | 0.9983980 | 0.9995551 |
| nemotron-3.5-embed-1b | f16: peregrine F16 vs llama.cpp F16 (2048-d) | 0.9999998 | 0.9999999 |
| nemotron-3.5-embed-1b | f16: peregrine F16 vs llama.cpp F16 (1024-d) | 0.9999996 | 0.9999999 |
| nemotron-3.5-embed-1b | f16: peregrine F16 vs llama.cpp F16 (512-d) | 0.9999992 | 0.9999999 |
| nemotron-3.5-embed-1b | q8_0: peregrine Q8_0 vs llama.cpp Q8_0 (2048-d) | 0.9999440 | 0.9999777 |
| nemotron-3.5-embed-1b | q8_0: peregrine Q8_0 vs llama.cpp F16 (2048-d) | 0.9997750 | 0.9998594 |
| nemotron-3.5-embed-1b | q8_0: llama.cpp Q8_0 vs llama.cpp F16 (2048-d) | 0.9997620 | 0.9998549 |
| nemotron-3.5-embed-1b | q8_0: peregrine Q8_0 vs llama.cpp Q8_0 (1024-d) | 0.9999440 | 0.9999774 |
| nemotron-3.5-embed-1b | q8_0: peregrine Q8_0 vs llama.cpp F16 (1024-d) | 0.9996860 | 0.9998428 |
| nemotron-3.5-embed-1b | q8_0: llama.cpp Q8_0 vs llama.cpp F16 (1024-d) | 0.9996800 | 0.9998384 |
| nemotron-3.5-embed-1b | q8_0: peregrine Q8_0 vs llama.cpp Q8_0 (512-d) | 0.9999400 | 0.9999766 |
| nemotron-3.5-embed-1b | q8_0: peregrine Q8_0 vs llama.cpp F16 (512-d) | 0.9995190 | 0.9998107 |
| nemotron-3.5-embed-1b | q8_0: llama.cpp Q8_0 vs llama.cpp F16 (512-d) | 0.9995070 | 0.9998062 |

## Where Peregrine is behind, and why

Two model families, and only at 128-512 tokens on the small BERT encoders. Every other measured
cell is at or above 1.05x the best equal-accuracy competitor.

### MiniLM

| cell | ratio | stage that explains it |
|---|---:|---|
| L512 b1 | **0.85x** (17.83 vs 20.87, OpenVINO fp32) | matmul 60.2%, attention 23.2%, other 16.7% |
| L512 b8 | **0.88x** (18.36 vs 20.92) | matmul 59.4%, attention 24.3%, other 16.3% |
| L512 b32 | **0.90x** (18.63 vs 20.60) | matmul 59.4%, attention 24.4%, other 16.3% |
| L128 b8 | **0.94x** (93.01 vs 98.91) | matmul 72.3%, attention 10.5%, other 17.3% |
| L128 b1 | **0.98x** (86.15 vs 87.93, ONNX Runtime fp32) | matmul 72.4%, attention 10.1%, other 17.6% |
| L128 b32 | **0.99x** (93.79 vs 95.09) | matmul 72.3%, attention 10.5%, other 17.2% |

The 512-token pass is the matrix kernel: peregrine's F16 row (17.83) and Q8_0 row (18.40) run
at the same speed, so the Q8_0 activation fill is not what is missing; OpenVINO's entire 47.5 ms
request is barely above Peregrine's matmul and attention alone (33.0 + 12.7 ms), and Peregrine
still pays its 9 ms of element-wise, embedding and pooling work on top. OpenVINO exports
the checkpoint to an IR and runs the CPU plugin's packed, register-blocked FMA GEMM (the S4.9
research note's "register-blocked FMA matrix kernel with weights packed at load"), while
Peregrine's tile is still the Q8_0/float panel kernel. The attention strip is the second lever
at this length (23-24% of the pass), and batching does not close the gap: OpenVINO keeps the
eight or 32 texts in one padded 512-token rectangle and its GEMM stays packed.
At 16 tokens the picture reversed: L16 b1 1.44x, L16 b8 1.10x and L16 b32 1.06x, against
0.87x / 0.74x in the 2026-11-11 standings. The element-wise kernels and the float path
(S4.9h, S4.9m) closed that short-batch gap; what is left is the long-text matrix chain.

### bge-small-en-v1.5

| cell | ratio | stage that explains it |
|---|---:|---|
| L512 b1 | **0.85x** (9.56 vs 11.22, OpenVINO fp32) | matmul 58.4%, attention 24.7%, other 16.9% |
| L512 b8 | **0.86x** (9.47 vs 10.97) | matmul 58.0%, attention 25.2%, other 16.8% |
| L512 b32 | **0.88x** (9.61 vs 10.88) | matmul 58.0%, attention 25.2%, other 16.8% |
| L128 b8 | **0.91x** (48.78 vs 53.70) | matmul 71.0%, attention 11.0%, other 18.0% |
| L128 b1 | **0.95x** (47.18 vs 49.59, ONNX Runtime fp32) | matmul 71.0%, attention 11.1%, other 17.9% |
| L16 b32 | **0.96x** (389.89 vs 407.20, fastembed fp32) | matmul 75.8%, attention 4.4%, other 19.8% |
| L128 b32 | **0.97x** (49.34 vs 50.80) | matmul 70.8%, attention 11.0%, other 18.1% |

The same two stages, with a little more attention: bge-small is a 12-layer encoder like MiniLM,
and the F32 and Q8_0 rows again run at the same speed (9.58 vs 9.56 at 512 b1), so the gap is
the packed GEMM plus the attention strip, not the activation quantisation. The losses narrow
with batching (0.85x at b1 to 0.88x at b32), which is the opposite of a padding explanation:
OpenVINO's advantage is the kernel, and it survives every batch size. At 16 tokens bge-small is
ahead at batch 1 (1.21x) and batch 8 (1.07x) and only 4% behind at batch 32.

### What is ahead, by model

- **bge-m3**: 1.18-1.53x over llama.cpp Q8_0 in every cell, including the two 2 000-token
  cells at **1.35x** (0.16 vs 0.12 texts/s). That cell was 0.72-0.73x in the 2026-11-11
  standings; the long-text attention work (S4.9g part 3) is why. Attention is still 36.5% of
  the 2 000-token pass and matmul 51.2%, and the F16 accuracy test keeps the vectors at
  0.9991 to fp32.
- **EmbeddingGemma 300M**: 1.30-1.51x over llama.cpp Q8_0, widest at 16 tokens (1.51x) and
  narrowest at 512 b1 (1.30x). The profile is 62-81% matmul, 13-21% attention.
- **Qwen3-Embedding 0.6B**: Q8_0 1.25-1.51x and F16 1.24-1.44x over the matching llama.cpp
  row; 90% of a 16-token pass is matmul and the float panel is the reason the F16 row is the
  slower of the two Peregrine rows.
- **Nemotron 3.5 Embed 1B**: 1.34-1.62x over llama.cpp Q8_0, the widest margin of the
  encoder family; attention is 57% of the 2 000-token pass in both engines (Peregrine 66.2%
  matmul / 57.2% attention), so the win is the matrix kernel and the encoder's strip attention,
  not a padding effect.
- **Mixed-length batch**: 1.23x on bge-small and 1.47x on MiniLM against the best competitor,
  and 1.6-4.5x against the padding engines (OpenVINO, ONNX Runtime, fastembed), because
  Peregrine never pads.
- **Decisions** (`/v1/systemone`): 1.33-1.84x llama.cpp on the same states and question
  counts, with llama.cpp at the fair setting the corrected table used (`-np 11 -c 16384
  -b 2048 -ub 512`, one slot per question of the largest request plus a spare); only that
  setting was measured this time. The smallest lead is one question on a 1 000-token state, the
  largest one question on a 64-token state.

## Accuracy, and what "equal accuracy" excluded

Every engine row is beside its cosine in the tables above. The two compact rows people will
compare are Peregrine Q8_0 and llama.cpp Q8_0: they are the same quantisation and land within
3e-4 of each other, and Peregrine is closer to the fp32 reference in all three BERT cases
(MiniLM 0.999764 vs 0.999474 mean, bge-small 0.999928 vs 0.999840, bge-m3 0.999139 vs
0.999122). The fp32/F16 rows are exact (1.000000) or within 1e-6.

**Excluded from the ratios:** the `int8`/`int8_avx2` ONNX exports served by ONNX Runtime and
fastembed. They are a different quantisation, not a faster version of Q8_0, and they measure
**cosine 0.981557 mean / 0.924134 min on MiniLM** and **0.972794 / 0.950791 on bge-small**.
Their speed is in the table with that cosine in the label; they are not the "best equal-accuracy
engine" in any cell. The fp32 rows of the same engines are included and are exactly 1.000000.

Two accuracy caveats are specific to the new models:

- **EmbeddingGemma** has no PyTorch reference on this machine (the checkpoint is licence-gated);
  its harness reference is the fp32 ONNX export, and both Q8_0 rows measure 0.99876 mean /
  0.9915 min to it. The repo's ignored test compares both engines to llama.cpp's own vectors:
  Peregrine 0.99946 mean and llama.cpp 0.99947 to the F16 reference at 768 d, and the two Q8_0
  files agree with each other at 0.99964. The relative rule above therefore includes llama.cpp:
  both are the same quantisation of the same checkpoint.
- **Nemotron 3.5 Embed 1B** has no fp32 PyTorch reference here either (its weights are 3.1 GB
  and were not downloaded). The repo's ignored test is the accuracy statement: Peregrine's F16
  vectors are 0.9999998 to llama.cpp's F16; the two Q8_0 files agree at 0.9999777 mean /
  0.999944 min, and each is 0.99986 mean to the F16 reference (Peregrine 0.9998594, llama.cpp
  0.9998549). llama.cpp is included as equal accuracy on the strength of that test, not a
  harness row.

The Qwen3 ignored test prints no cosine: it asserts bit-identical vectors alone, in a batch and
against llama.cpp's token ids, which is why the harness's 0.999682 (F16) and 0.999313 (Q8_0)
against torch-fp32 are the numbers in the table.

## Skips

- **Rerank**: `feat: rerank and classify for the bert family` is not on `origin/main` at
  `2a5e9f6` (it is `4196296` on `origin/s4.6b-bert-heads`), so the bge-reranker-v2-m3 cells are
  not measured here.
- **bge-m3 on OpenVINO / ONNX Runtime / fastembed / TEI**: only the Q8_0 GGUF and its tokenizer
  are on this machine, not the 2.3 GB fp32 XLM-R checkpoint; none of those engines reads a
  GGUF.
- **`L2000_b32`** (64 000 tokens) is above every model's 32 768-token batch cap and above
  llama.cpp's slot budget; not measured by any engine.
- **EmbeddingGemma, Qwen3 and Nemotron on the Python engines**: no ONNX/fp32 checkpoints for
  Qwen3 or Nemotron are on disk, and the brief's cell list for these three models is Peregrine
  and llama.cpp. An fp32 ONNX export of EmbeddingGemma does exist and was used to make the
  accuracy reference; it has no timing row (adding one after the three interleaved rounds would
  mix sessions), so its cells are llama.cpp-only by design and that is a limitation of this
  table, not a result.
- **Nemotron F16 timing** was not measured; the F16 file was downloaded only for the accuracy
  test. TEI reports no token usage, so its `tokens/s` uses the calibrated tokenizer count.

## Reproducing

The text sets and reference vectors are built once, then every row is one `embed.py` call:

```
python benchmarks/embed_texts.py --tokenizer-dir ~/models/embed/minilm/ref \
    --lengths 16,128,512 --per-length 128 --max-tokens 510 \
    --mixed-set 32 --mixed-seed 20261111 --out /var/tmp/embed-texts-s49l/minilm.json
# bge-small the same; bge-m3 / embeddinggemma / qwen3 with --lengths 16,128,512,2000
# --max-tokens 2010 --stream-words 20000; nemotron with --prefix "passage: "
# --mixed-prefixes "query: ,passage: "
python benchmarks/bert_fixtures.py --fixtures crates/peregrine-models/tests/fixtures/bert \
    --texts-dir /var/tmp/embed-texts-s49l
```

`benchmarks/proof_run.py` holds the row list and runs the missing rows in order:

```
for r in 1 2 3; do
  python benchmarks/proof_run.py --mode bench --round $r \
      --texts-dir /var/tmp/embed-texts-s49l \
      --out-dir benchmarks/results/bluefin-2026-11-12-s4.9l-proof/raw \
      --peregrine-binary /var/tmp/target-s49l/release/peregrine --max-seconds 300
done
python benchmarks/proof_run.py --mode accuracy ...   # 200-text accuracy rows
python benchmarks/proof_run.py --mode cache ...      # the cache-on table
```

Accuracy from the repo's own tests:

```
CARGO_TARGET_DIR=/var/tmp/target-s49l cargo test --release -p peregrine-models \
    --test embeddings --no-run
/var/tmp/target-s49l/release/deps/embeddings-* --ignored --nocapture --test-threads=1
```

The profile is `cellprof` built with the architecture features the bench crate does not enable
by default:

```
CARGO_TARGET_DIR=/var/tmp/target-s49l cargo build --release -p peregrine-bench \
    --example cellprof --features peregrine-models/arch-bert,peregrine-models/arch-gemma-embedding,\
peregrine-models/arch-qwen,peregrine-models/arch-llama,peregrine-models/arch-mistral3,\
peregrine-models/fmt-kquant,peregrine-models/fmt-legacy,peregrine-models/fmt-iq
PEREGRINE_PROF2=1 /var/tmp/target-s49l/release/examples/cellprof <gguf> <texts.json> \
    16,128,512,2000 1,8,32 4 --mixed 32
```

Decisions, three rounds per engine:

```
python benchmarks/decide.py --engine peregrine --binary /var/tmp/target-s49l/release/peregrine \
    --model ~/models/tinyopenjev-for-testing-Q8_0.gguf \
    --states /var/tmp/embed-texts-s49k/decide-states.json \
    --states-lengths 64,300,1000 --questions 1,3,10 --runs 7 --threads 4 --ctx 8192 --slots 4 \
    --round <r> --output-dir <dir>/decide
python benchmarks/decide.py --engine llama_cpp ... --llama-ctx 16384 --llama-slots 11 \
    --llama-batch 2048 --llama-ubatch 512 --round <r> --output-dir <dir>/decide
```

## Host noise

The three rounds were interleaved by row, but the host's 1-minute load average rose from 3.6 to
4.8 across them; `vmstat` showed 99% idle CPU between runs and the load came from a `find`
stuck in D state and `kswapd0`. Peregrine's own round-to-round spread is under 8% high in every
cell and under 3% low in all but four (Nemotron L16 b8 and L128 b8, bge-small L512 b8 and
bge-m3 L16 b1, where a single round was 8-22% low); the median absorbs those. A handful of rival short-text cells spread 20-35%
between rounds (fastembed and ONNX Runtime at 16 tokens, TEI at batch 1); where that happens the
same-round ratio is within 0.05x of the table's ratio, which is why the median is the reported
number and not the best round.

## Raw data

- `raw/<model>-<engine>-<precision>-r{1,2,3}.json` - the three rounds per row, with the exact
  command, versions, host state and per-cell latency, p50/p99 and texts.
- `raw/accuracy-<model>-<engine>-<precision>.json` - the harness cosine records.
- `raw/accuracy-real-file.json` - the ignored tests' printed cosines, parsed from
  `tests/real-file-tests.log`.
- `raw-cache/<model>-peregrine-<precision>-r{1,2,3}.json` - the cache-on, repeated-text rows.
- `profile/<model>-<precision>.jsonl` - one `cellprof` line per cell.
- `decide/decide-<engine>-r{1,2,3}.json` - the `/v1/systemone` rounds.
- `tests/real-file-tests.log` - the full output of the ignored real-file tests.
- Tables: `python benchmarks/proof_report.py --dir <this dir>`; the real-file accuracy JSON:
  `python benchmarks/real_file_accuracy.py --log tests/real-file-tests.log --out
  raw/accuracy-real-file.json`.
