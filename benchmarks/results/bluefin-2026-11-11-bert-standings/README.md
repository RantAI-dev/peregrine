# The bert family against every engine (bluefin, 2026-11-11)

The stage's claim is "fastest CPU engine for embedding models against every engine at equal
accuracy". Small BERT encoders are what OpenVINO, ONNX Runtime, fastembed and
text-embeddings-inference are built for, so this is the comparison that matters most. This
report measures all-MiniLM-L6-v2 and bge-small-en-v1.5 against every engine on this host,
and bge-m3 (XLM-R) against llama.cpp, with the accuracy of every row against the same fp32
PyTorch vectors.

## Machine and build

| | |
|---|---|
| Host | bluefin, AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads, AVX2+FMA, 13 GB RAM |
| Kernel | 7.2.7-200.fc44.x86_64 |
| Threads | **4** (one per physical core) for every engine |
| Load | 1-minute average 1.0-3.5 (median 2.8) during the runs: this agent's own process, `tailscaled` and swap reclaim, each under 4% CPU; no other benchmark ran. Round-to-round spread is under 2% except the two MiniLM short-batch cells noted below. |
| Toolchain | rustc 1.99.1, `cargo build --release -p peregrine` |

**What was built.** `origin/main` at `45291bc` does *not* contain either of the bert commits
(`feat: the bert family`, PR #178, and `perf: vector erf-GELU and LayerNorm for the bert
family`, PR #187). The measured binary is `origin/s4.9h-bert-elementwise` at `090170e`
merged locally with `origin/main` at `45291bc`, local merge commit `a67e2b5` (a measurement
merge only; it was not pushed). The tree is the bert family plus the S4.9a-g, S4.9h and
S4.14a work; no engine source was changed for this PR.

**Profiling.** The stage profile below comes from `crates/peregrine-bench/examples/cellprof`
with temporary phase timers added to `crates/peregrine-models/src/bert/mod.rs` (embedding
stage, attention, each LayerNorm, GELU, pooling, and the activation quantisation inside the
norm/GELU fills). The timers are off unless `PEREGRINE_BERT_PHASES` is set and are not part
of the measured binary: the throughput and accuracy rows were taken with the binary built
before the patch. The exact patch is `profile/bert-phases.patch`.

## Engines, versions and invocations

| Engine | Version | Invocation (4 threads) |
|---|---|---|
| Peregrine | merged build above, `peregrine 0.1.2` | `peregrine serve -m <gguf> -t 4 -c <model max tokens> --slots 40 --no-profile`; 40 slots so a 32-text request is admitted in one wave (bge-m3 2000-token texts still fit one slot each) |
| llama.cpp | v0.9.0-dev, commit `cd959ba` | `llama-server --embedding --pooling <mean,cls> -t 4 -c 16640 -b 1024 -ub 1024 -np 32 -fa auto` (bge-m3: `-np 8 -c 16448 -b 2048 -ub 2048`) |
| OpenVINO | `openvino 2026.4.1`, `sentence-transformers 6.2.0` | `servers/openvino_server.py`: `SentenceTransformer(backend="openvino")`, `INFERENCE_NUM_THREADS=4`, `PERFORMANCE_HINT=LATENCY` |
| ONNX Runtime | `onnxruntime 1.31.1` | `servers/ort_server.py`: plain ORT, `CPUExecutionProvider`, `intra_op_num_threads=4`, `ORT_ENABLE_ALL`; mean/cls pooling over `last_hidden_state` |
| fastembed | `fastembed 0.9.1` on ONNX Runtime 1.31.1 | `servers/fastembed_server.py`, `CPUExecutionProvider`, threads 4 |
| text-embeddings-inference | `1.9.4`, `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4` | `podman run --network host` with MKL/OMP/RAYON=4, `--dtype float32 --max-batch-tokens 16384` |

Python engines run from `/var/tmp/embed-venv`, outside the repository. No install was
needed for this run: all four were already on disk, including the TEI image.

## Method

- **Cells**: 16 / 128 / 512 tokens × batch 1, 8 and 32, plus one mixed-length batch of 32
  texts drawn from 8-512 tokens with the fixed seed `20261111` (3 516 tokens). bge-m3 also
  gets 2 000 tokens. One client, one request per wave for batch cells, 20 requests for
  batch 1, one request for the mixed cell; 0.5 s timed budget per cell.
- **Rounds**: three interleaved rounds, one engine and one model process at a time; the
  table is the median of the three per-round `texts/s`.
- **Texts**: `benchmarks/embed_texts.py` with each model's own tokenizer, so every engine
  sees byte-identical texts for a model. Costs per text: MiniLM and bge-small 16 / 128 /
  509-512 tokens; bge-m3 16-17 / 127-129 / 505-519 / 1 986-2 010; mixed 8-512. Nothing is
  truncated.
- **Batching**: Peregrine never pads (one row per token, texts share the weight passes).
  llama.cpp gives each text its own task/sequence and batches up to `-np` tasks in one
  weight pass, no padding. OpenVINO/sentence-transformers, ONNX Runtime (our server) and
  fastembed pad every request to the longest text in it; a mixed 32-text request therefore
  occupies 32 × 512 = 16 384 token positions for 3 516 real tokens. That padding is the
  reason their mixed cell collapses while their same-length cells do not.
- **Accuracy**: the 200 texts of `crates/peregrine-models/tests/fixtures/bert/texts.txt`
  and the corresponding `*_torch.f32` vectors (fp32 sentence-transformers, cosine in the
  table). Every engine embeds those texts; `benchmarks/bert_fixtures.py` converts the
  fixture vectors into the harness's reference JSON.
- **Equal accuracy**: the int8 ONNX exports are at cosine 0.973 (bge-small) and 0.982
  (MiniLM), well below Peregrine's Q8_0 at 0.9998 and the fp32 rows at 1.0000. Their speed
  is shown, but the "best equal-accuracy engine" column excludes anything more than
  0.0005 below the Peregrine row it is compared with. Peregrine Q8_0 and llama.cpp Q8_0 are
  the same quantisation; the ONNX `int8`/`int8_avx2` files are a different one and are
  labelled with their cosine in the speed table.
- **Percentages**: matmul and attention come from `StepStats`; element-wise, quantise,
  embedding and pool from the temporary timers; barrier wait from `PEREGRINE_PROF2`.
  Quantise is the activation fill inside LayerNorm/GELU, so it is shown separately and
  subtracted from element-wise. HTTP is the residual of the server p50 against the direct
  engine call plus tokenisation; on the two 2 000-token bge-m3 cells the direct call and
  the server differ by ~2%, so the residual is reported as 0 there. Element-wise is
  LayerNorm + GELU net of the activation quantisation: on a Q8_0 file the fill dominates
  that phase, which is why element-wise is near 0 and quantise is 15-18% for bge-small.

## Results

Generated from `raw/` and `profile/` by `benchmarks/bert_report.py --dir <this dir>`.

### MiniLM (all-MiniLM-L6-v2)

#### texts/s (median of three rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fastembed fp32 | 243.53 | 583.09 | 696.36 | 85.86 | 92.16 | 74.95 | 15.81 | 13.35 | 12.88 | 13.31 |
| fastembed int8 (0.982) | 232.01 | 398.19 | 629.60 | 85.03 | 89.53 | 86.36 | 15.01 | 14.00 | 13.84 | 14.39 |
| llama_cpp f16 | 244.24 | 408.56 | 429.80 | 57.33 | 55.36 | 55.99 | 13.13 | 12.08 | 12.13 | 59.12 |
| llama_cpp q8_0 | 279.86 | 467.88 | 484.72 | 57.85 | 63.60 | 63.63 | 14.88 | 13.52 | 13.64 | 66.72 |
| onnxruntime fp32 | 249.36 | 588.76 | 695.71 | 87.54 | 90.58 | 73.54 | 15.74 | 13.23 | 12.79 | 12.77 |
| onnxruntime int8 (0.982) | 284.14 | 609.63 | 715.70 | 89.74 | 94.65 | 85.46 | 14.91 | 13.94 | 13.70 | 14.20 |
| openvino fp32 | 201.85 | 543.03 | 669.91 | 80.24 | 98.48 | 94.18 | 21.00 | 20.79 | 20.20 | 21.82 |
| peregrine f16 | 366.59 | 515.03 | 517.92 | 74.27 | 79.56 | 79.93 | 16.09 | 16.28 | 16.39 | 84.27 |
| peregrine q8_0 | 371.13 | 526.79 | 558.81 | 69.61 | 74.10 | 74.66 | 15.05 | 15.27 | 15.22 | 78.71 |
| tei fp32 | 157.50 | 221.32 | 262.05 | 36.08 | 38.16 | 37.13 | 4.48 | 4.18 | 4.10 | 9.93 |

#### p50 latency, batch 1 (ms)

| engine / file | L16 b1 | L128 b1 | L512 b1 |
|---|---:|---:|---:|
| fastembed fp32 | 3.8 | 11.5 | 63.2 |
| fastembed int8 | 4.3 | 11.3 | 66.8 |
| llama_cpp f16 | 4.0 | 17.3 | 75.5 |
| llama_cpp q8_0 | 3.5 | 14.8 | 66.7 |
| onnxruntime fp32 | 3.7 | 11.3 | 63.3 |
| onnxruntime int8 | 3.1 | 11.1 | 66.2 |
| openvino fp32 | 4.9 | 12.3 | 46.9 |
| peregrine f16 | 2.5 | 13.4 | 61.9 |
| peregrine q8_0 | 2.6 | 14.2 | 66.5 |
| tei fp32 | 6.0 | 26.9 | 222.7 |

#### Peregrine profile

| cell | HTTP ms (residual) | engine ms | tokenise ms | matmul | attention | element-wise | quantise | embedding | barrier wait | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L16 b1 | 0.98 | 1.5 | 0.02 | 77.4% | 5.0% | 10.2% | 2.1% | 2.3% | 5.2% | 0.0% |
| L16 b8 | 2.17 | 13.1 | 0.08 | 79.2% | 4.0% | 8.4% | 2.3% | 3.6% | 4.2% | 0.0% |
| L16 b32 | 0.00 | 63.3 | 0.26 | 77.8% | 5.1% | 8.2% | 3.0% | 2.6% | 5.5% | 0.0% |
| L128 b1 | 0.00 | 15.6 | 0.11 | 72.6% | 9.7% | 5.5% | 4.5% | 3.5% | 3.0% | 1.0% |
| L128 b8 | 0.72 | 99.1 | 0.58 | 74.0% | 10.0% | 7.5% | 3.2% | 2.3% | 2.5% | 0.4% |
| L128 b32 | 15.56 | 383.1 | 1.44 | 74.9% | 9.6% | 7.6% | 3.0% | 2.1% | 2.1% | 0.6% |
| L512 b1 | 0.00 | 62.2 | 0.25 | 62.4% | 22.5% | 4.0% | 5.0% | 2.5% | 2.5% | 1.1% |
| L512 b8 | 5.65 | 484.2 | 1.46 | 61.2% | 23.9% | 4.8% | 4.5% | 1.7% | 2.6% | 1.2% |
| L512 b32 | 43.72 | 1903.4 | 5.47 | 61.8% | 23.3% | 4.9% | 4.4% | 1.8% | 1.9% | 1.9% |
| mixed b32 | 13.34 | 364.7 | 1.28 | 68.7% | 15.8% | 6.5% | 3.6% | 2.0% | 2.3% | 0.9% |

#### Peregrine against the best equal-accuracy engine

| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |
|---|---:|---:|---|---:|
| L16 b32 | 517.92 | 696.36 | fastembed fp32 | **0.74×** |
| L512 b1 | 16.09 | 21.00 | openvino fp32 | **0.77×** |
| L512 b8 | 16.28 | 20.79 | openvino fp32 | **0.78×** |
| L128 b8 | 79.56 | 98.48 | openvino fp32 | **0.81×** |
| L512 b32 | 16.39 | 20.20 | openvino fp32 | **0.81×** |
| L128 b1 | 74.27 | 87.54 | onnxruntime fp32 | **0.85×** |
| L128 b32 | 79.93 | 94.18 | openvino fp32 | **0.85×** |
| L16 b8 | 515.03 | 588.76 | onnxruntime fp32 | **0.87×** |
| mixed b32 | 84.27 | 59.12 | llama_cpp f16 | **1.43×** |
| L16 b1 | 366.59 | 249.36 | onnxruntime fp32 | **1.47×** |

### bge-small-en-v1.5

#### texts/s (median of three rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fastembed fp32 | 155.90 | 350.74 | 407.48 | 49.16 | 48.63 | 39.40 | 8.28 | 7.16 | 7.24 | 7.34 |
| fastembed int8 (0.973) | 189.61 | 280.80 | 404.69 | 48.13 | 49.19 | 46.86 | 8.47 | 8.32 | 8.48 | 8.38 |
| llama_cpp f32 | 150.94 | 312.63 | 330.39 | 42.42 | 40.36 | 40.52 | 9.08 | 8.21 | 8.29 | 41.07 |
| llama_cpp q8_0 | 172.24 | 263.00 | 269.99 | 35.14 | 32.41 | 32.45 | 7.58 | 6.84 | 6.91 | 33.87 |
| onnxruntime fp32 | 157.16 | 278.93 | 406.49 | 48.96 | 47.82 | 39.30 | 8.37 | 7.17 | 7.19 | 7.38 |
| onnxruntime int8 (0.973) | 224.66 | 362.72 | 408.20 | 48.76 | 49.43 | 47.55 | 8.52 | 8.39 | 8.42 | 8.26 |
| openvino fp32 | 142.16 | 344.67 | 400.19 | 45.59 | 53.49 | 50.70 | 11.22 | 10.90 | 10.75 | 11.21 |
| peregrine f32 | 205.35 | 272.29 | 281.20 | 40.87 | 42.24 | 42.39 | 8.38 | 8.44 | 8.42 | 43.54 |
| peregrine q8_0 | 230.10 | 285.51 | 294.96 | 36.22 | 36.58 | 36.84 | 7.55 | 7.59 | 7.61 | 39.22 |
| tei fp32 | 84.19 | 100.77 | 123.81 | 17.62 | 18.78 | 18.90 | 2.62 | 2.27 | 2.14 | 4.93 |

#### p50 latency, batch 1 (ms)

| engine / file | L16 b1 | L128 b1 | L512 b1 |
|---|---:|---:|---:|
| fastembed fp32 | 6.2 | 20.3 | 120.1 |
| fastembed int8 | 4.5 | 20.7 | 118.5 |
| llama_cpp f32 | 6.5 | 23.5 | 110.7 |
| llama_cpp q8_0 | 5.7 | 28.3 | 130.8 |
| onnxruntime fp32 | 6.7 | 20.1 | 119.6 |
| onnxruntime int8 | 4.4 | 20.4 | 117.6 |
| openvino fp32 | 6.9 | 21.7 | 88.8 |
| peregrine f32 | 4.7 | 24.3 | 119.3 |
| peregrine q8_0 | 4.2 | 27.5 | 132.5 |
| tei fp32 | 11.7 | 53.7 | 381.6 |

#### Peregrine profile

| cell | HTTP ms (residual) | engine ms | tokenise ms | matmul | attention | element-wise | quantise | embedding | barrier wait | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L16 b1 | 0.95 | 3.3 | 0.02 | 71.0% | 4.9% | 0.6% | 18.2% | 0.4% | 5.3% | 0.0% |
| L16 b8 | 2.22 | 25.5 | 0.10 | 74.1% | 4.1% | 0.0% | 17.5% | 0.9% | 4.2% | 0.0% |
| L16 b32 | 5.68 | 102.3 | 0.26 | 74.0% | 4.3% | 0.0% | 17.6% | 0.5% | 4.0% | 0.0% |
| L128 b1 | 1.10 | 26.3 | 0.08 | 71.6% | 8.3% | 0.0% | 17.1% | 0.8% | 2.5% | 0.0% |
| L128 b8 | 13.25 | 204.9 | 0.41 | 71.3% | 8.5% | 0.0% | 17.5% | 0.4% | 2.2% | 0.1% |
| L128 b32 | 36.06 | 830.8 | 1.44 | 71.2% | 8.5% | 0.0% | 17.5% | 0.3% | 2.5% | 0.0% |
| L512 b1 | 2.90 | 129.3 | 0.25 | 60.1% | 21.2% | 0.0% | 15.5% | 0.6% | 1.8% | 0.8% |
| L512 b8 | 37.89 | 1014.8 | 1.45 | 59.8% | 21.6% | 0.0% | 15.7% | 0.3% | 1.9% | 0.7% |
| L512 b32 | 124.75 | 4076.2 | 5.44 | 59.8% | 21.6% | 0.0% | 15.7% | 0.3% | 2.3% | 0.4% |
| mixed b32 | 30.86 | 783.4 | 1.28 | 65.9% | 14.6% | 0.0% | 16.6% | 0.3% | 2.6% | 0.0% |

#### Peregrine against the best equal-accuracy engine

| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |
|---|---:|---:|---|---:|
| L512 b1 | 7.55 | 11.22 | openvino fp32 | **0.67×** |
| L128 b8 | 36.58 | 53.49 | openvino fp32 | **0.68×** |
| L512 b8 | 7.59 | 10.90 | openvino fp32 | **0.70×** |
| L512 b32 | 7.61 | 10.75 | openvino fp32 | **0.71×** |
| L16 b32 | 294.96 | 407.48 | fastembed fp32 | **0.72×** |
| L128 b32 | 36.84 | 50.70 | openvino fp32 | **0.73×** |
| L128 b1 | 36.22 | 49.16 | fastembed fp32 | **0.74×** |
| L16 b8 | 285.51 | 350.74 | fastembed fp32 | **0.81×** |
| mixed b32 | 39.22 | 41.07 | llama_cpp f32 | **0.95×** |
| L16 b1 | 230.10 | 172.24 | llama_cpp q8_0 | **1.34×** |

### bge-m3 (XLM-R large)

Only Peregrine and llama.cpp: no fp32 checkpoint is on this machine (see Skips).

#### texts/s (median of three rounds)

| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | L2000 b1 | L2000 b8 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama_cpp q8_0 | 20.59 | 24.52 | 25.08 | 3.21 | 2.67 | 2.66 | 0.69 | 0.56 | 0.56 | 0.12 | 0.12 | 3.03 |
| peregrine q8_0 | 27.86 | 27.42 | 27.44 | 3.27 | 3.29 | 3.28 | 0.73 | 0.73 | 0.73 | 0.09 | 0.09 | 3.63 |

#### p50 latency, batch 1 (ms)

| engine / file | L16 b1 | L128 b1 | L512 b1 | L2000 b1 |
|---|---:|---:|---:|---:|
| llama_cpp q8_0 | 47.3 | 311.2 | 1441.5 | 8341.0 |
| peregrine q8_0 | 35.5 | 305.5 | 1374.9 | 11476.4 |

#### Peregrine profile

| cell | HTTP ms (residual) | engine ms | tokenise ms | matmul | attention | element-wise | quantise | embedding | barrier wait | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L16 b1 | 1.69 | 33.8 | 0.04 | 80.8% | 2.9% | 4.8% | 8.6% | 0.1% | 3.5% | 0.0% |
| L16 b8 | 5.34 | 286.0 | 0.16 | 82.4% | 2.7% | 3.7% | 8.4% | 0.2% | 1.9% | 0.8% |
| L16 b32 | 21.47 | 1143.5 | 0.45 | 83.0% | 2.6% | 3.4% | 8.3% | 0.1% | 1.5% | 1.2% |
| L128 b1 | 2.80 | 302.5 | 0.17 | 81.4% | 4.9% | 3.1% | 7.9% | 0.2% | 0.7% | 1.9% |
| L128 b8 | 12.19 | 2418.6 | 0.63 | 81.2% | 4.9% | 3.2% | 8.0% | 0.1% | 0.8% | 1.8% |
| L128 b32 | 94.69 | 9670.0 | 2.05 | 81.2% | 4.9% | 3.2% | 8.0% | 0.1% | 0.8% | 1.8% |
| L512 b1 | 0.00 | 1376.7 | 0.47 | 71.9% | 14.4% | 2.8% | 7.8% | 0.1% | 0.8% | 2.2% |
| L512 b8 | 22.55 | 10967.1 | 2.21 | 71.9% | 14.4% | 2.7% | 7.8% | 0.1% | 0.8% | 2.2% |
| L512 b32 | 17.91 | 44073.9 | 7.78 | 71.8% | 14.6% | 2.7% | 7.8% | 0.1% | 0.9% | 2.2% |
| L2000 b1 | 0.00 | 11729.4 | 1.29 | 33.0% | 60.5% | 1.2% | 3.8% | 0.1% | 0.5% | 1.0% |
| L2000 b8 | 0.00 | 93586.3 | 7.76 | 32.9% | 60.5% | 1.3% | 3.8% | 0.0% | 0.4% | 1.1% |
| mixed b32 | 11.81 | 8809.2 | 1.95 | 76.6% | 9.5% | 3.1% | 7.9% | 0.1% | 0.9% | 2.0% |

#### Peregrine against the best equal-accuracy engine

| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |
|---|---:|---:|---|---:|
| L2000 b1 | 0.09 | 0.12 | llama_cpp q8_0 | **0.72×** |
| L2000 b8 | 0.09 | 0.12 | llama_cpp q8_0 | **0.73×** |
| L128 b1 | 3.27 | 3.21 | llama_cpp q8_0 | **1.02×** |
| L512 b1 | 0.73 | 0.69 | llama_cpp q8_0 | **1.05×** |
| L16 b32 | 27.44 | 25.08 | llama_cpp q8_0 | **1.09×** |
| L16 b8 | 27.42 | 24.52 | llama_cpp q8_0 | **1.12×** |
| mixed b32 | 3.63 | 3.03 | llama_cpp q8_0 | **1.20×** |
| L128 b8 | 3.29 | 2.67 | llama_cpp q8_0 | **1.23×** |
| L128 b32 | 3.28 | 2.66 | llama_cpp q8_0 | **1.23×** |
| L512 b32 | 0.73 | 0.56 | llama_cpp q8_0 | **1.29×** |
| L512 b8 | 0.73 | 0.56 | llama_cpp q8_0 | **1.30×** |
| L16 b1 | 27.86 | 20.59 | llama_cpp q8_0 | **1.35×** |

## Accuracy

Cosine to the fp32 PyTorch (sentence-transformers) vectors, 200 texts of 16-428 tokens.

| model | engine / file | cosine mean | cosine min |
|---|---|---:|---:|
| bge-m3 | llama_cpp q8_0 | 0.999122 | 0.996671 |
| bge-m3 | peregrine q8_0 | 0.999139 | 0.996888 |
| bge-small | fastembed fp32 | 1.000000 | 1.000000 |
| bge-small | fastembed int8 | 0.972794 | 0.950791 |
| bge-small | llama_cpp f32 | 0.999999 | 0.999996 |
| bge-small | llama_cpp q8_0 | 0.999840 | 0.999740 |
| bge-small | onnxruntime fp32 | 1.000000 | 1.000000 |
| bge-small | onnxruntime int8 | 0.972794 | 0.950791 |
| bge-small | openvino fp32 | 1.000000 | 1.000000 |
| bge-small | peregrine f32 | 1.000000 | 1.000000 |
| bge-small | peregrine q8_0 | 0.999841 | 0.999755 |
| bge-small | tei fp32 | 0.999999 | 0.999996 |
| minilm | fastembed fp32 | 1.000000 | 1.000000 |
| minilm | fastembed int8 | 0.981557 | 0.924134 |
| minilm | llama_cpp f16 | 0.999998 | 0.999995 |
| minilm | llama_cpp q8_0 | 0.999472 | 0.999134 |
| minilm | onnxruntime fp32 | 1.000000 | 1.000000 |
| minilm | onnxruntime int8 | 0.981557 | 0.924134 |
| minilm | openvino fp32 | 1.000000 | 1.000000 |
| minilm | peregrine f16 | 1.000000 | 1.000000 |
| minilm | peregrine q8_0 | 0.999493 | 0.999138 |
| minilm | tei fp32 | 0.999998 | 0.999995 |

The int8 rows are a different quantisation and are 0.97-0.98, not a faster version of the
Q8_0 row. Peregrine Q8_0 (0.9995-0.9998) and llama.cpp Q8_0 (the same numbers within 3e-4)
are the compact rows; the fp32/F16 rows are the exactly-equal-accuracy rows.

## Where Peregrine is behind, and why

The three widest gaps are all bge-small at 128-512 tokens, where OpenVINO's packed fp32
GEMM and Peregrine's per-row activation quantisation meet:

1. **bge-small L512 b1: 7.55 vs 11.22 texts/s, 0.67× (OpenVINO fp32).** The profile puts
   60.1% of the pass in matmul, 21.2% in attention and 15.5% in quantise. The Q8_0 file
   re-quantises every activation row for every projection (that is the 15.5%), and the
   512-token text is a pure single-stream GEMM chain. The F32 row, which has no quantise,
   is 8.38 vs 11.22 (0.75×), so the larger part is the matrix kernel itself: OpenVINO
   exports the checkpoint to its IR and runs its CPU plugin's packed, register-blocked FMA
   GEMM (the reading in `docs/design/s4.9-sota.md` §0: "a register-blocked FMA matrix
   kernel with weights packed at load"), while Peregrine's Q8_0 tile is an int8 weight
   times f32 activation kernel. What OpenVINO does *not* do better here is batching: at
   batch 1 there is nothing to pad.
2. **bge-small L128 b8: 36.58 vs 53.49, 0.68× (OpenVINO fp32).** Matmul 71.3%, quantise
   17.5%, attention 8.5%. Here quantise is 36 ms of a 205 ms request: every one of the
   eight texts' rows is quantised again at each LayerNorm/GELU fill before the next
   projection, while OpenVINO keeps the whole padded 8 × 128 rectangle in fp32. The F32
   row is 42.24 vs 53.49 (0.79×), so the packed GEMM is the rest of the gap. This is the
   cell where the element-wise/quantise work is the largest single lever after the matrix
   kernel.
3. **bge-small L512 b8: 7.59 vs 10.90, 0.70× (OpenVINO fp32).** Matmul 59.8%, attention
   21.6%, quantise 15.7%. Batching does not close the gap: OpenVINO's eight 512-token
   texts are one padded rectangle and its GEMM stays packed, while Peregrine's attention
   share grows with length and its quantise stays at 15-16%.

The other two models' worst cells have different causes:

- **MiniLM L16 b32: 517.92 vs 696.36, 0.74× (fastembed fp32).** The 16-token pass is 77.8%
  matmul, 8.2% element-wise, 5.5% barrier wait and 3.0% quantise. There is no padding to
  exploit at 16 tokens, so this is a pure kernel/batching comparison: the ONNX Runtime rows
  turn a 32 × 16 batch into one GEMM per layer (the same MLAS-style packed kernel the design
  note describes), while Peregrine's engine admits the batch into slots and pays its
  quantise and fork-join barriers on a 16-token pass where they are a large fraction. MiniLM
  is ahead at batch 1 (1.47×) and in the mixed batch (1.43×), so the loss is specific to
  large short-text batches, not to the model.
- **bge-m3 L2000 b1/b8: 0.72-0.73× (llama.cpp Q8_0).** Attention is 60.5% of the pass and
  matmul 33%; llama.cpp runs `-fa auto`, i.e. its fused flash-attention path
  (`ggml_compute_forward_flash_attn_ext_tiled` in `ggml/src/ggml-cpu/ops.cpp`, selected in
  `ggml_compute_forward_flash_attn_ext_f16`), which keeps K/V in F16 and walks query rows in
  cache-sized tiles with an online softmax, never storing a full score row. Peregrine's
  bidirectional strip (`crates/peregrine-core/src/blocks/bidir.rs::attend_texts`) scores
  each row's band into an f32 buffer, then takes its max, exponentials and mix. At 2 000
  tokens the f32 score row and the extra pass are the whole difference. At every length up
  to 512 tokens Peregrine is ahead of llama.cpp (1.05-1.35×), so this is specifically the
  long-text attention path.

Where Peregrine wins: the mixed-length batch, by 1.43× on MiniLM against llama.cpp and
3.9× against the padding engines (Peregrine 84.3 against OpenVINO 21.8, ONNX Runtime 12.8
and fastembed 13.3 texts/s at 32 texts of 8-512 tokens), and by 3.5× on bge-small (39.2
against 11.2 / 7.2 / 7.3), because it computes no padding; MiniLM batch 1 at 16 tokens
(1.47×); and every bge-m3 cell up to 512 tokens.

## Skips

- **bge-m3 `L2000_b32`** (64 000 tokens) exceeds the model's 16 384-token batch cap; skipped
  by rule. `L2000_b8` (16 000 tokens) was measured.
- **bge-m3 on OpenVINO / ONNX Runtime / fastembed / TEI**: only the Q8_0 GGUF and the
  tokenizer are on disk, not the fp32 XLM-R checkpoint (2.3 GB). None of these engines can
  load the GGUF, so they were not run; a 2.3 GB download and OpenVINO IR conversion was
  outside the time budget. Their numbers on MiniLM and bge-small are the comparison.
- **bge-m3 on llama.cpp at 2 000 tokens** needs `-np 8` and `-c 16448` (32 slots would give
  a 642-token slot); the row records that setting. Batch 32 at 512 tokens therefore runs
  in four waves on llama.cpp, and that is what the `L512_b32` row shows.
- No engine was skipped for install time: every environment from the S4.2 baseline was
  still on disk.

## Raw data and reproduction

- `raw/<model>-<engine>-<precision>-r{1,2,3}.json` — the three rounds per row, with the
  exact command, versions, host state and per-cell latency. `raw/accuracy-*.json` — the
  cosine records. `raw/*-skips.json` — the skipped cells and why.
- `profile/<model>-<precision>.jsonl` — one line per cell: `StepStats`, the
  `PEREGRINE_PROF2` buckets and the temporary phase counters. `profile/bert-phases.patch` —
  the exact timers and the `cellprof` slot option.
- Texts and accuracy fixtures: `benchmarks/embed_texts.py` (with `--mixed-set 32
  --mixed-seed 20261111`, and `--stream-words 20000` for the 2 000-token set) and
  `benchmarks/bert_fixtures.py`.
- Tables: `python benchmarks/bert_report.py --dir <this dir>`.

The measurement loop, per round and per model, was
`python benchmarks/embed.py --select <model>:<engine> --lengths 16,128,512[,2000]
--batches 1,8,32 --clients 1 --budget 0.5 --max-requests 20 --max-big-requests 2
--mixed-cell ... --round <r>`; the accuracy pass is the same command with
`--mode accuracy --acc-batch 16`.
