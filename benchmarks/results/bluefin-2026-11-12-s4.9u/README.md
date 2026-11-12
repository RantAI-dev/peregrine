# S4.9u on bluefin: nomic-bert and jina-bert-v2 against every engine (2026-11-12/13)

The two newest encoder families had only ever been timed against llama.cpp, on a noisy
shared VM: nomic-embed-text v1.5 at 1.31–1.65× and jina-embeddings-v2-base-en at
1.04–1.50× (512 tokens its weakest cell). This run puts them through the same proof the
other four embedding families had — every engine on this host that can load the model, the
same cells, three interleaved rounds, the result cache off, accuracy beside speed — then
takes apart the cells behind 1.0 and closes the part the same bits allow.

The short answer: **Peregrine is the fastest equal-accuracy engine in 21 of the 36 cells**
these two models add (13 of 36 before this run's fixes), **1.71×/1.67× llama.cpp on the mixed
batch** and **1.04–1.11×** at 512 tokens. Fifteen cells remain behind: the 128-token cells
and the 16-token batch cells, where ONNX Runtime's MLAS GEMM is 12 % faster at 128 rows and
the matrix product is 90 % of the pass. That is the account's remaining item; the
element-wise block and jina's ALiBi bias, which the account also named, were closed with
every vector bit-identical.

## Machine, build and models

| | |
|---|---|
| machine | bluefin: AMD Ryzen 3 PRO 4350G (Renoir, Zen 2), 4 cores / 8 threads, 128 KiB L1d + 512 KiB L2 per core, 4 MiB L3, 13 GB, Fedora 44, kernel 7.2.7-200.fc44, governor `powersave` |
| flags | `fma avx avx2` (no AVX-512; tier `x86_64-avx2`) |
| threads | 4 everywhere (one per physical core) |
| before | `origin/main` at `773ebf8` — the "before" binary, and the standings before Part 3 |
| after | this branch on `773ebf8` + the three commits below |
| build | `cargo build --release -p peregrine`; `cellprof` from `peregrine-bench` with `--features peregrine-models/arch-bert,peregrine-models/arch-gemma-embedding,peregrine-models/arch-qwen,peregrine-models/arch-llama,peregrine-models/arch-mistral3,peregrine-models/arch-nomic-bert,peregrine-models/arch-jina-bert` |
| rustc | 1.99.1 |
| host load | **not quiet by the strict definition**: 1-minute load average 4.2–8.5 (median 5.8) in the before window and 5.1–6.9 (median 5.8) in the after window, but 97–100 % CPU idle between rows (`vmstat`): the load is a 14-hour D-state `find` and 6–7 GB of swap in use, the same condition S4.9l/S4.9p/S4.9r recorded. Every table is interleaved and reported as a median for that reason. |

### Model files (size and SHA-256 of every file used)

| model / file | source | bytes | SHA-256 |
|---|---|---:|---|
| nomic-embed-text-v1.5 F16 GGUF | `nomic-ai/nomic-embed-text-v1.5-GGUF` | 274 290 560 | `f7af6f66802f4df86eda10fe9bbcfc75c39562bed48ef6ace719a251cf1c2fdb` |
| nomic-embed-text-v1.5 Q8_0 GGUF | same | 146 146 432 | `3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7` |
| nomic ONNX fp32 (`model.onnx`) | `nomic-ai/nomic-embed-text-v1.5` | 547 310 275 | `147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965` |
| nomic ONNX int8 (`model_quantized.onnx`) | same | 137 296 292 | `b4342336debaea79de872370664b0aaeb67dea4605513d00ee236ea871a81f27` |
| nomic fp32 safetensors (TEI) | same | 546 938 168 | `9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c` |
| jina-embeddings-v2-base-en FP16 GGUF | `gpustack/jina-embeddings-v2-base-en-GGUF` | 274 367 136 | `8d29fa0c9b806ae5c165eaf1d8053820760c5d4a3b4f57dd203c426c4680c5f0` |
| jina-embeddings-v2-base-en Q8_0 GGUF | same | 146 218 656 | `123b13ec105de8e0d2becd7dca4f4c46cef3a535f6de9ff1b6ade3fbbf6f5184` |
| jina ONNX fp32 (`model.onnx`) | `Xenova/jina-embeddings-v2-base-en` | 547 362 259 | `f345c41919f5398da5aa3c300faed6ad58d93f96c0a2d990bc82e7012f64b6c8` |
| jina ONNX int8 (`model_int8.onnx`) | same | 137 390 704 | `5a01f42d114a6bb92228d1a47c4cccef79cfd931791a499134db80e177a3a93d` |
| jina fp32 reference ONNX | `jinaai/jina-embeddings-v2-base-en` (`model.onnx`) | 547 390 322 | `a6bccce798906f269ee6990d35b8a516390a9593cde824de2e6b9d087b07fa4d` |
| jina safetensors (TEI) | `jinaai/jina-embeddings-v2-base-en` | 274 757 256 | `6b70f1386f05b9703ea4edf7f1550a8925399f9580e4cc754cc099efc1e736d8` |

The nomic files are the same SHA-256 the family's reference note records. The jina
safetensors is stored at half precision by its publisher; TEI runs it at `--dtype float32`
and lands at cosine 1.000000 against the fp32 ONNX reference.

Every engine loaded the model natively. **No engine was excluded for support**: nomic's
custom-code ONNX export contains only standard ops (the rotary embedding is inlined), and
jina's ALiBi is inlined the same way, so both run in ONNX Runtime, OpenVINO and fastembed as
well as in TEI (candle) and llama.cpp. The two 2 000-token cells are reachable by all five
engines; nomic's GGUF declares `context_length = 2048` (not 8 192 — the converter caps it at
`max_trained_positions`), so 2 000 is its longest measured text, while jina's declares 8 192.

Two engine settings were needed and are recorded rather than hidden:

- **jina + TEI** starts only with a 2 048-token cap. With the file's 8 192 the candle JinaBert
  backend builds an 8 192² ALiBi mask (about 3.2 GB) and did not become ready within 300 s on
  this 13 GB host. The TEI row points at a copy of the same weights (hard-linked) whose
  `config.json` and `sentence_bert_config.json` cap the sequence at 2 048; since the measured
  texts stop at 2 000, the cap changes no measured cell.
- **fastembed** embeds a request with its own internal batch size; its server now honours
  `--batch-size`, and these rows use 2 texts per sub-batch so 8 × 2 000 tokens fits. The
  harness sends the whole request; fastembed's own sub-batching is its documented behaviour.

### Engines

| engine | version | settings |
|---|---|---|
| Peregrine | `peregrine 0.2.0`, this branch | `serve -t 4 -c <model context> --slots 40 --no-profile`, `PEREGRINE_EMBED_CACHE_MB=0` |
| llama.cpp | v0.9.0 (`cd959ba`) | `llama-server --embedding --pooling mean -t 4 -fa auto -b 2048 -ub 2048 -np 32 -c 66048` |
| OpenVINO | 2026.4.1, sentence-transformers 6.2.0 | `INFERENCE_NUM_THREADS=4`, `PERFORMANCE_HINT=LATENCY`, the same ONNX converted to an IR |
| ONNX Runtime | 1.31.1 | `CPUExecutionProvider`, `intra_op_num_threads=4`, `ORT_ENABLE_ALL`, mean pooling over `last_hidden_state` |
| fastembed | 0.9.1 on ONNX Runtime 1.31.1 | `CPUExecutionProvider`, threads 4, sub-batch 2 |
| text-embeddings-inference | 1.9.4 (candle) | `--dtype float32 --pooling mean --max-batch-tokens 16384`, 4 threads |

## Method

- **Cells**: 16/128/512/2 000 tokens × batch 1, 8 and 32, plus one mixed-length batch of 32
  texts (8–512 tokens, seed `20261111`, 3 516 tokens). Batch 32 × 2 000 is 64 000 tokens and
  is skipped by every row's 16 384-token batch cap; ONNX Runtime's 8 × 2 000 cell is skipped
  on the rows named below because its padded attention buffers were OOM-killed here.
- **Texts**: `benchmarks/embed_texts.py` with each model's own tokenizer, 128 distinct texts
  per length. Every nomic text carries the model's document prefix `search_document: `
  (tokens include it), applied identically for every engine; jina needs no prefix. The mixed
  batch is all passages for nomic.
- **Fresh texts inside a batch and per timed request**: `run_cell` slices
  `texts[start:start+batch]` with `start = (wave · batch) mod (len - batch + 1)`, so a batch
  is 8/32 different texts and each timed wave is a different window; `--repeat-texts` is off.
  The 128-text pool makes all 20 batch-1 requests distinct and the two batch-32 requests
  disjoint (checked in the harness and against the raw `tokens_calibrated` fields).
- **Rounds**: three interleaved rounds (all rows round 1, then round 2, then round 3), one
  engine and one model process at a time; medians of the three `texts/s`.
- **Cache off**: `PEREGRINE_EMBED_CACHE_MB=0` for every timed row; the proof runner sets it.
- **Accuracy**: `embed.py --mode accuracy --acc-batch 1` on 201 texts (67 each at 16, 128 and
  512 tokens) against a fp32 reference. The reference is the fp32 ONNX export with mean
  pooling (the same convention as EmbeddingGemma in S4.9l): the PyTorch remote code of both
  families does not load on this host's transformers 5.5 (`NomicBertConfig` expects a
  `GPT2Config`; jina's code imports `find_pruneable_heads_and_indices`, removed upstream), and
  the fp32 ONNX is the full-precision export of the same weights. Peregrine F16 and every
  fp32 engine land at cosine 1.000000 against it, which is the check that the reference and
  the engines compute the same graph.
- **Equal accuracy**: a competitor row is used for a ratio when its cosine mean is within
  0.0005 of Peregrine's row and not below 0.999. The int8 ONNX exports are 0.928–0.957 and
  are listed but never used. Nomic's Q8_0 file itself is 0.998330 (both engines' Q8_0 agree
  to 2e-6), below the 0.999 class floor, so **the nomic Q8_0 row has no equal-accuracy
  competitor and is listed without ratios**; the F16 row is the family's standings row.

## Part 1 — the standings

> **Superseded for the batch cells (2026-11-13).** S4.8b (`#251`) landed on `main` after this
> run and gives a float-only encoder over 256 MiB 128 rows a pass instead of 32. The two F16
> rows' batch cells were re-measured on the merged build: nomic's 16-token cells become
> **1.019×** and **1.088×**, jina's 0.938× and 0.964×, and the pair is **24 of 36** cells
> fastest (was 21), twelve behind (was fifteen). The 128-token and 512-token cells did not
> move. Tables and raws: `benchmarks/results/bluefin-2026-11-13-s4.9u-merge/`.

### Peregrine F16 against the best equal-accuracy engine, before → after

Cells are Peregrine texts/s, the winning engine and its texts/s, and Peregrine ÷ it. All
numbers are medians of the three interleaved rounds and are recomputed from the raw JSONs.

**nomic-embed-text v1.5 F16** (best other: ONNX Runtime fp32 or fastembed fp32; OpenVINO at
512/2 000):

| cell | before | after | best other engine (after) |
|---|---:|---:|---|
| L16 b1 | 1.462× | **1.515×** | ONNX Runtime fp32 47.37 |
| L16 b8 | 0.895× | 0.936× | ONNX Runtime fp32 83.51 |
| L16 b32 | 0.927× | 0.979× | ONNX Runtime fp32 80.99 |
| L128 b1 | 0.933× | 0.929× | fastembed fp32 11.43 |
| L128 b8 | 0.910× | 0.946× | fastembed fp32 11.50 |
| L128 b32 | 0.913× | 0.947× | fastembed fp32 11.48 |
| L512 b1 | 1.014× | **1.036×** | fastembed fp32 2.40 |
| L512 b8 | 1.055× | **1.078×** | OpenVINO fp32 2.31 |
| L512 b32 | 1.094× | **1.113×** | fastembed fp32 2.25 |
| L2000 b1 | 1.074× | **1.085×** | OpenVINO fp32 0.46 |
| L2000 b8 | 1.054× | **1.067×** | OpenVINO fp32 0.46 |
| mixed b32 | 1.679× | **1.708×** | fastembed fp32 7.05 |

**jina-embeddings-v2-base-en F16** (best other: ONNX Runtime fp32 at 16/128, fastembed fp32
at 128 b8/b32 and the mixed batch, OpenVINO fp32 at 512/2 000):

| cell | before | after | best other engine (after) |
|---|---:|---:|---|
| L16 b1 | 1.342× | **1.343×** | fastembed fp32 52.80 |
| L16 b8 | 0.843× | 0.889× | OpenVINO fp32 88.47 |
| L16 b32 | 0.876× | 0.891× | ONNX Runtime fp32 88.95 |
| L128 b1 | 0.895× | 0.911× | ONNX Runtime fp32 12.06 |
| L128 b8 | 0.899× | 0.916× | fastembed fp32 12.14 |
| L128 b32 | 0.884× | 0.920× | fastembed fp32 12.05 |
| L512 b1 | 0.975× | **1.055×** | OpenVINO fp32 2.44 |
| L512 b8 | 0.961× | **1.014×** | OpenVINO fp32 2.56 |
| L512 b32 | 1.000× | **1.031×** | OpenVINO fp32 2.51 |
| L2000 b1 | 1.092× | **1.208×** | OpenVINO fp32 0.41 |
| L2000 b8 | 0.892× | **1.006×** | OpenVINO fp32 0.49 |
| mixed b32 | 1.645× | **1.666×** | fastembed fp32 7.39 |

**jina Q8_0** (the same engine path; the Q8_0 file is in the 0.9996 class, so the fp32
engines remain eligible): 512 b1/b8/b32 **0.959/0.959/0.996× → 1.057/1.008/1.025×**, 2 000
b1/b8 1.074/0.884 → **1.208/1.002×**, L16 b1 1.368→**1.453×**, L16 b8/b32 0.915/0.955 →
0.945/0.974×, L128 b1/b8/b32 0.886/0.909/0.880 → 0.916/0.903/0.918×, mixed 1.653→**1.678×**.

**nomic Q8_0** is listed for speed only (cosine 0.998330, below the 0.999 floor): 74.71 /
81.00 / 79.15 texts/s at 16 b1/b8/b32, 10.28 / 10.36 / 10.32 at 128, 2.37 / 2.33 / 2.38 at
512, 0.47 / 0.48 at 2 000 and 11.54 on the mixed batch. Its only same-class competitor is
llama.cpp's Q8_0 (33.5–63.8 texts/s at 16, 5.9–8.1 at 128, 1.4–1.9 at 512), which it beats in
every cell; the nomic Q8_0 row's L16 b1 numbers in the after window are noisy (llama.cpp's own
three rounds span 33.5–51.0 texts/s on that cell), and no ratio depends on them.

**The cells below 1.0 after this run** (all of them named):

| model / file | cells | ratio |
|---|---|---|
| nomic F16 | L16 b8, L16 b32 | 0.936, 0.979 |
| nomic F16 | L128 b1, b8, b32 | 0.929, 0.946, 0.947 |
| jina F16 | L16 b8, L16 b32 | 0.889, 0.891 |
| jina F16 | L128 b1, b8, b32 | 0.911, 0.916, 0.920 |
| jina Q8_0 | L16 b8, L16 b32 | 0.945, 0.974 |
| jina Q8_0 | L128 b1, b8, b32 | 0.916, 0.903, 0.918 |

That is **21 of 36 cells fastest** for these two models after the fixes (13 of 36 before:
nomic L16 b1/b8/b32 → b1; nomic's three 512 and two 2 000 cells were already ahead; jina
gained 512 b1, b8 (from 0.961) and 2 000 b8, and kept the rest).

### The full after table, texts/s (median of three interleaved rounds)

#### nomic
| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | L2000 b1 | L2000 b8 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fastembed-fp32 | 46.08 | 61.80 | 62.30 | 11.43 | 11.50 | 11.48 | 2.40 | 2.24 | 2.25 | 0.40 | 0.40 | 7.05 |
| fastembed-int8 | 66.33 | 79.59 | 62.72 | 12.02 | 11.49 | 11.85 | 2.50 | 2.42 | 2.48 | 0.43 | 0.43 | 7.55 |
| llama_cpp-f16 | 41.96 | 52.75 | 52.97 | 6.79 | 6.68 | 6.09 | 1.62 | 1.46 | 1.46 | 0.31 | 0.31 | 6.99 |
| llama_cpp-q8_0 | 37.23 | 59.93 | 63.80 | 8.13 | 7.17 | 5.94 | 1.93 | 1.42 | 1.42 | 0.31 | 0.31 | 6.90 |
| onnxruntime-fp32 | 47.37 | 83.51 | 80.99 | 11.36 | 9.73 | 9.32 | 2.35 | 2.13 | 2.13 | 0.41 | n/a | 2.01 |
| onnxruntime-int8 | 62.46 | 86.25 | 89.09 | 12.07 | 10.92 | 10.85 | 2.50 | 2.46 | 2.46 | 0.44 | n/a | 2.42 |
| openvino-fp32 | 39.88 | 79.13 | 77.42 | 10.22 | 8.33 | 9.50 | 2.35 | 2.31 | 2.18 | 0.46 | 0.46 | 2.31 |
| openvino-int8 | 34.28 | 70.62 | 60.37 | 9.36 | 6.50 | 6.92 | 1.82 | 1.67 | 1.66 | 0.36 | 0.36 | 1.71 |
| peregrine-f16 | 71.74 | 78.13 | 79.31 | 10.62 | 10.87 | 10.88 | 2.49 | 2.49 | 2.50 | 0.50 | 0.49 | 12.04 |
| peregrine-q8_0 | 74.71 | 81.00 | 79.15 | 10.28 | 10.36 | 10.32 | 2.37 | 2.33 | 2.38 | 0.47 | 0.48 | 11.54 |
| tei-fp32 | 14.81 | 14.23 | 15.71 | 3.92 | 3.76 | 3.68 | 0.79 | 0.71 | 0.70 | 0.12 | 0.12 | 1.36 |

#### jina
| engine / file | L16 b1 | L16 b8 | L16 b32 | L128 b1 | L128 b8 | L128 b32 | L512 b1 | L512 b8 | L512 b32 | L2000 b1 | L2000 b8 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fastembed-fp32 | 52.80 | 68.04 | 67.95 | 11.91 | 12.14 | 12.05 | 2.41 | 2.28 | 2.32 | 0.34 | 0.36 | 7.39 |
| fastembed-int8 | 70.75 | 74.34 | 84.41 | 12.29 | 12.12 | 12.13 | 2.48 | 2.56 | 2.54 | 0.37 | 0.41 | 7.82 |
| llama_cpp-f16 | 41.36 | 53.03 | 53.91 | 6.80 | 6.72 | 6.07 | 1.63 | 1.45 | 1.45 | 0.31 | 0.31 | 6.98 |
| llama_cpp-q8_0 | 50.12 | 63.54 | 64.42 | 8.22 | 7.17 | 6.01 | 1.92 | 1.43 | 1.44 | 0.30 | 0.30 | 6.97 |
| onnxruntime-fp32 | 51.34 | 85.53 | 88.95 | 12.06 | 10.48 | 9.83 | 2.39 | 2.17 | 2.14 | 0.34 | 0.35 | 2.01 |
| onnxruntime-int8 | 70.65 | 95.15 | 93.57 | 12.33 | 11.74 | 11.20 | 2.48 | 2.51 | 2.50 | 0.36 | n/a | 2.45 |
| openvino-fp32 | 45.21 | 88.47 | 83.45 | 11.40 | 9.82 | 10.73 | 2.44 | 2.56 | 2.51 | 0.41 | 0.49 | 2.52 |
| openvino-int8 | 36.87 | 70.81 | 63.65 | 9.27 | 6.95 | 7.32 | 1.89 | 1.76 | 1.75 | 0.32 | 0.36 | 1.78 |
| peregrine-f16 | 70.88 | 78.62 | 79.28 | 10.99 | 11.12 | 11.08 | 2.58 | 2.59 | 2.59 | 0.49 | 0.50 | 12.31 |
| peregrine-q8_0 | 76.72 | 83.64 | 86.67 | 11.05 | 10.96 | 11.06 | 2.58 | 2.58 | 2.57 | 0.49 | 0.49 | 12.40 |
| tei-fp32 | 14.46 | 15.36 | 15.55 | 4.11 | 4.03 | 3.88 | 0.78 | 0.77 | 0.74 | 0.11 | 0.11 | 1.57 |

n/a is a cell the row could not run: 32 × 2 000 exceeds the 16 384-token batch cap for every
row; ONNX Runtime's 8 × 2 000 is the OOM skip named in its skips JSON (the fp32 nomic row and
the int8 rows; the other rows measured it).

### Accuracy (cosine to the fp32 ONNX reference, 201 texts)

#### nomic

| engine / file | cosine mean | cosine min | reference |
|---|---:|---:|---|
| fastembed fp32 | 1.000000 | 1.000000 | onnx-fp32 |
| fastembed int8 | 0.941088 | 0.912848 | onnx-fp32 |
| llama.cpp F16 | 1.000000 | 0.999999 | onnx-fp32 |
| llama.cpp Q8_0 | 0.998328 | 0.997902 | onnx-fp32 |
| ONNX Runtime fp32 | 1.000000 | 1.000000 | onnx-fp32 |
| ONNX Runtime int8 | 0.941088 | 0.912848 | onnx-fp32 |
| OpenVINO fp32 | 1.000000 | 1.000000 | onnx-fp32 |
| OpenVINO int8 | 0.956579 | 0.938159 | onnx-fp32 |
| Peregrine F16 | 1.000000 | 1.000000 | onnx-fp32 |
| Peregrine Q8_0 | 0.998330 | 0.997958 | onnx-fp32 |
| TEI fp32 | 1.000000 | 1.000000 | onnx-fp32 |

#### jina

| engine / file | cosine mean | cosine min | reference |
|---|---:|---:|---|
| fastembed fp32 | 1.000000 | 1.000000 | onnx-fp32 |
| fastembed int8 | 0.927825 | 0.897579 | onnx-fp32 |
| llama.cpp F16 | 1.000000 | 1.000000 | onnx-fp32 |
| llama.cpp Q8_0 | 0.999596 | 0.999060 | onnx-fp32 |
| ONNX Runtime fp32 | 1.000000 | 1.000000 | onnx-fp32 |
| ONNX Runtime int8 | 0.927825 | 0.897579 | onnx-fp32 |
| OpenVINO fp32 | 1.000000 | 1.000000 | onnx-fp32 |
| OpenVINO int8 | 0.927848 | 0.899944 | onnx-fp32 |
| Peregrine F16 | 1.000000 | 1.000000 | onnx-fp32 |
| Peregrine Q8_0 | 0.999597 | 0.998971 | onnx-fp32 |
| TEI fp32 | 1.000000 | 1.000000 | onnx-fp32 |

The Q8_0 pair agrees with itself to 2e-6 (nomic) and 1e-6 (jina), so neither engine is the
source of the Q8_0 distance to the F16 class: it is the quantised weights. The int8 ONNX
exports are 0.93–0.96 and are not used for any ratio. The same convention as S4.9l: the
comparison class is cosine mean, and every fp32 row (Peregrine included) is 1.000000.

## Part 2 — the account, cell by cell

`cellprof` drives the same `Engine` as the server, one text per pass, 4 threads, one warm-up
then the timed pass; the stage times are `StepStats` (`matmul`, `attention`, and everything
else). The winning engine's own profile is beside it: ONNX Runtime's profiling JSON for the
fp32 ONNX (the winner at 128 tokens) and OpenVINO's per-node `PERF_COUNT` profile for the same
ONNX (the winner at 512 for jina). The absolute per-node sums are only comparable for shares:
OpenVINO's `PERF_COUNT` inflates the 4-thread sum by about 2× here (398 ms without it, 675–890
ms with it), so the table quotes its shares at 4 threads and its shares at 1 thread (where
`real_time` sums to the wall exactly), never its 4-thread total.

### The stage split (after), 4 threads, one text

| cell | pass ms | matmul | attention | element-wise / other |
|---|---:|---:|---:|---:|
| nomic F16 128 | 89.7 | **80.2 (89.4 %)** | 3.07 | 6.42 |
| nomic F16 512 | 395.8 | **324.1 (81.9 %)** | 33.52 | 38.24 |
| jina F16 128 | 88.8 | **80.1 (90.2 %)** | 3.22 | 5.52 |
| jina F16 512 | 388.7 | **324.6 (83.5 %)** | 37.32 | 26.81 |

Before the fixes: nomic 128 90.9 / 80.4 / 3.09 / 7.35; nomic 512 402.5 / 322.3 / 33.26 /
46.87; jina 128 88.9 / 78.9 / 3.91 / 6.04; jina 512 403.7 / 321.3 / **48.82** / 33.60.

### The matrix product against this CPU's roof

The pass's matrix work is 14.50 GMAC at 128 tokens and 57.98 GMAC at 512 (12 blocks of
d 768 / ff 3 072). This core's measured 4-thread f32 roof is **259.6 GMAC/s** (S4.9p:
`telem::Machine::probe`, and `fpbench`).

| matmul | 128 tokens | 512 tokens |
|---|---:|---:|
| Peregrine before | 80.4 ms = 180.4 GMAC/s (69.5 %) | 322.3 ms = 179.9 GMAC/s (69.3 %) |
| Peregrine after | 80.2 ms = 180.8 GMAC/s (69.6 %) | 324.1 ms = 178.9 GMAC/s (68.9 %) |
| ONNX Runtime fp32, `MatMul` + `FusedMatMul` | **70.4–71.9 ms = 202–206 GMAC/s (78–79 %)** | 305–315 ms = 184–190 GMAC/s (71–73 %) |
| OpenVINO fp32, `FullyConnected` share × wall | ~65 ms ≈ 222 GMAC/s (85 %) | ~280 ms ≈ 207 GMAC/s (80 %) |

The ONNX Runtime numbers are its own node durations (wall, 4 threads) and the winner at 128
tokens; the OpenVINO numbers are its `FullyConnected` share (66.9 % of a 97.8 ms pass at 128,
66.9 % of 419 ms at 512) and are marked approximate. At 512 the two are close and Peregrine
wins the cell on the other stages; at 128 rows MLAS's small-M GEMM is **12 % faster** and that
is the whole gap, because the matmul is 90 % of the pass.

### The winner's own profile

ONNX Runtime fp32, 4 threads, `ORT_ENABLE_ALL` (node wall times; the same ONNX the other
engines got):

| node | nomic 128 | nomic 512 | jina 128 | jina 512 |
|---|---:|---:|---:|---:|
| `MatMul` (72/84 nodes) | 70.65 | 296.36 | 69.43 | 286.44 |
| `FusedMatMul` | 1.26 | 18.45 | 1.01 | 18.59 |
| `Add` | 0.99 | 18.87 | 2.42 | 42.83 |
| `SkipLayerNormalization` | 3.28 | 11.81 | 0.34 | — |
| `Mul` | 2.34 | 16.10 | 0.94 | 12.57 |
| attention (`Einsum` / `Softmax` / `Expand`) | 2.18 | 11.04 | 0.47 | 27.36 |
| `Concat` / `Split` / `Transpose` / `Slice` | 6.34 | 40.46 | 5.45 | 35.32 |
| `Gelu` / `QuickGelu` | 0.84 | 6.85 | 1.67 | 7.94 |
| LayerNorm (`SkipLayer…`/`LayerNormalization`) | 3.28 | 11.81 | 0.93 | 1.57 |
| **total** | **92.0** | **426.5** | **81.9** | **423.1** |

OpenVINO fp32 shares (447 nodes nomic, 267 jina; 4-thread `PERF_COUNT`):

| node type | nomic 128 | nomic 512 | jina 128 | jina 512 |
|---|---:|---:|---:|---:|
| `FullyConnected` (60/72) | 86.9 % | 64.2 % | 91.6 % | 72.2 % |
| `Subgraph` (attention) | 10.3 % | 22.0 % | 6.2 % | 18.2 % |
| `Concat` / `VariadicSplit` / `Transpose` / `Gather` / `MVN` | 2.8 % | 13.8 % | 2.3 % | 9.6 % |

(The 1-thread profiles in `profile/ov-*-t1.txt` sum to the wall exactly: nomic 512 2 541.7 ms
with `FullyConnected` 2 184.1, jina 512 2 355.7 with 2 001.7; they are the same shares and
show that the 4-thread sum's inflation is `PERF_COUNT`'s, not a stage the engine hides.)

### What loses, stage by stage

- **128 tokens, both models: the matmul.** It is 89–90 % of the pass and runs at 180–181
  GMAC/s against MLAS's 202–206 (12 %), a fixed ~9–10 ms per pass. Neither engine's
  element-wise block explains the gap: ONNX Runtime's non-matmul work is 11.5 ms (jina) and
  20.1 ms (nomic, the `Slice`/`Concat`/`Split` transposes its graph pays) against Peregrine's
  8.7 and 9.5 ms — Peregrine is *ahead* there. The loss is entirely the GEMM's small-M rate:
  Peregrine's tile runs at the same ~179–181 GMAC/s from 1 to 512 rows, while MLAS has a
  128-row path at 202–206 and drops to 184–190 at 512. The 16-token batch cells are the same
  block seen as 8 × 16 = 128 rows and 32 × 16 = 512 rows.
- **512 tokens, nomic: nothing loses any more.** Peregrine's matmul is 3 % below ONNX
  Runtime's and 3.7× faster than OpenVINO's attention on the same cell; the pass is 1.04×
  ahead.
- **512 tokens, jina before: the ALiBi bias.** `alibi_row` was a scalar multiply-add over
  37.7 M score entries per pass; attention was 48.8 ms against nomic's 33.3 ms for the same
  shape. After the AVX2 body it is 37.3 ms and the cell is 1.055× ahead. The remaining 37 ms
  is the same score/softmax/mix strip as nomic's 33.5 and is not the losing stage.
- **Element-wise, both models before:** nomic's 46.9 ms and jina's 33.6 ms at 512 were
  single-thread bias/residual/query-scale walks (and, for jina, the scalar ALiBi's neighbour).
  They are 38.2 / 26.8 ms after Part 3. What is left (LayerNorms, the SwiGLU/GEGLU, pooling,
  the F16 embedding widen) is already on the pool and at the memory-pass limit; the exact GELU
  is not in these families (nomic uses SwiGLU, jina the tanh GELU), so the bit-changing
  `erf` question does not arise here.

## Part 3 — the fixes, same bits

Three commits. The ALiBi one carries its own unit twin test; all three are held to the same
measured before/after vectors (1 and 4 threads, single and batch of 4) and the in-round A/B
below measures the combined effect. The rule for all of them: the scalar or serial walk is the
definition, a row's bits cannot depend on the batch, the pass budget, the chunk or the thread,
and every new body is `to_bits`-equal to its twin. `git diff --stat origin/main...HEAD --
'*.golden'` is empty; the real-file accuracy gates still pass.

1. **`alibi_row` AVX2** (`peregrine-kernels/src/bidir.rs`). Eight consecutive keys per step:
   `i32` distance, `cvtepi32_ps` (exact below 2²⁴; no encoder context is above 8 192), a
   sign-bit negation (so `−0.0` stays) and a separate multiply and add — no FMA, so the
   product is rounded before the add exactly as the scalar `s += slope * -(d as f32)`.
   Test: every length 0–70, five query positions, three band shapes, seven slopes (±0,
   1e-30 included), plus ±0/1e-38/±inf scores, all `to_bits`-equal.
2. **nomic's q/k/v bias, query scale and residual walks** (`nomic_bert.rs`). The fused bias's
   three slices go into `attn::prepare_rows`'s row task (bias then RoPE, per row);
   `qscale_rows_par` scales Q after the rotation; `add_bias_residual_par` /
   `add_residual_par` do the `wo` and FFN residuals over rows, keeping `(o + b) + r` and the
   zero-bias skip.
3. **jina's bias and residual walks** (`jina_bert.rs`). One fork-join for the three q/k/v
   biases, one for the `ffn_up` bias, `add_bias_residual_par` for `attn_output` and
   `ffn_down`, `add_residual_par` for the optional `attn_norm_2`.

Bit identity, measured, not only argued: `peregrine embed` on the same 512-token and
2 000-token texts gives the same 768 floats before and after, at 1 and 4 threads and in a
batch of 4, for both models (`/var/tmp/bits_check.py` in the command log; the checked values
are in `tests`-equivalent form — no golden moved).

### The in-round A/B (two builds, five interleaved rounds, cache off)

Alternating before/after, one model and one engine at a time, `L128_b1` and `L512_b1`, four
threads, medians of five (`ab/before`, `ab/after`):

| model | cell | before | after | change |
|---|---|---:|---:|---:|
| nomic | 128 b1 | 10.471 | 10.652 | +1.7 % |
| nomic | 512 b1 | 2.475 | 2.475 | level |
| jina | 128 b1 | 10.797 | 10.980 | +1.7 % |
| jina | 512 b1 | 2.425 | 2.587 | **+6.7 %** |

The jina 512 change is the ALiBi body plus the element-wise block; the stage profile separates
them (attention 48.8 → 37.3 ms, other 33.6 → 26.8). Nomic's 512 pass lost 6.7 ms of
element-wise time (402.5 → 395.8 ms in `cellprof`) but the server cell is 2.4–2.5 texts/s with
a ±2 % round spread, so the A/B says level; the 128 cell, where the same work is a larger
share, reads +1.7 %.

## What is left, and what it would be worth

**The 128-token cells (and the 16-token batch cells) are behind because of the GEMM's
small-M rate, not the element-wise block.** Peregrine runs at 180–181 GMAC/s at 128 rows and
179 at 512; MLAS runs at 202–206 at 128 and 184–190 at 512. Closing the 12 % at 128 is a
small-M tile/loop-shape change in `fpack` (a different row block or a different panel walk),
not a bit-changing one: every candidate preserves the per-output `k`-chain, as S4.9p's
widened/direct split already does. Its worth is the fixed ~9–10 ms per pass: at 128 tokens
that is **+10–11 %**, which would put nomic at ~1.03–1.05× and jina at ~1.01–1.02× against
ONNX Runtime; at 512 it is +3 %, which would widen the 1.04–1.11× cells. It was not attempted
here because it is a kernel-shape task of its own (and the first attempt — running the
existing direct F16 loop at 128 rows instead of widening — measured *slower*: nomic 10.31 and
jina 10.75 texts/s against 10.65/10.98, so the widened panel is the right base to tune).

No cell is at a bit-changing stop: the two families' activations are SwiGLU and the tanh
GELU, both already the engine's definition, and no exact-`erf` pass is on their path. The
only bit-changing option in reach would be fusing LayerNorm/bias/residual into the GEMM
epilogue the way oneDNN does (which could be done in the same arithmetic order, so it is not
even necessarily bit-changing), and the account says its ceiling is the ~9 ms of element-wise
work at 128 — smaller than the matmul item above.

## Checks

| check | result |
|---|---|
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| `cargo test --release -p peregrine-kernels -p peregrine-core` | pass (188 + the new ALiBi twin test) |
| `cargo test --release -p peregrine-models --test embeddings --test synthetic --test no_alloc`, native and `PEREGRINE_ISA=scalar` | pass |
| `cargo test --release -p peregrine-server -p peregrine` | pass |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace` | pass |
| no file under `tests/golden/` changed | confirmed (`git diff --stat origin/main...HEAD -- '*.golden'`) |
| real-file accuracy gates (nomic F16/Q8_0, jina F16/Q8_0) | pass on this host (the notes' numbers) |
| absolute home paths in this directory | none (the results-wide check prints nothing) |

## Exact commands

```sh
# models
hf download nomic-ai/nomic-embed-text-v1.5-GGUF --include 'nomic-embed-text-v1.5.f16.gguf' \
  --include 'nomic-embed-text-v1.5.Q8_0.gguf' --local-dir ...
hf download nomic-ai/nomic-embed-text-v1.5 --include 'onnx/*' --include 'model.safetensors' ...
hf download gpustack/jina-embeddings-v2-base-en-GGUF --include '*FP16.gguf' \
  --include '*Q8_0.gguf' --local-dir ...
hf download Xenova/jina-embeddings-v2-base-en --include 'onnx/*' ...
hf download jinaai/jina-embeddings-v2-base-en --exclude '*.h5' --exclude '*.msgpack' ...

# texts (prefix applied once, to every engine)
python benchmarks/embed_texts.py --tokenizer-dir ~/models/embed/ref-nomic \
  --lengths 16,128,512,2000 --per-length 128 --max-tokens 2000 --stream-words 12000 \
  --prefix 'search_document: ' --mixed-set 32 --out /var/tmp/embed-texts-s49u/nomic.json
python benchmarks/embed_texts.py --tokenizer-dir ~/models/embed/ref-jina \
  --lengths 16,128,512,2000 --per-length 128 --max-tokens 2000 --stream-words 12000 \
  --mixed-set 32 --out /var/tmp/embed-texts-s49u/jina.json

# fp32 reference (ONNX, mean pooling over last_hidden_state)
python benchmarks/embed_ref_onnx.py --onnx ~/models/embed/nomic/onnx-fp32/model.onnx \
  --tokenizer ~/models/embed/nomic/onnx-fp32/tokenizer.json \
  --texts /var/tmp/embed-texts-s49u/nomic-acc.json \
  --out /var/tmp/embed-texts-s49u/ref/nomic-onnx-fp32.json \
  --output-name last_hidden_state --pooling mean --batch-size 2
# …the same for jina with the official model.onnx

# the proof run (one engine/model per invocation; three interleaved rounds)
python benchmarks/proof_run.py --mode bench --round 1 \
  --texts-dir /var/tmp/embed-texts-s49u --out-dir benchmarks/results/bluefin-2026-11-12-s4.9u/raw-after \
  --peregrine-binary ./target/release/peregrine --models nomic,jina --max-load 20 --min-avail-gb 2.0

# the in-round A/B (two builds, five rounds, cache off)
export PEREGRINE_EMBED_CACHE_MB=0
for r in 1 2 3 4 5; do for m in nomic jina; do
  for arm in before after; do
    bin=/var/tmp/target-s49u-before/release/peregrine
    [ $arm = after ] && bin=./target/release/peregrine
    python benchmarks/embed.py --select $m:peregrine:f16 --round $r --threads 4 \
      --texts-dir /var/tmp/embed-texts-s49u --output-dir .../ab/$arm \
      --peregrine-binary $bin --lengths 128,512 --batches 1 --cells L128_b1_c1,L512_b1_c1 \
      --budget 0.5 --max-requests 20 --ctx-tokens 2048
  done
done; done

# stage profiles and the other engines' own profiles
cargo build --release -p peregrine-bench --example cellprof --features ...
./target/release/examples/cellprof ~/models/embed/nomic/nomic-f16.gguf \
  /var/tmp/embed-texts-s49u/nomic.json 128,512 1 4
python appendix/ovprof.py ~/models/embed/nomic/onnx-fp32/model.onnx 512 4
python appendix/ortprof.py ~/models/embed/nomic/onnx-fp32/model.onnx 128 profile/ort-nomic-128.json
```
