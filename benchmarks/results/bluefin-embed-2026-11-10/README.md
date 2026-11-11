# Embedding baselines on bluefin (S4.2)

Competitor baselines for roadmap step **S4.2** (`docs/design/s4-embeddings-decisions.md` §4):
what Peregrine's `/v1/embeddings` endpoint (S4.1, PR #158) has to beat, measured on the
same host with the same texts, and the harness it plugs into.

Raw JSON for every round and every accuracy row is in this directory. The tables at the
bottom are generated from it by `benchmarks/embed_report.py`; `summary.json` holds the
merged medians and spreads.

## Machine

| | |
|---|---|
| Host | bluefin, bare metal, no other benchmark running |
| CPU | AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads |
| SIMD | AVX2 + FMA (no AVX-512) |
| Clock | `powersave` governor, but `scaling_cur_freq` 4.09 GHz against a 4.11 GHz max during runs |
| Memory | 13.5 GB RAM, 8 GB swap; swap was **7/8 GB used all evening** (left over from earlier work) |
| Kernel | 7.2.7-200.fc44.x86_64 |
| Threads | **4** (one per physical core) for every engine |
| Python | 3.14.8 (`/var/tmp/embed-venv`, nothing installed system-wide) |

Every timed invocation refused to start unless `load average (1 min) < 0.5` and
`MemAvailable >= 5 GB`; the measured values are recorded per row in the JSON
(`preflight`, `host_before`, `host_after`). One engine and one model were loaded at a time
and every server was stopped before the next row started.

The host's load average is dominated by this agent's own process, so each invocation waited
for the 1-minute average to fall below 0.5 before timing (the waits are not part of any
measurement).

## Versions and exact invocations

| Engine | Version | Invocation |
|---|---|---|
| llama.cpp | `0.9.0-dev (build 1, commit cd959ba)`, `/var/tmp/llama.cpp/build/bin/llama-server` | `llama-server -m <gguf> --embedding --pooling {mean,cls,last} -t 4 -c <max_tokens+8>*10 -b 1024 -ub 1024 -np 10 -fa auto --host 127.0.0.1 --port N` |
| text-embeddings-inference | `1.9.4`, image `ghcr.io/huggingface/text-embeddings-inference:cpu-latest` (digest `sha256:3352cc5bcdeaa2d86681139e94cd5cf33eed4700ab20279c8abd8fed6587e4ce`) | `podman run --rm --network host -e MKL/OMP/RAYON_NUM_THREADS=4 -v <model>:/data:ro <image> --model-id /data --pooling <p> --dtype float32 --tokenization-workers 1 --max-client-batch-size 128 --max-batch-tokens 16384 --max-batch-requests 128` |
| fastembed | `0.9.1` on `onnxruntime 1.31.1` | `servers/fastembed_server.py --model-path <dir> --threads 4` (`CPUExecutionProvider`, `intra_op_num_threads=4`) |
| sentence-transformers + OpenVINO | `6.2.0`, `openvino 2026.4.1`, `torch 2.15.0` | `servers/openvino_server.py --model-path <dir> --threads 4` (`backend="openvino"`, `INFERENCE_NUM_THREADS=4`, `PERFORMANCE_HINT=LATENCY`) |
| Peregrine (**pre-merge**) | worktree of `origin/s4.1-embeddings` at `63288ec` (PR #158), `cargo build --release -p peregrine` | `peregrine serve -m <gguf> -t 4 -c 512 --slots 10 --no-profile` |

Two notes on the engines:

- **TEI's CPU image runs candle, not ONNX Runtime.** The design note says "ONNX Runtime
  backend"; no published TEI CPU image uses ONNX Runtime, so the pinned CPU image (candle) is
  what is measured. Its numbers are the slowest in every row, which is a property of that
  backend on this host, not a flag mistake: the same host, threads and texts produce 2–6×
  more through llama.cpp.
- **The Peregrine row is a pre-merge branch build** and is labelled as such in every table
  and in the JSON (`row.notes`). It exists to check the harness end to end and to get a first
  number; it is not a result on `main`.

### Flags that mattered

- `llama-server` caps `n_ctx_slot` at the model's training context and processes **one
  embedding input per task even inside a single request**, so `-c` is sized as
  `(max_tokens + 8) × slots`; a `-b` larger than the total context aborts the process in
  `build_pooling` (`GGML_ASSERT(ggml_can_mul_mat)`).
- Some GGUFs (all-MiniLM) abort in the same place as soon as `-np > 1`. The adapter retries
  with `-np 1` and records `engine.slots_fallback`. bge-small, bge-base, Qwen3, Gemma and
  bge-m3 accept `-np 10`.
- The fastembed/OpenVINO servers serve one request at a time on a single thread with a
  128-deep listen backlog. A threaded server behind a lock made ten clients fight over the
  GIL (10× throughput loss), and the default backlog of 5 dropped SYNs so every drop cost a
  one-second retransmit. Both were measurement artefacts and are fixed; the concurrency
  cells for these two engines therefore measure **queueing**, not parallel inference, which
  is how sentence-transformers is actually deployed.

## Method

- Grid: 16, 128, 512 tokens × 1, 8, 32, 128 texts per request × 1 client, plus 10 concurrent
  clients at batch 1. Three interleaved rounds (round 1 over every row, then 2, then 3),
  medians reported with the min–max spread across rounds.
- Texts: 128 distinct texts per length per model, built from `benchmarks/prompts/corpus.txt`
  and calibrated with that model's own tokenizer. 16 and 128 are exact; 512 lands in
  508–512. One text of 128 is 17 tokens for the Qwen3 and Gemma tokenizers (no 16-token
  window exists); everything else is inside 5%. **Every engine receives byte-identical
  texts.**
- `texts/s` and `tokens/s` are wall clock over the whole cell. Tokens come from the engine's
  own `usage` when it reports one, otherwise from the tokenizer count (`tokens_source`).
- p50/p99 are whole-request latencies (a request of 128 texts is one sample).
- A cell that would need more than 120 s, or more batch tokens than the model's cap, is
  **skipped by rule** and recorded as skipped with the reason (`<row>-skips.json`). The caps
  are 65536 batch tokens for MiniLM and bge-small, 32768 for bge-base/Qwen3/Gemma, 16384 for
  bge-m3. That is why 128 × 512 is missing for the larger models: its attention buffers do
  not fit the memory budget (one fastembed run was OOM-killed before the cap existed).
- Accuracy: mean cosine similarity to the model's fp32 reference on 200 mixed-length texts
  (5 lengths × 40 texts, different from the benchmark texts), plus the vector norms to prove
  the engine normalises. Reference is PyTorch sentence-transformers fp32, except
  EmbeddingGemma whose PyTorch weights are licence-gated: there the reference is the fp32
  ONNX graph's own `sentence_embedding` output (`embed_ref.py` / `embed_ref_onnx.py`).

## Models

| Model | Precision rows | Files | SHA-256 (weights / manifest) |
|---|---|---|---|
| all-MiniLM-L6-v2 | llama.cpp F16, Q8_0; TEI fp32; fastembed fp32, int8; OpenVINO fp32 | `minilm-f16.gguf` 45.9 MB, `minilm-q8_0.gguf` 25.0 MB, `model.safetensors` 90.9 MB, `model.onnx` 90.4 MB, `model_int8`/`model_quint8_avx2.onnx` 23.0 MB | `797b70c4…`, `263215c3…`, dir `aa42abb2…`, `69242f59…`, `70e6fa44…` |
| bge-small-en-v1.5 | llama.cpp F32, Q8_0; TEI fp32; fastembed fp32, int8; OpenVINO fp32 | `bge-small-f32.gguf` 133.6 MB, `bge-small-q8_0.gguf` 36.8 MB, `model.safetensors` 133.5 MB, ONNX fp32 133.1 MB, ONNX int8 33.8 MB | in `row.fingerprint` of each round JSON |
| bge-base-en-v1.5 | llama.cpp F32, Q8_0; TEI fp32; fastembed fp32, int8; OpenVINO fp32 | `bge-base-f32.gguf` 436.3 MB, `bge-base-q8_0.gguf` 118.0 MB, safetensors 438.0 MB, ONNX fp32 435.8 MB, ONNX int8 109.6 MB | ditto |
| Qwen3-Embedding-0.6B | llama.cpp Q8_0; **Peregrine Q8_0 (pre-merge)** | `qwen3-embed-0.6b-q8_0.gguf` 639.2 MB | ditto |
| EmbeddingGemma-300M | llama.cpp Q8_0 | `embeddinggemma-Q8_0.gguf` 333.6 MB | ditto |
| bge-m3 | llama.cpp Q8_0 | `bge-m3-Q8_0.gguf` 634.6 MB | ditto |

Every round JSON carries `row.fingerprint`: the file's SHA-256, or for directory models a
manifest hash over every file's name, size and SHA-256. Sources: the GGUFs are the
`second-state`/`CompendiumLabs`/`ggml-org`/`Qwen` HF repos named in `embed_models.json`;
the safetensors are the upstream model repos; ONNX fp32/int8 are the published ONNX exports
of the same weights.

## Accuracy

Cosine to the fp32 reference, mean and minimum over 200 texts. Every engine returns unit
norm (`norm.mean` 1.0), so the pooling and normalisation match the model cards.

| model | row | cosine mean | cosine min |
|---|---|---|---|
| MiniLM | llama.cpp F16 | 0.999999 | 0.999997 |
| MiniLM | llama.cpp Q8_0 | 0.999614 | 0.999416 |
| MiniLM | TEI fp32 | 0.999999 | 0.999998 |
| MiniLM | fastembed int8 | 0.989039 | 0.970046 |
| MiniLM | fastembed fp32 / OpenVINO fp32 | 1.000000 | 1.000000 |
| bge-small | llama.cpp F32 | 0.999999 | 0.999997 |
| bge-small | llama.cpp Q8_0 | 0.999850 | 0.999748 |
| bge-small | fastembed int8 | 0.979639 | 0.960613 |
| bge-small | fastembed fp32 / OpenVINO fp32 / TEI fp32 | 1.000000 | 1.000000 |
| bge-base | llama.cpp F32 | 0.999994 | 0.999986 |
| bge-base | llama.cpp Q8_0 | 0.999829 | 0.999690 |
| bge-base | fastembed int8 | 0.979545 | 0.969984 |
| bge-base | fastembed fp32 / OpenVINO fp32 / TEI fp32 | 1.000000 | 1.000000 |
| Qwen3-0.6B | llama.cpp Q8_0 | 0.999319 | 0.998576 |
| Qwen3-0.6B | **Peregrine Q8_0 (pre-merge)** | 0.999313 | 0.998573 |
| EmbeddingGemma | llama.cpp Q8_0 (vs fp32 ONNX) | 0.998751 | 0.991447 |
| bge-m3 | llama.cpp Q8_0 | **not measured** — no fp32 weights on this host; downloading 2.2 GB and holding them in fp32 exceeds the ~3 GB per-load budget | |

The published int8 ONNX is the accuracy cost in this table: ~1–2 points of cosine
(0.989/0.980), minimum 0.960–0.970 on the hardest texts. Q8_0 in llama.cpp costs 0.0004.
Peregrine's pre-merge Q8_0 path agrees with llama.cpp's to six digits, which is the
strongest signal that the pooling and normalisation are right.

## What Peregrine has to beat

Fastest **competitor** per model and cell (texts/s, median of 3 rounds), and that number as
a multiple of llama.cpp on the same GGUF. On the only model where both exist, the pre-merge
Peregrine row is shown for scale — it is **not** a `main` result.

### all-MiniLM-L6-v2 (22 M parameters) — the guard/router case

| cell | best competitor | texts/s | × llama.cpp F16 | tokens/s |
|---|---|---|---|---|
| L16 b1 | fastembed int8 | 267 | 1.09× | 4,276 |
| L16 b8 | OpenVINO fp32 | 537 | 2.00× | 8,586 |
| L16 b32 | OpenVINO fp32 | 655 | 2.19× | 10,484 |
| L16 b128 | OpenVINO fp32 | 653 | 2.10× | 10,454 |
| L128 b1 | fastembed int8 | 87.0 | 1.54× | 11,132 |
| L128 b8 | OpenVINO fp32 | 98.5 | 1.63× | 12,613 |
| L128 b32 | OpenVINO fp32 | 95.9 | 1.58× | 12,279 |
| L128 b128 | OpenVINO fp32 | 96.6 | 1.59× | 12,368 |
| L512 b1 | OpenVINO fp32 | 20.7 | 1.58× | 10,591 |
| L512 b8 | OpenVINO fp32 | 20.8 | 1.57× | 10,663 |
| L512 b32 | OpenVINO fp32 | 21.2 | 1.59× | 10,844 |
| L512 b128 | OpenVINO fp32 | 20.5 | 1.54× | 10,488 |

Single 16-token text: **fastembed int8 3.3 ms p50 / 4.4 ms p99**, llama.cpp 3.4/5.0,
OpenVINO 4.9/5.3, TEI 6.1/7.6. At 10 clients, batch 1, 16 tokens: llama.cpp 19.8/39.7 ms
(it queues on one slot), fastembed 19.3/36.4, OpenVINO 23.5/42.6, TEI 34.8/54.3.

### bge-small-en-v1.5 (33 M parameters)

| cell | best competitor | texts/s | × llama.cpp F32 |
|---|---|---|---|
| L16 b1 | fastembed int8 | 206 | 1.37× |
| L16 b32 | OpenVINO fp32 | 400 | 1.25× |
| L128 b1 | fastembed int8 | 47.0 | 1.12× |
| L128 b8 | OpenVINO fp32 | 53.2 | 1.30× |
| L512 b1 | OpenVINO fp32 | 11.1 | 1.21× |
| L512 b128 | OpenVINO fp32 | 10.8 | 1.30× |

### bge-base-en-v1.5 (110 M parameters)

| cell | best competitor | texts/s | × llama.cpp F32 |
|---|---|---|---|
| L16 b1 | fastembed int8 | 90.0 | 1.49× |
| L16 b128 | fastembed int8 | 123 | 1.16× |
| L128 b1 | fastembed fp32 | 16.3 | 1.23× |
| L128 b128 | fastembed int8 | 15.9 | 1.20× |
| L512 b1 | fastembed int8 | 3.5 | 1.11× |
| L512 b32 | OpenVINO fp32 | 3.5 | 1.18× |

bge-base 128 × 512 is skipped on every engine (memory cap), so the largest cell is 32 × 512.

### Larger models (no ONNX competitor measured)

| model | best competitor | L16 b1 | L128 b1 | L512 b1 | note |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | llama.cpp Q8_0 | 15.7 | 2.19 | 0.50 | Peregrine (pre-merge, Q8_0): 20.8 / 2.73 / 0.62 = **1.32× / 1.25× / 1.24× llama.cpp**, cosine 0.9993 vs 0.9993 |
| EmbeddingGemma-300M | llama.cpp Q8_0 | 52.8 | 7.8 | 1.83 | single engine measured |
| bge-m3 | llama.cpp Q8_0 | 21.0 | 3.2 | 0.70 | single engine measured |

### Does §4.2 look right?

- **"≥ 1.0× llama.cpp on every row"** — already true for the pre-merge Peregrine on Qwen3
  (1.24–1.32× in every cell). Keep it; it is the floor, not the target.
- **"≥ 1.5× on batches"** is **the wrong bar, and it is too low.** llama.cpp does not batch:
  it processes one embedding input per task even when the request carries 128 of them, so
  its batch curve is flat (MiniLM 246 → 310 texts/s from batch 1 to 128) and almost any
  batching implementation clears 1.5× against it. The engines that actually batch clear
  **2.0–2.2×** at short texts. The target should be **≥ 2.2× llama.cpp at batch 32/128**.
- **"≥ 1.0× the best of ONNX Runtime / OpenVINO at equal accuracy"** is the binding
  constraint, and it is much harder than the llama.cpp target: on MiniLM it means
  **655 texts/s at L16 b128 (10.5k tok/s), 96.6 at L128 b128 (12.4k tok/s) and 20.5 at
  L512 b128 (10.5k tok/s)** at fp32-ONNX accuracy; on bge-base, 123 / 15.9 / 3.5. The
  int8 rows are faster still but pay 1–2 points of cosine, so at "equal accuracy" the fp32
  OpenVINO/fastembed rows are the number to beat.
- **"p50 of one 16-token text ≤ the best competitor"** — for MiniLM that is **3.3 ms**
  (fastembed int8) or 3.9 ms at fp32-equivalent accuracy. No Peregrine MiniLM row exists
  yet (S4.6 adds `bert`), so this is unmeasured; the Qwen3 analogue (46.8 ms pre-merge vs
  llama.cpp 63.1) suggests the target is reachable there.
- **"Rerank and decide ≥ 2× llama.cpp"** — **not measured in this pass**; see below.

## Left out, and honest caveats

- **Rerank** (`bge-reranker-v2-m3` on llama.cpp `--reranking` and TEI): not run. The
  throughput grid for six models already used the evening, and the reranker file was never
  downloaded. Documents/s at 1 query × 32 documents × 128/512 tokens is the missing table.
- **bge-m3 accuracy**: no fp32 reference on the host (see above). Its throughput row stands;
  its accuracy column is empty and marked so.
- **No mixed-length batches.** Every text inside a cell has the same token length, which is
  the fair way to compare padded (ONNX/TEI) and ragged (Peregrine) batching but hides the
  ragged-batch advantage the design claims. A batch of 32 texts of 16–512 tokens would show
  it; that is the next thing to add.
- **All engines are "batch" only where the engine batches.** The batch axis is therefore an
  HTTP-amortisation axis for llama.cpp and for Peregrine's pre-merge path, and a real
  batching axis for ONNX Runtime and OpenVINO. This is the single most important shape in
  these tables.
- **The concurrency column is queueing for three of five engines** (llama.cpp with
  `-np 1` on MiniLM, fastembed, OpenVINO). Only TEI merges concurrent requests into a batch;
  its aggregate throughput does not rise either (35.8 → 35.2 texts/s at L128), which is
  consistent with candle serialising internally.
- **Swap was full (7/8 GB) throughout.** Memory available stayed above 6.4 GB in every
  recorded row, and no row shows the sawtooth a swap-in would produce, but a run on a host
  with free swap could be a few percent faster.
- **One fastembed run was OOM-killed** on bge-base 128 × 512 before the batch-token cap
  existed. Nothing from that run is in the tables; the cell is skipped by rule everywhere.
- **llama.cpp batch 128 at 512 tokens** is one 9.6 s request on MiniLM and 8.6 s on bge-small.
  Those cells have a single sample per round, so their p99 is that sample, not a tail.
- **TEI is 2–6× slower than llama.cpp here.** Rounds were consistent (spread < 5%), so it is
  not noise; the candle CPU backend is simply slow on these small encoders. It also caps a
  batch at 16k tokens, splitting 128 × 512 into four forwards.

## Full tables (generated by `benchmarks/embed_report.py`)

**bge-base** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 66.12 | 123 | 124 | 104 | 16.33 | 14.47 | 13.02 | 12.94 | 3.32 | 2.88 | 2.88 | skip |
| fastembed int8 | 0.9795 | 90.02 | 106 | 124 | 123 | 16.30 | 16.40 | 15.96 | 15.93 | 3.53 | 3.44 | 3.47 | skip |
| llama.cpp f32 | 1.0000 | 60.64 | 102 | 104 | 106 | 13.29 | 13.36 | 13.32 | 13.31 | 3.16 | 3.00 | 2.98 | skip |
| llama.cpp q8_0 | 0.9998 | 63.40 | 82.46 | 82.90 | 84.09 | 10.69 | 9.33 | 9.42 | 9.21 | 2.44 | 2.16 | 2.16 | skip |
| OpenVINO fp32 | 1.0000 | 60.08 | 111 | 117 | 116 | 14.45 | 13.69 | 15.21 | 14.76 | 3.35 | 3.55 | 3.54 | skip |
| TEI fp32 | 1.0000 | 26.44 | 27.97 | 29.14 | 29.10 | 5.80 | 5.67 | 5.86 | 5.80 | 1.08 | 1.03 | 1.00 | skip |

**bge-m3** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp q8_0 | — | 20.99 | 25.27 | 25.39 | 25.41 | 3.21 | 2.72 | 2.73 | 2.71 | 0.70 | 0.63 | 0.63 | skip |

**bge-small** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 174 | 267 | 368 | 342 | 46.95 | 48.23 | 38.76 | 38.94 | 8.20 | 7.27 | 7.17 | 6.98 |
| fastembed int8 | 0.9796 | 206 | 310 | 383 | 389 | 47.02 | 48.36 | 46.88 | 47.15 | 8.37 | 8.32 | 8.31 | 8.47 |
| llama.cpp f32 | 1.0000 | 150 | 313 | 321 | 331 | 42.08 | 40.76 | 40.94 | 41.08 | 9.20 | 8.34 | 8.24 | 8.31 |
| llama.cpp q8_0 | 0.9999 | 166 | 257 | 264 | 271 | 34.75 | 31.92 | 32.29 | 32.16 | 7.47 | 6.81 | 6.71 | 6.75 |
| OpenVINO fp32 | 1.0000 | 142 | 344 | 400 | 385 | 45.47 | 53.22 | 50.86 | 50.87 | 11.11 | 10.79 | 10.98 | 10.78 |
| TEI fp32 | 1.0000 | 88.58 | 102 | 127 | 127 | 18.16 | 18.04 | 18.84 | 19.39 | 2.19 | 2.18 | 2.12 | 2.04 |

**embeddinggemma** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp q8_0 | 0.9988 | 52.83 | 55.06 | 54.77 | 55.38 | 7.80 | 7.83 | 7.83 | 7.81 | 1.83 | 1.83 | 1.83 | skip |

**minilm** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 240 | 506 | 643 | 600 | 84.60 | 80.01 | 73.54 | 74.96 | 15.46 | 13.57 | 13.26 | 12.51 |
| fastembed int8 | 0.9890 | 267 | 437 | 636 | 645 | 86.97 | 90.06 | 86.06 | 85.71 | 14.82 | 13.92 | 13.52 | 13.64 |
| llama.cpp f16 | 1.0000 | 246 | 269 | 299 | 310 | 56.59 | 60.34 | 60.62 | 60.81 | 13.13 | 13.27 | 13.30 | 13.27 |
| llama.cpp q8_0 | 0.9996 | 262 | 316 | 365 | 365 | 63.52 | 69.91 | 70.48 | 70.43 | 14.75 | 14.83 | 14.90 | 14.95 |
| OpenVINO fp32 | 1.0000 | 199 | 537 | 655 | 653 | 78.04 | 98.54 | 95.93 | 96.62 | 20.70 | 20.84 | 21.20 | 20.50 |
| TEI fp32 | 1.0000 | 153 | 223 | 228 | 247 | 35.77 | 36.39 | 37.75 | 37.45 | 4.26 | 4.46 | 4.10 | 4.01 |

**qwen3-embed-0.6b** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp q8_0 | 0.9993 | 15.71 | 15.87 | 15.97 | 15.82 | 2.19 | 2.22 | 2.20 | 2.19 | 0.50 | 0.50 | 0.49 | skip |
| Peregrine q8_0 | 0.9993 | 20.78 | 21.66 | 20.45 | 20.85 | 2.73 | 2.73 | 2.72 | 2.69 | 0.62 | 0.62 | 0.62 | skip |


**bge-base** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 1,058 | 1,971 | 1,988 | 1,667 | 2,090 | 1,853 | 1,667 | 1,656 | 1,696 | 1,472 | 1,474 | skip |
| fastembed int8 | 0.9795 | 1,440 | 1,701 | 1,991 | 1,963 | 2,087 | 2,099 | 2,043 | 2,039 | 1,804 | 1,758 | 1,774 | skip |
| llama.cpp f32 | 1.0000 | 970 | 1,634 | 1,668 | 1,688 | 1,700 | 1,710 | 1,705 | 1,704 | 1,616 | 1,534 | 1,524 | skip |
| llama.cpp q8_0 | 0.9998 | 1,014 | 1,319 | 1,326 | 1,346 | 1,369 | 1,194 | 1,205 | 1,180 | 1,247 | 1,105 | 1,105 | skip |
| OpenVINO fp32 | 1.0000 | 961 | 1,780 | 1,872 | 1,852 | 1,849 | 1,753 | 1,948 | 1,890 | 1,710 | 1,814 | 1,813 | skip |
| TEI fp32 | 1.0000 | 423 | 448 | 466 | 466 | 743 | 725 | 750 | 743 | 550 | 527 | 514 | skip |

**bge-m3** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp q8_0 | — | 336 | 405 | 407 | 408 | 411 | 348 | 350 | 346 | 355 | 324 | 324 | skip |

**bge-small** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 2,784 | 4,266 | 5,888 | 5,476 | 6,009 | 6,173 | 4,961 | 4,984 | 4,193 | 3,720 | 3,666 | 3,570 |
| fastembed int8 | 0.9796 | 3,296 | 4,962 | 6,126 | 6,220 | 6,018 | 6,191 | 6,001 | 6,035 | 4,282 | 4,259 | 4,251 | 4,332 |
| llama.cpp f32 | 1.0000 | 2,402 | 5,001 | 5,134 | 5,295 | 5,386 | 5,217 | 5,240 | 5,258 | 4,703 | 4,266 | 4,218 | 4,250 |
| llama.cpp q8_0 | 0.9999 | 2,659 | 4,106 | 4,231 | 4,335 | 4,448 | 4,085 | 4,133 | 4,117 | 3,822 | 3,484 | 3,431 | 3,454 |
| OpenVINO fp32 | 1.0000 | 2,275 | 5,508 | 6,398 | 6,155 | 5,820 | 6,812 | 6,510 | 6,511 | 5,681 | 5,519 | 5,616 | 5,517 |
| TEI fp32 | 1.0000 | 1,417 | 1,636 | 2,026 | 2,028 | 2,325 | 2,309 | 2,412 | 2,481 | 1,120 | 1,116 | 1,084 | 1,043 |

**embeddinggemma** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp q8_0 | 0.9988 | 845 | 881 | 876 | 886 | 998 | 1,002 | 1,002 | 1,000 | 938 | 937 | 936 | skip |

**minilm** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 3,836 | 8,090 | 10,289 | 9,595 | 10,829 | 10,241 | 9,413 | 9,594 | 7,910 | 6,942 | 6,780 | 6,397 |
| fastembed int8 | 0.9890 | 4,276 | 6,988 | 10,177 | 10,319 | 11,132 | 11,528 | 11,016 | 10,971 | 7,581 | 7,121 | 6,914 | 6,977 |
| llama.cpp f16 | 1.0000 | 3,943 | 4,309 | 4,790 | 4,968 | 7,244 | 7,723 | 7,760 | 7,784 | 6,718 | 6,788 | 6,801 | 6,788 |
| llama.cpp q8_0 | 0.9996 | 4,197 | 5,050 | 5,843 | 5,847 | 8,130 | 8,949 | 9,022 | 9,015 | 7,545 | 7,588 | 7,622 | 7,647 |
| OpenVINO fp32 | 1.0000 | 3,187 | 8,586 | 10,484 | 10,454 | 9,989 | 12,613 | 12,279 | 12,368 | 10,591 | 10,663 | 10,844 | 10,488 |
| TEI fp32 | 1.0000 | 2,455 | 3,562 | 3,647 | 3,944 | 4,579 | 4,658 | 4,832 | 4,793 | 2,177 | 2,279 | 2,096 | 2,052 |

**qwen3-embed-0.6b** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp q8_0 | 0.9993 | 251 | 254 | 256 | 253 | 281 | 284 | 281 | 281 | 255 | 254 | 252 | skip |
| Peregrine q8_0 | 0.9993 | 332 | 347 | 327 | 334 | 350 | 350 | 348 | 344 | 316 | 316 | 317 | skip |


## single-text latency (batch 1)


| model | engine · precision | p50 1×16 (ms) | p99 1×16 (ms) | p50 10×1×16 (ms) | p99 10×1×16 (ms) |
|---|---|---|---|---|---|
| bge-base | fastembed fp32 | 14.8 | 18.9 | 85.3 | 151.4 |
| bge-base | fastembed int8 | 10.5 | 16.3 | 60.3 | 109.0 |
| bge-base | llama.cpp f32 | 16.4 | 17.1 | 99.8 | 110.0 |
| bge-base | llama.cpp q8_0 | 15.4 | 22.1 | 133.8 | 153.2 |
| bge-base | OpenVINO fp32 | 16.5 | 18.1 | 87.9 | 159.3 |
| bge-base | TEI fp32 | 36.0 | 44.7 | 241.6 | 397.4 |
| bge-m3 | llama.cpp q8_0 | 46.7 | 58.9 | 394.7 | 415.5 |
| bge-small | fastembed fp32 | 5.6 | 6.1 | 33.0 | 63.4 |
| bge-small | fastembed int8 | 4.5 | 6.3 | 28.5 | 55.4 |
| bge-small | llama.cpp f32 | 6.5 | 7.9 | 32.7 | 42.0 |
| bge-small | llama.cpp q8_0 | 5.8 | 8.5 | 38.2 | 55.5 |
| bge-small | OpenVINO fp32 | 7.0 | 7.2 | 35.4 | 63.6 |
| bge-small | TEI fp32 | 10.8 | 13.5 | 60.5 | 99.1 |
| embeddinggemma | llama.cpp q8_0 | 18.8 | 21.1 | 98.8 | 178.5 |
| minilm | fastembed fp32 | 3.9 | 4.7 | 20.5 | 39.4 |
| minilm | fastembed int8 | 3.3 | 4.4 | 19.3 | 36.4 |
| minilm | llama.cpp f16 | 3.9 | 4.3 | 19.8 | 39.7 |
| minilm | llama.cpp q8_0 | 3.4 | 5.0 | 17.4 | 38.6 |
| minilm | OpenVINO fp32 | 4.9 | 5.3 | 23.5 | 42.6 |
| minilm | TEI fp32 | 6.1 | 7.6 | 34.8 | 54.3 |
| qwen3-embed-0.6b | llama.cpp q8_0 | 63.1 | 65.2 | 622.4 | 648.8 |
| qwen3-embed-0.6b | Peregrine q8_0 | 46.8 | 55.8 | 328.0 | 591.8 |


## fastest competitor per cell


| model | cell | best competitor | texts/s | tokens/s | worst-cell note |
|---|---|---|---|---|---|
| bge-base | L16 b1 | fastembed int8 | 90.0 | 1,440 | 3 rounds |
| bge-base | L16 b8 | fastembed fp32 | 123 | 1,971 | 3 rounds |
| bge-base | L16 b32 | fastembed int8 | 124 | 1,991 | 3 rounds |
| bge-base | L16 b128 | fastembed int8 | 123 | 1,963 | 3 rounds |
| bge-base | L128 b1 | fastembed fp32 | 16.3 | 2,090 | 3 rounds |
| bge-base | L128 b8 | fastembed int8 | 16.4 | 2,099 | 3 rounds |
| bge-base | L128 b32 | fastembed int8 | 16.0 | 2,043 | 3 rounds |
| bge-base | L128 b128 | fastembed int8 | 15.9 | 2,039 | 3 rounds |
| bge-base | L512 b1 | fastembed int8 | 3.5 | 1,804 | 3 rounds |
| bge-base | L512 b8 | OpenVINO fp32 | 3.5 | 1,814 | 3 rounds |
| bge-base | L512 b32 | OpenVINO fp32 | 3.5 | 1,813 | 3 rounds |
| bge-m3 | L16 b1 | llama.cpp q8_0 | 21.0 | 336 | 3 rounds |
| bge-m3 | L16 b8 | llama.cpp q8_0 | 25.3 | 405 | 3 rounds |
| bge-m3 | L16 b32 | llama.cpp q8_0 | 25.4 | 407 | 3 rounds |
| bge-m3 | L16 b128 | llama.cpp q8_0 | 25.4 | 408 | 3 rounds |
| bge-m3 | L128 b1 | llama.cpp q8_0 | 3.2 | 411 | 3 rounds |
| bge-m3 | L128 b8 | llama.cpp q8_0 | 2.7 | 348 | 3 rounds |
| bge-m3 | L128 b32 | llama.cpp q8_0 | 2.7 | 350 | 3 rounds |
| bge-m3 | L128 b128 | llama.cpp q8_0 | 2.7 | 346 | 3 rounds |
| bge-m3 | L512 b1 | llama.cpp q8_0 | 0.7 | 355 | 3 rounds |
| bge-m3 | L512 b8 | llama.cpp q8_0 | 0.6 | 324 | 3 rounds |
| bge-m3 | L512 b32 | llama.cpp q8_0 | 0.6 | 324 | 3 rounds |
| bge-small | L16 b1 | fastembed int8 | 206 | 3,296 | 3 rounds |
| bge-small | L16 b8 | OpenVINO fp32 | 344 | 5,508 | 3 rounds |
| bge-small | L16 b32 | OpenVINO fp32 | 400 | 6,398 | 3 rounds |
| bge-small | L16 b128 | fastembed int8 | 389 | 6,220 | 3 rounds |
| bge-small | L128 b1 | fastembed int8 | 47.0 | 6,018 | 3 rounds |
| bge-small | L128 b8 | OpenVINO fp32 | 53.2 | 6,812 | 3 rounds |
| bge-small | L128 b32 | OpenVINO fp32 | 50.9 | 6,510 | 3 rounds |
| bge-small | L128 b128 | OpenVINO fp32 | 50.9 | 6,511 | 3 rounds |
| bge-small | L512 b1 | OpenVINO fp32 | 11.1 | 5,681 | 3 rounds |
| bge-small | L512 b8 | OpenVINO fp32 | 10.8 | 5,519 | 3 rounds |
| bge-small | L512 b32 | OpenVINO fp32 | 11.0 | 5,616 | 3 rounds |
| bge-small | L512 b128 | OpenVINO fp32 | 10.8 | 5,517 | 3 rounds |
| embeddinggemma | L16 b1 | llama.cpp q8_0 | 52.8 | 845 | 3 rounds |
| embeddinggemma | L16 b8 | llama.cpp q8_0 | 55.1 | 881 | 3 rounds |
| embeddinggemma | L16 b32 | llama.cpp q8_0 | 54.8 | 876 | 3 rounds |
| embeddinggemma | L16 b128 | llama.cpp q8_0 | 55.4 | 886 | 3 rounds |
| embeddinggemma | L128 b1 | llama.cpp q8_0 | 7.8 | 998 | 3 rounds |
| embeddinggemma | L128 b8 | llama.cpp q8_0 | 7.8 | 1,002 | 3 rounds |
| embeddinggemma | L128 b32 | llama.cpp q8_0 | 7.8 | 1,002 | 3 rounds |
| embeddinggemma | L128 b128 | llama.cpp q8_0 | 7.8 | 1,000 | 3 rounds |
| embeddinggemma | L512 b1 | llama.cpp q8_0 | 1.8 | 938 | 3 rounds |
| embeddinggemma | L512 b8 | llama.cpp q8_0 | 1.8 | 937 | 3 rounds |
| embeddinggemma | L512 b32 | llama.cpp q8_0 | 1.8 | 936 | 3 rounds |
| minilm | L16 b1 | fastembed int8 | 267 | 4,276 | 3 rounds |
| minilm | L16 b8 | OpenVINO fp32 | 537 | 8,586 | 3 rounds |
| minilm | L16 b32 | OpenVINO fp32 | 655 | 10,484 | 3 rounds |
| minilm | L16 b128 | OpenVINO fp32 | 653 | 10,454 | 3 rounds |
| minilm | L128 b1 | fastembed int8 | 87.0 | 11,132 | 3 rounds |
| minilm | L128 b8 | OpenVINO fp32 | 98.5 | 12,613 | 3 rounds |
| minilm | L128 b32 | OpenVINO fp32 | 95.9 | 12,279 | 3 rounds |
| minilm | L128 b128 | OpenVINO fp32 | 96.6 | 12,368 | 3 rounds |
| minilm | L512 b1 | OpenVINO fp32 | 20.7 | 10,591 | 3 rounds |
| minilm | L512 b8 | OpenVINO fp32 | 20.8 | 10,663 | 3 rounds |
| minilm | L512 b32 | OpenVINO fp32 | 21.2 | 10,844 | 3 rounds |
| minilm | L512 b128 | OpenVINO fp32 | 20.5 | 10,488 | 3 rounds |
| qwen3-embed-0.6b | L16 b1 | llama.cpp q8_0 | 15.7 | 251 | 3 rounds |
| qwen3-embed-0.6b | L16 b8 | llama.cpp q8_0 | 15.9 | 254 | 3 rounds |
| qwen3-embed-0.6b | L16 b32 | llama.cpp q8_0 | 16.0 | 256 | 3 rounds |
| qwen3-embed-0.6b | L16 b128 | llama.cpp q8_0 | 15.8 | 253 | 3 rounds |
| qwen3-embed-0.6b | L128 b1 | llama.cpp q8_0 | 2.2 | 281 | 3 rounds |
| qwen3-embed-0.6b | L128 b8 | llama.cpp q8_0 | 2.2 | 284 | 3 rounds |
| qwen3-embed-0.6b | L128 b32 | llama.cpp q8_0 | 2.2 | 281 | 3 rounds |
| qwen3-embed-0.6b | L128 b128 | llama.cpp q8_0 | 2.2 | 281 | 3 rounds |
| qwen3-embed-0.6b | L512 b1 | llama.cpp q8_0 | 0.5 | 255 | 3 rounds |
| qwen3-embed-0.6b | L512 b8 | llama.cpp q8_0 | 0.5 | 254 | 3 rounds |
| qwen3-embed-0.6b | L512 b32 | llama.cpp q8_0 | 0.5 | 252 | 3 rounds |
