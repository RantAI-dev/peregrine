# The ARM embedding table with the result cache off

The ARM table on the benchmark page compared engines on batch cells that had
been timed with Peregrine's exact embedding-vector cache **on**. A repeated
text list inside a cell was then a lookup after the first wave: Qwen3-Embedding
0.6B F16's "batch of 32" read 2.72–3.21× when the same 32 texts were served from
the cache, not when all 32 were computed. This run re-measures every row of the
table with the cache off, adds the two `gte-modernbert-base` rows and one `bert`
family file in F16 and Q8_0, and adds the `bge-reranker-v2-m3` row on
`/v1/rerank`. Nothing in the engine changed.

- Date: 2026-11-12, 13:39–14:35 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two
  clusters: 10 × Cortex-X925 at 3.9 GHz (CPUs 5–9 and 15–19) and
  10 × Cortex-A725 at 2.8 GHz, 121 GB unified memory, Linux
  `6.17.0-1008-nvidia`, governor `performance`, kernel tier
  `aarch64-neon-i8mm`, rustc 1.99.1.
- Peregrine: `origin/main` at `b4e206c` (the merge of S4.9q's #235), a clean
  worktree, `cargo build --release -p peregrine`.
- llama.cpp: v0.9.0, `cd959ba`, GNU 13.3.0 for Linux aarch64,
  `~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-server`.
- Threads: 10, the Cortex-X925 cluster, every engine pinned with
  `taskset -c 5-9,15-19`.

## The host is shared

A co-tenant (vLLM engine processes) was on the machine for the whole window.
The 1-minute load average was 0.76 at the first model, rose to 6.5–7.8 by the
end, and `MemAvailable` stayed at 35–36 GB of 121. Both engines of a model were
up at once and the three timed rounds alternated their order, so a drift is
charged to both arms alike. The short cells (10–200 ms) carry 15–95 % spreads
because the co-tenant moved during them; the ≥512-token and batch cells are
steady (0.2–5 %). Absolute throughput is indicative; the ratios and the
cache-off/cache-on difference are the claims.

## Method

`run.py` (standard library only) drives both engines over HTTP. For each model
it starts llama-server, builds the texts with that server's own `/tokenize`
(so both engines see byte-identical texts, calibrated to the target within
5 %), stops it, then starts one fresh Peregrine server and one fresh
llama-server and keeps both up for the model. After one warm-up request per
cell it times three rounds in a rotating order and reports the median; the
spread is min–max over the three.

- **Cells**: one text of 16, 128, 512 and 2 000 tokens; one request of 32
  distinct 128-token texts; one mixed request of 32 texts whose lengths were
  drawn from 8..512 with seed `20261111`. The `all-MiniLM` rows stop at 512
  tokens (the file's own limit), so they have no 2 000-token column.
- **Peregrine**: `serve -m <file> -t 10 -c <2048|512> --slots 32 --no-profile
  --opt EMBED_CACHE_MB=0`. `EMBED_CACHE_MB=0` is on every row: the exact cache
  of finished embedding vectors is off, so every text is computed every round.
- **llama.cpp**: `--embedding --pooling <mean|last|cls> -t 10 -c
  (ctx+8)·32 -b 2048 -ub 2048 -np 32 -fa auto`, one slot per text. A non-causal
  model needs the whole prompt in one ubatch; that is the shape the earlier ARM
  embedding runs used.
- **Rerank**: `POST /v1/rerank`, one query and 8 or 32 documents of 124–128
  tokens, five interleaved rounds, `pairs/s` = documents ÷ request wall.
  Peregrine `serve -c 8192 --slots 1`; llama-server `--reranking -c 8192 -b
  2048 -ub 2048 -np 1`. Both report the same 1 171 prompt tokens for 8
  documents and 4 689 for 32, so the pair texts are identical.

Reproduce:

```sh
python3 run.py --model gte-modernbert-base-f16 --out raw/gte-modernbert-base-f16.json
python3 run.py --model gte-modernbert-base-q8_0 --out raw/gte-modernbert-base-q8_0.json
# ... one --model per row, then
python3 rerank.py --runs 5 --out raw/rerank-bge-reranker-v2-m3-q8_0.json
```

The exact engine command lines are in each `raw/*.json` (`cmd`, with `~` for
the home directory).

## Models

| Model | Bytes | SHA-256 |
|---|---:|---|
| EmbeddingGemma 300M Q8_0 | 333,590,944 | `b5ce9d77…490d63` |
| Qwen3-Embedding 0.6B Q8_0 | 639,150,592 | `06507c7b…c3e439` |
| Qwen3-Embedding 0.6B F16 | 1,197,629,632 | `421a27e5…c54340` |
| Nemotron 3.5 Embed 1B Q8_0 | 1,220,296,800 | `27c8e82c…b2ec298` |
| bge-m3 Q8_0 | 634,553,760 | `950f4a8e…921167` |
| gte-modernbert-base F16 | 299,876,992 | `2cda419c…c4e437` |
| gte-modernbert-base Q8_0 | 160,208,512 | `d3c637bb…c93301` |
| all-MiniLM-L6-v2 F16 | 45,949,216 | `797b70c4…be4662` |
| all-MiniLM-L6-v2 Q8_0 | 25,008,064 | `263215c3…c2f8ed` |
| bge-reranker-v2-m3 Q8_0 | 635,676,416 | `a43c7c9b…aa1d3` |

The `gte-modernbert-base` files are `keisuke-miyako/gte-modernbert-base-gguf`;
`all-MiniLM-L6-v2` is `second-state/All-MiniLM-L6-v2-Embedding-GGUF`;
`bge-reranker-v2-m3-Q8_0.gguf` is `gpustack/bge-reranker-v2-m3-GGUF`.

## Results

`texts/s` is the median of three interleaved rounds; `ratio` is Peregrine ÷
llama.cpp, above 1.00 means Peregrine is ahead. Spread is the Peregrine
min–max over the three rounds.

### EmbeddingGemma 300M Q8_0 (333,590,944 bytes)

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine spread |
|---|---:|---:|---:|---:|
| one text, 16 tok | 80.63 | 37.90 | **2.13×** | 70.44–146.70 (94.6%) |
| one text, 128 tok | 15.97 | 6.63 | **2.41×** | 15.42–17.90 (15.5%) |
| one text, 512 tok | 3.92 | 1.88 | **2.09×** | 3.82–3.96 (3.5%) |
| one text, 2 000 tok | 1.67 | 0.83 | **2.02×** | 1.67–1.69 (1.2%) |
| batch 32 × 128 tok | 17.09 | 8.57 | **1.99×** | 17.06–17.12 (0.3%) |
| mixed batch of 32 | 9.04 | 4.60 | **1.96×** | 9.01–9.06 (0.5%) |

### Qwen3-Embedding 0.6B Q8_0 (639,150,592 bytes)

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine spread |
|---|---:|---:|---:|---:|
| one text, 16 tok | 64.53 | 20.19 | **3.20×** | 25.41–64.55 (60.6%) |
| one text, 128 tok | 4.58 | 2.32 | **1.97×** | 4.25–5.07 (17.9%) |
| one text, 512 tok | 0.97 | 0.57 | **1.71×** | 0.94–0.97 (3.6%) |
| one text, 2 000 tok | 0.30 | 0.21 | **1.38×** | 0.29–0.30 (1.4%) |
| batch 32 × 128 tok | 4.73 | 2.38 | **1.99×** | 4.73–4.74 (0.2%) |
| mixed batch of 32 | 2.35 | 1.30 | **1.80×** | 2.35–2.36 (0.5%) |

### Qwen3-Embedding 0.6B F16 (1,197,629,632 bytes)

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine spread |
|---|---:|---:|---:|---:|
| one text, 16 tok | 40.47 | 25.70 | **1.57×** | 16.55–41.20 (60.9%) |
| one text, 128 tok | 2.61 | 2.29 | **1.14×** | 2.53–3.02 (18.4%) |
| one text, 512 tok | 0.61 | 0.58 | **1.06×** | 0.60–0.61 (2.5%) |
| one text, 2 000 tok | 0.21 | 0.22 | **0.96×** | 0.21–0.21 (0.5%) |
| batch 32 × 128 tok | 2.73 | 2.33 | **1.17×** | 2.71–2.74 (1.0%) |
| mixed batch of 32 | 1.43 | 1.30 | **1.10×** | 1.42–1.43 (0.6%) |

### Nemotron 3.5 Embed 1B Q8_0 (1,220,296,800 bytes)

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine spread |
|---|---:|---:|---:|---:|
| one text, 16 tok | 40.28 | 15.85 | **2.54×** | 17.72–41.34 (58.6%) |
| one text, 128 tok | 3.44 | 1.26 | **2.72×** | 3.25–3.86 (17.7%) |
| one text, 512 tok | 0.82 | 0.32 | **2.57×** | 0.79–0.82 (3.3%) |
| one text, 2 000 tok | 0.30 | 0.13 | **2.40×** | 0.30–0.30 (0.0%) |
| batch 32 × 128 tok | 3.56 | 1.32 | **2.70×** | 3.54–3.56 (0.7%) |
| mixed batch of 32 | 1.89 | 0.72 | **2.62×** | 1.88–1.89 (0.3%) |

### bge-m3 Q8_0 (634,553,760 bytes)

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine spread |
|---|---:|---:|---:|---:|
| one text, 16 tok | 29.43 | 13.42 | **2.19×** | 23.11–30.89 (26.4%) |
| one text, 128 tok | 4.90 | 2.33 | **2.11×** | 4.72–5.04 (6.7%) |
| one text, 512 tok | 1.38 | 0.68 | **2.04×** | 1.36–1.41 (3.2%) |
| one text, 2 000 tok | 0.59 | 0.30 | **1.99×** | 0.59–0.60 (0.5%) |
| batch 32 × 128 tok | 4.93 | 2.37 | **2.08×** | 4.93–4.94 (0.2%) |
| mixed batch of 32 | 2.65 | 1.28 | **2.07×** | 2.65–2.66 (0.2%) |

### gte-modernbert-base F16 (299,876,992 bytes)

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine spread |
|---|---:|---:|---:|---:|
| one text, 16 tok | 54.48 | 34.29 | **1.59×** | 47.43–60.42 (23.8%) |
| one text, 128 tok | 10.65 | 7.62 | **1.40×** | 10.07–11.83 (16.5%) |
| one text, 512 tok | 2.65 | 1.98 | **1.34×** | 2.62–2.75 (4.7%) |
| one text, 2 000 tok | 1.16 | 0.82 | **1.42×** | 1.15–1.16 (1.4%) |
| batch 32 × 128 tok | 11.12 | 8.34 | **1.33×** | 11.09–11.19 (0.9%) |
| mixed batch of 32 | 6.13 | 4.48 | **1.37×** | 6.04–6.13 (1.6%) |

### gte-modernbert-base Q8_0 (160,208,512 bytes)

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine spread |
|---|---:|---:|---:|---:|
| one text, 16 tok | 61.75 | 35.03 | **1.76×** | 53.60–62.55 (14.5%) |
| one text, 128 tok | 12.79 | 7.49 | **1.71×** | 11.72–13.78 (16.1%) |
| one text, 512 tok | 3.30 | 1.89 | **1.74×** | 3.23–3.41 (5.5%) |
| one text, 2 000 tok | 1.39 | 0.78 | **1.78×** | 1.39–1.39 (0.5%) |
| batch 32 × 128 tok | 13.14 | 7.91 | **1.66×** | 13.08–13.17 (0.7%) |
| mixed batch of 32 | 7.49 | 4.28 | **1.75×** | 7.48–7.54 (0.8%) |

### all-MiniLM-L6-v2 F16 (45,949,216 bytes)

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine spread |
|---|---:|---:|---:|---:|
| one text, 16 tok | 236.16 | 143.10 | **1.65×** | 235.03–245.33 (4.4%) |
| one text, 128 tok | 75.45 | 50.71 | **1.49×** | 69.08–84.24 (20.1%) |
| one text, 512 tok | 37.94 | 24.86 | **1.53×** | 37.79–37.97 (0.5%) |
| batch 32 × 128 tok | 85.78 | 64.41 | **1.33×** | 85.71–85.88 (0.2%) |
| mixed batch of 32 | 76.77 | 56.37 | **1.36×** | 76.32–76.90 (0.8%) |

### all-MiniLM-L6-v2 Q8_0 (25,008,064 bytes)

| Cell | Peregrine texts/s | llama.cpp texts/s | Ratio | Peregrine spread |
|---|---:|---:|---:|---:|
| one text, 16 tok | 248.13 | 130.79 | **1.90×** | 226.84–251.17 (9.8%) |
| one text, 128 tok | 83.31 | 52.61 | **1.58×** | 77.87–91.69 (16.6%) |
| one text, 512 tok | 42.32 | 23.37 | **1.81×** | 42.04–42.55 (1.2%) |
| batch 32 × 128 tok | 95.03 | 58.34 | **1.63×** | 94.89–95.04 (0.2%) |
| mixed batch of 32 | 85.35 | 51.49 | **1.66×** | 85.09–85.47 (0.5%) |
## Rerank

bge-reranker-v2-m3 Q8_0, one query, documents of 124–128 tokens; `pairs/s`,
median of five interleaved rounds.

| Documents | Peregrine pairs/s | llama.cpp pairs/s | Ratio | Prompt tokens |
|---:|---:|---:|---:|---:|
| 8 | 7.41 | 5.39 | **1.38×** | 1 171 |
| 32 | 7.19 | 5.31 | **1.35×** | 4 689 |

## What changed against the cache-on numbers

The single-text cells barely move: their texts were already distinct inside a
cell. The batch cells do move, because the old harness reused one text list
inside a timed cell and the result cache answered every repeat. The clearest
case is Qwen3-Embedding 0.6B F16:

| Cell | Cache on (old table) | Cache off (this run) |
|---|---:|---:|
| batch 32 × 128 | 2.72–3.21× | **1.17×** |
| mixed batch of 32 | 1.11× | **1.10×** |

The old 2.72–3.21× was a lookup after the first wave, not a pass. With every
text computed, Peregrine is 17 % ahead on that batch, not 2.7–3.2×. The Q8_0
rows looked closer to their cache-off numbers because their batch texts were
already distinct enough that fewer were reused; the table above is the
consistent cache-off set.

**Cells below 1.0×:** one, Qwen3-Embedding 0.6B F16 at a single 2 000-token
text (0.96×). Its 512-token cell is 1.06× and its 128-token cell 1.14×: the
F16 matrix pass and llama.cpp's CPU flash attention meet at long context, while
the Q8_0 file of the same model stays 1.38–3.20× ahead. Every other cell of
every other row is above 1.0; the narrowest are MiniLM F16 batch (1.33×),
gte F16 batch (1.33×), gte F16 512 tokens (1.34×) and the rerank 32-document
cell (1.35×).
