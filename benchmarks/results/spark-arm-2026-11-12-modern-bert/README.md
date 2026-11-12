# gte-modernbert-base on the ARM host: Peregrine before/after S4.9n vs llama.cpp

S4.9n gives the `modern-bert` family's two scalar element-wise kernels NEON bodies
(`geglu_erf`, `layer_norm_rows_nobias`) that are bit-identical to their scalar twins,
and records the family's synthetic aarch64 goldens. This run measures the family on the
project's ARM host with the same texts and server shape as the latest ARM embedding
comparison ([`spark-2026-11-12-embeddings`](../spark-2026-11-12-embeddings/README.md)),
and checks that the NEON bodies change neither throughput materially nor a vector bit.

- Date: 2026-11-12.
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores (10 × Cortex-X925 +
  10 × Cortex-A725), 121 GB unified memory, Linux `6.17.0-1008-nvidia`, governor
  `performance`, kernel tier `aarch64-neon-i8mm`, rustc 1.99.1.
- Threads: 10 (the Cortex-X925 cluster), the main-table count of the earlier ARM runs.
- Peregrine before: `origin/main` at `7328026`, built from a clean worktree,
  `cargo build --release -p peregrine`.
- Peregrine after: this branch, `cargo build --release -p peregrine`.
- llama.cpp: v0.9.0-dev (build 1, commit `cd959ba`), GNU 13.3.0 for Linux aarch64,
  `~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-server` — the same harness
  cache build the earlier ARM embedding comparison used. It loads `modern-bert`
  without trouble.

## Method

`run.py` in this directory (standard library only). One process per engine, all three
kept up at once, and the nine timed rounds interleave them in a rotating order so a
frequency or thermal drift is not charged to one engine. Each cell is one
`POST /v1/embeddings` request, timed wall-clock; `texts/s` is texts ÷ wall. The table
reports the median of nine runs and the min–max spread.

- **Cells**: one text of 16, 128, 512 and 2 000 tokens, plus one batch of 16 texts of
  32 tokens (512 tokens per request). Every text is built from
  `crates/peregrine-models/tests/fixtures/modern-bert/texts.txt` and calibrated with
  llama-server's own `/tokenize` (`add_special: true`): the four windows came out at
  exactly 16, 128, 512 and 2 000 tokens, and each batch text at 32.
- **Peregrine**: `serve -m <file> -t 10 -c 2048 --slots 32 --no-profile --opt
  EMBED_CACHE_MB=0`. The exact embedding-vector cache is **off** for every row: left
  on (the 64 MiB default), a repeated text is a lookup, not a pass, and the same text
  measured 1.35 s cold and 6 ms cached. Both engines therefore compute every text.
- **llama.cpp**: `-t 10 -c 65792 -b 2048 -ub 2048 -np 32 -fa auto --embedding
  --pooling cls`, i.e. `2048 + 8` tokens per slot × 32 slots, the flags of the earlier
  ARM embedding comparison. `--pooling cls` is the file's own pooling type
  (`peregrine inspect` reports `pooling cls (2)`); forcing `mean` gives a different
  vector (cosine 0.14 at 2 000 tokens), and llama.cpp is never handicapped here.
- **Vectors**: the run's first response per cell is kept; the cosine is Peregrine's
  vector against llama.cpp's for the same text, min and mean over the cell's texts.

The engine command lines are recorded verbatim in `summary.json` (with `~` for the
home directory). Reproduce with:

```sh
python3 run.py --out summary.json --runs 9
```

## Results

`texts/s`, median of nine runs; `spread` is the after-build's min–max. `ratio` is
Peregrine after ÷ llama.cpp, above 1.00 means Peregrine is ahead.

### gte-modernbert-base Q8_0 (160,208,512 bytes, `d3c637bb…`)

| Cell | Tokens | Before | After | llama.cpp | Ratio after ÷ llama | Spread (after) | Before ÷ llama |
|---|---|---|---|---|---|---|---|
| L16 b1 | 16 | 77.457 | 76.686 | 70.578 | **1.09×** | 74.090–80.411 | 1.10× |
| L128 b1 | 128 | 12.387 | 12.345 | 14.164 | 0.87× | 12.310–12.417 | 0.87× |
| L512 b1 | 512 | 3.146 | 3.144 | 3.584 | 0.88× | 2.990–3.152 | 0.88× |
| L2000 b1 | 2000 | 0.745 | 0.746 | 0.792 | 0.94× | 0.740–0.750 | 0.94× |
| batch 16×32 | 512 | 48.388 | 48.321 | 62.390 | 0.77× | 47.689–48.610 | 0.78× |

### gte-modernbert-base F16 (299,876,992 bytes, `2cda419c…`)

| Cell | Tokens | Before | After | llama.cpp | Ratio after ÷ llama | Spread (after) | Before ÷ llama |
|---|---|---|---|---|---|---|---|
| L16 b1 | 16 | 106.935 | 106.395 | 68.151 | **1.56×** | 103.446–111.334 | 1.57× |
| L128 b1 | 128 | 20.340 | 20.234 | 14.235 | **1.42×** | 20.151–20.356 | 1.43× |
| L512 b1 | 512 | 5.270 | 5.282 | 3.680 | **1.44×** | 5.253–5.315 | 1.43× |
| L2000 b1 | 2000 | 1.179 | 1.178 | 0.820 | **1.44×** | 1.174–1.185 | 1.44× |
| batch 16×32 | 512 | 74.456 | 73.746 | 64.041 | **1.15×** | 73.687–73.893 | 1.16× |

The two builds are within the run spread of each other in every cell: the after ÷ before
ratios are 0.990–1.001 on Q8_0 and 0.995–1.002 on F16. That is expected. The two NEON
bodies cover the LayerNorm and the GeGLU over 768 and 2×1 152 values per layer; the Q8_0
and F16 matrices are 22 × (768·2 304 + 768·1 152) multiply-adds, so the element-wise work
is around 1 % of a pass even after the scalar `erf` series. The change is a correctness
and completeness step (every pinned ARM tier now has a vector path), not a throughput
step.

llama.cpp's batch-1 cells at 16 and 128 tokens are short (14–70 ms) and its spread there
is 12–19 %; its medians vary by a few percent between runs. The L512, L2000 and batch
cells are steady (spread under 6 %), and those are the columns to read for the ratio.
On Q8_0 llama.cpp's i8mm GEMM is 6–29 % ahead of Peregrine's from 128 tokens up; on F16
Peregrine is 15–56 % ahead everywhere. That is a kernel-format difference, not a
`modern-bert` graph difference: the same family was 1.29× ahead on Q8_0 at 128 tokens on
the x86 host, and llama.cpp's ARM Q8_0 path is the stronger one here.

### Vectors: cosine of Peregrine against llama.cpp

| Model | L16 | L128 | L512 | L2000 | batch 16×32 (min / mean) |
|---|---|---|---|---|---|
| Q8_0 | 0.9998901 | 0.9999010 | 0.9997947 | 0.9997825 | 0.9997992 / 0.9998498 |
| F16 | 0.9999999 | 1.0000000 | 0.9999999 | 0.9999999 | 0.9999999 / 1.0000000 |

Before and after are the same to the last bit in every cell, as the bit-equality tests
require. The Q8_0 distance is the int8 tile's, and matches the 200-text fixture run
(min 0.9969122 over 200 texts, mean 0.9998028, against llama.cpp's own Q8_0) — these
five texts happen to sit closer.

## The ignored real-file test

With `gte-modernbert-base-F16.gguf` and `-Q8_0.gguf` in `~/models/`
(`keisuke-miyako/gte-modernbert-base-gguf`, SHA-256 `2cda419c…` / `d3c637bb…`):

```sh
cargo test --release -p peregrine-models --test embeddings \
  the_modern_bert_real_files_match_llama_cpp -- --ignored --nocapture
```

**Passes** in 47 s: gte F16 200 texts min/mean cosine 0.9999936/0.9999997 against
llama.cpp; gte Q8_0 0.9969122/0.9998028, above the F16 reference's Q8_0 distance
(0.9963540/0.9996125) by 1e-4 on the mean and 6e-6 on the min, inside the S4 band. The
two Granite Embedding R2 files are not on this machine and their cases skip with a
notice, as the test documents.

## What was not done

- Only `gte-modernbert-base` was measured. Granite Embedding R2 is the same graph and
  was not downloaded for this run; the earlier x86 table covers the family on llama.cpp.
- The AVX2 bodies for the same two kernels are another worker's change; this branch
  only adds the aarch64 dispatch and body.
- No batch-32-at-2 000-tokens cell (it exceeds the model's batch cap, the same skip the
  earlier embedding table records).
