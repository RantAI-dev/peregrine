# An `i8mm` body for the Q8_0 tile (S4.9q): what it buys on the int8 tile route

S4.9o made a Q8_0 `modern-bert` file on the two int8 ARM tiers take the f32-panel route, so
what is left on the repacked Q8_0 int8 tile on `aarch64-neon-i8mm` is the other Q8_0 files:
`gemma-embedding`, `qwen3`, `mistral3` encoders, and any `bert`/`modern-bert` Q8_0 file too big
for the load-time float conversion. All of them ran the tile's `sdot` body, because
`q4k8_gemm_i8mm`, `q5_08_gemm_i8mm` and `q6k8_gemm_i8mm` existed and `q8_08_gemm_i8mm` did not.
This round writes that body and measures it, kernel alone and end to end.

- Date: 2026-11-12, 09:40–11:05 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters: 10 × Cortex-X925
  at 3.9 GHz (CPUs 5–9 and 15–19) and 10 × Cortex-A725 at 2.8 GHz (the rest), 121 GB unified
  memory, Linux `6.17.0-1008-nvidia`, governor `performance`, kernel tier `aarch64-neon-i8mm`,
  rustc 1.99.1.
- Peregrine before: `374e3cc`, the merge of `origin/main` (`84b7d2b`, which is S4.9o's #228)
  into the S4.9o branch — i.e. main with S4.9o and without this change, built from a clean
  worktree with `cargo build --release -p peregrine`.
- Peregrine after: `55386b2` (`perf: an i8mm body for the Q8_0 tile (S4.9q)`), same command.
- llama.cpp: v0.9.0-dev (build 1, commit `cd959ba`), GNU 13.3.0 for Linux aarch64,
  `~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-server`, `CUDA_VISIBLE_DEVICES=""`
  so the CPU path is measured.
- Threads: 10, the Cortex-X925 cluster, every engine pinned — `taskset -c 5-9,15-19`.

## The host is shared

A co-tenant (`vllm`'s engine processes) sat on the machine for the whole window: the 1-minute
load average was 2.7–3.0 at the start of the EmbeddingGemma run, 8.1 at its end, and 8–10 for
the bge-m3, Qwen3 and Nemotron runs; `MemAvailable` stayed at 35–36 GB of 121. Every arm of a
model stays up and the five timed rounds interleave them in a rotating order, so a drifting
co-tenant is charged to all three arms alike; the medians' spreads are 0.4–6.8 % (1 % on most
cells). Absolute throughput is indicative; the before/after and against-llama ratios are the
claims.

## Per-kernel speed: `sdot` body against the `smmla` body

`cargo run --release -p peregrine-bench --example neonbench -- 4096 512 300`, one 4 096-weight
layer row (the example's shape, which the earlier ARM rounds used) and the tile's eight output
rows; the reported GMAC/s counts `8 · nb · n` multiply-accumulates per call, i.e. the same
arithmetic the `sdot` and `smmla` bodies do. One thread is the example pinned to core 5, median
of ten runs. Ten threads is three rounds of ten copies pinned one per X925 core, the ten
reported GMAC/s summed per line, the median over the rounds — the example is single-threaded, so
the aggregate is the tile's throughput with the cluster busy.

| activation rows | `sdot` 1T | `smmla` 1T | ratio | `sdot` 10T | `smmla` 10T | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (GEMV tail, still `sdot`) | 36.4 | 36.5 | 1.00× | 144.8 | 142.4 | 0.98× |
| 8 | 37.9 | **115.8** | **3.06×** | 150.3 | **454.0** | **3.02×** |
| 64 | 37.9 | **115.9** | **3.06×** | 150.4 | **453.0** | **3.01×** |
| 512 | 37.4 | **114.0** | **3.05×** | 147.1 | **442.0** | **3.00×** |

The eight-row batch, which is the smallest group the `smmla` body takes, already runs at the
`smmla` rate; the one-row case is the `sdot` GEMV and is unchanged by design. Raw runs:
`microbench.txt`.

## End to end

`texts/s`, median of five interleaved rounds; `spread` is the min–max over the five, for the
before arm unless noted (the after arm's spreads are 0.4–2.5 %). Cells: one text of 16, 128, 512
and 2 000 tokens; a batch of 32 texts of 128 tokens; and the mixed-length batch of 32 texts
drawn from 8–512 tokens with seed `20261111` (3 505–3 567 tokens), the same recipe as the
`spark-2026-11-12-embeddings` run — the three files that run measured reuse its texts verbatim,
bge-m3's batch was built with the same recipe and llama-server's tokenizer.

### EmbeddingGemma 300M Q8_0 (334 MB)

| Cell | Before | After | llama.cpp | Gain | After ÷ llama | Before ÷ llama | Spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| one text, 16 tok | 89.645 | **138.569** | 69.776 | 1.55× | **1.99×** | 1.28× | 6.6 % |
| one text, 128 tok | 14.820 | **29.327** | 11.874 | 1.98× | **2.47×** | 1.25× | 1.5 % |
| one text, 512 tok | 3.517 | **6.709** | 3.242 | 1.91× | **2.07×** | 1.08× | 1.4 % |
| one text, 2 000 tok | 0.895 | **1.669** | 0.843 | 1.86× | **1.98×** | 1.06× | 1.0 % |
| batch 32 × 128 tok | 15.257 | **30.632** | 16.051 | 2.01× | **1.91×** | 0.95× | 1.3 % |
| mixed batch of 32 | 17.023 | **33.425** | 17.215 | 1.96× | **1.94×** | 0.99× | 1.6 % |

### Qwen3-Embedding 0.6B Q8_0 (639 MB)

| Cell | Before | After | llama.cpp | Gain | After ÷ llama | Before ÷ llama | Spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| one text, 16 tok | 32.233 | **63.619** | 28.888 | 1.97× | **2.20×** | 1.12× | 5.0 % |
| one text, 128 tok | 4.240 | **8.625** | 4.163 | 2.03× | **2.07×** | 1.02× | 0.4 % |
| one text, 512 tok | 0.931 | **1.778** | 0.994 | 1.91× | **1.79×** | 0.94× | 1.8 % |
| one text, 2 000 tok | 0.145 | **0.286** | 0.215 | 1.97× | **1.33×** | 0.67× | 6.8 % |
| batch 32 × 128 tok | 4.202 | **8.643** | 4.271 | 2.06× | **2.02×** | 0.98× | 1.0 % |
| mixed batch of 32 | 4.647 | **9.217** | 4.765 | 1.98× | **1.93×** | 0.98× | 1.1 % |

### Nemotron 3.5 Embed 1B Q8_0 (1.22 GB)

| Cell | Before | After | llama.cpp | Gain | After ÷ llama | Before ÷ llama | Spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| one text, 16 tok | 18.760 | **40.602** | 17.101 | 2.16× | **2.37×** | 1.10× | 1.5 % |
| one text, 128 tok | 2.410 | **6.241** | 2.352 | 2.59× | **2.65×** | 1.02× | 1.6 % |
| one text, 512 tok | 0.586 | **1.453** | 0.553 | 2.48× | **2.63×** | 1.06× | 1.2 % |
| one text, 2 000 tok | 0.139 | **0.301** | 0.126 | 2.16× | **2.39×** | 1.11× | 0.9 % |
| batch 32 × 128 tok | 2.428 | **6.243** | 2.374 | 2.57× | **2.63×** | 1.02× | 1.1 % |
| mixed batch of 32 | 2.799 | **7.021** | 2.671 | 2.51× | **2.63×** | 1.05× | 1.1 % |

### bge-m3 Q8_0 (635 MB)

| Cell | Before | After | llama.cpp | Gain | After ÷ llama | Before ÷ llama | Spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| one text, 16 tok | 42.953 | **63.533** | 34.636 | 1.48× | **1.83×** | 1.24× | 5.3 % |
| one text, 128 tok | 5.915 | **12.605** | 6.013 | 2.13× | **2.10×** | 0.98× | 1.0 % |
| one text, 512 tok | 1.424 | **2.911** | 1.460 | 2.04× | **1.99×** | 0.98× | 1.2 % |
| one text, 2 000 tok | 0.321 | **0.588** | 0.297 | 1.83× | **1.98×** | 1.08× | 1.0 % |
| batch 32 × 128 tok | 5.963 | **12.786** | 6.304 | 2.14× | **2.03×** | 0.95× | 0.9 % |
| mixed batch of 32 | 6.742 | **14.144** | 7.013 | 2.10× | **2.02×** | 0.96× | 1.3 % |

The bge-m3 file was not on this machine. It is `bge-m3-Q8_0.gguf` from
[`gpustack/bge-m3-GGUF`](https://huggingface.co/gpustack/bge-m3-GGUF) (the repository
`benchmarks/embed_models.json` names), 634,553,760 bytes, SHA-256
`950f4a8e5e19477a6d3c26d2f162233c20002c601f75e4b002e3239997821167`.

**Cells below 1.0×: none.** Every cell of every file is now ahead of llama.cpp; the narrowest
is Qwen3-Embedding Q8_0 at one 2 000-token text, 1.33×. Before this change ten cells of these
four files were below 1.0× (bge-m3 128 / 512 / batch / mixed at 0.95–0.98×, EmbeddingGemma's
batch of 32 at 0.95× and its mixed batch at 0.99×, Qwen3's 512-token, batch and mixed cells at
0.94–0.98×, and Qwen3's 2 000-token cell at 0.67×).

## The vectors did not move

Nothing about the arithmetic changed — see below — and the measurement says so: the raw f32
bits of every returned vector are **identical** between the two Peregrine arms, in every cell of
every file (69,632 / 52,224 / 139,264 / 69,632 values for bge-m3, EmbeddingGemma, Nemotron and
Qwen3), so the cosine between them is exactly 1.0. The cosine against llama.cpp's vectors is
identical to the last bit between the arms too, which is the end-to-end form of the kernel test:
the `aarch64-neon-i8mm` Q8_0 goldens were produced by the `sdot` body and the new body
reproduces them.

| File | Cell | cosine vs llama.cpp, min / mean (both arms) |
|---|---|---|
| EmbeddingGemma | one text (each) | 0.9995710 / — , 0.9997321, 0.9996630, 0.9997417 |
| EmbeddingGemma | batch 32 × 128 / mixed | 0.9992886 / 0.9996446, 0.9993783 / 0.9996249 |
| Qwen3-Embedding | one text (each) | 0.9997221, 0.9997695, 0.9997494, 0.9997489 |
| Qwen3-Embedding | batch 32 × 128 / mixed | 0.9995452 / 0.9997105, 0.9995392 / 0.9997182 |
| Nemotron | one text (each) | 0.9996776, 0.9999586, 0.9999844, 0.9999955 |
| Nemotron | batch 32 × 128 / mixed | 0.9997040 / 0.9999238, 0.9993188 / 0.9998000 |
| bge-m3 | one text (each) | 0.9997996, 0.9998452, 0.9996267, 0.9994535 |
| bge-m3 | batch 32 × 128 / mixed | 0.9994888 / 0.9997705, 0.9995629 / 0.9997783 |

## Why the bits cannot move

`smmla` computes a 2×8 signed by 8×2 signed product into four int32 sums; the Q8_0 tile's
window is exactly two output rows of eight codes, and two activation rows' eight codes joined
are the other operand, so one instruction forms 32 products. The integer sums are exact in any
grouping, and a Q8_0 block has one scale and no offset, so a block's four windows accumulate
into one int32 sum. The float epilogue is `kq::dot_q8_08_ref`'s: `(d·dk)` rounded once, then
one fused multiply-add per block, block by block in the tile's order — the `sdot` body's
expression in its order. The tests compare the new body with the `sdot` body and with
`kq::dot_q8_08_ref` bit for bit at every row count 1..=40 plus 63, 64, 65, 127, 128, 129, 511,
512, 513, at widths 32/64/96/160, on -128/127/alternating codes, a zero scale and activations at
the int8 limits.

## What the change is

- `peregrine_kernels::neon::q8_08_gemm_i8mm`, beside the `sdot` body
  (`q8_08_gemm`/`q8_08_gemv`): whole groups of four activation rows on `smmla`, the 1–3-row tail
  and the GEMV on `sdot`.
- `MatKernels::kq_tiles_q8_0` picks it when `kq_i8mm` (the `aarch64-neon-i8mm` tier); the
  dotprod tier is unchanged, so its measurements are the control.
- `kernels_for(Q8_0)` lists `neon-i8mm`, and `neonbench` benches both bodies.

## What was not done

- No Q8_0 chat model is on this machine (the chat files are Q4_K_M, Q2_K and Bonsai 2's PQ2_0),
  so there is no prompt/decode row to add; the tile path is the same one an encoder uses, and
  the kernel table is the decode-sized evidence (one row = +0 %, eight rows = 3.06×).
- The GEMV stays on `sdot`: `smmla` needs two activation rows, and one row's bits are pinned by
  the same test on both paths.
- Nothing changed on x86 (the AVX2 and AVX-512 Q8_0 tiles already have their own bodies) or on
  the two lower aarch64 tiers (no int8 tile there).
- The Qwen3-Embedding 2 000-token before arm is 0.145 texts/s where the S4.9l-era run recorded
  0.213 for the same file. A check of that run's binary (`3030172`, rebuilt) on the same cells
  measured 0.177 with this protocol, so something between that commit and S4.9o's head costs
  this one cell ~18 % against llama.cpp's unchanged 0.215; the same-round before/after A/B is
  not affected, and it is not this kernel. Not chased here.

## Commands

```sh
# Peregrine, both arms
cd ~/wt-s49q-before && git checkout 374e3cc && cargo build --release -p peregrine
cd ~/wt-s49q        && cargo build --release -p peregrine

# Kernel, both bodies (writes microbench.txt)
sh benchmarks/results/spark-arm-2026-11-12-q8-i8mm/bench-kernel.sh \
  > benchmarks/results/spark-arm-2026-11-12-q8-i8mm/microbench.txt

# End to end: one invocation per model, five interleaved rounds
cd benchmarks/results/spark-arm-2026-11-12-q8-i8mm
PIN_CPUS=5-9,15-19 python3 run.py --out summary-embeddinggemma.json --runs 5 \
  --models EmbeddingGemma-300M-Q8_0
PIN_CPUS=5-9,15-19 python3 run.py --out summary-qwen3.json --runs 5 \
  --models Qwen3-Embedding-0.6B-Q8_0
PIN_CPUS=5-9,15-19 python3 run.py --out summary-nemotron.json --runs 5 \
  --models Nemotron-3.5-Embed-1B-Q8_0
PIN_CPUS=5-9,15-19 python3 run.py --out summary-bge-m3.json --runs 5 --models bge-m3-Q8_0
```

`run.py` is in this directory (standard library only). It starts `peregrine serve -m <file>
-t 10 -c 2048 --slots 32 --no-profile --opt EMBED_CACHE_MB=0` with
`PEREGRINE_PACK_CACHE=0`, and llama-server with `-t 10 -np 32 -c 65792 -b 2048 -ub 2048 -fa
auto --embedding --pooling <the file's own>` (bge-m3: `-np 8 -c 16448`, the registry's row), all
under `taskset -c 5-9,15-19`. The exact command lines and the `PEREGRINE_ISA`/`CUDA` environment
of each arm are recorded in `summary.json`, with `~` for the home directory. `summary.json` also
holds every round's `texts/s`, the cell token counts, the cosine against llama.cpp, and the
before/after bit-identity verdict per cell. The raw vectors are not committed (43 MB of JSON);
the verdict is `identical: true` in every cell of every file.
