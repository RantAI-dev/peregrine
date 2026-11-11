# S4.9k baseline: where Peregrine stands after S4.9a-g (bluefin, 2026-11-11)

Machine: bluefin, Ryzen 3 PRO 4350G (4 cores / 8 threads, AVX2+FMA, 512 KiB L2 per core),
13 GB RAM, Linux. Engine: `peregrine serve` + `/v1/embeddings` and `/v1/systemone`, built
from `origin/main` at `d211c49` (S4.9e merged) with
`CARGO_TARGET_DIR=/var/tmp/target-s49k cargo build --release -p peregrine` and rustc 1.99.1.
Competitor: llama.cpp v0.9.0 (`cd959ba`, `/var/tmp/llama.cpp/build/bin/llama-server`).
Models: `EmbeddingGemma-300M` Q8_0 and `Qwen3-Embedding-0.6B` Q8_0 and F16, under
`~/models/embed/`; decisions on `tinyopenjev-for-testing-Q8_0.gguf`.

## Method

- **Embeddings**: `benchmarks/embed.py`, three interleaved rounds (one engine at a time,
  each round a separate invocation), median of the three per-round `texts/s`. Same
  40 calibrated texts per length for every engine (`benchmarks/embed_texts.py`,
  16/128/512/2000 tokens within 5%; the mixed set is the S4.9b fixed seed 20261111,
  32 texts of 8-512 tokens, 3 516 tokens). `texts/s` and `tokens/s` use the engine's
  usage count; p50 is the whole-request latency. One request per timed wave for big
  batches, 20-30 requests for batch 1 (`--budget 1.5`). Serving context 2 048 tokens.
  llama.cpp runs with `-b 2048 -ub 2048`: a non-causal embedding model must have the
  whole prompt in one ubatch, and its default 1 024 fails a 2 000-token text. `-fa auto`
  was the faster flash-attention setting on this host (at 16 tokens `-fa off` was 48.8
  against 54.9 texts/s; at 128 and 512 they were level), so every llama.cpp row is
  `-fa auto`.
- **Decisions**: `benchmarks/decide.py` (new; one state, `noul` questions in one
  `/v1/systemone` request), `benchmarks/decide_texts.py` for the token-calibrated states
  (64/300/1000 tokens). Seven requests per cell per round, three rounds, median. Four
  slots on Peregrine; llama.cpp runs `-np 11`, one slot for each of the largest request's
  ten questions plus a spare, with `-c 16384` so a 1 000-token state fits every slot.
  That is the fair setting: llama.cpp answers one request's questions as a group that
  shares its common prefix (`server_decision_group_tasks`), so with fewer slots than
  questions the request is split into several groups and the state is evaluated once per
  group. Both engines see the same requests. Three llama.cpp settings were measured per
  cell (below); the table reports the fastest, because the engines are compared at their
  best documented setting.
- **Profile**: `crates/peregrine-bench/examples/cellprof.rs` (new) drives the same
  `Engine` and prints `StepStats` plus the `PEREGRINE_PROF2=1` buckets for one request of
  each cell, with tokenisation timed apart. `benchmarks/prof_cells.py` (new) is the
  same measurement over the HTTP endpoint; the table here uses `cellprof` because the
  encoder's step statistics are the only matmul/attention split the endpoint exposes.
- **Idle host**: `ps` showed no work between runs; the harness's 1-minute load gate was
  relaxed above 0.5 because the only load was the previous timed run decaying. Medians
  of the three rounds agree within 3% except the two llama.cpp 2 000-token cells, whose
  single-request rounds spread up to 12%.
- **Skips**: `L2000_b32` (64 000 tokens) exceeds every model's 32 768-token batch cap
  and llama.cpp's 2 048-token-per-slot × 10-slot context; it is not measured by either
  engine. No other cell was skipped.

Every number in the tables below came from the harness (`embed.py`, `decide.py`); the
profile numbers came from `cellprof.rs` and are not used as throughput.

## Results, 4 threads

`ratio` is Peregrine / llama.cpp: above 1.00 is Peregrine ahead.

# Embedding standings

## embeddinggemma q8_0

| cell | peregrine texts/s | llama.cpp texts/s | ratio | peregrine p50 ms | llama.cpp p50 ms |
|---|---:|---:|---:|---:|---:|
| L16_b1_c1 | 75.97 | 50.68 | **1.50×** | 12.9 | 19.1 |
| L16_b8_c1 | 84.45 | 54.51 | **1.55×** | 94.4 | 146.5 |
| L16_b32_c1 | 85.41 | 54.91 | **1.56×** | 374.2 | 582.2 |
| L128_b1_c1 | 10.07 | 7.72 | **1.31×** | 99.2 | 129.2 |
| L128_b8_c1 | 10.17 | 7.79 | **1.31×** | 786.2 | 1026.6 |
| L128_b32_c1 | 10.16 | 7.80 | **1.30×** | 3150.0 | 4102.0 |
| L512_b1_c1 | 2.02 | 1.82 | **1.11×** | 495.9 | 547.1 |
| L512_b8_c1 | 2.01 | 1.84 | **1.09×** | 3983.6 | 4344.5 |
| L512_b32_c1 | 2.01 | 1.84 | **1.09×** | 15937.8 | 17354.4 |
| L2000_b1_c1 | 0.42 | 0.39 | **1.07×** | 2402.9 | 2583.5 |
| L2000_b8_c1 | 0.42 | 0.39 | **1.08×** | 19161.3 | 20730.5 |
| L2000_b32_c1 | skip | skip | – | – | – |
| mixed_b32_c1 | 10.70 | 8.78 | **1.22×** | 2991.1 | 3646.1 |

## qwen3-embed-0.6b f16

| cell | peregrine texts/s | llama.cpp texts/s | ratio | peregrine p50 ms | llama.cpp p50 ms |
|---|---:|---:|---:|---:|---:|
| L16_b1_c1 | 17.97 | 12.78 | **1.41×** | 55.4 | 78.1 |
| L16_b8_c1 | 16.86 | 12.95 | **1.30×** | 474.1 | 617.7 |
| L16_b32_c1 | 17.37 | 12.96 | **1.34×** | 1841.4 | 2469.3 |
| L128_b1_c1 | 2.09 | 1.82 | **1.15×** | 478.4 | 550.3 |
| L128_b8_c1 | 2.16 | 1.82 | **1.18×** | 3709.7 | 4394.9 |
| L128_b32_c1 | 2.13 | 1.82 | **1.17×** | 15005.8 | 17593.2 |
| L512_b1_c1 | 0.51 | 0.43 | **1.19×** | 1963.1 | 2335.9 |
| L512_b8_c1 | 0.50 | 0.43 | **1.17×** | 15982.0 | 18667.3 |
| L512_b32_c1 | 0.51 | 0.43 | **1.18×** | 63188.0 | 74663.1 |
| L2000_b1_c1 | 0.10 | 0.08 | **1.23×** | 9669.2 | 11910.7 |
| L2000_b8_c1 | 0.10 | 0.08 | **1.24×** | 77167.1 | 94699.9 |
| L2000_b32_c1 | skip | skip | – | – | – |
| mixed_b32_c1 | 2.42 | 2.05 | **1.18×** | 13232.6 | 15619.8 |

## qwen3-embed-0.6b q8_0

| cell | peregrine texts/s | llama.cpp texts/s | ratio | peregrine p50 ms | llama.cpp p50 ms |
|---|---:|---:|---:|---:|---:|
| L16_b1_c1 | 21.43 | 15.63 | **1.37×** | 46.2 | 63.7 |
| L16_b8_c1 | 21.77 | 15.89 | **1.37×** | 367.2 | 503.2 |
| L16_b32_c1 | 21.86 | 15.89 | **1.38×** | 1463.1 | 2012.9 |
| L128_b1_c1 | 2.71 | 2.20 | **1.23×** | 367.1 | 453.7 |
| L128_b8_c1 | 2.70 | 2.21 | **1.22×** | 2966.0 | 3628.1 |
| L128_b32_c1 | 2.66 | 2.20 | **1.21×** | 12023.2 | 14542.5 |
| L512_b1_c1 | 0.62 | 0.50 | **1.25×** | 1604.2 | 1999.3 |
| L512_b8_c1 | 0.62 | 0.50 | **1.23×** | 12975.8 | 15938.1 |
| L512_b32_c1 | 0.62 | 0.50 | **1.24×** | 51431.7 | 63894.0 |
| L2000_b1_c1 | 0.12 | 0.08 | **1.49×** | 8270.1 | 12290.8 |
| L2000_b8_c1 | 0.12 | 0.08 | **1.44×** | 66136.8 | 95697.3 |
| L2000_b32_c1 | skip | skip | – | – | – |
| mixed_b32_c1 | 2.99 | 2.44 | **1.23×** | 10691.1 | 13107.0 |

## Stage profile (Peregrine)

### embeddinggemma-Q8_0, 4 threads

| cell | request ms | tokenise ms | engine ms | matmul | attention | quantise | barrier wait | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L16_b1 | 15.4 | 0.08 | 15.3 | 81.4% | 2.5% | 1.7% | 1.3% | 13.1% |
| L16_b8 | 91.9 | 0.48 | 91.8 | 79.9% | 2.7% | 1.7% | 2.2% | 13.5% |
| L16_b32 | 360.7 | 1.31 | 360.5 | 80.1% | 2.6% | 1.7% | 2.3% | 13.2% |
| L128_b1 | 98.2 | 0.44 | 98.2 | 77.4% | 7.1% | 1.6% | 2.2% | 11.7% |
| L128_b8 | 779.0 | 2.38 | 778.9 | 77.3% | 7.2% | 1.6% | 2.0% | 11.8% |
| L128_b32 | 3132.5 | 8.58 | 3132.0 | 77.2% | 7.1% | 1.6% | 2.6% | 11.5% |
| L512_b1 | 528.6 | 1.54 | 528.6 | 60.9% | 25.5% | 1.7% | 8.3% | 3.7% |
| L512_b8 | 3993.4 | 8.88 | 3993.3 | 61.3% | 25.3% | 1.7% | 7.3% | 4.5% |
| L512_b32 | 16000.2 | 33.47 | 15999.7 | 61.3% | 25.2% | 1.7% | 7.3% | 4.6% |
| L2000_b1 | 2475.4 | 4.36 | 2475.4 | 48.8% | 38.8% | 1.5% | 4.6% | 6.2% |
| L2000_b8 | 19668.0 | 32.08 | 19667.9 | 48.8% | 38.8% | 1.6% | 4.7% | 6.1% |
| mixed_b32 | 2991.6 | 7.52 | 2991.2 | 69.8% | 15.6% | 1.7% | 4.2% | 8.8% |

### qwen3-embed-0.6b-f16, 4 threads

| cell | request ms | tokenise ms | engine ms | matmul | attention | quantise | barrier wait | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L16_b1 | 54.9 | 0.08 | 54.5 | 90.1% | 1.6% | 2.1% | 2.4% | 3.9% |
| L16_b8 | 476.4 | 0.45 | 473.9 | 90.3% | 1.5% | 2.0% | 3.4% | 3.0% |
| L16_b32 | 1849.6 | 1.31 | 1838.8 | 90.0% | 1.5% | 2.0% | 3.3% | 3.2% |
| L128_b1 | 485.6 | 0.43 | 484.0 | 89.0% | 3.0% | 1.9% | 2.8% | 3.3% |
| L128_b8 | 3754.7 | 2.31 | 3741.0 | 88.7% | 3.1% | 2.0% | 2.9% | 3.4% |
| L128_b32 | 15013.7 | 8.29 | 14960.8 | 88.6% | 3.1% | 2.0% | 3.1% | 3.3% |
| L512_b1 | 2003.7 | 1.47 | 1997.8 | 83.2% | 8.8% | 1.9% | 2.9% | 3.2% |
| L512_b8 | 16032.5 | 8.72 | 15979.8 | 83.2% | 8.9% | 1.9% | 3.0% | 3.1% |
| L512_b32 | 63262.8 | 33.28 | 63152.9 | 84.1% | 9.0% | 1.9% | 3.2% | 1.9% |
| L2000_b1 | 9591.9 | 4.50 | 9565.6 | 67.7% | 25.7% | 1.7% | 2.8% | 2.1% |
| L2000_b8 | 76706.1 | 31.93 | 76488.5 | 67.6% | 25.7% | 1.7% | 2.8% | 2.3% |
| mixed_b32 | 13237.4 | 7.30 | 13187.1 | 86.4% | 5.6% | 2.0% | 2.9% | 3.1% |

### qwen3-embed-0.6b-q8_0, 4 threads

| cell | request ms | tokenise ms | engine ms | matmul | attention | quantise | barrier wait | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L16_b1 | 45.4 | 0.08 | 45.1 | 88.6% | 2.0% | 2.1% | 2.9% | 4.4% |
| L16_b8 | 364.7 | 0.47 | 362.4 | 88.6% | 2.0% | 2.1% | 2.7% | 4.7% |
| L16_b32 | 1456.0 | 1.31 | 1446.3 | 88.9% | 2.0% | 2.1% | 2.3% | 4.7% |
| L128_b1 | 375.2 | 0.41 | 373.8 | 86.2% | 4.1% | 2.0% | 2.9% | 4.8% |
| L128_b8 | 3168.4 | 2.29 | 3156.4 | 85.8% | 4.4% | 2.2% | 3.7% | 4.0% |
| L128_b32 | 12032.8 | 8.16 | 11979.8 | 86.6% | 4.2% | 2.1% | 2.8% | 4.4% |
| L512_b1 | 1614.3 | 1.45 | 1607.9 | 80.3% | 11.4% | 1.9% | 2.1% | 4.3% |
| L512_b8 | 13011.7 | 8.60 | 12960.8 | 79.8% | 11.4% | 1.9% | 2.8% | 4.1% |
| L512_b32 | 51474.6 | 33.06 | 51362.4 | 80.9% | 11.6% | 2.0% | 2.6% | 3.0% |
| L2000_b1 | 8214.1 | 4.49 | 8186.3 | 62.2% | 30.9% | 1.6% | 2.0% | 3.3% |
| L2000_b8 | 65584.9 | 32.03 | 65367.0 | 62.1% | 30.8% | 1.7% | 2.3% | 3.1% |
| mixed_b32 | 10700.3 | 7.30 | 10651.6 | 83.8% | 7.4% | 2.0% | 2.5% | 4.4% |

## Decisions (/v1/systemone)

### peregrine, 4 threads

| state tokens | questions | questions/s |
|---:|---:|---:|
| 64 | 1 | 174.27 |
| 64 | 3 | 234.24 |
| 64 | 10 | 292.35 |
| 300 | 1 | 62.07 |
| 300 | 3 | 120.91 |
| 300 | 10 | 189.92 |
| 1000 | 1 | 19.64 |
| 1000 | 3 | 46.83 |
| 1000 | 10 | 91.55 |

### llama_cpp, 4 threads

| state tokens | questions | questions/s |
|---:|---:|---:|
| 64 | 1 | 115.29 |
| 64 | 3 | 121.77 |
| 64 | 10 | 125.11 |
| 300 | 1 | 45.74 |
| 300 | 3 | 46.80 |
| 300 | 10 | 47.97 |
| 1000 | 1 | 15.15 |
| 1000 | 3 | 15.59 |
| 1000 | 10 | 15.72 |

## 8 threads, 16-token cells

On this 4-core/8-thread part, 8 threads never helped Peregrine on the encoder (68-77
against 76-85 texts/s) and never helped either engine at 128 tokens or more (a full
8-thread screening of EmbeddingGemma was 9.5 against 10.1 at 128, 2.0 against 2.0 at
512, 0.43 against 0.43 at 2 000). It helped at 16 tokens: both engines gain 4-8% on the
causal Qwen3 file, and llama.cpp gains 4-8% on the encoder. The cells where it helps
are below; medians of three rounds.

| cell | peregrine 4t | peregrine 8t | llama.cpp 4t | llama.cpp 8t | 8t helps |
|---|---:|---:|---:|---:|---|
| Qwen3 Q8 L16_b1 | 21.43 | **22.86** | 15.63 | **16.50** | both, +5-7% |
| Qwen3 Q8 L16_b8 | 21.77 | **22.96** | 15.89 | **16.84** | both, +6% |
| Qwen3 Q8 L16_b32 | 21.86 | **22.93** | 15.89 | **16.81** | both, +5-6% |
| Qwen3 F16 L16_b1 | 17.97 | 17.74 | 12.78 | **13.43** | llama only |
| Qwen3 F16 L16_b8 | 16.86 | 16.56 | 12.95 | **13.75** | llama only |
| Qwen3 F16 L16_b32 | 17.37 | 17.20 | 12.96 | **13.76** | llama only |
| Gemma Q8 L16_b1 | **75.97** | 68.25 | 50.68 | **54.36** | llama only |
| Gemma Q8 L16_b8 | **84.45** | 76.49 | 54.51 | **57.13** | llama only |
| Gemma Q8 L16_b32 | **85.41** | 77.23 | 54.91 | **57.85** | llama only |

The rest of this report is 4 threads, the count both engines are configured for by the
stage's instruction.

## Where Peregrine is behind

**Nowhere in the measured grid.** Every 4-thread cell is at or above 1.07x llama.cpp;
the 8-thread table has the same shape. The closest cells, and the stage that explains
each:

| cell | ratio | the stage |
|---|---:|---|
| EmbeddingGemma L2000_b1 | **1.07x** | attention, 38.8% of the pass (matmul 48.8%); the encoder's attention is still scalar-tiled per text |
| EmbeddingGemma L2000_b8 | **1.08x** | the same 38.8% attention share, and the 2 000-token text is one pass in both engines |
| EmbeddingGemma L512_b32 | **1.09x** | attention 25.2% against matmul 61.3%; the gap is the strip's score stride (S4.9g "what is left") |
| Qwen3-Embedding F16 L128_b1 | **1.15x** | matmul is 89-90% of the pass and the float panel kernel is 0.75x the Q8_0 tile per multiply; llama.cpp quantises activations |

The causal Q8_0 file, which S4.9f measured 1.29-1.33x behind at 109/437 tokens, is now
1.21-1.49x **ahead** at every length and batch; the S4.9b token-budget pass and the
pass-size changes closed that gap, not a new kernel.

The profile says the next step should look at (a) the encoder's attention at 512-2 000
tokens (25-39% of the pass, the largest single lever left), (b) the float panel matmul
(90% of an F16 short-text pass), and (c) small-work overheads: 336-2 688 barriers for a
16-token pass, 2-3% barrier wait plus the dispatch cost inside the matmul wall, and
0.4-4.5 ms of tokenisation for the long texts (0.08 ms at 16 tokens).

## Decisions, `/v1/systemone`, 4 threads

One shared state, 1/3/10 `noul` questions in one request; questions per second and the
ratio (Peregrine / llama.cpp). Medians of three rounds, seven requests per cell per round,
both engines re-measured in the same session on 2026-11-11. The llama.cpp column is that
engine's fastest of the three settings measured per cell: `-np 11, -c 16384, -ub 512`
(the fair default in `decide.py`), and for the two 10-question cells `-np 11, -c 16384,
-ub 2048 -kvu --cache-ram 256`, which is 3-6% faster there. `-np 1, -c 8192` was measured
as the historical setting.

| state tokens | questions | peregrine q/s | llama.cpp q/s | ratio | llama.cpp setting |
|---:|---:|---:|---:|---:|---|
| 64 | 1 | 175.37 | 118.62 | **1.48x** | -np 11, -c 16384, -ub 512 |
| 64 | 3 | 208.42 | 162.04 | **1.29x** | -np 11, -c 16384, -ub 512 |
| 64 | 10 | 262.87 | 216.01 | **1.22x** | -np 11, -c 16384, -ub 2048, -kvu, cache-ram 256 |
| 300 | 1 | 62.79 | 46.73 | **1.34x** | -np 11, -c 16384, -ub 512 |
| 300 | 3 | 123.47 | 88.82 | **1.39x** | -np 11, -c 16384, -ub 512 |
| 300 | 10 | 193.79 | 138.98 | **1.39x** | -np 11, -c 16384, -ub 2048, -kvu, cache-ram 256 |
| 1000 | 1 | 20.13 | 15.64 | **1.29x** | -np 11, -c 16384, -ub 512 |
| 1000 | 3 | 47.79 | 35.70 | **1.34x** | -np 11, -c 16384, -ub 512 |
| 1000 | 10 | 93.63 | 67.75 | **1.38x** | -np 11, -c 16384, -ub 512 |

The first version of this table ran llama.cpp on **one slot** and reported 1.30-5.83x for
Peregrine. That was wrong: llama.cpp groups the questions of one request into slots and
forwards their common prefix once per group (`server_decision_group_tasks`), so a single
slot evaluates the state once per question and the ratio grew with the question count.
With as many slots as questions the state is forwarded once, llama.cpp's per-question cost
falls from 63 ms to 14.8 ms at 1 000 tokens and from 20.6 ms to 7.3 ms at 300, and the real
same-day advantage is 1.22-1.48x, not a multiple.

Being exact about what remains: llama.cpp still pays one tail per question and its Q8_0
group decode spreads the ten tails over ten slots, while Peregrine's S4.9j fork forwards
the shared state once into four slots and then shares the tails' weight passes. On this
host the honest decision table is 1.2-1.5x for Peregrine, and the 10-question row is the
narrowest because llama.cpp's grouping is at its best there.

The one-slot numbers, kept in `decide/one-slot/`, are with `-np 1, -c 8192, -ub 512`:
115-125 q/s at 64 and 46-48 at 300 whatever the question count, and 15.1-15.9 at 1 000.

## Raw data

- `*-r{1,2,3}.json` — the three embedding rounds per model/engine/precision.
- `t8/*-r{1,2,3}.json` — the 8-thread 16-token cells.
- `decide/fair/decide-{peregrine,llama_cpp}-r{1,2,3}.json` — the same-day decision cells
  (4 Peregrine slots; llama.cpp `-np 11, -c 16384, -ub 512`).
- `decide/kvu/decide-llama_cpp-r{1,2,3}.json` — llama.cpp `-np 11, -c 16384, -ub 2048,
  -kvu, --cache-ram 256`, the fastest setting at 10 questions of a 64- or 300-token state.
- `decide/one-slot/decide-llama_cpp-r{1,2,3}.json` — the historical `-np 1, -c 8192`
  setting, kept to show why it was the wrong comparison.
- `profile/*-t4-profile.json` — the per-cell stage profile.
- `*-skips.json` — the one skipped cell and why.

Tables above are reproduced with `python benchmarks/standings_report.py --dir <this dir>`.
