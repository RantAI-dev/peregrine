# S4.9b — token-budget passes and prompt-sized caches (bluefin, 2026-11-11)

Machine: bluefin, Ryzen 3 PRO 4350G (4 cores, AVX2+FMA, L2 512 KiB/core), 13 GB RAM,
4 threads. Engine: `peregrine serve` + `/v1/embeddings`, one server at a time.
Competitor: llama.cpp v0.9.0 (`cd959ba`, `/var/tmp/llama.cpp/build/bin/llama-server`).
Models: `Qwen3-Embedding-0.6B` Q8_0 (causal, 0.6 GB) and `EmbeddingGemma-300M` Q8_0
(bidirectional encoder, 0.3 GB), under `~/models/embed/`.

The timings compare **main at 1213f5d** — the commit this branch was cut from — with this branch at
`38dd0ea` (its engine change, before later `main` commits were merged) and with llama.cpp. `main`
has since gained S4.9a/c/f; those are merged into the branch but their gains are not claimed here
and are not in these tables, and the merge was re-validated by the test suite rather than re-timed
(the host was busy with another build by then). The Q8_0 path these tables measure is untouched by
S4.9f.

Method: `benchmarks/embed.py` (three interleaved rounds, median shown), one request per
timed wave, 2 waves per cell, 4 threads, `--budget 1.0`. `texts/s` and `tokens/s` (engine
usage), same 128 calibrated texts per length for every engine. `mixed_b32` is 32 texts whose
lengths were drawn from 8..512 with a fixed seed (`embed_texts.py --mixed-set 32
--mixed-seed 20261111`); the same 3 516 tokens for every engine.

The host was not quiescent (a desktop browser kept ~2 cores busy; load average 2-3,
`MemAvailable` ~3 GB), so treat differences under ~10% as noise.

## `main` (1213f5d) vs this branch vs llama.cpp

Cells are `L<token length>_b<batch>_c1`, texts/s / tokens/s. `skip`: the cell exceeds the
model's 32 768-token batch cap (128 x 512); llama.cpp would pad it to 65 536 tokens.

### Qwen3-Embedding 0.6B Q8_0 (causal)

| cell | main | branch | llama.cpp |
|---|---|---|---|
| L16_b1 | 19.43 / 311 | **21.42 / 343** | 15.53 / 248 |
| L16_b8 | 15.96 / 255 | **21.87 / 350** | 15.59 / 249 |
| L16_b32 | 18.83 / 301 | **21.70 / 347** | 15.70 / 251 |
| L16_b128 | 20.45 / 327 | **21.69 / 347** | 15.68 / 251 |
| L128_b1 | 2.59 / 332 | **2.71 / 346** | 2.18 / 279 |
| L128_b8 | 2.59 / 331 | **2.71 / 347** | 2.18 / 279 |
| L128_b32 | 2.63 / 337 | 2.65 / 339 | 2.18 / 279 |
| L128_b128 | 2.67 / 342 | 2.68 / 343 | 2.18 / 279 |
| L512_b1 | 0.60 / 307 | **0.62 / 318** | 0.50 / 255 |
| L512_b8 | 0.60 / 310 | **0.61 / 313** | 0.50 / 255 |
| L512_b32 | 0.61 / 314 | 0.61 / 315 | 0.50 / 254 |
| L512_b128 | skip | skip | skip |
| mixed_b32 | 2.96 / 325 | 2.98 / 327 | 2.41 / 265 |

### EmbeddingGemma 300M Q8_0 (bidirectional encoder)

| cell | main | branch | llama.cpp |
|---|---|---|---|
| L16_b1 | 47.99 / 768 | 47.99 / 768 | 50.75 / 812 |
| L16_b8 | 50.04 / 801 | **52.51 / 840** | 45.54 / 729 |
| L16_b32 | 51.10 / 818 | **52.59 / 842** | 50.72 / 812 |
| L16_b128 | 49.59 / 794 | **52.18 / 835** | 51.46 / 824 |
| L128_b1 | 4.51 / 577 | 4.64 / 595 | **7.03 / 900** |
| L128_b8 | 4.45 / 570 | 4.64 / 594 | **6.94 / 888** |
| L128_b32 | 4.41 / 564 | 4.56 / 583 | **7.22 / 924** |
| L128_b128 | 4.43 / 567 | 4.61 / 590 | **7.06 / 904** |
| L512_b1 | 0.64 / 327 | 0.64 / 326 | **1.49 / 761** |
| L512_b8 | 0.68 / 348 | 0.64 / 325 | **1.68 / 859** |
| L512_b32 | 0.66 / 335 | 0.63 / 323 | **1.67 / 856** |
| L512_b128 | skip | skip | skip |
| mixed_b32 | 3.13 / 344 | **3.92 / 430** | **8.36 / 919** |

The encoder's long-text gap to llama.cpp is the known S4.9g work (the scalar attention loop);
S4.9b is not claimed against it. The mixed batch is where the branch gains most on the encoder
(+25%): main packs a mixed batch into passes of up to 2 048 rows; the budget keeps them at 32.

## Row-budget sweep (`PEREGRINE_PASS_ROWS`)

Same method, one round per budget, cells `L16_b32`, `L128_b32` and `mixed_b32`
(texts/s). Raw JSON under `sweep/<model>/pass-rows-<N>/`.

| rows | qwen Q8 L16 | qwen Q8 L128 | qwen Q8 mixed | gemma Q8 L16 | gemma Q8 L128 | gemma Q8 mixed |
|---|---|---|---|---|---|---|
| 16 | **18.74** | 2.19 | 2.32 | 38.37 | 4.03 | 3.65 |
| 32 | 17.93 | 2.15 | 2.42 | **50.86** | 4.36 | **3.91** |
| 64 | **18.76** | 2.37 | **2.55** | 50.40 | 4.53 | 3.92 |
| 128 | 18.20 | 2.27 | 2.44 | 49.40 | **4.59** | 3.80 |
| 256 | 15.33 | 2.00 | 2.11 | 48.35 | 4.45 | 3.86 |
| 512 | 15.52 | 1.90 | 2.48 | 48.77 | 4.37 | 3.63 |

A dedicated 3-round A/B on the causal model (same cells) put 16 rows ahead: L16_b32
20.2 / 17.7 / 17.9 / 17.3 texts/s at 16 / 32 / 64 / 256, and mixed 2.77 / 2.53 / 2.46 / 2.41.
Q8_0 is compute-bound at the 16-row prefill chunk on this host (417 GFLOP/s measured), so
packing more rows only loses: hence the causal default is 16. The 300M encoder is
memory-bound and flat from 32 to 512; its floor is 32.

An F16 widening of the same Qwen3 weights (`llama-quantize F16`) behaves as the ridge rule
predicts: 179.9 / 189.8 / 174.3 / 126.3 tokens/s at 16 / 32 / 64 / 128 rows for L16_b32, i.e.
2 bytes per weight puts the ridge near 22-23 rows.

## Chat decode beside embedding work

Qwen3-Embedding 0.6B, `--slots 2`, 24-token chat requests, median of 7, beside two threads
posting 32-text embedding requests (`chat-latency-*.json`).

| | chat alone | chat beside embeddings | ratio |
|---|---|---|---|
| main | 55.98 tok/s | 7.53 tok/s | 7.4x slower |
| branch, work rows capped at one prefill chunk | 54.03 | 17.08 | 3.2x slower |
| branch, work rows capped at a quarter chunk (shipped) | 55.20 | 38.87 | **1.4x slower** |

The shipped bound is `max(1, prefill_chunk / 4)` rows of embedding work in a pass that also
carries decode rows: the chat keeps 70% of its token rate and four rows of embedding work
still move per pass.

## Decision requests (tiny openjev, 4 slots)

`/v1/systemone`, one state, 3 and 10 `noul` questions, median of 25 (`decide-*.json`).

| questions | main | branch |
|---|---|---|
| 3 | 15.5 ms | 12.4 ms |
| 10 | 36.3 ms | 39.1 ms |

The 10-question request is ~8% slower on this 40 ms request (the questions' tails now share one
pass instead of four-slot passes); 3 questions are 20% faster. Nothing material regressed.

## Memory per slot

Qwen3-Embedding 0.6B Q8_0: 112 KiB of cache per position (`KvCache::bytes_per_position`).
At the 8 192-token context the server used before, a slot mapped **896 MiB**.

| slot | before | after |
|---|---|---|
| pooled, 16-token text | 896 MiB | 1.75 MiB (16 positions) |
| pooled, 100-token text | 896 MiB | 12.3 MiB (112 positions) |
| labels, 113-token question | 896 MiB | 14.0 MiB (128 positions) |
| each forked follower | 896 MiB | its own prompt (112 positions here) |
| chat | 896 MiB | 896 MiB (unchanged) |

The engine admits at most `PEREGRINE_WORK_CACHE_MB` (512 MiB default) of these caches at once,
so 128 sixteen-token texts (224 MiB) are admitted whole on a two-chat-slot server.
