# The fast prompt mode's attention order on NEON (S5.4n)

S5.4j defined the fast prompt mode's attention order and built it for AVX-512.
S5.4k's account put the 4 096-token fast pass at 0.78–1.02× ik_llama.cpp on this
host and named attention as the gap. This round builds the order's aarch64 bodies,
benchmarks them against the exact attention this tier already had, and — because
they win — turns them on for the three NEON tiers.

- Date: 2026-11-13 08:30 – 12:30 local (UTC+7).
- Host: DGX Spark, aarch64, 20 cores in two clusters (10 × Cortex-X925 at CPUs 5–9
  and 15–19). Every engine and disturber is pinned to the ten X925s with
  `taskset -c 5-9,15-19` (one thread: CPU 5). Shared with other tenants all window;
  the 1-minute load average is on every raw record and ran 1.0–28.6 (the host was
  busiest during the yardstick cells).
- Models: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
  and — for the head-width-128 kernel bench — `Llama-2-7b-chat-hf-Q4_K_M.gguf`.
- Two builds of the same branch: `old` is `fast_attn()` forced false (the exact
  attention the tier had), `new` is the order on. The kernel bench is one binary
  with `PEREGRINE_FAST_ATTN=1` marking every row.
- `PEREGRINE_EMBED_CACHE_MB=0` everywhere; every cell is one fresh engine.

## 1. The kernel bench (B1)

`attnbench` (extended to print the tier, the flag and the head width it times), chunk
128, one repetition of the shortest wall; the fast arm marks every row a prompt row.
This is the whole attention of a pass — score, soft-max and value mix — and the
exact arm is the aarch64 attention the engine had (S4.9s's pair mix, S4.9t's
four-row tile, F16 read in registers).

| model / head width | positions / threads | exact | fast | exact ÷ fast |
|---|---|---|---|---|
| Llama 3.2 1B, 64 | 512 / 1 | 330.5 ms | **298.9 ms** | **1.106** |
| Llama 3.2 1B, 64 | 512 / 10 | 45.0 ms | **38.3 ms** | **1.175** |
| Llama 3.2 1B, 64 | 4 096 / 1 | 19 840.9 ms | **18 345.6 ms** | **1.081** |
| Llama 3.2 1B, 64 | 4 096 / 10 | 2 603.4 ms | **2 463.6 ms** | **1.057** |
| Llama 2 7B, 128 | 512 / 1 | 1 817.4 ms | **1 030.9 ms** | **1.763** |
| Llama 2 7B, 128 | 512 / 10 | 242.1 ms | **142.0 ms** | **1.705** |
| Llama 2 7B, 128 | 4 096 / 1 | 113 622.4 ms | **65 175.8 ms** | **1.743** |
| Llama 2 7B, 128 | 4 096 / 10 | 14 623.7 ms | **8 548.0 ms** | **1.711** |

The head-width-64 win is the wide-score reduction and the four-row value pass; the
head-width-128 win is bigger because the exact kernel's two eight-lane accumulators
per key and its per-row conversion of every value element cost more there. Raw:
`raw/attnbench-head64.txt`, `raw/attnbench-head128.txt`.

**The bodies are the twins, bit for bit.** `the_fast_score_body_is_the_twin`,
`the_fast_mix_body_is_the_twin` and `the_fast_denominator_body_is_the_twin` are now
`cfg(any(x86_64, aarch64))` and call the tier's body through one helper; they sweep
every key count 1…70, 127/128/129, 511/512/513, 4 095/4 096/4 097, head dims 64,
128, 80 and 16, random and soft-max edge values, at one, three and four rows, on
aarch64 natively. A fourth test, `the_fast_f16_bodies_are_the_widened_ones`, pins the
F16-direct bodies against the widened ones on the same grid.

## 2. In situ: the stop-rule cell

Llama 3.2 1B Q4_K_M, `PEREGRINE_FAST_PROMPT=1`, 4 096 tokens, interleaved rounds
(`raw/insitu-4096.jsonl` five rounds, `raw/stop-t10.jsonl` nine rounds at ten
threads):

| threads | old | new | new / old | paired median | spread |
|---|---|---|---|---|---|
| 1 (5 rounds) | 111.2 | **117.0** | **1.053** | 1.053 | ±0.2 % |
| 10 (5 rounds) | 698.0 | 674.4 | 0.966 | **1.048** | ±18 % |
| 10 (9 rounds) | 743.7 | 747.2 | 1.005 | **1.035** | ±18 % |

At one thread the order is 5.3 % on a cell whose spread is 0.2 %. The ten-thread
cell is the host's: rounds read 594–754 tok/s for the *same* arm, so the ratio of
medians moves with which round drew a co-tenant spike and the paired median (each
round its own control) is the reading that survives — +3.5 % to +4.8 %. The B5 grid
below, three rounds in a quieter window, reads +4.2 %.

## 3. End to end (B5)

`promptbench` medians of three interleaved rounds (`raw/b5-fast.jsonl`,
`raw/b5-exact.jsonl`); `new/old` is the order's gain with the mode held fixed.

| model | length | threads | fast old | fast new | **new/old** | exact old | exact new | exact new/old |
|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 512 | 1 | 208.8 | 211.4 | **1.012** | 95.6 | 97.1 | 1.016 |
| Llama 3.2 1B | 512 | 10 | 1 225.1 | 1 249.8 | **1.020** | 719.1 | 724.0 | 1.007 |
| Llama 3.2 1B | 4 096 | 1 | 111.4 | 117.5 | **1.054** | 68.1 | 68.4 | 1.005 |
| Llama 3.2 1B | 4 096 | 10 | 808.1 | 842.0 | **1.042** | 420.3 | 444.0 | 1.056 |
| Qwen 2.5 0.5B | 512 | 1 | 163.6 | 167.4 | **1.023** | 145.7 | 146.8 | 1.007 |
| Qwen 2.5 0.5B | 512 | 10 | 1 124.0 | 1 149.0 | **1.022** | 879.4 | 859.1 | 0.977 |
| Qwen 2.5 0.5B | 4 096 | 1 | 108.7 | 118.5 | **1.090** | 99.8 | 99.7 | 0.999 |
| Qwen 2.5 0.5B | 4 096 | 10 | 861.7 | 925.8 | **1.074** | 561.9 | 570.1 | 1.015 |

**The exact engine must not move by one bit, and the exact columns are the proof
that it did not:** the two binaries differ only in the tier's `fast_attn()` flag, so
every exact cell is the same arithmetic; the ratios above 1.0 there are the host's
drift (spreads ±12–31 % on the ten-thread cells in this window, ±0.3–0.7 % on the
one-thread ones). The fast ÷ exact gain of the mode itself (same window):

| model | 512/1 | 512/10 | 4 096/1 | 4 096/10 |
|---|---|---|---|---|
| Llama 3.2 1B | 2.18 | 1.70 | 1.64 | 1.92 |
| Qwen 2.5 0.5B | 1.12 | 1.28 | 1.09 | 1.53 |

### Attention's bucket

`PEREGRINE_PROF2=full`, fast mode, Llama 3.2 1B, medians of three (`raw/buckets.jsonl`):

| length / threads | attention, old | attention, new | change | share of the pass, old → new |
|---|---|---|---|---|
| 512 / 1 | 341 ms | **302 ms** | **−11.4 %** | 13.6 % → 12.2 % |
| 512 / 10 | 62 ms | **56 ms** | **−9.7 %** | 12.2 % → 11.2 % |
| 4 096 / 1 | 20 171 ms | **18 218 ms** | **−9.7 %** | 54.5 % → 51.9 % |
| 4 096 / 10 | 2 750 ms | **2 591 ms** | **−5.8 %** | 48.5 % → 46.1 % |

### Against the other engines

`llama-bench` (`raw/b5-llama.jsonl`, `raw/b5-ik.jsonl`, `raw/b5-ik-rtr.jsonl`); every
ratio is Peregrine's fast mode ÷ the engine, in the same window. ik's 4 096/1 and
llama.cpp's 4 096/1 cells are one round each (llama.cpp needs 5–6 minutes for one
4 096-token single-thread pass on this host), the rest three.

| model | cell | llama.cpp | fast ÷ llama | ik | fast ÷ ik | ik -rtr | fast ÷ ik -rtr |
|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 512 / 1 | 30.4 | 6.96 | 117.7 | 1.80 | 123.2 | 1.72 |
| Llama 3.2 1B | 512 / 10 | 212.7 | 5.88 | 624.4 | 2.00 | 434.8 | 2.88 |
| Llama 3.2 1B | 4 096 / 1 | 25.9 | 4.53 | 86.1 | 1.36 | — | — |
| Llama 3.2 1B | 4 096 / 10 | 189.2 | 4.45 | 366.1 | 2.30 | 403.0 | 2.09 |
| Qwen 2.5 0.5B | 512 / 1 | 61.8 | 2.71 | 305.7 | 0.55 | 261.5 | 0.64 |
| Qwen 2.5 0.5B | 512 / 10 | 442.0 | 2.60 | 669.4 | 1.72 | 446.9 | 2.57 |
| Qwen 2.5 0.5B | 4 096 / 1 | 50.7 | 2.34 | 187.9 | 0.63 | — | — |
| Qwen 2.5 0.5B | 4 096 / 10 | 354.0 | 2.62 | 592.3 | 1.56 | 712.6 | 1.30 |

The Qwen single-thread cells stay below ik (0.55–0.64×): ik's `iqk_flash_attn` on one
core is still far ahead of a widened f32 attention, and the fast pass there is 65 %
attention (§5k). Every ten-thread cell is ahead of both engines except ik's bare
4 096/10 on the 1B, where the fast pass is 2.30× — but read that against S5.4k's
0.78× for the same cell only with the window in mind: ik's own rounds read 366–624
tok/s here against 592–1 022 in S5.4k's loaded window, so most of the difference is
the host, not the order. The order's own share of that cell is the +4.2 % of §3.

## 4. Accuracy (B5)

`kld --ref-self --fast-prompt` on wikitext-2's test split, 40 chunks of 512 tokens,
10 200 scored positions — the corpus and chunking S5.4h/S5.4c/S5.4k used — the fast
mode against the exact mode of the same file, the old order and the new one
(`raw/kld-wikitext-*.txt`):

| model | mean KLD old | mean KLD new | same top p old → new | RMS Δp old → new |
|---|---|---|---|---|
| Llama 3.2 1B | 0.003487 ± 0.000047 | **0.003496 ± 0.000045** | 96.451 ± 0.183 % → 96.422 ± 0.184 % | 1.357 → 1.413 % |
| Qwen 2.5 0.5B | 0.003850 ± 0.000050 | **0.003854 ± 0.000053** | 96.196 ± 0.189 % → 96.324 ± 0.186 % | 1.492 → 1.444 % |

+0.26 % and +0.10 % of itself — a fifth of the run's own standard error. A summation
order cannot move the KL, and it did not.

## 5. Tests and goldens

- `cargo test --release -p peregrine-core` (292 tests) natively; `fast_prompt.rs`
  whole (10 tests) natively and under `PEREGRINE_ISA=neon`, `neon-dotprod`,
  `scalar`, and once with `PEREGRINE_ADAPTIVE_STALL=claim`.
- `a_fast_rows_attention_is_its_own_at_every_group_size`,
  `a_marked_prompt_row_takes_the_fast_attention_order` (its aarch64 branch now takes
  the fast order) and `a_fast_row_alone_in_its_cache_is_the_solo_row` pass natively.
- The 4 096-token invariance test is two layers deep now, so an attention row's own
  output reaches the next layer's K/V; it caught a real bug in the F16-direct walk
  (the level slice, fixed in the second commit) and fails if the bug is put back.
- The Q5_0 padded-form test is 896 wide and two layers, and it caught the same bug
  through a different path.
- **Goldens**: eight files moved, all aarch64 NEON-tier `*-fast`:
  `aarch64-neon/llama-{Q4_K,Q6_K}-fast`, `aarch64-neon-dotprod/llama-{Q4_K,Q5_0,Q6_K}-fast`,
  `aarch64-neon-i8mm/llama-{Q4_K,Q5_0,Q6_K}-fast`, re-recorded natively per
  `PEREGRINE_ISA` cap. The macOS twins
  (`aarch64-neon-dotprod-macos/llama-{Q4_K,Q5_0,Q6_K}-fast`) were recorded by
  `record-goldens.yml` on `macos-15` (run 43356516220) and are the Linux values bit
  for bit. The scalar-forced `aarch64/` directory did not move, and neither did any
  exact golden or any x86 file.

## Files

- `bench_ab.py` — two builds (or two modes) per cell, arms rotating per round.
- `profile_ab.py` — the same for the `StepStats` buckets.
- `summarize.py` — medians, spreads and every ratio (`table.txt`).
- `raw/` — the JSONL grids, the kernel bench output, the kld reports.
