# The fast prompt pass against ik_llama.cpp's `-rtr` path, and a form with one scale per 32 (S5.4i)

Where the opt-in 8-bit prompt pass (`--fast-prompt`) spends its time and why it was 0.74-0.94×
ik_llama.cpp's `-rtr` path on every cell (S5.4c); what three same-bits candidates bought when
built and measured; the per-32-weight form's kernel, its bytes, its build time and its per-weight
error, as the input to the owner's decision between the two arithmetics; and one quiet re-measure
of the exact engine against llama.cpp v0.9.0 and ik_llama.cpp.

- Date: 2026-11-12 (22:00) – 2026-11-13 (01:00), UTC+7.
- Host: AMD Ryzen AI 9 HX 370 (Zen 5), 12 physical cores / 24 logical, AVX-512 (VNNI, GFNI, VBMI,
  BF16), WSL2, 11.7 GiB, kernel `6.18.33.2-microsoft-standard-WSL2`, **idle: the only load was
  this session's own runs**. The clock still runs ~10 % fast (MONOTONIC), the same for every
  engine, so every absolute tokens/s is ~10 % conservative and the ratios are what to read.
  This session's sweep gives Llama 3.2 1B 512 tokens at 12 threads: Peregrine's exact engine
  349.1 tok/s against S5.4c's 356.3 (-2 %), so the two windows are comparable.
- Peregrine: branch `s5.4i-fast-pass-vs-rtr` = `origin/main` (`25ff094`, S5.4f) merged with
  `origin/s5.4c-fast-prompt-accuracy` (`68dd04f`, which had not merged when this step started, so
  its report and the `kld` switch are in this branch). AVX-512 tier, Rust 1.99.1.
- ik_llama.cpp: `~/.cache/peregrine-bench/ik_llama.cpp-479074c` (`479074c`), its `llama-bench`,
  and `benchmarks/ik_kernel_probe.cpp` built against its own `libggml.so`.
- llama.cpp: `~/.cache/peregrine-bench/llama.cpp-v0.9.0` (v0.9.0, `cd959ba`).
- Models: Llama 3.2 1B Instruct Q4_K_M `~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (807 694 464 B),
  Qwen3 1.7B Q4_K_M `/var/tmp/pg/s54c/models/Qwen3-1.7B-Q4_K_M.gguf` (1 282 439 264 B), Qwen2.5
  0.5B Instruct Q4_K_M `~/models-ab/qwen2.5-0.5b-instruct-q4_k_m.gguf` (491 400 032 B) — all regular
  files (`test -L` clean), the same uploads S5.4c hashed.
- Corpus for the account: `~/.cache/peregrine-bench/s52b/corpus.txt` (927 064 B of plain prose, the
  S5.2 corpus), read by `promptbench`.

## 0. What this round found, in one paragraph

**The fast pass's kernel is not where it loses.** In situ, at one thread, the 8-bit form's Q4_K
kernel runs at **96.2 GMAC/s** and its Q6_K at 103.9 (Llama 3.2 1B, 512 tokens, `PEREGRINE_PROF2`),
against its own isolated rate of 91.9-95.5 (S5.4a) and against **the fork's own `q8_k_r16` kernel
at 94.9 GMAC/s** measured through the fork's library at 512 activation rows (`85.1` at 256). So
the form's arithmetic is level with the fork's at both engines' own chunk widths. The loss at 12
threads is what is *not* the kernel: at 512 tokens **36 % of the fast pass is not the kernel**
(barrier wait 10.4 %, the caller's serial steps 13.5 %, attention 9.1 %, the activation's fill and
the output's untile ~3 %), and at 4096 tokens **attention is 38.0 % of the fast pass** against
20.5 % of the fork's, because the fork's flash-attention kernels are 2.8× ours in absolute time.
Three same-bits candidates were built and measured — a sixteen-output-row body (the fork's `r16`
shape, +4-6 % at one thread, -2 to -4 % at twelve), a 128-row prompt chunk (+3-5 % at 512 tokens,
-2.4 % at 4096) and the head's one-row kernel — and **none is shipped**, for the reason each
paragraph gives. The per-32 form is **0.58-0.65× the per-256 form's kernel** (it would make a fast
pass slower than the *exact* engine) for a per-weight RMS error only **1.39×** smaller.

## 1. Method

### The account

`promptbench` drives the real `Engine` with real token ids from the corpus file and prints, with
`PEREGRINE_PROF2=full`, the caller's `StepStats` buckets (matmul, head, attention, other,
barriers) and every `peregrine_core::prof` slot: the weight matmul by matrix group (qkv, wo,
gate/up, down, head) and by weight format (Q4_K, Q6_K), attention's three stages, and each of the
caller's serial steps. `scripts/account-cell.sh` runs one cell and records the 1-minute load and
the CPU-seconds spent on this VM **outside the pinning set** during the run (`other_busy_s`,
a better quietness metric here: `scripts/account-cell.sh` reads `/proc/stat` before and after).

`account.py` turns those into the tables below. The MAC counts come from **the model's own tensor
table** (`gguf_macs.py`, a GGUF header reader), so the in-situ GMAC/s is the engine's own rate and
not a microbenchmark's: one activation row through a group is `Σ dims[0]·dims[1]` over its
matrices, and the group's share of the pass is the `prof` slot's time over the pass wall.

**One correction to S5.2b's arithmetic.** The `prof` format subtimers are *worker-side* sums, so
they are core-seconds: at 12 threads the Q4_K slot reads 442.9 % of the pass wall on Llama 1B at
4096 tokens, i.e. twelve cores' worth. The per-core rate is the MACs over that sum, and the share
of the wall is the sum over the pool width. Both are printed where they appear.

### The fork's side

`llama-bench` exposes no per-op timings, so the fork's side is two measurements of its own binary
and one of its own library:

* `benchmarks/ik_kernel_probe.cpp`, built against the fork's `libggml.so` (the probe S5.2 used),
  calls the same `mul_mat` node the engine calls, for one real layer's shape (K = 2048, 8192 output
  rows), and prints **which of the fork's code paths that shape takes** (`iqk_dequant_type`) and
  its rate. This is the fork's per-op timing, by shape;
* `llama-bench` with `-rtr` / without and with `-fa 0` / without isolates the fork's two wide
  kernels and its flash attention in its own pass.

**`perf` is not available on this host**: no `perf` binary, no `sudo` to install one
(`sudo: a password is required`), `perf_event_paranoid` = 2, and WSL2's kernel exposes no arm
PMU counters to a plain user. So there is no sampling profile of either engine; the account is the
engines' own timers, as S5.2b/S5.2c did, and the fork's library called directly.

### Speed cells

`scripts/speed-sweep.sh` runs every arm once per cell, in the same order, in every round, so a
drift in the host hits every engine: `pg-exact`, `pg-fast-base` (this branch's fast mode),
`pg-fast-pair` (the sixteen-row candidate, §3.1), `ik` (defaults), `ik-rtr`, `llama`. 12-thread
cells are pinned to `taskset -c 0-23` and one-thread cells to cpu0, as S5.4c did (`0-11` is six
physical cores on this VM). Medians; the rounds are in `raw/speed/sweep-*.jsonl`.

**The 1-minute load average is not a usable quietness metric on this VM.** It rises with the
sweep's own 12-thread arms and decays slowly: during the 4096-token cells it reads 8.5-11.8 while
`/proc/stat` says the CPUs outside the pinning set are ~0.1 CPU busy. It is recorded beside every
run as before, and every table below also gives the loads, but the readings that matter are the
CPU-seconds outside the pinning set (`other_busy_s` in `raw/account/*.jsonl`: 0.5-1.6 CPU-seconds
per 10-second one-thread cell, i.e. ~0.1 CPU) and the spreads.

## 2. A: the account of a fast pass

Four tables, medians of two rounds (`raw/account/account-512-4096t12.txt`,
`raw/account/account-4096t1.txt`). "Wall" is the engine's own prefill seconds for the whole pass;
the shares are of that wall. Every GMAC/s is per core (one thread: the caller; twelve: the MACs
over the worker-side sum, i.e. per core, and the share is the sum over 12 workers where it says so).

### 2.1 Llama 3.2 1B Q4_K_M, 512 tokens

| stage | 1 thread, exact | 1 thread, fast | 12 threads, exact | 12 threads, fast |
|---|---|---|---|---|
| wall (s) | 9.218 | **6.283** | 1.573 | **1.103** |
| prompt tok/s | 55.5 | 81.3 | 326.8 | 463.3 |
| end to end GMAC/s | 55.0 | 80.7 | 322.4 | 459.8 |
| matmul (share) | 88.3 % | 83.0 % | 80.8 % | 79.3 % |
| attention (share) | 6.2 % | 9.1 % | 5.9 % | 9.1 % |
| other (share) | 6.0 % | 8.8 % | 14.1 % | 13.5 % |
| barrier wait | 0 % | 0 % | 9.1 % | 10.4 % |
| **matmul by group, GMAC/s in situ** | | | | |
| qkv | 60.4 | 94.5 | 404 (aggregate) | 590 |
| wo | 74.0 | 97.0 | 348 | 447 |
| gate/up | 74.1 | 94.6 | 501 | 682 |
| down | 44.5 | 100.1 | 279 | 474 |
| head (1 row) | 5.9 | 5.8 | 2.0 | 31.1 |
| **matmul by format, GMAC/s per core** | | | | |
| Q4_K | 73.1 | **96.2** | 42.6 | 58.9 |
| Q6_K | 32.2 | **103.9** | 19.9 | 54.5 |
| attention: scores / mix, GMAC/s/core | 23.9 / 13.3 | 23.2 / 13.3 | 15.0 / 9.3 | 12 / 8 |
| caller: quantise / swiglu / resid / rope / kv / norms | 1.9 / 3.2 / 0.1 / 0.2 / 0.3 / 0.2 % | 3.0 / 4.6 / 0.1 / 0.3 / 0.3 / 0.4 % | 1.5 / 3.0 / 0.6 / 0.4 / 2.2 / 1.0 % | 2.8 / 4.2 / 0.6 / 0.5 / 1.9 / 1.2 % |

(The `head` row at 12 threads is the second round's page-cache-cold read of the 215 MiB output
tensor: 130 ms, 8.3 % of that pass. Its warm value is 7.9 ms, measured in the S5.4c window.)

### 2.2 Llama 3.2 1B Q4_K_M, 4096 tokens

| stage | 1 thread, exact | 1 thread, fast | 12 threads, exact | 12 threads, fast |
|---|---|---|---|---|
| wall (s) | 98.705 | **76.571** | 15.615 | **12.087** |
| prompt tok/s | 41.5 | 53.5 | 262.2 | 338.8 |
| end to end GMAC/s | 46.0 | 59.2 | 290.5 | 375.3 |
| matmul (share) | 63.8 % | 53.5 % | 61.0 % | 52.9 % |
| **attention (share)** | 32.0 % | **41.0 %** | 30.1 % | **38.0 %** |
| other (share) | 4.3 % | 5.6 % | 9.0 % | 9.4 % |
| barrier wait | 0 % | 0 % | 11.5 % | 10.6 % |
| Q4_K / Q6_K, GMAC/s/core | 71 / 30 | **89 / 97** | 44.3 / 22.7 | 63.5 / 57.4 |
| scores / mix, GMAC/s/core | 19.5 / 9.4 | 20.3 / 10.2 | 14.9 / 10.0 | 13 / 9 |

### 2.3 Qwen3 1.7B Q4_K_M, 512 and 4096 tokens, 12 threads

| stage | 512 exact | 512 fast | 4096 exact | 4096 fast |
|---|---|---|---|---|
| wall (s) | 2.185 | **1.476** | 23.9 | **18.5** |
| prompt tok/s | 233.8 | 309.3 | 172 | 220.9 |
| matmul / attention / other | 82.0 / 6.7 / 12.3 % | 79.0 / 13.0 / 14.6 % | 62.5 / 32.5 / 5.1 % | **49.8 / 38.6** / 11.9 % |
| barrier wait | 8.9 % | 11.9 % | 11.1 % | 11.1 % |
| Q4_K / Q6_K, GMAC/s/core | 43.4 / 56.1 | 47.5 / 42.8 | — / — | 63.2 / 43.0 |

Qwen3's Q6_K is 424 MB of the file and includes the 151 936-row output head (255 MB): the head is
one row of a prompt pass, so a format table that multiplies the head by the prompt's rows
over-counts. `gguf_macs.py` excludes it and the group table carries it on its own line.

### 2.4 The fork's side, through its own library

`ik_kernel_probe`, one real layer (2048 × 8192), the fork's own quantiser, repack and `mul_mat`
node, one thread, median of 5:

| activation rows | Q4_K_R4 path | GMAC/s | Q6_K_R4 path | GMAC/s |
|---|---|---|---|---|
| 8 | `q4_k_r4` (its 4-bit kernel) | 65.5 | `q6_k_r4` | 57.4 |
| 64 | `q8_k_r16` | 52.4 | `q8_k_r16` | 56.7 |
| 128 | `q8_k_r16` | 75.5 | `q8_k_r16` | 74.2 |
| **256** | `q8_k_r16` | **85.1** | `q8_k_r16` | **88.9** |
| **512** (its `-ub` default) | `q8_k_r16` | **94.9** | `q8_k_r16` | **93.7** |

Two things follow. The fork's 8-bit kernel is **slower than its own 4-bit kernel at 64 rows**
(52.4 against 65.5) and only 1.45× faster at 512 (94.9 against 65.5), which is why it switches
at 32 rows and why its own wide path is worth what it is. And at Peregrine's chunk (256 rows) the
fork's kernel is 85.1-88.9 against Peregrine's **96.2/103.9 in situ** — Peregrine's form is *ahead*
of the fork's at the width it runs, and level at the fork's 512.

### 2.5 The fork's pass, by ablation, in its own binary

`llama-bench -p 512|4096 -n 0 -t 12 -r 3`, Llama 3.2 1B, medians:

| arm | 512 tok/s | 4096 tok/s |
|---|---|---|
| `-rtr` (its fast wide path, flash attention on) | **587.2** | **399.9** |
| `-rtr -fa 0` | 534.2 | 317.9 |
| defaults (no `-rtr`, flash attention on) | 455.0 | 334.2 |
| defaults, `-fa 0` | 415.3 | 282.0 |

So the fork's flash attention is worth **9.0 % of its 512-token pass and 20.5 % of its 4096-token
one**, and its requantising kernels **22.5 % and 16.4 %**. Peregrine's attention is 6.2 % of an
exact 512-token pass and 30.1-38.0 % of a 4096-token one — at 512 the shares agree, at 4096 ours is
1.85× the fork's share of a pass that is itself 1.31× longer, i.e. **our attention is ~2.4× the
fork's in absolute seconds** (5.89 s against ~2.1 s on the 1B at 4096 tokens, 12 threads).

## 3. A: what was built, and what each candidate bought

Three same-bits candidates were built and measured with a five-round in-round alternating A/B of
two builds at 1 and 12 threads. **None is shipped**; each is in the crate if it is a kernel, and
the reason is below the table.

| candidate | 1B 512/12 | 1B 512/1 | Q3 512/12 | Q3 512/1 | 1B 4096/12 | Q3 4096/12 | verdict |
|---|---|---|---|---|---|---|---|
| sixteen-row body (fork's `r16` shape) | 0.964 | **1.023** | 0.983 | **1.061** | **1.021** | **1.015** | not shipped |
| prompt chunk 256 → 128 | **1.045** | **1.021** | **1.032** | — | 0.976 | — | not shipped |
| head's one-row kernel | 1.00 (0.8 % of the pass) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | not worth it |

### 3.1 The sixteen-output-row body (the fork's `r16`)

`kqsimd.rs` gains `q8k_gemm16_zmm` and `kq8.rs` its twin `gemm_q8k_pair`: two adjacent eight-row
tiles of the form multiplied at once, one activation broadcast (`vpbroadcastq`) and one activation
cache line feeding two `vpdpbusd` instead of one — the trade the fork's `q8_k_r16` makes with its
sixteen weight rows. Four activation rows is the widest group that keeps both tiles' int32 and
float accumulators in the ZMM file. `the_sixteen_row_kernel_is_bit_identical_to_two_eight_row_ones`
holds it to the twin and to two calls of the eight-row body at every width from 1 to 129.

**Isolated** (`KQBENCH_PAIR=1`, one layer streamed, one core): 103.5 against 89.5 GMAC/s at 8
activation rows (**-14 %**),
100.8 against 93.5 at 16 (**+8 %**), 106.5 against 103.4 at 64 (**+3 %**), 110.9 against 102.4 at
128 (**+8 %**), 110.3 against 107.9 at 256 (**+2 %**), 113.2 against 110.4 at 512 (**+3 %**).

**End to end**, five rounds, two builds alternating: +2.3 % and +6.1 % at one thread, and at twelve
threads -3.6 % (1B 512), -1.7 % (Qwen3 512), +2.1 % (1B 4096), +1.5 % (Qwen3 4096). Geometric mean
**+1.1 %**, but the regression lands on the cell furthest below the bar (1B 512 tokens at 12
threads), and the account says why: at 12 threads the pass is not kernel-bound (§4), so a kernel
that is 3-8 % faster at the widths a 12-thread pass runs is worth nothing there, while its larger
working set costs 2-4 %. So the body stays in the crate with its test, and the dispatch does not
use it — the same disposition S5.1b/S5.2 gave the widened exact Q4_K kernel.

### 3.2 The prompt chunk, 128 against 256

Five rounds, one build, alternating: 1B 512/12 **1.045**, 1B 512/1 **1.021**, Q3 512/12 **1.032**,
1B 4096/12 **0.976**. 128 rows wins at 512 tokens and loses at 4096, the trade S5.4b measured
(2.2 % the other way) — a default cannot be both, and the engine does not know the prompt's total
length when it prefills its first chunk. **Left at 256**, with the numbers recorded so the next
round can pick a rule.

### 3.3 The output head's one-row kernel

The head is multiplied for **one row** of a prompt pass (the last), through `quant::dot_batch`,
at **5.8 GMAC/s**: 46 ms at one thread (0.7 % of a 512-token pass) and 8 ms at twelve. It is the
whole of decode's cost per token on the 1B at one thread (254 ms a token in a cold-cache run,
the 215 MiB output tensor read at 0.85 GB/s), but in a prompt pass it is 0.7 % at 512 tokens and
0.1 % at 4096: **not worth a kernel this round**. It is named here because S5.4a's account had it
at 2.6 % in one cold-cache run and a reader should not chase it.

## 4. A: where the fast pass loses, named

At **512 tokens, 12 threads** (Llama 1B, fast: 1.103 s; the terms sum to 1.067 s and the difference
is the loop's own rounding):

| term | seconds | share | note |
|---|---|---|---|
| the 8-bit kernel | 0.703 | 64 % | 497 GMAC at 707 GMAC/s aggregate (58.9 per core) |
| barrier wait | 0.115 | **10.4 %** | 481 forks, ~240 µs of caller wait each |
| attention | 0.100 | 9.1 % | the same code as the exact engine |
| the caller's serial steps | 0.149 | **13.5 %** | quantise, SwiGLU, norms, RoPE, residual, KV write |
| activation fill + untile + loop | ~0.036 | 3 % | `KqActs::fill_with` (1.1 % measured) and the write-out |

The fork's pass is 0.864 s. Its own ablation puts 9.0 % of it in flash attention (0.078 s), close
to our 9.1 %, so at this length attention is proportional and not the difference; our kernel's rate
is the fork's at its own width (§2.4: 96.2 in situ against 85.1-94.9). What is left is our
non-kernel share: **the 22 % gap at this cell is the pass's non-kernel share, not the 8-bit
multiply**, and 10.4 of those points are the barrier wait, which is a property of the pool and of
a pass this short, not of the arithmetic (S5.2b/S5.2c measured it: a worker that finds no job
parks, and the next job waits for the wake).

At **4096 tokens, 12 threads** (fast: 12.087 s): kernel 5.31 s (44 %), **attention 4.59 s (38 %)**,
barrier wait 1.28 s (10.6 %), the caller's serial steps 1.14 s (9.4 %), fill and untile 0.17 s.
The fork's whole pass is 10.56 s and **its own ablation puts 20.5 % of it in flash attention**
(2.16 s), against our 4.59 s — so our attention is **2.1× the fork's in absolute time and is the
only stage where we are further behind than the pass as a whole**: it accounts for 2.4 s of a
1.53 s gap, which means the rest of our pass is *ahead* of the fork's. That is the whole finding
of this cell: at 4096 tokens the fast pass's loss is a flash-attention kernel, and the 8-bit
multiply is 44 % of the pass at 63.5 GMAC/s per core.

Why the kernel cannot simply be made faster: its isolated rate is 91.9-95.5 GMAC/s at prompt widths
and its in-situ rate is 96.2 (1 thread), against a measured byte-dot roof for this core of
121 GMAC/s (S5.1b: `vpdpbusd` 512-bit at one per two cycles). It is at 79 % of that roof, and so is
the fork's (94.9/121 = 78 %). The remaining 20 % is the epilogue and the activation broadcasts,
which the sixteen-row body addresses (§3.1) and which are worth 3-8 % — not 22 %.

## 5. A: the eight cells, before and after

Prompt tokens per second, medians, this window. **No engine change ships in this step**, so
"before" and "after" are the same build: the column is labelled *this round* and the S5.4c column
is given so the two windows can be compared. Every cell below 1.0 is named.

| Model | prompt / threads | exact | fast | ik (defaults) | ik (`-rtr`) | llama.cpp | fast ÷ exact | fast ÷ ik | fast ÷ **ik `-rtr`** | fast ÷ llama | fast ÷ `-rtr`, S5.4c |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 512 / 12 | 349.1 | 486.5 | 440.2 | 592.3 | 290.7 | 1.394 | 1.105 | **0.821** | 1.673 | 0.79 |
| Llama 3.2 1B | 512 / 1 | 55.4 | 80.7 | 67.4 | 85.3 | 42.0 | 1.457 | 1.197 | **0.946** | 1.921 | 0.91 |
| Llama 3.2 1B | 4096 / 12 | 264.6 | 339.3 | 326.4 | 388.0 | 195.0 | 1.282 | 1.040 | **0.874** | 1.740 | 0.81 |
| Llama 3.2 1B | 4096 / 1 | 41.2 | 53.3 | — | 57.6 | 28.2 | 1.294 | — | **0.925** | 1.890 | 0.79 |
| Qwen3 1.7B | 512 / 12 | 233.4 | 309.3 | 296.1 | 387.8 | 193.2 | 1.325 | 1.045 | **0.798** | 1.601 | 0.74 |
| Qwen3 1.7B | 512 / 1 | 37.6 | 52.6 | 43.5 | 57.9 | 27.6 | 1.399 | 1.209 | **0.909** | 1.906 | 0.94 |
| Qwen3 1.7B | 4096 / 12 | 173.8 | 214.1 | 215.5 | 249.5 | 126.0 | 1.232 | **0.994** | **0.858** | 1.699 | 0.79 |
| Qwen3 1.7B | 4096 / 1 | 27.6 | 35.4 | — | 38.0 | — | 1.283 | — | **0.932** | — | 0.85 |

Every ratio is recomputed from the medians in the row. Rounds: 5 for both 512-token cells (two in
the first sweep window, three in the second — `sweep-a` and `sweep-b`/`sweep-c` — because the
sweeps ran in three windows), 3 for the 4096-token 12-thread cells, **1 for the two 4096-token
one-thread cells** (one round each, 7-8 minutes a round; `raw/speed/sweep-d.jsonl`). The Qwen3
4096/1 llama.cpp cell and both 4096/1 `ik` (defaults) cells are missing for that reason and are
marked as empty, not as zero.

**Cells below 1.0, all named:** against ik's `-rtr` path, **all eight** (0.798-0.946); against ik
at its defaults, **one** (Qwen3 4096 tokens at 12 threads, 0.994, i.e. level). Nothing else is
below 1.0: the fast mode is 1.23-1.46× the exact default and 1.60-1.92× llama.cpp on every cell.

The before/after read: against `-rtr` the eight cells move from S5.4c's 0.74-0.94 to 0.80-0.95 —
which is the window, not a change (this window's exact engine is 2 % faster than S5.4c's and the
fork's 12-thread cells are 2-7 % faster). **Nothing about the pass's arithmetic changed, so nothing
about the eight cells did either**, and the report says so rather than crediting a candidate that
was not shipped.

## 6. B: the form with one scale per 32 weights

`crates/peregrine-kernels/src/p32.rs`: the finer form's builder, its scalar twin and an AVX-512
body, bit-identical to each other
(`the_per_32_simd_body_is_the_twin`, widths 1-13, and `the_builder_is_the_fork_s_arithmetic_one_block_down`),
**called by nothing in the engine** — not in the kernel table, not in `q8form`, not on the
dispatch, no golden. One unit of the form is 32 weights × 8 rows: 256 bytes of codes (biased by
128, as the per-256 form stores them) + 32 of f32 scales + 32 of padding = 320 bytes, **1.25 bytes
per weight** against the per-256 form's 1.03125. The activation needs its own layout (`P32Acts`,
one int8 per weight plus a per-32 scale and sum), because the per-256 form reads llama.cpp's Q8_K
(one scale per 256) and this one reads the per-32 quantisation the exact kernels already have.

### 6.1 Speed, bytes and build

`KQBENCH_PER32=1`, one real layer (2048 × 8192, 1024 tiles, 17 MB streamed), GMAC/s per core at 1
core / 12 cores, medians of five (`raw/partb/per32-kernel.txt`):

| activation rows | exact | per 256 (shipped) | per 32 | per-32 ÷ per-256 | per-32 ÷ exact |
|---|---|---|---|---|---|
| 8 | 80.5 / 921.8 | **102.2 / 1249.7** | 44.3 / 561.5 | **0.433 / 0.449** | 0.550 / 0.609 |
| 64 | 81.1 / 967.4 | 104.3 / 1252.5 | 66.0 / 763.3 | **0.633 / 0.609** | 0.814 / 0.789 |
| 128 | 74.3 / 972.0 | 104.4 / 1259.2 | 64.8 / 808.7 | **0.621 / 0.642** | 0.872 / 0.832 |
| 256 | 79.2 / 900.9 | 105.7 / 1296.6 | 65.2 / 816.5 | **0.617 / 0.630** | 0.823 / 0.906 |
| 512 | 77.9 / 848.2 | 107.5 / 1279.8 | 65.1 / 706.7 | **0.606 / 0.552** | 0.837 / 0.833 |

| | per 256 | per 32 |
|---|---|---|
| bytes per weight | 1.03125 | **1.25** |
| the 1B's form (its 930 M Q4_K/Q6_K weights) | 957 MiB (measured, S5.4b) | **1 164 MiB** |
| build, scalar builder, dequantise + requantise | 0.08 G weights/s (the crate's twin) | **0.65 G weights/s** |
| build, the engine's builder | 1.9 G weights/s (the AVX-512 body: 957 MiB in 0.52 s) | no SIMD builder yet |

The reason is the epilogue, not the dot: the byte-dot instruction count is identical (one
`vpdpbusd` per 8 weights per activation row, 128 per 512 MACs either way), but the bias payback,
the `d · dk` multiply and the fused multiply-add run once per 32 weights instead of once per 256 —
eight times as often, on the same ports. That is exactly why the fork's default wide path is the
slower of its two (`q8_k_r16` at 94.9 GMAC/s against `q4_k_r4`'s 65.5 at 8 rows shows the 4-bit
kernel is not the comparison; the fork's own 64-row switch to 8 bits at 52.4 GMAC/s is).

### 6.2 The per-weight error on the real files

`kq8model p32 <model.gguf>`: every 16th super-block of every Q4_K/Q6_K tensor, all eight rows,
each error relative to **the super-block's** largest weight (the denominator S5.4a's tables used),
so the two forms are directly comparable (`raw/partb/per32-errors-*.txt`):

| model | form | format | RMS | max | mean / block max | max / block max |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | per 256 | Q4_K | 1.458e-4 | 8.953e-4 | 0.00198 | 0.00394 |
| | per 32 | Q4_K | **1.048e-4** | 8.689e-4 | **0.00142** | 0.00394 |
| | per 256 | Q6_K | 1.212e-4 | 5.509e-4 | 0.00194 | 0.00394 |
| | per 32 | Q6_K | **8.620e-5** | 5.353e-4 | **0.00138** | 0.00394 |
| Qwen3 1.7B | per 256 | Q4_K | 2.712e-4 | 1.792e-3 | 0.00197 | 0.00394 |
| | per 32 | Q4_K | **1.991e-4** | 1.759e-3 | **0.00143** | 0.00394 |
| | per 256 | Q6_K | 2.624e-4 | 9.766e-4 | 0.00193 | 0.00394 |
| | per 32 | Q6_K | **1.901e-4** | 9.293e-4 | **0.00138** | 0.00394 |

**The finer grid is 1.39× better in RMS, not 8×.** Each block's step is its own max/127, and a
32-weight block's max is close to its super-block's (0.7-0.8× on these files), so the RMS improves
by the square root of that ratio. The mean |error| / block max falls 0.00198 → 0.00142 on the 1B
and 0.00197 → 0.00143 on Qwen3. The worst single weight is the same in both (0.00394, the
`max/254` bound) because any block's own bound is reached somewhere.

**This sits oddly against the fork's own numbers, and the owner should know it.** S5.4c measured
the fork's default (per-32) wide path at 0.000945 nats of mean KL on the 1B and its `-rtr`
(per-256) path at 0.003401 — a factor of 3.6 — while the per-32 form's per-weight RMS is only 1.39×
better on the same file. Either the two fork paths differ in more than the grid (they do: also the
rounding and the integer bias), or the per-weight RMS is a poor predictor of the KL, or the fork's
two measurements are themselves a pair of different experiments (they are: the 0.000945 compares
its narrow pass with its wide default, the 0.003401 compares narrow-with-`-rtr` with wide-with-`-rtr`,
and `-rtr` moves the narrow pass's own perplexity too). A per-32 *Peregrine* mode's KL was not
measured — that needs a `kld` run with the form wired in, which this step does not do — so the
accuracy side of the decision has one number from this code (the per-weight error) and one from the
fork (0.000945).

### 6.3 What it would cost end to end (an estimate, not a measurement)

By part A's account: at 512 tokens and one thread the fast pass's weight matmul is 5.11 s of 6.28 s
(81 %), and the kernel is 96.2 GMAC/s for Q4_K. A per-32 kernel at **0.58×** that rate (what the
prototype measures at 64-512 activation rows) makes the matmul 8.8 s and the pass ~10.7 s, i.e.
**0.59× the current fast mode and 0.86× the exact engine** — a per-32 fast prompt mode would be
*slower than turning the option off*. At 12 threads the picture is the same (the kernel is 64 % of
that pass). The estimate is the account's shares times the measured kernel ratio; it is not a
measured end-to-end mode, and it does not include the extra 207 MiB or the load.

## 7. C: the exact engine, quiet

The cell S5.2c flagged (Qwen 2.5 0.5B Q4_K_M, 512 tokens, 12 threads, 0.89× llama.cpp, measured at
load 16-22) and its three neighbours, medians of three interleaved rounds in this window, exact
engine against llama.cpp v0.9.0 and ik_llama.cpp at its defaults (`raw/speed/sweep-b.jsonl`):

| model | prompt / threads | Peregrine | llama.cpp | ik (defaults) | P ÷ llama | P ÷ ik |
|---|---|---|---|---|---|---|
| Qwen 2.5 0.5B | 512 / 12 | **641.1** | 477.5 | 1051.0 | **1.343** | **0.610** |
| Qwen 2.5 0.5B | 4096 / 12 | **450.1** | 305.1 | 691.8 | **1.475** | **0.651** |
| Llama 3.2 1B | 512 / 12 | 349.1 | 290.7 | 440.2 | 1.201 | 0.793 |
| Llama 3.2 1B | 4096 / 12 | 264.6 | 195.0 | 326.4 | 1.357 | 0.811 |

Per round, so the spread is visible: the Qwen 0.5B 512 cell reads 646.0/634.3/641.1 against
477.5/475.7/477.5 (ratio 1.353/1.334/1.343), and its 4096 cell 450.3/450.1/446.3 against
305.1/303.5/308.0 (1.476/1.483/1.449). **Every one of the four cells is above llama.cpp and below
ik at its defaults**, and the Qwen 512 cell that S5.2c reported at 0.89× is **1.34×** when the host
is quiet: that 0.89 was the loaded window, as S5.2c itself said ("does not separate the two engines
and is to be measured again on a quiet host").

**Why the Qwen file trails ik more than the 1B does** (0.61-0.65 against 0.79-0.81): it is 44 %
Q5_0 by bytes, and Q5_0's block is 32 weights, so its exact kernel runs its float fold eight times
as often per weight — **47.9-52.0 GMAC/s isolated against Q4_K's 74.0** (S5.2, §11.5) — and the
Qwen 0.5B file is mostly Q5_0 where the 1B is Q4_K. That is a kernel work item (a 16-output-row
fold, which §11.5 measured *slower* at every width for the same format, or a different store),
not a small same-bits change, so nothing was attempted. The 512-token cell is 1.34× llama.cpp
either way, so it is not a regression against the engine the S5.2c cell was compared with.

## 8. What surprised this round

1. **The form's kernel is faster in situ than the fork's own, at the width it runs.** 96.2 against
   the fork's 85.1 (256 rows) and 94.9 (512 rows), measured through the fork's `libggml.so`. S5.4c
   concluded "the kernel is level and the pass is not" from microbenchmarks; the in-situ number is
   the same verdict from the other side, and it is worth stating plainly because it means no kernel
   work closes the gap.
2. **The fork deliberately runs its 8-bit kernel slower than its 4-bit one below 64 rows** (52.4
   against 65.5 GMAC/s) — its switch at 32 rows costs it at 64. Our own form's 8-row kernel is
   faster than the fork's at every width we measured, which is a small win nobody had noticed.
3. **The per-32 grid buys 1.39× in RMS, not 8×**, because the error is bounded by each block's own
   maximum and 32-weight maxima are nearly as large as 256-weight ones. The 3.6× KL difference
   between the fork's two paths therefore cannot be attributed to the grid alone.
4. **The per-32 kernel is 0.58-0.65× the per-256 one** — enough to make a per-32 fast prompt mode
   slower than the *exact* engine — while its bytes are 1.25 against 1.03. The decision is not
   "finer is better": it is 1.7× the kernel time and 21 % more memory for 1.39× the per-weight
   accuracy, and the fork's own end-to-end accuracy number for that path is 0.000945.
5. **The `-fa 0` ablation is the cheapest account of the fork's attention** (9 % of its 512-token
   pass, 20.5 % of its 4096-token one) and needs no profiler — worth remembering, because `perf` is
   not available here at all.
6. **The load average on this VM is not a quietness metric**, and the `prof` format subtimers are
   core-seconds, not wall: both cost this round time (the first account table mixed the two models
   under one key, and its group rates read 3× too high until the key was fixed).

## 9. What was not done

* **A per-32 fast prompt mode end to end** (part B is a kernel prototype by the brief): no `kld`
  run, no dispatch, no golden. Its KL on the 1B is not measured.
* **A SIMD builder for the per-32 form**: the prototype's builder is scalar (0.65 G weights/s), so
  its build time is not comparable to the engine's 1.9 G weights/s AVX-512 builder. The ratio of
  the two *scalar* builders is not meaningful either, because the per-256 twin `build_q4k` is much
  slower than the per-32 path's dequantise-once-then-requantise loop for reasons this round did not
  chase.
* **`perf` profiles**: not available (no binary, no `sudo`, `perf_event_paranoid` 2, WSL2).
* **The 4096-token one-thread cells at five rounds**: one round each (§5), and llama.cpp for the
  Qwen3 4096/1 cell and ik's defaults for both 4096/1 cells are missing.
* **A 512-bit attention score kernel or a wider value mix**: the account says attention is 38 % of a
  4096-token fast pass and 2.4× the fork's in absolute time, and S5.2b measured a 512-bit score row
  30-70 % *slower* when tried. That is a kernel project, not a this-round one, and the same-bits
  ceiling S5.2c worked out (the f32 roof with two rows per V pass) still holds.

## 10. Exact commands

```sh
# the account: one cell, and the two sweep drivers
./scripts/account-cell.sh pg-fast ~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf 512 1 raw/a.jsonl
./scripts/account-rounds.sh 2 raw/account/rounds-a.jsonl
./scripts/account-slow.sh raw/account/rounds-slow.jsonl 1
python3 account.py raw/account/rounds-a.jsonl      # the tables
python3 gguf_macs.py <model.gguf>                  # the MAC counts the tables divide by

# the sixteen-row body against two eight-row calls
KQBENCH_PAIR=1 taskset -c 0 target/release/examples/kqbench 8 40 1

# the per-32 form: kernel, bytes and build
KQBENCH_PER32=1 taskset -c 0-11 target/release/examples/kqbench 8 30 12
KQBENCH_PER32=1 taskset -c 0    target/release/examples/kqbench 8 30 1

# the per-weight error of both forms, on the real files
cargo run --release -p peregrine-bench --example kq8model -- p32 \
    ~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf
cargo run --release -p peregrine-bench --example kq8model -- p32 \
    /var/tmp/pg/s54c/models/Qwen3-1.7B-Q4_K_M.gguf

# the fork's own kernel, by shape, through its own library
F=~/.cache/peregrine-bench/ik_llama.cpp-479074c
g++ -O2 -I $F/ggml/include benchmarks/ik_kernel_probe.cpp -o /tmp/ik_kernel_probe \
    -L $F/build/ggml/src -lggml -Wl,-rpath,$F/build/ggml/src
taskset -c 0 /tmp/ik_kernel_probe q4k 8192 8,64,128,256,512 5
taskset -c 0 /tmp/ik_kernel_probe q6k 8192 8,64,128,256,512 5

# the fork's pass, by ablation
IK=~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/bin/llama-bench
for f in "-rtr" "-rtr -fa 0" "" "-fa 0"; do
    taskset -c 0-23 $IK -m <1B>.gguf -p 512 -n 0 -t 12 -r 3 -o json $f
    taskset -c 0-23 $IK -m <1B>.gguf -p 4096 -n 0 -t 12 -r 3 -o json $f
done

# the eight cells, six arms per cell, interleaved rounds
./scripts/speed-sweep.sh 2 raw/speed/sweep-a.jsonl \
    "$L|512|12|exact,fast-base,fast-pair,ik,ik-rtr,llama" \
    "$L|512|1|exact,fast-base,fast-pair,ik,ik-rtr,llama" \
    "$Q|512|12|exact,fast-base,fast-pair,ik,ik-rtr,llama" \
    "$Q|512|1|exact,fast-base,fast-pair,ik,ik-rtr,llama"
./scripts/speed-sweep.sh 3 raw/speed/sweep-b.jsonl \
    "$L|4096|12|exact,fast-base,fast-pair,ik,ik-rtr,llama" \
    "$S|512|12|exact,ik,llama" "$S|4096|12|exact,ik,llama"
./scripts/speed-sweep.sh 3 raw/speed/sweep-c.jsonl \
    "$Q|4096|12|exact,fast-base,fast-pair,ik,ik-rtr,llama"
./scripts/speed-sweep.sh 2 raw/speed/sweep-d.jsonl "$L|4096|1|exact,fast-base,ik-rtr,llama"

# the A/Bs of the two candidates
./scripts/ab.sh 5 raw/account/ab-pair.jsonl 0-23 "<cells>"     # /tmp/pg-base against /tmp/pg-pair
./scripts/chunk-ab.sh 5 raw/account/ab-chunk.jsonl /tmp/pg-pair 256 128 "<cells>"
```

`/tmp/pg-base` and `/tmp/pg-pair` were built from the same tree with `git stash` / `git stash pop`
around `cargo build --release -p peregrine`; the sixteen-row dispatch was then dropped, so
`pg-base` is this branch's engine and `pg-pair` differs only in the two call sites in
`blocks/linear.rs`.

## 11. The files

| file | what |
|---|---|
| `README.md` | this report |
| `account.py`, `gguf_macs.py` | the account's tables, from `promptbench`'s counters and the models' tensor tables |
| `scripts/account-cell.sh`, `account-rounds.sh`, `account-slow.sh` | one account cell, the 512/4096 × 12-thread rounds, the 4096 × 1-thread rounds |
| `scripts/speed-sweep.sh` | one round of six arms, interleaved |
| `scripts/ab.sh`, `chunk-ab.sh` | the two in-round A/Bs of two builds |
| `scripts/speed-cell.sh`, `speed-one.sh` | S5.4c's drivers, kept for the load-average record |
| `raw/account/` | the `promptbench` JSON, the A/Bs, and the account tables |
| `raw/speed/` | the four sweeps |
| `raw/partb/` | the per-32 kernel table and the per-weight error tables |
