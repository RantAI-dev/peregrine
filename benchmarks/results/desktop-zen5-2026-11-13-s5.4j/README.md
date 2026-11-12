# The fast prompt mode's attention order, and the pass's non-kernel share (S5.4j)

Where S5.4i left the opt-in 8-bit prompt pass (`--fast-prompt`) against ik_llama.cpp's `-rtr`
path — attention 9.1 % of a 512-token pass and 38 % of a 4 096-token one, 2.1× the fork's
absolute time, and 10.4 % of the pass the caller waiting for its workers — what a prompt row of
the mode gains from a **summation order of its own**, what the pass's other terms are, and the
one scheduling lever that paid.

- Date: 2026-11-13 (00:50–) UTC+7.
- Host: AMD Ryzen AI 9 HX 370 (Zen 5), 12 physical cores / 24 logical, AVX-512 (VNNI, GFNI, VBMI,
  BF16), WSL2, 11.7 GiB, kernel `6.18.33.2-microsoft-standard-WSL2`. **The 1-minute load average
  is not a quietness metric on this VM** (S5.4i §1): the readings are recorded beside every run,
  and the readings that matter are the spread of the interleaved rounds (given for every cell
  below) — this window's spreads are 0.5-7 % on the 4 096-token cells and 2-20 % on the
  512-token ones, with one 20 % outlier caused by a single slow first round. The clock still runs
  ~10 % fast (MONOTONIC), the same for every engine, so the ratios are what to read.
- Peregrine: branch `s5.4j-fast-attention` = `origin/main` (`fc7ba98`, S5.4h) merged with
  `origin/s5.4i-fast-pass-vs-rtr` (`7fd9083`, which had not merged when this step started), plus
  this step's commits. AVX-512 tier, Rust 1.99.1.
- ik_llama.cpp: `~/.cache/peregrine-bench/ik_llama.cpp-479074c` (`479074c`), its `llama-bench`.
- llama.cpp: `~/.cache/peregrine-bench/llama.cpp-v0.9.0` (v0.9.0).
- Models: Llama 3.2 1B Instruct Q4_K_M `~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf`
  (807 694 464 B), Qwen3 1.7B Q4_K_M `/var/tmp/pg/s54c/models/Qwen3-1.7B-Q4_K_M.gguf`
  (1 282 439 264 B) — both regular files (`test -L` clean), the uploads S5.4c hashed.
- Corpus for the account and the accuracy: `~/.cache/peregrine-bench/s52b/corpus.txt` (the S5.2
  corpus) for the pass account, and wikitext-2's chunks through the S5.4c base files
  (`/var/tmp/pg/s54c/base/*.base.bin`) for the `kld` runs.

## 0. What this round found, in one paragraph

**A prompt row of the mode now has its own summation order for attention on AVX-512**, defined in
writing and implemented twice (a scalar twin that is the definition and a 16-lane body,
bit-identical to it on the whole length grid). Four query rows share one pass over a decoded key
and value tile; the score tree is 16-lane and four keys' trees share one transposed reduction.
Attention is **16 % faster at 512 tokens and 12 threads in situ** (10-16 % in isolation at 512)
and **3 % faster at 4 096 and 12 threads**, which is **2.1-5.6 % of a whole pass** — attention is
only 9 % of a 512-token pass, so this cannot be more. At one thread the engine's attention is
level: there the bucket is the tile's decode and the caller's row setup, not the bodies. Accuracy is unmoved at the reported precision (**0.003514 → 0.003464**
nats of mean KL on the 1B, −1.4 %, inside the run's error; 96.75 → 96.67 % same top token), the
mode's bound is re-baselined to **0.0040** on the project lead's recommendation (the owner's
confirmation pending, so the option stays Experimental), and **one golden moved**: the AVX-512
`llama-Q6_K-fast`. The pass's **non-kernel share** is now a list with numbers: the only serial
steps are the KV write (1.9 %) and the form's interleaved activation fill (~1.5 %, an estimate),
and the largest term is **the caller waiting for its workers (10.4 %)**, which is workers
*parking* between joins — 1 006 parks in one pass, 507 of them 1-5 ms — because the pool's spin
window (calibrated from the wake latency at startup, 1 ms here) is shorter than a prompt pass's
1-5 ms gaps. The pool now **measures the gap and stretches the window to cover it**: the fast
512-token 12-thread cell gains **3.3 %**, the exact engine is level, and nothing a worker
computes changes. The eight cells move from S5.4i's 0.80-0.95× ik's `-rtr` to **0.82-0.97×**; the
gap that is left is the fork's **int8** dot product in flash attention, which is a different
arithmetic and cannot be reached by reordering.

## 1. Method

### The account

`promptbench` drives the real `Engine` with real token ids from the corpus and prints, with
`PEREGRINE_PROF2=full`, the caller's `StepStats` buckets (matmul, attention, other, barriers) and
every `peregrine_core::prof` slot. The pool's own log (`PEREGRINE_ADAPTIVE_LOG=1`) prints the
park durations, the adaptive ladder's rungs and each thread's runnable wall. The kernels are
timed on their own with `attnbench`, which fills a real cache and times `attend` over every
layer; `PEREGRINE_FAST_ATTN=1` marks every row a prompt row of the mode, so the same binary
times both orders on the same cache. That is the step's kernel bench: the exact order against
the fast order, at 1 and 12 threads, before anything is dispatched.

### Speed cells

`scripts/sweep.sh` runs every arm once per cell, in the same order, in every round, so a drift
in the host hits every engine: `exact`, `fast-before` (S5.4i's build, `/tmp/pg-s54j-base`),
`fast` (this branch), `ik` (defaults), `ik-rtr`, `llama`. 12-thread cells are pinned to
`taskset -c 0-23` and one-thread cells to cpu0, as S5.4c and S5.4i did (`0-11` is six physical
cores on this VM). Medians; the rounds are in `raw/sweep-*.jsonl` and `scripts/table.py`
recomputes every ratio from them.

**The load average is recorded, not believed.** Every raw line carries the 1-minute load; it
reads 1.5-11 in this window while the pinned CPUs are otherwise idle, which is the S5.4i finding
again. The quietness evidence is the per-cell spread of the rounds (printed by `scripts/table.py`
and in §4) and the fact that the same window's exact engine and both forks' cells agree with
S5.4i's to 1-4 %.

### The A/Bs

Two builds alternating inside each round, for the two changes this step makes: the fast
attention order (`/tmp/pg-s54j-attn1` against the build before it) and the pool's gap-matched
spin window (`/tmp/pg-s54j-gap` against `/tmp/pg-s54j-attn1`). The kld tool compares fast
against exact *inside one binary* (`--ref-self`), so the accuracy "before" is the pre-change
build's own measurement, made in the same window.

## 2. A: the fast mode's attention order

The definition, the tier table and the tests are `docs/design/s5.4-fast-prompt.md` §5k; the code
is `mod fast` in `crates/peregrine-core/src/attn.rs`, beside the exact attention bodies, which
is where this tier's attention kernels already live.

### 2.1 The kernel bench, before anything was dispatched

`attnbench <model> <ctx> <threads> <reps> 128`, Llama 3.2 1B Q4_K_M, attention over every layer
of a prompt of `ctx` positions read in pieces of 128 rows. "Exact order" is the engine's own
attention; "fast order" is the same binary with `PEREGRINE_FAST_ATTN=1`. `raw/kernels.txt` is one
run of each cell; the 4 096-token one-thread pair was repeated three times because the first
reading disagreed with the engine (§2.2), and all three agreed with each other.

| positions / threads | exact order | fast order | ratio |
|---|---|---|---|
| 512 / 1 | 583.8 ms | **503.7 ms** | **1.159** |
| 512 / 12 | 95.4 ms | **86.7 ms** | **1.100** |
| 4 096 / 1 | 32 458-32 827 ms | 33 778-33 919 ms | **0.96** |
| 4 096 / 12 | 4 851 ms | 4 980 ms | **0.974** |

Those are whole-attention numbers (scores + soft-max + mix + the KV tile's decode + the caller's
row setup). The count of multiply-accumulates is `2 · layers · heads · N(N+1)/2 · head_dim` =
8.6 GMAC at 512 tokens and 550 GMAC at 4 096, so at 512 tokens the exact order runs at 14.7 GMAC/s
(one thread) and the fast order at 17.1, against this core's f32 FMA roof of 59.0 GMAC/s (§12.2 of
the prefill study): 25 → 29 % of the roof.

**The 4 096-token rows are not the engine's data, and the engine is what to read.** `attnbench`
fills every key and value with a regular small pattern, so a 4 096-position prompt's soft-max is
nearly uniform and the widened f32 tile the fast bodies read (this step keeps one tile choice for
every length: the widened one, see §2.2) is 8 KB a tile against the exact path's f16 4 KB — and
at 4 096 positions that traffic is out of L2 where at 512 it is not. In the engine, whose cache
holds real text, the same order *saves* 3 % of the attention at 4 096/12 (§2.2). The 512-token
rows, where both paths read the same widened tile, are the bench's honest verdict on the bodies:
**1.10-1.16×**.

### 2.2 In situ

`PEREGRINE_PROF2` on a real fast pass, Llama 3.2 1B, one interleaved pair of `promptbench` runs
per cell (the same binary against S5.4i's build), so the two columns differ only in this step's
code:

| stage | 512 / 12 before → now | 4 096 / 12 before → now | 512 / 1 | 4 096 / 1 |
|---|---|---|---|---|
| attention (`StepStats::attention`) | 0.118 → **0.098 s** (−16.8 %) | 4.69 → **4.55 s** (−3.0 %) | 517.6 → 522.2 ms (level) | 30.64 → 30.75 s (level) |
| barrier wait | 0.154 → **0.148 s** | 1.56 → **1.46 s** | 0 | 0 |
| parks over 1 ms, of the pass | 835 → **581** | 6 721 → **5 339** | 0 | 0 |
| the pass | 1.270 → **1.231 s** | 13.26 → **12.89 s** | 6.39 → 6.39 s | 77.56 → 77.58 s |

At one thread the engine's attention is the same time with either order: at 512 tokens both paths
read the widened tile and the bucket is what the bench's 1.16× does not survive (the tile's
decode and the caller's per-row setup, neither of which the order touches, are most of it), and at
4 096 tokens the exact path reads the f16 halves in registers and pays half the bytes. The order
is kept at every length — a row's bits may not depend on which pool it runs on, so there is one
definition per row and not one per thread count — and the 12-thread cells, where attention is
what the pass waits on, are where it pays.

### 2.3 Accuracy

`kld --ref-self --fast-prompt`, the same corpus and chunking S5.4c used (wikitext-2, 40 chunks
of 512 tokens, 10 200 scored positions), the mode against the exact mode of the same file, before
(`/tmp/kld-base`) and after (`/tmp/kld-new`):

| model | mean KLD before | mean KLD after | median before → after | same top p before → after |
|---|---|---|---|---|
| Llama 3.2 1B | 0.003514 ± 0.000049 | **0.003464 ± 0.000047** | 0.002472 → 0.002456 | 96.745 ± 0.176 % → 96.667 ± 0.178 % |
| Qwen3 1.7B | 0.017115 ± 0.000993 | **0.017336 ± 0.001186** | 0.006523 → 0.006527 | 93.912 ± 0.237 % → 94.196 ± 0.232 % |

−1.4 % and +1.3 % of itself: inside the run's own error and far under the 2 % bar the step was
given ("if the mean KL moves by more than 2 % of itself, STOP"). A summation order cannot move
the KL, and it did not. RMS Δp 1.408 → 1.359 % (1B) and 3.751 → 3.817 % (Qwen3).

### 2.4 The goldens

One file moved: `crates/peregrine-models/tests/golden/synthetic/x86_64-avx2-avx512-vnni-gfni-vbmi/llama-Q6_K-fast.golden`,
re-recorded natively here. The AVX-512 `llama-Q4_K-fast` was rewritten identically (that
synthetic model's logits are insensitive to the reordering: its argmax is one token
throughout). Nothing else: the AVX2 and scalar fast goldens, every exact golden and the
`aarch64*` set are untouched.

## 3. B: the pass's non-kernel share

`docs/design/s5.4-fast-prompt.md` §5l has the step list. In one sentence: the only **serial**
steps are the KV write (1.9 %) and the form's interleaved activation fill (~1.5 %, an estimate
from 24 MiB of zeroing and copying a pass); every other named step is threaded; and the largest
non-kernel term, **10.4 %**, is the caller waiting for workers that have **parked**.

The pool's log, one fast 512-token 12-thread pass, 1 006 parks: 3 under 100 µs, 149 under 500 µs,
270 under 1 ms, **507 of 1-5 ms**, 11 over 5 ms. The window that produces them is
`calib`'s `2 · L_wake` (1 000 µs here, clamped to [50, 2 000]), sized for decode's tens of
microseconds between joins; a prompt pass's joins are 1-5 ms apart (the caller's serial work
between them), so the window expires first. A fixed window against it:

| `PEREGRINE_SPIN_US` | 1 000 (calibrated) | 2 000 | 4 000 | 8 000 |
|---|---|---|---|---|
| fast 512/12, tok/s | 429.5 | 454.9 | **463.9** | 446.5 |
| ratio | 1.00 | 1.06 | **1.08** | 1.04 |

A fixed default is what S5.2b refused (4 000 µs costs 2-13 % on an oversubscribed host), so the
pool **measures the gap between the end of one fork–join and the start of the next** — one clock
read and one atomic per join — smooths it (one-quarter weight), and stretches the worker's spin
window to cover it (cap 8 000 µs; a gap over 20 ms is a pass boundary and ignored; a configured
0 and `PEREGRINE_ADAPTIVE_SPIN=0` are left alone). Three interleaved rounds, fast 512/12:

| build | round 1 | round 2 | round 3 | median | ratio |
|---|---|---|---|---|---|
| before (`attn1`) | 446.9 | 446.3 | 462.1 | 446.9 | 1.00 |
| gap-matched | **465.3** | **461.5** | 461.1 | **461.5** | **1.033** |
| exact engine, same pair | 343.1 | — | 349.3 | level | 1.018 |

At 4 096 tokens the lever is worth nothing measurable (the pass is 12 s, its joins 25 ms apart,
so few workers park); the exact engine does not get slower anywhere. Nothing a worker computes
changes, so no golden moves for this half.

## 4. The eight cells, before and after

Prompt tokens per second, medians, this window, `fast ÷ rtr` recomputed from the medians in each
row. Rounds: **five** for the four 512-token cells (three rounds in `raw/sweep-512.jsonl` plus
`-r3`/`-r4`/`-r5`), **three** for the 4 096-token 12-thread cells, **three** for the 4 096-token
one-thread cell of Llama 3.2 1B and **one** for Qwen3 1.7B's (both without `ik` defaults, as
S5.4i left them, and Qwen3's also without llama.cpp: at one thread a 4 096-token Qwen3 round is
~3.5 minutes an arm in this window, and the cell reads 0.972 against `-rtr` in its one round).
"before" is S5.4i's build measured in this window, not S5.4i's table.

| Model | prompt / threads | exact | fast before | **fast now** | ik (defaults) | ik (`-rtr`) | llama.cpp | fast ÷ exact | fast ÷ ik | fast ÷ **ik `-rtr`** | fast ÷ llama | before → after (fast) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 512 / 12 | 332.8 | 452.7 | **462.2** | 431.0 | 561.9 | 287.4 | 1.389 | 1.073 | **0.823** | 1.608 | +2.1 % |
| Llama 3.2 1B | 512 / 1 | 54.1 | 77.6 | **79.1** | 65.3 | 83.8 | 43.8 | 1.462 | 1.211 | **0.944** | 1.808 | +1.9 % |
| Llama 3.2 1B | 4096 / 12 | 257.6 | 325.9 | **338.6** | 319.5 | 380.2 | 198.9 | 1.315 | 1.060 | **0.890** | 1.702 | +3.9 % |
| Llama 3.2 1B | 4096 / 1 | 39.8 | 51.0 | **52.6** | — | 57.0 | 30.5 | 1.322 | — | **0.923** | 1.725 | +3.1 % |
| Qwen3 1.7B | 512 / 12 | 227.9 | 284.5 | **300.5** | 289.7 | 368.3 | 189.7 | 1.319 | 1.038 | **0.816** | 1.584 | +5.6 % |
| Qwen3 1.7B | 512 / 1 | 36.7 | 51.3 | **53.3** | 43.6 | 55.3 | 26.8 | 1.452 | 1.223 | **0.963** | 1.989 | +3.9 % |
| Qwen3 1.7B | 4096 / 12 | 168.8 | 209.0 | **215.5** | 206.1 | 245.2 | 125.5 | 1.276 | 1.046 | **0.879** | 1.717 | +3.1 % |
| Qwen3 1.7B | 4096 / 1 | 24.9 | 30.0 | **31.3** | — | 32.2 | — | 1.257 | — | **0.972** | — | +4.3 % |

**Cells below 1.0, all named:** against ik's `-rtr` path, **all eight** (0.816-0.972); against
ik at its defaults, **none** (1.038-1.223); against llama.cpp and the exact engine, none.

**Where each cell below 1.0 is held** (design §5l):

- **512 tokens, 12 threads (0.823, 0.816)**: the caller's wait for parked workers (10.4 % of the
  pass) and the two serial steps (KV write 1.9 %, the form's fill ~1.5 %). The kernel is 64 % of
  the pass and is level with the fork's own (§5i §2.4); attention is 9 % and is now 21 % faster.
  This is a pool-and-pass-shape gap, and the gap-matched window takes 3.3 % of it.
- **512 tokens, one thread (0.944, 0.963) and 4 096 tokens on one thread (0.923, 0.972)**: nothing
  but the kernel's rate (81 % and 55 % of the pass) and the caller's serial steps; S5.4i measured
  the kernel at 96.2/103.9 GMAC/s in situ against the fork's 94.9, so these cells are as close to
  1.0 as the same arithmetic allows.
- **4 096 tokens (0.890, 0.879)**: attention, which is 36 % of the pass and still 2.1× the
  fork's absolute time because the fork's flash attention is **int8** (§5k's last paragraph).
  This step takes 3 % of that stage at twelve threads and none of it at one; the rest is a
  quantising attention kernel.

**Decode and ten users did not move.** Decode after a fast prompt reads the exact kernels from a
cache the prompt wrote, so it is the same code as before this step; the ten-user cell's rows are
never fast rows and their attention is the exact path (`fast_rows` is filled from
`QuantAct::prompt`, which the scheduler does not set for generated or pooled rows). Measured
anyway (one interleaved pair each, no change expected — decode reads the exact kernels and a
generated or pooled row is never marked): decode 32 tokens after the fast prompt **57.7 → 58.0**
tok/s on the 1B; ten users (10 slots, 512-token prompts, one pool, greedy) **228.5 → 227.9**
tok/s of decode, with the ten-user pass's attention 105.5 → 93.7 ms.

## 5. Bits

- `git diff --name-status origin/main...HEAD -- '*.golden'` lists exactly one `M`:
  `x86_64-avx2-avx512-vnni-gfni-vbmi/llama-Q6_K-fast.golden`.
- The AVX-512 fast goldens were re-recorded natively (`PEREGRINE_GOLDEN=record`) and the test
  passes again without the switch.
- `PEREGRINE_ISA=avx2` and `PEREGRINE_ISA=scalar`: `synthetic`, `prefill_chunk` and `fast_prompt`
  pass and no golden of theirs moves (the tier table gives them the exact attention).
- The exact engine is untouched: with no row marked, `plan` and the row loops take the same paths
  as before (`fast_row` is false everywhere), and every exact golden is unchanged.
- The twin-vs-body tests sweep every key count 1…70, 127/128/129, 511/512/513, 4 095/4 096/4 097,
  head dims 64/128/80/16, random and soft-max-edge values, at one, three and four rows.

## 6. What surprised this round

1. **The reduction, not the loads, was the score kernel's cost.** The first 16-lane body (eight
   rows × two keys, one `T16` per key) was *slower* than the exact eight-lane kernel; the win
   came from grouping four keys so one transposed `hadd` tree reduces four dots. The 4×4 tile is
   not the shape the brief suggested first (a wider row tile) but the one the reduction cost
   picked.
2. **A spilling body is a 40 % regression.** The same arithmetic with a dynamically indexed
   accumulator array gave 0.60× the exact path; with constant trip counts it gives 1.11-1.21×.
3. **The pool's spin window was sized for decode and a prompt pass is a different animal.** 1 006
   parks in a 1 s pass, 507 of them 1-5 ms, and the fix is to measure the join gap rather than to
   lengthen a constant: +3.3 % on the cell furthest below the bar, for one clock read and one
   atomic per join.
4. **The accuracy did not move, and it should not have.** A reordering of the same sums is
   rounding; the measured change (−1.4 % on the 1B) is smaller than the run's own error, which is
   the point §4's bar was written to catch.
5. **The form's interleaved activation fill is the second serial step nobody had named.** It is
   ~1.5 % of a pass and it is the caller's alone, three times a layer.

## 7. What was not done

- **A quantising attention kernel.** The fork's 2.1× is an int8 dot product, not a tile shape;
  reaching it needs its own accuracy report (a Q8 query/key attention changes the distribution
  more than an order does).
- **Fewer, larger fork–joins.** The steps that could share one region are separated by matmuls
  that need the previous step's whole output in a transposed layout; that is a change to
  `matmul_group`'s interface, priced at ~10 % of a 512-token pass (§5l).
- **The form's fill on the pool.** It adds a fork–join to a step that has none, for ~1.5 %.
- **The AVX2, aarch64 and scalar attention bodies.** The tier table gives them the exact
  attention in this step; their fast goldens therefore did not move.
- **`ik` at its defaults for the two 4 096-token one-thread cells** and **llama.cpp for Qwen3
  1.7B at 4 096 tokens on one thread** (S5.4i left the same two cells empty: at one thread a
  4 096-token Qwen3 round is ~6 minutes an arm).

## 8. Exact commands

```sh
# the kernel bench: the same binary, both orders, on the same cache
cargo build --release -p peregrine-bench --example attnbench --features arch-llama,arch-qwen
taskset -c 0-23 target/release/examples/attnbench <1B>.gguf 4096 12 2 128
PEREGRINE_FAST_ATTN=1 taskset -c 0-23 target/release/examples/attnbench <1B>.gguf 4096 12 2 128

# the eight cells, six arms, interleaved rounds
L=~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf
Q=/var/tmp/pg/s54c/models/Qwen3-1.7B-Q4_K_M.gguf
A=exact,fast-before,fast,ik,ik-rtr,llama
PG_NEW=/tmp/pg-s54j-final ./scripts/sweep.sh 5 raw/sweep-512.jsonl \
    "$L|512|12|$A" "$L|512|1|$A" "$Q|512|12|$A" "$Q|512|1|$A"
PG_NEW=/tmp/pg-s54j-final ./scripts/sweep.sh 3 raw/sweep-4096.jsonl \
    "$L|4096|12|$A" "$Q|4096|12|$A"

# the two changes' own A/Bs (fast attention, then the gap-matched window)
./scripts/ab.sh 5 raw/ab-a.jsonl 0-23 /tmp/pg-s54j-base /tmp/pg-s54j-attn1 "$L|512|12" "$L|512|1"

# the pass account and the pool's log
PEREGRINE_FAST_PROMPT=1 PEREGRINE_PROF2=full PEREGRINE_ADAPTIVE_LOG=1 \
    /tmp/pb-new <1B>.gguf ~/.cache/peregrine-bench/s52b/corpus.txt 512 12

# the accuracy, before and after, inside one binary each
PEREGRINE_FAST_PROMPT=1 /tmp/kld-base --base /var/tmp/pg/s54c/base/llama32-1b.base.bin \
    --model <1B>.gguf --threads 12 --ref-self --fast-prompt
PEREGRINE_FAST_PROMPT=1 /tmp/kld-new --base /var/tmp/pg/s54c/base/llama32-1b.base.bin \
    --model <1B>.gguf --threads 12 --ref-self --fast-prompt
```

`/tmp/pg-s54j-base` is the tree before this step, `/tmp/pg-s54j-attn1` adds the attention order,
`/tmp/pg-s54j-gap`/`-final` adds the pool's gap window; `/tmp/kld-base` and `/tmp/kld-new` are the
`kld` example built from the same two trees.

## 9. The files

| file | what |
|---|---|
| `README.md` | this report |
| `scripts/sweep.sh`, `scripts/table.py` | one round of six arms, and the tables above from the raw files |
| `scripts/ab.sh` | the alternating A/B of two builds |
| `raw/sweep-*.jsonl` | the eight cells' rounds |
| `raw/ab-a.jsonl` | the two changes' A/Bs |
| `raw/account-*.txt` | the `promptbench` profiles and the pool's park log |
| `raw/kld-*.txt` | the accuracy runs, before and after |
| `raw/kernels.txt` | the kernel bench |
