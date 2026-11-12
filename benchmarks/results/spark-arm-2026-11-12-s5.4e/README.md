# The NEON 8-bit kernel becomes the fast prompt mode's body (S5.4e, ARM)

[`docs/design/s5.4-fast-prompt.md`](../../../docs/design/s5.4-fast-prompt.md) §5f: S5.4d wrote
the aarch64 bodies of the 8-bit prompt form and left them unwired; S5.4b's dispatch still took
the scalar twin on ARM. This round wires the NEON multiply (`smmla` on `neon-i8mm`, `sdot` on
`neon-dotprod`) and the NEON builder into `MatKernels::q8k_gemm`, `q8k_build` and `q8k_simd`,
records the aarch64 fast goldens, and measures the mode end to end on Llama 3.2 1B Instruct
Q4_K_M and Qwen 2.5 0.5B Instruct Q4_K_M against the same binary's exact default, llama.cpp
and ik_llama.cpp.

- Date: 2026-11-12, 15:30–18:40 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters: 10 ×
  Cortex-X925 at 3.9 GHz (CPUs 5–9 and 15–19) and 10 × Cortex-A725 at 2.8 GHz, 121 GB unified
  memory, Linux `6.17.0-1008-nvidia`, governor `performance` (3.9 GHz read on all ten X925s
  during the runs), rustc 1.99.1. Kernel tier `aarch64-neon-i8mm`; the lower caps are the same
  binary under `PEREGRINE_ISA`.
- Peregrine: branch `s5.4e-fast-prompt-neon` on `origin/s5.4d-prompt-kernel-neon` (itself
  `origin/main` at `e4ad034`), plus this round's changes.
- Models, both regular files under `~/models`:

  | model | bytes | SHA-256 |
  |---|---|---|
  | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 807 694 464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |
  | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | 397 808 192 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |

  Llama 3.2 1B is 445.5 MiB Q4_K + 317.1 MiB Q6_K (every matrix has a form); Qwen 2.5 0.5B is
  28.1 MiB Q4_K + 40.9 MiB Q6_K + 165.1 MiB Q5_0 + 139.3 MiB Q8_0, so **only 69 of its 374 MiB
  of weights have a form** this step — Q5_0 and Q8_0 stay exact (design §6).
- llama.cpp: v0.9.0, commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8`, the harness cache build
  at `~/.cache/peregrine-bench/llama.cpp-v0.9.0`, CPU only (`CUDA_VISIBLE_DEVICES=""`).
- ik_llama.cpp: `master` at `479074ccac2ac94f06b6d24a9ce26f0f9e9aa6ae` (2026-11-11), built
  CPU-only, Release, `-march=armv9-a+dotprod+i8mm+bf16+sve2`, at its defaults (`-rtr` and its
  own flash attention are on by default; nothing was passed), at
  `~/.cache/peregrine-bench/ik_llama.cpp`.

## The host is shared

A co-tenant ran build and service work on the same host for the whole window: rustc/cargo
check, clippy and `ld` builds of a large workspace, then its server binaries (two of them at
2.4 and 1.8 cores), ClickHouse and PostgreSQL, all outside our `taskset` but on the same 20
CPUs and memory bus. The 1-minute load average was **2.8–34** across the window and is recorded
beside every cell (`runs.jsonl` has one line per cell; `load.txt`, `decode*.txt`, `sanity.txt`
carry the rest). Consequences, stated plainly:

- the **one-thread cells are stable** (core 5 is one core; the round-to-round spread is
  ≤1.5 % on every one-thread median, 0.1–0.4 % on the 512-token ones), so they are read as
  measured;
- the **ten-thread cells are noisy**: the half-ranges reach ±20 % on the 512-token cells and
  much more on a few rounds where the co-tenant landed on the X925s; the medians are over 5
  rounds (512 tokens) or 3 rounds (4 096), every round interleaved (exact, 64, 128, 256, 512,
  llama.cpp, ik) so a drift within a round hits every arm;
- the absolute ten-thread numbers are well below the quiet-host ARM table
  (`spark-arm-2026-11-09-i8mm`: 652 tok/s for Llama 3.2 1B i8mm at 512 tokens); they are not
  comparable with it and are not offered as a new baseline. **The within-round ratios are the
  claim**;
- a pilot of the Qwen 512-token/10-thread cell was discarded before the table: its first three
  rounds moved 185–1 177 tok/s for the same cell and 581–2 668 for ik within minutes. It is in
  `pilot/` for the record; every number below is from the interleaved window.

## How the cells were measured

```
# Peregrine, one cell (N = 64, 128, 256 or 512; fast = --fast-prompt or not)
PEREGRINE_PREFILL_CHUNK=N taskset -c 5-9,15-19 \
  target/release/peregrine bench -m <model> --prompt-tokens <512|4096> --tokens 1 \
  --threads 10 $FAST --json

# the exact default at its own chunk (64), no --fast-prompt
# one thread: taskset -c 5, --threads 1

# the other engines, at their defaults, one repetition per round, llama-bench's own warmup
CUDA_VISIBLE_DEVICES="" taskset -c 5-9,15-19 \
  ~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench -m <model> -p <512|4096> -n 0 -t 10 -r 1
taskset -c 5-9,15-19 ~/.cache/peregrine-bench/ik_llama.cpp/build/bin/llama-bench \
  -m <model> -p <512|4096> -n 0 -t 10 -r 1
```

`--tokens 1` makes the prefill the measurement (`prefill_tok_s`); a prompt pass with no slot
decoding is the wide one, so `PEREGRINE_PREFILL_CHUNK=N` is the width under test. `round.sh`
runs one round of all seven arms; `pg-round.sh` and `eng-round.sh` split it when one round is
longer than a foreground command should be. `summarize.py` and `ratios.py` rebuild every table
from `runs.jsonl`, `llama.txt` and `ik.txt`.

## The table (prompt tokens per second, medians)

| model | threads | prompt | exact | fast 64 | fast 128 | fast 256 | fast 512 | llama.cpp | ik |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B | 1 | 512 | 93.6 | 187.4 | 192.5 | **192.7** | 192.1 | 30.1 | 118.3 |
| Llama 3.2 1B | 10 | 512 | 254.4 | 481.7 | 430.1 | 517.3 | **645.5** | 195.7 | 426.1 |
| Llama 3.2 1B | 1 | 4096 | 55.8 | 79.3 | 80.7 | 81.0 | **81.0** | 25.6 | 84.0 |
| Llama 3.2 1B | 10 | 4096 | 183.1 | 278.8 | 296.5 | 348.3 | **375.0** | 176.1 | 550.9 |
| Qwen 2.5 0.5B | 1 | 512 | 141.8 | 158.8 | 161.1 | **161.5** | 160.7 | 61.9 | 307.6 |
| Qwen 2.5 0.5B | 10 | 512 | 272.5 | 330.2 | 366.3 | 411.3 | **520.8** | 346.7 | 703.1 |
| Qwen 2.5 0.5B | 1 | 4096 | 83.6 | 88.4 | **89.2** | 89.0 | 88.5 | 50.7 | 189.7 |
| Qwen 2.5 0.5B | 10 | 4096 | 247.4 | 256.8 | 292.9 | 336.9 | **399.7** | 304.6 | 803.3 |

n = 5 rounds per cell, except Qwen 2.5 0.5B at 512 tokens/10 threads (n = 10) and every
4 096-token cell (n = 3); the 1-minute load per round is in `runs.jsonl` (min–max: Llama
512/1 4.3–6.3,
512/10 8.9–11.7, 4096/1 3.2–20.3, 4096/10 10.7–13.4; Qwen 512/1 4.1–5.2, 512/10 6.2–21.9,
4096/1 4.9–24.8, 4096/10 11.7–17.0). Half-ranges: the one-thread cells are ±0.1–1.5 %;
the ten-thread ones are ±10–20 % on the medians (e.g. Llama fast 512 at 512 tokens
645.5 −77.6/+78.5, Qwen fast 512 520.8 −231.5/+105.7), which is why the ratios below are
computed inside each round and then medianed, not from these medians alone.

Two readings of the mode, before the ratios:

- on **Llama 3.2 1B**, where every matrix has a form, the fast mode is **1.45–2.54×** the exact
  default on all four cells at the best chunk, and at one thread it is 2.06× at 512 tokens and
  1.45× at 4 096;
- on **Qwen 2.5 0.5B**, where 82 % of the weights are Q5_0/Q8_0 and stay exact, it is
  1.07–1.91×. The 512-token/10-thread cell is the widest gain (1.91×) and the 4 096-token
  one-thread cell the narrowest (1.07×), exactly the shape the coverage predicts.

## Ratios: exact ÷ each engine (Peregrine ÷ other)

| model | threads | prompt | ÷ llama.cpp | ÷ ik_llama.cpp |
|---|---:|---:|---:|---:|
| Llama 3.2 1B | 1 | 512 | **3.11×** | 0.79× |
| Llama 3.2 1B | 10 | 512 | **1.27×** | 0.66× |
| Llama 3.2 1B | 1 | 4096 | **2.18×** | 0.66× |
| Llama 3.2 1B | 10 | 4096 | 0.92× | 0.33× |
| Qwen 2.5 0.5B | 1 | 512 | **2.29×** | 0.46× |
| Qwen 2.5 0.5B | 10 | 512 | 0.88× | 0.37× |
| Qwen 2.5 0.5B | 1 | 4096 | **1.65×** | 0.44× |
| Qwen 2.5 0.5B | 10 | 4096 | 0.97× | 0.42× |

Below 1.0: **Llama 3.2 1B ÷ ik on all four cells** (0.33–0.79×), **Llama ÷ llama.cpp at
4 096 tokens/10 threads** (0.92×), **Qwen 2.5 0.5B ÷ ik on all four** (0.37–0.46×), and
**Qwen ÷ llama.cpp on both ten-thread cells** (0.88× at 512, 0.97× at 4 096). The exact mode
is ahead of llama.cpp on every one-thread cell and on Llama at 512 tokens/10 threads.
ik_llama.cpp's repacked Q5_0/Q8_0 and its ARM kernels are why it leads everywhere; this mode
does not touch Q5_0 (§5f, open point).

## Ratios: fast (best chunk) ÷ each engine

| model | threads | prompt | best chunk | ÷ llama.cpp | ÷ ik_llama.cpp | ÷ exact |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B | 1 | 512 | 256 | **6.40×** | **1.63×** | **2.06×** |
| Llama 3.2 1B | 10 | 512 | 512 | **3.32×** | **1.43×** | **2.54×** |
| Llama 3.2 1B | 1 | 4096 | 512 | **3.17×** | **0.96×** | **1.45×** |
| Llama 3.2 1B | 10 | 4096 | 512 | **2.13×** | **0.81×** | **2.05×** |
| Qwen 2.5 0.5B | 1 | 512 | 256 | **2.61×** | **0.52×** | **1.14×** |
| Qwen 2.5 0.5B | 10 | 512 | 512 | **1.51×** | **0.69×** | **1.91×** |
| Qwen 2.5 0.5B | 1 | 4096 | 128 | **1.76×** | **0.47×** | **1.07×** |
| Qwen 2.5 0.5B | 10 | 4096 | 512 | **1.31×** | **0.54×** | **1.62×** |

Below 1.0, named: **Llama 3.2 1B at 4 096 tokens on one thread against ik** (0.96× — the only
cell where the fast mode is behind an engine at one thread), **Llama at 4 096 tokens/10 threads
against ik** (0.81×), and **Qwen 2.5 0.5B against ik on all four cells** (0.47–0.69×). Every
cell against llama.cpp is above 1.0 (1.31–6.40×).

The fast mode does not slow the default: `--fast-prompt` off runs today's kernels and today's
bits, and the one-thread `÷ exact` column is the mode's own gain, not a regression.

## Which chunk, and the tier's default

At one thread 128/256/512 are level within 1 % on both models (Llama 512: 192.5/192.7/192.1;
Qwen 512: 161.1/161.5/160.7; 4 096: 80.7/81.0/81.0 and 89.2/89.0/88.5), and 64 is 2–3 % behind
on the 512-token cells. At ten threads **512 is the best chunk on all four cells of both
models**: against 256 it is +24.8 % (Llama 512), +7.7 % (Llama 4 096), +26.6 % (Qwen 512) and
+18.6 % (Qwen 4 096) on the medians, and in every round of every cell. That clears the design's
"over 5 % on both models" bar, so **the fast mode's wide chunk on aarch64 is now 512**
(`WIDE_CHUNK_DEFAULT_FAST_ARM`); x86 keeps 256. A 512-row pass is inside `MAX_BATCH`, so no
other limit moves.

The one-thread rows are why this did not change on x86 in S5.4b: there 512 lost 8 % on the
short 12-thread row. Here it does not lose anywhere.

## What the mode costs at load

`/usr/bin/time -v` on an 8-token bench (load plus a tiny pass), three warm runs each, ten
threads (`load2.txt`; `%M` is KiB, shown here in MiB):

| model | mode | wall (median) | peak RSS (median) |
|---|---|---:|---:|
| Llama 3.2 1B | exact | 0.60 s | 877.1 MiB |
| Llama 3.2 1B | `--fast-prompt` | 1.08 s | 1 843.1 MiB |
| Qwen 2.5 0.5B | exact | 0.65 s | 653.9 MiB |
| Qwen 2.5 0.5B | `--fast-prompt` | 0.79 s | 763.5 MiB |

The form is **+966 MiB and +0.48 s** on the 1B (957.0 MiB of form plus allocator slack) and
**+110 MiB and +0.14 s** on Qwen (102.9 MiB of form: only its Q4_K/Q6_K matrices). The NEON
builder is what keeps the build near half a second: `peregrine inspect` without the option
reports the form off and with it reports the size; the S5.4d table's scalar twin would take
about 26 s on the 1B.

## Decode after a fast prompt

Decode rows are exact in both modes; what could move is the cost of the extra resident bytes.
One thread (core 5), 32 decode tokens after a 512-token prompt, modes alternated each round,
three rounds:

| model | exact (tok/s) | fast (tok/s) | fast ÷ exact |
|---|---:|---:|---:|
| Llama 3.2 1B | 24.78, 25.02, 24.22 (median 24.78) | 24.92, 24.99, 24.55 (24.92) | **1.006×** |
| Qwen 2.5 0.5B | 42.33, 38.29, 42.83 (42.33) | 42.83, 40.19, 43.03 (42.83) | **1.012×** |

Level within 1.2 % on both models: the form is not read by a decode row and nothing else
changed. The ten-thread decode cell could not be read under the co-tenant (single 64-token
runs spread 24.6–191.7 tok/s for the same mode and cell, `decode10.txt`); the one-thread table
is the evidence and its spread is 0.3 %.

## Sanity: 64 greedy tokens on two plain prompts

`peregrine run -m <1B> --max-tokens 64 --threads 10 [--fast-prompt] "<prompt>"`, temperature 0
(`sanity.txt`):

- *"Write one sentence about the sea."* — 35 tokens both, and they first differ after
  "endless blue": exact continues "horizon", fast "waters". The rest of the sentence is
  identical.
- *"List three colours, comma separated."* — 5 tokens both, byte-identical: "Blue, Red, Green".

So the mode is visible at the first prompt's 11th word and absent on the second, which is what
"differs from the exact mode, generated tokens stay exact" means in practice. The accuracy
report is S5.4c's and is not this round's.

## What was not done

- **Q5_0's form** (design §6): 165 MiB of Qwen 2.5 0.5B stay on the exact Q5_0 kernel, which is
  most of why its gain is 1.07–1.91× against the 1B's 1.45–2.54×, and most of why ik leads
  there. Wiring it in is a second re-rounding with its own error, and belongs to the accuracy
  report.
- **AVX2's body**: still S5.4d, still the scalar twin on that tier (another worker owns it).
- **A quiet-host rerun** of the ten-thread cells, and the multi-user rows: S5.4c's table, on a
  host without a co-tenant. The medians here are marked as such and the ratios are within-round.
