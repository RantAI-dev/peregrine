# `--fast-prompt` first look (S5.4b)

The opt-in 8-bit prompt pass of [`docs/design/s5.4-fast-prompt.md`](../../../docs/design/s5.4-fast-prompt.md)
§5b: prompt tokens of Llama 3.2 1B Q4_K_M, read with the resident 8-bit form, against the
same binary's exact default and against ik_llama.cpp. This is a first look for choosing the
fast mode's default wide chunk, not the report of S5.4c (which adds the accuracy numbers and
the multi-user rows).

- Date: 2026-11-12.
- Host: AMD Ryzen AI 9 HX 370 (Zen 5), 12 physical cores / 24 logical, AVX-512 (VNNI, VBMI,
  BF16), WSL2, 11.7 GiB RAM. Pinned with `taskset` as noted.
- **The 1-minute load is recorded beside every cell**: another worker (S5.2b) was running its
  test suite on this machine during the window, so the absolute numbers are low and some are
  noisy. The ratio within one round is the number to read.
- Model: `~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (762.6 MiB of weights).
  Raw JSONL: `runs.jsonl`; ik_llama.cpp's output: `ik.txt`.

## The form's cost at load

`PEREGRINE_FAST_PROMPT=1 peregrine inspect …` reports the resident form: **957.0 MiB**.
`/usr/bin/time -v` on a 8-token bench: 1.29 s wall and 1,013,572 KiB peak RSS without the
form, 1.81 s and 1,993,688 KiB with it (1-minute load 8.4): **+0.52 s and +957.1 MiB**. The
AVX-512 builder is what makes the build half a second; the scalar twin took 26 s on the same
file before it was dispatched.

## How the table was measured

```
# Peregrine, one cell (N = 64, 128, 256 or 512; fast = --fast-prompt or not)
PEREGRINE_PREFILL_CHUNK=N taskset -c 0-11 \
  target/release/peregrine bench -m <model> --prompt-tokens <512|4096> --tokens 1 $FAST --json

# the default mode at its default chunk
taskset -c 0-11 target/release/peregrine bench -m <model> --prompt-tokens 512 --tokens 1 --json

# ik_llama.cpp at its defaults (no -rtr: that is its default)
taskset -c 0-11 ik_llama.cpp/build/bin/llama-bench -m <model> -p 512 -n 0 -t 12 -r 5
```

`--tokens 1` makes the prefill the measurement (`prefill_tok_s`); a prompt pass with no slot
decoding is the wide one, so `PEREGRINE_PREFILL_CHUNK=N` is the width under test.

## The table

| Mode | chunk | 512 tok, 12 th | 4096 tok, 12 th | 512 tok, 1 th | 4096 tok, 1 th |
|---|---|---|---|---|---|
| default (exact) | 64 (wide default) | 327.0 | 226.7 | 46.0 | 25.7 |
| `--fast-prompt` | 64 | 410.8 | 273.2 | 53.8 | 34.1 |
| `--fast-prompt` | 128 | **437.6** | 283.2 | **67.5** | 33.7 |
| `--fast-prompt` | **256** | 428.3 | 285.0 | **67.5** | 36.6 |
| `--fast-prompt` | 512 | 402.9 | **288.2** | 67.6 | **42.5** |
| ik_llama.cpp (its defaults) | its default | 410.4 ± 2.3 | 326.7 ± 46.0 | 69.5 ± 16.0 | 46.6 ± 3.0 |

Prompt tokens per second. The fast rows are medians of three interleaved rounds at 512 tokens
and two at 4096/12 threads, and one round at 4096/1 thread (a 4096-token pass on one thread is
95 s); ik's are `llama-bench -r 5` (512) and `-r 3` (4096), median ± half-range. **256 is the
fast mode's default wide chunk** (`PEREGRINE_PREFILL_CHUNK` unset, `--fast-prompt` on).

The same numbers as ratios:

| Mode | 512/12 | 4096/12 | 512/1 | 4096/1 |
|---|---|---|---|---|
| fast 256 ÷ default | **1.31×** | **1.26×** | **1.47×** | **1.42×** |
| fast 128 ÷ default | 1.34× | 1.25× | 1.47× | 1.31× |
| fast 512 ÷ default | 1.23× | 1.27× | 1.47× | 1.65× |
| fast 256 ÷ ik (default) | 1.04× | 0.87× | 0.97× | 0.79× |
| fast 512 ÷ ik (default) | 0.98× | 0.88× | 0.97× | 0.91× |

Read this as: the fast mode is **1.26–1.47× the exact default** on the cell the mode is for
(no decoder in the pass, so the wide chunk applies). Against ik_llama.cpp **at its defaults**
(no `-rtr`; `llama-bench`'s default is off, while the fork's published 526.8 tok/s at 512 was
measured through its server with `-rtr` on) the fast mode is level at 512 tokens and 0.87–0.91×
at 4096, which is the design's §5 target of 0.85× (a first look, not S5.4c's table). The Q4_K
and Q6_K integers of the form are exact, so chunk width changes only time — the invariance test
in `crates/peregrine-models/tests/fast_prompt.rs` pins the bits.

The 1-minute load was between 2.3 and 10.2 across the cells (the per-run value is in
`runs.jsonl`); another worker's test suite was running for part of the window, and the 12-thread
4096 row shows it (±46 on ik). The cells were interleaved (default, 64, 128, 256, 512 per
round) so a drift hits every mode.

### A pinning note

`taskset -c 0-11` on this WSL2 host is **6 physical cores**, not 12: the same bench measures
141 tok/s there against 327 tok/s with the whole CPU set (`-c 0-23`, 12 threads, Llama 3.2 1B,
default mode, one-minute load 6.7). The 12-thread cells above pin to `0-23` and let the
scheduler place the twelve workers; the single-thread cells pin to core 0. A sparse
one-per-`lscpu`-core list (0,2,…,22) measured 144 tok/s: `lscpu`'s core ids do not describe this
VM's real topology.

## What was not done

The accuracy report and the full speed table (S5.4c): KL divergence against the exact mode and
against F16, the task-accuracy suite, multi-user rows, Gemma 3 1B and Qwen 2.5 0.5B. AVX2 and
NEON bodies (S5.4d). Q5_0's form: not wired in this step (Qwen 2.5's Q5_0 tensors stay exact).
