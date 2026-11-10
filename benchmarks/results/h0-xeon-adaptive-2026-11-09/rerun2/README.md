# H0: adaptive spin-wait, second run (2026-11-09)

This is the re-run that follows the review of #125. It replaces the first ladder (configured /
half / zero, chosen from the run-queue reading) with a gate plus an outcome-based A/B of the pool's
own join wait, after instrumenting the first version showed why Gemma-3-1B regressed under load.

- Host: Intel Xeon E5-2680 v4 under KVM, 8 logical CPUs, 1 thread/core, 11 GiB RAM.
  `taskset -c 4-7`, `peregrine bench --threads 4`, `--prompt-tokens 512 --tokens 64`.
- Branch at `origin/main` `600357e` (which now has #126 sliding-window attention) plus this change.
  Base is the same tree without the spin-wait commits, with only `bench --threads` added so both
  binaries can be pinned to four workers.
- H0 was shared throughout and never quiet: the 1-minute load average at the start of a measured run
  was 6.6-56.2, median 13.3. Every raw JSON is paired with `load0`/`load1` in `raw/meta.txt`.
  Compare same-day ratios, not absolute tok/s.
- 5 rounds per cell, engines alternating order each round (A,B,C / C,B,A), medians. `branch/base`
  is the median of per-round ratios, not the ratio of medians.

## Results

### Quiet (no hog of ours; co-tenant load 6.6-30)

| Model | Engine | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| Llama-3.2-1B | base | 39.0 | 9.08 |
| | branch | 50.4 | 8.10 |
| | `PEREGRINE_SPIN_US=0` | 50.6 | 10.16 |
| | **branch/base** | **1.65x** | **1.15x** |
| Qwen2.5-0.5B | base | 84.9 | 13.61 |
| | branch | 117.4 | 20.31 |
| | `PEREGRINE_SPIN_US=0` | 128.5 | 21.94 |
| | **branch/base** | **1.41x** | **1.69x** |
| Gemma-3-1B | base | 45.9 | 6.90 |
| | branch | 72.9 | 12.02 |
| | `PEREGRINE_SPIN_US=0` | 66.0 | 13.15 |
| | **branch/base** | **1.55x** | **1.54x** |

The "quiet" condition still had co-tenant load above the 35 % gate at times, so the adaptive pool
was allowed to run its A/B and often chose parking. These are not quiet-host numbers; the clean
bluefin quiet-host A/B follows separately.

### Four CPU hogs on the same four cores

| Model | Engine | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| Llama-3.2-1B | base | 13.2 | 1.98 |
| | branch | 27.2 | 6.17 |
| | `PEREGRINE_SPIN_US=0` | 27.4 | 6.31 |
| | **branch/base** | **2.15x** | **2.75x** |
| Qwen2.5-0.5B | base | 12.4 | 1.93 |
| | branch | 51.7 | 11.54 |
| | `PEREGRINE_SPIN_US=0` | 55.7 | 13.24 |
| | **branch/base** | **3.88x** | **5.88x** |
| Gemma-3-1B | base | 11.1 | 1.41 |
| | branch | 28.4 | 6.07 |
| | `PEREGRINE_SPIN_US=0` | 28.1 | 6.11 |
| | **branch/base** | **2.63x** | **4.05x** |

Gemma is no longer the outlier: the old window gives 11.1 / 1.41 and the branch 28.4 / 6.07, against
17.3 / 3.72 for the first version of the adaptive pool. The branch matches the zero-spin control
(28.1 / 6.11), which is what its A/B picked.

### Eight CPU hogs on the same four cores

| Model | Engine | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| Llama-3.2-1B | base | 10.3 | 1.43 |
| | branch | 24.4 | 4.62 |
| | `PEREGRINE_SPIN_US=0` | 21.6 | 5.20 |
| | **branch/base** | **2.36x** | **3.32x** |
| Qwen2.5-0.5B | base | 9.1 | 1.18 |
| | branch | 38.3 | 9.50 |
| | `PEREGRINE_SPIN_US=0` | 42.5 | 10.40 |
| | **branch/base** | **4.02x** | **7.97x** |
| Gemma-3-1B | base | 7.7 | 0.95 |
| | branch | 22.8 | 4.13 |
| | `PEREGRINE_SPIN_US=0` | 20.6 | 4.54 |
| | **branch/base** | **2.90x** | **4.51x** |

## What the instrumentation found

`PEREGRINE_ADAPTIVE_LOG=1` prints the rung, the smoothed run-queue reading and every trial decision.
On the first implementation, Gemma-3-1B under four hogs sat on the parking rung for 98 % of the run
while the base pool (1 ms window) was faster; the parked-duration histogram showed tens of thousands
of sub-500 us parks, i.e. the pool was parking for regions that were about to arrive. Two things
were wrong:

1. At the bottom rung an 8 ms sample window rarely contained the 4 ms of runnable time the sampler
   requires, and the code threw the window (and its reference point) away. The "host eased" streak
   therefore almost never completed: a level-2 dwell lasted 10-18 s, and the mid-rung probe put it
   back for the rest of the run. Runnable and parked time are now accumulated across windows.
2. The run-queue reading cannot choose the rung. The same 40-90 % reading maps to different winners
   on different days: with four hogs at times the full window won by 2x, at other times parking won
   by 2x. A fixed threshold (and a middle rung) got it wrong in both directions.

The final policy: the smoothed reading is only a gate (trials start at 35 %, stop below 17.5 %). Once
open, the pool holds a rung, measures the caller's join wait for one window on each rung, and keeps
the lower mean with a 10 % margin. The starting rung is parking when the reading is >= 60 %, the
configured window otherwise, and the first hold is short (256 joins) so a wrong start is corrected
in about a second. `raw/meta.txt` records the decision log for the runs that enabled it; the trial
picked the same rung as `PEREGRINE_SPIN_US=0` in every loaded and heavy run above.

## Quiet-host neutrality

The code path when the smoothed reading stays below 17.5 % is the old pool: `wants_sample()` is two
relaxed loads and returns false, `budgets()` is one relaxed load of 0 (the configured window), and
no mutex or clock read is added to `run` beyond the 8 ms sample that was already there. The unit
tests pin `!wants_sample()` and the configured window for a quiet host.

H0 never gave a genuinely quiet window for the requested A/B. The lowest-load measurements:

- 200,000-join, 4-worker microbenchmark, `taskset -c 0-3`, 5 alternating rounds, load 7.8-9.4:
  median **1.79 us/join with adaptation on** and **3.21 us/join with
  `PEREGRINE_ADAPTIVE_SPIN=0`**; per-round ranges 1.74-4.02 and 1.99-5.98. The spread is co-tenant
  scheduling, not the pool. On a quiet host the gate stays closed and both builds run the same
  fork–join code.
- 128-token decode-only A/B (16-token prompt), `taskset -c 0-3`, 5 alternating rounds, load
  12.8-30.2: Gemma-3-1B median **13.81 tok/s adaptive vs 13.19 fixed**, Llama-3.2-1B
  **18.26 vs 15.94**. The host was too busy for this to be the clean check the review asked for;
  bluefin will provide it.

## Verification

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`,
`cargo check --workspace --target aarch64-unknown-linux-gnu`,
`PEREGRINE_GOLDEN=require cargo test --workspace --release --locked` (native and
`PEREGRINE_ISA=scalar`), and `cargo test -p peregrine-core`. Goldens are unchanged: this is
scheduling only.
