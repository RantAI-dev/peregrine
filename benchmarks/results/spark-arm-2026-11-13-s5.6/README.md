# The pool on a contended host (S5.6)

S5.4k left one product problem: at ten threads, 3.7–4.0 s of a 7.5–8.8 s prompt pass
was the caller waiting for a pool the co-tenant descheduled, and one thread sweep read
879 tok/s at eight threads against 543 at ten. This round reproduces that with a
controlled disturbance, accounts for it, and removes most of it (the design is in
[`docs/design/s5.6-pool-contended.md`](../../../docs/design/s5.6-pool-contended.md)).

- Date: 2026-11-13 02:00 – 10:00 local (UTC+7).
- Host: DGX Spark, aarch64, 20 cores in two clusters: 10 × Cortex-X925 at 3.9 GHz
  (CPUs 5–9 and 15–19) and 10 × Cortex-A725. Every engine and every disturber is
  pinned to the ten X925 CPUs (`taskset -c 5-9,15-19`); the unpinned rows are the
  exception and say so. The host was shared all window: the 1-minute load average is
  beside every raw record, and the per-core idle share of the ten CPUs over each run
  is beside it too.
- Peregrine exact mode, `PEREGRINE_EMBED_CACHE_MB=0`; llama.cpp v0.9.0
  (`cd959ba`), CPU only; ik_llama.cpp `479074cc` at its defaults. ik's build cannot
  load `modern-bert`, so the embedding cell has no ik column.
- Models: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` and `gte-modernbert-base-F16.gguf`.
- Disturbance: `crates/peregrine-bench/examples/loadgen` runs N busy integer threads
  (N = 0, 1, 2, 4) under the same CPU mask. `run_grid.py` starts it for one cell and
  stops it by PID in a `finally`; no `pkill`, no process outlives its command.

Method: rounds interleave the levels, so a co-tenant that arrives or leaves affects
each level of that round alike. Prompt and decode cells are 5 rounds, the long prompt
cells 2–5 rounds depending on the engine's run time, the embedding wave 3 rounds.
The tables are medians; spreads are in `table.txt`. All ratios are recomputed from
the medians (`summarize.py`), never typed.

## The disturbance curve (ten threads, pinned, medians)

`own` is the engine against its own quiet cell; the ratio is that engine ÷ each
other engine in the same cell and level.

### Peregrine before

| cell | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| prefill 512 tok/s | 719.0 | 213.6 (0.30) | 201.8 (0.28) | 280.1 (0.39) |
| prefill 4 096 tok/s | 518.4 | 180.9 (0.35) | 169.2 (0.33) | 234.3 (0.45) |
| decode tok/s | 124.5 | **9.9 (0.08)** | 59.2 (0.48) | 64.1 (0.52) |
| embed 32×128 texts/s | 22.3 | **2.66 (0.12)** | 4.42 (0.20) | 6.87 (0.31) |

### Peregrine after

| cell | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| prefill 512 tok/s | 715.5 | 591.9 (0.83) | 554.6 (0.78) | 436.7 (0.61) |
| prefill 4 096 tok/s | 515.3 | 442.6 (0.86) | 414.1 (0.80) | 322.3 (0.63) |
| decode tok/s | 124.4 | 105.7 (0.85) | 104.7 (0.84) | 92.2 (0.74) |
| embed 32×128 texts/s | 22.6 | 17.4 (0.77*) | 16.4 (0.73*) | 13.3 (0.59*) |

\* the after embedding run's quiet wave before the levels read 22.6 in the controlled
A/B; the grid's own quiet wave was depressed by host drift, so the fractions use the
A/B's quiet number. Raw values are in `raw/after-pg-embed.jsonl`.

### The two other engines (ten threads, pinned)

| cell | engine | 0 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| prefill 512 | llama.cpp | 287.4 | 255.7 (0.89) | 229.8 (0.80) | 175.4 (0.61) |
| prefill 512 | ik | 1 097.5 | 459.0 (0.42) | 382.4 (0.35) | 312.3 (0.28) |
| prefill 4 096 | llama.cpp | 241.6 | 213.5 (0.88) | 192.3 (0.80) | 145.4 (0.60) |
| prefill 4 096 | ik | 813.4 | 370.4 (0.46) | 293.3 (0.36) | 240.6 (0.30) |
| decode | llama.cpp | 250.1 | 156.9 (0.63) | 145.1 (0.58) | 103.9 (0.42) |
| decode | ik | 454.2 | 144.6 (0.32) | 146.6 (0.32) | 160.0 (0.35) |
| embed | llama.cpp | 17.0 | 12.1 (0.71) | 10.7 (0.63) | 8.3 (0.49) |

ik is much faster in absolute terms (its aarch64 k-quant wide path) and loses a
larger *fraction* under the same disturbance; llama.cpp is the graceful-degradation
yardstick.

## The account

4 096-token prefill, ten threads, 12 385 joins, `PEREGRINE_PROF2=1`
(`raw/account.txt`):

| | quiet | 1 disturber | 4 disturbers |
|---|---|---|---|
| wall, before | 7.82 s | 22.65 s | 17.41 s |
| caller's wait, before | 1.35 s (17 %) | 15.20 s (67 %) | 7.57 s (43 %) |
| wall, after | 7.87 s | 9.42 s | 12.74 s |
| caller's wait, after | 1.34 s (17 %) | 0.78 s (8.7 %) | 0.91 s (7.4 %) |
| straggler gap, before (1 disturber) | — | mean 0.97 ms, max 3.82 ms | — |
| straggler gap, after (1 disturber) | — | mean 0.019 ms, max 1.68 ms | — |
| last worker, 1 disturber | — | uniform over workers 1–9 | — |

**Where the time went.** The linear matmuls already claim dynamic chunks (51 rows at
4 096/10), so a late worker holds one chunk, not a share; attention and the norms use
static shares, and their loss is bounded by one share. The join's cost is the barrier
contract, not the split: a worker that has not run yet must run and decrement before
the caller may return. The nine finished workers then spin on the epoch for the 1 ms
window while the late worker needs their cores. The adaptive-spin ladder did not
react at one disturber — its schedstat gate read 85–207 permille against a 350 gate —
while the join waited 67 % of its wall; `PEREGRINE_SPIN_US=0` alone recovered
181 → 305 tok/s, which is the proof that the spin was the first cause.

The two mechanisms (park on a stalled join; claim ranges from a pool counter so a
join does not wait for a worker that has not arrived) are described with their
numbers in the design note. Their effect, one run per cell in the same window:

| | before | park only | park + claim |
|---|---|---|---|
| prefill 4 096, 1 disturber | 181.7 | 311.1 | **443.9** |
| prefill 4 096, 4 disturbers | 235.5 | 241.5 | **323.8** |
| decode, 1 disturber | 9.9 | — | **102.6** |
| embed 32×128, 1 disturber | 2.66 | — | **17.4** |
| quiet prefill 4 096 | 524.5 | 521.0 | 521.0 |

## Part 3: the goal

The goal has two halves. First, at every disturbance level Peregrine's loss relative
to its own quiet cell is no larger than llama.cpp's. Second, asking for more threads
is never slower than asking for fewer by more than the spread.

At ten threads, after the fix:

| cell | level | Peregrine own | llama.cpp own | met? |
|---|---|---|---|---|
| prefill 512 | 1 | 0.827 | 0.89 | **no**, −6.3 pts |
| prefill 512 | 2 | 0.775 | 0.80 | **no**, −2.5 pts |
| prefill 512 | 4 | 0.610 | 0.61 | yes |
| prefill 4 096 | 1 | 0.859 | 0.88 | **no**, −2.1 pts |
| prefill 4 096 | 2 | 0.804 | 0.80 | yes |
| prefill 4 096 | 4 | 0.625 | 0.60 | yes |
| decode | 1 / 2 / 4 | 0.850 / 0.842 / 0.741 | 0.63 / 0.58 / 0.42 | yes |
| embed 32×128 | 1 / 2 / 4 | 0.911 / 0.858 / 0.693 | 0.71 / 0.63 / 0.49 | yes |

The three unmet cells are the prefill cells at one and two disturbers. The caller's
wait there is already ~0 (the claimed join), so the residual is the throughput cost
of eleven or fourteen runnable threads on ten cores and the X925 cluster's clock
under all-core load: every stage is 15–30 % slower, uniformly (`raw/account.txt`).
Closing 1.3 points on prefill 4 096/1 would need a narrower pool, and the sweep says
that is not a win at one disturber (449.6 tok/s at ten threads against 421.0 at nine
and 369.7 at eight). At two disturbers the same sweep would favour eight, but only by
a few points, and it loses at one; that is the active-width candidate, measured and
dropped.

Thread monotonicity at ten threads is met: prefill 4 096 reads 322.3 at ten against
327.3 at eight (within spread) and 161.6 at four; decode 92.2 at ten against 85.9 at
eight and 51.7 at four; prefill 512 436.7 at ten against 402.4 at eight and 252.3 at
four. The 879 → 543 case from S5.4k is gone: the quiet ten-thread cell is the same,
and the disturbed ten-thread cell is far above the eight-thread one's old value.

Unpinned (ten threads, 3 rounds), the OS can move workers off the disturbers' CPUs:
prefill 4 096 reads 329.1 at four disturbers against 329 (pinned) and decode 110.7
against 92.2; prefill 512 464.8 against 436.7. Pinning is still the right comparison
against an engine started with `-t N`, and it is the worst case for the pool.

## What was not measured

- **ik_llama.cpp's embedding cell**: its `479074c` build refuses `modern-bert`
  (`unknown model architecture`), so the embedding yardstick is llama.cpp only.
- **A quiet window**: there was none. Every record carries the load average and the
  idle share; the interleaved rounds and medians are the defence.
- **macOS and Windows**: the stall trigger's host gate reads Linux's schedstat; on
  the other platforms the mechanism stays off (the claimed partitions themselves are
  portable and compile). The native ARM tests and the x86_64, windows-gnu and
  aarch64-apple-darwin `cargo check`s are the coverage.

## Files

- `run_grid.py` — the decode/prefill grid: levels, threads, cells, interleaved
  rounds, the disturber by PID, idle share and load per record.
- `embed_wave.py` — the 32-text embedding wave against a server per engine.
- `summarize.py` — medians, spreads, and the Part 3 loss table from the raw records.
- `idle_share.py` — per-core idle share of CPUs 5–9,15–19 over a window.
- `account.sh` — the join counters for one 4 096-token pass, before and after.
- `embed_quiet_ab.py` — the quiet-host A/B of the two binaries on the embedding cell.
- `raw/` — one JSONL per engine/phase, `account.txt`, `table.txt`, `part3.txt`.
- the disturber: `crates/peregrine-bench/examples/loadgen.rs`.
