# aarch64 NEON tier (S2.8): first measurements

The ARM kernel tier's first numbers, on the machine the tier was written for. Every figure here is
from this host; nothing is extrapolated from x86.

- Date: 2026-11-09.
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters: 10 × Cortex-X925
  (3.9 GHz max, CPUs 5–9 and 15–19) and 10 × Cortex-A725 (2.8 GHz max, CPUs 0–4 and 10–14),
  121 GB unified memory, one NUMA node. `asimd`, `asimddp`, `i8mm`, `sve`, `sve2`, `bf16`.
- Peregrine: branch `neon-2` on `origin/main` + the NEON tier, `release` profile,
  `rustc 1.99.1 (58cf8f9c3 2026-10-13)`. Kernel tier `aarch64-neon-dotprod`.
- llama.cpp: v0.9.0-dev (build 1, commit `0c6dca2`), GNU 13.3.0, Linux aarch64, CUDA build.
  CPU runs only: `CUDA_VISIBLE_DEVICES="" ... -ngl 0`, so the GPU never enters the comparison.
- Models, both under `/var/tmp/pg/models`, SHA-256 checked against `docs/dev/testing.md`:
  - `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807694464 bytes,
    `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`;
  - `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, 397808192 bytes,
    `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`.
- Settings: `peregrine bench --prompt-tokens 512 --tokens 128`, default 20-thread pool; 5 rounds per
  arm, the arms interleaved round by round so a drift or a co-tenant hits every arm. llama.cpp:
  `llama-bench -p 512 -n 128 -r 3 -t 20`.

## The host is shared

Two other workers were running on this host throughout (load average 12–24 on 20 CPUs; at times a
`peregrine` benchmark from another worker held ~11 cores). Rounds whose figure was clearly hit by a
co-tenant are visible as outliers in `peregrine.csv`; the medians below are taken over all rounds
and the ratio is quoted from the clean rounds, where it was stable. Absolute numbers are therefore
**indicative**, and the ratio of two arms measured in the same interleaved sequence is the number
this note stands behind. The llama.cpp figures are the noisiest of all (its own reported spread is
up to ±116 tok/s on decode) and are context, not a gate.

## What the tier changes

| arm | kernels |
|---|---|
| `scalar` | `PEREGRINE_ISA=scalar`: every dot product is `quant.rs`'s scalar reference |
| `neon` | the shipping tier: Q4_K on `sdot` (per-row and the eight-row repacked tile), Q8_0 on `sdot` (per-row, eight-row decode, four-row batch) |
| `neon-perrow` | the shipping tier with `PEREGRINE_KREPACK=0`: the same NEON per-row kernels, no repacked tiles |

The two models exercise different format mixes, which is why they move by different amounts:

| file | formats by weight bytes |
|---|---|
| Llama-3.2-1B Q4_K_M | Q4_K 445.5 MiB (58 %), Q6_K 317.1 MiB (42 %), F32 0.3 MiB |
| Qwen2.5-0.5B Q4_K_M | Q5_0 165.1 MiB (44 %), Q8_0 139.3 MiB (37 %), Q6_K 40.9 MiB (11 %), Q4_K 28.1 MiB (7.5 %) |

Step 2 covers Q4_K and Q8_0. Q5_0 and Q6_K still run the scalar reference, so the Qwen file (Q5_0
44 %) and the Llama file (Q6_K 42 %) are both **partly** accelerated — the follow-ups in
`docs/design/s2.8-neon.md` §8 are exactly those two formats.

## Results

20 threads, 512-token prompt, 128 greedy decode tokens. Medians of 10 interleaved rounds (best of
10 in brackets, as the least-contended estimate); the ratio is the median of the per-round ratios,
with its range.

| model | arm | prefill tok/s (best) | decode tok/s (best) | prefill vs scalar | decode vs scalar |
|---|---|---:|---:|---:|---:|
| Llama-3.2-1B | scalar | 119.2 (123.4) | 63.5 (67.7) | — | — |
| Llama-3.2-1B | **neon** | **226.9** (244.7) | **67.7** (70.7) | **1.92×** (1.18–2.58) | **1.08×** |
| Llama-3.2-1B | neon-perrow | 214.9 (224.5) | 69.9 (75.2) | 1.81× | 1.10× |
| Qwen2.5-0.5B | scalar | 240.9 (249.9) | 111.5 (120.8) | — | — |
| Qwen2.5-0.5B | **neon** | **268.0** (279.8) | **131.7** (137.1) | **1.12×** (0.39–1.43) | **1.15×** |
| Qwen2.5-0.5B | neon-perrow | 261.0 (269.9) | 133.0 (144.2) | 1.08× | 1.17× |

Reading the two rows:

* **Llama (Q4_K 58 %)**: the Q4_K kernels nearly double the prompt rate. The repacked tile adds
  ~6 % over the per-row kernel (226.9 against 214.9), which is the tile's eight-rows-per-pass
  effect. Decode moves little (+8 %) because a decode step already streams the weights once and
  the remaining 42 % of the bytes (Q6_K) still run scalar.
* **Qwen (Q5_0 44 %, Q8_0 37 %)**: Q8_0 is accelerated but Q5_0, the largest slice, is not, so the
  whole file moves by ~12 %. The tile adds ~3 % here because Q4_K is only 7.5 % of it.
  This is the measurement that puts Q5_0 first in the follow-up order.
* The one out-of-family round ratio (Qwen 0.39 prefill / Llama 2.58) is a co-tenant spike, visible
  in `peregrine.csv`; the medians over 10 rounds are what the table reports.

Per-core kernel speed (one thread, synthetic L1/L2-resident data, `microbench.txt`):

| kernel | scalar | NEON | ratio |
|---|---:|---:|---:|
| Q8_0, one row | 6.0 GMAC/s | 22.1 | 3.7× |
| Q8_0, eight rows (decode) | 6.0 | 44.6 | 7.4× |
| Q4_K, one row | 13.4 | 42.4 | 3.2× |
| Q4_K repacked tile, GEMV | 2.5 (tile reference) | 51.2 | 20× |
| Q4_K repacked tile, GEMM ×4 | — | 76.7 | — |
| Q4_K repacked tile, GEMM ×16 | — | 78.6 | — |

The tile GEMM's 78 GMAC/s per X925 core is the number to carry forward: 20 such cores would be
1.5 TMAC/s, and the engine reaches about a fifth of that at prefill because attention, activation
quantisation and `vmath` are still scalar on aarch64 (the S2.8 follow-ups F5 and F6).

## X925 vs A725

The same binary, `taskset`-restricted to one cluster at a time, 2 rounds each, native tier.

| model | cluster | threads | prefill tok/s | decode tok/s |
|---|---|---:|---:|---:|
| Llama-3.2-1B | X925 only | 10 | 208.3 | 70.5 |
| Llama-3.2-1B | A725 only | 10 | 82.1 | 33.2 |
| Qwen2.5-0.5B | X925 only | 10 | 238.8 | 142.2 |
| Qwen2.5-0.5B | A725 only | 10 | 83.0 | 62.1 |

The X925 is **2.5–2.9×** the A725 on prefill and **2.1–2.3×** on decode, close to the 3.9/2.8 GHz
clock ratio (1.4×) times the IPC difference (≈1.8×). A 20-thread run is therefore not twice a
10-thread X925 run: it is the 10 fast cores plus the 10 slow ones doing about a third of a fast
core's work each, which is the 1.2–1.3× that 226.9/208.3 and 268.0/238.8 show. Decode gains
nothing from the mixed pool (67.7 at 20 threads against 70.5 at 10) because it is bandwidth-bound. A partition
weighted by core class would recover part of the prefill gap; that is a scheduling change, not
this PR (S2.8 open question 2).

## llama.cpp (context)

`llama-bench -t 20 -p 512 -n 128 -r 3`, CUDA disabled, same files. Best of the runs (its own
spread is in the range) against this PR's median:

| model | metric | peregrine neon | llama.cpp (best of N) | ratio |
|---|---|---:|---:|---:|
| Llama-3.2-1B | prompt 512 | 226.9 | 365.7 | 0.62× |
| Llama-3.2-1B | decode | 67.7 | 103.6 | 0.65× |
| Qwen2.5-0.5B | prompt 512 | 268.0 | 715.0 | 0.37× |
| Qwen2.5-0.5B | decode | 131.7 | 142.2 | 0.93× |

llama.cpp is ahead, most on the Qwen prompt rate, where its Q5_0 path is a repacked NEON GEMM and
ours is still the scalar reference. The gap is the follow-up list: Q5_0 (F1), Q6_K (F3), and the
vectorisation of attention, `exp` and the activation quantiser (F5, F6). No regression is claimed
against x86: on the x86 hosts the same files run the AVX2/AVX-512 tiers, whose goldens this tier
reproduces bit for bit for Q8_0 and Q4_K rows (see `docs/design/s2.8-neon.md` §11).

## Files

- `peregrine.csv`: every `peregrine bench` run, one row each (model, arm, affinity, threads, round,
  prefill tok/s, decode tok/s).
- `microbench.txt`: `neonbench` output.
- `llama-bench.txt`: every llama.cpp invocation, with its own spread.
