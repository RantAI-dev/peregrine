# Bonsai 2 on the aarch64 NEON tier (S2.8 F4)

The `PQ2_0` ternary kernels, the Hadamard rotation, the Gated DeltaNet step and the causal
conv, measured against the scalar tier and against PrismML's `llama.cpp` fork.

- Date: 2026-11-10.
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64: 10 × Cortex-X925 (3.9 GHz, CPUs 5–9 and
  15–19) and 10 × Cortex-A725 (2.8 GHz, CPUs 0–4 and 10–14), 121 GB LPDDR5X, one NUMA node.
  `asimd`, `asimddp`, `i8mm`, `sve`, `sve2`, `bf16`.
- Peregrine: `bonsai2-neon` on `neon-3` (the #129 tier), release profile,
  `rustc 1.99.1 (58cf8f3c3 2026-10-13)`, `CARGO_TARGET_DIR` under `/var/tmp/pg/work`.
  `native` = `aarch64-neon-i8mm`; `PEREGRINE_ISA=scalar` is the same binary with every vector
  kernel off, which is the A/B.
- Fork: PrismML `llama.cpp`, branch `prism`, commit
  `62c418ccd7aefee37997e94c2d9fac1eed7fc89b`, CMake Release, `GGML_CUDA=OFF`, native dispatch.
- Models, both under `/var/tmp/pg/models`, SHA-256 checked against `docs/dev/testing.md`:
  - `Ternary-Bonsai-2-2B-PQ2_0.gguf`, 517 312 992 B,
    `d5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928`
  - `Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf`, 1 274 642 176 B,
    `0acf517de1dc3d85c403a60df6b8b07a4878ac174f368d4390be98406a4d6c0d`
- `peregrine bench --prompt-tokens <p> --tokens 64`, 10-thread runs pinned to the X925s
  (`taskset -c 5-9,15-19`), 20-thread runs use all cores. PrismML is `llama-bench -p <p>
  -n 64 -t <t>`, the same affinity.

## The host is shared

Another worker ran a `peregrine serve` load test for the whole window, pinned to the same
10 X925 cores (8 slots, 10 threads; 680–830 % CPU, load 10–27). That is the worst case for
this comparison: the NEON prefill is at the memory-bandwidth ceiling (~220 GB/s at 400 tok/s),
so the co-tenant takes its bandwidth, while the scalar arm is compute-bound and barely
notices. Every cell below is therefore the **best of N rounds** (the rounds that caught a gap
between the co-tenant's phases), and the raw per-round data — including the medians — is in
`sweep.csv`, with the load average of each run. A dash means no round escaped the co-tenant;
those cells are marked "contended" and are not used for a ratio.

The one comparison that is clean within a single window is the MTP A/B on the 4B (both arms
took 4.4 s a round, back to back).

## Per-kernel speed (one thread, L1/L2-resident, X925)

`cargo run --release -p peregrine-bench --example tbench`, three runs in `microbench.txt`;
run 2 caught a quiet window and is the one quoted. One 12288 × 4096 PQ2_0 matrix (13.4 MB),
width = activations per pass.

| width | GMAC/s (quiet) | GMAC/s (shared host) |
|---|---:|---:|
| 1 (one row) | **56.6** | 19.5–19.9 |
| 2 (two rows, decode) | **74.2** | 25.6–26.3 |
| 4 (GEMM) | **88.7** | 29.3–29.8 |
| 8 | 87.2 | 29.2 |
| 16 | 82.4 | 80.6 (one quiet run) |

The two-row decode kernel is 1.31× the one-row kernel and the four-activation GEMM 1.57×,
which is the activation reuse the two passes are for. For scale, #129's Q4_K `sdot` GEMM on
the same core measured 77.6 GMAC/s and its `smmla` GEMM 125 GMAC/s.

**The `smmla` GEMM that was not shipped.** A 2 × 2-activation `smmla` prototype (natural-order
weight expansion, per-16 fold) measured 14.1 GMAC/s at two activations and 42.7 GMAC/s at
eight, against the `sdot` GEMM's 25.7 and 30.1 in the same runs — a win only when the weight
expansion hoists out of the activation loop, and its block-before-scale fold would make a
row's bits depend on the pass width. See `docs/design/s2.8-bonsai2.md` §1.

## End to end (best of N rounds)

| model | threads | prompt | arm | prefill tok/s | decode tok/s | prefill vs scalar | decode vs scalar |
|---|---|---:|---|---:|---:|---:|---:|
| 2B | 10 (X925) | 512 | scalar | 14.8 | 11.0 | — | — |
| | 10 | 512 | **NEON** | **404.0** | **140.4** | **27.3×** | **12.8×** |
| | 10 | 512 | llama.cpp | 167.7 | 88.7 | 11.3× | 8.1× |
| | 20 | 512 | scalar | 15.6 | 9.2 | — | — |
| | 20 | 512 | **NEON** | **433.1** | **146.8** | **27.8×** | **16.0×** |
| | 20 | 512 | llama.cpp | 200.5 | 64.5 | 12.9× | 7.0× |
| | 10 | 4096 | NEON | 102.5 | 10.1 | — | — |
| | 10 | 4096 | llama.cpp | 98.6 | 5.9 | — | — |
| | 20 | 4096 | NEON | 83.4 | 3.6 | — | — |
| 4B-MTP | 10 (X925) | 512 | scalar | 5.7 | 2.1 | — | — |
| | 10 | 512 | **NEON** | **164.1** | **64.8** | **28.8×** | **30.9×** |
| | 10 | 512 | **NEON + MTP** | **163.8** | **80.7** | 28.7× | 38.4× |
| | 10 | 512 | llama.cpp | 65.3 | 6.6 | 11.5× | 3.1× |
| | 20 | 512 | scalar | 7.8 | 1.8 | — | — |
| | 20 | 512 | **NEON** | **185.2** | **67.7** | **23.7×** | **37.6×** |
| | 20 | 512 | llama.cpp | 80.9 | 32.5 | 10.4× | 18.1× |
| | 10 | 4096 | **NEON** | **134.4** | **56.9** | — | — |
| | 10 | 4096 | llama.cpp | 62.2 | 38.9 | — | — |
| | 20 | 4096 | NEON | 51.2 | 2.7 | — | — |

The 2B scalar row is the pre-PR measurement on this host (14.8 / 11.0 tok/s at ten X925
cores); the same binary's scalar tier measured 13.7 / 3.3 as a median under the co-tenant, so
the pre-PR number is the one the ratio uses. The 4B has no pre-PR measurement; its scalar row
is the scalar tier's median, and its decode (2.1) is burst-sensitive — the best scalar round
was 4.8.

**Peregrine ÷ PrismML at 512:** 2B 2.41× / 1.58× (10 threads) and 2.16× / 2.28× (20);
4B 2.51× / 9.8× (10) and 2.29× / 2.08× (20). PrismML's PQ2_0 is a scalar kernel on aarch64,
so its decode is compute-bound; Peregrine's is at the bandwidth ceiling.

**MTP, 4B, ten X925 cores, same window:** decode 64.8 → 80.7 tok/s (**+24.5 %**), prefill
unchanged (163.3 → 162.4). On PrismML's fork, `llama-cli --spec-type draft-mtp` on the same
file measured 14.7 → 12.7 tok/s on a short chat prompt under the co-tenant, i.e. no gain; the
fork's MTP is not a useful comparison on this host.

**4096 context.** No quiet window was captured for the 2B at 4096 (its runs took 44–60 s
against 10 s uncontended), so the 4096 rows are shared-host only and no ratio is claimed from
them. The 4B's 134 tok/s prefill at 4096 is from a quiet round; the 2B's is not, and the two
are not comparable. At 4096 the same 543 MB of weights cross the bus per 16-token chunk, so
the NEON prefill is bandwidth-bound and the co-tenant's share of the bus is the whole story.

## What this says

The kernels, not the model, were the problem: a row of `PQ2_0` is 34 bytes for 128 weights, so
at 56.6 GMAC/s per core the 2B's 1.88 G MAC/token need ~1.3 ms of the ten cores' time and the
543 MB/token need ~2 ms at 273 GB/s — decode should be near 300 tok/s, and the measured 140 is
the rest of the engine (rotation, delta state, barriers). On the scalar tier the same row costs
~40× more. PrismML's fork still runs the scalar PQ2_0 kernel on aarch64, which is why it sits
at 65–89 tok/s decode and Peregrine is 1.6–2.3× ahead there, and 2.2–2.5× ahead on prefill.
