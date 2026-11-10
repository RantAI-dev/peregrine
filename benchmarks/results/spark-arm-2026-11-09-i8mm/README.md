# The aarch64 i8mm tier (S2.8 F2): `smmla` tile GEMM

The repacked tile GEMMs on `smmla`, and the tier that names them. The GEMV stays on `sdot`.

- Date: 2026-11-09.
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64: 10 × Cortex-X925 (3.9 GHz, CPUs 5–9 and
  15–19) and 10 × Cortex-A725 (2.8 GHz, CPUs 0–4 and 10–14), 121 GB LPDDR5X, one NUMA node.
  `asimd`, `asimddp`, `i8mm`, `sve`, `sve2`, `bf16`.
- Peregrine: `neon-7` on `neon-6`, release profile, `rustc 1.99.1 (58cf8f3c3 2026-10-13)`.
  `native` = `aarch64-neon-i8mm`; `PEREGRINE_ISA=neon-dotprod` is the same binary with the
  `smmla` GEMM off, which is the A/B.
- Models, both under `/var/tmp/pg/models`, SHA-256 checked against `docs/dev/testing.md`:
  `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (807694464 B) and
  `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (397808192 B).
- `peregrine bench --prompt-tokens 512 --tokens 128`; 10-thread runs pinned to the X925s
  (`taskset -c 5-9,15-19`), 20-thread runs use all cores; five rounds per arm, arms interleaved.

## The host is shared

A 16-thread `peregrine serve` from another worker ran for most of the window (load 9–29 on 20
CPUs). The first Llama run was discarded: the co-tenant moved in and out between the arms within a
round and the ratios were unusable (0.5×–2.2×). The four runs below were taken afterwards, when
the co-tenant's phase was stable: every arm moved by 1–4 % round to round, so the medians and the
per-round ratios agree. Absolute throughput is indicative; the i8mm/dotprod ratio is the claim.

## Per-kernel speed (one thread, L1/L2-resident, X925)

`cargo run --release -p peregrine-bench --example neonbench -- 4096 512 300`, three runs; the table
is the median GMAC/s of the ten GEMM lines in `microbench.txt`.

| format | `sdot` GEMM | `smmla` GEMM | ratio |
|---|---:|---:|---:|
| Q4_K tile, ×4 | 77.6 | **125.1** | **1.61×** |
| Q4_K tile, ×16 | 77.3 | **125.4** | 1.62× |
| Q5_0 tile, ×4 | 34.9 | **56.8** | **1.63×** |
| Q6_K tile, ×4 | 52.4 | **78.8** | **1.50×** |
| Q6_K tile, ×16 | 52.4 | **78.9** | 1.51× |

Q8_0 is not in the table because it has no repacked **tile** on aarch64: its rank-1 rows would
have to be gathered into the 2×8 operand, and the gather costs the loads the wider multiply saves.
It stays on the `sdot` batch kernel in this tier, exactly as on `neon-dotprod`.

## End to end (median of 5 rounds)

| model | threads | arm | prefill tok/s | decode tok/s | prefill vs scalar | decode vs scalar |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 10 (X925) | scalar | 95.1 | 57.8 | — | — |
| | 10 | dotprod | 465.1 | 119.8 | 4.89× | 2.07× |
| | 10 | **i8mm** | **652.0** | **120.9** | **6.86×** | **2.09×** |
| | 10 | llama.cpp | 266.5 | 106.0 | — | — |
| | 20 | scalar | 125.4 | 64.8 | — | — |
| | 20 | dotprod | 496.0 | 108.9 | 3.96× | 1.68× |
| | 20 | **i8mm** | **629.8** | **111.6** | **5.02×** | **1.72×** |
| | 20 | llama.cpp | 322.0 | 86.4 | — | — |
| Qwen2.5-0.5B Q4_K_M | 10 (X925) | scalar | 222.2 | 119.2 | — | — |
| | 10 | dotprod | 636.8 | 198.6 | 2.87× | 1.67× |
| | 10 | **i8mm** | **935.3** | **199.9** | **4.21×** | **1.68×** |
| | 10 | llama.cpp | 524.5 | 194.2 | — | — |
| | 20 | scalar | 251.0 | 117.1 | — | — |
| | 20 | dotprod | 798.5 | 204.1 | 3.18× | 1.74× |
| | 20 | **i8mm** | **1026.6** | **209.6** | **4.09×** | **1.79×** |
| | 20 | llama.cpp | 633.5 | 135.2 | — | — |

**This PR alone** (the same binary with `PEREGRINE_ISA=neon-dotprod`, so only the GEMM kernel
changes): prefill **+40.5 %** (Llama, 10 threads), **+26.0 %** (Llama, 20), **+46.9 %** (Qwen, 10),
**+29.2 %** (Qwen, 20); decode flat (0.5–2.4 %, inside the run-to-run spread). That is the format
mix: Qwen's Q5_0 tile was the slowest `sdot` GEMM and gains the most; decode is bandwidth-bound and
reads the same bytes either way.

Peregrine ÷ llama.cpp: Llama 2.45× / 1.96× prefill and 1.14× / 1.29× decode at 10 / 20 threads;
Qwen 1.78× / 1.62× and 1.03× / 1.55×.

## Bit identity

The integer sums are exact, and the fold is `kq::dot_*_ref`'s expression in its order, so the
`smmla` GEMM is bit-identical to the `sdot` GEMV of the same tier. That is tested per format
(`q4k8_i8mm_gemm_is_bit_identical_to_the_sdot_gemv`,
`q5_08_…`, `q6k8_…`) at widths 4/8/12, and end to end: the `aarch64-neon-i8mm` goldens are
byte-identical to `aarch64-neon-dotprod` on all 51 synthetic models and both real models, apart
from the `tier` header line. `PEREGRINE_ISA=neon-dotprod`, `=neon` and `=scalar` all still pass
`PEREGRINE_GOLDEN=require`.

## Files

| file | contents |
|---|---|
| `llama-1b-t10.csv`, `llama-1b-t20.csv` | scalar / dotprod / i8mm / llama.cpp, 5 interleaved rounds |
| `qwen-05b-t10.csv`, `qwen-05b-t20.csv` | the same for the small model |
| `microbench.txt` | three `neonbench` runs, the ten tile-GEMM lines |
