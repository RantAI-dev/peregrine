# S4.9v: the float matrix product at small row counts, the account (2026-11-13)

S4.9u left twelve cells behind for nomic-bert and jina-bert-v2 and named one cause:
"the 128-token cells need a small-M tile in `fpack`" — the 128-token pass is 89–90 % the
matrix product, and Peregrine's GEMM runs at 180–181 GMAC/s there where ONNX Runtime's MLAS
reaches 202–206. A tile worth 9–10 ms a pass would be +10–11 % at 128 tokens. This step
builds the instrument that claim needed, times the kernel and MLAS against the row count on
the same shapes, and reports what the numbers say.

**The short answer: the tile is not the deficit.** With the engine's own call shape — units
of two panels over a four-thread pool, the strided output — the 6-row × 16-column tile runs at
**199–228 GMAC/s at 128 rows** over the eight shapes (median of three sweeps), which is
**MLAS's 208–222 within −1 % to +8 %**: level on the narrow matrices and ahead on the wide
ones. Our kernel falls behind MLAS only at 512 rows (**0.85–0.99×**). The engine's *pass* runs
the same mix at **180** by its own counters, 14 % below its own isolated kernel, and that gap
is the per-call work around it — **no engine file changed**, because the changes the account
named either did not pay end to end (unit granularity) or change the bits (a wider
accumulator tile).
Numbers, raw sweeps and the ONNX Runtime script: this directory.

## Machine, build and host state

| | |
|---|---|
| machine | bluefin: AMD Ryzen 3 PRO 4350G (Renoir, Zen 2), 4 cores / 8 threads, 128 KiB L1d + 512 KiB L2 per core, 4 MiB L3, 13 GB, Fedora 44, kernel 7.2.7-200.fc44, governor `powersave`, tier `x86_64-avx2` |
| roof | the register-only FMA chain, measured in every run: **64.2 GMAC/s one thread, 256.8–257.0 four** (S4.9p: 64.9 / 259.6) |
| build | this branch on `32c222e` (S4.9u's merge of `main`), `cargo build --release`; instrument `crates/peregrine-bench/examples/mgemm.rs`, `mlas.py`, `summarize.py` |
| pinning | `taskset -c 0-3` (four threads), `taskset -c 0` (one) |
| host | `/proc/loadavg` 2.4–3.2 through the three sweeps (`/proc/loadavg` is host-wide in this PID namespace; the CPU was the measurement's), swap 7.4 GB of 8 GB in use, the condition S4.9l–S4.9u recorded. Three whole sweeps per thread count, 15 calls per cell, the minimum of the 15, the median across the sweeps. |

## What is measured, exactly

**Peregrine** (`mgemm`): the packed float GEMM as the engine calls it — panels of 16 output
rows, cut into **units of two** (`blocks::linear::FP_UNIT_PANELS`), shared out dynamically
over a persistent pool, each unit one `fpack::gemm_strided` writing the projection's
row-major output (`out_stride = n_out`). **The caller is one of the workers** (`threads`
threads, not `threads + 1`) and the pool spins instead of parking, as the engine's does: a
futex-waiting pool measured 84 GMAC/s where this one measures 175 at 16 rows, which is the
park latency, not the kernel. Activations are `QuantAct`s with `keep_f32` on — the engine's
float path. The timed window is one whole call.

**MLAS** (`mlas.py`): one `MatMul` node, activation `[M, K]` × weight `[K, N]` with the
weight a graph initializer, so ONNX Runtime 1.31.1 packs it once at session creation — the
state `fpack` is in. `intra_op_num_threads` 4 or 1, `ORT_ENABLE_ALL`, three warm-ups, median
of five calls, three runs. It measures **MLAS's packed f32 GEMM through ONNX Runtime's own
graph**; a single node leaves nothing else in it.

## The rate against the row count

Four threads, GMAC/s, F16 panels (the widened f32 panel where the kernel takes it — what the
engine runs), median of three sweeps:

| rows | 384×384 | 384×1536 | 768×768 | 768×3072 | 3072×768 | 2304×768 | 384×3072 | 768×1536 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 140.1 | 168.4 | 162.5 | 186.5 | 185.8 | 187.2 | 185.9 | 188.0 |
| 32 | 146.7 | 195.5 | 198.4 | 192.4 | 186.3 | 194.4 | 194.5 | 196.7 |
| 64 | 170.6 | 223.1 | 225.6 | 220.3 | 188.6 | 205.1 | 221.9 | 224.2 |
| 96 | 234.6 | 234.6 | 224.3 | 212.9 | 204.6 | 218.1 | 232.6 | 221.1 |
| 128 | 228.0 | 226.9 | 223.0 | 213.4 | 199.0 | 217.0 | 224.9 | 216.9 |
| 192 | 226.6 | 225.3 | 226.1 | 212.9 | 202.5 | 220.2 | 217.6 | 222.9 |
| 256 | 225.4 | 222.5 | 221.2 | 206.4 | 196.3 | 215.0 | 204.7 | 214.8 |
| 512 | 224.9 | 198.9 | 206.7 | 202.9 | 185.7 | 190.6 | 185.0 | 202.5 |

One thread, GMAC/s, same cells:

| rows | 384×384 | 384×1536 | 768×768 | 768×3072 | 3072×768 | 2304×768 | 384×3072 | 768×1536 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 47.5 | 47.0 | 47.5 | 46.6 | 46.7 | 47.2 | 47.2 | 47.2 |
| 32 | 49.3 | 49.2 | 50.2 | 48.2 | 48.5 | 49.0 | 49.1 | 50.0 |
| 64 | 56.2 | 56.5 | 57.2 | 55.7 | 48.2 | 52.3 | 55.9 | 56.5 |
| 96 | 59.1 | 58.2 | 56.2 | 54.8 | 53.5 | 56.3 | 58.5 | 55.7 |
| 128 | 57.3 | 57.0 | 55.5 | 54.8 | 52.0 | 56.2 | 57.0 | 55.3 |
| 192 | 56.9 | 56.7 | 56.8 | 56.3 | 52.5 | 57.5 | 56.2 | 56.5 |
| 256 | 56.5 | 56.6 | 56.2 | 55.5 | 51.7 | 56.2 | 56.3 | 55.9 |
| 512 | 56.8 | 55.8 | 55.5 | 52.9 | 47.2 | 49.9 | 53.2 | 54.3 |

MLAS through ONNX Runtime, four threads, GMAC/s (the two transposed shapes were not measured
for MLAS; that is why the ratio table below has six columns):

| rows | 384×384 | 384×1536 | 768×768 | 768×3072 | 384×3072 | 768×1536 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 127.3 | 136.6 | 189.5 | 114.8 | 118.2 | 156.7 |
| 32 | 165.9 | 155.6 | 202.3 | 154.3 | 178.5 | 179.2 |
| 64 | 201.7 | 221.8 | 221.0 | 194.8 | 201.5 | 199.9 |
| 96 | 216.3 | 220.7 | 218.8 | 211.3 | 208.9 | 208.5 |
| 128 | 218.6 | 221.9 | 222.0 | 215.2 | 208.1 | 210.2 |
| 192 | 230.0 | 229.7 | 228.0 | 225.2 | 219.3 | 222.5 |
| 256 | 231.9 | 225.9 | 228.3 | 228.0 | 221.3 | 225.7 |
| 512 | 228.2 | 218.9 | 225.4 | 207.5 | 216.9 | 208.4 |

Peregrine ÷ MLAS, four threads:

| rows | 384×384 | 384×1536 | 768×768 | 768×3072 | 384×3072 | 768×1536 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.10 | 1.23 | 0.86 | 1.62 | 1.57 | 1.20 |
| 32 | 0.88 | 1.26 | 0.98 | 1.25 | 1.09 | 1.10 |
| 64 | 0.85 | 1.01 | 1.02 | 1.13 | 1.10 | 1.12 |
| 96 | 1.08 | 1.06 | 1.03 | 1.01 | 1.11 | 1.06 |
| 128 | 1.04 | 1.02 | 1.00 | 0.99 | 1.08 | 1.03 |
| 192 | 0.99 | 0.98 | 0.99 | 0.95 | 0.99 | 1.00 |
| 256 | 0.97 | 0.98 | 0.97 | 0.91 | 0.92 | 0.95 |
| 512 | 0.99 | 0.91 | 0.92 | 0.98 | 0.85 | 0.97 |

MLAS one thread is 40–62 GMAC/s at every cell, i.e. it scales more evenly over M below 64
rows than Peregrine's one-thread numbers (47–58) do.

## Where the rate falls off, and why

**At 128 rows the tile is at MLAS's rate: 199–228 GMAC/s, 78–89 % of the 256.8 roof,
0.99–1.08× MLAS.** Peregrine is ahead on the wide matrices (1.08× on 384 × 3 072, 1.03× on
768 × 1 536, 1.04× on 384 × 384) and level on the narrow ones (1.00 on 768 × 768, 0.99 on
768 × 3 072). **The crossing is at 512 rows**, where MLAS is 1–15 % ahead (0.85–0.99×) — the
opposite of the small-M story S4.9u assumed.

**Below ~64 rows the rate is the weight stream, not the tile.** A unit reads its two panels
once per call, so the arithmetic intensity is `M/2` MAC per weight byte: 8 MAC/byte at
`M = 16`, where 256 GMAC/s would need 32 GB/s from L2/L3, and the measured 162.5 GMAC/s on
768 × 768 is 20 GB/s — that stream, at 63 % of the roof. At `M = 128` the same panel buys
64 MAC/byte, the stream is 4 GB/s, and the tile is FMA-bound. That is the whole shape of the
curve, and it is why the 16-token batch cells are a pass-width question (S4.8b's 128-row
pass already answers it) rather than a tile one.

**Peregrine's pass is 14 % below its own kernel, at every M.** nomic F16 at one 128-token
text: `StepStats.matmul` 80.6 ms for 14.50 GMAC = **180 GMAC/s**, where the same six shapes at
128 rows are 199.0–228.0 isolated, MAC-weighted **213**. At 512 tokens the pass is 179 and the
isolated cells 185.0–224.9 (weighted 202), a 12 % gap. Flat with M, so not a small-M effect.
What is inside that bucket, measured one at a time:

- **The activation copy** (`linear_rows`' `QuantAct::fill`; `quantize` is off on a float-only
  model, so it is the f32 copy): 0.062 ms for 128 × 384, 0.871 ms for 128 × 3 072 — about
  **1 ms of the 80 ms bucket**, 1.2 %.
- **The unit granularity.** `FP_UNIT_PANELS` 2 → 4: +1 % to +6 % on the kernel at 128 rows
  (384 × 1 536: 205.3 → 218.2 in the barrier-pool sweep) and **neutral end to end** (nomic
  128 b1: 80.3–81.2 ms of matmul against 79.5–80.6, 11.07 texts/s against 11.08, three runs
  each). Reverted.
- **The store layout.** A contiguous private block against the engine's `out_stride = n_out`
  store: the same rate (768 × 768, 128 rows: 212.0 both).
- **The row-block split.** `row_block(3 072) = 21`, so a 128-row call streams its weights in
  seven blocks; the shape that pays it, 3 072 × 768, is 199.0 at 128 rows — level with the
  shapes that do not (199–228). Not the 14 %.
- **What is left is the dispatch itself**: 48 GEMM calls, 3 456 two-panel units and 289
  barriers per 128-row pass, and `PEREGRINE_PROF2` puts every worker's time inside the float
  path (`fmt other` 314 ms over four workers for a 79.5 ms pass, `barrier wait` 1.1 ms). A
  `peregrine-core` scheduling question — fusing a layer's projections into fewer, wider
  calls — not a kernel one. That is the next step this account names, and it is same-bits work.

## The same-bits bound

`fpack` defines each output element as **one ascending FMA chain over `k` in a single f32
accumulator initialised to the bias** (zero when there is none): `gemm_scalar_strided` is the
definition, and the AVX2, AVX-512 and NEON bodies are held to it by the reduced-precision
tests (row counts 1–40 and 127/128/129/511/512/513 at ten widths, plus NaN and subnormals).
A row's bits are fixed by that chain and nothing else.

Admissible — any schedule that keeps each output's chain:

- **rows per tile** 1…6 with 16 columns (two `ymm` per row: 12 accumulators), or fewer
  columns in exchange for more rows, as long as the accumulators stay ≤ 12;
- **panels of 16 output rows** (a packing layout, not an arithmetic one);
- **row blocks** of any size, **units** of any number of panels, and **any order** of panels
  and blocks — a row's chain does not know which block or worker computed it (pinned by the
  bit-identity tests since S4.9f);
- **a k-slab decomposition with an exact f32 spill** between slabs: `Σ 0..K1` then `K1..K2`
  in one f32 accumulator is the same sequence of FMAs, so both weights and activations could
  stream once per call.

Not admissible: **more than one accumulator per output** summed at the end (split-k by
halves, pairwise accumulation, a changed FMA order); and **8 rows × 16 columns**, which needs
16 accumulators plus two weight registers plus the activation broadcast = 19 `ymm` against
16 — the wider tile *requires* exactly that reassociation. That is the option the bit rule
forbids, and what a library free to change bits would buy: the ceiling at 128 rows is the
256.8 GMAC/s roof against the 199–228 measured, i.e. **+13–29 % on the tile** and about
+10–11 % on a 128-token pass, the figure S4.9u estimated.

## What this changes in the standings

Nothing. No engine file is in this branch, so every standings cell in `docs/benchmark.md`
stands as the S4.9u re-read left it: **24 of the 36 cells these families add, 79 of 104 with
the existing table, twelve behind** — nomic F16 128 b32 (0.929×), jina F16 16 b8/b32
(0.938/0.964×) and 128 b1/b8/b32 (0.911/0.917/0.910×), jina Q8_0's five. The account moves
the next step: the pass's 14 % over its own kernel is `peregrine-core`'s per-layer dispatch,
not `fpack`'s tile, and it is worth the ~9 ms per 128-token pass the S4.9u account attributed
to the GEMM.

## Reproducing

```sh
cargo build --release -p peregrine-bench --example mgemm
for s in 1 2 3; do
  taskset -c 0-3 ./target/release/examples/mgemm --reps 15 > peregrine-4t-sweep$s.txt
  taskset -c 0   ./target/release/examples/mgemm --threads 1 --reps 15 > peregrine-1t-sweep$s.txt
done
for t in 1 4; do for r in 1 2 3; do
  taskset -c 0-3 /var/tmp/embed-venv/bin/python mlas.py $t > mlas-${t}t-run$r.txt
done; done
python3 summarize.py
# the engine's own number for the same pass
cargo build --release -p peregrine-bench --example cellprof \
  --features peregrine-models/arch-nomic-bert,peregrine-models/arch-jina-bert
taskset -c 0-3 ./target/release/examples/cellprof ~/models/embed/nomic/nomic-f16.gguf \
  /var/tmp/embed-texts-s49u/nomic.json 128 1 4
```
