# S4.9w: an encoder pass's fork-joins, the dispatch account (2026-11-13)

S4.9v left one sentence for the next step: the 128-token encoder pass runs at 180 GMAC/s
against its own isolated kernel's 199–228, a flat 12–14 % at every row count, and "what is
left is the dispatch: 48 GEMM calls, 3 456 two-panel units and 289 barriers per 128-row
pass". This step takes that pass apart on the build that has S5.4j's gap-following spin
window and S5.6's stall gate, finds the dispatch's *fork-joins* are not the loss, finds the
loss one level down — the **row block each unit reads its weights in** — and removes it with
the same bits.

**The short answer.** The old `fp_row_block` capped a block at half of L2. At 128 activation
rows that cut a 768-wide matrix into two blocks and a 3 072-wide one into seven, so the down
projection's 4.7 MB of weights were streamed seven times a layer — **6.9 ms of the 12.2 ms
matmul gap**. Removing the cap (a row block is a schedule; every output keeps its ascending
`k` chain) moves nomic F16 at one 128-token text from 10.96 to 11.93 texts/s and jina from
11.10 to 12.12 (**+8.2–9.6 %**), 512 tokens **+5.6–6.7 %**, 2 000 tokens **+4.3–5.8 %**, and
flips **six of the twelve** standings cells that were behind. The other half of the gap
(5.3 ms) is the one pass over 212 MB of weights that the isolated `mgemm` benchmark
amortises over fifteen calls. The 289 fork-joins cost 1.5 ms of wait (1.8 %), so fewer
fork-joins would not have paid; the next real lever for the remaining cells is the
bit-changing 8 × 16 tile, not a schedule.

## Part 0 — the two pool changes on this host

**Builds.** BEFORE = `5d603a48` (`main` with #262, before S5.4j and S5.6); AFTER =
`23d86c8` (`origin/s5.6-pool-contended`, both merged, with S4.9v's `mgemm`). Five
interleaved rounds, BEFORE → AFTER → AFTER with `PEREGRINE_ADAPTIVE_STALL=0`; medians.
Cache off (`PEREGRINE_EMBED_CACHE_MB=0`), four threads, fresh different texts inside every
batch and per timed request (the S4.9u texts, `benchmarks/embed.py --budget 0.5
--max-requests 20 --max-big-requests 2`).

**Host.** bluefin, Ryzen 3 PRO 4350G (Zen 2, AVX2), 4 cores / 8 threads, 13 GB; load average
2.1–5.0 through the windows, **CPU idle 49.7–54.8 % of every round** (the measurement's own
four threads on eight logical CPUs, `/proc/stat` deltas beside every round in
`parts/part0/r*-cpu.txt`). Swap the usual 7.4 GB of 8.

**Embeddings, texts/s, AFTER ÷ BEFORE** (spreads are the five rounds; every deficit is inside
the spread — no cell is more than its spread below 1.0):

| cell | BEFORE | AFTER | AFTER/BEFORE | AFTER, stall gate off | off/BEFORE |
|---|---:|---:|---:|---:|---:|
| nomic F16 L16 b8 | 80.84 | 84.32 | 1.043 | 84.46 | 1.045 |
| nomic F16 L16 b32 | 86.14 | 85.66 | 0.994 | 84.95 | 0.986 |
| nomic F16 L128 b1 | 10.73 | 10.64 | 0.992 | 10.51 | 0.980 |
| nomic F16 L128 b8 | 10.73 | 10.73 | 1.000 | 10.76 | 1.003 |
| nomic F16 L128 b32 | 10.59 | 10.65 | 1.006 | 10.74 | 1.014 |
| nomic F16 L512 b8 | 2.466 | 2.496 | 1.012 | 2.488 | 1.009 |
| jina F16 L16 b8 | 86.77 | 85.60 | 0.986 | 86.44 | 0.996 |
| jina F16 L16 b32 | 88.20 | 87.29 | 0.990 | 87.55 | 0.993 |
| jina F16 L128 b1 | 10.97 | 10.87 | 0.991 | 10.95 | 0.998 |
| jina F16 L128 b8 | 11.07 | 11.03 | 0.997 | 11.04 | 0.998 |
| jina F16 L128 b32 | 11.02 | 10.97 | 0.995 | 11.05 | 1.003 |
| jina F16 L512 b8 | 2.590 | 2.587 | 0.999 | 2.582 | 0.997 |
| bge-small F32 L128 b32 | 50.05 | 49.59 | 0.991 | 49.80 | 0.995 |
| MiniLM F16 L128 b32 | 101.98 | 100.15 | 0.982 | 100.51 | 0.986 |
| Qwen3-Embedding 0.6B F16 L512 b1 | 0.574 | 0.569 | 0.991 | 0.569 | 0.991 |

**Generation, Llama 3.2 1B Q4_K_M**, 512-token prompt, 128 decoded tokens, medians:

| cell | BEFORE | AFTER | AFTER/BEFORE | stall gate off | off/BEFORE |
|---|---:|---:|---:|---:|---:|
| prompt tok/s | 220.16 | 219.58 | 0.997 | 220.81 | 1.003 |
| decode tok/s | 46.46 | 46.37 | 0.998 | 45.68 | 0.983 |
| ten-user aggregate tok/s | 38.78 | 38.47 | 0.992 | 38.36 | 0.989 |

The ten-user decode's off arm has one 42.7 tok/s round (the quiet arm's rounds are 45.4–46.5);
its median is 45.68 and its deficit is inside that spread. **No cell is more than its spread
below 1.0**, and the `PEREGRINE_ADAPTIVE_STALL=0` arm is within 2 % of both, so on this host
neither the gap window nor the stall gate costs a cell; the measured join accounting agrees
(AFTER, one 128-token pass: 289 joins, caller's post-claim wait **1.2–1.5 ms, 1.5–1.8 %**,
parks: none inside the pass; `PEREGRINE_ADAPTIVE_LOG` shows level 0 throughout, 0 transitions,
three ≥5 ms park durations per process at the idle exit). This host is not the loaded one
either change was measured on, and the two are level here.

## Part 1 — an encoder pass's dispatch, taken apart

`cellprof` with `PEREGRINE_PROF2=2 PEREGRINE_ADAPTIVE_LOG=1`, one 128-token text and thirty-two
(nomic and jina F16, four threads). The join totals include the warm-up pass; the per-pass
column divides by two (b1) or thirty-three (b32).

| | nomic b1 | nomic b32 | jina b1 | jina b32 |
|---|---:|---:|---:|---:|
| wall, ms / pass | 82.1 | 82.9 | 81.5 | 81.8 |
| matmul (`StepStats`), ms | 73.4 | 74.1 | 73.0 | 73.5 |
| attention, ms | 3.12 | 3.18 | 3.36 | 3.32 |
| element-wise + serial (`other`), ms | 5.60 | 5.57 | 5.09 | 5.00 |
| fork-joins / pass | 289 | 289 | 277 | 277 |
| GEMM calls / pass | 48 | 48 | 48 | 48 |
| two-panel units / pass | 3 744 | 3 744 | 3 456 | 3 456 |
| caller's post-claim wait, ms / pass | 1.5 | 1.5 | 2.0 | 1.5 |
| caller's share, ms / pass | 78.5 | 79.1 | 79.5 | 81.6 |
| parks inside the pass | 0 | 0 | 0 | 0 |

**The 12–14 % split.** The old pass's matmul bucket was 80.3 ms for 14.50 GMAC = 180 GMAC/s
(nomic b1, S4.9v). The isolated kernel on the same six shapes at 128 rows is 199.0–228.0,
MAC-weighted **213 GMAC/s = 68.1 ms**. The 12.2 ms gap is:

| part | ms | how measured |
|---|---:|---|
| weight re-read by the row block | **6.9** | A/B, same build, `fp_row_block` capped vs whole pass: matmul 80.3 → 73.4 |
| one pass over 212 MB of weights | **5.3** | 73.4 − 68.1; 212 MB at 39 GB/s = 5.4 ms, the stream `mgemm`'s 15 calls amortise |
| fork-join dispatch | **0** of the matmul gap | the caller waits 1.5 ms, and that wait is in the pass's wall, not in the matmul bucket |

They add to 12.2 ms. The pass's total over the isolated 68.1 is 21.6 ms: the 12.2 above,
attention 3.2 (a join of its own), and 6.2 of element-wise joins + the serial activation
copy (~1.0 ms; `QuantAct::fill` is still the caller's, S4.9v measured 0.062 ms at 128 × 384
and 0.871 at 128 × 3 072), pooling and the loop between joins. **The rows are balanced**:
the caller's wait is 1.5–1.8 % of the pass, the last-worker histogram is even (0:162 1:90
2:165 3:161 over the warm-up and cell), and the straggler gap averages 3 µs (max 0.28 ms).
"Fork-joins are the dispatch cost" does not survive the measurement; the units' *weight
reads* do.

## Part 2 — the lever the account supports, and the ones it does not

**Built: the float path's row block is the pass, not half of L2** (commit 1). `fp_row_block`
now cuts a block only to fill the pool; it no longer caps at
`peregrine_kernels::fpack::row_block` (half of L2). At 128 rows a 3 072-input matrix was seven
blocks, its 4.7 MB of weights read seven times a layer; the pass's activation (up to
512 × 3 072 × 4 = 6 MB) fits L3, where the weights never do. A/B through `cellprof`, three
interleaved rounds, medians, same build:

| cell | capped (old) | whole pass | ratio | matmul old → new, ms |
|---|---:|---:|---:|---:|
| nomic L128 b1 | 10.964 | **11.925** | 1.088 | 81.6 → 75.0 |
| nomic L128 b8 | 10.876 | **11.921** | 1.096 | 657.3 → 600.4 |
| nomic L128 b32 | 10.915 | **11.917** | 1.092 | 2 618.5 → 2 403.0 |
| nomic L512 b1 | 2.489 | **2.645** | 1.063 | 329.2 → 305.0 |
| nomic L512 b8 | 2.532 | **2.701** | 1.067 | 2 599.2 → 2 399.5 |
| nomic L512 b32 | 2.519 | **2.680** | 1.064 | 10 452.0 → 9 664.2 |
| nomic L2000 b1 | 0.497 | **0.526** | 1.058 | 1 265.2 → 1 164.4 |
| jina L128 b1 | 11.098 | **12.120** | 1.092 | 81.0 → 74.2 |
| jina L128 b8 | 11.037 | **12.078** | 1.094 | 650.9 → 595.7 |
| jina L128 b32 | 11.112 | **12.024** | 1.082 | 2 597.5 → 2 396.7 |
| jina L512 b1 | 2.543 | **2.685** | 1.056 | 328.8 → 307.8 |
| jina L512 b8 | 2.580 | **2.746** | 1.064 | 2 596.4 → 2 411.4 |
| jina L512 b32 | 2.581 | **2.742** | 1.062 | 10 380.2 → 9 660.3 |
| jina L2000 b1 | 0.491 | **0.512** | 1.043 | 1 255.3 → 1 170.6 |

Every output keeps its one ascending `k` chain; a row block is a schedule, not an
arithmetic. No golden moved, and the row is the same at any pass width, batch or thread
count.

**Dropped: a 256-row pass (the float path's block cap at `MAX_BATCH`)**. With the 128-row
cap gone, a 256-row pass reads a batch's weights once for two 128-token texts. Measured with
the same `cellprof` build (3 rounds): nomic F16 L128 b8 **11.71 vs 12.04** (0.973), L128 b32
**11.75 vs 12.04** (0.976), L512 b8 2.714 vs 2.724 (0.996), L512 b32 2.713 vs 2.725 (0.996).
The 3 MB activation working set costs more in L3 than the halved weight stream saves, so the
lever was reverted, not shipped.

**Dropped: fewer fork-joins per layer.** Nomic's q/k/v projection is one matrix and jina's
three matrices are already one `linear_rows_shared` group (one join); both FFN halves are
already one group. What remains is 24 joins a layer, 20 of them element-wise, and the whole
pass's caller wait is **1.5 ms (1.8 %)**. Removing the two joins a layer still available
(nomic's q/k/v split and query-scale passes could fold into `prepare_rows`) would remove
24 of 289 joins, ≤0.15 ms by the measured per-join wait — under 0.2 % of a pass. Not built;
the account says the joins are not the cost.

**Dropped: larger unit claims.** One atomic claim per two-panel unit is 3 456–3 744 claims a
pass. At ~50 ns an atomic that is <0.02 % of the pass, and S4.9v's `FP_UNIT_PANELS` 2 → 4
test was already neutral end to end. Not built.

## Part 3 — the standings

Three interleaved rounds, the S4.9u method (`--budget 0.5 --max-requests 20
--max-big-requests 2`, cache off, fresh texts), medians; the best equal-accuracy engine on
the same cell (ONNX Runtime 1.31 fp32, fastembed 0.9 fp32, OpenVINO 2026.4 fp32; nomic F16
and Q8_0 use them because no int8 row clears the 0.999 accuracy floor). "before" is the
S4.9u-merge re-read.

**nomic-embed-text v1.5 F16**

| cell | before | now | Peregrine | best other | ratio |
|---|---:|---:|---:|---|---:|
| L16 b8 | 1.019× | **1.126×** | 92.17 | ONNX Runtime fp32 81.84 | 1.126 |
| L16 b32 | 1.088× | **1.086×** | 92.83 | ONNX Runtime fp32 85.48 | 1.086 |
| L128 b1 | 0.929× | **1.028×** | 11.40 | ONNX Runtime fp32 11.09 | 1.028 |
| L128 b8 | 1.002× | **1.034×** | 11.81 | fastembed fp32 11.42 | 1.034 |
| L128 b32 | 0.929× | **1.019×** | 11.69 | fastembed fp32 11.47 | 1.019 |
| L512 b8 | 1.090× | **1.160×** | 2.680 | OpenVINO fp32 2.310 | 1.160 |
| L512 b32 | 1.102× | **1.191×** | 2.680 | fastembed fp32 2.250 | 1.191 |
| mixed b32 | 1.717× | **1.804×** | 12.81 | fastembed fp32 7.10 | 1.804 |

**jina-embeddings-v2-base-en F16**

| cell | before | now | Peregrine | best other | ratio |
|---|---:|---:|---:|---|---:|
| L16 b8 | 0.938× | **1.014×** | 93.57 | ONNX Runtime fp32 92.24 | 1.014 |
| L16 b32 | 0.964× | **1.045×** | 95.51 | ONNX Runtime fp32 91.43 | 1.045 |
| L128 b1 | 0.911× | **1.011×** | 11.87 | fastembed fp32 11.74 | 1.011 |
| L128 b8 | 0.917× | **1.033×** | 12.00 | fastembed fp32 11.62 | 1.033 |
| L128 b32 | 0.910× | 0.993× | 12.05 | fastembed fp32 12.13 | 0.993 |
| L512 b8 | 1.002× | **1.066×** | 2.730 | OpenVINO fp32 2.560 | 1.066 |
| L512 b32 | 1.042× | **1.087×** | 2.740 | OpenVINO fp32 2.520 | 1.087 |
| mixed b32 | 1.692× | **1.780×** | 13.19 | fastembed fp32 7.41 | 1.780 |

**jina Q8_0** (not touched by this change — it runs the Q8_0 eight-row tile, not the packed
float path):

| cell | before | now | Peregrine | best other | ratio |
|---|---:|---:|---:|---|---:|
| L16 b8 | 0.945× | 0.922× | 85.07 | ONNX Runtime fp32 92.24 | 0.922 |
| L16 b32 | 0.974× | 0.945× | 86.38 | ONNX Runtime fp32 91.43 | 0.945 |
| L128 b1 | 0.916× | 0.946× | 11.11 | fastembed fp32 11.74 | 0.946 |
| L128 b8 | 0.903× | 0.961× | 11.17 | fastembed fp32 11.62 | 0.961 |
| L128 b32 | 0.918× | 0.917× | 11.12 | fastembed fp32 12.13 | 0.917 |
| L512 b8 | — | **1.008×** | 2.580 | OpenVINO fp32 2.560 | 1.008 |
| L512 b32 | — | **1.008×** | 2.540 | OpenVINO fp32 2.520 | 1.008 |
| mixed b32 | — | **1.690×** | 12.52 | fastembed fp32 7.41 | 1.690 |

The 16 b1, 512 b1 and 2 000-token cells were not re-run (the change moves the same pass
+4.3–5.8 % in the `cellprof` control and every one of them was already ahead; the 2 000
cells stay ahead by more). Nomic's Q8_0 row is listed without ratios (0.9983 against the
fp32 reference, below the 0.999 class floor).

**The count.** nomic F16 **12 of 12**, jina F16 **11 of 12**, jina Q8_0 **7 of 12** — **30 of
the 36 cells these two models add** (24 before), i.e. **85 of 104** on the table's rule
(79 before). Six of the twelve cells that were behind flipped: nomic F16 128 b1/b32, jina F16
16 b8/b32 and 128 b1/b8. Six remain, and each is named:

- **jina F16 L128 b32, 0.993× (12.05 against fastembed's 12.13).** Five rounds put it at
  12.02–12.13 against 11.88–12.43; the pass is 0.6 % of wall behind a competitor whose
  `MatMul` node is at MLAS's 128-row rate. It is the same weight-stream remainder the account
  names, not a join.
- **jina Q8_0 16 b8/b32 and 128 b1/b8/b32, 0.917–0.962×.** This file does not take the
  float pack (`q8_panels()` is `Keep` for jina-bert-v2 on x86); it runs the Q8_0 eight-row
  tile, whose block stays at the desktop's k-quant `matmul_group`, outside this PR's files.

**What the bit-changing tile would be worth.** S4.9v's §26 ceiling: 8 rows × 16 columns
needs 19 `ymm` against 16 and therefore a second accumulator per output, which changes the
summation order. At the post-fix 73.4 ms matmul it is worth **+13–29 % on the tile**, i.e.
64.9–56.7 ms, a pass of 73.6–65.4 ms against 82.1 — **+11.5 % to +25 % texts/s at 128
tokens** (the conservative S4.9u estimate was +10–11 %). That would put jina F16 L128 b32 at
~1.10× and is the only lever that clears it; it would not touch the Q8_0 row, which runs a
different kernel. Not built: the bits are `fpack`'s definition.

## Machine, build and host

| | |
|---|---|
| machine | bluefin: AMD Ryzen 3 PRO 4350G (Renoir, Zen 2), 4 cores / 8 threads, 512 KiB L2 per core, 4 MiB L3, 13 GB, Fedora 44, kernel 7.2.7-200.fc44, governor `powersave`, tier `x86_64-avx2` |
| flags | `fma avx avx2`, 4 threads, cache off (`PEREGRINE_EMBED_CACHE_MB=0`) |
| BEFORE | `5d603a48` (`main` with #262) |
| AFTER | `23d86c8` (`origin/s5.6-pool-contended`) |
| this PR | `s4.9w-encoder-dispatch`, one engine commit on the AFTER tree |
| host | load average 2.1–5.0, **CPU idle 49.7–54.8 % of every Part 0 round**, steal not recorded separately; swap 7.4 GB of 8 GB in use (the condition S4.9l–S4.9v recorded) |

## Reproducing

```sh
# Part 0: two binaries
git checkout --detach 5d603a48 && cargo build --release -p peregrine
cp target/release/peregrine /tmp/opencode/bin/peregrine-before
git checkout --detach 23d86c8 && cargo build --release -p peregrine
cp target/release/peregrine /tmp/opencode/bin/peregrine-after
for r in 1 2 3 4 5; do for a in before after after_nostall; do
  benchmarks/results/bluefin-2026-11-13-s4.9w/part0.sh $r $a
  benchmarks/results/bluefin-2026-11-13-s4.9w/part0-gen.sh $r $a
done; done
python3 benchmarks/results/bluefin-2026-11-13-s4.9w/summarize0.py
python3 benchmarks/results/bluefin-2026-11-13-s4.9w/summarize0gen.py

# Part 1
cargo build --release -p peregrine-bench --example cellprof \
  --features peregrine-models/arch-nomic-bert,peregrine-models/arch-jina-bert
for m in nomic jina; do for b in 1 32; do
  PEREGRINE_PROF2=2 PEREGRINE_ADAPTIVE_LOG=1 PEREGRINE_EMBED_CACHE_MB=0 \
    taskset -c 0-3 ./target/release/examples/cellprof \
      ~/models/embed/$m/$m-f16.gguf /var/tmp/embed-texts-s49u/$m.json 128 $b 4
done; done

# Part 2: the row-block A/B
benchmarks/results/bluefin-2026-11-13-s4.9w/rowblock-ab.sh <round> orig|noblock nomic|jina
python3 benchmarks/results/bluefin-2026-11-13-s4.9w/summarize2.py

# Part 3
for r in 1 2 3; do for a in peregrine competitors; do
  benchmarks/results/bluefin-2026-11-13-s4.9w/part3.sh $r $a
  benchmarks/results/bluefin-2026-11-13-s4.9w/part3-b1.sh $r $a
done; done
python3 benchmarks/results/bluefin-2026-11-13-s4.9w/summarize3.py
```

Raw files: `parts/part0/` (JSON per arm/round, one per model, and the `/proc/stat` deltas),
`parts/part0-gen/` (the generation run sets), `parts/part1/` (`cellprof` stdout and the
join/adaptive stderr), `parts/rowblock/` and `parts/pass256/` (the lever JSONL), and
`parts/part3/` (the harness logs, one per row and round — the raw record
`summarize3.py` reads; the runner also writes a per-cell JSON, and the logs carry the same
numbers plus the spread, so only the logs were retained for this report).
