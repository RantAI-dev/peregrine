# Benchmarks: Peregrine against llama.cpp

Where Peregrine stands against llama.cpp on the same machine, the same model file and the same thread
count. Wins and losses are both here. The raw data, method and machine records are in
[`benchmarks/results/`](../benchmarks/results/); the targets these numbers are held to are in
[targets.md](targets.md).

![Where Peregrine is fastest](img/bench-wins.svg)

## The short version

- **Single-user decode is Peregrine's strongest result, and it now wins every completed
  throughput row.** On the 12-core desktop the decode lead runs from 1.13× to 1.69× against
  llama.cpp's faster flash-attention setting: Llama 3.2 1B 1.31× at 512 and 1.60× at 4096, Qwen 2.5
  0.5B 1.37× and 1.57×, SmolLM2 360M 1.34× and 1.69×, gemma-3-1b-it 1.35× and 1.13×. On the 4-core
  bluefin all three models are ahead, from 1.07× to 1.34×. The gap grows with context: attention
  reads each cached key and value once per group of heads.
- **Many users on a small machine, and now on the desktop too.** At 512 tokens with 10 users the
  desktop serves 1.28-1.43× llama.cpp's aggregate throughput (it was 0.87-0.89× on 2026-11-08), and
  the 4-user rows are 1.13-1.38×. On the 4-core bluefin, Peregrine serves 10 users at once up to
  1.80× faster than llama.cpp (SmolLM2 360M at 4096; 1.73× at 512), and 1.47× on Llama 3.2 1B at 512
  (1.52× at 4096).
- **Prompt processing is ahead on both hosts.** The desktop moved from 0.69-0.74× on 2026-11-08 to
  1.03-1.30× after the repacked Q4_K/Q5_0 work (#108, #111, #118, #120); on bluefin it is 1.03-1.04×
  on Llama 3.2 1B, 1.43× and 1.17× on Qwen 2.5 0.5B, and 1.39× and 1.05× on SmolLM2 360M.
- **Bonsai 2 (ternary): 2.0-2.3× the decode and 2.1-2.5× the prompt speed of PrismML's own
  llama.cpp fork, 2.6-2.8× with 10 users** (12-core desktop), against the fork's faster
  flash-attention setting for each row. Upstream llama.cpp cannot load these PQ2_0 files, so the
  fork is the only comparison that exists for them.

Losses are in the same report, and there are now very few. **No completed throughput row is below
1.0×**, on either host. The one remaining loss is time to first token with 10 users on bluefin for
the 1B at 4096 (1.04×, Peregrine slower); at 512 Peregrine is faster on every TTFT row (0.66-0.86×
on bluefin, 0.70-0.86× on the desktop). The desktop 4096-token multi-user rows do not finish inside
the 10-minute command cap and are recorded as not completed. Every number, chart and machine record
is on this page and in the result folders.

## The machines

| | Desktop (H1) | Bluefin (H2) |
|---|---|---|
| CPU | AMD Ryzen AI 9 HX 370 (Zen 5), 12 cores / 24 threads | AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads |
| SIMD | AVX2, AVX-512 (VNNI, GFNI, VBMI) | AVX2 |
| Threads used | 12 (one per core) | 4 (one per core) |
| OS | Linux under WSL2 | Fedora, bare metal |
| Peregrine | `b88fcd7` (2026-11-09 re-run), defaults: repacked k-quant tiles on (AVX-512) | `09214de` (2026-11-08, the #105 tree; #105 was open when the runs started and merged as `b86f355` during the session; main has since moved on), defaults: repacked Q4_K kernels on |
| Results | [desktop-zen5-2026-11-09](../benchmarks/results/desktop-zen5-2026-11-09/README.md), with both llama.cpp flash-attention settings in the same folder | [bluefin-zen2-2026-11-08-q5](../benchmarks/results/bluefin-zen2-2026-11-08-q5/README.md), with the earlier same-day run in [bluefin-zen2-2026-11-08-fa-off](../benchmarks/results/bluefin-zen2-2026-11-08-fa-off/README.md) |

The desktop's 2026-11-09 run is the first after the November 8-9 kernel work (#108, #111, #113,
#114, #115, #118, #120, #121). Compare ratios measured on the same day, not numbers across days:
the host's llama.cpp moved by up to 7% between the 2026-11-08 and 2026-11-09 sessions, and so did
Peregrine. The first desktop baseline is
[`desktop-zen5-2026-11-07`](../benchmarks/results/desktop-zen5-2026-11-07/README.md).

Both run llama.cpp v0.9.0 (`cd959ba`) as a server against Peregrine's server, with 256 greedy tokens
per request, prompts calibrated to the target context, three interleaved rounds and the median
reported. The models are the Q4_K_M files of Llama 3.2 1B Instruct and Qwen 2.5 0.5B Instruct on
both hosts, plus SmolLM2 360M Instruct on both and gemma-3-1b-it on the desktop, with the same file
(by SHA-256) for both engines. The Qwen 0.5B and SmolLM2 files are mixed-format GGUFs whose tensors
are mostly Q5_0; the Llama 1B file is Q4_K/Q6_K only and is the control for the Q5_0 kernels.

**Each ratio is against llama.cpp's faster flash-attention setting for that row.** llama.cpp
defaults to `-fa auto`, which resolves to flash attention on here. Which setting wins varies by row
and host: `-fa off` is faster for some desktop single-user decode rows and for all the bluefin
decode rows, while the default wins prompt processing and multi-user throughput everywhere (and
gemma's desktop prompt rows). Both settings are measured and reported; the default-setting number is
kept as a secondary note. The latest bluefin run and both settings are in
[`bluefin-zen2-2026-11-08-q5`](../benchmarks/results/bluefin-zen2-2026-11-08-q5/README.md); the
earlier same-day baseline is in
[`bluefin-zen2-2026-11-08-fa-off`](../benchmarks/results/bluefin-zen2-2026-11-08-fa-off/README.md).
The desktop re-run is in
[`desktop-zen5-2026-11-08-fa-off`](../benchmarks/results/desktop-zen5-2026-11-08-fa-off/README.md)
and finds `-fa off` faster for single-user decode and for the 0.5B at 4096, and slower for the 1B's
prompt processing and for every multi-user row.

## Single-user decode

![Single-user decode](img/bench-decode.svg)

| Machine | Model | Context | Peregrine | llama.cpp | Peregrine ÷ llama.cpp |
|---|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 512 | 60.6 | 46.4 | **1.31×** |
| Desktop | Llama 3.2 1B | 4096 | 48.5 | 30.4 | **1.60×** |
| Desktop | Qwen 2.5 0.5B | 512 | 95.5 | 69.5 | **1.37×** |
| Desktop | Qwen 2.5 0.5B | 4096 | 71.9 | 45.8 | **1.57×** |
| Desktop | SmolLM2 360M | 512 | 102.9 | 77.0 | **1.34×** |
| Desktop | SmolLM2 360M | 4096 | 70.8 | 41.9 | **1.69×** |
| Desktop | gemma-3-1b-it | 512 | 59.9 | 44.3 | **1.35×** |
| Desktop | gemma-3-1b-it | 4096 | 46.3 | 41.0 | **1.13×** |
| Bluefin | Llama 3.2 1B | 512 | 46.8 | 42.5 | **1.10×** |
| Bluefin | Llama 3.2 1B | 4096 | 36.8 | 29.3 | **1.26×** |
| Bluefin | Qwen 2.5 0.5B | 512 | 83.5 | 77.7 | **1.07×** |
| Bluefin | Qwen 2.5 0.5B | 4096 | 65.9 | 52.2 | **1.26×** |
| Bluefin | SmolLM2 360M | 512 | 108.9 | 94.5 | **1.15×** |
| Bluefin | SmolLM2 360M | 4096 | 63.8 | 47.5 | **1.34×** |

Tokens per second. Bluefin compares against llama.cpp's faster flash-attention setting: `-fa off`
for these rows. Against llama.cpp's defaults the bluefin rows are 1.19×, 2.82× (13.1 tok/s), 1.12×,
1.60× (41.1 tok/s), 1.35× and 3.11× (20.5 tok/s). On the desktop `-fa off` is the faster setting for
SmolLM2 360M and gemma-3-1b-it at 512 and for gemma at 4096; the default wins the other desktop
decode rows. At llama.cpp's defaults the desktop ratios would be 1.31×, 1.60×, 1.37×, 1.57×, 1.46×,
1.69×, 1.40× and 1.21×.

The Qwen 0.5B is now ahead on both hosts. Its file and SmolLM2's are mostly Q5_0 tensors, so the
Q5_0 kernels that landed on 2026-11-08 (#103 wide prefill, #104 eight-row decode GEMV, #105 repacked
tiles) lifted them first; the desktop's 2026-11-09 jump also carries the repacked Q4_K/Q5_0 kernels
on AVX-512 (#108, #111) and the attention work (#113-#115, #121). The Llama 1B file has no Q5_0
tensors and is the control.

## Prompt processing

![Prompt processing](img/bench-prefill.svg)

| Machine | Model | Context | Peregrine | llama.cpp | Peregrine ÷ llama.cpp |
|---|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 512 | 272.6 | 252.7 | **1.08×** |
| Desktop | Llama 3.2 1B | 4096 | 195.3 | 174.3 | **1.12×** |
| Desktop | Qwen 2.5 0.5B | 512 | 479.3 | 423.8 | **1.13×** |
| Desktop | Qwen 2.5 0.5B | 4096 | 355.1 | 272.4 | **1.30×** |
| Desktop | SmolLM2 360M | 512 | 451.1 | 436.3 | **1.03×** |
| Desktop | SmolLM2 360M | 4096 | 292.9 | 263.9 | **1.11×** |
| Desktop | gemma-3-1b-it | 512 | 284.9 | 258.1 | **1.10×** |
| Desktop | gemma-3-1b-it | 4096 | 279.6 | 247.9 | **1.13×** |
| Bluefin | Llama 3.2 1B | 512 | 195.7 | 188.1 | **1.04×** |
| Bluefin | Llama 3.2 1B | 4096 | 145.0 | 141.4 | **1.03×** |
| Bluefin | Qwen 2.5 0.5B | 512 | 434.8 | 303.7 | **1.43×** |
| Bluefin | Qwen 2.5 0.5B | 4096 | 277.6 | 236.5 | **1.17×** |
| Bluefin | SmolLM2 360M | 512 | 443.2 | 318.9 | **1.39×** |
| Bluefin | SmolLM2 360M | 4096 | 228.5 | 217.1 | **1.05×** |

On bluefin, llama.cpp's default (`-fa auto` on) is faster than `-fa off` for prompt processing
(`-fa off` gives 174.7 and 123.0 tok/s on the 1B, 278.1 and 194.1 on the 0.5B, and 284.5 and 172.5 on
SmolLM2), so the default is the like-for-like competitor here. The Qwen and SmolLM2 prompt rows are
where the wide Q5_0 prefill kernel shows most: their files are mostly Q5_0, while the 1B's is
Q4_K/Q6_K.

On the desktop llama.cpp's default is the faster setting for Llama 3.2 1B (the `-fa off` number is
232.3 and 163.0 tok/s, 8% and 6% slower), for Qwen 2.5 0.5B at 512 (8% slower), for Qwen at 4096
(0.5% slower) and for SmolLM2 at 512 (1% slower); `-fa off` is 2% faster for SmolLM2 at 4096 and
0.2% and 6% faster for gemma-3-1b-it at 512 and 4096, and those rows use it. The desktop's prompt
processing moved from 0.69-0.74× on 2026-11-08 to 1.03-1.30×. The repacked Q4_K/Q5_0 store is now on
by default on AVX-512 (#108) with its own AVX-512 kernels (#111), and the prefill GEMM and Q6_K
changes (#118, #120) lifted the rest; the 2026-11-08 A/B that held the repack back (decode −1.2%,
prefill +26%) no longer describes the default.

## Many users at once

![Total throughput, 10 users at once](img/bench-users10.svg)

| Machine | Model | 4 users | 10 users | 10 users, Peregrine ÷ llama.cpp |
|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 70.3 vs 60.3 | 83.6 vs 61.8 | **1.35×** |
| Desktop | Qwen 2.5 0.5B | 108.4 vs 96.0 | 121.7 vs 93.6 | **1.30×** |
| Desktop | SmolLM2 360M | 117.1 vs 85.0 | 146.6 vs 102.4 | **1.43×** |
| Desktop | gemma-3-1b-it | 63.4 vs 49.6 | 69.3 vs 54.1 | **1.28×** |
| Bluefin | Llama 3.2 1B | 53.7 vs 39.6 | 55.5 vs 37.7 | **1.47×** |
| Bluefin | Qwen 2.5 0.5B | 104.3 vs 65.7 | 108.4 vs 68.4 | **1.59×** |
| Bluefin | SmolLM2 360M | 114.2 vs 66.2 | 116.8 vs 67.6 | **1.73×** |

Total tokens per second over all users at a 512-token context, Peregrine vs llama.cpp. On bluefin
the default is the faster llama.cpp setting for multi-user throughput: `-fa off` drops it to 34.0
and 23.8 tok/s (1B, 4 and 10 users), 54.9 and 42.4 tok/s (0.5B), and 54.3 and 37.4 tok/s (SmolLM2),
so these ratios use the default. The desktop agrees: `-fa off` drops llama.cpp to 41.3 and 36.1
tok/s (1B), 58.7 and 54.4 (0.5B), 59.6 and 50.4 (SmolLM2) and 45.5 and 45.4 (gemma), so the default
is the competitor for every desktop multi-user row too.

The desktop's multi-user rows are the biggest change of the run: 10 users at 512 went from 0.87×
and 0.89× on 2026-11-08 to 1.28-1.43× across the four models, and the 4-user rows from 0.95-0.96× to
1.13-1.38×. That is the one-attention-region work for all decoding users (#114) plus the attention
rewrites (#113, #115, #121). Time to first token with 10 users is faster than llama.cpp on every
desktop row (0.70-0.86×). At 4096 tokens on bluefin the 1B is 7.69 vs 5.38 (4 users, 1.43×) and
7.71 vs 5.07 (10 users, 1.52×); the 0.5B is 14.75 vs 10.91 (1.35×) and 14.28 vs 10.20 (1.40×);
SmolLM2 is 11.99 vs 7.74 (1.55×) and 12.11 vs 6.74 (1.80×). The desktop's 4096-token multi-user
rows do not fit the 10-minute command cap and are recorded as not completed in its result folder;
one round of the cheapest combination already takes 3m43.

## What changed in November

![Prompt processing: the November changes](img/bench-speedups.svg)

### What changed today (2026-11-09)

The desktop was re-measured after the November 8-9 work: repacked k-quant tiles on by default on
AVX-512 (#108), AVX-512 repacked Q4_K/Q5_0 kernels (#111), the prefill-attention rewrites (#113,
#115), one attention region for all decoding users (#114), the Q6_K and prefill-GEMM changes (#118,
#120) and the AVX-512 attention mix (#121). Peregrine against its own 2026-11-08 build, same host
and files:

| Model | Metric | 2026-11-08 | 2026-11-09 | Change |
|---|---|---|---|---|
| Llama 3.2 1B | Prompt 512 tok/s | 183.8 | 272.6 | +48% |
| Llama 3.2 1B | Prompt 4096 tok/s | 130.9 | 195.3 | +49% |
| Llama 3.2 1B | 10-user tok/s | 53.4 | 83.6 | +57% |
| Llama 3.2 1B | 4-user tok/s | 51.6 | 70.3 | +36% |
| Qwen 2.5 0.5B | Prompt 512 tok/s | 321.9 | 479.3 | +49% |
| Qwen 2.5 0.5B | Prompt 4096 tok/s | 215.1 | 355.1 | +65% |
| Qwen 2.5 0.5B | 10-user tok/s | 83.4 | 121.7 | +46% |
| Qwen 2.5 0.5B | Decode 512 tok/s | 70.0 | 95.5 | +37% |
| Bonsai 2 2B | Decode 512 tok/s | 70.6 | 75.1 | +6% |
| Bonsai 2 2B | Prompt 4096 tok/s | 202.9 | 247.3 | +22% |
| Bonsai 2 9B | Decode 4096 tok/s | 20.4 | 22.7 | +11% |
| Bonsai 2 9B | Prompt 4096 tok/s | 57.4 | 63.0 | +10% |

The Llama 1B file has no Q5_0 tensors, so its prompt gain is the repacked Q4_K path (#108, #111,
#118, #120). The Qwen and SmolLM2 files are mostly Q5_0. No per-PR A/B was run for this report; the
table is the combined end-to-end change, and the fork also moved by 5-11% between the two sessions
(same-day host drift). The full tables and raw manifests are in
[`desktop-zen5-2026-11-09`](../benchmarks/results/desktop-zen5-2026-11-09/README.md).

### What changed on 2026-11-08

Three Q5_0 changes are in the bluefin build:

- **#103 — wide Q5_0 prefill kernel.** Prompt processing for Q5_0 weights with a 16-row-wide tile.
- **#104 — eight-row Q5_0/Q8_0 decode GEMV.** Less serial work per layer in the decode matrix-vector
  path.
- **#105 — repacked Q5_0 tiles.** Q5_0 weights repacked into the tile layout the wide kernels read.

The Qwen 2.5 0.5B "Q4_K_M" file is 132-of-169 Q5_0 tensors and SmolLM2's is 176-of-274; the Llama
3.2 1B file has none. On bluefin, against the earlier same-day build, Qwen 2.5 0.5B went from 52.87
to 83.48 tok/s decode at 512 (+58%) and 263.19 to 434.78 prompt (+65%), and from 45.20 to 65.89
decode at 4096 (+46%) and 198.23 to 277.58 prompt (+40%); 10-user throughput rose 51% and 46%. Every
Llama 3.2 1B row moved by less than 2%, as expected for a file with no Q5_0 tensors. SmolLM2 360M is
new in this run and is ahead of llama.cpp on every throughput row. Full tables:
[`bluefin-zen2-2026-11-08-q5`](../benchmarks/results/bluefin-zen2-2026-11-08-q5/README.md).

Peregrine against its own earlier build on the same machine and file:

| Change | Machine | Model | Prompt processing |
|---|---|---|---|
| Wide Q5_0 prefill + eight-row decode + repacked tiles (#103-#105) | Bluefin | Qwen 2.5 0.5B | 263 → 435 tok/s (+65%) |
| Repacked Q4_K kernels (#88) | Bluefin | Llama 3.2 1B | 104 → 152 tok/s (+46%) |
| Repacked Q4_K kernels (#88) | Bluefin | Qwen 3.5 2B | 72 → 96 tok/s (+33%) |
| Prefill chunk follows the worker pool (#92) | Desktop | Bonsai 2 2B | +43% |
| Prefill chunk follows the worker pool (#92) | Desktop | Llama 3.2 1B | +19% |
| Prefill chunk follows the worker pool (#92) | Desktop | Bonsai 2 9B | +14% |

Decode also moved by +0.6% to +2.9% with the finer attention split (#94). The prefill chunk and the attention
split leave the output bit for bit the same; the repacked Q4_K kernels round differently, and their
KL divergence against llama.cpp stayed flat on every model checked. The Q5_0 kernels keep each row's
bits independent of batch width and worker; the AVX2 Q5_0 goldens were re-recorded for the new fold,
and the synthetic greedy tokens are unchanged.

## Reproducing

```sh
python3 benchmarks/run.py --help     # the harness: one engine, one model, the four numbers
python3 benchmarks/charts.py         # redraws the charts on this page from the numbers in it
```

Pass `--llama-fa off` (or `on`) to pin llama.cpp's flash attention; the default `auto` reproduces
llama.cpp's own default. The choice is written to the run manifest. Ratios on this page use the
faster setting per row.

The charts are drawn by [`benchmarks/charts.py`](../benchmarks/charts.py) from the medians in the
result folders; update its tables when a new result lands, then rerun it.

## Bonsai 2 vs PrismML's fork

Bonsai 2's ternary files (PQ2_0, ggml type 142) do not load in upstream llama.cpp, so the
llama.cpp comparison above cannot cover them. [PrismML's
fork](https://github.com/PrismML-Eng/llama.cpp) is the other engine that reads them. Measured
2026-11-09 on the Ryzen AI 9 HX 370 desktop (12 threads, WSL2), 3 interleaved rounds, 256
greedy tokens, prompt lengths calibrated per engine. Peregrine at `b88fcd7`; fork at
`62c418ccd7aefee37997e94c2d9fac1eed7fc89b` (branch `prism`). Full method and raw manifests:
[`benchmarks/results/2026-11-09-bonsai2-prism/`](../benchmarks/results/2026-11-09-bonsai2-prism/).

![Bonsai 2: Peregrine against PrismML's fork](img/bench-bonsai.svg)

Medians. "Fork best" is the faster of the fork's effective `-fa on` and `-fa off` server
settings for that row (lower for TTFT); the fork's default `-fa auto` resolves to on here, and
explicit-on runs on the 512-token rows confirm it (2B decode 34.77 vs 34.45, 9B 10.35 vs
10.75). "Ratio" is Peregrine ÷ fork best, so above 1 means Peregrine is faster and, for TTFT,
below 1 means Peregrine's first token arrives sooner. The fork's own default is the secondary
column.

| Model | Context | Users | Metric | Peregrine | Fork best | Ratio | Fork default |
|---|---|---|---|---|---|---|---|
| Bonsai 2 2B PQ2_0 | 512 | 1 | Decode tok/s | 75.14 | 34.89 (`-fa off`) | 2.15x | 34.45 |
| Bonsai 2 2B PQ2_0 | 512 | 1 | Prompt tok/s | 270.24 | 119.35 (`-fa on`) | 2.26x | 119.35 |
| Bonsai 2 2B PQ2_0 | 512 | 10 | Aggregate tok/s | 75.77 | 29.56 (`-fa on`) | 2.56x | 29.56 |
| Bonsai 2 2B PQ2_0 | 512 | 10 | TTFT (s) | 14.10 | 44.38 (`-fa on`) | 0.32x | 44.38 |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Decode tok/s | 64.60 | 32.65 (`-fa off`) | 1.98x | 32.54 |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Prompt tok/s | 247.30 | 116.35 (`-fa off`) | 2.13x | 113.46 |
| Bonsai 2 9B PQ2_0 | 512 | 1 | Decode tok/s | 25.47 | 10.92 (`-fa off`) | 2.33x | 10.75 |
| Bonsai 2 9B PQ2_0 | 512 | 1 | Prompt tok/s | 71.65 | 28.45 (`-fa on`) | 2.52x | 28.45 |
| Bonsai 2 9B PQ2_0 | 512 | 10 | Aggregate tok/s | 21.66 | 7.87 (`-fa on`) | 2.75x | 7.87 |
| Bonsai 2 9B PQ2_0 | 512 | 10 | TTFT (s) | 52.12 | 181.96 (`-fa on`) | 0.29x | 181.96 |
| Bonsai 2 9B PQ2_0 | 4096 | 1 | Decode tok/s | 22.69 | 10.12 (`-fa on`) | 2.24x | 10.12 |
| Bonsai 2 9B PQ2_0 | 4096 | 1 | Prompt tok/s | 63.04 | 27.85 (`-fa off`) | 2.26x | 27.48 |

MTP speculation on the 9B `-MTP-Q8_0` file (512 tokens, 1 user): Peregrine decodes 28.80
tok/s with MTP and 25.14 without (1.15x). The fork's `--spec-type draft-mtp` did not beat its
own non-speculative decode (9.33 vs 10.62); against it the ratios are 3.09x (MTP on) and 2.37x
(MTP off).

Flash attention: explicit `-fa on` and `-fa off` land within 6% of each other on every
server row. The large depth penalty upstream llama.cpp's default showed on the bluefin 1B
(13.0 vs 29.1 tok/s at 4096) does not reproduce on these hybrid-architecture Bonsai 2 files;
turning flash attention off is not a win here, and the default is the faster or equal
setting. The full on/off tables are in the results folder.

The manifests record `machine.simd_tier: x86_64-avx2-avx512-vnni-gfni-vbmi`, and `peregrine
tier` prints the same kernel tier, so Bonsai 2 ran the AVX-512 tier: PQ2_0's fastest kernel is
the 16-row repacked-tile path built on AVX-512 VNNI and GFNI (`MatPath::Tiles` in
`peregrine-kernels/src/table.rs`), selected with `zmm=true`, `tiles=true` and `isa=native`. The
fork was built with `-march=native`, so both engines use AVX-512 here. (The earlier
`avx2+fma` label in the 2026-11-07 manifests was a probe bug, fixed by #107; it never meant
the AVX-512 tier was skipped.)
