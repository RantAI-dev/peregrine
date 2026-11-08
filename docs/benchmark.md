# Benchmarks: Peregrine against llama.cpp

Where Peregrine stands against llama.cpp on the same machine, the same model file and the same thread
count. Wins and losses are both here. The raw data, method and machine records are in
[`benchmarks/results/`](../benchmarks/results/); the targets these numbers are held to are in
[targets.md](targets.md).

![Where Peregrine is fastest](img/bench-wins.svg)

## The short version

- **Single-user decode is Peregrine's strongest result.** It beats llama.cpp in 9 of the 10 confirmed
  cases, by up to 1.39× (Llama 3.2 1B at a 4096-token context on the desktop). The gap grows with
  context: Peregrine's attention reads each cached key and value once per group of heads, and spends
  about 2.3× less time per context position than llama.cpp. Against llama.cpp's faster
  flash-attention setting, the desktop 1B is ahead 1.21× at 512 and 1.39× at 4096; on bluefin all
  three models are ahead, from 1.07× to 1.34×.
- **Many users on a small machine.** On the 4-core bluefin, Peregrine serves 10 users at once up to
  1.80× faster than llama.cpp (SmolLM2 360M at 4096; 1.73× at 512), and 1.47× on Llama 3.2 1B at 512
  (1.52× at 4096), against llama.cpp's faster setting.
- **Prompt processing has caught up on AVX2, and moved ahead on Q5_0-heavy files.** On bluefin,
  Peregrine is at 1.03-1.04× llama.cpp on Llama 3.2 1B, 1.43× and 1.17× on Qwen 2.5 0.5B, and 1.39×
  and 1.05× on SmolLM2 360M, after the Q5_0 kernels (#103-#105). On the desktop it is at 0.69-0.74×;
  the repacked Q4_K kernels lift the desktop's Llama 3.2 1B from 0.70× to 0.89× at 512 (0.83× at
  4096), but they are not on by default there yet.
- **Where it loses:** the 0.5B model at 512 on the desktop (decode 0.92×), prompt processing on the
  desktop (0.69-0.74×), multi-user throughput on the desktop (0.87× at 10 users, 0.95× at 4), and
  time to first token with 10 users on bluefin for the 1B at 4096 (1.04×). At 512 Peregrine is faster
  on all three TTFT rows (0.66-0.86×), and at 4096 on the two small models (0.85× and 0.96×). No
  bluefin throughput row is below 1.0× any more.

## The machines

| | Desktop (H1) | Bluefin (H2) |
|---|---|---|
| CPU | AMD Ryzen AI 9 HX 370 (Zen 5), 12 cores / 24 threads | AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads |
| SIMD | AVX2, AVX-512 (VNNI, GFNI, VBMI) | AVX2 |
| Threads used | 12 (one per core) | 4 (one per core) |
| OS | Linux under WSL2 | Fedora, bare metal |
| Peregrine | `e48aea5` (2026-11-08 re-run, 20:00), defaults: repacked Q4_K kernels off | `09214de` (2026-11-08, the #105 tree; #105 was open when the runs started and merged as `b86f355` during the session; main has since moved on), defaults: repacked Q4_K kernels on |
| Results | [desktop-zen5-2026-11-08-fa-off](../benchmarks/results/desktop-zen5-2026-11-08-fa-off/README.md), with the same-day default-setting baseline in [desktop-zen5-2026-11-08](../benchmarks/results/desktop-zen5-2026-11-08/README.md) | [bluefin-zen2-2026-11-08-q5](../benchmarks/results/bluefin-zen2-2026-11-08-q5/README.md), with the earlier same-day run in [bluefin-zen2-2026-11-08-fa-off](../benchmarks/results/bluefin-zen2-2026-11-08-fa-off/README.md) |

The desktop's same-day llama.cpp ran 0-7% faster than on 2026-11-07
([first desktop baseline](../benchmarks/results/desktop-zen5-2026-11-07/README.md)), so compare
ratios measured on the same day, not numbers across days.

Both run llama.cpp v0.9.0 (`cd959ba`) as a server against Peregrine's server, with 256 greedy tokens
per request, prompts calibrated to the target context, three interleaved rounds and the median
reported. The models are the Q4_K_M files of Llama 3.2 1B Instruct and Qwen 2.5 0.5B Instruct, plus
SmolLM2 360M Instruct on bluefin, with the same file (by SHA-256) for both engines. The Qwen 0.5B and
SmolLM2 files are mixed-format GGUFs whose tensors are mostly Q5_0; the Llama 1B file is Q4_K/Q6_K
only and is the control for the Q5_0 kernels.

**Each ratio is against llama.cpp's faster flash-attention setting for that row.** llama.cpp
defaults to `-fa auto`, which resolves to flash attention on here; that path is slower for
single-user decode, and `-fa off` is used for those rows, while the default is faster for prompt
processing and multi-user throughput and is used there. The default-setting number is kept as a
secondary note. The latest bluefin run and both settings are in
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
| Desktop | Llama 3.2 1B | 512 | 62.6 | 51.8 | **1.21×** |
| Desktop | Llama 3.2 1B | 4096 | 47.0 | 33.8 | **1.39×** |
| Desktop | Qwen 2.5 0.5B | 512 | 70.0 | 75.8 | 0.92× |
| Desktop | Qwen 2.5 0.5B | 4096 | 54.9 | 49.1 | **1.12×** |
| Bluefin | Llama 3.2 1B | 512 | 46.8 | 42.5 | **1.10×** |
| Bluefin | Llama 3.2 1B | 4096 | 36.8 | 29.3 | **1.26×** |
| Bluefin | Qwen 2.5 0.5B | 512 | 83.5 | 77.7 | **1.07×** |
| Bluefin | Qwen 2.5 0.5B | 4096 | 65.9 | 52.2 | **1.26×** |
| Bluefin | SmolLM2 360M | 512 | 108.9 | 94.5 | **1.15×** |
| Bluefin | SmolLM2 360M | 4096 | 63.8 | 47.5 | **1.34×** |

Tokens per second. Bluefin compares against llama.cpp's faster flash-attention setting: `-fa off`
for these rows. Against llama.cpp's defaults the bluefin rows are 1.19×, 2.82× (13.1 tok/s), 1.12×,
1.60× (41.1 tok/s), 1.35× and 3.11× (20.5 tok/s). On the desktop `-fa off` is the faster setting for
the 1B at both contexts and for the 0.5B at 4096 (and 1.7% slower on the 0.5B at 512, where the
default is used); at llama.cpp's defaults the desktop 1B ratios would be 1.23× and 1.45×.

The Qwen 0.5B no longer loses on bluefin. Its file and SmolLM2's are mostly Q5_0 tensors, so the
Q5_0 kernels that landed on 2026-11-08 (#103 wide prefill, #104 eight-row decode GEMV, #105 repacked
tiles) lifted them; the Llama 1B file has no Q5_0 tensors and stayed within 2% of its earlier
same-day numbers.

## Prompt processing

![Prompt processing](img/bench-prefill.svg)

| Machine | Model | Context | Peregrine | llama.cpp | Peregrine ÷ llama.cpp |
|---|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 512 | 183.8 | 262.0 | 0.70× |
| Desktop | Llama 3.2 1B | 4096 | 130.9 | 189.0 | 0.69× |
| Desktop | Qwen 2.5 0.5B | 512 | 321.9 | 435.2 | 0.74× |
| Desktop | Qwen 2.5 0.5B | 4096 | 215.1 | 293.1 | 0.73× |
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

On the desktop, llama.cpp's default stays the faster prompt setting for the 1B: `-fa off` gives 251.5
and 178.5 tok/s, 4.0% and 5.6% slower. On the 0.5B it is 1.0% slower at 512 and 4.9% faster at 4096,
so that row uses `-fa off`.

On the desktop the repacked Q4_K kernels are off by default (AVX-512 has no kernel of its own yet).
Five alternating rounds with `PEREGRINE_KREPACK=1` on Llama 3.2 1B take prompt processing from 185.5
to 233.9 tok/s at 512 (0.89× of llama.cpp's default) and from 130.6 to 156.1 at 4096 (0.83×), with
decode 1.2% lower at 512 and flat at 4096. The 0.5B does not gain from them.

## Many users at once

![Total throughput, 10 users at once](img/bench-users10.svg)

| Machine | Model | 4 users | 10 users | 10 users, Peregrine ÷ llama.cpp |
|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 51.6 vs 54.5 | 53.4 vs 61.7 | 0.87× |
| Desktop | Qwen 2.5 0.5B | 78.3 vs 81.8 | 83.4 vs 93.4 | 0.89× |
| Bluefin | Llama 3.2 1B | 53.7 vs 39.6 | 55.5 vs 37.7 | **1.47×** |
| Bluefin | Qwen 2.5 0.5B | 104.3 vs 65.7 | 108.4 vs 68.4 | **1.59×** |
| Bluefin | SmolLM2 360M | 114.2 vs 66.2 | 116.8 vs 67.6 | **1.73×** |

Total tokens per second over all users at a 512-token context, Peregrine vs llama.cpp. On bluefin
the default is the faster llama.cpp setting for multi-user throughput: `-fa off` drops it to 34.0
and 23.8 tok/s (1B, 4 and 10 users), 54.9 and 42.4 tok/s (0.5B), and 54.3 and 37.4 tok/s (SmolLM2),
so these ratios use the default. The desktop agrees, and by a wider margin: `-fa off` drops
llama.cpp to 41.6 and 36.0 tok/s (1B) and 63.7 and 56.8 (0.5B) at 4 and 10 users, so the default is
the competitor for every desktop multi-user row too.

At 4096 tokens on bluefin the 1B is 7.69 vs 5.38 (4 users, 1.43×) and 7.71 vs 5.07 (10 users,
1.52×); the 0.5B is 14.75 vs 10.91 (1.35×) and 14.28 vs 10.20 (1.40×); SmolLM2 is 11.99 vs 7.74
(1.55×) and 12.11 vs 6.74 (1.80×). The 10-user, 4096, `-fa off` phase does not finish inside the
10-minute command cap on either host, so those rows use the default where one exists (bluefin) and
are recorded as not completed on the desktop, where the default-setting numbers from the same day
are still in its baseline folder. At 4096 tokens both engines fall to 5-15 tok/s in total, too
coarse to compare closely; those numbers are in the result folders.

## What changed in November

![Prompt processing: the November changes](img/bench-speedups.svg)

### What changed today (2026-11-08)

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
2026-11-08 on two hosts: the Ryzen AI 9 HX 370 desktop (12 threads, WSL2, AVX-512) and the
Ryzen 3 PRO 4350G bluefin (4 threads, bare metal, AVX2), 3 interleaved rounds, 256 greedy
tokens, prompt lengths calibrated per engine. Desktop: Peregrine at `bf241fe`; bluefin:
`cf9e81d`. The fork is `62c418ccd7aefee37997e94c2d9fac1eed7fc89b` (branch `prism`) on both.
Full method, raw manifests and the greedy-text check:
[desktop](../benchmarks/results/2026-11-08-bonsai2-prism/) and
[bluefin](../benchmarks/results/bluefin-zen2-2026-11-08-bonsai2-prism/).

![Bonsai 2: Peregrine against PrismML's fork](img/bench-bonsai.svg)

Medians. "Fork best" is the faster of the fork's flash-attention settings for that row
(explicit `-fa on`/`-fa off` on the desktop; `-fa auto`, which resolves to on, and `-fa off`
on the bluefin; lower for TTFT); "ratio" is Peregrine ÷ fork best, so above 1 means Peregrine
is faster and, for TTFT, below 1 means Peregrine's first token arrives sooner. The fork's own
default (`-fa auto`, which resolves to on here) is the secondary column.

### Desktop (12 cores, AVX-512)

| Model | Context | Users | Metric | Peregrine | Fork best | Ratio | Fork default |
|---|---|---|---|---|---|---|---|
| Bonsai 2 2B PQ2_0 | 512 | 1 | Decode tok/s | 70.64 | 32.66 (`-fa on`) | 2.16x | 32.67 |
| Bonsai 2 2B PQ2_0 | 512 | 1 | Prompt tok/s | 259.25 | 108.24 (`-fa on`) | 2.40x | 112.19 |
| Bonsai 2 2B PQ2_0 | 512 | 10 | Aggregate tok/s | 80.59 | 29.96 (`-fa on`) | 2.69x | 29.79 |
| Bonsai 2 2B PQ2_0 | 512 | 10 | TTFT (s) | 12.92 | 42.79 (`-fa on`) | 0.30x | 45.40 |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Decode tok/s | 60.64 | 30.47 (`-fa on`) | 1.99x | 30.76 |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Prompt tok/s | 202.94 | 105.19 (`-fa off`) | 1.93x | 103.90 |
| Bonsai 2 9B PQ2_0 | 512 | 1 | Decode tok/s | 22.97 | 9.85 (`-fa off`) | 2.33x | 9.82 |
| Bonsai 2 9B PQ2_0 | 512 | 1 | Prompt tok/s | 67.24 | 26.12 (`-fa off`) | 2.57x | 26.63 |
| Bonsai 2 9B PQ2_0 | 512 | 10 | Aggregate tok/s | 21.62 | 7.83 (`-fa on`) | 2.76x | 7.76 |
| Bonsai 2 9B PQ2_0 | 512 | 10 | TTFT (s) | 51.66 | 183.40 (`-fa on`) | 0.28x | 185.48 |
| Bonsai 2 9B PQ2_0 | 4096 | 1 | Decode tok/s | 20.36 | 9.22 (`-fa on`) | 2.21x | 8.93 |
| Bonsai 2 9B PQ2_0 | 4096 | 1 | Prompt tok/s | 57.42 | 25.53 (`-fa off`) | 2.25x | 25.33 |

### Bluefin (4 cores, AVX2)

This is the AVX2-vs-AVX2 point: the host has no AVX-512, so Peregrine runs the rows-path
PQ2_0 kernel and the fork its AVX2 path, instead of the AVX-512 tile kernels both used on the
desktop. The 4B file is the `-MTP-Q8_0` one (no plain 4B PQ2_0 exists on this host), measured
with Peregrine's default MTP on; the fork's 4B rows are non-speculative.

| Model | Context | Users | Metric | Peregrine | Fork best | Ratio | Fork default |
|---|---|---|---|---|---|---|---|
| Bonsai 2 2B PQ2_0 | 512 | 1 | Decode tok/s | 46.40 | 33.63 (`-fa off`) | 1.38x | 33.46 |
| Bonsai 2 2B PQ2_0 | 512 | 1 | Prompt tok/s | 118.67 | 56.95 (`-fa off`) | 2.08x | 55.15 |
| Bonsai 2 2B PQ2_0 | 512 | 10 | Aggregate tok/s | 33.69 | 19.08 (`-fa auto`) | 1.77x | 19.08 |
| Bonsai 2 2B PQ2_0 | 512 | 10 | TTFT (s) | 32.82 | 82.05 (`-fa auto`) | 0.40x | 82.05 |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Decode tok/s | 42.28 | 30.79 (`-fa off`) | 1.37x | 25.80 |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Prompt tok/s | 109.04 | 58.07 (`-fa auto`) | 1.88x | 58.07 |
| Bonsai 2 4B PQ2_0 | 512 | 1 | Decode tok/s | 24.59 | 14.95 (`-fa off`) | 1.64x | 14.82 |
| Bonsai 2 4B PQ2_0 | 512 | 1 | Prompt tok/s | 48.05 | 18.55 (`-fa off`) | 2.59x | 18.07 |
| Bonsai 2 4B PQ2_0 | 512 | 10 | Aggregate tok/s | 14.12 | 6.37 (`-fa auto`) | 2.22x | 6.37 |
| Bonsai 2 4B PQ2_0 | 512 | 10 | TTFT (s) | 78.84 | 279.73 (`-fa auto`) | 0.28x | 279.73 |
| Bonsai 2 4B PQ2_0 | 4096 | 1 | Decode tok/s | 21.28 | not completed | — | — |
| Bonsai 2 4B PQ2_0 | 4096 | 1 | Prompt tok/s | 43.91 | not completed | — | — |

The bluefin 4B row at 4096 is not completed: the fork's calibration-plus-measurement phase was
stopped at the 10-minute command cap, so no estimate is substituted. Peregrine's own 4B 4096
row completed (43.91 tok/s prompt, 21.28 tok/s decode).

**What the 4-core point says.** Peregrine's lead is smaller on AVX2. The same 2B file, measured
the same way, was 2.16x decode and 2.40x prompt at 512 on the desktop, against 1.38x and 2.08x
here; at 4096 it was 1.99x/1.93x on the desktop, against 1.37x/1.88x; 10-user aggregate falls
from 2.69x to 1.77x. Prompt processing keeps most of the gap, because the fork's AVX2 prefill
is relatively weak; decode narrows most, which is what a rows-path PQ2_0 kernel against a
mature AVX2 fork looks like. The 4B, absent on the desktop, is at 1.64x decode, 2.59x prompt
and 2.22x at 10 users.

MTP speculation on the 9B `-MTP-Q8_0` file (512 tokens, 1 user): Peregrine decodes 25.01
tok/s with MTP and 22.75 without (1.10x). The fork's `--spec-type draft-mtp` did not beat its
own non-speculative decode; against its faster measured setting the ratios are 2.59x (MTP on)
and 2.13x (MTP off). The fork's two MTP rows were measured an hour apart and their on/off
difference is inside the host's between-batch drift, so treat the fork's MTP as parity.

On the bluefin's 4B `-MTP-Q8_0` file, MTP adds 1.18x single-user decode at 512 (24.59 vs
20.79 tok/s) and 1.13x at 4096 (21.28 vs 18.77), and nothing at 10 users (14.12 vs 14.07,
1.00x), where prefill and batching dominate. The fork never enables the MTP head in these
rows, so its on/off is parity by construction. Against the fork's faster setting the
Peregrine MTP-off row is 1.39x and the MTP-on row 1.64x at 512/1u.

Flash attention: on the desktop, explicit `-fa on` and `-fa off` land within 0–2% of each
other on every server row; the large depth penalty upstream llama.cpp's default showed on the
bluefin 1B (13.0 vs 29.1 tok/s at 4096) did not reproduce on the desktop's hybrid-architecture
Bonsai 2 files. On the bluefin it partly does: on/off stay within 3–5% at 512 and for prompt
processing, but the 2B at 4096 decodes 19% faster with `-fa off` (30.79 vs 25.80 tok/s), so
that row's ratio uses off. The default is still faster for the 2B 4096 prompt and for both
10-user rows. The full on/off tables are in the bluefin results folder. No `llama-bench`
cross-check was run on this host.

The desktop manifests record `machine.simd_tier: avx2+fma`, but that is a stale label,
not the tier the kernels ran at. `peregrine tier` on the desktop prints
`kernel tier x86_64-avx2-avx512-vnni-gfni-vbmi` beside `simd avx2+fma`: two functions feed
those lines, and only the first resolves the kernel tier (AVX-512 here, `isa=native`).
`peregrine_core::simd_name()` knows `scalar`, `avx2`, `avx2+fma` and `neon` only and returns
the first that matches, and the hardware probe writes that coarse name into
`machine.simd_tier`. Bonsai 2 does run the AVX-512 tier: PQ2_0's fastest kernel is the
16-row repacked-tile path built on AVX-512 VNNI and GFNI (`MatPath::Tiles` in
`peregrine-kernels/src/table.rs`), selected with `zmm=true`, `tiles=true` and `isa=native`.
So both engines use AVX-512 here; the fork was also built with `-march=native`. The label is
a probe bug, not an AVX-512 tier that was skipped, and it does not change any number on this
page.

The bluefin has no stale label: `peregrine tier` prints `kernel tier x86_64-avx2`, every
manifest records `machine.simd_tier: x86_64-avx2`, and the fork was built with
`GGML_NATIVE=ON` (which resolves to `-march=native = znver2`, AVX2/FMA) plus explicit
`-DGGML_AVX2/AVX/FMA=ON`. PQ2_0 therefore runs `MatPath::RowsPq2` there, not the AVX-512
`MatPath::Tiles` path, which is why the bluefin is the AVX2-vs-AVX2 data point and why its
ratios sit below the desktop's.
