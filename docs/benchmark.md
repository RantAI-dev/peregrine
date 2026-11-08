# Benchmarks: Peregrine against llama.cpp

Where Peregrine stands against llama.cpp on the same machine, the same model file and the same thread
count. Wins and losses are both here. The raw data, method and machine records are in
[`benchmarks/results/`](../benchmarks/results/); the targets these numbers are held to are in
[targets.md](targets.md).

![Where Peregrine is fastest against llama.cpp](img/bench-wins.svg)

## The short version

- **Single-user decode is Peregrine's strongest result.** It beats llama.cpp in 5 of the 7 confirmed cases,
  by up to 1.36× (Llama 3.2 1B at a 4096-token context on the desktop). The gap grows with context:
  Peregrine's attention reads each cached key and value once per group of heads, and spends about
  2.3× less time per context position than llama.cpp.
- **Many users on a small machine.** On the 4-core bluefin, Peregrine serves 10 users at once 1.48×
  faster than llama.cpp on Llama 3.2 1B.
- **Prompt processing has caught up on AVX2.** On bluefin, with the repacked Q4_K kernels, Peregrine
  matches llama.cpp on Llama 3.2 1B (1.02-1.04×). On the desktop it is at 0.62-0.66×; the same
  kernels lift the desktop's Llama 3.2 1B from 0.66× to 0.81×, but they are not on by default there yet.
- **Where it loses:** the 0.5B model at a short context (decode 0.70× on bluefin, 0.90× on the
  desktop), prompt processing on the desktop, and multi-user throughput on the desktop (0.84×).

## The machines

| | Desktop (H1) | Bluefin (H2) |
|---|---|---|
| CPU | AMD Ryzen AI 9 HX 370 (Zen 5), 12 cores / 24 threads | AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads |
| SIMD | AVX2, AVX-512 (VNNI, GFNI, VBMI) | AVX2 |
| Threads used | 12 (one per core) | 4 (one per core) |
| OS | Linux under WSL2 | Fedora, bare metal |
| Peregrine | `5f76d48` (2026-11-08), defaults: repacked Q4_K kernels off | `5f76d48` (2026-11-08), defaults: repacked Q4_K kernels on |
| Results | [desktop-zen5-2026-11-08](../benchmarks/results/desktop-zen5-2026-11-08/README.md) | [bluefin-zen2-2026-11-08](../benchmarks/results/bluefin-zen2-2026-11-08/README.md) |

The desktop's same-day llama.cpp ran 0-7% faster than on 2026-11-07
([first desktop baseline](../benchmarks/results/desktop-zen5-2026-11-07/README.md)), so compare
ratios measured on the same day, not numbers across days.

Both run llama.cpp v0.9.0 (`cd959ba`) as a server against Peregrine's server, with 256 greedy tokens
per request, prompts calibrated to the target context, three interleaved rounds and the median
reported. The models are the Q4_K_M files of Llama 3.2 1B Instruct and Qwen 2.5 0.5B Instruct, with
the same file (by SHA-256) for both engines.

## Single-user decode

![Single-user decode](img/bench-decode.svg)

| Machine | Model | Context | Peregrine | llama.cpp | Peregrine ÷ llama.cpp |
|---|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 512 | 59.4 | 51.0 | **1.16×** |
| Desktop | Llama 3.2 1B | 4096 | 44.2 | 32.4 | **1.36×** |
| Desktop | Qwen 2.5 0.5B | 512 | 68.1 | 75.8 | 0.90× |
| Desktop | Qwen 2.5 0.5B | 4096 | 51.6 | 48.8 | **1.06×** |
| Bluefin | Llama 3.2 1B | 512 | 45.4 | 39.5 | **1.15×** |
| Bluefin | Llama 3.2 1B | 4096 | 36.7 | 13.0 | 2.82×, being checked |
| Bluefin | Qwen 2.5 0.5B | 512 | 52.8 | 74.9 | 0.70× |
| Bluefin | Qwen 2.5 0.5B | 4096 | 45.1 | 41.0 | **1.10×** |

Tokens per second. The bluefin Llama 3.2 1B row at 4096 is left out of the charts until it is
re-measured: llama.cpp fell from 39.5 to 13.0 tok/s between the two contexts there, against 51.0 to
32.4 on the desktop, and the host was using some swap during the run.

Why the 0.5B loses at a short context: with only two KV heads its attention splits into fewer tasks
than there are cores, and a few steps in each layer still run on one thread. Both matter more on a
small model, where each token's weight pass is short.

## Prompt processing

![Prompt processing](img/bench-prefill.svg)

| Machine | Model | Context | Peregrine | llama.cpp | Peregrine ÷ llama.cpp |
|---|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 512 | 173.5 | 262.0 | 0.66× |
| Desktop | Llama 3.2 1B | 4096 | 125.4 | 189.0 | 0.66× |
| Desktop | Qwen 2.5 0.5B | 512 | 270.6 | 435.2 | 0.62× |
| Desktop | Qwen 2.5 0.5B | 4096 | 184.9 | 279.4 | 0.66× |
| Bluefin | Llama 3.2 1B | 512 | 196.3 | 188.3 | **1.04×** |
| Bluefin | Llama 3.2 1B | 4096 | 143.7 | 141.2 | **1.02×** |
| Bluefin | Qwen 2.5 0.5B | 512 | 260.0 | 302.9 | 0.86× |
| Bluefin | Qwen 2.5 0.5B | 4096 | 195.2 | 236.0 | 0.83× |

On the desktop the repacked Q4_K kernels are off by default (AVX-512 has no kernel of its own yet).
Turned on with `PEREGRINE_KREPACK=1`, Llama 3.2 1B goes to 211.5 tok/s at 512 (0.81×) and 144.6 at
4096 (0.77×), and its 10-user throughput to 62.7 tok/s (1.02×), with decode within 1%. The 0.5B
does not gain from them.

## Many users at once

![Total throughput, 10 users at once](img/bench-users10.svg)

| Machine | Model | 4 users | 10 users | 10 users, Peregrine ÷ llama.cpp |
|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 52.9 vs 54.5 | 51.8 vs 61.7 | 0.84× |
| Desktop | Qwen 2.5 0.5B | 70.6 vs 81.8 | 78.0 vs 93.4 | 0.84× |
| Bluefin | Llama 3.2 1B | 53.5 vs 39.5 | 55.5 vs 37.4 | **1.48×** |
| Bluefin | Qwen 2.5 0.5B | 70.1 vs 65.2 | 71.3 vs 68.7 | **1.04×** |

Total tokens per second over all users at a 512-token context, Peregrine vs llama.cpp. At 4096 tokens
both engines fall to 5-14 tok/s in total, too coarse to compare closely; those numbers are in the
result folders.

## What changed in November

![Prompt processing: the November changes](img/bench-speedups.svg)

Peregrine against its own earlier build on the same machine and file:

| Change | Machine | Model | Prompt processing |
|---|---|---|---|
| Repacked Q4_K kernels (#88) | Bluefin | Llama 3.2 1B | 104 → 152 tok/s (+46%) |
| Repacked Q4_K kernels (#88) | Bluefin | Qwen 3.5 2B | 72 → 96 tok/s (+33%) |
| Prefill chunk follows the worker pool (#92) | Desktop | Bonsai 2 2B | +43% |
| Prefill chunk follows the worker pool (#92) | Desktop | Llama 3.2 1B | +19% |
| Prefill chunk follows the worker pool (#92) | Desktop | Bonsai 2 9B | +14% |

Decode also moved by +0.6% to +2.9% with the finer attention split (#94). The prefill chunk and the attention
split leave the output bit for bit the same; the repacked kernels round differently, and their KL
divergence against llama.cpp stayed flat on every model checked.

## Reproducing

```sh
python3 benchmarks/run.py --help     # the harness: one engine, one model, the four numbers
python3 benchmarks/charts.py         # redraws the charts on this page from the numbers in it
```

The charts are drawn by [`benchmarks/charts.py`](../benchmarks/charts.py) from the medians in the
result folders; update its tables when a new result lands, then rerun it.
