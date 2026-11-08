# Bluefin (H2): after the Q5_0 speed-ups

The bluefin suite re-measured on the build that carries **#103** (wide Q5_0 prefill), **#104**
(eight-row Q5_0/Q8_0 decode GEMV) and **#105** (repacked Q5_0 tiles). It adds SmolLM2 360M to the
two models from the earlier bluefin runs, and measures llama.cpp with both flash-attention settings.

The earlier same-day folder, [`bluefin-zen2-2026-11-08-fa-off`](../bluefin-zen2-2026-11-08-fa-off/README.md),
is the baseline this one is compared against. Every row below was measured again; no number is carried
over from it.

- Date: 2026-11-08, 22:27-03:01 local (15:27-20:01 UTC). The three 10-user, 4096-token, `-fa off`
  attempts were stopped at the 10-minute cap and are recorded as not completed.
- Host: AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads, AVX2/FMA, 13.5 GiB RAM, 8 GiB swap.
  Benchmark threads: 4 (one per physical core). Governor `powersave`, unchanged.
- Peregrine: commit `09214de7b6fe61db381b42512a01094e9e9acbc8`, built from
  `origin/perf-q5-0-repack` with `cargo build --release -p peregrine`, `rustc 1.99.1 (58cf8f9c3
  2026-10-13)`. **PR #105 was still open when the runs started** (`gh pr view 105 --json state`
  returned `OPEN`; its head was `09214de`), so `origin/main` `77638b2` did not contain it. #105
  merged as `b86f355` at 15:27 UTC while the session was running, and `origin/main` has since moved to
  `ec56a92` (#108, #110, #111); those later commits are not in these numbers. The measured tree is
  the tree #105 squashed to.
- llama.cpp: v0.9.0 (`cd959ba`), prebuilt at `/var/tmp/llama.cpp/build/bin` (CMake Release,
  `GGML_NATIVE=ON`). Same binary as the earlier bluefin runs.
- Models, all from `~/models`, SHA-256 recorded in every manifest:
  - `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807694464 bytes,
    `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` — Q4_K/Q6_K only, no Q5_0
    tensors (it is the control for today's kernels);
  - `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, 397808192 bytes,
    `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` — 132 of 169 tensors are
    Q5_0;
  - `SmolLM2-360M-Instruct.Q4_K_M.gguf`, 270590976 bytes,
    `ade7aee1e5bd7c3b8f1fedafc61a1a0be31e8afa64e006607cd202323e2ccce0` — 176 of 274 tensors are
    Q5_0.
- Settings: contexts 512 and 4096, 3 interleaved rounds per engine, 256 greedy decode tokens per
  request, server mode, prefix-cache defeat by per-request nonce, prompts calibrated per engine and
  model (every measured prompt within 5% of target).

The "Q4_K_M" Qwen 2.5 0.5B and SmolLM2 files are mixed-format GGUFs: most of their tensors are Q5_0,
with Q4_K, Q6_K and Q8_0 for the rest. That is why today's Q5_0 kernels move them and leave the
all-Q4_K/Q6_K Llama 1B essentially unchanged.

## The policy

For every published row:

1. measure llama.cpp with its default (`-fa auto`, which resolves to flash attention on here) and
   with `-fa off`;
2. compare Peregrine against whichever of the two is **faster on that row**;
3. show the default-setting number only as a secondary note.

`-fa off` wins single-user decode; the default wins prompt processing and multi-user throughput on
this host. The harness records the setting in every manifest (`settings.llama_flash_attn`,
`engine.flash_attn`), so a run cannot be misread.

## Method and command splitting

The 512-token phases ran as one `-r 3` invocation per model and setting. At 4096 tokens that would
exceed the 10-minute command cap, so the phases were split by user count and, where needed, by round;
the per-round samples were merged into one manifest per engine and model with the harness's own
medians and half-range spreads, exactly the shape a single `-r 3` invocation writes. The raw
per-round manifests are not committed; the merged JSONs carry every individual sample.

The only change to llama.cpp's command line is the flash-attention flag:

```sh
LLAMA_SERVER_BIN=/var/tmp/llama.cpp/build/bin/llama-server \
python3 benchmarks/run.py -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -e llama_cpp -c 4096 -r 1 -n 256 -t 4 --llama-fa off \
  --user-counts 1,4 -o /tmp/bench-q5/raw-llama-off
```

At 4096 tokens with `-fa off`, the 10-user phase did not finish within the cap on any of the three
models (each attempt was stopped at 9.5 minutes). Those rows are **not completed** and carry no
estimate; the default setting is faster there and is the one used in the ratio.

## Quiet host

The host ran its usual background (`kswapd0` about 2-3% CPU, `localsearch-3` about 2%, session
processes about 1% total). Load average before the first command was 1.90 (1 min), 3.08 (5 min);
after the last it was 1.26 and 2.91. Swap was about 4.1 GiB used of 8191 MiB and did not grow during
the session. No process was stopped or killed for these runs.

## Results

`Peregrine ÷ llama.cpp` is against the faster setting named in the "faster" column. Medians, tok/s
unless noted; spreads are in the JSON manifests.

### Llama 3.2 1B Instruct (Q4_K_M)

| Context | Metric | Peregrine | llama.cpp default | llama.cpp `-fa off` | Faster | P ÷ faster |
|---|---|---|---|---|---|---|
| 512 | Single decode | 46.83 | 39.49 | **42.50** | off | **1.10×** |
| 512 | Prompt processing | 195.67 | **188.09** | 174.69 | default | **1.04×** |
| 512 | 4 users total | 53.73 | **39.61** | 34.00 | default | **1.36×** |
| 512 | 10 users total | 55.53 | **37.74** | 23.79 | default | **1.47×** |
| 4096 | Single decode | 36.81 | 13.05 | **29.25** | off | **1.26×** |
| 4096 | Prompt processing | 144.95 | **141.37** | 123.00 | default | **1.03×** |
| 4096 | 4 users total | 7.69 | **5.38** | 3.11 | default | **1.43×** |
| 4096 | 10 users total | 7.71 | **5.07** | not completed | default | **1.52×** |

### Qwen 2.5 0.5B Instruct (Q4_K_M)

| Context | Metric | Peregrine | llama.cpp default | llama.cpp `-fa off` | Faster | P ÷ faster |
|---|---|---|---|---|---|---|
| 512 | Single decode | 83.48 | 74.61 | **77.69** | off | **1.07×** |
| 512 | Prompt processing | 434.78 | **303.73** | 278.14 | default | **1.43×** |
| 512 | 4 users total | 104.33 | **65.73** | 54.86 | default | **1.59×** |
| 512 | 10 users total | 108.41 | **68.36** | 42.44 | default | **1.59×** |
| 4096 | Single decode | 65.89 | 41.12 | **52.20** | off | **1.26×** |
| 4096 | Prompt processing | 277.58 | **236.46** | 194.12 | default | **1.17×** |
| 4096 | 4 users total | 14.75 | **10.91** | 5.03 | default | **1.35×** |
| 4096 | 10 users total | 14.28 | **10.20** | not completed | default | **1.40×** |

### SmolLM2 360M Instruct (Q4_K_M)

| Context | Metric | Peregrine | llama.cpp default | llama.cpp `-fa off` | Faster | P ÷ faster |
|---|---|---|---|---|---|---|
| 512 | Single decode | 108.92 | 80.71 | **94.45** | off | **1.15×** |
| 512 | Prompt processing | 443.19 | **318.91** | 284.46 | default | **1.39×** |
| 512 | 4 users total | 114.18 | **66.15** | 54.30 | default | **1.73×** |
| 512 | 10 users total | 116.76 | **67.63** | 37.38 | default | **1.73×** |
| 4096 | Single decode | 63.79 | 20.49 | **47.51** | off | **1.34×** |
| 4096 | Prompt processing | 228.45 | **217.05** | 172.53 | default | **1.05×** |
| 4096 | 4 users total | 11.99 | **7.74** | 3.83 | default | **1.55×** |
| 4096 | 10 users total | 12.11 | **6.74** | not completed | default | **1.80×** |

### Time to first token at 10 users (s, lower is better)

| Model | Context | Peregrine | llama.cpp default | llama.cpp `-fa off` | Faster | P ÷ faster |
|---|---|---|---|---|---|---|
| Llama 3.2 1B Instruct | 512 | 18.80 | **21.76** | 35.43 | default | **0.86×** (P faster) |
| Llama 3.2 1B Instruct | 4096 | 193.00 | **184.80** | not completed | default | **1.04×** (P slower) |
| Qwen 2.5 0.5B Instruct | 512 | 9.12 | **13.68** | 22.31 | default | **0.67×** (P faster) |
| Qwen 2.5 0.5B Instruct | 4096 | 104.21 | **122.22** | not completed | default | **0.85×** (P faster) |
| SmolLM2 360M Instruct | 512 | 8.63 | **13.01** | 25.20 | default | **0.66×** (P faster) |
| SmolLM2 360M Instruct | 4096 | 122.82 | **128.22** | not completed | default | **0.96×** (P faster) |

## What changed today

All three changes are in the measured tree:

- **#103 — wide Q5_0 prefill kernel.** Prompt processing for Q5_0 weights with a 16-row-wide tile.
- **#104 — eight-row Q5_0/Q8_0 decode GEMV.** Less serial work per layer in the decode matrix-vector
  path.
- **#105 — repacked Q5_0 tiles.** Q5_0 weights repacked into the tile layout the wide kernels read.

Against the earlier same-day run in
[`bluefin-zen2-2026-11-08-fa-off`](../bluefin-zen2-2026-11-08-fa-off/README.md), Peregrine against
itself:

| Model | Context | Decode | Prompt | 10 users |
|---|---|---|---|---|
| Qwen 2.5 0.5B (mostly Q5_0) | 512 | 52.87 → 83.48 (+58%) | 263.19 → 434.78 (+65%) | 71.72 → 108.41 (+51%) |
| Qwen 2.5 0.5B (mostly Q5_0) | 4096 | 45.20 → 65.89 (+46%) | 198.23 → 277.58 (+40%) | 9.81 → 14.28 (+46%) |
| Llama 3.2 1B (no Q5_0) | 512 | 46.06 → 46.83 (+1.7%) | 197.34 → 195.67 (-0.8%) | 55.18 → 55.53 (+0.6%) |
| Llama 3.2 1B (no Q5_0) | 4096 | 36.85 → 36.81 (-0.1%) | 144.59 → 144.95 (+0.2%) | 7.73 → 7.71 (-0.3%) |

SmolLM2 360M is new in this folder, so it has no earlier bluefin number; its ratios above are against
llama.cpp.

## What the numbers say

- **The Qwen 0.5B loss on bluefin is gone.** The earlier bluefin run had it at 0.68× decode at 512
  and 0.87× at 4096 against llama.cpp's faster setting. With the Q5_0 kernels it is 1.07× and 1.26×,
  and it also leads prompt processing (1.43× at 512, 1.17× at 4096) and multi-user throughput.
- **The 1B is unchanged, as it should be.** Its file has no Q5_0 tensors, and every 1B row moved by
  less than 2% against the earlier same-day run. The 1B's wins are the pre-existing ones: 1.10× and
  1.26× decode, 1.36-1.52× multi-user, and prompt processing at parity (1.03-1.04×).
- **SmolLM2 360M follows the Q5_0 pattern.** It is ahead on every throughput row, by 1.15× and 1.34×
  on single-user decode, 1.39× and 1.05× on prompt processing, and 1.73-1.80× with 10 users.
- **Time to first token at 10 users is mixed.** At 512 tokens Peregrine is faster on all three
  models (0.66-0.86×). At 4096 tokens it is slightly slower on the 1B (1.04×) and faster on the two
  small models (0.85× and 0.96×) once prefill is included.

## Files

Nine merged manifests, one per engine, model and (for llama.cpp) flash-attention setting:

- `2026-11-08-153153-llama.cpp-Llama-3.2-1B-Instruct-Q4_K_M-fa-auto.json`
- `2026-11-08-153730-llama.cpp-Llama-3.2-1B-Instruct-Q4_K_M-fa-off.json`
- `2026-11-08-184946-peregrine-Llama-3.2-1B-Instruct-Q4_K_M.json`
- `2026-11-08-152412-llama.cpp-Qwen2.5-0.5B-Instruct-Q4_K_M-fa-auto.json`
- `2026-11-08-152723-llama.cpp-Qwen2.5-0.5B-Instruct-Q4_K_M-fa-off.json`
- `2026-11-08-185349-peregrine-Qwen2.5-0.5B-Instruct-Q4_K_M.json`
- `2026-11-08-154517-llama.cpp-SmolLM2-360M-Instruct.Q4_K_M-fa-auto.json`
- `2026-11-08-154824-llama.cpp-SmolLM2-360M-Instruct.Q4_K_M-fa-off.json`
- `2026-11-08-185559-peregrine-SmolLM2-360M-Instruct.Q4_K_M.json`

Each carries the machine record, model SHA-256, engine build, settings, every raw sample, and notes
describing the command split.
