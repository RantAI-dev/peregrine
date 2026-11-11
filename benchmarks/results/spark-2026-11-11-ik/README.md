# DGX Spark chat baselines: Peregrine vs llama.cpp vs ik_llama.cpp

The published ARM table in [`docs/benchmark.md`](../../../docs/benchmark.md) compares
Peregrine with llama.cpp only. The desktop run
[`desktop-zen5-2026-11-11-ik`](../desktop-zen5-2026-11-11-ik/README.md) added the
ik_llama.cpp fork on x86, where its repacked GEMMs process prompts about twice as fast as
Peregrine. This run measures the same thing on the project's ARM host: four Q4_K_M chat
models, 512- and 4096-token contexts, single-user decode and prefill, 4 and 10 concurrent
users, and time to first token at 10 users. Nothing in the engine changed.

- Date: 2026-11-11, 20:45–23:45 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters:
  10 × Cortex-X925 (3.9 GHz max) and 10 × Cortex-A725 (2.8 GHz max), 121 GB unified
  memory, one NUMA node; `asimd`, `asimddp`, `i8mm`, `bf16`, `sve`, `sve2`.
  Linux `6.17.0-1008-nvidia`, governor `performance`, kernel tier `aarch64-neon-i8mm`.
  The harness's CPU probe reports `cpu_model = unknown` on this kernel (it reads
  `/proc/cpuinfo` model names, which this host does not fill in); the cluster layout is
  from the earlier `spark-arm-2026-11-09` runs, which identified the same machine.
- Peregrine: `origin/main` at `487f0a2cf42200b05192dff423784d6ea2605648`, `cargo build
  --release -p peregrine`, rustc 1.99.1 (58cf8f9c3 2026-10-13). `origin/main` advanced to
  `3030172` while the runs were in progress; every row here is the one binary at `487f0a2`.
- llama.cpp: v0.9.0, commit `cd959ba`, the harness cache build
  (`~/.cache/peregrine-bench/llama.cpp-v0.9.0`). It is a CUDA-capable build, so the GPU was
  kept out of every row twice: `CUDA_VISIBLE_DEVICES=""` in the environment and `-ngl 0`
  on the server. `llama-bench` defaults to no GPU layers.
- ik_llama.cpp: `master` at `479074ccac2ac94f06b6d24a9ce26f0f9e9aa6ae` (2026-11-11
  08:54:37 +0200), built CPU-only, Release, with `-rtr` and its own flash attention on.
- Host quiet before each invocation. The 1-minute load average was usually below 5
  (our own benchmark threads dominate the 5- and 15-minute averages while a run is in
  progress); no other benchmark process was running. All commands ran in the foreground
  under the ten-minute cap, so the grid is split across invocations (see "How the runs
  were split").

## Engines and settings

| Engine | Version / commit | Build |
|---|---|---|
| Peregrine | `origin/main` `487f0a2cf42200b05192dff423784d6ea2605648` | `cargo build --release -p peregrine`, rustc 1.99.1 |
| llama.cpp | v0.9.0, `cd959ba` | harness cache build, CMake Release, `llama-server`, CPU only |
| ik_llama.cpp | `master` `479074ccac2ac94f06b6d24a9ce26f0f9e9aa6ae` | CMake Release, CPU only, IQK multiply/FA compiled in |

### ik_llama.cpp: ARM build flags

`docs/build.md` requires `GGML_NATIVE=ON` for native tuning and names `GGML_ARCH_FLAGS`
as the fallback when `-march=native` does not propagate the ARM feature macros. On this
host it does not: GCC 13.3 does not know Cortex-X925 (`0xd87`), so `-march=native`
silently resolves to the baseline `armv8-a` and defines neither `__ARM_FEATURE_DOTPROD`
nor `__ARM_FEATURE_MATMUL_INT8`. The build therefore used the documented fallback:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=OFF \
    -DGGML_ARCH_FLAGS="-march=armv9-a+dotprod+i8mm+bf16+sve2"
cmake --build build --config Release -j 18
```

This is what native tuning means for this CPU on this compiler. The configure step printed
`Using optimized iqk matrix multiplications`, `Enabling IQK Flash Attention kernels` and
`Including all IQK FA kernels`; `ARM detected`; the effective C++ flags are the
`-march=armv9-a+dotprod+i8mm+bf16+sve2` above. The server banner at load reports:

```
NEON = 1 | SVE = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1
HAVE_FANCY_SIMD is NOT defined
```

`HAVE_FANCY_SIMD` is the fork's x86 AVX-512 path; on ARM it is expected to be absent, and
the fork's ARM kernels are the NEON/i8mm ones. The server was run with `-rtr` (run-time
repack, its documented CPU setting) and its default flash attention (`-fa on`), `-ngl 0`,
`-t 10`, one slot per phase.

### ik_llama.cpp: `-rtr` check (its documented setting, measured with and without)

`llama-bench`, 3 repetitions, 10 threads, 64 decode tokens, median tok/s:

| Model | Context | no `-rtr` pp / tg | `-rtr` pp / tg |
|---|---|---|---|
| Llama 3.2 1B | 512 | 869.1 / 97.6 | **980.8 / 99.2** |
| Llama 3.2 1B | 4096 | 810.2 / 94.5 | **890.7 / 97.3** |
| Qwen 2.5 0.5B | 512 | **2484.1** / 148.0 | 2006.1 / **166.0** |
| Qwen 2.5 0.5B | 4096 | 1632.9 / 151.0 | **1662.6 / 158.6** |

`-rtr` helps decode on both files and both contexts, and helps prompt processing on the
1B; on the Qwen 0.5B file at 512 it costs about 19% of prompt throughput in `llama-bench`.
It is the fork's documented setting and the server's own 512-token prompt rows (measured
separately below) are as high as the no-`-rtr` `llama-bench` number, so every row uses
`-rtr`. The Qwen 512 prompt row is the one where the competitor's best prompt setting
is arguable; even there Peregrine is 0.39×, so the conclusion does not depend on it.

### llama.cpp: flash attention

`llama-bench` on this host, 3 repetitions, 10 threads, 64 decode tokens, median tok/s:

| Model | Context | `-fa auto` (on) pp / tg | `-fa off` pp / tg |
|---|---|---|---|
| Llama 3.2 1B | 512 | **289.2 / 114.8** | 274.1 / 107.7 |
| Llama 3.2 1B | 4096 | **241.1 / 112.0** | 212.2 / 105.2 |
| Qwen 2.5 0.5B | 512 | **562.7 / 206.3** | 527.4 / 193.8 |
| Qwen 2.5 0.5B | 4096 | **449.2 / 207.2** | 392.4 / 192.5 |

Unlike the desktop, on this ARM host flash attention on is faster for both prompt and
decode on both files and both contexts, so every llama.cpp row is `-fa auto` (on), the
build's own default. No `-fa off` arm was measured through the server.

## Method

`benchmarks/run.py --mode server` with 256 greedy tokens per request (`ignore_eos`),
prompts calibrated per engine/model to the target context within 5% (records show 0–2%),
a unique nonce line on every request to defeat the prefix cache, three interleaved rounds
(A, B, C per round), median ± half-range. Servers are restarted per (context, user) phase
on a fresh free port. Cells that would not fit the ten-minute command cap are recorded as
not measured rather than estimated.

**Thread count.** Every engine was swept at 10, 16 and 20 threads on the two 512-token
single-user rows (`llama-bench -p 512 -n 128 -r 5`, `peregrine bench --prompt-tokens 512
--tokens 128`, medians; tok/s):

| Model | Threads | Peregrine pp / tg | llama.cpp pp / tg | ik_llama.cpp pp / tg |
|---|---|---|---|---|
| Llama 3.2 1B | 10 | 633.8 / 120.5 | 287.0 / 111.7 | **1075.7 / 102.1** |
| Llama 3.2 1B | 16 | **706.6 / 121.5** | 332.7 / 95.1 | 679.0 / 64.3 |
| Llama 3.2 1B | 20 | 642.7 / 111.8 | **365.0** / 99.4 | 827.2 / 81.8 |
| Qwen 2.5 0.5B | 10 | 903.0 / 199.5 | 569.2 / **206.4** | 2252.6 / **169.8** |
| Qwen 2.5 0.5B | 16 | 974.8 / 211.4 | 651.4 / 190.8 | 1409.4 / 131.7 |
| Qwen 2.5 0.5B | 20 | **1143.8** / 209.8 | **709.8** / 156.7 | 1681.8 / 135.4 |

The **main table uses 10 threads for every engine** — the width of the Cortex-X925
performance cluster, and the count at which both competitors decode fastest. Peregrine's
prefill thread pool was pinned to the same 10 (`--opt decode_threads=10`), so the number
in the table is a real 10-thread number, not a 20-thread decode beside a 10-thread
prefill. Two engines prefer another count on some rows and this is a note, not a change to
the table: Peregrine's prefill is 5–27% faster at 16–20 threads (most on Qwen 0.5B),
llama.cpp's prefill is 15–24% faster at 20, and ik_llama.cpp is clearly fastest at 10.
At 20 threads ik decodes 20–25% slower than at 10 on this asymmetric machine.

## Models

| Model | Bytes | SHA-256 |
|---|---|---|
| Llama 3.2 1B Instruct Q4_K_M | 807,694,464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |
| Qwen 2.5 0.5B Instruct Q4_K_M | 397,808,192 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| SmolLM2 360M Instruct Q4_K_M | 270,590,976 | `ade7aee1e5bd7c3b8f1fedafc61a1a0be31e8afa64e006607cd202323e2ccce0` |
| gemma-3-1b-it Q4_K_M | 806,058,496 | `12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d` |

The Qwen file is the ARM benchmarks' upload (`6eb923e7…`, 397,808,192 bytes), not the
desktop table's `74a4da8c…` (491,400,032 bytes). SmolLM2 is the pinned oracle file from
`docs/dev/testing.md` (`ade7aee1…`), the one the bluefin manifests also use.

## How the runs were split

Each command is capped at ten minutes; the grid is split so that every invocation stays
inside it, while keeping the three engines interleaved inside each invocation:

- 512 tokens: one invocation per model with users 1 and 4, one with users 10, engines
  `peregrine,llama_cpp,ik_llama_cpp`, `--runs 3`.
- 4096 tokens, single user: one invocation per model, all three engines, `--runs 3`.
- 4096 tokens, 4 and 10 users: one invocation per engine per model, `--runs 1`, because a
  three-engine three-round invocation at that context does not fit the cap. Those rows
  are marked `n=1`; their spread is zero by construction and they are weaker evidence
  than the `n=3` rows.

The first 512/4096 pass resolved `llama-server` from `PATH` (a local
`v0.9.0-dev` build, `0c6dca2`) instead of the pinned `v0.9.0` (`cd959ba`). Every llama.cpp
number below is from a re-measurement with `LLAMA_SERVER_BIN` pinned to the harness cache
build; the whole 512 and 4096 single-user grid was re-run with all three engines, and the
4096 multi-user llama arm was re-run with the same binary. No row mixes the two binaries.

## 512-token context

Medians of three interleaved rounds, 10 threads, 256 greedy tokens. `P ÷ llama` and
`P ÷ ik` are Peregrine over that engine, so above 1 means Peregrine is faster. For TTFT
lower is better, and the ratio is still Peregrine over the competitor.

### Llama 3.2 1B Instruct Q4_K_M

| Metric | Peregrine | llama.cpp | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 119.76 ± 0.69 | 88.06 ± 0.56 | 90.20 ± 1.21 | **1.36×** | **1.33×** | 3 |
| Prompt tok/s | 646.90 ± 6.03 | 280.52 ± 3.53 | 1204.53 ± 13.49 | **2.31×** | 0.54× | 3 |
| 4 users, total tok/s | 144.13 ± 0.46 | 72.37 ± 0.44 | 79.54 ± 1.03 | **1.99×** | **1.81×** | 3 |
| 10 users, total tok/s | 154.66 ± 1.75 | 74.82 ± 0.05 | 91.84 ± 1.92 | **2.07×** | **1.68×** | 3 |
| 10 users, TTFT (s) | 6.21 ± 0.05 | 14.67 ± 1.91 | 5.41 ± 1.14 | **0.42× (faster)** | 1.15× (slower) | 3 |

### Qwen 2.5 0.5B Instruct Q4_K_M

| Metric | Peregrine | llama.cpp | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 196.31 ± 13.88 | 138.75 ± 2.83 | 125.77 ± 4.59 | **1.41×** | **1.56×** | 3 |
| Prompt tok/s | 943.84 ± 7.17 | 551.66 ± 6.17 | 2446.67 ± 89.12 | **1.71×** | 0.39× | 3 |
| 4 users, total tok/s | 226.82 ± 0.65 | 128.78 ± 0.85 | 217.77 ± 18.07 | **1.76×** | **1.04×** | 3 |
| 10 users, total tok/s | 242.76 ± 2.70 | 141.50 ± 0.32 | 165.80 ± 11.88 | **1.72×** | **1.46×** | 3 |
| 10 users, TTFT (s) | 4.16 ± 0.03 | 7.43 ± 0.97 | 3.02 ± 0.51 | **0.56× (faster)** | 1.38× (slower) | 3 |

### SmolLM2 360M Instruct Q4_K_M

| Metric | Peregrine | llama.cpp | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 247.63 ± 3.16 | 148.50 ± 4.09 | 203.21 ± 5.87 | **1.67×** | **1.22×** | 3 |
| Prompt tok/s | 1031.60 ± 32.48 | 590.76 ± 9.47 | 2396.01 ± 804.54 | **1.75×** | 0.43× | 3 |
| 4 users, total tok/s | 264.30 ± 0.47 | 145.20 ± 0.57 | 228.17 ± 13.82 | **1.82×** | **1.16×** | 3 |
| 10 users, total tok/s | 285.20 ± 3.44 | 158.07 ± 0.11 | 108.91 ± 3.11 | **1.80×** | **2.62×** | 3 |
| 10 users, TTFT (s) | 3.60 ± 0.04 | 6.89 ± 0.46 | 3.82 ± 0.23 | **0.52× (faster)** | **0.94× (faster)** | 3 |

### gemma-3-1b-it Q4_K_M

| Metric | Peregrine | llama.cpp | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 110.00 ± 0.09 | 77.94 ± 0.92 | 71.78 ± 0.97 | **1.41×** | **1.53×** | 3 |
| Prompt tok/s | 535.90 ± 5.87 | 295.90 ± 0.74 | 1393.89 ± 13.45 | **1.81×** | 0.38× | 3 |
| 4 users, total tok/s | 126.51 ± 0.64 | 73.28 ± 0.19 | 104.43 ± 5.83 | **1.73×** | **1.21×** | 3 |
| 10 users, total tok/s | 133.89 ± 0.85 | 78.78 ± 0.15 | 135.50 ± 4.54 | **1.70×** | 0.99× | 3 |
| 10 users, TTFT (s) | 7.75 ± 0.05 | 17.63 ± 0.01 | 3.70 ± 0.69 | **0.44× (faster)** | 2.10× (slower) | 3 |

## 4096-token context

Medians of three interleaved rounds where `n=3`; the 4- and 10-user rows are one round
per engine and marked `n=1`.

### Llama 3.2 1B Instruct Q4_K_M

| Metric | Peregrine | llama.cpp | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 94.42 ± 0.70 | 60.07 ± 1.55 | 66.59 ± 3.41 | **1.57×** | **1.42×** | 3 |
| Prompt tok/s | 407.98 ± 1.19 | 234.34 ± 0.44 | 831.52 ± 2.43 | **1.74×** | 0.49× | 3 |
| 4 users, total tok/s | 21.68 | 12.15 | 9.94 | **1.78×** | **2.18×** | 1 |
| 10 users, total tok/s | 22.23 | 12.17 | 8.15 | **1.83×** | **2.73×** | 1 |
| 10 users, TTFT (s) | 65.03 | 108.07 | 94.43 | **0.60× (faster)** | **0.69× (faster)** | 1 |

### Qwen 2.5 0.5B Instruct Q4_K_M

| Metric | Peregrine | llama.cpp | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 156.50 ± 1.50 | 90.71 ± 3.53 | 105.11 ± 2.61 | **1.73×** | **1.49×** | 3 |
| Prompt tok/s | 664.63 ± 1.51 | 441.05 ± 0.71 | 1572.31 ± 1.35 | **1.51×** | 0.42× | 3 |
| 4 users, total tok/s | 35.20 | 21.84 | 16.49 | **1.61×** | **2.13×** | 1 |
| 10 users, total tok/s | 36.31 | 21.36 | 13.47 | **1.70×** | **2.70×** | 1 |
| 10 users, TTFT (s) | 40.20 | 57.95 | 52.55 | **0.69× (faster)** | **0.77× (faster)** | 1 |

### SmolLM2 360M Instruct Q4_K_M

| Metric | Peregrine | llama.cpp | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 157.50 ± 4.53 | 91.26 ± 4.77 | 119.10 ± 1.53 | **1.73×** | **1.32×** | 3 |
| Prompt tok/s | 617.22 ± 4.78 | 451.35 ± 2.42 | 1340.27 ± 120.89 | **1.37×** | 0.46× | 3 |
| 4 users, total tok/s | 32.98 | 22.61 | 15.05 | **1.46×** | **2.19×** | 1 |
| 10 users, total tok/s | 33.52 | 21.28 | 10.76 | **1.58×** | **3.12×** | 1 |
| 10 users, TTFT (s) | 42.98 | 57.44 | 65.54 | **0.75× (faster)** | **0.66× (faster)** | 1 |

### gemma-3-1b-it Q4_K_M

| Metric | Peregrine | llama.cpp | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 107.25 ± 0.11 | 66.37 ± 1.97 | 71.12 ± 1.63 | **1.62×** | **1.51×** | 3 |
| Prompt tok/s | 531.27 ± 0.11 | 283.40 ± 0.15 | 1325.22 ± 4.87 | **1.87×** | 0.40× | 3 |
| 4 users, total tok/s | 28.29 | 15.54 | 16.69 | **1.82×** | **1.69×** | 1 |
| 10 users, total tok/s | 28.46 | 15.67 | 14.20 | **1.82×** | **2.00×** | 1 |
| 10 users, TTFT (s) | 51.25 | 100.36 | 51.49 | **0.51× (faster)** | **1.00× (faster, 0.5%)** | 1 |

## Where Peregrine is behind

Every throughput row against llama.cpp is a Peregrine win: decode 1.36–1.73×, prompt
1.37–2.31×, 4 users 1.46–1.99×, 10 users 1.58–2.07×. Against ik_llama.cpp:

- **Prompt processing, every one of the eight cells, is behind by 0.38–0.54×**:
  Llama 1B 0.54× at 512 and 0.49× at 4096, Qwen 0.5B 0.39×/0.42×, SmolLM2 0.43×/0.46×,
  gemma 0.38×/0.40×. This is the fork's repacked IQK GEMM, the same gap the desktop run
  measured (0.39–0.58×); on ARM it is 1.9–2.6× Peregrine's prompt rate.
- **10-user TTFT at 512 is behind on three of four models**: Llama 1B 6.21 s vs 5.41 s
  (1.15× slower), Qwen 0.5B 4.16 s vs 3.02 s (1.38×), gemma 7.75 s vs 3.70 s (2.10×).
  SmolLM2 is the exception at 3.60 s vs 3.82 s (0.94×, Peregrine faster). The gemma loss
  is the largest single gap in the table; the fork's strong per-slot prefill gets all ten
  slots to their first token sooner.
- **gemma 512, 10 users total throughput is 0.99×** (133.89 vs 135.50 tok/s): 1.2%
  behind, inside the ik arm's ±4.5 tok/s spread, so this is parity rather than a loss.
- **gemma 4096, 10-user TTFT is 1.00×** (51.25 vs 51.49 s): parity, 0.5% ahead.

Multi-user aggregate throughput is otherwise ahead of ik_llama.cpp by 1.04–3.12×: its
single-slot path is strong, but at 4 and 10 users it loses the batching that Peregrine and
llama.cpp share. The 4096 multi-user rows are `n=1`, so the exact ratios there are
weaker evidence than the 512 `n=3` rows.

## What was not measured

- 16k context: out of scope; the published table is 512/4096.
- No cell hit the ten-minute cap. Every planned cell finished; the only weaker rows are
  the 4096 4/10-user rows, intentionally `n=1` to stay inside the cap.
- llama.cpp `-fa off` through the server: not measured. The `llama-bench` check above
  already shows `-fa auto` faster on this host for both files and both contexts, so the
  server rows use the faster setting.
- ik_llama.cpp without `-rtr` through the server: not measured as a full arm; the
  `llama-bench` check above is the sensitivity, and the server used the documented `-rtr`.

## Files

- `README.md`: this report.
- `2026-11-11-*.json`: raw run sets, one per engine per invocation, with the machine
  record, engine builds, settings, every sample and the calibrated prompt lengths.
- `summarize.py`: rebuilds the tables from the JSON (concatenates split samples and takes
  medians/half-ranges).
