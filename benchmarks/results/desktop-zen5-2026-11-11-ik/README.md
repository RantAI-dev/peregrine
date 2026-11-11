# Desktop chat baselines: Peregrine vs llama.cpp vs ik_llama.cpp (and OpenVINO)

The published desktop table in [`docs/benchmark.md`](../../../docs/benchmark.md) compares
Peregrine with llama.cpp only. [`docs/targets.md`](../../../docs/targets.md) Section 2 names
**ik_llama.cpp** (and OpenVINO GenAI) as competitors, so this run adds the fork that is most
often reported faster than upstream on x86 CPUs. Nothing in the engine changed: Peregrine is
`origin/main` at `99eabe9`, exactly as fetched.

- Date: 2026-11-11, 14:00–19:00 local (UTC+7).
- Host: AMD Ryzen AI 9 HX 370 (Zen 5), 12 physical cores / 24 logical, AVX-512
  (F/VBMI/VNNI/BF16), WSL2, Linux `6.18.33.2-microsoft-standard-WSL2`, 11,763 MiB RAM,
  3,072 MiB swap. No cgroup or governor control is exposed under WSL2.
- Method: `benchmarks/run.py --mode server`, 12 threads, 256 greedy tokens per request
  (`ignore_eos`), prompts calibrated per engine/model to the target context within 5%
  (records show 1–2%), a unique nonce line on every request to defeat the prefix cache, three
  interleaved rounds, median ± half-range.
- Host quiet before each invocation (`uptime` 1-minute load below 1, no benchmark process
  left running). All commands were run in the foreground under a 10-minute cap, so a model's
  rows are split over several invocations (see "How the runs were split").

## Engines

| Engine | Version / commit | Build |
|---|---|---|
| Peregrine | `origin/main` `99eabe9d9edb1fee43410d12b82de03c085e9280` | `cargo build --release -p peregrine`, rustc 1.99.1 (58cf8f9c3 2026-10-13) |

`origin/main` advanced to `af7eb09` while these runs were in progress; the measured tree stayed
at `99eabe9` so that every row in this folder is one binary, and the rebuilt post-reboot binary
is the same commit.
| llama.cpp | v0.9.0, commit `cd959ba` | harness cache build, CMake Release AVX2/FMA, `llama-server` |
| ik_llama.cpp | `master` `479074ccac2ac94f06b6d24a9ce26f0f9e9aa6ae` (2026-11-11 08:54:37 +0200) | CMake Release, `-DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON` |

The ik_llama.cpp build is the one its `docs/build.md` requires on AVX-512; a plain Release
build silently falls back to AVX2. Verification: the configure step printed
`Using optimized iqk matrix multiplications` and `Enabling IQK Flash Attention kernels`, the
server banner printed `HAVE_FANCY_SIMD is defined` and `AVX512_VNNI = 1`, and
`objdump -d build/ggml/src/libggml.so | grep -c vpdpbusd` returns 30,615. At load with `-rtr`
the server printed `Repacked 169 tensors` for the Qwen 0.5B file.

### ik_llama.cpp settings (best documented CPU settings)

| Setting | Value | Why |
|---|---|---|
| `-rtr` / `--run-time-repack` | **on** | `docs/parameters.md`: repacks tensors that have an interleaved kernel. On these stock k-quant files it turns `Q4_K→Q4_K_R4`, `Q5_0→Q5_0_R4`, `Q6_K→Q6_K_R4`, `Q8_0→Q8_0_R8` in memory at load, without touching the file. |
| `-fa` | **on** (the fork's default) | `docs/parameters.md` default; its own CPU flash-attention kernels. Measured against `-fa off` (below). |
| `-t` | 12 | One per physical core; also the fork's own default on this host (`--help` prints `default: 12`). |
| `-b` / `-ub` | defaults (2048 / 512) | `docs/parameters.md`: "safe to leave default" on CPU. |
| `-np` | exact phase user count | Same slot policy as llama.cpp: `-c ctx*slots`, `-np slots`; the fork has no `--kv-unified`. |
| KV cache | f16 (default) | No `--no-kv-offload`-style option is relevant with `-ngl 0` (CPU only). |

Setting check with `llama-bench -r 2` on the same Qwen 0.5B file (median tok/s):

| Context | Test | default (`-fa on`, no `-rtr`) | `-rtr` | `-rtr -fa off` |
|---|---|---|---|---|
| 512 | pp512 | 967.5 | **1056.4** | 968.7 |
| 512 | tg64 | 98.3 | 103.9 | 108.0 |
| 4096 | pp4096 | — | **670.6** | 643.5 |
| 4096 | tg64 | — | 101.1 | 98.5 |

`-rtr` is a clear win for prompt processing and a small win for decode, so it is on for every
row. `-fa on` (the fork's default) is used; at 512 `-fa off` decodes ~4% faster but pre-fills
~8% slower, and at 4096 the default is faster on both. The comparison therefore gives the fork
its documented default, not a setting chosen to make it lose.

Some gains in the fork's own benchmarks come from its `_R4`/`_R8` **quantisation types**
(`iqk_quants`), which need re-quantisation. This run does **not** use them: every engine reads
the same stock GGUF file (SHA-256 below), and `-rtr` only changes the in-memory layout.

## Models

Same files as the other desktop runs where they still exist, SHA-256 verified against the
manifests in `benchmarks/results/`:

| Model | Bytes | SHA-256 | Note |
|---|---|---|---|
| Llama 3.2 1B Instruct Q4_K_M | 807,694,464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` | matches the 2026-11-08/09 desktop manifests |
| Qwen 2.5 0.5B Instruct Q4_K_M | 491,400,032 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` | matches the published desktop manifests |
| SmolLM2 360M Instruct Q4_K_M | 270,590,880 | `2fa3f013dcdd7b99f9b237717fa0b12d75bbb89984cc1274be1471a465bac9c2` | **the file on disk today differs from the bluefin manifest's** (`ade7aee1…`, 270,590,976 bytes); this row is a different file and is labelled as such |
| gemma-3-1b-it Q4_K_M | 806,058,496 | `12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d` | no earlier desktop manifest carries a gemma hash; this is the file in `models/` |

The published desktop table itself has only Llama 1B and Qwen 0.5B; SmolLM2 and gemma-3 were
measured here because the task asks for them, not because an old desktop row exists.

## How the runs were split

Each command is capped at ten minutes, and a host-quiet check (`load < 1`) precedes each one.
Interleaving is kept inside every invocation (`run.py` runs A, B, C per round), and the 512
sweep was split by user count so that each command stays inside the cap:

- 512: one invocation per model with users 1 and 4, one with users 10, engines
  `peregrine,llama_cpp,ik_llama_cpp`, `--runs 3`.
- 512 llama.cpp sensitivity: `--engines llama_cpp --llama-fa off`, one invocation per pair of
  models, same method.
- 4096, 1 user: one invocation per model, all three engines, `--runs 3` (fits in 3–6 min).
- 4096, 4 users and 10 users: a full three-engine, three-round invocation does not fit in the
  cap at this context (one 10-user 4096 round is 2–9 min depending on the engine). Those rows
  were therefore measured **one round per engine**, and for 1B the peregrine and llama arms
  shared one invocation while ik ran separately. The JSON files record `runs: 1` and the tables
  below mark `n=1`; their spread column is zero by construction and they are not as strong as
  the `n=3` rows.
- **A host reboot at 17:36 wiped `/tmp` and the worktree.** The 512 results had already been
  committed and pushed; the first pass of 4096 single-user measurements was lost and re-run
  after rebuilding both engines at the same pinned commits (18:01 onward). Only the `n=3`
  4096 single-user rows in this folder are from that second pass; nothing was merged from the
  lost pass.

`run.py` was changed to give every phase a fresh free port (`pick_free_port`). ik_llama.cpp
does not set `SO_REUSEADDR`, so a fixed port reused by the next engine can fail to bind while
the previous server's connections are in TIME_WAIT; a stale server on the shared port would
also silently answer the next engine's requests. Neither changes what is measured.

## 512-token context

Medians of three interleaved rounds. "llama.cpp faster" is whichever of `-fa auto` (on) and
`-fa off` measured faster for that row (higher tok/s, lower TTFT); the ratio is Peregrine ÷
that number, so above 1 means Peregrine is faster. ik_llama.cpp is `-rtr -fa on`.

### Llama 3.2 1B Instruct Q4_K_M (`6f85a640…`)

| Metric | Peregrine | llama.cpp `auto` | llama.cpp `off` | llama faster | ik_llama.cpp | P ÷ llama | P ÷ ik |
|---|---|---|---|---|---|---|---|
| Decode tok/s | 56.57 ± 2.18 | 48.60 ± 4.00 | 46.70 ± 1.36 | auto 48.60 | 54.60 ± 0.85 | **1.16×** | 1.04× |
| Prompt tok/s | 274.05 ± 12.07 | 265.51 ± 2.36 | 237.50 ± 9.98 | auto 265.51 | 526.81 ± 8.09 | **1.03×** | 0.52× |
| 4 users, total tok/s | 85.64 ± 4.49 | 62.58 ± 4.14 | 41.97 ± 1.81 | auto 62.58 | 71.89 ± 4.71 | **1.37×** | 1.19× |
| 10 users, total tok/s | 85.11 ± 0.94 | 63.07 ± 2.78 | 35.77 ± 0.63 | auto 63.07 | 58.52 ± 2.22 | **1.35×** | 1.45× |
| 10 users, TTFT (s) | 10.46 ± 1.26 | 15.24 ± 0.28 | 22.58 ± 4.16 | auto 15.24 | 13.20 ± 1.97 | **0.69× (faster)** | 0.79× (faster) |

### Qwen 2.5 0.5B Instruct Q4_K_M (`74a4da8c…`)

| Metric | Peregrine | llama.cpp `auto` | llama.cpp `off` | llama faster | ik_llama.cpp | P ÷ llama | P ÷ ik |
|---|---|---|---|---|---|---|---|
| Decode tok/s | 93.89 ± 4.06 | 73.80 ± 1.68 | 67.15 ± 3.16 | auto 73.80 | 91.92 ± 0.80 | **1.27×** | 1.02× |
| Prompt tok/s | 470.91 ± 29.49 | 423.28 ± 5.69 | 390.48 ± 12.58 | auto 423.28 | 1116.90 ± 40.42 | **1.11×** | 0.42× |
| 4 users, total tok/s | 113.70 ± 4.26 | 97.26 ± 8.85 | 65.75 ± 6.63 | auto 97.26 | 102.75 ± 17.67 | **1.17×** | 1.11× |
| 10 users, total tok/s | 150.43 ± 1.79 | 92.75 ± 2.86 | 55.43 ± 4.95 | auto 92.75 | 100.36 ± 0.28 | **1.62×** | 1.50× |
| 10 users, TTFT (s) | 6.30 ± 1.23 | 9.46 ± 1.20 | 12.89 ± 3.71 | auto 9.46 | 6.46 ± 0.56 | **0.67× (faster)** | 0.97× (faster) |

### SmolLM2 360M Instruct Q4_K_M (`2fa3f013…`)

| Metric | Peregrine | llama.cpp `auto` | llama.cpp `off` | llama faster | ik_llama.cpp | P ÷ llama | P ÷ ik |
|---|---|---|---|---|---|---|---|
| Decode tok/s | 97.85 ± 4.04 | 78.01 ± 2.98 | 69.87 ± 0.25 | auto 78.01 | 92.44 ± 2.73 | **1.25×** | 1.06× |
| Prompt tok/s | 426.48 ± 8.04 | 434.06 ± 5.83 | 398.00 ± 6.15 | auto 434.06 | 1030.59 ± 43.70 | 0.98× | 0.41× |
| 4 users, total tok/s | 115.09 ± 9.40 | 85.86 ± 11.36 | 60.25 ± 4.27 | auto 85.86 | 97.26 ± 14.34 | **1.34×** | 1.18× |
| 10 users, total tok/s | 132.02 ± 2.55 | 90.98 ± 5.00 | 49.69 ± 1.00 | auto 90.98 | 74.84 ± 5.16 | **1.45×** | 1.76× |
| 10 users, TTFT (s) | 7.95 ± 0.28 | 9.46 ± 1.25 | 14.97 ± 1.09 | auto 9.46 | 7.32 ± 1.26 | **0.84× (faster)** | 1.09× (slower) |

### gemma-3-1b-it Q4_K_M (`12bf0fff…`)

| Metric | Peregrine | llama.cpp `auto` | llama.cpp `off` | llama faster | ik_llama.cpp | P ÷ llama | P ÷ ik |
|---|---|---|---|---|---|---|---|
| Decode tok/s | 54.33 ± 2.37 | 42.13 ± 1.45 | 44.17 ± 0.87 | off 44.17 | 54.02 ± 0.74 | **1.23×** | 1.01× |
| Prompt tok/s | 254.30 ± 7.87 | 235.00 ± 3.59 | 234.85 ± 3.78 | auto 235.00 | 644.72 ± 29.44 | **1.08×** | 0.39× |
| 4 users, total tok/s | 63.43 ± 5.58 | 53.30 ± 2.82 | 49.12 ± 0.39 | auto 53.30 | 74.57 ± 5.69 | **1.19×** | 0.85× |
| 10 users, total tok/s | 76.28 ± 0.58 | 55.53 ± 1.24 | 46.65 ± 0.24 | auto 55.53 | 72.67 ± 1.73 | **1.37×** | 1.05× |
| 10 users, TTFT (s) | 12.69 ± 1.20 | 20.23 ± 1.17 | 23.33 ± 0.21 | auto 20.23 | 9.48 ± 1.30 | **0.63× (faster)** | 1.34× (slower) |

## 4096-token context

Medians of three interleaved rounds where `n=3`; single-round rows are marked `n=1`. At 4096
llama.cpp was **not** re-measured with `-fa off` (the sweep hit the ten-minute cap before it
reached 4096); the comparison uses `-fa auto`, which is on and was the faster setting for every
512 throughput row except gemma decode. ik_llama.cpp is `-rtr -fa on`.

### Llama 3.2 1B Instruct Q4_K_M (`6f85a640…`)

| Metric | Peregrine | llama.cpp `auto` | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 40.64 ± 0.52 | 33.62 ± 0.51 | 42.26 ± 0.81 | **1.21×** | 0.96× | 3 |
| Prompt tok/s | 208.41 ± 3.52 | 186.87 ± 1.43 | 358.95 ± 3.04 | **1.12×** | 0.58× | 3 |
| 4 users, total tok/s | 11.53 | 9.42 | 5.67 | **1.22×** | **2.03×** | 1 |
| 10 users, total tok/s | 11.77 | 8.83 | not completed | **1.33×** | — | 1 |
| 10 users, TTFT (s) | 122.09 | 137.20 | not completed | **0.89× (faster)** | — | 1 |

### Qwen 2.5 0.5B Instruct Q4_K_M (`74a4da8c…`)

| Metric | Peregrine | llama.cpp `auto` | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 65.32 ± 5.88 | 48.73 ± 0.55 | 69.05 ± 1.42 | **1.34×** | 0.95× | 3 |
| Prompt tok/s | 379.11 ± 5.66 | 290.55 ± 0.70 | 662.86 ± 2.14 | **1.30×** | 0.57× | 3 |
| 4 users, total tok/s | 20.01 | 14.71 | 9.96 | **1.36×** | **2.01×** | 1 |
| 10 users, total tok/s | 20.20 | 13.74 | 6.53 | **1.47×** | **3.09×** | 1 |
| 10 users, TTFT (s) | 71.24 | 87.60 | 149.86 | **0.81× (faster)** | **0.48× (faster)** | 1 |

### SmolLM2 360M Instruct Q4_K_M (`2fa3f013…`)

| Metric | Peregrine | llama.cpp `auto` | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 57.80 ± 1.20 | 41.95 ± 1.07 | 55.52 ± 1.16 | **1.38×** | **1.04×** | 3 |
| Prompt tok/s | 304.53 ± 2.84 | 257.89 ± 3.80 | 555.98 ± 14.99 | **1.18×** | 0.55× | 3 |
| 4 users, total tok/s | not measured | not measured | not measured | — | — | 0 |
| 10 users, total tok/s | not measured | not measured | not measured | — | — | 0 |
| 10 users, TTFT (s) | not measured | not measured | not measured | — | — | 0 |

### gemma-3-1b-it Q4_K_M (`12bf0fff…`)

| Metric | Peregrine | llama.cpp `auto` | ik_llama.cpp | P ÷ llama | P ÷ ik | n |
|---|---|---|---|---|---|---|
| Decode tok/s | 47.45 ± 2.06 | 39.72 ± 0.31 | 53.43 ± 1.76 | **1.19×** | 0.89× | 3 |
| Prompt tok/s | 281.16 ± 1.99 | 222.94 ± 0.21 | 569.42 ± 5.56 | **1.26×** | 0.49× | 3 |
| 4 users, total tok/s | not measured | not measured | not measured | — | — | 0 |
| 10 users, total tok/s | not measured | not measured | not measured | — | — | 0 |
| 10 users, TTFT (s) | not measured | not measured | not measured | — | — | 0 |

## What the 4096 rows say

- Against llama.cpp, Peregrine wins every measured 4096 row: decode 1.19–1.38×, prompt
  1.12–1.30×, 4 users 1.22–1.36×, 10 users 1.33–1.47×, TTFT at 10 users 0.81–0.89×.
- **The depth decode gap against ik_llama.cpp closes.** ik decodes 42.3 vs 40.6 tok/s on the
  1B (Peregrine 0.96×), 69.1 vs 65.3 on the 0.5B (0.95×), 55.5 vs 57.8 on SmolLM2 (1.04×),
  and 53.4 vs 47.5 on gemma (**0.89×**, the largest decode loss in this report). At 512
  Peregrine was at parity or ahead on the same files, so the fork's repacked GEMMs hold up
  better as context grows.
- **ik_llama.cpp's multi-user path does not.** At 4 users Peregrine is 2.0× its aggregate
  (11.5 vs 5.7 on the 1B, 20.0 vs 10.0 on the 0.5B), and at 10 users 3.1× (20.2 vs 6.5 on the
  0.5B) with TTFT 71 s vs 150 s. Its 1B 10-user round did not finish inside the ten-minute
  command cap (10 concurrent 4096-token prefills plus 2,560 decode tokens); that row is
  recorded as not completed rather than estimated.
- Prompt processing remains the fork's win: 0.49–0.58× Peregrine at 4096 (358 vs 208 on the
  1B, 663 vs 379 on the 0.5B).

## What the 512 rows say

- **Against llama.cpp, Peregrine wins every throughput row except SmolLM2 prompt
  (0.98×)**; decode is 1.16–1.27×, prompt 1.01–1.11×, 4 users 1.17–1.37×, 10 users
  1.35–1.62×, and TTFT at 10 users is 0.63–0.84× (Peregrine earlier). `-fa auto` (on) is
  the faster llama.cpp setting for every 512 row except gemma decode, where `-fa off` is 4.8%
  faster (1.23× vs 1.29× against the default).
- **ik_llama.cpp is a different shape.** Its prompt processing is 1.9–2.6× Peregrine's
  (1117 vs 471 tok/s on the 0.5B, 1031 vs 426 on SmolLM2, 645 vs 254 on gemma, 527 vs 274 on
  the 1B). That is the fork's IQK/repacked GEMM path and it is the largest gap in this table.
- **ik_llama.cpp loses the multi-user rows** to Peregrine at 512 except gemma 4 users
  (0.85×) and gemma 10 users (1.05×, inside the run-to-run range): 1B 1.19×/1.45×, 0.5B
  1.11×/1.50×, SmolLM2 1.18×/1.76×. Its decode is within 1–6% of Peregrine on three models
  and 1% behind on gemma.
- **TTFT at 10 users:** Peregrine beats llama.cpp on all four (0.63–0.84×); against
  ik_llama.cpp it wins on 1B (0.79×), 0.5B (0.97×) and loses on SmolLM2 (1.09×) and gemma
  (1.34×). The gemma loss tracks the fork's shorter per-slot prefill and its 10-user aggregate
  being close to Peregrine's.

## What was not measured

- **OpenVINO GenAI (step 4): not attempted.** It is the second priority in the brief and the
  ik_llama.cpp sweep is not complete (below), so no `openvino_genai` adapter or IR export was
  produced. No OpenVINO number appears anywhere in this folder.
- ik_llama.cpp, Llama 3.2 1B, 4096 tokens, 10 users: one round did not finish inside the
  ten-minute command cap (slowest engine on the slowest model/context combination). Not
  estimated.
- SmolLM2 and gemma-3, 4096 tokens, 4 and 10 users: not measured (time). Their 512 multi-user
  rows are complete.
- llama.cpp `-fa off` at 4096: not measured. The 512 arms show `-fa auto` (on) faster for
  every throughput row except gemma decode, and the published 2026-11-08 desktop run also
  found `-fa off` slower at 4096, so `auto` is the competitor used at 4096; this is a
  like-for-like choice supported by the 512 arms, not a measured 4096 arm.
- 16k context: out of scope for this run (the published table is 512/4096).

## Measurement notes and artefacts

- The host ran 8–10% slower than the 2026-11-09 session for **both** established engines
  (Peregrine 1B decode 62.6 → 56.6, llama.cpp 51.8 → 48.6), so compare ratios, not absolute
  tok/s across days. The same-day ratios are the like-for-like quantity.
- Two single-user `ttft_s` samples in the SmolLM2 llama.cpp arm are negative (`-1.24 s`) and
  one ik sample is `-0.23 s`. That is a wall-clock jump in WSL2 (`time.time()` moving
  backwards), not a real TTFT; medians are unaffected because only one of three samples is
  affected, and TTFT at 10 users (the reported number) is computed from the concurrent
  runner. Flagged rather than smoothed.
- The `bandwidth_ceiling_gb_s` probe read 42.6 GB/s in this session (29–48 across sessions on
  this host), so `% of ceiling` is not used here.
- SmolLM2 is a different file from the one in the bluefin manifest (see Models); its numbers
  are not comparable with the bluefin SmolLM2 rows.

## Files

- `report.md`: the narrative comparison and the wins/losses list.
- `2026-11-11-*.json`: raw run sets, one per engine per invocation, with the machine record,
  engine builds, settings, every raw sample and the derived medians.
- `summarize.py`: rebuilds the tables above from the JSON (concatenates split samples).
