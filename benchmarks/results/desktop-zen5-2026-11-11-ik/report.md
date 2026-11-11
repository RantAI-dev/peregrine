# Desktop chat baselines: ik_llama.cpp and llama.cpp against Peregrine

**Date:** 2026-11-11. **Host:** Ryzen AI 9 HX 370, 12 physical cores, AVX-512, WSL2
(Linux 6.18.33.2-microsoft-standard-WSL2). **Peregrine:** `origin/main` `99eabe9`, release
build with rustc 1.99.1. **llama.cpp:** v0.9.0 `cd959ba` (harness cache build).
**ik_llama.cpp:** `master` `479074ccac2ac94f06b6d24a9ce26f0f9e9aa6ae` (2026-11-11 08:54:37
+0200), built for AVX-512 with its IQK kernels enabled.

The brief compared the four headline numbers per model and context: single-user decode,
prompt processing, total throughput at 4 and 10 users, and time to first token at 10 users.
Method is the published one: server mode, 12 threads, 256 greedy tokens per request,
per-engine prompt calibration to the target context (all measured prompts within 2%), a
per-request nonce that defeats the prefix cache, interleaved rounds, medians.

## Engine settings

ik_llama.cpp ran with its documented CPU performance settings, not a neutral default:

- `-rtr` (run-time repack): the stock Q4_K/Q5_0/Q6_K/Q8_0 tensors are converted in memory to
  the fork's interleaved `_R4`/`_R8` layouts at load; the files are unchanged.
- `-fa on`: the fork's own default and its CPU flash-attention kernels.
- `-t 12` (also the fork's default on this host), default `-b 2048 -ub 512`, f16 KV cache,
  `-c ctx*slots -np slots`.
- Build: CMake Release with `-DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON
  -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON`. A plain Release build silently falls back to
  AVX2 on this host. Verified: `HAVE_FANCY_SIMD is defined`, `AVX512_VNNI = 1`, 30,615
  `vpdpbusd` instructions in `libggml.so`, `Repacked 169 tensors` at load.

A setting check on the same Qwen 0.5B file (`llama-bench`, median of 2) shows `-rtr` is worth
keeping: pp512 967 → 1056 tok/s, tg64 98 → 104; at 4096 `-rtr` gives pp4096 671 and tg 101.
`-fa off` is 4% faster at 512 decode but 8% slower at 512 prompt and slower at 4096, so the
fork's default is the fair competitor.

llama.cpp was measured with both flash-attention settings at 512 and the faster one is used
per row (`-fa auto`, i.e. on, wins every 512 row except gemma decode). The same files are used
by all three engines; the only ik-specific layout change is `-rtr`, done in memory.

## Head-to-head, ratios (Peregrine ÷ competitor)

Above 1.00 means Peregrine is faster; for TTFT below 1.00 means Peregrine is faster.
`n=1` rows are one round, no spread.

### 512-token context

| Model | Decode | Prompt | 4 users | 10 users | TTFT 10u |
|---|---|---|---|---|---|
| **Llama 3.2 1B** vs llama.cpp | **1.16×** | **1.03×** | **1.37×** | **1.35×** | **0.69×** |
| Llama 3.2 1B vs ik_llama.cpp | 1.04× | 0.52× | 1.19× | 1.45× | 0.79× |
| **Qwen 2.5 0.5B** vs llama.cpp | **1.27×** | **1.11×** | **1.17×** | **1.62×** | **0.67×** |
| Qwen 2.5 0.5B vs ik_llama.cpp | 1.02× | 0.42× | 1.11× | 1.50× | 0.97× |
| **SmolLM2 360M** vs llama.cpp | **1.25×** | 0.98× | **1.34×** | **1.45×** | **0.84×** |
| SmolLM2 360M vs ik_llama.cpp | 1.06× | 0.41× | 1.18× | 1.76× | 1.09× |
| **gemma-3-1b** vs llama.cpp | **1.23×** | **1.08×** | **1.19×** | **1.37×** | **0.63×** |
| gemma-3-1b vs ik_llama.cpp | 1.01× | 0.39× | 0.85× | 1.05× | 1.34× |

### 4096-token context

| Model | Decode | Prompt | 4 users | 10 users | TTFT 10u |
|---|---|---|---|---|---|
| **Llama 3.2 1B** vs llama.cpp | **1.21×** | **1.12×** | **1.22×** (n=1) | **1.33×** (n=1) | **0.89×** (n=1) |
| Llama 3.2 1B vs ik_llama.cpp | 0.96× | 0.58× | **2.03×** (n=1) | — | — |
| **Qwen 2.5 0.5B** vs llama.cpp | **1.34×** | **1.30×** | **1.36×** (n=1) | **1.47×** (n=1) | **0.81×** (n=1) |
| Qwen 2.5 0.5B vs ik_llama.cpp | 0.95× | 0.57× | **2.01×** (n=1) | **3.09×** (n=1) | **0.48×** (n=1) |
| **SmolLM2 360M** vs llama.cpp | **1.38×** | **1.18×** | not measured | not measured | not measured |
| SmolLM2 360M vs ik_llama.cpp | **1.04×** | 0.55× | not measured | not measured | not measured |
| **gemma-3-1b** vs llama.cpp | **1.19×** | **1.26×** | not measured | not measured | not measured |
| gemma-3-1b vs ik_llama.cpp | 0.89× | 0.49× | not measured | not measured | not measured |

Absolute medians, spreads and per-setting numbers are in the tables in `README.md`; the raw
JSON files have every sample.

## Where Peregrine wins, and where it loses

**Wins against llama.cpp.** Every measured throughput row at both contexts (decode
1.16–1.38×, prompt 1.03–1.30×, 4 users 1.17–1.37×, 10 users 1.33–1.62×) except SmolLM2
prompt at 512 (0.98×, inside the ±8 tok/s spread). TTFT at 10 users is faster on every row
where it was measured (0.63–0.89×).

**Wins against ik_llama.cpp.** Multi-user throughput: 1.11–1.76× at 512, and 2.0–3.1× at
4096, where the fork's scheduler degrades badly with ten 4k-token prefills. TTFT at 10 users
at 4096 is 0.48× (Peregrine 71 s vs 150 s on the 0.5B). Single-user decode is at parity
(1.01–1.06×) on three of four models at 512 and on SmolLM2 at 4096 (1.04×).

**Losses to ik_llama.cpp.**

- **Prompt processing, everywhere and decisively: 0.39–0.58×.** 645 vs 254 tok/s (gemma 512),
  1117 vs 471 (Qwen 512), 1031 vs 426 (SmolLM2 512), 527 vs 274 (1B 512); at 4096 the same
  shape (663 vs 379, 556 vs 305, 569 vs 281, 359 vs 208). This is the fork's repacked IQK
  GEMM path and it is the single largest gap in the report.
- **Decode at 4096: 0.89–0.96×.** 47.5 vs 53.4 tok/s on gemma, 40.6 vs 42.3 on the 1B, 65.3
  vs 69.1 on the 0.5B. At 512 Peregrine is 1.01–1.06×, so the fork gains with depth on these
  files while Peregrine's lead erodes.
- **gemma-3 4 users at 512: 0.85×** (63.4 vs 74.6 tok/s). The only multi-user row Peregrine
  loses at 512.
- **TTFT at 10 users on SmolLM2 512 (1.09×) and gemma 512 (1.34×).** The fork pre-fills each
  slot faster; Peregrine's better aggregate throughput does not always buy a lower first token.

## What looks like an artefact, and what could not be measured

- **Host drift.** Both established engines ran 8–10% below the 2026-11-09 desktop session
  (Peregrine 1B 512 decode 62.6 → 56.6; llama.cpp 51.8 → 48.6). Ratios are the same-day
  quantity; absolute tok/s should not be compared across days.
- **WSL2 wall-clock jumps.** A few single-user `ttft_s` samples are negative (minimum
  −1.24 s) because `time.time()` moved backwards between request start and first chunk.
  Medians of three are unaffected; the 10-user TTFT is from the concurrent runner. Two
  load-average readings at command start were residual from the immediately preceding
  benchmark, not competing work; the host was checked for processes and listeners before
  every invocation.
- **A host reboot at 17:36 wiped `/tmp`.** The 512 results had been pushed; the first 4096
  pass was lost and re-run after rebuilding. All 4096 `n=3` rows here are from the second
  pass. The reboot is also why some JSON files predate and some postdate it.
- **Not measured:** OpenVINO GenAI (second priority; not attempted because the ik sweep is
  not fully complete and the session ran out of room), 4096 multi-user for SmolLM2 and
  gemma-3, ik_llama.cpp 1B 4096 10 users (one round exceeded the ten-minute cap),
  llama.cpp `-fa off` at 4096, and 16k context.
- SmolLM2's current file (`2fa3f013…`, 270,590,880 B) differs from the file in the bluefin
  manifest (`ade7aee1…`, 270,590,976 B), so its numbers are not comparable with bluefin.
