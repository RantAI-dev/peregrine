# Bonsai 2 vs PrismML's llama.cpp fork (roadmap S3.2)

The Bonsai 2 speed numbers measured against the only other engine that loads these files.
Upstream llama.cpp v0.9.0 rejects the PQ2_0 tensor type (ggml type 142), so the desktop
baseline in [`../desktop-zen5-2026-11-07/`](../desktop-zen5-2026-11-07/) could not include
Bonsai 2 at all. PrismML's fork accepts PQ2_0 and PTQ1_0, and this folder compares the two
engines on the same bytes, the same host, the same thread count and the same requests.

- Date: 2026-11-08.
- Peregrine: `main` at `bf241fe` (`bench: desktop baseline after the November speed-ups`),
  built `release` (`cargo build --release -p peregrine`), `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- Fork: PrismML `llama.cpp` branch `prism`, pinned commit
  `62c418ccd7aefee37997e94c2d9fac1eed7fc89b` (the commit `docs/reference/bonsai2_spec.md`
  was written against). It fetched and built as pinned; no fallback to the branch head.
  CMake Release, default native CPU dispatch (`-march=native`, AVX-512 on this host).
- Models (SHA-256 taken from the manifests):  - `Ternary-Bonsai-2-2B-PQ2_0.gguf`, 517,312,992 bytes,
    `d5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928`
  - `Ternary-Bonsai-2-9B-PQ2_0.gguf`, 2,416,851,840 bytes,
    `7a89ab48f5038c704480f60781f9b8d4035351f5f6beb7d11fad80001fde23f9`
  - `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf`, 2,675,410,720 bytes,
    `c79fcdfaf6b7bf47089b50dadcfd11f9ce7d3e96adf3bbefde7912e5ef69e22b`
- The manifests record each model as its `models/<dir>/<file>` path inside the checkout
  (`models/` is the local symlink to the model store), not the host path it resolves to.
- Settings: contexts 512 and 4096, 3 interleaved rounds per engine, 256 greedy decode tokens
  per request, server mode, 1 and 10 users at 512, 1 user at 4096, prefix-cache defeat by a
  per-request nonce, prompt lengths calibrated per engine and model (every measured prompt
  within 5% of the target). Threads: **12**, one per physical core, the same count as the
  desktop baseline in PR #98.

The 4B file is not on this host; the only MTP-capable file available here is the 9B
`-MTP-Q8_0` file, so the MTP comparison uses that one.

## The machine

| Property | Value |
|---|---|
| CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M (Zen 5), family 26, model 36 |
| Topology | 1 socket, 12 physical cores, 2 threads per core, 24 logical CPUs |
| SIMD | AVX2, FMA, AVX-512F/DQ/BW/VL/IFMA/VNNI, AVX-VNNI, AVX512-BF16, GFNI, VAES |
| RAM | 11,763 MiB total, 3,072 MiB swap; 9.7 GiB available at the end of the run |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2 (x86_64) |
| Virtualization | WSL2 under the Microsoft hypervisor, full virtualization (AMD-V) |
| CPU governor | unknown (VM or fixed frequency) — WSL2 does not expose it |
| Benchmark threads | 12 |

A note on SIMD tiers, because it is the one place the two engines are not configured
identically: Peregrine's telemetry reports `avx2+fma` for every run here (the same tier the
2026-11-07 baseline reported), while the fork was built with `-march=native` and therefore
has AVX-512 kernels available. If that matters at all, it favours the fork, so it does not
explain Peregrine's margin.

The measured streaming-bandwidth ceiling moved between invocations (32.8 to 44.7 GB/s, the
probe's usual spread on this WSL2 host), so the `% of ceiling` column is not comparable
across folders and is left out of the tables below; the engine-to-engine ratios are measured
within the same invocation and are the numbers to read.

## How the runs were split

The host caps a command at ten minutes, and a full 3-round sweep of the 9B does not fit in
one. Each round is therefore several invocations:

- 512 tokens, 1 user and 10 users: both engines in one invocation where the 9B fork's
  10-user phase allowed it; otherwise one engine per invocation.
- 4096 tokens, 1 user: one engine per invocation (the prompt calibration alone re-prefills
  several 4k prompts).
- The engine order alternates between rounds (Peregrine first in round 1 and 3, the fork
  first in round 2; the 10-user and 4096 measurement order alternates the same way).

The raw per-invocation manifests are in [`raw/`](raw/). [`merge.py`](merge.py) concatenates
their sample arrays into the one-manifest-per-engine shape a single `--runs 3` invocation
would write, recomputing medians and half-ranges with the harness's rule. Run it with
`python3 merge.py` from this directory to regenerate the eight manifests beside it.

Two harness additions were needed for this model family and are in the same PR:

- `--chat-template-kwargs '{"enable_thinking": false}'`, sent to both engines. Bonsai 2 uses
  the Qwen3 template; with thinking on, both servers put the whole answer in
  `reasoning_content`, which the streaming client does not count and which makes the greedy
  text check unreadable. Thinking is off for every measured request on both engines.
- `--token-tolerance 1`. Peregrine occasionally reports 255 of 256 tokens when the last
  generated token is an empty control token (`<|im_end|>`): the server does not count a
  control token that carries no text. The sample is kept, and its throughput is still
  `reported_tokens / reported_decode_time`; the fork reported 256 on every request. This is
  at most a 0.4% difference in the token count, and it is applied to both engines.

## Results

Medians of 3 rounds, spread is the half-range. "Peregrine ÷ fork" above 1 means Peregrine is
faster; for TTFT, below 1 means Peregrine's first token arrives sooner.

### Bonsai 2 2B, PQ2_0

| Context | Users | Metric | Peregrine | Fork | Peregrine ÷ fork |
|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 70.64 ± 0.77 | 32.67 ± 0.61 | **2.16x** |
| 512 | 1 | Prompt tok/s | 259.25 | 112.19 | **2.31x** |
| 512 | 10 | Aggregate tok/s | 80.59 | 29.79 | **2.71x** |
| 512 | 10 | TTFT (s) | 12.92 | 45.40 | **0.28x** |
| 4096 | 1 | Decode tok/s | 60.64 ± 0.59 | 30.76 ± 0.49 | **1.97x** |
| 4096 | 1 | Prompt tok/s | 202.94 | 103.90 | **1.95x** |

### Bonsai 2 9B, PQ2_0

| Context | Users | Metric | Peregrine | Fork | Peregrine ÷ fork |
|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 22.97 ± 0.22 | 9.82 ± 0.09 | **2.34x** |
| 512 | 1 | Prompt tok/s | 67.24 | 26.63 | **2.52x** |
| 512 | 10 | Aggregate tok/s | 21.62 | 7.76 | **2.79x** |
| 512 | 10 | TTFT (s) | 51.66 | 185.48 | **0.28x** |
| 4096 | 1 | Decode tok/s | 20.36 ± 0.22 | 8.93 ± 0.38 | **2.28x** |
| 4096 | 1 | Prompt tok/s | 57.42 | 25.33 | **2.27x** |

### MTP on the 9B `-MTP-Q8_0` file, 512 tokens, 1 user

Peregrine's MTP speculation is on by default and was disabled with `--opt MTP=0`. The fork
enables the same head with `--spec-type draft-mtp` (its `--mtp` flag is a download option,
not a server option); the "off" row is the fork with no speculative flag.

| Configuration | Peregrine decode tok/s | Fork decode tok/s | Peregrine ÷ fork |
|---|---|---|---|
| MTP off | 22.75 ± 0.09 | 9.83 ± 0.07 | 2.31x |
| MTP on | 25.01 ± 0.77 | 9.77 ± 0.53 | 2.56x |
| On ÷ off | **1.10x** | **0.99x** | — |

Peregrine's MTP adds about 10% at this prompt length. The fork's speculative mode runs (its
log says `speculative decoding enabled: draft-mtp`, and its timings report `draft_n` and
`draft_n_accepted`), but at 256 tokens of this corpus it did not move decode beyond the
round-to-round spread. A separate 64-token check on the same file reported 66 drafted and 40
accepted tokens for the fork.

## The models produce the same text

Greedy, `enable_thinking: false`, "Explain in two sentences why the sky looks blue." Both
engines return the same opening sentence, which is what confirms the fork is running these
weights rather than a placeholder path:

2B, Peregrine:

> Sunlight contains all colors of the rainbow, but only short wavelengths (blue) are scattered by air molecules in a process known as Rayleigh scattering. This causes the blue light to be scattered more strongly than the red light, which travels straight through the atmosphere, giving the sky its blue appearance.

2B, fork:

> Sunlight contains all colors of the rainbow, but when it travels through the Earth's atmosphere, shorter wavelengths (blue) are scattered more strongly by gas molecules than longer wavelengths (red). This phenomenon is known as Rayleigh scattering, which is why the sky appears blue during the day.

9B, both engines begin with the same sentence:

> The sky appears blue because sunlight is scattered by the molecules in the Earth's atmosphere, a process known as Rayleigh scattering.

They diverge after that, as two independent implementations of the same architecture with
different numerics do; the benchmark requests use `ignore_eos: true`, so generation continues
past the model's end-of-turn token and the tail of a 256-token reply is not meaningful text.

## What was checked, and what was not

- Checked: both engines load and decode all three files; every measured prompt is within 5%
  of its target length; every phase has 3 samples per engine; the 9B fits at 4096 with one
  user on this host, and 10 users fit at 512; the host did not become memory-bound (9.7 GiB
  still available after the last run).
- Not measured: 10 users at 4096 (not in the requested matrix), 4 users (not requested), any
  other quantisation, and the 4B file (absent).
- The fork's 10-user TTFT at 9B (185 s) is a real number, not a timeout: ten 512-token
  prefills at about 26 tok/s serialise before the first user sees a token.
