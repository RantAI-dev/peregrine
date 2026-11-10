# Bonsai 2 vs PrismML's llama.cpp fork, after the November 8-9 speed-ups (roadmap S3.2)

The Bonsai 2 speed numbers measured against the only other engine that loads these files. Upstream
llama.cpp v0.9.0 rejects the PQ2_0 tensor type (ggml type 142), so the desktop llama.cpp comparison
cannot include Bonsai 2 at all. PrismML's fork accepts PQ2_0 and PTQ1_0, and this folder compares
the two engines on the same bytes, the same host, the same thread count and the same requests. The
previous run is [`../2026-11-08-bonsai2-prism/`](../2026-11-08-bonsai2-prism/).

- Date: 2026-11-09 (runs 20:00-23:45 local, UTC+7).
- Peregrine: `origin/main` at `b88fcd7` (`perf: AVX-512 attention kernels (#121)`), built
  `release` (`cargo build --release -p peregrine`), `rustc 1.99.1 (58cf8f9c3 2026-10-13)`. The
  measured tree was that commit plus a test-only golden file change; the built binary is
  byte-identical to the one built from `b88fcd7` (SHA-256
  `5cba085372b724fd0f153443b605831efe98283ba9f2663bfd125542351da73d`).
- Fork: PrismML `llama.cpp` branch `prism`, pinned commit
  `62c418ccd7aefee37997e94c2d9fac1eed7fc89b`. CMake Release, default native CPU dispatch
  (`-march=native`, AVX-512 on this host).
- Models (SHA-256; all three match the 2026-11-08 run byte for byte):
  - `Ternary-Bonsai-2-2B-PQ2_0.gguf`, 517,312,992 bytes,
    `d5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928`
  - `Ternary-Bonsai-2-9B-PQ2_0.gguf`, 2,416,851,840 bytes,
    `7a89ab48f5038c704480f60781f9b8d4035351f5f6beb7d11fad80001fde23f9`
  - `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf`, 2,675,410,720 bytes,
    `c79fcdfaf6b7bf47089b50dadcfd11f9ce7d3e96adf3bbefde7912e5ef69e22b`
- Settings: contexts 512 and 4096, 3 interleaved rounds per engine, 256 greedy decode tokens per
  request, server mode, 1 and 10 users at 512, 1 user at 4096, prefix-cache defeat by a
  per-request nonce, prompt lengths calibrated per engine and model (every measured prompt within
  5% of the target). Threads: **12**, one per physical core.
- Requests use `--chat-template-kwargs '{"enable_thinking": false}'` on both engines and
  `--token-tolerance 1`, the same harness settings as the 2026-11-08 run.

## The machine

| Property | Value |
|---|---|
| CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M (Zen 5), family 26, model 36 |
| Topology | 1 socket, 12 physical cores, 2 threads per core, 24 logical CPUs |
| SIMD | AVX2, FMA, AVX-512F/DQ/BW/VL/IFMA/VNNI, AVX-VNNI, AVX512-BF16, GFNI, VAES |
| RAM | 11,763 MiB total, 3,072 MiB swap |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2 (x86_64), glibc 2.39 |
| Virtualization | WSL2 under the Microsoft hypervisor, full virtualization (AMD-V) |
| CPU governor | unknown (VM or fixed frequency) - WSL2 does not expose it |
| Benchmark threads | 12 |
| Streaming-bandwidth probe | 35.5-50.3 GB/s across the invocations |

The probe's spread on this WSL2 host is wide, so the `% of ceiling` column is not carried over and
the engine-to-engine ratios are the numbers to read. `peregrine tier` prints
`kernel tier x86_64-avx2-avx512-vnni-gfni-vbmi`; PQ2_0 runs its 16-row repacked-tile kernel built on
AVX-512 VNNI and GFNI (`MatPath::Tiles`), so the comparison is AVX-512 against AVX-512, not AVX2
against AVX-512.

## How the runs were split

The host caps each command at ten minutes, and a full 3-round sweep of the 9B does not fit in one:

- 2B, 512 tokens, 1 and 10 users: both engines in one invocation; 4096 tokens, 1 user: both engines
  in one invocation.
- 9B, 512 tokens, 1 user: both engines in one invocation; 10 users: Peregrine in one invocation,
  and the fork as **three one-round invocations** (one round of the fork's 10-user phase takes about
  7 minutes). 4096 tokens, 1 user: one engine per invocation, and the fork as three one-round
  invocations (about 8 minutes per round).
- The flash-attention sweeps are fork-only (Peregrine has no such switch).

The raw per-invocation manifests are in [`raw/`](raw/). [`merge.py`](merge.py) concatenates their
sample arrays into the one-manifest-per-engine shape a single `--runs 3` invocation would write,
recomputing medians and half-ranges with the harness's rule. Run it with `python3 merge.py` from
this directory to regenerate the manifests beside it.

## Results

Medians of 3 rounds, +- half-range. "Fork best" is the faster of the fork's effective
flash-attention settings for that row (lower for TTFT); the ratio is Peregrine / fork best. "Fork
default" is the fork at its own default, `-fa auto`, which resolves to on. Above 1 means Peregrine
is faster; for TTFT, below 1 means Peregrine's first token arrives sooner.

### Bonsai 2 2B, PQ2_0

| Context | Users | Metric | Peregrine | Fork best (-fa) | Ratio | Fork default |
|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 75.14 ± 1.77 | 34.89 ± 0.49 (off) | **2.15x** | 34.45 |
| 512 | 1 | Prompt tok/s | 270.24 ± 10.84 | 119.35 ± 2.18 (on) | **2.26x** | 119.35 |
| 512 | 10 | Aggregate tok/s | 75.77 ± 1.23 | 29.56 ± 0.43 (on) | **2.56x** | 29.56 |
| 512 | 10 | TTFT (s) | 14.10 ± 0.23 | 44.38 ± 1.17 (on) | **0.32x** | 44.38 |
| 4096 | 1 | Decode tok/s | 64.60 ± 2.87 | 32.65 ± 0.27 (off) | **1.98x** | 32.54 |
| 4096 | 1 | Prompt tok/s | 247.30 ± 1.65 | 116.35 ± 1.76 (off) | **2.13x** | 113.46 |

### Bonsai 2 9B, PQ2_0

| Context | Users | Metric | Peregrine | Fork best (-fa) | Ratio | Fork default |
|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 25.47 ± 0.13 | 10.92 ± 0.01 (off) | **2.33x** | 10.75 |
| 512 | 1 | Prompt tok/s | 71.65 ± 0.66 | 28.45 ± 1.08 (on) | **2.52x** | 28.45 |
| 512 | 10 | Aggregate tok/s | 21.66 ± 0.06 | 7.87 ± 0.10 (on) | **2.75x** | 7.87 |
| 512 | 10 | TTFT (s) | 52.12 ± 0.26 | 181.96 ± 3.47 (on) | **0.29x** | 181.96 |
| 4096 | 1 | Decode tok/s | 22.69 ± 0.22 | 10.12 ± 0.14 (on) | **2.24x** | 10.12 |
| 4096 | 1 | Prompt tok/s | 63.04 ± 0.47 | 27.85 ± 0.15 (off) | **2.26x** | 27.48 |

### MTP on the 9B `-MTP-Q8_0` file, 512 tokens, 1 user

Peregrine's MTP speculation is on by default and was disabled with `--opt MTP=0`. The fork enables
the same head with `--spec-type draft-mtp` (its `--mtp` flag is a download option, not a server
option); the "off" row is the fork with no speculative flag.

| Configuration | Peregrine | Fork | Ratio |
|---|---|---|---|
| MTP off | 25.14 ± 0.62 | 10.62 ± 0.05 | **2.37x** |
| MTP on | 28.80 ± 0.96 | 9.33 ± 0.37 | **3.09x** |
| On / off | **1.15x** | 0.88x | - |

Peregrine's MTP adds 15% at this prompt length. The fork's speculative mode runs (its log says
`speculative decoding enabled: draft-mtp`), but it did not beat its own non-speculative decode on
this corpus; treat the fork's MTP as a small loss and the 2.37x-3.09x against Peregrine as the
measured result.

## Flash attention: on, off and the default

Upstream llama.cpp showed a large depth penalty from its default flash attention on the bluefin
1B (13.0 tok/s at 4096 with the default, 29.1 with `-fa off`). That is **not** what this fork does
to Bonsai 2. Explicit `-fa on` and `-fa off` land within a few percent on every server row, and the
default (`-fa auto`) is the on setting:

| Row | Fork `-fa on` | Fork `-fa off` |
|---|---|---|
| 2B, 512, 1 user, decode tok/s | 34.77 | 34.89 |
| 2B, 512, 1 user, prompt tok/s | 117.04 | 117.63 |
| 2B, 512, 10 users, aggregate tok/s | 29.97 | 28.72 |
| 9B, 512, 1 user, decode tok/s | 10.35 | 10.92 |
| 9B, 512, 1 user, prompt tok/s | 27.93 | 28.17 |

The explicit-on runs were measured for 512 tokens on both models; their agreement with `-fa auto`
(2B decode 34.77 vs 34.45, prompt 117.04 vs 119.35; 9B decode 10.35 vs 10.75, prompt 27.93 vs
28.45) is the check that the default column is the on setting. The 4096 rows use `-fa auto` for the
on side for the same reason.

## What moved against the 2026-11-08 run

Both engines moved on this host between the two runs, so the ratios are the stable comparison:

| Row | 2026-11-08 ratio | 2026-11-09 ratio |
|---|---|---|
| 2B, 512, decode | 2.16x | 2.15x |
| 2B, 512, prompt | 2.40x | 2.26x |
| 2B, 512, 10 users | 2.69x | 2.56x |
| 2B, 4096, decode | 1.99x | 1.98x |
| 2B, 4096, prompt | 1.93x | 2.13x |
| 9B, 512, decode | 2.33x | 2.33x |
| 9B, 512, prompt | 2.57x | 2.52x |
| 9B, 512, 10 users | 2.76x | 2.75x |
| 9B, 4096, decode | 2.21x | 2.24x |
| 9B, 4096, prompt | 2.25x | 2.26x |

Peregrine's own numbers rose across the board: 2B decode 70.64 -> 75.14 (+6%) and prompt 259.25 ->
270.24 (+4%) at 512, prompt 202.94 -> 247.30 (+22%) at 4096; 9B decode 22.97 -> 25.47 (+11%) and
prompt 57.42 -> 63.04 (+10%) at 4096. Its 10-user 2B row slipped 6% (80.59 -> 75.77) while the 9B
10-user row was flat (21.62 -> 21.66). The fork also moved (2B 512 decode 32.66 -> 34.45, 9B 512
decode 9.85 -> 10.92), which is why the ratios are nearly unchanged. No per-PR A/B was run for this
report; the Peregrine change is the combined effect of #108, #111, #113, #114, #115, #118, #120 and
#121 plus same-day host drift.

## What was checked, and what was not

- Checked: all three model files hash to the 2026-11-08 run's SHA-256; every measured prompt is
  within 5% of its target; 3 samples per phase per engine (and per flash-attention setting); the
  `-fa auto` and `-fa off` settings are recorded in every fork manifest.
- Checked: the 9B 10-user and 9B 4096 fork rows are medians of three one-round invocations, not of
  one long invocation; `merge.py` combines them the way a `--runs 3` run would.
- Not measured: 4 users at 512 (not in the 2026-11-08 matrix either), 4 users at 4096, 16k context,
  any other quantisation, and the 4B file (not present on this host).
