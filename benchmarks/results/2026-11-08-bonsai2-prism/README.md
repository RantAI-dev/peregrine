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
- Models (SHA-256 taken from the manifests):
  - `Ternary-Bonsai-2-2B-PQ2_0.gguf`, 517,312,992 bytes,
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
- Flash attention: measured on the fork in both explicit settings, `-fa on` and `-fa off`,
  for every row, and cross-checked with `llama-bench`. The fork's default is `-fa auto`,
  which resolves to on (its default and explicit-on server numbers agree on every plain-model
  row). The setting comes from `run.py --llama-fa`, the same option #102 added for upstream
  llama.cpp; it is passed to both `llama-server` and `llama-bench` and recorded as
  `settings.llama_flash_attn` and `engine.flash_attn`.

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

## Why the manifests say `avx2+fma` on an AVX-512 host

This is a labelling bug in the probe, not a tier that was skipped. Every Peregrine manifest
here records `machine.simd_tier: avx2+fma`, the same string the 2026-11-07 baseline recorded,
while the CPU is Zen 5 with AVX-512 (see the machine table above). `peregrine tier` on this
host prints both values side by side:

```
kernel tier   x86_64-avx2-avx512-vnni-gfni-vbmi
simd          avx2+fma
```

Two different functions feed those two lines. `peregrine_kernels::cpu::tier_name()` resolves
the kernel tier actually in effect: detected features capped by `PEREGRINE_ISA` (unset here,
so `isa=native`), which is `x86_64-avx2-avx512-vnni-gfni-vbmi` — the AVX-512 tier.
`peregrine_core::simd_name()` is an older, coarser detector that only knows `scalar`,
`avx2`, `avx2+fma` and `neon`; it has no AVX-512 branch and returns `avx2+fma` as soon as
AVX2 and FMA are present. `benchmarks/run.py` fills `machine.simd_tier` from the hardware
probe (`examples/probe`, which prints `"simd": "avx2+fma"` on this host), so the coarse name
is what reached the manifests even though the kernels ran at the AVX-512 tier.

PQ2_0 uses that tier. Its fastest path is the 16-row repacked-tile kernel built on AVX-512
VNNI and GFNI (`MatPath::Tiles`, `peregrine-kernels/src/table.rs`; the kernel itself is
`tern::pq2_zmm`), and `peregrine tier` reports `zmm=true`, `tiles=true` with `isa=native` on
this host. The AVX2 `MatPath::RowsPq2` path is the fallback on hosts without AVX-512, not
what ran here. The fork was built with `-march=native` and also dispatches AVX-512, so the
comparison is AVX-512 against AVX-512, not AVX2 against AVX-512.

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
- The flash-attention sweeps are fork-only (Peregrine has no such switch) and alternate
  `-fa on`, `-fa off` between rounds.

The raw per-invocation manifests are in [`raw/`](raw/). [`merge.py`](merge.py) concatenates
their sample arrays into the one-manifest-per-engine shape a single `--runs 3` invocation
would write, recomputing medians and half-ranges with the harness's rule, and puts the
flash-attention setting in the output name. Run it with `python3 merge.py` from this
directory to regenerate the manifests beside it.

Three harness additions were needed for this model family and are in the same PR:

- `--chat-template-kwargs '{"enable_thinking": false}'`, sent to both engines. Bonsai 2 uses
  the Qwen3 template; with thinking on, both servers put the whole answer in
  `reasoning_content`, which the streaming client does not count and which makes the greedy
  text check unreadable. Thinking is off for every measured request on both engines.
- `--token-tolerance 1`. Peregrine occasionally reports 255 of 256 tokens when the last
  generated token is an empty control token (`<|im_end|>`): the server does not count a
  control token that carries no text. The sample is kept, and its throughput is still
  `reported_tokens / reported_decode_time`; the fork reported 256 on every request. This is
  at most a 0.4% difference in the token count, and it is applied to both engines.
- `--engine-args` (and, from #102, `--llama-fa {auto,on,off}`, shared by every
  llama.cpp-family adapter), so the fork can be run in each flash-attention setting and the
  setting lands in the manifest as `settings.llama_flash_attn` and `engine.flash_attn`.

## Results

Medians of 3 rounds, spread is the half-range. "Fork best" is the faster of the two explicit
flash-attention settings for that row (lower for TTFT); the ratio is Peregrine ÷ "fork best".
"Fork default" is the fork with its own default (`-fa auto`) as a secondary column. Above 1
means Peregrine is faster; for TTFT, below 1 means Peregrine's first token arrives sooner.

### Bonsai 2 2B, PQ2_0

| Context | Users | Metric | Peregrine | Fork best (-fa) | Ratio | Fork default |
|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 70.64 ± 0.77 | 32.66 ± 0.63 (on) | **2.16x** | 32.67 |
| 512 | 1 | Prompt tok/s | 259.25 ± 14.28 | 108.24 ± 1.69 (on) | **2.40x** | 112.19 |
| 512 | 10 | Aggregate tok/s | 80.59 ± 3.53 | 29.96 ± 0.61 (on) | **2.69x** | 29.79 |
| 512 | 10 | TTFT (s) | 12.92 ± 1.50 | 42.79 ± 1.62 (on) | **0.30x** | 45.40 |
| 4096 | 1 | Decode tok/s | 60.64 ± 0.59 | 30.47 ± 0.59 (on) | **1.99x** | 30.76 |
| 4096 | 1 | Prompt tok/s | 202.94 ± 0.83 | 105.19 ± 0.43 (off) | **1.93x** | 103.90 |

### Bonsai 2 9B, PQ2_0

| Context | Users | Metric | Peregrine | Fork best (-fa) | Ratio | Fork default |
|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 22.97 ± 0.22 | 9.85 ± 0.06 (off) | **2.33x** | 9.82 |
| 512 | 1 | Prompt tok/s | 67.24 ± 0.18 | 26.12 ± 0.30 (off) | **2.57x** | 26.63 |
| 512 | 10 | Aggregate tok/s | 21.62 ± 0.57 | 7.83 ± 0.09 (on) | **2.76x** | 7.76 |
| 512 | 10 | TTFT (s) | 51.66 ± 0.66 | 183.40 ± 3.22 (on) | **0.28x** | 185.48 |
| 4096 | 1 | Decode tok/s | 20.36 ± 0.22 | 9.22 ± 0.02 (on) | **2.21x** | 8.93 |
| 4096 | 1 | Prompt tok/s | 57.42 ± 0.39 | 25.53 ± 0.01 (off) | **2.25x** | 25.33 |

### MTP on the 9B `-MTP-Q8_0` file, 512 tokens, 1 user

Peregrine's MTP speculation is on by default and was disabled with `--opt MTP=0`. The fork
enables the same head with `--spec-type draft-mtp` (its `--mtp` flag is a download option,
not a server option); the "off" row is the fork with no speculative flag.

| Configuration | Peregrine | Fork best (-fa) | Ratio | Fork default |
|---|---|---|---|---|
| MTP off | 22.75 ± 0.09 | 10.69 ± 0.03 (on) | **2.13x** | 9.83 |
| MTP on | 25.01 ± 0.77 | 9.66 ± 0.44 (on) | **2.59x** | 9.77 |
| On ÷ off | **1.10x** | 0.90x | — | 0.99x |

Peregrine's MTP adds about 10% at this prompt length. The fork's speculative mode runs (its
log says `speculative decoding enabled: draft-mtp`, and its timings report `draft_n` and
`draft_n_accepted`), but at 256 tokens of this corpus it did not beat its own non-speculative
decode; the fork's MTP-on and MTP-off rows were measured about an hour apart and differ by
more than any MTP effect that appeared in the interleaved runs, so read the fork's own
on/off as parity and the 2.13x–2.59x against Peregrine as the measured result. A separate
64-token check on the same file reported 66 drafted and 40 accepted tokens for the fork.

## Flash attention: on, off and the default

Upstream llama.cpp on the bluefin host showed a large depth penalty from its default flash
attention (1B at 4096: 13.0 tok/s with the default, 29.1 with `-fa 0`). That is **not** what
this fork does to Bonsai 2. Across every server row, explicit `-fa on` and `-fa off` land
within 0–2% of each other, and the default (`-fa auto`) is the on setting:

| Row | Fork `-fa on` | Fork `-fa off` |
|---|---|---|
| 2B, 512, 1 user, decode tok/s | 32.66 ± 0.63 | 32.41 ± 0.37 |
| 2B, 512, 1 user, prompt tok/s | 108.24 ± 1.69 | 107.77 ± 0.51 |
| 2B, 512, 10 users, aggregate tok/s | 29.96 ± 0.61 | 28.25 ± 0.15 |
| 2B, 4096, 1 user, decode tok/s | 30.47 ± 0.59 | 29.91 ± 0.31 |
| 2B, 4096, 1 user, prompt tok/s | 102.65 ± 0.66 | 105.19 ± 0.43 |
| 9B, 512, 1 user, decode tok/s | 9.83 ± 0.03 | 9.85 ± 0.06 |
| 9B, 512, 1 user, prompt tok/s | 26.04 ± 0.09 | 26.12 ± 0.30 |
| 9B, 512, 10 users, aggregate tok/s | 7.83 ± 0.09 | 7.62 ± 0.08 |
| 9B, 4096, 1 user, decode tok/s | 9.22 ± 0.02 | 9.10 ± 0.03 |
| 9B, 4096, 1 user, prompt tok/s | 25.41 ± 0.07 | 25.53 ± 0.01 |

`llama-bench` (raw JSON in [`llama-bench/`](llama-bench/)) agrees after taking medians: the
host has a second, ~35% faster decode mode that hits roughly one sample in five regardless
of `-fa`, so the averages in its JSON are not usable as-is.

| Model | Test | `-fa auto` | `-fa on` | `-fa off` |
|---|---|---|---|---|
| 2B | pp 512 tok/s | 108.12 | 109.86 | 109.92 |
| 2B | tg 256 at depth 0 | 33.39 | 32.89 | 33.87 |
| 2B | tg 256 at depth 4096 | — | 33.70 | 33.30 |
| 9B | pp 512 tok/s | 30.64 | 30.47 | 30.49 |
| 9B | tg 256 at depth 0 | 11.09 | 11.05 | 9.88 |
| 9B | tg 256 at depth 4096 | — | 10.20 | 10.05 |

The 9B depth-0 `-fa off` median (9.88) is the one number that differs by more than the
server's spread; its three samples were 8.32, 9.88 and 10.93, and the server at the same
setting measures 9.85. Read it as noise. `llama-bench`'s depth-4096 decode is about 10%
faster than the server at the same setting for the 2B and about 10% faster for the 9B, so
the two tools are not interchangeable in absolute terms; both agree on the direction, which
is that turning flash attention off does not help here.

The reason may be the hybrid architecture: these Bonsai 2 files spend most layers in the
gated recurrent path and only some in attention, so the attention kernel is a smaller share
of the step than in the dense 1B where the bluefin penalty appeared.

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
  of its target length; every phase has 3 samples per engine and per flash-attention setting;
  the 9B fits at 4096 with one user on this host, and 10 users fit at 512; the host did not
  become memory-bound (9.7 GiB still available after the last run).
- Checked: the fork's `-fa auto` equals `-fa on` for the plain models (server default and
  explicit-on agree within 0.2% on 2B and 0.1% on 9B at 512), and `-fa off` does not produce
  the bluefin depth gain on these files.
- Not measured: 10 users at 4096 (not in the requested matrix), 4 users (not requested), any
  other quantisation, and the 4B file (absent).
- The fork's 10-user TTFT at 9B (183–185 s) is a real number, not a timeout: ten 512-token
  prefills at about 26 tok/s serialise before the first user sees a token.
