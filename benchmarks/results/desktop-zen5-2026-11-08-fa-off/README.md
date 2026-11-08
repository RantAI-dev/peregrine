# Desktop re-check with llama.cpp's flash attention off (roadmap S3.2)

The desktop baseline in [`../desktop-zen5-2026-11-08/`](../desktop-zen5-2026-11-08/) was taken
at llama.cpp's defaults. llama.cpp v0.9.0 defaults to `-fa auto`, which resolves to flash
attention **on** on the CPU backend, and on the 4-core bluefin host that default collapsed
decode at depth (Llama 3.2 1B at 4096 tokens: 13.0 tok/s default, 29.1 with `-fa 0`). The
[harness policy](../../README.md#flash-attention-llamacpp-and-its-forks-compare-against-the-faster-setting)
is to compare against the faster setting per row, so this folder re-measures llama.cpp with
`-fa off` on the desktop and re-runs Peregrine on the same day.

- Date: 2026-11-08 (llama.cpp `-fa off` and Peregrine runs 20:04–21:10 local, UTC+7).
- Peregrine: `origin/main` at `e48aea5` (`perf: wide Q5_0 prefill kernel`), `cargo build
  --release -p peregrine`, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`. The branch this ran from
  adds only files under `benchmarks/` and `docs/`; `git diff origin/main HEAD -- crates/
  Cargo.toml Cargo.lock` is empty, so the engine binary is main's.
- llama.cpp: v0.9.0 (`cd959ba`), the harness cache build (CMake Release, AVX2/FMA CPU),
  driven with `run.py --llama-fa off`, which passes `-fa off` to `llama-server`.
- Models, same files as the 2026-11-08 baseline (SHA-256 re-checked against its manifests,
  both match byte for byte):
  - `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807,694,464 bytes,
    `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`
  - `qwen2.5-0.5b-instruct-q4_k_m.gguf`, 491,400,032 bytes,
    `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- Settings: 12 threads (one per physical core), 256 greedy decode tokens per request,
  server mode, 3 interleaved rounds per engine, prefix-cache defeat by a per-request nonce,
  prompt lengths calibrated per engine and model (every measured prompt within 5% of the
  target).

## The machine

| Property | Value |
|---|---|
| CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M (Zen 5), family 26, model 36 |
| Topology | 1 socket, 12 physical cores, 2 threads per core, 24 logical CPUs |
| SIMD | AVX2, FMA, AVX-512F/DQ/BW/VL, AVX-VNNI, AVX-VNNI-256, AVX512-BF16, GFNI, VAES |
| RAM | 11,763 MiB total, 3,072 MiB swap |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2 (x86_64), glibc 2.39 |
| Virtualization | WSL2 under the Microsoft hypervisor, full virtualization (AMD-V) |
| CPU governor | unknown (VM or fixed frequency) — WSL2 does not expose it |
| Benchmark threads | 12 |

The streaming-bandwidth probe returned 35.79 GB/s in the first invocation of this session;
it has ranged 29–48 GB/s across sessions on this host, so the `% of ceiling` column is not
carried over and the tok/s numbers are the comparable ones. The manifests still record
`machine.simd_tier: avx2+fma`; that is the probe's coarse label, not the kernel tier. The
engine runs the AVX-512 tier (`peregrine tier` prints `kernel tier
x86_64-avx2-avx512-vnni-gfni-vbmi`); see [`../2026-11-08-bonsai2-prism/README.md`](../2026-11-08-bonsai2-prism/README.md#why-the-manifests-say-avx2fma-on-an-avx-512-host).

## How the runs were split

The host keeps each command under ten minutes, so the sweep is several invocations:

- 512 tokens, 1 and 10 users: both engines in one invocation per model (interleaved ABAB
  rounds).
- 512 tokens, 4 users: both engines in one invocation per model.
- 4096 tokens, 1 user: both engines in one invocation per model.
- 4096 tokens, 10 users: **not completed**. One round of the cheapest combination (Qwen
  0.5B, llama.cpp `-fa off`, `--runs 1`) did not finish inside the 9m50 command cap, and the
  same attempt on the 1B did not either; 10 concurrent 4096-token prefills plus 2,560 decode
  tokens do not fit the cap on this host. The rows are recorded as not completed rather than
  estimated. The default-setting 4096 multi-user numbers from the 2026-11-08 baseline are
  unchanged and still in that folder.

The raw manifests are beside this file; each carries the engine build, the machine record
and every individual sample. The `llama.cpp` manifests record `settings.llama_flash_attn:
"off"` and `engine.flash_attn: "off"`.

## Results

Medians of 3 rounds, ± half-range. "llama.cpp default" is the same-day baseline in
[`../desktop-zen5-2026-11-08/`](../desktop-zen5-2026-11-08/) (llama.cpp `-fa auto`, which is
on). "Faster llama.cpp" is whichever of the two settings measured faster for that row (lower
for TTFT); the ratio is Peregrine ÷ that number, so above 1 means Peregrine is faster.

### Llama 3.2 1B Instruct Q4_K_M

| Context | Users | Metric | Peregrine (`e48aea5`) | llama.cpp default (`-fa auto`) | llama.cpp `-fa off` | Faster llama.cpp | Ratio vs faster |
|---|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 62.63 ± 1.06 | 50.98 ± 0.10 | 51.76 ± 0.50 | `-fa off` | **1.21x** |
| 512 | 1 | Prompt tok/s | 183.76 ± 6.37 | 261.96 ± 3.90 | 251.54 ± 5.98 | default | 0.70x |
| 512 | 4 | Aggregate tok/s | 51.62 ± 0.21 | 54.50 ± 0.50 | 41.64 ± 0.15 | default | 0.95x |
| 512 | 10 | Aggregate tok/s | 53.40 ± 0.58 | 61.71 ± 0.20 | 35.97 ± 0.05 | default | 0.87x |
| 512 | 10 | TTFT (s) | 19.66 ± 0.23 | 15.58 ± 3.09 | 23.37 ± 2.10 | default | 1.26x (Peregrine slower) |
| 4096 | 1 | Decode tok/s | 47.03 ± 0.82 | 32.44 ± 0.60 | 33.78 ± 0.39 | `-fa off` | **1.39x** |
| 4096 | 1 | Prompt tok/s | 130.87 ± 0.48 | 189.03 ± 8.40 | 178.48 ± 6.58 | default | 0.69x |
| 4096 | 10 | Aggregate tok/s | — | 8.56 (2026-11-08) | not completed | — | — |

### Qwen 2.5 0.5B Instruct Q4_K_M

| Context | Users | Metric | Peregrine (`e48aea5`) | llama.cpp default (`-fa auto`) | llama.cpp `-fa off` | Faster llama.cpp | Ratio vs faster |
|---|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 69.96 ± 0.93 | 75.80 ± 0.60 | 74.51 ± 2.05 | default | 0.92x |
| 512 | 1 | Prompt tok/s | 321.90 ± 2.41 | 435.25 ± 2.10 | 430.90 ± 9.83 | default | 0.74x |
| 512 | 4 | Aggregate tok/s | 78.27 ± 0.35 | 81.80 ± 8.10 | 63.67 ± 0.19 | default | 0.96x |
| 512 | 10 | Aggregate tok/s | 83.35 ± 0.51 | 93.41 ± 0.80 | 56.77 ± 0.37 | default | 0.89x |
| 512 | 10 | TTFT (s) | 12.12 ± 0.11 | 9.54 ± 1.56 | 14.24 ± 1.20 | default | 1.27x (Peregrine slower) |
| 4096 | 1 | Decode tok/s | 54.87 ± 0.47 | 48.75 ± 0.60 | 49.06 ± 0.86 | `-fa off` | **1.12x** |
| 4096 | 1 | Prompt tok/s | 215.09 ± 1.31 | 279.44 ± 1.40 | 293.07 ± 9.48 | `-fa off` | 0.73x |
| 4096 | 10 | Aggregate tok/s | — | 13.36 (2026-11-08) | not completed | — | — |

## What the re-check says

- **The bluefin depth collapse does not reproduce on the desktop.** `-fa off` is 1.5% faster
  at 512 (1B) and 4.1% faster at 4096; on the 0.5B it is 1.7% *slower* at 512 and 0.6%
  faster at 4096. Against the faster setting the 1B 4096 decode ratio moves from 1.36x to
  **1.39x**; it is the only headline number that changes.
- **Prompt processing is mixed.** `-fa off` is 4.9% *faster* on the 0.5B at 4096, and 4.0%
  (1B, 512) to 5.6% (1B, 4096) *slower* on the 1B, so the default stays the competitor there.
  Ratios move from 0.62–0.66x to 0.69–0.74x. Most of that is Peregrine's own move (its prompt
  numbers are 4.4–19.0% higher today than in the morning baseline); main now also contains
  #103's wide Q5_0 prefill kernel, but this run did not A/B it, so it is not attributed here.
- **Multi-user throughput is much worse with `-fa off` on this host**: 10 users at 512 drops
  from 61.71 to 35.97 tok/s (1B, 0.58x) and from 93.41 to 56.77 (0.5B, 0.61x); 4 users drops
  from 54.50 to 41.64 and 81.80 to 63.67. The default (`-fa auto` → on) remains the
  competitor for every multi-user row, and it is a bigger gap than on bluefin.
- Peregrine itself moved between the two same-day sessions: decode +2.7% to +6.4% and prompt
  +4.4% to +19.0% against its own 2026-11-08 baseline, with the largest gain on the 0.5B
  prompt (270.6 → 321.9 tok/s). The baseline's llama.cpp ran around 11:00 local and this
  Peregrine run around 20:00, so same-day host drift is part of that; the ratios here use
  the faster of the two same-day llama.cpp settings, as the policy asks.

## Is the repacked Q4_K kernel worth turning on for AVX-512?

The repacked kernels (#88) are off by default on AVX-512 because that tier has no repacked
kernel of its own yet. To decide whether to turn them on there, the 1B was measured with
`PEREGRINE_KREPACK=1` (`--opt KREPACK=1`) against the same binary at its default, five rounds
alternating, single user, 512 and 4096 tokens. The per-round numbers and raw manifests are in
[`krepack-ab/`](krepack-ab/); medians:

| Test | Default | `KREPACK=1` | Change |
|---|---|---|---|
| Decode 512 tok/s | 64.00 | 63.24 | −1.2% |
| Decode 4096 tok/s | 46.77 | 46.75 | 0.0% |
| Prompt 512 tok/s | 185.48 | 233.88 | +26.1% |
| Prompt 4096 tok/s | 130.56 | 156.05 | +19.5% |

Decode at 4096 is flat; at 512 it is 0.76 tok/s lower (−1.2%), inside the overlap of the two
arms' ranges (default 62.64–65.50, repacked 60.17–65.00). Prompt processing gains a fifth to a
quarter. The decision belongs to whoever turns the default on for AVX-512; on these numbers
the decode cost is small enough that it is a question of appetite, not of a regression.

## What was checked, and what was not

- Checked: both model files hash to the 2026-11-08 baseline's SHA-256; every measured prompt
  is within 5% of its target; 3 samples per phase per engine (and per flash-attention
  setting); the `-fa off` setting is recorded in every llama.cpp manifest.
- Checked: 10 users at 4096 with llama.cpp does not complete inside the 10-minute cap on this
  host, on either model (two attempts, both capped at 9m50). Peregrine's 4096 multi-user rows
  were not attempted separately: it serves 7.3–10.5 tok/s in total there, so it cannot be
  faster than the llama.cpp attempt that already failed the cap.
- Not measured: 4 users and 10 users at 4096, 16k context, any other quantisation.
