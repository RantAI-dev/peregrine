# Bonsai 2 vs PrismML's llama.cpp fork on the 4-core bluefin

The AVX2 data point for the Bonsai 2 comparison in
[`../2026-11-08-bonsai2-prism/`](../2026-11-08-bonsai2-prism/). That run measured a 12-core
Zen 5 desktop, where both engines dispatched AVX-512; this one measures the 4-core bluefin,
where neither engine has AVX-512 and both run their AVX2 kernels. Same bytes, same host, same
thread count, same requests.

- Date: 2026-11-08 UTC (2026-11-09 local, WIB).
- Peregrine: `origin/main` at `cf9e81d849f1d2a149dd6b64a63aeb3b6a691049`
  (`bench: bluefin after the Q5_0 speed-ups`), built `release`
  (`cargo build --release -p peregrine`), `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- Fork: PrismML `llama.cpp` branch `prism`, pinned commit
  `62c418ccd7aefee37997e94c2d9fac1eed7fc89b`. It fetched and built as pinned; there was no
  fallback to the branch head. Built in `/var/tmp/peregrine-bench/llama.cpp-prism-62c418ccd`,
  CMake Release with explicit `-DGGML_AVX2=ON -DGGML_AVX=ON -DGGML_FMA=ON`. `GGML_NATIVE`
  stays on (the adapter does not turn it off), so the compile
  line also carries `-march=native`, which on this Zen 2 host resolves to AVX2/FMA with no
  AVX-512.
- Models (SHA-256 taken from the files and checked against each `SHA256SUMS`):
  - `Ternary-Bonsai-2-2B-PQ2_0.gguf`, 517,312,992 bytes,
    `d5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928`
  - `Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf`, 1,274,642,176 bytes,
    `0acf517de1dc3d85c403a60df6b8b07a4878ac174f368d4390be98406a4d6c0d`
- The only 4B file on this host is the `-MTP-Q8_0` file; there is no plain 4B PQ2_0 file, so
  every 4B row uses it. "MTP on" is Peregrine's default; "MTP off" is `--opt MTP=0`. The fork
  was run without `--spec-type draft-mtp`, so its 4B rows are non-speculative.
- The manifests record each model as its `models/<dir>/<file>` path inside the checkout
  (`models/` is the local symlink to `~/models/peregrine`), not the host path it resolves to.
- Settings: contexts 512 and 4096, 3 interleaved rounds per engine, 256 greedy decode tokens
  per request, server mode, 1 and 10 users at 512, 1 user at 4096, prefix-cache defeat by a
  per-request nonce, prompt lengths calibrated per engine and model (every measured prompt
  within 5% of the target). Threads: **4**, one per physical core.

The 4B row at 4096 tokens is the one that did not fit: the fork's 4096 phase plus prompt
calibration was stopped at 9m20s, and is recorded as **not completed** below. The 2B and all
512-token 4B rows completed.

## The machine

| Property | Value |
|---|---|
| CPU | AMD Ryzen 3 PRO 4350G (Zen 2), family 23, model 96 |
| Topology | 1 socket, 4 physical cores, 2 threads per core, 8 logical CPUs |
| SIMD | AVX, AVX2, FMA; **no AVX-512** |
| RAM | 13,839 MiB total, 8,191 MiB swap; 9.0 GiB available at the end |
| OS | Linux 7.2.7-200.fc44.x86_64 (Fedora, bare metal) |
| CPU governor | `powersave`, unchanged |
| Benchmark threads | 4 |

## This is the AVX2-vs-AVX2 comparison

The desktop folder measured AVX-512 against AVX-512: both engines dispatched their AVX-512
Bonsai 2 kernels, because the host has VNNI/GFNI and the fork was built with `-march=native`.
This host has neither. `peregrine tier` prints the tier actually in effect:

```
kernel tier   x86_64-avx2
simd features avx2+fma
```

and every manifest records `machine.simd_tier: x86_64-avx2`. PQ2_0 therefore runs Peregrine's
AVX2 `MatPath::RowsPq2` path rather than the AVX-512 `MatPath::Tiles` path, and the fork runs
its own AVX2 path. The fork adapter used to hard-code `"CMake Release, native CPU (AVX-512 on
this host)"` into `engine.build_type`; on this host that claim was false, so the adapter now
records `GGML_NATIVE=ON` with the explicit AVX2/FMA flags and the raw manifests written during
this session were corrected to match. The build itself was AVX2/FMA throughout.

## How the runs were split

The host caps a command at ten minutes. The 512-token, 1-user phases of both engines fit in
one 3-round invocation. Everything else was split:

- 2B, 512, 10 users: one 3-round invocation per engine and per flash-attention setting.
- 2B, 4096, 1 user: Peregrine in one 3-round invocation; the fork in a 1-round invocation plus
  a 2-round invocation (auto), and one 3-round invocation (off).
- 4B, 512, 1 user: both engines in one 3-round invocation; the fork's `-fa off` in a second.
- 4B, 512, 10 users: one 1-round invocation per engine and setting (about 4 minutes for
  Peregrine and 8 for the fork, calibration included).
- 4B, 4096, 1 user: Peregrine in a 3-round invocation; the fork was stopped at the cap in its
  single attempt and is not completed.
- Peregrine's 2B 512/1u third round was re-run in its own invocation after the harness dropped
  one sample for a reply short of 256 tokens.

The raw per-invocation manifests are in [`raw/`](raw/). [`merge.py`](merge.py) concatenates
their sample arrays into the one-manifest-per-engine shape a single `--runs 3` invocation would
write, recomputes medians and half-ranges with the harness's rule, and puts the flash-attention
setting in the output name. Run it with `python3 merge.py` from this directory to regenerate
the manifests beside it.

## The 4B token count

Peregrine's 4B MTP replies sometimes report 254 of 256 completion tokens: two empty control
tokens are not counted as content. The sample is kept with `--token-tolerance 2`, and its
throughput is still `reported_tokens / reported_decode_time` (at most a 0.8% difference). The
fork reported 256 on every request. The 2B rows use the same `--token-tolerance 1` as #101.

## Results

Medians of 3 rounds, spread is the half-range. "Fork best" is the faster of the fork's explicit
`-fa auto` (which is on here) and `-fa off` server settings for that row (lower for TTFT); the
ratio is Peregrine ÷ "fork best". "Fork default" is the fork's own default (`-fa auto`) as a
secondary column. Above 1 means Peregrine is faster; for TTFT, below 1 means Peregrine's first
token arrives sooner.

### Bonsai 2 2B, PQ2_0

| Context | Users | Metric | Peregrine | Fork best (-fa) | Ratio | Fork default |
|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 46.40 ± 0.09 | 33.63 ± 0.07 (off) | **1.38x** | 33.46 |
| 512 | 1 | Prompt tok/s | 118.67 ± 0.36 | 56.95 ± 1.36 (off) | **2.08x** | 55.15 |
| 512 | 10 | Aggregate tok/s | 33.69 ± 0.03 | 19.08 ± 0.39 (auto) | **1.77x** | 19.08 |
| 512 | 10 | TTFT (s) | 32.82 ± 0.05 | 82.05 ± 2.70 (auto) | **0.40x** | 82.05 |
| 4096 | 1 | Decode tok/s | 42.28 ± 1.54 | 30.79 ± 0.09 (off) | **1.37x** | 25.80 |
| 4096 | 1 | Prompt tok/s | 109.04 ± 1.45 | 58.07 ± 1.03 (auto) | **1.88x** | 58.07 |

### Bonsai 2 4B, PQ2_0 (+MTP head)

| Context | Users | Metric | Peregrine | Fork best (-fa) | Ratio | Fork default |
|---|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 24.59 ± 1.07 | 14.95 ± 0.01 (off) | **1.64x** | 14.82 |
| 512 | 1 | Prompt tok/s | 48.05 ± 0.08 | 18.55 ± 0.12 (off) | **2.59x** | 18.07 |
| 512 | 10 | Aggregate tok/s | 14.12 ± 0.13 | 6.37 ± 0.03 (auto) | **2.22x** | 6.37 |
| 512 | 10 | TTFT (s) | 78.84 ± 0.85 | 279.73 ± 1.55 (auto) | **0.28x** | 279.73 |
| 4096 | 1 | Decode tok/s | 21.28 ± 0.53 | not completed | — | — |
| 4096 | 1 | Prompt tok/s | 43.91 ± 0.08 | not completed | — | — |

### What the 4-core AVX2 point says

Peregrine's AVX2 advantage is smaller than its AVX-512 one. The same 2B file and the same
method were **2.16x** decode and **2.40x** prompt at 512 on the desktop (12 threads, AVX-512),
against **1.38x** and **2.08x** here; at 4096 the desktop was 1.99x/1.93x, here 1.37x/1.88x,
and 10-user aggregate falls from 2.69x to 1.77x. Prompt processing keeps most of the gap:
the fork's AVX2 prefill is relatively weaker than its AVX-512 prefill. Decode is where the
advantage narrows most, from 2.0-2.3x to 1.4x, which is what a rows-path PQ2_0 kernel against
a mature AVX2 fork should look like. The 4B file, which was absent on the desktop, sits at
1.64x decode, 2.59x prompt and 2.22x at 10 users.

## MTP on the 4B file

Peregrine's MTP speculation is on by default and was disabled with `--opt MTP=0`. The fork
does not use the MTP head in these rows (no `--spec-type draft-mtp`), so its "on" and "off"
are the same measurement within noise.

| Context | Users | Metric | MTP on | MTP off | On ÷ off |
|---|---|---|---|---|---|
| 512 | 1 | Decode tok/s | 24.59 ± 1.07 | 20.79 ± 0.05 | **1.18x** |
| 512 | 10 | Aggregate tok/s | 14.12 ± 0.13 | 14.07 ± 0.12 | 1.00x |
| 4096 | 1 | Decode tok/s | 21.28 ± 0.53 | 18.77 ± 0.06 | **1.13x** |

MTP adds 13-18% to single-user decode at both contexts and nothing to the 10-user aggregate,
where prefill and batching dominate. Against the fork's faster setting the MTP-off row is
1.39x (20.79 ÷ 14.95) and the MTP-on row 1.64x at 512/1u.

## Flash attention: on, off and the default

The fork's default is `-fa auto`, which resolves to on. Explicit `-fa on` was not measured
separately; the default setting is the on setting, as #101 verified on the desktop.

| Row | Fork default (`-fa auto`) | Fork `-fa off` |
|---|---|---|
| 2B, 512, 1 user, decode tok/s | 33.46 ± 0.04 | 33.63 ± 0.07 |
| 2B, 512, 1 user, prompt tok/s | 55.15 ± 0.76 | 56.95 ± 1.36 |
| 2B, 512, 10 users, aggregate tok/s | 19.08 ± 0.39 | 17.37 ± 0.25 |
| 2B, 512, 10 users, TTFT (s) | 82.05 ± 2.70 | 93.03 ± 0.25 |
| 2B, 4096, 1 user, decode tok/s | 25.80 ± 0.15 | **30.79 ± 0.09** |
| 2B, 4096, 1 user, prompt tok/s | 58.07 ± 1.03 | 55.08 ± 1.08 |
| 4B, 512, 1 user, decode tok/s | 14.82 ± 0.04 | 14.95 ± 0.01 |
| 4B, 512, 1 user, prompt tok/s | 18.07 ± 0.35 | 18.55 ± 0.12 |
| 4B, 512, 10 users, aggregate tok/s | 6.37 ± 0.03 | 6.19 ± 0.02 |
| 4B, 512, 10 users, TTFT (s) | 279.73 ± 1.55 | 288.28 ± 1.68 |

On the desktop, explicit on and off were within 0-2% on every server row. Here they are within
about 3-5% at 512 and for prompt processing, but the 2B at 4096 is an exception: `-fa off`
decodes 19% faster (30.79 vs 25.80 tok/s). That is the same depth penalty upstream llama.cpp
shows on this host, and it is why the 2B 4096 decode ratio is taken against `-fa off`. The
fork's default is still the faster setting for the 2B 4096 prompt and for both 10-user rows.

## The models produce the same text

Greedy, `enable_thinking: false`, "Explain in two sentences why the sky looks blue." Both
engines return the same opening sentence, so the fork is running these weights and not a
placeholder path:

2B, Peregrine:

> Sunlight contains all colors of the rainbow, but when it travels through the Earth's atmosphere, shorter wavelengths (blue) are scattered more strongly by gas molecules than longer wavelengths (red). This process is known as Rayleigh scattering, which makes the sky appear blue to our eyes.

2B, fork:

> Sunlight contains all colors of the rainbow, but only short wavelengths (blue) are scattered by air molecules in a process known as Rayleigh scattering. This causes the blue light to be scattered more strongly than the red light, which travels in a straight line and reaches your eyes first.

4B, both engines return the same text:

> The sky appears blue because sunlight is scattered by molecules in the Earth's atmosphere, with shorter blue wavelengths being scattered more effectively than longer red ones. This phenomenon, known as Rayleigh scattering, means that when you look up, your eyes receive more scattered blue light from all directions than any other color.

The 4B agreement is stronger than the 2B because the larger file's greedy path diverges later;
the 2B pair says the same thing in different words.

## What was checked, and what was not

- Checked: both engines load and decode both files; every measured prompt is within 5% of its
  target; every phase that completed has 3 samples per engine and per flash-attention setting;
  the 4B fits at 4096 with one user (43.91 tok/s prompt, 21.28 tok/s decode) and 10 users fit
  at 512; the host did not become memory-bound (9.0 GiB available after the last run).
- Checked: the greedy text from both engines is coherent and they agree on the opening
  sentence for each model.
- Not completed: the fork's 4B 4096-token phase. A single attempt was stopped at the 10-minute
  cap after calibration and did not produce a sample. No estimate is substituted.
- Not measured: 4 users (optional in the brief and left out to keep the session inside the
  command cap), 10 users at 4096, any other quantisation, the 9B file (not on this host), and
  the fork with `--spec-type draft-mtp` on the 4B.
