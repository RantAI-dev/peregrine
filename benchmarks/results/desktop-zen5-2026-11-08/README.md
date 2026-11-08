# Desktop baseline after the November speed-ups: Peregrine vs llama.cpp (roadmap S3.2)

The S3.2 desktop baseline re-run on the same machine, the same model files and the same
harness settings as [`../desktop-zen5-2026-11-07/`](../desktop-zen5-2026-11-07/), so the two
runs can be read side by side. This run adds a second Peregrine pass with the repacked
Q4_K kernels forced on (`PEREGRINE_KREPACK=1`), which the AVX-512 tier leaves off by default.

- Date: 2026-11-08.
- Peregrine: `main` at `5f76d488` (the November speed-ups: repacked k-quant store, prefill
  chunk default, more attention tasks at short context, startup thread/spin calibration),
  built `release` (`cargo build --release -p peregrine`), `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- llama.cpp: v0.9.0 (`cd959ba`), CMake release build, AVX2/FMA CPU (unchanged, from the
  harness cache).
- Models: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
  (SHA-256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`,
  491,400,032 bytes) and `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
  (SHA-256 `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`,
  807,694,464 bytes). Both hashes match the 2026-11-07 manifests exactly, so the two
  baselines are measured on identical bytes.
- Settings: contexts 512 and 4096, 3 interleaved rounds per engine, 256 greedy decode tokens
  per request, server mode, prefix-cache defeat by per-request nonce, prompt lengths
  calibrated per engine and model (every measured prompt within 5% of the target).
- The default 512-token sweeps ran both engines in one invocation (interleaved ABAB
  rounds); the 4096-token and KREPACK sweeps ran one engine per invocation, because the
  full sweep exceeds a 10-minute command on this host. The split is tabulated under
  "How the runs were split" in [`report.md`](report.md).

## The model files

The Qwen file was the one open question. `/tmp/gold-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
is a **different upload**: SHA-256 `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`,
397,808,192 bytes. It does not match the 2026-11-07 manifest. The file the first baseline
hashed is `74a4da8c…` at 491,400,032 bytes, and that is what this run used, so the
2026-11-07 → 2026-11-08 deltas below compare the same weights. The 1B file matched on the
first try (`6f85a640…`).

## The machine

| Property | Value |
|---|---|
| CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M (Zen 5), family 26, model 36 |
| Topology | 1 socket, 12 physical cores, 2 threads per core, 24 logical CPUs |
| Caches | L1d 576 KiB (12x), L1i 384 KiB (12x), L2 12 MiB (12x), L3 16 MiB (1x) |
| SIMD | AVX2, FMA, AVX-512F/DQ/BW/VL, AVX-VNNI, AVX-VNNI-256, AVX512-BF16, GFNI, VAES |
| RAM | 11 GiB total (12046016 kB), 3.0 GiB swap |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2 (x86_64), glibc 2.39 |
| Virtualization | WSL2 under the Microsoft hypervisor, full virtualization (AMD-V) |
| CPU governor | unknown (VM or fixed frequency) — WSL2 does not expose the governor |
| Benchmark threads | 12 (one per physical core) |

Same host as the 2026-11-07 baseline, again inside WSL2, so the same caveats apply: the
Windows host owns the CPU, the governor is not visible and cannot be pinned.

The harness's streaming-bandwidth probe was **not** stable this session: it returned
29.29 GB/s for the first invocation, 47.91 GB/s for the second, and 30.55–45.12 GB/s for the
rest, against a single 35.43 GB/s in the 2026-11-07 machine record. The probe streams a
192 MiB buffer three times and keeps the best pass, so this is host-level variance (the
earlier README already reported an independent probe spanning 33–72 GB/s). Consequences:
the "% of ceiling" column is only meaningful per run and is not compared across days here,
and everything below uses raw tok/s.

## What the numbers say

The November work shows up in exactly one place, and it is the one the roadmap expected:
**prompt throughput on the 1B**. Peregrine's prefill at 512 tokens went from 147.0 to
173.5 tok/s (+18%) and at 4096 from 116.8 to 125.4 tok/s (+7%); the 0.5B is unchanged
(271.1 → 270.6 at 512, 200.7 → 184.9 at 4096). So llama.cpp still leads prefill on both
models and both contexts, but the 1B gap narrowed from 0.58x to 0.66x at 512 and from
0.62x to 0.66x at 4096.

The repacked kernels are a clear win, but only on the 1B. With `PEREGRINE_KREPACK=1` the 1B
prefills at 211.5 tok/s at 512 (+22% over the default build, 0.81x of llama.cpp) and 144.6
tok/s at 4096 (+15%, 0.77x), its 10-user throughput reaches 62.7 tok/s at 512 (1.02x
llama.cpp, and its 4-user figure is 56.4 vs 54.5 — the first time Peregrine leads
llama.cpp on the multi-user numbers) and 8.4 tok/s
at 4096 (0.98x), and its 10-user TTFT drops from 20.95 s to 15.16 s at 512 (2.7% *better*
than llama.cpp) and from 204.6 s to 177.5 s at 4096 (still 1.23x llama.cpp's). On the 0.5B
the same switch does nothing measurable (prefill −2.6% at 512, −2.4% at 4096, within the
spread or just outside it) and costs up to 2% of decode on both models. A plausible reading is
that the repack pays where the weight matrices are large enough for the eight-row tiles to
earn their extra packing, and the 0.5B's are not; the 1B's are. That is a hypothesis, not a
measurement, and the A/B in `docs/design/s3.4.md` should be extended to the 0.5B before the
default is changed for it.

Decode is roughly where it was: Peregrine still beats llama.cpp on the 1B by a wide margin
(1.16x at 512, 1.36x at 4096, unchanged) and on the 0.5B at 4096 (1.06x), and still trails on
the 0.5B at 512 (0.90x).

Three numbers moved the wrong way and are worth calling out rather than smoothing over:

- **0.5B at 4096 lost ~13% of multi-user throughput** (12.0 → 10.5 tok/s at 10 users,
  12.1 → 10.3 at 4 users) and its 10-user TTFT rose 16% (121.3 → 141.2 s), while llama.cpp on
  the same file and context was flat (13.4 tok/s, 91.0 s). This is not a machine-wide drift.
  The KREPACK pass shows the same numbers, so the repack is not the cause.
- **10-user TTFT at 512 got worse on both models** (0.5B 8.96 → 12.16 s, 1B 18.29 → 20.95 s)
  even though llama.cpp's TTFT improved or held (11.06 → 9.54 s, 15.19 → 15.58 s). The
  prefill gains do not show up in the 10-user queueing behaviour; if anything they are
  swamped by whatever changed. The 2026-11-07 TTFT samples had large spreads (±1.3 s and
  ±4.8 s), but the 2026-11-08 intervals do not overlap the old medians on the 0.5B.
- **Decode is 3% lower on both models** at both contexts (0.5B 66.5 → 68.1 at 512 is the
  exception, +2.5%). Small, but consistent, and not explained by the repack.

One caveat on all of the above: the 4096-token sweeps ran one engine per invocation instead
of interleaving Peregrine and llama.cpp inside one invocation, because the full sweep does
not fit a 10-minute command on this host. The 512-token sweeps kept the interleaved ABAB
order. llama.cpp's own numbers moved by up to 7.1% (Qwen 512 decode 70.8 → 75.8 tok/s) and
by 0–4% on the other rate metrics, so the host ran slightly faster than on 2026-11-07; the
Peregrine/llama comparison at 4096 is therefore less tight than the 2026-11-07 one, and the
Peregrine deltas above should be read against that drift rather than as pure code effects.

One more detail about the 4096 Peregrine pass: an attempted interleaved Llama 4096 run was
started by hand while the queued Peregrine pass was starting, the two fought over port 8201,
and the hand-started run was discarded after its calibration produced no valid sample. The
kept Peregrine pass overlapped it for about ninety seconds; its three single-user samples
(44.7, 43.6 and 44.2 tok/s) show no outlier, and its multi-user samples are tight.

The full tables, with medians and spreads, are in [`report.md`](report.md). The raw JSON
manifests next to it carry every individual sample, the machine record and the engine builds.
