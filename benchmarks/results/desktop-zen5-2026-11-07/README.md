# Desktop baseline: Peregrine vs llama.cpp (roadmap S3.2)

The first desktop baseline for Peregrine: the harness (`benchmarks/run.py`) run against
llama.cpp v0.9.0 on the same machine, the same model files and the same thread count.

- Date: 2026-11-07.
- Peregrine: `main` at `ce67e762`, built `release` (`cargo build --release -p peregrine`),
  `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- llama.cpp: v0.9.0 (`cd959ba`), CMake release build, AVX2/FMA CPU.
- Models: `qwen2.5-0.5b-instruct-q4_k_m.gguf` and `Llama-3.2-1B-Instruct-Q4_K_M.gguf`,
  both from the same local model directory, hashed in the manifests.
- Settings: contexts 512 and 4096, 3 interleaved rounds per engine, 256 greedy decode tokens
  per request, server mode, prefix-cache defeat by per-request nonce, prompt lengths calibrated
  per engine and model (every measured prompt within 5% of the target, most within 1%).

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

The host is **WSL2**, not bare metal: the Windows host owns the CPU and WSL2 presents the
virtualized Linux environment, so the scaling governor is not visible and cannot be pinned.
The `Hypervisor vendor: Microsoft` line in `lscpu` and the `-microsoft-standard-WSL2` suffix
in the kernel release both mark it.

## What the numbers say

At 4096 tokens of context Peregrine decodes faster than llama.cpp on both models
(1.15x on the 0.5B, 1.43x on the 1B), and at 512 tokens it wins on the 1B (1.24x) but loses
on the 0.5B (0.94x). Peregrine's prompt throughput is below llama.cpp's at every context
(0.58x to 0.74x), and its multi-user throughput is below it too (0.77x to 0.96x).
Peregrine's 10-user TTFT is better at 512 tokens and worse at 4096.

Two readings of these numbers are worth stating plainly:

- The "% of ceiling" column exceeds 100% for the 1B model (138.8% at 512 tokens). The ceiling
  the harness measures is a conservative estimate: it streams a 192 MiB buffer once per pass
  and keeps the best of three. An independent multithreaded streaming-read probe on this host
  ranged from 33 to 72 GB/s depending on the buffer size and the run, so a decode measured
  at 49 GB/s of weight traffic is physically consistent here even though it is 138% of the
  harness's 35.43 GB/s figure. Read the column as "how far the decode pushes against a
  conservative ceiling", not as a saturation ratio.
- The 10-user figures at 4096 tokens are small in absolute terms (7 to 14 tok/s) and the
  absolute spread is correspondingly large relative to the median.

The full table, with medians and spreads, is in [`report.md`](report.md). The raw JSON
manifests next to it carry every individual sample, the machine record and the engine builds.
