# Bluefin (H2) baseline: Peregrine vs llama.cpp (roadmap S3.2)

The first baseline for host H2, the Ryzen 3 4350G desktop: the harness (`benchmarks/run.py`) run
against llama.cpp v0.9.0 on the same machine, the same model files and the same thread count. H2
and H3 are the AVX2-only machines in the S3.2 plan; every figure here is from this host, not from
the Zen 5 desktop.

- Date: 2026-11-08 (runs 10:25-12:40 local).
- Peregrine: `main` at `5f76d48` (`origin/main`), built `release`
  (`cargo build --release -p peregrine`), `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- llama.cpp: v0.9.0 (`cd959ba`), prebuilt at `/var/tmp/llama.cpp/build/bin` (CMake Release,
  `GGML_NATIVE=ON`, so the Zen 2 host's AVX2/FMA paths are compiled in), GNU 16.2.1.
- Models, both from `~/models`, SHA-256 in the manifests:
  - `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807694464 bytes,
    `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`;
  - `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, 397808192 bytes,
    `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`.
- Settings: contexts 512 and 4096, 3 interleaved rounds per engine, 256 greedy decode tokens per
  request, server mode, prefix-cache defeat by per-request nonce, prompt lengths calibrated per
  engine and model (every measured prompt within 5% of the target, all within 1.3%).

## The machine

| Property | Value |
|---|---|
| CPU | AMD Ryzen 3 PRO 4350G with Radeon Graphics (Zen 2), family 23, model 96, stepping 1 |
| Topology | 1 socket, 4 physical cores, 2 threads per core, 8 logical CPUs |
| Caches | L1d 128 KiB (4x), L1i 128 KiB (4x), L2 2 MiB (4x), L3 4 MiB (1x) |
| SIMD | AVX, AVX2, FMA (no AVX-512, no VNNI) |
| RAM | 13.5 GiB total (14171752 kB), 8 GiB swap (3.9 GiB in use) |
| OS | Linux 7.2.7-200.fc44.x86_64 (Fedora 44), glibc 2.43 |
| Virtualization | none (bare metal) |
| CPU governor | `powersave` (amd-pstate; boost enabled, frequency not pinned) |
| Benchmark threads | 4 (one per physical core) |

The governor is `powersave` and was left alone: this run does not change host state. Frequency
therefore floats with the load the benchmark itself creates, and the first seconds of each phase
are at a lower clock than the steady state. That affects the absolute numbers, not the
engine-to-engine comparison, since both engines ran under the same governor in interleaved rounds.

Peregrine's startup line recorded `avx2+fma`, 4 threads and "engine options: all defaults", so the
repacked Q4_K kernels were on. No `EngineOptions` were overridden and no environment variables
were set.

## Quiet host

Load average was 0.08 (1 min) before the first run. During the runs it sat at 3.7-4.2, which is
the benchmark itself (4 worker threads). Nothing else was killed or stopped. Background load
observed while benchmarking: `localsearch-3` (desktop indexer, ~2% CPU), `gnome-shell` and other
session processes (~1% total), and `kswapd0` at ~3% with 3.9 GiB of swap already in use. Nothing
appeared that would move a decode number materially; the round-to-round spread in the report
(mostly under 1%) supports that.

## How the runs were split

A full invocation (both engines, 1/4/10 users, one round) takes 2-3 minutes at 512 tokens but
7-9.5 minutes per engine at 4096 on this 4-core host. To keep every command under the 10-minute
cap, runs were split while preserving the A,B,A,B round order:

- 512 tokens: one command per round, both engines in it (2m12s per round for the 0.5B, 3m25s for the 1B).
- 4096 tokens, Qwen2.5-0.5B: one command per engine per round (7m15s Peregrine, 6m47s llama.cpp).
- 4096 tokens, Llama-3.2-1B: one command per round for Peregrine (9m19s); llama.cpp needed two,
  `--user-counts 1,4` (4m58s) and `--user-counts 10` (9m23s), because a single 10-user phase at
  4096 is already most of the cap on this host.

`--user-counts` is a filter added to the harness in this branch (`benchmarks/run.py`); it only
selects which of the three fixed phases execute and changes nothing about how a phase is measured.
The per-invocation manifests were reassembled into one manifest per engine and model with the
same statistics the harness computes in-process (medians and half-range spreads), so the four JSON
files here have the shape a single `-r 3` invocation writes, including every raw sample.

The one command shape, with the engine, context and phase selection varied as above:

```sh
LLAMA_SERVER_BIN=/var/tmp/llama.cpp/build/bin/llama-server \
python3 benchmarks/run.py -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -e peregrine -c 4096 -r 1 -n 256 -t 4 \
  -o /tmp/bench-raw/llama-3.2-1b
#   ... -e llama_cpp --user-counts 1,4    (then)    -e llama_cpp --user-counts 10
```

`LLAMA_SERVER_BIN` points the adapter at the confirmed v0.9.0 build instead of letting it clone and
build its own copy; `run_cli_bench`'s `bytes_per_token` probe still uses the local Peregrine
binary. `-r 1` per command, repeated three times in round order, is what the `-r 3` interleave
does in one invocation.

## What the numbers say

Peregrine wins single-user decode on the 1B model at both contexts (1.15x at 512, 2.82x at 4096)
and on the 0.5B at 4096 (1.10x), but loses the 0.5B at 512 (0.70x). Prompt throughput is at parity
on the 1B (1.04x, 1.02x) and below llama.cpp on the 0.5B (0.86x, 0.83x). Multi-user throughput is
ahead on the 1B (1.35x and 1.49x at 512, 1.43x and 1.52x at 4096) and mixed on the 0.5B (1.08x and
1.04x at 512, 0.91x and 0.96x at 4096). 10-user TTFT is worse on the 0.5B (0.92x and 0.75x), better
on the 1B at 512 (1.17x) and slightly worse at 4096 (0.96x).

Two things are worth calling out:

- llama.cpp's single-user decode on the 1B model falls from 39.5 tok/s at 512 to 13.0 tok/s at
  4096 (0.33x), while Peregrine falls only from 45.4 to 36.7 (0.81x). This is the default
  `llama-server` configuration from the pinned build, with no `-fa` or other extra flags, so it is
  what a default user of that build gets on this host; it is also the single largest gap in the
  suite. The same effect shows up in multi-user: 39.5 -> 5.4 tok/s for llama.cpp at 4 users versus
  53.5 -> 7.7 for Peregrine.
- At 4096 tokens the 10-user figures are small (5-10 tok/s) and the wall-clock phases are minutes
  long, so those medians carry more weight from cache/clock state than the 512-token ones. The
  spreads are still small (mostly under 1%), but read them as "this is what a 4-core Zen 2 does at
  10 x 4096", not as a fine-grained ranking.

The "% of ceiling" column is Peregrine's `bytes_per_token` telemetry times the measured decode rate
over the harness's streaming-bandwidth probe. The probe reported 39.27-39.77 GB/s across the four
invocations that fed these manifests (an earlier standalone probe on this host reported 32.54
GB/s), which is why the same decoder can show different percentages in different rows. Read it as
"how far the decode pushes against a conservative ceiling".

The full table, with medians and spreads, is in [`report.md`](report.md). The raw manifests next
to it carry every individual sample, the machine record and the engine builds.
