# S5.1, round 3: the prompt-speed measurements reconciled

Raw measurements for `docs/design/s5-prefill-avx512.md` §10, 2026-11-12 (03:40–05:10 local,
UTC+7) on the desktop: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads, AVX-512
F/BW/VL/VNNI/VBMI/BF16), WSL2, 11.7 GiB.

Peregrine: this branch (`s5.1-reconcile`, cut from `s5.1-q4k-kernel-2`, i.e. `origin/main` +
the wide pass + §9's write-up), `cargo build --release -p peregrine` with
`CARGO_TARGET_DIR=~/.cache/peregrine-bench/target-reconcile`.
ik_llama.cpp: `479074c`, the build PR #177 measured, at
`~/.cache/peregrine-bench/ik_llama.cpp-479074c` — not rebuilt, because it is that build:
`isa.txt` shows the runtime instruction-set probe, `ldd` of the `llama-bench` that produced
every fork row, its CMake flags and its 26,526 `vpdpbusd`.

One model process at a time; every command in the foreground under ten minutes; the host's
1-minute load is recorded in every row (`load`).

## The clock, first

This host's clocks disagree with each other and the disagreement **changes between runs**, so
no number here is trusted alone. Three independent references:

| probe | what it says |
|---|---|
| `clockprobe.c` (3 s burn, pinned) | MONOTONIC 3.000 s, MONOTONIC_RAW 2.78 s, CPU time 2.78 s, TSC 5.5 Gticks |
| `hostclock.py 40` (40 s burn, remote HTTP `Date` as the clock) | host 38 ± 2 s, MONOTONIC 40.02 s (**+5.3 % fast**), CPU 37.18 s |
| `roof_abs.sh` (a fixed count of `vpdpbusd`, bracketed by two `Date` headers) | 113.8 GMAC/s in **real** seconds, twice, against 111–120 by the guest's own clocks |

So: CLOCK_MONOTONIC — which both engines time with — ran 5–8 % fast in this session, and
earlier in the same session it ran 1.6–1.8× fast (a 3 s burn read MONOTONIC 1.76 s against
1.00 s of CPU time). That is the error class that produced the 455 GMAC/s in §1.2: a rate
measured on a fast clock is *understated*, and the reverse. Every number below is the
engines' own wall-clock rate, which is what a user waits for; every rate derived from a
probe I timed (`peak.c`, `roof_abs.c`, `kqbench`) uses the *minimum* of CLOCK_MONOTONIC_RAW
and CPU time over 5–7 timed passes, and `roof_abs.sh` checks the absolute value against the
host clock.

Cores: `lscpu -e` gives CPUs 0–7 = cores 0–3 (the four full cores) and CPUs 8–23 = cores
4–11 (the compact ones), two SMT threads per core. A fixed-work loop timed on every one of
the 24 CPUs gave the same CPU time on all of them, so no core is distinguishably faster on
this host today; one-thread rows are pinned to **CPU 4** (core 2, a full core) and the
"fast core" question is answered in §10.1 of the design note.

## Files

| file | what it is |
|---|---|
| `sweep.py` | the harness: one point at a time, points interleaved round by round, 5 rounds, raw JSONL. Records each invocation's wall time *and* its CPU time (the two clocks above) and the host load. |
| `fork-{1,12}t-*.jsonl` | the fork at `-p 512 -n 0 -t N -rtr 1 -ub 8…512`, `llama-bench`, medians of 5 |
| `peregrine-{1,12}t-*.jsonl` | `peregrine bench --prompt-tokens 512 --tokens 1 --threads N --no-profile --json` with `PEREGRINE_PREFILL_CHUNK` 8…128 |
| `*-run2.jsonl` | the second run of the same series, 70 minutes later |
| `clock.jsonl`, `clockprobe.c` | the clock calibration above |
| `hostclock.txt`, `hostclock.py` | the same against the host's clock |
| `peak.c`, `peak.txt` | the roof and the core clock: `vpdpbusd` ymm/zmm with 1–16 named accumulators, register and L1 operands, `vfmadd512`, dependent `add` and `imul` chains |
| `roof_abs.c`, `roof_abs.sh`, `roof_abs.txt` | the same roof at a fixed instruction count, timed by the host clock |
| `ik_kernel_probe.cpp`, `ik_kernel_probe.txt` | the fork's kernel through the fork's own library at a real layer's shape (2048 × 8192), 8/16/64/512 activation rows, printing the code path the library picks |
| `kqbench-layer.txt` | the same shapes for Peregrine's kernels: `KQBENCH_LAYER=1 kqbench 8 40 1` (the mode added to the example in this branch) |
| `macs.py`, `macs.txt` | multiply-accumulates per prompt token, read from each model's own tensor shapes |
| `table.py`, `table.txt` | the ground-truth tables of §10.2, generated from the JSONL |
| `isa.c`, `isa.txt` | the fork library's instruction sets, `ldd`, CMake flags and `vpdpbusd` count |

## Commands

```sh
# end to end, 5 interleaved rounds per point; ONLY picks the model, TAG names the run
ONLY=llama1b python3 sweep.py . fork-1t
ONLY=llama1b python3 sweep.py . peregrine-1t
TAG=-run2 ONLY=llama1b python3 sweep.py . fork-1t       # the second run
# ... and the same for qwen05, fork-12t, peregrine-12t

# clocks and the roof
python3 sweep.py . clock
taskset -c 4 python3 hostclock.py 40
gcc -O2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mfma peak.c -o peak && taskset -c 4 ./peak 7
gcc -O2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni roof_abs.c -o roof_abs && ./roof_abs.sh 2000000000

# the fork's kernel at a real layer's shape (no fork source in this repository)
FORK=~/.cache/peregrine-bench/ik_llama.cpp-479074c
g++ -O2 -std=c++17 -I $FORK/ggml/include benchmarks/ik_kernel_probe.cpp -o ik_kernel_probe \
    -L $FORK/build/ggml/src -lggml -Wl,-rpath,$FORK/build/ggml/src
taskset -c 4 ./ik_kernel_probe q4k 8192 8,16,64,512 5
taskset -c 4 ./ik_kernel_probe q6k 8192 8,16,64,512 5

# Peregrine's kernels at the same shape
KQBENCH_LAYER=1 taskset -c 4 kqbench 8 40 1

# the MAC counts
python3 macs.py ~/.cache/peregrine-bench/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

## Models

The same two files as every earlier desktop run, checked by size and by the shapes
`macs.py` reads out of them: Llama 3.2 1B Instruct Q4_K_M (807,694,464 B, 16 layers,
0.9731 GMAC per prompt token) and Qwen 2.5 0.5B Instruct Q4_K_M (491,400,032 B, 24 layers,
0.3579 GMAC per prompt token). Both are symlinks into `~/.cache/peregrine-bench/models/`.
No Bonsai file was opened.
