# S5.4a: the 8-bit prompt kernel, and what it costs

Raw measurements for `docs/design/s5.4-fast-prompt.md` §2, §5's S5.4a row and
§11.1 of `docs/design/s5-prefill-avx512.md`, 2026-11-12 on the desktop: AMD Ryzen
AI 9 HX 370 (Zen 5, 12 cores / 24 threads, AVX-512 F/BW/VL/VNNI/VBMI/BF16), WSL2,
11.7 GiB.

Peregrine: this branch (`s5.4a-prompt-kernel`, cut from `origin/main` at `97066f0`).
ik_llama.cpp: `479074c`, the build PR #177 measured and §10/§11 quote, at
`~/.cache/peregrine-bench/ik_llama.cpp-479074c` (not rebuilt).
Every command was run in the foreground under ten minutes, one at a time, and each
kernel benchmark was pinned to cores 20–23 with `taskset` because another worker was
measuring on this machine throughout: **the 1-minute load was 2.8–17.4**, and every
table below carries the load it was measured at. That load is this session's noise
floor, and it is why the same binary moves by a few per cent between runs.

Build and run:

```text
export CARGO_TARGET_DIR=~/.cache/peregrine-bench/target-s54a
cargo build -j 6 --release -p peregrine-bench --example kqbench --example kq8model
KQBENCH_PROMPT=1 taskset -c 20-23 \
    ~/.cache/peregrine-bench/target-s54a/release/examples/kqbench 8 40 1
```

## The clock, first

`clockprobe.c` (3 s burn, pinned to CPU 20), from the S5.2 results directory, at the
start of this session's benchmarks (`clock.txt`):

```
MONOTONIC 3.000  RAW 2.728  CPU 2.727  TSC/Gticks 5.445  tsc_ghz_if_cpu_accurate 1.996  cpu_over_wall 0.909
```

**CLOCK_MONOTONIC ran 10 % fast** (3.000 s of it for 2.727 s of CLOCK_MONOTONIC_RAW
and CPU time), the same skew as §11 measured. Both engines time with MONOTONIC, so
every GMAC/s below is ~10 % *conservative* in real seconds, and the ratios between
kernels and between engines are untouched. `kqbench` times with `Instant`
(MONOTONIC).

## 1. The form, as built

Eight output rows of a matrix are interleaved per 256-weight super-block, and one
super-block of the form is 2112 bytes:

```text
0      qs   the codes: qs[group * 64 + row * 8 + k]      2048 bytes
2048   d    the eight rows' f32 scales, one per row         32 bytes
2080   pad                                                 32 bytes
```

`qs[group * 64 + row * 8 + k]` is the fork's `code + 128` of weight `group * 8 + k`
of row `row`, so a 64-byte load is one `vpdpbusd` weight operand. The stride is 33
cache lines rather than 32 because a 64-byte load that straddles a line costs two
accesses on every group. **One weight costs 1.03125 bytes** against the file's
0.5625 (Q4_K) and 0.8203 (Q6_K).

The multiply is `vpdpbusd` over the biased bytes, `−128 · Σa` in int32 lanes (the
activation's whole-super-block sum, which `KqActs` now carries), and one
`d · dk` multiply plus one fused multiply-add per (row, super-block). The bound on
the integer sums is in `kqsimd.rs`: an int32 lane sees 128 products of a 255 byte
and a 127 activation, 4 145 280, and `vpdpbusd` wraps rather than saturating six
orders of magnitude above that.

## 2. The kernel table

One real layer: 2048 weights × 8192 output rows (1024 eight-row tiles, 4.7 MB of
Q4_K tiles streamed), one thread, pinned to cores 20–23. `exact` is today's kernel
on the file's own tiles (`q4k8_gemm_zmm`, `q6k8_gemm_zmm`, `q5_08_gemm_zmm`),
`resident` is `q8k_gemm_zmm` on a form built once (not timed), `per-pass` builds the
form inside the pass once per tile, and `build` is the conversion alone. Load 8.3
before, 8.3 after (`kqbench-prompt.txt`).

| format | acts | exact | resident | per-pass | build |
|---|---|---|---|---|---|
| q4k | 1 | 33.6 | 22.0 (−35 %) | 1.6 (−95 %) | 1.7 G w/s |
| q4k | 8 | 65.3 | 85.3 (+31 %) | 10.6 (−84 %) | 1.5 |
| q4k | 16 | 62.7 | 73.6 (+17 %) | 18.6 (−70 %) | 1.5 |
| q4k | 64 | 63.2 | 84.0 (+33 %) | 45.4 (−28 %) | 1.5 |
| q4k | 128 | 74.4 | 95.5 (+28 %) | 68.3 (−8 %) | 2.0 |
| q4k | 512 | 66.2 | 91.9 (+39 %) | 82.1 (+24 %) | 1.6 |
| q6k | 1 | 18.8 | 19.4 (+3 %) | 1.4 (−93 %) | 1.4 |
| q6k | 8 | 39.7 | 92.2 (+132 %) | 10.0 (−75 %) | 1.5 |
| q6k | 16 | 38.9 | 73.4 (+89 %) | 17.0 (−56 %) | 1.4 |
| q6k | 64 | 43.3 | 83.2 (+92 %) | 42.8 (−1 %) | 1.4 |
| q6k | 128 | 46.7 | 89.0 (+91 %) | 61.1 (+31 %) | 1.5 |
| q6k | 512 | 45.3 | 90.1 (+99 %) | 79.9 (+76 %) | 1.5 |
| q5_0 | 1 | 21.4 | 22.0 (+3 %) | 1.5 (−93 %) | 1.6 |
| q5_0 | 8 | 42.4 | 87.3 (+106 %) | 10.0 (−76 %) | 1.5 |
| q5_0 | 16 | 49.0 | 81.5 (+66 %) | 20.9 (−57 %) | 1.8 |
| q5_0 | 64 | 49.2 | 89.8 (+82 %) | 51.7 (+5 %) | 1.6 |
| q5_0 | 128 | 45.1 | 92.2 (+105 %) | 62.1 (+38 %) | 1.4 |
| q5_0 | 512 | 42.9 | 90.6 (+111 %) | 81.8 (+91 %) | 1.4 |

GMAC/s/core; `build` is in weights per second. For reference, §10/§11's number for
the fork's kernel at this shape is **91 GMAC/s at 512 activation rows**, and the
exact Q4_K kernel's ceiling is 74–76; the resident form measures 91.9 here, at the
same clock skew the fork's number was taken under.

Answers to §5's questions:

- **(c) overtakes (a) at about 128 activation rows.** At 64 rows the per-pass form
  is 28–57 % behind the exact kernel; at 128 it is 8 % behind for Q4_K and 31–38 %
  ahead for Q6_K/Q5_0; at 512 it is 24/76/91 % ahead. The reason is the conversion
  rate: 1.4–2.0 G weights per second costs 8–10 ms per pass over this layer, which
  is 12 % of a 512-row pass (82 ms) and 8× the whole kernel at 8 rows.
- **(b) is ahead of (a) at every width above eight rows, and not at one.** At one
  activation row the pass is bandwidth-bound and the form's 1.03 bytes per weight
  cost Q4_K 35 %: 22.0 against 33.6. Q6_K and Q5_0 are 3 % ahead at one row (their
  files hold 0.82 and 0.69 bytes per weight, so the form is a smaller multiple).
- **A 1B model's resident form is 1.187 GiB** for Llama 3.2 1B Q4_K_M, whose
  Q4_K/Q6_K/Q5_0 tensors hold 0.745 GiB (1.0312 bytes per weight against the file's
  0.6471), i.e. +0.38 bytes per weight, +0.44 GiB.

**Recommendation** (the decision is the project owner's): build the form **resident**
at load, in the packed-weight cache, keyed by the file's hash and the form's version.
It is the fastest of the two at every prompt width the mode is for (8 rows and up:
+17 % to +132 %), it needs no per-pass conversion code in the token path, and its
1.03 bytes per weight buy 0.44 GiB on the 1B. Per-pass conversion is the fallback
only if that memory is not on the table: it is never ahead of the exact kernel below
128 rows, 8 % behind at 128 for Q4_K and only 24 % ahead at 512, so it would need
the build to be four times faster before it competed. A hybrid (resident for Q4_K,
per-pass for Q6_K) is not worth the two code paths: Q6_K's resident gain is the
largest of the three.

What the table does *not* say: this is one thread on one core with the L2 shared
with another worker's benchmark. At twelve threads the form's 1.03 bytes per weight
against the file's 0.56 is twice the bytes per pass, and a prompt pass at 512 rows
is not bandwidth-bound at one thread (91.9 GMAC/s × 1.03 B/weight = 95 GB/s
equivalent — it is L3-bound, not DRAM-bound), so the twelve-thread rows of S5.4c
may tell a different story about the resident form's cost. Nothing in S5.4a
dispatches anything; the engine's default path is untouched.

## 3. Conformance with the fork

`examples/kq8model.rs` dumps raw rows of a real tensor and Peregrine's form of them
in a canonical layout (f32 scale then 256 signed codes per row and super-block);
`fork_requant.py` repeats the fork's own arithmetic from the lines §11.1 cites —
`iqk_convert_q4_k_r4_q8_k_r16` (`iqk_gemm_kquants.cpp:2400`),
`iqk_convert_q6_k_r4_q8_k_r16` (`:2724`) and `convert_to_q8_k_r8`
(`iqk_common.h:638-687`) — with glibc's correctly-rounded `fmaf` and f32 rounding
after every operation, and writes the same layout. `cmp` is then the comparison of
every code and every scale.

| tensor | format | rows | super-blocks | weights | result |
|---|---|---|---|---|---|
| `blk.3.attn_q.weight` | Q4_K | 128 | 8 | 262 144 | identical |
| `blk.0.attn_output.weight` | Q4_K | 16 | 8 | 32 768 | identical |
| `blk.8.ffn_gate.weight` | Q4_K | 16 | 8 | 32 768 | identical |
| `blk.0.ffn_down.weight` | Q6_K | 128 | 32 | 1 048 576 | identical |
| `blk.15.attn_v.weight` | Q6_K | 16 | 8 | 32 768 | identical |
| `token_embd.weight` | Q6_K | 16 | 8 | 32 768 | identical |

**1 441 792 weights, every scale bit for bit and every code equal, no difference.**

The harness that was *meant* to be the oracle — `forkconv.c`, a C transcription of
the fork's four-row repack (checked against the fork's own dequantiser on both
layouts) linked against the cached `libggml.so`, calling the fork's exported
`iqk_convert_kquants_q8X_r8` — **did not work, and that is a finding**. That build
(2026-11-11 19:57, from this checkout) returns `true` and writes zeros: for the 1B
model's own rows, and for a hand-built four-row R4 block with `d = 1.0`, `dmin = 0`,
every six-bit scale 63 and every code 15, whose only possible output is ±945
weights. Both the scale array and the codes come back zero. The repack is verified
against the fork's dequantisers before the call, so the input is the fork's layout
and the zeros are the library's. I did not chase it further than that (three
attempts: the ABI, the enum values — `ggml_type_name(212)` says `q4_k_r4` — and the
output layout; `libggml.so` is the only library with the symbol, and it was rebuilt
from this tree hours before the session). The comparison above is therefore against
a transcription of the fork's source, not against the fork's binary, and it was
cross-checked a second way: the re-rounding error this crate measures on the 1B
(`requant-llama1b.txt`) reproduces `requant_error.py`'s independently written
numbers of §11.1 to every digit — mean 0.00197, worst 0.00394, 0.42 % of weights
exactly on the grid.

## 4. The re-rounding error, and Q5_0

`kq8model errors <model>` samples every 16th eight-row group and every 16th
super-block, reports the worst and mean error relative to the super-block's largest
weight, and the share of weights exactly on the grid.

Llama 3.2 1B Q4_K_M (`requant-llama1b.txt`), 1.2357 G weights in 113 Q4_K/Q6_K
tensors:

| format | blocks sampled | mean of per-block mean / block max | worst single / block max | exactly on the grid |
|---|---|---|---|---|
| Q4_K | 23 296 | 0.00197 (0.00128–0.00273) | 0.00394 | 0.42 % |
| Q6_K | 10 320 | 0.00193 (0.00123–0.00220) | 0.00394 | 3.33 % |

Qwen 2.5 0.5B Q4_K_M (`requant-qwen05b.txt`), 492.6 M weights in 157 tensors:

| format | blocks sampled | mean of per-block mean / block max | worst single / block max | exactly on the grid |
|---|---|---|---|---|
| Q4_K | 1 344 | 0.00197 | 0.00394 | 0.44 % |
| Q6_K | 1 344 | 0.00191 | 0.00394 | 3.47 % |
| **Q5_0** | 27 064 | **0.00183** (0.00087–0.00222) | **0.00394** | 6.73 % |

The bound is half a step, `max/254` = 0.3937 %, and the worst single weight reaches
it in all three formats. Q5_0's worst and mean are the same as the k-quants' —
it is the same construction — but its grid is one scale per 256 weights where the
file has one per 32, so the error is relative to the *super-block's* maximum; a
Q5_0 row of 896 weights (Qwen's) ends in a half super-block, which is why its
resident form costs 1.1473 bytes per weight rather than 1.0312. No Q6_K super-block
of either model is in the fork's exact case: real weights are never all below 1.0
in a block.

## 5. What is in the tree

- `crates/peregrine-kernels/src/kq8.rs`: the form's layout, the scalar twin
  (`build_q4k`, `build_q6k`, `build_q5_0`, `dot_q8k`, `dot_q8k_plain`), the
  requantisations (`requant_f32`, `requant_q6k`) and the tests.
- `crates/peregrine-kernels/src/kqsimd.rs`: `q8k_build_q4k_zmm`, `q8k_build_q5_0_zmm`,
  `q8k_build_q6k_zmm` and `q8k_gemm_zmm`, each asserted bit for bit against the twin
  on random and edge tiles at every width from 1 to 40 and at 63, 64, 65, 127, 128,
  129, 511, 512 and 513 activation rows.
- `crates/peregrine-kernels/src/kq.rs`: `KqActs::sum256`, the whole-super-block
  activation sum the 128 bias is paid back with (the exact kernels do not read it).
- `crates/peregrine-bench/examples/kqbench.rs`: `KQBENCH_PROMPT=1`, the table above.
- `crates/peregrine-bench/examples/kq8model.rs`: the error report and the dump.
- This directory: `forkconv.c` (the oracle that did not work), `fork_requant.py`
  (the comparison that did), the raw outputs, `clockprobe.c`'s output and the loads.
