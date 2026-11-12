# S5.2: is the fork's wide pass exact, and what an exact one can reach

Raw measurements for `docs/design/s5-prefill-avx512.md` §11, 2026-11-12 (06:20–08:10 local,
UTC+7) on the desktop: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads, AVX-512
F/BW/VL/VNNI/VBMI/BF16), WSL2, 11.7 GiB.

Peregrine: this branch (`s5.2-exactness`, cut from `origin/main` at `dd6d64c`),
`cargo build --release -p peregrine -p peregrine-bench --example kqbench` with
`CARGO_TARGET_DIR=~/.cache/peregrine-bench/target-s52`.
ik_llama.cpp: `479074c`, the build PR #177 measured, at
`~/.cache/peregrine-bench/ik_llama.cpp-479074c` (not rebuilt; the same binaries round 3 used).
llama.cpp v0.9.0: `~/.cache/peregrine-bench/llama.cpp-v0.9.0`, the build `docs/targets.md`
quotes.

One model process at a time; every command in the foreground under ten minutes; the host's
1-minute load is recorded in every end-to-end row (`load`), and it was 2–5 throughout (other
tenants on the box), which is the noise floor these numbers carry.

## The clock, first

`clockprobe.c` (3 s burn, pinned to CPU 4), three times, at the start of the session:

```
MONOTONIC 3.000  RAW 2.727  CPU 2.725  TSC/Gticks 5.444  tsc_ghz_if_cpu_accurate 1.998  cpu_over_wall 0.908
MONOTONIC 3.000  RAW 2.727  CPU 2.724  TSC/Gticks 5.445  tsc_ghz_if_cpu_accurate 1.999  cpu_over_wall 0.908
MONOTONIC 3.000  RAW 2.728  CPU 2.726  TSC/Gticks 5.445  tsc_ghz_if_cpu_accurate 1.998  cpu_over_wall 0.908
```

So this session is the round-3 situation again: **CLOCK_MONOTONIC ran ~10 % fast** (3.000 s of
it for 2.727 s of CLOCK_MONOTONIC_RAW and CPU time). Both engines time with MONOTONIC, so every
tok/s below is ~10 % conservative in real seconds and every *ratio between the two engines* is
untouched. `kqbench` times with `Instant` (MONOTONIC), so its GMAC/s are 10 % low against the
roof `peak.c` measures with CPU time — the same bias round 3's kernel table carries, which is
why those numbers are compared with those.

## Part 1: the fork's wide pass

The corpus is 40 chunks of 512 tokens of plain text built from this repository's own prose (no
chat template anywhere):

```sh
cat README.md ARCHITECTURE.md CHANGELOG.md CONTRIBUTING.md GOVERNANCE.md SECURITY.md \
    docs/README.md docs/adr/*.md docs/backlog.md docs/benchmark.md docs/compat.md \
    docs/docker.md docs/product.md docs/quickstart.md docs/service.md docs/targets.md \
    docs/design/moe.md docs/design/prefill.md docs/design/s2.6.md docs/design/s2.8-bonsai2.md \
    docs/design/s2.8-neon.md docs/design/s3.4.md docs/design/s3.5-delta-scan.md \
    docs/design/s4-embeddings-decisions.md docs/design/s4.9-sota.md \
    docs/design/s5-prefill-avx512.md docs/design/s5.1-kq-epilogue.md \
    docs/design/s5.1-q6k-zmm.md docs/design/s5.1-wide-pass.md docs/dev/*.md \
    docs/reference/bert.md docs/reference/bonsai2_spec.md \
    docs/reference/gemma-embedding.md docs/reference/mistral3.md > corpus.txt
```

(The `docs/design/s6-*` and `docs/reference/systemone.md` files are excluded: they are chat
templates.)

Every run is `llama-perplexity -m <model> -f corpus.txt -c 512 -t 4 --chunks 40`, with `-ub 16`
or `-ub 512`; the base run additionally writes `--save-all-logits`, and the compared run reads
it with `--kl-divergence --kl-divergence-base`. Four log-probability files of 2.6 GB each were
produced (`u512.logits`, `u16.logits` for the fork; `l512.logits`, `l16.logits` for llama.cpp
v0.9.0) and are **not** committed; what is committed is the tool's own summary (`kl.txt`), the
byte comparison of the two files (`bitcmp.txt`) and the commands above.

## Part 2: the widened exact wide pass

`kqbench`'s `KQBENCH_WIDE` mode (this branch) widens a real layer's tile (2048 x 8192, 1024
eight-row tiles, 9.4 MB streamed) to one byte per weight once, then runs the same kernel over
the pass; every width is checked bit for bit against `q4k8_gemm_zmm` before it is timed, and
`crates/peregrine-kernels/src/kqsimd.rs`'s
`the_widened_wide_pass_is_bit_identical_to_the_narrow_one` asserts the same at every batch
width from 4 to 40.

```sh
KQBENCH_LAYER=1 KQBENCH_WIDE=1 taskset -c 4 kqbench 8 40 1
```

Two runs, 30 minutes apart, in `kqbench-wide.txt`.

## Part 3: the Q5_0 kernel

```sh
KQBENCH_LAYER=1 taskset -c 4 kqbench 8 40 1        # one real layer's stream
taskset -c 4 kqbench 8 40 1                        # one L1-resident tile
```

The variants measured and withdrawn (a 16-output-row fold over two adjacent tiles, and a
cheaper fifth-bit spread with a prefetch) are in `kqbench-q5.txt`; only the numbers are kept,
the code was reverted, because none of them was faster and the design note says so.

## End to end

`e2e.py` (this directory): both engines at their default pass widths — the fork's `-ub 512`
default and Peregrine's default chunk (the adaptive wide pass) — 512 prompt tokens with no
decode, then 128 decode tokens after a 16-token prompt, one thread (pinned to CPU 4) and
twelve, three rounds interleaved, medians in `e2e.txt`, every row in `e2e.jsonl` with its wall
time, its CPU time and the host load.

```sh
python3 e2e.py e2e.jsonl
```

## Models

Llama 3.2 1B Instruct Q4_K_M (807,694,464 B, 0.9731 GMAC per prompt token) and Qwen 2.5 0.5B
Instruct Q4_K_M (491,400,032 B, 0.3579 GMAC per prompt token), the same two files as every
earlier desktop run, under `~/.cache/peregrine-bench/models/`. No Bonsai file was opened.
