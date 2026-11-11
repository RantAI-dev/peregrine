# S5.1a: a wide prefill pass when nobody is decoding

Raw measurements for `docs/design/s5-prefill-avx512.md` §8, taken on the desktop
(AMD Ryzen AI 9 HX 370, WSL2) on 2026-11-12 with the S5.1 stack (`s5.1-kq-epilogue`).

- `measure.py` — one interleaved sweep of `peregrine bench --json` over prompt lengths and
  chunk widths (or the engine's default, `chunk 0`). `--isa avx2` caps the tier.
- `mixed.sh` — one `benchmarks/run.py --mode server` invocation; `before` exports
  `PEREGRINE_PREFILL_CHUNK=16` and `after` leaves it unset, so the two variants are the same
  binary.
- `avx512-*.json`, `avx2-*.json` — the chunk sweep (16, 64, 128) that chose the wide chunk.
- `after-avx512-*.json`, `after-avx2-*.json` — before (chunk 16) against after (default), three
  interleaved rounds per model and tier.
- Multi-user rows are the harness's own JSON under `~/.cache/peregrine-bench/pr1-mixed-*`
  (four invocations per variant for the 1B, three plus two for Qwen); the tables in §8.3 are
  the medians of those samples.
