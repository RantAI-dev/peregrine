# Desktop prefill study: why ik_llama.cpp processes prompts faster (2026-11-11)

Raw measurements behind [`docs/design/s5-prefill-avx512.md`](../../../docs/design/s5-prefill-avx512.md).
Nothing here changes the engine: every file is a run of the two binaries, or a probe that
measures the machine.

- Host: AMD Ryzen AI 9 HX 370 (Zen 5, 4 full + 8 compact cores, 12 cores / 24 threads),
  AVX-512 F/BW/VL/VNNI/VBMI/BF16, WSL2, 11.7 GiB RAM. `lscpu -e` reports no MHz here.
- Peregrine: `origin/main` `267ac22`, `cargo build --release -p peregrine`.
- ik_llama.cpp: `master` `479074ccac2ac94f06b6d24a9ce26f0f9e9aa6ae`, built in
  `~/.cache/peregrine-bench/ik_llama.cpp-479074c` with
  `-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON
  -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON`.
- Models (the same files as `desktop-zen5-2026-11-11-ik/`):
  Qwen 2.5 0.5B Instruct Q4_K_M `74a4da8c…` (491,400,032 B),
  Llama 3.2 1B Instruct Q4_K_M `6f85a640…` (807,694,464 B).
- Prompt length 512 tokens throughout. Decode is not measured here.

## The driver

`benchmarks/prefill_study.py` runs every series and writes one JSON per series. Every series
runs its points **interleaved** (one round = every point once), and every reported number is a
median over the rounds. Each JSON holds every raw sample, the environment of the run
(`env`, `taskset`), and a `summary` keyed by tag.

```
python3 benchmarks/prefill_study.py <this dir> <series> [...]
```

| file | series | what it holds |
|---|---|---|
| `peregrine-threads.json` | `peregrine-threads` | prompt tok/s at 1, 2, 4, 8, 12, 24 threads, both models, `PEREGRINE_PREFILL_CHUNK=16` fixed so the curve is threads alone |
| `peregrine-chunk.json` | `peregrine-chunk` | `PEREGRINE_PREFILL_CHUNK` 16/32/64/128 at 12 threads |
| `peregrine-chunk-1t.json` | `peregrine-chunk-1t` | the same sweep at one thread |
| `peregrine-krepack.json` | `peregrine-krepack` | `PEREGRINE_KREPACK` default / `1` / `0`, 5 rounds |
| `peregrine-isa.json` | `peregrine-isa` | `PEREGRINE_ISA=native` (ZMM kernel) against `avx2`, 5 rounds |
| `peregrine-pin.json` | `peregrine-pin` | `taskset` on cores 0-3, 4-7, 8-11, SMT, free; plus 24 threads at chunk 64 |
| `peregrine-profile.json` | `peregrine-profile` | `PEREGRINE_PROF2=full` phase dumps, repack on and off (the dumps are in the rows' `prof` field) |
| `ik-threads-q.json`, `ik-threads-l.json` | `ik-threads-q`, `ik-threads-l` | `llama-bench` prompt tok/s at the same thread counts (one model per invocation: the 1-thread rows do not fit the ten-minute cap together) |
| `ik-ub.json` | `ik-ub` | `-ub` 64/128/256/512 |
| `ik-rtr.json` | `ik-rtr` | `-rtr 0/1` × `-fa 0/1` |

`llama-bench` rows carry `avg_ts` (a *mean*) and `samples_ts`; the summary uses the median of
`samples_ts` because WSL2's clock jumps make single samples meaningless (one sample in this
session read a negative time).

## The kernel measurements

| file | what it is |
|---|---|
| `kernels-peregrine.txt` | `kqbench` (added widths 16 and 128 in this branch), 3 rounds, `taskset -c 2`: GMAC/s per core for the Q4_K/Q5_K/Q6_K/Q5_0 row and tile kernels at 1, 16, 64, 128 activation rows |
| `ik_kqbench.cpp`, `kernels-ik.txt` | ik_llama.cpp's own `ggml_mul_mat`, timed inside its build, at the same widths, with `-rtr` on and off (`iqk_repack_tensor` is what `-rtr` calls). 3 rounds. Not part of either engine; it links against the ik build. |
| `peak4.c`, `peak6.c`, `peak9.c`, `clock.c`, `roofs.txt` | the machine's roofs: 16 independent accumulators give the real rate (7.1 G instructions/s = two per cycle at 3.55 GHz; 455 GMAC/s for `vpdpbusd zmm`). `peak4.c` with 8 accumulators reads half that, which is the latency of the instruction, not its throughput; `clock.c`'s dependent-add chain reads 1.7 GHz for the same reason and is recorded only to show why it was rejected. |

`ik_kqbench` uses a 2048×4096 synthetic matrix (4.7 MB) so the weights stream from L3, while
`kqbench` keeps one 8-row tile in L1. That difference favours Peregrine; it is stated in the
note.

## Caveats

- **The host was not idle.** The agent's own UI (a `node` process) kept a load average of 1–8
  through several series, and `uptime` is recorded before every series in the driver's
  stderr. Medians of interleaved rounds are the mitigation; the same-day ratio of the two
  engines is the quantity to trust, not absolute tok/s.
- **The 24-thread rows are confounded** with `PEREGRINE_PREFILL_CHUNK`: at more threads than
  physical cores the engine's default chunk becomes 64. Both the 16 and the 64 rows are
  reported.
- **`--tokens 0` for the profile series.** With `--tokens 1` the dump's per-step wall line
  disagrees with the engine's own `prefill_s` by ~3× (a display inconsistency in
  `peregrine bench`, not a measurement error in the buckets: the buckets sum to the engine's
  step time, and with `--tokens 0` the two agree exactly). See the note.
- **No `perf` under WSL2**, so ik's matmul share inside its own prefill is inferred, not
  measured.
- `benchmarks/results/desktop-zen5-2026-11-11-ik/` is the other harness's run of the same
  question (server mode, three engines); this folder is the single-user, kernel-level study.
