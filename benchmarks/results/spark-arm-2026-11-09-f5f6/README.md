# aarch64 NEON tier F5/F6: `vmath`, the value mix and activation quantisation

The third follow-up to the NEON tier (S2.8 F5/F6): the softmax/activation vectors, the attention
value mix and `QuantAct::fill` on NEON. Branch `neon-5`, stacked on `neon-4` (F3, #130) and
`neon-3` (F1, #129), themselves on `neon-2` (#128).

- Date: 2026-11-09.
- Host: DGX Spark, 10 × Cortex-X925 (CPUs 5–9, 15–19) + 10 × Cortex-A725, 121 GB.
- Peregrine `neon-5` at `a44691c` + this step, `release`, `rustc 1.99.1`. Tier
  `aarch64-neon-dotprod`.

## What this step changes

| layer | before (aarch64) | now |
|---|---|---|
| `exp` | libm `expf` in the scalar fallbacks | `exp4`: `exp8`'s constants, order and FMAs, four lanes; **bit-identical to `exp_scalar`**, which is `exp8`'s scalar twin |
| softmax (`exp_shift_sum`, `exp_scaled_sum`) | scalar exp + sequential f64 sum | `exp4` + the same sequential f64 sum (index order kept, so the denominator's summation order does not change) |
| SwiGLU / SiLU / `mul_silu` / `mul_sigmoid` / `swiglu_oai` | scalar libm exp | `exp4`, four lanes; padded tails, so a row split across workers has the same bits |
| attention value mix (`mix`) | scalar loop | `mix_neon`: four dimensions per pass, keys in the scalar order, multiply then add — **bit-identical to the scalar loop** |
| `QuantAct::fill` | scalar absmax + rounded codes | `absmax`/`quantize32` on NEON, rebuilding `round` as `trunc(|y|) + (frac ≥ 0.5)`; **bit-identical to `fill_scalar`** |

`max` is deliberately still scalar: LLVM auto-vectorises the fold to 62 GB/s on this core, twice
what a hand-written four-lane loop measured, and the result is exact either way.

Attention scores (`scores`) stay scalar on aarch64: a NEON dot has a different reduction order, so
it would move the tier's bits. It is the obvious next step if the attention share justifies it.

## Why 50 goldens were re-recorded

The aarch64 tier used libm's `expf` in the scalar `vmath` fallbacks; the new vector paths use the
engine's own `exp`, which is the x86 tier's (issue #64's deterministic transcendental). That is a
legitimate fold change: every softmax and every SwiGLU/SiLU on this tier moves in the last bits, so
all `aarch64-neon-dotprod` goldens are re-recorded from this binary. The **scalar** tier is
unchanged: `PEREGRINE_ISA=scalar` reproduces its stored `aarch64` goldens bit for bit, verified.

## Micro-benchmarks (the reliable step measurement)

One thread, 8192-element arrays, X925, `vmathbench` under `PEREGRINE_ISA=scalar` and native
(`vmathbench.txt`). Scalar here is the same binary with every NEON branch disabled.

| operation | scalar | NEON | speed-up |
|---|---:|---:|---:|
| SwiGLU | 14688.7 ns | 6288.2 ns | 2.34× |
| SiLU | 17135.2 ns | 6189.6 ns | 2.77× |
| `mul_silu` | 14523.0 ns | 6226.3 ns | 2.33× |
| `mul_sigmoid` | 14130.4 ns | 6214.0 ns | 2.27× |
| `exp_shift_sum` (softmax) | 14593.4 ns | 6295.6 ns | 2.32× |
| `exp_scaled_sum` (sampling) | 14976.7 ns | 6390.2 ns | 2.34× |
| `QuantAct::fill` | 29263.5 ns | 12591.5 ns | 2.32× |

## Engine profile

`PEREGRINE_PROF2=full`, Qwen2.5-0.5B, 32 decode steps, X925, one run each back to back. The host
was shared, so the totals move; the phase split is the comparison:

| phase (µs/decode step) | F3 tree | this step |
|---|---:|---:|
| quantise | 172.4 | 75.7 |
| SwiGLU | 355.6 | 180.9 |
| attention | 1356.8 | 1285.0 |
| wall per step | 7949.0 | 6590.0 |

The two vectorised phases halve (2.3× and 2.0×, matching the micro-benchmark); attention gains the
softmax's `exp` and the value mix (~5 %). Both are small next to the matmuls on decode, which is
the bandwidth-bound part.

## Engine, best of 5 (contended)

5 interleaved rounds per arm, 512-token prompt, 128 decode tokens. Two other workers held most of
the machine for long stretches of this run (individual rounds at 25–80 tok/s against 400–600), so
**medians are meaningless** and the table quotes the best round per arm as the least-contended
estimate. `peregrine.csv` has every run.

| model | threads | arm | prefill tok/s | decode tok/s |
|---|---|---:|---:|---:|
| Llama-3.2-1B | 10 (X925) | F3 | 359.7 | 103.9 |
| Llama-3.2-1B | 10 (X925) | **this step** | **406.8** | **104.6** |
| Llama-3.2-1B | 20 | F3 | 104.0* | 8.2* |
| Llama-3.2-1B | 20 | **this step** | **419.5** | **90.3** |
| Qwen2.5-0.5B | 10 (X925) | F3 | 573.0 | 175.9 |
| Qwen2.5-0.5B | 10 (X925) | **this step** | 567.8 | 173.0 |
| Qwen2.5-0.5B | 20 | F3 | 628.6 | 175.0 |
| Qwen2.5-0.5B | 20 | **this step** | 631.5 | 182.4 |

\* the F3 20-thread round was hit by the co-tenant. The Qwen rows are flat within the noise, which
is what the 5 % phase saving predicts on a bandwidth-bound step; the Llama prefill best is +13 %,
where the Q6_K-heavy prefill's softmax/activation share is larger.

## Batch invariance

- `vmath::tests::neon_exp4_is_exp_scalar_bit_for_bit` (20 000 inputs over −130…130, `to_bits()`).
- `vmath::tests::neon_vmath_matches_the_scalar_definition`: SwiGLU/SiLU/`mul_silu`/`mul_sigmoid`/
  `swiglu_oai`/`exp_shift_sum`/`exp_scaled_sum` against a scalar definition built on `exp_scalar`,
  at lengths 1–17 and 1001, **and every split point** of a row (a worker's block range cannot move
  a bit).
- `attn::tests::neon_mix_is_bit_identical_to_the_scalar_mix` at head widths 4–128 (including lane
  remainders) and every `vis`.
- `quant::tests::neon_fill_matches_the_scalar_fill` for n = 128…4864 (all buffers, exact).
- `neon::tests::quantisation_matches_the_scalar_loop`: 2 000 seeded 32-value blocks, including the
  half-way rule and the clamp, against `round().clamp(±127)`.
- The whole golden and synthetic suites pass with `PEREGRINE_GOLDEN=require`; the scalar tier's
  stored goldens still pass unchanged.

## Files

- `vmathbench.txt`: the micro-benchmark, scalar and native.
- `profile.txt`: the two phase profiles.
- `peregrine.csv`: every engine run.
