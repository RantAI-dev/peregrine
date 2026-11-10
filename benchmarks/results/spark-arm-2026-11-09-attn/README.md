# NEON attention scores (S2.8 F5): measurements

The attention half of S2.8's F5, on the machine the tier is written for. This PR adds the
aarch64 score kernel (`vcvt_f32_f16` + NEON FMA, the canonical per-row reduction) and lets the
prefill tile read F16 K/V in registers; the value mix was already NEON (F5, `neon-5`).

- Date: 2026-11-09.
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64: 10 × Cortex-X925 (3.9 GHz, CPUs 5–9 and
  15–19) and 10 × Cortex-A725 (2.8 GHz, CPUs 0–4 and 10–14), 121 GB unified LPDDR5X, one NUMA
  node. `asimd`, `asimddp`, `i8mm`, `sve`, `sve2`, `bf16`.
- Peregrine: `neon-6` on `neon-5` (`release`), `rustc 1.99.1 (58cf8f3c3 2026-10-13)`. Tier
  `aarch64-neon-dotprod`. The A/B arm is the `neon-5` binary of the same tree (identical
  sources except this PR).
- llama.cpp: v0.9.0-dev (build 1, commit `0c6dca2`), GNU 13.3.0, Linux aarch64. CPU only:
  `CUDA_VISIBLE_DEVICES=""`, `-ngl 0`, `-fa 0` (CPU flash attention measured 4–30× slower
  here and is not the comparison), `-p 512 -n 128 -t <threads> -r 1`, five interleaved rounds.
- Models, both in `/var/tmp/pg/models`, SHA-256 checked against `docs/dev/testing.md`:
  `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (807694464 B) and
  `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (397808192 B).
- Settings: `peregrine bench --prompt-tokens 512 --tokens 128`, release profile. 10-thread runs
  are pinned to the X925s (`taskset -c 5-9,15-19`); 20-thread runs use all cores. Five rounds per
  arm, the arms interleaved round by round.

## The host is shared

Another worker ran a 16-thread `peregrine serve` (affinity 0–19) for most of the window, so load
averaged 11–31 on 20 CPUs. The tables use the median of the five rounds; rounds that a co-tenant
visibly hit are shown in the CSV and marked below. The attention A/B (same binary tree, two
commits, same process settings) is the number this PR stands behind: its per-round ratios
were stable at 1–2 % on the clean rounds, and the one dropped round is the co-tenant, not the
code. Absolute throughput is indicative.

## The canonical aarch64 score order

A key's dot now runs in two `f32x4` accumulators — dimensions `0..4` and `4..8` modulo 8 — over
ascending eight-wide blocks of the head, reduced as
`((d0+d1)+(d2+d3)) + ((d4+d5)+(d6+d7))`. That is the AVX2 tier's `hsum_grouped_ps` tree with
four lanes per half: the same association of the same products with the same fused multiply-add.
It is defined once (`attn.rs`, `hsum4_neon` / `scores_neon`) and every aarch64 path reaches it:
single-row decode (`run`/`run_f16`), the prefill tile (`run_range`/`score_tile`) and the
multi-user mapped path (`run_row`). A row's scores therefore do not depend on which path ran,
which is what the existing `prefill_tiles_are_bit_identical_to_a_solo_decode`,
`a_f16_rows_bits_are_identical_alone_in_a_chunk_and_at_any_pool_width` and
`a_rows_attention_does_not_depend_on_the_batch_width` tests re-check on this tier.

The F16 tile is converted in the kernel (`vcvt_f32_f16` is exact), so the f16 and widened-f32
kernels are bit-identical (`neon_f16_scores_equal_the_widened_scores`); the pairing kernels share
the loaded key but keep each row's own accumulators
(`the_paired_f16_score_is_each_rows_own_score`).

### Why the `aarch64` tier's goldens were re-recorded

The scalar reference sums a key's products left to right, so the new tree changes the last bits of
attention. Every `aarch64-neon` and `aarch64-neon-dotprod` golden that uses attention therefore
changed at the `logits_fnv` line and was re-recorded on this host
(`PEREGRINE_GOLDEN=record`, after deleting the old directories). What did **not** change:

- the `aarch64` (scalar) goldens, which still pass with `PEREGRINE_GOLDEN=require` — the scalar
  reference is untouched;
- every argmax/greedy token on the synthetic models, and the prompt argmax on both real models;
- `llama-Q4_K` synthetic, which is still bit-identical to `x86_64-avx2`: on a model where every
  other kernel already agreed, the new attention bits are the AVX2 tier's bits too.

The two real-model greedy streams diverge after a few tokens (Llama prompt 1 at token 5, Qwen
prompt 0 at token 16, Qwen prompt 2 at the first token) — the usual consequence of moving the
last bits of a softmax input where two candidate logits are close. That is the re-record this
order requires, not a regression: the scalar arm is unchanged and the AVX2-equal model stayed
equal.

## Results

### The tier against scalar and llama.cpp (median of 5 rounds)

| model | threads | arm | prefill tok/s | decode tok/s | prefill vs scalar | decode vs scalar |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 10 (X925) | scalar | 95.0 | 57.9 | — | — |
| | 10 | **neon** | **462.3** | **120.0** | **4.87×** | **2.07×** |
| | 10 | llama.cpp | 264.9 | 106.2 | — | — |
| | 20 | scalar | 127.7 | 68.3 | — | — |
| | 20 | **neon** | **500.2** | **113.6** | **3.92×** | **1.66×** |
| | 20 | llama.cpp | 325.9 | 97.5 | — | — |
| Qwen2.5-0.5B Q4_K_M | 10 (X925) | scalar | 218.4 | 119.2 | — | — |
| | 10 | **neon** | **641.5** | **194.7** | **2.94×** | **1.63×** |
| | 10 | llama.cpp | 519.5 | 193.6 | — | — |
| | 20 | scalar | 253.4 | 120.3 | — | — |
| | 20 | **neon** | **802.2** | **213.8** | **3.17×** | **1.78×** |
| | 20 | llama.cpp | 638.5 | 160.4 | — | — |

Peregrine ÷ llama.cpp: Llama 1.74× / 1.53× prefill and 1.13× / 1.17× decode at 10 / 20
threads; Qwen 1.23× / 1.26× and 1.01× / 1.33×. (The tier's earlier README compared an older
build to llama.cpp on a different day; these rows are the same-day, interleaved comparison.)

### What this PR alone changes (`neon-5` vs `neon-6`, same tree)

| model | threads | prefill `neon-5` → `neon-6` | decode `neon-5` → `neon-6` |
|---|---|---:|---:|
| Llama-3.2-1B | 10 (X925) | 404.3 → 463.2 (**+14.6 %**) | 104.3 → 120.6 (**+15.6 %**) |
| Llama-3.2-1B | 20 | 416.4 → 498.6 (**+19.7 %**) | 89.9 → 111.5 (**+24.1 %**) |
| Qwen2.5-0.5B | 10 (X925) | 557.5 → 638.5 (**+14.5 %**, rounds 1–4) | 172.7 → 194.5 (**+12.6 %**) |
| Qwen2.5-0.5B | 20 | 597.7 → 745.7 (**+24.7 %**, rounds 1–4) | 180.3 → 205.8 (**+13.8 %**, rounds 2–3) |

The last Qwen rows exclude rounds the co-tenant hit (round 5 of both, the decode of round 4 of
`neon-6`); the raw per-round values are in `ab-qwen-05b.csv`. Decode gains more than the score
kernel alone would suggest because a decode step re-reads every key each token, and the cache is
now read as f16 instead of being widened element-at-a-time.

## Files

| file | contents |
|---|---|
| `llama-1b-t10.csv`, `llama-1b-t20.csv` | scalar / neon / llama.cpp, 5 interleaved rounds |
| `qwen-05b-t10.csv`, `qwen-05b-t20.csv` | the same for the small model |
| `ab-llama-1b.csv`, `ab-qwen-05b.csv` | `neon-5` (before) against `neon-6` (this PR), 10 and 20 threads |
