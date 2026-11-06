# `quick` vs `release` build profile (roadmap S0.4)

- Date: 2026-11-06. Commit: `main` at the time (after S0.6).
- Rebuild times: H0 (Xeon E5-2680 v4 VM, 8 vCPU), `cargo build --profile <p> -p peregrine-server`, sccache off,
  after changing one constant in `serve.rs` (three edits each).
- Decode speed: H2 (Ryzen 3 PRO 4350G, bare metal, AVX2), Bonsai 2 2B PQ2_0, greedy, 160 tokens, one warm-up
  request per server start, interleaved release / quick, three rounds.

| | `release` | `quick` |
|---|---|---|
| Cold build | 79.0 s | 85.8 s |
| Rebuild after a one-line code edit | 29.6 / 29.3 / 29.6 s | 2.6 / 2.7 / 2.6 s |
| Decode, tok/s (rounds 1-3) | 46.49 / 46.32 / 46.18 | 46.27 / 46.04 / 45.92 |

**Result.** `quick` rebuilds about 11× faster and decodes about 0.5% slower (the gap is consistent across
rounds). Use it for daily work; publish numbers from `release` only.
