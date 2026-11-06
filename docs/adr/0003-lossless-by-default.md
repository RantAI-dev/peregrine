# 0003: Lossless by default

- Status: Accepted
- Date: 2026-11-06

## Context

Speed claims are often won with quality losses users cannot see (aggressive cache quantisation, lossy
drafting, kernels that change rounding). The engine's value depends on users trusting that a faster
setting gives the same answers.

## Decision

- Every default path produces the same tokens as the exact reference path for a fixed seed, independent
  of batch size, other users, speculation and the CPU kernel tier; and stays within the KL budget in
  `docs/targets.md`.
- A lossy feature (for example a q8 KV cache or LittleBit weights) is off by default, behind an explicit
  option, and ships with an accuracy report on the gate models.

## Consequences

Some speedups other engines enable by default stay opt-in here, and published comparisons say which
settings each engine used.
