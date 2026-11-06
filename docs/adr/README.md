# Architecture decision records

Short records of decisions that are expensive to reverse: what was decided, why, and what was given up.
Format: [`0000-template.md`](0000-template.md). Numbered in order; a superseded record stays, marked as such.

| # | Decision | Status |
|---|---|---|
| [0001](0001-build-with-cargo.md) | Build with cargo; blaze rejected for now | Accepted |
| [0002](0002-dependency-policy.md) | Own the hot path; audited crates for the cold path | Accepted |
| [0003](0003-lossless-by-default.md) | Lossless by default; lossy only opt-in with an accuracy report | Accepted |
| [0004](0004-name.md) | The name Peregrine | Accepted |
| [0005](0005-modular-structure.md) | Modular structure by layer, features for optional parts, runtime CPU dispatch | Accepted (planning) |
| [0006](0006-architecture-interface.md) | The `Architecture` interface and shared layer blocks (S1.2) | Accepted |
| [0007](0007-one-runtime.md) | One runtime (`Engine`) and one drafter interface (S1.4) | Accepted |
