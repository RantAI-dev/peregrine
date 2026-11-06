# How the roadmap is implemented

Work on `docs/roadmap.md` is split across machines and agents. This page is the operating manual: who
does what, where, and what a change must pass before it merges.

## Machines

| Host | Hardware | Role |
|---|---|---|
| **H1 desktop** | Ryzen AI 9 HX 370, WSL2, ~5.7 GiB usable, AVX-512 VNNI/GFNI/VBMI, all model files | the checking machine: real-model tests and goldens on the AVX-512 tier, A/B benchmarks. One heavy job at a time. |
| **H2 bluefin** | Ryzen 3 PRO 4350G, bare metal, AVX2 only, 13 GB; Qwen3-1.7B, SmolLM2-360M, Bonsai 2 2B and 4B (MTP) | AVX2 tier goldens, quiet benchmarks, a worker host |
| **H0 coordinator** | Xeon E5-2680 v4 VM, AVX2 only, no models | orchestration, review, merges, fast checks, a worker host. Never used for speed numbers. |

## Agents

| Agent | Gets |
|---|---|
| **Orchestrator** (Claude Opus) | interfaces others build on (`EngineOptions`, kernel table, `Architecture`, drafter interface), task briefs, final review, merges, conflict resolution |
| **Gemini 3.8 Flash High** (Antigravity) | larger changes with a clear spec: a kernel family into the table, one architecture onto shared blocks, CLI and features |
| **DeepSeek V4.1 Flash** (OpenCode) | narrow, checkable tasks: fixtures, scalar references, CI jobs, small kernels with a spec, docs, mechanical moves |

Workers review each other's PRs before the orchestrator's review.

## Flow of one task

1. **Brief.** The orchestrator writes a brief (template below) and opens an issue with the roadmap ID.
2. **Branch.** One branch per task, named `<id>-<slug>` (for example `s0.2-synthetic-models`), from
   current `main`. Each brief lists the files it may touch; parallel tasks never share a file.
3. **Work.** The worker implements, runs the checks, pushes the branch and opens a PR that fills in the
   PR template, labelled `worker` and the stage.
4. **Gates.** In order, and each must pass before the next:
   1. CI (`ci-ok`): fmt, clippy, tests on x86-64 / aarch64 Linux and macOS, MSRV, supply chain.
   2. Goldens on H2 (AVX2) and H1 (AVX-512): `cargo test --release --test golden` identical.
   3. If a hot path changed: an interleaved A/B on H1 and H2 (`docs/dev/benchmarking.md`), no loss.
5. **Review.** The other worker reviews, then the orchestrator. Requested changes go back to the author.
6. **Merge.** Squash-merge in dependency order by the orchestrator. Goldens are re-recorded only in a PR
   whose purpose is to change output, and that PR says why.

## Brief template

```
ID / title:
Goal: (one paragraph: what and why, link to roadmap/ADR)
Files you may change: ...
Files you must not change: ...
Design constraints: (from ADR 0005, coding-rules.md, structure-rules.md that apply)
Done when:
  - cargo xclippy and cargo xtest pass
  - <task-specific tests>
  - goldens identical (cargo test --release --test golden) on this host
Deliver: branch <id>-<slug> pushed, PR opened with the template filled in; reply with the PR URL,
the head SHA and anything you were unsure about.
Do not: change goldens, add dependencies to peregrine-core, touch files outside the list, force-push main.
```

## Waves

| Wave | Contents | Parallel |
|---|---|---|
| 0 | hub repo, branch protection, models on H2, **golden outputs**, this page | done by the orchestrator |
| 1 | Stage 0: S0.1 `EngineOptions` (orchestrator), S0.2 synthetic models, S0.6 sampling penalties; then S0.3 forced tier + SDE, S0.4/S0.5 measurements | 3 at a time |
| 2 | S1.1: the kernel table and one format through it (orchestrator), then one task per format family | 3-4 at a time |
| 3 | S1.2 interface and blocks (orchestrator), then dense and hybrid families in parallel; S1.3-S1.7 | 2-3 at a time |

## Goldens

`crates/peregrine-core/tests/golden.rs` runs three fixed prompts through every model file present and
compares, per kernel tier, a hash of every prefill logit's bits, the argmax at every position, and 24
greedy tokens; it also checks that results do not depend on thread count.

| Tier | Recorded on | Models |
|---|---|---|
| `x86_64-avx2` | H2 | SmolLM2-360M Q4_K_M, Qwen3-1.7B Q4_K_M, Bonsai 2 2B PQ2_0, Bonsai 2 4B PQ2_0+MTP |
| `x86_64-avx2-avx512-vnni-gfni-vbmi` | H1 | to record: the same, plus Bonsai 2 9B and 27B |
