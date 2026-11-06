# GitHub setup

How the repository is configured, so it can be recreated.

## Branch protection on `main`

- Pull requests required; squash merge only; linear history; branches deleted on merge.
- Required check: **`ci-ok`** (it aggregates every CI job, so the rule does not change when jobs do),
  and the branch must be up to date with `main`.
- Conversations must be resolved. No approval count is required while every PR is pushed from the
  maintainer's account (GitHub does not let an author approve their own PR); review follows
  `docs/dev/implementation.md`. Raise it to 1 when a second maintainer joins.
- No force pushes, no deletion.

## Security

- Private vulnerability reporting: on.
- Dependabot alerts and security updates: on. Version updates come from `.github/dependabot.yml`.
- Secret scanning and push protection: on.
- Actions: pinned to commit SHAs; default `GITHUB_TOKEN` permission read-only.

## Labels

| Label | Use |
|---|---|
| `bug`, `enhancement`, `triage` | issue forms set these |
| `perf` | speed work or reports |
| `models`, `formats`, `isa` | support for architectures, weight formats, CPU tiers |
| `accuracy` | anything that may change tokens or KL |
| `structure` | refactors from roadmap Stage 1 |
| `good first issue`, `help wanted` | for new contributors |
| `breaking`, `skip-changelog` | release notes |
| `stage-0` ... `stage-3` | roadmap stage |

## Project board

One board, columns *Backlog / Next / In progress / Review / Done*, one issue per roadmap ID
(`S0.1`, `S1.1`, ...), milestones per release (`v0.2` = Stage 0-1, `v0.3` = Stage 2).

## Discussions

On, with categories *Q&A*, *Ideas*, *Show and tell* and *Benchmarks*.
