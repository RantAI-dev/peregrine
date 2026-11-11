# Contributing to Peregrine

Thanks for helping. This page is the short version; the rules behind it are in `docs/dev/`.

## Build and check

You need Rust (the pinned toolchain in `rust-toolchain.toml` installs itself through rustup).

```sh
cargo build --release          # engine + server
cargo xtest                    # all tests, release mode
cargo xclippy                  # lints, warnings are errors
cargo fmt --all
cargo deny check               # licenses, advisories, sources (install: cargo install cargo-deny)
cargo quick                    # fast-rebuild release-speed build for daily work
```

Model-dependent tests skip without model files; see `docs/dev/testing.md` for where they look.

## Hygiene

`scripts/hygiene.sh` fails on a tracked credential file, on a credential-shaped string in a tracked
text file, and (when `HYGIENE_BANNED_PATTERN` is set) on a banned name. Run it before a PR; CI runs the
same script on the tree and on the PR's commit range, and `scripts/hygiene-selftest.sh` tests it.
A deliberate false positive can carry a `hygiene:allow` comment on its line. Maintainers export
`HYGIENE_BANNED_PATTERN` from their own notes when running it locally; the pattern is never written
into the repository.

## Before you open a PR

- **Small and focused.** One change per PR. Refactors that move code do not also change behaviour.
- **Same tokens.** Greedy output and the KL check must not change unless the PR is about that and says so
  (`ARCHITECTURE.md`, invariants).
- **Numbers for hot paths.** If you touched a kernel, the pool, attention, the cache or the scheduler,
  include an interleaved A/B (`docs/dev/benchmarking.md`).
- **Follow the rules** in `docs/dev/coding-rules.md` and `docs/dev/structure-rules.md`, especially on
  `unsafe` and on where new code goes.
- **Update docs** (`CHANGELOG.md` under *Unreleased*, the support table, `ARCHITECTURE.md`) when behaviour
  changes.

## Commits and PR titles

Conventional prefixes, used for the changelog and release notes:
`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `deps:`, `chore:`; add `!` for breaking
changes (`feat!:`). Scope is optional: `perf(tern): ...`. PRs are squash-merged, so the PR title is the commit.

## Bigger changes

Open an issue first for a new architecture, format, CPU tier, dependency or public API change. Decisions
that are expensive to reverse are written up as an ADR (`docs/adr/`).

## Reporting

- Bugs, model requests and performance reports: use the issue forms.
- Security problems: privately, see `SECURITY.md`.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0, as stated in
section 5 of the license. No separate CLA is required.
