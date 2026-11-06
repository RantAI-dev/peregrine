# Governance

Peregrine is maintained by RantAI. Decisions are made in the open, on GitHub.

## Roles

- **Maintainers** review and merge, own releases and set the roadmap. Listed below.
- **Reviewers** for an area are listed in `.github/CODEOWNERS`; a PR touching kernels or `unsafe` code needs
  one of them.
- **Contributors**: everyone who sends an issue, review or PR.

## Maintainers

| Name | GitHub | Area |
|---|---|---|
| Evan Pradipta Hardinatha | @kleopasevan | lead maintainer; engine, kernels, releases |

## How decisions are made

1. Most changes: a PR, a review from a maintainer or the area's code owner, green CI.
2. Changes that are expensive to reverse (a dependency, a public API, a file format, a default that affects
   accuracy, the project's scope): an issue or an ADR in `docs/adr/` first, open for comments for at least
   3 working days.
3. If maintainers disagree, the lead maintainer decides and records the reasoning in the ADR.

## Becoming a maintainer

Sustained, high-quality contributions and reviews over several months, then an invitation from the existing
maintainers.
