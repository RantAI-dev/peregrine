## What and why

<!-- One paragraph. Link the issue or roadmap item (docs/roadmap.md). -->

## How it was checked

- [ ] `cargo xclippy` and `cargo xtest` pass
- [ ] Output is unchanged: greedy tokens and KL against the exact path (or the change is lossy, off by default, and has an accuracy report)
- [ ] Performance: A/B numbers below if a hot path changed (docs/dev/benchmarking.md)
- [ ] Docs, `CHANGELOG.md` and the support table updated if behaviour changed

## Numbers

<!-- before / after, machine, model, command. Delete if no hot path changed. -->
