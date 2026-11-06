# Releasing

Until 1.0: minor versions (`0.x.0`) may break the API, CLI or options; patch versions (`0.x.y`) do not.

1. All crates share one version (`[workspace.package] version`).
2. Before a release: CI green on `main`; the real-model tests pass on H1 (AVX-512) and H2 (AVX2); the
   accuracy gate holds; `CHANGELOG.md` *Unreleased* moved under the new version.
3. Tag `vX.Y.Z` on `main`; GitHub generates release notes from PR labels (`.github/release.yml`).
4. Binaries: built by a release workflow (to be added with the first release) for x86-64 Linux, aarch64
   Linux and macOS, with checksums.
5. Crates are not published to crates.io until the API settles (`publish = false` today).
