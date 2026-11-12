# S7: macOS and Windows

- Status: design, decided 2026-11-12. S7.1 (macOS on Apple Silicon) built 2026-11-12; the rest not built.
- Owner request: "add support to windows, mac".

## 0. Where things stand

Peregrine releases Linux binaries only (x86-64 and aarch64). The code is closer to the other two
systems than the releases say:

- **macOS on Apple Silicon** builds and passes the test suite in CI on every PR (`test
  (macos-aarch64)`), with the NEON kernels and its own QoS call for the pool. It has no release
  archive, the installer refuses it, `peregrine upgrade` has no asset for it, the real and
  synthetic goldens are not recorded for it (macOS keys its own golden directories and skips), and
  nothing has been measured on it.
- **Windows** has never been built. The whole workspace type-checks for `x86_64-pc-windows-gnu`
  today (the dependencies are portable: `memmap2`, `tokio`, `axum`; the Unix-only pieces already
  sit behind `cfg`), which says the port is mostly behaviour, tests and packaging, not code that
  cannot compile.

## 1. What "supported" means

A platform is supported when all of these hold, the same bar Linux is held to:

1. CI builds it and runs the whole test suite on it on every PR, in release mode.
2. The release workflow builds a binary for it natively, tests the tagged commit on it, and
   uploads an archive with a checksum.
3. A one-line installer puts the binary on the machine and `peregrine upgrade` replaces it.
4. `run`, `serve`, `embed`, `rerank`, `decide`, `bench`, `tier`, `inspect`, `hf:` downloads and
   the packed-weight cache work. The server's lifetime commands (`serve --background`, `status`,
   `stop`) work, or say plainly what the platform offers instead.
5. Results are held by goldens recorded for that platform's tiers, and the engine's invariants
   (batch, chunk, thread and speculation independence) are tested on it like anywhere else.
6. The quickstart has the platform's commands, and the support table names it.

Speed is a separate claim. A platform can be supported before it has a published comparison;
`docs/benchmark.md` only gains rows that were measured on real hardware of that platform.

## 2. Targets

| Target | Step | Notes |
|---|---|---|
| macOS, Apple Silicon (`aarch64-apple-darwin`) | S7.1 | the Macs that matter for local inference; NEON, dotprod, and `i8mm` from the M2 on |
| Windows, x86-64 (`x86_64-pc-windows-msvc`) | S7.2–S7.4 | AVX2, AVX-VNNI and AVX-512 chosen at run time, as on Linux |
| macOS, Intel (`x86_64-apple-darwin`) | not planned | built from source if asked for; no archive, no CI |
| Windows on ARM (`aarch64-pc-windows-msvc`) | later | after x86-64 Windows; needs its CI runner |

## 3. Decisions that hold for both

**Numbers do not depend on the C library.** Coding rule 5 already says the forward path does not
call libm, and v0.2.0's release gate showed why (a `tanhf` that glibc changed). Off Linux there is
no history of goldens to protect, so on macOS and Windows every forward-path function is the
engine's own (`vmath`), on x86-64 as it already is on aarch64. A Windows x86-64 golden is then
the arithmetic's, not the Universal C Runtime's, and stays put across Windows versions.

**Goldens per platform.** The golden directory of a tier carries the system when it is not
Linux (`…-macos` today; `…-windows` is added). They are recorded on the platform itself: a
manually started CI job records the synthetic goldens on the runner and uploads them as an
artifact, which is then committed. Real-model goldens need model files and a machine of that
platform; until one exists they are skipped there, and that is stated in `docs/dev/testing.md`.

**One binary per (system, architecture)**, portable across that architecture's CPUs, with the
kernel tier chosen at run time; no `target-cpu`.

**No new dependencies** (ADR 0002). What the standard library lacks is written against the
system's own API in one small module per platform, with the same review as any `unsafe`.

## 4. macOS (S7.1)

- **Release**: a `macos-aarch64` job in `release.yml` on the Apple Silicon runner: the full suite
  in release mode, the tier check, `peregrine-vX.Y.Z-macos-aarch64.tar.gz`, one `SHA256SUMS` for
  all archives. A `workflow_dispatch` dry run builds and uploads the archives as artifacts without
  creating a release, so the workflow can be tested before a tag.
- **Installer**: `install.sh` accepts Darwin on arm64 (it says so, and still refuses Intel Macs
  with the build-from-source line). A binary fetched with `curl` carries no quarantine flag and
  the linker's ad-hoc signature is enough to run it; a binary downloaded with a browser is
  blocked by Gatekeeper until the user clears it, and the quickstart says how. Notarisation is
  not done in this step.
- **`peregrine upgrade`** maps the platform to its asset name.
- **Lifetime commands** already work through the Unix paths (`kill -0` for liveness). The
  `service` command writes a systemd unit; on macOS it says so and points at `serve --background`
  (a launchd agent is a later item).
- **Threads**: the default decode width is the performance cores (`hw.perflevel0.physicalcpu`),
  not every core: the efficiency cores slow a synchronous pool down. Checked on the CI runner,
  measured properly when a Mac is available.
- **Goldens**: the synthetic set recorded on the runner for the macOS tiers it reports.
- **Not in this step**: a published comparison on Apple Silicon. That needs a Mac in the
  benchmark fleet; the CI runners are three shared cores.

## 5. Windows (S7.2–S7.4)

**S7.2: it builds and passes.** A `test (windows-x86_64)` job on the Windows runner
(`x86_64-pc-windows-msvc`, release mode), required like the others. Expected work, to be
confirmed by the first run rather than assumed:

- line endings: byte-exact fixtures (`*.jsonl`, `*.txt`, `*.golden`, `*.jinja`) must check out
  with LF (`.gitattributes`);
- files that are mapped cannot be renamed over or deleted on Windows: the packed-weight cache's
  publish step, the `hf:` download's final rename and the tests' clean-up are the places that
  assume they can;
- tests that are Unix by construction (`SIGTERM`, `#![cfg(unix)]`) get a Windows counterpart
  where the behaviour exists and an explicit skip where it does not;
- paths: the cache and state directories follow the platform (`%LOCALAPPDATA%\peregrine`), and
  nothing prints or parses a path with a `/` assumption;
- the pool's default width: physical cores through the system's processor information call, not
  the logical count.

**S7.3: lifetime, installer, release.**

- `serve` stops cleanly on Ctrl-C (it does today, through `tokio`) and on a stop request.
  Windows has no `SIGTERM`: `peregrine stop` asks the running server to drain through a
  per-user stop file the server watches, and falls back to ending the process after the drain
  timeout. `serve --background` starts a detached process with no console and writes the same
  state file; `status` reads it.
- `peregrine upgrade` cannot overwrite a running `.exe`: it renames the old binary aside, puts
  the new one in place, and removes the old one on the next start. The hand-over of a listening
  socket (`SIGHUP`, `SO_REUSEPORT`) is Unix-only; on Windows an upgrade restarts the server and
  the command says so.
- `scripts/install.ps1`: `irm https://peregrine.rantai.dev/install.ps1 | iex` downloads the zip,
  checks its SHA-256 against `SHA256SUMS`, installs to `%LOCALAPPDATA%\Programs\peregrine` and
  tells the user how to put it on `PATH`. The download host gets the matching redirect rule.
- `release.yml`: a `windows-x86_64` job: the suite in release mode, the run-time tier check,
  `peregrine-vX.Y.Z-windows-x86_64.zip`.

**S7.4: goldens, docs, a first measurement.** The synthetic goldens for the Windows tiers
(scalar, AVX2; AVX-512 under the emulator if it runs on the runner). Quickstart, README install
section and the support table for both systems. Native Windows numbers on the AVX-512 desktop
(it is a Windows machine; the Linux numbers published so far come from WSL2 on it) against
llama.cpp's Windows build, for decode, prompt and ten users on the three gate models.

## 6. Steps

| Step | What | Depends on |
|---|---|---|
| S7.1 | macOS on Apple Silicon: release job with a dry run, installer, `upgrade`, performance-core default, synthetic goldens, quickstart | — |
| S7.2 | Windows x86-64 builds and passes the suite in CI | — |
| S7.3 | Windows lifetime commands, `upgrade`, `install.ps1`, release job | S7.2 |
| S7.4 | Windows goldens, docs for both systems, native Windows measurement on the desktop | S7.3 |
| S7.5 | A release that ships all four archives | S7.1, S7.4 |

## 7. Open points

- **A Mac for measurement.** Without one, macOS is supported but unmeasured.
- **Signing.** Unsigned binaries run when fetched by the installers. A Developer ID signature and
  notarisation for macOS, and Authenticode for Windows (SmartScreen warns on a downloaded
  unsigned `.exe`), need certificates the project does not have.
- **Windows Defender and mapped model files.** First-run scanning of a multi-gigabyte mapped file
  can dominate load time; measured in S7.4, with the documented exclusion if it does.

## 8. S7.1 as built

macOS on Apple Silicon is a released platform in this step; the bar in §1 is met for it:

- **CI.** `test (macos-aarch64)` in `ci.yml` runs the workspace suite in release mode on every pull
  request as before, and after it builds the release binary, prints `peregrine --version` and
  `peregrine tier`, and generates from a synthetic model written by `peregrine-bench`'s
  `write_synthetic` example. `ci.yml`'s `lint` job runs `scripts/install-test.sh`, the installer's
  platform matrix, without a network or a Rust build.
- **Release.** `release.yml`'s matrix has `macos-15` / `aarch64-apple-darwin` / `macos-aarch64`, and
  `scripts/release/build.sh` builds the archive there: `MACOSX_DEPLOYMENT_TARGET=11.0`,
  `strip -x`, then `codesign --force --sign -` (the linker's ad-hoc signature is invalidated by
  `strip`) and `codesign --verify --strict`. The archive is
  `peregrine-vX.Y.Z-macos-aarch64.tar.gz`, and `SHA256SUMS` is written with `shasum -a 256` because
  macOS has no `sha256sum`. A manual start is a dry run (`workflow_dispatch` with `dry_run=true`):
  the build jobs upload their archives as workflow artefacts and the publish job is guarded on a
  tag push, so it cannot run. The binary is ad-hoc signed, not notarised (§7). The dry run on
  `macos-15` produced `peregrine-v0.2.0-macos-aarch64.tar.gz` (2,717,446 bytes, SHA-256
  `6fca9591…e17b0`), and the installer installed that artefact on the runner and ran
  `peregrine --version` and `peregrine tier` (`aarch64-neon-dotprod`).
- **Installer.** `scripts/install.sh` accepts Darwin on arm64 (asset `macos-aarch64`), refuses
  Darwin on x86-64 with the build-from-source line, and uses `sha256sum` where it exists and
  `shasum -a 256` otherwise; the Linux paths are unchanged. `scripts/install-test.sh` exercises all
  four platform cases against stub archives, and the macOS workflow installs the real archive and
  runs it.
- **`peregrine upgrade`.** `release_platform(os, arch)` maps the four cases (Linux x86-64/aarch64,
  macOS aarch64; Intel macOS gets the build-from-source answer) and the checksum uses `shasum` when
  `sha256sum` is absent. The mapping is unit-tested by argument.
- **Threads.** `pool::physical_cores()` on macOS reads `hw.perflevel0.physicalcpu` through
  `sysctlbyname` (one hand-written `extern "C"` declaration, no new dependency), falls back to
  `hw.physicalcpu`, then to `available_parallelism`, and is clamped to the CPUs the process may run
  on. The default decode width, the narrow decode pool, the prefill-chunk default, calibration and
  `tune` all use it; an explicit width still wins. A macOS-only unit test prints both counts and
  checks the performance-core count is between 1 and `available_parallelism`. On the `macos-15`
  runner the two counts are both 3 (`hw.perflevel0.physicalcpu = hw.physicalcpu = hw.ncpu = 3`): a
  three-core VM with no efficiency cores, so the check is of the API, not of a mixed-chip default.
- **Goldens.** The synthetic set for the runner's tier is recorded on `macos-15` and committed
  (`tests/golden/synthetic/aarch64-neon-dotprod-macos/`, 58 files, and
  `tests/fixtures/session/aarch64-neon-dotprod-macos.txt`). Before recording, the five golden
  suites printed 52 "no golden for tier" skips on the runner; after committing the set and running
  again without `PEREGRINE_GOLDEN`, 0. `.github/workflows/record-goldens.yml` is the permanent,
  manually started recorder: it runs the suite with `PEREGRINE_GOLDEN=record` and
  `PEREGRINE_SESSION_FIXTURE=record`, runs it again without the switch to prove the values
  reproduce, counts the skip lines before and after, and uploads the changed files. Real-model
  goldens still need a Mac with model files; until one exists they are skipped there, as §3 says.
  (The calibration test's golden name was `llama-q8_0`, which on macOS's case-insensitive
  filesystem aliased the dense test's `llama-Q8_0` golden for a different seed; it is
  `llama-calibration-q8_0` now, so the collision cannot come back.)
- **Docs.** `docs/quickstart.md` has the macOS section (install line, quarantine, cache location,
  thread choice), `README.md` names the platform, and `docs/dev/releasing.md` has the archive and the
  dry-run command.
- **Not done.** No notarisation (needs a Developer ID certificate; §7), no Intel Mac archive, and no
  measurement on real Mac hardware: the CI runner's three shared cores are not a benchmark host, so
  `docs/benchmark.md` has no macOS row.
