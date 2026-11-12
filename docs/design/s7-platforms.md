# S7: macOS and Windows

- Status: design, decided 2026-11-12. Not built.
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
