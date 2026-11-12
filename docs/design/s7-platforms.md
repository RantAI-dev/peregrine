# S7: macOS and Windows

- Status: design, decided 2026-11-12. S7.2 (Windows builds and passes the suite) and S7.3 (Windows
  lifetime, installer and release archive) built; the rest is not built.
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

**S7.2 as built (2026-11-12).** What the first Windows run showed, and what it took:

- **Fixtures.** `* text=auto eol=lf` was already in `.gitattributes`, so the first run's
  fixture-comparing tests passed as they were. `*.jsonl`, `*.golden`, `*.jinja`, `*.txt` and
  `*.tsv` now carry an explicit `text eol=lf` as well, and `*.f32`, `*.gguf`, `*.bin` and
  `*.slb` stay binary: a checkout with `core.autocrlf=true` cannot re-encode them.
- **Mapped files.** Two write paths assumed a file can be rewritten or renamed over while it is
  mapped, which Windows refuses (`ERROR_USER_MAPPED_FILE`, `ERROR_ACCESS_DENIED`):
  the synthetic-model writers in `crates/peregrine-models/tests/support/models.rs` now give
  every write a fresh name (`unique_path`), because a test that writes two models of one family
  and format in one directory kept the first mapped; the pack cache's publish
  (`crates/peregrine-core/src/packcache.rs`) keeps a file already published for the same key
  instead of replacing it — the bytes a key names are the same bytes — and drops its temporary
  copy; the `hf:` download drops the header mapping before the final `.part` rename; and the
  pack-cache test that touches a model's mtime opens it for writing and drops the model first.
  `Gguf::read_data` on Windows now uses `seek_read` (positioned `ReadFile`) rather than copying
  out of the mapping, so the mapping's pages for the repacked matrices are not faulted in.
- **Numbers off the C runtime** (decision in §3). The `det_*` forward-path helpers in
  `crates/peregrine-kernels/src/vmath.rs` use the engine's own functions on aarch64 **and on
  every non-Linux system**; only Linux x86-64 keeps libm, so only its goldens stay put. macOS is
  aarch64, so this changes nothing there. The first run confirmed the need: several synthetic
  models and the session fixture differed from the Linux goldens in `logits_fnv` alone (the
  argmax and greedy tokens were identical), which is exactly the UCRT's rounding.
- **Goldens.** Every place that keyed a golden directory `-macos` now also keys `-windows`
  (11 sites in six test files), so Windows compares against its own directories and skips until
  S7.4 records them. The Windows job never sets `PEREGRINE_GOLDEN=require`.
- **Paths.** The model cache, the pack cache and the server state directory default to
  `%LOCALAPPDATA%\peregrine` (and `…\peregrine\packs`) on Windows, after the explicit
  `PEREGRINE_*`/XDG overrides.
- **The pool's width.** `physical_cores()` on Windows counts `RelationProcessorCore` records from
  `GetLogicalProcessorInformationEx` through a hand-written `extern "system"` in the new
  `crates/peregrine-core/src/win32.rs` (no dependency; SAFETY comments on every `unsafe`), with
  the logical count as the fallback when the call fails.
- **Unix-only tests.** `serve_lifetime.rs` now runs its lifetime half (the server outlives the
  drain timeout and keeps serving) everywhere; the `SIGTERM` half is `#[cfg(unix)]` with the
  reason. `server_multi_model.rs` lost its file-level `#![cfg(unix)]`: every test except the
  `SIGTERM` drain runs on Windows.
- **Lifetime commands are S7.3.** `serve --background`, `status` and `stop` return an explicit
  "not available on Windows yet (S7.3)" on Windows instead of failing through `kill`; `upgrade`
  already refuses non-Linux. A foreground `serve` stops on Ctrl-C, through `tokio`.
- **The job.** `test (windows-x86_64)` on `windows-2025` runs the same release-mode suite as the
  other x86-64 entries, and a smoke step builds `peregrine.exe`, checks `--version`, `tier`
  (`x86_64-avx2…` on the runner) and `run` on a model that `write_synthetic` writes. A green run
  takes about seven minutes.

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

## 8. S7.3 as built

Windows x86-64 reaches the bar in §1 for the lifetime commands, `upgrade`, the installer and the
release archive; the goldens and the measurement are S7.4's.

- **Stop and the drain.** Windows has no `SIGTERM`. `peregrine stop` writes
  `serve-<port>.stop` in the state directory (beside `serve.json`, named for the port like the
  log), the server watches it every 250 ms, removes it and runs the same drain a Unix `SIGTERM`
  starts (`serve_with_drain`, 30 s cap), and exits with the same status. `stop` waits the Unix
  40 s `DRAIN_TIMEOUT` and only then ends the process with `TerminateProcess`. Ctrl-C still works:
  the Windows `shutdown_signal` selects over `tokio`'s console handler and the stop-file tick.
- **Liveness and the background server.** `serve --background` spawns the child with
  `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`, redirects it to the same log file and writes the
  same state record; the parent waits for the child's state file (written after `bind`, before
  `listening on` is logged) before it reports success. `status` reads the record and checks the
  process with `OpenProcess` + `GetExitCodeProcess`; it also probes `/health` and prints the answer.
  The two calls are hand-written `extern "system"` declarations in
  `crates/peregrine-core/src/win32.rs` with SAFETY comments on every `unsafe`, like the S7.2
  topology call; there is no new dependency.
- **A detached child and the caller's pipes.** Windows' `CreateProcess(bInheritHandles = TRUE)`
  handed the detached server the caller's stdin/stdout/stderr pipe handles, so
  `peregrine serve --background | cat` and a test's `Command::output()` never saw EOF. The spawn
  now clears `HANDLE_FLAG_INHERIT` on this process's three standard handles for the duration of
  `CreateProcess` and restores them (`win32::NoInheritStdHandles`); the child's own stdio is passed
  explicitly to the log file and is unaffected.
- **Tests.** `serve_lifetime.rs` and `server_multi_model.rs` keep their Unix `SIGTERM` tests
  unchanged and gain Windows counterparts that ask for the drain through the stop file: the
  in-flight requests finish, a new connection is refused once the listener closes, and the process
  exits 0. `serve_lifetime.rs` also exercises `serve --background`, `status` (address and
  `health     ok`) and `stop` end to end. The Windows job runs all of them.
- **`upgrade`.** `release_platform(os, arch)` maps `windows`/`x86_64` to `windows-x86_64` (Windows
  on ARM gets the build-from-source answer), the asset is the `.zip`, and the SHA-256 comes from
  `certutil -hashfile` where Unix uses `sha256sum`/`shasum`. A running `.exe` cannot be overwritten
  but can be renamed: the installed binary becomes `peregrine.exe.old` (a unique name if that one
  is still running), the new one is moved into place, and every start removes `<name>.old` copies.
  Extraction uses PowerShell 5.1's `Expand-Archive`, the one archive tool every Windows has. The
  mapping is unit-tested by argument, and the rename swap with ordinary temporary files on every
  platform.
- **Installer.** `scripts/install.ps1` is Windows PowerShell 5.1 (no `??`, no ternary), needs no
  administrator rights, and follows `install.sh`'s shape: GitHub's latest tag (or `-Version`),
  `peregrine-vX.Y.Z-windows-x86_64.zip` and `SHA256SUMS` from `-BaseUrl` (a URL, a `file://` URL or
  a local directory, as `PEREGRINE_BASE_URL` is for the shell script), a `Get-FileHash` check, and
  `Expand-Archive` into `%LOCALAPPDATA%\Programs\peregrine` (or `-InstallDir`). It appends the
  directory to the **user** `Path` only when it is missing, prints the install and the undo
  commands, and refuses ARM64 with the build-from-source line.
- **CI.** `test (windows-x86_64)` gains an installer step after its smoke step: build.ps1 builds
  the zip, `install.ps1 -BaseUrl <repo>/dist -InstallDir <temp>` installs it, and the installed
  `peregrine.exe` must answer `--version` and `tier` (`x86_64-avx2…`). The job is green in about
  ten minutes cold.
- **Release.** `release.yml`'s matrix has `windows-2025` / `x86_64-pc-windows-msvc` /
  `windows-x86_64`; `scripts/release/build.ps1` is the PowerShell twin of `build.sh` (the Windows
  runner has no `strip` or `sha256sum`, and `Compress-Archive` writes `\` as an entry separator, so
  the zip entries are written with `/` through `System.IO.Compression`). The build sets
  `RUSTFLAGS=-C target-feature=+crt-static` for the release job only, so the `.exe` needs no Visual
  C++ redistributable; the binary still has no `target-cpu` and dispatches kernels at run time. The
  archive carries `peregrine.exe`, `LICENSE` and `NOTICE`, exactly like the Linux archives, and
  its SHA-256 joins `SHA256SUMS` in the same `hash  name` format. The publish job uploads `*.zip`
  and `install.ps1` beside `*.tar.gz` and `install.sh`, and is guarded on a tag push.
- **The dry run.** `gh workflow run release.yml --ref s7.3-windows-lifetime -f dry_run=true` was
  accepted: the three build jobs ran and uploaded artefacts, and the `release` job was skipped.
  The Windows build (`build (windows-x86_64)`) took 7 m 38 s and produced
  `peregrine-v0.2.0-windows-x86_64.zip` (3,111,510 bytes, SHA-256 `f900f0cd…908eb0`); its listing is
  `peregrine-v0.2.0-windows-x86_64/{LICENSE,NOTICE,peregrine.exe}` (8,186,389 bytes uncompressed).
- **Not done.** The Windows goldens (`-windows` directories) and a native Windows measurement are
  S7.4; every golden suite still skips on Windows. The `.exe` is unsigned (Authenticode needs a
  certificate, §7), and the installer's `irm … | iex` path is the one that avoids SmartScreen's
  browser-download mark rather than a signature.
