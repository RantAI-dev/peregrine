# S7: macOS and Windows

- Status: design, decided 2026-11-12. S7.1 (macOS on Apple Silicon), S7.2 (Windows x86-64 builds and
  passes the suite in CI), S7.3 (Windows lifetime, installer and release archive) and S7.4 (platform
  goldens, docs and the first measurements) built. §8, §9 and §10 are the "as built" sections.
- Owner request: "add support to windows, mac".

## 0. Where things stand

Peregrine releases Linux binaries only (x86-64 and aarch64). The code is closer to the other two
systems than the releases say:

- **macOS on Apple Silicon** builds and passes the test suite in CI on every PR (`test
  (macos-aarch64)`), with the NEON kernels and its own QoS call for the pool; S7.1 added the
  release archive, the installer path, the `upgrade` asset and the runner's synthetic goldens (§8).
  **Windows x86-64** builds and passes the suite too (`test (windows-x86_64)`, S7.2), and S7.3
  added the lifetime commands, `upgrade`, the installer and the release archive (§9).
- Neither has a release archive in **v0.2.0**, which shipped the two Linux archives only. Both
  platforms' archives and installers ship **from the release after v0.2.0**; until then they
  install with `cargo build --release`. Nothing has been measured on either yet (S7.4).

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
(scalar, AVX2; AVX-512 when a recording happens to draw an AVX-512 runner), compared on every pull
request on a pinned tier. Quickstart, README install section and the support table for both systems.
The first measurements are on GitHub's Windows and Linux runners, which are different machines; a
native Windows number from the AVX-512 desktop (it is a Windows machine; the Linux numbers published
so far come from WSL2 on it) against llama.cpp's Windows build is still to be taken.

## 6. Steps

| Step | What | Depends on |
|---|---|---|
| S7.1 | macOS on Apple Silicon: release job with a dry run, installer, `upgrade`, performance-core default, synthetic goldens, quickstart | — |
| S7.2 | Windows x86-64 builds and passes the suite in CI | — |
| S7.3 | Windows lifetime commands, `upgrade`, `install.ps1`, release job | S7.2 |
| S7.4 | Windows goldens, docs for both systems, the first measurement (GitHub's runners; native hardware still open) | S7.3 |
| S7.5 | A release that ships all four archives | S7.1, S7.4 |

## 7. Open points

- **A Mac for measurement.** Without one, macOS is supported but unmeasured.
- **Signing.** Unsigned binaries run when fetched by the installers. A Developer ID signature and
  notarisation for macOS, and Authenticode for Windows (SmartScreen warns on a downloaded
  unsigned `.exe`), need certificates the project does not have.
- **Windows Defender and mapped model files (closed by S7.4).** Measured on GitHub's Windows
  runners: the first open of a freshly downloaded 271 MB model took 2.3–3.6 s against 0.1–0.3 s for
  the second open on the same machine, about 9–13 ms per MB, and the second open was 1.2–1.3×
  Linux's. That is Defender's real-time scan of a file it has not seen, and it is paid once per
  file: a multi-gigabyte model costs tens of seconds on its first run after a download and nothing
  afterwards. Where that matters, exclude the model directory from real-time scanning
  (`Add-MpPreference -ExclusionPath '<dir with the models>'`), the documented exclusion; the engine
  needs no change (§10).

## 8. S7.1 as built

macOS on Apple Silicon is built, tested and packaged in this step; the bar in §1 is met for it. Its
archive and installer ship from the release after v0.2.0 — v0.2.0 itself has the two Linux archives
only, and `scripts/install.sh` says so on a Mac pointed at it.

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
  tag push, so it cannot run. The binary is ad-hoc signed, not notarised (§7). A dry run of this
  branch on `macos-15` produced the archive for the then-current workspace version (`vX.Y.Z`,
  2,717,446 bytes, SHA-256 `6fca9591…e17b0`), and the installer installed that artefact on the
  runner and ran `peregrine --version` and `peregrine tier` (`aarch64-neon-dotprod`).
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

## 9. S7.3 as built

Windows x86-64 reaches the bar in §1 for the lifetime commands, `upgrade`, the installer and the
release archive; the goldens and the measurement are S7.4's. As with macOS, the archive and the
installer ship from the release after v0.2.0: v0.2.0 has no Windows zip.

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
  The Windows build (`build (windows-x86_64)`) took 7 m 38 s and produced the archive for the
  then-current workspace version (`vX.Y.Z`, 3,111,510 bytes, SHA-256 `f900f0cd…908eb0`); its
  listing is `peregrine-vX.Y.Z-windows-x86_64/{LICENSE,NOTICE,peregrine.exe}` (8,186,389 bytes
  uncompressed).
- **Not done.** The Windows goldens (`-windows` directories) and a native Windows measurement are
  S7.4; every golden suite still skips on Windows. The `.exe` is unsigned (Authenticode needs a
  certificate, §7), and the installer's `irm … | iex` path is the one that avoids SmartScreen's
  browser-download mark rather than a signature.

## 10. S7.4 as built

Windows has its own goldens and both systems compare theirs on every pull request; the first
numbers from GitHub's runners are in. The bar in §1 holds for both platforms except item 5's
real-model goldens (CI has no model files) and a published performance comparison (§7).

- **The recorder.** `record-goldens.yml` takes an `isa` input that sets `PEREGRINE_ISA` for every
  step (`scalar`, `avx2`, …; empty is the runner's own tier) and names its artefact
  `goldens-<runner>-<isa|native>`, and it prints the runner's CPU. Two defects only a Windows run
  showed: an apostrophe inside `${VAR:-word}` in a double-quoted bash line opens a single-quoted
  string there (the CPU step died on it), and the runner's `tar` reads an archive path of the form
  `D:\…` as a remote host (`D:`), so the archive is made from the shell's POSIX path.
- **The Windows sets.** Three sets of synthetic goldens, as many as each tier's Linux directory
  (60, 61 and 62, in the table's order) plus the session fixture
  (`tests/fixtures/session/x86_64-*-windows.txt`), each recorded with the workflow's own verify run
  reporting **0 skips against 52 before** (`no golden for tier` / `no recording for tier`), and each
  proven by a later run of the same tier on a different runner and CPU, with 0 skips before and
  after:

  | tier (input) | recorded on | proven on |
  |---|---|---|
  | `x86_64-windows` (`isa=scalar`) | AMD EPYC 7763 | AMD EPYC 9V74 |
  | `x86_64-avx2-windows` (`isa=avx2`) | Intel Xeon Platinum 8573C (an AVX-512 machine, capped) | AMD EPYC 9V45, and again on an EPYC 7763 drawn natively |
  | `x86_64-avx2-avx512-vnni-gfni-vbmi-windows` (native) | Intel Xeon Platinum 8370C | Intel Xeon Platinum 8573C |

  The recordings crossing CPU families are the point: the scalar and AVX2 sets came out identical on
  an AMD and an Intel machine, which is what §3's "the arithmetic's, not the C runtime's" buys. The
  AVX-512 set was found by the first native start and proven by a fourth (the pool's AVX-512 draws
  were an Intel 8370C and an Intel 8573C; its AVX2 draws were EPYC 7763/9V74/9V45 machines). The
  sweep after merging S5.4e recorded nothing on any of the three: 0 skips before, 0 after.
  The final sweep, after `main` moved under the branch (S4.8b's `nomic-bert-moe` and S5.4f's
  form-only-where-it-wins), recorded `nomic-bert-moe-Q8_0` for the scalar set (AMD EPYC 7763), the
  AVX2 set (EPYC 7763) and the AVX-512 set (a third native start drew an Intel Xeon 6973P-C), and
  nothing else; each run's verify step reported 0 skips before and after, and the next recordings of
  macOS, scalar and AVX2 recorded nothing at all (EPYC 9V74, EPYC 7763, EPYC 7763). It also removed
  three Windows files nothing read any more: `llama-Q4_K-fast` and `llama-Q6_K-fast` from
  `x86_64-windows` and `llama-Q4_K-fast` from `x86_64-avx2-windows`. S5.4f takes the 8-bit form only
  where it wins on the tier — no format on the scalar tier, Q6_K only on AVX2 — so the fast test
  compares the fast run with the exact engine and the tier's exact golden there and has no fast
  golden of its own. The AVX-512 set's two fast goldens are still read and reproduced byte for byte
  in that recording.
- **The pool and the pin.** The Windows pool draws AVX2-only and AVX-512 machines, and
  `PEREGRINE_ISA` takes capability names, not tier names. `ci.yml` therefore runs the
  golden-bearing targets (`synthetic`, `embeddings`, `bert_heads`, `modern_bert_heads`,
  `session_fixture`) a second time with `PEREGRINE_ISA=avx2`, so the job compares
  `x86_64-avx2-windows` whatever CPU it drew, and fails if a "no golden for tier" / "no recording
  for tier" line appears — counted the way the recorder counts them. The step sets
  `PEREGRINE_GOLDEN=require`, without which the embedding families' golden tests skip silently and
  the grep would have nothing to match. The rest of the suite still runs natively, where
  `PEREGRINE_GOLDEN` is unset, so a real-model golden skips without its file, as everywhere in CI.
- **macOS.** The guard is the same step at `aarch64-neon-dotprod` (the `macos-15` runner reports
  that tier and stays there under the cap), so a golden added on Linux and not swept onto macOS
  fails a PR. The S7.4 sweep on `macos-15` (Apple M1) recorded nothing: 0 skips before, 0 after,
  the set had kept up with the new families and the `*-fast` goldens. It was swept again after the
  S5.4e merge, which added `*-fast` goldens for the four aarch64 tiers; those already existed here
  and are byte-identical to the `aarch64-neon-dotprod` ones, so nothing was added. The final sweep,
  after S4.8b merged, added `nomic-bert-moe-Q8_0` and nothing else (0 skips before and after; the
  next recording added nothing). The macOS directory is a superset of the `aarch64-neon-dotprod`
  one: `jina-bert-Q8_0`, `nomic-bert-Q8_0` and now `nomic-bert-moe-Q8_0` exist only here, recorded
  in earlier macOS sweeps, and their tests skip on the aarch64 Linux tiers, which record no golden
  for them.
- **The installer.** `scripts/install.ps1` read the user `Path` with
  `[Environment]::GetEnvironmentVariable("Path","User")` — which expands `%VARIABLES%` — and wrote it
  back with `[Environment]::SetEnvironmentVariable`, which stores a plain `REG_SZ`, so
  `%USERPROFILE%\bin` became a literal path and the value's kind changed. It now opens
  `HKCU:\Environment` for writing with `[Microsoft.Win32.Registry]::CurrentUser` (the key that
  `Get-Item` returns is read-only — the first CI run failed on exactly that), reads the raw value with
  `DoNotExpandEnvironmentNames`, compares entries by what they expand to, appends the install
  directory and writes it back with `RegistryValueKind::ExpandString`; the broadcast is a no-op
  `[Environment]::SetEnvironmentVariable` on a scratch name, the one PowerShell 5.1 has without
  `Add-Type`. The undo text is unchanged. The CI installer step starts from a user `Path` of
  `%USERPROFILE%\peregrine-test-bin`, and after the install checks the raw value still contains the
  literal `%USERPROFILE%`, contains the install directory, and is still typed `ExpandString`; it
  restores the runner's value in a `finally`.
- **The measurements.** Informational, on GitHub's runners, not standings: the two jobs are
  different machines and the pool draws vary. Both fetched SmolLM2-360M-Instruct Q4_K_M (271 MB, the
  `compat` job's pinned SHA-256), ran `peregrine bench --prompt-tokens 512 --tokens 64 --threads 4`
  five times, and the load is `serve --background`'s wall clock to ready and its own
  `loaded in … s` line. Pairs where both jobs drew the same CPU model and tier:

  | cell | Windows | Linux | W ÷ L |
  |---|---|---|---|
  | prefill tok/s, AMD EPYC 9V45, AVX-512 | 626.1 (545.0–642.1) | 661.0 (659.5–678.8) | 0.95 |
  | decode tok/s, AMD EPYC 9V45, AVX-512 | 114.2 (109.4–121.0) | 121.1 (114.2–127.8) | 0.94 |
  | prefill tok/s, AMD EPYC 7763, AVX2 | 253.6 (248.2–255.7) | 282.4 (240.9–283.8) | 0.90 |
  | decode tok/s, AMD EPYC 7763, AVX2 | 74.1 (70.3–75.2) | 92.7 (91.8–93.5) | 0.80 |

  Medians with the min–max of the five runs in brackets. The other Windows draw was an Intel Xeon
  Platinum 8370C (AVX-512: 310.8 prefill, 58.5 decode) and the other Linux draw an AMD EPYC 7763 at
  AVX2; a cross-CPU ratio only says "not grossly different", and these are between 0.80 and 0.95 on
  the same CPU, so nothing points at the pool's wait path and it was not changed.

  **Load and Defender** (the §7 open point): first open of the freshly downloaded 271 MB file, then
  the second.

  | machine | first load | second load |
  |---|---|---|
  | Windows, AMD EPYC 7763 | 3.26 s | 0.24 s |
  | Windows, AMD EPYC 9V45 | 3.64 s | 0.13 s |
  | Windows, Intel Xeon 8370C | 2.45 s | 0.25 s |
  | Linux, AMD EPYC 7763 | 0.30 s | 0.20 s |
  | Linux, AMD EPYC 9V45 | 0.30 s | 0.10 s |

  Wall clock from `serve --background` to ready, which includes process start (the engine's own line
  reads 2.3–3.5 s against 0.0–0.1 s). Windows ÷ Linux is 10.7× and 12.0× on the first load and 1.2×
  and 1.3× on the second: the first open of an unseen file is Defender's real-time scan, about
  9–13 ms per MB, paid once. See §7 for the exclusion and what it means for multi-gigabyte files.

  **The Windows lifetime commands**, timed on the runners: from `serve --background` to `/health`
  0.31 s (EPYC 7763), 0.16 s (EPYC 9V45) and 0.30 s (Xeon 8370C); one 8-token chat request 0.21 s,
  0.11 s and 0.21 s; `stop` 0.14 s, 0.13 s and 0.15 s; `status` reported no managed server after
  each.
- **Docs.** `docs/dev/testing.md` and `CONTRIBUTING.md` carry the three recording commands and what
  a golden-adding change owes macOS and Windows; `docs/dev/releasing.md` has the pre-release golden
  sweep and the archive smoke tests; `CHANGELOG.md`, `docs/roadmap.md` and §7 record the rest.
- **Not done.** No native measurement on real Windows hardware (these are shared CI runners, three
  or four cores of a large machine, and not a benchmark host, so `docs/benchmark.md` gains no row),
  and no signing or notarisation (§7).
