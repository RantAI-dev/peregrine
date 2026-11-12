# Running Peregrine as a service

Peregrine can manage one long-lived server for you — start it in the background, inspect it, stop it,
and upgrade it while it keeps serving.

## Background server

```sh
peregrine serve -m hf:mradermacher/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct.Q4_K_M.gguf \
  --port 8201 --background
# peregrine: listening on http://127.0.0.1:8201/ (pid 22144)
# peregrine: background server ready at http://127.0.0.1:8201/ (pid 22144)

peregrine status
peregrine stop
```

`--background` re-executes the same binary in its own process group (Unix) or detached from the
console (`CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`, Windows), redirects stdout/stderr to a log
and records lifecycle state; the command returns only after the server is listening, so the first
`curl` works immediately. `--startup-timeout` (default 300 s) bounds the wait for a large model.

| | |
|---|---|
| State | `$PEREGRINE_STATE_DIR` → `$XDG_STATE_HOME/peregrine` → `%LOCALAPPDATA%\peregrine` on Windows → `~/.local/state/peregrine` (`serve.json`) |
| Log | `serve-<port>.log` in the state directory, or `--log-file PATH` |
| Address | `status` prints the URL and the pid |

`peregrine status` is also the check for a supervisor: exit status 0 means a managed server exists,
1 means none (or a stale state file, which the message names). The state file is removed on a graceful
stop, and a stale file (dead pid) is reported rather than treated as a server. On Windows `status`
also probes `/health` and prints its answer (a server that is still loading a model is alive but not
yet healthy).

### Stopping on Windows

Windows has no `SIGTERM`. `peregrine stop` writes `serve-<port>.stop` in the state directory; the
server watches it every 250 ms, removes it and drains exactly as a Unix server does on `SIGTERM`.
`stop` waits 40 s for the drain (the Unix `DRAIN_TIMEOUT`) and only then ends the process with
`TerminateProcess`. A foreground `serve` stops on Ctrl-C, which `tokio` delivers on Windows too.

`status` checks liveness with `OpenProcess` and `GetExitCodeProcess` (hand-written Win32 in
`crates/peregrine-core/src/win32.rs`, no new dependency).

## Upgrading without downtime

```sh
peregrine upgrade --check     # running vs newest release, installs nothing
peregrine upgrade             # install the newest release, replace the managed server
```

`upgrade`:

1. reads the newest release tag from `peregrine.rantai.dev`, falling back to GitHub
   (`--version vX.Y.Z` pins one; `--base-url`/`PEREGRINE_RELEASE_BASE` and
   `PEREGRINE_RELEASE_LATEST` point at a mirror);
2. downloads the platform's archive (`linux-x86_64`, `linux-aarch64`, `macos-aarch64`,
   `windows-x86_64`) and its `SHA256SUMS` (vanity host first, GitHub as the fallback), verifies the
   checksum, extracts it, and runs the new binary with `--version` **before** touching anything.
   Intel Macs and Windows on ARM64 have no archive: there `upgrade` says to build from source;
3. copies it beside the installed binary and renames it into place, so the replacement is atomic and
   a running server keeps serving from its already-mapped old image. On Windows a running `.exe`
   cannot be overwritten but can be renamed, so `peregrine.exe` becomes `peregrine.exe.old` and the
   new one takes its place; the next start removes the `.old` copy;
4. if a managed server is running, on Linux and macOS starts the new binary with the same `serve`
   arguments and `SO_REUSEPORT`, waits for its lifecycle state, and only then sends the old process
   `SIGTERM`. The old process stops accepting, drains in-flight requests (30 s cap), and exits. On
   Windows there is no `SO_REUSEPORT` and no `SIGTERM`: the old server is asked to drain through the
   stop file first, and the new binary starts after it has exited, so a Windows upgrade has a short
   gap rather than a zero-downtime handoff.

On Linux and macOS both processes listen on the same port during the handoff, so no connection is
refused. Tested with 300 requests in flight across the swap: zero failures.

The running server must have been started with `--background` (which enables `SO_REUSEPORT`).
`--reuse-port` exposes the same socket option for a foreground server; without it a replacement
cannot bind the port while the old process is still draining. `--no-restart` installs the binary but
leaves the running server on the old image for a controlled restart. Foreground servers are not
managed: `upgrade` replaces the file and tells you to restart.

`upgrade` refuses to replace a cargo build under `target/` unless `--force` is given; use
`--exe PATH` to install somewhere else explicitly.

## Reloading a rebuilt binary (SIGHUP)

Unix only: Windows has neither `SIGHUP` nor `SO_REUSEPORT`. There, replace the binary with
`peregrine upgrade` or `peregrine stop` followed by `peregrine serve --background`.

Rebuilding over the installed path does not change a running process. To hand off to the new binary
without downtime:

```sh
cargo build --release
install -m 0755 target/release/peregrine /usr/local/bin/peregrine   # new inode
kill -HUP "$(peregrine status | awk '/^pid/{print $2}')"
```

On `SIGHUP` the server re-executes the binary at its own path with the same `serve` arguments and
`SO_REUSEPORT`, waits for the replacement to report ready, then drains. The same inode means the same
binary: a SIGHUP with no rebuild logs `unchanged` and keeps the current process, so the signal cannot
restart in a loop. The replacement re-reads the environment, the tuning profile (`peregrine tune`) and
the model file, but the arguments are the ones the server was started with.

Requirements and failure behaviour:

- the old process must have `SO_REUSEPORT` (a background server does; a foreground one needs
  `--reuse-port`), or the replacement cannot bind while the old one drains;
- if the replacement fails to start (bad binary, model error, port conflict), the old process keeps
  serving and logs why — a SIGHUP never takes a working server down.

`peregrine-server` (the standalone binary) supports the same signal.

## systemd

For boot-time start, run the server in the foreground under systemd rather than with `--background`:

```ini
# /etc/systemd/system/peregrine.service
[Unit]
Description=Peregrine OpenAI-compatible server
After=network.target

[Service]
User=peregrine
ExecStart=/usr/local/bin/peregrine serve \
  -m /srv/models/model.gguf --port 8201 --host 127.0.0.1
Environment=PEREGRINE_CALIBRATE=1
Restart=on-failure
KillSignal=SIGTERM
TimeoutStopSec=40

[Install]
WantedBy=multi-user.target
```

`SIGTERM` triggers the same graceful drain. `TimeoutStopSec` should exceed the server's 30 s drain
cap. Upgrade a systemd deployment by replacing the binary and restarting the unit; systemd does not
share the listening socket, so use `peregrine upgrade` with `--background` when zero downtime
matters, or run two units behind a reverse proxy.

## Notes

- One managed server per user/state directory. Run one process per model, as with llama-server.
- The log is appended, not rotated; point `--log-file` at a logrotate-managed path for long-lived services.
- `SO_REUSEPORT` and the zero-downtime handoff are Unix features; Windows uses the stop file and a
  short restart gap instead. `--background`, `status`, `stop` and `upgrade` exist on Linux, macOS
  and Windows on x86-64. The release archives exist on Linux x86-64 and aarch64 today; the
  `macos-aarch64` and `windows-x86_64` archives (and their installers) ship from the release after
  v0.2.0, so `upgrade` names the right asset but v0.2.0 has none to download until then.
