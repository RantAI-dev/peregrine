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

`--background` re-executes the same binary in its own process group, redirects stdout/stderr to a log
and records lifecycle state; the command returns only after the server is listening, so the first
`curl` works immediately. `--startup-timeout` (default 300 s) bounds the wait for a large model.

| | |
|---|---|
| State | `$PEREGRINE_STATE_DIR` → `$XDG_STATE_HOME/peregrine` → `~/.local/state/peregrine/serve.json` |
| Log | `serve-<port>.log` in the state directory, or `--log-file PATH` |
| Address | `status` prints the URL and the pid |

`peregrine status` is also the check for a supervisor: exit status 0 means a managed server exists,
1 means none (or a stale state file, which the message names). The state file is removed on a graceful
stop, and a stale file (dead pid) is reported rather than treated as a server.

## Upgrading without downtime

```sh
peregrine upgrade --check     # running vs newest release, installs nothing
peregrine upgrade             # install the newest release, replace the managed server
```

`upgrade`:

1. reads the newest release tag from `peregrine.rantai.dev`, falling back to GitHub
   (`--version vX.Y.Z` pins one; `--base-url`/`PEREGRINE_RELEASE_BASE` and
   `PEREGRINE_RELEASE_LATEST` point at a mirror);
2. downloads the Linux archive and its `SHA256SUMS` (vanity host first, GitHub as the fallback),
   verifies the checksum, extracts it, and runs the new binary with `--version` **before** touching
   anything;
3. copies it beside the installed binary and renames it into place, so the replacement is atomic and a
   running server keeps serving from its already-mapped old image;
4. if a managed server is running, starts the new binary with the same `serve` arguments and
   `SO_REUSEPORT`, waits for its lifecycle state, and only then sends the old process `SIGTERM`.
   The old process stops accepting, drains in-flight requests (30 s cap), and exits.

Because both processes listen on the same port during the handoff, no connection is refused. Tested
with 300 requests in flight across the swap: zero failures.

The running server must have been started with `--background` (which enables `SO_REUSEPORT`).
`--reuse-port` exposes the same socket option for a foreground server; without it a replacement
cannot bind the port while the old process is still draining. `--no-restart` installs the binary but
leaves the running server on the old image for a controlled restart. Foreground servers are not
managed: `upgrade` replaces the file and tells you to restart.

`upgrade` refuses to replace a cargo build under `target/` unless `--force` is given; use
`--exe PATH` to install somewhere else explicitly.

## Reloading a rebuilt binary (SIGHUP)

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
- `SO_REUSEPORT` and the background mode are Linux features; the release binaries are Linux-only.
