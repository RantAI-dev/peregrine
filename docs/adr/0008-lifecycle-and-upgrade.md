# 0008: Managed background server and self-update

- Status: Implemented
- Date: 2026-11-10

## Context

The binary is distributed as a release archive, not through a package manager (`publish = false`), so
`peregrine upgrade` has to replace the executable itself. Operators also want to start a server, leave
it running, and upgrade it without dropping the requests a client is waiting on.

The server had no lifecycle beyond the foreground process: `TcpListener::bind` then
`axum::serve(...).await`, no `SIGTERM` handling and no graceful drain. Replacing the file is atomic, but
a restart closes the listening socket, and a connection accepted at that moment is refused.

## Decision

1. **Background mode is a re-exec, not a `fork()` daemon.** `serve --background` spawns the same binary
   with the lifecycle flags in its own process group and waits until the child has bound and written a
   state file, then returns. No `unsafe` `fork`/`setsid`, and the parent can report a startup failure
   instead of daemonising a process that dies a second later.
2. **One state file is the interface** (`$XDG_STATE_HOME/peregrine/serve.json`): pid, version, address,
   model, log, and the `serve` arguments. `status`, `stop` and `upgrade` read that one file; the server
   writes it after `bind` (so its presence means listening) and removes it on shutdown only when the pid
   still matches. No second parser keeps the restarted command line in sync with the flags.
3. **Zero downtime is a `SO_REUSEPORT` handoff.** A background server binds with `SO_REUSEPORT`;
   `upgrade` installs the new binary, starts it with the same arguments beside the old one, waits for the
   new state file, then sends the old pid `SIGTERM`. The old process stops accepting, drains in-flight
   requests (30 s cap), and exits. Connections are load-balanced between the two processes during the
   window, so none is refused. Reuseport is opt-in (`--reuse-port`, implied by `--background`) so a typo
   cannot silently start a second server on an occupied port.
4. **Verify before replacing, replace atomically.** `upgrade` checks `SHA256SUMS`, extracts, and runs
   the new binary's `--version` before copying it beside the installed one and renaming it in. A bad
   archive leaves the old binary and the old server untouched; a running process keeps serving from its
   mapped image either way. Cargo builds under `target/` are refused without `--force`.
5. **Graceful shutdown on `SIGTERM` and `Ctrl-C`.** Stop accepting, drain, remove the state file, exit.
   `peregrine stop` waits for that drain; systemd gets the same behaviour with `KillSignal=SIGTERM`.

6. **`SIGHUP` re-execs the binary on disk.** The running process starts the path it came from with
   the same `serve` arguments and `SO_REUSEPORT`, waits for the replacement's state file, then drains.
   The on-disk inode is compared with `/proc/self/exe`: the same inode means the same binary, so the
   signal is a no-op instead of a restart loop. A handoff that fails to bind or to load leaves the
   old process serving; the only requirement is that the old process bound with `SO_REUSEPORT`
   (always true for `--background`, opt-in for foreground). This is a binary reload, not a config
   reload: the arguments are replayed and the profile is re-resolved, but flags cannot change.

## Consequences

- One managed server per user/state directory. Several models need separate state directories or the
  systemd/Docker paths; this is the same "one process, one model" rule as llama-server.
- A foreground `serve` is not managed: `upgrade` replaces the binary and asks for a restart.
- A zero-downtime upgrade or SIGHUP requires the running server to have `SO_REUSEPORT`; an old
  deployment started without `--reuse-port` needs one ordinary restart first.
- The background log is appended, not rotated; long-lived services point `--log-file` at a logrotate
  path.
- Passing the listening socket to a new process (`SCM_RIGHTS`) would also be zero-downtime without
  reuseport, but needs a supervisor protocol and a second long-lived channel; reuseport is one
  `setsockopt` and works with the kernel's own balancing.

## Alternatives rejected

- **systemd-only**: not every deployment uses systemd, and the install story is a bare archive.
- **In-process daemonisation** (`fork`, `setsid`, closing fds): `unsafe` outside the kernels, and it
  cannot report a bind or model-load failure to the shell.
- **Stop-then-start**: simpler, but it is the downtime this record exists to remove.
- **Automatic `systemctl restart`**: couples the CLI to one supervisor.
