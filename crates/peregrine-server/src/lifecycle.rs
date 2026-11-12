//! The managed server's lifecycle: the state file `peregrine status`, `stop` and `upgrade` read.
//!
//! A server started with `peregrine serve --background` binds with `SO_REUSEPORT`, writes this
//! file once it is listening, and removes it on a graceful shutdown. The file carries the argv
//! needed to start the same server again, so `peregrine upgrade` can start the new binary beside
//! the old one, wait until it is listening, and only then stop the old one.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// What a running managed server records about itself.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct State {
    pub pid: u32,
    /// `CARGO_PKG_VERSION` of the binary that is running, which may be older than the file on disk.
    pub version: String,
    pub host: String,
    pub port: u16,
    pub model: String,
    /// The log the background server writes to.
    pub log: Option<PathBuf>,
    /// The `serve` arguments, without the lifecycle flags (`--state-file`, `--log-file`,
    /// `--reuse-port`); `upgrade` re-runs them.
    pub args: Vec<String>,
    pub started: u64,
}

impl State {
    /// The address a client uses, with a wildcard bind written as localhost.
    pub fn url(&self) -> String {
        let host = match self.host.as_str() {
            "0.0.0.0" | "::" | "" => "127.0.0.1",
            h => h,
        };
        format!("http://{host}:{}/", self.port)
    }
}

/// Where the state file lives: `$PEREGRINE_STATE_DIR`, else `$XDG_STATE_HOME/peregrine`, else
/// `%LOCALAPPDATA%\peregrine` on Windows, else `~/.local/state/peregrine` (S7.2).
pub fn state_dir() -> PathBuf {
    if let Ok(d) = std::env::var("PEREGRINE_STATE_DIR") {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    if let Ok(d) = std::env::var("XDG_STATE_HOME") {
        if !d.is_empty() {
            return PathBuf::from(d).join("peregrine");
        }
    }
    #[cfg(windows)]
    {
        // Windows has no XDG state directory; the per-user local application data is where a
        // program keeps state of its own (`%LOCALAPPDATA%\peregrine\serve.json`).
        if let Ok(d) = std::env::var("LOCALAPPDATA") {
            if !d.is_empty() {
                return PathBuf::from(d).join("peregrine");
            }
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".local/state/peregrine")
}

/// The one managed server's state file.
pub fn state_path() -> PathBuf {
    state_dir().join("serve.json")
}

/// The default log for a background server on `port`.
pub fn log_path(port: u16) -> PathBuf {
    state_dir().join(format!("serve-{port}.log"))
}

/// The stop file a Windows server watches, named for its port like the log.
///
/// Windows has no `SIGTERM`. `peregrine stop` creates this file; the serving process notices it
/// within about 250 ms, removes it, and drains exactly as `SIGTERM` makes a Unix server drain
/// (S7.3). Unix servers never look at it.
pub fn stop_path(port: u16) -> PathBuf {
    state_dir().join(format!("serve-{port}.stop"))
}

/// Ask the Windows server on `port` to drain by creating its stop file.
#[cfg(windows)]
pub fn request_stop(port: u16) -> Result<(), String> {
    let path = stop_path(port);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    std::fs::write(&path, b"stop\n").map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Read the state file. `None` when there is none or it does not parse.
pub fn read(path: &Path) -> Option<State> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Write the state file atomically, so a reader never sees half a file.
pub fn write(path: &Path, state: &State) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let text = serde_json::to_string_pretty(state).map_err(|e| format!("state: {e}"))?;
    std::fs::write(&tmp, text).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Remove the state file only when it still names `pid`: a newer server may have replaced it.
pub fn remove_if_ours(path: &Path, pid: u32) {
    if read(path).is_some_and(|s| s.pid == pid) {
        let _ = std::fs::remove_file(path);
    }
}

/// Whether a process is running. Linux reads `/proc`; the other Unix systems ask `kill -0`.
///
/// Windows has no `/proc` and no `kill`: `win32::process_alive` opens the process and reads its
/// exit code (S7.3), which is also what `status` and `stop` need.
#[cfg(windows)]
pub fn alive(pid: u32) -> bool {
    peregrine_core::win32::process_alive(pid)
}

/// Whether a process is running. Linux reads `/proc`; elsewhere `kill -0` is asked.
#[cfg(not(windows))]
pub fn alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .is_ok_and(|s| s.success())
    }
}

/// End a process hard (`TerminateProcess`). The stop file is the graceful path; this is the
/// last resort `peregrine stop` (and, on Windows, `upgrade`) use after the drain timeout.
#[cfg(windows)]
pub fn terminate(pid: u32) -> Result<(), String> {
    peregrine_core::win32::terminate_process(pid)
        .map_err(|e| format!("cannot terminate pid {pid}: {e}"))
}

/// Ask a process to shut down (`SIGTERM`), and say whether the signal was sent.
#[cfg(not(windows))]
pub fn terminate(pid: u32) -> Result<(), String> {
    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .map_err(|e| format!("cannot run kill: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("kill -TERM {pid} failed"))
    }
}

/// `clock` seconds since the epoch, for the state file.
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The `serve` arguments a background server should be re-run with: the process's own argv minus
/// the leading `serve` (when the CLI supplied it) and minus the lifecycle flags.
///
/// This is what `upgrade` replays, so everything a user typed — `--opt`, `--slots`, `--kv`,
/// `--chat-template` — survives the replacement without a second parser to keep in sync.
pub fn forwarded_args<I: IntoIterator<Item = String>>(argv: I) -> Vec<String> {
    let mut argv = argv.into_iter().peekable();
    if argv.peek().map(String::as_str) == Some("serve") {
        argv.next();
    }
    let mut out = Vec::new();
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--state-file" | "--log-file" | "--startup-timeout" => {
                let _ = argv.next();
            }
            "--background" | "--reuse-port" => {}
            _ if arg.starts_with("--state-file=")
                || arg.starts_with("--log-file=")
                || arg.starts_with("--startup-timeout=") => {}
            _ => out.push(arg),
        }
    }
    out
}

/// The result of a `SIGHUP` handoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handoff {
    /// The binary on disk is the one already running: nothing was spawned.
    Unchanged,
    /// A replacement is listening; the caller should drain and exit.
    Restarted(u32),
}

/// Start a detached server with `SO_REUSEPORT` and wait until it writes its own state.
///
/// The caller must already be listening on the same port with `SO_REUSEPORT` (a background server
/// does; a foreground one needs `--reuse-port`). `log` is None for a foreground handoff, where the
/// child inherits the terminal and stays in its process group.
pub fn spawn_server(
    exe: &Path,
    args: &[String],
    log: Option<&Path>,
    detach: bool,
    state_path: &Path,
    timeout: Duration,
    cancel: Option<&AtomicBool>,
) -> Result<State, String> {
    let mut cmd = Command::new(exe);
    // The `peregrine-server` binary runs the server directly; any other name (the `peregrine` CLI,
    // including a renamed or versioned copy) takes the `serve` subcommand.
    if !exe
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("peregrine-server"))
    {
        cmd.arg("serve");
    }
    cmd.args(args)
        .arg("--reuse-port")
        .arg("--state-file")
        .arg(state_path);
    if let Some(log) = log {
        if let Some(dir) = log.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)
            .map_err(|e| format!("cannot open {}: {e}", log.display()))?;
        let _ = writeln!(
            file,
            "peregrine: starting {} for {}",
            env!("CARGO_PKG_VERSION"),
            state_path.display()
        );
        cmd.arg("--log-file")
            .arg(log)
            .stdin(Stdio::null())
            .stdout(Stdio::from(
                file.try_clone()
                    .map_err(|e| format!("cannot share {}: {e}", log.display()))?,
            ))
            .stderr(Stdio::from(file));
    } else {
        // Foreground handoff: keep the terminal, so Ctrl-C still reaches the replacement.
        cmd.stdin(Stdio::null());
    }
    if detach {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        // `CREATE_NEW_PROCESS_GROUP` keeps Ctrl-C in the parent's console from reaching the
        // detached server; `DETACHED_PROCESS` gives it no console at all (S7.3).
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
            const DETACHED_PROCESS: u32 = 0x0000_0008;
            cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS);
        }
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", exe.display()))?;
    let id = child.id();
    let started = Instant::now();
    loop {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("handoff cancelled by shutdown".into());
        }
        if let Some(state) = read(state_path) {
            if state.pid == id {
                if !alive(id) {
                    return Err(format!(
                        "the replacement exited immediately{}",
                        log.map_or(String::new(), |p| format!("; see {}", p.display()))
                    ));
                }
                return Ok(state);
            }
        }
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "the replacement exited with {status}{}",
                log.map_or(String::new(), |p| format!("; see {}", p.display()))
            ));
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "the replacement did not report ready in {} s{}",
                timeout.as_secs(),
                log.map_or(String::new(), |p| format!("; see {}", p.display()))
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Re-exec the binary on disk beside this process (`SIGHUP`). Returns [`Handoff::Unchanged`] when
/// the path still names the running inode, so a repeated signal cannot restart in a loop.
///
/// `cancel` lets a shutdown signal stop a handoff that is waiting for the replacement to load a
/// large model: the in-flight child is killed and reaped rather than orphaned.
pub fn handoff(
    state_file: Option<&Path>,
    log: Option<&Path>,
    cancel: Option<&AtomicBool>,
) -> Result<Handoff, String> {
    let (exe, unchanged) = running_exe()?;
    if unchanged {
        return Ok(Handoff::Unchanged);
    }
    let args = forwarded_args(std::env::args().skip(1));
    let child_state = state_file.map_or_else(
        || std::env::temp_dir().join(format!("peregrine-handoff-{}.json", std::process::id())),
        Path::to_path_buf,
    );
    if state_file.is_none() {
        let _ = std::fs::remove_file(&child_state);
    }
    let state = spawn_server(
        &exe,
        &args,
        log,
        log.is_some(),
        &child_state,
        Duration::from_secs(300),
        cancel,
    );
    if state_file.is_none() {
        // A foreground handoff owns a temporary state file in `/tmp`; it is only a readiness
        // signal for this call, so remove it whether the replacement started or not.
        let _ = std::fs::remove_file(&child_state);
    }
    state.map(|state| Handoff::Restarted(state.pid))
}

/// The executable path, without the ` (deleted)` suffix Linux appends after a replacement, and
/// whether it still names the running inode.
fn running_exe() -> Result<(PathBuf, bool), String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate this binary: {e}"))?;
    let text = exe.to_string_lossy().into_owned();
    let path = text.strip_suffix(" (deleted)").map_or(exe, PathBuf::from);
    if !path.exists() {
        return Err(format!(
            "{} no longer exists; build or install the replacement first",
            path.display()
        ));
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(new), Ok(running)) = (
            std::fs::metadata(&path),
            std::fs::metadata("/proc/self/exe"),
        ) {
            if new.ino() == running.ino() && new.dev() == running.dev() {
                return Ok((path, true));
            }
        }
    }
    Ok((path, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fwd(args: &[&str]) -> Vec<String> {
        forwarded_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn forwarded_args_drop_only_the_lifecycle_flags() {
        assert_eq!(
            fwd(&[
                "serve",
                "-m",
                "hf:org/name/model.gguf",
                "--port",
                "8201",
                "--state-file",
                "/tmp/serve.json",
                "--log-file=/tmp/serve.log",
                "--reuse-port",
                "--startup-timeout",
                "30",
                "--background",
                "--opt",
                "SPIN_US=500",
            ]),
            vec![
                "-m",
                "hf:org/name/model.gguf",
                "--port",
                "8201",
                "--opt",
                "SPIN_US=500"
            ]
        );
    }

    #[test]
    fn forwarded_args_work_for_the_server_binary_without_a_subcommand() {
        assert_eq!(
            fwd(&["-m", "model.gguf", "--port", "9000"]),
            vec!["-m", "model.gguf", "--port", "9000"]
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_running_binary_is_recognised_as_unchanged() {
        let (_, unchanged) = running_exe().expect("the test binary has a path");
        assert!(unchanged, "the on-disk path must name the running inode");
    }

    #[test]
    fn a_state_file_round_trips_and_is_removed_only_by_its_owner() {
        let dir = std::env::temp_dir().join(format!("pg-lifecycle-{}", std::process::id()));
        let path = dir.join("serve.json");
        let state = State {
            pid: 42,
            version: "0.1.0".into(),
            host: "127.0.0.1".into(),
            port: 8201,
            model: "m.gguf".into(),
            log: Some(PathBuf::from("/tmp/log")),
            args: vec!["-m".into(), "m.gguf".into()],
            started: 1,
        };
        write(&path, &state).unwrap();
        assert_eq!(read(&path), Some(state.clone()));
        remove_if_ours(&path, 7);
        assert!(read(&path).is_some(), "another pid must not remove it");
        remove_if_ours(&path, 42);
        assert!(read(&path).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_wildcard_bind_is_reported_as_localhost() {
        let mut s = State {
            pid: 1,
            version: "0".into(),
            host: "0.0.0.0".into(),
            port: 8201,
            model: String::new(),
            log: None,
            args: Vec::new(),
            started: 0,
        };
        assert_eq!(s.url(), "http://127.0.0.1:8201/");
        s.host = "127.0.0.1".into();
        assert_eq!(s.url(), "http://127.0.0.1:8201/");
    }
}
