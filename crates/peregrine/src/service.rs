//! The managed server lifecycle: background start, status, stop and zero-downtime `upgrade`.
//!
//! `serve --background` re-executes this binary with `SO_REUSEPORT` and a lifecycle state file.
//! `upgrade` installs the new release and starts it *beside* the old server; when the new process
//! reports ready, the old one is sent `SIGTERM` and drains. Both bind the same port while the
//! handoff happens, so a client's connection is never refused.

use crate::server::lifecycle::{self, State};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// How long a new server gets to load its model and report ready.
const START_TIMEOUT: Duration = Duration::from_secs(300);
/// How long a stopped server gets to drain its in-flight requests.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(40);

/// Run `serve --background`: start the detached child, wait until it is listening, and return its
/// state. The parent exits; the child keeps running in its own process group (Unix) or detached
/// from the console (`CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS` on Windows, S7.3).
pub fn start_background(
    port: u16,
    log: Option<PathBuf>,
    timeout: Duration,
) -> Result<State, String> {
    let state_path = lifecycle::state_path();
    if let Some(old) = lifecycle::read(&state_path) {
        if lifecycle::alive(old.pid) {
            return Err(format!(
                "a managed server is already running (pid {}, {}); use `peregrine status`, \
                 `peregrine stop` or `peregrine upgrade`",
                old.pid,
                old.url()
            ));
        }
        let _ = std::fs::remove_file(&state_path);
    }
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate this binary: {e}"))?;
    let log = log.unwrap_or_else(|| lifecycle::log_path(port));
    spawn(
        &exe,
        lifecycle::forwarded_args(std::env::args().skip(1)),
        &log,
        timeout,
    )
}

/// `peregrine status`: the managed server, or why there is none.
pub fn status() -> Result<String, String> {
    let path = lifecycle::state_path();
    let Some(state) = lifecycle::read(&path) else {
        return Err(format!("no managed server ({})", path.display()));
    };
    if !lifecycle::alive(state.pid) {
        return Err(format!(
            "stale state (pid {} is not running): {}",
            state.pid,
            path.display()
        ));
    }
    let out = format!(
        "pid        {}\nversion    {}\nurl        {}\nmodel      {}\nlog        {}\nstarted    {}\nstate      {}\n",
        state.pid,
        state.version,
        state.url(),
        state.model,
        state
            .log
            .as_ref()
            .map_or_else(|| "-".to_string(), |p| p.display().to_string()),
        state.started,
        path.display()
    );
    // Liveness is the exit status; the health line says whether the process is answering yet
    // (it may still be loading a large model). Windows only: on Unix `status` has never probed
    // the port, and the job is to stay identical there.
    #[cfg(windows)]
    let out = format!("{out}health     {}\n", windows_health(&state));
    Ok(out)
}

/// Whether the server answers `/health`, for the `status` line. Informational: a process that is
/// alive but still loading its model is not an error.
#[cfg(windows)]
fn windows_health(state: &State) -> String {
    use std::io::{Read, Write};
    use std::net::{TcpStream, ToSocketAddrs};
    let host = match state.host.as_str() {
        "0.0.0.0" | "::" | "" => "127.0.0.1",
        h => h,
    };
    let attempt = || -> Option<bool> {
        let addr = (host, state.port).to_socket_addrs().ok()?.next()?;
        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).ok()?;
        stream.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            .ok()?;
        let mut out = String::new();
        stream.read_to_string(&mut out).ok()?;
        Some(out.starts_with("HTTP/1.1 200") || out.starts_with("HTTP/1.0 200"))
    };
    match attempt() {
        Some(true) => "ok".into(),
        Some(false) => "not ok".into(),
        None => "not reachable".into(),
    }
}

/// `peregrine stop` on Windows: create the stop file the server watches, wait for the drain, and
/// end the process hard only when the drain timeout passes. `SIGTERM` is the Unix path below.
#[cfg(windows)]
pub fn stop() -> Result<String, String> {
    let path = lifecycle::state_path();
    let Some(state) = lifecycle::read(&path) else {
        return Ok("no managed server\n".into());
    };
    if !lifecycle::alive(state.pid) {
        let _ = std::fs::remove_file(lifecycle::stop_path(state.port));
        let _ = std::fs::remove_file(&path);
        return Ok(format!("pid {} was not running\n", state.pid));
    }
    stop_managed(&state, DRAIN_TIMEOUT)?;
    lifecycle::remove_if_ours(&path, state.pid);
    Ok(format!("stopped pid {}\n", state.pid))
}

/// Ask a Windows server to drain and wait for it to exit, ending the process only after
/// `timeout`. The server removes the stop file as it starts the drain.
#[cfg(windows)]
fn stop_managed(state: &State, timeout: Duration) -> Result<(), String> {
    let stop = lifecycle::stop_path(state.port);
    lifecycle::request_stop(state.port)?;
    match wait_gone(state.pid, timeout) {
        Ok(()) => {
            let _ = std::fs::remove_file(&stop);
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&stop);
            lifecycle::terminate(state.pid).map_err(|e2| format!("{e}; then {e2}"))?;
            wait_gone(state.pid, Duration::from_secs(10))
        }
    }
}

/// `peregrine stop`: ask the managed server to drain and exit.
#[cfg(not(windows))]
pub fn stop() -> Result<String, String> {
    let path = lifecycle::state_path();
    let Some(state) = lifecycle::read(&path) else {
        return Ok("no managed server\n".into());
    };
    if !lifecycle::alive(state.pid) {
        let _ = std::fs::remove_file(&path);
        return Ok(format!("pid {} was not running\n", state.pid));
    }
    lifecycle::terminate(state.pid)?;
    wait_gone(state.pid, DRAIN_TIMEOUT)?;
    let _ = std::fs::remove_file(&path);
    Ok(format!("stopped pid {}\n", state.pid))
}

/// `peregrine upgrade`: install a release and, when a managed server is running, replace it
/// without refusing a connection.
pub struct UpgradeArgs {
    pub check: bool,
    pub version: Option<String>,
    pub base_url: Option<String>,
    pub exe: Option<PathBuf>,
    pub no_restart: bool,
    pub force: bool,
}

/// The release tag currently running.
pub fn current_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub fn upgrade(args: &UpgradeArgs) -> Result<String, String> {
    let platform =
        release_platform(std::env::consts::OS, std::env::consts::ARCH).ok_or_else(|| {
            format!(
                "no release archive for {} on {}; build from source: cargo build --release",
                std::env::consts::OS,
                std::env::consts::ARCH
            )
        })?;
    let explicit = args
        .base_url
        .clone()
        .or_else(|| std::env::var("PEREGRINE_RELEASE_BASE").ok())
        .filter(|b| !b.is_empty());
    let tag = match &args.version {
        Some(t) => t.clone(),
        None => latest_tag(explicit.as_deref())?,
    };

    if args.check {
        return Ok(format!(
            "running    {}\nnewest     {}\nupdate     {}\n",
            current_version(),
            tag,
            if newer(&tag, current_version()).unwrap_or(false) {
                "available"
            } else {
                "not needed"
            }
        ));
    }

    let target = match &args.exe {
        Some(p) => p.clone(),
        None => std::env::current_exe()
            .map_err(|e| format!("cannot locate this binary: {e}"))?
            .canonicalize()
            .map_err(|e| format!("cannot resolve this binary: {e}"))?,
    };
    if !args.force
        && target
            .to_string_lossy()
            .replace('\\', "/")
            .contains("/target/")
    {
        return Err(format!(
            "{} is a cargo build; `cargo build --release` it, or pass --exe/--force",
            target.display()
        ));
    }
    let asset = format!("peregrine-{tag}-{platform}.{}", archive_extension(platform));
    let tmp = std::env::temp_dir().join(format!("peregrine-upgrade-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
    // The vanity host first, GitHub as the fallback: the public download URL is branded, while a
    // mirror that is down or not yet configured cannot break an upgrade.
    let mut failure = None;
    let mut installed = false;
    for (i, base) in release_bases(explicit).iter().enumerate() {
        let attempt = tmp.join(format!("try-{i}"));
        std::fs::create_dir_all(&attempt)
            .map_err(|e| format!("cannot create {}: {e}", attempt.display()))?;
        match install(base, &tag, platform, &asset, &attempt, &target) {
            Ok(()) => {
                installed = true;
                break;
            }
            Err(e) => failure = Some(format!("{base}: {e}")),
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    if !installed {
        return Err(failure.unwrap_or_else(|| "the release could not be installed".into()));
    }

    // Replace a running managed server while it keeps accepting connections.
    let state = lifecycle::read(&lifecycle::state_path()).filter(|s| lifecycle::alive(s.pid));
    let Some(old) = state else {
        return Ok(format!(
            "installed {} ({}); no managed server is running, so nothing was restarted\n",
            tag,
            target.display()
        ));
    };
    if args.no_restart {
        return Ok(format!(
            "installed {} ({}); pid {} still serves the old binary until `peregrine stop` and \
             `peregrine serve --background`\n",
            tag,
            target.display(),
            old.pid
        ));
    }
    let log = old
        .log
        .clone()
        .unwrap_or_else(|| lifecycle::log_path(old.port));
    // Windows cannot bind the port beside the old server (`SO_REUSEPORT` is Unix-only) and a
    // running `.exe` cannot be replaced while it serves, so an upgrade drains the old server
    // first and starts the new binary after it: a short gap, and the command says so. Linux and
    // macOS keep the zero-downtime handoff.
    #[cfg(windows)]
    {
        stop_managed(&old, DRAIN_TIMEOUT).map_err(|e| {
            format!(
                "installed {tag}, but pid {} did not stop ({e}); `peregrine stop` it and start \
                 the new binary",
                old.pid
            )
        })?;
        let new = spawn(&target, old.args.clone(), &log, START_TIMEOUT).map_err(|e| {
            format!(
                "installed {tag} and stopped pid {}, but the replacement did not start ({e}); \
                 start it with `peregrine serve --background`",
                old.pid
            )
        })?;
        Ok(format!(
            "upgraded to {tag}: pid {} -> {}\nWindows has no zero-downtime handoff: the old \
             server drained and the new one started after it\n",
            old.pid, new.pid
        ))
    }
    #[cfg(not(windows))]
    {
        let new = spawn(&target, old.args.clone(), &log, START_TIMEOUT).map_err(|e| {
            format!(
                "installed {tag}, but the replacement did not start ({e}); pid {} still serves the \
                 old binary from its mapped image",
                old.pid
            )
        })?;
        lifecycle::terminate(old.pid)?;
        wait_gone(old.pid, DRAIN_TIMEOUT).map_err(|e| {
            format!(
                "installed {tag}; pid {} is still draining ({e}), pid {} is serving",
                old.pid, new.pid
            )
        })?;
        Ok(format!(
            "upgraded to {tag}: pid {} -> {}\nno connection was refused: both servers shared {} during the handoff\n",
            old.pid,
            new.pid,
            old.url()
        ))
    }
}

/// Start one detached server and wait until it has written its own ready state.
fn spawn(exe: &Path, args: Vec<String>, log: &Path, timeout: Duration) -> Result<State, String> {
    let state = lifecycle::spawn_server(
        exe,
        &args,
        Some(log),
        true,
        &lifecycle::state_path(),
        timeout,
        None,
    )?;
    println!(
        "peregrine: listening on {} (pid {})",
        state.url(),
        state.pid
    );
    Ok(state)
}

fn wait_gone(pid: u32, timeout: Duration) -> Result<(), String> {
    let started = Instant::now();
    while lifecycle::alive(pid) {
        if started.elapsed() > timeout {
            return Err(format!(
                "pid {pid} is still running after {} s",
                timeout.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

/// The release asset's platform name for a system and architecture, e.g. `macos-aarch64`.
///
/// The names match the release archives' (`scripts/release/build.sh`, `scripts/release/build.ps1`).
/// Intel Macs and Windows on ARM have no archive: they build from source
/// (docs/design/s7-platforms.md §2).
fn release_platform(os: &str, arch: &str) -> Option<&'static str> {
    match (os, arch) {
        ("linux", "x86_64") => Some("linux-x86_64"),
        ("linux", "aarch64") => Some("linux-aarch64"),
        ("macos", "aarch64") => Some("macos-aarch64"),
        ("windows", "x86_64") => Some("windows-x86_64"),
        _ => None,
    }
}

/// The archive format a platform ships: `.zip` on Windows, `.tar.gz` elsewhere.
fn archive_extension(platform: &str) -> &'static str {
    if platform.starts_with("windows") {
        "zip"
    } else {
        "tar.gz"
    }
}

/// The public release host, which redirects to GitHub (the lakehouse pattern).
const VANITY: &str = "https://peregrine.rantai.dev";
/// The release archives, on GitHub.
const GITHUB_RELEASES: &str = "https://github.com/RantAI-dev/peregrine/releases/download";
/// The newest release, from the GitHub API.
const GITHUB_LATEST: &str = "https://api.github.com/repos/RantAI-dev/peregrine/releases/latest";

/// Where to look for archives: an explicit `--base-url`/`PEREGRINE_RELEASE_BASE` alone, else the
/// vanity host and then GitHub.
fn release_bases(explicit: Option<String>) -> Vec<String> {
    if let Some(base) = explicit {
        return vec![base];
    }
    vec![
        format!("{VANITY}/releases/download"),
        GITHUB_RELEASES.to_string(),
    ]
}

/// The newest release tag: `PEREGRINE_RELEASE_LATEST`, else the vanity host (when nothing was
/// overridden), else the GitHub API.
fn latest_tag(explicit: Option<&str>) -> Result<String, String> {
    let sources = match std::env::var("PEREGRINE_RELEASE_LATEST") {
        Ok(url) if !url.is_empty() => vec![url],
        _ if explicit.is_none() => vec![format!("{VANITY}/latest"), GITHUB_LATEST.into()],
        _ => vec![GITHUB_LATEST.into()],
    };
    let mut failure = None;
    for url in &sources {
        match tag_from(url) {
            Ok(tag) => return Ok(tag),
            Err(e) => failure = Some(format!("{url}: {e}")),
        }
    }
    Err(format!(
        "cannot read the newest release: {}",
        failure.unwrap_or_else(|| "no source".into())
    ))
}

fn tag_from(url: &str) -> Result<String, String> {
    let out = Command::new("curl")
        .args(["-fsSL", url])
        .output()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let key = text
        .find("\"tag_name\"")
        .ok_or("the release response has no tag_name")?;
    let rest = &text[key..];
    let start = rest.find(':').ok_or("the release response is malformed")?;
    let rest = &rest[start..];
    let start = rest.find('"').ok_or("the release response is malformed")? + 1;
    let rest = &rest[start..];
    let end = rest.find('"').ok_or("the release response is malformed")?;
    Ok(rest[..end].to_string())
}

/// Whether `tag` names a newer version than `current` (both `vX.Y.Z` or `X.Y.Z`).
fn newer(tag: &str, current: &str) -> Option<bool> {
    Some(parse_version(tag)? > parse_version(current)?)
}

fn parse_version(v: &str) -> Option<(u64, u64, u64)> {
    let v = v.trim().trim_start_matches('v');
    let core = v.split(['-', '+']).next()?;
    let mut parts = core.split('.');
    Some((
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next().unwrap_or("0").parse().ok()?,
    ))
}

/// Download the archive and its `SHA256SUMS`, verify, and install over `target` atomically.
fn install(
    base: &str,
    tag: &str,
    platform: &str,
    asset: &str,
    tmp: &Path,
    target: &Path,
) -> Result<(), String> {
    let dir = format!("{}/{tag}", base.trim_end_matches('/'));
    let archive = tmp.join(asset);
    let archive_s = archive.to_string_lossy().into_owned();
    curl(&["-o", &archive_s, &format!("{dir}/{asset}")])?;
    let sums = tmp.join("SHA256SUMS");
    let sums_s = sums.to_string_lossy().into_owned();
    curl(&["-o", &sums_s, &format!("{dir}/SHA256SUMS")])?;
    verify_checksum(&archive, asset, &sums)?;

    unpack(&archive, tmp)?;
    let binary = if cfg!(windows) {
        "peregrine.exe"
    } else {
        "peregrine"
    };
    let staged = tmp.join(format!("peregrine-{tag}-{platform}")).join(binary);
    if !staged.is_file() {
        return Err(format!("the archive has no {}", staged.display()));
    }
    verify(&staged)?;
    place(&staged, target)?;
    Ok(())
}

/// Extract `archive` into `dest` with the system's own tool: `tar` on Unix and PowerShell's
/// `Expand-Archive` on Windows (PowerShell 5.1 ships with every Windows). The paths go through
/// the environment so no quoting rule differs between the two.
fn unpack(archive: &Path, dest: &Path) -> Result<(), String> {
    #[cfg(windows)]
    let mut command = {
        let mut command = Command::new("powershell");
        command.args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Expand-Archive -LiteralPath $env:PEREGRINE_UNZIP_FROM -DestinationPath $env:PEREGRINE_UNZIP_TO -Force",
        ]);
        command
    };
    #[cfg(not(windows))]
    let mut command = {
        let mut command = Command::new("tar");
        command.args([
            "-xzf",
            &archive.to_string_lossy(),
            "-C",
            &dest.to_string_lossy(),
        ]);
        command
    };
    #[cfg(windows)]
    {
        command.env("PEREGRINE_UNZIP_FROM", archive);
        command.env("PEREGRINE_UNZIP_TO", dest);
    }
    let status = command
        .status()
        .map_err(|e| format!("cannot extract {}: {e}", archive.display()))?;
    if !status.success() {
        return Err(format!("the archive {} did not extract", archive.display()));
    }
    Ok(())
}

/// Move the staged binary over `target`.
///
/// Unix replaces it with an atomic rename (a running server keeps its old inode). Windows cannot
/// replace a running `.exe`, but it can rename one: the current binary becomes `<name>.old` and
/// the new one takes its place; [`clean_old_binaries`] removes the leftover on the next start.
fn place(staged: &Path, target: &Path) -> Result<(), String> {
    // Copy beside the target first, so a failed copy never disturbs the installed binary.
    let beside = target.with_file_name(format!(".peregrine-{}", std::process::id()));
    std::fs::copy(staged, &beside)
        .map_err(|e| format!("cannot write {}: {e}", beside.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&beside, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("cannot chmod {}: {e}", beside.display()))?;
    }
    #[cfg(windows)]
    {
        if target.exists() {
            let mut old = old_binary_path(target);
            // A previous `.old` may still be running and cannot be removed yet: give this one a
            // unique name instead of failing the upgrade.
            if old.exists() && std::fs::remove_file(&old).is_err() {
                let name = target
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "peregrine.exe".into());
                old = target.with_file_name(format!("{name}.{}.old", std::process::id()));
            }
            std::fs::rename(target, &old).map_err(|e| {
                format!(
                    "cannot move {} aside for the upgrade: {e}",
                    target.display()
                )
            })?;
        }
    }
    std::fs::rename(&beside, target)
        .map_err(|e| format!("cannot replace {}: {e}", target.display()))
}

/// `<name>.old`, the name `place` gives the replaced binary on Windows.
#[cfg(windows)]
fn old_binary_path(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "peregrine.exe".into());
    target.with_file_name(format!("{name}.old"))
}

/// Remove the `<name>.old` copies a previous Windows upgrade left behind. Called at every start;
/// a copy whose process still runs cannot be removed and is retried next time.
#[cfg(windows)]
pub fn clean_old_binaries() {
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let (Some(dir), Some(name)) = (
        exe.parent(),
        exe.file_name().map(|n| n.to_string_lossy().into_owned()),
    ) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let file = entry.file_name();
        let file = file.to_string_lossy();
        if file.starts_with(name.as_str()) && file.ends_with(".old") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn curl(args: &[&str]) -> Result<(), String> {
    let out = Command::new("curl")
        .arg("-fsSL")
        .args(args)
        .output()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "download failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

fn verify_checksum(archive: &Path, asset: &str, sums: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(sums).map_err(|e| format!("cannot read checksums: {e}"))?;
    let expected = text.lines().find_map(|line| {
        let mut it = line.split_whitespace();
        let hash = it.next()?;
        let name = it.next()?.trim_start_matches('*').rsplit('/').next()?;
        (name == asset).then(|| hash.to_string())
    });
    let expected = expected.ok_or_else(|| format!("SHA256SUMS has no entry for {asset}"))?;
    let actual = sha256_hex(archive)?;
    if actual != expected {
        return Err(format!(
            "checksum mismatch for {asset}: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

/// The SHA-256 of a file as lower-case hex, with the system's own tool: `certutil` on Windows,
/// `sha256sum` on Linux, `shasum -a 256` on macOS.
fn sha256_hex(path: &Path) -> Result<String, String> {
    #[cfg(windows)]
    {
        let out = Command::new("certutil")
            .args(["-hashfile", &path.to_string_lossy(), "SHA256"])
            .output()
            .map_err(|e| format!("cannot run certutil: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "certutil failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        // certutil prints a header, the hash (possibly grouped in bytes), and a footer.
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            let hex: String = line.chars().filter(|c| !c.is_whitespace()).collect();
            if hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Ok(hex.to_ascii_lowercase());
            }
        }
        Err(format!(
            "certutil printed no SHA-256 for {}",
            path.display()
        ))
    }
    #[cfg(not(windows))]
    {
        let (program, args): (&str, &[&str]) = if command_exists("sha256sum") {
            ("sha256sum", &[])
        } else if command_exists("shasum") {
            ("shasum", &["-a", "256"])
        } else {
            return Err("neither sha256sum nor shasum is installed".into());
        };
        let out = Command::new(program)
            .args(args)
            .arg(path)
            .output()
            .map_err(|e| format!("cannot run {program}: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "{program} failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase())
    }
}

#[cfg(not(windows))]
fn command_exists(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn verify(binary: &Path) -> Result<(), String> {
    let out = Command::new(binary)
        .arg("--version")
        .output()
        .map_err(|e| format!("cannot run {}: {e}", binary.display()))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{} did not run after installation",
            binary.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_with_or_without_a_v_prefix() {
        assert_eq!(newer("v0.2.0", "0.1.0"), Some(true));
        assert_eq!(newer("v0.1.0", "0.1.0"), Some(false));
        assert_eq!(newer("0.1.0", "v0.1.1"), Some(false));
        assert_eq!(newer("v1.0.0-rc1", "0.9.9"), Some(true));
        assert_eq!(newer("nonsense", "0.1.0"), None);
    }

    #[test]
    fn releases_prefer_the_vanity_host_and_fall_back_to_github() {
        let bases = release_bases(None);
        assert_eq!(bases[0], "https://peregrine.rantai.dev/releases/download");
        assert_eq!(
            bases[1],
            "https://github.com/RantAI-dev/peregrine/releases/download"
        );
        assert_eq!(
            release_bases(Some("file:///mirror".into())),
            vec!["file:///mirror"]
        );
    }

    #[test]
    fn the_release_platform_names_the_system_and_architecture() {
        // The mapping is a function of the system and architecture, not of the machine this test
        // runs on, so every case is checked by argument. The four names are the four archives the
        // release workflow builds; the two `None`s are the platforms that build from source.
        let cases = [
            (("linux", "x86_64"), Some("linux-x86_64")),
            (("linux", "aarch64"), Some("linux-aarch64")),
            (("macos", "aarch64"), Some("macos-aarch64")),
            (("windows", "x86_64"), Some("windows-x86_64")),
            // Intel macOS and Windows on ARM have no release archive.
            (("macos", "x86_64"), None),
            (("windows", "aarch64"), None),
        ];
        for ((os, arch), expected) in cases {
            assert_eq!(release_platform(os, arch), expected, "{os} on {arch}");
        }
    }

    #[test]
    fn the_archive_name_matches_the_platform() {
        // The tag is a stand-in, never a released version: the name is what the mapping plus the
        // extension produces.
        let tag = "vX.Y.Z";
        let cases = [
            ("linux", "x86_64", "peregrine-vX.Y.Z-linux-x86_64.tar.gz"),
            ("linux", "aarch64", "peregrine-vX.Y.Z-linux-aarch64.tar.gz"),
            ("macos", "aarch64", "peregrine-vX.Y.Z-macos-aarch64.tar.gz"),
            ("windows", "x86_64", "peregrine-vX.Y.Z-windows-x86_64.zip"),
        ];
        for (os, arch, expected) in cases {
            let platform = release_platform(os, arch).unwrap();
            assert_eq!(
                format!("peregrine-{tag}-{platform}.{}", archive_extension(platform)),
                expected
            );
        }
    }

    #[test]
    fn the_new_binary_replaces_the_old_one_in_place() {
        // Runs on every platform: on Windows the running binary's rename-swap is exercised with
        // ordinary files (nothing is running here), elsewhere the atomic rename.
        let dir = std::env::temp_dir().join(format!(
            "pg-upgrade-place-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join(if cfg!(windows) {
            "peregrine.exe"
        } else {
            "peregrine"
        });
        let staged = dir.join("staged");
        std::fs::write(&target, b"old binary").unwrap();
        std::fs::write(&staged, b"new binary").unwrap();

        place(&staged, &target).unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new binary");
        #[cfg(windows)]
        {
            let old = old_binary_path(&target);
            assert_eq!(
                std::fs::read(&old).unwrap(),
                b"old binary",
                "the replaced binary is kept as {} until the next start",
                old.display()
            );
        }
        #[cfg(not(windows))]
        {
            let leftovers: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().ends_with(".old"))
                .collect();
            assert!(
                leftovers.is_empty(),
                "Unix replaces with one rename and leaves no copy behind"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}
