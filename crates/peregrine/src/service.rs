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
/// state. The parent exits; the child keeps running in its own process group.
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
    Ok(format!(
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
    ))
}

/// `peregrine stop`: ask the managed server to drain and exit.
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
    if std::env::consts::OS != "linux" {
        return Err(format!(
            "release archives are Linux-only; on {} build from source",
            std::env::consts::OS
        ));
    }
    let arch = asset_arch()?;
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
    if !args.force && target.to_string_lossy().contains("/target/") {
        return Err(format!(
            "{} is a cargo build; `cargo build --release` it, or pass --exe/--force",
            target.display()
        ));
    }
    let asset = format!("peregrine-{tag}-linux-{arch}.tar.gz");
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
        match install(base, &tag, &asset, &attempt, &target) {
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

fn asset_arch() -> Result<&'static str, String> {
    match std::env::consts::ARCH {
        "x86_64" => Ok("x86_64"),
        "aarch64" => Ok("aarch64"),
        other => Err(format!("no release archive for {other}; build from source")),
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
fn install(base: &str, tag: &str, asset: &str, tmp: &Path, target: &Path) -> Result<(), String> {
    let dir = format!("{}/{tag}", base.trim_end_matches('/'));
    let archive = tmp.join(asset);
    let archive_s = archive.to_string_lossy().into_owned();
    curl(&["-o", &archive_s, &format!("{dir}/{asset}")])?;
    let sums = tmp.join("SHA256SUMS");
    let sums_s = sums.to_string_lossy().into_owned();
    curl(&["-o", &sums_s, &format!("{dir}/SHA256SUMS")])?;
    verify_checksum(&archive, asset, &sums)?;

    let status = Command::new("tar")
        .args([
            "-xzf",
            &archive.to_string_lossy(),
            "-C",
            &tmp.to_string_lossy(),
        ])
        .status()
        .map_err(|e| format!("cannot run tar: {e}"))?;
    if !status.success() {
        return Err("the archive did not extract".into());
    }
    let staged = tmp
        .join(format!("peregrine-{tag}-linux-{}", archive_arch(asset)?))
        .join("peregrine");
    if !staged.is_file() {
        return Err(format!("the archive has no {}", staged.display()));
    }
    verify(&staged)?;
    // Copy beside the target and rename: an atomic replace, never a half-written binary.
    let beside = target.with_file_name(format!(".peregrine-{}", std::process::id()));
    std::fs::copy(&staged, &beside)
        .map_err(|e| format!("cannot write {}: {e}", beside.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&beside, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("cannot chmod {}: {e}", beside.display()))?;
    }
    std::fs::rename(&beside, target)
        .map_err(|e| format!("cannot replace {}: {e}", target.display()))?;
    Ok(())
}

fn archive_arch(asset: &str) -> Result<&str, String> {
    if asset.contains("-linux-x86_64.") {
        Ok("x86_64")
    } else if asset.contains("-linux-aarch64.") {
        Ok("aarch64")
    } else {
        Err(format!("{asset}: cannot tell the architecture"))
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
    let out = Command::new("sha256sum")
        .arg(archive)
        .output()
        .map_err(|e| format!("cannot run sha256sum: {e}"))?;
    let actual = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string();
    if actual != expected {
        return Err(format!(
            "checksum mismatch for {asset}: expected {expected}, got {actual}"
        ));
    }
    Ok(())
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
    fn the_archive_names_its_architecture() {
        assert_eq!(
            archive_arch("peregrine-v0.1.0-linux-x86_64.tar.gz").unwrap(),
            "x86_64"
        );
        assert_eq!(
            archive_arch("peregrine-v0.1.0-linux-aarch64.tar.gz").unwrap(),
            "aarch64"
        );
        assert!(archive_arch("peregrine.tar.gz").is_err());
    }
}
