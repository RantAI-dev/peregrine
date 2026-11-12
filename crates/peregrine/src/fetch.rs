//! Model references: a local path, or `hf:<namespace>/<repo>/<file>[@<rev>]` downloaded once
//! into a cache.
//!
//! A hub reference is fetched with `curl` (the only external tool the CLI needs) into
//! `$PEREGRINE_CACHE_DIR`, `$XDG_CACHE_HOME/peregrine` or `~/.cache/peregrine`, under
//! `models/<repo>/<file>` (the repository is part of the key, so two repos can publish the same
//! file name without colliding).
//! The download is resumable: it starts with the GGUF header, checks that this build can run the
//! file *before* the rest is fetched, and leaves a `.part` file that a later run resumes rather
//! than a truncated model a later run would try to load.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

/// How much of the file the header check asks for. A GGUF header is the metadata plus the tensor
/// directory; even a heavily split 30B file needs a few hundred KiB, so 16 MiB leaves a wide
/// margin while keeping a refused download small.
const PROBE_BYTES: u64 = 16 * 1024 * 1024;

/// A `-m` value: a local file, or a Hugging Face hub reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Spec {
    Local(PathBuf),
    Hub {
        repo: String,
        file: String,
        rev: String,
    },
}

impl Spec {
    /// Parse `value`: anything starting with `hf:` is a hub reference, everything else is a path.
    ///
    /// `hf:org/name/file.gguf` uses the `main` revision; `hf:org/name/file.gguf@v2` pins one. The
    /// repository is the first two path segments (`namespace/name`, or one for a legacy
    /// single-segment repo), and any remaining segments are the file's path inside it.
    pub fn parse(value: &str) -> Result<Spec, String> {
        let Some(rest) = value.strip_prefix("hf:") else {
            return Ok(Spec::Local(PathBuf::from(value)));
        };
        if rest.is_empty() {
            return Err("hf: needs a repository and a file, e.g. hf:org/model.gguf".into());
        }
        let (rest, rev) = match rest.rsplit_once('@') {
            Some((r, v)) if !v.is_empty() && !v.contains('/') => (r, v),
            Some((_, "")) => {
                return Err(format!(
                    "hf:{rest}: the revision after '@' is empty; use hf:<namespace>/<repo>/<file>[@<rev>]"
                ));
            }
            _ => (rest, "main"),
        };
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() < 2 {
            return Err(format!(
                "hf:{rest}: expected hf:<namespace>/<repo>/<file>, e.g. hf:org/name/model.gguf"
            ));
        }
        // Hugging Face repository ids are `namespace/name`; a file may sit in a subdirectory
        // (`hf:org/name/quant/model.gguf`). With exactly two segments the whole string is the
        // old single-segment `repo/file` form.
        let repo_len = if parts.len() == 2 {
            parts[0].len()
        } else {
            parts[0].len() + 1 + parts[1].len()
        };
        let repo = &rest[..repo_len];
        let file = &rest[repo_len + 1..];
        if repo.is_empty()
            || file.is_empty()
            || repo.starts_with('.')
            || file.ends_with('/')
            || parts
                .iter()
                .any(|s| s.is_empty() || *s == "." || *s == "..")
        {
            return Err(format!(
                "hf:{rest}: expected a repository and a file name, e.g. hf:org/name/model.gguf"
            ));
        }
        Ok(Spec::Hub {
            repo: repo.to_string(),
            file: file.to_string(),
            rev: rev.to_string(),
        })
    }

    /// The HTTPS URL the file is downloaded from, for a hub reference.
    pub fn url(&self) -> Option<String> {
        match self {
            Spec::Local(_) => None,
            Spec::Hub { repo, file, rev } => Some(format!(
                "{}/{repo}/resolve/{rev}/{file}",
                hub_base().trim_end_matches('/')
            )),
        }
    }

    /// The file name used inside the cache directory.
    pub fn file_name(&self) -> Option<&str> {
        match self {
            Spec::Local(p) => p.file_name().and_then(|n| n.to_str()),
            Spec::Hub { file, .. } => file.rsplit('/').next(),
        }
    }

    /// Where a hub reference is cached: `models/<repo>/<file>`, keyed by the pinned revision when
    /// it is not `main`.
    ///
    /// The repository segment matters: many GGUF repos publish `model-Q4_K_M.gguf` or
    /// `ggml-model-Q4_K_M.gguf`, so a flat `models/<file>` would let one repo silently serve
    /// another repo's file. `None` for a local path, which is already on this machine.
    pub fn cache_path(&self) -> Option<PathBuf> {
        let Spec::Hub { repo, file, rev } = self else {
            return None;
        };
        let name = file.rsplit('/').next()?;
        let key = if rev == "main" {
            repo.clone()
        } else {
            format!("{repo}@{rev}")
        };
        Some(cache_dir().join("models").join(key).join(name))
    }
}

/// The hub that `hf:` references resolve against: `PEREGRINE_HF_BASE` for a mirror, else
/// Hugging Face.
fn hub_base() -> String {
    std::env::var("PEREGRINE_HF_BASE")
        .ok()
        .filter(|b| !b.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".into())
}

/// Where downloaded models are kept.
///
/// `$PEREGRINE_CACHE_DIR`, else `$XDG_CACHE_HOME/peregrine`, else `%LOCALAPPDATA%\peregrine` on
/// Windows, else `~/.cache/peregrine` (S7.2).
pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("PEREGRINE_CACHE_DIR") {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
        if !d.is_empty() {
            return PathBuf::from(d).join("peregrine");
        }
    }
    #[cfg(windows)]
    {
        // There is no `$XDG_CACHE_HOME` on Windows and `HOME` is usually unset: the per-user
        // local application data directory is the platform's answer (`%LOCALAPPDATA%`).
        if let Ok(d) = std::env::var("LOCALAPPDATA") {
            if !d.is_empty() {
                return PathBuf::from(d).join("peregrine");
            }
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("peregrine")
}

/// Turn a `-m` value into a path on this machine, downloading a hub reference if it is not cached.
pub fn resolve(value: impl AsRef<Path>) -> Result<PathBuf, String> {
    match Spec::parse(&value.as_ref().to_string_lossy())? {
        Spec::Local(p) => Ok(p),
        spec @ Spec::Hub { .. } => fetch(&spec),
    }
}

/// Download a hub reference into the cache and return the cached path. A cached file is used as
/// it is; delete it to fetch it again.
pub fn fetch(spec: &Spec) -> Result<PathBuf, String> {
    if let Spec::Local(p) = spec {
        return Ok(p.clone());
    }
    let url = spec.url().expect("every hub reference has a url");
    let name = spec
        .file_name()
        .ok_or_else(|| format!("hf:{url}: cannot derive a file name"))?;
    let final_path = spec
        .cache_path()
        .ok_or_else(|| format!("hf:{url}: cannot derive a cache path"))?;
    let dir = final_path
        .parent()
        .map_or_else(|| cache_dir().join("models"), Path::to_path_buf);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let part = final_path.with_file_name(format!("{name}.part"));

    if let Ok(m) = std::fs::metadata(&final_path) {
        if m.len() > 0 {
            return Ok(final_path);
        }
    }
    require_curl()?;

    let total = head_len(&url);
    let size = total.map_or_else(String::new, |n| format!(", {:.1} MiB", mib(n)));

    // A previous run's `.part` is resumed, not restarted: the header probe only runs when there
    // is no partial file yet.
    let existing = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let probed = existing == 0;
    if probed {
        if total.is_none_or(|t| t > PROBE_BYTES) {
            eprintln!("peregrine: fetching {name}{size} (header first)");
        }
        let end = PROBE_BYTES.saturating_sub(1);
        curl(&[
            "-fsSL".into(),
            "-r".into(),
            format!("0-{end}"),
            "-o".into(),
            part.to_string_lossy().into_owned(),
            url.clone(),
        ])?;
    }

    let part_len = std::fs::metadata(&part)
        .map_err(|e| format!("cannot read {}: {e}", part.display()))?
        .len();
    // Complete already: the server said so, the file ended before the probe range did, or a
    // resumed part loads as a whole file.
    let complete = total.is_some_and(|t| part_len >= t)
        || (probed && total.is_none() && part_len < PROBE_BYTES)
        || (total.is_none() && peregrine_core::gguf::Gguf::open_plain(&part).is_ok());
    if complete {
        if part_len == 0 {
            return Err(format!("{name}: the download was empty"));
        }
        // The pre-flight runs before the command line sets the process options, so it uses the
        // option-free reader and the metadata-only compatibility check. Reading the options here
        // would latch them and make `options::init` fail afterwards (the first-download bug).
        let g = peregrine_core::gguf::Gguf::open_plain(&part)
            .map_err(|e| format!("{name}: downloaded file is not a usable GGUF: {e}"))?;
        peregrine_models::check_compatible_metadata(&g)
            .map_err(|e| format!("{name} is not runnable by this build: {e}"))?;
        // The mapping must go before the rename: Windows refuses to rename a file that is
        // mapped, in this process or another (S7.2).
        drop(g);
        std::fs::rename(&part, &final_path)
            .map_err(|e| format!("cannot finish {}: {e}", final_path.display()))?;
        return Ok(final_path);
    }

    // The header decides before the weights arrive. A header that does not parse yet is not a
    // verdict: the probe may be smaller than a large tensor directory, so the download continues.
    match peregrine_core::gguf::Gguf::open_header_plain(&part) {
        Ok(g) => match peregrine_models::check_compatible_metadata(&g) {
            Ok(()) => {}
            Err(e) => {
                let _ = std::fs::remove_file(&part);
                return Err(format!("{name} is not runnable by this build: {e}"));
            }
        },
        Err(e) => {
            if total.is_some_and(|t| part_len >= t) {
                let _ = std::fs::remove_file(&part);
                return Err(format!("{name} is not a usable GGUF file: {e}"));
            }
            eprintln!("peregrine: header check skipped ({e})");
        }
    }

    eprintln!(
        "peregrine: downloading {name}{size} to {}",
        final_path.display()
    );
    let mut args = vec!["-fL".to_string(), "-C".to_string(), "-".to_string()];
    if std::io::stderr().is_terminal() {
        args.push("--progress-bar".into());
    } else {
        args.push("-sS".into());
    }
    args.push("-o".into());
    args.push(part.to_string_lossy().into_owned());
    args.push(url.clone());
    curl(&args)?;

    let len = std::fs::metadata(&part)
        .map_err(|e| format!("cannot read {}: {e}", part.display()))?
        .len();
    if let Some(t) = total {
        if len < t {
            return Err(format!(
                "{name}: download stopped at {len} of {t} bytes; run again to resume"
            ));
        }
    }
    if len == 0 {
        return Err(format!("{name}: the download was empty"));
    }
    // The complete file must also pass the check the partial one skipped.
    let g = peregrine_core::gguf::Gguf::open_plain(&part)
        .map_err(|e| format!("{name}: downloaded file is not a usable GGUF: {e}"))?;
    peregrine_models::check_compatible_metadata(&g)
        .map_err(|e| format!("{name} is not runnable by this build: {e}"))?;
    // The mapping must go before the rename; Windows cannot rename a mapped file (S7.2).
    drop(g);
    std::fs::rename(&part, &final_path)
        .map_err(|e| format!("cannot finish {}: {e}", final_path.display()))?;
    Ok(final_path)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

fn require_curl() -> Result<(), String> {
    match std::process::Command::new("curl").arg("--version").output() {
        Ok(_) => Ok(()),
        Err(_) => Err(
            "curl is required to fetch hf: models (install it, or download the .gguf and pass its path)"
                .into(),
        ),
    }
}

/// The file's size from a `HEAD` request, following redirects. `None` when the server does not say.
fn head_len(url: &str) -> Option<u64> {
    let mut cmd = std::process::Command::new("curl");
    add_token(&mut cmd);
    let out = cmd.args(["-fsSLI", url]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // Redirects append another header block: the last `content-length` is the file's.
    text.lines().rev().find_map(|line| {
        let (k, v) = line.split_once(':')?;
        k.eq_ignore_ascii_case("content-length")
            .then(|| v.trim().parse().ok())
            .flatten()
    })
}

/// Send `HF_TOKEN` as a bearer token, for gated or private repositories. Absent or empty, no
/// header is sent, so public downloads are anonymous.
fn add_token(cmd: &mut std::process::Command) {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            cmd.args(["-H", &format!("Authorization: Bearer {token}")]);
        }
    }
}

fn curl(args: &[String]) -> Result<(), String> {
    let mut cmd = std::process::Command::new("curl");
    add_token(&mut cmd);
    let out = cmd
        .args(args)
        .output()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    Err(format!(
        "curl {} failed ({}): {}",
        args.join(" "),
        out.status.code().map_or("signal".into(), |c| c.to_string()),
        err.trim()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_local_path_is_not_a_hub_reference() {
        assert_eq!(
            Spec::parse("./models/x.gguf").unwrap(),
            Spec::Local(PathBuf::from("./models/x.gguf"))
        );
        assert_eq!(Spec::parse("x.gguf").unwrap().url(), None);
    }

    #[test]
    fn hub_references_carry_repo_file_and_revision() {
        let s = Spec::parse("hf:org/model-Q4_K_M.gguf").unwrap();
        assert_eq!(
            s,
            Spec::Hub {
                repo: "org".into(),
                file: "model-Q4_K_M.gguf".into(),
                rev: "main".into()
            }
        );
        assert_eq!(
            s.url().unwrap(),
            "https://huggingface.co/org/resolve/main/model-Q4_K_M.gguf"
        );
        assert_eq!(s.file_name(), Some("model-Q4_K_M.gguf"));

        // A namespace/name repository: the first two segments are the repo.
        let namespaced = Spec::parse("hf:org/name/model.gguf").unwrap();
        assert_eq!(
            namespaced.url().unwrap(),
            "https://huggingface.co/org/name/resolve/main/model.gguf"
        );

        // A file in a repository subdirectory keeps its path; the cache stores the file name.
        let sub = Spec::parse("hf:ggml-org/models/tinyllamas/stories260K.gguf@v2").unwrap();
        assert_eq!(
            sub.url().unwrap(),
            "https://huggingface.co/ggml-org/models/resolve/v2/tinyllamas/stories260K.gguf"
        );
        assert_eq!(sub.file_name(), Some("stories260K.gguf"));
    }

    #[test]
    fn hub_references_reject_escapes_and_missing_parts() {
        for bad in [
            "hf:",
            "hf:org",
            "hf:/model.gguf",
            "hf:org/",
            "hf:org/../escape.gguf",
            "hf:org/sub/../../escape.gguf",
            "hf:../repo/model.gguf",
            "hf:org/model.gguf@",
        ] {
            assert!(Spec::parse(bad).is_err(), "{bad} was accepted");
        }
    }

    #[test]
    fn hub_files_are_cached_under_their_repository() {
        let one = Spec::parse("hf:org/one/model-Q4_K_M.gguf").unwrap();
        let two = Spec::parse("hf:org/two/model-Q4_K_M.gguf").unwrap();
        let p1 = one.cache_path().unwrap();
        let p2 = two.cache_path().unwrap();
        assert_ne!(
            p1, p2,
            "the same file name in two repositories must not share a cache path"
        );
        let models = cache_dir().join("models");
        assert_eq!(p1, models.join("org").join("one").join("model-Q4_K_M.gguf"));
        assert_eq!(p2, models.join("org").join("two").join("model-Q4_K_M.gguf"));
        assert_eq!(
            one.cache_path().unwrap().parent().unwrap(),
            models.join("org").join("one")
        );

        // A pinned revision is a separate key, so `main` moving cannot serve the old bytes.
        let pinned = Spec::parse("hf:org/one/model-Q4_K_M.gguf@v2").unwrap();
        assert_eq!(
            pinned.cache_path().unwrap(),
            models.join("org").join("one@v2").join("model-Q4_K_M.gguf")
        );

        // A file in a repository subdirectory keeps its base name under the repository key.
        let sub = Spec::parse("hf:ggml-org/models/tinyllamas/stories260K.gguf@v2").unwrap();
        assert_eq!(
            sub.cache_path().unwrap(),
            models
                .join("ggml-org")
                .join("models@v2")
                .join("stories260K.gguf")
        );

        assert_eq!(Spec::Local(PathBuf::from("x.gguf")).cache_path(), None);
    }

    #[test]
    fn the_cache_dir_can_be_redirected() {
        // `cache_dir` reads the process environment, so this test does not set it; it only pins
        // the suffix the downloader appends.
        assert_eq!(cache_dir().file_name().unwrap(), "peregrine");
    }
}
