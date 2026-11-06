//! Model- and machine-specific tuning profiles (ADR 0005, roadmap S1.7).
//!
//! Tuning is data, not code: `peregrine tune` measures pool and kernel knobs on the host for a model
//! and writes a profile file. A profile records CPU model and features, Peregrine version, and the
//! model file's size and FNV-1a hash; any mismatch ignores it and falls back to safe defaults.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MIB: usize = 1 << 20;

/// FNV-1a 64-bit hash over a byte slice, as in `tests/oracle.rs`.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// FNV-1a 64 over the first and last MiB of the model file (or whole file if < 2 MiB), as `tests/oracle.rs` does.
pub fn compute_model_fnv(path: &Path) -> std::io::Result<u64> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let step = |mut acc: u64, buf: &[u8]| -> u64 {
        for &b in buf {
            acc ^= u64::from(b);
            acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
        }
        acc
    };

    if len < 2 * MIB as u64 {
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        h = step(h, &buf);
    } else {
        let mut first = vec![0u8; MIB];
        file.read_exact(&mut first)?;
        h = step(h, &first);
        file.seek(SeekFrom::End(-(MIB as i64)))?;
        let mut last = vec![0u8; MIB];
        file.read_exact(&mut last)?;
        h = step(h, &last);
    }
    Ok(h)
}

/// The CPU model name (e.g. from `/proc/cpuinfo` on Linux, `sysctl` on macOS, or arch name).
pub fn cpu_model_name() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in info.lines() {
                if let Some(rest) = line.strip_prefix("model name") {
                    if let Some((_, val)) = rest.split_once(':') {
                        let name = val.trim();
                        if !name.is_empty() {
                            return name.to_string();
                        }
                    }
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

/// The CPU feature string the tier detection uses (`crate::cpu::tier_name()`).
pub fn cpu_features_string() -> String {
    crate::cpu::tier_name()
}

/// A parsed tuning profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Profile {
    pub peregrine_version: String,
    pub cpu: String,
    pub cpu_features: String,
    pub model_size: u64,
    pub model_fnv: u64,
    pub created: u64,
    pub options: Vec<(String, String)>,
}

impl Profile {
    /// Create a new profile template for `model_path` on the current machine and version.
    pub fn new_for_model(model_path: &Path) -> Result<Self, String> {
        let meta = std::fs::metadata(model_path)
            .map_err(|e| format!("cannot stat model {}: {e}", model_path.display()))?;
        let model_size = meta.len();
        let model_fnv = compute_model_fnv(model_path)
            .map_err(|e| format!("cannot hash model {}: {e}", model_path.display()))?;
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Ok(Profile {
            peregrine_version: env!("CARGO_PKG_VERSION").to_string(),
            cpu: cpu_model_name(),
            cpu_features: cpu_features_string(),
            model_size,
            model_fnv,
            created,
            options: Vec::new(),
        })
    }

    /// Look up an option value by name (case-insensitive, as in `EngineOptions::pairs()`).
    pub fn get(&self, key: &str) -> Option<&str> {
        self.options
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    /// Set an option value.
    pub fn set(&mut self, key: impl Into<String>, val: impl Into<String>) {
        let key = key.into();
        let val = val.into();
        if let Some((_, v)) = self
            .options
            .iter_mut()
            .find(|(k, _)| k.eq_ignore_ascii_case(&key))
        {
            *v = val;
        } else {
            self.options.push((key, val));
        }
    }
}

/// Default profile path: `$XDG_CONFIG_HOME` (or `~/.config`)`/peregrine/profiles/<model file stem>-<model_fnv hex>.profile`.
pub fn default_path(model_path: &Path) -> Result<PathBuf, String> {
    let fnv = compute_model_fnv(model_path)
        .map_err(|e| format!("cannot hash model {}: {e}", model_path.display()))?;
    let stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let base = match std::env::var_os("XDG_CONFIG_HOME") {
        Some(val) if !val.is_empty() => PathBuf::from(val),
        _ => match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(".config"),
            None => PathBuf::from("."),
        },
    };

    Ok(base
        .join("peregrine")
        .join("profiles")
        .join(format!("{stem}-{fnv:016x}.profile")))
}

/// Write a profile to `path`.
pub fn write(path: &Path, profile: &Profile) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    writeln!(file, "# Peregrine tuning profile")?;
    writeln!(file, "peregrine_version = {}", profile.peregrine_version)?;
    writeln!(file, "cpu = {}", profile.cpu)?;
    writeln!(file, "cpu_features = {}", profile.cpu_features)?;
    writeln!(file, "model_size = {}", profile.model_size)?;
    writeln!(file, "model_fnv = {:016x}", profile.model_fnv)?;
    writeln!(file, "created = {}", profile.created)?;
    writeln!(file)?;
    writeln!(file, "# Tuned options")?;
    for (k, v) in &profile.options {
        writeln!(file, "{k} = {v}")?;
    }
    Ok(())
}

/// Read a profile from `path`.
pub fn read(path: &Path) -> Result<Profile, String> {
    let file =
        File::open(path).map_err(|e| format!("cannot open profile {}: {e}", path.display()))?;
    let reader = BufReader::new(file);

    let mut version = None;
    let mut cpu = None;
    let mut cpu_features = None;
    let mut model_size = None;
    let mut model_fnv = None;
    let mut created = None;
    let mut options = Vec::new();

    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|e| format!("{}:{}: {e}", path.display(), line_no + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, val)) = trimmed.split_once('=') else {
            return Err(format!(
                "{}:{}: expected key = value, got {:?}",
                path.display(),
                line_no + 1,
                trimmed
            ));
        };
        let key = key.trim();
        let val = val.trim();
        match key {
            "peregrine_version" => version = Some(val.to_string()),
            "cpu" => cpu = Some(val.to_string()),
            "cpu_features" => cpu_features = Some(val.to_string()),
            "model_size" => {
                model_size = Some(
                    val.parse::<u64>()
                        .map_err(|e| format!("invalid model_size {val:?}: {e}"))?,
                );
            }
            "model_fnv" => {
                let clean = val.strip_prefix("0x").unwrap_or(val);
                model_fnv = Some(
                    u64::from_str_radix(clean, 16)
                        .map_err(|e| format!("invalid model_fnv hex {val:?}: {e}"))?,
                );
            }
            "created" => {
                created = Some(
                    val.parse::<u64>()
                        .map_err(|e| format!("invalid created timestamp {val:?}: {e}"))?,
                );
            }
            other => {
                options.push((other.to_string(), val.to_string()));
            }
        }
    }

    Ok(Profile {
        peregrine_version: version.ok_or_else(|| "missing peregrine_version header".to_string())?,
        cpu: cpu.ok_or_else(|| "missing cpu header".to_string())?,
        cpu_features: cpu_features.ok_or_else(|| "missing cpu_features header".to_string())?,
        model_size: model_size.ok_or_else(|| "missing model_size header".to_string())?,
        model_fnv: model_fnv.ok_or_else(|| "missing model_fnv header".to_string())?,
        created: created.unwrap_or(0),
        options,
    })
}

/// Check whether `profile` matches the current engine version, host CPU, and `model_path`.
/// Returns `Ok(())` on match, or `Err(mismatch_reason)`.
pub fn check(profile: &Profile, model_path: &Path) -> Result<(), String> {
    let current_version = env!("CARGO_PKG_VERSION");
    if profile.peregrine_version != current_version {
        return Err(format!(
            "version mismatch: profile has {}, engine is {current_version}",
            profile.peregrine_version
        ));
    }

    let current_cpu = cpu_model_name();
    if profile.cpu != current_cpu {
        return Err(format!(
            "cpu mismatch: profile was created for {:?}, current cpu is {:?}",
            profile.cpu, current_cpu
        ));
    }

    let current_features = cpu_features_string();
    if profile.cpu_features != current_features {
        return Err(format!(
            "cpu features mismatch: profile was created for {:?}, current features are {:?}",
            profile.cpu_features, current_features
        ));
    }

    let meta = std::fs::metadata(model_path)
        .map_err(|e| format!("cannot stat model {}: {e}", model_path.display()))?;
    if meta.len() != profile.model_size {
        return Err(format!(
            "model size mismatch: profile expects {} bytes, file has {} bytes",
            profile.model_size,
            meta.len()
        ));
    }

    let current_fnv = compute_model_fnv(model_path)
        .map_err(|e| format!("cannot hash model {}: {e}", model_path.display()))?;
    if current_fnv != profile.model_fnv {
        return Err(format!(
            "model hash mismatch: profile expects {:016x}, file has {:016x}",
            profile.model_fnv, current_fnv
        ));
    }

    Ok(())
}

/// Resolve tuning profile for a model given optional explicit path or `--no-profile`.
///
/// Returns `(Option<Profile>, String)` where the string is the single line describing
/// which profile was applied or why it was ignored.
pub fn resolve(
    model_path: &Path,
    explicit_profile: Option<&Path>,
    no_profile: bool,
) -> (Option<Profile>, String) {
    if no_profile {
        return (None, "profile: none (--no-profile)".to_string());
    }

    if let Some(path) = explicit_profile {
        match read(path) {
            Ok(p) => match check(&p, model_path) {
                Ok(()) => (Some(p), format!("profile: applied {}", path.display())),
                Err(reason) => (
                    None,
                    format!("profile: ignored {}: {reason}", path.display()),
                ),
            },
            Err(e) => (None, format!("profile: ignored {}: {e}", path.display())),
        }
    } else {
        match default_path(model_path) {
            Ok(path) => {
                if path.exists() {
                    match read(&path) {
                        Ok(p) => match check(&p, model_path) {
                            Ok(()) => (Some(p), format!("profile: applied {}", path.display())),
                            Err(reason) => (
                                None,
                                format!("profile: ignored {}: {reason}", path.display()),
                            ),
                        },
                        Err(e) => (None, format!("profile: ignored {}: {e}", path.display())),
                    }
                } else {
                    (
                        None,
                        format!("profile: none (no profile at {})", path.display()),
                    )
                }
            }
            Err(e) => (None, format!("profile: none ({e})")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_file(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("peregrine-profile-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        d.join(name)
    }

    #[test]
    fn profile_round_trip() {
        let p = tmp_file("roundtrip.profile");
        let prof = Profile {
            peregrine_version: "0.1.0".into(),
            cpu: "Test CPU".into(),
            cpu_features: "x86_64-avx2".into(),
            model_size: 1024,
            model_fnv: 0x1234_5678_9abc_def0,
            created: 1700000000,
            options: vec![
                ("spin_us".into(), "500".into()),
                ("decode_threads".into(), "4".into()),
                ("prefetch_bytes".into(), "1024".into()),
            ],
        };

        write(&p, &prof).expect("write profile");
        let loaded = read(&p).expect("read profile");
        assert_eq!(prof, loaded);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn mismatch_detection_each_field() {
        let model = tmp_file("fake_model.bin");
        std::fs::write(&model, b"hello peregrine model file content").unwrap();

        let base_prof = Profile::new_for_model(&model).expect("new profile");
        assert!(check(&base_prof, &model).is_ok());

        // Version mismatch
        let mut p_ver = base_prof.clone();
        p_ver.peregrine_version = "0.0.0".into();
        let err = check(&p_ver, &model).unwrap_err();
        assert!(err.contains("version mismatch"), "{err}");

        // CPU mismatch
        let mut p_cpu = base_prof.clone();
        p_cpu.cpu = "Different CPU Model".into();
        let err = check(&p_cpu, &model).unwrap_err();
        assert!(err.contains("cpu mismatch"), "{err}");

        // CPU features mismatch
        let mut p_feat = base_prof.clone();
        p_feat.cpu_features = "scalar".into();
        let err = check(&p_feat, &model).unwrap_err();
        assert!(err.contains("cpu features mismatch"), "{err}");

        // Model size mismatch
        let mut p_size = base_prof.clone();
        p_size.model_size += 1;
        let err = check(&p_size, &model).unwrap_err();
        assert!(err.contains("model size mismatch"), "{err}");

        // Model hash mismatch
        let mut p_hash = base_prof.clone();
        p_hash.model_fnv ^= 0x1;
        let err = check(&p_hash, &model).unwrap_err();
        assert!(err.contains("model hash mismatch"), "{err}");

        let _ = std::fs::remove_file(&model);
    }
}
