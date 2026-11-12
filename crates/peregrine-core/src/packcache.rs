//! The load-time pack cache (S4.9e): the packed weights on disk, mapped at load.
//!
//! Every start used to repack the Q4_K/Q6_K/Q5_0/Q8_0 matrices into eight-row
//! tiles (`kq.rs`) and the F32/F16/BF16 matrices into sixteen-row panels
//! (`fpack.rs`). That is 1.7-22 s for the integer store and 1.3-4.1 s for the
//! float panels, paid on every process start even when the same model was loaded
//! a second ago. This module writes the two stores to one file after packing and
//! maps that file read-only on the next load, so the packed bytes are the page
//! cache's — shared by every process reading the model — instead of private
//! anonymous memory.
//!
//! # The key
//!
//! Everything the packed bytes depend on:
//!
//! * the model's identity — size, mtime (seconds and nanoseconds) and the
//!   FNV-1a hash of the GGUF header plus tensor directory. The header names every
//!   tensor, its shape, format and offset, so it identifies the layout without
//!   reading gigabytes of weights;
//! * [`FORMAT_VERSION`], bumped whenever either layout changes;
//! * the tensors each store holds, in store order, with their names, formats,
//!   shapes, offsets and lengths;
//! * the switches that choose the layouts (the k-quant repack, the float pack),
//!   the kernel tier, and the layout constants (tile rows, panel width).
//!
//! A mismatch is not an error: the file is ignored and the stores are packed
//! again.
//!
//! # The file
//!
//! A fixed header (magic, version, key, lengths, the model identity, a header
//! checksum and a sampled body checksum) followed by the two stores, each at a
//! 4 KiB-aligned offset so it can be mapped on its own. The body checksum hashes
//! 64 evenly spaced 4 KiB chunks, including the first and last, so a truncated or
//! badly corrupted file is rejected without reading (or hashing) the whole pack.
//!
//! Writes go to a temporary file in the same directory and are renamed into
//! place, so a reader sees a complete old file or a complete new one, and a crash
//! mid-write leaves only a `.tmp-` file. Two processes may pack the same model at
//! once: both write, one rename wins, and both load. Windows cannot rename over a
//! file that is mapped (S7.2), so when the destination already holds a valid pack
//! for the same key it is kept and the loser's temporary file is dropped: the
//! bytes a key names are the same bytes whoever wrote them.
//!
//! A cache directory that cannot be created or written (read-only, full) is not
//! an error: the load packs in memory and says so once.

use crate::fpack::FpEntry;
use crate::gguf::Gguf;
use crate::kq::KqEntry;
use crate::options::{EngineOptions, PackCacheSetting};
use crate::profile::fnv1a64;
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
#[cfg(not(unix))]
use std::io::{Read, SeekFrom};
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// The packed-layout generation. Bump this whenever `kq.rs` or `fpack.rs`
/// changes the bytes they write, so old cache files are never trusted.
pub const FORMAT_VERSION: u32 = 2;

/// The on-disk magic: eight bytes, then the version.
const MAGIC: &[u8; 8] = b"PRGPACK\0";

/// Bytes before the key text: magic, version, flags, lengths, offsets, the model
/// identity, the header checksum's input and the body checksum.
const FIXED: usize = 96;

/// The largest header a cache file may claim to have. A tree of ~1000 tensors is
/// a few tens of KiB; 16 MiB is a generous bound that still refuses garbage.
const MAX_HEADER: usize = 16 << 20;

/// One sampled chunk of the body, and how many chunks are sampled.
const SAMPLE_CHUNK: usize = 4096;
const SAMPLES: usize = 64;

/// The smallest packed store a cache is worth writing for. A synthetic test model
/// packs a couple of MiB; a real model orders of magnitude more. Below this the
/// pack is faster than the cache bookkeeping.
pub const MIN_PACK_BYTES: u64 = 4 << 20;

const MIB: u64 = 1 << 20;

/// Where the pack cache lives when `PEREGRINE_PACK_CACHE` gives no directory.
///
/// `$PEREGRINE_CACHE_DIR/packs`, else `$XDG_CACHE_HOME/peregrine/packs`, else
/// `%LOCALAPPDATA%\peregrine\packs` on Windows, else `~/.cache/peregrine/packs` — the model
/// cache's directory and conventions (`peregrine/src/fetch.rs`).
pub fn default_dir() -> PathBuf {
    if let Ok(d) = std::env::var("PEREGRINE_CACHE_DIR") {
        if !d.is_empty() {
            return PathBuf::from(d).join("packs");
        }
    }
    if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
        if !d.is_empty() {
            return PathBuf::from(d).join("peregrine").join("packs");
        }
    }
    #[cfg(windows)]
    {
        // The model cache's Windows default, plus the pack subdirectory (`fetch::cache_dir`).
        if let Ok(d) = std::env::var("LOCALAPPDATA") {
            if !d.is_empty() {
                return PathBuf::from(d).join("peregrine").join("packs");
            }
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".cache")
        .join("peregrine")
        .join("packs")
}

/// The cache's directory and policy, or `None` when it is off.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackCache {
    pub dir: PathBuf,
    /// The directory's size cap; `u64::MAX` means no eviction.
    pub cap_bytes: u64,
    /// The layout generation written into the key (a test may bump it).
    pub version: u32,
    /// Stores smaller than this are not cached (see [`MIN_PACK_BYTES`]).
    pub min_bytes: u64,
}

impl PackCache {
    /// A cache in `dir` with `cap_bytes` and the current layout version.
    pub fn new(dir: impl Into<PathBuf>, cap_bytes: u64) -> PackCache {
        PackCache {
            dir: dir.into(),
            cap_bytes,
            version: FORMAT_VERSION,
            min_bytes: 0,
        }
    }

    /// The configured cache, or `None` when `PEREGRINE_PACK_CACHE=0`.
    pub fn from_options(o: &EngineOptions) -> Option<PackCache> {
        let dir = match &o.pack_cache {
            PackCacheSetting::Off => return None,
            PackCacheSetting::Default => default_dir(),
            PackCacheSetting::Path(p) => p.clone(),
        };
        let cap_bytes = if o.pack_cache_mb == 0 {
            u64::MAX
        } else {
            o.pack_cache_mb as u64 * MIB
        };
        Some(PackCache {
            dir,
            cap_bytes,
            version: FORMAT_VERSION,
            min_bytes: MIN_PACK_BYTES,
        })
    }

    /// A different layout generation (a test hook for invalidation).
    pub fn with_version(mut self, version: u32) -> PackCache {
        self.version = version;
        self
    }

    /// Whether a pack of `bytes` is worth caching.
    pub fn worth_caching(&self, bytes: u64) -> bool {
        bytes >= self.min_bytes && bytes > 0
    }

    /// The file a key maps to.
    pub fn path(&self, key: &PackKey) -> PathBuf {
        self.dir
            .join(format!("pack-{}-{:016x}.bin", key.stem, key.hash))
    }

    /// Compute the key for this model and these layouts.
    pub fn key(
        &self,
        g: &Gguf,
        kq: &[KqEntry],
        fp: &[FpEntry],
        kq_on: bool,
        fp_on: bool,
    ) -> Result<PackKey, String> {
        let md = std::fs::metadata(g.path())
            .map_err(|e| format!("pack cache: cannot stat {}: {e}", g.path().display()))?;
        let (secs, nanos) = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| (d.as_secs(), d.subsec_nanos()))
            .unwrap_or((0, 0));
        let header_fnv = g.header_fnv();
        let tier = peregrine_kernels::cpu::tier_name();
        let mut text = String::with_capacity(256 + 96 * (kq.len() + fp.len()));
        text.push_str("peregrine-pack\n");
        text.push_str(&format!("version={}\n", self.version));
        text.push_str(&format!("model_size={}\n", md.len()));
        text.push_str(&format!("model_mtime={secs}.{nanos:09}\n"));
        text.push_str(&format!("header_fnv={header_fnv:016x}\n"));
        // The weights themselves, sampled: size, mtime and the header do not tell two files with
        // the same tensors apart.
        text.push_str(&format!("data_fnv={:016x}\n", g.data_sample_fnv()));
        text.push_str(&format!("tier={tier}\n"));
        text.push_str(&format!("kq={}\n", u8::from(kq_on)));
        text.push_str(&format!("fp={}\n", u8::from(fp_on)));
        text.push_str(&format!("tile_rows={}\n", crate::kq::TILE_ROWS));
        text.push_str(&format!("panel={}\n", crate::fpack::PANEL));
        text.push_str("endian=le\n");
        for e in kq {
            text.push_str(&format!(
                "kq_entry={}|{}|{}|{}|{}|{}\n",
                e.name,
                e.kind.label(),
                e.rows,
                e.cols,
                e.offset,
                e.len
            ));
        }
        for e in fp {
            text.push_str(&format!(
                "fp_entry={}|{}|{}|{}|{}|{}|{}|{}\n",
                e.name,
                match e.expert {
                    Some(x) => x.to_string(),
                    None => "-".into(),
                },
                e.kind.label(),
                e.source.label(),
                e.rows,
                e.cols,
                e.offset,
                e.len
            ));
        }
        let hash = fnv1a64(text.as_bytes());
        Ok(PackKey {
            text,
            hash,
            stem: stem_of(g.path()),
            version: self.version,
            model_size: md.len(),
            model_mtime: (secs, nanos),
            header_fnv,
        })
    }

    /// Open the cache file for `key`, verifying its header, identity, length and
    /// sampled body checksum. `None` on any mismatch, so the caller packs.
    pub fn open(&self, key: &PackKey) -> Option<CacheHit> {
        let path = self.path(key);
        let file = File::open(&path).ok()?;
        let len = file.metadata().ok()?.len();
        let h = read_header(&file, len)?;
        if h.version != key.version
            || h.key != key.text.as_bytes()
            || h.model_size != key.model_size
            || (h.mtime_secs, h.mtime_nanos) != key.model_mtime
            || h.header_fnv != key.header_fnv
        {
            return None;
        }
        let want_flags = u32::from(h.kq_len > 0) | (u32::from(h.fp_len > 0) << 1);
        if h.flags != want_flags {
            return None;
        }
        let expected = if h.fp_len > 0 {
            h.fp_off.checked_add(h.fp_len)?
        } else if h.kq_len > 0 {
            h.kq_off.checked_add(h.kq_len)?
        } else {
            h.header_len as u64
        };
        if len != expected {
            return None;
        }
        if h.kq_off % 4096 != 0
            || (h.fp_len > 0 && h.fp_off % 4096 != 0)
            || h.kq_off < h.header_len as u64
            || (h.fp_len > 0 && h.fp_off < h.kq_off + h.kq_len)
        {
            return None;
        }
        let total = h.kq_len.checked_add(h.fp_len)?;
        let sampled = file_body_fnv(&file, &h, total).ok()?;
        if sampled != h.body_fnv {
            return None;
        }
        // The old file's mtime is the LRU order; a hit counts as a use.
        if let Ok(f) = OpenOptions::new().write(true).open(&path) {
            let _ = f.set_modified(SystemTime::now());
        }
        let kq = if h.kq_len > 0 {
            Some(map_section(&file, h.kq_off, h.kq_len)?)
        } else {
            None
        };
        let fp = if h.fp_len > 0 {
            Some(map_section(&file, h.fp_off, h.fp_len)?)
        } else {
            None
        };
        Some(CacheHit { kq, fp })
    }

    /// Write the packed stores for `key` atomically, then evict old packs to stay
    /// under the cap. Best effort: an unwritable directory only logs and returns.
    pub fn write(&self, key: &PackKey, kq: &[u8], fp: &[u8]) -> Option<PathBuf> {
        if !self.worth_caching((kq.len() + fp.len()) as u64) {
            return None;
        }
        if let Err(e) = std::fs::create_dir_all(&self.dir) {
            eprintln!(
                "peregrine: pack cache unavailable ({}): {e}; packing in memory",
                self.dir.display()
            );
            return None;
        }
        let path = self.path(key);
        let header_len = FIXED + key.text.len() + 8;
        let kq_off = align_up(header_len as u64, 4096);
        let fp_off = align_up(kq_off + kq.len() as u64, 4096);
        let body_fnv = body_fnv(kq, fp);
        let header = build_header(key, kq.len(), fp.len(), kq_off, fp_off, body_fnv);
        let tmp = self.tmp_path(key);
        let r = write_file(&tmp, &header, kq_off, kq, fp_off, fp);
        if let Err(e) = r {
            eprintln!(
                "peregrine: pack cache unwritable ({}): {e}; packing in memory",
                self.dir.display()
            );
            let _ = std::fs::remove_file(&tmp);
            return None;
        }
        self.publish(&tmp, &path, key)?;
        self.evict(&path);
        Some(path)
    }

    /// Move the finished temporary file to `path`; `None` when it could not be published.
    ///
    /// A rename replaces the destination atomically on Unix. Windows refuses to rename over a
    /// file that is mapped — by another process that won the race, or by this one — so a failed
    /// rename asks whether the destination is now a valid pack for `key`. It is the same bytes
    /// by construction, so keeping it is correct; only when it is not a valid pack does the
    /// publish count as failed. The temporary file is removed either way.
    fn publish(&self, tmp: &Path, path: &Path, key: &PackKey) -> Option<()> {
        match std::fs::rename(tmp, path) {
            Ok(()) => Some(()),
            Err(e) => {
                let _ = std::fs::remove_file(tmp);
                if self.open(key).is_some() {
                    return Some(());
                }
                eprintln!(
                    "peregrine: pack cache rename failed ({}): {e}",
                    path.display()
                );
                None
            }
        }
    }

    fn tmp_path(&self, key: &PackKey) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
        self.dir.join(format!(
            ".tmp-{}-{:016x}-{uniq}",
            std::process::id(),
            key.hash
        ))
    }

    /// Drop least-recently-used pack files until the directory fits the cap.
    /// `keep` (the file just written or read) is never removed.
    fn evict(&self, keep: &Path) {
        if self.cap_bytes == u64::MAX {
            return;
        }
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return;
        };
        let mut files: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
        let mut total = 0u64;
        for entry in rd.flatten() {
            let p = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("pack-") || !name.ends_with(".bin") {
                continue;
            }
            let Ok(md) = entry.metadata() else { continue };
            if !md.is_file() {
                continue;
            }
            total += md.len();
            let used = md.modified().unwrap_or(UNIX_EPOCH);
            files.push((used, md.len(), p));
        }
        files.sort_by_key(|f| f.0);
        for (_, len, p) in files {
            if total <= self.cap_bytes {
                break;
            }
            if p == keep {
                continue;
            }
            if std::fs::remove_file(&p).is_ok() {
                total = total.saturating_sub(len);
            }
        }
    }
}

/// One model's identity and layouts, hashed into the cache file's header.
#[derive(Clone, Debug)]
pub struct PackKey {
    text: String,
    hash: u64,
    stem: String,
    version: u32,
    model_size: u64,
    model_mtime: (u64, u32),
    header_fnv: u64,
}

impl PackKey {
    /// The file this key names, for diagnostics and tests.
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

/// The two stores mapped from a cache file.
pub struct CacheHit {
    pub kq: Option<Mmap>,
    pub fp: Option<Mmap>,
}

/// What a model's pack cache holds, for `peregrine inspect` / `tier`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelStatus {
    /// The directory consulted (even when the cache is off).
    pub dir: PathBuf,
    /// Whether the cache is enabled.
    pub enabled: bool,
    /// How many cache files exist for this model's identity.
    pub files: usize,
    /// Their total size.
    pub bytes: u64,
}

impl ModelStatus {
    /// A one-line description, e.g. `2 files, 511.2 MiB in /…/packs`.
    pub fn describe(&self) -> String {
        if !self.enabled {
            return format!("off ({})", self.dir.display());
        }
        if self.files == 0 {
            return format!("none ({})", self.dir.display());
        }
        format!(
            "{} file{}, {:.1} MiB in {}",
            self.files,
            if self.files == 1 { "" } else { "s" },
            self.bytes as f64 / (1 << 20) as f64,
            self.dir.display()
        )
    }
}

/// Whether `path` has one or more pack cache files, and how large they are.
///
/// The files are matched by the model's identity in their headers (size, mtime,
/// header hash), so a cache written for another model, or for another option set,
/// is not counted. This is a header reading only: it does not latch the engine
/// options or map the weights.
pub fn model_status(path: &Path, opts: &EngineOptions) -> ModelStatus {
    let dir = match &opts.pack_cache {
        PackCacheSetting::Off => default_dir(),
        PackCacheSetting::Default => default_dir(),
        PackCacheSetting::Path(p) => p.clone(),
    };
    let enabled = !matches!(opts.pack_cache, PackCacheSetting::Off);
    let mut st = ModelStatus {
        dir: dir.clone(),
        enabled,
        files: 0,
        bytes: 0,
    };
    if !enabled {
        return st;
    }
    let Ok(g) = Gguf::open_plain(path) else {
        return st;
    };
    let Ok(md) = std::fs::metadata(path) else {
        return st;
    };
    let (secs, nanos) = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| (d.as_secs(), d.subsec_nanos()))
        .unwrap_or((0, 0));
    let header_fnv = g.header_fnv();
    let prefix = format!("pack-{}-", stem_of(path));
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return st;
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with(&prefix) {
            continue;
        }
        let Ok(file) = File::open(entry.path()) else {
            continue;
        };
        let Ok(len) = file.metadata().map(|m| m.len()) else {
            continue;
        };
        let Some(h) = read_header(&file, len) else {
            continue;
        };
        if h.model_size == md.len()
            && (h.mtime_secs, h.mtime_nanos) == (secs, nanos)
            && h.header_fnv == header_fnv
        {
            st.files += 1;
            st.bytes += len;
        }
    }
    st
}

// ---------------------------------------------------------------- the file --

struct Header {
    version: u32,
    flags: u32,
    header_len: usize,
    key: Vec<u8>,
    kq_len: u64,
    fp_len: u64,
    kq_off: u64,
    fp_off: u64,
    model_size: u64,
    mtime_secs: u64,
    mtime_nanos: u32,
    header_fnv: u64,
    body_fnv: u64,
}

fn build_header(
    key: &PackKey,
    kq_len: usize,
    fp_len: usize,
    kq_off: u64,
    fp_off: u64,
    body_fnv: u64,
) -> Vec<u8> {
    let header_len = FIXED + key.text.len() + 8;
    let mut h = vec![0u8; header_len];
    h[0..8].copy_from_slice(MAGIC);
    h[8..12].copy_from_slice(&key.version.to_le_bytes());
    let flags = u32::from(kq_len > 0) | (u32::from(fp_len > 0) << 1);
    h[12..16].copy_from_slice(&flags.to_le_bytes());
    h[16..20].copy_from_slice(&(key.text.len() as u32).to_le_bytes());
    h[20..24].copy_from_slice(&(header_len as u32).to_le_bytes());
    h[24..32].copy_from_slice(&(kq_len as u64).to_le_bytes());
    h[32..40].copy_from_slice(&(fp_len as u64).to_le_bytes());
    h[40..48].copy_from_slice(&kq_off.to_le_bytes());
    h[48..56].copy_from_slice(&fp_off.to_le_bytes());
    h[56..64].copy_from_slice(&key.model_size.to_le_bytes());
    h[64..72].copy_from_slice(&key.model_mtime.0.to_le_bytes());
    h[72..76].copy_from_slice(&key.model_mtime.1.to_le_bytes());
    h[80..88].copy_from_slice(&key.header_fnv.to_le_bytes());
    h[88..96].copy_from_slice(&body_fnv.to_le_bytes());
    h[FIXED..FIXED + key.text.len()].copy_from_slice(key.text.as_bytes());
    let sum = fnv1a64(&h[..header_len - 8]);
    h[header_len - 8..].copy_from_slice(&sum.to_le_bytes());
    h
}

fn read_header(file: &File, len: u64) -> Option<Header> {
    if len < (FIXED + 8) as u64 {
        return None;
    }
    let mut fixed = [0u8; FIXED];
    read_exact_at(file, 0, &mut fixed).ok()?;
    if &fixed[0..8] != MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(fixed[8..12].try_into().ok()?);
    let flags = u32::from_le_bytes(fixed[12..16].try_into().ok()?);
    let key_len = u32::from_le_bytes(fixed[16..20].try_into().ok()?) as usize;
    let header_len = u32::from_le_bytes(fixed[20..24].try_into().ok()?) as usize;
    if !(FIXED + 8..=MAX_HEADER).contains(&header_len)
        || header_len as u64 > len
        || key_len != header_len - FIXED - 8
    {
        return None;
    }
    let mut h = vec![0u8; header_len];
    read_exact_at(file, 0, &mut h).ok()?;
    let want = u64::from_le_bytes(h[header_len - 8..].try_into().ok()?);
    if fnv1a64(&h[..header_len - 8]) != want {
        return None;
    }
    let key = h[FIXED..FIXED + key_len].to_vec();
    Some(Header {
        version,
        flags,
        header_len,
        key,
        kq_len: u64::from_le_bytes(h[24..32].try_into().ok()?),
        fp_len: u64::from_le_bytes(h[32..40].try_into().ok()?),
        kq_off: u64::from_le_bytes(h[40..48].try_into().ok()?),
        fp_off: u64::from_le_bytes(h[48..56].try_into().ok()?),
        model_size: u64::from_le_bytes(h[56..64].try_into().ok()?),
        mtime_secs: u64::from_le_bytes(h[64..72].try_into().ok()?),
        mtime_nanos: u32::from_le_bytes(h[72..76].try_into().ok()?),
        header_fnv: u64::from_le_bytes(h[80..88].try_into().ok()?),
        body_fnv: u64::from_le_bytes(h[88..96].try_into().ok()?),
    })
}

/// Write `header` at 0, `kq` at `kq_off`, `fp` at `fp_off`; the gaps stay zero.
fn write_file(
    path: &Path,
    header: &[u8],
    kq_off: u64,
    kq: &[u8],
    fp_off: u64,
    fp: &[u8],
) -> std::io::Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(header)?;
    if !kq.is_empty() {
        let at = f.stream_position()?;
        if at < kq_off {
            f.write_all(&vec![0u8; (kq_off - at) as usize])?;
        }
        f.write_all(kq)?;
    }
    if !fp.is_empty() {
        let at = f.stream_position()?;
        if at < fp_off {
            let pad = (fp_off - at) as usize;
            // A big gap between the two stores is normal (the k-quant store is
            // page-aligned); write it in chunks so no temporary allocation is large.
            const Z: [u8; 8192] = [0u8; 8192];
            let mut left = pad;
            while left > 0 {
                let n = left.min(Z.len());
                f.write_all(&Z[..n])?;
                left -= n;
            }
        }
        f.write_all(fp)?;
    }
    f.flush()
}

fn map_section(file: &File, off: u64, len: u64) -> Option<Mmap> {
    // SAFETY: the file is the cache file the caller just verified; its length is
    // checked before the mapping is made, and the mapping is read-only.
    let mut opts = memmap2::MmapOptions::new();
    opts.offset(off).len(len as usize).populate();
    // SAFETY: the file is the cache file the caller just verified; its length is
    // checked before the mapping is made, and the mapping is read-only.
    let map = unsafe { opts.map(file).ok()? };
    // File-backed transparent huge pages are a hint the kernel may ignore (on
    // Linux, a non-tmpfs file mapping ignores MADV_HUGEPAGE); try it when the
    // in-memory stores would have asked for huge pages, so the A/B is honest.
    #[cfg(target_os = "linux")]
    if crate::options::get().tile_hugepages {
        let _ = map.advise(memmap2::Advice::HugePage);
    }
    Some(map)
}

// ------------------------------------------------------------ body checksum --

/// The (start, length) of every sampled chunk of a logical body of `total` bytes.
/// The first chunk and the last chunk are always included.
fn sample_ranges(total: usize) -> Vec<(usize, usize)> {
    if total == 0 {
        return Vec::new();
    }
    let n = SAMPLES.min(total.div_ceil(SAMPLE_CHUNK).max(1)).max(1);
    let last = total - total.min(SAMPLE_CHUNK);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = if n == 1 { 0 } else { i * last / (n - 1) };
        out.push((start, SAMPLE_CHUNK.min(total - start)));
    }
    out
}

fn body_fnv(kq: &[u8], fp: &[u8]) -> u64 {
    let total = kq.len() + fp.len();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut buf = Vec::new();
    for (start, len) in sample_ranges(total) {
        buf.clear();
        if start + len <= kq.len() {
            buf.extend_from_slice(&kq[start..start + len]);
        } else if start >= kq.len() {
            let at = start - kq.len();
            buf.extend_from_slice(&fp[at..at + len]);
        } else {
            // The chunk straddles the two stores.
            buf.extend_from_slice(&kq[start..]);
            buf.extend_from_slice(&fp[..len - (kq.len() - start)]);
        }
        h = fnv1a64_cont(h, &buf);
    }
    h
}

fn fnv1a64_cont(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn file_body_fnv(file: &File, h: &Header, total: u64) -> std::io::Result<u64> {
    let mut sum = 0xcbf2_9ce4_8422_2325u64;
    let mut buf = vec![0u8; SAMPLE_CHUNK];
    for (start, len) in sample_ranges(total as usize) {
        read_logical(file, h, start as u64, &mut buf[..len])?;
        sum = fnv1a64_cont(sum, &buf[..len]);
    }
    Ok(sum)
}

fn read_logical(file: &File, h: &Header, start: u64, buf: &mut [u8]) -> std::io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let logical = start + done as u64;
        let (phys, avail) = if logical < h.kq_len {
            (h.kq_off + logical, h.kq_len - logical)
        } else {
            (
                h.fp_off + (logical - h.kq_len),
                h.fp_len - (logical - h.kq_len),
            )
        };
        let n = (buf.len() - done).min(avail as usize);
        read_exact_at(file, phys, &mut buf[done..done + n])?;
        done += n;
    }
    Ok(())
}

// ------------------------------------------------------------------ helpers --

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

fn stem_of(path: &Path) -> String {
    let raw = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let mut s: String = raw
        .chars()
        .take(48)
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() {
        s.push_str("model");
    }
    s
}

#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    let mut f = file.try_clone()?;
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "peregrine-packcache-unit-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn key() -> PackKey {
        PackKey {
            text: "unit-test-key\n".into(),
            hash: 0x1234_5678,
            stem: "model".into(),
            version: FORMAT_VERSION,
            model_size: 0,
            model_mtime: (0, 0),
            header_fnv: 0,
        }
    }

    /// Publishing over a pack that is already mapped (S7.2): Unix replaces the file atomically;
    /// Windows cannot rename over a mapped file, so the file already published for the key is
    /// kept. Either way the second publish succeeds, a mapping made before it still reads its
    /// bytes, one pack file exists, and no `.tmp-` file is left behind.
    #[test]
    fn publishing_over_a_mapped_pack_keeps_one_valid_file() {
        let dir = tmp("publish");
        let cache = PackCache::new(&dir, u64::MAX);
        let key = key();
        let path = cache
            .write(&key, b"kq-bytes", b"fp-bytes")
            .expect("the first publish");
        let hit = cache.open(&key).expect("the file just published verifies");
        assert_eq!(&hit.kq.as_ref().unwrap()[..], b"kq-bytes".as_slice());
        // The same key published again while the first file is mapped.
        assert_eq!(
            cache.write(&key, b"kq-bytes", b"fp-bytes"),
            Some(path.clone())
        );
        assert_eq!(&hit.fp.as_ref().unwrap()[..], b"fp-bytes".as_slice());
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names.iter().filter(|n| n.starts_with("pack-")).count(),
            1,
            "{names:?}"
        );
        assert!(names.iter().all(|n| !n.starts_with(".tmp-")), "{names:?}");
        drop(hit);
        let again = cache.open(&key).expect("the kept file still verifies");
        assert_eq!(&again.kq.as_ref().unwrap()[..], b"kq-bytes".as_slice());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
