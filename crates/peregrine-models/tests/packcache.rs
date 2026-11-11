//! The pack cache (S4.9e): the packed weights written once and mapped at load.
//!
//! Each test owns a temporary model directory and a temporary cache directory, so
//! a test never reads another's pack. The engine options are process-wide, but
//! every load here passes the cache and the two pack switches explicitly.

mod support;

use peregrine_core::embed::{self, Pooling};
use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, StepStats};
use peregrine_core::packcache::PackCache;
use peregrine_core::pool::Pool;
use std::path::{Path, PathBuf};
use std::time::Duration;
use support::models::{write_dense, Dense};

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-packcache-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &Path, cache: &PackCache, kq: bool, fp: bool) -> Model {
    Model::load_with_kq_fp_cache(
        Gguf::open(path).expect("open the synthetic model"),
        peregrine_models::FAMILIES,
        kq,
        fp,
        Some(cache),
    )
    .expect("load the synthetic model")
}

fn load_plain(path: &Path, kq: bool, fp: bool) -> Model {
    Model::load_with_kq_fp_cache(
        Gguf::open(path).expect("open the synthetic model"),
        peregrine_models::FAMILIES,
        kq,
        fp,
        None,
    )
    .expect("load the synthetic model")
}

fn logits(model: &Model) -> Vec<f32> {
    let threads = 2;
    let pool = Pool::new(threads);
    let tokens: Vec<u32> = (1..=8).collect();
    let mut st = model.new_state(16, tokens.len(), 128, threads);
    let mut kv = model.new_cache(KvPrec::F16, 128);
    let mut ss = StepStats::default();
    let want: Vec<usize> = (0..tokens.len()).collect();
    model
        .forward(&mut st, &mut kv, &pool, &tokens, 0, &want, &mut ss)
        .expect("the prefill runs");
    st.logits_row(0, model.cfg.n_vocab).to_vec()
}

fn embedding(model: &Model) -> Vec<f32> {
    let threads = 2;
    let pool = Pool::new(threads);
    let tokens: Vec<u32> = (3..=10).collect();
    let mut st = model.new_state(tokens.len(), 1, 128, threads);
    st.want_hidden = true;
    let mut kv = model.new_cache(KvPrec::F16, 128);
    let mut ss = StepStats::default();
    model
        .forward(&mut st, &mut kv, &pool, &tokens, 0, &[], &mut ss)
        .expect("the forward runs");
    let mut out = Vec::new();
    embed::pool_rows(
        &st.hidden[..tokens.len() * model.cfg.d_model],
        model.cfg.d_model,
        Pooling::Mean,
        &mut out,
    );
    out
}

fn pack_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .map(|n| n.to_string_lossy().starts_with("pack-"))
                        .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn only_pack(dir: &Path) -> PathBuf {
    let files = pack_files(dir);
    assert_eq!(files.len(), 1, "expected one pack in {dir:?}: {files:?}");
    files[0].clone()
}

#[test]
fn a_second_load_maps_the_pack_and_is_bit_identical() {
    let _ = peregrine_core::options::init(peregrine_core::options::EngineOptions::default());
    let root = tmpdir("hit");
    let models = root.join("models");
    let packs = root.join("packs");
    std::fs::create_dir_all(&models).unwrap();
    let path = write_dense(&models, "llama", Format::Q4_K, 7, Dense::default());
    let in_memory = load_plain(&path, true, true);
    let cache = PackCache::new(&packs, u64::MAX);

    let first = load(&path, &cache, true, true);
    assert!(!first.packed_from_cache(), "the first load packs");
    assert!(!first.kq().is_empty(), "the first load built the store");
    assert!(only_pack(&packs).metadata().unwrap().len() > 0);

    let second = load(&path, &cache, true, true);
    assert!(second.packed_from_cache(), "the second load maps the pack");
    // The bytes are the same bytes, not a copy that happens to compute the same.
    assert_eq!(first.kq(), second.kq());
    assert_eq!(first.fp(), second.fp());
    assert_eq!(in_memory.kq(), second.kq());
    let a = logits(&in_memory);
    let b = logits(&second);
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "logit {i}: {x} against {y}");
    }
    let e1 = embedding(&in_memory);
    let e2 = embedding(&second);
    for (i, (x, y)) in e1.iter().zip(&e2).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "embedding {i}: {x} against {y}");
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_float_store_round_trips_through_the_cache() {
    let _ = peregrine_core::options::init(peregrine_core::options::EngineOptions::default());
    let root = tmpdir("fp");
    let models = root.join("models");
    let packs = root.join("packs");
    std::fs::create_dir_all(&models).unwrap();
    let path = write_dense(&models, "llama", Format::F16, 11, Dense::default());
    let cache = PackCache::new(&packs, u64::MAX);
    let plain = load_plain(&path, false, true);
    assert!(plain.kq().is_empty() && !plain.fp().is_empty());
    let first = load(&path, &cache, false, true);
    assert!(!first.packed_from_cache());
    let second = load(&path, &cache, false, true);
    assert!(second.packed_from_cache());
    assert_eq!(second.fp(), plain.fp());
    assert_eq!(second.fp(), first.fp());
    assert!(peregrine_core::fpack::layout(&Gguf::open(&path).unwrap()).len() > 1);
    let a = logits(&plain);
    let b = logits(&second);
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "logit {i}: {x} against {y}");
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn every_key_component_invalidates() {
    let _ = peregrine_core::options::init(peregrine_core::options::EngineOptions::default());
    let root = tmpdir("invalidate");
    let models = root.join("models");
    let packs = root.join("packs");
    std::fs::create_dir_all(&models).unwrap();
    let path = write_dense(&models, "llama", Format::Q4_K, 21, Dense::default());
    let cache = PackCache::new(&packs, u64::MAX);

    // The model's mtime: touching the file must not serve the old pack.
    let first = load(&path, &cache, true, false);
    assert!(!first.packed_from_cache());
    assert!(load(&path, &cache, true, false).packed_from_cache());
    std::fs::File::open(&path)
        .unwrap()
        .set_modified(std::time::SystemTime::now() + Duration::from_secs(5))
        .unwrap();
    let touched = load(&path, &cache, true, false);
    assert!(!touched.packed_from_cache(), "a touched model rebuilds");
    assert_eq!(pack_files(&packs).len(), 2, "a new key is a new file");

    // The layout version: a test bump makes the old file invisible.
    let v2 = PackCache::new(&packs, u64::MAX)
        .with_version(peregrine_core::packcache::FORMAT_VERSION + 1);
    let bumped = load(&path, &v2, true, false);
    assert!(!bumped.packed_from_cache(), "a version bump rebuilds");
    assert_eq!(
        pack_files(&packs).len(),
        3,
        "the version is part of the key"
    );
    assert!(load(&path, &v2, true, false).packed_from_cache());

    // A switch that changes the layout: kq off must not read a kq pack.
    let plain = load(&path, &cache, false, false);
    assert!(!plain.packed_from_cache());
    assert!(plain.kq().is_empty());
    assert_eq!(pack_files(&packs).len(), 3, "no store, no cache file");
    let _ = std::fs::remove_dir_all(&root);
}

/// A model file replaced by another with the same tensors, the same size and the same
/// timestamp (a fine-tune copied over the old file with its mtime kept) must not be served the
/// old file's pack: the key samples the weights themselves.
#[test]
fn different_weights_under_the_same_name_size_and_mtime_do_not_share_a_pack() {
    let root = tmpdir("same-identity");
    let models = root.join("models");
    let other_models = root.join("other-models");
    let packs = root.join("packs");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::create_dir_all(&other_models).unwrap();
    let path = write_dense(&models, "llama", Format::Q4_K, 61, Dense::default());
    let other = write_dense(&other_models, "llama", Format::Q4_K, 62, Dense::default());
    let cache = PackCache::new(&packs, u64::MAX);

    let first = logits(&load(&path, &cache, true, true));
    let want = logits(&load_plain(&other, true, true));
    assert_ne!(first, want, "the two seeds give the same model");

    // Put the other weights under the first file's name, size and mtime.
    let meta = std::fs::metadata(&path).unwrap();
    assert_eq!(meta.len(), std::fs::metadata(&other).unwrap().len());
    let mtime = meta.modified().unwrap();
    std::fs::copy(&other, &path).unwrap();
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), mtime);

    let got = logits(&load(&path, &cache, true, true));
    assert_eq!(got, want, "the old file's pack was served for new weights");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn truncated_corrupted_and_foreign_files_are_rejected() {
    let _ = peregrine_core::options::init(peregrine_core::options::EngineOptions::default());
    let root = tmpdir("bad");
    let models = root.join("models");
    let packs = root.join("packs");
    std::fs::create_dir_all(&models).unwrap();
    let path = write_dense(&models, "llama", Format::Q4_K, 31, Dense::default());
    let cache = PackCache::new(&packs, u64::MAX);

    load(&path, &cache, true, false);
    let file = only_pack(&packs);
    let full = file.metadata().unwrap().len();

    // Truncated: the length in the header and the body checksum disagree.
    let f = std::fs::OpenOptions::new().write(true).open(&file).unwrap();
    f.set_len(full / 2).unwrap();
    drop(f);
    let truncated = load(&path, &cache, true, false);
    assert!(
        !truncated.packed_from_cache(),
        "a truncated pack is not trusted"
    );
    assert_eq!(
        file.metadata().unwrap().len(),
        full,
        "the pack was rewritten"
    );

    // Corrupted body: the first sampled chunk no longer matches.
    let mut head = [0u8; 48];
    {
        use std::io::Read as _;
        std::fs::File::open(&file)
            .unwrap()
            .read_exact(&mut head)
            .unwrap();
    }
    let kq_off = u64::from_le_bytes(head[40..48].try_into().unwrap());
    let mut one = [0u8; 1];
    {
        use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&file)
            .unwrap();
        f.seek(SeekFrom::Start(kq_off + 17)).unwrap();
        f.read_exact(&mut one).unwrap();
        one[0] ^= 0xff;
        f.seek(SeekFrom::Start(kq_off + 17)).unwrap();
        f.write_all(&one).unwrap();
    }
    let corrupted = load(&path, &cache, true, false);
    assert!(
        !corrupted.packed_from_cache(),
        "a corrupted pack is rebuilt"
    );

    // Foreign: another model's valid pack copied over this one's file.
    let other_models = root.join("other");
    std::fs::create_dir_all(&other_models).unwrap();
    let other = write_dense(&other_models, "llama", Format::Q4_K, 32, Dense::default());
    let foreign_cache = PackCache::new(root.join("foreign"), u64::MAX);
    load(&other, &foreign_cache, true, false);
    let foreign = only_pack(&root.join("foreign"));
    std::fs::copy(&foreign, &file).unwrap();
    let foreign_load = load(&path, &cache, true, false);
    assert!(
        !foreign_load.packed_from_cache(),
        "a foreign pack is rebuilt"
    );
    assert!(load(&path, &cache, true, false).packed_from_cache());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn concurrent_threads_both_load() {
    let _ = peregrine_core::options::init(peregrine_core::options::EngineOptions::default());
    let root = tmpdir("threads");
    let models = root.join("models");
    let packs = root.join("packs");
    std::fs::create_dir_all(&models).unwrap();
    let path = write_dense(&models, "llama", Format::Q4_K, 41, Dense::default());
    let reference = logits(&load_plain(&path, true, true));
    let cache = PackCache::new(&packs, u64::MAX);
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|s| {
        let mut got = Vec::new();
        for _ in 0..2 {
            let p = path.clone();
            let c = cache.clone();
            let b = &barrier;
            let h = s.spawn(move || {
                b.wait();
                logits(&load(&p, &c, true, true))
            });
            got.push(h);
        }
        for h in got {
            let v = h.join().expect("the thread loaded");
            assert_eq!(v.len(), reference.len());
            for (i, (x, y)) in v.iter().zip(&reference).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "logit {i}: {x} against {y}");
            }
        }
    });
    assert!(load(&path, &cache, true, true).packed_from_cache());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
#[ignore]
fn pack_cache_child_load() {
    let Ok(path) = std::env::var("PEREGRINE_PACK_CHILD_MODEL") else {
        return;
    };
    let Ok(dir) = std::env::var("PEREGRINE_PACK_CHILD_DIR") else {
        return;
    };
    let cache = PackCache::new(dir, u64::MAX);
    let model = load(Path::new(&path), &cache, true, true);
    assert!(!model.kq().is_empty());
    assert_eq!(logits(&model).len(), model.cfg.n_vocab);
}

#[test]
fn concurrent_processes_both_load() {
    let _ = peregrine_core::options::init(peregrine_core::options::EngineOptions::default());
    let root = tmpdir("processes");
    let models = root.join("models");
    let packs = root.join("packs");
    std::fs::create_dir_all(&models).unwrap();
    let path = write_dense(&models, "llama", Format::Q4_K, 51, Dense::default());
    let exe = std::env::current_exe().expect("the test binary");
    let mut children = Vec::new();
    for _ in 0..2 {
        children.push(
            std::process::Command::new(&exe)
                .args([
                    "--exact",
                    "pack_cache_child_load",
                    "--ignored",
                    "--nocapture",
                ])
                .env("PEREGRINE_PACK_CHILD_MODEL", &path)
                .env("PEREGRINE_PACK_CHILD_DIR", &packs)
                .env("PEREGRINE_PACK_CACHE", &packs)
                .env_remove("PEREGRINE_ISA")
                .spawn()
                .expect("spawn a child loader"),
        );
    }
    for mut c in children {
        let st = c.wait().expect("the child finishes");
        assert!(st.success(), "a concurrent child failed: {st:?}");
    }
    let cache = PackCache::new(&packs, u64::MAX);
    assert!(load(&path, &cache, true, true).packed_from_cache());
    assert_eq!(pack_files(&packs).len(), 1, "the race leaves one file");
    let _ = std::fs::remove_dir_all(&root);
}

#[cfg(unix)]
#[test]
fn a_read_only_cache_directory_falls_back() {
    use std::os::unix::fs::PermissionsExt;
    let _ = peregrine_core::options::init(peregrine_core::options::EngineOptions::default());
    let root = tmpdir("readonly");
    let models = root.join("models");
    let packs = root.join("packs");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::create_dir_all(&packs).unwrap();
    let path = write_dense(&models, "llama", Format::Q4_K, 61, Dense::default());
    // A root user can write into a 0555 directory; the test then proves nothing.
    if std::fs::write(packs.join("probe"), b"x").is_ok() {
        let _ = std::fs::remove_file(packs.join("probe"));
        let _ = std::fs::remove_dir_all(&root);
        eprintln!("running as root: the read-only cache test is skipped");
        return;
    }
    std::fs::set_permissions(&packs, std::fs::Permissions::from_mode(0o555)).unwrap();
    let cache = PackCache::new(&packs, u64::MAX);
    let model = load(&path, &cache, true, true);
    assert!(!model.packed_from_cache(), "the load packed in memory");
    assert!(!model.kq().is_empty(), "the load still succeeded");
    std::fs::set_permissions(&packs, std::fs::Permissions::from_mode(0o755)).unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn eviction_respects_the_cap() {
    let _ = peregrine_core::options::init(peregrine_core::options::EngineOptions::default());
    let root = tmpdir("evict");
    let a = root.join("a");
    let b = root.join("b");
    let packs = root.join("packs");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    let pa = write_dense(&a, "llama", Format::Q4_K, 71, Dense::default());
    let pb = write_dense(&b, "llama", Format::Q4_K, 72, Dense::default());

    // The first file's size is the cap, so writing the second must evict the first.
    let cache = PackCache::new(&packs, u64::MAX);
    load(&pa, &cache, true, false);
    let cap = only_pack(&packs).metadata().unwrap().len();
    let capped = PackCache::new(&packs, cap);
    load(&pb, &capped, true, false);
    let total: u64 = pack_files(&packs)
        .iter()
        .map(|p| p.metadata().unwrap().len())
        .sum();
    assert!(total <= cap, "{total} bytes in the cache, cap {cap}");
    assert_eq!(pack_files(&packs).len(), 1, "the older pack was evicted");
    let _ = std::fs::remove_dir_all(&root);
}
