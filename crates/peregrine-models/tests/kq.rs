//! The k-quant store (S3.4): the Q4_K and Q6_K matrices, repacked into eight-row
//! tiles at load, and the repacked kernel that multiplies them.
//!
//! With the repack on, the loader rewrites every eligible matrix into the repacked
//! layout and keeps the result in the model's k-quant store, and the Q4_K and Q6_K
//! matrices are multiplied from it. The repacked kernel reduces a super-block's eight
//! position classes in int32, where the ordinary kernel reduces them in float at
//! the very end, so the two compute the same real number with a different
//! rounding: the logits move in the last bits (the design note's §4.2).
//!
//! The store has to hold exactly the repack of every eligible matrix all the same:
//! the kernel multiplies those bytes, and a store that is one tile out of place
//! would multiply the wrong weights.
//!
//! One test, because the engine's options are process-wide and read once: the repack
//! is turned on here, before anything reads them.

#[path = "support/mod.rs"]
mod support;

use peregrine_core::gguf::{Format, Gguf, TensorInfo};
use peregrine_core::model::{Model, StepStats};
use peregrine_core::pool::Pool;
use peregrine_kernels::kq::{unrepack_q4k8, unrepack_q6k8, Q4K8, Q6K8};
use std::path::PathBuf;
use support::models::{write_dense, Dense};

/// The rows one tile holds.
const TILE_ROWS: usize = 8;

/// The token embedding: gathered row by row, never multiplied, so it is not repacked.
const EMBEDDING: &str = "token_embd.weight";

/// A directory of its own for the test (it removes it when it is done).
fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-kq-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The bytes one repacked tile of the given format takes.
fn tile_size(kind: Format) -> usize {
    match kind {
        Format::Q4_K => Q4K8::SIZE,
        Format::Q6_K => Q6K8::SIZE,
        other => panic!("{other:?} has no repacked layout"),
    }
}

/// Whether the store takes this tensor: the rule the loader uses.
fn eligible(t: &TensorInfo) -> bool {
    matches!(t.kind, Format::Q4_K | Format::Q6_K)
        && t.dims.len() == 2
        && t.name != EMBEDDING
        && t.dims[1] as usize >= TILE_ROWS
        && (t.dims[0] as usize).is_multiple_of(t.kind.block())
}

/// The repacked rows of a matrix: whole groups of eight from the top.
fn repacked_rows(t: &TensorInfo) -> usize {
    (t.dims[1] as usize / TILE_ROWS) * TILE_ROWS
}

/// The logits of a short prefill, and the activation bytes its matmuls read.
fn prefill(model: &Model) -> (Vec<f32>, u64) {
    let threads = 2;
    let pool = Pool::new(threads);
    let tokens: Vec<u32> = (1..=16).collect();
    let mut st = model.new_state(32, tokens.len(), 256, threads);
    let mut kv = model.new_cache(peregrine_core::kv::KvPrec::F16, 256);
    let mut ss = StepStats::default();
    let want: Vec<usize> = (0..tokens.len()).collect();
    model
        .forward(&mut st, &mut kv, &pool, &tokens, 0, &want, &mut ss)
        .expect("the prefill runs");
    (st.logits_row(0, model.cfg.n_vocab).to_vec(), ss.bytes_acts)
}

/// The store holds the repack of every eligible matrix, and the repack moves no
/// output.
#[test]
fn the_store_holds_the_repack_of_every_eligible_matrix() {
    // The engine's options are read once, so the repack is turned on here, before
    // anything reads them.
    peregrine_core::options::init(peregrine_core::options::EngineOptions {
        kq_repack: true,
        ..Default::default()
    })
    .ok();
    assert!(
        peregrine_core::options::get().kq_repack,
        "the k-quant repack is on"
    );

    let dir = tmpdir("store");
    for fmt in [Format::Q4_K, Format::Q6_K] {
        let path = write_dense(&dir, "llama", fmt, 4242, Dense::default());
        let g = Gguf::open(&path).expect("the synthetic model opens");

        // The ordinary load builds the store only where a kernel reads it: the repack is
        // on in this process, so the store appears exactly when the table has the
        // repacked kernel (not on a `PEREGRINE_ISA=scalar` run, and never off x86-64).
        let ordinary = peregrine_models::load(Gguf::open(&path).expect("the model opens"))
            .expect("the synthetic model loads");
        assert_eq!(
            !ordinary.kq().is_empty(),
            peregrine_kernels::table::MatKernels::get().kq_tiles(),
            "{fmt:?}: the ordinary load and the kernel table disagree about the store"
        );

        // The store's own contents, forced on: this is the test of the layout, so it runs on
        // every tier and every architecture, including the ones no kernel reads the store on.
        let model = Model::load_with_kq(
            Gguf::open(&path).expect("the model opens"),
            peregrine_models::FAMILIES,
            true,
        )
        .expect("the synthetic model loads");
        let store = model.kq();
        assert!(!store.is_empty(), "{fmt:?}: the store is empty");

        // The matrices the store takes, in the store's order: the file's own.
        let mut mats: Vec<(&String, &TensorInfo)> =
            g.tensors.iter().filter(|(_, t)| eligible(t)).collect();
        mats.sort_by_key(|(name, t)| (t.offset, name.as_str()));
        assert!(!mats.is_empty(), "{fmt:?}: no eligible matrix");

        let mut at = 0usize;
        for (_name, t) in mats {
            let rows = repacked_rows(t);
            let cols = t.dims[0] as usize;
            let n_sb = cols / t.kind.block();
            let len = (rows / TILE_ROWS) * n_sb * tile_size(t.kind);
            let stored = &store[at..at + len];
            check_matrix(t.kind, g.bytes(t), stored, rows, cols);
            at += len;
        }
        // Nothing else is in the store: not the embedding, not the norms, and not
        // the rows past the last whole group of eight.
        assert_eq!(at, store.len(), "{fmt:?}: the store holds something else");

        // The same file, loaded without the repack, is multiplied the ordinary
        // way. The two compute the same real number with a different rounding,
        // so the logits move in the last bits — not by more than the fold's
        // last bit.
        let plain = Model::load_with_kq(
            Gguf::open(&path).expect("the model opens"),
            peregrine_models::FAMILIES,
            false,
        )
        .expect("the synthetic model loads");
        assert!(
            plain.kq().is_empty(),
            "{fmt:?}: the plain model has a store"
        );
        let (a, plain_acts) = prefill(&plain);
        let (b, acts) = prefill(&model);
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            // Both eligible formats now run the repacked kernel, whose fold
            // rounds once where the ordinary kernel rounds twice: the same
            // real number, not the same bits. This bound holds here only
            // because a random-weight model doesn't amplify the few-ulp
            // difference; a trained model moves its logits by percent
            // (docs/design/s3.4.md §10). The kernel's own bound is the
            // per-matmul check in kqsimd.rs.
            assert!(
                (x - y).abs() <= 1e-3 * x.abs().max(1.0),
                "{fmt:?}: logit {i} moved: {x} against {y}"
            );
        }
        if peregrine_kernels::table::MatKernels::get().kq_tiles() {
            // The repacked tile reads the activation once per eight output rows
            // instead of once per row: the activation traffic falls by the
            // same factor. This is the design note's claim, measured. (Only
            // where the repacked kernel runs: on a CPU without AVX2 the
            // matrices are multiplied the ordinary way.)
            assert!(
                plain_acts > acts * 4,
                "{fmt:?}: the activation traffic did not fall: {acts} against {plain_acts}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every tile of the store is the repack of the file's bytes it stands for.
fn check_matrix(kind: Format, src: &[u8], stored: &[u8], rows: usize, cols: usize) {
    let n_sb = cols / kind.block();
    let blk = kind.block_bytes();
    let rb = n_sb * blk;
    let tile = tile_size(kind);
    let mut back = vec![0u8; 8 * blk];
    for g in 0..rows / TILE_ROWS {
        for s in 0..n_sb {
            let at = (g * n_sb + s) * tile;
            let t = &stored[at..at + tile];
            match kind {
                Format::Q4_K => unrepack_q4k8(t, &mut back),
                _ => unrepack_q6k8(t, &mut back),
            }
            for r in 0..8 {
                let want = &src[(g * TILE_ROWS + r) * rb + s * blk..][..blk];
                assert_eq!(
                    &back[r * blk..(r + 1) * blk],
                    want,
                    "{kind:?}: tile {g},{s} row {r} is not the file's bytes"
                );
            }
        }
    }
}
