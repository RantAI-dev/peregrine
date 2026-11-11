//! The packed float store (S4.9f): the F32, F16 and BF16 matrices, packed into
//! 16-output-row panels at load, and the packed kernel that multiplies them.
//!
//! With the pack on, the loader rewrites every eligible matrix into the packed
//! layout and keeps the result in the model's float store, and every float matrix is
//! multiplied from it. The packed kernel computes each output element as one
//! sequential FMA chain, where the ordinary tile multiplies and adds without FMA and
//! folds four partial sums; the two compute the same real number with a different
//! rounding, so the logits move in the last bits.
//!
//! The store has to hold exactly the pack of every eligible matrix all the same: the
//! kernel multiplies those bytes, and a store that is one panel out of place would
//! multiply the wrong weights.
//!
//! One test, because the engine's options are process-wide and read once.

#[path = "support/mod.rs"]
mod support;

use peregrine_core::gguf::{Format, Gguf, TensorInfo};
use peregrine_core::model::{Model, StepStats};
use peregrine_core::pool::Pool;
use peregrine_kernels::fpack::{elem_bytes, pack_panel, packed_len, panel_bytes, PANEL};
use std::path::PathBuf;
use support::models::{write_dense, Dense};

/// The token embedding: gathered row by row, never multiplied, so it is not packed.
const EMBEDDING: &str = "token_embd.weight";

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-fpack-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Whether the store takes this tensor: the rule the loader uses.
fn eligible(t: &TensorInfo) -> bool {
    matches!(t.kind, Format::F32 | Format::F16 | Format::BF16)
        && t.dims.len() == 2
        && t.name != EMBEDDING
        && t.dims[0] > 0
        && t.dims[1] > 0
}

/// The logits of a short prefill.
fn prefill(model: &Model) -> Vec<f32> {
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
    st.logits_row(0, model.cfg.n_vocab).to_vec()
}

/// One matrix's panels, unpacked back into the file's row layout, are the file's bytes.
fn check_matrix(kind: Format, file: &[u8], packed: &[u8], rows: usize, cols: usize) {
    let eb = elem_bytes(kind);
    let rb = cols * eb;
    let pb = panel_bytes(kind, cols);
    assert_eq!(packed.len(), packed_len(kind, rows, cols));
    let mut got = vec![0u8; rows * rb];
    for panel in 0..rows.div_ceil(PANEL) {
        let p = &packed[panel * pb..(panel + 1) * pb];
        for lane in 0..PANEL {
            let row = panel * PANEL + lane;
            if row >= rows {
                // The padding panel is zeros.
                for k in 0..cols {
                    let at = (k * PANEL + lane) * eb;
                    assert_eq!(&p[at..at + eb], &vec![0u8; eb][..], "padding row {row}");
                }
                continue;
            }
            for k in 0..cols {
                let at = (k * PANEL + lane) * eb;
                got[row * rb + k * eb..row * rb + (k + 1) * eb].copy_from_slice(&p[at..at + eb]);
            }
        }
    }
    assert_eq!(
        got,
        file[..rows * rb],
        "{kind:?}: the panels are not the matrix"
    );
    // And packing the file's own rows gives these bytes: the layout is a permutation.
    let mut again = vec![0u8; packed.len()];
    for panel in 0..rows.div_ceil(PANEL) {
        let first = panel * PANEL;
        let n = (rows - first).min(PANEL);
        let refs: Vec<&[u8]> = (0..n)
            .map(|r| &file[(first + r) * rb..(first + r + 1) * rb])
            .collect();
        pack_panel(kind, &refs, cols, &mut again[panel * pb..(panel + 1) * pb]);
    }
    assert_eq!(
        again, packed,
        "{kind:?}: pack_panel disagrees with the store"
    );
}

#[test]
fn the_store_holds_the_pack_of_every_float_matrix() {
    // The engine's options are read once, so the pack is turned on here.
    peregrine_core::options::init(peregrine_core::options::EngineOptions {
        fpack: true,
        ..Default::default()
    })
    .ok();
    assert!(peregrine_core::options::get().fpack, "the float pack is on");

    let dir = tmpdir("store");
    for fmt in [Format::F32, Format::F16, Format::BF16] {
        let path = write_dense(&dir, "llama", fmt, 4242, Dense::default());
        let g = Gguf::open(&path).expect("the synthetic model opens");

        // The ordinary load builds the store only where a kernel reads it: on a
        // target with the packed kernel (x86-64), and only with the switch on.
        let ordinary = peregrine_models::load(Gguf::open(&path).expect("the model opens"))
            .expect("the synthetic model loads");
        let packed_here = peregrine_kernels::table::MatKernels::get().fp_panels();
        assert_eq!(
            !ordinary.fp().is_empty(),
            packed_here,
            "{fmt:?}: the ordinary load and the kernel table disagree about the store"
        );

        // The store's own contents, forced on: this is the test of the layout, so
        // it runs on every tier and every architecture.
        let model = Model::load_with_kq_fp(
            Gguf::open(&path).expect("the model opens"),
            peregrine_models::FAMILIES,
            true,
            true,
        )
        .expect("the synthetic model loads");
        let store = model.fp();
        assert!(!store.is_empty(), "{fmt:?}: the store is empty");

        // The matrices the store takes, in the store's order: the file's own.
        let entries = peregrine_core::fpack::layout(&g);
        assert!(!entries.is_empty(), "{fmt:?}: no eligible matrix");
        let mut at = 0usize;
        for e in &entries {
            let t = g.tensor(&e.name).expect("the matrix is in the file");
            assert!(eligible(t), "{fmt:?} {}: not eligible", e.name);
            assert_eq!(e.offset, at, "{fmt:?} {}: laid out twice", e.name);
            assert_eq!(e.len, packed_len(t.kind, e.rows, e.cols));
            check_matrix(t.kind, g.bytes(t), &store[at..at + e.len], e.rows, e.cols);
            at += e.len;
        }
        // Nothing else is in the store: not the embedding, not the norms.
        assert_eq!(at, store.len(), "{fmt:?}: the store holds something else");

        // The same file, loaded without the pack, is multiplied the ordinary way.
        let plain = Model::load_with_kq_fp(
            Gguf::open(&path).expect("the model opens"),
            peregrine_models::FAMILIES,
            true,
            false,
        )
        .expect("the synthetic model loads");
        assert!(
            plain.fp().is_empty(),
            "{fmt:?}: the plain model has a store"
        );
        let a = prefill(&plain);
        let b = prefill(&model);
        assert_eq!(a.len(), b.len());
        if packed_here {
            // The packed kernel ran: the logits are not the old path's bits.
            assert!(
                a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits()),
                "{fmt:?}: the packed path did not change a logit — was it used?"
            );
            // The same real number with a different rounding: one FMA chain
            // instead of four partial sums and a fold. A random-weight model does
            // not amplify the few-ulp difference; a trained model moves its logits
            // by about 1e-6 in cosine (docs/design/s4.9-sota.md, S4.9f as built).
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-3 * x.abs().max(1.0),
                    "{fmt:?}: logit {i} moved: {x} against {y}"
                );
            }
        } else {
            // No packed kernel on this target: the store is built and never read,
            // so the forward is the ordinary one, bit for bit.
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{fmt:?}: logit {i} moved without a kernel: {x} against {y}"
                );
            }
        }
    }
}
