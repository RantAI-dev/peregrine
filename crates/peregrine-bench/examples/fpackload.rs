//! Time the load-time float pack (S4.9f) on one file.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example fpackload -- MODEL.gguf
//! ```
//!
//! Reports the matrix directory's scan, the store's layout and size, and the repack
//! itself — the three parts of a start-up that `PEREGRINE_FPACK=0` avoids.

use peregrine_core::gguf::Gguf;
use peregrine_core::model::FpStore;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("a model file");
    let t0 = Instant::now();
    let mut g = Gguf::open(std::path::Path::new(&path)).expect("the model opens");
    println!("open     {:>8.3?}", t0.elapsed());

    let t1 = Instant::now();
    let layout = peregrine_core::fpack::layout(&g);
    let total: usize = layout.iter().map(|e| e.len).sum();
    println!(
        "layout   {:>8.3?}  {} matrices, {:.1} MiB",
        t1.elapsed(),
        layout.len(),
        total as f64 / (1u64 << 20) as f64
    );

    let t2 = Instant::now();
    let store = peregrine_core::fpack::repack(&mut g);
    println!(
        "repack   {:>8.3?}  store {:.1} MiB",
        t2.elapsed(),
        store.bytes().len() as f64 / (1u64 << 20) as f64
    );

    // First touch of the store, which the forward would pay on its first pass.
    let t3 = Instant::now();
    let mut s = 0u8;
    for b in store.bytes().iter().step_by(4096) {
        s = s.wrapping_add(*b);
    }
    println!("touch    {:>8.3?}  ({s})", t3.elapsed());
    let _ = FpStore::new(1);
}
