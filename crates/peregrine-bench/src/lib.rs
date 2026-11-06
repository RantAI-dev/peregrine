//! Benchmarks and one-off experiments for the Peregrine engine, as examples of this crate:
//!
//! ```sh
//! cargo run --release -p peregrine-bench --example kbench -- <args>
//! ```
//!
//! Most are x86-64 only (they call the SIMD kernels directly). Nothing depends on this crate
//! (docs/dev/structure-rules.md); the benchmark harness (roadmap S3.1) will live here too.
