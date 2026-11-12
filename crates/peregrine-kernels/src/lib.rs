//! Peregrine's CPU kernels: every (weight format, instruction-set tier) pair, the scalar references they
//! are tested against, activation quantisation and vector math. All SIMD `unsafe` code lives here
//! (docs/adr/0005-modular-structure.md). Kernel tiers are chosen at run time (`cpu`), capped by
//! `PEREGRINE_ISA`.

// The workspace builds with `-D warnings`; these idioms are kept as they are.
// `chunks_exact_to_as_chunks` exists only in newer clippy (CI's stable is ahead of older local toolchains).
#![allow(
    unknown_lints,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_map_or,
    clippy::chunks_exact_to_as_chunks
)]

pub mod bidir;
pub mod cpu;
pub mod delta;
pub mod format;
pub mod fpack;
pub mod hadamard;
pub mod kq;
pub mod kq8;
#[cfg(target_arch = "x86_64")]
pub mod kqsimd;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod options;
pub mod p32;
pub mod prof;
pub mod quant;
#[cfg(target_arch = "x86_64")]
pub mod simd;
pub mod table;
#[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
pub mod tern;
pub mod vmath;

pub use format::{f16, Format};
pub use table::kernels_for;
