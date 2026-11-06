//! Layer building blocks shared by the model families (ADR 0006).
//!
//! # Unstable interface
//!
//! Items in this module are exposed for model families (ADR 0005: the interface is
//! unstable until 1.0).
//!
//! Holds the worker hand-over types (`par`), the attention scratch (`attn`), and the shared layer
//! computations moved here in roadmap S1.2 step 3.

pub mod attn;
pub mod embed;
pub mod ffn;
pub mod linear;
pub mod norm;
pub mod par;
pub mod recurrent;
pub mod rope;
pub mod rotate;

/// Batch rows at which per-row work (norm, SwiGLU, prep) is split across the pool.
pub const ROW_PARALLEL: usize = 2;
