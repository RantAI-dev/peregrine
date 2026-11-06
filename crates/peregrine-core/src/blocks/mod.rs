//! Layer building blocks shared by the model families (ADR 0006).
//!
//! Today this holds the worker hand-over types (`par`), the attention scratch (`attn`), and the shared layer
//! computations moved here in roadmap S1.2 step 3.

pub(crate) mod attn;
pub(crate) mod embed;
pub(crate) mod ffn;
pub(crate) mod linear;
pub(crate) mod norm;
pub(crate) mod par;
pub(crate) mod recurrent;
pub(crate) mod rope;
pub(crate) mod rotate;

/// Batch rows at which per-row work (norm, SwiGLU, prep) is split across the pool.
pub(crate) const ROW_PARALLEL: usize = 2;
