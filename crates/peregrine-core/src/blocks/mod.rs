//! Layer building blocks shared by the model families (ADR 0006).
//!
//! Today this holds the worker hand-over types (`par`) and the attention scratch (`attn`); the shared layer
//! computations move here in roadmap S1.2 step 3.

pub(crate) mod attn;
pub(crate) mod par;
pub(crate) mod recurrent;
pub(crate) mod rotate;
