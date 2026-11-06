//! Re-exports for MTP blocks (moved to `crate::families::qwen35::mtp` and `crate::arch` in S1.2 step 4).

pub use crate::arch::{MtpHead, MtpState, MTP_MAX_ROWS};
pub use crate::families::qwen35::mtp::Mtp;
