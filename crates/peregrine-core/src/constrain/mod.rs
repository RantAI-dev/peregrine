//! Constrained decoding: the sampler may only pick tokens that keep the reply a valid
//! prefix of a constraint (a JSON schema, a literal sequence, free text up to a terminator,
//! or a combination).
//!
//! The constraint is applied between the logits and the existing sampler: the forward pass
//! is untouched, and a request without a constraint takes exactly the path it took before.
//! See `docs/design/s6.6-constrained-decoding.md`.
//!
//! # Building a constraint
//!
//! ```
//! use peregrine_core::constrain::{Constraint, Part, Schema};
//! use peregrine_core::json::Value;
//!
//! let schema = Schema::new(Value::parse(r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#).unwrap());
//! // A made-up call syntax: the real tool format is the caller's business.
//! let call = Constraint::sequence(vec![
//!     Part::Literal("<<A>>".into()),
//!     Part::Json(schema),
//!     Part::Literal("<<B>>".into()),
//! ]);
//! // Several calls in one reply, and one of two shapes.
//! let several = Constraint::repeat(call.clone(), 1, Some(3));
//! let either = Constraint::any_of(vec![call.clone(), several]);
//! // Free text up to a terminator (a thinking block), then the call.
//! let after_thinking = Constraint::concat(vec![
//!     Constraint::sequence(vec![Part::Until("</think>".into()), Part::Literal("\n\n".into())]),
//!     call,
//! ]);
//! ```
//!
//! Attach one to a generation request with `SlotCfg::constraint`; the server's
//! `response_format` builds one from the request.

mod mask;
mod matcher;
mod schema;

pub use mask::{TokenMask, Vocabulary};
pub use matcher::Matcher;
pub use schema::{Constraint, Part, Schema, SchemaError, MAX_FINITE_INT_VALUES};
