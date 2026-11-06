//! Test support: tiny random GGUF models written at test time (roadmap S0.2).
//!
//! CI has no real model files, so these models are how the model code paths, every weight format and
//! every kernel tier get exercised there. Nothing here is committed as a binary: each test writes its
//! models into a temporary directory.
#![allow(dead_code)]

pub mod models;
pub mod vocab;
pub mod weights;
pub mod writer;
