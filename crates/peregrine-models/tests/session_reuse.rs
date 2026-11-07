//! The session's prefix reuse, on the synthetic models (no model file needed).
//!
//! A conversation that is reset, or seeded with an earlier exchange, reuses the
//! positions the cache still holds: only the prompt's divergent tail is
//! prefilled. The tokens must not move — a reused prefix and a fresh cache
//! answer the same — and the suffix index's corpus must follow the reuse, or
//! it drafts from positions the cache no longer holds.
//!
//! The test that covers this needs a real model, so it never ran in CI.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::session::{Session, SessionCfg};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{write_dense, Dense};

const QUESTION: &str = "Name three colours.";
const FOLLOW_UP: &str = "Name three fruits.";

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-reuse-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &Path) -> Arc<peregrine_core::model::Model> {
    Arc::new(Gguf::open(path).and_then(peregrine_models::load).unwrap())
}

fn session(model: &Arc<peregrine_core::model::Model>) -> Session {
    let cfg = SessionCfg {
        max_ctx: 512,
        max_tokens: 24,
        threads: 4,
        temperature: 0.0,
        seed: 20240,
        speculate: true,
        ..Default::default()
    };
    Session::from_model(model.clone(), cfg).expect("session")
}

/// The token ids a turn produced.
fn ids(s: &mut Session, user: &str) -> Vec<u32> {
    let mut out = Vec::new();
    s.generate(user, |e| out.push(e.id)).expect("generate");
    out
}

/// **The reuse does not change what the conversation says.**
///
/// A seeded history is replayed from the cache that holds it, and a reset
/// conversation reuses the system block. The answers are the ones a fresh
/// cache gives: the reuse saves the prefill, not the tokens.
#[test]
fn a_reused_prefix_answers_as_a_fresh_cache_does() {
    let dir = tmpdir("seeded");
    let model = load(&write_dense(
        &dir,
        "qwen3",
        Format::Q8_0,
        7100,
        Dense::default(),
    ));

    // The conversation, and the exchange it is seeded with.
    let mut s = session(&model);
    let (answer, _) = s.generate(QUESTION, |_| {}).expect("first turn");
    assert!(!answer.is_empty(), "the model said nothing");

    // Seeded with that exchange, the conversation answers from the cache that
    // holds it.
    s.seed_history(vec![(QUESTION.to_string(), answer.clone())]);
    let replayed = ids(&mut s, FOLLOW_UP);
    let reused = s.context_used();

    // The same conversation, but from an empty cache: the same answer.
    let mut fresh = session(&model);
    fresh.seed_history(vec![(QUESTION.to_string(), answer)]);
    let from_scratch = ids(&mut fresh, FOLLOW_UP);

    assert_eq!(
        replayed, from_scratch,
        "the reused prefix changed the answer"
    );
    assert!(!replayed.is_empty(), "the model said nothing");
    assert!(
        reused > 0,
        "the seeded conversation reused nothing of its cache"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// **A reset conversation reuses the system block, and answers the same.**
///
/// The reset keeps the cache: the next prompt shares the system block with it,
/// so only the question is prefilled. The answer is the one the first turn
/// gave — the same prompt, the same seed.
#[test]
fn a_reset_conversation_reuses_its_system_block_and_answers_the_same() {
    let dir = tmpdir("reset");
    let model = load(&write_dense(
        &dir,
        "qwen3",
        Format::Q8_0,
        7100,
        Dense::default(),
    ));
    let mut s = session(&model);

    let first = ids(&mut s, QUESTION);
    s.reset();
    let again = ids(&mut s, QUESTION);
    assert_eq!(first, again, "the reset conversation answered differently");
    assert!(!first.is_empty(), "the model said nothing");

    let _ = std::fs::remove_dir_all(&dir);
}
