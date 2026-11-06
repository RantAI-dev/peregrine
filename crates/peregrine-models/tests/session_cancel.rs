//! The session's cancellation, on the synthetic models (no model file needed).
//!
//! A cancelled turn is not a failure: it returns what it produced, marked
//! `cancelled`, and the conversation stays consistent for the next message.
//! The session runs on a one-slot engine now (ADR 0007, decision 4), so the
//! cancellation is the engine's: the slot goes, and the conversation keeps the
//! cache it had.
//!
//! Two cases are pinned here:
//!
//! * a cancel during the decode keeps the conversation: the tokens the turn
//!   produced are in it, and the next message is answered from it;
//! * a cancel before anything was produced leaves the conversation exactly as
//!   it was: the turn's prompt is not in it.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::session::{Session, SessionCfg};
use peregrine_core::telem::TurnTelem;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use support::models::{write_dense, Dense};

/// The prompts the test asks. The synthetic model's answer differs per kernel
/// tier, and it sometimes opens with its end-of-turn token — a turn with no
/// tokens has nothing to cancel.
const PROMPTS: [&str; 8] = [
    "Name three colours.",
    "Name three fruits.",
    "Name three animals.",
    "Name a colour.",
    "Name a fruit.",
    "Name an animal.",
    "Name three birds.",
    "Name three trees.",
];

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-cancel-{}-{test}", std::process::id()));
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

/// A turn that is cancelled as soon as it says something.
///
/// The prompts are tried until one of them produces a token: the turn has to
/// say something for the cancellation to have anything to keep.
fn cancelled_turn(s: &mut Session) -> (String, TurnTelem) {
    for user in PROMPTS {
        let stop = AtomicBool::new(false);
        let mut seen = 0usize;
        let turn = s
            .generate_with(user, Some(&stop), |_| {
                seen += 1;
                if seen == 1 {
                    stop.store(true, Ordering::Relaxed);
                }
            })
            .expect("a cancelled turn is not an error");
        if seen > 0 {
            return turn;
        }
    }
    panic!("the model never says anything: there is nothing to cancel");
}

/// A cancel in the middle of a turn keeps the conversation: the next message
/// is answered from it, and the exchange is in the history.
#[test]
fn a_cancelled_turn_keeps_the_conversation() {
    let dir = tmpdir("mid");
    let model = load(&write_dense(
        &dir,
        "qwen3",
        Format::Q8_0,
        7100,
        Dense::default(),
    ));
    let mut s = session(&model);

    let (text, telem) = cancelled_turn(&mut s);
    assert!(telem.cancelled, "the turn must be marked cancelled");
    // The flag is checked before every step, so at most one more step runs.
    assert!(telem.n() <= 2, "kept generating: {} tokens", telem.n());
    assert!(!text.is_empty(), "the turn produced nothing to keep");

    // The conversation keeps it: the next turn is answered from the context
    // the cancelled turn left.
    let used = s.context_used();
    assert!(
        used > 0,
        "the cancelled turn left nothing in the conversation"
    );
    let exchanges = s.exchanges();
    let (_, next) = s.generate("Name three fruits.", |_| {}).expect("next turn");
    assert!(!next.cancelled);
    assert!(s.context_used() > used, "the conversation did not grow");
    assert_eq!(
        s.exchanges(),
        exchanges + 1,
        "the next turn is not in the history"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A cancel before the turn produced anything leaves the conversation exactly
/// as it was: the prompt of the cancelled turn is not in it.
#[test]
fn a_cancelled_turn_before_any_output_leaves_the_conversation_alone() {
    let dir = tmpdir("early");
    let model = load(&write_dense(
        &dir,
        "qwen3",
        Format::Q8_0,
        7100,
        Dense::default(),
    ));
    let mut s = session(&model);

    // One turn first, so the conversation holds something.
    s.generate("Name three colours.", |_| {})
        .expect("first turn");
    let used = s.context_used();
    let exchanges = s.exchanges();

    // The flag is already set: the turn stops before it says anything.
    let stop = AtomicBool::new(true);
    let (text, telem) = s
        .generate_with("Name three fruits.", Some(&stop), |_| {})
        .expect("a cancelled turn is not an error");
    assert!(telem.cancelled);
    assert_eq!(text, "", "the turn said something");
    assert_eq!(telem.n(), 0);
    assert_eq!(
        s.context_used(),
        used,
        "the cancelled turn left its prompt in the conversation"
    );

    // The conversation still answers.
    let (_, next) = s
        .generate("Name three animals.", |_| {})
        .expect("next turn");
    assert!(!next.cancelled);
    assert_eq!(
        s.exchanges(),
        exchanges + 1,
        "the cancelled turn left an exchange in the history"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
