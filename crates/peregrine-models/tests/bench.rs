//! `Session::bench`: the measurement runs on the engine (ADR 0007), and it
//! leaves the conversation it measured around alone.
//!
//! The benchmark used to walk a prefill and a decode loop of its own, next to
//! the runtime the server runs. Now it admits a slot and lets the engine do the
//! work, so the numbers it reports are the numbers the engine charges a
//! request — and a conversation that runs a benchmark afterwards must be the
//! same conversation it was.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::model::Model;
use peregrine_core::session::{Session, SessionCfg};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{write_dense, Dense};

/// The turn the conversation makes.
const USER: &str = "Name three colours.";
/// The tokens a turn may produce.
const MAX_TOKENS: usize = 12;
/// The seed every run samples with.
const SEED: u64 = 20240;
/// The threads every run uses.
const THREADS: usize = 4;

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-bench-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &Path) -> Arc<Model> {
    Arc::new(Gguf::open(path).and_then(peregrine_models::load).unwrap())
}

fn session_cfg() -> SessionCfg {
    SessionCfg {
        max_ctx: 512,
        max_tokens: MAX_TOKENS,
        threads: THREADS,
        temperature: 0.0,
        seed: SEED,
        ..Default::default()
    }
}

/// One turn's tokens.
fn turn(s: &mut Session) -> Vec<u32> {
    let mut ids = Vec::new();
    s.generate(USER, |e| ids.push(e.id)).expect("generate");
    ids
}

/// **ADR 0007, decision 1: the engine is the one runtime.**
///
/// The benchmark reports what the engine spent: a prefill, a decode, and the
/// bytes a token moved. A report of zeroes would mean the measurement never
/// ran, so the numbers have to be real.
#[test]
fn the_benchmark_measures_the_engine() {
    let dir = tmpdir("measure");
    let path = write_dense(&dir, "qwen3", Format::Q8_0, 7200, Dense::default());
    let model = load(&path);
    let mut s = Session::from_model(model, session_cfg()).expect("session");

    let report = s.bench(64, 8).expect("bench");
    assert_eq!(report.prompt_tokens, 64);
    assert_eq!(report.decode_tokens, 8);
    assert!(
        report.prefill_s > 0.0,
        "the prefill took no time: {report:?}"
    );
    assert!(report.decode_s > 0.0, "the decode took no time: {report:?}");
    assert!(
        report.prefill_tok_s > 0.0,
        "the prefill measured no tokens: {report:?}"
    );
    assert!(
        report.decode_tok_s > 0.0,
        "the decode measured no tokens: {report:?}"
    );
    assert!(
        report.bytes_per_token > 0.0,
        "the decode moved no bytes: {report:?}"
    );

    // The same measurement again: the engine is left clean, so the second run
    // measures the same thing as the first.
    let again = s.bench(64, 8).expect("bench again");
    assert!(again.prefill_s > 0.0, "the second prefill took no time");
    assert!(again.decode_s > 0.0, "the second decode took no time");

    let _ = std::fs::remove_dir_all(&dir);
}

/// **ADR 0007, decision 4: the session is a layer over the engine.**
///
/// A benchmark is a measurement, not a conversation: it must not change what the
/// conversation says next. Two sessions that have had the same first turn must
/// still have the same second turn, whether or not a benchmark ran in between.
#[test]
fn the_benchmark_leaves_the_conversation_alone() {
    let dir = tmpdir("conversation");
    let path = write_dense(&dir, "qwen3", Format::Q8_0, 7201, Dense::default());
    let model = load(&path);

    // The reference: two turns, no benchmark.
    let mut plain = Session::from_model(model.clone(), session_cfg()).expect("session");
    turn(&mut plain);
    let plain_second = turn(&mut plain);

    // The same conversation, with a benchmark in between.
    let mut measured = Session::from_model(model, session_cfg()).expect("session");
    turn(&mut measured);
    measured.bench(32, 4).expect("bench");
    let measured_second = turn(&mut measured);

    assert_eq!(
        plain_second, measured_second,
        "the benchmark changed what the conversation said next"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
