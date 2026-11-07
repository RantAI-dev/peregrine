//! The session's own behaviour, pinned before it moves onto the engine (ADR 0007, S1.4 step 6).
//!
//! `Session` becomes a conversation layer over a one-slot `Engine` (ADR 0007,
//! decision 4). The tokens it produces must not move while it moves: this file
//! records what a three-turn conversation produces today, on the synthetic
//! models (no model file needed), and the test asserts it.
//!
//! Kernel tiers round differently, so the recording is per tier, as the goldens
//! are: `tests/fixtures/session/<tier>.txt`. A tier with no recording is
//! skipped, and says so.
//!
//!   PEREGRINE_SESSION_FIXTURE=record cargo test --release --test session_fixture
//!   PEREGRINE_ISA=scalar PEREGRINE_SESSION_FIXTURE=record cargo test --release --test session_fixture

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::model::Model;
use peregrine_core::session::{Session, SessionCfg};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{write_dense, write_hybrid, Dense};

/// The conversation: three turns, so the session's own turn counter is in the
/// sampling key of the second and the third.
const TURNS: [&str; 3] = [
    "Name three colours.",
    "Name three fruits.",
    "Name three animals.",
];
/// The tokens a turn may produce.
const MAX_TOKENS: usize = 12;
/// The seed every run samples with.
const SEED: u64 = 20240;
/// The threads every run uses: the recording must not depend on the machine.
const THREADS: usize = 4;
/// The temperature the sampled case runs at.
const TEMP: f32 = 0.7;
/// The suffix drafter's own settings, as the session's defaults have them.
const NGRAM: usize = 2;
const DRAFT_LEN: usize = 12;

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-session-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &Path) -> Arc<Model> {
    Arc::new(Gguf::open(path).and_then(peregrine_models::load).unwrap())
}

/// The session's settings for one case.
fn session_cfg(temperature: f32, suffix: bool) -> SessionCfg {
    SessionCfg {
        max_ctx: 512,
        max_tokens: MAX_TOKENS,
        threads: THREADS,
        temperature,
        top_p: 0.95,
        top_k: 0,
        min_p: 0.0,
        seed: SEED,
        // The drafter: the suffix index, or none at all.
        speculate: suffix,
        ngram: NGRAM,
        draft_len: DRAFT_LEN,
        thinking: true,
        ..Default::default()
    }
}

/// One configuration the fixture records.
struct Case {
    /// The name in the fixture file.
    name: &'static str,
    /// The model the case runs on: the dense one or the hybrid one.
    model: &'static str,
    temperature: f32,
    /// The suffix drafter is on.
    suffix: bool,
}

/// The cases, in the order they are recorded.
fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "qwen3-Q8_0-T0-plain",
            model: "dense",
            temperature: 0.0,
            suffix: false,
        },
        Case {
            name: "qwen3-Q8_0-T0-suffix",
            model: "dense",
            temperature: 0.0,
            suffix: true,
        },
        Case {
            name: "qwen3-Q8_0-T0.7-plain",
            model: "dense",
            temperature: TEMP,
            suffix: false,
        },
        Case {
            name: "qwen3-Q8_0-T0.7-suffix",
            model: "dense",
            temperature: TEMP,
            suffix: true,
        },
        // A hybrid model's recurrent state cannot be rolled back, so it never
        // speculates: the session turns the suffix drafter off for it.
        Case {
            name: "qwen35-PQ2_0-T0-plain",
            model: "hybrid",
            temperature: 0.0,
            suffix: false,
        },
        Case {
            name: "qwen35-PQ2_0-T0.7-plain",
            model: "hybrid",
            temperature: TEMP,
            suffix: false,
        },
    ]
}

/// The token ids one case produces, turn by turn, and how many drafts it kept.
fn run(model: &Arc<Model>, case: &Case) -> (Vec<Vec<u32>>, u64, u64) {
    let mut s = Session::from_model(model.clone(), session_cfg(case.temperature, case.suffix))
        .expect("session");
    let mut turns = Vec::new();
    let mut drafted = 0u64;
    let mut accepted = 0u64;
    for user in TURNS {
        let mut ids = Vec::new();
        let (_, telem) = s.generate(user, |e| ids.push(e.id)).expect("generate");
        drafted += telem.drafted();
        accepted += telem.accepted();
        turns.push(ids);
    }
    (turns, drafted, accepted)
}

/// The fixture file: one line per case and turn.
fn render(tier: &str, rows: &[(String, usize, Vec<u32>)]) -> String {
    let mut out = String::from("# peregrine session fixture v1\n");
    out.push_str(&format!("# tier {tier}\n"));
    for (case, turn, ids) in rows {
        let ids: Vec<String> = ids.iter().map(u32::to_string).collect();
        out.push_str(&format!("{case} turn {turn} {}\n", ids.join(",")));
    }
    out
}

/// The fixture file's rows: (case, turn, ids).
fn parse(text: &str) -> Vec<(String, usize, Vec<u32>)> {
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let case = it.next().expect("a case name").to_string();
        assert_eq!(it.next(), Some("turn"), "a turn number: {line}");
        let turn: usize = it
            .next()
            .expect("a turn number")
            .parse()
            .expect("a turn number");
        let ids: Vec<u32> = it
            .next()
            .unwrap_or("")
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().expect("a token id"))
            .collect();
        rows.push((case, turn, ids));
    }
    rows
}

/// **ADR 0007, decision 5: no output change in S1.4.**
///
/// The session's token streams, recorded before it moves onto the engine. A
/// conversation that produces different tokens after the move has changed what
/// the engine says, which S1.4 does not allow.
#[test]
fn the_session_s_token_streams_are_the_recorded_ones() {
    let dir = tmpdir("fixture");
    let dense = load(&write_dense(
        &dir,
        "qwen3",
        Format::Q8_0,
        7100,
        Dense::default(),
    ));
    let hybrid = load(&write_hybrid(&dir, Format::PQ2_0, 7101, false));

    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for case in cases() {
        let model = if case.model == "dense" {
            &dense
        } else {
            &hybrid
        };
        let (turns, drafted, accepted) = run(model, &case);
        for (i, ids) in turns.iter().enumerate() {
            rows.push((case.name.to_string(), i + 1, ids.clone()));
        }
        eprintln!(
            "session fixture: {} — {} drafts verified, {} kept",
            case.name, drafted, accepted
        );
        // The suffix drafter must actually have engaged, or the case records
        // nothing about speculation at all.
        if case.suffix && drafted == 0 {
            failures.push(format!(
                "{}: the suffix drafter never proposed anything, so the case \
                 records no speculation",
                case.name
            ));
        }
        // A turn may end on its first token (the model opens with the
        // end-of-turn token): that is behaviour worth pinning, not a fault.
        for (i, ids) in turns.iter().enumerate() {
            if ids.is_empty() {
                eprintln!(
                    "session fixture: {} turn {} ended on its first token",
                    case.name,
                    i + 1
                );
            }
        }
    }

    let tier = peregrine_core::cpu::tier_name();
    let text = render(&tier, &rows);
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/session")
        .join(format!("{tier}.txt"));
    let record = std::env::var("PEREGRINE_SESSION_FIXTURE").is_ok_and(|v| v == "record");
    if record {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &text).unwrap();
        eprintln!("session fixture: recorded {}", path.display());
    } else if !path.is_file() {
        eprintln!(
            "session fixture: skip (no recording for tier {tier}; record one with \
             PEREGRINE_SESSION_FIXTURE=record)"
        );
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            failures.is_empty(),
            "the session misbehaved:\n{}",
            failures.join("\n")
        );
        return;
    } else {
        let want = parse(&std::fs::read_to_string(&path).unwrap());
        for (got, wot) in rows.iter().zip(want.iter()) {
            if got != wot {
                failures.push(format!(
                    "{} turn {}: the tokens moved\n  recorded: {:?}\n  now:      {:?}",
                    got.0, got.1, wot.2, got.2
                ));
            }
        }
        if rows.len() != want.len() {
            failures.push(format!(
                "the recording has {} rows, this run has {}",
                want.len(),
                rows.len()
            ));
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the session's behaviour moved:\n{}",
        failures.join("\n")
    );
}
