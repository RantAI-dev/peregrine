//! `peregrine inspect` names a decision model (roadmap S4.3).
//!
//! The metadata itself is pinned in `peregrine-models/tests/decision_meta.rs`; this checks the one
//! line `inspect` prints for a file that declares a decision type. The template's bytes are not
//! printed, only their length.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use std::path::PathBuf;
use support::models::{write_dense_extra, DecisionSpec, Dense};

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-inspect-decision-{}-{test}",
        std::process::id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn inspect_reports_the_decision_type_the_temperatures_and_the_template_length() {
    let dir = tmpdir("line");
    let mut spec = DecisionSpec::new("lev");
    spec.temperatures = vec![
        ("choice".to_string(), 0.5),
        ("choice.small".to_string(), 0.25),
    ];
    let template = spec.template.clone().unwrap();
    let path = write_dense_extra(
        &dir,
        "llama",
        Format::F16,
        4,
        Dense::default(),
        &spec.keys("llama"),
    );
    let text = peregrine::inspect(&path).expect("inspect");
    let want = format!(
        "decision      lev (2 temperatures, template {} bytes)",
        template.len()
    );
    assert!(text.contains(&want), "missing {want:?} in:\n{text}");
    assert!(
        !text.contains(&template),
        "the template text leaked into inspect:\n{text}"
    );
}

#[test]
fn a_model_without_a_decision_type_prints_no_decision_line() {
    let dir = tmpdir("none");
    let path = write_dense_extra(&dir, "llama", Format::F16, 4, Dense::default(), &[]);
    let text = peregrine::inspect(&path).expect("inspect");
    assert!(!text.contains("decision      "), "{text}");
}
