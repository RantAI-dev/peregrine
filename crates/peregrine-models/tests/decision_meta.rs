//! Decision metadata read from a model file (roadmap S4.3).
//!
//! `DecisionInfo::from_gguf` is llama.cpp's `server_decision_context` constructor as far as the
//! answer math needs it (`docs/reference/systemone.md` §1.1, §2.1, §2.2): the type, every
//! temperature, every score bias, the labels and the presence of the `systemone` template. These
//! tests write synthetic llama files with those keys and check the round trip and each refusal.

mod support;

use peregrine_core::decision::{DecisionInfo, DecisionType};
use peregrine_core::gguf::{Format, Gguf};
use std::path::PathBuf;
use support::models::{write_dense_extra, DecisionSpec, Dense};

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-decision-meta-{}-{test}",
        std::process::id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn read(dir: &std::path::Path, spec: &DecisionSpec) -> Result<Option<DecisionInfo>, String> {
    let path = write_dense_extra(
        dir,
        "llama",
        Format::F16,
        3,
        Dense::default(),
        &spec.keys("llama"),
    );
    let g = Gguf::open_plain(&path).expect("open the synthetic model");
    DecisionInfo::from_gguf(&g)
}

#[test]
fn every_decision_key_round_trips() {
    let dir = tmpdir("round-trip");
    let mut spec = DecisionSpec::new("lev");
    spec.temperatures = vec![
        ("choice".to_string(), 0.625),
        ("choice.small".to_string(), 0.375),
        ("score".to_string(), 2.0),
        ("noul.mid".to_string(), 0.75),
    ];
    spec.score_bias = vec![((3, 0), 0.5), ((3, 2), -1.5), ((5, 4), 0.25)];
    spec.labels = Some(vec![
        "no".to_string(),
        "maybe".to_string(),
        "yes".to_string(),
    ]);
    let template = spec.template.clone().unwrap();
    let info = read(&dir, &spec).unwrap().expect("a decision model");

    assert_eq!(info.kind(), &DecisionType::Lev);
    assert_eq!(info.temperatures().len(), 4);
    assert_eq!(info.temperatures()["choice.small"], 0.375);
    assert_eq!(info.temperatures()["noul.mid"], 0.75);
    assert_eq!(info.score_bias()[&3], vec![0.5, 0.0, -1.5]);
    assert_eq!(info.score_bias()[&5], vec![0.0, 0.0, 0.0, 0.0, 0.25]);
    assert_eq!(info.labels(), ["no", "maybe", "yes"]);
    assert_eq!(info.template_len(), template.len());

    // `Debug` reports the template as a length, never as text.
    let debug = format!("{info:?}");
    assert!(debug.contains("template_bytes"), "{debug}");
    assert!(
        !debug.contains(&template),
        "the template text leaked: {debug}"
    );
}

#[test]
fn a_file_without_a_decision_type_is_not_a_decision_model() {
    let dir = tmpdir("none");
    let mut spec = DecisionSpec::new("lev");
    spec.type_name = None;
    assert!(read(&dir, &spec).unwrap().is_none());

    // A model with no architecture at all is not one either.
    let path = write_dense_extra(&dir, "llama", Format::F16, 3, Dense::default(), &[]);
    let g = Gguf::open_plain(&path).unwrap();
    let mut g = g;
    g.meta.remove("general.architecture");
    assert!(DecisionInfo::from_gguf(&g).unwrap().is_none());
}

#[test]
fn an_unknown_type_and_a_missing_template_are_refused() {
    let dir = tmpdir("refusals");
    let mut unknown = DecisionSpec::new("a-model-from-the-future");
    let err = read(&dir, &unknown).unwrap_err();
    assert!(
        err.contains("unsupported decision model type: a-model-from-the-future"),
        "{err}"
    );
    // The type is still reported by name before the refusal.
    unknown.type_name = Some("h2o-lightning".into());
    assert!(read(&dir, &unknown).is_ok());

    let mut no_template = DecisionSpec::new("openjev");
    no_template.template = None;
    let err = read(&dir, &no_template).unwrap_err();
    assert!(err.contains("no \"systemone\" template"), "{err}");
}

#[test]
fn a_non_positive_temperature_is_refused() {
    let dir = tmpdir("temperature");
    for bad in [0.0f32, -0.5] {
        let mut spec = DecisionSpec::new("openjev");
        spec.temperatures = vec![("choice".to_string(), bad)];
        let err = read(&dir, &spec).unwrap_err();
        assert!(err.contains("invalid decision temperature"), "{bad}: {err}");
        assert!(err.contains("choice"), "the key is named: {err}");
    }
}

#[test]
fn a_malformed_score_bias_is_refused() {
    let dir = tmpdir("bias");
    // The number of levels must be 2..=255 and the level inside it.
    for ((n, level), what) in [
        ((1, 0), "one level"),
        ((256, 0), "256 levels"),
        ((3, 3), "level == n"),
        ((3, 9), "level > n"),
    ] {
        let mut spec = DecisionSpec::new("vllm-decision2");
        spec.score_bias = vec![((n, level), 0.5)];
        let err = read(&dir, &spec).unwrap_err();
        assert!(err.contains("invalid decision score bias"), "{what}: {err}");
        assert!(
            err.contains(&format!("score_bias.{n}.{level}")),
            "{what}: {err}"
        );
    }
}

#[test]
fn the_sixteen_known_types_are_kept_and_an_unknown_one_keeps_its_name() {
    for name in [
        "openjev",
        "lev",
        "kev",
        "nimble",
        "laya",
        "clef",
        "pplx-decider",
        "vllm-decision2",
        "vllm-d3",
        "lfm2-d1",
        "lfm2-d1-omni",
        "granite-switch",
        "h2o-lightning",
        "openbmb-minicpm-ocr-jev",
        "lfm3-d1",
        "lfm3-d1-omni",
    ] {
        assert_eq!(DecisionType::from_name(name).name(), name);
        assert!(!matches!(
            DecisionType::from_name(name),
            DecisionType::Unknown(_)
        ));
    }
}
