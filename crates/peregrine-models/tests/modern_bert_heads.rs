//! The `modern-bert` family's classifier head (roadmap S4.7b): rerankers and
//! classifiers.
//!
//! A `modern-bert` file may carry `cls` (the dense layer, bias-less), `cls.norm` (a
//! weight-only LayerNorm), `cls.output` (the projection to the file's outputs) and
//! `classifier.output_labels`, with `classifier.activation` (erf GELU by default) and
//! `classifier.pooling_type` (mean by default). The head's row must be exactly the
//! reference forward's (`tests/support/modern_bert_ref.rs`), and it must be
//! bit-identical whatever else shares the pass: alone, in a batch of 2, 7 or 64 texts,
//! at any pass size and thread count. These run on synthetic files, so CI needs no
//! model file.

mod support;

use peregrine_core::gguf::Format;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::Model;
use peregrine_core::serve::{Engine, Finish, SlotCfg};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use support::models::{write_modern_bert, ModernBertConfig, ModernBertHeadConfig};

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-modern-bert-heads-{}-{test}",
        std::process::id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &std::path::Path) -> Arc<Model> {
    let g = peregrine_core::Gguf::open(path).expect("open the synthetic model");
    Arc::new(peregrine_models::load(g).expect("load the synthetic model"))
}

/// A text of token ids inside the synthetic vocabulary (all ids are valid).
fn modern_bert_text(seed: u32, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| 7 + (seed.wrapping_mul(53).wrapping_add(i as u32 * 17)) % 280)
        .collect()
}

/// A synthetic `modern-bert` file with the given head.
fn head_model(
    dir: &std::path::Path,
    fmt: Format,
    seed: u64,
    name: &str,
    head: ModernBertHeadConfig,
) -> Arc<Model> {
    let path = write_modern_bert(
        dir,
        fmt,
        seed,
        ModernBertConfig {
            head: Some(head),
            ..Default::default()
        },
    );
    let to = dir.join(format!("{name}.gguf"));
    std::fs::rename(&path, &to).unwrap();
    load(&to)
}

/// A reranker's head: one output, no labels, the family's defaults (mean pooling, erf
/// GELU, no `classifier.*` keys at all).
fn rank_head() -> ModernBertHeadConfig {
    ModernBertHeadConfig {
        n_out: 1,
        ..Default::default()
    }
}

/// A labelled classifier: three outputs named by `classifier.output_labels`, the tanh
/// activation and row 0 (cls) pooling, so both non-default keys are exercised.
fn labelled_head() -> ModernBertHeadConfig {
    ModernBertHeadConfig {
        n_out: 3,
        labels: vec!["negative".into(), "neutral".into(), "positive".into()],
        activation: Some("tanh"),
        pooling_type: Some(2),
        ..Default::default()
    }
}

/// The head rows of a batch of texts, through the engine (`Want::Head` slots).
fn head_batch(
    model: &Arc<Model>,
    texts: &[Vec<u32>],
    threads: usize,
    chunk: usize,
    pass_rows: Option<usize>,
) -> Vec<Vec<f32>> {
    let mut engine = Engine::new_with_chunk_and_pass(
        model.clone(),
        threads,
        model.cfg.ctx_train,
        texts.len().max(1),
        KvPrec::F16,
        0,
        chunk,
        pass_rows,
    );
    let mut by_slot: HashMap<u64, usize> = HashMap::new();
    for (i, t) in texts.iter().enumerate() {
        let id = engine.admit_head(t.clone(), SlotCfg::default()).unwrap();
        by_slot.insert(id, i);
    }
    let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
    let mut steps = 0;
    while engine.active() > 0 {
        for em in engine.step() {
            let Some(&i) = by_slot.get(&em.slot) else {
                continue;
            };
            if let Some(f) = &em.finish {
                assert!(
                    !matches!(f, Finish::Error(_)),
                    "the head slot failed: {f:?}"
                );
            }
            out[i] = em.head.clone();
        }
        steps += 1;
        assert!(steps < 100_000, "the engine made no progress");
    }
    out.into_iter()
        .map(|v| v.expect("a head row per text"))
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn a_modern_bert_rank_head_is_the_exact_scalar_reference_forward() {
    let dir = tmpdir("reference");
    let model = head_model(&dir, Format::F32, 71, "mb-rank-f32", rank_head());
    assert_eq!(model.head_width(), 1);
    assert!(model.has_head());
    assert_eq!(model.head_labels(), &[] as &[String]);
    for (i, len) in [1usize, 2, 7, 31, 64].into_iter().enumerate() {
        let text = modern_bert_text(1 + i as u32, len);
        let want = support::modern_bert_ref::reference_head(&model, &text);
        let got = head_batch(&model, std::slice::from_ref(&text), 1, 64, None);
        assert_eq!(
            bits(&got[0]),
            bits(&want),
            "len {len}: the engine's head row differs from the reference"
        );
    }
}

#[test]
fn a_modern_bert_labelled_head_is_the_exact_scalar_reference_forward() {
    let dir = tmpdir("labelled-reference");
    let model = head_model(&dir, Format::F32, 72, "mb-labels-f32", labelled_head());
    assert_eq!(model.head_width(), 3);
    assert_eq!(
        model.head_labels(),
        ["negative", "neutral", "positive"].map(String::from)
    );
    for (i, len) in [1usize, 5, 17, 48].into_iter().enumerate() {
        let text = modern_bert_text(3 + i as u32, len);
        let want = support::modern_bert_ref::reference_head(&model, &text);
        // Two threads: the head's projections are one row, so the pool's split cannot
        // touch them.
        let got = head_batch(&model, std::slice::from_ref(&text), 2, 64, None);
        assert_eq!(bits(&got[0]), bits(&want), "len {len}: the labelled head");
    }
}

/// The head the family ships by default — mean pooling, erf GELU, no `classifier.*`
/// key — on an F32 file, and the mean against row 0 explicitly.
#[test]
fn the_default_head_pools_the_mean_of_the_last_rows() {
    let dir = tmpdir("mean-pool");
    let model = head_model(&dir, Format::F32, 73, "mb-mean-f32", rank_head());
    let text = modern_bert_text(5, 23);
    let want = support::modern_bert_ref::reference_head(&model, &text);
    let got = head_batch(&model, std::slice::from_ref(&text), 1, 64, None);
    assert_eq!(bits(&got[0]), bits(&want));

    // The same file with `classifier.pooling_type = 2` starts from row 0 instead, so the
    // row differs.
    let cls = head_model(
        &dir,
        Format::F32,
        73,
        "mb-cls-f32",
        ModernBertHeadConfig {
            pooling_type: Some(2),
            ..rank_head()
        },
    );
    let cls_want = support::modern_bert_ref::reference_head(&cls, &text);
    let cls_got = head_batch(&cls, std::slice::from_ref(&text), 1, 64, None);
    assert_eq!(bits(&cls_got[0]), bits(&cls_want));
    assert_ne!(bits(&cls_got[0]), bits(&got[0]), "mean and cls agree");
}

/// `cls.bias` and `cls.norm.bias` are in the transformers config only when
/// `classifier_bias` / `norm_bias` are on, and llama.cpp creates neither for this family
/// (`src/models/modern-bert.cpp:90-93`). A file that carries them computes the row of
/// the same file without them.
#[test]
fn the_head_ignores_the_biases_llama_cpp_does_not_create() {
    let dir = tmpdir("biases");
    let plain = head_model(
        &dir,
        Format::F32,
        74,
        "mb-head-plain",
        ModernBertHeadConfig {
            labels: vec!["a".into(), "b".into()],
            n_out: 2,
            ..Default::default()
        },
    );
    let biased = head_model(
        &dir,
        Format::F32,
        74,
        "mb-head-biased",
        ModernBertHeadConfig {
            labels: vec!["a".into(), "b".into()],
            n_out: 2,
            extra_biases: true,
            ..Default::default()
        },
    );
    let text = modern_bert_text(8, 19);
    let a = head_batch(&plain, std::slice::from_ref(&text), 1, 64, None);
    let b = head_batch(&biased, std::slice::from_ref(&text), 1, 64, None);
    assert_eq!(bits(&a[0]), bits(&b[0]));
}

#[test]
fn a_head_row_is_bit_identical_alone_in_batches_and_at_any_pass_size() {
    let dir = tmpdir("invariance");
    for (fmt, seed, head, name) in [
        (Format::F32, 75u64, rank_head(), "mb-rank-inv"),
        (
            Format::Q8_0,
            76,
            ModernBertHeadConfig {
                labels: vec!["negative".into(), "neutral".into(), "positive".into()],
                n_out: 3,
                ..Default::default()
            },
            "mb-labels-inv",
        ),
    ] {
        let model = head_model(&dir, fmt, seed, name, head);
        let probe = modern_bert_text(9, 37);
        let alone = head_batch(&model, std::slice::from_ref(&probe), 1, 64, None)[0].clone();

        for (n, threads, pass) in [
            (2usize, 1usize, None),
            (7, 2, None),
            (64, 4, None),
            (7, 1, Some(1)),
            (64, 2, Some(13)),
        ] {
            let mut texts = Vec::with_capacity(n);
            for i in 0..n {
                texts.push(if i == n / 2 {
                    probe.clone()
                } else {
                    modern_bert_text(100 + i as u32, 1 + (i * 11) % 53)
                });
            }
            if n % 2 == 0 {
                texts.swap(0, n / 2);
            }
            let rows = head_batch(&model, &texts, threads, 64, pass);
            let at = texts.iter().position(|t| t == &probe).unwrap();
            assert_eq!(
                bits(&rows[at]),
                bits(&alone),
                "{}: the probe in a batch of {n}, {threads} threads, pass {pass:?}",
                fmt.label()
            );
        }

        // A text longer than the matmul's 128-row chunk takes several weight passes.
        let long = modern_bert_text(77, 64);
        let long_alone = head_batch(&model, std::slice::from_ref(&long), 2, 64, None)[0].clone();
        let mixed = vec![modern_bert_text(4, 3), modern_bert_text(5, 5), long.clone()];
        let rows = head_batch(&model, &mixed, 4, 64, None);
        assert_eq!(
            bits(&rows[2]),
            bits(&long_alone),
            "{}: the long text's row alone and in a mixed pass",
            fmt.label()
        );
    }
}

#[test]
fn a_file_without_a_head_refuses_a_head_slot_by_name() {
    let dir = tmpdir("no-head");
    let path = write_modern_bert(&dir, Format::F32, 78, ModernBertConfig::default());
    let model = load(&path);
    assert!(!model.has_head());
    assert_eq!(model.head_width(), 0);
    let mut engine = Engine::new(model, 1, 64, 1, KvPrec::F16, 0);
    let err = engine
        .admit_head(modern_bert_text(1, 8), SlotCfg::default())
        .unwrap_err();
    assert!(
        err.1.contains("no classifier head") && err.1.contains("modern-bert"),
        "{err:?}"
    );
}

#[test]
fn the_head_loader_refuses_metadata_it_cannot_honour() {
    let dir = tmpdir("bad-head");
    // Three labels, two outputs: llama.cpp sizes `cls.output` by the labels.
    let path = write_modern_bert(
        &dir,
        Format::F32,
        79,
        ModernBertConfig {
            head: Some(ModernBertHeadConfig {
                n_out: 2,
                labels: vec!["a".into(), "b".into(), "c".into()],
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    let err = match peregrine_core::Gguf::open(&path).and_then(peregrine_models::load) {
        Ok(_) => panic!("a head whose labels disagree with cls.output loaded"),
        Err(e) => e,
    };
    assert!(
        err.contains("cls.output.weight") && err.contains("3"),
        "{err}"
    );

    // An activation llama.cpp does not read.
    let path = write_modern_bert(
        &dir,
        Format::F32,
        80,
        ModernBertConfig {
            head: Some(ModernBertHeadConfig {
                activation: Some("relu"),
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    let err = match peregrine_core::Gguf::open(&path).and_then(peregrine_models::load) {
        Ok(_) => panic!("an activation llama.cpp does not read loaded"),
        Err(e) => e,
    };
    assert!(err.contains("relu") && err.contains("tanh"), "{err}");

    // Labels with no projection to produce them.
    let path = write_modern_bert(
        &dir,
        Format::F32,
        81,
        ModernBertConfig {
            head: Some(ModernBertHeadConfig {
                labels: vec!["a".into(), "b".into()],
                n_out: 2,
                out: false,
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    let err = match peregrine_core::Gguf::open(&path).and_then(peregrine_models::load) {
        Ok(_) => panic!("labels without cls.output loaded"),
        Err(e) => e,
    };
    assert!(
        err.contains("classifier.output_labels") && err.contains("cls.output.weight"),
        "{err}"
    );

    // A head norm with no dense layer and no projection: nothing to apply it to.
    let path = write_modern_bert(
        &dir,
        Format::F32,
        82,
        ModernBertConfig {
            head: Some(ModernBertHeadConfig {
                dense: false,
                norm: true,
                out: false,
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    let err = match peregrine_core::Gguf::open(&path).and_then(peregrine_models::load) {
        Ok(_) => panic!("a bare cls.norm loaded as a head"),
        Err(e) => e,
    };
    assert!(
        err.contains("cls.norm.weight") && err.contains("cls.output.weight"),
        "{err}"
    );
}

/// The two synthetic heads' golden for the current kernel tier: the whole row's bits.
#[test]
fn the_synthetic_modern_bert_heads_match_their_golden_for_the_tier() {
    let dir = tmpdir("golden");
    let mut out = String::new();
    let mut cases = vec![(rank_head(), "rank".to_string())];
    cases.push((labelled_head(), "labels".to_string()));
    for (head, name) in cases {
        let model = head_model(&dir, Format::Q8_0, 83, &format!("mb-{name}-q8"), head);
        for len in [7usize, 37] {
            let text = modern_bert_text(21 + len as u32, len);
            let row = head_batch(&model, std::slice::from_ref(&text), 3, 64, None);
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for x in &row[0] {
                for b in x.to_bits().to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
            }
            use std::fmt::Write as _;
            writeln!(out, "{name} len {len} width {} fnv {h:016x}", row[0].len()).unwrap();
        }
    }
    let tier = peregrine_core::cpu::tier_name();
    let golden_tier = format!(
        "{}{}",
        if tier == "x86_64-avx2-vnni256" {
            "x86_64-avx2"
        } else {
            tier.as_str()
        },
        if cfg!(target_os = "macos") {
            "-macos"
        } else {
            ""
        }
    );
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/synthetic")
        .join(golden_tier);
    let gpath = dir.join("modern-bert-heads-Q8_0.golden");
    let record = std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "record");
    let require = std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "require");
    if record && tier != "x86_64-avx2-vnni256" {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&gpath, &out).unwrap();
        eprintln!("synthetic: recorded modern-bert-heads-Q8_0 ({tier})");
    } else if let Ok(want) = std::fs::read_to_string(&gpath) {
        assert_eq!(want, out, "{}", gpath.display());
    } else if require {
        panic!(
            "modern-bert-heads-Q8_0 has no golden for tier {tier} ({}); record it with \
             PEREGRINE_GOLDEN=record",
            gpath.display()
        );
    }
}
