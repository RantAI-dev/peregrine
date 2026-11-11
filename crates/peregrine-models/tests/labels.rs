//! The decision label read-out (roadmap S4.3).
//!
//! A `Want::Labels` slot is prefilled like a pooled slot and finishes when its last prompt row is
//! computed; its emission carries that row's logits at the slot's label token ids. Those values
//! must be bit-identical to the same entries of the full logits row the model computes for the
//! position, alone, after a sibling question with the same prompt prefix (the prefix store), and in
//! a batch of questions.
//!
//! The synthetic `llama` (dense, prefix-reusing) and `qwen35` (hybrid: recurrent state, so the
//! store declines to keep its cache) files exercise both families without model files.

mod support;

use peregrine_core::gguf::Format;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, StepStats};
use peregrine_core::pool::Pool;
use peregrine_core::serve::{Emission, Engine, Finish, SlotCfg};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{write_dense, write_hybrid_plain, Dense};

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-labels-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &Path) -> Arc<Model> {
    let g = peregrine_core::Gguf::open(path).expect("open the synthetic model");
    Arc::new(peregrine_models::load(g).expect("load the synthetic model"))
}

/// The label token ids every question is read at (all inside the synthetic vocabulary).
const LABELS: [u32; 3] = [11, 12, 13];

/// The shared beginning every question of a request has: long enough to be worth a prefix
/// (`prefix::MIN_PREFIX` is 16).
fn shared_prefix() -> Vec<u32> {
    (0..24).map(|i| 100 + (i * 7) % 90).collect()
}

/// One question: the shared state and a tail of its own.
fn question(i: usize) -> Vec<u32> {
    let mut p = shared_prefix();
    for t in 0..6 {
        p.push(30 + ((i * 11 + t * 5) % 120) as u32);
    }
    p
}

/// The reference: one forward over the whole prompt, asking for its last row's logits, then the
/// same gather the engine does.
fn reference(model: &Model, prompt: &[u32], labels: &[u32], threads: usize) -> Vec<f32> {
    let pool = Pool::new(threads);
    let mut st = model.new_state(prompt.len(), 1, model.cfg.ctx_train, threads);
    let mut kv = model.new_cache(KvPrec::F16, model.cfg.ctx_train);
    let mut stats = StepStats::default();
    model
        .forward(
            &mut st,
            &mut kv,
            &pool,
            prompt,
            0,
            &[prompt.len() - 1],
            &mut stats,
        )
        .expect("reference forward");
    let logits = st.logits_row(0, model.cfg.n_vocab);
    labels.iter().map(|&l| logits[l as usize]).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

struct Run {
    /// The label logits of every admitted question, by admission order.
    logits: Vec<Vec<f32>>,
    /// The prefilled token count of every question.
    prefilled: Vec<usize>,
}

/// Admit `prompts` as labels slots in one engine and run it to completion. `prefix_bytes` sizes the
/// shared prefix store (0 turns it off).
fn run(
    model: &Arc<Model>,
    prompts: &[Vec<u32>],
    threads: usize,
    chunk: usize,
    prefix_bytes: usize,
) -> Run {
    let mut engine = Engine::new_with_chunk(
        model.clone(),
        threads,
        model.cfg.ctx_train,
        prompts.len(),
        KvPrec::F16,
        prefix_bytes,
        chunk,
    );
    let mut by_slot: HashMap<u64, usize> = HashMap::new();
    for (i, p) in prompts.iter().enumerate() {
        let id = engine
            .admit_labels(p.clone(), SlotCfg::default(), LABELS.to_vec())
            .expect("admit a labels slot");
        by_slot.insert(id, i);
    }
    let mut logits: Vec<Option<Vec<f32>>> = vec![None; prompts.len()];
    let mut prefilled = vec![0usize; prompts.len()];
    let mut steps = 0;
    while engine.active() > 0 {
        for em in engine.step() {
            let Some(&i) = by_slot.get(&em.slot) else {
                continue;
            };
            if let Some(Finish::Error(e)) = &em.finish {
                panic!("the labels slot failed: {e}");
            }
            if em.finish.is_some() {
                logits[i] = Some(em.label_logits.clone().expect("label logits"));
                prefilled[i] = em.cost.as_ref().expect("cost").prefilled;
            }
        }
        steps += 1;
        assert!(steps < 100_000, "the engine made no progress");
    }
    Run {
        logits: logits
            .into_iter()
            .map(|v| v.expect("one emission per question"))
            .collect(),
        prefilled,
    }
}

fn models(dir: &Path) -> Vec<(&'static str, Arc<Model>)> {
    vec![
        (
            "llama",
            load(&write_dense(
                dir,
                "llama",
                Format::F16,
                31,
                Dense::default(),
            )),
        ),
        ("qwen35", load(&write_hybrid_plain(dir, Format::Q8_0, 31))),
    ]
}

#[test]
fn the_label_logits_are_the_full_rows_entries_alone_and_in_batches() {
    let dir = tmpdir("invariance");
    let questions: Vec<Vec<u32>> = (0..4).map(question).collect();
    for (arch, model) in models(&dir) {
        for threads in [1usize, 2, 4] {
            for chunk in [1usize, 16, 64] {
                let reference: Vec<Vec<f32>> = questions
                    .iter()
                    .map(|p| reference(&model, p, &LABELS, threads))
                    .collect();

                // (a) alone, from position 0: one engine per question, no prefix store.
                for (i, prompt) in questions.iter().enumerate() {
                    let r = run(&model, std::slice::from_ref(prompt), threads, chunk, 0);
                    assert_eq!(
                        bits(&r.logits[0]),
                        bits(&reference[i]),
                        "{arch}: question {i} alone, {threads} threads, chunk {chunk}"
                    );
                    assert_eq!(r.prefilled[0], prompt.len());
                }

                // (c) in the same batch as the others.
                let r = run(&model, &questions, threads, chunk, 0);
                for (i, want) in reference.iter().enumerate() {
                    assert_eq!(
                        bits(&r.logits[i]),
                        bits(want),
                        "{arch}: question {i} in a batch, {threads} threads, chunk {chunk}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_question_after_another_with_the_same_prefix_is_bit_identical() {
    let dir = tmpdir("prefix");
    let questions: Vec<Vec<u32>> = (0..4).map(question).collect();
    for (arch, model) in models(&dir) {
        for threads in [1usize, 2, 4] {
            for chunk in [1usize, 16, 64] {
                let reference: Vec<Vec<f32>> = questions
                    .iter()
                    .map(|p| reference(&model, p, &LABELS, threads))
                    .collect();

                // The first question fills the store; the others then restore the shared prefix.
                let mut engine = Engine::new_with_chunk(
                    model.clone(),
                    threads,
                    model.cfg.ctx_train,
                    1,
                    KvPrec::F16,
                    8 << 20,
                    chunk,
                );
                let mut hits_before = 0;
                let mut reused = Vec::new();
                for (i, prompt) in questions.iter().enumerate() {
                    let id = engine
                        .admit_labels(prompt.clone(), SlotCfg::default(), LABELS.to_vec())
                        .expect("admit");
                    let mut got: Option<(Vec<f32>, usize)> = None;
                    let mut steps = 0;
                    while engine.active() > 0 {
                        for em in engine.step() {
                            assert_eq!(em.slot, id);
                            if let Some(Finish::Error(e)) = &em.finish {
                                panic!("the labels slot failed: {e}");
                            }
                            if em.finish.is_some() {
                                got = Some((
                                    em.label_logits.clone().expect("label logits"),
                                    em.cost.as_ref().expect("cost").prefilled,
                                ));
                            }
                        }
                        steps += 1;
                        assert!(steps < 100_000, "the engine made no progress");
                    }
                    let (logits, prefilled) = got.expect("one emission");
                    assert_eq!(
                        bits(&logits),
                        bits(&reference[i]),
                        "{arch}: question {i} after its siblings, {threads} threads, chunk {chunk}"
                    );
                    let (_, _, hits, _) = engine.prefix_stats();
                    reused.push((hits > hits_before, prefilled));
                    hits_before = hits;
                }
                // A dense model shares the prefix (the earlier question's cache is in the store); a
                // hybrid one cannot — its snapshot would carry no recurrent state, which the store
                // already declines to keep — so it recomputes and is still bit-identical.
                if arch == "llama" {
                    for (i, (did_reuse, prefilled)) in reused.iter().enumerate().skip(1) {
                        assert!(
                            *did_reuse,
                            "{arch}: question {i} did not restore the prefix"
                        );
                        assert!(
                            *prefilled < questions[i].len(),
                            "{arch}: question {i} prefilled {prefilled} of {}",
                            questions[i].len()
                        );
                    }
                } else {
                    assert!(
                        reused.iter().all(|&(did_reuse, _)| !did_reuse),
                        "{arch} is hybrid: its cache is not kept in the prefix store"
                    );
                }
            }
        }
    }
}

#[test]
fn label_ids_are_validated_at_admission() {
    let dir = tmpdir("validation");
    let path = write_dense(&dir, "llama", Format::F16, 31, Dense::default());
    let model = load(&path);
    let mut engine = Engine::new(model.clone(), 1, 128, 1, KvPrec::F16, 0);
    let n_vocab = model.cfg.n_vocab as u32;
    let err = engine
        .admit_labels(vec![11, 12], SlotCfg::default(), vec![n_vocab])
        .unwrap_err();
    assert!(err.1.contains("outside the vocabulary"), "{err:?}");
    let err = engine
        .admit_labels(vec![11, 12], SlotCfg::default(), vec![])
        .unwrap_err();
    assert!(err.1.contains("needs a label token"), "{err:?}");
}

#[test]
fn a_labels_slot_beside_a_chat_slot_still_reads_its_own_row() {
    // A decode row and a labels row share one weight pass; the label logits must not depend on it.
    let dir = tmpdir("mixed");
    let path = write_dense(&dir, "llama", Format::F16, 32, Dense::default());
    let model = load(&path);
    let prompt = question(9);
    let want = reference(&model, &prompt, &LABELS, 2);
    let cfg = SlotCfg {
        max_tokens: 4,
        ..Default::default()
    };
    let mut engine =
        Engine::new_with_chunk(model.clone(), 2, model.cfg.ctx_train, 2, KvPrec::F16, 0, 2);
    let chat = engine
        .admit(question(10), cfg)
        .expect("admit the chat slot");
    let labels = engine
        .admit_labels(prompt, SlotCfg::default(), LABELS.to_vec())
        .expect("admit the labels slot");
    let mut got: Option<Vec<f32>> = None;
    let mut chat_tokens = 0;
    let mut steps = 0;
    while engine.active() > 0 {
        for em in engine.step() {
            if em.slot == labels && em.finish.is_some() {
                got = em.label_logits.clone();
            } else if em.slot == chat {
                chat_tokens += 1;
            }
        }
        steps += 1;
        assert!(steps < 100_000, "the engine made no progress");
    }
    assert!(chat_tokens >= 4, "the chat slot decoded its four tokens");
    assert_eq!(bits(&got.expect("label logits")), bits(&want));
}

/// The unmixed scheduler (`PEREGRINE_MIX=0`) prefills through `prefill_chunk` and reads a labels
/// slot's row there. The switch is process-wide, so the child process runs the check.
#[test]
fn labels_through_the_unmixed_scheduler() {
    if std::env::var_os("PEREGRINE_LABELS_CHILD").is_none() {
        let exe = std::env::current_exe().expect("the test binary's path");
        let status = std::process::Command::new(exe)
            .args([
                "--exact",
                "labels_through_the_unmixed_scheduler",
                "--nocapture",
            ])
            .env("PEREGRINE_MIX", "0")
            .env("PEREGRINE_LABELS_CHILD", "1")
            .status()
            .expect("spawn the unmixed child");
        assert!(status.success(), "the PEREGRINE_MIX=0 child failed");
        return;
    }
    let dir = tmpdir("unmixed");
    let prompts: Vec<Vec<u32>> = (0..3).map(question).collect();
    for (arch, model) in models(&dir) {
        for threads in [1usize, 2] {
            for chunk in [1usize, 16] {
                let reference: Vec<Vec<f32>> = prompts
                    .iter()
                    .map(|p| reference(&model, p, &LABELS, threads))
                    .collect();
                for (i, prompt) in prompts.iter().enumerate() {
                    let r = run(&model, std::slice::from_ref(prompt), threads, chunk, 0);
                    assert_eq!(
                        bits(&r.logits[0]),
                        bits(&reference[i]),
                        "{arch}: question {i} alone (unmixed), {threads} threads, chunk {chunk}"
                    );
                }
                let r = run(&model, &prompts, threads, chunk, 0);
                for (i, want) in reference.iter().enumerate() {
                    assert_eq!(
                        bits(&r.logits[i]),
                        bits(want),
                        "{arch}: question {i} in a batch (unmixed), {threads} threads, chunk {chunk}"
                    );
                }
            }
        }
    }
}

/// The label logits of a question are the same bits when the same weight passes also serve an
/// embedding and a chat slot (S4.3c): a decision request runs beside the other endpoints on one
/// server, and its answer must not depend on what else is in the batch.
#[test]
fn a_question_beside_an_embedding_and_a_chat_slot_is_bit_identical() {
    let dir = tmpdir("mixed");
    for (arch, model) in models(&dir) {
        let alone = run(&model, &[question(0)], 2, 64, 1 << 20).logits[0].clone();
        let ctx = model.cfg.ctx_train;
        let mut engine = Engine::new(model.clone(), 2, ctx, 4, KvPrec::F16, 1 << 20);
        let id = engine
            .admit_labels(question(0), SlotCfg::default(), LABELS.to_vec())
            .expect("admit the labels slot");
        engine
            .admit_pooled(
                vec![70, 71, 72, 73, 74, 75],
                SlotCfg::default(),
                peregrine_core::embed::Pooling::Last,
            )
            .expect("admit an embedding slot");
        let chat = SlotCfg {
            max_tokens: 4,
            ..SlotCfg::default()
        };
        engine
            .admit(vec![80, 81, 82, 83, 84, 85], chat)
            .expect("admit a chat slot");
        let mut got: Option<Vec<f32>> = None;
        let mut steps = 0;
        while engine.active() > 0 {
            for em in engine.step() {
                if let Some(Finish::Error(e)) = &em.finish {
                    panic!("{arch}: a slot failed: {e}");
                }
                if em.slot == id {
                    if let Some(v) = &em.label_logits {
                        got = Some(v.clone());
                    }
                }
            }
            steps += 1;
            assert!(steps < 100_000, "{arch}: the engine made no progress");
        }
        assert_eq!(
            bits(&got.expect("the labels slot emitted")),
            bits(&alone),
            "{arch}: the logits changed beside an embedding and a chat slot"
        );
    }
}

#[test]
fn a_pooled_emission_has_no_label_logits() {
    // The two read-outs do not leak into each other.
    let dir = tmpdir("emission");
    let path = write_dense(&dir, "llama", Format::F16, 33, Dense::default());
    let model = load(&path);
    let mut engine = Engine::new(model.clone(), 1, 128, 1, KvPrec::F16, 0);
    engine
        .admit_pooled(
            vec![11, 12, 13, 14],
            SlotCfg::default(),
            peregrine_core::embed::Pooling::Mean,
        )
        .unwrap();
    let mut emissions: Vec<Emission> = Vec::new();
    while engine.active() > 0 {
        emissions.extend(engine.step());
    }
    let em = emissions.last().expect("an emission");
    assert!(em.embedding.is_some());
    assert!(em.label_logits.is_none());
}
