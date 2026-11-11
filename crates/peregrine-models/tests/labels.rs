//! The decision label read-out (roadmap S4.3).
//!
//! A `Want::Labels` slot is prefilled like a pooled slot and finishes when its last prompt row is
//! computed; its emission carries that row's logits at the slot's label token ids. Those values
//! must be bit-identical to the same entries of the full logits row the model computes for the
//! position, alone, after a sibling question with the same prompt prefix (the prefix store), and in
//! a batch of questions.
//!
//! The synthetic `llama` (dense) and `qwen35` (hybrid: a snapshot carries the recurrent state too,
//! so it restores only at its own length) files exercise both families without model files.

mod support;

use peregrine_core::gguf::Format;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, StepStats};
use peregrine_core::pool::Pool;
use peregrine_core::serve::{Emission, Engine, Finish, SlotCfg};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{write_dense, write_hybrid, write_hybrid_plain, Dense};

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
    reference_prec(model, prompt, labels, threads, KvPrec::F16)
}

fn reference_prec(
    model: &Model,
    prompt: &[u32],
    labels: &[u32],
    threads: usize,
    prec: KvPrec,
) -> Vec<f32> {
    let pool = Pool::new(threads);
    let mut st = model.new_state(prompt.len(), 1, model.cfg.ctx_train, threads);
    let mut kv = model.new_cache(prec, model.cfg.ctx_train);
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
    /// The batch rows forwarded to answer the questions, summed over the steps.
    rows: usize,
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
    run_prec_pass(
        model,
        prompts,
        threads,
        chunk,
        prefix_bytes,
        KvPrec::F16,
        None,
        None,
    )
}

/// [`run`] with a cache precision, and — when `group` is `Some(min_prefix)` — the questions admitted
/// as one group (S4.9j) rather than one by one.
fn run_prec(
    model: &Arc<Model>,
    prompts: &[Vec<u32>],
    threads: usize,
    chunk: usize,
    prefix_bytes: usize,
    prec: KvPrec,
    group: Option<usize>,
) -> Run {
    run_prec_pass(
        model,
        prompts,
        threads,
        chunk,
        prefix_bytes,
        prec,
        group,
        None,
    )
}

/// [`run_prec`] with an explicit pass budget (S4.9b; `None`: the machine's default).
#[allow(clippy::too_many_arguments)]
fn run_prec_pass(
    model: &Arc<Model>,
    prompts: &[Vec<u32>],
    threads: usize,
    chunk: usize,
    prefix_bytes: usize,
    prec: KvPrec,
    group: Option<usize>,
    pass_rows: Option<usize>,
) -> Run {
    let mut engine = Engine::new_with_chunk_and_pass(
        model.clone(),
        threads,
        model.cfg.ctx_train,
        prompts.len(),
        prec,
        prefix_bytes,
        chunk,
        pass_rows,
    );
    let mut by_slot: HashMap<u64, usize> = HashMap::new();
    match group {
        Some(min_prefix) => {
            let labels = vec![LABELS.to_vec(); prompts.len()];
            let ids = engine
                .admit_labels_group_from(prompts.to_vec(), SlotCfg::default(), labels, min_prefix)
                .expect("admit a group of labels slots");
            for (i, id) in ids.into_iter().enumerate() {
                by_slot.insert(id, i);
            }
        }
        None => {
            for (i, p) in prompts.iter().enumerate() {
                let id = engine
                    .admit_labels(p.clone(), SlotCfg::default(), LABELS.to_vec())
                    .expect("admit a labels slot");
                by_slot.insert(id, i);
            }
        }
    }
    let mut logits: Vec<Option<Vec<f32>>> = vec![None; prompts.len()];
    let mut prefilled = vec![0usize; prompts.len()];
    let mut rows = 0usize;
    let mut steps = 0;
    while engine.active() > 0 {
        let emitted = engine.step();
        rows += engine.last_step.rows;
        for em in emitted {
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
        rows,
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
                // A dense model shares the prefix (the earlier question's cache is in the store). A
                // hybrid model does not: the stored entry is a whole question, longer than the
                // beginning these prompts share, and a hybrid snapshot restores only at its own
                // length (a group's shared beginning — the same length for every question — is what
                // it keeps; see `a_repeated_state_is_restored_across_requests`).
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
                        "{arch} is hybrid: a longer stored question cannot restore a shorter beginning"
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
    // Each test writes into a directory of its own: the tests run in parallel threads, and a
    // second writer would truncate the file another test has memory-mapped (a SIGBUS on macOS).
    let dir = tmpdir("beside-chat");
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
                // S4.9j: the fork is taken in `prefill_chunk` on this path.
                let r = run_prec(
                    &model,
                    &prompts,
                    threads,
                    chunk,
                    1 << 20,
                    KvPrec::F16,
                    Some(1),
                );
                for (i, want) in reference.iter().enumerate() {
                    assert_eq!(
                        bits(&r.logits[i]),
                        bits(want),
                        "{arch}: forked question {i} (unmixed), {threads} threads, chunk {chunk}"
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
    let dir = tmpdir("beside-endpoints");
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

/// Questions that share exactly `shared` tokens and then differ: question `i` appends a tail of
/// `tails[i % tails.len()]` tokens, whose first token differs per question (so the longest common
/// beginning is exactly `shared`).
fn forked_questions(shared: usize, tails: &[usize], n: usize) -> Vec<Vec<u32>> {
    let base: Vec<u32> = (0..shared as u32).map(|i| 100 + (i * 7) % 90).collect();
    (0..n)
        .map(|i| {
            let t = tails[i % tails.len()];
            let mut p = base.clone();
            for j in 0..t {
                p.push(30 + ((i * 11 + j * 5) % 120) as u32);
            }
            p
        })
        .collect()
}

/// S4.9j: a question forked from a group's one evaluation of the shared beginning must equal the
/// same question computed alone from position 0, bit for bit — for groups of 2, 3 and 10 with tails
/// of different lengths, with the fork anywhere (a single token in, a prefill chunk boundary, the
/// middle of a chunk, one row before the end), on a dense (`llama`) and a hybrid (`qwen35`) model,
/// in the f16 and f32 caches, at 1, 2 and 4 threads and prefill chunks of 1, 16 and 64.
#[test]
fn a_forked_question_is_bit_identical_wherever_the_fork_is() {
    let dir = tmpdir("fork");
    // (shared tokens, tail lengths the questions cycle through): the fork at 1, at chunk
    // boundaries and past one, and one row before the shortest question's end.
    let cases: Vec<(usize, Vec<usize>)> = vec![
        (1, vec![1, 2, 7]),
        (16, vec![1, 3, 9]),
        (17, vec![2, 4, 6]),
        (24, vec![1]),
        (64, vec![1, 5, 33]),
    ];
    for (arch, model) in models(&dir) {
        for prec in [KvPrec::F16, KvPrec::F32] {
            for threads in [1usize, 2, 4] {
                for chunk in [1usize, 16, 64] {
                    for (shared, tails) in &cases {
                        for n in [2usize, 3, 10] {
                            let asked = forked_questions(*shared, tails, n);
                            let want: Vec<Vec<f32>> = asked
                                .iter()
                                .map(|p| reference_prec(&model, p, &LABELS, threads, prec))
                                .collect();
                            let r = run_prec(
                                &model,
                                &asked,
                                threads,
                                chunk,
                                /* prefix_bytes */ 1 << 20,
                                prec,
                                Some(1),
                            );
                            for (i, w) in want.iter().enumerate() {
                                assert_eq!(
                                    bits(&r.logits[i]),
                                    bits(w),
                                    "{arch} {prec:?}: question {i} of {n}, shared {shared}, \
                                     {threads} threads, chunk {chunk}, tails {tails:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// S4.9j: a group forwards the shared beginning once and each question only its own tail. Rows
/// forwarded are `shared + Σ tail`, not `Σ prompt`; and a shared beginning below the threshold is
/// admitted as separate slots, which forward every token as before.
#[test]
fn a_group_forwards_the_shared_beginning_once() {
    let dir = tmpdir("rows");
    for (arch, model) in models(&dir) {
        let (shared, tails, n) = (24usize, [2usize, 5, 9], 3usize);
        let asked = forked_questions(shared, &tails, n);
        let want_rows: usize = shared + tails.iter().sum::<usize>();
        for threads in [1usize, 2, 4] {
            for chunk in [1usize, 16, 64] {
                let r = run_prec(
                    &model,
                    &asked,
                    threads,
                    chunk,
                    0,
                    KvPrec::F16,
                    Some(peregrine_core::prefix::MIN_PREFIX),
                );
                assert_eq!(
                    r.rows, want_rows,
                    "{arch}: {threads} threads, chunk {chunk}: rows forwarded"
                );
                assert_eq!(
                    r.rows,
                    asked.iter().map(Vec::len).sum::<usize>() - (n - 1) * shared,
                    "{arch}: the saving is one shared beginning"
                );
                // Every question still paid for its own prompt in its own accounting (the rows it
                // shares were forwarded once for the group).
                for (i, p) in asked.iter().enumerate() {
                    assert_eq!(r.prefilled[i], p.len(), "{arch}: question {i}");
                }
                // Below the threshold, the slots are independent and forward everything.
                let short = forked_questions(8, &tails, n);
                let r = run_prec(
                    &model,
                    &short,
                    threads,
                    chunk,
                    0,
                    KvPrec::F16,
                    Some(peregrine_core::prefix::MIN_PREFIX),
                );
                assert_eq!(
                    r.rows,
                    short.iter().map(Vec::len).sum::<usize>(),
                    "{arch}: an 8-token beginning is not worth a fork"
                );
            }
        }
    }
}

/// S4.9j: a group beside a chat slot and an embedding slot — one weight pass carrying a forked
/// question's tail rows, other users' decode rows and an embedding's prompt rows — still reads the
/// same bits as the question alone.
#[test]
fn a_forked_question_beside_other_slots_is_bit_identical() {
    let dir = tmpdir("fork-beside");
    for (arch, model) in models(&dir) {
        let asked = forked_questions(24, &[3, 5, 7], 3);
        let want: Vec<Vec<f32>> = asked
            .iter()
            .map(|p| reference_prec(&model, p, &LABELS, 2, KvPrec::F16))
            .collect();
        let mut engine = Engine::new_with_chunk(
            model.clone(),
            2,
            model.cfg.ctx_train,
            5,
            KvPrec::F16,
            1 << 20,
            16,
        );
        let ids = engine
            .admit_labels_group_from(
                asked.clone(),
                SlotCfg::default(),
                vec![LABELS.to_vec(); asked.len()],
                1,
            )
            .expect("admit the group");
        engine
            .admit_pooled(
                vec![70, 71, 72, 73, 74, 75],
                SlotCfg::default(),
                peregrine_core::embed::Pooling::Last,
            )
            .expect("admit an embedding slot");
        engine
            .admit(
                vec![80, 81, 82, 83, 84, 85],
                SlotCfg {
                    max_tokens: 4,
                    ..SlotCfg::default()
                },
            )
            .expect("admit a chat slot");
        let mut got: Vec<Option<Vec<f32>>> = vec![None; ids.len()];
        let mut steps = 0;
        while engine.active() > 0 {
            for em in engine.step() {
                if let Some(Finish::Error(e)) = &em.finish {
                    panic!("{arch}: a slot failed: {e}");
                }
                if let Some(i) = ids.iter().position(|id| *id == em.slot) {
                    if let Some(v) = &em.label_logits {
                        got[i] = Some(v.clone());
                    }
                }
            }
            steps += 1;
            assert!(steps < 100_000, "{arch}: the engine made no progress");
        }
        for (i, w) in want.iter().enumerate() {
            assert_eq!(
                bits(got[i].as_ref().expect("the group's question answered")),
                bits(w),
                "{arch}: forked question {i} changed beside other endpoints"
            );
        }
    }
}

/// S4.9j: with a store budget, the shared beginning a group forks is kept, so the next request with
/// the same state restores it instead of forwarding it — on a hybrid model too, whose snapshot now
/// carries the recurrent state.
#[test]
fn a_repeated_state_is_restored_across_requests() {
    let dir = tmpdir("fork-store");
    for (arch, model) in models(&dir) {
        let asked = forked_questions(24, &[3, 5, 7], 3);
        let mut engine = Engine::new_with_chunk(
            model.clone(),
            2,
            model.cfg.ctx_train,
            3,
            KvPrec::F16,
            8 << 20,
            16,
        );
        let mut answers: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut hits = Vec::new();
        for _ in 0..2 {
            let ids = engine
                .admit_labels_group(asked.clone(), SlotCfg::default(), vec![LABELS.to_vec(); 3])
                .expect("admit the group");
            let mut got: Vec<Option<Vec<f32>>> = vec![None; ids.len()];
            while engine.active() > 0 {
                for em in engine.step() {
                    if let Some(i) = ids.iter().position(|id| *id == em.slot) {
                        if let Some(v) = &em.label_logits {
                            got[i] = Some(v.clone());
                        }
                    }
                }
            }
            answers.push(got.into_iter().map(|v| v.expect("an answer")).collect());
            hits.push(engine.prefix_stats().2);
        }
        let want: Vec<Vec<f32>> = asked
            .iter()
            .map(|p| reference_prec(&model, p, &LABELS, 2, KvPrec::F16))
            .collect();
        for (i, w) in want.iter().enumerate() {
            assert_eq!(
                bits(&answers[0][i]),
                bits(w),
                "{arch}: question {i} of the first request"
            );
            assert_eq!(
                bits(&answers[1][i]),
                bits(w),
                "{arch}: question {i} of the repeated request"
            );
        }
        assert!(
            hits[1] > hits[0],
            "{arch}: the repeated state was not restored from the store ({hits:?})"
        );
    }
}

/// S4.9b: the pass budget is scheduling only for a decision too. A forked question's label logits
/// are bit-identical at budgets 1, 16, 64, 256 and the default, at 1, 2 and 4 threads, chunks 1
/// and 16, in groups of 3 and 10 and (for a couple of budgets) as independent slots.
#[test]
fn a_forked_questions_logits_are_the_same_for_every_pass_budget() {
    let dir = tmpdir("budget");
    for (arch, model) in models(&dir) {
        for (shared, tails, n) in [(24usize, [2usize, 5, 9], 3usize), (16, [1, 3, 7], 10)] {
            let asked = forked_questions(shared, &tails, n);
            let want: Vec<Vec<f32>> = asked
                .iter()
                .map(|p| reference_prec(&model, p, &LABELS, 2, KvPrec::F16))
                .collect();
            for budget in [Some(1usize), Some(16), Some(64), Some(256), None] {
                for threads in [1usize, 2, 4] {
                    for chunk in [1usize, 16] {
                        let r = run_prec_pass(
                            &model,
                            &asked,
                            threads,
                            chunk,
                            1 << 20,
                            KvPrec::F16,
                            Some(1),
                            budget,
                        );
                        for (i, w) in want.iter().enumerate() {
                            assert_eq!(
                                bits(&r.logits[i]),
                                bits(w),
                                "{arch}: question {i} of {n}, shared {shared}, budget {budget:?}, \
                                 {threads} threads, chunk {chunk}"
                            );
                        }
                    }
                }
            }
            // The same questions as independent slots at two budgets: the budget must not matter
            // there either.
            for budget in [Some(16usize), None] {
                let r = run_prec_pass(&model, &asked, 2, 16, 0, KvPrec::F16, None, budget);
                for (i, w) in want.iter().enumerate() {
                    assert_eq!(
                        bits(&r.logits[i]),
                        bits(w),
                        "{arch}: question {i} of {n} ungrouped, budget {budget:?}"
                    );
                }
            }
        }
    }
}
// ---------------------------------------------------------------------------
// S4.9d: the label only path over every head format and every tier.
// ---------------------------------------------------------------------------

/// Every weight format the synthetic generator can give the output head.
const HEAD_FORMATS: &[Format] = &[
    Format::F32,
    Format::F16,
    Format::BF16,
    Format::Q4_0,
    Format::Q5_0,
    Format::Q8_0,
    Format::Q2_K,
    Format::Q3_K,
    Format::Q4_K,
    Format::Q5_K,
    Format::Q6_K,
    Format::IQ4_NL,
    Format::IQ4_XS,
    Format::MXFP4,
];

/// The whole vocabulary row of a prompt's last position, computed by the ordinary
/// forward path: the reference the label entries are compared with.
fn full_row(model: &Model, prompt: &[u32], threads: usize) -> Vec<f32> {
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
        .expect("the reference forward");
    st.logits_row(0, model.cfg.n_vocab).to_vec()
}

/// Admit one labels slot and return its emission's logits.
fn labels_via_engine(
    model: &Arc<Model>,
    prompt: &[u32],
    labels: &[u32],
    threads: usize,
) -> Vec<f32> {
    let mut engine = Engine::new_with_chunk(
        model.clone(),
        threads,
        model.cfg.ctx_train,
        1,
        KvPrec::F16,
        0,
        16,
    );
    let id = engine
        .admit_labels(prompt.to_vec(), SlotCfg::default(), labels.to_vec())
        .expect("admit a labels slot");
    let mut got: Option<Vec<f32>> = None;
    let mut steps = 0;
    while engine.active() > 0 {
        for em in engine.step() {
            if em.slot != id {
                continue;
            }
            if let Some(Finish::Error(e)) = &em.finish {
                panic!("the labels slot failed: {e}");
            }
            if em.finish.is_some() {
                got = em.label_logits.clone();
            }
        }
        steps += 1;
        assert!(steps < 10_000, "the engine made no progress");
    }
    got.expect("one emission per question")
}

fn check_loaded(model: &Arc<Model>, what: &str) {
    let n_vocab = model.cfg.n_vocab as u32;
    let prompt: Vec<u32> = (0..12).map(|i| 100 + (i * 7) % 90).collect();
    let last = n_vocab - 1;
    let many: Vec<u32> = (0..255u32).map(|i| (i * 37 + 11) % n_vocab).collect();
    let cases: [Vec<u32>; 6] = [
        vec![0],
        vec![last],
        vec![7, 100, last],
        vec![5, 5, 17, 5],
        vec![42],
        many,
    ];
    for threads in [1usize, 4] {
        let row = full_row(model, &prompt, threads);
        for labels in &cases {
            let want: Vec<f32> = labels.iter().map(|&l| row[l as usize]).collect();
            let got = labels_via_engine(model, &prompt, labels, threads);
            assert_eq!(
                bits(&got),
                bits(&want),
                "{what}: {} labels, {threads} threads",
                labels.len()
            );
        }
    }
}

fn check_head_format(dir: &Path, fmt: Format, tied: bool, seed: u64) {
    let shape = Dense {
        tied,
        ..Dense::default()
    };
    let path = write_dense(dir, "llama", fmt, seed, shape);
    check_loaded(&load(&path), &format!("{fmt:?} tied={tied}"));
}

/// The ternary files (rotated hybrid) give the head PQ2_0 or PTQ1_0, whose full path runs
/// 8-, 16- or row kernels; the label path must agree there too.
fn check_ternary_format(dir: &Path, fmt: Format, seed: u64) {
    let path = write_hybrid(dir, fmt, seed, false);
    check_loaded(&load(&path), &format!("{fmt:?} hybrid"));
}

fn check_head_formats() {
    let dir = tmpdir(&format!(
        "head-formats-{}",
        std::env::var("PEREGRINE_ISA").unwrap_or_else(|_| "native".into())
    ));
    let mut seed = 9100u64;
    for &fmt in HEAD_FORMATS {
        for tied in [false, true] {
            seed += 1;
            check_head_format(&dir, fmt, tied, seed);
        }
    }
    for &fmt in &[Format::PQ2_0, Format::PTQ1_0] {
        seed += 1;
        check_ternary_format(&dir, fmt, seed);
    }
}

/// S4.9d: in every head format (tied and untied), the label logits are the full row's
/// entries bit for bit, on every kernel tier this machine runs. The tier is
/// process-wide and read once, so `native` runs here and `avx2`/`scalar` in children.
#[test]
fn label_rows_match_the_full_row_in_every_head_format() {
    if std::env::var_os("PEREGRINE_LABELS_FORMAT_CHILD").is_some() {
        check_head_formats();
        return;
    }
    check_head_formats();
    let exe = std::env::current_exe().expect("the test binary's path");
    for isa in ["avx2", "scalar"] {
        let status = std::process::Command::new(&exe)
            .args([
                "--exact",
                "label_rows_match_the_full_row_in_every_head_format",
                "--nocapture",
            ])
            .env("PEREGRINE_ISA", isa)
            .env("PEREGRINE_LABELS_FORMAT_CHILD", "1")
            .status()
            .expect("spawn the tier child");
        assert!(status.success(), "the PEREGRINE_ISA={isa} child failed");
    }
}

/// S4.9d with S4.9b: a wide pass packs many label rows beside each other, so the head
/// takes each label row from the `MAX_BATCH`-row block the matmul driver would hand the
/// kernel — including the batched k-quant tile kernel, which reads the interleaved
/// activation filled once per block.
#[test]
fn a_wide_pass_of_label_rows_is_bit_identical() {
    let dir = tmpdir("wide");
    for fmt in [Format::Q4_K, Format::Q5_0, Format::Q6_K, Format::Q8_0] {
        let path = write_dense(&dir, "llama", fmt, 9500, Dense::default());
        let model = load(&path);
        let prompts = forked_questions(24, &[3, 4, 5], 10);
        let want: Vec<Vec<f32>> = prompts
            .iter()
            .map(|p| reference(&model, p, &LABELS, 4))
            .collect();
        for budget in [Some(4usize), Some(16), Some(64), Some(256)] {
            let r = run_prec_pass(
                &model,
                &prompts,
                4,
                16,
                1 << 20,
                KvPrec::F16,
                Some(1),
                budget,
            );
            for (i, w) in want.iter().enumerate() {
                assert_eq!(
                    bits(&r.logits[i]),
                    bits(w),
                    "{fmt:?}: question {i} of 10, budget {budget:?}"
                );
            }
        }
    }
}
