//! Embeddings on causal models (roadmap S4.1).
//!
//! A pooled slot's vector must be exactly the pooling of the final hidden states the model
//! computes for that text — the rows llama.cpp's `result_norm` holds — and it must be
//! bit-identical whatever else shares the batch. These run on synthetic llama and qwen3 files,
//! so CI needs no model files.

mod support;

use peregrine_core::embed::{self, Pooling};
use peregrine_core::gguf::Format;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::Model;
use peregrine_core::serve::{Emission, Engine, Finish, SlotCfg};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use support::models::{write_dense_pooled, Dense};

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-embeddings-{}-{test}",
        std::process::id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &std::path::Path) -> Arc<Model> {
    let g = peregrine_core::Gguf::open(path).expect("open the synthetic model");
    Arc::new(peregrine_models::load(g).expect("load the synthetic model"))
}

/// A deterministic text of `n` token ids inside the synthetic vocabulary (all ids are valid).
fn text(seed: u32, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| 11 + (seed.wrapping_mul(37).wrapping_add(i as u32 * 13)) % 280)
        .collect()
}

/// The reference: one forward over the whole text with `want_hidden`, then the same pooling.
///
/// `want_hidden` writes `output_norm(x)` for every row — llama.cpp's `result_norm`, which its
/// pooling reads — so this is the exact vector the design says the engine must produce.
fn reference(model: &Model, tokens: &[u32], pooling: Pooling, threads: usize) -> Vec<f32> {
    let c = model.cfg;
    let pool = peregrine_core::pool::Pool::new(threads);
    let mut st = model.new_state(tokens.len(), 1, c.ctx_train, threads);
    st.want_hidden = true;
    let mut kv = model.new_cache(KvPrec::F16, c.ctx_train);
    let mut stats = peregrine_core::model::StepStats::default();
    model
        .forward(&mut st, &mut kv, &pool, tokens, 0, &[], &mut stats)
        .unwrap();
    let mut out = Vec::new();
    embed::pool_rows(
        &st.hidden[..tokens.len() * c.d_model],
        c.d_model,
        pooling,
        &mut out,
    );
    out
}

struct Run {
    /// The vector of every admitted text, in admission order.
    vectors: Vec<Vec<f32>>,
    /// Every emission the engine returned, for the chat slot's tokens.
    emissions: Vec<Emission>,
}

/// Admit every text as a pooled slot at once, run the engine to completion, and collect the
/// vectors in admission order. `extra` is an optional chat job admitted beside them.
fn embed_batch(
    model: &Arc<Model>,
    texts: &[Vec<u32>],
    pooling: Pooling,
    threads: usize,
    chunk: usize,
    ctx: usize,
    extra: Option<(Vec<u32>, SlotCfg)>,
) -> Run {
    embed_batch_pass(model, texts, pooling, threads, chunk, ctx, None, extra)
}

/// [`embed_batch`] with an explicit pass budget (`None`: the machine's default, S4.9b).
#[allow(clippy::too_many_arguments)]
fn embed_batch_pass(
    model: &Arc<Model>,
    texts: &[Vec<u32>],
    pooling: Pooling,
    threads: usize,
    chunk: usize,
    ctx: usize,
    pass_rows: Option<usize>,
    extra: Option<(Vec<u32>, SlotCfg)>,
) -> Run {
    let mut engine = Engine::new_with_chunk_and_pass(
        model.clone(),
        threads,
        ctx,
        texts.len().max(1) + usize::from(extra.is_some()),
        KvPrec::F16,
        0,
        chunk,
        pass_rows,
    );
    let mut by_slot: HashMap<u64, usize> = HashMap::new();
    for (i, t) in texts.iter().enumerate() {
        let id = engine
            .admit_pooled(t.clone(), SlotCfg::default(), pooling)
            .unwrap();
        by_slot.insert(id, i);
    }
    if let Some((prompt, cfg)) = extra {
        engine.admit(prompt, cfg).unwrap();
    }
    let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
    let mut emissions = Vec::new();
    let mut steps = 0;
    while engine.active() > 0 {
        for em in engine.step() {
            if let Some(&i) = by_slot.get(&em.slot) {
                if let Some(f) = &em.finish {
                    assert!(
                        !matches!(f, Finish::Error(_)),
                        "the pooled slot failed: {f:?}"
                    );
                }
                out[i] = em.embedding.clone();
            }
            emissions.push(em);
        }
        steps += 1;
        assert!(steps < 100_000, "the engine made no progress");
    }
    Run {
        vectors: out
            .into_iter()
            .map(|v| v.expect("a vector per text"))
            .collect(),
        emissions,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn model_files(dir: &std::path::Path) -> Vec<(&'static str, Arc<Model>)> {
    let mut out = Vec::new();
    for arch in ["llama", "qwen3"] {
        for (id, pooling) in [(3u32, Pooling::Last), (1, Pooling::Mean)] {
            let path = write_dense_pooled(dir, arch, Format::F16, 7, Dense::default(), id);
            let model = load(&path);
            assert_eq!(model.pooling().unwrap(), Some(pooling), "{arch} {id}");
            out.push((arch, model));
        }
    }
    out
}

#[test]
fn the_pooled_vector_is_the_reference_forwards_pooled_hidden_states() {
    let dir = tmpdir("reference");
    for (arch, model) in model_files(&dir) {
        for pooling in [Pooling::Last, Pooling::Mean] {
            // Skip the pair the model was not written with; both poolings must still match the
            // reference for the file's own weights (the pooling is a request, not a weight).
            for (i, len) in [1usize, 2, 7, 31].into_iter().enumerate() {
                let tokens = text(i as u32 + 1, len);
                let want = reference(&model, &tokens, pooling, 1);
                let got = embed_batch(
                    &model,
                    std::slice::from_ref(&tokens),
                    pooling,
                    1,
                    64,
                    model.cfg.ctx_train,
                    None,
                )
                .vectors;
                assert_eq!(bits(&got[0]), bits(&want), "{arch} {pooling:?} len {len}");
            }
        }
    }
}

#[test]
fn a_duplicate_in_a_batch_matches_the_text_alone() {
    // The result cache computes a repeated text once (S4.9k); the engine's own
    // invariance is what makes that safe. This pins it: the same text at two
    // places in one batch, and alone, is one vector bit for bit.
    let dir = tmpdir("duplicate-batch");
    for (arch, model) in model_files(&dir) {
        let pooling = model.pooling().unwrap().unwrap();
        let probe = text(9, 37);
        let alone = embed_batch(
            &model,
            std::slice::from_ref(&probe),
            pooling,
            4,
            16,
            model.cfg.ctx_train,
            None,
        )
        .vectors[0]
            .clone();
        let texts = vec![text(4, 11), probe.clone(), text(5, 23), probe.clone()];
        let run = embed_batch(&model, &texts, pooling, 4, 16, model.cfg.ctx_train, None);
        assert_eq!(
            bits(&run.vectors[1]),
            bits(&alone),
            "{arch} {pooling:?}: first copy of a duplicated text"
        );
        assert_eq!(
            bits(&run.vectors[3]),
            bits(&alone),
            "{arch} {pooling:?}: second copy of a duplicated text"
        );
        assert_eq!(
            bits(&run.vectors[1]),
            bits(&run.vectors[3]),
            "{arch} {pooling:?}: the two copies agree"
        );
    }
}

#[test]
fn a_texts_vector_is_bit_identical_alone_in_batches_and_beside_a_chat_slot() {
    let dir = tmpdir("invariance");
    for (arch, model) in model_files(&dir) {
        let pooling = model.pooling().unwrap().unwrap();
        // The text whose vector every arrangement must reproduce.
        let probe = text(9, 37);
        let alone = embed_batch(
            &model,
            std::slice::from_ref(&probe),
            pooling,
            1,
            64,
            model.cfg.ctx_train,
            None,
        )
        .vectors[0]
            .clone();

        // Batching: 2, 7 and 64 texts of mixed lengths, with different chunks and thread counts.
        for (n, chunk, threads) in [
            (2usize, 64usize, 1usize),
            (7, 1, 2),
            (64, 64, 4),
            (7, 64, 1),
        ] {
            let mut texts = Vec::with_capacity(n);
            for i in 0..n {
                // The probe sits at a different place in each batch; every other text has a
                // different length, so the ragged rows are not a padded rectangle.
                texts.push(if i == n / 2 {
                    probe.clone()
                } else {
                    text(100 + i as u32, 1 + (i * 7) % 23)
                });
            }
            let run = embed_batch(
                &model,
                &texts,
                pooling,
                threads,
                chunk,
                model.cfg.ctx_train,
                None,
            );
            assert_eq!(
                bits(&run.vectors[n / 2]),
                bits(&alone),
                "{arch} {pooling:?}: probe in a batch of {n}, chunk {chunk}, {threads} threads"
            );
        }

        // Beside a running chat slot: the chat slot decodes while the pooled text is prefilled,
        // so the two share mixed weight passes.
        let cfg = SlotCfg {
            max_tokens: 4,
            ..Default::default()
        };
        let run = embed_batch(
            &model,
            std::slice::from_ref(&probe),
            pooling,
            2,
            2,
            model.cfg.ctx_train,
            Some((text(55, 9), cfg)),
        );
        assert_eq!(
            bits(&run.vectors[0]),
            bits(&alone),
            "{arch} {pooling:?}: probe beside a chat slot"
        );
        assert!(
            run.emissions
                .iter()
                .any(|e| e.finish.is_some() && e.embedding.is_some()),
            "the pooled slot finished"
        );
        assert!(
            run.emissions
                .iter()
                .filter(|e| e.embedding.is_none())
                .count()
                >= 4,
            "the chat slot decoded its four tokens"
        );
    }
}

#[test]
fn pooling_defaults_come_from_the_file_and_cls_is_refused() {
    let dir = tmpdir("pooling");
    for (arch, id, want) in [
        ("llama", 1u32, Some(Pooling::Mean)),
        ("llama", 3, Some(Pooling::Last)),
        ("qwen3", 0, None),
        ("qwen3", 4, None),
    ] {
        let path = write_dense_pooled(&dir, arch, Format::F16, 5, Dense::default(), id);
        let model = load(&path);
        assert_eq!(model.pooling().unwrap(), want, "{arch} pooling_type {id}");
    }
    // A model with no pooling type cannot serve a pooled request by default, and `Cls` is
    // not implemented for causal models.
    let path = write_dense_pooled(&dir, "llama", Format::F16, 5, Dense::default(), 3);
    let model = load(&path);
    let mut engine = Engine::new(model.clone(), 1, 128, 1, KvPrec::F16, 0);
    let err = engine
        .admit_pooled(vec![11, 12], SlotCfg::default(), Pooling::Cls)
        .unwrap_err();
    assert!(err.1.contains("cls"), "{err:?}");
}

#[test]
fn a_pooled_slot_that_finishes_in_a_decode_pass_is_reaped() {
    // A pooled slot that completes in a pass which also decodes a chat slot must leave the slot
    // list: `reap` used to be handed the pooled emissions after they had already moved into the
    // output, so the slot stayed forever, `active()` never reached zero and the next admission
    // was refused.
    let dir = tmpdir("reap");
    let path = write_dense_pooled(&dir, "llama", Format::F16, 7, Dense::default(), 3);
    let model = load(&path);
    let mut engine =
        Engine::new_with_chunk(model.clone(), 1, model.cfg.ctx_train, 2, KvPrec::F16, 0, 2);
    let chat = engine
        .admit(
            text(55, 9),
            SlotCfg {
                max_tokens: 500,
                ..Default::default()
            },
        )
        .unwrap();

    // Once the chat slot has emitted its first token it is decoding, so every following pass
    // has a decode row and the pooled text finishes in a mixed pass.
    let mut decoding = false;
    for _ in 0..100 {
        for em in engine.step() {
            if em.slot == chat && em.finish.is_none() {
                decoding = true;
            }
        }
        if decoding {
            break;
        }
    }
    assert!(decoding, "the chat slot never decoded");

    let pooled = engine
        .admit_pooled(text(9, 37), SlotCfg::default(), Pooling::Last)
        .unwrap();
    let mut vector = None;
    for _ in 0..1000 {
        for em in engine.step() {
            if em.slot == pooled {
                assert!(
                    !matches!(em.finish, Some(Finish::Error(_))),
                    "{:?}",
                    em.finish
                );
                vector = em.embedding;
            }
        }
        if vector.is_some() {
            break;
        }
    }
    let vector = vector.expect("the pooled slot never finished");
    assert_eq!(vector.len(), model.cfg.d_model);
    assert_eq!(engine.active(), 1, "only the chat slot is still active");

    // Its place is free again: the engine admits another text.
    let next = engine
        .admit_pooled(text(10, 5), SlotCfg::default(), Pooling::Last)
        .expect("a finished pooled slot must free its place");
    engine.cancel(next);
    engine.cancel(chat);
}

// ---------------------------------------------------------------------------
// S4.4: the bidirectional encoder, `gemma-embedding`
// ---------------------------------------------------------------------------

fn gemma_embedding(dir: &std::path::Path, fmt: Format, seed: u64) -> Arc<Model> {
    let path = support::models::write_gemma_embedding(
        dir,
        fmt,
        seed,
        support::models::GemmaEmbeddingConfig::default(),
    );
    load(&path)
}

/// A text whose length exercises both the sliding window (8 tokens) and the global
/// layer: the probe is longer than the window so both mask rules run.
fn encoder_text(seed: u32, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| 7 + (seed.wrapping_mul(53).wrapping_add(i as u32 * 17)) % 280)
        .collect()
}

#[test]
fn a_gemma_embedding_file_reports_itself_as_an_encoder() {
    let dir = tmpdir("encoder-pooling");
    let model = gemma_embedding(&dir, Format::F32, 41);
    assert_eq!(
        model.causality(),
        peregrine_core::arch::Causality::Bidirectional
    );
    assert_eq!(model.pooling().unwrap(), Some(Pooling::Mean));
    assert_eq!(model.embedding_width(), model.cfg.d_model);
}

#[test]
fn the_pooled_vector_is_the_exact_scalar_reference_forward() {
    let dir = tmpdir("encoder-reference");
    let model = gemma_embedding(&dir, Format::F32, 42);
    for pooling in [Pooling::Mean, Pooling::Cls, Pooling::Last] {
        for (i, len) in [1usize, 2, 7, 31, 37, 64, 100].into_iter().enumerate() {
            let text = encoder_text(1 + i as u32, len);
            let want = support::embed_ref::reference_embedding(&model, &text, pooling);
            let got = embed_batch(
                &model,
                std::slice::from_ref(&text),
                pooling,
                1,
                64,
                model.cfg.ctx_train,
                None,
            )
            .vectors;
            assert_eq!(
                bits(&got[0]),
                bits(&want),
                "{pooling:?} len {len}: the engine differs from the reference"
            );
        }
    }
}

#[test]
fn an_encoders_vector_is_bit_identical_alone_in_batches_and_in_any_order() {
    let dir = tmpdir("encoder-invariance");
    // Both an exact F32 file and the format a real file ships in exercise the same
    // scheduling: the vector is a function of the text and its own length only.
    for (fmt, seed) in [(Format::F32, 43u64), (Format::Q8_0, 44)] {
        let model = gemma_embedding(&dir, fmt, seed);
        let pooling = Pooling::Mean;
        let probe = encoder_text(9, 37);
        let alone = embed_batch(
            &model,
            std::slice::from_ref(&probe),
            pooling,
            1,
            64,
            model.cfg.ctx_train,
            None,
        )
        .vectors[0]
            .clone();
        // A Cls-pooled text beside it is a different reduction over the same rows.
        let _ = embed_batch(
            &model,
            std::slice::from_ref(&encoder_text(3, 5)),
            Pooling::Cls,
            1,
            64,
            model.cfg.ctx_train,
            None,
        );

        for (n, threads) in [(2usize, 1usize), (7, 2), (64, 4), (7, 1), (64, 8)] {
            let mut texts = Vec::with_capacity(n);
            for i in 0..n {
                texts.push(if i == n / 2 {
                    probe.clone()
                } else {
                    // Different lengths on both sides of the window, so the ragged
                    // rows are not a padded rectangle.
                    encoder_text(100 + i as u32, 1 + (i * 11) % 53)
                });
            }
            // Admit the probe first in one arrangement and last in another: the order
            // of admission must not matter either.
            if n % 2 == 0 {
                texts.swap(0, n / 2);
            }
            let run = embed_batch(
                &model,
                &texts,
                pooling,
                threads,
                64,
                model.cfg.ctx_train,
                None,
            );
            let probe_at = texts.iter().position(|t| t == &probe).unwrap();
            assert_eq!(
                bits(&run.vectors[probe_at]),
                bits(&alone),
                "{}: probe in a batch of {n}, {threads} threads, admitted at {probe_at}",
                fmt.label()
            );
        }

        // Short texts too: a one- or two-row pass can take a different matmul kernel than a
        // wide batch, and the vector must not notice.
        for len in [1usize, 2, 3] {
            let short = encoder_text(31 + len as u32, len);
            let short_alone = embed_batch(
                &model,
                std::slice::from_ref(&short),
                pooling,
                1,
                64,
                model.cfg.ctx_train,
                None,
            )
            .vectors[0]
                .clone();
            let mut texts = vec![short.clone()];
            for i in 0..7 {
                texts.push(encoder_text(200 + i, 13 + i as usize));
            }
            let run = embed_batch(&model, &texts, pooling, 4, 64, model.cfg.ctx_train, None);
            assert_eq!(
                bits(&run.vectors[0]),
                bits(&short_alone),
                "{}: a {len}-token text is the same alone and in a batch",
                fmt.label()
            );
        }

        // A text long enough to run alone in a whole pass, beside shorts that fit one.
        let long = encoder_text(77, 200);
        let long_alone = embed_batch(
            &model,
            std::slice::from_ref(&long),
            pooling,
            2,
            64,
            model.cfg.ctx_train,
            None,
        )
        .vectors[0]
            .clone();
        let mut mixed = vec![encoder_text(4, 3), encoder_text(5, 5), long.clone()];
        let run = embed_batch(&model, &mixed, pooling, 4, 64, model.cfg.ctx_train, None);
        assert_eq!(
            bits(&run.vectors[2]),
            bits(&long_alone),
            "{}: the long text is the same alone and in a mixed pass",
            fmt.label()
        );
        mixed.clear();
    }
}

#[test]
fn a_chat_slot_is_refused_by_an_encoder() {
    let dir = tmpdir("encoder-chat");
    let model = gemma_embedding(&dir, Format::F32, 45);
    let mut engine = Engine::new(model, 1, 64, 1, KvPrec::F16, 0);
    let err = engine
        .admit(encoder_text(1, 8), SlotCfg::default())
        .unwrap_err();
    assert!(
        err.1.contains("embedding model") && err.1.contains("cannot generate"),
        "{err:?}"
    );
    // The encoder's slot carries no cache, so an embedding slot is admitted.
    let id = engine
        .admit_pooled(encoder_text(2, 8), SlotCfg::default(), Pooling::Mean)
        .unwrap();
    engine.cancel(id);
}

#[test]
fn a_label_slot_is_refused_by_an_encoder() {
    let dir = tmpdir("encoder-labels");
    let model = gemma_embedding(&dir, Format::F32, 47);
    let mut engine = Engine::new(model, 1, 64, 1, KvPrec::F16, 0);
    // A label read-out is a decoder's last row's logits (S4.3); an encoder has no
    // decode path and no logits to read.
    let err = engine
        .admit_labels(encoder_text(2, 8), SlotCfg::default(), vec![1, 2])
        .unwrap_err();
    assert!(
        err.1.contains("embedding model") && err.1.contains("label logits"),
        "{err:?}"
    );
    // The refusal leaves the engine free for the slot it does serve.
    let id = engine
        .admit_pooled(encoder_text(3, 8), SlotCfg::default(), Pooling::Mean)
        .unwrap();
    engine.cancel(id);
}

/// S4.9b: the pass budget is scheduling only. A text's vector is bit-identical alone and in
/// batches at budgets 1, 16, 64, 256 and the default, at 1, 2 and 4 threads, with prefill chunks
/// 1 and 16, and beside a chat slot that shares the pass.
#[test]
fn a_texts_vector_is_bit_identical_for_every_pass_budget() {
    let dir = tmpdir("budget-invariance");
    for (arch, model) in model_files(&dir) {
        let pooling = model.pooling().unwrap().unwrap();
        let probe = text(9, 37);
        let alone = embed_batch_pass(
            &model,
            std::slice::from_ref(&probe),
            pooling,
            1,
            64,
            model.cfg.ctx_train,
            None,
            None,
        )
        .vectors[0]
            .clone();
        for budget in [Some(1usize), Some(16), Some(64), Some(256), None] {
            for threads in [1usize, 2, 4] {
                for chunk in [1usize, 16] {
                    let n = 7;
                    let mut texts = Vec::with_capacity(n);
                    for i in 0..n {
                        texts.push(if i == n / 2 {
                            probe.clone()
                        } else {
                            text(100 + i as u32, 1 + (i * 7) % 23)
                        });
                    }
                    let run = embed_batch_pass(
                        &model,
                        &texts,
                        pooling,
                        threads,
                        chunk,
                        model.cfg.ctx_train,
                        budget,
                        None,
                    );
                    assert_eq!(
                        bits(&run.vectors[n / 2]),
                        bits(&alone),
                        "{arch} {pooling:?}: probe in a batch of {n}, budget {budget:?}, \
                         {threads} threads, chunk {chunk}"
                    );
                }
            }
        }
        // A wider batch (the batch of 64 the earlier test uses) at the budgets where it runs
        // in a few passes each; budget 1 would be one text per pass.
        for budget in [Some(64usize), Some(256), None] {
            for threads in [2usize, 4] {
                let n = 64;
                let mut texts = Vec::with_capacity(n);
                for i in 0..n {
                    texts.push(if i == n / 2 {
                        probe.clone()
                    } else {
                        text(300 + i as u32, 1 + (i * 7) % 23)
                    });
                }
                let run = embed_batch_pass(
                    &model,
                    &texts,
                    pooling,
                    threads,
                    64,
                    model.cfg.ctx_train,
                    budget,
                    None,
                );
                assert_eq!(
                    bits(&run.vectors[n / 2]),
                    bits(&alone),
                    "{arch} {pooling:?}: probe in a batch of {n}, budget {budget:?}, {threads} threads"
                );
            }
        }
        // Beside a decoding chat slot, at every budget: the pass is shared with its decode rows.
        for budget in [Some(1usize), Some(16), Some(64), Some(256), None] {
            let cfg = SlotCfg {
                max_tokens: 3,
                ..Default::default()
            };
            let run = embed_batch_pass(
                &model,
                std::slice::from_ref(&probe),
                pooling,
                2,
                4,
                model.cfg.ctx_train,
                budget,
                Some((text(55, 9), cfg)),
            );
            assert_eq!(
                bits(&run.vectors[0]),
                bits(&alone),
                "{arch} {pooling:?}: probe beside a chat slot, budget {budget:?}"
            );
        }
    }
}

/// The encoder's whole-text pass uses the same budget: a batch of short texts fills a pass and
/// a text is bit-identical at every budget and thread count.
#[test]
fn an_encoders_vector_is_bit_identical_for_every_pass_budget() {
    let dir = tmpdir("encoder-budget");
    for fmt in [Format::F32, Format::Q8_0] {
        let model = gemma_embedding(&dir, fmt, 71 + usize::from(fmt == Format::Q8_0) as u64);
        let pooling = Pooling::Mean;
        let probe = encoder_text(9, 37);
        let alone = embed_batch_pass(
            &model,
            std::slice::from_ref(&probe),
            pooling,
            1,
            64,
            model.cfg.ctx_train,
            None,
            None,
        )
        .vectors[0]
            .clone();
        for budget in [Some(1usize), Some(16), Some(64), Some(256), None] {
            for threads in [1usize, 2, 4] {
                let n = 7;
                let mut texts = Vec::with_capacity(n);
                for i in 0..n {
                    texts.push(if i == n / 2 {
                        probe.clone()
                    } else {
                        encoder_text(100 + i as u32, 1 + (i * 13) % 29)
                    });
                }
                let run = embed_batch_pass(
                    &model,
                    &texts,
                    pooling,
                    threads,
                    64,
                    model.cfg.ctx_train,
                    budget,
                    None,
                );
                assert_eq!(
                    bits(&run.vectors[n / 2]),
                    bits(&alone),
                    "{}: probe in a batch of {n}, budget {budget:?}, {threads} threads",
                    fmt.label()
                );
            }
        }
    }
}

/// Run `texts` as pooled slots and count the engine's steps (one weight pass each). With no chat
/// slot in flight, every step is a non-decoding pass.
fn steps_to_finish(
    model: &Arc<Model>,
    texts: &[Vec<u32>],
    pooling: Pooling,
    threads: usize,
    chunk: usize,
    pass_rows: Option<usize>,
) -> usize {
    let mut engine = Engine::new_with_chunk_and_pass(
        model.clone(),
        threads,
        model.cfg.ctx_train,
        1,
        KvPrec::F16,
        0,
        chunk,
        pass_rows,
    );
    for t in texts {
        engine
            .admit_pooled(t.clone(), SlotCfg::default(), pooling)
            .expect("admit a pooled slot");
    }
    let mut steps = 0;
    while engine.active() > 0 {
        engine.step();
        steps += 1;
        assert!(steps < 100_000, "the engine made no progress");
    }
    steps
}

/// S4.9b: 32 texts of 16 tokens (512 rows) take ceil(rows / budget) passes, not one per text.
#[test]
fn a_non_decoding_pass_carries_the_whole_budget() {
    let dir = tmpdir("pass-count");
    let texts: Vec<Vec<u32>> = (0..32).map(|i| text(200 + i, 16)).collect();
    let total: usize = 32 * 16;
    for (arch, model) in model_files(&dir) {
        let pooling = model.pooling().unwrap().unwrap();
        for budget in [16usize, 64, 256] {
            let steps = steps_to_finish(&model, &texts, pooling, 2, 16, Some(budget));
            assert_eq!(
                steps,
                total.div_ceil(budget),
                "{arch}: 32 texts at budget {budget}"
            );
        }
    }
    // The encoder packs whole texts: 32 eight-token texts are ceil(256 / budget) passes.
    let model = gemma_embedding(&dir, Format::F32, 72);
    let short: Vec<Vec<u32>> = (0..32).map(|i| encoder_text(210 + i, 8)).collect();
    for budget in [16usize, 64, 256] {
        let steps = steps_to_finish(&model, &short, Pooling::Mean, 2, 64, Some(budget));
        assert_eq!(
            steps,
            (32 * 8usize).div_ceil(budget),
            "encoder: 32 eight-token texts at budget {budget}"
        );
    }
}

/// S4.9b: a pooled slot's cache is sized to its prompt, rounded to whole prefill chunks, not to
/// the model's context; a chat slot keeps the context.
#[test]
fn a_pooled_slots_cache_is_sized_to_its_prompt() {
    let dir = tmpdir("prompt-cache");
    let dense = Dense {
        ctx: 2048,
        ..Default::default()
    };
    let path = write_dense_pooled(&dir, "llama", Format::F16, 73, dense, 3);
    let model = load(&path);
    let per_position = peregrine_core::kv::KvCache::new(
        KvPrec::F16,
        model.cfg.kv_layers(),
        model.cfg.n_kv_head,
        model.cfg.head_dim,
        1,
    )
    .bytes_per_position();
    let mut engine = Engine::new(model.clone(), 1, 2048, 1, KvPrec::F16, 0);
    let context_cache = engine.cache_bytes_per_slot();
    assert_eq!(context_cache, per_position * 2048);

    let id = engine
        .admit_pooled(text(1, 16), SlotCfg::default(), Pooling::Last)
        .unwrap();
    assert_eq!(
        engine.slot_cache_bytes(id).unwrap(),
        per_position * 16,
        "a 16-token pooled prompt maps 16 positions"
    );
    let id = engine
        .admit_pooled(text(2, 100), SlotCfg::default(), Pooling::Last)
        .unwrap();
    assert_eq!(
        engine.slot_cache_bytes(id).unwrap(),
        per_position * 112,
        "a 100-token prompt rounds up to whole prefill chunks"
    );
    assert_eq!(
        engine.cache_bytes_for_prompt(100),
        per_position * 112,
        "the admission budget counts the same bytes"
    );
    let chat = engine
        .admit(
            text(3, 10),
            SlotCfg {
                max_tokens: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        engine.slot_cache_bytes(chat).unwrap(),
        context_cache,
        "a chat slot still maps the context it may decode into"
    );
}

/// S4.9b: embedding texts do not consume chat slots and do not wait for them. One chat slot
/// (`--slots 1`) is busy decoding; 64 pooled texts are admitted anyway (the work budget, a
/// memory bound, not a slot count), and the chat's tokens come out while the batch is still
/// running.
#[test]
fn a_wide_embedding_batch_is_admitted_beside_a_chat_slot() {
    let dir = tmpdir("work-admission");
    let path = write_dense_pooled(&dir, "llama", Format::F16, 75, Dense::default(), 3);
    let model = load(&path);
    let pooling = Pooling::Last;
    let mut engine = Engine::new_with_chunk_and_pass(
        model.clone(),
        2,
        model.cfg.ctx_train,
        1,
        KvPrec::F16,
        0,
        16,
        Some(16),
    );
    let chat = engine
        .admit(
            text(55, 9),
            SlotCfg {
                max_tokens: 4,
                ..Default::default()
            },
        )
        .expect("the one chat place");
    assert!(!engine.has_room(), "the only chat place is taken");
    let err = engine
        .admit(text(56, 9), SlotCfg::default())
        .expect_err("a second chat slot is refused");
    assert!(err.1.contains("busy"), "{err:?}");

    let n = 64;
    for i in 0..n {
        engine
            .admit_pooled(text(600 + i as u32, 16), SlotCfg::default(), pooling)
            .expect("a pooled slot beside the chat slot");
    }
    assert_eq!(
        engine.active(),
        n + 1,
        "the chat slot and every text are in"
    );
    assert_eq!(engine.work_rows(), n * 16);

    let mut done = 0usize;
    let mut first_chat_token_at: Option<usize> = None;
    let mut chat_tokens = 0usize;
    let mut steps = 0usize;
    while engine.active() > 0 {
        for em in engine.step() {
            if em.slot == chat {
                if em.finish.is_none() {
                    chat_tokens += 1;
                    first_chat_token_at.get_or_insert(done);
                }
            } else if em.embedding.is_some() {
                done += 1;
            }
        }
        steps += 1;
        assert!(steps < 100_000, "the engine made no progress");
    }
    assert_eq!(done, n, "every text of the batch answered");
    assert!(chat_tokens >= 1, "the chat slot decoded");
    let at = first_chat_token_at.expect("the chat slot decoded");
    assert!(
        at < n,
        "the chat's first token came after all {n} texts; the batch starved it"
    );
}

/// The synthetic encoder's golden for the current kernel tier.
///
/// The vector is the whole output, so the golden hashes every bit of the two vectors
/// (mean- and cls-pooled, 384 and 512 dimensions after Matryoshka truncation) instead of
/// logits. Recorded with `PEREGRINE_GOLDEN=record` on each pinned tier, like the
/// synthetic goldens.
#[test]
fn the_synthetic_encoder_matches_its_golden_for_the_tier() {
    let dir = tmpdir("encoder-golden");
    let model = gemma_embedding(&dir, Format::Q8_0, 46);

    let mut out = String::new();
    for (pooling, dims) in [(Pooling::Mean, None), (Pooling::Cls, Some(128usize))] {
        let text = encoder_text(21, 37);
        let run = embed_batch(
            &model,
            std::slice::from_ref(&text),
            pooling,
            3,
            64,
            model.cfg.ctx_train,
            None,
        );
        let mut v = run.vectors[0].clone();
        embed::finish(&mut v, dims).unwrap();
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for x in &v {
            for b in x.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        use std::fmt::Write as _;
        writeln!(
            out,
            "{} {dims:?} len {} fnv {h:016x}",
            pooling.name(),
            v.len()
        )
        .unwrap();
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
    let gpath = dir.join("gemma-embedding-Q8_0.golden");
    let record = std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "record");
    let require = std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "require");
    if record && tier != "x86_64-avx2-vnni256" {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&gpath, &out).unwrap();
        eprintln!("synthetic: recorded gemma-embedding-Q8_0 ({tier})");
    } else if let Ok(want) = std::fs::read_to_string(&gpath) {
        assert_eq!(want, out, "{}", gpath.display());
    } else if require {
        panic!(
            "gemma-embedding-Q8_0 has no golden for tier {tier} ({}); record it with PEREGRINE_GOLDEN=record",
            gpath.display()
        );
    }
}

/// EmbeddingGemma 300M against llama.cpp v0.9.0, when the file is on this machine.
///
/// It checks the tokenizer ids, the batch invariance and the two Matryoshka truncations
/// against the reference forward; the cross-engine cosine table is in the PR body and
/// `docs/reference/gemma-embedding.md`. `cargo test --release -p peregrine-models
/// --test embeddings -- --ignored` runs it.
#[test]
#[ignore = "needs embeddinggemma-300M-Q8_0.gguf in models/, ~/models/ or the peregrine cache"]
fn embeddinggemma_real_file_is_batch_invariant() {
    const FILE: &str = "embeddinggemma-300M-Q8_0.gguf";
    let mut candidates = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(FILE)];
    if let Ok(h) = std::env::var("HOME") {
        candidates.push(PathBuf::from(&h).join("models").join(FILE));
        candidates.push(
            PathBuf::from(&h)
                .join(".cache/peregrine/models/ggml-org/embeddinggemma-300M-GGUF")
                .join(FILE),
        );
    }
    let Some(path) = candidates.into_iter().find(|p| p.is_file()) else {
        eprintln!("{FILE} is not on this machine; skipped");
        return;
    };
    let model = load(&path);
    assert_eq!(model.pooling().unwrap(), Some(Pooling::Mean));
    assert!(model.tk.add_bos && model.tk.add_eos);
    let probe = model
        .tk
        .encode_special("task: search result | query: What is the capital of France?");
    let alone = embed_batch(
        &model,
        std::slice::from_ref(&probe),
        Pooling::Mean,
        8,
        512,
        512,
        None,
    );
    let mut a = alone.vectors[0].clone();
    embed::finish(&mut a, None).unwrap();
    assert!((norm(&a) - 1.0).abs() < 1e-5);
    let texts: Vec<Vec<u32>> = [
        "Paris.",
        "The quick brown fox jumps over the lazy dog.",
        "task: clustering | query: machine learning",
    ]
    .iter()
    .map(|t| model.tk.encode_special(t))
    .collect();
    let mut all = vec![probe];
    all.extend(texts.iter().cloned());
    let batch = embed_batch(&model, &all, Pooling::Mean, 8, 512, 512, None);
    let mut b0 = batch.vectors[0].clone();
    embed::finish(&mut b0, None).unwrap();
    assert_eq!(bits(&b0), bits(&a), "the probe is batch-invariant");
    for (i, t) in texts.iter().enumerate() {
        let one = embed_batch(
            &model,
            std::slice::from_ref(t),
            Pooling::Mean,
            4,
            512,
            512,
            None,
        );
        let mut x = one.vectors[0].clone();
        let mut y = batch.vectors[i + 1].clone();
        embed::finish(&mut x, None).unwrap();
        embed::finish(&mut y, None).unwrap();
        assert_eq!(bits(&x), bits(&y), "text {i} is batch-invariant");
    }
}

/// Qwen3-Embedding 0.6B, the S4.1 real-file check, when the file is on this machine.
///
/// The file is looked up where the real-model goldens look (`models/`, `~/models/`) and in the
/// `peregrine` model cache, which is where `peregrine embed -m hf:...` puts it. `cargo test
/// --release -- --ignored` runs it; CI has no model files, so it is skipped by default.
///
/// What is pinned: the file's own pooling type and EOS flag, the `Last` vector against the
/// exact reference forward, and batch invariance on a real 28-layer model.
#[test]
#[ignore = "needs Qwen3-Embedding-0.6B-Q8_0.gguf in models/, ~/models/ or the peregrine cache"]
fn qwen3_embedding_real_file_matches_the_reference_and_is_batch_invariant() {
    const FILE: &str = "Qwen3-Embedding-0.6B-Q8_0.gguf";
    let mut candidates = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(FILE)];
    if let Ok(h) = std::env::var("HOME") {
        candidates.push(PathBuf::from(&h).join("models").join(FILE));
        candidates.push(
            PathBuf::from(&h)
                .join(".cache/peregrine/models/Qwen/Qwen3-Embedding-0.6B-GGUF")
                .join(FILE),
        );
    }
    let Some(path) = candidates.into_iter().find(|p| p.is_file()) else {
        eprintln!("{FILE} is not on this machine; skipped");
        return;
    };
    let model = load(&path);
    assert_eq!(model.pooling().unwrap(), Some(Pooling::Last));
    assert_eq!(model.embedding_width(), 1024);
    // Qwen3-Embedding pools the appended EOS token, so a plain encoding must end with it.
    assert!(model.tk.add_eos, "tokenizer.ggml.add_eos_token is set");
    let tokens = model.tk.encode_special("What is the capital of France?");
    assert_eq!(tokens.last(), Some(&model.tk.eos));

    let text = tokens;
    let want = reference(&model, &text, Pooling::Last, 12);
    let mut want = want;
    embed::finish(&mut want, None).unwrap();
    let alone = embed_batch(
        &model,
        std::slice::from_ref(&text),
        Pooling::Last,
        12,
        512,
        512,
        None,
    );
    let mut got = alone.vectors[0].clone();
    embed::finish(&mut got, None).unwrap();
    assert_eq!(bits(&got), bits(&want), "the vector is the reference's");
    assert!((norm(&got) - 1.0).abs() < 1e-5);

    // Two more texts of different lengths; every vector the same as it is alone.
    let other: Vec<Vec<u32>> = ["The quick brown fox jumps over the lazy dog.", "Paris."]
        .iter()
        .map(|t| model.tk.encode_special(t))
        .collect();
    let mut texts = vec![text.clone()];
    texts.extend(other.iter().cloned());
    let batch = embed_batch(&model, &texts, Pooling::Last, 12, 256, 512, None);
    let mut b0 = batch.vectors[0].clone();
    embed::finish(&mut b0, None).unwrap();
    assert_eq!(bits(&b0), bits(&got), "bit-identical in a batch of three");
    for (i, t) in other.iter().enumerate() {
        let one = embed_batch(
            &model,
            std::slice::from_ref(t),
            Pooling::Last,
            4,
            256,
            512,
            None,
        );
        let mut a = one.vectors[0].clone();
        let mut b = batch.vectors[i + 1].clone();
        embed::finish(&mut a, None).unwrap();
        embed::finish(&mut b, None).unwrap();
        assert_eq!(bits(&a), bits(&b), "text {i} is batch-invariant");
    }
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

/// Cosine similarity to the same vector truncated to `dims` and normalised again (Matryoshka).
fn cosine_at(a: &[f32], b: &[f32], dims: usize) -> f64 {
    let dot = |x: &[f32], y: &[f32]| {
        x[..dims]
            .iter()
            .zip(&y[..dims])
            .map(|(&p, &q)| (p as f64) * (q as f64))
            .sum::<f64>()
    };
    let n = |x: &[f32]| {
        x[..dims]
            .iter()
            .map(|&p| (p as f64) * (p as f64))
            .sum::<f64>()
            .sqrt()
    };
    dot(a, b) / (n(a) * n(b))
}

fn read_vectors(path: &std::path::Path) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).unwrap();
    let (rows, rest) = bytes.as_chunks::<3072>();
    assert!(
        rest.is_empty(),
        "{}: not a whole number of 768-d rows",
        path.display()
    );
    rows.iter()
        .map(|row| {
            let (words, rest) = row.as_chunks::<4>();
            assert!(rest.is_empty());
            words.iter().map(|b| f32::from_le_bytes(*b)).collect()
        })
        .collect()
}

/// The cross-engine half of the gate: against llama.cpp's vectors for the same 200 texts.
///
/// `llama_q8_0.f32` is llama.cpp v0.9.0's output on `embeddinggemma-300M-Q8_0.gguf` (the file
/// this test loads) and `llama_f16.f32` is its output on the F16 conversion of the same weights
/// — the full-precision reference with `dense_2`/`dense_3`. The per-format bound is the one
/// measured for this file and written into the design note §5 item 2: Peregrine's Q8_0 is within
/// 0.998 (min) / 0.9995 (mean) of llama.cpp's Q8_0 at 768 dimensions and 0.9985 / 0.9996 at 256,
/// and no further from the F16 reference than llama.cpp's Q8_0 is (1e-4 slack). The fixtures and
/// their provenance are in `tests/fixtures/embeddinggemma/`.
#[test]
#[ignore = "needs embeddinggemma-300M-Q8_0.gguf in models/, ~/models/ or the peregrine cache"]
fn embeddinggemma_real_file_is_within_the_bound_of_llama_cpp() {
    const FILE: &str = "embeddinggemma-300M-Q8_0.gguf";
    let mut candidates = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(FILE)];
    if let Ok(h) = std::env::var("HOME") {
        candidates.push(PathBuf::from(&h).join("models").join(FILE));
        candidates.push(
            PathBuf::from(&h)
                .join(".cache/peregrine/models/ggml-org/embeddinggemma-300M-GGUF")
                .join(FILE),
        );
    }
    let Some(path) = candidates.into_iter().find(|p| p.is_file()) else {
        eprintln!("{FILE} is not on this machine; skipped");
        return;
    };
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embeddinggemma");
    let texts: Vec<String> = std::fs::read_to_string(dir.join("texts.txt"))
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(texts.len(), 200);
    let llama_q8 = read_vectors(&dir.join("llama_q8_0.f32"));
    let llama_f16 = read_vectors(&dir.join("llama_f16.f32"));
    assert_eq!(llama_q8.len(), texts.len());
    assert_eq!(llama_f16.len(), texts.len());

    let model = load(&path);
    let tokens: Vec<Vec<u32>> = texts.iter().map(|t| model.tk.encode_special(t)).collect();
    let run = embed_batch(&model, &tokens, Pooling::Mean, 8, 512, 512, None);
    let mut ours = run.vectors;
    for v in &mut ours {
        embed::finish(v, None).unwrap();
    }

    for dims in [768usize, 256] {
        let to_llama: Vec<f64> = ours
            .iter()
            .zip(&llama_q8)
            .map(|(a, b)| cosine_at(a, b, dims))
            .collect();
        let to_ref: Vec<f64> = ours
            .iter()
            .zip(&llama_f16)
            .map(|(a, b)| cosine_at(a, b, dims))
            .collect();
        let llama_to_ref: Vec<f64> = llama_q8
            .iter()
            .zip(&llama_f16)
            .map(|(a, b)| cosine_at(a, b, dims))
            .collect();
        let min_mean = |v: &[f64]| {
            (
                v.iter().copied().fold(f64::INFINITY, f64::min),
                v.iter().sum::<f64>() / v.len() as f64,
            )
        };
        let (min_l, mean_l) = min_mean(&to_llama);
        let (min_r, mean_r) = min_mean(&to_ref);
        let (min_rl, mean_rl) = min_mean(&llama_to_ref);
        let (want_min, want_mean) = if dims == 768 {
            (0.998, 0.9995)
        } else {
            (0.9985, 0.9996)
        };
        assert!(
            min_l >= want_min && mean_l >= want_mean,
            "{dims}-d: Peregrine Q8_0 vs llama.cpp Q8_0 min {min_l:.6} mean {mean_l:.6}, \
             bound {want_min} / {want_mean}"
        );
        // The mean is the accuracy statement; the min is a gross-break check. An encoder's
        // Q8_0 path still quantises activations to int8 (decision 2 of the S4.9 design makes
        // that an S4.9k report), and at 127 levels a change of one ulp anywhere re-rolls the
        // quantised code of a few texts: when S4.9g replaced attention's arithmetic the two
        // engines' vectors moved by up to 1.2e-3 on 5 of these 200 texts (the mean moved by
        // 1e-5, and the agreement with llama.cpp's Q8_0 *improved*, min 0.998168 -> 0.999099
        // at 768 d). A 1e-4 slack on one order statistic tested that lottery, not the graph.
        let min_slack = 1.5e-3;
        let mean_slack = 1e-4;
        assert!(
            min_r >= min_rl - min_slack && mean_r >= mean_rl - mean_slack,
            "{dims}-d: Peregrine Q8_0 vs the F16 reference min {min_r:.6} mean {mean_r:.6}, \
             llama.cpp's Q8_0 min {min_rl:.6} mean {mean_rl:.6}"
        );
        eprintln!(
            "{dims}-d: vs llama.cpp Q8_0 min {min_l:.6} mean {mean_l:.7}; \
             vs the F16 reference Peregrine min {min_r:.6} mean {mean_r:.7}, \
             llama.cpp min {min_rl:.6} mean {mean_rl:.7}"
        );
    }
}

// ---------------------------------------------------------------------------
// S4.6a: the `bert` family (MiniLM, BGE, E5, GTE and the XLM-R models)
// ---------------------------------------------------------------------------

/// A synthetic `bert` file: post-norm blocks, absolute positions, token types,
/// biases on every projection, LayerNorm and the exact GELU.
fn bert_model(dir: &std::path::Path, fmt: Format, seed: u64, pooling: u32) -> Arc<Model> {
    let path = support::models::write_bert(
        dir,
        fmt,
        seed,
        support::models::BertConfig {
            pooling_type: pooling,
            ..Default::default()
        },
    );
    load(&path)
}

/// A text of token ids inside the synthetic vocabulary (all ids are valid).
fn bert_text(seed: u32, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| 6 + (seed.wrapping_mul(59).wrapping_add(i as u32 * 19)) % 300)
        .collect()
}

#[test]
fn a_bert_file_reports_itself_as_an_encoder_with_its_pooling() {
    let dir = tmpdir("bert-encoder");
    for (id, want) in [(2u32, Pooling::Cls), (1, Pooling::Mean)] {
        let model = bert_model(&dir, Format::F32, 41, id);
        assert_eq!(
            model.causality(),
            peregrine_core::arch::Causality::Bidirectional
        );
        assert_eq!(model.pooling().unwrap(), Some(want), "pooling_type {id}");
        assert_eq!(model.embedding_width(), model.cfg.d_model);
        // The position table is the context: the text limit is the file's.
        assert_eq!(
            model.cfg.ctx_train,
            support::models::BertConfig::default().ctx
        );
    }
}

#[test]
fn the_bert_pooled_vector_is_the_exact_scalar_reference_forward() {
    let dir = tmpdir("bert-reference");
    let model = bert_model(&dir, Format::F32, 42, 2);
    for pooling in [Pooling::Cls, Pooling::Mean, Pooling::Last] {
        for (i, len) in [1usize, 2, 7, 31, 64].into_iter().enumerate() {
            let text = bert_text(1 + i as u32, len);
            let want = support::bert_ref::reference_embedding(&model, &text, pooling);
            let got = embed_batch(
                &model,
                std::slice::from_ref(&text),
                pooling,
                1,
                64,
                model.cfg.ctx_train,
                None,
            )
            .vectors;
            assert_eq!(
                bits(&got[0]),
                bits(&want),
                "{pooling:?} len {len}: the engine differs from the reference"
            );
        }
    }
}

#[test]
fn a_berts_vector_is_bit_identical_alone_in_batches_and_in_any_order() {
    let dir = tmpdir("bert-invariance");
    for (fmt, seed) in [(Format::F32, 43u64), (Format::Q8_0, 44)] {
        let model = bert_model(&dir, fmt, seed, 2);
        let pooling = Pooling::Cls;
        let probe = bert_text(9, 37);
        let alone = embed_batch(
            &model,
            std::slice::from_ref(&probe),
            pooling,
            1,
            64,
            model.cfg.ctx_train,
            None,
        )
        .vectors[0]
            .clone();
        // A mean-pooled text beside it is a different reduction over the same rows.
        let _ = embed_batch(
            &model,
            std::slice::from_ref(&bert_text(3, 5)),
            Pooling::Mean,
            1,
            64,
            model.cfg.ctx_train,
            None,
        );

        for (n, threads) in [(2usize, 1usize), (7, 2), (64, 4), (7, 1), (64, 8)] {
            let mut texts = Vec::with_capacity(n);
            for i in 0..n {
                texts.push(if i == n / 2 {
                    probe.clone()
                } else {
                    bert_text(100 + i as u32, 1 + (i * 11) % 53)
                });
            }
            // Both admission orders: the probe first in one arrangement, last in another.
            if n % 2 == 0 {
                texts.swap(0, n / 2);
            }
            let run = embed_batch(
                &model,
                &texts,
                pooling,
                threads,
                64,
                model.cfg.ctx_train,
                None,
            );
            let probe_at = texts.iter().position(|t| t == &probe).unwrap();
            assert_eq!(
                bits(&run.vectors[probe_at]),
                bits(&alone),
                "{}: probe in a batch of {n}, {threads} threads, admitted at {probe_at}",
                fmt.label()
            );
        }

        // Short texts too: a one- or two-row pass can take a different matmul kernel.
        for len in [1usize, 2, 3] {
            let short = bert_text(31 + len as u32, len);
            let short_alone = embed_batch(
                &model,
                std::slice::from_ref(&short),
                pooling,
                1,
                64,
                model.cfg.ctx_train,
                None,
            )
            .vectors[0]
                .clone();
            let mut texts = vec![short.clone()];
            for i in 0..7 {
                texts.push(bert_text(200 + i, 13 + i as usize));
            }
            let run = embed_batch(&model, &texts, pooling, 4, 64, model.cfg.ctx_train, None);
            assert_eq!(
                bits(&run.vectors[0]),
                bits(&short_alone),
                "{}: a {len}-token text is the same alone and in a batch",
                fmt.label()
            );
        }

        // A text longer than the matmul's 128-row chunk runs in several weight
        // passes, and must equal itself alone and in a mixed pass.
        let long = bert_text(77, 64);
        let long_alone = embed_batch(
            &model,
            std::slice::from_ref(&long),
            pooling,
            2,
            64,
            model.cfg.ctx_train,
            None,
        )
        .vectors[0]
            .clone();
        let mixed = vec![bert_text(4, 3), bert_text(5, 5), long.clone()];
        let run = embed_batch(&model, &mixed, pooling, 4, 64, model.cfg.ctx_train, None);
        assert_eq!(
            bits(&run.vectors[2]),
            bits(&long_alone),
            "{}: the long text is the same alone and in a mixed pass",
            fmt.label()
        );
    }
}

/// The unigram (`t5`) tokenizer variant: the XLM-R models' vocabulary. The weights
/// are the same `bert` graph, so the vector is too — only the ids differ.
#[test]
fn a_bert_file_with_the_unigram_tokenizer_embeds() {
    let dir = tmpdir("bert-unigram");
    let path = support::models::write_bert(
        &dir,
        Format::F32,
        48,
        support::models::BertConfig {
            unigram: true,
            ..Default::default()
        },
    );
    let model = load(&path);
    assert_eq!(model.pooling().unwrap(), Some(Pooling::Cls));
    let text = model.tk.encode_special("hello world");
    // `<s> … </s>`: the file's BOS and EOS, both added.
    assert_eq!(text.first(), Some(&1u32), "{text:?}");
    assert_eq!(text.last(), Some(&2u32), "{text:?}");
    let want = support::bert_ref::reference_embedding(&model, &text, Pooling::Cls);
    let got = embed_batch(
        &model,
        std::slice::from_ref(&text),
        Pooling::Cls,
        2,
        64,
        model.cfg.ctx_train,
        None,
    )
    .vectors;
    assert_eq!(bits(&got[0]), bits(&want), "the unigram file's vector");
}

#[test]
fn a_bert_text_longer_than_its_position_table_is_refused_by_name() {
    let dir = tmpdir("bert-long");
    let model = bert_model(&dir, Format::F32, 49, 2);
    let ctx = model.cfg.ctx_train;
    // The engine admits one position past the file's context (a chat turn needs the
    // last one), so a `ctx + 1`-token text reaches the family, which refuses it: the
    // position table has exactly `ctx` rows.
    let mut engine = Engine::new(model, 2, ctx + 1, 1, KvPrec::F16, 0);
    let id = engine
        .admit_pooled(bert_text(1, ctx + 1), SlotCfg::default(), Pooling::Cls)
        .expect("admitted: the engine allows ctx + 1 rows");
    let mut errors = Vec::new();
    let mut steps = 0;
    while engine.active() > 0 {
        for em in engine.step() {
            if em.slot == id {
                if let Some(Finish::Error(e)) = em.finish {
                    errors.push(e);
                }
            }
        }
        steps += 1;
        assert!(steps < 100, "no progress");
    }
    assert_eq!(errors.len(), 1, "one refusal");
    assert!(
        errors[0].contains("position table") && errors[0].contains(&format!("{ctx}")),
        "{errors:?}"
    );
    // The server/CLI rule is the file's context: `ctx + 1` is refused before admission.
    let model = peregrine_core::Gguf::open(&support::models::write_bert(
        &dir,
        Format::F32,
        50,
        support::models::BertConfig::default(),
    ))
    .map(peregrine_models::load)
    .unwrap()
    .unwrap();
    let mut engine = Engine::new(Arc::new(model), 2, ctx + 1, 1, KvPrec::F16, 0);
    let err = engine
        .admit_pooled(bert_text(1, ctx + 2), SlotCfg::default(), Pooling::Cls)
        .unwrap_err();
    assert!(err.1.contains("does not fit"), "{err:?}");
}

#[test]
fn a_chat_or_label_slot_is_refused_by_a_bert_encoder() {
    let dir = tmpdir("bert-refuse");
    let model = bert_model(&dir, Format::F32, 45, 2);
    let mut engine = Engine::new(model, 1, 64, 1, KvPrec::F16, 0);
    let err = engine
        .admit(bert_text(1, 8), SlotCfg::default())
        .unwrap_err();
    assert!(
        err.1.contains("embedding model") && err.1.contains("cannot generate"),
        "{err:?}"
    );
    let err = engine
        .admit_labels(bert_text(2, 8), SlotCfg::default(), vec![1, 2])
        .unwrap_err();
    assert!(
        err.1.contains("embedding model") && err.1.contains("label logits"),
        "{err:?}"
    );
    let id = engine
        .admit_pooled(bert_text(3, 8), SlotCfg::default(), Pooling::Cls)
        .unwrap();
    engine.cancel(id);
}

/// The synthetic `bert` encoder's golden for the current kernel tier: the whole
/// vector's bits, for cls- and mean-pooled texts and one Matryoshka truncation.
#[test]
fn the_synthetic_bert_matches_its_golden_for_the_tier() {
    let dir = tmpdir("bert-golden");
    let model = bert_model(&dir, Format::Q8_0, 46, 2);
    let mut out = String::new();
    for (pooling, dims) in [(Pooling::Cls, None), (Pooling::Mean, Some(64usize))] {
        let text = bert_text(21, 37);
        let run = embed_batch(
            &model,
            std::slice::from_ref(&text),
            pooling,
            3,
            64,
            model.cfg.ctx_train,
            None,
        );
        let mut v = run.vectors[0].clone();
        embed::finish(&mut v, dims).unwrap();
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for x in &v {
            for b in x.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        use std::fmt::Write as _;
        writeln!(
            out,
            "{} {dims:?} len {} fnv {h:016x}",
            pooling.name(),
            v.len()
        )
        .unwrap();
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
    let gpath = dir.join("bert-Q8_0.golden");
    let record = std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "record");
    let require = std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "require");
    if record && tier != "x86_64-avx2-vnni256" {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&gpath, &out).unwrap();
        eprintln!("synthetic: recorded bert-Q8_0 ({tier})");
    } else if let Ok(want) = std::fs::read_to_string(&gpath) {
        assert_eq!(want, out, "{}", gpath.display());
    } else if require {
        panic!(
            "bert-Q8_0 has no golden for tier {tier} ({}); record it with PEREGRINE_GOLDEN=record",
            gpath.display()
        );
    }
}

/// The Q8_0 file against the reference on the engine's *int8* path: the same graph,
/// with the activation row quantised once per projection (`QuantAct::fill`) and the
/// weight row dotted by `dot_scalar`, the scalar twin of the int8 kernel.
///
/// On the scalar tier the engine *is* `dot_scalar`, so the vector is bit for bit the
/// reference's. On a SIMD tier the repacked eight-row tile computes the same integer
/// sums but fuses the block scale into its accumulator (`d·d_a·Σ` in one FMA), which
/// rounds differently by 1-2 ulp; there the check is the tight bound below, and the
/// exact bits are the tier golden's.
#[test]
fn the_bert_q8_0_vector_is_the_engine_int8_path() {
    let dir = tmpdir("bert-q8-reference");
    let model = bert_model(&dir, Format::Q8_0, 51, 2);
    for pooling in [Pooling::Cls, Pooling::Mean] {
        for (i, len) in [1usize, 2, 7, 31].into_iter().enumerate() {
            let text = bert_text(61 + i as u32, len);
            let want = support::bert_ref::reference_embedding(&model, &text, pooling);
            let got = embed_batch(
                &model,
                std::slice::from_ref(&text),
                pooling,
                1,
                64,
                model.cfg.ctx_train,
                None,
            )
            .vectors;
            let max_abs = got[0]
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() as f64)
                .fold(0.0f64, f64::max);
            let cos = cosine_at(&got[0], &want, got[0].len());
            eprintln!("Q8_0 {pooling:?} len {len}: max abs {max_abs:.3e}, cosine {cos:.12}");
            let scalar_tier = peregrine_core::cpu::tier_name() == "x86_64"
                || peregrine_core::cpu::tier_name() == "aarch64";
            if scalar_tier {
                assert_eq!(
                    bits(&got[0]),
                    bits(&want),
                    "{pooling:?} len {len}: the scalar tier is the reference's own kernel"
                );
            } else {
                assert!(
                    max_abs < 1e-6 && cos > 0.9999999,
                    "{pooling:?} len {len}: max abs {max_abs:.3e}, cosine {cos:.12}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// S4.6a on real files: llama.cpp's ids and vectors, and the fp32 reference
// ---------------------------------------------------------------------------

/// The directory the fixture vectors live in.
fn bert_fixtures() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bert")
}

/// A committed fixture vector file: little-endian f32, `dim` per vector.
fn read_fixture_vectors(path: &std::path::Path, dim: usize) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let n = bytes.len() / 4;
    assert_eq!(
        n % dim,
        0,
        "{}: {n} floats is not a whole number of {dim}-d vectors",
        path.display()
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<f32>>()
        .chunks(dim)
        .map(|v| v.to_vec())
        .collect()
}

fn read_fixture_ids(path: &std::path::Path) -> Vec<Vec<u32>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect()
}

fn bert_cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// MiniLM-L6-v2, bge-small-en-v1.5 and bge-m3 against llama.cpp v0.9.0 and the fp32
/// PyTorch reference (S4.6a's accuracy gate; see `docs/reference/bert.md`).
///
/// The model file is looked for in `models/`, `~/models/` and the `hf:` cache — the
/// same places the EmbeddingGemma test uses — and the case is skipped with a notice
/// when it is absent, so CI needs no files.
/// `cargo test --release -p peregrine-models --test embeddings -- --ignored` runs it.
#[test]
#[ignore = "needs the MiniLM / bge-small / bge-m3 GGUF files from tests/fixtures/bert/README.md"]
fn the_bert_real_files_match_llama_cpp_and_the_fp32_reference() {
    let dir = bert_fixtures();
    let texts: Vec<String> = std::fs::read_to_string(dir.join("texts.txt"))
        .unwrap()
        .lines()
        .map(|l| l.to_string())
        .collect();

    #[derive(Debug)]
    struct Case {
        file: &'static str,
        llama: &'static str,
        reference: Option<&'static str>,
        ids: &'static str,
        dim: usize,
        /// (min, mean) cosine to the fp32 reference this case must reach, when it has one.
        bound: Option<(f64, f64)>,
    }
    let cases = [
        Case {
            file: "all-MiniLM-L6-v2-ggml-model-f16.gguf",
            llama: "minilm_f16_llama.f32",
            reference: Some("minilm_torch.f32"),
            ids: "minilm.ids",
            dim: 384,
            bound: Some((0.9999, 0.9999)),
        },
        Case {
            file: "all-MiniLM-L6-v2-Q8_0.gguf",
            llama: "minilm_q8_0_llama.f32",
            reference: Some("minilm_torch.f32"),
            ids: "minilm.ids",
            dim: 384,
            bound: None,
        },
        Case {
            file: "bge-small-en-v1.5-f32.gguf",
            llama: "bge_small_f32_llama.f32",
            reference: Some("bge_small_torch.f32"),
            ids: "bge_small.ids",
            dim: 384,
            bound: Some((0.9999, 0.9999)),
        },
        Case {
            file: "bge-small-en-v1.5-q8_0.gguf",
            llama: "bge_small_q8_0_llama.f32",
            reference: Some("bge_small_torch.f32"),
            ids: "bge_small.ids",
            dim: 384,
            bound: None,
        },
        Case {
            file: "bge-m3-Q8_0.gguf",
            llama: "bge_m3_q8_0_llama.f32",
            reference: Some("bge_m3_torch.f32"),
            ids: "bge_m3.ids",
            dim: 1024,
            bound: None,
        },
    ];

    for case in cases {
        let mut candidates = vec![
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../models")
                .join(case.file),
            std::path::Path::new("/var/tmp/bert-models").join(case.file),
        ];
        if let Ok(h) = std::env::var("HOME") {
            candidates.push(std::path::Path::new(&h).join("models").join(case.file));
        }
        let Some(path) = candidates.into_iter().find(|p| p.is_file()) else {
            eprintln!("{} is not on this machine; skipped", case.file);
            continue;
        };
        let model = load(&path);
        let llama = read_fixture_vectors(&dir.join(case.llama), case.dim);
        let ids = read_fixture_ids(&dir.join(case.ids));
        let reference = case
            .reference
            .map(|f| read_fixture_vectors(&dir.join(f), case.dim));
        assert_eq!(llama.len(), texts.len(), "{case:?}");

        // 1. The tokenizer: identical ids to llama.cpp on every text.
        let tokens: Vec<Vec<u32>> = texts.iter().map(|t| model.tk.encode_special(t)).collect();
        for (i, (ours, theirs)) in tokens.iter().zip(&ids).enumerate() {
            assert_eq!(ours, theirs, "{}: text {i} token ids", case.file);
        }

        // 2. The vectors, in one engine, batches of 16 texts.
        let pooling = model.pooling().unwrap().unwrap();
        let mut ours: Vec<Vec<f32>> = Vec::new();
        for chunk in tokens.chunks(16) {
            let run = embed_batch(&model, chunk, pooling, 8, 1024, 1024, None);
            for mut v in run.vectors {
                embed::finish(&mut v, None).unwrap();
                ours.push(v);
            }
        }

        let to_llama: Vec<f64> = ours
            .iter()
            .zip(&llama)
            .map(|(a, b)| bert_cosine(a, b))
            .collect();
        let min_mean = |v: &[f64]| {
            (
                v.iter().copied().fold(f64::INFINITY, f64::min),
                v.iter().sum::<f64>() / v.len() as f64,
            )
        };
        let (min_l, mean_l) = min_mean(&to_llama);
        eprintln!(
            "{}: {} texts, {}-d, vs llama.cpp min {min_l:.7} mean {mean_l:.7}",
            case.file,
            ours.len(),
            case.dim
        );
        // A quantised file is allowed the distance its quantisation costs (the S4 gate's
        // measured bounds); an F32/F16 file is at the 0.9999 class.
        let is_quant = case.file.contains("Q8_0") || case.file.contains("q8_0");
        if is_quant {
            assert!(
                min_l >= 0.998 && mean_l >= 0.9995,
                "{}: vs llama.cpp min {min_l:.6} mean {mean_l:.6}",
                case.file
            );
        } else {
            assert!(
                min_l >= 0.9999,
                "{}: vs llama.cpp min {min_l:.6} mean {mean_l:.6}",
                case.file
            );
        }

        // 3. And the more important number: distance to the fp32 reference. A Q8_0 file
        // must be no further from it than llama.cpp's Q8_0 is (1e-4 slack on min and mean).
        if let Some(reference) = reference {
            let to_ref: Vec<f64> = ours
                .iter()
                .zip(&reference)
                .map(|(a, b)| bert_cosine(a, b))
                .collect();
            let llama_ref: Vec<f64> = llama
                .iter()
                .zip(&reference)
                .map(|(a, b)| bert_cosine(a, b))
                .collect();
            let (min_r, mean_r) = min_mean(&to_ref);
            let (min_rl, mean_rl) = min_mean(&llama_ref);
            eprintln!(
                "{}: vs fp32 reference min {min_r:.7} mean {mean_r:.7} \
                 (llama.cpp min {min_rl:.7} mean {mean_rl:.7})",
                case.file
            );
            if let Some((bmin, bmean)) = case.bound {
                assert!(
                    min_r >= bmin && mean_r >= bmean,
                    "{}: vs the reference min {min_r:.6} mean {mean_r:.6}, bound {bmin}/{bmean}",
                    case.file
                );
            }
            assert!(
                min_r >= min_rl - 1e-4 && mean_r >= mean_rl - 1e-4,
                "{}: vs the reference min {min_r:.6} mean {mean_r:.6}; \
                 llama.cpp min {min_rl:.6} mean {mean_rl:.6}",
                case.file
            );
        }
    }
}
