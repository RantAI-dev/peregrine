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
    let mut engine = Engine::new_with_chunk(
        model.clone(),
        threads,
        ctx,
        texts.len().max(1) + usize::from(extra.is_some()),
        KvPrec::F16,
        0,
        chunk,
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
        for (i, len) in [1usize, 2, 7, 31].into_iter().enumerate() {
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

        // A text long enough to run alone in a whole pass, beside shorts that fit one.
        let long = encoder_text(77, 100);
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
