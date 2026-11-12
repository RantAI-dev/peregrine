//! Invariant 4: a decode step allocates nothing (roadmap S1.2).
//!
//! A counting global allocator wraps `System` and, while a flag is set, counts every allocation in the whole
//! process (so the pool's worker threads are included, not only the calling thread). Synthetic models are warmed
//! up first — that is where a session legitimately allocates — then further single-token decode steps, and one
//! MTP draft step, must count zero.
//!
//! Everything that allocates per token used to be visible here: `Caches::runs` built a `Vec` once per attention
//! layer, `Model::forward` collected a position vector per call, the hybrid pass built four pointer vectors and a
//! `seen` vector per Gated DeltaNet layer, and the MTP block allocated `e`, `qg`, `gate`, two `to_vec` rows and
//! `y` per draft.
//!
//! One MTP *verification* step is covered too: the drafter proposes into a buffer the caller owns and the
//! engine commits, the way `Engine::verify_step` asks of it. The proposal is the engine's own buffer and the
//! drafter's scratch is reused, so neither allocates.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, State, StepStats};
use peregrine_core::pool::Pool;
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use support::models::{
    write_bert, write_dense, write_gemma_embedding, write_hybrid, write_mistral3,
    write_modern_bert, write_moe, write_nomic_bert, BertConfig, BertHeadConfig, Dense,
    GemmaEmbeddingConfig, Mistral3Config, ModernBertConfig, MoeShape, NomicBertConfig,
};

/// Allocations counted while `COUNTING` was set.
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations made in the whole process while `f` runs.
fn allocations_during(f: impl FnOnce()) -> usize {
    ALLOCS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    f();
    COUNTING.store(false, Ordering::SeqCst);
    ALLOCS.load(Ordering::SeqCst)
}

/// The fewest allocations in any of three windows of `f`, run back to back.
///
/// A per-token allocation shows in every window. A one-off does not: on macOS, std's
/// `Mutex` and `Condvar` allocate their pthread object on first use, so a pool worker
/// that first sleeps (past its spin) inside a window counts once there and never again.
/// That made this test fail now and then on the macOS runner.
fn allocations_steady(mut f: impl FnMut()) -> usize {
    (0..3).map(|_| allocations_during(&mut f)).min().unwrap()
}

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-noalloc-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &Path) -> Arc<Model> {
    let g = Gguf::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Arc::new(peregrine_models::load(g).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

/// No model file here carries a tokenizer vocabulary large enough for the prompt used by the synthetic tests;
/// any id inside the vocabulary works for an allocation test.
const TOKEN: u32 = 11;

fn check_decode(model: &Model, threads: usize, prec: KvPrec) -> usize {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let mut st = State::new(&c, 8, 1, 512, threads, model.rope_factors.as_deref());
    let mut kv = model.new_cache(prec, 512);
    let mut stats = StepStats::default();
    let mut pos = 0usize;
    let mut decode = |pos: &mut usize| {
        model
            .forward(&mut st, &mut kv, &pool, &[TOKEN], *pos, &[0], &mut stats)
            .unwrap();
        *pos += 1;
    };
    for _ in 0..4 {
        decode(&mut pos);
    }
    allocations_steady(|| {
        for _ in 0..3 {
            decode(&mut pos);
        }
    })
}

/// A bidirectional pass over whole texts (roadmap S4.4): the projections, the per-text
/// attention, the pooled dense head and (S4.6b) the classifier head allocate nothing once
/// the state's scratch is warm.
fn check_forward_texts(model: &Model, threads: usize) -> usize {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let mut st = model.new_state(c.ctx_train.max(8), 1, c.ctx_train.max(8), threads);
    st.hidden.resize(st.max_batch * c.d_model, 0.0);
    st.pool_sum = vec![0.0; c.d_model];
    let a: Vec<u32> = (11..48).collect();
    let b: Vec<u32> = (20..35).collect();
    let c3: Vec<u32> = (13..22).collect();
    let texts: [&[u32]; 3] = [&a, &b, &c3];
    let mut wants =
        [peregrine_core::arch::TextWant::Pooled(peregrine_core::embed::Pooling::Mean); 3];
    // A model with a classifier head reads one text out as the head's row; the head's
    // projections then run in the same pass (S4.6b). Without a head, every text pools.
    if model.has_head() {
        wants[1] = peregrine_core::arch::TextWant::Head;
    }
    let mut stats = StepStats::default();
    let mut run = |st: &mut State| {
        model
            .forward_texts(st, &pool, &texts, &wants, &mut stats)
            .unwrap();
    };
    for _ in 0..4 {
        run(&mut st);
    }
    allocations_steady(|| {
        for _ in 0..3 {
            run(&mut st);
        }
    })
}

fn check_mtp_draft(model: &Model, threads: usize) -> usize {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let mut ms = model.mtp_state(512, threads).unwrap();
    let mut stats = StepStats::default();
    let hidden = vec![0.0f32; c.d_model];
    let draft = |ms: &mut peregrine_core::model::mtp::MtpState, stats: &mut StepStats| {
        let pos = ms.kv.filled();
        model
            .mtp_forward(ms, &pool, &[TOKEN], &hidden, &[pos], true, stats)
            .unwrap()
            .expect("a draft");
    };
    for _ in 0..4 {
        draft(&mut ms, &mut stats);
    }
    allocations_steady(|| {
        for _ in 0..3 {
            draft(&mut ms, &mut stats);
        }
    })
}

/// One MTP verification step: the drafter proposes into the caller's buffer
/// and the engine commits what was sampled, the way `Engine::verify_step`
/// asks of it.
///
/// The proposal is written into a buffer the caller owns and reused, and the
/// drafter keeps its own scratch: neither allocates per step.
fn check_mtp_verify(model: &Arc<Model>, threads: usize) -> usize {
    use peregrine_core::draft::{Drafter, MtpDrafter, Proposal};
    let c = model.cfg;
    let pool = Pool::new(threads);
    let mut drafter = MtpDrafter::new(model.clone(), 512, threads).unwrap();
    let mut out = Proposal::default();
    let hidden = vec![0.0f32; c.d_model];
    // The sampler's stand-in: the MTP block does not draw with it.
    let mut pick = |_i: usize, _logits: &[f32]| 0u32;
    let mut step = |drafter: &mut MtpDrafter, out: &mut Proposal, pos: usize| {
        // The target's committed row: the token and the hidden state of the
        // position before it.
        drafter.commit(pos, TOKEN, &[], &hidden, &pool);
        drafter.propose(TOKEN, pos + 1, 1, &pool, &mut pick, out);
    };
    for pos in 0..8 {
        step(&mut drafter, &mut out, pos);
    }
    allocations_during(|| {
        for pos in 8..12 {
            step(&mut drafter, &mut out, pos);
        }
    })
}

/// One suffix verification step: the drafter proposes into the caller's buffer
/// and is told what the target committed, the way `Engine::verify_step` asks of
/// it.
///
/// The proposal is written into a buffer the caller owns and reused, and the
/// drafter keeps its own scratch: neither allocates per step. The corpus is a
/// run of one token, so the index's tables hold one key each and stop growing;
/// the commits in the measured window are past the corpus, so it does not grow
/// either.
fn check_suffix_verify() -> usize {
    use peregrine_core::draft::{Drafter, Proposal, SuffixDrafter};
    let pool = Pool::new(4);
    let mut drafter = SuffixDrafter::new(2, 6);
    let mut out = Proposal::default();
    // The sampler's stand-in: the suffix index does not draw with it.
    let mut pick = |_i: usize, _logits: &[f32]| 0u32;
    // The corpus: the same token over and over, so the index finds a match and
    // its tables stop growing.
    for pos in 0..64 {
        drafter.commit(pos, TOKEN, &[], &[], &pool);
    }
    // A warm proposal: the drafter's own buffers and the caller's have their
    // capacity.
    drafter.propose(TOKEN, 64, 6, &pool, &mut pick, &mut out);
    let base = 64usize + 8;
    allocations_during(|| {
        for i in 0..4 {
            drafter.propose(TOKEN, base + i, 6, &pool, &mut pick, &mut out);
            // Past the corpus: the corpus keeps its length and its capacity, and
            // the index indexes nothing new.
            drafter.commit(base + i + 1, TOKEN, &[], &[], &pool);
        }
    })
}

#[test]
fn a_decode_step_allocates_nothing() {
    // Prove the counter sees allocations, so a silently broken allocator cannot make this test vacuous.
    let probe = allocations_during(|| {
        std::hint::black_box(vec![0u8; 16]);
    });
    assert!(
        probe > 0,
        "the counting allocator did not observe an allocation"
    );

    let dir = tmpdir("decode");
    let mut failures = Vec::new();

    // A dense qwen3 in Q8_0: every matmul, the norms, RoPE, attention and the head, on one and four threads.
    let dense = load(&write_dense(
        &dir,
        "qwen3",
        Format::Q8_0,
        7000,
        Dense::default(),
    ));
    // The hybrid model: Gated DeltaNet layers (conv, delta rule, snapshots), gated attention, Hadamard rotation.
    let hybrid = load(&write_hybrid(&dir, Format::PQ2_0, 7001, false));
    // The mixture of experts: the router, the expert gather and the slot-ordered sum, all in
    // the scratch the family sized once.
    let moe = load(&write_moe(&dir, Format::Q8_0, 7003, MoeShape::default()));
    // Gemma 3 in Q8_0: sliding-window and global layers, per-head norms, GeGLU, 4 norms per layer.
    let gemma3 = load(&support::models::write_gemma3(
        &dir,
        Format::Q8_0,
        7004,
        support::models::Gemma3Config::default(),
    ));
    // Phi-3 in Q8_0: fused qkv, fused ffn_up, RMSNorm, NeoX rotary pairing.
    let phi3 = load(&support::models::write_phi3(
        &dir,
        Format::Q8_0,
        7005,
        Dense::default(),
    ));
    // gpt-oss in Q8_0: attention sinks, the sliding window, YaRN, the experts' biases.
    let gpt_oss = load(&support::models::write_gpt_oss(
        &dir,
        Format::Q8_0,
        7006,
        support::models::GptOssShape::default(),
    ));
    for (name, model) in [
        ("qwen3-Q8_0", &dense),
        ("qwen35-PQ2_0", &hybrid),
        ("qwen3moe-Q8_0", &moe),
        ("gemma3-Q8_0", &gemma3),
        ("phi3-Q8_0", &phi3),
        ("gpt-oss-Q8_0", &gpt_oss),
    ] {
        for threads in [1, 4] {
            // The f16 cache and the f32 one (the opt-in full-precision cache) must
            // both decode without allocating.
            for prec in [KvPrec::F16, KvPrec::F32] {
                let n = check_decode(model, threads, prec);
                if n != 0 {
                    failures.push(format!(
                        "{name}, {threads} threads, {} cache: {n} allocations per decode step",
                        prec.name()
                    ));
                }
            }
        }
    }

    // A bidirectional encoder: the whole-text pass and its pooled dense head.
    let encoder = load(&write_gemma_embedding(
        &dir,
        Format::Q8_0,
        7007,
        GemmaEmbeddingConfig::default(),
    ));
    for threads in [1, 4] {
        let n = check_forward_texts(&encoder, threads);
        if n != 0 {
            failures.push(format!(
                "gemma-embedding whole-text pass, {threads} threads: {n} allocations per pass"
            ));
        }
    }

    // The second encoder: mistral3's wider queries, YaRN table, attention temperature
    // and mean pooling, over the same shared whole-text pass.
    let mistral = load(&write_mistral3(
        &dir,
        Format::Q8_0,
        7008,
        Mistral3Config::default(),
    ));
    for threads in [1, 4] {
        let n = check_forward_texts(&mistral, threads);
        if n != 0 {
            failures.push(format!(
                "mistral3 whole-text pass, {threads} threads: {n} allocations per pass"
            ));
        }
    }

    // The BERT family (S4.6a): the same whole-text pass, with the embedding LayerNorm,
    // the GELU rows and the cls pooling its graph adds.
    let bert = load(&write_bert(&dir, Format::Q8_0, 7008, BertConfig::default()));
    for threads in [1, 4] {
        let n = check_forward_texts(&bert, threads);
        if n != 0 {
            failures.push(format!(
                "bert whole-text pass, {threads} threads: {n} allocations per pass"
            ));
        }
    }

    // A BERT file with a classifier head (S4.6b): the head's two projections run in the
    // pass, and they must allocate nothing either.
    let bert_head = load(&write_bert(
        &dir,
        Format::Q8_0,
        7009,
        BertConfig {
            pooling_type: 4,
            head: Some(BertHeadConfig {
                n_out: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
    ));
    for threads in [1, 4] {
        let n = check_forward_texts(&bert_head, threads);
        if n != 0 {
            failures.push(format!(
                "bert classifier head, {threads} threads: {n} allocations per pass"
            ));
        }
    }

    // ModernBERT (S4.7): the same whole-text pass with the fused q/k/v scratch,
    // the two LayerNorm-without-bias calls and the gated feed-forward.
    let modern = load(&write_modern_bert(
        &dir,
        Format::Q8_0,
        7009,
        ModernBertConfig::default(),
    ));
    for threads in [1, 4] {
        let n = check_forward_texts(&modern, threads);
        if n != 0 {
            failures.push(format!(
                "modern-bert whole-text pass, {threads} threads: {n} allocations per pass"
            ));
        }
    }

    // NomicBERT (S4.8a): the same whole-text pass with a fused q/k/v, NeoX RoPE,
    // the SwiGLU pair and LayerNorms with a bias, and no final norm.
    let nomic = load(&write_nomic_bert(
        &dir,
        Format::Q8_0,
        7010,
        NomicBertConfig::default(),
    ));
    for threads in [1, 4] {
        let n = check_forward_texts(&nomic, threads);
        if n != 0 {
            failures.push(format!(
                "nomic-bert whole-text pass, {threads} threads: {n} allocations per pass"
            ));
        }
    }

    // The same hybrid file with one NextN block: a draft step through the MTP head.
    let mtp = load(&write_hybrid(&dir, Format::PQ2_0, 7002, true));
    if !mtp.has_mtp() {
        failures.push("qwen35-MTP loaded without an MTP block".into());
    } else {
        for threads in [1, 4] {
            let n = check_mtp_draft(&mtp, threads);
            if n != 0 {
                failures.push(format!(
                    "qwen35-MTP draft, {threads} threads: {n} allocations per step"
                ));
            }
            let n = check_mtp_verify(&mtp, threads);
            if n != 0 {
                failures.push(format!(
                    "qwen35-MTP verification, {threads} threads: {n} allocations per step"
                ));
            }
        }
    }

    // The suffix drafter: the proposal, the index and the corpus, all in
    // buffers the drafter and the caller own.
    let n = check_suffix_verify();
    if n != 0 {
        failures.push(format!("suffix verification: {n} allocations per step"));
    }

    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "a per-token allocation remains:\n{}",
        failures.join("\n")
    );
}
