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
use support::models::{write_dense, write_hybrid, write_moe, Dense, MoeShape};

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

fn check_decode(model: &Model, threads: usize) -> usize {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let mut st = State::new(&c, 8, 1, 512, threads, model.rope_factors.as_deref());
    let mut kv = model.new_cache(KvPrec::F16, 512);
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
    allocations_during(|| {
        for _ in 0..3 {
            decode(&mut pos);
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
    allocations_during(|| {
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
    for (name, model) in [
        ("qwen3-Q8_0", &dense),
        ("qwen35-PQ2_0", &hybrid),
        ("qwen3moe-Q8_0", &moe),
        ("gemma3-Q8_0", &gemma3),
    ] {
        for threads in [1, 4] {
            let n = check_decode(model, threads);
            if n != 0 {
                failures.push(format!(
                    "{name}, {threads} threads: {n} allocations per decode step"
                ));
            }
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

    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "a per-token allocation remains:\n{}",
        failures.join("\n")
    );
}
