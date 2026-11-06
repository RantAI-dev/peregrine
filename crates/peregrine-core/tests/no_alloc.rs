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

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, State, StepStats};
use peregrine_core::pool::Pool;
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use support::models::{write_dense, write_hybrid, Dense};

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

fn load(path: &Path) -> Model {
    let g = Gguf::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Model::load(g).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
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
    for (name, model) in [("qwen3-Q8_0", &dense), ("qwen35-PQ2_0", &hybrid)] {
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
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "a per-token allocation remains:\n{}",
        failures.join("\n")
    );
}
