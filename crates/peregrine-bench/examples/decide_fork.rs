//! What forking a decision request's questions is worth on this machine (S4.9j).
//!
//! A decision request asks several questions about one `state`. Every question's prompt begins with
//! the same tokens and differs only in its question text, so the engine can forward that beginning
//! once and fork the cache into one slot per question, each continuing with its own tail. This
//! measures the group admission against admitting the questions one by one, on the same model and
//! machine, alternating the arms: rows forwarded and wall time.
//!
//!     cargo run --release -p peregrine-bench --example decide_fork -- <model.gguf> [questions] [threads] [chunk]
//!
//! `questions` is a comma list (default `1,3,10`). The shared prefix is 300 tokens and each tail 30
//! (env `STATE_TOKENS` / `TAIL_TOKENS`).
//!
//! The model must be a decision model? No: any model runs the label read-out. A decision model's
//! own prompts are the real thing, but this example builds token ids so it does not need a
//! tokenizer, and it never prints a prompt.

use peregrine_core::kv::KvPrec;
use peregrine_core::serve::{Engine, Finish, SlotCfg};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// The prompts: a shared beginning of `state` tokens and a tail of `tail` per question, each tail
/// starting with a token of its own so the longest common beginning is exactly `state`.
fn questions(
    model: &peregrine_core::model::Model,
    state: usize,
    tail: usize,
    n: usize,
) -> Vec<Vec<u32>> {
    let vocab = model.cfg.n_vocab as u32;
    let mut next = |i: usize| 1 + ((i as u32 * 37 + 11) % (vocab - 2).max(1));
    let base: Vec<u32> = (0..state).map(&mut next).collect();
    (0..n)
        .map(|q| {
            let mut p = base.clone();
            for t in 0..tail {
                // The first tail token differs per question.
                p.push(1 + ((q as u32 * 101 + t as u32 * 37 + 7) % (vocab - 2).max(1)));
            }
            p
        })
        .collect()
}

struct Measured {
    rows: usize,
    secs: f64,
    prefilled: Vec<usize>,
}

/// Run `prompts` through one engine: `group` admits them in one group (forked), otherwise one by
/// one. Counts the batch rows every step forwarded and the wall time.
fn run(
    model: &Arc<peregrine_core::model::Model>,
    prompts: &[Vec<u32>],
    threads: usize,
    chunk: usize,
    group: bool,
) -> Measured {
    let labels = vec![vec![11u32, 12, 13]; prompts.len()];
    // One position more than the longest prompt (a labels slot forwards its last row), clamped to
    // what the model was trained for. A cache per slot at the model's whole context would be far
    // more memory than the measurement needs.
    let ctx = (prompts.iter().map(Vec::len).max().unwrap_or(0) + 1).clamp(32, model.cfg.ctx_train);
    let mut engine = Engine::new_with_chunk(
        model.clone(),
        threads,
        ctx,
        prompts.len(),
        KvPrec::F16,
        0,
        chunk,
    );
    let mut by_slot: HashMap<u64, usize> = HashMap::new();
    let t0 = Instant::now();
    if group {
        let ids = engine
            .admit_labels_group(prompts.to_vec(), SlotCfg::default(), labels)
            .expect("admit the group");
        for (i, id) in ids.into_iter().enumerate() {
            by_slot.insert(id, i);
        }
    } else {
        for (i, p) in prompts.iter().enumerate() {
            let id = engine
                .admit_labels(p.clone(), SlotCfg::default(), labels[i].clone())
                .expect("admit a labels slot");
            by_slot.insert(id, i);
        }
    }
    let mut rows = 0usize;
    let mut prefilled = vec![0usize; prompts.len()];
    while engine.active() > 0 {
        for em in engine.step() {
            rows += engine.last_step.rows;
            let Some(&i) = by_slot.get(&em.slot) else {
                continue;
            };
            if let Some(Finish::Error(e)) = &em.finish {
                panic!("a labels slot failed: {e}");
            }
            if em.finish.is_some() {
                prefilled[i] = em.cost.as_ref().map_or(0, |c| c.prefilled);
            }
        }
    }
    Measured {
        rows,
        secs: t0.elapsed().as_secs_f64(),
        prefilled,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(
        args.next()
            .expect("usage: decide_fork <model.gguf> [questions] [threads] [chunk]"),
    );
    let counts: Vec<usize> = args
        .next()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 3, 10]);
    let threads: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(4);
    let chunk: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(16);
    let state: usize = std::env::var("STATE_TOKENS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(300);
    let tail: usize = std::env::var("TAIL_TOKENS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(30);
    let model = Arc::new(peregrine_models::open(&path).expect("open the model"));
    println!(
        "{}: {} threads, prefill chunk {chunk}, {state}-token shared prefix + {tail}-token tails\n",
        path.file_name().unwrap().to_string_lossy(),
        threads
    );
    println!(
        "{:>9} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "questions", "rows old", "rows fork", "ms old", "ms fork", "speedup"
    );
    let prompts = questions(&model, state, tail, *counts.iter().max().unwrap_or(&1));
    for &n in &counts {
        let qs = &prompts[..n];
        // Two rounds, alternating which arm runs first (the cache is cold each time).
        let (mut old, mut new) = (0.0f64, f64::MAX);
        let (mut rows_old, mut rows_new) = (0usize, 0usize);
        for round in 0..2 {
            for arm in if round == 0 { [0, 1] } else { [1, 0] } {
                let m = run(&model, qs, threads, chunk, arm == 1);
                if arm == 0 {
                    old = old.max(m.secs);
                    rows_old = m.rows;
                } else {
                    new = new.min(m.secs);
                    rows_new = m.rows;
                }
                println!(
                    "  {:>7}: {} {:>7.3} s, {} rows forwarded, prefilled {:?}",
                    if arm == 1 { "fork" } else { "single" },
                    n,
                    m.secs,
                    m.rows,
                    m.prefilled
                );
            }
        }
        println!(
            "{:>9} {:>10} {:>10} {:>12.1} {:>12.1} {:>8.2}x",
            n,
            rows_old,
            rows_new,
            old * 1e3,
            new * 1e3,
            old / new.max(1e-9)
        );
    }
}
