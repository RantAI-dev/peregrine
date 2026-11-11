//! One text's embedding pass through the engine, with the stage split (S4.9g part 3).
//!
//! ```text
//! cargo run --release -p peregrine --example embedstage -- <gguf> <tokens,...> [threads] [reps]
//! ```
//!
//! Every text is a real pass of the engine's embedding path: the tokens are the model's own
//! (a seed text encoded and cycled to the requested length), the model is loaded once, and
//! each `reps` pass uses a fresh engine so no cache is warm. The printed times are the stage
//! buckets `Engine::phases_ms` keeps — matmul, output head, attention, other — plus the wall
//! time of the pass and texts/s. Use it for the "before" and "after" of a kernel change on
//! the same file, interleaved, one process at a time.

use peregrine_core::embed::Pooling;
use peregrine_core::kv::KvPrec;
use peregrine_core::serve::{Engine, SlotCfg};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("gguf path");
    let lens: Vec<usize> = args
        .get(1)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![512, 2000]);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let model = peregrine_models::open(std::path::Path::new(path)).expect("open model");
    let c = &model.cfg;
    println!(
        "{}: {} layers, {} heads, {} kv heads, head_dim {}, d_model {}",
        c.arch_name, c.n_layer, c.n_head, c.n_kv_head, c.head_dim, c.d_model
    );
    let max_ctx = c.ctx_train;
    let seed = model.tk.encode_special(
        "The quick brown fox jumps over the lazy dog, and then it keeps running through the \
         quiet field while the morning light comes up over the hills and the river.",
    );
    assert!(!seed.is_empty());
    let pooling = model.pooling().expect("pooling").unwrap_or(Pooling::Mean);

    for &n in &lens {
        let ids: Vec<u32> = (0..n).map(|i| seed[i % seed.len()]).collect();
        let mut best = f64::INFINITY;
        let mut best_phases = [0.0f64; 6];
        for _ in 0..reps {
            let mut engine = Engine::new(model.clone(), threads, max_ctx, 1, KvPrec::F16, 0);
            let before = engine.phases_ms();
            let t = Instant::now();
            let id = engine
                .admit_pooled(ids.clone(), SlotCfg::default(), pooling)
                .expect("admit");
            let mut done = false;
            while !done {
                for em in engine.step() {
                    if em.slot != id {
                        continue;
                    }
                    if let Some(peregrine_core::serve::Finish::Error(e)) = em.finish {
                        panic!("{e}");
                    }
                    done = true;
                }
            }
            let secs = t.elapsed().as_secs_f64();
            let after = engine.phases_ms();
            if secs < best {
                best = secs;
                for i in 0..6 {
                    best_phases[i] = after[i] - before[i];
                }
            }
        }
        let names = [
            "matmul",
            "head",
            "attention",
            "other",
            "sampling",
            "drafting",
        ];
        let total: f64 = best_phases.iter().sum::<f64>().max(1e-9);
        print!(
            "n {:>5}  {:>8.3} s  {:>7.3} texts/s  {:>8.1} tok/s |",
            n,
            best,
            1.0 / best,
            n as f64 / best
        );
        for i in 0..4 {
            print!(" {} {:.0}%", names[i], 100.0 * best_phases[i] / total);
        }
        println!(" | stages {:?} ms", best_phases.map(|x| x.round()));
    }
}
