//! Attention-only micro-benchmark: the cost of one token's attention at a context.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example attnbench -- <gguf> <ctx> [threads] [reps]
//! ```
//!
//! Fills a cache with `ctx` positions, then times `attend` over every layer (the
//! per-token attention cost) and prints ms/token and GFLOP/s.
use peregrine_core::blocks::attn::attend;
use peregrine_core::gguf::Gguf;
use peregrine_core::model::Model;
use peregrine_core::pool::Pool;
use peregrine_core::state::StepStats;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("gguf path");
    let ctx: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let g = Gguf::open(std::path::Path::new(path)).expect("open");
    let model = Model::load_with(g, peregrine_models::FAMILIES).expect("load");
    let cfg = model.cfg;
    let pool = Pool::new(threads);
    let mut st = model.new_state(1, 1, ctx, threads);
    let kv = model.new_cache(peregrine_core::kv::KvPrec::F16, ctx);
    let mut kv = if model.session_kv_rotation() {
        kv
    } else {
        kv.without_rotation()
    };
    // Fill: every position of every KV head with a small varied value.
    let hd = cfg.head_dim;
    let kvd = cfg.kv_dim();
    for l in 0..cfg.n_layer {
        for h in 0..cfg.n_kv_head {
            for p in 0..ctx {
                let k: Vec<f32> = (0..hd).map(|i| ((p * 7 + i) % 17) as f32 * 0.01).collect();
                let v: Vec<f32> = (0..hd).map(|i| ((p * 3 + i) % 13) as f32 * 0.01).collect();
                kv.put(l, h, p, &k, &v);
            }
        }
    }
    kv.advance(ctx);
    let qd = cfg.q_dim();
    let q: Vec<f32> = (0..qd).map(|i| (i % 9) as f32 * 0.01).collect();
    let mut xb = vec![0.0f32; qd];
    let mut stats = StepStats::default();
    let mut best = f64::INFINITY;
    let mut out_chk = 0.0f32;
    for _ in 0..reps {
        let t = Instant::now();
        for l in 0..cfg.n_layer {
            let _ = kvd;
            attend(
                &cfg,
                &mut st.attn,
                &q,
                &mut xb,
                &kv,
                &pool,
                l,
                1,
                ctx,
                0,
                &mut stats,
            );
            out_chk += xb[0];
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    let flops = cfg.n_layer as f64 * cfg.n_head as f64 * ctx as f64 * hd as f64 * 4.0;
    println!(
        "ctx {ctx} threads {threads}: {:.3} ms/token  {:.1} GFLOP/s  (chk {out_chk:.3})",
        best * 1e3,
        flops / best / 1e9
    );
    println!(
        "  bytes_kv {:.1} MB  barriers {}",
        stats.bytes_kv as f64 / 1e6,
        stats.barriers / reps as u64
    );
}
