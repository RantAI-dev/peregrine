//! Attention-only micro-benchmark: one token's attention at a context, or a prompt's.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example attnbench -- <gguf> <ctx> [threads] [reps] [chunk]
//! ```
//!
//! Fills a cache with `ctx` positions, then times `attend` over every layer and prints
//! ms and GFLOP/s. With `chunk` 0 or absent it times one decode token at `ctx − 1`;
//! with a `chunk` of 2 or more it times the prompt as the engine reads it, in pieces
//! of `chunk` rows (the default chunk is 16), covering every position up to `ctx`.
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
    let chunk: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let g = Gguf::open(std::path::Path::new(path)).expect("open");
    let model = Model::load_with(g, peregrine_models::FAMILIES).expect("load");
    let cfg = model.cfg;
    let pool = Pool::new(threads);
    // The prompt path reads `chunk` rows at a time; a decode token is one row.
    let batch = chunk.max(1);
    let mut st = model.new_state(batch, 1, ctx, threads);
    let kv = model.new_cache(peregrine_core::kv::KvPrec::F16, ctx);
    let mut kv = if model.session_kv_rotation() {
        kv
    } else {
        kv.without_rotation()
    };
    // Fill: every position of every KV head with a small varied value.
    let hd = cfg.head_dim;
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
    let q: Vec<f32> = (0..batch * qd)
        .map(|i| (i % 9) as f32 * 0.01 - 0.04)
        .collect();
    let mut xb = vec![0.0f32; batch * qd];
    let mut stats = StepStats::default();
    let flags = vec![true; batch];
    // Which order this run times: the tier's verdict and the head width the 16-lane
    // bodies cover (S5.4n). `PEREGRINE_FAST_ATTN=1` marks every row a prompt row.
    println!(
        "tier {} fast_attn {} head_dim {} fast_rows {}",
        peregrine_kernels::cpu::cap(),
        peregrine_kernels::table::MatKernels::get().fast_attn(),
        hd,
        std::env::var("PEREGRINE_FAST_ATTN").as_deref() == Ok("1"),
    );
    let mut best = f64::INFINITY;
    let mut out_chk = 0.0f32;
    for _ in 0..reps {
        let t = Instant::now();
        if chunk < 2 {
            for l in 0..cfg.n_layer {
                attend(
                    &cfg,
                    &mut st.attn,
                    &q,
                    &mut xb,
                    &kv,
                    &pool,
                    l,
                    1,
                    ctx - 1,
                    0,
                    &mut stats,
                );
                out_chk += xb[0];
            }
        } else {
            let mut start = 0;
            while start < ctx {
                let nb = chunk.min(ctx - start);
                // `PEREGRINE_FAST_ATTN=1` marks every row a prompt row of the fast
                // prompt mode (S5.4j), so the fast attention order is what is timed.
                let on = std::env::var("PEREGRINE_FAST_ATTN").as_deref() == Ok("1");
                let fast: Option<&[bool]> = on.then_some(&flags[..nb]);
                for l in 0..cfg.n_layer {
                    peregrine_core::blocks::attn::attend_marked(
                        &cfg,
                        &mut st.attn,
                        &q,
                        &mut xb,
                        &kv,
                        &pool,
                        l,
                        nb,
                        start,
                        0,
                        fast,
                        &mut stats,
                    );
                    out_chk += xb[0];
                }
                start += nb;
            }
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    let positions = if chunk < 2 {
        ctx as f64
    } else {
        (ctx * (ctx + 1) / 2) as f64
    };
    let flops = cfg.n_layer as f64 * cfg.n_head as f64 * positions * hd as f64 * 4.0;
    if chunk < 2 {
        println!(
            "decode ctx {ctx} threads {threads}: {:.3} ms/token  {:.1} GFLOP/s  (chk {out_chk:.3})",
            best * 1e3,
            flops / best / 1e9
        );
    } else {
        println!(
            "prefill ctx {ctx} chunk {chunk} threads {threads}: {:.1} ms  {:.1} tok/s  {:.1} GFLOP/s  (chk {out_chk:.3})",
            best * 1e3,
            ctx as f64 / best,
            flops / best / 1e9
        );
    }
    println!(
        "  bytes_kv {:.1} MB  barriers {}",
        stats.bytes_kv as f64 / 1e6,
        stats.barriers
            / (reps as u64
                * if chunk < 2 {
                    1
                } else {
                    (ctx.div_ceil(chunk)) as u64
                })
    );
    if peregrine_core::prof::enabled() {
        let snap = peregrine_core::prof::snapshot();
        for (name, ns) in peregrine_core::prof::names().iter().zip(snap) {
            if ns > 0 {
                println!("  prof {name:16} {:8.1} ms", ns as f64 / 1e6);
            }
        }
    }
}
