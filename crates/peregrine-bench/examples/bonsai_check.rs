// Bonsai 2 against the reference fixture (PrismML llama.cpp fork, greedy, top-5), and a decode timing.
// cargo run --release -p peregrine-bench --example bonsai_check -- <model.gguf> [threads] [decode_tokens]
use peregrine_core::gguf::Gguf;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, State, StepStats};
use peregrine_core::pool::Pool;
use std::time::Instant;

fn softmax_top(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = logits.iter().map(|&x| ((x - m) as f64).exp()).sum();
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|a, b| {
        logits[*b as usize]
            .partial_cmp(&logits[*a as usize])
            .unwrap()
    });
    idx[..k]
        .iter()
        .map(|&i| (i, (((logits[i as usize] - m) as f64).exp() / z) as f32))
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let path = a.first().expect("model path");
    let threads: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(12);
    let ndec: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(32);
    let t = Instant::now();
    let model = Model::load(Gguf::open(std::path::Path::new(path)).unwrap()).unwrap();
    eprintln!(
        "loaded in {:.1}s, simd {}",
        t.elapsed().as_secs_f64(),
        peregrine_core::simd_name()
    );
    let c = model.cfg;
    let pool = Pool::new(threads);
    let mut st = State::new(&c, 64, 1, 4096, threads, None);
    let mut kv = model.new_cache(KvPrec::F16, 4096);
    let prompt = model.tk.encode("The capital of France is");
    println!(
        "prompt tokens {:?} (reference [760, 6511, 314, 9338, 369])",
        prompt
    );
    let mut ss = StepStats::default();
    let t = Instant::now();
    model
        .forward(
            &mut st,
            &mut kv,
            &pool,
            &prompt,
            0,
            &[prompt.len() - 1],
            &mut ss,
        )
        .unwrap();
    eprintln!(
        "prefill {} tokens {:.3}s",
        prompt.len(),
        t.elapsed().as_secs_f64()
    );
    let mut pos = prompt.len();
    let mut out = Vec::new();
    for step in 0..16 {
        let top = softmax_top(st.logits_row(0, c.n_vocab), 5);
        if step < 3 || step == 8 {
            let s: Vec<String> = top
                .iter()
                .map(|(i, p)| format!("{:?} {:.4}", model.tk.piece(*i), p))
                .collect();
            println!("step {step}: {}", s.join(" | "));
        }
        let next = top[0].0;
        out.push(next);
        model
            .forward(&mut st, &mut kv, &pool, &[next], pos, &[0], &mut ss)
            .unwrap();
        pos += 1;
    }
    println!("greedy {:?}", out);
    println!("text {:?}", model.tk.decode(&out));
    // Decode timing.
    let mut ss = StepStats::default();
    let _ = peregrine_core::model::hybrid::profile();
    let t = Instant::now();
    let mut next = out[out.len() - 1];
    for _ in 0..ndec {
        model
            .forward(&mut st, &mut kv, &pool, &[next], pos, &[0], &mut ss)
            .unwrap();
        let l = st.logits_row(0, c.n_vocab);
        next = (0..l.len())
            .max_by(|&x, &y| l[x].partial_cmp(&l[y]).unwrap())
            .unwrap() as u32;
        pos += 1;
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "decode {ndec} tokens: {:.2} tok/s, {:.2} GB/token weights, {:.1} GB/s, matmul {:.0}% head {:.0}% attn {:.0}% other {:.0}%",
        ndec as f64 / dt,
        ss.bytes_weights as f64 / ndec as f64 / 1e9,
        ss.bytes_total() as f64 / dt / 1e9,
        100.0 * ss.ns_matmul as f64 / ss.ns_total as f64,
        100.0 * ss.ns_head as f64 / ss.ns_total as f64,
        100.0 * ss.ns_attn as f64 / ss.ns_total as f64,
        100.0 * ss.ns_other() as f64 / ss.ns_total as f64
    );
    let tot = ss.ns_total as f64;
    let pr = peregrine_core::model::hybrid::profile();
    let p: Vec<String> = pr[..10]
        .iter()
        .map(|(n, v)| format!("{n} {:.1}%", 100.0 * *v as f64 / tot))
        .collect();
    println!("profile: {}", p.join(", "));
    for k in [10usize, 12, 14] {
        let (ns, by) = (pr[k].1 as f64, pr[k + 1].1 as f64);
        if ns > 0.0 {
            println!(
                "  {}: {:.1} GB/s, {:.1}% of time",
                pr[k].0.trim_end_matches(" ns"),
                by / ns,
                100.0 * ns / tot
            );
        }
    }
}
