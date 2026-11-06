use peregrine_core::gguf::Gguf;
use peregrine_core::kv::{KvCache, KvPrec};
use peregrine_core::model::{Caches, State, StepStats};
use peregrine_core::pool::Pool;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let model = peregrine_models::load(Gguf::open(std::path::Path::new(&a[0])).unwrap()).unwrap();
    let threads: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(12);
    let np: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(512);
    let users: Vec<usize> = a
        .get(3)
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![1, 2, 4, 8]);
    let c = model.cfg;
    let pool = Pool::new(threads);
    let chunk = 64;
    let mut st = State::new(&c, chunk.max(16), 16, 8192, threads, None);
    // Prefill: `np` tokens of real text, in chunks of 64.
    let text = "The history of the printing press begins in the fifteenth century, when movable metal type, oil-based ink and a screw press were combined. ".repeat(64);
    let toks: Vec<u32> = model.tk.encode(&text).into_iter().take(np).collect();
    let mut kv = model.new_cache(KvPrec::F16, 8192);
    let mut ss = StepStats::default();
    let _ = peregrine_models::hybrid::profile();
    let t = Instant::now();
    let mut p = 0;
    while p < toks.len() {
        let e = (p + chunk).min(toks.len());
        let want = if e == toks.len() {
            vec![e - p - 1]
        } else {
            vec![]
        };
        model
            .forward(&mut st, &mut kv, &pool, &toks[p..e], p, &want, &mut ss)
            .unwrap();
        p = e;
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "prefill {} tokens: {:.1} tok/s ({:.2}s)",
        toks.len(),
        toks.len() as f64 / dt,
        dt
    );
    let pr = peregrine_models::hybrid::profile();
    let tot = dt * 1e9;
    let pp: Vec<String> = pr[..10]
        .iter()
        .map(|(n, v)| format!("{n} {:.1}%", 100.0 * *v as f64 / tot))
        .collect();
    println!("  profile: {}", pp.join(", "));
    // Several users: each with a short prompt, then 32 decode steps together.
    for &nu in &users {
        let mut caches: Vec<KvCache> = (0..nu)
            .map(|_| model.new_cache(KvPrec::F16, 1024))
            .collect();
        let prompt = model
            .tk
            .encode("Write a short story about a lighthouse keeper.");
        for kv in caches.iter_mut() {
            model
                .forward(&mut st, kv, &pool, &prompt, 0, &[], &mut ss)
                .unwrap();
        }
        let mut pos = vec![prompt.len(); nu];
        let mut next = vec![prompt[prompt.len() - 1]; nu];
        let want: Vec<usize> = (0..nu).collect();
        let steps = 32;
        let _ = peregrine_models::hybrid::profile();
        let t = Instant::now();
        for _ in 0..steps {
            let mut refs: Vec<&mut KvCache> = caches.iter_mut().collect();
            let mut cs = Caches::PerRow(&mut refs);
            model
                .forward_rows(&mut st, &mut cs, &pool, &next, &pos, &want, &mut ss)
                .unwrap();
            for u in 0..nu {
                let l = st.logits_row(u, c.n_vocab);
                next[u] = (0..l.len())
                    .max_by(|&x, &y| l[x].partial_cmp(&l[y]).unwrap())
                    .unwrap() as u32;
                pos[u] += 1;
            }
        }
        let dt = t.elapsed().as_secs_f64();
        let pr = peregrine_models::hybrid::profile();
        let pp: Vec<String> = pr[..10]
            .iter()
            .map(|(n, v)| format!("{n} {:.0}%", 100.0 * *v as f64 / (dt * 1e9)))
            .collect();
        println!("  profile: {}", pp.join(", "));
        println!(
            "{nu:2} users: {:.1} tok/s total, {:.1} tok/s each",
            (nu * steps) as f64 / dt,
            steps as f64 / dt
        );
    }
}
