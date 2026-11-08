//! Greedy decoding with the top-5 logits at every step, for a comparison against llama.cpp.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example phi4_greedy -- \
//!     --features peregrine-models/arch-phi3,peregrine-models/fmt-kquant -- \
//!     <model.gguf> <n_predict> <prompt> [threads] [mask_eog]
//! ```
//!
//! `mask_eog` (default 1) takes the end-of-generation tokens out of the running, exactly as
//! llama.cpp's `ignore_eos` does: it puts a -inf logit bias on every EOG token, so the model
//! keeps generating instead of stopping. With it off, the end token can be chosen, and the
//! generated sequence stops there.

use peregrine_core::gguf::Gguf;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::StepStats;
use peregrine_core::pool::Pool;
use std::path::Path;

fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best as u32
}

/// The five highest logits, as (id, logit) pairs, best first.
fn top5(xs: &[f32]) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| {
        xs[b]
            .partial_cmp(&xs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(5);
    idx.into_iter().map(|i| (i, xs[i])).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("the model file");
    let n_predict: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let prompt = args.get(2).cloned().unwrap_or_default();
    let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let mask_eog: bool = args.get(4).map(|s| s != "0").unwrap_or(true);

    let g = Gguf::open(Path::new(path)).expect("open");
    // The tokens llama.cpp treats as end-of-generation: the file's own EOS and `<|endoftext|>`.
    let mut eog: Vec<u32> = Vec::new();
    if let Ok(e) = g.u32("tokenizer.ggml.eos_token_id") {
        eog.push(e);
    }
    for (i, t) in g
        .arr_str("tokenizer.ggml.tokens")
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        if t == "<|endoftext|>" {
            eog.push(i as u32);
        }
    }
    eog.sort_unstable();
    eog.dedup();
    let model = peregrine_models::load(g).expect("load");
    let toks = model.tk.encode(&prompt);
    eprintln!("prompt tokens ({}): {:?}", toks.len(), toks);
    eprintln!("end-of-generation tokens masked: {eog:?}");

    let pool = Pool::new(threads);
    let mut st = model.new_state(128, toks.len() + n_predict, 1024, threads);
    let mut kv = model.new_cache(KvPrec::F16, 1024);
    let mut ss = StepStats::default();
    let want: Vec<usize> = vec![toks.len() - 1];
    model
        .forward(&mut st, &mut kv, &pool, &toks, 0, &want, &mut ss)
        .unwrap();

    println!("PROMPT_TOKENS {}", toks.len());
    let mut tokens: Vec<u32> = Vec::with_capacity(n_predict);
    for step in 0..n_predict {
        let mut logits = if step == 0 {
            st.logits_row(0, model.cfg.n_vocab).to_vec()
        } else {
            let pos = toks.len() + step - 1;
            model
                .forward(
                    &mut st,
                    &mut kv,
                    &pool,
                    &[tokens[step - 1]],
                    pos,
                    &[0],
                    &mut ss,
                )
                .unwrap();
            st.logits_row(0, model.cfg.n_vocab).to_vec()
        };
        if mask_eog {
            for &id in &eog {
                logits[id as usize] = f32::NEG_INFINITY;
            }
        }
        let next = argmax(&logits);
        let tops: Vec<String> = top5(&logits)
            .iter()
            .map(|(i, v)| format!("{i}={v:.6}"))
            .collect();
        println!("STEP {step} token {next} top {}", tops.join(";"));
        tokens.push(next);
        if !mask_eog && eog.contains(&next) {
            eprintln!("stopped on the end token {next} at step {step}");
            break;
        }
    }
    println!(
        "TOKENS {}",
        tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
}
