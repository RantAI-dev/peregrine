//! Check a plain Qwen 3.5 file against llama.cpp: the same greedy tokens, and how far the
//! logits are apart.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example qwen35_check -- <model.gguf> <n_predict> <prompt>
//! ```

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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("the model file");
    let n_predict: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let prompt = args.get(2).cloned().unwrap_or_default();

    let model = Gguf::open(Path::new(path))
        .and_then(peregrine_models::load)
        .expect("load");
    let toks = model.tk.encode(&prompt);
    eprintln!("prompt tokens ({}): {:?}", toks.len(), toks);

    let pool = Pool::new(1);
    let mut st = model.new_state(128, toks.len() + n_predict, 256, 1);
    let mut kv = model.new_cache(KvPrec::F16, 256);
    let mut ss = StepStats::default();
    let want: Vec<usize> = vec![toks.len() - 1];
    model
        .forward(&mut st, &mut kv, &pool, &toks, 0, &want, &mut ss)
        .unwrap();
    let logits = st.logits_row(0, model.cfg.n_vocab).to_vec();
    let mut next = argmax(&logits);
    let mut tokens = vec![next];
    for step in 1..n_predict {
        let pos = toks.len() + step - 1;
        model
            .forward(&mut st, &mut kv, &pool, &[next], pos, &[0], &mut ss)
            .unwrap();
        next = argmax(st.logits_row(0, model.cfg.n_vocab));
        tokens.push(next);
    }
    println!(
        "TOKENS: {}",
        tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "LOGITS: {}",
        logits
            .iter()
            .map(|v| format!("{v:e}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
}
