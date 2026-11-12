//! Prompt processing of a real text, through the engine, with its own phase split.
//!
//! `peregrine bench --prompt-tokens N` reads synthetic token ids. This reads the first
//! `N` tokens of a text file instead — the same plain prose every earlier desktop round
//! used, no chat template — and reports what the engine's own accounting says:
//! prefill tok/s, decode tok/s, and (with `PEREGRINE_PROF2=full`) the `StepStats`
//! buckets and the `peregrine_core::prof` slots, split into the caller's own time and
//! the workers' (summed over the pool's width, as `peregrine bench` prints them).
//!
//! ```text
//! cargo run --release -p peregrine-bench --example promptbench -- \
//!     model.gguf corpus.txt 512,4096 1,6,12,24 [--decode 32] [--slots 1] [--chunk 16]
//! ```
//!
//! One cell is one fresh engine: the KV cache is empty, as a benchmark's must be, and
//! the weights are shared. Every cell prints one JSON line; the driver interleaves the
//! cells across rounds and takes medians, because this host's clock has moved before
//! (`docs/design/s5-prefill-avx512.md` §10.0).

use peregrine_core::kv::KvPrec;
use peregrine_core::options;
use peregrine_core::serve::{Emission, Engine, Finish};
use peregrine_core::state::StepStats;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_path = PathBuf::from(args.first().expect("model path"));
    let corpus_path = PathBuf::from(args.get(1).expect("corpus path"));
    let lengths: Vec<usize> = args
        .get(2)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![512, 4096]);
    let threads: Vec<usize> = args
        .get(3)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 6, 12, 24]);
    let mut decode = 0usize;
    let mut slots = 1usize;
    let mut chunk: Option<usize> = None;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--decode" => {
                decode = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--slots" => {
                slots = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1);
                i += 2;
            }
            "--chunk" => {
                chunk = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            _ => i += 1,
        }
    }

    options::init(options::EngineOptions::from_env_and_profile(None)).expect("engine options");
    let model = peregrine_models::open(&model_path).expect("open the model");
    let text = std::fs::read_to_string(&corpus_path).expect("corpus");
    let ids = model.tk.encode_special(&text);
    let longest = *lengths.iter().max().expect("a length");
    assert!(
        ids.len() >= longest + slots * 7,
        "corpus is {} tokens, need {}",
        ids.len(),
        longest + slots * 7
    );

    for &len in &lengths {
        for &th in &threads {
            let max_ctx = len + decode + 16;
            let mut engine = match chunk {
                Some(c) => {
                    Engine::new_with_chunk(model.clone(), th, max_ctx, slots, KvPrec::F16, 0, c)
                }
                None => Engine::new(model.clone(), th, max_ctx, slots, KvPrec::F16, 0),
            };
            let width = engine.pool().width().max(1);
            // Every slot gets its own window of the corpus, so a shared prefix cannot
            // flatter the batched cell.
            let prompts: Vec<Vec<u32>> = (0..slots)
                .map(|s| ids[s * 7..s * 7 + len + decode.min(1)].to_vec())
                .collect();
            for p in &prompts {
                engine.admit(p.clone(), Default::default()).expect("admit");
            }
            peregrine_core::prof::reset();
            let before = engine.stats;
            let pre_prof = peregrine_core::prof::snapshot();
            let t0 = Instant::now();
            let mut prefill_rows = 0usize;
            let mut prefill_s = 0.0f64;
            let mut decoded = 0usize;
            let mut decode_s = 0.0f64;
            let mut emitted = 0usize;
            let mut prof_pre = [0u64; peregrine_core::prof::N];
            let mut prof_dec = [0u64; peregrine_core::prof::N];
            let mut steps_pre = 0usize;
            let mut steps_dec = 0usize;
            while engine.active() > 0 {
                let mark = peregrine_core::prof::snapshot();
                let ems: Vec<Emission> = engine.step();
                let rep = engine.last_step;
                if ems.is_empty() {
                    prefill_rows += rep.rows;
                    prefill_s += rep.secs;
                    steps_pre += 1;
                    if peregrine_core::prof::enabled() {
                        let now = peregrine_core::prof::snapshot();
                        for (k, slot) in prof_pre.iter_mut().enumerate() {
                            *slot += now[k].saturating_sub(mark[k]);
                        }
                    }
                    if decode == 0 && prefill_rows >= len * slots {
                        break;
                    }
                    continue;
                }
                decode_s += rep.secs;
                decoded += rep.rows.max(1);
                emitted += ems.len();
                steps_dec += 1;
                for em in &ems {
                    if let Some(Finish::Error(e)) = &em.finish {
                        panic!("slot error: {e}");
                    }
                }
                if peregrine_core::prof::enabled() {
                    let now = peregrine_core::prof::snapshot();
                    for (k, slot) in prof_dec.iter_mut().enumerate() {
                        *slot += now[k].saturating_sub(mark[k]);
                    }
                }
                if emitted >= slots * decode {
                    break;
                }
            }
            let wall = t0.elapsed().as_secs_f64();
            let st: StepStats = engine.stats.since(&before);
            let prof = peregrine_core::prof::snapshot();
            let names = peregrine_core::prof::names();
            let _ = (prof, pre_prof);
            let mut prof_txt = String::new();
            for (k, name) in names.iter().enumerate() {
                if prof_pre[k] > 0 {
                    prof_txt.push_str(&format!("\"{name}\":{},", prof_pre[k]));
                }
            }
            for (k, name) in names.iter().enumerate() {
                if prof_dec[k] > 0 {
                    prof_txt.push_str(&format!("\"dec {name}\":{},", prof_dec[k]));
                }
            }
            let prof_txt = prof_txt.trim_end_matches(',').to_string();
            println!(
                "{{\"len\":{len},\"threads\":{th},\"slots\":{slots},\"decode_want\":{decode},\
                 \"width\":{width},\"prefill_rows\":{prefill_rows},\"prefill_s\":{prefill_s:.6},\
                 \"prefill_tok_s\":{:.2},\"decode_rows\":{decoded},\"decode_s\":{decode_s:.6},\
                 \"decode_tok_s\":{:.2},\"wall_s\":{wall:.6},\"steps_pre\":{steps_pre},\
                 \"steps_dec\":{steps_dec},\"stats_ms\":{{\"matmul\":{:.3},\"head\":{:.3},\
                 \"attention\":{:.3},\"other\":{:.3},\"total\":{:.3},\"barriers\":{}}},\
                 \"prof_ms\":{{{prof_txt}}}}}",
                if prefill_s > 0.0 {
                    prefill_rows as f64 / prefill_s
                } else {
                    0.0
                },
                if decode_s > 0.0 {
                    decoded as f64 / decode_s
                } else {
                    0.0
                },
                st.ns_matmul as f64 / 1e6,
                st.ns_head as f64 / 1e6,
                st.ns_attn as f64 / 1e6,
                st.ns_other() as f64 / 1e6,
                st.ns_total as f64 / 1e6,
                st.barriers,
            );
        }
    }
}
