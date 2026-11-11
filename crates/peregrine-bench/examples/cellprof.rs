//! Per-cell stage profile of the embedding engine, for the S4.9 standings.
//!
//! Drives the same `Engine` the server and `peregrine embed` use, one cell at a
//! time, and prints what the phase counters say: the step statistics
//! (matmul/head/attention/other, from `StepStats`) and the `PEREGRINE_PROF2`
//! buckets (quantise, barrier wait, norms, ...). Tokenisation is timed apart,
//! before the engine is given the token ids, so a cell's request wall is
//! `tokenise + engine + the harness's own HTTP`.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example cellprof -- \
//!     model.gguf texts.json 16,128,512,2000 1,8,32 4 [--mixed 32]
//! ```
//!
//! The texts file is the one `benchmarks/embed_texts.py` writes.

use peregrine_core::embed::{self, Pooling};
use peregrine_core::kv::KvPrec;
use peregrine_core::options;
use peregrine_core::serve::Engine;
use peregrine_core::state::StepStats;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_path = PathBuf::from(args.first().expect("model path"));
    let texts_path = PathBuf::from(args.get(1).expect("texts json"));
    let lengths: Vec<usize> = args
        .get(2)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![16, 128, 512, 2000]);
    let batches: Vec<usize> = args
        .get(3)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 8, 32]);
    let threads: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let mut mixed_batch = 0usize;
    let mut i = 5;
    while i < args.len() {
        if args[i] == "--mixed" {
            mixed_batch = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(32);
            i += 2;
        } else {
            i += 1;
        }
    }

    options::init(options::EngineOptions::from_env_and_profile(None)).expect("engine options");
    let model = peregrine_models::open(&model_path).expect("open the model");
    let pooling = model
        .pooling()
        .expect("a pooling type")
        .unwrap_or(Pooling::Mean);
    let max_ctx = 2048usize;

    let json = std::fs::read_to_string(&texts_path).expect("texts json");
    let mut cells: Vec<(String, Vec<String>, bool)> = Vec::new();
    for &l in &lengths {
        let texts = texts_at(&json, &l.to_string());
        for &b in &batches {
            if b * l > 32768 {
                continue;
            }
            cells.push((format!("L{l}_b{b}"), texts.clone(), false));
        }
    }
    if mixed_batch > 0 {
        cells.push((
            format!("mixed_b{mixed_batch}"),
            texts_at(&json, "mixed"),
            true,
        ));
    }

    let mut engine = Engine::new(model.clone(), threads, max_ctx, 1, KvPrec::F16, 0);
    // Warm-up: the first pass faults in the mapped weights and allocates the
    // state, so it is not part of any cell.
    warm(
        &mut engine,
        &model.tk,
        &json,
        lengths.first().copied().unwrap_or(16),
        pooling,
    );
    let mut out = Vec::new();
    for (key, texts, mixed) in cells {
        let batch = if mixed {
            mixed_batch
        } else {
            key.split("_b")
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1)
        };
        let texts = &texts[..batch.min(texts.len())];

        // Tokenise first, timed on its own: the engine below receives ids.
        let t_tok = Instant::now();
        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
        for t in texts {
            ids.push(model.tk.encode_special(t));
        }
        let tokenise_s = t_tok.elapsed().as_secs_f64();
        let total_tokens: usize = ids.iter().map(Vec::len).sum();
        let bytes: usize = texts.iter().map(String::len).sum();

        peregrine_core::prof::reset();
        let before = engine.stats;
        let t0 = Instant::now();

        let mut by_slot: HashMap<u64, usize> = HashMap::new();
        let mut next = 0usize;
        let mut done = vec![false; ids.len()];
        while next < ids.len() && engine.work_room_for(ids[next].len()) {
            let id = engine
                .admit_pooled(
                    ids[next].clone(),
                    peregrine_core::serve::SlotCfg::default(),
                    pooling,
                )
                .expect("admit");
            by_slot.insert(id, next);
            next += 1;
        }
        let mut remaining = ids.len();
        while remaining > 0 {
            for em in engine.step() {
                let Some(&idx) = by_slot.get(&em.slot) else {
                    continue;
                };
                if let Some(peregrine_core::serve::Finish::Error(e)) = em.finish {
                    panic!("text {idx}: {e}");
                }
                let mut v = em.embedding.expect("an embedding");
                embed::finish(&mut v, None).expect("normalise");
                done[idx] = true;
                remaining -= 1;
            }
            while next < ids.len() && engine.work_room_for(ids[next].len()) {
                let id = engine
                    .admit_pooled(
                        ids[next].clone(),
                        peregrine_core::serve::SlotCfg::default(),
                        pooling,
                    )
                    .expect("admit");
                by_slot.insert(id, next);
                next += 1;
            }
        }
        let wall = t0.elapsed().as_secs_f64();
        let st: StepStats = engine.stats.since(&before);
        let prof = peregrine_core::prof::snapshot();
        let names = peregrine_core::prof::names();
        let mut prof_txt = String::new();
        for (name, ns) in names.iter().zip(prof.iter()) {
            if *ns > 0 {
                if !prof_txt.is_empty() {
                    prof_txt.push(',');
                }
                prof_txt.push_str(&format!("\"{name}\":{:.3}", *ns as f64 / 1e6));
            }
        }
        let line = format!(
            "{{\"cell\":\"{key}\",\"batch\":{batch},\"tokens\":{total_tokens},\"bytes\":{bytes},\"texts_per_s\":{:.3},\"tokens_per_s\":{:.1},\"wall_s\":{:.4},\"tokenise_s\":{:.6},\"tokenise_ms_per_text\":{:.3},\"stats_ms\":{{\"matmul\":{:.3},\"head\":{:.3},\"attention\":{:.3},\"other\":{:.3},\"total\":{:.3},\"barriers\":{}}},\"prof_ms\":{{{prof_txt}}}}}",
            (batch as f64) / wall,
            (total_tokens as f64) / wall,
            wall,
            tokenise_s,
            tokenise_s * 1e3 / batch as f64,
            st.ns_matmul as f64 / 1e6,
            st.ns_head as f64 / 1e6,
            st.ns_attn as f64 / 1e6,
            st.ns_other() as f64 / 1e6,
            st.ns_total as f64 / 1e6,
            st.barriers,
        );
        println!("{line}");
        out.push(line);
    }
    let _ = out;
}

/// The `"text"` strings of one length key in the texts file.
///
/// The file is plain JSON written by `benchmarks/embed_texts.py`; this reads
/// just what that writer emits, without pulling a JSON crate into the bench.
fn texts_at(json: &str, key: &str) -> Vec<String> {
    let marker = format!("\"{key}\": [");
    let mut rest = &json[json.find(&marker).expect("a length key") + marker.len()..];
    let mut texts = Vec::new();
    while let Some(i) = rest.find("\"text\": \"") {
        rest = &rest[i + 9..];
        let mut out = String::new();
        let mut esc = false;
        for ch in rest.chars() {
            if esc {
                out.push(match ch {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    other => other,
                });
                esc = false;
            } else if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                break;
            } else {
                out.push(ch);
            }
        }
        texts.push(out);
        rest = &rest[rest.find("}").map(|j| j + 1).unwrap_or(rest.len())..];
    }
    texts
}

/// One text through the engine, its vector dropped: the cold pass of a cell.
fn warm(
    engine: &mut Engine,
    tk: &peregrine_core::tokenizer::Tokenizer,
    json: &str,
    length: usize,
    pooling: Pooling,
) {
    let text = texts_at(json, &length.to_string())
        .into_iter()
        .next()
        .expect("a text");
    let id = engine
        .admit_pooled(
            tk.encode_special(&text),
            peregrine_core::serve::SlotCfg::default(),
            pooling,
        )
        .expect("warm-up admit");
    while engine.active() > 0 {
        let emitted = engine.step();
        if emitted.iter().any(|e| e.slot == id) {
            break;
        }
    }
}
