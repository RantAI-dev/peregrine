//! The real rerankers against llama.cpp v0.9.0 (roadmap S4.6b).
//!
//! `tests/fixtures/bert/rerank/` holds 320 (query, document) pairs in eight languages,
//! llama.cpp's own pair encoding per line (`ids.txt`, from `format_prompt_rerank`) and
//! its `/v1/rerank` scores on the Q8_0 file (`scores-llama-q8_0.f32`) and on the FP16
//! one (`scores-llama-fp16.f32`, the full-precision reference). The test needs the GGUF
//! files; CI has none, so it is `#[ignore]`d and prints `SKIPPED` without them.
//!
//! ```
//! cargo test --release -p peregrine-models --test rerank_real -- --ignored --nocapture
//! ```
//!
//! `RERANK_MODEL` and `RERANK_SCORES` compare another file against another score file:
//! with `RERANK_MODEL` set to the FP16 conversion and `RERANK_SCORES=scores-llama-fp16.f32`
//! the test measures the two engines at the same precision, where the only difference
//! left is llama.cpp's `GGML_GELU_FP16` table (the tanh approximation evaluated through
//! f16, `ggml-cpu/vec.h:1000-1013`) against this family's exact `erf` GELU.
//!
//! Measured on bluefin (4 threads, both files in `~/models/embed/`):
//!
//! | files | mean abs | p95 | max abs |
//! |---|---|---|---|
//! | Q8_0 vs llama.cpp Q8_0 | 0.0107 | 0.0588 | 0.186 |
//! | FP16 vs llama.cpp FP16 | 0.0047 | 0.0266 | 0.0688 |
//! | FP16 with llama.cpp's GELU substituted | 0.00056 | 0.0031 | 0.0067 |
//!
//! llama.cpp's own Q8_0 scores are 0.0158 (max 0.222) from its FP16 run, so the
//! quantisation moves its scores more than the two engines differ on the same file.
//! The design's 1e-3 was an F16-class number; the bound here is the measured one.

use peregrine_core::gguf::Format;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::Model;
use peregrine_core::serve::{Engine, SlotCfg};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One pair: the query, the document, llama.cpp's ids, and its two scores.
struct Pair {
    query: String,
    document: String,
    ids: Vec<u32>,
    q8: f32,
    fp16: f32,
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bert/rerank")
}

fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn f32s(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let (words, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "a score file is whole f32 values");
    words.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn pairs(ids_file: &str, scores_file: &str, fp16_file: &str) -> Vec<Pair> {
    let dir = fixtures();
    let texts = std::fs::read_to_string(dir.join("texts.tsv")).unwrap();
    let ids = std::fs::read_to_string(dir.join(ids_file)).unwrap();
    let q8 = f32s(&dir.join(scores_file));
    // Not every upload has an FP16 conversion beside it.
    let fp16 = if fp16_file.is_empty() {
        vec![f32::NAN; q8.len()]
    } else {
        f32s(&dir.join(fp16_file))
    };
    let texts: Vec<&str> = texts.lines().collect();
    let ids: Vec<&str> = ids.lines().collect();
    assert_eq!(texts.len(), q8.len(), "one score per pair");
    assert_eq!(texts.len(), fp16.len(), "one score per pair");
    texts
        .iter()
        .zip(ids)
        .enumerate()
        .map(|(i, (line, ids))| {
            let (q, d) = line.split_once('\t').expect("query<TAB>document");
            let ids: Vec<u32> = ids
                .trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse().unwrap())
                .collect();
            Pair {
                query: unescape(q),
                document: unescape(d),
                ids,
                q8: q8[i],
                fp16: fp16[i],
            }
        })
        .collect()
}

/// The head rows of a batch of texts, in order, through the engine.
fn head_rows(model: &Arc<Model>, texts: &[Vec<u32>], threads: usize) -> Vec<Vec<f32>> {
    let mut engine = Engine::new(
        model.clone(),
        threads,
        model.cfg.ctx_train,
        1,
        KvPrec::F16,
        0,
    );
    let mut by_slot: HashMap<u64, usize> = HashMap::new();
    let mut next = 0usize;
    let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
    while next < texts.len() && engine.work_room_for(texts[next].len()) {
        let id = engine
            .admit_head(texts[next].clone(), SlotCfg::default())
            .unwrap();
        by_slot.insert(id, next);
        next += 1;
    }
    let mut remaining = texts.len();
    while remaining > 0 {
        let emissions = engine.step();
        assert!(!emissions.is_empty(), "the engine made no progress");
        for em in emissions {
            let Some(i) = by_slot.remove(&em.slot) else {
                continue;
            };
            if let Some(peregrine_core::serve::Finish::Error(e)) = em.finish {
                panic!("pair {i}: {e}");
            }
            out[i] = em.head;
            remaining -= 1;
        }
        while next < texts.len() && engine.work_room_for(texts[next].len()) {
            let id = engine
                .admit_head(texts[next].clone(), SlotCfg::default())
                .unwrap();
            by_slot.insert(id, next);
            next += 1;
        }
    }
    out.into_iter()
        .map(|v| v.expect("a head row per pair"))
        .collect()
}

fn model_at(names: &[&str]) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    for base in [
        PathBuf::from(&home).join("models/embed"),
        PathBuf::from(&home).join("models"),
        PathBuf::from("models"),
    ] {
        for name in names {
            let p = base.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// The files this test knows: (label, names it may have on disk, ids fixture, score fixture).
/// `RERANK_MODEL` (with `RERANK_IDS` and `RERANK_SCORES`) compares another upload of the
/// same tokenizer against another recorded score file — the FP16 conversion, typically.
fn real_models() -> Vec<(String, PathBuf, String, String, String)> {
    if let Ok(p) = std::env::var("RERANK_MODEL") {
        let p = PathBuf::from(p);
        if !p.is_file() {
            return Vec::new();
        }
        let ids = std::env::var("RERANK_IDS").unwrap_or_else(|_| "v2_m3.ids".into());
        let scores =
            std::env::var("RERANK_SCORES").unwrap_or_else(|_| "v2_m3_q8_0_llama.f32".into());
        let fp16 = std::env::var("RERANK_FP16").unwrap_or_else(|_| "v2_m3_fp16_llama.f32".into());
        return vec![(p.display().to_string(), p, ids, scores, fp16)];
    }
    let mut out = Vec::new();
    for (label, names, ids, scores, fp16) in [
        (
            "bge-reranker-v2-m3",
            &["bge-reranker-v2-m3-Q8_0.gguf"][..],
            "v2_m3.ids",
            "v2_m3_q8_0_llama.f32",
            "v2_m3_fp16_llama.f32",
        ),
        (
            "bge-reranker-base",
            &["bge-reranker-base-q8_0.gguf"][..],
            "base.ids",
            "base_q8_0_llama.f32",
            "",
        ),
    ] {
        if let Some(path) = model_at(names) {
            out.push((
                label.to_string(),
                path,
                ids.to_string(),
                scores.to_string(),
                fp16.to_string(),
            ));
        }
    }
    out
}

fn load(path: &Path) -> Arc<Model> {
    let g = peregrine_core::Gguf::open(path).expect("open the model");
    assert_eq!(
        peregrine_models::load(g)
            .expect("load the model")
            .cfg
            .arch_name,
        "bert"
    );
    let g = peregrine_core::Gguf::open(path).unwrap();
    Arc::new(peregrine_models::load(g).unwrap())
}

/// Order agreement per query: the documents whose two scores are within `tie` of each
/// other may swap; every other pair of documents must keep its order.
fn order_disagreements(ours: &[f32], theirs: &[f32], tie: f32) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..ours.len() {
        for j in i + 1..ours.len() {
            let near = (theirs[i] - theirs[j]).abs() < tie || (ours[i] - ours[j]).abs() < tie;
            let same = (ours[i] > ours[j]) == (theirs[i] > theirs[j]);
            if !same && !near {
                out.push(format!(
                    "documents {i} and {j}: {:.4} vs {:.4} (llama {:.4} vs {:.4})",
                    ours[i], ours[j], theirs[i], theirs[j]
                ));
            }
        }
    }
    out
}

#[test]
#[ignore = "needs bge-reranker-v2-m3-Q8_0.gguf (and its FP16 conversion) and bge-reranker-base-q8_0.gguf in ~/models/embed/"]
fn the_real_rerankers_match_llama_cpp() {
    let models = real_models();
    if models.is_empty() {
        eprintln!("SKIPPED: no reranker GGUF file is in ~/models/embed/");
        return;
    }
    for (label, path, ids_file, scores_file, fp16_file) in models {
        let pairs = pairs(&ids_file, &scores_file, &fp16_file);
        let model = load(&path);
        assert!(
            model.has_head(),
            "{label}: the reranker's file has a classifier head"
        );
        assert_eq!(model.head_width(), 1, "{label}");

        // 1. Token ids: llama.cpp's `format_prompt_rerank`, exactly.
        let mut id_mismatches = 0;
        for (i, p) in pairs.iter().enumerate() {
            let got = model.tk.encode_pair(&p.query, &p.document);
            if got != p.ids {
                id_mismatches += 1;
                if id_mismatches <= 3 {
                    eprintln!("{label} pair {i}: ids {got:?} != llama {:?}", p.ids);
                }
            }
        }
        assert_eq!(id_mismatches, 0, "{label}: pair ids differ from llama.cpp");

        // 2. Scores: our head row[0] against llama.cpp's `/v1/rerank` score. A pair the
        // file's context cannot hold has no llama.cpp score (NaN) and is skipped — the
        // engine would refuse it too.
        let usable: Vec<usize> = (0..pairs.len())
            .filter(|&i| !pairs[i].q8.is_nan())
            .collect();
        let texts: Vec<Vec<u32>> = usable.iter().map(|&i| pairs[i].ids.clone()).collect();
        let rows = head_rows(&model, &texts, 4);
        let mut diffs: Vec<f32> = Vec::new();
        let mut ours: Vec<f32> = Vec::new();
        let mut llama: Vec<f32> = Vec::new();
        for (&i, row) in usable.iter().zip(&rows) {
            let ours_score = row[0];
            ours.push(ours_score);
            llama.push(pairs[i].q8);
            diffs.push((ours_score - pairs[i].q8).abs());
        }
        let used = usable;
        diffs.sort_by(f32::total_cmp);
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let max = *diffs.last().unwrap();
        let p95 = diffs[diffs.len() * 95 / 100];
        let fp16: Vec<f32> = used.iter().map(|&i| pairs[i].fp16).collect();
        let has_fp16 = fp16.iter().all(|s| !s.is_nan()) && pairs.iter().any(|p| !p.fp16.is_nan());
        let mean_of = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let max_of = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
        eprintln!(
            "{label}: {} pairs ({} scored), ids identical; ours vs llama.cpp: mean {mean:.5}, \
             p95 {p95:.5}, max {max:.5}",
            pairs.len(),
            diffs.len()
        );
        if has_fp16 && fp16.len() == ours.len() {
            let ours_vs_fp16: Vec<f32> =
                ours.iter().zip(&fp16).map(|(a, b)| (a - b).abs()).collect();
            let llama_vs_fp16: Vec<f32> = llama
                .iter()
                .zip(&fp16)
                .map(|(a, b)| (a - b).abs())
                .collect();
            eprintln!(
                "  {label}: ours vs llama.cpp FP16: mean {:.5}, max {:.5}; llama.cpp Q8_0 vs \
                 its FP16: mean {:.5}, max {:.5}",
                mean_of(&ours_vs_fp16),
                max_of(&ours_vs_fp16),
                mean_of(&llama_vs_fp16),
                max_of(&llama_vs_fp16),
            );
        }

        // 3. Order, per query of eight documents (a query whose documents all have a
        // score; the one pair a 512-token file cannot hold leaves its query out).
        let tie = 1e-2;
        let mut disagreements = Vec::new();
        let mut scored = 0usize;
        for q in 0..pairs.len() / 8 {
            let s = q * 8;
            let idx: Vec<usize> = (s..s + 8).filter(|i| !pairs[*i].q8.is_nan()).collect();
            if idx.len() != 8 {
                continue;
            }
            let mut a = Vec::with_capacity(8);
            let mut b = Vec::with_capacity(8);
            for i in idx {
                let at = used.iter().position(|&u| u == i).expect("a scored pair");
                a.push(ours[at]);
                b.push(llama[at]);
            }
            scored += 1;
            disagreements.extend(order_disagreements(&a, &b, tie));
        }
        eprintln!("  {label}: {scored} queries of eight documents compared for order");
        for d in &disagreements {
            eprintln!("{label} order: {d}");
        }
        assert!(
            disagreements.is_empty(),
            "{label}: {} document pairs changed order outside a {tie} near-tie",
            disagreements.len()
        );

        // The bound is the measured one (the doc comment's table): llama.cpp's own Q8_0
        // scores move by more than this against its FP16 run, so a tighter number would
        // be a claim about quantisation, not about the graph. The order check above is
        // the gate that matters for a reranker.
        assert!(
            mean < 2e-2,
            "{label}: mean score difference from llama.cpp is {mean:.5}"
        );
        assert!(
            max < 3e-1,
            "{label}: max score difference from llama.cpp is {max:.5}"
        );
    }
}

/// Invariance on the real file: a pair's score is bit-identical alone and in a batch of
/// two, seven and eight, at one, two and four threads.
#[test]
#[ignore = "needs bge-reranker-v2-m3-Q8_0.gguf in ~/models/embed/"]
fn a_real_rerankers_score_is_bit_identical_in_any_batch() {
    let Some(path) = model_at(&["bge-reranker-v2-m3-Q8_0.gguf"]) else {
        eprintln!("SKIPPED: bge-reranker-v2-m3 Q8_0 is not in ~/models/embed/");
        return;
    };
    let pairs = pairs("v2_m3.ids", "v2_m3_q8_0_llama.f32", "v2_m3_fp16_llama.f32");
    let model = load(&path);
    let probe = &pairs[0];
    let alone = head_rows(&model, std::slice::from_ref(&probe.ids), 1)[0][0];
    for (n, threads) in [(2usize, 1usize), (7, 2), (8, 4)] {
        let texts: Vec<Vec<u32>> = (0..n).map(|i| pairs[i * 3].ids.clone()).collect();
        let rows = head_rows(&model, &texts, threads);
        assert_eq!(
            rows[0][0].to_bits(),
            alone.to_bits(),
            "the probe's score changed in a batch of {n} at {threads} threads"
        );
    }
    // Every format of the file was read: the Q8_0 head's two matrices are in the file.
    let g = model.gguf();
    assert_eq!(g.tensor("cls.weight").unwrap().kind, Format::Q8_0);
    assert_eq!(g.tensor("cls.output.weight").unwrap().numel(), 1024);
}
