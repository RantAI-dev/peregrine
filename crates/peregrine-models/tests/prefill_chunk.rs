//! The prefill chunk (`PEREGRINE_PREFILL_CHUNK`) is how a prompt is *read*, not what it produces.
//!
//! A prompt is read in pieces of `prefill_chunk` tokens, and the engine's mixed step reads one
//! piece beside the other users' decode rows. This test pins that the piece does not change the
//! numbers: the same prompt, read in pieces of 16, 32, 64 or 128 tokens, gives the same KV cache,
//! the same logits and the same tokens.
//!
//! The test is about the read, not about the model: a synthetic model (no model file needed) and,
//! with `PEREGRINE_MODEL` set, a real one.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, StepStats};
use peregrine_core::pool::Pool;
use peregrine_core::serve::{Engine, Finish, SlotCfg};
use std::path::PathBuf;
use std::sync::Arc;
use support::models::{write_dense, Dense};

/// The chunk sizes the A/B measured: 16 (the default), 32, 64 (the old default) and 128 (the
/// matmul scratch's row limit).
const CHUNKS: [usize; 4] = [16, 32, 64, 128];

/// The prompt: longer than the widest chunk, so every size splits it.
fn prompt() -> Vec<u32> {
    (0..200u32).map(|i| (i * 37 + 11) % 300).collect()
}

/// A directory of its own for each test (tests run in parallel and each removes its directory).
fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-chunk-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The model's KV cache, read back and widened: the bytes the attention will read.
fn cache_floats(
    kv: &mut peregrine_core::kv::KvCache,
    layers: usize,
    heads: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = Vec::new();
    let mut buf = vec![0.0f32; dim];
    for l in 0..layers {
        for h in 0..heads {
            for pos in 0..200 {
                kv.decode_block(true, l, h, pos, 1, &mut buf);
                out.extend_from_slice(&buf);
                kv.decode_block(false, l, h, pos, 1, &mut buf);
                out.extend_from_slice(&buf);
            }
        }
    }
    out
}

/// The prompt read in pieces of `chunk` tokens: the cache it left and the logits of its last
/// position.
fn read_in_pieces(model: &Model, chunk: usize, toks: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let c = model.cfg;
    let threads = 4;
    let pool = Pool::new(threads);
    let mut st = model.new_state(128, 1, 512, threads);
    let mut kv = model.new_cache(KvPrec::F16, 512);
    let mut ss = StepStats::default();
    let mut fed = 0usize;
    let mut logits = Vec::new();
    while fed < toks.len() {
        let take = chunk.min(toks.len() - fed);
        let piece = &toks[fed..fed + take];
        // Only the last piece wants logits: the position the prompt ends on.
        let want: Vec<usize> = if fed + take == toks.len() {
            vec![take - 1]
        } else {
            Vec::new()
        };
        model
            .forward(&mut st, &mut kv, &pool, piece, fed, &want, &mut ss)
            .unwrap();
        if !want.is_empty() {
            // The logits of the wanted row are the first (and only) logits row.
            logits = st.logits_row(0, c.n_vocab).to_vec();
        }
        fed += take;
    }
    let cache = cache_floats(&mut kv, c.n_layer, c.n_kv_head, c.head_dim);
    (cache, logits)
}

/// The same prompt read in pieces of `chunk` tokens gives the same cache and the same logits.
#[test]
fn the_prompt_says_the_same_whatever_the_chunk_size() {
    let dir = tmpdir("pieces");
    let mut failures = Vec::new();
    for (i, fmt) in [Format::Q8_0, Format::Q4_K].into_iter().enumerate() {
        let path = write_dense(&dir, "llama", fmt, 7100 + i as u64, Dense::default());
        let model = match Gguf::open(&path).and_then(peregrine_models::load) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("llama-{}: does not load: {e}", fmt.label()));
                continue;
            }
        };
        let toks = prompt();
        let (cache0, logits0) = read_in_pieces(&model, CHUNKS[0], &toks);
        for &chunk in &CHUNKS[1..] {
            let (cache, logits) = read_in_pieces(&model, chunk, &toks);
            if cache != cache0 {
                let first = cache.iter().zip(cache0.iter()).position(|(a, b)| a != b);
                failures.push(format!(
                    "llama-{}: the cache differs at chunk {chunk} (first at {:?})",
                    fmt.label(),
                    first
                ));
            }
            if logits != logits0 {
                failures.push(format!(
                    "llama-{}: the logits differ at chunk {chunk}",
                    fmt.label()
                ));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the prefill chunk changed the numbers:\n{}",
        failures.join("\n")
    );
}

/// The engine reads the prompt in pieces of `chunk` tokens and gives the same tokens either way.
#[test]
fn the_engine_gives_the_same_tokens_at_any_chunk_size() {
    let dir = tmpdir("engine");
    let mut failures = Vec::new();
    for (i, fmt) in [Format::Q8_0, Format::Q4_K].into_iter().enumerate() {
        let path = write_dense(&dir, "llama", fmt, 7200 + i as u64, Dense::default());
        let model = match Gguf::open(&path).and_then(peregrine_models::load) {
            Ok(m) => Arc::new(m),
            Err(e) => {
                failures.push(format!("llama-{}: does not load: {e}", fmt.label()));
                continue;
            }
        };
        let toks = prompt();
        let mut want: Option<Vec<u32>> = None;
        for &chunk in &CHUNKS {
            let mut e = Engine::new_with_chunk(model.clone(), 4, 512, 1, KvPrec::F16, 0, chunk);
            let cfg = SlotCfg {
                temperature: 0.0,
                max_tokens: 16,
                ..Default::default()
            };
            if let Err((_, e)) = e.admit(toks.clone(), cfg) {
                failures.push(format!("llama-{}: does not admit: {e}", fmt.label()));
                break;
            }
            let mut got = Vec::new();
            while e.active() > 0 {
                for em in e.step() {
                    if let Some(Finish::Error(e)) = em.finish {
                        failures.push(format!("llama-{}: {e}", fmt.label()));
                        break;
                    }
                    got.push(em.token);
                }
            }
            match &want {
                None => want = Some(got),
                Some(w) if *w != got => failures.push(format!(
                    "llama-{}: the tokens differ at chunk {chunk}",
                    fmt.label()
                )),
                Some(_) => {}
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the prefill chunk changed the tokens:\n{}",
        failures.join("\n")
    );
}

/// The same on a real model: `PEREGRINE_MODEL` else the repository's Qwen3 model.
#[test]
fn a_real_model_gives_the_same_tokens_at_any_chunk_size() {
    let p = match std::env::var_os("PEREGRINE_MODEL") {
        Some(p) => PathBuf::from(p),
        None => {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen3-1.7B-Q4_K_M.gguf")
        }
    };
    if !p.exists() {
        eprintln!("skipped: set PEREGRINE_MODEL");
        return;
    }
    let model = peregrine_models::open(&p).unwrap();
    let toks: Vec<u32> = (0..200u32).map(|i| (i * 37 + 11) % 900).collect();
    let mut want: Option<Vec<u32>> = None;
    for &chunk in &CHUNKS {
        let mut e = Engine::new_with_chunk(model.clone(), 4, 512, 1, KvPrec::F16, 0, chunk);
        let cfg = SlotCfg {
            temperature: 0.0,
            max_tokens: 16,
            ..Default::default()
        };
        e.admit(toks.clone(), cfg).unwrap();
        let mut got = Vec::new();
        while e.active() > 0 {
            for em in e.step() {
                got.push(em.token);
            }
        }
        match &want {
            None => want = Some(got),
            Some(w) => assert_eq!(*w, got, "the tokens differ at chunk {chunk}"),
        }
    }
}
