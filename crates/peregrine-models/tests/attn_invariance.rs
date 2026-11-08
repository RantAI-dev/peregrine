//! A row's attention arithmetic must not depend on the other rows in its call.
//!
//! The position split of attention (`crates/peregrine-core/src/attn.rs`) is a decode
//! measure. If it followed the call width (`nb`), the same row would get different
//! bits when read alone, as part of a prefill chunk, or as one of a speculative
//! verify's rows. These tests compare the whole cache and the logits *exactly*.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::{KvCache, KvPrec};
use peregrine_core::model::{Model, StepStats};
use peregrine_core::pool::Pool;
use std::path::PathBuf;
use support::models::{write_dense, Dense};

/// Wide enough that one 300-position row clears the attention task gate
/// (`work = n_head * ctx * head_dim * 2 >= 1 << 19`), so positions split at short
/// context — the case S3.6/PR #94 lowers the threshold for.
fn wide() -> Dense {
    Dense {
        d_model: 1024,
        n_layer: 2,
        n_head: 8,
        n_head_kv: 2,
        n_ff: 2048,
        ctx: 1024,
    }
}

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-attn-inv-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n).map(|i| (i * 37 + 11) as u32 % 320).collect()
}

fn cache_floats(kv: &mut KvCache, layers: usize, heads: usize, dim: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::new();
    let mut buf = vec![0.0f32; dim];
    for l in 0..layers {
        for h in 0..heads {
            for pos in 0..n {
                kv.decode_block(true, l, h, pos, 1, &mut buf);
                out.extend_from_slice(&buf);
                kv.decode_block(false, l, h, pos, 1, &mut buf);
                out.extend_from_slice(&buf);
            }
        }
    }
    out
}

/// Read `toks` in pieces of `chunk`; return the cache and the logits of the last position.
fn read_in_pieces(
    model: &Model,
    chunk: usize,
    toks: &[u32],
    threads: usize,
) -> (Vec<f32>, Vec<f32>) {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let n = toks.len();
    let mut st = model.new_state(128, 1, 1024, threads);
    let mut kv = model.new_cache(KvPrec::F16, 1024);
    let mut ss = StepStats::default();
    let mut fed = 0usize;
    let mut logits = Vec::new();
    while fed < n {
        let take = chunk.min(n - fed);
        let piece = &toks[fed..fed + take];
        let want: Vec<usize> = if fed + take == n {
            vec![take - 1]
        } else {
            Vec::new()
        };
        model
            .forward(&mut st, &mut kv, &pool, piece, fed, &want, &mut ss)
            .unwrap();
        if !want.is_empty() {
            logits = st.logits_row(0, c.n_vocab).to_vec();
        }
        fed += take;
    }
    let cache = cache_floats(&mut kv, c.n_layer, c.n_kv_head, c.head_dim, n);
    (cache, logits)
}

#[test]
fn prefill_chunk_does_not_change_a_long_prompt() {
    let dir = tmpdir("pieces");
    let path = write_dense(&dir, "llama", Format::F16, 9100, wide());
    let model = Gguf::open(&path).and_then(peregrine_models::load).unwrap();
    let mut failures = Vec::new();
    for &n in &[300usize, 700] {
        let toks = prompt(n);
        let (cache0, logits0) = read_in_pieces(&model, 1, &toks, 4);
        for &chunk in &[7usize, 64, 128] {
            let (cache, logits) = read_in_pieces(&model, chunk, &toks, 4);
            if let Some(at) = cache.iter().zip(&cache0).position(|(a, b)| a != b) {
                failures.push(format!("{n} tokens: cache differs at chunk {chunk} ({at})"));
            }
            if let Some(at) = logits.iter().zip(&logits0).position(|(a, b)| a != b) {
                failures.push(format!("{n} tokens: logits differ at chunk {chunk} ({at})"));
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

/// One prefill of `n` tokens, then the next `k` tokens either one at a time or as a
/// speculative verify's k rows over one cache. Every row of the batch must equal the
/// solo decode of that position, bit for bit.
#[test]
fn verify_rows_match_their_solo_decodes() {
    let dir = tmpdir("verify");
    let path = write_dense(&dir, "llama", Format::F16, 9200, wide());
    let model = Gguf::open(&path).and_then(peregrine_models::load).unwrap();
    let c = model.cfg;
    let k = 5usize;
    let threads = 4;
    let pool = Pool::new(threads);

    fn base(
        model: &Model,
        pool: &Pool,
        toks: &[u32],
        n: usize,
        threads: usize,
        ss: &mut StepStats,
    ) -> (KvCache, peregrine_core::model::State) {
        let mut kv = model.new_cache(KvPrec::F16, 1024);
        let mut st = model.new_state(128, 8, 1024, threads);
        let mut fed = 0;
        while fed < n {
            let take = 16.min(n - fed);
            model
                .forward(&mut st, &mut kv, pool, &toks[fed..fed + take], fed, &[], ss)
                .unwrap();
            fed += take;
        }
        (kv, st)
    }

    let mut failures = Vec::new();
    for &n in &[300usize, 700] {
        let toks = prompt(n + k);

        // Solo: one row per call.
        let mut solo_logits = Vec::new();
        {
            let mut ss = StepStats::default();
            let (mut kv, mut st) = base(&model, &pool, &toks, n, threads, &mut ss);
            for b in 0..k {
                model
                    .forward(
                        &mut st,
                        &mut kv,
                        &pool,
                        &toks[n + b..n + b + 1],
                        n + b,
                        &[0],
                        &mut ss,
                    )
                    .unwrap();
                solo_logits.push(st.logits_row(0, c.n_vocab).to_vec());
            }
        }

        // Verify: k rows over the one cache, as the engine's speculative path does.
        let mut batch_logits = Vec::new();
        {
            let mut ss = StepStats::default();
            let (mut kv, mut st) = base(&model, &pool, &toks, n, threads, &mut ss);
            let want: Vec<usize> = (0..k).collect();
            model
                .forward(&mut st, &mut kv, &pool, &toks[n..n + k], n, &want, &mut ss)
                .unwrap();
            for b in 0..k {
                batch_logits.push(st.logits_row(b, c.n_vocab).to_vec());
            }
        }

        for b in 0..k {
            if let Some(at) = solo_logits[b]
                .iter()
                .zip(&batch_logits[b])
                .position(|(x, y)| x != y)
            {
                failures.push(format!(
                    "{n} tokens: row {b} (ctx {}): logit {at} differs",
                    n + b + 1
                ));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "a verify row changed when batched:\n{}",
        failures.join("\n")
    );
}

/// A mixed step: one user's prompt chunk beside another user's decode row, one
/// weight pass. Neither row may notice the other. The engine reaches this with
/// `Caches::Mapped`, which groups rows by cache and calls attention once per run.
#[test]
fn a_mixed_prefill_and_decode_step_matches_each_user_alone() {
    let dir = tmpdir("mixed");
    let path = write_dense(&dir, "llama", Format::F16, 9300, wide());
    let model = Gguf::open(&path).and_then(peregrine_models::load).unwrap();
    let c = model.cfg;
    let threads = 4;
    let pool = Pool::new(threads);
    let toks_a = prompt(340);
    let toks_b: Vec<u32> = prompt(260).iter().map(|t| (t + 7) % 320).collect();
    let (pa, pb, take) = (300usize, 200usize, 16usize);

    // A cache holding `toks[..n]`, read in the engine's own prefill pieces.
    fn prefix(model: &Model, pool: &Pool, toks: &[u32], n: usize, threads: usize) -> KvCache {
        let mut kv = model.new_cache(KvPrec::F16, 1024);
        let mut st = model.new_state(128, 1, 1024, threads);
        let mut ss = StepStats::default();
        let mut fed = 0;
        while fed < n {
            let k = 16.min(n - fed);
            model
                .forward(
                    &mut st,
                    &mut kv,
                    pool,
                    &toks[fed..fed + k],
                    fed,
                    &[],
                    &mut ss,
                )
                .unwrap();
            fed += k;
        }
        kv
    }

    // Alone: A's next prompt chunk (16 rows) and B's next decode row.
    let mut alone_a = Vec::new();
    {
        let mut kv = prefix(&model, &pool, &toks_a, pa, threads);
        let mut st = model.new_state(128, take, 1024, threads);
        let mut ss = StepStats::default();
        let want: Vec<usize> = (0..take).collect();
        model
            .forward(
                &mut st,
                &mut kv,
                &pool,
                &toks_a[pa..pa + take],
                pa,
                &want,
                &mut ss,
            )
            .unwrap();
        for i in 0..take {
            alone_a.push(st.logits_row(i, c.n_vocab).to_vec());
        }
    }
    let alone_b = {
        let mut kv = prefix(&model, &pool, &toks_b, pb, threads);
        let mut st = model.new_state(128, 1, 1024, threads);
        let mut ss = StepStats::default();
        model
            .forward(
                &mut st,
                &mut kv,
                &pool,
                &toks_b[pb..pb + 1],
                pb,
                &[0],
                &mut ss,
            )
            .unwrap();
        st.logits_row(0, c.n_vocab).to_vec()
    };

    // Together: one forward_rows over both caches.
    let mut kv_a = prefix(&model, &pool, &toks_a, pa, threads);
    let mut kv_b = prefix(&model, &pool, &toks_b, pb, threads);
    let mut st = model.new_state(128, take + 1, 1024, threads);
    let mut ss = StepStats::default();
    let mut tokens: Vec<u32> = toks_a[pa..pa + take].to_vec();
    tokens.push(toks_b[pb]);
    let mut positions: Vec<usize> = (pa..pa + take).collect();
    positions.push(pb);
    let map: Vec<usize> = std::iter::repeat_n(0, take).chain([1]).collect();
    let want: Vec<usize> = (0..take + 1).collect();
    {
        let mut caches = [&mut kv_a, &mut kv_b];
        model
            .forward_rows(
                &mut st,
                &mut peregrine_core::model::Caches::Mapped(&mut caches, &map),
                &pool,
                &tokens,
                &positions,
                &want,
                &mut ss,
            )
            .unwrap();
    }

    let mut failures = Vec::new();
    for (i, want) in alone_a.iter().enumerate() {
        if st.logits_row(i, c.n_vocab) != &want[..] {
            failures.push(format!("A row {i} changed in the mixed step"));
        }
    }
    if st.logits_row(take, c.n_vocab) != &alone_b[..] {
        failures.push("B's decode row changed in the mixed step".into());
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the mixed step changed a user:\n{}",
        failures.join("\n")
    );
}
