//! The opt-in fast prompt mode (S5.4b; the per-format table is S5.4f): what
//! `--fast-prompt` guarantees.
//!
//! With the option on, a *prompt* row's weights are read from an 8-bit form built
//! at load (one scale per 256-weight super-block) for the formats whose form's
//! kernel beats this tier's exact kernel (`MatKernels::q8k_uses_form`: Q4_K and
//! Q6_K on AVX-512, Q6_K on AVX2, Q4_K/Q6_K and — since S5.4h — Q5_0 on aarch64's
//! `dotprod`/`i8mm` caps, none on a scalar tier), while a *generated* row,
//! an embedding, a rerank and a decision keep the exact kernels
//! (`docs/design/s5.4-fast-prompt.md` §3). This file pins each part of that:
//!
//! * the same prompt gives the same cache and logits at prefill chunks 16, 32, 64
//!   and 128 and at 1, 2 and 4 threads, alone and beside a decoding user;
//! * decode from a cache a fast-mode prefill produced is the exact engine;
//! * the option is not a no-op: the prompt's logits differ from the exact mode's
//!   for every format the tier's table selects;
//! * the per-weight error bound holds through the engine's own load path;
//! * embeddings, decisions and the prefix store behave as documented.
//!
//! The mode is process-wide in a server, so the library exposes explicit switches for
//! this file to compare the two modes in one process: `peregrine_models::load_with_fast`
//! builds the form whatever the environment says, and `Engine::set_fast_prompt` marks
//! the rows or does not. On a model loaded without the form the mode is a no-op, which
//! is what the default engine sees.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::{KvCache, KvPrec};
use peregrine_core::model::{Caches, Model, StepStats};
use peregrine_core::pool::Pool;
use peregrine_core::serve::{Engine, Finish, SlotCfg, SlotId};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{write_dense, Dense};

/// The formats the tier's table gives an 8-bit form (S5.4f, S5.4h). On AVX2 that is
/// Q6_K alone; on AVX-512 Q4_K and Q6_K; on aarch64 Q4_K and Q6_K on every cap and
/// Q5_0 where the multiply is a NEON body; on a tier with no SIMD body, none, so
/// these tests have nothing to check there (the synthetic goldens assert the
/// no-op).
fn form_formats() -> Vec<Format> {
    [Format::Q4_K, Format::Q6_K, Format::Q5_0]
        .into_iter()
        .filter(|&f| peregrine_kernels::table::MatKernels::get().q8k_uses_form(f))
        .collect()
}

/// The chunk sizes `crates/peregrine-models/tests/prefill_chunk.rs` measures: the
/// mode's bits must not depend on any of them.
const CHUNKS: [usize; 4] = [16, 32, 64, 128];

/// The prompt: longer than the widest chunk, so every size splits it.
fn prompt() -> Vec<u32> {
    (0..200u32).map(|i| (i * 37 + 11) % 300).collect()
}

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-fast-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Load a synthetic file with the resident 8-bit form, whatever the process says.
fn load_fast(path: &Path) -> Model {
    Gguf::open(path)
        .and_then(peregrine_models::load_with_fast)
        .expect("load the synthetic model with the 8-bit form")
}

/// Load the same file the default way (no form): the exact engine.
fn load_exact(path: &Path) -> Model {
    Gguf::open(path)
        .and_then(peregrine_models::load)
        .expect("load the synthetic model")
}

/// The model's KV cache, read back and widened: the bytes the attention will read.
fn cache_floats(kv: &mut KvCache, layers: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut out = Vec::new();
    let mut buf = vec![0.0f32; dim];
    for l in 0..layers {
        for h in 0..heads {
            for pos in 0..prompt().len() {
                kv.decode_block(true, l, h, pos, 1, &mut buf);
                out.extend_from_slice(&buf);
                kv.decode_block(false, l, h, pos, 1, &mut buf);
                out.extend_from_slice(&buf);
            }
        }
    }
    out
}

/// The prompt read in pieces of `chunk` tokens, every row a prompt row: the cache it
/// left and the logits of its last position.
fn read_in_pieces(
    model: &Model,
    chunk: usize,
    threads: usize,
    toks: &[u32],
) -> (Vec<f32>, Vec<f32>) {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let ctx = toks.len() + 1;
    let mut st = model.new_state(1024, 1, ctx, threads);
    let mut kv = model.new_cache(KvPrec::F16, ctx);
    let mut ss = StepStats::default();
    let mut fed = 0usize;
    let mut logits = Vec::new();
    while fed < toks.len() {
        let take = chunk.min(toks.len() - fed);
        let piece = &toks[fed..fed + take];
        let want: Vec<usize> = if fed + take == toks.len() {
            vec![take - 1]
        } else {
            Vec::new()
        };
        let flags = vec![true; take];
        model
            .forward_marked(&mut st, &mut kv, &pool, piece, fed, &want, &flags, &mut ss)
            .unwrap();
        if !want.is_empty() {
            logits = st.logits_row(0, c.n_vocab).to_vec();
        }
        fed += take;
    }
    let cache = cache_floats(&mut kv, c.n_layer, c.n_kv_head, c.head_dim);
    (cache, logits)
}

/// The same prompt, with another user decoding in every pass: the prompt's cache and
/// logits must be the ones it has alone (S5.4b, §3's batch-independence row).
fn read_beside_a_decoder(model: &Model, chunk: usize, threads: usize, toks: &[u32]) -> Vec<f32> {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let mut st = model.new_state(256, 1, toks.len() + 16, threads);
    let mut kv_a = model.new_cache(KvPrec::F16, toks.len() + 16);
    let mut kv_b = model.new_cache(KvPrec::F16, toks.len() + 16);
    let mut ss = StepStats::default();
    // The decoder's own beginning, then one generated row per pass. Its rows are
    // never fast rows; the prompt's are.
    let b_prompt: Vec<u32> = (0..8u32).map(|i| (i * 7 + 3) % 300).collect();
    model
        .forward_marked(
            &mut st,
            &mut kv_b,
            &pool,
            &b_prompt,
            0,
            &[],
            &vec![true; b_prompt.len()],
            &mut ss,
        )
        .unwrap();
    let mut b_next = b_prompt[b_prompt.len() - 1];
    let mut b_pos = b_prompt.len() - 1;
    // The prompt rows are the trailing rows of each pass, the decoder's row leads.
    let mut fed = 0usize;
    while fed < toks.len() {
        let take = chunk.min(toks.len() - fed);
        let mut tokens = Vec::with_capacity(take + 1);
        let mut positions = Vec::with_capacity(take + 1);
        let mut flags = Vec::with_capacity(take + 1);
        tokens.push(b_next);
        positions.push(b_pos);
        flags.push(false);
        for t in 0..take {
            tokens.push(toks[fed + t]);
            positions.push(fed + t);
            flags.push(true);
        }
        // Cache 0 is the decoder's, cache 1 is the prompt's.
        let mut map = Vec::with_capacity(take + 1);
        map.push(0);
        map.extend(std::iter::repeat_n(1, take));
        let mut refs = [&mut kv_b as &mut KvCache, &mut kv_a as &mut KvCache];
        let mut caches = Caches::Mapped(&mut refs, &map);
        model
            .forward_rows_marked(
                &mut st,
                &mut caches,
                &pool,
                &tokens,
                &positions,
                &[0],
                &flags,
                &mut ss,
            )
            .unwrap();
        b_next = (tokens[0] + 7 * (fed as u32 + 1)) % 300;
        b_pos += 1;
        fed += take;
    }
    cache_floats(&mut kv_a, c.n_layer, c.n_kv_head, c.head_dim)
}

/// Greedy tokens from a cache that already holds `start_pos` positions.
fn decode_greedy(
    model: &Model,
    kv: &mut KvCache,
    start: u32,
    start_pos: usize,
    tokens: usize,
    threads: usize,
) -> (Vec<u32>, Vec<f32>) {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let mut st = model.new_state(8, 1, 512, threads);
    let mut ss = StepStats::default();
    let mut next = start;
    let mut out = Vec::with_capacity(tokens);
    let mut last = Vec::new();
    for step in 0..tokens {
        model
            .forward(&mut st, kv, &pool, &[next], start_pos + step, &[0], &mut ss)
            .unwrap();
        last = st.logits_row(0, c.n_vocab).to_vec();
        next = argmax(&last);
        out.push(next);
    }
    (out, last)
}

fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best as u32
}

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

/// The prompt's cache and logits are the same bits at every chunk and thread count,
/// alone and with a decoder in the same passes.
#[test]
fn fast_mode_is_prompt_invariant() {
    let dir = tmpdir("invariant");
    let mut failures = Vec::new();
    for (i, fmt) in form_formats().into_iter().enumerate() {
        let path = write_dense(&dir, "llama", fmt, 9150 + i as u64, Dense::default());
        let model = load_fast(&path);
        if model.q8_form().is_empty() {
            failures.push(format!("llama-{}: no form was built", fmt.label()));
            continue;
        }
        let toks = prompt();
        let (cache0, logits0) = read_in_pieces(&model, CHUNKS[0], 4, &toks);
        for &chunk in &CHUNKS[1..] {
            let (cache, logits) = read_in_pieces(&model, chunk, 4, &toks);
            if cache != cache0 {
                let first = cache.iter().zip(cache0.iter()).position(|(a, b)| a != b);
                failures.push(format!(
                    "llama-{}: the cache differs at chunk {chunk} (first at {first:?})",
                    fmt.label()
                ));
            }
            if logits != logits0 {
                failures.push(format!(
                    "llama-{}: the logits differ at chunk {chunk}",
                    fmt.label()
                ));
            }
        }
        for threads in [1usize, 2] {
            let (cache, logits) = read_in_pieces(&model, CHUNKS[1], threads, &toks);
            if cache != cache0 {
                failures.push(format!(
                    "llama-{}: the cache differs at {threads} threads",
                    fmt.label()
                ));
            }
            if logits != logits0 {
                failures.push(format!(
                    "llama-{}: the logits differ at {threads} threads",
                    fmt.label()
                ));
            }
        }
        // Beside a decoding user, at every chunk: the prompt's rows are still prompt
        // rows, and a row's result depends on its own token and the weights alone.
        for &chunk in &CHUNKS {
            let cache = read_beside_a_decoder(&model, chunk, 4, &toks);
            if cache != cache0 {
                let first = cache.iter().zip(cache0.iter()).position(|(a, b)| a != b);
                failures.push(format!(
                    "llama-{}: the cache differs beside a decoder at chunk {chunk} (first at {first:?})",
                    fmt.label()
                ));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the fast prompt mode is not prompt-invariant:\n{}",
        failures.join("\n")
    );
}

/// The same invariance at 4 096 tokens, where a prompt row's attention takes the
/// tiles past the exact engine's F16-direct crossover and the fast mode's order has
/// many more pieces to agree on (S5.4j). A small model, so the whole staged read is a
/// few seconds; the prompt is longer than any chunk the sweep uses and every row is a
/// prompt row.
#[test]
fn fast_attention_is_prompt_invariant_at_4096_tokens() {
    let dir = tmpdir("invariant-4096");
    let Some(fmt) = form_formats().first().copied() else {
        return;
    };
    let shape = Dense {
        d_model: 128,
        n_layer: 1,
        n_head: 4,
        n_head_kv: 2,
        n_ff: 256,
        ctx: 4096,
        tied: false,
    };
    let path = write_dense(&dir, "llama", fmt, 9600, shape);
    let model = load_fast(&path);
    assert!(!model.q8_form().is_empty(), "no form was built");
    let toks: Vec<u32> = (0..4096u32).map(|i| (i * 37 + 11) % 300).collect();
    let mut failures = Vec::new();
    let (cache0, logits0) = read_in_pieces(&model, 128, 4, &toks);
    for chunk in [16usize, 64, 128, 256] {
        let (cache, logits) = read_in_pieces(&model, chunk, 4, &toks);
        if cache != cache0 {
            let first = cache.iter().zip(cache0.iter()).position(|(a, b)| a != b);
            failures.push(format!(
                "the cache differs at chunk {chunk} (first at {first:?})"
            ));
        }
        if logits != logits0 {
            failures.push(format!("the logits differ at chunk {chunk}"));
        }
    }
    for threads in [1usize, 2, 6] {
        let (cache, logits) = read_in_pieces(&model, 128, threads, &toks);
        if cache != cache0 {
            failures.push(format!("the cache differs at {threads} threads"));
        }
        if logits != logits0 {
            failures.push(format!("the logits differ at {threads} threads"));
        }
    }
    // Beside a decoder, which walks the same tiles through the exact path.
    for &chunk in &[16usize, 128] {
        let cache = read_beside_a_decoder(&model, chunk, 4, &toks);
        if cache != cache0 {
            let first = cache.iter().zip(cache0.iter()).position(|(a, b)| a != b);
            failures.push(format!(
                "the cache differs beside a decoder at chunk {chunk} (first at {first:?})"
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the fast prompt mode is not prompt-invariant at 4 096 tokens:\n{}",
        failures.join("\n")
    );
}

/// A Q5_0 model whose rows are *not* a whole number of 256-weight super-blocks
/// (S5.4h): Qwen 2.5 0.5B's 896-wide shape, where the form pads a fourth
/// super-block. The prompt must still be the same bits at every chunk and thread
/// count and beside a decoder, and the prompt logits must differ from the exact
/// engine's (the form is in use, not silently skipped).
#[test]
fn a_q5_0_row_that_is_not_a_whole_super_block_is_prompt_invariant() {
    if !peregrine_kernels::table::MatKernels::get().q8k_uses_form(Format::Q5_0) {
        return;
    }
    let dir = tmpdir("invariant-padded");
    // 896 = 28 Q5_0 blocks = three and a half super-blocks; the head is 64 wide.
    let shape = Dense {
        d_model: 896,
        n_layer: 2,
        n_head: 14,
        n_head_kv: 2,
        n_ff: 1024,
        ctx: 512,
        tied: false,
    };
    let path = write_dense(&dir, "llama", Format::Q5_0, 9400, shape);
    let model = load_fast(&path);
    assert!(!model.q8_form().is_empty(), "no form was built");
    // Every 896-wide matrix's form is four super-blocks, one of them half padding.
    let t = model
        .tensor("blk.0.attn_q.weight")
        .expect("the synthetic model has the matrix");
    let off = t.q8_off.expect("the 896-wide matrix has a form");
    let cols = t.dims[0] as usize;
    assert_eq!(cols, 896);
    assert_eq!(
        model.tensor("blk.0.ffn_gate.weight").unwrap().dims[0] as usize,
        896
    );
    let ranges = peregrine_core::q8form::layout(&Gguf::open(&path).unwrap());
    let entry = ranges
        .iter()
        .find(|e| e.name == t.name)
        .expect("the layout holds the matrix");
    assert_eq!(
        entry.super_blocks(),
        4,
        "896 weights pad to four super-blocks"
    );
    assert_eq!(off, entry.offset);

    let toks = prompt();
    let mut failures = Vec::new();
    let (cache0, logits0) = read_in_pieces(&model, CHUNKS[0], 4, &toks);
    for &chunk in &CHUNKS[1..] {
        let (cache, logits) = read_in_pieces(&model, chunk, 4, &toks);
        if cache != cache0 {
            failures.push(format!("the cache differs at chunk {chunk}"));
        }
        if logits != logits0 {
            failures.push(format!("the logits differ at chunk {chunk}"));
        }
    }
    for threads in [1usize, 2] {
        let (cache, logits) = read_in_pieces(&model, CHUNKS[1], threads, &toks);
        if cache != cache0 {
            failures.push(format!("the cache differs at {threads} threads"));
        }
        if logits != logits0 {
            failures.push(format!("the logits differ at {threads} threads"));
        }
    }
    for &chunk in &CHUNKS {
        let cache = read_beside_a_decoder(&model, chunk, 4, &toks);
        if cache != cache0 {
            failures.push(format!(
                "the cache differs beside a decoder at chunk {chunk}"
            ));
        }
    }
    let exact = load_exact(&path);
    let (_, logits_exact) = read_in_pieces(&exact, CHUNKS[2], 4, &toks);
    if logits0 == logits_exact {
        failures.push("the fast mode gave the exact mode's logits".to_string());
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "a padded Q5_0 row is not prompt-invariant:\n{}",
        failures.join("\n")
    );
}

/// Starting from the cache a fast-mode prefill produced, decode is the exact engine:
/// the two models' generated tokens and logits are the same bits.
#[test]
fn fast_mode_decode_is_the_exact_engine() {
    let dir = tmpdir("decode");
    let mut failures = Vec::new();
    for (i, fmt) in form_formats().into_iter().enumerate() {
        let path = write_dense(&dir, "llama", fmt, 9200 + i as u64, Dense::default());
        let fast = load_fast(&path);
        let exact = load_exact(&path);
        let toks = prompt();
        let c = fast.cfg;
        let pool = Pool::new(4);
        // The same fast prefill twice: one cache for each continuation. The two
        // caches are compared too, so a difference cannot hide in non-determinism.
        let cache = |model: &Model| {
            let mut st = model.new_state(256, 1, 512, 4);
            let mut kv = model.new_cache(KvPrec::F16, 512);
            let mut ss = StepStats::default();
            let n = toks.len() - 1;
            model
                .forward_marked(
                    &mut st,
                    &mut kv,
                    &pool,
                    &toks[..n],
                    0,
                    &[n - 1],
                    &vec![true; n],
                    &mut ss,
                )
                .unwrap();
            (kv, st.logits_row(0, c.n_vocab).to_vec())
        };
        let (mut kv_fast, logits_fast) = cache(&fast);
        let (mut kv_copy, logits_copy) = cache(&fast);
        if logits_fast != logits_copy {
            failures.push(format!("llama-{}: two fast prefills differ", fmt.label()));
        }
        let n = toks.len();
        let first = argmax(&logits_fast);
        let (a, last_a) = decode_greedy(&fast, &mut kv_fast, first, n, 12, 4);
        let (b, last_b) = decode_greedy(&exact, &mut kv_copy, first, n, 12, 4);
        if a != b || bits(&last_a) != bits(&last_b) {
            failures.push(format!(
                "llama-{}: decode from the fast cache is not the exact engine ({a:?} vs {b:?})",
                fmt.label()
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "fast-mode decode is not exact:\n{}",
        failures.join("\n")
    );
}

/// The option is not a no-op: the prompt's logits are not the exact engine's bits.
#[test]
fn fast_mode_differs_from_exact_mode_on_prompt_rows() {
    let dir = tmpdir("differs");
    let mut failures = Vec::new();
    for (i, fmt) in form_formats().into_iter().enumerate() {
        let path = write_dense(&dir, "llama", fmt, 9250 + i as u64, Dense::default());
        let fast = load_fast(&path);
        let exact = load_exact(&path);
        let toks = prompt();
        let (_, logits_fast) = read_in_pieces(&fast, CHUNKS[2], 4, &toks);
        let (_, logits_exact) = read_in_pieces(&exact, CHUNKS[2], 4, &toks);
        if logits_fast == logits_exact {
            failures.push(format!(
                "llama-{}: the fast mode gave the exact mode's logits",
                fmt.label()
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the fast prompt mode did nothing:\n{}",
        failures.join("\n")
    );
}

/// The per-weight error bound (`max|w| / 254`, the kernel test's) holds end to end,
/// through the engine's own load path.
#[test]
fn every_rerounded_weight_is_within_half_a_step_after_loading() {
    use peregrine_kernels::kq8::{code_at, scale_at, Q8K_SIZE};
    let dir = tmpdir("bound");
    let mut failures = Vec::new();
    let mut checked = 0usize;
    let bound_formats: Vec<Format> = form_formats()
        .into_iter()
        .filter(|f| matches!(f, Format::Q4_K | Format::Q6_K))
        .collect();
    if bound_formats.is_empty() {
        return;
    }
    for (i, fmt) in bound_formats.into_iter().enumerate() {
        let path = write_dense(&dir, "llama", fmt, 9300 + i as u64, Dense::default());
        let model = load_fast(&path);
        let t = model
            .tensor("blk.0.ffn_gate.weight")
            .expect("the synthetic model has the matrix");
        let off = t
            .q8_off
            .unwrap_or_else(|| panic!("llama-{}: no form offset", fmt.label()));
        let cols = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        let n_sb = cols / 256;
        let rb = peregrine_core::quant::row_bytes(t.kind, cols);
        let blk = t.kind.block_bytes();
        let file = model.tensor_bytes(t);
        let form = model.q8_form();
        for gr in 0..rows / 8 {
            for sb in 0..n_sb {
                let mut blocks: [&[u8]; 8] = [&[]; 8];
                for (r, b) in blocks.iter_mut().enumerate() {
                    let at = (gr * 8 + r) * rb + sb * blk;
                    *b = &file[at..at + blk];
                }
                let mut tile = vec![
                    0u8;
                    peregrine_kernels::kq::Q4K8::SIZE
                        .max(peregrine_kernels::kq::Q6K8::SIZE,)
                ];
                match t.kind {
                    Format::Q4_K => peregrine_kernels::kq::repack_q4k8(&blocks, &mut tile),
                    _ => peregrine_kernels::kq::repack_q6k8(&blocks, &mut tile),
                }
                let at = off + (gr * n_sb + sb) * Q8K_SIZE;
                let one = &form[at..at + Q8K_SIZE];
                for r in 0..8 {
                    let w = match t.kind {
                        Format::Q4_K => peregrine_kernels::kq8::q4k_weights(&tile, r),
                        _ => {
                            let p = peregrine_kernels::kq8::q6k_products(&tile, r);
                            let d = peregrine_kernels::f16(u16::from_le_bytes([
                                tile[peregrine_kernels::kq::q6_d() + r * 2],
                                tile[peregrine_kernels::kq::q6_d() + r * 2 + 1],
                            ]));
                            // The form's reconstruction is `scale · code`; the exact
                            // weight is what the products with the block scale say.
                            let _ = d;
                            let mut w = [0.0f32; 256];
                            for (k, &v) in p.iter().enumerate() {
                                w[k] = v as f32 * d;
                            }
                            w
                        }
                    };
                    let mut m = 0.0f32;
                    for &x in w.iter() {
                        m = m.max(x.abs());
                    }
                    let d = f32::from_le_bytes(
                        one[scale_at(0, r)..scale_at(0, r) + 4].try_into().unwrap(),
                    );
                    for k in 0..256 {
                        let code = one[code_at(0, k / 8, r, k % 8)] as i32 - 128;
                        let got = d * code as f32;
                        let err = (got - w[k]).abs();
                        if err > m / 254.0 * (1.0 + 1e-5) + 1e-7 {
                            failures.push(format!(
                                "llama-{}: weight {k} of run {gr} block {sb} row {r} is {err} off (bound {})",
                                fmt.label(),
                                m / 254.0
                            ));
                            break;
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the resident form is outside its per-weight bound:\n{}",
        failures.join("\n")
    );
    assert!(checked > 100_000, "only {checked} weights were checked");
}

/// Embeds, reranks and decisions are bit-identical with the option on and off: their
/// rows are never fast rows, even when the model carries a form.
#[test]
fn embeddings_and_decisions_are_untouched() {
    use peregrine_core::embed::Pooling;
    let dir = tmpdir("pooled");
    let mut failures = Vec::new();
    for (i, fmt) in form_formats().into_iter().enumerate() {
        let path = write_dense(&dir, "llama", fmt, 9400 + i as u64, Dense::default());
        let model = Arc::new(load_fast(&path));
        let text: Vec<u32> = (0..24u32).map(|i| (i * 13 + 5) % 300).collect();
        let labels = vec![11u32, 12, 13];
        let run = |fast: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut e = Engine::new(model.clone(), 2, 512, 1, KvPrec::F16, 0);
            e.set_fast_prompt(fast);
            let cfg = SlotCfg {
                max_tokens: 1,
                ..Default::default()
            };
            let id = e
                .admit_pooled(text.clone(), cfg.clone(), Pooling::Mean)
                .unwrap();
            let mut pooled = Vec::new();
            while e.active() > 0 {
                for em in e.step() {
                    if em.slot == id {
                        pooled = em.embedding.clone().unwrap_or_default();
                    }
                }
            }
            let mut e = Engine::new(model.clone(), 2, 512, 1, KvPrec::F16, 0);
            e.set_fast_prompt(fast);
            let id = e
                .admit_labels(text.clone(), cfg.clone(), labels.clone())
                .unwrap();
            let mut decision = Vec::new();
            while e.active() > 0 {
                for em in e.step() {
                    if em.slot == id {
                        decision = em.label_logits.clone().unwrap_or_default();
                    }
                }
            }
            (pooled, decision, Vec::new())
        };
        let on = run(true);
        let off = run(false);
        if bits(&on.0) != bits(&off.0) {
            failures.push(format!("llama-{}: the embedding moved", fmt.label()));
        }
        if bits(&on.1) != bits(&off.1) {
            failures.push(format!("llama-{}: the decision moved", fmt.label()));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the fast mode changed a non-prompt read-out:\n{}",
        failures.join("\n")
    );
}

/// Two slots, one prefilling and one decoding in the same passes: each slot's tokens
/// are the tokens it gets alone in the same mode.
#[test]
fn a_multi_user_pass_serves_each_slot_alone() {
    let dir = tmpdir("multi");
    let Some(fmt) = form_formats().first().copied() else {
        return;
    };
    let path = write_dense(&dir, "llama", fmt, 9450, Dense::default());
    let model = Arc::new(load_fast(&path));
    let long: Vec<u32> = (0..200u32).map(|i| (i * 37 + 11) % 300).collect();
    let short: Vec<u32> = (0..8u32).map(|i| (i * 7 + 3) % 300).collect();
    type Run = Vec<(SlotId, u32)>;
    let run = |two: bool| -> Run {
        let mut e = Engine::new(model.clone(), 4, 512, 2, KvPrec::F16, 0);
        e.set_fast_prompt(true);
        let long_cfg = SlotCfg {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let short_cfg = SlotCfg {
            temperature: 0.0,
            max_tokens: 6,
            ..Default::default()
        };
        let mut out = Vec::new();
        if two {
            // The short one starts first, so it decodes while the long one is read.
            e.admit(short.clone(), short_cfg).unwrap();
            e.admit(long.clone(), long_cfg).unwrap();
        } else {
            let a = e.admit(short.clone(), short_cfg).unwrap();
            while e.active() > 0 {
                for em in e.step() {
                    out.push((a, em.token));
                }
            }
            let b = e.admit(long.clone(), long_cfg).unwrap();
            while e.active() > 0 {
                for em in e.step() {
                    out.push((b, em.token));
                }
            }
            return out;
        }
        while e.active() > 0 {
            for em in e.step() {
                if let Some(Finish::Error(err)) = &em.finish {
                    panic!("{err}");
                }
                out.push((em.slot, em.token));
            }
        }
        out
    };
    let together = run(true);
    let alone = run(false);
    let per_slot = |t: &[(SlotId, u32)]| -> Vec<Vec<u32>> {
        let mut ids: Vec<SlotId> = t.iter().map(|&(s, _)| s).collect();
        ids.sort();
        ids.dedup();
        ids.into_iter()
            .map(|id| {
                t.iter()
                    .filter(|&&(s, _)| s == id)
                    .map(|&(_, tok)| tok)
                    .collect()
            })
            .collect()
    };
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        per_slot(&together),
        per_slot(&alone),
        "a slot served beside another was served differently"
    );
}

/// The prefix store: a prompt restored from the store gives the same tokens as the
/// same prompt read again (both are prompt rows). The documented exception — a
/// continued conversation whose earlier reply was *generated* — is pinned separately.
#[test]
fn a_restored_prefix_gives_the_prompt_read_again() {
    let dir = tmpdir("prefix");
    let Some(fmt) = form_formats().first().copied() else {
        return;
    };
    let path = write_dense(&dir, "llama", fmt, 9500, Dense::default());
    let model = Arc::new(load_fast(&path));
    let opening: Vec<u32> = (0..40u32).map(|i| (i * 13) % 900 % 300).collect();
    let run = |engine: &mut Engine, tail: u32| -> Vec<u32> {
        let mut p = opening.clone();
        p.push(tail);
        let id = engine
            .admit(
                p,
                SlotCfg {
                    temperature: 0.0,
                    max_tokens: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut out = Vec::new();
        while engine.active() > 0 {
            for em in engine.step() {
                if em.slot == id {
                    out.push(em.token);
                }
            }
        }
        out
    };
    let mut warm = Engine::new(model.clone(), 3, 256, 1, KvPrec::F16, 1 << 26);
    warm.set_fast_prompt(true);
    run(&mut warm, 118);
    let restored = run(&mut warm, 204);
    let (entries, _, hits, _) = warm.prefix_stats();
    assert!(
        entries > 0 && hits > 0,
        "the second request did not restore a prefix"
    );
    let mut cold = Engine::new(model, 3, 256, 1, KvPrec::F16, 0);
    cold.set_fast_prompt(true);
    assert_eq!(
        run(&mut cold, 204),
        restored,
        "a restored prefix changed the fast mode's answer"
    );
}

/// The documented exception (design §3, last row): a token that was *generated* has an
/// exact cache entry, and the same token read again as *prompt* gets the 8-bit one. The
/// two caches differ, by construction. A test cannot ask for them to agree; it pins
/// which case is which.
#[test]
fn a_generated_tokens_cache_entry_differs_from_the_same_token_read_as_prompt() {
    let dir = tmpdir("exception");
    let Some(fmt) = form_formats().first().copied() else {
        return;
    };
    let path = write_dense(&dir, "llama", fmt, 9550, Dense::default());
    let model = load_fast(&path);
    let c = model.cfg;
    let pool = Pool::new(2);
    let opening: Vec<u32> = (0..16u32).map(|i| (i * 17 + 3) % 300).collect();
    let reply: Vec<u32> = (0..8u32).map(|i| (i * 29 + 7) % 300).collect();
    let mut st = model.new_state(64, 1, 512, 2);
    let mut ss = StepStats::default();
    // Case 1: the opening prefilled as prompt rows, then the reply fed as generated
    // (decode) rows — the exact kernels — into the same cache.
    let mut kv_gen = model.new_cache(KvPrec::F16, 512);
    model
        .forward_marked(
            &mut st,
            &mut kv_gen,
            &pool,
            &opening,
            0,
            &[],
            &vec![true; opening.len()],
            &mut ss,
        )
        .unwrap();
    for (t, &tok) in reply.iter().enumerate() {
        model
            .forward(
                &mut st,
                &mut kv_gen,
                &pool,
                &[tok],
                opening.len() + t,
                &[],
                &mut ss,
            )
            .unwrap();
    }
    // Case 2: the whole transcript read as one prompt, in fast mode.
    let mut kv_prompt = model.new_cache(KvPrec::F16, 512);
    let all: Vec<u32> = opening.iter().chain(reply.iter()).copied().collect();
    model
        .forward_marked(
            &mut st,
            &mut kv_prompt,
            &pool,
            &all,
            0,
            &[],
            &vec![true; all.len()],
            &mut ss,
        )
        .unwrap();
    let gen = cache_upto(&mut kv_gen, c.n_layer, c.n_kv_head, c.head_dim, all.len());
    let prompt = cache_upto(
        &mut kv_prompt,
        c.n_layer,
        c.n_kv_head,
        c.head_dim,
        all.len(),
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert_ne!(
        gen, prompt,
        "the documented exception did not happen: generated and prompt rows of the same \
         token gave the same cache, so the test is not pinning the case the design documents"
    );
}

/// The cache of the first `n` positions, read back and widened.
fn cache_upto(kv: &mut KvCache, layers: usize, heads: usize, dim: usize, n: usize) -> Vec<f32> {
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
