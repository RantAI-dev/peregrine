// lossless MTP speculation, end to end: decode with and without it and require the same tokens.
// cargo run --release --example spec_check -- <model-MTP.gguf> [threads] [tokens] [temperature]
use peregrine_core::gguf::Gguf;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Caches, Model, State, StepStats};
use peregrine_core::pool::Pool;
use peregrine_core::sample::{position_key, sample_filtered, Scratch};
use std::time::Instant;

struct Run {
    toks: Vec<u32>,
    secs: f64,
    drafts: usize,
    hits: usize,
}

fn sample(l: &[f32], t: f32, seed: u64, pos: usize, sc: &mut Scratch) -> u32 {
    sample_filtered(l, t, 0.8, 20, 0.0, position_key(seed, 0, pos), sc)
}

fn decode(
    model: &Model,
    pool: &Pool,
    pool1: &Pool,
    prompt: &[u32],
    n: usize,
    t: f32,
    spec: bool,
) -> Run {
    let c = model.cfg;
    let d = c.d_model;
    let threads = pool.width();
    let mut st = State::new(&c, 64, 2, 4096, threads, None);
    st.want_hidden = true;
    let mut kv = model.new_cache(KvPrec::F16, 4096);
    let mut ms = model.mtp_state(4096, threads).unwrap();
    let mut ss = StepStats::default();
    let mut sc = Scratch::default();
    let seed = 7u64;
    let np = prompt.len();
    model
        .forward(&mut st, &mut kv, pool, &prompt[..np - 1], 0, &[], &mut ss)
        .unwrap();
    let mut hrows = vec![0.0f32; np * d];
    hrows[d..np * d].copy_from_slice(&st.hidden[..(np - 1) * d]);
    let pos0: Vec<usize> = (0..np - 1).collect();
    if spec {
        model
            .mtp_forward(
                &mut ms,
                pool,
                &prompt[..np - 1],
                &hrows[..(np - 1) * d],
                &pos0,
                false,
                &mut ss,
            )
            .unwrap();
    }
    let mut hprev = hrows[(np - 1) * d..np * d].to_vec(); // h_{p-1}
    let (mut p, mut tok) = (np - 1, prompt[np - 1]);
    let mut pending: Option<(u32, Vec<f32>)> = None; // an MTP catch-up row owed after an accepted draft
    let mut out = Vec::new();
    let (mut drafts, mut hits) = (0, 0);
    let t0 = Instant::now();
    while out.len() < n {
        if !spec {
            model
                .forward(&mut st, &mut kv, pool1, &[tok], p, &[0], &mut ss)
                .unwrap();
            let s = sample(st.logits_row(0, c.n_vocab), t, seed, p + 1, &mut sc);
            out.push(s);
            tok = s;
            p += 1;
            continue;
        }
        // MTP: (catch-up row,) then the row for p, which drafts p+1.
        let mut mt = vec![];
        let mut mh = vec![];
        let mut mp = vec![];
        if let Some((pt, ph)) = pending.take() {
            mt.push(pt);
            mh.extend(ph);
            mp.push(p - 1);
        }
        mt.push(tok);
        mh.extend_from_slice(&hprev);
        mp.push(p);
        let draft = model
            .mtp_forward(&mut ms, pool, &mt, &mh, &mp, true, &mut ss)
            .unwrap()
            .unwrap();
        drafts += 1;
        // Verify [tok at p, draft at p+1], keeping the state after p.
        kv.request_snapshot(p);
        {
            let mut refs = [&mut kv];
            model
                .forward_rows(
                    &mut st,
                    &mut Caches::Mapped(&mut refs, &[0, 0]),
                    pool,
                    &[tok, draft],
                    &[p, p + 1],
                    &[0, 1],
                    &mut ss,
                )
                .unwrap();
        }
        let s = sample(st.logits_row(0, c.n_vocab), t, seed, p + 1, &mut sc);
        let h0 = st.hidden[..d].to_vec();
        if s == draft {
            hits += 1;
            kv.drop_snapshot();
            let u = sample(st.logits_row(1, c.n_vocab), t, seed, p + 2, &mut sc);
            out.push(s);
            out.push(u);
            pending = Some((s, h0));
            hprev = st.hidden[d..2 * d].to_vec();
            tok = u;
            p += 2;
        } else {
            kv.truncate(p + 1);
            out.push(s);
            hprev = h0;
            tok = s;
            p += 1;
        }
    }
    out.truncate(n);
    Run {
        toks: out,
        secs: t0.elapsed().as_secs_f64(),
        drafts,
        hits,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let model = Model::load(Gguf::open(std::path::Path::new(&a[0])).unwrap()).unwrap();
    let threads: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(16);
    let n: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(128);
    let t: f32 = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let pool = Pool::new(threads);
    let pool1 = peregrine_core::pool::narrow_pool(threads).unwrap_or_else(|| Pool::new(threads));
    for q in [
        "Explain how photosynthesis works, step by step.",
        "Write a Python function that merges two sorted lists.",
        "Summarize the causes of the French revolution.",
    ] {
        let prompt = model.tk.encode(&format!(
            "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        let base = decode(&model, &pool, &pool1, &prompt, n, t, false);
        let spec = decode(&model, &pool, &pool1, &prompt, n, t, true);
        let same = base.toks == spec.toks;
        let first_diff = base.toks.iter().zip(&spec.toks).position(|(x, y)| x != y);
        println!("t={t} identical={same} (first diff {first_diff:?})  plain {:.1} tok/s  spec {:.1} tok/s ({:.2}x)  accepted {}/{} = {:.2}",
            n as f64 / base.secs, n as f64 / spec.secs, base.secs / spec.secs, spec.hits, spec.drafts, spec.hits as f64 / spec.drafts as f64);
    }
}
