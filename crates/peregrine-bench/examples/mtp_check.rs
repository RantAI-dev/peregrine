// validate the MTP block: greedy decode with the main model and count how often the block's draft for the
// next position equals the main model's own pick (the fork measures ~85% greedy at depth 1 on the 9B).
// cargo run --release -p peregrine-bench --example mtp_check -- <model-MTP.gguf> [threads] [tokens]
use peregrine_core::gguf::Gguf;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{State, StepStats};
use peregrine_core::pool::Pool;
fn argmax(l: &[f32]) -> u32 {
    let mut b = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[b] {
            b = i;
        }
    }
    b as u32
}
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let model = peregrine_models::load(Gguf::open(std::path::Path::new(&a[0])).unwrap()).unwrap();
    let threads: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(12);
    let ntok: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(200);
    let c = model.cfg;
    let d = c.d_model;
    let pool = Pool::new(threads);
    let mut st = State::new(&c, 64, 1, 4096, threads, None);
    st.want_hidden = true;
    let mut kv = model.new_cache(KvPrec::F16, 4096);
    let mut ms = model.mtp_state(4096, threads).unwrap();
    let mut ss = StepStats::default();
    for prompt in ["<|im_start|>user\nExplain how photosynthesis works, step by step.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                   "<|im_start|>user\nWrite a Python function that merges two sorted lists.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"] {
        kv.clear(); ms.kv.clear();
        let toks = model.tk.encode(prompt);
        let n = toks.len();
        // prefill all but the last token, with hidden states; MTP catch-up for positions 0..n-1
        model.forward(&mut st, &mut kv, &pool, &toks[..n - 1], 0, &[], &mut ss).unwrap();
        let mut hprev = vec![0.0f32; n * d]; // row q = h_{q-1}; row 0 = zeros
        hprev[d..n * d].copy_from_slice(&st.hidden[..(n - 1) * d]);
        let pos: Vec<usize> = (0..n - 1).collect();
        model.mtp_forward(&mut ms, &pool, &toks[..n - 1], &hprev[..(n - 1) * d], &pos, false, &mut ss).unwrap();
        // decode: feed t at p -> h_p, greedy next; MTP at p+1 with (next, h_p) drafts p+2
        let (mut p, mut t) = (n - 1, toks[n - 1]);
        let mut hlast = hprev[(n - 1) * d..n * d].to_vec(); // h_{n-2}, for the MTP row at n-1
        let (mut hits, mut tries) = (0, 0);
        let mut draft: Option<u32> = None;
        let mut out = Vec::new();
        for _ in 0..ntok {
            // MTP row for position p (token t, h_{p-1}) — the catch-up entry; it also drafts p+1
            let dft = model.mtp_forward(&mut ms, &pool, &[t], &hlast, &[p], true, &mut ss).unwrap();
            if let Some(dd) = draft.take() { let _ = dd; }
            model.forward(&mut st, &mut kv, &pool, &[t], p, &[0], &mut ss).unwrap();
            let next = argmax(st.logits_row(0, c.n_vocab));
            tries += 1;
            if dft == Some(next) { hits += 1; }
            hlast = st.hidden[..d].to_vec();
            out.push(next);
            draft = dft;
            t = next; p += 1;
            if next == 248046 { break; }
        }
        println!("draft == main greedy next: {hits}/{tries} = {:.2}   text: {:?}", hits as f64 / tries as f64, &model.tk.decode(&out)[..80.min(model.tk.decode(&out).len())]);
    }
}
