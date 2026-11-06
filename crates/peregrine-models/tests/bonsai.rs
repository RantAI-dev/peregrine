// Bonsai 2 9B against PrismML's llama.cpp fork (benchmarks/bonsai2_9b_reference.json): tokenisation,
// the first step's top five, and the greedy continuation. Skips when the model file is absent (as the other
// model tests do). PEREGRINE_BONSAI9 overrides the path.
use peregrine_core::gguf::Gguf;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{State, StepStats};
use peregrine_core::pool::Pool;
use std::path::PathBuf;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("PEREGRINE_BONSAI9")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/bonsai2-9b/Ternary-Bonsai-2-9B-PQ2_0.gguf")
        });
    p.exists().then_some(p)
}

fn probs(l: &[f32]) -> Vec<f64> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = l.iter().map(|&x| ((x - m) as f64).exp()).sum();
    l.iter().map(|&x| ((x - m) as f64).exp() / z).collect()
}

#[test]
fn bonsai2_9b_matches_the_reference() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no Bonsai 2 9B file");
        return;
    };
    let model = peregrine_models::load(Gguf::open(&path).unwrap()).unwrap();
    let c = model.cfg;
    let toks = model.tk.encode("The capital of France is");
    assert_eq!(toks, vec![760, 6511, 314, 9338, 369]);
    let threads = 8;
    let pool = Pool::new(threads);
    let mut st = State::new(&c, 64, 1, 512, threads, None);
    let mut kv = model.new_cache(KvPrec::F16, 512);
    let mut ss = StepStats::default();
    model
        .forward(
            &mut st,
            &mut kv,
            &pool,
            &toks,
            0,
            &[toks.len() - 1],
            &mut ss,
        )
        .unwrap();
    // Reference step 0: " Paris" 0.5883, " not" 0.0650, " a" 0.0458, " in" 0.0270, " located" 0.0205.
    let p = probs(st.logits_row(0, c.n_vocab));
    assert!(
        (p[11751] - 0.5883).abs() < 0.02,
        "P(' Paris') = {}",
        p[11751]
    );
    // Greedy: " Paris.\nThe capital of France is" (the reference continues with a near-tie at step 8).
    let want = [11751u32, 13, 198, 760, 6511, 314, 9338, 369];
    let mut got = Vec::new();
    for pos in (toks.len()..).take(want.len()) {
        let l = st.logits_row(0, c.n_vocab);
        let next = (0..l.len())
            .max_by(|&a, &b| l[a].partial_cmp(&l[b]).unwrap())
            .unwrap() as u32;
        got.push(next);
        model
            .forward(&mut st, &mut kv, &pool, &[next], pos, &[0], &mut ss)
            .unwrap();
    }
    assert_eq!(got, want);
}

// MTP speculation through the serving engine is lossless: a lone user's tokens are the same with it and
// without it, greedy and sampled. Uses the 4B's MTP file (PEREGRINE_BONSAI4_MTP overrides); skips without it.
#[test]
fn mtp_speculation_changes_no_token() {
    use peregrine_core::serve::{Engine, SlotCfg};
    use std::sync::Arc;
    let path = std::env::var("PEREGRINE_BONSAI4_MTP")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/bonsai2-4b/Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf")
        });
    if !path.exists() {
        eprintln!("skipped: no Bonsai 2 4B MTP file");
        return;
    }
    let model = Arc::new(peregrine_models::load(Gguf::open(&path).unwrap()).unwrap());
    assert!(model.has_mtp());
    let prompt = model.tk.encode(
        "<|im_start|>user\nWrite a Python function that merges two sorted lists.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
    let run = |temperature: f32, spec: bool| -> Vec<u32> {
        let mut e = Engine::new(model.clone(), 8, 1024, 2, KvPrec::F16, 0);
        if !spec {
            e.disable_mtp();
        }
        assert_eq!(e.speculates(), spec || !model.has_mtp());
        let cfg = SlotCfg {
            temperature,
            top_p: 0.8,
            top_k: 20,
            min_p: 0.0,
            seed: 5,
            max_tokens: 64,
            ..Default::default()
        };
        e.admit(prompt.clone(), cfg).unwrap();
        let mut toks = vec![];
        while e.active() > 0 {
            for em in e.step() {
                assert!(!matches!(
                    em.finish,
                    Some(peregrine_core::serve::Finish::Error(_))
                ));
                toks.push(em.token);
            }
        }
        toks
    };
    for t in [0.0, 0.8] {
        let (a, b) = (run(t, true), run(t, false));
        assert_eq!(a.len(), 64);
        assert_eq!(a, b, "temperature {t}");
    }
}
