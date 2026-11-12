//! The DSpark head's draft count, pinned (`PEREGRINE_DSPARK_K`).
//!
//! The switch pins the number of draft tokens a round verifies, whatever the
//! confidences say: it is how the head's depth is measured. The session reads
//! it, and so does the engine, through the drafter's policy.
//!
//! The options are read once per process, so this lives in a binary of its
//! own: the switch is set before anything reads them.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::KvPrec;
use peregrine_core::serve::{DrafterChoice, Engine, SlotCfg};
use peregrine_core::session::{Session, SessionCfg};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{write_dense, Dense};

/// The prompt both paths are given.
const USER: &str = "Name three colours.";
/// The reply both paths are asked for.
const TOKENS: usize = 24;
/// The seed both paths sample with.
const SEED: u64 = 20240;
/// The temperature both paths sample at.
const TEMP: f32 = 0.7;
/// The nucleus both paths sample from.
const TOP_P: f32 = 0.95;
/// The number of draft tokens the pin holds the head to.
const PINNED_K: usize = 2;

fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-dspark-k-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &Path) -> Arc<peregrine_core::model::Model> {
    Arc::new(Gguf::open(path).and_then(peregrine_models::load).unwrap())
}

/// The token that ends the turn is the end of the answer, not part of it: the
/// engine reports it, the session does not.
fn drop_stop(out: &mut Vec<u32>, stop: &[u32]) {
    if stop.contains(&out.last().copied().unwrap_or(u32::MAX)) {
        out.pop();
    }
}

/// **The pin holds the head to its depth, and changes no token.**
///
/// With `PEREGRINE_DSPARK_K` set, every round verifies exactly that many
/// draft tokens — the price does not get a say. The tokens are the session's,
/// and the engine's without a drafter: the pin changes how much is verified,
/// never what is said.
#[test]
fn the_pinned_draft_count_holds_the_head_to_its_depth() {
    // The switch, before anything reads the options.
    std::env::set_var("PEREGRINE_DSPARK_K", PINNED_K.to_string());
    assert_eq!(
        peregrine_core::options::get().dspark.k,
        Some(PINNED_K),
        "the switch was not read"
    );

    let dir = tmpdir("pinned");
    let path = write_dense(&dir, "qwen3", Format::Q8_0, 7910, Dense::default());
    let model = load(&path);
    let drafter = support::dspark::write_drafter(
        &dir,
        support::dspark::DrafterShape {
            n_vocab: model.cfg.n_vocab,
            hidden: model.cfg.d_model,
            ..Default::default()
        },
        7911,
    );
    let prompt = model.tk.encode_prompt(&peregrine_core::serve::chat_prompt(
        peregrine_core::template::Template::detect(model.gguf()).expect("a template"),
        None,
        &[],
        USER,
        true,
        None,
    ));

    // The session, with the head attached and the pin on.
    let mut scfg = SessionCfg {
        max_ctx: 512,
        temperature: TEMP,
        top_p: TOP_P,
        top_k: 0,
        min_p: 0.0,
        seed: SEED,
        max_tokens: TOKENS,
        threads: 4,
        speculate: true,
        thinking: true,
        dspark: Some(drafter.clone()),
        dspark_max: DrafterChoice::DSPARK_MAX_K,
        ..Default::default()
    };
    scfg.dspark_max = DrafterChoice::DSPARK_MAX_K;
    let mut session = Session::from_model(model.clone(), scfg).expect("session");
    let mut s = Vec::new();
    session.generate(USER, |e| s.push(e.id)).expect("generate");

    // The engine, with a head of its own and the pin on.
    let stop = peregrine_core::template::Template::detect(model.gguf())
        .expect("a template")
        .stop_ids(&model.tk);
    let cfg = SlotCfg {
        temperature: TEMP,
        top_p: TOP_P,
        top_k: 0,
        min_p: 0.0,
        seed: SEED,
        max_tokens: TOKENS,
        stop,
        logprobs: false,
        penalties: Default::default(),
        turn: 1,
        kv_rotation: model.session_kv_rotation(),
        drafter: DrafterChoice::Dspark {
            dir: drafter.clone(),
            max_k: DrafterChoice::DSPARK_MAX_K,
        },
        constraint: None,
    };
    let mut e = Engine::new(model.clone(), 4, 512, 1, KvPrec::F16, 0);
    let id = e.admit(prompt.clone(), cfg.clone()).expect("admit");
    let mut out = Vec::new();
    while e.active() > 0 {
        for em in e.step() {
            assert_eq!(em.slot, id);
            out.push(em.token);
        }
    }
    drop_stop(&mut out, &cfg.stop);
    let drafts = e.spec_drafts;
    let tokens = e.spec_tokens;

    assert!(drafts > 0, "the engine never asked the head for a draft");
    assert!(
        tokens <= PINNED_K as u64 * drafts,
        "the pin did not hold: {tokens} draft tokens over {drafts} rounds"
    );
    assert_eq!(out, s, "the pinned run changed the tokens");

    // The engine without a drafter: the same tokens.
    let mut plain = cfg.clone();
    plain.drafter = DrafterChoice::None;
    let mut e = Engine::new(model.clone(), 4, 512, 1, KvPrec::F16, 0);
    let id = e.admit(prompt.clone(), plain.clone()).expect("admit");
    let mut without = Vec::new();
    while e.active() > 0 {
        for em in e.step() {
            assert_eq!(em.slot, id);
            without.push(em.token);
        }
    }
    drop_stop(&mut without, &plain.stop);
    assert_eq!(without, out, "the pinned run changed the tokens");

    let _ = std::fs::remove_dir_all(&dir);
}
