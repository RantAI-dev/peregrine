//! Today’s behaviour of the two runtimes, pinned before ADR 0007 unifies them.
//!
//! There are two runtimes. `session::Session` is one conversation: chat history,
//! template, prefix reuse inside the conversation, speculation with an adaptive
//! draft length. `serve::Engine` is the server: slots, batching, mixed
//! prefill/decode, MTP speculation, the cross-request prefix store.
//!
//! ADR 0007 makes the engine the one runtime and rebuilds the session as a thin
//! layer over a one-slot engine. Before that move, the two differences that change
//! output are pinned here, on the synthetic models (no model files needed):
//!
//! * the sampling key — the session counts conversation turns into it, the engine
//!   does not;
//! * the taps a DSpark drafter needs, which must not change a single logit bit;
//! * a second user arriving while the first one speculates;
//! * the prefix store, which is off while a drafter is attached.
//!
//! Every test names the ADR 0007 consequence it pins.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, StepStats};
use peregrine_core::pool::Pool;
use peregrine_core::sample::{position_key, sample_filtered, Scratch};
use peregrine_core::serve::{chat_prompt, Engine, Finish, SlotCfg};
use peregrine_core::session::{Session, SessionCfg};
use peregrine_core::template::Template;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{write_dense, write_hybrid, Dense};

/// The prompt both runtimes are given. Short enough to be one prefill chunk.
const USER: &str = "Name three colours.";
/// The reply both runtimes are asked for.
const TOKENS: usize = 24;
/// The seed both runtimes sample with.
const SEED: u64 = 20240;
/// The temperature at which the two sampling keys are compared.
const TEMP: f32 = 0.7;
/// The nucleus both runtimes sample from.
const TOP_P: f32 = 0.95;

/// A directory of its own for each test (tests run in parallel and each removes its directory).
fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-parity-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &Path) -> Arc<Model> {
    Arc::new(Gguf::open(path).and_then(peregrine_models::load).unwrap())
}

/// The template the file declares — the one the session detects and renders with.
fn template(model: &Model) -> Template {
    Template::detect(model.gguf()).expect("the synthetic models declare a template")
}

/// The system message in force, as the session computes it: the caller's, else the
/// one the model's own template would insert.
fn system(model: &Model) -> Option<String> {
    template(model)
        .default_system(model.name())
        .map(str::to_string)
}

/// The prompt the session builds for `user`, as token ids.
///
/// The session renders the chat template and encodes it (`Session::plan_prompt`);
/// the engine is given token ids. This is the same string through the same
/// template, so the two runtimes see the same prompt.
fn prompt_ids(model: &Model, user: &str) -> Vec<u32> {
    let text = chat_prompt(
        template(model),
        system(model).as_deref(),
        &[],
        user,
        true,
        None,
    );
    model.tk.encode_prompt(&text)
}

/// The token ids a session produces for one turn.
fn session_tokens(session: &mut Session, user: &str) -> Vec<u32> {
    let mut out = Vec::new();
    session
        .generate(user, |e| out.push(e.id))
        .expect("generate");
    out
}

/// The token ids a one-slot engine produces for one prompt.
///
/// The engine's stop list is the template's own, so both runtimes end the turn on
/// the same token; the engine reports that token, the session does not (it is the
/// turn's end, not part of the answer), so it is dropped here.
fn engine_tokens(model: &Arc<Model>, prompt: &[u32], cfg: &SlotCfg) -> Vec<u32> {
    engine_tokens_with(model, prompt, cfg, KvPrec::F16)
}

/// [`engine_tokens`] with the engine's KV cache in the given precision.
fn engine_tokens_with(model: &Arc<Model>, prompt: &[u32], cfg: &SlotCfg, kv: KvPrec) -> Vec<u32> {
    let mut e = Engine::new(model.clone(), 4, 512, 1, kv, 0);
    let id = e.admit(prompt.to_vec(), cfg.clone()).expect("admit");
    let mut out = Vec::new();
    while e.active() > 0 {
        for em in e.step() {
            if let Some(Finish::Error(err)) = em.finish {
                panic!("engine: {err}");
            }
            assert_eq!(em.slot, id);
            out.push(em.token);
        }
    }
    drop_stop(&mut out, &cfg.stop);
    out
}

/// The token that ends the turn is the end of the answer, not part of it: the
/// engine reports it, the session does not.
fn drop_stop(out: &mut Vec<u32>, stop: &[u32]) {
    if stop.contains(&out.last().copied().unwrap_or(u32::MAX)) {
        out.pop();
    }
}

/// The engine's request settings, matching the session's sampling controls.
///
/// The stop list is the template's own, so both runtimes end the turn on the same
/// token.
fn slot_cfg(model: &Model, temperature: f32) -> SlotCfg {
    SlotCfg {
        temperature,
        top_p: TOP_P,
        top_k: 0,
        min_p: 0.0,
        seed: SEED,
        max_tokens: TOKENS,
        stop: template(model).stop_ids(&model.tk),
        logprobs: false,
        penalties: Default::default(),
        // The engine's own defaults: no conversation turn in the sampling key,
        // and the engine's own KV cache.
        turn: 0,
        kv_rotation: true,
    }
}

/// The session's settings, matching [`slot_cfg`].
fn session_cfg(temperature: f32) -> SessionCfg {
    SessionCfg {
        max_ctx: 512,
        temperature,
        top_p: TOP_P,
        top_k: 0,
        min_p: 0.0,
        seed: SEED,
        max_tokens: TOKENS,
        threads: 4,
        // The drafter and the suffix drafter are not what this test compares: the
        // two runtimes' *own* decoding is.
        speculate: false,
        thinking: true,
        ..Default::default()
    }
}

/// The two runtimes, given the same prompt and the same sampling controls.
fn both(model: &Arc<Model>, temperature: f32) -> (Vec<u32>, Vec<u32>) {
    let mut session =
        Session::from_model(model.clone(), session_cfg(temperature)).expect("session");
    let cfg = slot_cfg(model, temperature);
    let prompt = prompt_ids(model, USER);
    let s = session_tokens(&mut session, USER);
    let e = engine_tokens(model, &prompt, &cfg);
    (s, e)
}

/// Whether `a` is the beginning of `b`.
fn starts_with(a: &[u32], b: &[u32]) -> bool {
    a.len() <= b.len() && a == &b[..a.len()]
}

/// The tokens a decode with `position_key(seed, turn, pos)` produces.
///
/// The session's and the engine's own decodes, written out: the same prompt, the
/// same greedy-or-sampled rules, the same positions — only the turn in the key
/// differs. This is what says *which* key each runtime uses.
fn reference_decode(model: &Model, prompt: &[u32], temperature: f32, turn: u64) -> Vec<u32> {
    let c = model.cfg;
    let pool = Pool::new(4);
    let mut st = model.new_state(64, 1, 512, 4);
    let mut kv = model.new_cache(KvPrec::F16, 512);
    let mut ss = StepStats::default();
    let mut sc = Scratch::default();
    let want = vec![prompt.len() - 1];
    model
        .forward(&mut st, &mut kv, &pool, prompt, 0, &want, &mut ss)
        .expect("prefill");
    // The first token is the one the prompt's last position predicts.
    let mut pos = prompt.len();
    let mut next = sample_filtered(
        &st.logits[..c.n_vocab],
        temperature,
        TOP_P,
        0,
        0.0,
        position_key(SEED, turn, pos),
        &mut sc,
    );
    let mut out = vec![next];
    for _ in 1..TOKENS {
        model
            .forward(&mut st, &mut kv, &pool, &[next], pos, &[0], &mut ss)
            .expect("decode");
        pos += 1;
        next = sample_filtered(
            st.logits_row(0, c.n_vocab),
            temperature,
            TOP_P,
            0,
            0.0,
            position_key(SEED, turn, pos),
            &mut sc,
        );
        out.push(next);
    }
    out
}

/// **ADR 0007, decision 5: no output change in S1.4.**
///
/// The same prompt through the session and through a one-slot engine, greedy: the
/// token streams must be identical. Greedy decoding ignores the sampling key, so
/// what this pins is everything else the two runtimes do differently — the prompt
/// they build, the positions they number, the batch widths they run, and the
/// kernel paths they take.
///
/// The other difference ADR 0007 names — the session's int8 KV rotation, which it
/// turns off for Qwen2 — is not exercised here: neither model is Qwen2, and both
/// runtimes use the f16 cache.
#[test]
fn greedy_decoding_is_the_same_through_the_session_and_the_engine() {
    let dir = tmpdir("greedy");
    let models = [
        (
            write_dense(&dir, "qwen3", Format::Q8_0, 7100, Dense::default()),
            "qwen3-Q8_0",
        ),
        (
            write_hybrid(&dir, Format::PQ2_0, 7101, false),
            "qwen35-PQ2_0",
        ),
    ];
    let mut failures = Vec::new();
    for (path, name) in &models {
        let model = load(path);
        let (s, e) = both(&model, 0.0);
        eprintln!("{name}: greedy session {s:?}");
        if s != e {
            failures.push(format!(
                "{name}: the two runtimes disagree\n  session: {s:?}\n  engine:  {e:?}"
            ));
        }
        // The turn may end before its budget, on the template's end-of-turn token
        // — which token it samples depends on the kernel tier, and both runtimes
        // must end on the same one.
        if s.is_empty() {
            failures.push(format!("{name}: the turn produced no tokens"));
        }
        if s.len() > TOKENS {
            failures.push(format!(
                "{name}: {} tokens, more than the budget of {TOKENS}",
                s.len()
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the runtimes disagree:\n{}",
        failures.join("\n")
    );
}

/// **ADR 0007, decision 5: the sampling key is a per-slot setting.**
///
/// The two runtimes key their randomness differently today:
///
/// * the engine samples with `position_key(seed, 0, pos)` — the position alone
///   (`serve.rs`, `finish_decode`), so the same prompt and seed give the same
///   tokens every time, and a second request with the same prompt repeats the
///   first one's tokens;
/// * the session samples with `position_key(seed, turn, pos)`, where `turn` counts
///   the conversation's turns and is incremented *before* the first one
///   (`session.rs`, `generate_with`), so the first turn already draws with
///   `turn = 1` and every later turn with its own number.
///
/// The two streams therefore differ from the very first turn, and the session's
/// own second turn differs from its first — while the engine's second request
/// repeats its first. This is what ADR 0007 turns into a `SlotCfg` field; until
/// then it is the behaviour to keep.
///
/// The keys are pinned exactly, not just their difference: a decode written out
/// with `position_key(seed, 1, pos)` reproduces the session's first turn, and one
/// with `position_key(seed, 0, pos)` reproduces the engine's.
#[test]
fn the_two_runtimes_sample_with_different_keys() {
    let dir = tmpdir("sampled");
    let path = write_dense(&dir, "qwen3", Format::Q8_0, 7200, Dense::default());
    let model = load(&path);
    let mut failures = Vec::new();

    // The first turn: the session's key carries turn = 1, the engine's turn = 0.
    let (s1, e1) = both(&model, TEMP);
    eprintln!("T={TEMP} session turn 1: {s1:?}");
    eprintln!("T={TEMP} engine        : {e1:?}");
    if s1 == e1 {
        failures.push(format!(
            "the session and the engine sampled the same tokens at T={TEMP}: {s1:?} \
             (their keys differ: the session counts turns, the engine does not)"
        ));
    }

    // Which key each of them uses, written out: the session's first turn is a
    // decode with turn = 1, the engine's is a decode with turn = 0.
    let prompt = prompt_ids(&model, USER);
    let with_turn_1 = reference_decode(&model, &prompt, TEMP, 1);
    let with_turn_0 = reference_decode(&model, &prompt, TEMP, 0);
    if !starts_with(&s1, &with_turn_1) {
        failures.push(format!(
            "the session's first turn is not a decode with turn = 1\n  session: {s1:?}\n  turn 1:  {with_turn_1:?}"
        ));
    }
    if !starts_with(&e1, &with_turn_0) {
        failures.push(format!(
            "the engine is not a decode with turn = 0\n  engine: {e1:?}\n  turn 0: {with_turn_0:?}"
        ));
    }

    // The turn alone changes the draw: the same prompt, the same positions, two
    // turns. This is what the session's counter does, and what the engine does not
    // have.
    let with_turn_2 = reference_decode(&model, &prompt, TEMP, 2);
    if with_turn_1 == with_turn_2 {
        failures.push(
            "two turns of the same prompt drew the same tokens: the turn is not in the key"
                .to_string(),
        );
    }

    // The session draws afresh every turn: the second turn is a new prompt and a
    // new turn number, and its tokens are none of the first turn's.
    let mut session = Session::from_model(model.clone(), session_cfg(TEMP)).expect("session");
    let s_first = session_tokens(&mut session, USER);
    let s_second = session_tokens(&mut session, "And two more?");
    if s_first == s_second {
        failures.push(
            "the session's second turn repeated its first: the turn counter is not in the key"
                .to_string(),
        );
    }

    // The engine's second request with the same prompt and seed: the same tokens,
    // because its key has no turn in it.
    let cfg = slot_cfg(&model, TEMP);
    let mut engine = Engine::new(model.clone(), 4, 512, 1, KvPrec::F16, 0);
    let mut runs = Vec::new();
    for _ in 0..2 {
        let id = engine.admit(prompt.clone(), cfg.clone()).expect("admit");
        let mut out = Vec::new();
        while engine.active() > 0 {
            for em in engine.step() {
                assert_eq!(em.slot, id);
                out.push(em.token);
            }
        }
        drop_stop(&mut out, &cfg.stop);
        runs.push(out);
    }
    if runs[0] != runs[1] {
        failures.push(format!(
            "the engine's second request with the same prompt and seed drew different \
             tokens: {:?} vs {:?}",
            runs[0], runs[1]
        ));
    }
    if runs[0] != e1 {
        failures.push(format!(
            "the engine's first request in a fresh engine differs from the one above: \
             {:?} vs {:?}",
            runs[0], e1
        ));
    }

    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the sampling keys do not behave as documented:\n{}",
        failures.join("\n")
    );
}

/// **ADR 0007, decision 5: the session's turn and KV rotation are per-slot settings.**
///
/// The two runtimes differ in two places that change the output, and both are
/// settings of a slot now:
///
/// * the sampling key — the session counts its conversation turns into it, so
///   its first turn draws with `turn = 1`; the engine keeps `turn = 0`;
/// * the KV cache — the session uses the family's own choice
///   (`Architecture::session_kv_rotation`), which is no rotation for Qwen2; the
///   engine keeps the rotation on.
///
/// Told both, a one-slot engine draws exactly the session's tokens at a
/// temperature above zero. This is the test that says the session may move onto
/// the engine without a single token changing.
///
/// The model is a Qwen2 (the family that measures worse with the rotation) and
/// the cache is int8: with an f16 cache the rotation is not used at all, and
/// the choice would make no difference.
#[test]
fn a_one_slot_engine_with_the_session_s_settings_reproduces_the_session_s_tokens() {
    let dir = tmpdir("session-settings");
    let path = write_dense(&dir, "qwen2", Format::Q8_0, 7700, Dense::default());
    let model = load(&path);
    assert!(
        !model.session_kv_rotation(),
        "the synthetic qwen2 is the family that turns the rotation off"
    );

    // The session: its own settings, its own turn counter, its own cache.
    let mut scfg = session_cfg(TEMP);
    scfg.kv_prec = KvPrec::I8;
    let mut session = Session::from_model(model.clone(), scfg).expect("session");
    let s = session_tokens(&mut session, USER);

    // The engine: the same prompt, the session's turn (its first turn is turn
    // one) and the session's cache.
    let prompt = prompt_ids(&model, USER);
    let mut cfg = slot_cfg(&model, TEMP);
    cfg.turn = 1;
    cfg.kv_rotation = model.session_kv_rotation();
    let e = engine_tokens_with(&model, &prompt, &cfg, KvPrec::I8);
    eprintln!("session {s:?}\nengine  {e:?}");
    assert_eq!(
        s, e,
        "the engine with the session's settings did not reproduce the session's tokens"
    );

    // The engine's own defaults do not: the turn is in the key, and the
    // rotation is in the cache.
    let mut plain = slot_cfg(&model, TEMP);
    plain.kv_rotation = true;
    let p = engine_tokens_with(&model, &prompt, &plain, KvPrec::I8);
    assert_ne!(
        s, p,
        "the engine's defaults reproduced the session's tokens: the turn and the \
         rotation changed nothing"
    );

    // The rotation alone changes the output: the same turn, the engine's own
    // cache and the family's own.
    let mut rotated = cfg.clone();
    rotated.kv_rotation = true;
    let r = engine_tokens_with(&model, &prompt, &rotated, KvPrec::I8);
    assert_ne!(
        e, r,
        "the KV rotation changed nothing: the int8 cache is the same either way"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// **ADR 0007, decision 2: one `Drafter` interface, told what the engine computed.**
///
/// A drafter conditions on the target's tapped hidden states, so the engine has to
/// copy them out (`State::set_taps`). That copy must not change what the model
/// computes: a forward with taps and one without must give bit-identical logits,
/// or every drafter would change the target's output.
#[test]
fn taps_do_not_change_the_logits() {
    let dir = tmpdir("taps");
    let path = write_dense(&dir, "qwen3", Format::Q8_0, 7300, Dense::default());
    let model = load(&path);
    let c = model.cfg;
    let prompt = prompt_ids(&model, USER);
    let pool = Pool::new(4);

    let forward = |taps: Option<&[usize]>| -> Vec<f32> {
        let mut st = model.new_state(64, prompt.len(), 512, 4);
        if let Some(layers) = taps {
            st.set_taps(layers, c.d_model);
        }
        let mut kv = model.new_cache(KvPrec::F16, 512);
        let mut ss = StepStats::default();
        let want: Vec<usize> = (0..prompt.len()).collect();
        model
            .forward(&mut st, &mut kv, &pool, &prompt, 0, &want, &mut ss)
            .expect("forward");
        st.logits[..prompt.len() * c.n_vocab].to_vec()
    };

    // The layers a DSpark drafter taps: a few of the target's own.
    let layers = [0usize, 1];
    let plain = forward(None);
    let tapped = forward(Some(&layers));
    assert_eq!(
        plain.len(),
        prompt.len() * c.n_vocab,
        "the forward did not produce a row of logits per prompt token"
    );
    assert!(
        plain.iter().all(|v| v.is_finite()),
        "the logits are not finite"
    );
    assert!(
        plain == tapped,
        "the taps changed the logits: {} of {} differ",
        plain.iter().zip(&tapped).filter(|(a, b)| a != b).count(),
        plain.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// **ADR 0007, decision 3: speculation stays single-slot.**
///
/// A second user turns the first one's speculation off (`serve.rs`, `admit`: a
/// second slot clears every slot's `spec`). That is a scheduling decision, and it
/// must not change what the first user is told: its tokens are the ones it would
/// have got alone.
#[test]
fn a_second_user_stops_the_first_users_speculation_without_changing_its_tokens() {
    let dir = tmpdir("second-user");
    let path = write_hybrid(&dir, Format::PQ2_0, 7400, true);
    let model = load(&path);
    assert!(
        model.has_mtp(),
        "the synthetic MTP model lost its MTP block"
    );
    let prompt = prompt_ids(&model, USER);
    let cfg = slot_cfg(&model, 0.0);

    // User A alone: the engine with one slot, speculating.
    let alone = Engine::new(model.clone(), 4, 512, 1, KvPrec::F16, 0);
    assert!(alone.speculates(), "the engine does not speculate");
    let a_alone = engine_tokens(&model, &prompt, &cfg);

    // User A speculating, then user B arrives.
    let mut e = Engine::new(model.clone(), 4, 512, 2, KvPrec::F16, 0);
    let a = e.admit(prompt.clone(), cfg.clone()).expect("admit A");
    let mut a_tokens = Vec::new();
    let mut speculated = false;
    // A few steps on its own: A speculates.
    for _ in 0..4 {
        for em in e.step() {
            assert_eq!(em.slot, a);
            a_tokens.push(em.token);
        }
        if e.spec_drafts > 0 {
            speculated = true;
        }
    }
    assert!(speculated, "the first user never speculated");

    // B arrives: speculation is over for the engine.
    let b = e.admit(prompt.clone(), cfg.clone()).expect("admit B");
    assert_ne!(a, b);
    let mut b_tokens = Vec::new();
    while e.active() > 0 {
        for em in e.step() {
            match em.slot {
                s if s == a => a_tokens.push(em.token),
                s if s == b => b_tokens.push(em.token),
                _ => panic!("a third slot"),
            }
        }
    }
    assert!(!b_tokens.is_empty(), "the second user produced no tokens");
    drop_stop(&mut a_tokens, &cfg.stop);
    drop_stop(&mut b_tokens, &cfg.stop);
    eprintln!(
        "second user: A alone {:?}\n            A with B {:?}\n            B alone {:?}",
        a_alone, a_tokens, b_tokens
    );
    assert_eq!(
        a_tokens, a_alone,
        "the second user changed the first user's tokens"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// **ADR 0007, decision 1: the engine owns the runtime.**
///
/// The prefix store keeps the caches of finished conversations, so a prompt that
/// shares a long beginning with one of them is not prefilled again. A DSpark
/// drafter has a cache of its own, which the store does not know about, so the
/// store is off while a drafter is attached (`session.rs`, `enable_prefix_store`
/// and `reuse_prefix`).
///
/// This pins that: with a drafter attached the store is refused, and nothing is
/// kept or restored. Without one it works: a conversation is kept, and a later
/// prompt that shares its beginning is prefilled from the store.
#[test]
fn the_prefix_store_stays_off_while_a_drafter_is_attached() {
    let dir = tmpdir("prefix-store");
    let model_path = write_dense(&dir, "qwen3", Format::Q8_0, 7500, Dense::default());
    let model = load(&model_path);
    let drafter = support::dspark::write_drafter(
        &dir,
        support::dspark::DrafterShape {
            n_vocab: model.cfg.n_vocab,
            hidden: model.cfg.d_model,
            ..Default::default()
        },
        7600,
    );

    let mut cfg = session_cfg(0.0);
    cfg.dspark = Some(drafter);
    let mut session = Session::from_model(model.clone(), cfg).expect("session with a drafter");
    assert!(session.dspark.is_some(), "the drafter was not attached");

    // The store is asked for, and refused: a drafter's own cache is not in it.
    session.enable_prefix_store(1 << 20);
    assert_eq!(
        session.prefix_store_stats(),
        None,
        "the prefix store was enabled while a drafter was attached"
    );

    // Three conversations: the first, one that shares only its beginning with it,
    // and the first one again.
    session.generate(USER, |_| {}).expect("first turn");
    session.reset();
    session
        .generate("Name three fruits.", |_| {})
        .expect("second turn");
    session.reset();
    session.generate(USER, |_| {}).expect("third turn");
    assert_eq!(
        session.prefix_store_stats(),
        None,
        "the store came up while a drafter was attached"
    );

    // The same three conversations without a drafter: the store keeps the first
    // one, and the third prompt is prefilled from it.
    let mut plain = Session::from_model(model.clone(), session_cfg(0.0)).expect("session");
    plain.enable_prefix_store(1 << 20);
    plain.generate(USER, |_| {}).expect("first turn");
    plain.reset();
    plain
        .generate("Name three fruits.", |_| {})
        .expect("second turn");
    plain.reset();
    plain.generate(USER, |_| {}).expect("third turn");
    let (entries, _, hits, saved) = plain.prefix_store_stats().expect("the store is off");
    assert!(entries > 0, "the store kept no conversation");
    assert!(hits > 0, "the third prompt did not restore a prefix");
    assert!(saved > 0, "the restore saved no positions");
    let _ = std::fs::remove_dir_all(&dir);
}
