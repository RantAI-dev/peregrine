//! The `Drafter` interface over the code that already drafts (ADR 0007).
//!
//! ADR 0007 gives speculation one interface: a drafter proposes, is told what
//! the target committed, and is told what was rolled back. The three drafters
//! that exist today are thin wrappers over the code that already implements
//! them — the suffix index, DeepSeek's DSpark head and the model's own MTP
//! block — and what they propose must be exactly what that code proposes, or
//! moving the session onto the engine would change the output.
//!
//! Every test here writes the same state twice: once through the code that
//! exists today, once through the wrapper, and compares.

mod support;

use peregrine_core::draft::{Drafter, Needs, Proposal, SuffixDrafter};
use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, StepStats};
use peregrine_core::pool::Pool;
use peregrine_core::sample::argmax;
use peregrine_core::serve::{Engine, SlotCfg};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{write_dense, write_hybrid, Dense};

/// A directory of its own for each test (tests run in parallel and each removes its directory).
fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-drafters-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn load(path: &Path) -> Arc<Model> {
    Arc::new(Gguf::open(path).and_then(peregrine_models::load).unwrap())
}

/// The sampler's stand-in: the target's own argmax, which is what a greedy
/// target draws.
fn greedy(_i: usize, row: &[f32]) -> u32 {
    argmax(row)
}

// ---------------------------------------------------------------------------
// The suffix drafter
// ---------------------------------------------------------------------------

/// **ADR 0007, decision 2: one `Drafter` interface.**
///
/// The suffix drafter proposes the longest match of the conversation's tail,
/// continued by what followed it last time. What the wrapper proposes must be
/// what the index proposes, with the confidence the index has learned for each
/// token — that confidence is what the engine prices the draft with.
#[test]
fn the_suffix_drafter_proposes_what_the_suffix_index_proposes() {
    // A conversation that repeats itself: the tail of the context occurred
    // earlier, and what followed it there is the proposal.
    let ctx: Vec<u32> = vec![9, 1, 2, 3, 4, 5, 6, 7, 1, 2];
    let anchor = 3u32; // the token the target sampled, not yet fed
    let min_order = 2;
    let max_len = 6;
    let pool = Pool::new(4);

    // Today's path: the index proposes, and the confidences come from its cells.
    let mut index = peregrine_core::suffix::SuffixIndex::new();
    index.sync(&ctx, min_order);
    let d = index.propose(&ctx, anchor, max_len, min_order);
    assert!(!d.toks.is_empty(), "the corpus has no match to propose");
    let conf: Vec<f32> = d.cells.iter().map(|&c| index.conf(c)).collect();

    // The wrapper: the same corpus, committed one position at a time.
    let mut drafter = SuffixDrafter::new(min_order, max_len);
    for (i, &t) in ctx.iter().enumerate() {
        drafter.commit(i, t, &[], &[], &pool);
    }
    assert_eq!(
        drafter.context(),
        &ctx[..],
        "the corpus did not reach the drafter"
    );
    let mut out = Proposal::default();
    drafter.propose(anchor, ctx.len(), max_len, &pool, &mut greedy, &mut out);

    assert_eq!(out.toks, d.toks, "the proposal differs from the index's");
    assert_eq!(out.conf, conf, "the confidences differ from the index's");
}

/// The proposal carries what the engine prices the draft with: the same
/// `choose_k` over the same confidences picks the same number of tokens.
#[test]
fn the_suffix_drafter_proposal_prices_the_same_way() {
    let ctx: Vec<u32> = vec![9, 1, 2, 3, 4, 5, 6, 7, 1, 2];
    let anchor = 3u32;
    let min_order = 2;
    let max_len = 6;
    let pool = Pool::new(4);

    let mut index = peregrine_core::suffix::SuffixIndex::new();
    index.sync(&ctx, min_order);
    let d = index.propose(&ctx, anchor, max_len, min_order);
    let conf: Vec<f32> = d.cells.iter().map(|&c| index.conf(c)).collect();
    let cost = |w: usize| 1.0 + 0.42 * (w as f64 - 1.0);
    let k = peregrine_core::draft::choose_k(&conf, cost);

    let mut drafter = SuffixDrafter::new(min_order, max_len);
    for (i, &t) in ctx.iter().enumerate() {
        drafter.commit(i, t, &[], &[], &pool);
    }
    let mut out = Proposal::default();
    drafter.propose(anchor, ctx.len(), max_len, &pool, &mut greedy, &mut out);
    assert_eq!(
        peregrine_core::draft::choose_k(&out.conf, cost),
        k,
        "the engine would verify a different number of tokens"
    );
}

/// What the target committed is what the drafter learns from: the confidence
/// table is booked exactly as the session books it.
#[test]
fn the_suffix_drafter_books_what_the_target_committed() {
    let ctx: Vec<u32> = vec![9, 1, 2, 3, 4, 5, 6, 7, 1, 2];
    let anchor = 3u32;
    let min_order = 2;
    let max_len = 6;
    let pool = Pool::new(4);

    // Today's path: the index proposes, and the outcome is booked on it.
    let mut index = peregrine_core::suffix::SuffixIndex::new();
    index.sync(&ctx, min_order);
    let d = index.propose(&ctx, anchor, max_len, min_order);

    // The wrapper: the same corpus, the same proposal, the same outcome.
    let mut drafter = SuffixDrafter::new(min_order, max_len);
    for (i, &t) in ctx.iter().enumerate() {
        drafter.commit(i, t, &[], &[], &pool);
    }
    let mut out = Proposal::default();
    drafter.propose(anchor, ctx.len(), max_len, &pool, &mut greedy, &mut out);
    // The draft is committed whole: every token survived.
    for (j, &t) in d.toks.iter().enumerate() {
        drafter.commit(ctx.len() + 1 + j, t, &[], &[], &pool);
    }

    // The same booking, on the same cells: the confidences agree.
    index.observe(&d.cells, d.toks.len());
    for &c in &d.cells {
        assert!(
            (index.conf(c) - drafter.index().conf(c)).abs() < 1e-6,
            "the drafter booked the outcome differently: {} vs {}",
            index.conf(c),
            drafter.index().conf(c)
        );
    }
}

// ---------------------------------------------------------------------------
// The engine's taps
// ---------------------------------------------------------------------------

/// What a drafter was told about one committed position: the position, the
/// token, the width of the taps, the width of the hidden state.
type Told = (usize, u32, usize, usize);

/// A drafter that records what the engine tells it.
struct Recorder {
    /// The tap layers it asks for.
    layers: Vec<usize>,
    /// Whether it asks for the target's hidden states.
    hidden: bool,
    /// What it was told.
    seen: Arc<std::sync::Mutex<Vec<Told>>>,
}

impl Drafter for Recorder {
    fn needs(&self) -> Needs {
        Needs {
            taps: !self.layers.is_empty(),
            hidden: self.hidden,
            history: false,
        }
    }

    fn tap_layers(&self) -> &[usize] {
        &self.layers
    }

    fn propose(
        &mut self,
        _anchor: u32,
        _anchor_pos: usize,
        _k_max: usize,
        _pool: &Pool,
        _pick: &mut dyn FnMut(usize, &[f32]) -> u32,
        out: &mut Proposal,
    ) {
        out.clear();
    }

    fn commit(&mut self, pos: usize, token: u32, taps: &[f32], hidden: &[f32], _pool: &Pool) {
        self.seen
            .lock()
            .expect("the recorder")
            .push((pos, token, taps.len(), hidden.len()));
    }

    fn rollback(&mut self, _pos: usize) {}

    fn reset(&mut self) {}
}

/// The engine copies out what a drafter asks for, and nothing else.
///
/// A drafter that conditions on the target's tapped hidden states needs those
/// rows copied out of the forward pass; a request without a drafter must not
/// pay for the copy. The copy must not change what the model computes either:
/// the tokens are the same with a drafter attached and without one.
#[test]
fn the_engine_copies_out_the_taps_a_drafter_asks_for() {
    let dir = tmpdir("engine-taps");
    let path = write_dense(&dir, "qwen3", Format::Q8_0, 8100, Dense::default());
    let model = load(&path);
    let d_model = model.cfg.d_model;
    let prompt: Vec<u32> = (0..8).map(|i| (i * 13 % 200) as u32).collect();
    let cfg = SlotCfg {
        temperature: 0.0,
        max_tokens: 8,
        ..Default::default()
    };

    // Without a drafter: nothing is copied out of the forward pass.
    let mut e = Engine::new(model.clone(), 4, 512, 1, KvPrec::F16, 0);
    assert!(
        e.tap_layers().is_empty(),
        "the engine copies taps out for nobody"
    );
    let id = e.admit(prompt.clone(), cfg.clone()).expect("admit");
    let plain = run(&mut e, id);

    // With one: the layers it asks for are copied, and it is told every
    // committed position with that row's taps.
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut e = Engine::new(model.clone(), 4, 512, 1, KvPrec::F16, 0);
    let id = e.admit(prompt.clone(), cfg.clone()).expect("admit");
    e.attach_drafter(
        id,
        Box::new(Recorder {
            layers: vec![0],
            hidden: true,
            seen: seen.clone(),
        }),
    )
    .expect("attach");
    assert_eq!(e.tap_layers(), &[0usize][..], "the taps were not set up");
    let with = run(&mut e, id);

    assert_eq!(plain, with, "the taps changed the engine's output");
    let seen = seen.lock().expect("the recorder");
    assert!(!seen.is_empty(), "the drafter was told nothing");
    for (pos, _token, taps, hidden) in seen.iter() {
        assert_eq!(
            *taps, d_model,
            "the taps of position {pos} are not the layer the drafter asked for"
        );
        // Only a hybrid forward writes the target's hidden states; a dense one
        // has none to give.
        assert_eq!(
            *hidden, 0,
            "the hidden state of position {pos} is not the target's row"
        );
    }
    // Every committed position, in order.
    let positions: Vec<usize> = seen.iter().map(|(p, ..)| *p).collect();
    assert_eq!(
        positions,
        (0..positions.len()).collect::<Vec<usize>>(),
        "the drafter was told the positions out of order"
    );
}

/// The token ids an engine produces for one slot.
fn run(e: &mut Engine, id: peregrine_core::serve::SlotId) -> Vec<u32> {
    let mut out = Vec::new();
    while e.active() > 0 {
        for em in e.step() {
            assert_eq!(em.slot, id);
            out.push(em.token);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The DSpark drafter
// ---------------------------------------------------------------------------

/// The DSpark head proposes a block after the anchor, with the confidence the
/// head's own head estimates. The wrapper must propose exactly that.
#[cfg(feature = "dspark")]
#[test]
fn the_dspark_drafter_proposes_what_the_dspark_head_proposes() {
    use peregrine_core::draft::DsparkDrafter;

    let dir = tmpdir("dspark");
    let model_path = write_dense(&dir, "qwen3", Format::Q8_0, 7800, Dense::default());
    let model = load(&model_path);
    let shape = support::dspark::DrafterShape {
        n_vocab: model.cfg.n_vocab,
        hidden: model.cfg.d_model,
        ..Default::default()
    };
    let drafter_dir = support::dspark::write_drafter(&dir, shape, 7900);
    let pool = Pool::new(4);
    let d_model = model.cfg.d_model;

    // The target's tapped hidden states: the drafter's context.
    let prompt: Vec<u32> = (0..4).map(|i| (i * 37 % 200) as u32).collect();
    let width = d_model; // one tapped layer
    let mut st = model.new_state(64, 1, 512, 4);
    st.set_taps(&[0], d_model);
    let mut kv = model.new_cache(KvPrec::F16, 512);
    let mut ss = StepStats::default();
    model
        .forward(&mut st, &mut kv, &pool, &prompt, 0, &[], &mut ss)
        .expect("forward");
    let taps = st.tap[..prompt.len() * width].to_vec();

    // Today's path: the head, its context, and the draft.
    let mut head = peregrine_core::dspark::Drafter::load(&drafter_dir, 4).expect("load");
    head.check_target(model.cfg.n_vocab, d_model, model.cfg.n_layer)
        .expect("the drafter is for this target");
    head.update_context(&taps, 0, &pool);
    let anchor = 7u32;
    let (toks, conf) = head.draft_with(anchor, 3, &pool, greedy);

    // The wrapper: the same head, the same context, learned through commits.
    let head2 = peregrine_core::dspark::Drafter::load(&drafter_dir, 4).expect("load");
    let mut drafter = DsparkDrafter::new(head2);
    for (i, &t) in prompt.iter().enumerate() {
        drafter.commit(i, t, &taps[i * width..(i + 1) * width], &[], &pool);
    }
    let mut out = Proposal::default();
    drafter.propose(anchor, prompt.len(), 3, &pool, &mut greedy, &mut out);

    assert_eq!(out.toks, toks, "the proposal differs from the head's");
    assert_eq!(out.conf, conf, "the confidences differ from the head's");
    assert_eq!(
        drafter.tap_layers(),
        &[0usize][..],
        "the drafter asks for the layers it taps"
    );
}

// ---------------------------------------------------------------------------
// The MTP drafter
// ---------------------------------------------------------------------------

/// The model's own MTP block proposes the token after the anchor. The wrapper
/// must propose exactly that, from the same block state.
#[cfg(feature = "mtp")]
#[test]
fn the_mtp_drafter_proposes_what_the_mtp_block_proposes() {
    use peregrine_core::draft::MtpDrafter;

    let dir = tmpdir("mtp");
    let path = write_hybrid(&dir, Format::PQ2_0, 8000, true);
    let model = load(&path);
    assert!(model.has_mtp(), "the synthetic model lost its MTP block");
    let pool = Pool::new(4);
    let d_model = model.cfg.d_model;

    // The target's hidden states for a prompt: what the block's rows read.
    let prompt: Vec<u32> = (0..6).map(|i| (i * 37 % 200) as u32).collect();
    let mut st = model.new_state(64, 1, 512, 4);
    st.want_hidden = true;
    let mut kv = model.new_cache(KvPrec::F16, 512);
    let mut ss = StepStats::default();
    model
        .forward(&mut st, &mut kv, &pool, &prompt, 0, &[], &mut ss)
        .expect("forward");
    let hidden = st.hidden.clone();
    assert_eq!(hidden.len(), prompt.len() * d_model);

    // Today's path: the engine catches the block up on the prompt, then feeds
    // it the anchor's row, and it drafts.
    let mut ms = model.mtp_state(512, 4).expect("mtp state");
    let mut h = vec![0.0f32; d_model];
    h.extend_from_slice(&hidden[..(prompt.len() - 1) * d_model]);
    let positions: Vec<usize> = (0..prompt.len()).collect();
    model
        .mtp_forward(&mut ms, &pool, &prompt, &h, &positions, false, &mut ss)
        .expect("catch-up");
    let anchor = 42u32;
    let draft = model
        .mtp_forward(
            &mut ms,
            &pool,
            &[anchor],
            &hidden[(prompt.len() - 1) * d_model..],
            &[prompt.len()],
            true,
            &mut ss,
        )
        .expect("draft")
        .expect("the block drafted nothing");

    // The wrapper: the same rows, told as commits.
    let mut drafter = MtpDrafter::new(model.clone(), 512, 4).expect("drafter");
    for (i, &t) in prompt.iter().enumerate() {
        drafter.commit(i, t, &[], &hidden[i * d_model..(i + 1) * d_model], &pool);
    }
    let mut out = Proposal::default();
    drafter.propose(anchor, prompt.len(), 1, &pool, &mut greedy, &mut out);

    assert_eq!(
        out.toks,
        vec![draft],
        "the proposal differs from the block's"
    );
}
