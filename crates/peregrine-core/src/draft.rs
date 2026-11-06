//! Drafting: what a draft costs, what verifying it costs, and how many of its tokens
//! are worth verifying.
//!
//! This module is the home of the verify-cost model ([`VerifyCost`]) and of the rule
//! that decides how much of a draft to check ([`choose_k`]). It measures the engine's
//! *own* forward passes, so it belongs with the runtime that runs them — ADR 0007
//! gives it to the one runtime (`serve::Engine`) and puts the `Drafter` interface
//! (MTP, DSpark, suffix speculation) here as well.
//!
//! # The model
//!
//! A verification pass of width `w` costs one plain step plus `w − 1` extra rows.
//! How much an extra row costs is measured, not assumed ([`ROW_COST`]), because on
//! this machine verification is compute-bound rather than bandwidth-bound: a width-8
//! pass costs about four steps, not one. The measured widths are kept as they are
//! seen (an EMA per width) and every width that was never measured is priced from
//! the slope the measured ones show.
//!
//! [`choose_k`] then asks the only question that matters: how many drafted tokens
//! to verify, given how likely each is to survive and what the verification costs.
//! A doubtful draft is not verified at all.
//!
//! # The drafter interface
//!
//! ADR 0007 gives speculation one interface: a [`Drafter`] proposes tokens, is told
//! what the target committed, and is told what was rolled back. It declares what it
//! needs from the engine ([`Needs`]) and writes its proposal into a buffer the engine
//! owns ([`Proposal`]), so a draft costs no allocation.
//!
//! The three drafters that exist today are thin wrappers over the code that already
//! implements them: [`SuffixDrafter`] over the suffix index, [`DsparkDrafter`] over
//! DeepSeek's DSpark head, [`MtpDrafter`] over the model's own MTP block. What the
//! engine does with a proposal — how much of it to verify, at what price — stays with
//! the engine: the verify-cost model measures the engine's own forward passes.
//!
//! The interface is unstable until 1.0, like the `Architecture` interface (ADR 0005).

use crate::pool::Pool;

/// What one extra row costs a verification pass, as a fraction of a plain step.
/// Measured with `examples/vcurve` (width 1..16, median of 9): SmolLM2-360M 0.47,
/// Llama-3.2-1B 0.40, Qwen2.5-0.5B 0.42. Verification on this CPU is compute-bound,
/// not bandwidth-bound — a width-8 pass costs about 4 steps, not 1 — which is why
/// the ceiling on any lossless speculation here is near 1/0.42 ≈ 2.4×. The engine
/// priced a row at 0.12 before this was measured, and drafted far too short.
pub const ROW_COST: f64 = 0.42;

/// Plain steps to take after drafting is measured to lose, before probing again.
pub const PARK_STEPS: u32 = 12;

/// How often a drafting round is probed even when the price says not to draft.
///
/// A confidence that is never tested never recovers, so every `PROBE_EVERY`th step
/// verifies a couple of tokens anyway.
pub const PROBE_EVERY: u64 = 16;

/// Seconds a verification forward of width `w` takes, learned online. A width seen
/// twice is its own measurement (an EMA; the first reading of a width is a cold one
/// and the second can only lower it). Any other width is the plain-step time scaled
/// by `1 + β·(w−1)`, with β the mean slope of the widths measured so far, or
/// `ROW_COST` before any. Extrapolating a line through two or three noisy points
/// out to width 12 — what this did before — priced a 12-token draft at 85 ms
/// against a true 77 ms-or-less and refused every long draft.
#[derive(Default)]
pub struct VerifyCost {
    /// The measured seconds per width, once a width has been seen.
    pub(crate) ema: [Option<f64>; 17],
    /// How many times each width has been measured.
    pub(crate) seen: [u8; 17],
}

impl VerifyCost {
    /// Record one measurement of a verification pass of width `w`.
    pub fn observe(&mut self, w: usize, secs: f64) {
        if w == 0 || w >= self.ema.len() {
            return;
        }
        self.seen[w] = self.seen[w].saturating_add(1);
        self.ema[w] = Some(match (self.ema[w], self.seen[w]) {
            (Some(e), 2) => e.min(secs),
            (Some(e), _) => 0.8 * e + 0.2 * secs,
            (None, _) => secs,
        });
    }
    /// Mean extra-row cost, in plain steps, over the widths measured at least twice.
    pub fn slope(&self) -> f64 {
        let Some(t1) = self.ema[1].filter(|_| self.seen[1] >= 2) else {
            return ROW_COST;
        };
        let s: Vec<f64> = (2..self.ema.len())
            .filter(|&w| self.seen[w] >= 2)
            .filter_map(|w| {
                self.ema[w].map(|t| ((t / t1 - 1.0) / (w as f64 - 1.0)).clamp(0.15, 0.9))
            })
            .collect();
        if s.is_empty() {
            ROW_COST
        } else {
            s.iter().sum::<f64>() / s.len() as f64
        }
    }
    /// The seconds a verification pass of width `w` is expected to take, or `None`
    /// while not even a plain step has been timed.
    pub fn estimate(&self, w: usize) -> Option<f64> {
        if let (Some(t), true) = (
            self.ema.get(w).copied().flatten(),
            self.seen.get(w).is_some_and(|&n| n >= 2),
        ) {
            return Some(t);
        }
        let t1 = self.ema[1]?;
        Some(t1 * (1.0 + self.slope() * (w as f64 - 1.0)))
    }
    /// `estimate`, or the prior curve while not even a plain step has been timed.
    pub fn cost(&self, w: usize) -> f64 {
        self.estimate(w)
            .unwrap_or(1.0 + ROW_COST * (w as f64 - 1.0))
    }
}

/// How many of `conf.len()` drafted tokens to verify: the prefix length `k` that
/// maximises expected committed tokens per second of verification, where
/// `conf[j]` is the head's probability that draft `j` survives given its prefix did.
/// The prefix survives with the running product, so expected committed tokens are
/// `1 + Σ_{j<k} Π_{i≤j} conf[i]` (the `1` is the target's own token). DeepInfer's
/// batch version of the same rule is `mtp_varlen.hpp`; at batch 1 it is this.
pub fn choose_k(conf: &[f32], cost: impl Fn(usize) -> f64) -> usize {
    let (mut best_k, mut best) = (0usize, 1.0 / cost(1));
    let (mut surv, mut expect) = (1.0f64, 1.0f64);
    for (j, &c) in conf.iter().enumerate() {
        surv *= c as f64;
        expect += surv;
        let rate = expect / cost(j + 2);
        if rate > best {
            best = rate;
            best_k = j + 1;
        }
    }
    best_k
}

// ---------------------------------------------------------------------------
// The drafter interface (ADR 0007)
// ---------------------------------------------------------------------------

/// What a drafter needs from the engine.
///
/// The engine reads this before a step and pays for what is asked for: the
/// hidden states and the taps are copies out of the forward pass, and the
/// history is the committed token sequence. A drafter that needs none of them
/// costs the engine nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Needs {
    /// The target's final hidden state per row (`State::want_hidden`).
    pub hidden: bool,
    /// The target's tapped hidden states per row (`State::set_taps`).
    pub taps: bool,
    /// The committed token history, position by position.
    pub history: bool,
}

/// A drafter's proposal, written into a buffer the engine owns and reuses (no
/// allocation per draft).
#[derive(Debug, Default)]
pub struct Proposal {
    /// The proposed tokens, in order.
    pub toks: Vec<u32>,
    /// Each token's conditional survival probability: the chance it is accepted
    /// given that the tokens before it were. The engine prices the proposal
    /// with it ([`choose_k`]).
    pub conf: Vec<f32>,
}

impl Proposal {
    /// Forget the proposal. The buffer keeps its capacity, so the next draft
    /// does not allocate.
    pub fn clear(&mut self) {
        self.toks.clear();
        self.conf.clear();
    }
}

/// One way of proposing tokens.
///
/// A drafter is told what the target committed and what was rolled back, and
/// asks for what it needs ([`Drafter::needs`]). It proposes into a buffer the
/// engine owns ([`Proposal`]) and keeps whatever bookkeeping it needs — the
/// cells a suffix draft is booked to, a DSpark block's context, the hidden
/// state an MTP row is owed — inside itself: the engine must not have to know
/// what a drafter draws drafts with.
///
/// The interface is unstable until 1.0 (ADR 0005).
pub trait Drafter: Send {
    /// What the drafter needs from the engine.
    fn needs(&self) -> Needs;

    /// Propose up to `k_max` tokens after `anchor` (sampled by the target, not
    /// yet fed), which will sit at `anchor_pos`.
    ///
    /// `pick(i, logits)` is the target's own position-keyed sampler for the
    /// `i`-th drafted position, for drafters that sample: a drafter that draws
    /// with the target's noise proposes tokens that survive sampling as often
    /// as they survive the argmax.
    ///
    /// Clear `out` and fill it; leave it empty to skip drafting.
    fn propose(
        &mut self,
        anchor: u32,
        anchor_pos: usize,
        k_max: usize,
        pool: &Pool,
        pick: &mut dyn FnMut(usize, &[f32]) -> u32,
        out: &mut Proposal,
    );

    /// Position `pos` was committed with `token`; `taps`/`hidden` are the
    /// target's rows for it (empty when not needed).
    fn commit(&mut self, pos: usize, token: u32, taps: &[f32], hidden: &[f32], pool: &Pool);

    /// Positions `pos..` were not committed: forget them.
    fn rollback(&mut self, pos: usize);

    /// Forget everything: the conversation the drafter was drafting for is over.
    fn reset(&mut self);

    /// The target layers whose hidden states this drafter needs copied out of
    /// the forward pass.
    ///
    /// Empty: it needs no taps. The engine copies them only for a drafter that
    /// asks, so a request without a drafter pays nothing for them.
    ///
    /// This is the one method the interface of ADR 0007 does not have: the
    /// engine has to know *which* layers to copy, and [`Needs::taps`] can only
    /// say that it needs some.
    fn tap_layers(&self) -> &[usize] {
        &[]
    }
}

// ---------------------------------------------------------------------------
// The suffix drafter
// ---------------------------------------------------------------------------

/// The proposal being verified: the anchor's position, the tokens proposed
/// after it, and the cells they are booked to when the target decides.
struct Pending {
    /// The position the anchor sits at; the proposal's tokens follow it.
    anchor_pos: usize,
    /// The proposed tokens.
    toks: Vec<u32>,
    /// The confidence cells the tokens are booked to.
    cells: Vec<crate::suffix::Cell>,
    /// How many of them the target has committed and matched.
    accepted: usize,
}

/// Suffix-indexed drafting as a [`Drafter`]: the longest match of the
/// conversation's tail, continued by what followed it last time.
///
/// The corpus it searches is the one it is told about: every committed
/// position joins it ([`Drafter::commit`]) and a rollback takes the tail back
/// ([`Drafter::rollback`]). What it proposes is the match's continuation with
/// the confidence the index has learned for each of its tokens; how much of it
/// is verified is the engine's decision ([`choose_k`]).
pub struct SuffixDrafter {
    /// The n-gram index over the corpus, and the confidence it has learned.
    index: crate::suffix::SuffixIndex,
    /// The committed tokens: the corpus the index searches.
    ctx: Vec<u32>,
    /// The proposal being verified, if any.
    pending: Option<Pending>,
    /// The shortest suffix accepted as a match; 0 disables drafting.
    min_order: usize,
    /// The most tokens one proposal may hold.
    max_len: usize,
}

impl SuffixDrafter {
    /// A suffix drafter.
    ///
    /// `min_order` is the shortest suffix (in tokens) accepted as a match; 0
    /// disables drafting. `max_len` is the most tokens one proposal may hold.
    pub fn new(min_order: usize, max_len: usize) -> Self {
        SuffixDrafter {
            index: crate::suffix::SuffixIndex::new(),
            ctx: Vec::new(),
            pending: None,
            min_order,
            max_len,
        }
    }

    /// The committed tokens the index searches: the conversation as the target
    /// has seen it.
    pub fn context(&self) -> &[u32] {
        &self.ctx
    }

    /// The index itself: the n-grams it holds and the confidence it has
    /// learned about them.
    pub fn index(&self) -> &crate::suffix::SuffixIndex {
        &self.index
    }
}

impl Drafter for SuffixDrafter {
    fn needs(&self) -> Needs {
        Needs {
            history: true,
            ..Needs::default()
        }
    }

    fn propose(
        &mut self,
        anchor: u32,
        anchor_pos: usize,
        k_max: usize,
        _pool: &Pool,
        _pick: &mut dyn FnMut(usize, &[f32]) -> u32,
        out: &mut Proposal,
    ) {
        out.clear();
        self.pending = None;
        let k = k_max.min(self.max_len);
        if k == 0 || self.min_order == 0 || self.ctx.is_empty() {
            return;
        }
        self.index.sync(&self.ctx, self.min_order);
        let d = self.index.propose(&self.ctx, anchor, k, self.min_order);
        if d.toks.is_empty() {
            // No match: whatever run was under way is over.
            self.index.cool();
            return;
        }
        out.toks.extend_from_slice(&d.toks);
        for &c in &d.cells {
            out.conf.push(self.index.conf(c));
        }
        let mut pending = Pending {
            anchor_pos,
            toks: Vec::with_capacity(d.toks.len()),
            cells: Vec::with_capacity(d.cells.len()),
            accepted: 0,
        };
        pending.toks.extend_from_slice(&d.toks);
        pending.cells.extend_from_slice(&d.cells);
        self.pending = Some(pending);
    }

    fn commit(&mut self, pos: usize, token: u32, _taps: &[f32], _hidden: &[f32], _pool: &Pool) {
        // The token joins the corpus at its position.
        if pos < self.ctx.len() {
            self.ctx.truncate(pos);
        }
        if pos == self.ctx.len() {
            self.ctx.push(token);
        }
        // Was it one of the drafted tokens? A match is an acceptance, a
        // mismatch ends the run: the tokens after it were never tested.
        let Some(p) = self.pending.as_mut() else {
            return;
        };
        let Some(j) = pos.checked_sub(p.anchor_pos + 1) else {
            return;
        };
        if j >= p.toks.len() {
            return;
        }
        if p.toks[j] == token {
            p.accepted = j + 1;
            if p.accepted == p.toks.len() {
                // The whole draft survived.
                let cells = std::mem::take(&mut p.cells);
                self.index.observe(&cells, p.accepted);
                self.pending = None;
            }
        } else {
            // Rejected at depth `j`: the tokens before it were accepted, this
            // one was not, and the rest were never tested.
            let cells = std::mem::take(&mut p.cells);
            self.index.observe(&cells, j);
            self.pending = None;
        }
    }

    fn rollback(&mut self, pos: usize) {
        if pos < self.ctx.len() {
            self.ctx.truncate(pos);
        }
        // A proposal whose positions were not committed was not verified: the
        // run it belonged to is over.
        if let Some(p) = &self.pending {
            if p.anchor_pos + 1 >= pos {
                self.index.cool();
                self.pending = None;
            }
        }
    }

    fn reset(&mut self) {
        self.ctx.clear();
        self.pending = None;
        // The corpus is forgotten; what the index learned about how reliable
        // each kind of match is is kept.
        self.index.clear();
    }
}

// ---------------------------------------------------------------------------
// The DSpark drafter
// ---------------------------------------------------------------------------

/// DSpark (DeepSeek's speculative head) as a [`Drafter`].
///
/// The drafter conditions on the target's tapped hidden states, so it needs
/// them ([`Needs::taps`]) and the engine copies them out of the forward pass.
/// The taps are batched: [`Drafter::commit`] buffers a position's row and the
/// next proposal hands the whole batch to the drafter's context, as the
/// session does.
#[cfg(feature = "dspark")]
pub struct DsparkDrafter {
    /// The drafter itself: its block, its context cache, its weights.
    drafter: crate::dspark::Drafter,
    /// The taps of the committed positions, not yet in the drafter's context.
    pending: Vec<f32>,
    /// The position the pending taps start at.
    pending_start: usize,
}

#[cfg(feature = "dspark")]
impl DsparkDrafter {
    /// A DSpark drafter over an already loaded head.
    pub fn new(drafter: crate::dspark::Drafter) -> Self {
        DsparkDrafter {
            drafter,
            pending: Vec::new(),
            pending_start: 0,
        }
    }

    /// The drafter itself, for the callers that read its statistics.
    pub fn inner(&self) -> &crate::dspark::Drafter {
        &self.drafter
    }

    /// The committed positions' taps, in one batch, join the drafter's context.
    fn flush(&mut self, pool: &Pool) {
        if self.pending.is_empty() {
            return;
        }
        let start = self.pending_start;
        let taps = std::mem::take(&mut self.pending);
        self.drafter.update_context(&taps, start, pool);
    }
}

#[cfg(feature = "dspark")]
impl Drafter for DsparkDrafter {
    fn needs(&self) -> Needs {
        Needs {
            taps: true,
            ..Needs::default()
        }
    }

    fn tap_layers(&self) -> &[usize] {
        &self.drafter.cfg.target_layer_ids
    }

    fn propose(
        &mut self,
        anchor: u32,
        _anchor_pos: usize,
        k_max: usize,
        pool: &Pool,
        pick: &mut dyn FnMut(usize, &[f32]) -> u32,
        out: &mut Proposal,
    ) {
        out.clear();
        // The committed positions' taps are the drafter's context; they must be
        // in it before it can read anything.
        self.flush(pool);
        let (toks, conf) = self
            .drafter
            .draft_with(anchor, k_max, pool, |i, row| pick(i, row));
        out.toks.extend_from_slice(&toks);
        out.conf.extend_from_slice(&conf);
    }

    fn commit(&mut self, pos: usize, _token: u32, taps: &[f32], _hidden: &[f32], _pool: &Pool) {
        if taps.is_empty() {
            return;
        }
        if self.pending.is_empty() {
            self.pending_start = pos;
        }
        self.pending.extend_from_slice(taps);
    }

    fn rollback(&mut self, pos: usize) {
        // The taps of the positions that were not committed are dropped.
        let width = self.drafter.cfg.target_layer_ids.len() * self.drafter.cfg.hidden;
        let keep = pos.saturating_sub(self.pending_start);
        self.pending.truncate(keep * width);
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.drafter.truncate(0);
    }
}

// ---------------------------------------------------------------------------
// The MTP drafter
// ---------------------------------------------------------------------------

/// The model's own MTP block as a [`Drafter`].
///
/// The block is depth-1: it proposes the token after the anchor, and the
/// engine verifies it against the target's next row. It reads the target's
/// final hidden state of the position *before* the anchor, so it needs them
/// ([`Needs::hidden`]).
///
/// The block keeps a cache of its own, one row per committed position: the
/// committed token and the hidden state of the position before it, which is
/// what the block's row at that position reads. The rows of the positions
/// that are committed are fed as they are committed; the anchor's row is fed
/// with the proposal, because the anchor is not committed yet when it is
/// proposed.
#[cfg(feature = "mtp")]
pub struct MtpDrafter {
    model: std::sync::Arc<crate::model::Model>,
    /// The MTP block's own cache.
    ms: crate::model::mtp::MtpState,
    /// The hidden state of the last committed position: the block's row for
    /// the anchor reads it.
    hprev: Vec<f32>,
    /// The row an accepted draft leaves owed.
    owed: Option<Owed>,
    /// The last proposal: the anchor's position and the token proposed after it.
    proposal: Option<(usize, u32)>,
    /// The block's own forward statistics.
    stats: crate::model::StepStats,
}

/// The row an accepted draft leaves owed: the token that survived, the hidden
/// state of the position before it (what the block's row at that position
/// reads) and the position the row sits at.
#[cfg(feature = "mtp")]
struct Owed {
    token: u32,
    hidden: Vec<f32>,
    pos: usize,
}

#[cfg(feature = "mtp")]
impl MtpDrafter {
    /// An MTP drafter for `model`, with a block cache of `max_ctx` positions.
    pub fn new(
        model: std::sync::Arc<crate::model::Model>,
        max_ctx: usize,
        threads: usize,
    ) -> Result<Self, String> {
        let ms = model.mtp_state(max_ctx, threads)?;
        let d = model.cfg.d_model;
        Ok(MtpDrafter {
            model,
            ms,
            hprev: vec![0.0; d],
            owed: None,
            proposal: None,
            stats: crate::model::StepStats::default(),
        })
    }

    /// The hidden state of a position, exactly `d_model` wide.
    fn set_hidden(&mut self, hidden: &[f32]) {
        let d = self.model.cfg.d_model;
        self.hprev.clear();
        self.hprev.resize(d, 0.0);
        let n = hidden.len().min(d);
        self.hprev[..n].copy_from_slice(&hidden[..n]);
    }

    /// Feed the block's row for `pos`: the token committed there and the hidden
    /// state of the position before it.
    fn feed_row(&mut self, pos: usize, token: u32, hidden: &[f32], pool: &Pool) {
        let _ = self.model.mtp_forward(
            &mut self.ms,
            pool,
            &[token],
            hidden,
            &[pos],
            false,
            &mut self.stats,
        );
    }
}

#[cfg(feature = "mtp")]
impl Drafter for MtpDrafter {
    fn needs(&self) -> Needs {
        Needs {
            hidden: true,
            ..Needs::default()
        }
    }

    fn propose(
        &mut self,
        anchor: u32,
        anchor_pos: usize,
        k_max: usize,
        pool: &Pool,
        _pick: &mut dyn FnMut(usize, &[f32]) -> u32,
        out: &mut Proposal,
    ) {
        out.clear();
        if k_max == 0 {
            return;
        }
        let d = self.model.cfg.d_model;
        let mut tokens = Vec::with_capacity(2);
        let mut hidden = Vec::with_capacity(2 * d);
        let mut positions = Vec::with_capacity(2);
        // The row an accepted draft left owed comes first: the block's rows
        // must continue its cache in order.
        if let Some(o) = self.owed.take() {
            tokens.push(o.token);
            hidden.extend_from_slice(&o.hidden);
            positions.push(o.pos);
        }
        tokens.push(anchor);
        hidden.extend_from_slice(&self.hprev);
        positions.push(anchor_pos);
        match self.model.mtp_forward(
            &mut self.ms,
            pool,
            &tokens,
            &hidden,
            &positions,
            true,
            &mut self.stats,
        ) {
            Ok(Some(t)) => {
                out.toks.push(t);
                // The block has no confidence head: its single draft is always
                // verified, so it is as good as it gets.
                out.conf.push(1.0);
                self.proposal = Some((anchor_pos, t));
            }
            // Not fatal: the slot decodes plainly from here.
            Ok(None) => {}
            Err(_) => {}
        }
    }

    fn commit(&mut self, pos: usize, token: u32, _taps: &[f32], hidden: &[f32], pool: &Pool) {
        // Was the committed token the drafted one? Then it survived, and the
        // block owes a row for it: the row reads the hidden state of the
        // position before it, which is the last committed one.
        let mut owed = None;
        if let Some((anchor_pos, drafted)) = self.proposal {
            if pos == anchor_pos + 1 {
                if drafted == token {
                    owed = Some(Owed {
                        token,
                        hidden: self.hprev.clone(),
                        pos,
                    });
                }
                self.proposal = None;
            }
        }
        if owed.is_none() && pos == self.ms.kv.filled() {
            // A position no proposal covered: the block's row for it is fed
            // now, as the engine's cache catch-up does.
            let h = self.hprev.clone();
            self.feed_row(pos, token, &h, pool);
        }
        if let Some(o) = owed {
            self.owed = Some(o);
        }
        if !hidden.is_empty() {
            self.set_hidden(hidden);
        }
    }

    fn rollback(&mut self, pos: usize) {
        // The block's cache drops the positions that were not committed.
        self.model.mtp_truncate(&mut self.ms, pos);
        if let Some((anchor_pos, _)) = self.proposal {
            if anchor_pos + 1 >= pos {
                self.proposal = None;
            }
        }
        if self.owed.as_ref().is_some_and(|o| o.pos >= pos) {
            self.owed = None;
        }
    }

    fn reset(&mut self) {
        self.model.mtp_truncate(&mut self.ms, 0);
        self.proposal = None;
        self.owed = None;
        self.set_hidden(&[]);
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn choose_k_prices_expected_tokens_against_verify_cost() {
        let flat = |_w: usize| 1.0;
        // Free verification: verify everything with any chance of surviving.
        assert_eq!(super::choose_k(&[0.9, 0.8, 0.7], flat), 3);
        // Each extra row costs 30% of a step: only drafts likely to survive pay.
        let steep = |w: usize| 1.0 + 0.3 * (w as f64 - 1.0);
        assert_eq!(super::choose_k(&[0.95, 0.9, 0.2], steep), 2);
        assert_eq!(super::choose_k(&[0.1, 0.9, 0.9], steep), 0);
        assert_eq!(super::choose_k(&[], flat), 0);
    }

    #[test]
    fn verify_cost_scales_the_plain_step_by_the_measured_row_cost() {
        let mut v = super::VerifyCost::default();
        assert_eq!(v.estimate(2), None);
        assert!(
            (v.cost(5) - (1.0 + 4.0 * super::ROW_COST)).abs() < 1e-12,
            "the prior before any measurement"
        );
        v.observe(1, 10.0); // a cold first step
        v.observe(1, 4.0);
        v.observe(1, 4.0);
        // Only the plain step is known: prior slope.
        assert!((v.estimate(3).unwrap() - 4.0 * (1.0 + 2.0 * super::ROW_COST)).abs() < 1e-9);
        // Width 2 measured twice: a cold 9.0 then 6.0 → min → 6.0, which is its own value
        // and moves the slope (6/4 − 1 = 0.5) used for unmeasured widths.
        v.observe(2, 9.0);
        v.observe(2, 6.0);
        assert!((v.estimate(2).unwrap() - 6.0).abs() < 1e-9);
        assert!((v.estimate(5).unwrap() - 4.0 * (1.0 + 4.0 * 0.5)).abs() < 1e-9);
        // A width seen once is not trusted: it is still the model.
        v.observe(8, 100.0);
        assert!((v.estimate(8).unwrap() - 4.0 * (1.0 + 7.0 * 0.5)).abs() < 1e-9);
    }
}
