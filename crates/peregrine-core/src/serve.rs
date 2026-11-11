// serving several users from one model, one weight pass at a time.
//
// # Why this exists
//
// A decode step reads every weight once to produce one token. Serving `n` users one after another reads the
// whole model `n` times; serving them in one pass reads it once and does `n` dot products per weight row. On a
// memory-bound machine that is nearly the whole difference between the two. Measured on the reference machine
// (4 cores, 19.5 GB/s, Qwen3-1.7B Q4_K_M, `examples/batch_bench.rs`): 2 users 1.56x, 4 users 2.57x, 8 users
// 2.27x the tokens per second of serving them one at a time. The roofline says why it stops there: Q4_K holds
// about 0.56 bytes per weight, so a batch of `b` rows has an arithmetic intensity near `3.6 b` FLOP/byte, and
// this machine's ridge (460 GFLOP/s over 19.5 GB/s) is near 24 — crossed at about seven rows, after which the
// kernels are compute-bound and another user buys nothing. `DEFAULT_MAX_SLOTS` sits below that knee.
//
// # The loop
//
// One `step` is one token for every user that is ready and, in the same pass, a chunk of the prompts still being
// read (`Caches::Mapped` gives a prompt chunk's rows one cache; before it, a step was one or the other,
// because a chunk and a decode row needed two mutable borrows of the same cache list). Chunking is what keeps a
// long prompt from stopping everyone else (Sarathi-Serve's argument, which matters more here than on a GPU:
// a 2,000-token prompt is about 20 seconds of this machine, and `PEREGRINE_PREFILL_CHUNK` bounds the stall to about one
// second).
//
// # What it does not do
//
// No paged cache: each slot owns a contiguous cache. Paging exists on GPUs because video memory is scarce and
// fragmentation is fatal; here a slot's cache is 112-144 KB per position of ordinary memory, the slot count is
// small, and a block table would cost an indirection on the hottest loop in the engine for nothing. No
// speculation while batching: verifying a draft multiplies the rows per pass, which past the ridge above is
// exactly where there are no spare FLOPs (measured elsewhere: 2.7x at one row falling to 1.6x at eight).

use crate::embed::Pooling;
use crate::kv::{KvCache, KvPrec};
use crate::model::{Caches, Model, State, StepStats};
use crate::pool::Pool;
use crate::prefix::PrefixStore;
use crate::tokenizer::Utf8Stream;
use std::sync::Arc;

/// The mixed schedule (prompt tokens and decode rows in one pass); `PEREGRINE_MIX=0` turns it off.
fn mixed() -> bool {
    use std::sync::OnceLock;
    static M: OnceLock<bool> = OnceLock::new();
    *M.get_or_init(|| crate::options::get().mix)
}

/// The prompt tokens one weight pass carries (`PEREGRINE_PREFILL_CHUNK`).
///
/// The engine prefills a prompt in pieces of this size, so a long prompt is read in several passes.
/// Smaller pieces keep the activation scratch small (it stays in cache) and read the weights more
/// often; the tokens do not depend on it — `crates/peregrine-models/tests/prefill_chunk.rs` pins
/// that. Unset, the chunk follows the pool: [`crate::options::prefill_chunk_default_for`].
fn prefill_chunk(threads: usize) -> usize {
    crate::options::get()
        .prefill_chunk
        .unwrap_or_else(|| crate::options::prefill_chunk_default_for(threads))
}

/// The prompt tokens one weight pass carries when no slot is decoding
/// ([`crate::options::wide_chunk_default_for`], by kernel tier).
///
/// `PEREGRINE_PREFILL_CHUNK` forces one chunk, so an explicit chunk is also the wide one.
fn wide_prefill_chunk(threads: usize) -> usize {
    crate::options::get().prefill_chunk.unwrap_or_else(|| {
        crate::options::wide_chunk_default_for(crate::cpu::resolved())
            .max(crate::options::prefill_chunk_default_for(threads))
    })
}

/// Users served at once by default. Above the machine's roofline ridge (about seven rows) another user adds no
/// throughput, and every user's own tokens arrive more slowly, so this is deliberately below it.
pub const DEFAULT_MAX_SLOTS: usize = 4;

/// The scratch's width in prompt tokens: the width the engine's scratch had before the prefill chunk
/// became an option. The scratch holds the speculation's rows as well as the prompt's, so a narrower
/// chunk must not narrow the drafts.
const SCRATCH_CHUNK: usize = 64;

/// A long prompt is prefetched in larger pieces than a short one (`LONG_PROMPT_CHUNK`).
///
/// The tuned chunk (16) is the best short-prompt setting: at 512 tokens a smaller chunk
/// keeps the activation scratch in cache. On a long prompt the fixed cost of a pass
/// dominates instead — 4096 tokens are 256 passes at 16 — and the measured
/// time-to-first-token at ten users, Llama-3.2-1B, 4096 positions, falls from 158.3 s to
/// 152.3 s with 64-token pieces (llama.cpp v0.9.0 `-fa auto`: 144.2 s), while
/// single-user prefill of the same prompt is no slower (24.0 s against 23.4 s in the A/B
/// that chose it). Neither the tokens nor the logits depend on the chunk
/// (`crates/peregrine-models/tests/prefill_chunk.rs`), so this is scheduling only.
const LONG_PROMPT: usize = 1024;
const LONG_PROMPT_CHUNK: usize = 64;

/// The most non-decoding slots one engine admits at once (S4.9b): a bound on bookkeeping, not on
/// the work. The memory bound is the caches themselves (`WORK_CACHE_BYTES`).
pub const MAX_WORK_SLOTS: usize = 1024;

/// The non-decoding caches one engine may hold at once: a pooled or labels slot's cache is sized
/// to its prompt (S4.9b), so a request's memory is its texts' tokens, not `slots × context`.
///
/// 512 MiB admits 128 short texts whole on a 0.6B model (128 × 16 positions) and still keeps a
/// long batch to a few waves. `PEREGRINE_WORK_CACHE_MB` overrides it.
pub const WORK_CACHE_BYTES: usize = 512 << 20;

/// The rounding of a non-decoding slot's cache: whole prefill chunks, so a short text does not
/// pay for a whole context and a long one does not pay for a second chunk of slack.
fn round_up_prompt(len: usize) -> usize {
    len.max(1).div_ceil(16) * 16
}

/// The memory budget of non-decoding caches: `PEREGRINE_WORK_CACHE_MB`, or
/// [`WORK_CACHE_BYTES`].
fn work_cache_budget() -> u64 {
    crate::options::get()
        .work_cache_mb
        .map_or(WORK_CACHE_BYTES as u64, |mb| (mb as u64) << 20)
}

/// Why a slot stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    /// A stop token, or the end of the turn.
    Stop,
    /// The request's own token limit.
    Length,
    /// The cache has no room left for this conversation.
    Context,
    /// The engine could not run the step.
    Error(String),
}

/// What a slot speculates with (ADR 0007).
///
/// The engine asks the slot's drafter for a proposal and verifies it against
/// the target's own samples (`verify_step`). Which drafter that is, is the
/// request's business: the server asks for none, a conversation may ask for
/// the suffix index or for a DSpark head.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum DrafterChoice {
    /// No drafter: the slot decodes plainly.
    ///
    /// The default, and what the server asks for. The engine's own MTP
    /// speculation is the engine's business, not the slot's, and goes on as
    /// before.
    #[default]
    None,
    /// Suffix-indexed drafting: the longest match of the conversation's tail,
    /// continued by what followed it last time.
    Suffix {
        /// The shortest suffix (in tokens) accepted as a match; 0 disables
        /// drafting.
        min_order: usize,
        /// The most tokens one proposal may hold.
        max_len: usize,
    },
    /// A DSpark drafter directory (`config.json` + `model.safetensors`) trained
    /// for this target.
    Dspark {
        /// The drafter's directory.
        dir: std::path::PathBuf,
        /// The most draft tokens verified per round (the policy may verify
        /// fewer).
        max_k: usize,
    },
}

impl DrafterChoice {
    /// The suffix drafter's own defaults: the shortest match the session uses
    /// and the length of the drafts it proposes.
    pub const SUFFIX_MIN_ORDER: usize = 2;
    /// The most tokens one suffix proposal may hold, as the session proposes.
    pub const SUFFIX_MAX_LEN: usize = 12;
    /// The most DSpark draft tokens verified per round, as the session does.
    pub const DSPARK_MAX_K: usize = 3;
}

/// The drafter a slot asked for, and how deep it may draft.
type AskedDrafter = Option<(Box<dyn crate::draft::Drafter>, usize)>;

/// What a slot's last row is for (ADR 0007: one runtime; roadmap S4.1).
///
/// A chat request samples from its last row and streams tokens. An embedding request pools
/// its text's final hidden states into one vector, and that is the whole answer: it never
/// samples and never decodes. Both are slots in the same engine, prefilled in the same
/// chunks and batched in the same weight passes.
///
/// The enum is the design's `Want` (`docs/design/s4-embeddings-decisions.md` §2.1); the decision
/// variants (`Choice`) arrive with the decision models' read-outs (S4.13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Want {
    /// Sample and stream tokens (today's request).
    #[default]
    Tokens,
    /// Pool the text's final hidden states into one vector and finish when its last prompt
    /// row is computed.
    Pooled(Pooling),
    /// Read the last prompt row's logits at a list of token ids and finish (S4.3): a decision
    /// model's raw score per option.
    Labels,
    /// Read the classifier head's row for the text and finish (S4.6b): a rerank score or a
    /// classifier's outputs, one row of `cls`/`cls.output` per text.
    Head,
}

impl Want {
    /// Whether this slot decodes and samples.
    pub fn is_tokens(self) -> bool {
        matches!(self, Want::Tokens)
    }
    /// Whether this slot is the label read-out.
    pub fn is_labels(self) -> bool {
        matches!(self, Want::Labels)
    }
    /// Whether this slot is the classifier head's read-out.
    pub fn is_head(self) -> bool {
        matches!(self, Want::Head)
    }
    /// The pooling this slot asks for, if it is an embedding slot.
    pub fn pooling(self) -> Option<Pooling> {
        match self {
            Want::Tokens | Want::Labels | Want::Head => None,
            Want::Pooled(p) => Some(p),
        }
    }
    /// What a bidirectional pass reads this slot's text out as: a pooled vector or the head's
    /// row. `None` for a token or labels slot, which a bidirectional model never admits.
    pub fn text_want(self) -> Option<crate::arch::TextWant> {
        match self {
            Want::Pooled(p) => Some(crate::arch::TextWant::Pooled(p)),
            Want::Head => Some(crate::arch::TextWant::Head),
            Want::Tokens | Want::Labels => None,
        }
    }
    /// Whether the slot forwards its whole prompt (a pooled or labels slot) rather than leaving
    /// the last token for the first decode step.
    fn whole_prompt(self) -> bool {
        !self.is_tokens()
    }
}

/// What one request asks of the sampler and when it should stop.
#[derive(Debug, Clone)]
pub struct SlotCfg {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    /// llama.cpp-compatible repetition penalties, all off by default.
    pub penalties: crate::sample::Penalties,
    pub seed: u64,
    pub max_tokens: usize,
    /// Token ids that end the turn.
    pub stop: Vec<u32>,
    /// Report each token's log-probability (for clients that request `logprobs`).
    pub logprobs: bool,
    /// The sampling turn mixed into every position's key (`sample::position_key`).
    ///
    /// 0 — the default — is what the engine has always done: the same prompt
    /// and the same seed draw the same tokens every time, whoever is being
    /// served. A conversation counts its turns instead, so its first turn draws
    /// with turn = 1 and a regenerated reply draws afresh (ADR 0007).
    pub turn: u32,
    /// Whether the slot's KV cache uses the rotated int8 layout.
    ///
    /// `true` — the default — is what the engine has always done. A family
    /// whose keys and queries share a few huge channels measures worse with the
    /// rotation (`Architecture::session_kv_rotation`), and a conversation that
    /// wants that behaviour says so here.
    pub kv_rotation: bool,
    /// The drafter this slot speculates with (ADR 0007).
    ///
    /// `None` — the default — is what the server asks for: the slot decodes
    /// plainly, and the engine's own MTP speculation goes on as it always has.
    /// A slot that names a drafter gets one of its own, built when the slot is
    /// admitted, and the engine verifies its proposals in the same
    /// verification step the MTP block goes through.
    pub drafter: DrafterChoice,
}

impl Default for SlotCfg {
    fn default() -> Self {
        SlotCfg {
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            penalties: crate::sample::Penalties::default(),
            seed: 42,
            max_tokens: 256,
            stop: vec![],
            logprobs: false,
            turn: 0,
            kv_rotation: true,
            drafter: DrafterChoice::None,
        }
    }
}

/// What one request cost, reported with its last token.
#[derive(Debug, Clone, Default)]
pub struct SlotCost {
    /// Tokens in the prompt, and how many of them had to be prefilled (the rest came from a restored prefix).
    pub prompt_tokens: usize,
    pub prefilled: usize,
    pub produced: usize,
    pub prefill_s: f64,
    /// Decode seconds attributed to this request: each step's wall time divided among the rows that shared it.
    pub decode_s: f64,
}

/// What one step of the engine cost, for a caller that reports per token.
///
/// The server counts tokens and bytes and needs nothing more. A conversation
/// does: it reports what each of its tokens cost, and the drafts that were
/// verified and kept (ADR 0007).
#[derive(Debug, Clone, Copy, Default)]
pub struct StepReport {
    /// What the step's forward pass moved.
    pub stats: StepStats,
    /// The step's wall time.
    pub secs: f64,
    /// The rows the step forwarded: more than one means a batched pass.
    pub rows: usize,
    /// The tokens the step emitted.
    pub tokens: usize,
    /// The draft tokens the step verified, and how many of them survived.
    pub drafted: u16,
    pub accepted: u16,
}

/// What a conversation's slot leaves behind when its turn ends.
///
/// The conversation keeps its cache: the next turn reuses the positions it
/// shares with this one. It keeps the tokens the cache holds (the same
/// sequence), and the drafter that drafted for it — a suffix index's corpus
/// and a DSpark head's context belong to the conversation, not to the turn.
pub struct SlotKeep {
    /// The cache: the positions the slot computed.
    pub kv: KvCache,
    /// The tokens the cache holds, position by position.
    pub tokens: Vec<u32>,
    /// The drafter the slot speculated with.
    pub drafter: Option<Box<dyn crate::draft::Drafter>>,
}

/// One token of one request, or the end of it.
#[derive(Debug, Clone)]
pub struct Emission {
    pub slot: SlotId,
    pub token: u32,
    /// The token's text, held back when it is a partial UTF-8 sequence.
    pub text: String,
    pub logprob: f32,
    /// Set on the last emission of a request.
    pub finish: Option<Finish>,
    /// Set with `finish`: what the request cost.
    pub cost: Option<SlotCost>,
    /// A pooled slot's answer: the vector, in the model's width. `None` for a token slot.
    /// Set once, on the slot's only emission.
    pub embedding: Option<Vec<f32>>,
    /// A labels slot's answer: the last prompt row's logits at the slot's label token ids, in
    /// that order. `None` for every other slot. Set once, on the slot's only emission.
    pub label_logits: Option<Vec<f32>>,
    /// A head slot's answer (S4.6b): the classifier head's row for the text, as wide as the
    /// file's head (`cls.output`'s rows, or `d_model` for a one-layer head). `None` for every
    /// other slot. Set once, on the slot's only emission.
    pub head: Option<Vec<f32>>,
}

/// A request's handle while it is being served.
pub type SlotId = u64;

struct Slot {
    id: SlotId,
    kv: KvCache,
    cfg: SlotCfg,
    /// What the slot does with its last row: sample, or pool one vector (S4.1), or read its
    /// logits at a list of tokens (S4.3).
    want: Want,
    /// The token ids a labels slot reads the last prompt row's logits at; empty for every other
    /// slot, so no request pays for it.
    labels: Vec<u32>,
    /// The whole prompt. For a token slot everything but its last token is prefilled; the last token is the
    /// first thing decoded. A pooled slot forwards every token, including the last, and never decodes.
    prompt: Vec<u32>,
    /// Mean pooling's accumulator: `d_model` f64 sums of the text's rows, in position order. Empty for
    /// `Last` and for token slots, so an ordinary request pays nothing for it.
    pool_sum: Vec<f64>,
    /// Prompt tokens already written to the cache.
    fed: usize,
    /// Position the next token will take.
    pos: usize,
    /// The token to feed on the next decode step.
    next: u32,
    produced: usize,
    /// Tokens written to the cache after the prompt (every sampled token is fed back on the next step), so a
    /// finished slot's cache can be described by the token sequence it holds.
    fed_back: Vec<u32>,
    seed: u64,
    utf8: Utf8Stream,
    /// Prompt tokens, how many of them were prefilled (the rest were restored), and the time that took.
    prompt_tokens: usize,
    prefilled: usize,
    prefill_s: f64,
    decode_s: f64,
    /// The slot's own token history and its reused penalty counts. Only touched when
    /// `cfg.penalties` is not off, so an ordinary request pays nothing for it.
    pen: crate::sample::PenaltyState,
    /// The slot's drafter, if one is attached (ADR 0007).
    ///
    /// The engine asks it for a proposal and verifies what it proposes against
    /// the target's own samples (`verify_step`). The engine's own speculation
    /// — the model's MTP block — is such a drafter, attached when the slot is
    /// admitted and the gates allow it.
    ///
    /// It is behind a lock because a slot is shared with the sampling workers,
    /// and a drafter is only `Send`: the lock is never contended, because only
    /// the engine's own thread talks to it.
    drafter: Option<std::sync::Mutex<Box<dyn crate::draft::Drafter>>>,
    /// The most draft tokens the slot's drafter is asked for; 0 — the engine's
    /// own MTP block — leaves the choice to the drafter.
    drafter_depth: usize,
    /// The cache belongs to the caller, not to the engine: a conversation lends
    /// it for its turn and takes it back when the turn ends.
    owned_cache: bool,
    /// S4.9j: the slots this slot's state is copied into once it has forwarded the group's shared
    /// beginning (it leads a decision request's questions). Empty for an ordinary slot.
    fork_to: Vec<SlotId>,
    /// S4.9j: the position the group's shared beginning ends at, and its state is forked there.
    /// 0 for an ordinary slot.
    fork_at: usize,
    /// S4.9j: this slot waits for its group's leader to reach `fork_at` and forwards nothing until
    /// then: its cache is empty, and the fork fills it.
    fork_wait: bool,
    /// S4.9j: room for the shared prefix's snapshot, reserved when the group is admitted so the fork
    /// can hand it to the prefix store without allocating, and the tokens it is kept under.
    fork_snap: Option<crate::kv::KvSnapshot>,
    fork_tokens: Vec<u32>,
}

/// MTP speculation for a lone user; `PEREGRINE_MTP=0` turns it off.
fn mtp_enabled() -> bool {
    crate::options::get().mtp
}

impl Slot {
    /// Whether this slot samples tokens.
    fn wants_tokens(&self) -> bool {
        self.want.is_tokens()
    }

    /// The prompt positions this slot forwards in prefill: a token slot leaves its last token for the
    /// first decode step; a pooled or labels slot forwards all of them.
    ///
    /// A group's leader forwards only the shared beginning first: its slot stops at `fork_at`, the state
    /// is copied into the other questions' slots there, and its own tail follows (S4.9j).
    fn prefill_len(&self) -> usize {
        if self.fork_wait {
            return 0;
        }
        let full = match self.want {
            Want::Tokens => self.prompt.len() - 1,
            Want::Pooled(_) | Want::Labels | Want::Head => self.prompt.len(),
        };
        if self.fed < self.fork_at {
            self.fork_at
        } else {
            full
        }
    }

    /// The prompt is in the cache (`fed` covers everything the slot forwards).
    fn prefill_done(&self) -> bool {
        self.fed >= self.prefill_len()
    }

    /// The slot can decode now: it is a token slot whose prompt is in the cache.
    fn ready(&self) -> bool {
        self.wants_tokens() && self.prefill_done()
    }

    /// Add this step's rows (`rows` is `take × d_model`, in position order) to the mean accumulator.
    fn pool_accumulate(&mut self, hidden: &[f32], d_model: usize) {
        for row in hidden.chunks_exact(d_model) {
            for (s, &x) in self.pool_sum.iter_mut().zip(row) {
                *s += x as f64;
            }
        }
    }

    /// The pooled vector of the whole text. `Last` reads the final row of `last_chunk` (the rows the
    /// completing step computed), `Mean` divides the f64 accumulator once and rounds once.
    ///
    /// `last_chunk` is the completing step's hidden rows for this slot; it is used by `Last` only.
    fn pool_finish(&self, last_chunk: &[f32], d_model: usize, pooling: Pooling) -> Vec<f32> {
        match pooling {
            Pooling::Last => {
                let start = (last_chunk.len() / d_model - 1) * d_model;
                last_chunk[start..start + d_model].to_vec()
            }
            Pooling::Mean => {
                let inv = 1.0 / self.prompt.len().max(1) as f64;
                self.pool_sum.iter().map(|s| (s * inv) as f32).collect()
            }
            // Refused at admission; a causal decoder's row 0 sees one token and is not a text's vector.
            Pooling::Cls => Vec::new(),
        }
    }

    /// Rebuild this slot's penalty window before its next token is sampled. A no-op for
    /// the default, penalty-free request.
    fn prepare_penalties(&mut self, n_vocab: usize) {
        let pen = self.cfg.penalties;
        if !pen.is_off() {
            self.pen.prepare(pen.last_n, n_vocab);
        }
    }
}

/// The prompt text of one chat turn: the system block, the earlier exchanges, this turn's message, and the
/// assistant's opening (plus whatever the assistant has already written, for a continuation).
///
/// One builder for both paths, so a prompt served by the batching engine is the same string a single session
/// would have built, token for token — and so a continuation restores the cache of the turn it continues.
pub fn chat_prompt(
    template: crate::template::Template,
    system: Option<&str>,
    pairs: &[(String, String)],
    user: &str,
    thinking: bool,
    assistant_prefix: Option<&str>,
) -> String {
    template.render(
        system,
        pairs,
        user,
        &crate::template::today_llama3(),
        thinking,
        assistant_prefix,
    )
}

/// Several users, one model, one weight pass per step.
pub struct Engine {
    model: Arc<Model>,
    pool: Pool,
    // the narrower pool for one-row steps (`pool::narrow_pool`); None: use `pool`.
    pool1: Option<Pool>,
    state: State,
    slots: Vec<Slot>,
    max_slots: usize,
    max_ctx: usize,
    kv_prec: KvPrec,
    /// The prompt tokens one weight pass carries: the engine prefills a prompt in pieces of this
    /// size (`PEREGRINE_PREFILL_CHUNK`).
    prefill_chunk: usize,
    /// The prompt tokens one pass carries when no slot is decoding (S5.1): the same read, wider,
    /// because no decoding user is waiting behind it.
    wide_chunk: usize,
    /// The rows one weight pass carries for work that never decodes: pooled and label slots
    /// (S4.9b, `PEREGRINE_PASS_ROWS`).
    pass_rows: usize,
    /// Bytes of non-decoding caches the engine admits at once (S4.9b).
    work_cache_bytes: u64,
    /// Caches of finished requests, so a prompt that shares a beginning with one of them (the system prompt
    /// every user of a deployment sends, the transcript a code agent extends) is not prefilled again. Shared
    /// across slots, which is where it pays most: one user's work serves the next.
    prefixes: PrefixStore,
    next_id: SlotId,
    /// Engine-wide totals, for the server's telemetry.
    pub stats: StepStats,
    /// Decode steps run, and the rows they carried: their ratio is the batching actually achieved.
    pub decode_steps: u64,
    pub decode_rows: u64,
    // one sampler scratch per pool worker: the rows of a step are sampled in parallel.
    samps: Vec<std::sync::Mutex<crate::sample::Scratch>>,
    /// The model's MTP block, when it has one and speculation is on: the
    /// engine's own drafter, attached to the slot that speculates.
    mtp_available: bool,
    /// The slot that speculates: the one whose drafter's proposals the engine
    /// verifies. `None` when nothing does.
    ///
    /// Speculation is single-slot (ADR 0007), so there is never more than one.
    spec_slot: Option<SlotId>,
    /// The verify-cost model: what a verification pass of a given width costs
    /// the engine. It measures the engine's own forward passes, so it lives
    /// here (ADR 0007).
    vcost: crate::draft::VerifyCost,
    /// The rows of logits the state holds: the anchor's and the drafted
    /// tokens'. A verification pass wants one row per row it forwards.
    max_logit_rows: usize,
    /// The proposal the drafter writes into: the engine's own buffer, reused,
    /// so a draft costs no allocation.
    proposal: crate::draft::Proposal,
    /// The verification pass's own rows: the anchor and the drafted tokens,
    /// their positions and the rows that want logits. The engine's buffers,
    /// reused.
    v_toks: Vec<u32>,
    v_pos: Vec<usize>,
    v_want: Vec<usize>,
    v_map: Vec<usize>,
    /// The target's own samples of the verification pass's rows, and their
    /// log-probabilities.
    v_samples: Vec<u32>,
    v_logprobs: Vec<f32>,
    /// The target layers the attached drafters need copied out of the forward pass.
    /// Empty: no drafter is attached, and the forward copies nothing.
    tap_layers: Vec<usize>,
    /// MTP drafts verified, and how many were kept.
    pub spec_drafts: u64,
    pub spec_hits: u64,
    /// The draft tokens verified: how many the engine asked the target to
    /// check, over every round.
    pub spec_tokens: u64,
    /// The rounds in which a drafter was asked for a proposal, whether or not
    /// the price said to verify any of it.
    pub spec_rounds: u64,
    /// What the last step cost: the bytes it moved, the rows it forwarded, the
    /// tokens it emitted and the drafts it verified.
    ///
    /// A conversation (one slot) reads this after every step, and reports what
    /// each of its tokens cost. With several slots the report covers the whole
    /// pass, not one slot's share of it.
    pub last_step: StepReport,
    /// What a conversation's slot left behind when its turn ended: the cache
    /// it keeps, the tokens it holds and the drafter that drafted for it.
    kept: Option<SlotKeep>,
    /// The decode steps run, for the suffix index's probe rule: a confidence
    /// that is never tested never recovers.
    steps: u64,
    /// Plain steps left before drafting is probed again (the DSpark head's
    /// parking rule).
    park: u32,
    /// The measured token rate of drafting rounds, against which a plain step
    /// is priced.
    draft_rate: Option<f64>,
    /// The drafting rounds measured.
    drafted_rounds: u64,
}

/// S4.9j: what a group admission asks of one question's slot.
#[derive(Clone, Copy)]
struct Fork {
    /// The position the group's shared beginning ends at (the state is forked there).
    at: usize,
    /// The slot waits for the leader to reach `at` and forwards nothing until then.
    wait: bool,
}

impl Engine {
    /// The engine with the process's prefill chunk (`PEREGRINE_PREFILL_CHUNK`, or the pool's default).
    pub fn new(
        model: Arc<Model>,
        threads: usize,
        max_ctx: usize,
        max_slots: usize,
        kv_prec: KvPrec,
        prefix_cache_bytes: usize,
    ) -> Engine {
        Self::build(
            model,
            threads,
            max_ctx,
            max_slots,
            kv_prec,
            prefix_cache_bytes,
            prefill_chunk(threads),
            wide_prefill_chunk(threads),
            None,
        )
    }

    /// The engine with a chosen prefill chunk: the prompt tokens one weight pass carries.
    ///
    /// The chunk is how the prompt is *read*, not what it produces: the tokens and the logits
    /// do not depend on it (`crates/peregrine-models/tests/prefill_chunk.rs` pins that). It is
    /// clamped to the matmul scratch's row limit, and a mixed step takes fewer rows when the
    /// other users' decode rows need the room. A chosen chunk is one chunk: the wide pass of a
    /// step with no decoder is this one too.
    pub fn new_with_chunk(
        model: Arc<Model>,
        threads: usize,
        max_ctx: usize,
        max_slots: usize,
        kv_prec: KvPrec,
        prefix_cache_bytes: usize,
        prefill_chunk: usize,
    ) -> Engine {
        Self::new_with_chunk_and_pass(
            model,
            threads,
            max_ctx,
            max_slots,
            kv_prec,
            prefix_cache_bytes,
            prefill_chunk,
            None,
        )
    }

    /// The engine with a chosen prefill chunk and pass budget (S4.9b).
    ///
    /// `pass_rows` is how many rows a weight pass may carry for work that never decodes: pooled
    /// (embedding) and label (decision) slots fill a pass together up to it, instead of one
    /// prefill chunk per slot. `None` takes `PEREGRINE_PASS_ROWS`, or the machine's default
    /// (`crate::options::pass_rows_default_for`). It is scheduling only: a text's vector and a
    /// question's label logits do not depend on it.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_chunk_and_pass(
        model: Arc<Model>,
        threads: usize,
        max_ctx: usize,
        max_slots: usize,
        kv_prec: KvPrec,
        prefix_cache_bytes: usize,
        prefill_chunk: usize,
        pass_rows: Option<usize>,
    ) -> Engine {
        Self::build(
            model,
            threads,
            max_ctx,
            max_slots,
            kv_prec,
            prefix_cache_bytes,
            prefill_chunk,
            prefill_chunk,
            pass_rows,
        )
    }

    /// The engine with a narrow chunk and a wide one (S5.1).
    ///
    /// `prefill_chunk` is the chunk of a pass that carries a decode row beside the prompt, and
    /// `wide_chunk` the chunk of a pass that carries none: with nobody decoding, the narrow chunk
    /// protects no one and the wider pass is faster. Both are how a prompt is read, not what it
    /// produces (`crates/peregrine-models/tests/prefill_chunk.rs` pins that a prompt read in
    /// pieces of different widths gives the same tokens).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_chunks_and_pass(
        model: Arc<Model>,
        threads: usize,
        max_ctx: usize,
        max_slots: usize,
        kv_prec: KvPrec,
        prefix_cache_bytes: usize,
        prefill_chunk: usize,
        wide_chunk: usize,
        pass_rows: Option<usize>,
    ) -> Engine {
        Self::build(
            model,
            threads,
            max_ctx,
            max_slots,
            kv_prec,
            prefix_cache_bytes,
            prefill_chunk,
            wide_chunk.max(prefill_chunk),
            pass_rows,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        model: Arc<Model>,
        threads: usize,
        max_ctx: usize,
        max_slots: usize,
        kv_prec: KvPrec,
        prefix_cache_bytes: usize,
        prefill_chunk: usize,
        wide_chunk: usize,
        pass_rows: Option<usize>,
    ) -> Engine {
        let c = model.cfg;
        let max_slots = max_slots.max(1);
        // A bidirectional model schedules whole texts, never chunks, and writes no KV
        // cache: the scratch must hold the longest accepted text (the engine's context)
        // and the budget of one step is that many rows (design §2.2).
        let bidirectional = model.causality() == crate::arch::Causality::Bidirectional;
        // A small float-only encoder (a converted Q8_0 one included) takes whole-text
        // passes of `SMALL_FLOAT_ENCODER_BYTES` rows (S4.9m).
        let float_only = !model.act_layout().quantize;
        // The prompt tokens one weight pass carries.
        let prefill_chunk = prefill_chunk.clamp(1, crate::options::MAX_PREFILL_CHUNK);
        // The same, for a pass with no decode row beside it (S5.1): never narrower than the
        // chunk, never wider than the scratch's row limit.
        let wide_chunk = wide_chunk.clamp(prefill_chunk, crate::options::MAX_PREFILL_CHUNK);
        // The rows one non-decoding pass carries (S4.9b). A bidirectional model schedules
        // whole texts, so its budget is the same number but a longer text still runs alone.
        let budget_rows = pass_rows
            .or(crate::options::get().pass_rows)
            .unwrap_or_else(|| {
                crate::options::pass_rows_default_for(
                    c.d_model,
                    model.bytes_per_param(),
                    bidirectional,
                    model.weight_bytes(),
                    float_only,
                )
            })
            .clamp(1, crate::options::MAX_PASS_ROWS);
        let work_cache_bytes = work_cache_budget();
        // One position more than the model's own context is allowed: the engine
        // stops a turn one position short of a full cache, so a caller that
        // wants the whole context asks for one more (ADR 0007: a conversation
        // runs on a one-slot engine).
        let max_ctx = max_ctx.clamp(32, c.ctx_train + 1);
        // The widest step is a prefill chunk beside one row per slot (a mixed step) and, since
        // S4.9b, a non-decoding pass of up to `pass_rows` rows beside the chat rows. Every decode
        // row wants logits, a prefill chunk wants none. The scratch keeps the width it had before
        // the chunk became an option, because it also holds the speculation's rows: a narrower
        // chunk must not narrow the drafts. The matmul driver splits a batch wider than its own
        // row limit into blocks, so this is bounded by the pass budget, not by that limit.
        let chat_rows = prefill_chunk
            .max(wide_chunk)
            .max(SCRATCH_CHUNK)
            .max(LONG_PROMPT_CHUNK)
            + max_slots;
        let max_batch = if bidirectional {
            max_ctx
        } else {
            budget_rows + chat_rows
        };
        // An MTP verification pass wants logits for two rows, whatever the slot count. An
        // encoder never samples: one row is all its state ever needs.
        let use_mtp = !bidirectional && model.has_mtp() && mtp_enabled() && mixed();
        let max_logit_rows = if bidirectional {
            1
        } else if use_mtp {
            max_slots.max(2)
        } else {
            max_slots
        };
        let mut state = model.new_state(max_batch, max_logit_rows, max_ctx, threads);
        // A pooled slot asks for every row's final hidden state, and a decode step must not allocate:
        // the buffer is sized once, here. (The hybrid family resizes it per forward, which keeps
        // the capacity.)
        state.hidden.resize(max_batch * c.d_model, 0.0);
        if bidirectional {
            // No cache is ever written, so the attention scratch is not needed. The
            // pooling accumulator is the family's, per text.
            state.attn = crate::blocks::attn::AttnScratch::new(&c, 1, threads);
            // The strip's panel (one head's keys per text) and one score strip per
            // worker, sized for the longest pass this engine admits (S4.9g).
            state.attn.bidir.resize(max_batch, c.head_dim, threads);
            state.pool_sum = vec![0.0; c.d_model];
        }
        Engine {
            spec_drafts: 0,
            spec_hits: 0,
            spec_tokens: 0,
            spec_rounds: 0,
            last_step: StepReport::default(),
            kept: None,
            model,
            pool: Pool::new(threads),
            pool1: crate::pool::narrow_pool(threads),
            state,
            slots: Vec::with_capacity(max_slots),
            max_slots,
            max_ctx,
            kv_prec,
            prefill_chunk,
            wide_chunk,
            pass_rows: budget_rows,
            work_cache_bytes,
            prefixes: PrefixStore::new(prefix_cache_bytes),
            next_id: 1,
            stats: StepStats::default(),
            decode_steps: 0,
            decode_rows: 0,
            tap_layers: Vec::new(),
            samps: (0..threads.max(1))
                .map(|_| std::sync::Mutex::new(crate::sample::Scratch::default()))
                .collect(),
            // The engine's own drafter needs the MTP feature; without it the
            // engine speculates with nothing.
            mtp_available: use_mtp && cfg!(feature = "mtp"),
            spec_slot: None,
            vcost: crate::draft::VerifyCost::default(),
            max_logit_rows,
            proposal: crate::draft::Proposal::default(),
            v_toks: Vec::new(),
            v_pos: Vec::new(),
            v_want: Vec::new(),
            v_map: Vec::new(),
            v_samples: Vec::new(),
            v_logprobs: Vec::new(),
            steps: 0,
            park: 0,
            draft_rate: None,
            drafted_rounds: 0,
        }
    }

    /// Whether a lone user's decode speculates with the model's MTP block.
    pub fn speculates(&self) -> bool {
        self.mtp_available
    }
    /// Turn MTP speculation off (for comparisons; `PEREGRINE_MTP=0` does it at startup).
    pub fn disable_mtp(&mut self) {
        self.mtp_available = false;
        self.stop_speculation();
    }

    /// Turn MTP speculation back on when this engine was built able to run it
    /// (`peregrine bench --mtp`). Does nothing when the model has no MTP block,
    /// the build has no `mtp` feature, or the state was allocated without the
    /// second logit row a verify pass needs.
    pub fn enable_mtp(&mut self) {
        if cfg!(feature = "mtp") && self.model.has_mtp() && mixed() && self.max_logit_rows >= 2 {
            self.mtp_available = true;
        }
    }

    pub fn max_slots(&self) -> usize {
        self.max_slots
    }
    pub fn active(&self) -> usize {
        self.slots.len()
    }
    /// Whether a chat (token) slot may be admitted: `--slots` bounds chat slots and only chat
    /// slots. A pooled or label slot asks [`Self::work_room_for`] instead (S4.9b).
    pub fn has_room(&self) -> bool {
        self.token_slots() < self.max_slots
    }
    /// The chat slots in flight.
    fn token_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.wants_tokens()).count()
    }
    /// The non-decoding slots in flight (pooled and label slots).
    fn work_slots(&self) -> usize {
        self.slots.len() - self.token_slots()
    }
    /// The rows of non-decoding work in flight: the tokens of every pooled and label slot's
    /// prompt, which is what the pass budget and the caches are spent on.
    pub fn work_rows(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !s.wants_tokens())
            .map(|s| s.prompt.len())
            .sum()
    }
    /// The bytes of the non-decoding slots' caches in flight (S4.9b).
    pub fn work_cache_used(&self) -> u64 {
        self.slots
            .iter()
            .filter(|s| !s.wants_tokens())
            .map(|s| s.kv.allocated_bytes())
            .sum()
    }
    /// Whether one more non-decoding slot of `prompt_len` tokens fits the work budget.
    ///
    /// The budget is the memory of the slot's own prompt-sized cache, not `--slots`: a request's
    /// texts do not consume chat slots and are not bounded by their number. An encoder's slots
    /// allocate no cache, so only the slot cap bounds them.
    pub fn work_room_for(&self, prompt_len: usize) -> bool {
        if self.work_slots() >= MAX_WORK_SLOTS {
            return false;
        }
        self.work_cache_used() + self.cache_bytes_for_prompt(prompt_len) <= self.work_cache_bytes
    }
    /// How many of `lens`, in order, fit the work budget if admitted now.
    pub fn work_room_prefix(&self, lens: &[usize]) -> usize {
        let mut used = self.work_cache_used();
        let room = MAX_WORK_SLOTS.saturating_sub(self.work_slots());
        let mut n = 0;
        for &len in lens {
            if n >= room {
                break;
            }
            let bytes = self.cache_bytes_for_prompt(len);
            if used + bytes > self.work_cache_bytes {
                break;
            }
            used += bytes;
            n += 1;
        }
        n
    }
    /// Bytes a pooled or label slot's cache holds for a prompt of `len` tokens: the prompt,
    /// rounded up to whole prefill chunks, not the model's whole context (S4.9b).
    pub fn cache_bytes_for_prompt(&self, len: usize) -> u64 {
        if self.model.causality() == crate::arch::Causality::Bidirectional {
            return 0;
        }
        let c = self.model.cfg;
        KvCache::new(self.kv_prec, c.kv_layers(), c.n_kv_head, c.head_dim, 1).bytes_per_position()
            * round_up_prompt(len) as u64
            + self.model.recurrent_shape().map_or(0, |r| {
                (r.n_layers * (r.state_floats + r.conv_floats) * 4) as u64
            })
    }
    /// The bytes of one admitted slot's cache, for a test or a report (S4.9b).
    pub fn slot_cache_bytes(&self, id: SlotId) -> Option<u64> {
        self.slots
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.kv.allocated_bytes())
    }
    /// Bytes of cache one prompt position costs (0 for an encoder).
    pub fn cache_bytes_per_position(&self) -> u64 {
        if self.model.causality() == crate::arch::Causality::Bidirectional {
            return 0;
        }
        let c = self.model.cfg;
        KvCache::new(self.kv_prec, c.kv_layers(), c.n_kv_head, c.head_dim, 1).bytes_per_position()
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    /// The rows one non-decoding pass carries (`PEREGRINE_PASS_ROWS`, or the machine's default).
    pub fn pass_rows(&self) -> usize {
        self.pass_rows
    }
    /// The target layers the engine copies out of the forward pass for the
    /// attached drafters. Empty: no drafter is attached, and nothing is copied.
    pub fn tap_layers(&self) -> &[usize] {
        &self.tap_layers
    }
    /// Bytes one slot's cache holds, for sizing a deployment.
    pub fn cache_bytes_per_slot(&self) -> u64 {
        // An encoder writes no cache: its slots are bounded by the text, not by a cache.
        if self.model.causality() == crate::arch::Causality::Bidirectional {
            return 0;
        }
        let c = self.model.cfg;
        KvCache::new(self.kv_prec, c.kv_layers(), c.n_kv_head, c.head_dim, 1).bytes_per_position()
            * self.max_ctx as u64
            // a hybrid model's recurrent state, per slot, whatever the context length.
            + self.model.recurrent_shape().map_or(0, |r| (r.n_layers * (r.state_floats + r.conv_floats) * 4) as u64)
    }

    /// Take a request, or give it back when every slot is busy or the prompt cannot fit.
    pub fn admit(&mut self, prompt: Vec<u32>, cfg: SlotCfg) -> Result<SlotId, (Vec<u32>, String)> {
        self.admit_inner(prompt, cfg, Want::Tokens, None, Vec::new(), None)
    }

    /// Admit an embedding request (S4.1): the slot is prefilled like any prompt, beside the
    /// others, and finishes when its last prompt row is computed. It never samples or decodes.
    ///
    /// `pooling` is the reduction the vector is read with; the file's own type is the default
    /// (`Model::pooling`), and the caller passes the type the request asked for.
    pub fn admit_pooled(
        &mut self,
        prompt: Vec<u32>,
        cfg: SlotCfg,
        pooling: Pooling,
    ) -> Result<SlotId, (Vec<u32>, String)> {
        self.admit_inner(prompt, cfg, Want::Pooled(pooling), None, Vec::new(), None)
    }

    /// Admit a decision model's label read-out (S4.3): the slot is prefilled like a pooled slot
    /// and finishes when its last prompt row is computed. That emission's `label_logits` holds
    /// the row's logits at `labels`, in that order; the slot never samples or decodes.
    ///
    /// Questions of one request share a prompt prefix, so a labels slot consults the prefix
    /// store like a token slot does: only its final row is read, and a restored prefix's rows are
    /// exactly the rows that were computed (the store copies the cache's bytes).
    pub fn admit_labels(
        &mut self,
        prompt: Vec<u32>,
        cfg: SlotCfg,
        labels: Vec<u32>,
    ) -> Result<SlotId, (Vec<u32>, String)> {
        self.admit_inner(prompt, cfg, Want::Labels, None, labels, None)
    }

    /// Admit a classifier-head read-out (S4.6b): a rerank pair or a classifier's text. The slot
    /// is prefilled like a pooled slot and finishes when its last prompt row is computed; its
    /// emission's `head` holds the head's row for the text (`cls` → activation → `cls.output`).
    ///
    /// A file without a head is refused by name, and so is a causal model: the head is the
    /// bidirectional encoder's read-out.
    pub fn admit_head(
        &mut self,
        prompt: Vec<u32>,
        cfg: SlotCfg,
    ) -> Result<SlotId, (Vec<u32>, String)> {
        self.admit_inner(prompt, cfg, Want::Head, None, Vec::new(), None)
    }

    /// Admit a decision request's questions as one group (S4.9j): the longest common beginning of
    /// their prompts is forwarded **once**, in the first question's slot, and its state is copied
    /// into every other question's slot at that position. Each then forwards only its own tail, and
    /// all tails share weight passes.
    ///
    /// `prompts` are the questions' token ids and `labels` their read-out ids, in request order. The
    /// group needs a free slot per question; a request with more questions than slots is admitted in
    /// several groups (each re-evaluates — or restores — the shared beginning). When the common
    /// beginning is shorter than [`crate::prefix::MIN_PREFIX`], the questions are admitted
    /// independently, exactly as [`Self::admit_labels`] does, because the fork would buy nothing.
    ///
    /// The shared beginning is also offered to the prefix store when the group forks (if the store
    /// has a budget), so the next request with the same state restores it instead of forwarding it.
    ///
    /// On `Err` no slot has been admitted.
    pub fn admit_labels_group(
        &mut self,
        prompts: Vec<Vec<u32>>,
        cfg: SlotCfg,
        labels: Vec<Vec<u32>>,
    ) -> Result<Vec<SlotId>, String> {
        self.admit_labels_group_from(prompts, cfg, labels, crate::prefix::MIN_PREFIX)
    }

    /// [`Self::admit_labels_group`] with the shortest common beginning worth a fork.
    ///
    /// The endpoint keeps the store's threshold; a test (or a measurement) that pins the fork at a
    /// position below it passes 1 here.
    pub fn admit_labels_group_from(
        &mut self,
        prompts: Vec<Vec<u32>>,
        cfg: SlotCfg,
        labels: Vec<Vec<u32>>,
        min_prefix: usize,
    ) -> Result<Vec<SlotId>, String> {
        let n = prompts.len();
        if n == 0 {
            return Err("a group needs at least one question".into());
        }
        if labels.len() != n {
            return Err(format!("{n} prompts but {} label lists", labels.len()));
        }
        for (prompt, label) in prompts.iter().zip(&labels) {
            self.check_request(prompt, Want::Labels, label)?;
        }
        // A decision group is admitted by the non-decoding budget, not by `--slots`: every
        // question is a label slot with its own prompt-sized cache (S4.9b).
        let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
        let room = self.work_room_prefix(&lens);
        if n > room {
            return Err(format!(
                "a group of {n} questions needs {n} places in the embedding budget, {room} are free"
            ));
        }
        // The longest common beginning. It stops one row short of the shortest question: every
        // question must keep a row of its own to perform the read-out (a labels slot's answer is its
        // last prompt row's logits).
        let mut at = prompts
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or(0)
            .saturating_sub(1);
        for q in &prompts[1..] {
            at = at.min(crate::prefix::common_prefix(&prompts[0], q));
        }
        if n == 1 || at < min_prefix {
            // Too little is shared to be worth a fork: admit them as separate slots, as before.
            let mut ids = Vec::with_capacity(n);
            for (prompt, label) in prompts.into_iter().zip(labels) {
                match self.admit_labels(prompt, cfg.clone(), label) {
                    Ok(id) => ids.push(id),
                    Err((_, e)) => {
                        self.slots.retain(|s| !ids.contains(&s.id));
                        return Err(e);
                    }
                }
            }
            return Ok(ids);
        }
        let leader_idx = self.slots.len();
        let mut ids: Vec<SlotId> = Vec::with_capacity(n);
        let mut questions = prompts.into_iter().zip(labels);
        let (first, first_labels) = questions.next().expect("a non-empty group");
        let leader = match self.admit_inner(
            first,
            cfg.clone(),
            Want::Labels,
            None,
            first_labels,
            Some(Fork { at, wait: false }),
        ) {
            Ok(id) => id,
            Err((_, e)) => return Err(e),
        };
        ids.push(leader);
        for (prompt, label) in questions {
            match self.admit_inner(
                prompt,
                cfg.clone(),
                Want::Labels,
                None,
                label,
                Some(Fork { at, wait: true }),
            ) {
                Ok(id) => ids.push(id),
                Err((_, e)) => {
                    self.slots.retain(|s| !ids.contains(&s.id));
                    return Err(e);
                }
            }
        }
        // Keep the shared beginning for the next request, if it is not there already. The snapshot's
        // room is taken now, so the fork itself copies into buffers the admission allocated.
        let uncovered = !self.prefixes.covers(&self.slots[leader_idx].prompt[..at]);
        let bytes = self.slots[leader_idx].kv.snapshot_bytes(at);
        if at >= crate::prefix::MIN_PREFIX && uncovered && bytes <= self.prefixes.budget() {
            let snap = self.slots[leader_idx].kv.snapshot_reserve(at);
            let tokens = self.slots[leader_idx].prompt[..at].to_vec();
            self.slots[leader_idx].fork_snap = Some(snap);
            self.slots[leader_idx].fork_tokens = tokens;
            // Both the snapshot's buffers and the store's next entry are made here, so the fork
            // itself (in a step) allocates nothing.
            self.prefixes.reserve(1);
        }
        self.slots[leader_idx].fork_to = ids[1..].to_vec();
        // A leader that restored the whole beginning from the store is already at the fork position.
        self.fork_groups();
        Ok(ids)
    }

    /// Admit a slot that runs on the caller's own cache (ADR 0007).
    ///
    /// A conversation's cache outlives its turn: the next turn reuses the
    /// positions it shares with the previous one. The engine takes the cache
    /// for the turn and hands it back when the turn ends (`take_kept`), and
    /// the engine's own prefix store is not consulted — the caller manages its
    /// own reuse, and a drafter's corpus starts at position 0 anyway.
    ///
    /// A request that is refused gives the cache back too: the engine keeps it
    /// until the caller takes it, because the turn did not start and the
    /// conversation still holds what it held.
    pub fn admit_with_cache(
        &mut self,
        prompt: Vec<u32>,
        cfg: SlotCfg,
        kv: KvCache,
    ) -> Result<SlotId, (Vec<u32>, String)> {
        self.admit_inner(prompt, cfg, Want::Tokens, Some(kv), Vec::new(), None)
    }

    /// What a conversation's slot left behind when its turn ended: the cache it
    /// keeps, the tokens it holds and the drafter that drafted for it.
    ///
    /// `None` when the last slot was not a conversation's (the server's slots
    /// keep their caches in the prefix store instead), or when the request was
    /// refused before it started.
    pub fn take_kept(&mut self) -> Option<SlotKeep> {
        self.kept.take()
    }

    /// The engine's worker pool, for a caller that runs a forward pass of its
    /// own (a conversation's benchmark, the verification-width probe).
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// The narrow pool for one-row steps, when there is one.
    pub fn narrow_pool(&self) -> Option<&Pool> {
        self.pool1.as_ref()
    }

    /// Refuse a request: the caller's cache goes back to it, held until the
    /// caller takes it.
    fn refuse(
        &mut self,
        prompt: Vec<u32>,
        cache: Option<KvCache>,
        e: String,
    ) -> Result<SlotId, (Vec<u32>, String)> {
        if let Some(kv) = cache {
            self.kept = Some(SlotKeep {
                kv,
                tokens: Vec::new(),
                drafter: None,
            });
        }
        Err((prompt, e))
    }

    /// The checks every request passes before a slot exists (S4.9j: a group checks every question
    /// first, so it either starts all of them or none).
    fn check_request(&self, prompt: &[u32], want: Want, labels: &[u32]) -> Result<(), String> {
        if prompt.is_empty() {
            return Err("the prompt is empty".into());
        }
        let bidirectional = self.model.causality() == crate::arch::Causality::Bidirectional;
        // An encoder has no decode path at all: a chat request on it is refused by name
        // (design §3). Everything it serves is a whole text, so it is pooled.
        if bidirectional && want.is_tokens() {
            return Err(format!(
                "{} is an embedding model and cannot generate text; use /v1/embeddings (or `peregrine embed`)",
                self.model.cfg.arch_name
            ));
        }
        // A labels slot reads one prompt row's logits at a few token ids (S4.3). That is a
        // decoder's read-out; an encoder has no decode path and no such rows, so it is refused
        // by name, as a token slot is above (design §3).
        if bidirectional && want.is_labels() {
            return Err(format!(
                "{} is an embedding model and cannot read label logits; use /v1/embeddings (or `peregrine embed`)",
                self.model.cfg.arch_name
            ));
        }
        // A pooled slot pools the rows the model computed; `Cls` needs an encoder (S4.4 onward),
        // and a causal decoder's row 0 is one token's view, not a text's vector.
        if let Want::Pooled(Pooling::Cls) = want {
            if !bidirectional {
                return Err("cls pooling is not implemented for causal models".into());
            }
        }
        // A head slot reads the classifier head's row: it needs a file that has a head, and a
        // bidirectional pass to compute the row in (S4.6b).
        if want.is_head() {
            if !bidirectional {
                return Err(format!(
                    "{} is a causal model: its classifier head is not served yet",
                    self.model.cfg.arch_name
                ));
            }
            if !self.model.has_head() {
                return Err(format!(
                    "{} has no classifier head (cls / cls.output), so it cannot rerank or classify",
                    self.model.cfg.arch_name
                ));
            }
        }
        if want.is_labels() {
            if labels.is_empty() {
                return Err("a labels slot needs a label token".into());
            }
            if let Some(&l) = labels
                .iter()
                .find(|&&l| l as usize >= self.model.cfg.n_vocab)
            {
                return Err(format!(
                    "label token {l} is outside the vocabulary of {}",
                    self.model.cfg.n_vocab
                ));
            }
        }
        // A token slot needs one position after its prompt for the first decode step; a pooled or
        // labels slot forwards the prompt itself and needs no more.
        let extra = usize::from(!want.whole_prompt());
        if prompt.len() + extra > self.max_ctx {
            return Err(format!(
                "prompt of {} tokens does not fit in {} positions",
                prompt.len(),
                self.max_ctx
            ));
        }
        Ok(())
    }

    fn admit_inner(
        &mut self,
        prompt: Vec<u32>,
        cfg: SlotCfg,
        want: Want,
        cache: Option<KvCache>,
        labels: Vec<u32>,
        forking: Option<Fork>,
    ) -> Result<SlotId, (Vec<u32>, String)> {
        // Chat slots are bounded by `--slots`; work that never decodes is admitted by its own
        // memory budget, so an embedding or decision request never waits for a chat slot (S4.9b).
        if want.is_tokens() && !self.has_room() {
            return self.refuse(prompt, cache, "every slot is busy".into());
        }
        if !want.is_tokens() && !self.work_room_for(prompt.len()) {
            return self.refuse(
                prompt,
                cache,
                format!(
                    "the embedding queue is full: {} slots and {} MiB of prompt caches are in flight",
                    self.work_slots(),
                    self.work_cache_used() >> 20
                ),
            );
        }
        let bidirectional = self.model.causality() == crate::arch::Causality::Bidirectional;
        if let Err(e) = self.check_request(&prompt, want, &labels) {
            return self.refuse(prompt, cache, e);
        }
        let whole = want.whole_prompt();
        // The last prompt token is never prefilled (it is what the first decode step feeds), so a prefix may
        // cover everything before it. A group's leader may only restore what the whole group shares.
        let cap = forking.map_or(prompt.len() - 1, |f| f.at);
        // The drafter the slot asks for, built for this slot. It is built
        // before the prefix store is consulted: a drafter's corpus and context
        // start at position 0, so a restored prefix would leave them short.
        let asked = match self.make_asked_drafter(&cfg.drafter) {
            Ok(d) => d,
            Err(e) => return self.refuse(prompt, cache, e),
        };
        // A labels slot has no token to draft: it reads one row and stops.
        if want.is_labels() && asked.is_some() {
            return self.refuse(prompt, cache, "a labels slot does not decode".into());
        }
        let id = self.next_id;
        self.next_id += 1;
        // The caller's cache is already the one it wants: its rotation, its
        // positions. The engine's own cache is made here.
        let given = cache.is_some();
        let mut kv = match cache {
            Some(kv) => {
                let held = kv.filled();
                if held > cap {
                    return self.refuse(
                        prompt,
                        Some(kv),
                        format!("the cache holds {held} positions, more than the prompt's {cap}"),
                    );
                }
                kv
            }
            None if bidirectional => {
                // An encoder never writes a cache; a zero-position one is a marker, not
                // an allocation (design §2.2, and the slot-memory limit S4.1 found).
                self.model.new_cache(self.kv_prec, 0)
            }
            None => {
                // A token slot may decode to the whole context. A pooled or label slot never
                // decodes: its cache holds its own prompt, rounded to whole prefill chunks
                // (S4.9b). It was admitted against that size, so the memory is accounted for.
                let positions = if whole {
                    round_up_prompt(prompt.len())
                } else {
                    self.max_ctx
                };
                let mut kv = self.model.new_cache(self.kv_prec, positions);
                // The rotation is the engine's default; a request may ask for the
                // family's own choice instead (ADR 0007).
                if !cfg.kv_rotation {
                    kv = kv.without_rotation();
                }
                kv
            }
        };
        let fed = if given {
            kv.filled()
        } else if forking.is_some_and(|f| f.wait) {
            // S4.9j: a group's follower forwards nothing of its own: its leader forwards the shared
            // beginning and the state is copied in at `fork_at`.
            0
        } else if asked.is_some() || matches!(want, Want::Pooled(_) | Want::Head) {
            // A pooled or head slot starts at position 0: `Mean` needs every row of the text,
            // and a head reads row 0 of the text (or its mean), so a restored prefix would
            // leave rows uncomputed.
            //
            // S4.3: a labels slot does consult the prefix store below. It reads only its final
            // row, and the store restores the cache's own bytes, so the rows a sibling question
            // computed are bit-identical to the rows this slot would have computed. (A token
            // slot has always reused a prefix the same way.)
            0
        } else {
            // A hybrid snapshot carries the recurrent state after its own length: the store is asked
            // for the longest beginning this cache can actually restore (S4.9j).
            match self
                .prefixes
                .best_where(&prompt, cap, |snap, lcp| kv.can_restore(snap, lcp))
            {
                Some((snap, lcp)) if kv.restore(snap, lcp) => {
                    self.prefixes.note_hit(lcp);
                    lcp
                }
                _ => 0,
            }
        };
        // A restored prefix can leave nothing to prefill, in which case the first decode step must still feed
        // the prompt's last token: `prefill_chunk`, which normally sets it, never runs.
        let next = if whole {
            0
        } else if fed + 1 >= prompt.len() {
            prompt[prompt.len() - 1]
        } else {
            0
        };
        // Speculation only for a user served alone, from position 0 (the MTP cache must cover the whole prompt).
        // A second user ends it for good on the first: verifying drafts while batching spends FLOPs the batch
        // already uses.
        //
        // Penalties keep speculation off. A draft extends the token window, so verifying it would sample the
        // next row from a different window than plain decoding would see; correct verification would have to
        // thread each accepted token into the window between the two row draws. Turning speculation off for
        // these slots is the simple, lossless choice (the request asked for penalties explicitly).
        let penalized = !cfg.penalties.is_off();
        let alone = !penalized && want.is_tokens() && self.slots.is_empty() && fed == 0;
        // The engine's own drafter is the MTP block, and only for a slot that
        // did not ask for one of its own.
        let own = if asked.is_none() && alone {
            self.make_drafter(alone)
        } else {
            None
        };
        let engine_own = own.is_some();
        let (drafter, depth) = match asked {
            Some((d, depth)) => (Some(d), depth),
            None => (own, 0),
        };
        // A verification pass is one weight pass over the anchor and the
        // drafts: it cannot be wider than the scratch the engine was built
        // with.
        let depth = depth.min(self.state.max_batch.saturating_sub(1));
        if !engine_own {
            // A second user ends the engine's own speculation for good.
            self.stop_speculation();
        }
        let speculating = engine_own;
        // The penalty window starts as the whole prompt, so the first sampled token sees
        // the same sequence plain sampling would (a restored prefix included).
        let mut pen = crate::sample::PenaltyState::default();
        if penalized {
            pen.start(&prompt, cfg.max_tokens.min(self.max_ctx));
        }
        // A pass that reads several labels slots' final rows needs one logits row per label slot
        // (plus the decode rows it may share the pass with). Admission grows it, so no step
        // allocates.
        if want.is_labels() {
            let labels = self.slots.iter().filter(|s| s.want.is_labels()).count();
            self.ensure_logit_rows(labels + self.max_slots + 1);
        }
        self.slots.push(Slot {
            drafter: drafter.map(std::sync::Mutex::new),
            drafter_depth: depth,
            id,
            kv,
            // The draw is keyed by (seed, position), so a user's tokens do not depend on who else is being served.
            seed: cfg.seed.max(1),
            pool_sum: if matches!(want, Want::Pooled(Pooling::Mean)) {
                vec![0.0; self.model.cfg.d_model]
            } else {
                Vec::new()
            },
            want,
            labels,
            cfg,
            prompt_tokens: prompt.len(),
            prefilled: 0,
            prompt,
            fed,
            pos: fed,
            next,
            produced: 0,
            fed_back: vec![],
            utf8: Utf8Stream::default(),
            prefill_s: 0.0,
            decode_s: 0.0,
            pen,
            owned_cache: given,
            fork_to: Vec::new(),
            fork_at: forking.map_or(0, |f| f.at),
            fork_wait: forking.is_some_and(|f| f.wait),
            fork_snap: None,
            fork_tokens: Vec::new(),
        });
        if speculating {
            self.spec_slot = Some(id);
        }
        // A drafter of the slot's own may want the target's taps, and a
        // verification pass wants a row of logits per drafted token.
        self.refresh_taps();
        if depth > 0 {
            self.ensure_logit_rows(depth + 1);
        }
        Ok(id)
    }

    /// The drafter a slot asks for, built for that slot, and how deep it may
    /// draft.
    ///
    /// `None` when the slot asks for none. `Err` when the request cannot be
    /// served: a drafter a hybrid model cannot use, or a drafter directory that
    /// does not load.
    fn make_asked_drafter(&self, choice: &DrafterChoice) -> Result<AskedDrafter, String> {
        match choice {
            DrafterChoice::None => Ok(None),
            DrafterChoice::Suffix { min_order, max_len } => {
                // A hybrid model's recurrent state rolls back only to its
                // turn-start checkpoint, so nothing may write positions it later
                // takes back: no k-deep speculation.
                if self.model.is_hybrid() {
                    return Err(
                        "suffix speculation is not supported with a hybrid (recurrent) model"
                            .into(),
                    );
                }
                let max_len = *max_len;
                let d: Box<dyn crate::draft::Drafter> =
                    Box::new(crate::draft::SuffixDrafter::new(*min_order, max_len));
                Ok(Some((d, max_len)))
            }
            DrafterChoice::Dspark { dir, max_k: _max_k } => {
                if self.model.is_hybrid() {
                    return Err(
                        "a DSpark drafter is not supported with a hybrid (recurrent) model".into(),
                    );
                }
                #[cfg(feature = "dspark")]
                {
                    let c = self.model.cfg;
                    let head = crate::dspark::Drafter::load(dir, self.pool.width())?;
                    head.check_target(c.n_vocab, c.d_model, c.n_layer)?;
                    let k = (*_max_k).clamp(1, head.cfg.block_size - head.cfg.logits_start);
                    let d: Box<dyn crate::draft::Drafter> =
                        Box::new(crate::draft::DsparkDrafter::new(head));
                    Ok(Some((d, k)))
                }
                #[cfg(not(feature = "dspark"))]
                {
                    Err(format!(
                        "dspark drafter at {dir:?} requested but 'dspark' feature is disabled in this build"
                    ))
                }
            }
        }
    }

    /// The engine's own drafter: the model's MTP block.
    ///
    /// `None` when the model has no MTP block, when speculation is off, or
    /// when the block's cache cannot be allocated. The drafter keeps the
    /// block's own cache, so it is created per speculating slot.
    #[cfg(feature = "mtp")]
    fn make_drafter(&self, allowed: bool) -> Option<Box<dyn crate::draft::Drafter>> {
        if !allowed || !self.mtp_available {
            return None;
        }
        crate::draft::MtpDrafter::new(self.model.clone(), self.max_ctx, self.pool.width())
            .ok()
            .map(|d| Box::new(d) as Box<dyn crate::draft::Drafter>)
    }

    /// The engine's own drafter: without the `mtp` feature there is none.
    #[cfg(not(feature = "mtp"))]
    fn make_drafter(&self, _allowed: bool) -> Option<Box<dyn crate::draft::Drafter>> {
        None
    }

    /// The engine's own speculation is over: its drafter goes and the slot
    /// decodes plainly from here. A drafter the caller attached is not the
    /// engine's and stays.
    fn stop_speculation(&mut self) {
        let Some(id) = self.spec_slot.take() else {
            return;
        };
        if let Some(i) = self.slots.iter().position(|s| s.id == id) {
            self.slots[i].drafter = None;
        }
    }

    /// Attach a drafter to a slot (ADR 0007).
    ///
    /// The engine verifies what the drafter proposes against the target's own
    /// samples (`verify_step`), and sets up the taps it asks for and tells it
    /// what was committed. A drafter that asks for nothing costs the engine
    /// nothing.
    ///
    /// A drafter attached here replaces the engine's own one: the speculation
    /// the MTP block stood for is over.
    pub fn attach_drafter(
        &mut self,
        id: SlotId,
        drafter: Box<dyn crate::draft::Drafter>,
    ) -> Result<(), String> {
        self.attach_drafter_with_depth(id, drafter, 0)
    }

    /// Attach a drafter to a slot, with the depth it may draft (ADR 0007).
    ///
    /// The depth is how many tokens the drafter is asked for at most: the
    /// DSpark head's block is what it is, and the suffix index proposes as far
    /// as the evidence supports. A caller that keeps its drafter between turns
    /// (a conversation) attaches the same drafter to each of them.
    pub fn attach_drafter_with_depth(
        &mut self,
        id: SlotId,
        drafter: Box<dyn crate::draft::Drafter>,
        depth: usize,
    ) -> Result<(), String> {
        let Some(i) = self.slots.iter().position(|s| s.id == id) else {
            return Err(format!("no slot {id}"));
        };
        self.slots[i].drafter = Some(std::sync::Mutex::new(drafter));
        self.slots[i].drafter_depth = depth.min(self.state.max_batch.saturating_sub(1));
        if self.spec_slot == Some(id) {
            self.spec_slot = None;
        }
        // A verification pass wants the logits of the anchor and of the
        // drafted tokens: room for one row per drafted token.
        self.ensure_logit_rows(self.slots[i].drafter_depth.max(1) + 1);
        self.refresh_taps();
        Ok(())
    }

    /// Make room for `rows` rows of logits: a verification pass wants one row
    /// per row it forwards, and the anchor's row is one of them.
    fn ensure_logit_rows(&mut self, rows: usize) {
        if self.max_logit_rows < rows {
            self.max_logit_rows = rows;
            let n_vocab = self.model.cfg.n_vocab;
            self.state.logits.resize(rows * n_vocab, 0.0);
        }
    }

    /// Set up the taps the attached drafters need, and take them down again
    /// when the last one goes: a forward copies the tapped layers out of the
    /// residual stream, and a request without a drafter must not pay for it.
    fn refresh_taps(&mut self) {
        let mut layers: Vec<usize> = Vec::new();
        for s in &self.slots {
            let Some(d) = &s.drafter else { continue };
            let d = d.lock().expect("drafter");
            if !d.needs().taps {
                continue;
            }
            for &l in d.tap_layers() {
                if !layers.contains(&l) {
                    layers.push(l);
                }
            }
        }
        if layers != self.tap_layers {
            self.tap_layers = layers;
            self.state
                .set_taps(&self.tap_layers, self.model.cfg.d_model);
        }
    }

    /// Whether any attached drafter needs the target's hidden states, or any slot pools them.
    fn wants_hidden(&self) -> bool {
        self.slots.iter().any(|s| {
            s.want.pooling().is_some()
                || s.drafter
                    .as_ref()
                    .is_some_and(|d| d.lock().expect("drafter").needs().hidden)
        })
    }

    /// Tell a slot's drafter what was committed in batch row `row`: the token
    /// at position `pos`, with the target's taps and hidden state for that row.
    ///
    /// A slot without a drafter pays nothing: nothing is copied out of the
    /// forward pass for it.
    fn commit_row(&mut self, si: usize, row: usize, pos: usize, token: u32) {
        let Some(d) = &self.slots[si].drafter else {
            return;
        };
        let mut d = d.lock().expect("drafter");
        let d_model = self.model.cfg.d_model;
        let needs = d.needs();
        let taps: &[f32] = if needs.taps {
            self.state.tap_row(row, d_model)
        } else {
            &[]
        };
        let hidden: &[f32] = if needs.hidden {
            let st = &self.state;
            let lo = row * d_model;
            let hi = (lo + d_model).min(st.hidden.len());
            if lo < hi {
                &st.hidden[lo..hi]
            } else {
                &[]
            }
        } else {
            &[]
        };
        d.commit(pos, token, taps, hidden, &self.pool);
    }

    /// Drop a request (its caller has gone, or it is being replanned); its cache is kept for whoever shares its
    /// beginning next.
    pub fn cancel(&mut self, id: SlotId) {
        if let Some(i) = self.slots.iter().position(|s| s.id == id) {
            let s = self.slots.remove(i);
            if self.spec_slot == Some(id) {
                self.spec_slot = None;
            }
            // S4.9j: a group whose leader went away before its fork leaves its questions with an
            // empty cache. They forward their own prompts instead of waiting for a prefix that will
            // never be copied in.
            for fid in &s.fork_to {
                if let Some(fi) = self.slots.iter().position(|x| x.id == *fid) {
                    let f = &mut self.slots[fi];
                    f.fork_wait = false;
                    f.fed = 0;
                    f.pos = 0;
                    f.prefilled = 0;
                }
            }
            // A conversation's cache is the conversation's: it goes back to the
            // caller, which decides what to keep of it.
            if s.owned_cache {
                self.kept = Some(self.keep_for(s));
            } else {
                self.keep_prefix(&s);
            }
            // The slot's drafter went with it: the taps it needed may not be needed any more.
            self.refresh_taps();
        }
    }

    /// What a finished slot leaves a conversation: its cache, the tokens the
    /// cache holds and the drafter that drafted for it.
    fn keep_for(&mut self, s: Slot) -> SlotKeep {
        // The cache holds exactly the prompt positions that were prefilled (`fed` of them) followed by the
        // tokens fed back during decode, in that order.
        let mut tokens: Vec<u32> = s.prompt[..s.fed.min(s.prompt.len())].to_vec();
        tokens.extend(&s.fed_back);
        let n = tokens.len().min(s.kv.filled());
        tokens.truncate(n);
        SlotKeep {
            kv: s.kv,
            tokens,
            drafter: s.drafter.map(|d| d.into_inner().expect("drafter")),
        }
    }

    /// Keep what a slot computed: the prompt positions that are in its cache, and the tokens it produced.
    fn keep_prefix(&mut self, s: &Slot) {
        // A pooled slot's prompt is a raw text, not a conversation: its cache is of no use to
        // the next request, and inserting it would evict prefixes that are. A labels slot's
        // prompt is a decision prompt whose prefix the sibling questions of the same request
        // share, so its cache is kept (S4.3).
        if s.want.pooling().is_some() {
            return;
        }
        // The cache holds exactly the prompt positions that were prefilled (`fed` of them) followed by the
        // tokens fed back during decode, in that order. Describing it any other way stores a snapshot under
        // the wrong tokens, and a later request restores a cache that does not match its prompt.
        let mut tokens: Vec<u32> = s.prompt[..s.fed.min(s.prompt.len())].to_vec();
        tokens.extend(&s.fed_back);
        let n = tokens.len().min(s.kv.filled());
        tokens.truncate(n);
        // S4.9j: a hybrid snapshot carries the recurrent state too, so it is kept like a dense one; it is
        // restorable at its own length only, which is what the store's lookup asks for. A group question's
        // full prompt is an exception: the beginning it shares with its siblings is kept once, when the
        // group forks, and a second snapshot of the same 20 MB state per question would only evict it.
        if n < crate::prefix::MIN_PREFIX || self.prefixes.covers(&tokens) {
            return;
        }
        if s.kv.has_recurrent() && s.fork_at > 0 {
            return;
        }
        let snap = s.kv.snapshot(n);
        debug_assert_eq!(snap.len(), tokens.len());
        self.prefixes.insert(tokens, snap);
    }

    /// S4.9j: copy a decision group's shared state from its leader into the question slots that wait
    /// for it.
    ///
    /// The leader is the group's first question: it forwarded rows `0..fork_at` once, and every
    /// follower continues at `fork_at` from a byte-for-byte copy of what the leader stored (attention
    /// K/V, and a hybrid model's recurrent state and convolution history). Nothing is recomputed and
    /// nothing allocates: the followers' caches were allocated at admission.
    ///
    /// Also offers the shared beginning to the prefix store, so a later request with the same state
    /// restores it instead of forwarding it. The snapshot's buffers were reserved at admission.
    fn fork_groups(&mut self) {
        let mut li = 0;
        while li < self.slots.len() {
            if self.slots[li].fork_to.is_empty() || self.slots[li].fed < self.slots[li].fork_at {
                li += 1;
                continue;
            }
            let at = self.slots[li].fork_at;
            let ids = std::mem::take(&mut self.slots[li].fork_to);
            self.keep_forked_prefix(li);
            for id in ids {
                let Some(fi) = self.slots.iter().position(|s| s.id == id) else {
                    continue;
                };
                let forked = if fi > li {
                    let (left, right) = self.slots.split_at_mut(fi);
                    right[0].kv.fork_from(&left[li].kv, at)
                } else if fi < li {
                    let (left, right) = self.slots.split_at_mut(li);
                    left[fi].kv.fork_from(&right[0].kv, at)
                } else {
                    false
                };
                let f = &mut self.slots[fi];
                if forked {
                    f.fork_wait = false;
                    f.fed = at;
                    f.pos = at;
                    // The shared rows were forwarded once for the whole group: they are not owed
                    // again, but this question did not compute them either. `prefilled` counts what
                    // the request paid for, and it paid for the group's one evaluation.
                    f.prefilled = at;
                } else {
                    // Cannot happen with caches of one engine (same geometry, same precision). If it
                    // ever did, the question would forward its own prompt rather than wait forever.
                    f.fork_wait = false;
                    f.fed = 0;
                    f.pos = 0;
                    f.prefilled = 0;
                }
            }
            li += 1;
        }
    }

    /// Keep a group's shared beginning under its tokens, if the store has room for it and does not
    /// hold it already.
    fn keep_forked_prefix(&mut self, li: usize) {
        let Some(mut snap) = self.slots[li].fork_snap.take() else {
            return;
        };
        let at = self.slots[li].fork_at;
        let tokens = std::mem::take(&mut self.slots[li].fork_tokens);
        self.slots[li].kv.snapshot_into(&mut snap, at);
        if tokens.len() == at {
            self.prefixes.insert(tokens, snap);
        }
    }

    /// Where the engine's time has gone, in milliseconds: matmul, output head, attention, other, sampling,
    /// drafting. Engine-wide rather than per request: a decode step is shared by every row in it.
    pub fn phases_ms(&self) -> [f64; 6] {
        let p = &self.stats;
        [p.ns_matmul, p.ns_head, p.ns_attn, p.ns_other(), 0, 0].map(|ns| ns as f64 / 1e6)
    }

    /// (entries, bytes, restores, positions not prefilled) of the shared prefix store.
    pub fn prefix_stats(&self) -> (usize, usize, u64, u64) {
        (
            self.prefixes.entries(),
            self.prefixes.bytes(),
            self.prefixes.hits,
            self.prefixes.positions_saved,
        )
    }

    /// One scheduler iteration. Mixed (the default): one token for every ready user and, in the same weight pass,
    /// up to `PEREGRINE_PREFILL_CHUNK` prompt tokens of the users still being prefilled (Sarathi-Serve's stall-free batching:
    /// nobody's decode waits behind a prompt, and the pass reads the weights once for both). `PEREGRINE_MIX=0`
    /// keeps the alternating schedule (a prompt chunk, or a decode step, never both). Returns what to send to each
    /// caller; an empty result with no active slots means there is nothing to do.
    pub fn step(&mut self) -> Vec<Emission> {
        if self.slots.is_empty() {
            return vec![];
        }
        if self.model.causality() == crate::arch::Causality::Bidirectional {
            return self.bidirectional_step();
        }
        if mixed() {
            return self.mixed_step();
        }
        // A prompt that is not yet in the cache comes first, oldest slot first: a user waiting for their first
        // token is the one most sensitive to delay, and one chunk is a bounded interruption for the others.
        if let Some(i) = self.slots.iter().position(|s| !s.prefill_done()) {
            return match self.prefill_chunk(i) {
                Ok(emitted) => {
                    // A pooled slot that finished in this chunk is reaped here; the token
                    // slots' emissions never carry a `finish`.
                    self.reap(&emitted);
                    emitted
                }
                Err(e) => {
                    let id = self.slots[i].id;
                    self.slots.remove(i);
                    vec![Emission {
                        slot: id,
                        token: 0,
                        text: String::new(),
                        logprob: 0.0,
                        finish: Some(Finish::Error(e)),
                        cost: None,
                        embedding: None,
                        label_logits: None,
                        head: None,
                    }]
                }
            };
        }
        self.decode()
    }

    /// One bidirectional step: whole texts, side by side, up to the scratch's row budget
    /// (design §2.2).
    ///
    /// A text is never split. The step takes the oldest texts whose lengths fit in
    /// `state.max_batch` rows together; a text longer than what is left runs alone in its
    /// own pass. Every text of the pass is one `forward_texts` call, so all of them share
    /// the layer projections' weight passes, and each one's attention is its own block:
    /// its vector cannot depend on who else was admitted (design §2.3).
    fn bidirectional_step(&mut self) -> Vec<Emission> {
        let t0 = std::time::Instant::now();
        let before = self.stats;
        // Whole texts, up to the pass budget (S4.9b: the same `PEREGRINE_PASS_ROWS` the causal
        // path uses). A text longer than the budget still runs alone in its own pass, as long as
        // the scratch holds it; an encoder writes no cache, so nothing else bounds it.
        let budget = self.pass_rows.min(self.state.max_batch);
        // (slot index, first row, length)
        let mut chosen: Vec<(usize, usize, usize)> = Vec::new();
        let mut rows = 0usize;
        for i in 0..self.slots.len() {
            let n = self.slots[i].prompt.len();
            if chosen.is_empty() {
                if n > self.state.max_batch {
                    let cap = self.state.max_batch;
                    let id = self.slots.remove(i).id;
                    return vec![Emission {
                        slot: id,
                        token: 0,
                        text: String::new(),
                        logprob: 0.0,
                        finish: Some(Finish::Error(format!(
                            "text of {n} tokens does not fit the bidirectional pass of {cap} rows"
                        ))),
                        cost: None,
                        embedding: None,
                        label_logits: None,
                        head: None,
                    }];
                }
                chosen.push((i, 0, n));
                rows = n;
            } else if rows + n <= budget {
                chosen.push((i, rows, n));
                rows += n;
            } else {
                break;
            }
        }
        if chosen.is_empty() {
            return vec![];
        }
        let texts: Vec<&[u32]> = chosen
            .iter()
            .map(|&(i, _, _)| self.slots[i].prompt.as_slice())
            .collect();
        let wants: Vec<crate::arch::TextWant> = chosen
            .iter()
            .map(|&(i, _, _)| {
                self.slots[i]
                    .want
                    .text_want()
                    .expect("only pooled and head slots reach a bidirectional step")
            })
            .collect();
        self.state.want_hidden = true;
        let r =
            self.model
                .forward_texts(&mut self.state, &self.pool, &texts, &wants, &mut self.stats);
        self.state.want_hidden = false;
        let secs = t0.elapsed().as_secs_f64();
        if let Err(e) = r {
            let ids: Vec<SlotId> = chosen.iter().map(|&(i, _, _)| self.slots[i].id).collect();
            let out: Vec<Emission> = ids
                .iter()
                .map(|&slot| Emission {
                    slot,
                    token: 0,
                    text: String::new(),
                    logprob: 0.0,
                    finish: Some(Finish::Error(e.clone())),
                    cost: None,
                    embedding: None,
                    label_logits: None,
                    head: None,
                })
                .collect();
            self.slots.retain(|s| !ids.contains(&s.id));
            return out;
        }
        let dm = self.model.cfg.d_model;
        let hw = self.model.head_width();
        let mut out = Vec::with_capacity(chosen.len());
        for &(i, row0, n) in &chosen {
            let head = if self.slots[i].want.is_head() {
                Some(self.state.head_out[row0 * hw..(row0 + 1) * hw].to_vec())
            } else {
                None
            };
            let vector = if head.is_some() {
                None
            } else {
                Some(self.state.hidden[row0 * dm..(row0 + 1) * dm].to_vec())
            };
            let s = &mut self.slots[i];
            s.fed = n;
            s.prefilled = n;
            s.pos = n;
            s.prefill_s = secs * n as f64 / rows.max(1) as f64;
            out.push(Emission {
                slot: s.id,
                token: 0,
                text: String::new(),
                logprob: 0.0,
                finish: Some(Finish::Stop),
                cost: Some(SlotCost {
                    prompt_tokens: s.prompt_tokens,
                    prefilled: s.prefilled,
                    produced: 0,
                    prefill_s: s.prefill_s,
                    decode_s: 0.0,
                }),
                embedding: vector,
                label_logits: None,
                head,
            });
        }
        self.reap(&out);
        self.last_step = StepReport {
            stats: self.stats.since(&before),
            secs: t0.elapsed().as_secs_f64(),
            rows,
            tokens: out.len(),
            drafted: 0,
            accepted: 0,
        };
        out
    }

    /// The mixed step: decode rows for the ready slots first (they want logits), then prompt runs for the others,
    /// oldest first, within the chunk budget. Each prompt run is consecutive positions of one cache
    /// (`Caches::Mapped`), so its attention is one grouped call, as in a plain prefill chunk.
    fn mixed_step(&mut self) -> Vec<Emission> {
        // A lone slot with a drafter verifies that drafter's proposals: one
        // weight pass for the anchor and the drafted tokens (ADR 0007,
        // decision 3: speculation stays single-slot).
        if self.slots.len() == 1
            && self.slots[0].ready()
            && self.slots[0].drafter.is_some()
            && self.slots[0].cfg.penalties.is_off()
        {
            let s = &self.slots[0];
            // Room for the draft's position, and more than one token still wanted.
            if s.pos + 2 < self.max_ctx && s.produced + 1 < s.cfg.max_tokens {
                return self.verify_step(0);
            }
            // No room for a draft, or the turn is over: the slot decodes plainly.
            self.stop_speculation();
        }
        let t0 = std::time::Instant::now();
        let before = self.stats;
        let dec: Vec<usize> = (0..self.slots.len())
            .filter(|&i| self.slots[i].ready())
            .collect();
        // Chat slots keep today's chunk: a prompt longer than `LONG_PROMPT` is read in
        // larger pieces (see `LONG_PROMPT_CHUNK`), and the chat rows never exceed one chunk.
        // A prompt whose owner never decodes fills the pass up to `pass_rows` instead (S4.9b),
        // taking whole short texts and pieces of a long one, from as many slots as fit.
        let long_waiting = self
            .slots
            .iter()
            .any(|s| s.wants_tokens() && !s.prefill_done() && s.prompt.len() > LONG_PROMPT);
        // With no decoding slot beside the prompt nobody is waiting for this pass, so it takes the
        // wide chunk (S5.1). With one, the chunk is what bounds that user's wait, as before. A
        // long prompt is read in larger pieces either way (`LONG_PROMPT_CHUNK`).
        let chunk = if dec.is_empty() {
            if long_waiting {
                LONG_PROMPT_CHUNK.max(self.wide_chunk)
            } else {
                self.wide_chunk
            }
        } else if long_waiting {
            LONG_PROMPT_CHUNK.max(self.prefill_chunk)
        } else {
            self.prefill_chunk
        };
        let mut budget = chunk.min(self.state.max_batch.saturating_sub(dec.len()));
        // The work rows: a full pass, except that a pass which also carries a decode row is
        // capped at one prefill chunk of work. A chat's token cadence must not wait behind a
        // wide embedding pass; measured on bluefin, decoding chat tokens/s beside a stream of
        // embedding texts is unchanged at this bound (S4.9b's chat-latency check).
        let mut work_budget = self.pass_rows;
        if !dec.is_empty() {
            // One prefill chunk of rows would cost a decoding chat 3.2x its token rate on
            // bluefin (54.0 to 17.1 tok/s beside two embedding streams); a quarter chunk
            // costs 1.4x (measured in the PR) and still moves four rows of embedding work
            // per pass. The price is paid where it is cheapest: an embedding text waits
            // behind another pass, not behind its own weight read.
            work_budget = work_budget.min((self.prefill_chunk / 4).max(1));
        }
        work_budget = work_budget.min(self.state.max_batch.saturating_sub(dec.len()));
        // (slot index, tokens taken, first batch row): the rows of one prefilled slot are
        // contiguous in the pass, so a slot's start is not its index.
        let mut pre: Vec<(usize, usize, usize)> = Vec::new();
        let mut pre_rows = dec.len();
        for (i, s) in self.slots.iter().enumerate() {
            if !s.wants_tokens() {
                continue;
            }
            if budget == 0 {
                break;
            }
            if !s.prefill_done() {
                let take = budget.min(s.prefill_len() - s.fed);
                if take > 0 {
                    pre.push((i, take, pre_rows));
                    pre_rows += take;
                    budget -= take;
                }
            }
        }
        for (i, s) in self.slots.iter().enumerate() {
            if s.wants_tokens() {
                continue;
            }
            if work_budget == 0 {
                break;
            }
            if !s.prefill_done() {
                let take = work_budget.min(s.prefill_len() - s.fed);
                if take > 0 {
                    pre.push((i, take, pre_rows));
                    pre_rows += take;
                    work_budget -= take;
                }
            }
        }
        let n_dec = dec.len();
        let mut tokens: Vec<u32> = Vec::new();
        let mut positions: Vec<usize> = Vec::new();
        let mut map: Vec<usize> = Vec::new();
        // Cache order: the decode slots, then the prefilled ones.
        let order: Vec<usize> = dec.iter().copied().chain(pre.iter().map(|p| p.0)).collect();
        for (ci, &i) in dec.iter().enumerate() {
            tokens.push(self.slots[i].next);
            positions.push(self.slots[i].pos);
            map.push(ci);
        }
        for (k, &(i, take, _)) in pre.iter().enumerate() {
            let s = &self.slots[i];
            for t in 0..take {
                tokens.push(s.prompt[s.fed + t]);
                positions.push(s.fed + t);
                map.push(n_dec + k);
            }
        }
        let mut want: Vec<usize> = (0..n_dec).collect();
        // A labels slot that completes its prompt in this pass wants its last row's logits.
        // They are gathered after the pass; the row is the same one the model computes for a
        // one-row pass, because the head is the same code (`layer::head_rows`). S4.9d: the
        // head evaluates only the label ids, not the whole vocabulary row.
        // (pre index, start in `state.label_out`, count) per labels slot finishing here.
        let mut label_rows: Vec<(usize, usize, usize)> = Vec::new();
        self.state.label_rows.clear();
        self.state.label_ids.clear();
        self.state.label_out.clear();
        for (k, &(i, take, row0)) in pre.iter().enumerate() {
            let s = &self.slots[i];
            if s.want.is_labels() && s.fed + take >= s.prefill_len() {
                let start = self.state.label_ids.len();
                let count = s.labels.len();
                self.state.label_ids.extend_from_slice(&s.labels);
                self.state.label_out.resize(start + count, 0.0);
                self.state.label_rows.push((want.len(), start, count));
                label_rows.push((k, start, count));
                want.push(row0 + take - 1);
            }
        }
        // A drafter that needs the target's hidden states gets them: the MTP
        // block's rows read the hidden state of the position before them.
        self.state.want_hidden = self.wants_hidden();
        let r = {
            let mut refs: Vec<Option<&mut KvCache>> =
                self.slots.iter_mut().map(|s| Some(&mut s.kv)).collect();
            let mut caches: Vec<&mut KvCache> = order
                .iter()
                .map(|&i| refs[i].take().expect("a slot appears once"))
                .collect();
            let p = if tokens.len() == 1 {
                self.pool1.as_ref().unwrap_or(&self.pool)
            } else {
                &self.pool
            };
            self.model.forward_rows(
                &mut self.state,
                &mut Caches::Mapped(&mut caches, &map),
                p,
                &tokens,
                &positions,
                &want,
                &mut self.stats,
            )
        };
        if let Err(e) = r {
            let out = self
                .slots
                .iter()
                .map(|s| Emission {
                    slot: s.id,
                    token: 0,
                    text: String::new(),
                    logprob: 0.0,
                    finish: Some(Finish::Error(e.clone())),
                    cost: None,
                    embedding: None,
                    label_logits: None,
                    head: None,
                })
                .collect();
            self.slots.clear();
            return out;
        }
        self.state.want_hidden = false;
        let elapsed = t0.elapsed().as_secs_f64();
        let rows = tokens.len().max(1) as f64;
        // The plain step is a verification pass of width one: the cost model
        // prices the drafts against it, and without it the prices are a prior
        // in the wrong units (a second against a millisecond).
        if n_dec == 1 && pre.is_empty() {
            self.vcost.observe(1, elapsed);
        }
        // The prefilled rows are committed positions too: a drafter is told about them, with the
        // target's taps and hidden state for the row.
        for &(i, take, row0) in &pre {
            if self.slots[i].drafter.is_none() {
                continue;
            }
            let fed = self.slots[i].fed;
            for t in 0..take {
                self.commit_row(i, row0 + t, fed + t, tokens[row0 + t]);
            }
        }
        // A pooled slot's rows are its text so far. Mean adds them to the f64 accumulator in
        // position order here, inside the completing chunk as well; Last reads the completing
        // chunk's final row once the slot is done.
        let d_model = self.model.cfg.d_model;
        for &(i, take, row0) in &pre {
            let Some(pooling) = self.slots[i].want.pooling() else {
                continue;
            };
            if pooling == Pooling::Mean {
                let base = row0 * d_model;
                self.slots[i]
                    .pool_accumulate(&self.state.hidden[base..base + take * d_model], d_model);
            }
        }
        let mut pooled_out: Vec<Emission> = Vec::new();
        for (k, &(i, take, row0)) in pre.iter().enumerate() {
            let last = self.slots[i].prefill_len();
            let s = &mut self.slots[i];
            s.fed += take;
            s.prefilled += take;
            s.pos = s.fed;
            s.prefill_s += elapsed * take as f64 / rows;
            if s.wants_tokens() && s.ready() {
                s.next = s.prompt[last];
            }
            if s.fed < s.prefill_len() {
                continue;
            }
            let vector = s.want.pooling().map(|pooling| {
                let base = row0 * d_model;
                s.pool_finish(
                    &self.state.hidden[base..base + take * d_model],
                    d_model,
                    pooling,
                )
            });
            let label_logits = if s.want.is_labels() {
                label_rows
                    .iter()
                    .find(|&&(pk, _, _)| pk == k)
                    .map(|&(_, start, count)| self.state.label_out[start..start + count].to_vec())
            } else {
                None
            };
            if vector.is_none() && label_logits.is_none() {
                continue;
            }
            pooled_out.push(Emission {
                slot: s.id,
                token: 0,
                text: String::new(),
                logprob: 0.0,
                finish: Some(Finish::Stop),
                cost: Some(SlotCost {
                    prompt_tokens: s.prompt_tokens,
                    prefilled: s.prefilled,
                    produced: 0,
                    prefill_s: s.prefill_s,
                    decode_s: 0.0,
                }),
                embedding: vector,
                label_logits,
                head: None,
            });
        }
        // S4.9j: a group's leader may have just reached the shared beginning's end; the other
        // questions continue from its state from the next pass on.
        self.fork_groups();
        // The decode slots are reaped by `finish_decode` after it reads them; the pooled ones
        // are reaped here, after `finish_decode` has used its slot indices (`reap` compacts
        // the slot list, so it must not run while an index into it is still live).
        if n_dec == 0 {
            self.reap(&pooled_out);
            // A prompt chunk: what it cost is the prompt's own time.
            self.last_step = StepReport {
                stats: self.stats.since(&before),
                secs: t0.elapsed().as_secs_f64(),
                rows: tokens.len(),
                tokens: 0,
                drafted: 0,
                accepted: 0,
            };
            return pooled_out;
        }
        let mut out = self.finish_decode(&dec, &tokens[..n_dec], elapsed * n_dec as f64 / rows);
        // `reap` reads the emissions, so it must run before they move into `out`; the decode
        // slots were already reaped by `finish_decode`, and their ids are no longer in `slots`.
        self.reap(&pooled_out);
        out.append(&mut pooled_out);
        self.last_step = StepReport {
            stats: self.stats.since(&before),
            secs: t0.elapsed().as_secs_f64(),
            rows: tokens.len(),
            tokens: out.len(),
            drafted: 0,
            accepted: 0,
        };
        out
    }
    /// Feed the next chunk of slot `i`'s prompt. A token slot's last token is left for the first decode step,
    /// which is what produces the logits the first token is sampled from; a pooled or labels slot forwards
    /// every token and is finished here when this was its last chunk.
    fn prefill_chunk(&mut self, i: usize) -> Result<Vec<Emission>, String> {
        let t0 = std::time::Instant::now();
        let before = self.stats;
        // A pass with no other slot decoding widens (S5.1): nobody waits behind it. A pass that
        // is read while someone decodes keeps the narrow chunk, which bounds that user's wait.
        let other_decoding = self
            .slots
            .iter()
            .enumerate()
            .any(|(j, o)| j != i && o.ready());
        let s = &mut self.slots[i];
        let last = s.prefill_len();
        // A chat slot keeps the prefill chunk; a slot that never decodes fills the pass up to
        // the pass budget (S4.9b).
        let width = if s.wants_tokens() {
            if other_decoding {
                self.prefill_chunk
            } else {
                self.wide_chunk
            }
        } else {
            self.pass_rows
        };
        let take = width.min(last - s.fed);
        let chunk: Vec<u32> = s.prompt[s.fed..s.fed + take].to_vec();
        let start = s.fed;
        // A labels slot reads its last prompt row: this chunk's last row, when it completes the
        // prompt. S4.9d: the head evaluates only the label ids of that row, which is bit-identical
        // to gathering them out of the full row.
        let labels_done = s.want.is_labels() && s.fed + take >= last;
        let want_logits: &[usize] = if labels_done { &[take - 1] } else { &[] };
        self.state.label_rows.clear();
        self.state.label_ids.clear();
        self.state.label_out.clear();
        if labels_done {
            self.state.label_ids.extend_from_slice(&s.labels);
            self.state.label_out.resize(s.labels.len(), 0.0);
            self.state.label_rows.push((0, 0, s.labels.len()));
        }
        self.state.want_hidden = self.wants_hidden();
        self.model.forward(
            &mut self.state,
            &mut self.slots[i].kv,
            &self.pool,
            &chunk,
            start,
            want_logits,
            &mut self.stats,
        )?;
        // The prefilled rows are committed positions too: a drafter is told about them.
        if self.slots[i].drafter.is_some() {
            for (t, &token) in chunk.iter().enumerate() {
                self.commit_row(i, t, start + t, token);
            }
        }
        self.state.want_hidden = false;
        let secs = t0.elapsed().as_secs_f64();
        let d_model = self.model.cfg.d_model;
        let pooling = self.slots[i].want.pooling();
        if pooling == Some(Pooling::Mean) {
            self.slots[i].pool_accumulate(&self.state.hidden[..take * d_model], d_model);
        }
        let label_logits = if labels_done {
            Some(self.state.label_out[..self.slots[i].labels.len()].to_vec())
        } else {
            None
        };
        let s = &mut self.slots[i];
        s.fed += take;
        s.prefilled += take;
        s.pos = s.fed;
        s.prefill_s += secs;
        if s.wants_tokens() && s.ready() {
            s.next = s.prompt[last];
        }
        let mut out = Vec::new();
        if s.fed >= s.prefill_len() {
            let vector = pooling.map(|pooling| {
                s.pool_finish(&self.state.hidden[..take * d_model], d_model, pooling)
            });
            if vector.is_some() || label_logits.is_some() {
                out.push(Emission {
                    slot: s.id,
                    token: 0,
                    text: String::new(),
                    logprob: 0.0,
                    finish: Some(Finish::Stop),
                    cost: Some(SlotCost {
                        prompt_tokens: s.prompt_tokens,
                        prefilled: s.prefilled,
                        produced: 0,
                        prefill_s: s.prefill_s,
                        decode_s: 0.0,
                    }),
                    embedding: vector,
                    label_logits,
                    head: None,
                });
            }
        }
        // S4.9j: this chunk may have completed a group's shared beginning.
        self.fork_groups();
        // What the chunk cost: a conversation reports the prompt's own time
        // apart from the decode steps'.
        self.last_step = StepReport {
            stats: self.stats.since(&before),
            secs,
            rows: take,
            tokens: 0,
            drafted: 0,
            accepted: 0,
        };
        Ok(out)
    }

    /// One token for every ready slot, in one pass over the weights.
    fn decode(&mut self) -> Vec<Emission> {
        let t0 = std::time::Instant::now();
        let before = self.stats;
        let dec: Vec<usize> = (0..self.slots.len())
            .filter(|&i| self.slots[i].ready())
            .collect();
        let n = dec.len();
        if n == 0 {
            return vec![];
        }
        let tokens: Vec<u32> = dec.iter().map(|&i| self.slots[i].next).collect();
        let positions: Vec<usize> = dec.iter().map(|&i| self.slots[i].pos).collect();
        let want: Vec<usize> = (0..n).collect();
        let hidden = self.wants_hidden();
        let mut caches: Vec<&mut KvCache> = {
            let mut refs: Vec<Option<&mut KvCache>> =
                self.slots.iter_mut().map(|s| Some(&mut s.kv)).collect();
            dec.iter()
                .map(|&i| refs[i].take().expect("a slot appears once"))
                .collect()
        };
        let p = if n == 1 {
            self.pool1.as_ref().unwrap_or(&self.pool)
        } else {
            &self.pool
        };
        self.state.want_hidden = hidden;
        self.state.label_rows.clear();
        self.state.label_ids.clear();
        self.state.label_out.clear();
        let r = self.model.forward_rows(
            &mut self.state,
            &mut Caches::PerRow(&mut caches),
            p,
            &tokens,
            &positions,
            &want,
            &mut self.stats,
        );
        self.state.want_hidden = false;
        if let Err(e) = r {
            // A failed step is the whole batch's failure: every caller is told, and the slots go.
            let out = self
                .slots
                .iter()
                .map(|s| Emission {
                    slot: s.id,
                    token: 0,
                    text: String::new(),
                    logprob: 0.0,
                    finish: Some(Finish::Error(e.clone())),
                    cost: None,
                    embedding: None,
                    label_logits: None,
                    head: None,
                })
                .collect();
            self.slots.clear();
            return out;
        }
        let secs = t0.elapsed().as_secs_f64();
        // A lone slot's plain step is a verification pass of width one: the cost
        // model prices the drafts against it.
        if n == 1 {
            self.vcost.observe(1, secs);
        }
        let out = self.finish_decode(&dec, &tokens, secs);
        self.last_step = StepReport {
            stats: self.stats.since(&before),
            secs: t0.elapsed().as_secs_f64(),
            rows: tokens.len(),
            tokens: out.len(),
            drafted: 0,
            accepted: 0,
        };
        out
    }

    /// Sample and account for the decode rows just computed: row `b` of the logits belongs to slot `dec[b]`, which
    /// was fed `fed[b]`; `secs` is the decode rows' share of the step.
    fn finish_decode(&mut self, dec: &[usize], fed: &[u32], secs: f64) -> Vec<Emission> {
        let n = dec.len();
        self.decode_steps += 1;
        self.decode_rows += n as u64;
        // The parking rule's countdown: a plain step is a step like any other.
        if self.park > 0 {
            self.park -= 1;
        }
        let share = secs / n as f64;
        let n_vocab = self.model.cfg.n_vocab;
        // Sampling, one row per worker: each is a pass over the whole vocabulary (~2 ms at 248 320 entries), and
        // serially they cost a 2B model's step 40% at ten users. The sampler is keyed by (seed, position), so which
        // worker samples a row does not change what it draws.
        let mut picks = vec![(0u32, 0.0f32); n];
        // Penalty windows are rebuilt serially (each is O(window)) so the parallel
        // sampling below only reads them; the default request skips this entirely.
        if dec.iter().any(|&si| !self.slots[si].cfg.penalties.is_off()) {
            for &si in dec {
                self.slots[si].prepare_penalties(n_vocab);
            }
        }
        {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let pk = std::sync::Mutex::new(&mut picks);
            let (slots, state, samps) = (&self.slots, &self.state, &self.samps);
            let body = |wk: usize, _nw: usize| {
                let mut sc = None;
                loop {
                    let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if b >= n {
                        break;
                    }
                    let sc = sc.get_or_insert_with(|| samps[wk].lock().expect("sampler scratch"));
                    let s = &slots[dec[b]];
                    let logits = state.logits_row(b, n_vocab);
                    let id = crate::sample::sample_penalized(
                        logits,
                        s.cfg.temperature,
                        s.cfg.top_p,
                        s.cfg.top_k,
                        s.cfg.min_p,
                        &s.cfg.penalties,
                        s.pen.window(),
                        crate::sample::position_key(s.seed, s.cfg.turn as u64, s.pos + 1),
                        sc,
                    );
                    let lp = if s.cfg.logprobs {
                        crate::session::logprob_of(logits, id)
                    } else {
                        0.0
                    };
                    pk.lock().expect("picks")[b] = (id, lp);
                }
            };
            if n >= 2 && self.pool.width() > 1 {
                self.pool.run(body);
            } else {
                body(0, 1);
            }
        }
        let mut out = Vec::with_capacity(n);
        for (b, &si) in dec.iter().enumerate() {
            let (id, logprob) = picks[b];
            out.push(self.account(si, b, fed[b], id, logprob, share));
        }
        self.reap(&out);
        // The step is done: the next one is the next step of the count the
        // suffix index's probe rule uses.
        self.steps += 1;
        out
    }

    /// Slot `si` was fed `fed` at its position and sampled `id` for the next: move it on one position, and
    /// say what to send.
    ///
    /// `row` is the batch row the slot's token was computed in: its taps and its hidden state are that
    /// row's, and a drafter that asked for them is told what was committed.
    fn account(
        &mut self,
        si: usize,
        row: usize,
        fed: u32,
        id: u32,
        logprob: f32,
        secs: f64,
    ) -> Emission {
        let pos = self.slots[si].pos;
        // The drafter is told what was committed, with the target's row for it.
        self.commit_row(si, row, pos, fed);
        let s = &mut self.slots[si];
        // The token just fed sat at `pos`; after the prompt's own positions it is part of what the cache
        // now describes.
        if s.pos >= s.prompt.len() - 1 {
            s.fed_back.push(fed);
        }
        s.pos += 1;
        s.next = id;
        // The sampled token joins the sequence for the next penalty window.
        if !s.cfg.penalties.is_off() {
            s.pen.push(id);
        }
        s.produced += 1;
        s.decode_s += secs;
        let stop = s.cfg.stop.contains(&id);
        // The stop token itself is not part of the answer.
        let text = if stop {
            String::new()
        } else {
            s.utf8.push(&self.model.tk.token_bytes(id))
        };
        let finish = if stop {
            Some(Finish::Stop)
        } else if s.produced >= s.cfg.max_tokens {
            Some(Finish::Length)
        } else if s.pos + 1 >= self.max_ctx {
            Some(Finish::Context)
        } else {
            None
        };
        let cost = finish.is_some().then_some(SlotCost {
            prompt_tokens: s.prompt_tokens,
            prefilled: s.prefilled,
            produced: s.produced,
            prefill_s: s.prefill_s,
            decode_s: s.decode_s,
        });
        Emission {
            slot: s.id,
            token: id,
            text,
            logprob,
            finish,
            cost,
            embedding: None,
            label_logits: None,
            head: None,
        }
    }

    /// Remove the slots whose last emission is in `out`, keeping their caches as prefixes.
    fn reap(&mut self, out: &[Emission]) {
        let finished: Vec<SlotId> = out
            .iter()
            .filter(|e| e.finish.is_some())
            .map(|e| e.slot)
            .collect();
        if finished.is_empty() {
            return;
        }
        let done: Vec<Slot> = {
            let mut keep = Vec::with_capacity(self.slots.len());
            let mut gone = vec![];
            for s in self.slots.drain(..) {
                if finished.contains(&s.id) {
                    gone.push(s)
                } else {
                    keep.push(s)
                }
            }
            self.slots = keep;
            gone
        };
        for s in done {
            if self.spec_slot == Some(s.id) {
                self.spec_slot = None;
            }
            if s.owned_cache {
                // A conversation's cache goes back to the conversation.
                self.kept = Some(self.keep_for(s));
            } else {
                self.keep_prefix(&s);
            }
        }
        // A drafter went with its slot: the taps it needed may not be needed any more.
        self.refresh_taps();
    }

    /// Sample logits row `row` for slot `si` at position `pos` (the sampler's key), with its log-probability
    /// when the request asked for them.
    fn sample_row(&mut self, si: usize, row: usize, pos: usize) -> (u32, f32) {
        let n_vocab = self.model.cfg.n_vocab;
        self.slots[si].prepare_penalties(n_vocab);
        let s = &self.slots[si];
        let logits = self.state.logits_row(row, n_vocab);
        let mut sc = self.samps[0].lock().expect("sampler scratch");
        let id = crate::sample::sample_penalized(
            logits,
            s.cfg.temperature,
            s.cfg.top_p,
            s.cfg.top_k,
            s.cfg.min_p,
            &s.cfg.penalties,
            s.pen.window(),
            crate::sample::position_key(s.seed, s.cfg.turn as u64, pos),
            &mut sc,
        );
        let lp = if s.cfg.logprobs {
            crate::session::logprob_of(logits, id)
        } else {
            0.0
        };
        (id, lp)
    }

    /// Ask the slot's drafter for a proposal, and price it: how many of the
    /// drafted tokens are worth verifying.
    ///
    /// The proposal is written into the engine's own buffer, which is reused,
    /// so a draft costs no allocation. The drafter draws with the target's own
    /// position-keyed sampler, so a draft that draws with the target's noise
    /// survives as often as it survives the argmax.
    fn ask_drafter(
        &mut self,
        si: usize,
        anchor: u32,
        pos: usize,
        policy: crate::draft::DraftPolicy,
    ) -> usize {
        let Some(drafter) = &self.slots[si].drafter else {
            return 0;
        };
        let mut d = drafter.lock().expect("drafter");
        let s = &self.slots[si];
        let mut sc = self.samps[0].lock().expect("sampler scratch");
        let (seed, turn) = (s.cfg.seed, s.cfg.turn as u64);
        let (temp, top_p, top_k, min_p) =
            (s.cfg.temperature, s.cfg.top_p, s.cfg.top_k, s.cfg.min_p);
        // Draft `i` lands at position `pos + 1 + i`: it is drawn with that
        // position's key, as the target will draw it.
        let mut pick = |i: usize, logits: &[f32]| {
            crate::sample::sample_penalized(
                logits,
                temp,
                top_p,
                top_k,
                min_p,
                &s.cfg.penalties,
                s.pen.window(),
                crate::sample::position_key(seed, turn, pos + 1 + i),
                &mut sc,
            )
        };
        // The room left in the cache bounds the proposal, and so does the
        // drafter's own depth: the DSpark head's block is what it is.
        let room = self.max_ctx.saturating_sub(pos + 2);
        let depth = self.slots[si].drafter_depth;
        let k_max = if depth == 0 { room } else { room.min(depth) };
        // The narrow pool: the draft is a small, memory-bound pass, and the
        // full pool's SMT siblings spin beside the cores doing the work (on a
        // 12-core / 24-thread host this step ran 5-6x slower than plain
        // decode).
        let pool = self.pool1.as_ref().unwrap_or(&self.pool);
        d.propose(anchor, pos, k_max, pool, &mut pick, &mut self.proposal);
        let n = self.proposal.toks.len();
        // The drafter's own policy first: an exact count (the DSpark head
        // under `PEREGRINE_DSPARK_K`) is measured, not priced.
        if let Some(k) = policy.pinned_k {
            return k.min(n);
        }
        // The price: a drafter that is certain of its tokens has them all
        // verified; one that doubts is priced against the cost of the pass
        // (the engine's own forward passes, measured).
        if self.proposal.conf.len() == n && self.proposal.conf.iter().any(|&c| c < 1.0) {
            let k = crate::draft::choose_k(&self.proposal.conf, |w| self.vcost.cost(w));
            if k == 0 && policy.probe && self.steps.is_multiple_of(crate::draft::PROBE_EVERY) {
                // A confidence that is never tested never recovers: every
                // `PROBE_EVERY`th step verifies a couple of tokens anyway.
                return n.min(2);
            }
            k
        } else {
            n
        }
    }

    /// One verification step for a slot with a drafter (lossless): the drafter
    /// proposes the tokens after the anchor, and one pass of the main model
    /// computes the next token and, assuming the drafts, the ones after it.
    ///
    /// The next token is sampled as always; a drafted token is kept only if it
    /// is the token the target would have sampled there (the sampler is keyed
    /// by position, so both are the tokens plain decoding would produce). A
    /// rejected draft is undone: the cache drops its position and the
    /// recurrent layers go back to the state the pass snapshotted after the
    /// anchor's row.
    ///
    /// The accepted run ends where the turn ends: the token that stops the
    /// turn is emitted but never fed, so the cache holds only what the turn
    /// keeps.
    fn verify_step(&mut self, si: usize) -> Vec<Emission> {
        let t0 = std::time::Instant::now();
        let before = self.stats;
        let pos = self.slots[si].pos;
        let anchor = self.slots[si].next;
        // The parking rule: a drafter whose rounds do not pay for themselves is
        // asked less often, and a parked step decodes plainly.
        if self.park > 0 {
            self.park -= 1;
        }
        let drafting = self.park == 0;
        if drafting {
            // The round in which the drafter was asked, whether or not the
            // price said to verify any of it.
            self.spec_rounds += 1;
        }
        // What the engine does with the proposal beyond pricing it: the
        // drafter's own policy.
        let policy = self.slots[si]
            .drafter
            .as_ref()
            .map(|d| d.lock().expect("drafter").policy())
            .unwrap_or_default();
        // The proposal: the drafter's tokens, and how many of them are worth
        // verifying.
        let k = if drafting {
            self.ask_drafter(si, anchor, pos, policy)
        } else {
            0
        };
        // The rows of logits the state holds bound the proposal: the anchor's
        // row is one of them.
        let k = k.min(self.max_logit_rows.saturating_sub(1));
        // A hybrid model's recurrent state can only be rolled back one row (the
        // snapshot below), so it verifies one draft at a time.
        let k = if self.slots[si].kv.has_recurrent() {
            k.min(1)
        } else {
            k
        };
        // The verification pass: the anchor and the drafted tokens, one weight
        // pass. The rows are the engine's own buffers: a verification step
        // allocates nothing.
        let n = 1 + k;
        self.v_toks.clear();
        self.v_toks.push(anchor);
        self.v_pos.clear();
        self.v_pos.push(pos);
        self.v_want.clear();
        self.v_map.clear();
        for i in 0..k {
            self.v_toks.push(self.proposal.toks[i]);
            self.v_pos.push(pos + 1 + i);
        }
        self.v_want.extend(0..n);
        self.v_map.extend(std::iter::repeat_n(0, n));
        // The narrow pool, as a one-row decode step: the pass is bounded by
        // memory, and on the full pool the SMT siblings spin beside the cores
        // doing the work (on a 12-core / 24-thread host this step ran 5-6x
        // slower than plain decode).
        let pool = self.pool1.as_ref().unwrap_or(&self.pool);
        self.state.want_hidden = self.wants_hidden();
        self.state.label_rows.clear();
        self.state.label_ids.clear();
        self.state.label_out.clear();
        let tv = std::time::Instant::now();
        let r = {
            let kv = &mut self.slots[si].kv;
            kv.request_snapshot(pos);
            let mut refs = [kv];
            self.model.forward_rows(
                &mut self.state,
                &mut Caches::Mapped(&mut refs, &self.v_map),
                pool,
                &self.v_toks,
                &self.v_pos,
                &self.v_want,
                &mut self.stats,
            )
        };
        self.state.want_hidden = false;
        let verify_s = tv.elapsed().as_secs_f64();
        if let Err(e) = r {
            let id = self.slots[si].id;
            self.slots.clear();
            return vec![Emission {
                slot: id,
                token: 0,
                text: String::new(),
                logprob: 0.0,
                finish: Some(Finish::Error(e)),
                cost: None,
                embedding: None,
                label_logits: None,
                head: None,
            }];
        }
        // What the pass cost is what the model measures.
        self.vcost.observe(n, verify_s);
        if k > 0 {
            self.spec_drafts += 1;
            self.spec_tokens += k as u64;
        }
        self.decode_steps += 1;
        // The target's own samples: the longest prefix of the proposal that
        // equals them survives, and the sample after it is the token plain
        // decoding would have produced.
        self.v_samples.clear();
        self.v_logprobs.clear();
        let mut accepted = 0usize;
        for j in 0..=k {
            let (id, lp) = self.sample_row(si, j, pos + 1 + j);
            self.v_samples.push(id);
            self.v_logprobs.push(lp);
            if j < k && id == self.proposal.toks[j] {
                accepted = j + 1;
            } else {
                break;
            }
        }
        // The accepted run ends where the turn ends: the token that stops the
        // turn is emitted but never fed, so it is not in the cache either.
        let cut = self.commit_cut(si, accepted);
        // The cache keeps the rows that survived: the anchor and the accepted
        // drafts. The rest is taken back.
        if accepted == k && cut == accepted {
            self.slots[si].kv.drop_snapshot();
        } else {
            self.slots[si].kv.truncate(pos + 1 + cut);
        }
        if accepted > 0 {
            self.spec_hits += 1;
        }
        let secs = t0.elapsed().as_secs_f64();
        let share = secs / (accepted + 1) as f64;
        let mut out = Vec::with_capacity(accepted + 1);
        for i in 0..=accepted {
            // The turn may have ended with the previous token: nothing more is
            // committed then.
            if out.last().is_some_and(|e: &Emission| e.finish.is_some()) {
                break;
            }
            let fed = if i == 0 {
                anchor
            } else {
                self.proposal.toks[i - 1]
            };
            out.push(self.account(si, i, fed, self.v_samples[i], self.v_logprobs[i], share));
        }
        // The drafts that were not committed are forgotten.
        if cut < k {
            if let Some(d) = &self.slots[si].drafter {
                d.lock().expect("drafter").rollback(pos + 1 + cut);
            }
        }
        // The parking rule: a drafting round that does not pay for itself
        // parks drafting for a while (the session's rule, for the DSpark
        // head). The round is priced against a plain step actually measured.
        if drafting && k > 0 && policy.park {
            let round_s = t0.elapsed().as_secs_f64();
            let rate = (1 + accepted) as f64 / round_s.max(1e-9);
            self.draft_rate = Some(match self.draft_rate {
                Some(r) => 0.7 * r + 0.3 * rate,
                None => rate,
            });
            self.drafted_rounds += 1;
            let plain = self.vcost.ema[1].map(|t| 1.0 / t);
            if policy.pinned_k.is_none() && self.drafted_rounds >= 8 {
                if let (Some(dr), Some(pr)) = (self.draft_rate, plain) {
                    if dr < 0.8 * pr {
                        self.park = crate::draft::PARK_STEPS;
                        self.draft_rate = None;
                    }
                }
            }
        }
        self.decode_rows += out.len() as u64;
        self.reap(&out);
        // The step is done: the next one is the next step of the count the
        // suffix index's probe rule uses.
        self.steps += 1;
        // What the step cost: a conversation reports it per token.
        self.last_step = StepReport {
            stats: self.stats.since(&before),
            secs: t0.elapsed().as_secs_f64(),
            rows: n,
            tokens: out.len(),
            drafted: k as u16,
            accepted: accepted as u16,
        };
        out
    }

    /// How many of the `accepted` drafted tokens are committed: the accepted
    /// run ends where the turn ends.
    ///
    /// The token that stops the turn is emitted but never fed — sequential
    /// decoding does not feed the last token of a turn either — so it must not
    /// stay in the cache, and the draft that carried it is not committed. The
    /// same holds for the token that reaches the turn's budget.
    ///
    /// The rule the plain decode path applies too: the token that ends the
    /// turn is not fed, so the cache does not hold it.
    fn commit_cut(&self, si: usize, accepted: usize) -> usize {
        let s = &self.slots[si];
        let pos = s.pos;
        let mut cut = accepted;
        let mut produced = s.produced;
        for (i, &id) in self.v_samples.iter().enumerate().take(accepted + 1) {
            if s.cfg.stop.contains(&id) {
                cut = cut.min(i);
                break;
            }
            produced += 1;
            if produced >= s.cfg.max_tokens {
                cut = cut.min(i);
                break;
            }
            // The cache is full: the turn ends there too, and the token at that
            // position is not fed.
            if pos + i + 2 >= self.max_ctx {
                cut = cut.min(i);
                break;
            }
        }
        cut
    }
}
