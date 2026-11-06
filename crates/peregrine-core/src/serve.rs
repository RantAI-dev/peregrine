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
// a 2,000-token prompt is about 20 seconds of this machine, and `PREFILL_CHUNK` bounds the stall to about one
// second).
//
// # What it does not do
//
// No paged cache: each slot owns a contiguous cache. Paging exists on GPUs because video memory is scarce and
// fragmentation is fatal; here a slot's cache is 112-144 KB per position of ordinary memory, the slot count is
// small, and a block table would cost an indirection on the hottest loop in the engine for nothing. No
// speculation while batching: verifying a draft multiplies the rows per pass, which past the ridge above is
// exactly where there are no spare FLOPs (measured elsewhere: 2.7x at one row falling to 1.6x at eight).

use crate::kv::{KvCache, KvPrec};
use crate::model::mtp::MtpState;
use crate::model::{Caches, Model, State, StepStats};
use crate::pool::Pool;
use crate::prefix::PrefixStore;
use crate::tokenizer::Utf8Stream;
use std::sync::Arc;

/// Prompt tokens fed in one step. The stall this imposes on everyone else's decode is one chunk's prefill,
/// about a second on the reference machine's 4B model.
pub const PREFILL_CHUNK: usize = 64;

/// The mixed schedule (prompt tokens and decode rows in one pass); `PEREGRINE_MIX=0` turns it off.
fn mixed() -> bool {
    use std::sync::OnceLock;
    static M: OnceLock<bool> = OnceLock::new();
    *M.get_or_init(|| crate::options::get().mix)
}

/// Users served at once by default. Above the machine's roofline ridge (about seven rows) another user adds no
/// throughput, and every user's own tokens arrive more slowly, so this is deliberately below it.
pub const DEFAULT_MAX_SLOTS: usize = 4;

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
}

/// A request's handle while it is being served.
pub type SlotId = u64;

struct Slot {
    id: SlotId,
    kv: KvCache,
    cfg: SlotCfg,
    /// The whole prompt. Everything but its last token is prefilled; the last token is the first thing decoded.
    prompt: Vec<u32>,
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
    // set while this slot is the only one and the model has an MTP block (see `spec_step`).
    spec: Option<Spec>,
    /// The slot's own token history and its reused penalty counts. Only touched when
    /// `cfg.penalties` is not off, so an ordinary request pays nothing for it.
    pen: crate::sample::PenaltyState,
}

/// A slot's place in MTP speculation: the main model's final hidden state at `pos - 1`, and the MTP row an
/// accepted draft left owed (the draft token at `pos - 1` and the hidden state before it).
struct Spec {
    hprev: Vec<f32>,
    pending: Option<(u32, Vec<f32>)>,
}

/// MTP speculation for a lone user; `PEREGRINE_MTP=0` turns it off.
fn mtp_enabled() -> bool {
    crate::options::get().mtp
}

impl Slot {
    /// The prompt is in the cache and the slot is ready to decode.
    fn ready(&self) -> bool {
        self.fed + 1 >= self.prompt.len()
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
    // the MTP block's cache and scratch, used by the one slot that speculates (there is never more
    // than one: speculation runs only while a single user is served).
    mtp: Option<MtpState>,
    /// MTP drafts verified, and how many were kept.
    pub spec_drafts: u64,
    pub spec_hits: u64,
}

impl Engine {
    pub fn new(
        model: Arc<Model>,
        threads: usize,
        max_ctx: usize,
        max_slots: usize,
        kv_prec: KvPrec,
        prefix_cache_bytes: usize,
    ) -> Engine {
        let c = model.cfg;
        let max_slots = max_slots.max(1);
        let max_ctx = max_ctx.clamp(32, c.ctx_train);
        // The widest batch is a prefill chunk beside one row per slot (a mixed step); every decode row wants
        // logits, a prefill chunk wants none.
        let max_batch = PREFILL_CHUNK + max_slots;
        // An MTP verification pass wants logits for two rows, whatever the slot count.
        let use_mtp = model.has_mtp() && mtp_enabled() && mixed();
        let max_logit_rows = if use_mtp { max_slots.max(2) } else { max_slots };
        let state = model.new_state(max_batch, max_logit_rows, max_ctx, threads);
        let mtp = if use_mtp {
            model.mtp_state(max_ctx, threads).ok()
        } else {
            None
        };
        Engine {
            mtp,
            spec_drafts: 0,
            spec_hits: 0,
            model,
            pool: Pool::new(threads),
            pool1: crate::pool::narrow_pool(threads),
            state,
            slots: Vec::with_capacity(max_slots),
            max_slots,
            max_ctx,
            kv_prec,
            prefixes: PrefixStore::new(prefix_cache_bytes),
            next_id: 1,
            stats: StepStats::default(),
            decode_steps: 0,
            decode_rows: 0,
            samps: (0..threads.max(1))
                .map(|_| std::sync::Mutex::new(crate::sample::Scratch::default()))
                .collect(),
        }
    }

    /// Whether a lone user's decode speculates with the model's MTP block.
    pub fn speculates(&self) -> bool {
        self.mtp.is_some()
    }
    /// Turn MTP speculation off (for comparisons; `PEREGRINE_MTP=0` does it at startup).
    pub fn disable_mtp(&mut self) {
        self.mtp = None;
        for s in &mut self.slots {
            s.spec = None;
        }
    }

    pub fn max_slots(&self) -> usize {
        self.max_slots
    }
    pub fn active(&self) -> usize {
        self.slots.len()
    }
    pub fn has_room(&self) -> bool {
        self.slots.len() < self.max_slots
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    /// Bytes one slot's cache holds, for sizing a deployment.
    pub fn cache_bytes_per_slot(&self) -> u64 {
        let c = self.model.cfg;
        KvCache::new(self.kv_prec, c.kv_layers(), c.n_kv_head, c.head_dim, 1).bytes_per_position()
            * self.max_ctx as u64
            // a hybrid model's recurrent state, per slot, whatever the context length.
            + self.model.recurrent_shape().map_or(0, |r| (r.n_layers * (r.state_floats + r.conv_floats) * 4) as u64)
    }

    /// Take a request, or give it back when every slot is busy or the prompt cannot fit.
    pub fn admit(&mut self, prompt: Vec<u32>, cfg: SlotCfg) -> Result<SlotId, (Vec<u32>, String)> {
        if !self.has_room() {
            return Err((prompt, "every slot is busy".into()));
        }
        if prompt.is_empty() {
            return Err((prompt, "the prompt is empty".into()));
        }
        if prompt.len() + 1 > self.max_ctx {
            let n = prompt.len();
            return Err((
                prompt,
                format!(
                    "prompt of {n} tokens does not fit in {} positions",
                    self.max_ctx
                ),
            ));
        }
        let c = self.model.cfg;
        let id = self.next_id;
        self.next_id += 1;
        let mut kv = self.model.new_cache(self.kv_prec, self.max_ctx);
        // The last prompt token is never prefilled (it is what the first decode step feeds), so a prefix may
        // cover everything before it.
        let cap = prompt.len() - 1;
        let fed = match self.prefixes.best(&prompt, cap) {
            Some((snap, lcp)) if kv.restore(snap, lcp) => {
                self.prefixes.note_hit(lcp);
                lcp
            }
            _ => 0,
        };
        // A restored prefix can leave nothing to prefill, in which case the first decode step must still feed
        // the prompt's last token: `prefill_chunk`, which normally sets it, never runs.
        let next = if fed + 1 >= prompt.len() {
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
        let spec = match self.mtp.as_mut() {
            Some(ms) if !penalized && self.slots.is_empty() && fed == 0 => {
                self.model.mtp_truncate(ms, 0);
                Some(Spec {
                    hprev: vec![0.0; c.d_model],
                    pending: None,
                })
            }
            _ => {
                for s in &mut self.slots {
                    s.spec = None;
                }
                None
            }
        };
        // The penalty window starts as the whole prompt, so the first sampled token sees
        // the same sequence plain sampling would (a restored prefix included).
        let mut pen = crate::sample::PenaltyState::default();
        if penalized {
            pen.start(&prompt, cfg.max_tokens.min(self.max_ctx));
        }
        self.slots.push(Slot {
            spec,
            id,
            kv,
            // The draw is keyed by (seed, position), so a user's tokens do not depend on who else is being served.
            seed: cfg.seed.max(1),
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
        });
        Ok(id)
    }

    /// Drop a request (its caller has gone, or it is being replanned); its cache is kept for whoever shares its
    /// beginning next.
    pub fn cancel(&mut self, id: SlotId) {
        if let Some(i) = self.slots.iter().position(|s| s.id == id) {
            let s = self.slots.remove(i);
            self.keep_prefix(&s);
        }
    }

    /// Keep what a slot computed: the prompt positions that are in its cache, and the tokens it produced.
    fn keep_prefix(&mut self, s: &Slot) {
        // The cache holds exactly the prompt positions that were prefilled (`fed` of them) followed by the
        // tokens fed back during decode, in that order. Describing it any other way stores a snapshot under
        // the wrong tokens, and a later request restores a cache that does not match its prompt.
        let mut tokens: Vec<u32> = s.prompt[..s.fed.min(s.prompt.len())].to_vec();
        tokens.extend(&s.fed_back);
        let n = tokens.len().min(s.kv.filled());
        tokens.truncate(n);
        // a hybrid model's snapshot would hold no recurrent state, so it could never be restored.
        if n < crate::prefix::MIN_PREFIX || self.prefixes.covers(&tokens) || s.kv.has_recurrent() {
            return;
        }
        let snap = s.kv.snapshot(n);
        debug_assert_eq!(snap.len(), tokens.len());
        self.prefixes.insert(tokens, snap);
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
    /// up to `PREFILL_CHUNK` prompt tokens of the users still being prefilled (Sarathi-Serve's stall-free batching:
    /// nobody's decode waits behind a prompt, and the pass reads the weights once for both). `PEREGRINE_MIX=0`
    /// keeps the alternating schedule (a prompt chunk, or a decode step, never both). Returns what to send to each
    /// caller; an empty result with no active slots means there is nothing to do.
    pub fn step(&mut self) -> Vec<Emission> {
        if self.slots.is_empty() {
            return vec![];
        }
        if mixed() {
            return self.mixed_step();
        }
        // A prompt that is not yet in the cache comes first, oldest slot first: a user waiting for their first
        // token is the one most sensitive to delay, and one chunk is a bounded interruption for the others.
        if let Some(i) = self.slots.iter().position(|s| !s.ready()) {
            if let Err(e) = self.prefill_chunk(i) {
                let id = self.slots[i].id;
                self.slots.remove(i);
                return vec![Emission {
                    slot: id,
                    token: 0,
                    text: String::new(),
                    logprob: 0.0,
                    finish: Some(Finish::Error(e)),
                    cost: None,
                }];
            }
            return vec![];
        }
        self.decode()
    }

    /// The mixed step: decode rows for the ready slots first (they want logits), then prompt runs for the others,
    /// oldest first, within the chunk budget. Each prompt run is consecutive positions of one cache
    /// (`Caches::Mapped`), so its attention is one grouped call, as in a plain prefill chunk.
    fn mixed_step(&mut self) -> Vec<Emission> {
        if self.slots.len() == 1 && self.slots[0].spec.is_some() && self.slots[0].ready() {
            let s = &self.slots[0];
            // Room for the draft's position, and more than one token still wanted.
            if s.pos + 2 < self.max_ctx && s.produced + 1 < s.cfg.max_tokens {
                return self.spec_step();
            }
            self.slots[0].spec = None;
        }
        let t0 = std::time::Instant::now();
        let dec: Vec<usize> = (0..self.slots.len())
            .filter(|&i| self.slots[i].ready())
            .collect();
        let mut budget = PREFILL_CHUNK;
        let mut pre: Vec<(usize, usize)> = Vec::new(); // (slot index, tokens taken)
        for (i, s) in self.slots.iter().enumerate() {
            if budget == 0 {
                break;
            }
            if !s.ready() {
                let last = s.prompt.len() - 1;
                let take = budget.min(last - s.fed);
                if take > 0 {
                    pre.push((i, take));
                    budget -= take;
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
        for (k, &(i, take)) in pre.iter().enumerate() {
            let s = &self.slots[i];
            for t in 0..take {
                tokens.push(s.prompt[s.fed + t]);
                positions.push(s.fed + t);
                map.push(n_dec + k);
            }
        }
        let want: Vec<usize> = (0..n_dec).collect();
        // A lone speculating slot's prompt also goes through the MTP block, which reads the main model's hidden
        // states.
        let mtp_pre =
            n_dec == 0 && pre.len() == 1 && self.slots.len() == 1 && self.slots[0].spec.is_some();
        self.state.want_hidden = mtp_pre;
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
                })
                .collect();
            self.slots.clear();
            return out;
        }
        if mtp_pre && self.mtp_catch_up(&tokens, &positions).is_err() {
            self.slots[0].spec = None;
        }
        self.state.want_hidden = false;
        let elapsed = t0.elapsed().as_secs_f64();
        let rows = tokens.len().max(1) as f64;
        for &(i, take) in &pre {
            let s = &mut self.slots[i];
            let last = s.prompt.len() - 1;
            s.fed += take;
            s.prefilled += take;
            s.pos = s.fed;
            s.prefill_s += elapsed * take as f64 / rows;
            if s.ready() {
                s.next = s.prompt[last];
            }
        }
        if n_dec == 0 {
            return vec![];
        }
        self.finish_decode(&dec, &tokens[..n_dec], elapsed * n_dec as f64 / rows)
    }
    /// Feed the next chunk of slot `i`'s prompt. Its last token is left for the first decode step, which is
    /// what produces the logits the first token is sampled from.
    fn prefill_chunk(&mut self, i: usize) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        let s = &mut self.slots[i];
        let last = s.prompt.len() - 1;
        let take = PREFILL_CHUNK.min(last - s.fed);
        let chunk: Vec<u32> = s.prompt[s.fed..s.fed + take].to_vec();
        let start = s.fed;
        self.model.forward(
            &mut self.state,
            &mut self.slots[i].kv,
            &self.pool,
            &chunk,
            start,
            &[],
            &mut self.stats,
        )?;
        let s = &mut self.slots[i];
        s.fed += take;
        s.prefilled += take;
        s.pos = s.fed;
        s.prefill_s += t0.elapsed().as_secs_f64();
        if s.ready() {
            s.next = s.prompt[last];
        }
        Ok(())
    }

    /// One token for every ready slot, in one pass over the weights.
    fn decode(&mut self) -> Vec<Emission> {
        let t0 = std::time::Instant::now();
        let n = self.slots.len();
        let tokens: Vec<u32> = self.slots.iter().map(|s| s.next).collect();
        let positions: Vec<usize> = self.slots.iter().map(|s| s.pos).collect();
        let want: Vec<usize> = (0..n).collect();
        let mut caches: Vec<&mut KvCache> = self.slots.iter_mut().map(|s| &mut s.kv).collect();
        let p = if n == 1 {
            self.pool1.as_ref().unwrap_or(&self.pool)
        } else {
            &self.pool
        };
        let r = self.model.forward_rows(
            &mut self.state,
            &mut Caches::PerRow(&mut caches),
            p,
            &tokens,
            &positions,
            &want,
            &mut self.stats,
        );
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
                })
                .collect();
            self.slots.clear();
            return out;
        }
        let all: Vec<usize> = (0..n).collect();
        self.finish_decode(&all, &tokens, t0.elapsed().as_secs_f64())
    }

    /// Sample and account for the decode rows just computed: row `b` of the logits belongs to slot `dec[b]`, which
    /// was fed `fed[b]`; `secs` is the decode rows' share of the step.
    fn finish_decode(&mut self, dec: &[usize], fed: &[u32], secs: f64) -> Vec<Emission> {
        let n = dec.len();
        self.decode_steps += 1;
        self.decode_rows += n as u64;
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
                        crate::sample::position_key(s.seed, 0, s.pos + 1),
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
            out.push(self.account(si, fed[b], id, logprob, share));
        }
        self.reap(&out);
        out
    }

    /// Slot `si` was fed `fed` at its position and sampled `id` for the next: move it on one position, and
    /// say what to send.
    fn account(&mut self, si: usize, fed: u32, id: u32, logprob: f32, secs: f64) -> Emission {
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
        for s in &done {
            self.keep_prefix(s);
        }
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
            crate::sample::position_key(s.seed, 0, pos),
            &mut sc,
        );
        let lp = if s.cfg.logprobs {
            crate::session::logprob_of(logits, id)
        } else {
            0.0
        };
        (id, lp)
    }

    /// Feed a lone speculating slot's prompt chunk (`tokens` at `positions`, just run with hidden states kept)
    /// through the MTP block. Its row at position q reads the main model's hidden state at q − 1.
    fn mtp_catch_up(&mut self, tokens: &[u32], positions: &[usize]) -> Result<(), String> {
        let d = self.model.cfg.d_model;
        let n = tokens.len();
        let sp = self.slots[0].spec.as_mut().ok_or("no speculation")?;
        let ms = self.mtp.as_mut().ok_or("no MTP state")?;
        let mut h = Vec::with_capacity(n * d);
        h.extend_from_slice(&sp.hprev);
        h.extend_from_slice(&self.state.hidden[..(n - 1) * d]);
        for k in (0..n).step_by(crate::model::mtp::MTP_MAX_ROWS) {
            let e = (k + crate::model::mtp::MTP_MAX_ROWS).min(n);
            self.model.mtp_forward(
                ms,
                &self.pool,
                &tokens[k..e],
                &h[k * d..e * d],
                &positions[k..e],
                false,
                &mut self.stats,
            )?;
        }
        sp.hprev
            .copy_from_slice(&self.state.hidden[(n - 1) * d..n * d]);
        Ok(())
    }

    /// One step of a lone user with MTP speculation (lossless): the MTP block drafts the token after next, one
    /// two-row pass of the main model computes the next token and, assuming the draft, the one after it. The
    /// next token is sampled as always; the draft is kept only if it is that token, and then the second row's
    /// token is sampled too (the sampler is keyed by position, so both are the tokens plain decoding would
    /// produce). A rejected draft is undone: the cache drops its position and the recurrent layers go back to
    /// the state the pass snapshotted after the first row.
    fn spec_step(&mut self) -> Vec<Emission> {
        let t0 = std::time::Instant::now();
        let d = self.model.cfg.d_model;
        let (tok, p) = (self.slots[0].next, self.slots[0].pos);
        let (mut mt, mut mh, mut mp) = (vec![], vec![], vec![]);
        {
            let sp = self.slots[0].spec.as_mut().expect("checked by the caller");
            if let Some((pt, ph)) = sp.pending.take() {
                mt.push(pt);
                mh.extend(ph);
                mp.push(p - 1);
            }
            mt.push(tok);
            mh.extend_from_slice(&sp.hprev);
            mp.push(p);
        }
        // One or two rows through the MTP block, then two through the model: bound by memory like a one-row
        // decode step, so on the narrow pool too. On the full pool the SMT siblings spin beside the cores doing
        // the work: on a 12-core / 24-thread host this step ran 5-6x slower than plain decode.
        let pool = self.pool1.as_ref().unwrap_or(&self.pool);
        let ms = self
            .mtp
            .as_mut()
            .expect("a speculating slot implies MTP state");
        let draft = match self
            .model
            .mtp_forward(ms, pool, &mt, &mh, &mp, true, &mut self.stats)
        {
            Ok(Some(t)) => t,
            // Not fatal: the slot decodes plainly from here.
            _ => {
                self.slots[0].spec = None;
                return self.mixed_step();
            }
        };
        self.state.want_hidden = true;
        let r = {
            let kv = &mut self.slots[0].kv;
            kv.request_snapshot(p);
            let mut refs = [kv];
            self.model.forward_rows(
                &mut self.state,
                &mut Caches::Mapped(&mut refs, &[0, 0]),
                pool,
                &[tok, draft],
                &[p, p + 1],
                &[0, 1],
                &mut self.stats,
            )
        };
        self.state.want_hidden = false;
        if let Err(e) = r {
            let id = self.slots[0].id;
            self.slots.clear();
            return vec![Emission {
                slot: id,
                token: 0,
                text: String::new(),
                logprob: 0.0,
                finish: Some(Finish::Error(e)),
                cost: None,
            }];
        }
        let (a, lpa) = self.sample_row(0, 0, p + 1);
        let h0 = self.state.hidden[..d].to_vec();
        self.spec_drafts += 1;
        self.decode_steps += 1;
        let mut out = Vec::with_capacity(2);
        if a == draft {
            self.spec_hits += 1;
            self.slots[0].kv.drop_snapshot();
            let (b, lpb) = self.sample_row(0, 1, p + 2);
            let h1 = self.state.hidden[d..2 * d].to_vec();
            let secs = t0.elapsed().as_secs_f64();
            out.push(self.account(0, tok, a, lpa, secs / 2.0));
            if out[0].finish.is_none() {
                out.push(self.account(0, draft, b, lpb, secs / 2.0));
                if let Some(sp) = self.slots[0].spec.as_mut() {
                    sp.pending = Some((a, h0));
                    sp.hprev = h1;
                }
            }
        } else {
            self.slots[0].kv.truncate(p + 1);
            let secs = t0.elapsed().as_secs_f64();
            out.push(self.account(0, tok, a, lpa, secs));
            if let Some(sp) = self.slots[0].spec.as_mut() {
                sp.hprev = h0;
            }
        }
        self.decode_rows += out.len() as u64;
        self.reap(&out);
        out
    }

    /// What one request cost: prompt tokens, tokens produced, and the seconds spent prefilling and decoding it
    /// (a decode step's time divided among the rows that shared it).
    pub fn slot_cost(&self, id: SlotId) -> Option<(usize, usize, f64, f64)> {
        self.slots
            .iter()
            .find(|s| s.id == id)
            .map(|s| (s.prompt_tokens, s.produced, s.prefill_s, s.decode_s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_tokens: usize) -> SlotCfg {
        SlotCfg {
            max_tokens,
            ..SlotCfg::default()
        }
    }

    /// The model these tests run against: `PEREGRINE_MODEL`, else `models/Qwen3-1.7B-Q4_K_M.gguf`.
    fn model() -> Option<Arc<Model>> {
        let p = match std::env::var_os("PEREGRINE_MODEL") {
            Some(p) => std::path::PathBuf::from(p),
            None => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/Qwen3-1.7B-Q4_K_M.gguf"),
        };
        p.exists().then(|| crate::model::keep::open(&p).ok())?
    }

    /// Two users served together must each get exactly the text they would get alone.
    #[test]
    fn users_served_together_get_what_they_would_get_alone() {
        let Some(m) = model() else {
            eprintln!("skipped: set PEREGRINE_MODEL");
            return;
        };
        let prompts = [
            vec![1u32, 450, 29871, 13, 5618],
            vec![1u32, 910, 338, 263, 1243],
        ];
        let steps = 8;
        let solo: Vec<String> = prompts
            .iter()
            .map(|p| {
                let mut e = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 0);
                e.admit(p.clone(), cfg(steps)).unwrap();
                let mut text = String::new();
                while e.active() > 0 {
                    for em in e.step() {
                        text.push_str(&em.text);
                    }
                }
                text
            })
            .collect();

        let mut e = Engine::new(m.clone(), 3, 256, 2, KvPrec::F16, 1 << 26);
        let a = e.admit(prompts[0].clone(), cfg(steps)).unwrap();
        let b = e.admit(prompts[1].clone(), cfg(steps)).unwrap();
        let (mut ta, mut tb) = (String::new(), String::new());
        while e.active() > 0 {
            for em in e.step() {
                if em.slot == a {
                    ta.push_str(&em.text);
                } else if em.slot == b {
                    tb.push_str(&em.text);
                }
            }
        }
        assert_eq!(ta, solo[0], "user A's text changed when sharing the engine");
        assert_eq!(tb, solo[1], "user B's text changed when sharing the engine");
        assert!(
            e.decode_rows > e.decode_steps,
            "the users never shared a pass"
        );
    }

    /// A user that arrives while another is already decoding joins the batch and is served correctly.
    #[test]
    fn a_user_can_arrive_mid_flight() {
        let Some(m) = model() else {
            eprintln!("skipped: set PEREGRINE_MODEL");
            return;
        };
        let late = vec![1u32, 910, 338, 263, 1243];
        let mut solo = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 0);
        solo.admit(late.clone(), cfg(6)).unwrap();
        let mut want = String::new();
        while solo.active() > 0 {
            for em in solo.step() {
                want.push_str(&em.text);
            }
        }

        let mut e = Engine::new(m.clone(), 3, 256, 2, KvPrec::F16, 1 << 26);
        let first = e.admit(vec![1u32, 450, 29871, 13, 5618], cfg(20)).unwrap();
        for _ in 0..6 {
            e.step();
        }
        let second = e.admit(late, cfg(6)).unwrap();
        let mut got = String::new();
        while e.active() > 0 {
            for em in e.step() {
                if em.slot == second {
                    got.push_str(&em.text);
                }
            }
        }
        assert_eq!(
            got, want,
            "a user that arrived later was served differently"
        );
        assert!(first != second);
    }

    /// A second user whose prompt begins like the first one's does not prefill that beginning again.
    #[test]
    fn a_shared_beginning_is_prefilled_once_for_everyone() {
        let Some(m) = model() else {
            eprintln!("skipped: set PEREGRINE_MODEL");
            return;
        };
        // A long shared opening (a system prompt), then a different question each.
        let opening: Vec<u32> = (0..80).map(|i| ((i * 13) % 900 + 11) as u32).collect();
        let run = |e: &mut Engine, tail: u32| {
            let mut p = opening.clone();
            p.push(tail);
            let id = e.admit(p, cfg(3)).unwrap();
            let mut text = String::new();
            while e.active() > 0 {
                for em in e.step() {
                    if em.slot == id {
                        text.push_str(&em.text);
                    }
                }
            }
            text
        };
        let mut e = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 1 << 26);
        run(&mut e, 5618);
        let before = e.stats.bytes_weights;
        let second = run(&mut e, 6324);
        let (entries, _, hits, saved) = e.prefix_stats();
        assert!(
            entries > 0 && hits > 0,
            "the second user did not restore a prefix"
        );
        assert!(
            saved as usize >= opening.len() - 1,
            "only {saved} positions were reused"
        );
        // With the opening already in the cache, the second user's prefill is a chunk of weights, not three.
        let second_cost = e.stats.bytes_weights - before;
        assert!(
            second_cost < before,
            "the second user read as many weights as the first ({second_cost} vs {before})"
        );
        // And it is still the right answer: the same question on a cold engine gives the same text.
        let mut cold = Engine::new(m, 3, 256, 1, KvPrec::F16, 0);
        assert_eq!(
            run(&mut cold, 6324),
            second,
            "a restored prefix changed the answer"
        );
    }

    #[test]
    fn a_full_engine_refuses_and_a_cancelled_slot_frees_its_room() {
        let Some(m) = model() else {
            eprintln!("skipped: set PEREGRINE_MODEL");
            return;
        };
        let mut e = Engine::new(m, 2, 128, 1, KvPrec::F16, 0);
        let a = e.admit(vec![1u32, 450, 29871], cfg(4)).unwrap();
        let refused = e.admit(vec![1u32, 450], cfg(4));
        assert!(refused.is_err(), "a full engine admitted a second request");
        assert_eq!(
            refused.unwrap_err().0,
            vec![1u32, 450],
            "the prompt is handed back"
        );
        e.cancel(a);
        assert_eq!(e.active(), 0);
        assert!(
            e.admit(vec![1u32, 450], cfg(4)).is_ok(),
            "cancelling did not free the slot"
        );
        // A prompt longer than the context is refused rather than truncated.
        let mut small = Engine::new(e.model.clone(), 2, 32, 1, KvPrec::F16, 0);
        assert!(small.admit(vec![7u32; 64], cfg(4)).is_err());
    }
}
