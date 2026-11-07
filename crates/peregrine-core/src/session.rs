//! The session API: load a model, hold a conversation, report what it cost.
//!
//! # A conversation over one slot of the engine
//!
//! ADR 0007 (decision 4) makes the engine the one runtime and keeps the session
//! as the public conversation type: a thin layer over a one-slot [`Engine`].
//! What stays here is what belongs to a conversation — the chat history, the
//! template, the prompt, the reuse of the positions the previous turn computed,
//! the shifting of the context when it fills, the cancellation, and the
//! per-token telemetry. What moved to the engine is the generation itself: the
//! sampling, the speculation and the verification of the drafts.
//!
//! The slot is told what the session's own runtime told it: the conversation's
//! turn in the sampling key (`SlotCfg::turn`), the family's own KV cache
//! (`SlotCfg::kv_rotation`) and the drafter the conversation speculates with.
//!
//! # Suffix-indexed speculation
//!
//! The paper proposed speculative decode with a draft model. This implements the
//! variant that needs no second model at all: match the longest suffix of what has
//! been generated against everything earlier in the context (`suffix.rs`: an n-gram
//! index, a vote across occurrences, a confidence table learned from what
//! verification accepted) and propose whatever followed. How many tokens to verify
//! is priced against the measured cost of a verification pass of that width
//! (`choose_k`), so a doubtful draft is not verified at all. The draft costs a few
//! hash lookups — microseconds against a decode step measured in tens of
//! milliseconds — and a few bytes per context token.
//!
//! The reason it works here is the workload. ParagonCorp's tasks are *input-grounded*:
//! product-data normalisation, ingredient extraction, invoice parsing, review scoring. Most
//! output tokens are copied or lightly transformed from the prompt, so the context
//! already contains the continuation. On free-form generation the acceptance rate
//! falls to near zero and the mechanism costs one extra batched forward per step —
//! which is why `acceptance_rate` is reported rather than assumed.
//!
//! Verification is exact: a drafted token is accepted only when it equals the token
//! the model would have produced at that position anyway, so decoding with
//! speculation emits *precisely* the sequence decoding without it would (acceptance
//! criterion A6). At temperature 0 that token is the argmax. At any other
//! temperature it is a pure function of the logits and a key derived from the
//! position (`sample.rs`, after uzu), so the same guarantee holds when sampling and
//! speculation stays on — it used to be switched off for every non-greedy chat.

use crate::dspark::Drafter;
use crate::kv::{KvCache, KvPrec};
use crate::model::Model;
use crate::serve::{Engine, Finish, SlotCfg};
use crate::telem::{Machine, TokenTelem, TurnTelem};
use crate::template::{today_llama3, Template};
use crate::tokenizer::Utf8Stream;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct SessionCfg {
    pub max_ctx: usize,
    pub kv_prec: KvPrec,
    /// 0 means greedy. Speculation is exact at any temperature (see module docs).
    pub temperature: f32,
    pub top_p: f32,
    // keep only the `top_k` most probable tokens (0 = all) and those with at least `min_p` times the
    // best token's probability (0 = off); llama-server's defaults are 40 and 0.05.
    pub top_k: usize,
    pub min_p: f32,
    pub seed: u64,
    pub max_tokens: usize,
    pub threads: usize,
    pub speculate: bool,
    /// Tokens proposed per speculative step.
    pub draft_len: usize,
    /// Shortest suffix (in tokens) accepted as a match in the context; 0 disables
    /// suffix drafting. Longer matches are preferred whenever they exist.
    pub ngram: usize,
    /// Prepended once per conversation. `None` uses the model's own default.
    pub system: Option<String>,
    /// The chat template to use, when the file declares none (or the caller
    /// prefers another one). `None` uses the file's own template.
    pub template: Option<Template>,
    /// Let a reasoning model think (Qwen3's `enable_thinking`). Off writes the
    /// template's empty reasoning block, so the reply starts with the answer.
    pub thinking: bool,
    /// A DSpark drafter directory (`config.json` + `model.safetensors`) trained for
    /// this target. When set, greedy decoding drafts with it instead of prompt lookup.
    pub dspark: Option<std::path::PathBuf>,
    /// Most draft tokens verified per round (≤ the drafter's block); the policy may
    /// verify fewer. `PEREGRINE_DSPARK_K` pins an exact count for measurement.
    pub dspark_max: usize,
    // text the assistant has already written for the turn being planned, appended
    // after the template's assistant header (and after the empty reasoning block when thinking
    // is off). A continuation: the server closes a reasoning span at its budget by replanning
    // the turn with the reasoning so far plus `</think>` here, and the cache prefix is reused.
    pub assistant_prefix: Option<String>,
    /// Run the projections a LittleBit sidecar (`<model>.slb`) covers from it. Off
    /// loads the model dense even when a sidecar is present.
    pub littlebit: bool,
}

impl Default for SessionCfg {
    fn default() -> Self {
        SessionCfg {
            max_ctx: 2048,
            kv_prec: KvPrec::F16,
            temperature: 0.0,
            top_p: 0.95,
            top_k: 0,
            min_p: 0.0,
            seed: 0,
            max_tokens: 256,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            speculate: true,
            draft_len: 12,
            ngram: 2,
            system: None,
            template: None,
            thinking: true,
            dspark: None,
            dspark_max: 3,
            assistant_prefix: None,
            littlebit: true,
        }
    }
}

pub struct Session {
    pub model: Arc<Model>,
    /// The one-slot engine the conversation runs on (ADR 0007, decision 4).
    engine: Engine,
    pub cfg: SessionCfg,
    pub machine: Machine,
    /// The conversation's cache. The engine borrows it for a turn and hands it
    /// back when the turn ends; `None` while the engine holds it.
    kv: Option<KvCache>,
    /// Every token fed to the model so far. Also the corpus the suffix index
    /// searches and the conversation the prefix reuse matches against.
    ctx: Vec<u32>,
    /// The sampling seed, and the turn counter mixed into every position's key, so a
    /// regenerated reply draws fresh randomness while each draw stays reproducible.
    seed: u64,
    turn: u64,
    started: bool,
    /// The chat template the file declares, and every token id that ends a turn
    /// under it (plus the tokenizer's own eos).
    template: Template,
    stops: Vec<u32>,
    default_system: Option<&'static str>,
    /// The conversation as text, (user, assistant) per exchange. The cache holds the
    /// same thing as positions; this copy is what lets the oldest exchanges be dropped
    /// and the rest re-encoded when the context fills, instead of failing the turn.
    history: Vec<(String, String)>,
    /// The suffix drafter: the conversation's own speculation. It keeps its corpus
    /// and what it learned between the turns.
    suffix: Option<Box<dyn crate::draft::Drafter>>,
    /// The attached DSpark drafter, fed the target's tapped hidden states.
    pub dspark: Option<Drafter>,
    /// The tokens the cache still holds after a `reset` or `seed_history`: the next
    /// prompt is matched against them and only its divergent tail is prefilled
    /// (`reuse_prefix`). Empty while a conversation is live.
    kept: Vec<u32>,
    /// Positions the last turn took from the cache instead of prefilling them.
    prefix_reused: usize,
    // snapshots of finished conversations, so a prompt that shares a long beginning with one of them (not
    // only with the last) does not recompute it. Off until `enable_prefix_store`.
    prefixes: Option<crate::prefix::PrefixStore>,
}

/// Reserve this many positions for the answer when deciding whether a prompt fits.
/// A prompt that leaves no room to answer is not a fit.
fn answer_reserve(cfg: &SessionCfg) -> usize {
    cfg.max_tokens.min(cfg.max_ctx / 4).max(16)
}

/// What the caller learns about each emitted token.
pub struct Emitted<'a> {
    pub id: u32,
    pub text: &'a str,
    pub telem: &'a TokenTelem,
    // log-probability of `id` under the softmax of the raw logits (temperature 1).
    pub logprob: f32,
}

// the log-probability of token `id` in one row of logits, with the full normaliser.
pub(crate) fn logprob_of(logits: &[f32], id: u32) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    logits
        .get(id as usize)
        .map_or(f32::NEG_INFINITY, |&v| v - max - sum.ln())
}

impl Session {
    /// Initialize a single-user `Session` from an already loaded model.
    pub fn from_model(model: Arc<Model>, cfg: SessionCfg) -> Result<Session, String> {
        let c = model.cfg;
        // a hybrid model's recurrent state rolls back only to its turn-start checkpoint, so nothing may
        // write positions it later takes back: no suffix speculation, no drafter.
        let mut cfg = cfg;
        if model.is_hybrid() {
            if cfg.dspark.is_some() {
                return Err(
                    "a DSpark drafter is not supported with a hybrid (recurrent) model".into(),
                );
            }
            cfg.draft_len = 0;
        }

        let max_ctx = cfg.max_ctx.clamp(32, c.ctx_train);
        let threads = cfg.threads.max(1);
        #[cfg(feature = "dspark")]
        let dspark = match &cfg.dspark {
            Some(dir) => {
                let d = Drafter::load(dir, threads)?;
                d.check_target(c.n_vocab, c.d_model, c.n_layer)?;
                Some(d)
            }
            None => None,
        };
        #[cfg(not(feature = "dspark"))]
        let dspark: Option<Drafter> = if let Some(dir) = &cfg.dspark {
            return Err(format!(
                "dspark drafter at {:?} requested but 'dspark' feature is disabled in this build",
                dir
            ));
        } else {
            None
        };
        let dspark_k = dspark.as_ref().map_or(0, |d| {
            cfg.dspark_max
                .clamp(1, d.cfg.block_size - d.cfg.logits_start)
        });
        // `new_cache` adds a hybrid model's recurrent state and sizes K/V for its attention layers only.
        let kv = model.new_cache(cfg.kv_prec, max_ctx);
        // The int8 rotation is measured per family: see `KvCache::without_rotation`.
        let kv = if !model.session_kv_rotation() {
            kv.without_rotation()
        } else {
            kv
        };
        // The engine the conversation runs on: one slot, and one position more
        // than the conversation's context — the engine stops a turn one position
        // short of a full cache, and the session fills it (ADR 0007).
        let mut engine = Engine::new(model.clone(), threads, max_ctx + 1, 1, cfg.kv_prec, 0);
        // The conversation does not speculate with the model's MTP block: it
        // speculates with the suffix index or with a DSpark head, as it always did.
        engine.disable_mtp();
        // The suffix drafter: the conversation's own speculation. A hybrid model
        // has none (its recurrent state cannot be rolled back), and the engine
        // would refuse it — the session turns it off silently, as it always did.
        let suffix = if cfg.speculate && cfg.ngram > 0 && cfg.draft_len > 0 {
            Some(
                Box::new(crate::draft::SuffixDrafter::new(cfg.ngram, cfg.draft_len))
                    as Box<dyn crate::draft::Drafter>,
            )
        } else {
            None
        };
        let machine = Machine::cached(threads);
        let template = match cfg.template {
            Some(t) => t,
            None => Template::detect(model.gguf())?,
        };
        let stops = template.stop_ids(&model.tk);
        let default_system = template.default_system(model.name());
        let seed = if cfg.seed == 0 {
            0x2545_F491_4F6C_DD1D
        } else {
            cfg.seed
        };

        Ok(Session {
            model,
            engine,
            cfg: SessionCfg {
                max_ctx,
                threads,
                dspark_max: dspark_k,
                ..cfg
            },
            machine,
            kv: Some(kv),
            ctx: Vec::new(),
            seed,
            turn: 0,
            started: false,
            template,
            stops,
            default_system,
            history: Vec::new(),
            suffix,
            dspark,
            kept: Vec::new(),
            prefix_reused: 0,
            prefixes: None,
        })
    }

    /// Open a model at `path` using the provided model families and initialize a `Session`.
    pub fn open_with(
        path: &Path,
        cfg: SessionCfg,
        families: &[&dyn crate::arch::Family],
    ) -> Result<Session, String> {
        let model =
            crate::model::keep::open_with_template(path, cfg.littlebit, cfg.template, families)?;
        Self::from_model(model, cfg)
    }

    /// Seconds for one verification forward of each width in `widths` (median of
    /// `reps`), at the end of the current context. The positions written are past
    /// `ctx` and are cut again, exactly as a rejected draft's are. Used to measure
    /// the batch→cost curve the draft length is priced against; call after a turn.
    pub fn probe_verify_curve(
        &mut self,
        widths: &[usize],
        reps: usize,
    ) -> Result<Vec<(usize, f64)>, String> {
        // it truncates to a position the recurrent state has no checkpoint for, which would empty a
        // hybrid model's cache while the session still counted its context.
        if self.model.is_hybrid() {
            return Err(
                "the verification-width probe is not available for a hybrid (recurrent) model"
                    .into(),
            );
        }
        let start = self.ctx.len();
        let tok = self.ctx.last().copied().unwrap_or(0);
        let mut out = Vec::new();
        // The scratch the probe needs is its own: the engine's state is the
        // engine's, and the probe runs between the turns.
        let mut st = self.model.new_state(
            crate::serve::PREFILL_CHUNK,
            1,
            self.cfg.max_ctx,
            self.cfg.threads,
        );
        for &w in widths {
            let batch = vec![tok; w];
            let want: Vec<usize> = (0..w).collect();
            let mut times = Vec::new();
            for _ in 0..reps + 1 {
                let mut ss = crate::model::StepStats::default();
                let t = Instant::now();
                let model = &self.model;
                let pool = self.engine.pool();
                let kv = self
                    .kv
                    .as_mut()
                    .expect("the cache is the engine's during a turn");
                model.forward(&mut st, kv, pool, &batch, start, &want, &mut ss)?;
                times.push(t.elapsed().as_secs_f64());
                self.kv
                    .as_mut()
                    .expect("the cache is the engine's during a turn")
                    .truncate(start);
            }
            times.remove(0); // the first touches pages the rest find warm
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out.push((w, times[times.len() / 2]));
        }
        Ok(out)
    }

    /// Forget the conversation. The cache is *not* cleared: its positions stay valid
    /// for the tokens that were fed (`kept`), and the next prompt reuses whatever
    /// prefix it shares with them — the system block at least, when the system
    /// prompt is unchanged, which is what the quality lab's forty probes share, and
    /// everything up to the edited message on a regenerate or an edit
    /// (FreeToken's radix prefix cache and semantic anchors, in their one-sequence
    /// form).
    pub fn reset(&mut self) {
        if !self.ctx.is_empty() {
            self.kept = std::mem::take(&mut self.ctx);
        } else {
            self.ctx.clear();
        }
        // The suffix index forgets the corpus it holds: it is rebuilt from the
        // conversation's tokens. What it learned about how reliable each kind of
        // match is stays.
        if let Some(d) = &mut self.suffix {
            d.reset();
        }
        self.started = false;
        self.history.clear();
    }

    /// After a `reset`/`seed_history`, keep the cache positions the new prompt `toks`
    /// shares with what was fed before and return how many; the caller prefills the
    /// rest. The match stops one token short of the whole prompt, because a turn
    /// needs at least one new position to produce logits.
    fn reuse_prefix(&mut self, toks: &[u32]) -> usize {
        let kept = std::mem::take(&mut self.kept);
        let limit = toks.len().saturating_sub(1).min(kept.len());
        let l = toks
            .iter()
            .zip(kept.iter())
            .take(limit)
            .take_while(|(a, b)| a == b)
            .count();
        // keep the conversation that just ended, then prefer the stored one that shares the longest beginning
        // with this prompt over the one the cache holds now when it saves more than a few positions. The drafter has
        // its own cache, which is not stored: with one attached the store is not used.
        if self.dspark.is_none() {
            if let Some(store) = self.prefixes.as_mut() {
                if kept.len() >= crate::prefix::MIN_PREFIX && !store.covers(&kept) {
                    let snap = self
                        .kv
                        .as_ref()
                        .expect("the cache is the engine's during a turn")
                        .snapshot(kept.len());
                    let n = snap.len();
                    store.insert(kept[..n].to_vec(), snap);
                }
                let cap = toks.len().saturating_sub(1);
                if let Some((snap, lcp)) = store.best(toks, cap) {
                    let restored = self
                        .kv
                        .as_mut()
                        .expect("the cache is the engine's during a turn")
                        .restore(snap, lcp);
                    if restored && lcp > l + 8 {
                        store.note_hit(lcp - l);
                        self.ctx.clear();
                        self.ctx.extend_from_slice(&toks[..lcp]);
                        // The corpus the suffix index searches is the conversation's
                        // tokens: the restored ones, not the ones before them.
                        self.reset_suffix_corpus();
                        return lcp;
                    }
                }
            }
        }
        self.kv
            .as_mut()
            .expect("the cache is the engine's during a turn")
            .truncate(l);
        // a hybrid model's cache may keep less than asked (only its checkpoint can be returned to).
        let l = l.min(
            self.kv
                .as_ref()
                .expect("the cache is the engine's during a turn")
                .filled(),
        );
        // The DSpark head's own context goes back with the cache: it drafts
        // from the positions the target has, and no further.
        if let Some(d) = &mut self.dspark {
            d.truncate(l);
        }
        self.ctx.clear();
        self.ctx.extend_from_slice(&kept[..l]);
        // The suffix index's corpus is the conversation's tokens: it is rebuilt
        // from them, because a reset (or a shifted context) left it with the
        // ones the cache no longer holds. What it learned about how reliable
        // each kind of match is stays.
        self.reset_suffix_corpus();
        l
    }

    /// The suffix index's corpus is the conversation's tokens: after a restore
    /// from the store, or a reset, it is a different sequence, so the index is
    /// rebuilt from it.
    fn reset_suffix_corpus(&mut self) {
        let Some(d) = &mut self.suffix else {
            return;
        };
        let ctx = self.ctx.clone();
        d.reset();
        let pool = self.engine.pool();
        for (i, &t) in ctx.iter().enumerate() {
            d.commit(i, t, &[], &[], pool);
        }
    }

    // keep up to `budget_bytes` of finished conversations' caches (0 = off). Ignored with a drafter attached.
    pub fn enable_prefix_store(&mut self, budget_bytes: usize) {
        // snapshots hold no recurrent state, so a hybrid model cannot use the store.
        self.prefixes = (budget_bytes > 0 && self.dspark.is_none() && !self.model.is_hybrid())
            .then(|| crate::prefix::PrefixStore::new(budget_bytes));
    }

    // (entries, bytes, restores, positions not prefilled), or None when the store is off.
    pub fn prefix_store_stats(&self) -> Option<(usize, usize, u64, u64)> {
        self.prefixes
            .as_ref()
            .map(|p| (p.entries(), p.bytes(), p.hits, p.positions_saved))
    }

    /// Exchanges currently held in the conversation.
    // re-seed the sampler for the next turn (`seed` is otherwise read once at open).
    pub fn reseed(&mut self, seed: u64) {
        self.cfg.seed = seed;
        self.seed = if seed == 0 { self.seed } else { seed };
    }

    pub fn exchanges(&self) -> usize {
        self.history.len()
    }

    /// Resume an earlier conversation: the exchanges are replayed from an empty
    /// cache on the next turn (`plan_prompt` re-encodes the full transcript when
    /// nothing is started), so a reopened chat continues where it left off.
    pub fn seed_history(&mut self, exchanges: Vec<(String, String)>) {
        self.reset();
        self.history = exchanges;
    }

    pub fn context_used(&self) -> usize {
        self.ctx.len()
    }

    pub fn kv_resident_bytes(&self) -> u64 {
        self.kv.as_ref().map_or(0, |kv| kv.resident_bytes())
    }

    /// The system message in force: the caller's, else the one the model's own
    /// template would insert (SmolLM and Qwen name themselves; Llama 3 has none).
    fn system_text(&self) -> Option<&str> {
        self.cfg.system.as_deref().or(self.default_system)
    }

    /// The template the file declares (`chatml` or `llama3`).
    pub fn template(&self) -> Template {
        self.template
    }

    /// The text the model's own chat template produces for one more user message.
    ///
    /// On a continuing conversation the text begins by *closing the previous
    /// assistant message*. A turn ends when the model samples the end-of-turn token,
    /// and that token is emitted but never fed — so without this the cache held
    /// `…Paris.<|im_start|>user` and the model saw its own answer run straight into
    /// the next question, a transcript its template never produces.
    fn build_prompt(&self, user: &str) -> String {
        let mut s = String::new();
        if !self.started {
            s.push_str(
                &self
                    .template
                    .system_block(self.system_text(), &today_llama3()),
            );
            // The system message, folded into this (the first) user turn for the
            // templates with no system branch of their own.
            let u = match self.template.system_fold(self.system_text()) {
                Some(sys) => format!("{sys}\n{user}"),
                None => user.to_string(),
            };
            s.push_str(&self.template.user_turn(&u));
        } else {
            s.push_str(self.template.close_previous());
            s.push_str(&self.template.user_turn(user));
        }
        s.push_str(self.template.assistant_open(self.cfg.thinking));
        if let Some(p) = &self.cfg.assistant_prefix {
            s.push_str(p);
        }
        s
    }

    /// The whole conversation from the start: system, the kept exchanges, then the
    /// new user message. Used after dropping the oldest exchanges.
    fn full_prompt(&self, kept: &[(String, String)], user: &str) -> String {
        // one builder, shared with the batching engine, so both paths produce the same prompt.
        crate::serve::chat_prompt(
            self.template,
            self.system_text(),
            kept,
            user,
            self.cfg.thinking,
            self.cfg.assistant_prefix.as_deref(),
        )
    }

    /// The tokens to prefill for `user`, and how many old exchanges had to go.
    ///
    /// The incremental prompt is used whenever it fits with room left to answer.
    /// Otherwise the oldest exchanges are dropped one at a time and the remainder is
    /// re-encoded from an empty cache — the same policy the Colibri and Foundry
    /// adapters apply, so a long conversation degrades by forgetting its beginning
    /// rather than by refusing to answer. Only a single message that cannot fit even
    /// alone is an error.
    fn plan_prompt(&mut self, user: &str) -> Result<(Vec<u32>, usize), String> {
        let reserve = answer_reserve(&self.cfg);
        let max = self.cfg.max_ctx;
        self.prefix_reused = 0;
        let inc = if !self.started && !self.history.is_empty() {
            // Seeded history: the whole transcript is the prompt, less what the cache
            // still holds of it.
            let toks = self
                .model
                .tk
                .encode_prompt(&self.full_prompt(&self.history.clone(), user));
            let l = self.reuse_prefix(&toks);
            self.prefix_reused = l;
            toks[l..].to_vec()
        } else if !self.started {
            // A fresh or reset conversation: the system block may be in the cache.
            let toks = self.model.tk.encode_prompt(&self.build_prompt(user));
            let l = self.reuse_prefix(&toks);
            self.prefix_reused = l;
            toks[l..].to_vec()
        } else {
            self.model.tk.encode(&self.build_prompt(user))
        };
        if self.ctx.len() + inc.len() + reserve <= max {
            return Ok((inc, 0));
        }
        for drop in 0..=self.history.len() {
            let toks = self
                .model
                .tk
                .encode_prompt(&self.full_prompt(&self.history[drop..], user));
            let last = drop == self.history.len();
            if toks.len() + reserve <= max || (last && toks.len() + 1 < max) {
                self.history.drain(..drop);
                // The re-encoded transcript shares its system block (and, when few
                // exchanges went, more) with the cache: keep that, prefill the rest.
                if self.kept.is_empty() {
                    self.kept = std::mem::take(&mut self.ctx);
                }
                let l = self.reuse_prefix(&toks);
                self.prefix_reused = l;
                self.started = false;
                return Ok((toks[l..].to_vec(), drop));
            }
        }
        let alone = self
            .model
            .tk
            .encode_prompt(&self.full_prompt(&[], user))
            .len();
        Err(format!(
            "this message needs {alone} positions on its own, more than the {max}-token \
             context holds — raise the context length in Settings"
        ))
    }

    /// The drafter this turn speculates with, and how deep it may draft.
    ///
    /// The DSpark head comes first: with one attached the suffix index is off, as
    /// it always was. The head is wrapped in the interface the engine asks, and the
    /// wrapper is handed back when the turn ends — the head's context (the tapped
    /// hidden states it drafts from) belongs to the conversation, not to the turn.
    fn take_drafter(&mut self) -> (Option<Box<dyn crate::draft::Drafter>>, usize, bool) {
        #[cfg(feature = "dspark")]
        {
            if let Some(head) = self.dspark.take() {
                return (
                    Some(Box::new(crate::draft::DsparkDrafter::new(head))),
                    self.cfg.dspark_max,
                    true,
                );
            }
        }
        match self.suffix.take() {
            Some(d) => (Some(d), self.cfg.draft_len, false),
            None => (None, 0, false),
        }
    }

    /// The drafter the turn ended with: the conversation keeps it.
    ///
    /// `is_dspark` says which drafter it is: the conversation attached it, so
    /// it knows. The DSpark head comes back out of its wrapper, with the taps
    /// the head had not read yet folded into it.
    fn put_drafter(&mut self, d: Box<dyn crate::draft::Drafter>, is_dspark: bool) {
        #[cfg(feature = "dspark")]
        if is_dspark {
            let any: Box<dyn std::any::Any> = d;
            let mut w = *any
                .downcast::<crate::draft::DsparkDrafter>()
                .expect("the drafter is the head's wrapper");
            // The taps of the positions the turn committed but the head never read
            // go into the head now: the next turn drafts from them.
            w.flush(self.engine.pool());
            self.dspark = Some(w.into_inner());
            return;
        }
        #[cfg(not(feature = "dspark"))]
        let _ = is_dspark;
        self.suffix = Some(d);
    }

    /// The engine hands the conversation back what the turn left: its cache,
    /// the tokens the cache holds and the drafter that drafted for it.
    fn take_back(&mut self, is_dspark: bool) -> Result<(), String> {
        let keep = self
            .engine
            .take_kept()
            .ok_or("the engine lost the turn's cache")?;
        self.kv = Some(keep.kv);
        self.ctx = keep.tokens;
        if let Some(d) = keep.drafter {
            self.put_drafter(d, is_dspark);
        }
        Ok(())
    }

    /// Run one conversational turn. `on_token` sees text as it is produced.
    pub fn generate<F: FnMut(Emitted)>(
        &mut self,
        user: &str,
        on_token: F,
    ) -> Result<(String, TurnTelem), String> {
        self.generate_with(user, None, on_token)
    }

    /// As `generate`, stopping cleanly as soon as `cancel` is set. The flag is
    /// checked between prefill chunks and before every decode step, so a Stop takes
    /// effect within one step instead of after the whole answer — which at 1 024
    /// tokens was most of a minute of a core-saturating thread nobody was reading.
    /// A cancelled turn returns what it produced, marked `cancelled`, and the
    /// conversation stays consistent for the next message.
    pub fn generate_with<F: FnMut(Emitted)>(
        &mut self,
        user: &str,
        cancel: Option<&AtomicBool>,
        mut on_token: F,
    ) -> Result<(String, TurnTelem), String> {
        let turn_start = Instant::now();
        self.turn += 1;
        let mut telem = TurnTelem {
            ctx_max: self.cfg.max_ctx,
            ..Default::default()
        };
        let stopped = |c: Option<&AtomicBool>| c.is_some_and(|f| f.load(Ordering::Relaxed));

        // ---- the prompt ----
        let (ptoks, dropped) = self.plan_prompt(user)?;
        telem.dropped_exchanges = dropped;
        telem.prefix_reused = self.prefix_reused;
        let started_before = self.started;
        self.started = true;
        telem.prompt_tokens = ptoks.len();
        // Where the turn starts: a cancelled turn goes back to here.
        let turn_start_pos = self.ctx.len();

        // The whole prompt, for the engine: the positions the cache already holds
        // and the ones this turn adds.
        let mut prompt = std::mem::take(&mut self.ctx);
        prompt.extend_from_slice(&ptoks);

        // ---- the slot ----
        let cfg = SlotCfg {
            temperature: self.cfg.temperature,
            top_p: self.cfg.top_p,
            top_k: self.cfg.top_k,
            min_p: self.cfg.min_p,
            penalties: Default::default(),
            seed: self.seed,
            max_tokens: self.cfg.max_tokens,
            stop: self.stops.clone(),
            // The caller wants the log-probability of every token.
            logprobs: true,
            // The conversation's own turn in the sampling key, and the family's own
            // KV cache (ADR 0007, decision 5).
            turn: self.turn as u32,
            kv_rotation: self.model.session_kv_rotation(),
            // The drafter is attached below: the conversation's own, kept between
            // its turns.
            drafter: Default::default(),
        };
        // A hybrid model's recurrent state rolls back only to its checkpoint: the
        // turn may be cancelled, and the cache must be able to go back.
        if self.model.is_hybrid() {
            self.kv
                .as_mut()
                .expect("the cache is the engine's during a turn")
                .checkpoint();
        }
        let kv = self
            .kv
            .take()
            .expect("the cache is the engine's during a turn");
        let id = match self.engine.admit_with_cache(prompt, cfg, kv) {
            Ok(id) => id,
            Err((_prompt, e)) => {
                // The turn did not start: the conversation keeps its cache.
                if let Some(keep) = self.engine.take_kept() {
                    self.kv = Some(keep.kv);
                }
                return Err(e);
            }
        };
        // The conversation's drafter, for this turn.
        let (drafter, depth, is_dspark) = self.take_drafter();
        if let Some(d) = drafter {
            if let Err(e) = self.engine.attach_drafter_with_depth(id, d, depth) {
                // The turn cannot run: the conversation takes its cache and its
                // drafter back.
                self.engine.cancel(id);
                let _ = self.take_back(is_dspark);
                return Err(e);
            }
        }
        // The engine's counters are cumulative: the turn's share is the difference.
        let rounds_before = self.engine.spec_rounds;
        let tokens_before = self.engine.spec_tokens;
        #[cfg(feature = "dspark")]
        let head_ns = self.dspark.as_ref().map_or(0, |d| d.stats.ns);
        #[cfg(feature = "dspark")]
        let head_ctx_ns = self.dspark.as_ref().map_or(0, |d| d.stats.ctx_ns);

        // ---- decode ----
        let mut out = String::new();
        let mut utf8 = Utf8Stream::default();
        let mut step = 0u32;
        let mut decoded = false;
        let mut roll_back = false;

        while self.engine.active() > 0 {
            if stopped(cancel) {
                telem.cancelled = true;
                // Nothing of this turn is in the conversation yet: the cache goes
                // back to where the turn started, so the next message sees a clean
                // context. A turn that produced something keeps it, as it always did.
                roll_back = !decoded;
                self.engine.cancel(id);
                break;
            }
            let ems = self.engine.step();
            let rep = self.engine.last_step;
            if ems.is_empty() {
                // A chunk of the prompt: its cost is the prompt's.
                telem.prefill_phases.add_step(&rep.stats);
                telem.prefill_s += rep.secs;
                telem.prefill_bytes += rep.stats.bytes_total();
                telem.prefill_macs += rep.stats.macs;
                continue;
            }
            // A decode step.
            decoded = true;
            telem.decode_phases.add_step(&rep.stats);
            telem.decode_steps_s += rep.secs;
            let n_em = ems.len();
            let mut emitted = 0usize;
            for (i, em) in ems.iter().enumerate() {
                match &em.finish {
                    Some(Finish::Error(e)) => {
                        // The step failed: the turn ends, and the conversation
                        // takes its cache and its drafter back.
                        self.engine.cancel(id);
                        let _ = self.take_back(is_dspark);
                        return Err(e.clone());
                    }
                    // The token that ends the turn is the end of the answer, not
                    // part of it.
                    Some(Finish::Stop) => break,
                    _ => {}
                }
                emitted += 1;
                step += 1;
                let tt = TokenTelem {
                    step,
                    dt_s: rep.secs / n_em as f64,
                    bytes_weights: rep.stats.bytes_weights / n_em as u64,
                    bytes_kv: rep.stats.bytes_kv / n_em as u64,
                    macs: rep.stats.macs / n_em as u64,
                    barriers: rep.stats.barriers / n_em as u64,
                    drafted: if i == 0 { rep.drafted as u8 } else { 0 },
                    accepted: if i == 0 { rep.accepted as u8 } else { 0 },
                    from_gemm: rep.rows > 1,
                    is_prefill: false,
                };
                let text = utf8.push(&self.model.tk.token_bytes(em.token));
                out.push_str(&text);
                telem.tokens.push(tt);
                on_token(Emitted {
                    id: em.token,
                    text: &text,
                    telem: telem.tokens.last().unwrap(),
                    logprob: em.logprob,
                });
            }
            // A pass counts toward tokens-per-pass only when it emitted a token.
            if emitted > 0 {
                telem.passes += 1;
            }
        }
        let tail = utf8.finish();
        out.push_str(&tail);

        // ---- what the turn left behind ----
        self.take_back(is_dspark)?;
        if roll_back {
            // The turn left nothing in the conversation: the cache, the drafter's
            // context and the conversation itself go back to where it started.
            let pos = turn_start_pos.min(self.ctx.len());
            self.kv
                .as_mut()
                .expect("the cache is the engine's during a turn")
                .truncate(pos);
            if let Some(d) = &mut self.suffix {
                d.rollback(pos);
            }
            if let Some(d) = &mut self.dspark {
                d.truncate(pos);
            }
            self.ctx.truncate(pos);
            self.started = started_before;
        }
        if is_dspark {
            telem.dspark_rounds = self.engine.spec_rounds - rounds_before;
            telem.dspark_verified = self.engine.spec_tokens - tokens_before;
        }
        #[cfg(feature = "dspark")]
        if let Some(d) = &mut self.dspark {
            telem.dspark_bytes = std::mem::take(&mut d.stats.bytes);
            telem.dspark_draft_s = d.stats.ns.saturating_sub(head_ns) as f64 / 1e9;
            telem.dspark_ctx_s = d.stats.ctx_ns.saturating_sub(head_ctx_ns) as f64 / 1e9;
        }

        // A turn that produced nothing is not in the conversation: it is not an
        // exchange, and the next message must not see it.
        if !roll_back {
            self.history.push((user.to_string(), out.clone()));
        }
        telem.ctx_used = self.ctx.len();
        telem.wall_s = turn_start.elapsed().as_secs_f64();
        Ok((out, telem))
    }

    /// Prefill a fixed synthetic prompt of `prompt_tokens` and decode `decode_tokens` greedy tokens.
    /// Returns prefill tok/s, decode tok/s, and bytes/token (from telemetry).
    pub fn bench(
        &mut self,
        prompt_tokens: usize,
        decode_tokens: usize,
    ) -> Result<BenchReport, String> {
        const PROMPT_SEED: [u32; 24] = [
            11, 300, 42, 7, 199, 256, 3, 88, 120, 5, 64, 31, 2, 270, 150, 9, 77, 1, 18, 240, 33,
            61, 100, 12,
        ];
        let n_vocab = self.model.cfg.n_vocab;
        let p = prompt_tokens.max(1);
        let ptoks: Vec<u32> = (0..p)
            .map(|i| PROMPT_SEED[i % PROMPT_SEED.len()] % n_vocab as u32)
            .collect();

        if ptoks.len() + decode_tokens > self.cfg.max_ctx {
            return Err(format!(
                "benchmark requires {} context positions ({} prompt + {} decode), but max_ctx is {}",
                ptoks.len() + decode_tokens,
                ptoks.len(),
                decode_tokens,
                self.cfg.max_ctx
            ));
        }

        self.reset();
        self.ctx.clear();

        // The scratch the benchmark needs is its own: the engine's state is the
        // engine's, and the benchmark runs between the turns.
        let mut st = self.model.new_state(
            crate::serve::PREFILL_CHUNK,
            1,
            self.cfg.max_ctx,
            self.cfg.threads,
        );

        // Prefill
        let mut pre = crate::model::StepStats::default();
        let t_pre = Instant::now();
        let mut fed = 0;
        while fed < ptoks.len() {
            let take = crate::serve::PREFILL_CHUNK.min(ptoks.len() - fed);
            let last = fed + take == ptoks.len();
            let start = fed;
            let want: &[usize] = if last { &[take - 1] } else { &[] };
            let model = &self.model;
            let pool = self.engine.pool();
            let kv = self
                .kv
                .as_mut()
                .expect("the cache is the engine's during a turn");
            model.forward(
                &mut st,
                kv,
                pool,
                &ptoks[fed..fed + take],
                start,
                want,
                &mut pre,
            )?;
            fed += take;
        }
        let prefill_s = t_pre.elapsed().as_secs_f64();
        self.ctx.extend_from_slice(&ptoks);

        // First greedy token from prefill logits
        let argmax = |row: &[f32]| -> u32 {
            let mut best = 0;
            for (i, &x) in row.iter().enumerate() {
                if x > row[best] {
                    best = i;
                }
            }
            best as u32
        };
        let mut next = argmax(&st.logits[..n_vocab]);

        // Decode
        let mut decode_s = 0.0;
        let mut total_bytes_weights = 0u64;
        let mut total_bytes_kv = 0u64;
        let start_pos = self.ctx.len();

        for step in 0..decode_tokens {
            let td = Instant::now();
            let mut ss = crate::model::StepStats::default();
            let pos = start_pos + step;
            let model = &self.model;
            let p = self.engine.narrow_pool().unwrap_or(self.engine.pool());
            let kv = self
                .kv
                .as_mut()
                .expect("the cache is the engine's during a turn");
            model.forward(&mut st, kv, p, &[next], pos, &[0], &mut ss)?;
            decode_s += td.elapsed().as_secs_f64();
            total_bytes_weights += ss.bytes_weights;
            total_bytes_kv += ss.bytes_kv;
            next = argmax(&st.logits[..n_vocab]);
        }

        let prefill_tok_s = if prefill_s > 0.0 {
            ptoks.len() as f64 / prefill_s
        } else {
            0.0
        };
        let decode_tok_s = if decode_s > 0.0 {
            decode_tokens as f64 / decode_s
        } else {
            0.0
        };
        let bytes_per_token = if decode_tokens > 0 {
            (total_bytes_weights + total_bytes_kv) as f64 / decode_tokens as f64
        } else {
            0.0
        };

        Ok(BenchReport {
            prompt_tokens: ptoks.len(),
            decode_tokens,
            prefill_s,
            decode_s,
            prefill_tok_s,
            decode_tok_s,
            bytes_per_token,
        })
    }
}

/// What one benchmark measurement produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchReport {
    pub prompt_tokens: usize,
    pub decode_tokens: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub bytes_per_token: f64,
}
