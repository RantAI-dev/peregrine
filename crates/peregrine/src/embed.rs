//! `peregrine embed`: one vector per text, from a causal model's final hidden states (S4.1).
//!
//! The CLI is the same engine the server runs: every text is a pooled slot, admitted together
//! so they share weight passes, and the vector comes back in input order. The pooling type is
//! the file's own unless `--pooling` names one; the vector is truncated to `--dimensions` and
//! L2-normalised, exactly as `/v1/embeddings` returns it.

use clap::Args;
use peregrine_core::embed::{self, Pooling};
use peregrine_core::kv::KvPrec;
use peregrine_core::options;
use peregrine_core::pool::physical_cores;
use peregrine_core::profile;
use peregrine_core::serve::Engine;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;

/// The most slots one `embed` call keeps busy (a batch bigger than this is admitted in waves).
const MAX_EMBED_SLOTS: usize = 64;

/// KV-cache memory the batches may hold at once (each slot owns a cache for the whole context).
const KV_BUDGET_BYTES: usize = 1 << 30;

/// Arguments for `peregrine embed`.
#[derive(Args, Debug, Clone)]
pub struct EmbedArgs {
    /// The GGUF file, or `hf:<namespace>/<repo>/<file>[@<rev>]` to download it into the cache once.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// The texts. With none, one text per stdin line is embedded, for pipes and scripts.
    #[arg(value_name = "TEXT")]
    pub text: Vec<String>,
    /// Truncate to this many dimensions before normalising (Matryoshka).
    #[arg(long)]
    pub dimensions: Option<usize>,
    /// Print one JSON object (`{"object":"list","data":[...],"usage":...}`, as the API returns it)
    /// instead of one line of space-separated floats per text.
    #[arg(long, default_value_t = false)]
    pub json: bool,
    /// The pooling to use, overriding the file's `<arch>.pooling_type`: `last` or `mean`.
    #[arg(long)]
    pub pooling: Option<String>,
    /// Context length in tokens (clamped to the model's training context). A text longer than
    /// the limit is an error unless `--truncate` is given. The batch's caches are sized to the
    /// longest text, not to this limit.
    #[arg(short = 'c', long, default_value_t = 8192)]
    pub ctx: usize,
    /// Truncate an over-length text to the context instead of refusing it.
    #[arg(long, default_value_t = false)]
    pub truncate: bool,
    /// Worker threads (default: the calibrated decode width).
    #[arg(short = 't', long)]
    pub threads: Option<usize>,
    /// A tuning profile file to apply (default: the profile for this model and machine).
    #[arg(long)]
    pub profile: Option<PathBuf>,
    /// Ignore tuning profiles; use environment variables and defaults.
    #[arg(long, default_value_t = false)]
    pub no_profile: bool,
}

/// Run `peregrine embed`.
pub fn run(args: &EmbedArgs) -> Result<(), String> {
    let texts = read_texts(args)?;
    if texts.is_empty() {
        return Err("no text given (pass arguments, or one text per stdin line)".into());
    }
    let (prof, prof_msg) = profile::resolve(&args.model, args.profile.as_deref(), args.no_profile);
    if prof.is_some() {
        eprintln!("peregrine: {prof_msg}");
    }
    let opts = options::EngineOptions::from_env_and_profile(prof.as_ref());
    options::init(opts.clone()).map_err(|e| format!("engine options: {e}"))?;
    let threads = args
        .threads
        .or(opts.decode_threads)
        .unwrap_or_else(|| physical_cores().unwrap_or(1))
        .max(1);

    let model = peregrine_models::open(&args.model)
        .map_err(|e| format!("cannot open {}: {e}", args.model.display()))?;
    let pooling = match args.pooling.as_deref() {
        Some(name) => Pooling::parse(name).ok_or_else(|| {
            format!("unknown pooling {name:?} — last or mean (cls is not implemented for causal models)")
        })?,
        None => model.pooling()?.ok_or_else(|| {
            format!(
                "{} declares no {}.pooling_type (pooling \"none\"); name one with --pooling last|mean",
                args.model.display(),
                model.cfg.arch_name
            )
        })?,
    };
    let bidirectional = model.causality() == peregrine_core::arch::Causality::Bidirectional;
    if pooling == Pooling::Cls && !bidirectional {
        return Err("cls pooling is not implemented for causal models".into());
    }
    if let Some(d) = args.dimensions {
        if d == 0 || d > model.embedding_width() {
            return Err(format!(
                "--dimensions must be from 1 to {}",
                model.embedding_width()
            ));
        }
    }

    let max_ctx = args.ctx.clamp(32, model.cfg.ctx_train);
    let mut tokens: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
    for (i, t) in texts.iter().enumerate() {
        let mut v = model.tk.encode_special(t);
        if v.is_empty() {
            return Err(format!("text {i} produced no tokens"));
        }
        if v.len() > max_ctx {
            if args.truncate {
                v.truncate(max_ctx);
            } else {
                return Err(format!(
                    "text {i} is {} tokens; this model is served with {max_ctx} positions (pass --truncate to truncate)",
                    v.len()
                ));
            }
        }
        tokens.push(v);
    }

    // All texts are slots at once when they fit: the engine batches whatever it has room for
    // and admits the rest as earlier ones finish. Each slot owns a KV cache of the engine's
    // context, and a pooled slot never decodes, so the engine's context is the longest text's
    // length (at most `--ctx`): a batch of short texts can then take many slots inside the KV
    // budget instead of one full-8k cache per text.
    let ctx = tokens
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
        .clamp(32, max_ctx);
    let slots = if bidirectional {
        // An encoder's slots hold no cache: one text costs its own rows, not a
        // `max_ctx`-position cache each.
        tokens.len().clamp(1, MAX_EMBED_SLOTS)
    } else {
        let per_slot = model.new_cache(KvPrec::F16, 1).bytes_per_position() as usize * ctx;
        tokens
            .len()
            .clamp(1, MAX_EMBED_SLOTS)
            .min(KV_BUDGET_BYTES / per_slot.max(1))
            .max(1)
    };
    let mut engine = Engine::new(model.clone(), threads, ctx, slots, KvPrec::F16, 0);
    let mut by_slot: HashMap<u64, usize> = HashMap::new();
    let mut next = 0usize;
    let mut done: Vec<Option<Vec<f32>>> = vec![None; tokens.len()];
    let admit = |engine: &mut Engine, by_slot: &mut HashMap<u64, usize>, i: usize| {
        let id = engine
            .admit_pooled(
                tokens[i].clone(),
                peregrine_core::serve::SlotCfg::default(),
                pooling,
            )
            .map_err(|(_, e)| e)?;
        by_slot.insert(id, i);
        Ok::<(), String>(())
    };
    while next < tokens.len() && engine.has_room() {
        admit(&mut engine, &mut by_slot, next)?;
        next += 1;
    }
    let mut remaining = tokens.len();
    while remaining > 0 {
        for em in engine.step() {
            let Some(i) = by_slot.remove(&em.slot) else {
                continue;
            };
            if let Some(peregrine_core::serve::Finish::Error(e)) = em.finish {
                return Err(e);
            }
            let Some(mut v) = em.embedding else {
                return Err("the engine returned no embedding".into());
            };
            embed::finish(&mut v, args.dimensions)?;
            done[i] = Some(v);
            remaining -= 1;
        }
        while next < tokens.len() && engine.has_room() {
            admit(&mut engine, &mut by_slot, next)?;
            next += 1;
        }
    }
    let vectors: Vec<Vec<f32>> = done
        .into_iter()
        .map(|v| v.expect("every text was admitted"))
        .collect();

    if args.json {
        let data: Vec<serde_json::Value> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| serde_json::json!({"object": "embedding", "index": i, "embedding": v}))
            .collect();
        let prompt_tokens: usize = tokens.iter().map(Vec::len).sum();
        let out = serde_json::json!({
            "object": "list",
            "data": data,
            "model": args.model.display().to_string(),
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        });
        println!(
            "{}",
            serde_json::to_string(&out).map_err(|e| format!("cannot format JSON: {e}"))?
        );
    } else {
        let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
        for v in &vectors {
            let line: Vec<String> = v.iter().map(|x| x.to_string()).collect();
            writeln!(stdout, "{}", line.join(" ")).map_err(|e| e.to_string())?;
        }
        stdout.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The texts to embed: the arguments, or one per stdin line when none are given.
fn read_texts(args: &EmbedArgs) -> Result<Vec<String>, String> {
    if !args.text.is_empty() {
        return Ok(args.text.clone());
    }
    let mut s = String::new();
    std::io::stdin()
        .read_to_string(&mut s)
        .map_err(|e| format!("cannot read stdin: {e}"))?;
    Ok(s.lines().map(str::to_string).collect())
}
