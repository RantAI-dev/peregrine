//! `peregrine rerank`: rank documents by their relevance to a query (S4.6b).
//!
//! A `bert` reranker carries a classifier head (`cls`, `cls.output`). Each (query, document)
//! pair is one text, built by the tokenizer's pair encoding exactly as llama.cpp's
//! `format_prompt_rerank` builds it — `[CLS] query [SEP] document` with a WordPiece
//! vocabulary, `<s> query </s> document </s>` with the XLM-R one — and scored by that head.
//! The score is the head's raw logit, as llama.cpp's `/v1/rerank` returns it; the documents
//! come back sorted by it, highest first.

use clap::Args;
use peregrine_core::kv::KvPrec;
use peregrine_core::options;
use peregrine_core::pool::physical_cores;
use peregrine_core::profile;
use peregrine_core::serve::Engine;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;

/// Arguments for `peregrine rerank`.
#[derive(Args, Debug, Clone)]
pub struct RerankArgs {
    /// The GGUF file, or `hf:<namespace>/<repo>/<file>[@<rev>]` to download it into the cache once.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// The query every document is scored against.
    #[arg(long)]
    pub query: String,
    /// The documents. With none, one document per stdin line is ranked.
    #[arg(value_name = "DOCUMENT")]
    pub documents: Vec<String>,
    /// Print only the first N documents.
    #[arg(long)]
    pub top_n: Option<usize>,
    /// Print one JSON object (`{"model":…,"results":[{"index":…,"relevance_score":…}]}`, as the
    /// API returns it) instead of one line per document.
    #[arg(long, default_value_t = false)]
    pub json: bool,
    /// Context length in tokens (clamped to the model's training context). A pair longer than
    /// the limit is an error unless `--truncate` is given.
    #[arg(short = 'c', long, default_value_t = 8192)]
    pub ctx: usize,
    /// Truncate an over-length pair to the context instead of refusing it.
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

/// Run `peregrine rerank`.
pub fn run(args: &RerankArgs) -> Result<(), String> {
    let documents = read_documents(args)?;
    if documents.is_empty() {
        return Err("no document given (pass arguments, or one document per stdin line)".into());
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
    if !model.has_head() {
        return Err(format!(
            "{} has no classifier head (cls / cls.output), so it cannot rerank; `peregrine embed` embeds texts with it",
            args.model.display()
        ));
    }
    if model.gguf().str("tokenizer.chat_template.rerank").is_ok() {
        return Err(
            "this file carries a \"rerank\" chat template (a generative reranker), which is not \
             supported yet; a bert reranker is scored with its classifier head"
                .into(),
        );
    }

    let max_ctx = args.ctx.clamp(32, model.cfg.ctx_train);
    let mut tokens: Vec<Vec<u32>> = Vec::with_capacity(documents.len());
    for (i, d) in documents.iter().enumerate() {
        let mut v = model.tk.encode_pair(&args.query, d);
        if v.is_empty() {
            return Err(format!("document {i} produced no tokens"));
        }
        if v.len() > max_ctx {
            if args.truncate {
                v.truncate(max_ctx);
            } else {
                return Err(format!(
                    "document {i} is {} tokens; this model is served with {max_ctx} positions (pass --truncate to truncate)",
                    v.len()
                ));
            }
        }
        tokens.push(v);
    }

    let prompt_tokens: usize = tokens.iter().map(Vec::len).sum();
    let mut engine = Engine::new(model.clone(), threads, max_ctx, 1, KvPrec::F16, 0);
    let mut by_slot: HashMap<u64, usize> = HashMap::new();
    let mut next = 0usize;
    let mut done: Vec<Option<f32>> = vec![None; tokens.len()];
    let admit = |engine: &mut Engine, by_slot: &mut HashMap<u64, usize>, i: usize| {
        let id = engine
            .admit_head(tokens[i].clone(), peregrine_core::serve::SlotCfg::default())
            .map_err(|(_, e)| e)?;
        by_slot.insert(id, i);
        Ok::<(), String>(())
    };
    while next < tokens.len() && engine.work_room_for(tokens[next].len()) {
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
            let Some(head) = em.head else {
                return Err("the engine returned no classifier head row".into());
            };
            let Some(&score) = head.first() else {
                return Err("the classifier head row is empty".into());
            };
            done[i] = Some(score);
            remaining -= 1;
        }
        while next < tokens.len() && engine.work_room_for(tokens[next].len()) {
            admit(&mut engine, &mut by_slot, next)?;
            next += 1;
        }
    }
    let scores: Vec<f32> = done
        .into_iter()
        .map(|s| s.expect("every document was admitted"))
        .collect();

    // Highest score first, ties by document index (the endpoint's order).
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let top_n = args.top_n.unwrap_or(order.len()).min(order.len());

    if args.json {
        let results: Vec<serde_json::Value> = order[..top_n]
            .iter()
            .map(|&i| serde_json::json!({"index": i, "relevance_score": scores[i]}))
            .collect();
        let out = serde_json::json!({
            "model": args.model.display().to_string(),
            "object": "list",
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            "results": results,
        });
        println!(
            "{}",
            serde_json::to_string(&out).map_err(|e| format!("cannot format JSON: {e}"))?
        );
    } else {
        let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
        for &i in &order[..top_n] {
            writeln!(stdout, "{:.6} {} {}", scores[i], i, documents[i])
                .map_err(|e| e.to_string())?;
        }
        stdout.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The documents to rank: the arguments, or one per stdin line when none are given.
fn read_documents(args: &RerankArgs) -> Result<Vec<String>, String> {
    if !args.documents.is_empty() {
        return Ok(args.documents.clone());
    }
    let mut s = String::new();
    std::io::stdin()
        .read_to_string(&mut s)
        .map_err(|e| format!("cannot read stdin: {e}"))?;
    Ok(s.lines().map(str::to_string).collect())
}
