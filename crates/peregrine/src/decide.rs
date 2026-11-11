//! `peregrine decide`: one answer per question about a state, from a native decision model
//! (S4.3c).
//!
//! The command is the same engine `/v1/systemone` runs: every question is a labels slot, admitted
//! together so they share weight passes, and the answers come back in request order. The request
//! is the API's body (`--state` and `--questions`, or a whole `--request` file), parsed with the
//! order-keeping JSON value, so the order of the questions and of each question's criteria is the
//! order of the answers and of the options.

use clap::Args;
use peregrine_core::decision::{self, Decider};
use peregrine_core::json::Value;
use peregrine_core::kv::KvPrec;
use peregrine_core::options;
use peregrine_core::pool::physical_cores;
use peregrine_core::profile;
use peregrine_core::serve::{Engine, Finish, SlotCfg};
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

/// The most slots one `decide` call keeps busy (more questions are admitted in waves).
const MAX_DECIDE_SLOTS: usize = 64;

/// KV-cache memory the batches may hold at once (each slot owns a cache for the whole context).
const KV_BUDGET_BYTES: usize = 1 << 30;

/// Arguments for `peregrine decide`.
#[derive(Args, Debug, Clone)]
pub struct DecideArgs {
    /// The GGUF file, or `hf:<namespace>/<repo>/<file>[@<rev>]` to download it into the cache once.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// The state (the text, object or array the questions are about): a value, `@file`, or `-` for
    /// stdin. Valid JSON is used as it is; anything else is a string.
    #[arg(long, value_name = "TEXT", required_unless_present = "request")]
    pub state: Option<String>,
    /// The questions object: JSON, or `@file`.
    #[arg(long, value_name = "JSON", required_unless_present = "request")]
    pub questions: Option<String>,
    /// A whole request body (`{"state":...,"questions":...}`): a file, or `-` for stdin.
    #[arg(long, value_name = "FILE", conflicts_with_all = ["state", "questions"])]
    pub request: Option<String>,
    /// Print the response JSON (the API's shape) instead of one line per answer.
    #[arg(long, default_value_t = false)]
    pub json: bool,
    /// Context length in tokens (clamped to the model's training context).
    #[arg(short = 'c', long, default_value_t = 8192)]
    pub ctx: usize,
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

/// Run `peregrine decide`.
pub fn run(args: &DecideArgs) -> Result<(), String> {
    let body = read_body(args)?;
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
    let decider = Decider::from_model(&model)?
        .ok_or_else(|| format!("{} is not a decision model", args.model.display()))?;
    let (state, questions) = decider.parse_request(&body)?;
    let max_ctx = args.ctx.clamp(32, model.cfg.ctx_train);
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(questions.len());
    for (id, question) in &questions {
        let tokens = decider.prompt_tokens(question, &state, &model.tk)?;
        if tokens.is_empty() {
            return Err(format!("question {id:?} produced no tokens"));
        }
        if tokens.len() >= max_ctx {
            return Err(format!(
                "question {id:?} is {} tokens; this model is run with {max_ctx} positions",
                tokens.len()
            ));
        }
        prompts.push(tokens);
    }

    let ctx = prompts
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
        .clamp(32, max_ctx);
    let per_slot = model.new_cache(KvPrec::F16, 1).bytes_per_position() as usize * ctx;
    let slots = prompts
        .len()
        .clamp(1, MAX_DECIDE_SLOTS)
        .min(KV_BUDGET_BYTES / per_slot.max(1))
        .max(1);
    let mut engine = Engine::new(model.clone(), threads, ctx, slots, KvPrec::F16, 0);
    let labels = decider.labels();
    let mut by_slot: HashMap<u64, usize> = HashMap::new();
    let mut next = 0usize;
    let mut done: Vec<Option<Vec<f32>>> = vec![None; prompts.len()];
    let admit = |engine: &mut Engine, by_slot: &mut HashMap<u64, usize>, i: usize| {
        let id = engine
            .admit_labels(
                prompts[i].clone(),
                SlotCfg::default(),
                labels[..questions[i].1.options.len()].to_vec(),
            )
            .map_err(|(_, e)| e)?;
        by_slot.insert(id, i);
        Ok::<(), String>(())
    };
    while next < prompts.len() && engine.has_room() {
        admit(&mut engine, &mut by_slot, next)?;
        next += 1;
    }
    let mut remaining = prompts.len();
    while remaining > 0 {
        for em in engine.step() {
            let Some(i) = by_slot.remove(&em.slot) else {
                continue;
            };
            if let Some(Finish::Error(e)) = em.finish {
                return Err(e);
            }
            let Some(scores) = em.label_logits else {
                return Err("the engine returned no label logits".into());
            };
            done[i] = Some(scores);
            remaining -= 1;
        }
        while next < prompts.len() && engine.has_room() {
            admit(&mut engine, &mut by_slot, next)?;
            next += 1;
        }
    }

    let info = decider.info();
    let input_tokens = prompts.iter().map(Vec::len).sum();
    let mut answers = Vec::with_capacity(questions.len());
    for ((id, question), scores) in questions.iter().zip(done) {
        let scores = scores.expect("every question was admitted");
        let answer = decision::format_answer(
            info.kind(),
            question,
            &[scores],
            info.temperatures(),
            info.score_bias(),
        )?;
        answers.push((id.clone(), answer));
    }
    let response =
        decision::response_json(&args.model.display().to_string(), &answers, input_tokens);
    if args.json {
        println!("{}", response.to_jinja_json());
    } else {
        for (id, answer) in &answers {
            println!("{id}: {}", line(answer));
        }
    }
    Ok(())
}

/// One answer as the one-line form: `choice (confidence)`, `score` or `probability`.
fn line(answer: &decision::Answer) -> String {
    match answer {
        decision::Answer::Choice {
            choice, confidence, ..
        } => format!(
            "{choice} ({})",
            peregrine_core::json::float_repr(*confidence)
        ),
        decision::Answer::Score { score, .. } => peregrine_core::json::float_repr(*score),
        decision::Answer::Noul { noul } => peregrine_core::json::float_repr(*noul),
    }
}

/// The request body: `--request` as it is, or `--state` and `--questions` composed in the API's
/// shape.
fn read_body(args: &DecideArgs) -> Result<Value, String> {
    if let Some(request) = &args.request {
        let text = read_source(request)?;
        return Value::parse(&text).map_err(|e| format!("--request: {e}"));
    }
    let state_text = read_source(args.state.as_deref().unwrap_or_default())?;
    // A state is any JSON value; text that is not JSON is the string the caller wrote.
    let state = Value::parse(state_text.trim()).unwrap_or(Value::Str(state_text));
    let questions_text = read_source(args.questions.as_deref().unwrap_or_default())?;
    let questions =
        Value::parse(&questions_text).map_err(|e| format!("--questions is not JSON: {e}"))?;
    Ok(Value::Obj(vec![
        ("state".to_string(), state),
        ("questions".to_string(), questions),
    ]))
}

/// A command-line value: the text itself, `@file`, or `-` for stdin.
fn read_source(value: &str) -> Result<String, String> {
    if value == "-" {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .map_err(|e| format!("cannot read stdin: {e}"))?;
        return Ok(s);
    }
    match value.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}")),
        None => Ok(value.to_string()),
    }
}
