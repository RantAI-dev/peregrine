//! `peregrine-server`: one GGUF model served by the Peregrine engine behind the subset of the OpenAI chat API
//! that llama-server-compatible clients speak: `POST /v1/chat/completions` (streamed or not, with `usage` and
//! llama-style `timings`), `GET /v1/models` (the health check), `GET /health`, `GET /metrics`.
//!
//! What it maps, and how:
//! - `messages` -> a Peregrine session: the system message is the session's system text, earlier user/assistant
//!   pairs are seeded as history, the last user message is the turn. The session is reset per request; its
//!   cache prefix is reused when the new prompt shares it (the system prompt, the agent's prompt).
//! - the chat template -> the one the model file declares, or the one `--chat-template` names for a file that
//!   declares none (a model prompted in a template it was not trained on answers fluently and wrongly).
//! - `chat_template_kwargs.enable_thinking` -> `SessionCfg::thinking` (Qwen3's empty reasoning block when off),
//!   and `reasoning_effort: "none"` switches it off too, as llama-server does (`tools/server/server-common.cpp`).
//! - `reasoning_format` (and `--reasoning-format`) -> how a reasoning reply is split: `none` leaves the
//!   analysis text in `content`, `auto`/`deepseek` put it in `reasoning_content`, and `deepseek-legacy` also
//!   mirrors it into streaming `content` deltas. gpt-oss's Harmony channels are parsed by `harmony.rs`; Qwen3's
//!   `<think>` span uses the same switch. The control tokens never appear in either field.
//! - `thinking_budget_tokens` -> the reasoning span is counted by token; at the budget the turn is cancelled and
//!   replanned with the reasoning so far plus `</think>` as the assistant's prefix (`SessionCfg::assistant_prefix`,
//!   a Peregrine addition), so the answer follows as llama-server's budget forcing does. This is Qwen3's
//!   `</think>` budget: Harmony has no such closing token, so the option is not applied to it.
//! - `logprobs` -> each content token's log-probability (`Emitted::logprob`), which a client's
//!   fixed rule reads.
//! - `ignore_eos` -> optional non-standard boolean (default false), matching llama.cpp's
//!   meaning: end-of-generation tokens do not stop generation; it continues until `max_tokens`.
//! - `n > 1` runs one generation per choice, in its own slot (so n choices share one weight pass);
//!   with `stream` it is refused. `grammar` is refused (400): the engine has no constrained sampler.
//!
//! One session per process (one stream at a time; requests queue): run one process per model, as with llama-server.
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use peregrine_core::model::Model;
use peregrine_core::sample::Penalties;
use peregrine_core::serve::{Emission, Engine, Finish, SlotCfg, SlotId};
use peregrine_core::template::Template;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

mod harmony;

use harmony::{Channel, Harmony, ReasoningFormat};

/// The server's command line (also `peregrine serve`).
#[derive(Parser, Debug)]
#[command(about = "Serve a GGUF model with Peregrine behind the OpenAI chat API")]
pub struct Args {
    /// The GGUF file (llama, qwen2 or qwen3; F32/F16/Q8_0/Q5_0/Q4_K/Q6_K tensors).
    #[arg(short = 'm', long)]
    pub model: std::path::PathBuf,
    /// An engine option, as `NAME=VALUE` with the name of its `PEREGRINE_<NAME>` variable (for example
    /// `--opt MTP=0`). Overrides the environment; repeatable. Every option is listed at startup.
    #[arg(long = "opt", value_name = "NAME=VALUE")]
    pub opts: Vec<String>,
    /// The model id reported by /v1/models and expected in requests (any id is accepted).
    #[arg(long, default_value = "peregrine")]
    pub alias: String,
    /// The chat template to use, by name (chatml, llama3, qwen3, gemma, mistral-v0,
    /// mistral-v1, mistral-v7, llama2, llama2-sys-bos). It overrides the template the
    /// file declares; a file that declares none needs it, since a model prompted in
    /// a template it was not trained on answers fluently and wrongly.
    #[arg(long = "chat-template", value_name = "NAME")]
    pub chat_template: Option<String>,
    /// How reasoning is returned (llama-server's `--reasoning-format`): `none` leaves the analysis text in
    /// `content`, `auto` and `deepseek` split it into `reasoning_content`, and `deepseek-legacy` also mirrors
    /// it into streaming `content` deltas. Per request, the body's `reasoning_format` wins.
    #[arg(
        long = "reasoning-format",
        value_name = "FORMAT",
        default_value = "auto"
    )]
    pub reasoning_format: String,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 8201)]
    pub port: u16,
    #[arg(short = 't', long)]
    pub threads: Option<usize>,
    /// Context length (clamped to the model's training context).
    #[arg(short = 'c', long, default_value_t = 8192)]
    pub ctx: usize,
    /// KV cache precision: f16, f32 or i8. The default is f16 (or whatever `PEREGRINE_KV`
    /// says); `f32` keeps the keys and values in full precision, which is what a
    /// like-for-like comparison against another engine needs.
    #[arg(long)]
    pub kv: Option<String>,
    /// A DSpark drafter directory (config.json + model.safetensors) trained for this model; greedy requests only.
    #[arg(long)]
    pub dspark: Option<std::path::PathBuf>,
    /// Prompt-lookup speculation on greedy requests (off by default: it applies only to greedy requests).
    #[arg(long, default_value_t = false)]
    pub speculate: bool,
    /// Memory (MB) for snapshots of finished conversations' caches, so a prompt that shares a long beginning with an
    /// earlier one (not only the last) is not prefilled again; 0 = off. Not used with a DSpark drafter.
    #[arg(long = "prefix-cache-mb", default_value_t = 768)]
    pub prefix_cache_mb: usize,
    /// Users served at once, each in one row of the same weight pass. Above the machine's roofline ridge
    /// (about seven rows here) another row adds no throughput and every user waits longer for its own tokens.
    #[arg(long, default_value_t = peregrine_core::serve::DEFAULT_MAX_SLOTS)]
    pub slots: usize,
    /// Run the projections a LittleBit sidecar (`<model>.gguf.slb`, built by `lb_convert`) covers from it. Off by
    /// default: post-training sidecars measured here produce wrong answers, so a stale one
    /// beside a model must not switch the weights silently. A missing or corrupt sidecar refuses to start.
    #[arg(long, default_value_t = false)]
    pub littlebit: bool,
    /// A tuning profile file to apply. Defaults to the default profile for this model and machine.
    #[arg(long)]
    pub profile: Option<std::path::PathBuf>,
    /// Ignore tuning profiles and use defaults or environment variables.
    #[arg(long, default_value_t = false)]
    pub no_profile: bool,
}

struct App {
    /// `/metrics` lines naming every engine option (`peregrine_option{name,value} 1`).
    option_metrics: String,
    /// The engine thread's queue: every request is served there, several at a time.
    jobs: std::sync::mpsc::Sender<EngineMsg>,
    engine: Arc<EngineShared>,
    model: Arc<Model>,
    template: Template,
    default_system: Option<&'static str>,
    /// Token ids that end a turn under this model's template.
    stop: Vec<u32>,
    max_ctx: usize,
    max_slots: usize,
    alias: String,
    think_open: Option<u32>,
    think_close: Option<u32>,
    /// The server's default reasoning format, overridden by a request's `reasoning_format`.
    reasoning_format: ReasoningFormat,
    /// Requests being served or waiting (llama-server's `requests_processing`).
    busy: AtomicUsize,
    requests: AtomicU64,
    tokens: AtomicU64,
    started: Instant,
    simd: &'static str,
    /// "dense" or "littlebit", reported in every response's timings.
    weights: &'static str,
}

/// Run the server until it is stopped. Logging must already be set up (see [`init_logging`]).
pub async fn run(a: Args) {
    let (profile, prof_msg) =
        peregrine_core::profile::resolve(&a.model, a.profile.as_deref(), a.no_profile);
    println!("{prof_msg}");
    let opts = match engine_options(&a.opts, profile.as_ref()) {
        Ok(o) => {
            if profile.is_some() {
                let p_path = a
                    .profile
                    .as_deref()
                    .map(|p| p.to_path_buf())
                    .or_else(|| peregrine_core::profile::default_path(&a.model).ok());
                if let Some(p) = p_path {
                    let dt = o
                        .decode_threads
                        .unwrap_or_else(|| peregrine_core::pool::physical_cores().unwrap_or(1));
                    eprintln!(
                        "peregrine: using tuning profile {} (decode_threads={}, spin_us={})",
                        p.display(),
                        dt,
                        o.spin_us
                    );
                }
            }
            o
        }
        Err(e) => {
            eprintln!("peregrine-server: {e}");
            std::process::exit(2);
        }
    };
    let changed = opts.non_default();
    tracing::info!(
        "engine options: {}",
        if changed.is_empty() {
            "all defaults".to_string()
        } else {
            changed
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        }
    );
    tracing::debug!("engine options in full:\n{}", opts.describe());
    let option_metrics: String = format!(
        "# HELP peregrine_calibrated_decode_threads Thread pool width for single-row decode\n# TYPE peregrine_calibrated_decode_threads gauge\nperegrine_calibrated_decode_threads {}\n\n# HELP peregrine_calibrated_spin_us Worker spin window in microseconds\n# TYPE peregrine_calibrated_spin_us gauge\nperegrine_calibrated_spin_us {}\n\n# HELP peregrine_calibration_duration_seconds Time spent during startup calibration\n# TYPE peregrine_calibration_duration_seconds gauge\nperegrine_calibration_duration_seconds {:.3}\n\n# HELP peregrine_calibration_source Source of active tuning parameters (0=default, 1=startup, 2=profile, 3=env)\n# TYPE peregrine_calibration_source gauge\nperegrine_calibration_source {}\n\n{}",
        opts.decode_threads.unwrap_or_else(|| peregrine_core::pool::physical_cores().unwrap_or(1)),
        opts.spin_us,
        opts.calibration_ms() / 1000.0,
        opts.tuning_source.code(),
        opts.pairs()
            .iter()
            .map(|(k, v)| format!("peregrine_option{{name=\"{k}\",value=\"{v}\"}} 1\n"))
            .collect::<String>()
    );
    peregrine_core::options::init(opts)
        .expect("engine options are set once, before the model loads");
    let threads = a
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let t0 = Instant::now();
    // The chat template: the one the caller named, else the one the file declares.
    let chosen = match a.chat_template.as_deref() {
        Some(name) => match Template::from_name(name) {
            Some(t) => Some(t),
            None => {
                eprintln!(
                    "unknown chat template {name:?} — Peregrine renders {}",
                    peregrine_core::template::TEMPLATES.join(", ")
                );
                std::process::exit(2);
            }
        },
        None => None,
    };
    let gguf = match peregrine_core::gguf::Gguf::open(&a.model).and_then(|g| {
        peregrine_models::check_compatible_with(&g, chosen)?;
        Ok(g)
    }) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("cannot open {}: {e}", a.model.display());
            std::process::exit(2);
        }
    };
    let template = match chosen {
        Some(t) => t,
        None => match Template::detect(&gguf) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("cannot serve {}: {e}", a.model.display());
                std::process::exit(2);
            }
        },
    };
    let model = match peregrine_models::keep::open_with_template(&a.model, a.littlebit, chosen) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot open {}: {e}", a.model.display());
            std::process::exit(2);
        }
    };
    let kv_prec = match a.kv.as_deref() {
        Some(s) => match peregrine_core::kv::KvPrec::parse(s) {
            Some(p) => p,
            None => {
                eprintln!("unknown KV cache precision {s:?} — f16, f32 or i8");
                std::process::exit(2);
            }
        },
        // No `--kv`: the environment decides (`PEREGRINE_KV`), else f16.
        None => peregrine_core::options::get().kv_prec,
    };
    let max_ctx = a.ctx.clamp(32, model.cfg.ctx_train);
    let reasoning_format = match ReasoningFormat::parse(&a.reasoning_format) {
        Some(f) => f,
        None => {
            eprintln!(
                "unknown reasoning format {:?} — none, auto, deepseek or deepseek-legacy",
                a.reasoning_format
            );
            std::process::exit(2);
        }
    };
    let engine = Engine::new(
        model.clone(),
        threads,
        max_ctx,
        a.slots,
        kv_prec,
        a.prefix_cache_mb << 20,
    );
    let cache_mb = engine.cache_bytes_per_slot() >> 20;
    let (jobs, rx) = std::sync::mpsc::channel::<EngineMsg>();
    let shared = Arc::new(EngineShared::default());
    {
        let shared = shared.clone();
        std::thread::Builder::new()
            .name("peregrine-engine".into())
            .spawn(move || engine_thread(engine, rx, shared))
            .expect("spawn the engine thread");
    }
    let app = Arc::new(App {
        option_metrics,
        think_open: model.tk.id_of("<think>"),
        think_close: model.tk.id_of("</think>"),
        reasoning_format,
        stop: template.stop_ids(&model.tk),
        default_system: template.default_system(model.name()),
        template,
        max_ctx,
        max_slots: a.slots,
        alias: a.alias.clone(),
        busy: AtomicUsize::new(0),
        requests: AtomicU64::new(0),
        tokens: AtomicU64::new(0),
        started: Instant::now(),
        simd: peregrine_core::simd_name(),
        weights: if model.lb_summary().is_some() {
            "littlebit"
        } else {
            "dense"
        },
        jobs,
        engine: shared,
        model,
    });
    let weights = match app.model.lb_summary() {
        Some((n, bytes)) => format!("LittleBit ({n} projections, {} MB of factors)", bytes >> 20),
        None => "dense".to_string(),
    };
    tracing::info!(
        "peregrine-core: {} loaded in {:.1} s, {} threads, {}, ctx {}, {} slots ({} MB of cache each), {} weights, {} chat template, think tokens {:?}/{:?}",
        a.model.display(),
        t0.elapsed().as_secs_f64(),
        threads,
        app.simd,
        max_ctx,
        a.slots,
        cache_mb,
        weights,
        template.name(),
        app.think_open,
        app.think_close
    );
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat))
        .with_state(app);
    let addr = format!("{}:{}", a.host, a.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    tracing::info!("listening on http://{addr}");
    axum::serve(listener, router).await.expect("serve");
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(s): State<Arc<App>>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": s.alias, "object": "model", "owned_by": "peregrine-core", "weights": s.weights}]}),
    )
}

async fn metrics(State(s): State<Arc<App>>) -> String {
    format!(
        "# Peregrine server\nllamacpp:requests_processing {}\nllamacpp:requests_deferred {}\nperegrine_slots_total {}\nperegrine_requests_total {}\nperegrine_tokens_predicted_total {}\nperegrine_decode_steps_total {}\nperegrine_decode_rows_total {}\nperegrine_mtp_drafts_total {}\nperegrine_mtp_accepted_total {}\nperegrine_uptime_seconds {}\n{}",
        s.engine.active.load(Ordering::SeqCst),
        s.engine.waiting.load(Ordering::SeqCst),
        s.max_slots,
        s.requests.load(Ordering::SeqCst),
        s.tokens.load(Ordering::SeqCst),
        s.engine.steps.load(Ordering::SeqCst),
        s.engine.rows.load(Ordering::SeqCst),
        s.engine.drafts.load(Ordering::SeqCst),
        s.engine.hits.load(Ordering::SeqCst),
        s.started.elapsed().as_secs(),
        s.option_metrics
    )
}

/// The engine options: the environment's `PEREGRINE_*` variables, overridden by `--opt NAME=VALUE`,
/// falling back to tuning profile, then defaults.
fn engine_options(
    cli: &[String],
    profile: Option<&peregrine_core::profile::Profile>,
) -> Result<peregrine_core::options::EngineOptions, String> {
    let mut over = HashMap::new();
    for o in cli {
        let (k, v) = o
            .split_once('=')
            .ok_or_else(|| format!("--opt {o:?}: expected NAME=VALUE"))?;
        let k = k
            .trim()
            .trim_start_matches("PEREGRINE_")
            .to_ascii_uppercase();
        if !peregrine_core::options::NAMES.contains(&k.as_str()) {
            return Err(format!(
                "--opt {k}: unknown option; known: {}",
                peregrine_core::options::NAMES.join(", ")
            ));
        }
        over.insert(format!("PEREGRINE_{k}"), v.to_string());
    }
    Ok(peregrine_core::options::EngineOptions::from_chained(
        |k| over.get(k).cloned().or_else(|| std::env::var(k).ok()),
        profile,
    ))
}

fn bad(msg: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(
            json!({"error": {"message": msg.into(), "type": "invalid_request_error", "code": 400}}),
        ),
    )
        .into_response()
}

/// What one request asks for, read from the OpenAI-style body.
#[derive(Clone)]
struct Plan {
    system: Option<String>,
    pairs: Vec<(String, String)>,
    user: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    /// Repetition penalties; all off unless the request asks for them.
    penalties: Penalties,
    seed: u64,
    thinking: bool,
    /// How this reply's reasoning is presented; the request's field, else the server's default.
    reasoning_format: ReasoningFormat,
    budget: Option<usize>,
    logprobs: bool,
    stream: bool,
    /// Choices asked for (`n`). Each is its own generation in its own slot, so the engine serves them in one
    /// weight pass; choice `i` samples with seed `seed + i`.
    n: usize,
    /// Non-standard option: whether to ignore end-of-generation tokens and decode until `max_tokens`.
    ignore_eos: bool,
}

/// Most choices one request may ask for.
const MAX_CHOICES: u64 = 16;

/// Sampling controls we do not implement, with the value that means "off" or "unchanged".
///
/// A request that sends one of these at any other value is refused by name: silently
/// ignoring it would sample differently from what the caller asked for, which is worse
/// than a clear 400. At their no-op values they have no effect, so they are accepted for
/// client compatibility.
const UNSUPPORTED_SAMPLING: &[(&str, f64)] = &[
    ("typical_p", 1.0),
    ("tfs_z", 1.0),
    ("mirostat", 0.0),
    ("mirostat_tau", 5.0),
    ("mirostat_eta", 0.1),
    ("dry_multiplier", 0.0),
    ("dry_base", 1.75),
    ("dry_allowed_length", 2.0),
    ("dry_penalty_last_n", 64.0),
    ("xtc_probability", 0.0),
    ("xtc_threshold", 0.1),
    ("top_n_sigma", -1.0),
];

/// Refuse sampling fields the engine cannot honour, but only when they would do something.
fn reject_unsupported_sampling(b: &Value) -> Result<(), String> {
    for &(field, no_op) in UNSUPPORTED_SAMPLING {
        if let Some(v) = b.get(field).filter(|v| !v.is_null()) {
            if v.as_f64() != Some(no_op) {
                return Err(format!(
                    "{field} is not supported by this backend (only its no-op value {no_op} is accepted)"
                ));
            }
        }
    }
    if let Some(v) = b.get("logit_bias").filter(|v| !v.is_null()) {
        // An empty object is how clients spell "no biases"; any entry is a real request.
        if !v.as_object().is_some_and(|o| o.is_empty()) {
            return Err(
                "logit_bias is not supported by this backend (only an empty object is accepted)"
                    .into(),
            );
        }
    }
    Ok(())
}

/// A numeric sampling field, defaulting when absent, with a typed error naming the field.
fn number(b: &Value, field: &str, default: f64) -> Result<f64, String> {
    match b.get(field) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| format!("{field} must be a finite number")),
    }
}

/// Read the four llama.cpp-compatible repetition-penalty fields and validate their ranges.
fn sampling_penalties(b: &Value) -> Result<Penalties, String> {
    let presence = number(b, "presence_penalty", 0.0)?;
    let frequency = number(b, "frequency_penalty", 0.0)?;
    let repeat = number(b, "repeat_penalty", 1.0)?;
    if !(-2.0..=2.0).contains(&presence) {
        return Err("presence_penalty must be between -2 and 2".into());
    }
    if !(-2.0..=2.0).contains(&frequency) {
        return Err("frequency_penalty must be between -2 and 2".into());
    }
    // 0 divides by zero; a negative value would reward repetition.
    if repeat <= 0.0 {
        return Err("repeat_penalty must be greater than 0".into());
    }
    let last_n = match b.get("repeat_last_n") {
        None | Some(Value::Null) => 64,
        Some(v) => v
            .as_i64()
            .ok_or_else(|| "repeat_last_n must be an integer".to_string())?,
    };
    if last_n < -1 || last_n > i32::MAX as i64 {
        return Err(
            "repeat_last_n must be -1 (the whole sequence), 0 (off) or a positive count".into(),
        );
    }
    Ok(Penalties {
        presence: presence as f32,
        frequency: frequency as f32,
        repeat: repeat as f32,
        last_n: last_n as i32,
    })
}

fn plan(b: &Value) -> Result<Plan, String> {
    reject_unsupported_sampling(b)?;
    let n = b["n"].as_u64().unwrap_or(1).max(1);
    if n > MAX_CHOICES {
        return Err(format!("n > {MAX_CHOICES} is not supported"));
    }
    if n > 1 && b["stream"].as_bool().unwrap_or(false) {
        return Err("n > 1 is not supported with stream".into());
    }
    if b.get("grammar").is_some_and(|g| !g.is_null())
        || b.get("json_schema").is_some_and(|g| !g.is_null())
    {
        return Err("grammar-constrained sampling is not supported by this backend".into());
    }
    let Some(msgs) = b["messages"].as_array() else {
        return Err("messages must be an array".into());
    };
    let text = |m: &Value| -> String {
        match &m["content"] {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        }
    };
    let mut system: Vec<String> = vec![];
    let mut pairs: Vec<(String, String)> = vec![];
    let mut pending_user: Option<String> = None;
    for m in msgs {
        match m["role"].as_str().unwrap_or("user") {
            "system" => system.push(text(m)),
            "user" => {
                if let Some(u) = pending_user.replace(text(m)) {
                    // Two user messages in a row: the earlier one got no answer.
                    pairs.push((u, String::new()));
                }
            }
            "assistant" => {
                let u = pending_user.take().unwrap_or_default();
                pairs.push((u, text(m)));
            }
            _ => {}
        }
    }
    let Some(user) = pending_user else {
        return Err("the last message must be from the user".into());
    };
    let kw = &b["chat_template_kwargs"];
    // llama.cpp v0.9.0 maps `reasoning_effort: "none"` to thinking off and drops the
    // effort from the template kwargs (`tools/server/server-common.cpp:1497-1505`).
    let effort_none = b["reasoning_effort"].as_str() == Some("none");
    let reasoning_format = match b["reasoning_format"].as_str() {
        Some(name) => ReasoningFormat::parse(name).ok_or_else(|| {
            format!("unknown reasoning_format {name:?} — none, auto, deepseek or deepseek-legacy")
        })?,
        None => ReasoningFormat::Auto,
    };
    Ok(Plan {
        system: (!system.is_empty()).then(|| system.join("\n\n")),
        pairs,
        user,
        max_tokens: b["max_tokens"].as_u64().unwrap_or(256).clamp(1, 32_768) as usize,
        temperature: b["temperature"].as_f64().unwrap_or(0.8) as f32,
        top_p: b["top_p"].as_f64().unwrap_or(0.95) as f32,
        // llama-server's defaults, so the two engines sample alike unless a request says otherwise.
        top_k: b["top_k"].as_u64().unwrap_or(40) as usize,
        min_p: b["min_p"].as_f64().unwrap_or(0.05) as f32,
        penalties: sampling_penalties(b)?,
        seed: b["seed"].as_u64().unwrap_or(42),
        n: n as usize,
        thinking: if effort_none {
            false
        } else {
            kw["enable_thinking"].as_bool().unwrap_or(true)
        },
        reasoning_format,
        budget: b["thinking_budget_tokens"]
            .as_u64()
            .or_else(|| b["reasoning_budget_tokens"].as_u64())
            .map(|v| v as usize),
        logprobs: b["logprobs"].as_bool().unwrap_or(false),
        stream: b["stream"].as_bool().unwrap_or(false),
        ignore_eos: b["ignore_eos"].as_bool().unwrap_or(false),
    })
}

enum Msg {
    /// A piece of text: (text, is_reasoning, logprob).
    Delta(String, bool, f32),
    Done {
        finish: &'static str,
        usage: Value,
        timings: Value,
    },
    Err(String),
}

/// What one request produced, for its usage and timings block.
#[derive(Default)]
struct Run {
    in_think: bool,
    /// The model wrote the opening think tag itself (it is not part of `reasoning`).
    saw_open: bool,
    reasoning: String,
    content_tokens: u64,
    reasoning_tokens: u64,
    budget_hit: bool,
    prompt_n: u64,
    cache_n: u64,
    prompt_ms: f64,
    predicted_ms: f64,
    /// Decode seconds attributed to this request by the engine (its share of the shared steps).
    decode_ms: f64,
}

/// What the assistant has written when the reasoning budget is hit: the text that was generated, token for token, with
/// the think span closed. The model wrote `<think>` and then a reasoning that starts with its own newline; a second
/// newline here made the first reasoning token a different one, so the cache stopped matching at that position and every
/// reasoning token was prefilled again (measured: 108 tokens and 2.2 s at a budget of 200 on the 1.7B model, 5 and 0.2 s
/// after). A model that never wrote the opening tag gets it, with its newline.
fn continuation_prefix(saw_open: bool, reasoning: &str, closing: &str) -> String {
    if saw_open {
        format!("<think>{reasoning}{closing}")
    } else {
        format!("<think>\n{reasoning}{closing}")
    }
}

/// A piece of a Harmony reply: count it under its channel and send it as the matching delta.
/// Control tokens never reach here; `deepseek-legacy` mirrors a reasoning piece into a
/// streaming `content` delta as llama.cpp does (`tools/server/server-task.cpp:78`).
fn send_harmony_delta(
    run: &mut Run,
    p: &Plan,
    tx: &tokio::sync::mpsc::UnboundedSender<Msg>,
    logprob: f32,
    channel: Channel,
    text: String,
) -> bool {
    let reasoning = channel == Channel::Reasoning && p.reasoning_format.extracts();
    if reasoning {
        run.reasoning_tokens += 1;
        run.reasoning.push_str(&text);
        if tx.send(Msg::Delta(text.clone(), true, logprob)).is_err() {
            return false;
        }
        if p.stream && p.reasoning_format.reasoning_in_content() {
            let _ = tx.send(Msg::Delta(text, false, logprob));
        }
    } else {
        run.content_tokens += 1;
        if tx.send(Msg::Delta(text, false, logprob)).is_err() {
            return false;
        }
    }
    true
}

/// A request handed to the engine thread, and where its tokens go.
struct Job {
    prompt: Vec<u32>,
    cfg: SlotCfg,
    out: tokio::sync::mpsc::UnboundedSender<Emission>,
}

enum EngineMsg {
    Run(Box<Job>),
    Cancel(SlotId),
}

/// The engine thread: it owns the model and serves every request, several at a time in one pass over the weights.
///
/// One thread rather than a lock each request takes in turn, because the batching only exists if the steps of
/// different requests meet in the same place. It sleeps when there is nothing to serve.
fn engine_thread(
    mut engine: Engine,
    rx: std::sync::mpsc::Receiver<EngineMsg>,
    shared: Arc<EngineShared>,
) {
    let mut routes: HashMap<SlotId, tokio::sync::mpsc::UnboundedSender<Emission>> = HashMap::new();
    let mut waiting: VecDeque<Job> = VecDeque::new();
    let take = |msg,
                waiting: &mut VecDeque<Job>,
                engine: &mut Engine,
                routes: &mut HashMap<_, _>| match msg {
        EngineMsg::Run(j) => waiting.push_back(*j),
        EngineMsg::Cancel(id) => {
            engine.cancel(id);
            routes.remove(&id);
        }
    };
    loop {
        loop {
            match rx.try_recv() {
                Ok(m) => take(m, &mut waiting, &mut engine, &mut routes),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
            }
        }
        while engine.has_room() {
            let Some(j) = waiting.pop_front() else { break };
            // A caller that has gone is not worth a prefill.
            if j.out.is_closed() {
                continue;
            }
            match engine.admit(j.prompt, j.cfg) {
                Ok(id) => {
                    routes.insert(id, j.out);
                }
                Err((_, why)) => {
                    let _ = j.out.send(Emission {
                        slot: 0,
                        token: 0,
                        text: String::new(),
                        logprob: 0.0,
                        finish: Some(Finish::Error(why)),
                        cost: None,
                    });
                }
            }
        }
        shared.active.store(engine.active(), Ordering::SeqCst);
        shared.waiting.store(waiting.len(), Ordering::SeqCst);
        if engine.active() == 0 {
            // Nothing to serve: block until something arrives rather than spin on four cores.
            match rx.recv() {
                Ok(m) => take(m, &mut waiting, &mut engine, &mut routes),
                Err(_) => return,
            }
            continue;
        }
        for em in engine.step() {
            let done = em.finish.is_some();
            let slot = em.slot;
            if let Some(tx) = routes.get(&slot) {
                if tx.send(em).is_err() && !done {
                    // The caller has gone: stop generating for it.
                    engine.cancel(slot);
                    routes.remove(&slot);
                    continue;
                }
            }
            if done {
                routes.remove(&slot);
            }
        }
        shared.steps.store(engine.decode_steps, Ordering::SeqCst);
        shared.rows.store(engine.decode_rows, Ordering::SeqCst);
        shared.drafts.store(engine.spec_drafts, Ordering::SeqCst);
        shared.hits.store(engine.spec_hits, Ordering::SeqCst);
        let (entries, bytes, hits, saved) = engine.prefix_stats();
        shared.prefix.store(
            ((entries as u64) << 48)
                | ((bytes as u64 >> 20) << 32)
                | ((hits & 0xffff) << 16)
                | (saved & 0xffff),
            Ordering::SeqCst,
        );
        *shared.prefix_full.lock().unwrap_or_else(|e| e.into_inner()) =
            (entries, bytes, hits, saved);
        *shared.phases.lock().unwrap_or_else(|e| e.into_inner()) = engine.phases_ms();
    }
}

/// Counters the engine thread publishes for `/metrics` and the timings block.
#[derive(Default)]
struct EngineShared {
    active: AtomicUsize,
    waiting: AtomicUsize,
    steps: AtomicU64,
    rows: AtomicU64,
    /// MTP drafts verified and kept (single-user speculation).
    drafts: AtomicU64,
    hits: AtomicU64,
    prefix: AtomicU64,
    prefix_full: Mutex<(usize, usize, u64, u64)>,
    phases: Mutex<[f64; 6]>,
}

impl EngineShared {
    fn phases(&self) -> [f64; 6] {
        *self.phases.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Serve one request: build its prompt, hand it to the engine, and turn the tokens it sends back into the
/// deltas the caller is streamed. The reasoning budget is enforced here, by replanning the turn with what the
/// model has written so far — the engine's prefix store then restores that reasoning instead of recomputing it.
async fn generate(app: &App, p: &Plan, tx: tokio::sync::mpsc::UnboundedSender<Msg>) {
    let t0 = Instant::now();
    let mut run = Run::default();
    let mut budget_left = p.max_tokens;
    let mut prefix: Option<String> = None;
    let mut finish = "stop";
    // gpt-oss's reply is split by control tokens; every other template's reasoning is Qwen3's
    // `<think>` span, below.
    let mut harmony = (app.template == Template::Harmony).then(Harmony::new);
    // At most two passes: the turn as asked, then the same turn continued with the reasoning closed.
    for pass in 0..2 {
        let thinking = p.thinking && pass == 0;
        let prompt_text = peregrine_core::serve::chat_prompt(
            app.template,
            p.system.as_deref().or(app.default_system),
            &p.pairs,
            &p.user,
            thinking || prefix.is_some(),
            prefix.as_deref(),
        );
        let prompt = app.model.tk.encode_prompt(&prompt_text);
        if prompt.len() + 1 >= app.max_ctx {
            let _ = tx.send(Msg::Err(format!(
                "the prompt is {} tokens; this model is served with {} positions",
                prompt.len(),
                app.max_ctx
            )));
            return;
        }
        let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel::<Emission>();
        let cfg = SlotCfg {
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            min_p: p.min_p,
            penalties: p.penalties,
            seed: p.seed,
            max_tokens: budget_left.max(1),
            stop: if p.ignore_eos {
                vec![]
            } else {
                app.stop.clone()
            },
            logprobs: p.logprobs,
            // The server serves independent requests: every one of them draws with the
            // engine's own key (no conversation turn in it) and with the engine's own
            // KV cache (the rotation on).
            turn: 0,
            kv_rotation: true,
            // The server asks for no drafter: the slot decodes plainly, and the
            // engine's own MTP speculation is the engine's business.
            drafter: Default::default(),
        };
        if app
            .jobs
            .send(EngineMsg::Run(Box::new(Job {
                prompt,
                cfg,
                out: etx,
            })))
            .is_err()
        {
            let _ = tx.send(Msg::Err("the engine is not running".into()));
            return;
        }
        let mut replan = false;
        let mut slot = 0;
        'emissions: while let Some(em) = erx.recv().await {
            slot = em.slot;
            if let Some(c) = &em.cost {
                run.prompt_n += c.prefilled as u64;
                run.cache_n += (c.prompt_tokens - c.prefilled) as u64;
                run.prompt_ms += c.prefill_s * 1e3;
                run.decode_ms += c.decode_s * 1e3;
            }
            if let Some(f) = &em.finish {
                match f {
                    Finish::Error(e) => {
                        let _ = tx.send(Msg::Err(e.clone()));
                        return;
                    }
                    Finish::Stop => finish = "stop",
                    Finish::Length | Finish::Context => finish = "length",
                }
                if em.text.is_empty() {
                    break;
                }
            }
            if let Some(h) = harmony.as_mut() {
                if !em.text.is_empty() {
                    let mut produced = false;
                    for (channel, text) in h.feed(&em.text) {
                        produced = true;
                        if !send_harmony_delta(&mut run, p, &tx, em.logprob, channel, text) {
                            break 'emissions;
                        }
                    }
                    // A channel header, `<|end|>` or `<|start|>` is a generated token too; it
                    // belongs to neither field but does belong in the token count.
                    if !produced {
                        run.content_tokens += 1;
                    }
                    budget_left = budget_left.saturating_sub(1);
                }
                if em.finish.is_some() {
                    break;
                }
                continue;
            }
            // The think tags mark the span; they are not part of either text.
            if Some(em.token) == app.think_open && pass == 0 {
                run.in_think = true;
                run.saw_open = true;
                continue;
            }
            if Some(em.token) == app.think_close {
                run.in_think = false;
                continue;
            }
            if run.in_think && p.reasoning_format.extracts() {
                run.reasoning_tokens += 1;
                run.reasoning.push_str(&em.text);
                budget_left = budget_left.saturating_sub(1);
                if tx
                    .send(Msg::Delta(em.text.clone(), true, em.logprob))
                    .is_err()
                {
                    break;
                }
                if p.stream && p.reasoning_format.reasoning_in_content() {
                    let _ = tx.send(Msg::Delta(em.text, false, em.logprob));
                }
                if p.budget.is_some_and(|b| run.reasoning_tokens as usize >= b) {
                    run.budget_hit = true;
                    replan = true;
                    break;
                }
            } else {
                // With `reasoning_format: none`, the think span is ordinary content.
                run.content_tokens += 1;
                budget_left = budget_left.saturating_sub(1);
                if tx.send(Msg::Delta(em.text, false, em.logprob)).is_err() {
                    break;
                }
            }
            if em.finish.is_some() {
                break;
            }
        }
        // Flush a reply that ended inside a channel (a `max_tokens` stop, for instance);
        // a half-written control token is dropped rather than leaked into the answer.
        if let Some(h) = harmony.as_mut() {
            for (channel, text) in h.finish() {
                if !send_harmony_delta(&mut run, p, &tx, 0.0, channel, text) {
                    break;
                }
            }
        }
        if !replan {
            break;
        }
        // Close the reasoning and continue the same turn. The cache of the pass just cancelled holds every
        // reasoning token, and the engine's prefix store hands it to the continuation.
        let _ = app.jobs.send(EngineMsg::Cancel(slot));
        let closing = "\n</think>\n\n";
        let _ = tx.send(Msg::Delta(closing.to_string(), true, 0.0));
        prefix = Some(continuation_prefix(run.saw_open, &run.reasoning, closing));
        run.in_think = false;
        if budget_left == 0 {
            finish = "length";
            break;
        }
    }
    let predicted_n = run.content_tokens + run.reasoning_tokens;
    app.tokens.fetch_add(predicted_n, Ordering::SeqCst);
    let (entries, bytes, hits, saved) = *app
        .engine
        .prefix_full
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (steps, rows) = (
        app.engine.steps.load(Ordering::SeqCst),
        app.engine.rows.load(Ordering::SeqCst),
    );
    // The decode time attributed to this request (its share of the steps it was in), not the wall clock: with
    // other users in the same pass the wall clock includes their tokens too.
    if run.predicted_ms == 0.0 {
        run.predicted_ms = run.decode_ms;
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    let _ = tx.send(Msg::Done {
        finish,
        usage: json!({"prompt_tokens": run.prompt_n + run.cache_n, "completion_tokens": predicted_n, "total_tokens": run.prompt_n + run.cache_n + predicted_n}),
        timings: json!({"prompt_n": run.prompt_n, "cache_n": run.cache_n, "prompt_ms": run.prompt_ms, "predicted_n": predicted_n, "predicted_ms": run.predicted_ms,
            "predicted_per_second": if run.predicted_ms > 0.0 { predicted_n as f64 / run.predicted_ms * 1e3 } else { 0.0 },
            "peregrine": {"simd": app.simd, "weights": app.weights, "budget_forced": run.budget_hit, "wall_ms": wall_ms,
                "phases_ms": {"names": ["matmul", "head", "attention", "other", "sampling", "drafting"], "decode": app.engine.phases(), "prefill": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                "concurrency": {"slots": app.max_slots, "active": app.engine.active.load(Ordering::SeqCst), "waiting": app.engine.waiting.load(Ordering::SeqCst),
                    "decode_steps": steps, "decode_rows": rows,
                    "rows_per_step": if steps > 0 { rows as f64 / steps as f64 } else { 0.0 }},
                "prefix_store": {"entries": entries, "mb": bytes >> 20, "restores": hits, "positions_saved": saved}}}),
    });
}

// requests in the building at once (served plus queued): at least 16, and twice the slots, so a full
// set of users plus their n-choice rounds is queued rather than refused.
const MIN_WAITING: usize = 16;

/// Reserve `k` places of `cap` at once, or none.
fn try_reserve_n(counter: &AtomicUsize, k: usize, cap: usize) -> bool {
    let mut cur = counter.load(Ordering::SeqCst);
    loop {
        if cur + k > cap {
            return false;
        }
        match counter.compare_exchange(cur, cur + k, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return true,
            Err(v) => cur = v,
        }
    }
}

struct BusyGuard(Arc<App>);
impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.0.busy.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn chat(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let mut p = match plan(&body) {
        Ok(p) => p,
        Err(e) => return bad(e),
    };
    if body.get("reasoning_format").is_none() {
        p.reasoning_format = app.reasoning_format;
    }
    // One session decodes one request at a time; the others wait. A bounded wait queue keeps a burst of (possibly
    // abandoned) long prompts from piling up blocking threads and CPU-hours of prefill.
    // every choice is a generation of its own, so a request with `n` choices reserves `n` places (and
    // each choice releases its own when it ends): `n` cannot carry a request past the queue limit.
    if !try_reserve_n(&app.busy, p.n, MIN_WAITING.max(2 * app.max_slots)) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": "too many requests are waiting for this model", "type": "overloaded_error", "code": 503}})),
        )
            .into_response();
    }
    let busy = BusyGuard(app.clone());
    app.requests.fetch_add(1, Ordering::SeqCst);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Msg>();
    let app2 = app.clone();
    let p = Arc::new(p);
    let p2 = p.clone();
    tokio::spawn(async move {
        let _busy = busy; // released when the generation ends, also if it panics
        generate(&app2, &p2, tx).await;
    });
    let id = format!("chatcmpl-{}", app.requests.load(Ordering::SeqCst));
    let model = app.alias.clone();
    if p.stream {
        let logprobs = p.logprobs;
        let stream = async_stream(rx, id, model, logprobs);
        return Sse::new(stream).into_response();
    }
    // Not streamed: collect everything. Choices after the first are their own generations, started here so the
    // engine admits them beside the first and runs them in the same weight passes.
    let mut rxs = vec![rx];
    for i in 1..p.n {
        let (txi, rxi) = tokio::sync::mpsc::unbounded_channel::<Msg>();
        let mut pi = (*p).clone();
        pi.seed = p.seed.wrapping_add(i as u64);
        let app3 = app.clone();
        let busy_i = BusyGuard(app.clone()); // this choice's place, reserved above
        tokio::spawn(async move {
            let _busy = busy_i;
            generate(&app3, &pi, txi).await;
        });
        rxs.push(rxi);
    }
    let mut choices = Vec::with_capacity(rxs.len());
    let (mut usage_all, mut timings0) = (json!(null), json!(null));
    let (mut prompt_tokens, mut completion_tokens) = (0u64, 0u64);
    for (index, mut rx) in rxs.into_iter().enumerate() {
        let (mut content, mut reasoning, mut lps) = (String::new(), String::new(), vec![]);
        let mut done = false;
        while let Some(m) = rx.recv().await {
            match m {
                Msg::Delta(t, r, lp) => {
                    if r {
                        reasoning.push_str(&t)
                    } else {
                        content.push_str(&t);
                        lps.push(json!({"token": t, "logprob": lp}));
                    }
                }
                Msg::Done {
                    finish,
                    usage,
                    timings,
                } => {
                    let mut choice = json!({"index": index, "message": {"role": "assistant", "content": content, "reasoning_content": reasoning}, "finish_reason": finish});
                    if p.logprobs {
                        choice["logprobs"] = json!({"content": lps});
                    }
                    choices.push(choice);
                    prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
                    completion_tokens += usage["completion_tokens"].as_u64().unwrap_or(0);
                    if index == 0 {
                        usage_all = usage;
                        timings0 = timings;
                    }
                    done = true;
                    break;
                }
                Msg::Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": {"message": e, "type": "server_error", "code": 500}})),
                    )
                        .into_response()
                }
            }
        }
        if !done {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"message": "the generation ended without a result", "type": "server_error", "code": 500}}))).into_response();
        }
    }
    if p.n > 1 {
        usage_all = json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens});
    }
    Json(json!({"id": id, "object": "chat.completion", "model": model, "choices": choices, "usage": usage_all, "timings": timings0})).into_response()
}

fn async_stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<Msg>,
    id: String,
    model: String,
    logprobs: bool,
) -> impl futures::Stream<Item = Result<Event, std::convert::Infallible>> {
    use tokio_stream::StreamExt;
    tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(move |m| {
        let v = match m {
            Msg::Delta(t, r, lp) => {
                let mut choice = json!({"index": 0, "delta": if r { json!({"reasoning_content": t}) } else { json!({"content": t}) }, "finish_reason": Value::Null});
                if logprobs && !r {
                    choice["logprobs"] = json!({"content": [{"token": t, "logprob": lp}]});
                }
                json!({"id": id, "object": "chat.completion.chunk", "model": model, "choices": [choice]})
            }
            Msg::Done {
                finish,
                usage,
                timings,
            } => {
                json!({"id": id, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage, "timings": timings})
            }
            Msg::Err(e) => {
                json!({"error": {"message": e}})
            }
        };
        Ok(Event::default().data(v.to_string()))
    })
    // `[DONE]` after the last chunk, as llama-server and vLLM send it.
    .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))))
}

/// Log to stderr at `info`, or as `RUST_LOG` says.
pub fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
}

#[cfg(test)]
mod tests {
    use super::{continuation_prefix, plan, Penalties, ReasoningFormat};
    use serde_json::{json, Value};

    fn body(extra: Value) -> Value {
        let mut b = json!({"messages": [{"role": "user", "content": "hi"}]});
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                b[k] = v.clone();
            }
        }
        b
    }

    #[test]
    fn the_chat_template_option_names_a_template() {
        use clap::Parser;
        let a = super::Args::try_parse_from([
            "peregrine-server",
            "-m",
            "model.gguf",
            "--chat-template",
            "llama2",
        ])
        .unwrap();
        assert_eq!(a.chat_template.as_deref(), Some("llama2"));
        assert_eq!(
            peregrine_core::template::Template::from_name(a.chat_template.as_deref().unwrap()),
            Some(peregrine_core::template::Template::Llama2Plain)
        );
        // Without the option, the file's own template is used.
        let a = super::Args::try_parse_from(["peregrine-server", "-m", "model.gguf"]).unwrap();
        assert_eq!(a.chat_template, None);
    }

    #[test]
    fn penalties_are_off_unless_asked_for() {
        let p = plan(&body(json!({}))).unwrap();
        assert_eq!(p.penalties, Penalties::default());
        assert!(p.penalties.is_off());
    }

    #[test]
    fn penalty_fields_are_read_in_openai_and_llama_cpp_spelling() {
        let p = plan(&body(json!({
            "presence_penalty": 1.5,
            "frequency_penalty": 0.25,
            "repeat_penalty": 1.1,
            "repeat_last_n": -1
        })))
        .unwrap();
        assert_eq!(p.penalties.presence, 1.5);
        assert_eq!(p.penalties.frequency, 0.25);
        assert_eq!(p.penalties.repeat, 1.1);
        assert_eq!(p.penalties.last_n, -1);
        assert!(!p.penalties.is_off());
    }

    #[test]
    fn nonsense_penalty_values_are_refused() {
        for bad in [
            json!({"repeat_penalty": -1.0}),
            json!({"repeat_penalty": 0.0}),
            json!({"presence_penalty": 3.0}),
            json!({"frequency_penalty": -2.5}),
            json!({"repeat_last_n": -2}),
            json!({"repeat_last_n": 1.5}),
            json!({"presence_penalty": "high"}),
        ] {
            assert!(plan(&body(bad.clone())).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn unsupported_sampling_fields_are_refused_by_name_when_active() {
        for (field, value) in [
            ("typical_p", json!(0.9)),
            ("tfs_z", json!(0.95)),
            ("mirostat", json!(2)),
            ("dry_multiplier", json!(0.8)),
            ("xtc_probability", json!(0.5)),
            ("top_n_sigma", json!(1.0)),
            ("logit_bias", json!({"15": 5})),
        ] {
            let err = plan(&body(json!({ field: value })))
                .err()
                .unwrap_or_default();
            assert!(err.contains(field), "{field}: {err:?}");
        }
        // At their no-op values they change nothing, so clients that always send them still work.
        for (field, value) in [
            ("typical_p", json!(1.0)),
            ("mirostat", json!(0)),
            ("top_n_sigma", json!(-1.0)),
            ("logit_bias", json!({})),
            ("xtc_probability", json!(null)),
        ] {
            assert!(
                plan(&body(json!({ field: value }))).is_ok(),
                "{field} at its no-op value"
            );
        }
    }

    #[test]
    fn ignore_eos_defaults_to_false_and_can_be_enabled() {
        let p_default = plan(&body(json!({}))).unwrap();
        assert!(!p_default.ignore_eos);

        let p_enabled = plan(&body(json!({"ignore_eos": true}))).unwrap();
        assert!(p_enabled.ignore_eos);

        let p_disabled = plan(&body(json!({"ignore_eos": false}))).unwrap();
        assert!(!p_disabled.ignore_eos);
    }

    #[test]
    fn ordinary_fields_are_not_refused() {
        let p = plan(&body(json!({
            "model": "peregrine",
            "stream": false,
            "max_tokens": 32,
            "stop": ["\n\n"],
            "n": 2,
            "seed": 7,
            "logprobs": true,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "min_p": 0.0
        })));
        assert!(p.is_ok(), "{:?}", p.err());
    }

    #[test]
    fn reasoning_format_defaults_to_auto_and_accepts_llama_cpp_names() {
        assert_eq!(
            plan(&body(json!({}))).unwrap().reasoning_format,
            ReasoningFormat::Auto
        );
        for (name, want) in [
            ("none", ReasoningFormat::None),
            ("auto", ReasoningFormat::Auto),
            ("deepseek", ReasoningFormat::Deepseek),
            ("deepseek-legacy", ReasoningFormat::DeepseekLegacy),
        ] {
            assert_eq!(
                plan(&body(json!({"reasoning_format": name})))
                    .unwrap()
                    .reasoning_format,
                want,
                "{name}"
            );
        }
        let err = plan(&body(json!({"reasoning_format": "thoughts"})))
            .err()
            .unwrap_or_default();
        assert!(err.contains("reasoning_format"), "{err:?}");
    }

    #[test]
    fn reasoning_effort_none_switches_thinking_off() {
        assert!(plan(&body(json!({}))).unwrap().thinking);
        assert!(
            !plan(&body(json!({"reasoning_effort": "none"})))
                .unwrap()
                .thinking
        );
        assert!(
            plan(&body(json!({"reasoning_effort": "high"})))
                .unwrap()
                .thinking
        );
        assert!(
            !plan(&body(
                json!({"chat_template_kwargs": {"enable_thinking": false}})
            ))
            .unwrap()
            .thinking
        );
    }

    #[test]
    fn reasoning_format_decides_which_delta_field_a_channel_goes_to() {
        use super::{send_harmony_delta, Channel, Msg, Run};
        let b = body(json!({}));
        let mut p = plan(&b).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut run = Run::default();
        // `none`: the analysis channel is content.
        p.reasoning_format = ReasoningFormat::None;
        assert!(send_harmony_delta(
            &mut run,
            &p,
            &tx,
            0.0,
            Channel::Reasoning,
            "R".into()
        ));
        match rx.try_recv().unwrap() {
            Msg::Delta(t, false, _) => assert_eq!(t, "R"),
            _ => panic!("reasoning_format none did not send content"),
        }
        // `auto`: it is reasoning_content.
        p.reasoning_format = ReasoningFormat::Auto;
        assert!(send_harmony_delta(
            &mut run,
            &p,
            &tx,
            0.0,
            Channel::Reasoning,
            "R".into()
        ));
        match rx.try_recv().unwrap() {
            Msg::Delta(t, true, _) => assert_eq!(t, "R"),
            _ => panic!("reasoning_format auto did not send reasoning_content"),
        }
        // `deepseek-legacy` with streaming also mirrors it into content.
        p.reasoning_format = ReasoningFormat::DeepseekLegacy;
        p.stream = true;
        assert!(send_harmony_delta(
            &mut run,
            &p,
            &tx,
            0.0,
            Channel::Reasoning,
            "R".into()
        ));
        assert!(matches!(rx.try_recv().unwrap(), Msg::Delta(t, true, _) if t == "R"));
        assert!(matches!(rx.try_recv().unwrap(), Msg::Delta(t, false, _) if t == "R"));
    }

    #[test]
    fn an_earlier_assistant_turn_replays_only_its_content() {
        // gpt-oss's Jinja drops earlier turns' analysis; the request's
        // `reasoning_content` is therefore not part of the replayed prompt.
        let b = json!({"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there", "reasoning_content": "SECRET"},
            {"role": "user", "content": "Again"}
        ]});
        let p = plan(&b).unwrap();
        assert_eq!(p.pairs, vec![("Hello".to_string(), "Hi there".to_string())]);
        let prompt = peregrine_core::serve::chat_prompt(
            peregrine_core::template::Template::Harmony,
            None,
            &p.pairs,
            &p.user,
            true,
            None,
        );
        assert!(!prompt.contains("SECRET"));
        assert!(prompt.contains("<|channel|>final<|message|>Hi there<|end|>"));
    }

    #[test]
    fn the_continuation_repeats_what_was_generated_without_an_extra_newline() {
        let closing = "\n</think>\n\n";
        // The model wrote <think>, then "\nOkay, ...": the prefix is those two pieces, not <think>\n + "\nOkay".
        assert_eq!(
            continuation_prefix(true, "\nOkay, let me think", closing),
            "<think>\nOkay, let me think\n</think>\n\n"
        );
        assert!(!continuation_prefix(true, "\nOkay", closing).contains("<think>\n\n"));
        // A model that never wrote the opening tag gets one.
        assert_eq!(
            continuation_prefix(false, "Okay", closing),
            "<think>\nOkay\n</think>\n\n"
        );
    }
    #[test]
    fn engine_options_and_metrics_include_calibration() {
        let opts =
            super::engine_options(&["CALIBRATE=0".into(), "SPIN_US=200".into()], None).unwrap();
        assert_eq!(opts.spin_us, 200);
        assert!(!opts.calibrate);
    }
}
