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
//! - `stop` (one string or up to four) is matched against the decoded text as it arrives: the reply
//!   ends before the sequence, streaming holds back a suffix that could still become one, and the
//!   match is per request, so it cannot affect another user in the same batched pass.
//!
//! One session per process (one stream at a time; requests queue): run one process per model, as with llama-server.
use axum::{
    extract::{FromRequest, Request, State},
    http::{header, StatusCode},
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use peregrine_core::decision::Decider;
use peregrine_core::embed::Pooling;
use peregrine_core::json::Value as OrderedValue;
use peregrine_core::model::Model;
use peregrine_core::sample::Penalties;
use peregrine_core::serve::{Emission, Engine, Finish, SlotCfg, SlotId, Want};
use peregrine_core::template::Template;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod embed_cache;
mod harmony;
pub mod lifecycle;
mod stop;

use embed_cache::{EmbedCache, Key};

use harmony::{Channel, Harmony, ReasoningFormat};
use stop::StopMatcher;

/// The server's command line (also `peregrine serve`).
#[derive(Parser, Debug)]
#[command(about = "Serve a GGUF model with Peregrine behind the OpenAI chat API")]
pub struct Args {
    /// The GGUF file (llama, qwen2 or qwen3; F32/F16/Q8_0/Q5_0/Q4_K/Q6_K tensors), or
    /// `hf:<namespace>/<repo>/<file>[@<rev>]` to download it into the cache once.
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
    /// Let a replacement process bind the same port while this one drains (`peregrine upgrade`).
    /// Off by default so a typo cannot quietly start a second server on an occupied port.
    #[arg(long, default_value_t = false)]
    pub reuse_port: bool,
    /// Write the lifecycle state file here (set by `peregrine serve --background`).
    #[arg(long, hide = true, value_name = "PATH")]
    pub state_file: Option<std::path::PathBuf>,
    /// The log a background server writes (recorded for `status` and `upgrade`).
    #[arg(long, hide = true, value_name = "PATH")]
    pub log_file: Option<std::path::PathBuf>,
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
    /// The file's pooling type (`<arch>.pooling_type`, read as llama.cpp reads it). `None`
    /// means the file declares none, and `/v1/embeddings` refuses it by name.
    pooling: Option<Pooling>,
    /// The width of one embedding vector (the final hidden state's width).
    embedding_width: usize,
    /// Exact cache of finished embedding vectors, keyed by token ids, pooling and
    /// dimensions (S4.9k). `0` MiB disables it.
    embed_cache: EmbedCache,
    /// The decision front end, when the file is a decision model (`/v1/systemone`).
    decider: Option<Decider>,
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
    /// The kernel tier the run takes, reported beside `simd` so a manifest names
    /// the kernels that ran rather than the build's instruction-set baseline.
    simd_tier: String,
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
    let model = match peregrine_models::keep::open_with_template(&a.model, a.littlebit, chosen) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot open {}: {e}", a.model.display());
            std::process::exit(2);
        }
    };
    // An embedding-only model has no chat template and needs none: it serves
    // `/v1/embeddings` and refuses generation (design §3). The placeholder is never
    // rendered because `chat` answers 400 before it.
    let template = match chosen {
        Some(t) => t,
        None if model.causality() == peregrine_core::arch::Causality::Bidirectional => {
            Template::ChatMl
        }
        None => match Template::detect(&gguf) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("cannot serve {}: {e}", a.model.display());
                std::process::exit(2);
            }
        },
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
    // S4.9b: embedding and decision slots are admitted by their own memory budget and their
    // caches are sized to their prompts, so they cost the prompt, not a context each.
    tracing::info!(
        "peregrine-core: embedding and decision slots: {} rows per pass, {} MB of prompt-sized caches, prompt cache {} KB per position",
        engine.pass_rows(),
        peregrine_core::options::get().work_cache_mb.unwrap_or(512),
        engine.cache_bytes_per_position().div_ceil(1024),
    );
    let (jobs, rx) = std::sync::mpsc::channel::<EngineMsg>();
    let shared = Arc::new(EngineShared::default());
    {
        let shared = shared.clone();
        std::thread::Builder::new()
            .name("peregrine-engine".into())
            .spawn(move || engine_thread(engine, rx, shared))
            .expect("spawn the engine thread");
    }
    let decider = match Decider::from_model(&model) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cannot serve {}: {e}", a.model.display());
            std::process::exit(2);
        }
    };
    let pooling = match model.pooling() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("peregrine-server: {e}");
            std::process::exit(2);
        }
    };
    let embedding_width = model.embedding_width();
    let app = Arc::new(App {
        option_metrics,
        pooling,
        embedding_width,
        embed_cache: EmbedCache::new(peregrine_core::options::get().embed_cache_mb),
        decider,
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
        simd_tier: peregrine_core::simd_tier(),
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
        "peregrine-core: {} loaded in {:.1} s, {} threads, {}, ctx {}, {} chat slots ({} MB of cache each), {} weights, {} chat template, think tokens {:?}/{:?}",
        a.model.display(),
        t0.elapsed().as_secs_f64(),
        threads,
        app.simd_tier,
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
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/systemone", post(systemone))
        .with_state(app);
    let addr = format!("{}:{}", a.host, a.port);
    let listener = match bind(&a.host, a.port, a.reuse_port).await {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("cannot listen on {addr}: {e}");
            std::process::exit(2);
        }
    };
    if let Some(path) = &a.state_file {
        let state = lifecycle::State {
            pid: std::process::id(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            host: a.host.clone(),
            port: a.port,
            model: a.model.display().to_string(),
            log: a.log_file.clone(),
            args: lifecycle::forwarded_args(std::env::args().skip(1)),
            started: lifecycle::now(),
        };
        if let Err(e) = lifecycle::write(path, &state) {
            eprintln!("cannot write {}: {e}", path.display());
            std::process::exit(2);
        }
    }
    tracing::info!("listening on http://{addr}");
    let drain = drain_timeout();
    match serve_with_drain(
        listener,
        router,
        shutdown_signal(a.state_file.clone(), a.log_file.clone()),
        drain,
    )
    .await
    {
        Ok(Drain::Finished) => {}
        Ok(Drain::TimedOut) => eprintln!("{}", drain_timeout_notice(drain)),
        Err(e) => tracing::error!("serve: {e}"),
    }
    if let Some(path) = &a.state_file {
        lifecycle::remove_if_ours(path, std::process::id());
    }
}

/// How long a drain may take before the process exits with requests still open.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// The drain timeout: [`DRAIN_TIMEOUT`], or `PEREGRINE_DRAIN_TIMEOUT_MS` when a test needs a
/// shorter one (`docs/dev/testing.md`). A zero or unparsable value is ignored.
fn drain_timeout() -> Duration {
    std::env::var("PEREGRINE_DRAIN_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&ms| ms > 0)
        .map_or(DRAIN_TIMEOUT, Duration::from_millis)
}

/// The line printed when a drain outlives its timeout.
fn drain_timeout_notice(timeout: Duration) -> String {
    let limit = if timeout.as_secs() > 0 {
        format!("{} s", timeout.as_secs())
    } else {
        format!("{} ms", timeout.as_millis())
    };
    format!("peregrine: shutdown took longer than {limit}; exiting with requests still open")
}

/// How a served [`Router`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drain {
    /// Every in-flight request finished (or there was none).
    Finished,
    /// The drain outlived its timeout, so the process exits with requests still open.
    TimedOut,
}

/// Serve `router` on `listener` until `shutdown` resolves, then drain for at most `drain`.
///
/// The timeout bounds the *shutdown*, not the server's lifetime: it starts when `shutdown`
/// resolves, so a server that nobody has asked to stop keeps serving. This used to be a
/// `timeout` around the whole serving future, which made every server exit 30 s after it started
/// (the bug in v0.1.1).
async fn serve_with_drain<F>(
    listener: tokio::net::TcpListener,
    router: Router,
    shutdown: F,
    drain: Duration,
) -> std::io::Result<Drain>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    // The signal future notifies this task when a drain begins; only then does the clock start.
    let (draining_tx, draining_rx) = tokio::sync::oneshot::channel::<()>();
    let signal = async move {
        shutdown.await;
        let _ = draining_tx.send(());
    };
    let server = axum::serve(listener, router).with_graceful_shutdown(signal);
    let mut server = Box::pin(std::future::IntoFuture::into_future(server));
    tokio::select! {
        served = &mut server => {
            served?;
            Ok(Drain::Finished)
        }
        _ = draining_rx => {
            // A drain must end: a client that holds a stream open cannot keep the process from exiting.
            match tokio::time::timeout(drain, &mut server).await {
                Ok(Ok(())) => Ok(Drain::Finished),
                Ok(Err(e)) => Err(e),
                Err(_) => Ok(Drain::TimedOut),
            }
        }
    }
}

/// Bind the listening socket. With `reuse_port` every process sharing the port must set
/// `SO_REUSEPORT`: that is what lets `peregrine upgrade` start the new server before the old one
/// has stopped, so no connection is refused during a replacement.
async fn bind(host: &str, port: u16, reuse_port: bool) -> std::io::Result<tokio::net::TcpListener> {
    let addr = match host.parse::<std::net::IpAddr>() {
        Ok(ip) => std::net::SocketAddr::new(ip, port),
        Err(_) => tokio::net::lookup_host((host, port))
            .await?
            .next()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::AddrNotAvailable,
                    format!("{host} has no address"),
                )
            })?,
    };
    if !reuse_port {
        return tokio::net::TcpListener::bind(addr).await;
    }
    let socket = if addr.is_ipv4() {
        tokio::net::TcpSocket::new_v4()?
    } else {
        tokio::net::TcpSocket::new_v6()?
    };
    socket.set_reuseaddr(true)?;
    #[cfg(all(
        unix,
        not(target_os = "solaris"),
        not(target_os = "illumos"),
        not(target_os = "cygwin"),
        not(target_os = "nuttx"),
    ))]
    socket.set_reuseport(true)?;
    socket.bind(addr)?;
    socket.listen(1024)
}

/// Resolve on `Ctrl-C` or `SIGTERM`; on `SIGHUP`, re-exec the binary on disk beside this process
/// with `SO_REUSEPORT` and drain once the replacement is listening. A handoff that fails leaves
/// this process serving.
async fn shutdown_signal(
    state_file: Option<std::path::PathBuf>,
    log_file: Option<std::path::PathBuf>,
) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("cannot listen for SIGTERM: {e}");
                let _ = tokio::signal::ctrl_c().await;
                tracing::info!("shutdown requested; draining in-flight requests");
                return;
            }
        };
        let mut hup = signal(SignalKind::hangup()).ok();
        // A `SIGHUP` handoff can wait up to 300 s for the replacement to load a large model, so it
        // runs in a blocking task instead of inside the loop: `SIGTERM` and Ctrl-C stay responsive,
        // and their arrival flips `cancel` so the in-flight replacement is killed, not orphaned.
        let mut in_flight: Option<tokio::task::JoinHandle<Result<lifecycle::Handoff, String>>> =
            None;
        let mut cancel: Option<Arc<AtomicBool>> = None;
        loop {
            let hangup = async {
                match hup.as_mut() {
                    Some(signal) => {
                        signal.recv().await;
                    }
                    None => std::future::pending().await,
                }
            };
            let handoff_done = async {
                match in_flight.as_mut() {
                    Some(task) => Some(task.await),
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    if let Some(flag) = cancel.as_ref() {
                        flag.store(true, Ordering::Relaxed);
                    }
                    tracing::info!("shutdown requested; draining in-flight requests");
                    return;
                }
                _ = term.recv() => {
                    if let Some(flag) = cancel.as_ref() {
                        flag.store(true, Ordering::Relaxed);
                    }
                    tracing::info!("shutdown requested; draining in-flight requests");
                    return;
                }
                result = handoff_done => {
                    in_flight = None;
                    cancel = None;
                    // `handoff_done` only resolves while a task is in flight.
                    match result.expect("handoff_done resolves only with a task") {
                        Ok(Ok(lifecycle::Handoff::Restarted(pid))) => {
                            tracing::info!("SIGHUP: replacement pid {pid} is listening; draining");
                            return;
                        }
                        Ok(Ok(lifecycle::Handoff::Unchanged)) => {
                            tracing::info!("SIGHUP: the binary on disk is unchanged; still serving");
                        }
                        Ok(Err(e)) => {
                            tracing::error!("SIGHUP: handoff failed; still serving: {e}");
                        }
                        Err(e) => {
                            tracing::error!("SIGHUP: handoff task failed; still serving: {e}");
                        }
                    }
                }
                _ = hangup => {
                    if in_flight.is_some() {
                        tracing::info!("SIGHUP: a handoff is already in progress; ignoring");
                        continue;
                    }
                    let flag = Arc::new(AtomicBool::new(false));
                    let state = state_file.clone();
                    let log = log_file.clone();
                    let child_cancel = Arc::clone(&flag);
                    in_flight = Some(tokio::task::spawn_blocking(move || {
                        lifecycle::handoff(state.as_deref(), log.as_deref(), Some(&child_cancel))
                    }));
                    cancel = Some(flag);
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (state_file, log_file);
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown requested; draining in-flight requests");
    }
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(s): State<Arc<App>>) -> Json<Value> {
    // What llama.cpp advertises for the file (`server_model_output_modalities`,
    // `server-common.cpp:207-241`): a native decision model answers decisions, not text, and an
    // `openjev` file has no text generation even though it carries an LM head.
    let output_modalities = if s.decider.is_some() {
        json!(["decisions"])
    } else {
        json!(["text"])
    };
    Json(json!({"object": "list", "data": [{
        "id": s.alias,
        "object": "model",
        "owned_by": "peregrine-core",
        "weights": s.weights,
        "architecture": {"input_modalities": ["text"], "output_modalities": output_modalities},
    }]}))
}

async fn metrics(State(s): State<Arc<App>>) -> String {
    format!(
        "# Peregrine server\nllamacpp:requests_processing {}\nllamacpp:requests_deferred {}\nperegrine_slots_total {}\nperegrine_requests_total {}\nperegrine_tokens_predicted_total {}\nperegrine_decode_steps_total {}\nperegrine_decode_rows_total {}\nperegrine_mtp_drafts_total {}\nperegrine_mtp_accepted_total {}\nperegrine_uptime_seconds {}\nperegrine_embed_cache_entries {}\nperegrine_embed_cache_bytes {}\n{}",
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
        s.embed_cache.len(),
        s.embed_cache.bytes(),
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
    Ok(
        peregrine_core::options::EngineOptions::from_chained_calibrating(
            |k| over.get(k).cloned().or_else(|| std::env::var(k).ok()),
            profile,
        ),
    )
}

fn bad(msg: impl Into<String>) -> Response {
    bad_json(msg.into()).into_response()
}

/// The OpenAI error body and status, before it becomes a response (so a unit test can read it).
fn bad_json(message: String) -> (StatusCode, Json<Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": message, "type": "invalid_request_error", "code": 400}})),
    )
}

/// `Json<Value>`, but a body that is not JSON is answered with the same error object as every
/// other bad request. Axum's own rejection is `text/plain`, which an OpenAI client cannot read as
/// an error.
struct OpenAiJson(Value);

impl<S> FromRequest<S> for OpenAiJson
where
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<Value>::from_request(req, state).await {
            Ok(Json(body)) => Ok(OpenAiJson(body)),
            Err(rejection) => Err(bad(format!(
                "the request body is not valid JSON: {rejection}"
            ))),
        }
    }
}

/// A `/v1/systemone` body, parsed with the order-keeping JSON value.
///
/// The order of the `questions` object is the order of the answers, and the order of each
/// `criteria` object is the order of the options and their probabilities: llama.cpp parses the
/// body into `nlohmann::ordered_json`, which `serde_json` in this workspace does not preserve.
struct OrderedJson(OrderedValue);

impl<S> FromRequest<S> for OrderedJson
where
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let bytes = match axum::body::Bytes::from_request(req, state).await {
            Ok(b) => b,
            Err(rejection) => {
                return Err(bad(format!(
                    "the request body is not valid JSON: {rejection}"
                )))
            }
        };
        let text = match std::str::from_utf8(&bytes) {
            Ok(t) => t,
            Err(_) => return Err(bad("the request body is not valid UTF-8")),
        };
        OrderedValue::parse(text)
            .map(OrderedJson)
            .map_err(|e| bad(format!("the request body is not valid JSON: {e}")))
    }
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
    /// Stop sequences: generation ends when the decoded text contains one, and it is not returned.
    stop: Vec<String>,
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

/// Most stop sequences one request may ask for, as OpenAI documents.
const MAX_STOP_SEQUENCES: usize = 4;

/// Read the OpenAI `stop` field: one string or an array of up to four.
fn stop_sequences(b: &Value) -> Result<Vec<String>, String> {
    let value = match b.get("stop") {
        None | Some(Value::Null) => return Ok(vec![]),
        Some(v) => v,
    };
    let mut sequences = match value {
        Value::String(s) => vec![s.clone()],
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item.as_str() {
                    Some(s) => out.push(s.to_string()),
                    None => return Err("stop must be a string or an array of strings".into()),
                }
            }
            out
        }
        _ => return Err("stop must be a string or an array of strings".into()),
    };
    if sequences.len() > MAX_STOP_SEQUENCES {
        return Err(format!(
            "stop accepts at most {MAX_STOP_SEQUENCES} sequences, got {}",
            sequences.len()
        ));
    }
    sequences.shrink_to_fit();
    Ok(sequences)
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
        stop: stop_sequences(b)?,
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

/// Send one piece of the reply as `content` or `reasoning_content`, mirroring `deepseek-legacy`
/// reasoning into streaming content. Counting is the caller's: one generated token can be split into
/// several pieces by the stop matcher, but it is still one token.
fn push_reply(
    run: &mut Run,
    p: &Plan,
    tx: &tokio::sync::mpsc::UnboundedSender<Msg>,
    text: String,
    reasoning: bool,
    logprob: f32,
) -> bool {
    if text.is_empty() {
        return true;
    }
    if reasoning {
        run.reasoning.push_str(&text);
        if tx.send(Msg::Delta(text.clone(), true, logprob)).is_err() {
            return false;
        }
        if p.stream && p.reasoning_format.reasoning_in_content() {
            let _ = tx.send(Msg::Delta(text, false, logprob));
        }
    } else if tx.send(Msg::Delta(text, false, logprob)).is_err() {
        return false;
    }
    true
}

/// A request handed to the engine thread, and where its tokens go.
struct Job {
    prompt: Vec<u32>,
    cfg: SlotCfg,
    /// Tokens sample and stream; `Pooled` pools one vector (S4.1); `Labels` reads the last prompt
    /// row's logits at `labels` (S4.3).
    want: Want,
    /// The token ids a labels slot reads; empty for every other want.
    labels: Vec<u32>,
    out: tokio::sync::mpsc::UnboundedSender<Emission>,
}

/// A decision request's questions, admitted as one group (S4.9j): their shared beginning is
/// forwarded once and forked into each question's slot.
struct GroupJob {
    prompts: Vec<Vec<u32>>,
    /// Each question's read-out ids, in request order.
    labels: Vec<Vec<u32>>,
    outs: Vec<tokio::sync::mpsc::UnboundedSender<Emission>>,
}

/// What the engine thread's queue holds: one request, or a decision group of several.
enum Queued {
    One(Box<Job>),
    Group(Box<GroupJob>),
}

enum EngineMsg {
    Run(Box<Job>),
    RunGroup(Box<GroupJob>),
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
    let mut waiting: VecDeque<Queued> = VecDeque::new();
    // `PEREGRINE_PROF2=1`: a periodic breakdown of the engine thread's step, on stderr.
    // The engine's own phase counters are caller-thread timers, so this is where they
    // can be read without touching the hot path. `=full` adds the inner-loop buckets.
    let profiled = peregrine_core::prof::enabled();
    let mut prof_last = peregrine_core::prof::snapshot();
    let mut prof_steps = 0u64;
    let mut prof_rows = 0u64;
    let mut prof_wall = Duration::ZERO;
    let mut prof_at = Instant::now();
    let take = |msg,
                waiting: &mut VecDeque<Queued>,
                engine: &mut Engine,
                routes: &mut HashMap<_, _>| match msg {
        EngineMsg::Run(j) => waiting.push_back(Queued::One(j)),
        EngineMsg::RunGroup(g) => waiting.push_back(Queued::Group(g)),
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
        // Admitting is zero-wait: whatever is queued when the engine is free runs together. Chat
        // requests are bounded by `--slots`; embedding texts and decision questions have their own
        // admission (a memory budget, not a slot count), so the queue is scanned rather than taken
        // head-first: a chat request blocked on its slots never holds an embedding batch behind it,
        // and a batch wider than the budget leaves only its tail queued (S4.9b).
        let mut i = 0;
        while i < waiting.len() {
            let q = waiting.remove(i).expect("an index below the length");
            match q {
                Queued::One(j) => {
                    // A caller that has gone is not worth a prefill.
                    if j.out.is_closed() {
                        continue;
                    }
                    let fits = match j.want {
                        Want::Tokens => engine.has_room(),
                        _ => engine.work_room_for(j.prompt.len()),
                    };
                    if !fits {
                        waiting.insert(i, Queued::One(j));
                        i += 1;
                        continue;
                    }
                    let admitted = match j.want {
                        Want::Tokens => engine.admit(j.prompt, j.cfg),
                        Want::Pooled(pooling) => engine.admit_pooled(j.prompt, j.cfg, pooling),
                        Want::Labels => engine.admit_labels(j.prompt, j.cfg, j.labels),
                    };
                    match admitted {
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
                                embedding: None,
                                label_logits: None,
                            });
                        }
                    }
                }
                Queued::Group(mut g) => {
                    // A question's caller that has gone takes its question with it.
                    while let Some(pos) = (0..g.outs.len()).find(|&i| g.outs[i].is_closed()) {
                        g.prompts.remove(pos);
                        g.labels.remove(pos);
                        g.outs.remove(pos);
                    }
                    if g.prompts.is_empty() {
                        continue;
                    }
                    // As many questions as the non-decoding budget admits: each group forwards its
                    // shared beginning once (the rest of the request waits for the next group).
                    let lens: Vec<usize> = g.prompts.iter().map(Vec::len).collect();
                    let k = engine.work_room_prefix(&lens).min(g.prompts.len());
                    if k == 0 {
                        waiting.insert(i, Queued::Group(g));
                        i += 1;
                        continue;
                    }
                    let tail = k < g.prompts.len();
                    let (tail_prompts, tail_labels, tail_outs) = if tail {
                        (
                            g.prompts.split_off(k),
                            g.labels.split_off(k),
                            g.outs.split_off(k),
                        )
                    } else {
                        (Vec::new(), Vec::new(), Vec::new())
                    };
                    let (prompts, labels, outs) = (g.prompts, g.labels, g.outs);
                    if tail {
                        waiting.insert(
                            i,
                            Queued::Group(Box::new(GroupJob {
                                prompts: tail_prompts,
                                labels: tail_labels,
                                outs: tail_outs,
                            })),
                        );
                    }
                    match engine.admit_labels_group(prompts, SlotCfg::default(), labels) {
                        Ok(ids) => {
                            for (id, out) in ids.into_iter().zip(outs) {
                                routes.insert(id, out);
                            }
                        }
                        Err(why) => {
                            for out in outs {
                                let _ = out.send(Emission {
                                    slot: 0,
                                    token: 0,
                                    text: String::new(),
                                    logprob: 0.0,
                                    finish: Some(Finish::Error(why.clone())),
                                    cost: None,
                                    embedding: None,
                                    label_logits: None,
                                });
                            }
                        }
                    }
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
        let t_step = profiled.then(Instant::now);
        let emitted = engine.step();
        if let Some(t) = t_step {
            prof_wall += t.elapsed();
            prof_steps += 1;
            prof_rows += engine.last_step.rows as u64;
        }
        if profiled && prof_steps >= 32 && prof_at.elapsed().as_secs_f64() >= 1.0 {
            let now = peregrine_core::prof::snapshot();
            let names = peregrine_core::prof::names();
            let mut parts = Vec::new();
            for (n, (a, b)) in names.iter().zip(prof_last.iter().zip(now.iter())) {
                let ms = b.saturating_sub(*a) as f64 / 1e6;
                if ms >= 0.05 {
                    parts.push(format!("{n} {ms:.1}"));
                }
            }
            eprintln!(
                "prof: {} steps, {} rows, wall {:.0} ms, per step {:.2} ms, rows/step {:.1}; {}",
                prof_steps,
                prof_rows,
                prof_wall.as_secs_f64() * 1e3,
                prof_wall.as_secs_f64() * 1e3 / prof_steps as f64,
                prof_rows as f64 / prof_steps as f64,
                parts.join(", ")
            );
            prof_last = now;
            prof_steps = 0;
            prof_rows = 0;
            prof_wall = Duration::ZERO;
            prof_at = Instant::now();
        }
        for em in emitted {
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
                want: Want::Tokens,
                labels: Vec::new(),
                out: etx,
            })))
            .is_err()
        {
            let _ = tx.send(Msg::Err("the engine is not running".into()));
            return;
        }
        let mut replan = false;
        let mut slot = 0;
        // This pass's stop sequences. A stop string spans token boundaries, so text that might be its
        // beginning is held back here; a stop found mid-token drops the rest of that token.
        let mut stops = StopMatcher::new(p.stop.clone());
        let mut stopped = false;
        // The field a held-back Harmony piece belongs to, so a flush goes back to the right one.
        let mut harmony_channel: Option<Channel> = None;
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
                        // A stop sequence belongs to one field: flush before the channel changes.
                        if let Some(prev) = harmony_channel {
                            if prev != channel {
                                let held = stops.flush();
                                if !held.is_empty()
                                    && !send_harmony_delta(&mut run, p, &tx, em.logprob, prev, held)
                                {
                                    break 'emissions;
                                }
                            }
                        }
                        harmony_channel = Some(channel);
                        let (text, hit) = stops.feed(&text);
                        if !text.is_empty()
                            && !send_harmony_delta(&mut run, p, &tx, em.logprob, channel, text)
                        {
                            break 'emissions;
                        }
                        if hit {
                            stopped = true;
                            finish = "stop";
                            break;
                        }
                    }
                    // A channel header, `<|end|>` or `<|start|>` is a generated token too; it
                    // belongs to neither field but does belong in the token count.
                    if !produced {
                        run.content_tokens += 1;
                    }
                    budget_left = budget_left.saturating_sub(1);
                    if stopped {
                        break 'emissions;
                    }
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
                // Do not carry a held reasoning suffix into `content`: a stop sequence never spans
                // the two fields.
                if !push_reply(&mut run, p, &tx, stops.flush(), true, em.logprob) {
                    break;
                }
                run.in_think = false;
                continue;
            }
            if run.in_think && p.reasoning_format.extracts() {
                run.reasoning_tokens += 1;
                budget_left = budget_left.saturating_sub(1);
                let (text, hit) = stops.feed(&em.text);
                if !push_reply(&mut run, p, &tx, text, true, em.logprob) {
                    break;
                }
                if hit {
                    stopped = true;
                    finish = "stop";
                    break;
                }
                if p.budget.is_some_and(|b| run.reasoning_tokens as usize >= b) {
                    // The continuation repeats what was generated, so the held suffix is part of it.
                    if !push_reply(&mut run, p, &tx, stops.flush(), true, em.logprob) {
                        break;
                    }
                    run.budget_hit = true;
                    replan = true;
                    break;
                }
            } else {
                // With `reasoning_format: none`, the think span is ordinary content.
                run.content_tokens += 1;
                budget_left = budget_left.saturating_sub(1);
                let (text, hit) = stops.feed(&em.text);
                if !push_reply(&mut run, p, &tx, text, false, em.logprob) {
                    break;
                }
                if hit {
                    stopped = true;
                    finish = "stop";
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
            if !stopped {
                for (channel, text) in h.finish() {
                    if let Some(prev) = harmony_channel {
                        if prev != channel {
                            let held = stops.flush();
                            if !held.is_empty()
                                && !send_harmony_delta(&mut run, p, &tx, 0.0, prev, held)
                            {
                                break;
                            }
                        }
                    }
                    harmony_channel = Some(channel);
                    let (text, hit) = stops.feed(&text);
                    if !text.is_empty() && !send_harmony_delta(&mut run, p, &tx, 0.0, channel, text)
                    {
                        break;
                    }
                    if hit {
                        stopped = true;
                        finish = "stop";
                        break;
                    }
                }
            }
        }
        // Whatever the matcher held back is not a stop sequence after all: the generation ended
        // (EOS or the token budget) before it could become one.
        if !stopped {
            let held = stops.flush();
            let reasoning =
                harmony_channel.is_none() && run.in_think && p.reasoning_format.extracts();
            if let Some(channel) = harmony_channel {
                if !held.is_empty() {
                    let _ = send_harmony_delta(&mut run, p, &tx, 0.0, channel, held);
                }
            } else if !push_reply(&mut run, p, &tx, held, reasoning, 0.0) {
                // The caller has gone; there is nothing left to send it.
            }
        }
        if stopped {
            // The engine would otherwise decode to EOS; nothing it produced next is part of the reply.
            let _ = app.jobs.send(EngineMsg::Cancel(slot));
            break;
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
            "peregrine": {"simd": app.simd, "simd_tier": app.simd_tier, "weights": app.weights, "budget_forced": run.budget_hit, "wall_ms": wall_ms,
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

/// One engine slot a request asks for: all of them are sent before any result is awaited, so they
/// share weight passes, and the answers come back in submission order.
struct SlotJob {
    prompt: Vec<u32>,
    want: Want,
    labels: Vec<u32>,
}

/// Submit one job per slot and collect their emissions in order.
///
/// A request with more slots than the server has busy waits in the engine's own queue (each
/// `EngineMsg::Run` is queued, not refused), so a decision with ten questions answers on a
/// four-slot server and an embedding of thirty texts answers on a four-slot one. The caller has
/// already reserved the request's place in the wait queue.
async fn run_jobs(app: &Arc<App>, jobs: Vec<SlotJob>) -> Result<Vec<Emission>, Box<Response>> {
    let mut rxs = Vec::with_capacity(jobs.len());
    for job in jobs {
        let (etx, erx) = tokio::sync::mpsc::unbounded_channel::<Emission>();
        if app
            .jobs
            .send(EngineMsg::Run(Box::new(Job {
                prompt: job.prompt,
                // A pooled or labels slot samples nothing; the rest of the config is the
                // engine's default.
                cfg: SlotCfg::default(),
                want: job.want,
                labels: job.labels,
                out: etx,
            })))
            .is_err()
        {
            return Err(Box::new(internal("the engine is not running")));
        }
        rxs.push(erx);
    }
    collect(rxs).await
}

/// Submit a decision request's questions as one group (S4.9j) and collect their emissions in
/// request order.
///
/// The engine forwards their shared beginning once, forks its state into one slot per question
/// and runs the tails together; a request with more questions than slots is split into groups by
/// the engine thread, each group forked from one evaluation of the shared beginning.
async fn run_group_jobs(
    app: &Arc<App>,
    prompts: Vec<Vec<u32>>,
    labels: Vec<Vec<u32>>,
) -> Result<Vec<Emission>, Box<Response>> {
    let mut rxs = Vec::with_capacity(prompts.len());
    let mut outs = Vec::with_capacity(prompts.len());
    for _ in 0..prompts.len() {
        let (etx, erx) = tokio::sync::mpsc::unbounded_channel::<Emission>();
        outs.push(etx);
        rxs.push(erx);
    }
    if app
        .jobs
        .send(EngineMsg::RunGroup(Box::new(GroupJob {
            prompts,
            labels,
            outs,
        })))
        .is_err()
    {
        return Err(Box::new(internal("the engine is not running")));
    }
    collect(rxs).await
}

/// The emissions of one submitted job per receiver, in submission order.
async fn collect(
    rxs: Vec<tokio::sync::mpsc::UnboundedReceiver<Emission>>,
) -> Result<Vec<Emission>, Box<Response>> {
    let mut out = Vec::with_capacity(rxs.len());
    for mut rx in rxs {
        let Some(em) = rx.recv().await else {
            return Err(Box::new(internal("the engine ended without a result")));
        };
        if let Some(Finish::Error(e)) = &em.finish {
            return Err(Box::new(bad(e.clone())));
        }
        out.push(em);
    }
    Ok(out)
}

/// A 500 with the OpenAI error shape.
fn internal(message: impl Into<String>) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": {"message": message.into(), "type": "server_error", "code": 500}})),
    )
        .into_response()
}

/// A 501 with llama.cpp's `not_supported_error` type.
fn not_supported(message: impl Into<String>) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({"error": {"message": message.into(), "type": "not_supported_error", "code": 501}})),
    )
        .into_response()
}

/// One input of an embedding request: text to tokenize, or token ids the caller already has.
#[derive(Debug, Clone, PartialEq)]
enum EmbedInput {
    Text(String),
    Tokens(Vec<u32>),
}

/// What `/v1/embeddings` asks for, read from the OpenAI-shaped body.
#[derive(Debug, Clone, PartialEq)]
struct EmbedSpec {
    inputs: Vec<EmbedInput>,
    /// Matryoshka truncation, applied before the L2 normalisation.
    dimensions: Option<usize>,
    /// `encoding_format`: `false` is `float` (the default), `true` is `base64` (little-endian f32).
    base64: bool,
    /// The non-standard opt-in to truncate an over-length text instead of refusing it.
    truncate: bool,
    /// The request's `model` (echoed); any value is accepted, as on the chat route.
    model: String,
}

/// The fields `/v1/embeddings` accepts. Anything else is refused by name: an embedding
/// endpoint that silently ignores an option returns a vector the caller did not ask for.
const EMBED_FIELDS: &[&str] = &[
    "input",
    "model",
    "dimensions",
    "encoding_format",
    "truncate",
    "user",
];

/// Read `/v1/embeddings` (`docs/design/s4-embeddings-decisions.md` §3).
///
/// `input` is a string, an array of strings, an array of token ids, or an array of token-id
/// arrays. `dimensions` truncates the vector (and must fit the model), `encoding_format` is
/// `float` or `base64`, and `truncate` is the opt-in for a text longer than the context.
fn parse_embedding_request(b: &Value, width: usize) -> Result<EmbedSpec, String> {
    if let Some(obj) = b.as_object() {
        for k in obj.keys() {
            if !EMBED_FIELDS.contains(&k.as_str()) {
                return Err(format!("{k} is not supported by /v1/embeddings"));
            }
        }
    } else {
        return Err("the request body must be a JSON object".into());
    }
    let input = b
        .get("input")
        .ok_or_else(|| "input must be provided".to_string())?;
    let ids = |v: &Value| -> Result<Vec<u32>, String> {
        let arr = v
            .as_array()
            .ok_or_else(|| "an input must be a string or an array of token ids".to_string())?;
        let mut out = Vec::with_capacity(arr.len());
        for x in arr {
            let id = x
                .as_u64()
                .filter(|&id| id <= u32::MAX as u64)
                .ok_or_else(|| "token ids must be non-negative integers".to_string())?;
            out.push(id as u32);
        }
        Ok(out)
    };
    let inputs = match input {
        Value::String(text) => vec![EmbedInput::Text(text.clone())],
        Value::Array(items) => {
            if items.is_empty() {
                return Err("input must not be an empty array".into());
            }
            let strings = items.iter().all(|v| v.is_string());
            let numbers = items.iter().all(|v| v.is_u64());
            if strings {
                items
                    .iter()
                    .map(|v| EmbedInput::Text(v.as_str().unwrap_or_default().to_string()))
                    .collect()
            } else if numbers {
                vec![EmbedInput::Tokens(ids(input)?)]
            } else if items.iter().all(|v| v.as_array().is_some()) {
                let mut out = Vec::with_capacity(items.len());
                for (i, v) in items.iter().enumerate() {
                    let t = ids(v)?;
                    if t.is_empty() {
                        return Err(format!("input {i} is an empty token-id array"));
                    }
                    out.push(EmbedInput::Tokens(t));
                }
                out
            } else {
                return Err(
                    "input must be a string, an array of strings, an array of token ids, or an array of token-id arrays"
                        .into(),
                );
            }
        }
        _ => return Err("input must be a string or an array".into()),
    };
    let dimensions = match b.get("dimensions") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let d = v
                .as_u64()
                .filter(|&d| d > 0 && d <= width as u64)
                .ok_or_else(|| format!("dimensions must be an integer from 1 to {width}"))?;
            Some(d as usize)
        }
    };
    let base64 = match b.get("encoding_format") {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) if s == "float" => false,
        Some(Value::String(s)) if s == "base64" => true,
        Some(other) => {
            return Err(format!(
                "encoding_format {other} is not supported: use \"float\" or \"base64\""
            ))
        }
    };
    let truncate = match b.get("truncate") {
        None | Some(Value::Null) => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| "truncate must be a boolean".to_string())?,
    };
    Ok(EmbedSpec {
        inputs,
        dimensions,
        base64,
        truncate,
        model: b
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
    })
}

/// Refuse a text longer than the served context, or truncate it when the request opted in.
///
/// An embedding is one vector for one text, so a silent truncation would be a different
/// vector than the caller asked for; the error names the limit, as the design requires.
fn fit_tokens(
    i: usize,
    tokens: &mut Vec<u32>,
    max_ctx: usize,
    truncate: bool,
) -> Result<(), String> {
    if tokens.is_empty() {
        return Err(format!("input {i} produced no tokens"));
    }
    if tokens.len() > max_ctx {
        if truncate {
            tokens.truncate(max_ctx);
        } else {
            return Err(format!(
                "input {i} is {} tokens; this model is served with {max_ctx} positions (set \"truncate\": true to truncate)",
                tokens.len()
            ));
        }
    }
    Ok(())
}

/// Base64 of the vector's little-endian f32 bytes (OpenAI's `encoding_format: "base64"`).
fn base64_f32(v: &[f32]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// `POST /v1/embeddings` (OpenAI shape; S4.1, causal models).
///
/// Every input of one request becomes a slot, so the texts share weight passes; the results
/// come back in input order.
async fn embeddings(State(app): State<Arc<App>>, OpenAiJson(body): OpenAiJson) -> Response {
    let Some(pooling) = app.pooling else {
        return bad(format!(
            "this model file declares no {}.pooling_type (pooling \"none\"), so /v1/embeddings has no vector to return",
            app.model.cfg.arch_name
        ));
    };
    let spec = match parse_embedding_request(&body, app.embedding_width) {
        Ok(s) => s,
        Err(e) => return bad(e),
    };
    // Tokenize before reserving a place: an over-length text is the caller's error, and it
    // must not hold the engine.
    let mut texts: Vec<Vec<u32>> = Vec::with_capacity(spec.inputs.len());
    for (i, input) in spec.inputs.iter().enumerate() {
        let mut tokens = match input {
            EmbedInput::Text(t) => app.model.tk.encode_special(t),
            EmbedInput::Tokens(t) => t.clone(),
        };
        if let Err(e) = fit_tokens(i, &mut tokens, app.max_ctx, spec.truncate) {
            return bad(e);
        }
        texts.push(tokens);
    }
    // One place for the whole request, not one per text: a batch larger than the slot count is
    // normal (32 texts on a 4-slot server), and the texts beyond the free slots wait in the
    // engine's queue — they are admitted as slots free up, in order, and never refused or
    // dropped here. Counting the request once is what keeps a 32-text batch from a 16-place cap.
    if !try_reserve_n(&app.busy, 1, MIN_WAITING.max(2 * app.max_slots)) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": "too many requests are waiting for this model", "type": "overloaded_error", "code": 503}})),
        )
            .into_response();
    }
    let _busy = BusyGuard(app.clone());
    app.requests.fetch_add(1, Ordering::SeqCst);
    // Usage counts every input, including a text the cache answered or a duplicate
    // that shares one computation (S4.9k).
    let prompt_tokens: u64 = texts.iter().map(|t| t.len() as u64).sum();
    let keys: Vec<Key> = texts
        .iter()
        .map(|t| Key::new(t.clone(), pooling, spec.dimensions))
        .collect();
    let plan = app.embed_cache.plan(keys);
    let mut computed: Vec<Vec<f32>> = Vec::with_capacity(plan.jobs.len());
    if !plan.jobs.is_empty() {
        let jobs = plan
            .jobs
            .iter()
            .map(|k| SlotJob {
                prompt: k.tokens.clone(),
                want: Want::Pooled(pooling),
                labels: Vec::new(),
            })
            .collect();
        let emissions = match run_jobs(&app, jobs).await {
            Ok(e) => e,
            Err(response) => return *response,
        };
        if emissions.len() != plan.jobs.len() {
            return internal("the engine returned the wrong number of embeddings");
        }
        for (key, em) in plan.jobs.iter().zip(emissions) {
            let Some(mut v) = em.embedding else {
                return internal("the engine returned no embedding");
            };
            if let Err(e) = peregrine_core::embed::finish(&mut v, spec.dimensions) {
                return bad(e);
            }
            app.embed_cache.put(key.clone(), v.clone());
            computed.push(v);
        }
    }
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(plan.cached.len());
    for (i, hit) in plan.cached.into_iter().enumerate() {
        match hit {
            Some(v) => vectors.push(v),
            None => {
                let j = plan.job_of[i].expect("a miss has a job");
                vectors.push(computed[j].clone());
            }
        }
    }
    let data: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            json!({
                "object": "embedding",
                "index": i,
                "embedding": if spec.base64 { Value::String(base64_f32(v)) } else { json!(v) },
            })
        })
        .collect();
    let model = if spec.model.is_empty() {
        app.alias.clone()
    } else {
        spec.model.clone()
    };
    Json(json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }))
    .into_response()
}

/// A response whose bytes are the order-keeping value's JSON text: `answers` keeps request order.
fn ordered_json(value: &OrderedValue) -> Response {
    (
        [(header::CONTENT_TYPE, "application/json")],
        value.to_jinja_json(),
    )
        .into_response()
}

/// `POST /v1/systemone` (S4.3c): every question is one labels slot in the engine, so they share
/// weight passes, and the answers come back in request order.
///
/// The request and its validation are [`Decider`]'s; the answer math is
/// [`peregrine_core::decision::format_answer`]. Image input is not supported in this stage, so
/// media is a 501 as llama.cpp answers without `--mmproj`.
async fn systemone(State(app): State<Arc<App>>, OrderedJson(body): OrderedJson) -> Response {
    let Some(decider) = app.decider.as_ref() else {
        return not_supported("This model is not a decision model");
    };
    let (state, questions) = match decider.parse_request(&body) {
        Ok(q) => q,
        Err(e) => return bad(e),
    };
    let media = match peregrine_core::decision::parse_media(&body) {
        Ok(m) => m,
        Err(e) => return bad(e),
    };
    if media {
        return not_supported(
            "This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`",
        );
    }
    // Render and tokenize before reserving a place: an over-length prompt is the caller's error
    // and must not hold the engine.
    let mut prompts = Vec::with_capacity(questions.len());
    for (_, question) in &questions {
        let tokens = match decider.prompt_tokens(question, &state, &app.model.tk) {
            Ok(t) => t,
            Err(e) => return bad(e),
        };
        if tokens.len() >= app.max_ctx {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": format!(
                    "request ({} tokens) exceeds the available context size ({} tokens), try increasing it",
                    tokens.len(), app.max_ctx), "type": "exceed_context_size_error", "code": 400}})),
            )
                .into_response();
        }
        prompts.push(tokens);
    }
    // One place for the whole request, as `/v1/embeddings`: questions beyond the free slots wait
    // in the engine's queue and are admitted in order.
    if !try_reserve_n(&app.busy, 1, MIN_WAITING.max(2 * app.max_slots)) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": "too many requests are waiting for this model", "type": "overloaded_error", "code": 503}})),
        )
            .into_response();
    }
    let _busy = BusyGuard(app.clone());
    app.requests.fetch_add(1, Ordering::SeqCst);
    let labels = decider.labels();
    // S4.9j: one group for the whole request — the questions' shared beginning is forwarded once
    // and forked into each question's slot; the engine splits a request wider than its slots into
    // several groups, each forked from one evaluation (or a prefix-store restore) of it.
    let labels: Vec<Vec<u32>> = questions
        .iter()
        // Option `i` is read at letter `i`; a question with fewer options reads fewer.
        .map(|(_, question)| labels[..question.options.len()].to_vec())
        .collect();
    let emissions = match run_group_jobs(&app, prompts, labels).await {
        Ok(e) => e,
        Err(response) => return *response,
    };
    let mut answers = Vec::with_capacity(questions.len());
    let mut input_tokens = 0usize;
    for ((id, question), em) in questions.iter().zip(emissions) {
        input_tokens += em.cost.as_ref().map_or(0, |c| c.prompt_tokens);
        let Some(scores) = em.label_logits else {
            return internal("the engine returned no label logits");
        };
        let info = decider.info();
        match peregrine_core::decision::format_answer(
            info.kind(),
            question,
            &[scores],
            info.temperatures(),
            info.score_bias(),
        ) {
            Ok(answer) => answers.push((id.clone(), answer)),
            Err(e) => return bad(e),
        }
    }
    ordered_json(&peregrine_core::decision::response_json(
        &app.alias,
        &answers,
        input_tokens,
    ))
}

async fn chat(State(app): State<Arc<App>>, OpenAiJson(body): OpenAiJson) -> Response {
    if app.decider.is_some() {
        return not_supported(
            "This model is a decision model without text generation, use /v1/systemone",
        );
    }
    // `gemma-embedding` (and, later, the other encoders) has no decode path: the
    // request is refused by name rather than served as a broken completion (design §3).
    if app.model.causality() == peregrine_core::arch::Causality::Bidirectional {
        return bad(format!(
            "{} is an embedding model and cannot generate text; POST /v1/embeddings returns its vectors",
            app.model.cfg.arch_name
        ));
    }
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
    use super::{
        base64_f32, continuation_prefix, fit_tokens, parse_embedding_request, plan, EmbedInput,
        Penalties, ReasoningFormat,
    };
    use axum::Json;
    use serde_json::{json, Value};
    use std::time::{Duration, Instant};

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
    fn malformed_json_has_the_openai_error_shape() {
        let (status, Json(body)) =
            super::bad_json("the request body is not valid JSON: messages: EOF".into());
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], 400);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not valid JSON"));
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
    fn stop_accepts_one_string_or_up_to_four_in_an_array() {
        assert_eq!(
            plan(&body(json!({"stop": "\n\n"}))).unwrap().stop,
            vec!["\n\n".to_string()]
        );
        assert_eq!(
            plan(&body(json!({"stop": ["a", "b", "c", "d"]})))
                .unwrap()
                .stop,
            vec!["a", "b", "c", "d"]
        );
        assert!(plan(&body(json!({"stop": []}))).unwrap().stop.is_empty());
        assert!(plan(&body(json!({"stop": null}))).unwrap().stop.is_empty());
        assert!(plan(&body(json!({}))).unwrap().stop.is_empty());
    }

    #[test]
    fn stop_refuses_more_than_four_or_non_string_sequences() {
        for bad in [
            json!({"stop": ["a", "b", "c", "d", "e"]}),
            json!({"stop": ["a", 1]}),
            json!({"stop": {"a": true}}),
            json!({"stop": 3}),
        ] {
            assert!(plan(&body(bad.clone())).is_err(), "accepted {bad}");
        }
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

        // With nothing pinned, `serve` calibrates: the options carry a measured thread count and
        // a spin window inside the calibration's clamp, and the source says where they came from.
        let calibrated = super::engine_options(&[], None).unwrap();
        assert!(calibrated.calibrate);
        assert!(calibrated.decode_threads.is_some());
        assert!((50..=2000).contains(&calibrated.spin_us));
    }

    /// One HTTP/1.1 GET, read to the end of the body (`Connection: close`).
    async fn http_get(port: u16, path: &str) -> String {
        tokio::time::timeout(Duration::from_secs(5), async {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .expect("connect to the test server");
            stream
                .write_all(
                    format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
                        .as_bytes(),
                )
                .await
                .unwrap();
            let mut out = Vec::new();
            stream.read_to_end(&mut out).await.unwrap();
            String::from_utf8_lossy(&out).into_owned()
        })
        .await
        .expect("the test server did not answer")
    }

    async fn wait_for_health(port: u16) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            let attempt = tokio::time::timeout(Duration::from_secs(2), async {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
                    .await
                    .ok()?;
                stream
                    .write_all(
                        b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .ok()?;
                let mut out = Vec::new();
                stream.read_to_end(&mut out).await.ok()?;
                Some(String::from_utf8_lossy(&out).into_owned())
            })
            .await;
            if matches!(attempt, Ok(Some(ref response)) if response.starts_with("HTTP/1.1 200")) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("the test server never answered /health");
    }

    /// The v0.1.1 bug: a `timeout` around the whole serving future made the server exit after
    /// the drain timeout even though nothing had asked it to stop. This test fails on that logic.
    #[tokio::test]
    async fn the_server_outlives_the_drain_timeout_when_nothing_asked_it_to_stop() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let router = axum::Router::new().route("/health", axum::routing::get(|| async { "ok" }));
        let task = tokio::spawn(super::serve_with_drain(
            listener,
            router,
            std::future::pending::<()>(),
            Duration::from_millis(300),
        ));
        wait_for_health(port).await;
        // Well past the 300 ms drain timeout, with no shutdown request.
        tokio::time::sleep(Duration::from_millis(900)).await;
        let response = http_get(port, "/health").await;
        assert!(response.starts_with("HTTP/1.1 200"), "{response:?}");
        assert!(
            !task.is_finished(),
            "the server exited without a shutdown request"
        );
        task.abort();
        let _ = task.await;
    }

    #[tokio::test]
    async fn a_shutdown_with_no_open_connection_finishes_promptly() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let router = axum::Router::new().route("/health", axum::routing::get(|| async { "ok" }));
        let task = tokio::spawn(super::serve_with_drain(
            listener,
            router,
            async move {
                let _ = shutdown_rx.await;
            },
            Duration::from_secs(5),
        ));
        tokio::time::sleep(Duration::from_millis(100)).await;
        let started = Instant::now();
        shutdown_tx.send(()).unwrap();
        let outcome = tokio::time::timeout(Duration::from_secs(10), task)
            .await
            .expect("a shutdown with no connections hung")
            .unwrap()
            .unwrap();
        assert_eq!(outcome, super::Drain::Finished);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "an idle drain took {:?}",
            started.elapsed()
        );
    }

    /// A client that holds a stream open keeps the drain alive until the timeout, and no longer.
    #[tokio::test]
    async fn an_open_stream_holds_the_drain_until_the_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let started_tx = std::sync::Arc::new(std::sync::Mutex::new(Some(started_tx)));
        let router = axum::Router::new().route(
            "/stream",
            axum::routing::get(move || {
                let started = started_tx.clone();
                async move {
                    if let Some(tx) = started.lock().unwrap().take() {
                        let _ = tx.send(());
                    }
                    // A body that never finishes, like a token stream whose client stays connected.
                    let body = axum::body::Body::from_stream(futures::stream::pending::<
                        Result<axum::body::Bytes, std::convert::Infallible>,
                    >());
                    axum::response::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(body)
                        .unwrap()
                }
            }),
        );
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(super::serve_with_drain(
            listener,
            router,
            async move {
                let _ = shutdown_rx.await;
            },
            Duration::from_millis(300),
        ));

        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        stream
            .write_all(b"GET /stream HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            .await
            .unwrap();
        started_rx.await.expect("the stream handler ran");
        let mut head = [0u8; 1024];
        let n = stream.read(&mut head).await.unwrap();
        assert!(
            String::from_utf8_lossy(&head[..n]).starts_with("HTTP/1.1 200"),
            "{:?}",
            String::from_utf8_lossy(&head[..n])
        );

        let started = Instant::now();
        shutdown_tx.send(()).unwrap();
        let outcome = tokio::time::timeout(Duration::from_secs(10), task)
            .await
            .expect("the drain never ended with a stream open")
            .unwrap()
            .unwrap();
        let elapsed = started.elapsed();
        assert_eq!(outcome, super::Drain::TimedOut);
        assert!(
            elapsed >= Duration::from_millis(300),
            "the in-flight request was cut off before its drain window: {elapsed:?}"
        );
        let notice = super::drain_timeout_notice(Duration::from_millis(300));
        assert!(
            notice.starts_with("peregrine: shutdown took longer than"),
            "{notice:?}"
        );
        assert_eq!(
            super::drain_timeout_notice(Duration::from_secs(30)),
            "peregrine: shutdown took longer than 30 s; exiting with requests still open"
        );
        drop(stream);
    }

    #[test]
    fn embedding_input_shapes_are_the_openai_ones() {
        let one = parse_embedding_request(&json!({"input": "hello"}), 8).unwrap();
        assert_eq!(one.inputs, vec![EmbedInput::Text("hello".into())]);

        let many = parse_embedding_request(&json!({"input": ["a", "b"]}), 8).unwrap();
        assert_eq!(
            many.inputs,
            vec![EmbedInput::Text("a".into()), EmbedInput::Text("b".into())]
        );

        let ids = parse_embedding_request(&json!({"input": [11, 12, 13]}), 8).unwrap();
        assert_eq!(ids.inputs, vec![EmbedInput::Tokens(vec![11, 12, 13])]);

        let arrays = parse_embedding_request(&json!({"input": [[11], [12, 13]]}), 8).unwrap();
        assert_eq!(
            arrays.inputs,
            vec![
                EmbedInput::Tokens(vec![11]),
                EmbedInput::Tokens(vec![12, 13])
            ]
        );
    }

    #[test]
    fn embedding_fields_are_validated_and_unknown_ones_refused() {
        assert!(parse_embedding_request(&json!({}), 8).is_err());
        assert!(parse_embedding_request(&json!({"input": []}), 8).is_err());
        assert!(parse_embedding_request(&json!({"input": [1, -2]}), 8).is_err());
        assert!(parse_embedding_request(&json!({"input": [1.5]}), 8).is_err());
        assert!(parse_embedding_request(&json!({"input": ["a", 3]}), 8).is_err());
        assert!(parse_embedding_request(&json!({"input": [[]]}), 8).is_err());
        // `dimensions` must be a positive integer that fits the model.
        assert_eq!(
            parse_embedding_request(&json!({"input": "a", "dimensions": 4}), 8)
                .unwrap()
                .dimensions,
            Some(4)
        );
        assert!(parse_embedding_request(&json!({"input": "a", "dimensions": 0}), 8).is_err());
        assert!(parse_embedding_request(&json!({"input": "a", "dimensions": 9}), 8).is_err());
        assert!(parse_embedding_request(&json!({"input": "a", "dimensions": "4"}), 8).is_err());
        // `encoding_format` is `float` or `base64`.
        assert!(
            !parse_embedding_request(&json!({"input": "a"}), 8)
                .unwrap()
                .base64
        );
        assert!(
            !parse_embedding_request(&json!({"input": "a", "encoding_format": "float"}), 8)
                .unwrap()
                .base64
        );
        assert!(
            parse_embedding_request(&json!({"input": "a", "encoding_format": "base64"}), 8)
                .unwrap()
                .base64
        );
        assert!(
            parse_embedding_request(&json!({"input": "a", "encoding_format": "int8"}), 8).is_err()
        );
        assert!(
            parse_embedding_request(&json!({"input": "a", "encoding_format": 1}), 8).is_err(),
            "a non-string format is refused, not read as float"
        );
        // The truncation opt-in is a boolean.
        assert!(
            parse_embedding_request(&json!({"input": "a", "truncate": true}), 8)
                .unwrap()
                .truncate
        );
        assert!(parse_embedding_request(&json!({"input": "a", "truncate": 1}), 8).is_err());
        // Unknown fields are refused by name, as chat refuses unsupported sampling fields.
        let err = parse_embedding_request(&json!({"input": "a", "pooling": "mean"}), 8)
            .err()
            .unwrap_or_default();
        assert!(err.contains("pooling"), "{err:?}");
        assert!(parse_embedding_request(&json!({"input": "a", "user": "u"}), 8).is_ok());
    }

    #[test]
    fn over_length_names_the_limit_unless_truncation_is_asked_for() {
        let mut t: Vec<u32> = (0..9).collect();
        let err = fit_tokens(0, &mut t, 8, false).err().unwrap_or_default();
        assert!(err.contains("9 tokens"), "{err:?}");
        assert!(err.contains("8 positions"), "{err:?}");
        fit_tokens(0, &mut t, 8, true).unwrap();
        assert_eq!(t.len(), 8);
        assert!(t == (0..8).collect::<Vec<u32>>());
        // A text that tokenizes to nothing is refused too.
        assert!(fit_tokens(2, &mut vec![], 8, false).is_err());
    }

    #[test]
    fn base64_is_little_endian_f32() {
        // 1.0f32 is 00 00 80 3f, whose base64 is AACAPw==.
        assert_eq!(base64_f32(&[1.0]), "AACAPw==");
        assert_eq!(base64_f32(&[0.0]), "AAAAAA==");
        // Eight bytes are three chunks: the last has two bytes and one `=` of padding.
        assert_eq!(base64_f32(&[1.0, 1.0]).len(), 12);
        assert!(base64_f32(&[1.0, 1.0]).ends_with('='));
        assert!(!base64_f32(&[1.0, 1.0]).ends_with("=="));
    }
}
