//! `peregrine run`: one prompt in, one reply out, streamed, without a server.

use clap::Args;
use peregrine_core::kv::KvPrec;
use peregrine_core::options::{self, EngineOptions};
use peregrine_core::profile;
use peregrine_core::session::SessionCfg;
use peregrine_core::template::Template;
use std::collections::HashMap;
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Arguments for `peregrine run`.
#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// The GGUF file, or `hf:<namespace>/<repo>/<file>[@<rev>]` to download it into the cache once.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// The prompt. With no arguments the prompt is read from stdin, for pipes and scripts.
    #[arg(value_name = "PROMPT")]
    pub prompt: Vec<String>,
    /// Tokens to generate at most.
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,
    /// Sampling temperature; 0 is greedy and deterministic.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    /// Keep only the most probable tokens (0 = all).
    #[arg(long, default_value_t = 0)]
    pub top_k: usize,
    /// Keep tokens with at least this share of the best token's probability (0 = off).
    #[arg(long, default_value_t = 0.0)]
    pub min_p: f32,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Context length in tokens (clamped to the model's training context).
    #[arg(short = 'c', long, default_value_t = 8192)]
    pub ctx: usize,
    /// Worker threads (default: every logical CPU).
    #[arg(short = 't', long)]
    pub threads: Option<usize>,
    /// A system message for the chat template.
    #[arg(long)]
    pub system: Option<String>,
    /// The chat template, by name, when the file declares none or another one is wanted.
    #[arg(long = "chat-template", value_name = "NAME")]
    pub chat_template: Option<String>,
    /// KV cache precision: f16 (default), f32 or i8.
    #[arg(long)]
    pub kv: Option<String>,
    /// Answer without the reasoning block (reasoning models only).
    #[arg(long)]
    pub no_thinking: bool,
    /// Prompt-lookup speculation (lossless; off by default, as in `serve`).
    #[arg(long, default_value_t = false)]
    pub speculate: bool,
    /// Run projections a LittleBit sidecar covers from it (off by default, as in `serve`).
    #[arg(long, default_value_t = false)]
    pub littlebit: bool,
    /// Experimental, lossy, and off by default: prompt tokens read the weights of the formats whose
    /// 8-bit form beats this CPU tier's exact kernel from an 8-bit copy built at
    /// load (AVX-512: Q4_K and Q6_K; AVX2: Q6_K; aarch64 with `dotprod`/`i8mm`:
    /// Q4_K, Q6_K and Q5_0; aarch64 without them: Q4_K and Q6_K; a tier with no SIMD
    /// body: none, so the option changes nothing), and on AVX-512 a prompt row's
    /// attention is computed in the mode's own summation order too (a tiled order of
    /// the same sums; generated rows keep the exact attention on every tier). A prompt
    /// pass is 1.28-1.46x the exact engine's speed on AVX-512, 1.05-1.08x on an AVX2
    /// host (S5.4f), and 1.72-2.73x on Qwen 2.5 0.5B on ARM (S5.4h). On Llama 3.2 1B
    /// Q4_K_M (AVX-512) the prompt's logits move 0.0035 nats of mean KL divergence
    /// from the default engine's and 96.7 % of positions keep the same top token (the
    /// report: benchmarks/results/desktop-zen5-2026-11-13-s5.4j/). Generated tokens
    /// keep the exact kernels, on a cache the 8-bit prompt wrote, so a reply can
    /// differ. `peregrine tier` names the formats this CPU takes; the numbers are in
    /// `docs/design/s5.4-fast-prompt.md` sections 5c, 5g, 5h, 5k and 5l.
    #[arg(long = "fast-prompt", default_value_t = false)]
    pub fast_prompt: bool,
    /// Print one JSON object (text, token counts and timings) instead of streaming text.
    #[arg(long, default_value_t = false)]
    pub json: bool,
    /// Do not print the timing line to stderr.
    #[arg(long, default_value_t = false)]
    pub quiet: bool,
    /// An engine option as `NAME=VALUE`, as in `peregrine serve --opt`.
    #[arg(long = "opt", value_name = "NAME=VALUE")]
    pub opts: Vec<String>,
    /// A tuning profile file to apply (default: the profile for this model and machine).
    #[arg(long)]
    pub profile: Option<PathBuf>,
    /// Ignore tuning profiles; use environment variables and defaults.
    #[arg(long, default_value_t = false)]
    pub no_profile: bool,
}

/// Run `peregrine run`.
pub fn run(args: &RunArgs) -> Result<(), String> {
    let prompt = read_prompt(args)?;
    let (profile, prof_msg) =
        profile::resolve(&args.model, args.profile.as_deref(), args.no_profile);
    if profile.is_some() {
        eprintln!("peregrine: {prof_msg}");
    }
    let mut cli_opts = args.opts.clone();
    if args.fast_prompt {
        cli_opts.push("FAST_PROMPT=1".into());
    }
    let opts = engine_options(&cli_opts, profile.as_ref())?;
    options::init(opts).map_err(|e| format!("engine options: {e}"))?;

    let template = match args.chat_template.as_deref() {
        Some(name) => Some(Template::from_name(name).ok_or_else(|| {
            format!(
                "unknown chat template {name:?}; Peregrine renders {}",
                peregrine_core::template::TEMPLATES.join(", ")
            )
        })?),
        None => None,
    };
    let kv = match args.kv.as_deref() {
        Some(name) => KvPrec::parse(name)
            .ok_or_else(|| format!("unknown KV precision {name:?}: f16, f32 or i8"))?,
        None => options::get().kv_prec,
    };

    let mut cfg = SessionCfg {
        max_ctx: args.ctx,
        kv_prec: kv,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        min_p: args.min_p,
        seed: args.seed,
        max_tokens: args.max_tokens,
        system: args.system.clone(),
        template,
        thinking: !args.no_thinking,
        speculate: args.speculate,
        littlebit: args.littlebit,
        ..Default::default()
    };
    if let Some(t) = args.threads {
        cfg.threads = t.max(1);
    }

    let mut session = peregrine_models::session::open(&args.model, cfg)
        .map_err(|e| format!("cannot run {}: {e}", args.model.display()))?;

    let started = Instant::now();
    let json = args.json;
    // Qwen 3.5's template (Bonsai 2 carries it) opens the reasoning block in the prompt,
    // so the model writes its reasoning and only the closing tag. The opening tag is
    // shown first, so the output reads as the whole block, as it does for a model that
    // writes the tag itself.
    let opened = if session.template().opens_reasoning(!args.no_thinking) {
        "<think>\n"
    } else {
        ""
    };
    if !json {
        print!("{opened}");
    }
    let (text, telem) = session
        .generate(&prompt, |e| {
            if !json {
                print!("{}", e.text);
                let _ = std::io::stdout().flush();
            }
        })
        .map_err(|e| format!("generation failed: {e}"))?;
    let text = format!("{opened}{text}");
    let wall = started.elapsed().as_secs_f64();

    if json {
        let decode_s = telem.decode_s();
        let value = serde_json::json!({
            "model": args.model.display().to_string(),
            "text": text,
            "prompt_tokens": telem.prompt_tokens,
            "completion_tokens": telem.n(),
            "decode_tokens": telem.n_decode(),
            "prefill_tokens_per_second": telem.prefill_tok_per_s(),
            "decode_tokens_per_second": if decode_s > 0.0 {
                telem.n_decode() as f64 / decode_s
            } else {
                0.0
            },
            "ttft_ms": telem.ttft_s().map(|s| s * 1000.0),
            "wall_ms": wall * 1000.0,
            "ctx_used": telem.ctx_used,
            "cancelled": telem.cancelled,
        });
        println!(
            "{}",
            serde_json::to_string(&value).map_err(|e| format!("cannot format JSON: {e}"))?
        );
    } else {
        if !text.ends_with('\n') {
            println!();
        }
        if !args.quiet {
            let decode_s = telem.decode_s();
            let decode_tps = if decode_s > 0.0 {
                telem.n_decode() as f64 / decode_s
            } else {
                0.0
            };
            eprintln!(
                "peregrine: {} prompt tokens, {} decode tokens, {:.1} tok/s decode, {} first token, {:.2} s total",
                telem.prompt_tokens,
                telem.n_decode(),
                decode_tps,
                telem
                    .ttft_s()
                    .map_or_else(|| "-".to_string(), |s| format!("{:.0} ms", s * 1000.0)),
                wall,
            );
        }
    }
    Ok(())
}

/// The prompt from the arguments, or from stdin when none is given (a pipe or a script).
fn read_prompt(args: &RunArgs) -> Result<String, String> {
    if !args.prompt.is_empty() {
        return Ok(args.prompt.join(" "));
    }
    if std::io::stdin().is_terminal() {
        return Err("no prompt: pass it as an argument, or pipe it on stdin".into());
    }
    let mut s = String::new();
    std::io::stdin()
        .read_to_string(&mut s)
        .map_err(|e| format!("cannot read the prompt from stdin: {e}"))?;
    let prompt = s.trim_end_matches(['\r', '\n']).to_string();
    if prompt.is_empty() {
        return Err("the prompt on stdin is empty".into());
    }
    Ok(prompt)
}

/// The engine options: `--opt` first, then the environment, then the profile, then defaults.
fn engine_options(
    cli: &[String],
    profile: Option<&profile::Profile>,
) -> Result<EngineOptions, String> {
    let mut over = HashMap::new();
    for o in cli {
        let (k, v) = o
            .split_once('=')
            .ok_or_else(|| format!("--opt {o:?}: expected NAME=VALUE"))?;
        let k = k
            .trim()
            .trim_start_matches("PEREGRINE_")
            .to_ascii_uppercase();
        if !options::NAMES.contains(&k.as_str()) {
            return Err(format!(
                "--opt {k}: unknown option; known: {}",
                options::NAMES.join(", ")
            ));
        }
        over.insert(format!("PEREGRINE_{k}"), v.to_string());
    }
    Ok(EngineOptions::from_chained(
        |k| over.get(k).cloned().or_else(|| std::env::var(k).ok()),
        profile,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_is_opt_in_for_one_shot_runs() {
        let off = engine_options(&[], None).unwrap();
        assert!(
            !off.calibrate,
            "a one-shot run must not pay startup calibration unless asked"
        );
        let on = engine_options(&["CALIBRATE=1".to_string()], None).unwrap();
        assert!(on.calibrate, "--opt CALIBRATE=1 must enable calibration");
    }
}
