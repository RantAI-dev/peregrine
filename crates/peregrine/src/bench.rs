//! `peregrine bench`: single-user throughput and bytes/token measurement (roadmap S1.7).

use clap::Args;
use peregrine_core::options::EngineOptions;
use peregrine_core::profile;
use peregrine_core::SessionCfg;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Arguments for `peregrine bench`.
#[derive(Args, Debug, Clone)]
pub struct BenchArgs {
    /// The GGUF model file to benchmark.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// Number of greedy decode tokens to generate.
    #[arg(long, default_value_t = 32)]
    pub tokens: usize,
    /// Number of synthetic prompt tokens to prefill.
    #[arg(long, default_value_t = 32)]
    pub prompt_tokens: usize,
    /// Output results as a single JSON object.
    #[arg(long, default_value_t = false)]
    pub json: bool,
    /// A tuning profile file to apply (default: default profile for model and machine).
    #[arg(long)]
    pub profile: Option<PathBuf>,
    /// Ignore tuning profiles and use defaults or environment variables.
    #[arg(long, default_value_t = false)]
    pub no_profile: bool,
}

/// Run `peregrine bench`.
pub fn run(args: &BenchArgs) -> Result<(), String> {
    let (prof, prof_msg) = profile::resolve(&args.model, args.profile.as_deref(), args.no_profile);

    let opts = EngineOptions::from_env_and_profile(prof.as_ref());
    let _ = peregrine_core::options::init(opts.clone());
    let effective_opts = peregrine_core::options::get();

    let needed = (args.prompt_tokens + args.tokens).max(64);
    let cfg = SessionCfg {
        speculate: false,
        temperature: 0.0,
        max_ctx: needed.max(512),
        ..Default::default()
    };

    let mut session = peregrine_models::session::open(&args.model, cfg)
        .map_err(|e| format!("cannot open session for {}: {e}", args.model.display()))?;

    let report = session.bench(args.prompt_tokens, args.tokens)?;

    if prof.is_some() {
        let p_path = args
            .profile
            .as_deref()
            .map(|p| p.to_path_buf())
            .or_else(|| profile::default_path(&args.model).ok());
        if let Some(p) = p_path {
            let dt = effective_opts
                .decode_threads
                .unwrap_or_else(|| peregrine_core::pool::physical_cores().unwrap_or(1));
            eprintln!(
                "peregrine: using tuning profile {} (decode_threads={}, spin_us={})",
                p.display(),
                dt,
                effective_opts.spin_us
            );
        }
    }

    if args.json {
        let opts_map: BTreeMap<&str, String> = effective_opts.pairs().into_iter().collect();
        let decode_th = effective_opts
            .decode_threads
            .unwrap_or_else(|| peregrine_core::pool::physical_cores().unwrap_or(1));
        let obj = serde_json::json!({
            "model": args.model.display().to_string(),
            "config": {
                "decode_threads": decode_th,
                "spin_us": effective_opts.spin_us,
                "tuning_source": effective_opts.tuning_source.as_str(),
                "calibration_ms": effective_opts.calibration_ms(),
            },
            "prefill_tok_s": report.prefill_tok_s,
            "decode_tok_s": report.decode_tok_s,
            "bytes_per_token": report.bytes_per_token,
            "prefill_bytes_acts": report.prefill_bytes_acts,
            "decode_bytes_acts": report.decode_bytes_acts,
            "prompt_tokens": report.prompt_tokens,
            "tokens": report.decode_tokens,
            "prefill_s": report.prefill_s,
            "decode_s": report.decode_s,
            "profile": prof_msg,
            "options": opts_map,
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap());
    } else {
        println!("{prof_msg}");
        println!("prefill tok/s   {:.1}", report.prefill_tok_s);
        println!("decode tok/s    {:.1}", report.decode_tok_s);
        println!("bytes/token     {:.1}", report.bytes_per_token);
        println!(
            "activation bytes  prefill {} MiB   decode {} MiB",
            report.prefill_bytes_acts / (1024 * 1024),
            report.decode_bytes_acts / (1024 * 1024)
        );
        println!(
            "options:\n{}",
            effective_opts
                .describe()
                .lines()
                .map(|l| format!("  {l}\n"))
                .collect::<String>()
        );
    }

    Ok(())
}
