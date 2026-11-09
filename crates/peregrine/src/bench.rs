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
    /// Speculate with the model's MTP block (files with an MTP head). Off by
    /// default, so a plain `bench` measures the target alone.
    #[arg(long, default_value_t = false)]
    pub mtp: bool,
    /// A tuning profile file to apply (default: default profile for model and machine).
    #[arg(long)]
    pub profile: Option<PathBuf>,
    /// Ignore tuning profiles and use defaults or environment variables.
    #[arg(long, default_value_t = false)]
    pub no_profile: bool,
    /// Worker threads (default: every logical CPU). Pin the process to as many CPUs when
    /// comparing against an engine started with `-t N`.
    #[arg(long)]
    pub threads: Option<usize>,
}

/// Run `peregrine bench`.
pub fn run(args: &BenchArgs) -> Result<(), String> {
    let (prof, prof_msg) = profile::resolve(&args.model, args.profile.as_deref(), args.no_profile);

    let opts = EngineOptions::from_env_and_profile(prof.as_ref());
    let _ = peregrine_core::options::init(opts.clone());
    let effective_opts = peregrine_core::options::get();

    let needed = (args.prompt_tokens + args.tokens).max(64);
    let mut cfg = SessionCfg {
        speculate: false,
        temperature: 0.0,
        max_ctx: needed.max(512),
        ..Default::default()
    };
    if let Some(t) = args.threads {
        cfg.threads = t.max(1);
    }
    let threads = cfg.threads;

    let mut session = peregrine_models::session::open(&args.model, cfg)
        .map_err(|e| format!("cannot open session for {}: {e}", args.model.display()))?;
    if args.mtp {
        session.set_mtp(true);
    }

    // The hybrid forward's own buckets (`PEREGRINE_PROF=1`) are process-wide
    // accumulators: clear anything before the measurement so the report covers
    // this run only.
    #[cfg(feature = "arch-qwen35")]
    let hybrid_prof = peregrine_core::options::get().prof;
    #[cfg(feature = "arch-qwen35")]
    if hybrid_prof {
        let _ = peregrine_models::hybrid::profile();
    }

    let report = session.bench(args.prompt_tokens, args.tokens)?;

    #[cfg(feature = "arch-qwen35")]
    if hybrid_prof {
        let pr = peregrine_models::hybrid::profile();
        let total: u64 = pr.iter().map(|&(_, v)| v).sum();
        let n = (report.prompt_tokens + report.decode_tokens).max(1) as f64;
        eprintln!(
            "peregrine hybrid profile: {} prompt + {} decode tokens, {:.1} ms, µs per token",
            report.prompt_tokens,
            report.decode_tokens,
            total as f64 / 1e6
        );
        for (name, ns) in pr {
            eprintln!(
                "  {name:<14} {:>10.1}  {:>5.1} %",
                ns as f64 / 1e3 / n,
                100.0 * ns as f64 / total.max(1) as f64
            );
        }
    }

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
            // The kernel tier this run takes, so a saved result names the kernels
            // that produced it (the instruction-set string alone stops at AVX2).
            "simd_tier": peregrine_core::simd_tier(),
            "config": {
                "threads": threads,
                "decode_threads": decode_th,
                "spin_us": effective_opts.spin_us,
                "tuning_source": effective_opts.tuning_source.as_str(),
                "calibration_ms": effective_opts.calibration_ms(),
            },
            "prefill_tok_s": report.prefill_tok_s,
            "decode_tok_s": report.decode_tok_s,
            "mtp": args.mtp,
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
