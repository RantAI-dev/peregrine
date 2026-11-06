//! `peregrine tune`: coordinate search over engine tuning knobs (roadmap S1.7).

use clap::Args;
use peregrine_core::profile::{self, Profile};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// Arguments for `peregrine tune`.
#[derive(Args, Debug, Clone)]
pub struct TuneArgs {
    /// The GGUF model file to tune.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// Time budget for tuning in seconds (default: 60).
    #[arg(long, default_value_t = 60)]
    pub budget_s: u64,
    /// Output path for the tuning profile (default: `$XDG_CONFIG_HOME/peregrine/profiles/<model>-<fnv>.profile`).
    #[arg(long)]
    pub out: Option<PathBuf>,
}

struct TrialResult {
    knob: String,
    value: String,
    median_tok_s: f64,
    spread: f64,
    result: String,
}

fn median_and_spread(mut samples: Vec<f64>) -> Option<(f64, f64)> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = samples[samples.len() / 2];
    let spread = samples.last().unwrap() - samples.first().unwrap();
    Some((median, spread))
}

fn run_candidate_bench(
    exe: &Path,
    model: &Path,
    env_vars: &BTreeMap<String, String>,
) -> Result<f64, String> {
    let mut cmd = Command::new(exe);
    cmd.arg("bench")
        .arg("-m")
        .arg(model)
        .arg("--tokens")
        .arg("32")
        .arg("--prompt-tokens")
        .arg("32")
        .arg("--json")
        .arg("--no-profile");

    for (k, v) in env_vars {
        cmd.env(k, v);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("failed to execute {}: {e}", exe.display()))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "candidate benchmark failed (status {:?}): {err}",
            output.status.code()
        ));
    }

    let json_val: serde_json::Value = serde_json::from_slice(&output.stdout).map_err(|e| {
        format!(
            "failed to parse benchmark JSON: {e}\nstdout: {}",
            String::from_utf8_lossy(&output.stdout)
        )
    })?;

    json_val
        .get("decode_tok_s")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "missing decode_tok_s in benchmark JSON".to_string())
}

/// Run `peregrine tune`.
pub fn run(args: &TuneArgs) -> Result<(), String> {
    let out_path = match &args.out {
        Some(p) => p.clone(),
        None => profile::default_path(&args.model)?,
    };

    let mut profile = Profile::new_for_model(&args.model)?;

    let current_exe = std::env::current_exe()
        .map_err(|e| format!("cannot determine current executable path: {e}"))?;

    let phys_cores = peregrine_core::pool::physical_cores()
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));

    let start_time = Instant::now();
    let budget = Duration::from_secs(args.budget_s);

    let mut trials = Vec::new();
    let mut current_best_opts: BTreeMap<String, String> = BTreeMap::new();

    const ROUNDS: usize = 3;

    // Baseline evaluation
    let mut baseline_samples = Vec::new();
    for _ in 0..ROUNDS {
        if start_time.elapsed() >= budget {
            break;
        }
        let tok_s = run_candidate_bench(&current_exe, &args.model, &current_best_opts)?;
        baseline_samples.push(tok_s);
    }

    let (mut best_median, mut best_spread) = median_and_spread(baseline_samples)
        .ok_or("time budget exhausted before baseline could finish")?;

    trials.push(TrialResult {
        knob: "(baseline)".into(),
        value: "defaults".into(),
        median_tok_s: best_median,
        spread: best_spread,
        result: "baseline".into(),
    });

    // Define search space
    // Each knob has (knob_name, env_var, candidate_values)
    let mut knobs: Vec<(&str, &str, Vec<String>)> = Vec::new();

    // 1. decode_threads: 1..=phys_cores
    let mut thread_vals = Vec::new();
    for t in 1..=phys_cores {
        thread_vals.push(t.to_string());
    }
    knobs.push(("decode_threads", "PEREGRINE_DECODE_THREADS", thread_vals));

    // 2. spin_us: 0, 50, 200, 1000, 5000
    knobs.push((
        "spin_us",
        "PEREGRINE_SPIN_US",
        vec![
            "0".into(),
            "50".into(),
            "200".into(),
            "1000".into(),
            "5000".into(),
        ],
    ));

    // 3. prefetch_bytes: 0, 512, 1024, 2048
    knobs.push((
        "prefetch_bytes",
        "PEREGRINE_PF",
        vec!["0".into(), "512".into(), "1024".into(), "2048".into()],
    ));

    // Coordinate search over knobs
    'knob_loop: for (knob_name, env_var, values) in knobs {
        if start_time.elapsed() >= budget {
            break 'knob_loop;
        }

        let mut candidate_samples: Vec<(String, Vec<f64>)> = values
            .into_iter()
            .map(|val| (val, Vec::with_capacity(ROUNDS)))
            .collect();

        // Run interleaved rounds
        for _ in 0..ROUNDS {
            for (val, samples) in &mut candidate_samples {
                if start_time.elapsed() >= budget {
                    break;
                }
                let mut env = current_best_opts.clone();
                env.insert(env_var.to_string(), val.clone());
                match run_candidate_bench(&current_exe, &args.model, &env) {
                    Ok(tok_s) => samples.push(tok_s),
                    Err(e) => eprintln!("warning: candidate {knob_name}={val} error: {e}"),
                }
            }
        }

        // Evaluate candidate results for this knob
        let mut best_cand: Option<(String, f64, f64)> = None;

        for (val, samples) in candidate_samples {
            if let Some((median, spread)) = median_and_spread(samples) {
                if let Some((_, b_med, _)) = &best_cand {
                    if median > *b_med {
                        best_cand = Some((val, median, spread));
                    }
                } else {
                    best_cand = Some((val, median, spread));
                }
            }
        }

        if let Some((best_val, cand_median, cand_spread)) = best_cand {
            let noise = best_spread.max(cand_spread);
            if cand_median > best_median + noise {
                let gain = (cand_median - best_median) / best_median * 100.0;
                trials.push(TrialResult {
                    knob: knob_name.into(),
                    value: best_val.clone(),
                    median_tok_s: cand_median,
                    spread: cand_spread,
                    result: format!("kept (+{gain:.1}%)"),
                });
                current_best_opts.insert(env_var.to_string(), best_val);
                best_median = cand_median;
                best_spread = cand_spread;
            } else {
                trials.push(TrialResult {
                    knob: knob_name.into(),
                    value: best_val,
                    median_tok_s: cand_median,
                    spread: cand_spread,
                    result: "rejected".into(),
                });
            }
        }
    }

    // Populate profile with chosen options
    // Map PEREGRINE_* env vars back to option keys
    for (k, v) in &current_best_opts {
        let opt_key = match k.as_str() {
            "PEREGRINE_DECODE_THREADS" => "decode_threads",
            "PEREGRINE_SPIN_US" => "spin_us",
            "PEREGRINE_PF" => "prefetch_bytes",
            other => other.strip_prefix("PEREGRINE_").unwrap_or(other),
        };
        profile.set(opt_key, v);
    }

    // Write profile
    profile::write(&out_path, &profile)
        .map_err(|e| format!("failed to write profile {}: {e}", out_path.display()))?;

    // Print table of what was tried
    println!(
        "{:<16} {:<10} {:>14} {:>10}  Result",
        "Knob", "Value", "Decode tok/s", "Spread"
    );
    println!(
        "{:-<16} {:-<10} {:-<14} {:-<10}  {:-<12}",
        "", "", "", "", ""
    );
    for t in &trials {
        println!(
            "{:<16} {:<10} {:>14.2} {:>10.2}  {}",
            t.knob, t.value, t.median_tok_s, t.spread, t.result
        );
    }
    println!();
    println!("wrote profile to {}", out_path.display());

    Ok(())
}
