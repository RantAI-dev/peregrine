//! `peregrine`: run, serve and inspect models.
#[cfg(feature = "server")]
use clap::Args;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "peregrine",
    version,
    about = "A CPU inference engine for language models"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve a model behind the OpenAI chat API (the same options as `peregrine-server`).
    #[cfg(feature = "server")]
    Serve(ServeCmd),
    /// Run one prompt and print the reply, without starting a server.
    Run(peregrine::run::RunArgs),
    /// Describe a model file and whether this build can run it.
    Inspect {
        /// The GGUF file, or `hf:<namespace>/<repo>/<file>[@<rev>]` (downloaded once into the cache).
        model: std::path::PathBuf,
    },
    /// Embed texts: one vector per text, from a causal model (an embedding model).
    Embed(peregrine::embed::EmbedArgs),
    /// Answer questions about a state with a native decision model (System One).
    Decide(peregrine::decide::DecideArgs),
    /// Benchmark single-user prefill and greedy decode throughput.
    Bench(peregrine::bench::BenchArgs),
    /// Search engine tuning knobs and write a machine- and model-specific profile.
    Tune(peregrine::tune::TuneArgs),
    /// Install the newest release and replace a running background server without downtime.
    #[cfg(feature = "server")]
    Upgrade(UpgradeCmd),
    /// Show the managed background server.
    #[cfg(feature = "server")]
    Status,
    /// Stop the managed background server and let it drain.
    #[cfg(feature = "server")]
    Stop,
    /// The CPU kernel tier and every engine option in effect.
    Tier {
        /// Also report the pack cache for this model (`-m`), if it has one.
        #[arg(short = 'm', long)]
        model: Option<std::path::PathBuf>,
    },
    /// Describe supported architectures, formats, tokenizers, templates, and CPU tiers.
    Support {
        /// Format as Markdown table for README.md.
        #[arg(long)]
        markdown: bool,
    },
}

#[cfg(feature = "server")]
#[derive(Args)]
struct ServeCmd {
    #[command(flatten)]
    args: peregrine::server::Args,
    /// Run detached: log to a file, record lifecycle state, print the address and return.
    /// `peregrine status`, `stop` and `upgrade` manage that server.
    #[arg(long)]
    background: bool,
    /// Log file for a background server (default: the state directory).
    #[arg(long)]
    log_file: Option<std::path::PathBuf>,
    /// Seconds a background server gets to load its model and report ready.
    #[arg(long, default_value_t = 300)]
    startup_timeout: u64,
}

#[cfg(feature = "server")]
#[derive(Args)]
struct UpgradeCmd {
    /// Only report whether a newer release exists; install nothing.
    #[arg(long)]
    check: bool,
    /// Install this tag instead of the newest release (for example v0.1.2).
    #[arg(long)]
    version: Option<String>,
    /// Release download base: a mirror, or a `file://` directory in tests.
    #[arg(long)]
    base_url: Option<String>,
    /// Replace this binary instead of the running one.
    #[arg(long)]
    exe: Option<std::path::PathBuf>,
    /// Install, but leave a running server on the old binary until it is restarted.
    #[arg(long)]
    no_restart: bool,
    /// Replace even a cargo build under `target/`.
    #[arg(long)]
    force: bool,
}

fn main() {
    match Cli::parse().cmd {
        #[cfg(feature = "server")]
        Cmd::Serve(cmd) => {
            let ServeCmd {
                mut args,
                background,
                log_file,
                startup_timeout,
            } = cmd;
            args.model = resolved(args.model, "peregrine serve");
            if background {
                match peregrine::service::start_background(
                    args.port,
                    log_file.or_else(|| args.log_file.clone()),
                    std::time::Duration::from_secs(startup_timeout),
                ) {
                    Ok(state) => {
                        println!(
                            "peregrine: background server ready at {} (pid {})",
                            state.url(),
                            state.pid
                        );
                    }
                    Err(e) => {
                        eprintln!("peregrine serve: {e}");
                        std::process::exit(1);
                    }
                }
                return;
            }
            peregrine::server::init_logging();
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("tokio runtime")
                .block_on(peregrine::server::run(args));
        }
        Cmd::Run(mut args) => {
            args.model = resolved(args.model, "peregrine run");
            if let Err(e) = peregrine::run::run(&args) {
                eprintln!("peregrine run: {e}");
                std::process::exit(1);
            }
        }
        Cmd::Inspect { model } => {
            let model = resolved(model, "peregrine inspect");
            match peregrine::inspect(&model) {
                Ok(s) => print!("{s}"),
                Err(e) => {
                    eprintln!("peregrine: {e}");
                    std::process::exit(1);
                }
            }
        }
        Cmd::Embed(mut args) => {
            args.model = resolved(args.model, "peregrine embed");
            if let Err(e) = peregrine::embed::run(&args) {
                eprintln!("peregrine embed: {e}");
                std::process::exit(1);
            }
        }
        Cmd::Decide(mut args) => {
            args.model = resolved(args.model, "peregrine decide");
            if let Err(e) = peregrine::decide::run(&args) {
                eprintln!("peregrine decide: {e}");
                std::process::exit(1);
            }
        }
        Cmd::Bench(mut args) => {
            args.model = resolved(args.model, "peregrine bench");
            if let Err(e) = peregrine::bench::run(&args) {
                eprintln!("peregrine bench: {e}");
                std::process::exit(1);
            }
        }
        Cmd::Tune(mut args) => {
            args.model = resolved(args.model, "peregrine tune");
            if let Err(e) = peregrine::tune::run(&args) {
                eprintln!("peregrine tune: {e}");
                std::process::exit(1);
            }
        }
        #[cfg(feature = "server")]
        Cmd::Upgrade(cmd) => {
            let args = peregrine::service::UpgradeArgs {
                check: cmd.check,
                version: cmd.version,
                base_url: cmd.base_url,
                exe: cmd.exe,
                no_restart: cmd.no_restart,
                force: cmd.force,
            };
            match peregrine::service::upgrade(&args) {
                Ok(out) => print!("{out}"),
                Err(e) => {
                    eprintln!("peregrine upgrade: {e}");
                    std::process::exit(1);
                }
            }
        }
        #[cfg(feature = "server")]
        Cmd::Status => match peregrine::service::status() {
            Ok(out) => print!("{out}"),
            Err(e) => {
                eprintln!("peregrine status: {e}");
                std::process::exit(1);
            }
        },
        #[cfg(feature = "server")]
        Cmd::Stop => match peregrine::service::stop() {
            Ok(out) => print!("{out}"),
            Err(e) => {
                eprintln!("peregrine stop: {e}");
                std::process::exit(1);
            }
        },
        Cmd::Tier { model } => {
            print!("{}", peregrine::describe_runtime());
            if let Some(model) = model {
                let model = resolved(model, "peregrine tier");
                let r =
                    peregrine_core::packcache::model_status(&model, peregrine_core::options::get());
                println!("pack cache    {}", r.describe());
            }
        }
        Cmd::Support { markdown } => print!("{}", peregrine::support(markdown)),
    }
}

/// A `-m` value as a path on this machine: `hf:` references are downloaded once into the cache.
fn resolved(model: std::path::PathBuf, cmd: &str) -> std::path::PathBuf {
    match peregrine::fetch::resolve(&model) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{cmd}: {e}");
            std::process::exit(1);
        }
    }
}
