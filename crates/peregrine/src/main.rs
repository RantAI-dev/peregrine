//! `peregrine`: run, serve and inspect models.
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
    Serve(peregrine::server::Args),
    /// Describe a model file and whether this build can run it.
    Inspect {
        /// The GGUF file.
        model: std::path::PathBuf,
    },
    /// Benchmark single-user prefill and greedy decode throughput.
    Bench(peregrine::bench::BenchArgs),
    /// Search engine tuning knobs and write a machine- and model-specific profile.
    Tune(peregrine::tune::TuneArgs),
    /// The CPU kernel tier and every engine option in effect.
    Tier,
}

fn main() {
    match Cli::parse().cmd {
        #[cfg(feature = "server")]
        Cmd::Serve(args) => {
            peregrine::server::init_logging();
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("tokio runtime")
                .block_on(peregrine::server::run(args));
        }
        Cmd::Inspect { model } => match peregrine::inspect(&model) {
            Ok(s) => print!("{s}"),
            Err(e) => {
                eprintln!("peregrine: {e}");
                std::process::exit(1);
            }
        },
        Cmd::Bench(args) => {
            if let Err(e) = peregrine::bench::run(&args) {
                eprintln!("peregrine bench: {e}");
                std::process::exit(1);
            }
        }
        Cmd::Tune(args) => {
            if let Err(e) = peregrine::tune::run(&args) {
                eprintln!("peregrine tune: {e}");
                std::process::exit(1);
            }
        }
        Cmd::Tier => print!("{}", peregrine::describe_runtime()),
    }
}
