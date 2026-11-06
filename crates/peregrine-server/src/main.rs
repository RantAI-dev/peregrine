//! `peregrine-server`: the OpenAI-compatible server as its own binary (also `peregrine serve`).
use clap::Parser;

#[tokio::main]
async fn main() {
    peregrine_server::init_logging();
    peregrine_server::run(peregrine_server::Args::parse()).await;
}
