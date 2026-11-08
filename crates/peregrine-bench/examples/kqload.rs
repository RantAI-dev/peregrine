//! Load a model file and report the load time and the resident set size.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example kqload -- <model.gguf>
//! ```
//!
//! This is the measurement S3.4 PR 1 asks for: what the k-quant repack costs at
//! load, and what it costs in resident memory. Run it twice, with
//! `PEREGRINE_KREPACK=0` and with `PEREGRINE_KREPACK=1`, on a file whose pages are
//! already in the page cache (run it once first, to warm it).
//!
//! The bench crate is built without the format features, so a quantised model needs
//! them on the command line:
//!
//! ```text
//! cargo run --release -p peregrine-bench --example kqload -- \
//!     --features peregrine-core/fmt-kquant,peregrine-models/fmt-kquant,peregrine-models/arch-llama -- \
//!     model.gguf
//! ```
//!
//! The resident set is read from `/proc/self/status`, the same file the engine's
//! telemetry reads. It is reported twice: after the load, and after a prefill that
//! has faulted in every weight — the second number is the one that matters, since a
//! model that has not run yet has none of its weights resident.

use peregrine_core::gguf::Gguf;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::StepStats;
use peregrine_core::pool::Pool;
use std::time::Instant;

/// The resident set size in kibibytes, from `/proc/self/status`.
fn vm_rss_kib() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        }
    }
    0
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: kqload <model.gguf>").clone();
    let rss0 = vm_rss_kib();
    let t = Instant::now();
    let model = peregrine_models::load(Gguf::open(std::path::Path::new(&path)).unwrap())
        .expect("the model loads");
    let dt = t.elapsed();
    let rss1 = vm_rss_kib();
    let store = model.kq().len();

    // A prefill that touches every weight: the resident set afterwards is what the
    // engine holds while it runs.
    let threads = 4;
    let pool = Pool::new(threads);
    let tokens: Vec<u32> = (1..=32).collect();
    let mut st = model.new_state(64, tokens.len(), 512, threads);
    let mut kv = model.new_cache(KvPrec::F16, 512);
    let mut ss = StepStats::default();
    let want: Vec<usize> = (0..tokens.len()).collect();
    let t = Instant::now();
    model
        .forward(&mut st, &mut kv, &pool, &tokens, 0, &want, &mut ss)
        .expect("the prefill runs");
    let prefill = t.elapsed();
    let rss2 = vm_rss_kib();

    println!(
        "load {:.3} s   RSS {} MiB -> {} MiB   after a prefill ({:.3} s): {} MiB   kq store {} MiB",
        dt.as_secs_f64(),
        rss0 / 1024,
        rss1 / 1024,
        prefill.as_secs_f64(),
        rss2 / 1024,
        store / (1024 * 1024),
    );
    println!(
        "options: {}",
        peregrine_core::options::get()
            .non_default()
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
}
