//! Probe machine streaming bandwidth and compute ceilings.

use peregrine_core::telem::Machine;

fn main() {
    let threads = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
    let m = Machine::probe(threads);
    println!(
        "{{\n  \"bandwidth_bytes_per_s\": {},\n  \"bandwidth_gb_s\": {:.3},\n  \"flops\": {},\n  \"gflops\": {:.3},\n  \"threads\": {},\n  \"simd\": \"{}\",\n  \"simd_tier\": \"{}\",\n  \"ridge\": {:.3}\n}}",
        m.bandwidth,
        m.bandwidth / 1e9,
        m.flops,
        m.flops / 1e9,
        m.threads,
        m.simd,
        m.simd_tier,
        m.ridge()
    );
}
