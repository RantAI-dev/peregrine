//! A controlled CPU disturbance for the S5.6 contended-host measurements.
//!
//! Runs `N` busy threads (default 1) until the process is killed. The caller pins the
//! process with `taskset` to the same CPUs the engine under test is pinned to, so the
//! busy threads compete for exactly those cores and for nothing else. Stop it by PID
//! (`kill <pid>`), never with a pattern match.
//!
//! ```text
//! taskset -c 5-9,15-19 target/release/examples/loadgen 4 &
//! pid=$!
//! ...measure...
//! kill "$pid"; wait "$pid" 2>/dev/null
//! ```
//!
//! The work is integer XOR and multiply on a value `black_box` cannot optimise away;
//! it saturates a core without touching memory (so it does not evict the engine's
//! weights from the last-level cache more than the scheduler's context switches do).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::with_capacity(n);
    for t in 0..n {
        let stop = Arc::clone(&stop);
        handles.push(std::thread::spawn(move || {
            let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ (t as u64);
            while !stop.load(Ordering::Relaxed) {
                for _ in 0..64 {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                    x ^= x >> 29;
                }
                std::hint::black_box(x);
            }
        }));
    }
    // The process lives until it is killed. A signal ends it; the threads are joined
    // only for a clean exit if that ever happens.
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
