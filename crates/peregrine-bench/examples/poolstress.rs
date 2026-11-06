// the pool's park/wake protocol under stress: every job must run on every worker exactly once.
use peregrine_core::pool::Pool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
fn main() {
    let th = 16;
    let pool = Pool::new(th);
    let count = AtomicU64::new(0);
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(30000);
    let t = Instant::now();
    for i in 0..n {
        pool.run(|_, _| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        if i % 5000 == 0 {
            eprintln!("{i} jobs, {:.1}s", t.elapsed().as_secs_f64());
        }
    }
    let got = count.load(Ordering::Relaxed);
    println!(
        "{n} jobs x {th} workers: counted {got} (expected {}), {:.1}s",
        n * th as u64,
        t.elapsed().as_secs_f64()
    );
    assert_eq!(got, n * th as u64);
}
