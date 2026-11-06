// the cost of one fork–join on this machine.
use peregrine_core::pool::Pool;
use std::time::Instant;
fn main() {
    for th in [2usize, 4, 8, 12, 16] {
        let pool = Pool::new(th);
        let n = 20000;
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..n {
                pool.run(|_, _| {
                    std::hint::black_box(0);
                });
            }
            best = best.min(t.elapsed().as_secs_f64() / n as f64);
        }
        // and with ~20 µs of work per worker
        let t = Instant::now();
        for _ in 0..2000 {
            pool.run(|_, _| {
                let mut x = 0u64;
                for i in 0..20000u64 {
                    x = x.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(x);
            });
        }
        let w = t.elapsed().as_secs_f64() / 2000.0;
        println!(
            "threads {th:2}: empty fork-join {:.2} us, with ~work {:.1} us",
            best * 1e6,
            w * 1e6
        );
    }
}
