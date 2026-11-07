//! Micro-benchmark: the GeGLU kernel, `vmath::geglu`.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example geglu_bench -- [n] [reps]
//! ```
//!
//! `n` is the width of one feed-forward row (elements), `reps` the number of passes over
//! it. The kernel is the one that ships: on a CPU with the FMA instruction it runs the
//! `#[target_feature(enable = "fma")]` path, where the `mul_add`s of `tanh` are the
//! hardware's `vfmadd`; on any other CPU it is the portable path, where each of them is
//! a call to the C library's `fmaf`. The two produce the same bits — this measures the
//! difference in speed.

use peregrine_core::vmath;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);

    // A row of activations with the range a feed-forward produces: mostly small, with
    // the tails that send `tanh` down its exponential branch.
    let mut s = 0x2545_f491_4f6c_dd1du64 | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let gate: Vec<f32> = (0..n)
        .map(|_| (next() >> 40) as f32 / (1u64 << 24) as f32 * 12.0 - 6.0)
        .collect();
    let up: Vec<f32> = (0..n)
        .map(|_| (next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)
        .collect();

    let mut g = gate.clone();
    // Warm the path up (and the branch predictor).
    for _ in 0..100 {
        vmath::geglu(&mut g, &up);
    }
    std::hint::black_box(&g);

    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            vmath::geglu(&mut g, &up);
            std::hint::black_box(&g);
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (reps * n) as f64;
        best = best.min(ns);
    }
    let mut h = 0u64;
    for v in &g {
        h = h.wrapping_mul(0x100_0000_01b3) ^ u64::from(v.to_bits());
    }
    println!("tier        {}", peregrine_core::cpu::tier_name());
    println!("fma         {}", peregrine_core::cpu::have_fma());
    println!("n           {n}");
    println!("reps        {reps}");
    println!("ns/element  {best:.3}");
    println!("checksum    {h:016x}");
}
