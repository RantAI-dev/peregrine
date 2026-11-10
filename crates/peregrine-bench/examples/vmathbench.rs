//! aarch64 `vmath` and activation-quantisation micro-benchmark (S2.8 F5/F6).
//!
//! ```text
//! cargo run --release -p peregrine-bench --example vmathbench -- [n] [reps]
//! ```
//!
//! One thread, L1/L2-resident arrays. Run it once with the native tier and once with
//! `PEREGRINE_ISA=scalar`: the scalar cap turns every NEON branch off, so the pair of
//! runs is the vector kernel beside the scalar loop it replaces.

use peregrine_kernels::quant::QuantAct;
use peregrine_kernels::vmath;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let n = args.first().copied().unwrap_or(4096).max(4);
    let reps = args.get(1).copied().unwrap_or(200).max(1);
    let mut s = 0x243f_6a88_85a3_08d3u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 24) as u8
    };
    let x: Vec<f32> = (0..n)
        .map(|_| (next() as f32 / 255.0) * 6.0 - 3.0)
        .collect();
    let u: Vec<f32> = (0..n)
        .map(|_| (next() as f32 / 255.0) * 2.0 - 1.0)
        .collect();
    let mut buf = x.clone();
    let mut act = QuantAct::with_capacity(n);

    let bench = |name: &str, bytes: usize, f: &mut dyn FnMut()| {
        f();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let ns = t.elapsed().as_nanos() as f64 / reps as f64;
        println!(
            "{name:<26} {:>10.1} ns  {:>7.1} GB/s",
            ns,
            bytes as f64 / ns
        );
    };

    println!(
        "n={n} reps={reps} tier={} isa={:?}",
        peregrine_kernels::cpu::tier_name(),
        std::env::var("PEREGRINE_ISA").ok()
    );

    bench("swiglu", n * 8, &mut || {
        buf.copy_from_slice(&x);
        vmath::swiglu(&mut buf, &u);
        black_box(buf[0]);
    });
    bench("silu", n * 4, &mut || {
        buf.copy_from_slice(&x);
        vmath::silu(&mut buf);
        black_box(buf[0]);
    });
    bench("mul_silu", n * 8, &mut || {
        buf.copy_from_slice(&x);
        vmath::mul_silu(&mut buf, &u);
        black_box(buf[0]);
    });
    bench("mul_sigmoid", n * 8, &mut || {
        buf.copy_from_slice(&x);
        vmath::mul_sigmoid(&mut buf, &u);
        black_box(buf[0]);
    });
    bench("exp_shift_sum", n * 4, &mut || {
        buf.copy_from_slice(&x);
        black_box(vmath::exp_shift_sum(&mut buf, 0.25));
    });
    bench("exp_scaled_sum", n * 4, &mut || {
        buf.copy_from_slice(&x);
        black_box(vmath::exp_scaled_sum(&mut buf, 0.25, 0.8));
    });
    bench("max", n * 4, &mut || {
        black_box(vmath::max(&x));
    });
    bench("quantise fill", n * 4, &mut || {
        act.fill(&x);
        black_box(act.q[0]);
    });
}
