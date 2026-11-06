// the pieces of one 4096-wide rotation + ternary fill.
use peregrine_core::quant::QuantAct;
use std::time::Instant;
fn main() {
    let n = 4096;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
        .collect();
    let signs: Vec<f32> = (0..n)
        .map(|i| if i % 3 == 0 { -0.015625 } else { 0.015625 })
        .collect();
    let mut r = x.clone();
    let mut a = QuantAct::with_capacity(n);
    a.ternary = true;
    let reps = 20000;
    let t = Instant::now();
    for _ in 0..reps {
        for ((v, &xi), s) in r.iter_mut().zip(&x).zip(&signs) {
            *v = xi * s;
        }
        std::hint::black_box(&r);
    }
    let t_sign = t.elapsed().as_secs_f64() / reps as f64;
    let t = Instant::now();
    for _ in 0..reps {
        peregrine_core::hadamard::fwht(&mut r);
        std::hint::black_box(&r);
    }
    let t_fwht = t.elapsed().as_secs_f64() / reps as f64;
    let t = Instant::now();
    for _ in 0..reps {
        a.fill(&x);
        std::hint::black_box(&a);
    }
    let t_fill = t.elapsed().as_secs_f64() / reps as f64;
    println!(
        "signs {:.2} us, fwht {:.2} us, fill {:.2} us",
        t_sign * 1e6,
        t_fwht * 1e6,
        t_fill * 1e6
    );
}
