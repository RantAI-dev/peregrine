// the cost of one sampling call over a 248 320-entry vocabulary.
use std::time::Instant;
fn main() {
    let n = 248320;
    let mut s = 1u64;
    let logits: Vec<f32> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 16777216.0) * 20.0 - 10.0
        })
        .collect();
    let mut sc = peregrine_core::sample::Scratch::default();
    for (name, t, p, k) in [
        ("argmax (t=0)", 0.0f32, 1.0f32, 0usize),
        ("t=0.7 top_p=0.8", 0.7, 0.8, 0),
        ("t=0.7 top_p=0.8 top_k=20", 0.7, 0.8, 20),
    ] {
        let reps = 200;
        let t0 = Instant::now();
        let mut acc = 0u64;
        for r in 0..reps {
            acc += peregrine_core::sample::sample_filtered(&logits, t, p, k, 0.0, r as u64, &mut sc)
                as u64;
        }
        println!(
            "{name:28} {:.3} ms per call ({acc})",
            t0.elapsed().as_secs_f64() * 1e3 / reps as f64
        );
    }
}
