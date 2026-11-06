#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
// the ternary kernel alone: one thread over a 12288×4096 PQ2_0 matrix, for widths 1..16.
use peregrine_core::gguf::Format;
use peregrine_core::quant::{dot_batch, f32_to_f16, row_bytes, QuantAct};
use std::time::Instant;
fn main() {
    let rows: usize = std::env::var("ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12288);
    let n = 4096usize;
    let reps: usize = (12288 / rows).max(1);
    let rb = row_bytes(Format::PQ2_0, n);
    let mut w = vec![0u8; rows * rb];
    let mut s = 1u64;
    for (i, b) in w.iter_mut().enumerate() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = if i % 34 < 2 { 0 } else { (s & 0x55) as u8 };
    }
    for blk in w.chunks_exact_mut(34) {
        blk[..2].copy_from_slice(&f32_to_f16(0.02).to_le_bytes());
    }
    for width in [1usize, 2, 4, 8, 16] {
        let acts: Vec<QuantAct> = (0..width)
            .map(|k| {
                let x: Vec<f32> = (0..n).map(|i| ((i * 31 + k) % 17) as f32 - 8.0).collect();
                let mut a = QuantAct::with_capacity(n);
                a.ternary = true;
                a.fill(&x);
                a
            })
            .collect();
        let mut out = vec![0.0f32; width];
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..reps {
                for r in 0..rows {
                    dot_batch(Format::PQ2_0, &w[r * rb..(r + 1) * rb], &acts, &mut out);
                }
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "width {width:2}: {:.2} ms, {:.2} GB/s weights, {:.1} G ternary-MAC/s",
            best * 1e3,
            (reps * rows * rb) as f64 / best / 1e9,
            (reps * rows * n * width) as f64 / best / 1e9
        );
    }
}
