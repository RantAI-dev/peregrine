#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
// the PTQ1_0 kernel alone (PTQ1 activation layout), one thread, 5120-wide rows.
#[cfg(target_arch = "x86_64")]
use peregrine_core::gguf::Format;
#[cfg(target_arch = "x86_64")]
use peregrine_core::quant::{dot_batch, f32_to_f16, row_bytes, QuantAct};
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("tbench1 benchmarks the x86-only PTQ1_0 kernel; nothing to run on this architecture");
}

#[cfg(target_arch = "x86_64")]
fn main() {
    let (rows, n) = (4096usize, 5120usize);
    let rb = row_bytes(Format::PTQ1_0, n);
    let mut w = vec![0u8; rows * rb];
    let mut s = 7u64;
    for (i, b) in w.iter_mut().enumerate() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = (s % 243) as u8;
        let _ = i;
    }
    for blk in w.chunks_exact_mut(28) {
        blk[26..28].copy_from_slice(&f32_to_f16(0.02).to_le_bytes());
    }
    for width in [1usize, 4, 8, 16] {
        let acts: Vec<QuantAct> = (0..width)
            .map(|k| {
                let x: Vec<f32> = (0..n).map(|i| ((i * 31 + k) % 17) as f32 - 8.0).collect();
                let mut a = QuantAct::with_capacity(n);
                a.ternary = true;
                a.ptq1 = true;
                a.fill(&x);
                a
            })
            .collect();
        let mut out = vec![0.0f32; width * rows];
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            unsafe { peregrine_core::tern::dot_rows_ptq1(&w, rb, rows, &acts, &mut out) };
            best = best.min(t.elapsed().as_secs_f64());
        }
        let _ = dot_batch;
        println!(
            "width {width:2}: {:.2} GB/s, {:.1} GMAC/s",
            (rows * rb) as f64 / best / 1e9,
            (rows * n * width) as f64 / best / 1e9
        );
    }
}
