#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
// the PQ2_0 kernel across the pool: one 12288×4096 matrix (13.4 MB) per call, chunked rows.
use peregrine_core::gguf::Format;
use peregrine_core::pool::Pool;
use peregrine_core::quant::{f32_to_f16, row_bytes, QuantAct};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
fn main() {
    let rows: usize = std::env::var("ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12288);
    let n = 4096usize;
    let rb = row_bytes(Format::PQ2_0, n);
    // 8 distinct matrices (107 MB) so the working set is out of cache, like a real decode step.
    let nm: usize = (8 * 12288 / rows).max(1);
    let mats: Vec<Vec<u8>> = (0..nm)
        .map(|m| {
            let mut w = vec![0u8; rows * rb];
            let mut s = 1u64 + m as u64;
            for (i, b) in w.iter_mut().enumerate() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                *b = if i % 34 < 2 { 0 } else { (s & 0x55) as u8 };
            }
            for blk in w.chunks_exact_mut(34) {
                blk[..2].copy_from_slice(&f32_to_f16(0.02).to_le_bytes());
            }
            w
        })
        .collect();
    let x: Vec<f32> = (0..n).map(|i| (i % 17) as f32 - 8.0).collect();
    let width: usize = std::env::var("WIDTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let acts: Vec<QuantAct> = (0..width)
        .map(|_| {
            let mut a = QuantAct::with_capacity(n);
            a.ternary = true;
            a.fill(&x);
            a
        })
        .collect();
    let ths: Vec<usize> = std::env::var("THS")
        .ok()
        .map(|v| v.split(",").map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![1, 2, 4, 8, 12, 16]);
    for th in ths {
        let pool = Pool::new(th);
        let out: Vec<f32> = vec![0.0; rows * width];
        let op = out.as_ptr() as usize;
        let mut best = f64::MAX;
        for _ in 0..12 {
            let t = Instant::now();
            for w in &mats {
                let next = AtomicUsize::new(0);
                let chunk = (rows / (th * 8)).clamp(8, 256);
                pool.run(|_, _| loop {
                    let lo = next.fetch_add(chunk, Ordering::Relaxed);
                    if lo >= rows {
                        break;
                    }
                    let hi = (lo + chunk).min(rows);
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(
                            (op as *mut f32).add(lo * width),
                            (hi - lo) * width,
                        )
                    };
                    unsafe {
                        peregrine_core::tern::dot_rows_pq2(
                            &w[lo * rb..hi * rb],
                            rb,
                            hi - lo,
                            &acts,
                            o,
                        )
                    };
                });
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "threads {th:2}: {:.2} GB/s, {:.0} GMAC/s",
            (nm * rows * rb) as f64 / best / 1e9,
            (nm * rows * n * width) as f64 / best / 1e9
        );
    }
}
