//! Kernel micro-benchmark: one thread, synthetic rows, every quantised format.
//!
//! ```text
//! cargo run --release -p peregrine-bench --example kbench -- [k] [rows] [reps]
//! ```
//!
//! Reports ns per row and the effective weight bandwidth for `dot_batch` with one
//! activation (the decode path), with four (the prefill tile) and with 32 (a prompt:
//! the wide kernels; `PEREGRINE_NO_WIDE=1` measures the tiled kernels at 32 instead). A kernel that
//! cannot sustain a few GB/s on one core is compute-bound before the memory system
//! enters the picture — which is the question this answers.

use peregrine_core::gguf::Format;
use peregrine_core::quant::{dot_batch, row_bytes, QuantAct};
use std::time::Instant;

fn prng_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

fn sane(fmt: Format, w: &mut [u8]) {
    let fix = |b: &mut [u8], at: usize| b[at + 1] = (b[at + 1] & 0x0F) | 0x30;
    for blk in w.chunks_exact_mut(fmt.block_bytes()) {
        match fmt {
            Format::Q8_0 | Format::Q5_0 | Format::IQ4_NL => fix(blk, 0),
            Format::Q2_K => {
                fix(blk, 80);
                fix(blk, 82);
            }
            Format::Q3_K => fix(blk, 108),
            Format::Q4_K | Format::Q5_K => {
                fix(blk, 0);
                fix(blk, 2);
            }
            Format::Q6_K => fix(blk, 208),
            Format::IQ4_XS => fix(blk, 0),
            Format::MXFP4 => blk[0] = (blk[0] % 30) + 115,
            _ => {}
        }
    }
}

fn act(n: usize, seed: u64) -> QuantAct {
    let mut s = seed | 1;
    let x: Vec<f32> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 32) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * 3.0
        })
        .collect();
    let mut a = QuantAct::with_capacity(n);
    a.fill(&x);
    a
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let k: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let rows: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    println!(
        "{:<6} {:>6} {:>12} {:>10} {:>12} {:>10} {:>12} {:>12}",
        "format", "acts", "ns/row", "GB/s", "ns/row×4", "GB/s×4", "ns/row×32", "ns/act×32"
    );
    // The decode-only eight-row kernels, against the one-row columns below.
    let mk = peregrine_core::table::MatKernels::get();
    for fmt in [Format::Q5_0, Format::Q8_0] {
        if !mk.rows8(fmt) {
            continue;
        }
        let rb = row_bytes(fmt, k);
        let n8 = rows / 8 * 8;
        let mut w = prng_bytes(7, rb * n8);
        sane(fmt, &mut w);
        let a = act(k, 11);
        let mut out8 = [0.0f32; 8];
        let mut sink = 0.0f32;
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let mut r = 0;
            while r + 8 <= n8 {
                mk.row8(fmt, &w[r * rb..(r + 8) * rb], rb, &a, &mut out8);
                sink += out8[0];
                r += 8;
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{fmt:?} rows8 (1 act) {:>8.0} ns/row {:>8.2} GB/s",
            best / n8 as f64 * 1e9,
            (rb * n8) as f64 / best / 1e9
        );
        if sink.is_nan() {
            println!("nan");
        }
    }

    for fmt in [
        Format::Q8_0,
        Format::Q5_0,
        Format::Q2_K,
        Format::Q3_K,
        Format::Q4_K,
        Format::Q5_K,
        Format::Q6_K,
        Format::IQ4_NL,
        Format::IQ4_XS,
        Format::MXFP4,
    ] {
        let rb = row_bytes(fmt, k);
        let mut w = prng_bytes(7, rb * rows);
        sane(fmt, &mut w);
        let acts: Vec<QuantAct> = (0..32).map(|i| act(k, 11 + i as u64)).collect();
        let mut out = vec![0.0f32; 32];
        let mut sink = 0.0f32;
        let mut best1 = f64::INFINITY;
        let mut best4 = f64::INFINITY;
        let mut best32 = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            for r in 0..rows {
                dot_batch(fmt, &w[r * rb..(r + 1) * rb], &acts[..1], &mut out);
                sink += out[0];
            }
            best1 = best1.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            for r in 0..rows {
                dot_batch(fmt, &w[r * rb..(r + 1) * rb], &acts[..4], &mut out);
                sink += out[3];
            }
            best4 = best4.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            for r in 0..rows {
                dot_batch(fmt, &w[r * rb..(r + 1) * rb], &acts, &mut out);
                sink += out[31];
            }
            best32 = best32.min(t.elapsed().as_secs_f64());
        }
        let bytes = (rb * rows) as f64;
        println!(
            "{:<6} {:>6} {:>12.0} {:>10.2} {:>12.0} {:>10.2} {:>12.0} {:>12.1}",
            format!("{fmt:?}"),
            1,
            best1 / rows as f64 * 1e9,
            bytes / best1 / 1e9,
            best4 / rows as f64 * 1e9,
            bytes / best4 / 1e9,
            best32 / rows as f64 * 1e9,
            best32 / rows as f64 * 1e9 / 32.0
        );
        if sink.is_nan() {
            println!("nan");
        }
    }
}
