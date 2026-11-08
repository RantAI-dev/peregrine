// The workspace lints with -D warnings; these idioms are kept as they are.
#![allow(clippy::needless_range_loop)]
//! Acceptance criterion A4: the AVX2 kernels agree with the scalar twins.
//!
//! Checked on the *real* weight file, not on synthetic bytes. Random bytes exercise
//! the arithmetic but not the distribution: real quantised weights cluster, real
//! block scales span a narrow range, and a bug in the high-bit expansion of Q5_0 or
//! the 2-bit gather of Q6_K can easily be invisible on uniform noise while being
//! systematic on trained weights.
//!
//! The scalar path is the specification. When these disagree, the SIMD kernel is
//! wrong by definition.

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::quant::{dot, dot_scalar, row_bytes, QuantAct};
use std::path::PathBuf;

fn model_path() -> PathBuf {
    std::env::var_os("PEREGRINE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join("models/SmolLM2-360M-Instruct.Q4_K_M.gguf")
        })
}

/// Deterministic activations with a realistic spread, including values that push the
/// int8 quantiser to its clamp.
fn activations(n: usize, seed: u64) -> QuantAct {
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

#[test]
fn simd_and_scalar_agree_on_every_row_of_real_tensors() {
    let path = model_path();
    if !path.exists() {
        eprintln!("\n  SKIPPED: A4 (SIMD ≡ scalar) needs the model at {path:?}\n");
        return;
    }
    if !peregrine_core::simd_name().starts_with("avx2") {
        eprintln!("  SKIPPED: no AVX2 on this machine, both paths are the scalar one");
        return;
    }
    let g = Gguf::open(&path).expect("open model");

    // One tensor per quantised format, chosen so all four kernels are exercised.
    let names = [
        "token_embd.weight",      // Q8_0
        "blk.0.attn_q.weight",    // Q5_0
        "blk.0.ffn_down.weight",  // Q6_K
        "blk.10.ffn_down.weight", // Q4_K
    ];
    let mut seen: Vec<Format> = Vec::new();
    let mut rows_checked = 0usize;

    for name in names {
        let t = g.tensor(name).expect(name);
        let n = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        let rb = row_bytes(t.kind, n);
        let bytes = g.bytes(t);
        let a = activations(n, 0xC0FF_EE00 + n as u64);

        // Every row would be slow on a 49k-row embedding table; take a stride that
        // still covers thousands of rows and both ends of the tensor.
        let step = (rows / 512).max(1);
        let kq = matches!(t.kind, Format::Q4_K | Format::Q6_K);
        let mut deq = vec![0.0f32; n];
        for r in (0..rows).step_by(step) {
            let row = &bytes[r * rb..(r + 1) * rb];
            let want = dot_scalar(t.kind, row, &a);
            let got = dot(t.kind, row, &a);
            // Both accumulate the same integer products and differ only in the
            // order of the float accumulation — but a row whose terms nearly
            // cancel can leave a small result with an error that is a few ulps
            // of the *terms*. So each path is held to the exact (f64) value of
            // the decoded dot product within a few ulps of the terms' magnitude,
            // which catches a wrong lane (an error of a few percent of that
            // magnitude) and tolerates summation order.
            peregrine_core::quant::dequant_row(t.kind, row, &mut deq);
            let (mut exact, mut mag) = (0.0f64, 0.0f64);
            for i in 0..n {
                let act = if kq {
                    a.qk[i] as f64 * a.dk[i / 256] as f64
                } else {
                    a.q[i] as f64 * a.d[i / 32] as f64
                };
                let term = deq[i] as f64 * act;
                exact += term;
                mag += term.abs();
            }
            let tol = 4e-6 * mag + 1e-7;
            assert!(
                (got as f64 - exact).abs() <= tol && (want as f64 - exact).abs() <= tol,
                "{name} [{}] row {r}: simd {got} vs scalar {want} vs exact {exact} (|terms| {mag})",
                t.kind.label()
            );
            rows_checked += 1;
        }
        if !seen.contains(&t.kind) {
            seen.push(t.kind);
        }
    }
    assert_eq!(
        seen.len(),
        4,
        "not every quantised format was exercised: {seen:?}"
    );
    eprintln!(
        "A4: {rows_checked} rows agree across {} formats",
        seen.len()
    );
}

/// A dot product of all-zero activations must be exactly zero in both paths — the
/// bias corrections have to cancel rather than leave a residue proportional to the
/// block scale.
#[test]
fn both_paths_are_exactly_zero_on_a_zero_activation() {
    let path = model_path();
    if !path.exists() {
        eprintln!("  SKIPPED: needs the model");
        return;
    }
    let g = Gguf::open(&path).expect("open model");
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
        "blk.10.ffn_down.weight",
    ] {
        let t = g.tensor(name).expect(name);
        let n = t.dims[0] as usize;
        let mut a = QuantAct::with_capacity(n);
        a.fill(&vec![0.0f32; n]);
        let row = &g.bytes(t)[..row_bytes(t.kind, n)];
        assert_eq!(dot_scalar(t.kind, row, &a), 0.0, "{name} scalar");
        assert_eq!(dot(t.kind, row, &a), 0.0, "{name} simd");
    }
}

// ---------------------------------------------------------------------------
// The batched kernels. `dot_batch` decodes a weight row once per tile of four
// activations; it must agree with the per-activation scalar specification for every
// batch width, including the 1/2/3 remainders of the tile loop.
// ---------------------------------------------------------------------------

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

/// Raw random bytes hit half-precision Inf/NaN about one block in 128, and a NaN scale
/// makes both sides NaN, so the comparison would pass or fail for no reason that has
/// to do with the kernel. Clear the exponent's top bits: a normal half in [0.125, 0.25).
fn sane(fmt: Format, w: &mut [u8]) {
    let fix = |b: &mut [u8], at: usize| b[at + 1] = (b[at + 1] & 0x0F) | 0x30;
    for blk in w.chunks_exact_mut(fmt.block_bytes()) {
        match fmt {
            Format::Q8_0 | Format::Q5_0 | Format::Q4_0 | Format::IQ4_NL => fix(blk, 0),
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

#[test]
fn batched_kernels_match_the_scalar_specification_at_every_batch_width() {
    use peregrine_core::quant::dot_batch;
    for (fmt, n) in [
        (Format::Q8_0, 960),
        (Format::Q4_0, 960),
        (Format::Q5_0, 960),
        (Format::IQ4_NL, 960),
        (Format::Q2_K, 2560),
        (Format::Q3_K, 2560),
        (Format::Q4_K, 2560),
        (Format::Q5_K, 2560),
        (Format::Q6_K, 2560),
        (Format::IQ4_XS, 2560),
        (Format::MXFP4, 960),
    ] {
        let mut w = prng_bytes(0xBEEF_0000 + n as u64, row_bytes(fmt, n));
        sane(fmt, &mut w);
        // 1..=9 covers the empty remainder, each of 1/2/3, and two full tiles plus one.
        for nb in 1..=9usize {
            let acts: Vec<QuantAct> = (0..nb)
                .map(|i| activations(n, 0x1000 + i as u64 * 7919))
                .collect();
            let mut got = vec![0.0f32; nb];
            dot_batch(fmt, &w, &acts, &mut got);
            for (b, a) in acts.iter().enumerate() {
                let want = dot_scalar(fmt, &w, a);
                // Same integer products; only the float accumulation order and the
                // bias being subtracted at the end rather than per block differ.
                let tol = 2e-4 * want.abs().max(1.0);
                assert!(
                    (got[b] - want).abs() <= tol,
                    "{} nb={nb} act {b}: batched {} vs scalar {want}",
                    fmt.label(),
                    got[b]
                );
            }
        }
    }
}

#[test]
fn batched_kernels_agree_on_rows_of_the_real_model() {
    use peregrine_core::quant::dot_batch;
    let path = model_path();
    if !path.exists() {
        eprintln!("  SKIPPED: needs the model at {path:?}");
        return;
    }
    let g = Gguf::open(&path).expect("open model");
    let mut checked = 0;
    for name in [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
        "blk.10.ffn_down.weight",
    ] {
        let t = g.tensor(name).expect(name);
        let n = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        let rb = row_bytes(t.kind, n);
        let bytes = g.bytes(t);
        let acts: Vec<QuantAct> = (0..6).map(|i| activations(n, 0xAB00 + i)).collect();
        let step = (rows / 128).max(1);
        for r in (0..rows).step_by(step) {
            let row = &bytes[r * rb..(r + 1) * rb];
            let mut got = vec![0.0f32; acts.len()];
            dot_batch(t.kind, row, &acts, &mut got);
            for (b, a) in acts.iter().enumerate() {
                let want = dot_scalar(t.kind, row, a);
                // Relative, plus a small absolute floor: a near-cancelling dot product
                // (a result of 0.02 built from partial sums in the tens) carries float
                // rounding from the *order* of accumulation, which is a property of the
                // sum and not of the kernel. A wrong nibble order or scale is off by
                // 0.1 or more, five orders above this floor.
                let tol = 2e-4 * want.abs() + 2e-5;
                assert!(
                    (got[b] - want).abs() <= tol,
                    "{name} row {r} act {b}: {} vs {want}",
                    got[b]
                );
                checked += 1;
            }
        }
    }
    eprintln!("batched ≡ scalar on {checked} (row, activation) pairs of real weights");
}

/// Total-order key of a finite float, so `ulps` counts representable values between
/// two numbers regardless of sign.
#[cfg(target_arch = "x86_64")]
fn f32_ord(x: f32) -> u32 {
    let u = x.to_bits();
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

#[cfg(target_arch = "x86_64")]
fn ulps(a: f32, b: f32) -> u32 {
    f32_ord(a).abs_diff(f32_ord(b))
}

/// The wide Q5_0 kernel has to leave a row's value untouched by the batch it ran
/// in, and the narrow (decode) path must agree with it bit for bit. Q8_0 is the
/// control: it still takes the tiled kernel, and must stay batch-invariant too.
///
/// The canonical call runs the largest width (twenty activations: one sixteen-wide
/// window plus a four-activation tail), and every narrower call — including widths
/// 1, 2 and 3 that take the tiled kernel — must reproduce the same bits for the
/// rows it holds. Passes that start part-way through the batch stand in for a
/// worker that claimed a range of rows, and must agree too. Finally each result is
/// checked against the scalar twin of the kernel's fold within one ULP: the
/// kernels reorder the float sum, but not by more than the last bit of the result.
#[test]
#[cfg(target_arch = "x86_64")]
fn q5_0_wide_windows_are_invariant_and_match_the_scalar_fold() {
    use peregrine_core::quant::dot_batch;
    if !peregrine_core::simd::have_batched() {
        eprintln!("  SKIPPED: needs AVX2 + FMA + F16C, both paths are the scalar one");
        return;
    }
    // 896 is the Qwen2.5-0.5B width: twenty-eight 32-value blocks and not a multiple
    // of 256, the shape that puts this model on the legacy formats.
    const N: usize = 896;
    for fmt in [Format::Q5_0, Format::Q8_0] {
        let mut w = prng_bytes(0x5A17_0000 + fmt.block_bytes() as u64, row_bytes(fmt, N));
        sane(fmt, &mut w);
        const ROWS: usize = 20;
        let acts: Vec<QuantAct> = (0..ROWS)
            .map(|i| activations(N, 0xFACE_0000 + i as u64 * 7919))
            .collect();

        // The widest call is the canonical value of every row.
        let mut full = vec![0.0f32; ROWS];
        dot_batch(fmt, &w, &acts, &mut full);

        // Every narrower width, from one activation (the decode path) through the
        // five-group tail, must give the same bits for the rows it holds.
        for nb in 1..ROWS {
            let mut got = vec![0.0f32; nb];
            dot_batch(fmt, &w, &acts[..nb], &mut got);
            for b in 0..nb {
                assert_eq!(
                    got[b].to_bits(),
                    full[b].to_bits(),
                    "{} width {nb} disagrees with width {ROWS} on activation {b}",
                    fmt.label()
                );
            }
        }

        // A pass that starts at an offset — what a worker takes when it claims a
        // range of rows — must agree with the same rows in the whole call.
        for (start, nb) in [(13usize, 4usize), (9, 7), (16, 3), (6, 2), (0, 1)] {
            let mut part = vec![0.0f32; nb];
            dot_batch(fmt, &w, &acts[start..start + nb], &mut part);
            for b in 0..nb {
                assert_eq!(
                    part[b].to_bits(),
                    full[start + b].to_bits(),
                    "{} pass {start}..{} disagrees on activation {}",
                    fmt.label(),
                    start + nb,
                    start + b
                );
            }
        }

        // The scalar twin of the kernel's fixed fold, within one ULP. The format's
        // own scalar specification (`dot_scalar`) reduces all 32 products of a block
        // in integer and rounds once; the kernels reduce four products per lane and
        // fold the eight lanes in float, which the batched-kernel test above holds
        // to a relative tolerance. Here the reference is the scalar version of the
        // *kernel's* fold, so a divergence is a rounding bug rather than a
        // different summation order.
        for (b, a) in acts.iter().enumerate() {
            let want = fold_reference(fmt, &w, a);
            assert!(
                ulps(full[b], want) <= 1,
                "{} activation {b}: {} against the scalar fold {want} ({} ulps)",
                fmt.label(),
                full[b],
                ulps(full[b], want)
            );
        }
    }
}

/// The row's codes as `dot_scalar` reads them: 32 per block, `[0,31]` for Q5_0 and
/// signed for Q8_0.
#[cfg(target_arch = "x86_64")]
fn block_codes(fmt: Format, blk: &[u8], out: &mut [i32; 32]) {
    match fmt {
        Format::Q5_0 => {
            let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
            let qs = &blk[6..22];
            for j in 0..16 {
                let hi0 = ((qh >> j) << 4) as u8 & 0x10;
                let hi1 = (qh >> (j + 12)) as u8 & 0x10;
                out[j] = ((qs[j] & 0x0F) | hi0) as i32;
                out[j + 16] = ((qs[j] >> 4) | hi1) as i32;
            }
        }
        Format::Q8_0 => {
            for j in 0..32 {
                out[j] = blk[2 + j] as i8 as i32;
            }
        }
        other => panic!("no block decoder for {other:?}"),
    }
}

/// The scalar twin of `wide_q5_0` / `wide_q8_0`: four contiguous products per
/// integer lane, each lane converted to `f32` and fused into its own accumulator
/// once per block, then the eight lanes reduced in `hsum_ps`'s order.
#[cfg(target_arch = "x86_64")]
fn fold_reference(fmt: Format, w: &[u8], a: &QuantAct) -> f32 {
    let bb = fmt.block_bytes();
    let mut acc = [0.0f32; 8];
    let mut codes = [0i32; 32];
    for (b, blk) in w.chunks_exact(bb).enumerate() {
        block_codes(fmt, blk, &mut codes);
        let h = u16::from_le_bytes([blk[0], blk[1]]);
        let d = peregrine_core::format::f16(h) * a.d[b];
        for l in 0..8 {
            let mut lane = 0i32;
            for k in 0..4 {
                let i = 4 * l + k;
                lane += codes[i] * a.q[b * 32 + i] as i32;
            }
            if fmt == Format::Q5_0 {
                lane -= a.lane_sums[b * 8 + l] << 4;
            }
            acc[l] = (lane as f32).mul_add(d, acc[l]);
        }
    }
    let s0 = acc[0] + acc[4];
    let s1 = acc[1] + acc[5];
    let s2 = acc[2] + acc[6];
    let s3 = acc[3] + acc[7];
    (s0 + s2) + (s1 + s3)
}
