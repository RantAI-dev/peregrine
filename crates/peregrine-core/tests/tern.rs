// the ternary kernels against their scalar specification (quant::dot_scalar), on random rows.
#![cfg(target_arch = "x86_64")]
use peregrine_core::gguf::Format;
use peregrine_core::quant::{dequant_row, dot_scalar, f32_to_f16, QuantAct};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

fn pq2_row(r: &mut Rng, n: usize) -> Vec<u8> {
    let mut w = Vec::new();
    for _ in 0..n / 128 {
        w.extend_from_slice(&f32_to_f16(0.01 + 0.05 * r.f().abs()).to_le_bytes());
        for _ in 0..32 {
            let mut b = 0u8;
            for s in 0..4 {
                b |= ((r.next() % 3) as u8) << (2 * s);
            }
            w.push(b);
        }
    }
    w
}

fn ptq1_row(r: &mut Rng, n: usize) -> Vec<u8> {
    let mut w = Vec::new();
    for _ in 0..n / 128 {
        // Encode random trits the way quantize_row_ptq1_0_ref does: byte = ceil(Σ t_n 3^(4-n) · 256 / 243).
        let enc = |t: &[u8]| -> u8 {
            let mut q: u16 = 0;
            for &x in t {
                q = q * 3 + x as u16;
            }
            for _ in t.len()..5 {
                q *= 3;
            }
            (q * 256).div_ceil(243) as u8
        };
        for _ in 0..24 {
            let t: Vec<u8> = (0..5).map(|_| (r.next() % 3) as u8).collect();
            w.push(enc(&t));
        }
        for _ in 0..2 {
            let t: Vec<u8> = (0..4).map(|_| (r.next() % 3) as u8).collect();
            w.push(enc(&t));
        }
        w.extend_from_slice(&f32_to_f16(0.01 + 0.05 * r.f().abs()).to_le_bytes());
    }
    w
}

fn acts(r: &mut Rng, n: usize, k: usize) -> Vec<QuantAct> {
    (0..k)
        .map(|_| {
            let x: Vec<f32> = (0..n).map(|_| r.f() * 3.0).collect();
            let mut a = QuantAct::with_capacity(n);
            a.ternary = true;
            a.fill(&x);
            a
        })
        .collect()
}

fn check(fmt: Format, w: &[u8], a: &[QuantAct], got: &[f32], what: &str) {
    for (i, ai) in a.iter().enumerate() {
        let want = dot_scalar(fmt, w, ai);
        let tol = 1e-4 * want.abs().max(1.0);
        assert!(
            (got[i] - want).abs() <= tol,
            "{what} row {i}: {} vs {want}",
            got[i]
        );
    }
}

#[test]
fn pq2_kernels_match_scalar() {
    let mut r = Rng(0x9E3779B97F4A7C15);
    for n in [128usize, 384, 4096, 12288] {
        let w = pq2_row(&mut r, n);
        for k in 1..=17 {
            let a = acts(&mut r, n, k);
            let mut out = vec![0.0f32; k];
            peregrine_core::quant::dot_batch(Format::PQ2_0, &w, &a, &mut out);
            check(Format::PQ2_0, &w, &a, &out, "dispatch");
            unsafe { peregrine_core::tern::dot_batch_pq2_avx2(&w, &a, &mut out) };
            check(Format::PQ2_0, &w, &a, &out, "avx2");
            if peregrine_core::tern::have_vnni() {
                unsafe { peregrine_core::tern::dot_batch_pq2_vnni256(&w, &a, &mut out) };
                check(Format::PQ2_0, &w, &a, &out, "vnni256");
            }
        }
    }
}

#[test]
fn ptq1_kernel_matches_scalar_and_decode() {
    let mut r = Rng(12345);
    for n in [128usize, 4096] {
        let w = ptq1_row(&mut r, n);
        // decode gives only -d, 0, +d
        let mut y = vec![0.0f32; n];
        dequant_row(Format::PTQ1_0, &w, &mut y);
        for blk in 0..n / 128 {
            let d =
                peregrine_core::gguf::f16(u16::from_le_bytes([w[blk * 28 + 26], w[blk * 28 + 27]]));
            for v in &y[blk * 128..(blk + 1) * 128] {
                assert!(*v == 0.0 || (v.abs() - d).abs() < 1e-6, "{v} not ±{d}");
            }
        }
        for k in 1..=10 {
            let a = acts(&mut r, n, k);
            let mut out = vec![0.0f32; k];
            peregrine_core::quant::dot_batch(Format::PTQ1_0, &w, &a, &mut out);
            check(Format::PTQ1_0, &w, &a, &out, "ptq1");
        }
    }
}

#[test]
fn pq2_scalar_equals_dequantised_dot() {
    // The scalar kernel against a plain float dot of the decoded row and the decoded activation.
    let mut r = Rng(7);
    let n = 4096;
    let w = pq2_row(&mut r, n);
    let a = &acts(&mut r, n, 1)[0];
    let mut y = vec![0.0f32; n];
    dequant_row(Format::PQ2_0, &w, &mut y);
    let want: f64 = (0..n)
        .map(|i| y[i] as f64 * a.tq[i] as f64 * a.td[i / 16] as f64)
        .sum();
    let got = dot_scalar(Format::PQ2_0, &w, a) as f64;
    assert!(
        (got - want).abs() < 1e-3 * want.abs().max(1.0),
        "{got} vs {want}"
    );
}

#[test]
fn ternary_quantiser_vector_equals_scalar() {
    let mut r = Rng(99);
    let n = 4096;
    let x: Vec<f32> = (0..n)
        .map(|i| if i % 97 == 0 { 0.0 } else { r.f() * 5.0 })
        .collect();
    let mut a = QuantAct::with_capacity(n);
    a.ternary = true;
    a.fill(&x);
    let mut b = a.clone();
    b.quantize16_scalar(&x);
    assert_eq!(a.tq, b.tq);
    assert_eq!(a.td, b.td);
    assert_eq!(a.tls, b.tls);
}

#[test]
fn pq2_rows_entry_matches_per_row() {
    let mut r = Rng(4242);
    let n = 4096;
    let rows = 7;
    let w: Vec<u8> = (0..rows).flat_map(|_| pq2_row(&mut r, n)).collect();
    let rb = w.len() / rows;
    for k in [1usize, 2, 3, 9] {
        let a = acts(&mut r, n, k);
        let mut out = vec![0.0f32; rows * k];
        unsafe { peregrine_core::tern::dot_rows_pq2(&w, rb, rows, &a, &mut out) };
        for row in 0..rows {
            let wr = &w[row * rb..(row + 1) * rb];
            check(Format::PQ2_0, wr, &a, &out[row * k..(row + 1) * k], "rows");
        }
    }
}

#[test]
fn pq2_gemm_matches_gemv_bit_for_bit_at_every_width() {
    if !peregrine_core::simd::have_batched() {
        return;
    }
    let mut r = Rng(0xC0FFEE);
    // Widths that are not whole groups, row counts that are not whole 16-row tiles, and a
    // K that is not a whole K-block: every tail the prompt chunker can hand the kernel.
    for n in [384usize, 2048, 4224] {
        let rb = n / 128 * 34;
        for rows in [1usize, 2, 7, 15, 16, 17, 31] {
            let w: Vec<u8> = (0..rows).flat_map(|_| pq2_row(&mut r, n)).collect();
            for k in 1..=32usize {
                let a = acts(&mut r, n, k);
                let mut got = vec![0.0f32; rows * k];
                // SAFETY: AVX2/FMA/F16C checked above; the lengths are the kernel's.
                unsafe { peregrine_core::tern::dot_rows_pq2_gemm_avx2(&w, rb, rows, &a, &mut got) };
                for row in 0..rows {
                    for b in 0..k {
                        let mut want = [0.0f32; 1];
                        // SAFETY: as above; one row against one activation is the GEMV.
                        unsafe {
                            peregrine_core::tern::dot_batch_pq2_avx2(
                                &w[row * rb..(row + 1) * rb],
                                std::slice::from_ref(&a[b]),
                                &mut want,
                            )
                        };
                        assert_eq!(
                            got[row * k + b].to_bits(),
                            want[0].to_bits(),
                            "n={n} rows={rows} k={k} row={row} act={b}: {} vs {}",
                            got[row * k + b],
                            want[0]
                        );
                    }
                }
            }
        }
    }
}

/// The multi-row decode GEMV against the one-row kernel, bit for bit, at every row count
/// (including the ones that leave a shorter tail). A row's bits must not depend on how
/// many rows shared the pass — the batch-invariance rule the MTP verify path relies on.
#[test]
fn pq2_multirow_decode_matches_one_row_kernel_exactly() {
    if !peregrine_core::simd::have_batched() {
        return;
    }
    let mut r = Rng(0xA11CE);
    for n in [2048usize, 4096] {
        for rows in 1..=11usize {
            let w: Vec<u8> = (0..rows).flat_map(|_| pq2_row(&mut r, n)).collect();
            let rb = n / 128 * 34;
            let a = acts(&mut r, n, 1).pop().unwrap();
            let mut got = vec![0.0f32; rows];
            // SAFETY: AVX2/FMA/F16C checked above; the lengths are the kernel's.
            unsafe { peregrine_core::tern::dot_rows_pq2_avx2(&w, rb, rows, &a, &mut got) };
            for row in 0..rows {
                let mut want = [0.0f32; 1];
                // SAFETY: as above.
                unsafe {
                    peregrine_core::tern::dot_batch_pq2_avx2(
                        &w[row * rb..(row + 1) * rb],
                        std::slice::from_ref(&a),
                        &mut want,
                    )
                };
                assert_eq!(
                    got[row].to_bits(),
                    want[0].to_bits(),
                    "n={n} rows={rows} row={row}: {} vs {}",
                    got[row],
                    want[0]
                );
            }
        }
    }
}

#[test]
fn plane_permutation_vbmi_equals_scalar() {
    if !peregrine_core::tern::have_vbmi() {
        return;
    }
    let mut r = Rng(5);
    let n = 12288;
    let x: Vec<f32> = (0..n).map(|_| r.f() * 4.0).collect();
    let mut a = QuantAct::with_capacity(n);
    a.ternary = true;
    a.fill(&x);
    let mut want = vec![0i8; a.tp.len()];
    for g in 0..n / 128 {
        for k in 0..32 {
            for s in 0..4 {
                want[peregrine_core::quant::tp_off(g) + 64 * s + k] = a.tq[g * 128 + 4 * k + s];
            }
        }
    }
    assert_eq!(a.tp, want);
}

#[test]
fn ptq1_mode_kernel_matches_scalar() {
    let mut r = Rng(777);
    for n in [256usize, 5120, 17408] {
        let w = ptq1_row(&mut r, n);
        // decoded row · x directly, as a float reference for the scalar spec
        let mut y = vec![0.0f32; n];
        dequant_row(Format::PTQ1_0, &w, &mut y);
        for k in [1usize, 3, 8, 11] {
            let a: Vec<QuantAct> = (0..k)
                .map(|_| {
                    let x: Vec<f32> = (0..n).map(|_| r.f() * 3.0).collect();
                    let mut q = QuantAct::with_capacity(n);
                    q.ternary = true;
                    q.ptq1 = true;
                    q.fill(&x);
                    q
                })
                .collect();
            let mut out = vec![0.0f32; k];
            peregrine_core::quant::dot_batch(Format::PTQ1_0, &w, &a, &mut out);
            check(Format::PTQ1_0, &w, &a, &out, "ptq1 zmm");
        }
    }
}

#[test]
fn pq2_tiles_match_scalar_at_every_span() {
    if !peregrine_core::tern::have_zmm() {
        return;
    }
    let mut r = Rng(31337);
    let (n, rows) = (4096usize, 48usize);
    let w: Vec<u8> = (0..rows).flat_map(|_| pq2_row(&mut r, n)).collect();
    let rb = w.len() / rows;
    let mut tiles = vec![0u8; w.len()];
    peregrine_core::tern::repack_tiles(&w, rows, rb, &mut tiles);
    for span in [32usize, 64, 128] {
        for k in [1usize, 3, 8, 13] {
            let a: Vec<QuantAct> = (0..k)
                .map(|_| {
                    let x: Vec<f32> = (0..n).map(|_| r.f() * 3.0).collect();
                    let mut q = QuantAct::with_capacity(n);
                    q.ternary = true;
                    q.tb_span = span;
                    q.fill(&x);
                    q
                })
                .collect();
            let mut out = vec![0.0f32; rows * k];
            unsafe {
                peregrine_core::tern::dot_tiles_pq2(&tiles, n / 128, rows / 16, &a, &mut out)
            };
            for row in 0..rows {
                // dot_scalar reads tq/td/tls, which carry the span scale in tile mode.
                check(
                    Format::PQ2_0,
                    &w[row * rb..(row + 1) * rb],
                    &a,
                    &out[row * k..(row + 1) * k],
                    "tiles",
                );
            }
        }
    }
}
