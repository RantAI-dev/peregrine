//! **Experimental, measured, and called by nothing in the engine**: the 8-bit
//! prompt form with one scale per **32** weights instead of one per 256 (S5.4i).
//!
//! `docs/design/s5.4-fast-prompt.md` §2 fixes the form the fast prompt mode uses: 8-bit
//! codes, one f32 scale per 256 weights, which is ik_llama.cpp's `-rtr` arithmetic
//! (`q8_k_r16`). That form measures 0.0035 nats of mean KL on Llama 3.2 1B Q4_K_M. The
//! fork's *other* wide path — the one it takes without `-rtr` — converts to `Q8_1` /
//! `Q8_0_R8`, one scale per **32** weights, and that path measures 0.000945 on the same
//! file. The project owner has to choose between the two arithmetics, and this module is
//! the measurement that choice needs: the finer form's builder, its multiply (a scalar
//! twin and an AVX-512 body), its bytes per weight, its build time and its speed against
//! the per-256 form on the same microbenchmark.
//!
//! **Nothing in the engine reads this module.** It is not in the kernel table, not in
//! `q8form`, not on the dispatch; no golden is recorded from it and no model can be
//! loaded with it. It is `pub` so `peregrine-bench`'s `kqbench` and the error tables can
//! call it, and so the numbers in
//! `benchmarks/results/desktop-zen5-2026-11-12-s5.4i/` can be reproduced.
//!
//! # The form
//!
//! Per output row and per 32-weight block: `d = max|w| / 127` (f32, a division), the
//! reciprocal rounded to f32 and multiplied (`code = round_to_nearest_even(w · (1/d))`),
//! the code saturated to int8 and stored biased by 128 so `vpdpbusd` can take it as its
//! unsigned operand — the operations and the order of the per-256 form
//! ([`crate::kq8`]), applied one block level down. One block of a tile of eight rows is
//! one 320-byte unit:
//!
//! ```text
//!   0 .. 256   codes:   codes[group · 64 + row · 8 + k], group 0..4, k 0..8
//! 256 .. 288   scales:  the eight rows' f32 scales
//! 288 .. 320   padding: so every unit starts on a 64-byte boundary
//! ```
//!
//! **1.25 bytes per weight**, against the per-256 form's 1.03125, the file's 0.5625
//! (Q4_K) and 0.8203 (Q6_K). A 1B Q4_K_M file's form would be 1 160 MiB instead of
//! 957.
//!
//! # The activation
//!
//! The per-256 form reads the activation llama.cpp's Q8_K holds — one scale per 256
//! weights ([`crate::kq::KqActs`], built from `QuantAct::qk`). This form reads the
//! per-32 one (`QuantAct::q`, `d`, `sums`), interleaved four rows to a group:
//!
//! ```text
//!   q[(row/4) · n_blk · 128 + (row%4) · 32 + blk · 128 + j · 8 + k]
//! ```
//!
//! so one 32-weight block of four activation rows is 128 bytes, two cache lines, and the
//! kernel's broadcast for eight-weight group `j` is eight contiguous bytes. [`P32Acts`]
//! builds it from the activation rows the engine already has; its cost per weight is one
//! int8 copy, the same as the per-256 layout's.
//!
//! # The multiply
//!
//! The inner loop is the per-256 kernel's, group for group: one `vpdpbusd` per
//! (eight-weight group, activation row), and the same exact int32 arithmetic. What
//! changes is the **epilogue**: the 128-bias payback, the `d · dk` multiply and the fused
//! multiply-add happen once per 32 weights instead of once per 256 — eight times as
//! often, and on the same ports as the byte dot. That is the whole cost of the finer
//! grid, and it is why the fork's default wide path is the slower of its two.

use crate::kq::ACT_GROUP;

/// The weights one block of the per-32 form holds.
pub const P32_WEIGHTS: usize = 32;
/// The eight-weight groups one block splits into.
pub const P32_GROUPS: usize = 4;
/// The bytes one (block, eight-row) unit takes: 256 of codes, 32 of scales, 32 of padding.
pub const P32_SIZE: usize = 320;
/// The byte offset of the eight rows' scales within one unit.
pub const P32_D: usize = 256;
/// The output rows one unit holds.
pub const P32_ROWS: usize = 8;
/// The blocks a 256-weight super-block splits into.
pub const P32_PER_SB: usize = 8;
/// The bytes one 32-weight block of one group of [`ACT_GROUP`] activation rows takes.
pub const P32_ACTS: usize = ACT_GROUP * 32;

/// The byte offset of (group `g`, row `row`, code `k`) within one unit.
pub const fn code_at(g: usize, row: usize, k: usize) -> usize {
    g * 64 + row * 8 + k
}

/// The bytes a form of `n_blk` 32-weight blocks takes.
pub const fn form_size(n_blk: usize) -> usize {
    n_blk * P32_SIZE
}

/// Bytes per weight: the memory side of the decision between the two forms.
pub const fn bytes_per_weight() -> f64 {
    P32_SIZE as f64 / (P32_ROWS * P32_WEIGHTS) as f64
}

/// The per-32 activation a pass needs for this form: the interleaved int8 codes with
/// their block scales and block sums, four rows to a group.
#[derive(Default, Clone, Debug)]
pub struct P32Acts {
    /// The interleaved codes: `groups · n_blk · 128` bytes.
    pub q: Vec<i8>,
    /// One scale per (group, block, row): `groups · n_blk · 4`.
    pub d: Vec<f32>,
    /// `Σ q` per (group, block, row) — the `128 · Σa` of the bias identity.
    pub sums: Vec<i32>,
    /// The rows filled.
    pub rows: usize,
    /// The blocks per row.
    pub n_blk: usize,
}

impl P32Acts {
    /// Fill from the per-32 quantisation the engine's [`crate::quant::QuantAct`] already
    /// holds. The work per weight is one int8 copy plus one add, the same as
    /// `KqActs::fill_with`'s per-256 layout.
    ///
    /// # Panics
    ///
    /// If the rows are not all the same width, or the width is not a whole number of
    /// 32-weight blocks.
    pub fn fill(acts: &[crate::quant::QuantAct]) -> P32Acts {
        let n = acts[0].n;
        assert!(
            n.is_multiple_of(P32_WEIGHTS),
            "the activation is a whole number of 32-weight blocks"
        );
        assert!(acts.iter().all(|a| a.n == n), "the rows are one width");
        let n_blk = n / P32_WEIGHTS;
        let groups = acts.len().div_ceil(ACT_GROUP);
        let mut out = P32Acts {
            q: vec![0i8; groups * n_blk * P32_ACTS],
            d: vec![0.0f32; groups * n_blk * ACT_GROUP],
            sums: vec![0i32; groups * n_blk * ACT_GROUP],
            rows: acts.len(),
            n_blk,
        };
        for (b, a) in acts.iter().enumerate() {
            let (g, t) = (b / ACT_GROUP, b % ACT_GROUP);
            for blk in 0..n_blk {
                let dst = (g * n_blk + blk) * P32_ACTS + t * 32;
                out.q[dst..dst + 32].copy_from_slice(&a.q[blk * 32..blk * 32 + 32]);
                let at = (g * n_blk + blk) * ACT_GROUP + t;
                out.d[at] = a.d[blk];
                out.sums[at] = a.sums[blk];
            }
        }
        out
    }
}

/// The f32 weights of a tile's eight rows, in file order — the input the builder takes.
/// Q4_K and Q6_K differ only in how these f32 numbers are obtained
/// ([`crate::kq8::q4k_weights`], [`crate::kq8::q6k_weights`]), so this module has one
/// builder.
///
/// The scalar builder is the definition: a SIMD builder would write the same bytes
/// (`crate::kq8`'s AVX-512 builder does that for the per-256 form) and is not built
/// here, because the decision this form is for is a speed and accuracy decision, not a
/// load-time one.
///
/// # Panics
///
/// If a row is not a whole number of blocks, the rows disagree, or `out` is short.
pub fn build_tile(rows: &[&[f32]; P32_ROWS], out: &mut [u8]) {
    let n_blk = rows[0].len() / P32_WEIGHTS;
    assert_eq!(
        rows[0].len(),
        n_blk * P32_WEIGHTS,
        "a whole number of blocks"
    );
    assert!(
        rows.iter().all(|r| r.len() == rows[0].len()),
        "every row the same width"
    );
    assert!(out.len() >= form_size(n_blk), "the form holds every block");
    out[..form_size(n_blk)].fill(0);
    for blk in 0..n_blk {
        for (row, w) in rows.iter().enumerate() {
            let w = &w[blk * P32_WEIGHTS..(blk + 1) * P32_WEIGHTS];
            let unit = &mut out[blk * P32_SIZE..(blk + 1) * P32_SIZE];
            let mut m = 0.0f32;
            for &x in w {
                let a = x.abs();
                if a > m {
                    m = a;
                }
            }
            // `d = m / 127`, and the reciprocal rounded to f32 before the multiply: the
            // fork multiplies by a rounded reciprocal rather than dividing.
            let d = m / 127.0f32;
            let id = if d != 0.0 { 1.0f32 / d } else { 0.0 };
            for (g, group) in w.chunks(8).enumerate() {
                for (k, &x) in group.iter().enumerate() {
                    let c = (x * id).round_ties_even().clamp(-127.0, 127.0) as i8;
                    unit[code_at(g, row, k)] = (c as i32 + 128) as u8;
                }
            }
            let at = P32_D + row * 4;
            unit[at..at + 4].copy_from_slice(&d.to_le_bytes());
        }
    }
}

/// One output of the per-32 form: the definition the SIMD body must equal.
///
/// # Panics
///
/// If the form is shorter than `n_blk` blocks or the activation has fewer rows.
pub fn dot_p32(form: &[u8], n_blk: usize, acts: &P32Acts, row: usize, b: usize) -> f32 {
    assert!(form.len() >= form_size(n_blk), "the form holds every block");
    assert!(
        b < acts.rows && n_blk <= acts.n_blk,
        "the activation is large enough"
    );
    let (g, t) = (b / ACT_GROUP, b % ACT_GROUP);
    let mut acc = 0.0f32;
    for blk in 0..n_blk {
        let unit = &form[blk * P32_SIZE..(blk + 1) * P32_SIZE];
        let mut s = 0i32;
        for j in 0..P32_GROUPS {
            for k in 0..8 {
                let c = unit[code_at(j, row, k)] as i32 - 128;
                let a = acts.q[(g * n_blk + blk) * P32_ACTS + t * 32 + j * 8 + k] as i32;
                s += c * a;
            }
        }
        let at = (g * n_blk + blk) * ACT_GROUP + t;
        let off = P32_D + row * 4;
        let d = f32::from_le_bytes([unit[off], unit[off + 1], unit[off + 2], unit[off + 3]]);
        // The bias paid back in int32 lanes, then one multiply `d · dk` and one fused
        // multiply-add — the per-256 twin's epilogue, block for block.
        let x = d * acts.d[at];
        // The twin's codes are signed (`byte − 128`), so the 128 bias is already out of
        // `s`; the SIMD body takes the stored byte as `vpdpbusd`'s unsigned operand and
        // pays the bias back with `−128 · Σa`, which is the same number.
        acc = (s as f32).mul_add(x, acc);
    }
    acc
}

/// A batch of activation rows against eight output rows of one form: the scalar twin of
/// `crate::kqsimd::gemm_p32_zmm`.
///
/// `out[b · 8 + row]` is row `row` of the form for activation row `b` — the layout every
/// repacked tile kernel writes.
///
/// # Panics
///
/// If the form is shorter than `n_blk` blocks or `out` shorter than `nb · 8`.
pub fn gemm_p32(form: &[u8], n_blk: usize, acts: &P32Acts, nb: usize, out: &mut [f32]) {
    assert!(form.len() >= form_size(n_blk), "the form holds every block");
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(out.len() >= nb * P32_ROWS, "the output holds eight rows");
    for b in 0..nb {
        for row in 0..P32_ROWS {
            out[b * P32_ROWS + row] = dot_p32(form, n_blk, acts, row, b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 32) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * 0.05
            })
            .collect()
    }

    #[cfg_attr(not(all(target_arch = "x86_64", feature = "fmt-kquant")), allow(dead_code))]
    fn batch(rows: usize, n: usize, seed: u64) -> Vec<crate::quant::QuantAct> {
        let mut s = seed | 1;
        (0..rows)
            .map(|_| {
                let x: Vec<f32> = (0..n)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        ((s >> 32) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * 3.0
                    })
                    .collect();
                let mut a = crate::quant::QuantAct::with_capacity(n);
                a.fill(&x);
                a
            })
            .collect()
    }

    /// The codes are the fork's own operations, block for block: `d = m / 127`, the
    /// rounded reciprocal, round-to-nearest-even, the +128 bias.
    #[test]
    fn the_builder_is_the_fork_s_arithmetic_one_block_down() {
        let w = weights(256, 0x1234);
        let mut form = vec![0u8; form_size(8)];
        build_tile(&[&w; P32_ROWS], &mut form);
        for blk in 0..8 {
            let block = &w[blk * 32..(blk + 1) * 32];
            let m = block.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let d = m / 127.0f32;
            let id = 1.0f32 / d;
            let unit = &form[blk * P32_SIZE..(blk + 1) * P32_SIZE];
            assert_eq!(
                f32::from_le_bytes(unit[P32_D..P32_D + 4].try_into().unwrap()),
                d
            );
            for g in 0..P32_GROUPS {
                for k in 0..8 {
                    let want = (block[g * 8 + k] * id)
                        .round_ties_even()
                        .clamp(-127.0, 127.0) as i32;
                    assert_eq!(
                        unit[code_at(g, 0, k)] as i32,
                        want + 128,
                        "block {blk} group {g} code {k}"
                    );
                }
            }
            // Every code is inside the range `vpdpbusd` can take as an unsigned byte.
            assert!(unit[..P32_D].iter().all(|&c| (1..=255).contains(&c)));
        }
    }

    /// The AVX-512 body is the twin's, bit for bit, at every activation width the prompt
    /// pass runs at. The body exists only on x86-64 (`crate::kqsimd` is
    /// `#![cfg(target_arch = "x86_64")]`), so the test is gated to that architecture —
    /// which is the only architecture where the identity is defined.
    #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
    #[test]
    fn the_per_32_simd_body_is_the_twin() {
        if !crate::kqsimd::have_zmm512() {
            return;
        }
        let n = 512;
        let w = weights(n, 0xBEEF);
        let mut form = vec![0u8; form_size(n / 32)];
        build_tile(&[&w; P32_ROWS], &mut form);
        let rows = batch(13, n, 0x5EED);
        let acts = P32Acts::fill(&rows);
        for nb in 1..=13 {
            let mut twin = vec![0.0f32; nb * 8];
            gemm_p32(&form, n / 32, &acts, nb, &mut twin);
            let mut got = vec![0.0f32; nb * 8];
            // SAFETY: `have_zmm512()` checked; the form holds `n / 32` blocks and the
            // activation `nb` rows.
            unsafe { crate::kqsimd::gemm_p32_zmm(&form, n / 32, &acts, nb, &mut got) };
            for i in 0..nb * 8 {
                if got[i].to_bits() != twin[i].to_bits() && i < 8 {
                    println!(
                        "nb {nb} i {i}: got {} twin {} (all got {:?})",
                        got[i],
                        twin[i],
                        &got[..8]
                    );
                }
                assert_eq!(got[i].to_bits(), twin[i].to_bits(), "width {nb}, index {i}");
            }
        }
    }

    /// Each block's weights are within half its own step, `m / 254`, and a 32-weight
    /// block's `m` is the block's own maximum — which is the whole accuracy argument for
    /// the finer grid on this file.
    #[test]
    fn the_per_32_block_meets_its_own_bound() {
        let w = weights(256, 0x99);
        let mut form = vec![0u8; form_size(8)];
        build_tile(&[&w; P32_ROWS], &mut form);
        for blk in 0..8 {
            let mut worst = 0.0f32;
            let block = &w[blk * 32..(blk + 1) * 32];
            let d = f32::from_le_bytes(
                form[blk * P32_SIZE + P32_D..blk * P32_SIZE + P32_D + 4]
                    .try_into()
                    .unwrap(),
            );
            for g in 0..P32_GROUPS {
                for k in 0..8 {
                    let c = form[blk * P32_SIZE + code_at(g, 0, k)] as i32 - 128;
                    worst = worst.max((c as f32 * d - block[g * 8 + k]).abs());
                }
            }
            let m = block.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            assert!(worst <= m / 254.0 + 1e-9, "{worst} against {}", m / 254.0);
        }
    }
}
