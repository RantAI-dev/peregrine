//! The encoder attention strip (roadmap S4.9g, ADR 0005).
//!
//! An encoder's attention is bidirectional over a whole text: every query row of a
//! text scores every other row of that text, under the family's window. The scalar
//! loop this replaces (`crates/peregrine-core/src/blocks/bidir.rs` before S4.9g) walked
//! one `(query row, key)` pair at a time: the dot product was a strictly ordered f32
//! chain with a loop-carried dependency, so it ran at one multiply-add per four cycles
//! and spent 35 % of an EmbeddingGemma pass at 128 tokens and 60 % at 512.
//!
//! This module is that loop, restructured so the arithmetic is the same for every
//! kernel tier:
//!
//! 1. [`key_panel`] transposes one head's keys into `[head_dim][n]` — keys along the
//!    vector lanes, so a lane is one score and the chain over the head dimension
//!    stays sequential *inside* the lane. It is built once per `(text, KV head)` and
//!    serves every query head of that KV head.
//! 2. [`scores_row`] scores one query row against all of its keys. A lane holds one
//!    key and the head dimension is folded with `acc = fma(q[k], key[k], acc)`, so the
//!    scalar twin (the reference) and AVX2 agree bit for bit: no horizontal
//!    reduction, no reassociation. Only keys inside the row's window are computed.
//! 3. [`exp_row`] turns the row's scores into weights with the engine's vector `exp`
//!    (`vmath::exp8`) and its scalar twin (`vmath::exp_scalar`), which are
//!    bit-identical, and returns the denominator as an `f64` sum in ascending key
//!    order — the order the reference forward (`tests/support/embed_ref.rs`) uses.
//! 4. [`mix_rows`] accumulates `P · V` for a block of up to four rows that share one
//!    value head, ascending key order inside each row, `fma(p[j], v[j][d], acc[d])`.
//!
//! # The defined arithmetic
//!
//! For a text of `n` keys, query row `i` and its window `lo..hi = band(i, n, half)`:
//!
//! ```text
//! score[j] = (fold k = 0..hd: acc = fma(q[k], key[k], acc)) * scale     j in lo..hi
//! max      = max(score[j])                                              j in lo..hi
//! e[j]     = exp(score[j] - max)                                        j in lo..hi
//! l        = (fold j ascending: sum += e[j] as f64)                     j in lo..hi
//! out[d]   = (fold j ascending: acc = fma(e[j], v[j][d], acc)) * (1/l)  j in lo..hi
//! ```
//!
//! Every step is a fixed fold: a row's bits depend on its own text (length, position,
//! layer), never on the strip it fell in, the thread that ran it, the other texts of
//! the pass or the kernel tier. The strip's `n × n` scores are never materialised for
//! the whole text: a caller holds `strip × n` of them, and the head's keys are read
//! once per strip into L1-sized blocks ([`SCORE_BLOCK`] keys).
//!
//! # aarch64
//!
//! The scalar twins compile everywhere and are the reference. `// S4.9g:` marks where
//! the NEON kernels go (`simd.rs`/`neon.rs` style: `vfmaq_f32` with the same lane
//! meaning, and `exp4` from `vmath`).

use crate::vmath::{exp8, exp_scalar};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Keys per strip block. The block is what keeps a head's keys in L1 while every row
/// of the strip scores against them: `head_dim × SCORE_BLOCK × 4` bytes is 32 KiB at
/// the widest head this engine supports (256), so a block of keys stays in L1 across
/// the strip's rows instead of being re-read from L2 or L3 for every row.
pub const SCORE_BLOCK: usize = 32;

/// Query rows per strip, at most. A strip's scores are `strip × n` f32 — 64 KiB at a
/// 512-token text and 128 KiB at 2 000 — and the strip is what makes the mix read the
/// value rows once for four query rows at a time.
pub const STRIP_MAX: usize = 32;

/// The score floats one worker's strip buffer holds. The strip is sized so
/// `strip × n` stays inside it: 32 rows at 1 024 tokens, 16 at 2 048, 1 at 32 768.
pub const STRIP_AREA: usize = 32 * 1024;

/// Floats one worker owns: its strip's scores and, after them,
/// [`STRIP_MAX`] softmax denominators (the `1/l` of each row of the strip).
pub const STRIP_SLOT: usize = STRIP_AREA + STRIP_MAX;

/// Keys per vector pass in a score row (four eight-lane accumulators).
const KEYS_PER_GROUP: usize = 4 * LANES;
/// Value rows the mix shares one value load between.
const MIX_ROWS: usize = 4;
/// Head-dimension lanes per mix accumulator (two vectors).
const MIX_D: usize = 16;

/// Keys per vector accumulator — one `__m256` lane per key, so a lane is one score.
const LANES: usize = 8;

#[inline]
fn vector_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::simd::have_batched()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The window of query row `i` of a text of `n` keys: `lo..hi`.
///
/// The rule is llama.cpp's (`docs/reference/gemma-embedding.md`): on a sliding layer
/// key `j` is visible from row `i` when `|i - j| <= half`, so the band is
/// `max(0, i - half) .. min(n, i + half + 1)` — an exact band, never a mask over
/// computed scores. `None` is a global layer: every key of the text.
#[inline]
pub fn band(i: usize, n: usize, half: Option<usize>) -> (usize, usize) {
    match half {
        None => (0, n),
        Some(h) => (i.saturating_sub(h), (i + h + 1).min(n)),
    }
}

/// One head's keys of one text as `[head_dim][n]`: `panel[k * n + j] = keys[j * stride + base + k]`.
///
/// `keys` is the text's key rows (`n` rows of `stride` floats), `base` the KV head's
/// offset inside a row. Only [`key_panel_scalar`] exists: the copy is a transpose, not
/// arithmetic, and its layout is all the score kernel needs. It is written in tiles of
/// 16×16 so each source cache line serves sixteen head dimensions and each destination
/// line is written whole; `// S4.9g:` an eight-wide register transpose is the next
/// step here if the panel build ever shows in a profile (it is ~2 % of the score work
/// at 512 tokens).
pub fn key_panel(keys: &[f32], stride: usize, base: usize, hd: usize, n: usize, panel: &mut [f32]) {
    key_panel_scalar(keys, stride, base, hd, n, panel)
}

/// [`key_panel`]'s scalar twin.
pub fn key_panel_scalar(
    keys: &[f32],
    stride: usize,
    base: usize,
    hd: usize,
    n: usize,
    panel: &mut [f32],
) {
    debug_assert!(panel.len() >= hd * n);
    debug_assert!(keys.len() >= n * stride);
    const T: usize = 16;
    let mut j0 = 0;
    while j0 < n {
        let j1 = (j0 + T).min(n);
        let mut k0 = 0;
        while k0 < hd {
            let k1 = (k0 + T).min(hd);
            for k in k0..k1 {
                let dst = &mut panel[k * n + j0..k * n + j1];
                for (jj, d) in dst.iter_mut().enumerate() {
                    *d = keys[(j0 + jj) * stride + base + k];
                }
            }
            k0 = k1;
        }
        j0 = j1;
    }
}

/// `out[j] = scale · Σ_k q[k]·panel[k][j]` for `j` in `lo..hi`, one sequential FMA
/// chain per key.
///
/// `panel` is [`key_panel`]'s layout with `n` keys; `q` holds one row's head (at least
/// `hd` floats); `out` is that row's `n` scores.
#[allow(clippy::too_many_arguments)]
pub fn scores_row(
    q: &[f32],
    panel: &[f32],
    n: usize,
    hd: usize,
    lo: usize,
    hi: usize,
    scale: f32,
    out: &mut [f32],
) {
    debug_assert!(lo <= hi && hi <= n);
    debug_assert!(q.len() >= hd && panel.len() >= hd * n && out.len() >= n);
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 + FMA checked; every load and store is bounds-checked above.
            unsafe { scores_row_avx2(q, panel, n, hd, lo, hi, scale, out) };
            return;
        }
    }
    scores_row_scalar(q, panel, n, hd, lo, hi, scale, out)
}

/// [`scores_row`]'s scalar twin: the reference.
#[allow(clippy::too_many_arguments)]
pub fn scores_row_scalar(
    q: &[f32],
    panel: &[f32],
    n: usize,
    hd: usize,
    lo: usize,
    hi: usize,
    scale: f32,
    out: &mut [f32],
) {
    for (j, o) in out.iter_mut().enumerate().take(hi).skip(lo) {
        let mut acc = 0.0f32;
        for k in 0..hd {
            acc = q[k].mul_add(panel[k * n + j], acc);
        }
        *o = acc * scale;
    }
}

/// Four accumulators, eight keys each: the chain over the head dimension is the inner
/// loop, so each key's dependency is one FMA deep. `q[k]` is a broadcast load, the
/// panel row a full vector load.
///
/// # Safety
/// AVX2 and FMA must be available; `lo..hi ⊆ 0..n`, `q.len() >= hd`,
/// `panel.len() >= hd * n`, `out.len() >= n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn scores_row_avx2(
    q: &[f32],
    panel: &[f32],
    n: usize,
    hd: usize,
    lo: usize,
    hi: usize,
    scale: f32,
    out: &mut [f32],
) {
    let s = _mm256_set1_ps(scale);
    macro_rules! group {
        ($j:expr, $($a:ident),+) => {{
            $(let mut $a = _mm256_setzero_ps();)+
            for k in 0..hd {
                let qk = _mm256_set1_ps(*q.get_unchecked(k));
                let p = panel.as_ptr().add(k * n + $j);
                let mut off = 0;
                $(
                    $a = _mm256_fmadd_ps(qk, _mm256_loadu_ps(p.add(off)), $a);
                    off += LANES;
                )+
                let _ = off;
            }
            let p = out.as_mut_ptr().add($j);
            let mut off = 0;
            $(
                _mm256_storeu_ps(p.add(off), _mm256_mul_ps($a, s));
                off += LANES;
            )+
            let _ = off;
        }};
    }
    let mut j = lo;
    while j + KEYS_PER_GROUP <= hi {
        group!(j, a0, a1, a2, a3);
        j += KEYS_PER_GROUP;
    }
    while j + LANES <= hi {
        group!(j, a0);
        j += LANES;
    }
    if j < hi {
        if hi - lo >= LANES {
            // The last group overlaps the previous one: the keys it recomputes are the
            // same keys with the same arithmetic, and only lanes inside `hi` are read
            // by the softmax. This is what keeps the tail off a scalar chain (a
            // seven-key scalar tail costs four cycles per element).
            j = hi - LANES;
            group!(j, a0);
        } else {
            for j in j..hi {
                let mut acc = 0.0f32;
                for k in 0..hd {
                    acc = q[k].mul_add(*panel.get_unchecked(k * n + j), acc);
                }
                *out.get_unchecked_mut(j) = acc * scale;
            }
        }
    }
}

/// `x[j] = exp(x[j] - max)` for `j` in `lo..hi`; returns `Σ x[j]` as `f64`, in
/// ascending key order.
///
/// The exponentials are `vmath::exp8`'s and `vmath::exp_scalar`'s (bit-identical, the
/// scalar twin of the same polynomial), so every tier produces the same weights on
/// the x86 tiers and the scalar tier alike. The sum is scalar in ascending order in
/// every path, which is what makes the denominator order-independent of the kernel:
/// the *definition* is the ascending sum, not the hardware's grouping.
pub fn exp_row(x: &mut [f32], lo: usize, hi: usize, max: f32) -> f64 {
    debug_assert!(lo <= hi && hi <= x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 + FMA checked; the loop keeps every access inside `lo..hi`.
            unsafe { exp_row_avx2(x, lo, hi, max) };
            return sum_row(x, lo, hi);
        }
    }
    exp_row_scalar(x, lo, hi, max);
    sum_row(x, lo, hi)
}

/// [`exp_row`]'s scalar twin (the caller sums).
pub fn exp_row_scalar(x: &mut [f32], lo: usize, hi: usize, max: f32) {
    for v in x[lo..hi].iter_mut() {
        *v = exp_scalar(*v - max);
    }
}

/// The ascending `f64` sum of `x[lo..hi]` — the softmax denominator, defined this way
/// so it cannot depend on how a tier groups its lanes.
#[inline]
pub fn sum_row(x: &[f32], lo: usize, hi: usize) -> f64 {
    let mut sum = 0.0f64;
    for &v in &x[lo..hi] {
        sum += v as f64;
    }
    sum
}

/// # Safety
/// AVX2 and FMA must be available; `lo <= hi <= x.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_row_avx2(x: &mut [f32], lo: usize, hi: usize, max: f32) {
    let m = _mm256_set1_ps(max);
    let mut j = lo;
    while j + LANES <= hi {
        let v = _mm256_loadu_ps(x.as_ptr().add(j));
        _mm256_storeu_ps(x.as_mut_ptr().add(j), exp8(_mm256_sub_ps(v, m)));
        j += LANES;
    }
    while j < hi {
        *x.get_unchecked_mut(j) = exp_scalar(*x.get_unchecked(j) - max);
        j += 1;
    }
}

/// `out[r][d] = (Σ_j e[r][j]·v[j][d]) · inv[r]`, `j` ascending, for `rows` query rows
/// that start at text-local index `i0` and share one value head.
///
/// `e` is the strip's weights (`rows` rows of `e_stride`), `v` the pass's value rows
/// (`v_stride` floats each) with the head at `base`, `out` the strip's rows of the
/// attention output at `out_stride` with the head at `obase`. Each row's window comes
/// from [`band`].
#[allow(clippy::too_many_arguments)]
pub fn mix_rows(
    e: &[f32],
    e_stride: usize,
    i0: usize,
    rows: usize,
    n: usize,
    half: Option<usize>,
    v: &[f32],
    v_stride: usize,
    base: usize,
    hd: usize,
    inv: &[f32],
    out: &mut [f32],
    out_stride: usize,
    obase: usize,
) {
    debug_assert!(e.len() >= rows * e_stride);
    debug_assert!(inv.len() >= rows);
    debug_assert!(out.len() >= rows * out_stride);
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 + FMA checked; rows are inside `e`, `inv` and `out`, keys
            // inside `n`, and the value loads stay inside one row of `v` because
            // `d0 + 16 <= hd <= v_stride - base`.
            unsafe {
                mix_rows_avx2(
                    e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out, out_stride,
                    obase,
                )
            };
            return;
        }
    }
    mix_rows_scalar(
        e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out, out_stride, obase,
    );
}

/// [`mix_rows`]'s scalar twin: the reference.
#[allow(clippy::too_many_arguments)]
pub fn mix_rows_scalar(
    e: &[f32],
    e_stride: usize,
    i0: usize,
    rows: usize,
    n: usize,
    half: Option<usize>,
    v: &[f32],
    v_stride: usize,
    base: usize,
    hd: usize,
    inv: &[f32],
    out: &mut [f32],
    out_stride: usize,
    obase: usize,
) {
    for r in 0..rows {
        let (lo, hi) = band(i0 + r, n, half);
        let e_row = &e[r * e_stride..];
        let o = &mut out[r * out_stride + obase..];
        for d in 0..hd {
            let mut acc = 0.0f32;
            for j in lo..hi {
                acc = e_row[j].mul_add(v[j * v_stride + base + d], acc);
            }
            o[d] = acc * inv[r];
        }
    }
}

/// Four rows at a time share one value load; two accumulators (16 lanes) per row.
///
/// # Safety
/// AVX2 and FMA must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_rows_avx2(
    e: &[f32],
    e_stride: usize,
    i0: usize,
    rows: usize,
    n: usize,
    half: Option<usize>,
    v: &[f32],
    v_stride: usize,
    base: usize,
    hd: usize,
    inv: &[f32],
    out: &mut [f32],
    out_stride: usize,
    obase: usize,
) {
    let ep = e.as_ptr();
    let vp = v.as_ptr();
    let mut r0 = 0;
    while r0 + MIX_ROWS <= rows {
        let bands = [
            band(i0 + r0, n, half),
            band(i0 + r0 + 1, n, half),
            band(i0 + r0 + 2, n, half),
            band(i0 + r0 + 3, n, half),
        ];
        // The core is the key range every row of the block sees; the edges are the
        // (at most three) keys at each end that only some rows see. Accumulating the
        // low edge, then the core, then the high edge keeps every row's fold in
        // ascending key order.
        let core_lo = bands.iter().map(|b| b.0).max().unwrap_or(0);
        let core_hi = bands.iter().map(|b| b.1).min().unwrap_or(0);
        let mut d0 = 0;
        while d0 + MIX_D <= hd {
            let mut m00 = _mm256_setzero_ps();
            let mut m01 = _mm256_setzero_ps();
            let mut m10 = _mm256_setzero_ps();
            let mut m11 = _mm256_setzero_ps();
            let mut m20 = _mm256_setzero_ps();
            let mut m21 = _mm256_setzero_ps();
            let mut m30 = _mm256_setzero_ps();
            let mut m31 = _mm256_setzero_ps();
            // The block's rows are the strip's rows `r0..r0 + 4`.
            let eb = ep.add(r0 * e_stride);
            macro_rules! edge {
                ($a:ident, $b:ident, $r:expr, $range:expr) => {
                    for j in $range {
                        let ev = _mm256_set1_ps(*eb.add($r * e_stride + j));
                        let p = vp.add(j * v_stride + base + d0);
                        $a = _mm256_fmadd_ps(ev, _mm256_loadu_ps(p), $a);
                        $b = _mm256_fmadd_ps(ev, _mm256_loadu_ps(p.add(LANES)), $b);
                    }
                };
            }
            edge!(m00, m01, 0, bands[0].0..core_lo);
            edge!(m10, m11, 1, bands[1].0..core_lo);
            edge!(m20, m21, 2, bands[2].0..core_lo);
            edge!(m30, m31, 3, bands[3].0..core_lo);
            for j in core_lo..core_hi {
                let p = vp.add(j * v_stride + base + d0);
                let v0 = _mm256_loadu_ps(p);
                let v1 = _mm256_loadu_ps(p.add(LANES));
                let e0 = _mm256_set1_ps(*eb.add(j));
                let e1 = _mm256_set1_ps(*eb.add(e_stride + j));
                let e2 = _mm256_set1_ps(*eb.add(2 * e_stride + j));
                let e3 = _mm256_set1_ps(*eb.add(3 * e_stride + j));
                m00 = _mm256_fmadd_ps(e0, v0, m00);
                m01 = _mm256_fmadd_ps(e0, v1, m01);
                m10 = _mm256_fmadd_ps(e1, v0, m10);
                m11 = _mm256_fmadd_ps(e1, v1, m11);
                m20 = _mm256_fmadd_ps(e2, v0, m20);
                m21 = _mm256_fmadd_ps(e2, v1, m21);
                m30 = _mm256_fmadd_ps(e3, v0, m30);
                m31 = _mm256_fmadd_ps(e3, v1, m31);
            }
            edge!(m00, m01, 0, core_hi..bands[0].1);
            edge!(m10, m11, 1, core_hi..bands[1].1);
            edge!(m20, m21, 2, core_hi..bands[2].1);
            edge!(m30, m31, 3, core_hi..bands[3].1);
            let mut o = out.as_mut_ptr().add(r0 * out_stride + obase + d0);
            for (i, (a, b)) in [(m00, m01), (m10, m11), (m20, m21), (m30, m31)]
                .into_iter()
                .enumerate()
            {
                let k = _mm256_set1_ps(*inv.get_unchecked(r0 + i));
                _mm256_storeu_ps(o, _mm256_mul_ps(a, k));
                _mm256_storeu_ps(o.add(LANES), _mm256_mul_ps(b, k));
                o = o.add(out_stride);
            }
            d0 += MIX_D;
        }
        if d0 < hd {
            // Head dimensions narrower than one accumulator pair: scalar, in the same
            // ascending order as the reference (the loop's order over `d` is free).
            for r in 0..MIX_ROWS {
                let (lo, hi) = bands[r];
                let e_row = &e[r * e_stride..];
                let o = &mut out[(r0 + r) * out_stride + obase..];
                for d in d0..hd {
                    let mut acc = 0.0f32;
                    for j in lo..hi {
                        acc = e_row[j].mul_add(v[j * v_stride + base + d], acc);
                    }
                    o[d] = acc * inv[r0 + r];
                }
            }
        }
        r0 += MIX_ROWS;
    }
    while r0 < rows {
        let (lo, hi) = band(i0 + r0, n, half);
        let e_row = &e[r0 * e_stride..];
        let o = &mut out[r0 * out_stride + obase..];
        let mut d0 = 0;
        while d0 + MIX_D <= hd {
            let mut a = _mm256_setzero_ps();
            let mut b = _mm256_setzero_ps();
            for j in lo..hi {
                let ev = _mm256_set1_ps(*e_row.get_unchecked(j));
                let p = vp.add(j * v_stride + base + d0);
                a = _mm256_fmadd_ps(ev, _mm256_loadu_ps(p), a);
                b = _mm256_fmadd_ps(ev, _mm256_loadu_ps(p.add(LANES)), b);
            }
            let k = _mm256_set1_ps(*inv.get_unchecked(r0));
            _mm256_storeu_ps(o.as_mut_ptr().add(d0), _mm256_mul_ps(a, k));
            _mm256_storeu_ps(o.as_mut_ptr().add(d0 + LANES), _mm256_mul_ps(b, k));
            d0 += MIX_D;
        }
        for d in d0..hd {
            let mut acc = 0.0f32;
            for j in lo..hi {
                acc = e_row[j].mul_add(v[j * v_stride + base + d], acc);
            }
            o[d] = acc * inv[r0];
        }
        r0 += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prng(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 16_777_216.0).mul_add(hi - lo, lo)
        };
        (0..n).map(|_| next()).collect()
    }

    /// The first index at which two vectors differ in bits, or `None`.
    fn first_diff(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
        a.iter()
            .zip(b)
            .position(|(x, y)| x.to_bits() != y.to_bits())
            .map(|i| (i, a[i], b[i]))
    }

    fn same(a: &[f32], b: &[f32], what: &str) {
        if let Some((i, x, y)) = first_diff(a, b) {
            panic!("{what}: index {i} {x:?} != {y:?}");
        }
    }

    const HDS: [usize; 6] = [16, 32, 64, 96, 128, 256];
    const WINDOWS: [Option<usize>; 6] = [None, Some(1), Some(8), Some(128), Some(256), Some(4096)];

    /// One text's attention through the kernels, exactly as `blocks/bidir.rs` drives
    /// them: the panel per KV head, then strips of `strip` query rows scored,
    /// soft-maxed and mixed while hot.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        n: usize,
        hd: usize,
        n_head: usize,
        n_kv: usize,
        half: Option<usize>,
        strip: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        scalar: bool,
    ) -> Vec<f32> {
        let qd = n_head * hd;
        let kvd = n_kv * hd;
        let gqa = n_head / n_kv;
        let scale = 0.25f32;
        let mut out = vec![0.0f32; n * qd];
        let mut panel = vec![0.0f32; hd * n];
        let mut scores = vec![0.0f32; strip * n];
        let mut inv = vec![0.0f32; strip];
        for kh in 0..n_kv {
            key_panel(k, kvd, kh * hd, hd, n, &mut panel);
            for h in kh * gqa..(kh + 1) * gqa {
                let mut r0 = 0;
                while r0 < n {
                    let rows = strip.min(n - r0);
                    for r in 0..rows {
                        let i = r0 + r;
                        let (lo, hi) = band(i, n, half);
                        let s = &mut scores[r * n..(r + 1) * n];
                        if scalar {
                            scores_row_scalar(
                                &q[i * qd + h * hd..],
                                &panel,
                                n,
                                hd,
                                lo,
                                hi,
                                scale,
                                s,
                            );
                            let max = s[lo..hi].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            exp_row_scalar(s, lo, hi, max);
                            let l = sum_row(s, lo, hi);
                            inv[r] = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
                        } else {
                            scores_row(&q[i * qd + h * hd..], &panel, n, hd, lo, hi, scale, s);
                            let max = s[lo..hi].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let l = exp_row(s, lo, hi, max);
                            inv[r] = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
                        }
                    }
                    let tail = &mut out[r0 * qd..];
                    if scalar {
                        mix_rows_scalar(
                            &scores,
                            n,
                            r0,
                            rows,
                            n,
                            half,
                            v,
                            kvd,
                            kh * hd,
                            hd,
                            &inv,
                            tail,
                            qd,
                            h * hd,
                        );
                    } else {
                        mix_rows(
                            &scores,
                            n,
                            r0,
                            rows,
                            n,
                            half,
                            v,
                            kvd,
                            kh * hd,
                            hd,
                            &inv,
                            tail,
                            qd,
                            h * hd,
                        );
                    }
                    r0 += rows;
                }
            }
        }
        out
    }

    #[test]
    fn scalar_and_avx2_agree_bit_for_bit() {
        for &hd in &HDS {
            for (ci, &n) in [1usize, 2, 7, 8, 15, 16, 31, 32, 33, 64, 129, 512, 600]
                .iter()
                .enumerate()
            {
                for (wi, &half) in WINDOWS.iter().enumerate() {
                    let seed = 0x51a0_0000u64 ^ ((hd as u64) << 16) ^ ci as u64 ^ wi as u64;
                    let q = prng(seed ^ (n as u64), n * hd, -1.0, 1.0);
                    let k = prng(seed ^ 7, n * hd, -1.0, 1.0);
                    let v = prng(seed ^ 11, n * hd, -1.0, 1.0);
                    for strip in [1usize, 7, 16, 32, n.max(1)] {
                        let want = attend(n, hd, 1, 1, half, strip, &q, &k, &v, true);
                        let got = attend(n, hd, 1, 1, half, strip, &q, &k, &v, false);
                        same(
                            &got,
                            &want,
                            &format!("hd {hd} n {n} half {half:?} strip {strip}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_row_is_the_same_alone_in_a_strip_and_at_any_strip_size() {
        for &hd in &[16usize, 32, 256] {
            for &n in &[1usize, 7, 16, 33, 128, 300] {
                for &half in &[None, Some(1), Some(8), Some(256)] {
                    let q = prng(0xabc ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                    let k = prng(0xdef ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                    let v = prng(0x123 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                    let full = attend(n, hd, 1, 1, half, n, &q, &k, &v, false);
                    for strip in [1usize, 2, 5, 16, 32] {
                        let got = attend(n, hd, 1, 1, half, strip, &q, &k, &v, false);
                        same(
                            &got,
                            &full,
                            &format!("hd {hd} n {n} half {half:?} strip {strip}"),
                        );
                    }
                    // One row alone: a strip of exactly that row, at every position.
                    for i in 0..n {
                        let alone = attend(n, hd, 1, 1, half, 1, &q, &k, &v, false);
                        same(
                            &alone[i * hd..(i + 1) * hd],
                            &full[i * hd..(i + 1) * hd],
                            &format!("hd {hd} n {n} half {half:?} row {i} alone"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_group_of_query_heads_shares_its_kv_head() {
        for &(n_head, n_kv) in &[(3usize, 1usize), (4, 2), (4, 4)] {
            let hd = 32;
            let n = 40;
            let qd = n_head * hd;
            let kvd = n_kv * hd;
            let q = prng(0x9e37 ^ n_head as u64, n * qd, -1.0, 1.0);
            let k = prng(0x85eb ^ n_kv as u64, n * kvd, -1.0, 1.0);
            let v = prng(0xc2b2 ^ n_kv as u64, n * kvd, -1.0, 1.0);
            for half in [None, Some(6)] {
                let want = attend(n, hd, n_head, n_kv, half, 16, &q, &k, &v, true);
                let got = attend(n, hd, n_head, n_kv, half, 16, &q, &k, &v, false);
                same(&got, &want, &format!("gqa {n_head}:{n_kv} half {half:?}"));
                // A value head serves its query heads: each head's output is the
                // single-head kernel run on that head's own queries and its KV head.
                for h in 0..n_head {
                    let kh = h / (n_head / n_kv);
                    let mut qh = vec![0.0f32; n * hd];
                    for i in 0..n {
                        qh[i * hd..(i + 1) * hd]
                            .copy_from_slice(&q[i * qd + h * hd..i * qd + (h + 1) * hd]);
                    }
                    let kk: Vec<f32> = (0..n)
                        .flat_map(|i| k[i * kvd + kh * hd..i * kvd + (kh + 1) * hd].to_vec())
                        .collect();
                    let vv: Vec<f32> = (0..n)
                        .flat_map(|i| v[i * kvd + kh * hd..i * kvd + (kh + 1) * hd].to_vec())
                        .collect();
                    let one = attend(n, hd, 1, 1, half, 16, &qh, &kk, &vv, false);
                    for i in 0..n {
                        same(
                            &one[i * hd..(i + 1) * hd],
                            &got[i * qd + h * hd..i * qd + (h + 1) * hd],
                            &format!("head {h} row {i} is its own head's attention"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_window_is_the_symmetric_band() {
        let n = 20;
        assert_eq!(band(0, n, None), (0, 20));
        assert_eq!(band(0, n, Some(3)), (0, 4));
        assert_eq!(band(10, n, Some(3)), (7, 14));
        assert_eq!(band(19, n, Some(3)), (16, 20));
        // A window wider than the text is the whole text.
        assert_eq!(band(10, n, Some(4096)), (0, 20));
        // `n_swa / 2` at the model's 512 is a window of 256 each way.
        assert_eq!(band(100, 512, Some(256)), (0, 357));
        assert_eq!(band(500, 512, Some(256)), (244, 512));
    }

    #[test]
    fn the_panel_is_the_heads_keys_with_keys_along_the_lanes() {
        let (n, hd, kvd, base) = (5usize, 8usize, 24usize, 8usize);
        let k = prng(0xfeed, n * kvd, -1.0, 1.0);
        let mut panel = vec![0.0f32; hd * n];
        key_panel(&k, kvd, base, hd, n, &mut panel);
        for j in 0..n {
            for d in 0..hd {
                assert_eq!(panel[d * n + j], k[j * kvd + base + d], "{d},{j}");
            }
        }
    }

    #[test]
    fn a_softmax_keeps_its_terms_and_its_floor() {
        // Scores spanning far more than f32's exponent range: the ascending f64 sum
        // is the definition, so both paths compute the same weights and the same
        // denominator.
        let mut x: Vec<f32> = (0..600).map(|i| -(i as f32) * 0.3).collect();
        let mut y = x.clone();
        let a = exp_row(&mut x, 0, 600, 0.0);
        exp_row_scalar(&mut y, 0, 600, 0.0);
        same(&x, &y, "exp");
        assert_eq!(a, sum_row(&y, 0, 600));
        // The floor: a score 200 below the row's max is exactly 0; the max is 1.
        assert_eq!(x[599], 0.0);
        assert_eq!(x[0], 1.0);
        // And a row of two keys keeps both.
        let mut z = vec![0.0f32, 1.0];
        let s = exp_row(&mut z, 0, 2, 1.0);
        assert_eq!(z[0], exp_scalar(-1.0));
        assert_eq!(z[1], 1.0);
        assert_eq!(s, z[0] as f64 + 1.0);
    }
}
