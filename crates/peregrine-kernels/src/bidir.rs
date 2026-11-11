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
//! 2. [`scores_tile`] scores a strip of query rows against a block of keys. A lane
//!    holds one key and the head dimension is folded with
//!    `acc = fma(q[k], key[k], acc)`, so the scalar twin (the reference) and AVX2
//!    agree bit for bit: no horizontal reduction, no reassociation. A tile loads
//!    each panel step once and broadcasts one query value per row — the broadcast
//!    form of the float GEMM ([`crate::fpack`]'s 6×16 tile) — so the tile changes
//!    the rate, never a bit. Only keys inside a row's window are read back.
//! 3. [`exp_row`] turns the row's scores into weights with the engine's vector `exp`
//!    (`vmath::exp8`) and its scalar twin (`vmath::exp_scalar`), which are
//!    bit-identical, and returns the denominator as an `f64` sum in ascending key
//!    order — the order the reference forward (`tests/support/embed_ref.rs`) uses.
//! 4. [`mix_rows`] accumulates `P · V` for a block of up to six rows that share one
//!    value head and one value load, ascending key order inside each row,
//!    `fma(p[j], v[j][d], acc[d])`.
//!
//! The three are the same shape: `scores_tile` is `Q · Kᵀ` and `mix_rows` is `P · V`,
//! both broadcast-form products whose lanes are independent outputs and whose folded
//! index stays in its defined order inside each lane.
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

use crate::vmath::exp_scalar;

#[cfg(target_arch = "x86_64")]
use crate::vmath::exp8;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Keys per strip block: what bounds the panel slice one row of a strip walks, so a
/// strip of 32 rows re-walks an L2-sized slice instead of the whole panel.
/// `head_dim × SCORE_BLOCK × 4` bytes is 128 KiB at the widest head this engine
/// supports (256) — a quarter of this machine's 256 KiB L2 — and 32 KiB at a 64-wide
/// head, which is L1. A block is a whole number of 32-key groups, so an interior block
/// of a band costs no overlap.
///
/// Measured on H0 at 512 and 2 048 tokens with the S4.9g strip, the block changed the
/// score rate by under 1 %: what limits the score kernel is the *stride* of the panel
/// read (one 64-byte run per head dimension, `n × 4` bytes apart), not the slice's
/// residency. With the tile (S4.9g part 2) 64 and 128 keys measure the same to within
/// the machine's noise at every length, so the block stays at 128: it bounds the
/// working set on smaller caches, and a key-major tiled panel remains the one change
/// that would make the traversal sequential.
pub const SCORE_BLOCK: usize = 128;

/// Query rows per strip, at most. A strip's scores are `strip × n` f32 — 64 KiB at a
/// 512-token text and 128 KiB at 2 000 — and the strip is what makes the mix read the
/// value rows once for [`MIX_BLOCK`] query rows at a time.
pub const STRIP_MAX: usize = 32;

/// The score floats one worker's strip buffer holds. The strip is sized so
/// `strip × n` stays inside it: 32 rows at 1 024 tokens, 16 at 2 048, 1 at 32 768.
pub const STRIP_AREA: usize = 32 * 1024;

/// Floats one worker owns: its strip's scores and, after them,
/// [`STRIP_MAX`] softmax denominators (the `1/l` of each row of the strip).
pub const STRIP_SLOT: usize = STRIP_AREA + STRIP_MAX;

/// Keys per vector pass in a score row (four eight-lane accumulators).
#[cfg(target_arch = "x86_64")]
const KEYS_PER_GROUP: usize = 4 * LANES;

/// Query rows per score tile: the register block of [`scores_tile`].
///
/// Six rows × sixteen keys is twelve `ymm` accumulators (two panel loads and one
/// query broadcast per head dimension), the same shape as the float GEMM's 6×16 tile
/// and for the same reason: sixteen keys amply amortise the panel load, six rows
/// amortise the broadcast, and twelve accumulators leave room for the live loads on
/// AVX2's sixteen registers. Measured shapes and their rates are in
/// `docs/design/s4.9-sota.md`.
pub const TILE_ROWS: usize = 6;

/// Keys per score tile: two `ymm` of eight lanes (one lane per key).
pub const TILE_KEYS: usize = 16;

/// Value rows the mix shares one value load between: six rows × two `ymm` of head
/// dimensions is twelve accumulators that share each value load, the same register
/// block as the score tile and for the same reason.
pub const MIX_BLOCK: usize = 6;
/// Keys per vector lane group on every tier: one lane per key, so a lane is one score.
pub const VEC: usize = 8;

/// Keys per vector accumulator — one `__m256` lane per key, so a lane is one score.
#[cfg(target_arch = "x86_64")]
const LANES: usize = VEC;

/// Head-dimension steps of lookahead for the panel prefetch in a score tile.
///
/// A tile's step `k` reads one 64-byte run at `panel[k*n + j]`, and consecutive steps
/// are `n*4` bytes apart — 2 KiB at 512 keys, 8 KiB at 2 048, past the stride this
/// core's hardware prefetcher follows. Eight rows of lookahead covers the L3 latency
/// at roughly six cycles a step.
#[cfg(target_arch = "x86_64")]
const SCORE_PF: usize = 8;

/// Whether the AVX2 kernels may run on this machine and tier.
#[cfg(target_arch = "x86_64")]
#[inline]
fn vector_ok() -> bool {
    crate::simd::have_batched()
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
    // S4.9g: the NEON kernel goes here — `vfmaq_f32` with the same lane meaning (one key
    // per lane, the head dimension folded inside it) and `vmath`'s `exp4` for the weights,
    // beside `scores_row_scalar` as the reference, exactly as the AVX2 one does on x86.
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

/// `out[r*out_stride + j] = scale · Σ_k q[r*q_stride + k]·panel[k*n + j]`, for every
/// row `r` in `0..rows` and every key `j` in `j0..j1`; one sequential FMA chain per
/// `(row, key)`, in head-dimension order.
///
/// This is the strip's `Q · Kᵀ`: row `r` of `q` is a query head of the strip, key `j`
/// is a column of the panel, and the returned score is `acc * scale` — exactly the
/// chain [`scores_row`] runs, for [`TILE_ROWS`] rows at a time. A tile loads
/// `panel[k][j..j+16]` once per head dimension and broadcasts `q[r][k]` for each of
/// its rows into two `ymm` accumulators, so a panel step feeds twelve fused
/// multiply-adds instead of two. Every lane is still one `(row, key)`, so the tile
/// shape, the tails and the vector width cannot change a result: the scalar twin is
/// the reference and the tests hold every shape to it.
///
/// `j0..j1` is the key range of the caller's *band union*: a strip of rows on a
/// sliding layer has overlapping windows, and the tile computes the union, leaving
/// the keys outside a particular row's band in that row's buffer unread (the soft-max
/// and the mix read only the row's own `band`). On a global layer the union is the
/// text.
#[allow(clippy::too_many_arguments)]
pub fn scores_tile(
    q: &[f32],
    q_stride: usize,
    panel: &[f32],
    n: usize,
    hd: usize,
    rows: usize,
    j0: usize,
    j1: usize,
    scale: f32,
    out: &mut [f32],
    out_stride: usize,
) {
    scores_tile_shaped(
        TILE_ROWS, TILE_KEYS, q, q_stride, panel, n, hd, rows, j0, j1, scale, out, out_stride,
    )
}

/// [`scores_tile`] at an explicit tile shape, for the tests and the bench that hold
/// every shape to the same scores. A shape wider than the strip or the key range is
/// narrowed to it; unsupported shapes fall back to the scalar twin (never used by the
/// engine).
#[allow(clippy::too_many_arguments)]
pub fn scores_tile_shaped(
    rows_tile: usize,
    keys_tile: usize,
    q: &[f32],
    q_stride: usize,
    panel: &[f32],
    n: usize,
    hd: usize,
    rows: usize,
    j0: usize,
    j1: usize,
    scale: f32,
    out: &mut [f32],
    out_stride: usize,
) {
    debug_assert!(j0 <= j1 && j1 <= n);
    debug_assert!(rows > 0);
    debug_assert!(q.len() >= rows * q_stride);
    debug_assert!(out.len() >= rows * out_stride);
    debug_assert!(panel.len() >= hd * n);
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (rows_tile, keys_tile);
    #[cfg(target_arch = "x86_64")]
    {
        let len = j1 - j0;
        // A key tile needs a whole tile's worth of keys to run on: below sixteen the
        // tile halves to eight, and below eight the range is a scalar tail (the edge
        // of a sliding window is at most seven keys wide, and the scalar chain *is*
        // the reference).
        let kt = if len >= keys_tile {
            keys_tile
        } else if keys_tile > 16 && len >= 16 {
            16
        } else if len >= VEC {
            8
        } else {
            0
        };
        if kt != 0 && vector_ok() {
            let rt = match rows_tile.min(rows) {
                1 => 1,
                2 | 3 => 2,
                4 | 5 => 4,
                6 | 7 => 6,
                _ => 8,
            };
            // SAFETY: AVX2 + FMA checked; the shape is at most the strip and the key
            // range holds a whole key tile.
            unsafe {
                scores_tile_avx2(
                    rt, kt, q, q_stride, panel, n, hd, rows, j0, j1, scale, out, out_stride,
                )
            };
            return;
        }
    }
    scores_tile_scalar(
        q, q_stride, panel, n, hd, rows, j0, j1, scale, out, out_stride,
    )
}

/// [`scores_tile`]'s scalar twin: the reference, and the only path on a scalar tier.
#[allow(clippy::too_many_arguments)]
pub fn scores_tile_scalar(
    q: &[f32],
    q_stride: usize,
    panel: &[f32],
    n: usize,
    hd: usize,
    rows: usize,
    j0: usize,
    j1: usize,
    scale: f32,
    out: &mut [f32],
    out_stride: usize,
) {
    for r in 0..rows {
        let qr = &q[r * q_stride..];
        let s = &mut out[r * out_stride..];
        for j in j0..j1 {
            let mut acc = 0.0f32;
            for k in 0..hd {
                acc = qr[k].mul_add(panel[k * n + j], acc);
            }
            s[j] = acc * scale;
        }
    }
}

/// The tile dispatcher: one monomorphisation per measured shape.
///
/// # Safety
/// AVX2 and FMA must be available; the bounds are those of [`scores_tile`], and callers
/// have already narrowed the shape to the strip and to the key range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn scores_tile_avx2(
    rows_tile: usize,
    keys_tile: usize,
    q: &[f32],
    q_stride: usize,
    panel: &[f32],
    n: usize,
    hd: usize,
    rows: usize,
    j0: usize,
    j1: usize,
    scale: f32,
    out: &mut [f32],
    out_stride: usize,
) {
    macro_rules! run {
        ($rt:literal, $kt:literal) => {
            score_tile::<$rt, { $kt / VEC }>(
                q, q_stride, panel, n, hd, rows, j0, j1, scale, out, out_stride,
            )
        };
    }
    match (rows_tile, keys_tile) {
        (1, 8) => run!(1, 8),
        (1, 16) => run!(1, 16),
        (1, 24) => run!(1, 24),
        (1, 32) => run!(1, 32),
        (2, 8) => run!(2, 8),
        (2, 16) => run!(2, 16),
        (2, 24) => run!(2, 24),
        (2, 32) => run!(2, 32),
        (4, 8) => run!(4, 8),
        (4, 16) => run!(4, 16),
        (4, 24) => run!(4, 24),
        (4, 32) => run!(4, 32),
        (6, 8) => run!(6, 8),
        (6, 16) => run!(6, 16),
        (6, 24) => run!(6, 24),
        (6, 32) => run!(6, 32),
        (8, 8) => run!(8, 8),
        (8, 16) => run!(8, 16),
        _ => scores_tile_scalar(
            q, q_stride, panel, n, hd, rows, j0, j1, scale, out, out_stride,
        ),
    }
}

/// One [`scores_tile`] shape: `R` rows × `W` keys, `R*W/8` accumulators.
///
/// Strips are walked in whole `R`-row tiles; the tail narrows to four, two and one row
/// (the same chains, at a smaller block) so a strip of 16 or 32 rows never computes a
/// row it throws away. The last key tile of a range overlaps the previous one, as
/// [`scores_row`] does: it recomputes up to `W-1` keys with the same arithmetic into
/// the same slots, and the soft-max reads only the row's own band.
///
/// # Safety
/// AVX2 and FMA must be available; `j1 - j0 >= W`, `rows > 0`, `q.len() >= rows *
/// q_stride`, `out.len() >= rows * out_stride`, `panel.len() >= hd * n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_tile<const R: usize, const NV: usize>(
    q: &[f32],
    q_stride: usize,
    panel: &[f32],
    n: usize,
    hd: usize,
    rows: usize,
    j0: usize,
    j1: usize,
    scale: f32,
    out: &mut [f32],
    out_stride: usize,
) {
    let w = NV * VEC;
    debug_assert!(j1 - j0 >= w && rows > 0);
    let s = _mm256_set1_ps(scale);
    let pp = panel.as_ptr();
    let mut r0 = 0;
    while r0 + R <= rows {
        score_block::<R, NV>(
            q.as_ptr().add(r0 * q_stride),
            q_stride,
            pp,
            n,
            hd,
            j0,
            j1,
            s,
            out.as_mut_ptr().add(r0 * out_stride),
            out_stride,
            R,
        );
        r0 += R;
    }
    let rn = rows - r0;
    if rn > 0 {
        // The tail rows run at a smaller block: a whole R-row tile for the last two
        // rows of a strip would be work thrown away, and a two-row block still halves
        // the panel loads.
        let qs = &q[r0 * q_stride..];
        let os = &mut out[r0 * out_stride..];
        if rn >= 4 && R > 4 {
            score_tile::<4, NV>(
                qs, q_stride, panel, n, hd, rn, j0, j1, scale, os, out_stride,
            );
        } else if rn >= 2 && R > 2 {
            score_tile::<2, NV>(
                qs, q_stride, panel, n, hd, rn, j0, j1, scale, os, out_stride,
            );
        } else {
            score_tile::<1, NV>(
                qs, q_stride, panel, n, hd, rn, j0, j1, scale, os, out_stride,
            );
        }
    }
}

/// One register-blocked step of [`score_tile`]: `R` rows × `NV` vectors of keys, every
/// `(row, key)` its own head-dimension chain.
///
/// `stored` is the number of rows to write back (equal to `R` except in the tail,
/// where the missing rows are duplicates of the last one and are discarded).
///
/// # Safety
/// AVX2 and FMA must be available; pointers are those of [`score_tile`] with
/// `stored <= R` and `j0 + NV*VEC <= j1`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_block<const R: usize, const NV: usize>(
    q: *const f32,
    q_stride: usize,
    panel: *const f32,
    n: usize,
    hd: usize,
    j0: usize,
    j1: usize,
    scale: __m256,
    out: *mut f32,
    out_stride: usize,
    stored: usize,
) {
    // A row pointer per accumulator row; the tail repeats the last row and discards
    // its lanes (a few multiplies, and every load stays in bounds).
    let mut qp = [q; R];
    for (r, p) in qp.iter_mut().enumerate() {
        *p = q.add(r.min(stored - 1) * q_stride);
    }
    let mut j = j0;
    loop {
        let mut acc = [[_mm256_setzero_ps(); NV]; R];
        for k in 0..hd {
            if SCORE_PF != 0 && k + SCORE_PF < hd {
                // One line per head dimension; `_mm_prefetch` never faults.
                _mm_prefetch(panel.add((k + SCORE_PF) * n + j) as *const i8, _MM_HINT_T0);
            }
            let p = panel.add(k * n + j);
            let mut w = [_mm256_setzero_ps(); NV];
            for (v, wv) in w.iter_mut().enumerate() {
                *wv = _mm256_loadu_ps(p.add(v * VEC));
            }
            for (r, qr) in qp.iter().enumerate() {
                let qk = _mm256_broadcast_ss(&*qr.add(k));
                for (v, wv) in w.iter().enumerate() {
                    acc[r][v] = _mm256_fmadd_ps(qk, *wv, acc[r][v]);
                }
            }
        }
        for r in 0..stored {
            let o = out.add(r * out_stride + j);
            for (v, a) in acc[r].iter().enumerate() {
                _mm256_storeu_ps(o.add(v * VEC), _mm256_mul_ps(*a, scale));
            }
        }
        if j + 2 * NV * VEC <= j1 {
            j += NV * VEC;
        } else if j + NV * VEC >= j1 {
            break;
        } else {
            // One overlapping tile to the end of the range.
            j = j1 - NV * VEC;
        }
    }
}

/// The maximum of `x[lo..hi]`, the soft-max's shift.
///
/// A maximum is exact — associative and commutative on finite values — so the vector
/// path is bit-identical to the scalar fold it replaces; this is not a change of
/// arithmetic, only of how many cycles the reduction costs (a scalar `f32::max` fold
/// is a four-cycle dependency chain per key, one of the strip's larger non-FMA costs).
/// A NaN in a score would make the two paths differ in the same way the FPU's
/// `maxps` and `f32::max` always do; real scores are finite.
pub fn max_row(x: &[f32], lo: usize, hi: usize) -> f32 {
    debug_assert!(lo <= hi && hi <= x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 + FMA checked; the loop stays inside `lo..hi`.
            return unsafe { max_row_avx2(x, lo, hi) };
        }
    }
    max_row_scalar(x, lo, hi)
}

/// [`max_row`]'s scalar twin.
pub fn max_row_scalar(x: &[f32], lo: usize, hi: usize) -> f32 {
    x[lo..hi].iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// # Safety
/// AVX2 and FMA must be available; `lo <= hi <= x.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn max_row_avx2(x: &[f32], lo: usize, hi: usize) -> f32 {
    let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut j = lo;
    while j + 4 * LANES <= hi {
        let p = x.as_ptr().add(j);
        m = _mm256_max_ps(m, _mm256_loadu_ps(p));
        m = _mm256_max_ps(m, _mm256_loadu_ps(p.add(LANES)));
        m = _mm256_max_ps(m, _mm256_loadu_ps(p.add(2 * LANES)));
        m = _mm256_max_ps(m, _mm256_loadu_ps(p.add(3 * LANES)));
        j += 4 * LANES;
    }
    while j + LANES <= hi {
        m = _mm256_max_ps(m, _mm256_loadu_ps(x.as_ptr().add(j)));
        j += LANES;
    }
    // The lanes are a maximum over the whole range by now (this fold is exact), so
    // the horizontal reduction can take any order.
    let mut v = [0.0f32; LANES];
    _mm256_storeu_ps(v.as_mut_ptr(), m);
    let mut best = f32::NEG_INFINITY;
    for &e in &v {
        best = best.max(e);
    }
    while j < hi {
        best = best.max(*x.get_unchecked(j));
        j += 1;
    }
    best
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
                mix_rows_blocks_avx2(
                    MIX_BLOCK, e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out,
                    out_stride, obase,
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

/// [`mix_rows`] at an explicit row block, for the tests and the bench. `rows_block` is
/// how many query rows share one value load; unsupported values fall back to the
/// scalar twin.
#[allow(clippy::too_many_arguments)]
pub fn mix_rows_shaped(
    rows_block: usize,
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
    #[cfg(not(target_arch = "x86_64"))]
    let _ = rows_block;
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 + FMA checked; the bounds asserted above cover every row,
            // key and head dimension the block's clamped tails read.
            unsafe {
                mix_rows_blocks_avx2(
                    rows_block, e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out,
                    out_stride, obase,
                )
            };
            return;
        }
    }
    mix_rows_scalar(
        e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out, out_stride, obase,
    );
}

/// # Safety
/// AVX2 and FMA must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_rows_blocks_avx2(
    rows_block: usize,
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
    macro_rules! run {
        ($rb:literal, $vv:literal) => {
            mix_row_blocks::<$rb, $vv>(
                e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out, out_stride, obase,
            )
        };
    }
    match rows_block {
        1 => run!(1, 1),
        2 => run!(2, 2),
        4 => run!(4, 2),
        // Twelve rows × one `ymm`: half the value traffic, a value load for every
        // twelve fused multiply-adds instead of every six.
        12 => run!(12, 1),
        _ => run!(6, 2),
    }
}

/// [`mix_rows`] in blocks of `R` rows that share one value load, `V` `ymm` of
/// head-dimension lanes each; the tail narrows through eight, four, two and one row so a
/// strip of 16 or 32 rows never computes a row it discards.
///
/// # Safety
/// AVX2 and FMA must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_row_blocks<const R: usize, const V: usize>(
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
    let mut r0 = 0;
    while r0 + R <= rows {
        mix_block::<R, V>(
            e, e_stride, i0, n, half, v, v_stride, base, hd, inv, out, out_stride, obase, r0, R,
        );
        r0 += R;
    }
    if r0 == rows {
        return;
    }
    if R > 8 && rows - r0 >= 8 {
        mix_block::<8, V>(
            e, e_stride, i0, n, half, v, v_stride, base, hd, inv, out, out_stride, obase, r0, 8,
        );
        r0 += 8;
    }
    if R > 4 && rows - r0 >= 4 {
        mix_block::<4, V>(
            e, e_stride, i0, n, half, v, v_stride, base, hd, inv, out, out_stride, obase, r0, 4,
        );
        r0 += 4;
    }
    if rows - r0 >= 2 {
        mix_block::<2, V>(
            e, e_stride, i0, n, half, v, v_stride, base, hd, inv, out, out_stride, obase, r0, 2,
        );
        r0 += 2;
    }
    if r0 < rows {
        mix_block::<1, V>(
            e, e_stride, i0, n, half, v, v_stride, base, hd, inv, out, out_stride, obase, r0, 1,
        );
    }
}

/// `R` rows at a time share one value load; `V` accumulators (`8·V` head-dimension
/// lanes) per row, so a value row is loaded once for every `R` query rows that read it.
///
/// The value traffic is what bounds this stage at long texts — a strip re-reads the
/// text's value rows once per row block, so `R` rows per load is `R` times fewer bytes
/// per multiply-add — and the register budget decides the pair: `V = 2` with `R = 6`
/// has the better load-per-multiply ratio (eight loads per twelve fused multiply-adds),
/// `V = 1` with `R = 12` has half the value traffic (thirteen loads per twelve) and is
/// the one to measure against it.
///
/// The core is the key range every row of the block sees; the edges are the (at most
/// `R - 1`) keys at each end that only some rows see. Accumulating the low edge, then
/// the core, then the high edge keeps every row's fold in ascending key order, which is
/// the defined order. Rows past `rn` (the tail of a strip) are duplicates of the last
/// real row: computed and discarded, never stored.
///
/// # Safety
/// AVX2 and FMA must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_block<const R: usize, const V: usize>(
    e: &[f32],
    e_stride: usize,
    i0: usize,
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
    r0: usize,
    rn: usize,
) {
    debug_assert!(rn > 0 && rn <= R);
    let ep = e.as_ptr();
    let vp = v.as_ptr();
    let mut bands = [(0usize, 0usize); R];
    for (r, b) in bands.iter_mut().enumerate() {
        *b = band(i0 + r0 + r.min(rn - 1), n, half);
    }
    let core_lo = bands.iter().map(|b| b.0).max().unwrap_or(0);
    let core_hi = bands.iter().map(|b| b.1).min().unwrap_or(0);
    if core_lo > core_hi {
        // The block's windows have no key in common — a window narrower than the row
        // block (`half < (R - 2) / 2`), which no real family has. One row at a time,
        // still vectorised over the head dimension.
        for r in 0..rn {
            mix_block::<1, V>(
                e,
                e_stride,
                i0,
                n,
                half,
                v,
                v_stride,
                base,
                hd,
                inv,
                out,
                out_stride,
                obase,
                r0 + r,
                1,
            );
        }
        return;
    }
    let step = V * VEC;
    let mut d0 = 0;
    while d0 + step <= hd {
        let mut acc = [[_mm256_setzero_ps(); V]; R];
        for r in 0..R {
            let er = ep.add((r0 + r.min(rn - 1)) * e_stride);
            for j in bands[r].0..core_lo {
                let ev = _mm256_broadcast_ss(&*er.add(j));
                let p = vp.add(j * v_stride + base + d0);
                for (vv, a) in acc[r].iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(ev, _mm256_loadu_ps(p.add(vv * VEC)), *a);
                }
            }
        }
        for j in core_lo..core_hi {
            let p = vp.add(j * v_stride + base + d0);
            let mut val = [_mm256_setzero_ps(); V];
            for (vv, w) in val.iter_mut().enumerate() {
                *w = _mm256_loadu_ps(p.add(vv * VEC));
            }
            for r in 0..R {
                let ev = _mm256_broadcast_ss(&*ep.add((r0 + r.min(rn - 1)) * e_stride + j));
                for (vv, w) in val.iter().enumerate() {
                    acc[r][vv] = _mm256_fmadd_ps(ev, *w, acc[r][vv]);
                }
            }
        }
        for r in 0..R {
            let er = ep.add((r0 + r.min(rn - 1)) * e_stride);
            for j in core_hi..bands[r].1 {
                let ev = _mm256_broadcast_ss(&*er.add(j));
                let p = vp.add(j * v_stride + base + d0);
                for (vv, a) in acc[r].iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(ev, _mm256_loadu_ps(p.add(vv * VEC)), *a);
                }
            }
        }
        for r in 0..rn {
            let o = out.as_mut_ptr().add((r0 + r) * out_stride + obase + d0);
            let k = _mm256_set1_ps(*inv.get_unchecked(r0 + r));
            for (vv, a) in acc[r].iter().enumerate() {
                _mm256_storeu_ps(o.add(vv * VEC), _mm256_mul_ps(*a, k));
            }
        }
        d0 += step;
    }
    if d0 < hd {
        // Head dimensions narrower than one accumulator pair: scalar, in the same
        // ascending order as the reference (the loop's order over `d` is free).
        for r in 0..rn {
            let (lo, hi) = bands[r];
            let er = ep.add((r0 + r) * e_stride);
            let o = out.as_mut_ptr().add((r0 + r) * out_stride + obase);
            for d in d0..hd {
                let mut a = 0.0f32;
                for j in lo..hi {
                    a = (*er.add(j)).mul_add(*vp.add(j * v_stride + base + d), a);
                }
                *o.add(d) = a * *inv.get_unchecked(r0 + r);
            }
        }
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

    /// The tile shapes the tests hold to the same scores: the engine's (6 × 16), a
    /// single row, two narrower row blocks, a wider one and two wider key blocks.
    const TILES: [(usize, usize); 8] = [
        (1, 8),
        (1, 16),
        (2, 8),
        (4, 16),
        (6, 16),
        (8, 16),
        (6, 24),
        (4, 32),
    ];

    /// The engine's shapes (what `blocks/bidir.rs` runs).
    const ENGINE_TILE: (usize, usize) = (TILE_ROWS, TILE_KEYS);
    const ENGINE_MIX: usize = MIX_BLOCK;

    /// One text's attention through the kernels, exactly as `blocks/bidir.rs` drives
    /// them: the panel per KV head, then strips of `strip` query rows scored with the
    /// `tile` shape, soft-maxed and mixed while hot by `mix` rows at a time.
    ///
    /// `scalar` takes the reference path: every row scored alone over its exact band by
    /// [`scores_row_scalar`], the maximum a scalar fold, `exp_row_scalar`, the mix
    /// scalar. The tile path scores the *union* of the strip's bands, as the engine
    /// does, so the two together also pin "a key outside a row's band changes nothing".
    #[allow(clippy::too_many_arguments)]
    fn attend(
        n: usize,
        hd: usize,
        n_head: usize,
        n_kv: usize,
        half: Option<usize>,
        strip: usize,
        tile: (usize, usize),
        mix: usize,
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
                    if scalar {
                        for r in 0..rows {
                            let i = r0 + r;
                            let (lo, hi) = band(i, n, half);
                            let s = &mut scores[r * n..(r + 1) * n];
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
                            let max = max_row_scalar(s, lo, hi);
                            exp_row_scalar(s, lo, hi, max);
                            let l = sum_row(s, lo, hi);
                            inv[r] = if l > 0.0 { (1.0 / l) as f32 } else { 0.0 };
                        }
                    } else {
                        let u_lo = band(r0, n, half).0;
                        let u_hi = band(r0 + rows - 1, n, half).1;
                        let qs = &q[r0 * qd + h * hd..];
                        let mut j0 = u_lo;
                        while j0 < u_hi {
                            let j1 = (j0 + SCORE_BLOCK).min(u_hi);
                            scores_tile_shaped(
                                tile.0,
                                tile.1,
                                qs,
                                qd,
                                &panel,
                                n,
                                hd,
                                rows,
                                j0,
                                j1,
                                scale,
                                &mut scores,
                                n,
                            );
                            j0 = j1;
                        }
                        for r in 0..rows {
                            let (lo, hi) = band(r0 + r, n, half);
                            let s = &mut scores[r * n..(r + 1) * n];
                            let max = max_row(s, lo, hi);
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
                        mix_rows_shaped(
                            mix,
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
                        let want = attend(
                            n,
                            hd,
                            1,
                            1,
                            half,
                            strip,
                            ENGINE_TILE,
                            ENGINE_MIX,
                            &q,
                            &k,
                            &v,
                            true,
                        );
                        let got = attend(
                            n,
                            hd,
                            1,
                            1,
                            half,
                            strip,
                            ENGINE_TILE,
                            ENGINE_MIX,
                            &q,
                            &k,
                            &v,
                            false,
                        );
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
                    let full = attend(
                        n,
                        hd,
                        1,
                        1,
                        half,
                        n,
                        ENGINE_TILE,
                        ENGINE_MIX,
                        &q,
                        &k,
                        &v,
                        false,
                    );
                    for strip in [1usize, 2, 5, 16, 32] {
                        let got = attend(
                            n,
                            hd,
                            1,
                            1,
                            half,
                            strip,
                            ENGINE_TILE,
                            ENGINE_MIX,
                            &q,
                            &k,
                            &v,
                            false,
                        );
                        same(
                            &got,
                            &full,
                            &format!("hd {hd} n {n} half {half:?} strip {strip}"),
                        );
                    }
                    // One row alone: a strip of exactly that row, at every position.
                    for i in 0..n {
                        let alone = attend(
                            n,
                            hd,
                            1,
                            1,
                            half,
                            1,
                            ENGINE_TILE,
                            ENGINE_MIX,
                            &q,
                            &k,
                            &v,
                            false,
                        );
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
                let want = attend(
                    n,
                    hd,
                    n_head,
                    n_kv,
                    half,
                    16,
                    ENGINE_TILE,
                    ENGINE_MIX,
                    &q,
                    &k,
                    &v,
                    true,
                );
                let got = attend(
                    n,
                    hd,
                    n_head,
                    n_kv,
                    half,
                    16,
                    ENGINE_TILE,
                    ENGINE_MIX,
                    &q,
                    &k,
                    &v,
                    false,
                );
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
                    let one = attend(
                        n,
                        hd,
                        1,
                        1,
                        half,
                        16,
                        ENGINE_TILE,
                        ENGINE_MIX,
                        &qh,
                        &kk,
                        &vv,
                        false,
                    );
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

    #[test]
    fn a_tile_shape_never_changes_a_score() {
        for &hd in &HDS {
            for &n in &[7usize, 16, 33, 64, 129] {
                for &half in &[None, Some(0), Some(3), Some(64)] {
                    let q = prng(0x7a11 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                    let k = prng(0x7a12 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                    let v = prng(0x7a13 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                    for strip in [7usize, 16, 32] {
                        let want = attend(
                            n,
                            hd,
                            1,
                            1,
                            half,
                            strip,
                            ENGINE_TILE,
                            ENGINE_MIX,
                            &q,
                            &k,
                            &v,
                            true,
                        );
                        for &tile in &TILES {
                            let got = attend(
                                n, hd, 1, 1, half, strip, tile, ENGINE_MIX, &q, &k, &v, false,
                            );
                            same(
                                &got,
                                &want,
                                &format!("hd {hd} n {n} half {half:?} strip {strip} tile {tile:?}"),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_mix_block_never_changes_a_row() {
        for &hd in &[16usize, 32, 64, 256] {
            for &n in &[16usize, 33, 128] {
                for &half in &[None, Some(5)] {
                    let q = prng(0x4b10 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                    let k = prng(0x4b11 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                    let v = prng(0x4b12 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                    let want = attend(
                        n,
                        hd,
                        1,
                        1,
                        half,
                        16,
                        ENGINE_TILE,
                        ENGINE_MIX,
                        &q,
                        &k,
                        &v,
                        true,
                    );
                    for &mix in &[1usize, 2, 4, 6, 12] {
                        let got =
                            attend(n, hd, 1, 1, half, 16, ENGINE_TILE, mix, &q, &k, &v, false);
                        same(
                            &got,
                            &want,
                            &format!("hd {hd} n {n} half {half:?} mix {mix}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_band_size_matches_the_scalar_twin() {
        let n = 40;
        for &hd in &[16usize, 32, 256] {
            let q = prng(0x8bad ^ hd as u64, n * hd, -1.0, 1.0);
            let k = prng(0x8bae ^ hd as u64, n * hd, -1.0, 1.0);
            let v = prng(0x8baf ^ hd as u64, n * hd, -1.0, 1.0);
            for half in (0..=n).map(Some).chain([None]) {
                for &strip in &[1usize, 7, 16, 32] {
                    let want = attend(
                        n,
                        hd,
                        1,
                        1,
                        half,
                        strip,
                        ENGINE_TILE,
                        ENGINE_MIX,
                        &q,
                        &k,
                        &v,
                        true,
                    );
                    let got = attend(
                        n,
                        hd,
                        1,
                        1,
                        half,
                        strip,
                        ENGINE_TILE,
                        ENGINE_MIX,
                        &q,
                        &k,
                        &v,
                        false,
                    );
                    same(&got, &want, &format!("hd {hd} half {half:?} strip {strip}"));
                }
            }
        }
    }

    #[test]
    fn the_tile_runs_the_rows_own_chain() {
        // A tile of one row over the row's band is exactly `scores_row`, and the tile's
        // scalar twin is `scores_row_scalar`: the tile is a scheduling change.
        for &hd in &HDS {
            for &n in &[16usize, 33, 129] {
                let q = prng(0x1a11 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                let k = prng(0x1a12 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                let mut panel = vec![0.0f32; hd * n];
                key_panel(&k, hd, 0, hd, n, &mut panel);
                let mut a = vec![0.0f32; n];
                let mut b = vec![0.0f32; n];
                let mut c = vec![0.0f32; n];
                for i in [0usize, 1, n / 2, n - 1] {
                    let (lo, hi) = band(i, n, Some(3));
                    scores_row(&q[i * hd..], &panel, n, hd, lo, hi, 0.5, &mut a);
                    scores_tile_shaped(
                        1,
                        8,
                        &q[i * hd..],
                        hd,
                        &panel,
                        n,
                        hd,
                        1,
                        lo,
                        hi,
                        0.5,
                        &mut b,
                        n,
                    );
                    scores_tile_scalar(&q[i * hd..], hd, &panel, n, hd, 1, lo, hi, 0.5, &mut c, n);
                    same(
                        &b[lo..hi],
                        &a[lo..hi],
                        &format!("tile hd {hd} n {n} row {i}"),
                    );
                    same(
                        &c[lo..hi],
                        &a[lo..hi],
                        &format!("scalar hd {hd} n {n} row {i}"),
                    );
                }
            }
        }
    }

    #[test]
    fn the_maximum_is_the_same_value() {
        for &n in &[1usize, 7, 8, 9, 33, 100] {
            let x = prng(0x3c3c ^ n as u64, n, -4.0, 4.0);
            for lo in 0..n {
                for hi in lo..=n {
                    let want = x[lo..hi].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    assert_eq!(
                        max_row(&x, lo, hi).to_bits(),
                        want.to_bits(),
                        "n {n} lo {lo} hi {hi}"
                    );
                }
            }
        }
    }
}
