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
//! 1. [`key_panel`] transposes one head's keys into a key-major panel — on x86_64
//!    tiles of [`PANEL_TILE`] keys, each head dimension's sixteen keys contiguous — so
//!    a score tile walks the panel front to back instead of one 64-byte run every
//!    `n × 4` bytes (S4.9g part 3). It is built once per `(text, KV head)` and serves
//!    every query head of that KV head.
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
//! 4. [`mix_rows`] accumulates `P · V` for a strip of query rows, ascending key order
//!    inside each row, `fma(p[j], v[j][d], acc[d])`. At long texts it walks the keys
//!    in blocks of [`MIX_KEY_BLOCK`] for every row block of the strip before moving
//!    on, so a text's value rows are read from L3 once per strip rather than once per
//!    row block (S4.9g part 3); the accumulator lives in the output rows between
//!    blocks, and the fold is still ascending.
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
//! once per strip in key blocks ([`SCORE_BLOCK`] keys).
//!
//! # aarch64
//!
//! The scalar twins compile everywhere and are the reference. The NEON kernels are
//! beside them: `vfmaq_f32` with one key per lane (four lanes, so a 4×16 tile),
//! `vmaxq_f32` for the soft-max's exact maximum, `vmath::exp4` for the weights
//! (bit-identical to `exp_scalar`), and a six-row value mix with two `float32x4_t`
//! per row. Every result is bit-identical to the scalar twins.
//!
//! Part 3's long-text layout is not x86_64's alone: aarch64 builds the same key-major
//! 16-key tiles, scores them with a 4-row × 16-key NEON tile (one key per lane), and
//! walks the value mix in 128-key blocks with a six-row NEON block. The defined
//! arithmetic is unchanged, so every tier's bits stay the scalar twins'.

use crate::vmath::exp_scalar;

#[cfg(target_arch = "aarch64")]
use crate::vmath::exp4;
#[cfg(target_arch = "x86_64")]
use crate::vmath::exp8;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Keys per strip block: what bounds the panel slice one row of a strip walks, so a
/// strip re-walks an L2-sized slice instead of the whole panel.
/// With the key-tiled panel (S4.9g part 3) the traversal is sequential, so the block is
/// only a call-granularity choice; 128 keys keeps a block's panel slice at 32-128 KiB
/// on every head the engine supports.
pub const SCORE_BLOCK: usize = 128;

/// Query rows per strip, at most (S4.9g part 3). A strip's scores are `strip × n` f32 —
/// 128 KiB at 512 tokens and 512 KiB at 2 000 with the 64-row cap — and the strip is
/// what makes the panel and the value rows be read once for all of its rows.
pub const STRIP_MAX: usize = 64;

/// The score floats one worker's strip buffer holds. The strip is sized so
/// `strip × n` stays inside it: 64 rows at 4 096 tokens, 32 at 8 192, 8 at 32 768
/// (1 MiB per worker). Part 2's 32 Ki (128 KiB) left a 2 048-token text at 16 rows,
/// which re-read the panel and the values 128 times per head.
pub const STRIP_AREA: usize = 256 * 1024;

/// Keys per key-major panel tile (S4.9g part 3): the score tile's sixteen keys are
/// contiguous for every head dimension, so a strip walks the panel front to back
/// instead of one 64-byte run every `n × 4` bytes. The last `n % 16` keys of a text are
/// stored row-major (one `[head_dim][rem]` region) so no key is padded. x86_64 and
/// aarch64 share the layout; only the tile's vector width differs.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub const PANEL_TILE: usize = 16;

/// Keys per value-mix key block (S4.9g part 3): one block's value rows —
/// `128 × head_dim × 4` bytes, 32 KiB at a 64-wide head and 128 KiB at 256 — stay in
/// L1/L2 while every row block of a strip walks them, so the text's values are read
/// from L3 once per strip instead of once per row block.
pub const MIX_KEY_BLOCK: usize = 128;

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
/// AVX2's sixteen registers. NEON is four lanes wide and has thirty-two registers, so
/// its tile is four rows × sixteen keys (sixteen accumulators). Measured shapes and
/// their rates are in `docs/design/s4.9-sota.md`.
#[cfg(target_arch = "x86_64")]
pub const TILE_ROWS: usize = 6;
#[cfg(target_arch = "aarch64")]
pub const TILE_ROWS: usize = 4;

/// Keys per score tile: four vectors of one lane per key (two `ymm` in AVX2, four
/// `float32x4_t` on NEON).
pub const TILE_KEYS: usize = 16;

/// Value rows the mix shares one value load between: six rows × two vectors of head
/// dimensions is twelve accumulators that share each value load, the same register
/// block as the score tile and for the same reason.
pub const MIX_BLOCK: usize = 6;
/// Keys per vector lane group on every tier: one lane per key, so a lane is one score.
#[cfg(target_arch = "x86_64")]
pub const VEC: usize = 8;
/// Four lanes on NEON, one key each.
#[cfg(target_arch = "aarch64")]
pub const VEC: usize = 4;

/// Keys per vector accumulator — one `__m256` lane per key, so a lane is one score.
#[cfg(target_arch = "x86_64")]
const LANES: usize = VEC;
/// Four lanes on NEON.
#[cfg(target_arch = "aarch64")]
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

/// Whether the NEON kernels may run on this machine and tier.
#[cfg(target_arch = "aarch64")]
#[inline]
fn vector_ok() -> bool {
    crate::cpu::have_neon()
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

/// llama.cpp's ALiBi slopes: the per-head scale of the `-|i - j|` bias.
///
/// Transcribed from the CPU soft-max (`ggml/src/ggml-cpu/ops.cpp`, the
/// `f_max_alibi_bias > 0` path): with `n_head_log2 = 1 << floor(log2(n_head))`,
/// `m0 = 2^(-max_bias/n_head_log2)` and `m1 = 2^(-(max_bias/2)/n_head_log2)`, head `h`
/// gets `m0^(h+1)` when `h < n_head_log2` and `m1^(2·(h - n_head_log2) + 1)` otherwise.
/// llama.cpp calls the C library's `powf` here. The powers go through
/// [`crate::vmath::det_pow_f32`] instead, so the slopes (and every score they touch) do
/// not depend on the platform's C library: the same split as the other `det_*` helpers.
/// They are built once at load, not in a pass.
///
/// `jina-bert-v2` is the family that asks for this (`f_max_alibi_bias = 8`,
/// `src/models/jina-bert-v2.cpp`), but the bias is generic and the kernel does not
/// know which model it serves.
pub fn alibi_slopes(n_head: usize, max_bias: f32) -> Vec<f32> {
    assert!(n_head > 0, "ALiBi needs at least one attention head");
    assert!(
        max_bias.is_finite() && max_bias > 0.0,
        "ALiBi's max_bias must be positive, got {max_bias}"
    );
    let n_head_log2 = 1usize << (usize::BITS - 1 - n_head.leading_zeros());
    let l = n_head_log2 as f32;
    let pow = crate::vmath::det_pow_f32;
    let m0 = pow(2.0, -max_bias / l);
    let m1 = pow(2.0, -(max_bias / 2.0) / l);
    (0..n_head)
        .map(|h| {
            if h < n_head_log2 {
                pow(m0, h as f32 + 1.0)
            } else {
                pow(m1, 2.0 * (h - n_head_log2) as f32 + 1.0)
            }
        })
        .collect()
}

/// Add ALiBi's bias to one query row's scores: `x[j] += slope · (-|j - qpos|)`,
/// `j` in `lo..hi`.
///
/// The order is llama.cpp's: the soft-max scales the score first (`scores_tile` has
/// already done that), then adds `slope · mask`, where the mask entry is
/// `-|p0 - p1|` as an `f32` (`src/llama-graph.cpp`, `llm_graph_input_attn_no_cache`)
/// and the slope is [`alibi_slopes`]'s. Only a row's own band is touched: the keys
/// outside it hold whatever the score tile left there and are never read.
pub fn alibi_row(x: &mut [f32], lo: usize, hi: usize, qpos: usize, slope: f32) {
    assert!(lo <= hi && hi <= x.len(), "ALiBi band outside the row");
    assert!(slope.is_finite(), "ALiBi slope must be finite, got {slope}");
    for (j, s) in x.iter_mut().enumerate().take(hi).skip(lo) {
        *s += slope * -(j.abs_diff(qpos) as f32);
    }
}

/// One head's keys of one text, key-major on x86_64 and aarch64 (S4.9g part 3).
///
/// `keys` is the text's key rows (`n` rows of `stride` floats), `base` the KV head's
/// offset inside a row. The destination is tiled: the first `n - n % 16` keys are
/// stored in tiles of [`PANEL_TILE`] keys,
/// `panel[(j / 16) · head_dim · 16 + k · 16 + j % 16] = keys[j][k]`, so the sixteen keys
/// of a score tile are contiguous for every head dimension; the last `n % 16` keys are
/// row-major (`panel[(n - rem) · head_dim + k · rem + (j - n0)]`), so a text is never
/// padded. Other architectures keep the lane panel of S4.9g part 2 (`panel[k · n + j]`).
///
/// [`key_panel_tile`] builds one tile, which is what the driver's per-head fork-join
/// hands to each worker; [`key_panel`] is every tile in order.
pub fn key_panel(keys: &[f32], stride: usize, base: usize, hd: usize, n: usize, panel: &mut [f32]) {
    debug_assert!(panel.len() >= hd * n);
    debug_assert!(keys.len() >= n * stride);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        for t in 0..panel_tiles(n) {
            let (off, len) = panel_tile_span(n, hd, t);
            key_panel_tile(keys, stride, base, hd, n, t, &mut panel[off..off + len]);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    key_panel_rows(keys, stride, base, hd, n, 0, n, panel);
}

/// [`key_panel`]'s twin: the same layout, scalar on every tier (the build is a copy,
/// not arithmetic).
pub fn key_panel_scalar(
    keys: &[f32],
    stride: usize,
    base: usize,
    hd: usize,
    n: usize,
    panel: &mut [f32],
) {
    key_panel(keys, stride, base, hd, n, panel);
}

/// The lane panel of S4.9g part 2: `panel[k · n + j] = keys[j · stride + base + k]`,
/// keys `ja..jb`. The reference layout [`scores_row`] reads, and the tail region of the
/// key-major panel.
#[allow(clippy::too_many_arguments)]
pub fn key_panel_rows(
    keys: &[f32],
    stride: usize,
    base: usize,
    hd: usize,
    n: usize,
    ja: usize,
    jb: usize,
    panel: &mut [f32],
) {
    debug_assert!(jb <= n && panel.len() >= hd * n);
    debug_assert!(keys.len() >= n * stride);
    const T: usize = 16;
    let mut j0 = ja;
    while j0 < jb {
        let j1 = (j0 + T).min(jb);
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

/// The number of panel tiles of a text of `n` keys: `n / 16` full tiles plus, when
/// `n % 16 != 0`, one row-major tail tile.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn panel_tiles(n: usize) -> usize {
    n / PANEL_TILE + usize::from(n % PANEL_TILE != 0)
}

/// The `(offset, length)` of tile `t` in a text's panel of `n` keys. Full tiles are
/// `head_dim × 16` floats; the last one, when `n % 16 != 0`, is the row-major tail of
/// `head_dim × (n % 16)` floats.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn panel_tile_span(n: usize, hd: usize, t: usize) -> (usize, usize) {
    let n0 = n - n % PANEL_TILE;
    if t * PANEL_TILE < n0 {
        (t * PANEL_TILE * hd, PANEL_TILE * hd)
    } else {
        (n0 * hd, (n - n0) * hd)
    }
}

/// Fill tile `t` of a text's key-major panel (`dst` is [`panel_tile_span`]'s slice).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn key_panel_tile(
    keys: &[f32],
    stride: usize,
    base: usize,
    hd: usize,
    n: usize,
    t: usize,
    dst: &mut [f32],
) {
    let n0 = n - n % PANEL_TILE;
    let tile_start = t * PANEL_TILE;
    if tile_start < n0 {
        debug_assert!(dst.len() >= hd * PANEL_TILE);
        for k in 0..hd {
            let o = k * PANEL_TILE;
            for jj in 0..PANEL_TILE {
                dst[o + jj] = keys[(tile_start + jj) * stride + base + k];
            }
        }
    } else {
        let rem = n - n0;
        debug_assert!(dst.len() >= hd * rem);
        for k in 0..hd {
            let o = k * rem;
            for jj in 0..rem {
                dst[o + jj] = keys[(n0 + jj) * stride + base + k];
            }
        }
    }
}

/// Where key `j` of a text of `n` keys lives in the panel, for head dimension `k`
/// (the key-major layout; the tests use it to read a key back).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
pub fn panel_index(n: usize, hd: usize, j: usize, k: usize) -> usize {
    let n0 = n - n % PANEL_TILE;
    if j < n0 {
        (j / PANEL_TILE) * hd * PANEL_TILE + k * PANEL_TILE + j % PANEL_TILE
    } else {
        n0 * hd + k * (n - n0) + (j - n0)
    }
}

/// `out[j] = scale · Σ_k q[k]·panel[k][j]` for `j` in `lo..hi`, one sequential FMA
/// chain per key.
///
/// `panel` is the lane panel ([`key_panel_rows`]) with `n` keys; the strip uses it for
/// a text's `< 16`-key tail. `q` holds one row's head (at least `hd` floats); `out` is
/// that row's `n` scores.
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
    // Checked in every build: the vector kernel below reads and writes through raw
    // pointers, so these bounds are what makes this safe function sound.
    assert!(lo <= hi && hi <= n, "score range outside the keys");
    assert!(
        q.len() >= hd && panel.len() >= hd * n && out.len() >= hi,
        "score buffers shorter than the range"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 + FMA checked; every load and store is bounds-checked above.
            unsafe { scores_row_avx2(q, panel, n, hd, lo, hi, scale, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if vector_ok() {
            // SAFETY: NEON checked; every load and store is bounds-checked above.
            unsafe { scores_row_neon(q, panel, n, hd, lo, hi, scale, out) };
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

/// NEON's four accumulators, four keys each: the chain over the head dimension is the
/// inner loop, so each key's dependency is one FMA deep.
///
/// # Safety
/// NEON must be available; `lo..hi ⊆ 0..n`, `q.len() >= hd`, `panel.len() >= hd * n`,
/// `out.len() >= n`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn scores_row_neon(
    q: &[f32],
    panel: &[f32],
    n: usize,
    hd: usize,
    lo: usize,
    hi: usize,
    scale: f32,
    out: &mut [f32],
) {
    let s = vdupq_n_f32(scale);
    macro_rules! group16 {
        ($j:expr) => {{
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            let mut a2 = vdupq_n_f32(0.0);
            let mut a3 = vdupq_n_f32(0.0);
            for k in 0..hd {
                let qk = vdupq_n_f32(*q.get_unchecked(k));
                let p = panel.as_ptr().add(k * n + $j);
                a0 = vfmaq_f32(a0, vld1q_f32(p), qk);
                a1 = vfmaq_f32(a1, vld1q_f32(p.add(4)), qk);
                a2 = vfmaq_f32(a2, vld1q_f32(p.add(8)), qk);
                a3 = vfmaq_f32(a3, vld1q_f32(p.add(12)), qk);
            }
            let o = out.as_mut_ptr().add($j);
            vst1q_f32(o, vmulq_f32(a0, s));
            vst1q_f32(o.add(4), vmulq_f32(a1, s));
            vst1q_f32(o.add(8), vmulq_f32(a2, s));
            vst1q_f32(o.add(12), vmulq_f32(a3, s));
        }};
    }
    macro_rules! group4 {
        ($j:expr) => {{
            let mut a0 = vdupq_n_f32(0.0);
            for k in 0..hd {
                let qk = vdupq_n_f32(*q.get_unchecked(k));
                let p = panel.as_ptr().add(k * n + $j);
                a0 = vfmaq_f32(a0, vld1q_f32(p), qk);
            }
            vst1q_f32(out.as_mut_ptr().add($j), vmulq_f32(a0, s));
        }};
    }
    let mut j = lo;
    while j + 16 <= hi {
        group16!(j);
        j += 16;
    }
    while j + LANES <= hi {
        group4!(j);
        j += LANES;
    }
    if j < hi {
        if hi - lo >= LANES {
            // The last group overlaps the previous one: the keys it recomputes are the
            // same keys with the same arithmetic, and only lanes inside `hi` are read.
            group4!(hi - LANES);
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
/// text. On x86_64 the union is widened to whole [`PANEL_TILE`] tiles, which is the
/// same scheduling choice.
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

/// [`scores_tile`] at an explicit row block, for the tests and the bench that hold
/// every shape to the same scores. A shape wider than the strip is narrowed to it.
/// On x86_64 the key tile is the panel's own [`PANEL_TILE`], so `keys_tile` is not a
/// knob there; on aarch64 it picks the NEON tile's width.
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
    if rows == 0 {
        return;
    }
    // Checked in every build: the vector kernel reads and writes through raw pointers,
    // so these bounds are what makes this safe function sound. The last row needs only
    // its own head (`hd` floats) and its scores up to the highest key the kernel stores,
    // not a whole stride: callers pass slices that end with the buffer. On x86_64 and
    // aarch64 the key range is widened to whole 16-key panel tiles, so the vector kernel
    // stores up to the end of the tile that holds `j1`.
    assert!(j0 <= j1 && j1 <= n, "key range outside the keys");
    assert!(
        q.len() >= (rows - 1) * q_stride + hd,
        "query rows shorter than the strip"
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let store_hi = {
        let n0 = n - n % PANEL_TILE;
        let full_hi = j1.min(n0);
        if j0 < full_hi {
            j1.max(full_hi.div_ceil(PANEL_TILE) * PANEL_TILE)
        } else {
            j1
        }
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let store_hi = j1;
    assert!(
        out.len() >= (rows - 1) * out_stride + store_hi,
        "score rows shorter than the strip"
    );
    assert!(panel.len() >= hd * n, "key panel shorter than hd * n");
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = (rows_tile, keys_tile);
    #[cfg(target_arch = "x86_64")]
    let _ = keys_tile;
    #[cfg(target_arch = "aarch64")]
    {
        let _ = keys_tile;
        if vector_ok() {
            // Full 16-key tiles are key-major, so they walk the panel front to back;
            // the text's last `n % 16` keys are row-major and take the row kernel.
            let n0 = n - n % PANEL_TILE;
            let full_hi = j1.min(n0);
            if j0 < full_hi {
                let a0 = j0 - j0 % PANEL_TILE;
                let a1 = full_hi.div_ceil(PANEL_TILE) * PANEL_TILE;
                let rt = match rows_tile.min(rows) {
                    1 => 1,
                    2 | 3 => 2,
                    _ => 4,
                };
                // SAFETY: NEON checked; the bounds are `scores_tile`'s and the range was
                // widened to whole tiles inside the panel.
                unsafe {
                    scores_tile_tiled_neon(
                        rt, q, q_stride, panel, hd, rows, a0, a1, scale, out, out_stride,
                    )
                };
            }
            if j1 > n0 {
                let rem = n - n0;
                let lo = j0.saturating_sub(n0);
                let hi = j1 - n0;
                let tp = &panel[n0 * hd..];
                let to = &mut out[n0..];
                let len = hi - lo;
                // Below sixteen keys the row kernel halves to eight and then four; below
                // four the range is a scalar tail (a window's edge is at most three wide).
                let kt = if len >= 16 {
                    16
                } else if len >= 8 {
                    8
                } else if len >= VEC {
                    4
                } else {
                    0
                };
                if kt != 0 {
                    // SAFETY: NEON checked; the tail region is the lane panel of `rem`
                    // keys, which the row kernel reads within `hd * rem`.
                    unsafe {
                        scores_tile_neon(
                            rows_tile.min(rows).clamp(1, 4),
                            kt,
                            q,
                            q_stride,
                            tp,
                            rem,
                            hd,
                            rows,
                            lo,
                            hi,
                            scale,
                            to,
                            out_stride,
                        )
                    };
                } else {
                    for r in 0..rows {
                        let qr = &q[r * q_stride..];
                        let s = &mut to[r * out_stride..];
                        for j in lo..hi {
                            let mut acc = 0.0f32;
                            for k in 0..hd {
                                acc = qr[k].mul_add(tp[k * rem + j], acc);
                            }
                            s[j] = acc * scale;
                        }
                    }
                }
            }
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            let rt = match rows_tile.min(rows) {
                1 => 1,
                2 | 3 => 2,
                4 | 5 => 4,
                6 | 7 => 6,
                _ => 8,
            };
            // SAFETY: AVX2 + FMA checked; the shape is at most the strip, and the
            // kernel reads the tiled panel within `hd * n` and writes `j0..j1`.
            unsafe {
                scores_tile_tiled_avx2(
                    rt, q, q_stride, panel, n, hd, rows, j0, j1, scale, out, out_stride,
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
///
/// On x86_64 and aarch64 it reads the key-major panel [`key_panel`] builds; elsewhere
/// the lane panel.
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
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        scores_tile_tiled_scalar(
            q, q_stride, panel, n, hd, rows, j0, j1, scale, out, out_stride,
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scores_tile_rows_scalar(
            q, q_stride, panel, n, hd, rows, j0, j1, scale, out, out_stride,
        );
    }
}

/// The lane-panel (`[head_dim][n]`) scalar reference: the other architectures' scalar
/// twin and the fallback for a range too short for a vector key tile.
#[allow(clippy::too_many_arguments)]
pub fn scores_tile_rows_scalar(
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

/// [`scores_tile_scalar`] on the key-major panel: one sequential head-dimension chain
/// per `(row, key)`, the reference every vector tile shape is held to.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub fn scores_tile_tiled_scalar(
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
                acc = qr[k].mul_add(panel[panel_index(n, hd, j, k)], acc);
            }
            s[j] = acc * scale;
        }
    }
}

/// The row-major score-tile dispatcher: one monomorphisation per measured shape. x86_64
/// uses it for a text's `n % 16`-key tail, which the key-major panel stores row-major;
/// the tests also sweep its shapes.
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
        _ => scores_tile_rows_scalar(
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
/// AVX2 and FMA must be available; `j1 - j0 >= W`, `j1 <= n`, `rows > 0`,
/// `q.len() >= (rows - 1) * q_stride + hd`, `out.len() >= (rows - 1) * out_stride + j1`,
/// `panel.len() >= hd * n` (what [`scores_tile_shaped`] asserts).
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

/// Score a strip against the key-major panel of S4.9g part 3.
///
/// The full tiles of the text are scored by [`score_tile_tiled`], whose sixteen keys per
/// tile are contiguous for every head dimension; the last `n % 16` keys are a row-major
/// region and run the row-major kernel. Both write the same `acc * scale`, so a key's
/// bits never depend on which region it is in. The range is widened to whole tiles, as
/// the caller's union already is: keys outside a row's band are written and never read.
///
/// # Safety
/// AVX2 and FMA must be available; the bounds are those of [`scores_tile`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn scores_tile_tiled_avx2(
    rows_tile: usize,
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
    let n0 = n - n % PANEL_TILE;
    let full_hi = j1.min(n0);
    if j0 < full_hi {
        let a0 = j0 - j0 % PANEL_TILE;
        let a1 = full_hi.div_ceil(PANEL_TILE) * PANEL_TILE;
        let rt = rows_tile.min(rows).clamp(1, 8);
        macro_rules! run {
            ($r:literal) => {
                score_tile_tiled::<$r>(q, q_stride, panel, hd, rows, a0, a1, scale, out, out_stride)
            };
        }
        match rt {
            1 => run!(1),
            2 | 3 => run!(2),
            4 | 5 => run!(4),
            6 | 7 => run!(6),
            _ => run!(8),
        }
    }
    if j1 > n0 {
        let rem = n - n0;
        let lo = j0.saturating_sub(n0);
        let hi = j1 - n0;
        let tp = &panel[n0 * hd..];
        let to = &mut out[n0..];
        let rt = rows_tile.min(rows);
        let len = hi - lo;
        let kt = if len >= 16 {
            16
        } else if len >= VEC {
            8
        } else {
            0
        };
        if kt != 0 {
            scores_tile_avx2(
                rt, kt, q, q_stride, tp, rem, hd, rows, lo, hi, scale, to, out_stride,
            );
        } else {
            // Fewer than eight keys: the scalar chain, exactly as the row kernel's tail.
            for r in 0..rows {
                let qr = &q[r * q_stride..];
                let s = &mut to[r * out_stride..];
                for j in lo..hi {
                    let mut acc = 0.0f32;
                    for k in 0..hd {
                        acc = qr[k].mul_add(tp[k * rem + j], acc);
                    }
                    s[j] = acc * scale;
                }
            }
        }
    }
}

/// One [`scores_tile`] shape over the key-major panel: `R` rows × one tile of sixteen
/// keys, `R × 2` accumulators. `j0..j1` must be tile-aligned and a whole number of
/// tiles. The row tail narrows to four, two and one row, as the row-major kernel's does.
///
/// # Safety
/// AVX2 and FMA must be available; `j1 - j0 >= PANEL_TILE`, `j0 % PANEL_TILE == 0`,
/// `j1 <= n - n % PANEL_TILE`, `rows > 0`, and the buffers are those of
/// [`scores_tile`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_tile_tiled<const R: usize>(
    q: &[f32],
    q_stride: usize,
    panel: &[f32],
    hd: usize,
    rows: usize,
    j0: usize,
    j1: usize,
    scale: f32,
    out: &mut [f32],
    out_stride: usize,
) {
    debug_assert!(j1 - j0 >= PANEL_TILE && rows > 0);
    let s = _mm256_set1_ps(scale);
    let pp = panel.as_ptr();
    let mut r0 = 0;
    while r0 + R <= rows {
        score_block_tiled::<R>(
            q.as_ptr().add(r0 * q_stride),
            q_stride,
            pp,
            hd,
            j0,
            j1,
            s,
            out.as_mut_ptr().add(r0 * out_stride),
            out_stride,
        );
        r0 += R;
    }
    let rn = rows - r0;
    if rn > 0 {
        let qs = &q[r0 * q_stride..];
        let os = &mut out[r0 * out_stride..];
        if rn >= 4 && R > 4 {
            score_tile_tiled::<4>(qs, q_stride, panel, hd, rn, j0, j1, scale, os, out_stride);
        } else if rn >= 2 && R > 2 {
            score_tile_tiled::<2>(qs, q_stride, panel, hd, rn, j0, j1, scale, os, out_stride);
        } else {
            score_tile_tiled::<1>(qs, q_stride, panel, hd, rn, j0, j1, scale, os, out_stride);
        }
    }
}

/// One register-blocked step of [`score_tile_tiled`]: `R` rows × sixteen keys, every
/// `(row, key)` its own head-dimension chain. A head-dimension step reads one
/// sixty-four-byte run of the tile, so a strip's panel traversal is sequential.
///
/// # Safety
/// AVX2 and FMA must be available; pointers and ranges are those of [`score_tile_tiled`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_block_tiled<const R: usize>(
    q: *const f32,
    q_stride: usize,
    panel: *const f32,
    hd: usize,
    j0: usize,
    j1: usize,
    scale: __m256,
    out: *mut f32,
    out_stride: usize,
) {
    const NV: usize = PANEL_TILE / VEC;
    let mut qp = [q; R];
    for (r, p) in qp.iter_mut().enumerate() {
        *p = q.add(r * q_stride);
    }
    let mut j = j0;
    while j < j1 {
        let base = (j / PANEL_TILE) * hd * PANEL_TILE;
        let mut acc = [[_mm256_setzero_ps(); NV]; R];
        for k in 0..hd {
            let p = panel.add(base + k * PANEL_TILE);
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
        for (r, a) in acc.iter().enumerate() {
            let o = out.add(r * out_stride + j);
            for (v, av) in a.iter().enumerate() {
                _mm256_storeu_ps(o.add(v * VEC), _mm256_mul_ps(*av, scale));
            }
        }
        j += PANEL_TILE;
    }
}

/// The NEON score-tile dispatcher: one monomorphisation per measured shape (four rows
/// and sixteen, eight or four keys; four-by-four is 1×4 rows).
///
/// # Safety
/// NEON must be available; the bounds are those of [`scores_tile`], and callers have
/// already narrowed the shape to the strip and to the key range.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn scores_tile_neon(
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
        ($rt:literal, $nv:literal) => {
            score_tile_neon::<$rt, $nv>(
                q, q_stride, panel, n, hd, rows, j0, j1, scale, out, out_stride,
            )
        };
    }
    match (rows_tile, keys_tile) {
        (1, 4) => run!(1, 1),
        (1, 8) => run!(1, 2),
        (1, 16) => run!(1, 4),
        (2, 4) => run!(2, 1),
        (2, 8) => run!(2, 2),
        (2, 16) => run!(2, 4),
        (4, 4) => run!(4, 1),
        (4, 8) => run!(4, 2),
        (4, 16) => run!(4, 4),
        _ => scores_tile_scalar(
            q, q_stride, panel, n, hd, rows, j0, j1, scale, out, out_stride,
        ),
    }
}

/// One NEON score-tile shape: `R` rows × `NV` vectors of `VEC` keys, `R*NV`
/// accumulators.
///
/// # Safety
/// NEON must be available; `j1 - j0 >= NV*VEC`, `j1 <= n`, `rows > 0`,
/// `q.len() >= (rows - 1) * q_stride + hd`, `out.len() >= (rows - 1) * out_stride + j1`,
/// `panel.len() >= hd * n`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_tile_neon<const R: usize, const NV: usize>(
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
    let s = vdupq_n_f32(scale);
    let pp = panel.as_ptr();
    let mut r0 = 0;
    while r0 + R <= rows {
        score_block_neon::<R, NV>(
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
        let qs = &q[r0 * q_stride..];
        let os = &mut out[r0 * out_stride..];
        if rn >= 4 && R > 4 {
            score_tile_neon::<4, NV>(
                qs, q_stride, panel, n, hd, rn, j0, j1, scale, os, out_stride,
            );
        } else if rn >= 2 && R > 2 {
            score_tile_neon::<2, NV>(
                qs, q_stride, panel, n, hd, rn, j0, j1, scale, os, out_stride,
            );
        } else {
            score_tile_neon::<1, NV>(
                qs, q_stride, panel, n, hd, rn, j0, j1, scale, os, out_stride,
            );
        }
    }
}

/// One register-blocked NEON step of [`score_tile`]: `R` rows × `NV` vectors,
/// every `(row, key)` its own head-dimension chain.
///
/// # Safety
/// NEON must be available; pointers are those of [`score_tile`] with `stored <= R`
/// and `j0 + NV*VEC <= j1`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_block_neon<const R: usize, const NV: usize>(
    q: *const f32,
    q_stride: usize,
    panel: *const f32,
    n: usize,
    hd: usize,
    j0: usize,
    j1: usize,
    scale: float32x4_t,
    out: *mut f32,
    out_stride: usize,
    stored: usize,
) {
    let mut qp = [q; R];
    for (r, p) in qp.iter_mut().enumerate() {
        *p = q.add(r.min(stored - 1) * q_stride);
    }
    let mut j = j0;
    loop {
        let mut acc = [[vdupq_n_f32(0.0); NV]; R];
        for k in 0..hd {
            let p = panel.add(k * n + j);
            let mut w = [vdupq_n_f32(0.0); NV];
            for (v, wv) in w.iter_mut().enumerate() {
                *wv = vld1q_f32(p.add(v * VEC));
            }
            for (r, qr) in qp.iter().enumerate() {
                let qk = vdupq_n_f32(*qr.add(k));
                for (v, wv) in w.iter().enumerate() {
                    acc[r][v] = vfmaq_f32(acc[r][v], *wv, qk);
                }
            }
        }
        for r in 0..stored {
            let o = out.add(r * out_stride + j);
            for (v, a) in acc[r].iter().enumerate() {
                vst1q_f32(o.add(v * VEC), vmulq_f32(*a, scale));
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

/// The NEON score tile over the key-major panel of S4.9g part 3: `R` rows × one tile
/// of sixteen keys, `R × 4` accumulators, and a head-dimension step reads one
/// contiguous sixty-four-byte run of the tile, so a strip walks the panel front to
/// back. The row tail narrows through four, two and one row, as the row kernel's does.
///
/// # Safety
/// NEON must be available; `j1 - j0 >= PANEL_TILE`, `j0 % PANEL_TILE == 0`,
/// `j1 <= n - n % PANEL_TILE`, `rows > 0`, and the buffers are those of
/// [`scores_tile`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn scores_tile_tiled_neon(
    rows_tile: usize,
    q: &[f32],
    q_stride: usize,
    panel: &[f32],
    hd: usize,
    rows: usize,
    j0: usize,
    j1: usize,
    scale: f32,
    out: &mut [f32],
    out_stride: usize,
) {
    macro_rules! run {
        ($r:literal) => {
            score_tile_tiled_neon::<$r>(
                q, q_stride, panel, hd, rows, j0, j1, scale, out, out_stride,
            )
        };
    }
    match rows_tile {
        1 => run!(1),
        2 | 3 => run!(2),
        _ => run!(4),
    }
}

/// One [`scores_tile_tiled_neon`] shape, `R` rows at a time.
///
/// # Safety
/// NEON must be available; ranges and buffers are those of
/// [`scores_tile_tiled_neon`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_tile_tiled_neon<const R: usize>(
    q: &[f32],
    q_stride: usize,
    panel: &[f32],
    hd: usize,
    rows: usize,
    j0: usize,
    j1: usize,
    scale: f32,
    out: &mut [f32],
    out_stride: usize,
) {
    debug_assert!(j1 - j0 >= PANEL_TILE && rows > 0);
    let s = vdupq_n_f32(scale);
    let pp = panel.as_ptr();
    let mut r0 = 0;
    while r0 + R <= rows {
        score_block_tiled_neon::<R>(
            q.as_ptr().add(r0 * q_stride),
            q_stride,
            pp,
            hd,
            j0,
            j1,
            s,
            out.as_mut_ptr().add(r0 * out_stride),
            out_stride,
        );
        r0 += R;
    }
    let rn = rows - r0;
    if rn > 0 {
        let qs = &q[r0 * q_stride..];
        let os = &mut out[r0 * out_stride..];
        if rn >= 4 && R > 4 {
            score_tile_tiled_neon::<4>(qs, q_stride, panel, hd, rn, j0, j1, scale, os, out_stride);
        } else if rn >= 2 && R > 2 {
            score_tile_tiled_neon::<2>(qs, q_stride, panel, hd, rn, j0, j1, scale, os, out_stride);
        } else {
            score_tile_tiled_neon::<1>(qs, q_stride, panel, hd, rn, j0, j1, scale, os, out_stride);
        }
    }
}

/// One register-blocked step of [`score_tile_tiled_neon`]: `R` rows × sixteen keys,
/// every `(row, key)` its own head-dimension chain.
///
/// # Safety
/// NEON must be available; pointers and ranges are those of
/// [`score_tile_tiled_neon`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_block_tiled_neon<const R: usize>(
    q: *const f32,
    q_stride: usize,
    panel: *const f32,
    hd: usize,
    j0: usize,
    j1: usize,
    scale: float32x4_t,
    out: *mut f32,
    out_stride: usize,
) {
    const NV: usize = PANEL_TILE / VEC;
    let mut qp = [q; R];
    for (r, p) in qp.iter_mut().enumerate() {
        *p = q.add(r * q_stride);
    }
    let mut j = j0;
    while j < j1 {
        let base = (j / PANEL_TILE) * hd * PANEL_TILE;
        let mut acc = [[vdupq_n_f32(0.0); NV]; R];
        for k in 0..hd {
            let p = panel.add(base + k * PANEL_TILE);
            let mut w = [vdupq_n_f32(0.0); NV];
            for (v, wv) in w.iter_mut().enumerate() {
                *wv = vld1q_f32(p.add(v * VEC));
            }
            for (r, qr) in qp.iter().enumerate() {
                let qk = vdupq_n_f32(*qr.add(k));
                for (v, wv) in w.iter().enumerate() {
                    acc[r][v] = vfmaq_f32(acc[r][v], *wv, qk);
                }
            }
        }
        for (r, a) in acc.iter().enumerate() {
            let o = out.add(r * out_stride + j);
            for (v, av) in a.iter().enumerate() {
                vst1q_f32(o.add(v * VEC), vmulq_f32(*av, scale));
            }
        }
        j += PANEL_TILE;
    }
}

/// # Safety
/// NEON must be available; `lo <= hi <= x.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn max_row_neon(x: &[f32], lo: usize, hi: usize) -> f32 {
    let mut m = vdupq_n_f32(f32::NEG_INFINITY);
    let mut j = lo;
    while j + LANES <= hi {
        m = vmaxq_f32(m, vld1q_f32(x.as_ptr().add(j)));
        j += LANES;
    }
    // A maximum is exact, so the lanes can be folded in any order.
    let mut lanes = [0.0f32; 4];
    vst1q_f32(lanes.as_mut_ptr(), m);
    let mut best = f32::NEG_INFINITY;
    for &e in &lanes {
        best = best.max(e);
    }
    while j < hi {
        best = best.max(*x.get_unchecked(j));
        j += 1;
    }
    best
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
    // Checked in every build: the vector path indexes without bounds checks.
    assert!(lo <= hi && hi <= x.len(), "band outside the row");
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 + FMA checked; the loop stays inside `lo..hi`.
            return unsafe { max_row_avx2(x, lo, hi) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if vector_ok() {
            // SAFETY: NEON checked; the loop stays inside `lo..hi`.
            return unsafe { max_row_neon(x, lo, hi) };
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
    // Checked in every build: the vector path indexes without bounds checks.
    assert!(lo <= hi && hi <= x.len(), "band outside the row");
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 + FMA checked; the loop keeps every access inside `lo..hi`.
            unsafe { exp_row_avx2(x, lo, hi, max) };
            return sum_row(x, lo, hi);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if vector_ok() {
            // SAFETY: NEON checked; the loop keeps every access inside `lo..hi`.
            unsafe { exp_row_neon(x, lo, hi, max) };
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

/// # Safety
/// NEON must be available; `lo <= hi <= x.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp_row_neon(x: &mut [f32], lo: usize, hi: usize, max: f32) {
    let m = vdupq_n_f32(max);
    let mut j = lo;
    while j + LANES <= hi {
        let v = exp4(vsubq_f32(vld1q_f32(x.as_ptr().add(j)), m));
        vst1q_f32(x.as_mut_ptr().add(j), v);
        j += LANES;
    }
    while j < hi {
        *x.get_unchecked_mut(j) = exp_scalar(*x.get_unchecked(j) - max);
        j += 1;
    }
}

/// The number of keys the rows of a strip share at most: the union of their bands.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn mix_band_len(i0: usize, rows: usize, n: usize, half: Option<usize>) -> usize {
    let mut lo = usize::MAX;
    let mut hi = 0;
    for r in 0..rows {
        let (a, b) = band(i0 + r, n, half);
        lo = lo.min(a);
        hi = hi.max(b);
    }
    hi.saturating_sub(lo)
}

/// `out[r][d] = (Σ_j e[r][j]·v[j][d]) · inv[r]`, `j` ascending, for `rows` query rows
/// that start at text-local index `i0` and share one value head.
///
/// `e` is the strip's weights (`rows` rows of `e_stride`), `v` the pass's value rows
/// (`v_stride` floats each) with the head at `base`, `out` the strip's rows of the
/// attention output at `out_stride` with the head at `obase`. Each row's window comes
/// from [`band`].
///
/// On x86_64 and aarch64 a band longer than [`MIX_KEY_BLOCK`] takes the key-blocked
/// kernel of S4.9g part 3; a shorter one keeps part 2's row-block kernel, whose whole
/// band is register-resident. Both accumulate each `(row, d)` over ascending `j`, so
/// they are the same fold and a bit-identical result.
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
    if rows == 0 || n == 0 {
        return;
    }
    // Checked in every build: the vector kernel reads and writes through raw pointers,
    // so these bounds are what makes this safe function sound. Every band lies inside
    // `0..n` because the rows are rows of the text (`i0 + rows <= n`).
    assert!(i0 + rows <= n, "mix rows outside the text");
    assert!(
        e.len() >= (rows - 1) * e_stride + n,
        "weight rows shorter than the strip"
    );
    assert!(inv.len() >= rows, "one denominator per row");
    assert!(
        v.len() >= (n - 1) * v_stride + base + hd,
        "value rows shorter than the text"
    );
    assert!(
        out.len() >= (rows - 1) * out_stride + obase + hd,
        "output rows shorter than the strip"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 + FMA checked; rows are inside `e`, `inv` and `out`, keys
            // inside `n`, and the value loads stay inside one row of `v` because
            // `d0 + 16 <= hd <= v_stride - base`.
            unsafe {
                if mix_band_len(i0, rows, n, half) > MIX_KEY_BLOCK {
                    mix_rows_keyblocked_avx2(
                        MIX_BLOCK, e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out,
                        out_stride, obase,
                    )
                } else {
                    mix_rows_blocks_avx2(
                        MIX_BLOCK, e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out,
                        out_stride, obase,
                    )
                }
            };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if vector_ok() {
            // SAFETY: NEON checked; rows are inside `e`, `inv` and `out`, keys inside
            // `n`, and the value loads stay inside one row of `v` because
            // `d0 + 8 <= hd <= v_stride - base`.
            unsafe {
                if mix_band_len(i0, rows, n, half) > MIX_KEY_BLOCK {
                    mix_rows_keyblocked_neon(
                        MIX_BLOCK, e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out,
                        out_stride, obase,
                    )
                } else {
                    mix_rows_blocks_neon(
                        MIX_BLOCK, e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out,
                        out_stride, obase,
                    )
                }
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
    if rows == 0 || n == 0 {
        return;
    }
    // Checked in every build: the vector kernel reads and writes through raw pointers,
    // so these bounds are what makes this safe function sound. Every band lies inside
    // `0..n` because the rows are rows of the text (`i0 + rows <= n`).
    assert!(i0 + rows <= n, "mix rows outside the text");
    assert!(
        e.len() >= (rows - 1) * e_stride + n,
        "weight rows shorter than the strip"
    );
    assert!(inv.len() >= rows, "one denominator per row");
    assert!(
        v.len() >= (n - 1) * v_stride + base + hd,
        "value rows shorter than the text"
    );
    assert!(
        out.len() >= (rows - 1) * out_stride + obase + hd,
        "output rows shorter than the strip"
    );
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = rows_block;
    #[cfg(target_arch = "x86_64")]
    {
        if vector_ok() {
            // SAFETY: AVX2 + FMA checked; the bounds asserted above cover every row,
            // key and head dimension the block's clamped tails read.
            unsafe {
                if mix_band_len(i0, rows, n, half) > MIX_KEY_BLOCK {
                    mix_rows_keyblocked_avx2(
                        rows_block, e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv,
                        out, out_stride, obase,
                    )
                } else {
                    mix_rows_blocks_avx2(
                        rows_block, e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv,
                        out, out_stride, obase,
                    )
                }
            };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if vector_ok() {
            // SAFETY: NEON checked; the bounds asserted above cover every row, key and
            // head dimension the block's clamped tails read.
            unsafe {
                if mix_band_len(i0, rows, n, half) > MIX_KEY_BLOCK {
                    mix_rows_keyblocked_neon(
                        rows_block, e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv,
                        out, out_stride, obase,
                    )
                } else {
                    mix_rows_blocks_neon(
                        rows_block, e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv,
                        out, out_stride, obase,
                    )
                }
            };
            return;
        }
    }
    mix_rows_scalar(
        e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out, out_stride, obase,
    );
}

/// The value mix over key blocks (S4.9g part 3). The strip's rows accumulate into
/// `out` itself, and the keys are walked in blocks of [`MIX_KEY_BLOCK`]: every row block
/// of the strip visits a block before the walk moves on, so the text's value rows are
/// read from L3 once per strip instead of once per row block. Within a row the keys are
/// still visited in ascending order — blocks ascend, and inside a block the low edge,
/// the shared core and the high edge ascend — so this is the same fold as
/// [`mix_rows_scalar`], not a reassociation.
///
/// # Safety
/// AVX2 and FMA must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_rows_keyblocked_avx2(
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
    if rows == 0 {
        return;
    }
    let op = out.as_mut_ptr();
    let ep = e.as_ptr();
    let vp = v.as_ptr();
    // The accumulator lives in the output rows themselves; start every one of them at
    // zero so a block can be loaded and stored without a separate strip buffer.
    for r in 0..rows {
        let o = op.add(r * out_stride + obase);
        let mut d = 0;
        while d + VEC <= hd {
            _mm256_storeu_ps(o.add(d), _mm256_setzero_ps());
            d += VEC;
        }
        while d < hd {
            *o.add(d) = 0.0;
            d += 1;
        }
    }
    let rb = rows_block.clamp(1, 8);
    let mut lo_all = n;
    let mut hi_all = 0;
    for r in 0..rows {
        let (a, b) = band(i0 + r, n, half);
        lo_all = lo_all.min(a);
        hi_all = hi_all.max(b);
    }
    let step = 2 * VEC;
    let mut jb = lo_all;
    while jb < hi_all {
        let be = (jb + MIX_KEY_BLOCK).min(hi_all);
        // Head dimensions outside the loop, row blocks inside: the block's value slice
        // for one accumulator pair (`MIX_KEY_BLOCK × 16 × 4` bytes) stays in L1 while
        // every row block of the strip walks it.
        let mut d0 = 0;
        while d0 + step <= hd {
            let mut r0 = 0;
            while r0 < rows {
                let rn = (rows - r0).min(rb);
                macro_rules! run {
                    ($r:literal) => {
                        mix_block_chunk::<$r, 2>(
                            ep, e_stride, i0, n, half, vp, v_stride, base, hd, op, out_stride,
                            obase, r0, rn, jb, be, d0,
                        )
                    };
                }
                match rb {
                    1 => run!(1),
                    2 => run!(2),
                    4 => run!(4),
                    6 => run!(6),
                    _ => run!(8),
                }
                r0 += rn;
            }
            d0 += step;
        }
        if d0 < hd {
            // Head dimensions narrower than one accumulator pair: scalar, the same
            // ascending order as the reference.
            for r in 0..rows {
                let (a, b) = band(i0 + r, n, half);
                let lo = a.max(jb);
                let hi = b.min(be);
                let er = ep.add(r * e_stride);
                let o = op.add(r * out_stride + obase);
                for d in d0..hd {
                    let mut acc = *o.add(d);
                    for j in lo..hi {
                        acc = (*er.add(j)).mul_add(*vp.add(j * v_stride + base + d), acc);
                    }
                    *o.add(d) = acc;
                }
            }
        }
        jb = be;
    }
    // The denominator, once per row, after the whole ascending chain.
    for r in 0..rows {
        let k = _mm256_set1_ps(*inv.get_unchecked(r));
        let o = op.add(r * out_stride + obase);
        let mut d = 0;
        while d + VEC <= hd {
            _mm256_storeu_ps(o.add(d), _mm256_mul_ps(_mm256_loadu_ps(o.add(d)), k));
            d += VEC;
        }
        while d < hd {
            *o.add(d) = *o.add(d) * *inv.get_unchecked(r);
            d += 1;
        }
    }
}

/// One row block × one key block × one head-dimension pair of
/// [`mix_rows_keyblocked_avx2`]: the accumulator is loaded from `out`, walked over the
/// block's keys in ascending order, and stored back unscaled. `rn` rows are real; the
/// rest of the `R` lanes duplicate the last one and are discarded. The key range is the
/// block clipped by every row's own band.
///
/// # Safety
/// AVX2 and FMA must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_block_chunk<const R: usize, const V: usize>(
    ep: *const f32,
    e_stride: usize,
    i0: usize,
    n: usize,
    half: Option<usize>,
    vp: *const f32,
    v_stride: usize,
    base: usize,
    hd: usize,
    op: *mut f32,
    out_stride: usize,
    obase: usize,
    r0: usize,
    rn: usize,
    jb: usize,
    je: usize,
    d0: usize,
) {
    debug_assert!(rn > 0 && rn <= R && d0 + V * VEC <= hd);
    if jb >= je {
        return;
    }
    let mut bands = [(0usize, 0usize); R];
    for (r, b) in bands.iter_mut().enumerate() {
        let (a, c) = band(i0 + r0 + r.min(rn - 1), n, half);
        *b = (a.max(jb), c.min(je));
    }
    let core_lo = bands.iter().map(|b| b.0).max().unwrap_or(0);
    let core_hi = bands.iter().map(|b| b.1).min().unwrap_or(0);
    if core_lo > core_hi {
        // The block's windows share no key (a window narrower than the row block, or a
        // row whose band is outside this block): one row at a time, in ascending order.
        if R == 1 {
            return;
        }
        for r in 0..rn {
            mix_block_chunk::<1, V>(
                ep,
                e_stride,
                i0,
                n,
                half,
                vp,
                v_stride,
                base,
                hd,
                op,
                out_stride,
                obase,
                r0 + r,
                1,
                jb,
                je,
                d0,
            );
        }
        return;
    }
    let mut acc = [[_mm256_setzero_ps(); V]; R];
    for (r, a) in acc.iter_mut().enumerate() {
        let o = op.add((r0 + r.min(rn - 1)) * out_stride + obase + d0);
        for (v, av) in a.iter_mut().enumerate() {
            *av = _mm256_loadu_ps(o.add(v * VEC));
        }
    }
    // The keys only the earliest-windowed rows see, ascending.
    for (r, a) in acc.iter_mut().enumerate() {
        let er = ep.add((r0 + r.min(rn - 1)) * e_stride);
        for j in bands[r].0..core_lo {
            let ev = _mm256_broadcast_ss(&*er.add(j));
            let p = vp.add(j * v_stride + base + d0);
            for (v, av) in a.iter_mut().enumerate() {
                *av = _mm256_fmadd_ps(ev, _mm256_loadu_ps(p.add(v * VEC)), *av);
            }
        }
    }
    // The core: every row sees these keys, so one value load feeds them all.
    for j in core_lo..core_hi {
        let p = vp.add(j * v_stride + base + d0);
        let mut val = [_mm256_setzero_ps(); V];
        for (v, w) in val.iter_mut().enumerate() {
            *w = _mm256_loadu_ps(p.add(v * VEC));
        }
        for (r, a) in acc.iter_mut().enumerate() {
            let ev = _mm256_broadcast_ss(&*ep.add((r0 + r.min(rn - 1)) * e_stride + j));
            for (v, w) in val.iter().enumerate() {
                a[v] = _mm256_fmadd_ps(ev, *w, a[v]);
            }
        }
    }
    // And the keys only the latest-windowed rows see, ascending.
    for (r, a) in acc.iter_mut().enumerate() {
        let er = ep.add((r0 + r.min(rn - 1)) * e_stride);
        for j in core_hi..bands[r].1 {
            let ev = _mm256_broadcast_ss(&*er.add(j));
            let p = vp.add(j * v_stride + base + d0);
            for (v, av) in a.iter_mut().enumerate() {
                *av = _mm256_fmadd_ps(ev, _mm256_loadu_ps(p.add(v * VEC)), *av);
            }
        }
    }
    for r in 0..rn {
        let o = op.add((r0 + r) * out_stride + obase + d0);
        for (v, av) in acc[r].iter().enumerate() {
            _mm256_storeu_ps(o.add(v * VEC), *av);
        }
    }
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

/// # Safety
/// NEON must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_rows_blocks_neon(
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
            mix_row_blocks_neon::<$rb, $vv>(
                e, e_stride, i0, rows, n, half, v, v_stride, base, hd, inv, out, out_stride, obase,
            )
        };
    }
    match rows_block {
        1 => run!(1, 2),
        2 => run!(2, 2),
        4 => run!(4, 2),
        12 => run!(12, 1),
        _ => run!(6, 2),
    }
}

/// [`mix_rows`] in blocks of `R` rows that share one value load, `NV` vectors of head
/// dimensions each; the tail narrows through eight, four, two and one row.
///
/// # Safety
/// NEON must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_row_blocks_neon<const R: usize, const NV: usize>(
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
        mix_block_neon::<R, NV>(
            e, e_stride, i0, n, half, v, v_stride, base, hd, inv, out, out_stride, obase, r0, R,
        );
        r0 += R;
    }
    if r0 == rows {
        return;
    }
    if R > 8 && rows - r0 >= 8 {
        mix_block_neon::<8, NV>(
            e, e_stride, i0, n, half, v, v_stride, base, hd, inv, out, out_stride, obase, r0, 8,
        );
        r0 += 8;
    }
    if R > 4 && rows - r0 >= 4 {
        mix_block_neon::<4, NV>(
            e, e_stride, i0, n, half, v, v_stride, base, hd, inv, out, out_stride, obase, r0, 4,
        );
        r0 += 4;
    }
    if rows - r0 >= 2 {
        mix_block_neon::<2, NV>(
            e, e_stride, i0, n, half, v, v_stride, base, hd, inv, out, out_stride, obase, r0, 2,
        );
        r0 += 2;
    }
    if r0 < rows {
        mix_block_neon::<1, NV>(
            e, e_stride, i0, n, half, v, v_stride, base, hd, inv, out, out_stride, obase, r0, 1,
        );
    }
}

/// `R` rows at a time, `NV` vectors of head dimensions per row, one value load for the
/// block; the ascending edge/core/edge fold of [`mix_block`].
///
/// # Safety
/// NEON must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_block_neon<const R: usize, const NV: usize>(
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
        // block. One row at a time, still vectorised over the head dimension.
        for r in 0..rn {
            mix_block_neon::<1, NV>(
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
    let step = NV * VEC;
    let mut d0 = 0;
    while d0 + step <= hd {
        let mut acc = [[vdupq_n_f32(0.0); NV]; R];
        for r in 0..R {
            let er = ep.add((r0 + r.min(rn - 1)) * e_stride);
            for j in bands[r].0..core_lo {
                let ev = vdupq_n_f32(*er.add(j));
                let p = vp.add(j * v_stride + base + d0);
                for (vv, a) in acc[r].iter_mut().enumerate() {
                    *a = vfmaq_f32(*a, vld1q_f32(p.add(vv * VEC)), ev);
                }
            }
        }
        for j in core_lo..core_hi {
            let p = vp.add(j * v_stride + base + d0);
            let mut val = [vdupq_n_f32(0.0); NV];
            for (vv, w) in val.iter_mut().enumerate() {
                *w = vld1q_f32(p.add(vv * VEC));
            }
            for r in 0..R {
                let ev = vdupq_n_f32(*ep.add((r0 + r.min(rn - 1)) * e_stride + j));
                for (vv, w) in val.iter().enumerate() {
                    acc[r][vv] = vfmaq_f32(acc[r][vv], *w, ev);
                }
            }
        }
        for r in 0..R {
            let er = ep.add((r0 + r.min(rn - 1)) * e_stride);
            for j in core_hi..bands[r].1 {
                let ev = vdupq_n_f32(*er.add(j));
                let p = vp.add(j * v_stride + base + d0);
                for (vv, a) in acc[r].iter_mut().enumerate() {
                    *a = vfmaq_f32(*a, vld1q_f32(p.add(vv * VEC)), ev);
                }
            }
        }
        for r in 0..rn {
            let o = out.as_mut_ptr().add((r0 + r) * out_stride + obase + d0);
            let k = vdupq_n_f32(*inv.get_unchecked(r0 + r));
            for (vv, a) in acc[r].iter().enumerate() {
                vst1q_f32(o.add(vv * VEC), vmulq_f32(*a, k));
            }
        }
        d0 += step;
    }
    if d0 < hd {
        // Head dimensions narrower than one accumulator pair: scalar, in the same
        // ascending order as the reference.
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

/// The value mix over key blocks (S4.9g part 3), NEON: the strip's rows accumulate
/// into `out` itself, and the keys are walked in blocks of [`MIX_KEY_BLOCK`] for every
/// row block of the strip before the walk moves on. Within a row the keys are still
/// visited in ascending order — blocks ascend, and inside a block the low edge, the
/// shared core and the high edge ascend — so this is the same fold as
/// [`mix_rows_scalar`], not a reassociation.
///
/// # Safety
/// NEON must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_rows_keyblocked_neon(
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
    if rows == 0 {
        return;
    }
    let op = out.as_mut_ptr();
    let ep = e.as_ptr();
    let vp = v.as_ptr();
    // The accumulator lives in the output rows themselves; start every one of them at
    // zero so a block can be loaded and stored without a separate strip buffer.
    for r in 0..rows {
        let o = op.add(r * out_stride + obase);
        let mut d = 0;
        while d + VEC <= hd {
            vst1q_f32(o.add(d), vdupq_n_f32(0.0));
            d += VEC;
        }
        while d < hd {
            *o.add(d) = 0.0;
            d += 1;
        }
    }
    let rb = rows_block.clamp(1, 8);
    let mut lo_all = n;
    let mut hi_all = 0;
    for r in 0..rows {
        let (a, b) = band(i0 + r, n, half);
        lo_all = lo_all.min(a);
        hi_all = hi_all.max(b);
    }
    let step = 2 * VEC;
    let mut jb = lo_all;
    while jb < hi_all {
        let be = (jb + MIX_KEY_BLOCK).min(hi_all);
        // Head dimensions outside the loop, row blocks inside: the block's value slice
        // for one accumulator pair stays in L1 while every row block walks it.
        let mut d0 = 0;
        while d0 + step <= hd {
            let mut r0 = 0;
            while r0 < rows {
                let rn = (rows - r0).min(rb);
                macro_rules! run {
                    ($r:literal) => {
                        mix_block_chunk_neon::<$r, 2>(
                            ep, e_stride, i0, n, half, vp, v_stride, base, hd, op, out_stride,
                            obase, r0, rn, jb, be, d0,
                        )
                    };
                }
                match rb {
                    1 => run!(1),
                    2 => run!(2),
                    4 => run!(4),
                    6 => run!(6),
                    _ => run!(8),
                }
                r0 += rn;
            }
            d0 += step;
        }
        if d0 < hd {
            // Head dimensions narrower than one accumulator pair: scalar, the same
            // ascending order as the reference.
            for r in 0..rows {
                let (a, b) = band(i0 + r, n, half);
                let lo = a.max(jb);
                let hi = b.min(be);
                let er = ep.add(r * e_stride);
                let o = op.add(r * out_stride + obase);
                for d in d0..hd {
                    let mut acc = *o.add(d);
                    for j in lo..hi {
                        acc = (*er.add(j)).mul_add(*vp.add(j * v_stride + base + d), acc);
                    }
                    *o.add(d) = acc;
                }
            }
        }
        jb = be;
    }
    // The denominator, once per row, after the whole ascending chain.
    for r in 0..rows {
        let k = vdupq_n_f32(*inv.get_unchecked(r));
        let o = op.add(r * out_stride + obase);
        let mut d = 0;
        while d + VEC <= hd {
            vst1q_f32(o.add(d), vmulq_f32(vld1q_f32(o.add(d)), k));
            d += VEC;
        }
        while d < hd {
            *o.add(d) = *o.add(d) * *inv.get_unchecked(r);
            d += 1;
        }
    }
}

/// One row block × one key block × one head-dimension pair of
/// [`mix_rows_keyblocked_neon`]: the accumulator is loaded from `out`, walked over the
/// block's keys in ascending order, and stored back unscaled. `rn` rows are real; the
/// rest of the `R` lanes duplicate the last one and are discarded.
///
/// # Safety
/// NEON must be available, and every row/key/head index must be in bounds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn mix_block_chunk_neon<const R: usize, const V: usize>(
    ep: *const f32,
    e_stride: usize,
    i0: usize,
    n: usize,
    half: Option<usize>,
    vp: *const f32,
    v_stride: usize,
    base: usize,
    hd: usize,
    op: *mut f32,
    out_stride: usize,
    obase: usize,
    r0: usize,
    rn: usize,
    jb: usize,
    je: usize,
    d0: usize,
) {
    debug_assert!(rn > 0 && rn <= R && d0 + V * VEC <= hd);
    if jb >= je {
        return;
    }
    let mut bands = [(0usize, 0usize); R];
    for (r, b) in bands.iter_mut().enumerate() {
        let (a, c) = band(i0 + r0 + r.min(rn - 1), n, half);
        *b = (a.max(jb), c.min(je));
    }
    let core_lo = bands.iter().map(|b| b.0).max().unwrap_or(0);
    let core_hi = bands.iter().map(|b| b.1).min().unwrap_or(0);
    if core_lo > core_hi {
        // The block's windows share no key: one row at a time, in ascending order.
        if R == 1 {
            return;
        }
        for r in 0..rn {
            mix_block_chunk_neon::<1, V>(
                ep,
                e_stride,
                i0,
                n,
                half,
                vp,
                v_stride,
                base,
                hd,
                op,
                out_stride,
                obase,
                r0 + r,
                1,
                jb,
                je,
                d0,
            );
        }
        return;
    }
    let mut acc = [[vdupq_n_f32(0.0); V]; R];
    for (r, a) in acc.iter_mut().enumerate() {
        let o = op.add((r0 + r.min(rn - 1)) * out_stride + obase + d0);
        for (v, av) in a.iter_mut().enumerate() {
            *av = vld1q_f32(o.add(v * VEC));
        }
    }
    // The keys only the earliest-windowed rows see, ascending.
    for (r, a) in acc.iter_mut().enumerate() {
        let er = ep.add((r0 + r.min(rn - 1)) * e_stride);
        for j in bands[r].0..core_lo {
            let ev = vdupq_n_f32(*er.add(j));
            let p = vp.add(j * v_stride + base + d0);
            for (v, av) in a.iter_mut().enumerate() {
                *av = vfmaq_f32(*av, vld1q_f32(p.add(v * VEC)), ev);
            }
        }
    }
    // The core: every row sees these keys, so one value load feeds them all.
    for j in core_lo..core_hi {
        let p = vp.add(j * v_stride + base + d0);
        let mut val = [vdupq_n_f32(0.0); V];
        for (v, w) in val.iter_mut().enumerate() {
            *w = vld1q_f32(p.add(v * VEC));
        }
        for (r, a) in acc.iter_mut().enumerate() {
            let ev = vdupq_n_f32(*ep.add((r0 + r.min(rn - 1)) * e_stride + j));
            for (v, w) in val.iter().enumerate() {
                a[v] = vfmaq_f32(a[v], *w, ev);
            }
        }
    }
    // And the keys only the latest-windowed rows see, ascending.
    for (r, a) in acc.iter_mut().enumerate() {
        let er = ep.add((r0 + r.min(rn - 1)) * e_stride);
        for j in core_hi..bands[r].1 {
            let ev = vdupq_n_f32(*er.add(j));
            let p = vp.add(j * v_stride + base + d0);
            for (v, av) in a.iter_mut().enumerate() {
                *av = vfmaq_f32(*av, vld1q_f32(p.add(v * VEC)), ev);
            }
        }
    }
    for r in 0..rn {
        let o = op.add((r0 + r) * out_stride + obase + d0);
        for (v, av) in acc[r].iter().enumerate() {
            vst1q_f32(o.add(v * VEC), *av);
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
                            // The arch's reference layout: the key-major panel's scalar
                            // twin on x86_64, the lane panel's elsewhere.
                            scores_tile_scalar(
                                &q[i * qd + h * hd..],
                                qd,
                                &panel,
                                n,
                                hd,
                                1,
                                lo,
                                hi,
                                scale,
                                s,
                                n,
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
                    for strip in [1usize, 7, 16, 32, 64, n.max(1)] {
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
                    for strip in [1usize, 2, 5, 16, 32, 64] {
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
        for &(n, hd, kvd, base) in &[
            (5usize, 8usize, 24usize, 8usize),
            (33, 8, 24, 8),
            (32, 16, 16, 0),
        ] {
            let k = prng(0xfeed ^ n as u64, n * kvd, -1.0, 1.0);
            let mut panel = vec![0.0f32; hd * n];
            key_panel(&k, kvd, base, hd, n, &mut panel);
            for j in 0..n {
                for d in 0..hd {
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    assert_eq!(
                        panel[panel_index(n, hd, j, d)],
                        k[j * kvd + base + d],
                        "{d},{j}"
                    );
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    assert_eq!(panel[d * n + j], k[j * kvd + base + d], "{d},{j}");
                }
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
                    for strip in [7usize, 16, 32, 64] {
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
        for &hd in &[7usize, 12, 16, 32, 64, 256] {
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
        for &hd in &[7usize, 12, 16, 32, 256] {
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
        // scalar twin is the same chain: the tile is a scheduling change. The row kernel
        // reads the lane panel (`key_panel_rows`); the tile reads whichever panel
        // `key_panel` builds on this architecture.
        for &hd in &HDS {
            for &n in &[16usize, 33, 129] {
                let q = prng(0x1a11 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                let k = prng(0x1a12 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                let mut rows_panel = vec![0.0f32; hd * n];
                key_panel_rows(&k, hd, 0, hd, n, 0, n, &mut rows_panel);
                let mut panel = vec![0.0f32; hd * n];
                key_panel(&k, hd, 0, hd, n, &mut panel);
                let mut a = vec![0.0f32; n];
                let mut a2 = vec![0.0f32; n];
                let mut b = vec![0.0f32; n];
                let mut c = vec![0.0f32; n];
                for i in [0usize, 1, n / 2, n - 1] {
                    let (lo, hi) = band(i, n, Some(3));
                    scores_row(&q[i * hd..], &rows_panel, n, hd, lo, hi, 0.5, &mut a);
                    scores_tile_rows_scalar(
                        &q[i * hd..],
                        hd,
                        &rows_panel,
                        n,
                        hd,
                        1,
                        lo,
                        hi,
                        0.5,
                        &mut a2,
                        n,
                    );
                    same(
                        &a2[lo..hi],
                        &a[lo..hi],
                        &format!("rows scalar hd {hd} n {n} row {i}"),
                    );
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

    #[test]
    fn every_key_block_edge_is_the_same_fold() {
        // Bands wider than `MIX_KEY_BLOCK` take the key-blocked value mix, and their
        // edges fall inside blocks: every strip size and window must still be the
        // scalar fold, key ascending.
        let n = 300;
        for &hd in &[16usize, 64] {
            let q = prng(0x8c01 ^ hd as u64, n * hd, -1.0, 1.0);
            let k = prng(0x8c02 ^ hd as u64, n * hd, -1.0, 1.0);
            let v = prng(0x8c03 ^ hd as u64, n * hd, -1.0, 1.0);
            for &half in &[
                None,
                Some(0),
                Some(1),
                Some(8),
                Some(64),
                Some(128),
                Some(256),
            ] {
                let want = attend(
                    n,
                    hd,
                    1,
                    1,
                    half,
                    64,
                    ENGINE_TILE,
                    ENGINE_MIX,
                    &q,
                    &k,
                    &v,
                    true,
                );
                for &strip in &[1usize, 7, 16, 32, 64, n] {
                    for &mix in &[1usize, 6, 12] {
                        let got = attend(
                            n,
                            hd,
                            1,
                            1,
                            half,
                            strip,
                            ENGINE_TILE,
                            mix,
                            &q,
                            &k,
                            &v,
                            false,
                        );
                        same(
                            &got,
                            &want,
                            &format!("hd {hd} half {half:?} strip {strip} mix {mix}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_long_text_matches_the_scalar_twin() {
        // Long enough for every panel tile and every mix key block to be visited many
        // times: 1 024 to 4 096 keys, strips of 1 to 64 rows, global and sliding.
        for &(n, hd) in &[(4096usize, 16usize), (2048, 32), (1024, 64), (1024, 128)] {
            for &half in &[None, Some(64)] {
                let q = prng(0x9d01 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                let k = prng(0x9d02 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                let v = prng(0x9d03 ^ hd as u64 ^ n as u64, n * hd, -1.0, 1.0);
                let want = attend(
                    n,
                    hd,
                    1,
                    1,
                    half,
                    64,
                    ENGINE_TILE,
                    ENGINE_MIX,
                    &q,
                    &k,
                    &v,
                    true,
                );
                for &strip in &[1usize, 7, 16, 32, 64, n] {
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

    /// The twelve slopes of `jina-bert-v2`'s 12 heads and `f_max_alibi_bias = 8`, as
    /// llama.cpp's soft-max computes them: two geometric groups of eight and four.
    #[test]
    fn alibi_slopes_follow_llama_cpps_two_groups() {
        let s = alibi_slopes(12, 8.0);
        let exact = [
            0.5f32, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        ];
        for (h, &want) in exact.iter().enumerate() {
            assert_eq!(s[h].to_bits(), want.to_bits(), "head {h}: {}", s[h]);
        }
        // The second group is sqrt(1/2) at odd powers; compare to the same f32 the
        // formula's own power function produces, not to a decimal literal.
        let m1 = crate::vmath::det_pow_f32(2.0, -4.0 / 8.0);
        for (h, e) in [(8usize, 1i32), (9, 3), (10, 5), (11, 7)] {
            let want = crate::vmath::det_pow_f32(m1, e as f32);
            assert_eq!(s[h].to_bits(), want.to_bits(), "head {h}: {}", s[h]);
        }
        // A single head is its own log2 group: `m0 = 2^-8` and the first slope is `m0^1`.
        assert_eq!(alibi_slopes(1, 8.0), vec![0.00390625]);
    }

    /// The bias is `slope · (−|j − qpos|)` on exactly the band it is given, and
    /// nothing outside it.
    #[test]
    fn alibi_row_touches_only_the_band() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        alibi_row(&mut x, 1, 4, 2, 0.5);
        assert_eq!(x[0].to_bits(), 1.0f32.to_bits());
        for (j, want) in [(1usize, 2.0 - 0.5), (2, 3.0), (3, 4.0 - 0.5)] {
            assert_eq!(x[j], want, "key {j}");
        }
        assert_eq!(x[4].to_bits(), 5.0f32.to_bits());

        // A whole row against the formula, for a position away from the middle.
        let mut row: Vec<f32> = (0..97).map(|i| i as f32 * 0.125).collect();
        let orig = row.clone();
        alibi_row(&mut row, 3, 91, 40, 0.03125);
        for j in 3..91 {
            let want = orig[j] + 0.03125 * -((j as i64 - 40).unsigned_abs() as f32);
            assert_eq!(row[j].to_bits(), want.to_bits(), "key {j}");
        }
    }
}
