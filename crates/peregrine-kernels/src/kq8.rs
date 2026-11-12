//! The 8-bit prompt form (S5.4a): the fork's `q8_k_r16` arithmetic as a tile.
//!
//! A prompt pass over a Q4_K or Q6_K matrix is bound by the per-32-weight scale
//! multiply and the minimum term, not by the byte dot product
//! (`docs/design/s5-prefill-avx512.md` §9.3, §10.4). ik_llama.cpp avoids both by
//! throwing the file's grid away: for every (matrix row, 256-weight super-block) it
//! dequantises the weights to f32, takes the largest magnitude and stores one f32
//! scale plus one int8 code per weight (`iqk_convert_q4_k_r4_q8_k_r16`,
//! `ggml/src/iqk/iqk_gemm_kquants.cpp:2400`). The product is then a plain byte dot
//! product with one scale multiply per (row, super-block). That is an
//! approximation, and Peregrine's default stays exact (ADR 0003); this module is
//! the kernel behind the opt-in `--fast-prompt` mode of
//! `docs/design/s5.4-fast-prompt.md`, and nothing dispatches it yet.
//!
//! # The arithmetic, per (row, 256 weights)
//!
//! 1. the 256 weights as f32, with the format's own formula and the fork's own
//!    operation order: Q4_K's `w = q·Ld·d − dmin·Lm` is
//!    `fma((q·Ld) as f32, d, (−dmin)·Lm)` — the products `q·Ld` are exact
//!    integers, the min term is one multiply, and the weight is one fma
//!    (`:2444-2463`). Q6_K never forms an f32 weight: its per-16 integer products
//!    `(q − 32)·L` are exact in int16 and the super-block's `d` enters only the
//!    scale (`convert_to_q8_k_r8`, `ggml/src/iqk/iqk_common.h:638-687`);
//! 2. `m = max |w|` over the 256;
//! 3. `scale = m / 127` — a *division* — and the codes are
//!    `round_to_nearest_even(w · (1/scale))` with the reciprocal rounded to f32
//!    first (`dq != 0 ? 1/dq : 0`, `:2464-2466`; `PackerQ80::convert`, `:16-26`).
//!    An all-zero block has `scale = 0` and codes 0;
//! 4. Q6_K's exact case: `dnew = max|p| · (1/127)`; when `dnew < 1` the codes are
//!    the products themselves and the scale is `d · dnew = d` (no rounding at all).
//!    Otherwise `scale = d · dnew` and `code = round(p · (1/dnew))`.
//!
//! The codes are stored biased by +128, because `vpdpbusd` takes its first
//! operand as unsigned bytes and its second as signed ones, and the activation is
//! the signed side. The bias is paid back once per (row, block) with
//! `−128 · Σ a`, taken from [`KqActs::sum256`], in int32 lanes — exact, and one
//! float multiply per (row, block) is all the epilogue costs.
//!
//! # Where this differs from the design note's wording
//!
//! `docs/design/s5.4-fast-prompt.md` §2 says `code = round(w / scale)` and, for
//! Q6_K, `w = q·L·d` with the exact case taken when "the integer products fit in
//! int8". The fork does none of those three literally: it multiplies by a rounded
//! reciprocal rather than dividing, it forms Q6_K's products as `(q − 32)·L` (the
//! six-bit code is signed; the +32 offset disappears into the product) and never
//! multiplies them by `d`, and its test is `max|p|·(1/127) < 1`, i.e.
//! `max|p| ≤ 126`, which is the same thing as "the products fit in int8" only
//! because 127 itself is the one value where the two tests could differ. This
//! module follows the fork (that is what makes the two engines' fast modes
//! comparable, and §2 of the note is corrected to say so).
//!
//! # The layout
//!
//! Eight output rows of a matrix are interleaved per super-block, the way
//! [`crate::kq::Q4K8`] interleaves them, because the AVX-512 kernel loads one
//! 512-bit weight operand per (eight-weight group, activation row): that operand
//! is eight rows × eight codes, and `vpdpbusd` then produces two int32 partials
//! per row. One super-block of the form is [`Q8K_SIZE`] bytes:
//!
//! ```text
//! 0     qs   the codes: qs[group * 64 + row * 8 + k]      2048 bytes
//! 2048  d    the eight rows' f32 scales, one per row         32 bytes
//! 2080  pad                                                 32 bytes
//! ```
//!
//! `qs[group * 64 + row * 8 + k]` is the biased code (`code + 128`, so 1..=255) of
//! weight `group * 8 + k` of row `row`. A 64-byte load at `group * 64` is one
//! operand, aligned because [`Q8K_SIZE`] is 33 cache lines and the tile's base is
//! 64-byte aligned. The scale is at `d + row * 4`, a scalar load per (row, block).
//!
//! The stride is 33 cache lines rather than the 32 the codes need because a
//! 64-byte load that straddles a line costs two accesses on every group; the
//! padding is 1.5 % of the form's bytes and worth it. One weight therefore costs
//! `2112 / 2048 = 1.03125` bytes, against the file's own 0.5625 (Q4_K) and 0.82
//! (Q6_K) — see the S5.4a results for what that is on a 1B model.
//!
//! # The scalar twin is the definition
//!
//! [`build_q4k`], [`build_q6k`], [`build_q5_0`] and [`dot_q8k`] are plain loops;
//! the AVX-512 bodies in `crate::kqsimd` are asserted bit for bit equal to them
//! for every width and every edge tile (ADR 0005).

use crate::format::f16;
use crate::kq::{tile_q4k_min, tile_q4k_scale, KqActs, ACT_GROUP, Q4K8, Q6K8};
#[cfg(feature = "fmt-legacy")]
use crate::kq::{Q5_08, Q5_0_WEIGHTS};
use crate::quant::QuantAct;

#[cfg(feature = "fmt-legacy")]
use crate::kq::Q5_0_WINDOWS;

/// The output rows one tile of the form holds.
pub const Q8K_ROWS: usize = 8;
/// The weights one super-block of the form holds.
pub const Q8K_WEIGHTS: usize = 256;
/// The eight-weight groups one super-block splits into.
pub const Q8K_GROUPS: usize = 32;
/// The bytes one super-block of the form takes: 2048 of codes, 32 of scales, 32 of
/// padding.
pub const Q8K_SIZE: usize = 2112;
/// The bytes of codes in one super-block of the form.
pub const Q8K_CODES: usize = 2048;
/// The byte offset of the eight rows' scales within one super-block of the form.
pub const Q8K_D: usize = Q8K_CODES;

/// The bytes a form of `n_sb` super-blocks takes.
pub const fn form_size(n_sb: usize) -> usize {
    n_sb * Q8K_SIZE
}

/// Whether this process holds a form for a Q5_0 matrix whose rows stop inside a
/// 256-weight super-block (S5.4f, S5.4h): Qwen 2.5 0.5B's 896-wide rows.
///
/// Such a matrix is multiplied against a *padded* activation buffer —
/// [`KqActs::fill_padded`](crate::kq::KqActs::fill_padded), which needs the
/// per-256 layout of the short last super-block with its missing codes zero —
/// rather than the plain one. [`crate::quant::QuantAct::fill`] builds that layout
/// for a width that is a whole number of super-blocks always, and for a shorter
/// width only when the mode will read it: `--fast-prompt` from the environment,
/// or a form built through the library's `load_with_fast`, which is what this
/// flag records. It is process-wide like the mode itself, and a matrix without a
/// form never reads the extra codes, so it cannot move a number.
static PADDED_Q5_FORM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Record that a form for a Q5_0 matrix with a short last super-block was built
/// (see [`padded_q5_form`]); called by the form's loader.
pub fn note_padded_q5_form() {
    PADDED_Q5_FORM.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether this process holds a form for a Q5_0 matrix whose rows stop inside a
/// 256-weight super-block (see [`note_padded_q5_form`]).
pub fn padded_q5_form() -> bool {
    PADDED_Q5_FORM.load(std::sync::atomic::Ordering::Relaxed)
}

/// The byte offset of (super-block `sb`, group `g`, row `row`, code `k`) in a form.
pub const fn code_at(sb: usize, g: usize, row: usize, k: usize) -> usize {
    sb * Q8K_SIZE + g * 64 + row * 8 + k
}

/// The byte offset of (super-block `sb`, row `row`)'s scale in a form.
pub const fn scale_at(sb: usize, row: usize) -> usize {
    sb * Q8K_SIZE + Q8K_D + row * 4
}

// ---------------------------------------------------------------------------
// The scalar twin: the format's weights, and the requantisation
// ---------------------------------------------------------------------------

/// The f32 weights of one row of a repacked Q4_K super-block, in file order.
///
/// This is the fork's own arithmetic (`:2444-2463`): the 32-weight sub-block's
/// scale and min are six-bit integers, the products `q·Ld` are exact, and each
/// weight is one fused multiply-add of the min term. It is exported because the
/// error report of `docs/design/s5.4-fast-prompt.md` §4 measures against these
/// numbers, not against a re-derived formula.
///
/// # Panics
///
/// If `row` is not one of the tile's eight rows.
pub fn q4k_weights(t: &[u8], row: usize) -> [f32; Q8K_WEIGHTS] {
    assert!(row < Q8K_ROWS, "a tile holds eight rows");
    let d = f16(u16::from_le_bytes([
        t[crate::kq::q4_d() + row * 2],
        t[crate::kq::q4_d() + row * 2 + 1],
    ]));
    let dmin = f16(u16::from_le_bytes([
        t[crate::kq::q4_dmin() + row * 2],
        t[crate::kq::q4_dmin() + row * 2 + 1],
    ]));
    let mut w = [0.0f32; Q8K_WEIGHTS];
    for sub in 0..8 {
        let ld = tile_q4k_scale(t, sub, row) as i32;
        let lm = tile_q4k_min(t, sub, row) as i32;
        // The fork's `vm = dmin * (-1)`, then `mm = vm * Lm`: one rounding, and
        // negation is exact, so this is -(dmin·Lm) with one rounding either way.
        let mm = (-dmin) * lm as f32;
        for l in 0..32 {
            // The sub-block's nibbles: the low nibbles of the row's bytes
            // 32·(sub/2) .. +32 for an even sub-block, the high nibbles for an odd one.
            let base = (sub / 2) * 32;
            let byte = t[crate::kq::q4_qs() + (base + l) / 8 * 64 + row * 8 + (base + l) % 8];
            let q = if sub % 2 == 0 {
                (byte & 0x0F) as i32
            } else {
                (byte >> 4) as i32
            };
            w[sub * 32 + l] = ((q * ld) as f32).mul_add(d, mm);
        }
    }
    w
}

/// The f32 weights of one row of a repacked Q5_0 super-block (`n_blk` blocks of
/// 32, at most eight), in file order; a short tail is zero.
///
/// The format's formula is `w = d · (q − 16)` with `q` the five-bit code; the
/// multiply is the only rounding, and `q − 16` is the seed of the exact kernel's
/// `−16·Σa` offset.
///
/// # Panics
///
/// If `row` is not one of the tile's eight rows, or `n_blk` is more than eight.
#[cfg(feature = "fmt-legacy")]
pub fn q5_0_weights(t: &[u8], n_blk: usize, row: usize) -> [f32; Q8K_WEIGHTS] {
    assert!(row < Q8K_ROWS, "a tile holds eight rows");
    assert!(
        n_blk <= Q8K_WEIGHTS / Q5_0_WEIGHTS,
        "a super-block is at most eight Q5_0 blocks"
    );
    let mut w = [0.0f32; Q8K_WEIGHTS];
    for b in 0..n_blk {
        let d = f16(u16::from_le_bytes([
            t[b * Q5_08::SIZE + crate::kq::q5_d() + row * 2],
            t[b * Q5_08::SIZE + crate::kq::q5_d() + row * 2 + 1],
        ]));
        for h in 0..Q5_0_WINDOWS {
            for k in 0..8 {
                let qh = t[b * Q5_08::SIZE + crate::kq::q5_qh() + h * 8 + row];
                let qs = b * Q5_08::SIZE
                    + crate::kq::q5_qs()
                    + (h % 2) * crate::kq::Q5_WINDOW
                    + row * crate::kq::Q5_QS_ROW;
                let nib = if h < 2 {
                    t[qs + k] & 0x0F
                } else {
                    t[qs + k] >> 4
                };
                let code = (nib + 16 * ((qh >> k) & 1)) as i32;
                w[b * Q5_0_WEIGHTS + h * 8 + k] = d * (code - 16) as f32;
            }
        }
    }
    w
}

/// The fork's Q4_K/Q5_0 requantisation of one (row, super-block): one f32 scale and
/// 256 signed codes, from the block's 256 f32 weights.
///
/// `dq = max|w| / 127` (a division), `id = 1/dq` (a division, 0 when `dq` is 0),
/// `code = round_to_nearest_even(w · id)` clamped to int8 the way the fork's
/// saturating packs are. The clamp cannot bite: `|w| ≤ max`, so `|w·id| ≤ 127`
/// up to the two roundings of the reciprocal, and rounding that reaches 128
/// saturates to 127 exactly as `vpacksdw` does.
pub fn requant_f32(w: &[f32; Q8K_WEIGHTS]) -> (f32, [i8; Q8K_WEIGHTS]) {
    let mut m = 0.0f32;
    for &x in w.iter() {
        let a = x.abs();
        if a > m {
            m = a;
        }
    }
    let dq = m / 127.0f32;
    let id = if dq != 0.0 { 1.0f32 / dq } else { 0.0 };
    let mut codes = [0i8; Q8K_WEIGHTS];
    for (c, &x) in codes.iter_mut().zip(w.iter()) {
        *c = round_i8(x * id);
    }
    (dq, codes)
}

/// The fork's Q6_K requantisation of one (row, super-block): its 256 integer
/// products `(q − 32)·L` and the block's `d` in, one f32 scale and 256 codes out.
///
/// The exact case is the fork's: `dnew = max|p| · (1/127)` and, when `dnew < 1`
/// (so `max|p| ≤ 126`, and every product fits int8), the codes are the products
/// and the scale is `d`. Otherwise the codes are `round(p · 1/dnew)` and the
/// scale is `d · dnew`. An all-zero block takes the exact branch and keeps the
/// scale `d`, not 0 — the codes are 0, so the product is 0 either way.
pub fn requant_q6k(p: &[i32; Q8K_WEIGHTS], d: f32) -> (f32, [i8; Q8K_WEIGHTS]) {
    let mut m = 0i32;
    for &x in p.iter() {
        let a = x.abs();
        if a > m {
            m = a;
        }
    }
    // The fork's `d0 = 1.f/127` and `dnew = hmax · d0`.
    let dnew = m as f32 * (1.0f32 / 127.0f32);
    let mut codes = [0i8; Q8K_WEIGHTS];
    if dnew < 1.0 {
        for (c, &x) in codes.iter_mut().zip(p.iter()) {
            *c = x as i8;
        }
        return (d, codes);
    }
    let id = if dnew.abs() > 1e-9 {
        1.0f32 / dnew
    } else {
        0.0
    };
    for (c, &x) in codes.iter_mut().zip(p.iter()) {
        *c = round_i8(x as f32 * id);
    }
    (d * dnew, codes)
}

/// `round_to_nearest_even` then a saturating int8 pack: the fork's
/// `_mm256_cvtps_epi32(_mm256_round_ps(x, _MM_ROUND_NEAREST))` followed by
/// `vpacksdw`/`vpacksswb` (`PackerQ80::convert`).
#[inline]
fn round_i8(x: f32) -> i8 {
    x.round_ties_even().clamp(-128.0, 127.0) as i8
}

/// Store one (row, super-block)'s scale and codes into the form.
fn store(form: &mut [u8], n_sb: usize, sb: usize, row: usize, scale: f32, codes: &[i8; 256]) {
    assert!(form.len() >= n_sb * Q8K_SIZE, "the form is short");
    let at = scale_at(sb, row);
    form[at..at + 4].copy_from_slice(&scale.to_le_bytes());
    for g in 0..Q8K_GROUPS {
        for k in 0..8 {
            // The bias: `vpdpbusd`'s first operand is unsigned, so the stored byte
            // is `code + 128` (the fork's xor with 0x80).
            form[code_at(sb, g, row, k)] = (codes[g * 8 + k] as u8) ^ 0x80;
        }
    }
}

// ---------------------------------------------------------------------------
// The scalar twin of the build
// ---------------------------------------------------------------------------

/// Build the 8-bit form of `n_sb` repacked Q4_K super-blocks.
///
/// # Panics
///
/// If the tile is shorter than `n_sb` super-blocks or the form is shorter than
/// `n_sb · Q8K_SIZE`.
pub fn build_q4k(tile: &[u8], n_sb: usize, form: &mut [u8]) {
    assert!(
        tile.len() >= n_sb * Q4K8::SIZE,
        "the tile holds fewer than n_sb super-blocks"
    );
    assert!(
        form.len() >= n_sb * Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    for sb in 0..n_sb {
        let t = &tile[sb * Q4K8::SIZE..(sb + 1) * Q4K8::SIZE];
        for row in 0..Q8K_ROWS {
            let w = q4k_weights(t, row);
            let (d, codes) = requant_f32(&w);
            store(form, n_sb, sb, row, d, &codes);
        }
    }
}

/// The 256 integer products `(q − 32)·L` of one row of a repacked Q6_K
/// super-block, in file order.
///
/// Q6_K has one int8 scale per sixteen weights and the products are exact in
/// int16; the super-block's `d` is not part of them (the fork multiplies by it in
/// the scale instead, `iqk_common.h:648-651`).
///
/// # Panics
///
/// If `row` is not one of the tile's eight rows.
pub fn q6k_products(t: &[u8], row: usize) -> [i32; Q8K_WEIGHTS] {
    assert!(row < Q8K_ROWS, "a tile holds eight rows");
    let mut p = [0i32; Q8K_WEIGHTS];
    for run in 0..16 {
        let c = run / 8;
        let g = (run % 8) / 2;
        let hf = run % 2;
        let nib_at = if g % 2 == 0 { 0 } else { 32 };
        let high = g >= 2;
        let sc = t[crate::kq::q6_sc() + run * 8 + row] as i8 as i32;
        for l in 0..16 {
            let li = hf * 16 + l;
            let ql =
                crate::kq::q6_ql() + (c * 8 + (nib_at + li) / 8) * 64 + row * 8 + (nib_at + li) % 8;
            let qh = crate::kq::q6_qh() + (c * 4 + li / 8) * 64 + row * 8 + li % 8;
            let nib = if high { t[ql] >> 4 } else { t[ql] & 0x0F };
            let code = (nib | (((t[qh] >> (2 * g)) & 3) << 4)) as i32 - 32;
            p[run * 16 + l] = code * sc;
        }
    }
    p
}

/// Build the 8-bit form of `n_sb` repacked Q6_K super-blocks.
///
/// # Panics
///
/// If the tile is shorter than `n_sb` super-blocks or the form is shorter than
/// `n_sb · Q8K_SIZE`.
pub fn build_q6k(tile: &[u8], n_sb: usize, form: &mut [u8]) {
    assert!(
        tile.len() >= n_sb * Q6K8::SIZE,
        "the tile holds fewer than n_sb super-blocks"
    );
    assert!(
        form.len() >= n_sb * Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    for sb in 0..n_sb {
        let t = &tile[sb * Q6K8::SIZE..(sb + 1) * Q6K8::SIZE];
        for row in 0..Q8K_ROWS {
            let d = f16(u16::from_le_bytes([
                t[crate::kq::q6_d() + row * 2],
                t[crate::kq::q6_d() + row * 2 + 1],
            ]));
            let p = q6k_products(t, row);
            let (scale, codes) = requant_q6k(&p, d);
            store(form, n_sb, sb, row, scale, &codes);
        }
    }
}

/// Build the 8-bit form of `n_blk` repacked Q5_0 blocks, eight to a 256-weight
/// super-block.
///
/// Q5_0 keeps one scale per 32 weights in the file and the form keeps one per 256,
/// as it does for the k-quants: the same construction, so the same kernel reads it.
/// A row whose width is not a multiple of 256 (Qwen 2.5's Q5_0 matrices are 896
/// wide, i.e. three and a half super-blocks) ends in a short super-block: its
/// missing weights are zero and its scale is the largest magnitude of the weights
/// that are there. Multiplying such a form needs the activation padded to a whole
/// super-block, which the kernel-only step of S5.4a does not build.
///
/// # Panics
///
/// If the tile is shorter than `n_blk` blocks or the form is shorter than the
/// super-blocks it needs.
#[cfg(feature = "fmt-legacy")]
pub fn build_q5_0(tile: &[u8], n_blk: usize, form: &mut [u8]) {
    assert!(
        tile.len() >= n_blk * Q5_08::SIZE,
        "the tile holds fewer than n_blk blocks"
    );
    let n_sb = n_blk.div_ceil(Q8K_WEIGHTS / Q5_0_WEIGHTS);
    assert!(
        form.len() >= n_sb * Q8K_SIZE,
        "the form holds fewer than n_blk blocks"
    );
    for sb in 0..n_sb {
        let b0 = sb * 8;
        let nb = (n_blk - b0).min(8);
        let t = &tile[b0 * Q5_08::SIZE..(b0 + nb) * Q5_08::SIZE];
        for row in 0..Q8K_ROWS {
            let w = q5_0_weights(t, nb, row);
            let (d, codes) = requant_f32(&w);
            store(form, n_sb, sb, row, d, &codes);
        }
    }
}

// ---------------------------------------------------------------------------
// The scalar twin of the multiply
// ---------------------------------------------------------------------------

/// One output row of the form against one activation row: the definition.
///
/// ```text
/// value = Σ_sb (d[sb] · dk[sb]) · (Σ_i code_i · a_i)
/// ```
///
/// with the inner sum an exact integer (it is at most `256 · 127 · 127 = 4 129 024`,
/// inside int32 and exactly representable in f32) and the epilogue one multiply
/// and one fused multiply-add per (row, super-block). The activation is the
/// interleaved [`KqActs`] buffer the exact Q4_K and Q6_K kernels read: row `arow`,
/// super-block `sb`, weight `256·sb + 8·g + k` at
/// `q[(g · n_sb + sb) · 1024 + g · 32 + (arow % 4) · 8 + k]`, scaled by
/// `d[(g · n_sb + sb) · 4 + arow % 4]`, with `g = arow / 4`.
///
/// # Panics
///
/// If `row` is not one of the eight rows, `arow` is not a filled activation row,
/// or either buffer is shorter than `n_sb` super-blocks.
pub fn dot_q8k(form: &[u8], n_sb: usize, row: usize, acts: &KqActs, arow: usize) -> f32 {
    assert!(row < Q8K_ROWS, "a tile holds eight rows");
    assert!(
        form.len() >= n_sb * Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    assert!(arow < acts.rows, "no such activation row");
    assert!(
        acts.super_blocks() >= n_sb,
        "the activation is shorter than the weights"
    );
    let ag = arow / ACT_GROUP;
    let at = arow % ACT_GROUP;
    let mut acc = 0.0f32;
    for sb in 0..n_sb {
        let d = f32::from_le_bytes(
            form[scale_at(sb, row)..scale_at(sb, row) + 4]
                .try_into()
                .expect("four bytes"),
        );
        let mut s = 0i32;
        for g in 0..Q8K_GROUPS {
            for k in 0..8 {
                let c = form[code_at(sb, g, row, k)] as i32 - 128;
                let a = acts.q[(ag * n_sb + sb) * 1024 + g * 32 + at * 8 + k] as i32;
                s += c * a;
            }
        }
        let dk = acts.d[(ag * n_sb + sb) * ACT_GROUP + at];
        let x = d * dk;
        acc = (s as f32).mul_add(x, acc);
    }
    acc
}

/// One output row of the form against one plain [`QuantAct`] row: the same real
/// number as [`dot_q8k`], for callers that hold one activation row and not the
/// interleaved buffer.
///
/// This is the form the accuracy report of §4 will want: a per-row dot product.
///
/// # Panics
///
/// If `row` is not one of the eight rows, or the activation is shorter than the
/// weights.
pub fn dot_q8k_plain(form: &[u8], n_sb: usize, row: usize, act: &QuantAct) -> f32 {
    assert!(row < Q8K_ROWS, "a tile holds eight rows");
    assert!(
        form.len() >= n_sb * Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    assert!(
        act.qk.len() >= n_sb * Q8K_WEIGHTS && act.dk.len() >= n_sb,
        "the activation is shorter than the weights"
    );
    let mut acc = 0.0f32;
    for sb in 0..n_sb {
        let d = f32::from_le_bytes(
            form[scale_at(sb, row)..scale_at(sb, row) + 4]
                .try_into()
                .expect("four bytes"),
        );
        let mut s = 0i32;
        for g in 0..Q8K_GROUPS {
            for k in 0..8 {
                let c = form[code_at(sb, g, row, k)] as i32 - 128;
                s += c * act.qk[sb * Q8K_WEIGHTS + g * 8 + k] as i32;
            }
        }
        let x = d * act.dk[sb];
        acc = (s as f32).mul_add(x, acc);
    }
    acc
}

/// A batch of activation rows against a run of eight output rows of one form:
/// the scalar twin of the AVX-512 `q8k_gemm_zmm` (`crate::kqsimd`).
///
/// `out[b · 8 + r]` receives row `r` of the form for activation row `b`, the
/// layout every repacked tile kernel writes. The body is [`dot_q8k`]: the
/// definition of the multiply. It is what the mode runs on a tier with no
/// SIMD body yet (AVX2 and NEON are S5.4d), so on those tiers the fast prompt
/// mode is correct and no faster than the exact one.
///
/// # Panics
///
/// If `nb` exceeds the activation's rows, the form is shorter than `n_sb`
/// super-blocks, or `out` is shorter than `nb · 8`.
pub fn gemm_q8k(form: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
    assert!(
        form.len() >= n_sb * Q8K_SIZE,
        "the form holds fewer than n_sb super-blocks"
    );
    assert!(nb <= acts.rows, "more activations than the buffer holds");
    assert!(
        acts.super_blocks() >= n_sb,
        "the activation is shorter than the weights"
    );
    assert!(
        out.len() >= nb * Q8K_ROWS,
        "the output holds fewer than eight rows per activation"
    );
    for b in 0..nb {
        for row in 0..Q8K_ROWS {
            out[b * Q8K_ROWS + row] = dot_q8k(form, n_sb, row, acts, b);
        }
    }
}

/// Tile builders for the tests: random and hand-built edge tiles.
///
/// They are shared by this module's tests and by the AVX-512 bodies' tests in
/// `crate::kqsimd`, so an edge case is defined once and the two paths are held
/// to the same input.
#[cfg(test)]
pub(crate) mod tiles {
    use super::*;
    use crate::kq::{repack_q4k8, repack_q6k8, Q4K_BLOCK, Q5_0_BLOCK, Q6K_BLOCK};
    use crate::quant::f32_to_f16;

    /// A seeded generator, so a failure is reproducible.
    pub struct Rng(pub u64);

    impl Rng {
        pub fn next_u8(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 24) as u8
        }

        pub fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next_u8()).collect()
        }
    }

    /// One Q4_K block with a sane scale and random codes and 6-bit scales.
    pub fn q4k_block(rng: &mut Rng) -> Vec<u8> {
        let mut b = rng.bytes(Q4K_BLOCK);
        b[0..2].copy_from_slice(&f32_to_f16(0.01).to_le_bytes());
        b[2..4].copy_from_slice(&f32_to_f16(0.005).to_le_bytes());
        for i in 0..12 {
            b[4 + i] = rng.next_u8() & 0x3F;
        }
        b
    }

    /// One Q6_K block with a sane scale and random codes and 8-bit scales.
    pub fn q6k_block(rng: &mut Rng) -> Vec<u8> {
        let mut b = rng.bytes(Q6K_BLOCK);
        b[208..210].copy_from_slice(&f32_to_f16(0.01).to_le_bytes());
        b
    }

    /// One Q5_0 block with a sane scale and random codes.
    #[cfg(feature = "fmt-legacy")]
    pub fn q5_0_block(rng: &mut Rng) -> Vec<u8> {
        let mut b = rng.bytes(Q5_0_BLOCK);
        b[0..2].copy_from_slice(&f32_to_f16(0.02).to_le_bytes());
        b
    }

    /// `n_sb` super-blocks of eight random Q4_K rows.
    pub fn q4k_tile(rng: &mut Rng, n_sb: usize) -> Vec<u8> {
        let mut tile = vec![0u8; n_sb * Q4K8::SIZE];
        for sb in 0..n_sb {
            let rows: Vec<Vec<u8>> = (0..8).map(|_| q4k_block(rng)).collect();
            let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
            repack_q4k8(&refs, &mut tile[sb * Q4K8::SIZE..(sb + 1) * Q4K8::SIZE]);
        }
        tile
    }

    /// `n_sb` super-blocks of eight random Q6_K rows.
    pub fn q6k_tile(rng: &mut Rng, n_sb: usize) -> Vec<u8> {
        let mut tile = vec![0u8; n_sb * Q6K8::SIZE];
        for sb in 0..n_sb {
            let rows: Vec<Vec<u8>> = (0..8).map(|_| q6k_block(rng)).collect();
            let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
            repack_q6k8(&refs, &mut tile[sb * Q6K8::SIZE..(sb + 1) * Q6K8::SIZE]);
        }
        tile
    }

    /// `n_blk` random Q5_0 blocks of eight rows.
    #[cfg(feature = "fmt-legacy")]
    pub fn q5_08_tile(rng: &mut Rng, n_blk: usize) -> Vec<u8> {
        let mut tile = vec![0u8; n_blk * Q5_08::SIZE];
        for t in 0..n_blk {
            let rows: Vec<Vec<u8>> = (0..8).map(|_| q5_0_block(rng)).collect();
            let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
            crate::kq::repack_q5_08(&refs, &mut tile[t * Q5_08::SIZE..(t + 1) * Q5_08::SIZE]);
        }
        tile
    }

    /// The file's own six-bit scale/min packing (`ggml`'s `quantize_row_q4_K_ref`,
    /// which `get_scale_min_k4` undoes).
    pub fn pack_scales(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
        let mut q = [0u8; 12];
        for j in 0..4 {
            q[j] = sc[j] & 0x3F;
            q[j + 4] = m[j] & 0x3F;
        }
        for j in 4..8 {
            q[j + 4] = (sc[j] & 0xF) | ((m[j] & 0xF) << 4);
            q[j - 4] |= (sc[j] >> 4) << 6;
            q[j] |= (m[j] >> 4) << 6;
        }
        q
    }

    /// One Q4_K block with chosen scales, mins and codes, in the file's layout.
    pub fn q4k_block_with(
        d: f32,
        dmin: f32,
        sc: &[u8; 8],
        m: &[u8; 8],
        codes: &[[u8; 32]; 8],
    ) -> Vec<u8> {
        let mut b = vec![0u8; Q4K_BLOCK];
        b[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        b[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
        b[4..16].copy_from_slice(&pack_scales(sc, m));
        // Sub-block j's 32 codes are the low nibbles of the row's bytes
        // 32*(j/2) .. +32 when j is even, the high nibbles when it is odd.
        for j in 0..8 {
            for l in 0..32 {
                let at = 16 + (j / 2) * 32 + l;
                if j % 2 == 0 {
                    b[at] = (b[at] & 0xF0) | (codes[j][l] & 0x0F);
                } else {
                    b[at] = (b[at] & 0x0F) | ((codes[j][l] & 0x0F) << 4);
                }
            }
        }
        b
    }

    /// One Q6_K block with chosen per-16 scales and six-bit codes (0..=63), in the
    /// file's layout.
    pub fn q6k_block_with(d: f32, sc: &[i8; 16], codes: &[u8; 256]) -> Vec<u8> {
        let mut b = vec![0u8; Q6K_BLOCK];
        b[208..210].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for run in 0..16 {
            b[192 + run] = sc[run] as u8;
        }
        for half in 0..2 {
            let base = half * 128;
            for l in 0..32 {
                let c0 = codes[base + l] & 0x0F;
                let c1 = codes[base + l + 32] & 0x0F;
                let c2 = codes[base + l + 64] & 0x0F;
                let c3 = codes[base + l + 96] & 0x0F;
                b[half * 64 + l] = c0 | (c2 << 4);
                b[half * 64 + l + 32] = c1 | (c3 << 4);
                let h0 = codes[base + l] >> 4;
                let h1 = codes[base + l + 32] >> 4;
                let h2 = codes[base + l + 64] >> 4;
                let h3 = codes[base + l + 96] >> 4;
                b[128 + half * 32 + l] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }
        b
    }

    /// A tile of eight rows, all of them `block`, repacked.
    pub fn tile_of(block: &[u8], size: usize, repack: fn(&[&[u8]], &mut [u8])) -> Vec<u8> {
        let blocks = vec![block.to_vec(); 8];
        let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
        let mut tile = vec![0u8; size];
        repack(&refs, &mut tile);
        tile
    }

    /// An all-zero Q4_K super-block.
    pub fn zero_tile() -> Vec<u8> {
        tile_of(&[0u8; Q4K_BLOCK], Q4K8::SIZE, repack_q4k8)
    }

    /// One weight of 945 and 255 zeros.
    pub fn huge_tile() -> Vec<u8> {
        let mut sc = [0u8; 8];
        let m = [0u8; 8];
        let mut codes = [[0u8; 32]; 8];
        sc[0] = 63;
        codes[0][0] = 15;
        let block = q4k_block_with(1.0, 0.0, &sc, &m, &codes);
        tile_of(&block, Q4K8::SIZE, repack_q4k8)
    }

    /// Every 6-bit scale and min at 63, with codes spread over all sixteen values.
    pub fn limits_tile() -> Vec<u8> {
        let sc = [63u8; 8];
        let m = [63u8; 8];
        let mut codes = [[0u8; 32]; 8];
        for (j, c) in codes.iter_mut().enumerate() {
            for (l, v) in c.iter_mut().enumerate() {
                *v = ((j * 31 + l * 7) % 16) as u8;
            }
        }
        let block = q4k_block_with(0.3, 0.2, &sc, &m, &codes);
        tile_of(&block, Q4K8::SIZE, repack_q4k8)
    }

    /// A block where four weights land exactly on half a step: `d = 1`, `dmin = 1`,
    /// so a weight is `q·Ld − 1` (an integer), the largest is 254 and `dq = 2`,
    /// `id = 0.5` are exact in f32.
    pub fn tie_tile() -> Vec<u8> {
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        let mut codes = [[0u8; 32]; 8];
        sc[0] = 17;
        m[0] = 1;
        codes[0][0] = 15; // 254
        sc[1] = 1;
        m[1] = 1;
        codes[1][0] = 2; // 1   -> 0.5 steps
        codes[1][1] = 4; // 3   -> 1.5
        codes[1][2] = 6; // 5   -> 2.5
        codes[1][3] = 8; // 7   -> 3.5
        codes[1][4] = 0; // -1  -> -0.5
        let block = q4k_block_with(1.0, 1.0, &sc, &m, &codes);
        tile_of(&block, Q4K8::SIZE, repack_q4k8)
    }

    /// A Q6_K super-block inside the exact case: every product fits in int8.
    pub fn exact_q6k_tile() -> Vec<u8> {
        let mut sc = [3i8; 16];
        sc[5] = -3;
        let mut codes = [0u8; 256];
        for (i, c) in codes.iter_mut().enumerate() {
            *c = 32 + ((i as i32 % 64) - 32).clamp(-31, 31) as u8;
        }
        let block = q6k_block_with(0.007, &sc, &codes);
        tile_of(&block, Q6K8::SIZE, repack_q6k8)
    }

    /// A Q6_K super-block just outside the exact case: the largest product is 128.
    pub fn outside_q6k_tile() -> Vec<u8> {
        let mut sc = [2i8; 16];
        sc[0] = 8;
        let mut codes = [32u8; 256];
        codes[0] = 48; // (48 - 32) * 8 = 128
        codes[1] = 33; // 8
        let block = q6k_block_with(0.5, &sc, &codes);
        tile_of(&block, Q6K8::SIZE, repack_q6k8)
    }
}

#[cfg(test)]
mod tests {
    use super::tiles::*;
    use super::*;
    use crate::kq::{Q4K8, Q5K8};

    /// The form's own scale for one (row, super-block).
    pub fn scale_of(form: &[u8], sb: usize, row: usize) -> f32 {
        f32::from_le_bytes(
            form[scale_at(sb, row)..scale_at(sb, row) + 4]
                .try_into()
                .unwrap(),
        )
    }

    /// The form's code for one (row, super-block, weight), unbiassed.
    pub fn code_of(form: &[u8], sb: usize, row: usize, i: usize) -> i32 {
        form[code_at(sb, i / 8, row, i % 8)] as i32 - 128
    }

    /// Every re-rounded weight is within half a step of the f32 weight, and the
    /// worst case reaches it.
    ///
    /// The step is `max|w|/127`, so the bound is `max|w|/254` — the design note's
    /// §11.1 bound, over the twin's own weights.
    #[test]
    fn every_rerounded_weight_is_within_half_a_step() {
        let mut rng = Rng(0x5EED_8A11);
        let mut worst: f32 = 0.0;
        let mut weights = 0usize;
        for _ in 0..40 {
            let n_sb = 4;
            let tile = q4k_tile(&mut rng, n_sb);
            let mut form = vec![0u8; form_size(n_sb)];
            build_q4k(&tile, n_sb, &mut form);
            for sb in 0..n_sb {
                let t = &tile[sb * Q4K8::SIZE..(sb + 1) * Q4K8::SIZE];
                for row in 0..8 {
                    let w = q4k_weights(t, row);
                    let m = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                    let d = scale_of(&form, sb, row);
                    let step = m / 127.0;
                    assert!(
                        (d - step).abs() <= 4.0 * f32::EPSILON * step,
                        "the scale is not max/127"
                    );
                    for (i, &x) in w.iter().enumerate() {
                        let got = code_of(&form, sb, row, i) as f32 * d;
                        let err = (x - got).abs();
                        assert!(
                            err <= m / 254.0 * (1.0 + 1e-5),
                            "weight {i} is {err} off, bound {}",
                            m / 254.0
                        );
                        worst = worst.max(err / m);
                        weights += 1;
                    }
                }
            }
        }
        // The worst case reaches the bound: with 256 weights per block the largest
        // error is within a few per mille of a half step.
        assert!(
            worst >= 0.49 / 127.0,
            "the worst error was {worst} of the block maximum, and the bound is {}",
            0.5 / 127.0
        );
        assert!(weights > 300_000);
    }

    /// An all-zero block: scale 0, codes 0 (and the stored byte is the bias, 128).
    #[test]
    fn the_all_zero_block_has_a_zero_scale_and_zero_codes() {
        let tile = zero_tile();
        let mut form = vec![0u8; form_size(1)];
        build_q4k(&tile, 1, &mut form);
        for row in 0..8 {
            assert_eq!(scale_of(&form, 0, row).to_bits(), 0.0f32.to_bits());
            for i in 0..Q8K_WEIGHTS {
                assert_eq!(code_of(&form, 0, row, i), 0);
                assert_eq!(form[code_at(0, i / 8, row, i % 8)], 128);
            }
        }
    }

    /// One huge weight and nothing else: the scale is that weight over 127 and every
    /// other code is 0.
    #[test]
    fn one_huge_weight_sets_the_scale() {
        let tile = huge_tile();
        let mut form = vec![0u8; form_size(1)];
        build_q4k(&tile, 1, &mut form);
        let w = q4k_weights(&tile, 0);
        assert_eq!(w[0], 945.0);
        for row in 0..8 {
            let d = scale_of(&form, 0, row);
            assert!((d - 945.0 / 127.0).abs() <= 1e-6);
            assert_eq!(code_of(&form, 0, row, 0), 127);
            for i in 1..Q8K_WEIGHTS {
                assert_eq!(code_of(&form, 0, row, i), 0);
            }
        }
    }

    /// Scales and mins at their six-bit limits: the weights are large and the codes
    /// still fit.
    #[test]
    fn six_bit_limits() {
        let tile = limits_tile();
        let mut form = vec![0u8; form_size(1)];
        build_q4k(&tile, 1, &mut form);
        for row in 0..8 {
            let w = q4k_weights(&tile, row);
            let d = scale_of(&form, 0, row);
            let big = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            assert!(big > 250.0);
            for (i, &x) in w.iter().enumerate() {
                let err = (x - code_of(&form, 0, row, i) as f32 * d).abs();
                assert!(err <= big / 254.0 * (1.0 + 1e-5));
            }
        }
    }

    /// A weight that lands exactly on half a step rounds to even, in both
    /// directions. The tile's arithmetic is exact: `dq = 2` and `id = 0.5`.
    #[test]
    fn a_weight_on_half_a_step_rounds_to_even() {
        let tile = tie_tile();
        let mut form = vec![0u8; form_size(1)];
        build_q4k(&tile, 1, &mut form);
        assert_eq!(scale_of(&form, 0, 0), 2.0);
        assert_eq!(code_of(&form, 0, 0, 0), 127);
        assert_eq!(code_of(&form, 0, 0, 32), 0);
        assert_eq!(code_of(&form, 0, 0, 33), 2);
        assert_eq!(code_of(&form, 0, 0, 34), 2);
        assert_eq!(code_of(&form, 0, 0, 35), 4);
        assert_eq!(code_of(&form, 0, 0, 36), 0);
    }

    /// The rounding helper is round-to-nearest-even, which is what the fork's
    /// `_MM_ROUND_NEAREST` is, and the saturating pack cannot fire.
    #[test]
    fn the_rounding_is_to_even() {
        assert_eq!(round_i8(0.5), 0);
        assert_eq!(round_i8(1.5), 2);
        assert_eq!(round_i8(2.5), 2);
        assert_eq!(round_i8(-0.5), 0);
        assert_eq!(round_i8(-1.5), -2);
        assert_eq!(round_i8(-2.5), -2);
        assert_eq!(round_i8(127.0), 127);
        assert_eq!(round_i8(127.4), 127);
        assert_eq!(round_i8(127.5), 127);
        assert_eq!(round_i8(-128.4), -128);
    }

    /// Q6_K's exact case: when every integer product fits in int8 the codes are the
    /// products and the scale is the block's `d`, with no rounding at all.
    #[test]
    fn q6k_keeps_the_exact_case() {
        let tile = exact_q6k_tile();
        let mut form = vec![0u8; form_size(1)];
        build_q6k(&tile, 1, &mut form);
        for row in 0..8 {
            let p = q6k_products(&tile, row);
            let m = p.iter().fold(0i32, |a, &x| a.max(x.abs()));
            assert!(m <= 126, "the block is not in the exact case: {m}");
            let d16 = f16(u16::from_le_bytes([tile[row * 2], tile[row * 2 + 1]]));
            assert_eq!(scale_of(&form, 0, row), d16);
            for i in 0..Q8K_WEIGHTS {
                assert_eq!(code_of(&form, 0, row, i), p[i]);
            }
        }
    }

    /// One just outside the exact case: the largest product is 128, so the block is
    /// rounded and the scale is `d·max/127`.
    #[test]
    fn q6k_rounds_just_outside_the_exact_case() {
        let tile = outside_q6k_tile();
        let mut form = vec![0u8; form_size(1)];
        build_q6k(&tile, 1, &mut form);
        let p = q6k_products(&tile, 0);
        assert_eq!(p[0], 128);
        let dnew = 128.0f32 * (1.0f32 / 127.0f32);
        assert!(dnew >= 1.0);
        for row in 0..8 {
            assert_eq!(scale_of(&form, 0, row), 0.5 * dnew);
            assert_eq!(code_of(&form, 0, row, 0), 127);
            assert_eq!(
                code_of(&form, 0, row, 1),
                round_i8(8.0 * (1.0 / dnew)) as i32
            );
        }
    }

    /// The bound on Q6_K too: every re-rounded product is within half a step of the
    /// product the file means, i.e. within `max|w|/254` of the weight.
    #[test]
    fn q6k_stays_within_half_a_step() {
        let mut rng = Rng(0x6EED_8A11);
        let mut worst = 0.0f32;
        for _ in 0..20 {
            let n_sb = 4;
            let tile = q6k_tile(&mut rng, n_sb);
            let mut form = vec![0u8; form_size(n_sb)];
            build_q6k(&tile, n_sb, &mut form);
            for sb in 0..n_sb {
                let t = &tile[sb * Q6K8::SIZE..(sb + 1) * Q6K8::SIZE];
                for row in 0..8 {
                    let d = f16(u16::from_le_bytes([
                        t[crate::kq::q6_d() + row * 2],
                        t[crate::kq::q6_d() + row * 2 + 1],
                    ]));
                    let p = q6k_products(t, row);
                    let m = p.iter().fold(0i32, |a, &x| a.max(x.abs()));
                    let scale = scale_of(&form, sb, row);
                    let top = m as f32 * d.abs();
                    for i in 0..Q8K_WEIGHTS {
                        let got = code_of(&form, sb, row, i) as f32 * scale;
                        let want = p[i] as f32 * d;
                        let err = (got - want).abs();
                        assert!(
                            err <= top / 254.0 * (1.0 + 1e-5) + f32::MIN_POSITIVE,
                            "Q6_K weight {i}: {err} off, bound {}",
                            top / 254.0
                        );
                        if top > 0.0 {
                            worst = worst.max(err / top);
                        }
                    }
                }
            }
        }
        assert!(worst >= 0.45 / 127.0, "worst {worst}");
    }

    /// Q5_0 goes through the same construction and keeps the same bound.
    #[cfg(feature = "fmt-legacy")]
    #[test]
    fn q5_0_keeps_the_bound() {
        let mut rng = Rng(0x50ED_8A11);
        let mut worst = 0.0f32;
        for _ in 0..20 {
            let n_blk = 16;
            let tile = q5_08_tile(&mut rng, n_blk);
            let mut form = vec![0u8; n_blk / 8 * Q8K_SIZE];
            build_q5_0(&tile, n_blk, &mut form);
            for sb in 0..n_blk / 8 {
                let t = &tile[sb * 8 * Q5_08::SIZE..(sb + 1) * 8 * Q5_08::SIZE];
                for row in 0..8 {
                    let w = q5_0_weights(t, 8, row);
                    let m = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                    let d = scale_of(&form, sb, row);
                    assert!((d - m / 127.0).abs() <= 8.0 * f32::EPSILON * (m / 127.0));
                    for (i, &x) in w.iter().enumerate() {
                        let err = (x - code_of(&form, sb, row, i) as f32 * d).abs();
                        assert!(err <= m / 254.0 * (1.0 + 1e-5));
                        if m > 0.0 {
                            worst = worst.max(err / m);
                        }
                    }
                }
            }
        }
        assert!(worst >= 0.45 / 127.0, "worst {worst}");
    }

    /// The form's geometry: every offset is inside its super-block, the codes fill
    /// exactly the 2048 bytes before the scales, and a super-block is 33 cache lines.
    #[test]
    fn the_geometry_is_what_the_docs_say() {
        assert_eq!(Q8K_SIZE, 33 * 64);
        assert_eq!(Q8K_D, 2048);
        assert_eq!(Q8K_GROUPS * 8, Q8K_WEIGHTS);
        assert_eq!(code_at(0, 31, 7, 7), 2047);
        assert_eq!(scale_at(0, 7), 2048 + 28);
        assert_eq!(form_size(3), 3 * Q8K_SIZE);
        assert!(code_at(2, 31, 7, 7) < form_size(3));
        // The form is one byte per weight plus the scales and the padding: 1.82x a
        // Q4_K file's bytes and 1.26x a Q6_K file's.
        assert_eq!(Q8K_SIZE as f32 / 2048.0, 1.03125);
        assert_eq!(Q5K8::SIZE, 1408);
    }

    /// The multiply's twin against the plain activation path: the two activation
    /// layouts give the same bits.
    #[test]
    fn the_two_activation_layouts_agree() {
        let mut rng = Rng(0xA11CE);
        let n_sb = 3;
        let tile = q4k_tile(&mut rng, n_sb);
        let mut form = vec![0u8; form_size(n_sb)];
        build_q4k(&tile, n_sb, &mut form);
        let rows = 9;
        let acts: Vec<QuantAct> = (0..rows)
            .map(|_| {
                let x: Vec<f32> = (0..n_sb * 256)
                    .map(|_| (rng.next_u8() as f32 / 127.0 - 1.0) * 2.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n_sb * 256);
                a.fill(&x);
                a
            })
            .collect();
        let mut kq = KqActs::default();
        kq.fill(&acts);
        for row in 0..8 {
            for r in 0..rows {
                let a = dot_q8k(&form, n_sb, row, &kq, r);
                let b = dot_q8k_plain(&form, n_sb, row, &acts[r]);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "row {row}, activation {r}: {a} against {b}"
                );
            }
        }
    }

    /// The shared super-block sum the kernel's bias needs is the activation's own.
    #[test]
    fn the_activation_block_sum_is_exact() {
        let mut rng = Rng(0xB105);
        let n = 512;
        let x: Vec<f32> = (0..n)
            .map(|_| (rng.next_u8() as f32 / 127.0 - 1.0) * 3.0)
            .collect();
        let mut a = QuantAct::with_capacity(n);
        a.fill(&x);
        let mut kq = KqActs::default();
        kq.fill(&[a.clone()]);
        for sb in 0..n / 256 {
            let want: i32 = a.qk[sb * 256..sb * 256 + 256]
                .iter()
                .map(|&v| v as i32)
                .sum();
            assert_eq!(kq.sum256[sb * ACT_GROUP], want);
        }
    }
}
