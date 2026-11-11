//! The packed float GEMM (S4.9f): F32, F16 and BF16 weights in 16-output panels,
//! multiplied by a broadcast-form FMA micro-kernel.
//!
//! # Why
//!
//! Before this, a float matrix ran one weight row against four activation rows,
//! multiplied and added *without* FMA, and summed horizontally once per output:
//! 3.7-4.2× slower per multiply than the repacked integer tiles (measured, BF16).
//! The layout is the difference. A panel holds sixteen output rows side by side, so
//! one weight load feeds sixteen outputs, and every lane of an accumulator is a
//! different output — no horizontal sum, no per-output reduction.
//!
//! # The layout
//!
//! A matrix `[n_out][n_in]` is cut into panels of [`PANEL`] consecutive output rows;
//! a panel is `n_in` steps of 16 contiguous values, step `k` holding
//! `W[r0..r0+16][k]`, in the file's own element type (f32 words, f16 halves or bf16
//! halves). The last panel is zero-padded. [`pack_panel`] writes one panel;
//! `peregrine-core`'s `fpack` module lays out the load-time store.
//!
//! # The invariant: one chain, whatever computed it
//!
//! Every output element is exactly
//!
//! ```text
//! acc = bias[r] (or +0.0); for k in 0..n_in { acc = fma(x[k], w[r][k], acc) }
//! ```
//!
//! one sequential fused-multiply-add chain over the input index, and nothing else.
//! The vector kernels compute that chain lane per output, so a result's bits do not
//! depend on the tile, the tails, the row block, the thread split or the vector
//! width: AVX2, AVX-512, NEON and [`gemm_scalar`] — the same chain with
//! `f32::mul_add`, and the scalar tier's kernel — agree bit for bit. F16 is widened
//! by `vcvtph2ps` (AVX) or `fcvtl` (NEON) and BF16 by shifting into the high half of
//! an f32; both conversions are exact, so they contribute nothing to the rounding.
//!
//! Fusing changes the value of every float matrix product once (one rounding per
//! term instead of two, and a different summation order from four partial sums and
//! a horizontal fold), which is the one-time golden change S4.9f accepts.
//!
//! NaN payloads are the one thing the chain does not pin down: a hardware
//! `vfmadd` and a libm `fmaf` may quiet a NaN to different payload bits (and a
//! signalling NaN to different sign bits) while both returning a NaN. The value
//! class agrees; the payload does not. Real weights and activations carry no NaN.
//!
//! # Measured shapes
//!
//! AVX2 runs 6 activation rows × 16 outputs (12 ymm accumulators, two panel loads
//! per step). AVX-512 runs 12 × 32 (two panels at once, 24 zmm accumulators), which
//! amortises each activation broadcast over two panels; the measurements behind the
//! choice are in `docs/design/s4.9-sota.md` (§ S4.9f as built). This CPU executes
//! 512-bit operations on 256-bit datapaths, so the width buys registers and
//! broadcast amortisation, not twice the FLOPs — measured, 12×32 zmm and 12×16 ymm
//! are within a few percent per multiply, and 12×32 wins on the panel loads.
//! NEON runs 4 activation rows × 16 outputs (16 `float32x4_t` accumulators), which
//! is one panel step's four loads feeding all four rows.
//!
//! # Threading and the row block
//!
//! Nothing here threads. `peregrine-core`'s `blocks::linear` splits work over
//! (panel, activation-row block) units; the row block is [`row_block`], sized so a
//! block's activations stay in half of L2, and it only changes how many times a
//! panel is read, never a result. A single activation row takes the same GEMM loop
//! with one row on NEON; on AVX2 it takes the GEMV path ([`gemv_scalar`] here,
//! `gemv_avx2`/`gemv_avx512` beside it), which streams each panel once and keeps the
//! same chain per output, so decode and prefill agree bit for bit.

use crate::format::{f16, Format};
use crate::quant::QuantAct;

/// The output rows one panel holds.
pub const PANEL: usize = 16;

/// The bytes one weight element of a float format takes.
#[inline]
pub fn elem_bytes(fmt: Format) -> usize {
    match fmt {
        Format::F32 => 4,
        Format::F16 | Format::BF16 => 2,
        other => panic!("{} is not a float format", other.label()),
    }
}

/// The bytes one panel of `n_in` weights takes.
#[inline]
pub fn panel_bytes(fmt: Format, n_in: usize) -> usize {
    n_in * PANEL * elem_bytes(fmt)
}

/// The bytes `n_out` rows of `n_in` weights take packed (the last panel zero-padded).
#[inline]
pub fn packed_len(fmt: Format, n_out: usize, n_in: usize) -> usize {
    n_out.div_ceil(PANEL) * panel_bytes(fmt, n_in)
}

/// Pack one panel from up to [`PANEL`] weight rows.
///
/// `rows[r]` is output row `r` in the file's layout (`n_in` elements of `fmt`);
/// a panel with fewer rows keeps zeros where the missing rows would be, so the
/// padding multiplies zero and adds nothing.
pub fn pack_panel(fmt: Format, rows: &[&[u8]], n_in: usize, dst: &mut [u8]) {
    let eb = elem_bytes(fmt);
    let pb = panel_bytes(fmt, n_in);
    assert!(rows.len() <= PANEL, "a panel holds {PANEL} rows");
    assert!(dst.len() >= pb, "the destination is smaller than a panel");
    let dst = &mut dst[..pb];
    dst.fill(0);
    // One element a step, unrolled over the panel's rows: `dst[k · 16 + lane]`
    // receives `rows[lane][k]`. The copies are raw because every bound is checked
    // above, and the element size is a compile-time constant per arm so the copy is
    // one 2- or 4-byte move rather than a slice call.
    let stride = PANEL * eb;
    unsafe {
        let d = dst.as_mut_ptr();
        for row in rows {
            assert!(row.len() >= n_in * eb, "a weight row is shorter than n_in");
        }
        for k in 0..n_in {
            let mut at = k * stride;
            for row in rows {
                std::ptr::copy_nonoverlapping(row.as_ptr().add(k * eb), d.add(at), eb);
                at += eb;
            }
        }
    }
}

/// The weight at index `i` of a panel's step array (`i = k · PANEL + lane`), as f32.
#[inline(always)]
fn w_at(fmt: Format, p: *const u8, i: usize) -> f32 {
    unsafe {
        match fmt {
            Format::F32 => f32::from_le_bytes([
                *p.add(i * 4),
                *p.add(i * 4 + 1),
                *p.add(i * 4 + 2),
                *p.add(i * 4 + 3),
            ]),
            Format::F16 => f16(u16::from_le_bytes([*p.add(i * 2), *p.add(i * 2 + 1)])),
            _ => f32::from_bits(
                (u16::from_le_bytes([*p.add(i * 2), *p.add(i * 2 + 1)]) as u32) << 16,
            ),
        }
    }
}

/// The specification: every output element as one sequential `f32::mul_add` chain.
///
/// This is the reference the vector kernels are tested against and the kernel the
/// scalar tier runs. `out[b · panels · PANEL + p · PANEL + lane]` receives the
/// product of activation row `b` with output row `p · PANEL + lane`; `bias`, when
/// given, is one value per output row of the run (`panels · PANEL` of them) and is
/// where the chain starts.
pub fn gemm_scalar(
    fmt: Format,
    packed: &[u8],
    n_in: usize,
    panels: usize,
    acts: &[QuantAct],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    gemm_scalar_strided(fmt, packed, n_in, panels, acts, bias, out, panels * PANEL);
}

/// [`gemm_scalar`] writing each activation row `b` at `b * out_stride`.
///
/// `out_stride` is the row width of the caller's output buffer: the panel's rows
/// land at `out[b · out_stride + p · PANEL + lane]`, so a caller whose output is
/// already row-major (a projection's `[row][out]`) is written in place instead of
/// through a temporary and a copy.
#[allow(clippy::too_many_arguments)]
pub fn gemm_scalar_strided(
    fmt: Format,
    packed: &[u8],
    n_in: usize,
    panels: usize,
    acts: &[QuantAct],
    bias: Option<&[f32]>,
    out: &mut [f32],
    out_stride: usize,
) {
    let pb = panel_bytes(fmt, n_in);
    debug_assert!(packed.len() >= panels * pb);
    debug_assert!(out_stride >= panels * PANEL);
    debug_assert!(acts.is_empty() || out.len() >= (acts.len() - 1) * out_stride + panels * PANEL);
    for (b, a) in acts.iter().enumerate() {
        debug_assert!(
            a.keep_f32 && a.xf.len() >= n_in,
            "float matmul without f32 activations"
        );
        for p in 0..panels {
            let w = unsafe { packed.as_ptr().add(p * pb) };
            for lane in 0..PANEL {
                let r = p * PANEL + lane;
                let mut acc = bias.map_or(0.0f32, |v| v[r]);
                for k in 0..n_in {
                    let wv = w_at(fmt, w, k * PANEL + lane);
                    acc = a.xf[k].mul_add(wv, acc);
                }
                out[b * out_stride + r] = acc;
            }
        }
    }
}

/// The single-row specification: one activation row, one sequential chain per output.
pub fn gemv_scalar(
    fmt: Format,
    packed: &[u8],
    n_in: usize,
    panels: usize,
    act: &QuantAct,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    gemm_scalar(
        fmt,
        packed,
        n_in,
        panels,
        std::slice::from_ref(act),
        bias,
        out,
    );
}

/// The activation rows one cache block holds: about `L2/2 / (4·n_in)` rows.
///
/// Read only for blocking, so the number itself never reaches a result; a block
/// that does not fit still computes the same bits.
pub fn row_block(n_in: usize) -> usize {
    ((l2_bytes() / 2) / (4 * n_in.max(1))).clamp(1, 128)
}

/// The L2 size in bytes: `/sys/devices/system/cpu/cpu0/cache/index2/size` on Linux,
/// one megabyte otherwise.
fn l2_bytes() -> usize {
    use std::sync::OnceLock;
    static L2: OnceLock<usize> = OnceLock::new();
    *L2.get_or_init(|| {
        #[cfg(target_os = "linux")]
        if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size") {
            if let Some(v) = parse_cache_size(s.trim()) {
                return v;
            }
        }
        1 << 20
    })
}

/// `"1024K"`, `"2M"` or a plain byte count.
///
/// Only Linux reads the cache size, and only the unit test needs the parser
/// elsewhere; a non-Linux build with `-D warnings` must not call it dead code.
#[cfg(any(target_os = "linux", test))]
fn parse_cache_size(s: &str) -> Option<usize> {
    let (num, mult) = match s.as_bytes().last()? {
        b'K' | b'k' => (&s[..s.len() - 1], 1 << 10),
        b'M' | b'm' => (&s[..s.len() - 1], 1 << 20),
        _ => (s, 1),
    };
    num.parse::<usize>().ok().map(|v| v * mult)
}

/// Whether the packed float path may run on this target: x86-64 and aarch64. The
/// store is worth building on aarch64 even on the scalar tier, whose kernel is the
/// same `f32::mul_add` chain.
pub fn have_packed() -> bool {
    cfg!(any(target_arch = "x86_64", target_arch = "aarch64"))
}

/// The kernel for this CPU: GEMV for one activation row, else the widest GEMM.
///
/// # Safety
/// `packed` must hold `panels` panels of `n_in` weights, `out` room for
/// `panels · PANEL · acts.len()` values, and every activation `xf` at least `n_in`
/// floats; the caller (`peregrine-core`'s `blocks::linear`) guarantees all three.
pub fn gemm(
    fmt: Format,
    packed: &[u8],
    n_in: usize,
    panels: usize,
    acts: &[QuantAct],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    gemm_strided(fmt, packed, n_in, panels, acts, bias, out, panels * PANEL);
}

/// [`gemm`] with a caller-chosen output stride: activation row `b` is written at
/// `out[b · out_stride ..][..panels · PANEL]`.
///
/// The kernel computes the same chain; only where it stores differs, so the values
/// are the row-major kernel's bit for bit. A projection whose output buffer is
/// `[row][out]` passes `out_stride = out_rows` and lets the kernel write it
/// directly, with no temporary block and no copy.
#[allow(clippy::too_many_arguments)]
pub fn gemm_strided(
    fmt: Format,
    packed: &[u8],
    n_in: usize,
    panels: usize,
    acts: &[QuantAct],
    bias: Option<&[f32]>,
    out: &mut [f32],
    out_stride: usize,
) {
    assert!(matches!(fmt, Format::F32 | Format::F16 | Format::BF16));
    assert!(packed.len() >= panels * panel_bytes(fmt, n_in));
    assert!(
        out_stride >= panels * PANEL,
        "output stride {out_stride} is narrower than {panels} panels"
    );
    assert!(
        acts.is_empty() || out.len() >= (acts.len() - 1) * out_stride + panels * PANEL,
        "strided float matmul output too short"
    );
    assert!(bias.is_none_or(|b| b.len() >= panels * PANEL));
    assert!(
        acts.iter().all(|a| a.keep_f32 && a.xf.len() >= n_in),
        "float matrix matmul reached with keep_f32 off on QuantAct"
    );
    if panels == 0 || acts.is_empty() {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON: F32, BF16 and F16. The half-to-single widening (`fcvtl`) is base
        // Advanced SIMD; the `fp16` extension is about half-precision arithmetic,
        // which this kernel does not use.
        if crate::cpu::have_neon() {
            if matches!(fmt, Format::F16) {
                // SAFETY: the preconditions above are this function's.
                unsafe { neon::gemm_f16(fmt, packed, n_in, panels, acts, bias, out, out_stride) };
                return;
            }
            // SAFETY: the preconditions above are this function's.
            unsafe { neon::gemm_neon(fmt, packed, n_in, panels, acts, bias, out, out_stride) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if acts.len() == 1 && x86::have_gemv() {
            // SAFETY: the preconditions above are this function's.
            unsafe { x86::gemv(fmt, packed, n_in, panels, &acts[0], bias, out) };
            return;
        }
        if x86::have_zmm() {
            // SAFETY: the preconditions above are this function's.
            unsafe { x86::gemm_avx512(fmt, packed, n_in, panels, acts, bias, out, out_stride) };
            return;
        }
        if crate::simd::have_batched() {
            // SAFETY: the preconditions above are this function's.
            unsafe { x86::gemm_avx2(fmt, packed, n_in, panels, acts, bias, out, out_stride) };
            return;
        }
    }
    gemm_scalar_strided(fmt, packed, n_in, panels, acts, bias, out, out_stride);
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    /// AVX2, FMA and F16C: the tile kernel's floor.
    #[inline]
    pub fn have_avx2() -> bool {
        crate::simd::have_batched()
    }

    /// AVX-512 (F and VL), FMA and F16C: the wide tile kernel.
    #[inline]
    pub fn have_zmm() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            crate::cpu_has!("avx512f")
                && crate::cpu_has!("avx512vl")
                && crate::cpu_has!("fma")
                && crate::cpu_has!("f16c")
        })
    }

    /// Whether the single-row path may run.
    #[inline]
    pub fn have_gemv() -> bool {
        have_avx2()
    }

    /// One activation row: the single-row kernel.
    ///
    /// AVX-512 does not get its own: the 256-bit GEMV measured faster on this CPU
    /// (19.8 against 15.7 GMAC/s for f16 at 384 wide), because a 512-bit load and
    /// FMA is two 256-bit halves here and the eight-panel ymm version already
    /// streams the weights once.
    ///
    /// # Safety
    /// As [`super::gemm`]: the caller checks the lengths and the feature sets.
    pub unsafe fn gemv(
        fmt: Format,
        packed: &[u8],
        n_in: usize,
        panels: usize,
        act: &QuantAct,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        gemv_avx2(fmt, packed, n_in, panels, act, bias, out)
    }

    /// Sixteen weights of one panel step as two ymm vectors of f32.
    #[inline(always)]
    unsafe fn w16(fmt: Format, p: *const u8) -> (__m256, __m256) {
        match fmt {
            Format::F32 => (
                _mm256_loadu_ps(p as *const f32),
                _mm256_loadu_ps((p as *const f32).add(8)),
            ),
            Format::F16 => {
                let h = _mm256_loadu_si256(p as *const __m256i);
                (
                    _mm256_cvtph_ps(_mm256_castsi256_si128(h)),
                    _mm256_cvtph_ps(_mm256_extracti128_si256(h, 1)),
                )
            }
            _ => {
                let h = _mm256_loadu_si256(p as *const __m256i);
                let lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(h));
                let hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(h, 1));
                (
                    _mm256_castsi256_ps(_mm256_slli_epi32::<16>(lo)),
                    _mm256_castsi256_ps(_mm256_slli_epi32::<16>(hi)),
                )
            }
        }
    }

    /// Six accumulator pairs: one `k` step over the six rows of an AVX2 tile.
    macro_rules! step6 {
        ($acc:ident, $xp:ident, $k:ident, $w0:ident, $w1:ident) => {
            step6!(@row $acc, $xp, $k, $w0, $w1, 0);
            step6!(@row $acc, $xp, $k, $w0, $w1, 1);
            step6!(@row $acc, $xp, $k, $w0, $w1, 2);
            step6!(@row $acc, $xp, $k, $w0, $w1, 3);
            step6!(@row $acc, $xp, $k, $w0, $w1, 4);
            step6!(@row $acc, $xp, $k, $w0, $w1, 5);
        };
        (@row $acc:ident, $xp:ident, $k:ident, $w0:ident, $w1:ident, $r:literal) => {{
            let x = _mm256_broadcast_ss(&*$xp[$r].add($k));
            $acc[$r][0] = _mm256_fmadd_ps(x, $w0, $acc[$r][0]);
            $acc[$r][1] = _mm256_fmadd_ps(x, $w1, $acc[$r][1]);
        }};
    }

    /// One panel, six activation rows at a time (AVX2).
    ///
    /// The tail of a block duplicates the last row's pointer; its lanes are computed
    /// and thrown away, which costs a few multiplies and keeps the stores in bounds.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_avx2(
        fmt: Format,
        packed: &[u8],
        n_in: usize,
        panels: usize,
        acts: &[QuantAct],
        bias: Option<&[f32]>,
        out: &mut [f32],
        out_stride: usize,
    ) {
        let eb = elem_bytes(fmt);
        let pb = panel_bytes(fmt, n_in);
        let nb = acts.len();
        let bs = row_block(n_in);
        let stride = out_stride;
        let pf = crate::simd::prefetch_distance();
        let mut b0 = 0;
        while b0 < nb {
            let bn = bs.min(nb - b0);
            for p in 0..panels {
                let w = packed.as_ptr().add(p * pb);
                let op = out.as_mut_ptr().add(b0 * stride + p * PANEL);
                let (ib0, ib1) = match bias {
                    Some(v) => {
                        let q = v.as_ptr().add(p * PANEL);
                        (_mm256_loadu_ps(q), _mm256_loadu_ps(q.add(8)))
                    }
                    None => (_mm256_setzero_ps(), _mm256_setzero_ps()),
                };
                let mut t = 0;
                while t < bn {
                    let tn = (bn - t).min(6);
                    let mut xp = [std::ptr::null(); 6];
                    for (r, x) in xp.iter_mut().enumerate() {
                        *x = acts[b0 + t + r.min(tn - 1)].xf.as_ptr();
                    }
                    let mut acc = [[ib0, ib1]; 6];
                    for k in 0..n_in {
                        crate::simd::prefetch_ahead(w.add(k * PANEL * eb), pf, 1);
                        let (w0, w1) = w16(fmt, w.add(k * PANEL * eb));
                        step6!(acc, xp, k, w0, w1);
                    }
                    for r in 0..tn {
                        let o = op.add((t + r) * stride);
                        _mm256_storeu_ps(o, acc[r][0]);
                        _mm256_storeu_ps(o.add(8), acc[r][1]);
                    }
                    t += tn;
                }
            }
            b0 += bn;
        }
    }

    /// Twelve accumulator pairs, two panels in flight: one `k` step.
    macro_rules! step6x2 {
        ($a:ident, $b:ident, $xp:ident, $k:ident, $w0:ident, $w1:ident, $v0:ident, $v1:ident) => {
            step6x2!(@row $a, $b, $xp, $k, $w0, $w1, $v0, $v1, 0);
            step6x2!(@row $a, $b, $xp, $k, $w0, $w1, $v0, $v1, 1);
            step6x2!(@row $a, $b, $xp, $k, $w0, $w1, $v0, $v1, 2);
            step6x2!(@row $a, $b, $xp, $k, $w0, $w1, $v0, $v1, 3);
            step6x2!(@row $a, $b, $xp, $k, $w0, $w1, $v0, $v1, 4);
            step6x2!(@row $a, $b, $xp, $k, $w0, $w1, $v0, $v1, 5);
        };
        (@row $a:ident, $b:ident, $xp:ident, $k:ident, $w0:ident, $w1:ident, $v0:ident, $v1:ident, $r:literal) => {{
            let x = _mm256_broadcast_ss(&*$xp[$r].add($k));
            $a[$r][0] = _mm256_fmadd_ps(x, $w0, $a[$r][0]);
            $a[$r][1] = _mm256_fmadd_ps(x, $w1, $a[$r][1]);
            $b[$r][0] = _mm256_fmadd_ps(x, $v0, $b[$r][0]);
            $b[$r][1] = _mm256_fmadd_ps(x, $v1, $b[$r][1]);
        }};
    }

    /// Two panels, six activation rows at a time, on 256-bit vectors (AVX-512's
    /// thirty-two registers).
    ///
    /// This CPU double-pumps a 512-bit FMA into two 256-bit halves and issues one of
    /// those per two cycles, so a zmm tile computes exactly what two ymm tiles do in
    /// the same time — measured, a 12×32 zmm tile reached 13.5 GMAC/s against this
    /// one's 27. What the AVX-512 tier does buy is registers: two panels in flight
    /// (24 accumulators) amortise each activation broadcast over two panels, and the
    /// panel loads stay 256 bits wide.
    #[target_feature(enable = "avx512vl,avx2,fma,f16c")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_avx512(
        fmt: Format,
        packed: &[u8],
        n_in: usize,
        panels: usize,
        acts: &[QuantAct],
        bias: Option<&[f32]>,
        out: &mut [f32],
        out_stride: usize,
    ) {
        let eb = elem_bytes(fmt);
        let pb = panel_bytes(fmt, n_in);
        let nb = acts.len();
        let bs = row_block(n_in);
        let stride = out_stride;
        let pf = crate::simd::prefetch_distance();
        let mut b0 = 0;
        while b0 < nb {
            let bn = bs.min(nb - b0);
            let mut p = 0;
            while p < panels {
                let two = p + 1 < panels;
                let wa = packed.as_ptr().add(p * pb);
                let wb = if two { wa.add(pb) } else { wa };
                let op = out.as_mut_ptr().add(b0 * stride + p * PANEL);
                let (ia0, ia1, ib0, ib1) = match bias {
                    Some(v) => {
                        let q = v.as_ptr().add(p * PANEL);
                        let q2 = if two { q.add(PANEL) } else { q };
                        (
                            _mm256_loadu_ps(q),
                            _mm256_loadu_ps(q.add(8)),
                            _mm256_loadu_ps(q2),
                            _mm256_loadu_ps(q2.add(8)),
                        )
                    }
                    None => (
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                    ),
                };
                let mut t = 0;
                while t < bn {
                    let tn = (bn - t).min(6);
                    let mut xp = [std::ptr::null(); 6];
                    for (r, x) in xp.iter_mut().enumerate() {
                        *x = acts[b0 + t + r.min(tn - 1)].xf.as_ptr();
                    }
                    let mut acc = [[ia0, ia1]; 6];
                    let mut acc2 = [[ib0, ib1]; 6];
                    for k in 0..n_in {
                        let at = k * PANEL * eb;
                        crate::simd::prefetch_ahead(wa.add(at), pf, 1);
                        if two {
                            crate::simd::prefetch_ahead(wb.add(at), pf, 1);
                        }
                        let (a0, a1) = w16(fmt, wa.add(at));
                        let (v0, v1) = w16(fmt, wb.add(at));
                        step6x2!(acc, acc2, xp, k, a0, a1, v0, v1);
                    }
                    for r in 0..tn {
                        let o = op.add((t + r) * stride);
                        _mm256_storeu_ps(o, acc[r][0]);
                        _mm256_storeu_ps(o.add(8), acc[r][1]);
                        if two {
                            _mm256_storeu_ps(o.add(PANEL), acc2[r][0]);
                            _mm256_storeu_ps(o.add(PANEL + 8), acc2[r][1]);
                        }
                    }
                    t += tn;
                }
                p += if two { 2 } else { 1 };
            }
            b0 += bn;
        }
    }

    /// Four panels' accumulator pairs: one `k` step of the four-panel AVX2 GEMV.
    macro_rules! gstep4 {
        ($acc:ident, $w:ident, $x:ident, $fmt:ident, $eb:ident, $k:ident) => {
            gstep4!(@p $acc, $w, $x, $fmt, $eb, $k, 0);
            gstep4!(@p $acc, $w, $x, $fmt, $eb, $k, 1);
            gstep4!(@p $acc, $w, $x, $fmt, $eb, $k, 2);
            gstep4!(@p $acc, $w, $x, $fmt, $eb, $k, 3);
        };
        (@p $acc:ident, $w:ident, $x:ident, $fmt:ident, $eb:ident, $k:ident, $i:literal) => {{
            let (w0, w1) = w16($fmt, $w[$i].add($k * PANEL * $eb));
            $acc[$i][0] = _mm256_fmadd_ps($x, w0, $acc[$i][0]);
            $acc[$i][1] = _mm256_fmadd_ps($x, w1, $acc[$i][1]);
        }};
    }

    /// One activation row against four panels at a time (AVX2).
    ///
    /// The chain is the GEMM's: per output lane, `k` ascending, one FMA per term.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn gemv_avx2(
        fmt: Format,
        packed: &[u8],
        n_in: usize,
        panels: usize,
        act: &QuantAct,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let eb = elem_bytes(fmt);
        let pb = panel_bytes(fmt, n_in);
        let pf = crate::simd::prefetch_distance();
        let x = act.xf.as_ptr();
        let mut p = 0;
        while p < panels {
            let group = (panels - p).min(4);
            let mut w = [packed.as_ptr(); 4];
            let mut acc = [[_mm256_setzero_ps(); 2]; 4];
            for i in 0..group {
                w[i] = packed.as_ptr().add((p + i) * pb);
                if let Some(v) = bias {
                    let q = v.as_ptr().add((p + i) * PANEL);
                    acc[i] = [_mm256_loadu_ps(q), _mm256_loadu_ps(q.add(8))];
                }
            }
            // A short tail repeats its own pointer; only its own store happens.
            for i in group..4 {
                w[i] = w[group - 1];
            }
            for k in 0..n_in {
                let xv = _mm256_broadcast_ss(&*x.add(k));
                for &wp in w.iter() {
                    crate::simd::prefetch_ahead(wp.add(k * PANEL * eb), pf, 1);
                }
                gstep4!(acc, w, xv, fmt, eb, k);
            }
            for i in 0..group {
                let o = out.as_mut_ptr().add((p + i) * PANEL);
                _mm256_storeu_ps(o, acc[i][0]);
                _mm256_storeu_ps(o.add(8), acc[i][1]);
            }
            p += group;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Test-only: how often the F16 kernel ran, so a test can pin the dispatch on
    /// every NEON tier (the result is bit-identical either way).
    #[cfg(test)]
    pub(super) static F16_CALLS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    /// Sixteen F32 or BF16 weights of one panel step as four f32 vectors.
    ///
    /// BF16 shifts the half's bits into the high half of an f32; the conversion is exact.
    #[inline(always)]
    unsafe fn w16(fmt: Format, p: *const u8) -> [float32x4_t; 4] {
        match fmt {
            Format::F32 => [
                vld1q_f32(p as *const f32),
                vld1q_f32((p as *const f32).add(4)),
                vld1q_f32((p as *const f32).add(8)),
                vld1q_f32((p as *const f32).add(12)),
            ],
            _ => [
                bf16x4(p),
                bf16x4(p.add(8)),
                bf16x4(p.add(16)),
                bf16x4(p.add(24)),
            ],
        }
    }

    /// Sixteen F16 weights of one panel step as four f32 vectors, widened by `fcvtl`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn w16_f16(p: *const u8) -> [float32x4_t; 4] {
        [
            f16x4(p),
            f16x4(p.add(8)),
            f16x4(p.add(16)),
            f16x4(p.add(24)),
        ]
    }

    /// Four f16 halves widened to f32 (`fcvtl`), exactly.
    ///
    /// `fcvtl` is base Advanced SIMD, not part of the `fp16` arithmetic extension.
    ///
    /// # Safety
    /// The CPU must have NEON and `p` must be readable for eight bytes.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn f16x4(p: *const u8) -> float32x4_t {
        let h = vld1_u16(p as *const u16);
        let mut r: float32x4_t;
        // The `float16x4_t` intrinsics are unstable at this workspace's MSRV, so the
        // instruction is written out; the register class is all it needs.
        std::arch::asm!(
            "fcvtl {r:v}.4s, {a:v}.4h",
            r = lateout(vreg) r,
            a = in(vreg) h,
            options(pure, nomem, nostack),
        );
        r
    }

    /// Four bf16 halves widened to f32 by the high-half shift, exactly.
    ///
    /// # Safety
    /// The CPU must have NEON and `p` must be readable for eight bytes.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn bf16x4(p: *const u8) -> float32x4_t {
        let h = vld1_u16(p as *const u16);
        vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(h)))
    }

    /// One `k` step of one panel against four activation rows.
    #[inline(always)]
    unsafe fn step4(
        acc: &mut [[float32x4_t; 4]; 4],
        xp: &[*const f32; 4],
        k: usize,
        w: &[float32x4_t; 4],
    ) {
        let x0 = vdupq_n_f32(*xp[0].add(k));
        let x1 = vdupq_n_f32(*xp[1].add(k));
        let x2 = vdupq_n_f32(*xp[2].add(k));
        let x3 = vdupq_n_f32(*xp[3].add(k));
        acc[0][0] = vfmaq_f32(acc[0][0], w[0], x0);
        acc[0][1] = vfmaq_f32(acc[0][1], w[1], x0);
        acc[0][2] = vfmaq_f32(acc[0][2], w[2], x0);
        acc[0][3] = vfmaq_f32(acc[0][3], w[3], x0);
        acc[1][0] = vfmaq_f32(acc[1][0], w[0], x1);
        acc[1][1] = vfmaq_f32(acc[1][1], w[1], x1);
        acc[1][2] = vfmaq_f32(acc[1][2], w[2], x1);
        acc[1][3] = vfmaq_f32(acc[1][3], w[3], x1);
        acc[2][0] = vfmaq_f32(acc[2][0], w[0], x2);
        acc[2][1] = vfmaq_f32(acc[2][1], w[1], x2);
        acc[2][2] = vfmaq_f32(acc[2][2], w[2], x2);
        acc[2][3] = vfmaq_f32(acc[2][3], w[3], x2);
        acc[3][0] = vfmaq_f32(acc[3][0], w[0], x3);
        acc[3][1] = vfmaq_f32(acc[3][1], w[1], x3);
        acc[3][2] = vfmaq_f32(acc[3][2], w[2], x3);
        acc[3][3] = vfmaq_f32(acc[3][3], w[3], x3);
    }

    /// The GEMM loop, shared by the F32/BF16 and F16 entry points: one panel, four
    /// activation rows at a time, sixteen accumulators, one weight load feeding all
    /// four rows. The tail duplicates the last row's pointer, as the AVX2 kernel does.
    macro_rules! gemm_body {
        ($fmt:ident, $packed:ident, $n_in:ident, $panels:ident, $acts:ident, $bias:ident, $out:ident, $stride:ident, $loader:expr) => {{
            let eb = elem_bytes($fmt);
            let pb = panel_bytes($fmt, $n_in);
            let nb = $acts.len();
            let bs = row_block($n_in);
            let stride = $stride;
            let mut b0 = 0;
            while b0 < nb {
                let bn = bs.min(nb - b0);
                for p in 0..$panels {
                    let w = $packed.as_ptr().add(p * pb);
                    let op = $out.as_mut_ptr().add(b0 * stride + p * PANEL);
                    let ib = match $bias {
                        Some(v) => {
                            let q = v.as_ptr().add(p * PANEL);
                            [
                                vld1q_f32(q),
                                vld1q_f32(q.add(4)),
                                vld1q_f32(q.add(8)),
                                vld1q_f32(q.add(12)),
                            ]
                        }
                        None => [vdupq_n_f32(0.0); 4],
                    };
                    let mut t = 0;
                    while t < bn {
                        let tn = (bn - t).min(4);
                        let mut xp = [std::ptr::null(); 4];
                        for (r, x) in xp.iter_mut().enumerate() {
                            *x = $acts[b0 + t + r.min(tn - 1)].xf.as_ptr();
                        }
                        let mut acc = [ib; 4];
                        for k in 0..$n_in {
                            let wv = $loader(w.add(k * PANEL * eb));
                            step4(&mut acc, &xp, k, &wv);
                        }
                        for r in 0..tn {
                            let o = op.add((t + r) * stride);
                            vst1q_f32(o, acc[r][0]);
                            vst1q_f32(o.add(4), acc[r][1]);
                            vst1q_f32(o.add(8), acc[r][2]);
                            vst1q_f32(o.add(12), acc[r][3]);
                        }
                        t += tn;
                    }
                }
                b0 += bn;
            }
        }};
    }

    /// One panel, four activation rows at a time (NEON), F32 and BF16.
    ///
    /// # Safety
    /// As [`super::gemm`]: the caller checks the lengths and the feature sets.
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_neon(
        fmt: Format,
        packed: &[u8],
        n_in: usize,
        panels: usize,
        acts: &[QuantAct],
        bias: Option<&[f32]>,
        out: &mut [f32],
        out_stride: usize,
    ) {
        gemm_body!(
            fmt,
            packed,
            n_in,
            panels,
            acts,
            bias,
            out,
            out_stride,
            |p| w16(fmt, p)
        );
    }

    /// The same loop for F16 weights, widened by `fcvtl`.
    ///
    /// # Safety
    /// As [`super::gemm`].
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_f16(
        fmt: Format,
        packed: &[u8],
        n_in: usize,
        panels: usize,
        acts: &[QuantAct],
        bias: Option<&[f32]>,
        out: &mut [f32],
        out_stride: usize,
    ) {
        #[cfg(test)]
        F16_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        gemm_body!(
            fmt,
            packed,
            n_in,
            panels,
            acts,
            bias,
            out,
            out_stride,
            |p| w16_f16(p)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small xorshift: the tests need reproducible weights, nothing more.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// A value in `[-3, 3)` with a few very small ones, so the f16 and bf16
        /// roundings are exercised rather than hidden by the exponent range.
        fn v(&mut self) -> f32 {
            let u = self.next();
            let mag = ((u >> 40) as f32 / (1u64 << 24) as f32) * 3.0;
            let sign = if u & 1 == 0 { 1.0 } else { -1.0 };
            sign * mag
        }
    }

    fn enc(fmt: Format, x: f32) -> [u8; 4] {
        let mut b = [0u8; 4];
        match fmt {
            Format::F32 => b.copy_from_slice(&x.to_le_bytes()),
            Format::F16 => b[..2].copy_from_slice(&crate::quant::f32_to_f16(x).to_le_bytes()),
            _ => b[..2].copy_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes()),
        }
        b
    }

    /// A random `[n_out][n_in]` matrix in the file's row layout.
    fn matrix(fmt: Format, n_out: usize, n_in: usize, seed: u64) -> Vec<u8> {
        let eb = elem_bytes(fmt);
        let mut rng = Rng(seed | 1);
        let mut w = vec![0u8; n_out * n_in * eb];
        for r in 0..n_out {
            for k in 0..n_in {
                let b = enc(fmt, rng.v());
                let at = (r * n_in + k) * eb;
                w[at..at + eb].copy_from_slice(&b[..eb]);
            }
        }
        w
    }

    /// The matrix's decoded values, through the file's own reference decode.
    fn decode(fmt: Format, w: &[u8], n_out: usize, n_in: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; n_out * n_in];
        let rb = n_in * elem_bytes(fmt);
        for r in 0..n_out {
            crate::quant::dequant_row(
                fmt,
                &w[r * rb..(r + 1) * rb],
                &mut y[r * n_in..(r + 1) * n_in],
            );
        }
        y
    }

    /// The panels of the whole matrix, built the way `peregrine-core`'s store is.
    fn pack(fmt: Format, w: &[u8], n_out: usize, n_in: usize) -> Vec<u8> {
        let rb = n_in * elem_bytes(fmt);
        let mut p = vec![0u8; packed_len(fmt, n_out, n_in)];
        let pb = panel_bytes(fmt, n_in);
        for panel in 0..n_out.div_ceil(PANEL) {
            let lo = panel * PANEL;
            let hi = (lo + PANEL).min(n_out);
            let rows: Vec<&[u8]> = (lo..hi).map(|r| &w[r * rb..(r + 1) * rb]).collect();
            pack_panel(fmt, &rows, n_in, &mut p[panel * pb..(panel + 1) * pb]);
        }
        p
    }

    /// One activation row, f32 only: the packed path reads `xf` and nothing else.
    /// (Built by hand rather than through `QuantAct::fill`, which wants a whole
    /// number of 32-blocks; the kernel must take any `n_in`.)
    fn act(x: &[f32]) -> QuantAct {
        QuantAct {
            n: x.len(),
            xf: x.to_vec(),
            keep_f32: true,
            ..Default::default()
        }
    }

    fn acts(rows: usize, n: usize, seed: u64) -> Vec<QuantAct> {
        let mut rng = Rng(seed | 1);
        (0..rows)
            .map(|_| {
                let x: Vec<f32> = (0..n).map(|_| rng.v()).collect();
                act(&x)
            })
            .collect()
    }

    /// The specification, written out independently of the kernel code.
    fn reference(
        vals: &[f32],
        n_out: usize,
        n_in: usize,
        xs: &[QuantAct],
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; n_out * xs.len()];
        for (b, a) in xs.iter().enumerate() {
            for r in 0..n_out {
                let mut acc = bias.map_or(0.0f32, |v| v[r]);
                for k in 0..n_in {
                    acc = a.xf[k].mul_add(vals[r * n_in + k], acc);
                }
                y[b * n_out + r] = acc;
            }
        }
        y
    }

    /// Bits, or both NaN: a fused multiply-add's NaN *payload* is not the same on
    /// every implementation (see the module docs), its value class is.
    fn same(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    fn check_all(fmt: Format, n_out: usize, n_in: usize, xs: &[QuantAct], bias: Option<&[f32]>) {
        let w = matrix(fmt, n_out, n_in, (n_out * 31 + n_in * 7 + xs.len()) as u64);
        let vals = decode(fmt, &w, n_out, n_in);
        let packed = pack(fmt, &w, n_out, n_in);
        let want = reference(&vals, n_out, n_in, xs, bias);
        let panels = n_out.div_ceil(PANEL);
        let mut got = vec![0.0f32; panels * PANEL * xs.len()];
        gemm(fmt, &packed, n_in, panels, xs, bias, &mut got);
        let mut got_s = vec![0.0f32; panels * PANEL * xs.len()];
        gemm_scalar(fmt, &packed, n_in, panels, xs, bias, &mut got_s);
        for b in 0..xs.len() {
            for r in 0..n_out {
                let g = got[b * panels * PANEL + r];
                assert!(
                    same(g, want[b * n_out + r]),
                    "{fmt:?} {n_out}x{n_in} b={b} r={r}: {g} vs {}",
                    want[b * n_out + r]
                );
                let s = got_s[b * panels * PANEL + r];
                assert!(
                    same(s, g),
                    "{fmt:?} scalar vs kernel {n_out}x{n_in} b={b} r={r}"
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        if crate::cpu::have_neon() {
            // The NEON kernel against the scalar reference — the bit-identity the
            // aarch64 tiers inherit from the x86 ones.
            let mut an = vec![0.0f32; panels * PANEL * xs.len()];
            if fmt == Format::F16 {
                unsafe {
                    neon::gemm_f16(
                        fmt,
                        &packed,
                        n_in,
                        panels,
                        xs,
                        bias,
                        &mut an,
                        panels * PANEL,
                    )
                };
            } else {
                unsafe {
                    neon::gemm_neon(
                        fmt,
                        &packed,
                        n_in,
                        panels,
                        xs,
                        bias,
                        &mut an,
                        panels * PANEL,
                    )
                };
            }
            for (i, (&x, &y)) in an.iter().zip(&got_s).enumerate() {
                assert!(same(x, y), "{fmt:?} NEON vs scalar {n_out}x{n_in} at {i}");
            }
        }
        #[cfg(target_arch = "x86_64")]
        if x86::have_avx2() {
            // The AVX2 kernel and, when the CPU has it, the AVX-512 kernel, both
            // against the scalar reference — the bit-identity across tiers.
            let mut a2 = vec![0.0f32; panels * PANEL * xs.len()];
            unsafe {
                x86::gemm_avx2(
                    fmt,
                    &packed,
                    n_in,
                    panels,
                    xs,
                    bias,
                    &mut a2,
                    panels * PANEL,
                )
            };
            for (i, (&x, &y)) in a2.iter().zip(&got_s).enumerate() {
                assert!(same(x, y), "{fmt:?} AVX2 vs scalar {n_out}x{n_in} at {i}");
            }
            if x86::have_zmm() {
                let mut a5 = vec![0.0f32; panels * PANEL * xs.len()];
                unsafe {
                    x86::gemm_avx512(
                        fmt,
                        &packed,
                        n_in,
                        panels,
                        xs,
                        bias,
                        &mut a5,
                        panels * PANEL,
                    )
                };
                for (i, (&x, &y)) in a5.iter().zip(&got_s).enumerate() {
                    assert!(
                        same(x, y),
                        "{fmt:?} AVX-512 vs scalar {n_out}x{n_in} at {i}"
                    );
                }
            }
        }
    }

    const FORMATS: [Format; 3] = [Format::F32, Format::F16, Format::BF16];

    /// Every size a tail can make, every batch a block can end on.
    #[test]
    fn the_vector_kernels_are_the_scalar_chain() {
        let sizes = [
            (1, 1),
            (3, 7),
            (15, 16),
            (16, 17),
            (17, 31),
            (31, 33),
            (33, 64),
            (40, 100),
            (48, 384),
            (16, 1024),
            (5, 1023),
            (17, 383),
        ];
        for fmt in FORMATS {
            for &(n_out, n_in) in &sizes {
                for nb in [1, 2, 5, 6, 7, 12, 13, 40] {
                    let xs = acts(nb, n_in, (n_out * 1000 + n_in + nb) as u64);
                    check_all(fmt, n_out, n_in, &xs, None);
                    // One value per output row of the *run*, so the padding lanes of
                    // the last panel have one too.
                    let run = n_out.div_ceil(PANEL) * PANEL;
                    let bias: Vec<f32> = (0..run).map(|r| r as f32 * 0.25 - 1.0).collect();
                    check_all(fmt, n_out, n_in, &xs, Some(&bias));
                }
            }
        }
    }

    /// The strided store writes the same values as the row-major one, and leaves
    /// every other element of the caller's buffer alone.
    #[test]
    fn the_strided_store_is_the_row_major_store() {
        let sizes = [(16, 384), (24, 384), (40, 384), (16, 96), (5, 37), (1, 1)];
        for fmt in FORMATS {
            for &(n_out, n_in) in &sizes {
                for nb in [1, 2, 5, 8] {
                    let xs = acts(nb, n_in, (n_out * 13 + n_in + nb) as u64);
                    let w = matrix(fmt, n_out, n_in, (n_out * 7 + n_in * 3 + nb) as u64);
                    let packed = pack(fmt, &w, n_out, n_in);
                    let panels = n_out.div_ceil(PANEL);
                    let run = panels * PANEL;
                    let mut want = vec![0.0f32; run * nb];
                    gemm(fmt, &packed, n_in, panels, &xs, None, &mut want);
                    // A wider, row-major buffer: the kernel writes the run at the
                    // start of each activation row.
                    let stride = run + 9;
                    let mut got = vec![-7.0f32; stride * nb];
                    gemm_strided(fmt, &packed, n_in, panels, &xs, None, &mut got, stride);
                    for b in 0..nb {
                        for r in 0..run {
                            assert!(
                                same(got[b * stride + r], want[b * run + r]),
                                "{fmt:?} {n_out}x{n_in} b={b} r={r}"
                            );
                        }
                        for r in run..stride {
                            assert_eq!(got[b * stride + r], -7.0, "wrote past the run");
                        }
                    }
                    // The scalar twin takes the same stride.
                    let mut gs = vec![0.0f32; stride * nb];
                    gemm_scalar_strided(fmt, &packed, n_in, panels, &xs, None, &mut gs, stride);
                    for b in 0..nb {
                        for r in 0..run {
                            assert!(same(gs[b * stride + r], want[b * run + r]));
                        }
                    }
                }
            }
        }
    }

    /// `fcvtl` is base Advanced SIMD, so F16 must take the packed NEON kernel on
    /// plain `neon` too, not fall through to the scalar chain.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn f16_takes_the_neon_kernel_on_every_neon_tier() {
        if !crate::cpu::have_neon() {
            return;
        }
        let (n_out, n_in) = (16, 32);
        let w = matrix(Format::F16, n_out, n_in, 9);
        let packed = pack(Format::F16, &w, n_out, n_in);
        let xs = acts(2, n_in, 4);
        let panels = n_out.div_ceil(PANEL);
        let mut out = vec![0.0f32; panels * PANEL * xs.len()];
        let before = neon::F16_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        gemm(Format::F16, &packed, n_in, panels, &xs, None, &mut out);
        assert!(
            neon::F16_CALLS.load(std::sync::atomic::Ordering::Relaxed) > before,
            "F16 did not reach the NEON kernel on this tier"
        );
    }

    /// A row computed alone, in the middle of a batch, or last must be the same bits.
    #[test]
    fn a_row_does_not_depend_on_the_batch() {
        for fmt in FORMATS {
            let (n_out, n_in) = (37, 260);
            let w = matrix(fmt, n_out, n_in, 77);
            let vals = decode(fmt, &w, n_out, n_in);
            let packed = pack(fmt, &w, n_out, n_in);
            let panels = n_out.div_ceil(PANEL);
            let all = acts(23, n_in, 5);
            let mut batch = vec![0.0f32; panels * PANEL * all.len()];
            gemm(fmt, &packed, n_in, panels, &all, None, &mut batch);
            let alone = reference(&vals, n_out, n_in, &all, None);
            for b in 0..all.len() {
                // Alone, through both paths.
                let mut solo = vec![0.0f32; panels * PANEL];
                let one = std::slice::from_ref(&all[b]);
                gemm(fmt, &packed, n_in, panels, one, None, &mut solo);
                for r in 0..n_out {
                    let batched = batch[b * panels * PANEL + r];
                    assert!(
                        same(solo[r], batched),
                        "{fmt:?} row {b}/{r}: alone vs batch"
                    );
                    assert!(same(batched, alone[b * n_out + r]), "{fmt:?} row {b}/{r}");
                }
            }
            // A batch of two around it, in both positions.
            for b in [0usize, all.len() - 1] {
                let pair = [all[b].clone(), all[all.len() - 1 - b].clone()];
                let mut got = vec![0.0f32; panels * PANEL * 2];
                gemm(fmt, &packed, n_in, panels, &pair, None, &mut got);
                for r in 0..n_out {
                    assert!(
                        same(got[r], batch[b * panels * PANEL + r]),
                        "{fmt:?} row {b}/{r}: pair position 0"
                    );
                }
            }
        }
    }

    /// The single-row kernel is the batched one: the chain is the same.
    #[test]
    fn the_single_row_path_equals_the_batched_path() {
        for fmt in FORMATS {
            let (n_out, n_in) = (33, 512);
            let w = matrix(fmt, n_out, n_in, 9);
            let packed = pack(fmt, &w, n_out, n_in);
            let panels = n_out.div_ceil(PANEL);
            let one = acts(1, n_in, 3);
            let mut row = vec![0.0f32; panels * PANEL];
            gemm(fmt, &packed, n_in, panels, &one, None, &mut row);
            let mut four = vec![0.0f32; panels * PANEL * 4];
            let all = acts(4, n_in, 3);
            gemm(fmt, &packed, n_in, panels, &all, None, &mut four);
            for r in 0..n_out {
                assert!(same(row[r], four[r]), "{fmt:?} row {r}");
            }
            #[cfg(target_arch = "x86_64")]
            if x86::have_avx2() {
                let mut g = vec![0.0f32; panels * PANEL];
                unsafe { x86::gemv_avx2(fmt, &packed, n_in, panels, &one[0], None, &mut g) };
                for r in 0..n_out {
                    assert!(same(g[r], row[r]), "{fmt:?} AVX2 GEMV row {r}");
                }
            }
        }
    }

    /// The row block is a cache decision: the same output whatever it is.
    #[test]
    fn the_row_block_does_not_change_a_result() {
        assert_eq!(row_block(0), row_block(0));
        assert!(row_block(1) >= 1);
        for n_in in [1usize, 384, 1024, 100_000] {
            assert!((1..=128).contains(&row_block(n_in)), "row_block({n_in})");
        }
        for fmt in FORMATS {
            let (n_out, n_in) = (17, 300);
            let w = matrix(fmt, n_out, n_in, 12);
            let packed = pack(fmt, &w, n_out, n_in);
            let vals = decode(fmt, &w, n_out, n_in);
            let panels = n_out.div_ceil(PANEL);
            let xs = acts(20, n_in, 8);
            // Compare the reference with a kernel run that is inside one block and
            // with runs that are split at different boundaries: the same numbers.
            let want = reference(&vals, n_out, n_in, &xs, None);
            let mut got = vec![0.0f32; panels * PANEL * xs.len()];
            gemm(fmt, &packed, n_in, panels, &xs, None, &mut got);
            for split in [1usize, 3, 7, 19] {
                let mut a = vec![0.0f32; panels * PANEL * xs.len()];
                let mut lo = 0;
                while lo < xs.len() {
                    let hi = (lo + split).min(xs.len());
                    gemm(
                        fmt,
                        &packed,
                        n_in,
                        panels,
                        &xs[lo..hi],
                        None,
                        &mut a[lo * panels * PANEL..],
                    );
                    lo = hi;
                }
                for b in 0..xs.len() {
                    for r in 0..n_out {
                        let g = a[b * panels * PANEL + r];
                        assert!(
                            same(g, want[b * n_out + r]),
                            "{fmt:?} split {split} {b}/{r}"
                        );
                        assert!(
                            same(g, got[b * panels * PANEL + r]),
                            "{fmt:?} split {split} {r}"
                        );
                    }
                }
            }
        }
    }

    /// Infinities, a NaN and denormals: the chain is IEEE, so every kernel agrees
    /// (NaN payloads aside, which the module docs explain).
    #[test]
    fn the_special_values_take_the_same_chain() {
        for fmt in FORMATS {
            let (n_out, n_in) = (18, 40);
            let mut rng = Rng(0xfeed_2026);
            let mut w = matrix(fmt, n_out, n_in, 4);
            let rb = n_in * elem_bytes(fmt);
            // Sprinkle specials into both operands.
            for (i, v) in [
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                f32::MIN_POSITIVE / 4.0,
                -1e-38,
                1e30,
            ]
            .iter()
            .enumerate()
            {
                let at = ((i * 7 + 3) % n_out) * rb + ((i * 13) % n_in) * elem_bytes(fmt);
                let b = enc(fmt, *v);
                w[at..at + elem_bytes(fmt)].copy_from_slice(&b[..elem_bytes(fmt)]);
            }
            let vals = decode(fmt, &w, n_out, n_in);
            let packed = pack(fmt, &w, n_out, n_in);
            let panels = n_out.div_ceil(PANEL);
            let mut xs = acts(5, n_in, 6);
            for a in xs.iter_mut() {
                a.xf[3] = f32::INFINITY;
                a.xf[11] = 0.0;
                a.xf[17] = 1e-42;
                a.xf[23] = f32::NAN;
                a.xf[29] = -0.0;
                let _ = &mut rng;
            }
            let want = reference(&vals, n_out, n_in, &xs, None);
            let mut got = vec![0.0f32; panels * PANEL * xs.len()];
            gemm(fmt, &packed, n_in, panels, &xs, None, &mut got);
            for b in 0..xs.len() {
                for r in 0..n_out {
                    let g = got[b * panels * PANEL + r];
                    assert!(same(g, want[b * n_out + r]), "{fmt:?} {b}/{r}: {g}");
                }
            }
        }
    }

    /// One packed panel is exactly the rows it was packed from, step by step.
    #[test]
    fn packing_moves_the_bytes_and_nothing_else() {
        for fmt in FORMATS {
            let (n_out, n_in) = (17, 33);
            let w = matrix(fmt, n_out, n_in, 21);
            let vals = decode(fmt, &w, n_out, n_in);
            let packed = pack(fmt, &w, n_out, n_in);
            assert_eq!(packed.len(), packed_len(fmt, n_out, n_in));
            let pb = panel_bytes(fmt, n_in);
            for panel in 0..n_out.div_ceil(PANEL) {
                for k in 0..n_in {
                    for lane in 0..PANEL {
                        let r = panel * PANEL + lane;
                        if r >= n_out {
                            // The padding is zero.
                            let v = w_at(
                                fmt,
                                unsafe { packed.as_ptr().add(panel * pb) },
                                k * PANEL + lane,
                            );
                            assert_eq!(v.to_bits(), 0.0f32.to_bits(), "{fmt:?} padding");
                            continue;
                        }
                        let v = w_at(
                            fmt,
                            unsafe { packed.as_ptr().add(panel * pb) },
                            k * PANEL + lane,
                        );
                        assert_eq!(
                            v.to_bits(),
                            vals[r * n_in + k].to_bits(),
                            "{fmt:?} panel {panel} k {k} lane {lane}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "keep_f32 off on QuantAct")]
    fn the_packed_kernel_insists_on_the_f32_activation() {
        let w = matrix(Format::F16, 16, 32, 1);
        let packed = pack(Format::F16, &w, 16, 32);
        let acts = vec![QuantAct::with_capacity(32)];
        let mut out = vec![0.0f32; 16];
        gemm(Format::F16, &packed, 32, 1, &acts, None, &mut out);
    }

    #[test]
    fn the_cache_size_parser_reads_the_kernel_and_the_sysfs() {
        assert_eq!(parse_cache_size("1024K"), Some(1 << 20));
        assert_eq!(parse_cache_size("2M"), Some(2 << 20));
        assert_eq!(parse_cache_size("65536"), Some(65536));
        assert_eq!(parse_cache_size("nonsense"), None);
    }
}
