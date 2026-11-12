//! Quantised weight kernels: fused dequantise-and-dot.
//!
//! The central rule (spec P1) is that a dequantised weight tensor is never
//! materialised. Decoding a Q4_K row into f32 and then running a float dot product
//! would move eight bytes for every four and a half the format actually stores,
//! which puts the engine below the bandwidth roof and makes every number it reports
//! a statement about our own buffer traffic rather than about the machine. So the
//! block decode happens inside the accumulation loop, in registers.
//!
//! # The integer identity this file is built on
//!
//! Activations are quantised to int8 in blocks of 32 (`QuantAct`), so the inner
//! product can be integer throughout. Every quantised format here stores its codes
//! *unsigned* and applies an offset at decode time — `−16` for Q5_0, `−32` for Q6_K,
//! an explicit `−dmin·m` for Q4_K. Reconstructing the signed weight just to multiply
//! it would throw away the one instruction that makes the integer path worth having
//! (`VPMADDUBSW` takes an unsigned operand and a signed one). Instead, for codes
//! `c_i` with offset `z` and activations `a_i`:
//!
//! ```text
//!   Σ (c_i − z)·a_i  =  Σ c_i·a_i  −  z·Σ a_i
//! ```
//!
//! exact in integers. `Σ a_i` is computed once per activation block when the
//! activation is quantised and reused by every output row, so the correction costs
//! O(1) per block rather than O(n). This is the paper's Theorem 1 generalised from
//! ternary weights to any offset format.
//!
//! # Why the scalar path is the specification
//!
//! Each kernel below mirrors llama.cpp's own reference `dequantize_row_*` exactly,
//! including the interleaved output orderings, which are the part everyone gets
//! wrong: a transposed nibble order still decodes to plausible-looking floats and
//! produces fluent, wrong text. `tests/oracle.rs` checks these against the real
//! weight file, and the AVX2 kernels in `simd.rs` are checked against these.
//!
//! # Float formats (F32, F16, BF16)
//!
//! Unlike the quantised formats, float weight matrices multiply against the full
//! unquantised `f32` activation row kept in `QuantAct::xf`. Accumulation order is
//! defined by [`dot_f32_ref`], matching the AVX2 SIMD kernel bit for bit.

use crate::format::{f16, Format};

/// Whether the fast prompt mode needs the per-256 activation layout of a row that
/// is `n` weights wide padded to a whole super-block (S5.4f).
///
/// Q5_0's file blocks are 32 weights and its matrices may stop inside a
/// super-block — Qwen 2.5 0.5B's are 896 wide, three and a half. The form's
/// kernel reads one scale per 256 weights, so its activation is built over four
/// super-blocks with the missing codes zero (and the form's missing codes zero
/// too, so the padded lanes contribute nothing). Only a tier that actually uses
/// the Q5_0 form pays this; with the option off or the format outside the tier's
/// table, the layout is exactly what it was.
#[inline]
fn pad_k_layout(n: usize) -> bool {
    !n.is_multiple_of(256)
        && crate::table::MatKernels::get().q8k_uses_form(Format::Q5_0)
        && (crate::options::get().fast_prompt || crate::kq8::padded_q5_form())
}

/// An activation vector quantised to int8 in blocks of 32.
///
/// Built once per matmul and shared across every output row — that sharing is what
/// makes the bias correction in the module docs O(1) rather than O(n).
#[derive(Clone)]
pub struct QuantAct {
    pub n: usize,
    /// Whether this row carries a *prompt* token for the opt-in fast prompt mode
    /// (`docs/design/s5.4-fast-prompt.md`, S5.4b). The engine sets it on the rows
    /// of a prompt chunk and leaves it false on generated (decode) rows and on
    /// every row of an embedding, rerank or decision pass. It changes nothing
    /// unless the process's `PEREGRINE_FAST_PROMPT` is on and the matrix has a
    /// resident 8-bit form (`kq8`): with the option off the field is always false.
    ///
    /// `fill` does not touch it, so the flag survives the re-quantisation of each
    /// layer's activation.
    pub prompt: bool,
    /// int8 codes, `n` of them.
    pub q: Vec<i8>,
    /// One scale per 32-block: `a_i ≈ q_i · d[i/32]`.
    pub d: Vec<f32>,
    /// `Σ q` over each 32-block — the `Σ a_i` of the bias identity, in code space.
    pub sums: Vec<i32>,
    /// `Σ q` over each 16-half. Q6_K changes scale every 16 weights, so it needs the
    /// finer partition; `sums` is derivable from this but kept separately because
    /// the 32-block kernels are the hot ones.
    pub sums16: Vec<i32>,
    /// `Σ q` over each run of four codes, eight per 32-block, in the lane order
    /// `VPMADDWD` produces. The batched kernels subtract an offset format's bias
    /// *inside the integer lanes* (`lane − offset·Σq`), which is exact; applying it in
    /// float after accumulation cancels two large, nearly equal numbers and lost three
    /// digits on small outputs (measured: −0.015595 against −0.015600).
    pub lane_sums: Vec<i32>,
    /// The same activation quantised with **one scale per 256** (llama.cpp's Q8_K),
    /// for the k-quant formats whose super-block is 256 weights. With one scale per
    /// super-block the eight sub-block dot products can be scaled and summed in
    /// integer lanes and converted to float once per 256 weights, instead of once per
    /// 32 — which is the difference between a kernel bound by vector-ALU latency and
    /// one bound by memory. Empty when `n` is not a multiple of 256.
    pub qk: Vec<i8>,
    /// One scale per 256-block: `a_i ≈ qk_i · dk[i/256]`.
    pub dk: Vec<f32>,
    /// `Σ qk` over each run of sixteen, sixteen per 256-block: the Q4_K mins and the
    /// Q6_K offset are applied against these, in integer lanes.
    pub bsums: Vec<i16>,
    // the ternary layout (Bonsai 2's PQ2_0 / PTQ1_0, group 128), filled instead of the others when set.
    /// Fill only the per-128 ternary layout below (a model whose matmuls are all ternary).
    pub ternary: bool,
    /// int8 codes, one scale per 16 (`td`), in natural order (PTQ1_0's trit planes are contiguous runs).
    pub tq: Vec<i8>,
    /// The same codes permuted to PQ2_0's byte layout: plane `s` of group `g` is `tp[tp_off(g) + 64s .. +32]` with
    /// `tp[tp_off(g) + 64s + k] = tq[128g + 4k + s]`, so bit-plane `s` of the 32 weight bytes lines up with 32
    /// contiguous activation bytes (no shuffles in the kernel), and groups `2p`, `2p+1` sit side by side per plane
    /// (one 512-bit operand holds both).
    pub tp: Vec<i8>,
    /// One scale per 16: `a_i ≈ tq_i · td[i/16]`. Lane `j` of the permuted kernel's `VPDPBUSD` sums exactly the
    /// 16 values of block `8g + j` (activations `16j..16j+16` of group `g`, over the four bit-planes), so a scale
    /// per 16 costs the kernel a vector load instead of a broadcast, and quantises 8× finer than one per 128.
    pub td: Vec<f32>,
    /// `Σ tq` over each 16-block, in the same lanes: subtracting it in the integer lanes turns `Σ code·a` into
    /// `Σ (code − 1)·a` exactly.
    pub tls: Vec<i32>,
    /// Tile mode (the row-interleaved PQ2_0 kernel, `tern::dot_tiles_pq2`): values sharing one activation scale
    /// (32, 64 or 128; 0 = off). The codes `tq` then carry that scale (`td`/`tls` stay per 16, consistent with it).
    pub tb_span: usize,
    /// The codes in broadcast order: per group g, quad q, plane s, the 4 bytes
    /// `[tq[128g+16q+s], tq[128g+16q+4+s], tq[128g+16q+8+s], tq[128g+16q+12+s]]`, which one `VPDPBUSD` lane meets
    /// against 4 code bytes of its row.
    pub tb: Vec<i8>,
    /// Scale per span, and `−Σ tq` per span (the code − 1 offset, as the accumulator's initial value).
    pub tbd: Vec<f32>,
    pub tbn: Vec<i32>,
    /// PTQ1_0 mode (a model whose ternary weights are all PTQ1_0): instead of the int8 layout, int16 codes with
    /// one scale per 128 in the PTQ1_0 kernel's order: per pair of groups p, 8 rounds r, `[g0 values 16r..16r+16 |
    /// g1 values 16r..16r+16]` (`t1a`), the offset init per pair (`t1init`, lane 0 = −Σ g0, lane 8 = −Σ g1) and the
    /// per-lane scale per pair (`t1sc`, lanes 0..8 = scale of g0, 8..16 = scale of g1).
    pub ptq1: bool,
    pub t1a: Vec<i16>,
    pub t1init: Vec<i32>,
    pub t1sc: Vec<f32>,
    /// The unquantised f32 activations (`x`), kept for F32/F16/BF16 matrix multiplies and exact reference paths.
    pub xf: Vec<f32>,
    /// Keep unquantised f32 activations in `xf` during `fill`.
    pub keep_f32: bool,
    /// Build the int8/ternary layouts in `fill`. False for a model whose every matrix
    /// reads the f32 rows (a pure F32/F16/BF16 file), where the codes would be
    /// produced and never read (S4.9h).
    pub quantize: bool,
    /// Build the per-256 layout (`qk`, `dk`, `bsums`) in `fill`. Set from the model's
    /// weight formats ([`Format::reads_k_layout`]): a model whose matmuls are all
    /// 32- or 128-weight formats never reads it, and the fill is a second, serial
    /// quantisation of the same row (S4.9a). Default on, so a bare `QuantAct` keeps
    /// the full layout every kernel table entry can be given.
    pub k_layout: bool,
}

/// `PEREGRINE_TQ_SPAN` (16, 32, 64 or 128): values sharing one activation scale in the ternary layout.
pub fn span_override() -> usize {
    use std::sync::OnceLock;
    static S: OnceLock<usize> = OnceLock::new();
    *S.get_or_init(|| crate::options::get().tq_span)
}

/// `PEREGRINE_EXACT=1`: ternary matmuls in f32 against the decoded weights (slow; a measurement switch).
pub fn exact_ternary() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| crate::options::get().exact)
}

/// Byte offset of group `g`'s planes in `QuantAct::tp` (pairs of groups interleaved by plane).
#[inline(always)]
pub fn tp_off(g: usize) -> usize {
    (g / 2) * 256 + (g % 2) * 32
}

impl Default for QuantAct {
    fn default() -> Self {
        QuantAct::with_capacity(0)
    }
}

impl QuantAct {
    pub fn with_capacity(n: usize) -> QuantAct {
        QuantAct {
            n: 0,
            prompt: false,
            q: Vec::with_capacity(n),
            d: Vec::with_capacity(n / 32 + 1),
            sums: Vec::with_capacity(n / 32 + 1),
            sums16: Vec::with_capacity(n / 16 + 1),
            lane_sums: Vec::with_capacity(n / 4 + 1),
            qk: Vec::with_capacity(n),
            dk: Vec::with_capacity(n / 256 + 1),
            bsums: Vec::with_capacity(n / 16 + 1),
            ternary: false,
            tq: Vec::with_capacity(n),
            tp: Vec::with_capacity(n),
            td: Vec::with_capacity(n / 16 + 1),
            tls: Vec::with_capacity(n / 16 + 1),
            xf: Vec::with_capacity(n),
            tb_span: 0,
            tb: Vec::new(),
            tbd: Vec::new(),
            tbn: Vec::new(),
            ptq1: false,
            t1a: Vec::new(),
            t1init: Vec::new(),
            t1sc: Vec::new(),
            keep_f32: false,
            quantize: true,
            k_layout: true,
        }
    }

    /// Whether later fills build the per-256 layout (`qk`, `dk`, `bsums`).
    ///
    /// A model whose matrices are all 32- or 128-weight formats never reads it
    /// ([`Format::reads_k_layout`]), so it is turned off once per model.
    pub fn set_k_layout(&mut self, on: bool) {
        self.k_layout = on;
    }

    // the per-128 ternary layout. Same rounding as the 32-block path (round half away from zero,
    // clamp to ±127), so the vector quantiser and this one agree code for code.
    pub fn fill_ternary(&mut self, x: &[f32]) {
        assert_eq!(
            x.len() % 128,
            0,
            "activation length {} is not a multiple of 128",
            x.len()
        );
        self.n = x.len();
        self.xf.clear();
        self.xf.extend_from_slice(x);
        if self.ptq1 {
            self.fill_ptq1(x);
            return;
        }
        self.tq.clear();
        self.tp.clear();
        self.td.clear();
        self.tls.clear();
        self.tq.resize(x.len(), 0);
        self.tp.resize(x.len().div_ceil(256) * 256, 0);
        self.td.resize(x.len() / 16, 0.0);
        self.tls.resize(x.len() / 16, 0);
        #[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
        {
            if self.tb_span > 0 {
                if crate::simd::have_avx2() {
                    // SAFETY: AVX2 checked; the buffers were sized above.
                    unsafe {
                        crate::tern::quantize_span_avx2(
                            x,
                            self.tb_span,
                            &mut self.tq,
                            &mut self.td,
                            &mut self.tls,
                        )
                    };
                } else {
                    self.quantize_span(x, self.tb_span);
                }
            } else if span_override() != 16 {
                self.quantize_span(x, span_override());
            } else if crate::simd::have_avx2() {
                // SAFETY: AVX2 checked; the buffers were sized above.
                unsafe {
                    crate::tern::quantize16_avx2(x, &mut self.tq, &mut self.td, &mut self.tls)
                };
            } else {
                self.quantize16_scalar(x);
            }
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "fmt-ternary")))]
        if self.tb_span > 0 {
            self.quantize_span(x, self.tb_span);
        } else {
            self.quantize16_scalar(x);
        }
        if self.tb_span > 0 {
            // Tile mode: every PQ2_0 matmul runs from tiles, so the per-row kernels' layout is not built.
            self.fill_broadcast();
            return;
        }
        // Pair-interleaved: groups 2p and 2p+1 alternate by plane, so one 512-bit load holds plane s of both.
        #[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
        {
            if crate::tern::have_vbmi() {
                // SAFETY: AVX-512 VBMI checked; tq holds x.len() codes and tp the padded layout.
                unsafe { crate::tern::permute_planes_vbmi(&self.tq, &mut self.tp) };
                return;
            }
        }
        for g in 0..x.len() / 128 {
            let base = tp_off(g);
            let q = &self.tq[g * 128..(g + 1) * 128];
            for k in 0..32 {
                for sl in 0..4 {
                    self.tp[base + 64 * sl + k] = q[4 * k + sl];
                }
            }
        }
    }

    /// One scale shared by every `span` values (a multiple of 16), written into each 16-block's slot: the
    /// accuracy comparison for `PEREGRINE_TQ_SPAN` net.
    pub fn quantize_span(&mut self, x: &[f32], span: usize) {
        for (c, blk) in x.chunks_exact(span).enumerate() {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for (i, &v) in blk.iter().enumerate() {
                self.tq[c * span + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
            for u in 0..span / 16 {
                let at = c * span / 16 + u;
                self.td[at] = d;
                self.tls[at] = self.tq[at * 16..at * 16 + 16]
                    .iter()
                    .map(|&q| q as i32)
                    .sum();
            }
        }
    }

    /// The tile kernel's broadcast order and per-span scales, from `tq`/`td`/`tls` (which carry the span scale).
    fn fill_broadcast(&mut self) {
        let n = self.n;
        let span = self.tb_span;
        self.tb.clear();
        self.tb.resize(n, 0);
        self.tbd.clear();
        self.tbn.clear();
        for c in 0..n / span {
            self.tbd.push(self.td[c * span / 16]);
            let sum: i32 = self.tls[c * span / 16..(c + 1) * span / 16].iter().sum();
            self.tbn.push(-sum);
        }
        // Within each 16 values the order is a 4×4 byte transpose (b[4s + i] = q[4i + s]).
        #[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
        {
            if crate::tern::have_vbmi() {
                // SAFETY: AVX-512 BW checked (with VBMI); both buffers hold n bytes, n a multiple of 128.
                unsafe { crate::tern::transpose4x4_bytes(&self.tq, &mut self.tb) };
                return;
            }
        }
        for g in 0..n / 128 {
            let q = &self.tq[g * 128..(g + 1) * 128];
            let b = &mut self.tb[g * 128..(g + 1) * 128];
            for qq in 0..8 {
                for s in 0..4 {
                    for i in 0..4 {
                        b[qq * 16 + s * 4 + i] = q[16 * qq + 4 * i + s];
                    }
                }
            }
        }
    }

    /// The PTQ1_0 layout: int16 codes (±32767) with one absmax scale per 128 values. `x.len()` must be a multiple
    /// of 256 (whole pairs of groups).
    pub fn fill_ptq1(&mut self, x: &[f32]) {
        assert_eq!(
            x.len() % 256,
            0,
            "PTQ1_0 activation length {} is not a multiple of 256",
            x.len()
        );
        let np = x.len() / 256;
        self.t1a.clear();
        self.t1a.resize(x.len(), 0);
        self.t1init.clear();
        self.t1init.resize(np * 16, 0);
        self.t1sc.clear();
        self.t1sc.resize(np * 16, 0.0);
        let mut q = [0i16; 128];
        for g in 0..x.len() / 128 {
            let blk = &x[g * 128..(g + 1) * 128];
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = if amax > 0.0 { amax / 32767.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut sum = 0i32;
            for (o, &v) in q.iter_mut().zip(blk) {
                *o = (v * inv).round().clamp(-32767.0, 32767.0) as i16;
                sum += *o as i32;
            }
            let (p, gi) = (g / 2, g % 2);
            for r in 0..8 {
                let at = p * 256 + r * 32 + gi * 16;
                self.t1a[at..at + 16].copy_from_slice(&q[16 * r..16 * r + 16]);
            }
            self.t1init[p * 16 + gi * 8] = -sum;
            for l in 0..8 {
                self.t1sc[p * 16 + gi * 8 + l] = d;
            }
        }
    }

    // the ternary layout's quantiser: one absmax scale per 16 values, round half away from zero.
    pub fn quantize16_scalar(&mut self, x: &[f32]) {
        for (u, blk) in x.chunks_exact(16).enumerate() {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut sum = 0i32;
            for (i, &v) in blk.iter().enumerate() {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                self.tq[u * 16 + i] = q;
                sum += q as i32;
            }
            self.td[u] = d;
            self.tls[u] = sum;
        }
    }

    /// Quantise `x` symmetrically, one absmax scale per 32 weights.
    ///
    /// Symmetric (no zero point) because every weight format here is either symmetric
    /// or carries its own min, so an activation zero point would have to be carried
    /// through the bias correction for no accuracy gain.
    pub fn fill(&mut self, x: &[f32]) {
        if self.keep_f32 {
            self.xf.clear();
            self.xf.extend_from_slice(x);
        }
        if !self.quantize {
            // No matrix of this model reads a quantised row (S4.9h): the f32 copy
            // above is the whole activation. `n` is still the row width, which the
            // float kernels read.
            self.n = x.len();
            return;
        }
        if self.ternary {
            self.fill_ternary(x);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if crate::simd::have_avx2() {
                // SAFETY: AVX2 checked above.
                unsafe { self.fill_avx2(x) };
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if crate::cpu::have_neon() {
                // SAFETY: NEON checked above.
                unsafe { self.fill_neon(x) };
                return;
            }
        }
        self.fill_scalar(x);
    }

    /// The vector `fill`: the same scales from the same scalar arithmetic, the codes
    /// eight at a time. It runs four times per layer per token, serially, and was a
    /// tenth of a SmolLM2 decode step on four threads.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn fill_avx2(&mut self, x: &[f32]) {
        assert_eq!(
            x.len() % 32,
            0,
            "activation length {} is not a multiple of 32",
            x.len()
        );
        self.n = x.len();
        self.q.clear();
        self.d.clear();
        self.sums.clear();
        self.sums16.clear();
        self.lane_sums.clear();
        self.qk.clear();
        self.dk.clear();
        self.bsums.clear();
        let mut codes = [0i8; 32];
        if self.k_layout && (x.len().is_multiple_of(256) || pad_k_layout(x.len())) {
            let n_sb = x.len().div_ceil(256);
            for sb in 0..n_sb {
                let lo = sb * 256;
                let blk = &x[lo..(lo + 256).min(x.len())];
                let amax = crate::simd::absmax(blk);
                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                let mut halves = 0;
                for half in blk.chunks_exact(32) {
                    crate::simd::quantize32(half.as_ptr(), inv, codes.as_mut_ptr());
                    self.qk.extend_from_slice(&codes);
                    for run in codes.chunks_exact(16) {
                        self.bsums.push(run.iter().map(|&q| q as i16).sum());
                    }
                    halves += 1;
                }
                for _ in halves..8 {
                    self.qk.extend_from_slice(&[0i8; 32]);
                    self.bsums.push(0);
                    self.bsums.push(0);
                }
                self.dk.push(d);
            }
        }
        for blk in x.chunks_exact(32) {
            let amax = crate::simd::absmax(blk);
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            crate::simd::quantize32(blk.as_ptr(), inv, codes.as_mut_ptr());
            self.q.extend_from_slice(&codes);
            for j in 0..8 {
                self.lane_sums
                    .push(codes[4 * j..4 * j + 4].iter().map(|&v| v as i32).sum());
            }
            let s_lo: i32 = codes[..16].iter().map(|&v| v as i32).sum();
            let s_hi: i32 = codes[16..].iter().map(|&v| v as i32).sum();
            self.d.push(d);
            self.sums16.push(s_lo);
            self.sums16.push(s_hi);
            self.sums.push(s_lo + s_hi);
        }
    }

    /// The vector `fill` for aarch64 (S2.8 F6): the scalar scales and the same integer
    /// sums, the codes four at a time through [`crate::neon::quantize32`]. The rounding
    /// is the scalar loop's, so the bytes are the same as [`QuantAct::fill_scalar`]'s.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn fill_neon(&mut self, x: &[f32]) {
        assert_eq!(
            x.len() % 32,
            0,
            "activation length {} is not a multiple of 32",
            x.len()
        );
        self.n = x.len();
        self.q.clear();
        self.d.clear();
        self.sums.clear();
        self.sums16.clear();
        self.lane_sums.clear();
        self.qk.clear();
        self.dk.clear();
        self.bsums.clear();
        let mut codes = [0i8; 32];
        if self.k_layout && (x.len().is_multiple_of(256) || pad_k_layout(x.len())) {
            let n_sb = x.len().div_ceil(256);
            for sb in 0..n_sb {
                let lo = sb * 256;
                let blk = &x[lo..(lo + 256).min(x.len())];
                let amax = crate::neon::absmax(blk);
                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                let mut halves = 0;
                for half in blk.chunks_exact(32) {
                    crate::neon::quantize32(half.as_ptr(), inv, codes.as_mut_ptr());
                    self.qk.extend_from_slice(&codes);
                    for run in codes.chunks_exact(16) {
                        self.bsums.push(run.iter().map(|&q| q as i16).sum());
                    }
                    halves += 1;
                }
                for _ in halves..8 {
                    self.qk.extend_from_slice(&[0i8; 32]);
                    self.bsums.push(0);
                    self.bsums.push(0);
                }
                self.dk.push(d);
            }
        }
        for blk in x.chunks_exact(32) {
            let amax = crate::neon::absmax(blk);
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            crate::neon::quantize32(blk.as_ptr(), inv, codes.as_mut_ptr());
            self.q.extend_from_slice(&codes);
            for j in 0..8 {
                self.lane_sums
                    .push(codes[4 * j..4 * j + 4].iter().map(|&v| v as i32).sum());
            }
            let s_lo: i32 = codes[..16].iter().map(|&v| v as i32).sum();
            let s_hi: i32 = codes[16..].iter().map(|&v| v as i32).sum();
            self.d.push(d);
            self.sums16.push(s_lo);
            self.sums16.push(s_hi);
            self.sums.push(s_lo + s_hi);
        }
    }

    /// Size and clear the buffers for a parallel fill of `n` activations.
    ///
    /// [`QuantAct::fill_range`] writes a worker's contiguous block range into these buffers; the
    /// per-32 scales are independent, so any split on a block boundary produces the
    /// same bytes as one call to [`QuantAct::fill`]. The 256-block layout is written only when
    /// `n` is a multiple of 256 and every range starts on a 256 boundary.
    pub fn prepare(&mut self, n: usize) {
        assert_eq!(n % 32, 0, "an activation is a whole number of 32-blocks");
        self.n = n;
        self.q.clear();
        self.q.resize(n, 0);
        self.d.clear();
        self.d.resize(n / 32, 0.0);
        self.sums.clear();
        self.sums.resize(n / 32, 0);
        self.sums16.clear();
        self.sums16.resize(n / 16, 0);
        self.lane_sums.clear();
        self.lane_sums.resize(n / 4, 0);
        self.qk.clear();
        self.dk.clear();
        self.bsums.clear();
        if self.k_layout && n.is_multiple_of(256) {
            self.qk.resize(n, 0);
            self.dk.resize(n / 256, 0.0);
            self.bsums.resize(n / 16, 0);
        }
        self.xf.clear();
        if self.keep_f32 {
            self.xf.resize(n, 0.0);
        }
    }

    /// Fill blocks starting at `lo` from `x`, after [`QuantAct::prepare`].
    ///
    /// `x` holds the activation values from block `lo/32` to the end of the worker's
    /// range; its length is a multiple of 32. When `n` is a multiple of 256, `lo` and
    /// the range length are multiples of 256, so the per-256 layout is filled too.
    ///
    /// The arithmetic is [`QuantAct::fill`]'s, block for block: a block's scale and codes depend
    /// only on that block, so who computed it cannot change the bytes.
    pub fn fill_range(&mut self, x: &[f32], lo: usize) {
        assert_eq!(x.len() % 32, 0, "a range is a whole number of 32-blocks");
        assert_eq!(lo % 32, 0, "a range starts on a 32-block");
        assert!(lo + x.len() <= self.n, "the range fits the activation");
        if self.keep_f32 {
            self.xf[lo..lo + x.len()].copy_from_slice(x);
        }
        #[cfg(target_arch = "x86_64")]
        {
            if crate::simd::have_avx2() {
                // SAFETY: AVX2 checked; the buffers were sized by `prepare` and the
                // range assertions keep every write inside them.
                unsafe { self.fill_range_avx2(x, lo) };
                return;
            }
        }
        self.fill_range_scalar(x, lo);
    }

    /// The vector form of [`QuantAct::fill_range`], the arithmetic of `fill_avx2`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn fill_range_avx2(&mut self, x: &[f32], lo: usize) {
        let mut codes = [0i8; 32];
        if self.k_layout && self.n.is_multiple_of(256) {
            assert_eq!(lo % 256, 0, "the per-256 layout needs 256-aligned ranges");
            for (s, blk) in x.chunks_exact(256).enumerate() {
                let sb = lo / 256 + s;
                let amax = crate::simd::absmax(blk);
                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for (h, half) in blk.chunks_exact(32).enumerate() {
                    crate::simd::quantize32(half.as_ptr(), inv, codes.as_mut_ptr());
                    let at = sb * 256 + h * 32;
                    self.qk[at..at + 32].copy_from_slice(&codes);
                    for (i, run) in codes.chunks_exact(16).enumerate() {
                        self.bsums[sb * 16 + h * 2 + i] = run.iter().map(|&q| q as i16).sum();
                    }
                }
                self.dk[sb] = d;
            }
        }
        for (s, blk) in x.chunks_exact(32).enumerate() {
            let b = lo / 32 + s;
            let amax = crate::simd::absmax(blk);
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            crate::simd::quantize32(blk.as_ptr(), inv, codes.as_mut_ptr());
            self.q[b * 32..b * 32 + 32].copy_from_slice(&codes);
            for j in 0..8 {
                self.lane_sums[b * 8 + j] = codes[4 * j..4 * j + 4].iter().map(|&v| v as i32).sum();
            }
            let s_lo: i32 = codes[..16].iter().map(|&v| v as i32).sum();
            let s_hi: i32 = codes[16..].iter().map(|&v| v as i32).sum();
            self.d[b] = d;
            self.sums16[b * 2] = s_lo;
            self.sums16[b * 2 + 1] = s_hi;
            self.sums[b] = s_lo + s_hi;
        }
    }

    /// The scalar specification of [`QuantAct::fill_range`].
    fn fill_range_scalar(&mut self, x: &[f32], lo: usize) {
        if self.k_layout && self.n.is_multiple_of(256) {
            assert_eq!(lo % 256, 0, "the per-256 layout needs 256-aligned ranges");
            for (s, blk) in x.chunks_exact(256).enumerate() {
                let sb = lo / 256 + s;
                let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for (h, run) in blk.chunks_exact(16).enumerate() {
                    let mut sum = 0i16;
                    for (i, &v) in run.iter().enumerate() {
                        let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                        self.qk[sb * 256 + h * 16 + i] = q;
                        sum += q as i16;
                    }
                    self.bsums[sb * 16 + h] = sum;
                }
                self.dk[sb] = d;
            }
        }
        for (s, blk) in x.chunks_exact(32).enumerate() {
            let b = lo / 32 + s;
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut s_lo = 0i32;
            let mut s_hi = 0i32;
            for (i, &v) in blk.iter().enumerate() {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                self.q[b * 32 + i] = q;
                if i < 16 {
                    s_lo += q as i32;
                } else {
                    s_hi += q as i32;
                }
            }
            for j in 0..8 {
                self.lane_sums[b * 8 + j] = self.q[b * 32 + 4 * j..b * 32 + 4 * j + 4]
                    .iter()
                    .map(|&v| v as i32)
                    .sum();
            }
            self.d[b] = d;
            self.sums16[b * 2] = s_lo;
            self.sums16[b * 2 + 1] = s_hi;
            self.sums[b] = s_lo + s_hi;
        }
    }

    /// The scalar specification of `fill`, which the vector form is tested against.
    pub fn fill_scalar(&mut self, x: &[f32]) {
        assert_eq!(
            x.len() % 32,
            0,
            "activation length {} is not a multiple of 32",
            x.len()
        );
        self.n = x.len();
        self.q.clear();
        self.d.clear();
        self.sums.clear();
        self.sums16.clear();
        self.lane_sums.clear();
        self.qk.clear();
        self.dk.clear();
        self.bsums.clear();

        if self.k_layout && (x.len().is_multiple_of(256) || pad_k_layout(x.len())) {
            let n_sb = x.len().div_ceil(256);
            for sb in 0..n_sb {
                let lo = sb * 256;
                let blk = &x[lo..(lo + 256).min(x.len())];
                let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                let mut runs = 0;
                for run in blk.chunks_exact(16) {
                    let mut s = 0i16;
                    for &v in run {
                        let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                        self.qk.push(q);
                        s += q as i16;
                    }
                    self.bsums.push(s);
                    runs += 1;
                }
                for _ in runs..16 {
                    self.qk.extend_from_slice(&[0i8; 16]);
                    self.bsums.push(0);
                }
                self.dk.push(d);
            }
        }

        for blk in x.chunks_exact(32) {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // An all-zero block is real (a masked position, a dead channel). Scale 0
            // with codes 0 reconstructs exactly, and avoids a division by zero.
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut s_lo = 0i32;
            let mut s_hi = 0i32;
            for (i, &v) in blk.iter().enumerate() {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                self.q.push(q);
                if i < 16 {
                    s_lo += q as i32;
                } else {
                    s_hi += q as i32;
                }
            }
            let base = self.q.len() - 32;
            for j in 0..8 {
                let l = &self.q[base + 4 * j..base + 4 * j + 4];
                self.lane_sums.push(l.iter().map(|&v| v as i32).sum());
            }
            self.d.push(d);
            self.sums16.push(s_lo);
            self.sums16.push(s_hi);
            self.sums.push(s_lo + s_hi);
        }
    }
}

/// `f32` → IEEE half, round-to-nearest-even, with overflow to infinity and gradual
/// underflow — the inverse of `gguf::f16`, for writing block scales.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal half: shift the full mantissa (with its implicit bit) into place.
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut q = m >> shift;
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
        return sign | q as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1; // a carry into the exponent is the correct rounding up
    }
    sign | h as u16
}

/// One row as Q8_0 blocks (34 bytes per 32 weights), appended to `out`. The same
/// arithmetic as llama.cpp's `quantize_row_q8_0_ref`: `d = amax/127` stored as f16,
/// codes `round(x / d)` from the unrounded `d`.
#[cfg(feature = "fmt-legacy")]
pub fn quantize_row_q8_0(x: &[f32], out: &mut Vec<u8>) {
    assert!(
        x.len().is_multiple_of(32),
        "Q8_0 rows are whole blocks of 32"
    );
    for blk in x.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend(blk.iter().map(|v| (v * id).round() as i8 as u8));
    }
}

/// ggml's `nearest_int`: round to nearest, ties to even, by the float-add trick —
/// reproduced exactly so the quantiser below picks the same codes as llama.cpp's.
#[cfg(feature = "fmt-kquant")]
#[inline]
fn nearest_int(f: f32) -> i32 {
    let v = f + 12_582_912.0;
    ((v.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

/// `make_qkx2_quants` from `ggml-quants.c`: a scale and min for one 32-weight
/// sub-block, searched over `nstep` candidate scales to minimise the weighted error.
#[cfg(feature = "fmt-kquant")]
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    x: &[f32],
    nmax: i32,
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> f32 {
    let n = x.len();
    let (mut min, mut max) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l.iter_mut().for_each(|v| *v = 0);
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = q as u8;
        let d = scale * q as f32 + min - x[i];
        best += weights[i] * d * d;
    }
    let mut laux = [0u8; 32];
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sl, mut sl2, mut sxl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = q as u8;
            let w = weights[i];
            sl += w * q as f32;
            sl2 += w * (q * q) as f32;
            sxl += w * q as f32 * x[i];
        }
        let det = sum_w * sl2 - sl * sl;
        if det > 0.0 {
            let mut this_scale = (sum_w * sxl - sum_x * sl) / det;
            let mut this_min = (sl2 * sum_x - sl * sxl) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sxl / sl2;
            }
            let mut err = 0.0f32;
            for i in 0..n {
                let d = this_scale * laux[i] as f32 + this_min - x[i];
                err += weights[i] * d * d;
            }
            if err < best {
                l[..n].copy_from_slice(&laux[..n]);
                best = err;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// One row as Q4_K super-blocks (144 bytes per 256 weights), appended to `out` —
/// `quantize_row_q4_K_ref` from `ggml-quants.c`, step for step, so a row quantised
/// here is the row llama.cpp would write. Used for the DSpark drafter, whose bf16
/// checkpoint would otherwise run at 8.5 bits per weight.
#[cfg(feature = "fmt-kquant")]
pub fn quantize_row_q4_k(x: &[f32], out: &mut Vec<u8>) {
    assert!(
        x.len().is_multiple_of(256),
        "Q4_K rows are whole super-blocks of 256"
    );
    for xb in x.chunks_exact(256) {
        let mut l = [0u8; 256];
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        let (mut max_scale, mut max_min) = (0.0f32, 0.0f32);
        for j in 0..8 {
            let sub = &xb[32 * j..32 * j + 32];
            let sum_x2: f32 = sub.iter().map(|v| v * v).sum();
            let av_x = (sum_x2 / 32.0).sqrt();
            let mut w = [0.0f32; 32];
            for (wi, v) in w.iter_mut().zip(sub) {
                *wi = av_x + v.abs();
            }
            scales[j] = make_qkx2_quants(
                sub,
                15,
                &w,
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                -1.0,
                0.1,
                20,
            );
            max_scale = max_scale.max(scales[j]);
            max_min = max_min.max(mins[j]);
        }
        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut sc = [0u8; 12];
        for j in 0..8 {
            let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
            let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
            if j < 4 {
                sc[j] = ls;
                sc[j + 4] = lm;
            } else {
                sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                sc[j - 4] |= (ls >> 4) << 6;
                sc[j] |= (lm >> 4) << 6;
            }
        }
        let d16 = f32_to_f16(max_scale / 63.0);
        let dmin16 = f32_to_f16(max_min / 63.0);
        let (d, dmin) = (f16(d16), f16(dmin16));
        for j in 0..8 {
            let (s6, m6) = scale_min_k4(j, &sc);
            let dj = d * s6 as f32;
            if dj == 0.0 {
                continue;
            }
            let dm = dmin * m6 as f32;
            for ii in 0..32 {
                l[32 * j + ii] = nearest_int((xb[32 * j + ii] + dm) / dj).clamp(0, 15) as u8;
            }
        }
        out.extend_from_slice(&d16.to_le_bytes());
        out.extend_from_slice(&dmin16.to_le_bytes());
        out.extend_from_slice(&sc);
        for j in (0..256).step_by(64) {
            for i in 0..32 {
                out.push(l[j + i] | (l[j + i + 32] << 4));
            }
        }
    }
}

/// `get_scale_min_k4` from `ggml-quants.c`: unpack the 6-bit scale and min for
/// sub-block `j` of a k-quant super-block out of the 12 packed bytes.
#[cfg(feature = "fmt-kquant")]
#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

#[inline(always)]
fn h(b: &[u8], i: usize) -> f32 {
    f16(u16::from_le_bytes([b[i], b[i + 1]]))
}

// dequantize_row_ptq1_0's order, as signed trits. 16 bytes × 5 trits (value 16n + m is trit n of byte m),
// 8 bytes × 5 trits (80 + 8n + m), then the 2 `qh` bytes × 4 trits (120 + 2n + h). Trit n of byte b is
// `((b · 3^n mod 256) · 3) >> 8`.
pub fn ptq1_trits(blk: &[u8], t: &mut [i8; 128]) {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let trit = |b: u8, n: usize| ((((b.wrapping_mul(POW3[n])) as u16) * 3) >> 8) as i8 - 1;
    for n in 0..5 {
        for m in 0..16 {
            t[16 * n + m] = trit(blk[m], n);
        }
        for m in 0..8 {
            t[80 + 8 * n + m] = trit(blk[16 + m], n);
        }
    }
    for n in 0..4 {
        for hh in 0..2 {
            t[120 + 2 * n + hh] = trit(blk[24 + hh], n);
        }
    }
}

// one PQ2_0 group of `a` (group index `g`), the scalar specification's inner sum, for kernel tails.
pub fn dot_scalar_pq2_group(blk: &[u8], a: &QuantAct, g: usize) -> f32 {
    let mut gs = 0.0f32;
    for u in 0..8 {
        let mut s = 0i32;
        for j in 16 * u..16 * u + 16 {
            let c = ((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) as i32;
            s += c * a.tq[g * 128 + j] as i32;
        }
        gs += a.td[8 * g + u] * (s - a.tls[8 * g + u]) as f32;
    }
    h(blk, 0) * gs
}

/// Number of bytes one row of `n` weights occupies in `fmt`.
pub fn row_bytes(fmt: Format, n: usize) -> usize {
    debug_assert_eq!(n % fmt.block(), 0);
    n / fmt.block() * fmt.block_bytes()
}

// ---------------------------------------------------------------------------
// Reference decode. Mirrors `dequantize_row_*` in ggml-quants.c element for
// element, including output interleaving. Used for embedding lookup, and as the
// oracle the fused kernels are checked against.
// ---------------------------------------------------------------------------

/// The 16-entry lookup table for GGML's non-linear 4-bit formats (IQ4_NL and IQ4_XS),
/// transcribed from `kvalues_iq4nl` in `ggml-common.h`.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// One row of `F16` weights, widened into `y` — the scalar loop by default, sixteen
/// halves at a time with F16C where the tier has it (`S4.9r`).
///
/// The scalar body is the definition: `f16`'s bit-for-bit widening of a half. F16C's
/// `vcvtph2ps` is the same conversion in hardware — every finite half is exact, an
/// infinity stays an infinity, and a NaN keeps its mantissa bits (they are the bits
/// `f16` shifts into place) — and the tests hold every one of the 65 536 half patterns
/// to it.
#[inline]
fn dequant_f16(w: &[u8], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::have_batched() {
            // SAFETY: AVX2 and F16C checked; `w` holds two bytes per `y` element.
            unsafe { dequant_f16_avx2(w, y) };
            return;
        }
    }
    dequant_f16_scalar(w, y);
}

/// [`dequant_f16`]'s scalar twin.
#[inline]
fn dequant_f16_scalar(w: &[u8], y: &mut [f32]) {
    for (o, c) in y.iter_mut().zip(w.chunks_exact(2)) {
        *o = f16(u16::from_le_bytes(c.try_into().unwrap()));
    }
}

/// [`dequant_f16`] on AVX2: sixteen halves per step with `vcvtph2ps`.
///
/// # Safety
/// AVX2 and F16C must be available, and `w` must hold at least two bytes per element
/// of `y`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn dequant_f16_avx2(w: &[u8], y: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = y.len();
    let src = w.as_ptr();
    let dst = y.as_mut_ptr();
    let mut i = 0;
    while i + 16 <= n {
        let h0 = _mm_loadu_si128(src.add(i * 2) as *const __m128i);
        let h1 = _mm_loadu_si128(src.add(i * 2 + 16) as *const __m128i);
        let f0 = _mm256_cvtph_ps(h0);
        let f1 = _mm256_cvtph_ps(h1);
        _mm256_storeu_ps(dst.add(i), f0);
        _mm256_storeu_ps(dst.add(i + 8), f1);
        // A NaN's payload is not the hardware's to canonicalise: `vcvtph2ps` quiets it
        // (and drops its mantissa bits), while the definition shifts them into place. No
        // real weight is a NaN, so this branch never runs on a model; when one is, the
        // sixteen values are re-widened by the scalar body, which is the definition.
        let nan = _mm256_or_ps(
            _mm256_cmp_ps(f0, f0, _CMP_UNORD_Q),
            _mm256_cmp_ps(f1, f1, _CMP_UNORD_Q),
        );
        if _mm256_movemask_ps(nan) != 0 {
            for j in i..i + 16 {
                *dst.add(j) = f16(u16::from_le_bytes([*src.add(j * 2), *src.add(j * 2 + 1)]));
            }
        }
        i += 16;
    }
    while i < n {
        *dst.add(i) = f16(u16::from_le_bytes([*src.add(i * 2), *src.add(i * 2 + 1)]));
        i += 1;
    }
}

/// The 16-entry lookup table for GGML's MXFP4 format, transcribed from `kvalues_fp4` / `kvalues_mxfp4`
/// in `ggml-common.h`. Note that values are doubled relative to OCP E2M1 floats: [0, 1, 2, 3, 4, 6, 8, 12, ...].
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

// The Q6_K decoders below contain `qh[l] >> 0`. It is an identity, and it is kept:
// the four shifts form the series 0/2/4/6, and writing the first one as a bare
// `qh[l]` breaks the column alignment that lets a reader check this transcription
// line-for-line against `ggml-quants.c`. That checkability is the whole reason the
// scalar path is the specification.
#[allow(clippy::identity_op, clippy::needless_range_loop)]
pub fn dequant_row(fmt: Format, w: &[u8], y: &mut [f32]) {
    let n = y.len();
    debug_assert_eq!(w.len(), row_bytes(fmt, n));
    match fmt {
        Format::F32 => {
            for (o, c) in y.iter_mut().zip(w.chunks_exact(4)) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        Format::F16 => dequant_f16(w, y),
        Format::BF16 => {
            for (o, c) in y.iter_mut().zip(w.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16);
            }
        }
        #[cfg(feature = "fmt-ternary")]
        Format::PQ2_0 => {
            // dequantize_row_pq2_0: byte j/4, bits 2(j%4); 00=-1, 01=0, 10=+1 (11=+2, unused by ternary files).
            for (blk, out) in w.chunks_exact(34).zip(y.chunks_exact_mut(128)) {
                let d = h(blk, 0);
                for j in 0..128 {
                    let q = (blk[2 + j / 4] >> ((j % 4) * 2)) & 3;
                    out[j] = (q as i32 - 1) as f32 * d;
                }
            }
        }
        #[cfg(feature = "fmt-ternary")]
        Format::PTQ1_0 => {
            for (blk, out) in w.chunks_exact(28).zip(y.chunks_exact_mut(128)) {
                let d = h(blk, 26);
                let mut t = [0i8; 128];
                ptq1_trits(blk, &mut t);
                for j in 0..128 {
                    out[j] = t[j] as f32 * d;
                }
            }
        }
        #[cfg(feature = "fmt-legacy")]
        Format::Q8_0 => {
            for (blk, out) in w.chunks_exact(34).zip(y.chunks_exact_mut(32)) {
                let d = h(blk, 0);
                for j in 0..32 {
                    out[j] = d * (blk[2 + j] as i8) as f32;
                }
            }
        }
        #[cfg(feature = "fmt-legacy")]
        Format::Q4_0 => {
            for (blk, out) in w.chunks_exact(18).zip(y.chunks_exact_mut(32)) {
                let d = h(blk, 0);
                let qs = &blk[2..18];
                for j in 0..16 {
                    out[j] = d * ((qs[j] & 0x0F) as i32 - 8) as f32;
                    out[j + 16] = d * ((qs[j] >> 4) as i32 - 8) as f32;
                }
            }
        }
        #[cfg(feature = "fmt-legacy")]
        Format::Q5_0 => {
            for (blk, out) in w.chunks_exact(22).zip(y.chunks_exact_mut(32)) {
                let d = h(blk, 0);
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let qs = &blk[6..22];
                // The two halves are interleaved: low nibbles produce outputs 0..16,
                // high nibbles produce 16..32, and the fifth bit for output j lives
                // at bit j of qh (and bit j+16 for the upper half).
                for j in 0..16 {
                    let hi0 = ((qh >> j) << 4) as u8 & 0x10;
                    let hi1 = (qh >> (j + 12)) as u8 & 0x10;
                    out[j] = d * (((qs[j] & 0x0F) | hi0) as i32 - 16) as f32;
                    out[j + 16] = d * (((qs[j] >> 4) | hi1) as i32 - 16) as f32;
                }
            }
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q2_K => {
            for (blk, out) in w.chunks_exact(84).zip(y.chunks_exact_mut(256)) {
                let scales = &blk[0..16];
                let qs = &blk[16..80];
                let d = h(blk, 80);
                let dmin = h(blk, 82);
                let mut is = 0;
                for n in 0..2 {
                    let q = &qs[n * 32..(n + 1) * 32];
                    for j in 0..4 {
                        let shift = j * 2;
                        let sc0 = scales[is];
                        let dl0 = d * (sc0 & 0x0F) as f32;
                        let ml0 = dmin * (sc0 >> 4) as f32;
                        let o0 = &mut out[is * 16..(is + 1) * 16];
                        for l in 0..16 {
                            o0[l] = dl0 * ((q[l] >> shift) & 3) as f32 - ml0;
                        }
                        is += 1;

                        let sc1 = scales[is];
                        let dl1 = d * (sc1 & 0x0F) as f32;
                        let ml1 = dmin * (sc1 >> 4) as f32;
                        let o1 = &mut out[is * 16..(is + 1) * 16];
                        for l in 0..16 {
                            o1[l] = dl1 * ((q[l + 16] >> shift) & 3) as f32 - ml1;
                        }
                        is += 1;
                    }
                }
            }
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q3_K => {
            for (blk, out) in w.chunks_exact(110).zip(y.chunks_exact_mut(256)) {
                let hmask = &blk[0..32];
                let qs = &blk[32..96];
                let scales_raw = &blk[96..108];
                let d = h(blk, 108);

                let a0 = u32::from_le_bytes(scales_raw[0..4].try_into().unwrap());
                let a1 = u32::from_le_bytes(scales_raw[4..8].try_into().unwrap());
                let tmp = u32::from_le_bytes(scales_raw[8..12].try_into().unwrap());
                const KMASK1: u32 = 0x03030303;
                const KMASK2: u32 = 0x0f0f0f0f;
                let s0 = (a0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
                let s1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
                let s2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
                let s3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
                let mut scales = [0i8; 16];
                for (chunk_idx, &word) in [s0, s1, s2, s3].iter().enumerate() {
                    for (i, b) in word.to_le_bytes().into_iter().enumerate() {
                        scales[chunk_idx * 4 + i] = b as i8;
                    }
                }

                let mut is = 0;
                let mut m = 1u8;
                for n in 0..2 {
                    let q = &qs[n * 32..(n + 1) * 32];
                    for j in 0..4 {
                        let shift = j * 2;
                        let dl0 = d * (scales[is] as i32 - 32) as f32;
                        let o0 = &mut out[is * 16..(is + 1) * 16];
                        for l in 0..16 {
                            let hbit = (hmask[l] & m) != 0;
                            let q_val = ((q[l] >> shift) & 3) as i32 - if hbit { 0 } else { 4 };
                            o0[l] = dl0 * q_val as f32;
                        }
                        is += 1;

                        let dl1 = d * (scales[is] as i32 - 32) as f32;
                        let o1 = &mut out[is * 16..(is + 1) * 16];
                        for l in 0..16 {
                            let hbit = (hmask[l + 16] & m) != 0;
                            let q_val =
                                ((q[l + 16] >> shift) & 3) as i32 - if hbit { 0 } else { 4 };
                            o1[l] = dl1 * q_val as f32;
                        }
                        is += 1;

                        m <<= 1;
                    }
                }
            }
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q4_K => {
            for (blk, out) in w.chunks_exact(144).zip(y.chunks_exact_mut(256)) {
                let d = h(blk, 0);
                let dmin = h(blk, 2);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                for is in 0..8 {
                    let (sc, m) = scale_min_k4(is, scales);
                    let d1 = d * sc as f32;
                    let m1 = dmin * m as f32;
                    // Sub-blocks alternate between the low and high nibble of the
                    // same 32 bytes: is=0 low of qs[0..32], is=1 high of qs[0..32],
                    // is=2 low of qs[32..64], and so on.
                    let base = (is / 2) * 32;
                    let o = &mut out[is * 32..is * 32 + 32];
                    if is % 2 == 0 {
                        for l in 0..32 {
                            o[l] = d1 * (qs[base + l] & 0x0F) as f32 - m1;
                        }
                    } else {
                        for l in 0..32 {
                            o[l] = d1 * (qs[base + l] >> 4) as f32 - m1;
                        }
                    }
                }
            }
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q5_K => {
            for (blk, out) in w.chunks_exact(176).zip(y.chunks_exact_mut(256)) {
                let d = h(blk, 0);
                let dmin = h(blk, 2);
                let scales = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                for is in 0..8 {
                    let (sc, m) = scale_min_k4(is, scales);
                    let d1 = d * sc as f32;
                    let m1 = dmin * m as f32;
                    let base = (is / 2) * 32;
                    let o = &mut out[is * 32..is * 32 + 32];
                    let u = 1u8 << is;
                    if is % 2 == 0 {
                        for l in 0..32 {
                            let hi = if (qh[l] & u) != 0 { 16 } else { 0 };
                            o[l] = d1 * ((qs[base + l] & 0x0F) | hi) as f32 - m1;
                        }
                    } else {
                        for l in 0..32 {
                            let hi = if (qh[l] & u) != 0 { 16 } else { 0 };
                            o[l] = d1 * ((qs[base + l] >> 4) | hi) as f32 - m1;
                        }
                    }
                }
            }
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q6_K => {
            for (blk, out) in w.chunks_exact(210).zip(y.chunks_exact_mut(256)) {
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc: &[i8] =
                    unsafe { std::slice::from_raw_parts(blk[192..208].as_ptr() as *const i8, 16) };
                let d = h(blk, 208);
                for c in 0..2 {
                    let ql = &ql[c * 64..c * 64 + 64];
                    let qh = &qh[c * 32..c * 32 + 32];
                    let sc = &sc[c * 8..c * 8 + 8];
                    let out = &mut out[c * 128..c * 128 + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        out[l] = d * sc[is] as f32 * q1 as f32;
                        out[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                        out[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                        out[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                    }
                }
            }
        }
        #[cfg(feature = "fmt-iq")]
        Format::IQ4_NL => {
            for (blk, out) in w.chunks_exact(18).zip(y.chunks_exact_mut(32)) {
                let d = h(blk, 0);
                let qs = &blk[2..18];
                for j in 0..16 {
                    out[j] = d * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32;
                    out[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }
            }
        }
        #[cfg(feature = "fmt-iq")]
        Format::IQ4_XS => {
            for (blk, out) in w.chunks_exact(136).zip(y.chunks_exact_mut(256)) {
                let d = h(blk, 0);
                let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
                let scales_l = &blk[4..8];
                let qs = &blk[8..136];
                for ib in 0..8 {
                    let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as i32
                        | ((((scales_h >> (2 * ib)) & 3) << 4) as i32);
                    let dl = d * (ls - 32) as f32;
                    let sub_qs = &qs[ib * 16..(ib + 1) * 16];
                    let o = &mut out[ib * 32..ib * 32 + 32];
                    for j in 0..16 {
                        o[j] = dl * KVALUES_IQ4NL[(sub_qs[j] & 0x0F) as usize] as f32;
                        o[j + 16] = dl * KVALUES_IQ4NL[(sub_qs[j] >> 4) as usize] as f32;
                    }
                }
            }
        }
        #[cfg(feature = "fmt-mxfp4")]
        Format::MXFP4 => {
            for (blk, out) in w.chunks_exact(17).zip(y.chunks_exact_mut(32)) {
                let d = crate::format::e8m0_to_f32_half(blk[0]);
                let qs = &blk[1..17];
                for j in 0..16 {
                    out[j] = d * KVALUES_MXFP4[(qs[j] & 0x0F) as usize] as f32;
                    out[j + 16] = d * KVALUES_MXFP4[(qs[j] >> 4) as usize] as f32;
                }
            }
        }
        #[allow(unreachable_patterns)]
        _ => panic!("format {} disabled in this build", fmt.label()),
    }
}

// ---------------------------------------------------------------------------
// Fused dequantise-and-dot, scalar. `Σ w_i · â_i` where `â` is the int8
// reconstruction of the activation. No allocation, no float weight buffer.
// ---------------------------------------------------------------------------

/// Fused dequantise-and-dot, taking whichever kernel this machine can run.
///
/// The scalar twin is the specification (spec P3); this dispatch exists only to make
/// it fast. `tests/simd.rs` asserts the two agree on the real weight file, so a
/// divergence is a test failure rather than a subtly wrong model.
#[inline]
pub fn dot(fmt: Format, w: &[u8], a: &QuantAct) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::have_avx2() {
            if fmt == Format::F16 && !crate::cpu_has!("f16c") {
                return dot_scalar(fmt, w, a);
            }
            // SAFETY: feature checked immediately above; lengths are the same ones
            // `dot_scalar` asserts on.
            return unsafe { crate::simd::dot(fmt, w, a) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_dotprod() {
            // SAFETY: `asimddp` checked immediately above; the lengths are the same
            // ones `dot_scalar` asserts on.
            if let Some(v) = unsafe { crate::neon::dot(fmt, w, a) } {
                return v;
            }
        }
    }
    dot_scalar(fmt, w, a)
}

/// One weight row against every activation in `acts`, written to `out[..acts.len()]`.
///
/// This is what `matmul` calls. On a machine with AVX2 and FMA the row is decoded
/// once per tile of four activations instead of once per activation (see the batched
/// kernels in `simd.rs`); otherwise it is the plain per-activation loop, so the
/// result never depends on which path ran by more than float rounding.
#[inline]
pub fn dot_batch(fmt: Format, w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
    debug_assert!(out.len() >= acts.len());
    #[cfg(feature = "fmt-ternary")]
    if matches!(fmt, Format::PQ2_0 | Format::PTQ1_0) && exact_ternary() {
        let n = acts[0].n;
        let mut y = vec![0.0f32; n];
        dequant_row(fmt, w, &mut y);
        for (b, a) in acts.iter().enumerate() {
            out[b] = y
                .iter()
                .zip(&a.xf)
                .map(|(p, q)| *p as f64 * *q as f64)
                .sum::<f64>() as f32;
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::have_batched() {
            // SAFETY: feature checked immediately above; the length preconditions are
            // the same ones `matmul` already guarantees for `dot`.
            return unsafe { crate::simd::dot_batch(fmt, w, acts, out) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::cpu::have_dotprod() {
            // SAFETY: `asimddp` checked immediately above; the width precondition is the
            // same one the per-row loop below relies on.
            if unsafe { crate::neon::dot_batch(fmt, w, acts, out) } {
                return;
            }
        }
    }
    for (b, a) in acts.iter().enumerate() {
        out[b] = dot(fmt, w, a);
    }
}

#[allow(clippy::identity_op, clippy::needless_range_loop)]
pub fn dot_scalar(fmt: Format, w: &[u8], a: &QuantAct) -> f32 {
    debug_assert_eq!(w.len(), row_bytes(fmt, a.n));
    match fmt {
        #[cfg(feature = "fmt-legacy")]
        Format::Q8_0 => {
            let mut acc = 0.0f32;
            for (b, blk) in w.chunks_exact(34).enumerate() {
                let mut s = 0i32;
                let aq = &a.q[b * 32..b * 32 + 32];
                for j in 0..32 {
                    s += (blk[2 + j] as i8) as i32 * aq[j] as i32;
                }
                // Signed codes already, so no bias term.
                acc += h(blk, 0) * a.d[b] * s as f32;
            }
            acc
        }
        #[cfg(feature = "fmt-legacy")]
        Format::Q4_0 => {
            let mut acc = 0.0f32;
            for (b, blk) in w.chunks_exact(18).enumerate() {
                let qs = &blk[2..18];
                let aq = &a.q[b * 32..b * 32 + 32];
                let mut s = 0i32;
                for j in 0..16 {
                    s += (qs[j] & 0x0F) as i32 * aq[j] as i32;
                    s += (qs[j] >> 4) as i32 * aq[j + 16] as i32;
                }
                // Bias identity: codes are [0,15] with offset 8, so subtract
                // 8·Σa once for the whole block instead of per weight.
                acc += h(blk, 0) * a.d[b] * (s - 8 * a.sums[b]) as f32;
            }
            acc
        }
        #[cfg(feature = "fmt-legacy")]
        Format::Q5_0 => {
            let mut acc = 0.0f32;
            for (b, blk) in w.chunks_exact(22).enumerate() {
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let qs = &blk[6..22];
                let aq = &a.q[b * 32..b * 32 + 32];
                let mut s = 0i32;
                for j in 0..16 {
                    let hi0 = ((qh >> j) << 4) as u8 & 0x10;
                    let hi1 = (qh >> (j + 12)) as u8 & 0x10;
                    s += ((qs[j] & 0x0F) | hi0) as i32 * aq[j] as i32;
                    s += ((qs[j] >> 4) | hi1) as i32 * aq[j + 16] as i32;
                }
                // Bias identity: codes are [0,31] with offset 16, so subtract
                // 16·Σa once for the whole block instead of per weight.
                acc += h(blk, 0) * a.d[b] * (s - 16 * a.sums[b]) as f32;
            }
            acc
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q2_K => {
            assert!(
                !a.qk.is_empty(),
                "a k-quant row needs a multiple-of-256 activation"
            );
            let mut acc = 0.0f32;
            for (sb, blk) in w.chunks_exact(84).enumerate() {
                let scales = &blk[0..16];
                let qs = &blk[16..80];
                let d = h(blk, 80);
                let dmin = h(blk, 82);
                let aq = &a.qk[sb * 256..sb * 256 + 256];
                let bs = &a.bsums[sb * 16..sb * 16 + 16];
                let mut sumi = 0i32;
                let mut summ = 0i32;
                let mut is = 0;
                for n in 0..2 {
                    let q = &qs[n * 32..(n + 1) * 32];
                    for j in 0..4 {
                        let shift = j * 2;
                        let sc0 = scales[is];
                        let sc0_val = (sc0 & 0x0F) as i32;
                        let m0_val = (sc0 >> 4) as i32;
                        let act0 = &aq[is * 16..(is + 1) * 16];
                        let mut s0 = 0i32;
                        for l in 0..16 {
                            s0 += ((q[l] >> shift) & 3) as i32 * act0[l] as i32;
                        }
                        sumi += sc0_val * s0;
                        summ += m0_val * bs[is] as i32;
                        is += 1;

                        let sc1 = scales[is];
                        let sc1_val = (sc1 & 0x0F) as i32;
                        let m1_val = (sc1 >> 4) as i32;
                        let act1 = &aq[is * 16..(is + 1) * 16];
                        let mut s1 = 0i32;
                        for l in 0..16 {
                            s1 += ((q[l + 16] >> shift) & 3) as i32 * act1[l] as i32;
                        }
                        sumi += sc1_val * s1;
                        summ += m1_val * bs[is] as i32;
                        is += 1;
                    }
                }
                acc += a.dk[sb] * (d * sumi as f32 - dmin * summ as f32);
            }
            acc
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q3_K => {
            assert!(
                !a.qk.is_empty(),
                "a k-quant row needs a multiple-of-256 activation"
            );
            let mut acc = 0.0f32;
            for (sb, blk) in w.chunks_exact(110).enumerate() {
                let hmask = &blk[0..32];
                let qs = &blk[32..96];
                let scales_raw = &blk[96..108];
                let d = h(blk, 108);

                let a0 = u32::from_le_bytes(scales_raw[0..4].try_into().unwrap());
                let a1 = u32::from_le_bytes(scales_raw[4..8].try_into().unwrap());
                let tmp = u32::from_le_bytes(scales_raw[8..12].try_into().unwrap());
                const KMASK1: u32 = 0x03030303;
                const KMASK2: u32 = 0x0f0f0f0f;
                let s0 = (a0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
                let s1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
                let s2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
                let s3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
                let mut scales = [0i8; 16];
                for (chunk_idx, &word) in [s0, s1, s2, s3].iter().enumerate() {
                    for (i, b) in word.to_le_bytes().into_iter().enumerate() {
                        scales[chunk_idx * 4 + i] = b as i8;
                    }
                }

                let aq = &a.qk[sb * 256..sb * 256 + 256];
                let mut sumi = 0i32;
                let mut is = 0;
                let mut m = 1u8;
                for n in 0..2 {
                    let q = &qs[n * 32..(n + 1) * 32];
                    for j in 0..4 {
                        let shift = j * 2;
                        let sc0 = scales[is] as i32 - 32;
                        let act0 = &aq[is * 16..(is + 1) * 16];
                        let mut s0 = 0i32;
                        for l in 0..16 {
                            let hbit = (hmask[l] & m) != 0;
                            let code = ((q[l] >> shift) & 3) as i32 - if hbit { 0 } else { 4 };
                            s0 += code * act0[l] as i32;
                        }
                        sumi += sc0 * s0;
                        is += 1;

                        let sc1 = scales[is] as i32 - 32;
                        let act1 = &aq[is * 16..(is + 1) * 16];
                        let mut s1 = 0i32;
                        for l in 0..16 {
                            let hbit = (hmask[l + 16] & m) != 0;
                            let code = ((q[l + 16] >> shift) & 3) as i32 - if hbit { 0 } else { 4 };
                            s1 += code * act1[l] as i32;
                        }
                        sumi += sc1 * s1;
                        is += 1;

                        m <<= 1;
                    }
                }
                acc += a.dk[sb] * (d * sumi as f32);
            }
            acc
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q4_K => {
            // Against the per-256 activation (`qk`, `dk`, `bsums`): the eight
            // sub-block dot products are scaled and summed as integers and the
            // super-block's one float scale is applied once.
            assert!(
                !a.qk.is_empty(),
                "a k-quant row needs a multiple-of-256 activation"
            );
            let mut acc = 0.0f32;
            for (sb, blk) in w.chunks_exact(144).enumerate() {
                let d = h(blk, 0);
                let dmin = h(blk, 2);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let aq = &a.qk[sb * 256..sb * 256 + 256];
                let bs = &a.bsums[sb * 16..sb * 16 + 16];
                let mut sumi = 0i32;
                let mut summ = 0i32;
                for is in 0..8 {
                    let (sc, m) = scale_min_k4(is, scales);
                    let base = (is / 2) * 32;
                    let act = &aq[is * 32..is * 32 + 32];
                    let mut s = 0i32;
                    if is % 2 == 0 {
                        for l in 0..32 {
                            s += (qs[base + l] & 0x0F) as i32 * act[l] as i32;
                        }
                    } else {
                        for l in 0..32 {
                            s += (qs[base + l] >> 4) as i32 * act[l] as i32;
                        }
                    }
                    sumi += sc as i32 * s;
                    // w = d·sc·q − dmin·m, so the min contributes −dmin·m·Σa.
                    summ += m as i32 * (bs[2 * is] as i32 + bs[2 * is + 1] as i32);
                }
                acc += a.dk[sb] * (d * sumi as f32 - dmin * summ as f32);
            }
            acc
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q5_K => {
            assert!(
                !a.qk.is_empty(),
                "a k-quant row needs a multiple-of-256 activation"
            );
            let mut acc = 0.0f32;
            for (sb, blk) in w.chunks_exact(176).enumerate() {
                let d = h(blk, 0);
                let dmin = h(blk, 2);
                let scales = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let aq = &a.qk[sb * 256..sb * 256 + 256];
                let bs = &a.bsums[sb * 16..sb * 16 + 16];
                let mut sumi = 0i32;
                let mut summ = 0i32;
                for is in 0..8 {
                    let (sc, m) = scale_min_k4(is, scales);
                    let base = (is / 2) * 32;
                    let act = &aq[is * 32..is * 32 + 32];
                    let mut s = 0i32;
                    let u = 1u8 << is;
                    if is % 2 == 0 {
                        for l in 0..32 {
                            let hi = if (qh[l] & u) != 0 { 16 } else { 0 };
                            s += (((qs[base + l] & 0x0F) | hi) as i32) * act[l] as i32;
                        }
                    } else {
                        for l in 0..32 {
                            let hi = if (qh[l] & u) != 0 { 16 } else { 0 };
                            s += (((qs[base + l] >> 4) | hi) as i32) * act[l] as i32;
                        }
                    }
                    sumi += sc as i32 * s;
                    // w = d·sc·q − dmin·m, so the min contributes −dmin·m·Σa.
                    summ += m as i32 * (bs[2 * is] as i32 + bs[2 * is + 1] as i32);
                }
                acc += a.dk[sb] * (d * sumi as f32 - dmin * summ as f32);
            }
            acc
        }
        #[cfg(feature = "fmt-kquant")]
        Format::Q6_K => {
            assert!(
                !a.qk.is_empty(),
                "a k-quant row needs a multiple-of-256 activation"
            );
            let mut acc = 0.0f32;
            for (sb, blk) in w.chunks_exact(210).enumerate() {
                let ql_all = &blk[0..128];
                let qh_all = &blk[128..192];
                let sc_all: &[i8] =
                    unsafe { std::slice::from_raw_parts(blk[192..208].as_ptr() as *const i8, 16) };
                let d = h(blk, 208);
                let mut sumi = 0i32;
                for c in 0..2 {
                    let ql = &ql_all[c * 64..c * 64 + 64];
                    let qh = &qh_all[c * 32..c * 32 + 32];
                    let sc = &sc_all[c * 8..c * 8 + 8];
                    // Outputs of this 128-chunk start here, in activation terms.
                    let obase = sb * 256 + c * 128;
                    // Four interleaved groups of 32, each split into two 16-runs that
                    // carry different scales — hence sums16 rather than sums.
                    for g in 0..4 {
                        for hf in 0..2 {
                            let scale = sc[2 * g + hf] as i32;
                            let o = obase + g * 32 + hf * 16;
                            let aq = &a.qk[o..o + 16];
                            let mut s = 0i32;
                            for l in 0..16 {
                                let li = hf * 16 + l;
                                let code = match g {
                                    0 => (ql[li] & 0x0F) | (((qh[li] >> 0) & 3) << 4),
                                    1 => (ql[li + 32] & 0x0F) | (((qh[li] >> 2) & 3) << 4),
                                    2 => (ql[li] >> 4) | (((qh[li] >> 4) & 3) << 4),
                                    _ => (ql[li + 32] >> 4) | (((qh[li] >> 6) & 3) << 4),
                                };
                                s += code as i32 * aq[l] as i32;
                            }
                            // Offset 32, applied once per 16-run via bsums.
                            let bias = 32 * a.bsums[o / 16] as i32;
                            sumi += scale * (s - bias);
                        }
                    }
                }
                acc += d * a.dk[sb] * sumi as f32;
            }
            acc
        }
        #[cfg(feature = "fmt-iq")]
        Format::IQ4_NL => {
            let mut acc = 0.0f32;
            for (b, blk) in w.chunks_exact(18).enumerate() {
                let qs = &blk[2..18];
                let aq = &a.q[b * 32..b * 32 + 32];
                let mut s = 0i32;
                for j in 0..16 {
                    s += KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as i32 * aq[j] as i32;
                    s += KVALUES_IQ4NL[(qs[j] >> 4) as usize] as i32 * aq[j + 16] as i32;
                }
                acc += h(blk, 0) * a.d[b] * s as f32;
            }
            acc
        }
        #[cfg(feature = "fmt-iq")]
        Format::IQ4_XS => {
            assert!(
                !a.qk.is_empty(),
                "a k-quant row needs a multiple-of-256 activation"
            );
            let mut acc = 0.0f32;
            for (sb, blk) in w.chunks_exact(136).enumerate() {
                let d = h(blk, 0);
                let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
                let scales_l = &blk[4..8];
                let qs = &blk[8..136];
                let aq = &a.qk[sb * 256..sb * 256 + 256];
                let mut sumi = 0i32;
                for ib in 0..8 {
                    let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as i32
                        | ((((scales_h >> (2 * ib)) & 3) << 4) as i32);
                    let scale = ls - 32;
                    let sub_qs = &qs[ib * 16..(ib + 1) * 16];
                    let sub_act = &aq[ib * 32..(ib + 1) * 32];
                    let mut s = 0i32;
                    for j in 0..16 {
                        s += KVALUES_IQ4NL[(sub_qs[j] & 0x0F) as usize] as i32 * sub_act[j] as i32;
                        s += KVALUES_IQ4NL[(sub_qs[j] >> 4) as usize] as i32
                            * sub_act[j + 16] as i32;
                    }
                    sumi += scale * s;
                }
                acc += a.dk[sb] * (d * sumi as f32);
            }
            acc
        }
        #[cfg(feature = "fmt-mxfp4")]
        Format::MXFP4 => {
            let mut acc = 0.0f32;
            for (b, blk) in w.chunks_exact(17).enumerate() {
                let qs = &blk[1..17];
                let aq = &a.q[b * 32..b * 32 + 32];
                let mut s = 0i32;
                for j in 0..16 {
                    s += KVALUES_MXFP4[(qs[j] & 0x0F) as usize] as i32 * aq[j] as i32;
                    s += KVALUES_MXFP4[(qs[j] >> 4) as usize] as i32 * aq[j + 16] as i32;
                }
                acc += crate::format::e8m0_to_f32_half(blk[0]) * a.d[b] * s as f32;
            }
            acc
        }
        #[cfg(feature = "fmt-ternary")]
        Format::PQ2_0 => {
            let mut acc = 0.0f32;
            for (g, blk) in w.chunks_exact(34).enumerate() {
                let mut gs = 0.0f32;
                for u in 0..8 {
                    let mut s = 0i32;
                    for j in 16 * u..16 * u + 16 {
                        let c = ((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) as i32;
                        s += c * a.tq[g * 128 + j] as i32;
                    }
                    gs += a.td[8 * g + u] * (s - a.tls[8 * g + u]) as f32;
                }
                acc += h(blk, 0) * gs;
            }
            acc
        }
        #[cfg(feature = "fmt-ternary")]
        Format::PTQ1_0 if a.ptq1 => {
            let mut acc = 0.0f32;
            let mut t = [0i8; 128];
            for (g, blk) in w.chunks_exact(28).enumerate() {
                ptq1_trits(blk, &mut t);
                let (p, gi) = (g / 2, g % 2);
                let mut s = 0i64;
                for r in 0..8 {
                    for m in 0..16 {
                        let qv = a.t1a[p * 256 + r * 32 + gi * 16 + m] as i64;
                        s += (t[16 * r + m] as i64 + 1) * qv;
                    }
                }
                s += a.t1init[p * 16 + gi * 8] as i64;
                acc += h(blk, 26) * a.t1sc[p * 16 + gi * 8] * s as f32;
            }
            acc
        }
        #[cfg(feature = "fmt-ternary")]
        Format::PTQ1_0 => {
            let mut acc = 0.0f32;
            let mut t = [0i8; 128];
            for (g, blk) in w.chunks_exact(28).enumerate() {
                ptq1_trits(blk, &mut t);
                let mut gs = 0.0f32;
                for u in 0..8 {
                    let s: i32 = (16 * u..16 * u + 16)
                        .map(|j| t[j] as i32 * a.tq[g * 128 + j] as i32)
                        .sum();
                    gs += a.td[8 * g + u] * s as f32;
                }
                acc += h(blk, 26) * gs;
            }
            acc
        }
        Format::F32 | Format::F16 | Format::BF16 => {
            assert!(
                a.keep_f32,
                "float matrix matmul reached with keep_f32 off on QuantAct"
            );
            assert_eq!(
                a.xf.len(),
                a.n,
                "activation f32 vector xf must hold a.n values"
            );
            dot_f32_ref(fmt, w, &a.xf)
        }
        #[allow(unreachable_patterns)]
        _ => panic!("format {} disabled in this build", fmt.label()),
    }
}

/// Horizontal sum of an 8-lane f32 accumulator in the fixed tree reduction order of `simd::hsum_ps`.
///
/// Accumulation order:
/// 1. `s0 = acc[0] + acc[4]`, `s1 = acc[1] + acc[5]`, `s2 = acc[2] + acc[6]`, `s3 = acc[3] + acc[7]`
/// 2. `s02 = s0 + s2`, `s13 = s1 + s3`
/// 3. `result = s02 + s13`
#[inline(always)]
pub fn hsum_8(acc: [f32; 8]) -> f32 {
    let s0 = acc[0] + acc[4];
    let s1 = acc[1] + acc[5];
    let s2 = acc[2] + acc[6];
    let s3 = acc[3] + acc[7];
    let s02 = s0 + s2;
    let s13 = s1 + s3;
    s02 + s13
}

/// Reference scalar dot product for float weight formats (F32, F16, BF16) against f32 activations.
///
/// # Accumulation Order
/// Defined to be bit-for-bit identical to the AVX2 kernel across all tiers:
/// 1. Accumulates into 8 independent f32 partial sums (`acc[i % 8] += w_i * x_i`),
///    using standard uncontracted IEEE-754 single-precision multiplication and addition
///    (without FMA contraction, matching `_mm256_mul_ps` followed by `_mm256_add_ps`).
/// 2. If row length `n` is not a multiple of 8, remaining elements `n % 8` are accumulated
///    into lanes `0 .. n % 8`.
/// 3. The 8 accumulator lanes are horizontally reduced using [`hsum_8`].
pub fn dot_f32_ref(fmt: Format, w: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    debug_assert_eq!(w.len(), row_bytes(fmt, n));
    let mut acc = [0.0f32; 8];
    match fmt {
        Format::F32 => {
            for (i, &xv) in x.iter().enumerate() {
                let wv = f32::from_le_bytes(w[i * 4..i * 4 + 4].try_into().unwrap());
                acc[i % 8] += wv * xv;
            }
        }
        Format::F16 => {
            for (i, &xv) in x.iter().enumerate() {
                let h = u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]);
                let wv = f16(h);
                acc[i % 8] += wv * xv;
            }
        }
        Format::BF16 => {
            for (i, &xv) in x.iter().enumerate() {
                let bits = (u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]) as u32) << 16;
                let wv = f32::from_bits(bits);
                acc[i % 8] += wv * xv;
            }
        }
        _ => panic!("dot_f32_ref called on non-float format {:?}", fmt),
    }
    hsum_8(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion_round_trips_and_rounds_to_even() {
        for h in 0..=u16::MAX {
            let f = f16(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "{h:#06x} ({f})");
        }
        // Halfway between 1.0 and the next half (1 + 2^-10): ties go to even (1.0).
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
    }

    /// The oracle: a row that is already exactly representable in Q4_K — one decoded
    /// from a real Q4_K block — must quantise back to essentially the same values.
    /// And a smooth random row must come back with a small error.
    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn q4_k_quantisation_matches_ggml_on_representable_rows() {
        let mut s = 0x9E37_79B9u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let x: Vec<f32> = (0..1024)
            .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2)
            .collect();
        let mut q = Vec::new();
        quantize_row_q4_k(&x, &mut q);
        assert_eq!(q.len(), row_bytes(Format::Q4_K, 1024));
        let mut y = vec![0.0f32; 1024];
        dequant_row(Format::Q4_K, &q, &mut y);
        let rmse = (x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / 1024.0)
            .sqrt();
        let rms = (x.iter().map(|a| a * a).sum::<f32>() / 1024.0).sqrt();
        assert!(rmse / rms < 0.12, "relative RMSE {}", rmse / rms);
        // Re-quantising the decoded row is (near) idempotent.
        let mut q2 = Vec::new();
        quantize_row_q4_k(&y, &mut q2);
        let mut z = vec![0.0f32; 1024];
        dequant_row(Format::Q4_K, &q2, &mut z);
        let drift = y
            .iter()
            .zip(&z)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(drift < 2e-3, "re-quantising moved a value by {drift}");
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn q8_0_quantisation_round_trips_within_half_a_step() {
        let x: Vec<f32> = (0..256)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect();
        let mut q = Vec::new();
        quantize_row_q8_0(&x, &mut q);
        assert_eq!(q.len(), row_bytes(Format::Q8_0, 256));
        let mut y = vec![0.0f32; 256];
        dequant_row(Format::Q8_0, &q, &mut y);
        for (blk_x, blk_y) in x.chunks(32).zip(y.chunks(32)) {
            let step = blk_x.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
            for (a, b) in blk_x.iter().zip(blk_y) {
                // Half a step from rounding the code, plus up to 127·d·2⁻¹¹ because
                // the stored scale is rounded to f16 while the codes used the exact one.
                assert!(
                    (a - b).abs() <= step * (0.5 + 127.0 / 2048.0) + 1e-6,
                    "{a} vs {b}"
                );
            }
        }
    }

    /// A deterministic pseudo-random byte stream. Real weight bytes are exercised by
    /// `tests/oracle.rs`; this is for the structural properties.
    fn prng(seed: u64, n: usize) -> Vec<u8> {
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

    /// Force every f16 block scale in a randomly generated row to a finite, sane
    /// value. Raw random bytes hit the half-precision Inf/NaN encodings about one
    /// block in 128, and a NaN scale propagates through both the kernel and the
    /// reference, so the comparison would pass vacuously (NaN == NaN is never
    /// asserted) or fail for a reason that has nothing to do with the kernel.
    /// Clearing the top exponent bits leaves a normal half in [0.125, 0.25).
    #[cfg_attr(
        not(all(
            feature = "fmt-kquant",
            feature = "fmt-ternary",
            feature = "fmt-iq",
            feature = "fmt-mxfp4",
            feature = "fmt-legacy"
        )),
        allow(unused_variables, dead_code)
    )]
    fn sane_scales(fmt: Format, w: &mut [u8]) {
        let fix = |b: &mut [u8], at: usize| b[at + 1] = (b[at + 1] & 0x0F) | 0x30;
        let sz = fmt.block_bytes();
        for blk in w.chunks_exact_mut(sz) {
            match fmt {
                #[cfg(feature = "fmt-legacy")]
                Format::Q4_0 | Format::Q8_0 | Format::Q5_0 => fix(blk, 0),
                #[cfg(feature = "fmt-iq")]
                Format::IQ4_NL => fix(blk, 0),
                #[cfg(feature = "fmt-kquant")]
                Format::Q2_K => {
                    fix(blk, 80);
                    fix(blk, 82);
                }
                #[cfg(feature = "fmt-kquant")]
                Format::Q3_K => fix(blk, 108),
                #[cfg(feature = "fmt-kquant")]
                Format::Q4_K | Format::Q5_K => {
                    fix(blk, 0);
                    fix(blk, 2);
                }
                #[cfg(feature = "fmt-kquant")]
                Format::Q6_K => fix(blk, 208),
                #[cfg(feature = "fmt-iq")]
                Format::IQ4_XS => fix(blk, 0),
                #[cfg(feature = "fmt-mxfp4")]
                Format::MXFP4 => blk[0] = (blk[0] % 30) + 115,
                #[cfg(feature = "fmt-ternary")]
                Format::PQ2_0 => fix(blk, 0),
                #[cfg(feature = "fmt-ternary")]
                Format::PTQ1_0 => fix(blk, 26),
                _ => {}
            }
        }
    }

    fn acts(n: usize, seed: u64) -> (Vec<f32>, QuantAct) {
        let raw = prng(seed, n * 2);
        let x: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| (u16::from_le_bytes(c.try_into().unwrap()) as f32 / 32768.0) - 1.0)
            .collect();
        let mut q = QuantAct::with_capacity(n);
        q.fill(&x);
        (x, q)
    }

    /// The central invariant of this file: the fused kernel must equal the reference
    /// decode followed by a plain dot product. If these disagree, the fusion is
    /// wrong — which is a class of bug that still produces fluent text.
    #[allow(clippy::needless_range_loop)]
    #[cfg_attr(
        not(all(
            feature = "fmt-kquant",
            feature = "fmt-ternary",
            feature = "fmt-iq",
            feature = "fmt-mxfp4",
            feature = "fmt-legacy"
        )),
        allow(dead_code)
    )]
    fn fused_equals_reference(fmt: Format, n: usize) {
        let mut w = prng(0xA5A5_1234, row_bytes(fmt, n));
        sane_scales(fmt, &mut w);
        let (_, a) = acts(n, 0x1357_9BDF);

        let mut deq = vec![0.0f32; n];
        dequant_row(fmt, &w, &mut deq);
        // Reference: dot of the decoded weights against the *reconstructed*
        // activation, which is what the fused kernel actually computes — the
        // per-256 representation for the k-quants, the per-32 one otherwise.
        let kq = match fmt {
            #[cfg(feature = "fmt-kquant")]
            Format::Q2_K | Format::Q3_K | Format::Q4_K | Format::Q5_K | Format::Q6_K => true,
            #[cfg(feature = "fmt-iq")]
            Format::IQ4_XS => true,
            _ => false,
        };
        let mut want = 0.0f64;
        for i in 0..n {
            let act = if kq {
                a.qk[i] as f32 * a.dk[i / 256]
            } else {
                a.q[i] as f32 * a.d[i / 32]
            };
            want += (deq[i] * act) as f64;
        }
        let got = dot_scalar(fmt, &w, &a) as f64;
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() / scale < 1e-5,
            "{}: fused {} vs reference {} (rel {:.2e})",
            fmt.label(),
            got,
            want,
            (got - want).abs() / scale
        );
    }

    /// The per-256 activation is a coarser quantisation of the same vector: it
    /// must reconstruct within its own step and carry exact run sums.
    #[test]
    fn per_256_activation_reconstructs_and_sums_exactly() {
        let (x, a) = acts(512, 0x2468_ACE0);
        assert_eq!(a.qk.len(), 512);
        assert_eq!(a.dk.len(), 2);
        assert_eq!(a.bsums.len(), 32);
        for (i, &v) in x.iter().enumerate() {
            let r = a.qk[i] as f32 * a.dk[i / 256];
            assert!(
                (r - v).abs() <= a.dk[i / 256] * 0.5 + 1e-6,
                "{i}: {r} vs {v}"
            );
        }
        for run in 0..32 {
            let s: i32 = a.qk[run * 16..run * 16 + 16]
                .iter()
                .map(|&q| q as i32)
                .sum();
            assert_eq!(s, a.bsums[run] as i32);
        }
        let (_, b) = acts(96, 1);
        assert!(
            b.qk.is_empty() && b.dk.is_empty(),
            "not a multiple of 256: no per-256 form"
        );
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn fused_dot_matches_reference_decode_q8_0() {
        fused_equals_reference(Format::Q8_0, 960);
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn fused_dot_matches_reference_decode_q4_0() {
        fused_equals_reference(Format::Q4_0, 960);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_fill_matches_the_scalar_fill() {
        if !crate::cpu::have_neon() {
            eprintln!("skip: no NEON on this CPU");
            return;
        }
        let mut s = 0x0bad_c0de_2468_1357u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        };
        for n in [128usize, 256, 896, 1024, 4864] {
            let x: Vec<f32> = (0..n)
                .map(|_| (next() as f32 / 255.0) * 6.0 - 3.0)
                .collect();
            let mut want = QuantAct::with_capacity(n);
            want.fill_scalar(&x);
            let mut got = QuantAct::with_capacity(n);
            // SAFETY: NEON checked above.
            unsafe { got.fill_neon(&x) };
            assert_eq!(got.n, want.n, "n={n}");
            assert_eq!(got.q, want.q, "n={n}: codes");
            assert_eq!(got.qk, want.qk, "n={n}: 256-block codes");
            assert_eq!(got.d, want.d, "n={n}: scales");
            assert_eq!(got.dk, want.dk, "n={n}: 256-block scales");
            assert_eq!(got.sums, want.sums, "n={n}: sums");
            assert_eq!(got.sums16, want.sums16, "n={n}: 16-sums");
            assert_eq!(got.lane_sums, want.lane_sums, "n={n}: lane sums");
            assert_eq!(got.bsums, want.bsums, "n={n}: run sums");
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn fused_dot_matches_reference_decode_q5_0() {
        fused_equals_reference(Format::Q5_0, 960);
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn fused_dot_matches_reference_decode_q2_k() {
        fused_equals_reference(Format::Q2_K, 2560);
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn fused_dot_matches_reference_decode_q3_k() {
        fused_equals_reference(Format::Q3_K, 2560);
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn fused_dot_matches_reference_decode_q4_k() {
        fused_equals_reference(Format::Q4_K, 2560);
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn fused_dot_matches_reference_decode_q5_k() {
        fused_equals_reference(Format::Q5_K, 2560);
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn fused_dot_matches_reference_decode_q6_k() {
        fused_equals_reference(Format::Q6_K, 2560);
    }

    #[test]
    #[cfg(feature = "fmt-iq")]
    fn fused_dot_matches_reference_decode_iq4_nl() {
        fused_equals_reference(Format::IQ4_NL, 960);
    }

    #[test]
    #[cfg(feature = "fmt-iq")]
    fn fused_dot_matches_reference_decode_iq4_xs() {
        fused_equals_reference(Format::IQ4_XS, 2560);
    }

    #[test]
    #[cfg(feature = "fmt-mxfp4")]
    fn fused_dot_matches_reference_decode_mxfp4() {
        fused_equals_reference(Format::MXFP4, 960);
    }

    #[test]
    #[cfg(feature = "fmt-mxfp4")]
    fn dequant_row_mxfp4_matches_hand_computed_block() {
        let mut blk = vec![0u8; 17];
        // e = 129 (0x81) -> e8m0_to_f32_half(129) = 2.0
        blk[0] = 129;
        // qs[0]: low nibble = 0 (KVALUES_MXFP4[0] = 0), high nibble = 1 (KVALUES_MXFP4[1] = 1)
        blk[1] = 0x10;
        // qs[1]: low nibble = 7 (KVALUES_MXFP4[7] = 12), high nibble = 9 (KVALUES_MXFP4[9] = -1)
        blk[2] = 0x97;
        // qs[2]: low nibble = 14 (KVALUES_MXFP4[14] = -8), high nibble = 15 (KVALUES_MXFP4[15] = -12)
        blk[3] = 0xFE;
        // qs[3]: low nibble = 4 (KVALUES_MXFP4[4] = 4), high nibble = 5 (KVALUES_MXFP4[5] = 6)
        blk[4] = 0x54;

        let mut out = vec![0.0f32; 32];
        dequant_row(Format::MXFP4, &blk, &mut out);

        assert_eq!(out[0], 0.0);
        assert_eq!(out[16], 2.0);
        assert_eq!(out[1], 24.0);
        assert_eq!(out[17], -2.0);
        assert_eq!(out[2], -16.0);
        assert_eq!(out[18], -24.0);
        assert_eq!(out[3], 8.0);
        assert_eq!(out[19], 12.0);

        // Test subnormal exponent edge values:
        // e = 0 -> e8m0_to_f32_half(0) = 2^(-128)
        blk[0] = 0;
        dequant_row(Format::MXFP4, &blk, &mut out);
        assert_eq!(out[16], f32::from_bits(0x0020_0000));

        // e = 1 -> e8m0_to_f32_half(1) = 2^(-127)
        blk[0] = 1;
        dequant_row(Format::MXFP4, &blk, &mut out);
        assert_eq!(out[16], f32::from_bits(0x0040_0000));
    }

    #[test]
    #[cfg(feature = "fmt-iq")]
    fn dequant_row_iq4_nl_matches_hand_computed_block() {
        let mut blk = vec![0u8; 18];
        // d = 2.0 (0x4000)
        blk[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
        // qs[0]: low nibble = 0 (KVALUES_IQ4NL[0] = -127), high nibble = 8 (KVALUES_IQ4NL[8] = 1)
        blk[2] = 0x80;
        // qs[1]: low nibble = 15 (KVALUES_IQ4NL[15] = 113), high nibble = 7 (KVALUES_IQ4NL[7] = -10)
        blk[3] = 0x7F;
        // qs[2]: low nibble = 8 (KVALUES_IQ4NL[8] = 1), high nibble = 9 (KVALUES_IQ4NL[9] = 13)
        blk[4] = 0x98;

        let mut out = vec![0.0f32; 32];
        dequant_row(Format::IQ4_NL, &blk, &mut out);

        assert_eq!(out[0], -254.0);
        assert_eq!(out[16], 2.0);
        assert_eq!(out[1], 226.0);
        assert_eq!(out[17], -20.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[18], 26.0);
    }

    #[test]
    #[cfg(feature = "fmt-iq")]
    fn dequant_row_iq4_xs_matches_hand_computed_block() {
        let mut blk = vec![0u8; 136];
        // d = 0.5 (0x3800)
        blk[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        // Sub-block 0 (ib=0): ls0 = 34 -> ls0 - 32 = 2. scales_l[0] & 0xF = 2, (scales_h >> 0) & 3 = 2
        // Sub-block 1 (ib=1): ls1 = 31 -> ls1 - 32 = -1. (scales_l[0] >> 4) & 0xF = 0xF, (scales_h >> 2) & 3 = 1
        // scales_l[0] = 0xF2
        // Sub-block 4 (ib=4): ls4 = 36 -> ls4 - 32 = 4. scales_l[2] & 0xF = 4, (scales_h >> 8) & 3 = 2
        let scales_h: u16 = 2 | (1 << 2) | (2 << 8);
        blk[2..4].copy_from_slice(&scales_h.to_le_bytes());
        blk[4] = 0xF2;
        blk[6] = 0x04;

        // qs at 8..136
        // Sub-block 0: qs[0] (weights 0 and 16): low = 0 (-127), high = 8 (1)
        // dl = 0.5 * 2 = 1.0 -> out[0] = -127.0, out[16] = 1.0
        blk[8] = 0x80;
        // Sub-block 1: qs[16] (weights 32 and 48): low = 15 (113), high = 9 (13)
        // dl = 0.5 * (-1) = -0.5 -> out[32] = -56.5, out[48] = -6.5
        blk[8 + 16] = 0x9F;
        // Sub-block 4: qs[64] (weights 128 and 144): low = 8 (1), high = 0 (-127)
        // dl = 0.5 * 4 = 2.0 -> out[128] = 2.0, out[144] = -254.0
        blk[8 + 64] = 0x08;

        let mut out = vec![0.0f32; 256];
        dequant_row(Format::IQ4_XS, &blk, &mut out);

        assert_eq!(out[0], -127.0);
        assert_eq!(out[16], 1.0);
        assert_eq!(out[32], -56.5);
        assert_eq!(out[48], -6.5);
        assert_eq!(out[128], 2.0);
        assert_eq!(out[144], -254.0);
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn dequant_row_q2_k_matches_hand_computed_block() {
        let mut blk = vec![0u8; 84];
        // d = 1.0 (0x3C00) at 80..82, dmin = 0.5 (0x3800) at 82..84
        blk[80..82].copy_from_slice(&0x3C00u16.to_le_bytes());
        blk[82..84].copy_from_slice(&0x3800u16.to_le_bytes());

        // scales[0..16]:
        // subblock 0 (is=0): sc = 4, m = 2 -> 0x24
        // subblock 1 (is=1): sc = 3, m = 1 -> 0x13
        // subblock 2 (is=2): sc = 5, m = 4 -> 0x45
        // subblock 8 (is=8): sc = 2, m = 3 -> 0x32
        blk[0] = 0x24;
        blk[1] = 0x13;
        blk[2] = 0x45;
        blk[8] = 0x32;

        // qs: 64 bytes at 16..80
        // qs[0] (is=0, shift=0, and is=2, shift=2):
        // weights 0 and 32: set qs[0] = 0b00_01_01_11 (low 2 bits = 3, bits 2..3 = 1) -> 0x17
        blk[16] = 0x17;
        // qs[1] (weight 1): low 2 bits = 0
        blk[17] = 0x00;
        // qs[16] (is=1, shift=0, weight 16): low 2 bits = 2
        blk[16 + 16] = 0x02;
        // qs[32] (is=8, shift=0, weight 128): low 2 bits = 3
        blk[16 + 32] = 0x03;

        let mut out = vec![0.0f32; 256];
        dequant_row(Format::Q2_K, &blk, &mut out);

        // is=0: dl = 1.0 * 4 = 4.0, ml = 0.5 * 2 = 1.0
        // weight 0: 4.0 * 3 - 1.0 = 11.0
        assert_eq!(out[0], 11.0);
        // weight 1: 4.0 * 0 - 1.0 = -1.0
        assert_eq!(out[1], -1.0);
        // is=1: dl = 1.0 * 3 = 3.0, ml = 0.5 * 1 = 0.5
        // weight 16: 3.0 * 2 - 0.5 = 5.5
        assert_eq!(out[16], 5.5);
        // is=2: dl = 1.0 * 5 = 5.0, ml = 0.5 * 4 = 2.0
        // weight 32: 5.0 * 1 - 2.0 = 3.0
        assert_eq!(out[32], 3.0);
        // is=8: dl = 1.0 * 2 = 2.0, ml = 0.5 * 3 = 1.5
        // weight 128: 2.0 * 3 - 1.5 = 4.5
        assert_eq!(out[128], 4.5);
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn dequant_row_q3_k_matches_hand_computed_block() {
        let mut blk = vec![0u8; 110];
        // d = 1.0 (0x3C00) at 108..110
        blk[108..110].copy_from_slice(&0x3C00u16.to_le_bytes());

        // scales: 12 bytes at 96..108.
        // a0 at 96..100, a1 at 100..104, a2 at 104..108
        // is=0: (0x32 & 0x0F) | (((0x22 >> 0) & 3) << 4) = 2 | (2 << 4) = 34 -> scale - 32 = +2
        // is=1: (0x1F & 0x0F) | (((0x22 >> 2) & 3) << 4) = 15 | (0 << 4) = 15 -> scale - 32 = -17
        // is=2: (0x04 & 0x0F) | (((0x02 >> 0) & 3) << 4) = 4 | (2 << 4) = 36 -> scale - 32 = +4
        // is=8: ((0x32 >> 4) & 0x0F) | (((0x22 >> 4) & 3) << 4) = 3 | (2 << 4) = 35 -> scale - 32 = +3
        blk[96] = 0x32;
        blk[97] = 0x1F;
        blk[98] = 0x04;
        blk[104] = 0x22;
        blk[106] = 0x02;

        // hmask: 32 bytes at 0..32
        // For is=0 (m=1): weight 0 has hbit 1 (hmask[0] & 1 != 0), weight 1 has hbit 0
        blk[0] = 0x01 | 0x10; // bit 0 for is=0 (weight 0), bit 4 for is=8 (weight 128)
                              // For is=1 (m=1): weight 16 has hbit 1 (hmask[16] & 1 != 0)
        blk[16] = 0x01;

        // qs: 64 bytes at 32..96
        // qs[0] (is=0, shift=0, and is=2, shift=2):
        // weight 0: low 2 bits = 3. weight 32: bits 2..3 = 2 -> 0b0000_1011 = 0x0B
        blk[32] = 0x0B;
        // qs[1] (weight 1): low 2 bits = 2
        blk[33] = 0x02;
        // qs[16] (weight 16): low 2 bits = 1
        blk[32 + 16] = 0x01;
        // qs[32] (weight 128): low 2 bits = 1
        blk[32 + 32] = 0x01;

        let mut out = vec![0.0f32; 256];
        dequant_row(Format::Q3_K, &blk, &mut out);

        // is=0: dl = 1.0 * (34 - 32) = 2.0.
        // weight 0: q=3, hbit=1 -> code = 3 - 0 = 3 -> 2.0 * 3 = 6.0
        assert_eq!(out[0], 6.0);
        // weight 1: q=2, hbit=0 -> code = 2 - 4 = -2 -> 2.0 * (-2) = -4.0
        assert_eq!(out[1], -4.0);
        // is=1: dl = 1.0 * (15 - 32) = -17.0.
        // weight 16: q=1, hbit=1 -> code = 1 - 0 = 1 -> -17.0 * 1 = -17.0
        assert_eq!(out[16], -17.0);
        // is=2: dl = 1.0 * (36 - 32) = 4.0.
        // weight 32: q=2, m=2 -> hmask[0] & 2 = 0 -> hbit=0 -> code = 2 - 4 = -2 -> 4.0 * (-2) = -8.0
        assert_eq!(out[32], -8.0);
        // is=8: dl = 1.0 * (35 - 32) = 3.0.
        // weight 128: q=1, m=16 -> hmask[0] & 16 != 0 -> hbit=1 -> code = 1 - 0 = 1 -> 3.0 * 1 = 3.0
        assert_eq!(out[128], 3.0);
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn dequant_row_q5_k_matches_hand_computed_block() {
        let mut blk = vec![0u8; 176];
        // d = 2.0 (0x4000), dmin = 0.5 (0x3800)
        blk[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        // Sub-block 0: sc = 3, m = 1 -> scales[0] = 3, scales[4] = 1
        // Sub-block 1: sc = 4, m = 2 -> scales[1] = 4, scales[5] = 2
        // Sub-block 4 (is = 4, j = 4):
        // For j >= 4:
        // scale_min_k4(4): sc = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
        //                  m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
        // Set scales[8] = 0x56: sc = 6, m = 5
        blk[4] = 3;
        blk[5] = 4;
        blk[8] = 1;
        blk[9] = 2;
        blk[12] = 0x56;

        // qh: 32 bytes at 16..48
        // For weight 0:
        // bit 0 of qh[0] is for is=0
        // bit 1 of qh[0] is for is=1
        // bit 4 of qh[0] is for is=4
        // Set qh[0] = 0x11 (bits 0 and 4 set, bit 1 clear)
        blk[16] = 0x11;
        // For weight 1:
        // bit 0 clear, bit 1 set -> qh[1] = 0x02
        blk[17] = 0x02;

        // qs: 128 bytes at 48..176
        // Chunk 0 (j=0) covers sub-block 0 (low nibbles) and 1 (high nibbles), weights 0..63:
        // qs[0]: low nibble = 5, high nibble = 9 -> qs[0] = 0x95
        // qs[1]: low nibble = 12, high nibble = 1 -> qs[1] = 0x1C
        blk[48] = 0x95;
        blk[49] = 0x1C;
        // Chunk 2 (j=2) covers sub-block 4 (low nibbles) and 5 (high nibbles), weights 128..191:
        // qs[64]: low nibble = 7, high nibble = 0 -> qs[64] = 0x07
        blk[48 + 64] = 0x07;

        let mut out = vec![0.0f32; 256];
        dequant_row(Format::Q5_K, &blk, &mut out);

        // Sub-block 0: is = 0: d1 = 2.0 * 3 = 6.0, m1 = 0.5 * 1 = 0.5
        // Weight 0: q = low nibble(5) | (qh[0] & 1 ? 16 : 0) = 5 | 16 = 21
        // Value = 6.0 * 21 - 0.5 = 125.5
        assert_eq!(out[0], 125.5);
        // Weight 1: q = low nibble(12) | (qh[1] & 1 ? 16 : 0) = 12 | 0 = 12
        // Value = 6.0 * 12 - 0.5 = 71.5
        assert_eq!(out[1], 71.5);

        // Sub-block 1: is = 1: d2 = 2.0 * 4 = 8.0, m2 = 0.5 * 2 = 1.0
        // Weight 32 (l=0): q = high nibble(9) | (qh[0] & 2 ? 16 : 0) = 9 | 0 = 9
        // Value = 8.0 * 9 - 1.0 = 71.0
        assert_eq!(out[32], 71.0);
        // Weight 33 (l=1): q = high nibble(1) | (qh[1] & 2 ? 16 : 0) = 1 | 16 = 17
        // Value = 8.0 * 17 - 1.0 = 135.0
        assert_eq!(out[33], 135.0);

        // Sub-block 4: is = 4: sc = 6, m = 5, d1 = 2.0 * 6 = 12.0, m1 = 0.5 * 5 = 2.5
        // Weight 128 (l=0): q = low nibble(7) | (qh[0] & 16 ? 16 : 0) = 7 | 16 = 23
        // Value = 12.0 * 23 - 2.5 = 273.5
        assert_eq!(out[128], 273.5);

        // Weight 2: all zero codes and bits, but has min:
        // q = 0, value = 6.0 * 0 - 0.5 = -0.5
        assert_eq!(out[2], -0.5);
    }

    #[test]
    #[cfg(feature = "fmt-kquant")]
    fn scale_min_k4_unpacks_six_bit_fields() {
        // Construct a byte pattern where every field is distinguishable, then check
        // the two branches agree with the packing rule in ggml-quants.c.
        let q: Vec<u8> = (0..12).map(|i| (i * 17 + 3) as u8).collect();
        for j in 0..4 {
            let (sc, m) = scale_min_k4(j, &q);
            assert_eq!(sc, q[j] & 63);
            assert_eq!(m, q[j + 4] & 63);
        }
        for j in 4..8 {
            let (sc, m) = scale_min_k4(j, &q);
            assert_eq!(sc, (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            assert_eq!(m, (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
            // Six-bit fields: nothing may leak into the top two bits.
            assert!(sc < 64 && m < 64, "j={j} sc={sc} m={m}");
        }
    }

    /// Every half pattern widens to the same bit pattern the vector path gives, and
    /// every row length the encoders use takes the same route.
    #[test]
    fn every_f16_row_is_the_scalar_widening() {
        // The entire half space, as one row of 65 536.
        let mut w = Vec::with_capacity(65536 * 2);
        for h in 0u32..65536 {
            w.extend_from_slice(&(h as u16).to_le_bytes());
        }
        let mut fast = vec![0.0f32; 65536];
        let mut slow = vec![0.0f32; 65536];
        dequant_f16(&w, &mut fast);
        dequant_f16_scalar(&w, &mut slow);
        for h in 0..65536usize {
            let (a, b) = (fast[h], slow[h]);
            assert!(
                (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                "half {h:#06x}: {a:?} != {b:?}"
            );
        }
        // Every length from 0 to 40, and the encoder widths and their neighbours.
        let mut lens: Vec<usize> = (0..=40).collect();
        lens.extend([127, 128, 129, 383, 384, 385, 511, 512, 513, 1536]);
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for n in lens {
            let mut w = Vec::with_capacity(n * 2);
            for _ in 0..n {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                w.extend_from_slice(&((state >> 48) as u16).to_le_bytes());
            }
            let mut fast = vec![0.0f32; n];
            let mut slow = vec![0.0f32; n];
            dequant_row(Format::F16, &w, &mut fast);
            dequant_f16_scalar(&w, &mut slow);
            assert_eq!(
                fast.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                slow.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "n {n}"
            );
        }
    }

    #[test]
    fn activation_quantisation_round_trips_within_one_step() {
        let (x, a) = acts(256, 0xFEED);
        for (i, &v) in x.iter().enumerate() {
            let back = a.q[i] as f32 * a.d[i / 32];
            // One quantisation step is amax/127; reconstruction must be within half.
            let step = a.d[i / 32];
            assert!((back - v).abs() <= step * 0.5 + 1e-9, "i={i} {v} -> {back}");
        }
        // And the cached sums must match what the codes actually are.
        for b in 0..a.n / 32 {
            let s: i32 = a.q[b * 32..b * 32 + 32].iter().map(|&q| q as i32).sum();
            assert_eq!(s, a.sums[b], "block {b} sum cache is stale");
            assert_eq!(a.sums16[2 * b] + a.sums16[2 * b + 1], a.sums[b]);
            assert_eq!(a.lane_sums[b * 8..b * 8 + 8].iter().sum::<i32>(), a.sums[b]);
            let lo: i32 = a.lane_sums[b * 8..b * 8 + 4].iter().sum();
            assert_eq!(lo, a.sums16[2 * b], "lanes 0..4 are the first 16 codes");
        }
    }

    #[test]
    #[cfg(feature = "fmt-legacy")]
    fn an_all_zero_activation_block_does_not_divide_by_zero() {
        let mut a = QuantAct::with_capacity(32);
        a.fill(&[0.0f32; 32]);
        assert_eq!(a.d[0], 0.0);
        assert!(a.q.iter().all(|&q| q == 0));
        let mut w = prng(7, row_bytes(Format::Q5_0, 32));
        sane_scales(Format::Q5_0, &mut w);
        assert_eq!(dot_scalar(Format::Q5_0, &w, &a), 0.0);
        let mut w4 = prng(7, row_bytes(Format::Q4_0, 32));
        sane_scales(Format::Q4_0, &mut w4);
        assert_eq!(dot_scalar(Format::Q4_0, &w4, &a), 0.0);
    }

    #[test]
    #[should_panic(expected = "float matrix matmul reached with keep_f32 off on QuantAct")]
    fn float_matmul_panics_when_keep_f32_is_off() {
        let mut a = QuantAct::with_capacity(32);
        a.n = 32;
        let w = [0u8; 128];
        dot_scalar(Format::F32, &w, &a);
    }
    #[test]
    fn activation_quantiser_clamps_to_pm_127_so_sign_trick_pairs_stay_under_i16_max() {
        // Decision 5: A Q8_0 weight of -128 makes `sign_epi8(w, w)` give 128 (unsigned byte).
        // If an activation were allowed to quantise to -128, the sign trick would map it to 128 unsigned.
        // Then a `vpmaddubsw` pair of two adjacent products would reach:
        //   128 * 128 + 128 * 128 = 32768, which is one past i16::MAX (32767) and silently saturates.
        // Because `QuantAct::fill` and `fill_scalar` clamp activations to ±127 rather than [-128, 127],
        // the worst-case pair is:
        //   128 * 127 + 128 * 127 = 32512 <= i16::MAX (headroom of 255).
        let mut x = vec![0.0f32; 256];
        for (i, v) in x.iter_mut().enumerate() {
            *v = if i % 2 == 0 { -1e9 } else { 1e9 };
        }
        x[0] = -128.0;
        x[1] = -127.0;
        x[2] = 127.0;
        x[3] = 128.0;

        // Test scalar path
        let mut a_scalar = QuantAct::with_capacity(256);
        a_scalar.fill_scalar(&x);
        for &q in &a_scalar.q {
            assert!(
                (-127..=127).contains(&q),
                "scalar q contains {q}, must clamp to [-127, 127]"
            );
        }
        for &q in &a_scalar.qk {
            assert!(
                (-127..=127).contains(&q),
                "scalar qk contains {q}, must clamp to [-127, 127]"
            );
        }

        // Test general fill path (which uses vector on x86 if available)
        let mut a_vec = QuantAct::with_capacity(256);
        a_vec.fill(&x);
        for &q in &a_vec.q {
            assert!(
                (-127..=127).contains(&q),
                "vector q contains {q}, must clamp to [-127, 127]"
            );
        }
        for &q in &a_vec.qk {
            assert!(
                (-127..=127).contains(&q),
                "vector qk contains {q}, must clamp to [-127, 127]"
            );
        }

        // Verify the sign trick's arithmetic bounds:
        let worst_w = -128i8;
        // In the sign trick, w is multiplied by sign_epi8(a, a) or sign_epi8(w, w):
        // (-128i8).wrapping_neg() is -128 as i8, which as unsigned u8 is 128.
        let w_abs = (worst_w as i16).unsigned_abs() as u32; // 128
        let max_act_abs = 127u32;
        let worst_pair = 2 * (w_abs * max_act_abs);
        assert_eq!(worst_pair, 32512);
        assert!(
            worst_pair <= i16::MAX as u32,
            "worst-case pair {worst_pair} must stay under i16::MAX"
        );

        let saturated_pair = 2 * (w_abs * 128);
        assert_eq!(saturated_pair, 32768);
        assert!(
            saturated_pair > i16::MAX as u32,
            "if activation reached -128 (magnitude 128), pair would overflow i16::MAX"
        );
    }

    /// `prepare` + `fill_range` over one range, and over several disjoint ranges, must
    /// equal `fill` byte for byte: the parallel quantise must not change the activation
    /// a row's matmul reads, whatever the worker split.
    #[test]
    fn fill_range_matches_fill_for_any_split() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for &n in &[32usize, 96, 896, 4864] {
            let x: Vec<f32> = (0..n)
                .map(|_| (next() as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * 3.0)
                .collect();
            let mut want = QuantAct::with_capacity(n);
            want.fill(&x);
            // One range covering everything.
            let mut got = QuantAct::with_capacity(n);
            got.prepare(n);
            got.fill_range(&x, 0);
            assert_eq!(got.q, want.q, "n {n}: codes");
            assert_eq!(got.d, want.d, "n {n}: scales");
            assert_eq!(got.sums, want.sums, "n {n}: sums");
            assert_eq!(got.sums16, want.sums16, "n {n}: half sums");
            assert_eq!(got.lane_sums, want.lane_sums, "n {n}: lane sums");
            assert_eq!(got.qk, want.qk, "n {n}: per-256 codes");
            assert_eq!(got.dk, want.dk, "n {n}: per-256 scales");
            assert_eq!(got.bsums, want.bsums, "n {n}: per-256 sums");
            // Four ranges, 256-aligned (all of these n are multiples of 256 except 96).
            let mut split = QuantAct::with_capacity(n);
            split.prepare(n);
            let step = if n.is_multiple_of(256) {
                (n / 4).div_ceil(256) * 256
            } else {
                32
            };
            let mut lo = 0;
            while lo < n {
                let hi = (lo + step).min(n);
                split.fill_range(&x[lo..hi], lo);
                lo = hi;
            }
            assert_eq!(split.q, want.q, "n {n}: split codes");
            assert_eq!(split.d, want.d, "n {n}: split scales");
            assert_eq!(split.sums, want.sums, "n {n}: split sums");
            assert_eq!(split.lane_sums, want.lane_sums, "n {n}: split lane sums");
            assert_eq!(split.qk, want.qk, "n {n}: split per-256 codes");
            assert_eq!(split.bsums, want.bsums, "n {n}: split per-256 sums");
        }
    }
}
