//! The kernel table for matrix products (roadmap S1.1b, ADR 0005).
//!
//! Which kernel computes a matrix's rows depends on the weight format, how the matrix is stored (repacked
//! tiles or plain rows), how the activations were quantised, and what the CPU can run. That choice is made
//! here, from answers computed once per process ([`MatKernels::get`]), instead of with feature checks inside
//! the model code. The table also hands out the fast kernels as safe functions: a `MatKernels` that reports a
//! path exists only on a CPU that has it, and each wrapper checks the call's other preconditions.

use crate::cpu::Tier;
use crate::format::Format;
use crate::quant::{row_bytes, QuantAct};
use std::sync::OnceLock;

use crate::kq::KqActs;

/// How one matrix's rows are computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatPath {
    /// Repacked 16-row PQ2_0 tiles against tile-mode activations (AVX-512 VNNI + GFNI).
    Tiles,
    /// PQ2_0 rows in blocks of up to 32, against the ternary activation layout (AVX2).
    RowsPq2,
    /// PTQ1_0 rows in blocks, against the PTQ1 activation layout (AVX-512 VBMI).
    RowsPtq1,
    /// One row at a time through `quant::dot_batch` (every format; scalar fallback inside).
    PerRow,
    /// Repacked eight-row k-quant, Q5_0 or Q8_0 tiles (AVX2 and up).
    KqTiles,
    /// Float weights packed into 16-output-row panels (S4.9f; x86-64 and aarch64).
    FloatPanels,
}

/// The matrix kernels this process runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatKernels {
    tiles: bool,
    pq2_rows: bool,
    ptq1_rows: bool,
    kq_tiles: bool,
    kq_i8mm: bool,
    kq_zmm: bool,
    fp_panels: bool,
    exact: bool,
}

impl MatKernels {
    /// The table for this CPU and these options, computed once.
    pub fn get() -> &'static MatKernels {
        static T: OnceLock<MatKernels> = OnceLock::new();
        T.get_or_init(MatKernels::detect)
    }

    fn detect() -> MatKernels {
        #[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
        let (tiles, pq2_rows, ptq1_rows) = (
            crate::tern::have_zmm(),
            crate::simd::have_batched(),
            crate::tern::have_ptq1_zmm(),
        );
        // The NEON tier has the per-row PQ2_0 kernels (`sdot`); PTQ1_0 stays on
        // the scalar reference until it has one.
        #[cfg(all(target_arch = "aarch64", feature = "fmt-ternary"))]
        let (tiles, pq2_rows, ptq1_rows) = (false, crate::cpu::have_dotprod(), false);
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-ternary"),
            all(target_arch = "aarch64", feature = "fmt-ternary")
        )))]
        let (tiles, pq2_rows, ptq1_rows) = (false, false, false);
        // The repacked k-quant kernels need the batched kernels' instruction
        // sets, and the repack switch.
        #[cfg(target_arch = "x86_64")]
        let kq_tiles = crate::simd::have_batched() && crate::options::get().kq_repack;
        // On aarch64 the repacked tiles run on `sdot` (S2.8); Q5_0 and Q6_K have
        // tiles too, and `path` keeps every other format on the per-row kernel.
        #[cfg(target_arch = "aarch64")]
        let kq_tiles = crate::cpu::have_dotprod() && crate::options::get().kq_repack;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let kq_tiles = false;
        // The i8mm rung runs the tile GEMM on `smmla` (S2.8 F2). The GEMV stays on
        // `sdot` — the integer sums and the fold are the same, and the test pins the
        // per-row bits — and a CPU without `i8mm` is the dotprod tier.
        #[cfg(target_arch = "aarch64")]
        let kq_i8mm = kq_tiles && crate::cpu::have_i8mm();
        #[cfg(not(target_arch = "aarch64"))]
        let kq_i8mm = false;
        // The AVX-512 variant of the same kernels (S3.4 follow-up): the same
        // tiles, 64 bytes at a time. `cpu_has!` honours `PEREGRINE_ISA`, so an
        // `avx2` cap keeps the AVX2 kernel.
        #[cfg(target_arch = "x86_64")]
        let kq_zmm = kq_tiles && crate::kqsimd::have_zmm512();
        #[cfg(not(target_arch = "x86_64"))]
        let kq_zmm = false;
        // The packed float GEMM (S4.9f) runs on x86-64 and aarch64 — the scalar tier
        // included, on the scalar chain — and its load-time store is built only where
        // it does.
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let fp_panels = crate::fpack::have_packed() && crate::options::get().fpack;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let fp_panels = false;
        MatKernels {
            tiles,
            pq2_rows,
            ptq1_rows,
            kq_tiles,
            kq_i8mm,
            kq_zmm,
            fp_panels,
            exact: crate::quant::exact_ternary(),
        }
    }

    /// Whether PQ2_0 matrices may be repacked into tiles (the tile kernel can run).
    pub fn tiles(&self) -> bool {
        self.tiles
    }

    /// Whether activations of a PTQ1_0 model should use the PTQ1 layout (its kernel can run).
    pub fn ptq1_layout(&self) -> bool {
        self.ptq1_rows
    }

    /// Whether `PEREGRINE_EXACT=1` sends ternary matmuls to the f32 reference.
    pub fn exact(&self) -> bool {
        self.exact
    }

    /// Whether the repacked k-quant kernels may run.
    pub fn kq_tiles(&self) -> bool {
        self.kq_tiles
    }

    /// Whether the packed float GEMM may run, and the load-time float store is worth building.
    pub fn fp_panels(&self) -> bool {
        self.fp_panels
    }

    /// Whether `fmt` has a packed float kernel on this CPU.
    #[inline]
    pub fn fp_panels_for(&self, fmt: Format) -> bool {
        self.fp_panels && matches!(fmt, Format::F32 | Format::F16 | Format::BF16)
    }

    /// Whether `fmt` has a repacked tile kernel on this CPU.
    ///
    /// The store carries a Q4_K, Q5_K, Q5_0, Q6_K and Q8_0 tile wherever the format has a
    /// repacked layout; this table decides which of them run: Q4_K, Q5_K, Q5_0 and Q8_0 on
    /// x86-64, Q4_K, Q5_0, Q6_K and Q8_0 on aarch64. A format without a kernel keeps the
    /// per-row path even when the store holds its tiles.
    #[inline]
    pub fn kq_tiles_for(&self, fmt: Format) -> bool {
        if !self.kq_tiles {
            return false;
        }
        match fmt {
            #[cfg(feature = "fmt-kquant")]
            Format::Q4_K => true,
            // Q5_0 has a repacked tile on both kernel architectures (S3.4 on x86, S2.8 F1
            // on NEON). Every other format keeps the per-row path even when the store
            // holds its tiles.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "aarch64"),
                feature = "fmt-legacy"
            ))]
            Format::Q5_0 => true,
            #[cfg(all(
                feature = "fmt-legacy",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ))]
            Format::Q8_0 => true,
            // Q5_K has an x86 tile since S4.9i. Q6_K's tile is NEON-only for
            // now: on x86 its GEMV loses to the row kernel at one to three rows
            // (21 % on the bluefin host), so decision 9 keeps the row path
            // there; the tile's GEMM and GEMV are implemented and tested for a
            // tier whose GEMV wins (S4.9i's follow-up).
            #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
            Format::Q5_K => true,
            #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
            Format::Q6_K => true,
            _ => false,
        }
    }

    /// The path for a matrix in `fmt`, repacked into tiles or not, against activations in the PTQ1 layout or not.
    ///
    /// `kq` says whether the matrix carries repacked k-quant tiles (`TensorInfo::kq_off`);
    /// `fp` whether it carries packed float panels (`TensorInfo::fp_off`).
    #[inline]
    pub fn path(&self, fmt: Format, tiled: bool, kq: bool, fp: bool, act_ptq1: bool) -> MatPath {
        if self.exact {
            // The f32 reference lives in `dot_batch`.
            MatPath::PerRow
        } else if fp && self.fp_panels_for(fmt) {
            MatPath::FloatPanels
        } else if tiled && self.tiles {
            MatPath::Tiles
        } else if fmt == Format::PTQ1_0 && act_ptq1 && self.ptq1_rows {
            MatPath::RowsPtq1
        } else if fmt == Format::PQ2_0 && self.pq2_rows {
            MatPath::RowsPq2
        } else if kq && self.kq_tiles_for(fmt) {
            // The repacked k-quant and Q5_0/Q8_0 tiles: eight output rows at once.
            MatPath::KqTiles
        } else {
            MatPath::PerRow
        }
    }

    /// Eight rows of a repacked Q4_K tile against a batch of four or more
    /// activations, through the interleaved layout.
    ///
    /// `tile` holds `n_sb` tiles of eight rows; `out` receives `out[b · 8 + r]`.
    pub fn kq_tiles_q4k(
        &self,
        tile: &[u8],
        n_sb: usize,
        acts: &KqActs,
        nb: usize,
        out: &mut [f32],
    ) {
        assert!(
            self.kq_tiles,
            "the repacked k-quant kernel used where it cannot run"
        );
        assert!(
            nb >= 4,
            "the batched k-quant kernel takes four rows or more"
        );
        assert!(
            acts.rows >= nb,
            "the interleaved activation holds fewer rows"
        );
        assert!(
            out.len() >= nb * 8,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        // SAFETY: `kq_tiles` is `have_batched()` (AVX2, FMA, F16C); the tile holds
        // `n_sb` tiles and the activation the rows the caller says. `kq_zmm` is
        // `have_zmm512()`, which is a superset of those.
        unsafe {
            if self.kq_zmm {
                crate::kqsimd::q4k8_gemm_zmm(tile, n_sb, acts, nb, out)
            } else {
                crate::kqsimd::q4k8_gemm(tile, n_sb, acts, nb, out)
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
        // SAFETY: `kq_tiles` is `have_dotprod()`; `kq_i8mm` is `have_i8mm()`, and
        // both kernels take the same tile and activation layouts. The tile holds
        // `n_sb` super-blocks and the interleaved buffer the rows the caller says.
        unsafe {
            if self.kq_i8mm {
                crate::neon::q4k8_gemm_i8mm(tile, n_sb, acts, nb, out)
            } else {
                crate::neon::q4k8_gemm(tile, n_sb, acts, nb, out)
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-kquant"),
            all(target_arch = "aarch64", feature = "fmt-kquant")
        )))]
        {
            let _ = (tile, n_sb);
            unreachable!("no repacked k-quant kernel on this architecture")
        }
    }

    /// A run of packed float panels against a batch of activation rows (S4.9f).
    ///
    /// `packed` holds `panels` panels of `n_in` weights; `out[b · panels · 16 + i]`
    /// receives output row `i` of the run for activation row `b`. Every element is
    /// one sequential FMA chain over `k`, so the bits do not depend on the batch.
    #[allow(clippy::too_many_arguments)]
    pub fn fp_panels_gemm(
        &self,
        fmt: Format,
        packed: &[u8],
        n_in: usize,
        panels: usize,
        acts: &[QuantAct],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        assert!(
            self.fp_panels_for(fmt),
            "the packed float kernel used where it cannot run"
        );
        assert!(packed.len() >= panels * crate::fpack::panel_bytes(fmt, n_in));
        assert!(out.len() >= panels * crate::fpack::PANEL * acts.len());
        crate::fpack::gemm(fmt, packed, n_in, panels, acts, bias, out);
    }

    /// [`MatKernels::fp_panels_gemm`] with the caller's output stride: activation
    /// row `b` is written at `out[b · out_stride ..][..panels · 16]`, so a
    /// projection's row-major buffer is filled in place.
    #[allow(clippy::too_many_arguments)]
    pub fn fp_panels_gemm_strided(
        &self,
        fmt: Format,
        packed: &[u8],
        n_in: usize,
        panels: usize,
        acts: &[QuantAct],
        bias: Option<&[f32]>,
        out: &mut [f32],
        out_stride: usize,
    ) {
        assert!(
            self.fp_panels_for(fmt),
            "the packed float kernel used where it cannot run"
        );
        assert!(packed.len() >= panels * crate::fpack::panel_bytes(fmt, n_in));
        assert!(out_stride >= panels * crate::fpack::PANEL);
        assert!(
            acts.is_empty()
                || out.len() >= (acts.len() - 1) * out_stride + panels * crate::fpack::PANEL
        );
        crate::fpack::gemm_strided(fmt, packed, n_in, panels, acts, bias, out, out_stride);
    }

    /// Eight rows of a repacked Q4_K tile against one to three activations, in
    /// the plain per-row layout.
    ///
    /// The decode path: the same fold as the batched kernel, so a row's result
    /// does not depend on the batch it ran in.
    pub fn kq_tiles_q4k_gemv(
        &self,
        tile: &[u8],
        n_sb: usize,
        acts: &[QuantAct],
        nb: usize,
        out: &mut [f32],
    ) {
        assert!(
            self.kq_tiles,
            "the repacked k-quant kernel used where it cannot run"
        );
        assert!(
            (1..=3).contains(&nb),
            "the single-row k-quant kernel takes one to three rows"
        );
        assert!(
            out.len() >= nb * 8,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        // SAFETY: as above.
        unsafe {
            if self.kq_zmm {
                crate::kqsimd::q4k8_gemv_zmm(tile, n_sb, acts, nb, out)
            } else {
                crate::kqsimd::q4k8_gemv(tile, n_sb, acts, nb, out)
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
        // SAFETY: `kq_tiles` is `have_dotprod()`; the tile holds `n_sb` super-blocks
        // and the plain activation rows the caller says.
        unsafe {
            crate::neon::q4k8_gemv(tile, n_sb, acts, nb, out)
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-kquant"),
            all(target_arch = "aarch64", feature = "fmt-kquant")
        )))]
        {
            let _ = (tile, n_sb, acts);
            unreachable!("no repacked k-quant kernel on this architecture")
        }
    }

    /// The weight formats whose prompt rows take the 8-bit form on this tier
    /// (S5.4f).
    ///
    /// This is the one table the load path, the dispatch and the startup line
    /// read. A format belongs here only when the form's kernel is at least 1.10×
    /// this tier's exact kernel for that format at prompt widths (64 activation
    /// rows and up); the measurements behind each entry are in
    /// `docs/design/s5.4-fast-prompt.md` §5g, §5h and
    /// `benchmarks/results/bluefin-2026-11-12-s5.4f/` and
    /// `benchmarks/results/spark-arm-2026-11-12-s5.4h/`.
    ///
    /// * **AVX-512** (unchanged by S5.4f: it cannot be measured on the AVX2 host):
    ///   Q4_K and Q6_K.
    /// * **AVX2**: Q6_K only. Q4_K's exact wide kernel is 1.26× the form's, and
    ///   Q5_0 is level at 2048 weights (1.00×) and 0.88× at Qwen 2.5's 896-wide
    ///   rows, where the form pays a padded fourth super-block — below the 1.10×
    ///   bar both ways.
    /// * **aarch64**: Q4_K and Q6_K on every cap (S5.4e's set), and Q5_0 where the
    ///   multiply is a NEON body (`neon-dotprod` and `neon-i8mm`; S5.4h). The form
    ///   is 2.4–2.7× the exact Q4_K kernel, 3.4–4.1× Q6_K and 4.7–5.7× Q5_0 from
    ///   64 activation rows up at one and ten workers, on a whole 2048-weight row
    ///   and on Qwen 2.5's 896-wide one (the padded fourth super-block costs
    ///   13.7 % of the rate, and the form is still 4.8× there). The `neon` cap
    ///   keeps S5.4e's two
    ///   formats: its multiply *is* the scalar twin, which is no faster than the
    ///   exact kernel for any format, so another format would only add memory.
    /// * **Scalar, and any tier without a SIMD body**: nothing. The mode then
    ///   computes exactly what the exact engine computes.
    #[inline]
    pub fn q8k_uses_form(&self, kind: Format) -> bool {
        self.q8k_formats().contains(&kind)
    }

    /// [`MatKernels::q8k_uses_form`] as a list, for the startup line, `peregrine
    /// tier` and the option's help text.
    pub fn q8k_formats(&self) -> &'static [Format] {
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        {
            if crate::kqsimd::have_zmm512() {
                return &[Format::Q4_K, Format::Q6_K];
            }
            if crate::kqsimd::have_avx2() {
                return &[Format::Q6_K];
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
        {
            // S5.4h: Q5_0 joins where the multiply is a NEON body. The `neon` cap
            // runs the scalar twin, which gains nothing from a third format.
            if crate::cpu::have_dotprod() && cfg!(feature = "fmt-legacy") {
                &[Format::Q4_K, Format::Q6_K, Format::Q5_0]
            } else {
                &[Format::Q4_K, Format::Q6_K]
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "fmt-kquant")))]
        {
            &[]
        }
    }

    /// The resident 8-bit prompt form of one run of eight output rows against a
    /// batch of activation rows (`crate::kq8`, S5.4a; dispatched here since
    /// S5.4b).
    ///
    /// `form` is one run of eight rows over `n_sb` 256-weight super-blocks, as
    /// the load-time builder lays it out; `out[b · 8 + r]` receives output row
    /// `r` for activation row `b`. The form is format-agnostic (Q4_K and Q6_K
    /// produce the same object), so one entry point multiplies both.
    ///
    /// The AVX-512 body is `crate::kqsimd::q8k_gemm_zmm`, the AVX2 body is
    /// `crate::kqsimd::q8k_gemm_avx2`; on aarch64 the NEON entry point
    /// `crate::neon::q8k_gemm` picks `smmla` on `neon-i8mm` and `sdot` on
    /// `neon-dotprod` (S5.4e); a tier with none of them runs the scalar twin,
    /// [`crate::kq8::gemm_q8k`]. They are bit-identical on the same tier
    /// (`crates/peregrine-kernels/src/kqsimd.rs`'s and `neon.rs`'s tests), so this
    /// choice moves time, never a number.
    pub fn q8k_gemm(&self, form: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
        assert!(
            form.len() >= n_sb * crate::kq8::Q8K_SIZE,
            "the form holds fewer than n_sb super-blocks"
        );
        assert!(nb <= acts.rows, "more activations than the buffer holds");
        assert!(
            out.len() >= nb * crate::kq8::Q8K_ROWS,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        {
            if crate::kqsimd::have_zmm512() {
                // SAFETY: `have_zmm512()` checked; `q8k_gemm_zmm` checks the same
                // bounds this wrapper does.
                unsafe { crate::kqsimd::q8k_gemm_zmm(form, n_sb, acts, nb, out) };
                return;
            }
            if crate::kqsimd::have_avx2() {
                // SAFETY: `have_avx2()` checked; `q8k_gemm_avx2` checks the same
                // bounds this wrapper does.
                unsafe { crate::kqsimd::q8k_gemm_avx2(form, n_sb, acts, nb, out) };
                return;
            }
        }
        // SAFETY: `have_dotprod()` is the tier's own check (it honours
        // `PEREGRINE_ISA`); `neon::q8k_gemm` picks `smmla` only where
        // `have_i8mm()` holds and checks the same bounds this wrapper does.
        #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
        {
            if crate::cpu::have_dotprod() {
                unsafe { crate::neon::q8k_gemm(form, n_sb, acts, nb, out) };
                return;
            }
        }
        crate::kq8::gemm_q8k(form, n_sb, acts, nb, out)
    }

    /// Sixteen output rows — two adjacent tiles of the form — against a batch of
    /// activation rows (S5.4i).
    ///
    /// `form` holds two tiles of `n_sb` super-blocks, the second first at
    /// `n_sb * Q8K_SIZE`; `out[b · 16 + r]` is tile 0's row `r` for `r < 8` and tile
    /// 1's row `r − 8` otherwise. It is exactly two [`MatKernels::q8k_gemm`] calls
    /// into one buffer, row for row and bit for bit: the sixteen-row AVX-512 body
    /// shares each activation broadcast between the two tiles and changes nothing
    /// else. A tier without that body takes the scalar pair, which is two calls of
    /// the twin.
    pub fn q8k_gemm16(&self, form: &[u8], n_sb: usize, acts: &KqActs, nb: usize, out: &mut [f32]) {
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        if crate::kqsimd::have_zmm512() {
            // SAFETY: `have_zmm512()` checked; `q8k_gemm16_zmm` checks the same
            // bounds this wrapper does.
            unsafe { crate::kqsimd::q8k_gemm16_zmm(form, n_sb, acts, nb, out) };
            return;
        }
        crate::kq8::gemm_q8k_pair(form, n_sb, acts, nb, out)
    }

    /// Build the 8-bit form of one run of eight output rows from a repacked tile:
    /// the AVX-512 builder when the tier has it, the AVX2 one next, the NEON
    /// builder on aarch64, the scalar twin otherwise (S5.4b, S5.4d, S5.4e, Q5_0
    /// since S5.4f).
    ///
    /// `n_sb` is the number of 256-weight super-blocks for Q4_K and Q6_K, and the
    /// number of 32-weight blocks for Q5_0 (the form is written one 256-weight
    /// super-block at a time, so the last one may be short); the Q5_0 builders
    /// take the block count, as `crate::kq8::build_q5_0` does.
    ///
    /// All of them write the same bytes: the conformance tests in `crate::kqsimd`,
    /// `crate::neon` and the per-weight bound hold either way.
    pub fn q8k_build(&self, kind: Format, tile: &[u8], n_sb: usize, form: &mut [u8]) {
        assert!(
            matches!(kind, Format::Q4_K | Format::Q6_K)
                || (cfg!(feature = "fmt-legacy") && kind == Format::Q5_0),
            "only Q4_K, Q6_K and Q5_0 have an 8-bit form"
        );
        let form_sb = if kind == Format::Q5_0 {
            n_sb.div_ceil(crate::kq8::Q8K_WEIGHTS / crate::kq::Q5_0_WEIGHTS)
        } else {
            n_sb
        };
        assert!(
            form.len() >= form_sb * crate::kq8::Q8K_SIZE,
            "the form holds fewer than n_sb super-blocks"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        {
            if crate::kqsimd::have_zmm512() {
                // SAFETY: `have_zmm512()` checked; both builders check the tile's and
                // the form's lengths.
                unsafe {
                    match kind {
                        Format::Q4_K => crate::kqsimd::q8k_build_q4k_zmm(tile, n_sb, form),
                        Format::Q6_K => crate::kqsimd::q8k_build_q6k_zmm(tile, n_sb, form),
                        #[cfg(feature = "fmt-legacy")]
                        Format::Q5_0 => crate::kqsimd::q8k_build_q5_0_zmm(tile, n_sb, form),
                        _ => unreachable!("the format has no 8-bit form"),
                    }
                }
                return;
            }
            if crate::kqsimd::have_avx2() {
                // SAFETY: `have_avx2()` checked; both builders check the tile's and
                // the form's lengths.
                unsafe {
                    match kind {
                        Format::Q4_K => crate::kqsimd::q8k_build_q4k_avx2(tile, n_sb, form),
                        Format::Q6_K => crate::kqsimd::q8k_build_q6k_avx2(tile, n_sb, form),
                        #[cfg(feature = "fmt-legacy")]
                        Format::Q5_0 => crate::kqsimd::q8k_build_q5_0_avx2(tile, n_sb, form),
                        _ => unreachable!("the format has no 8-bit form"),
                    }
                }
                return;
            }
        }
        // SAFETY: `have_neon()` is the tier's own check (it honours
        // `PEREGRINE_ISA`); both builders need only `neon` and check the tile's
        // and the form's lengths.
        #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
        {
            if crate::cpu::have_neon() {
                unsafe {
                    match kind {
                        Format::Q4_K => crate::neon::q8k_build_q4k_neon(tile, n_sb, form),
                        Format::Q6_K => crate::neon::q8k_build_q6k_neon(tile, n_sb, form),
                        #[cfg(feature = "fmt-legacy")]
                        Format::Q5_0 => crate::neon::q8k_build_q5_0_neon(tile, n_sb, form),
                        _ => unreachable!("the format has no 8-bit form"),
                    }
                }
                return;
            }
        }
        match kind {
            Format::Q4_K => crate::kq8::build_q4k(tile, n_sb, form),
            Format::Q6_K => crate::kq8::build_q6k(tile, n_sb, form),
            #[cfg(feature = "fmt-legacy")]
            Format::Q5_0 => crate::kq8::build_q5_0(tile, n_sb, form),
            _ => unreachable!("the format has no 8-bit form"),
        }
    }

    /// The name of the 8-bit prompt form's SIMD body on this tier: `Some("AVX-512")`,
    /// `Some("AVX2")`, `Some("NEON smmla")` or `Some("NEON sdot")`, and `None` where
    /// the mode runs through the scalar twin (the option's startup line says so).
    pub fn q8k_body(&self) -> Option<&'static str> {
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        {
            if crate::kqsimd::have_zmm512() {
                Some("AVX-512")
            } else if crate::kqsimd::have_avx2() {
                Some("AVX2")
            } else {
                None
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
        {
            if crate::cpu::have_dotprod() {
                Some(if crate::cpu::have_i8mm() {
                    "NEON smmla"
                } else {
                    "NEON sdot"
                })
            } else {
                None
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-kquant"),
            all(target_arch = "aarch64", feature = "fmt-kquant")
        )))]
        {
            None
        }
    }

    /// Whether the 8-bit prompt form has a SIMD body on this tier (AVX-512, AVX2 or
    /// NEON).
    ///
    /// A tier without one runs the mode correctly through the scalar twin, just
    /// no faster than the exact kernel; the option's startup line says so.
    pub fn q8k_simd(&self) -> bool {
        self.q8k_body().is_some()
    }

    /// Eight rows of a repacked Q6_K tile against a batch of four or more
    /// activations, through the interleaved layout.
    ///
    /// `tile` holds `n_sb` tiles of eight rows; `out` receives `out[b · 8 + r]`.
    pub fn kq_tiles_q6k(
        &self,
        tile: &[u8],
        n_sb: usize,
        acts: &KqActs,
        nb: usize,
        out: &mut [f32],
    ) {
        assert!(
            self.kq_tiles,
            "the repacked k-quant kernel used where it cannot run"
        );
        assert!(
            nb >= 4,
            "the batched k-quant kernel takes four rows or more"
        );
        assert!(
            acts.rows >= nb,
            "the interleaved activation holds fewer rows"
        );
        assert!(
            out.len() >= nb * 8,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        // SAFETY: `kq_tiles` is `have_batched()` (AVX2, FMA, F16C); `kq_zmm` is
        // `have_zmm512()`, which is a superset of those. The tile holds `n_sb`
        // tiles and the activation the rows the caller says.
        unsafe {
            if self.kq_zmm {
                crate::kqsimd::q6k8_gemm_zmm(tile, n_sb, acts, nb, out)
            } else {
                crate::kqsimd::q6k8_gemm(tile, n_sb, acts, nb, out)
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
        // SAFETY: `kq_tiles` is `have_dotprod()`; `kq_i8mm` is `have_i8mm()`. The
        // tile holds `n_sb` super-blocks and the interleaved buffer the rows the
        // caller says.
        unsafe {
            if self.kq_i8mm {
                crate::neon::q6k8_gemm_i8mm(tile, n_sb, acts, nb, out)
            } else {
                crate::neon::q6k8_gemm(tile, n_sb, acts, nb, out)
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-kquant"),
            all(target_arch = "aarch64", feature = "fmt-kquant")
        )))]
        {
            let _ = (tile, n_sb, acts, nb, out);
            unreachable!("no repacked Q6_K kernel on this architecture")
        }
    }

    /// Eight rows of a repacked Q6_K tile against one to three plain activations.
    pub fn kq_tiles_q6k_gemv(
        &self,
        tile: &[u8],
        n_sb: usize,
        acts: &[QuantAct],
        nb: usize,
        out: &mut [f32],
    ) {
        assert!(
            self.kq_tiles,
            "the repacked k-quant kernel used where it cannot run"
        );
        assert!(
            (1..=3).contains(&nb),
            "the Q6_K GEMV takes one to three rows"
        );
        assert!(
            out.len() >= nb * 8,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        // SAFETY: `kq_tiles` is `have_batched()` (AVX2, FMA, F16C); the tile holds
        // `n_sb` tiles and the plain activation rows the caller says.
        unsafe {
            crate::kqsimd::q6k8_gemv(tile, n_sb, acts, nb, out)
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
        // SAFETY: `kq_tiles` is `have_dotprod()`; the tile holds `n_sb` super-blocks
        // and the plain activation rows the caller says.
        unsafe {
            crate::neon::q6k8_gemv(tile, n_sb, acts, nb, out)
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-kquant"),
            all(target_arch = "aarch64", feature = "fmt-kquant")
        )))]
        {
            let _ = (tile, n_sb, acts, nb, out);
            unreachable!("no repacked Q6_K kernel on this architecture")
        }
    }

    /// Eight rows of a repacked Q5_K tile against a batch of four or more
    /// activations, through the interleaved layout.
    ///
    /// `tile` holds `n_sb` tiles of eight rows; `out` receives `out[b · 8 + r]`.
    pub fn kq_tiles_q5k(
        &self,
        tile: &[u8],
        n_sb: usize,
        acts: &KqActs,
        nb: usize,
        out: &mut [f32],
    ) {
        assert!(
            self.kq_tiles_for(Format::Q5_K),
            "the repacked Q5_K kernel used where it cannot run"
        );
        assert!(nb >= 4, "the batched Q5_K kernel takes four rows or more");
        assert!(
            acts.rows >= nb,
            "the interleaved activation holds fewer rows"
        );
        assert!(
            out.len() >= nb * 8,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        // SAFETY: `kq_tiles` is `have_batched()` (AVX2, FMA, F16C); the tile holds
        // `n_sb` tiles and the activation the rows the caller says.
        unsafe {
            crate::kqsimd::q5k8_gemm(tile, n_sb, acts, nb, out)
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "fmt-kquant")))]
        {
            let _ = (tile, n_sb, acts, nb, out);
            unreachable!("no repacked Q5_K kernel on this architecture")
        }
    }

    /// Eight rows of a repacked Q5_K tile against one to three plain activations.
    pub fn kq_tiles_q5k_gemv(
        &self,
        tile: &[u8],
        n_sb: usize,
        acts: &[QuantAct],
        nb: usize,
        out: &mut [f32],
    ) {
        assert!(
            self.kq_tiles_for(Format::Q5_K),
            "the repacked Q5_K kernel used where it cannot run"
        );
        assert!(
            (1..=3).contains(&nb),
            "the Q5_K GEMV takes one to three rows"
        );
        assert!(
            out.len() >= nb * 8,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-kquant"))]
        // SAFETY: `kq_tiles` is `have_batched()` (AVX2, FMA, F16C); the tile holds
        // `n_sb` tiles and the plain activation rows the caller says.
        unsafe {
            crate::kqsimd::q5k8_gemv(tile, n_sb, acts, nb, out)
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "fmt-kquant")))]
        {
            let _ = (tile, n_sb, acts, nb, out);
            unreachable!("no repacked Q5_K kernel on this architecture")
        }
    }

    /// Eight rows of a repacked Q5_0 tile against a batch of activations.
    ///
    /// Q5_0's activation is the ordinary per-32 layout, so the batched and the
    /// single-row paths read the same bytes and the kernel is one code path.
    pub fn kq_tiles_q5_0(
        &self,
        tile: &[u8],
        n_blk: usize,
        acts: &[QuantAct],
        nb: usize,
        out: &mut [f32],
    ) {
        assert!(
            self.kq_tiles_for(Format::Q5_0),
            "the repacked Q5_0 kernel used where it cannot run"
        );
        assert!(
            !acts.is_empty() && nb <= acts.len(),
            "more rows than activations"
        );
        assert!(
            out.len() >= nb * 8,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-legacy"))]
        // SAFETY: `kq_tiles` is `have_batched()` (AVX2, FMA, F16C); the tile holds
        // `n_blk` tiles and the activation the rows the caller says. `kq_zmm` is
        // `have_zmm512()`, which is a superset of those.
        unsafe {
            if self.kq_zmm {
                crate::kqsimd::q5_08_gemm_zmm(tile, n_blk, acts, nb, out)
            } else {
                crate::kqsimd::q5_08_gemm(tile, n_blk, acts, nb, out)
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-legacy"))]
        // SAFETY: `kq_tiles` is `have_dotprod()`; `kq_i8mm` is `have_i8mm()`. The
        // tile holds `n_blk` blocks and the activation rows the caller says.
        unsafe {
            if self.kq_i8mm {
                crate::neon::q5_08_gemm_i8mm(tile, n_blk, acts, nb, out)
            } else {
                crate::neon::q5_08_gemm(tile, n_blk, acts, nb, out)
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-legacy"),
            all(target_arch = "aarch64", feature = "fmt-legacy")
        )))]
        {
            let _ = (tile, n_blk, acts, nb, out);
            unreachable!("no repacked Q5_0 kernel on this architecture")
        }
    }

    /// Eight rows of a repacked Q5_0 tile against one to three activations.
    pub fn kq_tiles_q5_0_gemv(
        &self,
        tile: &[u8],
        n_blk: usize,
        acts: &[QuantAct],
        nb: usize,
        out: &mut [f32],
    ) {
        assert!(
            self.kq_tiles_for(Format::Q5_0),
            "the repacked Q5_0 kernel used where it cannot run"
        );
        assert!(
            (1..=3).contains(&nb),
            "the Q5_0 GEMV takes one to three rows"
        );
        assert!(
            out.len() >= nb * 8,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-legacy"))]
        // SAFETY: as above.
        unsafe {
            if self.kq_zmm {
                crate::kqsimd::q5_08_gemv_zmm(tile, n_blk, acts, nb, out)
            } else {
                crate::kqsimd::q5_08_gemv(tile, n_blk, acts, nb, out)
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-legacy"))]
        // SAFETY: as above.
        unsafe {
            crate::neon::q5_08_gemv(tile, n_blk, acts, nb, out)
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-legacy"),
            all(target_arch = "aarch64", feature = "fmt-legacy")
        )))]
        {
            let _ = (tile, n_blk, acts, nb, out);
            unreachable!("no repacked Q5_0 kernel on this architecture")
        }
    }

    /// Eight rows of a repacked Q8_0 tile against a batch of activations.
    ///
    /// Q8_0's activation is the ordinary per-32 layout, so the batched and the
    /// single-row paths read the same bytes and the kernel is one code path.
    pub fn kq_tiles_q8_0(
        &self,
        tile: &[u8],
        n_blk: usize,
        acts: &[QuantAct],
        nb: usize,
        out: &mut [f32],
    ) {
        assert!(
            self.kq_tiles,
            "the repacked Q8_0 kernel used where it cannot run"
        );
        assert!(
            !acts.is_empty() && nb <= acts.len(),
            "more rows than activations"
        );
        assert!(
            out.len() >= nb * 8,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-legacy"))]
        // SAFETY: `kq_tiles` is `have_batched()` (AVX2, FMA, F16C); the tile holds
        // `n_blk` tiles and the activation the rows the caller says. `kq_zmm` is
        // `have_zmm512()`, which is a superset of those.
        unsafe {
            if self.kq_zmm {
                crate::kqsimd::q8_08_gemm_zmm(tile, n_blk, acts, nb, out)
            } else {
                crate::kqsimd::q8_08_gemm(tile, n_blk, acts, nb, out)
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-legacy"))]
        // SAFETY: `kq_tiles` is `have_dotprod()`; `kq_i8mm` is `have_i8mm()`. The
        // tile holds `n_blk` blocks and the activation rows the caller says.
        unsafe {
            if self.kq_i8mm {
                crate::neon::q8_08_gemm_i8mm(tile, n_blk, acts, nb, out)
            } else {
                crate::neon::q8_08_gemm(tile, n_blk, acts, nb, out)
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-legacy"),
            all(target_arch = "aarch64", feature = "fmt-legacy")
        )))]
        {
            let _ = (tile, n_blk, acts, nb, out);
            unreachable!("no repacked Q8_0 kernel on this architecture")
        }
    }

    /// Eight rows of a repacked Q8_0 tile against one to three activations.
    pub fn kq_tiles_q8_0_gemv(
        &self,
        tile: &[u8],
        n_blk: usize,
        acts: &[QuantAct],
        nb: usize,
        out: &mut [f32],
    ) {
        assert!(
            self.kq_tiles,
            "the repacked Q8_0 kernel used where it cannot run"
        );
        assert!(
            (1..=3).contains(&nb),
            "the Q8_0 GEMV takes one to three rows"
        );
        assert!(
            out.len() >= nb * 8,
            "the output holds fewer than eight rows per activation"
        );
        #[cfg(all(target_arch = "x86_64", feature = "fmt-legacy"))]
        // SAFETY: as above.
        unsafe {
            if self.kq_zmm {
                crate::kqsimd::q8_08_gemv_zmm(tile, n_blk, acts, nb, out)
            } else {
                crate::kqsimd::q8_08_gemv(tile, n_blk, acts, nb, out)
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-legacy"))]
        // SAFETY: as above.
        unsafe {
            crate::neon::q8_08_gemv(tile, n_blk, acts, nb, out)
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-legacy"),
            all(target_arch = "aarch64", feature = "fmt-legacy")
        )))]
        {
            let _ = (tile, n_blk, acts, nb, out);
            unreachable!("no repacked Q8_0 kernel on this architecture")
        }
    }

    /// `nt` tiles of a repacked PQ2_0 matrix with `ng` groups of 128 per row, against every activation, into
    /// `out[r·nb + b]` for the tiles' 16·nt rows.
    pub fn tiles_pq2(&self, wt: &[u8], ng: usize, nt: usize, acts: &[QuantAct], out: &mut [f32]) {
        assert!(self.tiles, "tile kernel used where it cannot run");
        assert!(!acts.is_empty());
        let span = acts[0].tb_span;
        assert!(
            matches!(span, 32 | 64 | 128),
            "activations are not in tile mode"
        );
        assert!(acts.iter().all(|a| a.tb_span == span && a.n == ng * 128));
        assert!(wt.len() >= nt * 16 * row_bytes(Format::PQ2_0, ng * 128));
        assert!(out.len() >= 16 * nt * acts.len());
        #[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
        // SAFETY: `tiles` is `have_zmm()`; the activations are in one tile mode; lengths checked above.
        unsafe {
            crate::tern::dot_tiles_pq2(wt, ng, nt, acts, out)
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "fmt-ternary")))]
        unreachable!("no tile kernel on this architecture")
    }

    /// `rows` PQ2_0 rows of `rb` bytes against every activation (ternary layout), into `out[r·nb + b]`.
    pub fn rows_pq2(&self, w: &[u8], rb: usize, rows: usize, acts: &[QuantAct], out: &mut [f32]) {
        assert!(self.pq2_rows, "PQ2_0 row kernel used where it cannot run");
        assert!(!acts.is_empty());
        assert!(acts
            .iter()
            .all(|a| a.ternary && rb == row_bytes(Format::PQ2_0, a.n)));
        assert!(w.len() >= rows * rb && out.len() >= rows * acts.len());
        #[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
        // SAFETY: `pq2_rows` is `have_batched()` (AVX2, FMA, F16C); the activations are in the ternary layout and
        // match the row width; lengths checked above.
        unsafe {
            crate::tern::dot_rows_pq2(w, rb, rows, acts, out)
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-ternary"))]
        // SAFETY: `pq2_rows` is `have_dotprod()` (NEON + `sdot`); the activations are in the ternary layout and
        // match the row width; lengths checked above.
        unsafe {
            crate::neon::dot_rows_pq2(w, rb, rows, acts, out)
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-ternary"),
            all(target_arch = "aarch64", feature = "fmt-ternary")
        )))]
        unreachable!("no PQ2_0 row kernel on this architecture")
    }

    /// `rows` PTQ1_0 rows of `rb` bytes against every activation (PTQ1 layout), into `out[r·nb + b]`.
    pub fn rows_ptq1(&self, w: &[u8], rb: usize, rows: usize, acts: &[QuantAct], out: &mut [f32]) {
        assert!(self.ptq1_rows, "PTQ1_0 row kernel used where it cannot run");
        assert!(!acts.is_empty());
        assert!(acts
            .iter()
            .all(|a| a.ptq1 && a.n % 256 == 0 && rb == row_bytes(Format::PTQ1_0, a.n)));
        assert!(w.len() >= rows * rb && out.len() >= rows * acts.len());
        #[cfg(all(target_arch = "x86_64", feature = "fmt-ternary"))]
        // SAFETY: `ptq1_rows` is `have_ptq1_zmm()`; the activations are in the PTQ1 layout, width a multiple of
        // 256; lengths checked above.
        unsafe {
            crate::tern::dot_rows_ptq1(w, rb, rows, acts, out)
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "fmt-ternary")))]
        unreachable!("no PTQ1_0 row kernel on this architecture")
    }

    /// One row in any format against every activation (`quant::dot_batch`).
    #[inline]
    pub fn row(&self, fmt: Format, w: &[u8], acts: &[QuantAct], out: &mut [f32]) {
        crate::quant::dot_batch(fmt, w, acts, out)
    }

    /// Whether a decode step (one activation) can compute `fmt` eight rows at a time.
    ///
    /// The eight-row kernel shares one activation's block and lane sums across eight
    /// output rows. Each row's block order is the per-row kernel's, so the result is
    /// the same bits; only the loop that holds them differs.
    #[inline]
    pub fn rows8(&self, fmt: Format) -> bool {
        #[cfg(all(target_arch = "x86_64", feature = "fmt-legacy"))]
        if matches!(fmt, Format::Q5_0 | Format::Q8_0) {
            return crate::simd::have_batched();
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-legacy"))]
        if matches!(fmt, Format::Q5_0 | Format::Q8_0) {
            return crate::cpu::have_dotprod();
        }
        let _ = fmt;
        false
    }

    /// Eight consecutive rows of `fmt` (a legacy 32-weight block) of row size `rb`
    /// against one activation, into `out[r]`.
    ///
    /// # Panics
    /// If [`MatKernels::rows8`] is false for the format, or `w`/`out` are short.
    pub fn row8(&self, fmt: Format, w: &[u8], rb: usize, a: &QuantAct, out: &mut [f32]) {
        assert!(self.rows8(fmt), "the eight-row kernel cannot run");
        #[cfg(all(target_arch = "x86_64", feature = "fmt-legacy"))]
        // SAFETY: `have_batched()` is AVX2+FMA+F16C, checked by `rows8`; the lengths
        // are asserted by the kernel.
        unsafe {
            match fmt {
                Format::Q5_0 => crate::simd::q5_0_rows8(w, rb, a, out),
                Format::Q8_0 => crate::simd::q8_0_rows8(w, rb, a, out),
                _ => unreachable!("rows8 advertised a format without a kernel"),
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fmt-legacy"))]
        // SAFETY: `have_dotprod()` is checked by `rows8`; the lengths are asserted by
        // the kernel.
        unsafe {
            match fmt {
                Format::Q5_0 => crate::neon::q5_0_rows8(w, rb, a, out),
                Format::Q8_0 => crate::neon::q8_0_rows8(w, rb, a, out),
                _ => unreachable!("rows8 advertised a format without a kernel"),
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "fmt-legacy"),
            all(target_arch = "aarch64", feature = "fmt-legacy")
        )))]
        {
            let _ = (fmt, w, rb, a, out);
            unreachable!("no eight-row kernel on this architecture")
        }
    }

    /// The kernel tiers that provide a dedicated kernel for `fmt`.
    pub fn kernels_for(fmt: Format) -> &'static [Tier] {
        kernels_for(fmt)
    }
}

/// The kernel tiers that provide a dedicated kernel for `fmt`.
pub fn kernels_for(fmt: Format) -> &'static [Tier] {
    if !fmt.is_supported() {
        return &[];
    }
    // aarch64 has the NEON per-row kernels and repacked tiles (S2.8) and the scalar reference
    // for everything else; the x86 tiers are not compiled in there, so they are not listed.
    // `neon-i8mm` is the same kernels with the tile GEMM on `smmla` (F2, S4.9q).
    #[cfg(target_arch = "aarch64")]
    let tiers: &[Tier] = match fmt {
        // The packed float GEMM's NEON kernel (S4.9f): F32, BF16 and F16 all run on
        // baseline NEON; `fcvtl` is base Advanced SIMD, not the `fp16` arithmetic
        // extension.
        Format::F32 | Format::F16 | Format::BF16 => &[Tier::Scalar, Tier::Neon],
        #[cfg(feature = "fmt-legacy")]
        Format::Q8_0 => &[Tier::Scalar, Tier::NeonDotprod, Tier::NeonI8mm],
        #[cfg(feature = "fmt-legacy")]
        Format::Q5_0 => &[Tier::Scalar, Tier::NeonDotprod, Tier::NeonI8mm],
        #[cfg(feature = "fmt-kquant")]
        Format::Q4_K => &[Tier::Scalar, Tier::NeonDotprod, Tier::NeonI8mm],
        #[cfg(feature = "fmt-kquant")]
        Format::Q6_K => &[Tier::Scalar, Tier::NeonDotprod, Tier::NeonI8mm],
        #[cfg(feature = "fmt-ternary")]
        Format::PQ2_0 => &[Tier::Scalar, Tier::NeonDotprod],
        _ => &[Tier::Scalar],
    };
    #[cfg(not(target_arch = "aarch64"))]
    let tiers: &[Tier] = match fmt {
        Format::F32 | Format::F16 | Format::BF16 => &[Tier::Scalar, Tier::Avx2],
        #[cfg(feature = "fmt-legacy")]
        Format::Q4_0 => &[Tier::Scalar, Tier::Avx2],
        // Q8_0's repacked tile has an AVX-512 variant, like Q5_0's.
        #[cfg(feature = "fmt-legacy")]
        Format::Q8_0 => &[Tier::Scalar, Tier::Avx2, Tier::Avx512],
        // Q5_0 has a repacked tile with an AVX-512 variant (S3.4 §11 and §12).
        #[cfg(feature = "fmt-legacy")]
        Format::Q5_0 => &[Tier::Scalar, Tier::Avx2, Tier::Avx512],
        #[cfg(feature = "fmt-kquant")]
        Format::Q2_K | Format::Q3_K | Format::Q5_K | Format::Q6_K => &[Tier::Scalar, Tier::Avx2],
        // Q4_K's repacked tile has an AVX-512 variant (S3.4 §12).
        #[cfg(feature = "fmt-kquant")]
        Format::Q4_K => &[Tier::Scalar, Tier::Avx2, Tier::Avx512],
        #[cfg(feature = "fmt-iq")]
        Format::IQ4_NL | Format::IQ4_XS => &[Tier::Scalar, Tier::Avx2],
        #[cfg(feature = "fmt-mxfp4")]
        Format::MXFP4 => &[Tier::Scalar, Tier::Avx2],
        #[cfg(feature = "fmt-ternary")]
        Format::PQ2_0 | Format::PTQ1_0 => &[Tier::Scalar, Tier::Avx2, Tier::Avx512],
        #[allow(unreachable_patterns)]
        _ => &[],
    };
    tiers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn test_kernels_for_tiers() {
        assert_eq!(kernels_for(Format::F32), &[Tier::Scalar, Tier::Avx2]);
        assert_eq!(kernels_for(Format::F16), &[Tier::Scalar, Tier::Avx2]);
        assert_eq!(kernels_for(Format::BF16), &[Tier::Scalar, Tier::Avx2]);
        #[cfg(feature = "fmt-legacy")]
        {
            assert_eq!(kernels_for(Format::Q4_0), &[Tier::Scalar, Tier::Avx2]);
            assert_eq!(
                kernels_for(Format::Q5_0),
                &[Tier::Scalar, Tier::Avx2, Tier::Avx512]
            );
            assert_eq!(
                kernels_for(Format::Q8_0),
                &[Tier::Scalar, Tier::Avx2, Tier::Avx512]
            );
        }
        #[cfg(feature = "fmt-kquant")]
        {
            assert_eq!(
                kernels_for(Format::Q4_K),
                &[Tier::Scalar, Tier::Avx2, Tier::Avx512]
            );
            assert_eq!(kernels_for(Format::Q5_K), &[Tier::Scalar, Tier::Avx2]);
            assert_eq!(kernels_for(Format::Q6_K), &[Tier::Scalar, Tier::Avx2]);
        }
        #[cfg(feature = "fmt-iq")]
        {
            assert_eq!(kernels_for(Format::IQ4_NL), &[Tier::Scalar, Tier::Avx2]);
            assert_eq!(kernels_for(Format::IQ4_XS), &[Tier::Scalar, Tier::Avx2]);
        }
        #[cfg(feature = "fmt-mxfp4")]
        {
            assert_eq!(kernels_for(Format::MXFP4), &[Tier::Scalar, Tier::Avx2]);
        }
        #[cfg(feature = "fmt-ternary")]
        {
            assert_eq!(
                kernels_for(Format::PQ2_0),
                &[Tier::Scalar, Tier::Avx2, Tier::Avx512]
            );
            assert_eq!(
                kernels_for(Format::PTQ1_0),
                &[Tier::Scalar, Tier::Avx2, Tier::Avx512]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn the_arm_table_lists_the_neon_tiers() {
        #[cfg(feature = "fmt-legacy")]
        {
            assert_eq!(
                kernels_for(Format::Q8_0),
                &[Tier::Scalar, Tier::NeonDotprod, Tier::NeonI8mm]
            );
            // Q5_0 has a NEON tile and per-row kernel, and the tile GEMM on i8mm (S2.8 F1/F2).
            assert_eq!(
                kernels_for(Format::Q5_0),
                &[Tier::Scalar, Tier::NeonDotprod, Tier::NeonI8mm]
            );
        }
        #[cfg(feature = "fmt-kquant")]
        {
            assert_eq!(
                kernels_for(Format::Q4_K),
                &[Tier::Scalar, Tier::NeonDotprod, Tier::NeonI8mm]
            );
            // Q6_K has the per-row and tile kernels, and the tile GEMM on i8mm (S2.8 F3/F2).
            assert_eq!(
                kernels_for(Format::Q6_K),
                &[Tier::Scalar, Tier::NeonDotprod, Tier::NeonI8mm]
            );
        }
        assert_eq!(kernels_for(Format::F32), &[Tier::Scalar, Tier::Neon]);
        assert_eq!(kernels_for(Format::BF16), &[Tier::Scalar, Tier::Neon]);
        // `fcvtl` is base Advanced SIMD, so F16's packed kernel is on plain NEON too.
        assert_eq!(kernels_for(Format::F16), &[Tier::Scalar, Tier::Neon]);
    }

    #[test]
    fn exact_overrides_every_fast_path() {
        let t = MatKernels {
            tiles: true,
            pq2_rows: true,
            ptq1_rows: true,
            kq_tiles: true,
            kq_i8mm: true,
            kq_zmm: true,
            fp_panels: true,
            exact: true,
        };
        for f in [Format::PQ2_0, Format::PTQ1_0, Format::Q4_K, Format::Q5_0] {
            assert_eq!(t.path(f, true, true, true, false), MatPath::PerRow);
        }
    }

    #[test]
    fn paths_follow_format_storage_and_cpu() {
        let all = MatKernels {
            tiles: true,
            pq2_rows: true,
            ptq1_rows: true,
            kq_tiles: true,
            kq_i8mm: true,
            kq_zmm: true,
            fp_panels: true,
            exact: false,
        };
        assert_eq!(
            all.path(Format::PQ2_0, true, false, false, false),
            MatPath::Tiles
        );
        assert_eq!(
            all.path(Format::PQ2_0, false, false, false, false),
            MatPath::RowsPq2
        );
        assert_eq!(
            all.path(Format::PTQ1_0, false, false, false, true),
            MatPath::RowsPtq1
        );
        // PTQ1_0 against int8 activations has no block kernel.
        assert_eq!(
            all.path(Format::PTQ1_0, false, false, false, false),
            MatPath::PerRow
        );
        // The packed float panels (S4.9f): a float matrix with a packed store takes
        // them, and one without (the pack off, a format that is not float) does not.
        assert_eq!(
            all.path(Format::F16, false, false, true, false),
            MatPath::FloatPanels
        );
        for fmt in [Format::F32, Format::F16, Format::BF16] {
            assert_eq!(
                all.path(fmt, false, false, true, false),
                MatPath::FloatPanels
            );
            assert_eq!(all.path(fmt, false, false, false, false), MatPath::PerRow);
        }
        assert!(all.fp_panels_for(Format::F32));
        assert!(!all.fp_panels_for(Format::Q8_0));
        // A repacked Q4_K matrix goes to the repacked kernel; one that is not
        // repacked goes the old way.
        assert_eq!(
            all.path(Format::Q4_K, false, true, false, false),
            MatPath::KqTiles
        );
        assert_eq!(
            all.path(Format::Q4_K, false, false, false, false),
            MatPath::PerRow
        );
        // Q5_0 and Q6_K have their own repacked tiles on aarch64 too (S2.8 F1/F3).
        assert_eq!(
            all.path(Format::Q5_0, false, true, false, false),
            MatPath::KqTiles
        );
        assert_eq!(
            all.path(Format::Q5_0, false, false, false, false),
            MatPath::PerRow
        );
        // Q8_0's repacked tile: the tied output head and the other Q8_0 matrices.
        assert_eq!(
            all.path(Format::Q8_0, false, true, false, false),
            MatPath::KqTiles
        );
        assert_eq!(
            all.path(Format::Q8_0, false, false, false, false),
            MatPath::PerRow
        );
        // Q6_K's tile is NEON-only: on x86 its GEMV loses to the row kernel at
        // one row, so decision 9 keeps the row path there. Q5_K's tile is x86.
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(
            all.path(Format::Q6_K, false, true, false, false),
            MatPath::PerRow
        );
        #[cfg(target_arch = "aarch64")]
        assert_eq!(
            all.path(Format::Q6_K, false, true, false, false),
            MatPath::KqTiles
        );
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(
            all.path(Format::Q5_K, false, true, false, false),
            MatPath::KqTiles
        );
        #[cfg(target_arch = "aarch64")]
        assert_eq!(
            all.path(Format::Q5_K, false, true, false, false),
            MatPath::PerRow
        );
        let avx2 = MatKernels {
            tiles: false,
            pq2_rows: true,
            ptq1_rows: false,
            kq_tiles: true,
            kq_i8mm: false,
            kq_zmm: false,
            fp_panels: true,
            exact: false,
        };
        // A tiled matrix on a CPU without the tile kernel cannot happen (no repack), but stays correct.
        assert_eq!(
            avx2.path(Format::PQ2_0, true, false, false, false),
            MatPath::RowsPq2
        );
        assert_eq!(
            avx2.path(Format::PTQ1_0, false, false, false, true),
            MatPath::PerRow
        );
        let scalar = MatKernels {
            tiles: false,
            pq2_rows: false,
            ptq1_rows: false,
            kq_tiles: false,
            kq_i8mm: false,
            kq_zmm: false,
            fp_panels: false,
            exact: false,
        };
        assert_eq!(
            scalar.path(Format::PQ2_0, false, false, false, false),
            MatPath::PerRow
        );
        // Without the repacked kernel, a repacked matrix still goes the old way.
        assert_eq!(
            scalar.path(Format::Q4_K, false, true, false, false),
            MatPath::PerRow
        );
        // A table without the float pack keeps the row kernel even for a matrix
        // whose packed panels are in the store.
        assert_eq!(
            scalar.path(Format::F16, false, false, true, false),
            MatPath::PerRow
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "fmt-legacy"))]
    fn the_q8_0_tile_takes_the_smmla_body_on_the_i8mm_tier_only() {
        use crate::kq::{dot_q8_08_ref, repack_q8_08, Q8_08};

        let n_blk = 3usize;
        let n = n_blk * 32;
        let mut tile = vec![0u8; n_blk * Q8_08::SIZE];
        for t in 0..n_blk {
            let rows: Vec<Vec<u8>> = (0..8)
                .map(|r| {
                    let mut w = vec![0u8; 34];
                    let d = crate::quant::f32_to_f16(0.01 + r as f32 * 0.001);
                    w[..2].copy_from_slice(&d.to_le_bytes());
                    for (k, c) in w[2..].iter_mut().enumerate() {
                        *c = (k * 11 + r * 7 + t * 3) as u8;
                    }
                    w
                })
                .collect();
            let src: Vec<&[u8]> = rows.iter().map(|r| &r[..34]).collect();
            repack_q8_08(&src, &mut tile[t * Q8_08::SIZE..]);
        }
        let acts: Vec<QuantAct> = (0..8)
            .map(|b| {
                let x: Vec<f32> = (0..n)
                    .map(|i| ((i * 7 + b * 13) % 23) as f32 - 11.0)
                    .collect();
                let mut a = QuantAct::with_capacity(n);
                a.fill(&x);
                a
            })
            .collect();
        let table = |kq_i8mm: bool| MatKernels {
            tiles: false,
            pq2_rows: false,
            ptq1_rows: false,
            kq_tiles: true,
            kq_i8mm,
            kq_zmm: false,
            fp_panels: false,
            exact: false,
        };
        // Below the rung the `sdot` body runs; on it, the `smmla` body. Both are the
        // specification's bits, and only the i8mm table calls the new body.
        let rungs: &[bool] = if crate::cpu::have_i8mm() {
            &[false, true]
        } else {
            &[false]
        };
        for &i8mm in rungs {
            crate::neon::Q8_08_I8MM_CALLS.with(|c| c.set(0));
            crate::neon::Q8_08_SDOT_CALLS.with(|c| c.set(0));
            let t = table(i8mm);
            let mut got = vec![0.0f32; acts.len() * 8];
            // The table's wrapper checks its own preconditions: the tile holds the
            // blocks, the activation the rows, and the output the floats.
            t.kq_tiles_q8_0(&tile, n_blk, &acts, acts.len(), &mut got);
            let smmla = crate::neon::Q8_08_I8MM_CALLS.with(|c| c.get());
            let sdot = crate::neon::Q8_08_SDOT_CALLS.with(|c| c.get());
            if i8mm {
                assert!(smmla > 0, "the i8mm tier did not take the smmla body");
                assert_eq!(sdot, 0, "the i8mm tier also ran the sdot body");
            } else {
                assert_eq!(smmla, 0, "a tier below i8mm took the smmla body");
                assert!(sdot > 0, "the tile below i8mm did not take the sdot body");
            }
            for (b, a) in acts.iter().enumerate() {
                for r in 0..8 {
                    let want = dot_q8_08_ref(&tile, n_blk, r, a);
                    assert_eq!(
                        got[b * 8 + r].to_bits(),
                        want.to_bits(),
                        "i8mm={i8mm} b={b} r={r}: {} vs {want}",
                        got[b * 8 + r]
                    );
                }
            }
        }
        // The table this CPU builds follows the rung, and Q8_0 lists it.
        let k = MatKernels::get();
        assert_eq!(k.kq_i8mm, k.kq_tiles && crate::cpu::have_i8mm());
        assert!(MatKernels::kernels_for(Format::Q8_0).contains(&Tier::NeonI8mm));
    }
}
