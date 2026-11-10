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
        MatKernels {
            tiles,
            pq2_rows,
            ptq1_rows,
            kq_tiles,
            kq_i8mm,
            kq_zmm,
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

    /// Whether `fmt` has a repacked tile kernel on this CPU.
    ///
    /// The tile store holds a Q4_K, Q5_0 and Q8_0 tile on every tier with a repacked kernel:
    /// Q4_K on both architectures, Q5_0 and Q8_0 on x86 and (S2.8 F1) on aarch64. The aarch64
    /// tier has Q4_K, Q5_0 and Q8_0 here; Q6_K follows (F3). A format without a kernel keeps
    /// the per-row path even when the store holds its tiles.
    #[inline]
    pub fn kq_tiles_for(&self, fmt: Format) -> bool {
        if !self.kq_tiles {
            return false;
        }
        match fmt {
            #[cfg(feature = "fmt-kquant")]
            Format::Q4_K => true,
            // Q5_0 has a repacked tile on both kernel architectures (S3.4 on x86, S2.8 F1
            // on NEON); Q6_K has one on aarch64 (S2.8 F3). Every other format keeps the
            // per-row path even when the store holds its tiles.
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
            #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
            Format::Q6_K => true,
            _ => false,
        }
    }

    /// The path for a matrix in `fmt`, repacked into tiles or not, against activations in the PTQ1 layout or not.
    ///
    /// `kq` says whether the matrix carries repacked k-quant tiles (`TensorInfo::kq_off`).
    #[inline]
    pub fn path(&self, fmt: Format, tiled: bool, kq: bool, act_ptq1: bool) -> MatPath {
        if self.exact {
            // The f32 reference lives in `dot_batch`.
            MatPath::PerRow
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
        #[cfg(not(all(target_arch = "aarch64", feature = "fmt-kquant")))]
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
        #[cfg(all(target_arch = "aarch64", feature = "fmt-kquant"))]
        // SAFETY: `kq_tiles` is `have_dotprod()`; the tile holds `n_sb` super-blocks
        // and the plain activation rows the caller says.
        unsafe {
            crate::neon::q6k8_gemv(tile, n_sb, acts, nb, out)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "fmt-kquant")))]
        {
            let _ = (tile, n_sb, acts, nb, out);
            unreachable!("no repacked Q6_K kernel on this architecture")
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
        // SAFETY: `kq_tiles` is `have_dotprod()`; the tile holds `n_blk` blocks and the
        // activation rows the caller says.
        unsafe {
            crate::neon::q8_08_gemm(tile, n_blk, acts, nb, out)
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
    // `neon-i8mm` is the same kernels with the tile GEMM on `smmla` (F2); Q8_0 has no tile, so
    // it stays on the `sdot` batch kernel and is not listed there.
    #[cfg(target_arch = "aarch64")]
    let tiers: &[Tier] = match fmt {
        #[cfg(feature = "fmt-legacy")]
        Format::Q8_0 => &[Tier::Scalar, Tier::NeonDotprod],
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
                &[Tier::Scalar, Tier::NeonDotprod]
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
        assert_eq!(kernels_for(Format::F16), &[Tier::Scalar]);
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
            exact: true,
        };
        for f in [Format::PQ2_0, Format::PTQ1_0, Format::Q4_K, Format::Q5_0] {
            assert_eq!(t.path(f, true, true, true), MatPath::PerRow);
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
            exact: false,
        };
        assert_eq!(all.path(Format::PQ2_0, true, false, false), MatPath::Tiles);
        assert_eq!(
            all.path(Format::PQ2_0, false, false, false),
            MatPath::RowsPq2
        );
        assert_eq!(
            all.path(Format::PTQ1_0, false, false, true),
            MatPath::RowsPtq1
        );
        // PTQ1_0 against int8 activations has no block kernel.
        assert_eq!(
            all.path(Format::PTQ1_0, false, false, false),
            MatPath::PerRow
        );
        // A repacked Q4_K matrix goes to the repacked kernel; one that is not
        // repacked goes the old way.
        assert_eq!(all.path(Format::Q4_K, false, true, false), MatPath::KqTiles);
        assert_eq!(all.path(Format::Q4_K, false, false, false), MatPath::PerRow);
        // Q5_0 and Q6_K have their own repacked tiles on aarch64 too (S2.8 F1/F3).
        assert_eq!(all.path(Format::Q5_0, false, true, false), MatPath::KqTiles);
        assert_eq!(all.path(Format::Q5_0, false, false, false), MatPath::PerRow);
        // Q8_0's repacked tile: the tied output head and the other Q8_0 matrices.
        assert_eq!(all.path(Format::Q8_0, false, true, false), MatPath::KqTiles);
        assert_eq!(all.path(Format::Q8_0, false, false, false), MatPath::PerRow);
        // Q6_K's tile is NEON-only for now: x86 keeps the per-row kernel (S3.4 PR 4).
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(all.path(Format::Q6_K, false, true, false), MatPath::PerRow);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(all.path(Format::Q6_K, false, true, false), MatPath::KqTiles);
        let avx2 = MatKernels {
            tiles: false,
            pq2_rows: true,
            ptq1_rows: false,
            kq_tiles: true,
            kq_i8mm: false,
            kq_zmm: false,
            exact: false,
        };
        // A tiled matrix on a CPU without the tile kernel cannot happen (no repack), but stays correct.
        assert_eq!(
            avx2.path(Format::PQ2_0, true, false, false),
            MatPath::RowsPq2
        );
        assert_eq!(
            avx2.path(Format::PTQ1_0, false, false, true),
            MatPath::PerRow
        );
        let scalar = MatKernels {
            tiles: false,
            pq2_rows: false,
            ptq1_rows: false,
            kq_tiles: false,
            kq_i8mm: false,
            kq_zmm: false,
            exact: false,
        };
        assert_eq!(
            scalar.path(Format::PQ2_0, false, false, false),
            MatPath::PerRow
        );
        // Without the repacked kernel, a repacked matrix still goes the old way.
        assert_eq!(
            scalar.path(Format::Q4_K, false, true, false),
            MatPath::PerRow
        );
    }
}
