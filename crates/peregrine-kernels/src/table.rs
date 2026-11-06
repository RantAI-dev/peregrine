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
}

/// The matrix kernels this process runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatKernels {
    tiles: bool,
    pq2_rows: bool,
    ptq1_rows: bool,
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
        {
            MatKernels {
                tiles: crate::tern::have_zmm(),
                pq2_rows: crate::simd::have_batched(),
                ptq1_rows: crate::tern::have_ptq1_zmm(),
                exact: crate::quant::exact_ternary(),
            }
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "fmt-ternary")))]
        {
            MatKernels {
                tiles: false,
                pq2_rows: false,
                ptq1_rows: false,
                exact: crate::quant::exact_ternary(),
            }
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

    /// The path for a matrix in `fmt`, repacked into tiles or not, against activations in the PTQ1 layout or not.
    #[inline]
    pub fn path(&self, fmt: Format, tiled: bool, act_ptq1: bool) -> MatPath {
        if self.exact {
            // The f32 reference lives in `dot_batch`.
            MatPath::PerRow
        } else if tiled && self.tiles {
            MatPath::Tiles
        } else if fmt == Format::PTQ1_0 && act_ptq1 && self.ptq1_rows {
            MatPath::RowsPtq1
        } else if fmt == Format::PQ2_0 && self.pq2_rows {
            MatPath::RowsPq2
        } else {
            MatPath::PerRow
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
        #[cfg(not(all(target_arch = "x86_64", feature = "fmt-ternary")))]
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
    match fmt {
        Format::F32 | Format::F16 | Format::BF16 => &[Tier::Scalar],
        #[cfg(feature = "fmt-legacy")]
        Format::Q4_0 | Format::Q5_0 | Format::Q8_0 => &[Tier::Scalar, Tier::Avx2],
        #[cfg(feature = "fmt-kquant")]
        Format::Q4_K | Format::Q5_K | Format::Q6_K => &[Tier::Scalar, Tier::Avx2],
        #[cfg(feature = "fmt-iq")]
        Format::IQ4_NL | Format::IQ4_XS => &[Tier::Scalar, Tier::Avx2],
        #[cfg(feature = "fmt-mxfp4")]
        Format::MXFP4 => &[Tier::Scalar, Tier::Avx2],
        #[cfg(feature = "fmt-ternary")]
        Format::PQ2_0 | Format::PTQ1_0 => &[Tier::Scalar, Tier::Avx2, Tier::Avx512],
        #[allow(unreachable_patterns)]
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernels_for_tiers() {
        assert_eq!(kernels_for(Format::F32), &[Tier::Scalar]);
        assert_eq!(kernels_for(Format::F16), &[Tier::Scalar]);
        assert_eq!(kernels_for(Format::BF16), &[Tier::Scalar]);
        #[cfg(feature = "fmt-legacy")]
        {
            assert_eq!(kernels_for(Format::Q4_0), &[Tier::Scalar, Tier::Avx2]);
            assert_eq!(kernels_for(Format::Q5_0), &[Tier::Scalar, Tier::Avx2]);
            assert_eq!(kernels_for(Format::Q8_0), &[Tier::Scalar, Tier::Avx2]);
        }
        #[cfg(feature = "fmt-kquant")]
        {
            assert_eq!(kernels_for(Format::Q4_K), &[Tier::Scalar, Tier::Avx2]);
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
    fn exact_overrides_every_fast_path() {
        let t = MatKernels {
            tiles: true,
            pq2_rows: true,
            ptq1_rows: true,
            exact: true,
        };
        for f in [Format::PQ2_0, Format::PTQ1_0, Format::Q4_K] {
            assert_eq!(t.path(f, true, true), MatPath::PerRow);
        }
    }

    #[test]
    fn paths_follow_format_storage_and_cpu() {
        let all = MatKernels {
            tiles: true,
            pq2_rows: true,
            ptq1_rows: true,
            exact: false,
        };
        assert_eq!(all.path(Format::PQ2_0, true, false), MatPath::Tiles);
        assert_eq!(all.path(Format::PQ2_0, false, false), MatPath::RowsPq2);
        assert_eq!(all.path(Format::PTQ1_0, false, true), MatPath::RowsPtq1);
        // PTQ1_0 against int8 activations has no block kernel.
        assert_eq!(all.path(Format::PTQ1_0, false, false), MatPath::PerRow);
        assert_eq!(all.path(Format::Q4_K, false, false), MatPath::PerRow);
        let avx2 = MatKernels {
            tiles: false,
            pq2_rows: true,
            ptq1_rows: false,
            exact: false,
        };
        // A tiled matrix on a CPU without the tile kernel cannot happen (no repack), but stays correct.
        assert_eq!(avx2.path(Format::PQ2_0, true, false), MatPath::RowsPq2);
        assert_eq!(avx2.path(Format::PTQ1_0, false, true), MatPath::PerRow);
        let scalar = MatKernels {
            tiles: false,
            pq2_rows: false,
            ptq1_rows: false,
            exact: false,
        };
        assert_eq!(scalar.path(Format::PQ2_0, false, false), MatPath::PerRow);
    }
}
