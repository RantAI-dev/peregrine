//! Weight formats the kernels understand, and the f16 conversion every block scale needs.

/// Weight encodings this engine can read.
///
/// Named for the GGML types they mirror rather than renamed to satisfy Rust casing:
/// these strings appear in model cards, in llama.cpp's source and in this project's own
/// models, and a reader checking a layout against `ggml-common.h` should not have
/// to translate. The set is deliberately closed — an unsupported type is a load
/// error, never a silent fallback to a format with different numerics.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    F32,
    F16,
    Q4_0,
    Q5_0,
    Q8_0,
    /// GGML's 2-bit k-quant format, 256 weights in 84 bytes (`u8 scales[16], u8 qs[64], f16 d, f16 dmin`).
    Q2_K,
    /// GGML's 3-bit k-quant format, 256 weights in 110 bytes (`u8 hmask[32], u8 qs[64], u8 scales[12], f16 d`).
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    /// GGML's non-linear 4-bit format with 16-entry lookup table, 32 weights in 18 bytes (`f16 d, 16 bytes qs`).
    IQ4_NL,
    /// GGML's non-linear 4-bit format with 6-bit sub-block scales, 256 weights in 136 bytes (`f16 d, u16 scales_h, u8 scales_l[4], u8 qs[128]`).
    IQ4_XS,
    /// GGML's MXFP4 format (OCP microscaling FP4), 32 weights in 17 bytes (u8 e8m0 scale, 16 bytes qs).
    MXFP4,
    // PrismML's ternary formats (Bonsai 2), group size 128. PQ2_0: f16 scale, then 32 bytes of 2-bit
    // codes, weight j in bits 2(j%4) of byte j/4, value code − 1. PTQ1_0: 24 + 2 bytes of base-3 trits, then f16.
    PQ2_0,
    PTQ1_0,
    /// bfloat16 (Bonsai 2 27B's small `ssm_alpha`/`ssm_beta`, decoded to f32 at load).
    BF16,
}

impl Format {
    /// GGML type id as stored in the tensor directory.
    pub fn from_id(id: u32) -> Option<Format> {
        Some(match id {
            0 => Format::F32,
            1 => Format::F16,
            2 => Format::Q4_0,
            6 => Format::Q5_0,
            8 => Format::Q8_0,
            10 => Format::Q2_K,
            11 => Format::Q3_K,
            12 => Format::Q4_K,
            13 => Format::Q5_K,
            14 => Format::Q6_K,
            20 => Format::IQ4_NL,
            23 => Format::IQ4_XS,
            30 => Format::BF16,
            39 => Format::MXFP4,
            142 => Format::PQ2_0,
            143 => Format::PTQ1_0,
            _ => return None,
        })
    }

    /// Weights per block.
    pub const fn block(self) -> usize {
        match self {
            Format::F32 | Format::F16 | Format::BF16 => 1,
            Format::Q4_0 | Format::Q5_0 | Format::Q8_0 | Format::IQ4_NL | Format::MXFP4 => 32,
            Format::Q2_K
            | Format::Q3_K
            | Format::Q4_K
            | Format::Q5_K
            | Format::Q6_K
            | Format::IQ4_XS => 256,
            Format::PQ2_0 | Format::PTQ1_0 => 128,
        }
    }

    /// Bytes per block. These are the `static_assert`ed sizes in `ggml-common.h`.
    pub const fn block_bytes(self) -> usize {
        match self {
            Format::F32 => 4,
            Format::F16 => 2,
            Format::BF16 => 2,
            Format::Q4_0 => 18,    // f16 d + u8 qs[16]
            Format::Q5_0 => 22,    // f16 d + u8 qh[4] + u8 qs[16]
            Format::Q8_0 => 34,    // f16 d + i8 qs[32]
            Format::Q2_K => 84,    // u8 scales[16] + u8 qs[64] + f16 d + f16 dmin
            Format::Q3_K => 110,   // u8 hmask[32] + u8 qs[64] + u8 scales[12] + f16 d
            Format::Q4_K => 144,   // f16 d + f16 dmin + u8 scales[12] + u8 qs[128]
            Format::Q5_K => 176,   // f16 d + f16 dmin + u8 scales[12] + u8 qh[32] + u8 qs[128]
            Format::Q6_K => 210,   // u8 ql[128] + u8 qh[64] + i8 scales[16] + f16 d
            Format::IQ4_NL => 18,  // f16 d + u8 qs[16]
            Format::IQ4_XS => 136, // f16 d + u16 scales_h + u8 scales_l[4] + u8 qs[128]
            Format::MXFP4 => 17,   // u8 e + u8 qs[16]
            Format::PQ2_0 => 34,   // f16 d + u8 qs[32]
            Format::PTQ1_0 => 28,  // u8 qs[24] + u8 qh[2] + f16 d
        }
    }

    /// Bits per weight as actually stored, including the block's own scale metadata.
    /// This is the `bpw` of the spec's Equation (3), so it must count the overhead:
    /// quoting 4.0 for Q4_K when the block costs 4.5 would understate the floor.
    pub fn bpw(self) -> f64 {
        (self.block_bytes() * 8) as f64 / self.block() as f64
    }

    pub fn label(self) -> &'static str {
        match self {
            Format::F32 => "F32",
            Format::F16 => "F16",
            Format::Q4_0 => "Q4_0",
            Format::Q5_0 => "Q5_0",
            Format::Q8_0 => "Q8_0",
            Format::Q2_K => "Q2_K",
            Format::Q3_K => "Q3_K",
            Format::Q4_K => "Q4_K",
            Format::Q5_K => "Q5_K",
            Format::Q6_K => "Q6_K",
            Format::IQ4_NL => "IQ4_NL",
            Format::IQ4_XS => "IQ4_XS",
            Format::MXFP4 => "MXFP4",
            Format::PQ2_0 => "PQ2_0",
            Format::PTQ1_0 => "PTQ1_0",
            Format::BF16 => "BF16",
        }
    }

    /// The Cargo feature that enables kernels and runtime support for this format,
    /// or None if the format is always enabled (F32, F16, BF16).
    pub const fn required_feature(self) -> Option<&'static str> {
        match self {
            Format::F32 | Format::F16 | Format::BF16 => None,
            Format::Q4_0 | Format::Q5_0 | Format::Q8_0 => Some("fmt-legacy"),
            Format::Q2_K | Format::Q3_K | Format::Q4_K | Format::Q5_K | Format::Q6_K => {
                Some("fmt-kquant")
            }
            Format::IQ4_NL | Format::IQ4_XS => Some("fmt-iq"),
            Format::MXFP4 => Some("fmt-mxfp4"),
            Format::PQ2_0 | Format::PTQ1_0 => Some("fmt-ternary"),
        }
    }

    /// Whether this build includes kernels and support for this format.
    pub const fn is_supported(self) -> bool {
        match self {
            Format::F32 | Format::F16 | Format::BF16 => true,
            #[cfg(feature = "fmt-legacy")]
            Format::Q4_0 | Format::Q5_0 | Format::Q8_0 => true,
            #[cfg(feature = "fmt-kquant")]
            Format::Q2_K | Format::Q3_K | Format::Q4_K | Format::Q5_K | Format::Q6_K => true,
            #[cfg(feature = "fmt-iq")]
            Format::IQ4_NL | Format::IQ4_XS => true,
            #[cfg(feature = "fmt-mxfp4")]
            Format::MXFP4 => true,
            #[cfg(feature = "fmt-ternary")]
            Format::PQ2_0 | Format::PTQ1_0 => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Every weight format known to the engine, in canonical display order.
    pub const ALL: &[Format] = &[
        Format::F32,
        Format::F16,
        Format::BF16,
        Format::Q4_0,
        Format::Q5_0,
        Format::Q8_0,
        Format::Q2_K,
        Format::Q3_K,
        Format::Q4_K,
        Format::Q5_K,
        Format::Q6_K,
        Format::IQ4_NL,
        Format::IQ4_XS,
        Format::MXFP4,
        Format::PQ2_0,
        Format::PTQ1_0,
    ];

    /// Formats supported in this build (whose required feature is enabled).
    pub fn supported() -> Vec<Format> {
        Self::ALL
            .iter()
            .copied()
            .filter(|f| f.is_supported())
            .collect()
    }
}

/// IEEE-754 binary16 to binary32.
///
/// Written out rather than pulled from a crate because it is on the hot path for
/// every block scale in the model and the conversion is twenty lines. Handles
/// subnormals and infinities: a subnormal block scale is rare but a model that
/// contains one must not decode to zero.
#[inline(always)]
pub fn f16(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x03ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: value is `man · 2^-24`. Let `k` be the index of the highest
            // set bit of `man`, so the normalised exponent is `k - 24` and the
            // mantissa field is the remainder shifted to bit 23. With
            // `shift = 10 - k` (what it takes to bring bit k up to bit 10):
            //   exponent field = (k - 24) + 127 = 113 - shift
            //   mantissa field = man << (23 - k) = man << (13 + shift), masked
            // The mask drops the implicit bit that `man`'s own top bit becomes.
            let shift = man.leading_zeros() - 21; // man < 2^10, so this is 0..=9
            sign | ((113 - shift) << 23) | ((man << (shift + 13)) & 0x007f_ffff)
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Borrow an aligned `&[f32]` from raw tensor bytes if alignment and length match.
#[inline]
pub fn cast_f32_slice(b: &[u8], n: usize) -> Option<&[f32]> {
    if b.as_ptr() as usize % align_of::<f32>() == 0 && b.len() >= n * 4 {
        // SAFETY: 4-byte aligned, buffer holds at least n floats, and any 4 bytes represent a valid f32.
        Some(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, n) })
    } else {
        None
    }
}

/// Decode an 8-bit power-of-two exponent (e8m0) to f32.
///
/// Follows `ggml_e8m0_to_fp32` in `ggml-impl.h`:
/// - `x == 0`: 2^(-127) (represented as subnormal float with bits `0x0040_0000`)
/// - `x > 0`: 2^(x - 127) (represented as normalized float with bits `(x as u32) << 23`)
#[inline(always)]
pub const fn e8m0_to_f32(x: u8) -> f32 {
    let bits = if x == 0 {
        0x0040_0000
    } else {
        (x as u32) << 23
    };
    f32::from_bits(bits)
}

/// Decode an 8-bit power-of-two exponent (e8m0) halved (divided by 2.0) to f32.
///
/// Follows `ggml_e8m0_to_fp32_half` in `ggml-impl.h`:
/// Because `kvalues_mxfp4` contains doubled values (2 * E2M1), scales in `dequant_row`
/// and `dot_scalar` use this half-scale to compensate.
/// - `x == 0`: 2^(-128) (subnormal, bits `0x0020_0000`)
/// - `x == 1`: 2^(-127) (subnormal, bits `0x0040_0000`)
/// - `x >= 2`: 2^(x - 128) (normalized, bits `((x - 1) as u32) << 23`)
#[inline(always)]
pub const fn e8m0_to_f32_half(x: u8) -> f32 {
    let bits = if x < 2 {
        0x0020_0000 << x
    } else {
        ((x - 1) as u32) << 23
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Format::ALL` drives `peregrine support` and the README table: every format the GGUF reader
    /// knows must be in it, or a new format silently goes missing from the support list.
    #[test]
    fn all_lists_every_format_the_reader_knows() {
        for id in 0..=u16::MAX as u32 {
            if let Some(f) = Format::from_id(id) {
                assert!(
                    Format::ALL.contains(&f),
                    "{} (id {id}) is missing from Format::ALL",
                    f.label()
                );
            }
        }
    }
}
