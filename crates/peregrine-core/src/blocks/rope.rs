//! Rotary position embedding and precomputed table construction.

/// YaRN's parameters, as llama.cpp v0.9.0 computes them.
///
/// A YaRN-scaled model interpolates between the plain rotary frequency and one
/// divided by the scaling factor, with a linear ramp between the two over the
/// middle frequencies, and scales the angles' amplitude by a factor of its own.
/// llama.cpp keeps the three numbers here:
/// the frequency scale (`src/llama-context.cpp:285`, `factor = 1/freq_scale`),
/// the extrapolation factor (`src/llama-context.cpp:276-277`: 1 for YaRN, 0
/// otherwise) and the magnitude scale (`src/llama-context.cpp:280-317` and
/// `ggml/src/ggml-cpu/ops.cpp:6184`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Yarn {
    /// The frequency scale: `1 / factor`, the divisor of the interpolated angles.
    pub freq_scale: f32,
    /// The extrapolation factor: 1 for YaRN (the ramp mixes the two frequencies),
    /// 0 for plain linear scaling (every angle is interpolated).
    pub ext_factor: f32,
    /// The magnitude scale the cosines and the sines carry.
    pub mscale: f32,
    /// The ramp's two ends: the correction dimensions
    /// (`ggml_rope_yarn_corr_dims`, `ggml/src/ggml.c:4534-4542`).
    pub corr_dims: [f32; 2],
}

/// YaRN's parameters for a file that asks for it.
///
/// llama.cpp v0.9.0 computes them in two places, and this follows both:
///
/// * the correction dimensions, `ggml_rope_yarn_corr_dims`
///   (`ggml/src/ggml.c:4530-4542`):
///   `dim(n) = rope_dim · ln(n_ctx_orig / (n · 2π)) / (2 · ln(base))`, the start
///   is that with `beta_fast` rounded down and the end that with `beta_slow`
///   rounded up;
/// * the magnitude scale, `llama-context.cpp:280-317`: the attention factor
///   (`get_mscale(factor, 1)` cancelled by `1/(1 + 0.1·ln(factor))`, then the
///   file's own `rope.scaling.attn_factor`) times the factor the rope applies
///   itself (`ggml/src/ggml-cpu/ops.cpp:6184`).
///
/// `log_mul` is the file's `rope.scaling.yarn_log_multiplier`; llama.cpp reads it
/// for the magnitude scale (`src/llama-model.cpp:2635`), and no published gpt-oss
/// file carries it.
#[allow(clippy::too_many_arguments)]
pub fn yarn_params(
    rope_dim: usize,
    rope_base: f32,
    factor: f32,
    n_ctx_orig: usize,
    ext_factor: f32,
    attn_factor: f32,
    log_mul: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Yarn {
    let freq_scale = 1.0 / factor;
    // The correction dimensions (`ggml/src/ggml.c:4530-4542`).
    let corr_dim = |n_rot: f32| -> f32 {
        rope_dim as f32
            * crate::vmath::det_ln(n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI))
            / (2.0 * crate::vmath::det_ln(rope_base))
    };
    let start = corr_dim(beta_fast).floor();
    let end = corr_dim(beta_slow).ceil();
    let corr_dims = [start.max(0.0), end.min(rope_dim as f32 - 1.0)];
    // The magnitude scale (`src/llama-context.cpp:280-317`).
    let get_mscale = |scale: f32, m: f32| -> f32 {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * m * crate::vmath::det_ln(scale) + 1.0
        }
    };
    let mut yarn_attn = if log_mul != 0.0 {
        get_mscale(factor, 1.0) / get_mscale(factor, log_mul)
    } else {
        get_mscale(factor, 1.0)
    };
    // The factor the rope applies itself is cancelled in the attention factor and
    // applied to the angles instead (`src/llama-context.cpp:309-314`).
    yarn_attn *= 1.0 / (1.0 + 0.1 * crate::vmath::det_ln(factor));
    yarn_attn *= attn_factor;
    let mscale = if ext_factor != 0.0 {
        yarn_attn * (1.0 + 0.1 * crate::vmath::det_ln(1.0 / freq_scale))
    } else {
        yarn_attn
    };
    Yarn {
        freq_scale,
        ext_factor,
        mscale,
        corr_dims,
    }
}

/// The ramp between the extrapolated and the interpolated frequency, at the
/// rotary pair `i` (`ggml/src/ggml-cpu/ops.cpp:6164-6169`):
/// 1 below the first correction dimension, 0 above the second, linear between.
fn yarn_ramp(corr_dims: [f32; 2], i: usize) -> f32 {
    let (low, high) = (corr_dims[0], corr_dims[1]);
    let y = (i as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// Precompute the RoPE sin/cos table for positions `0..max_ctx`.
pub fn build_rope_table(
    rope_dim: usize,
    rope_base: f32,
    max_ctx: usize,
    rope_factors: Option<&[f32]>,
) -> Vec<f32> {
    let half = rope_dim / 2;
    let mut rope = vec![0.0f32; max_ctx * half * 2];
    // `powf` once per frequency, not once per (position, frequency): for a 32K
    // context that was millions of calls per session open. Same f64 expressions,
    // so the table is bit-identical.
    let freqs: Vec<(f64, f64)> = (0..half)
        .map(|i| {
            // With factors (Llama 3.x), ggml divides the angle by `factor[i]`.
            let freq =
                crate::vmath::det_pow_f64(rope_base as f64, -((2 * i) as f64) / rope_dim as f64);
            let factor = rope_factors.and_then(|f| f.get(i)).copied().unwrap_or(1.0) as f64;
            (freq, factor)
        })
        .collect();
    for pos in 0..max_ctx {
        for (i, &(freq, factor)) in freqs.iter().enumerate() {
            let theta = pos as f64 * freq / factor;
            let (sin, cos) = crate::vmath::det_sin_cos(theta);
            rope[(pos * half + i) * 2] = cos as f32;
            rope[(pos * half + i) * 2 + 1] = sin as f32;
        }
    }
    rope
}

/// The RoPE table of a YaRN-scaled model.
///
/// The angle of a rotary pair is the extrapolated one (`pos · base^(-2i/d)`,
/// divided by the file's per-frequency factor when it has one) or the
/// interpolated one (the same angle divided by the scaling factor), mixed by
/// the ramp; the cosines and the sines carry the magnitude scale
/// (`ggml/src/ggml-cpu/ops.cpp:6164-6187`, `rope_yarn`).
pub fn build_rope_table_yarn(
    rope_dim: usize,
    rope_base: f32,
    max_ctx: usize,
    rope_factors: Option<&[f32]>,
    yarn: &Yarn,
) -> Vec<f32> {
    let half = rope_dim / 2;
    let mut rope = vec![0.0f32; max_ctx * half * 2];
    let freqs: Vec<(f64, f64)> = (0..half)
        .map(|i| {
            let freq =
                crate::vmath::det_pow_f64(rope_base as f64, -((2 * i) as f64) / rope_dim as f64);
            let factor = rope_factors.and_then(|f| f.get(i)).copied().unwrap_or(1.0) as f64;
            (freq, factor)
        })
        .collect();
    for pos in 0..max_ctx {
        for (i, &(freq, factor)) in freqs.iter().enumerate() {
            let theta_extrap = pos as f64 * freq / factor;
            let theta_interp = yarn.freq_scale as f64 * theta_extrap;
            let ramp_mix = yarn_ramp(yarn.corr_dims, i) as f64 * yarn.ext_factor as f64;
            let theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
            let (sin, cos) = crate::vmath::det_sin_cos(theta);
            rope[(pos * half + i) * 2] = (cos as f32) * yarn.mscale;
            rope[(pos * half + i) * 2 + 1] = (sin as f32) * yarn.mscale;
        }
    }
    rope
}

/// Rotary embedding, ggml's "norm" layout: consecutive pairs `(2i, 2i+1)` are
/// rotated by `pos · base^(-2i/rope_dim)`. NeoX-style models pair `i` with
/// `i + dim/2` instead; llama does not, and using the wrong pairing produces a
/// model that is coherent for a few tokens and then drifts.
#[inline]
pub fn rope(
    rope: &[f32],
    v: &mut [f32],
    n_heads: usize,
    pos: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_neox: bool,
) {
    let hd = head_dim;
    let half = rope_dim / 2;
    let table = &rope[pos * half * 2..(pos + 1) * half * 2];
    for h in 0..n_heads {
        let head = &mut v[h * hd..(h + 1) * hd];
        if rope_neox {
            // Qwen2 (and GPT-NeoX): the two halves of the rotary span are paired.
            for i in 0..half {
                let (c, s) = (table[i * 2], table[i * 2 + 1]);
                let (x0, x1) = (head[i], head[i + half]);
                head[i] = x0 * c - x1 * s;
                head[i + half] = x0 * s + x1 * c;
            }
        } else {
            for i in 0..half {
                let (c, s) = (table[i * 2], table[i * 2 + 1]);
                let (x0, x1) = (head[2 * i], head[2 * i + 1]);
                head[2 * i] = x0 * c - x1 * s;
                head[2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The correction dimensions of gpt-oss's own geometry: a 64-wide rotary
    /// span, a base of 150 000 and a 4096-token original context.
    ///
    /// llama.cpp: `ggml_rope_yarn_corr_dims(64, 4096, 150000, 32, 1)`
    /// (`ggml/src/ggml.c:4534-4542`).
    #[test]
    fn the_correction_dimensions_follow_llama_cpp() {
        let y = yarn_params(64, 150000.0, 32.0, 4096, 1.0, 1.0, 0.0, 32.0, 1.0);
        assert_eq!(y.freq_scale, 1.0 / 32.0);
        // `64·ln(4096/(32·2π))/(2·ln(150000))` = 8.09…, rounded down.
        assert_eq!(y.corr_dims[0], 8.0);
        // `64·ln(4096/(1·2π))/(2·ln(150000))` = 17.40…, rounded up.
        assert_eq!(y.corr_dims[1], 18.0);
    }

    /// The magnitude scale of gpt-oss: llama.cpp cancels its own factor in the
    /// attention factor (`src/llama-context.cpp:309-314`) and applies it to the
    /// angles instead (`ggml/src/ggml-cpu/ops.cpp:6184`), so it is
    /// `1 + 0.1·ln(32)` = 1.34657….
    #[test]
    fn the_magnitude_scale_follows_llama_cpp() {
        let y = yarn_params(64, 150000.0, 32.0, 4096, 1.0, 1.0, 0.0, 32.0, 1.0);
        let want = 1.0 + 0.1 * 32f32.ln();
        assert!((y.mscale - want).abs() < 1e-6, "{} vs {want}", y.mscale);
    }

    /// The ramp: 1 below the first correction dimension, 0 above the second,
    /// linear between (`ggml/src/ggml-cpu/ops.cpp:6164-6169`).
    #[test]
    fn the_ramp_runs_from_the_first_to_the_second_correction_dimension() {
        let dims = [8.0, 18.0];
        assert_eq!(yarn_ramp(dims, 0), 1.0);
        assert_eq!(yarn_ramp(dims, 7), 1.0);
        assert!((yarn_ramp(dims, 13) - 0.5).abs() < 1e-6);
        assert_eq!(yarn_ramp(dims, 18), 0.0);
        assert_eq!(yarn_ramp(dims, 40), 0.0);
    }

    /// The table: the low frequencies are interpolated (their angles are the
    /// plain ones divided by the factor), the high ones are not, and the
    /// magnitude scale rides on both.
    #[test]
    fn the_yarn_table_interpolates_the_low_frequencies_only() {
        let rope_dim = 8;
        let rope_base = 150000.0f32;
        let yarn = yarn_params(rope_dim, rope_base, 32.0, 4096, 1.0, 1.0, 0.0, 32.0, 1.0);
        let t = build_rope_table_yarn(rope_dim, rope_base, 4, None, &yarn);
        let half = rope_dim / 2;
        for pos in 0..4 {
            for i in 0..half {
                let freq = (rope_base as f64).powf(-((2 * i) as f64) / rope_dim as f64);
                let extrap = pos as f64 * freq;
                let interp = extrap / 32.0;
                let ramp = yarn_ramp(yarn.corr_dims, i) as f64;
                let theta = interp * (1.0 - ramp) + extrap * ramp;
                let c = t[(pos * half + i) * 2];
                let s = t[(pos * half + i) * 2 + 1];
                assert!(
                    (c - (theta.cos() as f32) * yarn.mscale).abs() < 1e-6,
                    "pos {pos} pair {i}: {c}"
                );
                assert!(
                    (s - (theta.sin() as f32) * yarn.mscale).abs() < 1e-6,
                    "pos {pos} pair {i}: {s}"
                );
            }
        }
    }
}
