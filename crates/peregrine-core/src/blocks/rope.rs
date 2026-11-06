//! Rotary position embedding and precomputed table construction.

/// Precompute the RoPE sin/cos table for positions `0..max_ctx`.
pub(crate) fn build_rope_table(
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
            let freq = (rope_base as f64).powf(-((2 * i) as f64) / rope_dim as f64);
            let factor = rope_factors.and_then(|f| f.get(i)).copied().unwrap_or(1.0) as f64;
            (freq, factor)
        })
        .collect();
    for pos in 0..max_ctx {
        for (i, &(freq, factor)) in freqs.iter().enumerate() {
            let theta = pos as f64 * freq / factor;
            rope[(pos * half + i) * 2] = theta.cos() as f32;
            rope[(pos * half + i) * 2 + 1] = theta.sin() as f32;
        }
    }
    rope
}

/// Rotary embedding, ggml's "norm" layout: consecutive pairs `(2i, 2i+1)` are
/// rotated by `pos · base^(-2i/rope_dim)`. NeoX-style models pair `i` with
/// `i + dim/2` instead; llama does not, and using the wrong pairing produces a
/// model that is coherent for a few tokens and then drifts.
pub(crate) fn rope(
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
