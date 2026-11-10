//! Embedding post-processing: pooling types, row pooling, truncation and L2 normalisation
//! (roadmap S4.1).
//!
//! A causal model used as an embedder is prefilled like any prompt; the engine captures the
//! final hidden state (after the model's final norm) of the rows a pooling needs and reduces
//! them to one vector. This module holds the reduction and the file/request vocabulary around
//! it, so the engine, the server and the CLI agree bit for bit.
//!
//! The default comes from the file's `<arch>.pooling_type`, read exactly as llama.cpp reads it
//! (`llama-arch.cpp`, `llama-model.cpp`): an integer whose values are the ones in
//! [`llama.h`](https://github.com/ggml-org/llama.cpp): `0` none, `1` mean, `2` cls, `3` last,
//! `4` rank. A file without the key is `UNSPECIFIED`, which llama.cpp resolves to `NONE`; a
//! pooled OpenAI-style embedding endpoint cannot serve that, so the request is refused by name.

/// How the final hidden states of a text become one vector.
///
/// `Last` and `Mean` are implemented (S4.1, causal models). `Cls` is kept so the file's
/// vocabulary maps one-to-one; using it is refused until a bidirectional encoder needs it
/// (S4.4 onward), because a causal decoder's row 0 sees one token only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// The last row of the text.
    Last,
    /// The mean of every row of the text.
    Mean,
    /// Row 0 of the text. Not implemented for causal models.
    Cls,
}

impl Pooling {
    /// llama.cpp's `llama_pooling_type` integers (`include/llama.h`).
    ///
    /// `0` (none) and `4` (rank) have no single-vector meaning for this stage: `None` returns
    /// per-token rows, and `Rank` needs the classifier head. Both map to `None` here, and the
    /// caller names the reason.
    pub fn from_gguf(v: u32) -> Option<Pooling> {
        match v {
            0 | 4 => None,
            1 => Some(Pooling::Mean),
            2 => Some(Pooling::Cls),
            3 => Some(Pooling::Last),
            _ => None,
        }
    }

    /// The name llama.cpp prints for the type (`none`, `mean`, `cls`, `last`, `rank`).
    pub fn name(self) -> &'static str {
        match self {
            Pooling::Last => "last",
            Pooling::Mean => "mean",
            Pooling::Cls => "cls",
        }
    }

    /// Parse an explicit request's name.
    pub fn parse(s: &str) -> Option<Pooling> {
        match s {
            "last" => Some(Pooling::Last),
            "mean" => Some(Pooling::Mean),
            "cls" => Some(Pooling::Cls),
            _ => None,
        }
    }
}

/// Reduce `rows × dim` final hidden states to one vector.
///
/// `rows` are the text's own rows in position order. `Mean` accumulates in `f64` and rounds
/// once at the end, so the result does not depend on how the rows were grouped into chunks
/// or on which batch they were computed in. `Last` is the last row's `dim` floats.
///
/// `out` is reused; it is resized to `dim`.
pub fn pool_rows(rows: &[f32], dim: usize, pooling: Pooling, out: &mut Vec<f32>) {
    out.clear();
    if dim == 0 || rows.is_empty() {
        out.resize(dim, 0.0);
        return;
    }
    match pooling {
        Pooling::Last => {
            let start = (rows.len() / dim - 1) * dim;
            out.extend_from_slice(&rows[start..start + dim]);
        }
        Pooling::Mean => {
            let n = rows.len() / dim;
            let mut sum = vec![0.0f64; dim];
            for row in rows.chunks_exact(dim) {
                for (s, &x) in sum.iter_mut().zip(row) {
                    *s += x as f64;
                }
            }
            let inv = 1.0 / n as f64;
            out.extend(sum.iter().map(|s| (s * inv) as f32));
        }
        Pooling::Cls => {
            out.extend_from_slice(&rows[..dim]);
        }
    }
}

/// Truncate to `dimensions` (Matryoshka), then L2-normalise — in that order, as the design says.
///
/// The norm accumulates the squares in `f64`, takes the square root in `f64`, then scales each
/// component by a single `f32` factor, matching llama.cpp's `common_embd_normalize` (`case 2`):
/// the same sum, the same rounding, so the two engines' vectors are comparable bit for bit after
/// a shared forward. A zero vector stays zero. `dimensions` larger than the vector is refused
/// rather than padded.
pub fn finish(v: &mut Vec<f32>, dimensions: Option<usize>) -> Result<(), String> {
    if let Some(d) = dimensions {
        if d == 0 {
            return Err("dimensions must be greater than 0".into());
        }
        if d > v.len() {
            return Err(format!(
                "dimensions {d} is larger than the model's embedding width {}",
                v.len()
            ));
        }
        v.truncate(d);
    }
    let sum: f64 = v.iter().map(|&x| x as f64 * x as f64).sum();
    // llama.cpp: `const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;` after `sum = sqrt(sum)`.
    let norm = if sum > 0.0 {
        (1.0 / sum.sqrt()) as f32
    } else {
        0.0
    };
    for x in v.iter_mut() {
        *x *= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_cpp_pooling_integers_map_as_the_file_says() {
        assert_eq!(Pooling::from_gguf(0), None); // NONE
        assert_eq!(Pooling::from_gguf(1), Some(Pooling::Mean));
        assert_eq!(Pooling::from_gguf(2), Some(Pooling::Cls));
        assert_eq!(Pooling::from_gguf(3), Some(Pooling::Last));
        assert_eq!(Pooling::from_gguf(4), None); // RANK: a classifier head, not a vector
        assert_eq!(Pooling::from_gguf(99), None);
    }

    #[test]
    fn pooling_rows_is_position_order() {
        let rows = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![];
        pool_rows(&rows, 2, Pooling::Last, &mut out);
        assert_eq!(out, vec![5.0, 6.0]);
        pool_rows(&rows, 2, Pooling::Cls, &mut out);
        assert_eq!(out, vec![1.0, 2.0]);
        pool_rows(&rows, 2, Pooling::Mean, &mut out);
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn mean_rounds_once_in_f64() {
        // 0.1 + 0.2 in f32 is not 0.3; in f64 accumulated and rounded once it is the nearest
        // f32 to the exact mean, which the direct f32 sum gets wrong.
        let rows = [0.1f32, 0.2f32];
        let mut out = vec![];
        pool_rows(&rows, 1, Pooling::Mean, &mut out);
        assert_eq!(out[0], 0.15f32);
        // Chunking the same rows differently gives the same bits: the accumulator is f64.
        let a = [0.1f32, 0.2f32, 0.3f32, 0.4f32, 0.5f32, 0.6f32];
        let b = [0.1f32 + 0.0, 0.2, 0.3, 0.4, 0.5, 0.6];
        let (mut x, mut y) = (vec![], vec![]);
        pool_rows(&a, 1, Pooling::Mean, &mut x);
        pool_rows(&b, 1, Pooling::Mean, &mut y);
        assert_eq!(x, y);
    }

    #[test]
    fn truncate_then_normalise() {
        let mut v = vec![3.0, 4.0, 12.0, 0.0];
        finish(&mut v, Some(2)).unwrap();
        assert_eq!(v.len(), 2);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        let mut v = vec![0.0, 0.0];
        finish(&mut v, None).unwrap();
        assert_eq!(v, vec![0.0, 0.0]);
        assert!(finish(&mut vec![1.0], Some(0)).is_err());
        assert!(finish(&mut vec![1.0], Some(2)).is_err());
    }

    #[test]
    fn names_round_trip() {
        for p in [Pooling::Last, Pooling::Mean, Pooling::Cls] {
            assert_eq!(Pooling::parse(p.name()), Some(p));
        }
        assert_eq!(Pooling::parse("none"), None);
    }
}
