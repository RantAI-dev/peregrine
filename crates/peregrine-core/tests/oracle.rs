//! Acceptance criterion A1: the kernels agree with llama.cpp on real weights.
//!
//! This is the test that matters most in the crate, because the failure it catches
//! is invisible by any other means. A quantised block layout read with the wrong
//! interleaving — high nibbles where low ones belong, a super-block scale applied to
//! the wrong 32 of its 256 weights — still decodes to finite, plausibly-distributed
//! floats. The model still loads. It still generates fluent English. It is simply
//! the wrong model, and nothing short of comparing against a known-good decoder
//! will say so.
//!
//! The golden values in `fixtures/dequant-golden.f32` were produced by llama.cpp's
//! own dequantisation code, not by a second transcription of it:
//!
//! ```text
//! llama-quantize --allow-requantize SmolLM2-360M-Instruct.Q4_K_M.gguf smol-f32.gguf F32 4
//! ```
//!
//! with llama.cpp v0.9.0 (commit cd959bae99ff4bed22bef2604f45241c09bbb8c8), and then a
//! window of up to 2048 elements of one tensor per weight format the model uses was
//! copied out byte for byte. The windows start at element 4096 where the tensor is long
//! enough, so they exercise block-offset arithmetic rather than only the first block.
//! The Q4_K window is `blk.10.ffn_down`: in this upload `blk.12.ffn_down` is Q6_K, and
//! every format the model contains must stay covered.
//!
//! The fixture is pinned to one exact model file — mradermacher's upload,
//! <https://huggingface.co/mradermacher/SmolLM2-360M-Instruct-GGUF> — 270590976 bytes,
//! SHA-256 `ade7aee1e5bd7c3b8f1fedafc61a1a0be31e8afa64e006607cd202323e2ccce0`. Another
//! Q4_K_M upload is not interchangeable: the previous fixture, cut from a different
//! upload, stored `blk.12.ffn_down` as Q4_K and differed in one F32 norm weight. So a
//! different local file fails here with the expected hash, rather than as a kernel bug.
//!
//! The model file itself is 258 MiB and is not in the repository. Point
//! `PEREGRINE_MODEL` at it, or leave it at the default path below.

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::quant::dequant_row;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Identity of the model the fixture was cut from. The full SHA-256 is checked by hand
/// (there is no hashing dependency); the test itself compares the size and an FNV-1a of
/// the first and last MiB, which is enough to tell uploads apart at negligible cost.
const MODEL_SIZE: u64 = 270_590_976;
const MODEL_SHA256: &str = "ade7aee1e5bd7c3b8f1fedafc61a1a0be31e8afa64e006607cd202323e2ccce0";
const MODEL_FNV1A_FIRST_MIB: u64 = 12_794_659_150_644_441_972;
const MODEL_FNV1A_LAST_MIB: u64 = 9_807_110_217_639_181_536;
const MODEL_URL: &str = "https://huggingface.co/mradermacher/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct.Q4_K_M.gguf";
const MIB: u64 = 1 << 20;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Fail, with the expected identity, when the file at `path` is not the pinned upload.
fn verify_pinned_model(path: &Path) {
    let len = std::fs::metadata(path).expect("stat model").len();
    let mut file = std::fs::File::open(path).expect("open model");
    let mut first = vec![0u8; MIB as usize];
    let mut last = vec![0u8; MIB as usize];
    let (got_first, got_last) = if len >= 2 * MIB {
        file.read_exact(&mut first).expect("read first MiB");
        file.seek(SeekFrom::End(-(MIB as i64)))
            .expect("seek last MiB");
        file.read_exact(&mut last).expect("read last MiB");
        (fnv1a64(&first), fnv1a64(&last))
    } else {
        (0, 0)
    };
    assert!(
        len == MODEL_SIZE && got_first == MODEL_FNV1A_FIRST_MIB && got_last == MODEL_FNV1A_LAST_MIB,
        "\n  model file {path:?} is not the pinned SmolLM2-360M-Instruct.Q4_K_M.gguf.\n  \
         expected: size {MODEL_SIZE}, SHA-256 {MODEL_SHA256}\n            \
         first/last MiB FNV-1a {MODEL_FNV1A_FIRST_MIB:#018x} / {MODEL_FNV1A_LAST_MIB:#018x}\n  \
         got:      size {len}, first/last MiB FNV-1a {got_first:#018x} / {got_last:#018x}\n  \
         download the exact file from {MODEL_URL}\n"
    );
}

fn model_path() -> PathBuf {
    std::env::var_os("PEREGRINE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join("models/SmolLM2-360M-Instruct.Q4_K_M.gguf")
        })
}

fn fmt_of(s: &str) -> Format {
    match s {
        "F32" => Format::F32,
        "F16" => Format::F16,
        "Q5_0" => Format::Q5_0,
        "Q8_0" => Format::Q8_0,
        "Q4_K" => Format::Q4_K,
        "Q6_K" => Format::Q6_K,
        other => panic!("fixture names an unknown format {other:?}"),
    }
}

#[test]
fn dequantisation_matches_llama_cpp_on_the_real_model() {
    let path = model_path();
    if !path.exists() {
        // Skipping is a real cost — this is the only test that can catch a wrong
        // block layout — so say so loudly rather than passing quietly.
        eprintln!(
            "\n  SKIPPED: A1 (kernels vs llama.cpp) needs the model at {path:?}.\n  \
             Set PEREGRINE_MODEL to run it. No other test covers block layout.\n"
        );
        return;
    }
    verify_pinned_model(&path);
    let g = Gguf::open(&path).expect("open model");

    let tsv = include_str!("fixtures/dequant-golden.tsv");
    let blob: &[u8] = include_bytes!("fixtures/dequant-golden.f32");

    let mut checked = 0usize;
    let mut formats = Vec::new();
    for line in tsv
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 5, "malformed fixture line: {line}");
        let (name, fmt, start, count, off) = (
            f[0],
            fmt_of(f[1]),
            f[2].parse::<usize>().unwrap(),
            f[3].parse::<usize>().unwrap(),
            f[4].parse::<usize>().unwrap(),
        );

        let ti = g.tensor(name).expect(name);
        assert_eq!(
            ti.kind,
            fmt,
            "{name} is {} in the model but the fixture says {}; the fixture is stale",
            ti.kind.label(),
            fmt.label()
        );

        // Decode exactly the window the fixture covers. `start` is block-aligned, so
        // the byte offset is exact and no partial block is involved.
        let blk = fmt.block();
        assert_eq!(start % blk, 0);
        assert_eq!(count % blk, 0);
        let bytes = g.bytes(ti);
        let b0 = start / blk * fmt.block_bytes();
        let b1 = (start + count) / blk * fmt.block_bytes();
        let mut got = vec![0.0f32; count];
        dequant_row(fmt, &bytes[b0..b1], &mut got);

        for i in 0..count {
            let want = f32::from_le_bytes(blob[off + i * 4..off + i * 4 + 4].try_into().unwrap());
            // Both sides decode the same stored codes with the same arithmetic, so
            // this is an equality check up to f32 rounding order, not a tolerance
            // on a numerical method. A layout error is off by orders of magnitude,
            // not by an ulp.
            let tol = 1e-6 * want.abs().max(1e-3);
            assert!(
                (got[i] - want).abs() <= tol,
                "{name} [{}] element {}: peregrine-core {} vs llama.cpp {}",
                fmt.label(),
                start + i,
                got[i],
                want
            );
        }
        // A window of all zeros would pass the comparison while proving nothing.
        let nonzero = got.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > count / 2,
            "{name}: only {nonzero}/{count} decoded values are non-zero — \
             the window is not exercising the decoder"
        );
        checked += count;
        formats.push(fmt);
    }

    // The fixture must actually cover every format the model contains, or A1 is
    // only evidence about the formats it happens to include.
    let mut present: Vec<Format> = g.tensors.values().map(|t| t.kind).collect();
    present.sort_by_key(|f| f.label());
    present.dedup();
    for f in present {
        assert!(
            formats.contains(&f),
            "model contains {} but the golden fixture does not cover it",
            f.label()
        );
    }
    eprintln!(
        "A1: {checked} weights checked against llama.cpp across {} formats",
        formats.len()
    );
}

#[test]
fn the_model_loads_with_the_shape_the_spec_describes() {
    let path = model_path();
    if !path.exists() {
        eprintln!("  SKIPPED: needs the model at {path:?}");
        return;
    }
    verify_pinned_model(&path);
    let g = Gguf::open(&path).expect("open model");

    assert_eq!(g.str("general.architecture").unwrap(), "llama");
    assert_eq!(g.u32("llama.block_count").unwrap(), 32);
    assert_eq!(g.u32("llama.embedding_length").unwrap(), 960);
    assert_eq!(g.u32("llama.attention.head_count").unwrap(), 15);
    assert_eq!(g.u32("llama.attention.head_count_kv").unwrap(), 5);
    assert_eq!(g.u32("llama.feed_forward_length").unwrap(), 2560);
    assert_eq!(g.u32("llama.rope.dimension_count").unwrap(), 64);
    assert_eq!(g.tensors.len(), 290);

    // Tied embeddings: this model has no separate output projection, and the
    // loader must notice by absence rather than assume either way.
    assert!(g.tensors.contains_key("token_embd.weight"));
    assert!(!g.tensors.contains_key("output.weight"));

    // Every tensor's declared bytes must lie inside the map; `Gguf::open` checks
    // this, so reaching here at all is the assertion. Confirm the total is sane.
    let total = g.total_weight_bytes();
    let file = std::fs::metadata(&path).unwrap().len();
    assert!(
        total < file && total > file * 9 / 10,
        "weights {total} vs file {file} — directory and data disagree"
    );
}
