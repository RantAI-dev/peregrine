//! Tests for `peregrine support` CLI command (roadmap S2.7).

use std::path::PathBuf;
use std::process::Command;

fn find_peregrine_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_peregrine"))
}

#[test]
fn support_plain_smoke_test() {
    let bin = find_peregrine_binary();
    assert!(
        bin.exists(),
        "peregrine binary not found at {}",
        bin.display()
    );

    let out = Command::new(&bin)
        .arg("support")
        .output()
        .expect("run peregrine support");

    assert!(out.status.success(), "peregrine support failed: {:?}", out);
    let s = String::from_utf8(out.stdout).expect("utf-8");

    // Must list known architectures
    assert!(s.contains("llama"), "missing llama in support: {s}");
    assert!(s.contains("qwen2"), "missing qwen2 in support: {s}");
    assert!(s.contains("qwen3"), "missing qwen3 in support: {s}");
    assert!(s.contains("qwen35"), "missing qwen35 in support: {s}");
    assert!(s.contains("phi3"), "missing phi3 in support: {s}");

    // Must list known formats
    assert!(s.contains("F32"), "missing F32 in support: {s}");
    assert!(s.contains("F16"), "missing F16 in support: {s}");
    assert!(s.contains("BF16"), "missing BF16 in support: {s}");
    assert!(s.contains("Q4_0"), "missing Q4_0 in support: {s}");
    assert!(s.contains("Q8_0"), "missing Q8_0 in support: {s}");
    assert!(s.contains("Q4_K"), "missing Q4_K in support: {s}");
    assert!(s.contains("PQ2_0"), "missing PQ2_0 in support: {s}");
    assert!(s.contains("PTQ1_0"), "missing PTQ1_0 in support: {s}");

    // Must list tiers
    assert!(s.contains("scalar"), "missing scalar tier: {s}");
    assert!(s.contains("avx2"), "missing avx2 tier: {s}");

    // Must list tokenizers and chat templates
    assert!(s.contains("SentencePiece"), "missing sentencepiece: {s}");
    assert!(s.contains("chatml"), "missing chatml: {s}");
}

#[test]
fn support_markdown_smoke_test() {
    let bin = find_peregrine_binary();
    let out = Command::new(&bin)
        .args(["support", "--markdown"])
        .output()
        .expect("run peregrine support --markdown");

    assert!(
        out.status.success(),
        "peregrine support --markdown failed: {:?}",
        out
    );
    let s = String::from_utf8(out.stdout).expect("utf-8");

    assert!(
        s.contains("| | Supported now |"),
        "missing table header: {s}"
    );
    assert!(
        s.contains("| Architectures |"),
        "missing Architectures row: {s}"
    );
    assert!(s.contains("Llama"), "missing Llama in markdown: {s}");
    assert!(s.contains("Qwen 2"), "missing Qwen 2 in markdown: {s}");
    assert!(s.contains("Qwen 3.5"), "missing Qwen 3.5 in markdown: {s}");
    assert!(
        s.contains("Phi-3 / Phi-4"),
        "missing Phi-3 / Phi-4 in markdown: {s}"
    );

    assert!(
        s.contains("| Weight formats |"),
        "missing Weight formats row: {s}"
    );
    assert!(s.contains("Q4_K"), "missing Q4_K in markdown: {s}");
    assert!(s.contains("PQ2_0"), "missing PQ2_0 in markdown: {s}");

    assert!(s.contains("| CPUs |"), "missing CPUs row: {s}");
    assert!(s.contains("| Tokenizers |"), "missing Tokenizers row: {s}");
    assert!(
        s.contains("| Chat templates |"),
        "missing Chat templates row: {s}"
    );
}
