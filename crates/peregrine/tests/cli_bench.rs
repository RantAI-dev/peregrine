//! Smoke tests for `peregrine bench` and `peregrine tune` CLI (roadmap S1.7).

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use std::path::PathBuf;
use std::process::Command;
use support::models::{write_dense, Dense};

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-cli-test-{}-{name}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

fn find_peregrine_binary() -> PathBuf {
    // Cargo builds this package's binary for its integration tests and says where.
    PathBuf::from(env!("CARGO_BIN_EXE_peregrine"))
}

#[test]
fn bench_json_and_tune_smoke_test() {
    let dir = tmpdir("smoke");
    let model = write_dense(&dir, "llama", Format::F16, 42, Dense::default());
    let bin = find_peregrine_binary();
    assert!(
        bin.exists(),
        "peregrine binary not found at {}",
        bin.display()
    );

    // 1. Smoke test: `peregrine bench --json`
    let out = Command::new(&bin)
        .args([
            "bench",
            "-m",
            model.to_str().unwrap(),
            "--tokens",
            "8",
            "--prompt-tokens",
            "8",
            "--json",
            "--no-profile",
        ])
        .output()
        .expect("run bench --json");

    assert!(
        out.status.success(),
        "bench failed with stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout_str = String::from_utf8_lossy(&out.stdout);
    assert!(stdout_str.contains("\"tokens\": 8"), "stdout: {stdout_str}");
    assert!(
        stdout_str.contains("\"prompt_tokens\": 8"),
        "stdout: {stdout_str}"
    );
    assert!(
        stdout_str.contains("\"decode_tok_s\":"),
        "stdout: {stdout_str}"
    );
    assert!(
        stdout_str.contains("\"prefill_tok_s\":"),
        "stdout: {stdout_str}"
    );
    assert!(
        stdout_str.contains("\"bytes_per_token\":"),
        "stdout: {stdout_str}"
    );
    assert!(stdout_str.contains("\"config\":"), "stdout: {stdout_str}");
    assert!(
        stdout_str.contains("\"tuning_source\":"),
        "stdout: {stdout_str}"
    );
    assert!(
        stdout_str.contains("\"calibration_ms\":"),
        "stdout: {stdout_str}"
    );

    // 2. Smoke test: `peregrine tune`
    let profile_path = dir.join("test.profile");
    let tune_out = Command::new(&bin)
        .args([
            "tune",
            "-m",
            model.to_str().unwrap(),
            "--budget-s",
            "5",
            "--out",
            profile_path.to_str().unwrap(),
        ])
        .output()
        .expect("run tune");

    assert!(
        tune_out.status.success(),
        "tune failed with stderr: {}",
        String::from_utf8_lossy(&tune_out.stderr)
    );
    assert!(profile_path.exists(), "tune must write the profile");

    let prof = peregrine_core::profile::read(&profile_path).expect("parse generated profile");
    assert_eq!(prof.model_size, std::fs::metadata(&model).unwrap().len());

    // 3. Smoke test: `peregrine bench` applying the profile
    let bench_prof_out = Command::new(&bin)
        .args([
            "bench",
            "-m",
            model.to_str().unwrap(),
            "--tokens",
            "4",
            "--prompt-tokens",
            "4",
            "--profile",
            profile_path.to_str().unwrap(),
        ])
        .output()
        .expect("run bench with profile");

    assert!(bench_prof_out.status.success());
    let stdout = String::from_utf8_lossy(&bench_prof_out.stdout);
    assert!(
        stdout.contains(&format!("profile: applied {}", profile_path.display())),
        "stdout should report applied profile, got:\n{stdout}"
    );

    // 4. Mismatch test: modify one header field in the profile
    let mut modified = prof.clone();
    modified.cpu = "FakeCPU-1234".into();
    peregrine_core::profile::write(&profile_path, &modified).expect("write modified profile");

    let bench_mismatch_out = Command::new(&bin)
        .args([
            "bench",
            "-m",
            model.to_str().unwrap(),
            "--tokens",
            "4",
            "--prompt-tokens",
            "4",
            "--profile",
            profile_path.to_str().unwrap(),
        ])
        .output()
        .expect("run bench with mismatched profile");

    assert!(bench_mismatch_out.status.success());
    let mismatch_stdout = String::from_utf8_lossy(&bench_mismatch_out.stdout);
    assert!(
        mismatch_stdout.contains("profile: ignored") && mismatch_stdout.contains("cpu mismatch"),
        "stdout should report ignored profile with cpu mismatch, got:\n{mismatch_stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
