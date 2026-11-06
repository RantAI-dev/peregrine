//! Tiny random models through the whole forward pass (roadmap S0.2).
//!
//! For every synthetic model: it loads, a prefill gives finite logits, results are bit-identical across
//! runs and across 1, 2 and 4 threads, greedy decoding is deterministic, and the output matches the
//! stored golden for the current kernel tier (`tests/golden/synthetic/<tier>/`). `PEREGRINE_GOLDEN=record`
//! writes goldens. These run everywhere, CI included: no model files are needed.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, State, StepStats};
use peregrine_core::pool::Pool;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use support::models::{write_dense, write_hybrid, Dense};

const PROMPT: [u32; 24] = [
    11, 300, 42, 7, 199, 256, 3, 88, 120, 5, 64, 31, 2, 270, 150, 9, 77, 1, 18, 240, 33, 61, 100,
    12,
];
const GREEDY: usize = 16;

/// A directory of its own for each test (tests run in parallel and each removes its directory).
fn tmpdir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("peregrine-synthetic-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fnv(h: &mut u64, xs: &[f32]) {
    for x in xs {
        for b in x.to_bits().to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
}

fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best as u32
}

/// Prefill `PROMPT`, then decode greedily: the golden text of one run.
fn run(model: &Model, threads: usize) -> String {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let n = PROMPT.len();
    let mut st = State::new(&c, 128, n, 256, threads, model.rope_factors.as_deref());
    let mut kv = model.new_cache(KvPrec::F16, 256);
    let mut ss = StepStats::default();
    let want: Vec<usize> = (0..n).collect();
    model
        .forward(&mut st, &mut kv, &pool, &PROMPT, 0, &want, &mut ss)
        .unwrap();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut am = Vec::new();
    for i in 0..n {
        let row = st.logits_row(i, c.n_vocab);
        assert!(
            row.iter().all(|v| v.is_finite()),
            "non-finite logit at position {i}"
        );
        fnv(&mut h, row);
        am.push(argmax(row));
    }
    let mut next = *am.last().unwrap();
    let mut greedy = vec![next];
    for step in 1..GREEDY {
        model
            .forward(
                &mut st,
                &mut kv,
                &pool,
                &[next],
                n + step - 1,
                &[0],
                &mut ss,
            )
            .unwrap();
        next = argmax(st.logits_row(0, c.n_vocab));
        greedy.push(next);
    }
    let join = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let mut out = String::new();
    writeln!(out, "logits_fnv {h:016x}").unwrap();
    writeln!(out, "argmax {}", join(&am)).unwrap();
    writeln!(out, "greedy {}", join(&greedy)).unwrap();
    out
}

/// Every invariant for one model file; returns a failure description, if any.
fn check(path: &Path, name: &str) -> Option<String> {
    let model = match Gguf::open(path).and_then(Model::load) {
        Ok(m) => m,
        Err(e) => return Some(format!("{name}: does not load: {e}")),
    };
    let a = run(&model, 4);
    if run(&model, 4) != a {
        return Some(format!("{name}: two runs differ"));
    }
    for t in [1, 2] {
        if run(&model, t) != a {
            return Some(format!("{name}: {t} threads differ from 4"));
        }
    }
    let tier = peregrine_core::cpu::tier_name();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/synthetic")
        .join(&tier);
    let gpath = dir.join(format!("{name}.golden"));
    if std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "record") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&gpath, &a).unwrap();
        eprintln!("synthetic: recorded {name} ({tier})");
    } else if let Ok(want) = std::fs::read_to_string(&gpath) {
        if want != a {
            return Some(format!(
                "{name}: differs from {}\n  want:\n{want}  got:\n{a}",
                gpath.display()
            ));
        }
        eprintln!("synthetic: {name} identical ({tier})");
    } else {
        eprintln!("synthetic: {name} ok (no golden for tier {tier})");
    }
    None
}

#[test]
fn dense_models_in_every_format() {
    let dir = tmpdir("dense");
    let mut failures = Vec::new();
    let formats = [
        Format::F32,
        Format::F16,
        Format::Q8_0,
        Format::Q5_0,
        Format::Q4_K,
        Format::Q6_K,
    ];
    for arch in ["llama", "qwen3"] {
        for (i, &fmt) in formats.iter().enumerate() {
            let path = write_dense(&dir, arch, fmt, 1000 + i as u64, Dense::default());
            let name = format!("{arch}-{}", fmt.label());
            failures.extend(check(&path, &name));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic models failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn hybrid_models_in_both_ternary_formats() {
    let dir = tmpdir("hybrid");
    let mut failures = Vec::new();
    for (i, fmt) in [Format::PQ2_0, Format::PTQ1_0].into_iter().enumerate() {
        let path = write_hybrid(&dir, fmt, 2000 + i as u64);
        failures.extend(check(&path, &format!("qwen35-{}", fmt.label())));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic models failed:\n{}",
        failures.join("\n")
    );
}
