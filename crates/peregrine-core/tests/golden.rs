//! Golden outputs: the safety net for refactors (roadmap Wave 0).
//!
//! For every model file present, three fixed prompts are run through the engine. The test records, per
//! prompt, a hash of the exact bits of every prefill logit, the argmax at every position, and 24 greedy
//! tokens. A refactor that keeps behaviour identical reproduces all of it bit for bit.
//!
//! Kernel tiers (AVX2, AVX-512, ...) round differently, so goldens are kept per tier:
//! `tests/golden/<tier>/<model>.golden`. The test also checks that the result does not depend on the
//! thread count.
//!
//!   cargo test --release --test golden                     # compare against stored goldens
//!   PEREGRINE_GOLDEN=record cargo test --release --test golden -- --nocapture   # (re)write them
//!   PEREGRINE_GOLDEN_MODELS=Qwen3-1.7B,bonsai2-2b ...      # only models whose path contains one of these
//!
//! Models are looked up in `models/` at the repository root and in `~/models/`. A missing model is skipped.

use peregrine_core::gguf::Gguf;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, State, StepStats};
use peregrine_core::pool::Pool;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const PROMPTS: [&str; 3] = [
    "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined \
     movable metal type, oil-based ink and a screw press.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in range(n):",
    "Question: A train leaves at 9:40 and arrives at 13:05. How long is the journey? Answer step by step.",
];
const GREEDY_TOKENS: usize = 24;
const MAX_CTX: usize = 512;

/// Candidate model files, relative to `models/` (repo) or `~/models/`.
const MODELS: [&str; 11] = [
    "SmolLM2-360M-Instruct.Q4_K_M.gguf",
    "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "Llama-2-7b-chat-hf-Q4_K_M.gguf",
    "Mistral-7B-Instruct-v0.2.Q4_K_M.gguf",
    "Qwen3-1.7B-Q4_K_M.gguf",
    "Qwen3-4B-Q4_K_M.gguf",
    "bonsai2-2b/Ternary-Bonsai-2-2B-PQ2_0.gguf",
    "bonsai2-4b/Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf",
    "bonsai2-9b/Ternary-Bonsai-2-9B-PQ2_0.gguf",
    "bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn find(rel: &str) -> Option<PathBuf> {
    let mut dirs = vec![repo_root().join("models")];
    if let Ok(h) = std::env::var("HOME") {
        dirs.push(Path::new(&h).join("models"));
    }
    dirs.into_iter().map(|d| d.join(rel)).find(|p| p.is_file())
}

/// The kernel tier in effect (`PEREGRINE_ISA` can cap it), as a directory name.
fn tier() -> String {
    peregrine_core::cpu::tier_name()
}

/// FNV-1a over the bit patterns: any change in any logit changes the hash.
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

/// One model at one thread count: the golden text.
fn run(model: &Model, threads: usize) -> String {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let mut out = String::new();
    for (pi, text) in PROMPTS.iter().enumerate() {
        let toks = model.tk.encode(text);
        let n = toks.len();
        let mut st = State::new(&c, 128, n, MAX_CTX, threads, model.rope_factors.as_deref());
        let mut kv = model.new_cache(KvPrec::F16, MAX_CTX);
        let mut ss = StepStats::default();
        let want: Vec<usize> = (0..n).collect();
        model
            .forward(&mut st, &mut kv, &pool, &toks, 0, &want, &mut ss)
            .unwrap();
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut am = Vec::with_capacity(n);
        for i in 0..n {
            let row = st.logits_row(i, c.n_vocab);
            fnv(&mut h, row);
            am.push(argmax(row));
        }
        let mut next = *am.last().unwrap();
        let mut greedy = vec![next];
        for step in 1..GREEDY_TOKENS {
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
        writeln!(out, "prompt {pi} tokens {n} logits_fnv {h:016x}").unwrap();
        writeln!(out, "argmax {pi} {}", join(&am)).unwrap();
        writeln!(out, "greedy {pi} {}", join(&greedy)).unwrap();
    }
    out
}

#[test]
fn golden_outputs() {
    let record = std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "record");
    let only: Vec<String> = std::env::var("PEREGRINE_GOLDEN_MODELS")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let tier = tier();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(&tier);
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(2, 8);
    let mut checked = 0;
    let mut failures = Vec::new();
    for rel in MODELS {
        if !only.is_empty() && !only.iter().any(|o| rel.contains(o.as_str())) {
            continue;
        }
        let Some(path) = find(rel) else {
            eprintln!("golden: skip {rel} (model file not found)");
            continue;
        };
        let name = Path::new(rel)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let gpath = dir.join(format!("{name}.golden"));
        if !record && !gpath.is_file() {
            eprintln!("golden: skip {rel} (no golden for tier {tier}; record one)");
            continue;
        }
        let model = Model::load(Gguf::open(&path).unwrap()).unwrap();
        let a = run(&model, threads);
        let b = run(&model, 1.max(threads / 2));
        if a != b {
            failures.push(format!(
                "{name}: output depends on thread count ({threads} vs {})",
                threads / 2
            ));
            continue;
        }
        let size = std::fs::metadata(&path).unwrap().len();
        let text = format!("# peregrine golden v1\nmodel {name} bytes {size}\ntier {tier}\n{a}");
        if record {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&gpath, &text).unwrap();
            eprintln!("golden: recorded {}", gpath.display());
        } else {
            let want = std::fs::read_to_string(&gpath).unwrap();
            if want != text {
                let diff: Vec<String> = want
                    .lines()
                    .zip(text.lines())
                    .filter(|(w, g)| w != g)
                    .take(6)
                    .map(|(w, g)| format!("  want: {}\n  got:  {}", short(w), short(g)))
                    .collect();
                failures.push(format!(
                    "{name} differs from {}:\n{}",
                    gpath.display(),
                    diff.join("\n")
                ));
            } else {
                eprintln!("golden: {name} identical ({tier})");
            }
        }
        checked += 1;
    }
    eprintln!(
        "golden: {checked} model(s) {} on tier {tier}",
        if record { "recorded" } else { "checked" }
    );
    assert!(
        failures.is_empty(),
        "golden outputs changed:\n{}",
        failures.join("\n")
    );
}

fn short(s: &str) -> String {
    if s.len() > 140 {
        format!("{}...", &s[..140])
    } else {
        s.to_string()
    }
}
