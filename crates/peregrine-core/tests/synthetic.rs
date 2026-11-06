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
use peregrine_core::serve::{Engine, Finish, SlotCfg};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{
    write_dense, write_hybrid, write_hybrid_accepting, write_rope_dense, Dense, Rope,
};

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
    let (lx, lt) = model.lb_scratch(128);
    st.set_lb_scratch(lx, lt);
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
    check_model(model, name)
}

/// The same invariants and golden for a model loaded and converted from a LittleBit sidecar.
fn check_lb(path: &Path, sidecar: &Path, name: &str) -> Option<String> {
    let mut model = match Gguf::open(path).and_then(Model::load) {
        Ok(m) => m,
        Err(e) => return Some(format!("{name}: does not load: {e}")),
    };
    let slb = match peregrine_core::lbq::Slb::open(sidecar) {
        Ok(s) => s,
        Err(e) => return Some(format!("{name}: sidecar does not load: {e}")),
    };
    match model.attach_lb(slb) {
        Ok(n) if n > 0 => {}
        Ok(_) => return Some(format!("{name}: the sidecar replaced no tensor")),
        Err(e) => return Some(format!("{name}: attach_lb: {e}")),
    }
    check_model(model, name)
}

fn check_model(model: Model, name: &str) -> Option<String> {
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
    if std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "record") && !gpath.exists() {
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
        Format::Q4_0,
        Format::Q5_K,
        Format::IQ4_NL,
        Format::IQ4_XS,
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
        let path = write_hybrid(&dir, fmt, 2000 + i as u64, false);
        failures.extend(check(&path, &format!("qwen35-{}", fmt.label())));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic models failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn hybrid_mtp_variant_speculates_losslessly() {
    use peregrine_core::serve::{Engine, Finish, SlotCfg};
    use std::sync::Arc;

    let dir = tmpdir("hybrid-mtp");
    let mut failures = Vec::new();
    for (i, fmt) in [Format::PQ2_0, Format::PTQ1_0].into_iter().enumerate() {
        let path = write_hybrid(&dir, fmt, 3000 + i as u64, true);
        let name = format!("qwen35-mtp-{}", fmt.label());
        failures.extend(check(&path, &name));
        let model = match Gguf::open(&path).and_then(Model::load) {
            Ok(m) => Arc::new(m),
            Err(e) => {
                failures.push(format!("{name}: does not load: {e}"));
                continue;
            }
        };
        if !model.has_mtp() {
            failures.push(format!("{name}: loaded without an MTP block"));
            continue;
        }
        // Greedy through the batching engine, with the MTP drafter and with `disable_mtp()` (what
        // `PEREGRINE_MTP=0` does at startup): the token streams must be the same.
        let decode = |spec: bool| -> Result<(Vec<u32>, u64), String> {
            let mut e = Engine::new(model.clone(), 4, 512, 1, KvPrec::F16, 0);
            if !spec {
                e.disable_mtp();
            }
            if e.speculates() != spec {
                return Err(format!("speculation {spec} was not configured"));
            }
            let cfg = SlotCfg {
                temperature: 0.0,
                max_tokens: 24,
                ..Default::default()
            };
            e.admit(PROMPT.to_vec(), cfg).map_err(|(_, e)| e)?;
            let mut toks = Vec::new();
            while e.active() > 0 {
                for em in e.step() {
                    if let Some(Finish::Error(e)) = em.finish {
                        return Err(e);
                    }
                    toks.push(em.token);
                }
            }
            Ok((toks, e.spec_drafts))
        };
        match (decode(true), decode(false)) {
            (Ok((with, drafts)), Ok((without, _))) => {
                if drafts == 0 {
                    failures.push(format!("{name}: the drafter never ran"));
                }
                if with.len() != 24 {
                    failures.push(format!("{name}: {} tokens, want 24", with.len()));
                }
                if with != without {
                    failures.push(format!("{name}: MTP speculation changed the tokens"));
                }
            }
            (Err(e), _) | (_, Err(e)) => failures.push(format!("{name}: {e}")),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic MTP models failed:\n{}",
        failures.join("\n")
    );
}

// ---------------------------------------------------------------- qwen2, rope, LittleBit --

/// Qwen2 is llama with attention biases and NeoX rotary pairing. Both had loader and forward
/// code paths with no synthetic model before this test.
#[test]
fn qwen2_attention_biases_load_and_match_their_goldens() {
    let dir = tmpdir("qwen2");
    let mut failures = Vec::new();
    for (i, fmt) in [Format::F16, Format::Q8_0].into_iter().enumerate() {
        let path = write_dense(&dir, "qwen2", fmt, 4100 + i as u64, Dense::default());
        failures.extend(check(&path, &format!("qwen2-{}", fmt.label())));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic qwen2 models failed:\n{}",
        failures.join("\n")
    );
}

/// Llama 3.x's per-frequency factor table and linear rope scaling are separate loader paths
/// (`Model::load`'s rope_factors) that the base llama model never reaches.
#[test]
fn rope_scaling_variants_load_and_match_their_goldens() {
    let dir = tmpdir("rope");
    let mut failures = Vec::new();
    let variants = [
        (Rope::Freqs, "llama-rope-freqs"),
        (Rope::Linear { factor: 8.0 }, "llama-rope-linear"),
    ];
    for (i, (rope, name)) in variants.into_iter().enumerate() {
        let path = write_rope_dense(&dir, rope, 4200 + i as u64);
        failures.extend(check(&path, name));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic rope-scaling models failed:\n{}",
        failures.join("\n")
    );
}

/// Builds a tiny sidecar for a synthetic llama: two layer-0 projections are replaced, so the
/// forward pass runs a mixed dense/LittleBit layer and the fused Q/K/V group has one latent job.
#[test]
fn littlebit_sidecar_forward_is_finite_and_deterministic() {
    use peregrine_core::lbq::{build_tensor, default_sidecar_path, source_stamp, write_slb};
    use support::weights::Rng;

    let dir = tmpdir("lbq");
    let path = write_dense(&dir, "llama", Format::F16, 4300, Dense::default());
    let sidecar = default_sidecar_path(&path);
    // The sidecar's weights need only match the tensor geometry: attach_lb checks that, and the
    // forward pass is what this test exercises.
    let mut rng = Rng::new(4301);
    let pool = Pool::new(4);
    let mut builds = Vec::new();
    for (name, d_out, d_in) in [
        ("blk.0.attn_q.weight", 256usize, 256usize),
        ("blk.0.ffn_down.weight", 256, 512),
    ] {
        let w: Vec<f32> = (0..d_out * d_in)
            .map(|_| rng.f32_range(-0.04, 0.04))
            .collect();
        builds.push(build_tensor(name, w, d_out, d_in, 32, 2, 1, &pool));
    }
    let (size, hash) = source_stamp(&path).unwrap();
    write_slb(&sidecar, size, hash, &builds).unwrap();

    let failures = check_lb(&path, &sidecar, "llama-lb-F16");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_none(),
        "synthetic LittleBit model failed:\n{}",
        failures.unwrap_or_default()
    );
}

// ------------------------------------------------------------- batching through the engine --

const ENGINE_TOKENS: usize = 16;

fn greedy_cfg() -> SlotCfg {
    SlotCfg {
        temperature: 0.0,
        max_tokens: ENGINE_TOKENS,
        ..Default::default()
    }
}

fn load_arc(path: &Path) -> Arc<Model> {
    Arc::new(Gguf::open(path).and_then(Model::load).unwrap())
}

/// The long third prompt keeps one slot in prefill while the other two decode, which is the
/// mixed step (`Caches::Mapped` with decode rows beside a prompt chunk).
fn batch_prompts() -> Vec<Vec<u32>> {
    let long: Vec<u32> = (0..80).map(|i| ((i * 7) % 317 + 2) as u32).collect();
    vec![
        PROMPT.to_vec(),
        vec![17, 44, 201, 6, 99, 130, 250, 21, 300, 77, 4, 180],
        long,
    ]
}

fn shared_prefix() -> Vec<u32> {
    (0..40).map(|i| ((i * 13) % 317 + 2) as u32).collect()
}

/// One prompt through an engine, collecting its token stream. Used for the 1-slot solo runs and
/// for the two sequential prefix runs.
fn drive_one(e: &mut Engine, prompt: &[u32]) -> Vec<u32> {
    let id = e
        .admit(prompt.to_vec(), greedy_cfg())
        .unwrap_or_else(|(_, msg)| panic!("admit: {msg}"));
    let mut out = Vec::new();
    while e.active() > 0 {
        for em in e.step() {
            if let Some(Finish::Error(err)) = em.finish {
                panic!("engine: {err}");
            }
            assert_eq!(em.slot, id, "a single user's slot changed mid-request");
            out.push(em.token);
        }
    }
    out
}

fn solo_stream(model: &Arc<Model>, prompt: &[u32]) -> Vec<u32> {
    let mut e = Engine::new(model.clone(), 4, 512, 1, KvPrec::F16, 0);
    drive_one(&mut e, prompt)
}

/// The invariance checks for one model. `perrow` is true in the child process, where
/// `PEREGRINE_MIX=0` makes `Engine::step` prefill through `Caches::Shared` and decode through
/// `Caches::PerRow`; in the default (mixed) parent every engine step goes through
/// `Caches::Mapped`.
fn batching_failures(model: &Arc<Model>, label: &str, perrow: bool) -> Vec<String> {
    let mut failures = Vec::new();
    let prompts = batch_prompts();

    // (a) each prompt alone in a 1-slot engine.
    let solo: Vec<Vec<u32>> = prompts.iter().map(|p| solo_stream(model, p)).collect();
    for (i, s) in solo.iter().enumerate() {
        if s.len() != ENGINE_TOKENS {
            failures.push(format!(
                "{label}: solo prompt {i} produced {} tokens, want {ENGINE_TOKENS}",
                s.len()
            ));
        }
    }

    // (b) all three admitted together in a 4-slot engine. The default mixed schedule reaches
    // `Caches::Mapped`; the child process reaches `Caches::PerRow` (and `Caches::Shared` for
    // the prefill-only steps). The second step is mixed: two slots decode while the long third
    // prompt is still prefilling.
    let mut e = Engine::new(model.clone(), 4, 512, 4, KvPrec::F16, 1 << 26);
    let ids: Vec<_> = prompts
        .iter()
        .map(|p| {
            e.admit(p.clone(), greedy_cfg())
                .unwrap_or_else(|(_, msg)| panic!("admit: {msg}"))
        })
        .collect();
    let mut got = vec![Vec::new(); prompts.len()];
    let mut mixed_seen = false;
    while e.active() > 0 {
        let active = e.active();
        let ems = e.step();
        let empty = ems.is_empty();
        let mut emitted = 0usize;
        for em in ems {
            if let Some(Finish::Error(err)) = em.finish {
                failures.push(format!("{label}: {err}"));
            }
            let i = ids
                .iter()
                .position(|&x| x == em.slot)
                .expect("an emission for an admitted slot");
            got[i].push(em.token);
            emitted += 1;
        }
        if !empty && emitted < active {
            mixed_seen = true;
        }
    }
    if !perrow && !mixed_seen {
        failures.push(format!(
            "{label}: no step mixed a prefill chunk with decode rows"
        ));
    }
    if e.decode_rows <= e.decode_steps {
        failures.push(format!(
            "{label}: never decoded two rows in one pass ({} rows / {} steps)",
            e.decode_rows, e.decode_steps
        ));
    }
    for (i, want) in solo.iter().enumerate() {
        if got[i] != *want {
            failures.push(format!(
                "{label}: prompt {i} changed when served with the others\n  alone: {want:?}\n  batch: {:?}",
                got[i]
            ));
        }
    }

    // (c) two prompts sharing a 40-token opening, admitted one after the other: the second
    // restores the prefix store's snapshot instead of prefilling it again.
    let opening = shared_prefix();
    let mut first = opening.clone();
    first.extend_from_slice(&[301, 302, 303, 304]);
    let mut second = opening.clone();
    second.extend_from_slice(&[311, 312, 313, 314]);
    let first_solo = solo_stream(model, &first);
    let second_solo = solo_stream(model, &second);
    let mut warm = Engine::new(model.clone(), 4, 512, 1, KvPrec::F16, 1 << 26);
    let warm_first = drive_one(&mut warm, &first);
    let warm_second = drive_one(&mut warm, &second);
    let (entries, _, hits, saved) = warm.prefix_stats();
    if warm_first != first_solo {
        failures.push(format!("{label}: the first prefix prompt changed"));
    }
    if warm_second != second_solo {
        failures.push(format!(
            "{label}: a prompt with a restored prefix changed\n  cold: {second_solo:?}\n  warm: {warm_second:?}"
        ));
    }
    if model.is_hybrid() {
        // The prefix store deliberately keeps nothing for a hybrid model: a KvSnapshot holds
        // no recurrent state, so restoring it could never be correct.
        if entries != 0 || hits != 0 || saved != 0 {
            failures.push(format!(
                "{label}: the hybrid prefix store kept a snapshot ({entries} entries, {hits} hits)"
            ));
        }
    } else {
        if entries == 0 || hits == 0 {
            failures.push(format!(
                "{label}: the second prompt did not restore a prefix ({entries} entries, {hits} hits)"
            ));
        }
        if saved < opening.len() as u64 {
            failures.push(format!(
                "{label}: only {saved} prefix positions were reused, want at least {}",
                opening.len()
            ));
        }
    }

    failures
}

#[test]
fn batching_is_invariant_across_slots() {
    // `mix` is the process-wide switch, so `Caches::PerRow` needs its own process. The child
    // below re-runs these checks with PEREGRINE_MIX=0.
    let perrow = !peregrine_core::options::get().mix;
    let dir = tmpdir("batch");
    let mut failures = Vec::new();
    let models = [
        (
            write_dense(&dir, "qwen3", Format::Q8_0, 4400, Dense::default()),
            "qwen3-Q8_0",
        ),
        (
            write_hybrid(&dir, Format::PQ2_0, 4401, false),
            "qwen35-PQ2_0",
        ),
    ];
    for (path, label) in &models {
        let model = load_arc(path);
        failures.extend(batching_failures(&model, label, perrow));
    }
    if !perrow {
        let exe = std::env::current_exe().expect("the test binary's path");
        let status = std::process::Command::new(exe)
            .args(["--exact", "perrow_batching_through_engine", "--nocapture"])
            .env("PEREGRINE_MIX", "0")
            .env("PEREGRINE_SYNTHETIC_CHILD", "1")
            .status()
            .expect("spawn the per-row child");
        if !status.success() {
            failures.push("the Caches::PerRow child process failed".into());
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "batching invariance failed:\n{}",
        failures.join("\n")
    );
}

/// Child of `batching_is_invariant_across_slots`: it runs only when spawned with
/// `PEREGRINE_MIX=0`, which routes decode through `Caches::PerRow`.
#[test]
fn perrow_batching_through_engine() {
    if std::env::var_os("PEREGRINE_SYNTHETIC_CHILD").is_none() {
        return;
    }
    let dir = tmpdir("batch-perrow");
    let mut failures = Vec::new();
    let models = [
        (
            write_dense(&dir, "qwen3", Format::Q8_0, 4410, Dense::default()),
            "qwen3-Q8_0/perrow",
        ),
        (
            write_hybrid(&dir, Format::PQ2_0, 4411, false),
            "qwen35-PQ2_0/perrow",
        ),
    ];
    for (path, label) in &models {
        let model = load_arc(path);
        failures.extend(batching_failures(&model, label, true));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "per-row batching invariance failed:\n{}",
        failures.join("\n")
    );
}

// ------------------------------------------------------------------- MTP acceptance --

/// The ordinary random MTP model rejects every draft. This variant's output head is constant, so
/// every logit ties and both the main model and the MTP block argmax to token 0: drafts are
/// accepted, and the token stream must still be exactly the one speculation-free decoding gives.
#[test]
fn hybrid_mtp_accepting_variant_accepts_drafts_losslessly() {
    let dir = tmpdir("hybrid-mtp-accept");
    let mut failures = Vec::new();
    for (i, fmt) in [Format::PQ2_0, Format::PTQ1_0].into_iter().enumerate() {
        let path = write_hybrid_accepting(&dir, fmt, 4500 + i as u64);
        let name = format!("qwen35-mtp-accept-{}", fmt.label());
        failures.extend(check(&path, &name));
        let model = load_arc(&path);
        if !model.has_mtp() {
            failures.push(format!("{name}: loaded without an MTP block"));
            continue;
        }
        let decode = |spec: bool| -> Result<(Vec<u32>, u64, u64), String> {
            let mut e = Engine::new(model.clone(), 4, 512, 1, KvPrec::F16, 0);
            if !spec {
                e.disable_mtp();
            }
            let cfg = SlotCfg {
                temperature: 0.0,
                max_tokens: 24,
                ..Default::default()
            };
            e.admit(PROMPT.to_vec(), cfg).map_err(|(_, msg)| msg)?;
            let mut toks = Vec::new();
            while e.active() > 0 {
                for em in e.step() {
                    if let Some(Finish::Error(err)) = em.finish {
                        return Err(err);
                    }
                    toks.push(em.token);
                }
            }
            Ok((toks, e.spec_drafts, e.spec_hits))
        };
        match (decode(true), decode(false)) {
            (Ok((with, drafts, hits)), Ok((without, _, _))) => {
                if drafts == 0 {
                    failures.push(format!("{name}: the drafter never ran"));
                }
                if hits == 0 {
                    failures.push(format!("{name}: no draft was ever accepted"));
                }
                if with != without {
                    failures.push(format!(
                        "{name}: accepted drafts changed the tokens\n  spec: {with:?}\n  plain: {without:?}"
                    ));
                }
            }
            (Err(e), _) | (_, Err(e)) => failures.push(format!("{name}: {e}")),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic MTP acceptance models failed:\n{}",
        failures.join("\n")
    );
}
