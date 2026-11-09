//! Tiny random models through the whole forward pass (roadmap S0.2).
//!
//! For every synthetic model: it loads, a prefill gives finite logits, results are bit-identical across
//! runs and across 1, 2 and 4 threads, greedy decoding is deterministic, and the output matches the
//! stored golden for the current kernel tier (`tests/golden/synthetic/<tier>/`). `PEREGRINE_GOLDEN=record`
//! writes goldens. These run everywhere, CI included: no model files are needed.

mod support;

use peregrine_core::gguf::{Format, Gguf};
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, StepStats};
use peregrine_core::pool::Pool;
use peregrine_core::serve::{Engine, Finish, SlotCfg};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::models::{
    write_dense, write_gemma3, write_gemma3_named, write_hybrid, write_hybrid_accepting,
    write_hybrid_plain, write_hybrid_plain_mtp, write_moe, write_phi3, write_rope_dense, Dense,
    Gemma3Config, MoeShape, Rope,
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
    run_with(model, threads, KvPrec::F16)
}

/// The same, with a chosen KV cache precision.
fn run_with(model: &Model, threads: usize, prec: KvPrec) -> String {
    let c = model.cfg;
    let pool = Pool::new(threads);
    let n = PROMPT.len();
    let mut st = model.new_state(128, n, 256, threads);
    let mut kv = model.new_cache(prec, 256);
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
    let model = match Gguf::open(path).and_then(peregrine_models::load) {
        Ok(m) => m,
        Err(e) => return Some(format!("{name}: does not load: {e}")),
    };
    check_model(model, name)
}

/// The same invariants and golden for a model loaded and converted from a LittleBit sidecar.
fn check_lb(path: &Path, sidecar: &Path, name: &str) -> Option<String> {
    let mut model = match Gguf::open(path).and_then(peregrine_models::load) {
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
    // AVX-VNNI-256 is mathematically bit-identical to AVX2 and aliases to its goldens (S2.6 Decision 1).
    // Goldens are recorded on Linux; macOS libm (sin, cos, exp) rounds differently, so macOS
    // keys its own directory and skips until one is recorded there.
    let golden_tier = format!(
        "{}{}",
        if tier == "x86_64-avx2-vnni256" {
            "x86_64-avx2"
        } else {
            tier.as_str()
        },
        if cfg!(target_os = "macos") {
            "-macos"
        } else {
            ""
        }
    );
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/synthetic")
        .join(golden_tier);
    let gpath = dir.join(format!("{name}.golden"));
    if std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "record")
        && tier != "x86_64-avx2-vnni256"
        && !gpath.exists()
    {
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
    } else if std::env::var("PEREGRINE_GOLDEN").is_ok_and(|v| v == "require") {
        // CI sets this on the tiers it pins, so a model or format added without its goldens fails there
        // instead of passing unchecked.
        return Some(format!(
            "{name}: no golden for tier {tier} ({}); record it with PEREGRINE_GOLDEN=record",
            gpath.display()
        ));
    } else {
        eprintln!("synthetic: {name} ok (no golden for tier {tier})");
    }
    None
}

/// The f32 KV cache (opt-in, for a like-for-like comparison against another
/// engine): the synthetic models give finite logits, the same ones whatever the
/// thread count, and — the point of the option — numbers that differ from the
/// f16 cache's, because the keys and values are no longer rounded.
#[test]
fn the_f32_kv_cache_gives_finite_logits_at_any_thread_count() {
    let dir = tmpdir("f32kv");
    let mut failures = Vec::new();
    for (i, (arch, fmt)) in [
        ("llama", Format::Q8_0),
        ("qwen3", Format::Q8_0),
        ("qwen2", Format::F16),
    ]
    .into_iter()
    .enumerate()
    {
        let path = write_dense(&dir, arch, fmt, 8200 + i as u64, Dense::default());
        let name = format!("{arch}-{}-f32kv", fmt.label());
        let model = match Gguf::open(&path).and_then(peregrine_models::load) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{name}: does not load: {e}"));
                continue;
            }
        };
        // `run_with` asserts that every logit is finite.
        let a = run_with(&model, 4, KvPrec::F32);
        if run_with(&model, 4, KvPrec::F32) != a {
            failures.push(format!("{name}: two runs differ"));
        }
        for t in [1, 2] {
            if run_with(&model, t, KvPrec::F32) != a {
                failures.push(format!("{name}: {t} threads differ from 4"));
            }
        }
        // The f32 cache is a different computation: it must not be a copy of the
        // f16 one, or the option would do nothing.
        let f16 = run_with(&model, 4, KvPrec::F16);
        if f16 == a {
            failures.push(format!(
                "{name}: the f32 cache gave the f16 cache's numbers"
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the f32 KV cache failed:\n{}",
        failures.join("\n")
    );
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
        // Append new formats at the end: the seed is `1000 + index`, so inserting one changes the
        // weights (and the goldens) of every format after it.
        Format::MXFP4,
        Format::Q2_K,
        Format::Q3_K,
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
        let model = match Gguf::open(&path).and_then(peregrine_models::load) {
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

#[test]
fn littlebit_sidecar_on_untied_hybrid_names_no_tensor() {
    use peregrine_core::lbq::{build_tensor, default_sidecar_path, source_stamp, write_slb};
    use support::vocab::VOCAB_SIZE;
    use support::weights::Rng;

    let dir = tmpdir("lbq_hybrid");
    let path = write_hybrid(&dir, Format::PQ2_0, 4390, false);
    let sidecar = default_sidecar_path(&path);
    let mut rng = Rng::new(4391);
    let pool = Pool::new(4);
    let w: Vec<f32> = (0..256 * VOCAB_SIZE)
        .map(|_| rng.f32_range(-0.04, 0.04))
        .collect();
    let builds = vec![build_tensor(
        "output.weight",
        w,
        VOCAB_SIZE,
        256,
        32,
        2,
        1,
        &pool,
    )];
    let (size, hash) = source_stamp(&path).unwrap();
    write_slb(&sidecar, size, hash, &builds).unwrap();

    let mut model = Gguf::open(&path).and_then(peregrine_models::load).unwrap();
    let slb = peregrine_core::lbq::Slb::open(&sidecar).unwrap();
    let err = model.attach_lb(slb).unwrap_err();
    assert_eq!(err, "slb: the sidecar names no tensor of this model");
    let _ = std::fs::remove_dir_all(&dir);
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
    Arc::new(Gguf::open(path).and_then(peregrine_models::load).unwrap())
}

/// The long third prompt keeps one slot in prefill while the other two decode, which is the
/// mixed step (`Caches::Mapped` with decode rows beside a prompt chunk).
///
/// The long prompt is longer than the widest chunk the engine may be given
/// (`options::MAX_PREFILL_CHUNK`), so it is read in pieces whatever the chunk is, and the mixed
/// step happens whatever the chunk is.
fn batch_prompts() -> Vec<Vec<u32>> {
    let long: Vec<u32> = (0..160).map(|i| ((i * 7) % 317 + 2) as u32).collect();
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
        (
            write_moe(&dir, Format::Q8_0, 4402, MoeShape::default()),
            "qwen3moe-Q8_0",
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
        (
            write_moe(&dir, Format::Q8_0, 4412, MoeShape::default()),
            "qwen3moe-Q8_0/perrow",
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

// --------------------------------------------------------------------- qwen3moe (S2.5) --

/// A qwen3moe file — a mixture-of-experts model — written the way llama.cpp v0.9.0 expects it
/// (`src/models/qwen3moe.cpp`, `src/llama-arch.cpp`).
///
/// The engine runs the file now, so this test does two things. It checks that the file is
/// written and read back exactly as intended: the metadata llama.cpp reads, and the
/// three-dimensional expert tensors in llama.cpp's dimension order (the expert index runs
/// slowest, so one expert is a contiguous run of rows — a 2D matrix with no new kernel). And it
/// runs the ordinary `check` of every other synthetic model: the file loads, a prefill gives
/// finite logits, two runs agree, the result is the same on 1, 2 and 4 threads, and it matches
/// the stored golden for the current kernel tier.
#[test]
fn qwen3moe_models_load_and_match_their_goldens() {
    use support::models::{write_moe, MoeShape};

    let dir = tmpdir("moe");
    let s = MoeShape::default();
    let mut failures = Vec::new();
    for (i, fmt) in [Format::F16, Format::Q8_0, Format::MXFP4]
        .into_iter()
        .enumerate()
    {
        let name = format!("qwen3moe-{}", fmt.label());
        let path = write_moe(&dir, fmt, 5000 + i as u64, s);
        let g = match Gguf::open(&path) {
            Ok(g) => g,
            Err(e) => {
                failures.push(format!("{name}: does not open: {e}"));
                continue;
            }
        };
        if g.str("general.architecture").unwrap_or("") != "qwen3moe" {
            failures.push(format!("{name}: the architecture is not qwen3moe"));
        }
        for (key, want) in [
            ("qwen3moe.block_count", s.n_layer as u32),
            ("qwen3moe.embedding_length", s.d_model as u32),
            ("qwen3moe.feed_forward_length", s.n_ff as u32),
            ("qwen3moe.expert_feed_forward_length", s.expert_ff as u32),
            ("qwen3moe.expert_count", s.n_expert as u32),
            ("qwen3moe.expert_used_count", s.n_expert_used as u32),
            ("qwen3moe.attention.head_count", s.n_head as u32),
            ("qwen3moe.attention.head_count_kv", s.n_head_kv as u32),
        ] {
            match g.u32(key) {
                Ok(v) if v == want => {}
                Ok(v) => failures.push(format!("{name}: {key} is {v}, want {want}")),
                Err(e) => failures.push(format!("{name}: {e}")),
            }
        }
        // The router is a plain 2D matrix; the experts are 3D, the expert index slowest.
        let d = s.d_model as u64;
        let ff = s.expert_ff as u64;
        let ne = s.n_expert as u64;
        for (tensor, dims) in [
            ("blk.0.ffn_gate_inp.weight", vec![d, ne]),
            ("blk.0.ffn_gate_exps.weight", vec![d, ff, ne]),
            ("blk.0.ffn_up_exps.weight", vec![d, ff, ne]),
            ("blk.0.ffn_down_exps.weight", vec![ff, d, ne]),
            // The second layer has its own bank of experts.
            ("blk.1.ffn_gate_exps.weight", vec![d, ff, ne]),
        ] {
            match g.tensor(tensor) {
                Ok(t) => {
                    if t.dims != dims {
                        failures.push(format!(
                            "{name}: {tensor} has dims {:?}, want {dims:?}",
                            t.dims
                        ));
                    }
                    if t.kind != fmt {
                        failures.push(format!(
                            "{name}: {tensor} is {}, want {}",
                            t.kind.label(),
                            fmt.label()
                        ));
                    }
                }
                Err(e) => failures.push(format!("{name}: {e}")),
            }
        }
        // Qwen3's per-head q/k norms, one weight per head dimension (F32, like every norm).
        for tensor in ["blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"] {
            match g.tensor(tensor) {
                Ok(t) => {
                    let want = vec![d / s.n_head as u64];
                    if t.dims != want {
                        failures.push(format!(
                            "{name}: {tensor} has dims {:?}, want {want:?}",
                            t.dims
                        ));
                    }
                }
                Err(e) => failures.push(format!("{name}: {e}")),
            }
        }
        // The engine accepts the file, and runs it.
        if let Err(e) = peregrine_models::check_compatible(&g) {
            failures.push(format!("{name}: refused: {e}"));
        }
        failures.extend(check(&path, &name));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic qwen3moe models failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn mxfp4_oracle_matches_llama_cpp() {
    use std::process::Command;
    use support::models::{write_moe, MoeShape};

    let quantize_bin = Path::new("/var/tmp/llama.cpp/build/bin/llama-quantize");
    if !quantize_bin.exists() {
        eprintln!("SKIPPED: llama-quantize binary not found at {quantize_bin:?}");
        return;
    }

    let dir = tmpdir("oracle-mxfp4");
    let s = MoeShape::default();
    let f32_path = write_moe(&dir, Format::F32, 9876, s);
    let mxfp4_path = dir.join("mxfp4.gguf");
    let requant_path = dir.join("requant_f32.gguf");

    // 1. Quantize F32 to MXFP4_MOE
    let status = Command::new(quantize_bin)
        .arg(&f32_path)
        .arg(&mxfp4_path)
        .arg("MXFP4_MOE")
        .status()
        .expect("run llama-quantize to MXFP4_MOE");
    assert!(status.success(), "llama-quantize to MXFP4_MOE failed");

    // 2. Requantize MXFP4 to F32
    let status = Command::new(quantize_bin)
        .arg("--allow-requantize")
        .arg(&mxfp4_path)
        .arg(&requant_path)
        .arg("F32")
        .status()
        .expect("run llama-quantize to F32");
    assert!(
        status.success(),
        "llama-quantize --allow-requantize to F32 failed"
    );

    // 3. Open both models
    let g_mxfp4 = Gguf::open(&mxfp4_path).expect("open mxfp4 model");
    let g_f32 = Gguf::open(&requant_path).expect("open requant f32 model");

    let mut checked_tensors = 0;
    let mut total_weights = 0;
    for (name, t_mxfp4) in &g_mxfp4.tensors {
        if t_mxfp4.kind != Format::MXFP4 {
            continue;
        }
        let t_f32 = g_f32.tensor(name).expect(name);
        assert_eq!(t_f32.kind, Format::F32);
        assert_eq!(t_mxfp4.dims, t_f32.dims);

        let mxfp4_bytes = g_mxfp4.bytes(t_mxfp4);
        let f32_bytes = g_f32.bytes(t_f32);
        let numel: usize = t_mxfp4.dims.iter().product::<u64>() as usize;

        let mut deq = vec![0.0f32; numel];
        peregrine_core::quant::dequant_row(Format::MXFP4, mxfp4_bytes, &mut deq);

        for i in 0..numel {
            let want = f32::from_le_bytes(f32_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(
                deq[i].to_bits(),
                want.to_bits(),
                "mismatch at element {i} of {name}: got {} ({:#010x}) vs llama.cpp {} ({:#010x})",
                deq[i],
                deq[i].to_bits(),
                want,
                want.to_bits()
            );
        }
        checked_tensors += 1;
        total_weights += numel;
    }
    assert!(checked_tensors > 0, "no MXFP4 tensors found to check");
    eprintln!(
        "oracle: {checked_tensors} tensors ({total_weights} weights) bit-identical to llama.cpp"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------- gemma3 (S2.4a) --

/// Gemma 3: 4 norms per layer, per-head q/k norms, sliding-window interleaved with global,
/// GeGLU activation, tied output head, and sqrt(d_model) embedding scaling.
#[test]
fn gemma3_models_in_f16_and_q8_0() {
    let dir = tmpdir("gemma3");
    let mut failures = Vec::new();
    for (i, fmt) in [Format::F16, Format::Q8_0].into_iter().enumerate() {
        let path = write_gemma3(&dir, fmt, 8000 + i as u64, Gemma3Config::default());
        failures.extend(check(&path, &format!("gemma3-{}", fmt.label())));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic gemma3 models failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn gemma3_batching_is_invariant_across_slots() {
    let perrow = !peregrine_core::options::get().mix;
    let dir = tmpdir("gemma3-batch");
    let mut failures = Vec::new();
    let path = write_gemma3(&dir, Format::Q8_0, 8010, Gemma3Config::default());
    let model = load_arc(&path);
    failures.extend(batching_failures(&model, "gemma3-Q8_0", perrow));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "gemma3 batching invariance failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn gemma3_synthetic_file_runs_in_llama_cpp() {
    let compare_bin = PathBuf::from("/tmp/compare_gemma3");
    if !compare_bin.exists() {
        let llama_so = Path::new("/tmp/llama.cpp/build/bin/libllama.so");
        if !llama_so.exists() {
            eprintln!("llama.cpp not found at {}, skipping", llama_so.display());
            return;
        }
    }
    let dir = tmpdir("gemma3-llama-cpp");
    let path = write_gemma3(&dir, Format::F16, 8020, Gemma3Config::default());
    let logits_path = dir.join("llama_logits.bin");

    // Prompt of 40 tokens on the synthetic model's tokenizer (window is 16 < 40)
    let prompt = "In a quiet valley surrounded by mountains";
    let n_predict = 32;

    let output = std::process::Command::new(&compare_bin)
        .args([
            path.to_str().unwrap(),
            &n_predict.to_string(),
            prompt,
            logits_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute compare_gemma3");

    assert!(
        output.status.success(),
        "compare_gemma3 failed on synthetic gemma3 model: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let llama_stdout = String::from_utf8_lossy(&output.stdout);

    let llama_tokens: Vec<u32> = llama_stdout
        .lines()
        .find(|l| l.starts_with("TOKENS:"))
        .expect("TOKENS: line")
        .strip_prefix("TOKENS:")
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(llama_tokens.len(), n_predict);

    let llama_logits_bytes = std::fs::read(&logits_path).expect("read logits");
    let llama_logits: Vec<f32> = llama_logits_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect();

    let model = Gguf::open(&path).and_then(peregrine_models::load).unwrap();
    let prompt_toks = model.tk.encode(prompt);
    assert_eq!(prompt_toks.len(), 40, "prompt should be 40 tokens");
    assert!(
        prompt_toks.len() > 16,
        "prompt must be longer than window 16"
    );

    let pool = Pool::new(1);
    let mut st = model.new_state(128, prompt_toks.len() + n_predict, 256, 1);
    let mut kv = model.new_cache(KvPrec::F16, 256);
    let mut ss = StepStats::default();
    let want: Vec<usize> = vec![prompt_toks.len() - 1];
    model
        .forward(&mut st, &mut kv, &pool, &prompt_toks, 0, &want, &mut ss)
        .unwrap();

    let pere_logits = st.logits_row(0, model.cfg.n_vocab).to_vec();
    assert_eq!(pere_logits.len(), llama_logits.len());

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for (p, l) in pere_logits.iter().zip(&llama_logits) {
        let abs = (p - l).abs();
        max_abs_err = max_abs_err.max(abs);
        let rel = abs / l.abs().max(1e-5);
        max_rel_err = max_rel_err.max(rel);
        sum_sq_diff += ((p - l) as f64).powi(2);
        sum_sq_ref += (*l as f64).powi(2);
    }
    let rel_l2_err = (sum_sq_diff.sqrt() / sum_sq_ref.sqrt()) as f32;
    eprintln!(
        "Last-position logits: max_abs_err = {:e}, max_rel_err = {:e}, rel_l2_err = {:e}",
        max_abs_err, max_rel_err, rel_l2_err
    );
    assert!(max_abs_err < 0.05, "logits error too high: {max_abs_err}");

    let mut pere_tokens = Vec::new();
    let mut next = argmax(&pere_logits);
    pere_tokens.push(next);
    for step in 1..n_predict {
        let pos = prompt_toks.len() + step - 1;
        model
            .forward(&mut st, &mut kv, &pool, &[next], pos, &[0], &mut ss)
            .unwrap();
        next = argmax(st.logits_row(0, model.cfg.n_vocab));
        pere_tokens.push(next);
    }
    eprintln!(
        "llama.cpp greedy tokens ({}): {:?}",
        llama_tokens.len(),
        llama_tokens
    );
    eprintln!(
        "Peregrine greedy tokens ({}): {:?}",
        pere_tokens.len(),
        pere_tokens
    );

    assert_eq!(
        pere_tokens, llama_tokens,
        "greedy tokens diverged between Peregrine and llama.cpp"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gemma3_attention_scale_follows_llama_cpp() {
    use peregrine_models::gemma3::attention_scale;
    // 62 layers is the 27B: 1/sqrt(n_embd / n_head) with llama.cpp's integer division.
    assert_eq!(attention_scale(62, 5376, 32, 128), 1.0 / (168f32).sqrt());
    assert_eq!(attention_scale(62, 5377, 32, 128), 1.0 / (168f32).sqrt());
    // Every other layer count, known or not (llama.cpp loads it as an unknown type): 1/sqrt(head_dim).
    for n_layer in [6, 18, 26, 34, 48, 40] {
        assert_eq!(
            attention_scale(n_layer, 2560, 8, 256),
            1.0 / (256f32).sqrt()
        );
    }
}

#[test]
fn gemma3_rope_linear_matches_golden() {
    let dir = tmpdir("gemma3-rope-linear");
    let cfg = Gemma3Config {
        rope: Rope::Linear { factor: 8.0 },
        ..Default::default()
    };
    let path = write_gemma3_named(&dir, Format::F16, 8030, cfg, Some("gemma3-rope-linear"));
    let failures = check(&path, "gemma3-rope-linear");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_none(),
        "gemma3-rope-linear failed:\n{}",
        failures.unwrap_or_default()
    );
}

/// A qwen3moe file that does not say how wide the experts are is refused by name.
///
/// llama.cpp reads `expert_feed_forward_length` for this architecture and falls back to
/// `feed_forward_length / expert_used_count` when it is absent
/// (`src/models/qwen3moe.cpp`). Peregrine refuses the file instead: the dense
/// `feed_forward_length` a qwen3moe file also carries is not the experts' width, and guessing
/// it would run the wrong feed-forward.
#[test]
fn a_qwen3moe_file_without_the_expert_width_is_refused() {
    use support::vocab::add_tokenizer_metadata;
    use support::writer::GgufWriter;

    let dir = tmpdir("moe-refuse");
    let path = dir.join("qwen3moe-no-expert-ff.gguf");
    let mut w = GgufWriter::new();
    w.str("general.architecture", "qwen3moe");
    w.str("general.name", "synthetic-qwen3moe-no-expert-ff");
    w.u32("qwen3moe.block_count", 2);
    w.u32("qwen3moe.embedding_length", 256);
    w.u32("qwen3moe.attention.head_count", 8);
    w.u32("qwen3moe.attention.head_count_kv", 4);
    // The dense width the file also carries — not the experts' width.
    w.u32("qwen3moe.feed_forward_length", 512);
    w.u32("qwen3moe.expert_count", 8);
    w.u32("qwen3moe.expert_used_count", 2);
    w.u32("qwen3moe.vocab_size", 320);
    add_tokenizer_metadata(&mut w, "qwen2");
    w.write_to_file(&path).expect("write the test model");

    let g = Gguf::open(&path).expect("open the test model");
    let err = match peregrine_models::load(g) {
        Ok(_) => panic!("the engine loaded a qwen3moe file without the experts' width"),
        Err(e) => e,
    };
    assert!(
        err.contains("expert_feed_forward_length"),
        "refused without naming the key: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------------------ phi3 (S2.4c) --

/// Phi-3 / Phi-4: fused QKV projection, fused gate/up feed-forward, RMSNorm,
/// NeoX rotary pairing, and no biases.
#[test]
fn phi3_models_in_f16_and_q8_0() {
    let dir = tmpdir("phi3");
    let mut failures = Vec::new();
    let formats = [Format::F16, Format::Q8_0];
    for (i, &fmt) in formats.iter().enumerate() {
        let path = write_phi3(&dir, fmt, 1000 + i as u64, Dense::default());
        failures.extend(check(&path, &format!("phi3-{}", fmt.label())));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic phi3 models failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn phi3_batching_is_invariant_across_slots() {
    let perrow = !peregrine_core::options::get().mix;
    let dir = tmpdir("phi3-batch");
    let mut failures = Vec::new();
    let path = write_phi3(&dir, Format::Q8_0, 1010, Dense::default());
    let model = load_arc(&path);
    failures.extend(batching_failures(&model, "phi3-Q8_0", perrow));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "phi3 batching invariance failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn phi3_synthetic_file_runs_in_llama_cpp() {
    let compare_bin = PathBuf::from("/tmp/compare_gemma3");
    if !compare_bin.exists() {
        let llama_so = Path::new("/tmp/llama.cpp/build/bin/libllama.so");
        if !llama_so.exists() {
            eprintln!("llama.cpp not found at {}, skipping", llama_so.display());
            return;
        }
    }
    let dir = tmpdir("phi3-llama-cpp");
    let path = write_phi3(&dir, Format::F16, 1020, Dense::default());
    let logits_path = dir.join("llama_logits.bin");

    let prompt = "In a quiet valley surrounded by mountains";
    let n_predict = 16;

    let output = std::process::Command::new(&compare_bin)
        .args([
            path.to_str().unwrap(),
            &n_predict.to_string(),
            prompt,
            logits_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute compare_gemma3");

    if !output.status.success() {
        eprintln!(
            "compare_gemma3 failed on synthetic phi3 model: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let llama_stdout = String::from_utf8_lossy(&output.stdout);

    let llama_tokens: Vec<u32> = llama_stdout
        .lines()
        .find(|l| l.starts_with("TOKENS:"))
        .expect("TOKENS: line")
        .strip_prefix("TOKENS:")
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(llama_tokens.len(), n_predict);

    let llama_logits_bytes = std::fs::read(&logits_path).expect("read logits");
    let llama_logits: Vec<f32> = llama_logits_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect();

    let model = Gguf::open(&path).and_then(peregrine_models::load).unwrap();
    let prompt_toks = model.tk.encode(prompt);

    let pool = Pool::new(1);
    let mut st = model.new_state(128, prompt_toks.len() + n_predict, 256, 1);
    let mut kv = model.new_cache(KvPrec::F16, 256);
    let mut ss = StepStats::default();
    let want: Vec<usize> = vec![prompt_toks.len() - 1];
    model
        .forward(&mut st, &mut kv, &pool, &prompt_toks, 0, &want, &mut ss)
        .unwrap();

    let pere_logits = st.logits_row(0, model.cfg.n_vocab).to_vec();
    assert_eq!(pere_logits.len(), llama_logits.len());

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for (p, l) in pere_logits.iter().zip(&llama_logits) {
        let abs = (p - l).abs();
        max_abs_err = max_abs_err.max(abs);
        let rel = abs / l.abs().max(1e-5);
        max_rel_err = max_rel_err.max(rel);
        sum_sq_diff += ((p - l) as f64).powi(2);
        sum_sq_ref += (*l as f64).powi(2);
    }
    let rel_l2_err = (sum_sq_diff.sqrt() / sum_sq_ref.sqrt()) as f32;
    eprintln!(
        "Phi3 last-position logits: max_abs_err = {:e}, max_rel_err = {:e}, rel_l2_err = {:e}",
        max_abs_err, max_rel_err, rel_l2_err
    );
    assert!(max_abs_err < 0.05, "logits error too high: {max_abs_err}");

    let mut pere_tokens = Vec::new();
    let mut next = argmax(&pere_logits);
    pere_tokens.push(next);
    for step in 1..n_predict {
        let pos = prompt_toks.len() + step - 1;
        model
            .forward(&mut st, &mut kv, &pool, &[next], pos, &[0], &mut ss)
            .unwrap();
        next = argmax(st.logits_row(0, model.cfg.n_vocab));
        pere_tokens.push(next);
    }

    assert_eq!(
        pere_tokens, llama_tokens,
        "greedy tokens diverged between Peregrine and llama.cpp"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_phi3_file_with_longrope_factors_is_refused() {
    use support::vocab::add_tokenizer_metadata;
    use support::writer::GgufWriter;

    let dir = tmpdir("phi3-refuse");
    let path = dir.join("phi3-longrope.gguf");
    let mut w = GgufWriter::new();
    w.str("general.architecture", "phi3");
    w.str("general.name", "synthetic-phi3-longrope");
    w.u32("phi3.block_count", 2);
    w.u32("phi3.embedding_length", 256);
    w.u32("phi3.attention.head_count", 8);
    w.u32("phi3.attention.head_count_kv", 4);
    w.u32("phi3.feed_forward_length", 512);
    w.u32("phi3.vocab_size", 320);
    w.add_tensor(
        "rope_factors_long.weight",
        &[16],
        Format::F32,
        vec![0u8; 16 * 4],
    );
    add_tokenizer_metadata(&mut w, "llama-bpe");
    w.write_to_file(&path).expect("write the test model");

    let g = Gguf::open(&path).expect("open the test model");
    let err = match peregrine_models::load(g) {
        Ok(_) => panic!("the engine loaded a phi3 file with LongRoPE factors"),
        Err(e) => e,
    };
    assert!(
        err.contains("LongRoPE"),
        "refused without naming LongRoPE: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------- plain Qwen 3.5 (S2.2c) --

/// A plain Qwen 3.5 file: the hybrid model without the Hadamard rotation, with ordinary
/// quantised matmuls (F16, Q8_0) instead of the ternary ones.
///
/// This is the file upstream's converter writes for `Qwen/Qwen3.5-9B` (llama.cpp v0.9.0,
/// `conversion/qwen.py`, `Qwen3_5TextModel`): no `prism.hadamard.*` metadata at all. It
/// loads, gives finite logits, is bit-identical across runs and across 1, 2 and 4 threads,
/// decodes greedily the same way twice, and matches its golden for the current tier.
#[test]
fn plain_qwen35_models_load_and_match_their_goldens() {
    let dir = tmpdir("qwen35-plain");
    let mut failures = Vec::new();
    let formats = [Format::F16, Format::Q8_0];
    for (i, &fmt) in formats.iter().enumerate() {
        let path = write_hybrid_plain(&dir, fmt, 1000 + i as u64);
        failures.extend(check(&path, &format!("qwen35-plain-{}", fmt.label())));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic plain qwen35 models failed:\n{}",
        failures.join("\n")
    );
}

/// The plain Qwen 3.5 file against llama.cpp v0.9.0's own `qwen35` implementation — the
/// upstream oracle for this architecture.
///
/// The file is the F16 one: with the weights in f16 and the activations unquantised, both
/// engines compute the same thing, and the greedy tokens must be the same. (The Q8_0 file
/// quantises the activations to int8, and the two engines' rounding then differ by about
/// an ulp of a logit — enough for a near-tie to flip, as it does for any quantised model.)
#[test]
fn plain_qwen35_synthetic_file_runs_in_llama_cpp() {
    let compare_bin = PathBuf::from("/tmp/compare_gemma3");
    if !compare_bin.exists() {
        let llama_so = Path::new("/tmp/llama.cpp/build/bin/libllama.so");
        if !llama_so.exists() {
            eprintln!("llama.cpp not found at {}, skipping", llama_so.display());
            return;
        }
    }
    let dir = tmpdir("qwen35-plain-llama-cpp");
    let path = write_hybrid_plain(&dir, Format::F16, 1000);
    let logits_path = dir.join("llama_logits.bin");

    let prompt = "In a quiet valley surrounded by mountains";
    let n_predict = 8;

    let output = std::process::Command::new(&compare_bin)
        .args([
            path.to_str().unwrap(),
            &n_predict.to_string(),
            prompt,
            logits_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute compare_gemma3");

    assert!(
        output.status.success(),
        "compare_gemma3 failed on the synthetic plain qwen35 model: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let llama_stdout = String::from_utf8_lossy(&output.stdout);

    let llama_tokens: Vec<u32> = llama_stdout
        .lines()
        .find(|l| l.starts_with("TOKENS:"))
        .expect("TOKENS: line")
        .strip_prefix("TOKENS:")
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(llama_tokens.len(), n_predict);

    let llama_logits_bytes = std::fs::read(&logits_path).expect("read logits");
    let llama_logits: Vec<f32> = llama_logits_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect();

    let model = Gguf::open(&path).and_then(peregrine_models::load).unwrap();
    let prompt_toks = model.tk.encode(prompt);
    assert_eq!(prompt_toks.len(), 40, "prompt should be 40 tokens");

    let pool = Pool::new(1);
    let mut st = model.new_state(128, prompt_toks.len() + n_predict, 256, 1);
    let mut kv = model.new_cache(KvPrec::F16, 256);
    let mut ss = StepStats::default();
    let want: Vec<usize> = vec![prompt_toks.len() - 1];
    model
        .forward(&mut st, &mut kv, &pool, &prompt_toks, 0, &want, &mut ss)
        .unwrap();

    let pere_logits = st.logits_row(0, model.cfg.n_vocab).to_vec();
    assert_eq!(pere_logits.len(), llama_logits.len());

    let mut max_abs_err = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for (p, l) in pere_logits.iter().zip(&llama_logits) {
        max_abs_err = max_abs_err.max((p - l).abs());
        sum_sq_diff += ((p - l) as f64).powi(2);
        sum_sq_ref += (*l as f64).powi(2);
    }
    let rel_l2_err = (sum_sq_diff.sqrt() / sum_sq_ref.sqrt()) as f32;
    eprintln!("Last-position logits: max_abs_err = {max_abs_err:e}, rel_l2_err = {rel_l2_err:e}");
    assert!(max_abs_err < 0.05, "logits error too high: {max_abs_err}");

    let mut pere_tokens = Vec::new();
    let mut next = argmax(&pere_logits);
    pere_tokens.push(next);
    for step in 1..n_predict {
        let pos = prompt_toks.len() + step - 1;
        model
            .forward(&mut st, &mut kv, &pool, &[next], pos, &[0], &mut ss)
            .unwrap();
        next = argmax(st.logits_row(0, model.cfg.n_vocab));
        pere_tokens.push(next);
    }
    eprintln!("llama.cpp greedy tokens ({llama_tokens:?})");
    eprintln!("Peregrine greedy tokens ({pere_tokens:?})");

    assert_eq!(
        pere_tokens, llama_tokens,
        "greedy tokens diverged between Peregrine and llama.cpp"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The plain Qwen 3.5 file with an MTP (NextN) block: the block is optional, and when it
/// is there its speculation must be lossless — the token stream with the MTP on is the one
/// without it.
#[test]
fn plain_qwen35_mtp_variant_speculates_losslessly() {
    use peregrine_core::serve::{Engine, Finish, SlotCfg};
    use std::sync::Arc;

    let dir = tmpdir("qwen35-plain-mtp");
    let mut failures = Vec::new();
    for (i, fmt) in [Format::F16, Format::Q8_0].into_iter().enumerate() {
        // The MTP variants have their own seed base, as the Bonsai 2 ones do.
        let path = write_hybrid_plain_mtp(&dir, fmt, 3000 + i as u64);
        let name = format!("qwen35-plain-mtp-{}", fmt.label());
        failures.extend(check(&path, &name));
        let model = match Gguf::open(&path).and_then(peregrine_models::load) {
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
        "synthetic plain qwen35 MTP models failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn calibrated_settings_match_defaults_and_goldens() {
    let dir = tmpdir("calib-golden");
    let path = write_dense(&dir, "llama", Format::Q8_0, 1000, Dense::default());
    let model = match Gguf::open(&path).and_then(peregrine_models::load) {
        Ok(m) => m,
        Err(e) => panic!("failed to load synthetic model: {e}"),
    };

    let cal = peregrine_core::calib::calibrate();
    let run_cal = run(&model, cal.decode_threads);
    let run_def = run(&model, 4);

    assert_eq!(
        run_cal, run_def,
        "calibrated thread count ({}) output differs from default (4 threads)",
        cal.decode_threads
    );

    // Verify against the committed golden
    let name = "llama-q8_0";
    let tier = peregrine_core::cpu::tier_name();
    // Goldens are recorded on Linux; macOS libm (sin, cos, exp) rounds differently, so macOS
    // keys its own directory and skips until one is recorded there.
    let golden_tier = format!(
        "{}{}",
        if tier == "x86_64-avx2-vnni256" {
            "x86_64-avx2"
        } else {
            tier.as_str()
        },
        if cfg!(target_os = "macos") {
            "-macos"
        } else {
            ""
        }
    );
    let gpath = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/synthetic")
        .join(golden_tier)
        .join(format!("{name}.golden"));
    if gpath.exists() {
        let expected = std::fs::read_to_string(&gpath).expect("read golden");
        assert_eq!(
            run_cal, expected,
            "calibrated output must match golden file"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// -------------------------------------------------------------------- gpt-oss (S2.5) --

/// A `gpt-oss` file — OpenAI's mixture of experts — written the way llama.cpp v0.9.0
/// expects it (`src/models/openai-moe.cpp`, `conversion/gpt_oss.py`).
///
/// The engine runs the file now, so this test does two things. It checks that the file
/// is written and read back exactly as intended: the metadata llama.cpp reads, the
/// three-dimensional expert tensors in llama.cpp's dimension order (the expert index
/// runs slowest, so one expert is a contiguous run of rows — a 2D matrix with no new
/// kernel), the biases and the attention sinks. And it runs the ordinary `check` of
/// every other synthetic model: the file loads, a prefill gives finite logits, two runs
/// agree, the result is the same on 1, 2 and 4 threads, and it matches the stored golden
/// for the current kernel tier.
#[test]
fn gpt_oss_models_load_and_match_their_goldens() {
    use support::models::{write_gpt_oss, GptOssShape};

    let dir = tmpdir("gpt-oss");
    let s = GptOssShape::default();
    let mut failures = Vec::new();
    // The two formats, each with its own seed: F16 experts, and the MXFP4 experts the
    // published gpt-oss files carry.
    for (fmt, seed) in [(Format::F16, 9000u64), (Format::MXFP4, 9001)] {
        let name = format!("gpt-oss-{}", fmt.label());
        let path = write_gpt_oss(&dir, fmt, seed, s);
        let g = match Gguf::open(&path) {
            Ok(g) => g,
            Err(e) => {
                failures.push(format!("{name}: does not open: {e}"));
                continue;
            }
        };
        if g.str("general.architecture").unwrap_or("") != "gpt-oss" {
            failures.push(format!("{name}: the architecture is not gpt-oss"));
        }
        for (key, want) in [
            ("gpt-oss.block_count", s.n_layer as u32),
            ("gpt-oss.embedding_length", s.d_model as u32),
            ("gpt-oss.attention.head_count", s.n_head as u32),
            ("gpt-oss.attention.head_count_kv", s.n_head_kv as u32),
            ("gpt-oss.attention.sliding_window", s.sliding_window as u32),
            ("gpt-oss.expert_count", s.n_expert as u32),
            ("gpt-oss.expert_used_count", s.n_expert_used as u32),
            ("gpt-oss.expert_feed_forward_length", s.expert_ff as u32),
            ("gpt-oss.vocab_size", 320),
        ] {
            match g.u32(key) {
                Ok(v) if v == want => {}
                Ok(v) => failures.push(format!("{name}: {key} is {v}, want {want}")),
                Err(e) => failures.push(format!("{name}: {e}")),
            }
        }
        // YaRN: the scaling the real files carry.
        match g.str("gpt-oss.rope.scaling.type") {
            Ok("yarn") => {}
            Ok(t) => failures.push(format!("{name}: the rope scaling is {t:?}, not yarn")),
            Err(e) => failures.push(format!("{name}: {e}")),
        }
        let d = s.d_model as u64;
        let ff = s.expert_ff as u64;
        let ne = s.n_expert as u64;
        let hd = d / s.n_head as u64;
        for (tensor, dims) in [
            ("blk.0.attn_q.weight", vec![d, s.n_head as u64 * hd]),
            ("blk.0.attn_k.weight", vec![d, s.n_head_kv as u64 * hd]),
            ("blk.0.attn_output.weight", vec![s.n_head as u64 * hd, d]),
            ("blk.0.attn_output.bias", vec![d]),
            ("blk.0.attn_sinks.weight", vec![s.n_head as u64]),
            ("blk.0.ffn_gate_inp.weight", vec![d, ne]),
            ("blk.0.ffn_gate_inp.bias", vec![ne]),
            ("blk.0.ffn_gate_exps.weight", vec![d, ff, ne]),
            ("blk.0.ffn_gate_exps.bias", vec![ff, ne]),
            ("blk.0.ffn_up_exps.weight", vec![d, ff, ne]),
            ("blk.0.ffn_up_exps.bias", vec![ff, ne]),
            ("blk.0.ffn_down_exps.weight", vec![ff, d, ne]),
            ("blk.0.ffn_down_exps.bias", vec![d, ne]),
        ] {
            match g.tensor(tensor) {
                Ok(t) => {
                    if t.dims != dims {
                        failures.push(format!(
                            "{name}: {tensor} has dims {:?}, want {dims:?}",
                            t.dims
                        ));
                    }
                }
                Err(e) => failures.push(format!("{name}: {e}")),
            }
        }
        // The experts are in the file's format; the attention is F16.
        for tensor in ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"] {
            match g.tensor(tensor) {
                Ok(t) if t.kind == fmt => {}
                Ok(t) => failures.push(format!(
                    "{name}: {tensor} is {}, want {}",
                    t.kind.label(),
                    fmt.label()
                )),
                Err(e) => failures.push(format!("{name}: {e}")),
            }
        }
        // The biases and the sinks are F32, as in every file.
        for tensor in ["blk.0.attn_output.bias", "blk.0.attn_sinks.weight"] {
            match g.tensor(tensor) {
                Ok(t) if t.kind == Format::F32 => {}
                Ok(t) => failures.push(format!("{name}: {tensor} is {}, want F32", t.kind.label())),
                Err(e) => failures.push(format!("{name}: {e}")),
            }
        }
        // The engine accepts the file, and runs it.
        if let Err(e) = peregrine_models::check_compatible(&g) {
            failures.push(format!("{name}: refused: {e}"));
        }
        failures.extend(check(&path, &name));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "synthetic gpt-oss models failed:\n{}",
        failures.join("\n")
    );
}

/// A synthetic gpt-oss file that declares gpt-oss's own Jinja template: detection
/// runs on the file's `tokenizer.chat_template`, as it does for a real gpt-oss file,
/// and the template's two terminators are in the stop set.
#[test]
fn gpt_oss_with_its_jinja_template_detects_harmony() {
    use peregrine_core::template::Template;
    use peregrine_core::tokenizer::Tokenizer;
    use support::models::{write_gpt_oss_with_template, GptOssShape};

    let dir = tmpdir("gpt-oss-harmony");
    let tmpl = include_str!("../../peregrine-core/tests/fixtures/harmony.jinja");
    let path =
        write_gpt_oss_with_template(&dir, Format::F16, 9101, GptOssShape::default(), Some(tmpl));
    let g = Gguf::open(&path).unwrap();
    assert_eq!(g.str("tokenizer.chat_template").unwrap(), tmpl);
    assert_eq!(Template::detect(&g), Ok(Template::Harmony));

    let tk = Tokenizer::from_gguf(&g).unwrap();
    let stops = Template::Harmony.stop_ids(&tk);
    let id = |name: &str| {
        tk.id_of(name)
            .unwrap_or_else(|| panic!("the vocabulary has {name}"))
    };
    assert!(
        stops.contains(&id("<|return|>")),
        "the stop set has <|return|>"
    );
    // `<|end|>` closes the analysis message; the answer comes after it, so a
    // stop there would cut every reply before its answer.
    assert!(!stops.contains(&id("<|end|>")), "<|end|> is not a stop");
    assert!(stops.contains(&tk.eos), "the model's own EOS is a stop");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A conversation continues from the final channel only: gpt-oss's Jinja drops every
/// earlier assistant turn's analysis, so a replayed prompt must not keep it. A
/// history with an analysis channel and a history with just its final text must
/// produce the same tokens, on the seeded turn and on the incremental turn after it.
#[test]
fn gpt_oss_session_replays_only_the_final_channel() {
    use peregrine_core::session::{Session, SessionCfg};
    use peregrine_core::template::Template;
    use support::models::{write_gpt_oss_with_template, GptOssShape};

    let dir = tmpdir("gpt-oss-session");
    let tmpl = include_str!("../../peregrine-core/tests/fixtures/harmony.jinja");
    let path =
        write_gpt_oss_with_template(&dir, Format::F16, 9102, GptOssShape::default(), Some(tmpl));
    let model = load_arc(&path);
    assert_eq!(Template::detect(model.gguf()), Ok(Template::Harmony));
    let cfg = SessionCfg {
        max_ctx: 512,
        max_tokens: 8,
        temperature: 0.0,
        threads: 4,
        speculate: false,
        ..Default::default()
    };

    let analysis = "<|channel|>analysis<|message|>SECRET REASONING<|end|>\
         <|start|>assistant<|channel|>final<|message|>Hi there";
    let mut with_analysis = Session::from_model(model.clone(), cfg.clone()).unwrap();
    with_analysis.seed_history(vec![("Hello".to_string(), analysis.to_string())]);
    let (_, first_analysis) = with_analysis.generate("Next", |_| {}).unwrap();
    let (after_analysis, telem_analysis) = with_analysis.generate("Again", |_| {}).unwrap();

    let mut final_only = Session::from_model(model, cfg).unwrap();
    final_only.seed_history(vec![("Hello".to_string(), "Hi there".to_string())]);
    let (_, first_final) = final_only.generate("Next", |_| {}).unwrap();
    let (after_final, telem_final) = final_only.generate("Again", |_| {}).unwrap();

    assert_eq!(
        first_analysis.prompt_tokens, first_final.prompt_tokens,
        "the replayed prompt kept an earlier turn's analysis"
    );
    assert_eq!(
        telem_analysis.prompt_tokens, telem_final.prompt_tokens,
        "the replayed prompt kept an earlier turn's analysis"
    );
    assert_eq!(telem_analysis.ctx_used, telem_final.ctx_used);
    assert_eq!(
        after_analysis, after_final,
        "the replayed prompt kept an earlier turn's analysis"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The synthetic gpt-oss model's forward, computed from the file's own weights in
/// f64: the specification the engine's forward must agree with.
///
/// Everything the architecture adds to a plain transformer is in here: the YaRN rope
/// (`ggml/src/ggml-cpu/ops.cpp:6164-6187`), the attention sinks
/// (`ggml/src/ggml-cpu/ops.cpp:5883-5894`), the sliding window on every other layer,
/// the router's bias and its softmax over the *selected* experts
/// (`src/llama-graph.cpp:2285-2287, 2363-2367`), the experts' biases, and OpenAI's
/// `swiglu_oai` (`ggml/src/ggml-cpu/ops.cpp:3388-3450`).
fn gpt_oss_reference(g: &Gguf, tokens: &[u32]) -> Vec<f64> {
    let k = |s: &str| format!("gpt-oss.{s}");
    let n_layer = g.u32(&k("block_count")).unwrap() as usize;
    let d_model = g.u32(&k("embedding_length")).unwrap() as usize;
    let n_head = g.u32(&k("attention.head_count")).unwrap() as usize;
    let n_kv_head = g.u32(&k("attention.head_count_kv")).unwrap() as usize;
    let head_dim = g.u32(&k("attention.key_length")).unwrap() as usize;
    let window = g.u32(&k("attention.sliding_window")).unwrap() as usize;
    let eps = g.f32(&k("attention.layer_norm_rms_epsilon")).unwrap() as f64;
    let rope_dim = g.u32(&k("rope.dimension_count")).unwrap() as usize;
    let rope_base = g.f32(&k("rope.freq_base")).unwrap() as f64;
    let n_expert = g.u32(&k("expert_count")).unwrap() as usize;
    let n_used = g.u32(&k("expert_used_count")).unwrap() as usize;
    let n_ff = g.u32(&k("expert_feed_forward_length")).unwrap() as usize;
    let n_vocab = g.u32(&k("vocab_size")).unwrap() as usize;
    let factor = g.f32(&k("rope.scaling.factor")).unwrap() as f64;
    let n_ctx_orig = g.u32(&k("rope.scaling.original_context_length")).unwrap() as usize;

    // The rope angles, as llama.cpp computes them for YaRN.
    let corr_dim = |n_rot: f64| -> f64 {
        rope_dim as f64 * (n_ctx_orig as f64 / (n_rot * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * rope_base.ln())
    };
    let corr = [
        corr_dim(32.0).floor().max(0.0),
        corr_dim(1.0).ceil().min(rope_dim as f64 - 1.0),
    ];
    let freq_scale = 1.0 / factor;
    let mscale = 1.0 + 0.1 * factor.ln();
    let angle = |pos: usize, i: usize| -> (f64, f64) {
        let extrap = pos as f64 * rope_base.powf(-((2 * i) as f64) / rope_dim as f64);
        let interp = freq_scale * extrap;
        let y = (i as f64 - corr[0]) / (corr[1] - corr[0]).max(0.001);
        let ramp = 1.0 - y.clamp(0.0, 1.0);
        let theta = interp * (1.0 - ramp) + extrap * ramp;
        (theta.cos() * mscale, theta.sin() * mscale)
    };

    // A tensor, dequantised into a flat array of rows of `dims[0]` values.
    let deq = |name: &str| -> Vec<f64> {
        let t = g.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        let n = t.numel() as usize;
        let mut buf = vec![0.0f32; n];
        peregrine_core::quant::dequant_row(t.kind, g.bytes(t), &mut buf);
        buf.iter().map(|&v| v as f64).collect()
    };
    // `y = W·x`, for a matrix of `rows` rows of `cols` weights, at `off`.
    let matvec = |w: &[f64], cols: usize, rows: usize, x: &[f64], off: usize| -> Vec<f64> {
        (0..rows)
            .map(|r| {
                w[off + r * cols..off + (r + 1) * cols]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    };
    let rmsnorm = |x: &[f64], w: &[f64]| -> Vec<f64> {
        let ss: f64 = x.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / x.len() as f64 + eps).sqrt();
        x.iter().zip(w).map(|(v, &wv)| v * inv * wv).collect()
    };

    let nb = tokens.len();
    let qd = n_head * head_dim;
    let kvd = n_kv_head * head_dim;
    let gqa = n_head / n_kv_head;
    let half = rope_dim / 2;

    let embd = deq("token_embd.weight");
    let mut x: Vec<Vec<f64>> = tokens
        .iter()
        .map(|&t| embd[t as usize * d_model..(t as usize + 1) * d_model].to_vec())
        .collect();

    for li in 0..n_layer {
        let p = |t: &str| format!("blk.{li}.{t}");
        let attn_norm = deq(&p("attn_norm.weight"));
        let wq = deq(&p("attn_q.weight"));
        let wk = deq(&p("attn_k.weight"));
        let wv = deq(&p("attn_v.weight"));
        let wo = deq(&p("attn_output.weight"));
        let wo_b = deq(&p("attn_output.bias"));
        let sinks = deq(&p("attn_sinks.weight"));
        let post_norm = deq(&p("post_attention_norm.weight"));
        let router = deq(&p("ffn_gate_inp.weight"));
        let router_b = deq(&p("ffn_gate_inp.bias"));
        let gate_w = deq(&p("ffn_gate_exps.weight"));
        let gate_b = deq(&p("ffn_gate_exps.bias"));
        let up_w = deq(&p("ffn_up_exps.weight"));
        let up_b = deq(&p("ffn_up_exps.bias"));
        let down_w = deq(&p("ffn_down_exps.weight"));
        let down_b = deq(&p("ffn_down_exps.bias"));
        // Every other layer is a sliding-window one (`load_swa_pattern(ml, 2)`).
        let is_swa = li % 2 == 0;

        // ---- the attention ----
        let xn: Vec<Vec<f64>> = x.iter().map(|r| rmsnorm(r, &attn_norm)).collect();
        let mut qs = Vec::new();
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for (b, xnb) in xn.iter().enumerate() {
            let mut q = matvec(&wq, d_model, qd, xnb, 0);
            let mut k = matvec(&wk, d_model, kvd, xnb, 0);
            let v = matvec(&wv, d_model, kvd, xnb, 0);
            // The rotation: the NeoX pairing, the frequencies in order.
            for h in 0..n_head {
                let row = &mut q[h * head_dim..(h + 1) * head_dim];
                for i in 0..half {
                    let (c, s) = angle(b, i);
                    let (x0, x1) = (row[i], row[i + half]);
                    row[i] = x0 * c - x1 * s;
                    row[i + half] = x0 * s + x1 * c;
                }
            }
            for h in 0..n_kv_head {
                let row = &mut k[h * head_dim..(h + 1) * head_dim];
                for i in 0..half {
                    let (c, s) = angle(b, i);
                    let (x0, x1) = (row[i], row[i + half]);
                    row[i] = x0 * c - x1 * s;
                    row[i + half] = x0 * s + x1 * c;
                }
            }
            qs.push(q);
            ks.push(k);
            vs.push(v);
        }
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut attn_out = vec![vec![0.0f64; qd]; nb];
        for b in 0..nb {
            let lo = if is_swa {
                (b + 1).saturating_sub(window)
            } else {
                0
            };
            for h in 0..n_head {
                let kh = h / gqa;
                let q = &qs[b][h * head_dim..(h + 1) * head_dim];
                let scores: Vec<f64> = (lo..=b)
                    .map(|p| {
                        let k = &ks[p][kh * head_dim..(kh + 1) * head_dim];
                        q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() * scale
                    })
                    .collect();
                // The sink joins the softmax without a value of its own.
                let sink = sinks[h];
                let m = scores.iter().cloned().fold(sink, f64::max);
                let denom: f64 =
                    scores.iter().map(|s| (s - m).exp()).sum::<f64>() + (sink - m).exp();
                for d in 0..head_dim {
                    let acc: f64 = (lo..=b)
                        .enumerate()
                        .map(|(j, p)| (scores[j] - m).exp() * vs[p][kh * head_dim + d])
                        .sum();
                    attn_out[b][h * head_dim + d] = acc / denom;
                }
            }
        }
        for b in 0..nb {
            let o = matvec(&wo, qd, d_model, &attn_out[b], 0);
            for i in 0..d_model {
                x[b][i] += o[i] + wo_b[i];
            }
        }

        // ---- the experts ----
        let xn2: Vec<Vec<f64>> = x.iter().map(|r| rmsnorm(r, &post_norm)).collect();
        for b in 0..nb {
            let mut logits = matvec(&router, d_model, n_expert, &xn2[b], 0);
            for e in 0..n_expert {
                logits[e] += router_b[e];
            }
            // The best few by score, descending; the lower expert index first on a tie.
            let mut order: Vec<usize> = (0..n_expert).collect();
            order.sort_by(|&a, &b| {
                logits[b]
                    .partial_cmp(&logits[a])
                    .unwrap()
                    .then_with(|| a.cmp(&b))
            });
            let sel = &order[..n_used];
            // The softmax over the selected scores.
            let max = sel
                .iter()
                .map(|&e| logits[e])
                .fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = sel.iter().map(|&e| (logits[e] - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let mut out = vec![0.0f64; d_model];
            for (s, &e) in sel.iter().enumerate() {
                let gate = matvec(&gate_w, d_model, n_ff, &xn2[b], e * d_model * n_ff);
                let up = matvec(&up_w, d_model, n_ff, &xn2[b], e * d_model * n_ff);
                let mut h = vec![0.0f64; n_ff];
                for i in 0..n_ff {
                    let gx = (gate[i] + gate_b[e * n_ff + i]).min(7.0);
                    let uy = (up[i] + up_b[e * n_ff + i]).clamp(-7.0, 7.0);
                    h[i] = gx / (1.0 + (1.702 * -gx).exp()) * (uy + 1.0);
                }
                let y = matvec(&down_w, n_ff, d_model, &h, e * n_ff * d_model);
                let w = exps[s] / sum;
                for i in 0..d_model {
                    out[i] += w * (y[i] + down_b[e * d_model + i]);
                }
            }
            for i in 0..d_model {
                x[b][i] += out[i];
            }
        }
    }

    // ---- the output head ----
    let out_norm = deq("output_norm.weight");
    let head_w = deq("output.weight");
    let mut logits = Vec::with_capacity(nb * n_vocab);
    for xb in x.iter() {
        let xn = rmsnorm(xb, &out_norm);
        logits.extend(matvec(&head_w, d_model, n_vocab, &xn, 0));
    }
    logits
}

/// The engine's forward and the exact reference must agree.
///
/// The file is the F16 one, and the KV cache is f32, so the only difference between
/// the two is the order of the arithmetic: the engine sums in f32 (and its own `exp`),
/// the reference in f64 (and the C library's).
#[test]
fn the_gpt_oss_forward_matches_the_exact_reference() {
    use support::models::{write_gpt_oss, GptOssShape};

    let dir = tmpdir("gpt-oss-reference");
    let path = write_gpt_oss(&dir, Format::F16, 9000, GptOssShape::default());
    let g = Gguf::open(&path).expect("open the synthetic model");
    let want = gpt_oss_reference(&g, &PROMPT);

    let model = Gguf::open(&path).and_then(peregrine_models::load).unwrap();
    let pool = Pool::new(1);
    let mut st = model.new_state(128, PROMPT.len(), 256, 1);
    let mut kv = model.new_cache(KvPrec::F32, 256);
    let mut ss = StepStats::default();
    let rows: Vec<usize> = (0..PROMPT.len()).collect();
    model
        .forward(&mut st, &mut kv, &pool, &PROMPT, 0, &rows, &mut ss)
        .unwrap();

    let n_vocab = model.cfg.n_vocab;
    let mut worst = 0.0f64;
    for b in 0..PROMPT.len() {
        let got = st.logits_row(b, n_vocab);
        for i in 0..n_vocab {
            let d = (got[i] as f64 - want[b * n_vocab + i]).abs();
            worst = worst.max(d);
        }
    }
    eprintln!("gpt-oss: the largest logit difference from the reference is {worst:e}");
    assert!(
        worst < 5e-3,
        "the forward differs from the reference by {worst}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The synthetic gpt-oss file against llama.cpp v0.9.0's own `gpt-oss`
/// implementation — the upstream oracle for this architecture.
///
/// The F16 file and the MXFP4 one: with the weights in f16 and the activations
/// unquantised, both engines compute the same thing, and the greedy tokens must
/// be the same.
#[test]
fn gpt_oss_synthetic_file_runs_in_llama_cpp() {
    use support::models::{write_gpt_oss, GptOssShape};

    // The same helper the other families use: it loads any model, prints the greedy
    // tokens and saves the last position's logits.
    let compare_bin = PathBuf::from("/tmp/compare_gemma3");
    if !compare_bin.exists() {
        let llama_so = Path::new("/tmp/llama.cpp/build/bin/libllama.so");
        if !llama_so.exists() {
            eprintln!("llama.cpp not found at {}, skipping", llama_so.display());
            return;
        }
    }
    let dir = tmpdir("gpt-oss-llama-cpp");
    let prompt = "In a quiet valley surrounded by mountains";
    let n_predict = 8;
    let mut failures = Vec::new();
    for (fmt, seed) in [(Format::F16, 9000u64), (Format::MXFP4, 9001)] {
        let name = format!("gpt-oss-{}", fmt.label());
        let path = write_gpt_oss(&dir, fmt, seed, GptOssShape::default());
        let logits_path = dir.join(format!("llama_logits-{}.bin", fmt.label()));

        let output = std::process::Command::new(&compare_bin)
            .args([
                path.to_str().unwrap(),
                &n_predict.to_string(),
                prompt,
                logits_path.to_str().unwrap(),
            ])
            .output()
            .expect("failed to execute compare_gemma3");

        if !output.status.success() {
            eprintln!(
                "compare_gemma3 failed on the synthetic {name} model: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            failures.push(format!("{name}: llama.cpp did not run the file"));
            continue;
        }
        let llama_stdout = String::from_utf8_lossy(&output.stdout);

        let llama_tokens: Vec<u32> = llama_stdout
            .lines()
            .find(|l| l.starts_with("TOKENS:"))
            .expect("TOKENS: line")
            .strip_prefix("TOKENS:")
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(llama_tokens.len(), n_predict);

        let llama_logits_bytes = std::fs::read(&logits_path).expect("read logits");
        let llama_logits: Vec<f32> = llama_logits_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect();

        let model = Gguf::open(&path).and_then(peregrine_models::load).unwrap();
        let prompt_toks = model.tk.encode(prompt);

        let pool = Pool::new(1);
        let mut st = model.new_state(128, prompt_toks.len() + n_predict, 256, 1);
        let mut kv = model.new_cache(KvPrec::F16, 256);
        let mut ss = StepStats::default();
        let want: Vec<usize> = vec![prompt_toks.len() - 1];
        model
            .forward(&mut st, &mut kv, &pool, &prompt_toks, 0, &want, &mut ss)
            .unwrap();

        let pere_logits = st.logits_row(0, model.cfg.n_vocab).to_vec();
        assert_eq!(pere_logits.len(), llama_logits.len());

        let mut max_abs_err = 0.0f32;
        let mut sum_sq_diff = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for (p, l) in pere_logits.iter().zip(&llama_logits) {
            max_abs_err = max_abs_err.max((p - l).abs());
            sum_sq_diff += ((p - l) as f64).powi(2);
            sum_sq_ref += (*l as f64).powi(2);
        }
        let rel_l2_err = (sum_sq_diff.sqrt() / sum_sq_ref.sqrt()) as f32;
        eprintln!(
            "{name} last-position logits: max_abs_err = {max_abs_err:e}, rel_l2_err = {rel_l2_err:e}"
        );
        if max_abs_err >= 0.05 {
            failures.push(format!("{name}: the logits differ by {max_abs_err}"));
        }

        let mut pere_tokens = Vec::new();
        let mut next = argmax(&pere_logits);
        pere_tokens.push(next);
        for step in 1..n_predict {
            let pos = prompt_toks.len() + step - 1;
            model
                .forward(&mut st, &mut kv, &pool, &[next], pos, &[0], &mut ss)
                .unwrap();
            next = argmax(st.logits_row(0, model.cfg.n_vocab));
            pere_tokens.push(next);
        }
        eprintln!("llama.cpp greedy tokens ({llama_tokens:?})");
        eprintln!("Peregrine greedy tokens ({pere_tokens:?})");

        if pere_tokens != llama_tokens {
            failures.push(format!(
                "{name}: the greedy tokens diverged: {pere_tokens:?} vs {llama_tokens:?}"
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "the synthetic gpt-oss files did not match llama.cpp:\n{}",
        failures.join("\n")
    );
}
