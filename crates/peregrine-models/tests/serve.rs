use std::sync::Arc;

use peregrine_core::kv::KvPrec;
use peregrine_core::model::Model;
use peregrine_core::serve::{Engine, SlotCfg};

fn cfg(max_tokens: usize) -> SlotCfg {
    SlotCfg {
        max_tokens,
        ..SlotCfg::default()
    }
}

/// The model these tests run against: `PEREGRINE_MODEL`, else `models/Qwen3-1.7B-Q4_K_M.gguf`.
fn model() -> Option<Arc<Model>> {
    let p = match std::env::var_os("PEREGRINE_MODEL") {
        Some(p) => std::path::PathBuf::from(p),
        None => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/Qwen3-1.7B-Q4_K_M.gguf"),
    };
    p.exists().then(|| peregrine_models::open(&p).ok())?
}

/// Two users served together must each get exactly the text they would get alone.
#[test]
fn users_served_together_get_what_they_would_get_alone() {
    let Some(m) = model() else {
        eprintln!("skipped: set PEREGRINE_MODEL");
        return;
    };
    let prompts = [
        vec![1u32, 450, 29871, 13, 5618],
        vec![1u32, 910, 338, 263, 1243],
    ];
    let steps = 8;
    let solo: Vec<String> = prompts
        .iter()
        .map(|p| {
            let mut e = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 0);
            e.admit(p.clone(), cfg(steps)).unwrap();
            let mut text = String::new();
            while e.active() > 0 {
                for em in e.step() {
                    text.push_str(&em.text);
                }
            }
            text
        })
        .collect();

    let mut e = Engine::new(m.clone(), 3, 256, 2, KvPrec::F16, 1 << 26);
    let a = e.admit(prompts[0].clone(), cfg(steps)).unwrap();
    let b = e.admit(prompts[1].clone(), cfg(steps)).unwrap();
    let (mut ta, mut tb) = (String::new(), String::new());
    while e.active() > 0 {
        for em in e.step() {
            if em.slot == a {
                ta.push_str(&em.text);
            } else if em.slot == b {
                tb.push_str(&em.text);
            }
        }
    }
    assert_eq!(ta, solo[0], "user A's text changed when sharing the engine");
    assert_eq!(tb, solo[1], "user B's text changed when sharing the engine");
    assert!(
        e.decode_rows > e.decode_steps,
        "the users never shared a pass"
    );
}

/// A user that arrives while another is already decoding joins the batch and is served correctly.
#[test]
fn a_user_can_arrive_mid_flight() {
    let Some(m) = model() else {
        eprintln!("skipped: set PEREGRINE_MODEL");
        return;
    };
    let late = vec![1u32, 910, 338, 263, 1243];
    let mut solo = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 0);
    solo.admit(late.clone(), cfg(6)).unwrap();
    let mut want = String::new();
    while solo.active() > 0 {
        for em in solo.step() {
            want.push_str(&em.text);
        }
    }

    let mut e = Engine::new(m.clone(), 3, 256, 2, KvPrec::F16, 1 << 26);
    let first = e.admit(vec![1u32, 450, 29871, 13, 5618], cfg(20)).unwrap();
    for _ in 0..6 {
        e.step();
    }
    let second = e.admit(late, cfg(6)).unwrap();
    let mut got = String::new();
    while e.active() > 0 {
        for em in e.step() {
            if em.slot == second {
                got.push_str(&em.text);
            }
        }
    }
    assert_eq!(
        got, want,
        "a user that arrived later was served differently"
    );
    assert!(first != second);
}

/// A second user whose prompt begins like the first one's does not prefill that beginning again.
#[test]
fn a_shared_beginning_is_prefilled_once_for_everyone() {
    let Some(m) = model() else {
        eprintln!("skipped: set PEREGRINE_MODEL");
        return;
    };
    // A long shared opening (a system prompt), then a different question each.
    let opening: Vec<u32> = (0..80).map(|i| ((i * 13) % 900 + 11) as u32).collect();
    let run = |e: &mut Engine, tail: u32| {
        let mut p = opening.clone();
        p.push(tail);
        let id = e.admit(p, cfg(3)).unwrap();
        let mut text = String::new();
        while e.active() > 0 {
            for em in e.step() {
                if em.slot == id {
                    text.push_str(&em.text);
                }
            }
        }
        text
    };
    let mut e = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 1 << 26);
    run(&mut e, 5618);
    let before = e.stats.bytes_weights;
    let second = run(&mut e, 6324);
    let (entries, _, hits, saved) = e.prefix_stats();
    assert!(
        entries > 0 && hits > 0,
        "the second user did not restore a prefix"
    );
    assert!(
        saved as usize >= opening.len() - 1,
        "only {saved} positions were reused"
    );
    // With the opening already in the cache, the second user's prefill is a chunk of weights, not three.
    let second_cost = e.stats.bytes_weights - before;
    assert!(
        second_cost < before,
        "the second user read as many weights as the first ({second_cost} vs {before})"
    );
    // And it is still the right answer: the same question on a cold engine gives the same text.
    let mut cold = Engine::new(m, 3, 256, 1, KvPrec::F16, 0);
    assert_eq!(
        run(&mut cold, 6324),
        second,
        "a restored prefix changed the answer"
    );
}

#[test]
fn a_full_engine_refuses_and_a_cancelled_slot_frees_its_room() {
    let Some(m) = model() else {
        eprintln!("skipped: set PEREGRINE_MODEL");
        return;
    };
    let mut e = Engine::new(m.clone(), 2, 128, 1, KvPrec::F16, 0);
    let a = e.admit(vec![1u32, 450, 29871], cfg(4)).unwrap();
    let refused = e.admit(vec![1u32, 450], cfg(4));
    assert!(refused.is_err(), "a full engine admitted a second request");
    assert_eq!(
        refused.unwrap_err().0,
        vec![1u32, 450],
        "the prompt is handed back"
    );
    e.cancel(a);
    assert_eq!(e.active(), 0);
    assert!(
        e.admit(vec![1u32, 450], cfg(4)).is_ok(),
        "cancelling did not free the slot"
    );
    // A prompt longer than the context is refused rather than truncated.
    let mut small = Engine::new(m.clone(), 2, 32, 1, KvPrec::F16, 0);
    assert!(small.admit(vec![7u32; 64], cfg(4)).is_err());
}
