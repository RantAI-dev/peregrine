//! S6.6a measurements: decode throughput with and without a constraint, the mask's
//! per-token time, and the one-time vocabulary preparation.
//!
//! ```sh
//! cargo run --release -p peregrine-bench --example constrain_bench --features arch-qwen,arch-llama -- \
//!     ~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf 4
//! ```

use peregrine_core::constrain::{Constraint, Matcher, Schema, Vocabulary};
use peregrine_core::json::Value;
use peregrine_core::kv::KvPrec;
use peregrine_core::serve::Engine;
use peregrine_core::template::Template;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() {
        return 0.0;
    }
    v[v.len() / 2]
}

fn percentile(mut v: Vec<u64>, p: f64) -> u64 {
    v.sort_unstable();
    if v.is_empty() {
        return 0;
    }
    v[((v.len() - 1) as f64 * p) as usize]
}

fn schema(text: &str) -> Constraint {
    Constraint::from_schema(Schema::new(Value::parse(text).expect("schema")))
}

fn medium() -> Constraint {
    schema(
        r#"{"type":"object","properties":{
            "title":{"type":"string","maxLength":40},
            "count":{"type":"integer","minimum":0,"maximum":100},
            "kind":{"enum":["alpha","beta","gamma"]},
            "tags":{"type":"array","items":{"type":"string","maxLength":16},"maxItems":4},
            "meta":{"type":"object","properties":{"ok":{"type":"boolean"},"score":{"type":"number"}},"required":["ok"],"additionalProperties":false},
            "owner":{"type":"object","properties":{"name":{"type":"string","maxLength":20},"id":{"type":"integer","minimum":0,"maximum":999}},"required":["name","id"],"additionalProperties":false},
            "notes":{"type":"array","items":{"type":"string","maxLength":24},"maxItems":3},
            "active":{"type":"boolean"},
            "ratio":{"type":"number"},
            "label":{"type":"string","maxLength":12}
        },"required":["title","count","kind","tags","meta","owner","notes","active","ratio","label"],"additionalProperties":false}"#,
    )
}

fn large() -> Constraint {
    let mut props = String::new();
    let mut required = String::new();
    for i in 0..60 {
        if i > 0 {
            props.push(',');
            required.push(',');
        }
        let kind = match i % 6 {
            0 => format!(r#"{{"type":"integer","minimum":0,"maximum":999}}"#),
            1 => format!(r#"{{"type":"string","maxLength":16}}"#),
            2 => format!(r#"{{"type":"boolean"}}"#),
            3 => format!(r#"{{"type":"number"}}"#),
            4 => format!(r#"{{"enum":["a","b","c"]}}"#),
            _ => format!(
                r#"{{"type":"array","items":{{"type":"integer","minimum":0,"maximum":9}},"maxItems":2}}"#
            ),
        };
        props.push_str(&format!(r#""p{i}":{kind}"#));
        required.push_str(&format!(r#""p{i}""#));
    }
    let text = format!(
        r##"{{"$defs":{{"node":{{"type":"object","properties":{{"id":{{"type":"integer","minimum":0,"maximum":999}},"name":{{"type":"string","maxLength":16}},"children":{{"type":"array","items":{{"$ref":"#/$defs/node"}},"maxItems":2}}}},"required":["id","name"],"additionalProperties":false}}}},"type":"object","properties":{{"root":{{"$ref":"#/$defs/node"}},{props}}},"required":["root",{required}],"additionalProperties":false}}"##
    );
    schema(&text)
}

struct Run {
    tok_s: f64,
}

fn run(
    engine: &mut Engine,
    model: &std::sync::Arc<peregrine_core::model::Model>,
    prompt: &[u32],
    c: &Option<Constraint>,
) -> Run {
    let stop = Template::detect(model.gguf())
        .map(|t| t.stop_ids(&model.tk))
        .unwrap_or_else(|_| vec![model.tk.eos]);
    let cfg = peregrine_core::serve::SlotCfg {
        temperature: 0.0,
        top_p: 0.95,
        top_k: 0,
        min_p: 0.0,
        penalties: Default::default(),
        seed: 7,
        max_tokens: 400,
        stop,
        logprobs: false,
        turn: 0,
        kv_rotation: model.session_kv_rotation(),
        drafter: Default::default(),
        constraint: c.clone().map(std::sync::Arc::new),
    };
    let id = engine.admit(prompt.to_vec(), cfg).expect("admit");
    let mut cost = None;
    while engine.active() > 0 {
        for em in engine.step() {
            if let Some(peregrine_core::serve::Finish::Error(e)) = &em.finish {
                eprintln!("finish error: {e}");
            }
            if let Some(c) = &em.cost {
                cost = Some(c.clone());
            }
        }
    }
    engine.cancel(id);
    let cost = cost.expect("a finished slot");
    Run {
        tok_s: if cost.decode_s > 0.0 {
            cost.produced as f64 / cost.decode_s
        } else {
            0.0
        },
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("model path");
    let threads: usize = args.next().map_or(4, |s| s.parse().unwrap());
    let model = peregrine_models::open(std::path::Path::new(&path)).expect("load the model");
    let model = model;

    let t = Instant::now();
    let vocab = Vocabulary::new(&model.tk);
    let prep_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut preps = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        let v = Vocabulary::new(&model.tk);
        preps.push(t.elapsed().as_secs_f64() * 1e3);
        drop(v);
    }
    preps.sort_by(|a, b| a.total_cmp(b));
    println!(
        "model {} vocab {} prep_ms {:.1} prep_median_ms {:.1} prep_min_ms {:.1} prep_max_ms {:.1}",
        path,
        vocab.len(),
        prep_ms,
        preps[2],
        preps[0],
        preps[4]
    );

    if std::env::var_os("PEREGRINE_MICRO").is_some() {
        let cases: [(&str, Constraint, &[u8]); 6] = [
            ("any-object-key", Constraint::json_object(), b"{"),
            ("any-object-value", Constraint::json_object(), b"{\"a\":"),
            ("any-array-value", Constraint::json_object(), b"{\"a\":["),
            (
                "string-content",
                Constraint::json_object(),
                b"{\"abcdefghij\":",
            ),
            ("medium-value", medium(), b"{\"title\":"),
            ("medium-string", medium(), b"{\"title\":\"abcdefgh"),
        ];
        for (name, c, prefix) in cases {
            let mut m = Matcher::new(&c).unwrap();
            assert!(m.advance_bytes(prefix).is_ok(), "{name}");
            let t0 = Instant::now();
            let first = vocab.allowed(&m, &[0]).count();
            let first_ns = t0.elapsed().as_nanos();
            let n = 200;
            let t = Instant::now();
            let mut sink = 0usize;
            for _ in 0..n {
                sink += vocab.allowed(&m, &[0]).count();
            }
            let ns = t.elapsed().as_nanos() as f64 / n as f64;
            println!(
                "micro {name}: first {first_ns} ns, warm {ns:.0} ns/call (bits {sink} {first})"
            );
        }
    }

    // A compile check for the two schema shapes (and a Matcher, so the automaton is built).
    for c in [medium(), large(), Constraint::json_object()] {
        Matcher::new(&c).expect("the schema compiles");
    }

    let configs: [(&str, Option<Constraint>); 4] = [
        ("unconstrained", None),
        ("json_object", Some(Constraint::json_object())),
        ("medium", Some(medium())),
        ("large", Some(large())),
    ];
    let prompt = model.tk.encode_prompt(
        "Write a small JSON object describing a product with its name, price and tags.",
    );
    let mut engine = Engine::new(model.clone(), threads, 2048, 1, KvPrec::F16, 0);
    engine.disable_mtp();
    // One warm-up request per configuration builds the vocabulary's tries and the
    // boundary masks. The measured rounds then interleave the configurations, so a slow
    // spell on this shared host does not land on one row.
    let mut rates: Vec<Vec<f64>> = vec![Vec::new(); configs.len()];
    let mut masks: Vec<Vec<u64>> = vec![Vec::new(); configs.len()];
    let mut cold: Vec<Vec<u64>> = vec![Vec::new(); configs.len()];
    for (i, (_, c)) in configs.iter().enumerate() {
        let before = engine.mask_samples.len();
        let _ = run(&mut engine, &model, &prompt, c);
        cold[i].extend_from_slice(&engine.mask_samples[before..]);
    }
    for _ in 0..5 {
        for (i, (_, c)) in configs.iter().enumerate() {
            let before = engine.mask_samples.len();
            let r = run(&mut engine, &model, &prompt, c);
            masks[i].extend_from_slice(&engine.mask_samples[before..]);
            rates[i].push(r.tok_s);
        }
    }
    for (i, (name, _)) in configs.iter().enumerate() {
        let m = median(rates[i].clone());
        println!("{name}: tok_s median {m:.2} runs {:?}", rates[i]);
        for (label, list) in [("cold", &cold[i]), ("warm", &masks[i])] {
            if list.is_empty() {
                continue;
            }
            let mean = list.iter().sum::<u64>() as f64 / list.len() as f64;
            println!(
                "{name}: {label} mask calls {} median_ns {} p99_ns {} max_ns {} mean_ns {:.0}",
                list.len(),
                percentile(list.clone(), 0.5),
                percentile(list.clone(), 0.99),
                list.iter().copied().max().unwrap_or(0),
                mean
            );
        }
    }
}
