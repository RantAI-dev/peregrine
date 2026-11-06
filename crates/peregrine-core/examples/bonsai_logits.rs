// logits of every position of a fixed text, for accuracy comparisons between kernel paths.
// cargo run --release --example bonsai_logits -- <model.gguf> <out.bin>
use peregrine_core::gguf::Gguf;
use peregrine_core::kv::KvPrec;
use peregrine_core::model::{Model, State, StepStats};
use peregrine_core::pool::Pool;
use std::io::Write;
const TEXT: &str = "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable metal type, oil-based ink and a screw press. Within decades, presses spread across Europe, books became cheaper, and literacy rose. In 2024, a team measured 17 * 23 = 391 and wrote: def f(x): return x ** 2 + 1";
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let model = Model::load(Gguf::open(std::path::Path::new(&a[0])).unwrap()).unwrap();
    let c = model.cfg;
    let toks = model.tk.encode(TEXT);
    let n = toks.len();
    let pool = Pool::new(12);
    let mut st = State::new(&c, 128, n, 4096, 12, None);
    let mut kv = model.new_cache(KvPrec::F16, 4096);
    let want: Vec<usize> = (0..n).collect();
    let mut ss = StepStats::default();
    model
        .forward(&mut st, &mut kv, &pool, &toks, 0, &want, &mut ss)
        .unwrap();
    let mut f = std::fs::File::create(&a[1]).unwrap();
    f.write_all(&(n as u32).to_le_bytes()).unwrap();
    f.write_all(&(c.n_vocab as u32).to_le_bytes()).unwrap();
    for i in 0..n {
        for v in st.logits_row(i, c.n_vocab) {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    eprintln!("{n} positions");
}
