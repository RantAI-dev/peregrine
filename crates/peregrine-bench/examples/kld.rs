// The KL divergence of Peregrine against llama.cpp, measured the way
// `llama-perplexity --kl-divergence` measures it: the same tokens, the same
// "score only the second half of each chunk" rule, the same statistics.
//
//   cargo run --release -p peregrine-bench --example kld -- --base <base.bin> [--model <model.gguf>] [--threads N]
//
// `--base` is a file written by `llama-perplexity --kl-divergence-base`. With
// `--model` the tool runs Peregrine over the same tokens and prints the
// comparison; without it, it only decodes the file and prints what the file
// holds, which is the check that the reader understands the format.
//
// The file format is llama.cpp's (`tools/perplexity/perplexity.cpp`):
//
//   "_logits_"            8 bytes
//   u32 n_ctx
//   i32 n_vocab
//   i32 n_chunk
//   i32 tokens[n_ctx * n_chunk]        the evaluation tokens, before BOS
//   then, per chunk, per scored position:
//     f32 scale, f32 min_log_prob        (the first 8 bytes of the record)
//     u16 q[n_vocab]                     the log-probabilities, quantised
//
// One record is `2*((n_vocab + 1)/2) + 4` u16 wide; the quantised values start
// at byte 8 of the record.
use peregrine_core::gguf::Gguf;
use peregrine_core::kv::{KvCache, KvPrec};
use peregrine_core::model::{Model, State, StepStats};
use peregrine_core::pool::Pool;
use peregrine_core::serve::PREFILL_CHUNK;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

/// The eight bytes llama.cpp writes at the head of a `--kl-divergence-base` file.
const MAGIC: &[u8; 8] = b"_logits_";

/// The largest batch one matmul call takes (`peregrine_core::blocks::linear::MAX_BATCH`,
/// which is not public). A prefill piece may not be wider than this.
const MAX_BATCH: usize = 128;

/// The arguments.
struct Args {
    base: PathBuf,
    model: Option<PathBuf>,
    threads: usize,
    /// How many tokens the prefill is fed in at a time.
    prefill: usize,
    /// The precision of the KV cache. `f32` keeps the keys and values in full
    /// precision, which is what a like-for-like comparison against another engine
    /// (whose attention reads f16 operands) needs.
    kv: KvPrec,
}

/// What llama.cpp's `kl_divergence_result` accumulates, one entry per scored position.
#[derive(Default, Clone)]
struct Kld {
    sum_nll: f64,
    sum_nll2: f64,
    sum_nll_base: f64,
    sum_nll_base2: f64,
    sum_nll_nll_base: f64,
    sum_kld: f64,
    sum_kld2: f64,
    sum_p_diff: f64,
    sum_p_diff2: f64,
    sum_p_diff4: f64,
    max_p_diff: f64,
    n_same_top: usize,
    count: usize,
}

/// One scored position: llama.cpp's `log_softmax` (the `--kl-divergence` overload).
struct Pos {
    nll: f64,
    nll_base: f64,
    kld: f64,
    p_diff: f64,
    same_top: bool,
}

/// The comparison of one scored position, in llama.cpp's arithmetic and its precision.
///
/// `logits` are Peregrine's, `rec` is one base record and `tok` the token the
/// position predicts.
fn compare(n_vocab: usize, logits: &[f32], rec: &[u8], tok: usize) -> Pos {
    let scale = f32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
    let min_log_prob = f32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]);
    let q = &rec[8..8 + 2 * n_vocab];

    let mut max_logit = logits[0];
    let mut imax = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > max_logit {
            max_logit = l;
            imax = i;
        }
    }
    let mut sum_exp = 0.0f64;
    for &l in logits.iter() {
        sum_exp += (l - max_logit).exp() as f64;
    }
    let log_sum_exp = sum_exp.ln() as f32;

    let nll = (max_logit + log_sum_exp - logits[tok]) as f64;

    let q_tok = u16::from_le_bytes([q[2 * tok], q[2 * tok + 1]]) as f32;
    let nll_base = (-(scale * q_tok + min_log_prob)) as f64;

    // KL(base || Peregrine) over the base's non-negligible support, and the base's argmax.
    let lse = max_logit + log_sum_exp;
    let mut sum = 0.0f64;
    let mut imax_base = 0usize;
    let mut p_log_base_max = 0.0f32;
    for (i, (&l, w)) in logits.iter().zip(q.as_chunks::<2>().0.iter()).enumerate() {
        let qv = u16::from_le_bytes([w[0], w[1]]) as f32;
        let p_log_base = scale * qv + min_log_prob;
        if i == 0 || p_log_base > p_log_base_max {
            p_log_base_max = p_log_base;
            imax_base = i;
        }
        if p_log_base > -16.0f32 {
            let p_base = p_log_base.exp();
            sum += (p_base * (p_log_base - l + lse)) as f64;
        }
    }

    let p_diff = (-nll).exp() - (-nll_base).exp();
    Pos {
        nll,
        nll_base,
        kld: sum,
        p_diff,
        same_top: imax == imax_base,
    }
}

/// What the base file alone says about one scored position: the negative
/// log-likelihood of the target under the base distribution, and the base's argmax.
fn base_only(rec: &[u8], tok: usize) -> (f64, usize) {
    let scale = f32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
    let min_log_prob = f32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]);
    let q = &rec[8..];
    let q_tok = u16::from_le_bytes([q[2 * tok], q[2 * tok + 1]]) as f32;
    let nll_base = (-(scale * q_tok + min_log_prob)) as f64;
    let mut imax_base = 0usize;
    let mut p_log_base_max = f32::NEG_INFINITY;
    for (i, w) in q.as_chunks::<2>().0.iter().enumerate() {
        let qv = u16::from_le_bytes([w[0], w[1]]) as f32;
        let p_log_base = scale * qv + min_log_prob;
        if i == 0 || p_log_base > p_log_base_max {
            p_log_base_max = p_log_base;
            imax_base = i;
        }
    }
    (nll_base, imax_base)
}

fn mean_and_uncertainty(sum: f64, sum2: f64, count: usize) -> (f64, f64) {
    if count < 1 {
        return (0.0, 0.0);
    }
    let f = sum / count as f64;
    let mut df = sum2 / count as f64 - f * f;
    df = if df > 0.0 && count > 10 {
        (df / (count - 1) as f64).sqrt()
    } else {
        0.0
    };
    (f, df)
}

fn covariance(suma: f64, sumb: f64, sumab: f64, count: usize) -> f64 {
    if count < 10 {
        return 0.0;
    }
    let n = count as f64;
    let mut var = sumab / n - (suma / n) * (sumb / n);
    var /= n - 1.0;
    var
}

/// llama.cpp's percentile: linear interpolation between the two neighbours of the fraction.
fn percentile(values: &[f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    if fraction <= 0.0 {
        return values[0];
    }
    if fraction >= 1.0 {
        return values[values.len() - 1];
    }
    let p = fraction * (values.len() - 1) as f32;
    let ip = p as usize;
    let p = p - ip as f32;
    (1.0 - p) * values[ip] + p * values[ip.min(values.len() - 1)]
}

/// llama.cpp's rule for adding the BOS token: the pre-tokenizer decides, the file may override.
fn llama_add_bos(g: &Gguf) -> bool {
    let pre = g.str("tokenizer.ggml.pre").unwrap_or("default");
    let model = g.str("tokenizer.ggml.model").unwrap_or("gpt2");
    let mut add_bos = match model {
        // SentencePiece adds the BOS by default.
        "llama" => true,
        // BPE: only the Llama 3 family does.
        _ => matches!(
            pre,
            "llama3"
                | "llama-v3"
                | "llama-bpe"
                | "falcon3"
                | "falcon-h1"
                | "midm-2.0"
                | "jina-v5-nano"
                | "lfm2"
                | "tekken"
                | "tekken-num3"
                | "pixtral"
                | "chameleon"
                | "nemotron-3.5-embed"
        ),
    };
    if let Some(b) = g.bool("tokenizer.ggml.add_bos_token") {
        add_bos = b;
    }
    add_bos
}

fn parse_args() -> Result<Args, String> {
    let mut base = None;
    let mut model = None;
    let mut threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let mut prefill = PREFILL_CHUNK;
    let mut kv = KvPrec::F16;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--base" => base = Some(PathBuf::from(it.next().ok_or("--base needs a file")?)),
            "--model" => model = Some(PathBuf::from(it.next().ok_or("--model needs a file")?)),
            "--threads" => {
                threads = it
                    .next()
                    .ok_or("--threads needs a number")?
                    .parse()
                    .map_err(|_| "--threads needs a number")?;
            }
            "--prefill" => {
                prefill = it
                    .next()
                    .ok_or("--prefill needs a number")?
                    .parse()
                    .map_err(|_| "--prefill needs a number")?;
            }
            "--kv" => {
                let v = it.next().ok_or("--kv needs a precision")?;
                kv = KvPrec::parse(&v)
                    .ok_or_else(|| format!("--kv: unknown precision {v:?} — f16, f32 or i8"))?;
            }
            "-h" | "--help" => {
                println!(
                    "usage: kld --base <base.bin> [--model <model.gguf>] [--threads N] [--prefill N] [--kv f16|f32|i8]\n\
                     \n\
                     Reads a llama.cpp `--kl-divergence-base` file. With --model it runs\n\
                     Peregrine over the same tokens and prints the KL divergence against\n\
                     llama.cpp, the way `llama-perplexity --kl-divergence` does.\n\
                     \n\
                     --prefill is how many tokens the prefill is fed in at a time. The\n\
                     default is the engine's own {PREFILL_CHUNK}. The result does not\n\
                     depend on it: every row of a matmul is computed the same way,\n\
                     whatever else shares the call.\n\
                     \n\
                     --kv is the precision of the KV cache (f16 by default). With f32\n\
                     the keys and values attention reads are the ones the model produced,\n\
                     with no rounding — the setting a like-for-like comparison against\n\
                     another engine needs."
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        base: base.ok_or("usage: kld --base <base.bin> [--model <model.gguf>] [--threads N]")?,
        model,
        threads: threads.max(1),
        prefill: prefill.clamp(1, MAX_BATCH),
        kv,
    })
}

/// Read the header of a base file: the context size, the vocabulary size, the chunk count and the tokens.
fn read_header(r: &mut impl Read) -> Result<(u32, usize, usize, Vec<u32>), String> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)
        .map_err(|e| format!("reading the magic: {e}"))?;
    if &magic != MAGIC {
        return Err("this does not look like a file containing log-probabilities".into());
    }
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| format!("reading n_ctx: {e}"))?;
    let n_ctx = u32::from_le_bytes(buf);
    r.read_exact(&mut buf)
        .map_err(|e| format!("reading n_vocab: {e}"))?;
    let n_vocab = i32::from_le_bytes(buf);
    r.read_exact(&mut buf)
        .map_err(|e| format!("reading n_chunk: {e}"))?;
    let n_chunk = i32::from_le_bytes(buf);
    if n_vocab <= 0 || n_chunk <= 0 || n_ctx < 4 {
        return Err(format!(
            "nonsense header: n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk}"
        ));
    }
    let n = n_ctx as usize * n_chunk as usize;
    let mut raw = vec![0u8; n * 4];
    r.read_exact(&mut raw)
        .map_err(|e| format!("reading the evaluation tokens: {e}"))?;
    let tokens = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    Ok((n_ctx, n_vocab as usize, n_chunk as usize, tokens))
}

/// The model, and the engine state that goes with it.
struct Loaded {
    path: PathBuf,
    model: Model,
    pool: Pool,
    state: State,
    kv: KvCache,
    stats: StepStats,
    n_vocab: usize,
    add_bos: bool,
    prefill: usize,
}

impl Loaded {
    /// Run one chunk and compare it with the base records that go with it.
    ///
    /// `rec` holds the records of the chunk, one per scored position.
    #[allow(clippy::too_many_arguments)]
    fn run_chunk(
        &mut self,
        chunk_tokens: &[u32],
        first: usize,
        rec: &[u8],
        rec_bytes: usize,
        kld: &mut Kld,
        kld_values: &mut Vec<f32>,
        p_diff_values: &mut Vec<f32>,
    ) -> Result<(), String> {
        // llama.cpp replaces the first token of every chunk with the BOS before
        // the forward pass, and puts the original back afterwards.
        let mut toks: Vec<u32> = chunk_tokens.to_vec();
        if self.add_bos {
            toks[0] = self.model.tk.bos;
        }
        let n = toks.len();

        // The chunk starts at position 0 with an empty cache, exactly as llama.cpp
        // clears its KV cache between chunks. The prefill is fed in the engine's
        // own chunks: one matmul call takes at most `MAX_BATCH` rows, and the
        // engine prefills in `PREFILL_CHUNK` pieces.
        self.kv.clear();
        let mut p = 0usize;
        while p < n {
            let e = (p + self.prefill).min(n);
            // The rows that carry logits: the second half of the chunk, except
            // its last position, which predicts nothing. The rows are counted
            // from the start of this piece.
            let lo = first.max(p);
            let hi = e.min(n - 1);
            let want: Vec<usize> = if lo < hi {
                (lo - p..hi - p).collect()
            } else {
                Vec::new()
            };
            self.model.forward(
                &mut self.state,
                &mut self.kv,
                &self.pool,
                &toks[p..e],
                p,
                &want,
                &mut self.stats,
            )?;

            for (k, &row) in want.iter().enumerate() {
                let abs = p + row;
                let j = abs - first;
                let rec_j = &rec[j * rec_bytes..(j + 1) * rec_bytes];
                let tok = toks[abs + 1] as usize;
                let p = compare(
                    self.n_vocab,
                    self.state.logits_row(k, self.n_vocab),
                    rec_j,
                    tok,
                );
                kld.sum_nll += p.nll;
                kld.sum_nll2 += p.nll * p.nll;
                kld.sum_nll_nll_base += p.nll * p.nll_base;
                kld.sum_kld += p.kld;
                kld.sum_kld2 += p.kld * p.kld;
                kld.sum_p_diff += p.p_diff;
                kld.sum_p_diff2 += p.p_diff * p.p_diff;
                kld.sum_p_diff4 += p.p_diff * p.p_diff * p.p_diff * p.p_diff;
                kld.max_p_diff = kld.max_p_diff.max(p.p_diff.abs());
                if p.same_top {
                    kld.n_same_top += 1;
                }
                kld.count += 1;
                kld_values.push(p.kld as f32);
                p_diff_values.push(p.p_diff as f32);
            }
            p = e;
        }
        Ok(())
    }
}

fn load_model(
    path: &Path,
    threads: usize,
    n_ctx: usize,
    prefill: usize,
    kv: KvPrec,
) -> Result<Loaded, String> {
    let g = Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let add_bos = llama_add_bos(&g);
    let model = peregrine_models::load(g).map_err(|e| format!("{}: {e}", path.display()))?;
    let c = model.cfg;
    let pool = Pool::new(threads);
    let n_vocab = c.n_vocab;
    // One prefill piece is one batch; the rows that carry logits are the second half.
    let state = State::new(
        &c,
        prefill,
        prefill,
        n_ctx,
        pool.width(),
        model.rope_factors.as_deref(),
    );
    let kv = model.new_cache(kv, n_ctx);
    Ok(Loaded {
        path: path.to_path_buf(),
        model,
        pool,
        state,
        kv,
        stats: StepStats::default(),
        n_vocab,
        add_bos,
        prefill,
    })
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&args) {
        eprintln!("kld: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), String> {
    let f = std::fs::File::open(&args.base).map_err(|e| format!("{}: {e}", args.base.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let (n_ctx, n_vocab, n_chunk, tokens) = read_header(&mut r)?;

    // llama.cpp scores the second half of every chunk: `first = n_ctx/2`, and the
    // last position of the chunk is not scored because it predicts nothing.
    let first = n_ctx as usize / 2;
    let n_scored = n_ctx as usize - 1 - first;
    let nv = 2 * n_vocab.div_ceil(2) + 4; // the record stride, in u16
    let rec_bytes = nv * 2;

    println!("base file    {}", args.base.display());
    println!("  n_ctx {n_ctx}, n_vocab {n_vocab}, n_chunk {n_chunk}");
    println!(
        "  {n_scored} scored positions per chunk (the second half, the last one excluded), {} in total",
        n_scored * n_chunk
    );

    let mut engine = match &args.model {
        Some(p) => Some(load_model(
            p,
            args.threads,
            n_ctx as usize,
            args.prefill,
            args.kv,
        )?),
        None => None,
    };
    if let Some(m) = &engine {
        println!("model        {}", m.path.display());
        println!("  n_vocab {}, add_bos {}", m.n_vocab, m.add_bos);
        println!("  KV cache    {}", args.kv.name());
        if m.n_vocab != n_vocab {
            return Err(format!(
                "the base file was made with a vocabulary of {n_vocab}, this model has {}",
                m.n_vocab
            ));
        }
    }

    // One record per scored position, for the chunk being read.
    let mut rec = vec![0u8; n_scored * rec_bytes];
    let mut kld = Kld::default();
    let mut kld_values: Vec<f32> = Vec::with_capacity(n_scored * n_chunk);
    let mut p_diff_values: Vec<f32> = Vec::with_capacity(n_scored * n_chunk);
    let mut base_top1_hits = 0usize;
    let mut base_top1_count = 0usize;

    for chunk in 0..n_chunk {
        r.read_exact(&mut rec)
            .map_err(|e| format!("reading the log-probabilities of chunk {chunk}: {e}"))?;

        let chunk_tokens = &tokens[chunk * n_ctx as usize..(chunk + 1) * n_ctx as usize];

        // What the base file says, independent of any model: the perplexity the
        // reference engine recorded, and how often its own best guess was the
        // token that followed. The second number is the check that the token
        // array and the records line up.
        for j in 0..n_scored {
            let rec_j = &rec[j * rec_bytes..(j + 1) * rec_bytes];
            let tok = chunk_tokens[first + j + 1] as usize;
            let (nll_base, imax_base) = base_only(rec_j, tok);
            kld.sum_nll_base += nll_base;
            kld.sum_nll_base2 += nll_base * nll_base;
            base_top1_count += 1;
            if imax_base == tok {
                base_top1_hits += 1;
            }
        }

        if let Some(m) = &mut engine {
            m.run_chunk(
                chunk_tokens,
                first,
                &rec,
                rec_bytes,
                &mut kld,
                &mut kld_values,
                &mut p_diff_values,
            )?;
            // The same line llama.cpp prints per chunk, over the positions seen so far.
            let ppl = mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count)
                .0
                .exp();
            let kl = mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count).0;
            let top = 100.0 * kld.n_same_top as f64 / kld.count as f64;
            println!(
                "{:4}    PPL {:9.4}    KLD {:9.5}    same top p {:6.3} %",
                chunk + 1,
                ppl,
                kl,
                top
            );
        }
    }

    if base_top1_count > 0 {
        println!();
        println!("base file only");
        println!(
            "  Mean PPL(base)                : {:10.6}",
            mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, base_top1_count)
                .0
                .exp()
        );
        println!(
            "  base top-1 == target          : {:10.4} %   (the token array lines up)",
            100.0 * base_top1_hits as f64 / base_top1_count as f64
        );
    }

    if engine.is_none() {
        return Ok(());
    }

    // The report, in llama.cpp's layout.
    if kld.count < 100 {
        eprintln!(
            "kld: only {} scored positions, too few for statistics",
            kld.count
        );
        return Ok(());
    }
    kld_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    p_diff_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("====== Perplexity statistics ======");
    let log_ppl = mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
    let ppl_val = log_ppl.0.exp();
    let ppl_unc = ppl_val * log_ppl.1;
    println!("Mean PPL(Q)                   : {ppl_val:10.6} ± {ppl_unc:10.6}");

    let log_ppl_base = mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
    let ppl_base_val = log_ppl_base.0.exp();
    let ppl_base_unc = ppl_base_val * log_ppl_base.1;
    println!("Mean PPL(base)                : {ppl_base_val:10.6} ± {ppl_base_unc:10.6}");

    let log_ppl_cov = covariance(
        kld.sum_nll,
        kld.sum_nll_base,
        kld.sum_nll_nll_base,
        kld.count,
    );
    let log_ppl_cor = log_ppl_cov / (log_ppl.1 * log_ppl_base.1);
    println!(
        "Cor(ln(PPL(Q)), ln(PPL(base))): {:6.2}%",
        100.0 * log_ppl_cor
    );

    let ratio_val = log_ppl.0 - log_ppl_base.0;
    let ratio_unc =
        (log_ppl.1 * log_ppl.1 + log_ppl_base.1 * log_ppl_base.1 - 2.0 * log_ppl_cov).sqrt();
    println!("Mean ln(PPL(Q)/PPL(base))     : {ratio_val:10.6} ± {ratio_unc:10.6}");
    let ratio = ratio_val.exp();
    println!(
        "Mean PPL(Q)/PPL(base)         : {ratio:10.6} ± {:10.6}",
        ratio * ratio_unc
    );
    let ppl_cov = ppl_val * ppl_base_val * log_ppl_cov;
    let diff = ppl_val - ppl_base_val;
    let diff_unc = (ppl_unc * ppl_unc + ppl_base_unc * ppl_base_unc - 2.0 * ppl_cov).sqrt();
    println!("Mean PPL(Q)-PPL(base)         : {diff:10.6} ± {diff_unc:10.6}");

    println!();
    println!("====== KL divergence statistics ======");
    let kl = mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
    println!("Mean    KLD: {:10.6} ± {:10.6}", kl.0, kl.1);
    let median = if kld_values.len().is_multiple_of(2) {
        0.5 * (kld_values[kld_values.len() / 2] + kld_values[kld_values.len() / 2 - 1])
    } else {
        kld_values[kld_values.len() / 2]
    };
    println!("Maximum KLD: {:10.6}", kld_values[kld_values.len() - 1]);
    println!("99.9%   KLD: {:10.6}", percentile(&kld_values, 0.999));
    println!("99.0%   KLD: {:10.6}", percentile(&kld_values, 0.990));
    println!("95.0%   KLD: {:10.6}", percentile(&kld_values, 0.950));
    println!("90.0%   KLD: {:10.6}", percentile(&kld_values, 0.900));
    println!("Median  KLD: {median:10.6}");
    println!("10.0%   KLD: {:10.6}", percentile(&kld_values, 0.100));
    println!(" 5.0%   KLD: {:10.6}", percentile(&kld_values, 0.050));
    println!(" 1.0%   KLD: {:10.6}", percentile(&kld_values, 0.010));
    println!(" 0.1%   KLD: {:10.6}", percentile(&kld_values, 0.001));
    println!("Minimum KLD: {:10.6}", kld_values[0]);

    println!();
    println!("====== Token probability statistics ======");
    let p_diff = mean_and_uncertainty(kld.sum_p_diff, kld.sum_p_diff2, kld.count);
    println!(
        "Mean    Δp: {:6.3} ± {:5.3} %",
        100.0 * p_diff.0,
        100.0 * p_diff.1
    );
    let p_diff_median = if p_diff_values.len().is_multiple_of(2) {
        0.5 * (p_diff_values[p_diff_values.len() / 2] + p_diff_values[p_diff_values.len() / 2 - 1])
    } else {
        p_diff_values[p_diff_values.len() / 2]
    };
    println!(
        "Maximum Δp: {:6.3}%",
        p_diff_values[p_diff_values.len() - 1]
    );
    println!("99.9%   Δp: {:6.3}%", percentile(&p_diff_values, 0.999));
    println!("99.0%   Δp: {:6.3}%", percentile(&p_diff_values, 0.990));
    println!("95.0%   Δp: {:6.3}%", percentile(&p_diff_values, 0.950));
    println!("90.0%   Δp: {:6.3}%", percentile(&p_diff_values, 0.900));
    println!("75.0%   Δp: {:6.3}%", percentile(&p_diff_values, 0.750));
    println!("Median  Δp: {p_diff_median:6.3}%");
    println!("25.0%   Δp: {:6.3}%", percentile(&p_diff_values, 0.250));
    println!("10.0%   Δp: {:6.3}%", percentile(&p_diff_values, 0.100));
    println!(" 5.0%   Δp: {:6.3}%", percentile(&p_diff_values, 0.050));
    println!(" 1.0%   Δp: {:6.3}%", percentile(&p_diff_values, 0.010));
    println!(" 0.1%   Δp: {:6.3}%", percentile(&p_diff_values, 0.001));
    println!("Minimum Δp: {:6.3}%", p_diff_values[0]);

    let p_diff_mse = mean_and_uncertainty(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);
    let rms = p_diff_mse.0.sqrt();
    let rms_unc = 0.5 / rms * p_diff_mse.1;
    println!(
        "RMS Δp    : {:6.3} ± {:5.3} %",
        100.0 * rms,
        100.0 * rms_unc
    );

    let same_top_p = 1.0 * kld.n_same_top as f64 / kld.count as f64;
    println!(
        "Same top p: {:6.3} ± {:5.3} %",
        100.0 * same_top_p,
        100.0 * (same_top_p * (1.0 - same_top_p) / (kld.count - 1) as f64).sqrt()
    );

    Ok(())
}
