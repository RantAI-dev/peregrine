//! Sampling: argmax, or temperature + nucleus with position-keyed randomness.
//!
//! # Why the randomness is keyed by position (uzu)
//!
//! Speculative decoding verifies a drafted token by checking that it equals the token
//! the model would have produced at that position anyway. Under argmax that token is
//! a function of the logits alone, so verification is exact. Under sampling it used to
//! depend on a generator's *state* — how many draws came before — so a verification
//! pass that sampled several rows at once could not reproduce what one-at-a-time
//! decoding would have drawn, and the engine switched speculation off for every
//! non-greedy chat.
//!
//! uzu derives the noise from the position instead (`encodable_block/sampling/
//! prng.rs`, `PRng::derive(position)`): the token at position `p` becomes a pure
//! function of `(logits_p, key_p)`. A drafted token is then accepted exactly when
//! sequential sampling would have emitted it, and speculation is lossless at any
//! temperature — not merely "the same distribution" but the same sequence.
//!
//! # The draw
//!
//! Exponential race, the Gumbel-max trick's one-logarithm form: with
//! `E_i = −ln u_i` independent Exp(1) draws, `argmax_i p_i / E_i` is distributed
//! exactly as `p_i / Σp` over the candidates. `u_i` is a hash of `(key, i)`, so
//! nothing is stored and a row can be redrawn identically. Only the nucleus is
//! raced; the rest of the vocabulary is excluded before the draw.
//!
//! # The nucleus in O(V)
//!
//! The nucleus is the smallest set of most-probable tokens whose mass reaches
//! `top_p`. Sorting the vocabulary to find it (151 936 entries on Qwen3) cost
//! milliseconds per sampled row, and verifying a 12-token draft samples 13 rows. A
//! positive `f32`'s bit pattern orders like its value, so a histogram of the top 12
//! bits finds the bin where the cumulative mass crosses `top_p`, a second histogram
//! on the next 11 bits narrows it, and only that last bin is sorted. Ties inside the
//! boundary are broken by token id, so the set — and so the draw — is deterministic.

/// The key for the token produced at absolute position `pos` of turn `turn`.
pub fn position_key(seed: u64, turn: u64, pos: usize) -> u64 {
    splitmix64(splitmix64(seed ^ splitmix64(turn)).wrapping_add(pos as u64))
}

#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform in (0, 1), never 0 (so `ln` is finite), from `(key, i)`.
#[inline]
fn uniform(key: u64, i: u32) -> f64 {
    let h = splitmix64(key ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    ((h >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// llama.cpp-compatible repetition penalties.
///
/// They act on the raw logits before temperature, top-k, top-p and min-p, and before
/// the greedy argmax. Every default is the no-op value, so a request that sets none
/// takes the same code path and produces the same bits it did before they existed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Penalties {
    /// `presence_penalty`: subtracted once from every token that occurs in the window.
    pub presence: f32,
    /// `frequency_penalty`: subtracted once per occurrence in the window.
    pub frequency: f32,
    /// `repeat_penalty`: divides positive logits and multiplies non-positive ones. 1 = off.
    pub repeat: f32,
    /// `repeat_last_n`: tokens in the window. 0 = off, -1 = the whole sequence.
    pub last_n: i32,
}

impl Default for Penalties {
    fn default() -> Self {
        Penalties {
            presence: 0.0,
            frequency: 0.0,
            repeat: 1.0,
            last_n: 64,
        }
    }
}

impl Penalties {
    /// Whether these penalties can change a logit. When they cannot, the sampler runs
    /// the pre-penalty path exactly, without copying the row or scanning a window.
    pub fn is_off(&self) -> bool {
        self.last_n == 0 || (self.presence == 0.0 && self.frequency == 0.0 && self.repeat == 1.0)
    }
}

/// Apply `pen` to `logits` for the `(token, count)` pairs of the window.
///
/// llama.cpp's order: scale by `repeat_penalty` first, then subtract the frequency and
/// presence penalties. The same order matters when a token is both repeated and subject
/// to a presence penalty, so this is transcribed rather than rearranged.
pub fn apply_penalties(logits: &mut [f32], pen: &Penalties, window: &[(u32, u32)]) {
    for &(token, count) in window {
        let Some(logit) = logits.get_mut(token as usize) else {
            // A token id outside the row cannot come from this model's tokenizer; skip
            // rather than panic if a caller passes a window from elsewhere.
            continue;
        };
        if pen.repeat != 1.0 {
            *logit = if *logit > 0.0 {
                *logit / pen.repeat
            } else {
                *logit * pen.repeat
            };
        }
        *logit -= count as f32 * pen.frequency + if count > 0 { pen.presence } else { 0.0 };
    }
}

/// A slot's rolling token history and the counts of its penalty window, reused every token.
///
/// The window is rebuilt from the history when a token is sampled: O(window), never
/// O(vocabulary). The dense count buffer, the distinct-token list and the output pairs
/// are kept between calls, so after the first penalised token sampling allocates nothing.
#[derive(Default)]
pub struct PenaltyState {
    history: Vec<u32>,
    counts: Vec<u32>,
    tokens: Vec<u32>,
    window: Vec<(u32, u32)>,
}

impl PenaltyState {
    /// Begin a sequence: the prompt tokens form the first window.
    ///
    /// `max_new` reserves room for the tokens the request may generate, so the history
    /// does not grow (and reallocate) in the decode loop.
    pub fn start(&mut self, prompt: &[u32], max_new: usize) {
        self.history.clear();
        self.history.reserve(prompt.len().saturating_add(max_new));
        self.history.extend_from_slice(prompt);
        self.window.clear();
        // `counts` is cleared lazily by `prepare`, through `tokens`: keep that list so the counts it
        // names are zeroed on the next window. Nothing here allocates.
    }

    /// Add a token the sequence has just taken.
    pub fn push(&mut self, token: u32) {
        self.history.push(token);
    }

    /// Rebuild the `(token, count)` pairs of the last `last_n` tokens (-1: all) and
    /// return them. `n_vocab` bounds the reused count buffer.
    pub fn prepare(&mut self, last_n: i32, n_vocab: usize) -> &[(u32, u32)] {
        self.window.clear();
        if last_n == 0 {
            return &self.window;
        }
        if self.counts.len() < n_vocab {
            self.counts.resize(n_vocab, 0);
        }
        // Clear only the tokens the previous window touched: O(distinct), not O(vocab).
        for &token in &self.tokens {
            self.counts[token as usize] = 0;
        }
        self.tokens.clear();
        let from = if last_n < 0 {
            0
        } else {
            self.history.len().saturating_sub(last_n as usize)
        };
        for &token in &self.history[from..] {
            let count = &mut self.counts[token as usize];
            if *count == 0 {
                self.tokens.push(token);
            }
            *count += 1;
        }
        self.window.reserve(self.tokens.len());
        for &token in &self.tokens {
            self.window.push((token, self.counts[token as usize]));
        }
        &self.window
    }

    /// The window most recently built by [`prepare`](Self::prepare).
    pub fn window(&self) -> &[(u32, u32)] {
        &self.window
    }
}

/// Reusable scratch, so sampling a row allocates nothing after the first.
#[derive(Default)]
pub struct Scratch {
    p: Vec<f32>,
    kept: Vec<u32>,
    bin: Vec<(u32, f32)>,
    mass1: Vec<f64>,
    mass2: Vec<f64>,
}

/// The index of the largest logit (the first one on a tie).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

/// Argmax when `temperature <= 0`; otherwise the token drawn for `key` from the
/// temperature-scaled distribution restricted to its `top_p` nucleus.
pub fn sample(logits: &[f32], temperature: f32, top_p: f32, key: u64, sc: &mut Scratch) -> u32 {
    sample_filtered(logits, temperature, top_p, 0, 0.0, key, sc)
}

/// [`sample`] with two extra cuts, in llama.cpp's order: `top_k` (0 = off)
/// keeps the k most probable tokens, the nucleus is then taken over the renormalised
/// survivors, and `min_p` (0 = off) drops tokens below `min_p` times the best
/// probability at temperature 1 (llama.cpp applies temperature last). With both off
/// this is the histogram fast path; with either on the candidates are sorted, which
/// costs milliseconds per row on a 152k vocabulary and is why they default off.
pub fn sample_filtered(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    key: u64,
    sc: &mut Scratch,
) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let m = crate::vmath::max(logits);
    sc.p.clear();
    sc.p.extend_from_slice(logits);
    let z = crate::vmath::exp_scaled_sum(&mut sc.p, m, 1.0 / temperature);
    if top_k == 0 && min_p <= 0.0 {
        nucleus(
            &sc.p,
            z,
            top_p,
            &mut sc.kept,
            &mut sc.bin,
            &mut sc.mass1,
            &mut sc.mass2,
        );
    } else {
        cut_nucleus(&sc.p, top_p, top_k, min_p, temperature, &mut sc.kept);
    }
    // A row with no finite probability (all −∞ or NaN logits) has an empty nucleus:
    // fall back to the argmax rather than index an empty list.
    if sc.kept.is_empty() {
        return argmax(logits);
    }
    race(sc, key)
}

/// [`sample_filtered`] with llama.cpp-compatible repetition penalties applied to the raw
/// logits first. `window` holds the `(token, count)` pairs of the slot's last tokens;
/// when `pen` is off this is [`sample_filtered`] exactly, with no copy and no scan.
// Each argument is a separate sampling control; bundling them would hide which are per request.
#[allow(clippy::too_many_arguments)]
pub fn sample_penalized(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    pen: &Penalties,
    window: &[(u32, u32)],
    key: u64,
    sc: &mut Scratch,
) -> u32 {
    if pen.is_off() || window.is_empty() {
        return sample_filtered(logits, temperature, top_p, top_k, min_p, key, sc);
    }
    // The model's logits row must not be edited: other rows of the batch, the caller and
    // (when asked) the log-probability all read it. Work on the scratch copy instead.
    sc.p.clear();
    sc.p.extend_from_slice(logits);
    apply_penalties(&mut sc.p, pen, window);
    if temperature <= 0.0 {
        return argmax(&sc.p);
    }
    // The penalised row is the row the sampler saw, so the empty-nucleus fallback must
    // use it too.
    let fallback = argmax(&sc.p);
    let m = crate::vmath::max(&sc.p);
    let z = crate::vmath::exp_scaled_sum(&mut sc.p, m, 1.0 / temperature);
    if top_k == 0 && min_p <= 0.0 {
        nucleus(
            &sc.p,
            z,
            top_p,
            &mut sc.kept,
            &mut sc.bin,
            &mut sc.mass1,
            &mut sc.mass2,
        );
    } else {
        cut_nucleus(&sc.p, top_p, top_k, min_p, temperature, &mut sc.kept);
    }
    if sc.kept.is_empty() {
        return fallback;
    }
    race(sc, key)
}

/// The race over the nucleus: argmax p_i / E_i, compared as ln p_i − ln E_i.
fn race(sc: &mut Scratch, key: u64) -> u32 {
    let mut best = sc.kept[0];
    let mut bv = f64::NEG_INFINITY;
    for &i in &sc.kept {
        let e = -uniform(key, i).ln();
        let v = (sc.p[i as usize] as f64).ln() - e.ln();
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// The sorted path for `top_k` / `min_p`: candidates most probable first (ties by id),
/// cut to `top_k`, the `top_p` nucleus of the survivors, then the `min_p` floor.
fn cut_nucleus(p: &[f32], top_p: f32, top_k: usize, min_p: f32, t: f32, kept: &mut Vec<u32>) {
    let mut c: Vec<(u32, f32)> = p
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite() && **v > 0.0)
        .map(|(i, v)| (i as u32, *v))
        .collect();
    let desc = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if top_k > 0 && top_k < c.len() {
        c.select_nth_unstable_by(top_k - 1, desc);
        c.truncate(top_k);
    }
    c.sort_unstable_by(desc);
    kept.clear();
    let Some(best) = c.first().map(|e| e.1) else {
        return;
    };
    let z: f64 = c.iter().map(|e| e.1 as f64).sum();
    let floor = if min_p > 0.0 {
        min_p.powf(1.0 / t) * best
    } else {
        0.0
    };
    let mut acc = 0.0f64;
    for &(i, v) in &c {
        if (acc >= top_p as f64 * z || v < floor) && !kept.is_empty() {
            break;
        }
        kept.push(i);
        acc += v as f64;
    }
}

/// The `top_p` nucleus of the unnormalised probabilities `p` (sum `z`) into `kept`.
fn nucleus(
    p: &[f32],
    z: f64,
    top_p: f32,
    kept: &mut Vec<u32>,
    bin: &mut Vec<(u32, f32)>,
    mass1: &mut Vec<f64>,
    mass2: &mut Vec<f64>,
) {
    kept.clear();
    if top_p >= 1.0 {
        kept.extend((0..p.len() as u32).filter(|&i| p[i as usize] > 0.0));
        return;
    }
    let target = (top_p as f64).max(0.0) * z;
    // Level 1: bits 31..20 (sign, exponent, 3 mantissa bits) of a value in [0, 1].
    const B1: usize = 1 << 12;
    mass1.clear();
    mass1.resize(B1, 0.0);
    for &v in p {
        mass1[(v.to_bits() >> 20) as usize] += v as f64;
    }
    let (b1, above1) = crossing(mass1, target);
    // Level 2: bits 19..9 within the boundary bin.
    const B2: usize = 1 << 11;
    mass2.clear();
    mass2.resize(B2, 0.0);
    for &v in p {
        let b = v.to_bits();
        if (b >> 20) as usize == b1 {
            mass2[((b >> 9) & 0x7FF) as usize] += v as f64;
        }
    }
    let (b2, above2) = crossing(mass2, target - above1);
    let mut acc = above1 + above2;
    bin.clear();
    for (i, &v) in p.iter().enumerate() {
        let b = v.to_bits();
        let (h1, h2) = ((b >> 20) as usize, ((b >> 9) & 0x7FF) as usize);
        if h1 > b1 || (h1 == b1 && h2 > b2) {
            kept.push(i as u32);
        } else if h1 == b1 && h2 == b2 {
            bin.push((i as u32, v));
        }
    }
    // The boundary bin: most probable first, then lowest id, until the mass is in.
    bin.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    for &(i, v) in bin.iter() {
        if acc >= target && !kept.is_empty() {
            break;
        }
        kept.push(i);
        acc += v as f64;
    }
}

/// Walking down from the top bin: the bin where cumulative mass reaches `target`,
/// and the mass strictly above it.
fn crossing(mass: &[f64], target: f64) -> (usize, f64) {
    let mut above = 0.0f64;
    for b in (0..mass.len()).rev() {
        if above + mass[b] >= target && mass[b] > 0.0 {
            return (b, above);
        }
        above += mass[b];
    }
    (0, above)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(seed: u64, n: usize, spread: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * spread
            })
            .collect()
    }

    /// The reference nucleus: sort everything, take the smallest top prefix.
    fn sorted_nucleus(p: &[f32], top_p: f32) -> Vec<u32> {
        let z: f64 = p.iter().map(|v| *v as f64).sum();
        let mut idx: Vec<u32> = (0..p.len() as u32).collect();
        idx.sort_by(|&a, &b| p[b as usize].total_cmp(&p[a as usize]).then(a.cmp(&b)));
        let mut acc = 0.0f64;
        let mut out = Vec::new();
        for i in idx {
            if acc >= top_p as f64 * z && !out.is_empty() {
                break;
            }
            out.push(i);
            acc += p[i as usize] as f64;
        }
        out.sort();
        out
    }

    #[test]
    fn the_histogram_nucleus_is_the_sorted_nucleus() {
        for (seed, spread, top_p) in [
            (1, 30.0, 0.9),
            (2, 4.0, 0.95),
            (3, 0.5, 0.5),
            (4, 60.0, 0.99),
            (5, 0.01, 0.9),
        ] {
            let l = logits(seed, 151_936, spread);
            let m = crate::vmath::max(&l);
            let mut p = l.clone();
            let z = crate::vmath::exp_scaled_sum(&mut p, m, 1.0 / 0.8);
            let (mut kept, mut bin, mut m1, mut m2) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            nucleus(&p, z, top_p, &mut kept, &mut bin, &mut m1, &mut m2);
            kept.sort();
            assert_eq!(
                kept,
                sorted_nucleus(&p, top_p),
                "seed {seed} spread {spread} top_p {top_p}"
            );
        }
    }

    #[test]
    fn top_k_and_min_p_cut_the_tail() {
        // Probabilities proportional to e^3, e^2, e^1, e^0 at temperature 1.
        let l = [3.0f32, 2.0, 1.0, 0.0];
        let mut sc = Scratch::default();
        let mut draw = |k: usize, mp: f32, tp: f32, s: u64| {
            sample_filtered(&l, 1.0, tp, k, mp, position_key(s, 1, 0), &mut sc)
        };
        assert!((1..200u64).all(|s| draw(1, 0.0, 1.0, s) == 0));
        let picks: Vec<u32> = (1..400u64).map(|s| draw(2, 0.0, 1.0, s)).collect();
        assert!(picks.iter().all(|&t| t < 2) && picks.contains(&0) && picks.contains(&1));
        // e^-1 = 0.37 stays above 0.3 x best, e^-2 = 0.14 goes.
        let picks: Vec<u32> = (1..400u64).map(|s| draw(0, 0.3, 1.0, s)).collect();
        assert!(picks.iter().all(|&t| t < 2));
        let picks: Vec<u32> = (1..2000u64).map(|s| draw(0, 0.0, 1.0, s)).collect();
        assert!((0..4).all(|t| picks.contains(&t)));
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let l = logits(9, 5000, 8.0);
        let mut sc = Scratch::default();
        assert_eq!(sample(&l, 0.0, 0.9, 123, &mut sc), argmax(&l));
    }

    #[test]
    fn a_row_without_finite_probabilities_falls_back_instead_of_panicking() {
        let mut sc = Scratch::default();
        let row = vec![f32::NEG_INFINITY; 16];
        assert_eq!(sample(&row, 0.8, 1.0, 7, &mut sc), 0);
        assert_eq!(sample(&row, 0.8, 0.9, 7, &mut sc), 0);
    }

    #[test]
    fn a_key_determines_the_token() {
        let l = logits(10, 32_000, 6.0);
        let (mut a, mut b) = (Scratch::default(), Scratch::default());
        for pos in 0..50 {
            let k = position_key(42, 3, pos);
            assert_eq!(
                sample(&l, 0.9, 0.95, k, &mut a),
                sample(&l, 0.9, 0.95, k, &mut b)
            );
        }
        let distinct: std::collections::HashSet<u32> = (0..200)
            .map(|pos| sample(&l, 1.5, 1.0, position_key(42, 3, pos), &mut a))
            .collect();
        assert!(
            distinct.len() > 20,
            "different positions must draw differently"
        );
    }

    /// The race reproduces the distribution: 40 000 draws from a known 6-way
    /// distribution land within a chi-square bound of the expected counts.
    #[test]
    fn the_race_samples_the_specified_distribution() {
        let l = [2.0f32, 1.0, 0.5, 0.0, -1.0, -3.0];
        let t = 0.7f32;
        let w: Vec<f64> = l
            .iter()
            .map(|x| ((x - 2.0) / t) as f64)
            .map(f64::exp)
            .collect();
        let z: f64 = w.iter().sum();
        let n = 40_000;
        let mut counts = [0usize; 6];
        let mut sc = Scratch::default();
        for pos in 0..n {
            counts[sample(&l, t, 1.0, position_key(7, 0, pos), &mut sc) as usize] += 1;
        }
        let chi2: f64 = counts
            .iter()
            .zip(&w)
            .map(|(&c, &wi)| {
                let e = n as f64 * wi / z;
                (c as f64 - e).powi(2) / e
            })
            .sum();
        // 5 degrees of freedom: p = 0.001 at 20.5.
        assert!(chi2 < 20.5, "chi² = {chi2:.1}, counts {counts:?}");
    }

    #[test]
    fn nothing_outside_the_nucleus_is_ever_drawn() {
        // Token 0 holds ~88% of the mass at T = 1; top_p = 0.8 keeps it alone.
        let l = [4.0f32, 1.0, 1.0, 0.5, 0.0];
        let mut sc = Scratch::default();
        for pos in 0..2000 {
            assert_eq!(sample(&l, 1.0, 0.8, position_key(1, 0, pos), &mut sc), 0);
        }
    }

    #[test]
    fn penalties_follow_llama_cpps_arithmetic() {
        // Token 0 occurs twice, token 2 once, token 3 once. With repeat 2, presence 0.5
        // and frequency 0.25:
        //   t0: 2/2 - (2*0.25 + 0.5) = 0
        //   t2: -1*2 - (0.25 + 0.5)  = -2.75
        //   t3:  0*2 - (0.25 + 0.5)  = -0.75
        // Token 1 is outside the window and keeps its value.
        let pen = Penalties {
            presence: 0.5,
            frequency: 0.25,
            repeat: 2.0,
            last_n: 64,
        };
        let mut l = vec![2.0f32, 0.5, -1.0, 0.0];
        apply_penalties(&mut l, &pen, &[(0, 2), (2, 1), (3, 1)]);
        assert_eq!(l, vec![0.0, 0.5, -2.75, -0.75]);
    }

    #[test]
    fn default_penalties_take_the_exact_pre_penalty_path() {
        let pen = Penalties::default();
        assert!(pen.is_off());
        let l = logits(11, 4096, 7.0);
        let window = [(0u32, 3u32), (5, 1)];
        let mut a = Scratch::default();
        let mut b = Scratch::default();
        for pos in 0..32 {
            let k = position_key(5, 1, pos);
            assert_eq!(
                sample_penalized(&l, 0.8, 0.95, 40, 0.05, &pen, &window, k, &mut a),
                sample_filtered(&l, 0.8, 0.95, 40, 0.05, k, &mut b),
            );
        }
        // Greedy too: the defaults must leave the argmax alone.
        assert_eq!(
            sample_penalized(&l, 0.0, 0.95, 40, 0.05, &pen, &window, 1, &mut a),
            argmax(&l)
        );
    }

    #[test]
    fn the_window_is_the_last_repeat_last_n_tokens() {
        fn window(state: &mut PenaltyState, last_n: i32) -> Vec<(u32, u32)> {
            state.prepare(last_n, 16).to_vec()
        }
        let mut state = PenaltyState::default();
        state.start(&[1, 2, 3, 4, 5], 0);
        // 0 = off.
        assert!(window(&mut state, 0).is_empty());
        // The last n tokens.
        assert_eq!(window(&mut state, 2), vec![(4, 1), (5, 1)]);
        // -1 = the whole sequence.
        assert_eq!(
            window(&mut state, -1),
            vec![(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
        );
        // Counts are rebuilt from scratch: duplicates in the window are counted, tokens
        // that leave it are forgotten, and a later window does not inherit old counts.
        state.start(&[7, 7, 8, 7], 0);
        assert_eq!(window(&mut state, 3), vec![(7, 2), (8, 1)]);
        assert_eq!(window(&mut state, 1), vec![(7, 1)]);
        assert_eq!(window(&mut state, -1), vec![(7, 3), (8, 1)]);
        // A longer sequence keeps only its tail.
        state.start(&[1, 2, 3, 4, 5, 6], 0);
        assert_eq!(
            window(&mut state, 10),
            vec![(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
        );
    }

    #[test]
    fn greedy_respects_penalties() {
        let l = [3.0f32, 2.0, 0.0];
        let pen = Penalties {
            presence: 5.0,
            frequency: 0.0,
            repeat: 1.0,
            last_n: 64,
        };
        let mut sc = Scratch::default();
        // Token 0 is the raw argmax but its single occurrence costs it 5: 3 - 5 < 2.
        assert_eq!(
            sample_penalized(&l, 0.0, 1.0, 0, 0.0, &pen, &[(0, 1)], 9, &mut sc),
            1
        );
        // With the defaults the same row is still greedy on token 0.
        assert_eq!(
            sample_penalized(
                &l,
                0.0,
                1.0,
                0,
                0.0,
                &Penalties::default(),
                &[(0, 1)],
                9,
                &mut sc
            ),
            0
        );
    }
}
