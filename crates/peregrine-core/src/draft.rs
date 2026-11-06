//! Drafting: what a draft costs, what verifying it costs, and how many of its tokens
//! are worth verifying.
//!
//! This module is the home of the verify-cost model ([`VerifyCost`]) and of the rule
//! that decides how much of a draft to check ([`choose_k`]). It measures the engine's
//! *own* forward passes, so it belongs with the runtime that runs them — ADR 0007
//! gives it to the one runtime (`serve::Engine`) and puts the `Drafter` interface
//! (MTP, DSpark, suffix speculation) here as well.
//!
//! # The model
//!
//! A verification pass of width `w` costs one plain step plus `w − 1` extra rows.
//! How much an extra row costs is measured, not assumed ([`ROW_COST`]), because on
//! this machine verification is compute-bound rather than bandwidth-bound: a width-8
//! pass costs about four steps, not one. The measured widths are kept as they are
//! seen (an EMA per width) and every width that was never measured is priced from
//! the slope the measured ones show.
//!
//! [`choose_k`] then asks the only question that matters: how many drafted tokens
//! to verify, given how likely each is to survive and what the verification costs.
//! A doubtful draft is not verified at all.

/// What one extra row costs a verification pass, as a fraction of a plain step.
/// Measured with `examples/vcurve` (width 1..16, median of 9): SmolLM2-360M 0.47,
/// Llama-3.2-1B 0.40, Qwen2.5-0.5B 0.42. Verification on this CPU is compute-bound,
/// not bandwidth-bound — a width-8 pass costs about 4 steps, not 1 — which is why
/// the ceiling on any lossless speculation here is near 1/0.42 ≈ 2.4×. The engine
/// priced a row at 0.12 before this was measured, and drafted far too short.
pub const ROW_COST: f64 = 0.42;

/// Plain steps to take after drafting is measured to lose, before probing again.
pub const PARK_STEPS: u32 = 12;

/// How often a drafting round is probed even when the price says not to draft.
///
/// A confidence that is never tested never recovers, so every `PROBE_EVERY`th step
/// verifies a couple of tokens anyway.
pub const PROBE_EVERY: u64 = 16;

/// Seconds a verification forward of width `w` takes, learned online. A width seen
/// twice is its own measurement (an EMA; the first reading of a width is a cold one
/// and the second can only lower it). Any other width is the plain-step time scaled
/// by `1 + β·(w−1)`, with β the mean slope of the widths measured so far, or
/// `ROW_COST` before any. Extrapolating a line through two or three noisy points
/// out to width 12 — what this did before — priced a 12-token draft at 85 ms
/// against a true 77 ms-or-less and refused every long draft.
#[derive(Default)]
pub struct VerifyCost {
    /// The measured seconds per width, once a width has been seen.
    pub(crate) ema: [Option<f64>; 17],
    /// How many times each width has been measured.
    pub(crate) seen: [u8; 17],
}

impl VerifyCost {
    /// Record one measurement of a verification pass of width `w`.
    pub fn observe(&mut self, w: usize, secs: f64) {
        if w == 0 || w >= self.ema.len() {
            return;
        }
        self.seen[w] = self.seen[w].saturating_add(1);
        self.ema[w] = Some(match (self.ema[w], self.seen[w]) {
            (Some(e), 2) => e.min(secs),
            (Some(e), _) => 0.8 * e + 0.2 * secs,
            (None, _) => secs,
        });
    }
    /// Mean extra-row cost, in plain steps, over the widths measured at least twice.
    pub fn slope(&self) -> f64 {
        let Some(t1) = self.ema[1].filter(|_| self.seen[1] >= 2) else {
            return ROW_COST;
        };
        let s: Vec<f64> = (2..self.ema.len())
            .filter(|&w| self.seen[w] >= 2)
            .filter_map(|w| {
                self.ema[w].map(|t| ((t / t1 - 1.0) / (w as f64 - 1.0)).clamp(0.15, 0.9))
            })
            .collect();
        if s.is_empty() {
            ROW_COST
        } else {
            s.iter().sum::<f64>() / s.len() as f64
        }
    }
    /// The seconds a verification pass of width `w` is expected to take, or `None`
    /// while not even a plain step has been timed.
    pub fn estimate(&self, w: usize) -> Option<f64> {
        if let (Some(t), true) = (
            self.ema.get(w).copied().flatten(),
            self.seen.get(w).is_some_and(|&n| n >= 2),
        ) {
            return Some(t);
        }
        let t1 = self.ema[1]?;
        Some(t1 * (1.0 + self.slope() * (w as f64 - 1.0)))
    }
    /// `estimate`, or the prior curve while not even a plain step has been timed.
    pub fn cost(&self, w: usize) -> f64 {
        self.estimate(w)
            .unwrap_or(1.0 + ROW_COST * (w as f64 - 1.0))
    }
}

/// How many of `conf.len()` drafted tokens to verify: the prefix length `k` that
/// maximises expected committed tokens per second of verification, where
/// `conf[j]` is the head's probability that draft `j` survives given its prefix did.
/// The prefix survives with the running product, so expected committed tokens are
/// `1 + Σ_{j<k} Π_{i≤j} conf[i]` (the `1` is the target's own token). DeepInfer's
/// batch version of the same rule is `mtp_varlen.hpp`; at batch 1 it is this.
pub fn choose_k(conf: &[f32], cost: impl Fn(usize) -> f64) -> usize {
    let (mut best_k, mut best) = (0usize, 1.0 / cost(1));
    let (mut surv, mut expect) = (1.0f64, 1.0f64);
    for (j, &c) in conf.iter().enumerate() {
        surv *= c as f64;
        expect += surv;
        let rate = expect / cost(j + 2);
        if rate > best {
            best = rate;
            best_k = j + 1;
        }
    }
    best_k
}

#[cfg(test)]
mod tests {

    #[test]
    fn choose_k_prices_expected_tokens_against_verify_cost() {
        let flat = |_w: usize| 1.0;
        // Free verification: verify everything with any chance of surviving.
        assert_eq!(super::choose_k(&[0.9, 0.8, 0.7], flat), 3);
        // Each extra row costs 30% of a step: only drafts likely to survive pay.
        let steep = |w: usize| 1.0 + 0.3 * (w as f64 - 1.0);
        assert_eq!(super::choose_k(&[0.95, 0.9, 0.2], steep), 2);
        assert_eq!(super::choose_k(&[0.1, 0.9, 0.9], steep), 0);
        assert_eq!(super::choose_k(&[], flat), 0);
    }

    #[test]
    fn verify_cost_scales_the_plain_step_by_the_measured_row_cost() {
        let mut v = super::VerifyCost::default();
        assert_eq!(v.estimate(2), None);
        assert!(
            (v.cost(5) - (1.0 + 4.0 * super::ROW_COST)).abs() < 1e-12,
            "the prior before any measurement"
        );
        v.observe(1, 10.0); // a cold first step
        v.observe(1, 4.0);
        v.observe(1, 4.0);
        // Only the plain step is known: prior slope.
        assert!((v.estimate(3).unwrap() - 4.0 * (1.0 + 2.0 * super::ROW_COST)).abs() < 1e-9);
        // Width 2 measured twice: a cold 9.0 then 6.0 → min → 6.0, which is its own value
        // and moves the slope (6/4 − 1 = 0.5) used for unmeasured widths.
        v.observe(2, 9.0);
        v.observe(2, 6.0);
        assert!((v.estimate(2).unwrap() - 6.0).abs() < 1e-9);
        assert!((v.estimate(5).unwrap() - 4.0 * (1.0 + 4.0 * 0.5)).abs() < 1e-9);
        // A width seen once is not trusted: it is still the model.
        v.observe(8, 100.0);
        assert!((v.estimate(8).unwrap() - 4.0 * (1.0 + 7.0 * 0.5)).abs() < 1e-9);
    }
}
