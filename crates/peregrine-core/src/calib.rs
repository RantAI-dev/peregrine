//! Startup calibration of the decode thread count and the spin window (S2.6 PR 3).
//!
//! Two independent measurements:
//!
//! 1. **Decode thread count.** Every candidate count is measured: the powers of two up to
//!    the physical cores, plus `n_phys` and its halves, plus the floor `ceil(n_phys / 2)`.
//!    The sweep never stops early — a candidate that looks worse than the one before it says
//!    nothing about the ones after it — and the rounds are interleaved, so a machine that
//!    drifts drifts for every candidate alike.
//!    The default (`n_phys`) stands unless a smaller count is *at least as good*: its median
//!    bandwidth reaches [`AT_LEAST_AS_GOOD`] of the default's, and both measurements are
//!    quiet (CV ≤ [`CV_CUTOFF`]). The smallest such count is chosen, never below half the
//!    physical cores: below that the decode rate has collapsed before, and no measurement
//!    is worth that risk.
//! 2. **Spin window.** 50 back-to-back fork-join cycles with 0 µs spin give the wake-up
//!    latency `L_wake`; `spin_us = 2 * L_wake`, clamped to [50, 2000] µs.
//!
//! A measurement whose coefficient of variation exceeds 15% (a noisy host) is not evidence:
//! the default count stands. The sweep has a hard budget of 250 ms; overrunning it keeps the
//! default count too, and the spin window is still measured. A fork-join cycle that does not
//! come back within 2 s ends the calibration: startup must not wait for a machine that cannot
//! do it.
//!
//! The calibration is off by default: `PEREGRINE_CALIBRATE=1` turns it on (see
//! `docs/design/s2.6.md`, decision 6).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Above this coefficient of variation a measurement is too noisy to be evidence.
pub const CV_CUTOFF: f64 = 0.15;

/// The share of the default count's median bandwidth a smaller count must reach to be
/// *at least as good* as it.
pub const AT_LEAST_AS_GOOD: f64 = 0.98;

/// The bursts measured per candidate thread count (plus one warm-up burst).
const BURSTS: usize = 3;

/// The bytes a bandwidth burst reads. Bigger than any last-level cache, so the burst reads
/// DRAM and not the cache the previous burst left warm.
const BUF_BYTES: usize = 64 * 1024 * 1024;

/// The sweep's time budget. Overrunning it keeps the default thread count.
const BUDGET: Duration = Duration::from_millis(250);

/// The fork-join cycles the wake-up latency is the median of.
const WAKE_CYCLES: usize = 50;

/// How long one fork-join cycle may take before the calibration gives up. The workers are
/// woken by a condvar, so a cycle is a few microseconds; a machine that cannot even do that
/// is not a machine to measure on, and startup must not wait for it.
const JOB_TIMEOUT: Duration = Duration::from_secs(2);

/// Source of tuning parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuningSource {
    Default = 0,
    Startup = 1,
    Profile = 2,
    Env = 3,
}

impl TuningSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            TuningSource::Default => "default",
            TuningSource::Startup => "startup_calibration",
            TuningSource::Profile => "profile",
            TuningSource::Env => "env",
        }
    }

    pub fn code(&self) -> usize {
        *self as usize
    }
}

/// The result of startup calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibratedSettings {
    pub decode_threads: usize,
    pub prefill_threads: usize,
    pub spin_us: u64,
    pub duration_ms: f64,
    pub source: TuningSource,
    pub cv: f64,
}

/// One candidate's measured bandwidth bursts.
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    /// The thread count that was measured.
    pub threads: usize,
    /// The bandwidth of each burst (bytes per second).
    pub samples: Vec<f64>,
}

/// Why the calibration chose the decode thread count it chose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeReason {
    /// No smaller count was at least as good: the default stands.
    DefaultStands,
    /// The default's own measurement was too noisy to be evidence: the default stands.
    Noisy,
    /// The sweep ran out of its time budget: the default stands.
    BudgetOverrun,
    /// A smaller count was at least as good, and this is the smallest such count.
    Smaller,
}

impl DecodeReason {
    /// The reason in words, for the startup log.
    pub fn as_str(&self) -> &'static str {
        match self {
            DecodeReason::DefaultStands => "no smaller count was at least as good",
            DecodeReason::Noisy => "the machine was too noisy to measure",
            DecodeReason::BudgetOverrun => "the time budget ran out",
            DecodeReason::Smaller => "a smaller count was at least as good",
        }
    }
}

/// What the calibration decided about the decode thread count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeDecision {
    /// The thread count to use.
    pub threads: usize,
    /// Why that count.
    pub reason: DecodeReason,
}

/// The lowest count the calibration may choose: half the physical cores, rounded up.
pub fn floor_threads(n_phys: usize) -> usize {
    n_phys.div_ceil(2).max(1)
}

/// The thread counts to measure: the powers of two up to `n_phys`, plus `n_phys` and its
/// halves, plus the floor. The floor is always among them, so "never below `n_phys / 2`" is
/// always decidable.
pub fn candidates(n_phys: usize) -> Vec<usize> {
    let mut set = std::collections::BTreeSet::new();
    let mut t = 1usize;
    while t <= n_phys {
        set.insert(t);
        t = t.saturating_mul(2);
    }
    let mut h = n_phys;
    while h > 0 {
        set.insert(h);
        h /= 2;
    }
    set.insert(floor_threads(n_phys));
    set.into_iter().filter(|&c| c >= 1).collect()
}

/// The coefficient of variation of a set of samples: their standard deviation over their mean.
pub fn cv(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt() / mean
}

/// The middle sample of a set of samples.
pub fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    let mid = samples.len() / 2;
    samples[mid]
}

/// The median bandwidth of a candidate's bursts, and the noise of its quietest bursts.
///
/// The median is taken over every burst; the highest one is dropped before the coefficient of
/// variation is computed, so a single slow burst does not make a candidate look noisy, and a
/// single fast burst does not make it look fast.
pub fn summarise(samples: &[f64]) -> Option<(f64, f64)> {
    let mut s: Vec<f64> = samples
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if s.is_empty() {
        return None;
    }
    s.sort_by(|a, b| a.total_cmp(b));
    let med = if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2])
    };
    let kept: &[f64] = if s.len() > 2 {
        &s[..s.len() - 1]
    } else {
        &s[..]
    };
    Some((med, cv(kept)))
}

/// The decode thread count to use, from the measured bursts alone.
///
/// The default (`n_phys`) stands unless a smaller count is *at least as good*: its median
/// bandwidth reaches [`AT_LEAST_AS_GOOD`] of the default's median, and both measurements
/// pass the noise cutoff ([`CV_CUTOFF`]). The smallest such count is chosen, but never below
/// [`floor_threads`]: a count that low has collapsed the decode rate before (the 9B fell from
/// 23.6 to 5.6 tok/s with one or two threads), and no measurement on a busy machine is worth
/// that risk.
///
/// The choice is a function of the samples alone, so it can be tested on the sequences that
/// broke the first version of this sweep.
pub fn choose_decode_threads(n_phys: usize, measured: &[Measured]) -> DecodeDecision {
    let default = n_phys.max(1);
    let stands = DecodeDecision {
        threads: default,
        reason: DecodeReason::DefaultStands,
    };
    let Some(base) = measured.iter().find(|m| m.threads == default) else {
        return stands;
    };
    let Some((base_med, base_cv)) = summarise(&base.samples) else {
        return stands;
    };
    if base_cv > CV_CUTOFF {
        return DecodeDecision {
            threads: default,
            reason: DecodeReason::Noisy,
        };
    }
    let bar = AT_LEAST_AS_GOOD * base_med;
    let mut best: Option<usize> = None;
    for m in measured.iter().filter(|m| m.threads < default) {
        let Some((med, c)) = summarise(&m.samples) else {
            continue;
        };
        // A noisy measurement is not evidence, either way.
        if c > CV_CUTOFF || med < bar {
            continue;
        }
        best = Some(best.map_or(m.threads, |b| b.min(m.threads)));
    }
    let Some(best) = best else {
        return stands;
    };
    // The floor is a rail, not evidence: it holds the count at half the physical cores even
    // when the measurement points lower. It has to have been measured, or the default stands.
    let chosen = best.max(floor_threads(default));
    if !measured.iter().any(|m| m.threads == chosen) {
        return stands;
    }
    DecodeDecision {
        threads: chosen,
        reason: DecodeReason::Smaller,
    }
}

/// Fallback settings on noisy host, timeout, or calibration failure.
pub fn fallback_defaults(duration_ms: f64, source: TuningSource) -> CalibratedSettings {
    let n_phys = crate::pool::physical_cores().unwrap_or(1);
    let n_logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    CalibratedSettings {
        decode_threads: n_phys,
        prefill_threads: n_logical,
        spin_us: crate::options::SPIN_DEFAULT_US,
        duration_ms,
        source,
        cv: 0.0,
    }
}

struct ProbeState {
    epoch: AtomicUsize,
    shutdown: AtomicBool,
    active_workers: AtomicUsize,
    job_kind: AtomicUsize, // 0 = noop, 1 = bandwidth
    remaining: AtomicUsize,
    wake_cvar: Condvar,
    wake_mutex: Mutex<()>,
    done_cvar: Condvar,
    done_mutex: Mutex<()>,
}

impl ProbeState {
    fn new() -> Self {
        Self {
            epoch: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            active_workers: AtomicUsize::new(0),
            job_kind: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            wake_cvar: Condvar::new(),
            wake_mutex: Mutex::new(()),
            done_cvar: Condvar::new(),
            done_mutex: Mutex::new(()),
        }
    }

    /// One fork-join cycle: `None` when the workers did not come back in time.
    fn run_job(&self, active: usize, kind: usize) -> Option<Duration> {
        self.active_workers.store(active, Ordering::Release);
        self.job_kind.store(kind, Ordering::Release);
        self.remaining.store(active, Ordering::Release);

        let t0 = Instant::now();
        {
            let _g = self.wake_mutex.lock().unwrap();
            self.epoch.fetch_add(1, Ordering::Release);
            self.wake_cvar.notify_all();
        }
        {
            let mut lock = self.done_mutex.lock().unwrap();
            let deadline = Instant::now() + JOB_TIMEOUT;
            while self.remaining.load(Ordering::Acquire) > 0 {
                let now = Instant::now();
                if now >= deadline {
                    return None;
                }
                let (l, _) = self.done_cvar.wait_timeout(lock, deadline - now).unwrap();
                lock = l;
            }
        }
        Some(t0.elapsed())
    }
}

/// Run startup calibration.
pub fn calibrate() -> CalibratedSettings {
    let t0 = Instant::now();
    let n_phys = crate::pool::physical_cores().unwrap_or(1);
    let n_logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    let max_threads = n_logical.max(n_phys).max(1);
    let cands = candidates(n_phys);

    // Allocate the buffer the bandwidth bursts read (pre-faulted)
    let buf_len = BUF_BYTES / size_of::<u64>();
    let buf = vec![0x5555_aaaa_5555_aaaau64; buf_len];

    let state = std::sync::Arc::new(ProbeState::new());

    let res = std::thread::scope(|s| {
        // Spawn max_threads workers
        for w in 0..max_threads {
            let st = state.clone();
            let b = &buf;
            s.spawn(move || {
                let mut last_epoch = 0;
                loop {
                    let mut lock = st.wake_mutex.lock().unwrap();
                    while st.epoch.load(Ordering::Acquire) == last_epoch
                        && !st.shutdown.load(Ordering::Relaxed)
                    {
                        lock = st.wake_cvar.wait(lock).unwrap();
                    }
                    if st.shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    last_epoch = st.epoch.load(Ordering::Acquire);
                    drop(lock);

                    let active = st.active_workers.load(Ordering::Relaxed);
                    if w < active {
                        if st.job_kind.load(Ordering::Relaxed) == 1 {
                            // Bandwidth read
                            let (lo, hi) = crate::pool::share(b.len(), w, active);
                            let chunk = &b[lo..hi];
                            let mut a0 = 0u64;
                            let mut a1 = 0u64;
                            let mut a2 = 0u64;
                            let mut a3 = 0u64;
                            for w in chunk.chunks_exact(4) {
                                a0 ^= w[0];
                                a1 ^= w[1];
                                a2 ^= w[2];
                                a3 ^= w[3];
                            }
                            std::hint::black_box(a0 ^ a1 ^ a2 ^ a3);
                        }
                        if st.remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
                            let _g = st.done_mutex.lock().unwrap();
                            st.done_cvar.notify_all();
                        }
                    }
                }
            });
        }

        // The measurement, in a closure of its own: whatever happens in it, the workers are
        // shut down below.
        let measure = || -> Option<(DecodeDecision, f64, f64)> {
            // Warm up all workers once so TLB and CPU frequency scaling are active
            state.run_job(max_threads, 1)?;

            // Phase 2: bandwidth, every candidate, no early exit. The rounds are interleaved, so
            // a machine that drifts (a neighbour's job, a clock step) drifts for every candidate
            // alike instead of for the ones measured late.
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); cands.len()];
            let mut timed_out = false;
            'sweep: for round in 0..=BURSTS {
                for (i, &cand) in cands.iter().enumerate() {
                    if t0.elapsed() >= BUDGET {
                        timed_out = true;
                        break 'sweep;
                    }
                    let dt = state.run_job(cand, 1)?.as_secs_f64();
                    // Round 0 is the warm-up burst: it is measured, and thrown away.
                    if round > 0 && dt > 0.0 {
                        samples[i].push(BUF_BYTES as f64 / dt);
                    }
                }
            }

            let mut measured = Vec::with_capacity(cands.len());
            let mut highest_cv = 0.0f64;
            for (i, &cand) in cands.iter().enumerate() {
                if samples[i].is_empty() {
                    continue;
                }
                if let Some((_, c)) = summarise(&samples[i]) {
                    highest_cv = highest_cv.max(c);
                }
                measured.push(Measured {
                    threads: cand,
                    samples: std::mem::take(&mut samples[i]),
                });
            }
            let decision = if timed_out {
                DecodeDecision {
                    threads: n_phys.max(1),
                    reason: DecodeReason::BudgetOverrun,
                }
            } else {
                choose_decode_threads(n_phys, &measured)
            };

            // Phase 3: Spin window probe (5 warmup cycles, then 50 measured cycles)
            let spin_threads = decision.threads.clamp(1, max_threads);
            for _ in 0..5 {
                state.run_job(spin_threads, 0)?;
            }
            let mut wake_us = Vec::with_capacity(WAKE_CYCLES);
            for _ in 0..WAKE_CYCLES {
                let dur = state.run_job(spin_threads, 0)?;
                wake_us.push(dur.as_secs_f64() * 1e6);
            }
            let med_wake = median(wake_us);

            Some((decision, med_wake, highest_cv))
        };

        let outcome = measure();

        // Shutdown workers
        state.shutdown.store(true, Ordering::Release);
        {
            let _g = state.wake_mutex.lock().unwrap();
            state.wake_cvar.notify_all();
        }

        outcome
    });

    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let Some((decision, med_wake, highest_cv)) = res else {
        // The workers did not come back in time: the machine cannot be measured on, and
        // startup must not wait for it. The safe settings stand.
        eprintln!(
            "peregrine: calibration gave up (the workers did not come back in {}ms), falling back to defaults",
            JOB_TIMEOUT.as_millis()
        );
        let mut settings = fallback_defaults(dur_ms, TuningSource::Default);
        settings.duration_ms = dur_ms;
        return settings;
    };
    let spin_us = (2.0 * med_wake).round().clamp(50.0, 2000.0) as u64;
    eprintln!(
        "peregrine: calibrated in {:.0}ms: decode_threads={} (of {} physical), spin_us={} (source: startup-calibration, {})",
        dur_ms,
        decision.threads,
        n_phys,
        spin_us,
        decision.reason.as_str()
    );
    // The safe settings are the base; the measured parts overwrite them.
    let mut settings = fallback_defaults(dur_ms, TuningSource::Startup);
    settings.decode_threads = decision.threads;
    settings.spin_us = spin_us;
    settings.duration_ms = dur_ms;
    settings.cv = highest_cv;
    settings
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quiet measurement of `v`: the bursts agree to within 1%, far under the cutoff.
    fn quiet(threads: usize, v: f64) -> Measured {
        Measured {
            threads,
            samples: vec![v * 0.995, v, v * 1.005],
        }
    }

    #[test]
    fn calibration_finishes_under_budget() {
        let t0 = Instant::now();
        let cal = calibrate();
        let elapsed = t0.elapsed();
        println!(
            "Calibration finished in {:?}: decode_threads={} (of {}), spin_us={}, source={:?}, cv={:.3}",
            elapsed, cal.decode_threads, cal.prefill_threads, cal.spin_us, cal.source, cal.cv
        );
        // Requirement 1: <= 300 ms budget with slack for CI (< 1s)
        assert!(
            elapsed < Duration::from_millis(1000),
            "calibration took too long: {:?}",
            elapsed
        );
        assert!(cal.decode_threads >= 1);
        assert!(cal.spin_us >= 50 && cal.spin_us <= 2000);
    }

    #[test]
    fn cv_and_median_work_correctly() {
        let vals = vec![100.0, 102.0, 98.0];
        let c = cv(&vals);
        assert!(c < 0.05, "CV should be small: {c}");
        assert_eq!(median(vals), 100.0);

        // High noise
        let noisy = vec![100.0, 150.0, 50.0];
        let c_noisy = cv(&noisy);
        assert!(c_noisy > 0.15, "CV should be > 15%: {c_noisy}");
    }

    #[test]
    fn the_candidate_set_holds_the_powers_of_two_the_halves_and_the_floor() {
        assert_eq!(candidates(12), vec![1, 2, 3, 4, 6, 8, 12]);
        assert_eq!(candidates(8), vec![1, 2, 4, 8]);
        assert_eq!(candidates(1), vec![1]);
        assert_eq!(floor_threads(12), 6);
        assert_eq!(floor_threads(3), 2);
    }

    #[test]
    fn a_noisy_high_first_sample_keeps_the_default() {
        // The failure this rule exists for. The first candidate's bursts came back far too
        // high (one of them was a spike), and the first version of the sweep froze on that
        // measurement and chose 1 thread: the 9B then decoded at 5.6 tok/s instead of 23.6.
        // A measurement that noisy is not evidence, so the default stands.
        let measured = vec![
            Measured {
                threads: 1,
                samples: vec![100.0, 400.0, 410.0],
            },
            quiet(2, 190.0),
            quiet(3, 240.0),
            quiet(4, 280.0),
            quiet(6, 320.0),
            quiet(8, 330.0),
            quiet(12, 340.0),
        ];
        let d = choose_decode_threads(12, &measured);
        assert_eq!(d.threads, 12, "the default count must stand");
        assert_eq!(d.reason, DecodeReason::DefaultStands);
    }

    #[test]
    fn a_clear_plateau_at_six_of_twelve_chooses_six() {
        // Six threads already reach 99% of the bandwidth of twelve: the pool is narrowed to
        // the smallest count that is at least as good.
        let measured = vec![
            quiet(1, 40.0),
            quiet(2, 70.0),
            quiet(3, 85.0),
            quiet(4, 92.0),
            quiet(6, 99.0),
            quiet(8, 99.5),
            quiet(12, 100.0),
        ];
        let d = choose_decode_threads(12, &measured);
        assert_eq!(d.threads, 6);
        assert_eq!(d.reason, DecodeReason::Smaller);
    }

    #[test]
    fn three_of_twelve_at_ninety_nine_percent_lands_on_the_floor() {
        // Three threads reach 99% of the bandwidth of twelve, so the measurement points at
        // three. The floor holds the count at half the physical cores: below that the decode
        // rate collapses (the 9B: 23.6 tok/s at twelve threads, 5.6 at one or two).
        let measured = vec![
            quiet(1, 40.0),
            quiet(2, 70.0),
            quiet(3, 99.0),
            quiet(4, 92.0),
            quiet(6, 94.0),
            quiet(8, 96.0),
            quiet(12, 100.0),
        ];
        let d = choose_decode_threads(12, &measured);
        assert_eq!(d.threads, 6, "the floor is n_phys / 2");
        assert_eq!(d.reason, DecodeReason::Smaller);
    }

    #[test]
    fn a_noisy_measurement_keeps_the_default() {
        // The default's own bursts disagree: the machine is busy, and nothing measured on it
        // is evidence.
        let measured = vec![
            quiet(1, 40.0),
            quiet(2, 70.0),
            quiet(3, 85.0),
            quiet(4, 92.0),
            quiet(6, 99.0),
            quiet(8, 99.5),
            Measured {
                threads: 12,
                samples: vec![100.0, 140.0, 60.0],
            },
        ];
        let d = choose_decode_threads(12, &measured);
        assert_eq!(d.threads, 12);
        assert_eq!(d.reason, DecodeReason::Noisy);
    }

    #[test]
    fn a_noisy_candidate_is_not_evidence() {
        // The candidate is fast, but its bursts disagree: it does not count.
        let measured = vec![
            quiet(1, 40.0),
            quiet(2, 70.0),
            quiet(3, 85.0),
            quiet(4, 92.0),
            Measured {
                threads: 6,
                samples: vec![99.0, 130.0, 70.0],
            },
            quiet(8, 96.0),
            quiet(12, 100.0),
        ];
        let d = choose_decode_threads(12, &measured);
        assert_eq!(d.threads, 12);
        assert_eq!(d.reason, DecodeReason::DefaultStands);
    }

    #[test]
    fn a_missing_default_measurement_keeps_the_default() {
        let measured = vec![quiet(1, 40.0), quiet(2, 70.0)];
        let d = choose_decode_threads(12, &measured);
        assert_eq!(d.threads, 12);
        assert_eq!(d.reason, DecodeReason::DefaultStands);
    }
}
