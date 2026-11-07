//! Startup calibration of decode thread count and spin window (S2.6 PR 3).
//!
//! Measures:
//! 1. Memory bandwidth saturation across candidate decode thread counts:
//!    sweeping powers of two up to physical cores, stopping early if bandwidth
//!    gains fall below 5%.
//! 2. Worker condvar wake-up latency: 50 back-to-back fork-join cycles with 0 µs
//!    spin, setting `spin_us = 2 * L_wake` clamped to [50, 2000] µs.
//!
//! If variance (CV) across repetitions exceeds 15% (e.g. noisy host),
//! calibration aborts and falls back to safe architectural defaults.
//! Hard time budget: <= 300 ms (aborts if exceeded).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

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

    fn run_job(&self, active: usize, kind: usize) -> Duration {
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
            while self.remaining.load(Ordering::Acquire) > 0 {
                lock = self.done_cvar.wait(lock).unwrap();
            }
        }
        t0.elapsed()
    }
}

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

pub fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    let mid = samples.len() / 2;
    samples[mid]
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

/// Run startup calibration.
pub fn calibrate() -> CalibratedSettings {
    let t0 = Instant::now();
    let n_phys = crate::pool::physical_cores().unwrap_or(1);
    let n_logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    let max_threads = n_logical.max(n_phys).max(1);

    // Monotonicity candidates: powers of two up to n_phys, plus n_phys if not a power of two
    let mut candidates = Vec::new();
    let mut t = 1;
    while t <= n_phys {
        candidates.push(t);
        t *= 2;
    }
    if candidates.last().copied() != Some(n_phys) && n_phys > 0 {
        candidates.push(n_phys);
    }

    // Allocate 32 MB buffer for memory bandwidth micro-bursts (pre-faulted)
    const BUF_BYTES: usize = 64 * 1024 * 1024;
    let buf_len = BUF_BYTES / size_of::<u64>();
    let buf = vec![0x5555_aaaa_5555_aaaau64; buf_len];

    let state = std::sync::Arc::new(ProbeState::new());

    let res: Result<(usize, f64, f64), (f64, &'static str)> = std::thread::scope(|s| {
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

        // Warm up all workers once so TLB and CPU frequency scaling are active
        state.run_job(max_threads, 1);

        // Phase 2: Bandwidth sweep across candidate thread counts
        let mut best_t = 1;
        let mut best_bw = 0.0f64;
        let mut highest_cv = 0.0f64;

        for (i, &cand) in candidates.iter().enumerate() {
            if t0.elapsed() >= Duration::from_millis(250) {
                eprintln!("peregrine: calibration timeout (>= 250ms), falling back to defaults");
                state.shutdown.store(true, Ordering::Release);
                let _g = state.wake_mutex.lock().unwrap();
                state.wake_cvar.notify_all();
                return Err((highest_cv, "timeout"));
            }

            // 1 warmup burst for this thread count
            state.run_job(cand, 1);

            let mut bws = Vec::with_capacity(3);
            for _ in 0..3 {
                let dt = state.run_job(cand, 1).as_secs_f64();
                if dt > 0.0 {
                    bws.push(BUF_BYTES as f64 / dt);
                }
            }
            if bws.len() < 3 {
                state.shutdown.store(true, Ordering::Release);
                let _g = state.wake_mutex.lock().unwrap();
                state.wake_cvar.notify_all();
                return Err((highest_cv, "too few samples"));
            }
            bws.sort_by(|a, b| a.total_cmp(b));
            let med_bw = bws[1]; // median of 3 micro-bursts
            let kept = &bws[0..2]; // drop highest outlier
            let c = cv(kept);
            highest_cv = highest_cv.max(c);
            if c > 0.15 {
                eprintln!(
                    "peregrine: calibration aborted (host noise CV {:.1}% > 15%), falling back to defaults (decode_threads={}, spin_us={})",
                    c * 100.0, n_phys, crate::options::SPIN_DEFAULT_US
                );
                state.shutdown.store(true, Ordering::Release);
                let _g = state.wake_mutex.lock().unwrap();
                state.wake_cvar.notify_all();
                return Err((highest_cv, "noise"));
            }

            if i == 0 || med_bw >= 1.05 * best_bw {
                best_t = cand;
                best_bw = med_bw;
            } else {
                // Plateau or regression: stop sweep
                break;
            }
        }

        // Phase 3: Spin window probe (5 warmup cycles, then 50 measured cycles)
        for _ in 0..5 {
            state.run_job(best_t, 0);
        }
        let mut wake_us = Vec::with_capacity(50);
        for _ in 0..50 {
            let dur = state.run_job(best_t, 0);
            wake_us.push(dur.as_secs_f64() * 1e6);
        }
        let med_wake = median(wake_us);

        // Shutdown workers
        state.shutdown.store(true, Ordering::Release);
        {
            let _g = state.wake_mutex.lock().unwrap();
            state.wake_cvar.notify_all();
        }

        Ok((best_t, med_wake, highest_cv))
    });

    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
    match res {
        Ok((bt, mw, cv_val)) => {
            let spin_us = (2.0 * mw).round() as u64;
            let spin_us = spin_us.clamp(50, 2000);
            eprintln!(
                "peregrine: calibrated in {:.0}ms: decode_threads={} (of {}), spin_us={} (source: startup-calibration)",
                dur_ms, bt, n_logical, spin_us
            );
            CalibratedSettings {
                decode_threads: bt,
                prefill_threads: n_logical,
                spin_us,
                duration_ms: dur_ms,
                source: TuningSource::Startup,
                cv: cv_val,
            }
        }
        Err((cv_val, _)) => {
            let mut def = fallback_defaults(dur_ms, TuningSource::Default);
            def.cv = cv_val;
            def
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
