//! A persistent worker pool with exactly one fork–join per matmul.
//!
//! # Why not rayon, and why not `thread::scope`
//!
//! rayon is excluded by the spec: its work-stealing gives neither a stable thread
//! count nor a predictable barrier count, and this engine *reports* both as measured
//! quantities. A figure that changes with the scheduler's mood is not a measurement.
//!
//! `std::thread::scope` is safe and would need no unsafe code at all, but it spawns
//! and joins on every call. A single decode step runs seven matmuls in each of 32
//! layers — 224 fork–joins per token. At the ~25 µs it costs macOS to create a
//! thread that is 5.6 ms per token against a step budget of roughly 30 ms: a fifth
//! of the time spent creating threads that do 130 µs of work each. The workers here
//! are created once per session and park on a condvar between matmuls.
//!
//! # Partitioning
//!
//! Output rows are split into `n` contiguous ranges, each worker runs its range to
//! completion, and the call returns when all have. There is no stealing and no
//! per-row synchronisation: the ranges are disjoint by construction, which is what
//! makes the raw-pointer writes in `matmul` sound.

use std::cell::{RefCell, UnsafeCell};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// A type-erased `Fn(worker_index, worker_count)`.
///
/// Erased through a monomorphised trampoline rather than by transmuting a `&dyn` fat
/// pointer into two words — the trampoline is a plain function pointer and makes no
/// assumption about trait-object layout.
#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize, usize),
}

unsafe fn trampoline<F: Fn(usize, usize) + Sync>(p: *const (), w: usize, n: usize) {
    (*(p as *const F))(w, n)
}

/// How long a worker spins on the epoch before parking on the condvar.
///
/// # Why spin at all
///
/// A decode step is ~160 fork–joins (Q/K/V, O, gate/up and down in each of 32
/// layers, attention, the head), each a few tens of microseconds of work. Parking
/// every worker on a condvar between them made every fork–join pay a kernel wake-up
/// — measured as the layer matmuls sustaining 10 GB/s while the one large matmul
/// (the 49 152-row head) reached 18 GB/s on the same kernels. Workers now spin for a
/// short window first, so back-to-back matmuls within a token never sleep, and park
/// only when the session goes quiet (between turns), so an idle engine costs no CPU.
///
/// Spinning is also what makes an oversubscribed machine worse: four sessions each
/// spinning four workers on four cores starve one another (the throughput test
/// failed only when the suite ran its tests in parallel). On a shared host with fewer
/// free cores than workers, a waiter that spins takes the core the straggler it is
/// waiting for needs, and the whole fork–join takes longer. The waiting policy below
/// therefore shortens (and finally drops) the spin window when the host says it is
/// oversubscribed.
///
/// # Why the window adapts
///
/// One number cannot be right on both kinds of host. On a quiet machine a 1 ms spin
/// window is what keeps the 160 fork–joins of a token off the kernel scheduler; with
/// `PEREGRINE_SPIN_US=0` the same desktop decode ran 8.8x slower (97 forks per token
/// paid a wake-up). On a loaded machine the same window is a self-inflicted denial
/// of service: bluefin measured Gemma-1B prefill at ~70 tok/s with the default window
/// and 120-130 tok/s with `PEREGRINE_SPIN_US=0`, on the same loaded host. llama.cpp
/// survives both by polling a fixed number of `pause` rounds and then taking the
/// mutex, so it never spins for longer than a few microseconds at a time. That is a
/// good floor but, like a fixed window, it cannot use a quiet host's spare cycles.
///
/// The pool measures what it costs instead. A thread that is runnable but not
/// running accumulates run-queue delay; `/proc/thread-self/schedstat` reports it
/// per thread on Linux. Every [`LOAD_SAMPLE`] a worker compares the delay it
/// accumulated with the runnable wall time since its last sample. A smoothed reading
/// of that share is the **gate**: above [`LOAD_QUIET_PERMILLE`] the host is busy enough
/// that the window is worth adapting, below it the configured window is kept and
/// nothing else happens.
///
/// The gate does not choose the window. It cannot: the same 40-90 % reading maps to
/// different winners on different days and models, because a spinning waiter is itself
/// part of the queue it is measuring. (An earlier version mapped the reading to a
/// half-window rung; on this host that rung lost to both the full window and parking,
/// and Gemma-3-1B regressed under four hogs.) Once the gate is open, the pool runs a
/// short A/B test of its own join wait: it holds one rung for [`HOLD_JOINS`], then
/// measures one window on each rung and keeps the one whose mean wait is lower. The
/// join wait is exactly the cost the window changes — how long the caller waits for
/// the straggler — so it compares like with like even though the pool does not know
/// which model it is serving.
///
/// Nothing here changes what a worker computes.
///
/// The default windows and their measurements are in `options.rs` (`SPIN_DEFAULT_US`).
///
/// The spin and yield windows. `PEREGRINE_SPIN_US` / `PEREGRINE_YIELD_US`
/// override them (0 and 0 park at once, which is the original condvar pool), and
/// `PEREGRINE_ADAPTIVE_SPIN=0` keeps them fixed even on a busy host, so the
/// trade-off can be measured on a given machine instead of assumed.
///
/// # The two rungs
///
/// `level` 0 is the configured window. Level 2 parks at once, like
/// `PEREGRINE_SPIN_US=0`, which is what heavy oversubscription wants. There is no
/// middle rung: on the reference host the measured optimum at two runnable threads per
/// core was either the full window or none, never half. A quiet host never leaves
/// level 0, so its fork–join path is the old pool's plus the periodic sample.
const LOAD_SAMPLE: Duration = Duration::from_millis(8);

/// A sample needs at least this much runnable time behind it to be evidence: a
/// sample taken across a park (an idle session) would otherwise be mostly denominator.
const LOAD_MIN_RUNNABLE: Duration = Duration::from_millis(4);

/// Run-queue delay that means the host has more runnable threads than free cores.
///
/// Measured on the reference host (Xeon E5-2680 v4, four threads pinned to four cores):
/// 0.2-0.6 % quiet, 40-53 % with four CPU hogs on the same four cores and 61-100 % with
/// eight. The two rungs are the configured window and parking at once; the middle rung
/// an earlier version had was measured to be worse than both on this host, so it is
/// gone. The run-queue reading is only the *gate*: it says whether the host is busy
/// enough to be worth adapting at all, not which rung is faster. The choice between the
/// rungs is made by measuring the pool's own join wait (see [`WaitControl::decide`]),
/// because the same run-queue reading maps to different winners on different days and
/// models, and a fixed threshold got it wrong in both directions.
const LOAD_BUSY_PERMILLE: u32 = 600;

/// The gate is hysteretic: trials start at or above this delay and stop below half of
/// it. A quiet host (0.2-0.6 % measured) never enters them, and neither does a host
/// with only a little co-tenant noise: the original first step (35 %) is a safe place
/// to start paying for an A/B test, and below it the configured window is kept exactly.
const LOAD_QUIET_PERMILLE: u32 = 350;

/// A trial window: at least this long and this many joins, so one long matmul or one
/// preemption cannot decide the rung. The mean is the caller's join wait, which is the
/// scheduling cost the pool is choosing between.
const TRIAL_WINDOW: Duration = Duration::from_millis(100);
const TRIAL_MIN_JOINS: u32 = 16;

/// Joins to stay on a chosen rung before testing the other one again. The test costs
/// one window on the rung that turns out to be worse.
const HOLD_JOINS: u32 = 4096;

/// The first hold is short: the starting rung comes from the run-queue reading alone,
/// and on this host that reading does not always predict the winner. Half a second at
/// a typical join rate is enough to test it, and a wrong start then costs little.
const INITIAL_HOLD_JOINS: u32 = 256;

/// Reserved for the first window after a rung change, whose wait still contains the
/// previous rung's stragglers.
const SKIP_JOINS: u32 = 8;

/// The spin window, and how much of it the pool believes it can afford.
///
/// The common case on a quiet host is "level 0, delay below the gate". It must cost the
/// old pool plus the periodic sample and nothing else, so every part of that path is an
/// atomic: the smoothed delay, the trial switch and the seed flag. The mutex is taken
/// only while a trial is running, which implies a busy host.
struct WaitControl {
    /// The windows from `options.rs`, never exceeded.
    max_spin_us: u64,
    max_yield_us: u64,
    /// 0 = configured, 2 = park at once. Kept as one value rather than two durations
    /// so a waiter cannot observe a half-updated pair.
    level: AtomicU8,
    /// Smoothed run-queue delay in permille, scaled by 1024. The busy gate.
    ewma: AtomicU64,
    /// Whether the smoothing has a first sample.
    seeded: AtomicBool,
    /// Whether the join-wait trials are running (a cheap check before the mutex).
    trialing: AtomicBool,
    /// Last diagnostic print, for `PEREGRINE_ADAPTIVE_LOG` (milliseconds).
    logged: AtomicU64,
    /// `PEREGRINE_ADAPTIVE_SPIN=0` keeps the configured window.
    enabled: bool,
    /// Trial state; only the caller thread uses it, one join at a time.
    trial: Mutex<Trial>,
}

struct Trial {
    active: bool,
    rung: u8,
    mode: Mode,
    sum_ns: u128,
    count: u32,
    start: Instant,
    skip: u32,
    /// Mean wait of the incumbent's window in the current probe.
    incumbent: Option<f64>,
}

#[derive(Clone, Copy)]
enum Mode {
    /// Stay on the chosen rung for this many more joins.
    Hold(u32),
    /// Measure the current rung for one window, then `other`.
    Probe { other: u8 },
}

/// Diagnostic counters for the ladder, off unless `PEREGRINE_ADAPTIVE_LOG` is set.
///
/// The loaded matrix in the PR body had one result that did not fit the story
/// (Gemma-3-1B under four hogs), and the only way to see why was to know which rung
/// the pool actually sat on. These counters add nothing to the hot path beyond an
/// env-var check made once: transitions are rare, and the wall-time buckets are
/// already available at the sampling point that runs every 8 ms.
static ADAPTIVE_START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
static LEVEL_NANOS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static LEVEL_SAMPLES: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static TRANSITIONS: AtomicU64 = AtomicU64::new(0);
/// How long a parked worker actually stayed parked, in five buckets (100 µs, 500 µs,
/// 1 ms, 5 ms, more). This separates "parking saved a long wait" from "parking paid a
/// wake-up for a region that was about to arrive".
static PARK_BUCKETS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn park_bucket(d: Duration) -> usize {
    match d.as_micros() {
        0..=99 => 0,
        100..=499 => 1,
        500..=999 => 2,
        1000..=4999 => 3,
        _ => 4,
    }
}

/// Whether `PEREGRINE_ADAPTIVE_LOG` asks for the ladder diagnostic.
fn adaptive_log() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("PEREGRINE_ADAPTIVE_LOG").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

fn record_level_time(level: u8, d: Duration) {
    if adaptive_log() {
        let level = usize::from(level.min(2));
        LEVEL_NANOS[level].fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
        LEVEL_SAMPLES[level].fetch_add(1, Ordering::Relaxed);
    }
}

fn print_adaptive_totals() {
    if !adaptive_log() {
        return;
    }
    let ms = |ns: u64| ns as f64 / 1e6;
    eprintln!(
        "adaptive: ladder wall per thread: level0 {:.1} ms ({}), level1 {:.1} ms ({}), level2 {:.1} ms ({}), transitions {}",
        ms(LEVEL_NANOS[0].load(Ordering::Relaxed)),
        LEVEL_SAMPLES[0].load(Ordering::Relaxed),
        ms(LEVEL_NANOS[1].load(Ordering::Relaxed)),
        LEVEL_SAMPLES[1].load(Ordering::Relaxed),
        ms(LEVEL_NANOS[2].load(Ordering::Relaxed)),
        LEVEL_SAMPLES[2].load(Ordering::Relaxed),
        TRANSITIONS.load(Ordering::Relaxed),
    );
    eprintln!(
        "adaptive: park durations <100us {}, <500us {}, <1ms {}, <5ms {}, >=5ms {}",
        PARK_BUCKETS[0].load(Ordering::Relaxed),
        PARK_BUCKETS[1].load(Ordering::Relaxed),
        PARK_BUCKETS[2].load(Ordering::Relaxed),
        PARK_BUCKETS[3].load(Ordering::Relaxed),
        PARK_BUCKETS[4].load(Ordering::Relaxed),
    );
}

impl WaitControl {
    fn new(spin_us: u64, yield_us: u64, enabled: bool) -> WaitControl {
        WaitControl {
            max_spin_us: spin_us,
            max_yield_us: yield_us,
            level: AtomicU8::new(0),
            ewma: AtomicU64::new(0),
            seeded: AtomicBool::new(false),
            trialing: AtomicBool::new(false),
            logged: AtomicU64::new(0),
            enabled,
            trial: Mutex::new(Trial {
                active: false,
                rung: 0,
                mode: Mode::Hold(HOLD_JOINS),
                sum_ns: 0,
                count: 0,
                start: Instant::now(),
                skip: 0,
                incumbent: None,
            }),
        }
    }

    /// The (spin, yield) deadlines for the current level.
    ///
    /// The yield deadline is never earlier than the spin deadline: `backoff` treats both
    /// as absolute offsets from the start of the wait, so a configured `YIELD_US` of 0
    /// means "no yield phase" and not "yield for minus one millisecond".
    fn budgets(&self) -> (Duration, Duration) {
        let spin = match self.level.load(Ordering::Relaxed) {
            0 => self.max_spin_us,
            _ => 0,
        };
        let yld = self.max_yield_us.max(spin);
        (Duration::from_micros(spin), Duration::from_micros(yld))
    }

    /// Update the busy gate from one scheduling sample.
    ///
    /// This is the only thing the sampling does: it does not choose a rung. A smoothed
    /// reading (one eighth per sample) above [`LOAD_QUIET_PERMILLE`] lets the trials run;
    /// a reading below half that stops them and restores the configured window. A quiet
    /// host therefore never leaves level 0 and pays one atomic update per 8 ms sample
    /// and nothing else.
    fn observe(&self, delay_permille: u32, now: Instant) {
        if !self.enabled {
            return;
        }
        let new = u64::from(delay_permille) * 1024;
        if !self.seeded.swap(true, Ordering::Relaxed) {
            self.ewma.store(new, Ordering::Relaxed);
            return;
        }
        let old = self.ewma.load(Ordering::Relaxed);
        let ewma = old - old / 8 + new / 8;
        self.ewma.store(ewma, Ordering::Relaxed);
        let smoothed = u32::try_from(ewma / 1024).unwrap_or(1000);
        if adaptive_log() {
            let start = *ADAPTIVE_START.get_or_init(Instant::now);
            let ms = now.saturating_duration_since(start).as_millis() as u64;
            if ms.saturating_sub(self.logged.load(Ordering::Relaxed)) >= 500 {
                self.logged.store(ms, Ordering::Relaxed);
                eprintln!(
                    "adaptive: t={ms} ms level {} smoothed {smoothed} permille",
                    self.level.load(Ordering::Relaxed)
                );
            }
        }
    }

    /// Whether this join should be timed for the trial controller.
    ///
    /// Two relaxed loads on a quiet host; no clock read until the gate opens.
    fn wants_sample(&self) -> bool {
        self.enabled
            && (self.trialing.load(Ordering::Relaxed)
                || u32::try_from(self.ewma.load(Ordering::Relaxed) / 1024).unwrap_or(1000)
                    >= LOAD_QUIET_PERMILLE)
    }

    /// Feed the caller's join wait to the controller and return the rung it wants.
    ///
    /// The controller is an A/B test of the pool's own work, not a model of the host:
    /// it holds a rung for [`HOLD_JOINS`], then measures one window on each rung and
    /// keeps the one whose mean wait is lower (with a 10 % margin for staying put). The
    /// join wait is exactly the cost the rung changes — the caller's wait for the
    /// straggler — so it is comparable across windows even though the model is not
    /// known. Starting rung: parking when the gate says the queue is deep, the window
    /// otherwise.
    fn decide(&self, wait_ns: u64, now: Instant) {
        if !self.enabled {
            return;
        }
        let smoothed = u32::try_from(self.ewma.load(Ordering::Relaxed) / 1024).unwrap_or(1000);
        if !self.trialing.load(Ordering::Relaxed) && smoothed < LOAD_QUIET_PERMILLE {
            return;
        }
        let mut g = self.trial.lock().unwrap_or_else(|e| e.into_inner());
        if !g.active {
            if smoothed < LOAD_QUIET_PERMILLE {
                return;
            }
            g.active = true;
            g.rung = if smoothed >= LOAD_BUSY_PERMILLE { 2 } else { 0 };
            g.mode = Mode::Hold(INITIAL_HOLD_JOINS);
            g.incumbent = None;
            self.level.store(g.rung, Ordering::Relaxed);
            self.trialing.store(true, Ordering::Relaxed);
            if adaptive_log() {
                let start = *ADAPTIVE_START.get_or_init(Instant::now);
                eprintln!(
                    "adaptive: +{:>9.3} ms trial start on level {} (smoothed {smoothed})",
                    now.saturating_duration_since(start).as_secs_f64() * 1e3,
                    g.rung
                );
            }
            return;
        }
        if smoothed < LOAD_QUIET_PERMILLE / 2 {
            // The host is genuinely quiet: put the configured window back and stop.
            g.active = false;
            g.rung = 0;
            self.level.store(0, Ordering::Relaxed);
            self.trialing.store(false, Ordering::Relaxed);
            return;
        }
        match g.mode {
            Mode::Hold(n) if n > 0 => {
                g.mode = Mode::Hold(n - 1);
                return;
            }
            Mode::Hold(_) => {
                let other = if g.rung == 0 { 2 } else { 0 };
                g.mode = Mode::Probe { other };
                g.incumbent = None;
                g.skip = SKIP_JOINS;
                g.sum_ns = 0;
                g.count = 0;
                g.start = now;
            }
            Mode::Probe { .. } => {}
        }
        if g.skip > 0 {
            g.skip -= 1;
            return;
        }
        g.sum_ns += u128::from(wait_ns);
        g.count += 1;
        if g.count < TRIAL_MIN_JOINS || now.saturating_duration_since(g.start) < TRIAL_WINDOW {
            return;
        }
        let mean = g.sum_ns as f64 / f64::from(g.count);
        g.sum_ns = 0;
        g.count = 0;
        g.start = now;
        match g.mode {
            Mode::Probe { other } => {
                if let Some(incumbent) = g.incumbent.take() {
                    // Both rungs measured: keep the previous rung unless the challenger
                    // is at least 10 % better. Ties keep the current rung, which is what
                    // stops a quiet-ish host from flapping.
                    let keep = if mean < incumbent * 0.90 {
                        g.rung
                    } else if incumbent < mean * 0.90 {
                        if g.rung == 0 {
                            2
                        } else {
                            0
                        }
                    } else {
                        g.rung
                    };
                    g.rung = keep;
                    self.level.store(keep, Ordering::Relaxed);
                    g.mode = Mode::Hold(HOLD_JOINS);
                    if adaptive_log() {
                        let start = *ADAPTIVE_START.get_or_init(Instant::now);
                        eprintln!(
                            "adaptive: +{:>9.3} ms trial chose level {} (challenger {:.1} us, incumbent {:.1} us)",
                            now.saturating_duration_since(start).as_secs_f64() * 1e3,
                            keep,
                            mean / 1e3,
                            incumbent / 1e3
                        );
                    }
                } else {
                    // First window: the incumbent. Switch to the challenger.
                    g.incumbent = Some(mean);
                    g.rung = other;
                    self.level.store(other, Ordering::Relaxed);
                    g.skip = SKIP_JOINS;
                }
            }
            Mode::Hold(_) => unreachable!("hold was handled above"),
        }
    }
}

/// What one thread has learnt about being scheduled since its last sample.
struct LoadProbe {
    last: Option<(Instant, u64)>,
    /// Wall time spent parked, which is not time the thread was runnable: it must be
    /// taken out of the denominator or an idle session would look idle-but-loaded.
    parked: Duration,
}

impl LoadProbe {
    fn new() -> LoadProbe {
        LoadProbe {
            last: None,
            parked: Duration::ZERO,
        }
    }

    fn add_parked(&mut self, d: Duration) {
        self.parked += d;
        if adaptive_log() {
            PARK_BUCKETS[park_bucket(d)].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Sample the thread's run-queue delay, at most once per [`LOAD_SAMPLE`].
    fn sample(&mut self, ctl: &WaitControl, now: Instant) {
        let Some((then, before)) = self.last else {
            self.last = run_delay_ns().map(|ns| (now, ns));
            self.parked = Duration::ZERO;
            return;
        };
        let wall = now.saturating_duration_since(then);
        if wall < LOAD_SAMPLE {
            return;
        }
        let Some(after) = run_delay_ns() else {
            self.last = None;
            self.parked = Duration::ZERO;
            return;
        };
        let runnable = wall.saturating_sub(self.parked);
        if runnable < LOAD_MIN_RUNNABLE {
            // Too little running behind the sample to judge a ratio by. Keep the
            // earlier reference point and the parked time: a thread parked most of
            // the time (level 2 on a short-region model) accumulates enough runnable
            // time across several waits, and discarding it here is what used to leave
            // the ladder stuck at zero for tens of seconds.
            return;
        }
        let parked = std::mem::take(&mut self.parked);
        self.last = Some((now, after));
        let runnable = wall.saturating_sub(parked);
        let delayed = after.saturating_sub(before);
        let permille = (u128::from(delayed) * 1000 / runnable.as_nanos().max(1)) as u32;
        record_level_time(ctl.level.load(Ordering::Relaxed), wall);
        ctl.observe(permille.min(1000), now);
    }
}

thread_local! {
    /// One probe per thread: what it has to say about the scheduler is its own, and a
    /// lock around it would put a mutex in the fork–join path this pool exists to keep
    /// cheap.
    static PROBE: RefCell<LoadProbe> = RefCell::new(LoadProbe::new());
}

/// Sample this thread's scheduling pressure. Cheap when not due (a duration compare).
fn sample_load(ctl: &WaitControl, now: Instant) {
    PROBE.with(|p| p.borrow_mut().sample(ctl, now));
}

/// Time this thread has spent runnable but not running, in nanoseconds.
///
/// Linux reports it per thread in `/proc/thread-self/schedstat` (second field, in
/// nanoseconds since the thread started). `/proc/thread-self` is the calling thread,
/// so no thread id lookup is needed. Elsewhere there is no portable equivalent; the
/// probe returns `None` and the configured window is used unchanged.
#[cfg(target_os = "linux")]
fn run_delay_ns() -> Option<u64> {
    use std::io::Read;
    let mut buf = [0u8; 96];
    let mut f = std::fs::File::open("/proc/thread-self/schedstat").ok()?;
    let n = f.read(&mut buf).ok()?;
    parse_run_delay(std::str::from_utf8(&buf[..n]).ok()?)
}

#[cfg(not(target_os = "linux"))]
fn run_delay_ns() -> Option<u64> {
    None
}

/// The second field of a `schedstat` line: run-queue delay in nanoseconds.
///
/// Only Linux reads it in a normal build; the unit test exercises the parser on every
/// platform, so a `cfg(test)` build keeps it there too.
#[cfg(any(target_os = "linux", test))]
fn parse_run_delay(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Wait until `done()` holds: spin, then yield, then (if `park` is given) block.
#[inline]
fn backoff(t0: Instant, spins: &mut u32, spin: Duration, yld: Duration) -> Phase {
    *spins = spins.wrapping_add(1);
    if !spins.is_multiple_of(64) {
        std::hint::spin_loop();
        return Phase::Spin;
    }
    let e = t0.elapsed();
    if e < spin {
        std::hint::spin_loop();
        Phase::Spin
    } else if e < yld {
        std::thread::yield_now();
        Phase::Yield
    } else {
        Phase::Park
    }
}

#[derive(PartialEq)]
enum Phase {
    Spin,
    Yield,
    Park,
}

#[repr(align(64))]
struct Aligned<T>(T);

struct Inner {
    /// Bumped once per `run`; a worker runs the job when it sees a new value. Workers
    /// compare epochs rather than read a flag, so a worker that was slow to notice a
    /// job cannot run it twice or miss the next one.
    epoch: Aligned<AtomicU64>,
    /// Written by `run` before the epoch is published (Release) and read by workers
    /// after they observe it (Acquire).
    job: UnsafeCell<Option<Job>>,
    /// Workers still running the current job, on its own cache line so finishing
    /// workers do not invalidate `epoch` and `job` for workers still waking up.
    remaining: Aligned<AtomicUsize>,
    shutdown: AtomicBool,
    /// Set when a job panicked on a worker. The worker still counts itself done —
    /// otherwise the caller would spin forever — and `run` re-raises on its own
    /// thread, where the session's `catch_unwind` turns it into an error status.
    poisoned: AtomicBool,
    /// Non-zero when any worker is parked on `wake`, and true when the caller is
    /// parked on `done_cv`. Updated under the corresponding mutex with `SeqCst`
    /// ordering before re-checking the condition, so the hot spinning path takes
    /// neither mutex when nobody is asleep.
    sleepers: AtomicUsize,
    caller_sleeping: AtomicBool,
    /// True while a caller is inside `run`. A pool runs one job at a time; engines
    /// that share a pool take the server's `PassTurn` first, and a second caller is
    /// refused with a panic rather than left to overwrite a job workers still read.
    in_run: AtomicBool,
    /// Parking for idle workers. `park` is only touched under the mutex, which is
    /// what makes "publish then notify if anyone sleeps" race-free.
    park: Mutex<usize>,
    wake: Condvar,
    /// The caller sleeps here once its spin window has passed, and the last worker
    /// to finish wakes it. Yielding in a loop instead was measured to *lose* to the
    /// condvar pool under load: on macOS a yielding thread keeps competing with the
    /// very worker it is waiting for.
    done_mu: Mutex<()>,
    done_cv: Condvar,
    /// Fixed for the pool's lifetime. Held here rather than derived from the job
    /// count: `remaining` decrements as workers finish, so a worker that read the
    /// width late would be told there were fewer workers than there are and would
    /// compute a share overlapping someone else's.
    n: usize,
    /// The spin window and its adaptive ladder (see [`WaitControl`]).
    wait: WaitControl,
}

// SAFETY: `job` is written only by the thread inside `run`, before it publishes a new
// epoch with Release ordering and while no worker is running a job (the previous
// `run` waited for `remaining == 0`); workers read it only after an Acquire load of
// the new epoch. The pointer it holds is valid until `run` returns, which is after
// every worker has finished.
unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

/// Pin a pool thread to a physical core when `PEREGRINE_AFFINITY=1`; a no-op otherwise
/// and on platforms without `sched_setaffinity`.
///
/// Worker `w` takes core `w % cores` and, once the pool is wider than the machine's core count,
/// the `w / cores`-th logical CPU of that core — so a pool as wide as the physical cores puts one
/// worker on each core and never two on one core's siblings, and a wider pool fills the siblings
/// in a fixed order instead of leaving it to the scheduler. A pool wider than cores × threads per
/// core wraps, sharing the primary threads.
///
/// The caller is worker 0 and is pinned too, which is the point: leaving it free lets the OS put
/// it on a core that a worker already owns.
fn pin_worker(w: usize) {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        if !crate::options::get().affinity {
            return;
        }
        let Some(map) = physical_cpu_map() else {
            return;
        };
        let core = &map[w % map.len()];
        let cpu = core.get(w / map.len()).copied().unwrap_or(core[0]);
        // sched_setaffinity(0, ...) is the calling thread only, so a worker pins itself.
        const CPUSET_BYTES: usize = 128; // 1024 CPUs, the glibc cpu_set_t
        if cpu >= CPUSET_BYTES * 8 {
            return;
        }
        let mut mask = [0u64; CPUSET_BYTES / 8];
        mask[cpu / 64] |= 1u64 << (cpu % 64);
        extern "C" {
            fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
        }
        // SAFETY: a libc call that only sets the calling thread's CPU mask; the mask
        // is a live array of `CPUSET_BYTES` bytes for the duration of the call.
        unsafe {
            sched_setaffinity(0, CPUSET_BYTES, mask.as_ptr());
        }
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = w;
    }
}

/// The logical CPUs of every physical core, in the order `/proc/cpuinfo` lists them:
/// `[[0, 12], [1, 13], ...]` on a two-way-SMT machine. `None` when the topology is unavailable.
pub fn physical_cpu_map() -> Option<&'static Vec<Vec<usize>>> {
    static MAP: std::sync::OnceLock<Option<Vec<Vec<usize>>>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        let mut order: Vec<(String, String)> = Vec::new();
        let mut cpus: Vec<Vec<usize>> = Vec::new();
        let (mut pkg, mut core, mut cpu) = (None::<String>, None::<String>, None::<usize>);
        for line in info.lines().chain(std::iter::once("")) {
            if line.trim().is_empty() {
                if let (Some(p), Some(c), Some(u)) = (pkg.take(), core.take(), cpu.take()) {
                    let key = (p, c);
                    match order.iter().position(|k| *k == key) {
                        Some(i) => cpus[i].push(u),
                        None => {
                            order.push(key);
                            cpus.push(vec![u]);
                        }
                    }
                }
                continue;
            }
            let mut kv = line.splitn(2, ':');
            let (k, v) = (
                kv.next().unwrap_or("").trim(),
                kv.next().unwrap_or("").trim().to_string(),
            );
            match k {
                "physical id" => pkg = Some(v),
                "core id" => core = Some(v),
                "processor" => cpu = v.parse().ok(),
                _ => {}
            }
        }
        (!cpus.is_empty()).then_some(cpus)
    })
    .as_ref()
}

pub struct Pool {
    n: usize,
    inner: Arc<Inner>,
    handles: Vec<JoinHandle<()>>,
}

/// Ask the scheduler to treat the current thread as work a user is waiting on.
///
/// On macOS a thread's QoS class decides how it competes with everything else on
/// the machine. The default for a GUI app's threads is fine while the window is in
/// front; when it is not, the system demotes the process (App Nap) and its threads
/// ran at priority 4 against 31 for a terminal process. A decode step is about 130
/// fork-joins, each waiting on its slowest worker, so one demoted worker
/// behind a busy `mediaanalysisd` made the app 2.5x slower than the same engine
/// run from a shell. `USER_INITIATED` is the class Apple documents for exactly
/// this: work started by the user whose result they are waiting for.
/// Elsewhere this is a no-op; Linux and Windows do not demote a backgrounded app.
pub fn raise_priority() {
    #[cfg(target_os = "macos")]
    {
        const QOS_CLASS_USER_INITIATED: u32 = 0x19;
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        // SAFETY: a libSystem call that only changes the calling thread's own
        // scheduling class; it takes no pointers. A failure leaves the class as it was.
        unsafe {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
        }
    }
}

impl Pool {
    pub fn new(n: usize) -> Pool {
        let n = n.max(1);
        // Worker 0 is the calling thread, so it is raised along with the others.
        raise_priority();
        pin_worker(0);
        let o = crate::options::get();
        let inner = Arc::new(Inner {
            epoch: Aligned(AtomicU64::new(0)),
            job: UnsafeCell::new(None),
            remaining: Aligned(AtomicUsize::new(0)),
            shutdown: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            caller_sleeping: AtomicBool::new(false),
            in_run: AtomicBool::new(false),
            park: Mutex::new(0),
            wake: Condvar::new(),
            done_mu: Mutex::new(()),
            done_cv: Condvar::new(),
            n,
            wait: WaitControl::new(o.spin_us, o.yield_us, o.adaptive_spin),
        });
        let mut handles = Vec::with_capacity(n - 1);
        // Worker 0 is the calling thread, so only n-1 are spawned. That keeps the
        // caller busy instead of blocked, and makes `n == 1` allocate nothing.
        for w in 1..n {
            let inner = Arc::clone(&inner);
            handles.push(std::thread::spawn(move || {
                raise_priority();
                pin_worker(w);
                let mut seen = 0u64;
                loop {
                    // Spin, then park. The window is the current rung of the adaptive
                    // ladder, read once per wait so a job cannot see it change midway.
                    let t0 = Instant::now();
                    sample_load(&inner.wait, t0);
                    let (spin, yld) = inner.wait.budgets();
                    let mut spins = 0u32;
                    loop {
                        if inner.epoch.0.load(Ordering::Acquire) != seen
                            || inner.shutdown.load(Ordering::Acquire)
                        {
                            break;
                        }
                        if backoff(t0, &mut spins, spin, yld) == Phase::Park {
                            let mut g = inner.park.lock().unwrap_or_else(|e| e.into_inner());
                            *g += 1;
                            inner.sleepers.store(*g, Ordering::SeqCst);
                            let parked = Instant::now();
                            while inner.epoch.0.load(Ordering::SeqCst) == seen
                                && !inner.shutdown.load(Ordering::SeqCst)
                            {
                                g = inner.wake.wait(g).unwrap_or_else(|e| e.into_inner());
                            }
                            let idle = parked.elapsed();
                            *g -= 1;
                            inner.sleepers.store(*g, Ordering::SeqCst);
                            drop(g);
                            PROBE.with(|p| p.borrow_mut().add_parked(idle));
                            break;
                        }
                    }
                    if inner.shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    seen = inner.epoch.0.load(Ordering::Acquire);
                    // SAFETY: see the `Sync` impl — published before the epoch.
                    let job = unsafe { *inner.job.get() };
                    if let Some(j) = job {
                        // SAFETY: `j.data` points at the closure passed to `run`,
                        // which spins until `remaining` reaches zero — so the closure
                        // is still alive here.
                        let ok =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                                (j.call)(j.data, w, inner.n)
                            }));
                        if ok.is_err() {
                            inner.poisoned.store(true, Ordering::Release);
                        }
                    }
                    if inner.remaining.0.fetch_sub(1, Ordering::SeqCst) == 1
                        && inner.caller_sleeping.load(Ordering::SeqCst)
                    {
                        // Last one out: wake the caller if it went to sleep. Taking the
                        // lock orders this after the caller's check-then-wait.
                        let _g = inner.done_mu.lock().unwrap_or_else(|e| e.into_inner());
                        inner.done_cv.notify_one();
                    }
                }
            }));
        }
        Pool { n, inner, handles }
    }

    pub fn width(&self) -> usize {
        self.n
    }

    /// Run `f(worker, n)` on every worker and return once all have finished.
    ///
    /// A panic in any share — the caller's or a worker's — is re-raised on the
    /// caller only after every worker has finished with `f`: `f` and whatever it
    /// borrows live on the caller's stack, so unwinding earlier would free them under
    /// the workers (and leave `remaining` counting into the next job).
    pub fn run<F: Fn(usize, usize) + Sync>(&self, f: F) {
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        crate::blocks::par::bump_epoch();
        if self.n == 1 {
            f(0, 1);
            return;
        }
        // One caller at a time. This is what the SAFETY comment below rests on: a pool
        // shared by several engines (S4.10) is only sound while their passes take
        // turns, so a caller that arrives while another is inside is a bug in the
        // caller and is stopped here, before it can touch the job.
        if self.inner.in_run.swap(true, Ordering::Acquire) {
            panic!(
                "Pool::run was entered by two callers at once; engines that share a pool \
                 must hold the pass turn"
            );
        }
        struct Running<'a>(&'a AtomicBool);
        impl Drop for Running<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _running = Running(&self.inner.in_run);
        let job = Job {
            data: &f as *const F as *const (),
            call: trampoline::<F>,
        };
        // SAFETY: this is the only caller inside `run` (checked above), no worker is
        // running a job (the previous call waited for `remaining == 0`), and none reads
        // `job` until it sees the new epoch.
        unsafe { *self.inner.job.get() = Some(job) };
        self.inner.remaining.0.store(self.n - 1, Ordering::Release);
        self.inner.epoch.0.fetch_add(1, Ordering::SeqCst);
        if self.inner.sleepers.load(Ordering::SeqCst) > 0 {
            let g = self.inner.park.lock().unwrap_or_else(|e| e.into_inner());
            if *g > 0 {
                self.inner.wake.notify_all();
            }
        }
        // The caller takes share 0 rather than blocking immediately.
        let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(0, self.n)));
        // How long the caller then waits for the last worker. `PEREGRINE_PROF2` reports
        // it as "barrier wait"; it overlaps the matmul wall and is never added to it.
        let wait_t0 = crate::prof::enabled().then(Instant::now);
        // Spin for the stragglers: the remaining work is microseconds, and sleeping
        // here would hand the wake-up latency straight back. The window is the same
        // adaptive rung the workers use.
        let (spin, yld) = self.inner.wait.budgets();
        let t0 = Instant::now();
        sample_load(&self.inner.wait, t0);
        loop {
            if self.inner.remaining.0.load(Ordering::Acquire) == 0 {
                break;
            }
            let e = t0.elapsed();
            if e < spin {
                std::hint::spin_loop();
                continue;
            }
            if e < yld {
                std::thread::yield_now();
                continue;
            }
            let mut g = self.inner.done_mu.lock().unwrap_or_else(|e| e.into_inner());
            self.inner.caller_sleeping.store(true, Ordering::SeqCst);
            let parked = Instant::now();
            while self.inner.remaining.0.load(Ordering::SeqCst) > 0 {
                g = self
                    .inner
                    .done_cv
                    .wait(g)
                    .unwrap_or_else(|e| e.into_inner());
            }
            let idle = parked.elapsed();
            self.inner.caller_sleeping.store(false, Ordering::SeqCst);
            drop(g);
            PROBE.with(|p| p.borrow_mut().add_parked(idle));
        }
        if self.inner.wait.wants_sample() {
            self.inner
                .wait
                .decide(t0.elapsed().as_nanos() as u64, Instant::now());
        }
        if let Some(t) = wait_t0 {
            crate::prof::add(crate::prof::Slot::Wait, t.elapsed().as_nanos() as u64);
        }
        crate::blocks::par::bump_epoch();
        let poisoned = self.inner.poisoned.swap(false, Ordering::AcqRel);
        if let Err(p) = mine {
            std::panic::resume_unwind(p);
        }
        if poisoned {
            panic!("a pool worker panicked while running a job");
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        print_adaptive_totals();
        self.inner.shutdown.store(true, Ordering::Release);
        {
            let _g = self.inner.park.lock().unwrap_or_else(|e| e.into_inner());
            self.inner.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// One process-wide turn at a time, shared by the engines of a multi-model server.
///
/// A server with several models gives each model its own `Engine` but one process-wide
/// thread pool, so only one pass may run at a time: two passes at once would put twice
/// the threads on the same cores. Each engine thread takes a turn around one
/// `Engine::step`, so the models interleave pass by pass.
///
/// The next turn goes to the busy engine that has had the least pass time, with arrival
/// order breaking ties. A model whose passes are cheap (a tiny decision model computes
/// one in under a millisecond) therefore takes several turns in a row until it has had
/// its share of the machine, instead of receiving one pass for every long pass of a
/// chat model. Every turn is still exactly one pass, and a model that asks waits at
/// most about one pass of the other models. A model that stops having work calls
/// [`PassTurn::leave`], so the turn never rests on an idle engine.
pub struct PassTurn {
    n: usize,
    inner: Mutex<TurnState>,
    wake: Condvar,
}

#[derive(Default)]
struct TurnState {
    /// Nanoseconds of pass time each participant has received.
    served: Vec<u64>,
    /// The participants with work (an active slot or a queued job).
    busy: Vec<bool>,
    /// Who is inside `acquire`, waiting for its turn.
    wanting: Vec<bool>,
    /// The arrival ticket of each waiter.
    ticket: Vec<u64>,
    next_ticket: u64,
    /// The participant whose turn it is next.
    turn: usize,
    holders: usize,
    max_holders: usize,
    passes: u64,
}

impl PassTurn {
    /// A turn for `participants` engines, indexed `0..participants`.
    pub fn new(participants: usize) -> PassTurn {
        assert!(participants >= 1, "a turn needs a participant");
        PassTurn {
            n: participants,
            inner: Mutex::new(TurnState {
                served: vec![0; participants],
                busy: vec![false; participants],
                wanting: vec![false; participants],
                ticket: vec![0; participants],
                ..TurnState::default()
            }),
            wake: Condvar::new(),
        }
    }

    /// Mark participant `who` as having work; it stays in the rotation until
    /// [`PassTurn::leave`].
    pub fn enter(&self, who: usize) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !g.busy[who] {
            // No credit for a long idle spell, but a bounded one: a model that returns
            // may catch up by at most [`MAX_CREDIT_NS`] of service, so it can take a few
            // cheap passes in a row without holding the others for long.
            if let Some(front) = (0..self.n)
                .filter(|&j| j != who && g.busy[j])
                .map(|j| g.served[j])
                .max()
            {
                g.served[who] = g.served[who].max(front.saturating_sub(MAX_CREDIT_NS));
            }
        }
        g.busy[who] = true;
        if g.holders == 0 && g.turn == who {
            drop(g);
            self.wake.notify_all();
        }
    }

    /// Mark participant `who` as having no work. If the turn was resting on it, the next
    /// busy (or next asking) participant takes over.
    pub fn leave(&self, who: usize) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.busy[who] = false;
        if g.holders == 0 && g.turn == who {
            g.turn = next_participant(&g, who);
        }
        drop(g);
        self.wake.notify_all();
    }

    /// Wait for participant `who`'s turn and hold it until the guard is dropped.
    pub fn acquire(&self, who: usize) -> TurnGuard<'_> {
        assert!(who < self.n, "participant {who} is out of range");
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.ticket[who] = g.next_ticket;
        g.next_ticket += 1;
        g.wanting[who] = true;
        loop {
            // Run when no one holds the turn and either it is my turn, or the
            // participant scheduled now has stopped asking and has no work.
            if g.holders == 0 && (g.turn == who || (!g.busy[g.turn] && !g.wanting[g.turn])) {
                break;
            }
            g = self.wake.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        g.turn = who;
        g.holders = 1;
        g.max_holders = g.max_holders.max(g.holders);
        g.passes += 1;
        let started = Instant::now();
        drop(g);
        TurnGuard {
            turn: self,
            who,
            started,
        }
    }

    /// Participants that want the turn but do not hold it: the callers waiting.
    pub fn waiting(&self) -> usize {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.wanting.iter().filter(|w| **w).count() - g.holders
    }

    /// The most callers that ever held the turn at once. One means the turns did
    /// their job; a larger value is a bug.
    pub fn max_holders(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .max_holders
    }

    /// Passes run under this turn, for `/metrics` and tests.
    pub fn passes(&self) -> u64 {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).passes
    }
}

impl Default for PassTurn {
    fn default() -> Self {
        Self::new(1)
    }
}

/// How far behind an idle model may fall and still catch up in one burst.
///
/// About one long pass: a returning model takes a few cheap passes at most before the
/// rotation is even again, and no other model waits longer than that.
const MAX_CREDIT_NS: u64 = 50_000_000;

/// The busy participant after `who` with the least service; ties go to the earlier
/// ticket, then to the lower index. When none is busy, the next participant that is
/// asking, and failing that the next index, keeps the rotation moving.
fn next_participant(g: &TurnState, who: usize) -> usize {
    let n = g.served.len();
    let mut best: Option<usize> = None;
    for j in 0..n {
        if !g.busy[j] {
            continue;
        }
        best = Some(match best {
            None => j,
            Some(b) if (g.served[j], g.ticket[j], j) < (g.served[b], g.ticket[b], b) => j,
            Some(b) => b,
        });
    }
    if best.is_some() {
        return best.unwrap_or(who);
    }
    for step in 1..=n {
        let j = (who + step) % n;
        if g.wanting[j] {
            return j;
        }
    }
    (who + 1) % n
}

/// A held turn, released on drop.
pub struct TurnGuard<'a> {
    turn: &'a PassTurn,
    who: usize,
    started: Instant,
}

impl Drop for TurnGuard<'_> {
    fn drop(&mut self) {
        let mut g = self.turn.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.served[self.who] += self.started.elapsed().as_nanos() as u64;
        g.wanting[self.who] = false;
        g.holders = 0;
        g.turn = next_participant(&g, self.who);
        drop(g);
        self.turn.wake.notify_all();
    }
}

/// The physical cores — distinct `(package, core id)` pairs in `/proc/cpuinfo` on Linux, the
/// performance cores on Apple Silicon — when the OS says.
///
/// Linux on aarch64 does not print `physical id`/`core id` in `/proc/cpuinfo` (S2.8), so a host
/// that names neither falls back to the CPUs this process may actually run on. On the ARM
/// machines this was written for there is no SMT, so that number is the physical core count too;
/// the fallback also honours a `taskset` mask, which is what the benchmark's affinity runs need.
///
/// On macOS an Apple chip mixes performance and efficiency cores, and a synchronous pool as wide
/// as every core is slower than one on the performance cores: the efficiency cores take a share
/// of each fork–join and finish it later. The default width is `hw.perflevel0.physicalcpu`, the
/// performance-core count, falling back to `hw.physicalcpu` and then to the CPUs this process may
/// run on (a VM can cap them below the host's count).
pub fn physical_cores() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        let available = std::thread::available_parallelism().ok().map(|n| n.get());
        let cores = macos_sysctl_i32("hw.perflevel0.physicalcpu")
            .or_else(|| macos_sysctl_i32("hw.physicalcpu"))
            .map(|n| n as usize)
            .filter(|n| *n > 0);
        cores.map_or(available, |n| {
            Some(available.map_or(n, |max| n.min(max).max(1)))
        })
    }
    #[cfg(target_os = "linux")]
    {
        let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        let mut seen = std::collections::HashSet::new();
        let (mut pkg, mut core) = (None::<String>, None::<String>);
        for line in info.lines().chain(std::iter::once("")) {
            if line.trim().is_empty() {
                if let (Some(p), Some(c)) = (pkg.take(), core.take()) {
                    seen.insert((p, c));
                }
                continue;
            }
            let mut kv = line.splitn(2, ':');
            let (k, v) = (
                kv.next().unwrap_or("").trim(),
                kv.next().unwrap_or("").trim().to_string(),
            );
            match k {
                "physical id" => pkg = Some(v),
                "core id" => core = Some(v),
                _ => {}
            }
        }
        if seen.is_empty() {
            return std::thread::available_parallelism().ok().map(|n| n.get());
        }
        Some(seen.len())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        std::thread::available_parallelism().ok().map(|n| n.get())
    }
}

/// One integer `sysctl` value, e.g. `hw.perflevel0.physicalcpu`.
///
/// `libSystem`'s `sysctlbyname` is not in the standard library, and adding a crate for one call
/// is not worth a dependency (ADR 0002).
#[cfg(target_os = "macos")]
fn macos_sysctl_i32(name: &str) -> Option<i32> {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_void};
    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> i32;
    }
    let name = CString::new(name).ok()?;
    let mut value: i32 = 0;
    let mut len = size_of::<i32>();
    // SAFETY: `name` is a live NUL-terminated string; `value` and `len` are a live `i32` and its
    // byte length, and the call writes at most `len` bytes into `value`. No pointer outlives the
    // call.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut i32).cast::<c_void>(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len as usize == size_of::<i32>()).then_some(value)
}

/// Contiguous share of `total` items belonging to worker `w` of `n`.
///
/// Balanced to within one item, and every index belongs to exactly one worker —
/// which is the precondition the disjoint writes in `matmul` rely on.
#[inline]
pub fn share(total: usize, w: usize, n: usize) -> (usize, usize) {
    let lo = total * w / n;
    let hi = total * (w + 1) / n;
    (lo, hi)
}

/// The width of the narrower pool for one-row steps, given the explicit setting and the cores.
///
/// Split out from [`narrow_pool`] so the precedence is testable by argument, not by the process's
/// environment: an explicit `PEREGRINE_DECODE_THREADS` (or `--threads` at the call sites that
/// use it) always wins over the physical-core default.
fn narrow_width(
    decode_threads: Option<usize>,
    threads: usize,
    cores: Option<usize>,
) -> Option<usize> {
    let want = decode_threads.or(cores)?;
    (want < threads).then_some(want)
}

/// The pool for one-row steps (single-user decode), when it should be narrower than the main one.
///
/// A one-row step is bound by memory, and workers beyond the physical cores spin on SMT siblings of the cores doing
/// the work, taking their cycles: measured on the 9B (Ryzen AI 9 HX 370), one user decodes at 24.0-24.5 tok/s with
/// a 12-thread pool against 21.6-22.6 with 16, while ten users run faster with 16. Two pools, so the idle one parks
/// (after its spin window) instead of spinning beside the other. `PEREGRINE_DECODE_THREADS` sets the width
/// (default: the physical cores); `None` when it would not be narrower than `threads`.
pub fn narrow_pool(threads: usize) -> Option<Pool> {
    let want = narrow_width(
        crate::options::get().decode_threads,
        threads,
        physical_cores(),
    )?;
    Some(Pool::new(want))
}

#[cfg(test)]
mod tests {

    /// A pool shared by several engines is sound only while one caller is inside
    /// `run`. A second caller must be stopped, and the pool must keep working.
    #[test]
    fn a_second_caller_is_refused_and_the_pool_survives() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::Arc;
        let pool = Arc::new(Pool::new(2));
        let inside = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let first = {
            let (pool, inside, release) = (pool.clone(), inside.clone(), release.clone());
            std::thread::spawn(move || {
                pool.run(|_, _| {
                    inside.store(true, Ordering::SeqCst);
                    while !release.load(Ordering::SeqCst) {
                        std::thread::yield_now();
                    }
                });
            })
        };
        while !inside.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        let ran = AtomicUsize::new(0);
        let second = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(|_, _| {
                ran.fetch_add(1, Ordering::SeqCst);
            });
        }));
        assert!(
            second.is_err(),
            "a second caller must not get into the pool"
        );
        assert_eq!(ran.load(Ordering::SeqCst), 0, "and must not run anything");
        release.store(true, Ordering::SeqCst);
        first.join().expect("the first caller finishes normally");
        // The refusal left the pool usable.
        pool.run(|_, _| {
            ran.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(ran.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_caller_panic_waits_for_the_workers_and_the_pool_survives() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let pool = Pool::new(4);
        let touched = AtomicUsize::new(0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(|w, _| {
                if w == 0 {
                    panic!("caller share");
                }
                // Workers outlive the caller's panic; their captured state must too.
                std::thread::sleep(Duration::from_millis(5));
                touched.fetch_add(1, Ordering::SeqCst);
            })
        }));
        assert!(r.is_err(), "the caller's panic was swallowed");
        assert_eq!(
            touched.load(Ordering::SeqCst),
            3,
            "run returned before the workers finished"
        );
        // The next job starts from a clean count.
        let n = AtomicUsize::new(0);
        pool.run(|_, _| {
            n.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(n.load(Ordering::SeqCst), 4);
    }
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn the_cpu_map_groups_the_smt_siblings() {
        let Some(map) = physical_cpu_map() else {
            return; // no /proc/cpuinfo topology here
        };
        assert!(!map.is_empty());
        assert_eq!(
            Some(map.len()),
            physical_cores(),
            "the map and the core count must agree"
        );
        let flat: Vec<usize> = map.iter().flatten().copied().collect();
        let mut unique = flat.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            flat.len(),
            unique.len(),
            "a logical CPU is listed under two cores"
        );
        assert!(map.iter().all(|core| !core.is_empty()));
    }

    /// On Apple Silicon the default is the performance-core count: a plausible number of cores,
    /// never more than this process may run on (`hw.perflevel0.physicalcpu` can name the host's
    /// cores when a VM caps the runner).
    #[cfg(target_os = "macos")]
    #[test]
    fn the_performance_core_count_is_a_plausible_default() {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        let cores = physical_cores().expect("macOS reports a core count");
        eprintln!(
            "physical_cores (performance cores) = {cores}, available_parallelism = {available}"
        );
        assert!(
            (1..=available).contains(&cores),
            "performance cores {cores} is not between 1 and the {available} CPUs this process may use"
        );
    }

    /// An explicit width always beats the physical-core default, and a default that would not
    /// narrow the pool gives no second pool at all.
    #[test]
    fn an_explicit_narrow_width_wins_over_the_physical_cores() {
        assert_eq!(narrow_width(Some(2), 8, Some(6)), Some(2));
        assert_eq!(narrow_width(Some(7), 8, Some(6)), Some(7));
        assert_eq!(narrow_width(None, 8, Some(6)), Some(6));
        assert_eq!(narrow_width(None, 4, Some(6)), None);
        assert_eq!(narrow_width(None, 8, None), None);
    }

    #[test]
    fn shares_partition_exactly_and_evenly() {
        for total in [0usize, 1, 7, 960, 2560, 49152] {
            for n in 1..=8 {
                let mut covered = vec![0u8; total];
                let mut sizes = Vec::new();
                for w in 0..n {
                    let (lo, hi) = share(total, w, n);
                    assert!(lo <= hi && hi <= total);
                    sizes.push(hi - lo);
                    for c in covered.iter_mut().take(hi).skip(lo) {
                        *c += 1;
                    }
                }
                assert!(
                    covered.iter().all(|&c| c == 1),
                    "total={total} n={n}: not a partition"
                );
                let (min, max) = (
                    sizes.iter().min().copied().unwrap(),
                    sizes.iter().max().copied().unwrap(),
                );
                assert!(max - min <= 1, "total={total} n={n}: unbalanced {sizes:?}");
            }
        }
    }

    #[test]
    fn every_worker_runs_exactly_once_per_call() {
        for n in [1usize, 2, 4, 8] {
            let p = Pool::new(n);
            assert_eq!(p.width(), n);
            for _ in 0..50 {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                p.run(|w, nw| {
                    assert_eq!(nw, n, "worker told the wrong width");
                    hits[w].fetch_add(1, Ordering::SeqCst);
                });
                for (w, h) in hits.iter().enumerate() {
                    assert_eq!(h.load(Ordering::SeqCst), 1, "n={n} worker {w}");
                }
            }
        }
    }

    /// The property the raw-pointer writes in `matmul` depend on: workers only ever
    /// touch their own share, so concurrent writes never overlap.
    #[test]
    fn disjoint_shares_allow_concurrent_writes() {
        let n = 4;
        let p = Pool::new(n);
        let total = 10_000usize;
        let out: Vec<AtomicUsize> = (0..total).map(|_| AtomicUsize::new(0)).collect();
        p.run(|w, nw| {
            let (lo, hi) = share(total, w, nw);
            for (i, o) in out.iter().enumerate().take(hi).skip(lo) {
                o.store(i + 1, Ordering::Relaxed);
            }
        });
        for (i, o) in out.iter().enumerate() {
            assert_eq!(o.load(Ordering::Relaxed), i + 1);
        }
    }

    /// A panic on a spawned worker must surface on the caller, not hang it.
    #[test]
    fn a_worker_panic_is_raised_on_the_caller_and_the_pool_survives() {
        let p = Pool::new(3);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            p.run(|w, _| {
                if w == 2 {
                    panic!("boom");
                }
            })
        }));
        assert!(r.is_err(), "the worker's panic must reach the caller");
        let c = AtomicUsize::new(0);
        p.run(|_, _| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(
            c.load(Ordering::Relaxed),
            3,
            "the pool must still work afterwards"
        );
    }

    /// Workers park after the spin window and must still wake for the next job —
    /// the lost-wake-up case the publish-then-notify-under-the-lock order exists for.
    #[test]
    fn parked_workers_wake_for_the_next_job() {
        let p = Pool::new(4);
        for _ in 0..5 {
            std::thread::sleep(Duration::from_millis(3)); // well past SPIN: all parked
            let c = AtomicUsize::new(0);
            p.run(|_, _| {
                c.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(c.load(Ordering::Relaxed), 4);
        }
    }

    #[test]
    fn a_pool_can_be_reused_and_dropped_cleanly() {
        let p = Pool::new(4);
        let c = AtomicUsize::new(0);
        for _ in 0..1000 {
            p.run(|_, _| {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(c.load(Ordering::Relaxed), 4000);
        drop(p); // must join every worker rather than hang
    }

    /// Feed `n` samples `step` apart, starting at `t`.
    fn feed(w: &WaitControl, mut t: Instant, delay: u32, n: usize, step: Duration) -> Instant {
        for _ in 0..n {
            w.observe(delay, t);
            t += step;
        }
        t
    }

    /// Feed `n` joins `step` apart, starting at `t`.
    fn joins(w: &WaitControl, mut t: Instant, wait_ns: u64, n: usize, step: Duration) -> Instant {
        for _ in 0..n {
            w.decide(wait_ns, t);
            t += step;
        }
        t
    }

    /// Run the hold and the two probe windows, returning the time after them.
    fn probe_once(
        w: &WaitControl,
        mut t: Instant,
        incumbent_ns: u64,
        challenger_ns: u64,
    ) -> Instant {
        t = joins(
            w,
            t,
            incumbent_ns,
            INITIAL_HOLD_JOINS as usize + 1,
            Duration::from_millis(1),
        );
        // First probe window measures the incumbent, then switches to the challenger.
        t = joins(w, t, incumbent_ns, 24, Duration::from_millis(10));
        // Second window measures the challenger and decides.
        joins(w, t, challenger_ns, 24, Duration::from_millis(10))
    }

    #[test]
    fn a_quiet_host_keeps_the_configured_window() {
        let w = WaitControl::new(1000, 0, true);
        let t = feed(&w, Instant::now(), 0, 50, Duration::from_millis(10));
        assert!(!w.wants_sample());
        joins(&w, t, 10_000, 100, Duration::from_millis(10));
        assert_eq!(w.budgets().0, Duration::from_micros(1000));
    }

    #[test]
    fn a_busy_host_tries_both_rungs_and_keeps_the_faster_one() {
        // Two per core: the gate opens (smoothed above 150) but the starting rung is
        // the configured window. If parking makes the join wait longer, it goes back.
        let w = WaitControl::new(1000, 0, true);
        let mut t = feed(&w, Instant::now(), 530, 200, Duration::from_millis(10));
        assert!(w.wants_sample());
        t = joins(&w, t, 1000, 1, Duration::from_millis(1));
        assert_eq!(w.budgets().0, Duration::from_micros(1000));
        t = probe_once(&w, t, 1000, 5000);
        assert_eq!(
            w.budgets().0,
            Duration::from_micros(1000),
            "parking was slower, so the window stays"
        );
        let _ = t;

        // Three per core: the gate is deep, so the trial starts parked, and a window
        // whose wait is shorter than the parked one keeps it there.
        let w = WaitControl::new(1000, 0, true);
        let mut t = feed(&w, Instant::now(), 750, 200, Duration::from_millis(10));
        t = joins(&w, t, 5000, 1, Duration::from_millis(1));
        assert_eq!(w.budgets().0, Duration::ZERO);
        // First window (rung 2, parked) is slow in absolute terms, the challenger
        // (rung 0, spinning) is slower still: parking wins and is held.
        let _ = probe_once(&w, t, 5000, 10_000);
        assert_eq!(w.budgets().0, Duration::ZERO);
    }

    #[test]
    fn the_trials_stop_when_the_host_goes_quiet() {
        let w = WaitControl::new(1000, 0, true);
        let t = feed(&w, Instant::now(), 750, 200, Duration::from_millis(10));
        joins(&w, t, 5000, 1, Duration::from_millis(1));
        assert_eq!(w.budgets().0, Duration::ZERO);
        // The hogs stop: the gate decays and the configured window comes back.
        let t = feed(&w, t, 0, 500, Duration::from_millis(10));
        joins(&w, t, 10, 10, Duration::from_millis(10));
        assert_eq!(w.budgets().0, Duration::from_micros(1000));
        assert!(!w.wants_sample());
    }

    #[test]
    fn adaptation_can_be_turned_off() {
        let w = WaitControl::new(1000, 0, false);
        let t = Instant::now();
        for _ in 0..20 {
            w.observe(1000, t);
        }
        assert_eq!(w.budgets().0, Duration::from_micros(1000));
    }

    #[test]
    fn the_yield_deadline_is_never_before_the_spin_deadline() {
        // Defaults: YIELD_US is 0, which means "no yield phase", not "yield backwards".
        let w = WaitControl::new(1000, 0, true);
        assert_eq!(
            w.budgets(),
            (Duration::from_micros(1000), Duration::from_micros(1000))
        );
        // The rung below the configured window parks: the yield deadline is zero too.
        let t = feed(
            &w,
            Instant::now(),
            LOAD_BUSY_PERMILLE + 100,
            200,
            Duration::from_millis(10),
        );
        w.decide(0, t);
        assert_eq!(w.budgets(), (Duration::ZERO, Duration::ZERO));
        // A configured yield window still applies at full spin.
        let w = WaitControl::new(1000, 2000, true);
        assert_eq!(
            w.budgets(),
            (Duration::from_micros(1000), Duration::from_micros(2000))
        );
    }

    #[test]
    fn the_schedstat_parser_reads_the_run_queue_delay() {
        assert_eq!(parse_run_delay("794161 123456 1\n"), Some(123_456));
        assert_eq!(parse_run_delay("794161 0 1"), Some(0));
        assert_eq!(parse_run_delay(""), None);
        assert_eq!(parse_run_delay("1"), None);
        assert_eq!(parse_run_delay("1 x 2"), None);
    }

    /// The PR body's quiet-host neutrality check: 200 000 fork–joins on four workers,
    /// with the pool in whatever the environment selected. Run it on a quiet host with
    /// and without `PEREGRINE_ADAPTIVE_SPIN=0`:
    ///
    /// ```text
    /// cargo test -p peregrine-core --release --lib join_microbenchmark -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "microbenchmark: run with --ignored --nocapture"]
    fn join_microbenchmark() {
        let workers = 4;
        let joins = 200_000u64;
        let p = Pool::new(workers);
        for _ in 0..2_000 {
            p.run(|w, n| {
                std::hint::black_box((w + n) as u64);
            });
        }
        let t = Instant::now();
        for i in 0..joins {
            p.run(|w, n| {
                std::hint::black_box(i.wrapping_add((w + n) as u64));
            });
        }
        let ns = t.elapsed().as_nanos() as f64 / joins as f64;
        let us = ns / 1000.0;
        eprintln!(
            "join_microbenchmark: {joins} joins on {workers} workers, adaptive={}, {us:.3} us/join ({ns:.1} ns/join)",
            crate::options::get().adaptive_spin
        );
        assert!(us < 100.0, "{us} us/join is not a fork–join");
    }

    /// The turn is first come, first served: a caller that asked first runs first, even
    /// when a later caller is ready, and two callers never hold it at once.
    #[test]
    fn passes_take_turns_in_the_order_they_asked() {
        let turn = Arc::new(PassTurn::new(3));
        turn.enter(0);
        turn.enter(1);
        turn.enter(2);
        let order = Arc::new(Mutex::new(Vec::new()));
        let held = turn.acquire(0);
        let spawn = |n: usize| {
            let turn = turn.clone();
            let order = order.clone();
            std::thread::spawn(move || {
                let _guard = turn.acquire(n);
                order.lock().unwrap_or_else(|e| e.into_inner()).push(n);
            })
        };
        let first = spawn(1);
        wait_until(|| turn.waiting() == 1);
        let second = spawn(2);
        wait_until(|| turn.waiting() == 2);
        assert_eq!(
            turn.waiting(),
            2,
            "the second caller did not queue behind the first"
        );
        drop(held);
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(
            *order.lock().unwrap_or_else(|e| e.into_inner()),
            vec![1, 2],
            "the turns were not served in arrival order"
        );
        assert_eq!(turn.max_holders(), 1, "two passes held the turn at once");
        assert_eq!(turn.passes(), 3);
        assert_eq!(turn.waiting(), 0);
        turn.leave(0);
        turn.leave(1);
        turn.leave(2);
    }

    /// A participant that keeps asking cannot starve one that asks once: the waiting
    /// caller is served after at most the pass in flight.
    #[test]
    fn a_busy_participant_does_not_starve_another() {
        let turn = Arc::new(PassTurn::new(2));
        turn.enter(0);
        let hot = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let t = turn.clone();
        let h = hot.clone();
        let s = stop.clone();
        let worker = std::thread::spawn(move || {
            while !s.load(Ordering::SeqCst) {
                let _guard = t.acquire(0);
                h.fetch_add(1, Ordering::SeqCst);
            }
        });
        // Let the busy side take many turns, then ask once. The count starts once the
        // second participant has entered: the passes the busy side ran before that were
        // not taken from anyone, and on a loaded machine this thread can be off the CPU
        // for thousands of them. From `enter` on, the busy side can finish the pass in
        // flight and take at most one more before the turn is the newcomer's.
        wait_until(|| hot.load(Ordering::SeqCst) >= 100);
        turn.enter(1);
        let before = hot.load(Ordering::SeqCst);
        let _guard = turn.acquire(1);
        let after = hot.load(Ordering::SeqCst);
        assert!(
            after.saturating_sub(before) <= 4,
            "the waiting participant was passed over {} times",
            after - before
        );
        drop(_guard);
        turn.leave(1);
        stop.store(true, Ordering::SeqCst);
        worker.join().unwrap();
        turn.leave(0);
        assert_eq!(turn.max_holders(), 1);
    }

    /// Poll a condition from a test thread, with a deadline so a bug fails instead of hanging.
    fn wait_until(mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(Instant::now() < deadline, "the condition never became true");
            std::thread::yield_now();
        }
    }
}
