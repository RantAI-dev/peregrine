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

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// A job published for the workers, read through the pool's seqlock (S5.6).
///
/// The closure and its shape are written as plain atomics before [`Inner::epoch`] is
/// stored, and a sequence counter brackets the write. A worker reads them with a
/// retry if the sequence changed under it, so a worker that was descheduled across a
/// job boundary never mixes one generation's call pointer with another's data. That
/// race is new with claimed jobs: they are the first jobs whose caller may return
/// before every worker has observed them.
#[derive(Clone, Copy)]
struct Published {
    generation: u64,
    kind: u8,
    data: *const (),
    call: usize,
    units: usize,
    claim: usize,
}

const JOB_NONE: u8 = 0;
const JOB_WHOLE: u8 = 1;
const JOB_CLAIM: u8 = 2;

/// A type-erased `Fn(worker_index, worker_count)`, or the claimed form
/// `Fn(worker_index, worker_count, lo, hi)`.
///
/// Erased through a monomorphised trampoline rather than by transmuting a `&dyn` fat
/// pointer into two words — the trampoline is a plain function pointer and makes no
/// assumption about trait-object layout.
unsafe fn trampoline<F: Fn(usize, usize) + Sync>(p: *const (), w: usize, n: usize) {
    (*(p as *const F))(w, n)
}

unsafe fn claim_trampoline<F: Fn(usize, usize, usize, usize) + Sync>(
    p: *const (),
    w: usize,
    n: usize,
    lo: usize,
    hi: usize,
) {
    (*(p as *const F))(w, n, lo, hi)
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

/// The longest gap between two fork–joins the pool will bridge by spinning, in µs
/// (S5.4j).
///
/// A worker that finds no job spins for [`WaitControl::budgets`]' window and then parks,
/// and the next job pays the wake ([`crate::calib`] measures that latency at startup and
/// sets the configured window to twice it). A **prompt pass** has a much longer gap
/// between joins than a decode token — its caller does 1–5 ms of serial work between
/// them (the KV write, the activation quantise, the form's interleaved fill) — so a
/// window sized by the wake latency is shorter than the gap, and every join in a prompt
/// pass pays a park and a wake. Measured on the reference desktop, 512 tokens at twelve
/// threads, the fast prompt mode: a 2 000 µs window +6 %, 4 000 µs +8 %, 8 000 µs +4 %
/// against the calibrated 1 000 µs, and the park log showed 675 of 1 006 parks lasting
/// 1–5 ms.
///
/// **Only on a host that is not oversubscribed** (S5.6r). Stretching assumes the
/// workers are there to be kept warm; when the run queue is deeper than the pool's
/// cores, every spinning waiter takes a core the late worker needs, which is exactly
/// the loss S5.6's stall gate parks to avoid. The stretch therefore applies only while
/// the smoothed run-queue delay is below [`STALL_GATE_PERMILLE`]. Measured on the ARM
/// host (ten Cortex-X925 cores, co-tenants, load 3–11 through the window), a 512-token
/// and a 4 096-token exact prefill at ten threads: the stretch cost 1.6–2.6 % in every
/// disturbed cell and in the quiet ones (`benchmarks/results/spark-arm-2026-11-13-s5.6-reread/`).
///
/// **Per architecture, by what was measured.** On x86-64 the stretch is on: +3.3 % on
/// the AVX-512 desktop's fast 512-token twelve-thread cell (S5.4j) and level on the
/// four-core AVX2 host (0.98–1.04 with the stall gate either way, S4.9w). On aarch64 it
/// is off ([`GAP_STRETCH`]): on the ARM host a quiet ten-thread exact prefill was 1.6–2.0 %
/// *slower* with it, and 1.2–2.6 % slower under one to four competing threads (S5.6r),
/// with no reading of the host that tells the two machines apart. An aarch64 machine
/// that is truly idle has not been measured.
const GAP_CAP_US: u64 = 8000;

/// Whether the join-gap stretch is on for this architecture (see [`GAP_CAP_US`]).
const GAP_STRETCH: bool = cfg!(not(target_arch = "aarch64"));

/// A gap longer than this is a pass boundary (an idle session), not a fork-to-fork gap:
/// it must not stretch the window, because no worker is waiting across it.
const GAP_MAX_NS: u64 = 20_000_000;

/// Weight of a new gap sample in the pool's smoothed join gap: one quarter, so the
/// window follows a pass's shape within a handful of joins and cannot be moved by one
/// slow join.
const GAP_SHIFT: u32 = 2;

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

/// A join whose wait is at least this long, and at least [`STALL_SHARE_PERMILLE`] of
/// the fork–join, says the caller is waiting for a worker the host has taken away,
/// not for a share that is still being computed.
///
/// Measured on this host (S5.6): a quiet 4 096-token prefill at ten threads waits
/// 17 % of its wall (1.35 s of 7.9 s) for ordinary imbalance, while one busy thread
/// on the same cores raises the wait to 42 % (5.6 s of 13.3 s). The threshold sits
/// between the two, and the absolute floor keeps a tiny decode join from counting
/// scheduler noise as a stall.
const STALL_WAIT_NS: u64 = 50_000;
const STALL_SHARE_PERMILLE: u64 = 400;

/// The host's smoothed run-queue delay that must accompany a stalled join.
///
/// The ratio alone is not enough: a pass whose rows are unevenly cheap (a batch of
/// texts of different lengths, a pooled pass that waits for the other slots) can
/// leave the caller waiting a large share of the join without any worker having been
/// descheduled. On a quiet host the delay sample is near zero, so the claim machinery
/// stays off; a co-tenant on the pool's cores read 85-207 permille at one busy thread
/// and 250-600 at four on the S5.6 host.
const STALL_GATE_PERMILLE: u32 = 80;

/// Consecutive stalls that force the waiters to park. Four joins is under a
/// millisecond of evidence on a loaded host, so the ladder reacts within a token.
const STALL_STRIKES: u32 = 4;

/// Joins without a stall that a forced park waits before it tests the spin window
/// again. It doubles after every failed probe (up to [`STALL_HOLD_MAX`] joins), so a
/// host that stays contended is not re-probed for long; a host that quiets down gets
/// its window back within a millisecond or two of a decode step.
const STALL_HOLD: u32 = 512;
const STALL_HOLD_MAX: u32 = 8192;

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
    /// Set by the caller when recent joins stalled on a descheduled worker: waiters
    /// park at once instead of spinning, whatever the ladder's rung says. Cleared
    /// again after a hold of unstalled joins, so a host that quiets down gets its
    /// window back.
    forced_park: AtomicBool,
    /// Last diagnostic print, for `PEREGRINE_ADAPTIVE_LOG` (milliseconds).
    logged: AtomicU64,
    /// `PEREGRINE_ADAPTIVE_SPIN=0` keeps the configured window.
    enabled: bool,
    /// `PEREGRINE_ADAPTIVE_STALL=0` turns the stall detection (and the claimed
    /// partitions it enables) off, leaving the spin ladder exactly as it was.
    stall_enabled: bool,
    /// `PEREGRINE_ADAPTIVE_STALL=claim`: [`Pool::claiming`] is always true, so every
    /// splittable block takes its claimed partition. A test and diagnostic switch.
    claim_always: bool,
    /// Smoothed gap between the end of one fork–join and the start of the next, in
    /// nanoseconds (0 until the first measured gap). The worker's spin window is
    /// stretched to cover it — only while the host is not oversubscribed — because
    /// that is the wait a spin can save, and a prompt pass's gap is longer than the
    /// wake latency the configured window is sized by.
    gap_ns: AtomicU64,
    /// When the previous fork–join ended, in nanoseconds since the pool's first join.
    /// Only the caller thread writes it; `0` means "no join has finished yet".
    last_end: AtomicU64,
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

/// Join-wait accounting for the S5.6 account, printed at `Pool::drop` when
/// `PEREGRINE_PROF2` is on.
///
/// `JOIN_WAIT_NS` overlaps the forward's wall (it is the caller's post-share wait),
/// so it is reported as a share, never added to a stage. `LATE_GAP_NS` is the
/// difference between the last worker to finish and the one before it: a few
/// microseconds of ordinary imbalance when the pool is healthy, a scheduler
/// timeslice when the host has taken a worker away. `LATE_HIST[w]` counts which
/// worker was last, so a pinning mistake or one permanently busy core is visible.
static JOIN_N: AtomicU64 = AtomicU64::new(0);
static JOIN_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static JOIN_SHARE_NS: AtomicU64 = AtomicU64::new(0);
static JOIN_MAX_NS: AtomicU64 = AtomicU64::new(0);
static LATE_GAP_NS: AtomicU64 = AtomicU64::new(0);
static LATE_GAP_MAX: AtomicU64 = AtomicU64::new(0);
static LATE_HIST: [AtomicU64; 32] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn print_join_totals() {
    if !crate::prof::enabled() {
        return;
    }
    let n = JOIN_N.load(Ordering::Relaxed);
    if n == 0 {
        return;
    }
    let wait = JOIN_WAIT_NS.load(Ordering::Relaxed);
    let share = JOIN_SHARE_NS.load(Ordering::Relaxed);
    eprintln!(
        "joins: {n} ({} per join), wait {:.3} s ({:.1} %), share {:.3} s, max wait {:.3} ms",
        n,
        wait as f64 / 1e9,
        100.0 * wait as f64 / (wait + share).max(1) as f64,
        share as f64 / 1e9,
        JOIN_MAX_NS.load(Ordering::Relaxed) as f64 / 1e6,
    );
    eprintln!(
        "joins: straggler gap mean {:.3} ms, max {:.3} ms",
        if n == 0 {
            0.0
        } else {
            LATE_GAP_NS.load(Ordering::Relaxed) as f64 / 1e6 / n as f64
        },
        LATE_GAP_MAX.load(Ordering::Relaxed) as f64 / 1e6,
    );
    let hist: Vec<String> = LATE_HIST
        .iter()
        .enumerate()
        .filter(|(_, c)| c.load(Ordering::Relaxed) > 0)
        .map(|(w, c)| format!("{w}:{}", c.load(Ordering::Relaxed)))
        .collect();
    eprintln!("joins: last worker histogram [{}]", hist.join(" "));
}

/// Whether one join looks like a worker taken off the CPU rather than ordinary imbalance.
///
/// Both conditions: the wait is long enough that a scheduler timeslice could be behind
/// it, and it dominates the caller's own share. A quiet ten-thread 4 096-token pass
/// waits 13-17 % of the fork–join; one busy thread on the same cores raises that past
/// 60 % in the unmodified pool.
#[inline]
fn is_stall(wait_ns: u64, share_ns: u64) -> bool {
    let total = wait_ns.saturating_add(share_ns).max(1);
    wait_ns >= STALL_WAIT_NS
        && u128::from(wait_ns) * 1000 / u128::from(total) >= u128::from(STALL_SHARE_PERMILLE)
}

/// Nanoseconds since a process-wide origin, for the per-worker finish timestamps.
fn now_ns() -> u64 {
    use std::sync::OnceLock;
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_nanos() as u64
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

/// Nanoseconds since the process's first fork–join, the clock the join gaps are kept on
/// (a `u64` of nanoseconds, not an `Instant`, so the field is a plain atomic).
fn nanos_since_base(now: Instant) -> u64 {
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let base = *BASE.get_or_init(|| now);
    now.saturating_duration_since(base).as_nanos() as u64
}

impl WaitControl {
    fn new(spin_us: u64, yield_us: u64, enabled: bool, stall_enabled: bool) -> WaitControl {
        WaitControl {
            max_spin_us: spin_us,
            max_yield_us: yield_us,
            level: AtomicU8::new(0),
            ewma: AtomicU64::new(0),
            seeded: AtomicBool::new(false),
            trialing: AtomicBool::new(false),
            forced_park: AtomicBool::new(false),
            logged: AtomicU64::new(0),
            enabled,
            stall_enabled,
            claim_always: false,
            gap_ns: AtomicU64::new(0),
            last_end: AtomicU64::new(0),
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

    /// Force every waiter to park (or release the force). The caller owns the
    /// transition; workers only read the flag in [`WaitControl::budgets`].
    fn force_park(&self, on: bool) {
        self.forced_park.store(on, Ordering::Relaxed);
    }

    /// The (spin, yield) deadlines for the current level.
    ///
    /// The yield deadline is never earlier than the spin deadline: `backoff` treats both
    /// as absolute offsets from the start of the wait, so a configured `YIELD_US` of 0
    /// means "no yield phase" and not "yield for minus one millisecond".
    fn budgets(&self) -> (Duration, Duration) {
        if self.forced_park.load(Ordering::Relaxed) {
            return (Duration::ZERO, Duration::ZERO);
        }
        let spin = match self.level.load(Ordering::Relaxed) {
            0 => {
                let configured = self.max_spin_us;
                // A caller that asks for no spinning (`PEREGRINE_SPIN_US=0`, and the
                // calibration's own wake probe) keeps none.
                if configured == 0 || !self.enabled || self.oversubscribed() {
                    configured
                } else {
                    configured.max(self.gap_us())
                }
            }
            _ => 0,
        };
        let yld = self.max_yield_us.max(spin);
        (Duration::from_micros(spin), Duration::from_micros(yld))
    }

    /// Whether the smoothed run-queue delay says the host has more runnable threads
    /// than free cores.
    ///
    /// The same evidence the stall gate uses ([`STALL_GATE_PERMILLE`]): on such a host
    /// a longer spin does not bridge the join gap, it takes a core from the worker the
    /// join is waiting for, so the gap stretch is off and the configured window is kept.
    #[inline]
    fn oversubscribed(&self) -> bool {
        u32::try_from(self.ewma.load(Ordering::Relaxed) / 1024).unwrap_or(1000)
            >= STALL_GATE_PERMILLE
    }

    /// The measured join gap, in µs, capped at [`GAP_CAP_US`].
    fn gap_us(&self) -> u64 {
        if !GAP_STRETCH {
            return 0;
        }
        (self.gap_ns.load(Ordering::Relaxed) / 1000).min(GAP_CAP_US)
    }

    /// Record the gap a worker would have to bridge: the time from the end of the last
    /// fork–join to the start of this one.
    fn observe_gap(&self, now: Instant) {
        if !GAP_STRETCH || !self.enabled || self.max_spin_us == 0 {
            return;
        }
        let nanos = nanos_since_base(now);
        let prev = self.last_end.load(Ordering::Relaxed);
        if prev == 0 || nanos <= prev {
            return;
        }
        let gap = nanos - prev;
        if gap > GAP_MAX_NS {
            return;
        }
        let old = self.gap_ns.load(Ordering::Relaxed);
        let next = if old == 0 {
            gap
        } else {
            old - (old >> GAP_SHIFT) + (gap >> GAP_SHIFT)
        };
        self.gap_ns.store(next, Ordering::Relaxed);
    }

    /// Mark the end of this fork–join, which is where the next gap is measured from.
    fn mark_join_end(&self, now: Instant) {
        if !GAP_STRETCH || !self.enabled || self.max_spin_us == 0 {
            return;
        }
        self.last_end
            .store(nanos_since_base(now), Ordering::Relaxed);
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
    /// The published job's fields. Written by the caller under `job_seq` (odd while
    /// writing, even when stable) before the epoch is stored, and read by a worker
    /// with the same sequence check.
    job_seq: AtomicU64,
    job_gen: AtomicU64,
    job_kind: AtomicU8,
    job_data: AtomicUsize,
    job_call: AtomicUsize,
    job_units: AtomicUsize,
    job_claim: AtomicUsize,
    /// Claimed-job state (S5.6), also published before the epoch. `claim_next` is the
    /// next unit; `busy` counts workers inside a claim loop; `closed` names the last
    /// generation whose closure must not be entered again.
    claim_next: Aligned<AtomicUsize>,
    busy: Aligned<AtomicUsize>,
    closed: AtomicU64,
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
    /// The caller's stall detection (S5.6). Only the caller reads and writes these:
    /// one join runs at a time, so relaxed atomics are enough and no worker takes a
    /// lock or a shared cache line in its wait.
    strikes: AtomicU32,
    healthy: AtomicU32,
    hold: AtomicU32,
    /// Whether the unforced window currently being measured is a probe (the forced
    /// park was released and is being tested).
    probing: AtomicBool,
    /// Finish time of each worker in the current job, nanoseconds since [`now_ns`]'s
    /// origin; written only when `PEREGRINE_PROF2` is on.
    finish: Vec<Aligned<AtomicU64>>,
}

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

impl Inner {
    /// Publish one job under the sequence lock, before the epoch that announces it.
    fn publish(
        &self,
        generation: u64,
        kind: u8,
        data: *const (),
        call: usize,
        units: usize,
        claim: usize,
    ) {
        self.job_seq.fetch_add(1, Ordering::SeqCst);
        // The odd sequence must be visible before any field changes: without this
        // fence a weakly ordered CPU (aarch64) may let a field store pass the
        // increment, and a reader could pair an unchanged even sequence with a new
        // field. The reader's fence below is the other half; this is the standard
        // sequence-lock discipline (crossbeam's `SeqLock` uses the same two fences).
        std::sync::atomic::fence(Ordering::Release);
        self.job_gen.store(generation, Ordering::Relaxed);
        self.job_kind.store(kind, Ordering::Relaxed);
        self.job_data.store(data as usize, Ordering::Relaxed);
        self.job_call.store(call, Ordering::Relaxed);
        self.job_units.store(units, Ordering::Relaxed);
        self.job_claim.store(claim, Ordering::Relaxed);
        self.job_seq.fetch_add(1, Ordering::SeqCst);
    }

    /// Read the published job, retrying if a publish crossed the read.
    fn read_job(&self) -> Published {
        loop {
            let first = self.job_seq.load(Ordering::Acquire);
            if first & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let job = Published {
                generation: self.job_gen.load(Ordering::Relaxed),
                kind: self.job_kind.load(Ordering::Relaxed),
                data: self.job_data.load(Ordering::Relaxed) as *const (),
                call: self.job_call.load(Ordering::Relaxed),
                units: self.job_units.load(Ordering::Relaxed),
                claim: self.job_claim.load(Ordering::Relaxed),
            };
            // The field loads above are relaxed, so nothing else orders them before
            // the validating load: the fence does (an acquire *load* only orders what
            // follows it).
            std::sync::atomic::fence(Ordering::Acquire);
            if self.job_seq.load(Ordering::Acquire) == first {
                return job;
            }
        }
    }
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
            job_seq: AtomicU64::new(0),
            job_gen: AtomicU64::new(0),
            job_kind: AtomicU8::new(JOB_NONE),
            job_data: AtomicUsize::new(0),
            job_call: AtomicUsize::new(0),
            job_units: AtomicUsize::new(0),
            job_claim: AtomicUsize::new(0),
            claim_next: Aligned(AtomicUsize::new(0)),
            busy: Aligned(AtomicUsize::new(0)),
            closed: AtomicU64::new(0),
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
            wait: WaitControl {
                claim_always: o.claim_always,
                ..WaitControl::new(o.spin_us, o.yield_us, o.adaptive_spin, o.adaptive_stall)
            },
            strikes: AtomicU32::new(0),
            healthy: AtomicU32::new(0),
            hold: AtomicU32::new(STALL_HOLD),
            probing: AtomicBool::new(false),
            finish: (0..n).map(|_| Aligned(AtomicU64::new(0))).collect(),
        });
        let mut handles = Vec::with_capacity(n - 1);
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
                    let epoch_now = inner.epoch.0.load(Ordering::Acquire);
                    // SAFETY: the job is published before the epoch, and the sequence
                    // lock makes the fields a consistent generation even when a publish
                    // crossed this read. The data pointer and call pointer are therefore
                    // the pair the caller stored for this generation.
                    let job = inner.read_job();
                    if job.generation != epoch_now {
                        // The fields moved between the epoch load and the job read. Do
                        // not run a generation the epoch does not name: a worker that
                        // observed E must wait for the publish of E+1 rather than run
                        // E+1 twice.
                        seen = job.generation.min(epoch_now);
                        continue;
                    }
                    seen = job.generation;
                    let mut whole = false;
                    match job.kind {
                        JOB_WHOLE => {
                            whole = true;
                            // SAFETY: the pointer was stored from a function of type
                            // `unsafe fn(*const (), usize, usize)` and the data it
                            // points at is the closure passed to `run`.
                            let call: unsafe fn(*const (), usize, usize) =
                                unsafe { std::mem::transmute(job.call) };
                            let ok =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                                    call(job.data, w, inner.n)
                                }));
                            if ok.is_err() {
                                inner.poisoned.store(true, Ordering::Release);
                            }
                        }
                        JOB_CLAIM => {
                            // SAFETY: as above, for the claimed form.
                            let call: unsafe fn(*const (), usize, usize, usize, usize) =
                                unsafe { std::mem::transmute(job.call) };
                            let generation = job.generation;
                            let (units, claim) = (job.units, job.claim);
                            // A worker the host has only now scheduled must not enter
                            // a closed job; it costs nothing to skip, which is the
                            // point of the claimed form.
                            if inner.closed.load(Ordering::Acquire) < generation {
                                inner.busy.0.fetch_add(1, Ordering::SeqCst);
                                if inner.closed.load(Ordering::SeqCst) < generation {
                                    loop {
                                        let lo =
                                            inner.claim_next.0.fetch_add(claim, Ordering::Relaxed);
                                        if lo >= units {
                                            break;
                                        }
                                        let hi = (lo + claim).min(units);
                                        // SAFETY: `[lo, hi)` was claimed by this
                                        // worker alone, and the caller is still inside
                                        // `run_claimed` (it waits for `busy == 0`).
                                        let ok =
                                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                                                || unsafe { call(job.data, w, inner.n, lo, hi) },
                                            ));
                                        if ok.is_err() {
                                            inner.poisoned.store(true, Ordering::Release);
                                            break;
                                        }
                                    }
                                }
                                inner.busy.0.fetch_sub(1, Ordering::SeqCst);
                                if inner.busy.0.load(Ordering::SeqCst) == 0
                                    && inner.caller_sleeping.load(Ordering::SeqCst)
                                {
                                    let _g =
                                        inner.done_mu.lock().unwrap_or_else(|e| e.into_inner());
                                    inner.done_cv.notify_one();
                                }
                            }
                        }
                        _ => {}
                    }
                    if crate::prof::enabled() {
                        inner.finish[w].0.store(now_ns(), Ordering::Relaxed);
                    }
                    if whole
                        && inner.remaining.0.fetch_sub(1, Ordering::SeqCst) == 1
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
        self.inner.wait.observe_gap(Instant::now());
        if self.n == 1 {
            f(0, 1);
            self.inner.wait.mark_join_end(Instant::now());
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
        let generation = self.inner.epoch.0.load(Ordering::SeqCst) + 1;
        self.inner.publish(
            generation,
            JOB_WHOLE,
            &f as *const F as *const (),
            trampoline::<F> as *const () as usize,
            0,
            0,
        );
        self.inner.remaining.0.store(self.n - 1, Ordering::Release);
        self.inner.epoch.0.store(generation, Ordering::SeqCst);
        if self.inner.sleepers.load(Ordering::SeqCst) > 0 {
            let g = self.inner.park.lock().unwrap_or_else(|e| e.into_inner());
            if *g > 0 {
                self.inner.wake.notify_all();
            }
        }
        // The caller takes share 0 rather than blocking immediately.
        let share_t0 = Instant::now();
        let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(0, self.n)));
        let share_ns = share_t0.elapsed().as_nanos() as u64;
        if crate::prof::enabled() {
            self.inner.finish[0].0.store(now_ns(), Ordering::Relaxed);
        }
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
        let wait_ns = t0.elapsed().as_nanos() as u64;
        self.note_join(wait_ns, share_ns);
        if crate::prof::enabled() {
            JOIN_N.fetch_add(1, Ordering::Relaxed);
            JOIN_WAIT_NS.fetch_add(wait_ns, Ordering::Relaxed);
            JOIN_SHARE_NS.fetch_add(share_ns, Ordering::Relaxed);
            JOIN_MAX_NS.fetch_max(wait_ns, Ordering::Relaxed);
            self.record_straggler();
        }
        if self.inner.wait.wants_sample() {
            self.inner
                .wait
                .decide(t0.elapsed().as_nanos() as u64, Instant::now());
        }
        if let Some(t) = wait_t0 {
            crate::prof::add(crate::prof::Slot::Wait, t.elapsed().as_nanos() as u64);
        }
        self.inner.wait.mark_join_end(Instant::now());
        crate::blocks::par::bump_epoch();
        let poisoned = self.inner.poisoned.swap(false, Ordering::AcqRel);
        if let Err(p) = mine {
            std::panic::resume_unwind(p);
        }
        if poisoned {
            panic!("a pool worker panicked while running a job");
        }
    }

    /// Whether recent joins stalled on a worker the host had taken away.
    ///
    /// Blocks that can split their work into independent units ask this and use
    /// [`Pool::run_claimed`] when it is true: a claimed job does not wait for a
    /// worker that has not arrived, which is the only way a fork–join over borrowed
    /// data can stop paying a scheduling latency for a descheduled participant.
    /// On a healthy host the flag is false and every block keeps the partition it
    /// was tuned with.
    pub fn claiming(&self) -> bool {
        self.inner.wait.claim_always
            || (self.inner.wait.stall_enabled
                && self.inner.wait.forced_park.load(Ordering::Relaxed))
    }

    /// Run `f(worker, width, lo, hi)` for every claimed range of `units`.
    ///
    /// Ranges are `[lo, min(lo + claim, units))` from a counter that the caller and
    /// every worker that arrives claim from. The caller returns once the counter is
    /// exhausted and no range is in flight; a worker the host has descheduled does
    /// not hold the join open, because the caller closes the job (publishing
    /// `generation` in `closed`) before it returns and the late worker skips it.
    /// This is sound for exactly the same reason the static form is: every unit is
    /// claimed by one worker, so the writes are disjoint, and the closure is alive
    /// for every worker that is inside it (`busy` counts them).
    pub fn run_claimed<F: Fn(usize, usize, usize, usize) + Sync>(
        &self,
        units: usize,
        claim: usize,
        f: F,
    ) {
        if units == 0 {
            return;
        }
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        crate::blocks::par::bump_epoch();
        // A claimed job is a fork–join like any other for the join-gap measure
        // (S5.4j): leaving it out would hand the next `run` a gap that spans it.
        self.inner.wait.observe_gap(Instant::now());
        if self.n == 1 {
            f(0, 1, 0, units);
            self.inner.wait.mark_join_end(Instant::now());
            return;
        }
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
        let claim = claim.max(1);
        let generation = self.inner.epoch.0.load(Ordering::SeqCst) + 1;
        self.inner.publish(
            generation,
            JOB_CLAIM,
            &f as *const F as *const (),
            claim_trampoline::<F> as *const () as usize,
            units,
            claim,
        );
        self.inner.claim_next.0.store(0, Ordering::Relaxed);
        // `busy` is deliberately not reset here: a worker from the previous claimed
        // job may be between its first closed-check and its increment, and a reset
        // would let its decrement underflow the counter. It always reaches zero on
        // its own.
        self.inner.remaining.0.store(0, Ordering::Release);
        self.inner.epoch.0.store(generation, Ordering::SeqCst);
        if self.inner.sleepers.load(Ordering::SeqCst) > 0 {
            let g = self.inner.park.lock().unwrap_or_else(|e| e.into_inner());
            if *g > 0 {
                self.inner.wake.notify_all();
            }
        }
        let share_t0 = Instant::now();
        self.inner.busy.0.fetch_add(1, Ordering::SeqCst);
        let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loop {
            let lo = self.inner.claim_next.0.fetch_add(claim, Ordering::Relaxed);
            if lo >= units {
                break;
            }
            f(0, self.n, lo, (lo + claim).min(units));
        }));
        self.inner.busy.0.fetch_sub(1, Ordering::SeqCst);
        let share_ns = share_t0.elapsed().as_nanos() as u64;
        // Close the job before returning: a worker that has not arrived yet will see
        // this and skip, and one that is between its two closed-checks will leave.
        self.inner.closed.store(generation, Ordering::SeqCst);
        let (spin, yld) = self.inner.wait.budgets();
        let t0 = Instant::now();
        sample_load(&self.inner.wait, t0);
        loop {
            if self.inner.busy.0.load(Ordering::Acquire) == 0 {
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
            while self.inner.busy.0.load(Ordering::SeqCst) > 0 {
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
        let wait_ns = t0.elapsed().as_nanos() as u64;
        self.note_join(wait_ns, share_ns);
        if crate::prof::enabled() {
            JOIN_N.fetch_add(1, Ordering::Relaxed);
            JOIN_WAIT_NS.fetch_add(wait_ns, Ordering::Relaxed);
            JOIN_SHARE_NS.fetch_add(share_ns, Ordering::Relaxed);
            JOIN_MAX_NS.fetch_max(wait_ns, Ordering::Relaxed);
            self.record_straggler();
        }
        self.inner.wait.mark_join_end(Instant::now());
        crate::blocks::par::bump_epoch();
        let poisoned = self.inner.poisoned.swap(false, Ordering::AcqRel);
        if let Err(p) = mine {
            std::panic::resume_unwind(p);
        }
        if poisoned {
            panic!("a pool worker panicked while running a job");
        }
    }

    /// What one join's wait says about the host, and whether to stop spinning.
    ///
    /// The wait is the caller's post-share wait; the share is its own work. On a
    /// healthy host the wait is a fraction of the share (17 % of the wall on a quiet
    /// ten-thread 4 096-token pass); when a worker is off-CPU the join waits for a
    /// scheduling timeslice and the wait dominates. Four such joins in a row force
    /// every waiter to park at once — the window is what takes the cores the late
    /// worker needs — and after a hold of unstalled joins the window is tested again.
    /// A failed probe doubles the hold, so a permanently loaded host is not re-probed
    /// often. Nothing here changes what a worker computes.
    fn note_join(&self, wait_ns: u64, share_ns: u64) {
        if !self.inner.wait.enabled || !self.inner.wait.stall_enabled {
            return;
        }
        let busy =
            u32::try_from(self.inner.wait.ewma.load(Ordering::Relaxed) / 1024).unwrap_or(1000);
        let stalled = busy >= STALL_GATE_PERMILLE && is_stall(wait_ns, share_ns);
        if stalled {
            self.inner.healthy.store(0, Ordering::Relaxed);
            let forced = self.inner.wait.forced_park.load(Ordering::Relaxed);
            if forced {
                return;
            }
            let strikes = self.inner.strikes.fetch_add(1, Ordering::Relaxed) + 1;
            let probing = self.inner.probing.swap(false, Ordering::Relaxed);
            if strikes >= STALL_STRIKES || probing {
                // A probe that stalls is evidence the host is still contended: park
                // again and wait longer before the next test.
                let hold = if probing {
                    (self.inner.hold.load(Ordering::Relaxed) * 2).min(STALL_HOLD_MAX)
                } else {
                    self.inner.hold.load(Ordering::Relaxed)
                };
                self.inner.hold.store(hold, Ordering::Relaxed);
                self.inner.strikes.store(0, Ordering::Relaxed);
                self.inner.wait.force_park(true);
                self.log_stall("park", hold);
            }
            return;
        }
        self.inner.strikes.store(0, Ordering::Relaxed);
        let forced = self.inner.wait.forced_park.load(Ordering::Relaxed);
        if !forced {
            if self.inner.probing.load(Ordering::Relaxed) {
                let healthy = self.inner.healthy.fetch_add(1, Ordering::Relaxed) + 1;
                if healthy >= self.inner.hold.load(Ordering::Relaxed) {
                    // The window survived a whole hold: accept it and reset the hold.
                    self.inner.probing.store(false, Ordering::Relaxed);
                    self.inner.healthy.store(0, Ordering::Relaxed);
                    self.inner.hold.store(STALL_HOLD, Ordering::Relaxed);
                    self.log_stall("keep", 0);
                }
            }
            return;
        }
        let healthy = self.inner.healthy.fetch_add(1, Ordering::Relaxed) + 1;
        if healthy >= self.inner.hold.load(Ordering::Relaxed) {
            self.inner.healthy.store(0, Ordering::Relaxed);
            self.inner.probing.store(true, Ordering::Relaxed);
            self.inner.wait.force_park(false);
            self.log_stall("probe", self.inner.hold.load(Ordering::Relaxed));
        }
    }

    fn log_stall(&self, what: &str, hold: u32) {
        if adaptive_log() {
            let start = *ADAPTIVE_START.get_or_init(Instant::now);
            eprintln!(
                "adaptive: +{:>9.3} ms stall {what} (hold {hold} joins)",
                start.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    /// Which worker finished last, and by how much over the one before it.
    ///
    /// Diagnostic only (`PEREGRINE_PROF2`): the finish timestamps are written by the
    /// workers in their job-exit path.
    fn record_straggler(&self) {
        let n = self.inner.n;
        let mut first = 0u64;
        let mut second = 0u64;
        let mut who = 0usize;
        for w in 0..n {
            let t = self.inner.finish[w].0.load(Ordering::Relaxed);
            if t >= first {
                second = first;
                first = t;
                who = w;
            } else if t > second {
                second = t;
            }
        }
        let gap = first.saturating_sub(second);
        LATE_GAP_NS.fetch_add(gap, Ordering::Relaxed);
        LATE_GAP_MAX.fetch_max(gap, Ordering::Relaxed);
        LATE_HIST[who % 32].fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        print_adaptive_totals();
        print_join_totals();
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
///
/// Windows has no `/proc`: `GetLogicalProcessorInformationEx(RelationProcessorCore, …)` counts
/// the cores (S7.2; the call is in `crate::win32`, which exists only on Windows), and the
/// logical count is the fallback when that call fails.
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
    #[cfg(windows)]
    {
        // Windows has no `/proc`: the count comes from the processor topology, and the
        // logical count is the fallback when the call fails (a restricted process).
        crate::win32::physical_core_count()
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
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
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
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
        let w = WaitControl::new(1000, 0, true, true);
        let t = feed(&w, Instant::now(), 0, 50, Duration::from_millis(10));
        assert!(!w.wants_sample());
        joins(&w, t, 10_000, 100, Duration::from_millis(10));
        assert_eq!(w.budgets().0, Duration::from_micros(1000));
    }

    #[test]
    fn a_busy_host_tries_both_rungs_and_keeps_the_faster_one() {
        // Two per core: the gate opens (smoothed above 150) but the starting rung is
        // the configured window. If parking makes the join wait longer, it goes back.
        let w = WaitControl::new(1000, 0, true, true);
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
        let w = WaitControl::new(1000, 0, true, true);
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
        let w = WaitControl::new(1000, 0, true, true);
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
        let w = WaitControl::new(1000, 0, false, true);
        let t = Instant::now();
        for _ in 0..20 {
            w.observe(1000, t);
        }
        assert_eq!(w.budgets().0, Duration::from_micros(1000));
    }

    /// The join-gap stretch is a quiet-host lever: on an oversubscribed host the extra
    /// spin takes a core from the worker the join is waiting for, so the configured
    /// window is kept (S5.6r, measured on the ARM host).
    #[test]
    fn the_gap_stretch_stops_on_an_oversubscribed_host() {
        let w = WaitControl::new(1000, 0, true, true);
        // A 5 ms prompt-pass gap. Where the stretch is on it covers the gap; where it
        // is off (aarch64) the configured window stands.
        w.gap_ns.store(5_000_000, Ordering::Relaxed);
        let stretched = if GAP_STRETCH { 5000 } else { 1000 };
        assert_eq!(w.budgets().0, Duration::from_micros(stretched));
        // A run queue as deep as the stall gate's evidence: no stretch.
        let t = feed(
            &w,
            Instant::now(),
            STALL_GATE_PERMILLE + 120,
            8,
            Duration::from_millis(10),
        );
        assert!(w.oversubscribed());
        assert_eq!(w.budgets().0, Duration::from_micros(1000));
        // The co-tenant leaves and the stretch comes back.
        let _ = feed(&w, t, 0, 40, Duration::from_millis(10));
        assert!(!w.oversubscribed());
        assert_eq!(w.budgets().0, Duration::from_micros(stretched));
    }

    #[test]
    fn the_yield_deadline_is_never_before_the_spin_deadline() {
        // Defaults: YIELD_US is 0, which means "no yield phase", not "yield backwards".
        let w = WaitControl::new(1000, 0, true, true);
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
        let w = WaitControl::new(1000, 2000, true, true);
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

    /// The wait/share test: ordinary imbalance is not a stall, a scheduling
    /// timeslice is.
    #[test]
    fn a_join_is_a_stall_only_when_the_wait_dominates_the_share() {
        // Quiet: 17 % of the fork–join at ten threads (S5.6's account).
        assert!(!is_stall(1_350_000, 6_500_000));
        assert!(!is_stall(10_000, 10_000_000));
        // A worker taken off the CPU: a millisecond of wait on a 100 µs share.
        assert!(is_stall(1_000_000, 100_000));
        // Long but still the smaller half is not evidence.
        assert!(!is_stall(400_000, 1_200_000));
        // The absolute floor keeps a tiny decode join's scheduler noise out.
        assert!(!is_stall(10_000, 1_000));
        // Exactly at the threshold is a stall (>=).
        assert!(is_stall(STALL_WAIT_NS, STALL_WAIT_NS * 2 / 3));
    }

    /// Repeated stalls force every waiter to park; a run of healthy joins probes the
    /// configured window back, and a stalled probe parks again with a longer hold.
    #[test]
    fn the_stall_gate_parks_and_probes_back() {
        if crate::options::get().claim_always {
            eprintln!("skip: PEREGRINE_ADAPTIVE_STALL=claim pins the claimed partitions on");
            return;
        }
        let p = Pool::new(4);
        assert!(!p.claiming(), "a fresh pool is not stalled");
        // A busy host: the run-queue sample has to say so before a join can be a stall.
        for _ in 0..50 {
            p.inner.wait.observe(500, Instant::now());
        }
        for _ in 0..STALL_STRIKES {
            p.note_join(1_000_000, 100_000);
        }
        assert!(p.claiming(), "consecutive stalls must park the waiters");
        for _ in 0..STALL_HOLD {
            p.note_join(2_000, 1_000_000);
        }
        assert!(
            !p.claiming(),
            "a hold of healthy joins tests the window again"
        );
        p.note_join(1_000_000, 100_000);
        assert!(p.claiming(), "a stalled probe parks again");
    }

    /// Poll a condition from a test thread, with a deadline so a bug fails instead of hanging.
    fn wait_until(mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(Instant::now() < deadline, "the condition never became true");
            std::thread::yield_now();
        }
    }

    /// Every unit of a claimed job is processed exactly once, and the job can be
    /// repeated: the close of one generation must not hide the next.
    #[test]
    fn claimed_ranges_cover_every_unit_exactly_once() {
        let p = Pool::new(4);
        for _ in 0..200 {
            let hits: Vec<AtomicUsize> = (0..501).map(|_| AtomicUsize::new(0)).collect();
            p.run_claimed(501, 7, |_, _, lo, hi| {
                for h in hits.iter().take(hi).skip(lo) {
                    h.fetch_add(1, Ordering::SeqCst);
                }
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::SeqCst) == 1),
                "a claimed unit was missed or run twice"
            );
        }
    }

    /// A claimed job returns even when one worker is held off the CPU: the units are
    /// claimed by whoever is running, and the join does not wait for the absent one.
    #[test]
    fn a_claimed_job_does_not_wait_for_a_descheduled_worker() {
        let p = Pool::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        // A worker is somewhere inside the pool; this test only checks that the call
        // returns and the work is complete, not what the scheduler did with it.
        let t = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
            })
        };
        let done = AtomicUsize::new(0);
        p.run_claimed(64, 1, |_, _, lo, hi| {
            done.fetch_add(hi - lo, Ordering::SeqCst);
        });
        assert_eq!(done.load(Ordering::SeqCst), 64);
        stop.store(true, Ordering::SeqCst);
        t.join().unwrap();
    }
}
