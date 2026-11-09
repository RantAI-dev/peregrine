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
/// accumulated with the runnable wall time since its last sample. If a third or more
/// of that time was spent waiting for a CPU, the host has more runnable threads than
/// cores and the window is halved. Half is where two runnable threads per core want
/// it; whether the queue is actually three deep is then measured from the bottom rung
/// (a spinning thread's own samples cannot tell two from three, because the spin is
/// itself part of the queue), and if it is, the window drops to zero and waiters park.
/// Consecutive quiet samples widen it again one step at a time, so a transient
/// neighbour costs a few tokens and a host that becomes quiet gets the spinning back.
/// Nothing here changes what a worker computes.
///
/// The default windows and their measurements are in `options.rs` (`SPIN_DEFAULT_US`).
///
/// The spin and yield windows. `PEREGRINE_SPIN_US` / `PEREGRINE_YIELD_US`
/// override them (0 and 0 park at once, which is the original condvar pool), and
/// `PEREGRINE_ADAPTIVE_SPIN=0` keeps them fixed even on a busy host, so the
/// trade-off can be measured on a given machine instead of assumed.
///
/// # The ladder
///
/// `level` 0 is the configured window. Level 1 is half of it: the right compromise
/// when the host has more runnable threads than cores but not twice as many, where a
/// waiter that spins long enough still catches a straggler that was descheduled for a
/// moment, without taking a whole core from it. Level 2 parks at once, like
/// `PEREGRINE_SPIN_US=0`, which is what heavy oversubscription wants. A busy host
/// moves down the ladder; a quiet one moves up one level per [`QUIET_SAMPLES`] samples.
const LOAD_SAMPLE: Duration = Duration::from_millis(8);

/// A sample needs at least this much runnable time behind it to be evidence: a
/// sample taken across a park (an idle session) would otherwise be mostly denominator.
const LOAD_MIN_RUNNABLE: Duration = Duration::from_millis(4);

/// Run-queue delay that means the host has more runnable threads than free cores.
///
/// Measured on the reference host (Xeon E5-2680 v4, four threads pinned to four cores):
/// 0.2-0.6 % quiet, 40-53 % with four CPU hogs on the same four cores (two runnable
/// threads per core) and 61-100 % with eight (three per core). At two per core the
/// pool gains from a shorter window (half of 1 ms beat both 1 ms and 0 on this host);
/// at three it wants to park at once. The first step needs [`BUSY_SAMPLES`] consecutive
/// samples so a single preemption does not move the ladder.
const LOAD_BUSY_PERMILLE: u32 = 350;

/// At the bottom rung, delay below this means the host has eased to two-per-core or
/// better, so the window can widen to half. Measured 27-42 % at two per core while
/// parked and 46-50 % at three, so the rung stays put at three.
const LOAD_EASED_PERMILLE: u32 = 445;

/// Below this share the host is quiet again (0.2-0.6 % measured on a quiet host).
const LOAD_LOW_PERMILLE: u32 = 50;

/// Consecutive busy samples before the first step down.
const BUSY_SAMPLES: u32 = 2;

/// Consecutive samples before the window widens one step (about 24 ms).
const QUIET_SAMPLES: u32 = 3;

/// How long level 1 runs before it spends one sample window at level 2 to find out
/// whether the host has got worse.
///
/// Level 1's own samples cannot answer that: a thread at the middle rung spends much of
/// its runnable time spinning, so two threads per core and three both measure 80-100 %.
/// One window at level 2 measures the queue itself (20-36 % at two per core, 46-50 % at
/// three on the reference host), and then the ladder decides. The probe costs about
/// 24 ms of parked waiting per cycle, and only while the host is busy.
const PROBE_INTERVAL: Duration = Duration::from_millis(250);

/// The spin window, and how much of it the pool believes it can afford.
struct WaitControl {
    /// The windows from `options.rs`, never exceeded.
    max_spin_us: u64,
    max_yield_us: u64,
    /// 0 = configured, 1 = half, 2 = park at once. Kept as one value rather than
    /// two durations so a waiter cannot observe a half-updated pair.
    level: AtomicU8,
    state: Mutex<AdaptiveState>,
}

struct AdaptiveState {
    level: u8,
    busy: u32,
    quiet: u32,
    /// When the current dwell at level 1 began, or when the last probe ended.
    dwell: Option<Instant>,
    enabled: bool,
}

impl WaitControl {
    fn new(spin_us: u64, yield_us: u64, enabled: bool) -> WaitControl {
        WaitControl {
            max_spin_us: spin_us,
            max_yield_us: yield_us,
            level: AtomicU8::new(0),
            state: Mutex::new(AdaptiveState {
                level: 0,
                busy: 0,
                quiet: 0,
                dwell: None,
                enabled,
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
            1 => self.max_spin_us / 2,
            _ => 0,
        };
        let yld = self.max_yield_us.max(spin);
        (Duration::from_micros(spin), Duration::from_micros(yld))
    }

    /// Move down the ladder on a busy host, up on a quiet one.
    ///
    /// The caller holds no reference to the sampler: this is a function of the observed
    /// share of runnable time spent in the run queue, and it is the only state the
    /// waiting threads share. Down-steps are immediate; up-steps need a streak, so a
    /// transient neighbour does not make the window flap. Level 1 cannot classify the
    /// host from its own samples (spinning dominates them), so it periodically steps to
    /// level 2 for one measurement and lets that decide.
    fn observe(&self, delay_permille: u32, now: Instant) {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !g.enabled {
            return;
        }
        let mut next = g.level;
        let mut busy = g.busy;
        let mut quiet = g.quiet;
        match g.level {
            0 => {
                if delay_permille >= LOAD_BUSY_PERMILLE {
                    busy += 1;
                    if busy >= BUSY_SAMPLES {
                        next = 1;
                    }
                } else {
                    busy = 0;
                }
            }
            1 => {
                if delay_permille <= LOAD_LOW_PERMILLE {
                    quiet += 1;
                    if quiet >= QUIET_SAMPLES {
                        next = 0;
                    }
                } else {
                    quiet = 0;
                    if g.dwell
                        .is_none_or(|t| now.saturating_duration_since(t) >= PROBE_INTERVAL)
                    {
                        next = 2;
                    }
                }
            }
            _ => {
                if delay_permille <= LOAD_EASED_PERMILLE {
                    quiet += 1;
                    if quiet >= QUIET_SAMPLES {
                        next = 1;
                    }
                } else {
                    quiet = 0;
                }
            }
        }
        if next != g.level {
            g.level = next;
            busy = 0;
            quiet = 0;
            g.dwell = (next == 1).then_some(now);
            self.level.store(next, Ordering::Relaxed);
        }
        g.busy = busy;
        g.quiet = quiet;
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
        let runnable = wall.saturating_sub(std::mem::take(&mut self.parked));
        self.last = Some((now, after));
        if runnable < LOAD_MIN_RUNNABLE {
            return;
        }
        let delayed = after.saturating_sub(before);
        let permille = (u128::from(delayed) * 1000 / runnable.as_nanos().max(1)) as u32;
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
        let o = crate::options::get();
        let inner = Arc::new(Inner {
            epoch: Aligned(AtomicU64::new(0)),
            job: UnsafeCell::new(None),
            remaining: Aligned(AtomicUsize::new(0)),
            shutdown: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            caller_sleeping: AtomicBool::new(false),
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
        let job = Job {
            data: &f as *const F as *const (),
            call: trampoline::<F>,
        };
        // SAFETY: no worker is running a job (the previous call waited for
        // `remaining == 0`), and none reads `job` until it sees the new epoch.
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

/// Contiguous share of `total` items belonging to worker `w` of `n`.
///
/// Balanced to within one item, and every index belongs to exactly one worker —
/// which is the precondition the disjoint writes in `matmul` rely on.
#[inline]
/// physical cores (distinct (package, core id) pairs in /proc/cpuinfo), when the OS says.
pub fn physical_cores() -> Option<usize> {
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
    (!seen.is_empty()).then_some(seen.len())
}

/// the pool for one-row steps (single-user decode), when it should be narrower than the main one.
///
/// A one-row step is bound by memory, and workers beyond the physical cores spin on SMT siblings of the cores doing
/// the work, taking their cycles: measured on the 9B (Ryzen AI 9 HX 370), one user decodes at 24.0-24.5 tok/s with
/// a 12-thread pool against 21.6-22.6 with 16, while ten users run faster with 16. Two pools, so the idle one parks
/// (after its spin window) instead of spinning beside the other. `PEREGRINE_DECODE_THREADS` sets the width
/// (default: the physical cores); `None` when it would not be narrower than `threads`.
pub fn narrow_pool(threads: usize) -> Option<Pool> {
    let want = crate::options::get()
        .decode_threads
        .or_else(physical_cores)?;
    (want < threads).then(|| Pool::new(want))
}

pub fn share(total: usize, w: usize, n: usize) -> (usize, usize) {
    let lo = total * w / n;
    let hi = total * (w + 1) / n;
    (lo, hi)
}

#[cfg(test)]
mod tests {

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

    #[test]
    fn a_quiet_host_keeps_the_configured_window() {
        let w = WaitControl::new(1000, 0, true);
        let t = Instant::now();
        for _ in 0..20 {
            w.observe(0, t);
        }
        assert_eq!(w.budgets().0, Duration::from_micros(1000));
    }

    #[test]
    fn an_oversubscribed_host_steps_down_to_no_spin() {
        let w = WaitControl::new(1000, 0, true);
        let t = Instant::now();
        w.observe(LOAD_BUSY_PERMILLE, t);
        assert_eq!(
            w.budgets().0,
            Duration::from_micros(1000),
            "one busy sample is not evidence yet"
        );
        w.observe(LOAD_BUSY_PERMILLE, t);
        assert_eq!(
            w.budgets().0,
            Duration::from_micros(500),
            "two per core takes the window to half"
        );
        // Before the probe is due, level 1's own (spin-dominated) samples change nothing.
        w.observe(1000, t + Duration::from_millis(100));
        assert_eq!(w.budgets().0, Duration::from_micros(500));
        // A parking window then measures the queue, and three per core keeps it parked.
        w.observe(LOAD_EASED_PERMILLE + 1, t + PROBE_INTERVAL);
        assert_eq!(w.budgets().0, Duration::ZERO);
        w.observe(1000, t + PROBE_INTERVAL);
        assert_eq!(w.budgets().0, Duration::ZERO, "the ladder stops at zero");
    }

    #[test]
    fn the_window_widens_again_after_a_quiet_streak() {
        let w = WaitControl::new(1000, 0, true);
        let t = Instant::now();
        // Two busy samples take it to half; the probe parks it.
        w.observe(LOAD_BUSY_PERMILLE, t);
        w.observe(LOAD_BUSY_PERMILLE, t);
        w.observe(1000, t + PROBE_INTERVAL);
        assert_eq!(w.budgets().0, Duration::ZERO);

        // Two eased samples are not enough...
        w.observe(LOAD_EASED_PERMILLE - 1, t + PROBE_INTERVAL);
        w.observe(LOAD_EASED_PERMILLE - 1, t + PROBE_INTERVAL);
        assert_eq!(w.budgets().0, Duration::ZERO);
        // ...the third widens one step...
        w.observe(LOAD_EASED_PERMILLE - 1, t + PROBE_INTERVAL);
        assert_eq!(w.budgets().0, Duration::from_micros(500));
        // ...and a busy sample in between restarts the quiet streak.
        w.observe(LOAD_EASED_PERMILLE + 1, t + PROBE_INTERVAL * 2);
        assert_eq!(w.budgets().0, Duration::ZERO);
        w.observe(0, t + PROBE_INTERVAL * 2);
        w.observe(0, t + PROBE_INTERVAL * 2);
        assert_eq!(w.budgets().0, Duration::ZERO);
        w.observe(0, t + PROBE_INTERVAL * 2);
        assert_eq!(w.budgets().0, Duration::from_micros(500));
        // Level 1 needs a genuinely quiet host to return to the configured window.
        for _ in 0..3 {
            w.observe(LOAD_LOW_PERMILLE, t + PROBE_INTERVAL * 2);
        }
        assert_eq!(w.budgets().0, Duration::from_micros(1000));
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
        let t = Instant::now();
        w.observe(LOAD_BUSY_PERMILLE, t);
        w.observe(LOAD_BUSY_PERMILLE, t);
        assert_eq!(
            w.budgets(),
            (Duration::from_micros(500), Duration::from_micros(500))
        );
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
}
