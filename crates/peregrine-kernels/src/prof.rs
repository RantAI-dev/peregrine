//! Optional, per-thread phase timers for a forward pass.
//!
//! `PEREGRINE_PROF2=1` turns them on. Every timer is taken on the calling thread
//! (the thread that drives the pool), so the counters are thread-local `Cell`s
//! and no atomic is touched on the hot path. At the bench's scale one timer pair
//! is ~20 ns against a ~30 ms token, and the modules are compiled in always so a
//! measurement build is the published build.
//!
//! The buckets are attribution, not a partition of wall time: the matmul wall
//! already contains the caller's wait for its workers, and [`Slot::Wait`] reports
//! that wait again ("barrier wait"). A bucket therefore must never be added to
//! `ns_total` a second time.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// One measured phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Slot {
    /// Attention projections: Q, K, V together.
    Qkv = 0,
    /// The attention output projection.
    Wo,
    /// The feed-forward gate and up projections together.
    GateUp,
    /// The feed-forward down projection.
    Down,
    /// The output head (final norm is [`Slot::Norm`], the projection is [`Slot::Head`]).
    Head,
    /// Attention: scores, softmax and the value mix.
    Attn,
    /// RMSNorm itself (the sumsq and scale).
    Norm,
    /// Quantising an activation to the matmul format (`QuantAct::fill`).
    Quant,
    /// The attention prologue: biases, per-head q/k norms and RoPE.
    Rope,
    /// Writing this step's K/V rows into the cache.
    KvWrite,
    /// Residual adds and the activation copies between projections.
    Resid,
    /// SwiGLU's `silu(gate) ⊙ up`.
    Swiglu,
    /// The caller's wait for the pool in a fork–join.
    Wait,
    /// Everything else the caller did inside the forward: sampling, slot upkeep.
    Other,
    /// Attention subtimers (worker-side, so they use the shared counters).
    AttnScores,
    AttnSoftmax,
    AttnMix,
    AttnDecode,
    /// Matmul time by weight format (worker-side, attributed at the job's rows).
    MatQ5_0,
    MatQ8_0,
    MatQ6K,
    MatQ4K,
    MatQ5K,
    MatOther,
}

/// The number of slots.
pub const N: usize = 24;

thread_local! {
    static NS: [Cell<u64>; N] = const { [const { Cell::new(0) }; N] };
}

/// Counters that workers add to directly. Adding a shared counter is an atomic
/// read-modify-write, so only the attention subtimers (which run per block inside the
/// workers) use it, and only when the timers are on.
static SHARED: [AtomicU64; 10] = [
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

/// Whether the phase timers are on (`PEREGRINE_PROF2`).
pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("PEREGRINE_PROF2").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// Whether the inner-loop timers are on too (`PEREGRINE_PROF2=full` or `=2`).
///
/// These run per block inside the workers and cost real time (an `Instant` pair and an
/// atomic add per attention block). They answer "where inside attention", not "how much
/// of the token", so they are a second level.
pub fn detail() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("PEREGRINE_PROF2").is_ok_and(|v| v == "full" || v == "2"))
}

/// Add `ns` nanoseconds to `slot` on this thread. No-op unless enabled.
#[inline]
pub fn add(slot: Slot, ns: u64) {
    if enabled() {
        NS.with(|c| c[slot as usize].set(c[slot as usize].get() + ns));
    }
}

/// Time `f` into `slot`. No-op timing unless enabled.
#[inline]
pub fn time<R>(slot: Slot, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let r = f();
    add(slot, t.elapsed().as_nanos() as u64);
    r
}

/// The per-format matmul slot for a weight format.
#[inline]
pub fn format_slot(fmt: crate::format::Format) -> Slot {
    match fmt {
        #[cfg(feature = "fmt-legacy")]
        crate::format::Format::Q5_0 => Slot::MatQ5_0,
        #[cfg(feature = "fmt-legacy")]
        crate::format::Format::Q8_0 => Slot::MatQ8_0,
        #[cfg(feature = "fmt-kquant")]
        crate::format::Format::Q6_K => Slot::MatQ6K,
        #[cfg(feature = "fmt-kquant")]
        crate::format::Format::Q4_K => Slot::MatQ4K,
        #[cfg(feature = "fmt-kquant")]
        crate::format::Format::Q5_K => Slot::MatQ5K,
        _ => Slot::MatOther,
    }
}

/// Add `ns` nanoseconds to a worker-side `slot` (one of the detail subtimers).
#[inline]
pub fn add_shared(slot: Slot, ns: u64) {
    if enabled() {
        SHARED[slot as usize - Slot::AttnScores as usize].fetch_add(ns, Ordering::Relaxed);
    }
}

/// This thread's counters, in [`Slot`] order, plus the worker-side shared ones.
pub fn snapshot() -> [u64; N] {
    let mut a = NS.with(|c| std::array::from_fn::<u64, N, _>(|i| c[i].get()));
    for (i, s) in SHARED.iter().enumerate() {
        a[Slot::AttnScores as usize + i] += s.load(Ordering::Relaxed);
    }
    a
}

/// Zero this thread's counters.
pub fn reset() {
    NS.with(|c| {
        for v in c {
            v.set(0);
        }
    });
    for s in &SHARED {
        s.store(0, Ordering::Relaxed);
    }
}

/// The label of each slot, in [`Slot`] order.
pub fn names() -> [&'static str; N] {
    [
        "matmul qkv",
        "matmul wo",
        "matmul gate/up",
        "matmul down",
        "output head",
        "attention",
        "norms",
        "quantise",
        "rope+bias",
        "kv write",
        "residual",
        "swiglu",
        "barrier wait",
        "other",
        "  attn scores",
        "  attn softmax",
        "  attn mix",
        "  attn kv decode",
        "  fmt Q5_0",
        "  fmt Q8_0",
        "  fmt Q6_K",
        "  fmt Q4_K",
        "  fmt Q5_K",
        "  fmt other",
    ]
}
