//! Shared-mutable views that hand the pool's workers disjoint parts of one buffer.
//!
//! Workers of `Pool::run` write disjoint parts of shared buffers. `Disjoint<'a, T>` encapsulates
//! the shared-mutable access in safe methods with bounds-checking always.
//!
//! When overlap checking is enabled (`cfg(debug_assertions)` builds or `--cfg peregrine_check_disjoint`),
//! claims are tracked per `Pool::run` epoch in an atomic bitmap and any overlap panics immediately.
//! In release builds without checking, methods inline to direct pointer arithmetic.

use crate::attn;
use crate::pool::Pool;
use crate::quant::QuantAct;
use std::marker::PhantomData;

use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) static CURRENT_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Advance the disjoint claim epoch (called once per `Pool::run`).
#[inline]
pub(crate) fn bump_epoch() -> u64 {
    CURRENT_EPOCH.fetch_add(1, Ordering::SeqCst) + 1
}

#[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
struct BufferTracker {
    epoch: AtomicU64,
    mu: std::sync::Mutex<()>,
    words: Vec<AtomicU64>,
}

#[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
impl BufferTracker {
    fn new(len: usize) -> Self {
        let n_words = (len + 63) / 64;
        let mut words = Vec::with_capacity(n_words);
        for _ in 0..n_words {
            words.push(AtomicU64::new(0));
        }
        Self {
            epoch: AtomicU64::new(0),
            mu: std::sync::Mutex::new(()),
            words,
        }
    }

    #[inline]
    fn reset(&self) {
        let _g = self.mu.lock().unwrap();
        for w in &self.words {
            w.store(0, Ordering::Relaxed);
        }
        self.epoch
            .store(CURRENT_EPOCH.load(Ordering::Relaxed), Ordering::Release);
    }

    #[inline]
    fn ensure_epoch(&self) {
        let cur = CURRENT_EPOCH.load(Ordering::Acquire);
        if self.epoch.load(Ordering::Acquire) != cur {
            let _g = self.mu.lock().unwrap();
            if self.epoch.load(Ordering::Relaxed) != cur {
                for w in &self.words {
                    w.store(0, Ordering::Relaxed);
                }
                self.epoch.store(cur, Ordering::Release);
            }
        }
    }

    #[inline]
    fn claim_one(&self, i: usize) {
        self.ensure_epoch();
        let word_idx = i / 64;
        if word_idx >= self.words.len() {
            return;
        }
        let bit = 1u64 << (i % 64);
        let prev = self.words[word_idx].fetch_or(bit, Ordering::AcqRel);
        if (prev & bit) != 0 {
            let ep = CURRENT_EPOCH.load(Ordering::Relaxed);
            panic!(
                "Disjoint overlap detected! Element at index {} already claimed in epoch {}",
                i, ep
            );
        }
    }

    #[inline]
    fn claim_range(&self, off: usize, len: usize) {
        if len == 0 {
            return;
        }
        self.ensure_epoch();
        let end = off + len;
        let start_word = off / 64;
        let end_word = (end + 63) / 64;
        let n_words = self.words.len();
        let end_word = end_word.min(n_words);

        for w in start_word..end_word {
            let word_start = w * 64;
            let word_end = word_start + 64;
            let r_start = off.max(word_start) - word_start;
            let r_end = end.min(word_end) - word_start;
            let mask = if r_end - r_start == 64 {
                !0u64
            } else {
                let m = (1u64 << (r_end - r_start)) - 1;
                m << r_start
            };
            let prev = self.words[w].fetch_or(mask, Ordering::AcqRel);
            if (prev & mask) != 0 {
                let overlap_bit = (prev & mask).trailing_zeros() as usize;
                let overlap_idx = word_start + overlap_bit;
                let ep = CURRENT_EPOCH.load(Ordering::Relaxed);
                panic!(
                    "Disjoint overlap detected! Element at index {} in range {}..{} already claimed in epoch {}",
                    overlap_idx, off, end, ep
                );
            }
        }
    }
}

#[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
fn tracker_for(ptr: usize, len: usize) -> *const BufferTracker {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    static REGISTRY: OnceLock<Mutex<HashMap<usize, Arc<BufferTracker>>>> = OnceLock::new();
    let reg = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = reg.lock().unwrap();
    if let Some(existing) = map.get(&ptr) {
        if existing.words.len() * 64 >= len {
            return Arc::as_ptr(existing);
        }
    }
    let tracker = Arc::new(BufferTracker::new(len));
    let raw = Arc::as_ptr(&tracker);
    map.insert(ptr, tracker);
    raw
}

/// A shared-mutable view of a `&mut [T]` that worker threads can carve into disjoint chunks.
pub struct Disjoint<'a, T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    tracker: *const BufferTracker,
    _marker: PhantomData<&'a mut [T]>,
}

// SAFETY: Disjoint allows concurrent access to elements of T only at disjoint indices,
// which requires T: Send to transfer access across threads.
unsafe impl<'a, T: Send> Send for Disjoint<'a, T> {}
unsafe impl<'a, T: Send> Sync for Disjoint<'a, T> {}

impl<'a, T> Clone for Disjoint<'a, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for Disjoint<'a, T> {}

impl<'a, T> Disjoint<'a, T> {
    /// Create a new disjoint shared view over a mutable slice.
    #[inline(always)]
    pub fn new(slice: &'a mut [T]) -> Self {
        let ptr = slice.as_mut_ptr();
        let len = slice.len();
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        let tracker = {
            let t = tracker_for(ptr as usize, len);
            unsafe { (*t).reset() };
            t
        };
        Self {
            ptr,
            len,
            #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
            tracker,
            _marker: PhantomData,
        }
    }

    /// Construct a disjoint shared view from a raw pointer and length.
    ///
    /// # Safety
    /// `ptr` must be valid for `len` elements and point to memory uniquely owned for lifetime `'a`.
    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        let tracker = {
            let t = tracker_for(ptr as usize, len);
            unsafe { (*t).reset() };
            t
        };
        Self {
            ptr,
            len,
            #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
            tracker,
            _marker: PhantomData,
        }
    }

    /// The number of elements in the buffer.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Carve a contiguous mutable slice `[off..off+len]`.
    ///
    /// Bounds-checked always. Panics if overlapping an already claimed range in the current epoch.
    #[inline(always)]
    pub fn chunk(&self, off: usize, len: usize) -> &'a mut [T] {
        assert!(
            off.checked_add(len).map_or(false, |end| end <= self.len),
            "Disjoint::chunk out of bounds: off={off}, len={len}, total={}",
            self.len
        );
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        if !self.tracker.is_null() {
            // SAFETY: tracker is managed by registry and valid for program lifetime.
            unsafe { (*self.tracker).claim_range(off, len) };
        }
        // SAFETY: bounds checked above; disjoint access guaranteed by caller / checked above.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off), len) }
    }

    /// Alias for `chunk`.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn slice(&self, off: usize, len: usize) -> &'a mut [T] {
        self.chunk(off, len)
    }

    /// Carve a range of rows given row stride.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn rows(&self, range: std::ops::Range<usize>, stride: usize) -> &'a mut [T] {
        let off = range.start * stride;
        let len = (range.end - range.start) * stride;
        self.chunk(off, len)
    }

    /// Carve a single row given row stride.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn row(&self, r: usize, stride: usize) -> &'a mut [T] {
        self.chunk(r * stride, stride)
    }

    /// Write an element at index `i` (for strided outputs across batch rows).
    ///
    /// Bounds-checked always. Panics on duplicate write in the same epoch.
    #[inline(always)]
    pub fn set(&self, i: usize, v: T) {
        assert!(
            i < self.len,
            "Disjoint::set out of bounds: index {i} >= total {}",
            self.len
        );
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        if !self.tracker.is_null() {
            // SAFETY: tracker is managed by registry and valid for program lifetime.
            unsafe { (*self.tracker).claim_one(i) };
        }
        // SAFETY: bounds checked above; disjoint write guaranteed by caller / checked above.
        unsafe {
            *self.ptr.add(i) = v;
        }
    }

    /// Disjoint per-item mutable access.
    #[inline(always)]
    pub fn get_mut(&self, i: usize) -> &'a mut T {
        assert!(
            i < self.len,
            "Disjoint::get_mut out of bounds: index {i} >= total {}",
            self.len
        );
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        if !self.tracker.is_null() {
            unsafe { (*self.tracker).claim_one(i) };
        }
        // SAFETY: bounds checked above.
        unsafe { &mut *self.ptr.add(i) }
    }

    /// Alias for `get_mut` (for compatibility with previous `SharedActs::get`).
    #[inline(always)]
    pub fn get(&self, i: usize) -> &'a mut T {
        self.get_mut(i)
    }

    /// Shared read-only view of a range.
    #[inline(always)]
    pub fn slice_ref(&self, off: usize, len: usize) -> &'a [T] {
        assert!(
            off.checked_add(len).map_or(false, |end| end <= self.len),
            "Disjoint::slice_ref out of bounds: off={off}, len={len}, total={}",
            self.len
        );
        // SAFETY: bounds checked above; read-only slice.
        unsafe { std::slice::from_raw_parts(self.ptr.add(off), len) }
    }
}

// Type aliases for existing block and model interfaces.
pub(crate) type OutPtr<'a> = Disjoint<'a, f32>;
pub(crate) type SharedBuf<'a> = Disjoint<'a, f32>;
pub(crate) type SharedActs<'a> = Disjoint<'a, QuantAct>;
#[allow(dead_code)]
pub(crate) type SharedScratch<'a> = Disjoint<'a, attn::Scratch>;
#[allow(dead_code)]
pub(crate) type SharedMl<'a> = Disjoint<'a, (f32, f64)>;

/// Static partition driver: workers take equal contiguous shares of `total` rows.
#[allow(dead_code)]
#[inline]
pub(crate) fn for_rows<F>(pool: &Pool, total: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    bump_epoch();
    pool.run(|wk, nw| {
        let (lo, hi) = crate::pool::share(total, wk, nw);
        for i in lo..hi {
            f(i, wk);
        }
    });
}

/// Dynamic partition driver: workers claim dynamic chunks from an atomic counter.
#[allow(dead_code)]
#[inline]
pub(crate) fn for_rows_dynamic<F>(pool: &Pool, total: usize, chunk_size: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    bump_epoch();
    let next = AtomicUsize::new(0);
    pool.run(|wk, _| loop {
        let lo = next.fetch_add(chunk_size, Relaxed);
        if lo >= total {
            break;
        }
        let hi = (lo + chunk_size).min(total);
        for i in lo..hi {
            f(i, wk);
        }
    });
}

// ---- recurrent-state pointers of the hybrid (Gated DeltaNet) pass ----

/// Per-row raw pointers into the caches' recurrent state, and the snapshots a speculative verification asks
/// for, kept beside the other hybrid scratch so a decode step reuses them instead of allocating four vectors
/// per Gated DeltaNet layer.
pub(crate) struct RecScratch {
    /// Row `b`'s `(state, conv)` pointers for the recurrent layer being processed.
    pub(crate) rec: Vec<(*mut f32, *mut f32)>,
    /// Row `b`'s snapshot request `(position, state, conv)`, if its cache asked for one.
    pub(crate) snap: Vec<Option<(usize, *mut f32, *mut f32)>>,
}

// SAFETY: the vectors only hold pointers into the `KvCache`s passed to the same `forward_hybrid` call; they are
// rewritten by every `gdn`. `State` owns them and only one thread uses it at a time; the pool closures see copies
// through `RowPtrs`/`SnapPtrs`, whose `Send`/`Sync` impls carry the disjoint-write argument. The `Send` impl keeps
// `State` (and the `Engine` holding it) movable between threads like the rest of the scratch.
unsafe impl Send for RecScratch {}
// SAFETY: as `Send`; shared access only reads the vector's values (pointer copies) and never dereferences them.
unsafe impl Sync for RecScratch {}

/// Raw per-row state pointers for the delta-rule workers.
#[derive(Clone, Copy)]
pub(crate) struct RowPtrs(pub(crate) *const (*mut f32, *mut f32));
// SAFETY: each (row's state, head) region is written by exactly one task; rows of a shared cache are processed in
// order by the task that owns the head.
unsafe impl Send for RowPtrs {}
unsafe impl Sync for RowPtrs {}

impl RowPtrs {
    /// Row `b`'s (state, conv) pointers. A method, so a closure captures the wrapper (Send + Sync), not its field.
    ///
    /// # Safety
    /// `b` must be inside the pointer array, which must outlive the call.
    #[inline(always)]
    pub(crate) unsafe fn at(self, b: usize) -> (*mut f32, *mut f32) {
        *self.0.add(b)
    }
}

/// Per row, the snapshot request of its cache for recurrent layer `idx`: (position, state, conv) pointers.
#[derive(Clone, Copy)]
pub(crate) struct SnapPtrs(pub(crate) *const Option<(usize, *mut f32, *mut f32)>);
// SAFETY: as `RowPtrs`: each snapshot region is written by the task that owns it (a head slice, a channel range).
unsafe impl Send for SnapPtrs {}
unsafe impl Sync for SnapPtrs {}
impl SnapPtrs {
    /// # Safety
    /// `b` must be inside the array, which must outlive the call.
    #[inline(always)]
    pub(crate) unsafe fn at(self, b: usize) -> Option<(usize, *mut f32, *mut f32)> {
        *self.0.add(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disjoint_disjoint_claims() {
        let mut data = [0.0f32; 64];
        let d = Disjoint::new(&mut data);
        let s1 = d.chunk(0, 32);
        s1.fill(1.0);
        let s2 = d.chunk(32, 32);
        s2.fill(2.0);
        assert_eq!(data[0], 1.0);
        assert_eq!(data[31], 1.0);
        assert_eq!(data[32], 2.0);
        assert_eq!(data[63], 2.0);
    }

    #[test]
    fn test_disjoint_set_and_slice_ref() {
        let mut data = [0.0f32; 16];
        let d = Disjoint::new(&mut data);
        for i in 0..16 {
            d.set(i, i as f32);
        }
        let r = d.slice_ref(0, 16);
        assert_eq!(r[5], 5.0);
    }

    #[test]
    fn test_disjoint_epoch_reuse() {
        let mut data = [0.0f32; 32];
        let d = Disjoint::new(&mut data);
        let s1 = d.chunk(0, 16);
        s1.fill(1.0);
        bump_epoch();
        let s2 = d.chunk(0, 16);
        s2.fill(2.0);
        assert_eq!(data[0], 2.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_disjoint_out_of_bounds() {
        let mut data = [0.0f32; 16];
        let d = Disjoint::new(&mut data);
        let _ = d.chunk(10, 10);
    }

    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    #[test]
    #[should_panic(expected = "Disjoint overlap detected")]
    fn test_disjoint_chunk_overlap_panics() {
        bump_epoch();
        let mut data = [0.0f32; 64];
        let d = Disjoint::new(&mut data);
        let _s1 = d.chunk(0, 32);
        let _s2 = d.chunk(16, 32);
    }

    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    #[test]
    #[should_panic(expected = "Disjoint overlap detected")]
    fn test_disjoint_set_overlap_panics() {
        bump_epoch();
        let mut data = [0.0f32; 64];
        let d = Disjoint::new(&mut data);
        d.set(5, 1.0);
        d.set(5, 2.0);
    }
}
