//! Shared-mutable views and safe partition drivers that hand the pool's workers disjoint parts of one buffer.
//!
//! (a) Safe drivers (`for_row_chunks`, `for_row_chunks_2`, `for_row_chunks_3`):
//! Workers of `Pool::run` receive genuine, non-aliasing `&mut [T]` slices for their assigned row ranges.
//! Disjointness holds by construction: static `pool::share` or dynamic atomic counter partitions the row space
//! with no overlap. The one `unsafe` creating per-worker slices lives inside the driver in `par.rs`.
//!
//! (b) Checked raw views (`Disjoint<'a, T>`, `OutPtr<'a>`, `SharedBuf<'a>`, etc.):
//! For sites that do not fit contiguous row partitions (strided matmul outputs `b * rows + r`, attention's
//! partial `(o, m, l)` slots, recurrent-state pointers), `Disjoint` provides `unsafe fn` accessors with
//! safety comments at every call site.
//!
//! When overlap checking is enabled (`cfg(any(debug_assertions, peregrine_check_disjoint))`), claims are
//! tracked per `Pool::run` epoch in an atomic bitmap and any overlap panics immediately.
//! In release builds without checking, all tracker operations and epoch bumps compile to nothing, and
//! per-element accessors inline to direct pointer writes with zero bounds checks.

use crate::attn;
use crate::pool::Pool;
use crate::quant::QuantAct;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
use std::sync::atomic::AtomicU64;

#[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
pub(crate) static CURRENT_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Advance the disjoint claim epoch (called once per `Pool::run` or partition driver).
#[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
#[inline]
pub(crate) fn bump_epoch() -> u64 {
    CURRENT_EPOCH.fetch_add(1, Ordering::SeqCst) + 1
}

/// No-op in release builds without the overlap checker.
#[cfg(not(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint")))]
#[inline(always)]
pub(crate) fn bump_epoch() {}

#[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
struct BufferTracker {
    epoch: AtomicU64,
    mu: std::sync::Mutex<()>,
    words: Vec<AtomicU64>,
}

#[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
unsafe impl Send for BufferTracker {}
#[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
unsafe impl Sync for BufferTracker {}

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
fn tracker_for(ptr: usize, len: usize) -> &'static BufferTracker {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    static REGISTRY: OnceLock<Mutex<HashMap<usize, Arc<BufferTracker>>>> = OnceLock::new();
    let reg = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = reg.lock().unwrap();
    if let Some(existing) = map.get(&ptr) {
        if existing.words.len() * 64 >= len {
            let r: &BufferTracker = existing.as_ref();
            return unsafe { &*(r as *const BufferTracker) };
        }
    }
    let tracker = Arc::new(BufferTracker::new(len));
    let r: &BufferTracker = tracker.as_ref();
    let ptr_ref = unsafe { &*(r as *const BufferTracker) };
    map.insert(ptr, tracker);
    ptr_ref
}

/// Scheduling strategy for partitioning tasks across worker threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sched {
    /// Contiguous shares: worker `wk` of `nw` takes `pool::share(n_rows, wk, nw)`.
    Static,
    /// Contiguous block-aligned shares: workers take blocks of `align` rows via `share(n_rows / align, wk, nw)`.
    StaticAligned(usize),
    /// Dynamic chunks: workers take chunks of `chunk_size` rows from an atomic counter.
    Dynamic(usize),
}

/// Carve disjoint row ranges of `buf` across worker threads.
///
/// If `parallel` is false or `pool.width() <= 1`, runs synchronously on the calling thread as worker 0.
/// If `parallel` is true, partitions the rows across `pool` workers according to `sched`.
pub fn for_row_chunks<T, F>(
    pool: &Pool,
    parallel: bool,
    buf: &mut [T],
    stride: usize,
    n_rows: usize,
    sched: Sched,
    f: F,
) where
    T: Send,
    F: Fn(usize, Range<usize>, &mut [T]) + Sync,
{
    if n_rows == 0 {
        return;
    }
    assert!(
        n_rows.checked_mul(stride).map_or(false, |total| total <= buf.len()),
        "for_row_chunks out of bounds: n_rows={n_rows}, stride={stride}, total required={}, buf.len={}",
        n_rows * stride,
        buf.len()
    );

    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    bump_epoch();

    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    let tracker = tracker_for(buf.as_ptr() as usize, buf.len());

    let raw = buf.as_mut_ptr() as usize;

    if !parallel || pool.width() <= 1 {
        let len = n_rows * stride;
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        tracker.claim_range(0, len);
        // SAFETY: The caller guarantees exclusive access to `buf`. In serial mode, worker 0
        // is handed the entire exclusive range 0..n_rows * stride.
        let slice = unsafe { std::slice::from_raw_parts_mut(raw as *mut T, len) };
        f(0, 0..n_rows, slice);
        return;
    }

    match sched {
        Sched::Static => {
            pool.run(|wk, nw| {
                let (lo, hi) = crate::pool::share(n_rows, wk, nw);
                if lo < hi {
                    let off = lo * stride;
                    let len = (hi - lo) * stride;
                    #[cfg(any(
                        debug_assertions,
                        peregrine_check_disjoint,
                        feature = "check-disjoint"
                    ))]
                    tracker.claim_range(off, len);
                    // SAFETY: `pool::share` partitions 0..n_rows into strictly disjoint contiguous ranges.
                    // Each worker `wk` has an exclusive, non-overlapping slice `off..off+len`.
                    // The slice does not outlive this worker's execution within `pool.run`.
                    let chunk =
                        unsafe { std::slice::from_raw_parts_mut((raw as *mut T).add(off), len) };
                    f(wk, lo..hi, chunk);
                }
            });
        }
        Sched::StaticAligned(align) => {
            let align = align.max(1);
            pool.run(|wk, nw| {
                let (lo_blk, hi_blk) = crate::pool::share(n_rows / align, wk, nw);
                let lo = lo_blk * align;
                let hi = if wk + 1 == nw { n_rows } else { hi_blk * align };
                if lo < hi {
                    let off = lo * stride;
                    let len = (hi - lo) * stride;
                    #[cfg(any(
                        debug_assertions,
                        peregrine_check_disjoint,
                        feature = "check-disjoint"
                    ))]
                    tracker.claim_range(off, len);
                    // SAFETY: Block-aligned shares partition 0..n_rows into strictly disjoint ranges.
                    let chunk =
                        unsafe { std::slice::from_raw_parts_mut((raw as *mut T).add(off), len) };
                    f(wk, lo..hi, chunk);
                }
            });
        }
        Sched::Dynamic(chunk_size) => {
            let chunk_size = chunk_size.max(1);
            let next = AtomicUsize::new(0);
            pool.run(|wk, _nw| loop {
                let lo = next.fetch_add(chunk_size, Ordering::Relaxed);
                if lo >= n_rows {
                    break;
                }
                let hi = (lo + chunk_size).min(n_rows);
                let off = lo * stride;
                let len = (hi - lo) * stride;
                #[cfg(any(
                    debug_assertions,
                    peregrine_check_disjoint,
                    feature = "check-disjoint"
                ))]
                tracker.claim_range(off, len);
                // SAFETY: Atomic counter monotonically advances by `chunk_size`, so each claimed
                // range `lo..hi` is strictly disjoint from all others. The slice does not escape
                // the closure call.
                let chunk =
                    unsafe { std::slice::from_raw_parts_mut((raw as *mut T).add(off), len) };
                f(wk, lo..hi, chunk);
            });
        }
    }
}

/// Carve disjoint row ranges of two buffers across worker threads.
#[allow(clippy::too_many_arguments)]
pub fn for_row_chunks_2<T1, T2, F>(
    pool: &Pool,
    parallel: bool,
    buf1: &mut [T1],
    stride1: usize,
    buf2: &mut [T2],
    stride2: usize,
    n_rows: usize,
    sched: Sched,
    f: F,
) where
    T1: Send,
    T2: Send,
    F: Fn(usize, Range<usize>, &mut [T1], &mut [T2]) + Sync,
{
    if n_rows == 0 {
        return;
    }
    assert!(
        n_rows.checked_mul(stride1).map_or(false, |total| total <= buf1.len()),
        "for_row_chunks_2 buf1 out of bounds: n_rows={n_rows}, stride1={stride1}, required={}, buf1.len={}",
        n_rows * stride1,
        buf1.len()
    );
    assert!(
        n_rows.checked_mul(stride2).map_or(false, |total| total <= buf2.len()),
        "for_row_chunks_2 buf2 out of bounds: n_rows={n_rows}, stride2={stride2}, required={}, buf2.len={}",
        n_rows * stride2,
        buf2.len()
    );

    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    bump_epoch();

    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    let tracker1 = tracker_for(buf1.as_ptr() as usize, buf1.len());
    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    let tracker2 = tracker_for(buf2.as_ptr() as usize, buf2.len());

    let raw1 = buf1.as_mut_ptr() as usize;
    let raw2 = buf2.as_mut_ptr() as usize;

    if !parallel || pool.width() <= 1 {
        let len1 = n_rows * stride1;
        let len2 = n_rows * stride2;
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        {
            tracker1.claim_range(0, len1);
            tracker2.claim_range(0, len2);
        }
        // SAFETY: Caller guarantees exclusive access to buf1 and buf2. Single worker 0 takes entire range.
        let s1 = unsafe { std::slice::from_raw_parts_mut(raw1 as *mut T1, len1) };
        let s2 = unsafe { std::slice::from_raw_parts_mut(raw2 as *mut T2, len2) };
        f(0, 0..n_rows, s1, s2);
        return;
    }

    match sched {
        Sched::Static => {
            pool.run(|wk, nw| {
                let (lo, hi) = crate::pool::share(n_rows, wk, nw);
                if lo < hi {
                    let (off1, len1) = (lo * stride1, (hi - lo) * stride1);
                    let (off2, len2) = (lo * stride2, (hi - lo) * stride2);
                    #[cfg(any(
                        debug_assertions,
                        peregrine_check_disjoint,
                        feature = "check-disjoint"
                    ))]
                    {
                        tracker1.claim_range(off1, len1);
                        tracker2.claim_range(off2, len2);
                    }
                    // SAFETY: `pool::share` yields strictly disjoint row ranges across workers.
                    let c1 = unsafe {
                        std::slice::from_raw_parts_mut((raw1 as *mut T1).add(off1), len1)
                    };
                    let c2 = unsafe {
                        std::slice::from_raw_parts_mut((raw2 as *mut T2).add(off2), len2)
                    };
                    f(wk, lo..hi, c1, c2);
                }
            });
        }
        Sched::StaticAligned(align) => {
            let align = align.max(1);
            pool.run(|wk, nw| {
                let (lo_blk, hi_blk) = crate::pool::share(n_rows / align, wk, nw);
                let lo = lo_blk * align;
                let hi = if wk + 1 == nw { n_rows } else { hi_blk * align };
                if lo < hi {
                    let (off1, len1) = (lo * stride1, (hi - lo) * stride1);
                    let (off2, len2) = (lo * stride2, (hi - lo) * stride2);
                    #[cfg(any(
                        debug_assertions,
                        peregrine_check_disjoint,
                        feature = "check-disjoint"
                    ))]
                    {
                        tracker1.claim_range(off1, len1);
                        tracker2.claim_range(off2, len2);
                    }
                    // SAFETY: Block-aligned shares partition 0..n_rows into strictly disjoint ranges.
                    let c1 = unsafe {
                        std::slice::from_raw_parts_mut((raw1 as *mut T1).add(off1), len1)
                    };
                    let c2 = unsafe {
                        std::slice::from_raw_parts_mut((raw2 as *mut T2).add(off2), len2)
                    };
                    f(wk, lo..hi, c1, c2);
                }
            });
        }
        Sched::Dynamic(chunk_size) => {
            let chunk_size = chunk_size.max(1);
            let next = AtomicUsize::new(0);
            pool.run(|wk, _nw| loop {
                let lo = next.fetch_add(chunk_size, Ordering::Relaxed);
                if lo >= n_rows {
                    break;
                }
                let hi = (lo + chunk_size).min(n_rows);
                let (off1, len1) = (lo * stride1, (hi - lo) * stride1);
                let (off2, len2) = (lo * stride2, (hi - lo) * stride2);
                #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
                {
                    tracker1.claim_range(off1, len1);
                    tracker2.claim_range(off2, len2);
                }
                // SAFETY: Atomic counter guarantees disjoint ranges for each claim.
                let c1 =
                    unsafe { std::slice::from_raw_parts_mut((raw1 as *mut T1).add(off1), len1) };
                let c2 =
                    unsafe { std::slice::from_raw_parts_mut((raw2 as *mut T2).add(off2), len2) };
                f(wk, lo..hi, c1, c2);
            });
        }
    }
}

/// Carve disjoint row ranges of three buffers across worker threads.
#[allow(clippy::too_many_arguments)]
pub fn for_row_chunks_3<T1, T2, T3, F>(
    pool: &Pool,
    parallel: bool,
    buf1: &mut [T1],
    stride1: usize,
    buf2: &mut [T2],
    stride2: usize,
    buf3: &mut [T3],
    stride3: usize,
    n_rows: usize,
    sched: Sched,
    f: F,
) where
    T1: Send,
    T2: Send,
    T3: Send,
    F: Fn(usize, Range<usize>, &mut [T1], &mut [T2], &mut [T3]) + Sync,
{
    if n_rows == 0 {
        return;
    }
    assert!(
        n_rows.checked_mul(stride1).map_or(false, |total| total <= buf1.len()),
        "for_row_chunks_3 buf1 out of bounds: n_rows={n_rows}, stride1={stride1}, required={}, buf1.len={}",
        n_rows * stride1,
        buf1.len()
    );
    assert!(
        n_rows.checked_mul(stride2).map_or(false, |total| total <= buf2.len()),
        "for_row_chunks_3 buf2 out of bounds: n_rows={n_rows}, stride2={stride2}, required={}, buf2.len={}",
        n_rows * stride2,
        buf2.len()
    );
    assert!(
        n_rows.checked_mul(stride3).map_or(false, |total| total <= buf3.len()),
        "for_row_chunks_3 buf3 out of bounds: n_rows={n_rows}, stride3={stride3}, required={}, buf3.len={}",
        n_rows * stride3,
        buf3.len()
    );

    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    bump_epoch();

    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    let tracker1 = tracker_for(buf1.as_ptr() as usize, buf1.len());
    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    let tracker2 = tracker_for(buf2.as_ptr() as usize, buf2.len());
    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    let tracker3 = tracker_for(buf3.as_ptr() as usize, buf3.len());

    let raw1 = buf1.as_mut_ptr() as usize;
    let raw2 = buf2.as_mut_ptr() as usize;
    let raw3 = buf3.as_mut_ptr() as usize;

    if !parallel || pool.width() <= 1 {
        let len1 = n_rows * stride1;
        let len2 = n_rows * stride2;
        let len3 = n_rows * stride3;
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        {
            tracker1.claim_range(0, len1);
            tracker2.claim_range(0, len2);
            tracker3.claim_range(0, len3);
        }
        // SAFETY: Caller guarantees exclusive access to buf1, buf2, buf3. Single worker 0 takes entire range.
        let s1 = unsafe { std::slice::from_raw_parts_mut(raw1 as *mut T1, len1) };
        let s2 = unsafe { std::slice::from_raw_parts_mut(raw2 as *mut T2, len2) };
        let s3 = unsafe { std::slice::from_raw_parts_mut(raw3 as *mut T3, len3) };
        f(0, 0..n_rows, s1, s2, s3);
        return;
    }

    match sched {
        Sched::Static => {
            pool.run(|wk, nw| {
                let (lo, hi) = crate::pool::share(n_rows, wk, nw);
                if lo < hi {
                    let (off1, len1) = (lo * stride1, (hi - lo) * stride1);
                    let (off2, len2) = (lo * stride2, (hi - lo) * stride2);
                    let (off3, len3) = (lo * stride3, (hi - lo) * stride3);
                    #[cfg(any(
                        debug_assertions,
                        peregrine_check_disjoint,
                        feature = "check-disjoint"
                    ))]
                    {
                        tracker1.claim_range(off1, len1);
                        tracker2.claim_range(off2, len2);
                        tracker3.claim_range(off3, len3);
                    }
                    // SAFETY: `pool::share` yields strictly disjoint row ranges across workers.
                    let c1 = unsafe {
                        std::slice::from_raw_parts_mut((raw1 as *mut T1).add(off1), len1)
                    };
                    let c2 = unsafe {
                        std::slice::from_raw_parts_mut((raw2 as *mut T2).add(off2), len2)
                    };
                    let c3 = unsafe {
                        std::slice::from_raw_parts_mut((raw3 as *mut T3).add(off3), len3)
                    };
                    f(wk, lo..hi, c1, c2, c3);
                }
            });
        }
        Sched::StaticAligned(align) => {
            let align = align.max(1);
            pool.run(|wk, nw| {
                let (lo_blk, hi_blk) = crate::pool::share(n_rows / align, wk, nw);
                let lo = lo_blk * align;
                let hi = if wk + 1 == nw { n_rows } else { hi_blk * align };
                if lo < hi {
                    let (off1, len1) = (lo * stride1, (hi - lo) * stride1);
                    let (off2, len2) = (lo * stride2, (hi - lo) * stride2);
                    let (off3, len3) = (lo * stride3, (hi - lo) * stride3);
                    #[cfg(any(
                        debug_assertions,
                        peregrine_check_disjoint,
                        feature = "check-disjoint"
                    ))]
                    {
                        tracker1.claim_range(off1, len1);
                        tracker2.claim_range(off2, len2);
                        tracker3.claim_range(off3, len3);
                    }
                    // SAFETY: Block-aligned shares partition 0..n_rows into strictly disjoint ranges.
                    let c1 = unsafe {
                        std::slice::from_raw_parts_mut((raw1 as *mut T1).add(off1), len1)
                    };
                    let c2 = unsafe {
                        std::slice::from_raw_parts_mut((raw2 as *mut T2).add(off2), len2)
                    };
                    let c3 = unsafe {
                        std::slice::from_raw_parts_mut((raw3 as *mut T3).add(off3), len3)
                    };
                    f(wk, lo..hi, c1, c2, c3);
                }
            });
        }
        Sched::Dynamic(chunk_size) => {
            let chunk_size = chunk_size.max(1);
            let next = AtomicUsize::new(0);
            pool.run(|wk, _nw| loop {
                let lo = next.fetch_add(chunk_size, Ordering::Relaxed);
                if lo >= n_rows {
                    break;
                }
                let hi = (lo + chunk_size).min(n_rows);
                let (off1, len1) = (lo * stride1, (hi - lo) * stride1);
                let (off2, len2) = (lo * stride2, (hi - lo) * stride2);
                let (off3, len3) = (lo * stride3, (hi - lo) * stride3);
                #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
                {
                    tracker1.claim_range(off1, len1);
                    tracker2.claim_range(off2, len2);
                    tracker3.claim_range(off3, len3);
                }
                // SAFETY: Atomic counter guarantees disjoint ranges for each claim.
                let c1 =
                    unsafe { std::slice::from_raw_parts_mut((raw1 as *mut T1).add(off1), len1) };
                let c2 =
                    unsafe { std::slice::from_raw_parts_mut((raw2 as *mut T2).add(off2), len2) };
                let c3 =
                    unsafe { std::slice::from_raw_parts_mut((raw3 as *mut T3).add(off3), len3) };
                f(wk, lo..hi, c1, c2, c3);
            });
        }
    }
}

/// Carve disjoint chunks of a single 1D buffer across worker threads.
#[allow(dead_code)]
pub fn for_chunks<T, F>(pool: &Pool, buf: &mut [T], sched: Sched, f: F)
where
    T: Send,
    F: Fn(usize, Range<usize>, &mut [T]) + Sync,
{
    for_row_chunks(pool, true, buf, 1, buf.len(), sched, f);
}

/// A shared-mutable view of a `&mut [T]` that worker threads can access at disjoint locations.
///
/// Disjoint is `Copy`, but has NO safe methods that return `&mut` or write to elements.
/// All mutable access is mediated by `unsafe fn` methods (`chunk`, `slice`, `slice_ref`, `set`, `get_mut`),
/// which place the burden of ensuring non-overlapping claims on the caller.
pub struct Disjoint<'a, T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    tracker: Option<&'static BufferTracker>,
    _marker: PhantomData<&'a mut [T]>,
}

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
            t.reset();
            Some(t)
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
            t.reset();
            Some(t)
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
    /// # Safety
    /// `off..off+len` must be disjoint from any other slice claimed in the same epoch.
    #[inline(always)]
    pub unsafe fn chunk(&self, off: usize, len: usize) -> &'a mut [T] {
        assert!(
            off.checked_add(len).map_or(false, |end| end <= self.len),
            "Disjoint::chunk out of bounds: off={off}, len={len}, total={}",
            self.len
        );
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        if let Some(t) = self.tracker {
            t.claim_range(off, len);
        }
        std::slice::from_raw_parts_mut(self.ptr.add(off), len)
    }

    /// Alias for `chunk`.
    ///
    /// # Safety
    /// As `chunk`.
    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn slice(&self, off: usize, len: usize) -> &'a mut [T] {
        self.chunk(off, len)
    }

    /// Carve a range of rows given row stride.
    ///
    /// # Safety
    /// As `chunk`.
    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn rows(&self, range: Range<usize>, stride: usize) -> &'a mut [T] {
        let off = range.start * stride;
        let len = (range.end - range.start) * stride;
        self.chunk(off, len)
    }

    /// Carve a single row given row stride.
    ///
    /// # Safety
    /// As `chunk`.
    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn row(&self, r: usize, stride: usize) -> &'a mut [T] {
        self.chunk(r * stride, stride)
    }

    /// Write an element at index `i` (for strided outputs across batch rows).
    ///
    /// In release builds without checking, compiles to a direct pointer write with no bounds check.
    ///
    /// # Safety
    /// `i` must be owned by the calling task/worker and not written by any other task.
    #[inline(always)]
    pub unsafe fn set(&self, i: usize, v: T) {
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        {
            assert!(
                i < self.len,
                "Disjoint::set out of bounds: index {i} >= total {}",
                self.len
            );
            if let Some(t) = self.tracker {
                t.claim_one(i);
            }
        }
        *self.ptr.add(i) = v;
    }

    /// Disjoint per-item mutable access.
    ///
    /// # Safety
    /// `i` must be in bounds and owned by the calling task.
    #[inline(always)]
    pub unsafe fn get_mut(&self, i: usize) -> &'a mut T {
        assert!(
            i < self.len,
            "Disjoint::get_mut out of bounds: index {i} >= total {}",
            self.len
        );
        #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
        if let Some(t) = self.tracker {
            t.claim_one(i);
        }
        &mut *self.ptr.add(i)
    }

    /// Alias for `get_mut` (for compatibility with previous `SharedActs::get`).
    ///
    /// # Safety
    /// As `get_mut`.
    #[inline(always)]
    pub unsafe fn get(&self, i: usize) -> &'a mut T {
        self.get_mut(i)
    }

    /// Shared read-only view of a range.
    ///
    /// # Safety
    /// `off..off+len` must be inside the buffer and no concurrent writer may mutate it while the slice is live.
    #[inline(always)]
    pub unsafe fn slice_ref(&self, off: usize, len: usize) -> &'a [T] {
        assert!(
            off.checked_add(len).map_or(false, |end| end <= self.len),
            "Disjoint::slice_ref out of bounds: off={off}, len={len}, total={}",
            self.len
        );
        std::slice::from_raw_parts(self.ptr.add(off), len)
    }
}

// Type aliases for existing block and model interfaces.
pub type OutPtr<'a> = Disjoint<'a, f32>;
pub type SharedBuf<'a> = Disjoint<'a, f32>;
#[allow(dead_code)]
pub type SharedActs<'a> = Disjoint<'a, QuantAct>;
#[allow(dead_code)]
pub type SharedScratch<'a> = Disjoint<'a, attn::Scratch>;
#[allow(dead_code)]
pub type SharedMl<'a> = Disjoint<'a, (f32, f64)>;

// ---- recurrent-state pointers of the hybrid (Gated DeltaNet) pass ----

/// Per-row raw pointers into the caches' recurrent state, and the snapshots a speculative verification asks
/// for, kept beside the other hybrid scratch so a decode step reuses them instead of allocating four vectors
/// per Gated DeltaNet layer.
pub struct RecScratch {
    /// Row `b`'s `(state, conv)` pointers for the recurrent layer being processed.
    pub rec: Vec<(*mut f32, *mut f32)>,
    /// Row `b`'s snapshot request `(position, state, conv)`, if its cache asked for one.
    pub snap: Vec<Option<(usize, *mut f32, *mut f32)>>,
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
pub struct RowPtrs(pub *const (*mut f32, *mut f32));
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
    pub unsafe fn at(self, b: usize) -> (*mut f32, *mut f32) {
        *self.0.add(b)
    }
}

/// Per row, the snapshot request of its cache for recurrent layer `idx`: (position, state, conv) pointers.
#[derive(Clone, Copy)]
pub struct SnapPtrs(pub *const Option<(usize, *mut f32, *mut f32)>);
// SAFETY: as `RowPtrs`: each snapshot region is written by the task that owns it (a head slice, a channel range).
unsafe impl Send for SnapPtrs {}
unsafe impl Sync for SnapPtrs {}
impl SnapPtrs {
    /// # Safety
    /// `b` must be inside the array, which must outlive the call.
    #[inline(always)]
    pub unsafe fn at(self, b: usize) -> Option<(usize, *mut f32, *mut f32)> {
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
        unsafe {
            let s1 = d.chunk(0, 32);
            s1.fill(1.0);
            let s2 = d.chunk(32, 32);
            s2.fill(2.0);
        }
        assert_eq!(data[0], 1.0);
        assert_eq!(data[31], 1.0);
        assert_eq!(data[32], 2.0);
        assert_eq!(data[63], 2.0);
    }

    #[test]
    fn test_disjoint_set_and_slice_ref() {
        let mut data = [0.0f32; 16];
        let d = Disjoint::new(&mut data);
        unsafe {
            for i in 0..16 {
                d.set(i, i as f32);
            }
            let r = d.slice_ref(0, 16);
            assert_eq!(r[5], 5.0);
        }
    }

    #[test]
    fn test_disjoint_epoch_reuse() {
        let mut data = [0.0f32; 32];
        let d = Disjoint::new(&mut data);
        unsafe {
            let s1 = d.chunk(0, 16);
            s1.fill(1.0);
            bump_epoch();
            let s2 = d.chunk(0, 16);
            s2.fill(2.0);
        }
        assert_eq!(data[0], 2.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_disjoint_out_of_bounds() {
        let mut data = [0.0f32; 16];
        let d = Disjoint::new(&mut data);
        let _ = unsafe { d.chunk(10, 10) };
    }

    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    #[test]
    #[should_panic(expected = "Disjoint overlap detected")]
    fn test_disjoint_chunk_overlap_panics() {
        bump_epoch();
        let mut data = [0.0f32; 64];
        let d = Disjoint::new(&mut data);
        unsafe {
            let _s1 = d.chunk(0, 32);
            let _s2 = d.chunk(16, 32);
        }
    }

    #[cfg(any(debug_assertions, peregrine_check_disjoint, feature = "check-disjoint"))]
    #[test]
    #[should_panic(expected = "Disjoint overlap detected")]
    fn test_disjoint_set_overlap_panics() {
        bump_epoch();
        let mut data = [0.0f32; 64];
        let d = Disjoint::new(&mut data);
        unsafe {
            d.set(5, 1.0);
            d.set(5, 2.0);
        }
    }

    #[test]
    fn test_for_row_chunks_static() {
        let mut data = vec![0.0f32; 1024];
        let pool = Pool::new(4);
        for_row_chunks(
            &pool,
            true,
            &mut data,
            1,
            1024,
            Sched::Static,
            |_wk, range, chunk| {
                for (i, v) in chunk.iter_mut().enumerate() {
                    *v = (range.start + i) as f32;
                }
            },
        );
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, i as f32);
        }
    }

    #[test]
    fn test_for_row_chunks_dynamic() {
        let mut data = vec![0.0f32; 128];
        let pool = Pool::new(4);
        for_row_chunks(
            &pool,
            true,
            &mut data,
            4,
            32,
            Sched::Dynamic(2),
            |_wk, rows, chunk| {
                for (r_idx, r) in rows.enumerate() {
                    for (c_idx, val) in chunk[r_idx * 4..(r_idx + 1) * 4].iter_mut().enumerate() {
                        *val = (r * 4 + c_idx) as f32;
                    }
                }
            },
        );
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, i as f32);
        }
    }

    #[test]
    fn test_for_row_chunks_2_static() {
        let mut d1 = vec![0.0f32; 64];
        let mut d2 = vec![0u32; 16];
        let pool = Pool::new(4);
        for_row_chunks_2(
            &pool,
            true,
            &mut d1,
            4,
            &mut d2,
            1,
            16,
            Sched::Static,
            |_wk, rows, c1, c2| {
                for (idx, r) in rows.enumerate() {
                    c2[idx] = r as u32;
                    c1[idx * 4..(idx + 1) * 4].fill(r as f32);
                }
            },
        );
        for r in 0..16 {
            assert_eq!(d2[r], r as u32);
            assert_eq!(d1[r * 4], r as f32);
        }
    }
}
