//! Raw-pointer wrappers that hand the pool's workers disjoint parts of one buffer.
//!
//! Every `unsafe` handover of shared output between worker threads in the model code goes through the types
//! here, so their contracts sit in one place. Each keeps the safety argument it had beside its old use site.
//! Roadmap S1.3 replaces their insides with one checked helper; callers should not need to change.

use crate::attn;
use crate::quant::QuantAct;

/// A write cursor into the matmul's output, shared across workers.
///
/// Raw rather than a `&mut [f32]` split because the split is by row *stride*
/// (`b*rows + r`), not by a contiguous slice: with a batch, each worker writes an
/// interleaved set of ranges that `split_at_mut` cannot express. Disjointness is
/// guaranteed by `pool::share` instead, and tested there.
#[derive(Clone, Copy)]
pub(crate) struct OutPtr(pub(crate) *mut f32);
// SAFETY: workers write only indices inside their own share, which are disjoint.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    /// # Safety
    /// `i` must be inside the output buffer and must belong to the calling worker's
    /// share, so no two workers ever write the same index.
    ///
    /// This is a method rather than a bare field access on purpose: with edition
    /// 2021's disjoint closure capture, writing `dst.0` inside the worker closure
    /// captures the `*mut f32` field instead of the wrapper, and the closure then
    /// fails `Sync` however the wrapper is marked.
    #[inline(always)]
    pub(crate) unsafe fn set(self, i: usize, v: f32) {
        *self.0.add(i) = v;
    }
}

/// A shared base pointer from which workers carve out *disjoint* sub-slices: one
/// head's output, or one worker's softmax row. Same contract as `OutPtr`, for ranges.
#[derive(Clone, Copy)]
pub(crate) struct SharedBuf(pub(crate) *mut f32);
// SAFETY: every user takes sub-ranges no other worker takes (asserted by construction:
// ranges are keyed by the task or worker index).
unsafe impl Send for SharedBuf {}
unsafe impl Sync for SharedBuf {}

impl SharedBuf {
    /// # Safety
    /// `off..off+len` must be inside the buffer and used by exactly one worker.
    #[inline(always)]
    pub(crate) unsafe fn slice<'a>(self, off: usize, len: usize) -> &'a mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(off), len)
    }
    /// Shared read of a range no worker is writing — the LittleBit stage buffers,
    /// written in one phase and read by every worker in the next.
    ///
    /// # Safety
    /// `off..off+len` must be inside the buffer, and nothing may write it while
    /// the returned slice lives.
    #[inline(always)]
    pub(crate) unsafe fn slice_ref<'a>(self, off: usize, len: usize) -> &'a [f32] {
        std::slice::from_raw_parts(self.0.add(off), len)
    }
}

/// Disjoint per-row access to the activation scratch, for the row-parallel prologues.
#[derive(Clone, Copy)]
pub(crate) struct SharedActs(pub(crate) *mut QuantAct);
// SAFETY: as `SharedBuf` — each index is taken by exactly one worker.
unsafe impl Send for SharedActs {}
unsafe impl Sync for SharedActs {}

impl SharedActs {
    /// # Safety
    /// `i` must be in bounds and used by exactly one worker.
    #[inline(always)]
    pub(crate) unsafe fn get<'a>(self, i: usize) -> &'a mut QuantAct {
        &mut *self.0.add(i)
    }
}

/// Disjoint per-worker access to the attention scratch: worker `w` takes index `w`.
#[derive(Clone, Copy)]
pub(crate) struct SharedScratch(pub(crate) *mut attn::Scratch);
// SAFETY: as `SharedBuf` — each worker takes only its own index.
unsafe impl Send for SharedScratch {}
unsafe impl Sync for SharedScratch {}

impl SharedScratch {
    /// # Safety
    /// `w` must be in bounds and used by exactly one worker.
    #[inline(always)]
    pub(crate) unsafe fn get<'a>(self, w: usize) -> &'a mut attn::Scratch {
        &mut *self.0.add(w)
    }
}

/// Disjoint slots of the partial-softmax `(m, l)` buffer, keyed by task row.
#[derive(Clone, Copy)]
pub(crate) struct SharedMl(pub(crate) *mut (f32, f64));
// SAFETY: each slot is written by the one task that owns its (head, chunk, row).
unsafe impl Send for SharedMl {}
unsafe impl Sync for SharedMl {}

impl SharedMl {
    /// # Safety
    /// `i` must be in bounds and owned by the calling task.
    #[inline(always)]
    pub(crate) unsafe fn set(self, i: usize, v: (f32, f64)) {
        *self.0.add(i) = v;
    }
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
