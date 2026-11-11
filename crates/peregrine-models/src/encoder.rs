//! The shared piece of a bidirectional encoder pass: one layer's projections over the
//! rows of the whole batch (design §2.2, roadmap S4.4 and S4.14).
//!
//! An encoder forwards a whole text (up to the model's context) in one pass, but the
//! shared GEMM accepts at most [`peregrine_core::blocks::linear::MAX_BATCH`] rows per
//! call. These helpers run a layer's projection in as many calls as it takes, one
//! weight pass each. Rows of the batch are contiguous, so a chunk is a plain row range
//! and a row's bits do not depend on where a chunk boundary fell.
//!
//! The families differ in their blocks, not in this scheduling, so `gemma-embedding` and
//! `mistral3` share the code. It was moved out of `gemma3` unchanged when the second
//! encoder arrived.

use peregrine_core::blocks::linear::LinJob;
use peregrine_core::blocks::par::{Disjoint, OutPtr};
use peregrine_core::model::{Linear, StepStats, Weights};
use peregrine_core::pool::Pool;

/// One projection over many rows, split into matmul-sized chunks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_rows(
    w: &Weights,
    pool: &Pool,
    stats: &mut StepStats,
    lin: &Linear,
    acts: &mut [peregrine_core::quant::QuantAct],
    lb_x: &mut [f32],
    lb_t: &mut [f32],
    input: &[f32],
    d_in: usize,
    nb: usize,
    out: &mut [f32],
    out_dim: usize,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    let step = peregrine_core::blocks::linear::MAX_BATCH;
    let mut r0 = 0;
    while r0 < nb {
        let r1 = (r0 + step).min(nb);
        let take = r1 - r0;
        for (i, a) in acts[r0..r1].iter_mut().enumerate() {
            a.fill(&input[(r0 + i) * d_in..(r0 + i + 1) * d_in]);
        }
        let lbx = Disjoint::new(lb_x);
        let lbt = Disjoint::new(lb_t);
        let jobs = [LinJob {
            lin,
            out: Disjoint::new(&mut out[r0 * out_dim..r1 * out_dim]),
        }];
        peregrine_core::blocks::linear::linear_group(
            w,
            w.lb(),
            &jobs,
            &acts[r0..r1],
            &input[r0 * d_in..r1 * d_in],
            d_in,
            take,
            lbx,
            lbt,
            pool,
            stats,
            kq,
        );
        r0 = r1;
    }
}

/// One or more projections that read the **same** activation rows, in matmul-sized
/// chunks, without quantising anything: the rows are already in `acts`, filled once by
/// the norm or activation that produced them (S4.9a).
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_rows_shared(
    w: &Weights,
    pool: &Pool,
    stats: &mut StepStats,
    jobs: &mut [RowsOut<'_>],
    acts: &[peregrine_core::quant::QuantAct],
    lb_x: &mut [f32],
    lb_t: &mut [f32],
    input: &[f32],
    d_in: usize,
    nb: usize,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    debug_assert!(!jobs.is_empty() && jobs.len() <= 4);
    let step = peregrine_core::blocks::linear::MAX_BATCH;
    let empty: &mut [f32] = &mut [];
    let mut lj = [LinJob {
        lin: jobs[0].lin,
        out: Disjoint::new(empty),
    }; 4];
    let mut r0 = 0;
    let njobs = jobs.len();
    while r0 < nb {
        let r1 = (r0 + step).min(nb);
        let take = r1 - r0;
        for i in 0..njobs {
            lj[i].lin = jobs[i].lin;
            let width = jobs[i].out_dim;
            // SAFETY: each job's output is a different buffer (q, k, v; gate, up; one
            // projection), and this chunk of it belongs to this call alone.
            lj[i].out = Disjoint::new(unsafe { jobs[i].out.rows(r0..r1, width) });
        }
        let lbx = Disjoint::new(lb_x);
        let lbt = Disjoint::new(lb_t);
        peregrine_core::blocks::linear::linear_group(
            w,
            w.lb(),
            &lj[..njobs],
            &acts[r0..r1],
            &input[r0 * d_in..r1 * d_in],
            d_in,
            take,
            lbx,
            lbt,
            pool,
            stats,
            kq,
        );
        r0 = r1;
    }
}

/// One projection of [`linear_rows_shared`]: the matrix, its output buffer, and the
/// width of one output row. Every job's buffer must be a different one — the group
/// writes them in one fork-join.
pub(crate) struct RowsOut<'a> {
    pub(crate) lin: &'a Linear,
    pub(crate) out: OutPtr<'a>,
    pub(crate) out_dim: usize,
}
