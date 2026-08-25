//! Stride-0 (broadcast) problem builders and the baseline-then-broadcast check.
//! A stride of 0 on a logical dimension means it is broadcast: every index along
//! it shares one physical element.

use cubecl::{Runtime, TestRuntime, ir::AddressType, zspace::shape};
use cubek_matmul::{definition::MatmulProblem, strategy::Strategy};
use cubek_std::MatrixLayout;
use cubek_test_utils::{TestOutcome, ValidationResult};

use crate::harness::{f32_elems, run_with_strides};

/// Which logical axis carries the stride-0 broadcast, and on which operand.
#[derive(Clone, Copy)]
pub(crate) enum Broadcast {
    /// Batch dim broadcast on Lhs (logical batch B, one physical matrix).
    BatchLhs,
    /// Batch dim broadcast on Rhs.
    BatchRhs,
    /// Row dim of Lhs broadcast (every output row reads the same Lhs row).
    MLhs,
    /// Column dim of Rhs broadcast (every output column reads the same Rhs col).
    NRhs,
    /// Reduction dim of Lhs broadcast.
    KLhs,
    /// Reduction dim of Rhs broadcast.
    KRhs,
}

impl Broadcast {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Broadcast::BatchLhs => "batch-lhs",
            Broadcast::BatchRhs => "batch-rhs",
            Broadcast::MLhs => "m-lhs",
            Broadcast::NRhs => "n-rhs",
            Broadcast::KLhs => "k-lhs",
            Broadcast::KRhs => "k-rhs",
        }
    }
}

/// Build a problem on a fixed shape. With `zero_stride`, zero the stride of the
/// `broadcast` axis so it is broadcast; otherwise it is the contiguous baseline.
pub(crate) fn make_problem(
    broadcast: Broadcast,
    zero_stride: bool,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
) -> MatmulProblem {
    let (m, n, k) = (64, 64, 128);
    // Batch cases need a real two-batch output; the matrix-dim cases keep a
    // single batch and broadcast inside the matrix.
    let (lhs_batch, rhs_batch) = match broadcast {
        Broadcast::BatchLhs | Broadcast::BatchRhs => (2, 2),
        _ => (1, 1),
    };

    let mut problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![lhs_batch],
        shape![rhs_batch],
        lhs_layout,
        rhs_layout,
        MatrixLayout::RowMajor,
        None,
        None,
        f32_elems(),
        AddressType::U32,
    );

    if zero_stride {
        // rank-3 stride layout: [batch, row, col].
        match broadcast {
            Broadcast::BatchLhs => problem.lhs_strides[0] = 0,
            Broadcast::BatchRhs => problem.rhs_strides[0] = 0,
            Broadcast::MLhs => problem.lhs_strides[1] = 0,
            Broadcast::NRhs => problem.rhs_strides[2] = 0,
            Broadcast::KLhs => problem.lhs_strides[2] = 0,
            Broadcast::KRhs => problem.rhs_strides[1] = 0,
        }
    }

    problem
}

pub(crate) fn passed(outcome: &TestOutcome) -> bool {
    matches!(outcome, TestOutcome::Validated(ValidationResult::Pass))
}

/// A strategy that computes the contiguous baseline correctly must keep
/// computing correctly once a *batch* stride is zeroed. Strategies that can't
/// run the shape on this backend (e.g. cmma without the feature) are skipped.
pub(crate) fn assert_batch_broadcast(strategy: Strategy) {
    use MatrixLayout::{ColMajor as C, RowMajor as R};
    let client = TestRuntime::client(&Default::default());
    for broadcast in [Broadcast::BatchLhs, Broadcast::BatchRhs] {
        let baseline = run_with_strides(
            client.clone(),
            make_problem(broadcast, false, R, C),
            strategy.clone(),
        );
        if !passed(&baseline) {
            continue;
        }
        let out = run_with_strides(
            client.clone(),
            make_problem(broadcast, true, R, C),
            strategy.clone(),
        );
        assert!(
            passed(&out),
            "{strategy}: {} batch broadcast gave {out:?}",
            broadcast.label()
        );
    }
}
