//! Broadcast (stride-0) coverage for the public [`Strategy`] table.
//!
//! A stride of 0 on a logical dimension means it is broadcast: every index
//! along it shares one physical element. The valid, widely-supported case for
//! matmul is a broadcast *batch* dim (numpy/torch `expand` then matmul). Each
//! strategy gets its own `batch_broadcast_*` test so a failure names exactly
//! which strategy regressed.
//!
//! Stride-0 on the actual matrix dims (m, n, k) is a separate story: the
//! unit/naive path and gemm/cpu_gemm materialize it (via `into_contiguous`),
//! while the plane strategies still reject it. The `gemm_*` and
//! `reported_m_broadcast` tests pin that behaviour.

use cubecl::{Runtime, frontend::CubePrimitive, ir::AddressType, zspace::shape};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulProblem},
    routines::BlueprintStrategy,
    strategy::Strategy,
};
use cubek_std::MatrixLayout;
use cubek_test_utils::{TestOutcome, ValidationResult};

use crate::matmul::launcher_strategy::run_with_strides;

type TestRuntime = cubecl::TestRuntime;

fn f32_elems() -> MatmulGlobalElems {
    MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
}

/// Which logical axis carries the stride-0 broadcast, and on which operand.
#[derive(Clone, Copy)]
enum Broadcast {
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
    fn label(self) -> &'static str {
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
fn make_problem(
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

fn passed(outcome: &TestOutcome) -> bool {
    matches!(outcome, TestOutcome::Validated(ValidationResult::Pass))
}

/// A strategy that computes the contiguous baseline correctly must keep
/// computing correctly once a *batch* stride is zeroed. Strategies that can't
/// run the shape on this backend (e.g. cmma without the feature) are skipped.
fn assert_batch_broadcast(strategy: Strategy) {
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

// One independent test per public strategy: a failure names the strategy.
#[test]
fn batch_broadcast_simple_cyclic_cmma() {
    assert_batch_broadcast(Strategy::SimpleCyclicCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_cyclic_mma() {
    assert_batch_broadcast(Strategy::SimpleCyclicMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_strided_cmma() {
    assert_batch_broadcast(Strategy::SimpleStridedCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_strided_mma() {
    assert_batch_broadcast(Strategy::SimpleStridedMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_tilewise_cmma() {
    assert_batch_broadcast(Strategy::SimpleTilewiseCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_tilewise_mma() {
    assert_batch_broadcast(Strategy::SimpleTilewiseMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_async_strided_cmma() {
    assert_batch_broadcast(Strategy::SimpleAsyncStridedCmma(
        BlueprintStrategy::default(),
    ));
}

#[test]
fn batch_broadcast_simple_async_strided_mma() {
    assert_batch_broadcast(Strategy::SimpleAsyncStridedMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_async_cyclic_cmma() {
    assert_batch_broadcast(Strategy::SimpleAsyncCyclicCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_async_cyclic_mma() {
    assert_batch_broadcast(Strategy::SimpleAsyncCyclicMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_tma_cmma() {
    assert_batch_broadcast(Strategy::SimpleTmaCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_tma_mma() {
    assert_batch_broadcast(Strategy::SimpleTmaMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_cyclic_cmma() {
    assert_batch_broadcast(Strategy::DoubleCyclicCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_cyclic_mma() {
    assert_batch_broadcast(Strategy::DoubleCyclicMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_tilewise_cmma() {
    assert_batch_broadcast(Strategy::DoubleTilewiseCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_tilewise_mma() {
    assert_batch_broadcast(Strategy::DoubleTilewiseMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_hybrid_cmma() {
    assert_batch_broadcast(Strategy::DoubleHybridCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_hybrid_mma() {
    assert_batch_broadcast(Strategy::DoubleHybridMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_async_cyclic_cmma() {
    assert_batch_broadcast(Strategy::DoubleAsyncCyclicCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_async_cyclic_mma() {
    assert_batch_broadcast(Strategy::DoubleAsyncCyclicMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_async_strided_cmma() {
    assert_batch_broadcast(Strategy::DoubleAsyncStridedCmma(
        BlueprintStrategy::default(),
    ));
}

#[test]
fn batch_broadcast_double_async_strided_mma() {
    assert_batch_broadcast(Strategy::DoubleAsyncStridedMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_tma_cmma() {
    assert_batch_broadcast(Strategy::DoubleTmaCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_tma_mma() {
    assert_batch_broadcast(Strategy::DoubleTmaMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_specialized_cyclic_cmma() {
    assert_batch_broadcast(Strategy::SpecializedCyclicCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_specialized_cyclic_mma() {
    assert_batch_broadcast(Strategy::SpecializedCyclicMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_specialized_strided_cmma() {
    assert_batch_broadcast(Strategy::SpecializedStridedCmma(
        BlueprintStrategy::default(),
    ));
}

#[test]
fn batch_broadcast_specialized_strided_mma() {
    assert_batch_broadcast(Strategy::SpecializedStridedMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_specialized_tma_cmma() {
    assert_batch_broadcast(Strategy::SpecializedTmaCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_specialized_tma_mma() {
    assert_batch_broadcast(Strategy::SpecializedTmaMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_ordered_double_cmma() {
    assert_batch_broadcast(Strategy::OrderedDoubleCmma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_ordered_double_mma() {
    assert_batch_broadcast(Strategy::OrderedDoubleMma(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_unit() {
    assert_batch_broadcast(Strategy::SimpleUnit(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_unit() {
    assert_batch_broadcast(Strategy::DoubleUnit(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_simple_vecmat() {
    assert_batch_broadcast(Strategy::SimpleVecMat(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_double_vecmat() {
    assert_batch_broadcast(Strategy::DoubleVecMat(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_gemv_unit_perpendicular() {
    assert_batch_broadcast(Strategy::GemvUnitPerpendicular(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_gemm() {
    assert_batch_broadcast(Strategy::Gemm(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_cpu_gemm() {
    assert_batch_broadcast(Strategy::CpuGemm(BlueprintStrategy::default()));
}

#[test]
fn batch_broadcast_naive() {
    assert_batch_broadcast(Strategy::Naive);
}

#[test]
fn batch_broadcast_auto() {
    assert_batch_broadcast(Strategy::Auto);
}

/// The exact reported case: lhs `[300,4,256]` strides `[256,0,1]` (M broadcast on
/// lhs) @ rhs `[1,256,256]` strides `[65536,1,256]` (batch broadcast on rhs).
/// The default `Auto` path must compute it correctly (it does, via
/// `into_contiguous`); this guards against a backend regression.
#[test]
fn reported_m_broadcast() {
    use MatrixLayout::{ColMajor, RowMajor};
    let client = TestRuntime::client(&Default::default());
    let mut problem = MatmulProblem::from_parameters(
        4,
        256,
        256,
        shape![300],
        shape![1],
        RowMajor,
        ColMajor,
        RowMajor,
        None,
        None,
        f32_elems(),
        AddressType::U32,
    );
    problem.lhs_strides[1] = 0;
    let outcome = run_with_strides(client, problem, Strategy::Auto);
    assert!(
        passed(&outcome),
        "reported M-broadcast repro gave {outcome:?}"
    );
}

/// gemm's outcome on a stride-0 `broadcast` for the given layouts, or `None` if
/// the contiguous baseline can't run on this backend+layout (nothing to judge).
fn gemm_broadcast_outcome(
    broadcast: Broadcast,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
) -> Option<TestOutcome> {
    let client = TestRuntime::client(&Default::default());
    let gemm = || Strategy::Gemm(BlueprintStrategy::default());
    let baseline = run_with_strides(
        client.clone(),
        make_problem(broadcast, false, lhs_layout, rhs_layout),
        gemm(),
    );
    passed(&baseline).then(|| {
        run_with_strides(
            client,
            make_problem(broadcast, true, lhs_layout, rhs_layout),
            gemm(),
        )
    })
}

/// gemm must compute a broadcast batch correctly on every layout the family
/// supports (Dot / OuterN / OuterM); unsupported variants are skipped.
#[test]
fn gemm_handles_broadcast_batch_all_layouts() {
    use MatrixLayout::{ColMajor as C, RowMajor as R};
    for (lhs_l, rhs_l) in [(R, C), (R, R), (C, R), (C, C)] {
        for broadcast in [Broadcast::BatchLhs, Broadcast::BatchRhs] {
            if let Some(outcome) = gemm_broadcast_outcome(broadcast, lhs_l, rhs_l) {
                assert!(
                    passed(&outcome),
                    "gemm batch {} [{lhs_l:?}/{rhs_l:?}] gave {outcome:?}",
                    broadcast.label()
                );
            }
        }
    }
}

/// gemm now supports a broadcast (stride-0) M dim — the reported case: the
/// launch materializes the operand via `into_contiguous`, so it computes
/// correctly on both backends. N- and K-broadcast also materialize but then hit
/// pre-existing, unrelated constraints (a materialized N flips the layout into
/// gemm's `OuterN` variant, which is CPU-only; transposed K trips an upstream
/// cubecl-cpu `into_contiguous` bug), so they aren't asserted here.
#[test]
fn gemm_handles_broadcast_m() {
    use MatrixLayout::{ColMajor as C, RowMajor as R};
    if let Some(outcome) = gemm_broadcast_outcome(Broadcast::MLhs, R, C) {
        assert!(passed(&outcome), "gemm m-lhs broadcast gave {outcome:?}");
    }
}
