//! Broadcast (stride-0) coverage for the multi-level strategy table.
//!
//! Each strategy gets its own `batch_broadcast_*` test so a failure names exactly
//! which strategy regressed. Stride-0 on the actual matrix dims (m, n, k) is a
//! separate story: the unit/naive path and gemm materialize it (via
//! `into_contiguous`), while the plane strategies reject it. The `gemm_*` tests
//! pin that behaviour.

use cubecl::{Runtime, TestRuntime};
use cubek_matmul::{multi_level::Strategy as MultiLevel, routine::BlueprintStrategy};
use cubek_std::MatrixLayout;
use cubek_test_utils::TestOutcome;

use crate::harness::{Broadcast, assert_batch_broadcast, make_problem, passed, run_with_strides};

// One independent test per public strategy: a failure names the strategy.
#[test]
fn batch_broadcast_simple_cyclic_cmma() {
    assert_batch_broadcast(MultiLevel::SimpleCyclicCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_cyclic_mma() {
    assert_batch_broadcast(MultiLevel::SimpleCyclicMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_strided_cmma() {
    assert_batch_broadcast(MultiLevel::SimpleStridedCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_strided_mma() {
    assert_batch_broadcast(MultiLevel::SimpleStridedMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_tilewise_cmma() {
    assert_batch_broadcast(MultiLevel::SimpleTilewiseCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_tilewise_mma() {
    assert_batch_broadcast(MultiLevel::SimpleTilewiseMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_async_strided_cmma() {
    assert_batch_broadcast(MultiLevel::SimpleAsyncStridedCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_async_strided_mma() {
    assert_batch_broadcast(MultiLevel::SimpleAsyncStridedMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_async_cyclic_cmma() {
    assert_batch_broadcast(MultiLevel::SimpleAsyncCyclicCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_async_cyclic_mma() {
    assert_batch_broadcast(MultiLevel::SimpleAsyncCyclicMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_tma_cmma() {
    assert_batch_broadcast(MultiLevel::SimpleTmaCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_tma_mma() {
    assert_batch_broadcast(MultiLevel::SimpleTmaMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_cyclic_cmma() {
    assert_batch_broadcast(MultiLevel::DoubleCyclicCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_cyclic_mma() {
    assert_batch_broadcast(MultiLevel::DoubleCyclicMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_tilewise_cmma() {
    assert_batch_broadcast(MultiLevel::DoubleTilewiseCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_tilewise_mma() {
    assert_batch_broadcast(MultiLevel::DoubleTilewiseMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_hybrid_cmma() {
    assert_batch_broadcast(MultiLevel::DoubleHybridCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_hybrid_mma() {
    assert_batch_broadcast(MultiLevel::DoubleHybridMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_async_cyclic_cmma() {
    assert_batch_broadcast(MultiLevel::DoubleAsyncCyclicCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_async_cyclic_mma() {
    assert_batch_broadcast(MultiLevel::DoubleAsyncCyclicMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_async_strided_cmma() {
    assert_batch_broadcast(MultiLevel::DoubleAsyncStridedCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_async_strided_mma() {
    assert_batch_broadcast(MultiLevel::DoubleAsyncStridedMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_tma_cmma() {
    assert_batch_broadcast(MultiLevel::DoubleTmaCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_tma_mma() {
    assert_batch_broadcast(MultiLevel::DoubleTmaMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_specialized_cyclic_cmma() {
    assert_batch_broadcast(MultiLevel::SpecializedCyclicCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_specialized_cyclic_mma() {
    assert_batch_broadcast(MultiLevel::SpecializedCyclicMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_specialized_strided_cmma() {
    assert_batch_broadcast(MultiLevel::SpecializedStridedCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_specialized_strided_mma() {
    assert_batch_broadcast(MultiLevel::SpecializedStridedMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_specialized_tma_cmma() {
    assert_batch_broadcast(MultiLevel::SpecializedTmaCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_specialized_tma_mma() {
    assert_batch_broadcast(MultiLevel::SpecializedTmaMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_ordered_double_cmma() {
    assert_batch_broadcast(MultiLevel::OrderedDoubleCmma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_ordered_double_mma() {
    assert_batch_broadcast(MultiLevel::OrderedDoubleMma(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_unit() {
    assert_batch_broadcast(MultiLevel::SimpleUnit(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_unit() {
    assert_batch_broadcast(MultiLevel::DoubleUnit(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_simple_vecmat() {
    assert_batch_broadcast(MultiLevel::SimpleVecMat(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_double_vecmat() {
    assert_batch_broadcast(MultiLevel::DoubleVecMat(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_gemv_unit_perpendicular() {
    assert_batch_broadcast(MultiLevel::GemvUnitPerpendicular(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_gemm() {
    assert_batch_broadcast(MultiLevel::Gemm(BlueprintStrategy::default()).into());
}

#[test]
fn batch_broadcast_naive() {
    assert_batch_broadcast(MultiLevel::Naive.into());
}

/// gemm's outcome on a stride-0 `broadcast` for the given layouts, or `None` if
/// the contiguous baseline can't run on this backend+layout (nothing to judge).
fn gemm_broadcast_outcome(
    broadcast: Broadcast,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
) -> Option<TestOutcome> {
    let client = TestRuntime::client(&Default::default());
    let gemm = || MultiLevel::Gemm(BlueprintStrategy::default()).into();
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

/// gemm now supports a broadcast (stride-0) M dim, the reported case: the
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
