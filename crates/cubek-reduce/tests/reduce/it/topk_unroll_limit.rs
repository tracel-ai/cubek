//! Top-k past `TOPK_UNROLL_BUDGET`.
//!
//! `plane_topk_insert`, `plane_topk_merge` and `topk_finalize_*` each walk all
//! `k` accumulator slots for each of their `k` candidates. Both levels used to
//! be unrolled unconditionally, so the emitted kernel grew with `k²`: the
//! `topk(300)` an object-detection head asks for expanded to ~90k copies of the
//! insertion body and no backend compiler returned from it. Past the limit those
//! loops stay rolled up, and the cases below cover the rolled form of every one
//! of them.
//!
//! `k` is only just past the limit so the *unrolled* comparison stays cheap; the
//! point is which code path runs, not how large `k` gets.

use cubecl::{
    config::autotune::AutotuneLevel,
    prelude::CubeDim,
    zspace::{Shape, Strides},
};
use cubek_reduce::{
    BoundChecks, IdleMode, ReduceStrategy,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, PlaneMergeStrategy, PlaneReduceBlueprint, unit::UnitStrategy},
};

use crate::reduce::it::test_case::TestCase;

/// Past the budget for the `k`-by-`k` nests, which roll once `k * k` exceeds
/// `TOPK_UNROLL_BUDGET`. The `k * k * vector_size` nests roll well before this.
const K: usize = 40;

fn case(routine: RoutineStrategy, parallel_output_vectorization: bool) -> TestCase {
    TestCase::new::<f32>(
        Shape::new([4, 256]),
        Strides::new(&[256, 1]),
        Some(1),
        ReduceStrategy {
            autotune_level: AutotuneLevel::Full,
            vectorization: VectorizationStrategy {
                parallel_output_vectorization,
            },
            routine,
        },
    )
}

/// `Eager` folds every candidate into the plane-wide accumulator as it arrives,
/// which is `plane_topk_insert`; `Lazy` accumulates per lane and merges the
/// lanes once at the end, which is `plane_topk_merge`. Both are forced, since an
/// inferred blueprint picks one of them and would leave the other uncovered.
fn plane(plane_merge_strategy: PlaneMergeStrategy) -> RoutineStrategy {
    RoutineStrategy::Plane(BlueprintStrategy::Forced(
        PlaneReduceBlueprint {
            plane_idle: IdleMode::Terminate,
            bound_checks: BoundChecks::Mask,
            plane_merge_strategy,
            plane_dim_ceil: true,
        },
        CubeDim::new_2d(32, 2),
    ))
}

fn unit() -> RoutineStrategy {
    RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy))
}

/// The case from the bug report, and the only one here that actually fails
/// without the limit: `k = 300` is what an object-detection head asks for, and
/// unrolled it expands to ~90k copies of the insertion body. The failure mode is
/// a hang, not an assertion — expanding the kernel alone went from 15 ms to
/// minutes between `k = 64` and `k = 128`, and never returned at 300 — so
/// without the fix this test does not finish rather than reporting red.
///
/// The `K` cases below are the correctness half: they cover the rolled form of
/// every selection network, but they pass either way, since that `k` unrolled
/// still compiles.
#[test]
fn topk_300_terminates_and_matches_reference() {
    TestCase::new::<f32>(
        Shape::new([1, 1024]),
        Strides::new(&[1024, 1]),
        Some(1),
        ReduceStrategy {
            autotune_level: AutotuneLevel::Full,
            vectorization: VectorizationStrategy {
                parallel_output_vectorization: true,
            },
            routine: unit(),
        },
    )
    .test_topk_with_indices(300);
}

#[test]
fn plane_eager_topk_with_indices_past_unroll_limit() {
    case(plane(PlaneMergeStrategy::Eager), false).test_topk_with_indices(K);
}

#[test]
fn plane_eager_topk_past_unroll_limit() {
    case(plane(PlaneMergeStrategy::Eager), false).test_topk(K);
}

#[test]
fn plane_lazy_topk_with_indices_past_unroll_limit() {
    case(plane(PlaneMergeStrategy::Lazy), false).test_topk_with_indices(K);
}

#[test]
fn plane_lazy_topk_past_unroll_limit() {
    case(plane(PlaneMergeStrategy::Lazy), false).test_topk(K);
}

// Output vectorization is what routes the accumulator out through
// `topk_finalize_with_coords` / `topk_finalize_values`, whose `k`-by-`k` nest
// runs once more per lane of the vector.
#[test]
fn vectorized_topk_with_indices_past_unroll_limit() {
    case(unit(), true).test_topk_with_indices(K);
}

#[test]
fn vectorized_topk_past_unroll_limit() {
    case(unit(), true).test_topk(K);
}

#[test]
fn vectorized_argtopk_past_unroll_limit() {
    case(unit(), true).test_argtopk(K);
}
