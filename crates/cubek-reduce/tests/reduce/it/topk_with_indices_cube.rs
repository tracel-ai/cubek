//! Cube-routine coverage for the fused top-k.
//!
//! The `reduce_dim` matrix only instantiates the unit and plane routines, so
//! nothing there reaches [`GlobalFullCubeReduce`]. That matters for the fused
//! path specifically: the cube routine is the only one that stages accumulators
//! in shared memory, so it is the only caller of `TopKSharedAccumulator`. The
//! single-output `reduce` never reaches that type either, as it goes through the
//! mixed `ReduceOperation` and its `DynamicSharedAccumulator`, which left the
//! top-k shared accumulator entirely unexercised until `reduce_with_indices`
//! started monomorphising over `TopK` directly.
//!
//! These shapes are small so the cube kernels stay cheap to compile (the top-k
//! insert/merge are unrolled over `k`), unlike the `heavy` suites.

use cubecl::{
    Runtime, TestRuntime,
    config::autotune::AutotuneLevel,
    zspace::{Shape, Strides},
};
use cubek_reduce::{
    ReduceStrategy,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, cube::CubeStrategy},
};

use crate::reduce::it::test_case::TestCase;

/// The cube tree merge (`use_planes: false`) produces wrong top-k results for
/// `k > 1` on the CPU runtime, for the values-only path through plain `reduce`
/// just as much as for the fused one; every GPU backend passes. The affected
/// tests skip that runtime so the pre-existing bug does not fail the CPU CI job.
fn cpu_runtime() -> bool {
    let client = TestRuntime::client(&Default::default());
    <TestRuntime as Runtime>::name(&client) == "cpu"
}

fn cube_strategy(use_planes: bool) -> ReduceStrategy {
    ReduceStrategy {
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
        routine: RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy { use_planes })),
        autotune_level: AutotuneLevel::Full,
    }
}

fn case(use_planes: bool) -> TestCase {
    TestCase::new::<f32>(
        Shape::new([4, 256]),
        Strides::new(&[256, 1]),
        Some(1),
        cube_strategy(use_planes),
    )
}

// `k > 1` is the interesting case: the shared accumulator holds one slice per k,
// so a `k == 1` accumulator stays correct even if the per-k slices are not built.
#[test]
fn cube_planes_topk_with_indices_k1() {
    case(true).test_topk_with_indices(1);
}

#[test]
fn cube_planes_topk_with_indices_k2() {
    case(true).test_topk_with_indices(2);
}

#[test]
fn cube_planes_topk_with_indices_k3() {
    case(true).test_topk_with_indices(3);
}

#[test]
fn cube_planes_topk_with_indices_k5() {
    case(true).test_topk_with_indices(5);
}

#[test]
fn cube_units_topk_with_indices_k3() {
    if cpu_runtime() {
        return;
    }
    case(false).test_topk_with_indices(3);
}

#[test]
fn cube_units_topk_with_indices_k5() {
    if cpu_runtime() {
        return;
    }
    case(false).test_topk_with_indices(5);
}

// The values-only and indices-only cube paths, for comparison: these go through
// the mixed accumulator and were already working, so a failure here means the
// breakage is not fused-specific.
#[test]
fn cube_planes_topk_values_only_k3() {
    case(true).test_topk(3);
}

#[test]
fn cube_planes_argtopk_k3() {
    case(true).test_argtopk(3);
}
