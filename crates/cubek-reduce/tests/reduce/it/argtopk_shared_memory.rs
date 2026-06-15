//! End-to-end coverage for the shared-memory clamp on the inferred `Cube` routine.
//!
//! Before the clamp, the inferred Cube blueprint sized its shared allocation from
//! plane geometry alone, so `ArgTopK(k)` requested `8 * k * cube_width` bytes and the
//! launch was rejected once `k` passed ~12 on a 99 KiB device, even though a narrower
//! cube fit. Gated behind `heavy` like the other cube tests (the kernels are slow to
//! compile, the top-k insert/merge are unrolled over `k`).

#![cfg(feature = "heavy")]

use cubecl::zspace::{Shape, Strides};
use cubek_reduce::{
    ReduceStrategy,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, cube::CubeStrategy},
};

use crate::reduce::it::test_case::TestCase;

fn inferred_cube_strategy(use_planes: bool) -> ReduceStrategy {
    ReduceStrategy {
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
        routine: RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy { use_planes })),
    }
}

/// Run an inferred-Cube `ArgTopK(k)` over `[64, 4096]` (axis 1), the shape from the
/// bug report, and assert the output matches the CPU top-k reference. With the old
/// geometry the inferred cube width is 1024, so `8 * k * 1024` overruns shared
/// memory for `k >= 13`. The clamp must shrink the width and let the launch through.
fn run_wide_argtopk(k: usize) {
    TestCase::new::<f32>(
        Shape::new([64, 4096]),
        Strides::new(&[4096, 1]),
        Some(1),
        inferred_cube_strategy(false),
    )
    .test_argtopk(k);
}

#[test]
fn argtopk_k13_inferred_cube_launches() {
    // The smallest k that failed to launch before the fix (8 * 13 * 1024 = 106496
    // bytes vs the 101376-byte Ada limit). Larger k is launchable too, but the
    // top-k codegen is unrolled over k and compiles slowly (a separate issue), so
    // a single representative k keeps this suite fast.
    run_wide_argtopk(13);
}
