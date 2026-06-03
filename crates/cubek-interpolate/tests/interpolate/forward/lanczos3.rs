use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::{
    definition::{InterpolateMode, InterpolateOptions},
    launch::InterpolateStrategy,
    routines::{
        BlueprintStrategy, GlobalMemoryRoutine, GlobalMemoryStrategy, SharedMemoryRoutine,
        SharedMemoryStrategy,
    },
};

use super::{make_problem, run_interpolate_global_test};

const LANCZOS3_TOLERANCE: f32 = 0.00001;
const LANCZOS3_HIGH_RESOLUTION_TOLERANCE: f32 = 0.001;

const SHARED_MEMORY_HEIGHT: usize = 6;

#[test]
fn test_interpolate_lanczos3_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 16],
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_global_test(
        client,
        5678,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {}),
        ),
        LANCZOS3_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_shared_memory_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 16],
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_global_test(
        client,
        5678,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                shared_memory_height: SHARED_MEMORY_HEIGHT,
            }),
        ),
        LANCZOS3_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [10, 10],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_global_test(
        client,
        1234,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {}),
        ),
        LANCZOS3_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_shared_memory_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [10, 10],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_global_test(
        client,
        1234,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                shared_memory_height: SHARED_MEMORY_HEIGHT,
            }),
        ),
        LANCZOS3_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [2, 2],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_global_test(
        client,
        91011,
        -100.0,
        100.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {}),
        ),
        LANCZOS3_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_shared_memory_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [2, 2],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_global_test(
        client,
        91011,
        -100.0,
        100.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                shared_memory_height: SHARED_MEMORY_HEIGHT,
            }),
        ),
        LANCZOS3_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_resize() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [8, 16],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_global_test(
        client,
        25,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {}),
        ),
        LANCZOS3_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_shared_memory_resize() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [8, 16],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_global_test(
        client,
        25,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                shared_memory_height: SHARED_MEMORY_HEIGHT,
            }),
        ),
        LANCZOS3_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_without_align_corners() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [16, 16],
        InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(false),
    );
    run_interpolate_global_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {}),
        ),
        LANCZOS3_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_shared_memory_without_align_corners() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [16, 16],
        InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(false),
    );
    run_interpolate_global_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                shared_memory_height: SHARED_MEMORY_HEIGHT,
            }),
        ),
        LANCZOS3_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_high_resolution() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [5, 89, 43, 13],
        [321, 75],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_global_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {}),
        ),
        LANCZOS3_HIGH_RESOLUTION_TOLERANCE,
    );
}

#[test]
fn test_interpolate_lanczos3_shared_memory_high_resolution() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [5, 89, 43, 13],
        [321, 75],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_global_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                shared_memory_height: SHARED_MEMORY_HEIGHT,
            }),
        ),
        LANCZOS3_HIGH_RESOLUTION_TOLERANCE,
    );
}
