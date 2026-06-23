use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::{
    definition::{InterpolateMode, InterpolateOptions, TileSize},
    launch::InterpolateStrategy,
    routines::{
        BlueprintStrategy, GlobalMemoryRoutine, GlobalMemoryStrategy, SharedMemoryRoutine,
        SharedMemoryStrategy,
    },
};

use super::{make_problem, run_interpolate_global_test};

const BICUBIC_TOLERANCE: f32 = 0.0001;
const BICUBIC_HIGH_RESOLUTION_TOLERANCE: f32 = 0.001;

const TILE_SIZE: TileSize = TileSize::new(16, 16);

#[test]
fn test_interpolate_bicubic_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 16],
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        5678,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_shared_memory_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 16],
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        5678,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [10, 10],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        1234,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_shared_memory_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [10, 10],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        1234,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [2, 2],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        91011,
        -100.0,
        100.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_shared_memory_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [2, 2],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        91011,
        -100.0,
        100.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_resize() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [8, 16],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        25,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_shared_memory_resize() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [8, 16],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        25,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_without_align_corners() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [16, 16],
        InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false),
    );
    run_interpolate_global_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_shared_memory_without_align_corners() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [16, 16],
        InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false),
    );
    run_interpolate_global_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_high_resolution() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [5, 89, 43, 13],
        [321, 75],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_HIGH_RESOLUTION_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_shared_memory_high_resolution() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [5, 89, 43, 13],
        [321, 75],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_HIGH_RESOLUTION_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_bhwc_512() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [1, 512, 512, 1],
        [1024, 1024],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        122,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_HIGH_RESOLUTION_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bicubic_shared_memory_bhwc_512() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [1, 512, 512, 1],
        [1024, 1024],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_global_test(
        client,
        122,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                tile_size: TILE_SIZE,
            }),
        ),
        BICUBIC_HIGH_RESOLUTION_TOLERANCE,
    );
}
