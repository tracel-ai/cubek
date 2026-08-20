use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::{
    definition::{InterpolateMode, InterpolateOptions, TileSize},
    launch::{InterpolateStrategy, TileConfig, interpolate_tile_launch},
    routines::{
        BlueprintStrategy, GlobalMemoryRoutine, GlobalMemoryStrategy, SharedMemoryRoutine,
        SharedMemoryStrategy,
    },
};
use cubek_test_utils::TestInput;
use cubek_tile::Residence;

use super::{
    build_output_tensor, make_problem, output_host_f32, run_interpolate_global_test, validate_test,
};

const BILINEAR_TOLERANCE: f32 = 0.00001;
const BILINEAR_HIGH_RESOLUTION_TOLERANCE: f32 = 0.0001;

const TILE_SIZE: TileSize = TileSize::new(16, 16);

/// The geometry these tests validate against: the shape the plane-derived split produced
/// before every choice became the caller's.
const BASELINE: TileConfig = TileConfig::new(Residence::InPlace, 4, 2, 1);

#[test]
fn test_interpolate_tile_upsample_half_pixel() {
    let client = TestRuntime::client(&Default::default());
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    let problem = make_problem([2, 3, 5, 3], [7, 9], options);
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape())
        .uniform(42, -3.0, 3.0)
        .generate_with_f32_host_data();
    let expected = cubek_interpolate::eval::cpu_reference::cpu_reference_interpolate_from_host(
        &input_data,
        &problem.output_shape(),
        &options,
    );
    let output = build_output_tensor(&client, problem.output_shape().to_vec(), input.dtype);

    let result = interpolate_tile_launch(
        &client,
        input.clone().binding(),
        output.clone().binding(),
        options,
        input.dtype,
        BASELINE,
    );

    validate_test(
        result,
        output_host_f32(&client, output),
        expected,
        BILINEAR_TOLERANCE,
    );
}

/// A channel count that fills a plane, so the lanes divide the channels rather than the columns
/// (the lane split). That is a different space, and a different leaf shape.
#[test]
fn test_interpolate_tile_wide_channels() {
    let client = TestRuntime::client(&Default::default());
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    let problem = make_problem([2, 5, 4, 32], [9, 11], options);
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape())
        .uniform(7, -3.0, 3.0)
        .generate_with_f32_host_data();
    let expected = cubek_interpolate::eval::cpu_reference::cpu_reference_interpolate_from_host(
        &input_data,
        &problem.output_shape(),
        &options,
    );
    let output = build_output_tensor(&client, problem.output_shape().to_vec(), input.dtype);

    let result = interpolate_tile_launch(
        &client,
        input.clone().binding(),
        output.clone().binding(),
        options,
        input.dtype,
        BASELINE,
    );

    validate_test(
        result,
        output_host_f32(&client, output),
        expected,
        BILINEAR_TOLERANCE,
    );
}

/// A channel block of sixteen takes the `Const<4>` channel-line path while the bilinear gather
/// still clamps only the spatial coordinates at the image edge.
#[test]
fn test_interpolate_tile_vectorized_channels() {
    let client = TestRuntime::client(&Default::default());
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    let problem = make_problem([1, 3, 5, 16], [7, 9], options);
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape())
        .uniform(11, -3.0, 3.0)
        .generate_with_f32_host_data();
    let expected = cubek_interpolate::eval::cpu_reference::cpu_reference_interpolate_from_host(
        &input_data,
        &problem.output_shape(),
        &options,
    );
    let output = build_output_tensor(&client, problem.output_shape().to_vec(), input.dtype);

    let result = interpolate_tile_launch(
        &client,
        input.clone().binding(),
        output.clone().binding(),
        options,
        input.dtype,
        BASELINE,
    );

    validate_test(
        result,
        output_host_f32(&client, output),
        expected,
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 16],
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_shared_memory_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 16],
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [10, 10],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_shared_memory_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [10, 10],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [2, 2],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_shared_memory_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [2, 2],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_resize() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [8, 16],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_shared_memory_resize() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [8, 16],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_without_align_corners() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [16, 16],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
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
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_shared_memory_without_align_corners() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [16, 16],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
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
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_high_resolution() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [1, 89, 43, 13],
        [321, 75],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_HIGH_RESOLUTION_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_shared_memory_high_resolution() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [5, 89, 43, 13],
        [321, 75],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_HIGH_RESOLUTION_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_bhwc_512() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [1, 512, 512, 1],
        [1024, 1024],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_HIGH_RESOLUTION_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_shared_memory_bhwc_512() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [1, 512, 512, 1],
        [1024, 1024],
        InterpolateOptions::new(InterpolateMode::Bilinear),
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
        BILINEAR_HIGH_RESOLUTION_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_floor_global_flattened_tall_tile() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [1, 45, 14, 1],
        [4, 6],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );
    run_interpolate_global_test(
        client,
        5678,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {
                tile_size: TileSize::new(256, 1),
            }),
        ),
        BILINEAR_TOLERANCE,
    );
}

#[test]
fn test_interpolate_bilinear_shared_flattened_tall_tile() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [1, 46, 14, 1],
        [4, 6],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );
    run_interpolate_global_test(
        client,
        5678,
        -1.0,
        1.0,
        problem,
        InterpolateStrategy::SharedMemoryStrategy(
            BlueprintStrategy::<SharedMemoryRoutine>::Inferred(SharedMemoryStrategy {
                tile_size: TileSize::new(256, 1),
            }),
        ),
        BILINEAR_TOLERANCE,
    );
}
