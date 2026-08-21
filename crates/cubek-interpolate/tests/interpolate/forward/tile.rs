//! Validation of the experimental tile-backed forward path.

use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::{
    definition::{InterpolateMode, InterpolateOptions, NearestMode},
    eval::cpu_reference::cpu_reference_interpolate_from_host,
    launch::{TileConfig, interpolate_tile_launch},
};
use cubek_test_utils::{TestInput, assert_equals_approx};
use cubek_tile::Residence;

use super::{build_output_tensor, make_problem, output_host_f32, validate_test};

const TOLERANCE: f32 = 0.0001;

/// The geometry these tests validate against: the shape the plane-derived split produced
/// before every choice became the caller's.
const BASELINE: TileConfig = TileConfig::new(Residence::InPlace, 4, 2, 1);

fn tile_output(options: InterpolateOptions) {
    tile_output_with(options, BASELINE);
}

fn tile_output_with(options: InterpolateOptions, config: TileConfig) {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem([2, 8, 9, 4], [13, 15], options);
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape())
        .uniform(123, -3.0, 3.0)
        .generate_with_f32_host_data();
    let expected =
        cpu_reference_interpolate_from_host(&input_data, &problem.output_shape(), &options);
    let output = build_output_tensor(&client, problem.output_shape().to_vec(), input.dtype);
    let result = interpolate_tile_launch(
        &client,
        input.binding(),
        output.clone().binding(),
        options,
        output.dtype,
        config,
    );
    let actual = output_host_f32(&client, output);
    validate_test(result, actual, expected, TOLERANCE);
}

#[test]
fn test_interpolate_tile_staging_configurations() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for config in [
        TileConfig::new(Residence::Smem, 4, 2, 1),
        TileConfig::new(Residence::InPlace, 4, 2, 1),
    ] {
        tile_output_with(options, config);
    }
}

#[test]
fn test_interpolate_tile_geometry_configurations() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for config in [
        TileConfig::new(Residence::InPlace, 4, 2, 2),
        TileConfig::new(Residence::InPlace, 4, 4, 1),
        TileConfig::new(Residence::InPlace, 2, 2, 1),
        TileConfig::new(Residence::InPlace, 2, 4, 2),
    ] {
        tile_output_with(options, config);
    }
}

/// The channel run the lane takes. The problem's four channels solve to a block of four, so the
/// narrower splits are reachable only by stating them, and each one moves the accumulator's
/// innermost extent under the same contraction.
#[test]
fn test_interpolate_tile_channel_block_configurations() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for residence in [Residence::InPlace, Residence::Smem] {
        for block in [1, 2, 4] {
            tile_output_with(
                options,
                TileConfig::new(residence, 4, 2, 1).with_channel_block(block),
            );
        }
    }
}

#[test]
fn test_interpolate_tile_nearest_exact() {
    tile_output(
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact))
            .with_align_corners(false),
    );
}

#[test]
fn test_interpolate_tile_bicubic() {
    tile_output(InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false));
}

#[test]
fn test_interpolate_tile_lanczos3_identity() {
    let client = TestRuntime::client(&Default::default());
    let options = InterpolateOptions::new(InterpolateMode::Lanczos3);
    let problem = make_problem([2, 8, 9, 4], [8, 9], options);
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape())
        .uniform(123, -3.0, 3.0)
        .generate_with_f32_host_data();
    let expected =
        cpu_reference_interpolate_from_host(&input_data, &problem.output_shape(), &options);
    let output = build_output_tensor(&client, problem.output_shape().to_vec(), input.dtype);
    let result = interpolate_tile_launch(
        &client,
        input.binding(),
        output.clone().binding(),
        options,
        output.dtype,
        BASELINE,
    );
    let actual = output_host_f32(&client, output);

    result.unwrap();
    // At integral coordinates every non-central Lanczos weight is zero, so this validates the
    // six-tap path without exercising its intentionally unnormalized border behavior.
    assert_equals_approx(&actual, &expected, TOLERANCE)
        .as_test_outcome()
        .enforce();
}
