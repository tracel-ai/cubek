//! Validation of the forward kernel.

use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::{
    definition::{InterpolateMode, InterpolateOptions, NearestMode},
    eval::cpu_reference::cpu_reference_interpolate_from_host,
    interpolate,
    kernel::TileConfig,
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
    tile_output_shaped(options, config, 4);
}

fn tile_output_shaped(options: InterpolateOptions, config: TileConfig, channels: usize) {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem([2, 8, 9, channels], [13, 15], options);
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape())
        .uniform(123, -3.0, 3.0)
        .generate_with_f32_host_data();
    let expected =
        cpu_reference_interpolate_from_host(&input_data, &problem.output_shape(), &options);
    let output = build_output_tensor(&client, problem.output_shape().to_vec(), input.dtype);
    let result = interpolate(
        &client,
        input.binding(),
        output.clone().binding(),
        options,
        config,
        output.dtype,
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
        TileConfig::new(Residence::InPlace, 1, 1, 1),
        TileConfig::new(Residence::Smem, 1, 1, 1),
        TileConfig::new(Residence::InPlace, 1, 2, 1),
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

/// The padded stage. An `NHWC` tensor at `C = 3` starts every row at element `3r`, so no 4-wide
/// line ever lands on a row boundary and the operand must be read scalar. A shared-memory stage
/// has no such constraint: pinning the channel block to `4` pads the axis to one whole line, and
/// the contraction runs `4` wide against an output still addressed one scalar at a time. The
/// fourth lane is padding, dropped by the sink's own overhang mask, so the result has to equal the
/// reference exactly as the unpadded block of `3` does.
#[test]
fn test_interpolate_tile_padded_channel_stage() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for channels in [3, 2] {
        for block in [channels, 4] {
            tile_output_shaped(
                options,
                TileConfig::new(Residence::Smem, 4, 2, 1).with_channel_block(block),
                channels,
            );
        }
    }
}

/// The padded stage under every filter: the tap count is what the contraction amortizes the
/// scalar sink writes over, so each `kc` has to land on the reference too.
#[test]
fn test_interpolate_tile_padded_channel_stage_every_mode() {
    // Lanczos3 is left out: masked tap normalization requires the input window to remain in place.
    for mode in [
        InterpolateMode::Nearest(NearestMode::Exact),
        InterpolateMode::Bilinear,
        InterpolateMode::Bicubic,
    ] {
        tile_output_shaped(
            InterpolateOptions::new(mode).with_align_corners(false),
            TileConfig::new(Residence::Smem, 4, 2, 1).with_channel_block(4),
            3,
        );
    }
}

/// A channel axis the padded block covers in several whole blocks, the last part padding: the
/// mask has to drop the tail of *that* block rather than of the axis.
#[test]
fn test_interpolate_tile_padded_channel_stage_multi_block() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    tile_output_shaped(
        options,
        TileConfig::new(Residence::Smem, 4, 2, 1).with_channel_block(4),
        6,
    );
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
fn test_interpolate_tile_lanczos3() {
    tile_output(InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(false));
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
    let result = interpolate(
        &client,
        input.binding(),
        output.clone().binding(),
        options,
        BASELINE,
        output.dtype,
    );
    let actual = output_host_f32(&client, output);

    result.unwrap();
    assert_equals_approx(&actual, &expected, TOLERANCE)
        .as_test_outcome()
        .enforce();
}
