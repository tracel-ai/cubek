//! Validation of the forward kernel.

use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::{
    InterpolateBlueprint, InterpolateStrategy, Residence,
    definition::{InterpolateError, InterpolateMode, InterpolateOptions, NearestMode},
    eval::cpu_reference::cpu_reference_interpolate_from_host,
    interpolate,
};
use cubek_test_utils::{HostData, TestInput};

use super::super::{build_output_tensor, output_host_f32, validate_test};
use super::make_problem;

const TOLERANCE: f32 = 0.0001;

/// The geometry these tests validate against: the shape the plane-derived split produced
/// before every choice became the caller's.
const BASELINE: InterpolateBlueprint = InterpolateBlueprint::new(Residence::InPlace, 4, 2, 1);

fn kernel_output(options: InterpolateOptions) {
    kernel_output_with(options, BASELINE);
}

fn kernel_output_with(options: InterpolateOptions, blueprint: InterpolateBlueprint) {
    kernel_output_shaped(options, blueprint, 4);
}

fn kernel_output_shaped(
    options: InterpolateOptions,
    blueprint: InterpolateBlueprint,
    channels: usize,
) {
    kernel_output_on(options, blueprint, [2, 8, 9, channels], [13, 15]);
}

fn kernel_output_on(
    options: InterpolateOptions,
    blueprint: InterpolateBlueprint,
    input_shape: [usize; 4],
    output_size: [usize; 2],
) {
    kernel_output_using(
        options,
        InterpolateStrategy::Forced(blueprint),
        input_shape,
        output_size,
    );
}

fn kernel_output_using(
    options: InterpolateOptions,
    strategy: InterpolateStrategy,
    input_shape: [usize; 4],
    output_size: [usize; 2],
) {
    let (result, actual, expected) = kernel_run(options, strategy, input_shape, output_size);
    validate_test(result, actual, expected, TOLERANCE);
}

fn kernel_run(
    options: InterpolateOptions,
    strategy: InterpolateStrategy,
    input_shape: [usize; 4],
    output_size: [usize; 2],
) -> (Result<(), InterpolateError>, HostData, HostData) {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(input_shape, output_size, options);
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
        strategy,
        output.dtype,
    );
    let actual = output_host_f32(&client, output);
    (result, actual, expected)
}

/// The intents the selector resolves against this device. Whatever geometry it picks here has to
/// land on the reference, or a build that states nothing runs a kernel nothing validates.
#[test]
fn test_interpolate_kernel_inferred_strategies() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for strategy in [
        InterpolateStrategy::MaximizeThroughput,
        InterpolateStrategy::MinimizeLatency,
    ] {
        kernel_output_using(options, strategy, [2, 8, 9, 4], [13, 15]);
    }
}

/// A staged window no device holds, which a steep downsample reaches: the intent reads in place
/// instead and still lands on the reference, while the same geometry stated outright is refused.
///
/// The refusal is half the test. An autotune key buckets the extents this window is sized from, so
/// an intent that refused here would abort on a problem the tuner cached under a shape that fit.
#[test]
fn test_interpolate_kernel_intent_falls_back_when_the_stage_cannot_fit() {
    let options = InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(false);
    let input_shape = [1, 64, 64, 64];
    let output_size = [2, 2];

    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(input_shape, output_size, options);
    let blueprint =
        InterpolateStrategy::MinimizeLatency.blueprint(&client.properties().hardware, &problem);
    assert_eq!(blueprint.input_residence, Residence::Smem);

    let (refused, ..) = kernel_run(
        options,
        InterpolateStrategy::Forced(blueprint),
        input_shape,
        output_size,
    );
    assert!(
        matches!(
            refused,
            Err(InterpolateError::SharedMemoryLimitExceeded { .. })
        ),
        "the stage fits after all, so this shape no longer exercises the fallback: {refused:?}"
    );

    kernel_output_using(
        options,
        InterpolateStrategy::MinimizeLatency,
        input_shape,
        output_size,
    );
}

#[test]
fn test_interpolate_kernel_staging_configurations() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for blueprint in [
        InterpolateBlueprint::new(Residence::Smem, 4, 2, 1),
        InterpolateBlueprint::new(Residence::InPlace, 4, 2, 1),
    ] {
        kernel_output_with(options, blueprint);
    }
}

#[test]
fn test_interpolate_kernel_geometry_configurations() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for blueprint in [
        InterpolateBlueprint::new(Residence::InPlace, 1, 1, 1),
        InterpolateBlueprint::new(Residence::Smem, 1, 1, 1),
        InterpolateBlueprint::new(Residence::InPlace, 1, 2, 1),
        InterpolateBlueprint::new(Residence::InPlace, 4, 2, 2),
        InterpolateBlueprint::new(Residence::InPlace, 4, 4, 1),
        InterpolateBlueprint::new(Residence::InPlace, 2, 2, 1),
        InterpolateBlueprint::new(Residence::InPlace, 2, 4, 2),
    ] {
        kernel_output_with(options, blueprint);
    }
}

/// The channel run the lane takes. The problem's four channels solve to a block of four, so the
/// narrower splits are reachable only by stating them, and each one moves the accumulator's
/// innermost extent under the same contraction.
#[test]
fn test_interpolate_kernel_channel_block_configurations() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for residence in [Residence::InPlace, Residence::Smem] {
        for block in [1, 2, 4] {
            kernel_output_with(
                options,
                InterpolateBlueprint::new(residence, 4, 2, 1).with_channel_block(block),
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
fn test_interpolate_kernel_padded_channel_stage() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for channels in [3, 2] {
        for block in [channels, 4] {
            kernel_output_shaped(
                options,
                InterpolateBlueprint::new(Residence::Smem, 4, 2, 1).with_channel_block(block),
                channels,
            );
        }
    }
}

/// The padded stage under every filter: the tap count is what the contraction amortizes the
/// scalar sink writes over, so each `kc` has to land on the reference too.
#[test]
fn test_interpolate_kernel_padded_channel_stage_every_mode() {
    // Lanczos3 is left out: masked tap normalization requires the input window to remain in place.
    for mode in [
        InterpolateMode::Nearest(NearestMode::Exact),
        InterpolateMode::Bilinear,
        InterpolateMode::Bicubic,
    ] {
        kernel_output_shaped(
            InterpolateOptions::new(mode).with_align_corners(false),
            InterpolateBlueprint::new(Residence::Smem, 4, 2, 1).with_channel_block(4),
            3,
        );
    }
}

/// A channel axis the padded block covers in several whole blocks, the last part padding: the
/// mask has to drop the tail of *that* block rather than of the axis.
#[test]
fn test_interpolate_kernel_padded_channel_stage_multi_block() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    kernel_output_shaped(
        options,
        InterpolateBlueprint::new(Residence::Smem, 4, 2, 1).with_channel_block(4),
        6,
    );
}

/// Every filter under both coordinate transforms.
///
/// `align_corners` selects the transform, and it is a comptime input the kernel specializes on,
/// so a mode landing on the reference under one setting says nothing about the other. `Floor` and
/// `Exact` round the same coordinate differently, which is the other half of the same choice.
#[test]
fn test_interpolate_kernel_every_mode_and_transform() {
    for mode in [
        InterpolateMode::Nearest(NearestMode::Floor),
        InterpolateMode::Nearest(NearestMode::Exact),
        InterpolateMode::Bilinear,
        InterpolateMode::Bicubic,
        InterpolateMode::Lanczos3,
    ] {
        for align_corners in [true, false] {
            kernel_output(InterpolateOptions::new(mode).with_align_corners(align_corners));
        }
    }
}

/// The directions the transform treats differently. The configuration tests above only ever
/// upsample: an upsample re-reads input rows across the output rows drawn from them, a downsample
/// skips the rows no tap window reaches, and an identity has to map every coordinate back onto
/// itself rather than merely near it.
#[test]
fn test_interpolate_kernel_resampling_directions() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for (input_shape, output_size) in [([2, 16, 17, 4], [7, 9]), ([2, 8, 9, 4], [8, 9])] {
        kernel_output_on(options, BASELINE, input_shape, output_size);
    }
}

/// A channel axis wide enough that the lanes cover the channels and never ride the columns, which
/// is the opposite end of the split from the padded stages above.
#[test]
fn test_interpolate_kernel_wide_channel_axis() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    kernel_output_shaped(options, BASELINE, 32);
}

/// Lanczos3 mapping a shape onto itself: the widest window, with every tap in place.
#[test]
fn test_interpolate_kernel_lanczos3_identity() {
    kernel_output_on(
        InterpolateOptions::new(InterpolateMode::Lanczos3),
        BASELINE,
        [2, 8, 9, 4],
        [8, 9],
    );
}
