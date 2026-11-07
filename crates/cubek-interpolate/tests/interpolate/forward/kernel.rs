//! Validation of the forward kernel.

use cubecl::tile::launch::Refusal;
use cubek_interpolate::{
    InterpolateKernel, InterpolateStrategy,
    definition::{InterpolateError, InterpolateMode, InterpolateOptions, NearestMode},
    eval::cpu_reference::cpu_reference_interpolate_from_host,
    interpolate,
    kernels::{CubeInterpolateRoutineSettings, select},
};
use cubek_test_utils::{HostData, TestInput};

use super::super::{build_output_tensor, output_host_f32, validate_test};
use super::make_problem;

const TOLERANCE: f32 = 0.0001;

/// The geometry these tests validate against: the shape the plane-derived split produced
/// before every choice became the caller's.
const BASELINE: InterpolateKernel = cube(4, 2, 1);

/// The in-place routine at `planes` a cube of `rows` rows, `cols` columns a unit.
const fn cube(planes: usize, rows: usize, cols: usize) -> InterpolateKernel {
    InterpolateKernel::Cube(CubeInterpolateRoutineSettings::new(planes, rows, cols))
}

/// The staged routine at `planes` a cube of `rows` rows, `cols` columns a unit.
const fn staged(planes: usize, rows: usize, cols: usize) -> InterpolateKernel {
    InterpolateKernel::CubeStaged(CubeInterpolateRoutineSettings::new(planes, rows, cols))
}

/// `kernel` with the unit's channel run pinned at `block`.
fn with_channel_block(kernel: InterpolateKernel, block: usize) -> InterpolateKernel {
    match kernel {
        InterpolateKernel::Cube(settings) => {
            InterpolateKernel::Cube(settings.with_channel_block(block))
        }
        InterpolateKernel::CubeStaged(settings) => {
            InterpolateKernel::CubeStaged(settings.with_channel_block(block))
        }
    }
}

fn kernel_output(options: InterpolateOptions) {
    kernel_output_with(options, BASELINE);
}

fn kernel_output_with(options: InterpolateOptions, kernel: InterpolateKernel) {
    kernel_output_shaped(options, kernel, 4);
}

fn kernel_output_shaped(
    options: InterpolateOptions,
    kernel: InterpolateKernel,
    channels: usize,
) {
    kernel_output_on(options, kernel, [2, 8, 9, channels], [13, 15]);
}

fn kernel_output_on(
    options: InterpolateOptions,
    kernel: InterpolateKernel,
    input_shape: [usize; 4],
    output_size: [usize; 2],
) {
    kernel_output_using(
        options,
        InterpolateStrategy::Specific(kernel),
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
    let client = cubecl::test_device().client();
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

/// The routines at the settings the selector picks on this device. Whatever geometry it picks
/// here has to land on the reference, or a build that states nothing runs a kernel nothing
/// validates.
#[test]
fn test_interpolate_kernel_selected_strategies() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for strategy in [InterpolateStrategy::Cube, InterpolateStrategy::CubeStaged] {
        kernel_output_using(options, strategy, [2, 8, 9, 4], [13, 15]);
    }
}

/// A staged window no device holds, which a steep downsample reaches: the staged strategy reads
/// in place instead and still lands on the reference, while the staged routine named outright is
/// refused.
///
/// The refusal is half the test. An autotune key buckets the extents this window is sized from, so
/// a strategy that refused here would abort on a problem the tuner cached under a shape that fit.
#[test]
fn test_interpolate_kernel_staged_strategy_reads_in_place_when_the_stage_cannot_fit() {
    let options = InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(false);
    let input_shape = [1, 64, 64, 64];
    let output_size = [2, 2];

    let client = cubecl::test_device().client();
    let problem = make_problem(input_shape, output_size, options);

    // A CPU has no shared memory to stage in, so the staged routine declines there and the
    // in-place one runs. The overflow is a GPU-only path.
    if client.properties().hardware.num_cpu_cores.is_none() {
        let staged = select(
            InterpolateStrategy::CubeStaged,
            &client.properties().hardware,
            &problem,
        )[0];
        let (refused, ..) = kernel_run(
            options,
            InterpolateStrategy::Specific(staged),
            input_shape,
            output_size,
        );
        assert!(
            matches!(
                refused,
                Err(InterpolateError::Refused(
                    Refusal::SharedMemoryPastDevice { .. }
                ))
            ),
            "the stage fits after all, so this shape no longer exercises the fallback: {refused:?}"
        );
    }

    kernel_output_using(
        options,
        InterpolateStrategy::CubeStaged,
        input_shape,
        output_size,
    );
}

#[test]
fn test_interpolate_kernel_staging_configurations() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for kernel in [staged(4, 2, 1), cube(4, 2, 1)] {
        kernel_output_with(options, kernel);
    }
}

#[test]
fn test_interpolate_kernel_geometry_configurations() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for kernel in [
        cube(1, 1, 1),
        staged(1, 1, 1),
        cube(1, 2, 1),
        cube(4, 2, 2),
        cube(4, 4, 1),
        cube(2, 2, 1),
        cube(2, 4, 2),
    ] {
        kernel_output_with(options, kernel);
    }
}

/// The channel run the unit takes. The problem's four channels solve to a block of four, so the
/// narrower splits are reachable only by stating them, and each one moves the accumulator's
/// innermost extent under the same contraction.
#[test]
fn test_interpolate_kernel_channel_block_configurations() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for kernel in [cube(4, 2, 1), staged(4, 2, 1)] {
        for block in [1, 2, 4] {
            kernel_output_with(options, with_channel_block(kernel, block));
        }
    }
}

/// The padded stage. An `NHWC` tensor at `C = 3` starts every row at element `3r`, so no 4-wide
/// line ever lands on a row boundary and the operand must be read scalar. A shared-memory stage
/// has no such constraint: pinning the channel block to `4` pads the axis to one whole line, and
/// the contraction runs `4` wide against an output still addressed one scalar at a time. The
/// fourth unit is padding, dropped by the sink's own overhang mask, so the result has to equal the
/// reference exactly as the unpadded block of `3` does.
#[test]
fn test_interpolate_kernel_padded_channel_stage() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    for channels in [3, 2] {
        for block in [channels, 4] {
            kernel_output_shaped(
                options,
                with_channel_block(staged(4, 2, 1), block),
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
            with_channel_block(staged(4, 2, 1), 4),
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
        with_channel_block(staged(4, 2, 1), 4),
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

/// A channel axis wide enough that the units cover the channels and never ride the columns, which
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

/// An input extent of `1`. Under `align_corners` the transform spans `input - 1` cells, so the
/// source coordinate along that axis holds still and the input is read broadcast over the output
/// rows drawn from it. Every filter is run because the taps the degenerate axis reaches past the
/// single row are the boundary's to serve, and how far they reach is the filter's radius.
#[test]
fn test_interpolate_kernel_degenerate_input_extent() {
    for mode in [
        InterpolateMode::Nearest(NearestMode::Floor),
        InterpolateMode::Nearest(NearestMode::Exact),
        InterpolateMode::Bilinear,
        InterpolateMode::Bicubic,
        InterpolateMode::Lanczos3,
    ] {
        for align_corners in [true, false] {
            let options = InterpolateOptions::new(mode).with_align_corners(align_corners);
            // One row, then one column, so neither axis is exercised only by the other's turn.
            kernel_output_on(options, BASELINE, [2, 1, 9, 4], [1, 15]);
            kernel_output_on(options, BASELINE, [2, 8, 1, 4], [13, 1]);
            // The extent held at 1 while the other axis resamples, which is the shape a burn
            // 1-D interpolation takes: [N, C, 1, W] upsampled along W alone.
            kernel_output_on(options, BASELINE, [2, 1, 6, 4], [1, 9]);
        }
    }
}
