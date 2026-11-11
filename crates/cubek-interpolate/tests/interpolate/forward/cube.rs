//! The in-place routine: each cube's units read their windows' taps where they lie.

use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions};
use cubek_interpolate::kernels::CubeInterpolateRoutineSettings;
use cubek_test_utils::ScopeUnderTest;

use crate::interpolate::forward::{BASELINE, BILINEAR, Forward, ForwardCase, MODES, Resampling};

/// Planes a cube, rows a plane and columns a unit, the splits the geometry takes.
#[test]
fn test_interpolate_kernel_geometry_configurations() {
    Forward.check(
        [
            (1, 1, 1),
            (1, 2, 1),
            (4, 2, 2),
            (4, 4, 1),
            (2, 2, 1),
            (2, 4, 2),
        ]
        .map(|(planes, rows, cols)| {
            ForwardCase::cube(
                CubeInterpolateRoutineSettings::new(planes, rows, cols),
                Resampling::usual(BILINEAR),
            )
        }),
    );
}

/// The channel run the unit takes. The problem's four channels solve to a block of four, so the
/// narrower splits are reachable only by stating them, and each one moves the accumulator's
/// innermost extent under the same contraction.
#[test]
fn test_interpolate_kernel_channel_block_configurations() {
    Forward.check([1, 2, 4].map(|block| {
        ForwardCase::cube(
            BASELINE.with_channel_block(block),
            Resampling::usual(BILINEAR),
        )
    }));
}

/// Every filter under both coordinate transforms.
///
/// `align_corners` selects the transform, and it is a comptime input the kernel specializes on,
/// so a mode landing on the reference under one setting says nothing about the other. `Floor` and
/// `Exact` round the same coordinate differently, which is the other half of the same choice.
#[test]
fn test_interpolate_kernel_every_mode_and_transform() {
    let cases: Vec<_> = MODES
        .into_iter()
        .flat_map(|mode| {
            [true, false].map(|align_corners| {
                ForwardCase::cube(
                    BASELINE,
                    Resampling::usual(
                        InterpolateOptions::new(mode).with_align_corners(align_corners),
                    ),
                )
            })
        })
        .collect();
    Forward.check(cases);
}

/// The directions the transform treats differently. The configuration tests above only ever
/// upsample: an upsample re-reads input rows across the output rows drawn from them, a downsample
/// skips the rows no tap window reaches, and an identity has to map every coordinate back onto
/// itself rather than merely near it.
#[test]
fn test_interpolate_kernel_resampling_directions() {
    Forward.check(
        [([2, 16, 17, 4], [7, 9]), ([2, 8, 9, 4], [8, 9])].map(|(input, output)| {
            ForwardCase::cube(
                BASELINE,
                Resampling {
                    options: BILINEAR,
                    input,
                    output,
                },
            )
        }),
    );
}

/// A channel axis wide enough that the units cover the channels and never ride the columns, which
/// is the opposite end of the split from the staged routine's padded stages.
#[test]
fn test_interpolate_kernel_wide_channel_axis() {
    Forward.check([ForwardCase::cube(
        BASELINE,
        Resampling::usual(BILINEAR).with_channels(32),
    )]);
}

/// Lanczos3 mapping a shape onto itself: the widest window, with every tap in place.
#[test]
fn test_interpolate_kernel_lanczos3_identity() {
    Forward.check([ForwardCase::cube(
        BASELINE,
        Resampling {
            options: InterpolateOptions::new(InterpolateMode::Lanczos3),
            input: [2, 8, 9, 4],
            output: [8, 9],
        },
    )]);
}

/// An input extent of `1`. Under `align_corners` the transform spans `input - 1` cells, so the
/// source coordinate along that axis holds still and the input is read broadcast over the output
/// rows drawn from it. Every filter is run because the taps the degenerate axis reaches past the
/// single row are the boundary's to serve, and how far they reach is the filter's radius.
#[test]
fn test_interpolate_kernel_degenerate_input_extent() {
    // One row, then one column, so neither axis is exercised only by the other's turn; then the
    // extent held at 1 while the other axis resamples, which is the shape a burn 1-D
    // interpolation takes: [N, C, 1, W] upsampled along W alone.
    let shapes = [
        ([2, 1, 9, 4], [1, 15]),
        ([2, 8, 1, 4], [13, 1]),
        ([2, 1, 6, 4], [1, 9]),
    ];
    let cases: Vec<_> = MODES
        .into_iter()
        .flat_map(|mode| {
            [true, false].into_iter().flat_map(move |align_corners| {
                let options = InterpolateOptions::new(mode).with_align_corners(align_corners);
                shapes.map(|(input, output)| {
                    ForwardCase::cube(
                        BASELINE,
                        Resampling {
                            options,
                            input,
                            output,
                        },
                    )
                })
            })
        })
        .collect();
    Forward.check(cases);
}
