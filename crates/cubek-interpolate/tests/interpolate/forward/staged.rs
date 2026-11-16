//! The staged routine: each cube stages its windows' taps in shared memory before its units read
//! them.

use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions, NearestMode};
use cubek_interpolate::kernels::CubeInterpolateRoutineSettings;
use cubek_test_utils::ScopeUnderTest;

use crate::interpolate::forward::{BASELINE, BILINEAR, Forward, ForwardCase, Resampling};

/// The baseline geometry, and a cube of one plane of one row.
#[test]
fn test_interpolate_kernel_staging_configurations() {
    Forward.check(
        [BASELINE, CubeInterpolateRoutineSettings::new(1, 1, 1)]
            .map(|settings| ForwardCase::staged(settings, Resampling::usual(BILINEAR))),
    );
}

/// The channel run the unit takes, every split of the problem's four channels.
#[test]
fn test_interpolate_kernel_channel_block_configurations() {
    Forward.check([1, 2, 4].map(|block| {
        ForwardCase::staged(
            BASELINE.with_channel_block(block),
            Resampling::usual(BILINEAR),
        )
    }));
}

/// The padded stage. An `NHWC` tensor at `C = 3` starts every row at element `3r`, so no 4-wide
/// line ever lands on a row boundary and the operand must be read scalar. A shared-memory stage
/// has no such constraint: pinning the channel block to `4` pads the axis to one whole line, and
/// the contraction runs `4` wide against an output still addressed one scalar at a time. The
/// fourth unit is padding, dropped by the sink's own overhang mask, so the result has to equal the
/// reference exactly as the unpadded block of `3` does.
#[test]
fn test_interpolate_kernel_padded_channel_stage() {
    let cases: Vec<_> = [3, 2]
        .into_iter()
        .flat_map(|channels| {
            [channels, 4].map(|block| {
                ForwardCase::staged(
                    BASELINE.with_channel_block(block),
                    Resampling::usual(BILINEAR).with_channels(channels),
                )
            })
        })
        .collect();
    Forward.check(cases);
}

/// The padded stage under every filter: the tap count is what the contraction amortizes the
/// scalar sink writes over, so each `kc` has to land on the reference too. Lanczos3 is left out:
/// masked tap normalization requires the input window to remain in place.
#[test]
fn test_interpolate_kernel_padded_channel_stage_every_mode() {
    Forward.check(
        [
            InterpolateMode::Nearest(NearestMode::Exact),
            InterpolateMode::Bilinear,
            InterpolateMode::Bicubic,
        ]
        .map(|mode| {
            ForwardCase::staged(
                BASELINE.with_channel_block(4),
                Resampling::usual(InterpolateOptions::new(mode).with_align_corners(false))
                    .with_channels(3),
            )
        }),
    );
}

/// A channel axis the padded block covers in several whole blocks, the last part padding: the
/// mask has to drop the tail of *that* block rather than of the axis.
#[test]
fn test_interpolate_kernel_padded_channel_stage_multi_block() {
    Forward.check([ForwardCase::staged(
        BASELINE.with_channel_block(4),
        Resampling::usual(BILINEAR).with_channels(6),
    )]);
}
