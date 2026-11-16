//! The strategies a build states nothing for: each routine at the settings the selector picks on
//! this device, which has to land on the reference, or a build that states nothing runs a kernel
//! nothing validates.

use cubecl::tile::launch::Refusal;
use cubek_interpolate::definition::{
    InterpolateForwardProblem, InterpolateMode, InterpolateOptions,
};
use cubek_interpolate::kernels::select;
use cubek_interpolate::{InterpolateKernel, InterpolateStrategy};
use cubek_test_utils::ScopeUnderTest;

use crate::interpolate::forward::{Forward, ForwardCase, Resampling};

/// Both routines at the settings the selector picks.
#[test]
fn test_interpolate_kernel_selected_strategies() {
    let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    Forward.check(
        [InterpolateStrategy::Cube, InterpolateStrategy::CubeStaged].map(|strategy| ForwardCase {
            strategy,
            resampling: Resampling::usual(options),
        }),
    );
}

/// A staged window no device holds, which a steep downsample reaches: the staged strategy reads
/// in place instead and still lands on the reference, while the staged routine named outright is
/// refused.
///
/// The refusal is half the test. An autotune key buckets the extents this window is sized from, so
/// a strategy that refused here would abort on a problem the tuner cached under a shape that fit.
#[test]
fn test_interpolate_kernel_staged_strategy_reads_in_place_when_the_stage_cannot_fit() {
    let resampling = Resampling {
        options: InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(false),
        input: [1, 64, 64, 64],
        output: [2, 2],
    };
    let client = cubecl::test_device().client();

    // A CPU has no shared memory to stage in, so the staged routine declines there and the
    // in-place one runs. The overflow is a GPU-only path.
    if client.properties().hardware.num_cpu_cores.is_none() {
        let staged: InterpolateKernel = select(
            InterpolateStrategy::CubeStaged,
            &client.properties().hardware,
            &InterpolateForwardProblem::from_input_output_shapes(
                &resampling.input.into(),
                &resampling.output,
                resampling.options,
            ),
        )[0];
        let refused = Forward.launch(&client, &ForwardCase::specific(staged, resampling));
        assert!(
            matches!(refused, Err(Refusal::SharedMemoryPastDevice { .. })),
            "the stage fits after all, so this shape no longer exercises the fallback: {:?}",
            refused.err()
        );
    }

    Forward.check([ForwardCase {
        strategy: InterpolateStrategy::CubeStaged,
        resampling,
    }]);
}
