//! Tests for the cube staged convolution routine: which settings decline which shapes, and the
//! levels each prepares.

use super::*;
use crate::components::ConvolutionOperation;
use crate::eval::benchmarks::Conv2dProblem;
use crate::eval::benchmarks::settings;
use crate::kernels::tiled::routines::test_support::{cmma_only, nvidia, problems};
use crate::kernels::tiled::{CMMA_STEP_ACROSS_TAPS, ConvProblem};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::plane::PlaneFragmentInstruction;
use cubek_std::devices::topology;
use cubek_std::routine::Routine;

const ROUTINE: CubeStagedConvRoutine = CubeStagedConvRoutine;

fn conv2d(name: &str) -> Conv2dProblem {
    problems()
        .into_iter()
        .find(|(known, _)| *known == name)
        .expect("a catalog problem")
        .1
}

fn problem(name: &str) -> ConvProblem {
    ConvProblem::from(&conv2d(name))
}

/// A few channels make a step of taps, which cmma declines by name and the Mma settings serve, so
/// the first that serves is one of them.
#[test]
fn cmma_declines_a_step_of_taps_and_mma_serves_it() {
    let (hw, problem) = (nvidia(), problem("few_channels_deep_filter"));
    for settings in [settings::F2X2_P2X2_T16, settings::F1X2_P1X2_T16] {
        let prepared = ROUTINE.prepare(&hw, &problem, &settings);
        assert_eq!(prepared.err(), Some(CMMA_STEP_ACROSS_TAPS));
    }
    let (blueprint, _) = ROUTINE
        .prepare_first_served(&hw, &problem)
        .expect("a setting serves");
    assert!(matches!(
        blueprint.microkernel.plane.instruction,
        PlaneFragmentInstruction::Mma { .. }
    ));
    assert_eq!(
        ROUTINE.prepare_first_served(&cmma_only(), &problem).err(),
        Some(CMMA_STEP_ACROSS_TAPS)
    );
}

/// A cmma fragment's store cannot be cut short, so its output is masked and bounces; a manual mma
/// writes its own cells and masks only where the extent needs it.
#[test]
fn only_a_cmma_output_is_masked_to_bounce() {
    let problem = problem("channels_64");
    let masks_out = |settings| {
        ROUTINE
            .prepare(&nvidia(), &problem, &settings)
            .expect("the setting prepares")
            .0
            .masks_out
    };
    assert!(masks_out(settings::F2X2_P2X2_T16));
    assert!(!masks_out(settings::F2X2_P2X2_T16.mma()));
}

/// A split holding more planes than a cube of the device spawns is refused by name before any
/// launch, at either plane width.
#[test]
fn planes_across_past_the_cube_are_refused_by_name() {
    let narrow = problem("map_7_wide");
    let settings = settings::F1X1_P1X2_T8.across(7);
    for (plane_width, most_units) in [(32, 256), (64, 512)] {
        let mut hw = nvidia();
        let device = topology(&mut hw);
        device.plane_size_min = plane_width;
        device.plane_size_max = plane_width;
        device.max_units_per_cube = most_units;
        assert!(matches!(
            ROUTINE.prepare(&hw, &narrow, &settings).err(),
            Some(Refusal::CubePastDevice { .. })
        ));
    }
    assert!(ROUTINE.prepare(&nvidia(), &narrow, &settings).is_ok());
}

/// The catalog's 64 channel shape read as a data backward of `channels_out` output channels,
/// which it contracts.
fn data_backward(channels_out: usize) -> ConvProblem {
    let forward = conv2d("channels_64");
    let [_, channels_in, kh, kw] = forward.weight_shape;
    ConvProblem::from(&Conv2dProblem {
        operation: ConvolutionOperation::BackwardData,
        weight_shape: [channels_out, channels_in, kh, kw],
        ..forward
    })
}

/// A data backward reads its gradient back through the taps, so a box that overhangs the output
/// channels reads it one element wide, where whole boxes read vectors.
#[test]
fn a_data_backward_over_overhanging_channels_reads_one_element_wide() {
    let widths = |channels_out| {
        let settings = settings::F2X2_P2X2_T16;
        ROUTINE
            .prepare(&nvidia(), &data_backward(channels_out), &settings)
            .expect("cmma serves whole steps of channels")
            .0
            .widths
    };
    assert!(widths(32).lhs > 1);
    assert_eq!(widths(24).lhs, 1);
}

/// A few output channels make a data backward step of taps and channels, which cmma declines by
/// name and Mma serves.
#[test]
fn cmma_declines_a_data_backward_step_across_taps_and_mma_serves_it() {
    let problem = data_backward(3);
    let settings = settings::F2X2_P2X2_T16;
    assert_eq!(
        ROUTINE.prepare(&nvidia(), &problem, &settings).err(),
        Some(CMMA_STEP_ACROSS_TAPS)
    );
    assert!(
        ROUTINE
            .prepare(&nvidia(), &problem, &settings.mma())
            .is_ok()
    );
}
