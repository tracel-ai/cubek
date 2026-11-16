//! Tests for the cube staged convolution routine: which settings decline which shapes, and the
//! levels each prepares.

use super::*;
use crate::components::ConvolutionOperation;
use crate::eval::benchmarks::Conv2dProblem;
use crate::eval::benchmarks::settings;
use crate::kernels::tiled::routines::test_support::{
    cmma_8x32_only, cmma_only, mma_only, nvidia, problems,
};
use crate::kernels::tiled::{CMMA_STEP_ACROSS_TAPS, CONV_INSTRUCTION_UNOFFERED, ConvProblem};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::plane::PlaneFragmentInstruction;
use cubek_std::devices::topology;
use cubek_std::microkernel::KernelEstimate;
use cubek_std::routine::EstimatedProblem;
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

/// A few channels make a step of taps, which the fragment API declines by name; the tile picks
/// the manual mma only where the device offers no fragment API, and that serves it.
#[test]
fn cmma_declines_a_step_of_taps_and_mma_serves_it() {
    let problem = problem("few_channels_deep_filter");
    for settings in [settings::F2X2_P2X2_T16, settings::F1X2_P1X2_T16] {
        let prepared = ROUTINE.prepare(&nvidia(), &problem, &settings);
        assert_eq!(prepared.err(), Some(CMMA_STEP_ACROSS_TAPS));
    }
    let (blueprint, _) = ROUTINE
        .prepare_first_served(&mma_only(), &problem)
        .expect("a setting serves");
    assert!(matches!(
        blueprint.microkernel.plane.instruction,
        PlaneFragmentInstruction::Mma { .. }
    ));
    assert_eq!(
        ROUTINE.prepare_first_served(&cmma_only(), &problem).err(),
        Some(CMMA_STEP_ACROSS_TAPS)
    );
    assert!(ROUTINE.prepare_first_served(&nvidia(), &problem).is_err());
}

/// A cmma fragment's store cannot be cut short, so its output is masked and bounces; a manual mma
/// writes its own cells and masks only where the extent needs it.
#[test]
fn only_a_cmma_output_is_masked_to_bounce() {
    let problem = problem("channels_64");
    let masks_out = |hw| {
        ROUTINE
            .prepare(&hw, &problem, &settings::F2X2_P2X2_T16)
            .expect("the setting prepares")
            .0
            .masks_out
    };
    assert!(masks_out(nvidia()));
    assert!(!masks_out(mma_only()));
}

/// A split holding more planes than a cube of the device spawns is refused by name before any
/// launch, at either plane width, on a device whose tile picks the eight-row instruction.
#[test]
fn planes_across_past_the_cube_are_refused_by_name() {
    let narrow = problem("map_7_wide");
    let settings = settings::F1X1_P1X2_T8.across(7);
    for (plane_width, most_units) in [(32, 256), (64, 512)] {
        let mut hw = cmma_8x32_only();
        let device = topology(&mut hw);
        device.plane_size_min = plane_width;
        device.plane_size_max = plane_width;
        device.max_units_per_cube = most_units;
        assert!(matches!(
            ROUTINE.prepare(&hw, &narrow, &settings).err(),
            Some(Refusal::CubePastDevice { .. })
        ));
    }
    assert!(
        ROUTINE
            .prepare(&cmma_8x32_only(), &narrow, &settings)
            .is_ok()
    );
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
    assert!(ROUTINE.prepare(&mma_only(), &problem, &settings).is_ok());
}

/// The catalog problems as the launch states them.
fn staged_problems() -> Vec<(&'static str, ConvProblem)> {
    problems()
        .into_iter()
        .map(|(name, problem)| (name, ConvProblem::from(&problem)))
        .collect()
}

/// The refusals the routine states for a setting its device lists: a step of taps through cmma, an
/// instruction the device lacks, and a stage past the device's shared memory.
fn named(refusal: &Refusal) -> bool {
    *refusal == CMMA_STEP_ACROSS_TAPS
        || *refusal == CONV_INSTRUCTION_UNOFFERED
        || matches!(refusal, Refusal::SharedMemoryPastDevice { .. })
}

/// Every listed setting prepares each problem or is declined by name, and some setting serves
/// each problem: on a device whose tile picks the manual mma every one, on one whose tile picks
/// the fragment API all but those that take a step of taps, which every setting declines by name.
#[test]
fn every_setting_prepares_or_declines_by_name() {
    for hw in [nvidia(), mma_only()] {
        for (name, problem) in staged_problems() {
            for settings in ROUTINE.settings(&hw) {
                let Err(refusal) = ROUTINE.prepare(&hw, &problem, &settings) else {
                    continue;
                };
                assert!(named(&refusal), "{settings} on {name}: {refusal:?}");
            }
            let served = ROUTINE.prepare_first_served(&hw, &problem);
            assert!(
                served.is_ok() || served.as_ref().is_err_and(named),
                "no setting serves {name}"
            );
        }
    }
    for (name, problem) in staged_problems() {
        assert!(
            ROUTINE.prepare_first_served(&mma_only(), &problem).is_ok(),
            "no setting serves {name} through the manual mma"
        );
    }
}

/// A cube's estimate holds the staging its partitioning places, and the cubes perform at least
/// what the problem asks.
#[test]
fn every_setting_estimates_whole() {
    let hw = nvidia();
    let mut prepared = 0;
    for settings in ROUTINE.settings(&hw) {
        for (name, problem) in staged_problems() {
            let Ok((blueprint, grid)) = ROUTINE.prepare(&hw, &problem, &settings) else {
                continue;
            };
            prepared += 1;
            let mut estimate = KernelEstimate::from(&grid);
            ROUTINE.estimate(&problem, &blueprint, &grid, &mut estimate);
            let staging = estimate.staging.as_ref().expect("a staging");
            assert!(estimate.work.is_some(), "{settings} on {name}: no work");
            assert!(!staging.reads.is_empty(), "{settings} on {name}");
            let performed = staging.operations * estimate.cubes.expect("a static grid") as f64;
            let useful = problem.demand().compute_ops as f64;
            assert!(performed >= useful, "{settings} on {name}");
        }
    }
    assert!(prepared > 0, "no setting prepares here");
}
