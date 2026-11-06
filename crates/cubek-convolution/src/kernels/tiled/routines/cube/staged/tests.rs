//! Tests for the cube staged convolution routine: which settings decline which shapes, and the
//! levels each prepares.

use super::*;
use crate::kernels::tiled::routines::test_support::{F32, cmma_only, nvidia, problems};
use crate::kernels::tiled::{
    CI, CMMA_STEP_ACROSS_TAPS, CONV_INSTRUCTION_UNOFFERED, ConvOperation, ConvProblem, OW, RW,
};
use cubek_microkernel::matmul::MatmulOperands;
use cubek_microkernel::matmul::plane::{PlaneFragmentInstruction, PlaneFragmentTransport};
use cubek_std::microkernel::KernelEstimate;
use cubek_std::routine::EstimatedProblem;
use cubek_std::routine::Routine;

const ROUTINE: CubeStagedConvRoutine = CubeStagedConvRoutine;

fn problem(name: &str) -> ConvProblem {
    problems()
        .into_iter()
        .find(|(known, _)| *known == name)
        .expect("a catalog problem")
        .1
}

/// Whole steps of channels run through cmma, and the lines setting reads wider than width one.
#[test]
fn cmma_serves_whole_channel_steps() {
    let name = "conv3x3_c64";
    let prepared = |settings| {
        ROUTINE
            .prepare(&nvidia(), &problem(name), &settings)
            .expect("the setting prepares")
            .0
    };
    assert!(matches!(
        prepared(CubeStagedConvRoutine::GRID_2X2)
            .microkernel
            .plane
            .instruction,
        PlaneFragmentInstruction::Cmma { .. }
    ));
    let widths = |settings| prepared(settings).widths;
    assert!(widths(CubeStagedConvRoutine::GRID_2X2).lhs > 1);
    assert_eq!(widths(CubeStagedConvRoutine::GRID_2X2.width_one()).lhs, 1);
}

/// A few channels make a step of taps, which cmma declines by name and the Mma settings serve, so
/// the first that serves is one of them.
#[test]
fn cmma_declines_a_step_of_taps_and_mma_serves_it() {
    let (hw, problem) = (nvidia(), problem("alexnet_like"));
    for settings in [
        CubeStagedConvRoutine::GRID_2X2,
        CubeStagedConvRoutine::GRID_1X2,
    ] {
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
}

/// An element the device has no instruction for, and an instruction it does not offer at all, are
/// declined by name.
#[test]
fn a_pair_the_device_lacks_declines() {
    let f32_problem = ConvProblem {
        elements: MatmulOperands::all(F32),
        ..problem("conv3x3_c64")
    };
    let declined =
        |hw, problem: &ConvProblem, settings| ROUTINE.prepare(&hw, problem, &settings).err();
    assert_eq!(
        declined(nvidia(), &f32_problem, CubeStagedConvRoutine::GRID_2X2),
        Some(CONV_INSTRUCTION_UNOFFERED)
    );
    assert_eq!(
        declined(
            cmma_only(),
            &problem("conv3x3_c64"),
            CubeStagedConvRoutine::GRID_2X2.mma()
        ),
        Some(CONV_INSTRUCTION_UNOFFERED)
    );
    assert_eq!(
        ROUTINE
            .prepare_first_served(&cmma_only(), &problem("alexnet_like"))
            .err(),
        Some(CONV_INSTRUCTION_UNOFFERED)
    );
}

/// The space is the problem's whatever the tiles, and a box wider than a narrow map overhangs it.
#[test]
fn the_space_is_the_problems_and_a_wide_box_overhangs_it() {
    let narrow = problem("narrow_width");
    for settings in [
        CubeStagedConvRoutine::GRID_2X2,
        CubeStagedConvRoutine::GRID_1X2,
    ] {
        let (blueprint, _) = ROUTINE
            .prepare(&nvidia(), &narrow, &settings)
            .expect("the setting prepares");
        assert_eq!(blueprint.partitioning.space().extent(OW), narrow.out[1]);
        assert!(blueprint.partitioning.overhangs(OW));
    }
}

/// A deep filter over a few channels walks taps: the channels round up to a power of two, which
/// the tiles overhang, and so does the filter's width.
#[test]
fn a_step_of_taps_overhangs_the_channels() {
    let (blueprint, _) = ROUTINE
        .prepare(
            &nvidia(),
            &problem("alexnet_like"),
            &CubeStagedConvRoutine::GRID_2X2.mma(),
        )
        .expect("Mma serves a step of taps");
    let partitioning = &blueprint.partitioning;
    assert_eq!(
        (
            partitioning.space().extent(RW),
            partitioning.space().extent(CI)
        ),
        (11, 3)
    );
    assert!(partitioning.overhangs(CI));
}

/// A weight backward takes its rows from the input's channels and its contraction from the output
/// positions, which the boxes overhang where they do not divide the map.
#[test]
fn a_weight_backward_contracts_the_output_positions() {
    let problem = ConvProblem {
        operation: ConvOperation::BackwardWeight,
        ..problem("conv3x3_c64")
    };
    let (blueprint, _) = ROUTINE
        .prepare(&nvidia(), &problem, &CubeStagedConvRoutine::GRID_2X2)
        .expect("the setting prepares");
    assert!(blueprint.partitioning.overhangs(OW));
    assert!(!blueprint.partitioning.overhangs(CI));
}

/// A map seven wide makes a weight backward step of two axes, which cmma serves as Mma does.
#[test]
fn a_weight_backward_over_a_narrow_map_is_served_by_cmma() {
    let problem = ConvProblem {
        operation: ConvOperation::BackwardWeight,
        ..problem("narrow_width")
    };
    let settings = CubeStagedConvRoutine::GRID_2X2;
    for settings in [settings, settings.mma()] {
        assert!(ROUTINE.prepare(&nvidia(), &problem, &settings).is_ok());
    }
}

/// A cube's estimate holds the staging its partitioning places, and the cubes perform at least
/// what the problem asks.
#[test]
fn every_setting_estimates_whole() {
    let hw = nvidia();
    let mut prepared = 0;
    for settings in ROUTINE.settings(&hw) {
        for (name, problem) in problems() {
            let Ok((blueprint, grid)) = ROUTINE.prepare(&hw, &problem, &settings) else {
                continue;
            };
            prepared += 1;
            let mut estimate = KernelEstimate::from(&grid);
            ROUTINE.estimate(&problem, &blueprint, &grid, &mut estimate);
            let staging = estimate.staging.as_ref().expect("a staging");
            assert!(estimate.work.is_some(), "{settings} on {name}: no work");
            assert!(
                staging.stages > 0 && !staging.reads.is_empty(),
                "{settings} on {name}"
            );
            let performed = staging.operations * estimate.cubes.expect("a static grid") as f64;
            let useful = problem.demand().compute_ops as f64;
            assert!(
                performed >= useful,
                "{settings} on {name}: {performed} < {useful}"
            );
        }
    }
    assert!(prepared > 0, "no setting prepares here");
}

/// Every setting the routine lists prepares on every problem, or declines with a refusal the
/// routine names.
#[test]
fn every_setting_prepares_or_declines_by_name() {
    let hw = nvidia();
    for settings in ROUTINE.settings(&hw) {
        for (name, problem) in problems() {
            let prepared = ROUTINE.prepare(&hw, &problem, &settings);
            let cmma_on_taps =
                settings.instruction == PlaneFragmentTransport::Cmma && name == "alexnet_like";
            assert!(
                prepared.is_ok() != cmma_on_taps,
                "{settings} on {name}: {:?}",
                prepared.err()
            );
        }
    }
}
