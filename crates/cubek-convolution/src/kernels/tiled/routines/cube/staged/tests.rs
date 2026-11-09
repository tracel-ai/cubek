//! Tests for the cube staged convolution routine: which settings decline which shapes, and the
//! levels each prepares.

use super::*;
use crate::kernels::tiled::routines::test_support::{F32, cmma_only, nvidia, problems};
use crate::kernels::tiled::{
    CI, CMMA_STEP_ACROSS_TAPS, CONV_INSTRUCTION_UNOFFERED, ConvOperation, ConvProblem, OH, OW, RW,
};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::MatmulOperands;
use cubek_microkernel::matmul::plane::{PlaneFragmentInstruction, PlaneFragmentTransport};
use cubek_std::devices::topology;
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
        prepared(CubeStagedConvRoutine::F2X2_P2X2)
            .microkernel
            .plane
            .instruction,
        PlaneFragmentInstruction::Cmma { .. }
    ));
    let widths = |settings| prepared(settings).widths;
    assert!(widths(CubeStagedConvRoutine::F2X2_P2X2).lhs > 1);
    assert_eq!(widths(CubeStagedConvRoutine::F2X2_P2X2.width_one()).lhs, 1);
}

/// A few channels make a step of taps, which cmma declines by name and the Mma settings serve, so
/// the first that serves is one of them.
#[test]
fn cmma_declines_a_step_of_taps_and_mma_serves_it() {
    let (hw, problem) = (nvidia(), problem("alexnet_like"));
    for settings in [
        CubeStagedConvRoutine::F2X2_P2X2,
        CubeStagedConvRoutine::F1X2_P1X2,
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
    assert_eq!(
        ROUTINE.prepare_first_served(&cmma_only(), &problem).err(),
        Some(CMMA_STEP_ACROSS_TAPS)
    );
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
        declined(nvidia(), &f32_problem, CubeStagedConvRoutine::F2X2_P2X2),
        Some(CONV_INSTRUCTION_UNOFFERED)
    );
    assert_eq!(
        declined(
            cmma_only(),
            &problem("conv3x3_c64"),
            CubeStagedConvRoutine::F2X2_P2X2.mma()
        ),
        Some(CONV_INSTRUCTION_UNOFFERED)
    );
    assert_eq!(
        ROUTINE
            .prepare_first_served(&cmma_only(), &f32_problem)
            .err(),
        Some(CONV_INSTRUCTION_UNOFFERED)
    );
}

/// The eight row setting takes the 8x32 instruction, where a plain one takes the largest, and a
/// device without it declines by name.
#[test]
fn the_eight_row_setting_takes_the_8x32_instruction() {
    let narrow = problem("narrow_width");
    let rows = |settings| {
        let (blueprint, _) = ROUTINE
            .prepare(&nvidia(), &narrow, &settings)
            .expect("the setting prepares");
        let [(_, m), (_, n), _] = blueprint.microkernel.plane.instruction.shape();
        (m, n)
    };
    assert_eq!(rows(CubeStagedConvRoutine::F1X1_P1X2_T8), (8, 32));
    assert_eq!(rows(CubeStagedConvRoutine::F2X2_P2X2), (16, 16));
    assert_eq!(
        ROUTINE
            .prepare(&cmma_only(), &narrow, &CubeStagedConvRoutine::F1X1_P1X2_T8)
            .err(),
        Some(CONV_INSTRUCTION_UNOFFERED)
    );
}

/// The space is the problem's whatever the tiles, and a box wider than a narrow map overhangs it.
#[test]
fn the_space_is_the_problems_and_a_wide_box_overhangs_it() {
    let narrow = problem("narrow_width");
    for settings in [
        CubeStagedConvRoutine::F2X2_P2X2,
        CubeStagedConvRoutine::F1X2_P1X2,
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
            &CubeStagedConvRoutine::F2X2_P2X2.mma(),
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
        .prepare(&nvidia(), &problem, &CubeStagedConvRoutine::F2X2_P2X2)
        .expect("the setting prepares");
    assert!(blueprint.partitioning.overhangs(OW));
    assert!(!blueprint.partitioning.overhangs(CI));
}

/// Planes across cover that many rows of the map a cube, the last box overhanging where they do not
/// divide it. Each costs the lhs stage its rows and the rhs stage nothing.
#[test]
fn planes_across_share_the_rhs_stage_and_overhang_the_rows() {
    let narrow = problem("narrow_width");
    let prepared = |settings| {
        ROUTINE
            .prepare(&nvidia(), &narrow, &settings)
            .expect("the setting prepares")
            .0
            .partitioning
    };
    let plain = prepared(CubeStagedConvRoutine::F1X1_P1X2_T8);
    assert!(!plain.overhangs(OH));
    let bytes = |partitioning: &crate::kernels::tiled::ConvPartitioning| {
        partitioning.stage_bytes(&narrow.elements)
    };
    for across in [2, 4, 7] {
        let split = prepared(CubeStagedConvRoutine::F1X1_P1X2_T8.across(across));
        assert_eq!(split.planes(), plain.planes() * across);
        assert_eq!(split.overhangs(OH), narrow.out[0] % across != 0);
        assert!(bytes(&split) > bytes(&plain));
        assert!(bytes(&split) < across * bytes(&plain));
    }
}

/// A split holding more planes than a cube of the device spawns is refused by name before any
/// launch, at either plane width.
#[test]
fn planes_across_past_the_cube_are_refused_by_name() {
    let narrow = problem("narrow_width");
    let settings = CubeStagedConvRoutine::F1X1_P1X2_T8.across(7);
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

/// A map seven wide makes a weight backward step of two axes, which cmma serves as Mma does.
#[test]
fn a_weight_backward_over_a_narrow_map_is_served_by_cmma() {
    let problem = ConvProblem {
        operation: ConvOperation::BackwardWeight,
        ..problem("narrow_width")
    };
    let settings = CubeStagedConvRoutine::F2X2_P2X2;
    for settings in [settings, settings.mma()] {
        assert!(ROUTINE.prepare(&nvidia(), &problem, &settings).is_ok());
    }
}

/// The catalog's 64 channel shape read as a data backward of `channels_out` output channels,
/// which it contracts.
fn data_backward(channels_out: usize) -> ConvProblem {
    ConvProblem {
        operation: ConvOperation::BackwardData,
        channels_out,
        ..problem("conv3x3_c64")
    }
}

/// A data backward reads its gradient back through the taps, so a box that overhangs the output
/// channels reads it one element wide, where whole boxes read lines.
#[test]
fn a_data_backward_over_overhanging_channels_reads_one_element_wide() {
    let widths = |channels_out| {
        let settings = CubeStagedConvRoutine::F2X2_P2X2;
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
    let settings = CubeStagedConvRoutine::F2X2_P2X2;
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

/// Every setting the routine lists prepares on a problem or declines it by name, and a step of
/// taps is declined through cmma by its own refusal and served by some setting of the list.
#[test]
fn every_setting_prepares_or_declines_by_name() {
    let hw = nvidia();
    for (name, problem) in problems() {
        for settings in ROUTINE.settings(&hw) {
            let prepared = ROUTINE.prepare(&hw, &problem, &settings);
            let cmma_on_taps = settings.instruction.transport == PlaneFragmentTransport::Cmma
                && name == "alexnet_like";
            if cmma_on_taps {
                assert_eq!(prepared.err(), Some(CMMA_STEP_ACROSS_TAPS), "{settings}");
            }
        }
        assert!(
            ROUTINE.prepare_first_served(&hw, &problem).is_ok(),
            "no setting serves {name}"
        );
    }
}
