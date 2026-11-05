//! Tests for the plane fragment convolution routine: which settings decline which shapes, and
//! the levels each prepares.

use super::*;
use crate::kernels::tiled::routines::test_support::{cmma_only, nvidia, problems};
use crate::kernels::tiled::{CMMA_STEP_ACROSS_TAPS, CONV_INSTRUCTION_UNOFFERED, OW};
use cubek_microkernel::matmul::plane::PlaneFragmentTransport;
use cubek_std::microkernel::KernelEstimate;
use cubek_std::routine::{EstimatedProblem, Routine};

const ROUTINE: PlaneFragmentConvRoutine = PlaneFragmentConvRoutine;

/// Every setting prepares on the problems whose steps are whole channels, where each plane holds
/// the whole box it walks, and a setting through cmma declines a step of taps by name.
#[test]
fn settings_decline_a_step_of_taps_through_cmma_only() {
    let hw = nvidia();
    for settings in ROUTINE.settings(&hw) {
        for (name, problem) in problems() {
            let prepared = ROUTINE.prepare(&hw, &problem, &settings);
            match (name, settings.instruction) {
                ("alexnet_like", PlaneFragmentTransport::Cmma) => {
                    assert_eq!(prepared.err(), Some(CMMA_STEP_ACROSS_TAPS))
                }
                _ => assert!(
                    prepared.is_ok(),
                    "{settings} on {name}: {:?}",
                    prepared.err()
                ),
            }
        }
    }
}

/// A device offering cmma alone declines the Mma settings by name, and the first that serves a
/// problem of whole channel steps is the cmma one.
#[test]
fn a_device_without_mma_declines_its_settings() {
    let (hw, [_, (_, whole_steps), ..]) = (cmma_only(), problems());
    assert_eq!(
        ROUTINE
            .prepare(&hw, &whole_steps, &PlaneFragmentConvRoutine::GRID_2X2.mma())
            .err(),
        Some(CONV_INSTRUCTION_UNOFFERED)
    );
    let (blueprint, _) = ROUTINE
        .prepare_first_served(&hw, &whole_steps)
        .expect("cmma serves whole steps");
    assert_eq!(blueprint.microkernel.settings.along_m.get(), 2);
}

/// The space is the problem's whatever the tiles, and a plane's box overhangs a narrow map.
#[test]
fn a_plane_overhangs_a_narrow_map() {
    let (_, narrow) = problems()
        .into_iter()
        .find(|(name, _)| *name == "narrow_width")
        .expect("a catalog problem");
    for settings in [
        PlaneFragmentConvRoutine::GRID_2X2,
        PlaneFragmentConvRoutine::GRID_1X2,
    ] {
        let (blueprint, _) = ROUTINE
            .prepare(&nvidia(), &narrow, &settings)
            .expect("the setting prepares");
        assert_eq!(blueprint.partitioning.space().extent(OW), narrow.out[1]);
        assert!(blueprint.partitioning.overhangs(OW));
    }
}

/// A plane's estimate holds the staging its partitioning places, and the planes perform at least
/// what the problem asks.
#[test]
fn every_setting_estimates_whole() {
    let hw = nvidia();
    for settings in ROUTINE.settings(&hw) {
        for (name, problem) in problems() {
            let Ok((blueprint, grid)) = ROUTINE.prepare(&hw, &problem, &settings) else {
                continue;
            };
            let mut estimate = KernelEstimate::from(&grid);
            ROUTINE.estimate(&problem, &blueprint, &grid, &mut estimate);
            let staging = estimate.staging.as_ref().expect("a staging");
            let performed = staging.operations * estimate.cubes.expect("a static grid") as f64;
            let useful = problem.demand().compute_ops as f64;
            assert!(
                performed >= useful,
                "{settings} on {name}: {performed} < {useful}"
            );
        }
    }
}
