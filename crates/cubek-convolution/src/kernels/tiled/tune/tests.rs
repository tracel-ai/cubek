use super::{ConvCandidate, ConvTuneKey};
use crate::components::ConvolutionOperation;
use crate::eval::benchmarks::Conv2dProblem;
use crate::eval::benchmarks::settings;
use crate::kernels::tiled::routines::test_support::{cmma_8x32_only, dense, nvidia};
use crate::kernels::tiled::{ConvProblem, ConvSettings, CubeStagedConvRoutine};
use cubek_microkernel::matmul::cube::MatmulPlaneLayout;
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings as Fragments;
use cubek_std::autotune::{CandidatePriority, TuneCandidate};
use cubek_std::routine::Routine;

fn key(problem: Conv2dProblem) -> ConvTuneKey {
    ConvTuneKey::from(&ConvProblem::from(&problem))
}

/// `WIDE` batches of a map 56 wide, held by a few rows.
fn wide_map_56() -> Conv2dProblem {
    Conv2dProblem {
        input_shape: [WIDE, 64, 12, 58],
        weight_shape: [128, 64, 3, 3],
        ..dense()
    }
}

/// `WIDE` batches of a map 7 wide and 7 tall.
fn wide_map_7() -> Conv2dProblem {
    Conv2dProblem {
        input_shape: [WIDE, 64, 9, 9],
        ..dense()
    }
}

/// Every grid is wide enough to fill the device, so only the boxes' fit sets the band, and
/// stays within the 65535 cubes a device takes along z.
const WIDE: usize = 4096;

/// Seven 8-row instructions cover a map 56 wide whole, and the base's boxes overhang it
/// little: both are raced in the top band.
#[test]
fn a_map_56_wide_races_the_56_row_candidate_and_the_base_together() {
    let (hw, key) = (cmma_8x32_only(), key(wide_map_56()));
    let priority = |settings| ConvCandidate::new(settings, &hw).priority(&key);
    let base = ConvSettings::new(8, Fragments::grid(2, 2, 1), MatmulPlaneLayout::new(2, 2));
    assert_eq!(priority(settings::F7X1_P1X2_T8), CandidatePriority::FIRST);
    assert_eq!(priority(base), CandidatePriority::FIRST);
}

/// The same setting over a grid that cannot fill half the device is raced after the one over
/// a grid that fills it.
#[test]
fn a_grid_short_of_half_the_device_is_in_a_lower_band() {
    let hw = cmma_8x32_only();
    let settings = settings::F1X1_P1X1_T8_A7;
    let priority = |key| ConvCandidate::new(settings, &hw).priority(&key);
    assert!(
        priority(key(Conv2dProblem {
            input_shape: [1, 64, 9, 9],
            ..wide_map_7()
        })) < priority(key(wide_map_7()))
    );
}

/// Boxes covering at least twice the map are raced after those that tile it.
#[test]
fn boxes_covering_twice_the_map_are_in_a_lower_band() {
    let (hw, key) = (cmma_8x32_only(), key(wide_map_56()));
    let priority = |settings| ConvCandidate::new(settings, &hw).priority(&key);
    let overhanging = ConvSettings::new(8, Fragments::grid(16, 1, 1), MatmulPlaneLayout::new(1, 2));
    assert!(priority(overhanging) < priority(settings::F7X1_P1X2_T8));
}

/// A contraction of one stage compiles one launch at either ring, so the round keeps one; one
/// of many stages keeps both.
#[test]
fn a_double_ring_over_one_stage_is_the_single_ring_launch() {
    let hw = nvidia();
    let launch = |settings: ConvSettings, key: &ConvTuneKey| {
        ConvCandidate::new(settings, &hw)
            .blueprint(key)
            .expect("the candidate prepares the key")
    };
    let base = settings::F2X2_P2X2_T16;
    let one_stage = key(Conv2dProblem {
        input_shape: [1, 16, 8, 8],
        weight_shape: [32, 16, 1, 1],
        ..dense()
    });
    assert_eq!(
        launch(base, &one_stage),
        launch(base.double_ring(), &one_stage)
    );
    let many = key(dense());
    assert_ne!(launch(base, &many), launch(base.double_ring(), &many));
}

/// A view starting at an odd element narrows the key's width, and the launch the key's problem
/// prepares is the one the real problem prepares, so the candidate tuned is the one run.
#[test]
fn a_key_holds_the_width_a_view_start_leaves() {
    let hw = nvidia();
    let aligned = ConvProblem::from(&dense());
    let mut shifted = aligned.clone();
    shifted.start_bytes.lhs = 2;
    let (aligned_key, shifted_key) = (ConvTuneKey::from(&aligned), ConvTuneKey::from(&shifted));
    assert_ne!(aligned_key, shifted_key);
    for (problem, key) in [(&aligned, &aligned_key), (&shifted, &shifted_key)] {
        for settings in ConvCandidate::list(&hw) {
            let prepared = CubeStagedConvRoutine.prepare(&hw, problem, settings.settings());
            let real = prepared
                .ok()
                .map(|(blueprint, grid)| (blueprint, grid.launch.cube_dim));
            assert_eq!(real, settings.blueprint(key), "{}", settings.settings());
        }
    }
}

/// Batches that fill the grid's third dimension with the positions across are keyed exactly,
/// since settings may disagree within a bucket there; below that they share their power of two.
#[test]
fn a_batch_is_keyed_exactly_only_where_it_fills_the_grid() {
    let batched = |batch| {
        key(Conv2dProblem {
            input_shape: [batch, 64, 58, 58],
            ..dense()
        })
    };
    assert_eq!(batched(5), batched(7));
    assert_ne!(batched(1200), batched(1201));
}

/// A weight backward's batch is contracted, never a grid axis, so it is bucketed however large.
#[test]
fn a_contracted_batch_is_keyed_by_its_bucket() {
    let batched = |batch| {
        key(Conv2dProblem {
            operation: ConvolutionOperation::BackwardWeight,
            input_shape: [batch, 64, 58, 58],
            ..dense()
        })
    };
    assert_eq!(batched(1200), batched(1201));
}
