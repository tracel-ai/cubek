use super::{ConvCandidate, ConvTuneKey, candidates};
use crate::components::ConvolutionOperation;
use crate::eval::benchmarks::{Conv2dElement, Conv2dProblem};
use crate::kernels::tiled::routines::test_support::{cmma_8x32_only, dense, nvidia};
use crate::kernels::tiled::{
    ConvProblem, ConvSettings, CubeStagedConvRoutine, UnitRegistersConvRoutine,
};
use cubek_microkernel::matmul::cube::MatmulPlaneLayout;
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings as Fragments;
use cubek_std::Hardware;
use cubek_std::autotune::{CandidatePriority, TuneCandidate, priority_of};
use cubek_std::devices::topology;
use cubek_std::routine::Routine;
use std::cmp::Reverse;

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

/// The listed settings in the order a round tries them for `key`, the repeats left out.
fn ranked(hw: &Hardware, key: &ConvTuneKey) -> Vec<ConvCandidate> {
    let list = ConvCandidate::list(hw);
    let mut placed: Vec<_> = (0..list.len())
        .filter_map(|position| priority_of(&list, position, key).map(|p| (p, position)))
        .collect();
    placed.sort_by_key(|&(priority, position)| (Reverse(priority), position));
    placed
        .into_iter()
        .map(|(_, position)| list[position].clone())
        .collect()
}

fn rows(candidate: &ConvCandidate, key: &ConvTuneKey) -> usize {
    let (blueprint, _) = candidate
        .blueprint(key)
        .expect("the candidate prepares the key");
    blueprint.partitioning.block().rows
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
    assert_eq!(
        priority(CubeStagedConvRoutine::F7X1_P1X2_T8),
        CandidatePriority::FIRST
    );
    assert_eq!(priority(base), CandidatePriority::FIRST);
}

/// A map 7 wide and 7 tall is held by 8 rows, and by planes across its rows that share each
/// weight stage: such a candidate is raced in the top band.
#[test]
fn a_map_7_by_7_races_an_8_row_candidate_with_an_across_split_in_the_top_band() {
    let key = key(wide_map_7());
    let top: Vec<_> = ranked(&cmma_8x32_only(), &key)
        .into_iter()
        .filter(|candidate| candidate.priority(&key) == CandidatePriority::FIRST)
        .collect();
    assert!(
        top.iter()
            .any(|c| rows(c, &key) == 8 && c.strategy().across.get() > 1),
        "{:?}",
        top.iter()
            .map(|c| c.strategy().to_string())
            .collect::<Vec<_>>()
    );
}

/// The same setting over a grid that cannot fill half the device is raced after the one over
/// a grid that fills it.
#[test]
fn a_grid_short_of_half_the_device_is_in_a_lower_band() {
    let hw = cmma_8x32_only();
    let settings = CubeStagedConvRoutine::F1X1_P1X1_T8_A7;
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
    assert!(priority(overhanging) < priority(CubeStagedConvRoutine::F7X1_P1X2_T8));
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
    let base = CubeStagedConvRoutine::F2X2_P2X2_T16;
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

/// A key states the problem it describes: the key of that problem is the key.
#[test]
fn a_key_states_the_problem_it_describes() {
    let key = key(Conv2dProblem {
        input_shape: [5, 64, 58, 58],
        ..dense()
    });
    assert_eq!(ConvTuneKey::from(&ConvProblem::from(&key)), key);
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

/// The list and the set raced over it come out the same every time they are built.
#[test]
fn the_list_and_its_set_are_built_in_one_order() {
    let hw = nvidia();
    let names = |list: &[ConvCandidate]| {
        list.iter()
            .map(|candidate| candidate.settings().to_string())
            .collect::<Vec<_>>()
    };
    let (first, second) = (ConvCandidate::list(&hw), ConvCandidate::list(&hw));
    assert_eq!(names(&first), names(&second));
    let checksum =
        |list: Vec<ConvCandidate>| candidates(list.into(), hw.clone()).compute_checksum();
    assert_eq!(checksum(first), checksum(second));
}

/// Whether the unit routine, the floor under every tuned launch, serves `problem` on `hw`.
fn floor_serves(hw: &Hardware, problem: &Conv2dProblem) -> bool {
    UnitRegistersConvRoutine
        .prepare_first_served(hw, &ConvProblem::from(problem))
        .is_ok()
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

/// A device whose plane width is a range lists nothing to race, so the tuned launch floors on the
/// unit routine, which asks nothing of the plane.
#[test]
fn a_ranged_plane_width_floors_the_tuned_launch_on_the_unit_routine() {
    let mut hw = nvidia();
    let device = topology(&mut hw);
    (device.plane_size_min, device.plane_size_max) = (32, 64);
    assert!(ConvCandidate::list(&hw).is_empty());
    assert!(floor_serves(&hw, &dense()));
}

/// An element the device has no matrix instruction for leaves every listed setting declining the
/// key, so the tuned launch floors on the unit routine.
#[test]
fn an_f32_problem_without_f32_instructions_floors_on_the_unit_routine() {
    let hw = nvidia();
    let problem = Conv2dProblem {
        element: Conv2dElement::F32,
        ..dense()
    };
    let key = key(problem.clone());
    assert!(
        ConvCandidate::list(&hw)
            .iter()
            .all(|candidate| candidate.blueprint(&key).is_none())
    );
    assert!(floor_serves(&hw, &problem));
}
