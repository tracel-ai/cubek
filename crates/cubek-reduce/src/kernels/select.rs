//! The reduce family's rule: which routine runs a problem when nothing is measured.

use crate::kernels::{
    BATCH, CubeStagedReduceRoutine, HypercubeLastArrivalReduceRoutine,
    HypercubeLastArrivalReduceSettings, PlaneShuffleReduceRoutine, ReduceBlueprint, ReduceKernel,
    ReduceRoutine, ReduceRoutineProblem, ReduceSplitExtents, ReduceTensors,
    UnitDirectReduceRoutine, launch_kernel,
};
use cubecl::tile::launch::Decline;
use cubek_microkernel::reduce::ReduceCarrier;
use cubek_std::Hardware;
use cubek_std::routine::{Grid, Routine};

/// `tensors` reduced by the first routine, in the order the rule gives the problem the call's Tune Key
/// stands for ([`ReduceRoutineProblem::keyed`]) on `hw`, that does not decline the call, read top to
/// bottom:
///
/// 1. Rows enough that a unit each fills the device, each row short enough that a unit walks it
///    alone as fast as a plane would: the unit routine.
/// 2. Rows enough that a plane each fills the device: the plane routine, whose lanes share a row
///    without meeting in shared memory.
/// 3. Rows a cube each would leave the device idle, and a split that fills it: the hypercube, at
///    its first setting whose cubes fill the device, or at the one with the most cubes.
/// 4. Otherwise the cube routine.
/// 5. Where every one of these declines the call, the unit routine, which serves every problem.
///
/// # Errors
///
/// Why the unit routine cannot run the problem here, which no device refuses.
pub(crate) fn select(hw: &Hardware, tensors: &ReduceTensors) -> Result<(), Decline> {
    let keyed = tensors.problem().keyed();
    if let Some((settings, blueprint)) =
        filling_the_device::<UnitDirectReduceRoutine, _>(hw, &keyed)
        && unit_keeps_up(hw, &keyed, blueprint.lane_count)
        && launched::<UnitDirectReduceRoutine, _>(hw, tensors, &settings)
    {
        return Ok(());
    }
    if let Some((settings, _)) = filling_the_device::<PlaneShuffleReduceRoutine, _>(hw, &keyed)
        && launched::<PlaneShuffleReduceRoutine, _>(hw, tensors, &settings)
    {
        return Ok(());
    }
    if let Some(settings) = hypercube_filling_the_device(hw, &keyed)
        && launched::<HypercubeLastArrivalReduceRoutine, _>(hw, tensors, &settings)
    {
        return Ok(());
    }
    if let Some((settings, _, _)) = first_served::<CubeStagedReduceRoutine, _>(hw, &keyed)
        && launched::<CubeStagedReduceRoutine, _>(hw, tensors, &settings)
    {
        return Ok(());
    }
    launch_kernel(ReduceKernel::Unit, hw, tensors)
}

/// Whether a unit reading a row alone, `lane_count` values a line, keeps up with a plane reading it
/// together. Across the reduced dim it does: neighbouring units read neighbouring rows, so their
/// reads meet. Along it, a unit walks its row line after line where a plane takes a line a lane,
/// so it keeps up only while the row holds no more lines than a plane has lanes, or where the
/// device runs no plane routine.
fn unit_keeps_up(hw: &Hardware, problem: &ReduceRoutineProblem, lane_count: usize) -> bool {
    if problem.lines_along() == BATCH {
        return true;
    }
    hw.plane_dim()
        .ok()
        .is_none_or(|lanes| problem.reduced_extent().div_ceil(lane_count) <= lanes as usize)
}

/// The setting of the hypercube's launch where a box of rows a cube leaves the device idle and
/// splitting the rows along the reduced dim gives more cubes: the first of its settings whose cubes
/// fill the device, or the one with the most cubes where none does. A top-k takes the settings
/// whose units read the most lines only: the last cube merges a row's top-k lists one after
/// another, `k` inserts a part, so shorter parts cost more in the merge than their extra cubes
/// save. `None` where the device does not say how wide it is, the boxes already fill it, or every
/// row is one part.
fn hypercube_filling_the_device(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
) -> Option<HypercubeLastArrivalReduceSettings> {
    let mut most_cubes = None;
    let most_lines_only = matches!(
        problem.operation.carrier(),
        ReduceCarrier::TopK { .. } | ReduceCarrier::TopValues { .. }
    );
    let settings = HypercubeLastArrivalReduceRoutine::settings(hw)
        .into_iter()
        .filter(|settings| {
            !most_lines_only
                || settings.unit_line_count
                    == HypercubeLastArrivalReduceSettings::UNIT_LINE_COUNTS[0]
        });
    for settings in settings {
        let Ok((blueprint, _)) = HypercubeLastArrivalReduceRoutine::prepare(hw, problem, &settings)
        else {
            continue;
        };
        let units = blueprint.microkernel.part().units() as u32;
        let fill = hw.cubes_to_fill(units)? as usize;
        let boxes = problem.row_box_count_at_lane_count(blueprint.lane_count);
        let parts = ReduceSplitExtents::new(problem, &blueprint.partitioning).part_count as usize;
        if boxes >= fill {
            return None;
        }
        // A row in one part is the cube routine's, not a split.
        let Some(cubes) = (parts >= 2).then_some(boxes * parts) else {
            continue;
        };
        if cubes >= fill {
            return Some(settings);
        }
        if most_cubes.as_ref().is_none_or(|(most, _)| cubes > *most) {
            most_cubes = Some((cubes, settings));
        }
    }
    most_cubes.map(|(_, settings)| settings)
}

/// The first of `R`'s settings that prepares `problem` on `hw`, with the launch it prepares.
fn first_served<R: ReduceRoutine<M>, M>(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
) -> Option<(R::Settings, ReduceBlueprint<M>, Grid)> {
    R::settings(hw).into_iter().find_map(|settings| {
        R::prepare(hw, problem, &settings)
            .ok()
            .map(|(blueprint, grid)| (settings, blueprint, grid))
    })
}

/// [`first_served`]'s setting and Blueprint, where its launch fills the device.
fn filling_the_device<R: ReduceRoutine<M>, M>(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
) -> Option<(R::Settings, ReduceBlueprint<M>)> {
    first_served::<R, M>(hw, problem)
        .filter(|(_, _, grid)| hw.halvings_short_of_filling(&grid.launch) == 0)
        .map(|(settings, blueprint, _)| (settings, blueprint))
}

/// Whether `R` at `settings` launched on `tensors`: `false` where it declines them.
fn launched<R: ReduceRoutine<M>, M>(
    hw: &Hardware,
    tensors: &ReduceTensors,
    settings: &R::Settings,
) -> bool {
    R::prepare(hw, tensors.problem(), settings)
        .and_then(|(blueprint, grid)| R::launch(tensors, blueprint, grid))
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ReducePartials;
    use crate::kernels::testing::{
        BufferLaunch, BufferScopeChecker, given_too_little_scratch, sum_of_one_long_row,
    };
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;
    use cubek_test_utils::ScopeChecker;
    use std::sync::Arc;

    /// Across the reduced axis a unit always keeps up: each lane its own row.
    #[test]
    fn a_unit_keeps_up_across_the_reduced_axis() {
        let problem = ReduceRoutineProblem::dense(&[100_000, 8], 0, ReduceOperation::Sum);
        assert!(unit_keeps_up(&devices::m2_pro(), &problem, 1));
    }

    /// Values a line holds.
    const LANE_COUNT: usize = 4;

    /// Along it, a unit keeps up while the row holds no more lines than a plane has lanes.
    #[test]
    fn a_unit_keeps_up_along_a_row_no_longer_than_a_plane() {
        let hw = devices::m2_pro();
        let lanes = hw.plane_dim().expect("an M2 Pro commits to a plane width") as usize;
        let short = ReduceRoutineProblem::dense(&[8, lanes * LANE_COUNT], 1, ReduceOperation::Sum);
        assert!(unit_keeps_up(&hw, &short, LANE_COUNT));
    }

    /// Along it, a unit falls behind on a row longer than a plane's lanes' lines.
    #[test]
    fn a_unit_falls_behind_along_a_row_longer_than_a_plane() {
        let hw = devices::m2_pro();
        let lanes = hw.plane_dim().expect("an M2 Pro commits to a plane width") as usize;
        let long =
            ReduceRoutineProblem::dense(&[8, lanes * LANE_COUNT + 1], 1, ReduceOperation::Sum);
        assert!(!unit_keeps_up(&hw, &long, LANE_COUNT));
    }

    /// On a device with no committed plane width, nothing outruns the unit.
    #[test]
    fn a_unit_keeps_up_where_no_plane_routine_runs() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let long = ReduceRoutineProblem::dense(&[8, 100_000], 1, ReduceOperation::Sum);
        assert!(unit_keeps_up(&hw, &long, 1));
    }

    /// A sum of `shape` along its last axis, given the scratch the hypercube takes.
    fn sum_given_scratch(shape: &[usize]) -> ReduceRoutineProblem {
        ReduceRoutineProblem {
            partials: Some(ReducePartials::UNBOUNDED),
            ..ReduceRoutineProblem::dense(shape, shape.len() - 1, ReduceOperation::Sum)
        }
    }

    /// A row of few enough values that only the hypercube's shorter parts fill an M2 Pro, and of
    /// enough that its longest parts split it.
    const SPLIT_VALUE_COUNT: usize = 400_000;

    #[test]
    fn the_hypercube_takes_its_first_parts_that_fill_the_device() {
        let hw = devices::m2_pro();
        let problem = sum_given_scratch(&[1, SPLIT_VALUE_COUNT]);
        let settings = hypercube_filling_the_device(&hw, &problem).expect("the row splits");
        assert_eq!(
            settings.unit_line_count,
            HypercubeLastArrivalReduceSettings::UNIT_LINE_COUNTS[1]
        );
    }

    #[test]
    fn the_hypercube_takes_its_most_cubes_where_none_fills_the_device() {
        let [_, _, shortest] = HypercubeLastArrivalReduceSettings::UNIT_LINE_COUNTS;
        let problem = sum_given_scratch(&[1, 8192]);
        let settings =
            hypercube_filling_the_device(&devices::m2_pro(), &problem).expect("the row splits");
        assert_eq!(settings.unit_line_count, shortest);
    }

    #[test]
    fn a_top_k_takes_the_hypercubes_longest_parts() {
        let problem = ReduceRoutineProblem {
            operation: ReduceOperation::TopK { k: 2 },
            ..sum_given_scratch(&[1, SPLIT_VALUE_COUNT])
        };
        let settings =
            hypercube_filling_the_device(&devices::m2_pro(), &problem).expect("the row splits");
        assert_eq!(
            settings.unit_line_count,
            HypercubeLastArrivalReduceSettings::UNIT_LINE_COUNTS[0]
        );
    }

    #[test]
    fn rows_that_fill_the_device_are_not_split() {
        let problem = sum_given_scratch(&[10_000, SPLIT_VALUE_COUNT]);
        assert_eq!(
            hypercube_filling_the_device(&devices::m2_pro(), &problem),
            None
        );
    }

    #[test]
    fn a_row_of_one_part_is_not_split() {
        let problem = sum_given_scratch(&[1, 16]);
        assert_eq!(
            hypercube_filling_the_device(&devices::m2_pro(), &problem),
            None
        );
    }

    #[test]
    fn no_row_is_split_on_a_device_of_unstated_width() {
        let mut hw = devices::m2_pro();
        Arc::make_mut(&mut hw.properties)
            .hardware
            .num_streaming_multiprocessors = None;
        let problem = sum_given_scratch(&[1, SPLIT_VALUE_COUNT]);
        assert_eq!(hypercube_filling_the_device(&hw, &problem), None);
    }

    #[test]
    fn the_first_setting_that_prepares_is_served() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[8, 300], 1, ReduceOperation::Sum);
        let served = first_served::<UnitDirectReduceRoutine, _>(&hw, &problem);
        assert_eq!(
            served.map(|(settings, _, _)| settings),
            Some(UnitDirectReduceRoutine::settings(&hw)[0])
        );
    }

    #[test]
    fn a_routine_that_declines_every_setting_serves_none() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let problem = ReduceRoutineProblem::dense(&[8, 300], 1, ReduceOperation::Sum);
        assert!(first_served::<CubeStagedReduceRoutine, _>(&hw, &problem).is_none());
    }

    #[test]
    fn a_unit_a_row_fills_the_device_with_many_rows() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[100_000, 8], 1, ReduceOperation::Sum);
        let filling = filling_the_device::<UnitDirectReduceRoutine, _>(&hw, &problem);
        assert_eq!(
            filling.map(|(settings, _)| settings),
            Some(UnitDirectReduceRoutine::settings(&hw)[0])
        );
    }

    #[test]
    fn a_unit_a_row_leaves_the_device_idle_with_few_rows() {
        let problem = ReduceRoutineProblem::dense(&[8, 300], 1, ReduceOperation::Sum);
        let filling =
            filling_the_device::<UnitDirectReduceRoutine, _>(&devices::m2_pro(), &problem);
        assert!(filling.is_none());
    }

    /// The key gives the hypercube, which declines a call given too little scratch.
    #[test]
    fn a_call_given_too_little_scratch_runs_another_routine() {
        const LAUNCH: BufferLaunch = |hw, tensors| select(hw, &given_too_little_scratch(tensors));
        BufferScopeChecker::new(LAUNCH).check([sum_of_one_long_row()]);
    }

    /// The rule's routine, launched.
    mod launched {
        use super::*;
        strategy_reduces_each_problem!(select);
    }
}
