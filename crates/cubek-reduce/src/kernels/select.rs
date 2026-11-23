//! The reduce family's rule: which routine runs a problem when nothing is measured.

use crate::kernels::{
    BATCH, CubeStagedReduceRoutine, HypercubeLastArrivalReduceRoutine,
    HypercubeLastArrivalReduceSettings, PlaneShuffleReduceRoutine, ReduceBlueprint, ReduceKernel,
    ReduceLauncher, ReduceRoutineProblem, ReduceSplitExtents, ReduceTensors,
    UnitDirectReduceRoutine, launch_kernel,
};
use cubecl::tile::launch::Decline;
use cubek_microkernel::reduce::ReduceCarrier;
use cubek_std::Hardware;
use cubek_std::autotune::short_of_the_device;
use cubek_std::routine::{Grid, Routine};

/// `tensors` reduced by the routine the rule picks on `hw`, read top to bottom. The rule reads the
/// problem the call's Tune Key stands for ([`ReduceRoutineProblem::keyed`]), so every call of one
/// key runs the same routine at the same setting:
///
/// 1. Rows enough that a unit each fills the device, each row short enough that a unit walks it
///    alone as fast as a plane would: the unit routine.
/// 2. Rows enough that a plane each fills the device: the plane routine, whose lanes share a row
///    without meeting in shared memory.
/// 3. Rows a cube each would leave the device idle, and a split that fills it: the hypercube, at
///    its first setting whose cubes fill the device, or at the one with the most cubes.
/// 4. Otherwise the cube routine.
/// 5. Where the cube declines the problem, the unit routine, which serves every problem.
///
/// # Errors
///
/// Why the unit routine cannot run the problem here, which no device refuses.
pub(crate) fn select(hw: &Hardware, tensors: &ReduceTensors) -> Result<(), Decline> {
    let keyed = tensors.problem().keyed();
    if let Some((settings, blueprint)) =
        filling_the_device::<UnitDirectReduceRoutine, _>(hw, &keyed)
        && unit_keeps_up(hw, &keyed, blueprint.width)
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

/// Whether a unit reading a row alone, `width` values a line, keeps up with a plane reading it
/// together. Across the reduced dim it does: neighbouring units read neighbouring rows, so their
/// reads meet. Along it, a unit walks its row line after line where a plane takes a line a lane,
/// so it keeps up only while the row holds no more lines than a plane has lanes, or where the
/// device runs no plane routine.
fn unit_keeps_up(hw: &Hardware, problem: &ReduceRoutineProblem, width: usize) -> bool {
    if problem.lines_along() == BATCH {
        return true;
    }
    hw.plane_dim()
        .ok()
        .is_none_or(|lanes| problem.reduced_extent().div_ceil(width) <= lanes as usize)
}

/// The setting of the hypercube's launch where a box of rows a cube leaves the device idle and
/// splitting the rows along the reduced dim gives more cubes: the first of its settings, the
/// longest parts first, whose cubes fill the device, or the one with the most cubes where none
/// does. A top-k takes the longest parts only: the last cube merges a row's top-k lists one after
/// another, `k` inserts a part, so shorter parts cost more in the merge than their extra cubes
/// save. `None` where the device does not say how wide it is, the boxes already fill it, or every
/// row is one part.
fn hypercube_filling_the_device(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
) -> Option<HypercubeLastArrivalReduceSettings> {
    let mut most_cubes = None;
    let longest_only = matches!(
        problem.operation.carrier(),
        ReduceCarrier::TopK { .. } | ReduceCarrier::TopValues { .. }
    );
    let settings = HypercubeLastArrivalReduceRoutine::settings(hw)
        .into_iter()
        .filter(|settings| {
            !longest_only
                || settings.unit_lines == HypercubeLastArrivalReduceSettings::UNIT_LINES[0]
        });
    for settings in settings {
        let Ok((blueprint, _)) = HypercubeLastArrivalReduceRoutine::prepare(hw, problem, &settings)
        else {
            continue;
        };
        let units = blueprint.microkernel.part().units() as u32;
        let fill = hw.cubes_to_fill(units)? as usize;
        let boxes = problem.row_boxes_at_width(blueprint.width);
        let parts = ReduceSplitExtents::new(problem, &blueprint.partitioning).parts as usize;
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
fn first_served<R: ReduceLauncher<M>, M>(
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
fn filling_the_device<R: ReduceLauncher<M>, M>(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
) -> Option<(R::Settings, ReduceBlueprint<M>)> {
    first_served::<R, M>(hw, problem)
        .filter(|(_, _, grid)| short_of_the_device(hw, &grid.launch) == 0)
        .map(|(settings, blueprint, _)| (settings, blueprint))
}

/// Whether `R` at `settings` launched on `tensors`: `false` where it declines their problem.
fn launched<R: ReduceLauncher<M>, M>(
    hw: &Hardware,
    tensors: &ReduceTensors,
    settings: &R::Settings,
) -> bool {
    R::prepare(hw, tensors.problem(), settings)
        .map(|(blueprint, grid)| R::launch(tensors, blueprint, grid))
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// Across the reduced axis a unit always keeps up: each lane its own row.
    #[test]
    fn a_unit_keeps_up_across_the_reduced_axis() {
        let problem = ReduceRoutineProblem::dense(&[100_000, 8], 0, ReduceOperation::Sum);
        assert!(unit_keeps_up(&devices::m2_pro(), &problem, 1));
    }

    /// Lines of `WIDTH` values, rows as long as a plane's lanes' lines.
    const WIDTH: usize = 4;

    /// Along it, a unit keeps up while the row holds no more lines than a plane has lanes.
    #[test]
    fn a_unit_keeps_up_along_a_row_no_longer_than_a_plane() {
        let hw = devices::m2_pro();
        let lanes = hw.plane_dim().expect("an M2 Pro commits to a plane width") as usize;
        let short = ReduceRoutineProblem::dense(&[8, lanes * WIDTH], 1, ReduceOperation::Sum);
        assert!(unit_keeps_up(&hw, &short, WIDTH));
    }

    /// Along it, a unit falls behind on a row longer than a plane's lanes' lines.
    #[test]
    fn a_unit_falls_behind_along_a_row_longer_than_a_plane() {
        let hw = devices::m2_pro();
        let lanes = hw.plane_dim().expect("an M2 Pro commits to a plane width") as usize;
        let long = ReduceRoutineProblem::dense(&[8, lanes * WIDTH + 1], 1, ReduceOperation::Sum);
        assert!(!unit_keeps_up(&hw, &long, WIDTH));
    }

    /// On a device with no committed plane width, nothing outruns the unit.
    #[test]
    fn a_unit_keeps_up_where_no_plane_routine_runs() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let long = ReduceRoutineProblem::dense(&[8, 100_000], 1, ReduceOperation::Sum);
        assert!(unit_keeps_up(&hw, &long, 1));
    }
}
