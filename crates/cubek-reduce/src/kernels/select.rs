//! The reduce family's rule: which routine runs a problem when nothing is measured.

use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeLastArrivalReduceRoutine, HypercubeLastArrivalReduceSettings,
    PlaneShuffleReduceRoutine, ReduceKernel, ReduceLauncher, ReduceRoutineProblem,
    ReduceSplitExtents, ReduceTensors, UnitDirectReduceRoutine, launch_kernel,
};
use cubecl::tile::launch::Decline;
use cubek_microkernel::reduce::{ReduceCarrier, ReduceLayout};
use cubek_std::Hardware;
use cubek_std::autotune::short_of_the_device;
use cubek_std::routine::Routine;

/// `tensors` reduced by the routine the rule picks on `hw`, read top to bottom:
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
    let problem = tensors.problem();
    if let Ok((blueprint, grid)) = UnitDirectReduceRoutine::prepare_first_served(hw, problem)
        && short_of_the_device(hw, &grid.launch) == 0
        && unit_keeps_up(hw, problem, blueprint.width)
    {
        UnitDirectReduceRoutine::launch(tensors, blueprint, grid);
        return Ok(());
    }
    if let Ok((blueprint, grid)) = PlaneShuffleReduceRoutine::prepare_first_served(hw, problem)
        && short_of_the_device(hw, &grid.launch) == 0
    {
        PlaneShuffleReduceRoutine::launch(tensors, blueprint, grid);
        return Ok(());
    }
    if let Some((blueprint, grid)) = hypercube_filling_the_device(hw, tensors) {
        HypercubeLastArrivalReduceRoutine::launch(tensors, blueprint, grid);
        return Ok(());
    }
    if let Ok((blueprint, grid)) = CubeStagedReduceRoutine::prepare_first_served(hw, problem) {
        CubeStagedReduceRoutine::launch(tensors, blueprint, grid);
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
    match problem.layout() {
        ReduceLayout::Across => true,
        ReduceLayout::Along => hw
            .plane_dim()
            .ok()
            .is_none_or(|lanes| problem.reduced_extent().div_ceil(width) <= lanes as usize),
    }
}

/// The hypercube's launch where a box of rows a cube leaves the device idle and splitting the rows
/// along the reduced dim gives more cubes: at the first of its settings, the longest parts first,
/// whose cubes fill the device, or at the one with the most cubes where none does. A top-k takes
/// the longest parts only: the last cube merges a row's top-k lists one after another, `k` inserts
/// a part, so shorter parts cost more in the merge than their extra cubes save. `None` where the
/// device does not say how wide it is, the boxes already fill it, or every row is one part.
fn hypercube_filling_the_device(
    hw: &Hardware,
    tensors: &ReduceTensors,
) -> Option<(
    <HypercubeLastArrivalReduceRoutine as Routine>::Blueprint,
    cubek_std::routine::Grid,
)> {
    let problem = tensors.problem();
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
        let Ok((blueprint, grid)) =
            HypercubeLastArrivalReduceRoutine::prepare(hw, problem, &settings)
        else {
            continue;
        };
        let units =
            (blueprint.microkernel.part().lanes() * blueprint.microkernel.part().planes()) as u32;
        let fill = hw.cubes_to_fill(units)? as usize;
        let boxes = ReduceSplitExtents::row_boxes(problem, blueprint.width);
        let parts = ReduceSplitExtents::new(problem, &blueprint.partitioning, blueprint.width).parts
            as usize;
        if boxes >= fill {
            return None;
        }
        // A row in one part is the cube routine's, not a split.
        let Some(cubes) = (parts >= 2).then_some(boxes * parts) else {
            continue;
        };
        if cubes >= fill {
            return Some((blueprint, grid));
        }
        if most_cubes.as_ref().is_none_or(|(most, _, _)| cubes > *most) {
            most_cubes = Some((cubes, blueprint, grid));
        }
    }
    most_cubes.map(|(_, blueprint, grid)| (blueprint, grid))
}
