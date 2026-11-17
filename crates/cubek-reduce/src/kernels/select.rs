//! The reduce family's rule: which routine runs a problem when nothing is measured.

use crate::kernels::{
    CubeReduceRoutine, HypercubeReduceRoutine, HypercubeReduceRoutineSettings, ReduceKernel,
    ReduceRoutine, ReduceSplitExtents, ReduceTensors, UnitReduceRoutine,
};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::reduce::ReduceCarrier;
use cubek_std::Hardware;
use cubek_std::autotune::short_of_the_device;
use cubek_std::routine::Routine;

/// `tensors` reduced by the routine the rule picks on `hw`, read top to bottom:
///
/// 1. Rows enough that a unit each fills the device: the unit routine.
/// 2. Rows a cube each would leave the device idle, and a split that fills it: the hypercube, at
///    its first setting whose cubes fill the device, or at the one with the most cubes.
/// 3. Otherwise the cube routine.
/// 4. Where the cube declines the problem, the unit routine, which serves every problem.
///
/// # Errors
///
/// Why the unit routine cannot run the problem here, which no device refuses.
pub fn select(hw: &Hardware, tensors: &ReduceTensors) -> Result<(), Refusal> {
    let problem = tensors.problem();
    if let Ok((blueprint, grid)) = UnitReduceRoutine.prepare_first_served(hw, problem)
        && short_of_the_device(hw, &grid.launch) == 0
    {
        UnitReduceRoutine.launch(tensors, blueprint, grid);
        return Ok(());
    }
    if let Some((blueprint, grid)) = hypercube_filling_the_device(hw, tensors) {
        HypercubeReduceRoutine.launch(tensors, blueprint, grid);
        return Ok(());
    }
    if let Ok((blueprint, grid)) = CubeReduceRoutine.prepare_first_served(hw, problem) {
        CubeReduceRoutine.launch(tensors, blueprint, grid);
        return Ok(());
    }
    ReduceKernel::Unit.launch(hw, tensors)
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
    <HypercubeReduceRoutine as Routine>::Blueprint,
    cubek_std::routine::Grid,
)> {
    let problem = tensors.problem();
    let routine = HypercubeReduceRoutine;
    let mut most_cubes = None;
    let longest_only = matches!(
        problem.operation.carrier(),
        ReduceCarrier::TopK { .. } | ReduceCarrier::TopValues { .. }
    );
    let settings = routine.settings(hw).into_iter().filter(|settings| {
        !longest_only || settings.unit_lines == HypercubeReduceRoutineSettings::UNIT_LINES[0]
    });
    for settings in settings {
        let Ok((blueprint, grid)) = routine.prepare(hw, problem, &settings) else {
            continue;
        };
        let units = (blueprint.microkernel.part.lanes * blueprint.microkernel.part.planes) as u32;
        let fill = hw.cubes_to_fill(units)? as usize;
        let boxes = ReduceSplitExtents::row_boxes(problem, blueprint.width);
        let parts = ReduceSplitExtents::new(problem, &blueprint.partitioning, blueprint.width).parts
            as usize;
        if boxes >= fill {
            return None;
        }
        if parts < 2 {
            continue;
        }
        let cubes = boxes * parts;
        if cubes >= fill {
            return Some((blueprint, grid));
        }
        if most_cubes.as_ref().is_none_or(|(most, _, _)| cubes > *most) {
            most_cubes = Some((cubes, blueprint, grid));
        }
    }
    most_cubes.map(|(_, blueprint, grid)| (blueprint, grid))
}
