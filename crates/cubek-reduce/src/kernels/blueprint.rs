//! A reduce launch's compile-time decisions.

use crate::kernels::{BATCH, REDUCED, ReduceRoutineProblem, UNBOUND_LINES_NARROWER};
use cubecl::tile::launch::{Decline, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::space::UnitCube;
use cubecl::tile::{Level, Levels, Partitioning, Space};
use cubek_microkernel::reduce::{ReduceOperand, ReduceWorkItem};
use cubek_std::Hardware;
use cubek_std::microkernel::CubePlanes;
use cubek_std::routine::{CubeResources, Grid};

/// What a launch of this kind compiles: the microkernel, the partitioning that places its owners,
/// and the lanes the input binds at, which those
/// of the output follow: the rows a load of it reaches.
///
/// One type for every routine of this kind, named per routine by an alias: what differs between
/// them is the microkernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceBlueprint<M> {
    pub microkernel: M,
    pub partitioning: Partitioning,
    /// The lanes of a line of the input.
    pub lane_count: usize,
}

/// The lanes the input binds at on `hw`, at most `widest_line`, the widest line the
/// carrier keeps apart: to the requirements the microkernel enqueued in `vectorization`, the
/// binding adds each buffer's alignment, and a line that never runs past a row along the reduced
/// dim, or past the rows across it. Settled before the partitioning, whose owners hold a line of
/// rows across the reduced dim.
///
/// # Errors
///
/// [`UNBOUND_LINES_NARROWER`] where `problem`'s input is one the caller binds, stated at another
/// width than the lines the input is read in.
pub(crate) fn reduce_lane_count(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
    mut vectorization: Vectorization<ReduceOperand>,
    widest_line: Option<usize>,
) -> Result<usize, Decline> {
    for (operand, alignment) in [ReduceOperand::Input, ReduceOperand::Out]
        .into_iter()
        .zip(problem.elem_alignments)
    {
        vectorization.require(VectorRequirement::Aligned { operand, alignment });
    }
    // The indices go out at the values' lanes, so their buffer aligns them too.
    if let Some(indices) = problem.indices {
        vectorization.require(VectorRequirement::Aligned {
            operand: ReduceOperand::Out,
            alignment: indices.elem_alignment,
        });
    }
    vectorization.require(VectorRequirement::Access {
        operand: ReduceOperand::Input,
        elements: problem.line_stays_in(),
    });
    let ceiling = hw.vector_ceiling(problem.elements.input);
    let widest = widest_line.map_or(ceiling, |widest| widest.min(ceiling));
    let lane_count = vectorization.width(ReduceOperand::Input, |_| widest);
    let [input_lines, _] = problem.elem_alignments;
    if input_lines.is_stored_piece() && input_lines != ElemAlignment::stored_piece(lane_count) {
        return Err(UNBOUND_LINES_NARROWER);
    }
    Ok(lane_count)
}

/// What a reduce routine hands its microkernel's estimate of `problem` on `grid`: the work item and
/// the planes of a cube.
pub(crate) fn estimated_work(
    problem: &ReduceRoutineProblem,
    grid: &Grid,
) -> (ReduceWorkItem, CubePlanes) {
    (problem.work_item(), grid.planes())
}

/// `levels` placed over `space`, and the grid of cubes they deal to on `hw`, each cube `cube`.
///
/// # Errors
///
/// Where the device launches no such grid.
pub(crate) fn placed(
    hw: &Hardware,
    space: Space,
    levels: Vec<Level>,
    cube: CubeResources,
) -> Result<(Partitioning, Grid), Decline> {
    let partitioning = Partitioning::new(space, levels);
    let grid = Grid::new(hw, &partitioning, cube)?;
    Ok((partitioning, grid))
}

/// The launch of a unit an owner: `problem`'s owners at `lane_count`, each its own unit, in cubes of
/// `plane_count` planes as wide as `hw` dispatches one, holding no shared memory.
///
/// # Errors
///
/// Where the device cannot launch that grid.
pub(crate) fn unit_launch(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
    lane_count: usize,
    plane_count: usize,
) -> Result<(Partitioning, Grid), Decline> {
    let plane_width = hw.dispatch_plane_size();
    let unit_cube = UnitCube {
        plane_width: plane_width as usize,
        planes: plane_count,
    };
    let levels = Levels::leaf(&problem.owner_box_at_lane_count(lane_count))
        .launch_units(&[BATCH], unit_cube)
        .build();
    let cube = CubeResources {
        plane_width,
        shared_memory: 0,
    };
    placed(hw, problem.space(), levels, cube)
}

/// The launch of rows split across cubes: each row in parts `cube`'s units reduce,
/// `unit_line_count` lines of `lane_count` values a unit, a cube a part of a box of rows, over the
/// space `space_of_part` states for that part length. The parts ride the grid's `X`, the boxes of
/// rows the rest.
///
/// # Errors
///
/// Where the device cannot launch a cube a part.
pub(crate) fn split_launch(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
    space_of_part: impl FnOnce(usize) -> Space,
    cube: CubeResources,
    unit_line_count: usize,
    lane_count: usize,
) -> Result<(Partitioning, Grid), Decline> {
    let part = problem.split_part(cube.plane_width as usize, unit_line_count, lane_count);
    let levels = Levels::leaf(&problem.split_owner_box(lane_count, part))
        .cubes(&[REDUCED, BATCH])
        .build();
    placed(hw, space_of_part(part), levels, cube)
}

/// A routine's settings: each of its microkernel's `microkernels` with each of its own `knobs`,
/// the microkernel's order outermost, built by `settings`.
pub(crate) fn each_microkernel_with<M: Copy, K: Copy, S, const N: usize>(
    microkernels: Vec<M>,
    knobs: [K; N],
    settings: impl Fn(M, K) -> S,
) -> Vec<S> {
    microkernels
        .into_iter()
        .flat_map(|microkernel| knobs.map(|knob| settings(microkernel, knob)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ReduceIndices;
    use cubecl::prelude::CubeCount;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// Each microkernel setting crossed with each knob, the microkernel's outermost.
    #[test]
    fn settings_cross_the_microkernels_with_the_knobs() {
        let crossed = each_microkernel_with(vec!['a', 'b'], [1, 2], |microkernel, knob| {
            (microkernel, knob)
        });
        assert_eq!(crossed, vec![('a', 1), ('a', 2), ('b', 1), ('b', 2)]);
    }

    #[test]
    fn the_estimated_work_is_the_problems_item_on_the_grids_planes() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[64, 300], 1, ReduceOperation::Sum);
        let (_, grid) = unit_launch(&hw, &problem, 1, 4).expect("the unit launch fits");
        let (item, planes) = estimated_work(&problem, &grid);
        assert_eq!(
            (item, planes.count),
            (problem.work_item(), grid.planes().count)
        );
    }

    #[test]
    fn levels_a_device_cannot_launch_decline() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[64, 300], 1, ReduceOperation::Sum);
        let levels = Levels::leaf(&problem.owner_box_at_lane_count(1))
            .cubes(&[BATCH])
            .build();
        let cube = CubeResources {
            plane_width: hw.dispatch_plane_size(),
            shared_memory: hw.shared_memory_budget() + 1,
        };
        assert!(placed(&hw, problem.space(), levels, cube).is_err());
    }

    /// Two boxes of rows, each in three parts.
    #[test]
    fn a_split_launch_lays_its_parts_along_x() {
        let hw = devices::m2_pro();
        let cube = CubeResources {
            plane_width: 64,
            shared_memory: 0,
        };
        let problem = ReduceRoutineProblem::dense(&[2, 3 * 64], 1, ReduceOperation::Sum);
        let (_, grid) = split_launch(&hw, &problem, |_| problem.space(), cube, 1, 1)
            .expect("an M2 Pro launches six cubes");
        assert!(matches!(grid.launch.cube_count, CubeCount::Static(3, 2, 1)));
    }

    #[test]
    fn the_widest_line_bounds_the_width() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum);
        let lane_count = reduce_lane_count(&hw, &problem, Vectorization::new(), Some(1))
            .expect("a buffer's input should take any line the device reads");
        assert_eq!(lane_count, 1);
    }

    #[test]
    fn rows_across_the_axis_are_read_a_line_at_a_time() {
        let hw = devices::m2_pro();
        let dense = ReduceRoutineProblem::dense(&[4096, 4], 0, ReduceOperation::ArgMax);
        let lane_count = reduce_lane_count(&hw, &dense, Vectorization::new(), None)
            .expect("a buffer's input should take any line the device reads");
        assert!(lane_count > 1);
    }

    /// Indices beside the values go out at their lanes, so a strided indices buffer narrows them to
    /// one.
    #[test]
    fn strided_indices_narrow_the_width() {
        let hw = devices::m2_pro();
        let dense = ReduceRoutineProblem::dense(&[4096, 4], 0, ReduceOperation::ArgMax);
        let problem = ReduceRoutineProblem {
            indices: Some(ReduceIndices {
                element: cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U32),
                elem_alignment: ElemAlignment::ONE,
            }),
            ..dense
        };
        let mut vectorization = Vectorization::new();
        vectorization.require(problem.work_item().requirement());
        let lane_count = reduce_lane_count(&hw, &problem, vectorization, None)
            .expect("a buffer's input should take any line the device reads");
        assert_eq!(lane_count, 1);
    }

    /// A unit-an-owner launch groups its units in cubes of `PLANE_COUNT` planes the device's width.
    #[test]
    fn a_unit_launch_holds_its_planes_in_a_cube() {
        const PLANE_COUNT: usize = 4;
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[64, 300], 1, ReduceOperation::Sum);
        let (_, grid) = unit_launch(&hw, &problem, 1, PLANE_COUNT).expect("the unit launch fits");
        assert_eq!(
            grid.launch.units_per_cube(),
            hw.dispatch_plane_size() * PLANE_COUNT as u32
        );
    }

    const ROW_COUNT: usize = 64;

    /// A sum of [`ROW_COUNT`] rows whose input the caller binds `input_lane_count` wide.
    fn sum_with_fixed_input_lanes(
        row_value_count: usize,
        input_lane_count: usize,
    ) -> ReduceRoutineProblem {
        let mut problem =
            ReduceRoutineProblem::dense(&[ROW_COUNT, row_value_count], 1, ReduceOperation::Sum);
        problem.elem_alignments[0] = ElemAlignment::stored_piece(input_lane_count);
        problem
    }

    #[test]
    fn an_input_wider_than_the_device_can_read_declines() {
        let problem = sum_with_fixed_input_lanes(4096, 64);
        let declined = reduce_lane_count(&devices::m2_pro(), &problem, Vectorization::new(), None)
            .expect_err("the device cannot read 64 elements per line");
        assert_eq!(declined, UNBOUND_LINES_NARROWER);
    }

    #[test]
    fn a_two_wide_input_declines_when_the_row_requires_scalar_reads() {
        let problem = sum_with_fixed_input_lanes(7, 2);
        let declined = reduce_lane_count(&devices::m2_pro(), &problem, Vectorization::new(), None)
            .expect_err("a scalar read would split the caller's two-element line");
        assert_eq!(declined, UNBOUND_LINES_NARROWER);
    }
}
