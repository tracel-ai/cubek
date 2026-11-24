//! A reduce launch's compile-time decisions.

use crate::kernels::{BATCH, REDUCED, ReduceRoutineProblem, UNBOUND_LINES_NARROWER};
use cubecl::tile::launch::{Decline, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::space::UnitCube;
use cubecl::tile::{Levels, Partitioning, Space};
use cubek_microkernel::reduce::{ReduceOperand, ReduceWorkItem};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid};

/// What a launch of this kind compiles: the microkernel, the partitioning that places its owners,
/// and the width each operand binds at.
///
/// One type for every routine of this kind, named per routine by an alias: what differs between
/// them is the microkernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceBlueprint<M> {
    pub microkernel: M,
    pub partitioning: Partitioning,
    /// The input's width: what a line of it holds.
    pub width: usize,
    /// The output's width: one along the reduced dim, the input's across it.
    pub out_width: usize,
}

/// The widths the operands bind at on `hw`, at most `widest_line`, the widest line the carrier keeps apart: to
/// the requirements the microkernel enqueued in `vectorization`, the binding adds each buffer's
/// alignment, and a line that never runs past a row along the reduced dim, or past the rows across
/// it. Settled before the partitioning, whose owners hold a line of rows across the reduced dim.
///
/// # Errors
///
/// [`UNBOUND_LINES_NARROWER`] where `problem`'s input is one the caller binds, stated at another
/// width than the lines the input is read in.
pub(crate) fn reduce_widths(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
    mut vectorization: Vectorization<ReduceOperand>,
    widest_line: Option<usize>,
) -> Result<[usize; 2], Decline> {
    for (operand, alignment) in [ReduceOperand::Input, ReduceOperand::Out]
        .into_iter()
        .zip(problem.elem_alignments)
    {
        vectorization.require(VectorRequirement::Aligned { operand, alignment });
    }
    // The indices go out at the values' width, so their buffer aligns it too.
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
    let [width, out_width] = [ReduceOperand::Input, ReduceOperand::Out]
        .map(|operand| vectorization.width(operand, |_| widest));
    let [input_lines, _] = problem.elem_alignments;
    if input_lines.is_stored_piece() && input_lines != ElemAlignment::stored_piece(width) {
        return Err(UNBOUND_LINES_NARROWER);
    }
    Ok([width, out_width])
}

/// The widths `microkernel`, whose lines are at most `widest_line` wide, binds `problem`'s
/// operands at on `hw`: what it asks of them, then [`reduce_widths`].
///
/// # Errors
///
/// As [`reduce_widths`].
pub(crate) fn microkernel_widths<M: Microkernel<CubeSpace = (), Operand = ReduceOperand>>(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
    microkernel: &M,
    widest_line: Option<usize>,
) -> Result<[usize; 2], Decline> {
    let mut vectorization = Vectorization::new();
    microkernel.vectorize(&(), &mut vectorization);
    reduce_widths(hw, problem, vectorization, widest_line)
}

/// What `microkernel` costs reducing `problem` on `grid`, added to `estimate`.
pub(crate) fn estimate_reduce<M: Microkernel<WorkItem = ReduceWorkItem, CubeSpace = ()>>(
    problem: &ReduceRoutineProblem,
    microkernel: &M,
    grid: &Grid,
    estimate: &mut KernelEstimate,
) {
    microkernel.estimate(&problem.work_item(), &(), grid.planes(), estimate);
}

/// The launch of a unit an owner: `problem`'s owners at `width`, each its own unit, in cubes of
/// `planes` planes as wide as `hw` dispatches one, holding no shared memory.
///
/// # Errors
///
/// Where the device cannot launch that grid.
pub(crate) fn unit_launch(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
    width: usize,
    planes: usize,
) -> Result<(Partitioning, Grid), Decline> {
    let plane_width = hw.dispatch_plane_size() as usize;
    let partitioning = Partitioning::new(
        problem.space(),
        Levels::leaf(&problem.owner_box_at_width(width))
            .launch_units(
                &[BATCH],
                UnitCube {
                    plane_width,
                    planes,
                },
            )
            .build(),
    );
    let cube = CubeResources {
        plane_width: plane_width as u32,
        shared_memory: 0,
    };
    let grid = Grid::new(hw, &partitioning, cube)?;
    Ok((partitioning, grid))
}

/// The launch of rows split across cubes: each row in parts a cube of `units` units reduces,
/// `unit_lines` lines of `width` values a unit, a cube a part of a box of rows, over the space
/// `space_of_part` states for that part length. The parts ride the grid's `X`, the boxes of rows
/// the rest; each cube holds `shared_memory` bytes.
///
/// # Errors
///
/// Where the device cannot launch that grid.
pub(crate) fn split_launch(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
    space_of_part: impl FnOnce(usize) -> Space,
    units: usize,
    unit_lines: usize,
    width: usize,
    shared_memory: usize,
) -> Result<(Partitioning, Grid), Decline> {
    let part = problem.split_part(units, unit_lines, width);
    let partitioning = Partitioning::new(
        space_of_part(part),
        Levels::leaf(&problem.split_owner_box(width, part))
            .cubes(&[REDUCED, BATCH])
            .build(),
    );
    let cube = CubeResources {
        plane_width: units as u32,
        shared_memory,
    };
    let grid = Grid::new(hw, &partitioning, cube)?;
    Ok((partitioning, grid))
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

impl<M> ReduceBlueprint<M> {
    /// `microkernel` placed by `partitioning`, its operands bound at `widths`.
    pub fn new(microkernel: M, partitioning: Partitioning, widths: [usize; 2]) -> Self {
        let [width, out_width] = widths;
        Self {
            microkernel,
            partitioning,
            width,
            out_width,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ReduceIndices;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;
    use cubek_std::routine::Routine;

    /// Each microkernel setting crossed with each knob, the microkernel's outermost.
    #[test]
    fn settings_cross_the_microkernels_with_the_knobs() {
        let crossed = each_microkernel_with(vec!['a', 'b'], [1, 2], |microkernel, knob| {
            (microkernel, knob)
        });
        assert_eq!(crossed, vec![('a', 1), ('a', 2), ('b', 1), ('b', 2)]);
    }

    /// A carrier's widest line bounds the input's width.
    #[test]
    fn the_widest_line_bounds_the_width() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum);
        let [width, _] = reduce_widths(&hw, &problem, Vectorization::new(), Some(1))
            .expect("a buffer's input should take any line the device reads");
        assert_eq!(width, 1);
    }

    /// Rows across the reduced axis go out a line of rows at a time.
    #[test]
    fn rows_across_the_axis_go_out_a_line_at_a_time() {
        let hw = devices::m2_pro();
        let dense = ReduceRoutineProblem::dense(&[4096, 4], 0, ReduceOperation::ArgMax);
        let [_, dense_width] = reduce_widths(&hw, &dense, Vectorization::new(), None)
            .expect("a buffer's input should take any line the device reads");
        assert!(dense_width > 1);
    }

    /// Indices beside the values align the output's width too: a strided indices buffer narrows it
    /// to one.
    #[test]
    fn strided_indices_narrow_the_outputs_width() {
        let hw = devices::m2_pro();
        let dense = ReduceRoutineProblem::dense(&[4096, 4], 0, ReduceOperation::ArgMax);
        let problem = ReduceRoutineProblem {
            indices: Some(ReduceIndices {
                element: cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U32),
                elem_alignment: ElemAlignment::ONE,
            }),
            ..dense
        };
        let [_, out_width] = reduce_widths(&hw, &problem, Vectorization::new(), None)
            .expect("a buffer's input should take any line the device reads");
        assert_eq!(out_width, 1);
    }

    /// A unit-an-owner launch groups its units in cubes of `planes` planes the device's width.
    #[test]
    fn a_unit_launch_holds_its_planes_in_a_cube() {
        const PLANES: usize = 4;
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[64, 300], 1, ReduceOperation::Sum);
        let (_, grid) = unit_launch(&hw, &problem, 1, PLANES).expect("the unit launch fits");
        assert_eq!(
            grid.launch.units_per_cube(),
            hw.dispatch_plane_size() * PLANES as u32
        );
    }

    /// Settings of the unit routine: one plane of units a cube, the default top-k slots.
    fn unit_settings(hw: &Hardware) -> crate::kernels::UnitDirectReduceSettings {
        crate::kernels::UnitDirectReduceRoutine::settings(hw)[0]
    }

    fn sum_with_fixed_input_width(row_len: usize, input_width: usize) -> ReduceRoutineProblem {
        let mut problem = ReduceRoutineProblem::dense(&[64, row_len], 1, ReduceOperation::Sum);
        problem.elem_alignments[0] = ElemAlignment::stored_piece(input_width);
        problem
    }

    #[test]
    fn an_input_wider_than_the_device_can_read_declines() {
        use crate::kernels::UnitDirectReduceRoutine;
        let hw = devices::m2_pro();
        let problem = sum_with_fixed_input_width(4096, 64);
        let declined = UnitDirectReduceRoutine::prepare(&hw, &problem, &unit_settings(&hw))
            .expect_err("the device cannot read 64 elements per line");
        assert_eq!(declined.to_string(), UNBOUND_LINES_NARROWER.to_string());
    }

    #[test]
    fn a_two_wide_input_declines_when_the_row_requires_scalar_reads() {
        use crate::kernels::UnitDirectReduceRoutine;
        let hw = devices::m2_pro();
        let problem = sum_with_fixed_input_width(7, 2);
        let declined = UnitDirectReduceRoutine::prepare(&hw, &problem, &unit_settings(&hw))
            .expect_err("a scalar read would split the caller's two-element line");
        assert_eq!(declined.to_string(), UNBOUND_LINES_NARROWER.to_string());
    }
}
