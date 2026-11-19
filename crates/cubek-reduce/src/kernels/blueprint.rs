//! A reduce launch's compile-time decisions.

use crate::kernels::ReduceRoutineProblem;
use cubecl::tile::launch::{Decline, VectorRequirement, Vectorization};
use cubecl::tile::space::UnitCube;
use cubecl::tile::{Levels, Partitioning};
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperand, ReduceWorkItem};
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
pub(crate) fn reduce_widths(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
    mut vectorization: Vectorization<ReduceOperand>,
    widest_line: Option<usize>,
) -> [usize; 2] {
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
    let line_stays_in = problem
        .contiguous
        .filter(|_| problem.layout() == ReduceLayout::Across)
        .map_or(problem.reduced_extent(), |inner| problem.shape[inner]);
    vectorization.require(VectorRequirement::Access {
        operand: ReduceOperand::Input,
        elements: line_stays_in,
    });
    let ceiling = hw.vector_ceiling(problem.elements.input);
    let widest = widest_line.map_or(ceiling, |widest| widest.min(ceiling));
    [ReduceOperand::Input, ReduceOperand::Out]
        .map(|operand| vectorization.width(operand, |_| widest))
}

/// The widths `microkernel`, whose lines are at most `widest_line` wide, binds `problem`'s
/// operands at on `hw`: what it asks of them, then [`reduce_widths`].
pub(crate) fn microkernel_widths<M: Microkernel<CubeSpace = (), Operand = ReduceOperand>>(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
    microkernel: &M,
    widest_line: Option<usize>,
) -> [usize; 2] {
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
                &problem.owner_dimensions_innermost_first(),
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
