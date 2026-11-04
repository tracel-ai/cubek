//! A reduce launch's compile-time decisions.

use crate::kernels::ReduceRoutineProblem;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperand};
use cubek_std::Hardware;

/// What a launch of this kind compiles: the microkernel, the partitioning that places its owners,
/// and the width each operand binds at.
///
/// One type for every routine of this kind, named per routine by an alias: what differs between
/// them is the microkernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceRoutineBlueprint<M> {
    pub microkernel: M,
    pub partitioning: Partitioning,
    /// The input's width: what a line of it holds.
    pub width: usize,
    /// The output's width: one along the reduced dim, the input's across it.
    pub out_width: usize,
}

/// The widths the operands bind at on `hw`, at most the widest line the operation keeps apart: to
/// the requirements the microkernel enqueued in `vectorization`, the binding adds each buffer's
/// alignment, and a line that never runs past a row along the reduced dim, or past the rows across
/// it. Settled before the partitioning, whose owners hold a line of rows across the reduced dim.
pub fn reduce_widths(
    hw: &Hardware,
    problem: &ReduceRoutineProblem,
    mut vectorization: Vectorization<ReduceOperand>,
) -> [usize; 2] {
    for (operand, alignment) in [ReduceOperand::Input, ReduceOperand::Out]
        .into_iter()
        .zip(problem.elem_alignments)
    {
        vectorization.require(VectorRequirement::Aligned { operand, alignment });
    }
    let line_stays_in = match (problem.layout(), problem.contiguous) {
        (ReduceLayout::Across, Some(inner)) => problem.shape[inner],
        _ => problem.reduced_extent(),
    };
    vectorization.require(VectorRequirement::Access {
        operand: ReduceOperand::Input,
        elements: line_stays_in,
    });
    let ceiling = hw.vector_ceiling(problem.elements.input);
    let widest = problem
        .operation
        .widest_line()
        .map_or(ceiling, |widest| widest.min(ceiling));
    [ReduceOperand::Input, ReduceOperand::Out]
        .map(|operand| vectorization.width(operand, |_| widest))
}

impl<M> ReduceRoutineBlueprint<M> {
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
