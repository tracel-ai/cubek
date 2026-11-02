//! A walk launch's compile-time decisions.

use crate::kernels::walk::{GATED_DELTA_WALK_ELEMENT, GatedDeltaWalkProblem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::gated_delta::GatedDeltaWalkPartitioning;
use cubek_microkernel::gated_delta::walk::GatedDeltaWalkOperand;
use cubek_std::Hardware;

/// The width the operands bind at on `hw`: to the requirements enqueued in `vectorization`, the
/// binding adds its own, each operand's buffer alignment and where its view starts. The operands
/// then bind at the widest width this device reads their elements in that they all admit; the decay
/// and the strength, one scalar a position, at one.
pub fn settled_width(
    hw: &Hardware,
    problem: &GatedDeltaWalkProblem,
    mut vectorization: Vectorization<GatedDeltaWalkOperand>,
) -> usize {
    for operand in GatedDeltaWalkOperand::ALL {
        let element = operand_element(operand, problem);
        vectorization.require(VectorRequirement::Aligned {
            operand,
            alignment: problem.alignments.get(operand),
        });
        vectorization.require(VectorRequirement::Aligned {
            operand,
            alignment: ElemAlignment::at_start(problem.start_bytes.get(operand), element.size()),
        });
    }
    vectorization.width(GatedDeltaWalkOperand::Value, |_| {
        hw.vector_ceiling(GATED_DELTA_WALK_ELEMENT)
            .min(hw.vector_ceiling(problem.output_element))
    })
}

/// The element `operand` is bound at: the output's is the activations', every other's `f32`.
fn operand_element(operand: GatedDeltaWalkOperand, problem: &GatedDeltaWalkProblem) -> ElemType {
    match operand {
        GatedDeltaWalkOperand::Out => problem.output_element,
        _ => GATED_DELTA_WALK_ELEMENT,
    }
}

/// What a launch of the walk compiles: the microkernel, the partitioning that places its runs,
/// the one width the queries, keys, values, states and outputs bind at (a unit reads a line of each
/// beside the others), and how many runs a cube holds, which sizes its shared memory.
///
/// One type for every routine of this kind: what differs between them is the microkernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedDeltaWalkBlueprint<M> {
    pub microkernel: M,
    pub partitioning: GatedDeltaWalkPartitioning,
    pub width: usize,
    pub runs: usize,
}

impl<M> GatedDeltaWalkBlueprint<M> {
    /// `microkernel` placed by `partitioning`, `runs` a cube, the operands bound at `width`
    /// ([`settled_width`]).
    pub fn new(
        microkernel: M,
        partitioning: GatedDeltaWalkPartitioning,
        width: usize,
        runs: usize,
    ) -> Self {
        Self {
            microkernel,
            partitioning,
            width,
            runs,
        }
    }
}
