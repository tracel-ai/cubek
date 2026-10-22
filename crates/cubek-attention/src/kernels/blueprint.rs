//! An attention launch's compile-time decisions.

use crate::kernels::AttentionRoutineProblem;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::attention::AttentionMicrokernel;
use cubek_microkernel::attention::{AttentionOperand, AttentionOperands, AttentionPartitioning, S};
use cubek_microkernel::matmul::MatmulOrder;
use cubek_std::Hardware;

/// What a launch of this kind compiles: the microkernel, the partitioning that places its work
/// items, the width each operand binds at, and how the routine lays out shared memory.
///
/// One type for every routine of this kind, named per routine by an alias: what differs between
/// them is the microkernel, and the shared memory only the routine that walks at that level
/// decides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionBlueprint<M, SM = ()> {
    pub microkernel: M,
    pub partitioning: AttentionPartitioning,
    pub widths: AttentionOperands<usize>,
    /// How the routine lays out the shared memory its walk holds; `()` for one that holds none.
    pub shared_memory: SM,
}

impl<M: AttentionMicrokernel, SM> AttentionBlueprint<M, SM> {
    /// `microkernel` placed by `partitioning`, shared memory laid out as `shared_memory`: the
    /// blueprint once its widths are settled.
    ///
    /// To the requirements the routine and its microkernel enqueued in `vectorization`, the
    /// binding adds its own: each operand's buffer alignment, where its view starts, and the
    /// masking of an operand whose innermost bound axis the tiles overhang. The keys and values
    /// are read no further than the attended keys, a length only the kernel knows, so an operand
    /// lined along `S` is read one element at a time. Each operand then binds at the widest width
    /// this device reads its element in that they admit.
    pub fn new(
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        microkernel: M,
        partitioning: AttentionPartitioning,
        mut vectorization: Vectorization<AttentionOperand>,
        shared_memory: SM,
    ) -> Self {
        let key_order = microkernel.key_order();
        for operand in AttentionOperand::ALL {
            let alignment = match (operand, key_order) {
                // The keys bound along `S` are a stride over their buffer.
                (AttentionOperand::Key, MatmulOrder::Stated) => ElemAlignment::ONE,
                _ => problem.elem_alignments.get(operand),
            };
            vectorization.require(VectorRequirement::Aligned { operand, alignment });
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: ElemAlignment::at_start(
                    problem.start_bytes.get(operand),
                    problem.elements.get(operand).size(),
                ),
            });
            let innermost = operand.innermost(key_order);
            if innermost == S || partitioning.overhangs(innermost) {
                vectorization.require(VectorRequirement::Masked { operand });
            }
        }
        let widths = problem
            .elements
            .map(|operand, element| vectorization.width(operand, |_| hw.vector_ceiling(element)));
        Self {
            microkernel,
            partitioning,
            widths,
            shared_memory,
        }
    }
}
