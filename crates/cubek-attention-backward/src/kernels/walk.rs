//! One launch of a routine: the microkernel it runs and how it binds.

use crate::kernels::AttentionBackwardProblem;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::attention::{B, KV, Q, S};
use cubek_microkernel::attention_backward::{
    AttentionBackwardOperand, AttentionBackwardOperands, AttentionBackwardWorkItem,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// A microkernel of this kind, as a walk reads it.
pub trait AttentionBackwardMicrokernel:
    Microkernel<
        WorkItem = AttentionBackwardWorkItem,
        Operand = AttentionBackwardOperand,
        CubeSpace = Partitioning,
    > + core::fmt::Display
{
}

impl<M> AttentionBackwardMicrokernel for M where
    M: Microkernel<
            WorkItem = AttentionBackwardWorkItem,
            Operand = AttentionBackwardOperand,
            CubeSpace = Partitioning,
        > + core::fmt::Display
{
}

/// The decisions of one launch of a routine: its microkernel, the partitioning that places its
/// work items, the width each operand binds at, and how the routine lays out the shared memory the
/// walk holds; `()` for one that holds none. The grid it spawns is beside it
/// ([`AttentionBackwardGrid`](crate::kernels::AttentionBackwardGrid)).
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionBackwardWalk<M, SM = ()> {
    pub microkernel: M,
    pub partitioning: Partitioning,
    pub widths: AttentionBackwardOperands<usize>,
    pub shared_memory: SM,
}

impl<M: AttentionBackwardMicrokernel, SM> AttentionBackwardWalk<M, SM> {
    /// `microkernel` placed by `partitioning`, shared memory laid out as `shared_memory`, its
    /// widths settled.
    ///
    /// To the requirements the microkernel enqueues, the binding adds its own: each operand's
    /// buffer alignment, where its view starts, and the masking of an operand whose innermost
    /// axis the tiles overhang. Each operand then binds at the widest width this device reads its
    /// element in that they admit.
    pub(crate) fn new(
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        microkernel: M,
        partitioning: Partitioning,
        shared_memory: SM,
    ) -> Self {
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&partitioning, &mut vectorization);
        let elements = problem.elements();
        for operand in AttentionBackwardOperand::ALL {
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: problem.elem_alignments.get(operand),
            });
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: ElemAlignment::at_start(
                    problem.start_bytes.get(operand),
                    elements.get(operand).size(),
                ),
            });
            if partitioning.overhangs(operand.innermost()) {
                vectorization.require(VectorRequirement::masked(
                    operand,
                    partitioning.space(),
                    operand.innermost(),
                ));
            }
        }
        let widths = elements
            .map(|operand, &element| vectorization.width(operand, |_| hw.vector_ceiling(element)));
        Self {
            microkernel,
            partitioning,
            widths,
            shared_memory,
        }
    }

    /// The partitioning the walk's kernel compiles against: every extent a launch varies — the
    /// batches, the key heads, the queries and the keys — left to the launch, and the group and
    /// the dims a row spans compiled in, which a row's registers are sized by.
    pub(crate) fn kernel_partitioning(&self) -> Partitioning {
        Partitioning::new(
            self.partitioning
                .space()
                .clone()
                .with_dynamic(&[B, KV, Q, S]),
            self.partitioning.levels().to_vec(),
        )
    }
}
