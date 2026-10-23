//! What a routine of this kind states: the two walks it launches, and the launch around them.

use crate::{AttentionBackwardGradients, AttentionBackwardProblem, AttentionBackwardTensors};
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::attention::{B, KV, Q, S};
use cubek_microkernel::attention_backward::{
    AttentionBackwardOperand, AttentionBackwardOperands, AttentionBackwardWorkItem,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::Grid;

/// A microkernel of this kind, as a walk reads it.
pub trait AttentionBackwardMicrokernel:
    Microkernel<
        WorkItem = AttentionBackwardWorkItem,
        Operand = AttentionBackwardOperand,
        Partitioning = Partitioning,
    > + core::fmt::Display
{
}

impl<M> AttentionBackwardMicrokernel for M where
    M: Microkernel<
            WorkItem = AttentionBackwardWorkItem,
            Operand = AttentionBackwardOperand,
            Partitioning = Partitioning,
        > + core::fmt::Display
{
}

/// One launch of a routine: its microkernel, the partitioning that places its work items, the
/// width each operand binds at, the grid it spawns, and how the routine lays out the shared
/// memory the walk holds; `()` for one that holds none.
#[derive(Debug, Clone)]
pub struct AttentionBackwardWalk<M, SM = ()> {
    pub microkernel: M,
    pub partitioning: Partitioning,
    pub widths: AttentionBackwardOperands<usize>,
    pub grid: Grid,
    pub shared_memory: SM,
}

impl<M: AttentionBackwardMicrokernel, SM> AttentionBackwardWalk<M, SM> {
    /// `microkernel` placed by `partitioning` on `grid`, shared memory laid out as
    /// `shared_memory`, its widths settled.
    ///
    /// To the requirements the microkernel enqueues, the binding adds its own: each operand's
    /// buffer alignment, where its view starts, and the masking of an operand whose innermost
    /// axis the tiles overhang. Each operand then binds at the widest width this device reads its
    /// element in that they admit.
    pub fn new(
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        microkernel: M,
        partitioning: Partitioning,
        grid: Grid,
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
                vectorization.require(VectorRequirement::Masked { operand });
            }
        }
        let widths = elements
            .map(|operand, element| vectorization.width(operand, |_| hw.vector_ceiling(element)));
        Self {
            microkernel,
            partitioning,
            widths,
            grid,
            shared_memory,
        }
    }

    /// The partitioning the walk's kernel compiles against: every extent a launch varies — the
    /// batches, the key heads, the queries and the keys — left to the launch, and the group and
    /// the dims a row spans compiled in, which a row's registers are sized by.
    pub fn kernel_partitioning(&self) -> Partitioning {
        Partitioning::new(
            self.partitioning
                .space()
                .clone()
                .with_dynamic(&[B, KV, Q, S]),
            self.partitioning.levels().to_vec(),
        )
    }
}

/// What a routine of this kind compiles: the walk owned by the queries, which writes their
/// gradient and each one's dot product, then the walk owned by the keys, which reads the dot
/// products and writes the keys' and the values' gradients.
#[derive(Debug, Clone)]
pub struct AttentionBackwardBlueprint<QM, KM, SM = ()> {
    pub query_gradient: AttentionBackwardWalk<QM, SM>,
    pub key_gradient: AttentionBackwardWalk<KM, SM>,
}

/// What routine `R` compiles.
pub type AttentionBackwardRoutineBlueprint<R> = AttentionBackwardBlueprint<
    <R as AttentionBackwardRoutine>::QueryGradient,
    <R as AttentionBackwardRoutine>::KeyGradient,
    <R as AttentionBackwardRoutine>::SharedMemory,
>;

/// A routine of this kind: its two walks prepared for a problem, then launched in order.
///
/// A routine of the tile DSL states one launch ([`Routine`](cubek_std::routine::Routine)); one of
/// this kind is two, each placed by its own microkernel and partitioning, so it states both. What
/// autotune races is the routine: the two launches together.
pub trait AttentionBackwardRoutine {
    /// The knobs a measurement settles: each walk's microkernel's, and the routine's own.
    type Settings: core::fmt::Debug + Clone + core::fmt::Display + PartialEq;
    type QueryGradient: AttentionBackwardMicrokernel;
    type KeyGradient: AttentionBackwardMicrokernel;
    /// How a walk lays out the shared memory it holds; `()` for one that holds none.
    type SharedMemory;

    /// The settings worth measuring on this device, the one a caller gets by default first.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings>;

    /// Both walks for `problem` on this device.
    ///
    /// # Errors
    ///
    /// Why it cannot run here.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<AttentionBackwardRoutineBlueprint<Self>, Refusal>;

    /// Launch both walks `blueprint` compiles, the queries' first, over `tensors`.
    fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        blueprint: AttentionBackwardRoutineBlueprint<Self>,
    );

    /// The gradients `tensors` state, launched on their client's device, prepared for `hw`.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the backward pass at these settings.
    fn launch_on(
        &self,
        hw: &Hardware,
        tensors: AttentionBackwardTensors,
        settings: &Self::Settings,
    ) -> Result<AttentionBackwardGradients, Refusal> {
        let blueprint = self.prepare(hw, &tensors.problem(), settings)?;
        self.dispatch(&tensors, blueprint);
        Ok(tensors.into_gradients())
    }
}
