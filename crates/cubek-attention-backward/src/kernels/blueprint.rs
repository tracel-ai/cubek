//! What a routine compiles: its two walks.

use crate::kernels::{AttentionBackwardRoutine, AttentionBackwardWalk};

/// What a routine of this kind compiles: the walk owned by the queries, which writes each one's
/// dot product of the output with its gradient, then the walk owned by the keys, which reads the
/// dot products and writes the keys' and the values' gradients. The query's gradient is the first
/// walk's where the queries own it, and summed by the second where the keys do, as the hypercube
/// routine's cubes each add their keys' share.
#[derive(Debug, Clone)]
pub struct AttentionBackwardBlueprint<QM, KM, SM = ()> {
    pub query_walk: AttentionBackwardWalk<QM, SM>,
    pub key_walk: AttentionBackwardWalk<KM, SM>,
}

/// What routine `R` compiles.
pub type AttentionBackwardRoutineBlueprint<R> = AttentionBackwardBlueprint<
    <R as AttentionBackwardRoutine>::QueryMicrokernel,
    <R as AttentionBackwardRoutine>::KeyMicrokernel,
    <R as AttentionBackwardRoutine>::SharedMemory,
>;
