//! What a routine compiles: its two walks.

use crate::kernels::{AttentionBackwardRoutine, AttentionBackwardWalk};

/// What a routine of this kind compiles: the walk owned by the queries, which writes their
/// gradient and each one's dot product, then the walk owned by the keys, which reads the dot
/// products and writes the keys' and the values' gradients.
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
