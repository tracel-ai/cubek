//! What a routine compiles: its two walks.

use crate::kernels::{AttentionBackwardRoutine, AttentionBackwardWalk};

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
