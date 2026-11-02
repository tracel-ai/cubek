//! What a routine compiles, its two walks, and the grids they spawn beside them.

use cubek_std::routine::Grid;

/// What a routine of this kind compiles: the walk owned by the queries, which writes each one's
/// dot product of the output with its gradient, then the walk owned by the keys, which reads the
/// dot products and writes the keys' and the values' gradients. The query's gradient is the first
/// walk's where the queries own it, and summed by the second where the keys do, as the hypercube
/// routine's cubes each add their keys' share.
///
/// Each walk is an [`AttentionBackwardWalk`](crate::kernels::AttentionBackwardWalk), whose type
/// names its microkernel and how it hands its boxes out: a routine's blueprint reads as its two
/// launches.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionBackwardBlueprint<Q, K> {
    pub query_walk: Q,
    pub key_walk: K,
}

/// What the two walks of an [`AttentionBackwardBlueprint`] spawn on this device, beside it: the
/// grids read off their partitionings at the plane width this device runs.
#[derive(Debug, Clone)]
pub struct AttentionBackwardGrid {
    pub query_walk: Grid,
    pub key_walk: Grid,
}
