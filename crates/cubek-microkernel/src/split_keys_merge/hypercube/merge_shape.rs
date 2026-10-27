use crate::split_keys_merge::SlotShape;

/// What a merge of slots is compiled for: the slots of every query, the cubes sharing a box, which
/// count themselves in, and the queries a merging plane holds
/// ([`merge_shape`](crate::attention::hypercube::HypercubeAttentionMicrokernel::merge_shape)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeMergeShape {
    pub slots: SlotShape,
    pub cubes: usize,
    pub seq_q: usize,
}
