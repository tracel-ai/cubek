//! Concrete walk orders: example traversal policies plugged into the partitioner.

use super::{Distribution, Partitioner, PartitionerBuilder};
use crate::{ByAxis, Fold, FoldExpand};
use cubecl::prelude::*;

/// A new order is a new variant here plus a [`walk_index`] arm.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WalkOrder {
    /// step `i` visits odometer index `i` (the identity).
    RowMajor,
    /// step `i` visits `total - i - 1`.
    Reversed,
}

impl Partitioner {
    /// Declared axis order, last axis fastest.
    pub fn row_major(sub_tile: ByAxis<usize>, dists: ByAxis<Distribution>) -> PartitionerBuilder {
        PartitionerBuilder::new(sub_tile, dists, WalkOrder::RowMajor)
    }

    pub fn reversed(sub_tile: ByAxis<usize>, dists: ByAxis<Distribution>) -> PartitionerBuilder {
        PartitionerBuilder::new(sub_tile, dists, WalkOrder::Reversed)
    }
}

#[cube]
pub(crate) fn walk_index(i: usize, total: usize, #[comptime] order: WalkOrder) -> usize {
    match order {
        WalkOrder::RowMajor => i,
        // Folded: an unrolled walk's constant `i` must stay constant through the
        // reversal, or its regions lose their comptime coordinates.
        WalkOrder::Reversed => total.fsub(i).fsub(1),
    }
}
