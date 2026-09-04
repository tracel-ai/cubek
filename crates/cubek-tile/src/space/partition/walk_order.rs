//! The orders a [`Walk`](crate::Walk) visits its regions in.

use crate::{Fold, FoldExpand};
use cubecl::prelude::*;

/// A new order is a new variant here plus a [`walk_index`] arm.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WalkOrder {
    /// step `i` visits odometer index `i` (the identity).
    RowMajor,
    /// step `i` visits `total - i - 1`.
    Reversed,
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
