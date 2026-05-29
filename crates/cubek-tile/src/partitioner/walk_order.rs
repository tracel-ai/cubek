//! Concrete walk orders — example traversal policies plugged into the
//! [`Partitioner`] seam, not core machinery. The odometer in [`walk`](super::walk)
//! already walks natural order; an order just remaps step `i` to an odometer
//! index. New orders are added here, not in the engine.

use cubecl::prelude::*;

use crate::ByAxis;

use super::{Distribution, Partitioner};

/// The order a [`Partitioner`] visits its walk steps. A new order is a new
/// variant here plus a [`walk_index`] arm; nothing downstream branches on it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WalkOrder {
    /// Ascending: step `i` visits odometer index `i` (the identity).
    RowMajor,
    /// Descending: step `i` visits `total - i - 1`.
    Reversed,
}

impl Partitioner {
    /// Declared axis order, last axis fastest — the natural nested walk.
    pub fn row_major(sub_tile: ByAxis<usize>, dists: ByAxis<Distribution>) -> Self {
        Partitioner::new(sub_tile, dists, WalkOrder::RowMajor)
    }

    /// Same split, walked back-to-front.
    pub fn reversed(sub_tile: ByAxis<usize>, dists: ByAxis<Distribution>) -> Self {
        Partitioner::new(sub_tile, dists, WalkOrder::Reversed)
    }
}

/// The odometer index visited at walk step `i` of `total` — one arm per
/// [`WalkOrder`].
#[cube]
pub(crate) fn walk_index(i: usize, total: usize, #[comptime] order: WalkOrder) -> usize {
    match order {
        WalkOrder::RowMajor => i,
        WalkOrder::Reversed => total - i - 1,
    }
}
