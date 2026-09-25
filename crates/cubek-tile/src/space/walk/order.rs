//! The order a [`Walk`](crate::Walk)'s steps visit the odometer.

use crate::{Integer, IntegerExpand};
use cubecl::prelude::*;

/// The direction a walk's steps take through its grid. A new order is a new variant here plus a
/// [`step`](StepOrder::step) arm.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum StepOrder {
    /// Step `i` visits odometer index `i` (the identity).
    #[default]
    Forward,
    /// Step `i` visits `total - i - 1`.
    Reversed,
}

#[cube]
impl StepOrder {
    /// The odometer index step `i` of `total` visits under `order`.
    pub(crate) fn step(i: usize, total: usize, #[comptime] order: StepOrder) -> usize {
        match order {
            StepOrder::Forward => i,
            // Folded: an unrolled walk's constant `i` must stay constant through the
            // reversal, or its regions lose their comptime coordinates.
            StepOrder::Reversed => total.minus(i).minus(1),
        }
    }
}
