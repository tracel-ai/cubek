//! The algebra a leaf folds under: an identity and a combine.

use cubecl::prelude::*;

/// The arithmetic a leaf folds under. One vocabulary for the three layers that need it: the
/// plane instructions ([`plane`](super::plane)), the register nests
/// ([`instruction`](crate::instruction::registers)), and the verb that schedules them
/// ([`Tile::reduce_axis`](crate::Tile::reduce_axis)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LeafOp {
    /// Summation reduction (`acc += val`).
    Sum,
    /// Maximum reduction (`acc = max(acc, val)`).
    Max,
    /// Minimum reduction (`acc = min(acc, val)`).
    Min,
}

#[cube]
impl LeafOp {
    /// `op`'s identity element: folding it in leaves the other operand unchanged. What a masked
    /// read past an operand's valid extent must return instead of a shared zero, since zero is
    /// Sum's identity but biases Max toward it (any negative data) and Min away from it (any
    /// positive data).
    pub fn identity<E: Numeric>(#[comptime] op: LeafOp) -> E {
        match comptime!(op) {
            LeafOp::Sum => E::from_int(0),
            LeafOp::Max => E::min_value(),
            LeafOp::Min => E::max_value(),
        }
    }

    /// Fold `rhs` into `lhs` under `op`.
    ///
    /// Generic over scalars and lines alike: these three need only ordering and addition, which
    /// `Vector` has, so the bound stays well below `Numeric` (which it does not have).
    pub fn combine<T: CubePartialOrd + CubeAdd + core::ops::Add<T, Output = T>>(
        lhs: T,
        rhs: T,
        #[comptime] op: LeafOp,
    ) -> T {
        match comptime!(op) {
            LeafOp::Sum => lhs + rhs,
            LeafOp::Max => max(lhs, rhs),
            LeafOp::Min => min(lhs, rhs),
        }
    }
}
