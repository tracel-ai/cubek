//! Leaf reduction kinds for [`Tile::reduce_axis`](crate::Tile::reduce_axis).

use cubecl::prelude::*;

/// The arithmetic reduction operation executed at leaf tiles.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ReduceLeafKind {
    /// Summation reduction (`acc += val`).
    Sum,
    /// Maximum reduction (`acc = max(acc, val)`).
    Max,
    /// Minimum reduction (`acc = min(acc, val)`).
    Min,
}

/// `inst`'s identity element: folding it in leaves the other operand unchanged. What a masked
/// read past an operand's valid extent must return instead of a shared zero, since zero is Sum's
/// identity but biases Max toward it (any negative data) and Min away from it (any positive data).
#[cube]
pub fn reduce_identity<E: Numeric>(#[comptime] inst: ReduceLeafKind) -> E {
    match comptime!(inst) {
        ReduceLeafKind::Sum => E::from_int(0),
        ReduceLeafKind::Max => E::min_value(),
        ReduceLeafKind::Min => E::max_value(),
    }
}
