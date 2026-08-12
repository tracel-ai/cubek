//! Leaf reduction kinds for [`Tile::reduce_axis`](crate::Tile::reduce_axis).

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
