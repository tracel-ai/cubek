//! Leaf reduction kinds for [`Tile::reduce_axis`](crate::Tile::reduce_axis).

use cubecl::prelude::*;

use crate::instruction::{max as inst_max, min as inst_min, sum as inst_sum};

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

#[cube]
impl ReduceLeafKind {
    /// `inst`'s identity element: folding it in leaves the other operand unchanged. What a masked
    /// read past an operand's valid extent must return instead of a shared zero, since zero is
    /// Sum's identity but biases Max toward it (any negative data) and Min away from it (any
    /// positive data).
    pub fn identity<E: Numeric>(#[comptime] inst: ReduceLeafKind) -> E {
        match comptime!(inst) {
            ReduceLeafKind::Sum => E::from_int(0),
            ReduceLeafKind::Max => E::min_value(),
            ReduceLeafKind::Min => E::max_value(),
        }
    }

    /// Fold `rhs` into `lhs` under `inst`.
    pub(crate) fn combine<E: Numeric>(lhs: E, rhs: E, #[comptime] inst: ReduceLeafKind) -> E {
        match comptime!(inst) {
            ReduceLeafKind::Sum => lhs + rhs,
            ReduceLeafKind::Max => max(lhs, rhs),
            ReduceLeafKind::Min => min(lhs, rhs),
        }
    }

    /// Lane-wise [`combine`](ReduceLeafKind::combine). Separate because `Vector` is not `Numeric`.
    pub(crate) fn combine_vector<E: Numeric, V: Size>(
        lhs: Vector<E, V>,
        rhs: Vector<E, V>,
        #[comptime] inst: ReduceLeafKind,
    ) -> Vector<E, V> {
        match comptime!(inst) {
            ReduceLeafKind::Sum => lhs + rhs,
            ReduceLeafKind::Max => max(lhs, rhs),
            ReduceLeafKind::Min => min(lhs, rhs),
        }
    }

    /// Fold `value` across the plane under `inst`, leaving every lane holding the plane's total.
    pub(crate) fn plane_broadcast<E: Numeric, V: Size>(
        value: Vector<E, V>,
        #[comptime] inst: ReduceLeafKind,
    ) -> Vector<E, V> {
        match comptime!(inst) {
            ReduceLeafKind::Sum => plane_sum(value),
            ReduceLeafKind::Max => plane_max(value),
            ReduceLeafKind::Min => plane_min(value),
        }
    }

    /// Fold `value` across each lane group of `fold_mask` under `inst`, leaving every lane of a
    /// group holding that group's total. The mask bits must be the group's lane bits; see
    /// [`instruction::sum::group`](crate::instruction::sum::group).
    pub(crate) fn group_fold<E: Numeric, V: Size>(
        value: Vector<E, V>,
        #[comptime] fold_mask: usize,
        #[comptime] inst: ReduceLeafKind,
    ) -> Vector<E, V> {
        match comptime!(inst) {
            ReduceLeafKind::Sum => inst_sum::group::<E, V>(value, fold_mask),
            ReduceLeafKind::Max => inst_max::group::<E, V>(value, fold_mask),
            ReduceLeafKind::Min => inst_min::group::<E, V>(value, fold_mask),
        }
    }
}
