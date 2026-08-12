//! Sum microkernels: vector horizontal sum, array sum, plane-cooperative sum, and lane group folding.

use cubecl::prelude::*;

/// Horizontal sum of a vector's `width` lanes.
#[cube]
pub fn vector<E: Numeric, N: Size>(v: Vector<E, N>, #[comptime] width: usize) -> E {
    let mut s = E::from_int(0);
    #[unroll]
    for j in 0..width {
        s += v.extract(j);
    }
    s
}

/// Horizontal sum across an array of elements (identity-seeded with 0).
#[cube]
pub fn array<E: Numeric>(arr: &Array<E>, #[comptime] len: usize) -> E {
    array_from(arr, len, E::from_int(0))
}

/// Horizontal sum across an array of elements, starting from `seed`.
#[cube]
pub fn array_from<E: Numeric>(arr: &Array<E>, #[comptime] len: usize, seed: E) -> E {
    let mut s = seed;
    #[unroll]
    for i in 0..len {
        s += arr[i];
    }
    s
}

/// Plane-cooperative sum with fallback for 1-lane/CPU runtimes.
#[cube]
pub fn plane<E: Numeric>(val: E, #[comptime] lanes: usize) -> E {
    if comptime!(lanes > 1) {
        plane_sum(val)
    } else {
        val
    }
}

/// Combine a lane group's partials, leaving every lane of the group holding the group's total.
///
/// One butterfly step per bit of `fold_mask`. A cell's partials sit on the lanes that agree
/// outside the mask and differ inside it, so an xor by a single mask bit stays within the group
/// — every group folds at once, each over its own cell, with no guard and no branch.
///
/// If the whole plane shares one cell without carries ([`LaneShare::Plane`](crate::LaneShare::Plane)),
/// [`plane`] is the better instruction. The `fold_mask` bits must be the group's lane bits, since
/// a wrong mask gives silently wrong results rather than an error.
#[cube]
pub fn group<E: Numeric, V: Size>(
    value: Vector<E, V>,
    #[comptime] fold_mask: usize,
) -> Vector<E, V> {
    let mut total = value;
    #[unroll]
    for bit in 0..comptime!(usize::BITS - fold_mask.leading_zeros()) {
        if comptime!(fold_mask & (1 << bit) != 0) {
            total = total + plane_shuffle_xor(total, comptime!(1u32 << bit));
        }
    }
    total
}
