//! Plane-cooperative instructions: the hardware plane reduction, and the shuffle butterfly that
//! folds a lane group. Each is one fixed instruction sequence over a value already in a register,
//! with no loop over data; the nests that call them repeatedly live in
//! [`microkernel`](crate::microkernel).
//!
//! All three take scalars and lines alike, since the hardware plane ops are themselves generic
//! over a vectorized element.

use cubecl::prelude::*;

use super::LeafOp;

/// The plane instruction itself: fold `val` under `op` across the whole plane, leaving every lane
/// holding the total. Unguarded, for callers that already know the plane carries real lanes.
#[cube]
pub fn broadcast<T: CubePrimitive<Scalar: PlaneNumeric>>(val: T, #[comptime] op: LeafOp) -> T {
    match comptime!(op) {
        LeafOp::Sum => plane_sum(val),
        LeafOp::Max => plane_max(val),
        LeafOp::Min => plane_min(val),
    }
}

/// [`broadcast()`] with the 1-lane/CPU fallback: a plane of one is already its own total, and the
/// intrinsic does not lower there.
#[cube]
pub fn reduce<T: CubePrimitive<Scalar: PlaneNumeric>>(
    val: T,
    #[comptime] lanes: usize,
    #[comptime] op: LeafOp,
) -> T {
    if comptime!(lanes > 1) {
        broadcast::<T>(val, op)
    } else {
        val
    }
}

/// Combine a lane group's partials under `op`, leaving every lane of the group holding the
/// group's total.
///
/// One butterfly step per bit of `fold_mask`. A cell's partials sit on the lanes that agree
/// outside the mask and differ inside it, so an xor by a single mask bit stays within the group
/// — every group folds at once, each over its own cell, with no guard and no branch.
///
/// If the whole plane shares one cell without carries ([`LaneShare::Plane`](crate::LaneShare::Plane)),
/// [`broadcast()`] is the better instruction. The `fold_mask` bits must be the group's lane bits,
/// since a wrong mask gives silently wrong results rather than an error.
#[cube]
pub fn group<E: Numeric, V: Size>(
    value: Vector<E, V>,
    #[comptime] fold_mask: usize,
    #[comptime] op: LeafOp,
) -> Vector<E, V> {
    let mut total = value;
    #[unroll]
    for bit in 0..comptime!(usize::BITS - fold_mask.leading_zeros()) {
        if comptime!(fold_mask & (1 << bit) != 0) {
            total = LeafOp::combine::<Vector<E, V>>(
                total,
                plane_shuffle_xor(total, comptime!(1u32 << bit)),
                op,
            );
        }
    }
    total
}
