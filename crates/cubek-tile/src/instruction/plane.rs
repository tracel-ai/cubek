//! Plane-cooperative instructions: the hardware plane reduction, and the shuffle butterfly that
//! folds a lane group. Each is one fixed instruction sequence over a value already in a register,
//! with no loop over data; the nests that call them repeatedly live in
//! [`microkernel`](crate::microkernel).

use cubecl::prelude::*;

use super::LeafOp;

/// Plane-cooperative fold of `val` under `op`, leaving every lane holding the plane's total.
/// Falls back to the value itself on 1-lane/CPU runtimes.
#[cube]
pub fn reduce<E: Numeric>(val: E, #[comptime] lanes: usize, #[comptime] op: LeafOp) -> E {
    if comptime!(lanes > 1) {
        match comptime!(op) {
            LeafOp::Sum => plane_sum(val),
            LeafOp::Max => plane_max(val),
            LeafOp::Min => plane_min(val),
        }
    } else {
        val
    }
}

/// Lane-wise [`reduce()`]. Separate because `Vector` is not `Numeric`, and unconditional because its
/// caller reaches it only under [`LaneShare::Plane`](crate::LaneShare::Plane), which a runtime
/// with a 1-lane plane never carries.
#[cube]
pub fn reduce_vector<E: Numeric, V: Size>(
    value: Vector<E, V>,
    #[comptime] op: LeafOp,
) -> Vector<E, V> {
    match comptime!(op) {
        LeafOp::Sum => plane_sum(value),
        LeafOp::Max => plane_max(value),
        LeafOp::Min => plane_min(value),
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
/// [`reduce_vector()`] is the better instruction. The `fold_mask` bits must be the group's lane bits,
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
            total = LeafOp::combine_vector(
                total,
                plane_shuffle_xor(total, comptime!(1u32 << bit)),
                op,
            );
        }
    }
    total
}
