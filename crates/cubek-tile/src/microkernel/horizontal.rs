//! 1-D register folds: a vector's lanes, or an array's elements, folded under a [`LeafOp`].

use cubecl::prelude::*;

use crate::LeafOp;

/// Fold a vector's first `width` lanes under `op`, seeded with `op`'s identity.
#[cube]
pub fn vector<E: Numeric, N: Size>(
    v: Vector<E, N>,
    #[comptime] width: usize,
    #[comptime] op: LeafOp,
) -> E {
    let mut acc = LeafOp::identity::<E>(op);
    #[unroll]
    for j in 0..width {
        acc = LeafOp::combine::<E>(acc, v.extract(j), op);
    }
    acc
}

/// Fold `len` elements of `arr` under `op`, seeded with `op`'s identity.
#[cube]
pub fn array<E: Numeric>(arr: &Array<E>, #[comptime] len: usize, #[comptime] op: LeafOp) -> E {
    array_from(arr, len, LeafOp::identity::<E>(op), op)
}

/// Fold `len` elements of `arr` under `op`, starting from `seed`.
#[cube]
pub fn array_from<E: Numeric>(
    arr: &Array<E>,
    #[comptime] len: usize,
    seed: E,
    #[comptime] op: LeafOp,
) -> E {
    let mut acc = seed;
    #[unroll]
    for i in 0..len {
        acc = LeafOp::combine::<E>(acc, arr[i], op);
    }
    acc
}
