//! 1-D register folds: a vector's lanes, or an array's elements, folded under a [`Monoid`].

use cubecl::prelude::*;

use crate::Monoid;

/// Fold a vector's first `width` lanes under `monoid`, seeded with `monoid`'s identity.
#[cube]
pub fn vector<E: Numeric, N: Size>(
    v: Vector<E, N>,
    #[comptime] width: usize,
    #[comptime] monoid: Monoid,
) -> E {
    let mut acc = Monoid::identity::<E>(monoid);
    #[unroll]
    for j in 0..width {
        acc = monoid.fold::<E>(acc, v.extract(j));
    }
    acc
}

/// Fold `len` elements of `arr` under `monoid`, seeded with `monoid`'s identity.
#[cube]
pub fn array<E: Numeric>(arr: &Array<E>, #[comptime] len: usize, #[comptime] monoid: Monoid) -> E {
    array_from(arr, len, Monoid::identity::<E>(monoid), monoid)
}

/// Fold `len` elements of `arr` under `monoid`, starting from `seed`.
#[cube]
pub fn array_from<E: Numeric>(
    arr: &Array<E>,
    #[comptime] len: usize,
    seed: E,
    #[comptime] monoid: Monoid,
) -> E {
    let mut acc = seed;
    #[unroll]
    for i in 0..len {
        acc = monoid.fold::<E>(acc, arr[i]);
    }
    acc
}
