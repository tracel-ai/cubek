//! Max microkernels: vector horizontal max, array max, and plane-cooperative max.

use cubecl::prelude::*;

/// Horizontal maximum of a vector's `width` lanes (`width > 0`).
#[cube]
pub fn vector<E: Numeric, N: Size>(v: Vector<E, N>, #[comptime] width: usize) -> E {
    comptime!(assert!(width > 0, "max::vector requires width > 0"));
    let mut m = v.extract(0usize);
    #[unroll]
    for j in 1..width {
        m = max(m, v.extract(j));
    }
    m
}

/// Horizontal maximum across an array of elements (identity-seeded with `E::min_value()`).
#[cube]
pub fn array<E: Numeric>(arr: &Array<E>, #[comptime] len: usize) -> E {
    array_from(arr, len, E::min_value())
}

/// Horizontal maximum across an array of elements, starting from `seed`.
#[cube]
pub fn array_from<E: Numeric>(arr: &Array<E>, #[comptime] len: usize, seed: E) -> E {
    let mut m = seed;
    #[unroll]
    for i in 0..len {
        m = max(m, arr[i]);
    }
    m
}

/// Plane-cooperative maximum with fallback for 1-lane/CPU runtimes.
#[cube]
pub fn plane<E: Numeric>(val: E, #[comptime] lanes: usize) -> E {
    if comptime!(lanes > 1) {
        plane_max(val)
    } else {
        val
    }
}
