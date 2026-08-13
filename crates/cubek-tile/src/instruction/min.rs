//! Min microkernels: vector horizontal min, array min, and plane-cooperative min.

use cubecl::prelude::*;

/// Horizontal minimum of a vector's `width` lanes (`width > 0`).
#[cube]
pub fn vector<E: Numeric, N: Size>(v: Vector<E, N>, #[comptime] width: usize) -> E {
    comptime!(assert!(width > 0, "min::vector requires width > 0"));
    let mut m = v.extract(0usize);
    #[unroll]
    for j in 1..width {
        m = min(m, v.extract(j));
    }
    m
}

/// Horizontal minimum across an array of elements (identity-seeded with `E::max_value()`).
#[cube]
pub fn array<E: Numeric>(arr: &Array<E>, #[comptime] len: usize) -> E {
    array_from(arr, len, E::max_value())
}

/// Horizontal minimum across an array of elements, starting from `seed`.
#[cube]
pub fn array_from<E: Numeric>(arr: &Array<E>, #[comptime] len: usize, seed: E) -> E {
    let mut m = seed;
    #[unroll]
    for i in 0..len {
        m = min(m, arr[i]);
    }
    m
}

/// Plane-cooperative minimum with fallback for 1-lane/CPU runtimes.
#[cube]
pub fn plane<E: Numeric>(val: E, #[comptime] lanes: usize) -> E {
    if comptime!(lanes > 1) {
        plane_min(val)
    } else {
        val
    }
}
