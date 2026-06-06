use cubecl::prelude::*;

// Normalize each element to a 0/1 flag: non-zero -> 1, zero -> 0.
// On {0, 1}, OR is exactly `max` and AND is exactly `min`, which lets the logical
// reductions (`Any` / `All`) reuse the vectorized `plane_max` / `plane_min` machinery
// while keeping a narrow accumulator (no overflow, unlike a sum-based emulation).
#[cube]
pub(crate) fn normalize_to_flag<E: Numeric, N: Size>(item: Vector<E, N>) -> Vector<E, N> {
    let zero = Vector::empty().fill(E::from_int(0));
    let one = Vector::empty().fill(E::from_int(1));
    select_many(item.not_equal(&zero), one, zero)
}

// Using plane operations, return the lowest coordinate for each vector element
// for which the item equal the target.
#[cube]
pub(crate) fn lowest_coordinate_matching<E: Scalar, N: Size>(
    target: Vector<E, N>,
    item: Vector<E, N>,
    coordinate: Vector<u32, N>,
) -> Vector<u32, N> {
    let is_candidate = item.equal(&target);
    let candidate_coordinate =
        select_many(is_candidate, coordinate, Vector::empty().fill(u32::MAX));
    plane_min(candidate_coordinate)
}
