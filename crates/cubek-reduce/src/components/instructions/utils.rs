use cubecl::prelude::*;

// Using plane operations, return the lowest coordinate for each line element
// for which the item equal the target.
#[cube]
pub(crate) fn lowest_coordinate_matching<E: CubePrimitive, N: Size>(
    target: Line<E, N>,
    item: Line<E, N>,
    coordinate: Line<u32, N>,
) -> Line<u32, N> {
    let is_candidate = item.equal(target);
    let candidate_coordinate = select_many(is_candidate, coordinate, Line::empty().fill(u32::MAX));
    plane_min(candidate_coordinate)
}
