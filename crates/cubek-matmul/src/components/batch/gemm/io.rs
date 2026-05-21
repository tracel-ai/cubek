use cubecl::prelude::*;
use cubecl::{cube, std::tensor::View, std::tensor::layout::Coordinates};

use crate::components::batch::CheckBounds;

/// Read a value from `view` at `coords`
#[cube]
pub fn read<T: CubePrimitive, C: Coordinates>(
    view: &View<T, C>,
    coords: C,
    #[comptime] check_bounds: CheckBounds,
) -> T {
    if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
        view.read_checked(coords)
    } else {
        view.read_unchecked(coords)
    }
}

/// Write `value` into `view` at `coord`
#[cube]
pub fn write<T: CubePrimitive, C: Coordinates>(
    view: &View<T, C, ReadWrite>,
    coord: C,
    value: T,
    #[comptime] check_bounds: CheckBounds,
) {
    if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
        view.write_checked(coord, value);
    } else {
        view.write(coord, value);
    }
}
