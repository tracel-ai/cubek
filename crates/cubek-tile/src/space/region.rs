use super::{Axis, Space};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

/// One region of a partitioned [`Space`]: the subset the walk visits at a step,
/// a `Space` at an origin.
#[derive(CubeType)]
pub struct Region {
    coords: CoordsDyn,
    #[cube(comptime)]
    space: Space,
}

#[cube]
impl Region {
    pub fn new(coords: CoordsDyn, #[comptime] space: Space) -> Region {
        Region { coords, space }
    }

    pub fn coord(&self, #[comptime] axis: Axis) -> usize {
        self.coords[comptime!(self.space.position(axis))] as usize
    }
}
