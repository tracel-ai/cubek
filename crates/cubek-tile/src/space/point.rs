use super::{Axis, Space};
use cubecl::prelude::*;

/// A resolved grid coordinate: one index per axis of its frame (the operation
/// [`Space`] the walk ranges over). Tiles read it [by axis](Point::get), so they
/// need only their own axes.
#[derive(CubeType)]
pub struct Point {
    coords: Sequence<usize>,
    #[cube(comptime)]
    frame: Space,
}

#[cube]
impl Point {
    /// Wrap per-axis runtime coordinates (in `frame` order) as a point.
    pub fn new(coords: Sequence<usize>, #[comptime] frame: Space) -> Point {
        Point { coords, frame }
    }

    /// The coordinate along `axis`, located via the point's frame.
    pub fn get(&self, #[comptime] axis: Axis) -> usize {
        *self.coords.index(comptime!(self.frame.position(axis)))
    }
}
