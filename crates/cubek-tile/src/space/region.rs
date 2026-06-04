use super::{Axis, Space};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

/// One region of a partitioned [`Space`] — the (sub-)Space the walk visits at a
/// step. It is a `Space` at an origin: the operation space it lies in (read for
/// the axis order and composite-batch factorization) plus the per-axis tile
/// coordinate locating it (runtime, since a spatial walk folds in the hardware
/// position). A [`Tile`](crate::Tile) locates itself at a region with
/// [`at`](crate::Tile::at) — projecting the region onto its *own* axes internally,
/// so different operands stay aligned with no caller-side axis bookkeeping.
#[derive(CubeType)]
pub struct Region {
    coords: CoordsDyn,
    #[cube(comptime)]
    space: Space,
}

#[cube]
impl Region {
    /// Wrap per-axis tile coordinates (in `space` order) at an origin in `space`.
    pub fn new(coords: CoordsDyn, #[comptime] space: Space) -> Region {
        Region { coords, space }
    }

    /// The operation space this region lies in — its axes and extents (including a
    /// composite axis's levels).
    pub fn space(&self) -> comptime_type!(Space) {
        comptime!(self.space.clone())
    }

    /// The tile coordinate along `axis`, located via the space's axis order. A
    /// tile reads only the axes it carries, so different operands stay aligned.
    pub fn coord(&self, #[comptime] axis: Axis) -> usize {
        self.coords[comptime!(self.space.position(axis))] as usize
    }
}
