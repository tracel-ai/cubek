use super::Space;
use crate::Axis;
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

    /// The coordinate along `axis`; `0` when the axis is absent (broadcast by omission:
    /// the tile spans all of it).
    pub fn coord(&self, #[comptime] axis: Axis) -> usize {
        if comptime!(self.space.contains(axis)) {
            self.coords[comptime!(self.space.position(axis))] as usize
        } else {
            0usize.runtime()
        }
    }

    /// The runtime form of a static region. Memory windowing is coordinate-kind-agnostic,
    /// so the register-tier walk reuses [`Tile::at`](crate::Tile::at) through this.
    pub(crate) fn from_static(#[comptime] region: &StaticRegion) -> Region {
        let mut coords = CoordsDyn::new();
        #[unroll]
        for p in 0..comptime!(region.space().rank()) {
            coords.push(comptime!(region.coord_at(p)) as u32);
        }
        Region::new(coords, comptime!(region.space().clone()))
    }
}

/// [`Region`]'s static sibling, for the register tier: fragments are comptime-indexed,
/// so a walk over them carries its coordinates as host data.
#[derive(Clone, Debug)]
pub struct StaticRegion {
    coords: Vec<usize>,
    space: Space,
}

impl StaticRegion {
    pub fn new(coords: Vec<usize>, space: Space) -> StaticRegion {
        assert!(coords.len() == space.rank(), "StaticRegion: rank mismatch");
        StaticRegion { coords, space }
    }

    /// The region at trailing-two coordinates `(c0, c1)`, `0` elsewhere.
    pub fn trailing(space: &Space, c0: usize, c1: usize) -> StaticRegion {
        let rank = space.rank();
        let mut coords = vec![0; rank];
        coords[rank - 2] = c0;
        coords[rank - 1] = c1;
        StaticRegion::new(coords, space.clone())
    }

    /// The coordinate along `axis`; `0` when the axis is absent (broadcast by omission:
    /// the tile spans all of it).
    pub fn coord(&self, axis: Axis) -> usize {
        if self.space.contains(axis) {
            self.coords[self.space.position(axis)]
        } else {
            0
        }
    }

    pub fn coord_at(&self, p: usize) -> usize {
        self.coords[p]
    }

    pub fn space(&self) -> &Space {
        &self.space
    }
}
