use super::Space;
use crate::{Axis, Coords, Fold, FoldExpand};
use cubecl::prelude::*;

/// One region of a partitioned [`Space`]: the subset the walk visits at a step,
/// a `Space` at an origin. Coordinates carry their constness: a static walk's fold
/// to comptime constants, so a region can select fragments as well as window memory.
#[derive(CubeType)]
pub struct Region {
    coords: Coords<u32>,
    #[cube(comptime)]
    space: Space,
}

#[cube]
impl Region {
    pub fn new(coords: Coords<u32>, #[comptime] space: Space) -> Region {
        Region { coords, space }
    }

    /// The region at trailing-two coordinates `(c0, c1)`, `0` elsewhere. The coordinates carry
    /// their own constness ([`fcast`](crate::Fold::fcast) keeps a constant constant): comptime
    /// ones fold to constants and can select fragments, ones the kernel computed (the visit a
    /// worker picked out of a grid by hardware position) window memory.
    pub fn trailing(#[comptime] space: Space, c0: usize, c1: usize) -> Region {
        let rank = comptime!(space.rank());
        let mut coords = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            // `fcast`, not `as`: a comptime coordinate has to stay a constant or it could no
            // longer select a fragment. `runtime` on the `0` moves the literal into the expand
            // domain and keeps it constant too.
            let c = if comptime!(p == rank - 2) {
                c0.fcast::<u32>()
            } else if comptime!(p == rank - 1) {
                c1.fcast::<u32>()
            } else {
                0u32.runtime()
            };
            coords.push(c);
        }
        Region::new(coords, comptime!(space.clone()))
    }

    /// The coordinate along `axis`; `0` when the axis is absent (broadcast by omission:
    /// the tile spans all of it).
    pub fn coord(&self, #[comptime] axis: Axis) -> usize {
        if comptime!(self.space.contains(axis)) {
            self.coords
                .at(comptime!(self.space.position(axis)))
                .fcast::<usize>()
        } else {
            0usize.runtime()
        }
    }
}
