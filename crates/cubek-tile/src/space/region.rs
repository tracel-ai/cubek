use super::{Level, Space};
use crate::{Axis, Coords, Fold, FoldExpand};
use cubecl::prelude::*;

/// One region of a [`Space`] under a [`Level`]: the subset a walk visits at a step, the parent
/// space at an origin, and the level that cut it (what `at` reads to window a tile down).
/// Coordinates carry their constness: a static walk's fold to comptime constants, so a region
/// can select fragments as well as window memory.
#[derive(CubeType)]
pub struct Region {
    coords: Coords<u32>,
    #[cube(comptime)]
    pub(crate) space: Space,
    #[cube(comptime)]
    pub(crate) level: Level,
}

#[cube]
impl Region {
    pub fn new(coords: Coords<u32>, #[comptime] space: Space, #[comptime] level: Level) -> Region {
        Region {
            coords,
            space,
            level,
        }
    }

    /// The region at trailing-two coordinates `(c0, c1)` under `level`, `0` elsewhere. The
    /// coordinates carry their own constness ([`fcast`](crate::Fold::fcast) keeps a constant
    /// constant): comptime ones fold to constants and can select fragments, ones the kernel
    /// computed (the visit a worker picked out of a grid by hardware position) window memory.
    pub fn trailing(
        #[comptime] space: Space,
        #[comptime] level: Level,
        c0: usize,
        c1: usize,
    ) -> Region {
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
        Region::new(coords, comptime!(space.clone()), level)
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
