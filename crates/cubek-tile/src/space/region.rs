//! Where a loop is: a [`Region`] is the path of levels a kernel's loops took from a root space
//! to the box the innermost one handed out, one [`Step`] per level. A tile windows itself to a
//! region with `at`, applying the steps below its own depth, so the root tile and any window of
//! it read the same region.

use super::{Level, Space};
use crate::{Axis, Coords, Fold, FoldExpand};
use cubecl::prelude::*;

/// One level's cut of one space: the tile coordinates a loop over `level` handed out, and the
/// space they index.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Step {
    coords: Coords<u32>,
    #[cube(comptime)]
    pub(crate) space: Space,
    #[cube(comptime)]
    pub(crate) level: Level,
}

#[cube]
impl Step {
    pub(crate) fn new(
        coords: Coords<u32>,
        #[comptime] space: Space,
        #[comptime] level: Level,
    ) -> Step {
        Step {
            coords,
            space,
            level,
        }
    }

    /// The coordinate along `axis`; `0` when the axis is absent (broadcast by omission:
    /// the tile spans all of it).
    pub(crate) fn coord(&self, #[comptime] axis: Axis) -> usize {
        if comptime!(self.space.contains(axis)) {
            self.coords
                .at(comptime!(self.space.position(axis)))
                .fcast::<usize>()
        } else {
            0usize.runtime()
        }
    }
}

/// The path a kernel's loops took to a box: the levels from a root space down, outermost first,
/// and the coordinates each loop handed out. `base` is the depth of the first level, counted in
/// the nest the root sits in, so a tile at any depth on the path applies the steps below it.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Region {
    /// One coordinate set per level, outermost first.
    digits: Sequence<Coords<u32>>,
    /// The runtime sizes of the root's dynamic axes, positional: what a loop below this region
    /// walks its child space with.
    sizes: Sequence<usize>,
    #[cube(comptime)]
    pub(crate) base: usize,
    #[cube(comptime)]
    pub(crate) root: Space,
    #[cube(comptime)]
    pub(crate) levels: Vec<Level>,
}

#[cube]
impl Region {
    pub(crate) fn new(
        digits: Sequence<Coords<u32>>,
        sizes: Sequence<usize>,
        #[comptime] base: usize,
        #[comptime] root: Space,
        #[comptime] levels: Vec<Level>,
    ) -> Region {
        Region {
            digits,
            sizes,
            base,
            root,
            levels,
        }
    }

    /// The empty path at `space`, the space itself sitting at `depth` in its nest: where a loop
    /// over the space starts from.
    pub(crate) fn root(space: &Space, #[comptime] depth: usize) -> Region {
        Region::new(
            Sequence::new(),
            space.extents.sizes.clone(),
            depth,
            comptime!(space.clone()),
            comptime!(Vec::new()),
        )
    }

    /// The region one level below the root at trailing-two coordinates `(c0, c1)` under
    /// `level`, `0` elsewhere, for a tile at `depth`: what a leaf states when it cuts an operand
    /// its own way. The coordinates carry their own constness ([`fcast`](crate::Fold::fcast)
    /// keeps a constant constant): comptime ones fold to constants and can select fragments, ones
    /// the kernel computed (the visit a worker picked out of a grid by hardware position) window
    /// memory.
    pub fn trailing(
        #[comptime] depth: usize,
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
        let mut digits = Sequence::<Coords<u32>>::new();
        digits.push(coords);
        Region::new(
            digits,
            Sequence::new(),
            depth,
            comptime!(space.clone()),
            comptime!(vec![level]),
        )
    }

    /// The `i`-th step of the path.
    pub(crate) fn step(&self, #[comptime] i: usize) -> Step {
        Step::new(
            self.digits.index(i).clone(),
            comptime!(self.space_at(i)),
            comptime!(self.levels[i].clone()),
        )
    }

    /// The box this region covers, as the runtime space a loop below it walks: the root through
    /// every level, an axis left whole keeping the root's size.
    pub(crate) fn child(&self) -> Space {
        Space::with_sizes(comptime!(self.root.leaf(&self.levels)), self.sizes.clone())
    }

    /// The innermost level's coordinate along `axis`; `0` when the axis is absent (broadcast by
    /// omission: the tile spans all of it).
    pub fn coord(&self, #[comptime] axis: Axis) -> usize {
        let last = comptime!(self.levels.len() - 1);
        self.step(last).coord(axis)
    }

    /// This path with one more level's coordinates below it.
    pub(crate) fn below(&self, coords: Coords<u32>, #[comptime] level: Level) -> Region {
        let n = comptime!(self.digits.len());
        let mut digits = Sequence::<Coords<u32>>::new();
        #[unroll]
        for i in 0..n {
            digits.push(self.digits.index(i).clone());
        }
        digits.push(coords);
        let levels = comptime!({
            let mut levels = self.levels.clone();
            levels.push(level);
            levels
        });
        Region::new(
            digits,
            self.sizes.clone(),
            comptime!(self.base),
            comptime!(self.root.clone()),
            levels,
        )
    }
}

/// The comptime shape of a path, the same read on the host and the expand types.
impl Region {
    pub(crate) fn depth(&self) -> usize {
        self.base + self.levels.len()
    }

    pub(crate) fn space_at(&self, i: usize) -> Space {
        self.root.leaf(&self.levels[..i])
    }
}

impl RegionExpand {
    /// The depth of the box this region names: below every level of its path.
    pub(crate) fn depth(&self) -> usize {
        self.base + self.levels.len()
    }

    /// The space the `i`-th level of the path cuts: the root, through the levels above it.
    pub(crate) fn space_at(&self, i: usize) -> Space {
        self.root.leaf(&self.levels[..i])
    }
}
