//! Where a loop is: a [`Region`] is the path of levels a kernel's loops took from a root space
//! to the box the innermost one handed out, one [`Step`] per level. A tile windows itself to a
//! region with `at`, applying the steps below its own depth, so the root tile and any window of
//! it read the same region.

use super::{Level, Partitioning, Space};
use crate::{Axis, Coords, Fold, FoldExpand, Walk};
use cubecl::{prelude::*, unexpanded};

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
    /// Where `level` sits in the nest, outermost `0`: what a storage tile's level is matched
    /// against on the way down ([`Storage`](crate::Storage)).
    #[cube(comptime)]
    pub(crate) depth: usize,
}

#[cube]
impl Step {
    pub(crate) fn new(
        coords: Coords<u32>,
        #[comptime] space: Space,
        #[comptime] level: Level,
        #[comptime] depth: usize,
    ) -> Step {
        Step {
            coords,
            space,
            level,
            depth,
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
/// the partitioning the root sits in, so a tile at any depth on the path applies the steps below
/// it. Iterating a region deals the partitioning's next level: `for plane in cube`.
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
    /// The root space and every level of the partitioning it sits in, the path's own included:
    /// the level below this region is the one at its depth.
    #[cube(comptime)]
    pub(crate) root: Partitioning,
    #[cube(comptime)]
    pub(crate) levels: Vec<Level>,
}

#[cube]
impl Region {
    pub(crate) fn new(
        digits: Sequence<Coords<u32>>,
        sizes: Sequence<usize>,
        #[comptime] base: usize,
        #[comptime] root: Partitioning,
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

    /// The empty path at the partitioning's space: where a loop over it starts from.
    pub(crate) fn root(partitioning: &Partitioning) -> Region {
        Region::rooted(
            &partitioning.space,
            comptime!(partitioning.levels.clone()),
            0usize,
        )
    }

    /// The empty path at `space`, which sits at `depth` in a partitioning of `levels`: where a
    /// loop over a tile's own box starts from.
    pub(crate) fn rooted(
        space: &Space,
        #[comptime] levels: Vec<Level>,
        #[comptime] depth: usize,
    ) -> Region {
        Region::new(
            Sequence::new(),
            space.extents.sizes.clone(),
            depth,
            comptime!(Partitioning::new(space.clone(), levels)),
            comptime!(Vec::new()),
        )
    }

    /// The regions of the level below this one: what `for plane in cube` iterates, as a value
    /// for a schedule that indexes them by hand or a loop that unrolls or reverses.
    pub fn walk(&self) -> Walk {
        Walk::of(&self.child(), comptime!(self.next()), self.clone())
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
            comptime!(Partitioning::new(space.clone(), Vec::new())),
            comptime!(vec![level]),
        )
    }

    /// The `i`-th step of the path.
    pub(crate) fn step(&self, #[comptime] i: usize) -> Step {
        Step::new(
            self.digits.index(i).clone(),
            comptime!(self.space_at(i)),
            comptime!(self.levels[i].clone()),
            comptime!(self.base + i),
        )
    }

    /// The box this region covers, as the runtime space a loop below it walks: the root through
    /// every level, an axis left whole keeping the root's size.
    pub(crate) fn child(&self) -> Space {
        Space::with_sizes(
            comptime!(self.root.space().leaf(&self.levels)),
            self.sizes.clone(),
        )
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
macro_rules! path_shape {
    ($ty:ty) => {
        impl $ty {
            /// The depth of the box this region names: below every level of its path.
            pub(crate) fn depth(&self) -> usize {
                self.base + self.levels.len()
            }

            /// The space the `i`-th level of the path cuts: the root, through the levels above it.
            pub(crate) fn space_at(&self, i: usize) -> Space {
                self.root.space().leaf(&self.levels[..i])
            }

            /// The partitioning's level below this region: the one at its depth.
            pub(crate) fn next(&self) -> Level {
                let depth = self.depth();
                let levels = self.root.levels();
                assert!(
                    depth < levels.len(),
                    "this region sits {depth} levels down a partitioning of {} levels, so there \
                     is no level below it to iterate; a walk of the kernel's own says `over`",
                    levels.len()
                );
                levels[depth].clone()
            }
        }
    };
}
path_shape!(Region);
path_shape!(RegionExpand);

/// The runtime twin of `for plane in cube`, which a kernel's host-side body names but never
/// runs: every loop over a region expands in-kernel.
impl IntoIterator for Region {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

impl IntoIterator for &Region {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

/// `for plane in cube` iterates the level below `cube`. A walk of one region runs straight
/// through, as [`Walk`]'s own iteration does, so a level that cuts nothing costs no loop.
impl Iterable for RegionExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        let walk = self.__expand_walk_method(scope);
        if walk.const_len() == Some(1) {
            walk.expand_unroll(scope, body)
        } else {
            walk.expand(scope, body)
        }
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.__expand_walk_method(scope).expand_unroll(scope, body)
    }
}

/// `for plane in &cube`: the same, leaving `cube` to the body.
impl Iterable for &RegionExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.clone().expand(scope, body)
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.clone().expand_unroll(scope, body)
    }
}
