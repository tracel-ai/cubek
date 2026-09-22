//! Where a loop is: a [`Region`] is the path of levels a kernel's loops took from a root space to
//! the box the innermost one handed out, one [`Step`] per level. A tile windows itself to a region
//! with `at`, applying the steps below its own depth, so the root tile and any window read alike.

use super::{Level, Partitioning, Space};
use crate::{Axis, Coords, Known, KnownExpand, MatrixAxes, Walk};
use cubecl::{prelude::*, unexpanded};

/// The comptime shape of a [`Region`]'s path: the levels taken from a root space down, outermost
/// first, where the first sits in the root's partitioning, and the partitioning itself, whose
/// level at the path's depth is the one below.
#[derive(Clone, Debug)]
pub(crate) struct Path {
    /// The first level's depth in the root's partitioning.
    base: usize,
    /// The root space and every level of the partitioning it sits in, the path's own included.
    root: Partitioning,
    /// The levels taken so far, outermost first.
    levels: Vec<Level>,
}

impl Path {
    pub(crate) fn new(base: usize, root: Partitioning, levels: Vec<Level>) -> Self {
        Path { base, root, levels }
    }

    /// The depth of the box this path names: below every level of it.
    pub(crate) fn depth(&self) -> usize {
        self.base + self.levels.len()
    }

    /// The first level's depth in the root's partitioning.
    pub(crate) fn base(&self) -> usize {
        self.base
    }

    /// How many levels the path has taken.
    pub(crate) fn len(&self) -> usize {
        self.levels.len()
    }

    /// The `i`-th level taken, outermost first.
    pub(crate) fn level(&self, i: usize) -> &Level {
        &self.levels[i]
    }

    /// The space the `i`-th level of the path cuts: the root, through the levels above it.
    pub(crate) fn space_at(&self, i: usize) -> Space {
        self.root.space().leaf(&self.levels[..i])
    }

    /// The box this path names: the root through every level taken.
    pub(crate) fn child(&self) -> Space {
        self.root.space().leaf(&self.levels)
    }

    /// The partitioning's level below this path: the one at its depth.
    pub(crate) fn next(&self) -> Level {
        let depth = self.depth();
        let levels = self.root.levels();
        assert!(
            depth < levels.len(),
            "this region sits {depth} levels down a partitioning of {} levels, so there is no \
             level below it to iterate; a walk of the kernel's own says `over`",
            levels.len()
        );
        levels[depth].clone()
    }

    /// This path one level further down.
    pub(crate) fn below(&self, level: Level) -> Path {
        let mut levels = self.levels.clone();
        levels.push(level);
        Path::new(self.base, self.root.clone(), levels)
    }
}

/// The path a kernel's loops took to a box: the levels from a root space down, outermost first,
/// and the coordinates each loop handed out. A tile at any depth applies the steps below it.
/// Iterating deals the next level.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Region {
    /// One coordinate set per level, outermost first.
    digits: Sequence<Coords<u32>>,
    /// The runtime sizes of the root's dynamic axes, positional: what a loop below this region
    /// walks its child space with.
    sizes: Sequence<usize>,
    #[cube(comptime)]
    pub(crate) path: Path,
}

#[cube]
impl Region {
    pub(crate) fn new(
        digits: Sequence<Coords<u32>>,
        sizes: Sequence<usize>,
        #[comptime] path: Path,
    ) -> Region {
        Region {
            digits,
            sizes,
            path,
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
            space.sizes.clone(),
            comptime!(Path::new(
                depth,
                Partitioning::new(space.clone(), levels),
                Vec::new()
            )),
        )
    }

    /// The regions of the level below this one: what `for plane in cube` iterates, as a value
    /// for a schedule that indexes them by hand or a loop that unrolls or reverses.
    pub fn walk(&self) -> Walk {
        Walk::of(&self.child(), comptime!(self.path.next()), self.clone())
    }

    /// The region one level below the root at trailing-two coordinates `(c0, c1)` under `level`,
    /// `0` elsewhere, for a tile at `depth`: what a leaf states to cut an operand its own way.
    ///
    /// The coordinates carry their own constness ([`retyped`](crate::Known::retyped) keeps a constant
    /// constant): comptime ones fold to constants and can select fragments; kernel-computed ones
    /// (the visit a worker picked out of a grid by hardware position) window memory.
    pub fn trailing(
        #[comptime] depth: usize,
        #[comptime] space: Space,
        #[comptime] level: Level,
        c0: usize,
        c1: usize,
    ) -> Region {
        let rank = comptime!(space.rank());
        let edges = comptime!(MatrixAxes::edges(&space));
        let mut coords = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            // `fcast`, not `as`: a comptime coordinate has to stay a constant or it could no
            // longer select a fragment. `runtime` on the `0` moves the literal into the expand
            // domain and keeps it constant too.
            let c = if comptime!(p == edges.row_split) {
                c0.retyped::<u32>()
            } else if comptime!(p == edges.col_split) {
                c1.retyped::<u32>()
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
            comptime!(Path::new(
                depth,
                Partitioning::new(space.clone(), Vec::new()),
                vec![level]
            )),
        )
    }

    /// The `i`-th step of the path.
    pub(crate) fn step(&self, #[comptime] i: usize) -> Step {
        Step::new(
            self.digits.index(i).clone(),
            comptime!(self.path.space_at(i)),
            comptime!(self.path.level(i).clone()),
            comptime!(self.path.depth() - self.path.len() + i),
        )
    }

    /// The box this region covers, as the runtime space a loop below it walks: the root through
    /// every level, an axis left whole keeping the root's size.
    pub(crate) fn child(&self) -> Space {
        Space::with_sizes(comptime!(self.path.child()), self.sizes.clone())
    }

    /// The innermost level's coordinate along `axis`; `0` when the axis is absent (broadcast by
    /// omission: the tile spans all of it).
    pub fn coord(&self, #[comptime] axis: Axis) -> usize {
        let last = comptime!(self.path.len() - 1);
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
        Region::new(
            digits,
            self.sizes.clone(),
            comptime!(self.path.below(level)),
        )
    }
}

#[cube]
impl Region {
    /// The regions of `level` over this region's own box, a level of the kernel's own rather
    /// than the partitioning's next ([`walk`](Region::walk)).
    pub fn over(&self, #[comptime] level: &Level) -> Walk {
        Walk::of(&self.child(), comptime!(level.clone()), self.clone())
    }
}

/// One level's cut of one space: the tile coordinates a loop over `level` handed out, and the
/// space they index.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Step {
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
                .retyped::<usize>()
        } else {
            0usize.runtime()
        }
    }
}

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
