//! Where a loop is: the path of levels a kernel's loops took from a root space to a box.

use super::{ComputeScope, Level, Partitioning, Space};
use crate::{Axis, Coords, Integer, IntegerExpand, MatrixAxes, Walk};
use cubecl::{prelude::*, unexpanded};

/// The comptime shape of a [`Region`]'s path: the levels taken from a root space, outermost first.
#[derive(Clone, Debug)]
pub(crate) struct Path {
    base: usize,
    root: Partitioning,
    levels: Vec<Level>,
}

impl Path {
    pub(crate) fn new(base: usize, root: Partitioning, levels: Vec<Level>) -> Self {
        Path { base, root, levels }
    }

    /// The depth of the box this path names.
    pub(crate) fn depth(&self) -> usize {
        self.base + self.levels.len()
    }

    pub(crate) fn base(&self) -> usize {
        self.base
    }

    pub(crate) fn len(&self) -> usize {
        self.levels.len()
    }

    pub(crate) fn level(&self, i: usize) -> &Level {
        &self.levels[i]
    }

    /// The space the `i`-th level of the path cuts.
    pub(crate) fn space_at(&self, i: usize) -> Space {
        self.root.space().leaf(&self.levels[..i])
    }

    /// The box this path names.
    pub(crate) fn child(&self) -> Space {
        self.root.space().leaf(&self.levels)
    }

    /// The partitioning's level at this path's depth.
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

    /// Who runs the box this path names: the narrowest scope any level from the root down to it
    /// hands its tiles to — a unit, a plane, or, where every level is walked or distributed over
    /// cubes, the cube.
    pub(crate) fn scope(&self) -> ComputeScope {
        self.root.levels()[..self.base]
            .iter()
            .chain(&self.levels)
            .filter_map(|level| level.coverage().scope())
            .min()
            .unwrap_or(ComputeScope::Cube)
    }

    /// The root partitioning's levels, outermost first: what a tile placed on this path is walked
    /// with.
    pub(crate) fn root_levels(&self) -> Vec<Level> {
        self.root.levels().to_vec()
    }

    /// The planes one cube of the root's partitioning holds ([`Partitioning::planes_per_cube`]).
    pub(crate) fn planes_per_cube(&self) -> usize {
        self.root.planes_per_cube() as usize
    }

    /// This path one level further down.
    pub(crate) fn below(&self, level: Level) -> Path {
        let mut levels = self.levels.clone();
        levels.push(level);
        Path::new(self.base, self.root.clone(), levels)
    }
}

/// The path a kernel's loops took to a box, with the coordinates each loop handed out.
/// Iterating distributes the next level.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Region {
    digits: Sequence<Coords<u32>>,
    /// The runtime sizes of the root's dynamic axes, positional.
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

    /// The empty path at the partitioning's space.
    pub(crate) fn root(partitioning: &Partitioning) -> Region {
        Region::rooted(
            &partitioning.space,
            comptime!(partitioning.levels.clone()),
            0usize,
        )
    }

    /// The empty path at `space`, which sits at `depth` in a partitioning of `levels`.
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

    /// The regions of the level below this one.
    pub fn walk(&self) -> Walk {
        Walk::of(&self.child(), comptime!(self.path.next()), self.clone())
    }

    /// The region one level below the root at trailing-two coordinates `(c0, c1)` under `level`,
    /// `0` elsewhere, for a tile at `depth`. Comptime coordinates stay constant.
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
            // `cast`, not `as`, and `runtime` on the `0`: both keep a comptime coordinate constant.
            let c = if comptime!(p == edges.row_split) {
                c0.cast::<u32>()
            } else if comptime!(p == edges.col_split) {
                c1.cast::<u32>()
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

    /// The box this region covers, as a runtime space.
    pub(crate) fn child(&self) -> Space {
        Space::with_sizes(comptime!(self.path.child()), self.sizes.clone())
    }

    /// The innermost level's coordinate along `axis`; `0` when the axis is absent.
    pub fn coord(&self, #[comptime] axis: Axis) -> usize {
        let last = comptime!(self.path.len() - 1);
        self.step(last).coord(axis)
    }

    /// The first coordinate along `axis` of the box this region covers, counted from its root's:
    /// every level's coordinate along `axis` times the tile it cuts there. What a kernel bounds a
    /// walk of its own by, as an attention's causal rows bound the keys they read.
    pub fn origin(&self, #[comptime] axis: Axis) -> usize {
        let mut origin = 0usize;
        #[unroll]
        for i in 0..comptime!(self.path.len()) {
            if let Some(tile) = comptime!(self.path.level(i).tile(axis)) {
                origin += self.step(i).coord(axis) * tile;
            }
        }
        origin
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
    /// The regions of `level` over this region's own box.
    pub fn over(&self, #[comptime] level: &Level) -> Walk {
        Walk::of(&self.child(), comptime!(level.clone()), self.clone())
    }
}

/// One level's cut of one space: the tile coordinates a loop handed out, and the space.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Step {
    coords: Coords<u32>,
    #[cube(comptime)]
    pub(crate) space: Space,
    #[cube(comptime)]
    pub(crate) level: Level,
    /// Where `level` sits in the nest, outermost `0`.
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

    /// The coordinate along `axis`; `0` when the axis is absent.
    pub(crate) fn coord(&self, #[comptime] axis: Axis) -> usize {
        if comptime!(self.space.contains(axis)) {
            self.coords
                .at(comptime!(self.space.position(axis)))
                .cast::<usize>()
        } else {
            0usize.runtime()
        }
    }
}

/// Host-side stand-in for `for plane in cube`; never runs.
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

/// `for plane in cube` iterates the level below `cube`.
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
