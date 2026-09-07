//! The [`Walk`]: the regions one [`Level`] of a [`Space`] hands the instance running the code,
//! and the verbs a kernel's loop states it with ([`Space::cubes`], [`Space::planes`],
//! [`Space::lanes`], [`Space::walk`], and the same on a [`Region`] or a [`Tile`]).
//!
//! A walk is a sequence with random access and nothing else: it computes once how many regions
//! this cube or plane owns at the level ([`total`](Walk::total)) and how an index maps to one
//! ([`region`](Walk::region)): decode the index as an odometer over the level's walked axes,
//! last declared axis fastest, and add this instance's share on the distributed axes. It never
//! iterates the instances themselves, the hardware does that, and it holds no current region:
//! `for region in walk` is `for i in 0..total { region(i) }`. The only things a walk can be
//! told are its order ([`reversed`](Walk::reversed)) and whether it unrolls
//! ([`unrolled`](Walk::unrolled)). Holding several regions at once (double buffering) is a
//! schedule's doing ([`pipelined`](crate::pipelined)), which indexes the walk by hand.
//!
//! Each region is a [`Region`], the path of levels from the space to that box; a [`Tile`]
//! windows itself to it with `at`, applying the steps below its own depth.

use cubecl::prelude::*;

use crate::{
    Coords, Edge, Fold, FoldExpand, Level, Region, RegionExpand, Space, SpaceExpand,
    instance_count, instance_tiles, run_length,
};

use super::walk_order::walk_index;
use super::{ComputeScope, CubeAxis, Distribution, LevelScope, Spread, WalkOrder};

/// The runtime odometer over a [`Space`]'s tiles under one [`Level`].
#[derive(CubeType)]
pub struct Walk {
    /// Per-axis walk counts: this instance's share on `Spatial` axes, the whole grid on
    /// `Sequential` ones.
    counts: Coords<usize>,
    /// Per-axis hardware-instance coordinate, folded through any shared hardware dim;
    /// `0` for `Sequential`. Loop-invariant, so decoded once at construction rather
    /// than per region.
    positions: Coords<usize>,
    /// Per-axis spread factor combining a step with its position: the instance's tile
    /// share (`Contiguous`) or the instance count (`Interleaved`); `1` for `Sequential`.
    scales: Coords<usize>,
    /// Where in this level's flat step space the walk starts. `0` for a whole walk, so it
    /// folds away; a run dealt out of the flat grid ([`window`](Walk::window)) starts at its own.
    base: usize,
    steps: usize,
    /// The path above this walk: what every region it hands out is one level below.
    parent: Region,
    /// The space the regions are cut from, which is also what a ring sizes its slots to
    /// ([`Ring::smem`](crate::Ring::smem)).
    #[cube(comptime)]
    pub(crate) space: Space,
    /// The level this walk steps: the statement the loop made.
    #[cube(comptime)]
    pub(crate) level: Level,
    /// Whether iterating this walk unrolls (the one codegen choice folding cannot
    /// make): fragment outputs demand it, memory outputs prefer the compact loop.
    #[cube(comptime)]
    pub(crate) unroll: bool,
    /// The order the steps visit the odometer in ([`reversed`](Walk::reversed)).
    #[cube(comptime)]
    order: WalkOrder,
}

/// The loops a kernel writes over a space or a region, one verb per statement. A distribute verb
/// (`cubes`, `planes`, `lanes`) hands each instance of that scope its region and iterates that:
/// once when each instance takes one tile, its share otherwise. `walk` steps every region. The
/// verb checks the level it is handed: `for plane in stage.planes(level)` refuses a level that
/// deals to cubes or steps an axis, so the header cannot say one thing while the value does
/// another. A kernel generic over its levels, which cannot know the verb, says
/// [`level`](Space::level).
#[cube]
impl Space {
    /// The regions of `level` over this space, whatever verb the level is: what a kernel handed
    /// its levels states. Comptime for `Static` axes, runtime for `Dynamic`.
    pub fn level(&self, #[comptime] level: Level) -> Walk {
        Walk::of(self, level, Region::root(self, 0usize))
    }

    /// Each cube's box of this space under `level`, which deals to the cube grid and steps
    /// nothing.
    pub fn cubes(&self, #[comptime] level: Level) -> Walk {
        Walk::stated(
            self,
            level,
            Region::root(self, 0usize),
            comptime!(LevelScope::Cubes),
        )
    }

    /// Each plane's box of this space under `level`, which deals to the cube's planes and steps
    /// nothing.
    pub fn planes(&self, #[comptime] level: Level) -> Walk {
        Walk::stated(
            self,
            level,
            Region::root(self, 0usize),
            comptime!(LevelScope::Planes),
        )
    }

    /// Each lane's box of this space under `level`, which deals to the plane's lanes and steps
    /// nothing.
    pub fn lanes(&self, #[comptime] level: Level) -> Walk {
        Walk::stated(
            self,
            level,
            Region::root(self, 0usize),
            comptime!(LevelScope::Lanes),
        )
    }

    /// Every region of this space under `level`, which deals to nobody: the loop steps them all.
    pub fn walk(&self, #[comptime] level: Level) -> Walk {
        Walk::stated(
            self,
            level,
            Region::root(self, 0usize),
            comptime!(LevelScope::Sequential),
        )
    }
}

#[cube]
impl Region {
    /// [`Space::cubes`] over this region's own box, one level further down the path.
    pub fn cubes(&self, #[comptime] level: Level) -> Walk {
        Walk::stated(
            &self.child(),
            level,
            self.clone(),
            comptime!(LevelScope::Cubes),
        )
    }

    /// [`Space::planes`] over this region's own box, one level further down the path.
    pub fn planes(&self, #[comptime] level: Level) -> Walk {
        Walk::stated(
            &self.child(),
            level,
            self.clone(),
            comptime!(LevelScope::Planes),
        )
    }

    /// [`Space::lanes`] over this region's own box, one level further down the path.
    pub fn lanes(&self, #[comptime] level: Level) -> Walk {
        Walk::stated(
            &self.child(),
            level,
            self.clone(),
            comptime!(LevelScope::Lanes),
        )
    }

    /// [`Space::walk`] over this region's own box, one level further down the path.
    pub fn walk(&self, #[comptime] level: Level) -> Walk {
        Walk::stated(
            &self.child(),
            level,
            self.clone(),
            comptime!(LevelScope::Sequential),
        )
    }

    /// [`Space::level`] over this region's own box, one level further down the path.
    pub fn level(&self, #[comptime] level: Level) -> Walk {
        Walk::of(&self.child(), level, self.clone())
    }
}

#[cube]
impl Walk {
    /// [`of`](Walk::of) under a verb: the level must have been built under the same one.
    pub(crate) fn stated(
        space: &Space,
        #[comptime] level: Level,
        parent: Region,
        #[comptime] verb: LevelScope,
    ) -> Walk {
        comptime!({
            let built = level.scope();
            assert!(
                built == verb,
                "{}: this level was built by `Level::{}`, which is not what this loop says",
                verb.verb(),
                built.verb()
            );
        });
        Walk::of(space, level, parent)
    }

    pub(crate) fn of(space: &Space, #[comptime] level: Level, parent: Region) -> Walk {
        let mut counts = Coords::<usize>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            match comptime!(level.edge_kind(space.axis_at(p))) {
                Edge::Cut(edge) => counts.push(space.extents.count(p, edge)),
                Edge::Whole => counts.push(1usize),
            }
        }
        Walk::from_counts(comptime!(space.clone()), level, counts, parent)
    }

    /// Fold the per-axis grid `grid` into the walk: counts, total steps, and each
    /// `Spatial` axis's hardware decode (invariant across the walk, so paid once here).
    fn from_counts(
        #[comptime] space: Space,
        #[comptime] level: Level,
        grid: Coords<usize>,
        parent: Region,
    ) -> Walk {
        let rank = comptime!(space.rank());
        let mut counts = Coords::<usize>::new();
        let mut positions = Coords::<usize>::new();
        let mut scales = Coords::<usize>::new();

        // Per-axis instance counts, `1` for `Sequential`. Folded, so a constant grid's
        // decode below folds too (`/1` and `%1` vanish, `%` gets a constant divisor).
        let mut instances = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            let dist = comptime!(level.distribution(space.axis_at(p)));
            if comptime!(matches!(dist, Distribution::Spatial { .. })) {
                instances.push(instance_count(grid.at(p), comptime!(dist.coverage())));
            } else {
                instances.push(1usize);
            }
        }

        #[unroll]
        for p in 0..rank {
            let axis = comptime!(space.axis_at(p));
            let dist = comptime!(level.distribution(axis));

            if comptime!(matches!(dist, Distribution::Spatial { .. })) {
                // Mixed-radix stride for axes sharing one hardware dim: the product of the
                // later same-scope axes' instance counts (the earlier axis is the more
                // significant digit); `1` when this axis owns its scope.
                //
                // Both halves of that product, because the odometer is the level's and this
                // space may be a projection of it: the axes here carry a possibly-runtime count,
                // and the ones this space does not span a comptime one. Dropping the second half
                // reads a contracted axis as weight `1` and aliases every outer digit onto a
                // single value; see [`Level::inner_weight_unspanned`].
                let picks = comptime!(
                    ((p + 1)..rank)
                        .filter(|&q| level.distribution(space.axis_at(q)).scope() == dist.scope())
                        .collect::<Vec<_>>()
                );
                let unspanned = comptime!(level.inner_weight_unspanned(&space, axis));
                let inner_weight = instances.fproduct(picks) * comptime!(unspanned).runtime();
                let position = hardware_pos(comptime!(dist.scope_unchecked()))
                    .fdiv(inner_weight)
                    .frem(instances.at(p));
                // This instance's run of the grid, cut short where the grid does not divide.
                let run = run_length(grid.at(p), comptime!(dist.coverage()));
                counts.push(instance_tiles(
                    grid.at(p),
                    position,
                    instances.at(p),
                    run,
                    comptime!(dist.spread()),
                    comptime!(level.divides(&space, axis)),
                ));
                positions.push(position);
                if comptime!(matches!(dist.spread(), Spread::Contiguous)) {
                    scales.push(run);
                } else {
                    scales.push(instances.at(p));
                }
            } else {
                counts.push(grid.at(p));
                positions.push(0usize);
                scales.push(1usize);
            }
        }

        // Folded, not accumulated: a static walk's total stays a constant, so
        // `#[unroll] for region in walk` can unroll it.
        let steps = counts.fproduct(comptime!((0..rank).collect::<Vec<_>>()));

        Walk {
            counts,
            positions,
            scales,
            base: 0usize,
            steps,
            parent,
            order: comptime!(WalkOrder::RowMajor),
            space,
            level,
            unroll: comptime!(false),
        }
    }

    /// This walk with its steps visited last to first.
    pub fn reversed(self) -> Walk {
        Walk {
            counts: self.counts,
            positions: self.positions,
            scales: self.scales,
            base: self.base,
            steps: self.steps,
            parent: self.parent,
            space: comptime!(self.space.clone()),
            level: comptime!(self.level.clone()),
            unroll: comptime!(self.unroll),
            order: comptime!(WalkOrder::Reversed),
        }
    }

    /// This walk, unrolled when iterated: each region's coordinates fold to comptime
    /// constants (static spaces only; the trip count must be constant).
    pub fn unrolled(self) -> Walk {
        self.with_unroll(comptime!(true))
    }

    /// This walk, unrolled when `unroll`. Lets a caller pick the mode from a comptime flag
    /// without branching on the [`Walk`] value (which `#[cube]` would read as a runtime select).
    pub(crate) fn with_unroll(self, #[comptime] unroll: bool) -> Walk {
        Walk {
            counts: self.counts,
            positions: self.positions,
            scales: self.scales,
            base: self.base,
            steps: self.steps,
            parent: self.parent,
            space: comptime!(self.space.clone()),
            level: comptime!(self.level.clone()),
            unroll: comptime!(unroll),
            order: comptime!(self.order),
        }
    }

    /// This walk over the `steps` regions starting at flat step `base`, rather than all of its
    /// own from zero.
    ///
    /// The window is how a level deals its grid out as contiguous runs instead of as a
    /// rectangular block per axis: every axis stays `Sequential`, so the counts are the whole
    /// grid and the flat index already carries every coordinate, and an instance's share is a
    /// range of that index. `base` and `steps` are runtime values, so a run whose length only
    /// the launch knows walks the same loop a static one does.
    ///
    /// The caller owns the range: `base + steps` past this walk's own [`total`](Walk::total)
    /// reads coordinates that are not in the grid, and nothing here can check it.
    pub fn window(self, base: usize, steps: usize) -> Walk {
        Walk {
            counts: self.counts,
            positions: self.positions,
            scales: self.scales,
            base,
            steps,
            parent: self.parent,
            space: comptime!(self.space.clone()),
            level: comptime!(self.level.clone()),
            unroll: comptime!(self.unroll),
            order: comptime!(self.order),
        }
    }

    /// Returns the regions count
    pub fn total(&self) -> usize {
        self.steps
    }

    /// Returns the ith region of the walk
    pub fn region(&self, i: usize) -> Region {
        let idx = self
            .base
            .fadd(walk_index(i, self.steps, comptime!(self.order)));
        self.parent
            .below(self.resolve(idx), comptime!(self.level.clone()))
    }

    /// Unravel a runtime step `idx` to its per-axis coordinates: each axis's odometer
    /// [`digit`](Walk::digit), [`fold`](Walk::fold)ed with its instance position. A
    /// constant `idx` (an unrolled walk's) folds through, so a static walk's regions
    /// carry comptime coordinates and can select fragments.
    fn resolve(&self, idx: usize) -> Coords<u32> {
        let mut coords = Coords::<u32>::new();

        #[unroll]
        for p in 0..comptime!(self.space.rank()) {
            coords.push(self.fold(self.digit(idx, p), p).fcast::<u32>());
        }
        coords
    }

    /// The odometer digit of step `idx` along axis `p` (last axis fastest): divide off
    /// the later axes' counts, keep the remainder of this one. Constant counts fold.
    fn digit(&self, idx: usize, #[comptime] p: usize) -> usize {
        let rank = comptime!(self.space.rank());
        let quot = idx.fdiv(
            self.counts
                .fproduct(comptime!(((p + 1)..rank).collect::<Vec<_>>())),
        );
        // `% count` is a no-op when `idx` has no more significant digit: a range fact,
        // which folding (which only sees values) cannot know. A count of one is the exception:
        // there `% 1` folds to the constant `0`, which `quot` alone would not.
        let count = self.counts.at(p);
        let one = count.constant();
        if comptime!(
            one != Some(1)
                && (0..p).all(|e| self.level.single_tile(&self.space, self.space.axis_at(e)))
        ) {
            quot
        } else {
            quot.frem(count)
        }
    }

    /// Fold axis `p`'s instance position into its `digit` per the [`Spread`]: an
    /// instance owns a contiguous run (`digit + pos·share`) or the instances take
    /// turns (`digit·instances + pos`); a sequential digit passes through.
    fn fold(&self, digit: usize, #[comptime] p: usize) -> usize {
        let dist = comptime!(self.level.distribution(self.space.axis_at(p)));
        if comptime!(matches!(dist, Distribution::Sequential)) {
            digit
        } else if comptime!(matches!(dist.spread(), Spread::Contiguous)) {
            digit.fadd(self.positions.at(p).fmul(self.scales.at(p)))
        } else {
            digit.fmul(self.scales.at(p)).fadd(self.positions.at(p))
        }
    }
}

/// Iterating a `Walk` visits its regions in order, so `for region in walk` is equivalent to
/// `for i in 0..walk.total() {let region = walk.region(i); ...}`
/// Schedules that need random access (prefetch, double-buffering) still index by hand.
impl IntoIterator for Walk {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        let mut regions = Vec::new();
        for i in 0..self.total() {
            regions.push(self.region(i));
        }
        regions.into_iter()
    }
}

impl Iterable for WalkExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        let start = 0usize.into_expand(scope);
        let total = self.__expand_total_method(scope);
        let range = RangeExpand::new(start, total);
        if self.unroll {
            range.expand_unroll(scope, &mut |scope, i| {
                body(scope, self.__expand_region_method(scope, i));
            });
        } else {
            range.expand(scope, &mut |scope, i| {
                body(scope, self.__expand_region_method(scope, i));
            });
        }
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        let start = 0usize.into_expand(scope);
        let total = self.__expand_total_method(scope);
        RangeExpand::new(start, total).expand_unroll(scope, &mut |scope, i| {
            body(scope, self.__expand_region_method(scope, i));
        });
    }

    /// The region count when it folds to a constant, which is what lets `for region in walk`
    /// drop the loop around a single region: a level that cuts nothing is one region, walked
    /// straight through rather than under a one-trip loop.
    fn const_len(&self) -> Option<usize> {
        crate::fold::constant(&self.steps).map(|n| n as usize)
    }
}

/// The raw hardware position of a `Spatial` axis's scope; [`Walk::from_counts`] folds
/// it through the axis's shared-dim stride to the per-axis instance coordinate.
#[cube]
pub fn hardware_pos(#[comptime] unit: ComputeScope) -> usize {
    match comptime!(unit) {
        ComputeScope::Cube(dim) => {
            let cube_pos = match comptime!(dim) {
                CubeAxis::X => CUBE_POS_X,
                CubeAxis::Y => CUBE_POS_Y,
                CubeAxis::Z => CUBE_POS_Z,
            };
            cube_pos as usize
        }
        // cube_dim = new_2d(plane_size, num_planes): Y is the plane index, X the
        // plane-relative lane. Lanes agree on UNIT_POS_Y, so they cooperate.
        ComputeScope::Plane => UNIT_POS_Y as usize,
        // The plane-relative lane, not flat UNIT_POS: flat would fold in UNIT_POS_Y and
        // double-count a sibling Plane axis's digit. The plane's `plane_size` lanes ride
        // the X dim already, so a Unit axis divides them (instances == plane_size).
        ComputeScope::Unit => UNIT_POS_X as usize,
    }
}

impl Walk {
    /// The depth of the regions this walk hands out: one below its path.
    pub(crate) fn depth(&self) -> usize {
        self.parent.depth() + 1
    }
}

impl WalkExpand {
    pub(crate) fn depth(&self) -> usize {
        self.parent.depth() + 1
    }
}
