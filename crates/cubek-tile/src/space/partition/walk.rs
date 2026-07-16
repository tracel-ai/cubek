//! The [`Walk`]: the (sub-)Spaces partitioning a [`Space`] yields, as a runtime
//! odometer over the per-axis tile counts. Each step is a [`Region`] (a `Space` at
//! an origin); a [`Tile`] locates itself at it.

use cubecl::prelude::*;

use crate::{
    Axis, Coords, Fold, FoldExpand, Region, RegionExpand, Space, instance_count, tiles_per_instance,
};

use super::walk_order::walk_index;
use super::{ComputeScope, CubeAxis, Distribution, Spread};

/// The runtime odometer over a [`Space`]'s tiles.
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
    steps: usize,
    #[cube(comptime)]
    space: Space,
    /// Whether iterating this walk unrolls (the one codegen choice folding cannot
    /// make): fragment outputs demand it, memory outputs prefer the compact loop.
    #[cube(comptime)]
    unroll: bool,
}

#[cube]
impl Walk {
    /// The [`Walk`] over `space`'s tiles
    /// Comptime for `Static` axes, runtime for `Dynamic`.
    pub fn over(space: Space) -> Walk {
        let mut counts = Coords::<usize>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            let edge = comptime!(space.partitioner().edge(space.axis_at(p)));
            counts.push(space.extents.count(p, edge));
        }
        Walk::from_counts(comptime!(space.clone()), counts)
    }

    /// [`over`](Walk::over) with `fastest` walked innermost, so each operand fragment
    /// feeds a consecutive burst of executes (the legacy emission order, ~1.3% on
    /// Metal). Static spaces only: there are no runtime sizes to permute.
    pub fn over_fastest(#[comptime] space: Space, #[comptime] fastest: Axis) -> Walk {
        let reordered = comptime!({
            assert!(space.is_static(), "Walk::over_fastest: static spaces only");
            space.with_fastest(fastest)
        });
        Walk::over(Space::with_sizes(reordered, Sequence::new()))
    }

    /// Fold the per-axis grid `grid` into the walk: counts, total steps, and each
    /// `Spatial` axis's hardware decode (invariant across the walk, so paid once here).
    fn from_counts(#[comptime] space: Space, grid: Coords<usize>) -> Walk {
        let rank = comptime!(space.rank());
        let mut counts = Coords::<usize>::new();
        let mut positions = Coords::<usize>::new();
        let mut scales = Coords::<usize>::new();

        // Per-axis instance counts, `1` for `Sequential`. Folded, so a constant grid's
        // decode below folds too (`/1` and `%1` vanish, `%` gets a constant divisor).
        let mut instances = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            let dist = comptime!(space.partitioner().distribution(space.axis_at(p)));
            if comptime!(matches!(dist, Distribution::Spatial { .. })) {
                instances.push(instance_count(grid.at(p), comptime!(dist.coverage())));
            } else {
                instances.push(1usize);
            }
        }

        #[unroll]
        for p in 0..rank {
            let axis = comptime!(space.axis_at(p));
            let dist = comptime!(space.partitioner().distribution(axis));
            let count = axis_count(grid.at(p), dist);
            counts.push(count);

            if comptime!(matches!(dist, Distribution::Spatial { .. })) {
                // Mixed-radix stride for axes sharing one hardware dim: the product of the
                // later same-scope axes' instance counts (the earlier axis is the more
                // significant digit); `1` when this axis owns its scope.
                let picks = comptime!(
                    ((p + 1)..rank)
                        .filter(|&q| {
                            space.partitioner().distribution(space.axis_at(q)).scope()
                                == dist.scope()
                        })
                        .collect::<Vec<_>>()
                );
                let inner_weight = instances.fproduct(picks);
                positions.push(
                    hardware_pos(comptime!(dist.scope_unchecked()))
                        .fdiv(inner_weight)
                        .frem(instances.at(p)),
                );
                if comptime!(matches!(dist.spread(), Spread::Contiguous)) {
                    scales.push(tiles_per_instance(grid.at(p), comptime!(dist.coverage())));
                } else {
                    scales.push(instances.at(p));
                }
            } else {
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
            steps,
            space,
            unroll: comptime!(false),
        }
    }

    /// This walk, unrolled when iterated: each region's coordinates fold to comptime
    /// constants (static spaces only; the trip count must be constant).
    pub fn unrolled(self) -> Walk {
        Walk {
            counts: self.counts,
            positions: self.positions,
            scales: self.scales,
            steps: self.steps,
            space: comptime!(self.space.clone()),
            unroll: comptime!(true),
        }
    }

    /// Returns the regions count
    pub fn total(&self) -> usize {
        self.steps
    }

    /// Returns the ith region of the walk
    pub fn region(&self, i: usize) -> Region {
        let idx = walk_index(i, self.steps, comptime!(self.space.partitioner().order()));
        Region::new(self.resolve(idx), self.space.clone())
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
        // which folding (which only sees values) cannot know.
        if comptime!((0..p).all(|e| self.space.single_tile_at(e))) {
            quot
        } else {
            quot.frem(self.counts.at(p))
        }
    }

    /// Fold axis `p`'s instance position into its `digit` per the [`Spread`]: an
    /// instance owns a contiguous run (`digit + pos·share`) or the instances take
    /// turns (`digit·instances + pos`); a sequential digit passes through.
    fn fold(&self, digit: usize, #[comptime] p: usize) -> usize {
        let dist = comptime!(self.space.partitioner().distribution(self.space.axis_at(p)));
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

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, RegionExpand)) {
        let start = 0usize.into_expand(scope);
        let total = self.__expand_total_method(scope);
        let range = RangeExpand::new(start, total);
        if self.unroll {
            range.expand_unroll(scope, |scope, i| {
                body(scope, self.__expand_region_method(scope, i));
            });
        } else {
            range.expand(scope, |scope, i| {
                body(scope, self.__expand_region_method(scope, i));
            });
        }
    }

    fn expand_unroll(self, scope: &Scope, mut body: impl FnMut(&Scope, RegionExpand)) {
        let start = 0usize.into_expand(scope);
        let total = self.__expand_total_method(scope);
        RangeExpand::new(start, total).expand_unroll(scope, |scope, i| {
            body(scope, self.__expand_region_method(scope, i));
        });
    }
}

/// Whole `grid` when `Sequential`, else this instance's `Spatial` share.
#[cube]
fn axis_count(grid: usize, #[comptime] dist: Distribution) -> usize {
    if comptime!(matches!(dist, Distribution::Spatial { .. })) {
        tiles_per_instance(grid, dist.coverage())
    } else {
        grid
    }
}

/// The raw hardware position of a `Spatial` axis's scope; [`Walk::from_counts`] folds
/// it through the axis's shared-dim stride to the per-axis instance coordinate.
#[cube]
fn hardware_pos(#[comptime] unit: ComputeScope) -> usize {
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
