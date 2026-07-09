//! The [`Walk`]: the (sub-)Spaces partitioning a [`Space`] yields, as a runtime
//! odometer over the per-axis tile counts. Each step is a [`Region`] (a `Space` at
//! an origin); a [`Tile`] locates itself at it.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::{Axis, Region, RegionExpand, Space, StaticRegion, instance_count, tiles_per_instance};

use super::walk_order::walk_index;
use super::{ComputeScope, CubeAxis, Distribution, Spread};

/// The runtime odometer over a [`Space`]'s tiles.
#[derive(CubeType)]
pub struct Walk {
    /// Per-axis walk counts: this instance's share on `Spatial` axes, the whole grid on
    /// `Sequential` ones.
    counts: Sequence<usize>,
    /// Per-axis hardware-instance coordinate, folded through any shared hardware dim;
    /// `0` for `Sequential`. Loop-invariant, so decoded once at construction rather
    /// than per region.
    positions: Sequence<usize>,
    /// Per-axis spread factor combining a step with its position: the instance's tile
    /// share (`Contiguous`) or the instance count (`Interleaved`); `1` for `Sequential`.
    scales: Sequence<usize>,
    steps: usize,
    #[cube(comptime)]
    space: Space,
}

#[cube]
impl Walk {
    /// The [`Walk`] over `space`'s tiles
    /// Comptime for `Static` axes, runtime for `Dynamic`.
    pub fn over(space: Space) -> Walk {
        let mut counts = Sequence::<usize>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            let edge = comptime!(space.partitioner().edge(space.axis_at(p)));
            counts.push(space.extents.count(p, edge));
        }
        Walk::from_counts(comptime!(space.clone()), counts)
    }

    /// Fold the per-axis grid `grid` into the walk: counts, total steps, and each
    /// `Spatial` axis's hardware decode (invariant across the walk, so paid once here).
    fn from_counts(#[comptime] space: Space, grid: Sequence<usize>) -> Walk {
        let rank = comptime!(space.rank());
        let mut steps = 1usize;
        let mut counts = Sequence::<usize>::new();
        let mut positions = Sequence::<usize>::new();
        let mut scales = Sequence::<usize>::new();

        #[unroll]
        for p in 0..rank {
            let axis = comptime!(space.axis_at(p));
            let dist = comptime!(space.partitioner().distribution(axis));
            let count = axis_count(*grid.index(p), dist);
            steps *= count;
            counts.push(count);

            if comptime!(matches!(dist, Distribution::Spatial { .. })) {
                // Mixed-radix stride for axes sharing one hardware dim: the product of the
                // later same-scope axes' instance counts (the earlier axis is the more
                // significant digit). Computed from the runtime grid counts, so dynamic
                // extents work; `1` when this axis owns its scope.
                let mut inner_weight = 1usize;
                #[unroll]
                for q in comptime!(p + 1)..rank {
                    let other = comptime!(space.axis_at(q));
                    let other_dist = comptime!(space.partitioner().distribution(other));
                    if comptime!(other_dist.scope() == dist.scope()) {
                        inner_weight *=
                            instance_count(*grid.index(q), comptime!(other_dist.coverage()));
                    }
                }
                let instances = instance_count(*grid.index(p), comptime!(dist.coverage()));
                positions.push((hardware_pos(comptime!(dist.unit())) / inner_weight) % instances);
                if comptime!(matches!(dist.spread(), Spread::Contiguous)) {
                    scales.push(tiles_per_instance(*grid.index(p), comptime!(dist.coverage())));
                } else {
                    scales.push(instances);
                }
            } else {
                positions.push(0usize);
                scales.push(1usize);
            }
        }

        Walk {
            counts,
            positions,
            scales,
            steps,
            space,
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

    /// Unravel a runtime step `idx` to its per-axis coordinates. The hardware part is
    /// precomputed at construction; a digit's div/mod also drops when the axes after
    /// (resp. before) it are all comptime single-tile, leaving a pure-`TilesEach(1)`
    /// spatial grid with a lone sequential axis decode-free.
    fn resolve(&self, idx: usize) -> CoordsDyn {
        let rank = comptime!(self.space.rank());
        let mut coords = CoordsDyn::new();

        #[unroll]
        for p in 0..rank {
            let dist = comptime!(
                self.space
                    .partitioner()
                    .distribution(self.space.axis_at(p))
            );
            let pos = *self.positions.index(p);

            if comptime!(dist.single_tile()) {
                // One tile per instance: the coordinate is the hardware position alone.
                coords.push(pos as u32);
            } else {
                // weight = product of later axes' counts (last axis fastest); single-tile
                // axes count comptime 1, so only the rest multiply — and when none remain
                // the division vanishes. Likewise `% count` is a no-op when every earlier
                // axis is single-tile (`idx` then has no more significant digit).
                let quot = if comptime!(((p + 1)..rank).all(|e| self.space.single_tile_at(e))) {
                    idx
                } else {
                    let mut weight = 1usize;
                    #[unroll]
                    for e in comptime!(p + 1)..rank {
                        let other = comptime!(self.space.axis_at(e));
                        let other_dist = comptime!(self.space.partitioner().distribution(other));
                        if comptime!(!other_dist.single_tile()) {
                            weight *= *self.counts.index(e);
                        }
                    }
                    idx / weight
                };
                let local = if comptime!((0..p).all(|e| self.space.single_tile_at(e))) {
                    quot
                } else {
                    quot % *self.counts.index(p)
                };
                let coord = if comptime!(matches!(dist, Distribution::Sequential)) {
                    local
                } else if comptime!(matches!(dist.spread(), Spread::Contiguous)) {
                    local + pos * *self.scales.index(p)
                } else {
                    local * *self.scales.index(p) + pos
                };
                coords.push(coord as u32);
            }
        }
        coords
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
        RangeExpand::new(start, total).expand(scope, |scope, i| {
            body(scope, self.__expand_region_method(scope, i));
        });
    }

    fn expand_unroll(self, scope: &Scope, mut body: impl FnMut(&Scope, RegionExpand)) {
        let start = 0usize.into_expand(scope);
        let total = self.__expand_total_method(scope);
        RangeExpand::new(start, total).expand_unroll(scope, |scope, i| {
            body(scope, self.__expand_region_method(scope, i));
        });
    }
}

/// [`Walk`]'s static sibling, for the register tier: fragments are comptime-indexed, so
/// this odometer is host data and its loop unrolls where the runtime walk loops.
/// `Static` axes only, no hardware folding; the level it walks is instance-owned wholesale.
pub struct StaticWalk {
    counts: Vec<usize>,
    space: Space,
}

impl StaticWalk {
    pub fn over(space: &Space) -> StaticWalk {
        let counts = space.axes().map(|axis| space.count(axis)).collect();
        StaticWalk {
            counts,
            space: space.clone(),
        }
    }

    /// The walk over `space` with `fastest` walked innermost, so each operand fragment
    /// feeds a consecutive burst of executes (the legacy emission order, ~1.3% on Metal).
    pub fn over_fastest(space: &Space, fastest: Axis) -> StaticWalk {
        let mut axes: Vec<Axis> = space.axes().filter(|&a| a != fastest).collect();
        axes.push(fastest);
        StaticWalk::over(&space.project(&axes))
    }

    pub fn total(&self) -> usize {
        self.counts.iter().product()
    }

    /// The `i`th region, row-major (last axis fastest); the register walk's steps are
    /// independent MMAs, so no [`WalkOrder`] plugs in.
    pub fn region(&self, i: usize) -> StaticRegion {
        let rank = self.space.rank();
        let mut coords = vec![0; rank];
        let mut rem = i;
        for p in (0..rank).rev() {
            coords[p] = rem % self.counts[p];
            rem /= self.counts[p];
        }
        StaticRegion::new(coords, self.space.clone())
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
        // cube_dim = new_2d(plane_size, num_planes): Y is the plane index, the flat
        // position the unit index. Lanes agree on UNIT_POS_Y, so they cooperate.
        ComputeScope::Plane => UNIT_POS_Y as usize,
        ComputeScope::Unit => UNIT_POS as usize,
    }
}
