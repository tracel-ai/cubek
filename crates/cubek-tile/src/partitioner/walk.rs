//! The [`Walk`]: the (sub-)Spaces partitioning a [`Space`] yields, as a runtime
//! odometer over the per-axis tile counts. Each step is a [`Region`] (a `Space` at
//! an origin); a [`Tile`] locates itself at it.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::{Region, RegionExpand, Space, instance_count, tiles_per_instance};

use super::walk_order::walk_index;
use super::{ComputeScope, CubeAxis, Distribution, Spread};

/// The runtime odometer over a [`Space`]'s tiles.
#[derive(CubeType)]
pub struct Walk {
    counts: Sequence<usize>,
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

    /// Total step count from the per-axis grid `counts`, shared by both constructors.
    fn from_counts(#[comptime] space: Space, counts: Sequence<usize>) -> Walk {
        let mut steps = 1usize;
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            let axis = space.axis_at(p);
            let dist = space.partitioner().distribution(axis);
            steps *= axis_count(*counts.index(p), dist);
        }

        Walk {
            counts,
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

    /// Unravel a runtime step `idx` to its per-axis coordinates
    fn resolve(&self, idx: usize) -> CoordsDyn {
        let rank = comptime!(self.space.rank());
        let mut counts = Sequence::<usize>::new();

        #[unroll]
        for p in 0..rank {
            let axis = comptime!(self.space.axis_at(p));
            let dist = comptime!(self.space.partitioner().distribution(axis));
            counts.push(axis_count(*self.counts.index(p), dist));
        }

        let mut coords = CoordsDyn::new();
        #[unroll]
        for p in 0..rank {
            // weight = product of later axes' counts (last axis fastest).
            let mut weight = 1usize;
            #[unroll]
            for e in comptime!(p + 1)..comptime!(self.space.rank()) {
                weight *= *counts.index(e);
            }
            let local = (idx / weight) % *counts.index(p);
            let axis = comptime!(self.space.axis_at(p));
            let dist = comptime!(self.space.partitioner().distribution(axis));
            // Mixed-radix stride for axes sharing one hardware dim: the product of the later
            // same-scope axes' instance counts (the earlier axis is the more significant
            // digit). Computed from the runtime grid counts, so dynamic extents work; `1` when
            // this axis owns its scope or is sequential.
            let mut inner_weight = 1usize;
            #[unroll]
            for q in comptime!(p + 1)..rank {
                let other = comptime!(self.space.axis_at(q));
                let other_dist = comptime!(self.space.partitioner().distribution(other));
                if comptime!(dist.scope().is_some() && other_dist.scope() == dist.scope()) {
                    inner_weight *=
                        instance_count(*self.counts.index(q), comptime!(other_dist.coverage()));
                }
            }
            coords.push(coord_of(local, *self.counts.index(p), inner_weight, dist) as u32);
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

/// Whole `grid` when `Sequential`, else this instance's `Spatial` share.
#[cube]
fn axis_count(grid: usize, #[comptime] dist: Distribution) -> usize {
    if comptime!(matches!(dist, Distribution::Spatial { .. })) {
        tiles_per_instance(grid, dist.coverage())
    } else {
        grid
    }
}

/// Grid coordinate for a runtime local `step`: `step` for `Sequential`, else the
/// `Spatial` axis folds its hardware instance in (`Contiguous`: instance owns a run;
/// `Interleaved`: instances take turns). `inner_weight` is this axis's stride in a
/// hardware dim it may share with others: the raw hardware position is decoded to this
/// axis's own instance via `(pos / inner_weight) % instances`. With one axis on the dim
/// `inner_weight = 1` and the position is in range, so the decode is a no-op.
#[cube]
fn coord_of(
    step: usize,
    grid: usize,
    inner_weight: usize,
    #[comptime] dist: Distribution,
) -> usize {
    let mut coord = step;
    if comptime!(matches!(dist, Distribution::Spatial { .. })) {
        let cov = comptime!(dist.coverage());
        let unit = comptime!(dist.unit());
        let instances = instance_count(grid, cov);
        let pos = (hardware_pos(unit) / inner_weight) % instances;
        if comptime!(matches!(dist.spread(), Spread::Contiguous)) {
            coord = step + pos * tiles_per_instance(grid, cov);
        } else {
            coord = step * instances + pos;
        }
    }
    coord
}

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
        ComputeScope::Plane => UNIT_POS as usize,
        ComputeScope::Unit => {
            panic!("hardware_pos: Unit spreading is an inner-level seam, not yet implemented")
        }
    }
}
