//! The [`Walk`]: the (sub-)Spaces partitioning a [`Space`] yields, as a runtime
//! odometer over the per-axis tile counts. Each step is a [`Region`] (a `Space` at
//! an origin); a [`Tile`] locates itself at it.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::{Region, Space, instance_count, tiles_per_instance};

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
    /// To subdivide an operation, merge the operands' spaces then `Walk::over` the result.
    pub fn over(#[comptime] space: Space) -> Walk {
        let mut counts = Sequence::<usize>::new();
        #[unroll]
        for p in 0..space.rank() {
            counts.push(space.count(space.axis_at(p)))
        }

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

    pub fn total(&self) -> usize {
        self.steps
    }

    pub fn region(&self, i: usize) -> Region {
        let idx = walk_index(i, self.steps, comptime!(self.space.partitioner().order()));
        Region::new(self.resolve(idx), self.space.clone())
    }

    /// Unravel a runtime step `idx` to its per-axis coordinates: an odometer over
    /// the per-axis tile counts, last axis fastest.
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
            coords.push(coord_of(local, *self.counts.index(p), dist) as u32);
        }
        coords
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
/// `Interleaved`: instances take turns).
#[cube]
fn coord_of(step: usize, grid: usize, #[comptime] dist: Distribution) -> usize {
    let mut coord = step;
    if comptime!(matches!(dist, Distribution::Spatial { .. })) {
        let cov = comptime!(dist.coverage());
        let unit = comptime!(dist.unit());
        if comptime!(matches!(dist.spread(), Spread::Contiguous)) {
            coord = step + hardware_pos(unit) * tiles_per_instance(grid, cov);
        } else {
            coord = step * instance_count(grid, cov) + hardware_pos(unit);
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
