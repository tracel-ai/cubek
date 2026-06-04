//! The [`Walk`]: what partitioning a [`Space`] yields — the (sub-)Spaces to
//! visit, as a runtime odometer over the per-axis tile counts. Each step is a
//! [`Region`] (a `Space` at an origin); a [`Tile`] locates itself at it.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::{Region, Space};

use super::walk_order::walk_index;
use super::{ComputeScope, Coverage, CubeAxis, Distribution, Spread};

/// The runtime odometer over a [`Space`]'s tiles: its `space` (the partitioned
/// space, which owns the partitioner) plus the per-axis tile `counts`. The
/// odometer ([`total`](Walk::total) / [`region`](Walk::region)) lives here.
#[derive(CubeType)]
pub struct Walk {
    /// Per-axis tile count, in `space` order — the space measured in tiles.
    counts: Sequence<usize>,
    steps: usize,
    #[cube(comptime)]
    space: Space,
}

#[cube]
impl Walk {
    /// The walk over `space`'s regions: one tile [`count`](Space::count) per axis
    /// (`extent / edge`, read straight off the space — no operand, no view), folded
    /// by distribution into the step total. This is the runtime *positions* half of
    /// a subdivision (the comptime *shape* half is [`Space::divide`]); to subdivide
    /// an operation, merge the operands' spaces, then `Walk::over` the result.
    pub fn over(#[comptime] space: Space) -> Walk {
        let mut counts = Sequence::<usize>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            counts.push(usize::from_int(comptime!(
                space.count(space.axis_at(p)) as i64
            )));
        }

        let mut steps = 1usize;
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            let axis = comptime!(space.axis_at(p));
            let dist = comptime!(space.partitioner().distribution(axis));
            steps *= axis_count(*counts.index(p), dist);
        }
        Walk {
            counts,
            steps,
            space,
        }
    }

    /// Number of steps the walk visits.
    pub fn total(&self) -> usize {
        self.steps
    }

    /// The [`Region`] at walk step `i`: the (sub-)Space this step visits — the
    /// partitioned space at an origin. The partitioner maps `i` to an odometer
    /// index ([`walk_index`]); the consumer just iterates `0..total` and locates
    /// each tile at the region ([`Tile::at`](crate::Tile::at)).
    pub fn region(&self, i: usize) -> Region {
        let idx = walk_index(i, self.steps, comptime!(self.space.partitioner().order()));
        Region::new(self.resolve(idx), comptime!(self.space.clone()))
    }

    /// Unravel a runtime step `idx` to its per-axis coordinates: an odometer (last
    /// axis fastest) over the per-axis tile counts, each digit mapped to its
    /// coordinate.
    fn resolve(&self, idx: usize) -> CoordsDyn {
        // Per-instance counts (after distribution), in space order.
        let mut counts = Sequence::<usize>::new();
        #[unroll]
        for p in 0..comptime!(self.space.rank()) {
            let axis = comptime!(self.space.axis_at(p));
            let dist = comptime!(self.space.partitioner().distribution(axis));
            counts.push(axis_count(*self.counts.index(p), dist));
        }

        let mut coords = CoordsDyn::new();
        #[unroll]
        for p in 0..comptime!(self.space.rank()) {
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

/// Tiles this instance walks along an axis with `grid` tiles total: the whole
/// grid when `Sequential`, else its `Spatial` share.
#[cube]
fn axis_count(grid: usize, #[comptime] dist: Distribution) -> usize {
    let mut count = grid;
    if comptime!(matches!(dist, Distribution::Spatial { .. })) {
        count = tiles_each_rt(grid, comptime!(dist.coverage()));
    }
    count
}

/// Grid coordinate for a runtime local `step`: `step` for `Sequential`, else the
/// `Spatial` axis folds its hardware instance in (`Contiguous`: instance owns a
/// run; `Interleaved`: instances take turns).
#[cube]
fn coord_of(step: usize, grid: usize, #[comptime] dist: Distribution) -> usize {
    let mut coord = step;
    if comptime!(matches!(dist, Distribution::Spatial { .. })) {
        let cov = comptime!(dist.coverage());
        let unit = comptime!(dist.unit());
        if comptime!(matches!(dist.spread(), Spread::Contiguous)) {
            coord = step + hw_pos(unit) * tiles_each_rt(grid, cov);
        } else {
            coord = step * instances_rt(grid, cov) + hw_pos(unit);
        }
    }
    coord
}

/// Tiles each instance covers, given the axis's runtime tile `grid`. `TilesEach`
/// pins it; `Instances` splits the grid.
#[cube]
fn tiles_each_rt(grid: usize, #[comptime] cov: Coverage) -> usize {
    let mut out = usize::from_int(comptime!(cov.tiles_const().unwrap_or(0) as i64));
    if comptime!(cov.instances_const().is_some()) {
        out = grid / comptime!(cov.instances_const().unwrap());
    }
    out
}

/// Instances covering the axis, given its runtime tile `grid`. `Instances` pins
/// it; `TilesEach` derives it from the grid.
#[cube]
fn instances_rt(grid: usize, #[comptime] cov: Coverage) -> usize {
    let mut out = usize::from_int(comptime!(cov.instances_const().unwrap_or(0) as i64));
    if comptime!(cov.tiles_const().is_some()) {
        out = grid / comptime!(cov.tiles_const().unwrap());
    }
    out
}

/// This instance's position along the hardware primitive an axis rides: the cube
/// position for a `Cube` dimension, the plane index for `Plane`. On CPU a plane
/// is one core (plane length 1), so a `Plane`-spread axis is split across cores;
/// cubes are sequential launch-grid iterations. (`Unit` is an inner-level seam.)
#[cube]
fn hw_pos(#[comptime] unit: ComputeScope) -> usize {
    match comptime!(unit) {
        ComputeScope::Cube(dim) => {
            let cube_pos = match comptime!(dim) {
                CubeAxis::X => CUBE_POS_X,
                CubeAxis::Y => CUBE_POS_Y,
                CubeAxis::Z => CUBE_POS_Z,
            };
            cube_pos as usize
        }
        // On CPU a plane is one unit (plane length 1), so the unit position *is*
        // the plane/core index — and `cube_dim_for` lays the cores out as the
        // cube's units. (`PLANE_POS` is the GPU spelling, but the CPU backend
        // doesn't pass it.)
        ComputeScope::Plane => UNIT_POS as usize,
        ComputeScope::Unit => {
            panic!("hw_pos: Unit spreading is an inner-level seam, not yet implemented")
        }
    }
}
