//! The [`Walk`]: the runtime odometer that unravels a step into a [`Point`].

use cubecl::prelude::*;

use crate::{Grid, Point};

use super::walk_order::walk_index;
use super::{ComputePrimitive, Coverage, CubeDimension, Distribution, Partitioner, Spread};

/// A [`Partitioner`] instantiated against a [`Grid`]: the runtime half of the
/// walk, so the odometer ([`total`](Walk::total) / [`point`](Walk::point)) lives
/// here.
#[derive(CubeType)]
pub struct Walk {
    grid: Grid,
    steps: usize,
    partitioner: Partitioner,
}

#[cube]
impl Walk {
    /// Instantiate over `grid`; the total step count is the product of its tile
    /// counts.
    pub fn new(grid: Grid, partitioner: Partitioner) -> Walk {
        let frame = grid.frame();
        let mut steps = 1usize;
        #[unroll]
        for p in 0..comptime!(frame.rank()) {
            let axis = comptime!(frame.axis_at(p));
            let dist = partitioner.distribution(axis);
            steps *= axis_count(grid.tiles(axis), dist);
        }
        Walk {
            grid,
            steps,
            partitioner,
        }
    }

    /// Number of steps the walk visits.
    pub fn total(&self) -> usize {
        self.steps
    }

    /// The [`Point`] at walk step `i`. The partitioner maps `i` to an odometer
    /// index ([`walk_index`]); the consumer just iterates `0..total`.
    pub fn point(&self, i: usize) -> Point {
        let idx = walk_index(i, self.steps, self.partitioner.order());
        self.resolve(idx)
    }

    /// Unravel a runtime step `idx` to a [`Point`]: an odometer (last axis
    /// fastest) over the grid's tile counts, each digit mapped to its coordinate.
    fn resolve(&self, idx: usize) -> Point {
        let frame = self.grid.frame();
        // Per-axis tile counts (runtime), in frame order.
        let mut counts = Sequence::<usize>::new();
        #[unroll]
        for p in 0..comptime!(frame.rank()) {
            let axis = comptime!(frame.axis_at(p));
            let dist = self.partitioner.distribution(axis);
            counts.push(axis_count(self.grid.tiles(axis), dist));
        }

        let mut coords = Sequence::<usize>::new();
        #[unroll]
        for p in 0..comptime!(frame.rank()) {
            // weight = product of later axes' counts (last axis fastest).
            let mut weight = 1usize;
            #[unroll]
            for e in comptime!(p + 1)..comptime!(frame.rank()) {
                weight *= *counts.index(e);
            }
            let local = (idx / weight) % *counts.index(p);
            let axis = comptime!(frame.axis_at(p));
            let dist = self.partitioner.distribution(axis);
            coords.push(coord_of(local, self.grid.tiles(axis), dist));
        }
        Point::new(coords, frame)
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

/// This cube's position along the dimension a `Cube` primitive rides. (Plane and
/// Unit spreading land with the inner levels.)
#[cube]
fn hw_pos(#[comptime] unit: ComputePrimitive) -> usize {
    #[comptime]
    let dim = match unit {
        ComputePrimitive::Cube(dim) => dim,
        _ => {
            panic!("hw_pos: only Cube spreading is implemented (Plane/Unit are inner-level seams)")
        }
    };

    let cube_pos = match comptime!(dim) {
        CubeDimension::X => CUBE_POS_X,
        CubeDimension::Y => CUBE_POS_Y,
        CubeDimension::Z => CUBE_POS_Z,
    };

    cube_pos as usize
}
