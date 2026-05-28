//! How a level of the space is split (distribution / coverage / spread), the
//! per-axis walk coordinates they produce, and the [`Partitioner`] that ties them
//! into an ordered walk of the space.

use cubecl::prelude::*;

use crate::std::tile_dsl::{Grid, Point};

use super::{Axis, ByAxis, Space};

/// How a single axis is distributed when partitioned. `Sequential` is one
/// instance walking the whole axis; `Spatial` splits it across hardware
/// instances ([`Coverage`]) dealt out by a [`Spread`]. "Don't split" is just
/// `Sequential` with `sub_tile = extent`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Distribution {
    Sequential,
    Spatial {
        unit: ComputePrimitive,
        spread: Spread,
        coverage: Coverage,
    },
}

/// How a `Spatial` axis is sized across its instances — duals
/// (`instances · tiles_each = grid`); pin one, derive the other.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Coverage {
    /// Pin the instance count; each walks `grid / n` tiles.
    Instances(usize),
    /// Pin each instance's share to `t` tiles; use `grid / t` instances.
    TilesEach(usize),
}

impl Coverage {
    /// Tiles each instance walks, given the axis's full tile grid.
    pub fn tiles_each(self, grid: usize) -> usize {
        match self {
            Coverage::Instances(instances) => grid / instances,
            Coverage::TilesEach(tiles) => tiles,
        }
    }

    /// Instances covering the axis, given its full tile grid.
    pub fn instances(self, grid: usize) -> usize {
        match self {
            Coverage::Instances(instances) => instances,
            Coverage::TilesEach(tiles) => grid / tiles,
        }
    }
}

/// How a `Spatial` axis's tiles are dealt to its instances — disjoint either
/// way, differing only in locality.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Spread {
    /// Instance `i` owns a contiguous run (cube 0 → `{0,1}`, cube 1 → `{2,3}`).
    Contiguous,
    /// Instances take turns (cube 0 → `{0,2}`, cube 1 → `{1,3}`).
    Interleaved,
}

/// A dimension of a hardware grid (for `Cube`, the launch grid): `X`, `Y`, `Z`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CubeDimension {
    X,
    Y,
    Z,
}

/// A hardware primitive an axis can be distributed across, and which of its grid
/// dimensions to ride.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ComputePrimitive {
    Cube(CubeDimension),
    Plane,
    Unit,
}

impl Coverage {
    /// The pinned instance count, if this coverage pins instances (comptime).
    fn instances_const(self) -> Option<usize> {
        match self {
            Coverage::Instances(n) => Some(n),
            Coverage::TilesEach(_) => None,
        }
    }

    /// The pinned per-instance tile count, if this coverage pins tiles (comptime).
    fn tiles_const(self) -> Option<usize> {
        match self {
            Coverage::TilesEach(t) => Some(t),
            Coverage::Instances(_) => None,
        }
    }
}

impl Distribution {
    fn coverage(self) -> Coverage {
        match self {
            Distribution::Spatial { coverage, .. } => coverage,
            Distribution::Sequential => panic!("coverage: not a Spatial axis"),
        }
    }

    fn unit(self) -> ComputePrimitive {
        match self {
            Distribution::Spatial { unit, .. } => unit,
            Distribution::Sequential => panic!("unit: not a Spatial axis"),
        }
    }

    fn spread(self) -> Spread {
        match self {
            Distribution::Spatial { spread, .. } => spread,
            Distribution::Sequential => panic!("spread: not a Spatial axis"),
        }
    }
}

/// The order a [`Partitioner`] visits its `total` walk steps — the blackbox of
/// strategies. A new order is a new variant here (plus a [`walk_index`] arm);
/// nothing downstream branches on which one it is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WalkOrder {
    /// Ascending: step `i` visits odometer index `i`.
    RowMajor,
    /// Descending: step `i` visits `total - i - 1`.
    Reversed,
}

/// A descent strategy for one level of the space: the split
/// ([`sub_tile_edge`](Partitioner::sub_tile_edge) /
/// [`distribution`](Partitioner::distribution)) and the [`WalkOrder`]. A
/// `CubeType` of comptime fields that answers queries as methods — comptime ones
/// (sizes, distribution) via `comptime_type!`, the walk via
/// [`get_walk`](Partitioner::get_walk).
#[derive(CubeType, CubeLaunch, Clone, PartialEq, Eq, Hash, Debug)]
#[expand(derive(Clone))]
pub struct Partitioner {
    #[cube(comptime)]
    sub_tile: ByAxis<usize>,
    #[cube(comptime)]
    dists: ByAxis<Distribution>,
    #[cube(comptime)]
    order: WalkOrder,
}

impl Partitioner {
    /// Declared axis order, last axis fastest — the natural nested walk.
    pub fn row_major(sub_tile: ByAxis<usize>, dists: ByAxis<Distribution>) -> Self {
        Partitioner {
            sub_tile,
            dists,
            order: WalkOrder::RowMajor,
        }
    }

    /// Same split, walked back-to-front.
    pub fn reversed(sub_tile: ByAxis<usize>, dists: ByAxis<Distribution>) -> Self {
        Partitioner {
            sub_tile,
            dists,
            order: WalkOrder::Reversed,
        }
    }

    /// The launch arg for carrying this partitioner on a tile.
    pub fn launch<R: Runtime>(&self) -> PartitionerLaunch<R> {
        PartitionerLaunch::new(self.sub_tile.clone(), self.dists.clone(), self.order)
    }
}

#[cube]
impl Partitioner {
    /// Sub-tile edge along an axis — comptime, since it sizes shared memory.
    pub fn sub_tile_edge(&self, #[comptime] axis: Axis) -> comptime_type!(usize) {
        comptime!(self.sub_tile.get(axis))
    }

    /// How an axis is distributed.
    pub fn distribution(&self, #[comptime] axis: Axis) -> comptime_type!(Distribution) {
        comptime!(self.dists.get(axis))
    }

    /// The order this partitioner visits its steps in.
    pub fn order(&self) -> comptime_type!(WalkOrder) {
        comptime!(self.order)
    }

    /// The [`Walk`] over `grid` (per-axis tile counts, which carry their own
    /// frame). The caller supplies the grid — an operation reads it from its
    /// operands — and the partitioner only owns the order it's walked in.
    pub fn walk(&self, grid: Grid) -> Walk {
        Walk::new(grid, self.clone())
    }
}

/// A [`Partitioner`] instantiated against a [`Grid`]: the runtime half of the
/// walk, so the odometer lives here as methods ([`total`](Walk::total) /
/// [`point`](Walk::point)). The partitioner rides along as comptime config.
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

    /// The [`Point`] at walk step `i`. The partitioner decides which odometer
    /// index step `i` maps to ([`walk_index`]), so the consumer just iterates
    /// `0..total` and the order stays the partitioner's business.
    pub fn point(&self, i: usize) -> Point {
        let idx = walk_index(i, self.steps, self.partitioner.order());
        self.resolve(idx)
    }

    /// Unravel a runtime step `idx` to a [`Point`]: the odometer (last axis
    /// fastest) read off the grid's tile counts, each step mapped to its grid
    /// coordinate. Only the axis structure (order, distribution) is comptime.
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

/// The odometer index visited at walk step `i` of `total` — one arm per
/// [`WalkOrder`]. The caller never learns which order it is; a new order is just
/// a new arm.
#[cube]
fn walk_index(i: usize, total: usize, #[comptime] order: WalkOrder) -> usize {
    match order {
        WalkOrder::RowMajor => i,
        WalkOrder::Reversed => total - i - 1,
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

/// The launch geometry a partitioner implies: cube dimension `d` gets the
/// instance count of whichever axis is `Spatial { Cube(d), .. }`, else 1.
pub fn cube_count_for(partitioner: &Partitioner, space: &Space) -> CubeCount {
    let instances_along = |dim: CubeDimension| -> u32 {
        let mut i = 0;
        while i < space.rank() {
            let axis = space.axis_at(i);
            if let Distribution::Spatial {
                unit: ComputePrimitive::Cube(cube_dim),
                coverage,
                ..
            } = partitioner.distribution(axis)
                && cube_dim == dim
            {
                let grid = space.extent(axis) / partitioner.sub_tile_edge(axis);
                return coverage.instances(grid) as u32;
            }
            i += 1;
        }
        1
    };
    CubeCount::Static(
        instances_along(CubeDimension::X),
        instances_along(CubeDimension::Y),
        instances_along(CubeDimension::Z),
    )
}
