//! How a level of the space is split (distribution / coverage / spread), the
//! per-axis walk coordinates they produce, and the [`Partitioner`] that ties them
//! into an ordered walk of the space.

use cubecl::prelude::*;

use super::{Axis, ByAxis, MAX_AXES, Space};

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
    Instances(u32),
    /// Pin each instance's share to `t` tiles; use `grid / t` instances.
    TilesEach(u32),
}

impl Coverage {
    /// Tiles each instance walks, given the axis's full tile grid.
    pub fn tiles_each(self, grid: u32) -> u32 {
        match self {
            Coverage::Instances(instances) => grid / instances,
            Coverage::TilesEach(tiles) => tiles,
        }
    }

    /// Instances covering the axis, given its full tile grid.
    pub fn instances(self, grid: u32) -> u32 {
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

/// Where a tile sits along one axis at a walk [`Cell`], distribution folded in.
/// `Fixed` is a grid index chosen at walk time; `Affine` is
/// `hw_index · stride + offset` (how a `Spatial` axis turns its instance into a
/// grid coordinate). Resolved to a runtime index by `axis_pos`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Coord {
    Fixed(u32),
    Affine {
        unit: ComputePrimitive,
        stride: u32,
        offset: u32,
    },
}

impl Coord {
    /// The hardware-independent part: index for `Fixed`, offset for `Affine`.
    pub fn base(self) -> u32 {
        match self {
            Coord::Fixed(index) => index,
            Coord::Affine { offset, .. } => offset,
        }
    }

    pub fn is_affine(self) -> bool {
        matches!(self, Coord::Affine { .. })
    }

    pub fn unit(self) -> ComputePrimitive {
        match self {
            Coord::Affine { unit, .. } => unit,
            Coord::Fixed(_) => panic!("Coord::unit: not an Affine coordinate"),
        }
    }

    pub fn stride(self) -> u32 {
        match self {
            Coord::Affine { stride, .. } => stride,
            Coord::Fixed(_) => panic!("Coord::stride: not an Affine coordinate"),
        }
    }
}

/// One cell of a [`Partitioner::walk`]: each axis's [`Coord`]. `resolve` turns it
/// into a runtime [`Point`].
pub type Cell = ByAxis<Coord>;

/// A resolved grid coordinate: one index per axis of its frame (the operation
/// [`Space`] the walk ranges over). Tiles read it [by axis](Point::get), so they
/// need only their own axes.
#[derive(CubeType)]
pub struct Point {
    coords: Sequence<u32>,
    #[cube(comptime)]
    frame: Space,
}

#[cube]
impl Point {
    /// Wrap per-axis runtime coordinates (in `frame` order) as a point.
    pub fn new(coords: Sequence<u32>, #[comptime] frame: Space) -> Point {
        Point { coords, frame }
    }

    /// The coordinate along `axis`, located via the point's frame.
    pub fn get(&self, #[comptime] axis: Axis) -> u32 {
        *self.coords.index(comptime!(self.frame.position(axis)))
    }
}

/// A named strategy for descending one level of the space: the split
/// ([`sub_tile_edge`](Partitioner::sub_tile_edge) /
/// [`distribution`](Partitioner::distribution)) and the ordered
/// [`walk`](Partitioner::walk).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Partitioner {
    /// Declared axis order, last axis fastest — the natural nested walk.
    RowMajor {
        sub_tile: ByAxis<u32>,
        dists: ByAxis<Distribution>,
    },
    /// Same split, walked back-to-front.
    Reversed {
        sub_tile: ByAxis<u32>,
        dists: ByAxis<Distribution>,
    },
}

impl Partitioner {
    /// Sub-tile edge along an axis.
    pub fn sub_tile_edge(&self, axis: Axis) -> u32 {
        match self {
            Partitioner::RowMajor { sub_tile, .. } | Partitioner::Reversed { sub_tile, .. } => {
                sub_tile.get(axis)
            }
        }
    }

    /// How an axis is distributed.
    pub fn distribution(&self, axis: Axis) -> Distribution {
        match self {
            Partitioner::RowMajor { dists, .. } | Partitioner::Reversed { dists, .. } => {
                dists.get(axis)
            }
        }
    }

    /// Steps the current instance takes along an axis (its local share).
    fn count(&self, space: &Space, axis: Axis) -> u32 {
        let grid = space.extent(axis) / self.sub_tile_edge(axis);
        match self.distribution(axis) {
            Distribution::Sequential => grid,
            Distribution::Spatial { coverage, .. } => coverage.tiles_each(grid),
        }
    }

    /// The [`Coord`] one axis takes at local `step`.
    fn locate(&self, space: &Space, axis: Axis, step: u32) -> Coord {
        match self.distribution(axis) {
            Distribution::Sequential => Coord::Fixed(step),
            Distribution::Spatial {
                unit,
                spread,
                coverage,
            } => {
                let grid = space.extent(axis) / self.sub_tile_edge(axis);
                match spread {
                    Spread::Contiguous => Coord::Affine {
                        unit,
                        stride: coverage.tiles_each(grid),
                        offset: step,
                    },
                    Spread::Interleaved => Coord::Affine {
                        unit,
                        stride: 1,
                        offset: step * coverage.instances(grid),
                    },
                }
            }
        }
    }

    /// The ordered [`Cell`]s this level visits, as an odometer over the space's
    /// axes (last axis fastest).
    pub fn walk(&self, space: &Space) -> Vec<Cell> {
        let n = space.rank();

        let mut counts = [1u32; MAX_AXES];
        let mut total = 1u32;
        let mut p = 0;
        while p < n {
            counts[p] = self.count(space, space.axis_at(p));
            total *= counts[p];
            p += 1;
        }

        let mut cells = Vec::with_capacity(total as usize);
        let mut index = [0u32; MAX_AXES];
        let mut visited = 0u32;
        while visited < total {
            let mut entries = [(Axis(0), Coord::Fixed(0)); MAX_AXES];
            let mut q = 0;
            while q < n {
                let axis = space.axis_at(q);
                entries[q] = (axis, self.locate(space, axis, index[q]));
                q += 1;
            }
            cells.push(ByAxis::new(&entries[..n]));

            // Increment the odometer: last axis fastest.
            let mut d = n;
            while d > 0 {
                d -= 1;
                index[d] += 1;
                if index[d] < counts[d] {
                    break;
                }
                index[d] = 0;
            }
            visited += 1;
        }

        if matches!(self, Partitioner::Reversed { .. }) {
            cells.reverse();
        }
        cells
    }
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
                return coverage.instances(grid);
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
