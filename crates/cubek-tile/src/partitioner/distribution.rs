//! The split vocabulary: how a single axis is distributed, sized, and dealt out.

/// How a single axis is distributed. `Sequential` is one instance walking the
/// whole axis; `Spatial` splits it across hardware instances ([`Coverage`]) dealt
/// out by a [`Spread`].
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

    /// The pinned instance count, if this coverage pins instances (comptime).
    pub(crate) fn instances_const(self) -> Option<usize> {
        match self {
            Coverage::Instances(n) => Some(n),
            Coverage::TilesEach(_) => None,
        }
    }

    /// The pinned per-instance tile count, if this coverage pins tiles (comptime).
    pub(crate) fn tiles_const(self) -> Option<usize> {
        match self {
            Coverage::TilesEach(t) => Some(t),
            Coverage::Instances(_) => None,
        }
    }
}

impl Distribution {
    pub(crate) fn coverage(self) -> Coverage {
        match self {
            Distribution::Spatial { coverage, .. } => coverage,
            Distribution::Sequential => panic!("coverage: not a Spatial axis"),
        }
    }

    pub(crate) fn unit(self) -> ComputePrimitive {
        match self {
            Distribution::Spatial { unit, .. } => unit,
            Distribution::Sequential => panic!("unit: not a Spatial axis"),
        }
    }

    pub(crate) fn spread(self) -> Spread {
        match self {
            Distribution::Spatial { spread, .. } => spread,
            Distribution::Sequential => panic!("spread: not a Spatial axis"),
        }
    }
}
