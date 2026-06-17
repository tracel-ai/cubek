//! The split vocabulary: how a single axis is distributed, sized, and dealt out.

use cubecl::prelude::*;

/// `Sequential` is one instance walking the whole axis. `Spatial` splits it across
/// hardware instances ([`Coverage`]) dealt out by a [`Spread`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Distribution {
    Sequential,
    Spatial {
        scope: ComputeScope,
        spread: Spread,
        coverage: Coverage,
    },
}

/// How a `Spatial` axis is sized across its instances, where
/// `instances · tiles_per_instance = grid`. Pin one, derive the other.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Coverage {
    /// Pin the instance count; each walks `grid / n` tiles.
    Instances(usize),
    /// Pin each instance's share to `t` tiles; use `grid / t` instances.
    TilesEach(usize),
}

/// How a `Spatial` axis's tiles are dealt to its instances. Disjoint either way,
/// differing only in locality.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Spread {
    /// Instance `i` owns a contiguous run (cube 0 → `{0,1}`, cube 1 → `{2,3}`).
    Contiguous,
    /// Instances take turns (cube 0 → `{0,2}`, cube 1 → `{1,3}`).
    Interleaved,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CubeAxis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ComputeScope {
    Cube(CubeAxis),
    Plane,
    Unit,
}

impl Distribution {
    /// One tile per cube on `axis`, contiguous
    pub fn cube(axis: CubeAxis) -> Self {
        Distribution::Spatial {
            scope: ComputeScope::Cube(axis),
            spread: Spread::Contiguous,
            coverage: Coverage::TilesEach(1),
        }
    }

    /// One tile per plane, contiguous
    pub fn plane() -> Self {
        Distribution::Spatial {
            scope: ComputeScope::Plane,
            spread: Spread::Contiguous,
            coverage: Coverage::TilesEach(1),
        }
    }
}

impl Coverage {
    pub fn instances(self, grid: usize) -> usize {
        match self {
            Coverage::Instances(instances) => instances,
            Coverage::TilesEach(tiles) => grid / tiles,
        }
    }

    pub(crate) fn instances_const(self) -> Option<usize> {
        match self {
            Coverage::Instances(n) => Some(n),
            Coverage::TilesEach(_) => None,
        }
    }

    pub(crate) fn tiles_const(self) -> Option<usize> {
        match self {
            Coverage::TilesEach(t) => Some(t),
            Coverage::Instances(_) => None,
        }
    }
}

/// `TilesEach` pins it, `Instances` splits the `grid`.
#[cube]
pub(crate) fn tiles_per_instance(grid: usize, #[comptime] cov: Coverage) -> usize {
    match cov {
        Coverage::Instances(instances) => grid / instances,
        Coverage::TilesEach(tiles) => tiles.runtime(),
    }
}

/// `Instances` pins it, `TilesEach` derives it from the `grid`.
#[cube]
pub(crate) fn instance_count(grid: usize, #[comptime] cov: Coverage) -> usize {
    match cov {
        Coverage::Instances(instances) => instances.runtime(),
        Coverage::TilesEach(tiles) => grid / tiles,
    }
}

impl Distribution {
    pub(crate) fn coverage(self) -> Coverage {
        match self {
            Distribution::Spatial { coverage, .. } => coverage,
            Distribution::Sequential => panic!("coverage: not a Spatial axis"),
        }
    }

    pub(crate) fn unit(self) -> ComputeScope {
        match self {
            Distribution::Spatial { scope: unit, .. } => unit,
            Distribution::Sequential => panic!("unit: not a Spatial axis"),
        }
    }

    pub(crate) fn scope(self) -> Option<ComputeScope> {
        match self {
            Distribution::Spatial { scope, .. } => Some(scope),
            Distribution::Sequential => None,
        }
    }

    pub(crate) fn spread(self) -> Spread {
        match self {
            Distribution::Spatial { spread, .. } => spread,
            Distribution::Sequential => panic!("spread: not a Spatial axis"),
        }
    }
}
