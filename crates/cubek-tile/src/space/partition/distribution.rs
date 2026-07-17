//! The split vocabulary: how a single axis is distributed, sized, and dealt out.

use crate::{Fold, FoldExpand};
use cubecl::prelude::*;

/// What the plane's lanes each hold of a tile's cells, once a `Unit` split is dealt out.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LaneShare {
    /// The lane's cells are whole, so they read and write as they are.
    Whole,
    /// The lane's cells are a partial: an axis the tile doesn't span is spread across the lanes,
    /// so an accumulator only becomes true once combined across the plane.
    Partial,
}

/// A descent's share, given the parent's and the level's: once partial, always partial.
pub(crate) fn join_lane_share(parent: LaneShare, level: LaneShare) -> LaneShare {
    match (parent, level) {
        (LaneShare::Whole, LaneShare::Whole) => LaneShare::Whole,
        _ => LaneShare::Partial,
    }
}

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
    /// A `Unit` axis's deferred count: resolved to `Instances(plane_size)` at launch
    /// ([`resolve_lanes`](Distribution::resolve_lanes), driven by `Space::launcher`). Every
    /// accessor panics on it; it must never reach geometry or the walk unresolved.
    PlaneLanes,
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

    /// Spread across the plane's lanes, contiguous. The lane count is the hardware
    /// `plane_size`, unknown until launch; [`PlaneLanes`](Coverage::PlaneLanes) is a
    /// deferred count [`resolve_lanes`](Self::resolve_lanes) fills in.
    pub fn unit() -> Self {
        Distribution::Spatial {
            scope: ComputeScope::Unit,
            spread: Spread::Contiguous,
            coverage: Coverage::PlaneLanes,
        }
    }

    /// Resolve a deferred [`PlaneLanes`](Coverage::PlaneLanes) count to
    /// `Instances(plane_size)`; every other distribution passes through. Called once at
    /// launch, so geometry and the walk only ever see a concrete instance count.
    pub(crate) fn resolve_lanes(self, plane_size: usize) -> Self {
        match self {
            Distribution::Spatial {
                scope,
                spread,
                coverage: Coverage::PlaneLanes,
            } => Distribution::Spatial {
                scope,
                spread,
                coverage: Coverage::Instances(plane_size),
            },
            other => other,
        }
    }
}

impl Coverage {
    pub fn instances(self, grid: usize) -> usize {
        match self {
            Coverage::Instances(instances) => instances,
            Coverage::TilesEach(tiles) => grid / tiles,
            Coverage::PlaneLanes => panic!("{UNRESOLVED_LANES}"),
        }
    }

    pub(crate) fn instances_const(self) -> Option<usize> {
        match self {
            Coverage::Instances(n) => Some(n),
            Coverage::TilesEach(_) => None,
            Coverage::PlaneLanes => panic!("{UNRESOLVED_LANES}"),
        }
    }

    pub(crate) fn tiles_const(self) -> Option<usize> {
        match self {
            Coverage::TilesEach(t) => Some(t),
            Coverage::Instances(_) => None,
            Coverage::PlaneLanes => panic!("{UNRESOLVED_LANES}"),
        }
    }
}

/// The panic every [`Coverage::PlaneLanes`] accessor raises: the deferred lane count was
/// never resolved, so the space was not launched through [`Space::launcher`].
const UNRESOLVED_LANES: &str =
    "Coverage::PlaneLanes: unresolved Unit lane count; launch through space.launcher(client)";

/// `TilesEach` pins it, `Instances` splits the `grid` (folded, so a constant grid
/// keeps its constant).
#[cube]
pub(crate) fn tiles_per_instance(grid: usize, #[comptime] cov: Coverage) -> usize {
    match cov {
        Coverage::Instances(instances) => grid.fdiv(instances.runtime()),
        Coverage::TilesEach(tiles) => tiles.runtime(),
        Coverage::PlaneLanes => {
            panic!(
                "Coverage::PlaneLanes: unresolved Unit lane count; launch through space.launcher(client)"
            )
        }
    }
}

/// `Instances` pins it, `TilesEach` derives it from the `grid` (folded, so a constant
/// grid keeps its constant).
#[cube]
pub(crate) fn instance_count(grid: usize, #[comptime] cov: Coverage) -> usize {
    match cov {
        Coverage::Instances(instances) => instances.runtime(),
        Coverage::TilesEach(tiles) => grid.fdiv(tiles.runtime()),
        Coverage::PlaneLanes => {
            panic!(
                "Coverage::PlaneLanes: unresolved Unit lane count; launch through space.launcher(client)"
            )
        }
    }
}

impl Distribution {
    /// `Spatial` with `TilesEach(1)`: the instance owns exactly one tile, so its walk
    /// count is comptime `1` and its coordinate is the hardware position alone.
    pub(crate) fn single_tile(self) -> bool {
        matches!(
            self,
            Distribution::Spatial {
                coverage: Coverage::TilesEach(1),
                ..
            }
        )
    }

    pub(crate) fn coverage(self) -> Coverage {
        match self {
            Distribution::Spatial { coverage, .. } => coverage,
            Distribution::Sequential => panic!("coverage: not a Spatial axis"),
        }
    }

    /// The hardware scope of a `Spatial` axis (panics on `Sequential`); the non-optional
    /// [`scope`](Self::scope) for sites that already know the axis is split.
    pub(crate) fn scope_unchecked(self) -> ComputeScope {
        match self {
            Distribution::Spatial { scope, .. } => scope,
            Distribution::Sequential => panic!("scope_unchecked: not a Spatial axis"),
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
