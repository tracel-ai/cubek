//! The split vocabulary: how a single axis is distributed, sized, and dealt out. What a
//! [`Cut`](crate::Cut) entry of a level states, read back per axis.

use crate::{Fold, FoldExpand};
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

impl Coverage {
    /// How many instances take a `grid` of tiles: the pinned count, or as many runs of
    /// `tiles` as the grid holds, the last one short where it does not divide.
    pub fn instances(self, grid: usize) -> usize {
        match self {
            Coverage::Instances(instances) => instances,
            Coverage::TilesEach(tiles) => grid.div_ceil(tiles),
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

/// The run of tiles each instance is dealt: `TilesEach` pins it, `Instances` splits the `grid`,
/// rounded up so a grid that does not divide leaves its tail in the last runs rather than
/// nowhere (folded, so a constant grid keeps its constant).
#[cube]
pub(crate) fn run_length(grid: usize, #[comptime] cov: Coverage) -> usize {
    match cov {
        Coverage::Instances(instances) => grid
            .fadd(comptime!(instances - 1).runtime())
            .fdiv(instances.runtime()),
        Coverage::TilesEach(tiles) => tiles.runtime(),
    }
}

/// How many instances take the `grid`: `Instances` pins it, `TilesEach` derives it, rounded up
/// (folded, so a constant grid keeps its constant).
#[cube]
pub(crate) fn instance_count(grid: usize, #[comptime] cov: Coverage) -> usize {
    match cov {
        Coverage::Instances(instances) => instances.runtime(),
        Coverage::TilesEach(tiles) => grid
            .fadd(comptime!(tiles - 1).runtime())
            .fdiv(tiles.runtime()),
    }
}

/// The tiles instance `pos` of `instances` takes of a `grid` dealt in runs of `run`: the whole
/// run where the host proved the grid `divides`, else the run cut short where the grid ends
/// inside it (contiguous), or the turns left to it (interleaved). Saturating, so an instance
/// past the grid takes nothing.
#[cube]
pub(crate) fn instance_tiles(
    grid: usize,
    pos: usize,
    instances: usize,
    run: usize,
    #[comptime] spread: Spread,
    #[comptime] divides: bool,
) -> usize {
    if divides {
        run
    } else {
        match spread {
            Spread::Contiguous => {
                let start = pos.fmul(run);
                run.fmin(grid.fmax(start).fsub(start))
            }
            Spread::Interleaved => grid
                .fmax(pos)
                .fsub(pos)
                .fadd(instances.fsub(1usize))
                .fdiv(instances),
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
