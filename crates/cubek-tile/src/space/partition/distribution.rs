//! The split vocabulary: how a single axis is dealt out. What an entry of a
//! [`Level`](crate::Level) states beside its tile and its [`Count`](crate::Count), read back
//! per axis.

use crate::{Fold, FoldExpand};
use cubecl::prelude::*;

/// `Sequential` is one instance walking the whole axis. `Spatial` deals it to hardware
/// instances by a [`Spread`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Distribution {
    Sequential,
    Spatial { scope: ComputeScope, spread: Spread },
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
