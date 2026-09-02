//! The split vocabulary: how a single axis is distributed, sized, and dealt out.

use crate::{Fold, FoldExpand};
use cubecl::prelude::*;

/// A spatial distribution under construction: who runs the tiles, how many of them, and which
/// ones each takes.
///
/// The value [`cubes`], [`planes`] and [`lanes`] build. It names no axis, which is what lets one
/// value describe a single axis's tiles or several axes' work at once, whichever the level it is
/// handed to names ([`LevelCuts::distribute`](crate::LevelCuts::distribute)).
///
/// The knobs live here rather than on [`Distribution`] so they cannot be reached from
/// [`Sequential`](Distribution::Sequential): one instance walking the whole axis has nobody to
/// share with and nothing to take turns with.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Spatial {
    scope: ComputeScope,
    spread: Spread,
    coverage: Coverage,
}

/// The tiles ride the cubes of `axis`, one each.
pub fn cubes(axis: CubeAxis) -> Spatial {
    Spatial {
        scope: ComputeScope::Cube(axis),
        spread: Spread::Contiguous,
        coverage: Coverage::TilesEach(1),
    }
}

/// The tiles ride the cube's planes, one each.
pub fn planes() -> Spatial {
    Spatial {
        scope: ComputeScope::Plane,
        spread: Spread::Contiguous,
        coverage: Coverage::TilesEach(1),
    }
}

/// The tiles ride the plane's lanes. How many lanes is the hardware's `plane_size`, unknown until
/// launch, so the count is deferred ([`Coverage::PlaneLanes`]) and stamped by
/// [`Space::launcher`](crate::Space::launcher). State [`instances`](Spatial::instances) to take a
/// subset, which is what carving one plane between several axes needs.
pub fn lanes() -> Spatial {
    Spatial {
        scope: ComputeScope::Unit,
        spread: Spread::Contiguous,
        coverage: Coverage::PlaneLanes,
    }
}

impl Spatial {
    /// Instances take turns rather than each taking a contiguous run, so neighbouring instances
    /// touch neighbouring tiles. What a read wants whenever those tiles are neighbouring words.
    pub fn interleaved(mut self) -> Self {
        self.spread = Spread::Interleaved;
        self
    }

    /// Pin the instance count; each walks `grid / n` tiles. Replaces whatever count stood:
    /// `instances · tiles_each = grid`, so stating either states the other.
    pub fn instances(mut self, n: usize) -> Self {
        self.coverage = Coverage::Instances(n);
        self
    }

    /// Pin each instance's share; `grid / t` instances run. The twin of
    /// [`instances`](Spatial::instances), and the same field.
    pub fn tiles_each(mut self, t: usize) -> Self {
        self.coverage = Coverage::TilesEach(t);
        self
    }

    pub fn scope(self) -> ComputeScope {
        self.scope
    }

    pub fn coverage(self) -> Coverage {
        self.coverage
    }

    pub fn spread(self) -> Spread {
        self.spread
    }

    /// How this distribution hands `axes` axes' regions out.
    ///
    /// Not a knob: it follows from what was stated. One axis's runs are boxes of the grid, and so
    /// is one region each whatever the axes, so both get a dial per axis. Only several axes
    /// sharing a stated count need an index no box can describe.
    pub(crate) fn handout(self, axes: usize) -> Handout {
        match (axes, self.coverage) {
            (0 | 1, _) | (_, Coverage::TilesEach(1)) => Handout::Dial,
            _ => Handout::OneIndex,
        }
    }
}

/// How a level hands the axes it distributes to their scope's workers.
///
/// [`Spatial::handout`]'s answer, read once where a level is stated
/// ([`LevelCuts::distribute`](crate::LevelCuts::distribute)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Handout {
    /// A dial on each axis: its tiles ride the scope on their own, and several dials make a box
    /// of the grid.
    Dial,
    /// One index over every named axis, whose runs the workers take a share of each
    /// ([`Work`](crate::Work)).
    OneIndex,
}

impl From<Spatial> for Distribution {
    fn from(spatial: Spatial) -> Distribution {
        Distribution::Spatial {
            scope: spatial.scope,
            spread: spatial.spread,
            coverage: spatial.coverage,
        }
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
    /// One tile per cube on `axis`, contiguous: [`cubes`] with nothing else stated.
    pub fn cube(axis: CubeAxis) -> Self {
        cubes(axis).into()
    }

    /// One tile per plane, contiguous: [`planes`] with nothing else stated.
    pub fn plane() -> Self {
        planes().into()
    }

    /// Spread across the plane's lanes: [`lanes`] with nothing else stated.
    pub fn unit() -> Self {
        lanes().into()
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
