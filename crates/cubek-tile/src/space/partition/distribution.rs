//! The split vocabulary: how a single axis is distributed, sized, and dealt out.

use crate::{Fold, FoldExpand};
use cubecl::prelude::*;

/// What the plane's lanes each hold of a tile's cells, once a `Unit` split is dealt out.
///
/// A `Unit` axis the tile doesn't span is *folded*: the lanes cover disjoint slices of it, so
/// each holds a partial. One the tile does span is *carried*: it gives each lane a different
/// cell. Which of the three cases below a tile is in is what says how a partial drains.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LaneShare {
    /// Nothing folded: the lane's cells are whole, so they read and write as they are.
    Whole,
    /// Nothing carried either, so every lane of the plane holds a partial of the *same* cell and
    /// the plane's own reduction is the drain.
    Plane,
    /// Both, so the plane splits into groups: one cell each, several cells in flight at once.
    /// `fold_mask` is the set of lane-index bits the folded axes occupy, so a cell's partials
    /// live on exactly the lanes that agree outside it and differ inside.
    Group { fold_mask: usize },
}

/// A descent's share, given the parent's and the level's: the folds compose, since each level
/// takes its own bits of the lane index.
///
/// [`LaneShare::Plane`] already spans every lane, so nothing can fold under it, and nothing
/// builds that, since [`Space::cube_dim`](crate::Space::cube_dim) caps the tree's `Unit` instance
/// product at the plane width.
pub(crate) fn join_lane_share(parent: LaneShare, level: LaneShare) -> LaneShare {
    match (parent, level) {
        (LaneShare::Whole, share) | (share, LaneShare::Whole) => share,
        (LaneShare::Group { fold_mask: a }, LaneShare::Group { fold_mask: b }) => {
            LaneShare::Group { fold_mask: a | b }
        }
        _ => panic!("join_lane_share: {parent:?} under {level:?}: nothing folds under a plane"),
    }
}

/// How many of the plane's lanes run one tile's work.
///
/// A space that distributes nothing at `Unit` scope still launches a full plane
/// ([`Space::cube_dim`](crate::Space::cube_dim) sizes it at the hardware `plane_size`), and every
/// lane of it then runs the same code over the same cells. Identical stores land the same value
/// however many lanes make them, so only a write that accumulates has to count the writers.
///
/// A fold is not idempotent. `Repeated` lanes folding one cell add their contribution
/// `plane_size` times, so a folding drain elects one of them. Distinct from [`LaneShare`], which
/// says what the lanes hold of a cell rather than how many of them hold it: with nothing on the
/// lanes both answers are "whole", and only one of them is the one a fold needs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LaneWork {
    /// Something rides the lanes, so each has its own share and a cell is written once.
    Own,
    /// Nothing does, so every lane repeats the same work and a cell is written once per lane.
    Repeated,
}

/// What the plane's lanes are to a tile's cells: what each of them holds of one
/// ([`LaneShare`]), and how many of them hold it ([`LaneWork`]).
///
/// Two answers to one question. They are derived from the same space at the same moment and read
/// together on drain, where neither settles who writes on its own, so they travel together.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Lanes {
    pub share: LaneShare,
    pub work: LaneWork,
}

/// What one instance holds of a tile's cells, across the scopes whose instances can only meet in
/// the destination: `Plane` and `Cube`.
///
/// [`LaneShare`]'s counterpart, and deliberately a coarser answer, because the two combine in
/// different places. A plane's lanes share registers, so they combine there, and to elect one of
/// their own to write they have to know which lanes hold a cell: hence a mask. Planes and cubes
/// share no registers. Each folds its own contribution into the destination and never learns that
/// the others exist, so there is nothing to elect between them and no mask to read.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SplitShare {
    /// Every cell this instance writes is its own outright, so the drain is a store.
    Whole,
    /// Several instances hold partials of the same cell, so the drain has to fold rather than
    /// store. A contraction cut at plane or cube scope is the way to get here.
    Partial,
}

impl SplitShare {
    /// Refuse an accumulation this share leaves in pieces, unless the destination adds them
    /// together. Called where an accumulator is opened and where one is written, which are the two
    /// places a partial can escape.
    ///
    /// A destination that replaces is wrong twice over under a split, and silently: a
    /// register-resident accumulator drains by storing, so the last instance to arrive erases
    /// every other one's slice, and one accumulating in place reads the cell, folds, and writes
    /// it back, which is a lost update. Neither shows up as anything but a wrong number, so it is
    /// refused here instead. A destination that accumulates ([`Write::Accumulate`](crate::Write))
    /// is the case this exists to let through, and it serves both scopes alike: the drain's
    /// election is per plane (`UNIT_POS_X == 0`), so one lane of every plane of every cube adds
    /// its own.
    pub(crate) fn validate(self, write: crate::Write, site: &str) {
        match (self, write) {
            (SplitShare::Whole, _) | (SplitShare::Partial, crate::Write::Accumulate) => {}
            (SplitShare::Partial, crate::Write::Replace) => panic!(
                "{site}: this accumulator's cells are split across planes or cubes and its \
                 destination replaces rather than accumulates, so every partial but one would be \
                 lost. \
                 A contracted axis cut at plane or cube scope (`Cut::plane`, `Cut::cube`) gives \
                 each instance a slice of the contraction, and none of them holds a whole cell. \
                 Drain into an accumulating destination (bind it as an `AccumulateArg`), cut the \
                 contraction \
                 at unit scope (`Cut::unit`, combined in the plane's registers), or give the \
                 output an axis of its own for the split."
            ),
        }
    }
}

/// A spatial distribution under construction: who runs the tiles, how many of them, and which
/// ones each takes.
///
/// The value [`cubes`], [`planes`] and [`lanes`] build. It names no axis, which is what lets one
/// value describe a single axis's tiles ([`Cut::cube`](crate::Cut::cube) and its siblings) or
/// several axes' work at once ([`LevelCuts::distribute`](crate::LevelCuts::distribute)).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Write;

    /// A destination that replaces cannot take a cell several instances hold slices of. The
    /// guard is the only thing between that mistake and a wrong number: nothing about it shows up
    /// at compile time or in a crash.
    #[test]
    #[should_panic(expected = "split across planes or cubes")]
    fn a_partial_cell_may_not_be_stored() {
        SplitShare::Partial.validate(Write::Replace, "test");
    }

    /// A destination that accumulates is exactly the case it exists to let through.
    #[test]
    fn a_partial_cell_may_be_accumulated() {
        SplitShare::Partial.validate(Write::Accumulate, "test");
    }

    /// A whole cell is nobody else's business, whichever way it is written.
    #[test]
    fn a_whole_cell_is_written_either_way() {
        SplitShare::Whole.validate(Write::Replace, "test");
        SplitShare::Whole.validate(Write::Accumulate, "test");
    }
}
