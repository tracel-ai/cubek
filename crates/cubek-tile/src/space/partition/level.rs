//! One decomposition [`Level`]: how each axis of a space is cut and dealt out, and the axes dealt
//! as one. The value a kernel's loop states ([`Space::level`](crate::Space::level)), the value
//! a [`Region`](crate::Region) carries down to `at`, and the value a launch sizes its grid from
//! ([`Nest`](crate::Nest)). A blueprint hands the same value to both, one method per level, so
//! the grid and the loops cannot disagree.
//!
//! Built by [`Level::new`]: two verbs on a [`LevelCuts`] collector,
//! [`distribute`](LevelCuts::distribute) for axes a hardware scope's workers take and
//! [`walk`](LevelCuts::walk) for axes every one of them steps through. Between them they name
//! each of the space's axes exactly once.

use super::{ComputeScope, Coverage, Distribution, Handout, Spatial, Spread};
use crate::{Axis, ByAxis, Extent, LaneShare, Space, SplitShare};

/// One decomposition level of a space: every axis's sub-tile edge and distribution, in the
/// space's canonical axis order, plus the axes it distributes as one.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Level {
    edges: ByAxis<usize>,
    dists: ByAxis<Distribution>,
    scope: LevelScope,
    work: Option<Work>,
}

impl Level {
    /// The level `f` states over `axes`, which fix the canonical order (the last one is the
    /// fastest of a walk). The cuts may come in any order and are realigned to it; every axis
    /// must be named exactly once, by `walk` or `distribute`.
    pub fn new(axes: &[Axis], f: impl FnOnce(&mut LevelCuts)) -> Level {
        let mut cuts = LevelCuts::new();
        f(&mut cuts);
        // Per axis first: it names the one that is wrong, where the count only says the total
        // is off.
        for &axis in axes {
            let stated = cuts.cuts.iter().filter(|&&(a, _)| a == axis).count();
            assert!(stated > 0, "Level::new: axis {axis:?} has no cut");
            assert!(
                stated == 1,
                "Level::new: axis {axis:?} is cut {stated} times; a level states each of its \
                 axes once, by `walk` or `distribute`"
            );
        }
        assert_eq!(
            cuts.cuts.len(),
            axes.len(),
            "Level::new: {} cuts but {} axes",
            cuts.cuts.len(),
            axes.len()
        );
        let cut = |axis: Axis| {
            cuts.cuts
                .iter()
                .find(|&&(a, _)| a == axis)
                .expect("checked above")
                .1
        };
        let edges: Vec<_> = axes.iter().map(|&a| (a, cut(a).edge)).collect();
        let dists: Vec<_> = axes.iter().map(|&a| (a, cut(a).dist)).collect();
        Level::from_parts(ByAxis::new(&edges), ByAxis::new(&dists), cuts.work)
    }

    pub(crate) fn from_parts(
        edges: ByAxis<usize>,
        dists: ByAxis<Distribution>,
        work: Option<Work>,
    ) -> Level {
        // The finest scope any axis rides; `Sequential` when none spreads at all.
        let scope = dists.values().fold(LevelScope::Sequential, |scope, dist| {
            scope.max(LevelScope::of(dist))
        });
        Level {
            edges,
            dists,
            scope,
            work,
        }
    }

    pub fn edge(&self, axis: Axis) -> usize {
        self.edges.get(axis)
    }

    pub fn distribution(&self, axis: Axis) -> Distribution {
        self.dists.get(axis)
    }

    pub(crate) fn scope(&self) -> LevelScope {
        self.scope
    }

    pub(crate) fn role(&self) -> LevelRole {
        self.scope.role()
    }

    /// The axes this level cuts, in canonical order. A level keeps every axis of the operation,
    /// so an operand's projection (`{M, N}`) still finds its contraction here.
    pub(crate) fn axes(&self) -> Vec<Axis> {
        (0..self.dists.len())
            .map(|i| self.dists.axis_at(i))
            .collect()
    }

    pub(crate) fn contains(&self, axis: Axis) -> bool {
        self.dists.contains(axis)
    }

    /// The axes this level distributes as one, if any.
    pub fn work(&self) -> Option<&Work> {
        self.work.as_ref()
    }

    /// The space one region of this level covers: every axis of `space` cut to its edge, static
    /// whatever the parent was (an edge is comptime). Position-free; the positions are the walk.
    pub fn child(&self, space: &Space) -> Space {
        Space::new(
            &space
                .axes()
                .map(|axis| (axis, self.edge(axis)))
                .collect::<Vec<_>>(),
        )
    }

    /// Whether this level's edge on `axis` fails to divide the extent `space` hands it, leaving a
    /// partial tile that needs masking. Host-side, static extents.
    pub(crate) fn overhangs(&self, space: &Space, axis: Axis) -> bool {
        !space.extent(axis).is_multiple_of(self.edge(axis))
    }

    /// Tiles along `axis` of `space`: `ceil(extent / edge)`, so an indivisible axis gets a
    /// trailing partial tile (its overhang is masked at read/write). Host-side, static extents.
    pub(crate) fn count(&self, space: &Space, axis: Axis) -> usize {
        space.extent(axis).div_ceil(self.edge(axis))
    }

    /// Whether `axis` is `Spatial` `TilesEach(1)`: its walk count is comptime `1`, so a step
    /// decode can skip it.
    pub(crate) fn single_tile(&self, axis: Axis) -> bool {
        self.distribution(axis).single_tile()
    }

    /// Whether this level cuts `axis` of `space` into a single, statically-known tile, so its
    /// walk coordinate is a constant `0`, even on a rolled walk. A `Dynamic` axis has no comptime
    /// count and is never statically single.
    pub(crate) fn single_static_tile(&self, space: &Space, axis: Axis) -> bool {
        !space.is_dynamic(axis) && self.count(space, axis) == 1
    }

    /// The per-instance tile count of `axis`, `None` when it is runtime.
    pub(crate) fn per_instance_tiles(&self, space: &Space, axis: Axis) -> Option<usize> {
        let edge = self.edge(axis);
        match self.distribution(axis) {
            Distribution::Sequential => match space.extent_raw(axis) {
                Extent::Static(e) => Some(e.div_ceil(edge)),
                Extent::Dynamic => None,
            },
            Distribution::Spatial { coverage, .. } => match coverage {
                Coverage::TilesEach(t) => Some(t),
                Coverage::Instances(n) => match space.extent_raw(axis) {
                    Extent::Static(e) => Some(e.div_ceil(edge).div_ceil(n)),
                    Extent::Dynamic => None,
                },
            },
        }
    }

    /// The `m × n` grid a partition level cuts, read off `space`'s trailing two axes; leading
    /// (batch) axes must hand out one tile. Valid only on a [`Partition`](LevelRole::Partition)
    /// level; the role says whether it applies, this only reads the counts.
    pub(crate) fn partition_grid(&self, space: &Space) -> (usize, usize) {
        let rank = space.rank();
        for (p, axis) in space.axes().enumerate() {
            let tiles = self
                .per_instance_tiles(space, axis)
                .expect("plane partition level: tile counts must be comptime");
            assert!(
                p >= rank - 2 || tiles == 1,
                "plane partition level: leading (batch) axes must hand out one tile"
            );
        }
        (
            self.per_instance_tiles(space, space.axis_at(rank - 2))
                .unwrap(),
            self.per_instance_tiles(space, space.axis_at(rank - 1))
                .unwrap(),
        )
    }

    /// Whether this level cuts `space`'s tiles into an m×n grid larger than 1×1, so each region
    /// must be selected by a comptime coordinate. An instance level and a degenerate 1×1
    /// partition (a k-step walk) both cut nothing.
    pub(crate) fn cuts_tiles(&self, space: &Space) -> bool {
        match self.role() {
            LevelRole::Instance => false,
            LevelRole::Partition => self.partition_grid(space) != (1, 1),
        }
    }

    /// Whether a walk of this level over `space` leaves `operand`'s window unchanged: every axis
    /// the walk actually steps (more than one tile) is absent from the operand, the same
    /// structural fact as broadcast omission. A staged walk fills such an operand once, above
    /// the loop. Host-side, static extents.
    pub(crate) fn walk_invariant(&self, space: &Space, operand: &Space) -> bool {
        space
            .axes()
            .all(|axis| self.count(space, axis) == 1 || !operand.contains(axis))
    }

    /// How many instances `axis` is dealt out to at this level, where that is comptime: the
    /// pinned count, or the tile grid divided by each instance's share. `None` where the grid is
    /// not known here: the extent is [`Dynamic`](Extent::Dynamic), or `space` is a projection
    /// that dropped the axis (a drain descending an output through its own space).
    pub(crate) fn instances_along(&self, space: &Space, axis: Axis) -> Option<usize> {
        let coverage = self.distribution(axis).coverage();
        match coverage.instances_const() {
            Some(instances) => Some(instances),
            None if !space.contains(axis) => None,
            None => match space.extent_raw(axis) {
                Extent::Static(_) => Some(coverage.instances(self.count(space, axis))),
                Extent::Dynamic => None,
            },
        }
    }

    /// The instance-index weight `spanned`'s own axis list cannot see: the instance counts of the
    /// same-scope axes *inside* `axis` that this level distributes and `spanned` does not span.
    /// A projected space is why: the index's odometer belongs to the level, so an operand not
    /// spanning a contracted axis must still divide it out to find its own digit, and reading
    /// omitted axes as weight `1` aliases the outer digits onto one value. Panics where such an
    /// axis has no comptime count: assuming `1` is exactly that aliasing.
    pub(crate) fn inner_weight_unspanned(&self, spanned: &Space, axis: Axis) -> usize {
        let scope = self.distribution(axis).scope();
        self.axes()
            .iter()
            .skip_while(|&&a| a != axis)
            .skip(1)
            .filter(|&&a| !spanned.contains(a) && self.distribution(a).scope() == scope)
            .map(|&a| {
                self.distribution(a)
                    .coverage()
                    .instances_const()
                    .unwrap_or_else(|| {
                        panic!(
                            "Level::inner_weight_unspanned: {a:?} is distributed inside {axis:?} \
                             at the same scope but this operand does not span it, and its \
                             instance count is not comptime, so {axis:?}'s digit of the \
                             instance index cannot be decoded"
                        )
                    })
            })
            .product()
    }

    /// What the plane's lanes are to `spanned`'s cells once this level is dealt out: an axis the
    /// operand does not span is folded (lanes hold partials), one it spans is carried.
    pub(crate) fn lane_share(&self, spanned: &Space) -> LaneShare {
        // Innermost first, so `weight` is the axis's stride in the lane index as it is reached,
        // the same least-significant-last ordering `Walk::from_counts` decodes with.
        let (mut weight, mut fold_mask) = (1usize, 0usize);
        for axis in self.axes().into_iter().rev() {
            let Distribution::Spatial {
                scope: ComputeScope::Unit,
                coverage,
                ..
            } = self.distribution(axis)
            else {
                continue;
            };
            // Asserted, not skipped: a `Unit` axis always carries an `Instances` count, and
            // passing over one whose count we could not read would shift every inner axis's
            // bits by its width.
            let lanes = coverage
                .instances_const()
                .expect("Level::lane_share: a Unit axis must carry a const instance count");
            if lanes == 1 {
                continue;
            }
            assert!(
                lanes.is_power_of_two(),
                "Level::lane_share: {axis:?} rides {lanes} lanes, which is not a power of two, \
                 so its partials are not a bit range"
            );
            if !spanned.contains(axis) {
                fold_mask |= (lanes - 1) * weight;
            }
            weight *= lanes;
        }
        match fold_mask {
            0 => LaneShare::Whole,
            // Every lane's bit folded: nothing is carried, so the plane shares the one cell.
            mask if mask == weight - 1 => LaneShare::Plane,
            fold_mask => LaneShare::Group { fold_mask },
        }
    }

    /// What one instance of an operand spanning `spanned` holds of its cells after this level
    /// is dealt out over `space`: [`Partial`](SplitShare::Partial) where a `Plane` or `Cube` axis
    /// the operand does not span is dealt across several instances, so each contracts a slice.
    /// Asked with the level's whole space, not the operand's projection: a projection has dropped
    /// the contracted axis and so cannot tell a split from a cut whose edge is the whole axis.
    /// Answered conservatively where the instance count is not comptime, since calling it whole
    /// loses every partial but one.
    pub(crate) fn split_share_of(&self, space: &Space, spanned: &Space) -> SplitShare {
        // Work distributed as one is not an axis: a share of it covers part of a cell whenever
        // the index runs over an axis the operand does not span, and which part is not something
        // the level's per-axis distributions record.
        if let Some(work) = self.work()
            && work.axes().iter().any(|axis| !spanned.contains(*axis))
        {
            return SplitShare::Partial;
        }
        let split = self.axes().into_iter().any(|axis| {
            // An axis the operand spans is carried, not split: it gives each instance a cell of
            // its own rather than a slice of one.
            if spanned.contains(axis) {
                return false;
            }
            match self.distribution(axis).scope() {
                Some(ComputeScope::Cube(_)) | Some(ComputeScope::Plane) => {}
                Some(ComputeScope::Unit) | None => return false,
            }
            self.instances_along(space, axis) != Some(1)
        });
        if split {
            SplitShare::Partial
        } else {
            SplitShare::Whole
        }
    }

    /// Whether anything rides this level's lanes.
    pub(crate) fn rides_lanes(&self) -> bool {
        self.axes().into_iter().any(|axis| {
            let Distribution::Spatial {
                scope: ComputeScope::Unit,
                coverage,
                ..
            } = self.distribution(axis)
            else {
                return false;
            };
            coverage.instances_const() != Some(1)
        })
    }
}

/// Collects what one level says, in two verbs: [`distribute`](Self::distribute) hands some axes'
/// regions to a hardware scope's workers, [`walk`](Self::walk) leaves the rest to every one of
/// them. Between them they name each of the space's axes exactly once.
pub struct LevelCuts {
    cuts: Vec<(Axis, Cut)>,
    work: Option<Work>,
}

impl LevelCuts {
    fn new() -> Self {
        LevelCuts {
            cuts: Vec::new(),
            work: None,
        }
    }

    /// Nobody owns these axes: every worker on this level covers all of their tiles, walking them
    /// one at a time. `axes` pairs each axis with its tile edge.
    ///
    /// The contraction of a matmul is the everyday one. So is any axis a level leaves alone,
    /// which is most of them at most levels.
    pub fn walk(&mut self, axes: &[(Axis, usize)]) -> &mut Self {
        self.cuts.extend(
            axes.iter()
                .map(|&(axis, edge)| (axis, Cut::sequential(edge))),
        );
        self
    }

    /// Hand these axes' regions to `dist`'s workers, in order: as many workers as regions means
    /// one each, fewer means each takes a run. `axes` pairs each axis with its tile edge, and
    /// `dist` says who runs the tiles and how many of them there are ([`cubes`](crate::cubes),
    /// [`planes`](crate::planes), [`lanes`](crate::lanes) and their knobs).
    ///
    /// One axis, or one region each, is a box of the grid, so every axis gets a dial of its own
    /// and two lines make a two-dimensional box. Several axes sharing a stated count are read as a
    /// single index instead, so a worker takes a share of the whole rather than a box of it: a
    /// dial per axis always yields a box, and a share that begins inside one region and ends
    /// inside another is not one.
    ///
    /// That index runs over these axes' tiles at this level and one region of the level below, so
    /// a share can end part way through a region's own work ([`Walk::window`](crate::Walk::window)
    /// is how a kernel takes its share).
    ///
    /// An axis named here is not named by [`walk`](Self::walk): a level states each of its axes
    /// once.
    pub fn distribute(&mut self, dist: Spatial, axes: &[(Axis, usize)]) -> &mut Self {
        match dist.handout(axes.len()) {
            Handout::Dial => self.cuts.extend(axes.iter().map(|&(axis, edge)| {
                (
                    axis,
                    Cut {
                        edge,
                        dist: dist.into(),
                    },
                )
            })),
            Handout::OneIndex => {
                assert!(
                    self.work.is_none(),
                    "LevelCuts::distribute: this level already distributes work; state it once"
                );
                // A share is a range of one index, and lanes that reduce in registers have to
                // reach their reduction together. Ranges put them on different regions, so they
                // never would.
                assert!(
                    dist.scope() != ComputeScope::Unit,
                    "LevelCuts::distribute: the plane's lanes combine in registers, which needs \
                     them in lockstep, and lanes holding different shares never are. Distribute \
                     one axis at a time across them instead."
                );
                // A share is walked as a nest: consecutive steps of one region under one
                // accumulator, then the next region. Turns would put a different region under it
                // at every step.
                assert!(
                    dist.spread() == Spread::Contiguous,
                    "LevelCuts::distribute: a share is a run of the index, so its steps are \
                     consecutive; instances taking turns would leave no region long enough to \
                     accumulate in. Distribute one interleaved axis instead."
                );
                self.cuts.extend(
                    axes.iter()
                        .map(|&(axis, edge)| (axis, Cut::sequential(edge))),
                );
                self.work = Some(Work::new(
                    axes.iter().map(|&(axis, _)| axis).collect(),
                    dist,
                ));
            }
        }
        self
    }
}

/// Several axes' work distributed as one.
///
/// Dealing each axis on its own gives an instance the product of its per-axis runs, which is a
/// box of the grid. These axes are read as a single index instead, so an instance takes a share
/// of the whole rather than a box of it: the shares that no box can describe are exactly the ones
/// that balance a grid its shape cannot divide.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Work {
    axes: Vec<Axis>,
    dist: Spatial,
}

impl Work {
    pub(crate) fn new(axes: Vec<Axis>, dist: Spatial) -> Self {
        Work { axes, dist }
    }

    /// The axes read as one index.
    pub(crate) fn axes(&self) -> &[Axis] {
        &self.axes
    }

    pub(crate) fn scope(&self) -> ComputeScope {
        self.dist.scope()
    }

    /// How many instances share the work. Pinned, not derived: the index's length is the whole
    /// level's grid and can be runtime, so nothing here could divide it.
    pub(crate) fn instances(&self) -> usize {
        match self.dist.coverage() {
            Coverage::Instances(n) => n,
            Coverage::TilesEach(_) => panic!(
                "Level::new: state how many instances share the work (`.instances(n)`); a \
                 share of the whole cannot be derived from a grid whose length is only known at \
                 launch"
            ),
        }
    }
}

/// How finely a level separates its tiles: the smallest hardware scope any of its axes rides.
/// Decided once, when the level is built, so no consumer re-folds the per-axis distributions.
///
/// The finest scope wins. A level with an axis on a cube dim and another on planes reaches
/// inside a cube, and what a level separates inside a cube the cube's own cooperative
/// transports already spread.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub(crate) enum LevelScope {
    /// Every axis `Sequential`: one instance walks the whole grid.
    Sequential,
    /// Some axis rides a cube dim and none reaches inside a cube, so the level separates
    /// exactly what the launch grid does.
    Cubes,
    /// Some axis rides the cube's planes.
    Planes,
    /// Some axis rides a plane's lanes.
    Lanes,
}

impl LevelScope {
    fn of(dist: Distribution) -> Self {
        match dist.scope() {
            None => LevelScope::Sequential,
            Some(ComputeScope::Cube(_)) => LevelScope::Cubes,
            Some(ComputeScope::Plane) => LevelScope::Planes,
            Some(ComputeScope::Unit) => LevelScope::Lanes,
        }
    }

    /// The coarse reading, for consumers that only ask whether the level spreads at all.
    pub(crate) fn role(self) -> LevelRole {
        match self {
            LevelScope::Sequential => LevelRole::Partition,
            LevelScope::Cubes | LevelScope::Planes | LevelScope::Lanes => LevelRole::Instance,
        }
    }
}

/// Whether a level spreads its tiles across hardware at all, which is all most consumers ask.
/// A view over [`LevelScope`], never stored: the scope is the level's own state.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum LevelRole {
    /// Spreads its tiles across hardware instances (`Spatial` on some axis).
    Instance,
    /// Partitions its tiles sequentially across a grid (every axis `Sequential`).
    Partition,
}

/// How one axis is cut at one level: the sub-tile `edge` and how the level hands the tiles out.
#[derive(Clone, Copy, Debug)]
struct Cut {
    edge: usize,
    dist: Distribution,
}

impl Cut {
    fn sequential(edge: usize) -> Self {
        Cut {
            edge,
            dist: Distribution::Sequential,
        }
    }
}
