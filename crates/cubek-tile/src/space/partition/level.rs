//! One decomposition [`Level`]: which axes of a space a loop cuts, to what tile edge, and who
//! takes the tiles. The value a kernel's loop states, under the verb that says who takes the
//! regions ([`Space::cubes`](crate::Space::cubes), [`Space::walk`](crate::Space::walk), and the
//! rest), the value a [`Region`](crate::Region) carries down to `at`, and the value a launch
//! sizes its grid from ([`Nest`](crate::Nest)). A blueprint hands the same value to both, one
//! method per level, so the grid and the loops cannot disagree.
//!
//! One constructor per verb: [`Level::cubes`], [`Level::planes`], [`Level::lanes`] deal tiles
//! to a hardware scope's workers, [`Level::walk`] steps through them. A level names only the
//! axes it touches; every other axis is handed down whole.

use super::{ComputeScope, Coverage, CubeAxis, Distribution, Spread};
use crate::{Axis, ByAxis, Extent, LaneShare, Space, SplitShare};

/// What a level does to one axis: cuts it into tiles of `edge`, or leaves it whole. Whole is
/// what a level says of an axis it does not name, which is the only way to leave a
/// [`Dynamic`](Extent::Dynamic) axis alone, since its extent is no number a cut could name.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Edge {
    Cut(usize),
    Whole,
}

/// One axis's tiles dealt to a scope's workers: the tile edge, how many workers take them and
/// which ones each takes. The entry of [`Level::cubes`], [`Level::planes`] and
/// [`Level::lanes`]; a plain `(axis, edge)` converts to one tile per worker, in runs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Deal {
    axis: Axis,
    edge: usize,
    spread: Spread,
    coverage: Coverage,
}

impl Deal {
    /// `axis` cut to `edge`, one tile per worker.
    pub fn new(axis: Axis, edge: usize) -> Self {
        Deal {
            axis,
            edge,
            spread: Spread::Contiguous,
            coverage: Coverage::TilesEach(1),
        }
    }

    /// `n` workers take the tiles, each a run of `grid / n`. Replaces whatever count stood:
    /// `workers · tiles_each = grid`, so stating either states the other.
    pub fn across(mut self, n: usize) -> Self {
        self.coverage = Coverage::Instances(n);
        self
    }

    /// Each worker takes `t` tiles; `grid / t` workers run. The twin of [`across`](Self::across).
    pub fn each(mut self, t: usize) -> Self {
        self.coverage = Coverage::TilesEach(t);
        self
    }

    /// Workers take turns rather than each taking a run, so neighbouring workers touch
    /// neighbouring tiles. What a read wants whenever those tiles are neighbouring words.
    pub fn interleaved(mut self) -> Self {
        self.spread = Spread::Interleaved;
        self
    }

    fn distribution(&self, scope: ComputeScope) -> Distribution {
        Distribution::Spatial {
            scope,
            spread: self.spread,
            coverage: self.coverage,
        }
    }
}

impl From<(Axis, usize)> for Deal {
    fn from((axis, edge): (Axis, usize)) -> Deal {
        Deal::new(axis, edge)
    }
}

/// One decomposition level of a space: the axes it names, each with its tile edge and who takes
/// the tiles, plus the axes it deals as one. An axis it does not name is handed down whole.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Level {
    edges: ByAxis<Edge>,
    dists: ByAxis<Distribution>,
    scope: LevelScope,
    work: Option<Work>,
}

impl Level {
    /// Every worker steps through `steps`' tiles, one at a time; `(axis, edge)` each. The
    /// contraction of a matmul is the everyday one. Stated under [`walk`](crate::Space::walk).
    pub fn walk(steps: &[(Axis, usize)]) -> Level {
        let dists: Vec<_> = steps
            .iter()
            .map(|&(axis, _)| (axis, Distribution::Sequential))
            .collect();
        Level::build(steps, &dists, LevelScope::Sequential, None)
    }

    /// The tiles of each entry ride a cube dimension of the launch grid, in order: the first
    /// entry's cubes are `X`, the second's `Y`, the third's `Z`. One tile per cube unless the
    /// entry says otherwise ([`Deal`]). Batch axes go on `Z` too ([`batches`](Self::batches)).
    /// Stated under [`cubes`](crate::Space::cubes); every entry is one box of the grid.
    pub fn cubes<D: Into<Deal> + Clone>(deals: &[D]) -> Level {
        assert!(
            deals.len() <= 3,
            "Level::cubes: {} entries, but a launch grid has three dimensions",
            deals.len()
        );
        let scopes = [CubeAxis::X, CubeAxis::Y, CubeAxis::Z];
        Level::dealt(deals, |i| ComputeScope::Cube(scopes[i]), LevelScope::Cubes)
    }

    /// The tiles of each entry ride the cube's planes, one tile per plane unless the entry says
    /// otherwise ([`Deal`]). Several entries make a box per plane. Stated under
    /// [`planes`](crate::Space::planes).
    pub fn planes<D: Into<Deal> + Clone>(deals: &[D]) -> Level {
        Level::dealt(deals, |_| ComputeScope::Plane, LevelScope::Planes)
    }

    /// The tiles of each entry ride some of the plane's lanes; every entry states how many
    /// ([`Deal::across`]), since the plane is carved between the entries and their counts must
    /// multiply to its width. Stated under [`lanes`](crate::Space::lanes).
    pub fn lanes(deals: &[Deal]) -> Level {
        for deal in deals {
            assert!(
                matches!(deal.coverage, Coverage::Instances(_)),
                "Level::lanes: {:?} states no lane count; say how many lanes take it \
                 (`Deal::new(axis, edge).across(n)`)",
                deal.axis
            );
        }
        Level::dealt(deals, |_| ComputeScope::Unit, LevelScope::Lanes)
    }

    /// One tile of each of `axes` per cube, on the grid's `Z` dimension: the batch axes of a
    /// [`cubes`](Self::cubes) level.
    pub fn batches(mut self, axes: &[Axis]) -> Level {
        assert!(
            self.scope == LevelScope::Cubes,
            "Level::batches: batches ride cubes; this level deals to {:?}",
            self.scope
        );
        for &axis in axes {
            self.push(
                axis,
                Edge::Cut(1),
                Deal::new(axis, 1).distribution(ComputeScope::Cube(CubeAxis::Z)),
            );
        }
        self
    }

    /// The tiles of every entry read as one index, of which each of `n` workers takes a share:
    /// a run of the whole rather than a box of it, which is what balances a grid its shape
    /// cannot divide. The index runs over this level's tiles and one region of the level
    /// below, so a share can end part way through a region's own work
    /// ([`Walk::window`](crate::Walk::window) is how a kernel takes its share).
    ///
    /// The plain entries' boxes go: the workers ride the scope as one, on its first dimension.
    /// Not for lanes: they combine in registers, which needs them in lockstep, and lanes holding
    /// different shares never are.
    pub fn shared_by(mut self, n: usize) -> Level {
        let scope = match self.scope {
            LevelScope::Cubes => ComputeScope::Cube(CubeAxis::X),
            LevelScope::Planes => ComputeScope::Plane,
            LevelScope::Lanes => panic!(
                "Level::shared_by: the plane's lanes combine in registers, which needs them in \
                 lockstep, and lanes holding different shares never are"
            ),
            LevelScope::Sequential => {
                panic!("Level::shared_by: a walk has no workers to share its tiles")
            }
        };
        assert!(
            self.work.is_none(),
            "Level::shared_by: this level already deals its tiles as one"
        );
        let axes = self.axes();
        for &axis in &axes {
            let plain = matches!(
                self.dists.get(axis),
                Distribution::Spatial {
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                    ..
                }
            );
            assert!(
                plain,
                "Level::shared_by: {axis:?} states a count or a spread of its own, which a share \
                 of the whole has no use for"
            );
        }
        self.dists = self.dists.map(|_, _| Distribution::Sequential);
        self.work = Some(Work::new(axes, scope, n));
        self
    }

    fn dealt<D: Into<Deal> + Clone>(
        deals: &[D],
        scope_of: impl Fn(usize) -> ComputeScope,
        kind: LevelScope,
    ) -> Level {
        let deals: Vec<Deal> = deals.iter().cloned().map(Into::into).collect();
        let cuts: Vec<_> = deals.iter().map(|d| (d.axis, d.edge)).collect();
        let dists: Vec<_> = deals
            .iter()
            .enumerate()
            .map(|(i, d)| (d.axis, d.distribution(scope_of(i))))
            .collect();
        Level::build(&cuts, &dists, kind, None)
    }

    fn build(
        cuts: &[(Axis, usize)],
        dists: &[(Axis, Distribution)],
        scope: LevelScope,
        work: Option<Work>,
    ) -> Level {
        for (i, &(axis, _)) in cuts.iter().enumerate() {
            assert!(
                !cuts[..i].iter().any(|&(a, _)| a == axis),
                "Level: {axis:?} is named twice; a level states each of its axes once"
            );
        }
        let edges: Vec<_> = cuts
            .iter()
            .map(|&(axis, edge)| (axis, Edge::Cut(edge)))
            .collect();
        Level {
            edges: ByAxis::new(&edges),
            dists: ByAxis::new(dists),
            scope,
            work,
        }
    }

    fn push(&mut self, axis: Axis, edge: Edge, dist: Distribution) {
        assert!(
            !self.edges.contains(axis),
            "Level: {axis:?} is named twice; a level states each of its axes once"
        );
        let mut edges: Vec<_> = self
            .axes()
            .into_iter()
            .map(|a| (a, self.edges.get(a)))
            .collect();
        let mut dists: Vec<_> = self
            .axes()
            .into_iter()
            .map(|a| (a, self.dists.get(a)))
            .collect();
        edges.push((axis, edge));
        dists.push((axis, dist));
        self.edges = ByAxis::new(&edges);
        self.dists = ByAxis::new(&dists);
    }

    /// The tile edge this level cuts `axis` to. An axis the level does not name has no edge of
    /// its own; its extent is the space's.
    pub fn edge(&self, axis: Axis) -> usize {
        match self.edge_kind(axis) {
            Edge::Cut(edge) => edge,
            Edge::Whole => panic!(
                "Level::edge: {axis:?} is not named at this level, so it is handed down whole; \
                 its extent is the space's"
            ),
        }
    }

    /// The cut of `axis`: its edge where the level names it, [`Whole`](Edge::Whole) where not.
    pub fn edge_kind(&self, axis: Axis) -> Edge {
        if self.edges.contains(axis) {
            self.edges.get(axis)
        } else {
            Edge::Whole
        }
    }

    /// The extent one region of this level covers along `axis` of `space`: the cut edge, or the
    /// space's own extent (static or not) where the axis is handed down whole.
    pub(crate) fn edge_in(&self, space: &Space, axis: Axis) -> Extent {
        match self.edge_kind(axis) {
            Edge::Cut(edge) => Extent::Static(edge),
            Edge::Whole => space.extent_raw(axis),
        }
    }

    /// Who takes `axis`'s tiles: [`Sequential`](Distribution::Sequential) where the level walks
    /// it or does not name it.
    pub fn distribution(&self, axis: Axis) -> Distribution {
        if self.dists.contains(axis) {
            self.dists.get(axis)
        } else {
            Distribution::Sequential
        }
    }

    pub(crate) fn scope(&self) -> LevelScope {
        self.scope
    }

    pub(crate) fn role(&self) -> LevelRole {
        self.scope.role()
    }

    /// The axes this level names, in the order it named them.
    pub(crate) fn axes(&self) -> Vec<Axis> {
        (0..self.dists.len())
            .map(|i| self.dists.axis_at(i))
            .collect()
    }

    /// The axes this level distributes as one, if any.
    pub fn work(&self) -> Option<&Work> {
        self.work.as_ref()
    }

    /// The space one region of this level covers: every axis of `space` cut to its edge (static,
    /// an edge is comptime) or handed down whole. Position-free; the positions are the walk.
    pub fn child(&self, space: &Space) -> Space {
        Space::from_extents(
            &space
                .axes()
                .map(|axis| (axis, self.edge_in(space, axis)))
                .collect::<Vec<_>>(),
        )
    }

    /// Whether this level's edge on `axis` fails to divide the extent `space` hands it, leaving a
    /// partial tile that needs masking. Host-side, static extents.
    pub(crate) fn overhangs(&self, space: &Space, axis: Axis) -> bool {
        match self.edge_kind(axis) {
            Edge::Cut(edge) => !space.extent(axis).is_multiple_of(edge),
            Edge::Whole => false,
        }
    }

    /// Tiles along `axis` of `space`: `ceil(extent / edge)`, so an indivisible axis gets a
    /// trailing partial tile (its overhang is masked at read/write). Host-side, static extents.
    pub(crate) fn count(&self, space: &Space, axis: Axis) -> usize {
        match self.edge_kind(axis) {
            Edge::Cut(edge) => space.extent(axis).div_ceil(edge),
            Edge::Whole => 1,
        }
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
        match self.edge_kind(axis) {
            Edge::Cut(_) => !space.is_dynamic(axis) && self.count(space, axis) == 1,
            Edge::Whole => true,
        }
    }

    /// The per-instance tile count of `axis`, `None` when it is runtime.
    pub(crate) fn per_instance_tiles(&self, space: &Space, axis: Axis) -> Option<usize> {
        let edge = match self.edge_kind(axis) {
            Edge::Cut(edge) => edge,
            Edge::Whole => return Some(1),
        };
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

/// Several axes' work distributed as one.
///
/// Dealing each axis on its own gives an instance the product of its per-axis runs, which is a
/// box of the grid. These axes are read as a single index instead, so an instance takes a share
/// of the whole rather than a box of it: the shares that no box can describe are exactly the ones
/// that balance a grid its shape cannot divide.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Work {
    axes: Vec<Axis>,
    scope: ComputeScope,
    instances: usize,
}

impl Work {
    pub(crate) fn new(axes: Vec<Axis>, scope: ComputeScope, instances: usize) -> Self {
        Work {
            axes,
            scope,
            instances,
        }
    }

    /// The axes read as one index.
    pub(crate) fn axes(&self) -> &[Axis] {
        &self.axes
    }

    pub(crate) fn scope(&self) -> ComputeScope {
        self.scope
    }

    /// How many instances share the work. Pinned, not derived: the index's length is the whole
    /// level's grid and can be runtime, so nothing here could divide it.
    pub(crate) fn instances(&self) -> usize {
        self.instances
    }
}

/// Who takes a level's tiles: the verb it was built under, and the loop verb that must state
/// it. Set once, by the constructor, so no consumer re-folds the per-axis distributions.
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
    /// The loop verb that states a level of this scope.
    pub(crate) fn verb(self) -> &'static str {
        match self {
            LevelScope::Sequential => "walk",
            LevelScope::Cubes => "cubes",
            LevelScope::Planes => "planes",
            LevelScope::Lanes => "lanes",
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
    edge: Edge,
    dist: Distribution,
}

impl Cut {
    fn sequential(edge: Edge) -> Self {
        Cut {
            edge,
            dist: Distribution::Sequential,
        }
    }
}
