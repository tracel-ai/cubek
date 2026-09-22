//! One [`Level`]: which axes a loop steps, in tiles of what size, how many, and who takes them.
//! One entry of a [`Partitioning`](crate::Partitioning): a [`Region`](crate::Region) hands it to
//! `at`, a [`Launcher`](crate::Launcher) sizes its grid from it, so grid and loops cannot disagree.
//!
//! Built only by [`Tiling`](crate::Tiling), leaf up: a level's tile on an axis is the product of
//! what was stated below it, and its [`Count`] is what it stated. A level names only the axes it
//! touches; the rest pass down whole. [`Level::every`], a walk over a region, goes through it too.

use super::{ComputeScope, CubeAxis, CubeOrder, Distribution, Spread};
use crate::{Axis, AxisMap, Extent, LaneShare, MatrixAxes, Space, SplitShare, Tiling};

/// How many tiles a level takes along one of its axes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Count {
    /// This many, stated: a walk's steps, or a scope's workers with one tile each. A count
    /// cannot fail to divide, and a tile under it cannot overhang.
    Of(usize),
    /// Every tile the level above hands down, one per step or per worker: the count only the
    /// launch knows, and the one place a tile can reach past the extent.
    Every,
    /// Every tile, dealt across this many workers in runs: a closed axis returning to the cube
    /// level to be split, and the one run whose length the kernel computes.
    Across(usize),
}

/// What a walk counts along one axis of a level: a constant, or the extent handed down in
/// this tile ([`Level::grid`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GridCount {
    Const(usize),
    Extent(usize),
}

impl Count {
    /// The count where it is a stated number: `Of` and `Across` are, `Every` is the launch's.
    pub(crate) fn stated(self) -> Option<usize> {
        match self {
            Count::Of(n) | Count::Across(n) => Some(n),
            Count::Every => None,
        }
    }

    /// Tiles this count takes along an axis of `extent`, in tiles of `tile`: the stated number,
    /// or every tile the extent holds, the last one partial where it does not divide.
    pub(crate) fn tiles(self, extent: usize, tile: usize) -> usize {
        match self {
            Count::Of(n) => n,
            Count::Every | Count::Across(_) => extent.div_ceil(tile),
        }
    }

    /// [`tiles`](Count::tiles) where the host can prove it: a stated count needs no extent, and
    /// every tile of a [`Dynamic`](Extent::Dynamic) axis is the launch's to count.
    pub(crate) fn tiles_const(self, extent: Extent, tile: usize) -> Option<usize> {
        match (self, extent) {
            (Count::Of(n), _) => Some(n),
            (_, Extent::Static(extent)) => Some(self.tiles(extent, tile)),
            (_, Extent::Dynamic) => None,
        }
    }

    /// What a walk counts along an axis stepped in `tile`: the stated number, or the extent it
    /// is handed, in that tile.
    pub(crate) fn grid(self, tile: usize) -> GridCount {
        match self {
            Count::Of(n) => GridCount::Const(n),
            Count::Every | Count::Across(_) => GridCount::Extent(tile),
        }
    }
}

/// One axis of a level: the tile it steps in, how many, and who takes them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Entry {
    tile: usize,
    count: Count,
    dist: Distribution,
}

/// One decomposition level of a space: the axes it names, each with its tile, its count and who
/// takes the tiles, plus the axes it deals as one. An axis it does not name is handed down whole.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Level {
    entries: AxisMap<Entry>,
    scope: LevelScope,
    work: Option<Work>,
    /// Planes that fill this walk's stages and take no tile of any level
    /// ([`filled_by`](Level::filled_by)).
    fillers: usize,
    /// The order a cube level deals its boxes to the grid ([`ordered`](Level::ordered)).
    /// [`RowMajor`](CubeOrder::RowMajor) everywhere else, which is what every level that
    /// states nothing gets.
    order: CubeOrder,
}

impl Level {
    /// The one level that walks a region in tiles of `tile`, taking every one of them — what
    /// [`Region::over`](crate::Region::over) asks for, and the shape a stated count never has.
    ///
    /// The leaf-up spelling of the same thing, stated once instead of twice:
    /// `Tiling::leaf(tile).walk_every(its axes).level()`. It goes through
    /// [`Tiling`](crate::Tiling) like every other level, so there is still one builder.
    pub fn every(tile: &[(Axis, usize)]) -> Level {
        let axes: Vec<Axis> = tile.iter().map(|&(axis, _)| axis).collect();
        Tiling::leaf(tile).walk_every(&axes).level()
    }

    /// A level of `scope` over `entries`, each `(axis, tile, count, who takes it)`. The
    /// builder's constructor: [`Tiling`](crate::Tiling) is the only caller, and it states the
    /// tile as the product of the levels below.
    pub(crate) fn new(scope: LevelScope, entries: &[(Axis, usize, Count, Distribution)]) -> Level {
        for (i, &(axis, ..)) in entries.iter().enumerate() {
            assert!(
                !entries[..i].iter().any(|&(a, ..)| a == axis),
                "Level: {axis:?} is named twice; a level states each of its axes once"
            );
        }
        for &(axis, tile, count, dist) in entries {
            assert!(tile > 0, "Level: {axis:?} has a tile of nothing");
            match (count, dist) {
                (
                    Count::Across(_),
                    Distribution::Spatial {
                        scope: ComputeScope::Cube(_),
                        ..
                    },
                ) => {}
                (Count::Across(_), _) => {
                    panic!("Level: {axis:?} is dealt across workers in runs, which only cubes take")
                }
                (
                    Count::Every,
                    Distribution::Spatial {
                        scope: ComputeScope::Unit,
                        ..
                    },
                )
                | (
                    Count::Every,
                    Distribution::Spatial {
                        scope: ComputeScope::Plane,
                        ..
                    },
                ) => {
                    panic!(
                        "Level: {axis:?} takes every tile on a plane's workers, whose count is \
                         the device's; state how many"
                    )
                }
                _ => {}
            }
        }
        let entries: Vec<_> = entries
            .iter()
            .map(|&(axis, tile, count, dist)| (axis, Entry { tile, count, dist }))
            .collect();
        Level {
            entries: AxisMap::new(&entries),
            scope,
            work: None,
            fillers: 0,
            order: CubeOrder::RowMajor,
        }
    }

    /// One tile of each of `axes` per cube, on the grid's `Z` dimension: the batch axes of a
    /// cube level.
    pub(crate) fn batches(mut self, axes: &[Axis]) -> Level {
        assert!(
            self.scope == LevelScope::Cubes,
            "Level::batches: batches ride cubes; this level cuts to {:?}",
            self.scope
        );
        for &axis in axes {
            self.push(
                axis,
                Entry {
                    tile: 1,
                    count: Count::Every,
                    dist: Distribution::Spatial {
                        scope: ComputeScope::Cube(CubeAxis::Z),
                        spread: Spread::Contiguous,
                    },
                },
            );
        }
        self
    }

    /// Every entry's tiles read as one index, of which each of `n` workers takes a share: a run
    /// of the whole, not a box, which balances a grid its shape cannot divide. It also spans a
    /// region of the level below, so a [`Walk::window`](crate::Walk::window) can end mid-region.
    ///
    /// The plain entries' boxes go: the workers ride the scope as one, on its first dimension.
    /// Not for lanes: they combine in registers, which needs them in lockstep, and lanes holding
    /// different shares never are.
    pub(crate) fn shared_by(mut self, n: usize) -> Level {
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
            "Level::shared_by: this level already cuts its tiles as one"
        );
        let axes = self.axes();
        for &axis in &axes {
            let entry = self.entries.get(axis);
            let plain = entry.count == Count::Every
                && matches!(
                    entry.dist,
                    Distribution::Spatial {
                        spread: Spread::Contiguous,
                        ..
                    }
                );
            assert!(
                plain,
                "Level::shared_by: {axis:?} states a count or a spread of its own, which a share \
                 of the whole has no use for"
            );
        }
        self.entries = self.entries.map(|_, entry| Entry {
            dist: Distribution::Sequential,
            ..entry
        });
        self.work = Some(Work::new(axes, scope, n));
        self
    }

    /// The stages of this walk are filled by `n` planes of the cube, which take no tile of any
    /// level and do nothing else.
    ///
    /// Those `n` sit at the *end* of the cube, so no plane position below this level changes: the
    /// cube is `n` planes wider and nothing is renumbered. A ring built over this walk reads the
    /// count off the level and hands the two roles their own halves of each slot.
    ///
    /// Only a walk's regions are staged, so only a walk takes this.
    pub(crate) fn filled_by(mut self, n: usize) -> Level {
        assert!(
            self.scope == LevelScope::Sequential,
            "Level::filled_by: only a walk's regions are staged, and this level deals its tiles \
             to {:?}",
            self.scope
        );
        self.fillers = n;
        self
    }

    /// The order this cube level deals its boxes to the grid ([`CubeOrder`]).
    ///
    /// Only cubes take one: the order permutes which *instance* holds which box, and the
    /// grid is the one place instances are handed out by hardware position rather than by a
    /// loop the kernel writes.
    pub(crate) fn ordered(mut self, order: CubeOrder) -> Level {
        assert!(
            self.scope == LevelScope::Cubes,
            "Level::ordered: only a cube level is dealt to a grid, and this level deals its \
             tiles to {:?}",
            self.scope
        );
        self.order = order.canonicalize();
        self
    }

    fn push(&mut self, axis: Axis, entry: Entry) {
        assert!(
            !self.entries.contains(axis),
            "Level: {axis:?} is named twice; a level states each of its axes once"
        );
        let mut entries: Vec<_> = self
            .axes()
            .into_iter()
            .map(|a| (a, self.entries.get(a)))
            .collect();
        entries.push((axis, entry));
        self.entries = AxisMap::new(&entries);
    }

    /// The tile this level steps `axis` in, where it names it. An axis the level does not name
    /// has no tile of its own; its extent is the space's.
    pub fn tile(&self, axis: Axis) -> Option<usize> {
        self.entries
            .contains(axis)
            .then(|| self.entries.get(axis).tile)
    }

    /// How many tiles this level takes along `axis`, where it names it.
    pub fn count(&self, axis: Axis) -> Option<Count> {
        self.entries
            .contains(axis)
            .then(|| self.entries.get(axis).count)
    }

    /// The extent one region of this level covers along `axis` of `space`: its tile, or the
    /// space's own extent (static or not) where the axis is handed down whole.
    pub(crate) fn extent_in(&self, space: &Space, axis: Axis) -> Extent {
        match self.tile(axis) {
            Some(tile) => Extent::Static(tile),
            None => space.extent_raw(axis),
        }
    }

    /// Who takes `axis`'s tiles: [`Sequential`](Distribution::Sequential) where the level walks
    /// it or does not name it.
    pub fn distribution(&self, axis: Axis) -> Distribution {
        if self.entries.contains(axis) {
            self.entries.get(axis).dist
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
        (0..self.entries.len())
            .map(|i| self.entries.axis_at(i))
            .collect()
    }

    /// The axes this level distributes as one, if any.
    pub fn work(&self) -> Option<&Work> {
        self.work.as_ref()
    }

    /// Planes that fill this level's stages and do nothing else
    /// ([`filled_by`](Level::filled_by)).
    pub fn fillers(&self) -> usize {
        self.fillers
    }

    /// The order this level deals its boxes to the grid, which only a cube level ever states.
    pub fn order(&self) -> CubeOrder {
        self.order
    }

    /// The space one region of this level covers: every axis of `space` cut to its tile (static,
    /// a tile is comptime) or handed down whole. Position-free; the positions are the walk.
    pub fn child(&self, space: &Space) -> Space {
        Space::from_extents(
            &space
                .axes()
                .map(|axis| (axis, self.extent_in(space, axis)))
                .collect::<Vec<_>>(),
        )
    }

    /// Whether this level's tile on `axis` fails to divide the extent `space` hands it, leaving a
    /// partial tile that needs masking. Only [`Every`](Count::Every) or [`Across`](Count::Across)
    /// can: a stated count built the extent below it. Host-side, static extents.
    pub(crate) fn overhangs(&self, space: &Space, axis: Axis) -> bool {
        match self.tile(axis) {
            Some(tile) => !space.extent(axis).is_multiple_of(tile),
            None => false,
        }
    }

    /// Tiles along `axis` of `space` at this level: the stated count, or `ceil(extent / tile)`
    /// where the level takes every tile, so an indivisible axis gets a trailing partial tile
    /// (its overhang is masked at read/write). Host-side, static extents.
    pub(crate) fn tiles(&self, space: &Space, axis: Axis) -> usize {
        match self.count(axis) {
            None => 1,
            Some(count) => count.tiles(space.extent(axis), self.tile_of(axis)),
        }
    }

    /// Tiles along `axis` where the count is comptime: a stated count, or an every-level's
    /// over a static extent. `None` on a dynamic every-axis.
    pub(crate) fn tiles_const(&self, space: &Space, axis: Axis) -> Option<usize> {
        match self.count(axis) {
            None => Some(1),
            Some(count) => count.tiles_const(space.extent_raw(axis), self.tile_of(axis)),
        }
    }

    /// What a walk counts along `axis`: the stated count, `1` where the level does not name the
    /// axis, or the extent in this level's tile where it takes every tile.
    pub(crate) fn grid(&self, axis: Axis) -> GridCount {
        match self.count(axis) {
            None => GridCount::Const(1),
            Some(count) => count.grid(self.tile_of(axis)),
        }
    }

    fn tile_of(&self, axis: Axis) -> usize {
        self.entries.get(axis).tile
    }

    /// Whether every worker's run along an `axis` dealt across workers is the full one: the grid
    /// divides the worker count, provable only of a static extent. Any other count deals one tile
    /// a worker, which every grid divides. What lets the kernel skip clamping a run.
    pub(crate) fn divides(&self, space: &Space, axis: Axis) -> bool {
        match self.count(axis) {
            Some(Count::Across(workers)) => match space.extent_raw(axis) {
                Extent::Static(extent) => {
                    extent.div_ceil(self.tile_of(axis)).is_multiple_of(workers)
                }
                Extent::Dynamic => false,
            },
            _ => true,
        }
    }

    /// Whether this level cuts `axis` of `space` into a single, statically-known tile, so its
    /// walk coordinate is a constant `0`, even on a rolled walk. A `Dynamic` axis has no comptime
    /// count and is never statically single.
    pub(crate) fn single_static_tile(&self, space: &Space, axis: Axis) -> bool {
        match self.tile(axis) {
            Some(_) => self.tiles_const(space, axis) == Some(1),
            None => true,
        }
    }

    /// The `m × n` grid a partition level walks, read off `space`'s trailing two axes; leading
    /// (batch) axes must hand out one tile. Valid only on a [`Partition`](LevelRole::Partition)
    /// level; the role says whether it applies, this only reads the counts.
    pub(crate) fn partition_grid(&self, space: &Space) -> (usize, usize) {
        let edges = MatrixAxes::edges(space);
        for (p, axis) in space.axes().enumerate() {
            let tiles = self
                .tiles_const(space, axis)
                .expect("plane partition level: tile counts must be comptime");
            assert!(
                p == edges.row_split || p == edges.col_split || tiles == 1,
                "plane partition level: leading (batch) axes must hand out one tile"
            );
        }
        (
            self.tiles_const(space, space.axis_at(edges.row_split))
                .unwrap(),
            self.tiles_const(space, space.axis_at(edges.col_split))
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
    /// the walk steps (more than one tile) is absent from the operand, as in broadcast omission.
    /// A staged walk fills such an operand once, above the loop. Host-side, static extents.
    pub(crate) fn walk_invariant(&self, space: &Space, operand: &Space) -> bool {
        space
            .axes()
            .all(|axis| self.tiles(space, axis) == 1 || !operand.contains(axis))
    }

    /// How many workers `axis` is dealt out to at this level, where comptime: the stated count, or
    /// the tiles an every-level takes over a static extent. `None` where the grid is unknown here:
    /// a [`Dynamic`](Extent::Dynamic) extent, or `space` a projection dropping the axis (a drain).
    pub fn instances_along(&self, space: &Space, axis: Axis) -> Option<usize> {
        match self.count(axis) {
            None => Some(1),
            Some(count) => match count.stated() {
                Some(n) => Some(n),
                None if !space.contains(axis) => None,
                None => self.tiles_const(space, axis),
            },
        }
    }

    /// The instance-index weight `spanned`'s own axis list cannot see: the instance counts of the
    /// same-scope axes *inside* `axis` that this level distributes and `spanned` does not span.
    /// The odometer is the level's, so an operand divides out contracted axes it does not span.
    ///
    /// Reading omitted axes as weight `1` aliases the outer digits onto one value, so this panics
    /// where such an axis has no comptime count: assuming `1` would be exactly that aliasing.
    pub(crate) fn inner_weight_unspanned(&self, spanned: &Space, axis: Axis) -> usize {
        let scope = self.distribution(axis).scope();
        self.axes()
            .iter()
            .skip_while(|&&a| a != axis)
            .skip(1)
            .filter(|&&a| !spanned.contains(a) && self.distribution(a).scope() == scope)
            .map(|&a| {
                self.entries.get(a).count.stated().unwrap_or_else(|| {
                    panic!(
                        "Level::inner_weight_unspanned: {a:?} is distributed inside {axis:?} at \
                         the same scope but this operand does not span it, and its instance \
                         count is not comptime, so {axis:?}'s digit of the instance index \
                         cannot be decoded"
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
                ..
            } = self.distribution(axis)
            else {
                continue;
            };
            // Asserted, not skipped: a `Unit` axis always carries a stated count, and passing
            // over one whose count we could not read would shift every inner axis's bits by
            // its width.
            let lanes = self
                .entries
                .get(axis)
                .count
                .stated()
                .expect("Level::lane_share: a Unit axis must carry a stated count");
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

    /// What one instance of an operand spanning `spanned` holds of its cells after this level is
    /// dealt out over `space`: [`Partial`](SplitShare::Partial) where a `Plane` or `Cube` axis the
    /// operand does not span is dealt across several instances, so each contracts a slice.
    ///
    /// Asked with the level's whole space, not the operand's projection: a projection has dropped
    /// the contracted axis and so cannot tell a split from a cut whose edge is the whole axis.
    /// Conservative where the count is not comptime: whole would lose every partial but one.
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
            matches!(
                self.distribution(axis),
                Distribution::Spatial {
                    scope: ComputeScope::Unit,
                    ..
                }
            ) && self.entries.get(axis).count != Count::Of(1)
        })
    }
}

/// Several axes' work distributed as one.
///
/// Dealing each axis on its own gives an instance a box of the grid, the product of its per-axis
/// runs. Read as a single index instead, these axes give it a share of the whole: the shares no
/// box can describe are exactly the ones that balance a grid its shape cannot divide.
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
/// it. Set once, by the builder, so no consumer re-folds the per-axis distributions.
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
