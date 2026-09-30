//! One [`Level`]: which axes a loop steps, in tiles of what size, how many, and who takes them.

use super::CubeOrder;
use crate::{Axis, AxisMap, Extent, Space};

/// One decomposition level of a space: who takes its tiles and how it cuts each axis it names.
/// Unnamed axes are handed down whole.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Level {
    coverage: Coverage,
    cuts: AxisMap<Cut>,
    /// Axes a cube level hands one tile of, riding the grid's `Z` ([`batching`](Level::batching)).
    batches: Vec<Axis>,
    shared_by: Option<usize>,
    fillers: usize,
    order: CubeOrder,
}

/// One axis of a level: the tile it steps in, how many, and how its instances spread them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Cut {
    pub(crate) tile: usize,
    pub(crate) count: Count,
    pub(crate) spread: Spread,
}

/// How many tiles a level takes along one of its axes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Count {
    /// This many, stated: a walk's steps, or a scope's workers with one tile each.
    Stated(usize),
    /// Every tile the level above hands down; the one count whose tile can overhang.
    All,
    /// Every tile, distributed across this many workers in runs.
    AllAcross(usize),
    /// This many tiles, taken in turns by however many units the launch runs.
    Distributed(usize),
}

/// What a walk counts along one axis of a level: a constant, or the extent handed down.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GridCount {
    Const(usize),
    Extent(usize),
}

/// The hardware a level's tiles can go to, ordered inner to outer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum ComputeScope {
    Unit,
    Plane,
    Cube,
}

/// How a level covers its tiles.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Coverage {
    /// One instance steps the tiles in sequence.
    Walk,
    /// The tiles are distributed over the instances of a scope.
    Distribute(ComputeScope),
}

impl Coverage {
    /// The scope the tiles are distributed over, or `None` where one instance walks them.
    pub fn scope(self) -> Option<ComputeScope> {
        match self {
            Coverage::Walk => None,
            Coverage::Distribute(scope) => Some(scope),
        }
    }
}

/// How a distributed axis's tiles are handed to a scope's instances.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Spread {
    /// Instance `i` owns a contiguous run (cube 0 → `{0,1}`, cube 1 → `{2,3}`).
    #[default]
    Contiguous,
    /// Instances take turns (cube 0 → `{0,2}`, cube 1 → `{1,3}`).
    Interleaved,
}

/// One of the launch grid's three dimensions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CubeAxis {
    X,
    Y,
    Z,
}

impl Count {
    /// The workers the tiles are distributed to, where that is a stated number.
    pub(crate) fn stated(self) -> Option<usize> {
        match self {
            Count::Stated(n) | Count::AllAcross(n) => Some(n),
            Count::All | Count::Distributed(_) => None,
        }
    }

    /// Tiles this count takes along an axis of `extent`; a partial last tile counts.
    pub(crate) fn tiles(self, extent: usize, tile: usize) -> usize {
        match self {
            Count::Stated(n) | Count::Distributed(n) => n,
            Count::All | Count::AllAcross(_) => extent.div_ceil(tile),
        }
    }

    /// [`tiles`](Count::tiles) where the host can prove it.
    pub(crate) fn tiles_const(self, extent: Extent, tile: usize) -> Option<usize> {
        match (self, extent) {
            (Count::Stated(n) | Count::Distributed(n), _) => Some(n),
            (_, Extent::Static(extent)) => Some(self.tiles(extent, tile)),
            (_, Extent::Dynamic) => None,
        }
    }

    /// What a walk counts along an axis stepped in `tile`.
    pub(crate) fn grid(self, tile: usize) -> GridCount {
        match self {
            Count::Stated(n) | Count::Distributed(n) => GridCount::Const(n),
            Count::All | Count::AllAcross(_) => GridCount::Extent(tile),
        }
    }
}

impl Level {
    /// The walk level over a region in tiles of `tile`, taking every one of them.
    pub fn every(tile: &[(Axis, usize)]) -> Level {
        let cuts: Vec<_> = tile
            .iter()
            .map(|&(axis, tile)| (axis, tile, Count::All, Spread::Contiguous))
            .collect();
        Level::new(Coverage::Walk, &cuts)
    }

    /// A level covered as `coverage` says over `cuts`, each `(axis, tile, count, spread)`.
    pub(crate) fn new(coverage: Coverage, cuts: &[(Axis, usize, Count, Spread)]) -> Level {
        for (i, &(axis, ..)) in cuts.iter().enumerate() {
            assert!(
                !cuts[..i].iter().any(|&(a, ..)| a == axis),
                "Level: {axis:?} is named twice; a level states each of its axes once"
            );
        }
        for &(axis, tile, count, _) in cuts {
            assert!(tile > 0, "Level: {axis:?} has a tile of nothing");
            match (coverage, count) {
                (Coverage::Distribute(ComputeScope::Cube), Count::AllAcross(_)) => {}
                (_, Count::AllAcross(_)) => {
                    panic!(
                        "Level: {axis:?} is distributed across workers in runs, which only cubes take"
                    )
                }
                (
                    Coverage::Distribute(ComputeScope::Unit)
                    | Coverage::Distribute(ComputeScope::Plane),
                    Count::All,
                ) => panic!(
                    "Level: {axis:?} takes every tile on a plane's workers, whose count is the \
                     device's; state how many"
                ),
                (Coverage::Distribute(ComputeScope::Unit), Count::Distributed(_)) => assert!(
                    matches!(
                        cuts.iter().find(|&&(a, ..)| a == axis).map(|&(.., s)| s),
                        Some(Spread::Interleaved)
                    ),
                    "Level: {axis:?} is distributed in turns to however many units the launch runs, so \
                     the units take it interleaved"
                ),
                (_, Count::Distributed(_)) => panic!(
                    "Level: {axis:?} is distributed in turns to however many units the launch runs, \
                     which only a plane's units can be"
                ),
                _ => {}
            }
        }
        let distributed = cuts
            .iter()
            .any(|&(_, _, count, _)| matches!(count, Count::Distributed(_)));
        assert!(
            !distributed
                || (coverage == Coverage::Distribute(ComputeScope::Unit) && cuts.len() == 1),
            "Level: an axis distributed to however many units the launch runs takes all of them, so \
             the level distributes no other axis to the units; it distributes {}",
            cuts.len()
        );
        let cuts: Vec<_> = cuts
            .iter()
            .map(|&(axis, tile, count, spread)| {
                (
                    axis,
                    Cut {
                        tile,
                        count,
                        spread,
                    },
                )
            })
            .collect();
        Level {
            coverage,
            cuts: AxisMap::new(&cuts),
            batches: Vec::new(),
            shared_by: None,
            fillers: 0,
            order: CubeOrder::RowMajor,
        }
    }

    /// One tile of each of `axes` per cube, on the grid's `Z` dimension.
    pub(crate) fn batching(mut self, axes: &[Axis]) -> Level {
        assert!(
            self.coverage == Coverage::Distribute(ComputeScope::Cube),
            "Level::batching: batches ride cubes; this level's tiles are taken by {:?}",
            self.coverage
        );
        for &axis in axes {
            self.push(
                axis,
                Cut {
                    tile: 1,
                    count: Count::All,
                    spread: Spread::Contiguous,
                },
            );
            self.batches.push(axis);
        }
        self
    }

    /// Every cut's tiles read as one index, of which each of `n` workers takes a run.
    pub(crate) fn sharing(mut self, n: usize) -> Level {
        match self.coverage {
            Coverage::Distribute(ComputeScope::Cube)
            | Coverage::Distribute(ComputeScope::Plane) => {}
            Coverage::Distribute(ComputeScope::Unit) => panic!(
                "Level::sharing: the plane's units combine in registers, which needs them in \
                 lockstep, and units holding different shares never are"
            ),
            Coverage::Walk => panic!("Level::sharing: a walk has no workers to share its tiles"),
        }
        assert!(
            self.shared_by.is_none(),
            "Level::sharing: this level already cuts its tiles as one"
        );
        for axis in self.axes() {
            let cut = self.cuts.get(axis);
            assert!(
                cut.count == Count::All && cut.spread == Spread::Contiguous,
                "Level::sharing: {axis:?} states a count or a spread of its own, which a share \
                 of the whole has no use for"
            );
        }
        self.shared_by = Some(n);
        self
    }

    /// The stages of this walk are filled by `n` extra planes at the end of the cube.
    pub(crate) fn filling(mut self, n: usize) -> Level {
        assert!(
            self.coverage == Coverage::Walk,
            "Level::filling: only a walk's regions are staged, and this level's tiles are taken \
             by {:?}",
            self.coverage
        );
        self.fillers = n;
        self
    }

    /// The order this cube level distributes its boxes to the grid.
    pub(crate) fn distributed_in(mut self, order: CubeOrder) -> Level {
        assert!(
            self.coverage == Coverage::Distribute(ComputeScope::Cube),
            "Level::distributed_in: only a cube level is distributed to a grid, and this level's tiles are \
             taken by {:?}",
            self.coverage
        );
        self.order = order.canonicalize();
        self
    }

    fn push(&mut self, axis: Axis, cut: Cut) {
        assert!(
            !self.cuts.contains(axis),
            "Level: {axis:?} is named twice; a level states each of its axes once"
        );
        let mut cuts: Vec<_> = self
            .axes()
            .into_iter()
            .map(|a| (a, self.cuts.get(a)))
            .collect();
        cuts.push((axis, cut));
        self.cuts = AxisMap::new(&cuts);
    }

    /// Who takes this level's tiles.
    pub fn coverage(&self) -> Coverage {
        self.coverage
    }

    /// The axes this level names, in the order it named them.
    pub fn axes(&self) -> Vec<Axis> {
        (0..self.cuts.len()).map(|i| self.cuts.axis_at(i)).collect()
    }

    /// The tile this level steps `axis` in, where it names it.
    pub fn tile(&self, axis: Axis) -> Option<usize> {
        self.cuts.contains(axis).then(|| self.cuts.get(axis).tile)
    }

    /// How many tiles this level takes along `axis`, where it names it.
    pub fn count(&self, axis: Axis) -> Option<Count> {
        self.cuts.contains(axis).then(|| self.cuts.get(axis).count)
    }

    /// How the instances spread `axis`'s tiles, where the level names it.
    pub fn spread(&self, axis: Axis) -> Option<Spread> {
        self.cuts.contains(axis).then(|| self.cuts.get(axis).spread)
    }

    pub(crate) fn cut(&self, axis: Axis) -> Cut {
        self.cuts.get(axis)
    }

    /// Whether this level distributes `axis`'s tiles over its scope one by one.
    pub fn distributes(&self, axis: Axis) -> bool {
        self.coverage != Coverage::Walk && self.shared_by.is_none() && self.cuts.contains(axis)
    }

    /// The grid dimension `axis` rides on a cube level: batches on `Z`, the others `X`, `Y`, `Z`
    /// in the order named.
    pub(crate) fn cube_axis(&self, axis: Axis) -> Option<CubeAxis> {
        if self.coverage != Coverage::Distribute(ComputeScope::Cube) || !self.cuts.contains(axis) {
            return None;
        }
        if self.batches.contains(&axis) {
            return Some(CubeAxis::Z);
        }
        let dims = [CubeAxis::X, CubeAxis::Y, CubeAxis::Z];
        let position = self
            .axes()
            .into_iter()
            .filter(|a| !self.batches.contains(a))
            .position(|a| a == axis)
            .expect("the axis is named and not a batch");
        Some(dims[position])
    }

    /// The workers taking this level's grid as one flat index.
    pub fn shared_by(&self) -> Option<usize> {
        self.shared_by
    }

    /// Planes that fill this level's stages.
    pub fn fillers(&self) -> usize {
        self.fillers
    }

    /// The order this level distributes its boxes to the grid.
    pub fn order(&self) -> CubeOrder {
        self.order
    }

    /// The extent one region of this level covers along `axis` of `space`.
    pub(crate) fn extent_in(&self, space: &Space, axis: Axis) -> Extent {
        match self.tile(axis) {
            Some(tile) => Extent::Static(tile),
            None => space.extent_raw(axis),
        }
    }

    /// The space one region of this level covers.
    pub fn child(&self, space: &Space) -> Space {
        Space::from_extents(
            &space
                .axes()
                .map(|axis| (axis, self.extent_in(space, axis)))
                .collect::<Vec<_>>(),
        )
    }

    /// Whether this level's tile on `axis` fails to divide the extent `space` hands it.
    /// Static extents only.
    pub(crate) fn overhangs(&self, space: &Space, axis: Axis) -> bool {
        match self.tile(axis) {
            Some(tile) => !space.extent(axis).is_multiple_of(tile),
            None => false,
        }
    }

    /// Tiles along `axis` of `space` at this level, a partial tile counted; static extents only.
    pub(crate) fn tiles(&self, space: &Space, axis: Axis) -> usize {
        match self.count(axis) {
            None => 1,
            Some(count) => count.tiles(space.extent(axis), self.cut(axis).tile),
        }
    }

    /// Tiles along `axis` where the count is comptime.
    pub(crate) fn tiles_const(&self, space: &Space, axis: Axis) -> Option<usize> {
        match self.count(axis) {
            None => Some(1),
            Some(count) => count.tiles_const(space.extent_raw(axis), self.cut(axis).tile),
        }
    }

    /// What a walk counts along `axis`.
    pub(crate) fn grid(&self, axis: Axis) -> GridCount {
        match self.count(axis) {
            None => GridCount::Const(1),
            Some(count) => count.grid(self.cut(axis).tile),
        }
    }

    /// How many workers `axis` is distributed out to at this level, where comptime.
    pub(crate) fn instances_along(&self, space: &Space, axis: Axis) -> Option<usize> {
        match self.count(axis) {
            None => Some(1),
            Some(Count::Distributed(_)) => None,
            Some(count) => match count.stated() {
                Some(n) => Some(n),
                None if !space.contains(axis) => None,
                None => self.tiles_const(space, axis),
            },
        }
    }
}
