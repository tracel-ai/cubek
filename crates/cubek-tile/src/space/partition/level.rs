//! One [`Level`]: which axes a loop steps, in tiles of what size, how many, and who takes them.
//! One entry of a [`Partitioning`](crate::Partitioning): a [`Region`](crate::Region) hands it to
//! `at`, a [`Launcher`](crate::Launcher) sizes its grid from it, so grid and loops cannot disagree.
//!
//! Built only by [`Levels`](crate::Levels), leaf up: a level's tile on an axis is the product of
//! what was stated below it, and its [`Count`] is what it stated. A level names only the axes it
//! touches; the rest pass down whole. [`Level::every`], a walk over a region, is the one level
//! built directly.
//!
//! A level states; it does not answer for its consumers. What the plane's lanes are to an
//! operand's cells is [`LaneShare::new`](crate::LaneShare), what one instance holds of them
//! [`SplitShare::new`](crate::SplitShare), how a walk deals an axis `AxisDeal::new`: each asker
//! reads the statement and derives its own answer.

use super::CubeOrder;
use crate::{Axis, AxisMap, Extent, Space};

/// One decomposition level of a space: who takes its tiles, the axes it names (each cut to a tile,
/// a count and a spread), and the cube level's statements (its batch axes, whether its grid is
/// shared as one index, the order it is dealt in). An axis it does not name is handed down whole.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Level {
    takers: Takers,
    cuts: AxisMap<Cut>,
    /// Axes a cube level hands one tile of, riding the grid's `Z` ([`batching`](Level::batching)).
    batches: Vec<Axis>,
    /// Cubes taking this level's whole grid as one flat index ([`sharing`](Level::sharing)),
    /// each a run of it rather than a box.
    shared_by: Option<usize>,
    /// Planes that fill this walk's stages and take no tile of any level
    /// ([`filling`](Level::filling)).
    fillers: usize,
    /// The order a cube level deals its boxes to the grid ([`dealt_in`](Level::dealt_in)).
    /// [`RowMajor`](CubeOrder::RowMajor) everywhere else, which is what every level that
    /// states nothing gets.
    order: CubeOrder,
}

/// One axis of a level: the tile it steps in, how many, and how the takers spread them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Cut {
    pub(crate) tile: usize,
    pub(crate) count: Count,
    pub(crate) spread: Spread,
}

/// How many tiles a level takes along one of its axes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Count {
    /// This many, stated: a walk's steps, or a scope's workers with one tile each. A count
    /// cannot fail to divide, and a tile under it cannot overhang.
    Stated(usize),
    /// Every tile the level above hands down, one per step or per worker: the count only the
    /// launch knows, and the one place a tile can reach past the extent.
    All,
    /// Every tile, dealt across this many workers in runs: a closed axis returning to the cube
    /// level to be split, and the one run whose length the kernel computes.
    AllAcross(usize),
}

/// What a walk counts along one axis of a level: a constant, or the extent handed down in
/// this tile ([`Level::grid`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GridCount {
    Const(usize),
    Extent(usize),
}

/// Who takes a level's tiles: the loop verb that states it. One instance walks a level's grid
/// whole; the plane's lanes, the cube's planes or the launch's cubes each take a share of it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Takers {
    Walk,
    Lanes,
    Planes,
    Cubes,
}

/// How a dealt axis's tiles are handed to its takers. Disjoint either way, differing only in
/// locality.
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
pub enum CubeAxis {
    X,
    Y,
    Z,
}

impl Count {
    /// The count where it is a stated number: `Stated` and `AllAcross` are, `All` is the launch's.
    pub(crate) fn stated(self) -> Option<usize> {
        match self {
            Count::Stated(n) | Count::AllAcross(n) => Some(n),
            Count::All => None,
        }
    }

    /// Tiles this count takes along an axis of `extent`, in tiles of `tile`: the stated number,
    /// or every tile the extent holds, the last one partial where it does not divide.
    pub(crate) fn tiles(self, extent: usize, tile: usize) -> usize {
        match self {
            Count::Stated(n) => n,
            Count::All | Count::AllAcross(_) => extent.div_ceil(tile),
        }
    }

    /// [`tiles`](Count::tiles) where the host can prove it: a stated count needs no extent, and
    /// every tile of a [`Dynamic`](Extent::Dynamic) axis is the launch's to count.
    pub(crate) fn tiles_const(self, extent: Extent, tile: usize) -> Option<usize> {
        match (self, extent) {
            (Count::Stated(n), _) => Some(n),
            (_, Extent::Static(extent)) => Some(self.tiles(extent, tile)),
            (_, Extent::Dynamic) => None,
        }
    }

    /// What a walk counts along an axis stepped in `tile`: the stated number, or the extent it
    /// is handed, in that tile.
    pub(crate) fn grid(self, tile: usize) -> GridCount {
        match self {
            Count::Stated(n) => GridCount::Const(n),
            Count::All | Count::AllAcross(_) => GridCount::Extent(tile),
        }
    }
}

impl Level {
    /// The one level that walks a region in tiles of `tile`, taking every one of them: what
    /// [`Region::over`](crate::Region::over) asks for, and the shape a stated count never has.
    pub fn every(tile: &[(Axis, usize)]) -> Level {
        let cuts: Vec<_> = tile
            .iter()
            .map(|&(axis, tile)| (axis, tile, Count::All, Spread::Contiguous))
            .collect();
        Level::new(Takers::Walk, &cuts)
    }

    /// A level `takers` take over `cuts`, each `(axis, tile, count, spread)`. The builder's
    /// constructor: [`Levels`](crate::Levels) is the only other caller, and it states the tile
    /// as the product of the levels below.
    ///
    /// What each taker can state: a walk steps a stated count or every tile; lanes and planes
    /// take a stated count each, since their number is the device's; cubes take every tile, or
    /// every tile dealt across a stated number of them.
    pub(crate) fn new(takers: Takers, cuts: &[(Axis, usize, Count, Spread)]) -> Level {
        for (i, &(axis, ..)) in cuts.iter().enumerate() {
            assert!(
                !cuts[..i].iter().any(|&(a, ..)| a == axis),
                "Level: {axis:?} is named twice; a level states each of its axes once"
            );
        }
        for &(axis, tile, count, _) in cuts {
            assert!(tile > 0, "Level: {axis:?} has a tile of nothing");
            match (takers, count) {
                (Takers::Cubes, Count::AllAcross(_)) => {}
                (_, Count::AllAcross(_)) => {
                    panic!("Level: {axis:?} is dealt across workers in runs, which only cubes take")
                }
                (Takers::Lanes | Takers::Planes, Count::All) => panic!(
                    "Level: {axis:?} takes every tile on a plane's workers, whose count is the \
                     device's; state how many"
                ),
                _ => {}
            }
        }
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
            takers,
            cuts: AxisMap::new(&cuts),
            batches: Vec::new(),
            shared_by: None,
            fillers: 0,
            order: CubeOrder::RowMajor,
        }
    }

    /// One tile of each of `axes` per cube, on the grid's `Z` dimension: the batch axes of a
    /// cube level.
    pub(crate) fn batching(mut self, axes: &[Axis]) -> Level {
        assert!(
            self.takers == Takers::Cubes,
            "Level::batching: batches ride cubes; this level's tiles are taken by {:?}",
            self.takers
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

    /// Every cut's tiles read as one index, of which each of `n` workers takes a share: a run
    /// of the whole, not a box, which balances a grid its shape cannot divide. It also spans a
    /// region of the level below, so a [`Walk::range`](crate::Walk::range) can end mid-region.
    ///
    /// Not for lanes: they combine in registers, which needs them in lockstep, and lanes holding
    /// different shares never are.
    pub(crate) fn sharing(mut self, n: usize) -> Level {
        match self.takers {
            Takers::Cubes | Takers::Planes => {}
            Takers::Lanes => panic!(
                "Level::sharing: the plane's lanes combine in registers, which needs them in \
                 lockstep, and lanes holding different shares never are"
            ),
            Takers::Walk => panic!("Level::sharing: a walk has no workers to share its tiles"),
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

    /// The stages of this walk are filled by `n` planes of the cube, which take no tile of any
    /// level and do nothing else.
    ///
    /// Those `n` sit at the *end* of the cube, so no plane position below this level changes: the
    /// cube is `n` planes wider and nothing is renumbered. Stages built over this walk reads the
    /// count off the level and hands the two roles their own halves of each slot.
    pub(crate) fn filling(mut self, n: usize) -> Level {
        assert!(
            self.takers == Takers::Walk,
            "Level::filling: only a walk's regions are staged, and this level's tiles are taken \
             by {:?}",
            self.takers
        );
        self.fillers = n;
        self
    }

    /// The order this cube level deals its boxes to the grid ([`CubeOrder`]). Only cubes take
    /// one: the order permutes which *instance* holds which box, and the grid is the one place
    /// instances are handed out by hardware position rather than by a loop the kernel writes.
    pub(crate) fn dealt_in(mut self, order: CubeOrder) -> Level {
        assert!(
            self.takers == Takers::Cubes,
            "Level::dealt_in: only a cube level is dealt to a grid, and this level's tiles are \
             taken by {:?}",
            self.takers
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
    pub fn takers(&self) -> Takers {
        self.takers
    }

    /// The axes this level names, in the order it named them.
    pub fn axes(&self) -> Vec<Axis> {
        (0..self.cuts.len()).map(|i| self.cuts.axis_at(i)).collect()
    }

    /// The tile this level steps `axis` in, where it names it. An axis the level does not name
    /// has no tile of its own; its extent is the space's.
    pub fn tile(&self, axis: Axis) -> Option<usize> {
        self.cuts.contains(axis).then(|| self.cuts.get(axis).tile)
    }

    /// How many tiles this level takes along `axis`, where it names it.
    pub fn count(&self, axis: Axis) -> Option<Count> {
        self.cuts.contains(axis).then(|| self.cuts.get(axis).count)
    }

    /// How the takers spread `axis`'s tiles among themselves, where the level names it.
    pub fn spread(&self, axis: Axis) -> Option<Spread> {
        self.cuts.contains(axis).then(|| self.cuts.get(axis).spread)
    }

    /// The cut of an axis this level names.
    pub(crate) fn cut(&self, axis: Axis) -> Cut {
        self.cuts.get(axis)
    }

    /// Whether this level deals `axis`'s tiles to its takers one by one: it names the axis, its
    /// takers are hardware, and its grid is not shared as one index. A walk deals nothing; a
    /// shared level deals its whole index rather than any one axis.
    pub fn deals(&self, axis: Axis) -> bool {
        self.takers != Takers::Walk && self.shared_by.is_none() && self.cuts.contains(axis)
    }

    /// The grid dimension `axis` rides on a cube level: batches on `Z`, the others `X`, `Y`, `Z`
    /// in the order named. `None` on any other level, or an axis the level does not name.
    pub fn cube_axis(&self, axis: Axis) -> Option<CubeAxis> {
        if self.takers != Takers::Cubes || !self.cuts.contains(axis) {
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

    /// The workers taking this level's grid as one flat index ([`sharing`](Level::sharing)),
    /// `None` where each takes a box of it.
    pub fn shared_by(&self) -> Option<usize> {
        self.shared_by
    }

    /// Planes that fill this level's stages and do nothing else ([`filling`](Level::filling)).
    pub fn fillers(&self) -> usize {
        self.fillers
    }

    /// The order this level deals its boxes to the grid, which only a cube level ever states.
    pub fn order(&self) -> CubeOrder {
        self.order
    }

    /// The extent one region of this level covers along `axis` of `space`: its tile, or the
    /// space's own extent (static or not) where the axis is handed down whole.
    pub(crate) fn extent_in(&self, space: &Space, axis: Axis) -> Extent {
        match self.tile(axis) {
            Some(tile) => Extent::Static(tile),
            None => space.extent_raw(axis),
        }
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
    /// partial tile that needs masking. Only [`All`](Count::All) or [`AllAcross`](Count::AllAcross)
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
            Some(count) => count.tiles(space.extent(axis), self.cut(axis).tile),
        }
    }

    /// Tiles along `axis` where the count is comptime: a stated count, or an every-level's
    /// over a static extent. `None` on a dynamic every-axis.
    pub(crate) fn tiles_const(&self, space: &Space, axis: Axis) -> Option<usize> {
        match self.count(axis) {
            None => Some(1),
            Some(count) => count.tiles_const(space.extent_raw(axis), self.cut(axis).tile),
        }
    }

    /// What a walk counts along `axis`: the stated count, `1` where the level does not name the
    /// axis, or the extent in this level's tile where it takes every tile.
    pub(crate) fn grid(&self, axis: Axis) -> GridCount {
        match self.count(axis) {
            None => GridCount::Const(1),
            Some(count) => count.grid(self.cut(axis).tile),
        }
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
}
