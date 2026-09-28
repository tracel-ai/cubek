//! [`Levels`]: stating a partitioning's levels from the leaf up, in counts.
//!
//! A [`Level`] steps an axis in tiles of some size, which a caller almost always knows as a
//! product: the instruction is `16x8x16`, a plane holds `2x4` of them, a cube `2x4` planes. As
//! sizes (`16x8`, `32x32`, `64x128`) the reader divides to relate them; here it is stated once.
//!
//! ```ignore
//! Levels::leaf(&[(M, 16), (N, 8), (K, 16)])   // the instruction: the device's, not a choice
//!     .walk(&[(M, 2), (N, 4)])                // fragments a plane holds
//!     .walk(&[(K, 2)])                        // instruction steps in one stage
//!     .planes(&[(M, 2), (N, 4)])              // planes a cube holds
//!     .walk_every(&[K])                       // every stage `k` holds
//!     .cubes(&[M, N]).batches(&[B])           // every box `m x n` holds
//!     .build()
//! ```
//!
//! **The leaf is not a level.** It is the size the first level above it states as its tile, which
//! is why the instruction disappears from the list and becomes the first line instead.
//!
//! **Counts, and "all of it".** Every level says how many of the thing below it
//! ([`Count::Stated`]); units that cooperate on nothing may take their count in turns, however
//! many the launch runs ([`units_distributed`](Levels::units_distributed), [`Count::Distributed`]), which is still
//! a stated count and builds its tile the same way. The exception takes *every* tile of an axis
//! (a reduction's `K`, the output's boxes, the batch), a count nobody knows until launch
//! ([`Count::All`]).
//!
//! That is [`walk_every`](Levels::walk_every), [`cubes`](Levels::cubes) and
//! [`batches`](Levels::batches), which name axes and no number, and what they settle is this:
//!
//! * "every" means every tile **the level above hands down**: the whole axis only at the outermost
//!   level that names it; under a cube level that distributes the axis across cubes, the cube's run.
//!
//! * It closes the axis. Nothing above may count its tiles, because nothing above knows how many
//!   there are; a walk above it would have nothing to step through.
//!
//! * It returns exactly once, on the cube level, and only to be distributed across cubes
//!   ([`across`](Levels::across), [`Count::AllAcross`]): the split of a contraction is the walk
//!   below taking every stage of the run the grid hands its cube.
//!
//! * An axis a level does not name is not "all of it"; it is not that level's. The region is
//!   handed down whole, unstepped, and the [`Level`] has no tile for it.
//!
//! So the only division left in a launch is an every-level's tile against the extent it is
//! handed, taken once, ragged at the end; between two stated levels there is nothing to divide,
//! and a shape that does not divide cannot be written.
//!
//! **One reversal, named.** Levels are stated innermost first and a kernel's loops run outermost
//! first, so [`build`](Levels::build) reverses. That is the only place the two directions meet.

use super::CubeOrder;
use crate::{Axis, ComputeScope, Count, Coverage, Level, Spread};

/// One level, as the builder holds it before it is a [`Level`]: the axes it names with the tile
/// each is built of, and the modifiers stated after it.
#[derive(Clone, Debug)]
struct StatedLevel {
    /// Who takes this level's tiles, which is also the loop verb that states it.
    coverage: Coverage,
    /// `(axis, the size one tile of this level covers, how many)` — the running size below it,
    /// and the count stated. An every-level states no count and `takes` says so.
    tiles: Vec<(Axis, usize, usize)>,
    /// How this level takes the tiles below it.
    takes: Takes,
    /// A cube level distributing one of its axes across several cubes ([`Levels::across`]): the axis
    /// and how many.
    across: Option<(Axis, usize)>,
    /// A cube level distributing its tiles as one index ([`Levels::shared_by`]).
    shared_by: Option<usize>,
    /// Planes that only fill this walk's stages ([`Levels::filled_by`]).
    fillers: usize,
    /// Batch axes this cube level hands out one of ([`Levels::batches`]).
    batches: Vec<Axis>,
    /// Axes whose tiles the workers take in turns rather than in runs ([`Levels::interleaved`]).
    interleaved: Vec<Axis>,
    /// Axes a level below already took whole, which this cube level names again — legal only to
    /// distribute them across cubes, checked when the levels are built.
    reopened: Vec<Axis>,
    /// The order a cube level distributes its boxes to the grid ([`Levels::ordered`]).
    order: CubeOrder,
}

/// How a stated level takes the tiles below it: which [`Count`] its entries carry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Takes {
    /// The count stated, one tile a worker or a step ([`Count::Stated`]).
    Stated,
    /// Every tile the level above hands down ([`Count::All`]).
    Every,
    /// The count stated, taken in turns by as many units as the launch runs ([`Count::Distributed`]).
    DistributedToUnits,
}

/// A partitioning stated from the leaf up. See the module docs.
#[derive(Clone, Debug)]
pub struct Levels {
    /// What one tile of each axis covers so far, leaf up. An axis nobody built covers one.
    sizes: Vec<(Axis, usize)>,
    /// Axes an `every` level finished, which nothing above may multiply again.
    closed: Vec<Axis>,
    /// Innermost first; [`levels`](Self::levels) reverses.
    stated: Vec<StatedLevel>,
}

impl Levels {
    /// The tile one worker holds at the bottom — an instruction's shape, a register block, a
    /// unit's vector. The only sizes in a partitioning, and the only ones a device dictates.
    pub fn leaf(tile: &[(Axis, usize)]) -> Self {
        Levels {
            sizes: tile.to_vec(),
            closed: Vec::new(),
            stated: Vec::new(),
        }
    }

    /// Every worker steps through this many of the thing below, one at a time.
    pub fn walk(self, counts: &[(Axis, usize)]) -> Self {
        self.state(Coverage::Walk, counts)
    }

    /// Every worker steps through as many of the thing below as the axis holds — the count the
    /// levels do not know, and the launch does.
    pub fn walk_every(self, axes: &[Axis]) -> Self {
        self.every(Coverage::Walk, axes)
    }

    /// This many of the thing below, one per unit of the plane.
    pub fn units(self, counts: &[(Axis, usize)]) -> Self {
        self.state(Coverage::Distribute(ComputeScope::Unit), counts)
    }

    /// This many of the thing below, taken in turns by the plane's units, however many the
    /// launch runs ([`Count::Distributed`]): what units that cooperate on nothing take, so the plane
    /// width is read by the kernel rather than compiled into it. One axis, since the units are
    /// all its own.
    pub fn units_distributed(mut self, axis: Axis, count: usize) -> Self {
        let tiles = vec![(axis, self.size(axis), count)];
        let mut stated = StatedLevel::new(
            Coverage::Distribute(ComputeScope::Unit),
            tiles,
            Takes::DistributedToUnits,
        );
        stated.interleaved.push(axis);
        self.stated.push(stated);
        self.grow(axis, count);
        self
    }

    /// This many of the thing below, one per plane of the cube.
    pub fn planes(self, counts: &[(Axis, usize)]) -> Self {
        self.state(Coverage::Distribute(ComputeScope::Plane), counts)
    }

    /// One cube for every box these axes hold. The outermost level of a launch, and the one whose
    /// count is the problem's rather than the plan's. The entries ride the grid's dimensions in
    /// order: the first `X`, the second `Y`, the third `Z`.
    ///
    /// The two ways several cubes share a box are stated after it:
    /// [`across`](Self::across) cuts one axis among them, [`shared_by`](Self::shared_by) distributes
    /// this level's tiles to them as one index.
    pub fn cubes(self, axes: &[Axis]) -> Self {
        assert!(
            axes.len() <= 3,
            "Levels::cubes: {} axes, but a launch grid has three dimensions",
            axes.len()
        );
        self.every(Coverage::Distribute(ComputeScope::Cube), axes)
    }

    /// Distribute `axis` — one the cube level just stated — across `cubes` of them, each taking a run
    /// of its tiles. Split-K, where the axis is the contraction. The axis keeps its place among
    /// the level's entries, and so its grid dimension.
    pub fn across(mut self, axis: Axis, cubes: usize) -> Self {
        let stated = self.last("across");
        assert!(
            stated.coverage == Coverage::Distribute(ComputeScope::Cube),
            "Levels::across: only cubes take an axis in runs; the level just stated is {:?}",
            stated.coverage
        );
        assert!(
            stated.tiles.iter().any(|&(a, _, _)| a == axis),
            "Levels::across: {axis:?} is not an axis of the cube level just stated; name it there"
        );
        stated.across = Some((axis, cubes));
        self
    }

    /// Distribute this level's tiles to `cubes` of them **as one index**, so a cube's share is a run of
    /// the whole rather than a box of it. Stream-K.
    pub fn shared_by(mut self, cubes: usize) -> Self {
        self.last("shared_by").shared_by = Some(cubes);
        self
    }

    /// The order the cube level just stated distributes its boxes to the grid ([`CubeOrder`]).
    ///
    /// The grid's own order unless this is said, so every tiling that says nothing is the
    /// tiling it was.
    pub fn ordered(mut self, order: CubeOrder) -> Self {
        let stated = self.last("ordered");
        assert!(
            stated.coverage == Coverage::Distribute(ComputeScope::Cube),
            "Levels::ordered: only cubes are distributed to a grid; the level just stated is {:?}",
            stated.coverage
        );
        stated.order = order.canonicalize();
        self
    }

    /// This walk's stages are filled by `n` planes of the cube, which take no tile of any level.
    pub fn filled_by(mut self, n: usize) -> Self {
        self.last("filled_by").fillers = n;
        self
    }

    /// Batch axes this cube level hands out one of, however they are listed: one call or several.
    pub fn batches(mut self, axes: &[Axis]) -> Self {
        self.last("batches").batches.extend_from_slice(axes);
        self
    }

    /// The workers of the level just stated take `axis`'s tiles in turns (worker 0 the first,
    /// worker 1 the next) rather than each a contiguous run, so neighbouring units read
    /// neighbouring memory at the same instant, as a unit split of a contiguous contraction wants.
    pub fn interleaved(mut self, axis: Axis) -> Self {
        self.last("interleaved").interleaved.push(axis);
        self
    }

    /// The levels, **outermost first**: what a kernel's loops walk and a launch reads its grid
    /// off. The one place the two directions meet.
    pub fn build(self) -> Vec<Level> {
        for stated in &self.stated {
            for &axis in &stated.reopened {
                assert!(
                    matches!(stated.across, Some((a, _)) if a == axis),
                    "Levels::cubes: {axis:?} was taken whole by a level below; a cube level names \
                     it again only to distribute it across cubes (`.across({axis:?}, n)`)"
                );
            }
        }
        self.stated.iter().rev().map(StatedLevel::level).collect()
    }

    /// The one level of a tiling that states one: what a kernel walks a region with
    /// ([`Region::over`](crate::Region::over)), stated as the leaf it walks in and whether it takes
    /// every tile of it. A tiling of several levels has no one level to answer with.
    pub fn level(self) -> Level {
        assert!(
            self.stated.len() == 1,
            "Levels::level: {} levels are stated; a walk over a region takes exactly one",
            self.stated.len()
        );
        self.build().remove(0)
    }

    /// State a level of `counts` of the thing below, then grow each axis by its count.
    fn state(mut self, coverage: Coverage, counts: &[(Axis, usize)]) -> Self {
        let tiles = counts
            .iter()
            .map(|&(axis, count)| (axis, self.size(axis), count));
        self.stated
            .push(StatedLevel::new(coverage, tiles.collect(), Takes::Stated));
        for &(axis, count) in counts {
            self.grow(axis, count);
        }
        self
    }

    /// State a level covering every tile these axes hold, and close them: a count nothing above
    /// can multiply, because nothing above knows it.
    fn every(mut self, coverage: Coverage, axes: &[Axis]) -> Self {
        let reopened: Vec<Axis> = axes
            .iter()
            .copied()
            .filter(|axis| self.closed.contains(axis))
            .collect();
        if let (Coverage::Walk, Some(axis)) = (coverage, reopened.first()) {
            panic!(
                "Levels::walk_every: {axis:?} was taken whole by a level below, so a walk above \
                 it has nothing to step through"
            );
        }
        let tiles = axes.iter().map(|&axis| (axis, self.size(axis), 1));
        let mut stated = StatedLevel::new(coverage, tiles.collect(), Takes::Every);
        stated.reopened = reopened;
        self.stated.push(stated);
        self.closed.extend_from_slice(axes);
        self
    }

    /// What one tile of `axis` covers so far. An axis nobody built covers one.
    fn size(&self, axis: Axis) -> usize {
        self.sizes
            .iter()
            .find(|&&(a, _)| a == axis)
            .map_or(1, |&(_, size)| size)
    }

    fn grow(&mut self, axis: Axis, count: usize) {
        assert!(
            !self.closed.contains(&axis),
            "Levels: {axis:?} was taken whole by a level below, so nothing above it may state a \
             count of its tiles"
        );
        match self.sizes.iter_mut().find(|(a, _)| *a == axis) {
            Some((_, size)) => *size *= count,
            None => self.sizes.push((axis, count)),
        }
    }

    fn last(&mut self, verb: &str) -> &mut StatedLevel {
        self.stated
            .last_mut()
            .unwrap_or_else(|| panic!("Levels::{verb}: no level has been stated for it to modify"))
    }
}

impl StatedLevel {
    fn new(coverage: Coverage, tiles: Vec<(Axis, usize, usize)>, takes: Takes) -> Self {
        StatedLevel {
            coverage,
            tiles,
            takes,
            across: None,
            shared_by: None,
            fillers: 0,
            batches: Vec::new(),
            interleaved: Vec::new(),
            reopened: Vec::new(),
            order: CubeOrder::RowMajor,
        }
    }

    /// How the level's workers take `axis`'s tiles: in turns where it was said to be.
    fn spread(&self, axis: Axis) -> Spread {
        match self.interleaved.contains(&axis) {
            true => Spread::Interleaved,
            false => Spread::Contiguous,
        }
    }

    /// The [`Level`] this states: every entry with its built tile, its count and its coverage, and
    /// the modifiers applied in the order the level's own builders take them.
    fn level(&self) -> Level {
        let cuts: Vec<(Axis, usize, Count, Spread)> = self
            .tiles
            .iter()
            .map(|&(axis, tile, count)| {
                let count = match (self.takes, self.across) {
                    (Takes::Every, Some((across, cubes))) if across == axis => {
                        Count::AllAcross(cubes)
                    }
                    (Takes::Every, _) => Count::All,
                    (Takes::Stated, _) => Count::Stated(count),
                    (Takes::DistributedToUnits, _) => Count::Distributed(count),
                };
                (axis, tile, count, self.spread(axis))
            })
            .collect();
        let level = Level::new(self.coverage, &cuts);
        let level = match self.batches.is_empty() {
            true => level,
            false => level.batching(&self.batches),
        };
        let level = match self.fillers {
            0 => level,
            n => level.filling(n),
        };
        let level = match self.order.swizzles() {
            false => level,
            true => level.distributed_in(self.order),
        };
        match self.shared_by {
            None => level,
            Some(cubes) => level.sharing(cubes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Partitioning, Space};

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);
    const B: Axis = Axis(3);

    /// The gemm's stack, leaf up: every level's tile is the product of what was stated below it,
    /// and its count is what it stated.
    #[test]
    fn every_level_is_a_count_of_the_level_before() {
        let levels = Levels::leaf(&[(M, 16), (N, 8), (K, 16)])
            .walk(&[(M, 2), (N, 4)])
            .walk(&[(K, 2)])
            .planes(&[(M, 2), (N, 4)])
            .walk_every(&[K])
            .cubes(&[M, N])
            .batches(&[B])
            .build();
        assert_eq!(levels.len(), 5);
        let tiles = |i: usize, axis| levels[i].tile(axis);
        let counts = |i: usize, axis| levels[i].count(axis);

        assert_eq!((tiles(0, M), tiles(0, N)), (Some(64), Some(128)));
        assert_eq!(
            (counts(0, M), counts(0, N)),
            (Some(Count::All), Some(Count::All))
        );
        assert_eq!((tiles(0, B), counts(0, B)), (Some(1), Some(Count::All)));
        assert_eq!(tiles(0, K), None);

        assert_eq!((tiles(1, K), counts(1, K)), (Some(32), Some(Count::All)));
        assert_eq!(
            (tiles(2, M), counts(2, M)),
            (Some(32), Some(Count::Stated(2)))
        );
        assert_eq!(
            (tiles(2, N), counts(2, N)),
            (Some(32), Some(Count::Stated(4)))
        );
        assert_eq!(
            (tiles(3, K), counts(3, K)),
            (Some(16), Some(Count::Stated(2)))
        );
        assert_eq!(
            (tiles(4, M), counts(4, M)),
            (Some(16), Some(Count::Stated(2)))
        );
        assert_eq!(
            (tiles(4, N), counts(4, N)),
            (Some(8), Some(Count::Stated(4)))
        );
    }

    /// The split of a contraction: the cube level names the closed axis again to distribute it across
    /// cubes, in its place on the grid, and the walk below takes every stage of the cube's run.
    #[test]
    fn a_closed_axis_returns_to_the_cube_level_to_be_distributed_across() {
        let levels = Levels::leaf(&[(M, 16), (N, 8), (K, 16)])
            .walk_every(&[K])
            .cubes(&[M, N, K])
            .across(K, 4)
            .build();
        assert_eq!(levels[0].count(K), Some(Count::AllAcross(4)));
        assert_eq!(levels[0].tile(K), Some(16));
        assert_eq!(levels[0].cube_axis(K), Some(crate::CubeAxis::Z));
        assert_eq!(levels[1].count(K), Some(Count::All));
    }

    /// A count cannot fail to divide: three planes of a tile build a tile three times the size,
    /// and there is nothing left for a refusal to check. What the old spelling refused as "the
    /// planes do not distribute into whole tiles" cannot be written.
    #[test]
    fn a_count_builds_its_tile_and_nothing_refuses() {
        let levels = Levels::leaf(&[(M, 5)])
            .planes(&[(M, 3)])
            .walk(&[(M, 7)])
            .cubes(&[M])
            .build();
        assert_eq!(levels[0].tile(M), Some(105));
        assert_eq!(levels[1].tile(M), Some(15));
        assert_eq!(levels[1].count(M), Some(Count::Stated(7)));
        assert_eq!(levels[2].count(M), Some(Count::Stated(3)));
        // Only the every-level at the top can overhang, and only against the problem's extent.
        let space = Space::new(&[(M, 100)]);
        assert!(levels[0].overhangs(&space, M));
        let cube = levels[0].child(&space);
        assert!(!levels[1].overhangs(&cube, M));
        assert!(!levels[2].overhangs(&levels[1].child(&cube), M));
    }

    /// A level that walks every tile and one that walks one of them are different levels: the
    /// first's count is the launch's, the second's is one.
    #[test]
    fn every_is_not_one() {
        let every = Levels::leaf(&[(K, 16)]).walk_every(&[K]).level();
        let one = Levels::leaf(&[(K, 16)]).walk(&[(K, 1)]).level();
        assert_ne!(every, one);
        assert_eq!(every.count(K), Some(Count::All));
        assert_eq!(one.count(K), Some(Count::Stated(1)));
    }

    /// Units taking their tiles in turns state a count like any level, so the tile above is built
    /// the same way, and the launch's unit count is none of the partitioning's.
    #[test]
    fn a_count_distributed_to_the_units_builds_its_tile() {
        let levels = Levels::leaf(&[(M, 4), (N, 4)])
            .units_distributed(N, 64)
            .planes(&[(M, 2)])
            .cubes(&[M, N])
            .build();
        assert_eq!(levels[0].tile(N), Some(256));
        assert_eq!(levels[2].count(N), Some(Count::Distributed(64)));
        assert_eq!(
            levels[2].coverage(),
            Coverage::Distribute(ComputeScope::Unit)
        );
        assert_eq!(levels[2].spread(N), Some(Spread::Interleaved));
        let partitioning = Partitioning::new(Space::new(&[(M, 100), (N, 1000)]), levels);
        assert_eq!(partitioning.units(), 1);
        assert_eq!(partitioning.planes_per_cube(), 2);
    }

    #[test]
    #[should_panic(expected = "distributes no other axis to the units")]
    fn a_count_distributed_to_the_units_is_their_only_axis() {
        let _ = Level::new(
            Coverage::Distribute(ComputeScope::Unit),
            &[
                (M, 4, Count::Distributed(8), Spread::Interleaved),
                (N, 4, Count::Stated(4), Spread::Contiguous),
            ],
        );
    }

    /// The one-liner every walk over a region takes is the chain it stands for.
    #[test]
    fn a_region_walk_is_stated_once() {
        assert_eq!(
            Level::every(&[(M, 8), (K, 16)]),
            Levels::leaf(&[(M, 8), (K, 16)]).walk_every(&[M, K]).level()
        );
    }

    #[test]
    #[should_panic(
        expected = "taken whole by a level below, so nothing above it may state a count"
    )]
    fn nothing_above_all_of_it_may_count_it() {
        let _ = Levels::leaf(&[(K, 16)]).walk_every(&[K]).walk(&[(K, 2)]);
    }

    #[test]
    #[should_panic(expected = "a walk above it has nothing to step through")]
    fn all_of_it_is_walked_once() {
        let _ = Levels::leaf(&[(K, 16)]).walk_every(&[K]).walk_every(&[K]);
    }

    #[test]
    #[should_panic(expected = "names it again only to distribute it across cubes")]
    fn a_closed_axis_on_the_cube_level_must_be_distributed_across() {
        let _ = Levels::leaf(&[(M, 16), (K, 16)])
            .walk_every(&[K])
            .cubes(&[M, K])
            .build();
    }

    #[test]
    #[should_panic(expected = "only cubes take an axis in runs")]
    fn only_cubes_take_an_axis_in_runs() {
        let _ = Levels::leaf(&[(K, 16)]).planes(&[(K, 4)]).across(K, 2);
    }

    #[test]
    #[should_panic(expected = "a walk over a region takes exactly one")]
    fn a_walk_over_a_region_takes_one_level() {
        let _ = Levels::leaf(&[(K, 16)])
            .walk(&[(K, 2)])
            .walk_every(&[K])
            .level();
    }
}
