//! Stating a partitioning from the leaf up, in counts.
//!
//! A [`Level`] carries the *size* of one of its tiles, and a caller almost always knows that size
//! as a product: the instruction is `16x8x16`, a plane holds `2x4` of them, a cube holds `2x4`
//! planes. Written as sizes, those three facts read as `16x8`, `32x32` and `64x128`, and whether
//! two of them are the same kind of number is something the reader divides to find out. Written
//! here, the multiplication is stated once and the sizes are nobody's business.
//!
//! ```ignore
//! Tiling::leaf(&[(M, 16), (N, 8), (K, 16)])   // the instruction: the device's, not a choice
//!     .walk(&[(M, 2), (N, 4)])                // fragments a plane holds
//!     .walk(&[(K, 2)])                        // instruction steps in one stage
//!     .planes(&[(M, 2), (N, 4)])              // planes a cube holds
//!     .walk_every(&[K])                       // every stage `k` holds
//!     .cubes(&[M, N]).batches(&[B])           // every box `m x n` holds
//!     .levels()
//! ```
//!
//! **The leaf is not a level.** It is the size the first level above it states as its tile, which
//! is why the instruction disappears from the list and becomes the first line instead.
//!
//! **Counts, and "all of it".** Every level says how many of the thing below it. The one
//! exception is a level that takes *every* tile of an axis — a reduction's whole `K`, the boxes
//! of the output, the batch — a count nobody knows until launch. That is
//! [`walk_every`](Tiling::walk_every), [`cubes`](Tiling::cubes) and [`batches`](Tiling::batches),
//! which name axes and no number, and what they settle is this:
//!
//! * "every" means every tile **the level above hands down** — the whole axis only at the
//!   outermost level that names it. Under a cube level that deals the axis across cubes, it is
//!   the cube's run.
//! * It closes the axis. Nothing above may count its tiles, because nothing above knows how
//!   many there are; a walk above it would have nothing to step through.
//! * It returns exactly once, on the cube level, and only to be dealt across cubes
//!   ([`across`](Tiling::across)): the split of a contraction is the walk below taking every
//!   stage of the run the grid hands its cube.
//! * An axis a level does not name is not "all of it"; it is not that level's. The region is
//!   handed down whole, unstepped, and the [`Level`] says so with `Edge::Whole`.
//!
//! So the only division left in a launch is the outermost level's tile against the problem's
//! extent, taken once, ragged at the end; between two stated levels there is nothing to divide.
//!
//! **One reversal, named.** Levels are stated innermost first and a kernel's loops run outermost
//! first, so [`levels`](Tiling::levels) reverses. That is the only place the two directions meet.

use crate::{Axis, Cut, Level};

/// Who takes the tiles of one level.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Takers {
    Walk,
    Lanes,
    Planes,
    Cubes,
}

/// One level, as the builder holds it before it is a [`Level`]: the axes it names with the tile
/// each is built of, and the modifiers stated after it.
#[derive(Clone, Debug)]
struct Stated {
    takers: Takers,
    /// `(axis, the size one tile of this level covers, how many)` — the running size below it,
    /// and the count stated. Only a lanes level reads the count: a plane is carved between its
    /// entries, so each says how many lanes take it.
    tiles: Vec<(Axis, usize, usize)>,
    /// A cube level dealing one of its axes across several cubes ([`Tiling::across`]): the axis
    /// and how many.
    across: Option<(Axis, usize)>,
    /// A cube level dealing its tiles as one index ([`Tiling::shared_by`]).
    shared_by: Option<usize>,
    /// Planes that only fill this walk's stages ([`Tiling::filled_by`]).
    fillers: usize,
    /// Batch axes this cube level hands out one of ([`Tiling::batches`]).
    batches: Vec<Axis>,
    /// Axes whose tiles the workers take in turns rather than in runs ([`Tiling::interleaved`]).
    interleaved: Vec<Axis>,
    /// Axes a level below already took whole, which this cube level names again — legal only to
    /// deal them across cubes, checked when the levels are built.
    reopened: Vec<Axis>,
}

/// A partitioning stated from the leaf up. See the module docs.
#[derive(Clone, Debug)]
pub struct Tiling {
    /// What one tile of each axis covers so far, leaf up. An axis nobody built covers one.
    sizes: Vec<(Axis, usize)>,
    /// Axes an `every` level finished, which nothing above may multiply again.
    closed: Vec<Axis>,
    /// Innermost first; [`levels`](Self::levels) reverses.
    stated: Vec<Stated>,
}

impl Tiling {
    /// The tile one worker holds at the bottom — an instruction's shape, a register block, a
    /// lane's vector. The only sizes in a partitioning, and the only ones a device dictates.
    pub fn leaf(tile: &[(Axis, usize)]) -> Self {
        Tiling {
            sizes: tile.to_vec(),
            closed: Vec::new(),
            stated: Vec::new(),
        }
    }

    /// Every worker steps through this many of the thing below, one at a time.
    pub fn walk(self, counts: &[(Axis, usize)]) -> Self {
        self.state(Takers::Walk, counts)
    }

    /// Every worker steps through as many of the thing below as the axis holds — the count the
    /// levels do not know, and the launch does.
    pub fn walk_every(self, axes: &[Axis]) -> Self {
        self.every(Takers::Walk, axes)
    }

    /// This many of the thing below, one per lane of the plane.
    pub fn lanes(self, counts: &[(Axis, usize)]) -> Self {
        self.state(Takers::Lanes, counts)
    }

    /// This many of the thing below, one per plane of the cube.
    pub fn planes(self, counts: &[(Axis, usize)]) -> Self {
        self.state(Takers::Planes, counts)
    }

    /// One cube for every box these axes hold. The outermost level of a launch, and the one whose
    /// count is the problem's rather than the plan's.
    ///
    /// The two ways several cubes share a box are stated after it:
    /// [`across`](Self::across) cuts one axis among them, [`shared_by`](Self::shared_by) deals
    /// this level's tiles to them as one index.
    pub fn cubes(self, axes: &[Axis]) -> Self {
        self.every(Takers::Cubes, axes)
    }

    /// Deal `axis` — one the cube level just stated — across `cubes` of them, each taking a run
    /// of its tiles. Split-K, where the axis is the contraction. The axis keeps its place among
    /// the level's entries, and so its grid dimension.
    pub fn across(mut self, axis: Axis, cubes: usize) -> Self {
        let stated = self.last("across");
        assert!(
            stated.tiles.iter().any(|&(a, _, _)| a == axis),
            "Tiling::across: {axis:?} is not an axis of the cube level just stated; name it there"
        );
        stated.across = Some((axis, cubes));
        self
    }

    /// Deal this level's tiles to `cubes` of them **as one index**, so a cube's share is a run of
    /// the whole rather than a box of it. Stream-K.
    pub fn shared_by(mut self, cubes: usize) -> Self {
        self.last("shared_by").shared_by = Some(cubes);
        self
    }

    /// This walk's stages are filled by `n` planes of the cube, which take no tile of any level.
    pub fn filled_by(mut self, n: usize) -> Self {
        self.last("filled_by").fillers = n;
        self
    }

    /// Batch axes this cube level hands out one of.
    pub fn batches(mut self, axes: &[Axis]) -> Self {
        self.last("batches").batches = axes.to_vec();
        self
    }

    /// The workers of the level just stated take `axis`'s tiles in turns — worker 0 the first,
    /// worker 1 the next — rather than each a contiguous run. Neighbouring lanes then read
    /// neighbouring memory at the same instant, which is what a lane split of a contiguous
    /// contraction wants.
    pub fn interleaved(mut self, axis: Axis) -> Self {
        self.last("interleaved").interleaved.push(axis);
        self
    }

    /// The levels, **outermost first**: what a kernel's loops walk and a launch reads its grid
    /// off. The one place the two directions meet.
    pub fn levels(self) -> Vec<Level> {
        for stated in &self.stated {
            for &axis in &stated.reopened {
                assert!(
                    matches!(stated.across, Some((a, _)) if a == axis),
                    "Tiling::cubes: {axis:?} was taken whole by a level below; a cube level names \
                     it again only to deal it across cubes (`.across({axis:?}, n)`)"
                );
            }
        }
        self.stated.iter().rev().map(Stated::level).collect()
    }

    /// State a level of `counts` of the thing below, then grow each axis by its count.
    fn state(mut self, takers: Takers, counts: &[(Axis, usize)]) -> Self {
        let tiles = counts
            .iter()
            .map(|&(axis, count)| (axis, self.size(axis), count));
        self.stated.push(Stated::new(takers, tiles.collect()));
        for &(axis, count) in counts {
            self.grow(axis, count);
        }
        self
    }

    /// State a level covering every tile these axes hold, and close them: a count nothing above
    /// can multiply, because nothing above knows it.
    fn every(mut self, takers: Takers, axes: &[Axis]) -> Self {
        let reopened: Vec<Axis> = axes
            .iter()
            .copied()
            .filter(|axis| self.closed.contains(axis))
            .collect();
        if let (Takers::Walk, Some(axis)) = (takers, reopened.first()) {
            panic!(
                "Tiling::walk_every: {axis:?} was taken whole by a level below, so a walk above \
                 it has nothing to step through"
            );
        }
        let tiles = axes.iter().map(|&axis| (axis, self.size(axis), 1));
        let mut stated = Stated::new(takers, tiles.collect());
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
            "Tiling: {axis:?} was taken whole by a level below, so nothing above it may state a \
             count of its tiles"
        );
        match self.sizes.iter_mut().find(|(a, _)| *a == axis) {
            Some((_, size)) => *size *= count,
            None => self.sizes.push((axis, count)),
        }
    }

    fn last(&mut self, verb: &str) -> &mut Stated {
        self.stated
            .last_mut()
            .unwrap_or_else(|| panic!("Tiling::{verb}: no level has been stated for it to modify"))
    }
}

impl Stated {
    /// `cut` dealt in turns where its axis was said to be.
    fn spread(&self, cut: Cut, axis: Axis) -> Cut {
        match self.interleaved.contains(&axis) {
            true => cut.interleaved(),
            false => cut,
        }
    }

    fn new(takers: Takers, tiles: Vec<(Axis, usize, usize)>) -> Self {
        Stated {
            takers,
            tiles,
            across: None,
            shared_by: None,
            fillers: 0,
            batches: Vec::new(),
            interleaved: Vec::new(),
            reopened: Vec::new(),
        }
    }

    /// The [`Level`] this states: the tiles as edges, and the modifiers applied in the order the
    /// level's own builders take them.
    fn level(&self) -> Level {
        let edges: Vec<(Axis, usize)> = self.tiles.iter().map(|&(a, tile, _)| (a, tile)).collect();
        let level = match self.takers {
            Takers::Walk => Level::walk(&edges),
            Takers::Planes => Level::planes(&edges),
            // The plane is carved between the entries, so each says how many lanes take it.
            Takers::Lanes => {
                let cuts: Vec<Cut> = self
                    .tiles
                    .iter()
                    .map(|&(axis, tile, lanes)| {
                        self.spread(Cut::new(axis, tile).across(lanes), axis)
                    })
                    .collect();
                Level::lanes(&cuts)
            }
            Takers::Cubes => {
                let cuts: Vec<Cut> = edges
                    .iter()
                    .map(|&(axis, tile)| {
                        let cut = Cut::new(axis, tile);
                        match self.across {
                            Some((across, cubes)) if across == axis => {
                                self.spread(cut.across(cubes), axis)
                            }
                            _ => cut,
                        }
                    })
                    .collect();
                Level::cubes(&cuts)
            }
        };
        let level = match self.batches.is_empty() {
            true => level,
            false => level.batches(&self.batches),
        };
        let level = match self.fillers {
            0 => level,
            n => level.filled_by(n),
        };
        match self.shared_by {
            None => level,
            Some(cubes) => level.shared_by(cubes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);
    const B: Axis = Axis(3);

    /// The gemm's stack, leaf up: every level's tile is the product of what was stated below it.
    #[test]
    fn every_level_is_a_count_of_the_level_before() {
        let levels = Tiling::leaf(&[(M, 16), (N, 8), (K, 16)])
            .walk(&[(M, 2), (N, 4)])
            .walk(&[(K, 2)])
            .planes(&[(M, 2), (N, 4)])
            .walk_every(&[K])
            .cubes(&[M, N])
            .batches(&[B])
            .levels();
        assert_eq!(
            levels,
            vec![
                Level::cubes(&[(M, 64), (N, 128)]).batches(&[B]),
                Level::walk(&[(K, 32)]),
                Level::planes(&[(M, 32), (N, 32)]),
                Level::walk(&[(K, 16)]),
                Level::walk(&[(M, 16), (N, 8)]),
            ]
        );
    }

    /// The split of a contraction: the cube level names the closed axis again to deal it across
    /// cubes, in its place on the grid, and the walk below takes every stage of the cube's run.
    #[test]
    fn a_closed_axis_returns_to_the_cube_level_to_be_dealt_across() {
        let levels = Tiling::leaf(&[(M, 16), (N, 8), (K, 16)])
            .walk_every(&[K])
            .cubes(&[M, N, K])
            .across(K, 4)
            .levels();
        assert_eq!(
            levels,
            vec![
                Level::cubes(&[Cut::new(M, 16), Cut::new(N, 8), Cut::new(K, 16).across(4)]),
                Level::walk(&[(K, 16)]),
            ]
        );
    }

    #[test]
    #[should_panic(
        expected = "taken whole by a level below, so nothing above it may state a count"
    )]
    fn nothing_above_all_of_it_may_count_it() {
        let _ = Tiling::leaf(&[(K, 16)]).walk_every(&[K]).walk(&[(K, 2)]);
    }

    #[test]
    #[should_panic(expected = "a walk above it has nothing to step through")]
    fn all_of_it_is_walked_once() {
        let _ = Tiling::leaf(&[(K, 16)]).walk_every(&[K]).walk_every(&[K]);
    }

    #[test]
    #[should_panic(expected = "names it again only to deal it across cubes")]
    fn a_closed_axis_on_the_cube_level_must_be_dealt_across() {
        let _ = Tiling::leaf(&[(M, 16), (K, 16)])
            .walk_every(&[K])
            .cubes(&[M, K])
            .levels();
    }
}
