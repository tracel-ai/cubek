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
//! **Counts, except twice.** Every level says how many of the thing below it. The exception is an
//! axis whose count nobody knows until launch — a reduction's whole `K`, the boxes of the output
//! — and that is [`walk_every`](Tiling::walk_every) and [`cubes`](Tiling::cubes), which take axes
//! and no number. They are the outermost levels of every kernel here, and the only runtime
//! numbers in a launch.
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
    /// A cube level dealing one axis across several cubes ([`Tiling::across`]).
    across: Option<(Axis, usize, usize)>,
    /// A cube level dealing its tiles as one index ([`Tiling::shared_by`]).
    shared_by: Option<usize>,
    /// Planes that only fill this walk's stages ([`Tiling::filled_by`]).
    fillers: usize,
    /// Batch axes this cube level hands out one of ([`Tiling::batches`]).
    batches: Vec<Axis>,
    /// Axes whose tiles the workers take in turns rather than in runs ([`Tiling::interleaved`]).
    interleaved: Vec<Axis>,
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

    /// Deal `axis` across `cubes` of them, each taking a run of its tiles. Split-K, where the
    /// axis is the contraction.
    pub fn across(mut self, axis: Axis, cubes: usize) -> Self {
        let tile = self.size(axis);
        self.last("across").across = Some((axis, tile, cubes));
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
        let tiles = axes.iter().map(|&axis| (axis, self.size(axis), 1));
        self.stated.push(Stated::new(takers, tiles.collect()));
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
                let mut cuts: Vec<Cut> = edges
                    .iter()
                    .map(|&(axis, tile)| Cut::new(axis, tile))
                    .collect();
                if let Some((axis, tile, cubes)) = self.across {
                    cuts.push(self.spread(Cut::new(axis, tile).across(cubes), axis));
                }
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
