//! Where a tile sits in its partitioning: its own space, how many levels down it is, and the
//! levels it is walked with.

use crate::{Level, Space};

/// A tile's place in its nest: the box it covers (`space`), how many levels down the partitioning
/// it sits (`depth`, what `at` skips of a region's path, so a region names the same box from the
/// root tile and from any window of it), and every level of the partitioning it is walked with
/// (`levels`; the one at its depth is what `for plane in tile` deals). Empty levels for a tile no
/// partitioning states.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Placement {
    pub space: Space,
    pub depth: usize,
    pub levels: Vec<Level>,
}

impl Placement {
    pub fn new(space: Space, depth: usize, levels: Vec<Level>) -> Self {
        Placement {
            space,
            depth,
            levels,
        }
    }

    /// At the top of `levels`: what a tile served off a kernel argument is.
    pub fn root(space: Space, levels: Vec<Level>) -> Self {
        Placement::new(space, 0, levels)
    }

    /// Outside any partitioning: a tile no loop deals.
    pub fn alone(space: Space) -> Self {
        Placement::new(space, 0, Vec::new())
    }

    /// The levels below this depth, the first of which `for plane in tile` deals.
    pub fn below(&self) -> &[Level] {
        &self.levels[self.depth..]
    }

    /// One level down, through `level`: its child box, one deeper.
    pub(crate) fn at_step(&self, level: &Level) -> Placement {
        Placement::new(
            level.child(&self.space),
            self.depth + 1,
            self.levels.clone(),
        )
    }

    /// The same box at `depth` in its nest: what a stage allocated for a walk's regions sits at.
    pub(crate) fn at_depth(mut self, depth: usize) -> Placement {
        self.depth = depth;
        self
    }
}
