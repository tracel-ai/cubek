//! Where a tile sits in its partitioning.

use crate::{Level, Space};

/// A tile's place in its nest: its box (`space`), how many levels down it sits (`depth`), and the
/// partitioning levels it is walked with (`levels`).
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

    /// At the top of `levels`.
    pub fn root(space: Space, levels: Vec<Level>) -> Self {
        Placement::new(space, 0, levels)
    }

    /// Outside any partitioning.
    pub fn alone(space: Space) -> Self {
        Placement::new(space, 0, Vec::new())
    }

    /// The levels below this depth.
    pub fn below(&self) -> &[Level] {
        &self.levels[self.depth..]
    }

    /// One level down, through `level`.
    pub(crate) fn at_step(&self, level: &Level) -> Placement {
        Placement::new(
            level.child(&self.space),
            self.depth + 1,
            self.levels.clone(),
        )
    }

    /// The same box at `depth` in its nest.
    pub(crate) fn at_depth(mut self, depth: usize) -> Placement {
        self.depth = depth;
        self
    }
}
