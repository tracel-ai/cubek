//! Where a tile sits in its partitioning.

use cubecl::prelude::*;

use crate::{ComputeScope, Level, Space};

/// A tile's place in its nest: its box (`space`), how many levels down it sits (`depth`), and the
/// partitioning levels it is walked with (`levels`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Placement {
    pub space: Space,
    pub depth: usize,
    pub levels: Vec<Level>,
}

impl Placement {
    pub(crate) fn new(space: Space, depth: usize, levels: Vec<Level>) -> Self {
        Placement {
            space,
            depth,
            levels,
        }
    }

    /// At the top of `levels`.
    pub(crate) fn root(space: Space, levels: Vec<Level>) -> Self {
        Placement::new(space, 0, levels)
    }

    /// Outside any partitioning.
    pub(crate) fn alone(space: Space) -> Self {
        Placement::new(space, 0, Vec::new())
    }

    /// Who holds this tile: the narrowest scope a level above it in its nest hands its tiles to,
    /// the cube where none distributes below it. A plane's window of a cube's buffer is the
    /// plane's, whoever filled the buffer.
    pub(crate) fn holder(&self) -> ComputeScope {
        self.levels[..self.depth.min(self.levels.len())]
            .iter()
            .filter_map(|level| level.coverage().scope())
            .min()
            .unwrap_or(ComputeScope::Cube)
    }

    /// The levels below this depth.
    pub(crate) fn below(&self) -> &[Level] {
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

/// This unit's position among the units of the scope holding a tile ([`Placement::holder`]): its
/// position in the cube, in its plane, or zero for a unit's own tile.
#[cube]
pub(crate) fn holder_worker(#[comptime] holder: ComputeScope) -> usize {
    match comptime!(holder) {
        ComputeScope::Cube => UNIT_POS as usize,
        ComputeScope::Plane => UNIT_POS_PLANE as usize,
        ComputeScope::Unit => 0usize,
    }
}

/// How many units the scope holding a tile has ([`holder_worker`]).
#[cube]
pub(crate) fn holder_workers(#[comptime] holder: ComputeScope) -> usize {
    match comptime!(holder) {
        ComputeScope::Cube => CUBE_DIM as usize,
        ComputeScope::Plane => PLANE_DIM as usize,
        ComputeScope::Unit => 1usize,
    }
}
