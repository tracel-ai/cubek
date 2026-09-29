//! The axis-agnostic tile DSL engine: a [`Space`] cut into [`Level`]s forms a [`Partitioning`],
//! whose loops hand out [`Region`]s; the [`Launcher`] reads the grid off the same levels.

pub mod algebra;
pub mod launch;
pub mod layout;
pub mod ops;
pub mod space;
pub mod stage;
mod tile;

/// What a [`Tile`] can be in the kernel.
pub mod kind {
    pub use crate::tile::*;
}

/// Tiles whose values are computed from their coordinates.
pub mod procedural {
    pub use crate::tile::kind::procedural::*;
}

// Flat crate namespace for `use crate::*`.
#[allow(unused_imports)]
pub(crate) use {algebra::*, launch::*, layout::*, ops::*, space::*, stage::*, tile::*};

pub use algebra::{Monoid, Semiring};
pub use launch::{Launcher, TileArg, TileArgLaunch, TileSpec};
pub use layout::{Geometry, Projection};
pub use ops::matmul::{Instruction, RegisterBlock};
pub use space::{
    Axis, Level, Levels, Partitioning, PartitioningExpand, PartitioningLaunch, Region,
    RegionExpand, Space, Walk, WalkExpand,
};
pub use stage::{StageStorage, Stages, StagesExpand};
pub use tile::{Accumulate, AccumulateExpand, Scratch, Tile, TileExpand};
