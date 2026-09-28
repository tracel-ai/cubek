//! The axis-agnostic tile DSL engine.
//!
//! A [`Space`] is the axes and their extents. A [`Level`] is one decomposition of it (the axes a
//! loop steps, in what tile, how many, who takes them), stated leaf-up in counts ([`Levels`]). A
//! [`Partitioning`] is the space with its levels, outermost first, and is what a kernel is handed.
//!
//! `for cube in space` distributes the first level, `for plane in cube` the next, each loop handing
//! out a [`Region`] (the path down to `at`); a level of the kernel's own is [`Region::over`]. The
//! launch ([`Launcher`]) reads the grid off those levels and binds the tensors to the same extents.
//!
//! The rest is the kernel's: operand storage ([`Stages::smem`], [`Stages::pipelined`]), accumulator
//! ([`Accumulate`]), loaded fragments ([`PlanePartition::cmma_fragments`]), and the leaf it
//! contracts through ([`Tile::mm_with`], [`Tile::mma`]).
//!
//! A fragment's store to its output window is [`Tile::copy_cast_from`]. Where data decides what
//! a loop reaches, [`Walk::routed`] gives an axis a table's coordinate (an expert per token,
//! a page per logical one) and [`Tile::within`] puts a window at an element and bounds its reads.

pub mod algebra;
pub mod launch;
pub mod layout;
pub mod ops;
pub mod space;
pub mod stage;
mod tile;

/// What a [`Tile`] can be in the kernel: global or staged memory, plane fragments, register
/// lines, and the accumulators and packed fields built over them.
pub mod kind {
    pub use crate::tile::*;
}

/// Tiles whose values are computed from their coordinates rather than read: a [`Recipe`]
/// per element, or [`Factors`] combined along each axis.
pub mod procedural {
    pub use crate::tile::kind::procedural::*;
}

/// Quantized operands as one launch argument ([`QuantTileArg`]). On its way
/// out: scales are moving onto [`Tile::mul`](crate::Tile::mul).
pub mod quant {
    pub use crate::tile::deprecated::quant::*;
}

// The crate's own flat namespace: every module's items under `crate::`, for `use crate::*`.
#[allow(unused_imports)]
pub(crate) use {algebra::*, launch::*, layout::*, ops::*, space::*, stage::*, tile::*};

// The facade's core: what nearly every kernel names. Everything else lives in its module.
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
