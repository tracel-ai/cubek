#![warn(unreachable_pub)]
//! The axis-agnostic tile DSL engine: a [`Space`] cut into [`Level`]s forms a [`Partitioning`],
//! whose loops hand out [`Region`]s; the [`Launcher`] reads the grid off the same levels.

pub(crate) mod algebra;
pub mod launch;
pub mod layout;
pub mod ops;
pub mod space;
pub mod stage;
pub(crate) mod tile;

/// What a [`Tile`] can be in the kernel.
pub mod kind {
    pub use crate::tile::kind::memory::base::{Boundary, Storage, Write};
    pub use crate::tile::kind::memory::global::GlobalOperand;
    pub use crate::tile::kind::plane::base::PlanePartition;
    pub use crate::tile::packing::base::Field;
}

/// Tiles whose values are computed from their coordinates.
pub mod procedural {
    pub use crate::tile::kind::procedural::affine::{AffineCoordinate, affine_along};
    pub use crate::tile::kind::procedural::base::{Reads, Recipe, RecipeCoords, RecipeExpand};
    pub use crate::tile::kind::procedural::constant::Constant;
    pub use crate::tile::kind::procedural::kind::Procedural;
    pub use crate::tile::kind::procedural::normalization::{DivGuard, TapSupport};
    pub use crate::tile::kind::procedural::phase::Phase;
    pub use crate::tile::kind::procedural::product::{Product, product_of};
    pub use crate::tile::kind::procedural::separable::Factors;
    pub use crate::tile::kind::procedural::sum::{Sum, sum_of};
}

// Flat crate namespace for `use crate::*`.
#[allow(unused_imports)]
pub(crate) use {algebra::*, launch::*, layout::*, ops::*, space::*, stage::*, tile::*};

pub use algebra::monoid::Monoid;
pub use algebra::semiring::Semiring;
pub use launch::arg::tensor::{TileArg, TileArgLaunch};
pub use launch::base::Launcher;
pub use launch::spec::TileSpec;
pub use layout::geometry::Geometry;
pub use layout::projection::Projection;
pub use ops::matmul::config::RegisterBlock;
pub use ops::matmul::instruction::Instruction;
pub use space::axis::Axis;
pub use space::base::Space;
pub use space::partition::base::Partitioning;
pub use space::partition::level::Level;
pub use space::partition::levels::Levels;
pub use space::region::Region;
pub use space::walk::base::Walk;
pub use stage::base::StageStorage;
pub use stage::pipeline::base::Stages;
pub use tile::accumulator::base::{Accumulate, AccumulateExpand, Scratch};
pub use tile::base::{Tile, TileExpand};
