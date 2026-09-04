//! The axis-agnostic tile DSL engine.
//!
//! A [`Space`] is geometry only: the axes and their extents, and one level per decomposition
//! stating which hardware scope owns which axes ([`LevelCuts::distribute`]) and which axes are
//! stepped through ([`LevelCuts::walk`]). Everything else is the kernel's to write against
//! that space, level by level: the loops ([`Walk::over`] over a level's regions, `at` to
//! descend), where an operand is materialized ([`Ring::smem`] and [`pipelined`], which also
//! own how many regions are in flight), the accumulator it opens and drains
//! ([`Tile::block_accumulator`], [`Tile::cmma_accumulator`], [`Tile::drain_cast_into`]), the
//! fragments it loads ([`PlanePartition::cmma_fragments`]) and the instruction at the leaf
//! ([`Tile::mm_with`], [`Tile::mma`]). The launch ([`Launcher`]) sizes the grid and binds the
//! tensors from the same space function the kernel builds its space with.
#![allow(dead_code)]

mod axis;
mod fold;
pub mod instruction;
mod ops;
mod physical;
mod space;
mod staging;
mod tile;

pub use axis::*;
pub use fold::*;
pub use instruction::*;
pub use ops::*;
pub use physical::*;
pub use space::*;
pub use staging::*;
pub use tile::*;
