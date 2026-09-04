//! The axis-agnostic tile DSL engine.
//!
//! A [`Space`] is geometry only: the axes and their extents. A [`Level`] is one decomposition
//! of it, stating which hardware scope owns which axes ([`LevelCuts::distribute`]) and which
//! axes are stepped through ([`LevelCuts::walk`]); it lives on the loop that states it
//! ([`Space::level`]), and the [`Region`] that loop hands out carries it down to `at`. So the
//! kernel is the one source of its partitioning: it cannot walk a level it does not state,
//! and what it states is what it walks. Everything else is the kernel's to write, level by
//! level: where an operand is materialized ([`Ring::smem`] and [`pipelined`], which also own
//! how many regions are in flight), the accumulator it opens, shaped by the statement
//! ([`Fragments`]), and drains by replaying the levels its `at`s recorded
//! ([`Tile::block_accumulator`], [`Tile::cmma_accumulator`], [`Tile::drain_cast_into`]), the
//! fragments it loads ([`PlanePartition::cmma_fragments`]), the zero of what it holds where it
//! holds it, and the instruction at the leaf ([`Tile::mm_with`], [`Tile::mma`]). The launch
//! ([`Launcher`]) sizes the grid from the same levels the kernel's loops state, listed by the
//! blueprint, and binds the tensors to the same extents.
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
