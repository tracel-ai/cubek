//! The axis-agnostic tile DSL engine.
//!
//! A [`Space`] is the axes and their extents. A [`Level`] is one decomposition of it (the axes a
//! loop steps, in what tile, how many, who takes them), stated leaf-up in counts ([`Tiling`]). A
//! [`Partitioning`] is the space with its levels, outermost first, and is what a kernel is handed.
//!
//! `for cube in space` deals the first level, `for plane in cube` the next, each loop handing out
//! a [`Region`] (the path down to `at`); a level of the kernel's own is [`Region::over`]. The
//! launch ([`Launcher`]) reads the grid off those levels and binds the tensors to the same extents.
//!
//! The rest is the kernel's: operand storage ([`Stages::smem`], [`pipelined`]), accumulator
//! ([`Accumulate`]), loaded fragments ([`PlanePartition::cmma_fragments`]), leaf instruction
//! ([`Tile::mm_with`], [`Tile::mma`]).
//!
//! A fragment's store to its output window is [`Tile::copy_cast_from`]. Where data decides what
//! a loop reaches, [`Walk::routed`] gives an axis a table's coordinate (an expert per token,
//! a page per logical one) and [`Tile::within`] puts a window at an element and bounds its reads.

mod algebra;
pub mod instruction;
mod launch;
mod layout;
mod load;
mod ops;
mod space;
mod tile;

pub use algebra::*;
pub use instruction::*;
pub use launch::*;
pub use layout::*;
pub use load::*;
pub use ops::*;
pub use space::*;
pub use tile::*;
