//! The axis-agnostic tile DSL engine.
//!
//! A [`Space`] is the axes and their extents. A [`Level`] is one decomposition of it, naming
//! the axes a loop cuts and who takes the tiles ([`Level::cubes`], [`Level::planes`],
//! [`Level::lanes`], [`Level::walk`]). A [`Partitioning`] is the space with its levels, outermost
//! first, and is what a kernel is handed: `for cube in space` deals the first level, `for plane
//! in cube` the next, and the [`Region`] each loop hands out carries the path down to `at`. So
//! the launch is the one source of the partitioning: the kernel walks the levels it was handed,
//! one loop each, and a level of its own ([`Region::over`]) is stated beside them. Everything
//! else is the kernel's to write, level by level: where an operand is materialized ([`Ring::smem`] and [`pipelined`], which also own
//! how many regions are in flight), the accumulator it opens, shaped by the statement
//! ([`Fragments`], [`Tile::block_accumulator`], [`Tile::cmma_accumulator`]), the fragments it
//! loads ([`PlanePartition::cmma_fragments`]), the zero of what it holds where it holds it, the
//! instruction at the leaf ([`Tile::mm_with`], [`Tile::mma`]), and the store of each fragment
//! to its window of the output ([`Tile::copy_cast_from`]), one more loop over the cells. Where
//! data decides what a loop reaches, the kernel says that too: [`Walk::routed`] gives an axis
//! the coordinate a table named (an expert per token, a physical page per logical one), and
//! [`Tile::within`] places an operand's window at an element and says where its reads stop (a
//! sequence packed among others). A
//! [`Region`] is the path those loops took from the space, so the root tile and any window of
//! it read one region alike. The launch ([`Launcher`]) reads the grid off the same levels the
//! kernel's loops state, listed by the blueprint, and binds the tensors to the same extents.
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
