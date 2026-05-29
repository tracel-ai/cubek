//! The axis-agnostic tile DSL engine — works over a labeled axis set a client
//! declares (see `matmul.rs`), knowing nothing of `M`/`N`/`K` or axis count.
//!
//! - [`space`] — the [`Space`] a tile lives in (axes + extents).
//! - [`partitioner`] — how a level splits, the [`Walk`] over a [`Grid`], and the
//!   example walk orders plugged into its seam.
//! - [`tile`] — the [`Tile`], its [`TileKind`]s, and the `partition` seam.
//! - [`dim2`] — the 2-D leaf world `partition` collapses into.
#![allow(dead_code)]

mod dim2;
mod partitioner;
mod space;
mod tile;

pub use dim2::*;
pub use partitioner::*;
pub use space::*;
pub use tile::*;
