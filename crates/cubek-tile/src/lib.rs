//! The axis-agnostic tile DSL engine. Nothing here knows about `M`, `N`, `K`, or
//! how many axes a problem has; it works over a labeled axis set a client
//! declares (see `matmul.rs`).
//!
//! Read it in this order:
//!
//! - [`axis`] — [`Axis`] labels and [`ByAxis`], the comptime map that keys
//!   everything per-axis (never positional `{m, n, k}` fields).
//! - [`space`] — the [`Space`] a tile lives in (its own axes + extents); an
//!   operation's space is the [union](Space::union) of its operands' spaces, and
//!   the contracted axes are `union ∖ output`.
//! - [`partitioner`] — how a level is split ([`Distribution`]/[`Coverage`]/
//!   [`Spread`]) and the [`Walk`] over a [`Grid`] (the space sized in tiles — the
//!   runtime sibling of [`Space`]) whose odometer turns a step into a [`Point`].
//! - [`tile`] — the [`Tile`] itself, its memory-space [`TileKind`]s, and the
//!   arity-agnostic `partition` seam + `tiles` count. (The `mma` op is a client
//!   concern — it lives in `matmul.rs`.)
//! - [`dim2`] — the 2-D leaf world `partition` collapses into: the leaf layouts,
//!   the `copy_from`/`copy_2d` element copies, and the `stage_smem` factory. The
//!   only place `Coords2d` and `row`/`col` are allowed.
//!
//! Plus the DSL's own test surface: [`tile_input`] (a launchable-tile test input)
//! and [`recursive`] (a multi-level tiling round-trip test). Clients live outside
//! this module (e.g. `matmul.rs`).
#![allow(dead_code)]

mod dim2;
mod partitioner;
mod space;
mod tile;

pub use dim2::*;
pub use partitioner::*;
pub use space::*;
pub use tile::*;
