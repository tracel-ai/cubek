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
//! - [`tile`] — the [`Tile`] itself, its memory-space [`TileKind`]s, the three
//!   verbs (`mma`/`partition`/`copy_from`), and the [`Gmem`]/[`Smem`] factories.
//! - [`mma`] — the lowering strategies the verbs dispatch to.
//! - [`layout`] — 2-D layouts re-presenting tiled storage to the leaf.
#![allow(dead_code)]

mod layout;
mod mma;
mod partitioner;
mod space;
mod tile;

pub use layout::*;
pub use mma::*;
pub use partitioner::*;
pub use space::*;
pub use tile::*;
