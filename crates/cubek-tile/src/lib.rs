//! The axis-agnostic tile DSL engine — works over a labeled axis set a client
//! declares (see `matmul.rs`), knowing nothing of `M`/`N`/`K` or axis count.
//!
//! Three nouns — `Space`, `Region`, `Tile`:
//! - [`space`] — the [`Space`]: geometry only (axes + extents, origin implicit/zero)
//!   that owns its partitioner. **Partitioning a space yields a [`Walk`] of
//!   [`Region`]s**; the [`count`](Space::count) of pieces is the space's own property
//!   (no operand, no view).
//! - [`Region`] — a `Space` at an origin: what partitioning visits at each step (its
//!   origin runtime, since a spatial walk folds in the hardware position). The
//!   [`Walk`] is the runtime odometer that yields one per step.
//! - [`tile`] — the [`Tile`]: a projected, storage-bound view of an operand — its
//!   axes plus a buffer, not the whole operation space but the sub-Space the data
//!   lies in (`out ∈ {M,N}`, no `K`). **Locating a tile [`at`](Tile::at) a
//!   [`Region`] yields a [`Tile`]** — a pure within-level view; the region is chosen
//!   in the loop. Dropping axes is a separate op, [`project`](Space::project).
//!
//! There is no operation *type*: an operation is its lowering — reconstruct the
//! operation's space from the operand tiles (if needed), partition it, and locate
//! each tile at every region. [`matmul`] is the worked example (and ships one
//! built-in, [`mma_staged`]).
//!
//! Supporting machinery:
//! - [`partitioner`] — how a level splits, the [`Walk`] over a [`Space`], and the
//!   example walk orders plugged into its seam.
//! - [`dim2`] — the 2-D tile world a tile collapses into at the leaf.
#![allow(dead_code)]

mod dim2;
mod matmul;
mod partitioner;
mod ring;
mod space;
mod tile;

pub use dim2::*;
pub use matmul::*;
pub use partitioner::*;
pub use ring::*;
pub use space::*;
pub use tile::*;
