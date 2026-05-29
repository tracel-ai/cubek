//! How a level of the space is split, and the ordered [`Walk`] that splitting
//! produces.
//!
//! - [`base`] — the [`Partitioner`] itself: the per-axis split + walk order.
//! - [`distribution`] — the split vocabulary ([`Distribution`]/[`Coverage`]/
//!   [`Spread`] and the hardware primitives).
//! - [`walk`] — the [`Walk`] odometer that turns a step into a [`Point`].
//! - [`walk_order`] — example walk orders plugged into the partitioner seam, not
//!   core machinery.
//! - [`cube_count`] — the launch geometry a partitioner implies.

mod base;
mod cube_count;
mod distribution;
mod walk;
mod walk_order;

pub use base::*;
pub use cube_count::*;
pub use distribution::*;
pub use walk::*;
pub use walk_order::*;
