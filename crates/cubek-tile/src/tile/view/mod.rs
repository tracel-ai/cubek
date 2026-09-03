//! The layouts a leaf reads a [`Tile`](crate::Tile) through, and the views that wrap them:
//! flat, 2-D matrix, gathered, packed, masked, and the accumulating one.

mod accumulate;
mod coords;
mod flat;
mod masked;
mod matrix;
mod packed;
mod projected;
mod quant;

pub(crate) use accumulate::*;
// Crate-internal helpers, so this re-export carries no public item.
pub(crate) use coords::*;
pub use flat::*;
pub use masked::*;
pub use matrix::*;
pub(crate) use packed::*;
pub use projected::*;
pub use quant::*;
