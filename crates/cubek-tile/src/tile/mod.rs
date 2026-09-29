//! One operand's data in the kernel: the [`Tile`], its kinds, packing and accumulators.

mod accumulator;
mod base;
pub(crate) mod kind;
mod packing;
mod placement;

pub use accumulator::*;
pub use base::*;
pub use kind::*;
pub use packing::*;
pub use placement::*;
