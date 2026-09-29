//! One operand's data in the kernel: the [`Tile`], its kinds, packing and accumulators.

pub(crate) mod accumulator;
pub(crate) mod base;
pub(crate) mod kind;
pub(crate) mod packing;
pub(crate) mod placement;

pub use accumulator::*;
pub use kind::*;
pub use packing::*;
pub(crate) use placement::*;
