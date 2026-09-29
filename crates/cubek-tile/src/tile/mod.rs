//! One operand's data in the kernel: the [`Tile`] itself (`base`), the kinds it dispatches on
//! (`kind`), how packed values decode (`packing`), and the accumulators (`accumulator`).

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
