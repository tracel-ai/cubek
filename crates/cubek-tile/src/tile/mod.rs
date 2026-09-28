//! One operand's data in the kernel: the [`Tile`] itself (`base`), the kinds it dispatches on
//! (`kind`), how packed values decode (`packing`), the accumulators (`accumulator`), and
//! the scale paths on their way out ([`quant`](crate::quant)).

mod accumulator;
mod base;
pub(crate) mod deprecated;
pub(crate) mod kind;
mod packing;
mod placement;

pub use accumulator::*;
pub use base::*;
pub(crate) use deprecated::*;
pub use kind::*;
pub use packing::*;
pub use placement::*;
