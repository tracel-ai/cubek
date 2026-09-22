//! One operand's data in the kernel: the [`Tile`] itself ([`base`]), the kinds it dispatches on
//! ([`kind`]), how packed values decode ([`packing`]), the accumulators ([`accumulator`]), and
//! the scale paths on their way out ([`deprecated`]).

mod accumulator;
mod atomic;
mod base;
mod deprecated;
mod kind;
mod operand;
mod packing;

pub use accumulator::*;
pub use base::*;
pub use deprecated::*;
pub use kind::*;
pub use operand::*;
pub use packing::*;
