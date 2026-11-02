//! The cube scope: a cube's runs of units each walk a run of a head's value columns through every
//! position, their partial sums meeting in shared memory.

mod base;
mod held_state;
mod microkernel;
mod unit_partials;

pub use base::*;
pub use held_state::*;
pub use microkernel::*;
pub use unit_partials::*;
