//! The cube scope: a cube's runs of a head's value columns, each walked through every position by
//! its units, whose partial sums meet in shared memory.

mod base;
mod held_state;
mod microkernel;
mod unit_partials;

pub use base::*;
pub use held_state::*;
pub use microkernel::*;
pub use unit_partials::*;
