//! The cube scope: a cube's runs of a head's value columns, each walked through every position by
//! its units, whose partial sums meet in shared memory.

mod base;
mod held_state;
mod microkernel;
mod state_access;
mod unit_partials;

pub use base::*;
pub use held_state::*;
pub use microkernel::*;
pub use state_access::*;
pub use unit_partials::*;
