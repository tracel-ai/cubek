//! The cube scope: a cube's runs of a head's value columns, each walked through every position by
//! its units, whose partial sums meet in shared memory.

mod delta;
mod held_state;
mod state_access;
mod unit_partials;

pub use delta::*;
pub use held_state::*;
pub use state_access::*;
pub use unit_partials::*;
