//! The cube scope: a cube's runs of a head's value columns, each walked through every position by
//! its units, whose partial sums meet in shared memory.

mod additive;
mod delta;
mod held_state;
mod microkernel;
mod partials;
mod row_pass;
mod settings;
mod shape;
mod state_access;

pub use additive::*;
pub use delta::*;
pub use held_state::*;
pub use microkernel::*;
pub use partials::*;
pub use row_pass::*;
pub use settings::*;
pub use shape::*;
pub use state_access::*;
