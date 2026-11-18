//! The reduce family's autotune: every routine at every setting the device lists, raced per
//! problem, over the unit routine as the floor.

mod base;
mod candidate;
mod key;
mod routine_runner;

pub(crate) use base::*;
pub use candidate::*;
pub use key::*;
pub(crate) use routine_runner::*;
