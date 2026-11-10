//! The random routines: a plane routine whose lanes interleave their draws, and a unit routine
//! whose units each draw a run of their own.

mod plane;
mod unit;

pub use plane::*;
pub use unit::*;
