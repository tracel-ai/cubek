//! The random routines on the tile stack: each places a random microkernel
//! (`cubek_microkernel::random`) over the values of an output and launches it. The problem, its
//! tensor and its binding, the selector, and the routines, one folder per scope.

mod launch;
mod problem;
mod routines;
mod select;
mod strategy;

pub use launch::*;
pub use problem::*;
pub use routines::*;
pub use select::select;
pub use strategy::*;
