//! The reduce routines on the tile stack: each places a reduce microkernel
//! (`cubek_microkernel::reduce`) over a problem of any rank, reduced along any dim, and launches it.
//! The problem, its tensors and their binding, the blueprint a routine prepares, and the routines,
//! one folder per scope.

mod blueprint;
mod entry;
mod launch;
mod problem;
mod routines;
mod tensors;

pub use blueprint::*;
pub use entry::*;
pub use launch::*;
pub use problem::*;
pub use routines::*;
pub use tensors::*;
