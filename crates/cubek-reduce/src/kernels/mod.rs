//! The reduce routines on the tile stack: each places a reduce microkernel
//! (`cubek_microkernel::reduce`) over a problem of any rank, reduced along any dim, and launches it.
//! The problem, its tensors and their binding, the blueprint a routine prepares, the routines, one
//! folder per routine, and how a caller picks one: a strategy, the rule, and the autotune.

mod blueprint;
mod operation;
mod problem;
mod routines;
mod select;
mod strategy;
mod tensors;
#[cfg(test)]
mod tests;
mod tune;

pub use blueprint::*;
pub(crate) use cubek_microkernel::reduce::{BATCH, PARTIALS, REDUCED, SLOTS};
pub use operation::*;
pub use problem::*;
pub use routines::*;
pub(crate) use select::*;
pub use strategy::*;
pub use tensors::*;
pub use tune::*;
