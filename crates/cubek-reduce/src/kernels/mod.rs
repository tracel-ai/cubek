//! The reduce routines on the tile stack: each places a reduce microkernel
//! (`cubek_microkernel::reduce`) over a problem of any rank, reduced along any dim, and launches it.
//! The problem, its tensors and their binding, the blueprint a routine prepares, the routines, one
//! folder per scope, and how a caller picks one: a strategy, the rule, and the autotune.

mod blueprint;
mod entry;
mod launch;
mod problem;
mod routines;
mod select;
mod strategy;
mod tensors;
mod tune;

pub use blueprint::*;
pub use entry::*;
pub use launch::*;
pub use problem::*;
pub use routines::*;
pub use select::*;
pub use strategy::*;
pub use tensors::*;
pub use tune::*;
