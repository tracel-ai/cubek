//! The reduce routines on the tile stack: each places a reduce microkernel
//! (`cubek_microkernel::reduce`) over a problem of any rank, reduced along any dim, and launches it.
//! The problem, its tensors and their binding, the scratch their caller gives, the blueprint a
//! routine prepares, the routines, one folder per routine, and how a caller picks one: a strategy
//! or the rule. A caller that measures the routines races their settings
//! ([`ReduceRoutineSetting`]), keyed by [`ReduceKey`].

mod blueprint;
mod candidate;
mod key;
mod operation;
mod problem;
mod routine_setting;
mod routines;
mod select;
mod strategy;
mod tensors;
#[cfg(test)]
mod tests;
mod workspace;

pub use blueprint::*;
pub use candidate::*;
pub(crate) use cubek_microkernel::reduce::{BATCH, PARTIALS, REDUCED, SLOTS};
pub use key::*;
pub use operation::*;
pub use problem::*;
pub(crate) use routine_setting::*;
pub use routines::*;
pub(crate) use select::*;
pub use strategy::*;
pub use tensors::*;
pub use workspace::*;
