//! The reduce routines on the tile stack: each places a reduce microkernel
//! (`cubek_microkernel::reduce`) over a problem of any rank, reduced along any dim, and launches it.
//! The problem, its tensors and their binding, the scratch their caller gives, the blueprint a
//! routine prepares, the routines, one folder per routine, and how a caller picks one: a strategy
//! or the rule. A caller that measures the routines races their settings
//! ([`ReduceCandidate`]), keyed by [`ReduceKey`].

#[cfg(test)]
#[macro_use]
pub(crate) mod testing;

mod blueprint;
mod candidate;
mod key;
mod operation;
mod prepared;
mod problem;
mod routines;
mod select;
mod setting_launcher;
mod strategy;
mod tensors;
mod workspace;

pub use blueprint::*;
pub use candidate::*;
pub(crate) use cubek_microkernel::reduce::{BATCH, PARTIALS, REDUCED, SLOTS};
pub use key::*;
pub use operation::*;
pub use prepared::*;
pub use problem::*;
pub use routines::*;
pub(crate) use select::*;
pub(crate) use setting_launcher::*;
pub use strategy::*;
pub use tensors::*;
pub use workspace::*;
