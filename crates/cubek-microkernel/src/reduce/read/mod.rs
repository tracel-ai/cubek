//! How a unit reads its share of a row that several units read together, into its carrier: every
//! `stride`-th line ([`UnitStridedReduceMicrokernel`]), or the cells a region's walk gives it
//! ([`UnitWalkReduceMicrokernel`]).

mod base;
mod microkernel;

pub use base::*;
pub(crate) use microkernel::*;
