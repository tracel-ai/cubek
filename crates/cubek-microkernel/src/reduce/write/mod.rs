//! What every reduce microkernel ends on: one unit writing a carrier's lines to its window of the
//! output ([`UnitCarrierWriteMicrokernel`]).

mod base;
mod microkernel;

pub use base::*;
pub(crate) use microkernel::*;
