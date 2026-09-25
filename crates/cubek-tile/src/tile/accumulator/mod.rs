//! Accumulators: how one is opened over a tile ([`base`]), how its cells reach memory
//! ([`drain`]), and the shared-memory destination a contraction cut across planes drains into
//! ([`smem_accumulation`]).

mod base;
mod drain;
mod smem_accumulation;

pub use base::*;
pub(crate) use drain::*;
pub use smem_accumulation::*;
